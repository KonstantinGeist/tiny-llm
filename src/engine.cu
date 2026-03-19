extern "C" {
#include "engine.h"
#include "tokenizer.h"
#include "utils.h"
#include "gguf.h"
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define Q8_BLOCK_SIZE 32

typedef struct __attribute__((packed)) {
    uint16_t scale;
    int8_t   qs[Q8_BLOCK_SIZE];
} Q8Block;

typedef struct {
    int   d_model, n_layers, n_heads_q, n_heads_kv, head_dim, d_ff;
    int   vocab_size, max_seq_len;
    float rope_freq_base, rms_norm_eps;
} Config;

typedef struct {
    Q8Block *token_embd;          // [vocab_size, d_model]  Q8_0
    struct {
        float   *attn_norm;       // [d_model]              f32
        Q8Block *attn_q_w;        // [n_heads_q*head_dim, d_model]  Q8_0
        float   *attn_q_norm;     // [head_dim]             f32
        Q8Block *attn_k_w;        // [n_heads_kv*head_dim, d_model]  Q8_0
        float   *attn_k_norm;     // [head_dim]             f32
        Q8Block *attn_v_w;        // [n_heads_kv*head_dim, d_model]  Q8_0
        Q8Block *attn_out_w;      // [d_model, d_model]     Q8_0
        float   *ffn_norm;        // [d_model]              f32
        Q8Block *ffn_gate_w;      // [d_ff, d_model]        Q8_0
        Q8Block *ffn_up_w;        // [d_ff, d_model]        Q8_0
        Q8Block *ffn_down_w;      // [d_model, d_ff]        Q8_0
    } layer[32];
    float *output_norm;           // [d_model]              f32
} Weights;

typedef struct {
    float *k, *v;
    int n_layers, max_seq_len, n_heads_kv, head_dim;
} KVCache;

typedef struct {
    float *x, *xb;
    float *q, *k_cur, *v_cur;
    float *attn, *attn_out;
    float *gate, *up;
    float *tmp, *ffn_out;
    float *logits;
    float *logits_host;
} RunState;

struct Engine {
    Config    cfg;
    Weights   w;
    KVCache   kv;
    RunState  s;
    Tokenizer tok;
    int       pos;
    gguf_ctx_t *gguf;
    long   prefill_tokens;
    double prefill_ms;
    long   gen_tokens;
    double gen_ms;
};

#define GEMV_THREADS 128

__global__ void q8_gemv(float * __restrict__ y,
                        const Q8Block * __restrict__ W,
                        const float   * __restrict__ x,
                        int d_in)
{
    int row = blockIdx.x;
    int n_blocks = d_in / Q8_BLOCK_SIZE;
    const Q8Block *row_blocks = W + (size_t)row * n_blocks;

    float acc = 0.f;

    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        float scale = __half2float(__ushort_as_half(row_blocks[b].scale));
        int col_base = b * Q8_BLOCK_SIZE;
        float block_dot = 0.f;
        #pragma unroll
        for (int i = 0; i < Q8_BLOCK_SIZE; i++) {
            block_dot += (float)row_blocks[b].qs[i] * x[col_base + i];
        }
        acc += scale * block_dot;
    }

    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    __shared__ float smem[GEMV_THREADS / 32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = acc;
    __syncthreads();

    int n_warps = blockDim.x / 32;
    if (warp == 0) {
        acc = (lane < n_warps) ? smem[lane] : 0.f;
        #pragma unroll
        for (int offset = n_warps / 2; offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        if (lane == 0) y[row] = acc;
    }
}

static void gemv_q8(float *y, const Q8Block *W, const float *x,
                    int d_in, int d_out) {
    q8_gemv<<<d_out, GEMV_THREADS>>>(y, W, x, d_in);
}

#define NORM_THREADS 256

__global__ void rms_norm_kernel(float * __restrict__ out,
                                const float * __restrict__ x,
                                const float * __restrict__ w,
                                int n, float eps)
{
    __shared__ float smem[NORM_THREADS / 32];

    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        ss += x[i] * x[i];

    #pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1)
        ss += __shfl_down_sync(0xffffffff, ss, off);

    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = ss;
    __syncthreads();

    int n_warps = blockDim.x / 32;
    if (warp == 0) {
        ss = (lane < n_warps) ? smem[lane] : 0.f;
        #pragma unroll
        for (int off = n_warps / 2; off > 0; off >>= 1)
            ss += __shfl_down_sync(0xffffffff, ss, off);
        if (lane == 0) smem[0] = 1.f / sqrtf(ss / n + eps);
    }
    __syncthreads();

    float scale = smem[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = w[i] * x[i] * scale;
}

__global__ void rms_norm_heads_kernel(float * __restrict__ v,
                                      const float * __restrict__ w,
                                      int head_dim, float eps)
{
    __shared__ float smem[NORM_THREADS / 32];

    float *vh = v + (size_t)blockIdx.x * head_dim;

    float ss = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        ss += vh[i] * vh[i];

    #pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1)
        ss += __shfl_down_sync(0xffffffff, ss, off);

    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = ss;
    __syncthreads();

    int n_warps = blockDim.x / 32;
    if (warp == 0) {
        ss = (lane < n_warps) ? smem[lane] : 0.f;
        #pragma unroll
        for (int off = n_warps / 2; off > 0; off >>= 1)
            ss += __shfl_down_sync(0xffffffff, ss, off);
        if (lane == 0) smem[0] = 1.f / sqrtf(ss / head_dim + eps);
    }
    __syncthreads();

    float sc = smem[0];
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        vh[i] = w[i] * vh[i] * sc;
}

__global__ void rope_kernel(float * __restrict__ v,
                            int pos, int head_dim, float freq_base)
{
    int h = blockIdx.x;
    int i = threadIdx.x;   // i in [0, head_dim/2)
    float *vh = v + h * head_dim;

    float freq = 1.f / powf(freq_base, 2.f * i / head_dim);
    float angle = pos * freq;
    float c = cosf(angle), s = sinf(angle);
    float v0 = vh[i], v1 = vh[i + head_dim / 2];
    vh[i]               = v0 * c - v1 * s;
    vh[i + head_dim / 2] = v0 * s + v1 * c;
}

#define SOFTMAX_THREADS 128

__global__ void softmax_kernel(float * __restrict__ x, int n)
{
    __shared__ float smem[SOFTMAX_THREADS / 32];

    float mx = -1e38f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        mx = fmaxf(mx, x[i]);

    #pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));

    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = mx;
    __syncthreads();

    int n_warps = blockDim.x / 32;
    if (warp == 0) {
        mx = (lane < n_warps) ? smem[lane] : -1e38f;
        #pragma unroll
        for (int off = n_warps / 2; off > 0; off >>= 1)
            mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
        if (lane == 0) smem[0] = mx;
    }
    __syncthreads();
    mx = smem[0];

    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        x[i] = expf(x[i] - mx);
        s += x[i];
    }

    #pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffff, s, off);

    if (lane == 0) smem[warp] = s;
    __syncthreads();

    if (warp == 0) {
        s = (lane < n_warps) ? smem[lane] : 0.f;
        #pragma unroll
        for (int off = n_warps / 2; off > 0; off >>= 1)
            s += __shfl_down_sync(0xffffffff, s, off);
        if (lane == 0) smem[0] = s;
    }
    __syncthreads();
    s = smem[0];

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        x[i] /= s;
}

__global__ void add_residual_kernel(float * __restrict__ x,
                                    const float * __restrict__ y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

#define ATTN_THREADS 128

__global__ void attn_kernel(
    float       * __restrict__ attn_out,
    float       * __restrict__ scores_buf,
    const float * __restrict__ q,
    const float * __restrict__ k_cache,
    const float * __restrict__ v_cache,
    int pos,
    int layer,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    int max_seq_len)
{
    int h      = blockIdx.x;
    int group  = n_heads_q / n_heads_kv;
    int kv_h   = h / group;
    float scale = 1.f / sqrtf((float)head_dim);

    const float *qh = q + h * head_dim;
    size_t layer_off = (size_t)layer * max_seq_len * n_heads_kv * head_dim;
    float *scores = scores_buf + h * max_seq_len;

    __shared__ float smem[ATTN_THREADS / 32];

    for (int t = 0; t <= pos; t++) {
        const float *kt = k_cache + layer_off + ((size_t)t * n_heads_kv + kv_h) * head_dim;
        float dot = 0.f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
            dot += qh[i] * kt[i];

        #pragma unroll
        for (int off = warpSize / 2; off > 0; off >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, off);

        int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        if (lane == 0) smem[warp] = dot;
        __syncthreads();

        int n_warps = blockDim.x / 32;
        if (warp == 0) {
            dot = (lane < n_warps) ? smem[lane] : 0.f;
            #pragma unroll
            for (int off = n_warps / 2; off > 0; off >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, off);
            if (lane == 0) scores[t] = dot * scale;
        }
        __syncthreads();
    }

    float mx = -1e38f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x)
        mx = fmaxf(mx, scores[t]);
    #pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
    {
        int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        if (lane == 0) smem[warp] = mx;
        __syncthreads();
        int n_warps = blockDim.x / 32;
        if (warp == 0) {
            mx = (lane < n_warps) ? smem[lane] : -1e38f;
            #pragma unroll
            for (int off = n_warps / 2; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
            if (lane == 0) smem[0] = mx;
        }
        __syncthreads();
        mx = smem[0];
    }
    float s = 0.f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        scores[t] = expf(scores[t] - mx);
        s += scores[t];
    }
    #pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffff, s, off);
    {
        int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        if (lane == 0) smem[warp] = s;
        __syncthreads();
        int n_warps = blockDim.x / 32;
        if (warp == 0) {
            s = (lane < n_warps) ? smem[lane] : 0.f;
            #pragma unroll
            for (int off = n_warps / 2; off > 0; off >>= 1)
                s += __shfl_down_sync(0xffffffff, s, off);
            if (lane == 0) smem[0] = s;
        }
        __syncthreads();
        s = smem[0];
    }
    for (int t = threadIdx.x; t <= pos; t += blockDim.x)
        scores[t] /= s;
    __syncthreads();

    float *oh = attn_out + h * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        oh[i] = 0.f;
    __syncthreads();

    for (int t = 0; t <= pos; t++) {
        float sc = scores[t];
        const float *vt = v_cache + layer_off + ((size_t)t * n_heads_kv + kv_h) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        oh[i] += sc * vt[i];
    }
}

__global__ void swiglu_kernel(float * __restrict__ ffn_out,
                               const float * __restrict__ gate,
                               const float * __restrict__ up,
                               int d_ff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_ff) {
        float g = gate[i];
        ffn_out[i] = (g / (1.f + expf(-g))) * up[i];
    }
}

__global__ void embed_lookup_kernel(float * __restrict__ out,
                                    const Q8Block * __restrict__ embd,
                                    int token, int d_model)
{
    int block_idx = blockIdx.x;
    int elem      = threadIdx.x;
    int n_blocks  = d_model / Q8_BLOCK_SIZE;
    const Q8Block *row = embd + (size_t)token * n_blocks;
    float scale = __half2float(__ushort_as_half(row[block_idx].scale));
    out[block_idx * Q8_BLOCK_SIZE + elem] = scale * (float)row[block_idx].qs[elem];
}

static Q8Block *gpu_upload_q8(const void *host_ptr, size_t n_elems) {
    if (!host_ptr) return NULL;
    size_t n_blocks = n_elems / Q8_BLOCK_SIZE;
    size_t bytes    = n_blocks * sizeof(Q8Block);
    Q8Block *d;
    CUDA_CHECK(cudaMalloc(&d, bytes));
    CUDA_CHECK(cudaMemcpy(d, host_ptr, bytes, cudaMemcpyHostToDevice));
    return d;
}

static float *gpu_upload_f32(const void *host_ptr, size_t n_elems) {
    if (!host_ptr) return NULL;
    size_t bytes = n_elems * sizeof(float);
    float *d;
    CUDA_CHECK(cudaMalloc(&d, bytes));
    CUDA_CHECK(cudaMemcpy(d, host_ptr, bytes, cudaMemcpyHostToDevice));
    return d;
}

extern "C" Engine *engine_load(const char *model_path) {
    Engine *e = (Engine *)calloc(1, sizeof(Engine));

    e->gguf = gguf_load(model_path);
    if (!e->gguf) { free(e); return NULL; }
    gguf_ctx_t *ctx = e->gguf;

    Config *c = &e->cfg;
#define U32(k) ((int)gguf_get_val(ctx, k)->uint32)
    c->d_model     = U32("qwen3.embedding_length");
    c->n_layers    = U32("qwen3.block_count");
    c->n_heads_q   = U32("qwen3.attention.head_count");
    c->n_heads_kv  = U32("qwen3.attention.head_count_kv");
    c->d_ff        = U32("qwen3.feed_forward_length");
    c->max_seq_len = U32("qwen3.context_length");
    c->head_dim    = U32("qwen3.attention.key_length");
#undef U32
    c->vocab_size     = (int)gguf_get_val(ctx, "tokenizer.ggml.tokens")->array.len;
    c->rope_freq_base = gguf_get_val(ctx, "qwen3.rope.freq_base")->float32;
    c->rms_norm_eps   = gguf_get_val(ctx, "qwen3.attention.layer_norm_rms_epsilon")->float32;

    int fd = open(model_path, O_RDONLY);
    struct stat st; fstat(fd, &st);
    uint8_t *base = (uint8_t *)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    Weights *w  = &e->w;
    int dm      = c->d_model;
    int dff     = c->d_ff;
    int nkv_d   = c->n_heads_kv * c->head_dim;
    int nq_d    = c->n_heads_q  * c->head_dim;
    char name[128];

    w->token_embd  = gpu_upload_q8(gguf_tensor_ptr(ctx, base, "token_embd.weight"),
                                   (size_t)c->vocab_size * dm);
    w->output_norm = gpu_upload_f32(gguf_tensor_ptr(ctx, base, "output_norm.weight"), dm);

    for (int l = 0; l < c->n_layers; l++) {
#define UQ8(field, fmt, n)  \
        snprintf(name, sizeof(name), fmt, l); \
        w->layer[l].field = gpu_upload_q8(gguf_tensor_ptr(ctx, base, name), (size_t)(n))
#define UF32(field, fmt, n) \
        snprintf(name, sizeof(name), fmt, l); \
        w->layer[l].field = gpu_upload_f32(gguf_tensor_ptr(ctx, base, name), (size_t)(n))

        UF32(attn_norm,   "blk.%d.attn_norm.weight",   dm);
        UQ8 (attn_q_w,    "blk.%d.attn_q.weight",      nq_d  * dm);
        UF32(attn_q_norm, "blk.%d.attn_q_norm.weight", c->head_dim);
        UQ8 (attn_k_w,    "blk.%d.attn_k.weight",      nkv_d * dm);
        UF32(attn_k_norm, "blk.%d.attn_k_norm.weight", c->head_dim);
        UQ8 (attn_v_w,    "blk.%d.attn_v.weight",      nkv_d * dm);
        UQ8 (attn_out_w,  "blk.%d.attn_output.weight", dm    * dm);
        UF32(ffn_norm,    "blk.%d.ffn_norm.weight",    dm);
        UQ8 (ffn_gate_w,  "blk.%d.ffn_gate.weight",    dff   * dm);
        UQ8 (ffn_up_w,    "blk.%d.ffn_up.weight",      dff   * dm);
        UQ8 (ffn_down_w,  "blk.%d.ffn_down.weight",    dm    * dff);
#undef UQ8
#undef UF32
    }

    munmap(base, st.st_size);

    if (!w->token_embd || !w->output_norm) { engine_free(e); return NULL; }
    for (int l = 0; l < c->n_layers; l++) {
        if (!w->layer[l].attn_norm   || !w->layer[l].attn_q_w   ||
            !w->layer[l].attn_q_norm || !w->layer[l].attn_k_w   ||
            !w->layer[l].attn_k_norm || !w->layer[l].attn_v_w   ||
            !w->layer[l].attn_out_w  || !w->layer[l].ffn_norm   ||
            !w->layer[l].ffn_gate_w  || !w->layer[l].ffn_up_w   ||
            !w->layer[l].ffn_down_w) {
            engine_free(e); return NULL;
        }
    }

    const gguf_value_t *tv = gguf_get_val(ctx, "tokenizer.ggml.tokens");
    const gguf_value_t *mv = gguf_get_val(ctx, "tokenizer.ggml.merges");
    int vocab_size = (int)tv->array.len;
    int n_merges   = (int)mv->array.len;
    char **vocab  = (char **)malloc(vocab_size * sizeof(char *));
    char **merges = (char **)malloc(n_merges   * sizeof(char *));
    for (int i = 0; i < vocab_size; i++) vocab[i]  = tv->array.items[i].string.str;
    for (int i = 0; i < n_merges;   i++) merges[i] = mv->array.items[i].string.str;
    int bos_id = (int)gguf_get_val(ctx, "tokenizer.ggml.bos_token_id")->uint32;
    int eos_id = (int)gguf_get_val(ctx, "tokenizer.ggml.eos_token_id")->uint32;
    tok_init(&e->tok, vocab, vocab_size, merges, n_merges, bos_id, eos_id);
    free(merges);
    e->tok.vocab = vocab;

    RunState *s = &e->s;
    int nq = c->n_heads_q, hd = c->head_dim;
    int vs = c->vocab_size, ml = c->max_seq_len;
#define DMALLOC(ptr, count) CUDA_CHECK(cudaMalloc(&(ptr), (count) * sizeof(float)))
    DMALLOC(s->x,        dm);
    DMALLOC(s->xb,       dm);
    DMALLOC(s->q,        nq * hd);
    DMALLOC(s->k_cur,    c->n_heads_kv * hd);
    DMALLOC(s->v_cur,    c->n_heads_kv * hd);
    DMALLOC(s->attn,     nq * ml);
    DMALLOC(s->attn_out, dm);
    DMALLOC(s->gate,     dff);
    DMALLOC(s->up,       dff);
    DMALLOC(s->tmp,      dm);
    DMALLOC(s->ffn_out,  dff);
    DMALLOC(s->logits,   vs);
#undef DMALLOC
    CUDA_CHECK(cudaMallocHost(&s->logits_host, vs * sizeof(float)));

    KVCache *kv = &e->kv;
    kv->n_layers    = c->n_layers;
    kv->max_seq_len = c->max_seq_len;
    kv->n_heads_kv  = c->n_heads_kv;
    kv->head_dim    = hd;
    size_t kvsz = (size_t)c->n_layers * c->max_seq_len * c->n_heads_kv * hd * sizeof(float);
    CUDA_CHECK(cudaMalloc(&kv->k, kvsz));
    CUDA_CHECK(cudaMalloc(&kv->v, kvsz));
    CUDA_CHECK(cudaMemset(kv->k, 0, kvsz));
    CUDA_CHECK(cudaMemset(kv->v, 0, kvsz));

    e->pos = 0;
    return e;
}

static float *forward(Engine *e, int token, int pos) {
    Config   *cfg = &e->cfg;
    Weights  *w   = &e->w;
    KVCache  *kv  = &e->kv;
    RunState *s   = &e->s;

    int dm    = cfg->d_model;
    int nq    = cfg->n_heads_q;
    int nkv   = cfg->n_heads_kv;
    int hd    = cfg->head_dim;
    int dff   = cfg->d_ff;
    int nkv_d = nkv * hd;

    int emb_blocks = dm / Q8_BLOCK_SIZE;
    embed_lookup_kernel<<<emb_blocks, Q8_BLOCK_SIZE>>>(s->x, w->token_embd, token, dm);

    for (int l = 0; l < cfg->n_layers; l++) {

        rms_norm_kernel<<<1, NORM_THREADS>>>(s->xb, s->x, w->layer[l].attn_norm, dm, cfg->rms_norm_eps);

        gemv_q8(s->q,     w->layer[l].attn_q_w, s->xb, dm, nq  * hd);
        gemv_q8(s->k_cur, w->layer[l].attn_k_w, s->xb, dm, nkv * hd);
        gemv_q8(s->v_cur, w->layer[l].attn_v_w, s->xb, dm, nkv_d);

        rms_norm_heads_kernel<<<nq,  NORM_THREADS>>>(s->q,     w->layer[l].attn_q_norm, hd, cfg->rms_norm_eps);
        rms_norm_heads_kernel<<<nkv, NORM_THREADS>>>(s->k_cur, w->layer[l].attn_k_norm, hd, cfg->rms_norm_eps);

        rope_kernel<<<nq,  hd / 2>>>(s->q,     pos, hd, cfg->rope_freq_base);
        rope_kernel<<<nkv, hd / 2>>>(s->k_cur, pos, hd, cfg->rope_freq_base);

        size_t kv_row = ((size_t)l * cfg->max_seq_len + pos) * nkv_d;
        CUDA_CHECK(cudaMemcpy(kv->k + kv_row, s->k_cur, nkv_d * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(kv->v + kv_row, s->v_cur, nkv_d * sizeof(float), cudaMemcpyDeviceToDevice));

        attn_kernel<<<nq, ATTN_THREADS>>>(
            s->attn_out, s->attn,
            s->q, kv->k, kv->v,
            pos, l, nq, nkv, hd, cfg->max_seq_len);

        gemv_q8(s->tmp, w->layer[l].attn_out_w, s->attn_out, dm, dm);
        add_residual_kernel<<<(dm + 255) / 256, 256>>>(s->x, s->tmp, dm);

        rms_norm_kernel<<<1, NORM_THREADS>>>(s->xb, s->x, w->layer[l].ffn_norm, dm, cfg->rms_norm_eps);

        gemv_q8(s->gate, w->layer[l].ffn_gate_w, s->xb, dm, dff);
        gemv_q8(s->up,   w->layer[l].ffn_up_w,   s->xb, dm, dff);

        swiglu_kernel<<<(dff + 255) / 256, 256>>>(s->ffn_out, s->gate, s->up, dff);

        gemv_q8(s->tmp, w->layer[l].ffn_down_w, s->ffn_out, dff, dm);
        add_residual_kernel<<<(dm + 255) / 256, 256>>>(s->x, s->tmp, dm);
    }

    rms_norm_kernel<<<1, NORM_THREADS>>>(s->xb, s->x, w->output_norm, dm, cfg->rms_norm_eps);

    gemv_q8(s->logits, w->token_embd, s->xb, dm, cfg->vocab_size);

    CUDA_CHECK(cudaMemcpy(s->logits_host, s->logits,
                          cfg->vocab_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return s->logits_host;
}

static int host_argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

extern "C" void engine_generate(Engine *e, const char *prompt,
                     TokenCallback cb, void *cb_ctx) {
    int tokens[4096];
    int n = tok_encode(&e->tok, prompt, tokens);

    float *logits = NULL;
    double t0 = now_ms();
    for (int i = 0; i < n; i++)
        logits = forward(e, tokens[i], e->pos++);
    double t1 = now_ms();
    e->prefill_tokens += n;
    e->prefill_ms     += t1 - t0;

    for (;;) {
        int next = host_argmax(logits, e->cfg.vocab_size);
        if (next == e->tok.eos_id) break;
        double tg0 = now_ms();
        if (cb(e, next, cb_ctx) != 0) break;
        if (e->pos >= e->cfg.max_seq_len) break;
        logits = forward(e, next, e->pos++);
        e->gen_ms += now_ms() - tg0;
        e->gen_tokens++;
    }
}

extern "C" const char *engine_decode_token(Engine *e, int token_id) {
    return tok_decode(&e->tok, token_id);
}

extern "C" void engine_get_stats(const Engine *e, EngineStats *out) {
    out->prefill_tokens = e->prefill_tokens;
    out->prefill_ms     = e->prefill_ms;
    out->gen_tokens     = e->gen_tokens;
    out->gen_ms         = e->gen_ms;
}

extern "C" void engine_free(Engine *e) {
    if (!e) return;
    gguf_free(e->gguf);

    cudaFree(e->w.token_embd);
    cudaFree(e->w.output_norm);
    for (int l = 0; l < e->cfg.n_layers; l++) {
        cudaFree(e->w.layer[l].attn_norm);
        cudaFree(e->w.layer[l].attn_q_w);
        cudaFree(e->w.layer[l].attn_q_norm);
        cudaFree(e->w.layer[l].attn_k_w);
        cudaFree(e->w.layer[l].attn_k_norm);
        cudaFree(e->w.layer[l].attn_v_w);
        cudaFree(e->w.layer[l].attn_out_w);
        cudaFree(e->w.layer[l].ffn_norm);
        cudaFree(e->w.layer[l].ffn_gate_w);
        cudaFree(e->w.layer[l].ffn_up_w);
        cudaFree(e->w.layer[l].ffn_down_w);
    }

    cudaFree(e->s.x);      cudaFree(e->s.xb);
    cudaFree(e->s.q);      cudaFree(e->s.k_cur);   cudaFree(e->s.v_cur);
    cudaFree(e->s.attn);   cudaFree(e->s.attn_out);
    cudaFree(e->s.gate);   cudaFree(e->s.up);
    cudaFree(e->s.tmp);    cudaFree(e->s.ffn_out);
    cudaFree(e->s.logits);
    cudaFreeHost(e->s.logits_host);

    cudaFree(e->kv.k);
    cudaFree(e->kv.v);

    free(e->tok.vocab);
    tok_free(&e->tok);
    free(e);
}
