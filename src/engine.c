#include "engine.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

typedef struct {
    uint16_t scale;
    int8_t   qs[32];
} q8_0_block_t;

typedef struct {
    const q8_0_block_t *data;
    int d_out;
    int d_in;
} q8_0_mat_t;

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float r; __builtin_memcpy(&r, &f, 4); return r;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static float *copy_f32(const float *src, size_t n) {
    if (!src) return NULL;
    float *dst = malloc(n * sizeof(float));
    memcpy(dst, src, n * sizeof(float));
    return dst;
}

static inline q8_0_mat_t make_q8_mat(const void *ptr, int d_out, int d_in) {
    q8_0_mat_t m;
    m.data  = (const q8_0_block_t *)ptr;
    m.d_out = d_out;
    m.d_in  = d_in;
    return m;
}

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __AVX2__

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

static inline float hmax256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m  = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_movehdup_ps(m));
    return _mm_cvtss_f32(m);
}

static inline __m256 exp256(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps( 88.3762626647950f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647950f));
    __m256 z   = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896f),
                                     _mm256_set1_ps(0.5f));
    __m256i n  = _mm256_cvttps_epi32(z);
    __m256  fn = _mm256_cvtepi32_ps(n);
    __m256 f   = _mm256_fnmadd_ps(fn, _mm256_set1_ps( 0.693359375f),    x);
    f           = _mm256_fnmadd_ps(fn, _mm256_set1_ps(-2.12194440e-4f), f);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    __m256i pw2 = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(pw2));
}

#endif // __AVX2__

static void rms_norm(float *out, const float *x, const float *w, int n, float eps) {
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    float ss = hsum256(acc);
    for (; i < n; i++) ss += x[i] * x[i];
    ss = 1.f / sqrtf(ss / (float)n + eps);
    __m256 sc = _mm256_set1_ps(ss);
    i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(out + i,
            _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(w + i),
                                        _mm256_loadu_ps(x + i)), sc));
    for (; i < n; i++) out[i] = w[i] * x[i] * ss;
#else
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = w[i] * x[i] * ss;
#endif
}

static void softmax(float *x, int n) {
#ifdef __AVX2__
    __m256 vmx = _mm256_set1_ps(x[0]);
    int i = 0;
    for (; i <= n - 8; i += 8)
        vmx = _mm256_max_ps(vmx, _mm256_loadu_ps(x + i));
    float mx = hmax256(vmx);
    for (; i < n; i++) if (x[i] > mx) mx = x[i];

    __m256 vmxv = _mm256_set1_ps(mx);
    __m256 vsum = _mm256_setzero_ps();
    i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 v = exp256(_mm256_sub_ps(_mm256_loadu_ps(x + i), vmxv));
        _mm256_storeu_ps(x + i, v);
        vsum = _mm256_add_ps(vsum, v);
    }
    float s = hsum256(vsum);
    for (; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }

    __m256 vinv = _mm256_set1_ps(1.f / s);
    i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
    float inv = 1.f / s;
    for (; i < n; i++) x[i] *= inv;
#else
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0.f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
#endif
}

static int argmax(const float *v, int n) {
#ifdef __AVX2__
    __m256 vmx = _mm256_loadu_ps(v);
    int i = 8;
    for (; i <= n - 8; i += 8)
        vmx = _mm256_max_ps(vmx, _mm256_loadu_ps(v + i));
    float mx = hmax256(vmx);
    for (; i < n; i++) if (v[i] > mx) mx = v[i];
    for (int j = 0; j < n; j++) if (v[j] == mx) return j;
    return 0;
#else
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
#endif
}

static inline float dot_q8_f32(const q8_0_block_t *row, const float *x, int nb) {
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m256 vsc = _mm256_set1_ps(f16_to_f32(row[b].scale));
        const int8_t *qs = row[b].qs;
        __m128i q8_lo = _mm_loadu_si128((const __m128i *)(qs));
        __m128i q8_hi = _mm_loadu_si128((const __m128i *)(qs + 16));
        __m256i q16_lo = _mm256_cvtepi8_epi16(q8_lo);
        __m256i q16_hi = _mm256_cvtepi8_epi16(q8_hi);
        __m256 qf0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16_lo)));
        __m256 qf1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16_lo, 1)));
        __m256 qf2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16_hi)));
        __m256 qf3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16_hi, 1)));
        __m256 dot = _mm256_fmadd_ps(qf0, _mm256_loadu_ps(x + b*32),
                     _mm256_fmadd_ps(qf1, _mm256_loadu_ps(x + b*32 + 8),
                     _mm256_fmadd_ps(qf2, _mm256_loadu_ps(x + b*32 + 16),
                     _mm256_mul_ps  (qf3, _mm256_loadu_ps(x + b*32 + 24)))));
        acc = _mm256_fmadd_ps(vsc, dot, acc);
    }
    return hsum256(acc);
#else
    float s = 0.f;
    for (int b = 0; b < nb; b++) {
        float sc = f16_to_f32(row[b].scale);
        for (int i = 0; i < 32; i++)
            s += sc * (float)row[b].qs[i] * x[b * 32 + i];
    }
    return s;
#endif
}

static void gemv_q8(float *y, const q8_0_mat_t *W, const float *x,
                    int row_start, int row_end) {
    int nb = W->d_in / 32;
    for (int r = row_start; r < row_end; r++)
        y[r] = dot_q8_f32(W->data + (size_t)r * nb, x, nb);
}

static void gemv_q8_swiglu(float *out,
                            const q8_0_mat_t *gate_W, const q8_0_mat_t *up_W,
                            const float *x, int row_start, int row_end) {
    int nb = gate_W->d_in / 32;
    for (int r = row_start; r < row_end; r++) {
        float g = dot_q8_f32(gate_W->data + (size_t)r * nb, x, nb);
        float u = dot_q8_f32(up_W->data   + (size_t)r * nb, x, nb);
        out[r] = g / (1.f + expf(-g)) * u;
    }
}

typedef void (*tp_fn_t)(void *arg, int tid, int n_threads);

typedef struct {
    pthread_t       thread;
    int             id;
    int             n_threads;
    tp_fn_t         fn;
    void           *arg;
    pthread_mutex_t mu;
    pthread_cond_t  cv_wake;
    pthread_cond_t  cv_done;
    int             ready;
    int             done;
    int             quit;
} Worker;

static Worker  *g_workers   = NULL;
static int      g_n_threads = 1;

static void *worker_main(void *p) {
    Worker *w = (Worker *)p;
    for (;;) {
        pthread_mutex_lock(&w->mu);
        while (!w->ready && !w->quit)
            pthread_cond_wait(&w->cv_wake, &w->mu);
        if (w->quit) { pthread_mutex_unlock(&w->mu); break; }
        w->ready = 0;
        pthread_mutex_unlock(&w->mu);
        w->fn(w->arg, w->id, w->n_threads);
        pthread_mutex_lock(&w->mu);
        w->done = 1;
        pthread_cond_signal(&w->cv_done);
        pthread_mutex_unlock(&w->mu);
    }
    return NULL;
}

static void tp_init(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    g_n_threads = n_threads;
    if (n_threads == 1) return;
    g_workers = calloc(n_threads, sizeof(Worker));
    for (int i = 0; i < n_threads; i++) {
        Worker *w = &g_workers[i];
        w->id = i; w->n_threads = n_threads;
        pthread_mutex_init(&w->mu,      NULL);
        pthread_cond_init (&w->cv_wake, NULL);
        pthread_cond_init (&w->cv_done, NULL);
    }
    for (int i = 1; i < n_threads; i++)
        pthread_create(&g_workers[i].thread, NULL, worker_main, &g_workers[i]);
}

static void tp_free(void) {
    if (!g_workers) return;
    for (int i = 1; i < g_n_threads; i++) {
        Worker *w = &g_workers[i];
        pthread_mutex_lock(&w->mu);
        w->quit = 1;
        pthread_cond_signal(&w->cv_wake);
        pthread_mutex_unlock(&w->mu);
        pthread_join(w->thread, NULL);
        pthread_mutex_destroy(&w->mu);
        pthread_cond_destroy(&w->cv_wake);
        pthread_cond_destroy(&w->cv_done);
    }
    free(g_workers);
    g_workers = NULL;
    g_n_threads = 1;
}

static void tp_run(tp_fn_t fn, void *arg) {
    if (g_n_threads == 1) { fn(arg, 0, 1); return; }
    for (int i = 1; i < g_n_threads; i++) {
        Worker *w = &g_workers[i];
        pthread_mutex_lock(&w->mu);
        w->fn = fn; w->arg = arg; w->done = 0; w->ready = 1;
        pthread_cond_signal(&w->cv_wake);
        pthread_mutex_unlock(&w->mu);
    }
    fn(arg, 0, g_n_threads);
    for (int i = 1; i < g_n_threads; i++) {
        Worker *w = &g_workers[i];
        pthread_mutex_lock(&w->mu);
        while (!w->done) pthread_cond_wait(&w->cv_done, &w->mu);
        pthread_mutex_unlock(&w->mu);
    }
}

typedef struct {
    int   d_model, n_layers, n_heads_q, n_heads_kv, head_dim, d_ff;
    int   vocab_size, max_seq_len;
    float rope_freq_base, rms_norm_eps;
} Config;

typedef struct {
    q8_0_mat_t token_embd;
    struct {
        float      *attn_norm;
        q8_0_mat_t  attn_q_w;
        float      *attn_q_norm;
        q8_0_mat_t  attn_k_w;
        float      *attn_k_norm;
        q8_0_mat_t  attn_v_w;
        q8_0_mat_t  attn_out_w;
        float      *ffn_norm;
        q8_0_mat_t  ffn_gate_w;
        q8_0_mat_t  ffn_up_w;
        q8_0_mat_t  ffn_down_w;
    } layer[32];
    float *output_norm;
} Weights;

typedef struct {
    float *k, *v;
    int n_layers, max_seq_len, n_heads_kv, head_dim;
} KVCache;

typedef struct {
    float *x, *xb;
    float *q, *k_cur, *v_cur;
    float *attn;
    float *attn_out;
    float *ffn_out;
    float *tmp;
    float *logits;
} RunState;

struct Engine {
    Config     cfg;
    Weights    w;
    KVCache    kv;
    RunState   s;
    Tokenizer  tok;
    int        pos;
    gguf_ctx_t *gguf;
    uint8_t    *mmap_base;
    size_t      mmap_size;
    long        prefill_tokens;
    double      prefill_ms;
    long        gen_tokens;
    double      gen_ms;
};

typedef struct { float *y; const q8_0_mat_t *W; const float *x; } GemvArg;

static void gemv_worker(void *p, int tid, int nt) {
    GemvArg *a = (GemvArg *)p;
    int lo = (tid       * a->W->d_out) / nt;
    int hi = ((tid + 1) * a->W->d_out) / nt;
    gemv_q8(a->y, a->W, a->x, lo, hi);
}
static void par_gemv(float *y, const q8_0_mat_t *W, const float *x) {
    GemvArg a = { y, W, x };
    tp_run(gemv_worker, &a);
}

typedef struct {
    float *out; const q8_0_mat_t *gate_W, *up_W; const float *x;
} SwigluArg;

static void swiglu_worker(void *p, int tid, int nt) {
    SwigluArg *a = (SwigluArg *)p;
    int lo = (tid       * a->gate_W->d_out) / nt;
    int hi = ((tid + 1) * a->gate_W->d_out) / nt;
    gemv_q8_swiglu(a->out, a->gate_W, a->up_W, a->x, lo, hi);
}
static void par_swiglu(float *out,
                       const q8_0_mat_t *gate_W, const q8_0_mat_t *up_W,
                       const float *x) {
    SwigluArg a = { out, gate_W, up_W, x };
    tp_run(swiglu_worker, &a);
}

static void norm_rope_heads(float *v, int n_heads, int head_dim,
                             const float *w_norm, float eps,
                             int pos, float freq_base) {
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float *vh = v + h * head_dim;
        rms_norm(vh, vh, w_norm, head_dim, eps);
        for (int i = 0; i < half; i++) {
            float freq = 1.f / powf(freq_base, 2.f * (float)i / (float)head_dim);
            float c = cosf((float)pos * freq), s = sinf((float)pos * freq);
            float v0 = vh[i], v1 = vh[i + half];
            vh[i]        = v0 * c - v1 * s;
            vh[i + half] = v0 * s + v1 * c;
        }
    }
}

typedef struct {
    const Config *cfg; const KVCache *kv; RunState *s; int layer, pos;
} AttnArg;

static void attn_worker(void *p, int tid, int nt) {
    AttnArg      *a   = (AttnArg *)p;
    const Config *cfg = a->cfg;
    const KVCache *kv = a->kv;
    RunState     *s   = a->s;
    int l = a->layer, pos = a->pos;
    int nq    = cfg->n_heads_q, nkv = cfg->n_heads_kv;
    int hd    = cfg->head_dim,  nkv_d = nkv * hd;
    int group = nq / nkv;
    float scale = 1.f / sqrtf((float)hd);
    int h_lo = (tid * nq) / nt, h_hi = ((tid + 1) * nq) / nt;

    for (int h = h_lo; h < h_hi; h++) {
        int    kv_h   = h / group;
        float *qh     = s->q   + h * hd;
        float *scores = s->attn + h * cfg->max_seq_len;

        for (int t = 0; t <= pos; t++) {
            const float *kt = kv->k + ((size_t)l * cfg->max_seq_len + t) * nkv_d + kv_h * hd;
#ifdef __AVX2__
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            for (; i <= hd - 8; i += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i), _mm256_loadu_ps(kt + i), acc);
            float dot = hsum256(acc);
            for (; i < hd; i++) dot += qh[i] * kt[i];
#else
            float dot = 0.f;
            for (int i = 0; i < hd; i++) dot += qh[i] * kt[i];
#endif
            scores[t] = dot * scale;
        }
        softmax(scores, pos + 1);

        float *oh = s->attn_out + h * hd;
        memset(oh, 0, hd * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            const float *vt = kv->v + ((size_t)l * cfg->max_seq_len + t) * nkv_d + kv_h * hd;
            float sc = scores[t];
#ifdef __AVX2__
            __m256 vs = _mm256_set1_ps(sc);
            int i = 0;
            for (; i <= hd - 8; i += 8)
                _mm256_storeu_ps(oh + i,
                    _mm256_fmadd_ps(vs, _mm256_loadu_ps(vt + i), _mm256_loadu_ps(oh + i)));
            for (; i < hd; i++) oh[i] += sc * vt[i];
#else
            for (int i = 0; i < hd; i++) oh[i] += sc * vt[i];
#endif
        }
    }
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
    int nkv_d = nkv * hd;

    {
        int nb = dm / 32;
        const q8_0_block_t *row = w->token_embd.data + (size_t)token * nb;
        for (int b = 0; b < nb; b++) {
            float sc = f16_to_f32(row[b].scale);
            for (int i = 0; i < 32; i++)
                s->x[b * 32 + i] = sc * (float)row[b].qs[i];
        }
    }

    for (int l = 0; l < cfg->n_layers; l++) {
        // Attention block
        rms_norm(s->xb, s->x, w->layer[l].attn_norm, dm, cfg->rms_norm_eps);

        par_gemv(s->q,     &w->layer[l].attn_q_w, s->xb);
        par_gemv(s->k_cur, &w->layer[l].attn_k_w, s->xb);
        par_gemv(s->v_cur, &w->layer[l].attn_v_w, s->xb);

        norm_rope_heads(s->q,     nq,  hd, w->layer[l].attn_q_norm,
                        cfg->rms_norm_eps, pos, cfg->rope_freq_base);
        norm_rope_heads(s->k_cur, nkv, hd, w->layer[l].attn_k_norm,
                        cfg->rms_norm_eps, pos, cfg->rope_freq_base);

        float *kc = kv->k + ((size_t)l * cfg->max_seq_len + pos) * nkv_d;
        float *vc = kv->v + ((size_t)l * cfg->max_seq_len + pos) * nkv_d;
        memcpy(kc, s->k_cur, nkv_d * sizeof(float));
        memcpy(vc, s->v_cur, nkv_d * sizeof(float));

        AttnArg aa = { cfg, kv, s, l, pos };
        tp_run(attn_worker, &aa);

        par_gemv(s->tmp, &w->layer[l].attn_out_w, s->attn_out);
#ifdef __AVX2__
        { int i = 0;
          for (; i <= dm - 8; i += 8)
              _mm256_storeu_ps(s->x + i, _mm256_add_ps(_mm256_loadu_ps(s->x + i),
                                                        _mm256_loadu_ps(s->tmp + i)));
          for (; i < dm; i++) s->x[i] += s->tmp[i]; }
#else
        for (int i = 0; i < dm; i++) s->x[i] += s->tmp[i];
#endif

        rms_norm(s->xb, s->x, w->layer[l].ffn_norm, dm, cfg->rms_norm_eps);
        par_swiglu(s->ffn_out, &w->layer[l].ffn_gate_w, &w->layer[l].ffn_up_w, s->xb);
        par_gemv(s->tmp, &w->layer[l].ffn_down_w, s->ffn_out);
#ifdef __AVX2__
        { int i = 0;
          for (; i <= dm - 8; i += 8)
              _mm256_storeu_ps(s->x + i, _mm256_add_ps(_mm256_loadu_ps(s->x + i),
                                                        _mm256_loadu_ps(s->tmp + i)));
          for (; i < dm; i++) s->x[i] += s->tmp[i]; }
#else
        for (int i = 0; i < dm; i++) s->x[i] += s->tmp[i];
#endif
    }

    rms_norm(s->xb, s->x, w->output_norm, dm, cfg->rms_norm_eps);
    par_gemv(s->logits, &w->token_embd, s->xb);
    return s->logits;
}

Engine *engine_load(const char *model_path) {
    Engine *e = calloc(1, sizeof(Engine));

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
    e->mmap_base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    e->mmap_size = st.st_size;
    close(fd);
    if (e->mmap_base == MAP_FAILED) { engine_free(e); return NULL; }
    posix_madvise(e->mmap_base, e->mmap_size, POSIX_MADV_SEQUENTIAL);

    uint8_t *base = e->mmap_base;
    Weights *w = &e->w;
    int dm    = c->d_model, dff = c->d_ff;
    int nkv_d = c->n_heads_kv * c->head_dim;
    int nq_d  = c->n_heads_q  * c->head_dim;

    w->token_embd  = make_q8_mat(gguf_tensor_ptr(ctx, base, "token_embd.weight"),
                                 c->vocab_size, dm);
    w->output_norm = copy_f32(gguf_tensor_ptr(ctx, base, "output_norm.weight"), dm);

    char name[128];
    for (int l = 0; l < c->n_layers; l++) {
#define TQ8(field, fmt, rows, cols) \
        snprintf(name, sizeof(name), fmt, l); \
        w->layer[l].field = make_q8_mat(gguf_tensor_ptr(ctx, base, name), rows, cols)
#define TF32(field, fmt, n) \
        snprintf(name, sizeof(name), fmt, l); \
        w->layer[l].field = copy_f32(gguf_tensor_ptr(ctx, base, name), n)
        TF32(attn_norm,   "blk.%d.attn_norm.weight",   dm);
        TQ8 (attn_q_w,    "blk.%d.attn_q.weight",      nq_d,  dm);
        TF32(attn_q_norm, "blk.%d.attn_q_norm.weight", c->head_dim);
        TQ8 (attn_k_w,    "blk.%d.attn_k.weight",      nkv_d, dm);
        TF32(attn_k_norm, "blk.%d.attn_k_norm.weight", c->head_dim);
        TQ8 (attn_v_w,    "blk.%d.attn_v.weight",      nkv_d, dm);
        TQ8 (attn_out_w,  "blk.%d.attn_output.weight", dm,    dm);
        TF32(ffn_norm,    "blk.%d.ffn_norm.weight",     dm);
        TQ8 (ffn_gate_w,  "blk.%d.ffn_gate.weight",    dff,   dm);
        TQ8 (ffn_up_w,    "blk.%d.ffn_up.weight",      dff,   dm);
        TQ8 (ffn_down_w,  "blk.%d.ffn_down.weight",    dm,    dff);
#undef TQ8
#undef TF32
    }

    const gguf_value_t *tv = gguf_get_val(ctx, "tokenizer.ggml.tokens");
    const gguf_value_t *mv = gguf_get_val(ctx, "tokenizer.ggml.merges");
    int vocab_size = (int)tv->array.len, n_merges = (int)mv->array.len;
    char **vocab  = malloc(vocab_size * sizeof(char *));
    char **merges = malloc(n_merges   * sizeof(char *));
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
    s->x        = malloc(dm      * sizeof(float));
    s->xb       = malloc(dm      * sizeof(float));
    s->q        = malloc(nq * hd * sizeof(float));
    s->k_cur    = malloc(nkv_d   * sizeof(float));
    s->v_cur    = malloc(nkv_d   * sizeof(float));
    s->attn     = calloc(nq * ml,  sizeof(float));
    s->attn_out = calloc(dm,       sizeof(float));
    s->ffn_out  = malloc(dff     * sizeof(float));
    s->tmp      = malloc(dm      * sizeof(float));
    s->logits   = malloc(vs      * sizeof(float));

    KVCache *kv = &e->kv;
    kv->n_layers = c->n_layers; kv->max_seq_len = c->max_seq_len;
    kv->n_heads_kv = c->n_heads_kv; kv->head_dim = c->head_dim;
    size_t kvsz = (size_t)c->n_layers * c->max_seq_len * nkv_d * sizeof(float);
    kv->k = calloc(1, kvsz);
    kv->v = calloc(1, kvsz);

    {
        int nt = 1;
#ifdef _SC_NPROCESSORS_ONLN
        nt = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (nt < 1) nt = 1;
#endif
        tp_init(nt);
    }

    e->pos = 0;
    return e;
}

const char *engine_decode_token(Engine *e, int token_id) {
    return tok_decode(&e->tok, token_id);
}

void engine_generate(Engine *e, const char *prompt,
                     TokenCallback cb, void *cb_ctx) {
    int tokens[4096];
    int n = tok_encode(&e->tok, prompt, tokens);
    float *logits = NULL;
    double t0 = now_ms();
    for (int i = 0; i < n; i++) logits = forward(e, tokens[i], e->pos++);
    e->prefill_tokens += n;
    e->prefill_ms     += now_ms() - t0;

    for (;;) {
        int next = argmax(logits, e->cfg.vocab_size);
        if (next == e->tok.eos_id) break;
        double tg0 = now_ms();
        if (cb(e, next, cb_ctx) != 0) break;
        if (e->pos >= e->cfg.max_seq_len) break;
        logits = forward(e, next, e->pos++);
        e->gen_ms += now_ms() - tg0;
        e->gen_tokens++;
    }
}

void engine_get_stats(const Engine *e, EngineStats *out) {
    out->prefill_tokens = e->prefill_tokens;
    out->prefill_ms     = e->prefill_ms;
    out->gen_tokens     = e->gen_tokens;
    out->gen_ms         = e->gen_ms;
}

void engine_free(Engine *e) {
    if (!e) return;
    tp_free();
    gguf_free(e->gguf);
    free(e->w.output_norm);
    for (int l = 0; l < e->cfg.n_layers; l++) {
        free(e->w.layer[l].attn_norm);
        free(e->w.layer[l].attn_q_norm);
        free(e->w.layer[l].attn_k_norm);
        free(e->w.layer[l].ffn_norm);
    }
    if (e->mmap_base && e->mmap_base != MAP_FAILED)
        munmap(e->mmap_base, e->mmap_size);
    free(e->s.x); free(e->s.xb); free(e->s.q);
    free(e->s.k_cur); free(e->s.v_cur);
    free(e->s.attn); free(e->s.attn_out);
    free(e->s.ffn_out); free(e->s.tmp); free(e->s.logits);
    free(e->kv.k); free(e->kv.v);
    free(e->tok.vocab);
    tok_free(&e->tok);
    free(e);
}
