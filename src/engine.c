#include "engine.h"
#include "math_ops.h"
#include "tokenizer.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Гиперпараметры модели.
typedef struct {
    int   d_model, n_layers, n_heads_q, n_heads_kv, head_dim, d_ff;
    int   vocab_size, max_seq_len;
    float rope_freq_base, rms_norm_eps;
} Config;

typedef struct {
    float *token_embd;            /* [vocab_size, d_model] */
    struct {
        float *attn_norm;         /* [d_model]             */
        float *attn_q_w;          /* [n_heads_q * head_dim, d_model] */
        float *attn_q_norm;       /* [head_dim]  — RMSNorm на Q-головах (Qwen3) */
        float *attn_k_w;          /* [n_kv_dim, d_model]   */
        float *attn_k_norm;       /* [head_dim]  — RMSNorm на K-головах (Qwen3) */
        float *attn_v_w;          /* [n_kv_dim, d_model]   */
        float *attn_out_w;        /* [d_model, d_model]    */
        float *ffn_norm;          /* [d_model]             */
        float *ffn_gate_w;        /* [d_ff, d_model]       */
        float *ffn_up_w;          /* [d_ff, d_model]       */
        float *ffn_down_w;        /* [d_model, d_ff]       */
    } layer[32];
    float *output_norm;           /* [d_model]             */
} Weights;

// KV-кэш: [n_layers, max_seq_len, n_heads_kv, head_dim]
typedef struct {
    float *k, *v;
    int n_layers, max_seq_len, n_heads_kv, head_dim;
} KVCache;

// Временные буферы для forward pass
typedef struct {
    float *x, *xb;
    float *q, *k_cur, *v_cur;
    float *attn, *attn_out;
    float *gate, *up;
    float *tmp, *ffn_out;
    float *logits;
} RunState;

// Структура движка.
struct Engine {
    Config    cfg;
    Weights   w;
    KVCache   kv;
    RunState  s;
    Tokenizer tok;
    int pos; // текущая позиция в KV-кэше
    gguf_ctx_t *gguf; // GGUF-контекст, хранится, т.к. держит в себе строками словаря
    // статистика
    long   prefill_tokens;
    double prefill_ms;
    long   gen_tokens;
    double gen_ms;
};

// Применяет RMSNorm поголовно к вектору v размера (n_heads × head_dim),
// используя один набор весов w размера head_dim.
static void rms_norm_heads(float *v, int n_heads, int head_dim,
                            const float *w, float eps) {
    for (int h = 0; h < n_heads; h++) {
        float *vh = v + h * head_dim;
        rms_norm(vh, vh, w, head_dim, eps);
    }
}

// Выполняет forward pass для одного токена в позиции pos
// и возвращает указатель на логиты по всему словарю.
static float *forward(Engine *e, int token, int pos) {
    Config   *cfg = &e->cfg;
    Weights  *w   = &e->w;
    KVCache  *kv  = &e->kv;
    RunState *s   = &e->s;

    int dm    = cfg->d_model;      // размер скрытого состояния модели
    int nq    = cfg->n_heads_q;    // число query-heads
    int nkv   = cfg->n_heads_kv;   // число key/value-heads
    int hd    = cfg->head_dim;     // размер одной головы
    int dff   = cfg->d_ff;         // размер FFN-пространства
    int nkv_d = nkv * hd;          // общий размер K/V-представления

    // Копируем эмбеддинг входного токена в текущее скрытое состояние.
    memcpy(s->x, w->token_embd + (size_t)token * dm, dm * sizeof(float));

    // Последовательно прогоняем состояние через все слои трансформера.
    for (int l = 0; l < cfg->n_layers; l++) {
        // Attention block
        // Нормализуем вход слоя перед вычислением attention.
        rms_norm(s->xb, s->x, w->layer[l].attn_norm, dm, cfg->rms_norm_eps);

        // Строим Q, K и V из нормализованного состояния.
        // В Qwen3 у Q/K/V нет векторов смещений (bias = NULL).
        linear_layer(s->q,     w->layer[l].attn_q_w, s->xb, NULL, dm, nq  * hd);
        linear_layer(s->k_cur, w->layer[l].attn_k_w, s->xb, NULL, dm, nkv_d);
        linear_layer(s->v_cur, w->layer[l].attn_v_w, s->xb, NULL, dm, nkv_d);

        // Qwen3: нормализуем Q и K поголовно перед RoPE.
        rms_norm_heads(s->q,     nq,  hd, w->layer[l].attn_q_norm, cfg->rms_norm_eps);
        rms_norm_heads(s->k_cur, nkv, hd, w->layer[l].attn_k_norm, cfg->rms_norm_eps);

        // Применяем rotary positional embedding к Q и K
        // для учета позиции токена в последовательности.
        rope(s->q,     pos, nq,  hd, cfg->rope_freq_base);
        rope(s->k_cur, pos, nkv, hd, cfg->rope_freq_base);

        // Сохраняем K и V текущего токена в KV-кэш слоя,
        // чтобы использовать их на следующих шагах декодирования.
        float *kc = kv->k + ((size_t)l * cfg->max_seq_len + pos) * nkv_d;
        float *vc = kv->v + ((size_t)l * cfg->max_seq_len + pos) * nkv_d;
        memcpy(kc, s->k_cur, nkv_d * sizeof(float));
        memcpy(vc, s->v_cur, nkv_d * sizeof(float));

        // Grouped Query Attention
        // Несколько Q-голов могут делить одну K/V-голову.
        int group = nq / nkv;

        // Обнуляем буфер, в который собирается результат attention.
        memset(s->attn_out, 0, dm * sizeof(float));

        // Обрабатываем каждую query-head отдельно.
        for (int h = 0; h < nq; h++) {
            int    kv_h   = h / group;                      // какая K/V-голова соответствует этой Q-голове
            float *qh     = s->q + h * hd;                  // вектор query для головы h
            float *scores = s->attn + h * cfg->max_seq_len; // attention-логиты для головы h
            float  scale  = 1.f / sqrtf((float)hd);         // стандартное масштабирование dot-product attention

            // Считаем attention score между текущим query
            // и всеми ключами от позиции 0 до pos включительно.
            for (int t = 0; t <= pos; t++) {
                float *kt = kv->k + ((size_t)l * cfg->max_seq_len + t) * nkv_d + kv_h * hd;
                float  dot = 0.f;
                for (int i = 0; i < hd; i++) dot += qh[i] * kt[i];
                scores[t] = dot * scale;
            }

            // Преобразуем логиты в вероятности внимания.
            softmax(scores, pos + 1);

            // Взвешенно суммируем value-векторы по всем доступным позициям.
            float *oh = s->attn_out + h * hd;
            for (int t = 0; t <= pos; t++) {
                float *vt = kv->v + ((size_t)l * cfg->max_seq_len + t) * nkv_d + kv_h * hd;
                for (int i = 0; i < hd; i++) oh[i] += scores[t] * vt[i];
            }
        }

        // Проецируем результат attention обратно в пространство модели
        // и добавляем residual connection.
        linear_layer(s->tmp, w->layer[l].attn_out_w, s->attn_out, NULL, dm, dm);
        for (int i = 0; i < dm; i++) s->x[i] += s->tmp[i];

        // Feed-Forward block
        // Нормализуем состояние перед FFN.
        rms_norm(s->xb, s->x, w->layer[l].ffn_norm, dm, cfg->rms_norm_eps);

        // Две проекции для gated-FFN: gate и up.
        linear_layer(s->gate, w->layer[l].ffn_gate_w, s->xb, NULL, dm, dff);
        linear_layer(s->up,   w->layer[l].ffn_up_w,   s->xb, NULL, dm, dff);

        // Применяем swish к gate и поэлементно умножаем на up.
        for (int i = 0; i < dff; i++) {
            s->ffn_out[i] = (s->gate[i] / (1.f + expf(-s->gate[i]))) * s->up[i];
        }

        // Возвращаемся в размерность модели и добавляем residual connection.
        linear_layer(s->tmp, w->layer[l].ffn_down_w, s->ffn_out, NULL, dff, dm);
        for (int i = 0; i < dm; i++) s->x[i] += s->tmp[i];
    }

    // Финальная RMSNorm перед вычислением логитов.
    rms_norm(s->xb, s->x, w->output_norm, dm, cfg->rms_norm_eps);

    // Проецируем скрытое состояние в пространство словаря.
    linear_layer(s->logits, w->token_embd, s->xb, NULL, dm, cfg->vocab_size);

    // Возвращаем логиты для выбора следующего токена.
    return s->logits;
}

// Публичное API.

Engine *engine_load(const char *model_path) {
    Engine *e = calloc(1, sizeof(Engine));

    // Загрузка GGUF-файла.
    e->gguf = gguf_load(model_path);
    if (!e->gguf) { free(e); return NULL; }
    gguf_ctx_t *ctx = e->gguf;

    // Извлекаем гиперпараметры из метаданных.
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

    // Загружаем веса в наши float32-буферы.
    // Все матричные тензоры в модели имеют тип Q8_0 — деквантизируем в float32.
    int fd = open(model_path, O_RDONLY);
    struct stat st; fstat(fd, &st);
    uint8_t *base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    Weights *w = &e->w;
    char name[128];
    int dm = c->d_model, dff = c->d_ff, nkv_d = c->n_heads_kv * c->head_dim;
    int nq_d = c->n_heads_q * c->head_dim;

    w->token_embd  = copy_q8_0(gguf_tensor_ptr(ctx, base, "token_embd.weight"), (size_t)c->vocab_size * dm);
    w->output_norm = copy_f32(gguf_tensor_ptr(ctx, base, "output_norm.weight"), dm);

    for (int l = 0; l < c->n_layers; l++) {
#define TQ8(f, fmt, n) snprintf(name, sizeof(name), fmt, l); \
                        w->layer[l].f = copy_q8_0(gguf_tensor_ptr(ctx, base, name), n)
#define TF32(f, fmt, n) snprintf(name, sizeof(name), fmt, l); \
                        w->layer[l].f = copy_f32(gguf_tensor_ptr(ctx, base, name), n)
        TF32(attn_norm,   "blk.%d.attn_norm.weight",   dm);
        TQ8 (attn_q_w,    "blk.%d.attn_q.weight",      (size_t)nq_d  * dm);
        TF32(attn_q_norm, "blk.%d.attn_q_norm.weight", c->head_dim);
        TQ8 (attn_k_w,    "blk.%d.attn_k.weight",      (size_t)nkv_d * dm);
        TF32(attn_k_norm, "blk.%d.attn_k_norm.weight", c->head_dim);
        TQ8 (attn_v_w,    "blk.%d.attn_v.weight",      (size_t)nkv_d * dm);
        TQ8 (attn_out_w,  "blk.%d.attn_output.weight", (size_t)dm    * dm);
        TF32(ffn_norm,    "blk.%d.ffn_norm.weight",    dm);
        TQ8 (ffn_gate_w,  "blk.%d.ffn_gate.weight",    (size_t)dff   * dm);
        TQ8 (ffn_up_w,    "blk.%d.ffn_up.weight",      (size_t)dff   * dm);
        TQ8 (ffn_down_w,  "blk.%d.ffn_down.weight",    (size_t)dm    * dff);
#undef TQ8
#undef TF32
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

    // Токенизатор
    const gguf_value_t *tv = gguf_get_val(ctx, "tokenizer.ggml.tokens");
    const gguf_value_t *mv = gguf_get_val(ctx, "tokenizer.ggml.merges");
    int vocab_size = (int)tv->array.len;
    int n_merges   = (int)mv->array.len;
    char **vocab  = malloc(vocab_size * sizeof(char *));
    char **merges = malloc(n_merges   * sizeof(char *));
    for (int i = 0; i < vocab_size; i++)
        vocab[i]  = tv->array.items[i].string.str;
    for (int i = 0; i < n_merges; i++)
        merges[i] = mv->array.items[i].string.str;
    int bos_id = (int)gguf_get_val(ctx, "tokenizer.ggml.bos_token_id")->uint32;
    int eos_id = (int)gguf_get_val(ctx, "tokenizer.ggml.eos_token_id")->uint32;
    tok_init(&e->tok, vocab, vocab_size, merges, n_merges, bos_id, eos_id);
    free(merges);
    e->tok.vocab = vocab;

    // Состояние инференса
    RunState *s = &e->s;
    int nq = c->n_heads_q, hd = c->head_dim;
    int vs = c->vocab_size, ml = c->max_seq_len;
    s->x        = calloc(dm, sizeof(float));
    s->xb       = calloc(dm, sizeof(float));
    s->q        = calloc(nq * hd, sizeof(float));
    s->k_cur    = calloc(c->n_heads_kv * hd, sizeof(float));
    s->v_cur    = calloc(c->n_heads_kv * hd, sizeof(float));
    s->attn     = calloc(nq * ml, sizeof(float));
    s->attn_out = calloc(dm, sizeof(float));
    s->gate     = calloc(dff, sizeof(float));
    s->up       = calloc(dff, sizeof(float));
    s->tmp      = calloc(dm, sizeof(float));
    s->ffn_out  = calloc(dff, sizeof(float));
    s->logits   = calloc(vs, sizeof(float));

    // KV-кэш
    KVCache *kv = &e->kv;
    kv->n_layers    = c->n_layers;
    kv->max_seq_len = c->max_seq_len;
    kv->n_heads_kv  = c->n_heads_kv;
    kv->head_dim    = c->head_dim;
    size_t kvsz = (size_t)c->n_layers * c->max_seq_len * c->n_heads_kv * hd * sizeof(float);
    kv->k = calloc(1, kvsz);
    kv->v = calloc(1, kvsz);

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

    // prefill
    float *logits = NULL;
    double t0 = now_ms();
    for (int i = 0; i < n; i++)
        logits = forward(e, tokens[i], e->pos++);
    double t1 = now_ms();
    e->prefill_tokens += n;
    e->prefill_ms     += t1 - t0;

    // генерация
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
    gguf_free(e->gguf);

    // веса
    free(e->w.token_embd);
    free(e->w.output_norm);
    for (int l = 0; l < e->cfg.n_layers; l++) {
        free(e->w.layer[l].attn_norm);
        free(e->w.layer[l].attn_q_w);
        free(e->w.layer[l].attn_q_norm);
        free(e->w.layer[l].attn_k_w);
        free(e->w.layer[l].attn_k_norm);
        free(e->w.layer[l].attn_v_w);
        free(e->w.layer[l].attn_out_w);
        free(e->w.layer[l].ffn_norm);
        free(e->w.layer[l].ffn_gate_w);
        free(e->w.layer[l].ffn_up_w);
        free(e->w.layer[l].ffn_down_w);
    }

    // состояние инференса
    free(e->s.x); free(e->s.xb); free(e->s.q);
    free(e->s.k_cur); free(e->s.v_cur);
    free(e->s.attn); free(e->s.attn_out);
    free(e->s.gate); free(e->s.up);
    free(e->s.tmp); free(e->s.ffn_out);
    free(e->s.logits);

    // KV-кэш
    free(e->kv.k); free(e->kv.v);

    // токенизатор
    free(e->tok.vocab);
    tok_free(&e->tok);
    free(e);
}
