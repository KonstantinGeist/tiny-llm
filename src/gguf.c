#include "gguf.h"

#include <stdlib.h>
#include <string.h>
#include <errno.h>

static int read_exact(FILE *f, void *buf, size_t n) {
    if (fread(buf, 1, n, f) != n) {
        fprintf(stderr, "gguf: неожиданный конец файла\n");
        return -1;
    }
    return 0;
}

#define READ(f, var)  read_exact((f), &(var), sizeof(var))
#define CHECK(expr)   do { if ((expr) != 0) goto fail; } while (0)

static int read_string(FILE *f, gguf_string_t *s) {
    uint64_t len;
    CHECK(READ(f, len));
    s->len = len;
    s->str = (char *)malloc(len + 1);
    if (!s->str) { fprintf(stderr, "gguf: не хватает памяти\n"); return -1; }
    CHECK(read_exact(f, s->str, len));
    s->str[len] = '\0';
    return 0;
fail:
    s->str = NULL;
    return -1;
}

static void free_string(gguf_string_t *s) {
    free(s->str);
    s->str = NULL;
    s->len = 0;
}

static int read_value(FILE *f, gguf_val_type_t type, gguf_value_t *out);

static int read_array(FILE *f, gguf_array_t *arr) {
    uint32_t elem_type_raw;
    CHECK(READ(f, elem_type_raw));
    arr->type = (gguf_val_type_t)elem_type_raw;

    uint64_t len;
    CHECK(READ(f, len));
    arr->len = len;

    if (len == 0) {
        arr->items = NULL;
        return 0;
    }

    arr->items = (gguf_value_t *)calloc(len, sizeof(gguf_value_t));
    if (!arr->items) { fprintf(stderr, "gguf: не хватает памяти\n"); return -1; }

    for (uint64_t i = 0; i < len; i++) {
        CHECK(read_value(f, arr->type, &arr->items[i]));
    }
    return 0;
fail:
    free(arr->items);
    arr->items = NULL;
    return -1;
}

static void free_value(gguf_value_t *v);

static void free_array(gguf_array_t *arr) {
    if (arr->items) {
        for (uint64_t i = 0; i < arr->len; i++) {
            free_value(&arr->items[i]);
        }
        free(arr->items);
        arr->items = NULL;
    }
}

static void free_value(gguf_value_t *v) {
    switch (v->type) {
        case GGUF_VAL_STRING: free_string(&v->string); break;
        case GGUF_VAL_ARRAY:  free_array(&v->array);   break;
        default: break;
    }
}

static int read_value(FILE *f, gguf_val_type_t type, gguf_value_t *out) {
    out->type = type;
    switch (type) {
        case GGUF_VAL_UINT8:   CHECK(READ(f, out->uint8));   break;
        case GGUF_VAL_INT8:    CHECK(READ(f, out->int8));    break;
        case GGUF_VAL_UINT16:  CHECK(READ(f, out->uint16));  break;
        case GGUF_VAL_INT16:   CHECK(READ(f, out->int16));   break;
        case GGUF_VAL_UINT32:  CHECK(READ(f, out->uint32));  break;
        case GGUF_VAL_INT32:   CHECK(READ(f, out->int32));   break;
        case GGUF_VAL_FLOAT32: CHECK(READ(f, out->float32)); break;
        case GGUF_VAL_BOOL:    CHECK(READ(f, out->bool_));   break;
        case GGUF_VAL_UINT64:  CHECK(READ(f, out->uint64));  break;
        case GGUF_VAL_INT64:   CHECK(READ(f, out->int64));   break;
        case GGUF_VAL_FLOAT64: CHECK(READ(f, out->float64)); break;
        case GGUF_VAL_STRING:  CHECK(read_string(f, &out->string)); break;
        case GGUF_VAL_ARRAY:   CHECK(read_array(f, &out->array));   break;
        default:
            fprintf(stderr, "gguf: неизвестный тип значения %d\n", (int)type);
            return -1;
    }
    return 0;
fail:
    return -1;
}

static size_t ggml_type_size(ggml_type_t t) {
    switch (t) {
        case GGML_TYPE_F32:     return 4;
        case GGML_TYPE_F16:     return 2;
        case GGML_TYPE_BF16:    return 2;
        case GGML_TYPE_I8:      return 1;
        case GGML_TYPE_I16:     return 2;
        case GGML_TYPE_I32:     return 4;
        case GGML_TYPE_I64:     return 8;
        case GGML_TYPE_F64:     return 8;
        case GGML_TYPE_Q4_0:    return 18;
        case GGML_TYPE_Q4_1:    return 20;
        case GGML_TYPE_Q5_0:    return 22;
        case GGML_TYPE_Q5_1:    return 24;
        case GGML_TYPE_Q8_0:    return 34;
        case GGML_TYPE_Q8_1:    return 36;
        case GGML_TYPE_Q2_K:    return 84;
        case GGML_TYPE_Q3_K:    return 110;
        case GGML_TYPE_Q4_K:    return 144;
        case GGML_TYPE_Q5_K:    return 176;
        case GGML_TYPE_Q6_K:    return 210;
        case GGML_TYPE_Q8_K:    return 292;
        case GGML_TYPE_IQ1_S:   return 18;
        case GGML_TYPE_IQ1_M:   return 18;
        case GGML_TYPE_IQ2_XXS: return 22;
        case GGML_TYPE_IQ2_XS:  return 26;
        case GGML_TYPE_IQ2_S:   return 28;
        case GGML_TYPE_IQ3_XXS: return 34;
        case GGML_TYPE_IQ3_S:   return 38;
        case GGML_TYPE_IQ4_NL:  return 18;
        case GGML_TYPE_IQ4_XS:  return 136;
        case GGML_TYPE_TQ1_0:   return 54;
        case GGML_TYPE_TQ2_0:   return 66;
        case GGML_TYPE_MXFP4:   return 2;
        default:                return 0;
    }
}

static size_t ggml_blk_size(ggml_type_t t) {
    switch (t) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
            return 32;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
            return 256;
        default:
            return 1;
    }
}

static uint64_t tensor_num_elements(const gguf_tensor_info_t *ti) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < ti->n_dims; i++) n *= ti->dims[i];
    return n;
}

static uint64_t tensor_byte_size(const gguf_tensor_info_t *ti) {
    uint64_t ne   = tensor_num_elements(ti);
    size_t   ts   = ggml_type_size(ti->type);
    size_t   bs   = ggml_blk_size(ti->type);
    if (ts == 0 || bs == 0) return 0;
    return (ne / bs) * ts;
}

gguf_ctx_t *gguf_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "gguf: не могу открыть файл '%s': %s\n", path, strerror(errno));
        return NULL;
    }

    gguf_ctx_t *ctx = (gguf_ctx_t *)calloc(1, sizeof(gguf_ctx_t));
    if (!ctx) { fprintf(stderr, "gguf: не хватает памяти\n"); fclose(f); return NULL; }

    ctx->alignment = 32; // по умолчанию согласно спецификации

    // Заголовок
    uint32_t magic;
    if (READ(f, magic) != 0) goto fail;
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "gguf: неправильное магическое число 0x%08X (ожидалось 0x%08X)\n",
                magic, GGUF_MAGIC);
        goto fail;
    }

    uint32_t version;
    if (READ(f, version) != 0) goto fail;
    if (version < 1 || version > GGUF_VERSION) {
        fprintf(stderr, "gguf: неподдерживаемая версия %u\n", version);
        goto fail;
    }
    ctx->version = version;

    uint64_t tensor_count, kv_count;
    if (READ(f, tensor_count) != 0) goto fail;
    if (READ(f, kv_count)     != 0) goto fail;
    ctx->tensor_count      = tensor_count;
    ctx->metadata_kv_count = kv_count;

    // Пары "ключ-значене" для метаданных
    if (kv_count > 0) {
        ctx->kv = (gguf_kv_t *)calloc(kv_count, sizeof(gguf_kv_t));
        if (!ctx->kv) { fprintf(stderr, "gguf: не хватает памяти\n"); goto fail; }
    }

    for (uint64_t i = 0; i < kv_count; i++) {
        gguf_kv_t *kv = &ctx->kv[i];

        if (read_string(f, &kv->key) != 0) goto fail;

        uint32_t val_type_raw;
        if (READ(f, val_type_raw) != 0) goto fail;
        kv->value.type = (gguf_val_type_t)val_type_raw;

        if (read_value(f, kv->value.type, &kv->value) != 0) goto fail;

        if (strcmp(kv->key.str, "general.alignment") == 0
                && kv->value.type == GGUF_VAL_UINT32) {
            ctx->alignment = kv->value.uint32;
        }
    }

    // Тензоры
    if (tensor_count > 0) {
        ctx->tensors = (gguf_tensor_info_t *)calloc(
                tensor_count, sizeof(gguf_tensor_info_t));
        if (!ctx->tensors) { fprintf(stderr, "gguf: не хватает памяти\n"); goto fail; }
    }

    for (uint64_t i = 0; i < tensor_count; i++) {
        gguf_tensor_info_t *ti = &ctx->tensors[i];

        if (read_string(f, &ti->name) != 0) goto fail;

        uint32_t n_dims;
        if (READ(f, n_dims) != 0) goto fail;
        if (n_dims > GGUF_MAX_DIMS) {
            fprintf(stderr, "gguf: тензор '%s' имеет %u измерений (максимум %d)\n",
                    ti->name.str, n_dims, GGUF_MAX_DIMS);
            goto fail;
        }
        ti->n_dims = n_dims;
        for (uint32_t d = 0; d < n_dims; d++) {
            if (READ(f, ti->dims[d]) != 0) goto fail;
        }

        uint32_t type_raw;
        if (READ(f, type_raw) != 0) goto fail;
        ti->type = (ggml_type_t)type_raw;

        if (READ(f, ti->offset) != 0) goto fail;

        ti->size_bytes = tensor_byte_size(ti);
    }

    long cur = ftell(f);
    if (cur < 0) { fprintf(stderr, "gguf: ftell завершился с ошибкой\n"); goto fail; }
    uint64_t pos = (uint64_t)cur;
    uint64_t align = ctx->alignment;
    uint64_t padding = (align - (pos % align)) % align;
    ctx->tensor_data_offset = pos + padding;

    for (uint64_t i = 0; i < tensor_count; i++) {
        ctx->tensors[i].file_offset =
            ctx->tensor_data_offset + ctx->tensors[i].offset;
    }

    fclose(f);
    return ctx;

fail:
    fclose(f);
    gguf_free(ctx);
    return NULL;
}

void gguf_free(gguf_ctx_t *ctx) {
    if (!ctx) return;

    if (ctx->kv) {
        for (uint64_t i = 0; i < ctx->metadata_kv_count; i++) {
            free_string(&ctx->kv[i].key);
            free_value(&ctx->kv[i].value);
        }
        free(ctx->kv);
    }

    if (ctx->tensors) {
        for (uint64_t i = 0; i < ctx->tensor_count; i++) {
            free_string(&ctx->tensors[i].name);
        }
        free(ctx->tensors);
    }

    free(ctx);
}

const gguf_value_t *gguf_get_val(const gguf_ctx_t *ctx, const char *key) {
    for (uint64_t i = 0; i < ctx->metadata_kv_count; i++) {
        if (strcmp(ctx->kv[i].key.str, key) == 0) {
            return &ctx->kv[i].value;
        }
    }
    return NULL;
}

const void *gguf_tensor_ptr(const gguf_ctx_t *ctx, const uint8_t *base,
                             const char *name) {
    for (uint64_t i = 0; i < ctx->tensor_count; i++)
        if (strcmp(ctx->tensors[i].name.str, name) == 0)
            return base + ctx->tensors[i].file_offset;
    fprintf(stderr, "gguf: тензор не найден: %s\n", name);
    return NULL;
}
