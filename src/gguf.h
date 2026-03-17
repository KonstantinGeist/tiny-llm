#pragma once

// Код для загрузки весов моделей в формате GGUF (см. docs/gguf.md)

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

// Магическое число и версия
#define GGUF_MAGIC   0x46554747u  /* "GGUF" little-endian */
#define GGUF_VERSION 3

// Типы тензоров
typedef enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    GGML_TYPE_MXFP4   = 39,
    GGML_TYPE_COUNT   = 40,
} ggml_type_t;

// Типы значений метаданных
typedef enum {
    GGUF_VAL_UINT8   = 0,
    GGUF_VAL_INT8    = 1,
    GGUF_VAL_UINT16  = 2,
    GGUF_VAL_INT16   = 3,
    GGUF_VAL_UINT32  = 4,
    GGUF_VAL_INT32   = 5,
    GGUF_VAL_FLOAT32 = 6,
    GGUF_VAL_BOOL    = 7,
    GGUF_VAL_STRING  = 8,
    GGUF_VAL_ARRAY   = 9,
    GGUF_VAL_UINT64  = 10,
    GGUF_VAL_INT64   = 11,
    GGUF_VAL_FLOAT64 = 12,
} gguf_val_type_t;

struct gguf_value;

// Строка в файле GGUF
typedef struct {
    uint64_t len;
    char    *str;   /* NUL-терминированная копия */
} gguf_string_t;

// Массив в файле GGUF
typedef struct {
    gguf_val_type_t    type;
    uint64_t           len;
    struct gguf_value *items;
} gguf_array_t;

// Union всех возможных типов
typedef struct gguf_value {
    gguf_val_type_t type;
    union {
        uint8_t       uint8;
        int8_t        int8;
        uint16_t      uint16;
        int16_t       int16;
        uint32_t      uint32;
        int32_t       int32;
        float         float32;
        uint64_t      uint64;
        int64_t       int64;
        double        float64;
        uint8_t       bool_;
        gguf_string_t string;
        gguf_array_t  array;
    };
} gguf_value_t;

// Одна пара "ключ-значение"
typedef struct {
    gguf_string_t key;
    gguf_value_t  value;
} gguf_kv_t;

#define GGUF_MAX_DIMS 4

typedef struct {
    gguf_string_t name;
    uint32_t      n_dims;
    uint64_t      dims[GGUF_MAX_DIMS];
    ggml_type_t   type;
    uint64_t      offset;
    uint64_t      file_offset;
    uint64_t      size_bytes;
} gguf_tensor_info_t;

// Контекст загруженного GGUF-файла
typedef struct {
    uint32_t           version;
    uint64_t           tensor_count;
    uint64_t           metadata_kv_count;
    uint32_t           alignment;        /* general.alignment, по умолчанию 32 */

    gguf_kv_t         *kv;               /* [metadata_kv_count] */
    gguf_tensor_info_t *tensors;         /* [tensor_count] */

    uint64_t           tensor_data_offset; /* абсолютный оффсет к tensor_data */
} gguf_ctx_t;

// Загружает GGUF-файл. Возвращает контекст, созданный на куче, или NULL в случае ошибки (также пишет в stderr).
gguf_ctx_t *gguf_load(const char *path);

// Освободить GGUF-контекст.
void gguf_free(gguf_ctx_t *ctx);

// Найти значения метаданных по ключу. Возвращает NULL, если не найдено.
const gguf_value_t *gguf_get_val(const gguf_ctx_t *ctx, const char *key);

// Возвращает указатель на данные тензора по имени.
// Завершает программу с ошибкой, если тензор не найден.
const void *gguf_tensor_ptr(const gguf_ctx_t *ctx, const uint8_t *base, const char *name);
