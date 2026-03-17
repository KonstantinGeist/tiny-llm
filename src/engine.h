#pragma once

// Ядро инференса модели Qwen2.5

#include <stdint.h>
#include "gguf.h"
#include "hashmap.h"

// Opaque-структура, детали реализации в engine.c
typedef struct Engine Engine;

// Вызывается один раз на каждый сгенерированный токен.
typedef int (*TokenCallback)(Engine *e, int token_id, void *ctx);

// Запускает движок, загружая веса из GGUF-файла по указанному пути.
// Возвращает NULL в случае ошибки.
Engine *engine_load(const char *model_path);

// Запускает инференс на предформатированную строку промпта.
// Вызыввет cb(engine, token_id, cb_ctx) на каждый сгенерированный токен.
void engine_generate(Engine *e, const char *prompt,
                     TokenCallback cb, void *cb_ctx);

// Декодирует ID токена в UTF8-строку.
// Возвращённый указатель валиден до следующего вызова.
const char *engine_decode_token(Engine *e, int token_id);

// Статистика генерации.
typedef struct {
    long   prefill_tokens;
    double prefill_ms;
    long   gen_tokens;
    double gen_ms;
} EngineStats;

void engine_get_stats(const Engine *e, EngineStats *out);

void engine_free(Engine *e);
