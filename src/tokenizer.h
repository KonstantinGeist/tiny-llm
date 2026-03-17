#pragma once

// BPE-токенизатор.

#include "hashmap.h"

typedef struct {
    HashMap  vocab_map;   // строка => token_id
    HashMap  merge_map;   // пара токенов "A B" → ранг BPE-слияния
    char     **vocab;
    int      vocab_size;
    int      bos_id, eos_id;
} Tokenizer;

// Инициазилизировать токенизатор на основе готового словаря и merge-массивов (жизнью строк управляет код извне)
// vocab[i] это строка для token_id i.
// merges[i] это правила слияния строки "A B" для i. */
void tok_init(Tokenizer *t,
              char **vocab, int vocab_size,
              char **merges, int n_merges,
              int bos_id, int eos_id);

// Токенизировать текст в token ID.
// out должен иметь место хотя бы для 2 * strlen(text) + 16 записей
// Возвращает число токенов, записанных в out.
int tok_encode(const Tokenizer *t, const char *text, int *out);

// Декодировать ID токена в UTF8-строку.
// Возвращает указатель на статический буфер, который валиден до следующего вызова.
const char *tok_decode(const Tokenizer *t, int token_id);

// Удаляет память токенизатора (но не строки словари).
void tok_free(Tokenizer *t);
