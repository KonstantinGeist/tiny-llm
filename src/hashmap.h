#pragma once

// Хэш-таблица для быстрого доступа к метаданным по ключу.

#include <stdint.h>

#define HM_CAP (1 << 19)
typedef struct { const char *key; int val; } HMEntry;
typedef struct { HMEntry *entries; int cap; } HashMap;

HashMap  hm_new(void);
void     hm_put(HashMap *hm, const char *key, int val);
int      hm_get(const HashMap *hm, const char *key);
