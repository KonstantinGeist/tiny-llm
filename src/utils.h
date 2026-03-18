#pragma once

#include <stddef.h>
#include <stdint.h>

// Текущее время в миллисекундах
double now_ms(void);

// Выделяет память и копрует массив float32
float *copy_f32(const float *src, size_t n);

// Выделяет память и деквантизирует массив float16 → float32
float *copy_f16(const uint16_t *src, size_t n);

// Выделяет память и деквантизирует массив Q8_0 → float32.
// Q8_0: блоки по 32 элемента, каждый блок = 2 байта (f16 scale) + 32 байта (int8).
// n — количество float-элементов (должно быть кратно 32).
float *copy_q8_0(const void *src, size_t n);
