#pragma once

#include <stddef.h>
#include <stdint.h>

// Текущее время в миллисекундах
double now_ms(void);

// Выделяет память и копрует массив float32
float *copy_f32(const float *src, size_t n);

// Выделяет память и деквантизирует массив float16 → float32
float *copy_f16(const uint16_t *src, size_t n);
