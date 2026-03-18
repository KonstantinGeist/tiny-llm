#include "utils.h"
#include "math_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

float *copy_f32(const float *src, size_t n) {
    if (!src) return NULL;
    float *dst = malloc(n * sizeof(float));
    memcpy(dst, src, n * sizeof(float));
    return dst;
}

float *copy_f16(const uint16_t *src, size_t n) {
    if (!src) return NULL;
    float *dst = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) dst[i] = f16_to_f32(src[i]);
    return dst;
}

float *copy_q8_0(const void *src, size_t n) {
    if (!src) return NULL;
    float *dst = malloc(n * sizeof(float));
    // Q8_0: блоки по 32 элемента = 2 байта (f16 scale) + 32 байта (int8)
    const uint8_t *p = (const uint8_t *)src;
    size_t blocks = n / 32;
    for (size_t b = 0; b < blocks; b++) {
        // Первые 2 байта блока — масштаб в формате f16
        uint16_t scale_bits;
        memcpy(&scale_bits, p, sizeof(uint16_t));
        float scale = f16_to_f32(scale_bits);
        p += 2;
        // Следующие 32 байта — квантованные int8-значения
        for (int i = 0; i < 32; i++) {
            dst[b * 32 + i] = scale * (float)((int8_t)p[i]);
        }
        p += 32;
    }
    return dst;
}
