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
