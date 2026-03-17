// Реализация основных функций на CPU, у которых не ускорения.

#include "math_ops.h"

#include <math.h>

void rms_norm(
    float *out,     // выходной вектор
    const float *x, // входной вектор
    const float *w, // предобученная матрица
    int n,          // размер вектора
    float eps       // эпсилон: чтобы избежать деления на ноль
) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = w[i] * x[i] * ss;
}

void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0.f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

void rope(float *v, int pos, int n_heads, int head_dim, float freq_base) {
    for (int h = 0; h < n_heads; h++) {
        float *vh = v + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.f / powf(freq_base, 2.f * i / head_dim);
            float c = cosf(pos * freq), s = sinf(pos * freq);
            float v0 = vh[i], v1 = vh[i + head_dim / 2];
            vh[i]               = v0 * c - v1 * s;
            vh[i + head_dim / 2] = v0 * s + v1 * c;
        }
    }
}

float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float r; __builtin_memcpy(&r, &f, 4); return r;
}

int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}
