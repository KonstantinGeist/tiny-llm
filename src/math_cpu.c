// Простые реализаций функций на CPU, которые могут также ускоряться.

#include "math_ops.h"

#include <stddef.h>

void linear_layer(float *y, const float *W,
                const float *x, const float *bias,
                int d_in, int d_out) {
    for (int i = 0; i < d_out; i++) {
        float acc = bias ? bias[i] : 0.f;
        const float *row = W + (size_t)i * d_in;
        for (int j = 0; j < d_in; j++) acc += row[j] * x[j];
        y[i] = acc;
    }
}
