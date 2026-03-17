// Ускорение через OpenBLAS.

#include "math_ops.h"

#include <cblas.h>

void linear_layer(float *y, const float *W,
                const float *x, const float *bias,
                int d_in, int d_out) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                d_out, d_in,
                1.f, W, d_in,
                x, 1,
                0.f, y, 1);
    if (bias)
        for (int i = 0; i < d_out; i++) y[i] += bias[i];
}
