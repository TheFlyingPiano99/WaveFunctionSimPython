#include "PATH_TO_SOURCES/cuda_kernels/common.cu"

extern "C" __global__
void next_s(
    complex<float>* s_prev, complex<float>* s_next, const complex<float>* v, complex<float>* wave_function,
    float delta_t, float delta_x, float mass, int array_size, int n
)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = i * array_size * array_size + j * array_size + k;
    int idx_n00 = 0;
    if (i > 0) {
        idx_n00 = (i - 1) * array_size * array_size + j * array_size + k;
    }
    else {
        idx_n00 = (i) * array_size * array_size + j * array_size + k;
    }
    int idx_p00 = 0;
    if (i < array_size - 1) {
        idx_p00 = (i + 1) * array_size * array_size + j * array_size + k;
    }
    else {
        idx_p00 = (i) * array_size * array_size + j * array_size + k;
    }
    int idx_0n0 = 0;
    if (j > 0) {
        idx_0n0 = i * array_size * array_size + (j - 1) * array_size + k;
    }
    else {
        idx_0n0 = i * array_size * array_size + (j) * array_size + k;
    }
    int idx_0p0 = 0;
    if (j < array_size - 1) {
        idx_0p0 = i * array_size * array_size + (j + 1) * array_size + k;
    }
    else {
        idx_0p0 = i * array_size * array_size + (j) * array_size + k;
    }
    int idx_00n = 0;
    if (k > 0) {
        idx_00n = i * array_size * array_size + j * array_size + k - 1;
    }
    else {
        idx_00n = i * array_size * array_size + j * array_size + k;
    }
    int idx_00p = 0;
    if (k < array_size - 1) {
        idx_00p = i * array_size * array_size + j * array_size + k + 1;
    }
    else {
        idx_00p = i * array_size * array_size + j * array_size + k;
    }

    complex<float> laplace_s = (
          s_prev[idx_n00]
        + s_prev[idx_p00]
        + s_prev[idx_0n0]
        -6.0f * s_prev[idx]
        + s_prev[idx_0p0]
        + s_prev[idx_00n]
        + s_prev[idx_00p]
    ) / delta_x / delta_x;

    complex<float> s = complex<float>(0.0f, 1.0f) * complex<float>(delta_t / (float)n, 0.0f)
        * (complex<float>(1.0f / 2.0f / mass, 0.0f) * laplace_s - complex<float>(v[idx]) * s_prev[idx]);
    s_next[idx] = s;
    wave_function[idx] += s;
}
