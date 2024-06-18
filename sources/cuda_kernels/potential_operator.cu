#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


extern "C" __global__
void potential_operator_kernel(
    complex<T_WF_FLOAT>* __restrict__ wave_function,
    complex<float>* __restrict__ V,
    float delta_t
)
{
    int idx = get_array_index();
    //int inv_idx = get_array_index_inverted();

    float hBar = 1.0f;
    complex<float> angle = (T_WF_FLOAT)(-delta_t / hBar) * V[idx];

    wave_function[idx] *= cexp_i(angle);
}
