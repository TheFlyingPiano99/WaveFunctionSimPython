#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


extern "C" __global__
void potential_operator_kernel(
    complex<float>* __restrict__ potential_operator,
    complex<float>* __restrict__ V,
    float delta_t
)
{
    int idx = get_array_index();
    int inv_idx = get_array_index_inverted();

    complex<float> angle = -delta_t * V[inv_idx];

    potential_operator[idx] = cexp_i(angle);
}
