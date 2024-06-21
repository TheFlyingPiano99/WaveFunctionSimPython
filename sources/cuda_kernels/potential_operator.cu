#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


extern "C" __global__
void potential_operator_kernel(
    complex<T_WF_FLOAT>* __restrict__ wave_function,
    complex<float>* __restrict__ V, // V is always float!
    T_WF_FLOAT delta_t
)
{
    unsigned int idx = get_array_index();
    T_WF_FLOAT hBar = 1.0;
    complex<float> v = V[idx];
    complex<T_WF_FLOAT> angle = -delta_t / hBar * complex<T_WF_FLOAT>(v.real(), v.imag());
    wave_function[idx] *= cexp_i(angle);
}
