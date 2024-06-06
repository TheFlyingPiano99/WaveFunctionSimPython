#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


extern "C" __global__
void potential_operator_kernel(
    complex<float>* __restrict__ potential_operator,
    complex<float>* __restrict__ V,
    float delta_t
)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    int vK = gridDim.x * blockDim.x - k - 1;
    int vJ = gridDim.y * blockDim.y - j - 1;
    int vI = gridDim.z * blockDim.z - i - 1;
    int vIdx = vI * gridDim.x * blockDim.x * gridDim.y * blockDim.y
            + vJ * gridDim.x * blockDim.x
            + vK;
    complex<float> angle = -delta_t * V[vIdx];

    int idx = i * gridDim.x * blockDim.x * gridDim.y * blockDim.y
            + j * gridDim.x * blockDim.x
            + k;
    potential_operator[idx] = cexp_i(angle);
}
