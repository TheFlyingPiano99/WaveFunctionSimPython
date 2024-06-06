#include <cupy/complex.cuh>

extern "C" float M_PI = 3.14159265359;

extern "C" __device__ float3 scalarVectorMul(float s, const float3& v)
{
    return {s * v.x, s * v.y, s * v.z};
}

extern "C" __device__ float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

extern "C" __device__ complex<float> exp_i(float angle)
{
    return complex<float>(cosf(angle), sinf(angle));
}

extern "C" __device__ complex<float> cexp_i(complex<float> cangle)
{
    return complex<float>(cosf(cangle.real()), sinf(cangle.real())) * expf(-cangle.imag());
}

extern "C" __device__ float3 diff(float3 a, float3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

extern "C" __device__ float3 div(float3 a, float3 b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

extern "C" __global__
void potential_operator_kernel(
    complex<float>* potential_operator,
    complex<float>* V,
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
