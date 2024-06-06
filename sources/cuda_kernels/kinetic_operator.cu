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

extern "C" __device__ float3 diff(float3 a, float3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

extern "C" __device__ float3 div(float3 a, float3 b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

extern "C" __global__
void kinetic_operator_kernel(
    complex<float>* kinetic_operator,

    float delta_x,
    float delta_y,
    float delta_z,

    float delta_t
)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = i * gridDim.x * blockDim.x * gridDim.y * blockDim.y
            + j * gridDim.x * blockDim.x
            + k;

    float3 f = div(
        {(float)k, (float)j, (float)i},
        {(float)(gridDim.x * blockDim.x - 1), (float)(gridDim.y * blockDim.y - 1), (float)(gridDim.z * blockDim.z - 1)}
    );
    float3 delta_r = {delta_x, delta_y, delta_z};

    // Account for numpy fftn's "negative frequency in second half" pattern
    if (f.x > 0.5f)
        f.x = 1.0f - f.x;
    if (f.y > 0.5f)
        f.y = 1.0f - f.y;
    if (f.z > 0.5f)
        f.z = 1.0f - f.z;

    float3 momentum = scalarVectorMul(2.0f * M_PI, div(f, delta_r));
    float angle = -dot(momentum, momentum) * delta_t / 4.0f;
    kinetic_operator[idx] = exp_i(angle);
}
