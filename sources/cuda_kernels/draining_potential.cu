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

extern "C" __device__ float3 mul(float3 a, float3 b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

extern "C" __device__ float3 div(float3 a, float3 b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}


extern "C" __global__
void draining_potential_kernel(
    complex<float>* V,

    float delta_x,
    float delta_y,
    float delta_z,

    float ellipsoid_a,
    float ellipsoid_b,
    float ellipsoid_c,

    float inner_ellipsoid_distance_bohr_radii,

    float max_potential_hartree,
    float exponent
)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    int idx = i * gridDim.x * blockDim.x * gridDim.y * blockDim.y
            + j * gridDim.x * blockDim.x
            + k;

    float3 delta_r = {delta_x, delta_y, delta_z};
    float3 N = {(float)(gridDim.x * blockDim.x), (float)(gridDim.y * blockDim.y), (float)(gridDim.z * blockDim.z)};
    float3 pos = diff(
        mul(delta_r, {k, j, i}),
        scalarVectorMul(0.5f, mul(N, delta_r))
    );
    float ellipsoid_distance =
        pos.x * pos.x / ellipsoid_a / ellipsoid_a
        + pos.y * pos.y / ellipsoid_b / ellipsoid_b
        + pos.z * pos.z / ellipsoid_c / ellipsoid_c
        - 1.0f;     // The implicit equation of the ellipsoid

    float t = fmaxf(
            0.0f,
            ellipsoid_distance - inner_ellipsoid_distance_bohr_radii
        ) / -inner_ellipsoid_distance_bohr_radii;
    V[idx] += complex<float>(0.0f, powf(t, exponent) * max_potential_hartree);
}
