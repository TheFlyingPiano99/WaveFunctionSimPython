#ifndef CUDA_COMMON_INCLUDE
#define CUDA_COMMON_INCLUDE

#include <cupy/complex.cuh>


constexpr float M_PI = 3.14159265359;

__device__ float3 fabsf(const float3& v)
{
    return {fabsf(v.x), fabsf(v.y), fabsf(v.z)};
}

__device__ complex<float> conj(const complex<float>& c)
{
    return complex<float>(c.real(), -c.imag());
}

__device__ complex<float> mul(const complex<float>& a, const complex<float>& b)
{
    return complex<float>(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real() );
}

__device__ constexpr float3 scalarVectorMul(const float s, const float3& v)
{
    return {s * v.x, s * v.y, s * v.z};
}

__device__ constexpr float3 operator*(const float s, const float3& v)
{
    return {s * v.x, s * v.y, s * v.z};
}

__device__ constexpr float3 operator*(const float3& v, const float s)
{
    return {s * v.x, s * v.y, s * v.z};
}

__device__ constexpr float3 operator-(const float3& v)
{
    return {-v.x, -v.y, -v.z};
}

constexpr __device__ float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

constexpr __device__ float3 cross(const float3& a, const float3& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__device__ const complex<float> exp_i(float angle)
{
    return complex<float>(cosf(angle), sinf(angle));
}

__device__ const complex<float> cexp_i(const complex<float>& cangle)
{
    return complex<float>(cosf(cangle.real()), sinf(cangle.real())) * expf(-cangle.imag());
}

__device__ constexpr float3 add(const float3& a, const float3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ constexpr float3 operator+(const float3& a, const float3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ constexpr float3 diff(const float3& a, const float3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ constexpr float3 operator-(const float3& a, const float3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ float3 mul(const float3& a, const float3 b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ float3 operator*(const float3& a, const float3 b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ constexpr float3 div(const float3& a, const float3& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__device__ constexpr float3 operator/(const float3& a, const float3& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__device__ float length(const float3& a)
{
    return sqrtf(dot(a, a));
}

__device__ float3 normalize(const float3& a)
{
    return scalarVectorMul(1.0f / sqrtf(dot(a, a)), a);
}

__device__ constexpr float3 transform_corner_origin_to_center_origin_system(const float3& pos)
{
    return diff(
        pos,
        scalarVectorMul(0.5f, {(float)(gridDim.x * blockDim.x), (float)(gridDim.y * blockDim.y), (float)(gridDim.z * blockDim.z)})
    );
}

__device__ uint3 get_voxel_count()
{
    return {
        gridDim.x * blockDim.x,
        gridDim.y * blockDim.y,
        gridDim.z * blockDim.z
    };
}

__device__ uint3 get_voxel_coords()
{
    return {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z
    };
}

__device__ uint3 get_voxel_coords_inverted()
{
    unsigned int x = gridDim.x * blockDim.x - (blockIdx.x * blockDim.x + threadIdx.x) - 1;
    unsigned int y = gridDim.y * blockDim.y - (blockIdx.y * blockDim.y + threadIdx.y) - 1;
    unsigned int z = gridDim.z * blockDim.z - (blockIdx.z * blockDim.z + threadIdx.z) - 1;
    return {x, y, z};
}

__device__ int get_array_index()
{
    uint3 voxel = get_voxel_coords();
    return voxel.x * gridDim.y * blockDim.y * gridDim.z * blockDim.z
            + voxel.y * gridDim.z * blockDim.z
            + voxel.z;
}

__device__ int get_array_index(const uint3& voxel)
{
    return voxel.x * gridDim.y * blockDim.y * gridDim.z * blockDim.z
            + voxel.y * gridDim.z * blockDim.z
            + voxel.z;
}

__device__ int get_array_index(const uint3& voxel, const uint3& N)
{
    return voxel.x * N.y * N.z
            + voxel.y * N.z
            + voxel.z;
}

__device__ int get_array_index_inverted()
{
    uint3 voxel = get_voxel_coords_inverted();
    return voxel.x * gridDim.y * blockDim.y * gridDim.z * blockDim.z
            + voxel.y * gridDim.z * blockDim.z
            + voxel.z;
}


__device__ float3 operator*(const float (&m)[3][3], const float3& v)
{
    return {
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
    };
}

__device__ float4 operator*(const float (&m)[4][4], const float4& v)
{
    return {
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
        m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w,
    };
}

__device__ float3 rotate_vector(const float3& v, const float3& axis, float rad)
{
    float q0 = cosf(rad / 2.0f);
    float q1 = sinf(rad / 2.0f) * axis.x;
    float q2 = sinf(rad / 2.0f) * axis.y;
    float q3 = sinf(rad / 2.0f) * axis.z;
    float Q[3][3] = { { 0.0f } }; // 3x3 rotation matrix

    Q[0][0] = q0*q0 + q1*q1 - q2*q2 - q3*q3;
    Q[0][1] = 2.0f * (q1*q2 - q0*q3);
    Q[0][2] = 2.0f * (q1*q3 + q0*q2);

    Q[1][0] = 2.0f * (q2*q1 + q0*q3);
    Q[1][1] = q0*q0 - q1*q1 + q2*q2 - q3*q3;
    Q[1][2] = 2.0f * (q2*q3 - q0*q1);

    Q[2][0] = 2.0f * (q3*q1 - q0*q2);
    Q[2][1] = 2.0f * (q3*q2 + q0*q1);
    Q[2][2] = q0*q0 - q1*q1 - q2*q2 + q3*q3;

    return Q * v;
}

__device__ float mix(const float3& xyz, float u, float v)
{
    return xyz.z * v + (1.0f - v) * (xyz.y * u + xyz.x * (1.0f - u));
}

#endif  // CUDA_COMMON
