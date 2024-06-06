#ifndef CUDA_COMMON_INCLUDE
#define CUDA_COMMON_INCLUDE

#include <cupy/complex.cuh>


constexpr float M_PI = 3.14159265359;

constexpr __device__ float3 scalarVectorMul(const float s, const float3& v)
{
    return {s * v.x, s * v.y, s * v.z};
}

constexpr __device__ float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
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

__device__ constexpr float3 diff(const float3& a, const float3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ float3 mul(const float3& a, const float3 b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ constexpr float3 div(const float3& a, const float3& b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__device__ constexpr float3 transform_corner_origin_to_center_origin_system(const float3& pos)
{
    return diff(
        pos,
        scalarVectorMul(0.5f, {(float)(gridDim.x * blockDim.x), (float)(gridDim.y * blockDim.y), (float)(gridDim.z * blockDim.z)})
    );
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
    int x = gridDim.x * blockDim.x - (blockIdx.x * blockDim.x + threadIdx.x) - 1;
    int y = gridDim.y * blockDim.y - (blockIdx.y * blockDim.y + threadIdx.y) - 1;
    int z = gridDim.z * blockDim.z - (blockIdx.z * blockDim.z + threadIdx.z) - 1;
    return {x, y, z};
}

__device__ int get_array_index()
{
    uint3 voxel = get_voxel_coords();
    return voxel.x * gridDim.y * blockDim.y * gridDim.z * blockDim.z
            + voxel.y * gridDim.z * blockDim.z
            + voxel.z;
}

__device__ int get_array_index_inverted()
{
    uint3 voxel = get_voxel_coords_inverted();
    return voxel.x * gridDim.y * blockDim.y * gridDim.z * blockDim.z
            + voxel.y * gridDim.z * blockDim.z
            + voxel.z;
}


#endif  // CUDA_COMMON
