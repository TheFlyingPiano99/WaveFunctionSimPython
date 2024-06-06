#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


extern "C" __global__
void potential_wall_kernel(
    complex<float>* __restrict__ V,

    float delta_x,
    float delta_y,
    float delta_z,

    float center_x,
    float center_y,
    float center_z,

    float normal_x,
    float normal_y,
    float normal_z,

    float thickness_bohr_radius,
    float potential_hartree
)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    int idx = i * gridDim.x * blockDim.x * gridDim.y * blockDim.y
            + j * gridDim.x * blockDim.x
            + k;

    float3 center = {center_x, center_y, center_z};
    float3 normal = {normal_x, normal_y, normal_z};
    float3 delta_r = {delta_x, delta_y, delta_z};

    float3 r = mul(delta_r, transform_corner_origin_to_center_origin_system({(float)i, (float)j, (float)k}));
    float d = dot(normal, diff(center, r));
    if (d <= thickness_bohr_radius * 0.5f && d >= -thickness_bohr_radius * 0.5f)
    {
        V[idx] += potential_hartree * (1.0f - fmaxf(2.0f * fabsf(d / thickness_bohr_radius * 2.0f) - 1.0, 0.0f));
    }
}
