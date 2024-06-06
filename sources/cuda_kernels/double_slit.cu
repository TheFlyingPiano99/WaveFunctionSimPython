#include "PATH_TO_SOURCES/cuda_kernels/common.cu"

extern "C" __global__
void double_slit_kernel(
    complex<float>* __restrict__ V,

    float delta_x,
    float delta_y,
    float delta_z,

    float center_x,
    float center_y,
    float center_z,

    float thickness_bohr_radii,
    float potential_hartree,
    float space_between_slits_bohr_radii,
    float slit_width_bohr_radii
)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    int idx = i * gridDim.x * blockDim.x * gridDim.y * blockDim.y
            + j * gridDim.x * blockDim.x
            + k;

    float3 center = {center_x, center_y, center_z};
    float3 delta_r = {delta_x, delta_y, delta_z};
    float3 r = delta_r * transform_corner_origin_to_center_origin_system({(float)i, (float)j, (float)k});

    if (
        r.x > center.x - thickness_bohr_radii / 2.0f
        && r.x < center.x + thickness_bohr_radii / 2.0f
        && !(
            (
                r.z
                > center.z
                - space_between_slits_bohr_radii * 0.5f
                - slit_width_bohr_radii
                && r.z
                < center.z
                - space_between_slits_bohr_radii * 0.5f
            )
            or (
                r.z
                < center.z
                + space_between_slits_bohr_radii * 0.5f
                + slit_width_bohr_radii
                and r.z
                > center.z
                + space_between_slits_bohr_radii * 0.5f
            )
        )
    ) {
        V[idx] += potential_hartree;
    }

}
