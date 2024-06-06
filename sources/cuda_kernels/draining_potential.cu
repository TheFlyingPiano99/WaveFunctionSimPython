#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


extern "C" __global__
void draining_potential_kernel(
    complex<float>* __restrict__ V,

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
    uint3 voxel = get_voxel_coords();
    int idx = get_array_index();

    float3 delta_r = {delta_x, delta_y, delta_z};
    float3 N = {(float)(gridDim.x * blockDim.x), (float)(gridDim.y * blockDim.y), (float)(gridDim.z * blockDim.z)};
    float3 pos = diff(
        mul(delta_r, {voxel.x, voxel.y, voxel.z}),
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
