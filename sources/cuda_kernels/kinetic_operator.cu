#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


extern "C" __global__
void kinetic_operator_kernel(
    complex<T_WF_FLOAT>* __restrict__ k_space_wave_function,

    float delta_x,
    float delta_y,
    float delta_z,

    float delta_t,
    float mass,

    double* frequencies_x,
    double* frequencies_y,
    double* frequencies_z
)
{
    uint3 voxel = get_voxel_coords();
    int idx = get_array_index();

    // Calculate wavenumber:
    float3 k = 2.0f * M_PI * float3{frequencies_x[voxel.x], frequencies_y[voxel.y], frequencies_z[voxel.z]};
    float hBar = 1.0f;
    float angle = -dot(k, k) * hBar * delta_t / 4.0f / mass;
    k_space_wave_function[idx] *= exp_i(angle);
}
