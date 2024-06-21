#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


extern "C" __global__
void kinetic_operator_kernel(
    complex<T_WF_FLOAT>* __restrict__ k_space_wave_function,

    T_WF_FLOAT delta_t,
    T_WF_FLOAT mass,

    double* wave_number_x,
    double* wave_number_y,
    double* wave_number_z
)
{
    uint3 voxel = get_voxel_coords();
    unsigned int idx = get_array_index();

    // Calculate wavenumber:
    T_WF_FLOAT3 k = T_WF_FLOAT3{wave_number_x[voxel.x], wave_number_y[voxel.y], wave_number_z[voxel.z]};
    T_WF_FLOAT hBar = 1.0;
    T_WF_FLOAT angle = -dot(k, k) * hBar * delta_t / 4.0f / mass;
    k_space_wave_function[idx] *= exp_i(angle);
}
