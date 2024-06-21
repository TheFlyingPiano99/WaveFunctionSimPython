#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


extern "C" __global__
void wave_packet_kernel(
    complex<T_WF_FLOAT>* __restrict__ wave_tensor,

    T_WF_FLOAT delta_x,
    T_WF_FLOAT delta_y,
    T_WF_FLOAT delta_z,

    T_WF_FLOAT a,

    T_WF_FLOAT r_x,
    T_WF_FLOAT r_y,
    T_WF_FLOAT r_z,

    T_WF_FLOAT k_x,
    T_WF_FLOAT k_y,
    T_WF_FLOAT k_z
)
{
    uint3 voxel = get_voxel_coords();
    int idx = get_array_index();

    T_WF_FLOAT3 k_0 = {k_x, k_y, k_z};
    T_WF_FLOAT3 r_0 = {r_x, r_y, r_z};
    T_WF_FLOAT3 delta_r = {delta_x, delta_y, delta_z};
    // Position operator:
    T_WF_FLOAT3 r = {
        delta_r.x * (T_WF_FLOAT)((int)voxel.x - (int)(gridDim.x * blockDim.x) / 2),
        delta_r.y * (T_WF_FLOAT)((int)voxel.y - (int)(gridDim.y * blockDim.y) / 2),
        delta_r.z * (T_WF_FLOAT)((int)voxel.z - (int)(gridDim.z * blockDim.z) / 2)
    };
    wave_tensor[idx] = gaussian_wave_packet<T_WF_FLOAT, T_WF_FLOAT3>(a, r, r_0, k_0);
}
