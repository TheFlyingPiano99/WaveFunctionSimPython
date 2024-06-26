#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


template<unsigned int sample_count_x, unsigned int sample_count_y, unsigned int sample_count_z>
__global__ void volume_probability_kernel(
    complex<T_WF_FLOAT>* wave_function,
    T_WF_FLOAT* probabilityOut,

    T_WF_FLOAT delta_x,
    T_WF_FLOAT delta_y,
    T_WF_FLOAT delta_z,

    unsigned int bottom_voxel_x,    // Offset of the integrated volume inside the simulated volume
    unsigned int bottom_voxel_y,
    unsigned int bottom_voxel_z,

    unsigned int voxel_count_x, // For the whole simulated volume
    unsigned int voxel_count_y,
    unsigned int voxel_count_z

)
{

    extern __shared__ T_WF_FLOAT sdata[];

    T_WF_FLOAT3 delta_r = {delta_x, delta_y, delta_z};
    uint3 voxel = get_voxel_coords();   // In the integrated volume
    uint3 sample_count = {sample_count_x, sample_count_y, sample_count_z};
    unsigned int wf_idx = get_array_index(
        {bottom_voxel_x + voxel.x, bottom_voxel_y + voxel.y, bottom_voxel_z + voxel.z},
        {voxel_count_x, voxel_count_y, voxel_count_z}
    );  // Have to account for the offset and the size of the measured sub-volume

    unsigned int threadId = get_block_local_idx_3d();
    if (voxel.x < sample_count_x && voxel.y < sample_count_y && voxel.z < sample_count_z) {
        sdata[threadId] = (conj(wave_function[wf_idx]) * wave_function[wf_idx]).real()
                            * get_simpson_coefficient_3d<T_WF_FLOAT>(voxel, sample_count) * delta_r.x * delta_r.y * delta_r.z;
    }
    else {
        sdata[threadId] = (T_WF_FLOAT)0;
    }
    parallel_reduction_sequential(threadId, sdata);

    // Add the values calculated by the blocks and write the result into probabilityOut
    if (threadId == 0)
        atomicAdd(probabilityOut, sdata[0]);
}