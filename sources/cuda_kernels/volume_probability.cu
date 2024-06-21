#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


extern "C" __global__
void volume_probability_kernel(
    complex<T_WF_FLOAT>* wave_function,
    T_WF_FLOAT* probability,

    T_WF_FLOAT delta_x,
    T_WF_FLOAT delta_y,
    T_WF_FLOAT delta_z,

    unsigned int bottom_voxel_x,
    unsigned int bottom_voxel_y,
    unsigned int bottom_voxel_z,

    unsigned int voxel_count_x,
    unsigned int voxel_count_y,
    unsigned int voxel_count_z
)
{
    extern __shared__ T_WF_FLOAT sdata[];

    T_WF_FLOAT3 delta_r = {delta_x, delta_y, delta_z};
    uint3 voxel = get_voxel_coords();   // In the integrated volume
    uint3 n = get_voxel_count_3d();    // In the integrated volume
    unsigned int wf_idx = get_array_index(
        {bottom_voxel_x + voxel.x, bottom_voxel_y + voxel.y, bottom_voxel_z + voxel.z},
        {voxel_count_x, voxel_count_y, voxel_count_z}
    );  // Have to account for the offset and the size of the measured sub-volume

    unsigned int threadId = get_block_local_idx_3d();
    sdata[threadId] = (conj(wave_function[wf_idx]) * wave_function[wf_idx]).real()
                        * get_simpson_coefficient_3d<T_WF_FLOAT>(voxel) * delta_r.x * delta_r.y * delta_r.z;

    __syncthreads();

    // Reduction in shared memory
    unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;
    for (unsigned int s = 1; s < blockSize; s *= 2) {
        if (threadId % (2 * s) == 0 && (threadId + s) < blockSize) {
            sdata[threadId] += sdata[threadId + s];
        }
        __syncthreads();
    }

    // Add the values calculated by the blocks and write the result into probability
    if (threadId == 0)
        atomicAdd(probability, sdata[0]);
}