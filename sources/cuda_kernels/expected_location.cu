#include "PATH_TO_SOURCES/cuda_kernels/common.cu"

extern "C" __global__
void expected_location_kernel(
    complex<T_WF_FLOAT>* waveFunction,
    T_WF_FLOAT* expectedLocationOut,

    T_WF_FLOAT delta_x,
    T_WF_FLOAT delta_y,
    T_WF_FLOAT delta_z,

    int bottom_voxel_x,
    int bottom_voxel_y,
    int bottom_voxel_z,

    int voxel_count_x,  // For the whole simulated volume
    int voxel_count_y,
    int voxel_count_z
)
{
    extern __shared__ T_WF_FLOAT3 sdata[];

    float3 delta_r = {delta_x, delta_y, delta_z};
    uint3 voxel = get_voxel_coords();   // In the integrated volume

    // Position operator:
    T_WF_FLOAT3 r = {
        delta_r.x * (T_WF_FLOAT)((int)voxel.x + bottom_voxel_x - voxel_count_x / 2),
        delta_r.y * (T_WF_FLOAT)((int)voxel.y + bottom_voxel_y - voxel_count_y / 2),
        delta_r.z * (T_WF_FLOAT)((int)voxel.z + bottom_voxel_z - voxel_count_z / 2)
    };

    unsigned int wf_idx = get_array_index(
        {(unsigned int)(bottom_voxel_x + voxel.x), (unsigned int)(bottom_voxel_y + voxel.y), (unsigned int)(bottom_voxel_z + voxel.z)},
        {(unsigned int)voxel_count_x, (unsigned int)voxel_count_y, (unsigned int)voxel_count_z}
    );  // Have to account for the offset and the size of the measured sub-volume
    unsigned int threadId = get_block_local_idx_3d();
    sdata[threadId] = r * (conj(waveFunction[wf_idx]) * waveFunction[wf_idx]).real()
        * get_simpson_coefficient_3d<T_WF_FLOAT>(voxel) * delta_r.x * delta_r.y * delta_r.z;

    __syncthreads();

    // Reduction in shared memory:
    unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;
    for (unsigned int s = 1; s < blockSize; s *= 2) {
        if (threadId % (2 * s) == 0 && (threadId + s) < blockSize) {
            sdata[threadId] += sdata[threadId + s];
        }
        __syncthreads();
    }

    // Add the values calculated by the blocks and write the result into output buffer:
    if (threadId == 0)
        atomicAdd(&expectedLocationOut[0], sdata[0].x);
        atomicAdd(&expectedLocationOut[1], sdata[0].y);
        atomicAdd(&expectedLocationOut[2], sdata[0].z);
}