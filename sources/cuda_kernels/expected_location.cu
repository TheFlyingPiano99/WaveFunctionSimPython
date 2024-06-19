#include "PATH_TO_SOURCES/cuda_kernels/common.cu"

extern "C" __global__
void expected_location_kernel(
    complex<T_WF_FLOAT>* wave_function,
    T_WF_FLOAT* expected_location,

    float delta_x,
    float delta_y,
    float delta_z,

    int bottom_voxel_x,
    int bottom_voxel_y,
    int bottom_voxel_z,

    int voxel_count_x,
    int voxel_count_y,
    int voxel_count_z
)
{
    extern __shared__ T_WF_FLOAT3 sdata[];

    float3 delta_r = {delta_x, delta_y, delta_z};
    uint3 voxel = get_voxel_coords();
    // Position operator:
    T_WF_FLOAT3 r = {
        delta_x * (T_WF_FLOAT)((int)voxel.x + bottom_voxel_x - voxel_count_x / 2),
        delta_y * (T_WF_FLOAT)((int)voxel.y + bottom_voxel_y - voxel_count_y / 2),
        delta_z * (T_WF_FLOAT)((int)voxel.z + bottom_voxel_z - voxel_count_z / 2)
    };

    unsigned int wf_idx = get_array_index(
        {(unsigned int)(bottom_voxel_x + voxel.x), (unsigned int)(bottom_voxel_y + voxel.y), (unsigned int)(bottom_voxel_z + voxel.z)},
        {(unsigned int)voxel_count_x, (unsigned int)voxel_count_y, (unsigned int)voxel_count_z}
    );  // Have to account for the offset and the size of the measured sub-volume
    unsigned int threadId = get_block_local_idx();
    sdata[threadId] = r * (conj(wave_function[wf_idx]) * wave_function[wf_idx]).real() * delta_r.x * delta_r.y * delta_r.z;

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
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        atomicAdd(&expected_location[0], sdata[0].x);
        atomicAdd(&expected_location[1], sdata[0].y);
        atomicAdd(&expected_location[2], sdata[0].z);
}