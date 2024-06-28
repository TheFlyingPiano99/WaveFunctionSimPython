#include "PATH_TO_SOURCES/cuda_kernels/common.cu"

template<
    unsigned int sample_count_x,    // In the integrated volume
    unsigned int sample_count_y,
    unsigned int sample_count_z,
    int voxel_count_x,  // For the whole simulated volume
    int voxel_count_y,
    int voxel_count_z

>
__global__ void expected_location_kernel(
    complex<T_WF_FLOAT>* waveFunction,
    T_WF_FLOAT* expectedLocationOut,
    T_WF_FLOAT* expectedLocationSquaredOut,

    T_WF_FLOAT delta_x,
    T_WF_FLOAT delta_y,
    T_WF_FLOAT delta_z,

    int bottom_voxel_x,
    int bottom_voxel_y,
    int bottom_voxel_z
)
{
    extern __shared__ T_WF_FLOAT3 sdata[];

    float3 delta_r = {delta_x, delta_y, delta_z};
    uint3 voxel = get_voxel_coords();   // In the integrated volume
    constexpr uint3 sample_count = {sample_count_x, sample_count_y, sample_count_z};  // In the integrated volume

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
    if (voxel.x < sample_count_x && voxel.y < sample_count_y && voxel.z < sample_count_z) {
        sdata[threadId] = r * (conj(waveFunction[wf_idx]) * waveFunction[wf_idx]).real()
            * get_simpson_coefficient_3d<T_WF_FLOAT>(voxel, sample_count) * delta_r.x * delta_r.y * delta_r.z;
    }
    else {
        sdata[threadId] = {(T_WF_FLOAT)0.0, (T_WF_FLOAT)0.0, (T_WF_FLOAT)0.0};
    }
    parallel_reduction_sequential(threadId, sdata);
    // Add the values calculated by the blocks and write the result into output buffer:
    if (threadId == 0) {
        atomicAdd(&expectedLocationOut[0], sdata[0].x);
        atomicAdd(&expectedLocationOut[1], sdata[0].y);
        atomicAdd(&expectedLocationOut[2], sdata[0].z);
    }

    //----------------------------------------------------------------------------------------
    // Calculate E[r^2]
    if (voxel.x < sample_count_x && voxel.y < sample_count_y && voxel.z < sample_count_z) {
        sdata[threadId] = r * r * (conj(waveFunction[wf_idx]) * waveFunction[wf_idx]).real()
            * get_simpson_coefficient_3d<T_WF_FLOAT>(voxel, sample_count) * delta_r.x * delta_r.y * delta_r.z;
    }

    parallel_reduction_sequential(threadId, sdata);
    // Add the values calculated by the blocks and write the result into output buffer:
    if (threadId == 0) {
        atomicAdd(&expectedLocationSquaredOut[0], sdata[0].x);
        atomicAdd(&expectedLocationSquaredOut[1], sdata[0].y);
        atomicAdd(&expectedLocationSquaredOut[2], sdata[0].z);
    }
}