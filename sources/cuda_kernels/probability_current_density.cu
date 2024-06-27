#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


template<
    unsigned int sample_count_x,    // For the integrated plane only
    unsigned int sample_count_y,
    unsigned int volume_dim_x,  // For the whole simulated volume
    unsigned int volume_dim_y,
    unsigned int volume_dim_z
>
__global__ void probability_current_density_kernel(
    complex<T_WF_FLOAT>* __restrict__ wave_function,
    T_WF_FLOAT* __restrict__ probability_current_density,
    T_WF_FLOAT* probabilityCurrentOut,

    T_WF_FLOAT mass,

    T_WF_FLOAT delta_x,
    T_WF_FLOAT delta_y,
    T_WF_FLOAT delta_z,

    float center_x,
    float center_y,
    float center_z,

    float normal_x,
    float normal_y,
    float normal_z,

    float width,
    float height
)
{
    extern __shared__ T_WF_FLOAT sdata[];

    T_WF_FLOAT3 delta_r = {delta_x, delta_y, delta_z};
    float3 center = {center_x, center_y, center_z};
    float3 normal = {normal_x, normal_y, normal_z};
    constexpr uint2 sample_count = {sample_count_x, sample_count_y};
    constexpr uint3 N = {volume_dim_x, volume_dim_y, volume_dim_z};   // For the whole simulated volume
    float3 prefUp = {0.0f, 1.0f, 0.0f};
    if (fabsf(dot(normal, prefUp)) > 0.99)  // Change preferred up vector
    {
        prefUp.y = 0.0f;
        prefUp.z = 1.0f;
    }

    float3 right = normalize(cross(normal, prefUp));
    float3 up = normalize(cross(right, normal));

    T_WF_FLOAT dW = width / (T_WF_FLOAT)(sample_count_x - 1);
    T_WF_FLOAT dH = height / (T_WF_FLOAT)(sample_count_y - 1);

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    float3 f_r = center + right * (float)dW * (float)((int)i - (int)(sample_count_x / 2))
        + up * (float)dH * (float)((int)j - (int)(sample_count_y / 2));
    T_WF_FLOAT3 r = {f_r.x, f_r.y, f_r.z};
    int planeIdx = get_array_index();
    T_WF_FLOAT pcDensity;
    T_WF_FLOAT3 fVoxel = r / delta_r + 0.5f * T_WF_FLOAT3{(T_WF_FLOAT)N.x, (T_WF_FLOAT)N.y, (T_WF_FLOAT)N.z};
    uint3 voxel = {(unsigned int)fVoxel.x, (unsigned int)fVoxel.y, (unsigned int)fVoxel.z};

    // Calculate array indices:

    unsigned int idx = get_array_index(voxel, N);

    // 1th step:
    unsigned int idx_n00 = idx;
    if (voxel.x > 0) {
        idx_n00 = get_array_index({voxel.x - 1, voxel.y, voxel.z}, N);
    }
    unsigned int idx_p00 = idx;
    if (voxel.x < N.x - 1) {
        idx_p00 = get_array_index({voxel.x + 1, voxel.y, voxel.z}, N);
    }
    unsigned int idx_0n0 = idx;
    if (voxel.y > 0) {
        idx_0n0 = get_array_index({voxel.x, voxel.y - 1, voxel.z}, N);
    }
    unsigned int idx_0p0 = idx;
    if (voxel.y < N.y - 1) {
        idx_0p0 = get_array_index({voxel.x, voxel.y + 1, voxel.z}, N);
    }
    unsigned int idx_00n = idx;
    if (voxel.z > 0) {
        idx_00n = get_array_index({voxel.x, voxel.y, voxel.z - 1}, N);
    }
    unsigned int idx_00p = idx;
    if (voxel.z < N.z - 1) {
        idx_00p = get_array_index({voxel.x, voxel.y, voxel.z + 1}, N);
    }

    // 2nd step:
    unsigned int idx_2_n00 = idx;
    if (voxel.x > 1) {
        idx_2_n00 = get_array_index({voxel.x - 2, voxel.y, voxel.z}, N);
    }
    unsigned int idx_2_p00 = idx;
    if (voxel.x < N.x - 2) {
        idx_2_p00 = get_array_index({voxel.x + 2, voxel.y, voxel.z}, N);
    }
    unsigned int idx_2_0n0 = idx;
    if (voxel.y > 1) {
        idx_2_0n0 = get_array_index({voxel.x, voxel.y - 2, voxel.z}, N);
    }
    unsigned int idx_2_0p0 = idx;
    if (voxel.y < N.y - 2) {
        idx_2_0p0 = get_array_index({voxel.x, voxel.y + 2, voxel.z}, N);
    }
    unsigned int idx_2_00n = idx;
    if (voxel.z > 1) {
        idx_2_00n = get_array_index({voxel.x, voxel.y, voxel.z - 2}, N);
    }
    unsigned int idx_2_00p = idx;
    if (voxel.z < N.z - 2) {
        idx_2_00p = get_array_index({voxel.x, voxel.y, voxel.z + 2}, N);
    }

    // 3nd step:
    unsigned int idx_3_n00 = idx;
    if (voxel.x > 2) {
        idx_3_n00 = get_array_index({voxel.x - 3, voxel.y, voxel.z}, N);
    }
    unsigned int idx_3_p00 = idx;
    if (voxel.x < N.x - 3) {
        idx_3_p00 = get_array_index({voxel.x + 3, voxel.y, voxel.z}, N);
    }
    unsigned int idx_3_0n0 = idx;
    if (voxel.y > 2) {
        idx_3_0n0 = get_array_index({voxel.x, voxel.y - 3, voxel.z}, N);
    }
    unsigned int idx_3_0p0 = idx;
    if (voxel.y < N.y - 3) {
        idx_3_0p0 = get_array_index({voxel.x, voxel.y + 3, voxel.z}, N);
    }
    unsigned int idx_3_00n = idx;
    if (voxel.z > 2) {
        idx_3_00n = get_array_index({voxel.x, voxel.y, voxel.z - 3}, N);
    }
    unsigned int idx_3_00p = idx;
    if (voxel.z < N.z - 3) {
        idx_3_00p = get_array_index({voxel.x, voxel.y, voxel.z + 3}, N);
    }

    // Sample:

    // 1th step:
    complex<T_WF_FLOAT> psi = wave_function[idx];
    complex<T_WF_FLOAT> wf_n00 = wave_function[idx_n00];
    complex<T_WF_FLOAT> wf_p00 = wave_function[idx_p00];
    complex<T_WF_FLOAT> wf_0n0 = wave_function[idx_0n0];
    complex<T_WF_FLOAT> wf_0p0 = wave_function[idx_0p0];
    complex<T_WF_FLOAT> wf_00n = wave_function[idx_00n];
    complex<T_WF_FLOAT> wf_00p = wave_function[idx_00p];

    // 2th step:
    complex<T_WF_FLOAT> wf_2_n00 = wave_function[idx_2_n00];
    complex<T_WF_FLOAT> wf_2_p00 = wave_function[idx_2_p00];
    complex<T_WF_FLOAT> wf_2_0n0 = wave_function[idx_2_0n0];
    complex<T_WF_FLOAT> wf_2_0p0 = wave_function[idx_2_0p0];
    complex<T_WF_FLOAT> wf_2_00n = wave_function[idx_2_00n];
    complex<T_WF_FLOAT> wf_2_00p = wave_function[idx_2_00p];

    // 3th step:
    complex<T_WF_FLOAT> wf_3_n00 = wave_function[idx_3_n00];
    complex<T_WF_FLOAT> wf_3_p00 = wave_function[idx_3_p00];
    complex<T_WF_FLOAT> wf_3_0n0 = wave_function[idx_3_0n0];
    complex<T_WF_FLOAT> wf_3_0p0 = wave_function[idx_3_0p0];
    complex<T_WF_FLOAT> wf_3_00n = wave_function[idx_3_00n];
    complex<T_WF_FLOAT> wf_3_00p = wave_function[idx_3_00p];

    // Gradient (Seven-point stencil):
    complex<T_WF_FLOAT> dX = (wf_3_p00 - (T_WF_FLOAT)9.0 * wf_2_p00 + (T_WF_FLOAT)45.0 * wf_p00 - (T_WF_FLOAT)45.0 * wf_n00 + (T_WF_FLOAT)9.0 * wf_2_n00 - wf_3_n00) / (T_WF_FLOAT)60 / delta_x;
    complex<T_WF_FLOAT> dY = (wf_3_0p0 - (T_WF_FLOAT)9.0 * wf_2_0p0 + (T_WF_FLOAT)45.0 * wf_0p0 - (T_WF_FLOAT)45.0 * wf_0n0 + (T_WF_FLOAT)9.0 * wf_2_0n0 - wf_3_0n0) / (T_WF_FLOAT)60 / delta_y;
    complex<T_WF_FLOAT> dZ = (wf_3_00p - (T_WF_FLOAT)9.0 * wf_2_00p + (T_WF_FLOAT)45.0 * wf_00p - (T_WF_FLOAT)45.0 * wf_00n + (T_WF_FLOAT)9.0 * wf_2_00n - wf_3_00n) / (T_WF_FLOAT)60 / delta_z;

    /*
    // Gradient (Five-point stencil)
    complex<T_WF_FLOAT> dX = (-wf_2_p00 + (T_WF_FLOAT)8.0 * wf_p00 - (T_WF_FLOAT)8.0 * wf_n00 + wf_2_n00) / (T_WF_FLOAT)12 / delta_x;
    complex<T_WF_FLOAT> dY = (-wf_2_0p0 + (T_WF_FLOAT)8.0 * wf_0p0 - (T_WF_FLOAT)8.0 * wf_0n0 + wf_2_0n0) / (T_WF_FLOAT)12 / delta_y;
    complex<T_WF_FLOAT> dZ = (-wf_2_00p + (T_WF_FLOAT)8.0 * wf_00p - (T_WF_FLOAT)8.0 * wf_00n + wf_2_00n) / (T_WF_FLOAT)12 / delta_z;
    */
    complex<T_WF_FLOAT> dPsi = (T_WF_FLOAT)normal.x * dX + (T_WF_FLOAT)normal.y * dY + (T_WF_FLOAT)normal.z * dZ;

    complex<T_WF_FLOAT> iUnit = complex<T_WF_FLOAT>(0.0, 1.0);
    T_WF_FLOAT hBar = 1.0;
    pcDensity = ((T_WF_FLOAT)(-hBar / 2.0 / mass) * iUnit
        * (
            conj(psi) * dPsi - psi * conj(dPsi)
        )
    ).real();

    if (i < sample_count_x && j < sample_count_y) {
        probability_current_density[planeIdx] = pcDensity;
    }

    // Integrate:
    unsigned int threadId = get_block_local_idx_2d();
    uint2 pixel = {blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};

    if (i < sample_count_x && j < sample_count_y) {
        sdata[threadId] = pcDensity * get_simpson_coefficient_2d<T_WF_FLOAT>(pixel, sample_count) * dW * dH;
    }
    else {
        sdata[threadId] = (T_WF_FLOAT)0;
    }

    // Reduction in shared memory:
    parallel_reduction_sequential(threadId, sdata);

    // Add the values calculated by the blocks and write the result into probabilityCurrentOut
    if (threadId == 0)
        atomicAdd(probabilityCurrentOut, sdata[0]);
}