#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


extern "C" __global__
void probability_current_density_kernel(
    complex<T_WF_FLOAT>* wave_function,
    float* __restrict__ probability_current_density,

    float mass,

    float delta_x,
    float delta_y,
    float delta_z,

    float center_x,
    float center_y,
    float center_z,

    float normal_x,
    float normal_y,
    float normal_z,

    float width,
    float height,

    unsigned int volume_dim_x,
    unsigned int volume_dim_y,
    unsigned int volume_dim_z,

    float bounding_bottom_x,
    float bounding_bottom_y,
    float bounding_bottom_z,

    float bounding_top_x,
    float bounding_top_y,
    float bounding_top_z
)
{
    float3 delta_r = {delta_x, delta_y, delta_z};
    float3 center = {center_x, center_y, center_z};
    float3 normal = {normal_x, normal_y, normal_z};
    uint3 N = {volume_dim_x, volume_dim_y, volume_dim_z};
    float3 prefUp = {0.0f, 1.0f, 0.0f};
    if (fabsf(dot(normal, prefUp)) > 0.99)  // Change prefered up vector
    {
        prefUp.y = 0.0f;
        prefUp.z = 1.0f;
    }

    float3 right = cross(normal, prefUp);
    float3 up = cross(right, normal);

    float u = (blockIdx.x * blockDim.x + threadIdx.x) / (float)(gridDim.x * blockDim.x - 1);
    float v = (blockIdx.y * blockDim.y + threadIdx.y) / (float)(gridDim.y * blockDim.y - 1);

    float3 r = center + right * (width * (u - 0.5f)) + up * (height * (v - 0.5f));
    int planeIdx = get_array_index();
    if (r.x < bounding_bottom_x || r.x > bounding_top_x
    || r.y < bounding_bottom_y || r.y > bounding_top_y
    || r.z < bounding_bottom_z || r.z > bounding_top_z) // Terminate if outside the bounding box
    {
        probability_current_density[planeIdx] = 0.0f;
        return;
    }

    float3 fVoxel = r / delta_r + 0.5f * float3{(float)N.x, (float)N.y, (float)N.z};
    uint3 voxel = {(unsigned int)fVoxel.x, (unsigned int)fVoxel.y, (unsigned int)fVoxel.z};

    // Calculate array indices:

    int idx = get_array_index(voxel, N);

    // 1th step:
    int idx_n00 = idx;
    if (voxel.x > 0) {
        idx_n00 = get_array_index({voxel.x - 1, voxel.y, voxel.z}, N);
    }
    int idx_p00 = idx;
    if (voxel.x < N.x - 1) {
        idx_p00 = get_array_index({voxel.x + 1, voxel.y, voxel.z}, N);
    }
    int idx_0n0 = idx;
    if (voxel.y > 0) {
        idx_0n0 = get_array_index({voxel.x, voxel.y - 1, voxel.z}, N);
    }
    int idx_0p0 = idx;
    if (voxel.y < N.y - 1) {
        idx_0p0 = get_array_index({voxel.x, voxel.y + 1, voxel.z}, N);
    }
    int idx_00n = idx;
    if (voxel.z > 0) {
        idx_00n = get_array_index({voxel.x, voxel.y, voxel.z - 1}, N);
    }
    int idx_00p = idx;
    if (voxel.z < N.z - 1) {
        idx_00p = get_array_index({voxel.x, voxel.y, voxel.z + 1}, N);
    }

    // 2nd step:
    int idx_2_n00 = idx;
    if (voxel.x > 1) {
        idx_2_n00 = get_array_index({voxel.x - 2, voxel.y, voxel.z}, N);
    }
    int idx_2_p00 = idx;
    if (voxel.x < N.x - 2) {
        idx_2_p00 = get_array_index({voxel.x + 2, voxel.y, voxel.z}, N);
    }
    int idx_2_0n0 = idx;
    if (voxel.y > 1) {
        idx_2_0n0 = get_array_index({voxel.x, voxel.y - 2, voxel.z}, N);
    }
    int idx_2_0p0 = idx;
    if (voxel.y < N.y - 2) {
        idx_2_0p0 = get_array_index({voxel.x, voxel.y + 2, voxel.z}, N);
    }
    int idx_2_00n = idx;
    if (voxel.z > 1) {
        idx_2_00n = get_array_index({voxel.x, voxel.y, voxel.z - 2}, N);
    }
    int idx_2_00p = idx;
    if (voxel.z < N.z - 2) {
        idx_2_00p = get_array_index({voxel.x, voxel.y, voxel.z + 2}, N);
    }

    // 3nd step:
    int idx_3_n00 = idx;
    if (voxel.x > 2) {
        idx_3_n00 = get_array_index({voxel.x - 3, voxel.y, voxel.z}, N);
    }
    int idx_3_p00 = idx;
    if (voxel.x < N.x - 3) {
        idx_3_p00 = get_array_index({voxel.x + 3, voxel.y, voxel.z}, N);
    }
    int idx_3_0n0 = idx;
    if (voxel.y > 2) {
        idx_3_0n0 = get_array_index({voxel.x, voxel.y - 3, voxel.z}, N);
    }
    int idx_3_0p0 = idx;
    if (voxel.y < N.y - 3) {
        idx_3_0p0 = get_array_index({voxel.x, voxel.y + 3, voxel.z}, N);
    }
    int idx_3_00n = idx;
    if (voxel.z > 2) {
        idx_3_00n = get_array_index({voxel.x, voxel.y, voxel.z - 3}, N);
    }
    int idx_3_00p = idx;
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

    // Gradient:
    complex<T_WF_FLOAT> dX = -(-wf_3_p00 + (T_WF_FLOAT)9.0 * wf_2_p00 - (T_WF_FLOAT)45.0 * wf_p00 + (T_WF_FLOAT)45.0 * wf_n00 - (T_WF_FLOAT)9.0 * wf_2_n00 + wf_3_n00) / (T_WF_FLOAT)60.0 / (T_WF_FLOAT)delta_x;
    complex<T_WF_FLOAT> dY = -(-wf_3_0p0 + (T_WF_FLOAT)9.0 * wf_2_0p0 - (T_WF_FLOAT)45.0 * wf_0p0 + (T_WF_FLOAT)45.0 * wf_0n0 - (T_WF_FLOAT)9.0 * wf_2_0n0 + wf_3_0n0) / (T_WF_FLOAT)60.0 / (T_WF_FLOAT)delta_y;
    complex<T_WF_FLOAT> dZ = -(-wf_3_00p + (T_WF_FLOAT)9.0 * wf_2_00p - (T_WF_FLOAT)45.0 * wf_00p + (T_WF_FLOAT)45.0 * wf_00n - (T_WF_FLOAT)9.0 * wf_2_00n + wf_3_00n) / (T_WF_FLOAT)60.0 / (T_WF_FLOAT)delta_z;
    complex<T_WF_FLOAT> dPsi = (T_WF_FLOAT)normal.x * dX + (T_WF_FLOAT)normal.y * dY + (T_WF_FLOAT)normal.z * dZ;

    complex<T_WF_FLOAT> iUnit = complex<T_WF_FLOAT>(0.0f, 1.0f);
    float hBar = 1.0f;
    probability_current_density[planeIdx] = ((T_WF_FLOAT)(-hBar / 2.0f / mass) * iUnit
        * (
            conj(psi) * dPsi - psi * conj(dPsi)
        )
    ).real();
}