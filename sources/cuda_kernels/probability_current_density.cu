#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


extern "C" __global__
void probability_current_density_kernel(
    complex<float>* wave_function,
    float* __restrict__ probability_current_density,

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

    int idx = get_array_index(voxel, N);

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

    complex<float> psi = wave_function[idx];
    complex<float> wf_n00 = wave_function[idx_n00];
    complex<float> wf_p00 = wave_function[idx_p00];
    complex<float> wf_0n0 = wave_function[idx_0n0];
    complex<float> wf_0p0 = wave_function[idx_0p0];
    complex<float> wf_00n = wave_function[idx_00n];
    complex<float> wf_00p = wave_function[idx_00p];
    // Derivate:
    complex<float> dX = (wf_p00 - wf_n00) / 2.0f / delta_x;
    complex<float> dY = (wf_0p0 - wf_0n0) / 2.0f / delta_y;
    complex<float> dZ = (wf_00p - wf_00n) / 2.0f / delta_z;
    complex<float> gradPsi = normal.x * dX + normal.y * dY + normal.z * dZ;

    complex<float> iUnit = complex<float>(0.0f, 1.0f);
    float hBar = 1.0f;
    float mass = 1.0f;
    probability_current_density[planeIdx] = (-iUnit * hBar / 2.0f / mass * (
        mul(conj(psi), gradPsi) - mul(psi, conj(gradPsi))
    )).real();
}