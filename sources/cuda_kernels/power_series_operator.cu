#include "PATH_TO_SOURCES/cuda_kernels/common.cu"

extern "C" __global__
void next_s(
    complex<float>* s_prev,
    complex<float>* s_next,
    const complex<float>* v,
    complex<float>* wave_function,
    float delta_t,

    float delta_x,
    float delta_y,
    float delta_z,

    float mass,
    int n
)
{
    uint3 voxel = get_voxel_coords();
    uint3 N = get_voxel_count();
    int idx = get_array_index();

    int idx_n00 = idx;
    if (voxel.x > 0) {
        idx_n00 = get_array_index({voxel.x - 1, voxel.y, voxel.z});
    }
    int idx_p00 = idx;
    if (voxel.x < N.x - 1) {
        idx_p00 = get_array_index({voxel.x + 1, voxel.y, voxel.z});
    }
    int idx_0n0 = idx;
    if (voxel.y > 0) {
        idx_0n0 = get_array_index({voxel.x, voxel.y - 1, voxel.z});
    }
    int idx_0p0 = idx;
    if (voxel.y < N.y - 1) {
        idx_0p0 = get_array_index({voxel.x, voxel.y + 1, voxel.z});
    }
    int idx_00n = idx;
    if (voxel.z > 0) {
        idx_00n = get_array_index({voxel.x, voxel.y, voxel.z - 1});
    }
    int idx_00p = idx;
    if (voxel.z < N.z - 1) {
        idx_00p = get_array_index({voxel.x, voxel.y, voxel.z + 1});
    }

    float3 delta_r = {delta_x, delta_y, delta_z};

    complex<float> laplace_s = (
          s_prev[idx_n00]
        + s_prev[idx_p00]
        + s_prev[idx_0n0]
        -6.0f * s_prev[idx]
        + s_prev[idx_0p0]
        + s_prev[idx_00n]
        + s_prev[idx_00p]
    ) / delta_x / delta_x;

    complex<float> s = complex<float>(0.0f, 1.0f) * complex<float>(delta_t / (float)n, 0.0f)
        * (complex<float>(1.0f / 2.0f / mass, 0.0f) * laplace_s - complex<float>(v[idx]) * s_prev[idx]);

    s_next[idx] = s;
    wave_function[idx] += s;
}
