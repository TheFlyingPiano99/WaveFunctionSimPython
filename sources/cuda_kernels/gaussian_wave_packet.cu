#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


extern "C" __global__
void wave_packet_kernel(
    complex<T_WF_FLOAT>* __restrict__ wave_tensor,

    float delta_x,
    float delta_y,
    float delta_z,

    float a,

    float r_x,
    float r_y,
    float r_z,

    float k_x,
    float k_y,
    float k_z
)
{
    uint3 voxel = get_voxel_coords();
    int idx = get_array_index();

    float3 k_0 = {k_x, k_y, k_z};
    float3 r_0 = {r_x, r_y, r_z};
    float3 delta_r = {delta_x, delta_y, delta_z};
    float3 N = {
        (float)(gridDim.x * blockDim.x),
        (float)(gridDim.y * blockDim.y),
        (float)(gridDim.z * blockDim.z)
    };
    float3 r = diff(
        mul(delta_r, {(float)voxel.x, (float)voxel.y, (float)voxel.z}),
        scalarVectorMul(0.5f, mul(N, delta_r))
    );

    complex<T_WF_FLOAT> val =
        (T_WF_FLOAT)powf(2.0f / M_PI / a / a, 3.0f / 4.0f)
        * exp_i(dot(k_0, r))
        * (T_WF_FLOAT)expf(
            -dot(diff(r, r_0), diff(r, r_0)) / a / a
        );
    wave_tensor[idx] = val;
}
