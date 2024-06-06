#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


extern "C" __global__
void kinetic_operator_kernel(
    complex<float>* __restrict__ kinetic_operator,

    float delta_x,
    float delta_y,
    float delta_z,

    float delta_t
)
{
    uint3 voxel = get_voxel_coords();
    int idx = get_array_index();

    float3 f = div(
        {(float)voxel.x, (float)voxel.y, (float)voxel.z},
        {(float)(gridDim.x * blockDim.x - 1), (float)(gridDim.y * blockDim.y - 1), (float)(gridDim.z * blockDim.z - 1)}
    );
    float3 delta_r = {delta_x, delta_y, delta_z};

    // Account for numpy fftn's "negative frequency in second half" pattern
    if (f.x > 0.5f)
        f.x = 1.0f - f.x;
    if (f.y > 0.5f)
        f.y = 1.0f - f.y;
    if (f.z > 0.5f)
        f.z = 1.0f - f.z;

    float3 momentum = scalarVectorMul(2.0f * M_PI, div(f, delta_r));
    float angle = -dot(momentum, momentum) * delta_t / 4.0f;
    kinetic_operator[idx] = exp_i(angle);
}
