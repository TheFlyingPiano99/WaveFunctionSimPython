#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


extern "C" __global__
void probability_current_density_kernel(
    complex<float>* wave_function,
    float* probability_current_density,

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
    float height
)
{
    float3 delta_r = {delta_x, delta_y, delta_z};
    float3 center = {center_x, center_y, center_z};
    float3 normal = {normal_x, normal_y, normal_z};

    unsigned int x = gridDim.x * blockDim.x + threadIdx.x;
    unsigned int y = gridDim.y * blockDim.y + threadIdx.y;

    unsigned int planeIdx = get_array_index();

    probability_current_density[planeIdx] = 0.0001f;
}