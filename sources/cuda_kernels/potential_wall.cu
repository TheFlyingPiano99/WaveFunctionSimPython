#include "PATH_TO_SOURCES/cuda_kernels/common.cu"


extern "C" __global__
void potential_wall_kernel(
    complex<float>* __restrict__ V,
    float potential_hartree,

    float delta_x,
    float delta_y,
    float delta_z,

    float center_x,
    float center_y,
    float center_z,

    float normal_x,
    float normal_y,
    float normal_z,

    float plateau_thickness_bohr_radii,
    float slope_thickness_bohr_radii,
    float slope_exponent,
    unsigned int slit_count,
    float slit_spacing_bohr_radii,
    float slit_width_bohr_radii,
    float slit_rotation_radian
)
{
    uint3 voxel = get_voxel_coords();
    int idx = get_array_index();

    float3 delta_r = {delta_x, delta_y, delta_z};
    float3 center = {center_x, center_y, center_z};
    float3 normal = {normal_x, normal_y, normal_z};

    float3 r = mul(delta_r, transform_corner_origin_to_center_origin_system({(float)voxel.x, (float)voxel.y, (float)voxel.z}));
    float d = fabsf(dot(normal, diff(r, center)));  // distance
    float3 up = {0.0f, 1.0f, 0.0f};
    if (fabsf(dot(normal, up)) > 0.99f){    // normal and up vectors are almost parallel
        up.x = 0.0f;
        up.z = 1.0f;
    }
    float3 right = normalize(cross(normal, up));
    right = rotate_vector(right, normal, slit_rotation_radian);
    float dR = fabsf(dot(right, diff(r, center)));    // tangential distance
    switch (slit_count) {
        case 1: {
            if (dR < slit_width_bohr_radii * 0.5f)
                return;
            break;
        }
        case 2: {
            if (dR > (slit_spacing_bohr_radii - slit_width_bohr_radii) * 0.5f && dR < (slit_spacing_bohr_radii + slit_width_bohr_radii) * 0.5f)
                return;
            break;
        }
        case 3: {
            if (
                (dR < slit_width_bohr_radii * 0.5f) ||
                (dR > slit_spacing_bohr_radii - slit_width_bohr_radii * 0.5f && dR < slit_spacing_bohr_radii + slit_width_bohr_radii * 0.5f)
            )
                return;
            break;
        }
        case 4: {
            if (
                (dR > (slit_spacing_bohr_radii - slit_width_bohr_radii) * 0.5f && dR < (slit_spacing_bohr_radii + slit_width_bohr_radii) * 0.5f)
                ||
                (dR > slit_spacing_bohr_radii * 1.5f - slit_width_bohr_radii * 0.5f && dR < slit_spacing_bohr_radii * 1.5f + slit_width_bohr_radii * 0.5f)
            )
                return;
            break;
        }
    }
    if (d <= plateau_thickness_bohr_radii * 0.5f)  // Inside the plateau
    {
        V[idx] += potential_hartree;    // Apply max potential value
    }
    else if (d <= plateau_thickness_bohr_radii * 0.5f + slope_thickness_bohr_radii) {   // Inside the slope
        V[idx] += potential_hartree * powf(1.0f - (d - plateau_thickness_bohr_radii * 0.5f) / slope_thickness_bohr_radii, slope_exponent);
    }
}
