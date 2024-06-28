#include "PATH_TO_SOURCES/cuda_kernels/common.cu"

extern "C" __global__
void absorbing_potential_kernel(
    complex<float>* __restrict__ V,

    float delta_x,
    float delta_y,
    float delta_z,

    float bottom_x,
    float bottom_y,
    float bottom_z,

    float top_x,
    float top_y,
    float top_z,

    float offset,

    float potential_pos_x,
    float potential_pos_y,
    float potential_pos_z,

    float potential_neg_x,
    float potential_neg_y,
    float potential_neg_z,

    float exponent_pos_x,
    float exponent_pos_y,
    float exponent_pos_z,

    float exponent_neg_x,
    float exponent_neg_y,
    float exponent_neg_z
)
{
    float3 delta_r = {delta_x, delta_y, delta_z};
    float3 bottom = {bottom_x, bottom_y, bottom_z};
    float3 top = {top_x, top_y, top_z};
    float3 potential_pos = {potential_pos_x, potential_pos_y, potential_pos_z};
    float3 potential_neg = {potential_neg_x, potential_neg_y, potential_neg_z};

    float3 offset_pos = {offset, offset, offset};
    float3 offset_neg = {offset, offset, offset};

    float3 exponent_pos = {exponent_pos_x, exponent_pos_y, exponent_pos_z};
    float3 exponent_neg = {exponent_neg_x, exponent_neg_y, exponent_neg_z};

    float3 N = {(float)(gridDim.x * blockDim.x), (float)(gridDim.y * blockDim.y), (float)(gridDim.z * blockDim.z)};
    uint3 voxel = get_voxel_coords();
    float3 r = delta_r * float3{(float)voxel.x, (float)voxel.y, (float)voxel.z} - 0.5f * N * delta_r;
    float3 box_half = 0.5f * N * delta_r;  // Simulated box size / 2.0

    float potential = 0.0f;
    float3 dB = r - bottom;
    float3 dT = r - top;
    // Corners of the bottom plate:
    if (dB.x < 0.0f && dB.y < 0.0f && dB.z < 0.0f) {     // (-,-,- ; .,.,.)
        // Calculate blending parameters:
        float alfa = atanf(dB.y / dB.x);
        float u = alfa / M_PI_f * 2.0f;
        float beta = atanf(-dB.z / sqrtf(dB.x*dB.x + dB.y*dB.y));
        float v = beta / M_PI_f * 2.0f;
        // Blend parameters:
        float maxD = mix(box_half - (abs(bottom) + offset_neg), u, v);
        float exponent = mix(exponent_neg, u, v);
        float max_pot = mix(potential_neg, u, v);
        potential = powf(max(length(dB) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    else if (dB.x < 0.0f && dB.y < 0.0f && dT.z > 0.0f) {     // (-,-,. ; .,.,+)
        // Calculate blending parameters:
        float alfa = atanf(dB.y / dB.x);
        float u = alfa / M_PI_f * 2.0f;
        float beta = atanf(dT.z / sqrtf(dB.x*dB.x + dB.y*dB.y));
        float v = beta / M_PI_f * 2.0f;
        // Blend parameters:
        float maxD = mix(box_half - (abs(float3{bottom.x, bottom.y, top.z}) + float3{offset_neg.x, offset_neg.y, offset_pos.z}), u, v);
        float exponent = mix({exponent_neg.x, exponent_neg.y, exponent_pos.z}, u, v);
        float max_pot = mix({potential_neg.x, potential_neg.y, potential_pos.z}, u, v);
        potential = powf(max(length(float3{dB.x, dB.y, dT.z}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    else if (dB.y < 0.0f && dT.x > 0.0f && dT.z > 0.0f) {     // (.,-,. ; +,.,+)
        // Calculate blending parameters:
        float alfa = atanf(-dB.y / dT.x);
        float u = alfa / M_PI_f * 2.0f;
        float beta = atanf(dT.z / sqrtf(dT.x*dT.x + dB.y*dB.y));
        float v = beta / M_PI_f * 2.0f;
        // Blend parameters:
        float maxD = mix(box_half - (abs(float3{top.x, bottom.y, top.z}) + float3{offset_pos.x, offset_neg.y, offset_pos.z}), u, v);
        float exponent = mix({exponent_pos.x, exponent_neg.y, exponent_pos.z}, u, v);
        float max_pot = mix({potential_pos.x, potential_neg.y, potential_pos.z}, u, v);
        potential = powf(max(length(float3{dT.x, dB.y, dT.z}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    else if (dB.y < 0.0f && dB.z < 0.0f && dT.x > 0.0f) {     // (.,-,- ; +,.,.)
        // Calculate blending parameters:
        float alfa = atanf(-dB.y / dT.x);
        float u = alfa / M_PI_f * 2.0f;
        float beta = atanf(-dB.z / sqrtf(dT.x*dT.x + dB.y*dB.y));
        float v = beta / M_PI_f * 2.0f;
        // Blend parameters:
        float maxD = mix(box_half - (abs(float3{top.x, bottom.y, bottom.z}) + float3{offset_pos.x, offset_neg.y, offset_neg.z}), u, v);
        float exponent = mix({exponent_pos.x, exponent_neg.y, exponent_neg.z}, u, v);
        float max_pot = mix({potential_pos.x, potential_neg.y, potential_neg.z}, u, v);
        potential = powf(max(length(float3{dT.x, dB.y, dB.z}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    // Corners of the top plate:
    else if (dT.x > 0.0f && dT.y > 0.0f && dT.z > 0.0f) {   // (.,.,. ; +,+,+)
        // Calculate blending parameters:
        float alfa = atanf(dT.y / dT.x);
        float u = alfa / M_PI_f * 2.0f;
        float beta = atanf(dT.z / sqrtf(dT.x*dT.x + dT.y*dT.y));
        float v = beta / M_PI_f * 2.0f;
        // Blend parameters:
        float maxD = mix(box_half - (abs(top) + offset_pos), u, v);
        float exponent = mix(exponent_pos, u, v);
        float max_pot = mix(potential_pos, u, v);
        potential = powf(max(length(dT) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    else if (dB.z < 0.0f && dT.x > 0.0f && dT.y > 0.0f) {   // (.,.,- ; +,+,.)
        // Calculate blending parameters:
        float alfa = atanf(dT.y / dT.x);
        float u = alfa / M_PI_f * 2.0f;
        float beta = atanf(-dB.z / sqrtf(dT.x*dT.x + dT.y*dT.y));
        float v = beta / M_PI_f * 2.0f;
        // Blend parameters:
        float maxD = mix(box_half - (abs(float3{top.x, top.y, bottom.z}) + float3{offset_pos.x, offset_pos.y, offset_neg.z}), u, v);
        float exponent = mix({exponent_pos.x, exponent_pos.y, exponent_neg.z}, u, v);
        float max_pot = mix({potential_pos.x, potential_pos.y, potential_neg.z}, u, v);
        potential = powf(max(length(float3{dT.x, dT.y, dB.z}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    else if (dB.x < 0.0f && dB.z < 0.0f && dT.y > 0.0f) {   // (-,.,- ; .,+,.)
        // Calculate blending parameters:
        float alfa = atanf(dT.y / -dB.x);
        float u = alfa / M_PI_f * 2.0f;
        float beta = atanf(-dB.z / sqrtf(dB.x*dB.x + dT.y*dT.y));
        float v = beta / M_PI_f * 2.0f;
        // Blend parameters:
        float maxD = mix(box_half - (abs(float3{bottom.x, top.y, bottom.z}) + float3{offset_neg.x, offset_pos.y, offset_neg.z}), u, v);
        float exponent = mix({exponent_neg.x, exponent_pos.y, exponent_neg.z}, u, v);
        float max_pot = mix({potential_neg.x, potential_pos.y, potential_neg.z}, u, v);
        potential = powf(max(length(float3{dB.x, dT.y, dB.z}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    else if (dB.x < 0.0f && dT.y > 0.0f && dT.z > 0.0f) {   // (-,.,. ; .,+,+)
        // Calculate blending parameters:
        float alfa = atanf(dT.y / -dB.x);
        float u = alfa / M_PI_f * 2.0f;
        float beta = atanf(dT.z / sqrtf(dB.x*dB.x + dT.y*dT.y));
        float v = beta / M_PI_f * 2.0f;
        // Blend parameters:
        float maxD = mix(box_half - (abs(float3{bottom.x, top.y, top.z}) + float3{offset_neg.x, offset_pos.y, offset_pos.z}), u, v);
        float exponent = mix({exponent_neg.x, exponent_pos.y, exponent_pos.z}, u, v);
        float max_pot = mix({potential_neg.x, potential_pos.y, potential_pos.z}, u, v);
        potential = powf(max(length(float3{dB.x, dT.y, dT.z}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    // Bottom edges:
    else if (dB.x < 0.0f && dB.y < 0.0f && dB.z > 0.0f && dT.z < 0.0f) {    // (-,-,+ ; .,.,-)
        // Calculate blending parameter:
        float alfa = atanf(dB.y / dB.x);
        float u = alfa / M_PI_f * 2.0f;
        // Blend parameters:
        float maxD = mix(box_half - (abs(float3{bottom.x, bottom.y, 0.0f}) + float3{offset_neg.x, offset_neg.y, 0.0f}), u, 0.0f);
        float exponent = mix({exponent_neg.x, exponent_neg.y, 0.0f}, u, 0.0f);
        float max_pot = mix({potential_neg.x, potential_neg.y, 0.0f}, u, 0.0f);
        potential = powf(max(length(float3{dB.x, dB.y, 0.0f}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    else if (dB.x > 0.0f && dB.y < 0.0f && dT.x < 0.0f && dT.z > 0.0f) {    // (+,-,. ; -,.,+)
        // Calculate blending parameter:
        float beta = atanf(dT.z / -dB.y);
        float v = beta / M_PI_f * 2.0f;
        // Blend parameters
        float maxD = mix(box_half - (abs(float3{0.0f, bottom.y, top.z}) + float3{0.0f, offset_neg.y, offset_pos.z}), 0.0f, v);
        float exponent = mix({0.0f, exponent_neg.y, exponent_pos.z}, 1.0f, v);
        float max_pot = mix({0.0f, potential_neg.y, potential_pos.z}, 1.0f, v);
        potential = powf(max(length(float3{0.0f, dB.y, dT.z}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    else if (dB.y < 0.0f && dB.z > 0.0f && dT.x > 0.0f && dT.z < 0.0f) {    // (.,-,+ ; +,.,-)
        // Calculate blending parameter:
        float alfa = atanf(-dB.y / dT.x);
        float u = alfa / M_PI_f * 2.0f;
        // Blend parameters
        float maxD = mix(box_half - (abs(float3{top.x, bottom.y, 0.0f}) + float3{offset_pos.x, offset_neg.y, 0.0f}), u, 0.0f);
        float exponent = mix({exponent_pos.x, exponent_neg.y, 0.0f}, u, 0.0f);
        float max_pot = mix({potential_pos.x, potential_neg.y, 0.0f}, u, 0.0f);
        potential = powf(max(length(float3{dT.x, dB.y, 0.0f}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    else if (dB.x > 0.0f && dB.y < 0.0f && dB.z < 0.0f && dT.x < 0.0f) {    // (+,-,- ; -,.,.)
        // Calculate blending parameter:
        float beta = atanf(dB.z / dB.y);
        float v = beta / M_PI_f * 2.0f;
        // Blend parameters
        float maxD = mix(box_half - (abs(float3{0.0f, bottom.y, bottom.z}) + float3{0.0f, offset_neg.y, offset_neg.z}), 0.0f, v);
        float exponent = mix({0.0f, exponent_neg.y, exponent_neg.z}, 1.0f, v);
        float max_pot = mix({0.0f, potential_neg.y, potential_neg.z}, 1.0f, v);
        potential = powf(max(length(float3{0.0f, dB.y, dB.z}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    // "Pillar" edges:
    else if (dB.x < 0.0f && dB.y > 0.0f && dB.z < 0.0f && dT.y < 0.0f) {    // (-,+,- ; .,-,.)
        // Calculate blending parameter:
        float beta = atanf(dB.z / dB.x);
        float v = beta / M_PI_f * 2.0f;
        // Blend parameters
        float maxD = mix(box_half - (abs(float3{bottom.x, 0.0f, bottom.z}) + float3{offset_neg.x, 0.0f, offset_neg.z}), 0.0f, v);
        float exponent = mix({exponent_neg.x, 0.0f, exponent_neg.z}, 0.0f, v);
        float max_pot = mix({potential_neg.x, 0.0f, potential_neg.z}, 0.0f, v);
        potential = powf(max(length(float3{dB.x, 0.0f, dB.z}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    else if (dB.x < 0.0f && dB.y > 0.0f && dT.y < 0.0f && dT.z > 0.0f) {    // (-,+,. ; .,-,+)
        // Calculate blending parameter:
        float beta = atanf(dT.z / -dB.x);
        float v = beta / M_PI_f * 2.0f;
        // Blend parameters
        float maxD = mix(box_half - (abs(float3{bottom.x, 0.0f, top.z}) + float3{offset_neg.x, 0.0f, offset_pos.z}), 0.0f, v);
        float exponent = mix({exponent_neg.x, 0.0f, exponent_pos.z}, 0.0f, v);
        float max_pot = mix({potential_neg.x, 0.0f, potential_pos.z}, 0.0f, v);
        potential = powf(max(length(float3{dB.x, 0.0f, dT.z}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    else if (dB.y > 0.0f && dT.x > 0.0f && dT.y < 0.0f && dT.z > 0.0f) {    // (.,+,. ; +,-,+)
        // Calculate blending parameter:
        float beta = atanf(dT.z / dT.x);
        float v = beta / M_PI_f * 2.0f;
        // Blend parameters
        float maxD = mix(box_half - (abs(float3{top.x, 0.0f, top.z}) + float3{offset_pos.x, 0.0f, offset_pos.z}), 0.0f, v);
        float exponent = mix({exponent_pos.x, 0.0f, exponent_pos.z}, 0.0f, v);
        float max_pot = mix({potential_pos.x, 0.0f, potential_pos.z}, 0.0f, v);
        potential = powf(max(length(float3{dT.x, 0.0f, dT.z}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    else if (dB.y > 0.0f && dB.z < 0.0f && dT.x > 0.0f && dT.y < 0.0f) {    // (.,+,- ; +,-,.)
        // Calculate blending parameter:
        float beta = atanf(-dB.z / dT.x);
        float v = beta / M_PI_f * 2.0f;
        // Blend parameters
        float maxD = mix(box_half - (abs(float3{top.x, 0.0f, bottom.z}) + float3{offset_pos.x, 0.0f, offset_neg.z}), 0.0f, v);
        float exponent = mix({exponent_pos.x, 0.0f, exponent_neg.z}, 0.0f, v);
        float max_pot = mix({potential_pos.x, 0.0f, potential_neg.z}, 0.0f, v);
        potential = powf(max(length(float3{dT.x, 0.0f, dB.z}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    // Top plane edges:
    else if (dB.x < 0.0f && dB.z > 0.0f && dT.y > 0.0f && dT.z < 0.0f) {    // (-,.,+ ; .,+,-)
        // Calculate blending parameter:
        float alfa = atanf(dT.y / -dB.x);
        float u = alfa / M_PI_f * 2.0f;
        // Blend parameters
        float maxD = mix(box_half - (abs(float3{bottom.x, top.y, 0.0f}) + float3{offset_neg.x, offset_pos.y, 0.0f}), u, 0.0f);
        float exponent = mix({exponent_neg.x, exponent_pos.y, 0.0f}, u, 0.0f);
        float max_pot = mix({potential_neg.x, potential_pos.y, 0.0f}, u, 0.0f);
        potential = powf(max(length(float3{dB.x, dT.y, 0.0f}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    else if (dB.x > 0.0f && dT.x < 0.0f && dT.y > 0.0f && dT.z > 0.0f) {    // (+,.,. ; -,+,+)
        // Calculate blending parameter:
        float beta = atanf(dT.z / dT.y);
        float v = beta / M_PI_f * 2.0f;
        // Blend parameters
        float maxD = mix(box_half - (abs(float3{0.0f, top.y, top.z}) + float3{0.0f, offset_pos.y, offset_pos.z}), 1.0f, v);
        float exponent = mix({0.0f, exponent_pos.y, exponent_pos.z}, 1.0f, v);
        float max_pot = mix({0.0f, potential_pos.y, potential_pos.z}, 1.0f, v);
        potential = powf(max(length(float3{0.0f, dT.y, dT.z}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    else if (dB.z > 0.0f && dT.x > 0.0f && dT.y > 0.0f && dT.z < 0.0f) {    // (.,.,+ ; +,+,-)
        // Calculate blending parameter:
        float alfa = atanf(dT.y / dT.x);
        float u = alfa / M_PI_f * 2.0f;
        // Blend parameters
        float maxD = mix(box_half - (abs(float3{top.x, top.y, 0.0f}) + float3{offset_pos.x, offset_pos.y, 0.0f}), u, 0.0f);
        float exponent = mix({exponent_pos.x, exponent_pos.y, 0.0f}, u, 0.0f);
        float max_pot = mix({potential_pos.x, potential_pos.y, 0.0f}, u, 0.0f);
        potential = powf(max(length(float3{dT.x, dT.y, 0.0f}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    else if (dB.x > 0.0f && dB.z < 0.0f && dT.x < 0.0f && dT.y > 0.0f) {    // (+,.,- ; -,+,.)
        // Calculate blending parameter:
        float beta = atanf(-dB.z / dT.y);
        float v = beta / M_PI_f * 2.0f;
        // Blend parameters
        float maxD = mix(box_half - (abs(float3{0.0f, top.y, bottom.z}) + float3{0.0f, offset_pos.y, offset_neg.z}), 0.0f, v);
        float exponent = mix({0.0f, exponent_pos.y, exponent_neg.z}, 0.0f, v);
        float max_pot = mix({0.0f, potential_pos.y, potential_neg.z}, 0.0f, v);
        potential = powf(max(length(float3{0.0f, dT.y, dB.z}) - offset, 0.0f) / maxD, exponent) * max_pot;
    }
    // Box sides:
    else if (dB.x > 0.0f && dB.y < 0.0f && dB.z > 0.0f && dT.x < 0.0f && dT.z < 0.0f) {    // (+,-,+ ; -,.,-)
        float maxD = box_half.y - (fabsf(bottom.y) + offset_neg.y);
        potential = powf(max(-dB.y - offset_neg.y, 0.0f) / maxD, exponent_neg.y) * potential_neg.y;
    }
    else if (dB.x > 0.0f && dB.y > 0.0f && dT.x < 0.0f && dT.y < 0.0f && dT.z > 0.0f) {    // (+,+,. ; -,-,+)
        float maxD = box_half.z - (fabsf(top.z) + offset_pos.z);
        potential = powf(max(dT.z - offset_pos.z, 0.0f) / maxD, exponent_pos.z) * potential_pos.z;
    }
    else if (dB.y > 0.0f && dB.z > 0.0f && dT.x > 0.0f && dT.y < 0.0f && dT.z < 0.0f) {    // (.,+,+ ; +,-,-)
        float maxD = box_half.x - (fabsf(top.x) + offset_pos.x);
        potential = powf(max(dT.x - offset_pos.x, 0.0f) / maxD, exponent_pos.x) * potential_pos.x;
    }
    else if (dB.x > 0.0f && dB.y > 0.0f && dB.z < 0.0f && dT.x < 0.0f && dT.y < 0.0f) {    // (+,+,- ; -,-,.)
        float maxD = box_half.z - (fabsf(bottom.z) + offset_neg.z);
        potential = powf(max(-dB.z - offset_neg.z, 0.0f) / maxD, exponent_neg.z) * potential_neg.z;
    }
    else if (dB.x < 0.0f && dB.y > 0.0f && dB.z > 0.0f && dT.y < 0.0f && dT.z < 0.0f) {    // (-,+,+ ; .,-,-)
        float maxD = box_half.x - (fabsf(bottom.x) + offset_neg.x);
        potential = powf(max(-dB.x - offset_neg.x, 0.0f) / maxD, exponent_neg.x) * potential_neg.x;
    }
    else if (dB.x > 0.0f && dB.z > 0.0f && dT.x < 0.0f && dT.y > 0.0f && dT.z < 0.0f) {    // (+,.,+ ; -,+,-)
        float maxD = box_half.y - (fabsf(top.y) + offset_pos.y);
        potential = powf(max(dT.y - offset_pos.y, 0.0f) / maxD, exponent_pos.y) * potential_pos.y;
    }

    int idx = get_array_index();
    V[idx] += complex<float>(0.0f, potential);
}
