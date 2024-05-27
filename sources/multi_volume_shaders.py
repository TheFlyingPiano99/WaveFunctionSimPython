# This file implements a fragment shader that can be used to visualize multiple
# volumes simultaneously. It is derived from the original fragment shader in
# vispy.visuals.volume, which is releaed under a BSD license included here:
#
# ===========================================================================
# Vispy is licensed under the terms of the (new) BSD license:
#
# Copyright (c) 2015, authors of Vispy
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of Vispy Development Team nor the names of its
#   contributors may be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================
#
# This modified version is released under the BSD license given in the LICENSE
# file in this repository.

import textwrap

from vispy.visuals.volume import _VERTEX_SHADER

# Fragment shader
FRAG_SHADER = """
#version 130
// uniforms
{declarations}
uniform vec3 u_shape;
uniform float u_threshold;
uniform float u_relative_step_size;
uniform int u_n_tex;
uniform vec3 u_light_direction; 

in vec3 v_position;
in vec4 v_nearpos;
in vec4 v_farpos;

// uniforms for lighting. Hard coded until we figure out how to do lights
const vec4 u_ambient = vec4(0.2, 0.4, 0.2, 1.0);
const vec4 u_diffuse = vec4(0.8, 0.2, 0.2, 1.0);
const vec4 u_specular = vec4(1.0, 1.0, 1.0, 1.0);
const float u_shininess = 40.0;

//in vec3 lightDirs[1];

// global holding view direction in local coordinates
vec3 view_ray;

float rand(vec2 co)
{{
    // Create a pseudo-random number between 0 and 1.
    // http://stackoverflow.com/questions/4200224
    return fract(sin(dot(co.xy ,vec2(12.9898, 78.233))) * 43758.5453);
}}


vec2 trilinearInterpolation(vec3 currentPos, sampler3D voxels, vec3 delta) {{
	vec3 currentVoxel = vec3(ivec3(currentPos));
	vec3 inVoxelPos = currentPos - currentVoxel;
	vec2 sample000 = texture(voxels, (currentVoxel + vec3(0,0,0)) * delta).rg;
	vec2 sample100 = texture(voxels, (currentVoxel + vec3(1,0,0)) * delta).rg;
	vec2 sample010 = texture(voxels, (currentVoxel + vec3(0,1,0)) * delta).rg;
	vec2 sample001 = texture(voxels, (currentVoxel + vec3(0,0,1)) * delta).rg;
	vec2 sample110 = texture(voxels, (currentVoxel + vec3(1,1,0)) * delta).rg;
	vec2 sample101 = texture(voxels, (currentVoxel + vec3(1,0,1)) * delta).rg;
	vec2 sample011 = texture(voxels, (currentVoxel + vec3(0,1,1)) * delta).rg;
	vec2 sample111 = texture(voxels, (currentVoxel + vec3(1,1,1)) * delta).rg;

	vec2 filtered = (
				(sample000 * (1.0 - inVoxelPos.z) + sample001 * inVoxelPos.z) * (1.0 - inVoxelPos.y)
				+ (sample010 * (1.0 - inVoxelPos.z) + sample011 * inVoxelPos.z) * inVoxelPos.y
			) * (1.0 - inVoxelPos.x)
		 + (
				(sample100 * (1.0 - inVoxelPos.z) + sample101 * inVoxelPos.z) * (1.0 - inVoxelPos.y)
				+ (sample110 * (1.0 - inVoxelPos.z) + sample111 * inVoxelPos.z) * inVoxelPos.y
		   ) * inVoxelPos.x;

	return filtered;
}}

vec4 complexCentralDifferenceGradSample(sampler3D samplerUnit, vec3 position, vec3 size, int coord) {{
    
    vec3 delta = 1.0 / size;
    vec3 positionN = position - delta;
    vec3 positionP = position + delta;
    
    float sample = $sample(samplerUnit, position)[coord] * 2.0 - 1.0;
     
    vec2 NX = $sample(samplerUnit,vec3(positionN.x, position.y, position.z)).ra;
    vec2 NY = $sample(samplerUnit,vec3(position.x, positionN.y, position.z)).ra;
    vec2 NZ = $sample(samplerUnit,vec3(position.x, position.y, positionN.z)).ra;
    
    NX = NX * 2.0 - 1.0; 
    NY = NY * 2.0 - 1.0; 
    NZ = NZ * 2.0 - 1.0; 

    vec2 PX = $sample(samplerUnit,vec3(positionP.x, position.y, position.z)).ra;
    vec2 PY = $sample(samplerUnit,vec3(position.x, positionP.y, position.z)).ra;
    vec2 PZ = $sample(samplerUnit,vec3(position.x, position.y, positionP.z)).ra;
    
    PX = PX * 2.0 - 1.0; 
    PY = PY * 2.0 - 1.0; 
    PZ = PZ * 2.0 - 1.0; 

    vec3 sN = vec3(
        dot(NX, NX),
        dot(NY, NY),
        dot(NZ, NX)
    );
    vec3 sP = vec3(
        dot(PX, PX),
        dot(PY, PY),
        dot(PZ, PX)
    );
    vec3 grad = (sP - sN) / 2.0 / delta;
    return vec4(
        grad,
        sample);
}}

vec4 resampleGradientAndDensity(sampler3D samplerUnit, vec3 position, vec3 size, int coord) {{
    vec3 scaled_position = position * size - 0.5;
    vec3 beta = scaled_position - round(scaled_position);
    vec3 g0 = 0.5 - beta;
    vec3 delta0 = (0.5 + beta)*0.5;
    vec3 position0 = position - delta0 / size;
    vec3 position1 = position0 + 0.5 / size;
    vec4 s0 = vec4(
        $sample(samplerUnit,vec3(position0.x, position0.y, position0.z))[coord],
        $sample(samplerUnit,vec3(position0.x, position1.y, position0.z))[coord],
        $sample(samplerUnit,vec3(position0.x, position0.y, position1.z))[coord],
        $sample(samplerUnit,vec3(position0.x, position1.y, position1.z))[coord]
    );
    vec4 s1 = vec4(
        $sample(samplerUnit,vec3(position1.x, position0.y, position0.z))[coord],
        $sample(samplerUnit,vec3(position1.x, position1.y, position0.z))[coord],
        $sample(samplerUnit,vec3(position1.x, position0.y, position1.z))[coord],
        $sample(samplerUnit,vec3(position1.x, position1.y, position1.z))[coord]
    );
    vec4 s_xy0z0_xy1z0_xy0z1_xy1z1 = mix(s1, s0, g0.x);
    vec4 s_dxy0z0_dxy1z0_dxy0z1_dxy1z1 = s1 - s0;
    vec4 s_xyz0_xyz1_dxyz0_dxyz1 = mix(
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.yw,
            s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.yw),
        vec4(s_xy0z0_xy1z0_xy0z1_xy1z1.xz,
            s_dxy0z0_dxy1z0_dxy0z1_dxy1z1.xz), g0.y);
    vec2 s_xdyz0_xdyz1 =
        s_xy0z0_xy1z0_xy0z1_xy1z1.yw -
        s_xy0z0_xy1z0_xy0z1_xy1z1.xz;
    vec3 s_xyz_dxyz_xdyz = mix(
        vec3(s_xyz0_xyz1_dxyz0_dxyz1.yw, s_xdyz0_xdyz1.y),
        vec3(s_xyz0_xyz1_dxyz0_dxyz1.xz, s_xdyz0_xdyz1.x), g0.z);
    float s_xydz =
    s_xyz0_xyz1_dxyz0_dxyz1.y -
    s_xyz0_xyz1_dxyz0_dxyz1.x;

    return vec4(
        vec3(s_xyz_dxyz_xdyz.y, s_xyz_dxyz_xdyz.z, s_xydz),
        s_xyz_dxyz_xdyz.x);
}}


// Based on http://www.oscars.org/science-technology/sci-tech-projects/aces
vec3 acesTonemap(vec3 color){{
    mat3 m1 = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );
    mat3 m2 = mat3(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
    );
    vec3 v = m1 * color;    
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return pow(clamp(m2 * (a / b), 0.0, 1.0), vec3(1.0 / 2.2));
}}


// for some reason, this has to be the last function in order for the
// filters to be inserted in the correct place...

void main() {{
    vec3 farpos = v_farpos.xyz / v_farpos.w;
    vec3 nearpos = v_nearpos.xyz / v_nearpos.w;

    // Calculate unit vector pointing in the view direction through this
    // fragment.
    view_ray = normalize(farpos.xyz - nearpos.xyz);

    // Compute the distance to the front surface or near clipping plane
    float distance = dot(nearpos-v_position, view_ray);
    distance = max(distance, min((-0.5 - v_position.x) / view_ray.x,
                            (u_shape.x - 0.5 - v_position.x) / view_ray.x));
    distance = max(distance, min((-0.5 - v_position.y) / view_ray.y,
                            (u_shape.y - 0.5 - v_position.y) / view_ray.y));
    distance = max(distance, min((-0.5 - v_position.z) / view_ray.z,
                            (u_shape.z - 0.5 - v_position.z) / view_ray.z));

    // Now we have the starting position on the front surface
    vec3 front = v_position + view_ray * distance;
    
    vec3 background_color = vec3(1, 1, 1);

    // Decide how many steps to take
    int nsteps = int(-distance / u_relative_step_size + 0.5);
    if (nsteps < 1 )
        discard;

    // Get starting location and step vector in texture coordinates
    vec3 step = ((v_position - front) / u_shape) / nsteps;
    float stepSize = length(step);
    vec3 start_loc = front / u_shape;

    // For testing: show the number of steps. This helps to establish
    // whether the rays are correctly oriented
    //return;

    vec3 accumulated_color = vec3(0., 0., 0.);
    vec3 accumulated_attenuation = vec3(0., 0., 0.);

    // This outer loop seems necessary on some systems for large
    // datasets. Ugly, but it works ...
    vec3 loc = start_loc;
    int iter = 0;
    float colorDensity = 6000.0;
    float attenDensity = 500.0;
    while (iter < nsteps) {{
        for (iter=iter; iter<nsteps; iter++)
        {{
        
            // Shadow:
            vec3 shadow = vec3(0, 0, 0);
            float shadowStep = stepSize * 3.0;
            vec3 shadowLoc = loc + shadowStep * normalize(u_light_direction);
            float shadowDiscont = 1.0;
            float shadowAttenDensity = attenDensity * 50.0;
            for (int shadowIter = 0; shadowIter < 50; shadowIter++) {{
                 
                vec3 s = vec3(0,0,0);
                
{shadow_calculation}

                shadow = shadow + s * (1.0 - shadow);
                shadowLoc = shadowLoc + shadowStep * normalize(u_light_direction);
                shadowDiscont *= 0.96;
                shadowStep *= 1.0001;
            }}

            vec4 color = vec4(0, 0, 0, 0);

{color_calculation}
            
            vec4 fogRadianceAplha = vec4(0.0 * (1.0 - shadow) * vec3(1,1,1), 0.0);  // Now disabled!

            color += fogRadianceAplha;
            
            // Scaled current color and attenuation:
            vec3 c = vec3(
                        max(colorDensity * stepSize * color.r, 0.0),
                        max(colorDensity * stepSize * color.g, 0.0),
                        max(colorDensity * stepSize * color.b, 0.0)
                    );
            float tintedMix = 0.75;
            vec3 a = vec3(
                        max(min(attenDensity * stepSize * color.a * ((1 - tintedMix) + tintedMix * max(1.0 - color.r, 0.0)), 1.0), 0.0),
                        max(min(attenDensity * stepSize * color.a * ((1 - tintedMix) + tintedMix * max(1.0 - color.g, 0.0)), 1.0), 0.0),
                        max(min(attenDensity * stepSize * color.a * ((1 - tintedMix) + tintedMix * max(1.0 - color.b, 0.0)), 1.0), 0.0)
                    );

            // Accumulate color and attenuation:
            accumulated_color = accumulated_color + c * (1.0 - accumulated_attenuation);
            accumulated_attenuation = accumulated_attenuation + a * (1.0 - accumulated_attenuation);  
            
            // Early ray termination:
            if (accumulated_attenuation.x > 0.99 && accumulated_attenuation.y > 0.99 && accumulated_attenuation.z > 0.99) {{
                accumulated_attenuation = vec3(1, 1, 1);
                iter = nsteps;
            }}
            
            // Advance location deeper into the volume
            loc += step;
        }}
    }}

    gl_FragColor = vec4(acesTonemap(accumulated_color + background_color * (1.0 - accumulated_attenuation)), 1.0);

    /* Set depth value - from visvis TODO
    int iter_depth = int(maxi);
    // Calculate end position in world coordinates
    vec4 position2 = vertexPosition;
    position2.xyz += ray*shape*float(iter_depth);
    // Project to device coordinates and set fragment depth
    vec4 iproj = gl_ModelViewProjectionMatrix * position2;
    iproj.z /= iproj.w;
    gl_FragDepth = (iproj.z+1.0)/2.0;
    */
}}
"""


def get_shaders(n_volume_max):
    """
    Get the fragment shader code, supporting a maximum of ``n_volume_max``
    simultaneous textures and colormaps.
    """

    declarations = ""
    shadow_calculation = ""
    color_calculation = ""

    # Here I changed the way alpha value is calculated from different volume sources - Zoltán Simon
    # Now the source with the highest alpha is preserved in color and not the average of all the sources
    for i in range(n_volume_max):
        declarations += "uniform $sampler_type u_volumetex{0:d};\n".format(i)
        shadow_calculation += (
            """
                if (u_n_tex > {0:d}) {{
                    ivec3 size = textureSize(u_volumetex{0:d}, 0);
                    vec4 reGradDensity = complexCentralDifferenceGradSample(u_volumetex{0:d}, shadowLoc, size, 0);    // The grad is the same for both
                    vec4 imGradDensity = complexCentralDifferenceGradSample(u_volumetex{0:d}, shadowLoc, size, 3);    // The grad is the same for both
                    vec4 shadowAlbedoAlpha = $cmap{0:d}(reGradDensity.w, imGradDensity.w);
                    vec2 complexVal = vec2(reGradDensity.w, imGradDensity.w);

                    s = vec3(
                        max(min(s.r + shadowAlbedoAlpha.a * (0.5 + 0.5 * (1.0 - shadowAlbedoAlpha.r)) * shadowStep * shadowAttenDensity * shadowDiscont, 1.0), 0.0),
                        max(min(s.g + shadowAlbedoAlpha.a * (0.5 + 0.5 * (1.0 - shadowAlbedoAlpha.g)) * shadowStep * shadowAttenDensity * shadowDiscont, 1.0), 0.0),
                        max(min(s.b + shadowAlbedoAlpha.a * (0.5 + 0.5 * (1.0 - shadowAlbedoAlpha.b)) * shadowStep * shadowAttenDensity * shadowDiscont, 1.0), 0.0)
                    );
                }}
            """
        ).format(i)
        color_calculation += (
            """if (u_n_tex > {0:d}) {{
                    ivec3 size = textureSize(u_volumetex{0:d}, 0);
                    vec4 reGradDensity = complexCentralDifferenceGradSample(u_volumetex{0:d}, loc, size, 0);    // The grad is the same for both
                    vec4 imGradDensity = complexCentralDifferenceGradSample(u_volumetex{0:d}, loc, size, 3);    // The grad is the same for both
                    vec4 albedoAlpha = $cmap{0:d}(reGradDensity.w, imGradDensity.w);
                    vec3 radiance = vec3(0,0,0);
                    float gradLength = length(reGradDensity.xyz);
                    
                    if (gradLength > 0.000001) {{
                        vec3 normal = -reGradDensity.xyz;
                        normal = normal / gradLength;
                        vec3 halfway = normalize(-view_ray + normalize(u_light_direction));
                        vec3 lightPower = vec3(1.0, 0.976, 0.847);
                        float shininess = 60.0;
                        float diffuse = 30.0;
                        float specular = 20.0;
                        float ambient = 0.03;
                        float gradMix = 1.0;    // The amount of gradient scaling in diffuse and specular
                        float gradT = min(max(gradMix * gradLength + (1.0 - gradMix) * 1.0, 0.0), 1.0); 
                        radiance = albedoAlpha.a * (
                                    (
                                        lightPower * diffuse * (1.0 - shadow) * ((1.0 - gradT) + gradT * max(0.0, dot(normal, normalize(u_light_direction))))
                                        + ambient
                                    ) * albedoAlpha.rgb
                                    + lightPower * specular * (1.0 - shadow) * gradT * pow(max(0.0, dot(normal, halfway)), shininess)
                                );
                    }}
                    else{{
                        
                    }}
                    if (albedoAlpha.a > color.a)
                        color = vec4(radiance, albedoAlpha.a);
                    }}
            """
        ).format(i, abs(i - 1))

    # color_calculation += "color *= 1. / u_n_tex;".format(1. / n_volume_max)

    color_calculation = textwrap.indent(color_calculation, " " * 12)
    shadow_calculation = textwrap.indent(shadow_calculation, " " * 12)

    return _VERTEX_SHADER, FRAG_SHADER.format(
        declarations=declarations, color_calculation=color_calculation, shadow_calculation=shadow_calculation
    )

BACKGROUND_FRAG_SHADER = """
#version 130

// Based on http://www.oscars.org/science-technology/sci-tech-projects/aces
vec3 acesTonemap(vec3 color){{
    mat3 m1 = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );
    mat3 m2 = mat3(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
    );
    vec3 v = m1 * color;    
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return pow(clamp(m2 * (a / b), 0.0, 1.0), vec3(1.0 / 2.2));
}}


void main() {{
    vec3 background_color = vec3(1, 1, 1);
    gl_FragColor = vec4(acesTonemap(background_color), 1.0);
}}    
"""

def get_background_shader():
    return _VERTEX_SHADER, BACKGROUND_FRAG_SHADER

if __name__ == "__main__":
    print(get_shaders(6)[1])
