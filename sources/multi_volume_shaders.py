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


float trilinearInterpolation(vec3 currentPos, sampler3D voxels, vec3 delta) {{
	vec3 currentVoxel = vec3(ivec3(currentPos));
	vec3 inVoxelPos = currentPos - currentVoxel;
	float sample000 = texture(voxels, (currentVoxel + vec3(0,0,0)) * delta).r;
	float sample100 = texture(voxels, (currentVoxel + vec3(1,0,0)) * delta).r;
	float sample010 = texture(voxels, (currentVoxel + vec3(0,1,0)) * delta).r;
	float sample001 = texture(voxels, (currentVoxel + vec3(0,0,1)) * delta).r;
	float sample110 = texture(voxels, (currentVoxel + vec3(1,1,0)) * delta).r;
	float sample101 = texture(voxels, (currentVoxel + vec3(1,0,1)) * delta).r;
	float sample011 = texture(voxels, (currentVoxel + vec3(0,1,1)) * delta).r;
	float sample111 = texture(voxels, (currentVoxel + vec3(1,1,1)) * delta).r;

	float filtered = (
				(sample000 * (1.0 - inVoxelPos.z) + sample001 * inVoxelPos.z) * (1.0 - inVoxelPos.y)
				+ (sample010 * (1.0 - inVoxelPos.z) + sample011 * inVoxelPos.z) * inVoxelPos.y
			) * (1.0 - inVoxelPos.x)
		 + (
				(sample100 * (1.0 - inVoxelPos.z) + sample101 * inVoxelPos.z) * (1.0 - inVoxelPos.y)
				+ (sample110 * (1.0 - inVoxelPos.z) + sample111 * inVoxelPos.z) * inVoxelPos.y
		   ) * inVoxelPos.x;

	return filtered;
}}


vec4 resampleGradientAndDensity(sampler3D samplerUnit, vec3 position, vec3 size) {{
    vec3 scaled_position = position * size - 0.5;
    vec3 beta = scaled_position - round(scaled_position);
    vec3 g0 = 0.5 - beta;
    vec3 delta0 = (0.5 + beta)*0.5;
    vec3 position0 = position - delta0 / size;
    vec3 position1 = position0 + 0.5 / size;
    vec4 s0 = vec4(
        $sample(samplerUnit,vec3(position0.x, position0.y, position0.z)).r,
        $sample(samplerUnit,vec3(position0.x, position1.y, position0.z)).r,
        $sample(samplerUnit,vec3(position0.x, position0.y, position1.z)).r,
        $sample(samplerUnit,vec3(position0.x, position1.y, position1.z)).r
    );
    vec4 s1 = vec4(
        $sample(samplerUnit,vec3(position1.x, position0.y, position0.z)).r,
        $sample(samplerUnit,vec3(position1.x, position1.y, position0.z)).r,
        $sample(samplerUnit,vec3(position1.x, position0.y, position1.z)).r,
        $sample(samplerUnit,vec3(position1.x, position1.y, position1.z)).r
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
        normalize(vec3(s_xyz_dxyz_xdyz.y, s_xyz_dxyz_xdyz.z, s_xydz)),
        s_xyz_dxyz_xdyz.x);
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

    // Decide how many steps to take
    int nsteps = int(-distance / u_relative_step_size + 0.5);
    if( nsteps < 1 )
        discard;

    // Get starting location and step vector in texture coordinates
    vec3 step = ((v_position - front) / u_shape) / nsteps;
    vec3 start_loc = front / u_shape;

    // For testing: show the number of steps. This helps to establish
    // whether the rays are correctly oriented
    //return;

    vec4 integrated_color = vec4(0., 0., 0., 0.);

    // This outer loop seems necessary on some systems for large
    // datasets. Ugly, but it works ...
    vec3 loc = start_loc;
    int iter = 0;
    while (iter < nsteps) {{
        for (iter=iter; iter<nsteps; iter++)
        {{
            // Get sample color
            vec4 color = vec4(0, 0, 0, 0);

{color_calculation}
            
            // Translucent method:
            float a1 = integrated_color.a;
            float a2 = color.a * (1 - a1);
            float alpha = max(a1 + a2, 0.000001);
            integrated_color *= a1 / alpha;
            integrated_color += color * a2 / alpha;
            integrated_color.a = alpha;
            if (alpha > 0.99 ) {{
                iter = nsteps;
            }}
            
            // Old method:
            //integrated_color = 1.0 - (1.0 - integrated_color) * (1.0 - color);

            // Advance location deeper into the volume
            loc += step;
        }}
    }}

    gl_FragColor = integrated_color;

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
    color_calculation = ""

    # Here I changed the way alpha value is calculated from different volume sources - Zoltán Simon
    # Now the source with the highest alpha is preserved in color and not the average of all the sources
    for i in range(n_volume_max):
        declarations += "uniform $sampler_type u_volumetex{0:d};\n".format(i)
        color_calculation += (
            "if (u_n_tex > {0:d}) {{\n\
                            ivec3 size = textureSize(u_volumetex{0:d}, 0);\n\
                            vec4 gradDensity = resampleGradientAndDensity(u_volumetex{0:d}, loc, size);\n\
                            vec4 current_color = $cmap{0:d}(gradDensity.w);\n\
                            vec3 normal = -gradDensity.xyz;\n\
                            if (length(normal) > 0.0) {{\n\
                                float l = length(normal);\n\
                                normal = normalize(normal) * min(1.0, pow(l, 1.0));\n\
                            }}\n\
                            else{{\n\
                                normal = vec3(0,0,0); // Disable reflection for too homogenous density\n\
                            }}\n\
                            // Light reflection:\n\
                            //vec3 light_dir = vec3(-1, 1, 1);\n\
                            vec3 halfway = normalize(-view_ray + u_light_direction);\n\
                            float shininess = 30.0;\n\
                            float diffuse = 1.0;\n\
                            float specular = 1.0;\n\
                            float ambient = 0.2;\n\
                            vec3 radiance = (diffuse * max(0.0, dot(normal, u_light_direction)) + ambient) * current_color.rgb\n\
                            + specular * max(0.0, pow(dot(normal, halfway), shininess));\n\
                            if (current_color.a > color.a)\n\
                                color = vec4(radiance, current_color.a);\n\
                            }}\n"
        ).format(i, abs(i - 1))

    # color_calculation += "color *= 1. / u_n_tex;".format(1. / n_volume_max)

    color_calculation = textwrap.indent(color_calculation, " " * 12)

    return _VERTEX_SHADER, FRAG_SHADER.format(
        declarations=declarations, color_calculation=color_calculation
    )


if __name__ == "__main__":
    print(get_shaders(6)[1])
