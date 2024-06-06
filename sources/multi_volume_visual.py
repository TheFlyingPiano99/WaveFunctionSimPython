# This file implements a MultiVolumeVisual class that can be used to show
# multiple volumes simultaneously. It is derived from the original VolumeVisual
# class in vispy.visuals.volume, which is releaed under a BSD license included
# here:
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

# Modified by Zoltan Simon

from vispy.gloo import Texture3D, TextureEmulated3D, VertexBuffer, IndexBuffer
from vispy.visuals import Visual
from vispy.visuals.shaders import Function
from vispy.color import get_colormap, BaseColormap
from vispy.scene.visuals import create_visual_node
import numpy as np

from .multi_volume_shaders import get_shaders
from .callback_list import CallbackList


class Default2ParamColorMap(BaseColormap):
    glsl_map = """
    vec4 defaultColorMap(float x, float y) {
        return vec4(0, 0, 0, 0);    // Empty render
    }
    """


class MultiVolumeVisual(Visual):
    """
    Displays multiple 3D volumes simultaneously.

    Parameters
    ----------
    volumes : list of tuples
        The volumes to show. Each tuple should contain three elements: the data
        array, the clim values, and the colormap to use. The clim values should
        be either a 2-element tuple, or None.
    relative_step_size : float
        The relative step size to step through the volume. Default 0.8.
        Increase to e.g. 1.5 to increase performance, at the cost of
        quality.
    emulate_texture : bool
        Use 2D textures to emulate a 3D texture. OpenGL ES 2.0 compatible,
        but has lower performance on desktop platforms.
    n_volume_max : int
        Absolute maximum number of volumes that can be shown.
    """

    def __init__(
        self,
        volumes,
        clim=(-1.0, 1.0),
        threshold=None,
        relative_step_size=0.8,
        method="mip",
        emulate_texture=False,
        n_volume_max=5,
    ):
        # Choose texture class
        tex_cls = TextureEmulated3D if emulate_texture else Texture3D

        # We store the data and colormaps in a CallbackList which can warn us
        # when it is modified.
        self.volumes = CallbackList()
        self.volumes.on_size_change = self._update_all_volumes
        self.volumes.on_item_change = self._update_volume

        self._vol_shape = None
        self._need_vertex_update = True

        # Create OpenGL program
        vert_shader, frag_shader = get_shaders(n_volume_max)
        super(MultiVolumeVisual, self).__init__(vcode=vert_shader, fcode=frag_shader)
        self.method = method
        # Create gloo objects
        self._vertices = VertexBuffer()
        self._texcoord = VertexBuffer(
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1],
                ],
                dtype=np.float32,
            )
        )

        # Set up textures
        self.textures = []
        for i in range(n_volume_max):
            self.textures.append(
                tex_cls(
                    data=np.zeros(shape=(16, 16, 16, 2)),
                    interpolation="linear",
                    wrapping="clamp_to_edge",
                )
            )
            self.shared_program["u_volumetex{0}".format(i)] = self.textures[i]
            self.shared_program.frag["cmap{0:d}".format(i)] = Function(
                Default2ParamColorMap().glsl_map
            )

        self.shared_program["a_position"] = self._vertices
        self.shared_program["a_texcoord"] = self._texcoord
        self._draw_mode = "triangle_strip"
        self._index_buffer = IndexBuffer()

        self.shared_program.frag["sampler_type"] = self.textures[0].glsl_sampler_type
        self.shared_program.frag["sample"] = self.textures[0].glsl_sample

        # Only show back faces of cuboid. This is required because if we are
        # inside the volume, then the front faces are outside of the clipping
        # box and will not be drawn.
        self.set_gl_state("translucent", cull_face=False, depth_test=False)

        self.relative_step_size = relative_step_size
        self.light_direction = np.array([1.0, 1.0, -1.0], dtype=np.float_)
        self.freeze()

        # Add supplied volumes
        self.volumes.extend(volumes)

    def _update_all_volumes(self, volumes):
        """
        Update the number of simultaneous textures.

        Parameters
        ----------
        n_textures : int
            The number of textures to use
        """
        if len(self.volumes) > len(self.textures):
            raise ValueError(
                "Number of volumes ({0}) exceeds number of textures ({1})".format(
                    len(self.volumes), len(self.textures)
                )
            )
        for index, volume in enumerate(self.volumes):
            self._update_volume(volume, index)

    def _update_volume(self, volume, index):
        data, clim, cmap = volume
        cmap = get_colormap(cmap)

        if clim is None:
            clim = data.min(), data.max()
            data = data.astype(np.float32)
        if clim[1] == clim[0]:
            if clim[0] != 0.0:
                data *= 1.0 / clim[0]
        else:
            data -= clim[0]
            data /= clim[1] - clim[0]

        self.shared_program["u_volumetex{0:d}".format(index)].set_data(data)
        self.shared_program.frag["cmap{0:d}".format(index)] = Function(cmap.glsl_map)

        if self._vol_shape is None:
            self.shared_program["u_shape"] = (data.shape[2], data.shape[1], data.shape[0])  # TODO WARNING: Untested index order!!!
            self._vol_shape = data.shape
        elif data.shape != self._vol_shape:
            raise ValueError(
                "Shape of arrays should be {0} instead of {1}".format(
                    self._vol_shape, data.shape
                )
            )

        self.shared_program["u_n_tex"] = len(self.volumes)

    # Helper method that only requires the volume_data to update, and it's index in the multi_volume_visual
    def update_volume_data(self, volume_data, index):
        self._update_volume(
            (volume_data, self.volumes[index][1], self.volumes[index][2]), index
        )

    @property
    def relative_step_size(self):
        """The relative step size used during raycasting.

        Larger values yield higher performance at reduced quality. If
        set > 2.0 the ray skips entire voxels. Recommended values are
        between 0.5 and 1.5. The amount of quality degredation depends
        on the render method.
        """
        return self._relative_step_size

    @property
    def light_direction(self):
        return self._light_direction

    @relative_step_size.setter
    def relative_step_size(self, value):
        value = float(value)
        if value < 0.1:
            raise ValueError("relative_step_size cannot be smaller than 0.1")
        self._relative_step_size = value
        self.shared_program["u_relative_step_size"] = value

    @light_direction.setter
    def light_direction(self, light_dir):
        self._light_direction = light_dir
        self.shared_program["u_light_direction"] = light_dir

    def _create_vertex_data(self):
        """Create and set positions and texture coords from the given shape

        We have six faces with 1 quad (2 triangles) each, resulting in
        6*2*3 = 36 vertices in total.
        """
        shape = self._vol_shape

        # Get corner coordinates. The -0.5 offset is to center
        # pixels/voxels. This works correctly for anisotropic data.
        x0, x1 = -0.5, shape[2] - 0.5
        y0, y1 = -0.5, shape[1] - 0.5
        z0, z1 = -0.5, shape[0] - 0.5

        pos = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x0, y1, z0],
                [x1, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x0, y1, z1],
                [x1, y1, z1],
            ],
            dtype=np.float32,
        )

        """
          6-------7
         /|      /|
        4-------5 |
        | |     | |
        | 2-----|-3
        |/      |/
        0-------1
        """

        # Order is chosen such that normals face outward; front faces will be
        # culled.
        indices = np.array([2, 6, 0, 4, 5, 6, 7, 2, 3, 0, 1, 5, 3, 7], dtype=np.uint32)

        # Fullscreen quad for background rendering:
        quad_vertices = np.array(
            [
                [-1.0, -1.0, 0.0],
                [-1.0,  1.0, 0.0],
                [ 1.0, -1.0, 0.0],
                [ 1.0,  1.0, 0.0]
            ],
            dtype=np.float32
        )

        quad_uv = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0,  1.0, 0.0],
                [ 1.0, 0.0, 0.0],
                [ 1.0,  1.0, 0.0]
            ],
            dtype=np.float32
        )

        quad_indices = np.array([0, 2, 1, 2, 3, 1], dtype=np.uint32)

        # Apply
        self._vertices.set_data(pos)
        self._index_buffer.set_data(indices)

        #self._vertices.set_data(pos)
        #self._index_buffer.set_data(indices)

    def _compute_bounds(self, axis, view):
        return 0, self._vol_shape[axis]

    def _prepare_transforms(self, view):
        trs = view.transforms
        view.view_program.vert["transform"] = trs.get_transform()

        view_tr_f = trs.get_transform("visual", "document")
        view_tr_i = view_tr_f.inverse
        view.view_program.vert["viewtransformf"] = view_tr_f
        view.view_program.vert["viewtransformi"] = view_tr_i

    def _prepare_draw(self, view):
        if self._need_vertex_update:
            self._create_vertex_data()
            self._need_vertex_update = False


MultiVolume = create_visual_node(MultiVolumeVisual)
