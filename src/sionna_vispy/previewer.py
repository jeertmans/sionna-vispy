#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
3D scene and paths viewer using VisPy.

This file is a modified version of :mod:`sionna.rt.previewer` to use
VisPy as a viewer rather than pythreejs.
"""

from __future__ import annotations

import drjit as dr
import matplotlib
import mitsuba as mi
import numpy as np
import pythreejs as p3s
import vispy.util.transforms as tsf
from sionna.rt.renderer import coverage_map_color_mapping
from sionna.rt.utils import (
    mitsuba_rectangle_to_world,
    paths_to_segments,
    rotate,
    scene_scale,
)
from vispy.geometry.generation import create_cylinder
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Image, LinePlot, Markers, Mesh
from vispy.visuals.transforms import MatrixTransform, STTransform


class InteractiveDisplay(SceneCanvas):
    """
    Lightweight wrapper around the `vispy` library.

    Input
    -----
    resolution : [2], int
        Size of the viewer figure.

    fov : float
        Field of view, in degrees.

    background : str
        Background color in hex format prefixed by '#'.
    """

    def __init__(self, scene, resolution, fov, background) -> None:
        super().__init__(keys="interactive", size=resolution, bgcolor=background)
        print("Patched version")

        self.unfreeze()

        self._resolution = resolution
        self._sionna_scene = scene  # self._scene is already defined by VisPy

        # List of objects in the scene
        self._objects = []
        # Bounding box of the scene
        self._bbox = mi.ScalarBoundingBox3f()

        ####################################################
        # Setup the viewer
        ####################################################

        # View
        self._view = self.central_widget.add_view()

        self._view.camera = "turntable"
        # self._view.camera.transform.translate((+100, 100, 100))
        # self._view.camera.depth_value = 1e3

        self._camera = self._view.camera

        # Camera
        # self._camera = PerspectiveCamera(
        #     fov=fov,
        #     scale_factor=1000,
        #     center=[0, 0, 0],
        # )
        # self._view.camera = self._camera

        self.freeze()

        ####################################################
        # Plot the scene geometry
        ####################################################
        self.plot_scene()

        # Finally, ensure the camera is looking at the scene
        self.center_view()

    def reset(self):
        """
        Removes objects that are not flagged as persistent, i.e., the paths.
        """
        remaining = []
        for obj, persist in self._objects:
            if persist:
                remaining.append((obj, persist))
            else:
                obj.parent = None
        self._objects = remaining

    def redraw_scene_geometry(self):
        """
        Redraw the scene geometry.
        """
        remaining = []
        for obj, persist in self._objects:
            if not persist:  # Only scene objects are flagged as persistent
                remaining.append((obj, persist))
            else:
                obj.parent = None
        self._objects = remaining

        # Plot the scene geometry
        self.plot_scene()

    def center_view(self):
        """
        Automatically place the camera based on the scene's bounding box such
        that it is located at (-1, -1, 1) on the normalized bounding box, and
        oriented toward the center of the scene.
        """
        bbox = self._bbox if self._bbox.valid() else mi.ScalarBoundingBox3f(0.0)
        center = bbox.center()

        corner = [bbox.min.x, center.y, 1.5 * bbox.max.z]
        if np.allclose(corner, 0):
            corner = (-1, -1, 1)
        # self._camera.center = tuple(corner)

        self._view.camera.set_range()
        # self._camera.lookAt(center)

    def plot_radio_devices(self, show_orientations=False):
        """
        Plots the radio devices.

        Input
        -----
        show_orientations : bool
            Shows the radio devices' orientations.
            Defaults to `False`.
        """
        scene = self._sionna_scene
        sc, tx_positions, rx_positions, _, _ = scene_scale(scene)
        transmitter_colors = [
            transmitter.color.numpy() for transmitter in scene.transmitters.values()
        ]
        receiver_colors = [
            receiver.color.numpy() for receiver in scene.receivers.values()
        ]

        # Radio emitters, shown as points
        p = np.array(list(tx_positions.values()) + list(rx_positions.values()))
        albedo = np.array(transmitter_colors + receiver_colors)

        if p.shape[0] > 0:
            # Radio devices are not persistent
            radius = max(0.005 * sc, 1)
            self._plot_points(p, persist=False, colors=albedo, radius=radius)
        if show_orientations:
            line_length = 0.0075 * sc
            head_length = 0.15 * line_length
            zeros = np.zeros((1, 3))

            for devices in [
                scene.transmitters.values(),
                scene.receivers.values(),
                scene.ris.values(),
            ]:
                if len(devices) == 0:
                    continue
                starts, ends = [], []
                for rd in devices:
                    # Arrow line
                    color = rd.color
                    starts.append(rd.position)
                    endpoint = rd.position + rotate(
                        [line_length, 0.0, 0.0], rd.orientation
                    )
                    ends.append(endpoint)

                    meshdata = create_cylinder(
                        rows=80,
                        cols=80,
                        radius=(30 * head_length, 0),
                        length=100*head_length,
                    )
                    angles = rd.orientation.numpy()
                    mesh = Mesh(color=color, meshdata=meshdata)
                    mesh.transform = MatrixTransform()
                    mesh.transform.rotate(np.rad2deg(+angles[0] - np.pi / 2), (0, 0, 1))
                    mesh.transform.rotate(np.rad2deg(+angles[2]), (0, 1, 0))
                    mesh.transform.rotate(np.rad2deg(-angles[1]), (1, 0, 0))
                    mesh.transform.translate(np.append(endpoint, 1))


                    # geo = p3s.CylinderGeometry(
                    #     radiusTop=0,
                    #     radiusBottom=0.3 * head_length,
                    #     height=head_length,
                    #     radialSegments=8,
                    #     heightSegments=0,
                    #     openEnded=False,
                    # )
                    # mat = p3s.MeshLambertMaterial(color=color)
                    # mesh = p3s.Mesh(geo, mat)
                    # mesh.position = tuple(endpoint)
                    # angles = rd.orientation.numpy()
                    # mesh.rotateZ(angles[0] - np.pi / 2)
                    # mesh.rotateY(angles[2])
                    # mesh.rotateX(-angles[1])
                    self._add_child(mesh, zeros, zeros, persist=False)

                #self._plot_lines(np.array(starts), np.array(ends), width=2, color=color)

    def plot_paths(self, paths):
        """
        Plot the ``paths``.

        Input
        -----
        paths : :class:`~sionna.rt.Paths`
            Paths to plot
        """
        starts, ends = paths_to_segments(paths)
        if starts and ends:
            self._plot_lines(np.vstack(starts), np.vstack(ends))

    def plot_scene(self):
        """
        Plots the meshes that make the scene.
        """
        shapes = self._sionna_scene.mi_scene.shapes()
        n = len(shapes)
        if n <= 0:
            return

        palette = None
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wi = mi.Vector3f(0, 0, 1)

        # Shapes (e.g. buildings)
        vertices, faces, albedos = [], [], []
        f_offset = 0
        for i, s in enumerate(shapes):
            null_transmission = s.bsdf().eval_null_transmission(si).numpy()
            if np.min(null_transmission) > 0.99:
                # The BSDF for this shape was probably set to `null`, do not
                # include it in the scene preview.
                continue

            n_vertices = s.vertex_count()
            v = s.vertex_position(dr.arange(mi.UInt32, n_vertices))
            vertices.append(v.numpy())
            f = s.face_indices(dr.arange(mi.UInt32, s.face_count()))
            faces.append(f.numpy() + f_offset)
            f_offset += n_vertices

            albedo = s.bsdf().eval_diffuse_reflectance(si).numpy()
            if not np.any(albedo > 0.0):
                if palette is None:
                    palette = matplotlib.colormaps.get_cmap("Pastel1_r")
                albedo[:] = palette((i % palette.N + 0.5) / palette.N)[:3]

            albedos.append(np.tile(albedo, (n_vertices, 1)))

        # Plot all objects as a single PyThreeJS mesh, which is must faster
        # than creating individual mesh objects in large scenes.
        self._plot_mesh(
            np.concatenate(vertices, axis=0),
            np.concatenate(faces, axis=0),
            persist=True,  # The scene geometry is persistent
            colors=np.concatenate(albedos, axis=0),
        )

    def plot_coverage_map(
        self, coverage_map, tx=0, db_scale=True, vmin=None, vmax=None
    ):
        """
        Plot the coverage map as a textured rectangle in the scene. Regions
        where the coverage map is zero-valued are made transparent.
        """
        to_world = coverage_map.to_world()
        # coverage_map = resample_to_corners(
        #     coverage_map[tx, :, :].numpy()
        # )
        coverage_map = coverage_map[tx, :, :].numpy()

        # Create a rectangle from two triangles
        p00 = to_world.transform_affine([-1, -1, 0])
        p01 = to_world.transform_affine([1, -1, 0])
        p10 = to_world.transform_affine([-1, 1, 0])
        p11 = to_world.transform_affine([1, 1, 0])

        vertices = np.array([p00, p01, p10, p11])
        pmin = np.min(vertices, axis=0)
        pmax = np.max(vertices, axis=0)

        to_map, normalizer, color_map = coverage_map_color_mapping(
            coverage_map, db_scale=db_scale, vmin=vmin, vmax=vmax
        )
        texture = color_map(normalizer(to_map)).astype(np.float32)
        texture[:, :, 3] = (coverage_map > 0.0).astype(np.float32)
        # Pre-multiply alpha
        texture[:, :, :3] *= texture[:, :, 3, None]

        n, m = texture.shape[:2]

        xscale = abs(pmax[0] - pmin[0]) / m
        yscale = abs(pmax[1] - pmin[1]) / n
        xshift = pmin[0]
        yshift = pmin[1]
        zshift = (pmin[2] + pmax[2]) / 2

        transform = STTransform(
            scale=(xscale, yscale),
            translate=(xshift, yshift, zshift),
        )

        image = Image(data=(255.0 * texture).astype(np.uint8))
        image.transform = transform

        self._add_child(image, pmin, pmax, persist=False)

    def plot_ris(self):
        """
        Plot all RIS as a monochromatic rectangle in the scene
        """
        all_ris = list(self._sionna_scene.ris.values())

        for ris in all_ris:
            orientation = ris.orientation
            to_world = mitsuba_rectangle_to_world(
                ris.position, orientation, ris.size, ris=True
            )
            color = ris.color.numpy()

            # Create a rectangle from two triangles
            p00 = to_world.transform_affine([-1, -1, 0])
            p01 = to_world.transform_affine([1, -1, 0])
            p10 = to_world.transform_affine([-1, 1, 0])
            p11 = to_world.transform_affine([1, 1, 0])

            vertices = np.array([p00, p01, p10, p11])
            pmin = np.min(vertices, axis=0)
            pmax = np.max(vertices, axis=0)

            faces = np.array(
                [
                    [0, 1, 2],
                    [2, 1, 3],
                ],
                dtype=np.uint32,
            )

            mesh = Mesh(vertices=vertices, faces=faces, color=color)

            self._add_child(mesh, pmin, pmax, persist=False)

    def set_clipping_plane(self, offset, orientation):
        """
        Input
        -----
        clip_at : float
            If not `None`, the scene preview will be clipped (cut) by a plane
            with normal orientation ``clip_plane_orientation`` and offset
            ``clip_at``. This allows visualizing the interior of meshes such
            as buildings.

        clip_plane_orientation : tuple[float, float, float]
            Normal vector of the clipping plane.
        """
        return  # TODO
        if offset is None:
            self._renderer.localClippingEnabled = False
            self._renderer.clippingPlanes = []
        else:
            self._renderer.localClippingEnabled = True
            self._renderer.clippingPlanes = [p3s.Plane(orientation, offset)]

    @property
    def camera(self):
        """
        vispy.scene.cameras.perspective.PerspectiveCamera : Get the camera
        """
        return self._camera

    @property
    def orbit(self):
        """
        None : Get the orbit
        """
        raise AttributeError("VisPy has not orbit")

    def resolution(self):
        """
        Returns a tuple (width, height) with the rendering resolution.
        """
        return self._resolution

    ##################################################
    # Internal methods
    ##################################################

    def _plot_mesh(self, vertices, faces, persist, colors=None):
        """
        Plots a mesh.

        Input
        ------
        vertices : [n,3], float
            Position of the vertices

        faces : [n,3], int
            Indices of the triangles associated with ``vertices``

        persist : bool
            Flag indicating if the mesh is persistent, i.e., should not be
            erased when ``reset()`` is called.

        colors : [n,3] | [3] | None
            Colors of the vertices. If `None`, black is used.
            Defaults to `None`.
        """
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3
        n_v = vertices.shape[0]
        pmin, pmax = np.min(vertices, axis=0), np.max(vertices, axis=0)

        # Assuming per-vertex colors
        if colors is None:
            # Black is default
            colors = np.zeros((n_v, 3), dtype=np.float32)
        elif colors.ndim == 1:
            colors = np.tile(colors[None, :], (n_v, 1))
        colors = colors.astype(np.float32)
        assert (
            (colors.ndim == 2) and (colors.shape[1] == 3) and (colors.shape[0] == n_v)
        )

        # Closer match to Mitsuba and Blender
        colors = np.power(colors, 1 / 1.8)

        mesh = Mesh(
            vertices=vertices, faces=faces, vertex_colors=colors, shading="flat"
        )
        self._add_child(mesh, pmin, pmax, persist=persist)

    def _plot_points(self, points, persist, colors=None, radius=0.05):
        """
        Plots a set of `n` points.

        Input
        -------
        points : [n, 3], float
            Coordinates of the `n` points.

        persist : bool
            Indicates if the points are persistent, i.e., should not be erased
            when ``reset()`` is called.

        colors : [n, 3], float | [3], float | None
            Colors of the points.

        radius : float
            Radius of the points.
        """
        assert points.ndim == 2 and points.shape[1] == 3
        n = points.shape[0]
        pmin, pmax = np.min(points, axis=0), np.max(points, axis=0)

        # Assuming per-vertex colors
        if colors is None:
            colors = np.zeros((n, 3), dtype=np.float32)
        elif colors.ndim == 1:
            colors = np.tile(colors[None, :], (n, 1))
        colors = colors.astype(np.float32)
        assert (colors.ndim == 2) and (colors.shape[1] == 3) and (colors.shape[0] == n)

        markers = Markers(pos=points, size=2 * radius, face_color=colors, alpha=0.5)
        self._add_child(markers, pmin, pmax, persist=persist)

    def _add_child(self, obj, pmin, pmax, persist):
        """
        Adds an object for display

        Input
        ------
        obj : :class:`~pythreejs.Mesh`
            Mesh to display

        pmin : [3], float
            Lowest position for the bounding box

        pmax : [3], float
            Highest position for the bounding box

        persist : bool
            Flag that indicates if the object is persistent, i.e., if it should
            be removed from the display when `reset()` is called.
        """
        self._objects.append((obj, persist))
        self._view.add(obj)

        self._bbox.expand(pmin)
        self._bbox.expand(pmax)

    def _plot_lines(self, starts, ends, width=0.5, color="black"):
        """
        Plots a set of `n` lines. This is used to plot the paths.

        Input
        ------
        starts : [n, 3], float
            Coordinates of the lines starting points

        ends : [n, 3], float
            Coordinates of the lines ending points

        width : float
            Width of the lines.
            Defaults to 0.5.

        color : str
            Color of the lines.
            Defaults to 'black'.
        """
        assert starts.ndim == 2 and starts.shape[1] == 3
        assert ends.ndim == 2 and ends.shape[1] == 3
        assert starts.shape[0] == ends.shape[0]

        segments = np.hstack((starts, ends)).astype(np.float32).reshape(-1, 2, 3)
        pmin = np.min(segments, axis=(0, 1))
        pmax = np.max(segments, axis=(0, 1))

        line_plot = LinePlot(data=segments.reshape(-1, 3), color=color, width=width)

        # Lines are not flagged as persistent as they correspond to paths, which
        # can changes from one display to the next.
        self._add_child(line_plot, pmin, pmax, persist=False)
