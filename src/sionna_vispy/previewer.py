#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
3D scene and paths viewer using VisPy.

This file is a modified version of :mod:`sionna.rt.previewer` to use
VisPy as a viewer rather than pythreejs.
"""

import warnings

import drjit as dr
import matplotlib as mpl
import mitsuba as mi
import numpy as np
from sionna.rt.constants import (
    INTERACTION_TYPE_TO_COLOR,
    LOS_COLOR,
    InteractionType,
)
from sionna.rt.utils import (
    rotation_matrix,
    scene_scale,
)
from vispy.geometry.generation import create_cylinder
from vispy.scene import SceneCanvas
from vispy.scene.cameras.turntable import TurntableCamera
from vispy.scene.visuals import Image, LinePlot, Markers, Mesh
from vispy.visuals.filters.clipping_planes import PlanesClipper
from vispy.visuals.transforms import MatrixTransform, STTransform


class Previewer(SceneCanvas):
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

        self.unfreeze()  # allow creating attributes

        self._resolution = resolution
        self._sionna_scene = scene  # self._scene is already defined by VisPy

        # List of objects in the scene
        self._objects = []
        # Bounding box of the scene
        self._bbox = mi.ScalarBoundingBox3f()  # type: ignore[reportAttributeAccessIssue]

        ####################################################
        # Setup the viewer
        ####################################################

        # View
        self._view = self.central_widget.add_view()
        self._view.camera = TurntableCamera(fov=fov)
        self._camera = self._view.camera
        self._camera.depth_value = 1e6
        self._clipper = PlanesClipper()

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
        Automatically place the camera and observable range.
        """
        self._camera.set_range()

    def plot_radio_devices(self, show_orientations=False):  # noqa: C901
        """
        Plots the radio devices.

        If `show_orientations` is set to `True`, the orientation of each device
        is shown using an arrow.

        Input
        ------
        show_orientations : bool
            If set to `True`, the orientation of the radio device is shown using
            an arrow. Defaults to `False`.
        """
        scene = self._sionna_scene
        sc = scene_scale(scene)
        # If scene is empty, set the scene scale to 1
        if sc == 0.0:
            sc = 1.0

        tx_positions = [tx.position.numpy().T[0] for tx in scene.transmitters.values()]
        rx_positions = [rx.position.numpy().T[0] for rx in scene.receivers.values()]

        sources_colors = [tx.color for tx in scene.transmitters.values()]
        target_colors = [rx.color for rx in scene.receivers.values()]

        # Radio emitters, shown as points
        p = np.array(list(tx_positions) + list(rx_positions))
        # Stop here if no radio devices to plot
        if p.shape[0] == 0:
            return
        albedo = np.array(sources_colors + target_colors)

        # Expand the bounding box to include radio devices
        pmin = np.min(p, axis=0)
        pmax = np.max(p, axis=0)
        self._bbox.expand(pmin)
        self._bbox.expand(pmax)

        # Radio devices are not persistent.
        default_radius = max(0.0025 * sc, 1)
        only_default = True
        radii = []
        for devices in scene.transmitters, scene.receivers:
            for rd in devices.values():
                r = rd.display_radius
                if r is not None:
                    radii.append(r)
                    only_default = False
                else:
                    radii.append(default_radius)

        if only_default:
            self._plot_points(p, persist=False, colors=albedo, radius=default_radius)
        else:
            # Since we can only have one radius value per draw call,
            # we group the points to plot by radius.
            unique_radii, mapping = np.unique(radii, return_inverse=True)
            for i, r in enumerate(unique_radii):
                mask = mapping == i
                self._plot_points(p[mask], persist=False, colors=albedo[mask], radius=r)

        if show_orientations:
            line_length = 0.05 * sc
            head_length = 0.05 * line_length
            zeros = np.zeros((3,))

            for devices in [scene.transmitters.values(), scene.receivers.values()]:
                if len(devices) == 0:
                    continue
                starts, ends, colors = [], [], []
                for rd in devices:
                    # Arrow line
                    starts.append(rd.position.numpy()[:, 0])
                    rot_mat = rotation_matrix(rd.orientation)
                    local_endpoint = mi.Point3f(line_length, 0.0, 0.0)
                    endpoint = rd.position + rot_mat @ local_endpoint
                    endpoint = endpoint.numpy()[:, 0]
                    ends.append(endpoint)
                    colors.append([rd.color[0], rd.color[1], rd.color[2]])

                    meshdata = create_cylinder(
                        rows=80,
                        cols=80,
                        radius=(0.3 * head_length, 0),
                        length=head_length,
                    )
                    angles = rd.orientation.numpy()
                    mesh = Mesh(color=rd.color, meshdata=meshdata)
                    mesh.transform = MatrixTransform()
                    mesh.transform.rotate(np.rad2deg(angles[2]), (1, 0, 0))
                    mesh.transform.rotate(np.rad2deg(angles[1] + np.pi / 2), (0, 1, 0))
                    mesh.transform.rotate(np.rad2deg(angles[0]), (0, 0, 1))
                    mesh.transform.translate(np.append(endpoint, 1))

                    self._add_child(mesh, zeros, zeros, persist=False)

                self._plot_lines(
                    np.array(starts), np.array(ends), width=2, colors=colors
                )

    def plot_paths(self, paths, line_width=1.0):
        """
        Plot the ``paths``.

        Input
        -----
        paths : :class:`~rt.Paths`
            Paths to plot

        line_width : float
            Width of the lines.
            Defaults to 0.8.
        """

        vertices = paths.vertices.numpy()
        valid = paths.valid.numpy()
        types = paths.interactions.numpy()
        max_depth = vertices.shape[0]

        num_paths = vertices.shape[-2]
        if num_paths == 0:
            return  # Nothing to do

        # Build sources and targets
        src_positions, tgt_positions = paths.sources, paths.targets
        src_positions = src_positions.numpy().T
        tgt_positions = tgt_positions.numpy().T

        num_src = src_positions.shape[0]
        num_tgt = tgt_positions.shape[0]

        # Merge device and antenna dimensions if required
        if not paths.synthetic_array:
            # The dimension corresponding to the number of antenna patterns
            # is removed as it is a duplicate
            num_rx = paths.num_rx
            rx_array_size = paths.rx_array.array_size
            num_rx_patterns = len(paths.rx_array.antenna_pattern.patterns)
            #
            num_tx = paths.num_tx
            tx_array_size = paths.tx_array.array_size
            num_tx_patterns = len(paths.tx_array.antenna_pattern.patterns)
            #
            vertices = np.reshape(
                vertices,
                [
                    max_depth,
                    num_rx,
                    num_rx_patterns,
                    rx_array_size,
                    num_tx,
                    num_tx_patterns,
                    tx_array_size,
                    -1,
                    3,
                ],
            )
            valid = np.reshape(
                valid,
                [
                    num_rx,
                    num_rx_patterns,
                    rx_array_size,
                    num_tx,
                    num_tx_patterns,
                    tx_array_size,
                    -1,
                ],
            )
            types = np.reshape(
                types,
                [
                    max_depth,
                    num_rx,
                    num_rx_patterns,
                    rx_array_size,
                    num_tx,
                    num_tx_patterns,
                    tx_array_size,
                    -1,
                ],
            )
            vertices = vertices[:, :, 0, :, :, 0, :, :, :]
            types = types[:, :, 0, :, :, 0, :, :]
            valid = valid[:, 0, :, :, 0, :, :]
            vertices = np.reshape(vertices, [max_depth, num_tgt, num_src, -1, 3])
            valid = np.reshape(valid, [num_tgt, num_src, -1])
            types = np.reshape(types, [max_depth, num_tgt, num_src, -1])

        # Emit directly two lists of the beginnings and endings of line segments
        starts = []
        ends = []
        colors = []
        for rx in range(num_tgt):  # For each receiver
            for tx in range(num_src):  # For each transmitter
                for p in range(num_paths):  # For each path
                    if not valid[rx, tx, p]:
                        continue
                    start = src_positions[tx]
                    i = 0
                    color = LOS_COLOR
                    while i < max_depth:
                        t = types[i, rx, tx, p]
                        if t == InteractionType.NONE:
                            break
                        end = vertices[i, rx, tx, p]
                        starts.append(start)
                        ends.append(end)
                        colors.append(color)
                        start = end
                        color = INTERACTION_TYPE_TO_COLOR[t]
                        i += 1
                    # Explicitly add the path endpoint
                    starts.append(start)
                    ends.append(tgt_positions[rx])
                    colors.append(color)

        self._plot_lines(
            np.vstack(starts), np.vstack(ends), np.vstack(colors), line_width
        )

    def plot_scene(self):
        """
        Plots the meshes that make the scene.
        """
        shapes = self._sionna_scene.mi_scene.shapes()
        n = len(shapes)
        if n <= 0:
            return

        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wi = mi.Vector3f(0, 0, 1)  # type: ignore[reportAttributeAccessIssue]

        # Shapes (e.g. buildings)
        vertices, faces, albedos = [], [], []
        f_offset = 0
        for s in shapes:
            null_transmission = s.bsdf().eval_null_transmission(si).numpy()
            if np.min(null_transmission) > 0.99:
                # The BSDF for this shape was probably set to `null`, do not
                # include it in the scene preview.
                continue

            n_vertices = s.vertex_count()
            v = s.vertex_position(dr.arange(mi.UInt32, n_vertices))  # type: ignore[reportArgumentType]
            v = np.transpose(v.numpy())
            vertices.append(v)

            f = s.face_indices(dr.arange(mi.UInt32, s.face_count()))  # type: ignore[reportArgumentType]
            f = np.transpose(f.numpy())
            faces.append(f + f_offset)
            f_offset += n_vertices

            albedo = np.array(s.bsdf().radio_material.color)

            albedos.append(np.tile(albedo, (n_vertices, 1)))

        # Plot all objects as a single VisPy mesh, which is must faster
        # than creating individual mesh objects in large scenes.
        self._plot_mesh(
            np.concatenate(vertices, axis=0),
            np.concatenate(faces, axis=0),
            persist=True,  # The scene geometry is persistent
            colors=np.concatenate(albedos, axis=0),
        )

    def plot_radio_map(
        self,
        radio_map,
        tx=0,
        db_scale=True,
        vmin=None,
        vmax=None,
        metric="path_gain",
    ):
        """
        Plot the coverage map as a textured rectangle in the scene. Regions
        where the coverage map is zero-valued are made transparent.
        """
        to_world = radio_map.to_world

        tensor = radio_map.transmitter_radio_map(metric, tx)
        tensor = tensor.numpy()

        # Create a rectangle from two triangles
        p00 = to_world.transform_affine([-1, -1, 0]).numpy().T[0]
        p01 = to_world.transform_affine([1, -1, 0]).numpy().T[0]
        p10 = to_world.transform_affine([-1, 1, 0]).numpy().T[0]
        p11 = to_world.transform_affine([1, 1, 0]).numpy().T[0]

        vertices = np.array([p00, p01, p10, p11])
        pmin = np.min(vertices, axis=0)
        pmax = np.max(vertices, axis=0)

        to_map, normalizer, color_map = self._coverage_map_color_mapping(
            tensor, db_scale=db_scale, vmin=vmin, vmax=vmax
        )
        texture = color_map(normalizer(to_map)).astype(np.float32)
        texture[:, :, 3] = (tensor > 0.0).astype(np.float32)
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

    def set_clipping_plane(self, offset, orientation):
        """
        Set a plane such that the scene preview is clipped (cut) by that plane.

        The clipping plane has normal orientation ``orientation`` and
        offset ``offset``. This allows, e.g., visualizing the interior of meshes
        such as buildings.

        Input
        -----
        offset : float
            Offset to position the plane

        orientation : tuple[float, float, float]
            Normal vector of the clipping plane
        """
        if offset is None:
            self._clipper.clipping_planes = None
        else:
            orientation = np.asarray(orientation)
            self._clipper.clipping_planes = np.array(
                [[offset * -orientation, orientation]]
            )

    def show_legend(self, show_paths, show_devices):
        r"""
        Display the legend
        """
        del show_paths
        del show_devices
        warnings.warn(
            "The legend is not yet implemented in VisPy. \n"
            "If you want to help us and contribute, please reach out on GitHub!",
            stacklevel=2,
        )

    ##################################################
    # Accessors
    ##################################################

    def resolution(self):
        """
        (float, float) : Rendering resolution `(width, height)`
        """
        return self._resolution

    @property
    def camera(self):
        """
        vispy.scene.cameras.perspective.PerspectiveCamera : Get the camera
        """
        return self._camera

    @property
    def orbit(self):
        raise AttributeError("VisPy has not orbit controls like PyThreeJS")

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
        mesh.shading_filter.ambiant_light = (1, 1, 1, 0.8)  # type: ignore
        mesh.attach(self._clipper)
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

        markers = Markers(
            pos=points,
            size=2 * radius,
            edge_width_rel=0,
            face_color=colors,
            scaling="scene",
            alpha=0.5,  # type: ignore[reportArgumentType]
        )
        self._add_child(markers, pmin, pmax, persist=persist)

    def _add_child(self, obj, pmin, pmax, persist):
        """
        Adds an object for display.

        Input
        ------
        obj : VisPy node to display

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

    def _plot_lines(self, starts, ends, colors, width):
        """
        Plots a set of `n` lines. This is used to plot the paths.

        Input
        ------
        starts : [n, 3], float
            Coordinates of the lines starting points

        ends : [n, 3], float
            Coordinates of the lines ending points

        colors : str
            Color of the lines.

        width : float
            Width of the lines.
        """
        assert starts.ndim == 2 and starts.shape[1] == 3
        assert ends.ndim == 2 and ends.shape[1] == 3
        assert starts.shape[0] == ends.shape[0]

        paths = np.hstack((starts, ends)).astype(np.float32).reshape(-1, 3)
        connect = np.ones(paths.shape[0], dtype=bool)
        connect[1::2] = False

        pmin = np.min(paths, axis=0)
        pmax = np.max(paths, axis=0)

        colors = np.repeat(colors, 2, axis=0)

        line_plot = LinePlot(
            data=paths,
            color=colors,  # type: ignore[reportArgumentType]
            width=width,
            marker_size=0,
            connect=connect,  # type: ignore[reportArgumentType]
        )

        # Lines are not flagged as persistent as they correspond to paths, which
        # can changes from one display to the next.
        self._add_child(line_plot, pmin, pmax, persist=False)

    def _coverage_map_color_mapping(
        self, coverage_map, db_scale=True, vmin=None, vmax=None
    ):
        """
        Prepare a Matplotlib color maps and normalizing helper based on the
        requested value scale to be displayed.
        Also applies the dB scaling to a copy of the coverage map, if requested.
        """
        valid = np.logical_and(coverage_map > 0.0, np.isfinite(coverage_map))
        coverage_map = coverage_map.copy()
        if db_scale:
            coverage_map[valid] = 10.0 * np.log10(coverage_map[valid])
        else:
            coverage_map[valid] = coverage_map[valid]

        if vmin is None:
            vmin = coverage_map[valid].min()
        if vmax is None:
            vmax = coverage_map[valid].max()
        normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)  # type: ignore[reportAttributeAccessIssue]
        color_map = mpl.colormaps.get_cmap("viridis")
        return coverage_map, normalizer, color_map
