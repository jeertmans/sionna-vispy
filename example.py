# /// script
# dependencies = [
#   "sionna-rt==1.2.0",
#   "sionna-vispy[recommended]==1.2.0",
# ]
# requires-python = ">=3.10,<3.13"
# ///
# Inspired from:
# https://nvlabs.github.io/sionna/rt/api/paths_solvers.html
#
# type: ignore

import mitsuba as mi
import sionna
from sionna.rt import (
    PathSolver,
    PlanarArray,
    RadioMapSolver,
    Receiver,
    Transmitter,
    load_scene,
)

import sionna_vispy

# Load example scene
scene = load_scene(sionna.rt.scene.munich)

# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(
    num_rows=8,
    num_cols=2,
    vertical_spacing=0.7,
    horizontal_spacing=0.5,
    pattern="tr38901",
    polarization="VH",
)

# Configure antenna array for all receivers
scene.rx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="dipole",
    polarization="cross",
)

# Create transmitter
tx = Transmitter(
    name="tx", position=mi.Point3f(8.5, 21, 27), orientation=mi.Point3f(0, 0, 0)
)
scene.add(tx)

# Create a receiver
rx = Receiver(
    name="rx", position=mi.Point3f(45, 90, 1.5), orientation=mi.Point3f(0, 0, 0)
)
scene.add(rx)

# TX points towards RX
tx.look_at(rx)

# Compute paths
path_solver = PathSolver()
paths = path_solver(scene)
rm_solver = RadioMapSolver()
radio_map = rm_solver(scene, cell_size=(1.0, 1.0), samples_per_tx=100000000)

with sionna_vispy.patch():
    scene.preview(
        paths=paths,
        radio_map=radio_map,
        resolution=[1000, 600],
        clip_at=15.0,
        rm_vmin=-100.0,
    )


canvas = sionna_vispy.get_canvas(scene)
canvas.camera.elevation = 45
canvas.camera.azimuth = 175
canvas.camera.distance = 300

canvas.show()
canvas.app.run()
