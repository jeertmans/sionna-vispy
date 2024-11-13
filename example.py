# /// script
# dependencies = [
#   "sionna-vispy[recommended]",
# ]
# ///
# Inspired from:
# https://nvlabs.github.io/sionna/examples/Sionna_Ray_Tracing_RIS.html
#
# type: ignore

import sionna
from sionna.rt import (
    RIS,
    PlanarArray,
    Receiver,
    Transmitter,
    load_scene,
)

import sionna_vispy

scene = load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = 3e9
scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, "iso", "V")
scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, "iso", "V")

tx = Transmitter("tx", position=[-32, 10, 32], look_at=[0, 0, 0])
scene.add(tx)

rx = Receiver("rx", position=[22, 52, 1.7])
scene.add(rx)

ris = RIS(
    name="ris",
    position=[32, -9, 32],
    num_rows=100,
    num_cols=100,
    num_modes=1,
    look_at=(tx.position + rx.position) / 2,
)
scene.add(ris)

ris.phase_gradient_reflector(tx.position, rx.position)

cm = scene.coverage_map(
    num_samples=2e6,
    max_depth=5,
    los=True,
    reflection=True,
    diffraction=True,
    ris=True,
    cm_cell_size=[4, 4],
    cm_orientation=[0, 0, 0],
    cm_center=[0, 0, 1.5],
    cm_size=[200, 200],
)

paths = scene.compute_paths(max_depth=5, diffraction=True)

with sionna_vispy.patch():
    canvas = scene.preview(paths=paths, coverage_map=cm)

canvas.camera.elevation = 45
canvas.camera.azimuth = 175
canvas.camera.distance = 300
canvas.show()
canvas.app.run()
