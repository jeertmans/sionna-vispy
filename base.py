import sionna

# For link-level simulations
from sionna.rt import RIS, PlanarArray, Receiver, Transmitter, load_scene

import sionna_vispy

scene = load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = 3e9
scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, "iso", "V")
scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, "iso", "V")

tx = Transmitter("tx", position=[-32, 10, 32], look_at=[0, 0, 0])
scene.add(tx)

rx = Receiver("rx", position=[22, 52, 1.7])
scene.add(rx)

# Place RIS
ris = RIS(
    name="ris",
    position=[32, -9, 32],
    num_rows=100,
    num_cols=100,
    num_modes=1,
    look_at=(tx.position + rx.position) / 2,
)  # Look in between TX and RX
scene.add(ris)

ris.phase_gradient_reflector(tx.position, rx.position)


paths = scene.compute_paths(
    num_samples=2e1,  # increase for better resolution
    max_depth=5,
    los=True,
    reflection=True,
    diffraction=True,
    ris=True,
)


cm = scene.coverage_map(
    num_samples=2e1,  # increase for better resolution
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

paths = None

with sionna_vispy.patch():
    canvas = scene.preview(paths=paths, coverage_map=cm, show_orientations=True)
    canvas.show()
    canvas.app.run()
