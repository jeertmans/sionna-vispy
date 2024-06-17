import sionna

# For link-level simulations
from sionna.rt import PlanarArray, Receiver, Transmitter, load_scene

scene = load_scene(sionna.rt.scene.munich)

# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="tr38901",
    polarization="V",
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
tx = Transmitter(name="tx", position=[8.5, 21, 27])

# Add transmitter instance to scene
scene.add(tx)

# Create a receiver
rx = Receiver(name="rx", position=[45, 90, 1.5], orientation=[0, 0, 0])

# Add receiver instance to scene
scene.add(rx)

tx.look_at(rx)  # Transmitter points towards receiver

scene.frequency = 2.14e9  # in Hz; implicitly updates RadioMaterials

scene.synthetic_array = True  # If set to False, ray tracing will be done per antenna element (slower for large arrays)

paths = scene.compute_paths(max_depth=2, num_samples=1e3)

cm = scene.coverage_map(
    max_depth=2,
    diffraction=True,  # Disable to see the effects of diffraction
    cm_cell_size=(5.0, 5.0),  # Grid size of coverage map cells in m
    combining_vec=None,
    precoding_vec=None,
    num_samples=int(20e3),
)

import unittest.mock

from sionna_vispy.previewer import InteractiveDisplay

# with sionna_vispy.patch():

with unittest.mock.patch(
    "sionna.rt.previewer.InteractiveDisplay", new=InteractiveDisplay
):
    scene.preview(coverage_map=cm)
