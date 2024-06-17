import sionna
from sionna.rt import RIS, PlanarArray, Receiver, Transmitter, load_scene

scene = load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = 3e9  # Carrier frequency [Hz]
scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, "iso", "V")
scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, "iso", "V")

# Place a transmitter
tx = Transmitter("tx", position=[-32, 10, 32], look_at=[0, 0, 0])
scene.add(tx)

# Place a receiver (we will not actually use it
# for anything apart from referencing the position)
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

# Configure RIS as phase gradient reflector that reradiates energy
# toward the direction of the receivers
ris.phase_gradient_reflector(tx.position, rx.position)

# Compute coverage map without RIS
cm_no_ris = scene.coverage_map(
    num_samples=10e6,
    max_depth=5,
    los=True,
    reflection=True,
    diffraction=True,
    ris=False,
    cm_cell_size=[4, 4],
    cm_orientation=[0, 0, 0],
    cm_center=[0, 0, 1.5],
    cm_size=[200, 200],
)
cm_no_ris.show(vmax=-65, vmin=-150, show_ris=True, show_rx=True)
plt.title("Coverage without RIS")

# Compute coverage map with RIS
cm_ris = scene.coverage_map(
    num_samples=10e6,
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
cm_ris.show(vmax=-65, vmin=-150, show_ris=True, show_rx=True)
plt.title("Coverage with RIS")

# Visualize the coverage improvements thanks to the RIS
fig = plt.figure()
plt.imshow(
    10 * np.log10(cm_ris._value[0] / cm_no_ris._value[0]), origin="lower", vmin=0
)
plt.colorbar(label="Gain [dB]")
plt.xlabel("Cell index (X-axis)")
plt.ylabel("Cell index (Y-axis)")
plt.title("RIS Coverage Gain")
