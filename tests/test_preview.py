import pytest
import sionna
from sionna.rt import RIS, PlanarArray, Receiver, Transmitter, load_scene
from sionna.rt.coverage_map import CoverageMap
from sionna.rt.paths import Paths
from sionna.rt.scene import Scene
from vispy.scene import SceneCanvas

import sionna_vispy


@pytest.fixture
def num_samples() -> int:
    return 1e2


@pytest.fixture
def max_depth() -> int:
    return 5


@pytest.fixture
def los() -> bool:
    return True


@pytest.fixture
def reflection() -> bool:
    return True


@pytest.fixture
def diffraction() -> bool:
    return True


@pytest.fixture
def ris() -> bool:
    return True


@pytest.fixture
def scene() -> Scene:
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

    return scene


@pytest.fixture
def paths(
    scene: Scene,
    num_samples: int,
    max_depth: int,
    los: bool,
    reflection: bool,
    diffraction: bool,
    ris: bool,
) -> Paths:
    return scene.compute_paths(
        num_samples=num_samples,
        max_depth=max_depth,
        los=los,
        reflection=reflection,
        diffraction=diffraction,
        ris=ris,
    )


@pytest.fixture
def coverage_map(
    scene: Scene,
    num_samples: int,
    max_depth: int,
    los: bool,
    reflection: bool,
    diffraction: bool,
    ris: bool,
) -> CoverageMap:
    return scene.coverage_map(
        num_samples=num_samples,
        max_depth=max_depth,
        los=los,
        reflection=reflection,
        diffraction=diffraction,
        ris=ris,
    )


def test_preview(scene: Scene, paths: Paths, coverage_map: CoverageMap) -> None:
    with sionna_vispy.patch():
        canvas = scene.preview(
            paths=paths, coverage_map=coverage_map, show_orientations=True
        )
        assert isinstance(canvas, SceneCanvas)
