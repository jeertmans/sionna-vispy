import pytest
import sionna
from sionna.rt import (
    Paths,
    PathSolver,
    PlanarArray,
    RadioMap,
    RadioMapSolver,
    Receiver,
    Scene,
    Transmitter,
    load_scene,
)
from vispy.scene import SceneCanvas

import sionna_vispy


@pytest.fixture
def num_samples() -> int:
    return 10_000


@pytest.fixture
def max_depth() -> int:
    return 2


@pytest.fixture
def los() -> bool:
    return True


@pytest.fixture
def specular_reflection() -> bool:
    return True


@pytest.fixture
def diffuse_reflection() -> bool:
    return True


@pytest.fixture
def reflection() -> bool:
    return True


@pytest.fixture
def refraction() -> bool:
    return True


@pytest.fixture
def scene() -> Scene:
    scene = load_scene(sionna.rt.scene.simple_street_canyon)
    scene.frequency = 3e9
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    tx = Transmitter(name="tx", position=[-32, 10, 32], look_at=[0, 0, 0])  # type: ignore[reportArgumentType]
    scene.add(tx)

    rx = Receiver(name="rx", position=[22, 52, 1.7])  # type: ignore[reportArgumentType]
    scene.add(rx)

    return scene


@pytest.fixture
def paths(
    scene: Scene,
    num_samples: int,
    max_depth: int,
    los: bool,
    specular_reflection: bool,
    diffuse_reflection: bool,
    refraction: bool,
) -> Paths:
    path_solver = PathSolver()
    return path_solver(
        scene,
        samples_per_src=num_samples,
        max_depth=max_depth,
        los=los,
        specular_reflection=specular_reflection,
        diffuse_reflection=diffuse_reflection,
        refraction=refraction,
    )


@pytest.fixture
def radio_map(
    scene: Scene,
    num_samples: int,
    max_depth: int,
    los: bool,
    specular_reflection: bool,
    diffuse_reflection: bool,
    refraction: bool,
) -> RadioMap:
    rm_solver = RadioMapSolver()
    return rm_solver(
        scene,
        samples_per_tx=num_samples,
        max_depth=max_depth,
        los=los,
        specular_reflection=specular_reflection,
        diffuse_reflection=diffuse_reflection,
        refraction=refraction,
    )


def test_get_canvas(scene: Scene) -> None:
    if scene._preview_widget is not None:
        scene._preview_widget = None
    with pytest.raises(
        AttributeError, match="The scene does not have a preview widget"
    ):
        sionna_vispy.get_canvas(scene)


def test_preview(scene: Scene, paths: Paths, radio_map: RadioMap) -> None:
    with (
        sionna_vispy.patch(),
        pytest.warns(match="Point picking is not yet implemented in VisPy"),
    ):
        scene.preview(paths=paths, radio_map=radio_map, show_orientations=True)  # type: ignore[reportArgumentType]
        canvas = sionna_vispy.get_canvas(scene)
        assert isinstance(canvas, SceneCanvas)
