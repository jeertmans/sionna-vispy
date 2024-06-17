import importlib

import sionna.rt.scene
from vispy.scene import SceneCanvas

from sionna_vispy import patch
from sionna_vispy.previewer import InteractiveDisplay as NewInteractiveDisplay


def test_patch() -> None:
    importlib.reload(sionna.rt.scene)

    assert not issubclass(sionna.rt.scene.InteractiveDisplay, SceneCanvas)

    with patch() as cls:
        assert cls == NewInteractiveDisplay

        importlib.reload(sionna.rt.scene)

        assert not issubclass(sionna.rt.scene.InteractiveDisplay, SceneCanvas)

    importlib.reload(sionna.rt.scene)

    assert not issubclass(sionna.rt.scene.InteractiveDisplay, SceneCanvas)
