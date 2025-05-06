import importlib

import sionna.rt.scene
from vispy.scene import SceneCanvas

from sionna_vispy import patch
from sionna_vispy.previewer import Previewer as NewPreviewer


def test_patch() -> None:
    importlib.reload(sionna.rt.scene)

    assert not issubclass(sionna.rt.scene.Previewer, SceneCanvas)

    with patch() as cls:
        assert cls == NewPreviewer

        importlib.reload(sionna.rt.scene)

        assert not issubclass(sionna.rt.scene.Previewer, SceneCanvas)

    importlib.reload(sionna.rt.scene)

    assert not issubclass(sionna.rt.scene.Previewer, SceneCanvas)
