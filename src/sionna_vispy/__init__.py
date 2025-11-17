import inspect
import unittest.mock
from collections.abc import Iterator
from contextlib import contextmanager

from sionna.rt.scene import Scene

from .previewer import Previewer


def get_canvas(scene: Scene) -> Previewer:
    """
    Get the VisPy canvas associated with a Sionna scene.

    Input
    -----
    scene : Scene
        Sionna scene with a VisPy canvas.

    Returns
    -------
    previewer : Previewer
        The VisPy canvas associated with the scene.
    """
    previewer = scene._preview_widget
    if previewer is None:
        msg = (
            "The scene does not have a preview widget. Did you call 'scene.preview()'?"
        )
        raise AttributeError(msg)
    if not isinstance(previewer, Previewer):
        msg = (
            "The scene's preview widget is not a VisPy previewer. "
            "Make sure to use 'sionna_vispy.patch()' when creating the preview."
        )
        raise ValueError(msg)
    return previewer


@contextmanager
def patch(*, patch_existing: bool = True) -> Iterator[type[Previewer]]:
    """
    Monkey patch Sionna's scene previewer to use VisPy instead.

    Input
    -----
    patch_existing : bool
        Read the local variables from the callee and patch
        any existing scene to reset the preview widget.
        When leaving the context, the previous preview widgets will be put back,
        if any were present (otherwise, the new ones are kept).
        Defaults to `True`.
    """
    callee_locals = {}

    if patch_existing:
        callee_locals = inspect.stack()[2][0].f_locals

    scenes = [
        obj
        for obj in callee_locals.values()
        if isinstance(obj, Scene) and obj._preview_widget is not None
    ]
    previewers = [scene._preview_widget for scene in scenes]

    try:
        for scene in scenes:
            scene._preview_widget = None

        with unittest.mock.patch("sionna.rt.scene.Previewer", new=Previewer) as cls:
            yield cls

    finally:
        for scene, previewer in zip(scenes, previewers, strict=True):
            scene._preview_widget = previewer
