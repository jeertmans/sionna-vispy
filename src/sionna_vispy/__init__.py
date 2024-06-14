import unittest.mock

from .previewer import InteractiveDisplay


def patch() -> unittest.mock.patch[type[InteractiveDisplay]]:
    """
    Monkey patch Sionna's scene previewer to use VisPy instead.
    """
    return unittest.mock.patch(
        "sionna.rt.previewer.InteractiveDisplay", new=InteractiveDisplay
    )
