from __future__ import annotations

import unittest.mock
from typing import TYPE_CHECKING

from .previewer import InteractiveDisplay

if TYPE_CHECKING:
    import contextlib


def patch() -> contextlib.AbstractContextManager[InteractiveDisplay]:
    """
    Monkey patch Sionna's scene previewer to use VisPy instead.
    """
    return unittest.mock.patch(
        "sionna.rt.previewer.InteractiveDisplay", new=InteractiveDisplay
    )
