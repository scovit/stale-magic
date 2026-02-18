"""stale-magic: Makefile-like Jupyter cell magic with smart skipping."""

from ._version import __version__
from .magic import load_ipython_extension

__all__ = ["__version__", "load_ipython_extension"]
