"""Backend implementations for the simple browser tool."""

from .base_backend import Backend, BackendError
from .exa_backend import ExaBackend
from .tavily_backend import TavilyBackend

__all__ = ["Backend", "BackendError", "ExaBackend", "TavilyBackend"]