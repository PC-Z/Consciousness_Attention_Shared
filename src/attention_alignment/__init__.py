"""Multimodal alignment for the attention oddball experiment."""

from .config import ProjectConfig, load_config
from .pipeline import build_alignment
from .session import open_session

__all__ = ["ProjectConfig", "build_alignment", "load_config", "open_session"]
__version__ = "0.1.0"
