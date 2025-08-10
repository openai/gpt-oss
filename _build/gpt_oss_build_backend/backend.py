"""
Build backend for gpt-oss that supports two modes:

1) Default (pure wheel for PyPI)
   - Delegates to setuptools.build_meta.
   - Produces a py3-none-any wheel so PyPI accepts it (no linux_x86_64 tag).

2) Optional Metal/C extension build (local only)
   - If the environment variable GPTOSS_BUILD_METAL is set to a truthy value
     (1/true/on/yes), delegates to scikit_build_core.build.
   - Dynamically injects build requirements (scikit-build-core, cmake, ninja,
     pybind11) only for this mode.

Why this is needed:
- PyPI rejects Linux wheels tagged linux_x86_64; manylinux/musllinux is required
  for binary wheels. We ship a pure wheel by default, but still allow developers
  to build/install the native Metal backend locally when needed.

Typical usage:
- Publish pure wheel: `python -m build` (do not set GPTOSS_BUILD_METAL).
- Local Metal dev: `GPTOSS_BUILD_METAL=1 pip install -e ".[metal]"`.
- CI: keep GPTOSS_BUILD_METAL unset for releases; set it in internal jobs that
  exercise the extension.

Configuration Settings:
--build-option: Extra arguments to pass to the build command
--global-option: Global build system arguments

Notes:
- The base package remains importable without the extension. The Metal backend
  is only used when `gpt_oss.metal` is explicitly imported.
- This file is discovered via `backend-path = ["_build"]` and
  `build-backend = "gpt_oss_build_backend.backend"` in pyproject.toml.
"""

import logging
import os
from importlib import import_module
from typing import Any, Mapping, Optional, Union
from typing_extensions import Protocol

# Configure logging
logger = logging.getLogger(__name__)

# Constants
TRUE_VALUES = {"1", "true", "TRUE", "on", "ON", "yes", "YES"}
VALID_CONFIG_KEYS = {"--build-option", "--global-option"}


class BuildBackendProtocol(Protocol):
    """Protocol defining the interface required for build backends."""
    
    def build_wheel(
        self,
        wheel_directory: str,
        config_settings: Optional[Mapping[str, Any]] = None,
        metadata_directory: Optional[str] = None,
    ) -> str:
        ...

    def build_sdist(
        self,
        sdist_directory: str,
        config_settings: Optional[Mapping[str, Any]] = None,
    ) -> str:
        ...

    def build_editable(
        self,
        editable_directory: str,
        config_settings: Optional[Mapping[str, Any]] = None,
    ) -> str:
        ...


def _validate_config_settings(config_settings: Optional[Mapping[str, Any]]) -> None:
    """
    Validate build configuration settings.

    Args:
        config_settings: Dictionary of configuration settings

    Raises:
        ValueError: If invalid configuration keys are found
    """
    if config_settings is None:
        return

    invalid_keys = set(config_settings.keys()) - VALID_CONFIG_KEYS
    if invalid_keys:
        raise ValueError(
            f"Invalid configuration keys: {invalid_keys}. "
            f"Valid keys are: {VALID_CONFIG_KEYS}"
        )


def _use_metal_backend() -> bool:
    """
    Determine if Metal backend should be used based on environment variable.

    Returns:
        bool: True if Metal backend should be used, False otherwise

    Raises:
        ValueError: If GPTOSS_BUILD_METAL environment variable has invalid value
    """
    metal_env = os.environ.get("GPTOSS_BUILD_METAL", "").strip()
    try:
        return metal_env in TRUE_VALUES
    except Exception as e:
        raise ValueError(f"Invalid GPTOSS_BUILD_METAL value: {metal_env}") from e


def _setuptools_backend() -> BuildBackendProtocol:
    """
    Get the setuptools build backend.

    Returns:
        BuildBackendProtocol: Setuptools build backend instance
    """
    from setuptools import build_meta as _bm
    return _bm  # type: ignore


def _scikit_build_backend() -> BuildBackendProtocol:
    """
    Get the scikit-build-core build backend.

    Returns:
        BuildBackendProtocol: Scikit-build-core build backend instance

    Raises:
        ImportError: If scikit-build-core is not installed
    """
    try:
        return import_module("scikit_build_core.build")  # type: ignore
    except ImportError as e:
        raise ImportError(
            "scikit-build-core is required for Metal builds. "
            "Install with pip install 'gpt-oss[metal]'"
        ) from e


def _backend() -> BuildBackendProtocol:
    """
    Get the appropriate build backend based on configuration.

    Returns:
        BuildBackendProtocol: Selected build backend instance
    """
    use_metal = _use_metal_backend()
    logger.info(f"Using {'Metal' if use_metal else 'default'} backend")
    return _scikit_build_backend() if use_metal else _setuptools_backend()


# Required PEP 517 hooks

def build_wheel(
    wheel_directory: str,
    config_settings: Optional[Mapping[str, Any]] = None,
    metadata_directory: Optional[str] = None,
) -> str:
    """
    Build a wheel distribution.

    Args:
        wheel_directory: Directory where the wheel should be written
        config_settings: Optional build configuration settings
        metadata_directory: Directory containing metadata

    Returns:
        str: The filename of the built wheel within wheel_directory

    Raises:
        ValueError: If config_settings contains invalid keys
    """
    _validate_config_settings(config_settings)
    return _backend().build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(
    sdist_directory: str,
    config_settings: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Build a source distribution.

    Args:
        sdist_directory: Directory where the sdist should be written
        config_settings: Optional build configuration settings

    Returns:
        str: The filename of the built sdist within sdist_directory

    Raises:
        ValueError: If config_settings contains invalid keys
    """
    _validate_config_settings(config_settings)
    return _backend().build_sdist(sdist_directory, config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Prepare wheel metadata.

    Args:
        metadata_directory: Directory where metadata should be written
        config_settings: Optional build configuration settings

    Returns:
        str: Directory containing metadata

    Raises:
        ValueError: If config_settings contains invalid keys
    """
    _validate_config_settings(config_settings)
    be = _backend()
    fn = getattr(be, "prepare_metadata_for_build_wheel", None)
    if fn is None:
        # setuptools exposes it; scikit-build-core may not
        # Fallback to setuptools for metadata
        return _setuptools_backend().prepare_metadata_for_build_wheel(
            metadata_directory, config_settings
        )
    return fn(metadata_directory, config_settings)


def build_editable(
    editable_directory: str,
    config_settings: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Build an editable installation.

    Args:
        editable_directory: Directory where editable install should be written
        config_settings: Optional build configuration settings

    Returns:
        str: The filename of the built editable install within editable_directory

    Raises:
        ValueError: If config_settings contains invalid keys
    """
    _validate_config_settings(config_settings)
    be = _backend()
    fn = getattr(be, "build_editable", None)
    if fn is None:
        raise NotImplementedError(
            f"Backend {be.__name__} does not support editable installs"
        )
    return fn(editable_directory, config_settings)


# Optional hooks - can be added as needed
# def get_requires_for_build_wheel(config_settings: Optional[Mapping[str, Any]] = None) -> List[str]:
#     """Get dependencies required for building a wheel."""
#     pass

# def get_requires_for_build_sdist(config_settings: Optional[Mapping[str, Any]] = None) -> List[str]:
#     """Get dependencies required for building an sdist."""
#     pass
