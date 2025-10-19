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

Why this is needed
- PyPI rejects Linux wheels tagged linux_x86_64; manylinux/musllinux is required
  for binary wheels. We ship a pure wheel by default, but still allow developers
  to build/install the native Metal backend locally when needed.

Typical usage
- Publish pure wheel: `python -m build` (do not set GPTOSS_BUILD_METAL).
- Local Metal dev: `GPTOSS_BUILD_METAL=1 pip install -e ".[metal]"`.
- CI: keep GPTOSS_BUILD_METAL unset for releases; set it in internal jobs that
  exercise the extension.

Notes
- The base package remains importable without the extension. The Metal backend
  is only used when `gpt_oss.metal` is explicitly imported.
- This file is discovered via `backend-path = ["_build"]` and
  `build-backend = "gpt_oss_build_backend.backend"` in pyproject.toml.
"""
import os
from importlib import import_module
from typing import Any, Mapping, Sequence


TRUE_VALUES = {"1", "true", "TRUE", "on", "ON", "yes", "YES"}


def _use_metal_backend() -> bool:
    return str(os.environ.get("GPTOSS_BUILD_METAL", "")).strip() in TRUE_VALUES


def _setuptools_backend():
    from setuptools import build_meta as _bm  # type: ignore

    return _bm


def _scikit_build_backend():
    return import_module("scikit_build_core.build")


def _backend():
    return _scikit_build_backend() if _use_metal_backend() else _setuptools_backend()


# Required PEP 517 hooks

def build_wheel(
    wheel_directory: str,
    config_settings: Mapping[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    return _backend().build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(
    sdist_directory: str, config_settings: Mapping[str, Any] | None = None
) -> str:
    return _backend().build_sdist(sdist_directory, config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings: Mapping[str, Any] | None = None
) -> str:
    # Fallback if backend doesn't implement it
    be = _backend()
    fn = getattr(be, "prepare_metadata_for_build_wheel", None)
    if fn is None:
        # setuptools exposes it; scikit-build-core may not. Defer to building a wheel for metadata.
        return _setuptools_backend().prepare_metadata_for_build_wheel(
            metadata_directory, config_settings
        )
    return fn(metadata_directory, config_settings)


# Optional hooks

def build_editable(
    editable_directory: str, config_settings: Mapping[str, Any] | None = None, metadata_directory: str | None = None
) -> str:
    be = _backend()
    fn = getattr(be, "build_editable", None)
    if fn is None:
        # setuptools implements build_editable; if not available, raise the standard error
        raise RuntimeError("Editable installs not supported by the selected backend")
    return fn(editable_directory, config_settings)


def get_requires_for_build_wheel(
    config_settings: Mapping[str, Any] | None = None,
) -> Sequence[str]:
    if _use_metal_backend():
        # Add dynamic build requirements only when building the Metal backend
        return [
            "scikit-build-core>=0.10",
            "pybind11>=2.12",
            "cmake>=3.26",
            "ninja",
        ]
    # setuptools usually returns []
    return list(_setuptools_backend().get_requires_for_build_wheel(config_settings))


def get_requires_for_build_sdist(
    config_settings: Mapping[str, Any] | None = None,
) -> Sequence[str]:
    # No special requirements for SDist
    be = _backend()
    fn = getattr(be, "get_requires_for_build_sdist", None)
    if fn is None:
        return []
    return list(fn(config_settings))


def get_requires_for_build_editable(
    config_settings: Mapping[str, Any] | None = None,
) -> Sequence[str]:
    if _use_metal_backend():
        return [
            "scikit-build-core>=0.10",
            "pybind11>=2.12",
            "cmake>=3.26",
            "ninja",
        ]
    be = _setuptools_backend()
    fn = getattr(be, "get_requires_for_build_editable", None)
    if fn is None:
        return []
    return list(fn(config_settings)) 