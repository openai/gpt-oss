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
import sys
from importlib import import_module
from typing import Any, Callable, List, Mapping, Optional, Protocol, Sequence, Union


# Configuration constants
ENV_VAR_METAL_BUILD = "GPTOSS_BUILD_METAL"
TRUE_VALUES = {"1", "true", "on", "yes", "y", "t"}

# Build requirements
METAL_BUILD_REQUIREMENTS = [
    "scikit-build-core>=0.10",
    "pybind11>=2.12", 
    "cmake>=3.26",
    "ninja",
]

SETUPTOOLS_BUILD_REQUIREMENTS: List[str] = []


class BuildBackendProtocol(Protocol):
    """Protocol defining the interface for build backends."""
    
    def build_wheel(
        self,
        wheel_directory: str,
        config_settings: Optional[Mapping[str, Any]] = None,
        metadata_directory: Optional[str] = None,
    ) -> str:
        """Build a wheel and return its filename."""
        ...
    
    def build_sdist(
        self,
        sdist_directory: str,
        config_settings: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Build a source distribution and return its filename."""
        ...
    
    def get_requires_for_build_wheel(
        self,
        config_settings: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        """Get requirements for building a wheel."""
        ...
    
    def get_requires_for_build_sdist(
        self,
        config_settings: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        """Get requirements for building a source distribution."""
        ...


class BuildError(Exception):
    """Custom exception for build-related errors."""
    pass


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


def _log_info(message: str) -> None:
    """
    Simple logging function that prints to stderr with prefix.
    
    Args:
        message: The message to log.
    """
    print(f"[gpt-oss-build] {message}", file=sys.stderr)


def _log_error(message: str) -> None:
    """
    Log error messages to stderr with prefix.
    
    Args:
        message: The error message to log.
    """
    print(f"[gpt-oss-build] ERROR: {message}", file=sys.stderr)


def _validate_directory(directory: str, operation: str) -> None:
    """
    Validate that a directory exists and is writable.
    
    Args:
        directory: Path to the directory to validate.
        operation: Description of the operation for error messages.
        
    Raises:
        ConfigurationError: If directory validation fails.
    """
    if not directory:
        raise ConfigurationError(f"Directory for {operation} cannot be empty")
    
    directory_path = os.path.abspath(directory)
    
    # Create directory if it doesn't exist
    try:
        os.makedirs(directory_path, exist_ok=True)
    except OSError as e:
        raise ConfigurationError(f"Cannot create directory {directory_path} for {operation}: {e}")
    
    # Check if directory is writable
    if not os.access(directory_path, os.W_OK):
        raise ConfigurationError(f"Directory {directory_path} is not writable for {operation}")


def _parse_environment_variable(var_name: str, default: str = "") -> str:
    """
    Parse and normalize an environment variable value.
    
    Args:
        var_name: Name of the environment variable.
        default: Default value if the variable is not set.
        
    Returns:
        Normalized environment variable value.
    """
    value = os.environ.get(var_name, default).strip().lower()
    _log_info(f"Environment variable {var_name}='{os.environ.get(var_name, default)}' -> normalized: '{value}'")
    return value


def _use_metal_backend() -> bool:
    """
    Check if Metal backend should be used based on GPTOSS_BUILD_METAL environment variable.
    
    This function performs case-insensitive, whitespace-tolerant parsing of the
    environment variable and logs the decision for transparency.
    
    Returns:
        bool: True if Metal backend should be used, False otherwise.
        
    Raises:
        ConfigurationError: If there's an issue with configuration validation.
    """
    try:
        env_value = _parse_environment_variable(ENV_VAR_METAL_BUILD)
        use_metal = env_value in TRUE_VALUES
        
        if env_value:
            backend_type = "Metal (scikit-build-core)" if use_metal else "setuptools (invalid value)"
            _log_info(f"Backend selection: {backend_type}")
        else:
            _log_info("Backend selection: setuptools (default - no environment variable set)")
        
        return use_metal
    except Exception as e:
        _log_error(f"Failed to determine backend from environment: {e}")
        _log_info("Falling back to setuptools backend")
        return False


def _setuptools_backend() -> BuildBackendProtocol:
    """
    Get the setuptools build backend.
    
    Returns:
        BuildBackendProtocol: The setuptools build backend instance.
        
    Raises:
        BuildError: If setuptools backend cannot be imported.
    """
    try:
        from setuptools import build_meta as _bm  # type: ignore
        _log_info("Successfully imported setuptools build backend")
        return _bm
    except ImportError as e:
        _log_error(f"Failed to import setuptools build backend: {e}")
        raise BuildError("setuptools is required but not available") from e


def _scikit_build_backend() -> BuildBackendProtocol:
    """
    Get the scikit-build-core build backend.
    
    Returns:
        BuildBackendProtocol: The scikit-build-core build backend instance.
        
    Raises:
        BuildError: If scikit-build-core backend cannot be imported.
    """
    try:
        backend = import_module("scikit_build_core.build")
        _log_info("Successfully imported scikit-build-core backend")
        return backend
    except ImportError as e:
        _log_error(f"Failed to import scikit-build-core: {e}")
        _log_error("Install Metal build dependencies with: pip install 'scikit-build-core>=0.10' cmake ninja")
        raise BuildError("scikit-build-core is required for Metal backend but not available") from e


def _backend() -> BuildBackendProtocol:
    """
    Get the appropriate build backend based on configuration.
    
    This function selects between setuptools and scikit-build-core backends
    based on the environment configuration, with comprehensive error handling.
    
    Returns:
        BuildBackendProtocol: The selected build backend instance.
        
    Raises:
        BuildError: If the selected backend cannot be loaded.
        ConfigurationError: If there's a configuration issue.
    """
    try:
        if _use_metal_backend():
            _log_info("Configuring scikit-build-core backend for Metal extension")
            return _scikit_build_backend()
        else:
            _log_info("Configuring setuptools backend for pure Python wheel")
            return _setuptools_backend()
    except (BuildError, ConfigurationError):
        # Re-raise known exceptions
        raise
    except Exception as e:
        _log_error(f"Unexpected error while selecting build backend: {e}")
        raise BuildError(f"Failed to configure build backend: {e}") from e


def _safe_getattr(obj: Any, name: str, default: Optional[Callable] = None) -> Optional[Callable]:
    """
    Safely get an attribute from an object with logging.
    
    Args:
        obj: The object to get the attribute from.
        name: The name of the attribute.
        default: Default value if attribute doesn't exist.
        
    Returns:
        The attribute value or default.
    """
    attr = getattr(obj, name, default)
    if attr is None:
        _log_info(f"Backend doesn't implement {name}")
    else:
        _log_info(f"Backend supports {name}")
    return attr


# Required PEP 517 hooks

def build_wheel(
    wheel_directory: str,
    config_settings: Optional[Mapping[str, Any]] = None,
    metadata_directory: Optional[str] = None,
) -> str:
    """
    Build a wheel in the specified directory.
    
    Args:
        wheel_directory: Directory where the wheel should be built.
        config_settings: Optional configuration settings.
        metadata_directory: Optional metadata directory.
        
    Returns:
        str: The filename of the built wheel.
        
    Raises:
        BuildError: If the wheel build fails.
        ConfigurationError: If directory validation fails.
    """
    try:
        _validate_directory(wheel_directory, "wheel building")
        _log_info(f"Building wheel in directory: {wheel_directory}")
        
        backend = _backend()
        result = backend.build_wheel(wheel_directory, config_settings, metadata_directory)
        
        _log_info(f"Successfully built wheel: {result}")
        return result
    except (BuildError, ConfigurationError):
        raise
    except Exception as e:
        _log_error(f"Wheel build failed: {e}")
        raise BuildError(f"Failed to build wheel: {e}") from e


def build_sdist(
    sdist_directory: str, 
    config_settings: Optional[Mapping[str, Any]] = None
) -> str:
    """
    Build a source distribution in the specified directory.
    
    Args:
        sdist_directory: Directory where the sdist should be built.
        config_settings: Optional configuration settings.
        
    Returns:
        str: The filename of the built source distribution.
        
    Raises:
        BuildError: If the sdist build fails.
        ConfigurationError: If directory validation fails.
    """
    try:
        _validate_directory(sdist_directory, "source distribution building")
        _log_info(f"Building sdist in directory: {sdist_directory}")
        
        backend = _backend()
        result = backend.build_sdist(sdist_directory, config_settings)
        
        _log_info(f"Successfully built sdist: {result}")
        return result
    except (BuildError, ConfigurationError):
        raise
    except Exception as e:
        _log_error(f"Sdist build failed: {e}")
        raise BuildError(f"Failed to build sdist: {e}") from e


def prepare_metadata_for_build_wheel(
    metadata_directory: str, 
    config_settings: Optional[Mapping[str, Any]] = None
) -> str:
    """
    Prepare metadata for wheel building.
    
    Args:
        metadata_directory: Directory where metadata should be prepared.
        config_settings: Optional configuration settings.
        
    Returns:
        str: The name of the metadata directory.
        
    Raises:
        BuildError: If metadata preparation fails.
        ConfigurationError: If directory validation fails.
    """
    try:
        _validate_directory(metadata_directory, "metadata preparation")
        _log_info(f"Preparing metadata in directory: {metadata_directory}")
        
        backend = _backend()
        fn = _safe_getattr(backend, "prepare_metadata_for_build_wheel")
        
        if fn is None:
            # Fallback to setuptools if current backend doesn't support it
            _log_info("Using setuptools fallback for metadata preparation")
            setuptools_backend = _setuptools_backend()
            result = setuptools_backend.prepare_metadata_for_build_wheel(
                metadata_directory, config_settings
            )
        else:
            result = fn(metadata_directory, config_settings)
        
        _log_info(f"Successfully prepared metadata: {result}")
        return result
    except (BuildError, ConfigurationError):
        raise
    except Exception as e:
        _log_error(f"Metadata preparation failed: {e}")
        raise BuildError(f"Failed to prepare metadata: {e}") from e


# Optional hooks

def build_editable(
    main
    wheel_directory: str, 
    config_settings: Optional[Mapping[str, Any]] = None,
    metadata_directory: Optional[str] = None,

) -> str:
    """
    Build an editable wheel in the specified directory.
    
    Args:
        wheel_directory: Directory where the editable wheel should be built.
        config_settings: Optional configuration settings.
        metadata_directory: Optional metadata directory.
        
    Returns:
        str: The filename of the built editable wheel.
        
    Raises:
        BuildError: If the editable build fails or is not supported.
        ConfigurationError: If directory validation fails.
    """
    try:
        _validate_directory(wheel_directory, "editable wheel building")
        _log_info(f"Building editable install in directory: {wheel_directory}")
        
        backend = _backend()
        fn = _safe_getattr(backend, "build_editable")
        
        if fn is None:
            raise BuildError("Editable installs not supported by the selected backend")
        
        result = fn(wheel_directory, config_settings, metadata_directory)
        _log_info(f"Successfully built editable wheel: {result}")
        return result
    except (BuildError, ConfigurationError):
        raise
    except Exception as e:
        _log_error(f"Editable build failed: {e}")
        raise BuildError(f"Failed to build editable wheel: {e}") from e


def get_requires_for_build_wheel(
    config_settings: Optional[Mapping[str, Any]] = None,
) -> Sequence[str]:
    """
    Get the build requirements for building a wheel.
    
    Args:
        config_settings: Optional configuration settings.
        
    Returns:
        Sequence[str]: List of build requirements.
        
    Raises:
        ConfigurationError: If there's a configuration issue.
    """
    try:
        if _use_metal_backend():
            requirements = METAL_BUILD_REQUIREMENTS.copy()
            _log_info(f"Metal backend requires: {', '.join(requirements)}")
            return requirements
        else:
            # Get requirements from setuptools backend
            setuptools_backend = _setuptools_backend()
            fn = _safe_getattr(setuptools_backend, "get_requires_for_build_wheel")
            
            if fn is not None:
                requirements = list(fn(config_settings))
            else:
                requirements = SETUPTOOLS_BUILD_REQUIREMENTS.copy()
            
            _log_info(f"Setuptools backend requires: {', '.join(requirements) if requirements else 'no additional packages'}")
            return requirements
    except Exception as e:
        _log_error(f"Failed to get build requirements: {e}")
        raise ConfigurationError(f"Unable to determine build requirements: {e}") from e


def get_requires_for_build_sdist(
    config_settings: Optional[Mapping[str, Any]] = None,
) -> Sequence[str]:
    """
    Get the build requirements for building a source distribution.
    
    Args:
        config_settings: Optional configuration settings.
        
    Returns:
        Sequence[str]: List of build requirements.
        
    Raises:
        ConfigurationError: If there's a configuration issue.
    """
    try:
        backend = _backend()
        fn = _safe_getattr(backend, "get_requires_for_build_sdist")
        
        if fn is None:
            requirements: List[str] = []
        else:
            requirements = list(fn(config_settings))
        
        _log_info(f"SDist requires: {', '.join(requirements) if requirements else 'no additional packages'}")
        return requirements
    except Exception as e:
        _log_error(f"Failed to get sdist requirements: {e}")
        raise ConfigurationError(f"Unable to determine sdist requirements: {e}") from e


def get_requires_for_build_editable(
    config_settings: Optional[Mapping[str, Any]] = None,
) -> Sequence[str]:
    """
    Get the build requirements for building an editable install.
    
    Args:
        config_settings: Optional configuration settings.
        
    Returns:
        Sequence[str]: List of build requirements.
        
    Raises:
        ConfigurationError: If there's a configuration issue.
    """
    try:
        if _use_metal_backend():
            requirements = METAL_BUILD_REQUIREMENTS.copy()
            _log_info(f"Editable Metal backend requires: {', '.join(requirements)}")
            return requirements
        else:
            setuptools_backend = _setuptools_backend()
            fn = _safe_getattr(setuptools_backend, "get_requires_for_build_editable")
            
            if fn is None:
                requirements: List[str] = []
            else:
                requirements = list(fn(config_settings))
            
            _log_info(f"Editable setuptools backend requires: {', '.join(requirements) if requirements else 'no additional packages'}")
            return requirements
    except Exception as e:
        _log_error(f"Failed to get editable build requirements: {e}")
        raise ConfigurationError(f"Unable to determine editable build requirements: {e}") from e


# Future expansion hooks (currently unused but available for extension)

def get_requires_for_build_meta(
    config_settings: Optional[Mapping[str, Any]] = None,
) -> Sequence[str]:
    """
    Get requirements for building metadata (future PEP extension).
    
    Args:
        config_settings: Optional configuration settings.
        
    Returns:
        Sequence[str]: List of requirements (currently empty).
    """
    _log_info("get_requires_for_build_meta called (future extension)")
    return []


def build_meta(
    meta_directory: str,
    config_settings: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Build metadata only (future PEP extension).
    
    Args:
        meta_directory: Directory for metadata.
        config_settings: Optional configuration settings.
        
    Returns:
        str: Metadata filename.
        
    Raises:
        BuildError: Always, as this is not yet implemented.
    """
    _log_info("build_meta called (future extension)")
    raise BuildError("build_meta is not yet implemented") 