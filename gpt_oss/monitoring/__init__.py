"""
GPT-OSS Hallucination Monitor

A comprehensive monitoring system for detecting hallucination risk in LLM outputs.
"""

from .halluci_monitor import HallucinationMonitor
from .config import MonitorConfig, MonitorThresholds

__all__ = ["HallucinationMonitor", "MonitorConfig", "MonitorThresholds"]
__version__ = "0.1.0"

# Demo module
try:
    from . import demo
    __all__.append("demo")
except ImportError:
    pass
