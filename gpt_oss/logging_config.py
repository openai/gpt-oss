"""
Logging configuration for gpt-oss package.
Provides structured logging with appropriate levels and formatting.
"""

import logging
import sys
from typing import Any, Dict, Optional

import structlog

# Configure structlog processors
structlog.configure(
    processors=[
        # Add log level to log entries
        structlog.stdlib.add_log_level,
        # Add a timestamp in ISO format
        structlog.processors.TimeStamper(fmt="iso"),
        # If the "stack_info" key in the event dict is true, remove it and
        # render the current stack trace in the "stack" key.
        structlog.processors.StackInfoRenderer(),
        # Format the exception only once, even if there are multiple loggers
        structlog.processors.format_exc_info,
        # Render in JSON for production, or pretty print for development
        (
            structlog.dev.ConsoleRenderer()
            if sys.stderr.isatty()
            else structlog.processors.JSONRenderer()
        ),
    ],
    # Our `wrapper_class` is used for passing metadata when binding
    wrapper_class=structlog.stdlib.BoundLogger,
    # `logger_factory` is used to create wrapped loggers that are used for OUTPUT.
    logger_factory=structlog.stdlib.LoggerFactory(),
    # Cache the logger for better performance
    cache_logger_on_first_use=True,
)


def configure_logging(
    level: str = "INFO", format_json: bool = False, component: Optional[str] = None
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_json: Whether to format logs as JSON
        component: Optional component name to include in logs
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper()),
    )

    # Update structlog processors based on configuration
    processors = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if component:
        processors.append(structlog.processors.CallsiteParameterAdder())

    if format_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(processors=processors)


def get_logger(name: str, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger with optional initial values.

    Args:
        name: Logger name (typically __name__)
        **initial_values: Key-value pairs to include in all log messages

    Returns:
        Structured logger instance
    """
    logger = structlog.get_logger(name)
    if initial_values:
        logger = logger.bind(**initial_values)
    return logger
