"""Logging configuration using loguru.

Call ``setup_logging()`` once at application startup to configure the sink,
format, and log level based on application settings.

Exports:
    setup_logging -- configure loguru for the application
"""

from __future__ import annotations

import sys

from loguru import logger

from app.config import get_settings


def setup_logging() -> None:
    """Configure loguru logging based on application settings.

    Reads ``LOG_LEVEL`` and ``LOG_JSON`` from settings:
        - ``LOG_LEVEL`` : e.g. ``"INFO"``, ``"DEBUG"``
        - ``LOG_JSON``  : if True, emits structured JSON logs (for cloud/Railway)

    Removes the default loguru sink and adds a new configured one.
    """
    settings = get_settings()
    log_level = settings.log_level.upper()
    log_json = settings.log_json

    # Remove default sink
    logger.remove()

    if log_json:
        logger.add(
            sys.stdout,
            level=log_level,
            serialize=True,  # JSON output
            backtrace=False,
            diagnose=False,
        )
    else:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stdout,
            level=log_level,
            format=fmt,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    logger.info("Logging configured: level={}, json={}", log_level, log_json)
