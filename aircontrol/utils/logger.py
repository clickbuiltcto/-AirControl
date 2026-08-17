"""
utils/logger.py
================
Tiny, dependency-free logging helper so every module reports through a
consistent, timestamped format instead of ad-hoc print() calls.
"""

from __future__ import annotations

import logging
import sys


_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger with consistent formatting."""
    global _CONFIGURED
    if not _CONFIGURED:
        # A stock Windows console (cmd.exe) defaults to a legacy codepage
        # (e.g. cp437), not UTF-8. Any non-ASCII character in a log message
        # (e.g. an em dash) then raises UnicodeEncodeError inside logging's
        # emit(), which silently drops the message instead of printing it.
        # Reconfigure stdout to UTF-8 (with 'replace' as a last-resort
        # fallback) so log output is never lost because of this.
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)-7s %(name)-20s | %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
