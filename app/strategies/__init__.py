"""Strategy registry — import all concrete strategies so they self-register."""

import importlib.util
from pathlib import Path

from app.strategies.crypto_momentum import CryptoMomentumStrategy
from app.strategies.crypto_breakout import CryptoBreakoutStrategy
from app.strategies.crypto_slc import CryptoSLCStrategy

__all__ = [
    "CryptoMomentumStrategy",
    "CryptoBreakoutStrategy",
    "CryptoSLCStrategy",
]

# ── Auto-load strategies that passed the walk-forward blind backtest ──────────
_AUTO_DIR = Path(__file__).parent / "auto"
if _AUTO_DIR.exists():
    for _f in sorted(_AUTO_DIR.glob("*.py")):
        if _f.name.startswith("_") or _f.name == "__init__.py":
            continue   # skip temp/pending files
        try:
            _spec   = importlib.util.spec_from_file_location(f"auto_{_f.stem}", _f)
            _module = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_module)
            # BaseStrategy subclasses self-register on import via __init_subclass__
        except Exception as _e:
            import logging
            logging.getLogger(__name__).warning("Failed to load auto strategy %s: %s", _f.name, _e)
