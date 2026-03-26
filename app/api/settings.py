"""Settings API — read and persist API keys / configuration via the dashboard."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/settings", tags=["settings"])

_ENV_PATH = Path(".env")


# ---------------------------------------------------------------------------
# .env file helpers
# ---------------------------------------------------------------------------

def _read_env_file() -> dict[str, str]:
    """Parse the .env file into a key→value dict (ignores comments/blanks)."""
    result: dict[str, str] = {}
    if not _ENV_PATH.exists():
        return result
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            value = value.split("#")[0].strip()   # strip trailing inline comments
            result[key.strip()] = value
    return result


def _write_env_value(key: str, value: str) -> None:
    """Upsert a single key=value line in the .env file.

    If the key already exists (with or without whitespace around =) the line is
    replaced in-place.  Otherwise the key is appended at the end.
    """
    if _ENV_PATH.exists():
        content = _ENV_PATH.read_text(encoding="utf-8")
    else:
        content = ""

    pattern = re.compile(rf"^({re.escape(key)}\s*=).*$", re.MULTILINE)
    if pattern.search(content):
        content = pattern.sub(rf"\g<1>{value}", content)
    else:
        content = content.rstrip("\n") + f"\n{key}={value}\n"

    _ENV_PATH.write_text(content, encoding="utf-8")


def _mask(value: str) -> str:
    """Return a masked representation — shows first 4 chars then asterisks."""
    if not value:
        return ""
    visible = value[:4]
    stars = "*" * min(len(value) - 4, 12)
    return visible + stars


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SettingsResponse(BaseModel):
    # Binance Futures
    binance_testnet: bool
    binance_futures_api_key: str        # masked if set
    binance_futures_api_secret: str     # masked if set
    binance_futures_api_key_set: bool
    binance_futures_api_secret_set: bool
    binance_leverage: int
    # Crypto
    crypto_enabled: bool
    crypto_symbols: str
    # Trading
    account_balance: float
    # Telegram
    telegram_bot_token: str             # masked if set
    telegram_chat_id: str
    telegram_bot_token_set: bool
    # Gold / XAUUSD
    twelve_data_api_key: str            # masked if set
    twelve_data_api_key_set: bool


class SettingsUpdate(BaseModel):
    # All fields are optional — only provided fields are updated
    binance_testnet: bool | None = None
    binance_futures_api_key: str | None = None      # empty string = keep current
    binance_futures_api_secret: str | None = None   # empty string = keep current
    binance_leverage: int | None = None
    crypto_enabled: bool | None = None
    crypto_symbols: str | None = None
    account_balance: float | None = None
    telegram_bot_token: str | None = None           # empty string = keep current
    telegram_chat_id: str | None = None
    twelve_data_api_key: str | None = None          # empty string = keep current


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/", response_model=SettingsResponse)
async def get_settings_view():
    """Return current settings read from .env (secrets are masked)."""
    env = _read_env_file()

    raw_binance_key    = env.get("BINANCE_FUTURES_API_KEY", "")
    raw_binance_secret = env.get("BINANCE_FUTURES_API_SECRET", "")
    raw_tg_token       = env.get("TELEGRAM_BOT_TOKEN", "")
    raw_twelve         = env.get("TWELVE_DATA_API_KEY", "")

    return SettingsResponse(
        binance_testnet=env.get("BINANCE_TESTNET", "true").lower() not in ("false", "0", "no"),
        binance_futures_api_key=_mask(raw_binance_key),
        binance_futures_api_secret=_mask(raw_binance_secret),
        binance_futures_api_key_set=bool(raw_binance_key),
        binance_futures_api_secret_set=bool(raw_binance_secret),
        binance_leverage=int(env.get("BINANCE_LEVERAGE", "5")),
        crypto_enabled=env.get("CRYPTO_ENABLED", "false").lower() in ("true", "1", "yes"),
        crypto_symbols=env.get("CRYPTO_SYMBOLS", "BTCUSDT,ETHUSDT"),
        account_balance=float(env.get("ACCOUNT_BALANCE", "100000.0")),
        telegram_bot_token=_mask(raw_tg_token),
        telegram_chat_id=env.get("TELEGRAM_CHAT_ID", ""),
        telegram_bot_token_set=bool(raw_tg_token),
        twelve_data_api_key=_mask(raw_twelve),
        twelve_data_api_key_set=bool(raw_twelve),
    )


@router.post("/")
async def update_settings(payload: SettingsUpdate):
    """Persist updated settings to .env.

    Empty-string values for secret fields are treated as "no change".
    Changes take effect after application restart.
    """
    _KEY_MAP: dict[str, tuple[str, Any]] = {
        "binance_testnet":          ("BINANCE_TESTNET",          lambda v: "true" if v else "false"),
        "binance_futures_api_key":  ("BINANCE_FUTURES_API_KEY",  str),
        "binance_futures_api_secret": ("BINANCE_FUTURES_API_SECRET", str),
        "binance_leverage":         ("BINANCE_LEVERAGE",         str),
        "crypto_enabled":           ("CRYPTO_ENABLED",           lambda v: "true" if v else "false"),
        "crypto_symbols":           ("CRYPTO_SYMBOLS",           str),
        "account_balance":          ("ACCOUNT_BALANCE",          str),
        "telegram_bot_token":       ("TELEGRAM_BOT_TOKEN",       str),
        "telegram_chat_id":         ("TELEGRAM_CHAT_ID",         str),
        "twelve_data_api_key":      ("TWELVE_DATA_API_KEY",      str),
    }

    # Secret fields — skip update when the user leaves them blank
    _SECRET_FIELDS = {
        "binance_futures_api_key",
        "binance_futures_api_secret",
        "telegram_bot_token",
        "twelve_data_api_key",
    }

    updated: list[str] = []
    data = payload.model_dump(exclude_none=True)

    for field, value in data.items():
        if field not in _KEY_MAP:
            continue
        # Skip empty strings for secret fields (user left them blank intentionally)
        if field in _SECRET_FIELDS and isinstance(value, str) and not value.strip():
            continue
        env_key, transformer = _KEY_MAP[field]
        _write_env_value(env_key, transformer(value))
        updated.append(env_key)

    return {
        "updated": updated,
        "restart_required": bool(updated),
        "message": (
            f"Saved {len(updated)} setting(s). Restart the app to apply changes."
            if updated else "No changes to save."
        ),
    }
