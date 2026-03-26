"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import model_validator

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Database
    database_url: str

    # External APIs
    # Twelve Data is only needed for XAUUSD (Gold) candle data.
    # Leave blank if you are running crypto-only.
    twelve_data_api_key: str = ""

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    # Scheduling
    candle_refresh_delay_seconds: int = 60

    # Trading
    # Prop firm account balance in USD (sourced from ACCOUNT_BALANCE env var)
    account_balance: float = 100000.0

    # Telegram (optional -- system works without these configured)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── Crypto Futures (Binance) ─────────────────────────────────────────────
    # Public market data endpoints require no API key.
    # Keys are only needed for order execution (future feature).
    binance_futures_api_key: str = ""
    binance_futures_api_secret: str = ""

    # Comma-separated list of crypto futures symbols to track
    crypto_symbols: str = "BTCUSDT,ETHUSDT"

    # Master switch — set CRYPTO_ENABLED=true to activate the crypto pipeline.
    # Defaults to False so existing XAUUSD behaviour is unaffected on deploy.
    crypto_enabled: bool = False

    @property
    def xauusd_enabled(self) -> bool:
        """True when a Twelve Data key is configured (XAUUSD pipeline active)."""
        return bool(self.twelve_data_api_key.strip())

    @property
    def crypto_symbol_list(self) -> list[str]:
        """Return crypto_symbols as a list of stripped uppercase strings."""
        return [s.strip().upper() for s in self.crypto_symbols.split(",") if s.strip()]

    @model_validator(mode="after")
    def normalize_database_url(self) -> "Settings":
        """Ensure DATABASE_URL uses the asyncpg driver.

        Railway and other providers supply postgresql:// but SQLAlchemy async
        requires postgresql+asyncpg://.
        """
        url = self.database_url
        if url.startswith("postgresql://"):
            self.database_url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            self.database_url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()
