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

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    # Scheduling
    candle_refresh_delay_seconds: int = 60

    # Trading
    # Account balance in USD — set ACCOUNT_BALANCE env var to match your actual
    # Binance demo/live balance. Binance demo accounts start with 10 000 USDT.
    account_balance: float = 10000.0

    # Telegram (optional -- system works without these configured)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── Crypto Futures (Binance) ─────────────────────────────────────────────
    # Public market data endpoints require no API key.
    # Keys are only needed for order execution.
    binance_futures_api_key: str = ""
    binance_futures_api_secret: str = ""

    # Use Binance Futures TESTNET (fake money — safe for testing).
    # Set to false only when you are ready for real trading.
    binance_testnet: bool = True

    # Leverage applied to every futures position (default 5x — conservative).
    binance_leverage: int = 5

    # Comma-separated list of crypto futures symbols to track
    crypto_symbols: str = "BTCUSDT,ETHUSDT"

    # Master switch — set CRYPTO_ENABLED=true to activate the crypto pipeline.
    crypto_enabled: bool = False

    # ── Position sizing (confidence-tiered) ──────────────────────────────────
    trade_size_high_confidence: float = 75.0   # confidence % threshold → high tier
    trade_size_mid_confidence:  float = 65.0   # confidence % threshold → mid tier
    trade_size_low_confidence:  float = 50.0   # confidence % threshold → mid-low tier
    trade_size_high:    float = 1000.0          # notional USDT when conf >= high
    trade_size_mid:     float = 500.0           # notional USDT when conf >= mid
    trade_size_mid_low: float = 250.0           # notional USDT when conf >= low
    trade_size_low:     float = 100.0           # notional USDT when conf < low

    # ── News ─────────────────────────────────────────────────────────────────
    # Optional: register free at cryptopanic.com to get a token.
    # Without it, news is sourced from Binance announcements RSS only.
    cryptopanic_api_key: str = ""

    # ── Claude Autonomous Trading Agent ──────────────────────────────────────
    anthropic_api_key: str = ""
    claude_agent_enabled: bool = False
    claude_agent_risk_pct: float = 0.05        # 5% risk per trade
    claude_agent_daily_loss_limit: float = 0.10  # 10% daily loss limit

    @property
    def binance_base_url(self) -> str:
        """Binance Futures REST base URL — demo or mainnet."""
        if self.binance_testnet:
            return "https://demo-fapi.binance.com"
        return "https://fapi.binance.com"

    @property
    def binance_order_execution_enabled(self) -> bool:
        """True when both API key and secret are configured."""
        return bool(self.binance_futures_api_key.strip() and self.binance_futures_api_secret.strip())

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
        url = self.database_url.strip()
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        self.database_url = url
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()
