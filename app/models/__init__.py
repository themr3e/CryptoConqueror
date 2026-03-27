"""ORM model imports for Alembic autogenerate discovery."""

from app.models.base import Base  # noqa: F401
from app.models.candle import Candle  # noqa: F401
from app.models.strategy import Strategy  # noqa: F401
from app.models.signal import Signal  # noqa: F401
from app.models.outcome import Outcome  # noqa: F401
from app.models.backtest_result import BacktestResult  # noqa: F401
from app.models.strategy_performance import StrategyPerformance  # noqa: F401
from app.models.optimized_params import OptimizedParams  # noqa: F401
from app.models.crypto_fee_config import CryptoFeeConfig  # noqa: F401
from app.models.trade_order import TradeOrder  # noqa: F401
from app.models.claude_decision import ClaudeDecision  # noqa: F401
