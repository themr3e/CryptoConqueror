"""Gold-specific market intelligence service.

Identifies active trading sessions, applies London/NY overlap confidence
boost, attaches session metadata to signals, and monitors gold-DXY
correlation as informational enrichment.

Float math internally; Decimal(str(round(x, 2))) at CandidateSignal boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.candle import Candle
from app.strategies.base import CandidateSignal
from app.strategies.helpers.session_filter import get_active_sessions

OVERLAP_CONFIDENCE_BOOST: int = 5
DXY_SYMBOL: str = "DXY"
DXY_CORRELATION_WINDOW: int = 30
DXY_DIVERGENCE_THRESHOLD: float = -0.3

_VOLATILITY_PROFILES: dict[str, str] = {
    "asian": "Low volatility (typically 40-60% of London)",
    "london": "High volatility (London open drives significant price movement)",
    "new_york": "High volatility (NY open adds liquidity and direction)",
    "overlap": "Very high volatility (peak liquidity, largest moves)",
}


@dataclass(frozen=True)
class SessionInfo:
    active_sessions: list[str]
    is_overlap: bool
    timestamp: datetime


@dataclass(frozen=True)
class DXYCorrelation:
    correlation: float | None
    is_divergent: bool
    available: bool
    message: str


class GoldIntelligence:
    """Gold-specific market intelligence for the signal pipeline."""

    def get_session_info(self, timestamp: datetime | None = None) -> SessionInfo:
        """Return active trading sessions for timestamp."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        active = get_active_sessions(timestamp)
        is_overlap = "overlap" in active

        return SessionInfo(
            active_sessions=active,
            is_overlap=is_overlap,
            timestamp=timestamp,
        )

    def enrich(
        self,
        candidates: list[CandidateSignal],
        dxy_info: DXYCorrelation | None = None,
    ) -> list[CandidateSignal]:
        """Enrich candidate signals with session metadata and optional boosts."""
        enriched: list[CandidateSignal] = []

        now = datetime.now(timezone.utc)
        for candidate in candidates:
            session_info = self.get_session_info(now)

            if session_info.is_overlap:
                primary_session = "overlap"
            elif session_info.active_sessions:
                primary_session = session_info.active_sessions[0]
            else:
                primary_session = "off_hours"

            updates: dict = {"session": primary_session}
            reasoning = candidate.reasoning

            if session_info.is_overlap:
                boosted = float(candidate.confidence) + OVERLAP_CONFIDENCE_BOOST
                new_confidence = Decimal(str(round(min(boosted, 100.0), 2)))
                updates["confidence"] = new_confidence
                reasoning += " | London/NY overlap: +5 confidence"
                logger.info(
                    "Overlap boost applied | strategy={} old={} new={}",
                    candidate.strategy_name,
                    candidate.confidence,
                    new_confidence,
                )

            if dxy_info is not None and dxy_info.is_divergent:
                corr_str = (
                    f"{dxy_info.correlation:.2f}"
                    if dxy_info.correlation is not None
                    else "N/A"
                )
                reasoning += f" | DXY divergence detected (corr={corr_str})"

            updates["reasoning"] = reasoning
            enriched.append(candidate.model_copy(update=updates))

        return enriched

    async def get_dxy_correlation(self, session: AsyncSession) -> DXYCorrelation:
        """Compute rolling Pearson correlation between XAUUSD and DXY."""
        unavailable = DXYCorrelation(
            correlation=None,
            is_divergent=False,
            available=False,
            message="DXY data unavailable",
        )

        try:
            dxy_stmt = (
                select(Candle)
                .where(Candle.symbol == DXY_SYMBOL, Candle.timeframe == "D1")
                .order_by(Candle.timestamp.desc())
                .limit(60)
            )
            dxy_result = await session.execute(dxy_stmt)
            dxy_candles = dxy_result.scalars().all()

            if len(dxy_candles) < DXY_CORRELATION_WINDOW + 5:
                return unavailable

            gold_stmt = (
                select(Candle)
                .where(Candle.symbol == "XAUUSD", Candle.timeframe == "D1")
                .order_by(Candle.timestamp.desc())
                .limit(60)
            )
            gold_result = await session.execute(gold_stmt)
            gold_candles = gold_result.scalars().all()

            if len(gold_candles) < DXY_CORRELATION_WINDOW + 5:
                return unavailable

            dxy_df = pd.DataFrame(
                [{"date": c.timestamp.date(), "dxy_close": float(c.close)} for c in dxy_candles]
            ).sort_values("date")

            gold_df = pd.DataFrame(
                [{"date": c.timestamp.date(), "gold_close": float(c.close)} for c in gold_candles]
            ).sort_values("date")

            merged = pd.merge(dxy_df, gold_df, on="date", how="inner")

            if len(merged) < DXY_CORRELATION_WINDOW + 5:
                return unavailable

            rolling_corr = merged["gold_close"].rolling(DXY_CORRELATION_WINDOW).corr(merged["dxy_close"])
            valid_corr = rolling_corr.dropna()
            latest_corr = float(valid_corr.iloc[-1]) if not valid_corr.empty else None

            if latest_corr is None:
                return unavailable

            is_divergent = latest_corr > DXY_DIVERGENCE_THRESHOLD
            msg = (
                f"Gold-DXY 30-period correlation: {latest_corr:.3f}"
                f" ({'DIVERGENT' if is_divergent else 'normal inverse'})"
            )
            logger.info(msg)

            return DXYCorrelation(
                correlation=latest_corr,
                is_divergent=is_divergent,
                available=True,
                message=msg,
            )

        except Exception:
            logger.opt(exception=True).warning("DXY correlation computation failed")
            return unavailable

    def get_session_volatility_profile(self, session_name: str) -> str:
        """Return a qualitative volatility description for session_name."""
        return _VOLATILITY_PROFILES.get(session_name, f"Unknown session '{session_name}'")
