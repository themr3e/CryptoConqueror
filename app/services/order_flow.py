"""Order flow analysis using Binance Futures aggTrades.

Replicates the core insight of Footprint IQ — tracking who is aggressive
in the market (buyers lifting offers vs sellers hitting bids).

Key metrics:
    delta       — buy volume minus sell volume in USDT
    CVD trend   — direction of cumulative delta over recent periods
    buy_ratio   — fraction of total volume from aggressive buyers
    pressure    — composite signal: BULLISH / BEARISH / NEUTRAL
    large_trades — count of trades > $50k USDT (institutional footprint)

Always fetches from MAINNET even when the bot is on testnet — testnet has
no real order flow; real market data is what matters for decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from loguru import logger


# Always mainnet — order flow is market data, not execution
_AGG_TRADES_URL = "https://fapi.binance.com/fapi/v1/aggTrades"

# A "large" trade threshold in USDT
_LARGE_TRADE_USDT = 50_000.0


@dataclass
class OrderFlowSummary:
    symbol: str
    delta_30m: float        # buy vol − sell vol over last 30 min (USDT)
    delta_60m: float        # buy vol − sell vol over last 60 min (USDT)
    buy_ratio: float        # 0.0–1.0 fraction of aggressive buys
    sell_ratio: float       # 0.0–1.0 fraction of aggressive sells
    total_volume_usdt: float
    cvd_trend: str          # "rising" | "falling" | "flat"
    pressure: str           # "BULLISH" | "BEARISH" | "NEUTRAL"
    large_buys: int         # trades > $50k from buyers
    large_sells: int        # trades > $50k from sellers

    def to_context_string(self) -> str:
        emoji = "🟢" if self.pressure == "BULLISH" else ("🔴" if self.pressure == "BEARISH" else "⚪")
        d30 = f"{'+'if self.delta_30m>=0 else ''}{self.delta_30m:,.0f}"
        d60 = f"{'+'if self.delta_60m>=0 else ''}{self.delta_60m:,.0f}"
        return (
            f"{emoji} [ORDER FLOW]\n"
            f"  Delta 30m: {d30} USDT | Delta 60m: {d60} USDT\n"
            f"  Buy {self.buy_ratio*100:.0f}% / Sell {self.sell_ratio*100:.0f}% "
            f"| CVD: {self.cvd_trend.upper()}\n"
            f"  Large trades: {self.large_buys} buys / {self.large_sells} sells (>${_LARGE_TRADE_USDT/1000:.0f}k)\n"
            f"  Pressure: {self.pressure}"
        )


class OrderFlowAnalyzer:
    """Fetches and analyzes order flow for a Binance Futures symbol."""

    async def fetch(self, symbol: str, lookback_minutes: int = 90) -> OrderFlowSummary | None:
        """Return order flow summary for the last ``lookback_minutes`` minutes.

        Returns None if data cannot be fetched (network error, symbol not found).
        """
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = now_ms - lookback_minutes * 60 * 1000

        try:
            trades = await self._fetch_agg_trades(symbol, start_ms, now_ms)
        except Exception:
            logger.opt(exception=True).warning("[OrderFlow] Failed to fetch aggTrades for {}", symbol)
            return None

        if not trades:
            logger.debug("[OrderFlow] No aggTrades returned for {}", symbol)
            return None

        return self._compute(symbol, trades, now_ms)

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _fetch_agg_trades(
        self, symbol: str, start_ms: int, end_ms: int
    ) -> list[dict]:
        """Fetch up to 2 000 aggregated trades from Binance Futures mainnet."""
        trades: list[dict] = []
        params: dict = {
            "symbol": symbol,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            for _ in range(2):          # max 2 pages = 2 000 trades
                resp = await client.get(_AGG_TRADES_URL, params=params)
                if resp.status_code != 200:
                    logger.debug(
                        "[OrderFlow] aggTrades HTTP {} for {}", resp.status_code, symbol
                    )
                    break
                batch: list[dict] = resp.json()
                if not batch:
                    break
                trades.extend(batch)
                if len(batch) < 1000:
                    break
                # Advance start to avoid duplicating the last trade
                params["startTime"] = int(batch[-1]["T"]) + 1

        return trades

    def _compute(
        self, symbol: str, trades: list[dict], now_ms: int
    ) -> OrderFlowSummary:
        """Bucket trades and compute all order flow metrics."""
        cutoff_30m = now_ms - 30 * 60 * 1000
        cutoff_60m = now_ms - 60 * 60 * 1000

        # 15-min buckets for CVD trend (up to 6 buckets in 90 min window)
        bucket_ms = 15 * 60 * 1000
        buckets: dict[int, float] = {}

        buy_vol_total = 0.0
        sell_vol_total = 0.0
        delta_30m = 0.0
        delta_60m = 0.0
        large_buys = 0
        large_sells = 0

        for trade in trades:
            price = float(trade["p"])
            qty   = float(trade["q"])
            value = price * qty
            t     = int(trade["T"])
            # m=True → buyer is the maker → this is an aggressive SELL
            is_sell = bool(trade["m"])

            if is_sell:
                sell_vol_total += value
                delta = -value
            else:
                buy_vol_total += value
                delta = +value

            if t >= cutoff_30m:
                delta_30m += delta
            if t >= cutoff_60m:
                delta_60m += delta

            bucket_key = t // bucket_ms
            buckets[bucket_key] = buckets.get(bucket_key, 0.0) + delta

            if value >= _LARGE_TRADE_USDT:
                if is_sell:
                    large_sells += 1
                else:
                    large_buys += 1

        total_vol = buy_vol_total + sell_vol_total
        buy_ratio  = buy_vol_total / total_vol if total_vol > 0 else 0.5
        sell_ratio = 1.0 - buy_ratio

        # CVD trend: build running CVD from oldest bucket to newest,
        # compare first bucket vs last bucket
        cvd_trend = "flat"
        if len(buckets) >= 3:
            sorted_deltas = [v for _, v in sorted(buckets.items())]
            running = 0.0
            cvd_series = []
            for d in sorted_deltas:
                running += d
                cvd_series.append(running)
            first, last = cvd_series[0], cvd_series[-1]
            swing = abs(last - first)
            # Require at least 1% of total vol to call it a trend
            threshold = total_vol * 0.01
            if swing >= threshold:
                cvd_trend = "rising" if last > first else "falling"

        # Composite pressure signal
        if buy_ratio >= 0.58 and delta_30m > 0 and cvd_trend != "falling":
            pressure = "BULLISH"
        elif sell_ratio >= 0.58 and delta_30m < 0 and cvd_trend != "rising":
            pressure = "BEARISH"
        else:
            pressure = "NEUTRAL"

        return OrderFlowSummary(
            symbol=symbol,
            delta_30m=delta_30m,
            delta_60m=delta_60m,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
            total_volume_usdt=total_vol,
            cvd_trend=cvd_trend,
            pressure=pressure,
            large_buys=large_buys,
            large_sells=large_sells,
        )
