"""Footprint / order-flow analyzer.

Pure functions that mirror the concept from *Footprint IQ Pro* by TradingIQ
(Pine Script) — adapted for Binance Futures where each Binance aggTrade
already carries ``isBuyerMaker``. Classification rule:

    - ``isBuyerMaker == True``  → taker was SELLER  → aggressive SELL
    - ``isBuyerMaker == False`` → taker was BUYER   → aggressive BUY

Features ported:
    1. Volume profile on ATR-derived tick levels
    2. Point of Control (POC)        — level with max total volume
    3. Value Area (70%)              — smallest band around POC
    4. Per-level imbalance (≥ 70%)   — |delta / total| per level
    5. Stacked imbalance             — N consecutive imbalances same side
    6. Cumulative delta divergence   — price HH + cum_delta LH (and inverse)

Exports:
    Level                      -- single volume-profile level
    FootprintProfile           -- full profile for a symbol + minute
    build_profile              -- trades -> profile
    compute_poc_va             -- profile -> (POC, VAH, VAL)
    find_imbalances            -- profile -> list of (level, side)
    count_stacked              -- imbalances -> (count, side)
    cumulative_delta_divergence -- list[FootprintBar] -> divergence flag

The analyzer never touches the DB; the rollup caller persists results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Iterable, Sequence


# Typed in-memory view of one aggTrade / synthetic 1s trade.
@dataclass(frozen=True)
class Trade:
    price: float
    qty: float
    is_buyer_maker: bool  # see module docstring

    @property
    def signed_qty(self) -> float:
        """Qty signed by aggressor side: + for aggressive BUY, − for SELL."""
        return -self.qty if self.is_buyer_maker else self.qty


@dataclass
class Level:
    """A single price bucket in the volume profile."""
    price: float          # lower bound of the bucket
    buy_vol: float = 0.0
    sell_vol: float = 0.0

    @property
    def delta(self) -> float:
        return self.buy_vol - self.sell_vol

    @property
    def total_vol(self) -> float:
        return self.buy_vol + self.sell_vol

    @property
    def delta_pct(self) -> float:
        """Delta as fraction of total volume at this level (-1 to +1)."""
        tv = self.total_vol
        return self.delta / tv if tv > 0 else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "price": round(self.price, 8),
            "buy_vol": round(self.buy_vol, 8),
            "sell_vol": round(self.sell_vol, 8),
            "delta": round(self.delta, 8),
        }


@dataclass
class FootprintProfile:
    """Full footprint state for one (symbol, minute)."""
    levels: list[Level]
    tick_size: float
    high: float
    low: float
    close: float
    poc: float = 0.0
    vah: float = 0.0  # value-area high
    val: float = 0.0  # value-area low
    stacked_imb_count: int = 0
    stacked_imb_side: str | None = None  # "BUY" | "SELL" | None
    imbalances: list[tuple[int, str]] = field(default_factory=list)

    @property
    def total_vol(self) -> float:
        return sum(lv.total_vol for lv in self.levels)

    @property
    def buy_vol(self) -> float:
        return sum(lv.buy_vol for lv in self.levels)

    @property
    def sell_vol(self) -> float:
        return sum(lv.sell_vol for lv in self.levels)

    @property
    def delta(self) -> float:
        return self.buy_vol - self.sell_vol


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def build_profile(
    trades: Sequence[Trade],
    tick_size: float,
) -> FootprintProfile | None:
    """Build a volume profile from a minute of trades.

    Returns None if no trades. ``tick_size`` defines bucket width.
    """
    if not trades or tick_size <= 0:
        return None

    high = max(t.price for t in trades)
    low = min(t.price for t in trades)
    close = trades[-1].price

    # Build level buckets from low → high aligned to tick_size multiples.
    n_levels = max(1, int((high - low) / tick_size) + 1)
    base = (low // tick_size) * tick_size
    levels = [Level(price=base + i * tick_size) for i in range(n_levels + 1)]

    for t in trades:
        idx = min(
            len(levels) - 1,
            max(0, int((t.price - base) / tick_size)),
        )
        if t.is_buyer_maker:
            levels[idx].sell_vol += t.qty
        else:
            levels[idx].buy_vol += t.qty

    return FootprintProfile(
        levels=levels,
        tick_size=tick_size,
        high=high,
        low=low,
        close=close,
    )


def compute_poc_va(
    profile: FootprintProfile,
    va_pct: float = 0.70,
) -> tuple[float, float, float]:
    """Mutate ``profile`` with POC, VAH, VAL and also return them.

    Mirrors the Pine Script: POC = max-total-volume level; expand
    symmetrically from POC until cumulative volume >= ``va_pct`` × total.
    """
    if not profile.levels:
        return 0.0, 0.0, 0.0

    totals = [lv.total_vol for lv in profile.levels]
    total_sum = sum(totals)
    if total_sum <= 0:
        return 0.0, 0.0, 0.0

    poc_idx = max(range(len(totals)), key=totals.__getitem__)
    poc_price = profile.levels[poc_idx].price

    accum = totals[poc_idx]
    lo = hi = poc_idx
    target = va_pct * total_sum

    while accum < target and (lo > 0 or hi < len(totals) - 1):
        lo_vol = totals[lo - 1] if lo > 0 else -1.0
        hi_vol = totals[hi + 1] if hi < len(totals) - 1 else -1.0
        if hi_vol >= lo_vol:
            hi += 1
            accum += totals[hi]
        else:
            lo -= 1
            accum += totals[lo]

    val = profile.levels[lo].price
    vah = profile.levels[hi].price + profile.tick_size

    profile.poc = poc_price
    profile.vah = vah
    profile.val = val
    return poc_price, vah, val


def find_imbalances(
    profile: FootprintProfile,
    threshold: float = 0.70,
) -> list[tuple[int, str]]:
    """Return [(level_index, "BUY"|"SELL"), ...] for levels ≥ threshold |δ%|.

    Also mutates ``profile.imbalances``.
    """
    out: list[tuple[int, str]] = []
    for i, lv in enumerate(profile.levels):
        if lv.total_vol <= 0:
            continue
        dp = lv.delta_pct
        if abs(dp) >= threshold:
            side = "BUY" if dp > 0 else "SELL"
            out.append((i, side))
    profile.imbalances = out
    return out


def count_stacked(
    imbalances: list[tuple[int, str]],
    min_stack: int = 3,
) -> tuple[int, str | None]:
    """Scan ``imbalances`` (sorted ascending by level index) for the longest
    consecutive run of same-side imbalances on adjacent price levels.

    Returns ``(run_length, side)`` — ``(0, None)`` if no run ≥ ``min_stack``.
    """
    if not imbalances:
        return 0, None

    imbalances = sorted(imbalances, key=lambda x: x[0])
    best_len = 0
    best_side: str | None = None
    cur_len = 1
    cur_side = imbalances[0][1]
    cur_last_idx = imbalances[0][0]

    for idx, side in imbalances[1:]:
        if side == cur_side and idx == cur_last_idx + 1:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len, best_side = cur_len, cur_side
            cur_len = 1
            cur_side = side
        cur_last_idx = idx

    if cur_len > best_len:
        best_len, best_side = cur_len, cur_side

    if best_len >= min_stack:
        return best_len, best_side
    return 0, None


def cumulative_delta_divergence(
    closes: Sequence[float],
    cum_deltas: Sequence[float],
    lookback: int = 20,
) -> str | None:
    """Detect bullish/bearish cumulative-delta divergence.

    - "bearish": price makes a higher high while cum_delta makes a lower high
      → buying exhaustion, expect reversal down.
    - "bullish": price makes a lower low while cum_delta makes a higher low
      → selling exhaustion, expect reversal up.

    Returns None when no divergence is detected.
    """
    n = min(len(closes), len(cum_deltas))
    if n < lookback or lookback < 4:
        return None

    px = list(closes[-lookback:])
    cd = list(cum_deltas[-lookback:])
    half = lookback // 2

    px_prev_high = max(px[:half])
    px_cur_high = max(px[half:])
    cd_prev_high = max(cd[:half])
    cd_cur_high = max(cd[half:])
    if px_cur_high > px_prev_high and cd_cur_high < cd_prev_high:
        return "bearish"

    px_prev_low = min(px[:half])
    px_cur_low = min(px[half:])
    cd_prev_low = min(cd[:half])
    cd_cur_low = min(cd[half:])
    if px_cur_low < px_prev_low and cd_cur_low > cd_prev_low:
        return "bullish"

    return None


# ---------------------------------------------------------------------------
# Helper: convert raw_trade ORM rows into Trade dataclass objects
# ---------------------------------------------------------------------------

def rows_to_trades(rows: Iterable) -> list[Trade]:
    """Adapter for raw_trades query results (SQLAlchemy rows / ORM)."""
    out: list[Trade] = []
    for r in rows:
        price = float(r.price)
        qty = float(r.qty)
        if qty <= 0:
            continue
        out.append(Trade(price=price, qty=qty, is_buyer_maker=bool(r.is_buyer_maker)))
    return out


def derive_tick_size(high: float, low: float, fallback: float) -> float:
    """Pick a sensible tick_size when no ATR is available.

    The Pine indicator uses ATR/4; for a 1-minute window we approximate
    with (high − low) / 20 so we get ~20 buckets per minute, but never
    smaller than ``fallback`` (typically the symbol's Binance tick_size).
    """
    span = max(high - low, 0.0)
    est = span / 20.0 if span > 0 else fallback
    return max(est, fallback)
