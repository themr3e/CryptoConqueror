"""Smart Money Concepts analyzer using the smartmoneyconcepts library.

Wraps pip install smartmoneyconcepts to provide FVG, Order Block, and BOS/CHoCH
context for trading decisions. Used as a "second opinion" alongside the
custom crypto_slc.py implementation.

Gracefully falls back with empty results if the library is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger


@dataclass
class SMCContext:
    """Snapshot of Smart Money context at the latest candle."""

    # Market structure
    structure: str = "sideways"      # "uptrend" | "downtrend" | "sideways"
    last_bos: str | None = None      # "bullish" | "bearish" — most recent BOS
    last_choch: str | None = None    # "bullish" | "bearish" — most recent CHoCH

    # Nearest unfilled zones (price ordered by distance from current price)
    nearest_fvg_bull: dict | None = None   # {"top": float, "bottom": float}
    nearest_fvg_bear: dict | None = None

    nearest_ob_bull: dict | None = None    # {"top": float, "bottom": float}
    nearest_ob_bear: dict | None = None

    # Volume context
    volume_above_avg: bool = False    # current bar volume > 20-bar SMA

    # Summary for Claude prompt injection
    summary: str = ""

    # Raw errors (non-blocking)
    errors: list[str] = field(default_factory=list)


class SMCAnalyzer:
    """Compute Smart Money Concepts context from H1 OHLCV candles."""

    def analyze(self, candles: pd.DataFrame) -> SMCContext:
        """Run full SMC analysis on the candle DataFrame.

        Args:
            candles: DataFrame with columns open, high, low, close, volume.
                     Must have at least 100 rows for meaningful output.

        Returns:
            SMCContext with current market structure and nearest zones.
        """
        ctx = SMCContext()

        if len(candles) < 50:
            ctx.errors.append("Insufficient candles (need ≥ 50)")
            return ctx

        ohlcv = self._prepare(candles)

        try:
            import smartmoneyconcepts as smc
        except ImportError:
            ctx.errors.append("smartmoneyconcepts not installed — pip install smartmoneyconcepts")
            ctx.summary = "[SMC unavailable — library not installed]"
            return ctx

        # ── Volume context ────────────────────────────────────────────────────
        if "volume" in ohlcv.columns:
            try:
                vol_sma = ohlcv["volume"].rolling(20).mean().iloc[-1]
                ctx.volume_above_avg = float(ohlcv["volume"].iloc[-1]) > float(vol_sma)
            except Exception as exc:
                ctx.errors.append(f"volume: {exc}")

        # ── Swing highs / lows ────────────────────────────────────────────────
        swing_hl = None
        try:
            swing_hl = smc.swing_highs_lows(ohlcv, swing_length=50)
        except Exception as exc:
            ctx.errors.append(f"swing_highs_lows: {exc}")

        # ── BOS / CHoCH ───────────────────────────────────────────────────────
        try:
            if swing_hl is not None:
                bos_df = smc.bos_choch(ohlcv, swing_hl, close_break=True)
                if bos_df is not None and len(bos_df) > 0:
                    bos_rows = bos_df.dropna(subset=["BOS"])
                    choch_rows = bos_df.dropna(subset=["CHOCH"])

                    if len(bos_rows):
                        last = bos_rows.iloc[-1]
                        ctx.last_bos = "bullish" if last.get("Direction", 1) > 0 else "bearish"
                    if len(choch_rows):
                        last = choch_rows.iloc[-1]
                        ctx.last_choch = "bullish" if last.get("Direction", 1) > 0 else "bearish"

                    # Infer structure from most recent BOS/CHoCH
                    if ctx.last_choch:
                        ctx.structure = "uptrend" if ctx.last_choch == "bullish" else "downtrend"
                    elif ctx.last_bos:
                        ctx.structure = "uptrend" if ctx.last_bos == "bullish" else "downtrend"
        except Exception as exc:
            ctx.errors.append(f"bos_choch: {exc}")

        current_price = float(ohlcv["close"].iloc[-1])

        # ── Fair Value Gaps ───────────────────────────────────────────────────
        try:
            fvg_df = smc.fvg(ohlcv, join_consecutive=False)
            if fvg_df is not None and len(fvg_df) > 0:
                open_fvgs = fvg_df[fvg_df["MitigatedIndex"].isna()].copy()
                if len(open_fvgs):
                    open_fvgs["mid"] = (open_fvgs["Top"] + open_fvgs["Bottom"]) / 2
                    open_fvgs["dist"] = (open_fvgs["mid"] - current_price).abs()

                    bull_fvgs = open_fvgs[open_fvgs["FVG"] > 0].nsmallest(1, "dist")
                    bear_fvgs = open_fvgs[open_fvgs["FVG"] < 0].nsmallest(1, "dist")

                    if len(bull_fvgs):
                        r = bull_fvgs.iloc[0]
                        ctx.nearest_fvg_bull = {"top": float(r["Top"]), "bottom": float(r["Bottom"])}
                    if len(bear_fvgs):
                        r = bear_fvgs.iloc[0]
                        ctx.nearest_fvg_bear = {"top": float(r["Top"]), "bottom": float(r["Bottom"])}
        except Exception as exc:
            ctx.errors.append(f"fvg: {exc}")

        # ── Order Blocks ──────────────────────────────────────────────────────
        try:
            ob_df = smc.ob(ohlcv, close_mitigation=False)
            if ob_df is not None and len(ob_df) > 0:
                active_obs = ob_df[ob_df["MitigatedIndex"].isna()].copy()
                if len(active_obs):
                    active_obs["mid"] = (active_obs["Top"] + active_obs["Bottom"]) / 2
                    active_obs["dist"] = (active_obs["mid"] - current_price).abs()

                    bull_obs = active_obs[active_obs["OB"] > 0].nsmallest(1, "dist")
                    bear_obs = active_obs[active_obs["OB"] < 0].nsmallest(1, "dist")

                    if len(bull_obs):
                        r = bull_obs.iloc[0]
                        ctx.nearest_ob_bull = {"top": float(r["Top"]), "bottom": float(r["Bottom"])}
                    if len(bear_obs):
                        r = bear_obs.iloc[0]
                        ctx.nearest_ob_bear = {"top": float(r["Top"]), "bottom": float(r["Bottom"])}
        except Exception as exc:
            ctx.errors.append(f"ob: {exc}")

        ctx.summary = self._build_summary(ctx, current_price)
        return ctx

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _prepare(candles: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to match smartmoneyconcepts expectations."""
        rename = {}
        col_map = {
            "open": ["open", "Open"],
            "high": ["high", "High"],
            "low": ["low", "Low"],
            "close": ["close", "Close"],
            "volume": ["volume", "Volume"],
        }
        for target, candidates in col_map.items():
            for src in candidates:
                if src in candles.columns and target not in candles.columns:
                    rename[src] = target
        df = candles.rename(columns=rename).copy()
        return df.reset_index(drop=True)

    @staticmethod
    def _build_summary(ctx: SMCContext, price: float) -> str:
        parts = [f"SMC [{ctx.structure.upper()}]"]

        if ctx.last_choch:
            parts.append(f"CHoCH={ctx.last_choch}")
        if ctx.last_bos:
            parts.append(f"BOS={ctx.last_bos}")

        if ctx.nearest_fvg_bull:
            z = ctx.nearest_fvg_bull
            parts.append(f"Bull FVG {z['bottom']:.2f}-{z['top']:.2f}")
        if ctx.nearest_fvg_bear:
            z = ctx.nearest_fvg_bear
            parts.append(f"Bear FVG {z['bottom']:.2f}-{z['top']:.2f}")
        if ctx.nearest_ob_bull:
            z = ctx.nearest_ob_bull
            parts.append(f"Bull OB {z['bottom']:.2f}-{z['top']:.2f}")
        if ctx.nearest_ob_bear:
            z = ctx.nearest_ob_bear
            parts.append(f"Bear OB {z['bottom']:.2f}-{z['top']:.2f}")

        if ctx.volume_above_avg:
            parts.append("VOL↑")

        return " | ".join(parts)
