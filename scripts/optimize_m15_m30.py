"""
M15 / M30 MACD Divergence Parameter Optimizer
==============================================
Fetches 3 months of real Binance Futures data, runs walk-forward
optimization (75% train / 25% test) across a parameter grid, and
reports the best params for each coin × timeframe.

Usage:
    cd /Users/mw/Dhaferr
    python scripts/optimize_m15_m30.py

Output: prints a table + writes results to scripts/opt_results.json
"""
import json
import time
import itertools
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# ── Binance Futures OHLCV fetch ────────────────────────────────────────────────

def fetch_klines(symbol: str, interval: str, limit: int = 1500) -> pd.DataFrame:
    """Fetch up to `limit` klines from Binance Futures."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    all_rows = []
    end_time = None

    while len(all_rows) < limit:
        params = {"symbol": symbol, "interval": interval, "limit": min(1500, limit - len(all_rows))}
        if end_time:
            params["endTime"] = end_time

        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break

        all_rows = rows + all_rows
        end_time = rows[0][0] - 1  # go further back
        if len(rows) < 1500:
            break
        time.sleep(0.1)

    df = pd.DataFrame(all_rows, columns=[
        "ts", "open", "high", "low", "close", "volume",
        "close_time", "qv", "trades", "tbbv", "tbqv", "ignore"
    ])
    df["ts"]    = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["open"]  = df["open"].astype(float)
    df["high"]  = df["high"].astype(float)
    df["low"]   = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"]= df["volume"].astype(float)
    df = df[["ts","open","high","low","close","volume"]].set_index("ts").sort_index()
    return df.iloc[-limit:]


# ── Indicators ────────────────────────────────────────────────────────────────

def _macd(close: np.ndarray, fast=12, slow=26, sig=9):
    def ema(arr, span):
        k, out = 2/(span+1), np.zeros(len(arr))
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = arr[i]*k + out[i-1]*(1-k)
        return out
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, sig)
    return m, s, m - s

def _rsi(close: np.ndarray, period=14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    def smooth(arr):
        out = np.zeros(len(arr))
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = (out[i-1]*(period-1) + arr[i]) / period
        return out
    ag, al = smooth(gain), smooth(loss)
    rs = np.where(al == 0, 100, ag / (al + 1e-9))
    return 100 - 100/(1+rs)

def _atr(high, low, close, period=14) -> np.ndarray:
    tr = np.maximum(high-low, np.maximum(np.abs(high-np.roll(close,1)), np.abs(low-np.roll(close,1))))
    tr[0] = high[0]-low[0]
    out = np.zeros(len(tr))
    out[0] = tr[0]
    k = 1/period
    for i in range(1, len(tr)):
        out[i] = tr[i]*k + out[i-1]*(1-k)
    return out

def _swing_lows(arr, start, end, gap):
    result = []
    for i in range(start+gap, end-gap):
        if all(arr[i] <= arr[i-j] for j in range(1,gap+1)) and \
           all(arr[i] <= arr[i+j] for j in range(1,gap+1)):
            result.append(i)
    return result

def _swing_highs(arr, start, end, gap):
    result = []
    for i in range(start+gap, end-gap):
        if all(arr[i] >= arr[i-j] for j in range(1,gap+1)) and \
           all(arr[i] >= arr[i+j] for j in range(1,gap+1)):
            result.append(i)
    return result


# ── Single-bar signal check ────────────────────────────────────────────────────

def check_signal(i, closes, highs, lows, macd, hist, rsi, atr, params):
    """Check if bar i generates a signal. Returns list of (direction, entry, sl, tp1) or []"""
    lb  = params["div_lookback"]
    gap = params["swing_gap"]
    ht  = params["hist_trigger"]
    rbm = params["rsi_buy_max"]
    rsm = params["rsi_sell_min"]
    slm = params["atr_sl_mult"]
    rr  = params["rr"]

    start = max(0, i - lb)
    end   = i  # exclusive of current bar (no lookahead)
    if end - start < gap*2 + 2:
        return []

    entry = closes[i]
    atr_v = atr[i]
    if atr_v <= 0:
        return []

    rsi_v = rsi[i]
    signals = []

    # ── Bullish divergence ────────────────────────────────────────────────────
    if rsi_v <= rbm:
        plows = _swing_lows(closes, start, end, gap)
        mlows = _swing_lows(macd,   start, end, gap)
        if len(plows) >= 2 and len(mlows) >= 2:
            p1, p2 = plows[-2], plows[-1]
            m1, m2 = mlows[-2], mlows[-1]
            if closes[p2] < closes[p1] and macd[m2] > macd[m1]:
                # Histogram trigger
                win = ht + 2
                if i >= win:
                    recent = hist[i-win:i]
                    neg_count = sum(1 for x in recent if x < 0)
                    rising    = all(recent[j] > recent[j-1] for j in range(1,len(recent)))
                    if neg_count >= 2 and rising:
                        sl = entry - atr_v * slm
                        tp = entry + (entry - sl) * rr
                        signals.append(("BUY", entry, sl, tp))

    # ── Bearish divergence ────────────────────────────────────────────────────
    if rsi_v >= rsm:
        phighs = _swing_highs(closes, start, end, gap)
        mhighs = _swing_highs(macd,   start, end, gap)
        if len(phighs) >= 2 and len(mhighs) >= 2:
            p1, p2 = phighs[-2], phighs[-1]
            m1, m2 = mhighs[-2], mhighs[-1]
            if closes[p2] > closes[p1] and macd[m2] < macd[m1]:
                win = ht + 2
                if i >= win:
                    recent = hist[i-win:i]
                    pos_count = sum(1 for x in recent if x > 0)
                    falling   = all(recent[j] < recent[j-1] for j in range(1,len(recent)))
                    if pos_count >= 2 and falling:
                        sl = entry + atr_v * slm
                        tp = entry - (sl - entry) * rr
                        signals.append(("SELL", entry, sl, tp))

    return signals


# ── Backtester ─────────────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, params: dict, commission: float = 0.0005) -> dict:
    """Simulate strategy on df. Returns metrics dict."""
    closes = df["close"].to_numpy()
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    n = len(closes)

    macd, _, hist = _macd(closes)
    rsi  = _rsi(closes)
    atr  = _atr(highs, lows, closes)

    trades = []
    in_trade = False
    entry_dir = sl = tp = entry_price = 0

    warmup = max(50, params["div_lookback"])

    for i in range(warmup, n):
        if in_trade:
            # Check exit
            if entry_dir == "BUY":
                if lows[i] <= sl:
                    pnl = (sl - entry_price) / entry_price - 2*commission
                    trades.append(pnl)
                    in_trade = False
                elif highs[i] >= tp:
                    pnl = (tp - entry_price) / entry_price - 2*commission
                    trades.append(pnl)
                    in_trade = False
            else:
                if highs[i] >= sl:
                    pnl = (entry_price - sl) / entry_price - 2*commission
                    trades.append(pnl)
                    in_trade = False
                elif lows[i] <= tp:
                    pnl = (entry_price - tp) / entry_price - 2*commission
                    trades.append(pnl)
                    in_trade = False
            continue

        sigs = check_signal(i, closes, highs, lows, macd, hist, rsi, atr, params)
        if sigs:
            entry_dir, entry_price, sl, tp = sigs[0]
            in_trade = True

    if not trades:
        return {"trades": 0, "wr": 0, "pf": 0, "net": 0}

    wins   = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    wr     = len(wins) / len(trades)
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses)) if losses else 1e-9
    pf     = gross_profit / gross_loss
    net    = sum(trades) * 100  # as %

    return {"trades": len(trades), "wr": round(wr, 4), "pf": round(pf, 3), "net": round(net, 2)}


# ── Walk-forward wrapper ───────────────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, params: dict, train_ratio: float = 0.75) -> dict:
    """Split df 75/25, optimize on train, validate on test."""
    split = int(len(df) * train_ratio)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    train_m = backtest(train_df, params)
    test_m  = backtest(test_df,  params)

    return {
        "train": train_m,
        "test":  test_m,
        "score": test_m["pf"] * (1 if test_m["trades"] >= 20 else 0.3),
    }


# ── Grid search ───────────────────────────────────────────────────────────────

GRID = {
    "div_lookback": [20, 40, 60, 80],
    "swing_gap":    [1, 2, 3],
    "hist_trigger": [2, 3],
    "atr_sl_mult":  [1.0, 1.5, 2.0, 2.5],
    "rr":           [1.0, 1.3, 1.5, 2.0],
    "rsi_buy_max":  [68],
    "rsi_sell_min": [22],
}

SYMBOLS    = ["ZECUSDT", "BTCUSDT", "BCHUSDT", "ETHUSDT"]
TIMEFRAMES = {"M15": "15m", "M30": "30m"}
BARS       = 2500   # ~26 days M15, ~52 days M30


def run():
    total_combos = 1
    for v in GRID.values():
        total_combos *= len(v)
    print(f"\nGrid: {total_combos} combinations × {len(SYMBOLS)} coins × {len(TIMEFRAMES)} TFs\n")

    all_results = {}

    for symbol in SYMBOLS:
        all_results[symbol] = {}
        for tf_name, tf_interval in TIMEFRAMES.items():
            print(f"{'='*60}")
            print(f"  {symbol} {tf_name}  — fetching {BARS} bars...")
            try:
                df = fetch_klines(symbol, tf_interval, limit=BARS)
            except Exception as e:
                print(f"  ERROR fetching data: {e}")
                continue
            print(f"  Got {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")

            best = {"score": -1}
            tested = 0

            keys   = list(GRID.keys())
            values = list(GRID.values())

            for combo in itertools.product(*values):
                params = dict(zip(keys, combo))
                wf = walk_forward(df, params)
                tested += 1

                if wf["score"] > best["score"] and wf["test"]["trades"] >= 20:
                    best = {**wf, "params": params, "symbol": symbol, "tf": tf_name}

                if tested % 200 == 0:
                    print(f"  ... {tested}/{total_combos} tested, best test PF so far: {best.get('test',{}).get('pf','—')}")

            if best["score"] > 0:
                p = best["params"]
                t = best["test"]
                tr = best["train"]
                print(f"\n  ✅ BEST for {symbol} {tf_name}:")
                print(f"     Params : lb={p['div_lookback']} gap={p['swing_gap']} ht={p['hist_trigger']} "
                      f"atr={p['atr_sl_mult']} rr={p['rr']}")
                print(f"     Train  : {tr['trades']} trades  WR={tr['wr']*100:.1f}%  PF={tr['pf']:.2f}  net={tr['net']:.1f}%")
                print(f"     TEST   : {t['trades']} trades   WR={t['wr']*100:.1f}%  PF={t['pf']:.2f}  net={t['net']:.1f}%")
                all_results[symbol][tf_name] = best
            else:
                print(f"\n  ⚠️  No param set found ≥20 test trades for {symbol} {tf_name}")
                all_results[symbol][tf_name] = None

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Symbol':<10} {'TF':<6} {'Trades':>7} {'WR%':>6} {'PF':>5} {'Net%':>7}  Params")
    print(f"  {'-'*9} {'-'*5} {'-'*7} {'-'*6} {'-'*5} {'-'*7}  {'-'*40}")

    for symbol in SYMBOLS:
        for tf_name in TIMEFRAMES:
            r = all_results.get(symbol, {}).get(tf_name)
            if r:
                t = r["test"]
                p = r["params"]
                param_str = f"lb={p['div_lookback']} gap={p['swing_gap']} atr={p['atr_sl_mult']} rr={p['rr']}"
                print(f"  {symbol:<10} {tf_name:<6} {t['trades']:>7} {t['wr']*100:>5.1f}% {t['pf']:>5.2f} {t['net']:>7.1f}%  {param_str}")
            else:
                print(f"  {symbol:<10} {tf_name:<6}  {'—':>7}")

    # Save JSON
    import os
    os.makedirs("scripts", exist_ok=True)

    def safe(obj):
        if isinstance(obj, dict):
            return {k: safe(v) for k,v in obj.items()}
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    with open("scripts/opt_results.json", "w") as f:
        json.dump(safe(all_results), f, indent=2)
    print(f"\n  Results saved to scripts/opt_results.json\n")


if __name__ == "__main__":
    run()
