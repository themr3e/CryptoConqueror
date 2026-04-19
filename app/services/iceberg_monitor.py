"""Iceberg Order Monitor — background service for Railway.

Runs one daemon thread per symbol, polling Binance order book every 250ms.
Detections are stored in a bounded in-memory deque (last 50 per symbol).

Usage (from anywhere in the app):
    from app.services.iceberg_monitor import iceberg_monitor
    hits = iceberg_monitor.get_detections("BTCUSDT")   # returns list[dict]
    summary = iceberg_monitor.status()                  # health summary
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import requests
from loguru import logger

# ── Config ─────────────────────────────────────────────────────────────────────
POLL_MS         = 250       # book poll interval
WINDOW_SECS     = 60        # how long to track a price level
MIN_USD         = 50_000    # min USD size to care about
REFILL_MS       = 500       # mechanical refill threshold
REFILL_COUNT    = 3         # min fast refills to count as PERSISTENCE
ABSORPTION_MULT = 1.5       # traded > N× visible → ABSORPTION
MAX_DETECTIONS  = 50        # max stored detections per symbol
RESTART_DELAY   = 5         # seconds to wait before restarting a crashed thread


# ── Low-level helpers (sync, called from worker threads) ──────────────────────

def _fetch_depth(symbol: str, limit: int = 20) -> dict:
    url = f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit={limit}"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return r.json()


def _fetch_recent_trades(symbol: str, limit: int = 20) -> list:
    url = f"https://fapi.binance.com/fapi/v1/trades?symbol={symbol}&limit={limit}"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return r.json()


# ── Price-level tracker ────────────────────────────────────────────────────────

class _PriceLevel:
    __slots__ = (
        "price", "side", "min_size", "max_size", "last_size",
        "refills", "last_disappear", "refill_times",
        "traded_at_level", "first_seen", "last_seen",
    )

    def __init__(self, price: float, size: float, side: str):
        self.price           = price
        self.side            = side
        self.min_size        = size
        self.max_size        = size
        self.last_size       = size
        self.refills         = 0
        self.last_disappear: float | None = None
        self.refill_times:   list[float]  = []
        self.traded_at_level = 0.0
        self.first_seen      = time.monotonic()
        self.last_seen       = time.monotonic()

    def update(self, size: float) -> None:
        now = time.monotonic()
        self.last_seen = now
        if size > self.last_size * 1.5 and self.last_disappear is not None:
            self.refills += 1
            self.refill_times.append((now - self.last_disappear) * 1000)
            self.last_disappear = None
        self.min_size = min(self.min_size, size)
        self.max_size = max(self.max_size, size)
        self.last_size = size

    def mark_disappeared(self) -> None:
        self.last_disappear = time.monotonic()

    @property
    def avg_refill_ms(self) -> float:
        return sum(self.refill_times) / len(self.refill_times) if self.refill_times else 0.0

    @property
    def fast_refills(self) -> int:
        return sum(1 for t in self.refill_times if t < REFILL_MS)

    def iceberg_score(self) -> tuple[int, list[str]]:
        signals: list[str] = []
        score = 0

        if self.fast_refills >= REFILL_COUNT:
            score += 1
            signals.append(
                f"PERSISTENCE: refilled {self.fast_refills}× in <{REFILL_MS}ms "
                f"(avg {self.avg_refill_ms:.0f}ms)"
            )

        visible_usd = self.min_size * self.price
        if self.traded_at_level > visible_usd * ABSORPTION_MULT and self.traded_at_level > MIN_USD:
            score += 1
            signals.append(
                f"ABSORPTION: ${self.traded_at_level:,.0f} traded vs "
                f"${visible_usd:,.0f} visible ({self.traded_at_level/visible_usd:.1f}×)"
            )

        size_variance = (self.max_size - self.min_size) / (self.min_size + 1e-9)
        if self.refills >= 2 and size_variance < 0.15:
            score += 1
            signals.append(
                f"STABLE TIP: ~{self.min_size:.2f} units "
                f"({self.refills} refills, variance {size_variance:.1%})"
            )

        return score, signals


# ── Per-symbol scanner (runs in a thread) ─────────────────────────────────────

class _SymbolScanner:
    def __init__(self, symbol: str, detections: deque, stop_event: threading.Event):
        self.symbol       = symbol
        self.detections   = detections
        self.stop_event   = stop_event
        self.levels: dict[float, _PriceLevel] = {}
        self.prev_book: dict[str, dict[float, float]] = {"bid": {}, "ask": {}}
        self.last_trade_id = 0
        self._alerted: dict[float, float] = {}   # price → monotonic time of last alert

    def _parse_book(self, depth: dict) -> dict[str, dict[float, float]]:
        book: dict[str, dict[float, float]] = {"bid": {}, "ask": {}}
        for p, v in depth.get("bids", []):
            book["bid"][float(p)] = float(v)
        for p, v in depth.get("asks", []):
            book["ask"][float(p)] = float(v)
        return book

    def _process_trades(self) -> None:
        try:
            trades = _fetch_recent_trades(self.symbol)
            for t in trades:
                tid = int(t["id"])
                if tid <= self.last_trade_id:
                    continue
                self.last_trade_id = tid
                price = float(t["price"])
                usd   = price * float(t["qty"])
                if price in self.levels:
                    self.levels[price].traded_at_level += usd
        except Exception:
            pass

    def run(self) -> None:
        logger.info("[Iceberg] {} scanner started", self.symbol)
        while not self.stop_event.is_set():
            try:
                depth = _fetch_depth(self.symbol)
            except Exception:
                time.sleep(POLL_MS / 1000)
                continue

            book = self._parse_book(depth)
            now  = time.monotonic()

            for side in ("bid", "ask"):
                prev = self.prev_book[side]
                curr = book[side]

                for price in prev:
                    if price not in curr and price in self.levels:
                        self.levels[price].mark_disappeared()

                for price, size in curr.items():
                    if price * size < MIN_USD:
                        continue
                    if price in self.levels:
                        self.levels[price].update(size)
                    else:
                        self.levels[price] = _PriceLevel(price, size, side)

                self.prev_book[side] = curr

            self._process_trades()

            # Prune stale levels
            stale = [p for p, lv in self.levels.items() if now - lv.last_seen > WINDOW_SECS]
            for p in stale:
                del self.levels[p]

            # Check for icebergs
            for price, lv in list(self.levels.items()):
                score, signals = lv.iceberg_score()
                if score < 2:
                    continue
                last_alert = self._alerted.get(price, 0.0)
                if now - last_alert < 30:
                    continue     # don't re-alert same level within 30s
                self._alerted[price] = now
                entry: dict[str, Any] = {
                    "symbol":    self.symbol,
                    "time":      datetime.now(timezone.utc).isoformat(),
                    "side":      lv.side.upper(),
                    "price":     price,
                    "usd":       price * lv.min_size,
                    "score":     score,
                    "signals":   signals,
                    "refills":   lv.refills,
                }
                self.detections.appendleft(entry)
                logger.info(
                    "[Iceberg] {} ★×{} @ {:.2f} {} — {}",
                    self.symbol, score, price, lv.side.upper(),
                    " | ".join(signals),
                )

            time.sleep(POLL_MS / 1000)

        logger.info("[Iceberg] {} scanner stopped", self.symbol)


# ── Public monitor (module-level singleton) ────────────────────────────────────

class IcebergMonitor:
    """Manages one background daemon thread per symbol.

    Call ``start(symbols)`` once at app startup.  After that, use
    ``get_detections(symbol)`` anywhere to read iceberg hits.
    """

    def __init__(self) -> None:
        self._threads:    dict[str, threading.Thread]  = {}
        self._stop_events: dict[str, threading.Event]  = {}
        self._detections: dict[str, deque]             = {}
        self._running = False

    def start(self, symbols: list[str]) -> None:
        if self._running:
            return
        self._running = True
        for symbol in symbols:
            self._launch(symbol)
        logger.info("[Iceberg] Monitor started for {} symbol(s): {}", len(symbols), symbols)

    def _launch(self, symbol: str) -> None:
        self._detections.setdefault(symbol, deque(maxlen=MAX_DETECTIONS))
        stop_ev = threading.Event()
        self._stop_events[symbol] = stop_ev

        scanner = _SymbolScanner(symbol, self._detections[symbol], stop_ev)

        def _run_with_restart() -> None:
            while not stop_ev.is_set():
                try:
                    scanner.run()
                except Exception:
                    logger.opt(exception=True).warning(
                        "[Iceberg] {} scanner crashed — restarting in {}s",
                        symbol, RESTART_DELAY,
                    )
                    if not stop_ev.is_set():
                        time.sleep(RESTART_DELAY)

        t = threading.Thread(target=_run_with_restart, name=f"iceberg-{symbol}", daemon=True)
        t.start()
        self._threads[symbol] = t

    def stop(self) -> None:
        for ev in self._stop_events.values():
            ev.set()
        self._running = False
        logger.info("[Iceberg] Monitor stopped")

    def get_detections(self, symbol: str, max_items: int = 10) -> list[dict]:
        """Return up to *max_items* most recent iceberg detections for *symbol*."""
        dq = self._detections.get(symbol)
        if dq is None:
            return []
        return list(dq)[:max_items]

    def status(self) -> dict[str, Any]:
        """Return a health summary for all monitored symbols."""
        return {
            symbol: {
                "thread_alive":  self._threads[symbol].is_alive() if symbol in self._threads else False,
                "detections":    len(self._detections.get(symbol, [])),
                "latest":        self._detections[symbol][0]["time"]
                                 if self._detections.get(symbol) else None,
            }
            for symbol in self._detections
        }

    def check_threads(self) -> None:
        """Restart any threads that died unexpectedly (called by APScheduler job)."""
        for symbol, thread in list(self._threads.items()):
            if not thread.is_alive() and not self._stop_events[symbol].is_set():
                logger.warning("[Iceberg] {} thread died — restarting", symbol)
                self._launch(symbol)


# Singleton used everywhere in the app
iceberg_monitor = IcebergMonitor()
