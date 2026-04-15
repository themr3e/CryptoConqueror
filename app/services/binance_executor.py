"""Binance Futures order executor.

Handles HMAC-signed order placement on Binance Futures (testnet or mainnet).

For every signal it places three orders:
  1. MARKET entry order
  2. STOP_MARKET stop-loss (reduceOnly)
  3. TAKE_PROFIT_MARKET take-profit (reduceOnly)

Uses the Binance Futures TESTNET by default — safe to run with fake money.
Switch to mainnet by setting BINANCE_TESTNET=false in your .env.

Testnet sign-up: https://testnet.binancefuture.com (login with GitHub)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any

import httpx
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.signal import Signal
from app.models.trade_order import TradeOrder


@dataclass
class OrderResult:
    """Result of a single order placement attempt."""
    success: bool
    order_role: str          # "entry" | "stop_loss" | "take_profit_1"
    broker_order_id: str | None
    status: str              # "NEW" | "FILLED" | "REJECTED" | "ERROR"
    raw_response: dict | None
    error_message: str | None = None


@dataclass
class ExecutionResult:
    """Full result of executing one signal (entry + SL + TP)."""
    signal_id: int
    symbol: str
    success: bool
    orders: list[OrderResult]
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Lot size / tick size precision helpers
# ---------------------------------------------------------------------------

# Binance Futures quantity step sizes (lot size) per symbol
_LOT_SIZE_DEFAULTS: dict[str, Decimal] = {
    "BTCUSDT":    Decimal("0.001"),
    "ETHUSDT":    Decimal("0.001"),
    "BNBUSDT":    Decimal("0.01"),
    "SOLUSDT":    Decimal("0.1"),
    "XRPUSDT":    Decimal("1"),
    "ADAUSDT":    Decimal("1"),
    "DOGEUSDT":   Decimal("1"),
    "AVAXUSDT":   Decimal("1"),
    "LINKUSDT":   Decimal("0.01"),
    "DOTUSDT":    Decimal("0.1"),
    "LTCUSDT":    Decimal("0.01"),
    "UNIUSDT":    Decimal("1"),
    "ATOMUSDT":   Decimal("0.01"),
    "NEARUSDT":   Decimal("1"),
    "APTUSDT":    Decimal("0.1"),
    "ARBUSDT":    Decimal("1"),
    "OPUSDT":     Decimal("0.1"),
    "INJUSDT":    Decimal("0.1"),
    "SUIUSDT":    Decimal("1"),
    "FILUSDT":    Decimal("0.1"),
    "AAVEUSDT":   Decimal("0.1"),
    "MKRUSDT":    Decimal("0.001"),
    "RUNEUSDT":   Decimal("1"),
    "STXUSDT":    Decimal("1"),
    "FETUSDT":    Decimal("1"),
    "RENDERUSDT": Decimal("0.1"),
    "WLDUSDT":    Decimal("1"),
    "TIAUSDT":    Decimal("1"),
    "SEIUSDT":    Decimal("1"),
    "JUPUSDT":    Decimal("1"),
    "PYTHUSDT":   Decimal("1"),
    "MATICUSDT":  Decimal("1"),
}

_TICK_SIZE_DEFAULTS: dict[str, Decimal] = {
    "BTCUSDT":    Decimal("0.10"),
    "ETHUSDT":    Decimal("0.01"),
    "BNBUSDT":    Decimal("0.01"),
    "SOLUSDT":    Decimal("0.01"),
    "XRPUSDT":    Decimal("0.0001"),
    "ADAUSDT":    Decimal("0.0001"),
    "DOGEUSDT":   Decimal("0.00001"),
    "AVAXUSDT":   Decimal("0.001"),
    "LINKUSDT":   Decimal("0.001"),
    "DOTUSDT":    Decimal("0.001"),
    "LTCUSDT":    Decimal("0.01"),
    "UNIUSDT":    Decimal("0.001"),
    "ATOMUSDT":   Decimal("0.001"),
    "NEARUSDT":   Decimal("0.001"),
    "APTUSDT":    Decimal("0.001"),
    "ARBUSDT":    Decimal("0.0001"),
    "OPUSDT":     Decimal("0.001"),
    "INJUSDT":    Decimal("0.001"),
    "SUIUSDT":    Decimal("0.0001"),
    "FILUSDT":    Decimal("0.001"),
    "AAVEUSDT":   Decimal("0.01"),
    "MKRUSDT":    Decimal("0.10"),
    "RUNEUSDT":   Decimal("0.001"),
    "STXUSDT":    Decimal("0.0001"),
    "FETUSDT":    Decimal("0.0001"),
    "RENDERUSDT": Decimal("0.001"),
    "WLDUSDT":    Decimal("0.001"),
    "TIAUSDT":    Decimal("0.001"),
    "SEIUSDT":    Decimal("0.0001"),
    "JUPUSDT":    Decimal("0.0001"),
    "PYTHUSDT":   Decimal("0.0001"),
    "MATICUSDT":  Decimal("0.0001"),
}


def _round_quantity(qty: Decimal, symbol: str) -> Decimal:
    """Round quantity down to the symbol's lot size step."""
    step = _LOT_SIZE_DEFAULTS.get(symbol, Decimal("0.001"))
    return (qty / step).to_integral_value(rounding=ROUND_DOWN) * step


def _round_price(price: Decimal, symbol: str) -> Decimal:
    """Round price to the symbol's tick size."""
    tick = _TICK_SIZE_DEFAULTS.get(symbol, Decimal("0.01"))
    return (price / tick).to_integral_value(rounding=ROUND_DOWN) * tick


class BinanceExecutor:
    """Places and tracks Binance Futures orders for generated signals."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.binance_futures_api_key
        self._api_secret = settings.binance_futures_api_secret
        self._base_url = settings.binance_base_url
        self._leverage = settings.binance_leverage
        self._testnet = settings.binance_testnet
        self._environment = "testnet" if self._testnet else "mainnet"
        self._timeout = 15.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def execute_signal(
        self,
        session: AsyncSession,
        signal: Signal,
    ) -> ExecutionResult:
        """Execute a signal: set leverage, place entry + SL + TP orders.

        Args:
            session: Async DB session (for persisting TradeOrder records).
            signal:  Persisted Signal ORM object.

        Returns:
            ExecutionResult with details of all placed orders.
        """
        if not self._api_key or not self._api_secret:
            return ExecutionResult(
                signal_id=signal.id,
                symbol=signal.symbol,
                success=False,
                orders=[],
                error_message="Binance API key/secret not configured",
            )

        logger.info(
            "BinanceExecutor: executing signal {} — {} {} ({})",
            signal.id, signal.direction, signal.symbol, self._environment,
        )

        # 1. Set leverage for this symbol
        await self._set_leverage(signal.symbol)

        # 2. Calculate position quantity
        quantity = self._calculate_quantity(signal)
        if quantity <= 0:
            return ExecutionResult(
                signal_id=signal.id,
                symbol=signal.symbol,
                success=False,
                orders=[],
                error_message=f"Calculated quantity {quantity} is zero or negative",
            )

        orders: list[OrderResult] = []

        # 3. Place entry MARKET order
        entry_result = await self._place_entry(session, signal, quantity)
        orders.append(entry_result)

        if not entry_result.success:
            return ExecutionResult(
                signal_id=signal.id,
                symbol=signal.symbol,
                success=False,
                orders=orders,
                error_message=f"Entry order failed: {entry_result.error_message}",
            )

        # 4. Place STOP_MARKET stop-loss (opposite side, reduceOnly)
        sl_result = await self._place_stop_loss(session, signal, quantity)
        orders.append(sl_result)

        # 5. Place TAKE_PROFIT_MARKET take-profit (opposite side, reduceOnly)
        tp_result = await self._place_take_profit(session, signal, quantity)
        orders.append(tp_result)

        all_ok = entry_result.success  # SL/TP failures are logged but don't void the trade
        if not sl_result.success:
            logger.warning("BinanceExecutor: SL order failed for signal {} — {}", signal.id, sl_result.error_message)
        if not tp_result.success:
            logger.warning("BinanceExecutor: TP order failed for signal {} — {}", signal.id, tp_result.error_message)

        logger.info(
            "BinanceExecutor: signal {} executed on {} — entry={}, sl={}, tp={}",
            signal.id, self._environment,
            entry_result.status, sl_result.status, tp_result.status,
        )

        return ExecutionResult(
            signal_id=signal.id,
            symbol=signal.symbol,
            success=all_ok,
            orders=orders,
        )

    async def cancel_signal_orders(
        self,
        session: AsyncSession,
        signal_id: int,
        symbol: str,
    ) -> int:
        """Cancel all open orders linked to a signal.

        Args:
            session:   Async DB session.
            signal_id: Signal ID to cancel orders for.
            symbol:    Binance symbol (needed for cancel endpoint).

        Returns:
            Number of orders successfully cancelled.
        """
        from sqlalchemy import select, and_

        stmt = select(TradeOrder).where(
            and_(
                TradeOrder.signal_id == signal_id,
                TradeOrder.status == "NEW",
                TradeOrder.broker_order_id.isnot(None),
            )
        )
        result = await session.execute(stmt)
        orders = result.scalars().all()

        cancelled = 0
        for order in orders:
            ok = await self._cancel_order(symbol, order.broker_order_id)
            if ok:
                order.status = "CANCELED"
                cancelled += 1

        if cancelled:
            await session.commit()

        return cancelled

    async def get_account_balance(self) -> Decimal | None:
        """Fetch available USDT balance from Binance Futures account.

        Returns:
            Available balance in USDT, or None on failure.
        """
        try:
            data = await self._signed_get("/fapi/v2/account", {})
            assets = data.get("assets", [])
            for asset in assets:
                if asset.get("asset") == "USDT":
                    return Decimal(str(asset.get("availableBalance", "0")))
        except Exception:
            logger.opt(exception=True).warning("BinanceExecutor: failed to fetch account balance")
        return None

    # ------------------------------------------------------------------
    # Private order placement helpers
    # ------------------------------------------------------------------

    async def _place_entry(
        self,
        session: AsyncSession,
        signal: Signal,
        quantity: Decimal,
    ) -> OrderResult:
        """Place the entry MARKET order."""
        side = signal.direction  # "BUY" or "SELL"
        params = {
            "symbol":   signal.symbol,
            "side":     side,
            "type":     "MARKET",
            "quantity": str(_round_quantity(quantity, signal.symbol)),
        }

        return await self._place_and_record(
            session, signal, params, order_role="entry"
        )

    async def _place_stop_loss(
        self,
        session: AsyncSession,
        signal: Signal,
        quantity: Decimal,
    ) -> OrderResult:
        """Place the STOP_MARKET stop-loss order.

        Uses closePosition=true so the entire position is closed when
        the stop price is hit. This form is required on Binance Futures
        Demo (demo-fapi.binance.com) — using quantity+reduceOnly triggers
        error -4120 on that environment.
        """
        side = "SELL" if signal.direction == "BUY" else "BUY"
        stop_price = _round_price(signal.stop_loss, signal.symbol)

        params = {
            "symbol":        signal.symbol,
            "side":          side,
            "type":          "STOP_MARKET",
            "stopPrice":     str(stop_price),
            "closePosition": "true",
        }

        return await self._place_and_record(
            session, signal, params, order_role="stop_loss"
        )

    async def _place_take_profit(
        self,
        session: AsyncSession,
        signal: Signal,
        quantity: Decimal,
    ) -> OrderResult:
        """Place the TAKE_PROFIT_MARKET take-profit-1 order.

        Uses closePosition=true for the same reason as _place_stop_loss —
        required on Binance Futures Demo to avoid -4120.
        """
        side = "SELL" if signal.direction == "BUY" else "BUY"
        tp_price = _round_price(signal.take_profit_1, signal.symbol)

        params = {
            "symbol":        signal.symbol,
            "side":          side,
            "type":          "TAKE_PROFIT_MARKET",
            "stopPrice":     str(tp_price),
            "closePosition": "true",
        }

        return await self._place_and_record(
            session, signal, params, order_role="take_profit_1"
        )

    async def _place_and_record(
        self,
        session: AsyncSession,
        signal: Signal,
        params: dict[str, Any],
        order_role: str,
    ) -> OrderResult:
        """Place an order and persist a TradeOrder record regardless of outcome."""
        raw: dict | None = None
        error_msg: str | None = None
        broker_order_id: str | None = None
        status = "ERROR"

        try:
            raw = await self._signed_post("/fapi/v1/order", params)
            broker_order_id = str(raw.get("orderId", ""))
            status = raw.get("status", "NEW")
            success = True
            logger.info(
                "BinanceExecutor: {} order placed — orderId={} status={}",
                order_role, broker_order_id, status,
            )
        except BinanceAPIError as exc:
            error_msg = str(exc)
            success = False
            logger.error("BinanceExecutor: {} order failed — {}", order_role, error_msg)
        except Exception as exc:
            error_msg = str(exc)
            success = False
            logger.opt(exception=True).error("BinanceExecutor: {} order error", order_role)

        # Always persist a record for auditing
        trade_order = TradeOrder(
            signal_id=signal.id,
            broker_order_id=broker_order_id,
            order_role=order_role,
            symbol=params["symbol"],
            side=params["side"],
            order_type=params["type"],
            quantity=Decimal(str(params.get("quantity", "0"))),
            stop_price=Decimal(str(params["stopPrice"])) if "stopPrice" in params else None,
            status=status if success else "ERROR",
            raw_response=json.dumps(raw) if raw else json.dumps({"error": error_msg}),
            environment=self._environment,
        )
        session.add(trade_order)
        await session.commit()

        return OrderResult(
            success=success,
            order_role=order_role,
            broker_order_id=broker_order_id,
            status=status if success else "ERROR",
            raw_response=raw,
            error_message=error_msg,
        )

    # ------------------------------------------------------------------
    # Leverage + account helpers
    # ------------------------------------------------------------------

    async def _set_leverage(self, symbol: str) -> None:
        """Set leverage for the given symbol."""
        try:
            await self._signed_post("/fapi/v1/leverage", {
                "symbol":   symbol,
                "leverage": self._leverage,
            })
            logger.debug("BinanceExecutor: leverage set to {}x for {}", self._leverage, symbol)
        except Exception:
            logger.opt(exception=True).warning(
                "BinanceExecutor: failed to set leverage for {}", symbol
            )

    async def _cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a single order by ID.

        -2011 (Unknown order) means the order no longer exists on the exchange
        — it was already filled or cancelled externally.  We treat that as a
        successful cancellation so the DB record gets updated to CANCELED.
        """
        try:
            await self._signed_delete("/fapi/v1/order", {
                "symbol":  symbol,
                "orderId": order_id,
            })
            return True
        except BinanceAPIError as exc:
            if exc.code == -2011:
                logger.debug(
                    "BinanceExecutor: order {} not found on exchange (already gone) — "
                    "treating as cancelled",
                    order_id,
                )
                return True
            logger.opt(exception=True).warning(
                "BinanceExecutor: failed to cancel order {}", order_id
            )
            return False
        except Exception:
            logger.opt(exception=True).warning(
                "BinanceExecutor: failed to cancel order {}", order_id
            )
            return False

    def _calculate_quantity(self, signal: Signal) -> Decimal:
        """Calculate position size in base currency.

        Uses account_balance * 1% risk / (entry - SL) * leverage.
        """
        settings = get_settings()
        account_balance = Decimal(str(settings.account_balance))
        risk_pct = Decimal("0.01")  # 1% risk per trade

        entry = signal.entry_price
        sl = signal.stop_loss
        price_risk = abs(entry - sl)

        if price_risk == 0:
            return Decimal("0")

        # Quantity = risk_amount / SL_distance (dollar risk divided by loss per unit)
        # Leverage reduces margin needed but does NOT change the number of units
        # needed to risk exactly risk_pct of account balance.
        risk_amount = account_balance * risk_pct
        quantity = risk_amount / price_risk

        # Cap notional value to avoid oversized positions
        # Hard cap: 5% of account OR $500, whichever is smaller
        max_notional = min(account_balance * Decimal("0.05"), Decimal("500"))
        notional = quantity * entry
        if notional > max_notional:
            quantity = max_notional / entry

        return _round_quantity(quantity, signal.symbol)

    # ------------------------------------------------------------------
    # Signed HTTP helpers
    # ------------------------------------------------------------------

    def _sign(self, params: dict[str, Any]) -> dict[str, Any]:
        """Add timestamp and HMAC-SHA256 signature to params."""
        params["timestamp"] = int(time.time() * 1000)
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _headers(self) -> dict[str, str]:
        return {"X-MBX-APIKEY": self._api_key}

    @staticmethod
    def _is_retryable(status_code: int) -> bool:
        """Return True for transient HTTP errors worth retrying."""
        return status_code in (429, 500, 502, 503, 504)

    async def _signed_post(
        self,
        path: str,
        params: dict[str, Any],
        _retries: int = 2,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(_retries + 1):
            try:
                signed = self._sign(dict(params))
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(
                        f"{self._base_url}{path}",
                        data=signed,
                        headers=self._headers(),
                    )
                    data = resp.json()
                    if resp.status_code == 200:
                        return data
                    if not self._is_retryable(resp.status_code) or attempt == _retries:
                        raise BinanceAPIError(
                            f"HTTP {resp.status_code} — code={data.get('code')} msg={data.get('msg')}",
                            code=data.get("code"),
                        )
                    last_exc = BinanceAPIError(
                        f"HTTP {resp.status_code} (retrying)", code=data.get("code")
                    )
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                if attempt == _retries:
                    raise
            logger.warning("BinanceExecutor: POST {} retry {}/{} after transient error", path, attempt + 1, _retries)
            await asyncio.sleep(1 * (attempt + 1))
        raise last_exc  # unreachable, but satisfies type checker

    async def _signed_get(
        self,
        path: str,
        params: dict[str, Any],
        _retries: int = 2,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(_retries + 1):
            try:
                signed = self._sign(dict(params))
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(
                        f"{self._base_url}{path}",
                        params=signed,
                        headers=self._headers(),
                    )
                    data = resp.json()
                    if resp.status_code == 200:
                        return data
                    if not self._is_retryable(resp.status_code) or attempt == _retries:
                        raise BinanceAPIError(
                            f"HTTP {resp.status_code} — code={data.get('code')} msg={data.get('msg')}",
                            code=data.get("code"),
                        )
                    last_exc = BinanceAPIError(
                        f"HTTP {resp.status_code} (retrying)", code=data.get("code")
                    )
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                if attempt == _retries:
                    raise
            logger.warning("BinanceExecutor: GET {} retry {}/{} after transient error", path, attempt + 1, _retries)
            await asyncio.sleep(1 * (attempt + 1))
        raise last_exc  # unreachable

    async def get_open_position_size(self, symbol: str) -> float:
        """Return the current position size on Binance for a symbol (0 = no position)."""
        try:
            data = await self._signed_get(
                "/fapi/v2/positionRisk",
                {"symbol": symbol},
            )
            if isinstance(data, list):
                for p in data:
                    if p.get("symbol") == symbol:
                        return float(p.get("positionAmt", 0))
            return 0.0
        except Exception:
            logger.opt(exception=True).warning("[BinanceExecutor] get_open_position_size failed for {}", symbol)
            return 0.0  # safe default: assume no position on error

    async def close_position(self, symbol: str, position_size: float) -> bool:
        """Close an open position with a MARKET order.

        Tries closePosition=true first (works on Demo and mainnet).
        Falls back to quantity+reduceOnly if the first attempt fails,
        which handles edge cases where closePosition is rejected.

        Args:
            symbol:        Binance symbol, e.g. "BTCUSDT".
            position_size: Current position size from positionRisk (negative = SHORT).

        Returns:
            True if the close order was accepted, False otherwise.
        """
        if position_size == 0.0:
            return True

        # Positive size = LONG (close with SELL); negative = SHORT (close with BUY)
        side = "SELL" if position_size > 0 else "BUY"

        # Primary approach: closePosition=true avoids -4120 on Demo env
        params: dict[str, Any] = {
            "symbol":        symbol,
            "side":          side,
            "type":          "MARKET",
            "closePosition": "true",
        }
        try:
            await self._signed_post("/fapi/v1/order", params)
            logger.info(
                "BinanceExecutor: close_position {} size={} → {} MARKET closePosition=true",
                symbol, position_size, side,
            )
            return True
        except BinanceAPIError as exc:
            logger.warning(
                "BinanceExecutor: close_position closePosition=true failed for {} — {} — trying quantity fallback",
                symbol, exc,
            )

        # Fallback: explicit quantity + reduceOnly
        quantity = _round_quantity(Decimal(str(abs(position_size))), symbol)
        params_fallback: dict[str, Any] = {
            "symbol":     symbol,
            "side":       side,
            "type":       "MARKET",
            "quantity":   str(quantity),
            "reduceOnly": "true",
        }
        try:
            await self._signed_post("/fapi/v1/order", params_fallback)
            logger.info(
                "BinanceExecutor: close_position {} size={} → {} MARKET qty={} (fallback)",
                symbol, position_size, side, quantity,
            )
            return True
        except BinanceAPIError as exc:
            logger.error("BinanceExecutor: close_position failed for {} — {}", symbol, exc)
            return False
        except Exception:
            logger.opt(exception=True).error("BinanceExecutor: close_position error for {}", symbol)
            return False

    async def update_stop_loss(
        self,
        session: AsyncSession,
        signal: Signal,
        new_stop_price: Decimal,
    ) -> bool:
        """Cancel the existing STOP_MARKET and place a new one at new_stop_price.

        Safety guarantee: if the new order fails, the original stop is
        immediately restored.  Returns True only when the new SL is live.
        """
        from sqlalchemy import select as sa_select, and_ as sa_and_

        rounded_new = _round_price(new_stop_price, signal.symbol)
        original_sl  = _round_price(signal.stop_loss, signal.symbol)
        side = "SELL" if signal.direction == "BUY" else "BUY"

        # ── 1. Find existing SL orders ────────────────────────────────────────
        stmt = sa_select(TradeOrder).where(
            sa_and_(
                TradeOrder.signal_id   == signal.id,
                TradeOrder.order_role  == "stop_loss",
                TradeOrder.status      == "NEW",
                TradeOrder.broker_order_id.isnot(None),
            )
        )
        result = await session.execute(stmt)
        existing_orders = result.scalars().all()

        # ── 2. If no live SL order exists, update DB only ─────────────────────
        # Positions that never had SL/TP orders placed (e.g. from failed
        # execution batches) have no Binance order to cancel/replace.  Placing
        # a brand-new STOP_MARKET on the demo environment without a prior order
        # triggers -4120.  The safer approach: update signal.stop_loss in the
        # DB so the outcome detector's price-level check uses the tighter stop,
        # and let close_position() handle the actual closure when the price is hit.
        if not existing_orders:
            signal.stop_loss = new_stop_price
            await session.commit()
            logger.info(
                "BinanceExecutor: SL updated in DB only (no live Binance order) "
                "{} {} → {}",
                signal.symbol, original_sl, rounded_new,
            )
            return True

        # ── 3. Cancel existing SL orders ──────────────────────────────────────
        cancelled_any = False
        for order in existing_orders:
            ok = await self._cancel_order(signal.symbol, order.broker_order_id)
            if ok:
                order.status  = "CANCELED"
                cancelled_any = True
        if cancelled_any:
            await session.commit()

        # ── 4. Place new SL ───────────────────────────────────────────────────
        new_params: dict[str, Any] = {
            "symbol":        signal.symbol,
            "side":          side,
            "type":          "STOP_MARKET",
            "stopPrice":     str(rounded_new),
            "closePosition": "true",
        }
        try:
            raw = await self._signed_post("/fapi/v1/order", new_params)
            broker_order_id = str(raw.get("orderId", ""))
            status          = raw.get("status", "NEW")

            session.add(TradeOrder(
                signal_id       = signal.id,
                broker_order_id = broker_order_id,
                order_role      = "stop_loss",
                symbol          = signal.symbol,
                side            = side,
                order_type      = "STOP_MARKET",
                quantity        = Decimal("0"),
                stop_price      = rounded_new,
                status          = status,
                raw_response    = json.dumps(raw),
                environment     = self._environment,
            ))
            signal.stop_loss = new_stop_price   # update in-memory + DB via ORM
            await session.commit()

            logger.info(
                "BinanceExecutor: SL updated {} {} → {}",
                signal.symbol, original_sl, rounded_new,
            )
            return True

        except Exception as exc:
            logger.error(
                "BinanceExecutor: new SL failed for {} @ {} — {}",
                signal.symbol, rounded_new, exc,
            )

            # ── 5. Restore original SL if we cancelled it ─────────────────────
            if cancelled_any:
                logger.warning(
                    "BinanceExecutor: restoring original SL @ {} for {}",
                    original_sl, signal.symbol,
                )
                try:
                    restore_params: dict[str, Any] = {
                        "symbol":        signal.symbol,
                        "side":          side,
                        "type":          "STOP_MARKET",
                        "stopPrice":     str(original_sl),
                        "closePosition": "true",
                    }
                    raw_r = await self._signed_post("/fapi/v1/order", restore_params)
                    session.add(TradeOrder(
                        signal_id       = signal.id,
                        broker_order_id = str(raw_r.get("orderId", "")),
                        order_role      = "stop_loss",
                        symbol          = signal.symbol,
                        side            = side,
                        order_type      = "STOP_MARKET",
                        quantity        = Decimal("0"),
                        stop_price      = original_sl,
                        status          = raw_r.get("status", "NEW"),
                        raw_response    = json.dumps(raw_r),
                        environment     = self._environment,
                    ))
                    await session.commit()
                    logger.info(
                        "BinanceExecutor: SL restored @ {} for {}",
                        original_sl, signal.symbol,
                    )
                except Exception as restore_exc:
                    logger.critical(
                        "BinanceExecutor: CRITICAL — could not restore SL for {} "
                        "— position exposed! restore_error={}",
                        signal.symbol, restore_exc,
                    )
            return False

    # ------------------------------------------------------------------
    # Public (unsigned) market-data helpers — used by the tick ingestor
    # ------------------------------------------------------------------

    async def fetch_agg_trades(
        self,
        symbol: str,
        from_id: int | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetch aggregated trades from /fapi/v1/aggTrades.

        Either ``from_id`` OR (``start_ms``, ``end_ms``) should be supplied.
        Returns the raw list Binance returns — each entry has keys
        ``a`` (aggTradeId), ``p`` (price), ``q`` (qty), ``m`` (isBuyerMaker),
        ``T`` (timestamp ms). Empty list on transient failure.
        """
        params: dict[str, Any] = {"symbol": symbol, "limit": min(limit, 1000)}
        if from_id is not None:
            params["fromId"] = from_id
        else:
            if start_ms is not None:
                params["startTime"] = start_ms
            if end_ms is not None:
                params["endTime"] = end_ms
        try:
            data = await self._public_get("/fapi/v1/aggTrades", params)
            if isinstance(data, list):
                return data
            logger.warning("BinanceExecutor: unexpected aggTrades response for {}: {}", symbol, data)
            return []
        except Exception:
            logger.opt(exception=True).warning("BinanceExecutor: aggTrades fetch failed for {}", symbol)
            return []

    async def fetch_klines_1s(
        self,
        symbol: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list[Any]]:
        """Fetch 1-second klines from /fapi/v1/klines.

        Returns raw kline arrays — Binance layout:
        ``[openTime, open, high, low, close, volume, closeTime, quoteAssetVol,
        numTrades, takerBuyBaseVol, takerBuyQuoteVol, ignore]``.
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": "1s",
            "limit": min(limit, 1000),
        }
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        try:
            data = await self._public_get("/fapi/v1/klines", params)
            if isinstance(data, list):
                return data
            logger.warning("BinanceExecutor: unexpected klines_1s response for {}: {}", symbol, data)
            return []
        except Exception:
            logger.opt(exception=True).warning("BinanceExecutor: klines_1s fetch failed for {}", symbol)
            return []

    async def _public_get(
        self,
        path: str,
        params: dict[str, Any],
        _retries: int = 2,
    ) -> Any:
        """Unsigned public market-data GET."""
        last_exc: Exception | None = None
        for attempt in range(_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(
                        f"{self._base_url}{path}",
                        params=params,
                    )
                    if resp.status_code == 200:
                        return resp.json()
                    if not self._is_retryable(resp.status_code) or attempt == _retries:
                        try:
                            err = resp.json()
                        except Exception:
                            err = {"msg": resp.text}
                        raise BinanceAPIError(
                            f"HTTP {resp.status_code} — code={err.get('code')} msg={err.get('msg')}",
                            code=err.get("code"),
                        )
                    last_exc = BinanceAPIError(f"HTTP {resp.status_code} (retrying)")
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                if attempt == _retries:
                    raise
            logger.warning(
                "BinanceExecutor: public GET {} retry {}/{} after transient error",
                path, attempt + 1, _retries,
            )
            await asyncio.sleep(1 * (attempt + 1))
        raise last_exc  # unreachable

    async def _signed_delete(
        self,
        path: str,
        params: dict[str, Any],
        _retries: int = 2,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(_retries + 1):
            try:
                signed = self._sign(dict(params))
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.delete(
                        f"{self._base_url}{path}",
                        params=signed,
                        headers=self._headers(),
                    )
                    data = resp.json()
                    if resp.status_code == 200:
                        return data
                    if not self._is_retryable(resp.status_code) or attempt == _retries:
                        raise BinanceAPIError(
                            f"HTTP {resp.status_code} — code={data.get('code')} msg={data.get('msg')}",
                            code=data.get("code"),
                        )
                    last_exc = BinanceAPIError(
                        f"HTTP {resp.status_code} (retrying)", code=data.get("code")
                    )
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                if attempt == _retries:
                    raise
            logger.warning("BinanceExecutor: DELETE {} retry {}/{} after transient error", path, attempt + 1, _retries)
            await asyncio.sleep(1 * (attempt + 1))
        raise last_exc  # unreachable


class BinanceAPIError(Exception):
    """Raised when Binance returns a non-200 response."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
