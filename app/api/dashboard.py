"""Dashboard API endpoint — comprehensive interactive trading dashboard."""

import asyncio
import datetime
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import case, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.backtest_result import BacktestResult
from app.models.candle import Candle
from app.models.claude_decision import ClaudeDecision
from app.models.optimized_params import OptimizedParams
from app.models.outcome import Outcome
from app.models.signal import Signal
from app.models.strategy import Strategy
from app.models.strategy_performance import StrategyPerformance
from app.workers.scheduler import scheduler

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

templates = Jinja2Templates(
    directory=str(Path(__file__).resolve().parent.parent / "templates")
)

_AUTO_DIR = Path(__file__).resolve().parent.parent / "strategies" / "auto"

# Track app start time for uptime
_start_time: datetime.datetime = datetime.datetime.now(datetime.UTC)

# In-memory backtest job tracking
_backtest_jobs: dict[str, dict[str, Any]] = {}


# ── HTML page ─────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Serve the dashboard HTML page."""
    return templates.TemplateResponse(request=request, name="dashboard.html")


# ── Overview data ─────────────────────────────────────────────────────────────

@router.get("/data")
async def dashboard_data(
    session: AsyncSession = Depends(get_session),
):
    """Return all dashboard data as a single JSON payload."""
    now = datetime.datetime.now(datetime.UTC)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    uptime = (now - _start_time).total_seconds()

    db_status = "connected"
    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    scheduler_status = "running" if scheduler.running else "stopped"

    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger": str(job.trigger),
        })

    active_signals = 0
    signals_today = 0
    total_signals = 0
    try:
        r = await session.execute(
            select(func.count()).select_from(Signal).where(Signal.status == "active")
        )
        active_signals = r.scalar_one()
        r = await session.execute(
            select(func.count()).select_from(Signal).where(Signal.created_at >= today_start)
        )
        signals_today = r.scalar_one()
        r = await session.execute(select(func.count()).select_from(Signal))
        total_signals = r.scalar_one()
    except Exception:
        pass

    recent_signals = []
    try:
        query = (
            select(Signal, Outcome, Strategy.name)
            .outerjoin(Outcome, Signal.id == Outcome.signal_id)
            .outerjoin(Strategy, Signal.strategy_id == Strategy.id)
            .order_by(Signal.created_at.desc())
            .limit(20)
        )
        result = await session.execute(query)
        for signal, outcome, strategy_name in result.all():
            recent_signals.append({
                "id": signal.id,
                "symbol": signal.symbol,
                "direction": signal.direction,
                "entry": float(signal.entry_price),
                "sl": float(signal.stop_loss),
                "tp1": float(signal.take_profit_1),
                "tp2": float(signal.take_profit_2) if signal.take_profit_2 else None,
                "rr": float(signal.risk_reward),
                "confidence": float(signal.confidence),
                "status": signal.status,
                "strategy": strategy_name or "Unknown",
                "created": signal.created_at.isoformat() if signal.created_at else None,
                "result": outcome.result if outcome else None,
                "pnl": float(outcome.pnl_usdt) if outcome and outcome.pnl_usdt else None,
            })
    except Exception:
        pass

    last_signal_generated = None
    try:
        r = await session.execute(select(func.max(Signal.created_at)))
        ts = r.scalar_one()
        if ts:
            last_signal_generated = ts.isoformat()
    except Exception:
        pass

    wins = 0
    losses = 0
    total_pnl = 0.0
    try:
        r = await session.execute(
            select(
                func.count().filter(Outcome.result.in_(["tp1_hit", "tp2_hit", "tp_hit"])).label("wins"),
                func.count().filter(Outcome.result == "sl_hit").label("losses"),
                func.coalesce(func.sum(Outcome.pnl_usdt), 0).label("total_pnl"),
            ).select_from(Outcome)
        )
        row = r.one()
        wins = row.wins
        losses = row.losses
        total_pnl = float(row.total_pnl)
    except Exception:
        pass

    strategies = []
    try:
        query = (
            select(Strategy.name, StrategyPerformance)
            .join(Strategy, StrategyPerformance.strategy_id == Strategy.id)
            .where(StrategyPerformance.period == "30d")
            .order_by(StrategyPerformance.win_rate.desc())
        )
        result = await session.execute(query)
        for name, perf in result.all():
            strategies.append({
                "name": name,
                "win_rate": float(perf.win_rate),
                "profit_factor": float(perf.profit_factor),
                "avg_rr": float(perf.avg_rr),
                "total_signals": perf.total_signals,
                "is_degraded": perf.is_degraded,
            })
    except Exception:
        pass

    last_candle = None
    try:
        r = await session.execute(select(func.max(Candle.timestamp)))
        ts = r.scalar_one()
        if ts:
            last_candle = ts.isoformat()
    except Exception:
        pass

    backtests = []
    total_backtests = 0
    walk_forward = []
    opt_params_list = []
    try:
        r = await session.execute(select(func.count()).select_from(BacktestResult))
        total_backtests = r.scalar_one()

        from sqlalchemy import and_

        latest_sub = (
            select(
                BacktestResult.strategy_id,
                BacktestResult.window_days,
                func.max(BacktestResult.created_at).label("max_created"),
            )
            .where(BacktestResult.is_walk_forward.isnot(True))
            .group_by(BacktestResult.strategy_id, BacktestResult.window_days)
            .subquery()
        )
        bt_query = (
            select(BacktestResult, Strategy.name)
            .join(Strategy, BacktestResult.strategy_id == Strategy.id)
            .join(latest_sub, and_(
                BacktestResult.strategy_id == latest_sub.c.strategy_id,
                BacktestResult.window_days == latest_sub.c.window_days,
                BacktestResult.created_at == latest_sub.c.max_created,
            ))
            .order_by(Strategy.name, BacktestResult.window_days)
        )
        result = await session.execute(bt_query)
        for bt, strat_name in result.all():
            backtests.append({
                "strategy": strat_name,
                "window_days": bt.window_days,
                "win_rate": float(bt.win_rate) if bt.win_rate is not None else None,
                "profit_factor": float(bt.profit_factor) if bt.profit_factor is not None else None,
                "sharpe_ratio": float(bt.sharpe_ratio) if bt.sharpe_ratio is not None else None,
                "max_drawdown": float(bt.max_drawdown) if bt.max_drawdown is not None else None,
                "expectancy": float(bt.expectancy) if bt.expectancy is not None else None,
                "total_trades": bt.total_trades,
                "is_walk_forward": bt.is_walk_forward or False,
                "is_overfitted": bt.is_overfitted,
                "created": bt.created_at.isoformat() if bt.created_at else None,
            })

        wf_latest_sub = (
            select(
                BacktestResult.strategy_id,
                func.max(BacktestResult.created_at).label("max_created"),
            )
            .where(BacktestResult.is_walk_forward.is_(True))
            .group_by(BacktestResult.strategy_id)
            .subquery()
        )
        wf_query = (
            select(BacktestResult, Strategy.name)
            .join(Strategy, BacktestResult.strategy_id == Strategy.id)
            .join(wf_latest_sub, and_(
                BacktestResult.strategy_id == wf_latest_sub.c.strategy_id,
                BacktestResult.created_at == wf_latest_sub.c.max_created,
            ))
            .order_by(Strategy.name)
        )
        result = await session.execute(wf_query)
        for bt, strat_name in result.all():
            walk_forward.append({
                "strategy": strat_name,
                "win_rate": float(bt.win_rate) if bt.win_rate is not None else None,
                "profit_factor": float(bt.profit_factor) if bt.profit_factor is not None else None,
                "total_trades": bt.total_trades,
                "is_overfitted": bt.is_overfitted,
                "wfe": float(bt.walk_forward_efficiency) if bt.walk_forward_efficiency is not None else None,
                "created": bt.created_at.isoformat() if bt.created_at else None,
            })

        opt_latest_sub = (
            select(
                OptimizedParams.strategy_name,
                func.max(OptimizedParams.created_at).label("max_created"),
            )
            .where(OptimizedParams.is_active.is_(True))
            .group_by(OptimizedParams.strategy_name)
            .subquery()
        )
        opt_query = (
            select(OptimizedParams)
            .join(opt_latest_sub, and_(
                OptimizedParams.strategy_name == opt_latest_sub.c.strategy_name,
                OptimizedParams.created_at == opt_latest_sub.c.max_created,
            ))
            .order_by(OptimizedParams.strategy_name)
        )
        result = await session.execute(opt_query)
        for opt in result.scalars().all():
            opt_params_list.append({
                "strategy": opt.strategy_name,
                "win_rate": float(opt.win_rate) if opt.win_rate is not None else None,
                "profit_factor": float(opt.profit_factor) if opt.profit_factor is not None else None,
                "total_trades": opt.total_trades,
                "wfe_ratio": float(opt.wfe_ratio) if opt.wfe_ratio is not None else None,
                "is_overfitted": opt.is_overfitted,
                "combinations_tested": opt.combinations_tested,
                "created": opt.created_at.isoformat() if opt.created_at else None,
            })
    except Exception:
        pass

    return {
        "system": {
            "status": "operational" if db_status == "connected" and scheduler_status == "running" else "degraded",
            "database": db_status,
            "scheduler": scheduler_status,
            "uptime_seconds": round(uptime, 1),
            "last_candle": last_candle,
            "last_signal_generated": last_signal_generated,
            "timestamp": now.isoformat(),
        },
        "jobs": jobs,
        "signals": {
            "active": active_signals,
            "today": signals_today,
            "total": total_signals,
            "recent": recent_signals,
        },
        "performance": {
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
            "total_pnl": round(total_pnl, 2),
        },
        "strategies": strategies,
        "backtests": {
            "total": total_backtests,
            "results": backtests,
            "walk_forward": walk_forward,
            "optimized_params": opt_params_list,
        },
    }


# ── Trade Journal ─────────────────────────────────────────────────────────────

@router.get("/trades")
async def get_trades(
    page: int = 1,
    limit: int = 25,
    symbol: str = "",
    result_filter: str = "",
    session: AsyncSession = Depends(get_session),
):
    """Paginated trade journal with Claude reasoning."""
    offset = (page - 1) * limit

    try:
        query = (
            select(Signal, Outcome, Strategy.name)
            .outerjoin(Outcome, Signal.id == Outcome.signal_id)
            .outerjoin(Strategy, Signal.strategy_id == Strategy.id)
        )
        count_query = select(func.count()).select_from(Signal)

        if symbol:
            query = query.where(Signal.symbol == symbol.upper())
            count_query = count_query.where(Signal.symbol == symbol.upper())
        if result_filter:
            query = query.join(Outcome, Signal.id == Outcome.signal_id, isouter=False)
            query = query.where(Outcome.result == result_filter)

        total_r = await session.execute(count_query)
        total = total_r.scalar_one()

        query = query.order_by(Signal.created_at.desc()).offset(offset).limit(limit)
        result = await session.execute(query)
        rows = result.all()

        trades = []
        for signal, outcome, strategy_name in rows:
            # Find Claude's decision closest to signal time
            reasoning = None
            if signal.created_at:
                from datetime import timedelta
                window_start = signal.created_at - timedelta(minutes=10)
                window_end   = signal.created_at + timedelta(minutes=2)
                dec_r = await session.execute(
                    select(ClaudeDecision)
                    .where(
                        ClaudeDecision.symbol == signal.symbol,
                        ClaudeDecision.action.in_(["open_long", "open_short"]),
                        ClaudeDecision.created_at >= window_start,
                        ClaudeDecision.created_at <= window_end,
                    )
                    .order_by(ClaudeDecision.created_at.desc())
                    .limit(1)
                )
                dec = dec_r.scalar_one_or_none()
                if dec:
                    reasoning = dec.reasoning

            trades.append({
                "id": signal.id,
                "symbol": signal.symbol,
                "direction": signal.direction,
                "entry": float(signal.entry_price),
                "sl": float(signal.stop_loss),
                "tp1": float(signal.take_profit_1),
                "tp2": float(signal.take_profit_2) if signal.take_profit_2 else None,
                "rr": float(signal.risk_reward),
                "confidence": float(signal.confidence),
                "status": signal.status,
                "strategy": strategy_name or "Unknown",
                "created": signal.created_at.isoformat() if signal.created_at else None,
                "result": outcome.result if outcome else None,
                "pnl": float(outcome.pnl_usdt) if outcome and outcome.pnl_usdt else None,
                "reasoning": reasoning,
            })

        return {"total": total, "page": page, "limit": limit, "trades": trades}

    except Exception as e:
        return {"total": 0, "page": page, "limit": limit, "trades": [], "error": str(e)}


# ── P&L Equity Curve ─────────────────────────────────────────────────────────

@router.get("/pnl")
async def get_pnl(
    days: int = 30,
    session: AsyncSession = Depends(get_session),
):
    """P&L data points for equity curve chart."""
    try:
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
        result = await session.execute(
            select(Outcome.created_at, Outcome.pnl_usdt, Signal.symbol, Signal.direction)
            .join(Signal, Outcome.signal_id == Signal.id)
            .where(Outcome.created_at >= cutoff, Outcome.pnl_usdt != None)
            .order_by(Outcome.created_at.asc())
        )
        rows = result.all()

        cumulative = 0.0
        points = []
        daily: dict[str, float] = {}

        for created_at, pnl, symbol, direction in rows:
            pnl_f = float(pnl)
            cumulative += pnl_f
            ts = created_at.isoformat() if created_at else None
            points.append({"t": ts, "y": round(cumulative, 2), "pnl": round(pnl_f, 2), "symbol": symbol})
            day = created_at.strftime("%Y-%m-%d") if created_at else "unknown"
            daily[day] = round(daily.get(day, 0) + pnl_f, 2)

        daily_bars = [{"day": d, "pnl": v} for d, v in sorted(daily.items())]

        return {"points": points, "daily": daily_bars, "total_pnl": round(cumulative, 2)}
    except Exception as e:
        return {"points": [], "daily": [], "total_pnl": 0, "error": str(e)}


# ── Claude Decisions Log ──────────────────────────────────────────────────────

@router.get("/decisions")
async def get_decisions(
    page: int = 1,
    limit: int = 50,
    action_filter: str = "",
    symbol: str = "",
    session: AsyncSession = Depends(get_session),
):
    """Claude decision log with full reasoning."""
    offset = (page - 1) * limit
    try:
        query = select(ClaudeDecision)
        count_query = select(func.count()).select_from(ClaudeDecision)

        if action_filter:
            query = query.where(ClaudeDecision.action == action_filter)
            count_query = count_query.where(ClaudeDecision.action == action_filter)
        if symbol:
            query = query.where(ClaudeDecision.symbol == symbol.upper())
            count_query = count_query.where(ClaudeDecision.symbol == symbol.upper())

        total_r = await session.execute(count_query)
        total = total_r.scalar_one()

        query = query.order_by(ClaudeDecision.created_at.desc()).offset(offset).limit(limit)
        result = await session.execute(query)

        decisions = []
        for dec in result.scalars().all():
            decisions.append({
                "id": dec.id,
                "symbol": dec.symbol,
                "action": dec.action,
                "confidence": float(dec.confidence) if dec.confidence else None,
                "entry": float(dec.entry_price) if dec.entry_price else None,
                "sl": float(dec.stop_loss) if dec.stop_loss else None,
                "tp": float(dec.take_profit) if dec.take_profit else None,
                "executed": dec.executed,
                "execution_error": dec.execution_error,
                "reasoning": dec.reasoning,
                "created": dec.created_at.isoformat() if dec.created_at else None,
            })

        return {"total": total, "page": page, "limit": limit, "decisions": decisions}
    except Exception as e:
        return {"total": 0, "page": page, "limit": limit, "decisions": [], "error": str(e)}


# ── Self-Improvement Insights ─────────────────────────────────────────────────

@router.get("/insights")
async def get_insights(
    session: AsyncSession = Depends(get_session),
):
    """Latest self-improvement insights."""
    try:
        result = await session.execute(
            select(ClaudeDecision)
            .where(ClaudeDecision.action == "self_analysis")
            .order_by(ClaudeDecision.created_at.desc())
            .limit(10)
        )
        insights = []
        for dec in result.scalars().all():
            import json
            import re as _re
            raw = dec.reasoning or ""
            parsed = None
            m = _re.search(r"json=(\{.*\})", raw, _re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception:
                    pass
            insights.append({
                "id": dec.id,
                "created": dec.created_at.isoformat() if dec.created_at else None,
                "raw": raw,
                "parsed": parsed,
            })
        return {"insights": insights}
    except Exception as e:
        return {"insights": [], "error": str(e)}


# ── Auto Strategies (Strategy Lab) ───────────────────────────────────────────

@router.get("/auto-strategies")
async def list_auto_strategies():
    """List pending and live auto-generated strategies."""
    pending = []
    live = []

    if _AUTO_DIR.exists():
        for f in sorted(_AUTO_DIR.glob("*.py")):
            if f.name == "__init__.py":
                continue
            name = f.stem
            code_snippet = ""
            try:
                lines = f.read_text().splitlines()
                code_snippet = "\n".join(lines[:30])
            except Exception:
                pass

            stat = f.stat()
            created = datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.UTC).isoformat()

            entry = {
                "filename": f.name,
                "name": name,
                "created": created,
                "size_kb": round(stat.st_size / 1024, 1),
                "code_snippet": code_snippet,
            }

            if name.startswith("_pending_"):
                strategy_name = name[len("_pending_"):]
                entry["strategy_name"] = strategy_name
                pending.append(entry)
            else:
                entry["strategy_name"] = name
                live.append(entry)

    # Also list strategies from the main registry
    try:
        from app.strategies.base import _STRATEGY_REGISTRY
        registry_names = list(_STRATEGY_REGISTRY.keys())
    except Exception:
        registry_names = []

    return {"pending": pending, "live": live, "registry": registry_names}


@router.post("/auto-strategies/{filename}/approve")
async def approve_strategy(filename: str):
    """Approve a pending strategy — rename from _pending_X to X."""
    import re as _re
    safe = _re.sub(r"[^a-zA-Z0-9_.]", "", filename)
    pending_path = _AUTO_DIR / safe

    if not pending_path.exists():
        raise HTTPException(status_code=404, detail="Pending strategy file not found")

    if not safe.startswith("_pending_"):
        raise HTTPException(status_code=400, detail="File is not a pending strategy")

    strategy_name = safe[len("_pending_"):]
    approved_path = _AUTO_DIR / strategy_name

    try:
        code = pending_path.read_text()
        # Add approval header
        header = (
            "# Auto-generated strategy — manually approved via dashboard\n"
            "# Passed walk-forward blind backtest\n"
        )
        approved_path.write_text(header + code)
        pending_path.unlink()
        return {"ok": True, "message": f"Strategy '{strategy_name}' approved and activated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/auto-strategies/{filename}")
async def reject_strategy(filename: str):
    """Reject (delete) a strategy file."""
    import re as _re
    safe = _re.sub(r"[^a-zA-Z0-9_.]", "", filename)
    path = _AUTO_DIR / safe

    if not path.exists():
        raise HTTPException(status_code=404, detail="Strategy file not found")

    try:
        path.unlink()
        return {"ok": True, "message": f"Strategy '{safe}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auto-strategies/{filename}/source")
async def get_strategy_source(filename: str):
    """Return full source code of a strategy file."""
    import re as _re
    safe = _re.sub(r"[^a-zA-Z0-9_.]", "", filename)
    path = _AUTO_DIR / safe

    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return {"filename": safe, "source": path.read_text()}


# ── Order Flow ────────────────────────────────────────────────────────────────

@router.get("/orderflow/{symbol}")
async def get_orderflow(symbol: str):
    """Fetch live order flow for a symbol."""
    try:
        from app.services.order_flow import OrderFlowAnalyzer
        analyzer = OrderFlowAnalyzer()
        summary = await analyzer.fetch(symbol.upper(), lookback_minutes=90)
        if summary is None:
            return {"symbol": symbol.upper(), "available": False}

        return {
            "symbol": symbol.upper(),
            "available": True,
            "bias": summary.pressure,
            "buy_ratio": round(summary.buy_ratio, 4),
            "sell_ratio": round(summary.sell_ratio, 4),
            "delta_30m": round(summary.delta_30m, 2),
            "delta_60m": round(summary.delta_60m, 2),
            "cvd_trend": summary.cvd_trend,
            "large_buys": summary.large_buys,
            "large_sells": summary.large_sells,
            "total_volume_usdt": round(summary.total_volume_usdt, 2),
            "context": summary.to_context_string(),
        }
    except Exception as e:
        return {"symbol": symbol.upper(), "available": False, "error": str(e)}


# ── Walk-Forward Backtest (on-demand) ─────────────────────────────────────────

@router.get("/backtest/strategies")
async def list_backtest_strategies():
    """List all registered strategies available for backtesting."""
    try:
        from app.strategies import (  # noqa: F401 — trigger auto-registration
            CryptoBreakoutStrategy, CryptoMomentumStrategy, CryptoSLCStrategy,
        )
        from app.strategies.base import _STRATEGY_REGISTRY
        result = []
        for name, cls in _STRATEGY_REGISTRY.items():
            result.append({
                "name": name,
                "class": cls.__name__,
                "has_param_grid": bool(getattr(cls, "PARAM_GRID", None)),
                "default_params": getattr(cls, "DEFAULT_PARAMS", {}),
            })
        return {"strategies": result}
    except Exception as e:
        return {"strategies": [], "error": str(e)}


@router.post("/backtest/run")
async def run_backtest(
    payload: dict,
    background_tasks: BackgroundTasks,
):
    """Trigger a walk-forward backtest for a strategy. Runs in background."""
    strategy_name = payload.get("strategy_name", "")
    if not strategy_name:
        raise HTTPException(status_code=400, detail="strategy_name is required")

    try:
        from app.strategies.base import _STRATEGY_REGISTRY
        if strategy_name not in _STRATEGY_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Strategy '{strategy_name}' not found in registry")
        strategy_cls = _STRATEGY_REGISTRY[strategy_name]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    job_id = f"{strategy_name}_{datetime.datetime.now(datetime.UTC).strftime('%H%M%S')}"
    _backtest_jobs[job_id] = {"status": "running", "strategy": strategy_name, "result": None, "error": None}

    async def _run():
        from app.services.walk_forward_backtester import WalkForwardBacktester
        try:
            backtester = WalkForwardBacktester()
            result = await backtester.run(strategy_cls)
            _backtest_jobs[job_id]["status"] = "done"
            _backtest_jobs[job_id]["result"] = {
                "strategy_name": result.strategy_name,
                "symbols": result.symbols,
                "total_folds": result.total_folds,
                "blind_trades": result.blind_trades,
                "blind_wins": result.blind_wins,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "total_pnl_pct": result.total_pnl_pct,
                "passed": result.passed,
                "summary": result.summary(),
                "folds": [
                    {
                        "fold": f.fold,
                        "blind_start": f.blind_start.isoformat(),
                        "blind_end": f.blind_end.isoformat(),
                        "win_rate": f.win_rate,
                        "profit_factor": f.profit_factor,
                        "trades": len(f.trades),
                        "wins": sum(1 for t in f.trades if t.result == "tp_hit"),
                        "best_params": f.best_params,
                    }
                    for f in result.fold_results
                ],
            }
        except Exception as e:
            _backtest_jobs[job_id]["status"] = "error"
            _backtest_jobs[job_id]["error"] = str(e)

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "running", "strategy": strategy_name}


@router.get("/backtest/status/{job_id}")
async def backtest_job_status(job_id: str):
    """Poll status of a running backtest."""
    if job_id not in _backtest_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _backtest_jobs[job_id]


@router.get("/backtest/jobs")
async def list_backtest_jobs():
    """List all backtest jobs (running + completed)."""
    return {"jobs": [{"job_id": k, **v} for k, v in _backtest_jobs.items()]}
