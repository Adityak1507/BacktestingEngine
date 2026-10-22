"""HTTP API for the backtesting engine, consumed by the React frontend.

Run in development with:
    uvicorn api.main:app --reload

If `frontend/dist` exists (after `npm run build`), it is served at `/`, so a
single process hosts both the API and the UI.
"""

from __future__ import annotations

import io
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from backtester import (
    Backtest,
    BuyAndHold,
    RsiMeanReversion,
    SmaCrossover,
    generate_gbm,
    load_csv,
    load_yahoo,
)
from backtester.fast import HAS_NUMBA, sma_grid_search

MAX_SWEEP_PAIRS = 50_000
MAX_CSV_BYTES = 20 * 1024 * 1024

app = FastAPI(title="Backtesting Engine API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- request models --------------------------------------------------------
class DataSpec(BaseModel):
    source: Literal["synthetic", "yahoo", "csv"] = "synthetic"
    # synthetic
    days: int = Field(1260, ge=60, le=10_000)
    drift: float = Field(0.08, ge=-1, le=1)
    volatility: float = Field(0.2, gt=0, le=2)
    seed: int = 42
    # yahoo
    ticker: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    # csv
    csv_text: Optional[str] = Field(None, max_length=MAX_CSV_BYTES)
    csv_name: Optional[str] = None

    @model_validator(mode="after")
    def _check_source(self):
        if self.source == "yahoo" and not (self.ticker or "").strip():
            raise ValueError("ticker is required for Yahoo Finance data")
        if self.source == "csv" and not self.csv_text:
            raise ValueError("csv_text is required for CSV data")
        return self


class Costs(BaseModel):
    initial_cash: float = Field(100_000, gt=0)
    commission_bps: float = Field(10, ge=0, le=1000)
    slippage_bps: float = Field(5, ge=0, le=1000)

    @property
    def kwargs(self) -> Dict[str, float]:
        return {
            "initial_cash": self.initial_cash,
            "commission": self.commission_bps / 10_000,
            "slippage": self.slippage_bps / 10_000,
        }


class StrategySpec(BaseModel):
    name: Literal["sma_crossover", "rsi_mean_reversion", "buy_and_hold"] = "sma_crossover"
    params: Dict[str, float] = Field(default_factory=dict)


class BacktestRequest(BaseModel):
    data: DataSpec = Field(default_factory=DataSpec)
    strategy: StrategySpec = Field(default_factory=StrategySpec)
    costs: Costs = Field(default_factory=Costs)


class SweepRequest(BaseModel):
    data: DataSpec = Field(default_factory=DataSpec)
    costs: Costs = Field(default_factory=Costs)
    fast_min: int = Field(5, ge=2)
    fast_max: int = Field(50, ge=2)
    slow_min: int = Field(20, ge=3)
    slow_max: int = Field(200, ge=3)
    step: int = Field(5, ge=1)


# --- strategy registry -----------------------------------------------------
STRATEGIES: List[Dict[str, Any]] = [
    {
        "id": "sma_crossover",
        "name": "SMA crossover",
        "description": "Long while the fast moving average is above the slow one.",
        "params": [
            {"key": "fast", "label": "Fast window", "default": 20, "min": 2, "max": 400, "step": 1},
            {"key": "slow", "label": "Slow window", "default": 50, "min": 3, "max": 500, "step": 1},
        ],
    },
    {
        "id": "rsi_mean_reversion",
        "name": "RSI mean reversion",
        "description": "Buy when RSI is oversold, exit once it recovers.",
        "params": [
            {"key": "period", "label": "RSI period", "default": 14, "min": 2, "max": 100, "step": 1},
            {"key": "oversold", "label": "Buy below", "default": 30, "min": 1, "max": 60, "step": 1},
            {"key": "exit_level", "label": "Sell above", "default": 55, "min": 30, "max": 99, "step": 1},
        ],
    },
    {
        "id": "buy_and_hold",
        "name": "Buy & hold",
        "description": "Invest on the first bar and never sell.",
        "params": [],
    },
]


def build_strategy(spec: StrategySpec):
    schema = {p["key"]: p for s in STRATEGIES if s["id"] == spec.name for p in s["params"]}
    unknown = sorted(set(spec.params) - set(schema))
    if unknown:
        raise HTTPException(422, f"Unknown parameter(s) for {spec.name}: {', '.join(unknown)}.")
    p = {key: param["default"] for key, param in schema.items()}
    p.update(spec.params)
    for key, value in p.items():
        lo, hi = schema[key]["min"], schema[key]["max"]
        if not lo <= value <= hi:
            raise HTTPException(422, f"{schema[key]['label']} must be between {lo} and {hi}.")
    if spec.name == "rsi_mean_reversion" and p["oversold"] >= p["exit_level"]:
        raise HTTPException(422, "The buy level must be below the sell level.")
    try:
        if spec.name == "sma_crossover":
            return SmaCrossover(int(p["fast"]), int(p["slow"]))
        if spec.name == "rsi_mean_reversion":
            return RsiMeanReversion(int(p["period"]), float(p["oversold"]), float(p["exit_level"]))
    except ValueError as exc:
        raise HTTPException(422, f"{str(exc)[0].upper()}{str(exc)[1:]}.") from exc
    return BuyAndHold()


# --- helpers ---------------------------------------------------------------
def load_data(spec: DataSpec) -> Tuple[pd.DataFrame, str]:
    try:
        if spec.source == "synthetic":
            df = generate_gbm(spec.days, mu=spec.drift, sigma=spec.volatility, seed=spec.seed)
            return df, "Synthetic"
        if spec.source == "yahoo":
            ticker = spec.ticker.strip().upper()
            return load_yahoo(ticker, start=spec.start, end=spec.end), ticker
        df = load_csv(io.StringIO(spec.csv_text))
        name = (spec.csv_name or "CSV").rsplit(".", 1)[0]
        return df, name
    except ImportError as exc:
        raise HTTPException(501, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422, f"Could not load data: {exc}") from exc


def clean(value: Any) -> Any:
    """Make values JSON-safe: NaN/inf -> None, numpy/pandas scalars -> Python."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [clean(r) for r in df.to_dict(orient="records")]


# --- routes ----------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "numba": HAS_NUMBA}


@app.get("/api/strategies")
def strategies():
    return STRATEGIES


@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    data, label = load_data(req.data)
    strategy = build_strategy(req.strategy)
    started = time.perf_counter()
    try:
        result = Backtest({label: data}, strategy, **req.costs.kwargs).run()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    elapsed = time.perf_counter() - started

    dates = result.equity.index.strftime("%Y-%m-%d")
    series = [
        {"date": d, "equity": e, "benchmark": b, "drawdown": dd}
        for d, e, b, dd in zip(
            dates,
            result.equity.to_numpy(),
            result.benchmark.to_numpy(),
            result.drawdown.to_numpy(),
        )
    ]
    return clean({
        "meta": {
            "symbol": label,
            "bars": len(data),
            "start": data.index[0],
            "end": data.index[-1],
            "elapsed_ms": round(elapsed * 1000, 1),
        },
        "metrics": result.metrics,
        "series": series,
        "trades": records(result.trades),
        "fills": records(result.fills),
    })


@app.post("/api/sweep")
def sweep(req: SweepRequest):
    if req.fast_min > req.fast_max or req.slow_min > req.slow_max:
        raise HTTPException(422, "Range minimum must not exceed its maximum.")
    # range() objects are lazy, so oversized requests are rejected before allocating anything.
    fasts = range(req.fast_min, req.fast_max + 1, req.step)
    slows = range(req.slow_min, req.slow_max + 1, req.step)
    if len(fasts) * len(slows) > MAX_SWEEP_PAIRS:
        raise HTTPException(422, f"Too many combinations; keep it under {MAX_SWEEP_PAIRS:,}.")

    data, label = load_data(req.data)
    costs = req.costs.kwargs
    started = time.perf_counter()
    grid = sma_grid_search(
        data, fasts, slows,
        initial_cash=costs["initial_cash"],
        commission=costs["commission"],
        slippage=costs["slippage"],
    )
    elapsed = time.perf_counter() - started
    if grid.empty:
        raise HTTPException(422, "No valid pairs: every fast window is at least every slow window.")
    return clean({
        "meta": {"symbol": label, "pairs": len(grid), "elapsed_ms": round(elapsed * 1000, 1), "numba": HAS_NUMBA},
        "fast_values": sorted(grid["fast"].unique().tolist()),
        "slow_values": sorted(grid["slow"].unique().tolist()),
        "rows": records(grid),
    })


# Serve the built frontend, if present.
_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")
