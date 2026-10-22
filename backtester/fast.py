"""Numba-accelerated backtests for signal-based, single-asset strategies.

The event-driven `Backtest` is flexible but runs Python code on every bar.
When a strategy can be expressed as a precomputed signal array, the kernel
here simulates it in compiled code, which makes parameter sweeps over
thousands of combinations practical.

Execution rules match `Backtest`: decisions are made at bar t's close, orders
fill at bar t+1's open with slippage and commission, positions are whole
shares and buys are capped to available cash.

Numba is optional. Without it the same code runs as plain Python (slower).
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .data import validate_ohlcv

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda fn: fn


# Signal values: 1 = be long, 0 = be flat, -1 = no change.
LONG, FLAT, HOLD = 1, 0, -1


@njit(cache=True)
def _affordable(cash, price, commission_rate, commission_min):
    if cash <= 0 or price <= 0:
        return 0.0
    qty = float(int(cash / (price * (1 + commission_rate))))
    while qty > 0 and qty * price + max(qty * price * commission_rate, commission_min) > cash:
        qty -= 1
    return qty


@njit(cache=True)
def simulate(open_, close, signal, initial_cash, fraction, commission_rate, commission_min, slippage):
    """Run a long/flat strategy. Returns (equity array, number of fills)."""
    n = len(close)
    equity = np.empty(n)
    cash = initial_cash
    pos = 0.0
    pending = 0.0
    n_fills = 0
    for t in range(n):
        if pending != 0.0:
            if pending > 0:
                price = open_[t] * (1 + slippage)
                qty = min(pending, _affordable(cash, price, commission_rate, commission_min))
            else:
                price = open_[t] * (1 - slippage)
                qty = -min(-pending, pos)
            if qty != 0.0:
                commission = max(abs(qty) * price * commission_rate, commission_min)
                cash -= qty * price + commission
                pos += qty
                n_fills += 1
            pending = 0.0

        equity[t] = cash + pos * close[t]

        s = signal[t]
        if s == LONG and pos == 0.0:
            pending = float(int(equity[t] * fraction / close[t])) if close[t] > 0 else 0.0
        elif s == FLAT and pos > 0.0:
            pending = -pos
    return equity, n_fills


@njit(cache=True)
def rolling_mean(x, window):
    out = np.full(len(x), np.nan)
    if window > len(x):
        return out
    s = 0.0
    for i in range(len(x)):
        s += x[i]
        if i >= window:
            s -= x[i - window]
        if i >= window - 1:
            out[i] = s / window
    return out


@njit(cache=True)
def sma_signal(close, fast, slow):
    """1 when fast SMA > slow SMA, 0 when below, -1 during warm-up or ties."""
    f = rolling_mean(close, fast)
    s = rolling_mean(close, slow)
    sig = np.full(len(close), HOLD, dtype=np.int8)
    for i in range(slow - 1, len(close)):
        if f[i] > s[i]:
            sig[i] = LONG
        elif f[i] < s[i]:
            sig[i] = FLAT
    return sig


@njit(cache=True)
def summary_stats(equity, periods_per_year):
    """(total return, annualised Sharpe, max drawdown) of an equity curve."""
    n = len(equity)
    total = equity[-1] / equity[0] - 1
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        max_dd = min(max_dd, v / peak - 1)
    if n < 3:
        return total, np.nan, max_dd
    rets = equity[1:] / equity[:-1] - 1
    sd = rets.std() * np.sqrt((n - 1) / (n - 2))  # sample std (ddof=1)
    sharpe = rets.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else np.nan
    return total, sharpe, max_dd


def run_signals(
    data: pd.DataFrame,
    signal,
    initial_cash: float = 100_000.0,
    fraction: float = 0.99,
    commission: float = 0.0,
    commission_min: float = 0.0,
    slippage: float = 0.0,
) -> pd.Series:
    """Backtest a precomputed long/flat signal; returns the equity curve.

    `signal` is aligned to `data` and holds 1 (long), 0 (flat) or -1 (no change).
    A boolean signal is treated as long/flat with no warm-up.
    """
    data = validate_ohlcv(data)
    sig = np.asarray(signal)
    if sig.dtype == bool:
        sig = sig.astype(np.int8)
    if len(sig) != len(data):
        raise ValueError("signal length must match data length")
    equity, _ = simulate(
        data["open"].to_numpy(),
        data["close"].to_numpy(),
        sig.astype(np.int8),
        float(initial_cash),
        float(fraction),
        float(commission),
        float(commission_min),
        float(slippage),
    )
    return pd.Series(equity, index=data.index, name="equity")


def sma_grid_search(
    data: pd.DataFrame,
    fast_values: Iterable[int],
    slow_values: Iterable[int],
    initial_cash: float = 100_000.0,
    fraction: float = 0.99,
    commission: float = 0.0,
    commission_min: float = 0.0,
    slippage: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Evaluate every (fast, slow) SMA crossover pair with fast < slow.

    Returns one row per pair, sorted by Sharpe ratio (best first).

    Beware of overfitting: the best in-sample pair is rarely the best
    out-of-sample. Validate on data the search did not see.
    """
    data = validate_ohlcv(data)
    open_ = data["open"].to_numpy()
    close = data["close"].to_numpy()
    rows: list[Dict[str, float]] = []
    for fast, slow in itertools.product(fast_values, slow_values):
        if fast < 1:
            raise ValueError("SMA windows must be at least 1")
        if fast >= slow:
            continue
        sig = sma_signal(close, int(fast), int(slow))
        equity, n_fills = simulate(
            open_, close, sig, float(initial_cash), float(fraction),
            float(commission), float(commission_min), float(slippage),
        )
        total, sharpe, max_dd = summary_stats(equity, periods_per_year)
        rows.append({
            "fast": int(fast), "slow": int(slow), "total_return": total,
            "sharpe": sharpe, "max_drawdown": max_dd, "num_fills": n_fills,
        })
    df = pd.DataFrame(rows, columns=["fast", "slow", "total_return", "sharpe", "max_drawdown", "num_fills"])
    return df.sort_values("sharpe", ascending=False, na_position="last").reset_index(drop=True)
