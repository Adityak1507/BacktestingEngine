"""Performance statistics computed from an equity curve and trade list."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .portfolio import Trade


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional distance below the running peak (0 at a new high, negative otherwise)."""
    return equity / equity.cummax() - 1


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest number of bars spent below a previous peak."""
    underwater = (equity < equity.cummax()).astype(int).to_numpy()
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def compute_metrics(
    equity: pd.Series,
    trades: List[Trade],
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    returns = equity.pct_change().dropna()
    n = len(returns)
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    total_return = end / start - 1

    years = n / periods_per_year if n else 0
    cagr = (end / start) ** (1 / years) - 1 if years > 0 and end > 0 else float("nan")

    rf = risk_free_rate / periods_per_year
    excess = returns - rf
    vol = returns.std(ddof=1) * np.sqrt(periods_per_year) if n > 1 else float("nan")
    sharpe = (
        excess.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year)
        if n > 1 and returns.std(ddof=1) > 0
        else float("nan")
    )
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean()) if n else 0.0
    sortino = (
        excess.mean() / downside * np.sqrt(periods_per_year) if downside > 0 else float("nan")
    )

    dd = drawdown_series(equity)
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 and not np.isnan(cagr) else float("nan")

    pnls = np.array([t.pnl for t in trades])
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]  # break-even trades are neither
    gross_loss = abs(losses.sum())

    return {
        "start_equity": start,
        "end_equity": end,
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_duration": max_drawdown_duration(equity),
        "calmar": calmar,
        "num_trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else float("nan"),
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "profit_factor": wins.sum() / gross_loss if gross_loss > 0 else float("nan"),
    }


_PERCENT_KEYS = {
    "total_return", "cagr", "annual_volatility", "max_drawdown", "win_rate", "benchmark_return",
}


def format_metrics(metrics: Dict[str, float]) -> str:
    width = max(len(k) for k in metrics)
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float) and np.isnan(value):
            text = "n/a"
        elif key in _PERCENT_KEYS:
            text = f"{value:.2%}"
        elif isinstance(value, float):
            text = f"{value:,.2f}"
        else:
            text = str(value)
        lines.append(f"{key.replace('_', ' '):<{width}}  {text}")
    return "\n".join(lines)
