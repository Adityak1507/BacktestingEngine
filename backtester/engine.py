"""The event loop that ties data, strategy, broker and portfolio together."""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd

from .broker import Broker, CostModel
from .data import validate_ohlcv
from .metrics import compute_metrics, drawdown_series, format_metrics
from .portfolio import Portfolio
from .strategy import Context, Strategy

DataInput = Union[pd.DataFrame, Mapping[str, pd.DataFrame]]


class BacktestResult:
    def __init__(
        self,
        equity: pd.Series,
        portfolio: Portfolio,
        periods_per_year: int,
        risk_free_rate: float,
        benchmark: Optional[pd.Series] = None,
    ):
        self.equity = equity
        self.returns = equity.pct_change().fillna(0.0)
        self.drawdown = drawdown_series(equity)
        self.benchmark = benchmark
        self.fills = pd.DataFrame([asdict(f) for f in portfolio.fills])
        self.trades = pd.DataFrame([asdict(t) for t in portfolio.trades])
        self.final_positions = dict(portfolio.positions)
        self.metrics = compute_metrics(equity, portfolio.trades, periods_per_year, risk_free_rate)
        self.metrics["total_commission"] = portfolio.total_commission
        if benchmark is not None:
            self.metrics["benchmark_return"] = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1)

    def summary(self) -> str:
        return format_metrics(self.metrics)

    def plot(self, path: Optional[str] = None):
        """Plot equity (and benchmark) plus drawdown. Requires matplotlib."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax1.plot(self.equity.index, self.equity, label="Strategy")
        if self.benchmark is not None:
            ax1.plot(self.benchmark.index, self.benchmark, label="Buy & hold", alpha=0.7)
        ax1.set_ylabel("Equity")
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax2.fill_between(self.drawdown.index, self.drawdown * 100, 0, alpha=0.4, color="tab:red")
        ax2.set_ylabel("Drawdown %")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=120)
        return fig

    def plot_interactive(self, path: Optional[str] = None, title: str = "Backtest"):
        """Interactive equity, trade-marker and drawdown chart. Requires plotly.

        Returns a plotly Figure; if `path` is given, also writes a standalone HTML file.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04
        )
        fig.add_trace(go.Scatter(x=self.equity.index, y=self.equity, name="Strategy"), row=1, col=1)
        if self.benchmark is not None:
            fig.add_trace(
                go.Scatter(x=self.benchmark.index, y=self.benchmark, name="Buy & hold", opacity=0.6),
                row=1, col=1,
            )
        if not self.fills.empty:
            for side, mask, symbol, color in (
                ("Buy", self.fills["quantity"] > 0, "triangle-up", "#2ca02c"),
                ("Sell", self.fills["quantity"] < 0, "triangle-down", "#d62728"),
            ):
                f = self.fills[mask]
                if f.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=f["timestamp"],
                        y=self.equity.reindex(f["timestamp"]).to_numpy(),
                        mode="markers",
                        name=side,
                        marker=dict(symbol=symbol, size=10, color=color),
                        customdata=f[["symbol", "quantity", "price"]].to_numpy(),
                        hovertemplate=(
                            f"{side} %{{customdata[0]}}<br>qty %{{customdata[1]:,.0f}}"
                            " @ %{customdata[2]:,.2f}<extra></extra>"
                        ),
                    ),
                    row=1, col=1,
                )
        fig.add_trace(
            go.Scatter(
                x=self.drawdown.index, y=self.drawdown * 100, name="Drawdown %",
                fill="tozeroy", line=dict(color="#d62728"), showlegend=False,
            ),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="Equity", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        fig.update_layout(title=title, hovermode="x unified", height=650, legend=dict(orientation="h", y=1.06))
        if path:
            fig.write_html(path, include_plotlyjs="cdn")
        return fig


class Backtest:
    """Run a strategy over historical bars.

    Each bar is processed in this order:
      1. Orders submitted on the previous bar are filled at this bar's open
         (or at their limit/stop price if triggered within the bar's range).
      2. The portfolio is marked to market at this bar's close.
      3. The strategy's `on_bar` runs and may submit new orders.

    Parameters
    ----------
    data: a single OHLCV DataFrame, or a dict of symbol -> DataFrame.
    strategy: a `Strategy` instance.
    initial_cash: starting capital.
    commission / commission_min / slippage: see `CostModel`.
    allow_short: whether sells may take a position below zero.
    periods_per_year: bars per year, used to annualise metrics (252 for daily).
    """

    def __init__(
        self,
        data: DataInput,
        strategy: Strategy,
        initial_cash: float = 100_000.0,
        commission: float = 0.0,
        commission_min: float = 0.0,
        slippage: float = 0.0,
        allow_short: bool = False,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ):
        if isinstance(data, pd.DataFrame):
            data = {"ASSET": data}
        if not data:
            raise ValueError("no price data supplied")
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")

        frames = {sym: validate_ohlcv(df) for sym, df in data.items()}
        index = frames[next(iter(frames))].index
        for df in frames.values():
            index = index.union(df.index)

        # Align every symbol to a common calendar. Missing bars are forward-filled
        # for valuation but flagged so the broker never fills on them.
        self._has_bar = {sym: df["close"].reindex(index).notna() for sym, df in frames.items()}
        self.data: Dict[str, pd.DataFrame] = {sym: df.reindex(index).ffill() for sym, df in frames.items()}
        self.index = index
        # Plain numpy views for the hot loop; pandas row access per bar is slow.
        self._cols = {
            sym: {c: df[c].to_numpy() for c in ("open", "high", "low", "close")}
            for sym, df in self.data.items()
        }
        self._has_bar_np = {sym: s.to_numpy() for sym, s in self._has_bar.items()}

        self.strategy = strategy
        self.initial_cash = initial_cash
        self.costs = CostModel(commission, commission_min, slippage)
        self.allow_short = allow_short
        self.portfolio = Portfolio(initial_cash)
        self.broker = Broker(self.costs, allow_short)
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

    def run(self) -> BacktestResult:
        """Run the backtest from a clean state; calling it again gives the same result.

        The strategy is deep-copied so state it keeps between bars starts fresh each
        run. The instance that actually ran is available as `result.strategy`.
        """
        self.portfolio = Portfolio(self.initial_cash)
        self.broker = Broker(self.costs, self.allow_short)
        try:
            strategy = copy.deepcopy(self.strategy)
        except (TypeError, copy.Error):
            # Holds something uncopyable (a lock, a connection); run it as-is.
            strategy = self.strategy
        ctx = Context(self)
        strategy.init(ctx)

        equity = []
        valid_from = None
        for i, ts in enumerate(self.index):
            ctx._i, ctx.timestamp = i, ts

            # A symbol only becomes tradeable once its first real bar arrives.
            bars = {
                sym: {c: arr[i] for c, arr in cols.items()}
                for sym, cols in self._cols.items()
                if self._has_bar_np[sym][i]
            }
            fills = self.broker.process(ts, bars, self.portfolio.cash, self.portfolio.positions)
            for fill in fills:
                self.portfolio.apply_fill(fill)

            prices = {
                sym: float(cols["close"][i])
                for sym, cols in self._cols.items()
                if not np.isnan(cols["close"][i])
            }
            if len(prices) < len(self.data):
                continue  # some symbol has not started trading yet
            if valid_from is None:
                valid_from = i
            equity.append(self.portfolio.equity(prices))
            strategy.on_bar(ctx)

        if valid_from is None:
            raise ValueError("price data never overlaps for all symbols")
        eq_index = self.index[valid_from:]
        equity_series = pd.Series(equity, index=eq_index, name="equity")

        first = next(iter(self.data))
        bench_close = self.data[first]["close"].loc[eq_index]
        benchmark = self.portfolio.initial_cash * bench_close / bench_close.iloc[0]

        result = BacktestResult(
            equity_series,
            self.portfolio,
            self.periods_per_year,
            self.risk_free_rate,
            benchmark.rename("benchmark"),
        )
        result.strategy = strategy
        return result
