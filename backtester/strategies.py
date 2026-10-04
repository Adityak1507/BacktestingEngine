"""Ready-made example strategies."""

from __future__ import annotations

from typing import Optional

from .strategy import Context, Strategy


class BuyAndHold(Strategy):
    """Invest everything on the first bar and never sell."""

    def __init__(self, symbol: Optional[str] = None):
        self.symbol = symbol

    def on_bar(self, ctx: Context) -> None:
        if ctx.position(self.symbol) == 0:
            ctx.order_target_percent(0.99, self.symbol)


class SmaCrossover(Strategy):
    """Long when the fast moving average is above the slow one, flat otherwise."""

    def __init__(self, fast: int = 20, slow: int = 50, symbol: Optional[str] = None):
        if fast < 1:
            raise ValueError("fast window must be at least 1")
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast, self.slow, self.symbol = fast, slow, symbol

    def on_bar(self, ctx: Context) -> None:
        close = ctx.history(self.symbol, self.slow)["close"]
        if len(close) < self.slow:
            return
        fast_ma = close.iloc[-self.fast:].mean()
        slow_ma = close.mean()
        if fast_ma > slow_ma and ctx.position(self.symbol) == 0:
            ctx.order_target_percent(0.99, self.symbol)
        elif fast_ma < slow_ma and ctx.position(self.symbol) > 0:
            ctx.close_position(self.symbol)


class RsiMeanReversion(Strategy):
    """Buy when RSI is oversold, sell when it recovers above the exit level."""

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30,
        exit_level: float = 55,
        symbol: Optional[str] = None,
    ):
        if period < 1:
            raise ValueError("RSI period must be at least 1")
        if not 0 <= oversold < exit_level <= 100:
            raise ValueError("RSI levels must satisfy 0 <= oversold < exit_level <= 100")
        self.period, self.oversold, self.exit_level, self.symbol = period, oversold, exit_level, symbol

    def _rsi(self, ctx: Context) -> Optional[float]:
        close = ctx.history(self.symbol, self.period * 5)["close"]
        if len(close) <= self.period:
            return None
        delta = close.diff().dropna()
        gain = delta.clip(lower=0).ewm(alpha=1 / self.period, adjust=False).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / self.period, adjust=False).mean().iloc[-1]
        if loss == 0:
            return 100.0
        return 100 - 100 / (1 + gain / loss)

    def on_bar(self, ctx: Context) -> None:
        rsi = self._rsi(ctx)
        if rsi is None:
            return
        if rsi < self.oversold and ctx.position(self.symbol) == 0:
            ctx.order_target_percent(0.99, self.symbol)
        elif rsi > self.exit_level and ctx.position(self.symbol) > 0:
            ctx.close_position(self.symbol)
