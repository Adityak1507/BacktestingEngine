import math

import numpy as np
import pandas as pd
import pytest

from backtester import Backtest, BuyAndHold, SmaCrossover, Strategy, generate_gbm
from backtester.metrics import compute_metrics


def make_bars(closes, opens=None, start="2024-01-01"):
    closes = np.asarray(closes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 1,
            "low": np.minimum(opens, closes) - 1,
            "close": closes,
            "volume": 1000.0,
        },
        index=pd.bdate_range(start, periods=len(closes)),
    )


class BuyOnce(Strategy):
    def __init__(self, qty, **order_kwargs):
        self.qty, self.kwargs, self.done = qty, order_kwargs, False

    def on_bar(self, ctx):
        if not self.done:
            ctx.buy(self.qty, **self.kwargs)
            self.done = True


def test_orders_fill_on_next_bar_open():
    data = make_bars(closes=[10, 11, 12, 13], opens=[10, 20, 30, 40])
    result = Backtest(data, BuyOnce(10), initial_cash=1000).run()

    fill = result.fills.iloc[0]
    assert fill["timestamp"] == data.index[1]
    assert fill["price"] == 20  # next bar's open, not the signal bar's close
    # Equity at the end: cash 1000 - 200, plus 10 shares at 13.
    assert result.equity.iloc[-1] == pytest.approx(800 + 130)


def test_commission_and_slippage_are_charged():
    data = make_bars([100, 100, 100])
    result = Backtest(data, BuyOnce(10), initial_cash=10_000, commission=0.01, slippage=0.01).run()

    fill = result.fills.iloc[0]
    assert fill["price"] == pytest.approx(101)
    assert fill["commission"] == pytest.approx(10.1)
    assert result.equity.iloc[-1] == pytest.approx(10_000 - 1010 - 10.1 + 1000)


def test_buys_are_capped_by_available_cash():
    data = make_bars([100, 100, 100])
    result = Backtest(data, BuyOnce(1_000), initial_cash=1_050).run()
    assert result.fills.iloc[0]["quantity"] == 10
    assert result.final_positions["ASSET"] == 10


def test_cannot_sell_more_than_held_without_shorting():
    class SellFirst(Strategy):
        def on_bar(self, ctx):
            ctx.sell(5)

    result = Backtest(make_bars([10, 10, 10]), SellFirst()).run()
    assert result.fills.empty
    assert result.final_positions == {}


def test_short_selling_profits_when_price_falls():
    class ShortOnce(Strategy):
        def on_bar(self, ctx):
            if ctx.position() == 0 and len(ctx.history()) == 1:
                ctx.sell(10)

    data = make_bars([100, 100, 80])
    result = Backtest(data, ShortOnce(), initial_cash=10_000, allow_short=True).run()
    assert result.final_positions["ASSET"] == -10
    assert result.equity.iloc[-1] == pytest.approx(10_200)


def test_limit_order_fills_only_when_price_reached():
    data = make_bars(closes=[100, 100, 90], opens=[100, 100, 92])
    # Bar 1 low is 99 (no fill at 95); bar 2 low is 89, opens at 92 -> fills at 92.
    result = Backtest(data, BuyOnce(1, limit_price=95), initial_cash=1_000).run()
    assert len(result.fills) == 1
    assert result.fills.iloc[0]["timestamp"] == data.index[2]
    assert result.fills.iloc[0]["price"] == 92


def test_strategy_cannot_see_future_bars():
    seen = []

    class Spy(Strategy):
        def on_bar(self, ctx):
            hist = ctx.history()
            seen.append((ctx.timestamp, hist.index[-1], len(hist)))

    data = make_bars([10, 20, 30, 40, 50])
    Backtest(data, Spy()).run()
    for i, (ts, last, n) in enumerate(seen):
        assert ts == last
        assert n == i + 1


def test_round_trip_trade_pnl_recorded():
    class InAndOut(Strategy):
        def on_bar(self, ctx):
            n = len(ctx.history())
            if n == 1:
                ctx.buy(10)
            elif n == 3:
                ctx.close_position()

    data = make_bars(closes=[10, 10, 10, 10], opens=[10, 10, 10, 15])
    result = Backtest(data, InAndOut(), initial_cash=1_000).run()
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == 10 and trade["exit_price"] == 15
    assert trade["pnl"] == pytest.approx(50)
    assert result.metrics["win_rate"] == 1.0


def test_buy_and_hold_tracks_benchmark():
    data = generate_gbm(n_days=300, seed=1)
    result = Backtest(data, BuyAndHold(), initial_cash=100_000).run()
    # Invested ~99% from bar 2 onward, so returns should be close to the asset's.
    asset_ret = data["close"].iloc[-1] / data["open"].iloc[1] - 1
    assert result.metrics["total_return"] == pytest.approx(asset_ret * 0.99, abs=0.01)


def test_multi_symbol_backtest():
    class Equal(Strategy):
        def on_bar(self, ctx):
            for sym in ctx.symbols:
                if ctx.position(sym) == 0:
                    ctx.order_target_percent(0.49, sym)

    data = {"A": generate_gbm(200, seed=1), "B": generate_gbm(200, seed=2)}
    result = Backtest(data, Equal()).run()
    assert set(result.final_positions) == {"A", "B"}
    assert len(result.equity) == 200


def test_sma_crossover_runs_and_produces_metrics():
    data = generate_gbm(n_days=750, seed=7)
    result = Backtest(data, SmaCrossover(10, 30), commission=0.001).run()
    m = result.metrics
    assert m["num_trades"] > 0
    assert -1 <= m["max_drawdown"] <= 0
    assert m["total_commission"] > 0


def test_metrics_on_known_equity_curve():
    equity = pd.Series([100, 110, 99, 120], index=pd.bdate_range("2024-01-01", periods=4))
    m = compute_metrics(equity, [])
    assert m["total_return"] == pytest.approx(0.2)
    assert m["max_drawdown"] == pytest.approx(99 / 110 - 1)
    assert m["max_drawdown_duration"] == 1
    assert math.isnan(m["win_rate"])


def test_missing_columns_raise():
    bad = make_bars([1, 2, 3]).drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing columns"):
        Backtest(bad, BuyAndHold())
