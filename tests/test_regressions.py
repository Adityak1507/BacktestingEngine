"""Regression tests for bugs found in review."""

import numpy as np
import pandas as pd
import pytest

from backtester import Backtest, RsiMeanReversion, SmaCrossover, Strategy, generate_gbm
from backtester.data import validate_ohlcv
from backtester.metrics import compute_metrics
from backtester.portfolio import Trade

from test_engine import BuyOnce, make_bars


def test_run_twice_gives_identical_results():
    bt = Backtest(generate_gbm(300, seed=1), SmaCrossover(5, 20), commission=0.001)
    first, second = bt.run(), bt.run()
    pd.testing.assert_series_equal(first.equity, second.equity)
    assert len(first.trades) == len(second.trades)


def test_run_does_not_mutate_the_strategy_passed_in():
    strategy = BuyOnce(5)
    result = Backtest(make_bars([10, 10, 10]), strategy).run()
    assert strategy.done is False
    assert result.strategy.done is True


def test_trade_pnl_includes_entry_commission():
    class InOut(Strategy):
        def on_bar(self, ctx):
            n = len(ctx.history())
            if n == 1:
                ctx.buy(10)
            elif n == 3:
                ctx.close_position()

    result = Backtest(make_bars([100] * 5), InOut(), initial_cash=10_000, commission=0.01).run()
    # Flat prices: the only P&L is 10 commission in plus 10 out.
    assert result.trades["pnl"].tolist() == [pytest.approx(-20)]
    assert result.trades["pnl"].sum() == pytest.approx(result.equity.iloc[-1] - 10_000)


def test_partial_closes_split_entry_commission():
    class ScaleOut(Strategy):
        def on_bar(self, ctx):
            n = len(ctx.history())
            if n == 1:
                ctx.buy(10)
            elif n in (2, 3):
                ctx.sell(5)

    result = Backtest(make_bars([100] * 5), ScaleOut(), initial_cash=10_000, commission=0.01).run()
    # Entry commission of 10 is split 5/5; each exit pays 5.
    assert result.trades["pnl"].tolist() == [pytest.approx(-10), pytest.approx(-10)]


def test_flip_from_short_to_long_is_limited_by_cash():
    class Flip(Strategy):
        def on_bar(self, ctx):
            n = len(ctx.history())
            if n == 1:
                ctx.sell(10)
            elif n == 2:
                ctx.buy(10_000)

    result = Backtest(make_bars([100] * 4), Flip(), initial_cash=1_000, allow_short=True).run()
    # Shorting 10 @ 100 leaves 2,000 cash; covering costs 1,000, leaving 1,000 for 10 more shares.
    assert result.final_positions["ASSET"] == 10
    cash = result.equity.iloc[-1] - 10 * 100
    assert cash == pytest.approx(0)


def test_covering_a_short_is_not_limited_by_cash():
    class ShortThenCover(Strategy):
        def on_bar(self, ctx):
            n = len(ctx.history())
            if n == 1:
                ctx.sell(10)
            elif n == 2:
                ctx.buy(10)

    data = make_bars([100, 100, 300, 300])  # price triples: covering costs more than the cash on hand
    result = Backtest(data, ShortThenCover(), initial_cash=100, allow_short=True).run()
    assert result.final_positions == {}


def test_break_even_trades_are_not_losses():
    trade = Trade("A", 10, 10, 1, pd.Timestamp("2024-01-01"), 0.0)
    equity = pd.Series([100.0, 100.0, 100.0], index=pd.bdate_range("2024-01-01", periods=3))
    m = compute_metrics(equity, [trade])
    assert m["win_rate"] == 0 and np.isnan(m["avg_loss"])


@pytest.mark.parametrize(
    "make",
    [
        lambda: SmaCrossover(0, 50),
        lambda: SmaCrossover(-5, 50),
        lambda: RsiMeanReversion(period=0),
        lambda: RsiMeanReversion(oversold=60, exit_level=40),
        lambda: RsiMeanReversion(exit_level=150),
    ],
)
def test_invalid_strategy_params_raise(make):
    with pytest.raises(ValueError):
        make()


def test_stop_limit_orders_are_rejected():
    with pytest.raises(ValueError, match="stop-limit"):
        Backtest(make_bars([10, 10]), BuyOnce(1, limit_price=9, stop_price=11)).run()


@pytest.mark.parametrize(
    "column, value, message",
    [
        ("close", -1.0, "non-positive"),
        ("volume", -1.0, "negative volume"),
        ("high", 50.0, "high below low"),
    ],
)
def test_invalid_bars_raise(column, value, message):
    df = make_bars([100, 100, 100])
    df.iloc[1, df.columns.get_loc(column)] = value
    with pytest.raises(ValueError, match=message):
        validate_ohlcv(df)


def test_close_far_outside_range_raises():
    df = make_bars([100, 100, 100])
    df.iloc[1, df.columns.get_loc("close")] = 150.0  # high is 101
    with pytest.raises(ValueError, match="outside the high-low range"):
        validate_ohlcv(df)


def test_small_ohlc_rounding_errors_are_repaired():
    df = make_bars([100, 100, 100])
    df.iloc[1, df.columns.get_loc("close")] = 101.2  # high is 101: 0.2% over
    out = validate_ohlcv(df)
    assert out["high"].iloc[1] == 101.2


def test_history_does_not_expose_future_bars_through_numpy_base():
    reachable = {}

    class Peek(Strategy):
        def on_bar(self, ctx):
            if len(ctx.history()) == 5:
                for arr in (ctx.history()["close"].to_numpy(), ctx.history(length=3).to_numpy()):
                    base = arr.base if arr.base is not None else arr
                    reachable[arr.shape] = base.size

    Backtest(generate_gbm(100, seed=1), Peek()).run()
    # Only the visible bars are reachable on bar 5, not all 100 bars of the dataset.
    assert reachable and all(size <= 5 * 5 for size in reachable.values())
