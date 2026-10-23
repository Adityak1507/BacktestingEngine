"""A small event-driven backtesting engine."""

from .broker import CostModel, Fill, Order
from .data import generate_gbm, load_csv, load_sample, load_yahoo
from .engine import Backtest, BacktestResult
from .portfolio import Portfolio, Trade
from .strategies import BuyAndHold, RsiMeanReversion, SmaCrossover
from .strategy import Context, Strategy

__all__ = [
    "Backtest",
    "BacktestResult",
    "BuyAndHold",
    "Context",
    "CostModel",
    "Fill",
    "Order",
    "Portfolio",
    "RsiMeanReversion",
    "SmaCrossover",
    "Strategy",
    "Trade",
    "generate_gbm",
    "load_csv",
    "load_sample",
    "load_yahoo",
]
