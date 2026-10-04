"""Loading and generating OHLCV price data."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
OHLC_TOLERANCE = 0.005  # open/close may sit up to 0.5% outside high/low before it's an error


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, sort by date and check the required columns exist."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "adj close" in df.columns:
        df = df.drop(columns=["adj close"])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"price data is missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError("price data has duplicate timestamps")
    df = df[REQUIRED_COLUMNS].astype(float)

    prices = df[["open", "high", "low", "close"]]
    if (prices <= 0).any().any():
        raise ValueError(f"price data has non-positive prices at {_first_bad(prices.le(0).any(axis=1))}")
    if (df["volume"] < 0).any():
        raise ValueError(f"price data has negative volume at {_first_bad(df['volume'] < 0)}")
    if (df["high"] < df["low"]).any():
        raise ValueError(f"price data has high below low at {_first_bad(df['high'] < df['low'])}")
    # Adjusted data often has open/close a rounding error outside high/low; widen the
    # range to fit, but reject bars that are inconsistent by more than that.
    body_high = df[["open", "close"]].max(axis=1)
    body_low = df[["open", "close"]].min(axis=1)
    off = (body_high > df["high"] * (1 + OHLC_TOLERANCE)) | (body_low < df["low"] * (1 - OHLC_TOLERANCE))
    if off.any():
        raise ValueError(f"price data has open/close outside the high-low range at {_first_bad(off)}")
    df["high"] = df[["open", "high", "close"]].max(axis=1, skipna=False)
    df["low"] = df[["open", "low", "close"]].min(axis=1, skipna=False)
    return df


def _first_bad(mask: pd.Series) -> str:
    bad = mask[mask]
    return f"{bad.index[0]} ({len(bad)} rows)"


def load_csv(path: str, date_column: str = "date") -> pd.DataFrame:
    """Load an OHLCV CSV (e.g. a Yahoo Finance export) indexed by date."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if date_column.lower() not in cols:
        where = f" in {path}" if isinstance(path, str) else ""
        raise ValueError(f"column {date_column!r} not found{where}")
    df = df.set_index(cols[date_column.lower()])
    return validate_ohlcv(df)


def load_yahoo(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download split- and dividend-adjusted OHLCV bars from Yahoo Finance.

    Requires the optional `yfinance` package (`pip install -e ".[yahoo]"`).
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("load_yahoo needs yfinance: pip install yfinance") from exc

    df = yf.download(
        symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False
    )
    if df is None or df.empty:
        raise ValueError(f"no data returned from Yahoo Finance for {symbol!r}")
    if isinstance(df.columns, pd.MultiIndex):
        # Newer yfinance returns (field, ticker) columns even for one ticker.
        df = df.xs(symbol, axis=1, level=-1) if symbol in df.columns.get_level_values(-1) else df.droplevel(-1, axis=1)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return validate_ohlcv(df.dropna(how="all"))


def generate_gbm(
    n_days: int = 756,
    start_price: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.2,
    start: str = "2020-01-01",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate synthetic daily OHLCV data using geometric Brownian motion.

    Useful for demos and tests when no real market data is available.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_days)
    close = start_price * np.exp(np.cumsum(returns))
    prev_close = np.concatenate([[start_price], close[:-1]])
    # Open near the previous close with a small overnight gap.
    open_ = prev_close * np.exp(rng.normal(0, sigma * np.sqrt(dt) * 0.3, n_days))
    spread = np.abs(rng.normal(0, sigma * np.sqrt(dt) * 0.5, n_days))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(100_000, 1_000_000, n_days).astype(float)
    index = pd.bdate_range(start=start, periods=n_days, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
