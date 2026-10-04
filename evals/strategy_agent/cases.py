"""Eval cases for the strategy agent.

Each case is a plain-English request plus either a reference implementation
(the grader compares the agent's positions against it on hidden data) or
`expect="decline"` for requests that can't be implemented honestly.

Tags: tags[0] is the category; the rest give difficulty and what the case tests.
"""

from __future__ import annotations

from textwrap import dedent

SIZE = "Invest 95% of equity when entering a position (use order_target_percent(0.95))."

# Shared indicator snippets, written the way the prompts define them.
WILDER_RSI = '''
def wilder_rsi(close, period):
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    return 100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1])
'''


def _code(src: str) -> str:
    return dedent(src).strip().replace("__RSI__", WILDER_RSI.strip()) + "\n"


CASES = [
    # ---------------------------------------------------------------- trend
    {
        "id": "sma_cross",
        "tags": ["trend", "easy"],
        "prompt": f"Go long when the 10-day simple moving average of the close is above the 30-day SMA, "
                  f"and be flat when it is below. {SIZE} Wait until 30 bars of history exist.",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    c = ctx.history(length=30)["close"]
                    if len(c) < 30:
                        return
                    fast, slow = c.iloc[-10:].mean(), c.mean()
                    if fast > slow and ctx.position() == 0:
                        ctx.order_target_percent(0.95)
                    elif fast < slow and ctx.position() > 0:
                        ctx.close_position()
        '''),
    },
    {
        "id": "ema_cross",
        "tags": ["trend", "easy"],
        "prompt": "Long when the 12-period EMA of the close is above the 26-period EMA, flat otherwise. "
                  "Compute each EMA over all available history with pandas `ewm(span=N, adjust=False)`. "
                  f"Only start trading once 26 bars exist. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    c = ctx.history()["close"]
                    if len(c) < 26:
                        return
                    fast = c.ewm(span=12, adjust=False).mean().iloc[-1]
                    slow = c.ewm(span=26, adjust=False).mean().iloc[-1]
                    if fast > slow and ctx.position() == 0:
                        ctx.order_target_percent(0.95)
                    elif fast < slow and ctx.position() > 0:
                        ctx.close_position()
        '''),
    },
    {
        "id": "above_sma100",
        "tags": ["trend", "easy"],
        "prompt": f"Hold the asset whenever the close is above its 100-day simple moving average and be flat "
                  f"when the close is below it. Do nothing until 100 bars exist. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    c = ctx.history(length=100)["close"]
                    if len(c) < 100:
                        return
                    if c.iloc[-1] > c.mean() and ctx.position() == 0:
                        ctx.order_target_percent(0.95)
                    elif c.iloc[-1] < c.mean() and ctx.position() > 0:
                        ctx.close_position()
        '''),
    },
    {
        "id": "momentum_60",
        "tags": ["trend", "easy"],
        "prompt": "Time-series momentum: be long when today's close is above the close 60 bars ago "
                  "(i.e. the 60-bar return is positive) and flat when it is below. "
                  f"You need 61 bars of history before the first decision. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    c = ctx.history(length=61)["close"]
                    if len(c) < 61:
                        return
                    ret = c.iloc[-1] / c.iloc[0] - 1
                    if ret > 0 and ctx.position() == 0:
                        ctx.order_target_percent(0.95)
                    elif ret < 0 and ctx.position() > 0:
                        ctx.close_position()
        '''),
    },
    {
        "id": "macd",
        "tags": ["trend", "medium"],
        "prompt": "MACD strategy: MACD line = EMA12 - EMA26 of the close; signal line = 9-period EMA of the "
                  "MACD line. All EMAs use pandas `ewm(span=N, adjust=False)` over all available history. "
                  "Be long when the MACD line is above the signal line and flat when it is below. "
                  f"Don't trade until 35 bars exist. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    c = ctx.history()["close"]
                    if len(c) < 35:
                        return
                    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
                    signal = macd.ewm(span=9, adjust=False).mean()
                    if macd.iloc[-1] > signal.iloc[-1] and ctx.position() == 0:
                        ctx.order_target_percent(0.95)
                    elif macd.iloc[-1] < signal.iloc[-1] and ctx.position() > 0:
                        ctx.close_position()
        '''),
    },
    {
        "id": "sma_long_short",
        "tags": ["trend", "medium", "shorting"],
        "settings": {"allow_short": True},
        "prompt": "Always in the market once 30 bars exist: when the 10-day SMA is above the 30-day SMA hold a "
                  "long position worth 50% of equity; when it is below, hold a short position worth 50% of "
                  "equity. Only trade when the side needs to change (flip directly from long to short and back).",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    c = ctx.history(length=30)["close"]
                    if len(c) < 30:
                        return
                    fast, slow = c.iloc[-10:].mean(), c.mean()
                    pos = ctx.position()
                    if fast > slow and pos <= 0:
                        ctx.order_target_percent(0.5)
                    elif fast < slow and pos >= 0:
                        ctx.order_target_percent(-0.5)
        '''),
    },
    # ---------------------------------------------------------------- mean reversion
    {
        "id": "rsi_wilder",
        "tags": ["mean_reversion", "medium"],
        "prompt": "RSI mean reversion with a 14-period Wilder RSI computed over all available history: take "
                  "close-to-close changes (drop the first NaN), average gains and losses with pandas "
                  "`ewm(alpha=1/14, adjust=False)`, RSI = 100 - 100/(1 + avg_gain/avg_loss) (100 if avg_loss is 0). "
                  f"Buy when flat and RSI < 30; sell everything when long and RSI > 55. Start once 15 bars exist. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            __RSI__

            class Ref(Strategy):
                def on_bar(self, ctx):
                    c = ctx.history()["close"]
                    if len(c) < 15:
                        return
                    rsi = wilder_rsi(c, 14)
                    if rsi < 30 and ctx.position() == 0:
                        ctx.order_target_percent(0.95)
                    elif rsi > 55 and ctx.position() > 0:
                        ctx.close_position()
        '''),
    },
    {
        "id": "bollinger",
        "tags": ["mean_reversion", "easy"],
        "prompt": "Bollinger band reversion: middle band = 20-bar SMA of the close, lower band = middle - 2 x the "
                  "20-bar rolling standard deviation (pandas default, ddof=1). Buy when flat and the close is below "
                  f"the lower band; exit when long and the close is above the middle band. Needs 20 bars. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    c = ctx.history(length=20)["close"]
                    if len(c) < 20:
                        return
                    mid, sd = c.mean(), c.std()
                    if c.iloc[-1] < mid - 2 * sd and ctx.position() == 0:
                        ctx.order_target_percent(0.95)
                    elif c.iloc[-1] > mid and ctx.position() > 0:
                        ctx.close_position()
        '''),
    },
    {
        "id": "zscore",
        "tags": ["mean_reversion", "easy"],
        "prompt": "Z-score reversion: z = (close - 20-bar SMA) / 20-bar standard deviation (ddof=1), using the last "
                  f"20 closes including today. Buy when flat and z < -1.5; exit when long and z > 0. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    c = ctx.history(length=20)["close"]
                    if len(c) < 20:
                        return
                    z = (c.iloc[-1] - c.mean()) / c.std()
                    if z < -1.5 and ctx.position() == 0:
                        ctx.order_target_percent(0.95)
                    elif z > 0 and ctx.position() > 0:
                        ctx.close_position()
        '''),
    },
    {
        "id": "ten_day_low_hold5",
        "tags": ["mean_reversion", "medium", "stateful"],
        "prompt": "Buy when flat and today's close is the lowest close of the last 10 bars (today included; needs "
                  "10 bars). Exit with a market order exactly 5 bars after the bar on which you submitted the buy: "
                  f"if the buy was submitted on bar t, submit the sell on bar t+5. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def init(self, ctx):
                    self.entry_bar = None

                def on_bar(self, ctx):
                    hist = ctx.history()
                    i = len(hist)
                    if ctx.position() > 0:
                        if self.entry_bar is not None and i - self.entry_bar >= 5:
                            ctx.close_position()
                            self.entry_bar = None
                        return
                    c = hist["close"].iloc[-10:]
                    if len(c) == 10 and c.iloc[-1] == c.min():
                        ctx.order_target_percent(0.95)
                        self.entry_bar = i
        '''),
    },
    {
        "id": "three_down_days",
        "tags": ["mean_reversion", "easy"],
        "prompt": "Buy when flat after three consecutive lower closes (each of the last three closes is below the "
                  f"close before it). Exit when long on the first close that is above the previous close. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    c = ctx.history(length=4)["close"]
                    if len(c) < 4:
                        return
                    if ctx.position() == 0 and c.iloc[3] < c.iloc[2] < c.iloc[1] < c.iloc[0]:
                        ctx.order_target_percent(0.95)
                    elif ctx.position() > 0 and c.iloc[3] > c.iloc[2]:
                        ctx.close_position()
        '''),
    },
    {
        "id": "rsi2",
        "tags": ["mean_reversion", "medium"],
        "prompt": "Connors-style RSI(2): compute a 2-period Wilder RSI over all available history (close-to-close "
                  "changes with the first NaN dropped, gains and losses averaged with pandas "
                  "`ewm(alpha=1/2, adjust=False)`, RSI = 100 if average loss is 0). Buy when flat and RSI(2) < 10. "
                  f"Exit when long and the close is above its 5-bar SMA. Start once 5 bars exist. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            __RSI__

            class Ref(Strategy):
                def on_bar(self, ctx):
                    c = ctx.history()["close"]
                    if len(c) < 5:
                        return
                    if ctx.position() == 0 and wilder_rsi(c, 2) < 10:
                        ctx.order_target_percent(0.95)
                    elif ctx.position() > 0 and c.iloc[-1] > c.iloc[-5:].mean():
                        ctx.close_position()
        '''),
    },
    # ---------------------------------------------------------------- breakout
    {
        "id": "donchian",
        "tags": ["breakout", "medium"],
        "prompt": "Donchian breakout: buy when flat and today's close is above the highest HIGH of the previous 20 "
                  "bars (excluding today). Exit when long and today's close is below the lowest LOW of the previous "
                  f"10 bars (excluding today). Needs 21 bars of history. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    h = ctx.history(length=21)
                    if len(h) < 21:
                        return
                    close = h["close"].iloc[-1]
                    if ctx.position() == 0 and close > h["high"].iloc[:-1].max():
                        ctx.order_target_percent(0.95)
                    elif ctx.position() > 0 and close < h["low"].iloc[-11:-1].min():
                        ctx.close_position()
        '''),
    },
    {
        "id": "stop_breakout",
        "tags": ["breakout", "hard", "order_types", "stateful"],
        "prompt": "Breakout using stop orders. Once 20 bars exist, on every bar while flat: cancel any pending "
                  "orders, then place a buy STOP order at the highest high of the last 20 bars (today included) for "
                  "int(0.95 * equity / today's close) shares. Once you are long, stop placing entries; on the 10th "
                  "bar after the first bar on which you observe a long position (that bar counts as 0), close the "
                  "position with a market order.",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def init(self, ctx):
                    self.long_since = None

                def on_bar(self, ctx):
                    hist = ctx.history()
                    i = len(hist)
                    if i < 20:
                        return
                    if ctx.position() == 0:
                        self.long_since = None
                        ctx.cancel_orders()
                        qty = int(0.95 * ctx.equity / ctx.price())
                        ctx.buy(qty, stop_price=hist["high"].iloc[-20:].max())
                    else:
                        if self.long_since is None:
                            self.long_since = i
                        if i - self.long_since == 10:
                            ctx.close_position()
        '''),
    },
    {
        "id": "range_expansion",
        "tags": ["breakout", "medium", "stateful"],
        "prompt": "Volatility expansion: let today's range = high - low. Buy when flat, today's range is more than "
                  "2x the average range of the 20 bars before today, and today closed above its open. Exit with a "
                  f"market order 3 bars after the bar on which the buy was submitted (bar t+3). Needs 21 bars. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def init(self, ctx):
                    self.entry_bar = None

                def on_bar(self, ctx):
                    h = ctx.history()
                    i = len(h)
                    if ctx.position() > 0:
                        if self.entry_bar is not None and i - self.entry_bar >= 3:
                            ctx.close_position()
                            self.entry_bar = None
                        return
                    if i < 21:
                        return
                    rng = h["high"] - h["low"]
                    today = h.iloc[-1]
                    if rng.iloc[-1] > 2 * rng.iloc[-21:-1].mean() and today["close"] > today["open"]:
                        ctx.order_target_percent(0.95)
                        self.entry_bar = i
        '''),
    },
    {
        "id": "gap_up",
        "tags": ["breakout", "easy", "stateful"],
        "prompt": "Gap-up continuation: buy when flat and today's open is above yesterday's high. Hold for exactly "
                  f"one bar: submit the exit on the bar after the one where you submitted the buy. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def init(self, ctx):
                    self.entry_bar = None

                def on_bar(self, ctx):
                    h = ctx.history(length=2)
                    i = len(ctx.history())
                    if ctx.position() > 0:
                        if self.entry_bar is not None and i - self.entry_bar >= 1:
                            ctx.close_position()
                            self.entry_bar = None
                        return
                    if len(h) == 2 and h["open"].iloc[1] > h["high"].iloc[0]:
                        ctx.order_target_percent(0.95)
                        self.entry_bar = i
        '''),
    },
    # ---------------------------------------------------------------- risk management
    {
        "id": "trailing_stop",
        "tags": ["risk", "hard", "stateful"],
        "prompt": "Trend entry with a trailing stop. Signal: 20-bar SMA vs 50-bar SMA of the close (needs 50 bars). "
                  "Enter when flat and SMA20 > SMA50. While long, track the highest close since (and including) the "
                  "bar you submitted the entry; exit if the close drops below 92% of that peak, or if SMA20 falls "
                  "below SMA50. After a trailing-stop exit, do not re-enter until SMA20 has first been below SMA50 "
                  f"again. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def init(self, ctx):
                    self.peak = None
                    self.blocked = False

                def on_bar(self, ctx):
                    c = ctx.history(length=50)["close"]
                    if len(c) < 50:
                        return
                    fast, slow, close = c.iloc[-20:].mean(), c.mean(), c.iloc[-1]
                    pos = ctx.position()
                    if fast > slow:
                        if pos == 0 and not self.blocked:
                            ctx.order_target_percent(0.95)
                            self.peak = close
                        elif pos > 0:
                            self.peak = max(self.peak, close)
                            if close < 0.92 * self.peak:
                                ctx.close_position()
                                self.blocked = True
                    else:
                        self.blocked = False
                        if pos > 0:
                            ctx.close_position()
        '''),
    },
    {
        "id": "take_profit_stop_loss",
        "tags": ["risk", "medium", "stateful"],
        "prompt": "Enter long when flat and the 10-bar SMA is above the 30-bar SMA (needs 30 bars). Remember the "
                  "close on the bar where you submit the entry as the reference price. While long, exit when the "
                  "close reaches +10% above the reference (>=), drops to -5% below it (<=), or the 10-bar SMA "
                  f"falls below the 30-bar SMA. Re-entry is allowed immediately on later bars. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def init(self, ctx):
                    self.ref = None

                def on_bar(self, ctx):
                    c = ctx.history(length=30)["close"]
                    if len(c) < 30:
                        return
                    fast, slow, close = c.iloc[-10:].mean(), c.mean(), c.iloc[-1]
                    if ctx.position() == 0:
                        if fast > slow:
                            ctx.order_target_percent(0.95)
                            self.ref = close
                    elif close >= 1.10 * self.ref or close <= 0.95 * self.ref or fast < slow:
                        ctx.close_position()
        '''),
    },
    {
        "id": "limit_entry",
        "tags": ["risk", "hard", "order_types", "stateful"],
        "prompt": "Use the 14-period Wilder RSI (close changes with the first NaN dropped, pandas "
                  "`ewm(alpha=1/14, adjust=False)` averages, 100 if average loss is 0) over all history; start once "
                  "15 bars exist. When flat with no entry order working and RSI < 30, place a LIMIT buy for "
                  "int(0.95 * equity / close) shares at 98% of today's close. If that order is still unfilled at the "
                  "close of the 3rd bar after you placed it, cancel it (a new order may then be placed on that same "
                  "bar if RSI < 30). While long, exit with a market order when RSI > 55.",
        "reference": _code('''
            from backtester import Strategy

            __RSI__

            class Ref(Strategy):
                def init(self, ctx):
                    self.order_bar = None

                def on_bar(self, ctx):
                    c = ctx.history()["close"]
                    i = len(c)
                    if i < 15:
                        return
                    rsi = wilder_rsi(c, 14)
                    if ctx.position() > 0:
                        self.order_bar = None
                        if rsi > 55:
                            ctx.close_position()
                        return
                    if self.order_bar is not None:
                        if i - self.order_bar >= 3:
                            ctx.cancel_orders()
                            self.order_bar = None
                        else:
                            return
                    if rsi < 30:
                        price = ctx.price()
                        ctx.buy(int(0.95 * ctx.equity / price), limit_price=0.98 * price)
                        self.order_bar = i
        '''),
    },
    {
        "id": "vol_target",
        "tags": ["risk", "medium", "sizing"],
        "prompt": "Volatility-targeted trend following. Needs 50 bars. When the close is above its 50-bar SMA, "
                  "target a position of min(1.0, 0.15 / vol) of equity, where vol = standard deviation (ddof=1) of the "
                  "last 20 daily percentage returns x sqrt(252). When the close is at or below the SMA, target 0. "
                  "Rebalance to the target on every bar with order_target_percent.",
        "reference": _code('''
            import math
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    c = ctx.history(length=51)["close"]
                    if len(c) < 50:
                        return
                    if c.iloc[-1] > c.iloc[-50:].mean():
                        vol = c.pct_change().iloc[-20:].std() * math.sqrt(252)
                        ctx.order_target_percent(min(1.0, 0.15 / vol))
                    else:
                        ctx.order_target_percent(0.0)
        '''),
    },
    {
        "id": "scale_in",
        "tags": ["risk", "medium", "sizing"],
        "prompt": "Scale exposure with trend strength. Needs 100 bars. Count how many of the 20-, 50- and 100-bar "
                  "SMAs today's close is above (0 to 3). Target a position of 0.95 x count / 3 of equity, rebalancing "
                  "with order_target_percent on every bar.",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    c = ctx.history(length=100)["close"]
                    if len(c) < 100:
                        return
                    close = c.iloc[-1]
                    count = sum(close > c.iloc[-n:].mean() for n in (20, 50, 100))
                    ctx.order_target_percent(0.95 * count / 3)
        '''),
    },
    # ---------------------------------------------------------------- calendar
    {
        "id": "month_start",
        "tags": ["calendar", "medium", "stateful"],
        "prompt": "Turn-of-the-month: on the first bar of each calendar month (including the very first bar of the "
                  "data), buy if flat. On the 10th bar of the month (counting that first bar as 1), sell everything "
                  f"if long. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    idx = ctx.history().index
                    now = ctx.timestamp
                    n = sum(1 for d in idx if d.year == now.year and d.month == now.month)
                    if n == 1 and ctx.position() == 0:
                        ctx.order_target_percent(0.95)
                    elif n == 10 and ctx.position() > 0:
                        ctx.close_position()
        '''),
    },
    {
        "id": "weekend_hold",
        "tags": ["calendar", "easy"],
        "prompt": f"Weekend effect: on Fridays submit a buy if flat; on Mondays submit a sell of everything if long. {SIZE}",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    day = ctx.timestamp.weekday()
                    if day == 4 and ctx.position() == 0:
                        ctx.order_target_percent(0.95)
                    elif day == 0 and ctx.position() > 0:
                        ctx.close_position()
        '''),
    },
    # ---------------------------------------------------------------- multi-asset
    {
        "id": "relative_momentum",
        "tags": ["multi_asset", "medium"],
        "settings": {"symbols": ["SPY", "TLT"]},
        "prompt": "Relative momentum between SPY and TLT: once 61 bars exist, compute each symbol's 60-bar return "
                  "(today's close / close 60 bars ago - 1). Hold only the leader with 95% of equity and nothing in "
                  "the other. When the leader changes, close the old position and buy the new leader on the same bar.",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    rets = {}
                    for s in ctx.symbols:
                        c = ctx.history(s, 61)["close"]
                        if len(c) < 61:
                            return
                        rets[s] = c.iloc[-1] / c.iloc[0] - 1
                    leader = max(rets, key=rets.get)
                    for s in ctx.symbols:
                        if s != leader and ctx.position(s) > 0:
                            ctx.close_position(s)
                    if ctx.position(leader) == 0:
                        ctx.order_target_percent(0.95, leader)
        '''),
    },
    {
        "id": "equal_weight",
        "tags": ["multi_asset", "easy"],
        "settings": {"symbols": ["AAA", "BBB", "CCC"]},
        "prompt": "Equal-weight portfolio of AAA, BBB and CCC: on the first bar and then every 20 bars after it "
                  "(bars 1, 21, 41, ...), rebalance each symbol to 30% of equity with order_target_percent. "
                  "Do nothing on other bars.",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    i = len(ctx.history(ctx.symbols[0]))
                    if (i - 1) % 20 == 0:
                        for s in ctx.symbols:
                            ctx.order_target_percent(0.30, s)
        '''),
    },
    {
        "id": "ratio_reversion",
        "tags": ["multi_asset", "medium"],
        "settings": {"symbols": ["XXX", "YYY"]},
        "prompt": "Ratio mean reversion: ratio = close of XXX / close of YYY. Over the last 20 bars (today included) "
                  "compute z = (today's ratio - mean of the 20 ratios) / their standard deviation (ddof=1). Buy XXX with "
                  "95% of equity when flat and z < -2; sell all XXX when long and z > 0. Never trade YYY.",
        "reference": _code('''
            from backtester import Strategy

            class Ref(Strategy):
                def on_bar(self, ctx):
                    x = ctx.history("XXX", 20)["close"]
                    y = ctx.history("YYY", 20)["close"]
                    if len(x) < 20:
                        return
                    r = x / y
                    z = (r.iloc[-1] - r.mean()) / r.std()
                    if z < -2 and ctx.position("XXX") == 0:
                        ctx.order_target_percent(0.95, "XXX")
                    elif z > 0 and ctx.position("XXX") > 0:
                        ctx.close_position("XXX")
        '''),
    },
    # ---------------------------------------------------------------- should decline
    {
        "id": "decline_tomorrow_close",
        "tags": ["decline", "easy", "lookahead"],
        "prompt": "Buy whenever tomorrow's close will be higher than today's close, and sell whenever it will be lower.",
        "expect": "decline",
    },
    {
        "id": "decline_dataset_max",
        "tags": ["decline", "hard", "lookahead"],
        "prompt": "Normalise prices by dividing each close by the highest close in the entire dataset, then buy "
                  "with 95% of equity when the normalised close is below 0.5 and sell when it is above 0.8.",
        "expect": "decline",
    },
    {
        "id": "decline_earnings",
        "tags": ["decline", "medium", "missing_data"],
        "prompt": "Buy when the company's quarterly earnings beat analyst estimates, and sell after the next earnings report.",
        "expect": "decline",
    },
    {
        "id": "decline_leverage",
        "tags": ["decline", "medium", "unsupported"],
        "prompt": "Whenever the 50-day SMA is rising, borrow money so the position is worth 300% of equity (3x "
                  "leverage); otherwise stay flat.",
        "expect": "decline",
    },
]

# --------------------------------------------------------------------- loose prompts
# Worded the way a user would actually type them. Several readings are reasonable,
# so each case lists the acceptable ones under `references`; the agent passes if it
# matches any of them by position direction (long / flat / short), since sizing
# isn't specified.
CUTLER_RSI = '''
def cutler_rsi(close, period):
    delta = close.diff().dropna().iloc[-period:]
    gain, loss = delta.clip(lower=0).mean(), (-delta.clip(upper=0)).mean()
    return 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
'''


def _sma_state(fast: int, slow: int) -> str:
    return _code(f'''
        from backtester import Strategy

        class Ref(Strategy):
            def on_bar(self, ctx):
                c = ctx.history(length={slow})["close"]
                if len(c) < {slow}:
                    return
                f, s = c.iloc[-{fast}:].mean(), c.mean()
                if f > s and ctx.position() == 0:
                    ctx.order_target_percent(0.95)
                elif f < s and ctx.position() > 0:
                    ctx.close_position()
    ''')


def _sma_event(fast: int, slow: int) -> str:
    return _code(f'''
        from backtester import Strategy

        class Ref(Strategy):
            def on_bar(self, ctx):
                c = ctx.history(length={slow + 1})["close"]
                if len(c) < {slow + 1}:
                    return
                f_now, s_now = c.iloc[-{fast}:].mean(), c.iloc[-{slow}:].mean()
                f_prev, s_prev = c.iloc[-{fast + 1}:-1].mean(), c.iloc[:-1].mean()
                if f_prev <= s_prev and f_now > s_now and ctx.position() == 0:
                    ctx.order_target_percent(0.95)
                elif f_prev >= s_prev and f_now < s_now and ctx.position() > 0:
                    ctx.close_position()
    ''')


def _rsi_ref(kind: str, low: float, high: float) -> str:
    fn = "wilder_rsi" if kind == "wilder" else "cutler_rsi"
    helper = WILDER_RSI if kind == "wilder" else CUTLER_RSI
    return "from backtester import Strategy\n" + helper + _code(f'''
        class Ref(Strategy):
            def on_bar(self, ctx):
                c = ctx.history()["close"]
                if len(c) < 15:
                    return
                rsi = {fn}(c, 14)
                if rsi < {low} and ctx.position() == 0:
                    ctx.order_target_percent(0.95)
                elif rsi > {high} and ctx.position() > 0:
                    ctx.close_position()
    ''')


def _bollinger_ref(exit_at: str) -> str:
    target = "mid" if exit_at == "mid" else "mid + 2 * sd"
    return _code(f'''
        from backtester import Strategy

        class Ref(Strategy):
            def on_bar(self, ctx):
                c = ctx.history(length=20)["close"]
                if len(c) < 20:
                    return
                mid, sd = c.mean(), c.std()
                if c.iloc[-1] < mid - 2 * sd and ctx.position() == 0:
                    ctx.order_target_percent(0.95)
                elif c.iloc[-1] > {target} and ctx.position() > 0:
                    ctx.close_position()
    ''')


def _turtle_ref(entry: int, exit_: int) -> str:
    return _code(f'''
        from backtester import Strategy

        class Ref(Strategy):
            def on_bar(self, ctx):
                h = ctx.history(length={entry + 1})
                if len(h) < {entry + 1}:
                    return
                close = h["close"].iloc[-1]
                if ctx.position() == 0 and close > h["high"].iloc[:-1].max():
                    ctx.order_target_percent(0.95)
                elif ctx.position() > 0 and close < h["low"].iloc[-{exit_ + 1}:-1].min():
                    ctx.close_position()
    ''')


def _macd_ref(adjust: bool) -> str:
    return _code(f'''
        from backtester import Strategy

        class Ref(Strategy):
            def on_bar(self, ctx):
                c = ctx.history()["close"]
                if len(c) < 35:
                    return
                ema = lambda s, n: s.ewm(span=n, adjust={adjust}).mean()
                macd = ema(c, 12) - ema(c, 26)
                signal = ema(macd, 9)
                if macd.iloc[-1] > signal.iloc[-1] and ctx.position() == 0:
                    ctx.order_target_percent(0.95)
                elif macd.iloc[-1] < signal.iloc[-1] and ctx.position() > 0:
                    ctx.close_position()
    ''')


def _rotation_ref(lookback: int) -> str:
    return _code(f'''
        from backtester import Strategy

        class Ref(Strategy):
            def on_bar(self, ctx):
                rets = {{}}
                for s in ctx.symbols:
                    c = ctx.history(s, {lookback + 1})["close"]
                    if len(c) < {lookback + 1}:
                        return
                    rets[s] = c.iloc[-1] / c.iloc[0] - 1
                leader = max(rets, key=rets.get)
                for s in ctx.symbols:
                    if s != leader and ctx.position(s) > 0:
                        ctx.close_position(s)
                if ctx.position(leader) == 0:
                    ctx.order_target_percent(0.95, leader)
    ''')


def _slope_ref(lag: int) -> str:
    return _code(f'''
        from backtester import Strategy

        class Ref(Strategy):
            def on_bar(self, ctx):
                c = ctx.history(length={50 + lag})["close"]
                if len(c) < {50 + lag}:
                    return
                now, before = c.iloc[-50:].mean(), c.iloc[-50 - {lag}:-{lag}].mean()
                if now > before and ctx.position() == 0:
                    ctx.order_target_percent(0.95)
                elif now < before and ctx.position() > 0:
                    ctx.close_position()
    ''')


LOOSE_CASES = [
    {
        "id": "loose_golden_cross",
        "tags": ["loose", "trend", "easy"],
        "prompt": "can you code up a golden cross strategy",
        "references": [_sma_state(50, 200), _sma_event(50, 200)],
        "interpretations": ["long while SMA50 > SMA200", "enter/exit only on the actual crossover"],
    },
    {
        "id": "loose_rsi",
        "tags": ["loose", "mean_reversion", "easy"],
        "prompt": "buy when RSI says it's oversold and sell when it's overbought",
        "references": [_rsi_ref("wilder", 30, 70), _rsi_ref("cutler", 30, 70)],
        "interpretations": ["Wilder RSI(14), 30/70", "simple-average (Cutler) RSI(14), 30/70"],
    },
    {
        "id": "loose_bollinger",
        "tags": ["loose", "mean_reversion", "easy"],
        "prompt": "bollinger band bounce strategy - buy the lower band",
        "references": [_bollinger_ref("mid"), _bollinger_ref("upper")],
        "interpretations": ["20/2 bands, exit at the middle band", "20/2 bands, exit at the upper band"],
    },
    {
        "id": "loose_turtle",
        "tags": ["loose", "breakout", "medium"],
        "prompt": "classic turtle trading breakout system please, long only",
        "references": [_turtle_ref(20, 10), _turtle_ref(55, 20)],
        "interpretations": ["System 1: 20-day high entry, 10-day low exit", "System 2: 55-day high, 20-day low"],
    },
    {
        "id": "loose_macd",
        "tags": ["loose", "trend", "easy"],
        "prompt": "standard MACD crossover, long when bullish",
        "references": [_macd_ref(False), _macd_ref(True)],
        "interpretations": ["12/26/9, ewm adjust=False", "12/26/9, ewm adjust=True"],
    },
    {
        "id": "loose_rotation",
        "tags": ["loose", "multi_asset", "medium"],
        "settings": {"symbols": ["SPY", "TLT"]},
        "prompt": "just hold whichever of SPY or TLT has been doing better over the last 3 months",
        "references": [_rotation_ref(63), _rotation_ref(60), _rotation_ref(66)],
        "interpretations": ["63-bar return", "60-bar return", "66-bar return"],
    },
    {
        "id": "loose_ma_slope",
        "tags": ["loose", "trend", "medium"],
        "prompt": "be long when the 50 day moving average is going up, out when it's going down",
        "references": [_slope_ref(1), _slope_ref(5)],
        "interpretations": ["SMA50 above yesterday's SMA50", "SMA50 above its value 5 bars ago"],
    },
    {
        "id": "loose_decline_twitter",
        "tags": ["loose", "decline", "medium"],
        "prompt": "use twitter sentiment to time my entries on this stock",
        "expect": "decline",
    },
    {
        "id": "loose_decline_fed",
        "tags": ["loose", "decline", "medium"],
        "prompt": "buy the day before the fed cuts rates and sell a week later",
        "expect": "decline",
    },
]

CASES += LOOSE_CASES
CASES_BY_ID = {c["id"]: c for c in CASES}


def write_markdown(path: str = "evals/strategy_agent/CASES.md") -> None:
    """Render the cases as a reviewable markdown file: `python -m evals.strategy_agent.cases`."""
    lines = ["# Strategy agent eval cases", "",
             f"{len(CASES)} cases. Graded against the reference implementation(s) on hidden data, "
             "or expected to be declined. Loose cases are worded casually and accept any of several readings.", "",
             "| # | id | category | tags | settings | expected |", "|---|---|---|---|---|---|"]
    for i, c in enumerate(CASES, 1):
        settings = ", ".join(f"{k}={v}" for k, v in c.get("settings", {}).items()) or "1 symbol, long only"
        expected = ("decline" if c.get("expect") == "decline"
                    else f"match any of {len(c['references'])} readings" if "references" in c else "match reference")
        lines.append(f"| {i} | `{c['id']}` | {c['tags'][0]} | {', '.join(c['tags'][1:])} | {settings} | {expected} |")
    for i, c in enumerate(CASES, 1):
        lines += ["", f"## {i}. `{c['id']}`", "", "Prompt:", "", "````text", c["prompt"], "````"]
        if "references" in c:
            lines += ["", "Acceptable readings (matched by position direction):"]
            lines += [f"- {r}" for r in c["interpretations"]]
        for k, ref in enumerate(c.get("references") or ([c["reference"]] if "reference" in c else [])):
            title = c["interpretations"][k] if "references" in c else "Reference implementation"
            lines += ["", f"<details><summary>{title}</summary>", "", "````python", ref.rstrip(), "````", "", "</details>"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    write_markdown()
