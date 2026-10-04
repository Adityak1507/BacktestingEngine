# Strategy agent eval cases

39 cases. Graded against the reference implementation(s) on hidden data, or expected to be declined. Loose cases are worded casually and accept any of several readings.

| # | id | category | tags | settings | expected |
|---|---|---|---|---|---|
| 1 | `sma_cross` | trend | easy | 1 symbol, long only | match reference |
| 2 | `ema_cross` | trend | easy | 1 symbol, long only | match reference |
| 3 | `above_sma100` | trend | easy | 1 symbol, long only | match reference |
| 4 | `momentum_60` | trend | easy | 1 symbol, long only | match reference |
| 5 | `macd` | trend | medium | 1 symbol, long only | match reference |
| 6 | `sma_long_short` | trend | medium, shorting | allow_short=True | match reference |
| 7 | `rsi_wilder` | mean_reversion | medium | 1 symbol, long only | match reference |
| 8 | `bollinger` | mean_reversion | easy | 1 symbol, long only | match reference |
| 9 | `zscore` | mean_reversion | easy | 1 symbol, long only | match reference |
| 10 | `ten_day_low_hold5` | mean_reversion | medium, stateful | 1 symbol, long only | match reference |
| 11 | `three_down_days` | mean_reversion | easy | 1 symbol, long only | match reference |
| 12 | `rsi2` | mean_reversion | medium | 1 symbol, long only | match reference |
| 13 | `donchian` | breakout | medium | 1 symbol, long only | match reference |
| 14 | `stop_breakout` | breakout | hard, order_types, stateful | 1 symbol, long only | match reference |
| 15 | `range_expansion` | breakout | medium, stateful | 1 symbol, long only | match reference |
| 16 | `gap_up` | breakout | easy, stateful | 1 symbol, long only | match reference |
| 17 | `trailing_stop` | risk | hard, stateful | 1 symbol, long only | match reference |
| 18 | `take_profit_stop_loss` | risk | medium, stateful | 1 symbol, long only | match reference |
| 19 | `limit_entry` | risk | hard, order_types, stateful | 1 symbol, long only | match reference |
| 20 | `vol_target` | risk | medium, sizing | 1 symbol, long only | match reference |
| 21 | `scale_in` | risk | medium, sizing | 1 symbol, long only | match reference |
| 22 | `month_start` | calendar | medium, stateful | 1 symbol, long only | match reference |
| 23 | `weekend_hold` | calendar | easy | 1 symbol, long only | match reference |
| 24 | `relative_momentum` | multi_asset | medium | symbols=['SPY', 'TLT'] | match reference |
| 25 | `equal_weight` | multi_asset | easy | symbols=['AAA', 'BBB', 'CCC'] | match reference |
| 26 | `ratio_reversion` | multi_asset | medium | symbols=['XXX', 'YYY'] | match reference |
| 27 | `decline_tomorrow_close` | decline | easy, lookahead | 1 symbol, long only | decline |
| 28 | `decline_dataset_max` | decline | hard, lookahead | 1 symbol, long only | decline |
| 29 | `decline_earnings` | decline | medium, missing_data | 1 symbol, long only | decline |
| 30 | `decline_leverage` | decline | medium, unsupported | 1 symbol, long only | decline |
| 31 | `loose_golden_cross` | loose | trend, easy | 1 symbol, long only | match any of 2 readings |
| 32 | `loose_rsi` | loose | mean_reversion, easy | 1 symbol, long only | match any of 2 readings |
| 33 | `loose_bollinger` | loose | mean_reversion, easy | 1 symbol, long only | match any of 2 readings |
| 34 | `loose_turtle` | loose | breakout, medium | 1 symbol, long only | match any of 2 readings |
| 35 | `loose_macd` | loose | trend, easy | 1 symbol, long only | match any of 2 readings |
| 36 | `loose_rotation` | loose | multi_asset, medium | symbols=['SPY', 'TLT'] | match any of 3 readings |
| 37 | `loose_ma_slope` | loose | trend, medium | 1 symbol, long only | match any of 2 readings |
| 38 | `loose_decline_twitter` | loose | decline, medium | 1 symbol, long only | decline |
| 39 | `loose_decline_fed` | loose | decline, medium | 1 symbol, long only | decline |

## 1. `sma_cross`

Prompt:

````text
Go long when the 10-day simple moving average of the close is above the 30-day SMA, and be flat when it is below. Invest 95% of equity when entering a position (use order_target_percent(0.95)). Wait until 30 bars of history exist.
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 2. `ema_cross`

Prompt:

````text
Long when the 12-period EMA of the close is above the 26-period EMA, flat otherwise. Compute each EMA over all available history with pandas `ewm(span=N, adjust=False)`. Only start trading once 26 bars exist. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 3. `above_sma100`

Prompt:

````text
Hold the asset whenever the close is above its 100-day simple moving average and be flat when the close is below it. Do nothing until 100 bars exist. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 4. `momentum_60`

Prompt:

````text
Time-series momentum: be long when today's close is above the close 60 bars ago (i.e. the 60-bar return is positive) and flat when it is below. You need 61 bars of history before the first decision. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 5. `macd`

Prompt:

````text
MACD strategy: MACD line = EMA12 - EMA26 of the close; signal line = 9-period EMA of the MACD line. All EMAs use pandas `ewm(span=N, adjust=False)` over all available history. Be long when the MACD line is above the signal line and flat when it is below. Don't trade until 35 bars exist. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 6. `sma_long_short`

Prompt:

````text
Always in the market once 30 bars exist: when the 10-day SMA is above the 30-day SMA hold a long position worth 50% of equity; when it is below, hold a short position worth 50% of equity. Only trade when the side needs to change (flip directly from long to short and back).
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 7. `rsi_wilder`

Prompt:

````text
RSI mean reversion with a 14-period Wilder RSI computed over all available history: take close-to-close changes (drop the first NaN), average gains and losses with pandas `ewm(alpha=1/14, adjust=False)`, RSI = 100 - 100/(1 + avg_gain/avg_loss) (100 if avg_loss is 0). Buy when flat and RSI < 30; sell everything when long and RSI > 55. Start once 15 bars exist. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
from backtester import Strategy

def wilder_rsi(close, period):
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    return 100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1])

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
````

</details>

## 8. `bollinger`

Prompt:

````text
Bollinger band reversion: middle band = 20-bar SMA of the close, lower band = middle - 2 x the 20-bar rolling standard deviation (pandas default, ddof=1). Buy when flat and the close is below the lower band; exit when long and the close is above the middle band. Needs 20 bars. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 9. `zscore`

Prompt:

````text
Z-score reversion: z = (close - 20-bar SMA) / 20-bar standard deviation (ddof=1), using the last 20 closes including today. Buy when flat and z < -1.5; exit when long and z > 0. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 10. `ten_day_low_hold5`

Prompt:

````text
Buy when flat and today's close is the lowest close of the last 10 bars (today included; needs 10 bars). Exit with a market order exactly 5 bars after the bar on which you submitted the buy: if the buy was submitted on bar t, submit the sell on bar t+5. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 11. `three_down_days`

Prompt:

````text
Buy when flat after three consecutive lower closes (each of the last three closes is below the close before it). Exit when long on the first close that is above the previous close. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 12. `rsi2`

Prompt:

````text
Connors-style RSI(2): compute a 2-period Wilder RSI over all available history (close-to-close changes with the first NaN dropped, gains and losses averaged with pandas `ewm(alpha=1/2, adjust=False)`, RSI = 100 if average loss is 0). Buy when flat and RSI(2) < 10. Exit when long and the close is above its 5-bar SMA. Start once 5 bars exist. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
from backtester import Strategy

def wilder_rsi(close, period):
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    return 100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1])

class Ref(Strategy):
    def on_bar(self, ctx):
        c = ctx.history()["close"]
        if len(c) < 5:
            return
        if ctx.position() == 0 and wilder_rsi(c, 2) < 10:
            ctx.order_target_percent(0.95)
        elif ctx.position() > 0 and c.iloc[-1] > c.iloc[-5:].mean():
            ctx.close_position()
````

</details>

## 13. `donchian`

Prompt:

````text
Donchian breakout: buy when flat and today's close is above the highest HIGH of the previous 20 bars (excluding today). Exit when long and today's close is below the lowest LOW of the previous 10 bars (excluding today). Needs 21 bars of history. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 14. `stop_breakout`

Prompt:

````text
Breakout using stop orders. Once 20 bars exist, on every bar while flat: cancel any pending orders, then place a buy STOP order at the highest high of the last 20 bars (today included) for int(0.95 * equity / today's close) shares. Once you are long, stop placing entries; on the 10th bar after the first bar on which you observe a long position (that bar counts as 0), close the position with a market order.
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 15. `range_expansion`

Prompt:

````text
Volatility expansion: let today's range = high - low. Buy when flat, today's range is more than 2x the average range of the 20 bars before today, and today closed above its open. Exit with a market order 3 bars after the bar on which the buy was submitted (bar t+3). Needs 21 bars. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 16. `gap_up`

Prompt:

````text
Gap-up continuation: buy when flat and today's open is above yesterday's high. Hold for exactly one bar: submit the exit on the bar after the one where you submitted the buy. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 17. `trailing_stop`

Prompt:

````text
Trend entry with a trailing stop. Signal: 20-bar SMA vs 50-bar SMA of the close (needs 50 bars). Enter when flat and SMA20 > SMA50. While long, track the highest close since (and including) the bar you submitted the entry; exit if the close drops below 92% of that peak, or if SMA20 falls below SMA50. After a trailing-stop exit, do not re-enter until SMA20 has first been below SMA50 again. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 18. `take_profit_stop_loss`

Prompt:

````text
Enter long when flat and the 10-bar SMA is above the 30-bar SMA (needs 30 bars). Remember the close on the bar where you submit the entry as the reference price. While long, exit when the close reaches +10% above the reference (>=), drops to -5% below it (<=), or the 10-bar SMA falls below the 30-bar SMA. Re-entry is allowed immediately on later bars. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 19. `limit_entry`

Prompt:

````text
Use the 14-period Wilder RSI (close changes with the first NaN dropped, pandas `ewm(alpha=1/14, adjust=False)` averages, 100 if average loss is 0) over all history; start once 15 bars exist. When flat with no entry order working and RSI < 30, place a LIMIT buy for int(0.95 * equity / close) shares at 98% of today's close. If that order is still unfilled at the close of the 3rd bar after you placed it, cancel it (a new order may then be placed on that same bar if RSI < 30). While long, exit with a market order when RSI > 55.
````

<details><summary>Reference implementation</summary>

````python
from backtester import Strategy

def wilder_rsi(close, period):
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    return 100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1])

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
````

</details>

## 20. `vol_target`

Prompt:

````text
Volatility-targeted trend following. Needs 50 bars. When the close is above its 50-bar SMA, target a position of min(1.0, 0.15 / vol) of equity, where vol = standard deviation (ddof=1) of the last 20 daily percentage returns x sqrt(252). When the close is at or below the SMA, target 0. Rebalance to the target on every bar with order_target_percent.
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 21. `scale_in`

Prompt:

````text
Scale exposure with trend strength. Needs 100 bars. Count how many of the 20-, 50- and 100-bar SMAs today's close is above (0 to 3). Target a position of 0.95 x count / 3 of equity, rebalancing with order_target_percent on every bar.
````

<details><summary>Reference implementation</summary>

````python
from backtester import Strategy

class Ref(Strategy):
    def on_bar(self, ctx):
        c = ctx.history(length=100)["close"]
        if len(c) < 100:
            return
        close = c.iloc[-1]
        count = sum(close > c.iloc[-n:].mean() for n in (20, 50, 100))
        ctx.order_target_percent(0.95 * count / 3)
````

</details>

## 22. `month_start`

Prompt:

````text
Turn-of-the-month: on the first bar of each calendar month (including the very first bar of the data), buy if flat. On the 10th bar of the month (counting that first bar as 1), sell everything if long. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 23. `weekend_hold`

Prompt:

````text
Weekend effect: on Fridays submit a buy if flat; on Mondays submit a sell of everything if long. Invest 95% of equity when entering a position (use order_target_percent(0.95)).
````

<details><summary>Reference implementation</summary>

````python
from backtester import Strategy

class Ref(Strategy):
    def on_bar(self, ctx):
        day = ctx.timestamp.weekday()
        if day == 4 and ctx.position() == 0:
            ctx.order_target_percent(0.95)
        elif day == 0 and ctx.position() > 0:
            ctx.close_position()
````

</details>

## 24. `relative_momentum`

Prompt:

````text
Relative momentum between SPY and TLT: once 61 bars exist, compute each symbol's 60-bar return (today's close / close 60 bars ago - 1). Hold only the leader with 95% of equity and nothing in the other. When the leader changes, close the old position and buy the new leader on the same bar.
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 25. `equal_weight`

Prompt:

````text
Equal-weight portfolio of AAA, BBB and CCC: on the first bar and then every 20 bars after it (bars 1, 21, 41, ...), rebalance each symbol to 30% of equity with order_target_percent. Do nothing on other bars.
````

<details><summary>Reference implementation</summary>

````python
from backtester import Strategy

class Ref(Strategy):
    def on_bar(self, ctx):
        i = len(ctx.history(ctx.symbols[0]))
        if (i - 1) % 20 == 0:
            for s in ctx.symbols:
                ctx.order_target_percent(0.30, s)
````

</details>

## 26. `ratio_reversion`

Prompt:

````text
Ratio mean reversion: ratio = close of XXX / close of YYY. Over the last 20 bars (today included) compute z = (today's ratio - mean of the 20 ratios) / their standard deviation (ddof=1). Buy XXX with 95% of equity when flat and z < -2; sell all XXX when long and z > 0. Never trade YYY.
````

<details><summary>Reference implementation</summary>

````python
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
````

</details>

## 27. `decline_tomorrow_close`

Prompt:

````text
Buy whenever tomorrow's close will be higher than today's close, and sell whenever it will be lower.
````

## 28. `decline_dataset_max`

Prompt:

````text
Normalise prices by dividing each close by the highest close in the entire dataset, then buy with 95% of equity when the normalised close is below 0.5 and sell when it is above 0.8.
````

## 29. `decline_earnings`

Prompt:

````text
Buy when the company's quarterly earnings beat analyst estimates, and sell after the next earnings report.
````

## 30. `decline_leverage`

Prompt:

````text
Whenever the 50-day SMA is rising, borrow money so the position is worth 300% of equity (3x leverage); otherwise stay flat.
````

## 31. `loose_golden_cross`

Prompt:

````text
can you code up a golden cross strategy
````

Acceptable readings (matched by position direction):
- long while SMA50 > SMA200
- enter/exit only on the actual crossover

<details><summary>long while SMA50 > SMA200</summary>

````python
from backtester import Strategy

class Ref(Strategy):
    def on_bar(self, ctx):
        c = ctx.history(length=200)["close"]
        if len(c) < 200:
            return
        f, s = c.iloc[-50:].mean(), c.mean()
        if f > s and ctx.position() == 0:
            ctx.order_target_percent(0.95)
        elif f < s and ctx.position() > 0:
            ctx.close_position()
````

</details>

<details><summary>enter/exit only on the actual crossover</summary>

````python
from backtester import Strategy

class Ref(Strategy):
    def on_bar(self, ctx):
        c = ctx.history(length=201)["close"]
        if len(c) < 201:
            return
        f_now, s_now = c.iloc[-50:].mean(), c.iloc[-200:].mean()
        f_prev, s_prev = c.iloc[-51:-1].mean(), c.iloc[:-1].mean()
        if f_prev <= s_prev and f_now > s_now and ctx.position() == 0:
            ctx.order_target_percent(0.95)
        elif f_prev >= s_prev and f_now < s_now and ctx.position() > 0:
            ctx.close_position()
````

</details>

## 32. `loose_rsi`

Prompt:

````text
buy when RSI says it's oversold and sell when it's overbought
````

Acceptable readings (matched by position direction):
- Wilder RSI(14), 30/70
- simple-average (Cutler) RSI(14), 30/70

<details><summary>Wilder RSI(14), 30/70</summary>

````python
from backtester import Strategy

def wilder_rsi(close, period):
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    return 100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1])
class Ref(Strategy):
    def on_bar(self, ctx):
        c = ctx.history()["close"]
        if len(c) < 15:
            return
        rsi = wilder_rsi(c, 14)
        if rsi < 30 and ctx.position() == 0:
            ctx.order_target_percent(0.95)
        elif rsi > 70 and ctx.position() > 0:
            ctx.close_position()
````

</details>

<details><summary>simple-average (Cutler) RSI(14), 30/70</summary>

````python
from backtester import Strategy

def cutler_rsi(close, period):
    delta = close.diff().dropna().iloc[-period:]
    gain, loss = delta.clip(lower=0).mean(), (-delta.clip(upper=0)).mean()
    return 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
class Ref(Strategy):
    def on_bar(self, ctx):
        c = ctx.history()["close"]
        if len(c) < 15:
            return
        rsi = cutler_rsi(c, 14)
        if rsi < 30 and ctx.position() == 0:
            ctx.order_target_percent(0.95)
        elif rsi > 70 and ctx.position() > 0:
            ctx.close_position()
````

</details>

## 33. `loose_bollinger`

Prompt:

````text
bollinger band bounce strategy - buy the lower band
````

Acceptable readings (matched by position direction):
- 20/2 bands, exit at the middle band
- 20/2 bands, exit at the upper band

<details><summary>20/2 bands, exit at the middle band</summary>

````python
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
````

</details>

<details><summary>20/2 bands, exit at the upper band</summary>

````python
from backtester import Strategy

class Ref(Strategy):
    def on_bar(self, ctx):
        c = ctx.history(length=20)["close"]
        if len(c) < 20:
            return
        mid, sd = c.mean(), c.std()
        if c.iloc[-1] < mid - 2 * sd and ctx.position() == 0:
            ctx.order_target_percent(0.95)
        elif c.iloc[-1] > mid + 2 * sd and ctx.position() > 0:
            ctx.close_position()
````

</details>

## 34. `loose_turtle`

Prompt:

````text
classic turtle trading breakout system please, long only
````

Acceptable readings (matched by position direction):
- System 1: 20-day high entry, 10-day low exit
- System 2: 55-day high, 20-day low

<details><summary>System 1: 20-day high entry, 10-day low exit</summary>

````python
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
````

</details>

<details><summary>System 2: 55-day high, 20-day low</summary>

````python
from backtester import Strategy

class Ref(Strategy):
    def on_bar(self, ctx):
        h = ctx.history(length=56)
        if len(h) < 56:
            return
        close = h["close"].iloc[-1]
        if ctx.position() == 0 and close > h["high"].iloc[:-1].max():
            ctx.order_target_percent(0.95)
        elif ctx.position() > 0 and close < h["low"].iloc[-21:-1].min():
            ctx.close_position()
````

</details>

## 35. `loose_macd`

Prompt:

````text
standard MACD crossover, long when bullish
````

Acceptable readings (matched by position direction):
- 12/26/9, ewm adjust=False
- 12/26/9, ewm adjust=True

<details><summary>12/26/9, ewm adjust=False</summary>

````python
from backtester import Strategy

class Ref(Strategy):
    def on_bar(self, ctx):
        c = ctx.history()["close"]
        if len(c) < 35:
            return
        ema = lambda s, n: s.ewm(span=n, adjust=False).mean()
        macd = ema(c, 12) - ema(c, 26)
        signal = ema(macd, 9)
        if macd.iloc[-1] > signal.iloc[-1] and ctx.position() == 0:
            ctx.order_target_percent(0.95)
        elif macd.iloc[-1] < signal.iloc[-1] and ctx.position() > 0:
            ctx.close_position()
````

</details>

<details><summary>12/26/9, ewm adjust=True</summary>

````python
from backtester import Strategy

class Ref(Strategy):
    def on_bar(self, ctx):
        c = ctx.history()["close"]
        if len(c) < 35:
            return
        ema = lambda s, n: s.ewm(span=n, adjust=True).mean()
        macd = ema(c, 12) - ema(c, 26)
        signal = ema(macd, 9)
        if macd.iloc[-1] > signal.iloc[-1] and ctx.position() == 0:
            ctx.order_target_percent(0.95)
        elif macd.iloc[-1] < signal.iloc[-1] and ctx.position() > 0:
            ctx.close_position()
````

</details>

## 36. `loose_rotation`

Prompt:

````text
just hold whichever of SPY or TLT has been doing better over the last 3 months
````

Acceptable readings (matched by position direction):
- 63-bar return
- 60-bar return
- 66-bar return

<details><summary>63-bar return</summary>

````python
from backtester import Strategy

class Ref(Strategy):
    def on_bar(self, ctx):
        rets = {}
        for s in ctx.symbols:
            c = ctx.history(s, 64)["close"]
            if len(c) < 64:
                return
            rets[s] = c.iloc[-1] / c.iloc[0] - 1
        leader = max(rets, key=rets.get)
        for s in ctx.symbols:
            if s != leader and ctx.position(s) > 0:
                ctx.close_position(s)
        if ctx.position(leader) == 0:
            ctx.order_target_percent(0.95, leader)
````

</details>

<details><summary>60-bar return</summary>

````python
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
````

</details>

<details><summary>66-bar return</summary>

````python
from backtester import Strategy

class Ref(Strategy):
    def on_bar(self, ctx):
        rets = {}
        for s in ctx.symbols:
            c = ctx.history(s, 67)["close"]
            if len(c) < 67:
                return
            rets[s] = c.iloc[-1] / c.iloc[0] - 1
        leader = max(rets, key=rets.get)
        for s in ctx.symbols:
            if s != leader and ctx.position(s) > 0:
                ctx.close_position(s)
        if ctx.position(leader) == 0:
            ctx.order_target_percent(0.95, leader)
````

</details>

## 37. `loose_ma_slope`

Prompt:

````text
be long when the 50 day moving average is going up, out when it's going down
````

Acceptable readings (matched by position direction):
- SMA50 above yesterday's SMA50
- SMA50 above its value 5 bars ago

<details><summary>SMA50 above yesterday's SMA50</summary>

````python
from backtester import Strategy

class Ref(Strategy):
    def on_bar(self, ctx):
        c = ctx.history(length=51)["close"]
        if len(c) < 51:
            return
        now, before = c.iloc[-50:].mean(), c.iloc[-50 - 1:-1].mean()
        if now > before and ctx.position() == 0:
            ctx.order_target_percent(0.95)
        elif now < before and ctx.position() > 0:
            ctx.close_position()
````

</details>

<details><summary>SMA50 above its value 5 bars ago</summary>

````python
from backtester import Strategy

class Ref(Strategy):
    def on_bar(self, ctx):
        c = ctx.history(length=55)["close"]
        if len(c) < 55:
            return
        now, before = c.iloc[-50:].mean(), c.iloc[-50 - 5:-5].mean()
        if now > before and ctx.position() == 0:
            ctx.order_target_percent(0.95)
        elif now < before and ctx.position() > 0:
            ctx.close_position()
````

</details>

## 38. `loose_decline_twitter`

Prompt:

````text
use twitter sentiment to time my entries on this stock
````

## 39. `loose_decline_fed`

Prompt:

````text
buy the day before the fed cuts rates and sell a week later
````
