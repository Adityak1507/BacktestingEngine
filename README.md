# Backtesting Engine

[![CI](https://github.com/Adityak1507/BacktestingEngine/actions/workflows/ci.yml/badge.svg)](https://github.com/Adityak1507/BacktestingEngine/actions/workflows/ci.yml)

A small, event-driven backtesting engine in Python. Write a trading strategy,
run it over historical OHLCV bars, and get an equity curve, a trade log and
performance metrics.

## Features

- **Event-driven loop**: bars are processed one at a time, like a live system.
- **No look-ahead**: strategies only see data up to the current bar, and orders
  placed on bar *t* fill on bar *t+1*. `ctx.history()` returns a copy, so even
  numpy's `.base` can't reach later bars.
- **Order types**: market, limit and stop orders, plus helpers such as
  `order_target_percent` and `close_position`.
- **Realistic costs**: percentage commission, minimum commission and slippage.
- **Cash and position limits**: buys are capped to available cash; shorting is
  opt-in.
- **Multiple symbols**: pass a dict of DataFrames and they are aligned on a
  common calendar.
- **Metrics**: total return, CAGR, volatility, Sharpe, Sortino, max drawdown
  and its duration, Calmar, win rate, average win and loss, profit factor, and
  commission paid, with a buy-and-hold benchmark for comparison.
- **Real market data** from Yahoo Finance via `yfinance` (optional).
- **Fast parameter sweeps** with a Numba-compiled path for signal-based
  strategies (optional; falls back to plain Python).
- **Web app**: a React + TypeScript frontend on a FastAPI backend, with
  backtests, an optimizer heatmap, light and dark themes.
- **Interactive charts** in Python with Plotly (optional).
- **LLM strategy agent** that turns plain-English trading ideas into `Strategy`
  code, tests it in a sandbox and submits it, running on local or free-tier
  open-weight models with rate-limit-aware failover between providers. It ships
  with a programmatic **eval harness** covering 39 cases, with hidden-data grading
  and a look-ahead detector: gpt-oss-120b passes 92%, Qwen2.5-Coder 7B 28%
  ([results](#results)).

## Install

```bash
pip install -e ".[dev]"     # core engine + pytest
pip install -e ".[all]"     # everything below
```

| Extra | Installs | Enables |
|---|---|---|
| `yahoo` | yfinance | `load_yahoo()` |
| `fast` | numba | compiled `backtester.fast` sweeps |
| `viz` | matplotlib, plotly | `result.plot()`, `result.plot_interactive()` |
| `api` | fastapi, uvicorn, yfinance, numba | the HTTP API behind the web app |

## Quick start

```python
from backtester import Backtest, SmaCrossover, generate_gbm, load_csv

data = generate_gbm(n_days=1260, seed=42)   # or load_csv("AAPL.csv")

result = Backtest(
    data,
    SmaCrossover(fast=20, slow=50),
    initial_cash=100_000,
    commission=0.001,   # 10 bps
    slippage=0.0005,    # 5 bps
).run()

print(result.summary())
result.trades        # DataFrame of closed trades
result.equity        # equity curve (pd.Series)
result.plot("equity.png")
```

Use real data and an interactive chart:

```python
from backtester import Backtest, SmaCrossover, load_yahoo

spy = load_yahoo("SPY", start="2015-01-01")   # adjusted for splits and dividends
result = Backtest(spy, SmaCrossover(50, 200), commission=0.0005).run()
result.plot_interactive("spy.html")           # open in a browser: zoom, hover, trade markers
```

Or run the example:

```bash
python examples/run_sma_crossover.py            # synthetic data
python examples/run_sma_crossover.py data.csv   # CSV with date,open,high,low,close,volume
```

## Writing a strategy

Subclass `Strategy` and implement `on_bar`. Everything goes through `ctx`:

```python
from backtester import Strategy

class Breakout(Strategy):
    def __init__(self, lookback=20):
        self.lookback = lookback

    def on_bar(self, ctx):
        bars = ctx.history(length=self.lookback + 1)
        if len(bars) <= self.lookback:
            return
        prior_high = bars["high"].iloc[:-1].max()
        if ctx.position() == 0 and ctx.price() > prior_high:
            ctx.order_target_percent(0.95)
        elif ctx.position() > 0 and ctx.price() < bars["low"].iloc[:-1].min():
            ctx.close_position()
```

| `ctx` member | Description |
|---|---|
| `history(symbol, length)` | OHLCV bars up to and including the current one |
| `price(symbol)` | Latest close |
| `position(symbol)`, `cash`, `equity` | Account state |
| `buy(qty)`, `sell(qty)`, `order(qty, limit_price=, stop_price=)` | Submit orders |
| `order_target_percent(pct)`, `order_target_quantity(qty)`, `close_position()` | Position sizing helpers |
| `cancel_orders(symbol)` | Cancel pending orders |
| `symbols`, `timestamp` | Universe and current bar time |

With a single symbol the `symbol` argument can be omitted.

## Web app

A React frontend (`frontend/`) talks to a FastAPI backend (`api/`).

![Backtest view](docs/backtest.png)
![Optimizer view](docs/optimize.png)

**Development** (two terminals, hot reload):

```bash
pip install -e ".[api]"
uvicorn api.main:app --reload            # API on :8000

cd frontend
npm install
npm run dev                              # UI on :5173, proxies /api to :8000
```

**Production** (one process): build the UI and FastAPI serves it.

```bash
cd frontend && npm run build && cd ..
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- **Backtest**: choose synthetic data, a Yahoo Finance ticker or a CSV, a
  strategy and costs. See KPIs, equity against buy & hold with trade markers,
  drawdown, full statistics and the trade log.
- **Optimize**: grid-search SMA crossover windows with the Numba fast path and
  explore the result as a heatmap (Sharpe, return or drawdown). Click any cell or
  row to backtest that pair.
- Press `Ctrl/⌘ + Enter` to run. The theme follows your OS and can be toggled.

### API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | | status, whether Numba is active |
| GET | `/api/strategies` | | strategies and their parameters |
| POST | `/api/backtest` | `{data, strategy, costs}` | metrics, equity/drawdown series, trades, fills |
| POST | `/api/sweep` | `{data, costs, fast_min, fast_max, slow_min, slow_max, step}` | grid of results |

`data` is `{"source": "synthetic", "days", "drift", "volatility", "seed"}`,
`{"source": "yahoo", "ticker", "start", "end"}` or
`{"source": "csv", "csv_text", "csv_name"}`. Interactive docs are at
`/docs` while the server is running.

## Strategy agent and eval

`agent/` contains an LLM agent that writes strategies for this engine. It talks to
any OpenAI-compatible endpoint, such as Ollama, vLLM, LM Studio, Groq, OpenRouter or
NVIDIA NIM.

```python
from agent import OpenAICompatModel, StrategyAgent

model = OpenAICompatModel("qwen2.5-coder:7b", base_url="http://localhost:11434/v1")
result = StrategyAgent(model).run("Buy when RSI(14) drops below 30, sell above 55")
print(result.outcome, result.code)   # "submitted" + code, or "declined" + reason
```

The agent has three tools: `run_backtest` tests code on synthetic data,
`submit_strategy` hands in the final version, and `decline` refuses requests that
need future data, data the engine doesn't have, or unsupported features. Generated
code is statically checked and run in a subprocess sandbox. The agent can also
recover tool calls that a model writes as plain JSON text, which is common with
open-weight models.

[`evals/strategy_agent/`](evals/strategy_agent/README.md) grades the agent on 39
cases. It runs each submission on hidden data and compares positions with
reference implementations. It also checks for look-ahead by perturbing every bar
after a cut-off and confirming that earlier orders don't change.

### Results

39 cases, 1 rep each, graded programmatically on hidden data (pass = runs, no
look-ahead and positions match the reference, or a correct decline).

| Model | Where it ran | Agent | Pass rate (95% CI) | Time per case | Tool calls |
|---|---|---|---|---|---|
| **gpt-oss-120b** | Groq free tier | v1 | **92%** (35/38, 79–97%)\* | 40 s | native |
| Qwen2.5-Coder 7B | local, Ollama on a 4 GB GPU | baseline | 28% (11/39, 17–44%) | 71 s | all written as text, recovered by the agent |
| Qwen2.5-Coder 7B | local, Ollama on a 4 GB GPU | v1 | 21% (8/39, 11–36%) | 133 s | all written as text, recovered by the agent |

\* One case could not be scored: Groq rejected a malformed tool call from the model.
The harness now hands such replies back to the model, and the case will be scored on
the next run.

Per category, gpt-oss-120b passed every trend, breakout, risk, calendar and
multi-asset case; Qwen2.5-Coder 7B passed none of the mean-reversion, breakout or
risk cases.

**What the runs showed**

- **Model capability dominates agent design.** With the same agent and eval,
  gpt-oss-120b wrote correct strategies for 31 of the 33 implementable cases and
  never wrongly declined one, in about two model turns per case. It also correctly
  declined 4 of the 5 requests it should refuse.
- **Small models decline instead of debugging.** Qwen2.5-Coder 7B declined 24 of the
  30 implementable cases. In 17 of those, it declined right after its own code raised
  an error, often with reasons that misread the engine ("`close_position()` is not a
  valid method", "yesterday's high is future information").
- **A prompt fix didn't fix it (negative result).** v1 told the agent to fix its
  errors instead of declining, and rejected a decline that came right after a failed
  test. Wrong declines barely moved (24 → 22) and the pass rate changed within noise
  (28% → 21%). Blocking the escape route also showed that the 7B model's correct
  declines were hollow: once it couldn't decline after a crash, it wrote code for 3 of
  the 4 requests it should refuse. One of them reads *yesterday's* close into a
  variable named `tomorrow_close`.
- **gpt-oss-120b's three failures** were a misread rule (three falling closes counted
  before today instead of ending at today), writing "3x leverage" code instead of
  declining (the engine caps buys at cash, so it would silently run at 1x), and a
  62-bar lookback for "the last 3 months" that the grader's accepted readings (60, 63
  and 66 bars) don't cover. The last one is arguably the grader being too strict.

Reports with per-case transcripts are built from `evals/strategy_agent/runs/`; see the
[eval README](evals/strategy_agent/README.md) for how grading works and was validated.

### Running it

```bash
pip install -e ".[agent]"
python -m evals.strategy_agent.run_eval --fake oracle            # harness check, ~100%
python -m evals.strategy_agent.run_eval --model qwen2.5-coder:7b # local model via Ollama

cp .env.example .env                                             # add free-tier API keys
python -m agent.providers check gpt-oss-120b                     # verify the chain
python -m evals.strategy_agent.run_eval --chain gpt-oss-120b --concurrency 2
```

Chains in `agent/chains.json` serve one open-weight model from several free
providers (Groq, OpenRouter, NVIDIA NIM, Hugging Face...). On a rate limit, the
request moves to the next provider or waits for the reset the provider reports;
when every quota is used up, the run stops cleanly and resumes later.

## Fast parameter sweeps (Numba)

The event loop runs Python on every bar, which is flexible but too slow for
searching thousands of parameter combinations. If a strategy can be written as
a precomputed signal (1 = long, 0 = flat, -1 = no change), `backtester.fast`
simulates it in compiled code with **the same execution rules** as `Backtest`
(next-bar-open fills, slippage, commission, whole shares, cash limit). A test
checks the two give identical equity curves.

```python
from backtester.fast import run_signals, sma_grid_search

grid = sma_grid_search(data, range(5, 101), range(20, 301), commission=0.001)
grid.head()   # one row per (fast, slow), best Sharpe first
```

On 10 years of daily bars, about 25,000 combinations run in under a second.
Watch out for overfitting: check the best parameters on data the sweep didn't
see.

## How a bar is processed

1. Pending orders fill at this bar's open, or at their limit or stop price if
   the bar's range reaches it. Sells are processed before buys.
2. The portfolio is marked to market at the close.
3. `strategy.on_bar(ctx)` runs and may submit new orders.

## Project layout

```
backtester/
  data.py        CSV and Yahoo Finance loading, validation, synthetic GBM data
  broker.py      orders, cost model, order matching
  portfolio.py   cash, positions, average cost, realised trades
  strategy.py    Strategy base class and Context API
  strategies.py  BuyAndHold, SmaCrossover, RsiMeanReversion
  metrics.py     performance statistics
  engine.py      Backtest event loop and BacktestResult (+ charts)
  fast.py        Numba-compiled signal backtests and grid search
agent/           LLM strategy agent: tools, sandbox, OpenAI-compatible client,
                 free-tier provider chains with failover
evals/           eval harness for the agent (cases, grader, runner)
api/main.py      FastAPI backend
frontend/        React + TypeScript UI (Vite, Recharts)
examples/        runnable example
tests/           pytest suite
```

## Tests

```bash
pytest                                     # engine, fast path, API, agent, providers
python -m evals.strategy_agent.selfcheck   # eval harness: oracle must score 100%, null 0%
cd frontend && npm run build               # type-checks and builds the UI
```

GitHub Actions runs all three on every push and pull request (`.github/workflows/ci.yml`):
the test suite with an error-only lint on Python 3.11-3.13, the eval self-check (no LLM
calls), and the frontend build.

## Limitations

- Bar-level simulation only: there is no intrabar ordering of fills, and if a
  limit or stop is hit the fill is assumed to be complete.
- No margin, borrow costs or dividends, and short positions are not
  margin-checked.
- Orders are good-till-cancelled; there are no time-in-force options. Stop-limit
  orders (both `limit_price` and `stop_price`) are rejected.
- The Numba fast path handles single-asset long/flat signals only. Use
  `Backtest` for anything else.
