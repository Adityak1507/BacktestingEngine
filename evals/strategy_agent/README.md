# Strategy agent eval

Measures how well an LLM agent turns a plain-English trading idea into a correct
`backtester.Strategy`. The agent (`agent/`) writes code, tests it with a
`run_backtest` tool on synthetic data, then calls `submit_strategy` or `decline`.
It talks to any OpenAI-compatible endpoint, so it runs against local open-weight
models (Ollama, vLLM, LM Studio) or hosted ones (Groq, Together, OpenRouter).

## Cases

39 cases ([CASES.md](CASES.md)):

| Category | Count | What it tests |
|---|---|---|
| trend, mean_reversion, breakout | 16 | indicators with precise definitions (SMA/EMA/MACD/RSI/Bollinger/Donchian) |
| risk | 5 | trailing stops, take-profit/stop-loss, limit orders with expiry, volatility sizing, scaling in |
| calendar, multi_asset | 5 | month/weekday rules, rotation, rebalancing, pairs ratio |
| decline | 4 | requests that need future data, missing data or unsupported leverage |
| loose | 9 | casual wording with several acceptable readings, two of which must be declined |

## Grading (all programmatic, no LLM judge)

Submitted code runs in a sandbox (`agent/sandbox.py`: static checks plus a subprocess with
restricted builtins and a timeout) on **hidden** datasets: three seeded GBM paths with
different regimes, none of which the agent saw while testing.

| Metric | Meaning |
|---|---|
| **pass** | headline: runs, no look-ahead, and agreement at or above the threshold; or a correct decline |
| valid | passes static checks, defines a `Strategy`, constructs with no arguments |
| runs | backtests every hidden dataset without error or timeout |
| no_lookahead | orders up to bar 400 are identical when every later bar is replaced by an unrelated path |
| agreement | worst case over datasets of the share of *active* bars (either side holds a position) on which the agent's exposure is within 0.10 of the reference |

Precise cases pass at agreement ≥ 99.5%. Loose cases compare position *direction* against
the closest acceptable reading and pass at ≥ 95%.

**How the grader was validated.**
- **Oracle and null:** submitting each case's reference scores 100%; a strategy that never trades scores 0%.
- **Mutants:** 15 deliberately wrong variants were tried, such as a wrong window, `ddof`, hold
  period, threshold, sizing or EMA convention. All fail except one: rebalancing every 21 bars
  instead of 20, which no position-based check can see.
- **Equivalents:** correct code written differently (rolling vs. slicing, a hand-written RSI loop,
  rebalancing every bar, 99% vs. 95% sizing) still passes. The equivalents are written for the
  precise prompts, which state the 95% size; on those prompts, 99% passes only because it stays
  within the 0.10 exposure tolerance.

**Known limitations.**
- A decline is graded on the outcome only. A model can decline for the wrong reason (in the pilot,
  qwen2.5-coder declined the "tomorrow's close" case only after its own code crashed). The reason
  is recorded in each row's `explanation.pass` and `meta.detail.decline_reason`; spot-check it.
- Rebalance timing differences that barely move positions (e.g. every 21 vs 20 bars) are invisible
  to a position-based comparison.
- The sandbox is a guard against mistakes and casual cheating, not a security boundary. Run
  untrusted models in a container.

## Running it

```bash
pip install -e ".[agent]"

# Free harness checks: expect ~100% and ~0%
python -m evals.strategy_agent.run_eval --fake oracle
python -m evals.strategy_agent.run_eval --fake null

# Local model through Ollama (ollama pull qwen2.5-coder:7b)
python -m evals.strategy_agent.run_eval --model qwen2.5-coder:7b

# Hosted open-weight model
python -m evals.strategy_agent.run_eval --model <model-id> \
    --base-url https://api.groq.com/openai/v1 --api-key-env GROQ_API_KEY --concurrency 4
```

Results go to `runs/<model>/<variant>/`:
- `results.jsonl`: one graded row per case and rep, with explanations, token usage and latency
- `traces/`: the full agent transcript per case
- `errors.jsonl`: API errors and timeouts, kept separate so they are never scored as failures

Runs resume where they stopped. The runner refuses real-model runs until the current agent
and grader code is approved with `--approve-harness`, so scores from different harness
versions are never mixed.

With 39 cases × 1 rep the pass rate is only accurate to about ±15 points (95% CI). Use
`--reps 3` before comparing two models.
