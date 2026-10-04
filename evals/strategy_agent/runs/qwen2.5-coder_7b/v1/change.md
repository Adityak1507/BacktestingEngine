Fix your own errors instead of declining (prompt rules + one decline pushback)

Baseline finding: qwen2.5-coder:7b declined 24 of the 30 implementable cases; 17 of those
declines came right after its own run_backtest failed, and several reasons misread the
engine ("yesterday's high is future information", "close_position() is not a valid method").

Changes in agent/agent.py:
- System prompt: an error from run_backtest is a bug to fix, not a reason to decline;
  "no trades" is not a reason to decline; decline only when the request itself needs future
  bars, unavailable data or unsupported features, decided before writing code. Spells out
  that the current bar's OHLC and all earlier bars are not future data, and shows the
  warm-up guard `if len(ctx.history()) < N: return`.
- Guardrail: a decline issued right after a failed test run is rejected once, with the error
  sent back; a second decline is accepted, so genuine declines still work.
- Records `decline_pushbacks` per case (run_eval.py: new perf column, no grading change).
