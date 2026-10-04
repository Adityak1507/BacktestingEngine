"""The strategy-writing agent: prompt in, validated `Strategy` code (or a reasoned decline) out."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm import ChatModel, ModelTurn, parse_text_tool_calls
from .sandbox import BacktestSettings, gbm_dataset, run_job, validate_code

SYSTEM_PROMPT = """\
You write trading strategies for a Python event-driven backtesting engine.
The user describes a strategy in plain English. Implement it exactly as specified, test it with
the run_backtest tool, then call submit_strategy with the final code.

# The engine
Subclass `backtester.Strategy` and implement `on_bar(self, ctx)`. Optionally implement
`init(self, ctx)` for set-up. The class must be constructible with no arguments
(give any parameters default values). Each bar is processed in this order:
1. Orders submitted on the previous bar fill at this bar's open (limit/stop orders fill when
   the bar's range reaches their price).
2. The portfolio is marked to market at this bar's close.
3. `on_bar(ctx)` runs; orders it submits fill on the NEXT bar.

`ctx` API (only these public members exist):
- `ctx.history(symbol=None, length=None)` -> pandas DataFrame of OHLCV bars (columns open, high,
  low, close, volume) up to and including the current bar, indexed by date.
- `ctx.price(symbol=None)` latest close; `ctx.timestamp` current bar's pandas Timestamp.
- `ctx.position(symbol=None)` shares held (negative if short); `ctx.cash`; `ctx.equity`.
- `ctx.buy(qty, symbol=None, limit_price=None, stop_price=None)`, `ctx.sell(...)` (qty > 0),
  `ctx.order(signed_qty, symbol=None, limit_price=None, stop_price=None)`.
- `ctx.order_target_percent(pct, symbol=None)` size the position to pct of equity (whole shares;
  negative pct = short). `ctx.order_target_quantity(qty, symbol=None)`; `ctx.close_position(symbol=None)`.
- `ctx.cancel_orders(symbol=None)` cancels pending orders. Orders are good-till-cancelled.
- `ctx.symbols` list of symbols. With one symbol, `symbol` arguments can be omitted.
Stop-limit orders (both limit_price and stop_price) are not supported.

# Rules
- Only import from: backtester, math, statistics, numpy, pandas, collections, itertools,
  functools, dataclasses, typing. No file, network or OS access.
- Never access private attributes (anything starting with `_`) of ctx or other objects;
  `self._name` on your own strategy is fine.
- Never use information from the future: decisions on a bar may only use bars up to that bar.
- Follow the user's definitions exactly (window lengths, inclusive/exclusive ranges, sizing,
  indicator formulas, when to enter and exit). Keep state on `self` when the rules need it.
- Not future information: the current bar's open, high, low and close, and every earlier bar.
  Future information means bars AFTER the current one (e.g. "tomorrow's close").
- Indicators need history: start `on_bar` with a warm-up guard such as
  `if len(ctx.history()) < N: return` so early bars don't raise IndexError or use partial windows.

# Errors and declining
- An error from run_backtest is a bug in YOUR code, never a reason to decline. Read the traceback,
  fix the code and run run_backtest again. Repeat until it runs, then submit.
- No trades on the test data is also not a reason to decline: check the logic, and submit if it
  matches the request.
- Call `decline` ONLY when the request itself cannot be implemented faithfully with this engine and
  OHLCV data: it needs future information, data the engine does not have (fundamentals, news,
  earnings, sentiment, economic events...), or unsupported features such as leverage beyond
  available cash. Decide this from the request, before writing code; give that reason.

Respond only with tool calls."""

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_backtest",
            "description": (
                "Run strategy code on a synthetic test dataset with the same symbols and settings as "
                "the real backtest. Returns errors with a traceback, or the number of trades, total "
                "return, time in market and the first fills, so you can check the logic behaves as intended."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Complete Python source defining one Strategy subclass."}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_strategy",
            "description": "Submit the final strategy code. It is checked and run once more; if it fails you get the error back.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Complete Python source defining one Strategy subclass."}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "decline",
            "description": "Decline a request that cannot be implemented faithfully (needs future data, unavailable data, or unsupported features).",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]

DEV_SEED = 7  # the agent's test data; graders use different seeds


@dataclass
class AgentResult:
    outcome: str  # "submitted" | "declined" | "gave_up"
    code: Optional[str] = None
    reason: Optional[str] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)  # eval-report trace format
    turns: int = 0
    tool_calls: int = 0
    dev_runs: int = 0
    text_tool_calls: int = 0  # tool calls recovered from plain text (model didn't use the tool API)
    decline_pushbacks: int = 0  # declines rejected because the agent's own last test had failed
    providers: List[str] = field(default_factory=list)  # provider that served each model turn
    served_models: List[str] = field(default_factory=list)  # model id each turn was served as
    usage: Dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    model: str = ""
    stop_reason: str = ""
    retries: int = 0
    latency_s: float = 0.0


class StrategyAgent:
    def __init__(self, model: ChatModel, max_turns: int = 10, max_nudges: int = 2, sandbox_timeout_s: float = 60.0,
                 recover_text_tool_calls: bool = True, max_decline_pushbacks: int = 1):
        self.model = model
        self.recover_text_tool_calls = recover_text_tool_calls
        # Small models often "decline" as soon as their own code crashes. A decline that comes
        # right after a failed test is bounced back this many times before it is accepted.
        self.max_decline_pushbacks = max_decline_pushbacks
        self.max_turns = max_turns
        self.max_nudges = max_nudges
        self.sandbox_timeout_s = sandbox_timeout_s

    def run(self, request: str, settings: Optional[BacktestSettings] = None) -> AgentResult:
        settings = settings or BacktestSettings()
        user = f"{request}\n\n# Backtest settings\n{settings.describe()}"
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        self._last_test_error: Optional[str] = None
        res = AgentResult(outcome="gave_up", trace=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ])
        started = time.perf_counter()
        nudges = 0
        for _ in range(self.max_turns):
            turn = self.model.chat(messages, TOOLS)
            self._account(res, turn)
            if not turn.tool_calls and self.recover_text_tool_calls:
                recovered = parse_text_tool_calls(turn.content, {t["function"]["name"] for t in TOOLS})
                if recovered:
                    res.text_tool_calls += len(recovered)
                    if self._run_text_calls(turn, recovered, messages, settings, res):
                        break
                    continue
            if not turn.tool_calls:
                res.trace.append({"role": "assistant", "content": turn.content})
                messages.append({"role": "assistant", "content": turn.content})
                nudges += 1
                if nudges > self.max_nudges:
                    break
                reminder = ("Please respond with a tool call: run_backtest to test code, submit_strategy "
                            "to finish, or decline if the request can't be implemented.")
                messages.append({"role": "user", "content": reminder})
                res.trace.append({"role": "user", "content": reminder})
                continue

            messages.append({
                "role": "assistant",
                "content": turn.content or None,
                "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments or json.dumps(c.arguments)}}
                    for c in turn.tool_calls
                ],
            })
            if turn.content:
                res.trace.append({"role": "assistant", "content": turn.content})
            done = False
            for call in turn.tool_calls:
                res.tool_calls += 1
                res.trace.append({"role": "tool_call", "name": call.name,
                                  "content": call.arguments.get("code") or json.dumps(call.arguments, indent=2)})
                output, finished = self._handle(call, settings, res)
                res.trace.append({"role": "tool_result", "content": output})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
                done = done or finished
            if done:
                break
        res.latency_s = time.perf_counter() - started
        return res

    # --- helpers -----------------------------------------------------------------
    def _run_text_calls(self, turn: ModelTurn, calls, messages, settings, res) -> bool:
        """Execute tool calls recovered from text; results go back as a user message.

        The model never produced API tool calls, so its own text stays in the history
        verbatim and the results are fed back the same way.
        """
        messages.append({"role": "assistant", "content": turn.content})
        results, done = [], False
        for call in calls:
            res.tool_calls += 1
            res.trace.append({"role": "tool_call", "name": call.name,
                              "content": call.arguments.get("code") or json.dumps(call.arguments, indent=2)})
            output, finished = self._handle(call, settings, res)
            res.trace.append({"role": "tool_result", "content": output})
            results.append(f"Result of {call.name}:\n{output}")
            done = done or finished
        if not done:
            messages.append({"role": "user", "content": "\n\n".join(results)})
        return done

    @staticmethod
    def _account(res: AgentResult, turn: ModelTurn) -> None:
        res.turns += 1
        res.model = turn.model
        res.served_models.append(turn.model)
        if turn.provider:
            res.providers.append(turn.provider)
        res.stop_reason = turn.finish_reason
        res.retries += turn.retries
        for k, v in turn.usage.items():
            res.usage[k] = res.usage.get(k, 0) + v

    def _handle(self, call, settings: BacktestSettings, res: AgentResult):
        """Execute one tool call; returns (tool result text, whether the episode is over)."""
        if call.parse_error:
            return f"Error: {call.parse_error}. Send the arguments as a JSON object.", False
        if call.name == "decline":
            if self._last_test_error and res.decline_pushbacks < self.max_decline_pushbacks:
                res.decline_pushbacks += 1
                return ("Decline not accepted yet: your last test failed because of a bug in your own code, "
                        "which is not a reason to decline. Fix it and call run_backtest again.\n"
                        f"The error was:\n{self._last_test_error}\n"
                        "Only decline if the request itself needs future bars, data the engine doesn't "
                        "have, or unsupported features."), False
            res.outcome, res.reason = "declined", str(call.arguments.get("reason", ""))
            return "Declined.", True
        if call.name not in ("run_backtest", "submit_strategy"):
            return f"Error: unknown tool {call.name!r}.", False
        code = call.arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return "Error: `code` must be a non-empty string of Python source.", False

        res.dev_runs += 1
        report, ok = self.dev_backtest(code, settings)
        self._last_test_error = None if ok else report
        if call.name == "submit_strategy":
            if not ok:
                return "Submission rejected - fix the problem and submit again.\n" + report, False
            res.outcome, res.code = "submitted", code
            return "Submitted.\n" + report, True
        return report, False

    def dev_backtest(self, code: str, settings: BacktestSettings):
        problems = validate_code(code)
        if problems:
            return "Static check failed:\n- " + "\n- ".join(problems), False
        out = run_job(code, settings, [gbm_dataset(DEV_SEED, settings, n_days=500)], self.sandbox_timeout_s)
        if not out.get("ok"):
            detail = "\n".join(out.get("problems", [])) or out.get("traceback", "")
            return f"Backtest failed: {out.get('error')}\n{detail}".strip(), False
        run = out["runs"][0]
        bars = len(run["dates"])
        in_market = {
            sym: sum(1 for e in exp if abs(e) > 1e-9) / bars for sym, exp in run["exposure"].items()
        }
        lines = [
            f"Backtest OK on {bars} bars of synthetic data.",
            f"Orders submitted: {len(run['orders'])}, fills: {run['num_fills']}, closed trades: {run['num_trades']}, "
            f"total return: {run['total_return']:.2%}",
            "Time in market: " + ", ".join(f"{s} {v:.0%}" for s, v in in_market.items()),
        ]
        if run["fills_head"]:
            lines.append("First fills: " + "; ".join(
                f"{f['date']} {f['symbol']} {f['qty']:+g} @ {f['price']}" for f in run["fills_head"]))
        else:
            lines.append("WARNING: no orders were filled. Check that the entry conditions can trigger.")
        return "\n".join(lines), True
