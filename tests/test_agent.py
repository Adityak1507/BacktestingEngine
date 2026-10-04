"""Tests for the strategy agent, its sandbox and the eval grader (no LLM needed)."""

import json

import pytest

from agent import BacktestSettings, StrategyAgent, validate_code
from agent.llm import ModelTurn, ToolCall
from agent.sandbox import gbm_dataset, run_job
from evals.strategy_agent.cases import CASES, CASES_BY_ID
from evals.strategy_agent.grading import agreement, grade, lookahead_violations

IDLE = "from backtester import Strategy\n\nclass Idle(Strategy):\n    def on_bar(self, ctx):\n        pass\n"


# --- static checks -------------------------------------------------------------
@pytest.mark.parametrize(
    "snippet, message",
    [
        ("import os", "import of 'os'"),
        ("from subprocess import run", "import from 'subprocess'"),
        ("x = open('f')", "'open'"),
        ("x = getattr(1, 'real')", "'getattr'"),
        ("def f(ctx):\n    return ctx._engine.data", "private attribute '_engine'"),
        ("x = (1).__class__", "private attribute '__class__'"),
        ("def f():\n    global y", "global"),
    ],
)
def test_static_checks_reject_unsafe_code(snippet, message):
    problems = validate_code(IDLE + "\n" + snippet + "\n")
    assert any(message in p for p in problems), problems


def test_static_checks_allow_own_private_state_and_normal_imports():
    code = IDLE.replace("pass", "self._last = ctx.price()") + "import numpy as np\nimport pandas as pd\n"
    assert validate_code(code) == []


def test_syntax_errors_are_reported():
    assert "SyntaxError" in validate_code("class X(:\n")[0]


# --- sandbox -------------------------------------------------------------------------
def test_sandbox_reports_runtime_errors_with_stage():
    code = IDLE.replace("pass", "raise ValueError('boom')")
    out = run_job(code, BacktestSettings(), [gbm_dataset(1, BacktestSettings(), n_days=60)])
    assert not out["ok"] and out["stage"] == "run" and "boom" in out["error"]


def test_sandbox_rejects_strategy_needing_constructor_args():
    code = IDLE.replace("    def on_bar", "    def __init__(self, n):\n        self.n = n\n\n    def on_bar")
    out = run_job(code, BacktestSettings(), [gbm_dataset(1, BacktestSettings(), n_days=60)])
    assert not out["ok"] and out["stage"] == "load"


def test_sandbox_enforces_timeout():
    code = IDLE.replace("pass", "while True:\n            pass")
    out = run_job(code, BacktestSettings(), [gbm_dataset(1, BacktestSettings(), n_days=60)], timeout_s=5)
    assert out.get("timeout")


def test_restricted_builtins_block_dynamic_import():
    code = IDLE.replace("pass", "import pathlib")
    assert any("pathlib" in p for p in validate_code(code))


# --- grader ------------------------------------------------------------------------------
def test_lookahead_detector_flags_strategies_that_see_the_future():
    settings = BacktestSettings()
    base = gbm_dataset(5, settings, n_days=300)
    perturbed = gbm_dataset(5, settings, n_days=300, perturb_after=200)
    honest = CASES_BY_ID["sma_cross"]["reference"]
    out = run_job(honest, settings, [base, perturbed])
    assert lookahead_violations(out["runs"][0], out["runs"][1], cut=200) == []

    # A "strategy" whose orders depend on the final price of the dataset.
    def fake_run(final_price):
        orders = [{"t": "2020-01-02", "symbol": "ASSET", "qty": 1.0 if final_price > 100 else -1.0, "limit": None, "stop": None}]
        return {"dates": ["2020-01-01", "2020-01-02", "2020-01-03"], "orders": orders}

    assert lookahead_violations(fake_run(150), fake_run(50), cut=1)


def test_agreement_ignores_bars_where_both_are_flat():
    ref = {"exposure": {"A": [0, 0, 0, 0.95, 0.95]}}
    assert agreement({"exposure": {"A": [0, 0, 0, 0.95, 0.95]}}, ref) == 1.0
    assert agreement({"exposure": {"A": [0, 0, 0, 0.0, 0.95]}}, ref) == 0.5


def test_every_case_is_well_formed():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)) >= 30
    for c in CASES:
        assert c["tags"] and c["prompt"]
        refs = c.get("references") or ([c["reference"]] if "reference" in c else [])
        assert bool(refs) != (c.get("expect") == "decline"), c["id"]
        for code in refs:
            assert validate_code(code) == [], c["id"]


def test_grading_a_reference_passes_and_idle_fails():
    case = CASES_BY_ID["bollinger"]
    assert grade(case, "submitted", case["reference"])["grade"]["pass"] == 1.0
    assert grade(case, "submitted", IDLE)["grade"]["pass"] == 0.0
    assert grade(case, "declined", None)["grade"]["pass"] == 0.0
    assert grade(CASES_BY_ID["decline_earnings"], "declined", None)["grade"]["pass"] == 1.0


# --- agent loop -------------------------------------------------------------------------
class Script:
    """A fake model that replays a fixed list of turns."""

    name = "scripted"

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def chat(self, messages, tools):
        self.seen.append(json.loads(json.dumps(messages, default=str)))
        return self.turns.pop(0)


def call(name, **args):
    return ModelTurn("", [ToolCall(f"id_{name}", name, args, json.dumps(args))], "scripted", "tool_calls",
                     {"input_tokens": 10, "output_tokens": 5})


def test_agent_tests_then_submits():
    code = CASES_BY_ID["sma_cross"]["reference"]
    model = Script([call("run_backtest", code=code), call("submit_strategy", code=code)])
    res = StrategyAgent(model).run("SMA crossover please")
    assert res.outcome == "submitted" and res.code == code
    assert res.turns == 2 and res.dev_runs == 2 and res.usage == {"input_tokens": 20, "output_tokens": 10}
    tool_msg = model.seen[1][-1]
    assert tool_msg["role"] == "tool" and "Backtest OK" in tool_msg["content"]


def test_broken_submission_is_bounced_back_to_the_model():
    broken = IDLE.replace("pass", "return ctx.nope()")
    fixed = CASES_BY_ID["sma_cross"]["reference"]
    model = Script([call("submit_strategy", code=broken), call("submit_strategy", code=fixed)])
    res = StrategyAgent(model).run("SMA crossover please")
    assert res.outcome == "submitted" and res.code == fixed
    assert "Submission rejected" in model.seen[1][-1]["content"]


def test_agent_nudges_text_replies_then_gives_up():
    text = ModelTurn("Here is some code...", [], "scripted", "stop", {})
    res = StrategyAgent(Script([text] * 3), max_nudges=2).run("anything")
    assert res.outcome == "gave_up" and res.turns == 3


def test_agent_records_declines():
    res = StrategyAgent(Script([call("decline", reason="needs tomorrow's prices")])).run("cheat")
    assert res.outcome == "declined" and "tomorrow" in res.reason


def test_tool_calls_written_as_text_are_recovered():
    code = CASES_BY_ID["sma_cross"]["reference"]
    raw = json.dumps({"name": "submit_strategy", "arguments": {"code": code}})
    text = ModelTurn(f"Sure!\n```json\n{raw}\n```", [], "scripted", "stop", {})
    res = StrategyAgent(Script([text])).run("SMA crossover please")
    assert res.outcome == "submitted" and res.code == code and res.text_tool_calls == 1


def test_text_tool_call_parser_tolerates_raw_newlines_and_ignores_unknown_names():
    from agent.llm import parse_text_tool_calls

    text = '<tool_call>{"name": "run_backtest", "arguments": {"code": "line1\nline2"}}</tool_call> {"name": "rm_rf"}'
    calls = parse_text_tool_calls(text, {"run_backtest"})
    assert [(c.name, c.arguments["code"]) for c in calls] == [("run_backtest", "line1\nline2")]
