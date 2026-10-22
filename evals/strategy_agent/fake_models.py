"""Scripted stand-ins for an LLM, used to check the harness end to end for free.

- OracleModel submits each case's reference implementation (or declines when the
  case expects it): the eval should score ~100%.
- NullModel submits a strategy that never trades: the eval should score ~0%
  on code cases (it still "passes" nothing, since declines are not submitted).
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Dict, List

from agent.llm import ModelTurn, ToolCall

from .cases import CASES

NEVER_TRADES = "from backtester import Strategy\n\nclass Idle(Strategy):\n    def on_bar(self, ctx):\n        pass\n"


class _Scripted:
    name = "fake"
    _ids = itertools.count()

    def _call(self, name: str, args: Dict[str, Any]) -> ModelTurn:
        tid = f"call_{next(self._ids)}"
        return ModelTurn("", [ToolCall(tid, name, args, json.dumps(args))], self.name, "tool_calls",
                         {"input_tokens": 0, "output_tokens": 0})


class OracleModel(_Scripted):
    name = "fake-oracle"

    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> ModelTurn:
        request = messages[1]["content"]
        case = next(c for c in CASES if request.startswith(c["prompt"]))
        if case.get("expect") == "decline":
            return self._call("decline", {"reason": "needs data the engine cannot provide"})
        return self._call("submit_strategy", {"code": case.get("reference") or case["references"][0]})


class NullModel(_Scripted):
    name = "fake-null"

    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> ModelTurn:
        return self._call("submit_strategy", {"code": NEVER_TRADES})


FAKES = {"oracle": OracleModel, "null": NullModel}
