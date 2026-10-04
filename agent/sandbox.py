"""Check and run model-written strategy code away from the main process.

Two layers keep generated code in its lane:

1. `validate_code` statically rejects imports outside an allowlist, dangerous
   builtins (`open`, `exec`, `getattr`, ...) and any `_private` / `__dunder__`
   attribute access, which is the only route from `ctx` to future data.
2. `run_job` executes the code in a fresh Python subprocess with restricted
   builtins and a wall-clock timeout, so a crash or infinite loop can't take
   down the caller.

This is a guard against mistakes and casual cheating, not a security boundary
for hostile code; run untrusted models inside a container for that.
"""

from __future__ import annotations

import ast
import builtins
import json
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ALLOWED_IMPORTS = {
    "backtester", "math", "statistics", "numpy", "pandas",
    "collections", "itertools", "functools", "dataclasses", "typing",
}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help",
    "memoryview", "object",
}
SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "frozenset",
        "int", "isinstance", "issubclass", "len", "list", "map", "max", "min", "print",
        "range", "reversed", "round", "set", "slice", "sorted", "str", "sum", "super",
        "tuple", "zip", "ValueError", "ZeroDivisionError", "KeyError", "IndexError",
        "Exception", "TypeError", "ArithmeticError", "NotImplementedError", "property",
        "staticmethod", "classmethod", "None", "True", "False",
    )
    if hasattr(builtins, name)
}
REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_CODE_CHARS = 20_000


@dataclass
class BacktestSettings:
    """How a strategy is backtested: universe, shorting and starting cash."""

    symbols: List[str] = field(default_factory=lambda: ["ASSET"])
    allow_short: bool = False
    initial_cash: float = 100_000.0

    def describe(self) -> str:
        syms = ", ".join(self.symbols)
        single = len(self.symbols) == 1
        return (
            f"Symbols: {syms}" + (" (single symbol, so `symbol` arguments can be omitted)" if single else "")
            + f"\nShort selling: {'allowed' if self.allow_short else 'NOT allowed (sells are capped at the position held)'}"
            + f"\nInitial cash: {self.initial_cash:,.0f}; no commission or slippage; daily bars"
        )


# --- static checks -----------------------------------------------------------
def validate_code(code: str) -> List[str]:
    """Return a list of problems with `code`; empty means it may be executed."""
    if len(code) > MAX_CODE_CHARS:
        return [f"code is longer than {MAX_CODE_CHARS} characters"]
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc.msg} (line {exc.lineno})"]

    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"line {node.lineno}: import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"line {node.lineno}: import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"line {node.lineno}: use of {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_") and not (
            # The strategy's own state (`self._prev`) is fine; dunders and other objects' privates are not.
            isinstance(node.value, ast.Name) and node.value.id == "self" and not node.attr.startswith("__")
        ):
            problems.append(
                f"line {node.lineno}: access to private attribute {node.attr!r} is not allowed;"
                " use the public ctx API"
            )
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f"line {node.lineno}: global/nonlocal statements are not allowed")
    if not any(isinstance(n, ast.ClassDef) for n in tree.body):
        problems.append("no class defined; define a subclass of backtester.Strategy")
    return problems


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def load_strategy_class(code: str):
    """Execute validated code and return the last `Strategy` subclass it defines."""
    from backtester import Strategy

    problems = validate_code(code)
    if problems:
        raise ValueError("; ".join(problems))
    namespace: Dict[str, Any] = {
        "__builtins__": {**SAFE_BUILTINS, "__import__": _safe_import, "__build_class__": builtins.__build_class__},
        "__name__": "agent_strategy",
    }
    exec(compile(code, "<strategy>", "exec"), namespace)  # noqa: S102 - validated above
    classes = [
        v for v in namespace.values()
        if isinstance(v, type) and issubclass(v, Strategy) and v is not Strategy
    ]
    if not classes:
        raise ValueError("no subclass of backtester.Strategy found")
    return classes[-1]


# --- data ------------------------------------------------------------------------
def make_data(spec: Dict[str, Any]):
    """Build price data from a JSON-able spec (see `run_job`)."""
    from backtester import generate_gbm

    n = spec.get("n_days", 750)
    frames = {}
    for sym, p in spec["symbols"].items():
        df = generate_gbm(n, mu=p.get("mu", 0.08), sigma=p.get("sigma", 0.2), seed=p["seed"],
                          start_price=p.get("start_price", 100.0))
        cut = spec.get("perturb_after")
        if cut is not None:
            # Replace everything after bar `cut` with an unrelated path that starts
            # at the same price. Decisions up to `cut` must not change.
            alt = generate_gbm(n - cut - 1, mu=-p.get("mu", 0.08), sigma=p.get("sigma", 0.2) * 1.5,
                               seed=p["seed"] + 10_000, start_price=float(df["close"].iloc[cut]))
            df.iloc[cut + 1:] = alt.to_numpy()
        frames[sym] = df
    return frames


# --- execution (runs inside the subprocess) ------------------------------------------
def _execute(job: Dict[str, Any], stage: Dict[str, str]) -> Dict[str, Any]:
    import numpy as np

    from backtester import Backtest, broker

    settings = BacktestSettings(**job["settings"])
    cls = load_strategy_class(job["code"])
    cls()  # must be constructible with no arguments
    stage["name"] = "run"
    runs = []
    for data_spec in job["datasets"]:
        frames = make_data(data_spec)
        orders: List[Dict[str, Any]] = []
        original_submit = broker.Broker.submit

        def logging_submit(self, order, _orig=original_submit):
            orders.append({"t": str(order.created_at.date()), "symbol": order.symbol,
                           "qty": order.quantity, "limit": order.limit_price, "stop": order.stop_price})
            _orig(self, order)

        broker.Broker.submit = logging_submit
        try:
            strategy = cls()
            result = Backtest(frames, strategy, initial_cash=settings.initial_cash,
                              allow_short=settings.allow_short).run()
        finally:
            broker.Broker.submit = original_submit

        index = result.equity.index
        exposure = {}
        for sym, df in frames.items():
            fills = result.fills
            qty = (
                fills[fills["symbol"] == sym].groupby("timestamp")["quantity"].sum()
                if not fills.empty else None
            )
            pos = (qty.reindex(index, fill_value=0.0).cumsum() if qty is not None
                   else np.zeros(len(index)))
            exposure[sym] = list(np.asarray(pos) * df["close"].reindex(index).to_numpy()
                                 / result.equity.to_numpy())
        runs.append({
            "dates": [str(d.date()) for d in index],
            "exposure": exposure,
            "orders": orders,
            "num_fills": len(result.fills),
            "num_trades": int(result.metrics["num_trades"]),
            "total_return": float(result.metrics["total_return"]),
            "fills_head": [
                {"date": str(f["timestamp"].date()), "symbol": f["symbol"],
                 "qty": float(f["quantity"]), "price": round(float(f["price"]), 4)}
                for f in result.fills.head(6).to_dict("records")
            ] if not result.fills.empty else [],
        })
    return {"ok": True, "runs": runs}


def worker_main() -> None:
    """Entry point of the sandbox subprocess: JSON job on stdin, JSON result on stdout."""
    job = json.loads(sys.stdin.read())
    stage = {"name": "load"}  # "load" = defining/instantiating the class failed, "run" = the backtest did
    try:
        out = _execute(job, stage)
    except Exception as exc:  # report the strategy's own error back to the caller
        # Show the model only the frames inside its own code, plus the error itself.
        frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == "<strategy>"]
        lines = [f'  File "<strategy>", line {f.lineno}, in {f.name}' for f in frames[-5:]]
        code_lines = job.get("code", "").splitlines()
        for f, text in zip(frames[-5:], list(lines)):
            if f.lineno and 0 < f.lineno <= len(code_lines):
                lines[lines.index(text)] = f"{text}\n    {code_lines[f.lineno - 1].strip()}"
        lines.append(f"{type(exc).__name__}: {exc}")
        out = {"ok": False, "stage": stage["name"], "error": f"{type(exc).__name__}: {exc}",
               "traceback": "\n".join(lines)}
    sys.stdout.write(json.dumps(out))


def run_job(code: str, settings: BacktestSettings, datasets: List[Dict[str, Any]],
            timeout_s: float = 60.0) -> Dict[str, Any]:
    """Run `code` against each dataset spec in a subprocess and return the per-run records.

    A dataset spec is ``{"symbols": {sym: {"seed", "mu", "sigma"}}, "n_days": int,
    "perturb_after": Optional[int]}``. The result has ``ok`` plus either ``runs`` or
    ``error``/``traceback``; ``timeout`` is set when the wall-clock limit was hit.
    """
    problems = validate_code(code)
    if problems:
        return {"ok": False, "stage": "static", "error": "rejected by static checks", "problems": problems}
    job = {"code": code, "settings": asdict(settings), "datasets": datasets}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "from agent.sandbox import worker_main; worker_main()"],
            input=json.dumps(job), capture_output=True, text=True, timeout=timeout_s,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "timeout": True, "error": f"timed out after {timeout_s:.0f}s"}
    if proc.returncode != 0 or not proc.stdout:
        return {"ok": False, "error": "sandbox process failed", "traceback": proc.stderr[-2000:]}
    return json.loads(proc.stdout)


def gbm_dataset(seed: int, settings: BacktestSettings, n_days: int = 750, mu: float = 0.08,
                sigma: float = 0.2, perturb_after: Optional[int] = None) -> Dict[str, Any]:
    """A dataset spec with one GBM series per symbol, seeded deterministically."""
    return {
        "n_days": n_days,
        "perturb_after": perturb_after,
        "symbols": {
            sym: {"seed": seed + 97 * k, "mu": mu * (1 + 0.3 * k), "sigma": sigma, "start_price": 50.0 + 25 * k}
            for k, sym in enumerate(settings.symbols)
        },
    }
