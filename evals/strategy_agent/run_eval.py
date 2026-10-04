"""Run the strategy agent over the eval cases and grade every submission.

Examples (from the repo root):

    # Free end-to-end checks of the harness itself
    python -m evals.strategy_agent.run_eval --fake oracle --variant oracle-check
    python -m evals.strategy_agent.run_eval --fake null --variant null-check

    # Local open-weight model via Ollama
    python -m evals.strategy_agent.run_eval --model qwen2.5-coder:7b

    # Hosted open-weight model on any OpenAI-compatible provider
    python -m evals.strategy_agent.run_eval --model llama-3.3-70b-versatile \\
        --base-url https://api.groq.com/openai/v1 --api-key-env GROQ_API_KEY --variant v1

Layout written under --out-dir (default evals/strategy_agent/runs/<model>/):
    _state.json                 metric definitions for the report
    <variant>/results.jsonl     one graded row per (case, rep), written as each finishes
    <variant>/traces/<id>_rep<k>.json   the full agent transcript
    <variant>/errors.jsonl      attempts that failed before producing gradable output

Re-running resumes: (case, rep) pairs already in results.jsonl are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent import StrategyAgent
from agent.llm import ModelMismatchError, OpenAICompatModel

from .cases import CASES, CASES_BY_ID
from .fake_models import FAKES
from .grading import METRICS, PASS_AGREEMENT, settings_for, grade

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
HARNESS_PATHS = ["agent/agent.py", "agent/llm.py", "agent/sandbox.py", "evals/strategy_agent/cases.py",
                 "evals/strategy_agent/grading.py", "evals/strategy_agent/run_eval.py"]

STATE = {
    "flow": "strategy-agent",
    "metrics": [
        {"id": "pass", "label": "Pass", "kind": "binary"},
        {"id": "valid", "label": "Valid code", "kind": "binary"},
        {"id": "runs", "label": "Runs", "kind": "binary"},
        {"id": "no_lookahead", "label": "No lookahead", "kind": "binary"},
        {"id": "agreement", "label": "Agreement", "kind": "float", "scale": 1},
    ],
    "perf_fields": [
        {"id": "latency_s", "label": "Latency", "unit": "s"},
        {"id": "turns", "label": "Model turns"},
        {"id": "tool_calls", "label": "Tool calls"},
        {"id": "dev_runs", "label": "Test runs"},
        {"id": "text_tool_calls", "label": "Text tool calls"},
        {"id": "in_tokens", "label": "In tokens"},
        {"id": "out_tokens", "label": "Out tokens"},
    ],
    "harness_paths": HARNESS_PATHS,
}


def harness_sha() -> str:
    h = hashlib.sha256()
    for rel in HARNESS_PATHS:
        h.update(rel.encode())
        h.update((ROOT / rel).read_bytes())
    return h.hexdigest()


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "model"


class Writer:
    """Thread-safe append-only writers for the results and errors files."""

    def __init__(self, variant_dir: Path):
        self.dir = variant_dir
        (variant_dir / "traces").mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def done_keys(self) -> set:
        path = self.dir / "results.jsonl"
        if not path.exists():
            return set()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {(r["prompt_id"], r["rep"]) for r in rows}

    def append(self, name: str, row: Dict[str, Any]) -> None:
        with self.lock, open(self.dir / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def trace(self, case_id: str, rep: int, trace: List[Dict[str, Any]]) -> str:
        rel = f"traces/{case_id}_rep{rep}.json"
        (self.dir / rel).write_text(json.dumps(trace, indent=1), encoding="utf-8")
        return rel


def run_case(case: Dict[str, Any], rep: int, make_model, writer: Writer, timeout_s: float) -> str:
    """Run one (case, rep) with a hard wall-clock ceiling; returns a one-line status."""
    box: Dict[str, Any] = {}

    def work():
        try:
            agent = StrategyAgent(make_model())
            box["result"] = agent.run(case["prompt"], settings_for(case))
        except BaseException as exc:  # recorded below; must not kill the worker thread silently
            box["error"] = exc

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout_s)
    base = {"prompt_id": case["id"], "rep": rep}
    if t.is_alive():
        # The model call may still be running in the background; the slot is reclaimed anyway.
        writer.append("errors.jsonl", {**base, "class": "timeout", "error": f"exceeded {timeout_s:.0f}s"})
        return f"{case['id']}#{rep}: TIMEOUT"
    if "error" in box:
        exc = box["error"]
        cls = "model_mismatch" if isinstance(exc, ModelMismatchError) else "api_error"
        writer.append("errors.jsonl", {**base, "class": cls, "error": f"{type(exc).__name__}: {exc}"[:2000]})
        return f"{case['id']}#{rep}: ERROR {type(exc).__name__}"

    res = box["result"]
    try:
        graded = grade(case, res.outcome, res.code, res.reason)
    except Exception as exc:
        writer.append("errors.jsonl", {**base, "class": "grader_error", "error": f"{type(exc).__name__}: {exc}"[:2000],
                                       "model": res.model, "usage": res.usage})
        return f"{case['id']}#{rep}: GRADER ERROR {exc}"

    if res.outcome == "declined":
        res.trace.append({"role": "assistant", "content": f"[declined] {res.reason}"})
    trace_ref = writer.trace(case["id"], rep, res.trace)
    truncated = res.stop_reason == "length" and res.outcome == "gave_up"
    writer.append("results.jsonl", {
        **base,
        "prompt": case["prompt"],
        "tags": case["tags"],
        "status": "truncated" if truncated else "ok",
        "stop_reason": res.stop_reason,
        "outcome": res.outcome,
        "grade": graded["grade"],
        "explanation": graded["explanation"],
        "model": res.model,
        "usage": res.usage,
        "in_tokens": res.usage.get("input_tokens", 0),
        "out_tokens": res.usage.get("output_tokens", 0),
        "latency_s": round(res.latency_s, 2),
        "turns": res.turns,
        "tool_calls": res.tool_calls,
        "dev_runs": res.dev_runs,
        "text_tool_calls": res.text_tool_calls,
        "retries": res.retries,
        "trace": trace_ref,
        "meta": {"detail": graded["detail"], "code": res.code},
    })
    g = graded["grade"]
    return f"{case['id']}#{rep}: {'PASS' if g['pass'] else 'fail'} ({res.outcome}, agreement {g['agreement']:.0%}, {res.latency_s:.0f}s)"


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(variant_dir: Path) -> Dict[str, Any]:
    path = variant_dir / "results.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []
    ok = [r for r in rows if r["status"] == "ok"]
    errors_path = variant_dir / "errors.jsonl"
    errors = len(errors_path.read_text(encoding="utf-8").splitlines()) if errors_path.exists() else 0
    n = len(ok)
    k = sum(int(r["grade"]["pass"]) for r in ok)
    lo, hi = wilson(k, n)
    by_tag: Dict[str, List[float]] = defaultdict(list)
    for r in ok:
        by_tag[r["tags"][0]].append(r["grade"]["pass"])
    summary = {
        "cases_scored": n,
        "errors": errors,
        "truncated": len(rows) - n,
        "pass_rate": k / n if n else None,
        "pass_ci95": [lo, hi],
        "metric_means": {m: (sum(r["grade"][m] for r in ok) / n if n else None) for m in METRICS},
        "pass_by_category": {t: sum(v) / len(v) for t, v in sorted(by_tag.items())},
        "mean_latency_s": sum(r["latency_s"] for r in ok) / n if n else None,
        "tokens": {"input": sum(r["in_tokens"] for r in ok), "output": sum(r["out_tokens"] for r in ok)},
        "models": sorted({r["model"] for r in rows}),
    }
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="model id as the server knows it, e.g. qwen2.5-coder:7b")
    p.add_argument("--base-url", default="http://localhost:11434/v1", help="OpenAI-compatible endpoint (default: local Ollama)")
    p.add_argument("--api-key-env", help="name of the environment variable holding the API key (not the key itself)")
    p.add_argument("--fake", choices=sorted(FAKES), help="use a scripted model to check the harness for free")
    p.add_argument("--variant", default="baseline", help="output sub-directory: baseline, v1, v2, ...")
    p.add_argument("--out-dir", help="flow directory (default: evals/strategy_agent/runs/<model>)")
    p.add_argument("--cases", help="comma-separated case ids (default: all)")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=1, help="cases in flight; keep 1 for a single local GPU")
    p.add_argument("--timeout-s", type=float, default=900.0, help="hard wall-clock ceiling per case")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--approve-harness", action="store_true",
                   help="record the current harness files as approved (run once after reviewing a change)")
    args = p.parse_args(argv)

    if not args.fake and not args.model:
        p.error("--model is required unless --fake is given")
    model_name = f"fake-{args.fake}" if args.fake else args.model
    flow = Path(args.out_dir) if args.out_dir else HERE / "runs" / slug(model_name)
    flow.mkdir(parents=True, exist_ok=True)
    (flow / "_state.json").write_text(json.dumps(STATE, indent=2), encoding="utf-8")

    # Harness-integrity gate: refuse to run if the agent or grader changed since the last approval,
    # so scores from different runs are never silently produced by different harness code.
    sha_file = HERE / "runs" / ".approved_harness"
    current = harness_sha()
    if args.approve_harness:
        sha_file.parent.mkdir(parents=True, exist_ok=True)
        sha_file.write_text(current)
        print(f"Harness approved ({current[:12]}).")
    elif not args.fake and (not sha_file.exists() or sha_file.read_text().strip() != current):
        print("The agent/grader code differs from the last approved harness. Review the changes, then re-run "
              "with --approve-harness.", file=sys.stderr)
        return 2

    if args.fake:
        make_model = FAKES[args.fake]
    else:
        key = os.environ.get(args.api_key_env, "") if args.api_key_env else "not-needed"
        if args.api_key_env and not key:
            p.error(f"environment variable {args.api_key_env} is not set")
        make_model = lambda: OpenAICompatModel(args.model, base_url=args.base_url, api_key=key,  # noqa: E731
                                               temperature=args.temperature)

    cases = [CASES_BY_ID[c] for c in args.cases.split(",")] if args.cases else CASES
    writer = Writer(flow / args.variant)
    done = writer.done_keys()
    todo = [(c, r) for r in range(args.reps) for c in cases if (c["id"], r) not in done]
    print(f"{model_name}: {len(todo)} case runs to do ({len(done)} already done) -> {writer.dir}")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(run_case, c, r, make_model, writer, args.timeout_s) for c, r in todo]
        for i, f in enumerate(futures, 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)

    s = summarise(writer.dir)
    if s["pass_rate"] is not None:
        lo, hi = s["pass_ci95"]
        print(f"\npass rate {s['pass_rate']:.1%} (95% CI {lo:.0%}-{hi:.0%}) on {s['cases_scored']} case runs; "
              f"{s['errors']} errors, {s['truncated']} truncated; wall-clock {time.perf_counter() - started:.0f}s")
        print("  " + "  ".join(f"{m}={v:.2f}" for m, v in s["metric_means"].items()))
        print("  by category: " + ", ".join(f"{t} {v:.0%}" for t, v in s["pass_by_category"].items()))
        print(f"  (pass = runs, no look-ahead and >= {PASS_AGREEMENT:.1%} agreement with the reference; or a correct decline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
