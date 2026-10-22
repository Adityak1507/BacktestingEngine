"""Check that the eval harness can still tell right from wrong (used in CI).

Runs every case through the real runner and grader with two scripted models:
the oracle (submits each reference, declines when it should) must score 100%,
and the null model (submits a strategy that never trades) must score 0%.
No LLM is called and nothing is written outside a temporary directory.

    python -m evals.strategy_agent.selfcheck
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from .run_eval import main as run_eval

EXPECTED = {"oracle": 1.0, "null": 0.0}


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for fake, expected in EXPECTED.items():
            out = Path(tmp) / fake
            code = run_eval(["--fake", fake, "--out-dir", str(out), "--concurrency", "4"])
            summary = json.loads((out / "baseline" / "summary.json").read_text(encoding="utf-8"))
            rate, errors = summary["pass_rate"], summary["errors"]
            ok = code == 0 and errors == 0 and rate == expected
            print(f"{fake}: pass rate {rate:.1%}, {errors} errors (expected {expected:.0%}) -> {'ok' if ok else 'FAIL'}")
            if not ok:
                failures.append(fake)
    if failures:
        print(f"Eval self-check failed for: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("Eval self-check passed: the grader separates correct from incorrect strategies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
