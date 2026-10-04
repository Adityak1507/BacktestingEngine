"""Programmatic grading of the agent's output.

For a case with a reference implementation, the submitted code is run in the
sandbox on hidden datasets (different seeds from the agent's test data) and
scored on:

- valid:        passes the static checks, defines a Strategy, constructs with no args
- runs:         backtests on every hidden dataset without error or timeout
- no_lookahead: orders submitted up to bar CUT are identical when every bar after
                CUT is replaced with an unrelated price path
- agreement:    worst case over datasets of the fraction of bars, among those
                where either side holds a position, on which the agent's exposure
                (position value / equity, per symbol) is within EXPOSURE_TOL of
                the reference's
- pass:         all of the above, with agreement >= PASS_AGREEMENT

Loose prompts list several acceptable readings; the agent is compared with the
closest one, by position direction only (sizing isn't specified), and passes at
LOOSE_PASS_AGREEMENT.

For a case that should be declined, pass = the agent declined. If it wrote code
instead, valid/runs/no_lookahead are still measured on that code and
agreement is 0.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Dict, List, Optional, Tuple

from agent.sandbox import BacktestSettings, gbm_dataset, run_job

# (seed, drift, volatility): an up-trend, a falling and choppy market, and a quiet one.
GRADING_DATA = [(101, 0.10, 0.20), (202, -0.10, 0.30), (303, 0.05, 0.15)]
LOOKAHEAD_CUT = 400
EXPOSURE_TOL = 0.10
PASS_AGREEMENT = 0.995  # correct implementations score 1.000; known near-misses score 0.98-0.99
LOOSE_PASS_AGREEMENT = 0.95  # loose prompts: direction match against the closest acceptable reading
GRADE_TIMEOUT_S = 120.0

METRICS = ["pass", "valid", "runs", "no_lookahead", "agreement"]


def settings_for(case: Dict[str, Any]) -> BacktestSettings:
    return BacktestSettings(**case.get("settings", {}))


def datasets_for(settings: BacktestSettings) -> List[Dict[str, Any]]:
    """The hidden datasets, plus a copy of the first one perturbed after LOOKAHEAD_CUT."""
    sets = [gbm_dataset(seed, settings, mu=mu, sigma=sigma) for seed, mu, sigma in GRADING_DATA]
    seed, mu, sigma = GRADING_DATA[0]
    sets.append(gbm_dataset(seed, settings, mu=mu, sigma=sigma, perturb_after=LOOKAHEAD_CUT))
    return sets


def references(case: Dict[str, Any]) -> List[str]:
    """Acceptable implementations: one for precise prompts, several for loose ones."""
    return case.get("references") or ([case["reference"]] if "reference" in case else [])


def is_loose(case: Dict[str, Any]) -> bool:
    return "references" in case


@functools.lru_cache(maxsize=None)
def _reference_runs(case_id: str, index: int) -> Tuple[Dict[str, Any], ...]:
    from .cases import CASES_BY_ID

    case = CASES_BY_ID[case_id]
    settings = settings_for(case)
    out = run_job(references(case)[index], settings, datasets_for(settings), GRADE_TIMEOUT_S)
    if not out.get("ok"):
        raise RuntimeError(f"reference {index} for {case_id} failed: {out.get('error')}\n{out.get('traceback', '')}")
    return tuple(out["runs"])


def agreement(agent_run: Dict[str, Any], ref_run: Dict[str, Any], direction_only: bool = False) -> float:
    """Fraction of active (bar, symbol) points where the agent matches the reference.

    A point is active when either side holds a position. Counting the bars where
    both sit flat would let a strategy that is wrong whenever it trades still
    score highly, because most strategies are flat most of the time. Points match
    when exposures are within EXPOSURE_TOL or, with `direction_only`, when both
    are long, both flat or both short.
    """
    hits = total = 0
    for sym, ref_exp in ref_run["exposure"].items():
        for a, r in zip(agent_run["exposure"].get(sym, [0.0] * len(ref_exp)), ref_exp):
            if abs(a) <= 1e-9 and abs(r) <= 1e-9:
                continue
            total += 1
            if direction_only:
                hits += _side(a) == _side(r)
            else:
                hits += abs(a - r) <= EXPOSURE_TOL
    return hits / total if total else 1.0


def _side(x: float) -> int:
    return 0 if abs(x) <= 1e-9 else (1 if x > 0 else -1)


def lookahead_violations(base_run: Dict[str, Any], perturbed_run: Dict[str, Any], cut: int = LOOKAHEAD_CUT) -> List[str]:
    """Orders that differ, up to bar `cut`, between the original and the perturbed data."""
    cut_date = base_run["dates"][cut]
    before = lambda run: [o for o in run["orders"] if o["t"] <= cut_date]  # noqa: E731
    a, b = before(base_run), before(perturbed_run)
    problems = []
    for x, y in zip(a, b):
        if x["t"] != y["t"] or x["symbol"] != y["symbol"] or not _close(x["qty"], y["qty"]) \
                or not _close(x["limit"], y["limit"]) or not _close(x["stop"], y["stop"]):
            problems.append(f"order on {x['t']} differs: {x} vs {y}")
            break
    if len(a) != len(b):
        problems.append(f"{len(a)} orders up to {cut_date} on the original data vs {len(b)} on the perturbed data")
    return problems


def _close(x: Optional[float], y: Optional[float]) -> bool:
    if x is None or y is None:
        return x is y
    return math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9)


def grade(case: Dict[str, Any], outcome: str, code: Optional[str], reason: Optional[str] = None) -> Dict[str, Any]:
    """Return ``{"grade": {metric: score}, "explanation": {metric: str}, "detail": {...}}``.

    A decline is graded on the outcome only; its `reason` is recorded for human
    spot-checks (a model can decline for the wrong reason, e.g. after its own code crashed).
    """
    expects_decline = case.get("expect") == "decline"
    g = {m: 0.0 for m in METRICS}
    why: Dict[str, str] = {}
    detail: Dict[str, Any] = {"outcome": outcome}

    if outcome == "declined":
        detail["decline_reason"] = reason
        if expects_decline:
            g = {m: 1.0 for m in METRICS}
            why["pass"] = f"correctly declined - check the reason: {reason!r}"
        else:
            why["pass"] = f"declined a request that can be implemented: {reason!r}"
        return {"grade": g, "explanation": why, "detail": detail}
    if outcome != "submitted" or not code:
        why["pass"] = "no strategy submitted" + (" (expected a decline)" if expects_decline else "")
        return {"grade": g, "explanation": why, "detail": detail}

    settings = settings_for(case)
    out = run_job(code, settings, datasets_for(settings), GRADE_TIMEOUT_S)
    detail["sandbox"] = {k: v for k, v in out.items() if k != "runs"}
    if not out.get("ok"):
        stage = out.get("stage")
        g["valid"] = 0.0 if stage in ("static", "load") else 1.0
        why["valid"] = "ok" if g["valid"] else f"{out.get('error')} {'; '.join(out.get('problems', []))}".strip()
        why["runs"] = "timed out" if out.get("timeout") else f"{out.get('error')}"
        why["pass"] = why["runs"] if g["valid"] else why["valid"]
        return {"grade": g, "explanation": why, "detail": detail}

    runs = out["runs"]
    g["valid"] = g["runs"] = 1.0
    violations = lookahead_violations(runs[0], runs[-1])
    g["no_lookahead"] = 0.0 if violations else 1.0
    why["no_lookahead"] = "; ".join(violations) or "ok"
    detail["fills_per_dataset"] = [r["num_fills"] for r in runs[:-1]]

    if expects_decline:
        why["pass"] = "wrote code for a request that should have been declined"
        why["agreement"] = "no reference: request should be declined"
        return {"grade": g, "explanation": why, "detail": detail}

    loose = is_loose(case)
    best, best_scores, best_index = -1.0, [], 0
    for k in range(len(references(case))):
        refs = _reference_runs(case["id"], k)
        scores = [agreement(a, r, direction_only=loose) for a, r in zip(runs[:-1], refs[:-1])]
        if min(scores) > best:
            best, best_scores, best_index = min(scores), scores, k
    threshold = LOOSE_PASS_AGREEMENT if loose else PASS_AGREEMENT
    detail["agreement_per_dataset"] = [round(x, 4) for x in best_scores]
    g["agreement"] = best
    why["agreement"] = "per dataset: " + ", ".join(f"{x:.1%}" for x in best_scores)
    if loose:
        reading = case.get("interpretations", [])[best_index] if case.get("interpretations") else f"#{best_index}"
        detail["closest_interpretation"] = reading
        why["agreement"] += f" (closest reading: {reading}; direction only)"
    g["pass"] = float(g["no_lookahead"] == 1.0 and g["agreement"] >= threshold)
    if not g["pass"]:
        why["pass"] = ("look-ahead detected" if not g["no_lookahead"]
                       else f"behaviour differs from the spec (agreement {g['agreement']:.1%} < {threshold:.1%})")
    else:
        why["pass"] = "ok"
    return {"grade": g, "explanation": why, "detail": detail}
