"""Fail over between free-tier providers so a long run survives rate limits and quotas.

A *chain* is an ordered list of (provider, model id) entries, normally the same
open-weight model served by several providers, e.g. gpt-oss-120b on Groq, then
OpenRouter, then NVIDIA NIM. `ChainModel` sends each request to the first
provider that isn't cooling down:

- 429 rate limit: that provider cools down for as long as its `retry-after` /
  reset headers say (exponential backoff if it doesn't say), and the request
  moves to the next provider straight away.
- 5xx, timeouts, connection errors: a short cool-down, then the next provider.
- 401/403/404, 402 (out of credits), wrong model served: the provider is dropped
  for the rest of the run.
- 400/413/422: skipped for this request only (e.g. a provider that rejects the
  tool schema), others are tried.
- Everything cooling down: wait for the earliest reset, up to `max_wait_s`, then
  raise `QuotaExhausted` so the caller can stop cleanly and resume later.

Within a case the chain sticks to the provider that served the previous turn,
so a conversation only changes provider when it has to; every turn records the
provider that served it.

Chains live in `agent/chains.json`. Entries whose API-key variable isn't set are
skipped, so you only need keys for the providers you use. Check a chain against
the live providers with::

    python -m agent.providers check gpt-oss-120b
    python -m agent.providers models groq gpt-oss      # list a provider's model ids
"""

from __future__ import annotations

import email.utils
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .llm import EmptyResponseError, ModelMismatchError, ModelTurn, OpenAICompatModel

CHAINS_FILE = Path(__file__).with_name("chains.json")

# OpenAI-compatible endpoints. `key_env` is the environment variable holding the API key.
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY",
        "headers": {"X-Title": "backtesting-engine strategy agent"},
    },
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "key_env": "NVIDIA_API_KEY"},
    "huggingface": {"base_url": "https://router.huggingface.co/v1", "key_env": "HF_TOKEN"},
    "github": {"base_url": "https://models.github.ai/inference", "key_env": "GITHUB_TOKEN"},
    "mistral": {"base_url": "https://api.mistral.ai/v1", "key_env": "MISTRAL_API_KEY"},
    "google": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY"},
    "cerebras": {"base_url": "https://api.cerebras.ai/v1", "key_env": "CEREBRAS_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None},
}


class QuotaExhausted(RuntimeError):
    """Every provider in the chain is rate-limited for longer than we're willing to wait."""

    def __init__(self, chain: str, wait_s: float):
        super().__init__(f"all providers for {chain!r} are rate-limited; earliest reset in {wait_s:.0f}s")
        self.wait_s = wait_s


class AllProvidersFailed(RuntimeError):
    """No provider in the chain can serve this request (disabled, or rejected it)."""


@dataclass
class ProviderEntry:
    provider: str
    model: str
    base_url: str
    api_key: str
    headers: Dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def load_dotenv(path: Path = CHAINS_FILE.parent.parent / ".env") -> None:
    """Read KEY=value lines from the repo's `.env` (git-ignored) without overriding real env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_chain(name: str, path: Path = CHAINS_FILE, environ: Optional[Dict[str, str]] = None) -> List[ProviderEntry]:
    """Entries of chain `name` whose API key is available, in order."""
    if environ is None:
        load_dotenv()
    environ = os.environ if environ is None else environ
    chains = json.loads(Path(path).read_text(encoding="utf-8"))["chains"]
    if name not in chains:
        raise KeyError(f"unknown chain {name!r}; available: {', '.join(sorted(chains))}")
    entries = []
    for item in chains[name]["entries"]:
        spec = PROVIDERS[item["provider"]]
        key_env = spec.get("key_env")
        key = environ.get(key_env, "") if key_env else "not-needed"
        if key:
            entries.append(ProviderEntry(item["provider"], item["model"], item.get("base_url", spec["base_url"]),
                                         key, {**spec.get("headers", {}), **item.get("headers", {})}))
    return entries


# --- rate-limit headers ------------------------------------------------------------
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def parse_reset_seconds(headers: Any, now: Optional[float] = None) -> Optional[float]:
    """Seconds until a provider says we may retry, from common rate-limit headers."""
    if headers is None:
        return None
    get = lambda k: headers.get(k) if hasattr(headers, "get") else None  # noqa: E731
    now = time.time() if now is None else now
    ra = get("retry-after")
    if ra:  # authoritative when present
        try:
            return max(0.0, float(ra))
        except ValueError:
            try:
                return max(0.0, email.utils.parsedate_to_datetime(ra).timestamp() - now)
            except (TypeError, ValueError):
                pass
    candidates = []
    for kind in ("requests", "tokens"):
        val = get(f"x-ratelimit-reset-{kind}")
        remaining = get(f"x-ratelimit-remaining-{kind}")
        # Only the limit that actually ran out matters: Groq reports the daily request window's
        # reset on every response, even when it was the per-minute token budget that hit zero.
        exhausted = remaining is None or str(remaining).strip() in ("0", "0.0")
        if val and exhausted:  # Groq style: "2m59.56s", "7.66s", "120ms"
            parts = _DURATION.findall(str(val))
            if parts:
                scale = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
                candidates.append(sum(float(n) * scale[u] for n, u in parts))
    reset = get("x-ratelimit-reset")
    if reset:  # OpenRouter style: epoch milliseconds (or seconds)
        try:
            value = float(reset)
            epoch = value / 1000 if value > 1e11 else value
            candidates.append(epoch - now if epoch > 1e9 else value)
        except ValueError:
            pass
    positive = [c for c in candidates if c > 0]
    return max(positive) if positive else None


def classify(exc: BaseException, consecutive_failures: int):
    """Decide what a failure means: ("cooldown", seconds), ("disable", why) or ("skip", why)."""
    import openai

    backoff = lambda base, cap: min(cap, base * 2 ** consecutive_failures)  # noqa: E731
    if isinstance(exc, ModelMismatchError):
        return "disable", str(exc)
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)):
        return "disable", f"{type(exc).__name__}: {exc}"
    if isinstance(exc, openai.RateLimitError):
        wait = parse_reset_seconds(getattr(exc, "response", None) and exc.response.headers)
        return "cooldown", min(wait, 6 * 3600) if wait else backoff(30, 3600)
    if isinstance(exc, openai.APIStatusError) and exc.status_code == 402:
        return "disable", "out of credits (402)"
    if isinstance(exc, openai.APIStatusError) and exc.status_code == 410:
        return "disable", f"model retired by this provider (410): {exc}"
    if isinstance(exc, (openai.InternalServerError, openai.APIConnectionError, openai.APITimeoutError,
                        EmptyResponseError)):
        return "cooldown", backoff(15, 600)
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return "cooldown", backoff(15, 600)
    if isinstance(exc, openai.APIStatusError):
        # Any other 4xx (400 schema rejected, 413 too large, 422, ...): this provider can't serve
        # this request, but others might, so never let a provider-side error end the case here.
        return "skip", f"{exc.status_code}: {exc}"
    return None


class ProviderPool:
    """Provider state shared by every case in a run: cool-downs, disabled providers, stats."""

    def __init__(self, name: str, entries: List[ProviderEntry], make_client: Optional[Callable] = None,
                 clock: Callable[[], float] = time.monotonic, log: Callable[[str], None] = print):
        if not entries:
            raise ValueError(f"chain {name!r} has no usable providers: set at least one of its API-key variables")
        self.name = name
        self.entries = entries
        make_client = make_client or (lambda e: OpenAICompatModel(
            e.model, base_url=e.base_url, api_key=e.api_key, max_attempts=1, request_timeout_s=180.0,
            default_headers=e.headers or None))
        self.clients = [make_client(e) for e in entries]
        self.clock = clock
        self.log = log
        self.lock = threading.Lock()
        self.cool_until = [0.0] * len(entries)
        self.failures = [0] * len(entries)
        self.disabled: Dict[int, str] = {}
        self.stats = {e.label: {"ok": 0, "rate_limited": 0, "errors": 0} for e in entries}

    def available(self, i: int, skip) -> bool:
        return i not in self.disabled and i not in skip and self.clock() >= self.cool_until[i]

    def next_ready_in(self, skip) -> Optional[float]:
        with self.lock:
            waits = [self.cool_until[i] - self.clock() for i in range(len(self.entries))
                     if i not in self.disabled and i not in skip]
        return max(0.0, min(waits)) if waits else None

    def record_ok(self, i: int) -> None:
        with self.lock:
            self.failures[i] = 0
            self.stats[self.entries[i].label]["ok"] += 1

    def record_failure(self, i: int, exc: BaseException) -> Optional[str]:
        """Apply the failure to provider i; returns "skip" for request-local failures, None otherwise."""
        label = self.entries[i].label
        with self.lock:
            decision = classify(exc, self.failures[i])
            if decision is None:
                raise exc
            kind, value = decision
            self.failures[i] += 1
            if kind == "cooldown":
                self.cool_until[i] = max(self.cool_until[i], self.clock() + value)
                is_rate = "RateLimit" in type(exc).__name__
                self.stats[label]["rate_limited" if is_rate else "errors"] += 1
                self.log(f"  [{self.name}] {label} {'rate-limited' if is_rate else 'failed'}; cooling down {value:.0f}s")
                return None
            self.stats[label]["errors"] += 1
            if kind == "disable":
                self.disabled[i] = value
                self.log(f"  [{self.name}] {label} disabled for this run: {value}")
                return None
            return "skip"


class ChainModel:
    """A `ChatModel` that serves each request from the first available provider in a pool."""

    def __init__(self, pool: ProviderPool, max_wait_s: float = 1800.0, sleep: Callable[[float], None] = time.sleep):
        self.pool = pool
        self.name = pool.name
        self.max_wait_s = max_wait_s
        self.sleep = sleep
        self.sticky: Optional[int] = None
        self.switches = 0

    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> ModelTurn:
        pool = self.pool
        skip: set = set()
        last_error: Optional[BaseException] = None
        waited = 0.0
        while True:
            order = ([self.sticky] if self.sticky is not None else []) + \
                    [i for i in range(len(pool.entries)) if i != self.sticky]
            choice = next((i for i in order if pool.available(i, skip)), None)
            if choice is None:
                wait = pool.next_ready_in(skip)
                if wait is None:
                    raise AllProvidersFailed(f"no provider could serve the request; last error: {last_error}")
                if waited + wait > self.max_wait_s:
                    raise QuotaExhausted(pool.name, wait)
                pool.log(f"  [{pool.name}] all providers busy; waiting {wait:.0f}s for the earliest reset")
                self.sleep(wait + 0.5)
                waited += wait + 0.5
                continue
            try:
                turn = pool.clients[choice].chat(messages, tools)
            except Exception as exc:  # classified below; unknown errors are re-raised
                last_error = exc
                if pool.record_failure(choice, exc) == "skip":
                    skip.add(choice)
                continue
            pool.record_ok(choice)
            if self.sticky is not None and choice != self.sticky:
                self.switches += 1
            self.sticky = choice
            turn.provider = pool.entries[choice].label
            return turn


# --- CLI: check chains against the live providers ------------------------------------
def _cmd_models(provider: str, needle: str = "") -> int:
    from openai import OpenAI

    load_dotenv()
    spec = PROVIDERS[provider]
    key = os.environ.get(spec["key_env"], "") if spec.get("key_env") else "not-needed"
    if not key:
        print(f"{spec['key_env']} is not set")
        return 1
    client = OpenAI(base_url=spec["base_url"], api_key=key, default_headers=spec.get("headers"))
    ids = sorted(m.id for m in client.models.list())
    for mid in ids:
        if needle.lower() in mid.lower():
            print(mid)
    return 0


def _cmd_check(names: List[str]) -> int:
    """For each entry with a key: is the model listed, and does a tiny tool call work?"""
    from .agent import TOOLS

    chains = json.loads(CHAINS_FILE.read_text(encoding="utf-8"))["chains"]
    names = names or sorted(chains)
    status = 0
    for name in names:
        print(f"\n{name}: {chains[name].get('description', '')}")
        available = {e.label for e in load_chain(name)}
        for item in chains[name]["entries"]:
            label = f"{item['provider']}:{item['model']}"
            if label not in available:
                print(f"  - {label:60s} skipped ({PROVIDERS[item['provider']]['key_env']} not set)")
                continue
            entry = next(e for e in load_chain(name) if e.label == label)
            client = OpenAICompatModel(entry.model, base_url=entry.base_url, api_key=entry.api_key, max_attempts=1,
                                       request_timeout_s=60.0, max_tokens=256, default_headers=entry.headers or None)
            try:
                turn = client.chat([{"role": "user", "content": "Call the decline tool with reason 'ping'."}], TOOLS)
                kind = "native tool call" if turn.tool_calls else "text only (needs text-call recovery)"
                print(f"  ok {label:60s} served as {turn.model}; {kind}")
            except Exception as exc:
                status = 1
                print(f"  !! {label:60s} {type(exc).__name__}: {str(exc)[:160]}")
    return status


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["models"] and len(argv) >= 2:
        return _cmd_models(argv[1], argv[2] if len(argv) > 2 else "")
    if argv[:1] == ["check"]:
        return _cmd_check(argv[1:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
