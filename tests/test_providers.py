"""Tests for provider fail-over chains (no network: fake clients and a fake clock)."""

import json

import httpx2
import openai
import pytest

from agent.llm import ModelMismatchError, ModelTurn
from agent.providers import (
    AllProvidersFailed,
    ChainModel,
    ProviderEntry,
    ProviderPool,
    QuotaExhausted,
    load_chain,
    parse_reset_seconds,
)


def status_error(cls, code, headers=None):
    resp = httpx2.Response(code, headers=headers or {}, request=httpx2.Request("POST", "https://x.test/v1"))
    return cls("boom", response=resp, body=None)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeClient:
    """Raises the queued exceptions in order, then answers."""

    def __init__(self, name, errors=()):
        self.name, self.errors, self.calls = name, list(errors), 0

    def chat(self, messages, tools):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return ModelTurn(f"hi from {self.name}", [], self.name, "stop", {"input_tokens": 1, "output_tokens": 1})


def make_pool(*clients, clock=None):
    entries = [ProviderEntry(c.name, "m", "https://x.test", "k") for c in clients]
    by_label = {e.label: c for e, c in zip(entries, clients)}
    return ProviderPool("test", entries, make_client=lambda e: by_label[e.label], clock=clock or Clock(), log=lambda m: None)


def test_rate_limit_fails_over_to_next_provider_and_cools_down():
    clock = Clock()
    a = FakeClient("a", [status_error(openai.RateLimitError, 429, {"retry-after": "20"})])
    b = FakeClient("b")
    pool = make_pool(a, b, clock=clock)
    model = ChainModel(pool, sleep=clock.sleep)
    assert model.chat([], []).provider == "b:m"
    assert pool.cool_until[0] == pytest.approx(1020.0)
    # Sticky: the next turn stays on b even after a's cool-down ends.
    clock.t += 60
    assert model.chat([], []).provider == "b:m" and a.calls == 1


def test_waits_for_earliest_reset_when_every_provider_is_limited():
    clock = Clock()
    a = FakeClient("a", [status_error(openai.RateLimitError, 429, {"retry-after": "30"})])
    b = FakeClient("b", [status_error(openai.RateLimitError, 429, {"retry-after": "90"})])
    model = ChainModel(make_pool(a, b, clock=clock), max_wait_s=600, sleep=clock.sleep)
    turn = model.chat([], [])
    assert turn.provider == "a:m" and clock.t == pytest.approx(1030.5)


def test_quota_exhausted_when_reset_is_beyond_max_wait():
    clock = Clock()
    a = FakeClient("a", [status_error(openai.RateLimitError, 429, {"retry-after": "7200"})])
    model = ChainModel(make_pool(a, clock=clock), max_wait_s=600, sleep=clock.sleep)
    with pytest.raises(QuotaExhausted):
        model.chat([], [])


def test_bad_key_or_no_credits_disables_provider_for_the_run():
    a = FakeClient("a", [status_error(openai.AuthenticationError, 401)])
    b = FakeClient("b", [status_error(openai.APIStatusError, 402)])
    c = FakeClient("c")
    pool = make_pool(a, b, c)
    assert ChainModel(pool).chat([], []).provider == "c:m"
    assert set(pool.disabled) == {0, 1}
    # A fresh case never retries the disabled providers.
    assert ChainModel(pool).chat([], []).provider == "c:m" and a.calls == 1 and b.calls == 1


def test_wrong_model_served_disables_provider():
    pool = make_pool(FakeClient("a", [ModelMismatchError("x")]), FakeClient("b"))
    assert ChainModel(pool).chat([], []).provider == "b:m" and 0 in pool.disabled


def test_request_rejected_everywhere_raises():
    a = FakeClient("a", [status_error(openai.BadRequestError, 400)])
    b = FakeClient("b", [status_error(openai.BadRequestError, 400)])
    pool = make_pool(a, b)
    with pytest.raises(AllProvidersFailed):
        ChainModel(pool).chat([], [])
    assert not pool.disabled  # a 400 is request-local, not a reason to drop the provider


def test_server_errors_back_off_exponentially():
    clock = Clock()
    errs = [status_error(openai.InternalServerError, 503) for _ in range(3)]
    a = FakeClient("a", errs)
    model = ChainModel(make_pool(a, clock=clock), max_wait_s=10_000, sleep=clock.sleep)
    model.chat([], [])
    # Cool-downs of 15s, 30s and 60s before the fourth attempt succeeds.
    assert clock.t == pytest.approx(1000 + 15.5 + 30.5 + 60.5)


def test_parse_reset_headers():
    assert parse_reset_seconds({"retry-after": "12"}) == 12
    assert parse_reset_seconds({"x-ratelimit-reset-tokens": "2m59.5s"}) == pytest.approx(179.5)
    assert parse_reset_seconds({"x-ratelimit-reset-requests": "120ms"}) == pytest.approx(0.12)
    assert parse_reset_seconds({"x-ratelimit-reset": str(1_800_000_030 * 1000)}, now=1_800_000_000) == pytest.approx(30)
    assert parse_reset_seconds({}) is None


def test_load_chain_skips_entries_without_keys(tmp_path):
    path = tmp_path / "chains.json"
    path.write_text(json.dumps({"chains": {"c": {"entries": [
        {"provider": "groq", "model": "g"}, {"provider": "openrouter", "model": "o:free"},
        {"provider": "ollama", "model": "local"}]}}}))
    entries = load_chain("c", path, environ={"OPENROUTER_API_KEY": "k"})
    assert [e.label for e in entries] == ["openrouter:o:free", "ollama:local"]
    assert entries[0].headers  # OpenRouter attribution header is attached
    with pytest.raises(KeyError):
        load_chain("missing", path, environ={})


def test_shipped_chains_file_is_valid():
    from agent.providers import CHAINS_FILE, PROVIDERS

    chains = json.loads(CHAINS_FILE.read_text())["chains"]
    assert "gpt-oss-120b" in chains
    for chain in chains.values():
        assert chain["entries"] and all(e["provider"] in PROVIDERS for e in chain["entries"])


def test_agent_records_provider_per_turn():
    from agent import StrategyAgent
    from agent.llm import ToolCall

    class Answer(FakeClient):
        def chat(self, messages, tools):
            turn = super().chat(messages, tools)
            turn.tool_calls = [ToolCall("1", "decline", {"reason": "needs earnings"}, '{"reason": "needs earnings"}')]
            return turn

    a = FakeClient("a", [status_error(openai.RateLimitError, 429, {"retry-after": "5"})])
    pool = make_pool(a, Answer("b"))
    res = StrategyAgent(ChainModel(pool)).run("earnings strategy")
    assert res.outcome == "declined" and res.providers == ["b:m"]


def test_retired_model_and_unknown_4xx_never_end_the_case():
    clock = Clock()
    a = FakeClient("a", [status_error(openai.RateLimitError, 429, {"retry-after": "30"})])
    gone = FakeClient("gone", [status_error(openai.APIStatusError, 410)])
    teapot = FakeClient("teapot", [status_error(openai.APIStatusError, 418)])
    pool = make_pool(a, gone, teapot, clock=clock)
    # a is rate-limited, gone is retired, teapot rejects this request: wait for a's reset.
    turn = ChainModel(pool, max_wait_s=600, sleep=clock.sleep).chat([], [])
    assert turn.provider == "a:m" and 1 in pool.disabled and 2 not in pool.disabled


def test_reset_uses_only_the_exhausted_limit_and_prefers_retry_after():
    groq_tpm = {"x-ratelimit-remaining-requests": "990", "x-ratelimit-reset-requests": "7m12s",
                "x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "6.5s"}
    assert parse_reset_seconds(groq_tpm) == pytest.approx(6.5)
    assert parse_reset_seconds({**groq_tpm, "retry-after": "3"}) == 3


def test_server_rejected_tool_call_comes_back_as_a_model_reply(monkeypatch):
    from agent.llm import OpenAICompatModel

    model = OpenAICompatModel("m", base_url="https://x.test/v1", api_key="k", max_attempts=1)
    body = {"error": {"code": "tool_use_failed", "failed_generation": "<|channel|>commentary to=decline"}}
    resp = httpx2.Response(400, request=httpx2.Request("POST", "https://x.test/v1"))

    def reject(**kwargs):
        raise openai.BadRequestError("bad tool", response=resp, body=body)

    monkeypatch.setattr(model._client.chat.completions, "create", reject)
    turn = model.chat([], [])
    assert turn.tool_calls == [] and turn.finish_reason == "tool_use_failed" and "commentary" in turn.content
