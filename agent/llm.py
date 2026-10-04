"""A minimal chat-model interface, plus an implementation for OpenAI-compatible servers.

Ollama, vLLM, LM Studio, llama.cpp's server, Groq, Together and OpenRouter all
expose the same `/v1/chat/completions` endpoint with function calling, so one
client covers local and hosted open-weight models.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]
    raw_arguments: str = ""
    parse_error: Optional[str] = None


@dataclass
class ModelTurn:
    """One assistant response."""

    content: str
    tool_calls: List[ToolCall]
    model: str
    finish_reason: str
    usage: Dict[str, int] = field(default_factory=dict)
    retries: int = 0


class ChatModel(Protocol):
    name: str

    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> ModelTurn: ...


class ModelMismatchError(RuntimeError):
    """The server answered with a different model than the one requested."""


class OpenAICompatModel:
    """Chat model served over the OpenAI-compatible API.

    Examples::

        OpenAICompatModel("qwen2.5-coder:7b", base_url="http://localhost:11434/v1")  # Ollama
        OpenAICompatModel("llama-3.3-70b-versatile", base_url="https://api.groq.com/openai/v1",
                          api_key=os.environ["GROQ_API_KEY"])
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "not-needed",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_attempts: int = 5,
        request_timeout_s: float = 300.0,
    ):
        from openai import OpenAI

        self.name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        # Retries are done here (not by the client) so they can be counted.
        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=request_timeout_s)

    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> ModelTurn:
        import openai

        for attempt in range(self.max_attempts):
            try:
                resp = self._client.chat.completions.create(
                    model=self.name, messages=messages, tools=tools,
                    temperature=self.temperature, max_tokens=self.max_tokens,
                )
                break
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError):
                if attempt == self.max_attempts - 1:
                    raise
                # Jittered exponential backoff on transient errors.
                time.sleep(min(60.0, 2 ** attempt) * (0.5 + random.random()))
        served = resp.model or self.name
        if not _same_model(self.name, served):
            raise ModelMismatchError(f"requested {self.name!r} but the server answered as {served!r}")

        choice = resp.choices[0]
        msg = choice.message
        calls = []
        for tc in msg.tool_calls or []:
            raw = tc.function.arguments or "{}"
            try:
                args, err = json.loads(raw), None
                if not isinstance(args, dict):
                    args, err = {}, "arguments are not a JSON object"
            except json.JSONDecodeError as exc:
                args, err = {}, f"invalid JSON arguments: {exc}"
            calls.append(ToolCall(tc.id or f"call_{len(calls)}", tc.function.name, args, raw, err))
        usage = {}
        if resp.usage:
            usage = {"input_tokens": resp.usage.prompt_tokens or 0,
                     "output_tokens": resp.usage.completion_tokens or 0}
        return ModelTurn(msg.content or "", calls, served, choice.finish_reason or "", usage, attempt)


def parse_text_tool_calls(text: str, tool_names) -> List[ToolCall]:
    """Recover tool calls a model wrote as JSON text instead of using the tool-calling API.

    Many open-weight models (or their chat templates) emit `{"name": ..., "arguments": {...}}`
    in plain text, sometimes inside ```json fences or <tool_call> tags. Only objects whose
    `name` is a known tool are accepted.
    """
    decoder = json.JSONDecoder(strict=False)  # tolerate raw newlines inside code strings
    calls: List[ToolCall] = []
    i = 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict) and obj.get("name") in tool_names:
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = decoder.decode(args)
                except json.JSONDecodeError:
                    args = {}
            if isinstance(args, dict):
                calls.append(ToolCall(f"text_call_{len(calls)}", obj["name"], args, json.dumps(args)))
        i = end
    return calls


def _same_model(requested: str, served: str) -> bool:
    """Tolerate providers that echo a namespaced or tagged id (e.g. 'meta/llama-3' vs 'llama-3')."""
    norm = lambda s: s.lower().split("/")[-1].removesuffix(":latest")  # noqa: E731
    return norm(requested) == norm(served) or norm(served).startswith(norm(requested))
