"""OpenAI-compatible chat client used by all LLM architectures.

Configure with environment variables:
  VOAI_BASE_URL   base URL of an OpenAI-compatible endpoint (default: https://api.openai.com/v1)
  VOAI_API_KEY    API key for that endpoint
  VOAI_EXTRA_HEADERS  optional JSON object with extra HTTP headers
Per call it records latency, input/output/reasoning tokens and, when the endpoint reports it in
`usage`, the list-price cost (fields `cost` or `list_cost`).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class CallStats:
    calls: int = 0
    llm_s: float = 0.0
    tok_in: int = 0
    tok_out: int = 0
    tok_reason: int = 0
    cost_usd: float = 0.0
    allowance: float = 0.0
    errors: list = field(default_factory=list)

    def add(self, usage, dt):
        self.calls += 1
        self.llm_s += dt
        if usage is None:
            return
        u = usage if isinstance(usage, dict) else usage.model_dump()
        self.tok_in += u.get("prompt_tokens") or 0
        self.tok_out += u.get("completion_tokens") or 0
        self.tok_reason += ((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        extra = {k: v for d in u.values() if isinstance(d, dict) for k, v in d.items()}
        self.cost_usd += float(u.get("cost") or extra.get("list_cost") or 0.0)


class LLM:
    def __init__(self, model: str, temperature: float | None = 0.0, max_tokens: int = 8000, retries: int = 4):
        self.model_id = model
        self.name = model.split("/", 1)[-1] if model.count("/") == 1 and not model.startswith("http") else model
        self.client = OpenAI(base_url=os.environ.get("VOAI_BASE_URL", "https://api.openai.com/v1"),
                             api_key=os.environ.get("VOAI_API_KEY", ""), timeout=300)
        self.headers = json.loads(os.environ.get("VOAI_EXTRA_HEADERS", "{}"))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries

    def chat(self, messages, stats: CallStats, tools=None, session: str | None = None, json_mode=False):
        kw = dict(model=self.name, messages=messages, max_tokens=self.max_tokens)
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = "auto"
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        last = None
        for attempt in range(self.retries):
            t0 = time.time()
            try:
                r = self.client.chat.completions.create(extra_headers=self.headers, **kw)
                stats.add(r.usage, time.time() - t0)
                return r.choices[0].message
            except Exception as e:
                last = e
                msg = str(e)
                if "usage limit" in msg.lower() or "quota" in msg.lower():
                    stats.errors.append(f"{type(e).__name__}: {msg[:200]}")
                    raise
                if "temperature" in msg and "temperature" in kw:
                    kw.pop("temperature")
                    continue
                if "response_format" in msg and "response_format" in kw:
                    kw.pop("response_format")
                    continue
                time.sleep(min(60, 4 * 2 ** attempt))
        stats.errors.append(f"{type(last).__name__}: {str(last)[:200]}")
        raise last
