"""Bounded, explicit API transport for AI development references only.

Importing this module does not read credentials, call APIs, or change strategy.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, HTTPRedirectHandler, build_opener

from research.llm_contract import VERSION_V3, FactConstraintError, review_e_facts, validate_response

BASE_URLS = {"b.ai": "https://api.b.ai/v1", "openrouter": "https://openrouter.ai/api/v1",
             "deepseek_native": "https://api.deepseek.com"}
KEY_NAMES = {"b.ai": ("BAI_API_KEY", "B.ai_API_KEY"), "openrouter": ("OPENROUTER_API_KEY",),
             "deepseek_native": ("DEEPSEEK_API_KEY",)}
MODEL_IDS = {"b.ai": "deepseek-v4.1-flash", "openrouter": "deepseek/deepseek-v4.1-flash",
             "deepseek_native": "deepseek-flash"}


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    api_key: str = field(repr=False)


def load_providers(env_path):
    """Parse data, never source/execute .env. Accept the user's b.ai alias."""
    values = {}
    model_hints = set()
    for line in Path(env_path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("#"):
            model_hints.update(re.findall(r"deepseek-v[0-9.]+-flash", text, re.I))
            continue
        if "=" not in text:
            raise ValueError("invalid configuration line; values are not shell commands")
        key, value = text.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key in values:
            raise ValueError("duplicate configuration key")
        values[key] = value
    hints = model_hints | {values[k] for k in ("LLM_MODEL", "MODEL") if values.get(k)}
    if hints and hints != {"deepseek-v4.1-flash"}:
        raise ValueError("this bounded pilot requires the explicitly selected V4.1 Flash family")
    providers = []
    for name, aliases in KEY_NAMES.items():
        keys = {values[k] for k in aliases if values.get(k)}
        if len(keys) > 1:
            raise ValueError("conflicting credential aliases")
        if keys:
            api_key = next(iter(keys))
            if not re.fullmatch(r"[!-~]{8,1024}", api_key):
                raise ValueError("invalid credential shape")
            providers.append(Provider(name, MODEL_IDS[name], api_key))
    if not providers:
        raise ValueError("no configured provider")
    return providers


class ProviderFailure(Exception):
    """Safe error fields only; no URLs, headers, keys, or server error bodies."""
    def __init__(self, kind, status=None):
        self.kind, self.status = kind, status
        super().__init__(kind)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("nonfinite JSON number")
    return value


class Transport:
    def __init__(self, timeout=60):
        self.timeout = timeout
        self.opener = build_opener(_NoRedirect)

    def request(self, provider, *, path, payload=None):
        if provider.name not in BASE_URLS or path not in ("/models", "/chat/completions"):
            raise ValueError("unrecognized API destination")
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8") if payload is not None else None
        request = Request(BASE_URLS[provider.name] + path, data=body,
                          headers={"Authorization": "Bearer " + provider.api_key,
                                   "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ProviderFailure("response_too_large")
            try:
                value = json.loads(raw, parse_float=_finite_float,
                                   parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite envelope")))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ProviderFailure("non_json_envelope") from exc
            if not isinstance(value, dict):
                raise ProviderFailure("invalid_envelope")
            return value
        except HTTPError as exc:
            # Generic 400 is usually a request error: do not hide it by hopping.
            if exc.code == 400:
                try:
                    error = json.loads(exc.read(20_000)).get("error", {})
                    message = str(error.get("message", "")) if isinstance(error, dict) else ""
                except (ValueError, AttributeError):
                    message = ""
                unsupported = bool(re.search(r"(?:unsupported|not supported|not found|does not exist|invalid model)", message, re.I))
                raise ProviderFailure("unsupported_capability" if unsupported else "bad_request", 400) from None
            if exc.code in (401, 402, 403, 404, 408, 429) or 500 <= exc.code < 600:
                raise ProviderFailure("unavailable_http", exc.code) from None
            raise ProviderFailure("unexpected_http", exc.code) from None
        except (URLError, TimeoutError, socket.timeout, ConnectionError):
            raise ProviderFailure("network_unavailable") from None


def request_body(provider, prompt, *, max_tokens=8192, effort="low", profile="pilot"):
    if provider.name not in MODEL_IDS or provider.model != MODEL_IDS[provider.name]:
        raise ValueError("unverified pilot provider/model mapping")
    allowed = (type(max_tokens) is int and
               ((profile == "pilot" and 1 <= max_tokens <= 8192) or
                (profile in ("output_budget_diagnostic", "asof_revision_pilot", "facts_revision_pilot")
                 and max_tokens == 16384)))
    if not allowed or effort != "low":
        raise ValueError("pilot output/effort bounds exceeded")
    if profile == "asof_revision_pilot" and prompt.get("prompt_version") != "wyckoff-prompt-v2-asof":
        raise ValueError("asof pilot requires the explicit scoped prompt")
    if prompt.get("prompt_version") == "wyckoff-prompt-v2-asof" and profile != "asof_revision_pilot":
        raise ValueError("scoped prompt requires the bounded asof pilot profile")
    if (profile == "facts_revision_pilot") != (prompt.get("prompt_version") == "wyckoff-prompt-v3-facts"):
        raise ValueError("facts prompt and the bounded facts pilot profile require each other")
    body = {"model": provider.model, "messages": deepcopy(prompt["messages"]),
            "stream": False, "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
    if provider.name == "openrouter":
        body.update(reasoning={"effort": effort, "exclude": True},
                    provider={"only": ["deepseek"], "order": ["deepseek"], "allow_fallbacks": False,
                              "require_parameters": True,
                              "max_price": {"prompt": .30, "completion": 1.20}})
    else:
        body["reasoning_effort"] = effort
        # These exact DeepSeek models enable thinking by default. Use the
        # documented effort control; no ignored temperature/reasoning history.
    return body


@dataclass
class Budget:
    max_requests: int = 12
    max_usd: float = 1.
    requests: int = 0
    reserved_usd: float = 0.

    def __post_init__(self):
        if (type(self.max_requests) is not int or not 1 <= self.max_requests <= 12
                or type(self.max_usd) not in (int, float) or not math.isfinite(self.max_usd)
                or not 0 < self.max_usd <= 1):
            raise ValueError("bounded pilot requires <=12 requests and <=1 USD")

    def reserve(self, body):
        # Byte-based conservative allowance, not a tokenizer-exact guarantee.
        # Price uses full Peak miss/write rates; no discount/cache assumption.
        tokens_upper = 2 * len(json.dumps(body["messages"], ensure_ascii=False).encode("utf-8")) + 4096
        ceiling = (tokens_upper * .30 + body["max_tokens"] * 1.20) / 1_000_000
        if self.requests >= self.max_requests or self.reserved_usd + ceiling > self.max_usd:
            raise BudgetExceeded("pilot budget exhausted")
        self.requests += 1
        self.reserved_usd += ceiling
        return ceiling


class BudgetExceeded(ValueError):
    pass


def safe_response(value, providers, *, omit_reasoning=True):
    """Store final response/usage; omit private reasoning and redact known keys."""
    keys = [p.api_key for p in providers if p.api_key]
    def walk(item):
        if isinstance(item, dict):
            return {walk(k): walk(v) for k, v in item.items()
                    if not (omit_reasoning and k in ("reasoning_content", "reasoning", "reasoning_details"))}
        if isinstance(item, list):
            return [walk(v) for v in item]
        if isinstance(item, str):
            for key in keys:
                item = item.replace(key, "[REDACTED]")
        return item
    return walk(value)


def _usage_cost(raw):
    usage = raw.get("usage", {})
    if not isinstance(usage, dict):
        return {"reported_cost_usd": None, "peak_estimate_usd": None}
    reported = usage.get("cost")
    invalid_usage = reported is not None and (type(reported) not in (int, float) or not 0 <= reported <= 1_000_000)
    reported = reported if not invalid_usage else None
    p, c = usage.get("prompt_tokens"), usage.get("completion_tokens")
    invalid_usage = invalid_usage or any(x is not None and (type(x) is not int or not 0 <= x <= 10_000_000) for x in (p, c))
    estimate = (p * .30 + c * 1.20) / 1_000_000 if type(p) is int and type(c) is int and 0 <= p <= 10_000_000 and 0 <= c <= 10_000_000 else None
    return {"reported_cost_usd": reported, "peak_estimate_usd": estimate, "usage_invalid": invalid_usage}


FALLBACK_ERRORS = {"unavailable_http", "network_unavailable", "unsupported_capability",
                   "non_json_envelope", "invalid_envelope", "empty_content", "upstream_error"}


def evaluate_request(item, variant, providers, transport, budget, *, record=lambda event: None,
                     max_tokens=8192, effort="low", profile="pilot"):
    """Only availability failures advance channels. No semantic retries/repair."""
    prompt, bundle = item["prompts"][variant], item["bundle"]
    attempts = []
    for provider in providers:
        body = request_body(provider, prompt, max_tokens=max_tokens, effort=effort, profile=profile)
        try:
            reservation = budget.reserve(body)
        except BudgetExceeded:
            if not attempts:
                raise
            return {"case_ref": bundle["packet"]["case_ref"], "variant": variant,
                    "attempts": attempts, "state": "availability_failure", "budget_exhausted": True}
        attempt = {"provider": provider.name, "requested_model": provider.model,
                   "reserved_usd": reservation, "state": "reserved", "request": deepcopy(body)}
        record(deepcopy(attempt))  # checkpoint before a possibly charged call
        started = time.monotonic()
        try:
            raw = transport.request(provider, path="/chat/completions", payload=body)
            attempt.update(response=safe_response(raw, providers), **_usage_cost(raw),
                           returned_model=raw.get("model"), reasoning_omitted=True)
            if raw.get("error"):
                raise ProviderFailure("upstream_error")
            choices = raw.get("choices")
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise ProviderFailure("invalid_envelope")
            choice = choices[0]
            message = choice.get("message", {})
            if not isinstance(message, dict):
                raise ProviderFailure("invalid_envelope")
            content = message.get("content")
            attempt.update(finish_reason=choice.get("finish_reason"), native_finish_reason=choice.get("native_finish_reason"))
            if not isinstance(content, str) or not content.strip():
                if choice.get("finish_reason") == "stop":
                    raise ProviderFailure("empty_content")
            try:
                aliases = {x.lower() for x in MODEL_IDS.values()}
                if str(raw.get("model", "")).lower() not in aliases or message.get("refusal") or message.get("tool_calls"):
                    raise ValueError("unexpected model/refusal/tool output")
                result = validate_response(content, bundle, finish_reason=choice.get("finish_reason"))
                attempt.update(state="valid", answer=safe_response(result, providers))
                if bundle["packet"]["version"] == VERSION_V3:
                    attempt["fact_review"] = review_e_facts(result, bundle["packet"])
            except FactConstraintError as exc:
                # Well-formed but its declared facts fail the bars: keep the
                # answer and the classified findings apart from contract failures.
                attempt.update(state="fact_findings", answer=safe_response(exc.answer, providers),
                               fact_review=exc.review)
            except (ValueError, TypeError) as exc:
                # Returned content exists: keep this as a failed contract, not
                # a reason to hunt for a different judgment/provider.
                attempt.update(state="invalid_contract", contract_error=str(exc))
        except ProviderFailure as exc:
            attempt.update(state="availability_failure" if exc.kind in FALLBACK_ERRORS else "request_failure",
                           error_kind=exc.kind, http_status=exc.status)
        finally:
            # Even an empty/failed answer can have been charged. Check before
            # considering channel fallback, and redact all copied metadata.
            measured = attempt.get("reported_cost_usd")
            measured = attempt.get("peak_estimate_usd") if measured is None else measured
            if attempt.get("usage_invalid") or (measured is not None and measured > reservation):
                attempt["budget_mismatch"] = True
                budget.reserved_usd = budget.max_usd
            attempt["latency_seconds"] = round(time.monotonic() - started, 3)
            attempt = safe_response(attempt, providers, omit_reasoning=False)
            record(deepcopy(attempt))
            attempts.append(attempt)
        if attempt["state"] != "availability_failure" or attempt.get("budget_mismatch"):
            break
    return {"case_ref": bundle["packet"]["case_ref"], "variant": variant,
            "attempts": attempts, "state": attempts[-1]["state"]}
