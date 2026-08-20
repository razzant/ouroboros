"""Golden characterisation of every provider route in ``ouroboros/llm.py``.

The fixtures in ``tests/fixtures/llm_golden/`` record, per route, the exact
projection a route produces: the resolved provider target, the client the route
constructs (base url, header set, retry policy, proxy trust), every request
payload actually handed to a transport, the canonical byte digest of that
payload, the physical-attempt ledger rows the send produced, and the
``(message, usage)`` the route returns.

Nothing here touches the network: the OpenAI SDK, ``requests``, ``httpx`` and
the ``gigachat`` library are replaced by recording fakes driven by a scripted
response queue, and no fixture contains a real credential (every key is an
obviously synthetic ``*-fixture-key``).

The digests make the golden byte-level: reordering a payload key, adding a
header, resolving a different model slot, or changing the retry/fallback order
changes a recorded digest or the recorded attempt sequence and fails here.

Regenerating (deliberate re-baselining only — every diff must be explained):

    ~/ouro/venv/bin/python tests/test_llm_provider_golden.py --write
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import types
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

REPO = pathlib.Path(__file__).parents[1]
if str(REPO) not in sys.path:  # allows the ``--write`` entry point below
    sys.path.insert(0, str(REPO))

from ouroboros import llm as llm_module  # noqa: E402
from ouroboros.llm import LLMClient  # noqa: E402

FIXTURE_DIR = REPO / "tests" / "fixtures" / "llm_golden"

# Every environment name any route reads. Cleared before each case so a host
# export can never leak into a recorded projection.
_ROUTE_ENV_NAMES = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_COMPATIBLE_API_KEY",
    "OPENAI_COMPATIBLE_BASE_URL",
    "ANTHROPIC_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_REGION",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_BASE_URL",
    "GIGACHAT_CREDENTIALS",
    "GIGACHAT_USER",
    "GIGACHAT_PASSWORD",
    "GIGACHAT_BASE_URL",
    "GIGACHAT_SCOPE",
    "GIGACHAT_VERIFY_SSL_CERTS",
    "LOCAL_MODEL_PORT",
    "OUROBOROS_OR_PROVIDER",
    "OUROBOROS_RETURN_REASONING",
    "OUROBOROS_MAIN_WEB_SEARCH",
    "OUROBOROS_MAIN_WEB_SEARCH_ENGINE",
    "OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS",
    "OUROBOROS_PROMPT_CACHE_TTL",
    "OUROBOROS_RUB_USD_RATE",
    "OUROBOROS_MODEL",
    "OUROBOROS_MODEL_HEAVY",
    "OUROBOROS_MODEL_LIGHT",
    "OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC",
    "OUROBOROS_OBSERVABILITY_KEEP_RAW",
)

# Class-level caches LLMClient uses as process-global memory. Reset per case.
_CLASS_CACHE_NAMES = (
    "_SUPPORTED_PARAMS_CACHE",
    "_CONTEXT_LENGTH_CACHE",
    "_REJECTED_PARAMS_CACHE",
    "_REJECTED_PARAMS_LOADED",
    "_EFFORT_CEILING_CACHE",
    "_EFFORT_CEILING_LOADED",
    "_EFFORT_FLOOR_CACHE",
    "_EFFORT_FLOOR_LOADED",
)


# ---------------------------------------------------------------------------
# Synthetic provider failure. NOT a copy of any provider SDK type: it carries
# only the structural attributes llm.py reads (status_code / body / code).
# ---------------------------------------------------------------------------
class FixtureProviderError(Exception):
    """A scripted transport failure with provider-shaped structured facts."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        code: str = "",
        error_type: str = "",
        body: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = int(status_code)
        if code:
            self.code = code
        if error_type:
            self.type = error_type
        if body is not None:
            self.body = body


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=str,
    ).encode("utf-8")


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


class _Recorder:
    """Collects the ordered transport observations of one case."""

    def __init__(self, script: List[Dict[str, Any]]) -> None:
        self.script = list(script)
        self.sends: List[Dict[str, Any]] = []
        self.pricing_calls: List[Dict[str, Any]] = []
        self.sleeps: List[float] = []

    def next_step(self, transport: str) -> Dict[str, Any]:
        if not self.script:
            raise AssertionError(
                f"transport {transport} asked for send #{len(self.sends) + 1} "
                "but the fixture script is exhausted"
            )
        return self.script.pop(0)

    def record(self, transport: str, *, payload: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
        step = self.next_step(transport)
        row: Dict[str, Any] = {"transport": transport}
        row.update({key: _jsonable(value) for key, value in extra.items()})
        row["payload"] = _jsonable(payload)
        row["payload_sha256"] = _digest(payload)
        self.sends.append(row)
        return step


# ---------------------------------------------------------------------------
# Fake transports
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, body: Dict[str, Any]) -> None:
        self._body = body

    def model_dump(self) -> Dict[str, Any]:
        return copy.deepcopy(self._body)

    # probe_oversized_context reads the SDK object shape directly.
    @property
    def choices(self) -> Any:
        raw = self._body.get("choices") or []
        return [types.SimpleNamespace(message=types.SimpleNamespace(**(c.get("message") or {}))) for c in raw]

    @property
    def usage(self) -> Any:
        return types.SimpleNamespace(**(self._body.get("usage") or {}))


def _raise_step(step: Dict[str, Any]) -> None:
    raise FixtureProviderError(
        str(step.get("message") or "scripted failure"),
        status_code=step.get("status_code"),
        code=str(step.get("code") or ""),
        error_type=str(step.get("error_type") or ""),
        body=step.get("body"),
    )


def _resolve_step(step: Dict[str, Any]) -> Any:
    if step.get("kind") == "error":
        _raise_step(step)
    return _FakeResponse(step.get("body") or {})


class _FakeCompletions:
    def __init__(self, client: "_FakeOpenAI") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        step = self._client.recorder.record(
            "openai.chat.completions.create",
            payload=kwargs,
            client=self._client.observed,
        )
        return _resolve_step(step)


class _FakeAsyncCompletions:
    def __init__(self, client: "_FakeOpenAI") -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        step = self._client.recorder.record(
            "async_openai.chat.completions.create",
            payload=kwargs,
            client=self._client.observed,
        )
        return _resolve_step(step)


class _FakeOpenAI:
    recorder: _Recorder
    is_async = False

    def __init__(self, **kwargs: Any) -> None:
        http_client = kwargs.get("http_client")
        self.observed = {
            "api_key": kwargs.get("api_key"),
            "base_url": kwargs.get("base_url"),
            "default_headers": kwargs.get("default_headers"),
            "max_retries": kwargs.get("max_retries"),
            "http_client": getattr(http_client, "observed", None),
        }
        self._options: Dict[str, Any] = {}
        chat = types.SimpleNamespace()
        chat.completions = _FakeAsyncCompletions(self) if self.is_async else _FakeCompletions(self)
        self.chat = chat

    def with_options(self, **options: Any) -> "_FakeOpenAI":
        clone = copy.copy(self)
        clone.observed = dict(self.observed, request_options=_jsonable(options))
        chat = types.SimpleNamespace()
        chat.completions = _FakeAsyncCompletions(clone) if self.is_async else _FakeCompletions(clone)
        clone.chat = chat
        return clone


class _FakeAsyncOpenAI(_FakeOpenAI):
    is_async = True


class _FakeHttpxClient:
    recorder: _Recorder

    def __init__(self, **kwargs: Any) -> None:
        timeout = kwargs.get("timeout")
        self.observed = {
            "trust_env": kwargs.get("trust_env"),
            "mounts": kwargs.get("mounts"),
            "timeout": {
                "connect": getattr(timeout, "connect", None),
                "read": getattr(timeout, "read", None),
                "write": getattr(timeout, "write", None),
                "pool": getattr(timeout, "pool", None),
            },
        }

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class _FakeRequestsResponse:
    def __init__(self, step: Dict[str, Any], url: str) -> None:
        self.status_code = int(step.get("status_code") or 200)
        self.reason = str(step.get("reason") or "OK")
        self.url = url
        self._json = step.get("json") or {}
        self.text = str(step.get("text") or json.dumps(self._json, ensure_ascii=False))

    def json(self) -> Any:
        return copy.deepcopy(self._json)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} for url {self.url}", response=self)


def _install_fakes(stack: contextlib.ExitStack, recorder: _Recorder, spec: Dict[str, Any]) -> None:
    import httpx
    import requests

    openai_cls = type("_CaseOpenAI", (_FakeOpenAI,), {"recorder": recorder})
    async_cls = type("_CaseAsyncOpenAI", (_FakeAsyncOpenAI,), {"recorder": recorder})
    httpx_cls = type("_CaseHttpx", (_FakeHttpxClient,), {"recorder": recorder})
    stack.enter_context(mock.patch("openai.OpenAI", openai_cls))
    stack.enter_context(mock.patch("openai.AsyncOpenAI", async_cls))
    stack.enter_context(mock.patch.object(httpx, "Client", httpx_cls))
    stack.enter_context(mock.patch.object(httpx, "AsyncClient", httpx_cls))

    def _post(url: str, *, headers: Optional[Dict[str, str]] = None, json: Any = None,
              timeout: Any = None, trust_env: Any = None, **_rest: Any) -> Any:
        step = recorder.record(
            "requests.post",
            payload=json or {},
            url=url,
            headers=headers or {},
            timeout=timeout,
            session_trust_env=trust_env,
        )
        if step.get("kind") == "error":
            _raise_step(step)
        return _FakeRequestsResponse(step, url)

    def _get(url: str, *, headers: Optional[Dict[str, str]] = None, timeout: Any = None,
             **_rest: Any) -> Any:
        step = recorder.record(
            "requests.get",
            payload={},
            url=url,
            headers=sorted((headers or {}).keys()),
            timeout=timeout,
        )
        if step.get("kind") == "error":
            _raise_step(step)
        return _FakeRequestsResponse(step, url)

    class _FakeSession:
        def __init__(self) -> None:
            self.trust_env = True

        def __enter__(self) -> "_FakeSession":
            return self

        def __exit__(self, *_exc: Any) -> None:
            return None

        def post(self, url: str, **kwargs: Any) -> Any:
            return _post(url, trust_env=self.trust_env, **kwargs)

    stack.enter_context(mock.patch.object(requests, "post", _post))
    stack.enter_context(mock.patch.object(requests, "get", _get))
    stack.enter_context(mock.patch.object(requests, "Session", _FakeSession))

    # The gigachat library is optional; a synthetic module keeps the lane
    # replayable on a host (or CI runner) that never installed it.
    class _FakeGigaChat:
        def __init__(self, **kwargs: Any) -> None:
            self.observed = _jsonable(kwargs)

        def chat(self, payload: Dict[str, Any]) -> Any:
            step = recorder.record("gigachat.chat", payload=payload, client=self.observed)
            if step.get("kind") == "error":
                _raise_step(step)
            return _gigachat_completion(step.get("body") or {})

    gigachat_module = types.ModuleType("gigachat")
    gigachat_module.GigaChat = _FakeGigaChat  # type: ignore[attr-defined]

    # Same for the Anthropic SDK used by the provider-owned web_search tool.
    class _FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            self.observed = _jsonable(kwargs)
            self.messages = types.SimpleNamespace(create=self._create)

        def _create(self, **payload: Any) -> Any:
            step = recorder.record("anthropic.messages.create", payload=payload, client=self.observed)
            return _resolve_step(step)

    anthropic_module = types.ModuleType("anthropic")
    anthropic_module.Anthropic = _FakeAnthropic  # type: ignore[attr-defined]
    stack.enter_context(mock.patch.dict(
        sys.modules, {"gigachat": gigachat_module, "anthropic": anthropic_module}
    ))

    # Deterministic reroute affinity: the production key folds in time_ns().
    stack.enter_context(mock.patch("time.time_ns", lambda: 1_700_000_000_000_000_000))
    stack.enter_context(mock.patch("time.sleep", recorder.sleeps.append))

    # Cost projection: record the exact arguments each lane hands the pricer.
    estimate = spec.get("pricing_estimate", None)

    def _estimate_cost_optional(model: str, prompt_tokens: int, completion_tokens: int,
                                **kwargs: Any) -> Optional[float]:
        recorder.pricing_calls.append(_jsonable({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            **kwargs,
        }))
        return estimate

    import ouroboros.pricing as pricing_module
    import ouroboros.usage_accounting as usage_module

    stack.enter_context(
        mock.patch.object(pricing_module, "estimate_cost_optional", _estimate_cost_optional)
    )
    # The ledger's own admission pricer holds its own import-time reference and
    # would otherwise reach the live catalog (and the process-global pricing
    # cache) from inside a recorded route.
    stack.enter_context(mock.patch.object(
        usage_module, "estimate_cost_optional", lambda *_a, **_k: 0.001
    ))
    stack.enter_context(mock.patch.object(pricing_module, "_cached_pricing", {}))
    stack.enter_context(mock.patch.object(pricing_module, "_pricing_fetched_at", {}))

    ctx_len = spec.get("local_context_length")
    if ctx_len is not None:
        import ouroboros.local_model as local_model_module

        stack.enter_context(mock.patch.object(
            local_model_module, "get_manager",
            lambda: types.SimpleNamespace(get_context_length=lambda: int(ctx_len)),
        ))


def _gigachat_completion(body: Dict[str, Any]) -> Any:
    """Build a GigaChat-library-shaped completion object from fixture JSON."""
    message = body.get("message") or {}
    function_call = message.get("function_call")
    gmsg = types.SimpleNamespace(
        content=message.get("content", ""),
        function_call=types.SimpleNamespace(**function_call) if function_call else None,
    )
    usage = body.get("usage")
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=gmsg)],
        usage=types.SimpleNamespace(**usage) if usage else None,
    )


# ---------------------------------------------------------------------------
# Case execution
# ---------------------------------------------------------------------------
def _ledger_projection(root: pathlib.Path) -> List[Dict[str, Any]]:
    """Ordered physical attempts with their stable accounting facts."""
    path = root / "state" / "usage_attempts.jsonl"
    if not path.is_file():
        return []
    attempts: List[Dict[str, Any]] = []
    index: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        attempt_id = str(row.get("attempt_id") or "")
        entry = index.get(attempt_id)
        if entry is None:
            entry = {
                "source": row.get("source"),
                "model": row.get("model"),
                "provider": row.get("provider"),
                "candidate_raw_sha256": row.get("candidate_raw_sha256"),
                "candidate_raw_size_bytes": row.get("candidate_raw_size_bytes"),
                "candidate_measurement_kind": row.get("candidate_measurement_kind"),
                "states": [],
            }
            index[attempt_id] = entry
            attempts.append(entry)
        entry["states"].append(row.get("state"))
        if row.get("state") == "settled":
            entry["prompt_cache_ttl"] = row.get("prompt_cache_ttl")
            entry["cost_final"] = row.get("cost_final")
    return attempts


def _call_route(client: LLMClient, call: Dict[str, Any]) -> Any:
    # Deep-copied: a route may legitimately mutate an argument in place
    # (``add_usage`` accumulates into ``total``), and the fixture must describe
    # the call, not carry the residue of the last replay.
    call = copy.deepcopy(call)
    kind = str(call.get("kind") or "method")
    if kind == "resolve_target":
        return client._resolve_remote_target(call["model"])
    if kind == "build_kwargs":
        target = client._resolve_remote_target(call["model"])
        args = call.get("args") or {}
        return client._build_remote_kwargs(
            target,
            args.get("messages") or [],
            args.get("reasoning_effort", "medium"),
            int(args.get("max_tokens", 1024)),
            args.get("tool_choice", "auto"),
            args.get("temperature"),
            args.get("tools"),
            **{k: v for k, v in args.items() if k in {
                "skip_capability_fetch", "allow_server_web_search", "response_format",
                "cache_affinity", "bypass_response_cache",
            }},
        )
    if kind == "method":
        bound = getattr(client, call["name"])
        result = bound(**(call.get("kwargs") or {}))
        if asyncio.iscoroutine(result):
            return asyncio.run(result)
        return result
    if kind == "function":
        return getattr(llm_module, call["name"])(**(call.get("kwargs") or {}))
    if kind == "sequence":
        return [_call_route(client, step) for step in call["calls"]]
    raise AssertionError(f"unknown call kind {kind!r}")


def _observe(spec: Dict[str, Any]) -> Dict[str, Any]:
    recorder = _Recorder(spec.get("transport") or [])
    root = pathlib.Path(tempfile.mkdtemp(prefix="llm_golden_"))
    (root / "state").mkdir(parents=True, exist_ok=True)
    saved_caches = {name: copy.deepcopy(getattr(LLMClient, name)) for name in _CLASS_CACHE_NAMES}
    saved_flags = (LLMClient._SUPPORTED_PARAMS_FETCHED, LLMClient._CAPABILITIES_FETCH_OK)
    observed: Dict[str, Any] = {}
    with contextlib.ExitStack() as stack:
        env = stack.enter_context(mock.patch.dict(os.environ, {}, clear=False))
        del env
        for name in _ROUTE_ENV_NAMES:
            os.environ.pop(name, None)
        os.environ["OUROBOROS_DATA_DIR"] = str(root)
        os.environ["OUROBOROS_SETTINGS_PATH"] = str(root / "settings.json")
        os.environ["TOTAL_BUDGET"] = str(spec.get("total_budget", 1000))
        for key, value in (spec.get("env") or {}).items():
            os.environ[key] = str(value)

        # Durable capability evidence is read through the frozen ``config.DATA_DIR``
        # constant, so the env override alone would let host state leak into (and a
        # learned rejection leak out of) a recorded projection.
        import ouroboros.config as config_module

        stack.enter_context(mock.patch.object(config_module, "DATA_DIR", root))
        stack.enter_context(mock.patch.object(config_module, "SETTINGS_PATH", root / "settings.json"))

        for name in _CLASS_CACHE_NAMES:
            setattr(LLMClient, name, type(saved_caches[name])())
        LLMClient._SUPPORTED_PARAMS_FETCHED = bool(spec.get("capabilities_fetched", True))
        LLMClient._CAPABILITIES_FETCH_OK = bool(spec.get("capabilities_fetch_ok", True))
        for model_id, params in (spec.get("supported_parameters") or {}).items():
            LLMClient._SUPPORTED_PARAMS_CACHE[model_id] = set(params)

        durable = spec.get("durable_evidence") or {}
        if durable:
            from ouroboros import capability_evidence as ce

            for key, value in (durable.get("effort_ceilings") or {}).items():
                ce.record_effort_ceiling(root, key, value)
            for key, value in (durable.get("effort_floors") or {}).items():
                ce.record_effort_floor(root, key, value)
            for key, value in (durable.get("rejected_params") or {}).items():
                ce.record_rejected_params(root, key, value)

        _install_fakes(stack, recorder, spec)

        client_args = spec.get("client") or {}
        client = LLMClient(**client_args)
        try:
            result = _call_route(client, spec["call"])
        except BaseException as exc:  # noqa: BLE001 - the raise IS the projection
            observed["raised"] = {"type": type(exc).__name__, "message": str(exc)}
        else:
            observed["returned"] = _project_result(result, str(spec["call"].get("project") or ""))

    for name, value in saved_caches.items():
        setattr(LLMClient, name, value)
    LLMClient._SUPPORTED_PARAMS_FETCHED, LLMClient._CAPABILITIES_FETCH_OK = saved_flags

    observed["sends"] = recorder.sends
    if recorder.sleeps:
        observed["transport_sleeps"] = [round(float(v), 3) for v in recorder.sleeps]
    if recorder.pricing_calls:
        observed["pricing_calls"] = recorder.pricing_calls
    ledger = _ledger_projection(root)
    if ledger:
        observed["physical_attempts"] = ledger
    observed["unused_script_steps"] = len(recorder.script)
    return _jsonable(observed)


def _project_result(result: Any, projector: str = "") -> Any:
    if projector == "pricing_catalog":
        # A pricing row is a tuple subclass whose prompt-length tiers live on an
        # attribute JSON would silently drop.
        return {
            model_id: {
                "base": list(row),
                "tiers": [[int(size), list(tier)] for size, tier in getattr(row, "tiers", ())],
            }
            for model_id, row in sorted((result or {}).items())
        }
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        message, usage = result
        usage = dict(usage)
        attempt_ids = usage.pop("ledger_attempt_ids", None)
        projected: Dict[str, Any] = {"message": _jsonable(message), "usage": _jsonable(usage)}
        if attempt_ids is not None:
            projected["ledger_attempt_count"] = len(attempt_ids)
        return projected
    if isinstance(result, dict):
        return {"value": _jsonable(result), "value_sha256": _digest(result)}
    if hasattr(result, "model_dump"):
        return {"value": _jsonable(result.model_dump())}
    return {"value": _jsonable(result)}


# ---------------------------------------------------------------------------
# The table-driven replay
# ---------------------------------------------------------------------------
def _load_files() -> List[pathlib.Path]:
    return sorted(FIXTURE_DIR.glob("*.json"))


def _load_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for path in _load_files():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            case = dict(case)
            case["_file"] = path.name
            cases.append(case)
    return cases


_CASES = _load_cases()


def test_golden_fixture_ids_are_unique_and_documented():
    ids = [case["id"] for case in _CASES]
    assert len(ids) == len(set(ids)), "duplicate golden case id"
    for case in _CASES:
        assert case.get("route"), f"{case['id']} has no route description"


@pytest.mark.parametrize("case", _CASES, ids=[case["id"] for case in _CASES])
def test_llm_provider_route_matches_golden(case):
    observed = _observe(case["spec"])
    assert observed == case["expected"], (
        f"route {case['id']} drifted from {case['_file']}:\n"
        f"observed={json.dumps(observed, indent=2, sort_keys=True)}"
    )


def test_golden_covers_every_declared_provider_lane():
    """Coverage floor: dropping a lane's fixtures must fail, not go unnoticed."""
    covered = {case["id"].split(".", 1)[0] for case in _CASES}
    assert covered >= {
        "target", "openrouter", "openai", "compatible", "cloudru", "minimax",
        "anthropic", "gigachat", "local", "fallback", "aux",
    }
    ledger_sources = {
        attempt["source"]
        for case in _CASES
        for attempt in case["expected"].get("physical_attempts", [])
    }
    assert ledger_sources >= {
        "llm.chat", "llm.local", "llm.anthropic", "llm.gigachat",
        "capability_probe", "web_search.openrouter", "web_search.anthropic",
    }


def _write_golden() -> int:
    """Re-record every fixture's ``expected`` block from the live code."""
    changed = 0
    for path in _load_files():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            observed = _observe(case["spec"])
            if case.get("expected") != observed:
                changed += 1
                print(f"re-recorded {case['id']}")
            case["expected"] = observed
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
            encoding="utf-8",
        )
    print(f"{changed} case(s) re-recorded")
    return changed


if __name__ == "__main__":
    if "--write" not in sys.argv[1:]:
        raise SystemExit("usage: python tests/test_llm_provider_golden.py --write")
    _write_golden()
