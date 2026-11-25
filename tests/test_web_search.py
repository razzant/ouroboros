"""Tests for ``ouroboros.tools.search._web_search``: its legs (OpenAI Responses, DeepSeek Search) and cascade.

Merged from former ``test_search_tool.py`` (provider routing / required-env
contract) and ``test_web_search_streaming.py`` (streaming events, progress,
cost). Both files exercised the same `_web_search` function with overlapping
mocks; the merged file shares one ``_FakeStream`` / event factory.
"""
from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import httpx
import pytest

import ouroboros.tools.search as search_module
from ouroboros.tools.search import _web_search
from tests._tariff_guard import _no_unscripted_tariff_reads  # noqa: F401  (autouse: DeepSeek sends here)
from tests._usage_store_testing import ledger_rows


@pytest.fixture(autouse=True)
def _anthropic_sdk_present(monkeypatch):
    """These tests fake the Anthropic search call; the helper still imports the optional
    SDK first, so its presence is chosen here, not inherited from the interpreter."""
    monkeypatch.setitem(sys.modules, "anthropic", sys.modules.get("anthropic") or types.ModuleType("anthropic"))


# ---------------------------------------------------------------------------
# Shared streaming fixtures
# ---------------------------------------------------------------------------


def _make_event(etype: str, **kwargs):
    return types.SimpleNamespace(type=etype, **kwargs)


def _make_completed_event(input_tokens: int = 100, output_tokens: int = 50):
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    # SDK model_dump returns finite data. An unconstrained MagicMock instead
    # creates another model_dump mock recursively during citation extraction.
    resp_obj = types.SimpleNamespace(
        usage=types.SimpleNamespace(model_dump=lambda: dict(usage)),
        model_dump=lambda: {"usage": dict(usage), "output": []},
    )
    return _make_event("response.completed", response=resp_obj)


class _FakeStream:
    """Iterable that yields pre-built streaming events."""

    def __init__(self, events):
        self._events = events

    def __iter__(self):
        return iter(self._events)


@pytest.fixture
def ctx():
    c = MagicMock()
    c.pending_events = []
    c.emit_progress_fn = MagicMock()
    c.task_id = "task-web"
    c.task_metadata = {
        "root_task_id": "root-web",
        "parent_task_id": "parent-web",
        "delegation_role": "subagent",
    }
    return c


@pytest.fixture
def patch_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    # Deterministic cascade: disable the optional ddgs backend so tests do not
    # depend on whether `ddgs` happens to be installed (and do not make real
    # network calls). Tests that exercise ddgs inject their own fake module.
    monkeypatch.setitem(sys.modules, "ddgs", None)


@pytest.fixture
def mock_openai():
    """Inject a fake openai module so the lazy import inside _web_search works."""
    mock_client = MagicMock()
    mock_module = MagicMock()
    mock_module.OpenAI.return_value = mock_client
    with patch.dict(sys.modules, {"openai": mock_module}):
        yield mock_client


# ---------------------------------------------------------------------------
# Provider routing / required-env contract
# ---------------------------------------------------------------------------


def test_web_search_reports_unavailable_without_any_backend(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "compat-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "ddgs", None)

    result = json.loads(
        search_module._web_search(types.SimpleNamespace(pending_events=[]), "latest news")
    )

    assert result["error"].startswith("web_search unavailable")
    assert "backend_errors" in result


def test_responses_flex_refusal_reprices_standard_and_keeps_original_intent(ctx, patch_env, mock_openai, monkeypatch):
    from ouroboros import usage_accounting as ua

    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "economy")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "openai")
    prepared, holds = [], []
    original_reserve = search_module.reserve_attempt

    def reserve(request):
        prepared.append(request)
        hold = original_reserve(request)
        holds.append(hold)
        return hold

    monkeypatch.setattr(search_module, "reserve_attempt", reserve)
    error = RuntimeError("Resource unavailable")
    error.status_code = 429
    error.body = {"error": {"code": "resource_unavailable", "param": "service_tier"}}
    mock_openai.responses.create.side_effect = [error, _FakeStream([
        _make_event("response.output_text.delta", delta="answer"), _make_completed_event()])]
    result = json.loads(_web_search(ctx, "same query", reasoning_effort="high"))
    assert result["answer"] == "answer"
    assert [call.kwargs["service_tier"] for call in mock_openai.responses.create.call_args_list] == ["flex", "default"]
    assert [request.processing_preference for request in prepared] == ["economy", "economy"]
    assert prepared[0].candidate_raw_sha256 != prepared[1].candidate_raw_sha256
    with ua._locked(holds[0].drive_root):
        records = ledger_rows(holds[0].drive_root)
    matching = [row for row in records if row.get("candidate_raw_sha256") == prepared[0].candidate_raw_sha256]
    assert (matching[-1]["state"], matching[-1]["settle_reason"]) == ("settled", "processing_not_started")
    assert (matching[-1]["cost_usd"], matching[-1]["cost_final"]) == (None, False)
    for request in prepared:
        assert request.effort == {
            "requested": "high", "sent": {"reasoning": {"effort": "high"}},
            "sent_state": "explicit", "sent_source": "host_candidate",
            "reported": None, "report_source": None,
        }
        attempt_rows = [row for row in records if row.get("candidate_raw_sha256") == request.candidate_raw_sha256]
        assert all(row["effort"] == request.effort for row in attempt_rows)


@pytest.mark.parametrize(("work_seen", "input_tokens", "cost", "full_bill", "expected"), [
    (True, 0, 0.0, False, None), (True, 0, 0.125, False, 0.125), (False, 0, 0.0, False, 0.0),
    (True, 0, 0.0, True, 0.0),
    pytest.param(False, 10, 0.0, True, 0.0, id="confirmed-full-zero-with-terminal-tokens"),
    pytest.param(False, 10, 0.0, False, None, id="gateway-zero-with-terminal-tokens"),
])
def test_responses_failure_bills_zero_only_without_contradictory_prior_work(
    ctx, patch_env, mock_openai, monkeypatch, work_seen, input_tokens, cost, full_bill, expected,
):
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "openai")
    holds = []
    original_reserve = search_module.reserve_attempt

    def reserve(request):
        hold = original_reserve(request)
        holds.append(hold)
        return hold

    monkeypatch.setattr(search_module, "reserve_attempt", reserve)
    terminal = {"status": "failed", "error": {"code": "invalid_request_error", "status_code": 400},
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
    if full_bill:
        terminal["total_cost_usd"] = cost
    else:
        terminal["usage"]["cost"] = cost
    events = [_make_event("response.web_search_call.searching")] if work_seen else []
    events.append(_make_event("response.failed", response=types.SimpleNamespace(model_dump=lambda: terminal)))
    mock_openai.responses.create.return_value = _FakeStream(events)
    assert "error" in json.loads(_web_search(ctx, "once"))
    assert mock_openai.responses.create.call_count == len(holds) == 1
    [row] = [row for row in ledger_rows(holds[0].drive_root) if row["attempt_id"] == holds[0].attempt_id]
    assert (row["state"], row["cost_usd"], row["cost_final"]) == ("settled", expected, expected is not None)


def test_web_search_uses_official_openai_responses(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", raising=False)

    calls: dict = {}

    class _Usage:
        def model_dump(self):
            return {"input_tokens": 11, "output_tokens": 7}

    class _CompletedResponse:
        usage = _Usage()

    class _Stream:
        def __iter__(self):
            yield _make_event("response.web_search_call.searching",
                              item_id="ws1", output_index=0, sequence_number=1)
            yield _make_event("response.output_text.delta",
                              delta="fresh answer", content_index=0,
                              item_id="m1", output_index=1, sequence_number=2,
                              logprobs=[])
            yield _make_event("response.completed",
                              response=_CompletedResponse(), sequence_number=3)

    class _Responses:
        def create(self, **kwargs):
            calls["kwargs"] = kwargs
            return _Stream()

    class _Client:
        def __init__(self, api_key=None, base_url=None, **kwargs):  # v6.54.3: accepts timeout
            calls["api_key"] = api_key
            calls["base_url"] = base_url
            self.responses = _Responses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    request_ctx = types.SimpleNamespace(pending_events=[])

    result = json.loads(
        search_module._web_search(request_ctx, "latest news", model="gpt-5.2")
    )

    assert {key: result[key] for key in ("answer", "answer_type", "sources", "backend")} == {"answer": "fresh answer", "answer_type": "summary", "sources": [], "backend": "openai_responses"}
    assert result["legs_tried"][0]["model"] == "gpt-5.2"
    assert calls["api_key"] == "openai-key"
    assert calls["base_url"] == "https://api.openai.com/v1"
    assert calls["kwargs"]["model"] == "gpt-5.2"
    assert calls["kwargs"]["stream"] is True
    assert calls["kwargs"]["tools"][0]["type"] == "web_search"
    assert request_ctx.pending_events[0]["provider"] == "openai"
    assert request_ctx.pending_events[0]["model"] == "openai/gpt-5.2"


def test_web_search_falls_back_to_openrouter_server_tool(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    calls: dict = {}

    class _Response:
        def model_dump(self):
            return {"choices": [{"message": {"content": "fresh answer from openrouter"}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7}}

    class _Completions:
        def create(self, **kwargs):
            calls["kwargs"] = kwargs
            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, api_key=None, base_url=None, **kwargs):  # v6.54.3: accepts timeout
            calls["api_key"] = api_key
            calls["base_url"] = base_url
            self.chat = _Chat()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    ctx = types.SimpleNamespace(pending_events=[], emit_progress_fn=MagicMock(), task_metadata={})

    result = json.loads(search_module._web_search(ctx, "latest news", model="openai/gpt-5.5"))

    assert result["answer"] == "fresh answer from openrouter"
    assert result["backend"] == "openrouter_server_tool"
    assert calls["api_key"] == "or-test-key"
    assert calls["base_url"] == "https://openrouter.ai/api/v1"
    assert calls["kwargs"]["tools"][0]["type"] == "openrouter:web_search"
    assert ctx.pending_events[0]["provider"] == "openrouter"


def test_provider_owned_search_recomputes_the_owner_bounded_timeout(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from ouroboros import llm

    captured = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_TIMEOUT_SEC", "480")
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    ctx = types.SimpleNamespace(
        pending_events=[],
        task_metadata={
            "deadline_at": (datetime.now(timezone.utc) + timedelta(seconds=150)).isoformat(),
        },
    )

    def fake_openrouter(**kwargs):
        captured["openrouter"] = kwargs["timeout"]
        message = types.SimpleNamespace(content="openrouter answer")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)

    def fake_anthropic(**kwargs):
        captured["anthropic"] = kwargs["timeout"]
        block = types.SimpleNamespace(type="text", text="anthropic answer")
        return types.SimpleNamespace(content=[block], usage=None)

    monkeypatch.setattr(llm, "openrouter_web_search_server_tool", fake_openrouter)
    monkeypatch.setattr(llm, "anthropic_web_search_server_tool", fake_anthropic)
    assert json.loads(search_module._web_search_openrouter(ctx, "q"))["answer"] == "openrouter answer"
    assert json.loads(search_module._web_search_anthropic(ctx, "q"))["answer"] == "anthropic answer"
    assert 0 < captured["anthropic"] <= captured["openrouter"] <= 31


def test_provider_owned_search_refuses_inside_finalization_reserve(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from ouroboros import llm

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    calls = []

    def fake_openrouter(**_kwargs):
        calls.append(1)
        raise AssertionError("reserve-only owner window must not dispatch search")

    monkeypatch.setattr(llm, "openrouter_web_search_server_tool", fake_openrouter)
    ctx = types.SimpleNamespace(
        pending_events=[], emit_progress_fn=MagicMock(),
        task_metadata={
            "deadline_at": (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat(),
        },
    )

    result = json.loads(search_module._web_search_openrouter(ctx, "q"))

    assert result["reason_code"] == "deadline_exhausted"
    assert calls == []


def test_provider_owned_search_without_deadline_uses_the_configured_transport_cap(monkeypatch):
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_TIMEOUT_SEC", "321")
    ctx = types.SimpleNamespace(task_metadata={})
    assert search_module._web_search_transport_timeout(ctx) == 321


def test_web_search_outer_envelope_covers_configured_paid_cascade(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_TIMEOUT_SEC", "200")
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "10")
    monkeypatch.delenv("OUROBOROS_WEBSEARCH_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "ddgs", None)

    [entry] = [tool for tool in search_module.get_tools() if tool.name == "web_search"]
    assert entry.timeout_sec == 810  # Safe terminal retry + OpenRouter + Anthropic + grace.


def test_pinned_web_search_preserves_the_legacy_outer_floor(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "openrouter")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_TIMEOUT_SEC", "200")
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "10")

    [entry] = [tool for tool in search_module.get_tools() if tool.name == "web_search"]
    assert entry.timeout_sec == 600


def test_web_search_falls_back_to_ddgs(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    class _DDGS:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def text(self, query, max_results=10):
            assert query == "latest docs"
            assert max_results == 10
            return [{"href": "https://example.com", "title": "Example", "body": "Snippet"}]

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=_DDGS))

    result = json.loads(search_module._web_search(types.SimpleNamespace(pending_events=[]), "latest docs"))

    assert result["backend"] == "ddgs"
    assert result["sources"] == [{"url": "https://example.com", "title": "Example", "snippet": "Snippet"}]


def test_web_search_backend_pin_ddgs_forces_retrieval_no_cascade(monkeypatch):
    # All LLM keys present, but the pin forces ddgs (fixed-model "no second LLM" run).
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "ddgs")

    def _boom(*_a, **_k):  # any LLM backend touched => the pin leaked
        raise AssertionError("pinned ddgs must not cascade to an LLM backend")

    monkeypatch.setattr(search_module, "_web_search_openrouter", _boom)
    monkeypatch.setattr(search_module, "_web_search_anthropic", _boom)
    monkeypatch.setattr(search_module, "_web_search_ddgs", lambda q: json.dumps({"backend": "ddgs", "sources": []}))

    result = json.loads(_web_search(types.SimpleNamespace(pending_events=[]), "q"))
    assert result["backend"] == "ddgs"


def test_web_search_backend_pin_openai_hard_fails_no_cascade(monkeypatch):
    # 'openai' is a TRUE pin: with no OpenAI key it must NOT fall back to other backends.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "openai")

    def _boom(*_a, **_k):
        raise AssertionError("pinned openai must not cascade to another backend")

    monkeypatch.setattr(search_module, "_web_search_openrouter", _boom)
    monkeypatch.setattr(search_module, "_web_search_anthropic", _boom)
    monkeypatch.setattr(search_module, "_web_search_ddgs", _boom)

    result = json.loads(_web_search(types.SimpleNamespace(pending_events=[]), "q"))
    assert result["backend"] == "openai"
    assert "openai" in result["error"].lower()


# ---------------------------------------------------------------------------
# Streaming behavior
# ---------------------------------------------------------------------------


def test_streaming_emits_progress_on_search(ctx, patch_env, mock_openai):
    events = [
        _make_event("response.web_search_call.in_progress", item_id="ws1", output_index=0, sequence_number=1),
        _make_event("response.web_search_call.searching", item_id="ws1", output_index=0, sequence_number=2),
        _make_event("response.output_text.delta", delta="Hello ", content_index=0,
                    item_id="m1", output_index=1, sequence_number=3, logprobs=[]),
        _make_event("response.output_text.delta", delta="world", content_index=0,
                    item_id="m1", output_index=1, sequence_number=4, logprobs=[]),
        _make_completed_event(200, 80),
    ]
    order = []
    ctx.emit_progress_fn.side_effect = lambda text: order.append(text)

    def create(**kwargs):
        order.append("<dispatch>")
        return _FakeStream(events)

    mock_openai.responses.create.side_effect = create

    result = _web_search(ctx, "test query")

    # The attempt is said before dispatch; only the stream's own web_search_call is search evidence.
    assert order == ["🔍 Trying OpenAI web search: test query", "<dispatch>", "🔍 Searching: test query"]

    data = json.loads(result)
    assert data["answer"] == "Hello world"

    mock_openai.responses.create.assert_called_once()
    call_kwargs = mock_openai.responses.create.call_args[1]
    assert call_kwargs["stream"] is True


def test_streaming_direct_openai_cost_remains_nullable(ctx, patch_env, mock_openai):
    events = [
        _make_event("response.output_text.delta", delta="Answer", content_index=0,
                    item_id="m1", output_index=0, sequence_number=1, logprobs=[]),
        _make_completed_event(500, 100),
    ]
    mock_openai.responses.create.return_value = _FakeStream(events)

    _web_search(ctx, "cost test")

    assert len(ctx.pending_events) == 1
    ev = ctx.pending_events[0]
    assert ev["type"] == "llm_usage"
    assert ev["prompt_tokens"] == 500
    assert ev["completion_tokens"] == 100
    assert ev["model_category"] == "websearch"
    assert ev["task_id"] == "task-web"
    assert ev["root_task_id"] == "root-web"
    assert ev["parent_task_id"] == "parent-web"
    assert ev["delegation_role"] == "subagent"
    assert ev["source"] == "web_search"
    assert ev["cost"] is None


def test_streaming_returns_cited_sources(ctx, patch_env, mock_openai):
    class _Usage:
        def model_dump(self):
            return {"input_tokens": 50, "output_tokens": 10}

    class _CompletedResponse:
        usage = _Usage()

        def model_dump(self):
            return {
                "output": [{
                    "content": [{
                        "type": "output_text",
                        "annotations": [{
                            "type": "url_citation",
                            "url": "https://example.com/article",
                            "title": "Example Article",
                            "snippet": "Short source summary",
                        }],
                    }],
                }]
            }

    events = [
        _make_event("response.output_text.delta", delta="Answer", content_index=0,
                    item_id="m1", output_index=0, sequence_number=1, logprobs=[]),
        _make_event("response.completed", response=_CompletedResponse(), sequence_number=2),
    ]
    mock_openai.responses.create.return_value = _FakeStream(events)

    result = json.loads(_web_search(ctx, "source test"))

    assert result["answer"] == "Answer"
    assert result["sources"] == [{
        "url": "https://example.com/article",
        "title": "Example Article",
        "snippet": "Short source summary",
    }]


def test_streaming_sanitizes_progress_and_cited_sources(ctx, patch_env, mock_openai):
    leaked_secret = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"

    class _Usage:
        def model_dump(self):
            return {"input_tokens": 50, "output_tokens": 10}

    class _CompletedResponse:
        usage = _Usage()

        def model_dump(self):
            return {
                "output": [{
                    "content": [{
                        "type": "output_text",
                        "annotations": [{
                            "type": "url_citation",
                            "url": f"https://user:{leaked_secret}@example.com/article?token={leaked_secret}",
                            "title": f"Leaked {leaked_secret}",
                            "snippet": f"Snippet {leaked_secret}",
                        }],
                    }],
                }]
            }

    events = [
        _make_event("response.web_search_call.searching", item_id="ws1", output_index=0, sequence_number=1),
        _make_event("response.output_text.delta", delta="Answer", content_index=0,
                    item_id="m1", output_index=0, sequence_number=2, logprobs=[]),
        _make_event("response.completed", response=_CompletedResponse(), sequence_number=3),
    ]
    mock_openai.responses.create.return_value = _FakeStream(events)

    result = json.loads(_web_search(ctx, f"query {leaked_secret}"))

    rows = [call.args[0] for call in ctx.emit_progress_fn.call_args_list]
    serialized = json.dumps(result)
    assert [row.split(":")[0] for row in rows] == ["🔍 Trying OpenAI web search", "🔍 Searching"]
    assert all(leaked_secret not in row and "***REDACTED***" in row for row in rows)
    assert leaked_secret not in serialized
    assert "***REDACTED***" in serialized


def test_web_search_sanitizes_provider_errors(ctx, patch_env, monkeypatch):
    leaked_secret = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "ddgs", None)

    class _Responses:
        def create(self, **_kwargs):
            raise RuntimeError(f"provider rejected Authorization: Bearer {leaked_secret}")

    class _Client:
        def __init__(self, api_key=None, base_url=None, **kwargs):  # v6.54.3: accepts timeout
            self.responses = _Responses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))

    result = json.loads(_web_search(ctx, "error query"))

    assert result["reason_code"] == "provider_outcome_unknown"
    serialized = json.dumps(result)
    assert leaked_secret not in serialized
    assert result["backend"] == "openai_responses"


def test_dispatched_web_timeout_never_retries_or_cascades(ctx, patch_env, monkeypatch):
    calls = {"openai": 0, "fallback": 0}

    class _Responses:
        def create(self, **_kwargs):
            calls["openai"] += 1
            raise TimeoutError("read timed out after dispatch")

    class _Client:
        def __init__(self, **_kwargs):
            self.responses = _Responses()

    def forbidden(*_args, **_kwargs):
        calls["fallback"] += 1
        raise AssertionError("ambiguous work must not start another backend")

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    monkeypatch.setattr(search_module, "_web_search_openrouter", forbidden)
    monkeypatch.setattr(search_module, "_web_search_anthropic", forbidden)
    monkeypatch.setattr(search_module, "_web_search_ddgs", forbidden)

    result = json.loads(_web_search(ctx, "error query"))

    assert result["reason_code"] == "provider_outcome_unknown"
    assert calls == {"openai": 1, "fallback": 0}


def test_connect_phase_failure_releases_attempt_and_uses_fallback(ctx, patch_env, monkeypatch):
    import httpx

    calls = {"openai": 0, "fallback": 0}

    class _Responses:
        def create(self, **_kwargs):
            calls["openai"] += 1
            raise httpx.ConnectError("connection refused")

    class _Client:
        def __init__(self, **_kwargs):
            self.responses = _Responses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setattr(
        search_module,
        "_web_search_openrouter",
        lambda *_args, **_kwargs: calls.__setitem__("fallback", calls["fallback"] + 1)
        or json.dumps({"answer": "fallback", "backend": "openrouter_server_tool"}),
    )

    result = json.loads(_web_search(ctx, "connect failure"))

    assert result["answer"] == "fallback"
    assert calls == {"openai": 1, "fallback": 1}


def test_fallback_read_timeout_keeps_its_dispatched_attempt_unresolved(
    ctx, patch_env, monkeypatch, tmp_path,
):
    """An exception raised in a prior-leg handler inherits that leg as context."""
    import httpx
    from ouroboros import usage_accounting as ua

    class _Responses:
        def create(self, **_kwargs):
            raise httpx.ConnectError("first leg refused")

    class _Client:
        def __init__(self, **_kwargs):
            self.responses = _Responses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "ddgs", None)

    def fallback(_ctx, *_args, **_kwargs):
        request = ua.AttemptRequest(
            model="openai/fallback", provider="openrouter", reservation_usd=1.0,
            drive_root=tmp_path, task_id="fallback", root_task_id="fallback",
            source="test.web_search",
        )

        def send():
            raise httpx.ReadTimeout("second leg timed out after dispatch")

        return ua.execute_physical_attempt(request, send)

    monkeypatch.setattr(search_module, "_web_search_openrouter", fallback)
    ctx.task_metadata["budget_drive_root"] = str(tmp_path)
    result = json.loads(_web_search(ctx, "fallback timeout"))

    assert result["reason_code"] == "provider_outcome_unknown"
    rows = ledger_rows(tmp_path)
    final_states = {}
    for row in rows:
        final_states[row["attempt_id"]] = row["state"]
    assert list(final_states.values()) == ["released", "unresolved"]


def test_spent_owner_deadline_does_not_start_web_search(ctx, patch_env, monkeypatch):
    calls = []

    class _Client:
        def __init__(self, **_kwargs):
            calls.append(_kwargs)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    ctx.task_metadata = {"deadline_at": "2000-01-01T00:00:00Z"}

    result = json.loads(_web_search(ctx, "expired query"))

    assert result["reason_code"] == "deadline_exhausted"
    assert calls == []


def test_web_search_rechecks_owner_deadline_after_client_preparation(
    ctx, patch_env, monkeypatch, tmp_path,
):
    import ouroboros.deadline_utils as deadlines

    clock = [100.0]
    monkeypatch.setattr(deadlines.time, "time", lambda: clock[0])
    ctx.deadline_ts = 100.5
    ctx.task_metadata = {
        "budget_drive_root": str(tmp_path),
        "root_task_id": "root-web-prep",
        "parent_task_id": "parent-web-prep",
    }
    calls = []

    class _Responses:
        def create(self, **_kwargs):
            calls.append("provider")
            raise AssertionError("expired preparation must not dispatch search")

    class _Client:
        def __init__(self, **_kwargs):
            # Client construction stands in for a slow setup phase.
            clock[0] = 101.0
            self.responses = _Responses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))

    result = json.loads(_web_search(ctx, "expired after setup"))

    assert result["reason_code"] == "deadline_exhausted"
    assert calls == []


def _captured_provider_error(*, provider="openrouter", status=None, error_type=""):
    from ouroboros.usage_accounting import PhysicalAttemptCapture

    exc = RuntimeError("provider transport ended")
    exc.physical_attempt_capture = PhysicalAttemptCapture(
        attempt_id="attempt-1",
        model="model-1",
        provider=provider,
        state="unresolved",
        candidate_measurement_kind="opaque",
        provider_status_code=status,
        provider_error_type=error_type,
    )
    return exc


@pytest.mark.parametrize("status", [None, 503])
def test_openrouter_ambiguous_outcome_stops_cross_provider_cascade(ctx, monkeypatch, status):
    from ouroboros import llm

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.delenv("OUROBOROS_WEBSEARCH_BACKEND", raising=False)
    monkeypatch.setattr(
        llm, "openrouter_web_search_server_tool",
        lambda **_kwargs: (_ for _ in ()).throw(_captured_provider_error(status=status)),
    )
    monkeypatch.setattr(
        search_module, "_web_search_anthropic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ambiguous OpenRouter work must stop the cascade")
        ),
    )
    monkeypatch.setattr(
        search_module, "_web_search_ddgs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ambiguous OpenRouter work must stop the cascade")
        ),
    )

    result = json.loads(_web_search(ctx, "ambiguous OpenRouter query"))

    assert result["backend"] == "openrouter_server_tool"
    assert result["reason_code"] == "provider_outcome_unknown"


def test_r3_openrouter_error_finish_is_not_a_search_summary(ctx, monkeypatch):
    from ouroboros import llm

    body = {"choices": [{"message": {"content": "partial"}, "finish_reason": "error"}], "usage": {"cost": 0}}
    monkeypatch.setattr(search_module, "_search_target", lambda *_args: {"api_key": "test-key"})
    monkeypatch.setattr(llm, "openrouter_web_search_server_tool", lambda **_kwargs: body)
    result = json.loads(search_module._web_search_openrouter(ctx, "once", model="openrouter::test-model"))
    assert result["reason_code"] == "provider_outcome_unknown"
    assert "answer" not in result
    assert (ctx.pending_events[-1]["cost"], ctx.pending_events[-1]["cost_final"]) == (None, False)


def test_anthropic_ambiguous_outcome_stops_ddgs_cascade(ctx, monkeypatch):
    from ouroboros import llm

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.delenv("OUROBOROS_WEBSEARCH_BACKEND", raising=False)
    monkeypatch.setattr(
        search_module, "_web_search_openrouter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("pre-dispatch")),
    )
    monkeypatch.setattr(
        llm, "anthropic_web_search_server_tool",
        lambda **_kwargs: (_ for _ in ()).throw(
            _captured_provider_error(provider="anthropic")
        ),
    )
    monkeypatch.setattr(
        search_module, "_web_search_ddgs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ambiguous Anthropic work must stop the cascade")
        ),
    )

    result = json.loads(_web_search(ctx, "ambiguous Anthropic query"))

    assert result["backend"] == "anthropic_server_tool"
    assert result["reason_code"] == "provider_outcome_unknown"


def test_terminal_openrouter_status_allows_next_backend(ctx, monkeypatch):
    from ouroboros import llm

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.delenv("OUROBOROS_WEBSEARCH_BACKEND", raising=False)
    monkeypatch.setattr(
        llm, "openrouter_web_search_server_tool",
        lambda **_kwargs: (_ for _ in ()).throw(
            _captured_provider_error(status=503, error_type="overloaded_error")
        ),
    )
    monkeypatch.setattr(
        search_module, "_web_search_anthropic",
        lambda *_args, **_kwargs: json.dumps({
            "answer": "recovered", "backend": "anthropic_server_tool",
        }),
    )

    result = json.loads(_web_search(ctx, "terminal OpenRouter query"))

    assert result["answer"] == "recovered" and result["backend"] == "anthropic_server_tool"
    assert [leg["outcome"] for leg in result["legs_tried"]] == ["failed", "succeeded"]


_NO_ENDPOINTS = "No endpoints found for vendor/search-model."
_TYPED_404 = {"code": "model_not_found", "type": "invalid_request_error", "message": _NO_ENDPOINTS}


@pytest.mark.parametrize("error,envelope,advances", [
    ({"code": 404, "message": _NO_ENDPOINTS}, {}, False),
    ({"code": 404, "message": _NO_ENDPOINTS, "metadata": {"failed_routing_step": "Filter by Parameters"}}, {}, False),
    ({"code": 404, "message": "No endpoints found matching your data policy."}, {}, False),
    (_TYPED_404, {}, True),
    (_TYPED_404, {"usage": {"cost": 0.25}}, False),
    ({"code": 404, "message": _NO_ENDPOINTS}, {"usage": {"server_tool_use": {"web_search_requests": 1}}}, False),
], ids=["router-number", "router-number-naming-its-routing-step", "router-number-data-policy", "provider-typed",
        "provider-typed-beside-a-charge", "router-number-with-search-work"])
def test_a_404_reaches_another_search_source_only_as_a_typed_refusal_nothing_contradicts(
        ctx, monkeypatch, tmp_path, error, envelope, advances):
    """One OpenRouter search leg answered HTTP 404 through the real attempt ledger, with a second source eligible.

    OpenRouter types its own verdicts by a number and words only (``error.code`` equal to the status, a message,
    optional ``metadata``). The ledger settles the "No endpoints found" 404 free, which is a price; the same words also
    say that a data-policy or provider filter left no endpoint, and the routing step it may name serves the repair of
    that one route (``request_wire_recovery``). None of them is a typed refusal before work, so a second source is
    possible there and not supported: the search is held. With the status, the words and the free settlement all the
    same, the provider's own typed code is what lets Auto try its next source, unless the same reply shows a charge.
    """
    from ouroboros import llm
    from ouroboros import usage_accounting as ua

    class NotFoundError(Exception):  # the SDK's class name, which is no provider's typed word
        status_code = 404

    failure = NotFoundError(f"Error code: 404 - {{'error': {error}}}")
    failure.body = {"error": dict(error), **envelope}

    def send(**_kwargs):
        request = ua.AttemptRequest(
            model="vendor/search-model", provider="openrouter", prompt_tokens_estimate=1000, max_completion_tokens=100,
            reservation_usd=5.0, drive_root=tmp_path, task_id="task-web", root_task_id="root-web", global_limit_usd=100.0)
        return ua.execute_physical_attempt(request, lambda: (_ for _ in ()).throw(failure))

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.delenv("OUROBOROS_WEBSEARCH_BACKEND", raising=False)
    monkeypatch.setattr(llm, "openrouter_web_search_server_tool", send)
    another = MagicMock(return_value=json.dumps({"answer": "another source", "backend": "anthropic_server_tool"}))
    monkeypatch.setattr(search_module, "_web_search_anthropic", another)
    monkeypatch.setattr(search_module, "_web_search_ddgs", lambda *_args, **_kwargs: pytest.fail("no scraper leg"))

    result = json.loads(_web_search(ctx, "one search"))

    [row] = ledger_rows(tmp_path)
    assert result["legs_tried"][0]["ledger_attempt_ids"] == [row["attempt_id"]]
    if not envelope:  # the ledger says the same of all four: free, by the router's status and words
        assert (row["state"], row["settle_reason"], row["cost_usd"], row["cost_final"]) == (
            "settled", "pre_routing_rejection", 0, True)
    elif "server_tool_use" in envelope["usage"]:
        assert (row["state"], row.get("cost_usd"), bool(row.get("cost_final"))) == ("unresolved", None, False)
    assert another.called is advances
    if advances:
        assert result["answer"] == "another source" and result["backend"] == "anthropic_server_tool"
        assert [leg["outcome"] for leg in result["legs_tried"]] == ["failed", "succeeded"]
    else:
        assert result["reason_code"] == "provider_outcome_unknown" and result["backend"] == "openrouter_server_tool"
        assert [leg["outcome"] for leg in result["legs_tried"]] == ["provider_outcome_unknown"]


@pytest.mark.parametrize("status,body", [
    (408, None), (429, "gateway unavailable"), (503, {"error": {"code": 503}}),
    (502, {"error": {"type": "server_error"}}),
])
def test_opaque_http_failure_after_dispatch_holds_one_search(ctx, patch_env, mock_openai, monkeypatch, tmp_path,
                                                            status, body):
    ctx.task_metadata["budget_drive_root"] = str(tmp_path)
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "auto")
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-fallback")
    monkeypatch.setattr(search_module, "_web_search_openrouter", lambda *a, **k: pytest.fail("duplicate search"))
    error = RuntimeError("indeterminate upstream reply")
    error.status_code, error.body = status, body
    mock_openai.responses.create.side_effect = error
    result = json.loads(_web_search(ctx, "query"))
    mock_openai.responses.create.assert_called_once()
    [row] = ledger_rows(tmp_path)
    [leg] = result["legs_tried"]
    assert row["state"] == "unresolved" and leg["ledger_attempt_ids"] == [row["attempt_id"]]
    assert leg["outcome"] == result["reason_code"] == "provider_outcome_unknown"


@pytest.mark.parametrize("connect_timeout", [False, True], ids=["typed-rate-limit", "unsent-timeout"])
def test_known_search_outcome_allows_one_safe_retry(ctx, patch_env, monkeypatch, tmp_path, connect_timeout):
    import httpx

    ctx.task_metadata["budget_drive_root"] = str(tmp_path)
    calls = 0

    class _RateLimited(Exception):
        status_code = 429
        body = {"error": {"code": "rate_limit_exceeded"}, "usage": {"cost": 0, "output_tokens": 0}}

    class _Responses:
        def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                if connect_timeout:
                    raise httpx.ConnectTimeout("no connection")
                raise _RateLimited("rate limit")
            return _FakeStream([
                _make_event("response.output_text.delta", delta="recovered"),
                _make_completed_event(),
            ])

    class _Client:
        def __init__(self, **_kwargs):
            self.responses = _Responses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    result = json.loads(_web_search(ctx, "retry query"))

    assert calls == 2
    assert result["answer"] == "recovered"
    rows = ledger_rows(tmp_path)
    assert [row["state"] for row in rows] == ["released" if connect_timeout else "settled", "settled"]
    if not connect_timeout:
        assert (rows[0]["cost_usd"], rows[0]["cost_final"]) == (0, True)
    assert result["legs_tried"][0]["ledger_attempt_ids"] == [row["attempt_id"] for row in rows]


def test_streamed_typed_overload_before_work_allows_one_retry(ctx, patch_env, monkeypatch):
    calls = 0

    class _Responses:
        def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _FakeStream([
                    _make_event(
                        "response.failed",
                        response=types.SimpleNamespace(error=types.SimpleNamespace(
                            message="service unavailable", status_code=503, type="overloaded_error")),
                    )
                ])
            return _FakeStream([
                _make_event("response.output_text.delta", delta="recovered from event"),
                _make_completed_event(),
            ])

    class _Client:
        def __init__(self, **_kwargs):
            self.responses = _Responses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    result = json.loads(_web_search(ctx, "retry streamed failure"))

    assert calls == 2
    assert result["answer"] == "recovered from event"


def test_streaming_no_progress_without_search_events(ctx, patch_env, mock_openai):
    events = [
        _make_event("response.output_text.delta", delta="Direct answer", content_index=0,
                    item_id="m1", output_index=0, sequence_number=1, logprobs=[]),
        _make_completed_event(50, 20),
    ]
    mock_openai.responses.create.return_value = _FakeStream(events)

    result = _web_search(ctx, "simple query")

    # An attempt without a web_search_call event never claims that a search ran.
    assert [call.args[0] for call in ctx.emit_progress_fn.call_args_list] == ["🔍 Trying OpenAI web search: simple query"]
    data = json.loads(result)
    assert data["answer"] == "Direct answer"


@pytest.mark.parametrize("has_fallback", [False, True])
def test_streaming_empty_text_engages_cascade(ctx, patch_env, mock_openai, monkeypatch, has_fallback):
    """NW-11: a confirmed empty completion is a soft failure, not a fabricated answer or a same-leg repeat."""
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "auto")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_MODEL", "")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fallback = MagicMock(return_value=json.dumps({"answer": "another source"}))
    monkeypatch.setattr(search_module, "_web_search_openrouter", fallback)
    if has_fallback:
        monkeypatch.setenv("OPENROUTER_API_KEY", "fallback")
    else:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    events = [_make_completed_event(10, 0)]
    mock_openai.responses.create.return_value = _FakeStream(events)

    data = json.loads(_web_search(ctx, "empty query"))
    mock_openai.responses.create.assert_called_once()
    assert any("no answer and no sources" in e for e in data.get("backend_errors", []))
    if has_fallback:
        fallback.assert_called_once()
        assert data["answer"] == "another source"
        assert [(leg["source"], leg["outcome"]) for leg in data["legs_tried"]] == [
            ("openai", "failed"), ("openrouter", "succeeded")]
    else:
        fallback.assert_not_called()
        assert "error" in data and "answer" not in data
        assert len(data["legs_tried"]) == 1


def test_streaming_progress_fires_only_once(ctx, patch_env, mock_openai):
    events = [
        _make_event("response.web_search_call.in_progress", item_id="ws1", output_index=0, sequence_number=1),
        _make_event("response.web_search_call.searching", item_id="ws1", output_index=0, sequence_number=2),
        _make_event("response.web_search_call.searching", item_id="ws1", output_index=0, sequence_number=3),
        _make_event("response.output_text.delta", delta="Result", content_index=0,
                    item_id="m1", output_index=1, sequence_number=4, logprobs=[]),
        _make_completed_event(100, 50),
    ]
    mock_openai.responses.create.return_value = _FakeStream(events)

    _web_search(ctx, "multi-search query")

    assert [call.args[0] for call in ctx.emit_progress_fn.call_args_list] == [
        "🔍 Trying OpenAI web search: multi-search query", "🔍 Searching: multi-search query"]


def test_direct_search_dispatch_failure_releases_same_unsent_reservation(ctx, patch_env, mock_openai, monkeypatch):
    from ouroboros import usage_accounting as ua
    from ouroboros.usage_ledger import UsageLockUnavailable
    reservations = []
    original = search_module.reserve_attempt
    def reserve(request):
        result = original(request)
        reservations.append(result)
        return result
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "openai")
    monkeypatch.setattr(search_module, "reserve_attempt", reserve)
    def refused(*args, **kwargs):
        raise UsageLockUnavailable("synthetic kernel failure", reason="kernel_refused")
    monkeypatch.setattr(search_module, "mark_dispatched", refused)
    with pytest.raises(ua.PhysicalAttemptPreparationFailed) as error:
        _web_search(ctx, "one query")
    assert error.value.physical_attempt_capture.state == "released"
    mock_openai.responses.create.assert_not_called()
    assert len(reservations) == 1
    with ua._locked(reservations[0].drive_root):
        records = ledger_rows(reservations[0].drive_root)
    chain = [row for row in records if row["attempt_id"] == reservations[0].attempt_id]
    assert [row["state"] for row in chain] == ["released"]


def test_direct_search_failed_cleanup_retains_reservation_and_refuses_fallback(ctx, patch_env, mock_openai, monkeypatch):
    from ouroboros import usage_accounting as ua
    original = search_module._responses_search_candidate
    def candidate(*args, **kwargs):
        target, payload, request, before = original(*args, **kwargs)
        def fail(held):
            raise RuntimeError("synthetic preparation failure")
        return target, payload, request, fail
    monkeypatch.setattr(search_module, "_responses_search_candidate", candidate)
    def release(*args, **kwargs):
        raise ua.UsageLockUnavailable("synthetic failed cleanup", reason="contention")
    monkeypatch.setattr(ua, "release_attempt", release)
    monkeypatch.setattr(search_module, "_web_search_openrouter", lambda *a, **k: pytest.fail("competing fallback"))
    monkeypatch.setattr(search_module, "_web_search_anthropic", lambda *a, **k: pytest.fail("competing fallback"))
    with pytest.raises(ua.PhysicalAttemptPreparationFailed) as error:
        _web_search(ctx, "one query")
    assert error.value.physical_attempt_capture.state == "reserved"
    mock_openai.responses.create.assert_not_called()


# ---------------------------------------------------------------------------
# DeepSeek Search: an explicit Source only (owner decision, 11 Nov 2026). The real leg,
# the installed OpenAI SDK and the real ledger; only the HTTP transport under them is in memory.
# ---------------------------------------------------------------------------

_DEEPSEEK_ITEM = {"title": "Gates", "url": "https://example.invalid/gates", "snippet": "A short line.",
                  "summary": "The provider's own summary of the page.", "content": "", "site_name": "Example",
                  "site_icon": "https://example.invalid/icon.png", "date": "2026-11-01T00:00:00Z", "authority": 3,
                  "rerank_score": 2.1}


@pytest.fixture
def deepseek(monkeypatch, tmp_path):
    from ouroboros import net_transport
    from ouroboros import usage_accounting as ua

    wire = types.SimpleNamespace(root=tmp_path, sent=[], progress=[],
                                 answer=lambda _request: httpx.Response(200, json={"results": [_DEEPSEEK_ITEM]}))

    def respond(request):
        wire.sent.append(request)
        return wire.answer(request)

    monkeypatch.delenv("OUROBOROS_CONNECTIONS", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-deepseek")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "deepseek")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_MODEL", "")
    # A host proxy would make the SDK build its own transport, past this one, onto the network.
    monkeypatch.setattr(net_transport, "env_proxies_configured", lambda: False)
    monkeypatch.setattr(net_transport, "remote_httpx_transport", lambda *a, **k: httpx.MockTransport(respond))
    wire.ctx = types.SimpleNamespace(task_id="search", task_metadata={}, drive_root=tmp_path, pending_events=[],
                                     emit_progress_fn=wire.progress.append)
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="search", root_task_id="search")):
        yield wire


@pytest.mark.parametrize("query", [
    "q" * 128,  # the documented limit, inclusive
    "я" * 128,  # characters, not bytes: 256 bytes of UTF-8
    "😀" * 128,  # code points, not UTF-16 units: 512 bytes of UTF-8 (the live API is unchecked with no key here)
    '  site:example.invalid "exact phrase" -noise  ',  # operators and surrounding spaces travel unchanged
], ids=["ascii_limit", "cyrillic_limit", "astral_limit", "operators_and_spaces"])
def test_deepseek_sends_the_query_as_written_once_and_never_reads_its_unknown_price_as_free(deepseek, query):
    from ouroboros import usage_accounting as ua

    result = json.loads(_web_search(deepseek.ctx, query))
    [request] = deepseek.sent
    assert (request.method, str(request.url)) == ("POST", "https://api.deepseek.com/v1/tools/search")
    assert request.headers["authorization"] == "Bearer synthetic-deepseek"
    assert json.loads(request.content) == {"query": query}  # the provider's own count and need_content defaults
    assert [(leg["source"], leg["outcome"]) for leg in result["legs_tried"]] == [("deepseek", "succeeded")]
    assert result["backend"] == "deepseek_search_tool" and result["answer_type"] == "summary"
    assert result["answer"] == "- Gates: The provider's own summary of the page. (https://example.invalid/gates)"
    assert result["sources"] == [{"url": "https://example.invalid/gates", "title": "Gates", "snippet": "A short line.",
                                  "site_name": "Example", "date": "2026-11-01T00:00:00Z", "authority": 3,
                                  "rerank_score": 2.1}]
    [row] = ledger_rows(deepseek.root)
    assert (row["state"], row["provider"], row["model"], row["source"]) == ("settled", "deepseek", "", "web_search.deepseek")
    # The pin needs the catalog, which this case does not read: the outcome is kept, the amount unknown.
    assert (row["cost_usd"], row["cost_final"]) == (None, False)
    assert row["cost_evidence"] == {"results": 1, "http_status": 200, "outcome": "results"}
    assert result["ledger_attempt_ids"] == [row["attempt_id"]]
    spent = ua.usage_breakdown(deepseek.root)  # the public reader: an unknown amount, never a priced zero
    assert (spent["unknown_unmetered"], spent["priced_rows"], spent["cost_final"]) == (1, 0, False)
    assert deepseek.progress[0].startswith("🔍 Trying DeepSeek web search: ")


@pytest.mark.parametrize("query,chars", [("q" * 129, 129), ("   ", 3), ("", 0)],
                         ids=["one_over_the_limit", "only_whitespace", "empty"])
def test_deepseek_refuses_a_query_its_contract_refuses_before_reserving_or_sending(deepseek, query, chars):
    result = json.loads(_web_search(deepseek.ctx, query))
    assert deepseek.sent == [] and ledger_rows(deepseek.root) == []
    assert (result["reason_code"], result["query_chars"], result["query_max_chars"]) == ("query_limit", chars, 128)
    assert "Nothing was sent" in result["error"] and "answer" not in result
    assert deepseek.progress == ["⚠️ Web search not sent: DeepSeek does not accept this query as written"]


@pytest.mark.parametrize("query,code_point", [("caf\ud800", "U+D800"), ("\udfff", "U+DFFF"),
                                              ("\ud83d\ude00 smile", "U+D83D")],
                         ids=["lone_high", "lone_low", "pair_as_two_code_points"])
def test_deepseek_refuses_a_query_utf8_cannot_carry_before_reserving_or_sending(deepseek, query, code_point):
    result = json.loads(_web_search(deepseek.ctx, query))
    assert deepseek.sent == [] and ledger_rows(deepseek.root) == []  # no claim of a send that never happened
    assert result["reason_code"] == "argument_invalid" and code_point in result["error"]
    assert "Nothing was sent" in result["error"] and "answer" not in result
    assert [(leg["source"], leg["outcome"]) for leg in result["legs_tried"]] == [("deepseek", "argument_invalid")]
    assert deepseek.progress == ["⚠️ Web search not sent: DeepSeek does not accept this query as written"]


def test_deepseek_receives_its_own_key_and_never_the_openai_account_headers(deepseek, monkeypatch):
    monkeypatch.setenv("OPENAI_ORG_ID", "org-synthetic")  # the SDK reads both for any base URL
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_synthetic")
    assert "error" not in json.loads(_web_search(deepseek.ctx, "q"))
    [request] = deepseek.sent
    assert request.headers["authorization"] == "Bearer synthetic-deepseek"
    assert "openai-organization" not in request.headers and "openai-project" not in request.headers


def test_deepseek_without_a_match_is_an_answered_search(deepseek):
    deepseek.answer = lambda _request: httpx.Response(200, json={"results": []})
    result = json.loads(_web_search(deepseek.ctx, "nothing matches this"))
    assert "error" not in result and (result["answer"], result["sources"]) == ("(no matches)", [])
    [row] = ledger_rows(deepseek.root)
    assert (row["state"], row["cost_usd"]) == ("settled", None)
    assert row["cost_evidence"] == {"results": 0, "http_status": 200, "outcome": "empty"}


def _deepseek_tariff(monkeypatch, currency="USD"):
    """The retained DeepSeek catalog entries (tests/test_deepseek_pricing.py) behind the pin, and a scripted read
    of the key's currency; the keys it was asked with are returned."""
    import hashlib
    import pathlib

    from ouroboros import pricing_catalog

    raw = (pathlib.Path(__file__).parent / "fixtures" / "pricing" / "litellm-deepseek-20261115.json").read_text()
    snapshot = pricing_catalog._decode({"url": pricing_catalog.CATALOG_URL, "content": raw, "fetched_at": "",
                                        "sha256": hashlib.sha256(raw.encode()).hexdigest()})
    from ouroboros.utils import utc_now_iso

    snapshot["fetched_at"] = utc_now_iso()
    asked = []
    for name, value in (("_cache", {}), ("_retry_after", {}), ("_currency", {}), ("_currency_fetching", set())):
        monkeypatch.setattr(pricing_catalog, name, value)
    monkeypatch.setattr(pricing_catalog, "_fetch_catalog", lambda: snapshot)
    reading = pricing_catalog._CurrencyReading(currency, "" if currency else "http_401")
    monkeypatch.setattr(pricing_catalog, "_fetch_currency", lambda key: asked.append(key) or reading)
    return asked


@pytest.mark.parametrize("status, results, cost, outcome", [
    (200, [_DEEPSEEK_ITEM], 0.00075, "results"),
    (200, [_DEEPSEEK_ITEM] * 3, 0.00075, "results"),  # one request, however many results
    (200, [], None, "empty"),  # documented as not billed, not yet reconciled: unknown, never $0
    (200, [{}], None, "malformed_results"),  # an object that names no page is not a result
    (200, [_DEEPSEEK_ITEM, {"title": "no url"}], None, "malformed_results"),
    (201, [_DEEPSEEK_ITEM], None, "http_201"),  # outside the documented HTTP 200
], ids=["one_result", "three_results", "no_results", "blank_item", "partly_blank", "http_201"])
def test_a_deepseek_search_prices_one_unit_only_for_a_documented_answer_with_results(deepseek, monkeypatch, status,
                                                                                    results, cost, outcome):
    asked = _deepseek_tariff(monkeypatch)
    deepseek.answer = lambda _request: httpx.Response(status, json={"results": results})
    shown = json.loads(_web_search(deepseek.ctx, "q"))
    assert "error" not in shown and len(shown["sources"]) == len(results), "what is shown does not follow the price"
    [row] = ledger_rows(deepseek.root)
    assert (row["cost_usd"], row["cost_estimated"], row["cost_final"]) == (cost, cost is not None, False)
    assert row["reservation_upper_bound_usd"] is None, "the reservation stays unknown"
    evidence = row["cost_evidence"]
    assert evidence["tariff_quote"]["catalog_id"] == "deepseek/search"
    assert (evidence["http_status"], evidence["results"], evidence["outcome"]) == (status, len(results), outcome)
    assert evidence["tariff_pricing"] == ({"billable_requests": 1} if cost else {"unpriced": f"search_{outcome}"})
    assert row.get("price_source") == (f"litellm@{evidence['tariff_quote']['fetched_at']}" if cost else None)
    assert asked == ["synthetic-deepseek"], "the selected key's own currency, read before the send"


@pytest.mark.parametrize("currency", ["CNY", None])
def test_a_deepseek_search_on_a_key_not_proven_usd_stays_unknown(deepseek, monkeypatch, currency):
    _deepseek_tariff(monkeypatch, currency)
    json.loads(_web_search(deepseek.ctx, "q"))
    [row] = ledger_rows(deepseek.root)
    assert (row["cost_usd"], row["cost_estimated"]) == (None, False)
    assert row["cost_evidence"]["tariff_pricing"]["unpriced"] == ("currency_not_usd" if currency else "currency_unknown")


def _read_timeout(request):
    raise httpx.ReadTimeout("synthetic ReadTimeout", request=request)


@pytest.mark.parametrize("answer", [
    lambda _r: httpx.Response(500, json={"error": {"message": "Internal"}}),
    lambda _r: httpx.Response(400, json={"error": {"message": "Query refused"}}),
    lambda _r: httpx.Response(200, json={"data": []}),
    _read_timeout,
], ids=["server_error", "refused_400", "malformed_200", "read_timeout"])
def test_a_failed_or_unreadable_deepseek_search_is_never_priced(deepseek, monkeypatch, answer):
    _deepseek_tariff(monkeypatch)
    deepseek.answer = answer
    json.loads(_web_search(deepseek.ctx, "q"))
    [row] = ledger_rows(deepseek.root)
    assert (row["state"], row.get("cost_usd"), row.get("cost_estimated")) == ("unresolved", None, None)
    assert "tariff_pricing" not in (row.get("cost_evidence") or {})


def _raises(error):
    def respond(request):
        raise error(f"synthetic {error.__name__}", request=request)
    return respond


@pytest.mark.parametrize("answer,state,outcome,says", [
    (_raises(httpx.ConnectError), "released", "failed", "(APIConnectionError)"),
    (_raises(httpx.ConnectTimeout), "released", "failed", "(APITimeoutError)"),
    (_raises(httpx.ReadTimeout), "unresolved", "provider_outcome_unknown", "provider outcome is unknown"),
    (_raises(httpx.ReadError), "unresolved", "provider_outcome_unknown", "provider outcome is unknown"),
    (lambda _r: httpx.Response(302, headers={"location": "https://elsewhere.invalid/"}), "unresolved", "failed", "302"),
    (lambda _r: httpx.Response(402, json={"error": {"message": "Insufficient Balance"}}), "unresolved", "failed", "402"),
    (lambda _r: httpx.Response(429, json={"error": {"message": "Rate limited"}}), "unresolved", "failed", "429"),
    (lambda _r: httpx.Response(500, json={"error": {"message": "Internal"}}), "unresolved", "failed", "500"),
    (lambda _r: httpx.Response(200, content=b"<html>a proxy page</html>", headers={"content-type": "text/html"}),
     "unresolved", "failed", "without a readable results list"),
    (lambda _r: httpx.Response(200, json={"data": []}), "unresolved", "failed", "without a readable results list"),
    (lambda _r: httpx.Response(200, json={"results": ["text"]}), "unresolved", "failed", "without a readable results list"),
], ids=["connect_error", "connect_timeout", "read_timeout", "reset_after_send", "redirect", "payment", "rate_limit",
        "server_error", "html_200", "no_results_200", "results_not_objects_200"])
def test_a_failed_deepseek_search_keeps_its_cause_and_is_never_sent_again(deepseek, answer, state, outcome, says):
    deepseek.answer = answer
    result = json.loads(_web_search(deepseek.ctx, "q"))
    assert len(deepseek.sent) == 1 and "answer" not in result  # one physical send: no retry, no redirect followed
    [tried] = result["legs_tried"]
    assert tried["outcome"] == outcome and says in result["error"]
    [row] = ledger_rows(deepseek.root)
    assert row["state"] == state and row.get("cost_usd") is None and tried["ledger_attempt_ids"] == [row["attempt_id"]]


def test_a_deepseek_answer_survives_a_failed_settlement_and_its_attempt_stays_open(deepseek, monkeypatch):
    from ouroboros import usage_accounting as ua

    def unwritable(*_args, **_kwargs):
        raise OSError("ledger disk full")

    monkeypatch.setattr(ua, "settle_attempt", unwritable)
    result = json.loads(_web_search(deepseek.ctx, "q"))
    assert "error" not in result and result["sources"][0]["url"] == _DEEPSEEK_ITEM["url"]
    [row] = ledger_rows(deepseek.root)
    assert (row["state"], row.get("cost_usd")) == ("unresolved", None)


def test_a_spent_budget_refuses_deepseek_before_any_send(deepseek, monkeypatch):
    from ouroboros import usage_accounting as ua

    ua.record_received_attempt(ua.AttemptRequest(model="openrouter::spent", provider="openrouter", drive_root=deepseek.root,
                                                 task_id="earlier", root_task_id="earlier"), {}, cost_usd=5.0)
    monkeypatch.setenv("TOTAL_BUDGET", "1")
    with pytest.raises(ua.BudgetExceeded):
        _web_search(deepseek.ctx, "q")
    assert deepseek.sent == [] and [row["provider"] for row in ledger_rows(deepseek.root)] == ["openrouter"]


def test_a_saved_deepseek_head_sends_on_its_named_connection_only(deepseek, monkeypatch):
    monkeypatch.setenv("OUROBOROS_CONNECTIONS", json.dumps([{"kind": "deepseek", "name": "work", "api_key": "work-key"}]))
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_MODEL", "deepseek@work::")
    result = json.loads(_web_search(deepseek.ctx, "q"))
    [request] = deepseek.sent
    assert request.headers["authorization"] == "Bearer work-key"
    assert result["route"]["legs"][0]["connection"] == "work" and "error" not in result


# ---------------------------------------------------------------------------
# One money fact: a paid leg's llm_usage event states what its attempt row settled
# ---------------------------------------------------------------------------
# Only the provider SDK client is faked. The leg, the recovery driver, the attempt ledger, the event and the
# supervisor's llm_usage consumer are the real ones.

_SEARCH_COUNTERS = ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens")
_NATIVE_CACHE = {"input_tokens": 9, "output_tokens": 4, "cache_read_input_tokens": 300, "cache_creation_input_tokens": 0}
_CHAT_USAGE = {"prompt_tokens": 100, "completion_tokens": 20}
_RESPONSES_USAGE = {"input_tokens": 100, "output_tokens": 50, "input_tokens_details": {"cached_tokens": 40}}


def _sdk_object(payload, **attributes):
    """A provider SDK reply: public attributes the leg reads and the ``model_dump`` the ledger reads."""
    return types.SimpleNamespace(model_dump=lambda: json.loads(json.dumps(payload)), **attributes)


@pytest.fixture
def paid_leg(monkeypatch, tmp_path):
    """``run(source, usage, envelope)``: one real ``web_search`` call on a pinned paid Source whose SDK client
    answers with ``usage`` (and reply-level ``envelope`` fields); returns the tool result, the leg's event, the
    attempt row the event names and the supervisor's persisted row."""
    from ouroboros import pricing, usage_store
    from ouroboros import usage_accounting as ua
    from supervisor.events_budget import _handle_llm_usage

    catalog = {"rows": {}}  # no catalog read; a case may price its model here
    monkeypatch.setattr(pricing, "get_pricing", lambda **_: catalog["rows"])
    monkeypatch.delenv("OUROBOROS_CONNECTIONS", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_MODEL", "")
    monkeypatch.setitem(sys.modules, "ddgs", None)
    sent = []

    def run(source, usage, envelope=None):
        envelope = dict(envelope or {})
        sent.clear()
        monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", source)
        monkeypatch.setenv({"openrouter": "OPENROUTER_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
                            "openai": "OPENAI_API_KEY"}[source], "synthetic-key")

        def chat(**kwargs):
            sent.append(kwargs)
            payload = {"choices": [{"message": {"content": "answer"}}], "model": "m", **envelope,
                       **({} if usage is None else {"usage": usage})}
            return _sdk_object(payload, model="m", usage=None if usage is None else _sdk_object(usage),
                               choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="answer"))])

        def messages(**kwargs):
            sent.append(kwargs)
            payload = {"type": "message", "content": [{"type": "text", "text": "answer"}], "model": "m",
                       "usage": usage, **envelope}
            return _sdk_object(payload, model="m", usage=_sdk_object(usage), content=payload["content"])

        def responses(**kwargs):
            sent.append(kwargs)
            payload = {"object": "response", "output": [], "model": "m", "usage": usage, **envelope}
            return _FakeStream([_make_event("response.output_text.delta", delta="answer"),
                                _make_event("response.completed", response=_sdk_object(
                                    payload, model="m", usage=_sdk_object(usage)))])

        class _OpenAI:
            def __init__(self, **_kwargs):
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=chat))
                self.responses = types.SimpleNamespace(create=responses)

        class _Anthropic:
            def __init__(self, **_kwargs):
                self.messages = types.SimpleNamespace(create=messages)

        monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))
        monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Anthropic))
        ctx = types.SimpleNamespace(task_id="search", task_metadata={}, drive_root=tmp_path, pending_events=[],
                                    emit_progress_fn=None)
        with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="search", root_task_id="search")):
            result = json.loads(_web_search(ctx, "one query"))
        assert len(sent) == 1 and result.get("answer") == "answer", result
        [event] = [item for item in ctx.pending_events if item["type"] == "llm_usage"]
        assert event["ledger_attempt_ids"] and event["ledger_attempt_ids"] == result["ledger_attempt_ids"]
        with usage_store.read(tmp_path) as txn:
            row = txn.attempt(event["ledger_attempt_ids"][-1])
        logged = []
        _handle_llm_usage(event, types.SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=types.SimpleNamespace(
            push_log=logged.append)))
        [persisted] = logged
        return result, event, row, persisted

    run.catalog = catalog
    return run


def _one_fact(event, row, persisted, cost, final, estimated=False):
    """The event, the row its attempt id names and the supervisor's row state one amount and one finality."""
    assert row is not None and row["state"] == "settled"
    for holder, amount in ((row, row["cost_usd"]), (event, event["cost"]), (persisted, persisted["cost"])):
        assert amount == (None if cost is None else pytest.approx(cost, rel=1e-12, abs=0)), holder
    assert row["cost_final"] is event["cost_final"] is persisted["cost_final"] is final
    assert bool(row["cost_estimated"]) is persisted["cost_estimated"] is estimated
    assert persisted["cost_known"] is (cost is not None)
    # The event's usage copy, which the supervisor reads first, states the same fact.
    assert event["usage"]["cost"] == event["cost"]
    assert (event["usage"]["cost_final"], event["usage"]["cost_estimated"]) == (final, estimated)
    # Whether the bill was unreadable is the row's own fact too, never the reply's label.
    unreadable = row["cost_usd"] is None and bool(row.get("cost_invalid"))
    assert bool(event["usage"].get("cost_invalid")) is persisted["cost_invalid"] is unreadable
    for key in _SEARCH_COUNTERS:  # the event's four counts are the row's own: a measured zero and an unknown too
        assert event[key] == row[key] == persisted[key], key


@pytest.mark.parametrize("source,usage,envelope,cost,final", [
    pytest.param("openrouter", _CHAT_USAGE, {}, None, False, id="openrouter-unknown"),
    pytest.param("openrouter", None, {}, None, False, id="openrouter-no-usage-block"),
    pytest.param("openrouter", {"prompt_tokens": "bad", "completion_tokens": None}, {}, None, False,
                 id="openrouter-malformed-counts-no-bill"),
    pytest.param("openrouter", {**_CHAT_USAGE, "cost": "NaN"}, {}, None, False, id="openrouter-unreadable-bill"),
    pytest.param("openrouter", {**_CHAT_USAGE, "cost": 0}, {}, 0.0, True, id="openrouter-reported-zero"),
    pytest.param("openrouter", {"cost": 0}, {}, 0.0, True, id="openrouter-reported-zero-without-counts"),
    pytest.param("openrouter", {**_CHAT_USAGE, "cost": 0.031}, {}, 0.031, True, id="openrouter-reported-charge"),
    pytest.param("openrouter", {"cost": 0.031}, {}, 0.031, True, id="openrouter-reported-charge-without-counts"),
    pytest.param("openrouter", {**_CHAT_USAGE, "cost": 0, "is_byok": True, "cost_details": {}}, {}, None, False,
                 id="openrouter-byok-gateway-fee-alone"),
    pytest.param("openrouter", {"cost": 0, "is_byok": True}, {}, None, False,
                 id="openrouter-byok-gateway-fee-alone-without-counts"),
    pytest.param("openrouter", {**_CHAT_USAGE, "cost": 0.001, "is_byok": True,
                                "cost_details": {"upstream_inference_cost": 0.03}}, {}, 0.031, True,
                 id="openrouter-byok-both-shares"),
    pytest.param("openrouter", {"prompt_tokens": 100, "completion_tokens": "abc", "cost": 0.031}, {}, 0.031, True,
                 id="openrouter-malformed-output-reported-charge"),
    pytest.param("anthropic", _NATIVE_CACHE, {}, None, False, id="anthropic-native-cache-unknown"),
    pytest.param("anthropic", _NATIVE_CACHE, {"total_cost_usd": 0}, 0.0, True, id="anthropic-reported-zero"),
    pytest.param("anthropic", _NATIVE_CACHE, {"total_cost_usd": 0.031}, 0.031, True, id="anthropic-reported-charge"),
    pytest.param("anthropic", {"output_tokens": 4, "cache_read_input_tokens": 300}, {"total_cost_usd": 0.031},
                 0.031, True, id="anthropic-no-fresh-input-reported-charge"),
    pytest.param("openai", _RESPONSES_USAGE, {}, None, False, id="responses-unknown"),
    pytest.param("openai", _RESPONSES_USAGE, {"total_cost_usd": 0}, 0.0, True, id="responses-reported-zero"),
    pytest.param("openai", _RESPONSES_USAGE, {"total_cost_usd": 0.031}, 0.031, True, id="responses-reported-charge"),
    pytest.param("openai", {"input_tokens": "abc", "output_tokens": 50}, {"total_cost_usd": 0.031}, 0.031, True,
                 id="responses-malformed-input-reported-charge"),
])
def test_a_paid_search_event_states_the_amount_its_attempt_row_settled(paid_leg, source, usage, envelope, cost, final):
    """Without a provider amount the event states none and is not final; a reported $0 or $0.031 keeps its exact
    finality; half a BYOK price is unknown, never a proven zero; a malformed count costs neither the answer nor
    the reported bill. The search side prices nothing itself on any of the three paid legs."""
    result, event, row, persisted = paid_leg(source, usage, envelope)
    _one_fact(event, row, persisted, cost, final)
    assert row["source"] == {"openrouter": "web_search.openrouter", "anthropic": "web_search.anthropic",
                             "openai": "web_search.openai_responses"}[source]


_FRESH_ZERO = {"input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 5}
_ONE_BIN = {**_FRESH_ZERO, "cache_creation": {"ephemeral_1h_input_tokens": 5}}


@pytest.mark.parametrize("source,usage,envelope,counts,cost,final", [
    pytest.param("anthropic", _NATIVE_CACHE, {}, [309, 4, 300, 0], None, False, id="anthropic-aggregate"),
    pytest.param("openai", _RESPONSES_USAGE, {}, [100, 50, 40, None], None, False, id="responses-cached-input"),
    pytest.param("anthropic", {**_FRESH_ZERO, "cache_creation": {"ephemeral_5m_input_tokens": 3,
                                                                 "ephemeral_1h_input_tokens": 5}},
                 {}, [8, 5, 0, 8], None, False, id="anthropic-both-bins-reported-zero-fresh-input"),
    pytest.param("anthropic", _ONE_BIN, {"total_cost_usd": 0}, [None, 5, 0, None], 0.0, True,
                 id="anthropic-one-bin-reported-zero"),
    pytest.param("anthropic", _ONE_BIN, {"total_cost_usd": 0.031}, [None, 5, 0, None], 0.031, True,
                 id="anthropic-one-bin-reported-charge"),
])
def test_a_native_search_reply_states_the_rows_own_counts(paid_leg, source, usage, envelope, counts, cost, final):
    """The event, the settled row its attempt id names and the supervisor's row hold the same four counts, a
    measured zero and an unknown included. 0 fresh and 0 read tokens with write bins of 3 and 5 and no aggregate
    measure a prompt of 8 with 8 written. One bin alone is neither a write total nor a prompt, and a reported $0
    or $0.031 beside it stays exact and final."""
    _result, event, row, persisted = paid_leg(source, usage, envelope)
    _one_fact(event, row, persisted, cost, final)
    for holder in (row, event, persisted):
        assert [holder[key] for key in _SEARCH_COUNTERS] == counts


@pytest.mark.parametrize("source,usage,estimate", [
    pytest.param("openrouter", _CHAT_USAGE, None, id="openrouter-add-on-unpriced"),
    pytest.param("anthropic", _NATIVE_CACHE, 9 * 1.0 + 300 * 0.5 + 4 * 2.0, id="anthropic-native-cache"),
    pytest.param("openai", _RESPONSES_USAGE, 60 * 1.0 + 40 * 0.5 + 50 * 2.0, id="responses-cached-input"),
])
def test_a_search_event_carries_the_ledgers_own_estimate_as_an_estimate(paid_leg, source, usage, estimate):
    """With a tariff of $1/M input, $0.5/M cache reads and $2/M output the attempt row holds the ledger's own
    non-final estimate over the inclusive prompt and its cache reads. The event states that very amount as an
    estimate: never a second price over the reply's exclusive text counts, and never as final. A call with an
    unpriced add-on has no estimate in either reader."""
    from ouroboros import pricing

    class _EveryModel(dict):
        def get(self, _key, _default=None):
            return pricing.PricingSchedule((1.0, 0.5, 3.0, 2.0))

    paid_leg.catalog["rows"] = _EveryModel()
    _result, event, row, persisted = paid_leg(source, usage)
    expected = None if estimate is None else estimate / 1_000_000
    _one_fact(event, row, persisted, expected, False, estimated=estimate is not None)


_LABELLED = {"cost_invalid": True}  # a reply's own claim about its bill: no host fact


@pytest.mark.parametrize("source,usage,envelope,unreadable,cost", [
    pytest.param("anthropic", _NATIVE_CACHE, {"total_cost_usd": "NaN"}, True, None, id="anthropic-unreadable-bill"),
    pytest.param("openrouter", {**_CHAT_USAGE, "cost": "NaN"}, {}, True, None, id="openrouter-unreadable-bill"),
    pytest.param("openai", _RESPONSES_USAGE, {"total_cost_usd": "NaN"}, True, None, id="responses-unreadable-bill"),
    pytest.param("anthropic", {**_NATIVE_CACHE, **_LABELLED}, {}, False, None, id="anthropic-label-without-a-bill"),
    pytest.param("openrouter", {**_CHAT_USAGE, **_LABELLED}, {}, False, None, id="openrouter-label-without-a-bill"),
    pytest.param("openai", {**_RESPONSES_USAGE, **_LABELLED}, {}, False, None, id="responses-label-without-a-bill"),
    pytest.param("anthropic", {**_NATIVE_CACHE, **_LABELLED}, {"total_cost_usd": 0.031}, False, 0.031,
                 id="anthropic-label-beside-a-reported-charge"),
    pytest.param("anthropic", _NATIVE_CACHE, {}, False, None, id="anthropic-no-bill"),
    pytest.param("openrouter", {"cost": 0, "is_byok": True}, {}, False, None, id="openrouter-byok-gateway-fee-alone"),
    pytest.param("openrouter", {**_CHAT_USAGE, "cost": 0}, {}, False, 0.0, id="openrouter-reported-zero"),
])
def test_a_paid_search_event_states_whether_its_attempt_row_found_the_bill_unreadable(
        paid_leg, source, usage, envelope, unreadable, cost):
    """An unreadable bill (NaN) is unknown money the attempt row marks ``cost_invalid``; a bill that is merely
    absent, or half a BYOK price, is unknown and not marked. The event of that attempt id says the same in both
    of its copies and so does the supervisor's row, on each paid leg: a reply that labels itself ``cost_invalid``
    marks nothing, and the label never stands beside a reported $0.031."""
    _result, event, row, persisted = paid_leg(source, usage, envelope)
    _one_fact(event, row, persisted, cost, cost is not None)
    assert bool(row.get("cost_invalid")) is unreadable
    assert bool(event.get("cost_invalid")) is bool(event["usage"].get("cost_invalid")) is unreadable
    assert persisted["cost_invalid"] is unreadable
    assert persisted["ledger_attempt_ids"] == event["ledger_attempt_ids"] == [row["attempt_id"]]


def test_a_search_never_waits_out_the_money_lock_for_its_event_and_states_no_fact_it_could_not_read(
        paid_leg, monkeypatch, tmp_path):
    """Another connection holds the ledger EXCLUSIVE (a commit in progress) from the moment a leg reads its settled
    row for the display event. The search still answers at once, far inside the 45-second money wait: the event
    then states no amount, no finality, no validity and no count, never a $0 or a measured zero, and the attempt
    row keeps the $0.031 the provider reported, as every ordinary read sees once the hold ends."""
    import threading
    import time

    from ouroboros import usage_store
    from tests._store_journal_shared import exclusive_hold

    read, waits = search_module._settled_fact, []

    def held_off(attempt_ids, drive_root=None):
        hold = exclusive_hold(tmp_path / usage_store.STORE_REL)
        started = time.monotonic()
        try:
            return read(attempt_ids, drive_root)
        finally:
            waits.append(time.monotonic() - started)
            hold.execute("ROLLBACK")
            hold.close()

    monkeypatch.setattr(search_module, "_settled_fact", held_off)
    result, event, row, persisted = paid_leg("openrouter", {**_CHAT_USAGE, "cost": 0.031})
    assert result["answer"] == "answer" and len(waits) == 1
    assert waits[0] < 5.0, f"the event's read held a successful search for {waits[0]:.1f}s"
    assert (row["state"], row["cost_usd"], row["cost_final"]) == ("settled", 0.031, True)
    assert [row[key] for key in _SEARCH_COUNTERS] == [100, 20, None, None]
    for holder in (event, event["usage"], persisted):
        assert holder["cost"] is None, holder
        assert (holder["cost_final"], holder["cost_estimated"], holder["cost_invalid"]) == (False, False, False)
    assert persisted["cost_known"] is False
    assert [event[key] for key in _SEARCH_COUNTERS] == [persisted[key] for key in _SEARCH_COUNTERS] == [None] * 4
    assert persisted["ledger_attempt_ids"] == [row["attempt_id"]]  # the id still names the row that holds the money
    # Only the display fact is short. Held off by the same lock, the ledger's ordinary read still waits for the
    # exact row, as settlement does, while the display fact has already given up.
    from ouroboros._usage_settlement import attempt_money_fact

    def ordinary_read():
        with usage_store.read(tmp_path) as txn:
            exact.append(txn.attempt(row["attempt_id"]))

    exact, hold = [], exclusive_hold(tmp_path / usage_store.STORE_REL)
    waiting = threading.Thread(target=ordinary_read, daemon=True)
    try:
        waiting.start()
        waiting.join(timeout=1.0)
        assert waiting.is_alive() and not exact, "an ordinary ledger read refused instead of waiting"
        assert attempt_money_fact(tmp_path, row["attempt_id"]) == {}
    finally:
        hold.execute("ROLLBACK")
        hold.close()
    waiting.join(timeout=60)
    assert [(found["state"], found["cost_usd"], found["cost_final"]) for found in exact] == [("settled", 0.031, True)]
    # With the ledger free the same reader states the row's own fact again.
    monkeypatch.setattr(search_module, "_settled_fact", read)
    _result, event, row, persisted = paid_leg("openrouter", {**_CHAT_USAGE, "cost": 0.031})
    _one_fact(event, row, persisted, 0.031, True)


@pytest.mark.parametrize("beside,split", [
    pytest.param({"prompt_tokens_details": {"audio_tokens": 40}}, False, id="audio-beside-the-text-counts"),
    pytest.param({}, True, id="text-alone"),
])
def test_a_searchs_reported_charge_is_one_amount_from_its_row_to_its_event_and_the_costs_page(
        paid_leg, tmp_path, beside, split):
    """The provider billed one search $0.031 for 100 prompt and 20 output tokens. The attempt row, the leg's event,
    the supervisor's row and the Costs period state that same $0.031. With a price list of $1 input and $2 output a
    million at hand, Costs shares it out by class only when the text counts are all the reply reported: beside 40
    audio tokens no text price ratio explains the bill, so the amount stays whole and unshared."""
    import asyncio

    from ouroboros import pricing
    from ouroboros.gateway.history import make_cost_breakdown_endpoint
    from ouroboros.provider_models import normalize_model_identity

    usage = {**_CHAT_USAGE, "cached_tokens": 0, "cache_write_tokens": 0, "cost": 0.031, **beside}
    _result, event, row, persisted = paid_leg("openrouter", usage)
    _one_fact(event, row, persisted, 0.031, True)
    paid_leg.catalog["rows"] = {normalize_model_identity(row["model"]): pricing.PricingSchedule((1.0, 0.5, 3.0, 2.0))}
    served = json.loads(asyncio.run(make_cost_breakdown_endpoint(tmp_path)(
        types.SimpleNamespace(query_params={"period": "24h"}))).body)
    model, = served["period"]["models"]
    assert served["total_cost"] == model["cost"] == 0.031 and (model["model"], model["calls"]) == (row["model"], 1)
    if split:  # 100 prompt tokens at $1 and 20 output tokens at $2 a million weigh 100 : 40
        assert model["by_class"] == pytest.approx({"uncached_input": 0.031 * 100 / 140, "output": 0.031 * 40 / 140,
                                                   "cache_read": 0.0, "cache_write": 0.0})
        assert (model["unshared_calls"], model["unshared_cost"]) == (0, 0)
    else:
        assert model["by_class"] is None and (model["unshared_calls"], model["unshared_cost"]) == (1, 0.031)
