"""Tests for ``ouroboros.tools.search._web_search`` (OpenAI Responses API).

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

import pytest

import ouroboros.tools.search as search_module
from ouroboros.tools.search import _web_search


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
    result = json.loads(_web_search(ctx, "same query"))
    assert result["answer"] == "answer"
    assert [call.kwargs["service_tier"] for call in mock_openai.responses.create.call_args_list] == ["flex", "default"]
    assert [request.processing_preference for request in prepared] == ["economy", "economy"]
    assert prepared[0].candidate_raw_sha256 != prepared[1].candidate_raw_sha256
    with ua._locked(holds[0].drive_root):
        records = ua._read_records_locked_cached(holds[0].drive_root)
    matching = [row for row in records if row.get("candidate_raw_sha256") == prepared[0].candidate_raw_sha256]
    assert matching[-1]["state"] == "released"


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

    assert result == {"answer": "fresh answer", "answer_type": "summary", "sources": [], "backend": "openai_responses"}
    assert calls["api_key"] == "openai-key"
    assert calls["base_url"] is None
    assert calls["kwargs"]["model"] == "gpt-5.2"
    assert calls["kwargs"]["stream"] is True
    assert calls["kwargs"]["tools"][0]["type"] == "web_search"
    assert request_ctx.pending_events[0]["provider"] == "openai"
    assert request_ctx.pending_events[0]["model"] == "gpt-5.2"


def test_web_search_falls_back_to_openrouter_server_tool(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    calls: dict = {}

    class _Usage:
        def model_dump(self):
            return {"prompt_tokens": 11, "completion_tokens": 7}

    class _Message:
        content = "fresh answer from openrouter"

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]
        usage = _Usage()

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
    mock_openai.responses.create.return_value = _FakeStream(events)

    result = _web_search(ctx, "test query")

    ctx.emit_progress_fn.assert_called_once()
    call_text = ctx.emit_progress_fn.call_args[0][0]
    assert "test query" in call_text

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

    progress_text = ctx.emit_progress_fn.call_args[0][0]
    serialized = json.dumps(result)
    assert leaked_secret not in progress_text
    assert leaked_secret not in serialized
    assert "***REDACTED***" in progress_text
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
    rows = [
        json.loads(line)
        for line in (tmp_path / ua.LEDGER_REL).read_text().splitlines()
        if line.strip()
    ]
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


def _captured_provider_error(*, provider="openrouter", status=None):
    from ouroboros.usage_accounting import PhysicalAttemptCapture

    exc = RuntimeError("provider transport ended")
    exc.physical_attempt_capture = PhysicalAttemptCapture(
        attempt_id="attempt-1",
        model="model-1",
        provider=provider,
        state="unresolved",
        candidate_measurement_kind="opaque",
        provider_status_code=status,
    )
    return exc


def test_openrouter_ambiguous_outcome_stops_cross_provider_cascade(ctx, monkeypatch):
    from ouroboros import llm

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.delenv("OUROBOROS_WEBSEARCH_BACKEND", raising=False)
    monkeypatch.setattr(
        llm, "openrouter_web_search_server_tool",
        lambda **_kwargs: (_ for _ in ()).throw(_captured_provider_error()),
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
            _captured_provider_error(status=503)
        ),
    )
    monkeypatch.setattr(
        search_module, "_web_search_anthropic",
        lambda *_args, **_kwargs: json.dumps({
            "answer": "recovered", "backend": "anthropic_server_tool",
        }),
    )

    result = json.loads(_web_search(ctx, "terminal OpenRouter query"))

    assert result == {"answer": "recovered", "backend": "anthropic_server_tool"}


def test_explicit_web_503_allows_one_safe_retry(ctx, patch_env, monkeypatch):
    calls = 0

    class _Terminal503(Exception):
        status_code = 503

    class _Responses:
        def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _Terminal503("service unavailable")
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


def test_streamed_web_503_preserves_status_for_one_safe_retry(ctx, patch_env, monkeypatch):
    calls = 0

    class _Responses:
        def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _FakeStream([
                    _make_event(
                        "response.failed",
                        error=types.SimpleNamespace(message="service unavailable", status_code=503),
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

    ctx.emit_progress_fn.assert_not_called()
    data = json.loads(result)
    assert data["answer"] == "Direct answer"


def test_streaming_empty_text_engages_cascade(ctx, patch_env, mock_openai):
    """NW-11: an empty OpenAI result (no text AND no sources) is a soft failure,
    not a successful "(no answer)" — it must fall through to the provider
    cascade so a working fallback backend is not shadowed. With no fallback
    backend configured here, the cascade is exhausted and an error is returned
    (NOT a fake "(no answer)" success)."""
    events = [_make_completed_event(10, 0)]
    mock_openai.responses.create.return_value = _FakeStream(events)

    result = _web_search(ctx, "empty query")

    data = json.loads(result)
    assert "answer" not in data
    assert "error" in data
    assert any("no answer and no sources" in e for e in data.get("backend_errors", []))


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

    assert ctx.emit_progress_fn.call_count == 1
