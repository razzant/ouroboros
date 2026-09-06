"""Regression tests for LLM fork-safety: no_proxy parameter.

Covers:
- chat_async with no_proxy=True uses httpx.AsyncClient(trust_env=False) for non-Anthropic
- chat_async with no_proxy=True passes no_proxy through to _chat_anthropic for Anthropic
- _chat_anthropic with no_proxy=True uses requests.Session(trust_env=False)
- _chat_remote passes no_proxy through to _chat_anthropic for Anthropic provider
- plan_review, review.py, scope_review.py call chat_async with no_proxy=True
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch



# ---------------------------------------------------------------------------
# Test: _chat_anthropic uses requests.Session(trust_env=False) when no_proxy=True
# ---------------------------------------------------------------------------

def test_chat_anthropic_no_proxy_uses_session_trust_env_false():
    """_chat_anthropic(no_proxy=True) must use requests.Session with trust_env=False."""
    from ouroboros.llm import LLMClient

    target = {
        "provider": "anthropic",
        "resolved_model": "claude-opus-4-5",
        "usage_model": "anthropic/claude-opus-4-5",
        "api_key": "test-key",
        "base_url": "https://api.anthropic.com/v1",
        "default_headers": {},
        "supports_openrouter_extensions": False,
        "supports_generation_cost": False,
    }
    messages = [{"role": "user", "content": "hello"}]

    client = LLMClient()

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "content": [{"type": "text", "text": "Hi"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "stop_reason": "end_turn",
        "role": "assistant",
        "model": "claude-opus-4-5",
    }

    captured_session_trust_env = []



    class FakeSession:
        def __init__(self):
            self.trust_env = True  # Default

        def post(self, url, **kwargs):
            captured_session_trust_env.append(self.trust_env)
            return fake_response

        # Context manager support for `with requests.Session() as session:`
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    with patch("requests.Session", FakeSession):
        msg, usage = client._chat_anthropic(
            target, messages, None, "medium", 1024, "auto", None, no_proxy=True
        )

    assert len(captured_session_trust_env) == 1, "Session.post should be called once"
    assert captured_session_trust_env[0] is False, (
        f"Expected trust_env=False, got {captured_session_trust_env[0]}"
    )


def test_chat_anthropic_no_proxy_false_uses_requests_post():
    """_chat_anthropic(no_proxy=False) retries parameter rejection without Session."""
    from ouroboros.llm import LLMClient

    target = {
        "provider": "anthropic",
        "resolved_model": "claude-opus-4-5",
        "usage_model": "anthropic/claude-opus-4-5",
        "api_key": "test-key",
        "base_url": "https://api.anthropic.com/v1",
        "default_headers": {},
        "supports_openrouter_extensions": False,
        "supports_generation_cost": False,
    }
    messages = [{"role": "user", "content": "hello"}]
    client = LLMClient()

    ok_response = MagicMock()
    ok_response.raise_for_status = MagicMock()
    ok_response.status_code = 200
    ok_response.json.return_value = {
        "content": [{"type": "text", "text": "Hi"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "stop_reason": "end_turn",
        "role": "assistant",
    }
    rejected_response = MagicMock()
    rejected_response.status_code = 400
    rejected_response.reason = "Bad Request"
    rejected_response.url = "https://api.anthropic.com/v1/messages"
    rejected_response.text = "temperature is not supported for this model"

    session_called = []

    class FakeSession:
        def __init__(self):
            self.trust_env = True
        def post(self, url, **kwargs):
            session_called.append(True)
            return ok_response

    fake_post = MagicMock(side_effect=[ok_response, rejected_response, ok_response])

    with patch("requests.post", side_effect=fake_post), \
         patch("requests.Session", FakeSession):
        client._chat_anthropic(
            target, messages, None, "medium", 1024, "auto", None, no_proxy=False
        )
        target["resolved_model"] = "claude-opus-4-8"
        target["usage_model"] = "anthropic/claude-opus-4-8"
        client._chat_anthropic(
            target, messages, None, "medium", 1024, "auto", 0.2, no_proxy=False
        )

    direct_target = client._resolve_remote_target("anthropic::claude-opus-4.8")

    assert fake_post.call_count == 3, "requests.post should retry once for parameter rejection"
    assert len(session_called) == 0, "Session should NOT be used for no_proxy=False"
    captured_payloads = [call.kwargs.get("json") or {} for call in fake_post.call_args_list]
    assert captured_payloads[1]["model"] == "claude-opus-4-8"
    assert captured_payloads[1]["temperature"] == 0.2
    assert captured_payloads[2]["model"] == "claude-opus-4-8"
    assert "temperature" not in captured_payloads[2]
    # v6.73.2 named-param binding: the error names ONLY temperature, so the
    # thinking/output_config effort carrier SURVIVES the retry — a temperature
    # rejection no longer collaterally strips reasoning control.
    assert "thinking" in captured_payloads[2]
    assert captured_payloads[2]["output_config"] == {"effort": "medium"}
    assert direct_target["provider"] == "anthropic"
    assert direct_target["resolved_model"] == "claude-opus-4-8"
    assert direct_target["usage_model"] == "anthropic/claude-opus-4-8"


def test_chat_anthropic_honors_explicit_timeout():
    from ouroboros.llm import LLMClient

    target = {
        "provider": "anthropic",
        "resolved_model": "claude-opus-4-5",
        "usage_model": "anthropic/claude-opus-4-5",
        "api_key": "test-key",
        "base_url": "https://api.anthropic.com/v1",
        "default_headers": {},
        "supports_openrouter_extensions": False,
        "supports_generation_cost": False,
    }
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "content": [{"type": "text", "text": "Hi"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "stop_reason": "end_turn",
        "role": "assistant",
    }
    fake_post = MagicMock(return_value=fake_response)

    with patch("requests.post", side_effect=fake_post):
        LLMClient()._chat_anthropic(
            target, [{"role": "user", "content": "hello"}], None, "medium", 1024, "auto", None,
            no_proxy=False, timeout=77.0,
        )

    assert fake_post.call_args.kwargs["timeout"] == 77.0


# ---------------------------------------------------------------------------
# Test: chat_async with no_proxy=True passes through to Anthropic path
# ---------------------------------------------------------------------------

def test_chat_async_no_proxy_anthropic_path():
    """chat_async(no_proxy=True) on an Anthropic model must pass no_proxy=True
    to _chat_anthropic via asyncio.to_thread."""
    from ouroboros.llm import LLMClient

    client = LLMClient()
    messages = [{"role": "user", "content": "hello"}]
    model = "anthropic::claude-opus-4-5"

    captured_no_proxy = []

    def fake_chat_anthropic(target, messages, tools, effort, max_tokens, tool_choice, temp, np=False, timeout=None):
        captured_no_proxy.append(np)
        return {"role": "assistant", "content": "Hi"}, {"prompt_tokens": 10, "completion_tokens": 5}

    with patch.object(client, "_chat_anthropic", side_effect=fake_chat_anthropic):
        asyncio.run(
            client.chat_async(messages=messages, model=model, no_proxy=True)
        )

    assert len(captured_no_proxy) == 1, "_chat_anthropic should be called once"
    assert captured_no_proxy[0] is True, (
        f"no_proxy should be True, got {captured_no_proxy[0]}"
    )


# ---------------------------------------------------------------------------
# Test: chat_async with no_proxy=True uses httpx.AsyncClient for non-Anthropic
# ---------------------------------------------------------------------------

def test_chat_async_no_proxy_non_anthropic_uses_httpx_async_client():
    """chat_async(no_proxy=True) for a non-Anthropic model must create an
    httpx.AsyncClient with trust_env=False and mounts={}."""
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test-or-key")
    messages = [{"role": "user", "content": "hello"}]
    model = "openai/gpt-5.5"

    captured_httpx_kwargs = []


    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured_httpx_kwargs.append(kwargs)
            self.closed = False

        async def aclose(self):
            self.closed = True

    fake_oa_client = MagicMock()
    fake_create = AsyncMock(return_value=MagicMock(
        model_dump=lambda: {
            "choices": [{"message": {"role": "assistant", "content": "Hi", "tool_calls": None}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    ))
    fake_oa_client.chat.completions.create = fake_create

    # Resolve SDK inheritance before substituting its HTTP base class.
    with patch("openai.AsyncOpenAI", return_value=fake_oa_client), \
         patch("httpx.AsyncClient", FakeAsyncClient), \
         patch("requests.get", side_effect=AssertionError("no_proxy must not use requests.get")):
        asyncio.run(
            client.chat_async(messages=messages, model=model, no_proxy=True)
        )

    assert len(captured_httpx_kwargs) == 1, "httpx.AsyncClient should be created once"
    kw = captured_httpx_kwargs[0]
    assert kw.get("trust_env") is False, f"Expected trust_env=False, got {kw.get('trust_env')}"
    assert kw.get("mounts") == {}, f"Expected mounts={{}}, got {kw.get('mounts')}"


# ---------------------------------------------------------------------------
# Test: _chat_remote passes no_proxy to _chat_anthropic for Anthropic provider
# ---------------------------------------------------------------------------

def test_chat_remote_passes_no_proxy_to_anthropic():
    """_chat_remote with Anthropic target and no_proxy=True must call
    _chat_anthropic with no_proxy=True."""
    from ouroboros.llm import LLMClient

    client = LLMClient()
    messages = [{"role": "user", "content": "hello"}]

    target = {
        "provider": "anthropic",
        "resolved_model": "claude-opus-4-5",
        "usage_model": "anthropic/claude-opus-4-5",
        "api_key": "test-key",
        "base_url": "https://api.anthropic.com/v1",
        "default_headers": {},
        "supports_openrouter_extensions": False,
        "supports_generation_cost": False,
    }

    captured_no_proxy = []
    captured_timeout = []

    def fake_chat_anthropic(t, msgs, tools, effort, max_tok, tc, temp=None, no_proxy=False, timeout=None):
        captured_no_proxy.append(no_proxy)
        captured_timeout.append(timeout)
        return {"role": "assistant", "content": "Hi"}, {}

    with patch.object(client, "_chat_anthropic", side_effect=fake_chat_anthropic):
        client._chat_remote(
            target, messages, None, "medium", 1024, "auto", None, no_proxy=True, timeout=88.0
        )

    assert len(captured_no_proxy) == 1
    assert captured_no_proxy[0] is True, (
        f"no_proxy should be True when passed to _chat_remote, got {captured_no_proxy[0]}"
    )
    assert captured_timeout[0] == 88.0


def test_chat_remote_no_proxy_retries_openrouter_parameter_rejection():
    """OpenRouter no_proxy path retries once without optional sampling params."""
    from ouroboros.llm import LLMClient

    LLMClient._REJECTED_PARAMS_CACHE.clear()
    client = LLMClient(api_key="test-or-key")
    target = client._resolve_remote_target("anthropic/claude-opus-4.8")
    messages = [{"role": "user", "content": "hello"}]
    captured_kwargs = []

    class ParameterRejection(RuntimeError):
        status_code = 404
        body = {"error": {
            "message": "No endpoints found for requested parameter temperature",
        }}

    class FakeCompletions:
        def create(self, **kwargs):
            captured_kwargs.append(kwargs)
            if len(captured_kwargs) == 1:
                raise ParameterRejection(
                    "404 No endpoints found for requested parameter temperature"
                )
            return MagicMock(
                model_dump=lambda: {
                    "choices": [{"message": {"role": "assistant", "content": "ok", "tool_calls": None}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            )

    fake_oa_client = MagicMock()
    fake_oa_client.chat.completions = FakeCompletions()
    fake_http_client = MagicMock()

    with patch.object(client, "_make_no_proxy_client", return_value=(fake_oa_client, fake_http_client)), \
         patch("requests.get", side_effect=AssertionError("no_proxy must not fetch capabilities")):
        msg, usage = client._chat_remote(
            target,
            messages,
            None,
            "medium",
            1024,
            "auto",
            0.2,
            no_proxy=True,
        )

    assert msg["content"] == "ok"
    assert len(captured_kwargs) == 2
    assert captured_kwargs[0]["temperature"] == 0.2
    assert "temperature" not in captured_kwargs[1]
    assert captured_kwargs[1]["extra_body"]["reasoning"]["effort"] == "medium"
    assert captured_kwargs[1]["extra_body"]["provider"]["require_parameters"] is True
    fake_http_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Test: plan_review ReviewCoordinator path calls LLM with no_proxy=True
# ---------------------------------------------------------------------------

def test_plan_review_slots_use_no_proxy(tmp_path):
    """plan_review's shared ReviewCoordinator path must pass no_proxy=True."""
    from ouroboros.tools import plan_review, plan_review_runtime

    captured_kwargs = []

    class FakeLLMClient:
        def chat(self, **kwargs):
            captured_kwargs.append(kwargs)
            return {"content": "AGGREGATE: GREEN\nAll good."}, {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "resolved_model": "test-model",
            }

    import asyncio as _asyncio
    fake_ctx = type("FakeCtx", (), {
        "drive_root": tmp_path,
        "task_id": "test-plan-review",
        "pending_events": [],
        "event_queue": None,
    })()

    # The slot runner (and its LLMClient construction site) moved to
    # plan_review_runtime; patch the owning module, not the re-exporting one.
    from ouroboros.review_substrate import ReviewSlot

    with patch.object(plan_review_runtime, "LLMClient", return_value=FakeLLMClient()):
        result = _asyncio.run(
            plan_review._run_plan_review_slots(
                fake_ctx,
                [ReviewSlot(slot_id="slot_1", model="openai/gpt-5.5", effort="high")],
                system_prompt="system prompt",
                user_content="user content",
            )
        )

    assert result[0]["error"] is None
    assert len(captured_kwargs) == 1, "LLM should be called once"
    assert captured_kwargs[0].get("no_proxy") is True, (
        f"Expected no_proxy=True, got {captured_kwargs[0].get('no_proxy')}"
    )


# ---------------------------------------------------------------------------
# Test: review.py _query_model calls chat_async with no_proxy=True
# ---------------------------------------------------------------------------

def test_review_query_model_uses_no_proxy():
    """_query_model in review.py must call chat_async with no_proxy=True.

    _query_model signature: (llm_client, model, messages, semaphore)
    where messages is already a list of {role, content} dicts.
    """
    import asyncio
    from ouroboros.tools import review as review_mod

    captured_kwargs = []

    class FakeLLMClient:
        async def chat_async(self, **kwargs):
            captured_kwargs.append(kwargs)
            return {"content": "PASS"}, {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "resolved_model": "test-model",
            }

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user content"},
    ]

    asyncio.run(
        review_mod._query_model(
            FakeLLMClient(),
            "openai/gpt-5.5",
            messages,
            asyncio.Semaphore(1),
        )
    )

    assert len(captured_kwargs) == 1, "chat_async should be called once"
    assert captured_kwargs[0].get("no_proxy") is True, (
        f"Expected no_proxy=True in review._query_model, got {captured_kwargs[0].get('no_proxy')}"
    )


# ---------------------------------------------------------------------------
# Test: scope_review _call_scope_llm calls chat_async with no_proxy=True
# ---------------------------------------------------------------------------

def test_scope_review_call_scope_llm_uses_no_proxy():
    """_call_scope_llm in scope_review.py must call chat_async with no_proxy=True.

    Both the ThreadPoolExecutor path and the RuntimeError fallback path must pass
    no_proxy=True. We test the asyncio.run() fallback path (RuntimeError branch)
    by ensuring no running loop is active during the call.
    """
    from ouroboros.tools import scope_review

    captured_kwargs = []

    class FakeLLMClient:
        async def chat_async(self, **kwargs):
            captured_kwargs.append(kwargs)
            return {"content": "[]"}, {"prompt_tokens": 100, "completion_tokens": 50}

    prompt = "test prompt for scope review"

    with patch.object(scope_review, "LLMClient", return_value=FakeLLMClient()):
        raw_text, usage, error = scope_review._call_scope_llm(prompt)

    assert len(captured_kwargs) >= 1, "chat_async should be called at least once"
    for kw in captured_kwargs:
        assert kw.get("no_proxy") is True, (
            f"scope_review._call_scope_llm chat_async called without no_proxy=True: {kw}"
        )
