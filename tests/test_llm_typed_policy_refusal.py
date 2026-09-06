"""A typed policy refusal is never swallowed, repaired, or repeated.

Two seams answer for one fact. The recovery ladder must not consume a refusal
(it has no provider answer to fall back FROM), and the retry loop's classifier
must not call it retryable (repeating an unchanged refused call only re-runs the
refusal). Both read the declared code; neither reads the message.

The refusals here are SYNTHETIC. Nothing is copied from, imported from, or
shaped after any deployment's own refusal type: one subclasses the published
contract, the sibling only sets the declared ``code`` — the shape a transport
that cannot import Ouroboros uses to state the same fact. Neither is recognised
by any word in its message (BIBLE P5: no keyword gates), and the assertions below
would fail if the ladder matched prose instead of the typed fact.

What the fallback must do with one: nothing. A refused call never reached a
provider, so there is no provider answer to fall back FROM — dropping a
parameter, rerouting the endpoint or stripping replayed reasoning would all
re-attempt a call a policy layer declined, and the caller would be handed the
re-attempt's outcome (or the first errored response) instead of the refusal.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any, Dict, List

import pytest

from ouroboros.llm import PROVIDER_POLICY_REFUSAL, LLMClient, ProviderPolicyRefusal
from ouroboros.loop_llm_call import classify_llm_exception


class NoPermittedConnection(ProviderPolicyRefusal):
    """Fixture refusal: the host policy permits no connection for this call."""


class EgressDeniedByPolicy(RuntimeError):
    """Sibling refusal from a transport that never imports Ouroboros: it states
    the same fact with the declared code and nothing else."""

    code = PROVIDER_POLICY_REFUSAL


class TenantBlocked(RuntimeError):
    """Sibling refusal whose message is deliberately full of words the recovery
    ladder DOES key on for real provider failures (temperature, reasoning,
    unsupported parameter, rate limit). A prose matcher would repair it."""

    code = PROVIDER_POLICY_REFUSAL

    def __init__(self) -> None:
        super().__init__(
            "temperature reasoning unsupported parameter rate limit "
            "no endpoints found overloaded try again"
        )


_REFUSALS = [NoPermittedConnection("connection is not permitted"), EgressDeniedByPolicy("denied"),
             TenantBlocked()]
_REFUSAL_IDS = ["subclass", "code_only_sibling", "code_only_with_recoverable_prose"]

_BODY_429 = {
    "id": "gen-1", "choices": None,
    "error": {"code": 429, "message": "rate limit exceeded upstream"},
}
_BODY_400_ENCRYPTED = {
    "id": "gen-2", "choices": None,
    "error": {"code": 400, "message": "The encrypted content for item rs_a could not be verified"},
}
_BODY_400_PARAM = {
    "id": "gen-3", "choices": None,
    "error": {"code": 400, "message": "temperature: unsupported parameter for this model"},
}
_REPLAYED_REASONING: List[Dict[str, Any]] = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "a",
     "reasoning_details": [{"type": "reasoning.encrypted", "data": "rs_a"}]},
    {"role": "user", "content": "again"},
]


class _Resp:
    def __init__(self, body: Dict[str, Any]) -> None:
        self._body = body

    def model_dump(self) -> Dict[str, Any]:
        return copy.deepcopy(self._body)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-fixture-key")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    import ouroboros.llm_attempt as llm_attempt

    # The physical-attempt ledger is exercised by the golden fixtures; here the
    # subject is the ladder, so the executor is a pass-through.
    monkeypatch.setattr(llm_attempt, "execute_physical_attempt",
                        lambda _request, send, **_kw: send())

    async def _async_execute(_request, send, **_kw):
        return await send()

    monkeypatch.setattr(llm_attempt, "execute_physical_attempt_async", _async_execute)
    return LLMClient(api_key="or-fixture-key")


def _script(steps):
    """A create_fn that walks a scripted list of responses/raises."""
    calls: List[Dict[str, Any]] = []

    def create(**kwargs):
        calls.append(kwargs)
        step = steps[len(calls) - 1]
        if isinstance(step, BaseException):
            raise step
        return _Resp(step)

    return create, calls


def _async_script(steps):
    calls: List[Dict[str, Any]] = []

    async def create(**kwargs):
        calls.append(kwargs)
        step = steps[len(calls) - 1]
        if isinstance(step, BaseException):
            raise step
        return _Resp(step)

    return create, calls


def _target() -> Dict[str, Any]:
    return {
        "provider": "openrouter",
        "resolved_model": "openai/gpt-5.6",
        "usage_model": "openai/gpt-5.6",
        "api_key": "or-fixture-key",
        "base_url": "https://openrouter.ai/api/v1",
        "default_headers": {},
        "supports_openrouter_extensions": True,
        "supports_generation_cost": True,
    }


def _kwargs() -> Dict[str, Any]:
    return {
        "model": "openai/gpt-5.6",
        "messages": copy.deepcopy(_REPLAYED_REASONING),
        "max_tokens": 512,
        "temperature": 0.7,
        "extra_body": {"reasoning": {"effort": "high", "exclude": False},
                       "session_id": "ouroboros-session-fixture"},
    }


@pytest.mark.parametrize("refusal", _REFUSALS, ids=_REFUSAL_IDS)
@pytest.mark.parametrize("first_body", [_BODY_429, _BODY_400_ENCRYPTED, _BODY_400_PARAM],
                         ids=["transient_reroute", "encrypted_strip", "parameter_retry"])
def test_body_rung_does_not_swallow_a_typed_refusal(client, refusal, first_body):
    """Every 200-body rung answers a refused resend by returning the FIRST
    (errored) response. A typed refusal must surface instead — the caller would
    otherwise be told the call was rate-limited/bad-request by a provider that
    never saw it."""
    create, calls = _script([first_body, refusal])

    with pytest.raises(type(refusal)) as excinfo:
        client._create_chat_completion_with_retries(create, _kwargs(), _target())

    assert excinfo.value is refusal
    assert len(calls) == 2  # the rung's one resend, and no more


@pytest.mark.parametrize("refusal", _REFUSALS, ids=_REFUSAL_IDS)
def test_async_body_rung_does_not_swallow_a_typed_refusal(client, refusal):
    create, calls = _async_script([_BODY_429, refusal])

    with pytest.raises(type(refusal)) as excinfo:
        asyncio.run(
            client._create_chat_completion_with_retries_async(create, _kwargs(), _target())
        )

    assert excinfo.value is refusal
    assert len(calls) == 2


@pytest.mark.parametrize("refusal", _REFUSALS, ids=_REFUSAL_IDS)
def test_exception_ladder_never_re_attempts_a_refused_call(client, refusal):
    """A refusal on the FIRST send is not a parameter, cache or signature
    problem: no rung may spend a second physical attempt on it."""
    create, calls = _script([refusal, {"choices": [{"message": {"content": "never"}}]}])

    with pytest.raises(type(refusal)) as excinfo:
        client._create_chat_completion_with_retries(create, _kwargs(), _target())

    assert excinfo.value is refusal
    assert len(calls) == 1


@pytest.mark.parametrize("refusal", _REFUSALS, ids=_REFUSAL_IDS)
def test_refusal_surfaces_through_the_public_chat_surface(client, monkeypatch, refusal):
    """End to end: LLMClient.chat hands the refusal to its caller unchanged."""
    create, calls = _script([_BODY_429, refusal])

    import types

    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    monkeypatch.setattr(client, "_get_remote_client", lambda _target: fake_client, raising=False)
    # No capability fetch on the payload-build path: this test is about the ladder.
    monkeypatch.setattr(client, "_get_supported_parameters", lambda _model: None, raising=False)

    with pytest.raises(type(refusal)) as excinfo:
        # Replayed reasoning is what makes the 429 body eligible for the
        # same-model reroute whose resend the policy layer then refuses.
        client.chat(copy.deepcopy(_REPLAYED_REASONING), model="openai/gpt-5.6")

    assert excinfo.value is refusal
    assert len(calls) == 2


def test_an_ordinary_resend_failure_is_still_absorbed(client):
    """The guard is typed, not a blanket 'never absorb': an ordinary provider
    failure on the resend keeps returning the first response, exactly as before."""
    create, calls = _script([_BODY_429, RuntimeError("second endpoint also down")])

    resp = client._create_chat_completion_with_retries(create, _kwargs(), _target())

    assert resp.model_dump()["error"]["code"] == 429
    assert len(calls) == 2


def test_a_refusal_carrying_a_foreign_code_is_not_treated_as_a_refusal(client):
    """The code is an exact declared value, not a prefix or a substring scan."""

    class OtherCode(RuntimeError):
        code = "provider_policy_refusal_v2"

    create, calls = _script([_BODY_429, OtherCode("something else")])

    resp = client._create_chat_completion_with_retries(create, _kwargs(), _target())

    assert resp.model_dump()["error"]["code"] == 429
    assert len(calls) == 2


# --- the retry loop's own answer -------------------------------------------
#
# The ladder above declines to REPAIR a refusal; the retry loop must also decline
# to REPEAT it. Both seams read the same typed fact, so the assertions below go
# through `classify_llm_exception` — the function the loop actually consults —
# rather than any marker table it happens to consult on the way.


@pytest.mark.parametrize("refusal", _REFUSALS, ids=_REFUSAL_IDS)
def test_the_retry_loop_classifies_a_typed_refusal_as_permanent(refusal):
    """A refused call is a named, non-retryable class.

    Its kind is the declared code itself, so `events.jsonl` names the refusal
    instead of laundering it into the catch-all `provider_error`; and
    `retry_same_request` is False, so the loop stops rather than spending its
    whole attempt budget re-running a call no provider ever saw."""
    classification = classify_llm_exception(refusal)

    assert classification.kind == PROVIDER_POLICY_REFUSAL
    assert classification.retry_same_request is False
    assert classification.provider_code == PROVIDER_POLICY_REFUSAL
    # Recovery is not a known instant; nothing here may schedule a wake-up.
    assert classification.retry_after_sec is None


def test_the_retry_loop_prefers_the_typed_fact_over_recoverable_prose():
    """`TenantBlocked`'s message says "rate limit" — the text the loop keys on
    for its retryable class. The typed fact must win: prose from a call that
    never reached a provider describes nothing that could heal."""
    refusal = TenantBlocked()

    assert "rate limit" in str(refusal)  # the trap is really in the message
    assert classify_llm_exception(refusal).retry_same_request is False


def test_the_retry_loop_does_not_read_a_foreign_code_as_a_refusal():
    """The loop's branch is the same exact-value test the ladder makes, not a
    prefix match: a longer code that merely CONTAINS the declared one keeps the
    ordinary classification it would have had."""

    class OtherCode(RuntimeError):
        code = "provider_policy_refusal_v2"

    classification = classify_llm_exception(OtherCode("something else"))

    assert classification.kind != PROVIDER_POLICY_REFUSAL
    assert classification.retry_same_request is True
