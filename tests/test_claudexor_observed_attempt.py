"""Final-attempt route facts follow Claudexor's telemetry artifact, not request echoes."""

from pathlib import Path

import pytest
import yaml

from ouroboros.gateways.claudexor import final_attempt_facts
from ouroboros.llm_claudexor import (
    ClaudexorModelError,
    _ModelInvocation,
    _request,
)
from ouroboros.llm_substitution import remember_failed_profile


def _write_telemetry(tmp_path, attempts, *, final_id="a02", run_id="run-fixture"):
    # Claudexor 3.9.8 RunTelemetry shape, reduced to the fields this reader owns.
    # Route values below were observed on all three harnesses; IDs are fixtures.
    path = tmp_path / "final" / "telemetry.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "schema_version": 2, "run_id": run_id, "task_id": "task-fixture",
        "final_attempt_id": final_id, "attempts": attempts,
    }), encoding="utf-8")
    return {"summary": {
        "runDir": str(tmp_path), "model": "request-echo", "harnesses": ["requested-harness"],
        "route": {"observedModel": "earlier-model", "harnessId": "earlier-harness", "verified": True},
        "authRoute": {"attemptId": "a01", "profileId": "earlier-profile"},
    }}


@pytest.mark.parametrize("harness,requested,observed", [
    ("codex", "gpt-6-astra", "gpt-6-astra"),
    ("claude", "claude-fable-5-1", "claude-fable-5-1"),
    ("cursor", "cursor-grok-4.6-xhigh", "Cursor Grok 4.6 Extra High"),
])
def test_final_attempt_keeps_observed_values_together(tmp_path, harness, requested, observed):
    detail = _write_telemetry(tmp_path, [
        {"attempt_id": "a01", "harness_id": "earlier-harness", "observed_model": "earlier-model",
         "profile_id": "earlier-profile"},
        {"attempt_id": "a02", "harness_id": harness, "observed_model": observed,
         "requested_model": requested, "profile_id": "final-profile", "auth_mode": "local_session"},
        {"attempt_id": "a03", "harness_id": "later-harness", "observed_model": "later-model",
         "profile_id": "later-profile"},
    ])

    assert final_attempt_facts(detail, "run-fixture") == {
        "attempt_id": "a02", "harness_id": harness, "model": observed, "profile_id": "final-profile",
    }


@pytest.mark.parametrize("missing", [None, 12, {"model": "value"}])
def test_final_attempt_missing_facts_never_borrow_from_earlier_attempt(tmp_path, missing):
    detail = _write_telemetry(tmp_path, [
        {"attempt_id": "a01", "harness_id": "earlier-harness", "observed_model": "earlier-model",
         "profile_id": "earlier-profile"},
        {"attempt_id": "a02", "harness_id": missing, "observed_model": missing,
         "requested_model": "request-echo", "profile_id": missing},
    ])

    assert final_attempt_facts(detail, "run-fixture") == {
        "attempt_id": "a02", "harness_id": "", "model": "", "profile_id": "",
    }


@pytest.mark.parametrize("attempts,final_id", [
    ([{"attempt_id": "a01", "observed_model": "earlier-model"}], "a02"),
    ([{"attempt_id": "a01", "observed_model": "earlier-model"}], None),
    ([{"attempt_id": "a01", "observed_model": "earlier-model"}], ""),
])
def test_unbound_or_ambiguous_final_attempt_is_unknown(tmp_path, attempts, final_id):
    detail = _write_telemetry(tmp_path, attempts, final_id=final_id)

    assert final_attempt_facts(detail, "run-fixture") == {}


@pytest.mark.parametrize("run_id", ["other-run", "", 1])
def test_telemetry_must_belong_to_the_requested_run(tmp_path, run_id):
    detail = _write_telemetry(tmp_path, [{"attempt_id": "a02", "observed_model": "actual"}])

    assert final_attempt_facts(detail, run_id) == {}


@pytest.mark.parametrize("raw", [b"", b"bad: [", b"\xff"])
def test_unreadable_telemetry_stays_unknown(tmp_path, raw):
    detail = _write_telemetry(tmp_path, [{"attempt_id": "a02", "observed_model": "actual"}])
    (tmp_path / "final" / "telemetry.yaml").write_bytes(raw)

    assert final_attempt_facts(detail, "run-fixture") == {}


def test_missing_or_inaccessible_telemetry_stays_unknown(tmp_path, monkeypatch):
    detail = {"summary": {"runDir": str(tmp_path), "model": "request-echo"}}
    assert final_attempt_facts(detail, "run-fixture") == {}

    def inaccessible(*args, **kwargs):
        raise PermissionError("fixture denies the artifact read")

    monkeypatch.setattr(Path, "read_text", inaccessible)
    assert final_attempt_facts(detail, "run-fixture") == {}


@pytest.mark.parametrize("detail", [None, [], {"summary": {"runDir": 42}}])
def test_missing_engine_run_directory_never_reads_the_working_directory(detail, monkeypatch):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("missing runDir must not become a relative path")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    assert final_attempt_facts(detail, "run-fixture") == {}


@pytest.mark.parametrize("applied", [
    {"reasoningEffort": "xhigh"}, None, {"reasoningEffort": None},
])
def test_model_invocation_records_requested_and_applied_options(applied):
    requested = {"reasoningEffort": "xhigh"}
    invocation = _ModelInvocation(
        {"usage_model": "claudexor::codex=model"}, {"options": requested}, {}
    )
    result = {
        "outcome": "completed",
        "message": {"role": "assistant", "content": "done"},
    }
    if applied is not None:
        result["appliedOptions"] = applied

    _message, usage = invocation.finish(result)
    observed = usage["claudexor"]
    assert observed["requested_options"] == requested
    assert observed["applied_options"] == applied
    assert usage["effort"]["reported"] == (applied or {}).get("reasoningEffort")
    assert usage["effort"]["report_source"] == ("provider_applied_options" if isinstance(applied, dict) and applied.get("reasoningEffort") is not None else None)


def test_a_differently_echoed_cache_key_is_a_durable_mismatch_of_its_own():
    """The recorded state covers every submitted option, not the thinking horizon alone."""
    requested = {"reasoningEffort": "xhigh", "cacheKey": "execution-a"}
    invocation = _ModelInvocation(
        {"usage_model": "claudexor::codex=model"}, {"options": requested}, {}
    )

    _message, usage = invocation.finish({
        "outcome": "completed", "message": {"role": "assistant", "content": "done"},
        "appliedOptions": {"reasoningEffort": "xhigh", "cacheKey": "engine-b"},
    })

    assert usage["claudexor"]["requested_options"] == requested
    assert usage["claudexor"]["applied_options"] == {"reasoningEffort": "xhigh", "cacheKey": "engine-b"}


def _continuation(profile="profile-a", source="codex", model="gpt-6"):
    return [{
        "role": "assistant",
        "content": "prior answer",
        "nativeContinuation": {"route": {
            "source": source, "model": model, "credentialProfileId": profile,
        }},
    }]


def test_successful_profile_remains_an_auto_lane_preference():
    target = {"source": "codex", "resolved_model": "gpt-6"}

    payload = _request(target, _continuation(), None, {"cache_affinity": "execution-success"})

    assert payload["account"] == {"mode": "auto", "preferredProfileId": "profile-a"}


def test_status_null_failure_suppresses_only_the_next_same_route_preference():
    target = {"source": "codex", "resolved_model": "gpt-6"}
    parameters = {"cache_affinity": "execution-failed"}
    error = ClaudexorModelError(
        {"code": "server_error", "message": "stream ended"},
        route={"source": "codex", "model": "gpt-6", "credentialProfileId": "profile-a"},
        unknown=True,
    )
    remember_failed_profile(target, parameters, error)

    assert _request(target, _continuation(), None, parameters)["account"] == {"mode": "auto"}
    assert _request(target, _continuation(), None, parameters)["account"] == {
        "mode": "auto", "preferredProfileId": "profile-a",
    }


def test_failure_fact_survives_an_interleaved_request_on_another_route():
    target = {"source": "codex", "resolved_model": "gpt-6"}
    other = {"source": "claude", "resolved_model": "sonnet"}
    parameters = {"cache_affinity": "execution-interleaved"}
    error = ClaudexorModelError(
        {"code": "server_error", "message": "stream ended"},
        route={"source": "codex", "model": "gpt-6", "credentialProfileId": "profile-a"},
        unknown=True,
    )
    remember_failed_profile(target, parameters, error)

    # A request on another route neither consumes the fact nor loses its own preference.
    assert _request(other, _continuation("profile-b", "claude", "sonnet"), None, parameters)["account"] == {
        "mode": "auto", "preferredProfileId": "profile-b",
    }
    assert _request(target, _continuation(), None, parameters)["account"] == {"mode": "auto"}
    assert _request(target, _continuation(), None, parameters)["account"] == {
        "mode": "auto", "preferredProfileId": "profile-a",
    }


def test_failure_fact_does_not_change_pin_or_single_account_auto_mode():
    target = {"source": "codex", "resolved_model": "gpt-6"}
    parameters = {"cache_affinity": "execution-pin"}
    error = ClaudexorModelError(
        {"code": "subscription_window_exhausted", "message": "window spent",
         "context": {"httpStatus": 429}},
        route={"source": "codex", "model": "gpt-6", "credentialProfileId": "only-profile"},
    )
    remember_failed_profile(target, parameters, error)

    pinned = _request(target, _continuation("only-profile"), None, {
        **parameters, "model_account_override": "only-profile",
    })
    assert pinned["account"] == {"mode": "pin", "profileId": "only-profile"}

    remember_failed_profile(target, parameters, error)
    assert _request(target, _continuation("only-profile"), None, parameters)["account"] == {
        "mode": "auto",
    }


def _released(code):
    from types import SimpleNamespace
    from ouroboros.llm_claudexor import ClaudexorModelNotDispatched

    error = ClaudexorModelNotDispatched({"code": code, "message": "no request left the host"},
                                        route={"source": "codex", "model": "gpt-6", "credentialProfileId": "profile-a"})
    error.physical_attempt_capture = SimpleNamespace(state="released", provider="claudexor", route_is_loopback=False)
    return error


@pytest.mark.parametrize(("error", "kept"), [
    (ClaudexorModelError({"code": "context_length_exceeded", "message": "too long"},
                         route={"source": "codex", "model": "gpt-6", "credentialProfileId": "profile-a"}), True),
    (_released("model_account_unavailable"), False),  # an engine account verdict, though it reads as a transport wait
    (ClaudexorModelError({"code": "context_length_exceeded", "message": "too long", "context": {"poolCause": "quota"}},
                         route={"source": "codex", "model": "gpt-6", "credentialProfileId": "profile-a"}), False),
    (ClaudexorModelError({"code": "server_error", "message": "stream ended"}, unknown=True,
                         route={"source": "codex", "model": "gpt-6", "credentialProfileId": "profile-a"}), False),
    (ClaudexorModelError({"code": "rate_limited", "message": "slow down"},
                         route={"source": "codex", "model": "gpt-6", "credentialProfileId": "profile-a"}), False),
])
def test_a_size_refusal_keeps_the_conversations_account(error, kept):
    """C2 S5 (06.11.2026): a size refusal says nothing about the account, so the shrunk request stays
    where its prefix is cached; an unknown outcome or any account evidence still drops it once."""
    target = {"source": "codex", "resolved_model": "gpt-6"}
    parameters = {"cache_affinity": f"execution-{error.code}"}
    remember_failed_profile(target, parameters, error)

    preferred = _request(target, _continuation(), None, parameters)["account"].get("preferredProfileId")
    assert preferred == ("profile-a" if kept else None)
