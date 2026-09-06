"""Final-attempt route facts follow Claudexor's telemetry artifact, not request echoes."""

from pathlib import Path

import pytest
import yaml

from ouroboros.gateways.claudexor import final_attempt_facts


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


@pytest.mark.parametrize("missing", [None, "", 12, False, ["model"], {"model": "value"}])
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
    ([{"attempt_id": "a02"}, {"attempt_id": "a02"}], "a02"),
    ([{"attempt_id": "a01", "observed_model": "earlier-model"}], None),
    ([{"attempt_id": "a01", "observed_model": "earlier-model"}], ""),
    ([{"attempt_id": "a01", "observed_model": "earlier-model"}], " "),
    ([{"attempt_id": 2, "observed_model": "earlier-model"}], 2),
    ({"a02": {"observed_model": "earlier-model"}}, "a02"),
    ([None, "a02"], "a02"),
])
def test_unbound_or_ambiguous_final_attempt_is_unknown(tmp_path, attempts, final_id):
    detail = _write_telemetry(tmp_path, attempts, final_id=final_id)

    assert final_attempt_facts(detail, "run-fixture") == {}


@pytest.mark.parametrize("run_id", ["other-run", "", None, 1])
def test_telemetry_must_belong_to_the_requested_run(tmp_path, run_id):
    detail = _write_telemetry(tmp_path, [{"attempt_id": "a02", "observed_model": "actual"}])

    assert final_attempt_facts(detail, run_id) == {}


@pytest.mark.parametrize("raw", [b"", b"null", b"[]", b"bad: [", b"\xff"])
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


@pytest.mark.parametrize("detail", [None, [], {}, {"summary": []}, {"summary": {}},
                                     {"summary": {"runDir": ""}}, {"summary": {"runDir": 42}}])
def test_missing_engine_run_directory_never_reads_the_working_directory(detail, monkeypatch):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("missing runDir must not become a relative path")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    assert final_attempt_facts(detail, "run-fixture") == {}
