"""Scope packet preparation failures remain visible without changing admission."""

from __future__ import annotations

import json
import logging
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros import utils
from ouroboros.reviewer_window import ReviewerWindow
from ouroboros.tools import review_admission as admission
from ouroboros.tools import scope_review as sr
from supervisor.log_addressing import make_server_log_sink


pytestmark = pytest.mark.serial
_TASK_ADDRESS = {"chat_id": 42, "parent_task_id": "scope-parent", "root_task_id": "scope-root"}


@pytest.fixture
def scope_env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "CHECKLISTS.md").write_text(
        "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8")
    (repo / "docs" / "DEVELOPMENT.md").write_text("development\n", encoding="utf-8")
    (repo / "BIBLE.md").write_text("constitution\n", encoding="utf-8")
    (repo / "prompts").mkdir()
    (repo / "prompts" / "required.md").write_text("x" * 935_000, encoding="utf-8")
    (repo / "example.py").write_text("value = 1\n", encoding="utf-8")
    for args in (("init",), ("add", "."), ("commit", "-m", "initial")):
        subprocess.run(
            ["git", "-c", "user.email=test@example.test", "-c", "user.name=Test", *args],
            cwd=repo, check=True, capture_output=True,
        )
    (repo / "example.py").write_text("value = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    drive = tmp_path / "drive"
    ctx = SimpleNamespace(
        repo_dir=repo, drive_root=drive, task_id="scope-task",
        drive_logs=lambda: drive / "logs", pending_events=[],
    )
    monkeypatch.setattr(sr, "_scope_review_skipped_in_low_context", lambda: False)
    monkeypatch.setattr(sr, "_scope_window", lambda *_a, **_k: ReviewerWindow(
        window_tokens=1_000_000, status="confirmed"))
    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_k: 200_000)
    monkeypatch.setattr(admission, "density_probe_before_size_refusal", lambda *_a, **_k: "warm")
    forwarded = []
    bridge = SimpleNamespace(push_log=forwarded.append)
    monkeypatch.setattr(utils, "_log_sink", make_server_log_sink(
        bridge, drive, running={"scope-task": {"task": dict(_TASK_ADDRESS)}},
    ))
    return ctx, forwarded


def _events(ctx):
    path = ctx.drive_logs() / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


@pytest.mark.parametrize("window,status", [(1_000_000, "fixed_overflow"), (200_000, "sub_floor")])
@pytest.mark.parametrize("limit,mixed", [(200_000, False), (6_000, True)])
def test_unassembled_packet_persists_and_forwards_one_row(scope_env, monkeypatch, window, status, limit, mixed):
    ctx, forwarded = scope_env
    monkeypatch.setattr(sr, "_scope_window", lambda *_a, **_k: ReviewerWindow(
        window_tokens=window, status="confirmed"))
    monkeypatch.setattr(sr, "_effective_scope_input_limit", lambda **_k: limit)

    prepared, final = admission.prepare_scope_review(ctx, "Update value", scope_model="test/model", slot_id="scope-api")

    assert prepared is None and final.blocked and final.status == status
    assert "prompts/required.md" in final.block_message
    events = _events(ctx)
    assert len(events) == 1
    event = events[0]
    assert forwarded == [{**event, **_TASK_ADDRESS}]
    assert not _TASK_ADDRESS.keys() & event.keys(), "addressing must not mutate the durable row"
    assert event["ts"]
    assert {key: value for key, value in event.items() if key != "ts"} == {
        "type": "scope_review_pack_unassembled", "task_id": "scope-task",
        "slot_id": "scope-api", "model": "test/model", "status": status,
        "prompt_tokens": final.prompt_chars // 4, "prompt_tokens_source": "estimated",
        "prompt_tokens_budget": limit, "headroom_tokens": limit - final.prompt_chars // 4,
        "unassembled_required": ["prompts/required.md"], "atlas_overflowed": mixed,
    }
    if not mixed:
        assert event["headroom_tokens"] > 0


def test_fixed_overflow_warning_names_unassembled_artifact(scope_env, caplog):
    ctx, _ = scope_env
    with caplog.at_level(logging.WARNING, logger=sr.__name__):
        admission.prepare_scope_review(ctx, "Update value", scope_model="test/model")
    warnings = [record.getMessage() for record in caplog.records if record.name == sr.__name__]
    assert any("Scope review pack did not assemble:" in text and "prompts/required.md" in text for text in warnings)
    assert not any("irreducible scope prompt" in text for text in warnings)


@pytest.mark.parametrize("rebuilt_status", [None, "fixed_overflow"])
@pytest.mark.parametrize("model_override", [False, True])
def test_only_final_density_rebuild_state_is_logged(scope_env, monkeypatch, rebuilt_status, model_override):
    ctx, forwarded = scope_env
    calls = []
    limits = []
    waiter = SimpleNamespace(overrides={})
    monkeypatch.setattr("ouroboros.model_wait.current_model_wait", lambda: waiter)

    def probe(*_args, **_kwargs):
        if model_override:
            waiter.overrides["reviewer:scope-api"] = {
                "model": "test/final", "model_account_override": "profile-final", "use_local": False,
            }
        return "warm" if model_override else "measured"

    def build(*_args, **_kwargs):
        calls.append(1)
        sr._SCOPE_CONTEXT_MANIFEST.set({"selected": [], "build": len(calls)})
        if len(calls) == 1:
            return None, sr._TouchedContextStatus(status="fixed_overflow", token_count=900)
        return ("assembled", None) if rebuilt_status is None else (
            None, sr._TouchedContextStatus(status=rebuilt_status, token_count=350, atlas_overflowed=True))

    def limit(**kwargs):
        limits.append(kwargs)
        return 300 if len(limits) == 1 else 999

    monkeypatch.setattr(sr, "_build_scope_prompt", build)
    monkeypatch.setattr(sr, "_effective_scope_input_limit", limit)
    monkeypatch.setattr(admission, "density_probe_before_size_refusal", probe)
    prepared, final = admission.prepare_scope_review(ctx, "Update value", scope_model="test/model", slot_id="scope-api")

    assert len(calls) == 2
    expected_model = "test/final" if model_override else "test/model"
    binding = {"model_role": "reviewer:scope-api", "credential_profile_id": "profile-final" if model_override else ""}
    if model_override:
        binding["use_local"] = False
    assert limits == [{"scope_model": expected_model, "window_binding": binding}]
    if rebuilt_status is None:
        assert prepared is not None and final is None and _events(ctx) == forwarded == []
    else:
        assert prepared is None and final.context_manifest["build"] == 2
        event, = _events(ctx)
        assert forwarded == [{**event, **_TASK_ADDRESS}]
        assert event["model"] == expected_model
        assert event["prompt_tokens"] == 350 and event["prompt_tokens_budget"] == 300
        assert event["headroom_tokens"] == -50 and event["atlas_overflowed"] is True
        assert event["unassembled_required"] == []


@pytest.mark.parametrize("failure", ["false", "append_error", "missing_logs", "sink_error"])
def test_diagnostic_failure_preserves_preparation_result(scope_env, monkeypatch, failure):
    ctx, _ = scope_env
    calls = []

    def broken_append(*args):
        calls.append(args)
        if failure == "append_error":
            raise OSError("log unavailable")
        return False

    def broken_sink(event):
        calls.append(event)
        raise RuntimeError("forwarder unavailable")

    if failure in {"false", "append_error"}:
        monkeypatch.setattr(sr, "append_jsonl", broken_append)
    elif failure == "missing_logs":
        del ctx.drive_logs
    else:
        monkeypatch.setattr(utils, "_log_sink", broken_sink)
    prepared, final = admission.prepare_scope_review(ctx, "Update value", scope_model="test/model")

    assert prepared is None and final.blocked and final.status == "fixed_overflow"
    assert final.model_id == "test/model" and "prompts/required.md" in final.block_message
    assert final.context_manifest["ladder_steps"]
    if failure != "missing_logs":
        assert len(calls) == 1
    if failure == "sink_error":
        assert len(_events(ctx)) == 1


@pytest.mark.parametrize("status", ["empty", "omitted"])
def test_other_preparation_refusals_do_not_emit_fit_event(scope_env, monkeypatch, status):
    ctx, forwarded = scope_env
    monkeypatch.setattr(sr, "_build_scope_prompt", lambda *_a, **_k: (
        None, sr._TouchedContextStatus(status=status, omitted_paths=["unreadable.py"])))
    prepared, final = admission.prepare_scope_review(ctx, "Update value", scope_model="test/model")
    assert prepared is None and final.status == status
    assert _events(ctx) == forwarded == []


def test_fit_event_stays_row_local_when_retrieving_quorum_yields_seat(scope_env, monkeypatch):
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools import parallel_review
    from ouroboros.tools import scope_review_session

    ctx, forwarded = scope_env
    slots = [SimpleNamespace(
        slot_id=name, model="test/model", route=route, effort="", session_target="",
        session_profile="", subagent_id="",
    ) for name, route in [
        ("scope-api", ReviewRouteKind.API_CHAT),
        ("scope-session-one", ReviewRouteKind.AGENT_SESSION),
        ("scope-session-two", ReviewRouteKind.AGENT_SESSION),
    ]]
    monkeypatch.setattr(parallel_review, "scope_reviewer_slots", lambda: slots)
    monkeypatch.setattr(scope_review_session, "build_scope_session_task", lambda *_a, **_k: ("retrieve evidence", {}))
    rows = parallel_review._prepare_scope_rows(
        ctx, "Update value", goal="", scope="", review_rebuttal="",
        history_snapshot=[], scope_history=[],
    )

    assert rows[0]["final"].blocked is False and rows[0]["final"].block_message == ""
    assert all(row["prepared"] and row["final"] is None for row in rows[1:])
    event, = _events(ctx)
    assert forwarded == [{**event, **_TASK_ADDRESS}] and event["slot_id"] == "scope-api"
    assert event["status"] == "fixed_overflow"
    assert not {"blocked", "verdict", "block_message", "advisory_findings"} & event.keys()
