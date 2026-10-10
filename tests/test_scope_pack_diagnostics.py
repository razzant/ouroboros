"""Brief preparation diagnostics survive the move from packets to the one wave.

A retrieving seat's two-part brief is assembled before anything is dispatched.
When that assembly fails, the wave is blocked fail-closed and ONE durable,
seat-local row (`review_brief_preparation_failed`) lands in the task's events
log, forwarded live by the existing sink; a successful preparation emits
nothing. Logging can never change the typed block the gate returns.
"""

from __future__ import annotations

import json
import queue
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ouroboros import utils
from ouroboros.tools import review as review_mod
from ouroboros.tools import review_admission as admission
from ouroboros.tools import review_brief_coupling as brief_mod
from supervisor.log_addressing import make_server_log_sink


pytestmark = pytest.mark.serial
_TASK_ADDRESS = {"chat_id": 42, "parent_task_id": "brief-parent", "root_task_id": "brief-root"}


def _plan(*routes, model="test/model"):
    from ouroboros.review_execution import ReviewRouteKind

    kinds = [ReviewRouteKind.AGENT_SESSION if r == "agent_session" else ReviewRouteKind.API_CHAT for r in routes]
    n = len(kinds)
    return {"models": [model] * n, "routes": kinds, "efforts": [""] * n, "session_targets": [model] * n,
            "session_profiles": [""] * n, "subagent_ids": [""] * n, "use_local": [None] * n,
            "slot_ids": [f"seat-{i + 1}" for i in range(n)], "retrieves": [True] * n,
            "parts": [("change", "coupling")] * n}


@pytest.fixture
def brief_env(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext
    import ouroboros.reviewer_slot_config as slot_cfg

    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "CHECKLISTS.md").write_text(
        "## Coupling questions\n\nplaceholder\n", encoding="utf-8")
    (repo / "docs" / "DEVELOPMENT.md").write_text("development\n", encoding="utf-8")
    (repo / "BIBLE.md").write_text("constitution\n", encoding="utf-8")
    (repo / ".gitignore").write_text(".review-drive/\n", encoding="utf-8")
    (repo / "example.py").write_text("value = 1\n", encoding="utf-8")
    for args in (("init",), ("add", "."), ("commit", "-m", "initial")):
        subprocess.run(
            ["git", "-c", "user.email=test@example.test", "-c", "user.name=Test", *args],
            cwd=repo, check=True, capture_output=True,
        )
    (repo / "example.py").write_text("value = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    drive = tmp_path / "drive"
    drive.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    ctx.task_id = "brief-task"
    ctx.pending_events = []
    ctx._review_history, ctx._review_advisory, ctx._coupling_review_history = [], [], {}
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setattr(brief_mod, "first_send_bound", lambda _brief: 900_000)
    monkeypatch.setattr(slot_cfg, "commit_triad_delivery", lambda: _plan("api_chat"))
    dispatch = Mock(side_effect=AssertionError("preparation must not dispatch a reviewer"))
    monkeypatch.setattr(review_mod, "_handle_multi_model_review", dispatch)
    forwarded = []
    bridge = SimpleNamespace(push_log=forwarded.append)
    monkeypatch.setattr(utils, "_log_sink", make_server_log_sink(
        bridge, drive, running={"brief-task": {"task": dict(_TASK_ADDRESS)}},
    ))
    yield ctx, forwarded, dispatch
    dispatch.assert_not_called()


def _events(ctx):
    path = ctx.drive_root / "logs" / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _prepare(ctx, message="Update value"):
    return review_mod._prepare_unified_review(ctx, message)


@pytest.mark.parametrize("route", ["api_chat", "agent_session"])
def test_context_failure_persists_and_forwards_one_row(brief_env, monkeypatch, route):
    import ouroboros.reviewer_slot_config as slot_cfg

    ctx, forwarded, _ = brief_env
    monkeypatch.setattr(slot_cfg, "commit_triad_delivery", lambda: _plan(route))
    # Exercise the actual builder's missing-checklist refusal, through the gate's assembly.
    monkeypatch.setattr(brief_mod, "load_checklist_section", lambda _section: "")

    prepared, early, exited = _prepare(ctx)

    assert prepared is None and exited
    assert "REVIEW_BLOCKED: Failed to build the review brief" in early
    assert "Seat seat-1:" in early and "could not be loaded" in early
    assert ctx._last_review_block_reason == "infra_failure"
    event, = _events(ctx)
    assert forwarded == [{**event, **_TASK_ADDRESS}]
    assert not _TASK_ADDRESS.keys() & event.keys(), "addressing must not mutate the durable row"
    assert event["ts"]
    assert {key: value for key, value in event.items() if key != "ts"} == {
        "type": admission.BRIEF_PREPARATION_FAILED_EVENT, "task_id": "brief-task",
        "slot_id": "seat-1", "model": "test/model", "parts": ["change", "coupling"], "status": "error",
        "failure_phase": "context", "failure_code": "context_unavailable",
        "reason": (
            "Coupling questions could not be loaded from docs/CHECKLISTS.md — "
            "the coupling question cannot be asked without its checklist (fail-closed)."
        ),
    }
    assert ctx.pending_events == []


@pytest.mark.parametrize("failure", ["false", "append_error", "missing_logs", "sink_error"])
def test_diagnostic_failure_preserves_exact_preparation_result(brief_env, monkeypatch, failure):
    ctx, _, _ = brief_env
    monkeypatch.setattr(brief_mod, "load_checklist_section", lambda _section: "")
    with monkeypatch.context() as baseline:
        baseline.setattr(utils, "append_jsonl", lambda *_args: False)
        expected = _prepare(ctx)
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
        monkeypatch.setattr(utils, "append_jsonl", broken_append)
    elif failure == "missing_logs":
        monkeypatch.setattr(ctx, "drive_logs", None)
    else:
        monkeypatch.setattr(utils, "_log_sink", broken_sink)
    result = _prepare(ctx)

    assert result == expected and result[0] is None and result[2] is True
    assert len(calls) == (0 if failure == "missing_logs" else 1)
    assert len(_events(ctx)) == (1 if failure == "sink_error" else 0)


@pytest.mark.parametrize("route", ["api_chat", "agent_session"])
@pytest.mark.parametrize("delivery", ["inline", "paged"])
def test_successful_preparation_emits_no_failure(brief_env, monkeypatch, route, delivery):
    """The gate captures the staged diff itself and hands it to every seat's
    brief, so at the gate the diff is inline or paged — never left for the
    reviewer to retrieve (that delivery exists only for a brief built outside a
    repository and is pinned in tests/test_review_session_scope_wiring.py)."""
    import ouroboros.reviewer_slot_config as slot_cfg
    from ouroboros.tools import review_binary_context

    ctx, forwarded, _ = brief_env
    monkeypatch.setattr(slot_cfg, "commit_triad_delivery", lambda: _plan(route))
    if delivery == "paged":
        # Use the real source store and reader addresses, with a small test ceiling.
        monkeypatch.setattr(brief_mod, "first_send_bound", lambda _brief: 1)
        monkeypatch.setattr(brief_mod, "SESSION_INLINE_DIFF_CEILING_CHARS", 1)
    prepared, early, exited = _prepare(ctx)

    assert early is None and not exited
    manifest, = prepared["retrieving_manifests"]
    assert manifest["slot_id"] == "seat-1" and manifest["parts"] == ["change", "coupling"]
    assert manifest["diff_delivery"] == delivery, manifest
    assert prepared["row_plan"]["brief_shas"] == [manifest["sha"]["brief"]]
    assert _events(ctx) == forwarded == []
    if delivery == "paged":
        from ouroboros.artifacts import read_actor_source_bytes

        rows = prepared["row_plan"]["session_policies"][0]["native_required_sources"]
        diff_row = next(row for row in rows if row["disposition"] == "review_subject")
        assert manifest["diff_source"]["required_row"] == diff_row
        stored = read_actor_source_bytes(ctx.drive_root, ctx.task_id, manifest["diff_source"])
        assert stored.decode("utf-8") == review_binary_context.capture_staged_diff(ctx.repo_dir)


def test_model_control_error_propagates_before_diagnostic(brief_env, monkeypatch):
    from ouroboros.llm_claudexor import ClaudexorModelError

    ctx, forwarded, _ = brief_env
    error = ClaudexorModelError({"code": "model_outcome_unknown", "message": "original custody"})
    monkeypatch.setattr(admission, "retrieving_brief_for_seat", Mock(side_effect=error))
    with pytest.raises(ClaudexorModelError) as raised:
        _prepare(ctx)
    assert raised.value is error and _events(ctx) == forwarded == []


def test_worker_log_envelope_forwards_without_a_new_dispatch_kind(brief_env, monkeypatch):
    from supervisor import events
    from supervisor.worker_process import WORKER_LOG_SINK_SUPPRESSED_TYPES

    ctx, forwarded, _ = brief_env
    monkeypatch.setattr(brief_mod, "load_checklist_section", lambda _section: "")
    outgoing = queue.Queue()
    monkeypatch.setattr(utils, "_log_sink", lambda row: utils.emit_log_event(outgoing, row))
    _prepare(ctx)
    durable, = _events(ctx)
    assert durable["type"] not in WORKER_LOG_SINK_SUPPRESSED_TYPES
    envelope = outgoing.get_nowait()
    assert outgoing.empty() and envelope == {"type": "log_event", "data": durable}
    events.dispatch_event(envelope, SimpleNamespace(
        DRIVE_ROOT=ctx.drive_root, RUNNING={ctx.task_id: {"task": dict(_TASK_ADDRESS)}},
        bridge=SimpleNamespace(push_log=forwarded.append), append_jsonl=utils.append_jsonl,
    ))
    assert forwarded == [{**durable, **_TASK_ADDRESS}] and _events(ctx) == [durable]


def test_a_seat_preparation_failure_names_the_seat_and_blocks_the_whole_wave(brief_env, monkeypatch):
    """One wave: a seat whose brief cannot be built blocks the wave fail-closed
    before any seat is dispatched (no half-panel review), and the durable row
    names exactly the seat that failed."""
    import ouroboros.reviewer_slot_config as slot_cfg

    ctx, forwarded, _ = brief_env
    monkeypatch.setattr(slot_cfg, "commit_triad_delivery", lambda: _plan("api_chat", "agent_session"))
    original = brief_mod.build_retrieving_brief

    def build(repo, brief):
        if brief.slot_id == "seat-1":
            raise RuntimeError("fixture context unavailable")
        return original(repo, brief)

    monkeypatch.setattr(brief_mod, "build_retrieving_brief", build)
    prepared, early, exited = _prepare(ctx)
    assert prepared is None and exited
    assert "Seat seat-1: fixture context unavailable" in early
    event, = _events(ctx)
    assert forwarded == [{**event, **_TASK_ADDRESS}] and event["slot_id"] == "seat-1"
    assert not {"blocked", "verdict", "block_message", "context_manifest"} & event.keys()
