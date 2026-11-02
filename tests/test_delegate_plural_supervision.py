"""Interleaved waits, deliveries and migration preserve each selected run."""
from types import SimpleNamespace
import json

import pytest

from ouroboros import delegate_supervision as supervision
from ouroboros import delegate_supervision_state as checkpoints
from ouroboros.delegate_shared import delegate_payload


def context(root):
    return SimpleNamespace(task_id="nanny", task_attempt=1, drive_root=root, budget_drive_root=root, task_metadata={})


def wake(ctx, run_id, sequence=1):
    return supervision.supervised_wait(ctx, run_id, wait_once=lambda _ctx, rid, *_a, **_k: json.dumps({
        "status": "waiting_on_user", "run_id": rid, "last_seq": sequence,
        "pending_interactions": [{"interaction_id": f"q-{rid}"}]}))


def test_interleaved_runs_ack_only_the_wakes_delivered(tmp_path):
    ctx = context(tmp_path)
    a, b = delegate_payload(wake(ctx, "a", 11)), delegate_payload(wake(ctx, "b", 23))
    assert supervision.acknowledge_pending_wake(ctx, a)
    state = supervision.supervision_checkpoint(ctx)
    assert not state["runs"]["a"]["pending_wake"]
    assert state["runs"]["b"]["pending_wake"]["wake_id"] == b["supervision_wake_id"]
    assert state["runs"]["a"]["interaction_acknowledged_ids"] == ["q-a"]
    assert state["runs"]["b"].get("interaction_acknowledged_ids", []) == []
    assert supervision.acknowledge_pending_wake(ctx, {"wakes": [b]})
    assert {key: item["journal_cursor"] for key, item in supervision.supervision_checkpoint(ctx)["runs"].items()} == {"a": 11, "b": 23}


def test_stale_views_merge_only_their_run_and_changed_task_fields(tmp_path):
    ctx = context(tmp_path)
    a, b = supervision._load_state(ctx, "a"), supervision._load_state(ctx, "b")
    a.update(journal_cursor=4, coordination_cursor={"children": {"one": "done"}})
    b.update(journal_cursor=9, coordination_cursor={"children": {"two": "done"}})
    supervision._save_state(ctx, a)
    supervision._save_state(ctx, b)
    state = supervision.supervision_checkpoint(ctx)
    assert state["coordination_cursor"] == {"children": {"one": "done", "two": "done"}}
    assert state["runs"]["a"]["journal_cursor"] == 4 and state["runs"]["b"]["journal_cursor"] == 9


@pytest.mark.parametrize("concurrent", [False, True])
def test_same_run_publish_preserves_the_first_undelivered_wake(tmp_path, concurrent):
    ctx = context(tmp_path)
    first, second = supervision._load_state(ctx, "a"), supervision._load_state(ctx, "a")
    first.update(journal_cursor=4, pending_wake={"wake_id": "first"})
    second.update(journal_cursor=9, pending_wake={"wake_id": "second"})
    if concurrent:
        supervision._save_state(ctx, first)
    supervision._save_state(ctx, second)
    saved = supervision.supervision_checkpoint(ctx)["runs"]["a"]
    assert saved["pending_wake"] == second["pending_wake"] == {"wake_id": "first" if concurrent else "second"}
    assert saved["journal_cursor"] == (4 if concurrent else 9)


def test_failed_ack_persistence_never_marks_an_unseen_sibling(tmp_path, monkeypatch):
    from ouroboros.delegate_interactions import _REPORTED_INTERACTIONS

    ctx = context(tmp_path)
    a, b = delegate_payload(wake(ctx, "a")), delegate_payload(wake(ctx, "b"))
    _REPORTED_INTERACTIONS.clear()
    save = checkpoints.atomic_write_json
    monkeypatch.setattr(checkpoints, "atomic_write_json", lambda *_a, **_k: (_ for _ in ()).throw(OSError("unwritable")))
    with pytest.raises(OSError):
        supervision.acknowledge_pending_wake(ctx, a)
    assert not _REPORTED_INTERACTIONS
    assert all(item["pending_wake"] for item in supervision.supervision_checkpoint(ctx)["runs"].values())
    monkeypatch.setattr(checkpoints, "atomic_write_json", save)
    assert supervision.acknowledge_pending_wake(ctx, a)
    assert "b" not in _REPORTED_INTERACTIONS
    assert supervision.acknowledge_pending_wake(ctx, b)
    assert _REPORTED_INTERACTIONS["b"] == {"q-b"}


def test_plural_wait_wakes_on_any_selected_run_without_acking_others(tmp_path):
    ctx = context(tmp_path)
    observed = []

    def observer(_ctx, run, *_args, **_kwargs):
        observed.append(run)
        return json.dumps({"status": "terminal" if run == "b" else "progress", "run_id": run, "last_seq": 1})

    result = delegate_payload(supervision.supervised_wait(ctx, ["a", "b"], wait_once=observer))
    assert result["run_id"] == "b" and observed == ["a", "b"]
    assert supervision.supervision_checkpoint(ctx)["runs"]["a"].get("pending_wake") is None
    assert delegate_payload(supervision.delegate_wait_entry(ctx, "a", runs=["b"]))["reason"] == "run_selector_conflict"


def test_singleton_migration_is_read_only_until_next_mutation(tmp_path):
    ctx = context(tmp_path)
    path = supervision._state_path(ctx)
    path.parent.mkdir(parents=True)
    original = json.dumps({"schema": 1, "run_id": "a", "journal_cursor": 7,
                           "pending_wake": {"wake_id": "held"}, "unknown_provider_hold": {"run_id": "a"}})
    path.write_text(original)
    assert supervision._load_state(ctx, "a")["journal_cursor"] == 7
    assert path.read_text() == original
    b = supervision._load_state(ctx, "b")
    b["journal_cursor"] = 10
    supervision._save_state(ctx, b)
    saved = json.loads(path.read_text())
    assert saved["schema"] == 2 and set(saved["runs"]) == {"a", "b"}
    assert saved["runs"]["a"]["pending_wake"] == {"wake_id": "held"}
    path.write_text('{"schema":2,"runs":null}')
    refused = delegate_payload(supervision.supervised_wait(ctx, "a", wait_once=lambda *_a: pytest.fail("unknown state observed")))
    assert refused["reason"] == "supervision_state_unknown"


@pytest.mark.parametrize("corrupt", [False, True])
def test_unknown_hold_latch_never_replaces_unreadable_run_checkpoints(tmp_path, monkeypatch, corrupt):
    from ouroboros import delegate_hold
    from tests.test_delegate_hold import _configured_registry

    registry = _configured_registry(tmp_path)
    ctx = registry._ctx
    path = supervision._state_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = '{"schema":2,"runs":null}' if corrupt else '{"schema":2,"runs":{}}'
    path.write_text(original)
    monkeypatch.setattr(delegate_hold, "_eligible_live_runs", lambda _ctx: ["a", "b"])
    latched = delegate_hold.latch_after_unknown(registry, error_kind="provider_outcome_unknown",
        drive_logs=tmp_path, task_id=ctx.task_id, emit_progress=lambda *_a: None)
    assert latched is not corrupt
    if corrupt:
        assert path.read_text() == original
    else:
        assert supervision.read_unknown_hold(ctx)["run_ids"] == ["a", "b"]


def test_child_projection_reads_each_runs_own_observation(tmp_path):
    from ouroboros.tools.control_task_results import _delegated_run_facts
    from tests.test_wait_child_projection_and_window import _seed_child, CHILD

    _seed_child(tmp_path, record={"schema": 2, "runs": {
        "run-a": {"journal_cursor": 11, "observation": {"answered": True, "run_state": "running"}},
        "run-b": {"journal_cursor": 23, "observation": {"answered": False, "reason": "daemon_unreachable"}},
    }}, runs=("run-a", "run-b"))
    facts = {item["run_id"]: item["supervision"] for item in _delegated_run_facts(tmp_path, CHILD)}
    assert facts["run-a"]["journal_cursor"] == 11 and facts["run-a"]["observation_failure"] == ""
    assert facts["run-b"]["journal_cursor"] == 23 and facts["run-b"]["observation_failure"] == "daemon_unreachable"
