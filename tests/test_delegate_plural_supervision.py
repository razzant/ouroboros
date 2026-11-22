"""Interleaved waits, deliveries and migration preserve each selected run."""
from types import SimpleNamespace
import json
import errno
import os

import pytest

from ouroboros import delegate_supervision as supervision
from ouroboros import delegate_supervision_state as checkpoints
from ouroboros.delegate_shared import delegate_payload
from tests._run_wait_shared import held_run, wake_now


def context(root):
    return SimpleNamespace(task_id="nanny", task_attempt=1, drive_root=root, budget_drive_root=root, task_metadata={})


def wake(ctx, run_id, sequence=1):
    return wake_now(ctx, run_id, wait_once=lambda _ctx, rid, *_a, **_k: json.dumps({
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
    """The wake of a plural wait speaks for the selected run whose terminal the body recorded, read once; the
    quiet sibling is neither read nor given a wake."""
    from ouroboros import model_sleep

    ctx = context(tmp_path)
    held_run(tmp_path, "a", "nanny")
    held_run(tmp_path, "b", "nanny", ended="succeeded")
    observed = []

    def observer(_ctx, run, *_args, **_kwargs):
        observed.append(run)
        return json.dumps({"status": "terminal" if run == "b" else "progress", "run_id": run, "last_seq": 1})

    woke_by = model_sleep.wake_reason(ctx, supervision._choice({"runs": ["a", "b"]}))
    assert woke_by == "run:b"
    result = delegate_payload(wake_now(ctx, ["a", "b"], wait_once=observer, woke_by=woke_by))
    assert result["run_id"] == "b" and observed == ["b"]
    assert supervision.supervision_checkpoint(ctx)["runs"].get("a", {}).get("pending_wake") is None
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
    refused = delegate_payload(supervision.run_wait(ctx, "a"))
    assert refused["reason"] == "supervision_state_unknown" and not getattr(ctx, "_pending_task_wait", None)


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
    """Each run's cursor comes from the child's own record and each run's observation from the body's record of it."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.tools.control_task_results import _delegated_run_facts
    from tests.test_wait_child_projection_and_window import _observed, _seed_child, CHILD

    _seed_child(tmp_path, record={"schema": 2, "runs": {"run-a": {"journal_cursor": 11}, "run-b": {"journal_cursor": 23}}},
                runs=("run-a", "run-b"))
    _observed(tmp_path, "run-a")
    _observed(tmp_path, "run-b", error=ClaudexorUnavailable(
        "daemon_unreachable", "nothing answered", observation_timeout=True, observation_reason="daemon_unreachable"))
    facts = {item["run_id"]: item for item in _delegated_run_facts(tmp_path, CHILD)}
    assert facts["run-a"]["supervision"] == {"journal_cursor": 11} and facts["run-a"]["observation"]["observation_failure"] == ""
    assert facts["run-b"]["supervision"] == {"journal_cursor": 23}
    assert facts["run-b"]["observation"]["observation_failure"] == "daemon_unreachable"


def test_unsaved_wake_names_its_stage_and_platform_cause_without_deciding_custody(tmp_path, monkeypatch):
    """#3051 on the sleep: the wake publication is the wait's one save (the entry and quiet-renewal saves went with
    the tick); its failure names the stage and the platform cause, returns the observation already produced, writes no
    custody fact and leaves nothing pending, so a later wait publishes the same facts."""
    from ouroboros import delegate_custody as custody
    from ouroboros.gateways import claudexor as cx
    from tests._run_wait_shared import scripted_reads
    from tests.test_builtin_refusal_results import _delegate_registry
    from tests.test_delegate_body_collection import started

    registry = _delegate_registry(tmp_path, monkeypatch)
    entry = started(tmp_path, tid=registry._ctx.task_id, daemon_store_id="store-a")
    held_run(tmp_path, entry.run_id, registry._ctx.task_id, asks="q-1")  # a question already in custody: ready at once
    monkeypatch.setattr(cx, "ClaudexorGateway", lambda *a, **k: pytest.fail("reached real engine"))
    monkeypatch.setattr(supervision, "_loop_gateway", lambda *a: SimpleNamespace(close=lambda: None))

    def observe(_ctx, rid, *_a, **_k):
        return json.dumps({"status": "waiting_on_user", "run_id": rid, "last_seq": 5,
                           "pending_interactions": [{"interaction_id": "q-1"}]})

    scripted_reads(monkeypatch, observe)
    before = [row["type"] for row in custody.custody_rows(tmp_path)]
    real_save, real_open, armed, fired = checkpoints.save_view, os.open, [], []

    def create(path, flags, *args, **kwargs):
        if (armed and "delegate_actor_claims" in str(path) and str(path).endswith(".lock")
                and flags & os.O_CREAT and flags & os.O_EXCL):
            fired.append(armed.pop())
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(path))
        return real_open(path, flags, *args, **kwargs)

    def save(path, drive, task_id, view, *rest):
        if view.get("status") == "wake_pending" and not fired and not armed:
            armed.append("wake_publication")
        return real_save(path, drive, task_id, view, *rest)

    monkeypatch.setattr(os, "open", create)
    monkeypatch.setattr(checkpoints, "save_view", save)
    text = registry.execute("delegate_wait", {"run_id": entry.run_id})
    assert fired == ["wake_publication"]
    assert "errno 28 ENOSPC" in text and os.strerror(errno.ENOSPC) in text, text
    assert "supervision_state_unsaved" in text and "The wake_publication save did not land" in text, text
    assert "Nothing was cancelled" not in text
    assert '"observed_payload"' in text and '"waiting_on_user"' in text
    saved = supervision.supervision_checkpoint(registry._ctx)["runs"].get(entry.run_id, {})
    assert not saved.get("pending_wake") and int(saved.get("journal_cursor") or 0) == 0
    after = [row["type"] for row in custody.custody_rows(tmp_path)]
    assert after[:len(before)] == before
    assert all(kind.startswith("delegate_supervision_") for kind in after[len(before):])
    assert not custody.replay(tmp_path)[entry.run_id].settled
    again = registry.execute("delegate_wait", {"run_id": entry.run_id})
    assert "waiting_on_user" in again and "supervision_wake_id" in again, again
    assert "supervision_state_unsaved" not in again
