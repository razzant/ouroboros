"""Recovery regressions: actual admissions, launch races and durable custody."""
import json

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_owner_continue import _interrupted, NONCE


def test_publication_failure_cannot_dispatch_and_replay_repairs_same_action(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from supervisor import continuation_admission as ca
    from supervisor.queue_transitions import budget_pause_fact

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    from ouroboros import task_results as tr
    real_write = tr.write_task_result
    def fail(*args, **kwargs):
        if "continuation_admission" in kwargs:
            raise OSError("result disk unavailable")
        return real_write(*args, **kwargs)
    monkeypatch.setattr(tr, "write_task_result", fail)
    first = ca.admit_continuation("pred-1", action_nonce=NONCE)
    assert first["unconfirmed"] and not first["ok"]
    tid = first["successor_task_id"]
    assert budget_pause_fact(workers.PENDING[0]) is not None
    snapshot = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    assert tid in json.dumps(snapshot) and "_continuation_prepared" in json.dumps(snapshot)
    # Actual snapshot restore must preserve the preparation hold.
    workers.PENDING.clear()
    assert queue.restore_pending_from_snapshot() == 1
    assert budget_pause_fact(workers.PENDING[0]) is not None
    monkeypatch.setattr(tr, "write_task_result", real_write)
    replay = ca.admit_continuation("pred-1", action_nonce=NONCE)
    assert replay["ok"] and replay["successor_task_id"] == tid
    assert not workers.PENDING[0].get("_continuation_prepared")
    assert load_task_result(tmp_path, tid)["continuation_admission"]["binding"]["action_nonce"] == NONCE


@pytest.mark.parametrize("payload", ['[]\n', '{torn\n', '\udcff'])
def test_bad_mail_is_never_deleted_or_silently_dropped(tmp_path, monkeypatch, payload):
    from ouroboros.owner_mailbox import _mailbox_path
    from ouroboros.task_custody import settle_task_mailbox
    from supervisor.continuation_admission import admit_continuation

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, root_phase_checkpoint={}, child_ref_promotion={})
    path = _mailbox_path(tmp_path, "pred-1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.encode("utf-8", errors="surrogateescape"))
    assert not settle_task_mailbox(tmp_path, "pred-1", tmp_path)
    assert path.exists()
    assert not admit_continuation("pred-1", action_nonce=NONCE)["ok"]


def test_owner_mail_retention_failure_keeps_acknowledged_source(tmp_path, monkeypatch):
    from ouroboros import task_custody as tc
    from ouroboros.owner_mailbox import write_owner_message, acknowledge_transcript_entry, _mailbox_path

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, root_phase_checkpoint={}, child_ref_promotion={})
    write_owner_message(tmp_path, "exact correction", "pred-1", msg_id="owner-x")
    acknowledge_transcript_entry(tmp_path, "pred-1", {"msg_id": "owner-x"})
    monkeypatch.setattr(tc, "_write_custody_fields", lambda *a, **kw: None)  # lost publication
    assert not tc.settle_task_mailbox(tmp_path, "pred-1", tmp_path)
    assert "exact correction" in _mailbox_path(tmp_path, "pred-1").read_text()


def test_sleep_string_cannot_bypass_model_selection_policy(tmp_path, monkeypatch):
    from supervisor.budget_resume import grant_exact_budget_resume
    from supervisor.events_budget import budget_resume_dispatch_allowed

    q, _, _ = _install_queue(tmp_path, monkeypatch)
    assert grant_exact_budget_resume({}, {}, selected_by="sleep_wake")["error"] == "sleep_policy_required"
    forged = {"id": "child", "root_task_id": "root", "_budget_pause_resume": {"selected_by": "sleep_wake"}}
    assert not budget_resume_dispatch_allowed(q, forged)


def test_cold_sleep_retains_unknown_child_start_after_result_collection(tmp_path):
    from tests.test_model_sleep import _ctx, _result
    from ouroboros import delegate_custody as dc
    from ouroboros.model_sleep import request_sleep

    _result(tmp_path, "sleeper")
    assert dc.record_start_requested(tmp_path, task_id="collected-child", root_task_id="sleeper",
                                    invocation_id="uncertain", idempotency_key="key", route="test",
                                    request={"prompt": "work"})
    sleeper = _ctx(tmp_path)
    sleeper.task_metadata["project_id"] = "room"  # a top-level project sleeper: its cold sleep releases the lane
    with pytest.raises(ValueError, match="member_custody collected-child"):
        request_sleep(sleeper, {"tasks": [], "runs": [], "senders": [], "wake_at": ""}, "cold")
