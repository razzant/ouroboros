"""Peer writes reach inline Presence through the existing context-only mailbox."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros.loop_messages import _initialize_owner_directives, owner_authority_kinds
from ouroboros.loop_round_limits import _drain_incoming_messages
from ouroboros.owner_mailbox import drain_owner_entries, mail_read_state
from ouroboros.peer_roster import independent_message_target, independent_roots
from ouroboros.presence_runner import PresenceTurnGate, run_presence_turn
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.core import _forward_to_worker
from ouroboros.tools.registry import ToolContext
from tests.test_presence_runner import _admission, _event, _terminal

pytestmark = pytest.mark.serial


def _inline(root, task_id="inline-turn", **changes):
    fields = {"source": "presence", "_is_direct_chat": True,
              "metadata": {"presence": {"binding_id": "1" * 32}, "presence_event_identity": "event-identity"}}
    fields.update(changes)
    write_task_result(root, task_id, "running", **fields)
    return task_id


def _ctx(root, *, source_drive=None, task_id="peer-root"):
    return ToolContext(repo_dir=root, drive_root=source_drive or root, budget_drive_root=root,
                       task_id=task_id, task_metadata={})


_PEER_PROCESS = r'''
import json, os, sys
from pathlib import Path
root, repo, tid, sender = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
os.environ['OUROBOROS_IN_WORKER'] = '1'
from supervisor import queue, state
queue.DRIVE_ROOT = root
state.init(root)
from ouroboros.task_status import load_effective_task_result, wait_for_effective_tasks
from ouroboros.tools.core import _forward_to_worker
from ouroboros.tools.control_task_results import _get_task_result, _compact_child_projection
from ouroboros.tools.recent_tasks import _handle_recent_tasks
from ouroboros.tools.registry import ToolContext
ctx = ToolContext(repo_dir=repo, drive_root=sender, budget_drive_root=root,
                  task_id='peer-root', task_metadata={})
effective = load_effective_task_result(root, tid)
print(json.dumps({'forward': _forward_to_worker(ctx, tid, 'A peer found the missing context.'),
                  'get': _get_task_result(ctx, tid),
                  'recent': json.loads(_handle_recent_tasks(ctx)),
                  'wait': wait_for_effective_tasks(root, [tid], timeout_sec=0),
                  'compact': _compact_child_projection(tid, effective, None)}))
'''


def test_cross_process_peer_reaches_real_presence_drain_without_owner_authority(tmp_path, monkeypatch):
    """A real producer is live while a fresh interpreter writes/reads its result."""
    repo, data, sender = (tmp_path / name for name in ("repo", "data", "sender-private"))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))  # Parent and child share one install.
    for path in (repo, data, sender):
        path.mkdir()
    seen = {}

    class Agent:
        def handle_task(self, task):
            from ouroboros.agent import OuroborosAgent

            tid = task["id"]
            native = object.__new__(OuroborosAgent)
            native.env = SimpleNamespace(drive_root=data)
            native._persist_running_record(task)
            assert load_task_result(data, tid)["source"] == "presence"
            env = {**os.environ, "OUROBOROS_DATA_DIR": str(data), "OUROBOROS_APP_ROOT": str(tmp_path),
                   "OUROBOROS_REPO_DIR": str(repo), "PYTHONDONTWRITEBYTECODE": "1"}
            child = subprocess.run([sys.executable, "-c", _PEER_PROCESS, str(data), str(repo), tid, str(sender)],
                                   env=env, text=True, capture_output=True, timeout=45)
            assert child.returncode == 0, child.stdout + child.stderr
            seen.update(json.loads(child.stdout))
            assert "written to its mailbox" in seen["forward"]
            assert "unknown" in seen["forward"] and "not that its model read it" in seen["forward"]
            pending = drain_owner_entries(data, tid)
            assert len(pending) == 1
            entry = pending[0]
            assert entry["provenance"] == "independent_task"
            assert entry["source_task_id"] == "peer-root"
            assert owner_authority_kinds(pending) == []
            assert mail_read_state(data, tid, entry["msg_id"]) is False
            assert not (sender / "memory" / "owner_mailbox").exists()
            ctx = SimpleNamespace(task_id=tid, task_attempt=1, task_metadata=task["metadata"], _owner_directives=[])
            messages, events = [], queue.Queue()
            _initialize_owner_directives(ctx, messages)
            _drain_incoming_messages(messages, queue.Queue(), data, tid, events, set(), owner_ctx=ctx)
            assert ctx._owner_directives == []
            assert "A peer found the missing context." in str(messages)
            assert "[Message from independent task peer-root]" in str(messages)
            assert mail_read_state(data, tid, entry["msg_id"]) is True
            injected = [events.get_nowait() for _ in range(events.qsize())]
            assert any(row.get("provenance") == "independent_task" for row in injected)
            assert tid not in [row["task_id"] for row in independent_roots(data)["roots"]]
            for row in (seen["compact"], seen["wait"]["tasks"][tid], seen["recent"]["tasks"][0]):
                assert row["status"] == "running"
                assert row["execution_observation"]["kind"] == "presence"
                assert row["execution_observation"]["state"] == "unknown"
            assert '"execution_observation"' in seen["get"] and '"state": "unknown"' in seen["get"]
            assert seen["wait"]["all_terminal"] is False
            _terminal(data, task, "Context received")
            return [{"type": "presence_result", "outcome": "message", "text": "Context received"}]

    result = run_presence_turn(admission=_admission(), event=_event(), repo_dir=repo, drive_root=data,
                               agent_factory=lambda **_kwargs: Agent(), gate=PresenceTurnGate(2))
    assert result.text == "Context received"
    assert seen


def test_completion_between_admission_and_write_reports_retained_unread(tmp_path, monkeypatch):
    import ouroboros.owner_mailbox as mailbox

    tid = _inline(tmp_path)
    original = mailbox.write_task_message

    def settle_then_write(*args, **kwargs):
        write_task_result(tmp_path, tid, "completed", result="Finished before the write")
        return original(*args, **kwargs)

    monkeypatch.setattr(mailbox, "write_task_message", settle_then_write)
    out = _forward_to_worker(_ctx(tmp_path), tid, "late peer context")
    assert "retained_unread" in out and "no checkpoint will read it" in out
    assert "reads it at its next checkpoint" not in out
    assert load_task_result(tmp_path, tid)["status"] == "completed"


@pytest.mark.parametrize("changes", [
    {"source": "task"},
    {"_is_direct_chat": False},
    {"metadata": {"presence": {"binding_id": "1" * 32}}},
    {"metadata": {"presence": {}, "presence_event_identity": "event-identity"}},
])
def test_presence_metadata_alone_does_not_create_a_mailbox_target(tmp_path, changes):
    from ouroboros.task_status import load_effective_task_result

    tid = _inline(tmp_path, **changes)
    assert independent_message_target(tmp_path, tid, load_effective_task_result(tmp_path, tid)) is None
    assert "TASK_FORBIDDEN" in _forward_to_worker(_ctx(tmp_path), tid, "no")
    assert drain_owner_entries(tmp_path, tid) == []


def test_unknown_observation_is_disclosed_and_relay_is_still_ancestor_only(tmp_path):
    tid = _inline(tmp_path)
    refused = _forward_to_worker(_ctx(tmp_path), tid, "relayed", "unrelated-source")
    assert "relayed message reaches only your own descendants" in refused
    assert drain_owner_entries(tmp_path, tid) == []
    accepted = _forward_to_worker(_ctx(tmp_path), tid, "ordinary peer context")
    assert "unknown" in accepted and "independent_task, never owner text" in accepted
    assert "delivered" not in accepted and "it reads it" not in accepted
    assert len(drain_owner_entries(tmp_path, tid)) == 1


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_settled_presence_is_not_a_new_mailbox_target(tmp_path, status):
    tid = _inline(tmp_path)
    write_task_result(tmp_path, tid, status, result="settled")
    assert "TASK_NOT_ACTIVE" in _forward_to_worker(_ctx(tmp_path), tid, "no")
    assert drain_owner_entries(tmp_path, tid) == []


def test_cancellation_and_missing_sender_context_still_refuse(tmp_path, monkeypatch):
    import ouroboros.cancel_intents as cancel

    tid = _inline(tmp_path)
    assert "requires an active task context" in _forward_to_worker(_ctx(tmp_path, task_id=""), tid, "no")
    monkeypatch.setattr(cancel, "cancel_pending", lambda *args, **kwargs: True)
    assert "TASK_CANCEL_PENDING" in _forward_to_worker(_ctx(tmp_path), tid, "no")
    assert drain_owner_entries(tmp_path, tid) == []


@pytest.mark.parametrize("foreign", [True, False])
def test_presence_sender_ceiling_and_binding_are_not_widened(tmp_path, foreign):
    from tests.test_presence_own_work import BINDING, OTHER, _ceiling, _registry
    from tests.frozen_presence_capabilities import PresenceToolTarget

    tid = _inline(tmp_path, metadata={"presence": {"binding_id": OTHER if foreign else BINDING},
                                     "presence_event_identity": "event-identity"})
    selected = _ceiling(PresenceToolTarget("builtin", "forward_to_worker"))
    registry, _ = _registry(tmp_path, selected)
    # own_binding means this binding's promoted/follow-up work or own tree;
    # it never gave a Presence sender arbitrary access to inline turns.
    assert "PRESENCE" in registry.execute("forward_to_worker", {"task_id": tid, "message": "no"})
    assert drain_owner_entries(tmp_path, tid) == []
    registry, _ = _registry(tmp_path)
    assert "PRESENCE" in registry.execute("forward_to_worker", {"task_id": tid, "message": "no"})
    assert drain_owner_entries(tmp_path, tid) == []


def test_presence_origin_does_not_admit_a_now_pooled_or_redirected_target(tmp_path):
    from ouroboros.task_status import execution_owner_record, load_effective_task_result

    tid = "resumed-inline"
    _inline(tmp_path, tid, execution_owner=execution_owner_record(tmp_path, {"id": tid}, "pooled"))
    assert "TASK_FORBIDDEN" in _forward_to_worker(_ctx(tmp_path), tid, "no unlisted pooled target")
    assert drain_owner_entries(tmp_path, tid) == []
    tid = _inline(tmp_path, "exact-inline")
    effective = load_effective_task_result(tmp_path, tid)
    assert independent_message_target(tmp_path, tid, effective) is not None
    assert independent_message_target(tmp_path, tid, {**effective, "task_id": "different-turn"}) is None
    assert independent_message_target(tmp_path, tid, {**effective, "execution_observation": {}}) is None
