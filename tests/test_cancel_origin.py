"""A settled cancellation retains its recorded cause through result consumers."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros import cancel_intents
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.control_task_results import _get_task_result
from ouroboros.tools.join_ledger import _child_result_sha256
from tests._cancel_intents_shared import qenv as _qenv

qenv = _qenv


@pytest.mark.parametrize("queued", [False, True], ids=["missing-worker", "pending"])
def test_cancel_custody_keeps_origin_after_removing_intent(qenv, monkeypatch, queued):
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_kw: None)
    task = {"id": "cancelled", "chat_id": 0}
    if queued:
        qenv.q.PENDING.append(task)
    write_task_result(qenv.drive, task["id"], "scheduled" if queued else "running")
    intent = cancel_intents.request_cancel(qenv.drive, task["id"], source="http_single")

    assert qenv.tl.cancel_task_custody(task["id"]) == qenv.tl.CANCEL_CANCELLED

    stored = load_task_result(qenv.drive, task["id"])
    assert stored["cancel_origin"] == {
        "source": "http_single", "scope": "single",
        "request_id": intent["request_id"], "requested_at": intent["requested_at"],
    }
    assert "parent_decision" not in stored
    assert cancel_intents.active_intent(qenv.drive, task["id"]) is None


def test_cascade_origin_links_the_child_to_its_cancelled_root(qenv, monkeypatch):
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_kw: None)
    qenv.q.PENDING[:] = [
        {"id": "root", "chat_id": 0},
        {"id": "child", "chat_id": 0, "parent_task_id": "root", "root_task_id": "root"},
    ]
    for task in qenv.q.PENDING:
        write_task_result(qenv.drive, task["id"], "scheduled")
    cancel_intents.request_cancel(qenv.drive, "root", source="http_cascade", scope="cascade")

    assert qenv.tl.cancel_task_by_id("root")

    root = load_task_result(qenv.drive, "root")
    child = load_task_result(qenv.drive, "child")
    assert root["cancel_origin"]["source"] == "http_cascade"
    assert root["cancel_origin"]["scope"] == "cascade"
    assert child["cancel_origin"]["requested_by"] == "root"
    assert child["cancel_origin"]["reason"] == "subtree cancellation of root"
    assert child["parent_decision"] == "cancelled"


@pytest.mark.parametrize("queued", [False, True], ids=["pooled-missing-worker", "pooled-pending"])
def test_pooled_hardened_wrap_up_publishes_stop_now_cause(qenv, monkeypatch, queued):
    from pathlib import Path
    import subprocess

    from ouroboros.preflight_node import resolve_node
    from ouroboros.project_dialogue import _completion_verdict, completion_status_label
    from ouroboros.test_evidence import diagnostic_directory, write_record
    from supervisor import events

    task = {"id": "hardened-pooled", "chat_id": 7, "execution_owner": {"kind": "pooled"}}
    if queued:
        qenv.q.PENDING.append(task)
    write_task_result(qenv.drive, task["id"], "scheduled" if queued else "running",
                      chat_id=7, execution_owner=task["execution_owner"])
    original = cancel_intents.request_cancel(qenv.drive, task["id"], source="http_graceful",
        reason="owner requested finalize-then-stop", requested_stop_policy="finalize_then_cancel")
    hardened = cancel_intents.request_cancel(qenv.drive, task["id"], source="http_single",
                                            requested_stop_policy="immediate")
    assert hardened["request_id"] == original["request_id"]
    assert hardened["source"] == "http_graceful" and hardened["stop_policy"] == "immediate"
    published = []
    monkeypatch.setattr(events, "_finish_task_done_dispatch", lambda *_a, **kw: published.append(kw["task_done_event"]))
    assert qenv.tl.cancel_task_custody(task["id"], deliver=False) == qenv.tl.CANCEL_CANCELLED
    stored = load_task_result(qenv.drive, task["id"])
    assert cancel_intents.active_intent(qenv.drive, task["id"]) is None
    assert stored["execution_owner"]["kind"] == "pooled"
    assert stored["cancel_origin"]["stop_policy"] == "immediate"
    assert stored["cancel_origin"]["source"] == "http_graceful"
    # Rebuild the live publication from durable truth, as the task_done consumer does.
    events._handle_task_done({"task_id": task["id"], "status": "cancelled"},
                             SimpleNamespace(DRIVE_ROOT=qenv.drive, RUNNING={}, WORKERS={}))
    assert published[-1]["cancel_origin"] == stored["cancel_origin"]
    legacy = {**stored, "cancel_origin": {k: v for k, v in stored["cancel_origin"].items() if k != "stop_policy"}}
    rows = [stored, published[-1], legacy]
    causes = ["Stopped from the app (Stop now)"] * 2 + ["Stopped from the app (Wrap up)"]
    for row, cause in zip(rows, causes):
        assert completion_status_label(row, {}) == "Stopped"
        assert _completion_verdict(row, {}).rstrip(".") == cause
    node = resolve_node()
    assert node, "this publication proof requires Node"
    script = """import {taskOutcomeSeverity, taskReasonDetail} from './web/modules/log_events.js';
let text = ''; for await (const chunk of process.stdin) text += chunk;
console.log(JSON.stringify(JSON.parse(text).map(row => ({
  severity:taskOutcomeSeverity(row), cause:taskReasonDetail(row)
}))));"""
    checked = subprocess.run([node, "--input-type=module", "-e", script], input=json.dumps(rows),
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    views = json.loads(checked.stdout)
    assert [v["severity"] for v in views] == ["cancelled"] * 3
    assert [v["cause"].rstrip(".") for v in views] == causes
    write_record(diagnostic_directory(qenv.drive / "proof", "pooled-stop") / f"{queued}.json",
                 {"intent": hardened, "stored": stored, "published": published[-1], "js": views})


def test_history_terminal_row_retains_cancel_origin(tmp_path):
    origin = {"source": "http_single", "scope": "single", "request_id": "ci_recorded"}
    write_task_result(tmp_path, "task", "cancelled", cancel_origin=origin)
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "progress.jsonl").write_text(json.dumps({
        "ts": "2026-09-18T00:00:00Z", "task_id": "task", "content": "Working",
    }) + "\n", encoding="utf-8")
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "10"})))
    rows = json.loads(response.body.decode("utf-8"))["messages"]
    row = next(row for row in rows if row.get("task_id") == "task")
    assert row["task_terminal_status"] == "cancelled"
    assert row["cancel_origin"] == origin


def test_cancel_origin_is_current_fact_even_on_a_conditional_result_read(tmp_path):
    stored = write_task_result(tmp_path, "task", "cancelled", result="Stopped.")
    ctx = SimpleNamespace(drive_root=tmp_path, task_metadata={})
    before = _get_task_result(ctx, "task")
    origin = {"source": "agent_tool", "reason": "Хватит", "request_origin": {"kind": "agent_task", "task_id": "actor"}}
    write_task_result(tmp_path, "task", "cancelled", cancel_origin=origin)
    marker = f"\n\n[CANCELLED_BY] {json.dumps(origin, ensure_ascii=False)}"

    assert _get_task_result(ctx, "task") == before + marker
    conditional = _get_task_result(ctx, "task", known_result_sha256=_child_result_sha256(stored))
    assert '"result_unchanged": true' in conditional
    assert conditional.endswith(marker)


def test_rebuilt_task_done_carries_durable_cancel_origin(tmp_path, monkeypatch):
    from supervisor import events

    origin = {"source": "http_single", "request_id": "ci_recorded"}
    write_task_result(tmp_path, "task", "cancelled", cancel_origin=origin)
    published = []
    monkeypatch.setattr(events, "_finish_task_done_dispatch", lambda *_a, **kw: published.append(kw["task_done_event"]))
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, WORKERS={})

    events._handle_task_done({"task_id": "task", "status": "cancelled"}, ctx)

    assert published[0]["cancel_origin"] == origin
    rows = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert next(row for row in rows if row.get("type") == "task_done")["cancel_origin"] == origin


def test_ouroboros_cancel_task_speaks_its_producer_and_its_asker_through_custody(qenv, monkeypatch):
    """#1317 end to end: the real cancel_task tool mints the intent, custody
    settles it, and the stored origin speaks one truthful line. A parent that
    stops its own child ASKED -- it was not stopped with it; a foreign cancel
    names only that a task asked. The caller's own reason stays beside it."""
    import queue

    from ouroboros.project_dialogue import _completion_verdict
    from ouroboros.tools.join_ledger import _cancel_task

    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_kw: None)
    for task_id, parent in (("child", "parent"), ("stranger", "someone-else")):
        qenv.q.PENDING.append({"id": task_id, "chat_id": 0, "parent_task_id": parent, "root_task_id": parent})
        write_task_result(qenv.drive, task_id, "scheduled", parent_task_id=parent, root_task_id=parent,
                          delegation_role="subagent")
    caller = SimpleNamespace(task_id="parent", drive_root=qenv.drive, pending_events=[], event_queue=queue.Queue(),
                             task_metadata={"root_task_id": "parent", "delegation_role": "root"})
    _cancel_task(caller, "child", "the plan changed")
    _cancel_task(caller, "stranger", "not mine but stop it")
    for task_id in ("child", "stranger"):
        assert qenv.tl.cancel_task_custody(task_id) == qenv.tl.CANCEL_CANCELLED

    child, stranger = (load_task_result(qenv.drive, task_id) for task_id in ("child", "stranger"))
    assert child["cancel_origin"]["requested_by"] == "parent" and child["parent_decision"] == "cancelled"
    assert _completion_verdict(child, {}) == "Stopped by Ouroboros · the plan changed · Requested by its parent task."
    assert "requested_by" not in stranger["cancel_origin"]
    assert _completion_verdict(stranger, {}) == "Stopped by Ouroboros · not mine but stop it · Requested by a task."


def test_ouroboros_cancel_task_keeps_its_whole_reason_through_custody_and_receipt(qenv, monkeypatch):
    """#1317: the reason Ouroboros gives cancel_task is the record. Its tail past
    500 characters reaches the intent, the settled ``cancel_origin``, the parent
    decision and the receipt; only the ledger note, the tool's own reply and the
    owner line are previews, and each says so."""
    import queue

    from ouroboros import task_tree_ledger
    from ouroboros.project_dialogue import _completion_verdict
    from ouroboros.tools.join_ledger import _cancel_task
    from supervisor.cancel_publication import CANCEL_REASON_PREVIEW_NOTE
    from supervisor.terminal_delivery import build_unreviewed_salvage_event

    tail = "UNIQUE-TAIL-past-the-old-500-cut"
    reason = "the plan changed:\n" + "we no longer need this branch of the work " * 16 + tail
    whole = " ".join(reason.split())
    assert len(whole) > 600 and whole.endswith(tail)
    notes: list = []
    monkeypatch.setattr(task_tree_ledger, "tree_ledger_append", lambda _rid, _kind, text, **_kw: notes.append(text))
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_kw: None)
    qenv.q.PENDING.append({"id": "child", "chat_id": 7, "parent_task_id": "parent", "root_task_id": "parent"})
    write_task_result(qenv.drive, "child", "scheduled", parent_task_id="parent", root_task_id="parent",
                      delegation_role="subagent")
    caller = SimpleNamespace(task_id="parent", drive_root=qenv.drive, pending_events=[], event_queue=queue.Queue(),
                             task_metadata={"root_task_id": "parent", "delegation_role": "root"})

    reply = _cancel_task(caller, "child", reason)

    assert cancel_intents.active_intent(qenv.drive, "child")["reason"] == whole
    assert caller.event_queue.get_nowait()["reason"] == whole
    (note,) = notes
    for preview in (reply, note):
        assert tail not in preview and "more chars omitted" in preview
    assert qenv.tl.cancel_task_custody("child") == qenv.tl.CANCEL_CANCELLED
    build_unreviewed_salvage_event(qenv.drive, {"id": "child", "chat_id": 7}, "child",
                                   outcome="cancelled", settled_status="cancelled")
    child = load_task_result(qenv.drive, "child")
    assert child["cancel_origin"]["reason"] == whole
    assert child["parent_decision_reason"] == whole
    assert child["cancel_receipt"]["stop_reason"] == whole
    line = _completion_verdict(child, {})
    assert tail not in line and CANCEL_REASON_PREVIEW_NOTE.strip() in line
