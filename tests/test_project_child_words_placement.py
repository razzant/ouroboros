"""A row's task_id names its target, not its speaker, its placement or its end.

A steer the host refused for a child (an origin-addressed routing notice) is the
issuing chat's own standalone System row: counted, the newest arrival, never the
child's words or result. A child's own typed words are lines of its card: never
counted, never its final, one line each, the same words said again included. Only
the child's final concludes it, and a photo it delivers stands alone (DESIGN
"Unread"; the origin-addressed routing notices stay ordinary rows).

Every row comes from a real producer on its allowed branch: an acting child under
Cyber Pro (the only access that grants a child ``send_user_message``/``send_photo``;
the refusals below pin that) calls them through the registry's dispatch gate, the
owner-turn refusal is ``steering._handle_steer_task`` with the synthetic receipt id a
direct turn mints after its origin was routed, and all of them pass the real delivery
seam into a registered Project read by the real history endpoint. Those exact live
frames and history rows then drive the real chat.js consumer.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import shutil
from collections import deque
from types import SimpleNamespace

import pytest

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.gateway import history
from ouroboros.projects_registry import create_project, get_project
from ouroboros.task_results import write_task_result
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.utils import append_jsonl
from supervisor import events_chat_delivery as delivery
from supervisor import message_bus
from tests.group_run import run_in_group

pytestmark = pytest.mark.serial  # Access is read from the process environment

LINEAGE = {"delegation_role": "subagent", "parent_task_id": "root-1", "root_task_id": "root-1",
           "subagent_role": "researcher"}
REPO = pathlib.Path(__file__).resolve().parents[1]


class _Queue(list):
    put_nowait = put = list.append


@pytest.fixture
def room(tmp_path, monkeypatch):
    from supervisor import queue, state

    for module, names in ((state, ("DRIVE_ROOT", "STATE_PATH", "STATE_LAST_GOOD_PATH", "STATE_LOCK_PATH")),
                          (queue, ("DRIVE_ROOT", "QUEUE_SNAPSHOT_PATH"))):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    state.init(tmp_path)
    queue.init(tmp_path)
    monkeypatch.setattr(history, "_active_lifecycle_row", lambda _filter: None)
    chat_id = int(create_project(tmp_path, "racer")["chat_id"])
    frames = []
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7, "session_id": "s"})
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    host = SimpleNamespace(
        DRIVE_ROOT=tmp_path, append_jsonl=append_jsonl, bridge=bridge, send_with_budget=message_bus.send_with_budget,
        RUNNING={"root-1": {"task": {"id": "root-1"}}, "kid-1": {"task": {"id": "kid-1", **LINEAGE}}},
        PENDING=[], persist_queue_snapshot=lambda **_k: None,
    )
    # The child's durable result, as scheduling writes it: the history reader's lineage source.
    write_task_result(tmp_path, "kid-1", "running", **LINEAGE)

    def read() -> dict:
        response = asyncio.run(history.make_chat_history_endpoint(tmp_path)(
            SimpleNamespace(query_params={"chat_id": str(chat_id)})))
        assert response.status_code == 200, response.body
        return json.loads(response.body)

    def row_id(text: str) -> str:
        from ouroboros import dialogue_store as ds

        for entry in ds.iter_rows(tmp_path, "chat"):
            row = entry.row
            if text in str(row.get("text") or row.get("caption") or ""):
                return entry.history_id
        raise AssertionError(f"no stored row {text!r}")

    return SimpleNamespace(chat_id=chat_id, host=host, frames=frames, root=tmp_path, read=read, row_id=row_id,
                           revision=lambda: int(get_project(tmp_path, "racer")["visible_revision"]))


def _child_registry(room, monkeypatch, mode: str, profile: str = "acting_subagent"):
    """The child's own tool registry and context, gated by the real dispatch."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    repo, drive, worktree = (room.root / "child" / name for name in ("repo", "data", "wt"))
    for path in (repo, drive, worktree):
        path.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive, workspace_root=str(worktree), workspace_mode="self_worktree",
                      task_constraint=TaskConstraint(mode=profile, surface="self_worktree", write_root=str(worktree)),
                      task_id="kid-1", current_chat_id=room.chat_id, task_metadata=dict(LINEAGE),
                      event_queue=_Queue())
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx = ctx
    return registry, ctx, worktree


def _steer_refused(room, target: str = "kid-1") -> int:
    """An owner direct turn's later steer: its origin already routed, so the act carries
    the turn's own synthetic receipt id and the refusal is a chat row."""
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX
    from supervisor import steering

    before = room.revision()
    steering._handle_steer_task({
        "type": "steer_task", "routing_token": f"tok-{target}", "target_task_id": target,
        "message": "also check the README", "chat_id": room.chat_id,
        "client_message_id": f"{AGENT_RECEIPT_ID_PREFIX}tok-{target}", "issuer": {"kind": "owner_turn"},
        "ts": "2026-10-11T12:07:00+00:00"}, room.host)
    return room.revision() - before


@pytest.mark.parametrize("target, status, fields", [
    pytest.param("kid-1", "running", LINEAGE, id="running-child"),
    pytest.param("fin-1", "completed", {"root_phase_checkpoint": {"post_task_synthesis": "running"},
                                        "reason_code": "final_message"}, id="finalizing-root"),
    pytest.param("done-1", "completed", {"reason_code": "final_message"}, id="terminal-root"),
])
def test_a_routing_notice_borrows_nothing_from_the_task_it_refused(room, target, status, fields):
    """Neither lineage nor phase nor terminal truth: the notice is read exactly like one whose
    target the host does not know, so it can neither conclude nor reclassify that task."""
    write_task_result(room.root, target, status, **fields)
    assert _steer_refused(room, "gone-1") == 1
    assert _steer_refused(room, target) == 1
    page = room.read()
    unknown, notice = [row for row in page["messages"] if row.get("system_type") == "steer_not_delivered"]
    assert (unknown["task_id"], notice["task_id"]) == ("gone-1", target)
    identity = {"text", "ts", "task_id", "history_id", "history_position", "row_address"}
    assert {k: v for k, v in notice.items() if k not in identity} == \
        {k: v for k, v in unknown.items() if k not in identity}
    assert page["window"]["latest_message"] == {"history_id": notice["history_id"], "out_of_order": False}


def test_the_client_mirrors_the_hosts_origin_addressed_notices():
    """The client places rows by the same declared kinds the host keeps in the issuing chat."""
    import re

    from ouroboros.project_dialogue import ORIGIN_ADDRESSED_NOTICE_TYPES

    source = (REPO / "web" / "modules" / "chat_row_placement.js").read_text(encoding="utf-8")
    (members,) = re.findall(r"^export const ORIGIN_ADDRESSED_NOTICE_TYPES = new Set\(\[([^\]]*)\]\);$", source, re.MULTILINE)
    assert set(re.findall(r"'([a-z_]+)'", members)) == set(ORIGIN_ADDRESSED_NOTICE_TYPES)


def test_only_cyber_pro_lets_a_child_speak_to_the_owner(room, monkeypatch):
    """The reachability the placement below depends on, through the same dispatch gate."""
    for mode, profile, refusal in (("advanced", "acting_subagent", "ACTING_SUBAGENT_BLOCKED"),
                                   ("cyber_pro", "local_readonly_subagent", "LOCAL_READONLY_SUBAGENT_BLOCKED")):
        registry, ctx, _ = _child_registry(room, monkeypatch, mode, profile)
        assert refusal in registry.execute("send_user_message", {"text": "halfway"})
        assert ctx.event_queue == []
    registry, ctx, _ = _child_registry(room, monkeypatch, "cyber_pro")
    assert "sent to owner chat" in registry.execute("send_user_message", {"text": "halfway"})
    assert [frame["system_type"] for frame in ctx.event_queue] == ["proactive_message"]


def test_a_routing_notice_for_a_child_stands_alone_and_the_childs_words_stay_in_its_card(room, monkeypatch):
    def deliver(event: dict) -> int:
        before = room.revision()
        {"send_photo": delivery._handle_send_photo}.get(event["type"], delivery._handle_send_message)(
            event, room.host)
        return room.revision() - before

    def say(text: str, task_id: str = "root-1", **fields) -> int:
        return deliver({"type": "send_message", "chat_id": room.chat_id, "task_id": task_id, "text": text, **fields})

    registry, ctx, worktree = _child_registry(room, monkeypatch, "cyber_pro")
    assert say("Reading the brief.", is_progress=True) == 0
    assert say("ROOT_ANSWER: the scout is on it.") == 1
    assert say("Listing the repository root.", task_id="kid-1", is_progress=True) == 0
    # The same words again, right away and after other words: each send is its own row.
    for text in ("CHILD_WORDS_1: halfway through.", "CHILD_WORDS_2: found the README.",
                 "CHILD_WORDS_2: found the README.", "CHILD_WORDS_1: halfway through."):
        assert "sent to owner chat" in registry.execute("send_user_message", {"text": text})
        assert deliver(ctx.event_queue.pop()) == 0, "a child's own words change its card"
    assert _steer_refused(room) == 1, "the refusal is the owner's own conversation row"

    notice_read = room.read()
    (notice,) = [row for row in notice_read["messages"] if row.get("system_type") == "steer_not_delivered"]
    assert notice["task_id"] == "kid-1" and "Not delivered" in notice["text"]
    assert "delegation_role" not in notice and "parent_task_id" not in notice, \
        "the refused child is the notice's target, never its speaker"
    assert notice_read["window"]["latest_message"] == {"history_id": notice["history_id"], "out_of_order": False}, \
        "the counted notice is the newest standalone message, so reading the room means reading it"
    words = [row for row in notice_read["messages"] if str(row.get("text") or "").startswith("CHILD_WORDS")]
    assert [row["system_type"] for row in words] == ["proactive_message"] * 4
    assert len({row["history_id"] for row in words}) == len({row["ts"] for row in words}) == 4
    assert all(row.get("delegation_role") == "subagent" for row in words)

    (worktree / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 200)
    assert "photo accepted for delivery" in registry.execute("send_photo", {"file_path": str(worktree / "chart.png"),
                                                           "caption": "CHILD_PHOTO"})
    assert deliver(ctx.event_queue.pop()) == 1, "what a child delivers is shown alone and counts"
    assert say("CHILD_FINAL: root listed.", task_id="kid-1") == 0
    write_task_result(room.root, "kid-1", "completed", **LINEAGE)
    final_read = room.read()
    assert final_read["window"]["latest_message"] == {"history_id": room.row_id("CHILD_PHOTO"),
                                                      "out_of_order": False}

    rows = [row for row in final_read["messages"] if row.get("task_id") in ("root-1", "kid-1")]
    live = [frame for frame in room.frames if frame.get("type") == "chat"]
    fixture = room.root / "child-words-projection.json"
    fixture.write_text(json.dumps({"live": live, "history": rows,
                                   "running_history": notice_read["messages"]}), encoding="utf-8")
    node = shutil.which("node")
    assert node, "node is required for the real chat.js live/replay consumer"
    result = run_in_group([node, "--test", "--test-reporter=tap", "--test-name-pattern=child words",
                           "web/tests/chat_card_row_placement.test.js"],
                          cwd=REPO, env={**os.environ, "CHILD_WORDS_PROJECTION": str(fixture)}, timeout=60)
    assert result.returncode == 0 and "# pass 7\n" in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize("with_child_words", [False, True], ids=["notice-only", "actual-child-words"])
def test_notice_target_is_not_represented_work(tmp_path, monkeypatch, with_child_words):
    """A refused target alone buys no tool/result carrier; actual words still do."""
    from ouroboros import tool_call_log

    selected = []

    def evidence(_root, tasks):
        selected.append(set(tasks))
        return {task: {"legacy": {"calls": 1}, "coverage": {"shown": 1, "matched": 1}}
                for task in tasks}

    monkeypatch.setattr(tool_call_log, "replay_evidence_for_tasks", evidence)
    notice = {"task_id": "kid-1", "role": "system", "system_type": "steer_not_delivered",
              "text": "Not delivered", "ts": "2026-10-11T12:07:00+00:00"}
    rows = [notice.copy()]
    if with_child_words:
        rows.append({"task_id": "kid-1", "role": "assistant", "system_type": "proactive_message",
                     "text": "Still checking", "ts": "2026-10-11T12:06:00+00:00"})
    cache = {"kid-1": {"status": "running", "tool_calls": 1, **LINEAGE}}
    history._annotate_terminal_task_truth(rows, tmp_path, result_cache=cache)

    assert selected == [{"kid-1"} if with_child_words else set()]
    assert rows[0] == notice, "target identity lends neither task truth nor placement"
    carriers = [row for row in rows if row.get("system_type") == "task_evidence"]
    assert bool(carriers) is with_child_words
    if with_child_words:
        assert rows[1]["delegation_role"] == "subagent"
        assert rows[1]["task_phase"] == "unfinished"
