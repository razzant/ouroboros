"""The owner's words ride down the task tree by value.

``schedule_subagent`` computes them once from what the parent holds and carries them
in the existing tree-origin channel (``origin_metadata``) and the child's requested
record. A child passes on exactly what it inherited, so a grandchild sees the root's
words unchanged. The child's own owner corpus stays its assignment: the root's words
are a fact in its view, never its owner requirements or an owner-door stamp.
Every rule is checked in both directions.
"""
from __future__ import annotations

import ast
import json
import pathlib
import queue

import pytest

from ouroboros import owner_words as ow
from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.dialogue_provenance import run_origin
from ouroboros.loop_delivery import _effective_delivery_criteria
from ouroboros.loop_messages import _initialize_owner_directives, _record_owner_directive
from ouroboros.task_results import load_task_result
from ouroboros.tools import control
from ouroboros.tools.control_scheduling import _schedule_task
from ouroboros.tools.registry import ToolContext
from supervisor.task_dispatch import build_scheduled_task_payload
from tests.test_available_subagents_runtime import _api_row, _settings

REPO = pathlib.Path(__file__).resolve().parents[1]
ASKED = "Build the quarterly report\nwith the regional split"
LATER = "Add the charts too"
ORIGIN_REF = {"chat_id": 1, "ts": "2026-10-04T05:00:00+00:00", "client_message_id": "msg-1"}


@pytest.fixture(autouse=True)
def _one_api_actor(monkeypatch):
    monkeypatch.setattr(control, "load_settings", lambda: _settings(_api_row()))
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "4")


def _root(tmp_path, *, metadata=None, owner_said=True):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, current_task_type="task")
    ctx.task_id, ctx.event_queue = "root", queue.Queue()
    ctx.task_metadata = {"root_task_id": "root", **(metadata if metadata is not None else
                                                      {"origin_message_ref": dict(ORIGIN_REF)})}
    ctx.task_contract = build_task_contract({"id": "root", "type": "task", "text": ASKED})
    if owner_said:
        _initialize_owner_directives(ctx, [{"role": "user", "content": ASKED}])
        _record_owner_directive(ctx, source="owner_mailbox", content=LATER, msg_id="mail-1")
    return ctx


def _schedule(ctx, objective="Collect the Q3 figures", **extra):
    """One real schedule: the tool's event, its requested record and the supervisor's payload."""
    result = _schedule_task(ctx, subagent_id="api-builder", objective=objective,
                            expected_output="figures", memory_mode="empty", **extra)
    assert not result.startswith("⚠️"), result
    event = ctx.event_queue.get_nowait()
    stored = load_task_result(ctx.budget_drive_root or ctx.drive_root, event["task_id"])
    payload = build_scheduled_task_payload({**event, "tid": event["task_id"], "parent_id": ctx.task_id,
                                            "text": event["objective"], "desc": event["objective"]})
    return event, stored, payload


def _child(tmp_path, payload):
    """The child's run as its worker builds it: metadata from the payload, corpus from its first user turn."""
    ctx = ToolContext(repo_dir=tmp_path, drive_root=pathlib.Path(payload["drive_root"]),
                      budget_drive_root=str(tmp_path), current_task_type="task")
    ctx.task_id, ctx.task_depth, ctx.event_queue = payload["id"], payload["depth"], queue.Queue()
    ctx.task_metadata, ctx.task_contract = dict(payload["metadata"]), payload["task_contract"]
    _initialize_owner_directives(ctx, [{"role": "user", "content": payload["text"]}])
    return ctx


def _texts(rows):
    return [row["text"] for row in rows]


def _holds(value, text):
    """Whether ``text`` appears in the JSON form of ``value`` (its own JSON escaping, newlines included)."""
    return json.dumps(text, ensure_ascii=False)[1:-1] in json.dumps(value, ensure_ascii=False, default=str)


def test_the_owner_words_ride_the_event_the_record_and_the_payload(tmp_path):
    event, stored, payload = _schedule(_root(tmp_path))
    assert _texts(event["origin_metadata"][ow.FIELD]) == [ASKED, LATER]
    assert ow.ABSENT_FIELD not in event["origin_metadata"]
    assert _texts(stored[ow.FIELD]) == [ASKED, LATER]  # the requested record any reader projects
    assert _texts(payload["metadata"][ow.FIELD]) == [ASKED, LATER]  # task.metadata of the child
    assert stored[ow.FIELD] == event["origin_metadata"][ow.FIELD] == payload["metadata"][ow.FIELD]
    first = stored[ow.FIELD][0]
    assert first["source"] == "initial_user" and first["task_id"] == "root" and first["ts"] == ORIGIN_REF["ts"]
    # The child's view renders exactly these rows; a child scheduled without the field shows nothing.
    from ouroboros.memory_view import _owner_words_block

    block = _owner_words_block(payload)
    assert block.startswith("## Words of my human that caused this work (verbatim)\n") and "  " + LATER in block
    bare = {**payload, "metadata": {key: value for key, value in payload["metadata"].items() if key != ow.FIELD}}
    assert _owner_words_block(bare) == ""


def test_a_grandchild_receives_the_root_words_unchanged(tmp_path):
    _event, _stored, payload = _schedule(_root(tmp_path))
    child = _child(tmp_path, payload)
    grand_event, grand_stored, grand_payload = _schedule(child, objective="Check one region")
    assert grand_payload["metadata"][ow.FIELD] == payload["metadata"][ow.FIELD]
    assert grand_stored[ow.FIELD] == grand_event["origin_metadata"][ow.FIELD] == payload["metadata"][ow.FIELD]
    assert not _holds(grand_payload["metadata"][ow.FIELD], "Collect the Q3 figures")
    # The child's own corpus would give its assignment; what it passes on is what it inherited.
    assert _texts(ow.directive_owner_rows(child)) == []
    child.task_metadata.pop(ow.FIELD)
    child.task_metadata[ow.ABSENT_FIELD] = "not_recorded"
    _e, _s, orphan = _schedule(child, objective="Check another region")
    assert ow.FIELD not in orphan["metadata"] and orphan["metadata"][ow.ABSENT_FIELD] == "not_recorded"


def test_a_consciousness_root_carries_the_raw_marker_and_no_words(tmp_path):
    ctx = _root(tmp_path, metadata={"initiator": "consciousness", "usage_category": "consciousness_task"},
                owner_said=False)
    _initialize_owner_directives(ctx, [{"role": "user", "content": "A wake decided to look at the logs"}])
    event, stored, payload = _schedule(ctx)
    assert event["origin_metadata"][ow.ABSENT_FIELD] == "initiator=consciousness"
    assert ow.FIELD not in event["origin_metadata"] and ow.FIELD not in stored
    assert payload["metadata"][ow.ABSENT_FIELD] == "initiator=consciousness"
    assert payload["metadata"]["initiator"] == "consciousness"  # the consciousness origin still rides along
    assert ow.ABSENT_FIELD not in _schedule(_root(tmp_path))[2]["metadata"]


def test_a_swarm_roots_words_are_the_doors_text_not_the_host_notice(tmp_path, monkeypatch):
    """The host prefixes its [SWARM_INITIATIVE] notice to a Swarm root's first user turn; the
    door kept the owner's own text, and that text is what a child reads as my human's words."""
    from ouroboros.context import build_user_content
    from ouroboros.review_evidence_sections import _owner_content_projection

    monkeypatch.setattr("ouroboros.config.get_review_enforcement", lambda: "advisory")
    door = {"origin_message_ref": dict(ORIGIN_REF), "origin_message_text": ASKED,
            "force_plan": True, "force_plan_source": "swarm"}
    turn = build_user_content({"text": ASKED, "metadata": door})
    assert "[SWARM_INITIATIVE]" in _owner_content_projection(turn)  # the root's own first turn carries the notice
    swarm = _root(tmp_path, metadata=door, owner_said=False)
    _initialize_owner_directives(swarm, [{"role": "user", "content": turn}])
    _event, stored, payload = _schedule(swarm)
    assert _texts(payload["metadata"][ow.FIELD]) == [ASKED] and _texts(stored[ow.FIELD]) == [ASKED]
    assert payload["metadata"][ow.FIELD][0]["ref"] == "chat 1 / msg-1"  # still the door's stamp
    assert "[SWARM_INITIATIVE]" not in ow.owner_words_text(swarm)
    assert _holds(swarm._owner_directives, "[SWARM_INITIATIVE]")  # the root's corpus itself is unchanged
    # The other side: a door that kept no text leaves the corpus row's own projection.
    bare = _root(tmp_path, metadata={key: value for key, value in door.items() if key != "origin_message_text"},
                 owner_said=False)
    _initialize_owner_directives(bare, [{"role": "user", "content": turn}])
    assert _texts(ow.directive_owner_rows(bare)) == [_owner_content_projection(turn)]


def test_the_child_corpus_stays_its_assignment_and_no_owner_door(tmp_path):
    root = _root(tmp_path)
    _event, _stored, payload = _schedule(root)
    child = _child(tmp_path, payload)
    assert [row["source"] for row in child._owner_directives] == ["initial_text"]
    assert not _holds(child._owner_directives, ASKED) and _holds(child._owner_directives, "Collect the Q3 figures")
    assert run_origin({"metadata": child.task_metadata})["owner_ingress"] is False
    assert run_origin({"metadata": root.task_metadata})["owner_ingress"] is True
    assert not _holds(_effective_delivery_criteria(child), ASKED)
    assert not _holds(_effective_delivery_criteria(child), LATER)
    assert _holds(_effective_delivery_criteria(root), ASKED) and _holds(_effective_delivery_criteria(root), LATER)
    # The words a reviewer or a session of this run sees: the child's inherited, the root's own corpus.
    assert ASKED in ow.owner_words_text(child) and LATER in ow.owner_words_text(child)
    assert "Collect the Q3 figures" not in ow.owner_words_text(child)
    assert ASKED in ow.owner_words_text(root) and "from task root" in ow.owner_words_text(root, audience="session")


def test_the_parent_reads_the_childs_words_from_the_childs_record(tmp_path):
    from ouroboros.subagent_work_order import _source_task_from_context
    from ouroboros.task_results import write_task_result

    root = _root(tmp_path)
    _event, stored, _payload = _schedule(root)
    seen = _source_task_from_context(root, stored["task_id"])
    assert seen[ow.FIELD] == stored[ow.FIELD]  # the parent's own metadata carries no field
    root.task_metadata[ow.FIELD] = [{"text": "The parent's own inherited words", "source": "initial_user"}]
    assert _source_task_from_context(root, stored["task_id"])[ow.FIELD] == stored[ow.FIELD]
    write_task_result(tmp_path, "nofield", "requested", description="a child scheduled before the field")
    assert ow.FIELD not in _source_task_from_context(_root(tmp_path), "nofield")


def test_schedule_task_stays_within_the_function_ceiling():
    tree = ast.parse((REPO / "ouroboros" / "tools" / "control_scheduling.py").read_text(encoding="utf-8"))
    node = next(item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef) and item.name == "_schedule_task")
    assert node.end_lineno - node.lineno + 1 <= 300
