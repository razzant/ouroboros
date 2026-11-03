"""Every consumer of a predecessor carries the same brief; only the successor carries the notes row.

A Continue root's frozen prefix holds the compact brief and the exact reader of the
predecessor's record; the predecessor mind's selected account follows the assignment
as ONE foldable host prose row, only for the task that materialized that predecessor
at its own start. A shared or declared helper receives the pure brief and reader
through the contract spread; the context assembler re-expands nothing, and a helper's
own current review authority still reaches it. No role string decides any of this.
"""
from copy import deepcopy
import json

import pytest

from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
from tests.test_review_decision_notes import NOTE, first_plan, harness as _harness, notes_for
from tests.test_review_view_continue import successor
from tests.test_review_view_integration import _apply, _capture

harness = _harness


def _continued(harness):
    ctx = first_plan(harness)
    sub = harness.install({})
    _apply(ctx, _capture(ctx), note=NOTE, review_notes=notes_for(ctx))
    env, root = successor(ctx)
    return ctx, sub, env, root


def _helper(root, selection, task_id="helper-child"):
    from ouroboros.tools.control_scheduling import _build_child_subagent_contract

    parent = {**deepcopy(root["task_contract"]), "input_sources": selection}
    contract = _build_child_subagent_contract({"tid": task_id, "parent_contract": parent})
    return {"id": task_id, "type": "task", "text": "check the carried brief", "delegation_role": "subagent",
            "parent_task_id": root["id"], "root_task_id": root["id"], "input_sources": selection,
            "configured_subagent": {"route": {"kind": "api_model"}}, "task_contract": contract}


def _notes_rows(messages):
    from ouroboros.main_context_authority import PREDECESSOR_NOTES_NAME

    return [row for row in messages if row.get(HOST_CONTEXT_KIND_KEY) == PREDECESSOR_NOTES_NAME]


@pytest.mark.parametrize("selection", ["shared", "declared"])
def test_helpers_keep_the_brief_and_reader_while_their_own_review_authority_stays(harness, selection):
    from ouroboros.context import _task_authority_projection
    from ouroboros.main_context_authority import predecessor_notes_message
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control import _get_task_result
    from ouroboros.tools.registry import ToolContext
    from tests.test_plan_task_tool import _call

    ctx, sub, env, root = _continued(harness)
    root_runtime = json.dumps(_task_authority_projection(env, root), ensure_ascii=False)
    assert NOTE not in root_runtime and "historical_review_context" not in root_runtime
    assert NOTE in predecessor_notes_message(root, drive_root=ctx.drive_root)["content"]
    child = _helper(root, selection)
    canonical = deepcopy(child)
    before = {tid: load_task_result(ctx.drive_root, tid) for tid in (ctx.task_id, root["id"])}
    runtime = _task_authority_projection(env, child)
    assert NOTE not in json.dumps(runtime, ensure_ascii=False)
    assert predecessor_notes_message(child, drive_root=ctx.drive_root) is None, "a spread brief carries no notes row"
    brief = child["task_contract"]["predecessor_authority"]
    assert runtime["task_contract"]["predecessor_authority"] == brief
    assert brief["source"]["task_id"] == ctx.task_id and brief["source"]["arguments"]["include_authority"]
    assert child == canonical and before == {tid: load_task_result(ctx.drive_root, tid) for tid in before}
    # The brief's reader resolves the full predecessor result for the helper itself.
    reader_ctx = ToolContext(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root, task_id=child["id"],
                             task_contract=child["task_contract"], task_metadata=dict(child))
    payload = json.loads(_get_task_result(reader_ctx, **brief["source"]["arguments"]))
    assert payload["status"] == "available" and payload["source"]["task_id"] == ctx.task_id
    assert payload["authority"]["task_contract"] and payload["authority"]["status"] == "failed"
    # A helper's OWN current plan authority is still projected for it.
    own = {**child, "id": "helper-own-plan"}
    _call(harness.make_ctx(task_id=own["id"]), goal="The helper's own plan.", affected_paths=[],
          plan="A free planning-organ revision of the helper's own work.")
    own_runtime = _task_authority_projection(env, own)
    assert own_runtime["plan_review_authority"]["source"]["arguments"]["task_id"] == own["id"]
    assert NOTE not in json.dumps(own_runtime, ensure_ascii=False)


@pytest.mark.parametrize("selection", ["shared", "declared"])
def test_assembled_helper_context_carries_the_brief_not_the_notes(harness, tmp_path, selection):
    from ouroboros.context import build_llm_messages
    from tests.test_doc_context import _make_env_and_memory

    ctx, _, _, root = _continued(harness)
    env, memory = _make_env_and_memory(tmp_path / "assembly")
    child = {**_helper(root, selection), "budget_drive_root": str(ctx.drive_root)}
    messages, _ = build_llm_messages(env=env, memory=memory, task=child)
    rendered = json.dumps(messages, ensure_ascii=False)
    assert NOTE not in rendered and not _notes_rows(messages)
    assert child["task_contract"]["predecessor_authority"]["authority_sha256"] in rendered
    assert "get_task_result" in rendered
    root_messages, _ = build_llm_messages(env=env, memory=memory, task={**root, "budget_drive_root": str(ctx.drive_root)})
    system_text = "\n".join(block["text"] for block in root_messages[0]["content"])
    assert NOTE not in system_text, "the notes never sit in the frozen prefix"
    (note,) = _notes_rows(root_messages)
    assert root_messages.index(note) == 2 and NOTE in note["content"], "the first row after the assignment"
    assert json.dumps(root["text"], ensure_ascii=False)[1:-1] in json.dumps(root_messages[1], ensure_ascii=False)


@pytest.mark.parametrize("shape", [
    {"delegation_role": "root"},
    {"delegation_role": "", "root_task_id": ""},
    {"delegation_role": "subagent", "parent_task_id": "some-parent", "root_task_id": "some-parent"},
], ids=["root", "blank-role", "subagent"])
@pytest.mark.parametrize("edge", ["top-level", "contract-only"])
def test_only_the_materialized_start_carries_the_notes_row(harness, shape, edge):
    """The row follows the task that materialized its named predecessor at start (the top-level
    authority with the exact pointer), whatever its role; a contract-only edge (the spread a
    helper or a restored row carries) yields the brief and reader alone."""
    from ouroboros.context import _task_authority_projection
    from ouroboros.main_context_authority import predecessor_notes_message

    ctx, _, env, root = _continued(harness)
    task = {**deepcopy(root), **shape}
    if edge == "contract-only":
        task.pop("predecessor_authority")
        assert task["task_contract"]["predecessor_authority"]["source"]["task_id"] == ctx.task_id
    runtime = json.dumps(_task_authority_projection(env, task), ensure_ascii=False)
    assert "historical_review_context" not in runtime and NOTE not in runtime
    note = predecessor_notes_message(task, drive_root=ctx.drive_root)
    assert (note is not None) == (edge == "top-level")
    if note is not None:
        assert NOTE in note["content"] and f"Task {ctx.task_id}'s own working notes" in note["content"]
