"""A declared child carries the owner's words that caused its work (BIBLE P1), on every path.

``schedule_subagent`` records ``DECLARED_WORDS_FACT`` beside the words of a declared child; with
it the child's own first input (API child or nanny), its session's work order and a direct
delegate start carry the words verbatim, or the host's marker that none are recorded, and its
receipt names them included. A grandchild carries the root's words. A declared task scheduled
before the fact keeps the work order it started with: the same bytes and fingerprint, so its
recovery after an update finds no ``work_order_mismatch`` and a range read of its source no
``source_digest_mismatch``. A shared child keeps its bytes and carries no fact. The words block
is rendered from the task's frozen record, ahead of the dated runtime facts, so it is the same
bytes on every rebuild. Every rule is checked in both directions.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros import owner_words as ow
from ouroboros.subagent_work_order import (
    DECLARED_WORDS_FACT,
    assignment_instructions,
    canonical_work_order_source,
    compile_external_work_order,
    input_source_selection_receipt,
    work_order_fingerprint,
)
from tests.test_doc_context import _make_env_and_memory
from tests.test_helper_owner_words_flow import ASKED, LATER, _child, _root, _schedule
from tests.test_helper_owner_words_flow import _one_api_actor as _one_api_actor  # noqa: F401  (autouse fixture)

SESSION_HEADING, CHILD_HEADING = ow._HEADINGS["session"], ow._HEADINGS["child"]
WORDS_ITEM = "the owner's words that caused this work"
SAID = "Weigh both designs\n## not a heading of the work order\n  keep this indent  "
WORDS = [{"text": SAID, "source": "initial_user", "carrier": "ctx", "task_id": "root",
          "ts": "2026-10-04T05:00:00+00:00", "ref": "chat 1 / msg-1"}]
# A declared session child scheduled before the fact, carrying the words every child has carried since they
# existed, and its work order as the host rendered it then (base 28452e14f): the receipt names them omitted.
OLDER = {
    "id": "kid2", "parent_task_id": "root", "root_task_id": "root", "delegation_role": "subagent",
    "objective": "Take a position on design A",
    "context": "Design A lives in docs/a.md.", "expected_output": "A position with its reasons",
    "configured_subagent": {"route": {"kind": "agent_session"}, "selected_subagent_id": "codex-row",
                            "config_fingerprint": "cfg-1"},
    "task_contract": {"objective": "Take a position on design A", "expected_output": "A position with its reasons",
                      "input_sources": "declared", "context": "Design A lives in docs/a.md."},
    "metadata": {ow.FIELD: WORDS},
}
OLDER_SHA, OLDER_CHARS = "7940b64d86c36e029d9c354d009c56a4efaee4608e8a022846c1ac5a4ab33e13", 2543
# The same child shared, as the host rendered it before this change: the words after the parent context.
SHARED_SHA, SHARED_CHARS = "8df17d89e966c326aed56aa161f61059d37348873302aa5c05dafde2eb039dba", 754
MARKER = "No words of my human are recorded for this work (host marker: initiator=consciousness)."


def _with_fact(task):
    return {**task, "metadata": {**task["metadata"], DECLARED_WORDS_FACT: True}}


def _flat_ctx(tmp_path, task):
    """The child's own run as a source reader sees it: flat metadata, its contract, an empty store."""
    flat = {key: value for key, value in task.items() if key not in ("task_contract", "metadata")}
    return SimpleNamespace(task_id=task["id"], task_metadata={**flat, **task["metadata"]},
                           task_contract=dict(task["task_contract"]), drive_root=tmp_path, budget_drive_root=tmp_path)


def _first_input(tmp_path, payload):
    """The dynamic block of a child's first input, composed as its worker composes it."""
    from ouroboros import context

    env, memory = _make_env_and_memory(tmp_path / "ctx")
    return context._capture_context_core(env, memory, payload, None, None).dynamic_text


def test_a_task_scheduled_before_the_fact_keeps_its_bytes_recovery_and_range_reads(tmp_path):
    from ouroboros.delegate_recovery import _successor_binding_mismatch, authority_fingerprint_from_task

    rendered = compile_external_work_order(OLDER)
    assert work_order_fingerprint(OLDER) == OLDER_SHA and len(rendered) == OLDER_CHARS
    assert SESSION_HEADING not in rendered and "Weigh both designs" not in rendered
    assert [item for item in input_source_selection_receipt(OLDER)["omitted_automatic"] if item.startswith(WORDS_ITEM)]
    newer = _with_fact(OLDER)
    assert work_order_fingerprint(newer) != OLDER_SHA and "Weigh both designs" in compile_external_work_order(newer)
    # A restart after the update re-proves the binding the nanny recorded before it.
    row = {"authority_fingerprint": authority_fingerprint_from_task(OLDER), "actor_binding_source": "task_snapshot",
           "config_fingerprint": "cfg-1", "selected_subagent_id": "codex-row", "work_order_fingerprint": OLDER_SHA}
    assert _successor_binding_mismatch(row, OLDER) == ""
    assert _successor_binding_mismatch(row, newer) == "work_order_mismatch"
    assert authority_fingerprint_from_task(newer) == row["authority_fingerprint"]  # the fact grants no authority
    # A range read verifies against the bytes recorded before the update.
    request = {"complete_sha256": OLDER_SHA, "complete_chars": OLDER_CHARS,
               "source": {"kind": "task_result", "task_id": "kid2", "tool": "get_task_result",
                          "arguments": {"task_id": "kid2", "include_authority": True, "include_work_order_source": True},
                          "projection": "canonical_work_order"}}
    assert canonical_work_order_source(_flat_ctx(tmp_path, OLDER), request) == (rendered, "")
    assert canonical_work_order_source(_flat_ctx(tmp_path, newer), request) == ("", "source_digest_mismatch")
    # Its direct starts keep carrying none.
    marker = "HOST TASK CONTRACT AUTHORITY (normalized JSON; predecessor is a brief):\n"
    older_ctx, newer_ctx = _flat_ctx(tmp_path, OLDER), _flat_ctx(tmp_path, newer)
    assert assignment_instructions(older_ctx).startswith(marker) and SAID not in assignment_instructions(older_ctx)
    assert assignment_instructions(newer_ctx).endswith("\n\n" + SESSION_HEADING + "\nHost fact, carried by value "
                                                       "from task root.\n[2026-10-04T05:00:00+00:00 · owner · "
                                                       "initial_user of task root · chat 1 / msg-1]\n" + SAID)


def test_the_receipt_moves_only_the_words_between_its_lists():
    older, newer = input_source_selection_receipt(OLDER), input_source_selection_receipt(_with_fact(OLDER))
    moved = [item for item in newer["included"] if item not in older["included"]]
    dropped = [item for item in older["omitted_automatic"] if item not in newer["omitted_automatic"]]
    assert len(moved) == 1 and moved[0].startswith(WORDS_ITEM) and "verbatim" in moved[0]
    assert len(dropped) == 1 and dropped[0].startswith(WORDS_ITEM)
    rest = ("included", "omitted_automatic", "later_inputs")
    assert {k: v for k, v in newer.items() if k not in rest} == {k: v for k, v in older.items() if k not in rest}
    # the same fact also names the knowledge index a coordinating declared child meets later; older bytes stay
    assert newer["later_inputs"].startswith(older["later_inputs"]) and "knowledge index" not in older["later_inputs"]
    assert "knowledge index" in newer["later_inputs"] and "knowledge_list" in newer["later_inputs"]
    assert newer["included"].index(moved[0]) == older["included"].index(
        "explicit assignment, question, evidence and normalized task authority") + 1  # beside the assignment
    shared = {**_with_fact(OLDER), "task_contract": {"objective": "Take a position on design A"}}
    assert input_source_selection_receipt(shared) == {}  # a shared task has no receipt, fact or not


def test_a_shared_child_keeps_its_bytes_and_carries_no_fact(tmp_path):
    shared = {**OLDER, "task_contract": {key: value for key, value in OLDER["task_contract"].items()
                                         if key != "input_sources"}}
    rendered = compile_external_work_order(shared)
    assert work_order_fingerprint(shared) == SHARED_SHA and len(rendered) == SHARED_CHARS and SAID in rendered
    assert compile_external_work_order(_with_fact(shared)) == rendered  # a stray fact changes nothing there
    _event, stored, payload = _schedule(_root(tmp_path))
    assert DECLARED_WORDS_FACT not in payload["metadata"] and DECLARED_WORDS_FACT not in stored
    _event, stored, declared = _schedule(_root(tmp_path), input_sources="declared")
    assert declared["metadata"][DECLARED_WORDS_FACT] is True and stored[DECLARED_WORDS_FACT] is True


@pytest.mark.parametrize("kind", ["api_model", "agent_session"])
def test_a_declared_child_carries_the_words_on_every_path(tmp_path, kind):
    from ouroboros.memory_view import _owner_words_block

    _event, _stored, payload = _schedule(_root(tmp_path), input_sources="declared",
                                         context="Figures live in reports/q3/.")
    snapshot = payload["configured_subagent"]
    payload["configured_subagent"] = {**snapshot, "route": {**snapshot["route"], "kind": kind}}
    # 1. Its own first input (API child or nanny): the block a shared child reads, right after the receipt.
    dynamic = _first_input(tmp_path, payload)
    block = _owner_words_block(payload)
    assert block.startswith(CHILD_HEADING + "\n") and "\n  Add the charts too" in block and block in dynamic
    assert dynamic.startswith("## Input source selection") and dynamic.index(block) < dynamic.index("## Runtime context")
    assert "Figures live in reports/q3/." in dynamic  # the parent's authored context, as before
    # 2. The work order its session leaf receives: the words after the parent context.
    rendered = compile_external_work_order(payload)
    section = rendered.split("PARENT CONTEXT / REFERENCES\n", 1)[1].split("\n\nEXPECTED OUTPUT", 1)[0]
    context_part, words = section.split("\n\n", 1)
    assert context_part.endswith("\nFigures live in reports/q3/.") and words.startswith(SESSION_HEADING)
    assert ASKED in words and words.endswith(LATER)
    # 3. A direct delegate start from it appends the same words after the host contract.
    direct = assignment_instructions(_child(tmp_path, payload))
    assert direct.split("\n\n", 1)[1] == words


def test_a_grandchild_of_a_declared_child_carries_the_root_words(tmp_path):
    _event, _stored, payload = _schedule(_root(tmp_path), input_sources="declared")
    _event, _stored, grand = _schedule(_child(tmp_path, payload), objective="Check one region")
    assert grand["task_contract"]["input_sources"] == "declared" and grand["metadata"][DECLARED_WORDS_FACT] is True
    assert grand["metadata"][ow.FIELD] == payload["metadata"][ow.FIELD]
    first = _first_input(tmp_path, grand).split(CHILD_HEADING, 1)[1].split("\n\n## Runtime context", 1)[0]
    order = compile_external_work_order(grand).split(SESSION_HEADING, 1)[1].split("\n\nEXPECTED OUTPUT", 1)[0]
    for words in (first, order):
        assert "from task root." in words and "Build the quarterly report" in words and "Add the charts too" in words
        assert "Collect the Q3 figures" not in words  # the child's assignment is not the owner's words


def test_a_consciousness_root_gives_its_declared_child_the_absence_marker(tmp_path):
    from ouroboros.loop_messages import _initialize_owner_directives

    wake = _root(tmp_path, metadata={"initiator": "consciousness", "usage_category": "consciousness_task"},
                 owner_said=False)
    _initialize_owner_directives(wake, [{"role": "user", "content": "A wake decided to look at the logs"}])
    _event, _stored, payload = _schedule(wake, input_sources="declared")
    assert payload["metadata"][ow.ABSENT_FIELD] == "initiator=consciousness" and ow.FIELD not in payload["metadata"]
    dynamic = _first_input(tmp_path, payload)
    assert dynamic.index("## Input source selection") < dynamic.index(MARKER) < dynamic.index("## Runtime context")
    assert CHILD_HEADING not in dynamic and "A wake decided" not in dynamic
    rendered = compile_external_work_order(payload)
    assert f"OBJECTIVE\nCollect the Q3 figures\n\n{MARKER}\n\nEXPECTED OUTPUT" in rendered
    assert SESSION_HEADING not in rendered
    assert assignment_instructions(_child(tmp_path, payload)).endswith("\n\n" + MARKER)


def test_the_words_block_is_byte_stable_and_precedes_the_dated_facts(tmp_path, monkeypatch):
    """Rendered from the frozen record at capture: two captures at different instants share the head."""
    from ouroboros import context

    _event, _stored, payload = _schedule(_root(tmp_path), input_sources="declared")
    halves = []
    for instant in ("2026-10-04T06:00:00+00:00", "2026-10-05T07:30:00+00:00"):
        monkeypatch.setattr(context, "utc_now_iso", lambda instant=instant: instant)
        halves.append(_first_input(tmp_path, payload).split("## Runtime context", 1))
    (head, tail), (head_again, tail_again) = halves
    assert head == head_again and CHILD_HEADING in head and head.startswith("## Input source selection")
    assert tail != tail_again and "2026-10-05T07:30:00+00:00" in tail_again  # the dated facts follow it
