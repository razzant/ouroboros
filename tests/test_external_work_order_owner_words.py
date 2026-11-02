"""The owner's words in a configured session's work order and a direct start.

A task scheduled with the words carries them right after PARENT CONTEXT / REFERENCES;
a task scheduled before the field keeps the exact bytes and fingerprint it had, so a
nanny that started before the update still recovers. The payload, the flat task
metadata and the stored record render one work order. A direct delegate start appends
the same section after the host contract. A declared task scheduled with
``DECLARED_WORDS_FACT`` carries them in both, its receipt naming them included; one
scheduled before keeps a work order and direct start that carry none, its receipt naming
them among the omitted inputs (``tests/test_declared_owner_words.py`` pins its bytes).
Every rule is checked in both directions.
"""
from __future__ import annotations

import hashlib
import json

from ouroboros import owner_words as ow
from ouroboros.subagent_work_order import (
    DECLARED_WORDS_FACT,
    _source_task_from_context,
    assignment_instructions,
    canonical_work_order_source,
    compile_external_work_order,
    input_source_selection_receipt,
    work_order_fingerprint,
    work_order_source_projection,
)
from ouroboros.task_status import load_effective_task_result
from tests.test_helper_owner_words_flow import ASKED, LATER, _child, _root, _schedule
from tests.test_helper_owner_words_flow import _one_api_actor as _one_api_actor  # noqa: F401  (autouse fixture)

SESSION_HEADING = ow._HEADINGS["session"]
# A child scheduled before the field existed, and its work order as the host rendered it then.
SNAPSHOT_TASK = {
    "id": "kid1", "parent_task_id": "root", "root_task_id": "root", "objective": "Collect the Q3 figures",
    "context": "Figures live in reports/q3/.", "expected_output": "A table",
    "constraints": "Do not edit the reports.",
    "task_contract": {"objective": "Collect the Q3 figures", "expected_output": "A table",
                      "deadline_at": "2026-10-05T00:00:00Z"},
}
SNAPSHOT_TEXT = (
    "OBJECTIVE\nCollect the Q3 figures\n\nPARENT CONTEXT / REFERENCES\nDELEGATED ASSIGNMENT CONTEXT\n"
    "Figures live in reports/q3/.\n\nEXPECTED OUTPUT\nA table\n\nCONSTRAINTS / NON-GOALS\nDo not edit the reports.\n\n"
    "TASK CONTRACT AUTHORITY\n{'deadline_at': '2026-10-05T00:00:00Z'}\n\n"
    "HOST AUTHORITY BINDING (facts, not instructions to widen)\n"
    '{"allowed_resources": {}, "deadline_at": "2026-10-05T00:00:00Z", "origin_message_ref": {}, '
    '"parent_task_id": "root", "root_task_id": "root", "task_constraint": {}, "task_id": "kid1", '
    '"workspace_mode": "", "workspace_root": ""}'
)
SNAPSHOT_SHA = "e7a7268c05f10e7dd5ef55d33fc6f7f050dbc89268597bd85c8775daa7f8bc7f"
SAID = "Fix the import\n## not a heading of the work order\n  keep this indent  "
WORDS = [{"text": SAID, "source": "initial_user", "carrier": "ctx", "task_id": "root",
          "ts": "2026-10-04T05:00:00+00:00", "ref": "chat 1 / msg-1"}]


def _sections(text):
    return [block.split("\n", 1)[0] for block in text.split("\n\n")]


def test_a_task_without_the_field_keeps_its_pre_update_bytes_and_fingerprint():
    assert compile_external_work_order(SNAPSHOT_TASK) == SNAPSHOT_TEXT
    assert work_order_fingerprint(SNAPSHOT_TASK) == SNAPSHOT_SHA == hashlib.sha256(SNAPSHOT_TEXT.encode()).hexdigest()
    for carried in ({ow.FIELD: WORDS}, {ow.ABSENT_FIELD: "initiator=consciousness"}):
        task = {**SNAPSHOT_TASK, "metadata": carried}
        assert compile_external_work_order(task) != SNAPSHOT_TEXT
        assert work_order_fingerprint(task) != SNAPSHOT_SHA


def test_the_words_follow_the_parent_context_verbatim():
    rendered = compile_external_work_order({**SNAPSHOT_TASK, "metadata": {ow.FIELD: WORDS}})
    order = _sections(rendered)
    at = order.index(SESSION_HEADING)
    assert order[at - 1] == "PARENT CONTEXT / REFERENCES" and order[at + 1] == "EXPECTED OUTPUT"
    section = rendered.split("\n\n" + SESSION_HEADING, 1)[1].split("\n\nEXPECTED OUTPUT", 1)[0]
    assert section.endswith("\n" + SAID)  # verbatim: no strip, no indent, no cut
    assert "Host fact, carried by value from task root." in section
    assert "OBJECTIVE above" not in rendered and "OBJECTIVE above" not in SESSION_HEADING
    # Without a parent context the words still stand right after the objective.
    bare = compile_external_work_order({**SNAPSHOT_TASK, "context": None, "metadata": {ow.FIELD: WORDS}})
    assert _sections(bare)[:3] == ["OBJECTIVE", SESSION_HEADING, "EXPECTED OUTPUT"]
    marked = compile_external_work_order({**SNAPSHOT_TASK, "metadata": {ow.ABSENT_FIELD: "initiator=consciousness"}})
    absent = "No words of my human are recorded for this work (host marker: initiator=consciousness)."
    assert f"Figures live in reports/q3/.\n\n{absent}\n\nEXPECTED OUTPUT" in marked
    assert SESSION_HEADING not in marked


def test_the_payload_the_flat_metadata_and_the_record_render_one_work_order(tmp_path):
    root = _root(tmp_path)
    _event, stored, payload = _schedule(root)
    rendered = compile_external_work_order(payload)  # the nanny's bootstrap: the payload with nested metadata
    assert ASKED in rendered and LATER in rendered
    child = _child(tmp_path, payload)
    assert compile_external_work_order(_source_task_from_context(child, payload["id"])) == rendered  # flat
    record = load_effective_task_result(tmp_path, payload["id"], materialize_artifacts=False)
    assert ow.FIELD not in record.get("metadata", {}) and record[ow.FIELD]  # the record holds it at top level
    projected, reason = work_order_source_projection(record, 0, len(rendered))
    assert reason == "" and projected["text"] == rendered
    request = {"complete_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
               "complete_chars": len(rendered),
               "source": {"kind": "task_result", "task_id": payload["id"], "tool": "get_task_result",
                          "arguments": {"task_id": payload["id"], "include_authority": True,
                                        "include_work_order_source": True},
                          "projection": "canonical_work_order"}}
    assert canonical_work_order_source(child, request) == (rendered, "")
    # A work order rendered without the words is not the canonical source of this task.
    wordless = compile_external_work_order({**payload, "metadata": {
        key: value for key, value in payload["metadata"].items() if key != ow.FIELD}})
    assert SESSION_HEADING not in wordless
    stale = {**request, "complete_sha256": hashlib.sha256(wordless.encode("utf-8")).hexdigest(),
             "complete_chars": len(wordless)}
    assert canonical_work_order_source(child, stale) == ("", "source_digest_mismatch")


def test_a_parent_reading_the_childs_work_order_sees_the_childs_words(tmp_path):
    from ouroboros.tools.control_task_results import _get_task_result

    parent = _root(tmp_path, metadata={}, owner_said=False)  # no owner-door ref to project into the binding
    parent._owner_directives = [{"source": "initial_user", "content": ASKED},
                                {"source": "owner_mailbox", "content": LATER, "msg_id": "mail-1"}]
    _event, _stored, payload = _schedule(parent)
    rendered = compile_external_work_order(payload)
    parent._owner_directives = [{"source": "owner_mailbox", "content": "A later word to the parent only"}]
    answer = json.loads(_get_task_result(parent, payload["id"], include_authority=True,
                                         include_work_order_source=True, source_start_char=0,
                                         source_end_char=len(rendered)))
    assert answer["work_order_source"]["text"] == rendered
    assert "A later word to the parent only" not in answer["work_order_source"]["text"]


def test_a_direct_start_appends_the_words_after_the_host_contract(tmp_path):
    marker = "HOST TASK CONTRACT AUTHORITY (normalized JSON; predecessor is a brief):\n"
    root = _root(tmp_path)
    text = assignment_instructions(root)
    contract_json, words = text[len(marker):].split("\n\n", 1)
    assert text.startswith(marker) and json.loads(contract_json)["objective"] == root.task_contract["objective"]
    assert words.startswith(SESSION_HEADING + "\nHost fact, carried by value from task root.\n")
    assert words.endswith(ASKED + "\n\n[owner · owner_mailbox of task root · msg mail-1]\n" + LATER)
    # A child appends what it inherited, not its own assignment.
    _event, _stored, payload = _schedule(root)
    child_words = assignment_instructions(_child(tmp_path, payload)).split("\n\n", 1)[1]
    assert child_words == words
    wake = _root(tmp_path, metadata={"initiator": "consciousness"}, owner_said=False)
    assert assignment_instructions(wake).endswith(
        "\n\nNo words of my human are recorded for this work (host marker: initiator=consciousness).")
    root.task_contract, root.task_metadata = {}, {"origin_message_ref": root.task_metadata["origin_message_ref"]}
    assert assignment_instructions(root) == ""  # no contract: no assignment block, words or not


def test_a_declared_work_order_carries_the_words_and_one_scheduled_before_holds_none(tmp_path):
    """A declared child scheduled now carries the words in its work order and direct start, its receipt
    naming them included; the same child scheduled before the fact keeps its parent's selection."""
    marker = "HOST TASK CONTRACT AUTHORITY (normalized JSON; predecessor is a brief):\n"
    words_item = "the owner's words that caused this work"
    root = _root(tmp_path)
    _event, stored, declared = _schedule(root, input_sources="declared")
    assert declared["metadata"][ow.FIELD] and declared["metadata"][DECLARED_WORDS_FACT] is True
    assert stored[DECLARED_WORDS_FACT] is True and DECLARED_WORDS_FACT not in declared["task_contract"]
    rendered = compile_external_work_order(declared)
    receipt = input_source_selection_receipt(declared)
    assert [item for item in receipt["included"] if item.startswith(words_item)]
    assert not [item for item in receipt["omitted_automatic"] if item.startswith(words_item)]
    assert "the memory marks of that room and the global ones" in str(receipt["omitted_automatic"])  # still not loaded
    order = _sections(rendered)
    assert order[order.index(SESSION_HEADING) - 1] == "OBJECTIVE"  # no parent context: right after the objective
    assert rendered.split("\n\n" + SESSION_HEADING, 1)[1].split("\n\nEXPECTED OUTPUT", 1)[0].endswith(LATER)
    child = _child(tmp_path, declared)
    assert compile_external_work_order(_source_task_from_context(child, declared["id"])) == rendered  # flat
    direct = assignment_instructions(child)
    contract_json, words = direct[len(marker):].split("\n\n", 1)
    assert direct.startswith(marker) and json.loads(contract_json)["input_sources"] == "declared"
    assert words.startswith(SESSION_HEADING + "\nHost fact, carried by value from task root.\n") and ASKED in words
    # The other side: scheduled before the fact, the same child keeps its parent's selection; the words ride unread.
    older = {**declared, "metadata": {key: value for key, value in declared["metadata"].items()
                                      if key != DECLARED_WORDS_FACT}}
    older_text = compile_external_work_order(older)
    assert SESSION_HEADING not in older_text and ASKED not in older_text and LATER not in older_text
    assert [item for item in input_source_selection_receipt(older)["omitted_automatic"] if item.startswith(words_item)]
    bare = {**older, "metadata": {key: value for key, value in older["metadata"].items() if key != ow.FIELD}}
    assert compile_external_work_order(bare) == older_text  # the carried field changes nothing there
    child.task_metadata.pop(DECLARED_WORDS_FACT)
    assert assignment_instructions(child) == marker + contract_json
    # A shared sibling's work order and direct start carry the words, and no fact rides with it.
    _e, shared_stored, shared = _schedule(root, objective="Collect the Q4 figures")
    assert DECLARED_WORDS_FACT not in shared["metadata"] and DECLARED_WORDS_FACT not in shared_stored
    assert SESSION_HEADING in compile_external_work_order(shared) and ASKED in compile_external_work_order(shared)
    assert assignment_instructions(_child(tmp_path, shared)).split("\n\n", 1)[1].startswith(SESSION_HEADING)


def test_the_authority_fingerprint_does_not_move_with_the_words(tmp_path):
    from ouroboros.delegate_recovery import authority_fingerprint_from_context

    root = _root(tmp_path)
    before = authority_fingerprint_from_context(root)
    root._owner_directives.append({"source": "owner_mailbox", "content": "One more word", "msg_id": "mail-2"})
    root.task_metadata[ow.FIELD] = WORDS
    assert authority_fingerprint_from_context(root) == before
    root.task_contract = {**root.task_contract, "objective": "A different objective"}
    assert authority_fingerprint_from_context(root) != before  # the fingerprint is live on its own inputs


def test_the_parent_is_told_what_a_child_and_a_session_start_with():
    """The emitted contract says what the host supplies and what the parent writes."""
    from ouroboros.tools import delegate
    from ouroboros.tools.control import get_tools

    schedule = next(tool for tool in get_tools() if tool.name == "schedule_subagent").schema["parameters"]["properties"]
    start = next(tool for tool in delegate.get_tools() if tool.name == "delegate_start")
    prompt = start.schema["parameters"]["properties"]["prompt"]["description"]
    context, selection = schedule["context"]["description"], schedule["input_sources"]["description"]
    for text in (context + "\n" + selection, prompt):
        assert "my human's originating words" in text
    assert "input_sources says what it starts with" in context
    assert "write the orientation it lacks" in context and "none of my memory" in context
    assert "why the work exists" in prompt and "the host supplies the canonical work order" in prompt
    assert selection.startswith("Omit or shared: the child starts with ") and "one read away" in selection
    # The declared half carries the words too and keeps its own contract: no automatic shared memory, dialogue,
    # project knowledge, parent context or attachments; the assignment governs exchange.
    shared, declared = selection.split("declared selects only the authored ", 1)
    assert "my human's originating words verbatim" in shared and "my human's originating words verbatim" in declared
    assert ("excludes automatic shared memory, dialogue, project knowledge, inherited parent context and attachments"
            in " ".join(declared.split()))
