"""Source attribution of canonical chat rows: who wrote a row is read from its fields, never its text.

Every class is checked both ways: the rule fires on its own row and a neighbouring
row without the deciding field keeps its own signature, so deleting a rule turns a
test red.
"""
from __future__ import annotations

import ast
import json
import pathlib

import pytest

from ouroboros.dialogue_provenance import (
    dialogue_author,
    render_row_text,
    row_author,
    row_class,
    task_lineage_lookup,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EPOCH = {"pos": 100, "ts": "2026-08-21T20:18:52", "address": {"kind": "chat_row"}}


def _out(**fields):
    return {"ts": "2026-08-01T10:00:00+00:00", "direction": "out", "chat_id": 1, "task_id": "t1",
            "text": "## Summary\nDone.", **fields}


def _quiz_answer(**fields):
    return {"ts": "2026-09-01T10:03:00+00:00", "direction": "system", "chat_id": 1, "type": "quiz_answer",
            "source": "owner_quiz_answer", "user_id": 0, "task_id": "t1",
            "client_message_id": "quiz_answer:t1:q1", "text": "[Owner quiz answer] Owner chose option 1: five",
            "quiz": {"quiz_id": "q1", "question": "Five or six slides?", "options": ["five", "six"],
                     "answered_index": 0, "comment": "The board asked for five."}, **fields}


def _lookup(table):
    return lambda task_id: table.get(task_id)


def test_owner_quiz_answer_is_the_owner_not_ouroboros():
    author = row_author(_quiz_answer())
    assert author == {"kind": "human", "label": "Owner", "via": "quiz"}
    # The same system row without the quiz-answer type is a host fact, not the owner.
    host = row_author({**_quiz_answer(), "type": "acceptance_late_settlement"})
    assert host["kind"] == "host" and host["label"] != "Owner"
    assert row_class(_quiz_answer())["lane"] == 1


def test_child_final_is_the_child_not_ouroboros():
    final = _out(subagent_task_id="c1", task_id="c1", parent_task_id="p1", root_task_id="p1",
                 delegation_role="subagent", subagent_role="reviewer")
    author = row_author(final)
    assert author["kind"] == "child" and "Ouroboros" not in author["label"]
    assert (author["task_id"], author["parent_task_id"], author["root_task_id"], author["role"]) == (
        "c1", "p1", "p1", "reviewer")
    assert row_class(final)["lane"] == 2
    # delegation_role alone is lineage enough; without either field the row is mine.
    assert row_author(_out(delegation_role="subagent", parent_task_id="p1"))["kind"] == "child"
    plain = row_author(_out())
    assert plain["kind"] == "ouroboros" and plain["label"] == "Ouroboros"
    assert row_class(_out())["lane"] == 1


def test_legacy_light_retelling_is_the_helper():
    retelling = {"direction": "system", "type": "task_summary", "summary_kind": "authored_root_summary",
                 "task_id": "t1", "text": "I finished the migration."}
    assert row_author(retelling) == {"kind": "helper", "label": "Light (legacy retelling)"}
    assert row_class(retelling)["lane"] == 2
    projection = {**retelling, "summary_kind": "terminal_root_projection"}
    assert row_author(projection)["kind"] == "host"


def test_host_task_facts_render_their_status_fields_when_text_is_empty():
    facts = {"direction": "system", "type": "task_summary", "summary_kind": "host_task_facts", "task_id": "t9",
             "status": "completed", "outcome": "Completed", "outcome_phase": "done", "reason_code": "",
             "result_ref": {"kind": "task_result", "task_id": "t9", "reader": "get_task_result"}, "text": ""}
    author = row_author(facts)
    assert author == {"kind": "host", "label": "host", "type": "task_summary", "summary_kind": "host_task_facts"}
    assert row_class(facts)["lane"] == 2
    assert render_row_text(facts) == (
        "host facts for t9: status=completed; outcome=Completed; phase=done; result: get_task_result(task_id=t9)")
    # A summary row that has its own text keeps it; nothing is synthesized over it.
    spoken = {**facts, "summary_kind": "terminal_root_projection", "text": "Completed. Root task t9."}
    assert render_row_text(spoken) == "Completed. Root task t9."


def test_pre_epoch_outgoing_row_is_decided_by_the_task_result_fact():
    row = _out(task_id="t1")
    table = {"root1": {"is_root_task": True, "delegation_role": "", "parent_task_id": ""},
             "kid1": {"is_root_task": False, "delegation_role": "subagent", "parent_task_id": "root1",
                      "root_task_id": "root1"}}
    root = row_author({**row, "task_id": "root1"}, pos=5, lineage_epoch=EPOCH, lineage_lookup=_lookup(table))
    assert root["kind"] == "ouroboros" and root["lineage"] == "task_results"
    assert row_class({**row, "task_id": "root1"}, pos=5, lineage_epoch=EPOCH,
                     lineage_lookup=_lookup(table))["lane"] == 1
    child = row_author({**row, "task_id": "kid1"}, pos=5, lineage_epoch=EPOCH, lineage_lookup=_lookup(table))
    assert (child["kind"], child["parent_task_id"], child["lineage"]) == ("child", "root1", "task_results")
    unknown = row_class(row, pos=5, lineage_epoch=EPOCH, lineage_lookup=_lookup(table))
    assert unknown == {"lane": 2, "author": {"kind": "unattributed", "label": "outgoing, author not recorded",
                                             "lineage": "unrecorded"}}
    assert "Ouroboros" not in unknown["author"]["label"]
    # The same row at or after the epoch is my own words; no lookup is consulted.
    after = row_class(row, pos=100, lineage_epoch=EPOCH, lineage_lookup=None)
    assert after["lane"] == 1 and after["author"]["kind"] == "ouroboros"


def test_lineage_epoch_requires_the_row_position_and_only_then():
    with pytest.raises(TypeError, match="needs its stream pos"):
        row_author(_out(), lineage_epoch=EPOCH)
    with pytest.raises(TypeError, match="needs its stream pos"):
        row_class(_out(), lineage_epoch=EPOCH, lineage_lookup=_lookup({}))
    # Without an epoch (a fresh install records lineage from its first row) no pos is needed.
    assert row_author(_out())["kind"] == "ouroboros"
    # A row that carries lineage is signed without a position even when an epoch is known.
    assert row_author(_out(subagent_task_id="c1"), lineage_epoch=EPOCH)["kind"] == "child"


def test_strict_task_result_lookup_reads_facts_and_never_moves_a_file(tmp_path):
    from ouroboros.task_result_schema import SCHEMA_VERSION_KEY, TASK_RESULT_SCHEMA_VERSION
    from ouroboros.task_results import task_result_path

    def store(task_id, **fields):
        path = task_result_path(tmp_path, task_id)
        path.write_text(json.dumps({SCHEMA_VERSION_KEY: TASK_RESULT_SCHEMA_VERSION, "task_id": task_id,
                                    "status": "completed", **fields}), encoding="utf-8")
        return path

    store("root0001")
    store("kid00001", parent_task_id="root0001", root_task_id="root0001", delegation_role="subagent")
    broken = task_result_path(tmp_path, "bad00001")
    broken.write_text("{not json", encoding="utf-8")
    lookup = task_lineage_lookup(tmp_path)
    assert lookup("root0001")["is_root_task"] is True
    assert lookup("kid00001")["parent_task_id"] == "root0001"
    assert lookup("bad00001") is None and lookup("gone0001") is None and lookup("") is None
    assert broken.read_text(encoding="utf-8") == "{not json"  # strict read: no quarantine move
    signed = {tid: row_author(_out(task_id=tid), pos=1, lineage_epoch=EPOCH, lineage_lookup=lookup)["kind"]
              for tid in ("root0001", "kid00001", "bad00001", "gone0001")}
    assert signed == {"root0001": "ouroboros", "kid00001": "child", "bad00001": "unattributed",
                      "gone0001": "unattributed"}


def test_human_rows_use_dialogue_author_and_transport_provenance_survives():
    human = {"direction": "in", "text": "hello", "sender_label": "Alex", "source": "presence:telegram",
             "transport": {"provider": "telegram", "account_id": "bot-1", "conversation_id": "room-1"}}
    assert row_author(human) == {"kind": "human", "label": dialogue_author(human)}
    assert "provider=telegram" in row_author(human)["label"]
    delivered = _out(transport={"provider": "telegram", "conversation_id": "room-1",
                                "delivery": {"state": "accepted"}})
    label = row_author(delivered)["label"]
    assert label.startswith("Ouroboros [") and "provider=telegram" in label and "delivery=accepted" in label
    assert row_author(_out(transport={}))["label"] == "Ouroboros"


def test_consciousness_initiated_turn_is_my_own_words_with_its_focus():
    assert row_author(_out(initiator="consciousness"))["focus"] == "consciousness"
    assert "focus" not in row_author(_out())


def test_quiz_question_renders_option_labels_and_answer_uses_quiz_text(monkeypatch):
    question = _out(type="quiz", text="Which deck?", quiz={
        "quiz_id": "q7", "options": [{"label": "Short deck", "detail": "5 slides", "recommended": True},
                                     {"label": "Long deck", "detail": "12 slides"}]})
    text = render_row_text(question)
    assert text == "[question q7] Which deck? — options: (1) Short deck (2) Long deck; recommended (1)"
    assert "{" not in text and "'label'" not in text
    assert row_class(question)["lane"] == 1 and row_author(question)["kind"] == "ouroboros"

    from ouroboros.tools import plan_dialogue

    answer = _quiz_answer()
    assert render_row_text(answer) == plan_dialogue._quiz_text(answer)
    assert render_row_text(answer).startswith("[answer q1] chose (1) five")
    # Delegated, not copied: the plan-review renderer is the one source of the answer text.
    monkeypatch.setattr(plan_dialogue, "_quiz_text", lambda row: "SENTINEL " + row["quiz"]["quiz_id"])
    assert render_row_text(answer) == "SENTINEL q1"
    plain = _out(text="Just words")
    assert render_row_text(plain) == "Just words"


def test_row_projection_delegates_signature_and_keeps_owner_and_mailbox():
    from ouroboros.dialogue_evidence import _row_projection

    assert _row_projection(_quiz_answer(), "chat", 1)["author"] == "Owner"
    assert _row_projection({**_quiz_answer(), "provenance": "Telegram relay"}, "mailbox", 1)["author"] == "Owner"
    assert _row_projection({"kind": "owner_text", "text": "x", "provenance": "Owner via web"},
                           "mailbox", 1)["author"] == "Owner via web"
    assert _row_projection({"kind": "owner_text", "text": "x"}, "mailbox", 1)["author"] == "Owner"
    child = _row_projection(_out(subagent_task_id="c1", parent_task_id="p1"), "chat", 1)["author"]
    assert child.startswith("child c1") and "Ouroboros" not in child
    assert _row_projection(_out(), "chat", 1)["author"] == "Ouroboros"
    assert _row_projection({"direction": "system", "type": "cancel_receipt", "text": "x"}, "chat", 1)["author"] == "host"
    assert _row_projection({"direction": "in", "text": "hi"}, "chat", 1)["author"] == "User"


# --- dialogue_provenance (D15) reaches D06/D07/D17 only inside functions --------------------------------

_LAZY_ONLY_TARGETS = {"D06", "D07", "D17"}


def _imports(source: str) -> tuple[set[str], set[str]]:
    """``(module-level, function-level)`` imported ouroboros module names of one source."""
    module_level, nested = set(), set()

    def visit(node, in_function):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                visit(child, True)
                continue
            names = []
            if isinstance(child, ast.Import):
                names = [alias.name for alias in child.names]
            elif isinstance(child, ast.ImportFrom) and child.module and child.level == 0:
                names = [child.module] + [f"{child.module}.{alias.name}" for alias in child.names]
            (nested if in_function else module_level).update(n for n in names if n.startswith("ouroboros"))
            visit(child, in_function)

    visit(ast.parse(source), False)
    return module_level, nested


def _domain_of(module: str, modules: dict) -> str | None:
    base = module.replace(".", "/")
    return modules.get(base + ".py") or modules.get(base + "/__init__.py")


def _lazy_only_violations(source: str, modules: dict) -> list[str]:
    module_level, _ = _imports(source)
    return sorted(name for name in module_level if _domain_of(name, modules) in _LAZY_ONLY_TARGETS)


def test_dialogue_provenance_imports_review_delegation_and_task_domains_only_lazily():
    from scripts.domain_graph import load_manifest

    modules = load_manifest().modules
    source = (REPO_ROOT / "ouroboros" / "dialogue_provenance.py").read_text(encoding="utf-8")
    assert _lazy_only_violations(source, modules) == []
    # The scan sees the lazy reaches it permits: the quiz renderer (D06) and the task results (D17).
    _, nested = _imports(source)
    assert {_domain_of(name, modules) for name in nested} >= {"D06", "D17"}
    # A module-level import of the same renderer is reported.
    hoisted = "from ouroboros.tools.plan_dialogue import _quiz_text\n" + source
    assert _lazy_only_violations(hoisted, modules) == ["ouroboros.tools.plan_dialogue"]
