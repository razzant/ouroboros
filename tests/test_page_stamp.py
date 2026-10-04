"""A memory page's host stamp comes from ``terminal_projection.stamp_facts``.

The original ``host_stamp`` body moved without changing its order or format: per task its
terminal projection row, then its host facts row (both among the page's already-read
rows; a repeated row of one task takes the last), then the strict task result (the
file never moves), else ``not_recorded``; there is no pass over the chat chain.
``host_stamp`` stays the same call with the same ``{tasks, computed_at}`` result.
The entries keep every original key and gain only optional facts copied from the same
source: ``outcome_final``, ``reason_code``, ``objective_status`` and the source
row's ``source_address``; the full ``outcome_axes`` never enter a stamp.
``part_stamp`` summarizes member page stamps and keeps every failure in full.
No test calls a model or the network.
"""
from __future__ import annotations

import ast
import json
import pathlib
import types

from ouroboros import chat_chain
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.terminal_projection import _project_row, part_stamp, stamp_facts
from ouroboros.tools.chronicle import _chronicle_write, host_stamp, page_covers
from ouroboros.tools.registry import ToolContext

REPO = pathlib.Path(__file__).resolve().parents[1]
ORIGINAL_KEYS = {"task_id", "status", "outcome", "outcome_phase", "source", "result_ref", "review_verdict"}
OPTIONAL = {"outcome_final", "reason_code", "objective_status", "source_address"}


def ts(n: int) -> str:
    return f"2026-10-01T{n // 3600 % 24:02d}:{n // 60 % 60:02d}:{n % 60:02d}+00:00"


def chat(root: pathlib.Path, rows) -> list:
    path = root / "logs" / "chat.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return list(rows)


def addr(row) -> str:
    return chat_chain.format_address(chat_chain.row_address(row))


def terminal_row(tid: str, n: int, status: str, **fields) -> dict:
    """A terminal projection row exactly as the host publishes it at ``task_done``."""
    row = _project_row(tid, {"status": status, "task_id": tid, "chat_id": 1, **fields}, {}, {"chat_id": 1})
    return {**row, "ts": ts(n)}


def facts_row(scratch: pathlib.Path, tid: str, n: int, **usage) -> dict:
    """A host facts row exactly as the post-task phase appends it (written in a scratch root)."""
    from ouroboros.post_task_synthesis import _record_task_facts

    root = scratch / tid
    (root / "logs").mkdir(parents=True)
    _record_task_facts(types.SimpleNamespace(drive_root=root), {"id": tid, "chat_id": 1},
                       {"rounds": 1, **usage}, {}, root / "logs")
    return {**json.loads((root / "logs" / "chat.jsonl").read_text(encoding="utf-8")), "ts": ts(n)}


def original_host_stamp(root, task_ids, *, rows=()):
    """The original host_stamp body (tools/chronicle.py:233-287 at 4dc962d66), frozen here as the parity witness."""
    from ouroboros.project_dialogue import OUTCOME_PHASE_HEADLINE, outcome_phase
    from ouroboros.task_results import load_task_result

    def stamp_entry(task_id, row, source):
        entry = {"task_id": task_id, "status": str(row.get("status") or ""), "outcome": str(row.get("outcome") or ""),
                 "outcome_phase": str(row.get("outcome_phase") or ""), "source": source,
                 "result_ref": row.get("result_ref") or {"kind": "task_result", "task_id": task_id,
                                                          "reader": "get_task_result"}}
        if row.get("reason_detail"):
            entry["review_verdict"] = str(row["reason_detail"])
        return entry

    def result_entry(task_id):
        try:
            result = load_task_result(root, task_id, strict=True)
        except (OSError, ValueError):
            result = None
        if not isinstance(result, dict) or not result:
            return {"task_id": task_id, "status": "not_recorded"}
        try:
            phase = outcome_phase(result, {})
        except (KeyError, TypeError, ValueError, AttributeError):
            phase = ""
        return stamp_entry(task_id, {**result, "outcome_phase": phase,
                                     "outcome": OUTCOME_PHASE_HEADLINE.get(phase, "")}, "task_results")

    terminal, host_facts = {}, {}
    for entry in rows:
        row = entry[1]
        task = str(row.get("task_id") or "")
        if row.get("type") != "task_summary" or not task:
            continue
        if row.get("summary_kind") in {"terminal_root_projection", "terminal_result_projection"}:
            terminal[task] = row
        elif row.get("summary_kind") == "host_task_facts":
            host_facts[task] = row
    stamped = []
    for task in dict.fromkeys(str(t) for t in task_ids if str(t or "")):
        if task in terminal:
            stamped.append(stamp_entry(task, terminal[task], str(terminal[task]["summary_kind"])))
        elif task in host_facts:
            stamped.append(stamp_entry(task, host_facts[task], "host_task_facts"))
        else:
            stamped.append(result_entry(task))
    return stamped


def four_sources(tmp_path: pathlib.Path):
    """One room whose page covers a task of each source, a retried root and a child."""
    from ouroboros.task_results import write_task_result

    rows = chat(tmp_path, [
        {"chat_id": 1, "direction": "in", "ts": ts(1), "text": "Migrate the store and write the report",
         "client_message_id": "m1"},
        {"chat_id": 1, "direction": "out", "ts": ts(2), "task_id": "both", "text": "Migration attempt one."},
        facts_row(tmp_path / "producer", "both", 3),
        terminal_row("both", 4, "failed", reason_code="review_rejected",
                     outcome_axes={"objective": {"status": "fail", "source": "acceptance"}}),
        facts_row(tmp_path / "producer", "facts", 5, reason_code="tool_budget"),
        {"chat_id": 1, "direction": "out", "ts": ts(6), "task_id": "fromfile", "text": "Report written."},
        {"chat_id": 1, "direction": "out", "ts": ts(7), "task_id": "nowhere", "text": "Started, never recorded."},
        terminal_row("kid1", 8, "completed", parent_task_id="both", root_task_id="both", delegation_role="subagent"),
        # The retry publishes the same summary id again: the later row is the task's stamp.
        terminal_row("both", 9, "completed"),
    ])
    write_task_result(tmp_path, "both", "completed", result="Migrated on the second attempt.")
    write_task_result(tmp_path, "fromfile", "completed", result="Report written.", reason_code="owner_accepted",
                      outcome_axes={"objective": {"status": "pass", "source": "author_acceptance"}})
    found = page_covers(tmp_path, "1", from_addr=addr(rows[0]), to_addr=addr(rows[-1]))
    return rows, found


# --- parity with the original stamp and the order of sources ----------------------------------------

def test_host_stamp_keeps_its_original_order_and_format_for_all_four_sources(tmp_path):
    rows, found = four_sources(tmp_path)
    ids = found["covers"]["task_ids"]
    assert ids == ["both", "facts", "fromfile", "nowhere", "kid1"]
    result_file = tmp_path / "task_results" / "fromfile.json"
    before = result_file.read_bytes()
    stamp = host_stamp(tmp_path, ids, rows=found["rows"])
    assert set(stamp) == {"tasks", "computed_at"}
    old = original_host_stamp(tmp_path, ids, rows=found["rows"])
    assert [entry["task_id"] for entry in stamp["tasks"]] == [entry["task_id"] for entry in old] == ids
    for new, before_move in zip(stamp["tasks"], old):
        assert {key: new[key] for key in before_move} == before_move  # every original key and value, unchanged
        assert set(new) - set(before_move) <= OPTIONAL
    by_task = {entry["task_id"]: entry for entry in stamp["tasks"]}
    # 1: the terminal projection wins over host facts and the task result; the retry's later row wins.
    assert by_task["both"]["source"] == "terminal_root_projection" and by_task["both"]["status"] == "completed"
    assert by_task["both"]["outcome_final"] is True and by_task["both"]["source_address"] == addr(rows[8])
    # 2: host facts only, never final.
    assert by_task["facts"]["source"] == "host_task_facts" and by_task["facts"]["outcome_final"] is False
    assert by_task["facts"]["source_address"] == addr(rows[4]) and by_task["facts"]["reason_code"] == "tool_budget"
    # 3: the strict task result; the file is where it was, byte for byte.
    assert by_task["fromfile"]["source"] == "task_results" and by_task["fromfile"]["status"] == "completed"
    assert result_file.read_bytes() == before
    assert "outcome_final" not in by_task["fromfile"] and "source_address" not in by_task["fromfile"]
    # 4: nothing anywhere.
    assert by_task["nowhere"] == {"task_id": "nowhere", "status": "not_recorded"}
    # A child task is stamped from its own terminal projection.
    assert by_task["kid1"]["source"] == "terminal_result_projection" and by_task["kid1"]["outcome_phase"] == "done"
    assert all(set(entry) >= {"task_id", "status"} for entry in stamp["tasks"])
    assert all(set(entry) >= ORIGINAL_KEYS - {"review_verdict"} for entry in stamp["tasks"] if entry["task_id"] != "nowhere")


def test_the_last_terminal_row_of_a_task_is_its_stamp_either_way(tmp_path):
    rows, found = four_sources(tmp_path)
    failed, retried = found["rows"][3], found["rows"][8]
    assert stamp_facts(tmp_path, ["both"], rows=[failed, retried])["both"]["status"] == "completed"
    swapped = stamp_facts(tmp_path, ["both"], rows=[retried, failed])["both"]
    assert swapped["status"] == "failed" and swapped["outcome_phase"] == "error"
    assert swapped["source_address"] == addr(rows[3])


def test_stamp_facts_reads_only_the_rows_it_is_given_never_the_chain(tmp_path):
    _rows, found = four_sources(tmp_path)
    # The same task with its rows in hand stamps from its terminal row; without them, from its result file.
    assert stamp_facts(tmp_path, ["both"], rows=found["rows"])["both"]["source"] == "terminal_root_projection"
    assert stamp_facts(tmp_path, ["both"], rows=())["both"]["source"] == "task_results"
    assert stamp_facts(tmp_path, ["kid1"], rows=())["kid1"] == {"task_id": "kid1", "status": "not_recorded"}


def test_stamp_facts_keys_tasks_in_the_order_asked_once_each(tmp_path):
    _rows, found = four_sources(tmp_path)
    facts = stamp_facts(tmp_path, ["kid1", "", None, "both", "kid1", "nowhere"], rows=found["rows"])
    assert list(facts) == ["kid1", "both", "nowhere"]
    assert all(fact["task_id"] == task for task, fact in facts.items())


def test_a_strict_refusal_is_not_recorded_and_leaves_the_file_in_place(tmp_path):
    path = tmp_path / "task_results" / "broken.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert stamp_facts(tmp_path, ["broken"])["broken"] == {"task_id": "broken", "status": "not_recorded"}
    assert path.read_text(encoding="utf-8") == "{not json"


# --- optional facts: copied, never interpreted ---------------------------------------------------

def test_optional_facts_are_copied_from_the_source_and_the_axes_stay_there(tmp_path):
    verdict = "Rejected:  «тест не прошёл» — the fix\tbroke login.\n"
    denials = [{"tool": "run_shell", "reason": "x" * 400} for _ in range(100)]
    terminal = terminal_row("t1", 1, "failed", reason_code="review_rejected", outcome_axes={
        "objective": {"status": "fail", "source": "acceptance"}, "policy": {"status": "denied", "denials": denials}})
    terminal["reason_detail"] = verdict
    facts = facts_row(tmp_path / "producer", "t2", 2)
    rows = chat(tmp_path, [terminal, facts])
    found = list(chat_chain.iter_room_rows(tmp_path, "1"))
    entries = stamp_facts(tmp_path, ["t1", "t2"], rows=found)
    first, second = entries["t1"], entries["t2"]
    assert first["review_verdict"] == verdict  # byte for byte
    assert first["reason_code"] == "review_rejected" and first["objective_status"] == "fail"
    assert first["outcome_final"] is True and first["source_address"] == addr(rows[0])
    assert second["outcome_final"] is False and second["objective_status"] == "not_evaluated"
    assert "reason_code" not in second  # the source holds an empty code: nothing to copy
    for entry in entries.values():
        assert "outcome_axes" not in entry and len(json.dumps(entry)) < 1000
    assert len(json.dumps(terminal["outcome_axes"])) > 40_000  # the axes the stamp did not copy
    # A row without its optional facts gives an entry without them; a malformed address gives none.
    bare = {"type": "task_summary", "summary_kind": "terminal_root_projection", "task_id": "t3",
            "status": "completed", "outcome": "Done", "outcome_phase": "done"}
    assert set(stamp_facts(tmp_path, ["t3"], rows=[({}, bare)])["t3"]) == ORIGINAL_KEYS - {"review_verdict"}


# --- trap (b): a page that says done over a failed task -------------------------------------------

def test_a_failed_task_stays_failed_in_the_stamp_of_a_page_that_says_done(tmp_path):
    rows = chat(tmp_path, [
        {"chat_id": 1, "direction": "in", "ts": ts(1), "text": "Ship the migration", "client_message_id": "m1"},
        {"chat_id": 1, "direction": "out", "ts": ts(2), "task_id": "ship0001", "text": "Shipped."},
        terminal_row("ship0001", 3, "failed", reason_code="review_rejected",
                     outcome_axes={"objective": {"status": "fail", "source": "acceptance"}}),
    ])
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root0001", current_chat_id=1)
    reply = json.loads(_chronicle_write(ctx, kind="page", text="Done: the migration shipped and works.",
                                        covers={"from": addr(rows[0]), "to": addr(rows[2])}))
    assert reply["ok"] and reply["stamp"] == {"failed": 1}
    page = ChronicleStore(tmp_path).get(reply["node_id"])
    (entry,) = page["host_stamp"]["tasks"]
    assert entry["status"] == "failed" and entry["outcome_phase"] == "error" and entry["outcome"] == "Failed"
    assert entry["outcome_final"] is True and entry["objective_status"] == "fail"
    summary = part_stamp([page["host_stamp"]])
    assert summary["tasks"] == [entry]
    assert summary["counts"] == [{"source": "terminal_root_projection", "outcome_phase": "error", "tasks": 1}]


# --- part_stamp -----------------------------------------------------------------------------------

def entry(task_id: str, source: str, phase: str, **extra) -> dict:
    return {"task_id": task_id, "status": "completed" if phase == "done" else "failed", "outcome": phase,
            "outcome_phase": phase, "source": source, **extra}


def test_part_stamp_keeps_every_failure_and_every_non_terminal_stamp_and_counts_the_rest():
    page_a = {"tasks": [entry("t1", "terminal_root_projection", "done"),
                        entry("t2", "terminal_root_projection", "error"),
                        entry("t3", "host_task_facts", "done")], "computed_at": ts(1)}
    page_b = {"tasks": [entry("t4", "task_results", "done"), {"task_id": "t5", "status": "not_recorded"},
                        entry("t6", "terminal_result_projection", "done"),
                        entry("t7", "terminal_root_projection", "cancelled"),
                        entry("t1", "terminal_root_projection", "done")], "computed_at": ts(2)}
    summary = part_stamp([page_a, None, page_b])  # a member without a stamp (a legacy record) adds nothing
    kept = [item["task_id"] for item in summary["tasks"]]
    assert kept == ["t2", "t3", "t4", "t5", "t7"]  # never-done, or not from a terminal projection: in full
    assert {"t1", "t6"}.isdisjoint(kept)  # terminal and done: counted only
    assert summary["tasks"][3] == {"task_id": "t5", "status": "not_recorded"}
    assert summary["counts"] == [
        {"source": "", "outcome_phase": "", "tasks": 1},
        {"source": "host_task_facts", "outcome_phase": "done", "tasks": 1},
        {"source": "task_results", "outcome_phase": "done", "tasks": 1},
        {"source": "terminal_result_projection", "outcome_phase": "done", "tasks": 1},
        {"source": "terminal_root_projection", "outcome_phase": "cancelled", "tasks": 1},
        {"source": "terminal_root_projection", "outcome_phase": "done", "tasks": 1},
        {"source": "terminal_root_projection", "outcome_phase": "error", "tasks": 1}]
    assert sum(row["tasks"] for row in summary["counts"]) == 7  # each task once, though t1 is on two pages
    assert "computed_at" in summary


def test_part_stamp_takes_a_task_stamped_on_two_pages_by_its_later_stamp():
    early = {"tasks": [entry("t1", "terminal_root_projection", "done")]}
    late = {"tasks": [entry("t1", "terminal_root_projection", "error", review_verdict="Reverted.")]}
    assert part_stamp([early, late])["tasks"] == [late["tasks"][0]]
    assert part_stamp([late, early])["tasks"] == []
    assert part_stamp([late, early])["counts"] == [
        {"source": "terminal_root_projection", "outcome_phase": "done", "tasks": 1}]


# --- one source of the stamp ----------------------------------------------------------------------

def test_the_stamp_body_lives_only_in_terminal_projection_and_host_stamp_delegates_lazily():
    def defined(relative: str) -> set:
        tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
        return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    tools = (REPO / "ouroboros" / "tools" / "chronicle.py").read_text(encoding="utf-8")
    assert {"stamp_facts", "part_stamp", "_stamp_entry", "_result_entry"} <= defined("ouroboros/terminal_projection.py")
    assert not {"stamp_facts", "part_stamp", "_stamp_entry", "_result_entry"} & defined("ouroboros/tools/chronicle.py")
    tree = ast.parse(tools)
    host = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "host_stamp")
    lazy = {node.module for node in ast.walk(host) if isinstance(node, ast.ImportFrom)}
    top = {node.module for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert "ouroboros.terminal_projection" in lazy and "ouroboros.terminal_projection" not in top
