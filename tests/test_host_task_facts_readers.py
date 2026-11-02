"""A host task-facts row reaches its readers with the finality and authority it recorded.

The post-task producer appends a typed ``host_task_facts`` row before finalization: no text,
``outcome_final=false`` and ``outcome_authority=pre_finalization_host_facts``; a retained Presence
observation records ``retained_presence_observation``. Every reader that shows the row (chat
history and its search, a memory row, the memory view's task line, the page writer's input and
host stamp, the story's linked rows) keeps its task, status, reason, finality and authority
distinct. A later canonical terminal row wins where a reader picks one status, and the earlier
row stays as recorded. A missing finality stays unknown: an old row or a stored row fact
written before the field never gains one. No test calls a model or the network.
"""
from __future__ import annotations

import json
import types

import pytest

from ouroboros import chat_chain
from ouroboros import dialogue_store as ds
from ouroboros import memory_inventory as mi
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.dialogue_provenance import dialogue_text, render_row_text
from ouroboros.task_results import write_task_result
from tests import _dialogue as dialogue_test

PRELIMINARY = "final=false; authority=pre_finalization_host_facts"
OUTCOMES = (("t-ok", "completed", ""), ("t-fail", "failed", "provider_unavailable"), ("t-cancel", "cancelled", "cancelled"))
MAIN_TASK = {"id": "turn0001", "type": "task", "chat_id": 1, "_is_direct_chat": True}


def _activated(root):
    root.mkdir(parents=True, exist_ok=True)
    assert ChronicleStore(root).ensure_activated()["kind"] == "activation"
    ds.append(root, "chat", {"chat_id": 1, "direction": "in", "text": "Ship the release", "client_message_id": "m1",
                             "ts": "2026-10-26T10:00:00+00:00"})


def _produce(root, task_id, status, reason):
    """The real producer, on a stored result of ``status``; returns the row it appended, read back."""
    from ouroboros.post_task_synthesis import _record_task_facts

    write_task_result(root, task_id, status, chat_id=1, reason_code=reason)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    _record_task_facts(types.SimpleNamespace(drive_root=root), {"id": task_id, "chat_id": 1},
                       {"rounds": 1, "reason_code": reason}, {}, root / "logs")
    return next(row for row in reversed(dialogue_test.chat_rows(root))
                if row.get("summary_kind") == "host_task_facts" and row.get("task_id") == task_id)


def _terminal(root, task_id, status, reason):
    """The canonical terminal row the host writes once the result settles (its own builder and writer)."""
    from ouroboros.project_dialogue import append_canonical_task_summary
    from ouroboros.terminal_projection import _project_row

    row = _project_row(task_id, {"status": status, "task_id": task_id, "chat_id": 1, "reason_code": reason},
                       {}, {"chat_id": 1})
    assert append_canonical_task_summary(root, row)
    return row


def _lane_two(root):
    from ouroboros import memory_view as mv

    snapshot = mv.capture_memory_view(root, MAIN_TASK, mv.view_spec_for_task(MAIN_TASK, root))
    text = mv.render_room(snapshot)
    section = text[text.index("### Task facts of this conversation"):]
    return {line.split("; host; task ", 1)[1].split("]", 1)[0]: line
            for line in section.split("\n") if "; host; task " in line}


def test_every_reader_keeps_the_producers_task_status_reason_finality_and_authority(tmp_path):
    from ouroboros.memory import Memory

    _activated(tmp_path)
    rows = [_produce(tmp_path, task_id, status, reason) for task_id, status, reason in OUTCOMES]
    words = {}
    for row, (task_id, status, reason) in zip(rows, OUTCOMES):
        assert (row["text"], row["outcome_final"], row["outcome_authority"]) == ("", False, "pre_finalization_host_facts")
        assert row["status"] == status  # recorded "completed" stays beside final=false, never read as final
        words[task_id] = render_row_text(row)
        assert dialogue_text(row) == words[task_id]  # one renderer for history, search and memory rows
        assert words[task_id].startswith(f"host facts for {task_id}: status={status}; ")
        assert PRELIMINARY in words[task_id] and f"result: get_task_result(task_id={task_id})" in words[task_id]
        assert (f"reason={reason}" in words[task_id]) is bool(reason)
        assert "objective=not_evaluated" in words[task_id] and "policy" not in words[task_id]
    assert len(set(words.values())) == 3
    history = Memory(drive_root=tmp_path).chat_history(count=10)
    assert all(f"[task_summary] {text}" in history for text in words.values())
    found = Memory(drive_root=tmp_path).chat_history(count=10, search="provider_unavailable")
    assert words["t-fail"] in found and "host facts for t-ok" not in found
    lane = _lane_two(tmp_path)
    assert all(words[task_id] in lane[task_id] for task_id, _status, _reason in OUTCOMES)
    # The page writer's stamp keeps the same split: a recorded "completed" that is not final.
    from ouroboros.memory_fallback import _stamp_line
    from ouroboros.terminal_projection import stamp_facts

    stamp = stamp_facts(tmp_path, ["t-ok"], rows=[(chat_chain.row_address(row), row) for row in rows])["t-ok"]
    line = _stamp_line(stamp, 0)
    assert line.startswith("- task t-ok: status completed; ") and f"{PRELIMINARY}; stamp from host_task_facts" in line


def test_a_later_terminal_row_wins_the_status_while_the_preliminary_row_stays_as_recorded(tmp_path):
    from ouroboros.memory import Memory
    from ouroboros.memory_fallback import _stamp_line
    from ouroboros.terminal_projection import stamp_facts

    _activated(tmp_path)
    facts = _produce(tmp_path, "t-fail", "failed", "provider_unavailable")
    read = [(chat_chain.row_address(row), row) for row in dialogue_test.chat_rows(tmp_path)]
    early = stamp_facts(tmp_path, ["t-fail"], rows=read)["t-fail"]
    assert (early["source"], early["outcome_final"], early["outcome_authority"]) == (
        "host_task_facts", False, "pre_finalization_host_facts")
    assert f"{PRELIMINARY}; stamp from host_task_facts" in _stamp_line(early, 1)
    _terminal(tmp_path, "t-fail", "failed", "provider_unavailable")
    read = [(chat_chain.row_address(row), row) for row in dialogue_test.chat_rows(tmp_path)]
    late = stamp_facts(tmp_path, ["t-fail"], rows=read)["t-fail"]
    assert (late["source"], late["outcome_final"], late["outcome_authority"]) == (
        "terminal_root_projection", True, "canonical_task_result_after_finalization")
    assert "final=true; authority=canonical_task_result_after_finalization; stamp from terminal_root_projection" in (
        _stamp_line(late, 1))
    assert _lane_two(tmp_path)["t-fail"].startswith(f"[{facts['ts']}; host; task t-fail] Failed. Root task t-fail.")
    history = Memory(drive_root=tmp_path).chat_history(count=10)
    assert render_row_text(facts) in history and PRELIMINARY in history  # the earlier row, unchanged


def test_the_off_draft_keeps_the_rows_finality_in_its_stamp(tmp_path):
    from ouroboros.memory_fallback import build_writer_input, FallbackUnit

    _activated(tmp_path)
    _produce(tmp_path, "t-fail", "failed", "provider_unavailable")
    store = ChronicleStore(tmp_path)
    segment = next(mi.open_segments(tmp_path, store, "1"))
    unit = FallbackUnit("page", "1", "open:1", segment=segment)
    prompt = build_writer_input(tmp_path, store, unit, budget_tokens=None).prompt
    assert PRELIMINARY in prompt
    stamp = prompt[prompt.index("## Tasks of these rows"):]
    assert "- task t-fail: status failed; " in stamp
    assert f"{PRELIMINARY}; stamp from host_task_facts" in stamp


def test_a_retained_presence_observation_keeps_its_own_authority():
    row = {"type": "task_summary", "summary_kind": "host_task_facts", "task_id": "p1", "text": "", "status": "running",
           "presence_recovery": {"recorded_status": "running"}, "outcome_final": False,
           "outcome_authority": "retained_presence_observation"}
    words = render_row_text(row)
    assert words.startswith("host observation for Presence turn p1: recorded_status running; "
                            "final=false; authority=retained_presence_observation; no completed answer")
    assert dialogue_text(row) == words


@pytest.mark.parametrize("missing", [("outcome_final", "outcome_authority"), ("outcome_final",)])
def test_missing_finality_stays_unknown_and_is_never_invented(tmp_path, missing):
    from ouroboros.memory_fallback import _stamp_line
    from ouroboros.terminal_projection import stamp_facts

    _activated(tmp_path)
    row = {key: value for key, value in _produce(tmp_path, "t-ok", "completed", "").items() if key not in missing}
    # An authority the row still holds is kept beside the unknown finality; none is invented.
    unknown = "final=unknown" + ("; authority=pre_finalization_host_facts" if "outcome_authority" in row else "")
    words = render_row_text(row)
    assert f"{unknown}; " in words and "final=false" not in words and "final=true" not in words
    assert ("authority=" in words) is ("outcome_authority" in row)
    entry = stamp_facts(tmp_path, ["t-ok"], rows=[(chat_chain.row_address(row), row)])["t-ok"]
    assert "outcome_final" not in entry
    line = _stamp_line(entry, 0)
    assert f"{unknown}; stamp from host_task_facts" in line and "final=false" not in line and "final=true" not in line
    # A kindless empty summary row is not a host facts row: it carries no finality fact at all.
    plain = {"type": "task_summary", "task_id": "x", "status": "running", "text": "",
             "result_ref": {"reader": "get_task_result", "task_id": "x"}}
    assert render_row_text(plain) == "host facts for x: status=running; result: get_task_result(task_id=x)"


def test_a_row_unreadable_by_its_address_shows_only_its_stored_facts(tmp_path, monkeypatch):
    from ouroboros import memory_view as mv

    _activated(tmp_path)
    _produce(tmp_path, "t-fail", "failed", "provider_unavailable")
    monkeypatch.setattr(mv, "_row_texts", lambda root, entries: {})  # every body unreadable by its address
    line = _lane_two(tmp_path)["t-fail"]
    assert "status=failed" in line and PRELIMINARY in line  # the stored facts carry the recorded finality
    assert "objective=" not in line  # the axes stay in the chain, never in stored facts
    old = tuple(field for field in mi._META_FIELDS if field not in {"outcome_final", "outcome_authority"})
    monkeypatch.setattr(mi, "_META_FIELDS", old)  # facts a body stored before the fields existed
    line = _lane_two(tmp_path)["t-fail"]
    assert "status=failed; " in line and "final=unknown" in line and "final=false" not in line


def _page_with_later_rows(root):
    """A page the mind wrote as task ``w1``; the task's host facts row and terminal row come after it."""
    row = {"chat_id": 1, "direction": "out", "task_id": "w1", "text": "work", "ts": "2026-10-26T10:01:00+00:00"}
    ds.append(root, "chat", row)
    store = ChronicleStore(root)
    published = store.publish_page(room_id="1", text="page", covers={"rows": [chat_chain.source_row_id(row)],
                                                                     "stream_span": [1, 1]},
                                   author={"kind": "mind", "task_id": "w1", "focus": {"role": "root", "task_id": "w1"}})
    assert published.ok, published
    _produce(root, "w1", "completed", "")
    _terminal(root, "w1", "completed", "")
    return next(record for record in store.room_records("1") if record["kind"] == "page")


def test_story_links_say_which_linked_row_is_preliminary(tmp_path):
    from ouroboros.memory_sealing import terminal_links

    _activated(tmp_path)
    record = _page_with_later_rows(tmp_path)
    links = terminal_links(tmp_path, record)
    assert len(links) == 1  # one line per page; each row's authority stays in the row its reader opens
    assert ("Later terminal rows of task w1 (host-linked, not page coverage): host summary (completed, "
            f"{PRELIMINARY.partition(';')[0]}); host summary (completed, final=true); memory_read(") in links[0]


def test_story_links_from_facts_stored_before_the_fields_name_the_gap(tmp_path, monkeypatch):
    from ouroboros.memory_sealing import terminal_links

    _activated(tmp_path)
    record = _page_with_later_rows(tmp_path)
    with monkeypatch.context() as patch:  # the stored tail an older body filled keeps its facts as they were
        patch.setattr(mi, "_META_FIELDS", tuple(field for field in mi._META_FIELDS
                                                if field not in {"outcome_final", "outcome_authority"}))
        assert mi.fill_tail(tmp_path, lambda: False)["state"] == "done"
    links = terminal_links(tmp_path, record)
    assert "host summary (completed, final=unknown); host summary (completed); memory_read(" in links[0]
    assert json.dumps(links).count("final=") == 1  # the gap alone: no finality invented for either row


def test_memory_read_shows_each_page_stamp_with_the_finality_and_authority_it_recorded(tmp_path):
    from ouroboros.tools.chronicle import _chronicle_write, _memory_read
    from ouroboros.tools.registry import ToolContext

    _activated(tmp_path)
    _produce(tmp_path, "t-ok", "completed", "")  # no terminal row: the page's stamp is the preliminary one
    _produce(tmp_path, "t-fail", "failed", "provider_unavailable")
    _terminal(tmp_path, "t-fail", "failed", "provider_unavailable")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root0001", current_chat_id=1)
    reply = json.loads(_chronicle_write(ctx, kind="page", text="Both shipped.", covers={"task_ids": ["t-ok", "t-fail"]}))
    assert reply["ok"], reply
    store = ChronicleStore(tmp_path)
    # Pages an older body stamped: a host facts entry without a finality, one whose authority alone was kept.
    legacy = {"task_id": "t-old", "status": "completed", "outcome": "Done", "outcome_phase": "done",
              "source": "host_task_facts", "result_ref": {"kind": "task_result", "task_id": "t-old"}}
    for n, (text, entry) in enumerate((("Old.", legacy), ("Kept.", {**legacy, "task_id": "t-auth",
                                                                   "outcome_authority": "pre_finalization_host_facts"}))):
        row = {"chat_id": 1, "direction": "in", "text": text, "client_message_id": f"old{n}",
               "ts": f"2026-10-26T11:0{n}:00+00:00"}
        ds.append(tmp_path, "chat", row)
        published = store.publish_page(room_id="1", text=text, covers={"rows": [chat_chain.source_row_id(row)],
                                                                       "stream_span": [1, 1]},
                                       author={"kind": "mind", "task_id": "root0001", "focus": {"role": "root", "task_id": "root0001"}},
                                       host_stamp={"tasks": [entry], "computed_at": "2026-10-01T00:00:00+00:00"})
        assert published.ok, published
    pages = {record["text"]: record["id"] for record in store.room_records("1") if record["kind"] == "page"}
    final = "final=true; authority=canonical_task_result_after_finalization"
    node = _memory_read(ctx, node_id=pages["Both shipped."])
    header = node.split("\n", 1)[0]
    assert f"stamp: t-ok=completed ({PRELIMINARY}), t-fail=failed ({final});" in header
    assert f"stamp t-ok: status=completed; outcome=Done; phase=done; {PRELIMINARY}; from host_task_facts" in node
    assert "stamp t-fail: status=failed; outcome=Failed; phase=error; verdict: " in node
    assert f"; {final}; from terminal_root_projection" in node
    old = _memory_read(ctx, node_id=pages["Old."])
    assert "stamp: t-old=completed (final=unknown);" in old.split("\n", 1)[0]
    assert "stamp t-old: status=completed; outcome=Done; phase=done; final=unknown; from host_task_facts" in old
    kept = _memory_read(ctx, node_id=pages["Kept."])
    assert "t-auth=completed (final=unknown; authority=pre_finalization_host_facts)" in kept
    assert "final=false" not in old + kept and "final=true" not in old + kept
    listing = _memory_read(ctx, room_id=1)
    assert f"t-ok=completed ({PRELIMINARY}), t-fail=failed ({final})" in listing
    assert "t-old=completed (final=unknown)" in listing
