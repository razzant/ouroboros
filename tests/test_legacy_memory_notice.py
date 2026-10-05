"""The owner hears once, after an update, that the old memory folds gradually.

``upgrade_notices.startup_upgrade_notices`` owes a ``legacy_memory_notice`` only
when the chronicle imported old dialogue memory: the facts (pieces, periods,
span) come from the imported records without a model, the chat row is the
receipt, and ``state.json`` is marked only after that row was written. A fresh
install hears nothing and gets no chronicle; an unbound owner chat or a pending
import leaves the notice owed. Every rule is pinned in both directions with the
real chat, state and chronicle writers.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from ouroboros import upgrade_notices as notices
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.utils import iter_jsonl_chain_objects, jsonl_generation_signature

NOTICE = "legacy_memory_notice"


def _rows():
    """Three August rows (block 0), two September rows (block 1) and one row after the old cursor."""
    days = ["2026-08-01", "2026-08-02", "2026-08-03", "2026-09-04", "2026-09-05", "2026-09-06"]
    return [{"chat_id": 1, "direction": "in", "ts": f"{day}T10:00:00+00:00", "text": f"row {i}"}
            for i, day in enumerate(days)]


def _old_memory(root, *, blocks=True, flat=None):
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "chat.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _rows()), encoding="utf-8")
    memory = root / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    if blocks is True:
        blocks = [
            {"ts": "2026-08-04T01:00:00+00:00", "type": "summary", "range": "2026-08-01 to 2026-08-03",
             "message_count": 3, "content": "### August",
             "rooms": [{"room_id": "1", "label": "Main", "message_count": 2, "content": "I talked with the owner."},
                       {"room_id": "7", "label": "Project seven", "message_count": 1, "content": "Seven began."}]},
            {"ts": "2026-09-05T12:00:00+00:00", "type": "era", "range": "2026-09-04 to 2026-09-05",
             "message_count": 2, "content": "### September",
             "rooms": [{"room_id": "1", "label": "Main", "message_count": 2, "content": "An era of Main."}]},
        ]
    if blocks is not None:
        data = blocks if isinstance(blocks, bytes) else json.dumps(blocks).encode("utf-8")
        (memory / "dialogue_blocks.json").write_bytes(data)
    meta = {"chat_log_signature": jsonl_generation_signature(root / "logs" / "chat.jsonl"),
            "last_consolidated_offset": 5}
    (memory / "dialogue_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if flat is not None:
        (memory / "dialogue_summary.md").write_text(flat, encoding="utf-8")


@pytest.fixture
def boot(monkeypatch, tmp_path):
    """One install on the real state and owner-chat writers; the two settings notices already went."""
    from supervisor import message_bus as bus, state as ss

    for name, path in {"DRIVE_ROOT": tmp_path, "STATE_PATH": tmp_path / "state/state.json",
                       "STATE_LAST_GOOD_PATH": tmp_path / "state/state.last_good.json",
                       "STATE_LOCK_PATH": tmp_path / "locks/state.lock"}.items():
        monkeypatch.setattr(ss, name, path)
    monkeypatch.setattr(bus, "DATA_DIR", tmp_path)
    delivered = []
    monkeypatch.setattr(bus, "get_bridge", lambda: SimpleNamespace(send_message=lambda *a, **kw: delivered.append(a)))
    ss.save_state({"owner_chat_id": 7, "owner_id": 1, notices.REVIEWER_DEFAULT_NOTICE_KEY: "already",
                   notices.OPTIONAL_BOUNDS_NOTICE_KEY: "already"})

    def rows():
        return [row for row in iter_jsonl_chain_objects(tmp_path / "logs" / "chat.jsonl") if row.get("type") == NOTICE]

    return SimpleNamespace(root=tmp_path, state=ss, bus=bus, rows=rows, delivered=delivered,
                           run=lambda: notices.startup_upgrade_notices({}))


def test_an_install_with_old_memory_hears_once_how_much_and_how_to_fold_it(boot):
    _old_memory(boot.root)
    boot.run()
    rows = boot.rows()
    assert len(rows) == 1 and rows[0]["direction"] == "system" and rows[0]["chat_id"] == 7
    text = rows[0]["text"]
    assert "3 pieces over 2 periods (2026-08-01 to 2026-09-05)" in text  # the sections' own chat rows
    assert "works as it is" in text and "folded into the new format gradually, part by part" in text
    assert "ask Ouroboros to keep folding it; it will tell you how much is left" in text
    assert boot.state.load_state()[notices.LEGACY_MEMORY_NOTICE_KEY]
    # The facts came from the same import the first memory view runs, and the old files are untouched.
    store = ChronicleStore(boot.root)
    assert store.activation()["kind"] == "activation"
    assert {r["id"] for r in store.records(kinds=("legacy",))} == {"legacy-b00-r1", "legacy-b00-r7", "legacy-b01-r1"}
    boot.run()
    boot.run()
    assert len(boot.rows()) == 1 and len(boot.delivered) == 1  # never repeated


def test_a_written_row_without_its_marker_is_recovered_without_a_second_row(boot, monkeypatch):
    _old_memory(boot.root)
    real_update = boot.state.update_state
    failed = []

    def update(fn):
        if not failed:
            failed.append(True)
            raise OSError("injected state write failure")
        return real_update(fn)

    monkeypatch.setattr(boot.state, "update_state", update)
    boot.run()
    assert len(boot.rows()) == 1 and not boot.state.load_state().get(notices.LEGACY_MEMORY_NOTICE_KEY)
    boot.run()
    assert len(boot.rows()) == 1 and boot.state.load_state()[notices.LEGACY_MEMORY_NOTICE_KEY]


def test_a_failed_chat_write_leaves_the_notice_owed(boot, monkeypatch):
    _old_memory(boot.root)
    real_append = boot.bus.append_jsonl
    failed = []

    def append(*a, **kw):
        if not failed:
            failed.append(True)
            raise OSError("injected chat write failure")
        return real_append(*a, **kw)

    monkeypatch.setattr(boot.bus, "append_jsonl", append)
    boot.run()
    assert boot.rows() == [] and not boot.state.load_state().get(notices.LEGACY_MEMORY_NOTICE_KEY)
    boot.run()
    assert len(boot.rows()) == 1 and boot.state.load_state()[notices.LEGACY_MEMORY_NOTICE_KEY]


def test_a_fresh_install_hears_nothing_and_gets_no_chronicle(boot):
    (boot.root / "logs").mkdir(parents=True, exist_ok=True)
    (boot.root / "logs" / "chat.jsonl").write_text(json.dumps(_rows()[0]) + "\n", encoding="utf-8")
    (boot.root / "memory").mkdir()
    (boot.root / "memory" / "dialogue_meta.json").write_text("{}", encoding="utf-8")  # a cursor holds no retelling
    boot.run()
    assert boot.rows() == [] and boot.delivered == []
    assert not (boot.root / "memory" / "chronicle").exists()
    assert notices.LEGACY_MEMORY_NOTICE_KEY not in boot.state.load_state()
    # The same install, once the old writer's retelling is there, hears it.
    _old_memory(boot.root)
    boot.run()
    assert len(boot.rows()) == 1


@pytest.mark.parametrize("journal", ["activated_without_old_memory", "only_gaps"])
def test_an_activated_journal_without_legacy_pieces_hears_nothing(boot, journal):
    if journal == "only_gaps":
        _old_memory(boot.root, blocks=b"{broken")  # an unreadable block file imports as a gap, not a piece
    else:
        (boot.root / "logs").mkdir(parents=True, exist_ok=True)
        (boot.root / "logs" / "chat.jsonl").write_text(json.dumps(_rows()[0]) + "\n", encoding="utf-8")
    store = ChronicleStore(boot.root)
    assert store.ensure_activated()["kind"] == "activation"
    assert all(p["legacy_type"] in ("gap", "cursor_gap") for p in store.legacy_pointer_rows())
    boot.run()
    assert boot.rows() == [] and notices.LEGACY_MEMORY_NOTICE_KEY not in boot.state.load_state()


def test_a_flat_summary_is_a_piece_without_a_period(boot):
    _old_memory(boot.root, blocks=b"{broken", flat="What I remembered as one flat summary.")
    boot.run()
    rows = boot.rows()
    assert len(rows) == 1
    assert "previous format — 1 piece. It works" in rows[0]["text"]


def test_an_unbound_owner_chat_sends_marks_and_imports_nothing(boot):
    _old_memory(boot.root)
    boot.state.update_state(lambda st: st.__setitem__("owner_chat_id", 0))
    boot.run()
    assert boot.rows() == [] and boot.delivered == []
    assert notices.LEGACY_MEMORY_NOTICE_KEY not in boot.state.load_state()
    assert not (boot.root / "memory" / "chronicle").exists()
    boot.state.update_state(lambda st: st.__setitem__("owner_chat_id", 7))
    boot.run()
    assert len(boot.rows()) == 1 and boot.state.load_state()[notices.LEGACY_MEMORY_NOTICE_KEY]


def test_a_pending_import_keeps_the_notice_owed(boot):
    from ouroboros.platform_layer import file_lock_exclusive_nb, file_unlock

    _old_memory(boot.root)
    fd = os.open(str(boot.root / "memory" / ".consolidation.lock"), os.O_CREAT | os.O_WRONLY, 0o644)
    file_lock_exclusive_nb(fd)  # another importer (or an old writer of a running earlier version)
    try:
        boot.run()
        assert boot.rows() == [] and notices.LEGACY_MEMORY_NOTICE_KEY not in boot.state.load_state()
        assert ChronicleStore(boot.root).activation() is None
    finally:
        file_unlock(fd)
        os.close(fd)
    boot.run()
    assert len(boot.rows()) == 1 and boot.state.load_state()[notices.LEGACY_MEMORY_NOTICE_KEY]


def test_a_refused_import_keeps_the_notice_owed(boot):
    _old_memory(boot.root)
    squatter = {"id": "legacy-b00-r1", "kind": "legacy", "room_id": "1", "text": "someone else's",
                "author": {"kind": "legacy_helper"}}
    assert ChronicleStore(boot.root).publish([squatter]).ok
    boot.run()
    assert boot.rows() == [] and notices.LEGACY_MEMORY_NOTICE_KEY not in boot.state.load_state()


def test_an_unreadable_journal_leaves_only_this_notice_owed(boot, monkeypatch):
    _old_memory(boot.root)
    boot.state.update_state(lambda st: st.pop(notices.REVIEWER_DEFAULT_NOTICE_KEY))
    real_pointers, broken = ChronicleStore.legacy_pointer_rows, [True]

    def pointers(self):
        if broken:
            raise ValueError("chronicle authority shortened")
        return real_pointers(self)

    monkeypatch.setattr(ChronicleStore, "legacy_pointer_rows", pointers)
    boot.run()
    sent = [row.get("type") for row in iter_jsonl_chain_objects(boot.root / "logs" / "chat.jsonl")
            if row.get("direction") == "system"]
    assert sent == ["reviewer_default_notice"]
    assert notices.LEGACY_MEMORY_NOTICE_KEY not in boot.state.load_state()
    broken.clear()
    boot.run()
    assert len(boot.rows()) == 1


@pytest.mark.parametrize("facts, expected", [
    ({"pieces": 394, "periods": 23, "start": "2026-07-30", "end": "2026-09-30", "labels": []},
     "394 pieces over 23 periods (2026-07-30 to 2026-09-30)."),
    ({"pieces": 1, "periods": 1, "start": "2026-09-01", "end": "2026-09-01", "labels": ["x"]},
     "1 piece over 1 period (2026-09-01)."),
    ({"pieces": 2, "periods": 2, "start": None, "end": None,
      "labels": ["2026-07-30 to 2026-09-25", "2026-09-30 00:04 - 03:15"]},
     "2 pieces over 2 periods (labelled 2026-07-30 to 2026-09-25 … 2026-09-30 00:04 - 03:15)."),
    ({"pieces": 3, "periods": 0, "start": None, "end": None, "labels": []}, "3 pieces."),
])
def test_the_notice_states_facts_only(facts, expected):
    text = notices.legacy_memory_notice(facts)
    assert expected in text and text.endswith("it will tell you how much is left.")
    # No price, duration, schedule or UI surface: the owner decides on facts.
    for word in ("$", "cost", "minute", "hour", "second", "first turn", "Settings"):
        assert word not in text
    # No promise that one request folds everything: on the acceptance stand one request
    # folded 23 of 377 retellings. The previous wording ("To fold it all now, ask ...") trips this.
    for promise in ("all now", "at once", "fold it all"):
        assert promise not in text
    assert "part by part" in text and "keep folding" in text
    assert notices.legacy_memory_notice({**facts, "pieces": 0}) == ""
    assert notices.legacy_memory_notice(None) == ""
