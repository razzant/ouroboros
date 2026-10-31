"""The SQL open-row inventory preserves positions, bindings and imported history.

Reads after import never reopen a frozen chat file, even after it changes or moves.
New SQL rows remain visible across calls without a process-owned projection cache.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from ouroboros import chronicle_import as ci
from ouroboros import memory_inventory as mi
from ouroboros.chronicle_store import ChronicleStore
from tests import _memory_inventory_shared as shared


@pytest.fixture
def reads(monkeypatch):
    """Observe legacy chat-file reads; SQL queries may run on every inventory call."""
    calls = []
    original = pathlib.Path.open

    def recording(path, mode="r", *args, **kwargs):
        if path.name.startswith("chat") and path.suffix == ".jsonl" and "r" in mode:
            calls.append(path.name)
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", recording)
    return calls


def _frontier(root):
    return ci.legacy_frontier(ChronicleStore(root))["pos"]


def _live(root):
    return root / "logs" / "chat.jsonl"


def test_a_later_call_sees_appended_sql_rows_without_opening_legacy_files(tmp_path, reads):
    shared.world(tmp_path)
    reads.clear()
    first = mi.rows_after_frontier(tmp_path)
    assert first == shared.cold(tmp_path, 10) and [pos for _a, _m, pos in first] == list(range(10, 20))
    assert reads == []
    shared.append(_live(tmp_path), shared.msg("2026-09-04T00:00:00+00:00", "new one", client_message_id="n1"),
                  shared.msg("2026-09-04T00:01:00+00:00", "new two", direction="out", task_id="t9"))
    reads.clear()
    second = mi.rows_after_frontier(tmp_path)
    assert reads == []
    assert second == shared.cold(tmp_path, 10) and [pos for _a, _m, pos in second][-2:] == [20, 21]
    reads.clear()
    assert mi.rows_after_frontier(tmp_path) == second and reads == []  # nothing new, nothing read


def test_moving_the_frozen_file_does_not_change_sql_rows(tmp_path, reads):
    shared.world(tmp_path)
    reads.clear()
    mi.rows_after_frontier(tmp_path)
    # SQL rows land on both sides of a legacy-file move.
    shared.append(_live(tmp_path), shared.msg("2026-09-04T00:00:00+00:00", "before rotation", client_message_id="r0"))
    rotated = tmp_path / "archive" / "chat_20260904T120000.jsonl"
    os.replace(_live(tmp_path), rotated)
    shared.append(_live(tmp_path), shared.msg("2026-09-05T00:00:00+00:00", "after rotation", client_message_id="r1"))
    reads.clear()
    rows = mi.rows_after_frontier(tmp_path)
    assert rows == shared.cold(tmp_path, 10)
    assert [pos for _a, _m, pos in rows][-2:] == [20, 21]
    assert reads == []


def test_a_binding_made_later_moves_earlier_rows_without_rereading(tmp_path, reads):
    from ouroboros.projects_registry import bind_task_to_project

    rooms = shared.world(tmp_path)
    reads.clear()
    beta = str(rooms["beta"])
    assert [pos for _a, _m, pos in mi.open_room_rows(tmp_path, beta)] == [14]
    assert {12, 16} <= {pos for _a, _m, pos in mi.open_room_rows(tmp_path, "1")}
    reads.clear()
    bind_task_to_project(tmp_path, "t2", "beta", origin={"absent": "post_hoc_unresolved"})
    assert [pos for _a, _m, pos in mi.open_room_rows(tmp_path, beta)] == [12, 14, 16]
    assert not {12, 16} & {pos for _a, _m, pos in mi.open_room_rows(tmp_path, "1")}
    assert reads == []


def test_an_unimported_append_to_the_frozen_file_does_not_advance_the_inventory(tmp_path, reads):
    shared.world(tmp_path)
    reads.clear()
    first = mi.rows_after_frontier(tmp_path)
    with _live(tmp_path).open("ab") as handle:
        handle.write(b'{"chat_id": 1, "direction": "in", "text": "legacy append"}\n')
    assert mi.rows_after_frontier(tmp_path) == first == shared.cold(tmp_path, 10)
    assert reads == []


def test_a_rewritten_frozen_file_does_not_erase_imported_sql_rows(tmp_path, reads):
    shared.world(tmp_path)
    reads.clear()
    mi.rows_after_frontier(tmp_path)
    _live(tmp_path).write_text("", encoding="utf-8")
    shared.append(_live(tmp_path), shared.msg("2026-09-06T00:00:00+00:00", "replaced", client_message_id="x"))
    rows = mi.rows_after_frontier(tmp_path)
    assert rows == shared.cold(tmp_path, 10) and [pos for _a, _m, pos in rows] == list(range(10, 21))
    assert reads == []


def test_with_nothing_covered_every_row_is_read_from_the_chain_start(tmp_path, reads):
    shared.world(tmp_path, legacy=False)
    reads.clear()
    assert _frontier(tmp_path) == 0
    rows = mi.rows_after_frontier(tmp_path)
    assert rows == shared.cold(tmp_path, 0) and len(rows) == 20
    assert reads == []


def test_an_unknown_frontier_anchors_at_the_chain_end_seen_at_activation(tmp_path):
    shared.world(tmp_path, cursor=False)
    assert _frontier(tmp_path) == 20
    assert mi.rows_after_frontier(tmp_path) == []
    shared.append(_live(tmp_path), shared.msg("2026-09-04T00:00:00+00:00", "later", client_message_id="l1"))
    rows = mi.rows_after_frontier(tmp_path)
    assert rows == shared.cold(tmp_path, 20) and [pos for _a, _m, pos in rows] == [20]


def test_inbound_rows_carry_the_text_hash_a_source_ref_binding_matches(tmp_path, reads):
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import bind_task_to_project

    rooms = shared.world(tmp_path)
    reads.clear()
    mi.rows_after_frontier(tmp_path)
    reads.clear()
    ref = build_owner_message_ref(chat_id=1, client_message_id="m1", ts="2026-09-03T00:01:00+00:00",
                                  text="next please")
    bind_task_to_project(tmp_path, "tb3", "beta", origin={"ref": ref, "text": "next please"})
    # Membership follows the newly bound owner address; no legacy file is reopened.
    assert 11 in [pos for _a, _m, pos in mi.open_room_rows(tmp_path, str(rooms["beta"]))]
    assert 11 in [pos for _a, _m, pos in mi.open_room_rows(tmp_path, "1")]
    assert reads == []
