"""The open-row projection reads each chat byte once per process and agrees with a cold read.

``memory_inventory`` keeps row metadata after the legacy frontier and, on later calls,
reads only what the chain gained: appended bytes of the live file, a rotated
generation's tail, a new live file. Every check compares the projection with
``chat_chain.iter_rows`` read from the chain's start (same positions, same addresses,
same metadata), through rotation, rewrite, a Project binding made later and a line
still being written.
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
    """Every physical read of the projection as ``(file name, start byte)``."""
    calls = []
    original = mi._read_lines

    def recording(path, start, *, live):
        calls.append((pathlib.Path(path).name, start))
        return original(path, start, live=live)

    monkeypatch.setattr(mi, "_read_lines", recording)
    return calls


def _frontier(root):
    return ci.legacy_frontier(ChronicleStore(root))["pos"]


def _live(root):
    return root / "logs" / "chat.jsonl"


def test_a_later_call_reads_only_the_bytes_appended_to_the_live_file(tmp_path, reads):
    shared.world(tmp_path)
    first = mi.rows_after_frontier(tmp_path)
    assert first == shared.cold(tmp_path, 10) and [pos for _a, _m, pos in first] == list(range(10, 20))
    # The frontier's own generation is read once; the first archive (all legacy) never is.
    assert reads == [(shared.ARCHIVE_TWO, 0), ("chat.jsonl", 0)]
    size = _live(tmp_path).stat().st_size
    shared.append(_live(tmp_path), shared.msg("2026-09-04T00:00:00+00:00", "new one", client_message_id="n1"),
                  shared.msg("2026-09-04T00:01:00+00:00", "new two", direction="out", task_id="t9"))
    reads.clear()
    second = mi.rows_after_frontier(tmp_path)
    assert reads == [("chat.jsonl", size)]
    assert second == shared.cold(tmp_path, 10) and [pos for _a, _m, pos in second][-2:] == [20, 21]
    reads.clear()
    assert mi.rows_after_frontier(tmp_path) == second and reads == []  # nothing new, nothing read


def test_rotation_keeps_every_row_and_reads_only_the_new_bytes(tmp_path, reads):
    shared.world(tmp_path)
    mi.rows_after_frontier(tmp_path)
    read_up_to = _live(tmp_path).stat().st_size
    # A row lands before the rotation and is first seen inside the rotated archive.
    shared.append(_live(tmp_path), shared.msg("2026-09-04T00:00:00+00:00", "before rotation", client_message_id="r0"))
    rotated = tmp_path / "archive" / "chat_20260904T120000.jsonl"
    os.replace(_live(tmp_path), rotated)
    shared.append(_live(tmp_path), shared.msg("2026-09-05T00:00:00+00:00", "after rotation", client_message_id="r1"))
    reads.clear()
    rows = mi.rows_after_frontier(tmp_path)
    assert rows == shared.cold(tmp_path, 10)
    assert [pos for _a, _m, pos in rows][-2:] == [20, 21]
    # The rotated generation continues where the live file stopped; the new live file starts at 0.
    assert reads == [(rotated.name, read_up_to), ("chat.jsonl", 0)]


def test_a_binding_made_later_moves_earlier_rows_without_rereading(tmp_path, reads):
    from ouroboros.projects_registry import bind_task_to_project

    rooms = shared.world(tmp_path)
    beta = str(rooms["beta"])
    assert [pos for _a, _m, pos in mi.open_room_rows(tmp_path, beta)] == [14]
    assert {12, 16} <= {pos for _a, _m, pos in mi.open_room_rows(tmp_path, "1")}
    reads.clear()
    bind_task_to_project(tmp_path, "t2", "beta", origin={"absent": "post_hoc_unresolved"})
    assert [pos for _a, _m, pos in mi.open_room_rows(tmp_path, beta)] == [12, 14, 16]
    assert not {12, 16} & {pos for _a, _m, pos in mi.open_room_rows(tmp_path, "1")}
    assert reads == []


def test_a_line_still_being_written_waits_for_its_newline(tmp_path, reads):
    shared.world(tmp_path)
    mi.rows_after_frontier(tmp_path)
    complete = _live(tmp_path).stat().st_size
    with _live(tmp_path).open("ab") as handle:
        handle.write(b'{"chat_id": 1, "direction": "in", "ts": "2026-09-04T00:00:00+00:00", "text": "par')
    rows = mi.rows_after_frontier(tmp_path)
    assert [pos for _a, _m, pos in rows][-1] == 19 and rows == shared.cold(tmp_path, 10)
    projection = mi._PROJECTIONS[str(tmp_path.resolve())]
    assert projection.generations[-1].consumed == complete  # the offset does not move past the open line
    with _live(tmp_path).open("ab") as handle:
        handle.write(b'tial", "client_message_id": "p1"}\n')
    rows = mi.rows_after_frontier(tmp_path)
    assert rows == shared.cold(tmp_path, 10) and rows[-1][2] == 20 and rows[-1][1]["text_chars"] == len("partial")
    assert reads[-1] == ("chat.jsonl", complete)


def test_a_rewritten_live_file_rebuilds_the_projection(tmp_path):
    shared.world(tmp_path)
    mi.rows_after_frontier(tmp_path)
    _live(tmp_path).write_text("", encoding="utf-8")
    shared.append(_live(tmp_path), shared.msg("2026-09-06T00:00:00+00:00", "replaced", client_message_id="x"))
    rows = mi.rows_after_frontier(tmp_path)
    assert rows == shared.cold(tmp_path, 10) and [pos for _a, _m, pos in rows] == [10]


def test_with_nothing_covered_every_row_is_read_from_the_chain_start(tmp_path, reads):
    shared.world(tmp_path, legacy=False)
    assert _frontier(tmp_path) == 0
    rows = mi.rows_after_frontier(tmp_path)
    assert rows == shared.cold(tmp_path, 0) and len(rows) == 20
    assert [name for name, _start in reads] == [shared.ARCHIVE_ONE, shared.ARCHIVE_TWO, "chat.jsonl"]


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
    mi.rows_after_frontier(tmp_path)
    reads.clear()
    ref = build_owner_message_ref(chat_id=1, client_message_id="m1", ts="2026-09-03T00:01:00+00:00",
                                  text="next please")
    bind_task_to_project(tmp_path, "tb3", "beta", origin={"ref": ref, "text": "next please"})
    # Membership by the bound owner message comes from metadata alone: no row is read again.
    assert 11 in [pos for _a, _m, pos in mi.open_room_rows(tmp_path, str(rooms["beta"]))]
    assert 11 in [pos for _a, _m, pos in mi.open_room_rows(tmp_path, "1")]
    assert reads == []
