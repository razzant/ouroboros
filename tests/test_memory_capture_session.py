"""One memory capture reads the chronicle under ONE index session, whatever the chronicle holds.

``capture_memory_view`` copies the caught-up index once (``ChronicleStore.read_session``) and reads
every record from that copy with the publication lock released. The per-read path it replaces took
the lock, connected, ran the schema and caught up the journal once per read: once per part member
and account source, per part and per room. Counters, not clocks: acquisitions of the publication
lock. The production-shape world (``tests._memory_production_shape``) is the fixture; a larger one
adds shown accounts that cite every story record, so the per-read cost grows with the record count.
"""
from __future__ import annotations

import contextlib

import pytest

from ouroboros import chronicle_store
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.memory_view import capture_memory_view, snapshot_json, view_spec_for_task
from tests import _memory_inventory_shared as shared
from tests import _memory_production_shape as shape

MAIN = {"id": "turn0001", "chat_id": 1}
ACCOUNTS = 12


def _world(root, accounts: int = 0):
    world = shape.production(root)
    store = ChronicleStore(root)
    sources = [*world.parts.values(), world.outer, world.gap_page]
    for index in range(accounts):
        account = store.publish_account(room_id="1", text=f"Synthetic account {index} across the rooms.",
                                        sources=sources, author=shared.MIND)
        assert account.ok, account
        assert store.select_account(account.record["id"], replaces=[], author=shared.MIND, reason="beside them").ok
    return world, len(sources)


@pytest.fixture
def sessions(monkeypatch):
    """How many times the chronicle's publication lock is taken (one per index session)."""
    counts = {"n": 0}
    acquire = chronicle_store.acquire_exclusive_file_lock

    def counting(*args, **kwargs):
        counts["n"] += 1
        return acquire(*args, **kwargs)

    monkeypatch.setattr(chronicle_store, "acquire_exclusive_file_lock", counting)
    return counts


def _capture(root, sessions) -> tuple:
    sessions["n"] = 0
    snapshot = capture_memory_view(root, MAIN, view_spec_for_task(MAIN, root))
    assert snapshot.active and snapshot.story
    return snapshot_json(snapshot), sessions["n"]


@contextlib.contextmanager
def _per_read(self):
    """The path before read sessions: every read of the capture enters the live index again."""
    yield self


def test_a_capture_takes_the_index_lock_a_constant_number_of_times(tmp_path, sessions):
    small, large = tmp_path / "small", tmp_path / "large"
    _world(small)
    _world(large, ACCOUNTS)
    counts = [_capture(root, sessions)[1] for root in (small, large, small, large)]  # cold, then warm caches
    assert counts[0] == counts[1] == counts[2] == counts[3] <= 3, counts


def test_the_session_capture_is_byte_identical_to_the_per_read_path(tmp_path, sessions, monkeypatch):
    _unused, sources = _world(tmp_path, ACCOUNTS)
    session_text, session_count = _capture(tmp_path, sessions)
    monkeypatch.setattr(ChronicleStore, "read_session", _per_read)
    per_read_text, per_read_count = _capture(tmp_path, sessions)
    assert per_read_text == session_text
    assert per_read_count > ACCOUNTS * sources > session_count  # the cost the session removed grows with records


def test_a_read_session_releases_the_lock_keeps_its_copy_and_publishes_nothing(tmp_path, sessions):
    shared.world(tmp_path)
    store = ChronicleStore(tmp_path)
    with store.read_session() as session:
        copied, identity = session.records(), session.authority()
        sessions["n"] = 0
        note = store.write_note(room_id="1", task_id="t9", text="Written while a reader holds its copy.", author=shared.MIND)
        assert note.ok and sessions["n"] == 1  # the writer took the free lock at once
        assert session.records() == copied and session.authority() == identity
        assert store.authority() != identity and store.get(note.record["id"]) is not None
        with pytest.raises(RuntimeError, match="publishes nothing"):
            session.write_note(room_id="1", task_id="t9", text="Not from a copy.", author=shared.MIND)
    assert len(store.records()) == len(copied) + 1  # the refused write appended nothing
