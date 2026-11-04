"""A switch to a body that reads ``state/dialog.sqlite`` but no provider thread held by reference.

The release before threads were held once (``BASE``) has the dialogue store, schema 2 included, and no reader
of a reference: handed a new Slack, GitHub or transport-queue row as stored, it would rebuild the event with
``{"$shared": …}`` objects where the messages were. The switch seam therefore names that tree's need
(``store_switch.switch_needs``: ``threads``) and, before the file is handed over, writes every row held by
reference back as its exact line (``dialogue_import.expand_for_older_body``). These tests run the real readers
of ``BASE``, read from Git, over the rows this body wrote.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import types

import pytest

from ouroboros import chat_chain, dialogue_import as di, dialogue_store as ds, store, store_switch
from tests import _dialogue as dialogue_test
from tests.test_dialogue_threads_once import SHAPES, TS, presence_row, slack

REPO = pathlib.Path(__file__).resolve().parents[1]
BASE = "7ce4e1520"  # the release before this body held a thread by reference
_ENV = {**os.environ, "LC_ALL": "C", "LANG": "C", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True, env=_ENV).stdout.strip()


def _base_source(rel: str) -> bytes:
    found = subprocess.run(["git", "show", f"{BASE}:{rel}"], cwd=REPO, capture_output=True, env=_ENV)
    if found.returncode:
        pytest.skip(f"{BASE} is not in this clone's history")
    return found.stdout


def _trees(tmp_path: pathlib.Path):
    """``older``: every store module, its dialogue store the base body's own; ``current``: this body's."""
    repo = tmp_path / "repo"
    (repo / "ouroboros").mkdir(parents=True)
    _git(repo, "init", "-q")
    for name in ("store.py", "usage_store.py", "evidence_packs.py"):
        (repo / "ouroboros" / name).write_text("# store\n", encoding="utf-8")
    trees = []
    for name, source in (("older", _base_source("ouroboros/dialogue_store.py")),
                         ("current", (REPO / "ouroboros" / "dialogue_store.py").read_bytes())):
        (repo / "ouroboros" / "dialogue_store.py").write_bytes(source)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", name)
        trees.append(_git(repo, "rev-parse", "HEAD"))
    return repo, *trees


def _checkout(repo: pathlib.Path):
    """The door's switching owner, as the real ones leave it: HEAD on the target."""
    def owner(intent):
        _git(repo, "checkout", "-q", "--detach", intent["target_sha"])
        return True, f"switched to {intent['target_sha'][:8]}"
    return owner


def _bodies(root: pathlib.Path):
    with ds.transaction(root, write=False) as conn:
        return [record["body"] for record in conn.execute("SELECT body FROM dialogue WHERE stream='chat' ORDER BY seq")]


def _base_readers(monkeypatch):
    """``dialogue_store``, ``presence_events`` and ``place_intake`` of ``BASE``, bound where that body's own imports
    find them (``from ouroboros import dialogue_store``, ``presence_runner.PresenceTurnEvent``)."""
    import ouroboros
    from ouroboros import presence_runner

    loaded = {}
    for name in ("dialogue_store", "presence_events", "place_intake"):
        module = types.ModuleType(f"ouroboros.{name}")
        module.__package__, module.__file__ = "ouroboros", f"{BASE}:ouroboros/{name}.py"
        exec(compile(_base_source(f"ouroboros/{name}.py"), module.__file__, "exec"), module.__dict__)
        loaded[name] = module
    monkeypatch.setitem(sys.modules, "ouroboros.dialogue_store", loaded["dialogue_store"])
    monkeypatch.setattr(ouroboros, "dialogue_store", loaded["dialogue_store"])
    monkeypatch.setattr(presence_runner, "PresenceTurnEvent", loaded["presence_events"].PresenceTurnEvent)
    return types.SimpleNamespace(**loaded)


def _thread_rows(root: pathlib.Path):
    """One row of every thread shape (one revised after it was stored) and a row without a thread, as written."""
    rows = [presence_row(SHAPES[shape](3), cmid=f"rollback-{shape}", task_id=f"turn-{shape}",
                         ts=f"2026-11-04T10:00:0{i}+00:00") for i, shape in enumerate(sorted(SHAPES))]
    rows.append({"ts": TS, "direction": "out", "chat_id": 912345, "text": "a reply"})
    receipts = [ds.append(root, "chat", row) for row in rows]
    rows[0] = ds.revise_text(root, "chat", receipts[0].seq, "the message, edited")
    return rows


def test_a_switch_to_a_body_that_reads_no_reference_hands_its_real_readers_every_thread_whole(tmp_path, monkeypatch):
    root = tmp_path / "data"
    rows = _thread_rows(root)
    assert all(ds.ROW_SHA_KEY in ds.decode(body) for body in _bodies(root)[:3]), "the threads are held by reference"
    repo, older, _current = _trees(tmp_path)
    assert store_switch.switch_needs(repo, older) == {
        "facts": False, "money": False, "handles": False, "dialogue": False, "threads": True}
    ok, _message, deferred = store_switch.switch_or_reset(
        root, repo, target_sha=older, target_ref="older", reason="test", door="ui_rollback",
        reset=lambda: pytest.fail("a body that reads no reference must not take the file now"))
    assert ok and deferred, "the intent waits for the seam at the restart exit, every writer dead"
    outcome = store_switch.run_switch_seam(root, repo_dir=repo, owner=_checkout(repo))
    assert outcome["status"] == "switched" and outcome["needs"]["threads"], outcome
    assert _bodies(root) == [ds.encode(row) for row in rows], "each row is its exact line again"
    assert ds.meta(root, ds.SHARED_FROM + "chat") is None

    base = _base_readers(monkeypatch)
    for shape, row in zip(sorted(SHAPES), rows):
        _task, event = base.place_intake._captured_arrival(
            root, {"focus": {"place_id": "place-1"}}, f"turn-{shape}", "e" * 64, None)
        assert event.conversation == row["transport"]["conversation"] and event.text == row["text"], shape
    assert [entry.row for entry in base.dialogue_store.newest(root, "chat", limit=10)] == rows


def test_a_switch_to_a_body_that_reads_references_leaves_the_rows_as_stored(tmp_path):
    root = tmp_path / "data"
    rows = _thread_rows(root)
    stored = _bodies(root)
    repo, _older, current = _trees(tmp_path)
    assert store_switch.switch_needs(repo, current) == dict.fromkeys(
        ("facts", "money", "handles", "dialogue", "threads"), False)
    assert store_switch.switch_or_reset(root, repo, target_sha=current, target_ref="current", reason="test",
                                        door="ui_rollback", reset=lambda: (True, "reset now")) == (True, "reset now", False)
    assert _bodies(root) == stored and ds.meta(root, ds.SHARED_FROM + "chat") is not None
    assert dialogue_test.chat_rows(root) == rows


def test_this_bodys_dialogue_store_names_the_format_it_reads():
    """``switch_needs`` reads the marker from a tree's ``dialogue_store.py`` bytes: this body's must carry it, or
    every restart would take its own checkout for a body that reads no reference."""
    assert ds.THREAD_FORMAT.encode("ascii") in (REPO / "ouroboros" / "dialogue_store.py").read_bytes()


# ---- the expansion itself -----------------------------------------------------------------------

def test_the_first_row_held_by_reference_marks_where_the_expansion_starts(tmp_path):
    plain = ds.append(tmp_path, "chat", {"ts": TS, "direction": "in", "chat_id": 5, "text": "before"})
    assert ds.meta(tmp_path, ds.SHARED_FROM + "chat") is None, "a row without a thread marks nothing"
    first = ds.append(tmp_path, "chat", presence_row(slack(2), cmid="m-1", ts="2026-11-04T10:00:01+00:00"))
    ds.append(tmp_path, "chat", presence_row(slack(3), cmid="m-2", ts="2026-11-04T10:00:02+00:00"))
    assert ds.meta(tmp_path, ds.SHARED_FROM + "chat") == first.seq > plain.seq
    assert ds.meta(tmp_path, ds.SHARED_FROM + "progress") is None


def test_the_expansion_writes_each_held_row_as_its_line_batch_by_batch(tmp_path):
    rows = [presence_row(slack(n), cmid=f"m-{n}", ts=f"2026-11-04T10:00:0{n}+00:00") for n in range(1, 5)]
    rows.insert(2, {"ts": TS, "direction": "in", "chat_id": 5, "text": "between"})
    for row in rows:
        ds.append(tmp_path, "chat", row)
    with ds.transaction(tmp_path, write=False) as conn:
        spans = [(record["seq"], record["length"]) for record in conn.execute("SELECT seq, length FROM dialogue ORDER BY seq")]
    report = di.expand_for_older_body(tmp_path, batch=1)
    assert report == {str(tmp_path): {"chat": {"rows": 4}, "progress": {"rows": 0}}}
    assert _bodies(tmp_path) == [ds.encode(row) for row in rows]
    with ds.transaction(tmp_path, write=False) as conn:
        assert [(record["seq"], record["length"]) for record in conn.execute(
            "SELECT seq, length FROM dialogue ORDER BY seq")] == spans, "every coordinate and span stays"
    # This body reads the whole rows as it read the stored ones: the same rows, the same identities.
    assert dialogue_test.chat_rows(tmp_path) == rows
    assert [chat_chain.source_row_id(row) for row in rows] == [
        chat_chain.source_row_id(ds.decode(body)) for body in _bodies(tmp_path)]
    assert ds.meta(tmp_path, ds.SHARED_FROM + "chat") is None
    assert di.expand_for_older_body(tmp_path)[str(tmp_path)]["chat"] == {"rows": 0}, "nothing left to expand"


def test_a_part_the_store_does_not_hold_fails_the_switch_typed_and_no_row_is_shorter(tmp_path):
    root = tmp_path / "data"
    rows = _thread_rows(root)
    stored = _bodies(root)
    lost = ds.thread_references(ds.decode(stored[1])["transport"]["conversation"])[0]
    with ds.transaction(root, write=True) as conn:
        part = conn.execute("SELECT value FROM shared WHERE sha256=?", (lost,)).fetchone()["value"]
        conn.execute("DELETE FROM shared WHERE sha256=?", (lost,))
    with pytest.raises(store.StoreLost):
        di.expand_for_older_body(root, batch=1)
    after = _bodies(root)
    assert after[0] == ds.encode(rows[0]), "the batch before the lost part is whole"
    assert after[1:] == stored[1:], "the row naming the lost part and every later one stay as stored"
    repo, older, _current = _trees(tmp_path)
    store_switch.record_switch_intent(root, target_sha=older, target_ref="older", reason="test", door="ui_rollback")
    outcome = store_switch.run_switch_seam(root, repo_dir=repo, owner=_checkout(repo))
    assert outcome["status"] == "failed" and "StoreLost" in outcome["error"], outcome
    assert store_switch.read_switch_intent(root) is not None and _git(repo, "rev-parse", "HEAD") != older
    assert _bodies(root)[1:] == stored[1:]
    # The part found again: the next attempt resumes at the row it stopped at and finishes.
    with ds.transaction(root, write=True) as conn:
        conn.execute("INSERT INTO shared (sha256, value) VALUES (?, ?)", (lost, part))
    assert di.expand_for_older_body(root, batch=1)[str(root)]["chat"] == {"rows": 2}
    assert _bodies(root) == [ds.encode(row) for row in rows]
