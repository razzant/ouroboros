"""A busy recent-wakes read names the store it came from, named where the read is known.

The task and dialogue stores are real SQLite files held by another connection's
exclusive lock; the panel is read through the gateway route. Only the read's wait
is narrowed (the production acquisition budget). No sentence is derived from an
exception message: an unnamed busy read stays unknown, never the task store.
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone
import functools
import json
import sqlite3
import time

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros import consciousness_view as view, dialogue_store, store, task_store, usage_store
from ouroboros.consciousness import BackgroundConsciousness
from ouroboros.gateway.router import collect_routes
from tests._store_shared import seed_result
from tests.test_consciousness_view import FIXTURE, _call

COVERAGE = "wakes that reached a model call in the last 48 h"
FOCUS = {"wake_cause": "focus", "focus": {"place_id": "mail", "conversation_key": "chat:a:r"}}


def busy(name: str) -> str:
    return f"Recent wakes could not be read: {name} store busy."


@contextlib.contextmanager
def exclusive(path):
    """Another connection's exclusive lock: a reader's own wait runs out with SQLITE_BUSY."""
    # Taken inside the request's thread when a read must find it between two others; released here.
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=0, check_same_thread=False)
    try:
        conn.execute("BEGIN EXCLUSIVE")
        yield
    finally:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        conn.close()


@pytest.fixture
def wake(tmp_path, monkeypatch):
    """One focus wake with two released replies in its room, as the real writers store them."""
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")  # an exclusive lock also holds readers
    document = {**json.loads(FIXTURE.read_text(encoding="utf-8")), "metadata": FOCUS, "chat_id": -7}
    seed_result(tmp_path, document["task_id"], document)
    _call(tmp_path, monkeypatch, document["task_id"], at=time.time() - 60)  # the panel selects by the real clock
    reply = {"task_id": document["task_id"], "direction": "out", "chat_id": -7, "source": "presence:mail",
             "text": "answer", "transport": {"delivery": {"state": "authored"}}}
    for stamp in ("2026-10-21T10:00:15Z", "2026-10-21T10:00:20Z"):
        dialogue_store.append(tmp_path, "chat", {**reply, "ts": stamp})
    return document


@pytest.fixture
def client(tmp_path, monkeypatch, wake):
    import ouroboros.config as cfg
    from ouroboros.gateway import ui_i18n
    from supervisor import state

    files = tmp_path / "files"
    files.mkdir()
    monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(files))
    monkeypatch.setattr(cfg, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(ui_i18n, "_LANGUAGE_HOOKS", [])
    monkeypatch.setattr(state, "load_state", lambda: {"bg_consciousness_enabled": True})
    clock = object.__new__(BackgroundConsciousness)
    clock._drive_root = tmp_path
    next_wake = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    monkeypatch.setattr(clock, "status_snapshot", lambda: {"enabled": True, "places_brief": [],
                                                           "next_wake_at": next_wake})
    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path

    def describe():  # the server's own wiring, with the wait a busy read spends narrowed
        with store.acquisition_budget(0.25):
            return view.describe_consciousness(clock)

    app.state.describe_consciousness = describe
    with TestClient(app) as test_client:
        yield test_client


def panel(client):
    response = client.get("/api/consciousness")
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    return response.json()


def test_a_readable_panel_lists_the_wake_with_its_first_reply(client, wake):
    shown = panel(client)
    assert shown["recent_wakes_coverage"] == COVERAGE
    (row,) = shown["recent_wakes"]
    assert row["task_id"] == wake["task_id"] and row["first_reply_at"] == "2026-10-21T10:00:15Z"


@pytest.mark.parametrize("boundary", ["display_documents", "wake_notes"])
def test_both_task_reads_name_the_task_store(client, monkeypatch, tmp_path, boundary):
    reads, held = [], contextlib.ExitStack()
    real = task_store.display_documents

    def documents(*args, **kwargs):
        reads.append("documents")
        found = real(*args, **kwargs)
        held.enter_context(exclusive(store.store_path(tmp_path)))  # the notes read meets the lock
        return found

    if boundary == "wake_notes":
        monkeypatch.setattr(task_store, "display_documents", documents)
    else:
        held.enter_context(exclusive(store.store_path(tmp_path)))
    with held:
        shown = panel(client)
    assert reads == (["documents"] if boundary == "wake_notes" else [])
    assert shown["recent_wakes"] is None and shown["recent_wakes_coverage"] == busy("task")
    assert shown["status"] == "sleeping" and shown["enabled"] is True


@pytest.mark.parametrize("page", [1, 2], ids=["first-read", "advancement"])
def test_the_lazy_dialogue_iterator_names_the_dialogue_store(client, monkeypatch, tmp_path, page):
    real_probe, held, pages = dialogue_store.probe_rows, contextlib.ExitStack(), []
    # One row per page, so the second reply is read by the iterator's own advancement.
    monkeypatch.setattr(dialogue_store, "iter_probes", functools.partial(dialogue_store.iter_probes, batch=1))

    def probe(*args, **kwargs):
        pages.append(kwargs["lower"])
        if len(pages) == page:
            held.enter_context(exclusive(dialogue_store.path(tmp_path)))
        return real_probe(*args, **kwargs)

    monkeypatch.setattr(dialogue_store, "probe_rows", probe)
    if page == 1:  # the iterator's creation reads the stream end before any page
        held.enter_context(exclusive(dialogue_store.path(tmp_path)))
    with held:
        shown = panel(client)
    assert len(pages) == (0 if page == 1 else 2)
    assert shown["recent_wakes"] is None and shown["recent_wakes_coverage"] == busy("dialogue")


@pytest.mark.parametrize("message", ["dialogue store busy: database is locked", "current store busy"])
def test_an_unnamed_busy_read_is_unknown_never_the_task_store(client, monkeypatch, message):
    def unavailable(*_args, **_kwargs):
        raise store.StoreBusy(message, reason="contention")

    monkeypatch.setattr(usage_store, "read", unavailable)
    shown = panel(client)
    assert shown["recent_wakes"] is None and shown["recent_wakes_coverage"] == busy("data")


def test_the_usage_ledger_sentence_is_unchanged(client, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise usage_store.UsageLockUnavailable("busy", reason="contention")

    monkeypatch.setattr(usage_store, "read", unavailable)
    assert panel(client)["recent_wakes_coverage"] == "Recent wakes could not be read: usage ledger busy."


def test_a_busy_panel_recovers_with_the_same_rows(client, tmp_path):
    before = panel(client)
    with exclusive(dialogue_store.path(tmp_path)):
        during = panel(client)
    after = panel(client)
    assert during["recent_wakes"] is None and during["recent_wakes_coverage"] == busy("dialogue")
    assert before["recent_wakes"] == after["recent_wakes"] and after["recent_wakes"]
    assert after["recent_wakes_coverage"] == COVERAGE
    assert {k: v for k, v in during.items() if not k.startswith("recent_wakes")} == {
        k: v for k, v in after.items() if not k.startswith("recent_wakes")}


@pytest.mark.parametrize("name", ["task", "dialogue"])
def test_the_boundary_names_the_raised_exception_itself(monkeypatch, tmp_path, wake, name):
    raised = []
    source = (task_store, "open_store") if name == "task" else (dialogue_store, "transaction")
    real = getattr(*source)

    @contextlib.contextmanager
    def spied(*args, **kwargs):
        try:
            with real(*args, **kwargs) as handle:
                yield handle
        except store.StoreBusy as exc:
            raised.append(exc)
            raise

    monkeypatch.setattr(*source, spied)
    path = store.store_path(tmp_path) if name == "task" else dialogue_store.path(tmp_path)
    with exclusive(path), store.acquisition_budget(0.25), pytest.raises(store.StoreBusy) as caught:
        view.recent_wakes(tmp_path, [], ids=[wake["task_id"]])
    assert raised and caught.value is raised[-1], "the same exception, never a wrapper"
    assert caught.value.store_name == name and caught.value.reason == "contention"
    assert isinstance(caught.value.__cause__, sqlite3.OperationalError)
