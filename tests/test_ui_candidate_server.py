"""Exercise actual server/bootstrap custody and byte identity without a browser install."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
from threading import Barrier, Event, Thread
import time
from types import SimpleNamespace
import urllib.request
import uuid

import pytest

from tests import test_ui_smoke_playwright as ui
from tests.candidate_checkout import SENTINEL_ROUTE, candidate_checkout
from tests._store_shared import read_result

pytestmark = [pytest.mark.serial, pytest.mark.ui_browser]


def _served(running, route: str) -> bytes:
    with urllib.request.urlopen(running["url"] + route, timeout=5) as response:  # noqa: S310
        return response.read()


def _post(url: str, route: str, payload: dict) -> dict:
    request = urllib.request.Request(url + route, data=json.dumps(payload).encode("utf-8"),
                                     method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


@contextmanager
def _live_chat(url: str):
    """A WS client that keeps reading like the UI; an unread client stalls flow control."""
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

    stopped, errors = Event(), []
    with connect(url.replace("http://", "ws://", 1) + "/ws", open_timeout=30, proxy=None) as ws:
        def receive():
            while not stopped.is_set():
                try:
                    ws.recv(timeout=1)
                except TimeoutError:
                    continue
                except ConnectionClosed as exc:
                    if not stopped.is_set():
                        errors.append(exc)
                    return
        reader = Thread(target=receive, name="candidate-server-ws-reader")
        reader.start()
        try:
            yield ws
        finally:
            stopped.set()
            ws.close()
            reader.join(5)
    assert not reader.is_alive() and not errors, errors


def _wait_for(read, detail: str, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (value := read()):
            return value
        time.sleep(0.5)
    raise AssertionError(detail)


def _running_ids(data_dir: Path) -> set:
    try:
        snapshot = json.loads((data_dir / "state" / "queue_snapshot.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return {str(row.get("id") or "") for row in snapshot.get("running") or [] if isinstance(row, dict)}


def _terminal_result(data_dir: Path, task_id: str):
    try:
        row = read_result(data_dir, task_id)
    except (OSError, ValueError):
        return None
    return row if row.get("status") in {"completed", "failed", "cancelled", "rejected_duplicate"} else None


def test_real_server_pool_runs_a_submitted_task_to_completion(tmp_path, ui_candidate_checkout):
    """Readiness is not execution: a worker of the real pool must complete a queued task.

    `POST /api/tasks` admits a root task to the supervisor queue — unlike a chat
    message, which is a direct turn outside the pool. Seeing it in the supervisor's
    running set and then as a durable `completed` row with the model's answer proves
    the served candidate's pool executes: the claim a green `supervisor_ready` once
    made while every worker was dead. The streaming keyless stub replaces the UI
    fixture's non-streaming mock, whose answers end as typed provider failures.
    """
    from tests.system_e2e.harness import ScriptedStubModel, keyless_settings, write_settings_file

    fixture = ui.direct_server_with_data.__wrapped__(tmp_path, ui_candidate_checkout)
    with ScriptedStubModel(final_answer="Pooled worker answer: OK.") as stub:
        try:
            running = next(fixture)
            url, data_dir = running["url"], running["data_dir"]
            # Seeded while no server runs; the next incarnation re-proves its origin.
            running["stop_server"]()
            write_settings_file(data_dir / "settings.json", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
            running["start_server"]()
            with _live_chat(url):  # An attached UI client, as in real use; it only drains frames.
                task_id = str(_post(url, "/api/tasks", {
                    "description": f"Reply with OK. ({uuid.uuid4().hex[:8]})", "memory_mode": "forked",
                    "source": "cli", "metadata": {"source": "cli", "delegation_role": "root"},
                }).get("task_id") or "")
                assert task_id, "task creation returned no id"
                seen_running = False

                def terminal():
                    nonlocal seen_running
                    seen_running = seen_running or task_id in _running_ids(data_dir)
                    return _terminal_result(data_dir, task_id)

                result = _wait_for(terminal, f"pooled task {task_id} never reached a durable terminal row", 180)
            assert seen_running, "the task finished without ever being observed in the worker pool"
            assert result["status"] == "completed", json.dumps(result)[:3000]
            assert result["started_at"] and not result.get("_is_direct_chat")
            assert "Pooled worker answer: OK." in str(result.get("result") or ""), result.get("result")
            # Still the same served candidate after the pool executed.
            assert json.loads(_served(running, "/api/health"))["runtime_version"] == running["candidate_version"]
        finally:
            fixture.close()
    # Proven gone: the case hands the session's copy back to its owner, which deletes it once.
    assert ui_candidate_checkout.unproven == 0 and ui_candidate_checkout.path.exists()


@pytest.mark.parametrize("launcher", ["direct", "settings"])
def test_real_server_uses_disposable_candidate_and_keeps_identity_through_restart(
        tmp_path, monkeypatch, launcher, ui_candidate_checkout):
    fixture = ui.direct_server_with_data.__wrapped__(tmp_path, ui_candidate_checkout)
    try:
        if launcher == "settings":
            from tests.test_settings_restart_browser import settings_server

            request = SimpleNamespace(getfixturevalue=lambda name: next(fixture))
            running = settings_server.__wrapped__(request, tmp_path, monkeypatch)
        else:
            running = next(fixture)
        assert running["repo_dir"] != Path(ui.REPO_ROOT)
        assert running["repo_dir"] == ui_candidate_checkout.path  # the session's one copy, this case's own data
        assert running["data_dir"].is_relative_to(tmp_path)
        assert len(running["candidate_identity"]) == 64
        settings = (running["data_dir"] / "settings.json").read_text()
        assert "ui-smoke-key" in settings
        # The source VERSION never carries the suffix, so a server reporting it
        # imported THIS checkout's ouroboros package, not the source or HEAD.
        source_version = (Path(ui.REPO_ROOT) / "VERSION").read_text(encoding="utf-8").strip()
        assert running["candidate_version"] != source_version
        assert running["candidate_version"].startswith(source_version + "+candidate.")
        assert _served(running, SENTINEL_ROUTE) == running["candidate_sentinel"]
        assert json.loads(_served(running, "/api/health"))["runtime_version"] == running["candidate_version"]
        running["restart_server"]()
        # The restarted incarnation re-proves both origins (start_server asserts
        # them too; this is the same claim read from outside the fixture).
        assert _served(running, SENTINEL_ROUTE) == running["candidate_sentinel"]
        assert json.loads(_served(running, "/api/health"))["runtime_version"] == running["candidate_version"]
        assert os.environ["OUROBOROS_DATA_DIR"] != str(running["data_dir"])
    finally:
        fixture.close()
    assert ui_candidate_checkout.unproven == 0 and ui_candidate_checkout.path.exists()


def test_concurrent_servers_keep_distinct_roots_and_owner_sentinels(tmp_path, monkeypatch):
    owner = tmp_path / "owner"
    for key, name in (("HOME", "home"), ("OUROBOROS_DATA_DIR", "data"),
                      ("OUROBOROS_APP_ROOT", "app"), ("OUROBOROS_REPO_DIR", "repo"),
                      ("OUROBOROS_SUBAGENT_PROJECTS_ROOT", "projects"),
                      ("OUROBOROS_SUBAGENT_WORKTREE_ROOT", "worktrees"),
                      ("OUROBOROS_DELIVERABLES_ROOT", "Deliverables")):
        path = owner / name
        path.mkdir(parents=True)
        (path / "sentinel").write_bytes(b"owner state")
        monkeypatch.setenv(key, str(path))
    before = {path.relative_to(owner): path.read_bytes() for path in owner.rglob("sentinel")}
    barrier = Barrier(2)

    def run(number):
        root = tmp_path / f"run-{number}"
        # Each its own session-style copy, as separate xdist workers have.
        with candidate_checkout(Path(ui.REPO_ROOT), root / "candidate", origin_proof=True) as copy:
            fixture = ui.direct_server_with_data.__wrapped__(root, copy)
            try:
                running = next(fixture)
                assert running["data_dir"].is_relative_to(root)
                assert running["repo_dir"].is_relative_to(root)
                barrier.wait(timeout=90)  # Both owned servers must be alive together.
                assert _served(running, SENTINEL_ROUTE) == running["candidate_sentinel"]
                facts = running["url"], running["candidate_identity"], running["candidate_sentinel"]
            finally:
                fixture.close()
        assert not (root / "candidate").exists()
        return facts

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.map(run, range(2))
    assert first[0] != second[0]
    assert first[1] == second[1], "the same source selection is one candidate identity"
    assert first[2] != second[2], "each concurrent session's server proved its OWN copy"
    assert {path.relative_to(owner): path.read_bytes()
            for path in owner.rglob("*") if path.is_file()} == before
