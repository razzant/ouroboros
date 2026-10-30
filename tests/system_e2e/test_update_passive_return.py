"""Narrowed R1 on actual servers: a stylesheet-only Update whose successor wrote data and failed before ready.

One scratch Git checkout of this composition (``ouroboros``) and a target that changes one rule of
``web/style.css`` and nothing else. A fresh data root holds what an older install leaves behind: a
legacy chat row, a legacy task result and a settings file with the retired context-mode pair. Every
server is the real ``server.py``, keyless and isolated (own HOME, data root, ports and settings, no
provider credential, no ``CLAUDEXOR*`` variable, an engine home never provisioned, so
``claudexor_daemon.warm_owned_daemon`` starts nothing and discovery finds no engine).

1. The test process prepares and activates the Update with the real owners (``update_prepare``,
   ``body_adoption.authorize_update``/``arm``; only the preparation's selected tests are stubbed).
2. The successor boots: the captured helper switches the stylesheet, the real import probe passes, the
   entry is recorded, and the real boot imports the legacy task into the current store, cuts the chat
   row over into the dialogue store and persists the canonical context-mode pair, all before ready.
   Its bootstrap import test then fails once (``UPDATE_R1_FAULT_ONCE``: a one-shot fault outside the
   composition, through ``sitecustomize`` on this server's PYTHONPATH; no product code is changed and
   the boot, the writers and the readers stay the real ones), so the generation is not ready.
3. ``settle_on_boot`` asks ``body_switch.qualify_return``: stylesheets only, no owned engine home
   before or after, the same runtime and environment, the checkout still exactly the switched
   composition, no Panic: ``compatible``.
   The restart's next entry writes the previous files, index and branch back and hands over.
4. The previous generation boots ready on that data and serves, through its own readers, the task,
   the chat row and the settings its successor wrote.
"""
from __future__ import annotations

import calendar
import json
import os
import pathlib
import subprocess
import sys
import time
import types
import urllib.parse

import pytest

from ouroboros import body_adoption
from ouroboros.platform_layer import subprocess_new_group_kwargs
from ouroboros.task_results import task_result_path
from ouroboros.utils import atomic_write_json
from supervisor import update_prepare
from tests.system_e2e import harness

pytestmark = pytest.mark.serial

REPO = pathlib.Path(__file__).resolve().parents[2]
STOPPED = {"state": "completed", "unconfirmed": []}
EXITED = {"doomed": [1], "dead": [1], "unconfirmed": [], "cleanup_ok": True, "snapshot_ok": True}
TASK_ID, CHAT_TEXT = "passive-return-task", "passive-return chat row, written before the update"
FAULT = '''"""Test fixture: the successor's bootstrap import test fails exactly once (an outside failure).

The interpreter's own ``sitecustomize``, which this file shadows on PYTHONPATH, still runs first."""
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.realpath(os.path.dirname(__file__))
_SHADOWED = importlib.machinery.PathFinder.find_spec(
    "sitecustomize", [p for p in sys.path if os.path.realpath(p or os.curdir) != _HERE])
if _SHADOWED is not None:
    _SHADOWED.loader.exec_module(importlib.util.module_from_spec(_SHADOWED))

_TOKEN = os.environ.get("UPDATE_R1_FAULT_ONCE", "")


class _OnceFailingImportTest(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "supervisor.git_ops_reset":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        original = spec.loader.exec_module

        def exec_module(module):
            original(module)
            real = module.import_test

            def import_test():
                if os.path.exists(_TOKEN):
                    os.unlink(_TOKEN)
                    return {"ok": False, "returncode": 1, "stdout": "", "stderr": "injected outside failure"}
                return real()

            module.import_test = import_test

        spec.loader.exec_module = exec_module
        return spec


if _TOKEN:
    sys.meta_path.insert(0, _OnceFailingImportTest())
'''


def _git(repo, *args, env=None):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=str(repo), check=True,
                          capture_output=True, text=True, env=env).stdout.strip()


def _composition(tmp_path: pathlib.Path) -> tuple:
    """This checkout's composition (tracked and untracked, ignored files left out) as a fresh repository."""
    serving = tmp_path / "Ouroboros" / "repo"
    serving.mkdir(parents=True)
    _git(serving, "init", "-q", "-b", "ouroboros")
    reading = {**os.environ, "GIT_DIR": str(serving / ".git"), "GIT_WORK_TREE": str(REPO)}
    reading.pop("GIT_INDEX_FILE", None)
    _git(serving, "add", "-A", env=reading)  # reads this checkout; writes only the scratch repository
    _git(serving, "commit", "-q", "-m", "the composition under test", env=reading)
    _git(serving, "checkout", "-q", "-f", "ouroboros")
    base = _git(serving, "rev-parse", "HEAD")
    _git(serving, "checkout", "-q", "-b", "official")
    with open(serving / "web" / "style.css", "a", encoding="utf-8") as fh:
        fh.write("\n/* passive-return target */\n.passive-return-probe { outline: 1px solid; }\n")
    _git(serving, "commit", "-q", "-am", "official: one stylesheet rule")
    target = _git(serving, "rev-parse", "HEAD")
    _git(serving, "checkout", "-q", "ouroboros")
    return serving, base, target


def _seed(data: pathlib.Path) -> pathlib.Path:
    """What an older install left: its runtime state, a legacy task result, a legacy chat row and the retired
    context-mode pair (the state document is the state owner's own default, as a written one would be)."""
    from supervisor.state import ensure_state_defaults

    atomic_write_json(data / "state" / "state.json", {**ensure_state_defaults({}), "_schema_version": 1})
    atomic_write_json(task_result_path(data, TASK_ID), {
        "task_id": TASK_ID, "status": "completed", "ts": "2026-10-10T00:00:00+00:00",
        "updated_at": "2026-10-10T00:00:01+00:00", "_schema_version": 1, "result": "done before the update"})
    (data / "logs").mkdir(parents=True, exist_ok=True)
    (data / "logs" / "chat.jsonl").write_text(json.dumps({"ts": "2026-10-10T00:00:02Z", "chat_id": 1,
                                                          "direction": "in", "text": CHAT_TEXT}) + "\n")
    settings = harness.keyless_settings(types.SimpleNamespace(base_url="http://127.0.0.1:9/v1"),
                                        OUROBOROS_CONTEXT_MODE="nano", OUROBOROS_MAX_WORKERS=1)
    settings.pop("OUROBOROS_CONTEXT_MODE_AUTO_LOW", None)  # the boot's settings owner persists the canonical pair
    path = data / "settings.json"
    harness.write_settings_file(path, settings)
    return path


def _server_env(server, fault_dir: pathlib.Path, token: pathlib.Path) -> dict:
    env = {key: value for key, value in server._env().items() if not key.startswith("CLAUDEXOR")}
    env.update(PYTHONPATH=str(fault_dir), UPDATE_R1_FAULT_ONCE=str(token))
    return env


def _closed(data: pathlib.Path) -> list:
    log = data / "logs" / "supervisor.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.startswith("{")] if log.exists() else []
    return [row for row in rows if row.get("type") == "body_adoption_closed"]


def test_an_actual_stylesheet_successor_that_wrote_and_failed_is_returned_and_read(tmp_path, monkeypatch):
    harness.require_lane("update")
    from devtools.benchmarks.common.server_runner import _api, supervisor_state_is_ready
    from supervisor import git_ops

    serving, base, target = _composition(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    settings_path = _seed(data)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "worktrees"))
    monkeypatch.setattr(git_ops, "REPO_DIR", serving)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", data)
    monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros")
    monkeypatch.setattr("ouroboros.update_channels.get_update_channel", lambda: "development")
    monkeypatch.setattr(update_prepare, "resolve_target", lambda **_kw: {
        "channel": "development", "target_ref": "official", "target_sha": target})

    row = update_prepare.prepare(fetch=False, checks=lambda _row: {"ok": True, "stub": "preparation checks"})
    assert row["state"] == "ready", (row.get("reason"), row.get("native"))
    assert row["environment"]["same"] is True and row["native"]["status"] == "none"
    handoff = body_adoption.authorize_update(data, row)
    assert [entry["path"] for entry in handoff["switch"]] == ["web/style.css"]
    assert handoff["qualification"]["web_root"] == "web" and handoff["qualification"]["owned_engine"] == "absent"
    assert body_adoption.bind_update_restart(data)
    assert body_adoption.arm(data, worker_exits=EXITED, live_children=[], owned_stop=STOPPED,
                             owner_restart=True) == "armed"

    stores = ("state/current.sqlite", "state/dialog.sqlite")
    assert not any((data / store).exists() for store in stores)  # only the servers below create them
    fault_dir, token = tmp_path / "fault", tmp_path / "fault-once"
    fault_dir.mkdir()
    (fault_dir / "sitecustomize.py").write_text(FAULT)
    token.write_text("once")
    server = harness.KeylessIsolatedServer(serving, data, settings_path)
    server._patch_settings_ports()
    log = open(tmp_path / "servers.log", "wb")
    server.proc = subprocess.Popen([sys.executable, "server.py"], cwd=str(serving), stdout=log,
                                   stderr=subprocess.STDOUT, env=_server_env(server, fault_dir, token),
                                   **subprocess_new_group_kwargs())
    try:
        deadline, ready = time.monotonic() + 600, False
        while time.monotonic() < deadline:
            assert server.proc.poll() is None, (tmp_path / "servers.log").read_text(errors="replace")[-4000:]
            closed = _closed(data)
            try:
                ready = bool(closed) and supervisor_state_is_ready(_api(server.base_url, "GET", "/api/state",
                                                                        timeout=5))
            except Exception:
                ready = False
            if ready:
                break
            time.sleep(1)
        assert ready, (tmp_path / "servers.log").read_text(errors="replace")[-4000:]

        [closed] = _closed(data)
        assert closed["outcome"] == "returned:return_requested", closed
        phases = [event["phase"] for event in closed["events"]]
        assert "switched" in phases and phases[-1] == "returned", phases
        assert not token.exists()  # the one outside failure happened in the successor's bootstrap
        assert _git(serving, "rev-parse", "HEAD") == base
        assert "passive-return-probe" not in (serving / "web" / "style.css").read_text()

        # The successor wrote these before the return decision: its stores were created then, and the
        # canonical settings pair it persisted is unchanged by the previous generation's boot.
        returned_at = calendar.timegm(time.strptime(closed["events"][-1]["ts"], "%Y-%m-%dT%H:%M:%SZ"))
        for store in stores:
            assert os.stat(data / store).st_birthtime <= returned_at + 1, store
        written = json.loads(settings_path.read_text())
        assert written["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false" and settings_path.stat().st_mtime <= returned_at + 1

        # The previous generation reads them through its own readers.
        task = _api(server.base_url, "GET", f"/api/tasks/{urllib.parse.quote(TASK_ID)}", timeout=10)
        assert task.get("status") == "completed" and "done before the update" in json.dumps(task), task
        history = _api(server.base_url, "GET", "/api/chat/history?chat_id=1&limit=50", timeout=10)
        assert CHAT_TEXT in json.dumps(history), history
        current = _api(server.base_url, "GET", "/api/settings", timeout=10)
        assert current.get("OUROBOROS_CONTEXT_MODE") == "nano", current
    finally:
        server.stop()
        log.close()
