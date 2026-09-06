"""The extension reconcile marker queue and companion pickup.

Split out of ``tests/test_extension_loader.py`` when that module was divided by
theme; every moved block is verbatim. Covers the worker-side reconcile writing
server markers for enable and disable, the server pickup spawning, stopping and
redriving companion processes, a newer marker surviving a processing race, the
failed-marker overflow queue, the supervisor's redrive surface and the server
lifespan wiring of the pickup loop.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

from ouroboros import extension_loader
from ouroboros import extension_plugin_api
from ouroboros.extension_companion import CompanionSupervisor, init_server_process_pid
from ouroboros.extension_reconcile_queue import (
    MAX_ATTEMPTS,
    list_extension_reconcile_requests,
    process_extension_reconcile_requests,
    request_extension_reconcile,
)
from ouroboros.skill_loader import save_enabled

from tests._extension_loader_shared import (
    _prepare_extension,
)
from tests._extension_loader_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clear_loader_state,
)


def _prepare_companion_extension(tmp_path: pathlib.Path, name: str = "compskill"):
    return _prepare_extension(
        tmp_path,
        name,
        "def register(api):\n    api.register_companion_process('daemon')\n",
        permissions=["companion_process"],
        extra_frontmatter=(
            "companion_processes:\n"
            "  - name: daemon\n"
            "    runtime: python3\n"
            "    command: [\"python3\", \"scripts/daemon.py\"]\n"
        ),
    )


def test_worker_reconcile_writes_server_marker_for_enable_and_disable(tmp_path: pathlib.Path) -> None:
    init_server_process_pid(999999)
    loaded, repo_root, drive_root = _prepare_companion_extension(tmp_path)

    state = extension_loader.reconcile_extension(
        loaded.name,
        drive_root,
        lambda: {},
        repo_path=str(repo_root),
    )

    assert state["action"] == "extension_loaded"
    requests = list_extension_reconcile_requests(drive_root)
    assert [item["skill"] for item in requests] == [loaded.name]

    save_enabled(drive_root, loaded.name, False)
    state = extension_loader.reconcile_extension(
        loaded.name,
        drive_root,
        lambda: {},
        repo_path=str(repo_root),
    )

    assert state["action"] == "extension_unloaded"
    requests = list_extension_reconcile_requests(drive_root)
    assert {item["skill"] for item in requests} == {loaded.name}
    assert "desired_disabled" in {item["reason"] for item in requests}


def test_server_pickup_spawns_stops_and_redrives_missing_companion(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    init_server_process_pid()
    loaded, repo_root, drive_root = _prepare_companion_extension(tmp_path)

    class FakeSupervisor:
        def __init__(self):
            self.runtimes: Dict[str, Dict[str, Any]] = {}
            self.started: list[str] = []
            self.stopped: list[str] = []

        def start(self, descriptor):
            key = f"{descriptor.skill_name}:{descriptor.name}"
            self.runtimes[key] = {"skill_name": descriptor.skill_name, "name": descriptor.name}
            self.started.append(key)
            return True

        def snapshot(self):
            return dict(self.runtimes)

        def stop(self, skill_name: str, name: str):
            self.stopped.append(f"{skill_name}:{name}")
            self.runtimes.pop(f"{skill_name}:{name}", None)

        def stop_skill(self, skill_name: str):
            self.stopped.append(skill_name)
            self.runtimes = {
                key: value
                for key, value in self.runtimes.items()
                if value.get("skill_name") != skill_name
            }

    fake = FakeSupervisor()
    # The companion supervisor is read by PluginAPIImpl (its owner) at register
    # time and by the loader's ensure_companions_running/unload paths.
    monkeypatch.setattr(extension_plugin_api, "get_global_supervisor", lambda: fake)
    monkeypatch.setattr(extension_loader, "get_global_supervisor", lambda: fake)

    request_extension_reconcile(drive_root, loaded.name, reason="test")
    processed = process_extension_reconcile_requests(drive_root, lambda: {}, repo_path=str(repo_root))

    assert processed[0]["skill"] == loaded.name
    assert fake.started == [f"{loaded.name}:daemon"]
    assert list_extension_reconcile_requests(drive_root) == []

    request_extension_reconcile(drive_root, loaded.name, reason="idempotent")
    process_extension_reconcile_requests(drive_root, lambda: {}, repo_path=str(repo_root))
    assert fake.started == [f"{loaded.name}:daemon"]

    fake.runtimes.clear()
    state = extension_loader.reconcile_extension(
        loaded.name,
        drive_root,
        lambda: {},
        repo_path=str(repo_root),
    )
    assert state["action"] == "extension_already_live"
    assert fake.started == [f"{loaded.name}:daemon", f"{loaded.name}:daemon"]

    fake.runtimes.clear()
    request_extension_reconcile(drive_root, loaded.name, reason="redrive")
    process_extension_reconcile_requests(drive_root, lambda: {}, repo_path=str(repo_root))
    assert fake.started == [
        f"{loaded.name}:daemon",
        f"{loaded.name}:daemon",
        f"{loaded.name}:daemon",
    ]

    save_enabled(drive_root, loaded.name, False)
    request_extension_reconcile(drive_root, loaded.name, reason="disable")
    process_extension_reconcile_requests(drive_root, lambda: {}, repo_path=str(repo_root))

    assert fake.stopped == [f"{loaded.name}:daemon", loaded.name]
    assert fake.snapshot() == {}
    assert list_extension_reconcile_requests(drive_root) == []


def test_pickup_keeps_newer_marker_written_during_processing(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    request_extension_reconcile(drive_root, "race_skill", reason="old")

    def fake_reconcile(skill_name, drive_root_arg, settings_reader, **kwargs):
        request_extension_reconcile(drive_root_arg, skill_name, reason="newer")
        return {"action": "extension_loaded"}

    monkeypatch.setattr(extension_loader, "reconcile_extension", fake_reconcile)
    monkeypatch.setattr(
        extension_loader,
        "ensure_companions_running",
        lambda *args, **kwargs: {"action": "noop"},
    )

    processed = process_extension_reconcile_requests(drive_root, lambda: {})

    assert processed[0]["marker_removed"] is True
    requests = list_extension_reconcile_requests(drive_root)
    assert len(requests) == 1
    assert requests[0]["reason"] == "newer"


def test_repeatedly_failed_marker_moves_out_of_active_queue(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    request_extension_reconcile(drive_root, "broken_skill", reason="test")

    def fake_reconcile(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(extension_loader, "reconcile_extension", fake_reconcile)

    for _ in range(MAX_ATTEMPTS):
        process_extension_reconcile_requests(drive_root, lambda: {})

    assert list_extension_reconcile_requests(drive_root) == []
    failed = list((drive_root / "state" / "extension_reconcile" / "failed").glob("*.json"))
    assert len(failed) == 1
    assert json.loads(failed[0].read_text(encoding="utf-8"))["status"] == "failed"


def test_companion_supervisor_exposes_server_redrive_methods() -> None:
    assert callable(getattr(CompanionSupervisor, "snapshot"))
    assert callable(getattr(CompanionSupervisor, "stop_skill"))


def test_server_lifespan_wires_extension_reconcile_pickup() -> None:
    server_py = pathlib.Path(__file__).resolve().parents[1] / "server.py"
    text = server_py.read_text(encoding="utf-8")

    assert "from ouroboros.extension_reconcile_queue import extension_reconcile_pickup_loop" in text
    assert "extension_reconcile_task = asyncio.create_task" in text
    assert "name=\"extension-reconcile-pickup\"" in text
    assert "extension_reconcile_task.cancel()" in text
    assert "await asyncio.wait_for(extension_reconcile_task, timeout=30)" in text
