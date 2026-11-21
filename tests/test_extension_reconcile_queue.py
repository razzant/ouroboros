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

import pytest

from ouroboros import extension_companion, extension_loader
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
    extension_tool_answer,
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


@pytest.fixture
def queue_supervisor(monkeypatch):
    """A single signal-free process owner for registration, redrive and the health writer."""
    class FakeSupervisor:
        declined = False

        def __init__(self):
            self.runtimes: Dict[str, Dict[str, Any]] = {}
            self.started: list[str] = []
            self.stopped: list[str] = []

        def start(self, descriptor):
            key = f"{descriptor.skill_name}:{descriptor.name}"
            self.started.append(key)
            if self.declined:
                return False
            self.runtimes[key] = {"skill_name": descriptor.skill_name, "name": descriptor.name}
            return True

        def snapshot(self):
            return dict(self.runtimes)

        def companion_ownership(self, skill_name, names):
            return [name for name in names if f"{skill_name}:{name}" not in self.runtimes], {}

        def retained_companions(self, _skill_name, _skill_dir):
            return {}  # this fake always stops synchronously

        def observed_companion_starts(self, _skill_name, _skill_dir, _names):
            return {}, False, False  # this fake has no asynchronous transitions

        def companions_allow_known_good(self, skill_name, names):
            return all(f"{skill_name}:{name}" in self.runtimes for name in names)

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
    monkeypatch.setattr(extension_companion, "_GLOBAL_SUPERVISOR", fake)
    return fake


def test_server_pickup_spawns_stops_and_redrives_missing_companion(tmp_path, monkeypatch, queue_supervisor):
    from ouroboros.extension_health import read_extension_health

    init_server_process_pid()
    loaded, repo_root, drive_root = _prepare_companion_extension(tmp_path)
    fake = queue_supervisor
    request_extension_reconcile(drive_root, loaded.name, reason="test")
    processed = process_extension_reconcile_requests(drive_root, lambda: {}, repo_path=str(repo_root))

    assert processed[0]["skill"] == loaded.name
    assert fake.started == [f"{loaded.name}:daemon"]
    assert list_extension_reconcile_requests(drive_root) == []

    request_extension_reconcile(drive_root, loaded.name, reason="idempotent")
    process_extension_reconcile_requests(drive_root, lambda: {}, repo_path=str(repo_root))
    assert fake.started == [f"{loaded.name}:daemon"]

    fake.runtimes.clear()
    known_good = read_extension_health(drive_root, loaded.name)["last_known_good"]
    assert known_good
    with monkeypatch.context() as unavailable:
        unavailable.setattr(extension_companion, "_GLOBAL_SUPERVISOR", None)
        partial = extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root),
                                                       health_stamp=("test", "unavailable"))
    assert partial["live_loaded"] and partial["companions"]["action"] == "no_supervisor"
    assert read_extension_health(drive_root, loaded.name)["last_known_good"] == known_good
    state = extension_loader.reconcile_extension(
        loaded.name,
        drive_root,
        lambda: {},
        repo_path=str(repo_root),
        health_stamp=("test", "recovered"),
    )
    assert state["action"] == "extension_already_live"
    assert read_extension_health(drive_root, loaded.name)["last_known_good"]["sha"] == "recovered"
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


@pytest.mark.parametrize("recovering", [False, True], ids=["initial_registration", "missing_companion_recovery"])
def test_server_declined_companion_start_is_a_load_error_not_last_known_good(
    tmp_path, monkeypatch, recovering, queue_supervisor,
):
    from ouroboros import extension_health
    from ouroboros.gateway import ws

    init_server_process_pid()
    loaded, repo_root, drive_root = _prepare_companion_extension(tmp_path)
    stamp = ("test", "previous")
    monkeypatch.setattr(extension_health, "fresh_code_stamp", lambda: stamp)
    monkeypatch.setattr(ws, "notify_extension_health_changed", lambda: None)
    fake = queue_supervisor

    def reconcile():
        return extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))

    if recovering:
        assert reconcile()["action"] == "extension_loaded"
    else:
        extension_health.record_extension_health(drive_root, loaded.name, status="live", version=stamp[0], sha=stamp[1])
    known_good = extension_health.read_extension_health(drive_root, loaded.name)["last_known_good"]
    fake.runtimes.clear()
    stamp, fake.declined = ("test", "current"), True
    state = reconcile()
    health = extension_health.read_extension_health(drive_root, loaded.name)
    assert fake.started == [f"{loaded.name}:daemon"] * (2 if recovering else 1)
    assert (state["action"], state["live_loaded"]) == ("extension_load_error", False)
    assert "companion 'daemon' start was declined" in state["load_error"]
    assert loaded.name not in extension_loader._extensions
    assert health["status"] == "broken" and health["last_observed"]["sha"] == "current"
    assert health["last_known_good"] == known_good and known_good["sha"] == "previous"


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


# --- cost law L8: one reconcile per skill per pass ------------------------------------------


def _spy_reconcile(monkeypatch, answer=None):
    calls = []

    def reconcile(skill_name, *_args, **_kwargs):
        calls.append(skill_name)
        if isinstance(answer, BaseException):
            raise answer
        return dict(answer or {"action": "extension_loaded", "reason": "ready"})

    monkeypatch.setattr(extension_loader, "reconcile_extension", reconcile)
    monkeypatch.setattr(extension_loader, "ensure_companions_running", lambda *a, **k: {"action": "noop"})
    return calls


def test_a_backlog_of_requests_is_one_reconcile_per_skill_and_drains_in_one_pass(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """The 01.11 shape: duplicate per-request markers (78 per skill) re-read every pass, one
    reconcile EACH. Now a pass reconciles every named skill once and removes all it read; the
    next pass over an empty queue reconciles nothing."""
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    for n in range(60):
        request_extension_reconcile(drive_root, ("alpha", "beta", "gamma")[n % 3], reason=f"reload-{n}")
    calls = _spy_reconcile(monkeypatch)

    processed = process_extension_reconcile_requests(drive_root, lambda: {})

    assert sorted(calls) == ["alpha", "beta", "gamma"]
    assert sorted((row["skill"], row["requests"], row["marker_removed"]) for row in processed) == [
        ("alpha", 20, True), ("beta", 20, True), ("gamma", 20, True)]
    assert list_extension_reconcile_requests(drive_root) == []
    assert process_extension_reconcile_requests(drive_root, lambda: {}) == [] and len(calls) == 3


def test_a_deferred_skill_keeps_only_its_newest_request_until_the_call_ends(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    for reason in ("first", "second", "third"):
        request_extension_reconcile(drive_root, "voice", reason=reason)
    calls = _spy_reconcile(monkeypatch, {"action": "extension_reload_pending", "reason": "live_call_active"})

    [deferred] = process_extension_reconcile_requests(drive_root, lambda: {})
    assert (deferred["reason"], deferred["marker_removed"], deferred["requests"]) == ("live_call_active", False, 3)
    assert [row["reason"] for row in list_extension_reconcile_requests(drive_root)] == ["third"]

    calls = _spy_reconcile(monkeypatch)
    [done] = process_extension_reconcile_requests(drive_root, lambda: {})
    assert calls == ["voice"] and done["marker_removed"] is True
    assert list_extension_reconcile_requests(drive_root) == []


def test_a_failing_skill_counts_its_attempts_on_one_marker_and_records_the_terminal_failure(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    for _ in range(4):
        request_extension_reconcile(drive_root, "broken", reason="dup")
    calls = _spy_reconcile(monkeypatch, RuntimeError("boom"))

    process_extension_reconcile_requests(drive_root, lambda: {})
    [pending] = list_extension_reconcile_requests(drive_root)
    assert pending["attempts"] == 1 and "boom" in pending["last_error"]
    for _ in range(MAX_ATTEMPTS - 1):
        process_extension_reconcile_requests(drive_root, lambda: {})
    assert calls == ["broken"] * MAX_ATTEMPTS, "one attempt per pass, never one per duplicate"
    assert list_extension_reconcile_requests(drive_root) == []
    [failed] = list((drive_root / "state" / "extension_reconcile" / "failed").glob("*.json"))
    assert json.loads(failed.read_text(encoding="utf-8"))["status"] == "failed"
    events = [json.loads(line) for line in (drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(row["type"], row["skill"], row["attempts"]) for row in events] == [
        ("extension_reconcile_failed", "broken", MAX_ATTEMPTS)]


def test_a_companion_recovery_s_recorded_failure_is_still_a_failed_attempt(
    tmp_path: pathlib.Path, monkeypatch,
) -> None:
    """A reconcile answers a companion recovery that disposed its publication with the load error that recovery
    recorded, where it used to raise it: the pass still counts the attempt on the newest marker, with those words.
    An ordinary load error is a reconcile that ran, and removes its markers as before."""
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    for skill in ("recovering", "recovering", "plain"):
        request_extension_reconcile(drive_root, skill, reason="dup")
    words = "skill 'recovering' companion recovery failure: OSError: spawn refused"
    answers = {"recovering": {"action": "extension_load_error", "reason": "load_error", "load_error": words,
                              "companions": {"action": "recovery_failed"}},
               "plain": {"action": "extension_load_error", "reason": "load_error", "load_error": "import failed"}}
    ensured = []
    monkeypatch.setattr(extension_loader, "reconcile_extension", lambda skill, *_a, **_k: dict(answers[skill]))
    monkeypatch.setattr(extension_loader, "ensure_companions_running",
                        lambda skill, *_a, **_k: ensured.append(skill) or {"action": "not_live"})

    processed = process_extension_reconcile_requests(drive_root, lambda: {})
    [pending] = list_extension_reconcile_requests(drive_root)
    assert (pending["skill"], pending["attempts"], pending["last_error"]) == ("recovering", 1, words)
    assert ([(row["skill"], row["marker_removed"]) for row in processed], ensured) == ([("plain", True)], ["plain"])


_DRIVER_PLUGIN = (
    "import os\n"
    "from playwright._impl._driver import compute_driver_executable\n\n"
    "def register(api):\n"
    "    node, _cli = compute_driver_executable()  # the SDK's own resolution, as a real in-process launch does\n"
    "    if not os.path.isfile(node):\n"
    "        raise RuntimeError('Playwright driver Node missing: ' + node)\n"
    "    api.register_tool('driver', lambda ctx: node, description='driver node', schema={})\n"
)


def test_an_in_process_extension_failing_only_while_node_prepares_recovers_once_it_is_ready(tmp_path, monkeypatch):
    """Astra R2: the boot prepared Node but never selected it into the server's environment, so an in-process
    extension's Playwright ran the SDK's own Node. The step now selects locally before extensions load; when the pin's
    Node still has to be downloaded, such an extension fails on the planned path (never the SDK's Node) and recovers
    through the server's reconcile queue once the boot's preparation publishes exactly that path."""
    import os

    from playwright._impl._driver import compute_driver_executable

    from ouroboros import claudexor_runtime as runtime, node_runtime, server_boot
    from ouroboros.extension_registry_state import _load_failures
    from tests.test_claudexor_node_generations import _fixture

    original = os.environ.get("OUROBOROS_SERVER_PROCESS_PID")
    init_server_process_pid(os.getpid())
    try:
        for key in (node_runtime.PLAYWRIGHT_NODE_ENV, node_runtime.SELECTED_DRIVER_NODE_ENV):
            monkeypatch.delenv(key, raising=False)
        pin, fetches = _fixture(tmp_path, monkeypatch)
        manager = runtime.ClaudexorRuntimeManager(pin)
        monkeypatch.setattr(runtime, "get_runtime_manager", lambda: manager)
        sdk_own = compute_driver_executable()[0]
        assert sdk_own.endswith(("driver/node", "driver\\node.exe"))  # what an unselected server would launch
        fixture_fetch = runtime.fetch_node_archive
        monkeypatch.setattr(runtime, "fetch_node_archive", lambda *_a: (_ for _ in ()).throw(
            runtime.ClaudexorRuntimeError("runtime_node_download_failed", "offline")))
        loaded, repo_root, drive_root = _prepare_extension(tmp_path, "pwdriver", _DRIVER_PLUGIN, permissions=["tool"])
        # Still loading when the preparation reads (its failure would be recorded only after): enabled, not live.
        late = _prepare_extension(tmp_path, "pwlate", _DRIVER_PLUGIN, permissions=["tool"])[0]
        save_enabled(drive_root, _prepare_extension(tmp_path, "pwoff", _DRIVER_PLUGIN, permissions=["tool"])[0].name,
                     False)  # an owner's OFF is never asked again
        monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))  # the server's own skills checkout

        assert server_boot.prepare_pinned_node(drive_root) is None  # local only under pytest
        planned = runtime.ClaudexorRuntimeManager.planned_node(pin)
        assert compute_driver_executable()[0] == planned != sdk_own and not os.path.exists(planned)
        cold = extension_loader.reconcile_extension(loaded.name, drive_root, lambda: {}, repo_path=str(repo_root))
        assert cold["reason"] == "load_error" and planned in cold["load_error"] and loaded.name in _load_failures
        assert server_boot.finish_pinned_node(drive_root) == ""  # offline: logged, nothing published or asked
        assert list_extension_reconcile_requests(drive_root) == []

        monkeypatch.setattr(runtime, "fetch_node_archive", fixture_fetch)
        assert server_boot.finish_pinned_node(drive_root) == planned  # the path the extension was given
        assert sorted(row["skill"] for row in list_extension_reconcile_requests(drive_root)) == [loaded.name, late.name]
        processed = process_extension_reconcile_requests(drive_root, lambda: {}, repo_path=str(repo_root))
        assert [row["action"] for row in processed] == ["extension_loaded"] * 2 and loaded.name not in _load_failures
        assert extension_tool_answer(drive_root, loaded.name, "driver") == planned
        assert fetches and not runtime.managed_runtime_dir(pin).exists()  # Node only, never the engine
    finally:
        init_server_process_pid(int(original) if original else -1)
        if original is None:
            os.environ.pop("OUROBOROS_SERVER_PROCESS_PID", None)


def test_an_acting_consumer_recovers_the_pinned_node_after_a_failed_boot_preparation(tmp_path, monkeypatch):
    """R5 (Opus, Fable, Astra): after a failed boot preparation only a browser call or the next boot tried again, so
    skills, companions and the attach role waited until a restart. Now an acting consumer that finds the pin
    unprepared asks once (``server_boot.request_pinned_node``, single-flight, no timer): the network back, that one
    request prepares Node only and asks the reconcile queue again for every enabled extension that is not live; a
    disabled one is never asked and the queue's pass leaves it off. A lookup, an availability or list read, or an
    environment never asks."""
    import os
    import threading

    from ouroboros import claudexor_runtime as runtime
    from ouroboros import node_runtime, server_boot
    from ouroboros.tools import skill_exec
    from tests.test_claudexor_node_generations import _fixture

    original = os.environ.get("OUROBOROS_SERVER_PROCESS_PID")
    init_server_process_pid(os.getpid())
    try:
        for key in (node_runtime.PLAYWRIGHT_NODE_ENV, node_runtime.SELECTED_DRIVER_NODE_ENV,
                    node_runtime.CARRIED_NODE_ENV):
            monkeypatch.delenv(key, raising=False)
        pin, fetches = _fixture(tmp_path, monkeypatch)
        manager = runtime.ClaudexorRuntimeManager(pin)
        monkeypatch.setattr(runtime, "get_runtime_manager", lambda: manager)
        monkeypatch.setattr("ouroboros.platform_layer.bundled_resource_bases", lambda: [])  # no shipped copy here
        fixture_fetch = runtime.fetch_node_archive
        monkeypatch.setattr(runtime, "fetch_node_archive", lambda *_a: (_ for _ in ()).throw(
            runtime.ClaudexorRuntimeError("runtime_node_download_failed", "offline")))
        loaded, repo_root, drive_root = _prepare_extension(tmp_path, "pwdriver", _DRIVER_PLUGIN, permissions=["tool"])
        off = _prepare_extension(tmp_path, "pwoff", _DRIVER_PLUGIN, permissions=["tool"])[0]
        save_enabled(drive_root, off.name, False)
        monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))
        planned = runtime.ClaudexorRuntimeManager.planned_node(pin)
        assert server_boot.finish_pinned_node(drive_root) == ""  # the boot's preparation failed offline
        assert fetches == [] and not os.path.exists(planned)

        # Passive work never asks: lookups, child environments, the selector itself.
        asked = []
        real_request = server_boot.request_pinned_node
        monkeypatch.setattr(server_boot, "request_pinned_node", lambda root=None: asked.append(root))
        node_runtime.forward_playwright_node({})
        assert node_runtime.discovered_host_node() is None and node_runtime.pinned_node_wait() == (True, planned)
        selected, reason = node_runtime.select_skill_node_runtime()
        assert selected == "" and "pinned:" in reason and asked == []
        assert skill_exec._resolve_runtime_binary("node") == (None, reason) and asked == []  # a lookup never asks
        # R7 (Opus): that lookup is also the availability every context build and list_skills reads; a passive read of
        # an enabled, reviewed node script skill asked for a preparation each time. Only its permitted run asks.
        from ouroboros.skill_loader import discover_skills, summarize_skills
        from ouroboros.tools.registry import ToolContext
        from tests.test_skill_exec import _build_skill, _mark_reviewed_and_enabled, _valid_script_manifest

        script = _build_skill(repo_root, "nodescript", manifest=_valid_script_manifest("nodescript", runtime="node"))
        _mark_reviewed_and_enabled(drive_root, script, "nodescript")
        monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
        listed = next(skill for skill in discover_skills(drive_root) if skill.name == "nodescript")
        assert listed.enabled and not listed.available_for_execution  # waiting, honestly unavailable
        assert "nodescript" in json.dumps(summarize_skills(drive_root)) and asked == []
        (tmp_path / "repo").mkdir()
        ran = skill_exec._handle_skill_exec(ToolContext(repo_dir=tmp_path / "repo", drive_root=drive_root),
                                            skill="nodescript", script="scripts/hello.py")
        assert "SKILL_EXEC_ERROR" in ran and "pinned:" in ran and asked == [None], ran  # the run asks, once
        monkeypatch.setattr(server_boot, "request_pinned_node", real_request)

        monkeypatch.setattr(runtime, "fetch_node_archive", fixture_fetch)  # the network is back
        monkeypatch.delenv("OUROBOROS_PYTEST_ACTIVE")  # this test runs the real request
        gate = threading.Event()
        finish = server_boot.finish_pinned_node
        outcomes = []
        monkeypatch.setattr(server_boot, "finish_pinned_node", lambda root: gate.wait(30) and outcomes.append(
            (root, finish(root))))
        first = server_boot.request_pinned_node(drive_root)  # a consumer's request names this process's data root
        assert first is not None and server_boot.request_pinned_node(drive_root) is None  # single-flight while it runs
        gate.set()
        first.join(60)
        assert not first.is_alive() and outcomes == [(drive_root, planned)], outcomes
        assert os.path.isfile(planned) and len(fetches) == 1
        assert sorted(row["skill"] for row in list_extension_reconcile_requests(drive_root)) == [loaded.name]
        processed = process_extension_reconcile_requests(drive_root, lambda: {}, repo_path=str(repo_root))
        assert [row["action"] for row in processed] == ["extension_loaded"]
        assert extension_tool_answer(drive_root, loaded.name, "driver") == planned
        monkeypatch.setattr(node_runtime, "_probe_node_version_outcome", lambda path, timeout_sec=10: (
            (pin.node_version, "") if path == planned else ("", "exit:1")))  # the fixture archive's Node "runs"
        assert node_runtime.select_skill_node_runtime() == (planned, "bundled")  # the waiting skill runs now
        assert not runtime.managed_runtime_dir(pin).exists()  # Node only, never the engine
    finally:
        init_server_process_pid(int(original) if original else -1)
        if original is None:
            os.environ.pop("OUROBOROS_SERVER_PROCESS_PID", None)


def test_only_a_skill_exec_call_that_would_run_asks_for_the_pinned_node(tmp_path, monkeypatch):
    """R8 (the nanny's reading of the R7 fix): the run site asked for the pin's Node before its own declared-script,
    argument and grant checks, so an undeclared script, malformed args or a missing grant also started a preparation.
    Those refusals, and the earlier ones (disabled, edited since review), ask nothing; a call that would run asks once
    and keeps the reason it waits. Without a pin, or with an unreadable one, nothing is asked."""
    from ouroboros import node_runtime, platform_layer, server_boot
    from ouroboros.skill_loader import save_enabled
    from ouroboros.tools import skill_exec
    from tests.test_skill_exec import _build_skill, _make_ctx, _mark_reviewed_and_enabled, _valid_script_manifest

    skills_root = tmp_path / "skills"
    ctx = _make_ctx(tmp_path)
    for name, extra in (("nodescript", ""), ("nodegrant", "permissions: [inject_chat]\n"), ("nodestale", "")):
        manifest = _valid_script_manifest(name, runtime="node").replace("scripts:\n", extra + "scripts:\n", 1)
        _mark_reviewed_and_enabled(ctx.drive_root, _build_skill(skills_root, name, manifest=manifest), name)
    (skills_root / "nodestale" / "scripts" / "hello.py").write_text("print('edited after review')\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reason = "pinned:24.22.3:not_prepared (planned /planned/node)"
    monkeypatch.setattr(platform_layer, "select_skill_node_runtime", lambda timeout_sec=10: ("", reason))
    monkeypatch.setattr(node_runtime, "pinned_node_wait", lambda: (True, "/planned/node"))
    asked = []
    monkeypatch.setattr(server_boot, "request_pinned_node", lambda root=None: asked.append(root))

    def run(skill="nodescript", script="scripts/hello.py", **kwargs):
        return skill_exec._handle_skill_exec(ctx, skill=skill, script=script, **kwargs)

    refusals = {
        "undeclared script": (run(script="scripts/other.py"), "is not a declared script"),
        "args as one string": (run(args="alpha"), "not a single string"),
        "a non-scalar arg": (run(args=[{"x": 1}]), "is not allowed"),
        "missing grant": (run(skill="nodegrant"), "SKILL_EXEC_GRANT_REQUIRED"),
        "edited since review": (run(skill="nodestale"), "was edited since the last review"),
    }
    save_enabled(ctx.drive_root, "nodescript", False)
    refusals["disabled"] = (run(), "is disabled")
    save_enabled(ctx.drive_root, "nodescript", True)
    for case, (answer, expected) in refusals.items():
        assert expected in answer and "declared runtime" not in answer, (case, answer)
    assert asked == []  # no refused call asked for the pin's Node

    waiting = run(args=["alpha", 2])  # a call that would run: it waits, naming why, and asks once
    assert waiting == ("⚠️ SKILL_EXEC_ERROR: skill 'nodescript' declared runtime 'node' has no available binary "
                       f"on PATH. ({reason})")
    assert asked == [None]
    for pin in ((False, ""), (True, "")):  # no pin, or no archive for this host; an unreadable pin
        monkeypatch.setattr(node_runtime, "pinned_node_wait", lambda pin=pin: pin)
        assert "declared runtime 'node' has no available binary" in run() and asked == [None]
