"""The server→worker extension generation channel (W3B-F1).

Task workers load extensions ONCE, at spawn. A skill enabled after that was
invisible to every task the process went on to serve until the pool respawned:
the model calling the fresh surface got "Unknown tool" while ``/api/extensions``
truthfully reported ``live_loaded``. These pins cover the durable carrier
(``state/extension_generation.json``), both natural points a worker notices with
(task start and the dispatch miss), the symmetric disable direction, and the
three properties that keep the mechanism from being worse than the defect:
write-if-changed publication (the publish/adopt cycle terminates), one reload
per DISTINCT published generation (a worker that structurally cannot converge
degrades to the old behaviour instead of reloading before every task), and
fail-closed on a registry it cannot read.
"""

from __future__ import annotations

import os
import pathlib
import types

import pytest

from ouroboros import extension_loader
from ouroboros.extension_companion import init_server_process_pid
from ouroboros.extension_reconcile_queue import (
    adopt_published_extension_generation,
    extension_generation_path,
    publish_extension_generation,
    published_extension_generation,
)
from ouroboros.extension_registry_state import live_extension_fingerprint
from ouroboros.skill_loader import save_enabled
from ouroboros.utils import atomic_write_json

from tests._shared import clean_extension_runtime_state
from tests._extension_loader_shared import _prepare_extension
from tests._extension_loader_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clear_loader_state,
)

_PROBE_PLUGIN = (
    "def register(api):\n"
    "    api.register_tool('echo', lambda ctx, message='hi': f'echo: {message}',\n"
    "                      description='echo probe', schema={})\n"
)
# Imports cleanly in the publisher and raises in the adopter — the shape that
# would let a worker's local failure demote a globally healthy skill.
_BROKEN_PLUGIN = "raise RuntimeError('this extension never imports')\n"


@pytest.fixture(autouse=True)
def _restore_server_pid():
    original = os.environ.get("OUROBOROS_SERVER_PROCESS_PID")
    yield
    init_server_process_pid(int(original) if original else -1)
    if original is None:
        os.environ.pop("OUROBOROS_SERVER_PROCESS_PID", None)


def _as_server() -> None:
    init_server_process_pid(os.getpid())


def _as_worker() -> None:
    init_server_process_pid(999999)


def _published_by_a_server_that_loaded_it(tmp_path: pathlib.Path, name: str = "genprobe"):
    """Drive the SERVER side for real: publish empty, then load and publish again.

    Both generation values are genuine ``publish_extension_generation`` outputs
    of this tree, so a test that later hands one of them to the worker side is
    replaying a real publication rather than a value it invented.
    """
    _as_server()
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path, name, _PROBE_PLUGIN, permissions=["tool"])
    empty_generation = publish_extension_generation(drive_root)
    results = extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root))
    assert results == {name: None}, results  # None == live, no load error
    loaded_generation = published_extension_generation(drive_root)
    assert loaded_generation and loaded_generation != empty_generation
    surface = extension_loader.extension_surface_name(name, "echo")
    assert extension_loader.get_tool(surface) is not None
    return loaded, repo_root, drive_root, surface, empty_generation, loaded_generation


def _settings_and_repo(monkeypatch, repo_root: pathlib.Path) -> None:
    from ouroboros import config

    monkeypatch.setattr(config, "load_settings", lambda: {})
    monkeypatch.setattr(config, "get_skills_repo_path", lambda: str(repo_root))


# ---------------------------------------------------------------------------
# The defect itself, at both natural points.
# ---------------------------------------------------------------------------


def test_worker_with_a_stale_generation_loads_a_skill_enabled_after_its_spawn(
        tmp_path, monkeypatch):
    """THE defect pin: the task-start point, driven through the worker's own seam."""
    from supervisor import worker_process

    _loaded, repo_root, drive_root, surface, _empty, published = (
        _published_by_a_server_that_loaded_it(tmp_path))
    _settings_and_repo(monkeypatch, repo_root)

    # The worker spawned BEFORE the enable: nothing live, the published
    # generation already names a set it has never seen.
    clean_extension_runtime_state()
    _as_worker()
    assert extension_loader.get_tool(surface) is None, (
        "the worker's pre-enable registry is the premise of this pin")

    worker_process._adopt_published_extensions(str(drive_root))

    assert extension_loader.get_tool(surface) is not None, (
        "a skill enabled after the worker spawned is still invisible to its tasks")
    assert live_extension_fingerprint() == published


def test_dispatching_an_unknown_extension_surface_adopts_before_answering(
        tmp_path, monkeypatch):
    """The dispatch-miss point: an enable that lands MID-task is still dispatchable."""
    from ouroboros.tools.extension_dispatch import _extension_dispatch_candidate

    _loaded, repo_root, drive_root, surface, _empty, _published = (
        _published_by_a_server_that_loaded_it(tmp_path))
    _settings_and_repo(monkeypatch, repo_root)
    clean_extension_runtime_state()
    _as_worker()

    ctx = types.SimpleNamespace(
        task_metadata={}, budget_drive_root=str(drive_root), drive_root=str(drive_root))
    descriptor, host_attested_unavailable = _extension_dispatch_candidate(ctx, surface)

    assert descriptor is not None, "the dispatch miss answered Unknown tool without probing"
    assert host_attested_unavailable is False
    assert descriptor.get("skill") == "genprobe"


def test_a_worker_adopts_the_disable_direction_too(tmp_path, monkeypatch):
    """Symmetry: the surface LEAVES a worker that already had it live."""
    loaded, repo_root, drive_root, surface, empty_generation, _published = (
        _published_by_a_server_that_loaded_it(tmp_path))
    _settings_and_repo(monkeypatch, repo_root)

    # This process now holds exactly what a worker spawned before the disable
    # holds. The owner disables, and the server republishes — verbatim the
    # empty-set generation this same publisher minted above.
    _as_worker()
    save_enabled(drive_root, loaded.name, False)
    atomic_write_json(extension_generation_path(drive_root), {
        "schema_version": 1, "generation": empty_generation, "published_at": "2026-01-01T00:00:00Z",
    })

    result = adopt_published_extension_generation(drive_root, lambda: {}, repo_path=str(repo_root))

    assert result["action"] == "reloaded", result
    assert result["converged"] is True, result
    assert extension_loader.get_tool(surface) is None, "disabled surface survived the adopt"


# ---------------------------------------------------------------------------
# The three properties that keep the mechanism cheaper than the defect.
# ---------------------------------------------------------------------------


def test_publish_extension_generation_is_write_if_changed(tmp_path):
    """The property the publish/adopt cycle TERMINATES on.

    An adopting worker's ``reload_all`` sends a reconcile request per skill; the
    server reconciles and announces again. That loop closes only because
    reconciling an already-live extension leaves the live fingerprint alone and
    therefore publishes nothing new — otherwise every adopt would mint a
    generation for the next adopt to chase.
    """
    _loaded, _repo_root, drive_root, _surface, _empty, published = (
        _published_by_a_server_that_loaded_it(tmp_path))
    path = extension_generation_path(drive_root)
    before = path.read_bytes()
    stat_before = path.stat().st_mtime_ns

    assert publish_extension_generation(drive_root) == published
    assert publish_extension_generation(drive_root) == published

    assert path.read_bytes() == before, "an unchanged live set rewrote the marker"
    assert path.stat().st_mtime_ns == stat_before, "the marker was rewritten byte-identically"

    clean_extension_runtime_state()
    assert publish_extension_generation(drive_root) != published
    assert path.read_bytes() != before, "a CHANGED live set must publish"


def test_a_generation_a_worker_cannot_converge_on_costs_exactly_one_reload(
        tmp_path, monkeypatch):
    """Bounded: structural divergence degrades to the pre-fix behaviour.

    A skill that loads in the server and not here can never make the local
    fingerprint equal the published one. Without the per-generation guard the
    worker would spend a full ``reload_all`` before every single task.
    """
    _as_worker()
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path, "brokenprobe", _BROKEN_PLUGIN, permissions=["tool"])
    atomic_write_json(extension_generation_path(drive_root), {
        "schema_version": 1, "generation": "a" * 32, "published_at": "2026-01-01T00:00:00Z",
    })
    calls: list[str] = []
    real_reload = extension_loader.reload_all
    monkeypatch.setattr(
        extension_loader, "reload_all",
        lambda *a, **kw: (calls.append("reload"), real_reload(*a, **kw))[1])

    first = adopt_published_extension_generation(drive_root, lambda: {}, repo_path=str(repo_root))
    second = adopt_published_extension_generation(drive_root, lambda: {}, repo_path=str(repo_root))
    third = adopt_published_extension_generation(drive_root, lambda: {}, repo_path=str(repo_root))

    assert first["action"] == "reloaded" and first["converged"] is False, first
    assert second["action"] == "already_adopted", second
    assert third["action"] == "already_adopted", third
    assert calls == ["reload"], calls
    # ...and the worker's local load failure did NOT demote a globally enabled
    # skill: reload_all never asks for the enable-path revert.
    from ouroboros.skill_loader import load_enabled

    assert load_enabled(drive_root, loaded.name) is True, (
        "a worker-side adopt disabled a skill for the whole install")


def test_a_reload_that_raises_leaves_a_typed_failure_fact_and_still_costs_one_reload(tmp_path, monkeypatch):
    """A reload that RAISES keeps the exactly-one-reload contract (the generation
    stays marked, later calls answer ``already_adopted``) but must not vanish
    silently: one ``extension_generation_adoption_failed`` event row names the
    generation and the error (daemon audit #17 f3: a transient failure poisoned
    the generation with no durable fact anywhere)."""
    import json

    from ouroboros import extension_reconcile_queue as erq

    root = tmp_path / "data"
    (root / "logs").mkdir(parents=True)
    monkeypatch.setattr(erq, "_adopted_generations", {})
    monkeypatch.setattr(erq, "published_extension_generation", lambda _root: "gen-x")
    monkeypatch.setattr("ouroboros.extension_companion.is_server_process", lambda: False)
    monkeypatch.setattr("ouroboros.extension_registry_state.live_extension_fingerprint", lambda: "gen-old")

    def boom(*_a, **_k):
        raise OSError("registry unreadable mid-reload")

    monkeypatch.setattr("ouroboros.extension_loader.reload_all", boom)
    with pytest.raises(OSError):
        erq.adopt_published_extension_generation(root, lambda: {}, repo_path=tmp_path)
    rows = [json.loads(l) for l in (root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    failed = [r for r in rows if r.get("type") == "extension_generation_adoption_failed"]
    assert len(failed) == 1 and failed[0]["published_generation"] == "gen-x" and "OSError" in failed[0]["error"]
    second = erq.adopt_published_extension_generation(root, lambda: {}, repo_path=tmp_path)
    assert second["action"] == "already_adopted", second  # the contract: one reload, however it ended


@pytest.mark.parametrize("marker", ["absent", "unparseable", "empty_generation"])
def test_an_unreadable_published_registry_is_fail_closed(tmp_path, marker):
    """No readable marker is no evidence of divergence — never a blind reload."""
    _as_worker()
    _loaded, repo_root, drive_root = _prepare_extension(
        tmp_path, "failclosed", _PROBE_PLUGIN, permissions=["tool"])
    path = extension_generation_path(drive_root)
    if marker == "unparseable":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
    elif marker == "empty_generation":
        atomic_write_json(path, {"schema_version": 1, "generation": ""})

    result = adopt_published_extension_generation(drive_root, lambda: {}, repo_path=str(repo_root))

    assert result == {"action": "no_published_generation"}, result
    assert extension_loader.snapshot()["extensions"] == []


def test_the_steady_state_probe_reads_one_small_file_and_reloads_nothing(
        tmp_path, monkeypatch):
    """The overhead budget: O(read of a small file), no reload, no IPC, no sleep."""
    _loaded, repo_root, drive_root, _surface, _empty, published = (
        _published_by_a_server_that_loaded_it(tmp_path))
    _as_worker()
    monkeypatch.setattr(extension_loader, "reload_all", lambda *a, **kw: pytest.fail(
        "the in-sync steady state must not reload"))

    for _ in range(5):
        result = adopt_published_extension_generation(
            drive_root, lambda: {}, repo_path=str(repo_root))
        assert result == {"action": "in_sync", "generation": published}, result

    assert extension_generation_path(drive_root).stat().st_size < 512


def test_the_server_publishes_and_never_adopts(tmp_path, monkeypatch):
    """The direction switch is the process role, never the caller's opinion."""
    _loaded, repo_root, drive_root, _surface, _empty, _published = (
        _published_by_a_server_that_loaded_it(tmp_path))
    monkeypatch.setattr(extension_loader, "reload_all", lambda *a, **kw: pytest.fail(
        "the publisher adopted its own publication"))

    assert adopt_published_extension_generation(
        drive_root, lambda: {}, repo_path=str(repo_root)) == {"action": "server_process"}
