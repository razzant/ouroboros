"""ABI-9 registration atomicity suite (v7next Ф3.1-B).

The extension registration window is stage->validate->swap->attach: nothing an
extension registers is visible in the process-wide registries until the whole
``register()`` run validated and is published as one snapshot, a refused
registration leaves ZERO residue — no surfaces, no bundle, no event-bus
subscriptions, no companion processes and (the direct regression below) no
supervised asyncio task running outside any bundle's cancellation reach — and
the deferred side effects attach only AFTER the snapshot swap, so a handler is
visible to the bus only for an already-published extension (the mid-publication
race pin below) and a post-swap attach failure disposes through the standard
unload path.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time

import pytest

from ouroboros import extension_loader, extension_plugin_api
from ouroboros.contracts.plugin_api import ExtensionRegistrationError
from ouroboros.extension_registry_state import _PluginAPIConfig

from tests._extension_loader_shared import (
    _prepare_extension,
)
from tests._extension_loader_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clear_loader_state,
)


@pytest.fixture()
def _background_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    # run_forever needs a beat to actually start before is_running() is True.
    for _ in range(100):
        if loop.is_running():
            break
        time.sleep(0.01)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2.0)
    loop.close()


def test_supervised_future_never_leaks_when_unload_wins_the_registration_race(
    tmp_path, monkeypatch, _background_loop
):
    """DIRECT regression for the supervised-future leak (plan-pinned).

    Interleaving: ``register_supervised_task`` passes its permission gate, then
    a concurrent unload marks the skill unloading BEFORE the supervised runner
    would be scheduled. The registration must be refused AND the factory must
    never start: on the pre-fix code the future was created before the
    registration lock re-check, so the refusal leaked a running, uncancellable
    supervised task that no bundle (and therefore no unload) could ever reach.
    """
    skill_name = "leakprobe"
    factory_ran = threading.Event()

    class _Bus:
        pass

    bus = _Bus()
    bus._loop = _background_loop

    def _server_process_and_concurrent_unload() -> bool:
        # Deterministically land the racing unload inside the window between
        # the permission gate and the supervised-runner scheduling.
        with extension_loader._lock:
            extension_loader._unloading.add(skill_name)
        return True

    monkeypatch.setattr(extension_plugin_api, "get_global_event_bus", lambda: bus)
    monkeypatch.setattr(
        extension_plugin_api, "is_server_process", _server_process_and_concurrent_unload
    )

    api = extension_plugin_api.PluginAPIImpl(_PluginAPIConfig(
        skill_name=skill_name,
        permissions=["supervised_task"],
        env_allowlist=[],
        state_dir=tmp_path,
        settings_reader=lambda: {},
    ))
    try:
        refused_at_registration = False
        try:
            api.register_supervised_task("bg", lambda: factory_ran.set())
        except ExtensionRegistrationError:
            refused_at_registration = True
        # Whatever point the implementation consults the seam at, the unload
        # is now in flight; a deferred (staged) registration must refuse at
        # publication instead — and in EVERY case the factory never starts.
        with extension_loader._lock:
            extension_loader._unloading.add(skill_name)
        if not refused_at_registration:
            publish = getattr(api, "_publish_registrations", None)
            assert callable(publish), (
                "registration neither refused nor deferred to an atomic publication"
            )
            with pytest.raises(ExtensionRegistrationError, match="unload has started"):
                publish()
    finally:
        with extension_loader._lock:
            extension_loader._unloading.discard(skill_name)

    # The refusal must not have scheduled the runner: give the loop time to
    # betray a leaked future, then require the factory never started and no
    # bundle exists that could be holding (or failing to hold) it.
    assert not factory_ran.wait(0.5), (
        "supervised task factory ran although registration was refused — "
        "the future leaked outside every bundle's cancellation reach"
    )
    with extension_loader._lock:
        assert skill_name not in extension_loader._extensions


def test_failed_registration_publishes_nothing(tmp_path):
    """A register() that fails mid-way must leave zero global residue."""
    loaded, _repo, drive_root = _prepare_extension(
        tmp_path,
        "atomfail",
        plugin_body=(
            "def register(api):\n"
            "    api.register_tool('good', lambda **kw: 'ok', description='d', schema={})\n"
            "    api.register_ws_handler('evt', lambda **kw: None)\n"
            "    raise ValueError('boom after two staged surfaces')\n"
        ),
        permissions=["tool", "ws_handler"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is not None and "boom" in err
    snap = extension_loader.snapshot()
    assert snap["extensions"] == []
    assert snap["tools"] == []
    assert snap["ws_handlers"] == []
    with extension_loader._lock:
        assert "atomfail" not in extension_loader._extensions


def test_registration_error_disposes_live_event_subscription(tmp_path):
    """An event-bus subscription made during an aborted register() is disposed."""
    from ouroboros.event_bus import get_global_event_bus

    loaded, _repo, drive_root = _prepare_extension(
        tmp_path,
        "atomsub",
        plugin_body=(
            "def register(api):\n"
            "    api.subscribe_event('skill.lifecycle', lambda data: None)\n"
            "    api.register_tool('x' * 99, lambda **kw: 'ok', description='d', schema={})\n"
        ),
        permissions=["tool", "subscribe_event"],
        extra_frontmatter="subscribe_events: [skill.lifecycle]\n",
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is not None
    listing = get_global_event_bus().snapshot()
    assert all(sub.get("skill_name") != "atomsub" for sub in listing.values()), (
        "aborted registration left a live event-bus subscription behind"
    )


def test_surfaces_are_invisible_until_registration_completes(tmp_path):
    """No partial publication: mid-register() the registries show nothing."""
    loaded, _repo, drive_root = _prepare_extension(
        tmp_path,
        "atomvis",
        plugin_body=(
            "import ouroboros.extension_loader as el\n"
            "from ouroboros.extension_surface_names import extension_surface_name\n"
            "seen = {}\n"
            "def register(api):\n"
            "    api.register_tool('t1', lambda **kw: 'ok', description='d', schema={})\n"
            "    seen['mid_register'] = el.get_tool(extension_surface_name('atomvis', 't1'))\n"
        ),
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, _force_in_process=True)
    assert err is None, err
    from ouroboros.extension_import_staging import _module_key
    from ouroboros.extension_surface_names import extension_surface_name

    module = sys.modules[_module_key("atomvis")]
    assert module.seen["mid_register"] is None, (
        "a staged tool was globally visible before the registration snapshot swap"
    )
    published = extension_loader.get_tool(extension_surface_name("atomvis", "t1"))
    assert published is not None


def test_publication_carries_a_fresh_generation_digest(tmp_path):
    """Every publication mints a generation digest, stamped into dispatch
    surfaces (tools/routes/ws) so physical-call provenance can name the exact
    published generation; a reload mints a NEW generation."""
    loaded, _repo, drive_root = _prepare_extension(
        tmp_path,
        "atomgen",
        plugin_body=(
            "def register(api):\n"
            "    api.register_tool('t1', lambda **kw: 'ok', description='d', schema={})\n"
        ),
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, _force_in_process=True)
    assert err is None, err
    from ouroboros.extension_surface_names import extension_surface_name

    with extension_loader._lock:
        first_digest = extension_loader._extensions["atomgen"].generation_digest
    assert first_digest
    entry = extension_loader.get_tool(extension_surface_name("atomgen", "t1"))
    assert entry is not None and entry.get("extension_generation") == first_digest

    extension_loader.unload_extension("atomgen")
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, _force_in_process=True)
    assert err is None, err
    with extension_loader._lock:
        second_digest = extension_loader._extensions["atomgen"].generation_digest
    assert second_digest and second_digest != first_digest


def _load_dispatch_extension(tmp_path, name):
    loaded, _repo, drive_root = _prepare_extension(
        tmp_path,
        name,
        plugin_body=(
            "def register(api):\n"
            "    api.register_tool('t1', lambda **kw: 'ok', description='d', schema={})\n"
        ),
        permissions=["tool"],
    )
    err = extension_loader.load_extension(
        loaded, lambda: {}, drive_root=drive_root, _force_in_process=True
    )
    assert err is None, err
    return loaded, drive_root


def test_dispatch_provenance_carries_the_published_generation_digest(tmp_path, monkeypatch):
    """Ф3.2 seam: a physical call's typed meta names the published generation.

    The dispatch-side READ of the ABI-9 digest: a successful call carries
    ``extension_generation`` equal to the live publication's digest, and after
    a reload (a NEW publication) the next call carries the NEW digest. The
    read never gates dispatch — the model-facing text is unchanged."""
    from ouroboros.extension_surface_names import extension_surface_name
    from ouroboros.tools.extension_dispatch import _dispatch_extension_tool_result
    from ouroboros.tools.tool_context import ToolContext

    loaded, drive_root = _load_dispatch_extension(tmp_path, "provgen")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda *_a, **_k: True)
    name = extension_surface_name("provgen", "t1")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=drive_root, task_id="prov-task")

    ext_tool = extension_loader.get_tool(name)
    result = _dispatch_extension_tool_result(ctx, name, ext_tool, {})
    first_digest = extension_loader.extension_generation_digest("provgen")
    assert result.status == "ok" and result.text == "ok"
    assert first_digest
    assert result.meta.get("extension_generation") == first_digest

    extension_loader.unload_extension("provgen")
    err = extension_loader.load_extension(
        loaded, lambda: {}, drive_root=drive_root, _force_in_process=True
    )
    assert err is None, err
    ext_tool = extension_loader.get_tool(name)
    result = _dispatch_extension_tool_result(ctx, name, ext_tool, {})
    second_digest = extension_loader.extension_generation_digest("provgen")
    assert second_digest and second_digest != first_digest
    assert result.status == "ok" and result.text == "ok"
    assert result.meta.get("extension_generation") == second_digest


def test_dispatch_provenance_falls_back_to_the_registry_reader(tmp_path, monkeypatch):
    """A descriptor predating the per-surface stamp still names the live
    generation through the ``extension_generation_digest`` registry reader."""
    from ouroboros.extension_surface_names import extension_surface_name
    from ouroboros.tools.extension_dispatch import _dispatch_extension_tool_result
    from ouroboros.tools.tool_context import ToolContext

    _loaded, drive_root = _load_dispatch_extension(tmp_path, "provfall")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda *_a, **_k: True)
    name = extension_surface_name("provfall", "t1")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=drive_root, task_id="prov-task")

    ext_tool = extension_loader.get_tool(name)
    ext_tool.pop("extension_generation", None)
    result = _dispatch_extension_tool_result(ctx, name, ext_tool, {})
    assert result.status == "ok" and result.text == "ok"
    assert result.meta.get("extension_generation") == (
        extension_loader.extension_generation_digest("provfall")
    )


def test_unavailable_refusal_keeps_the_pre_seam_typed_shape(tmp_path, monkeypatch):
    """The typed EXTENSION_UNAVAILABLE refusal is NOT part of the Ф3.2 read:
    a not-live registration refuses with the exact pre-seam status/code/meta
    (no digest key), so the existing unavailable path breaks in no new way."""
    from ouroboros.extension_surface_names import extension_surface_name
    from ouroboros.tools.extension_dispatch import _dispatch_extension_tool_result
    from ouroboros.tools.tool_context import ToolContext

    _loaded, drive_root = _load_dispatch_extension(tmp_path, "provdead")
    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda *_a, **_k: False)
    name = extension_surface_name("provdead", "t1")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=drive_root, task_id="prov-task")

    ext_tool = extension_loader.get_tool(name)
    result = _dispatch_extension_tool_result(ctx, name, ext_tool, {})
    assert result.status == "unavailable"
    assert result.code == "EXTENSION_UNAVAILABLE"
    assert dict(result.meta) == {"dynamic_provider": True}


def _publish_oop(loaded, drive_root, catalog, *, current_hash=None, expected_generation=None):
    return extension_loader._publish_out_of_process_registration(
        loaded,
        catalog=catalog,
        drive_root=drive_root,
        state_dir=drive_root / "state",
        settings_reader=lambda: {},
        granted_keys=[],
        dependency_site_dirs_enabled=False,
        current_hash=current_hash,
        expected_generation=expected_generation,
    )


def test_out_of_process_catalog_publication_is_atomic(tmp_path):
    """The child-catalog install path is the same stage->validate->swap: a
    catalog with a conflicting surface publishes NOTHING, not a prefix."""
    loaded, _repo, drive_root = _prepare_extension(
        tmp_path,
        "atomoop",
        plugin_body="def register(api):\n    pass\n",
        permissions=["tool", "ws_handler"],
    )
    from ouroboros.extension_surface_names import extension_surface_name

    good_tool = extension_surface_name("atomoop", "t1")
    catalog = {
        "tools": [
            {"name": good_tool, "description": "d", "schema": {}, "timeout_sec": 5},
        ],
        "ws_handlers": [
            {"type": "not-namespaced-for-this-skill"},
        ],
    }
    with pytest.raises(ExtensionRegistrationError):
        _publish_oop(loaded, drive_root, catalog, current_hash=loaded.content_hash)
    snap = extension_loader.snapshot()
    assert snap["tools"] == []
    assert snap["ws_handlers"] == []
    assert snap["extensions"] == []


_COMPANION_FRONTMATTER = (
    "companion_processes:\n"
    "  - name: daemon\n"
    "    runtime: python3\n"
    "    command: [\"python3\", \"scripts/daemon.py\"]\n"
)


def _oop_companion_extension(tmp_path, name):
    return _prepare_extension(
        tmp_path,
        name,
        plugin_body="def register(api):\n    pass\n",
        permissions=["tool", "companion_process"],
        extra_frontmatter=_COMPANION_FRONTMATTER,
    )


def test_out_of_process_surfaces_and_companions_publish_as_one_transaction(
    tmp_path, monkeypatch
):
    """Ф3.1 fix-round-3 pin (ABI-9а): the OOP load path is ONE publication —
    surfaces and companion spawns stage on the same snapshot, so a companion
    that fails to start post-swap leaves NO published surface behind (before
    the fix, surfaces were published in a first transaction and the companion
    failure left a partially published extension for the caller to reap)."""
    from ouroboros.extension_surface_names import extension_surface_name

    loaded, _repo, drive_root = _oop_companion_extension(tmp_path, "oneshot")

    class ExplodingSupervisor:
        def start(self, descriptor):
            raise RuntimeError("companion refused post-swap")

        def stop(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        extension_plugin_api, "get_global_supervisor", lambda: ExplodingSupervisor()
    )
    monkeypatch.setattr(
        extension_loader, "get_global_supervisor", lambda: ExplodingSupervisor()
    )
    monkeypatch.setattr(extension_plugin_api, "is_server_process", lambda: True)

    catalog = {
        "tools": [{
            "name": extension_surface_name("oneshot", "t1"),
            "description": "d", "schema": {}, "timeout_sec": 5,
        }],
        "companions": ["daemon"],
    }
    with pytest.raises(RuntimeError, match="companion refused post-swap"):
        _publish_oop(loaded, drive_root, catalog, current_hash=loaded.content_hash)
    snap = extension_loader.snapshot()
    assert snap["tools"] == []
    assert snap["extensions"] == []
    with extension_loader._lock:
        assert "oneshot" not in extension_loader._extensions


def test_companion_recovery_failure_unloads_instead_of_silent_abort(
    tmp_path, monkeypatch
):
    """Ф3.1 fix-round-3 pin (ABI-9б): the server-side companion RECOVERY
    publication routes a failure into the standard dispose+unload path — the
    extension does not stay half-alive with published surfaces and a
    companion it could not start. (Fix-round-4: the disposal is
    generation-bound — it reaps exactly the publication this recovery
    swapped in, never a newer one.)"""
    from ouroboros.extension_surface_names import extension_surface_name

    loaded, _repo, drive_root = _oop_companion_extension(tmp_path, "recofail")
    tool_name = extension_surface_name("recofail", "t1")
    _publish_oop(
        loaded, drive_root,
        {"tools": [{"name": tool_name, "description": "d", "schema": {}, "timeout_sec": 5}]},
        current_hash=loaded.content_hash,
    )
    assert extension_loader.get_tool(tool_name) is not None

    class ExplodingSupervisor:
        def start(self, descriptor):
            raise RuntimeError("recovery spawn refused")

        def stop(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        extension_plugin_api, "get_global_supervisor", lambda: ExplodingSupervisor()
    )
    monkeypatch.setattr(
        extension_loader, "get_global_supervisor", lambda: ExplodingSupervisor()
    )
    monkeypatch.setattr(extension_plugin_api, "is_server_process", lambda: True)

    with pytest.raises(RuntimeError, match="recovery spawn refused"):
        _publish_oop(
            loaded, drive_root, {"companions": ["daemon"]},
            expected_generation=extension_loader.extension_generation_digest("recofail"),
        )
    assert extension_loader.get_tool(tool_name) is None, (
        "a failed recovery publication must unload the extension, not leave "
        "its surfaces half-alive"
    )
    with extension_loader._lock:
        assert "recofail" not in extension_loader._extensions


def test_late_publication_restamps_already_published_descriptors(
    tmp_path, monkeypatch
):
    """Ф3.1 fix-round-3 pin (staged protocol): a bundle publishing MORE than
    once (companion recovery onto live surfaces) mints ONE digest per
    publication and re-stamps every already-published descriptor — the
    per-surface provenance stamp never diverges from
    ``bundle.generation_digest``."""
    from ouroboros.extension_surface_names import extension_surface_name

    loaded, _repo, drive_root = _oop_companion_extension(tmp_path, "restamp")
    tool_name = extension_surface_name("restamp", "t1")
    _publish_oop(
        loaded, drive_root,
        {"tools": [{"name": tool_name, "description": "d", "schema": {}, "timeout_sec": 5}]},
        current_hash=loaded.content_hash,
    )
    with extension_loader._lock:
        first_digest = extension_loader._extensions["restamp"].generation_digest
    assert extension_loader.get_tool(tool_name)["extension_generation"] == first_digest

    class OkSupervisor:
        def start(self, descriptor):
            return True

        def stop(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        extension_plugin_api, "get_global_supervisor", lambda: OkSupervisor()
    )
    monkeypatch.setattr(
        extension_loader, "get_global_supervisor", lambda: OkSupervisor()
    )
    monkeypatch.setattr(extension_plugin_api, "is_server_process", lambda: True)

    _publish_oop(
        loaded, drive_root, {"companions": ["daemon"]},
        expected_generation=first_digest,
    )
    with extension_loader._lock:
        second_digest = extension_loader._extensions["restamp"].generation_digest
    assert second_digest and second_digest != first_digest
    assert extension_loader.get_tool(tool_name)["extension_generation"] == second_digest, (
        "the late publication left an already-published descriptor on the "
        "previous generation digest"
    )


def test_unload_closes_bus_and_runtime_visibility_before_surfaces_leave(tmp_path, monkeypatch):
    """Ф3.1 fix-round-3 pin (ABI-9в): unload closes the extension's INPUTS
    first — at the moment of the bus unsubscribe the bundle and its surfaces
    are still published; only then do they leave the registries."""
    from ouroboros.event_bus import EventBus

    skill_name = "unloadvis"
    real_bus = EventBus()
    monkeypatch.setattr(extension_plugin_api, "get_global_event_bus", lambda: real_bus)

    api = extension_plugin_api.PluginAPIImpl(_PluginAPIConfig(
        skill_name=skill_name,
        permissions=["tool", "subscribe_event"],
        env_allowlist=[],
        state_dir=tmp_path,
        settings_reader=lambda: {},
        subscribe_events=["skill.lifecycle"],
    ))
    api.register_tool("t1", lambda **kw: "ok", description="d", schema={})
    api.subscribe_event("skill.lifecycle", lambda data: None)
    api._publish_registrations()
    with extension_loader._lock:
        tool_key = list(extension_loader._extensions[skill_name].tools)[0]

    observed = {}

    class ProbeBus:
        def unsubscribe(self, sub_id):
            with extension_loader._lock:
                observed["tool_still_published"] = tool_key in extension_loader._tools
                observed["bundle_still_published"] = (
                    skill_name in extension_loader._extensions
                )
            real_bus.unsubscribe(sub_id)

    monkeypatch.setattr(extension_loader, "get_global_event_bus", lambda: ProbeBus())
    extension_loader.unload_extension(skill_name)

    assert observed == {
        "tool_still_published": True, "bundle_still_published": True,
    }, "the unsubscribe must run BEFORE the bundle/surfaces leave the registries"
    with extension_loader._lock:
        assert skill_name not in extension_loader._extensions
        assert tool_key not in extension_loader._tools
    assert real_bus.snapshot() == {}


def test_publish_started_after_unload_never_delivers(tmp_path, monkeypatch):
    """Ф3.1 fix-round-3 pin (the residual's supported half): after unload's
    unsubscribe, a NEW ``EventBus.publish`` finds no subscription and never
    invokes the handler. (The unsupported half — a publish that COPIED the
    handler before the unsubscribe — is the disclosed copy-semantics residual
    in ``EventBus.publish``.)"""
    from ouroboros.event_bus import EventBus

    skill_name = "unloadflow"
    real_bus = EventBus()
    monkeypatch.setattr(extension_plugin_api, "get_global_event_bus", lambda: real_bus)
    monkeypatch.setattr(extension_loader, "get_global_event_bus", lambda: real_bus)

    seen: list = []
    api = extension_plugin_api.PluginAPIImpl(_PluginAPIConfig(
        skill_name=skill_name,
        permissions=["subscribe_event"],
        env_allowlist=[],
        state_dir=tmp_path,
        settings_reader=lambda: {},
        subscribe_events=["skill.lifecycle"],
    ))
    api.subscribe_event("skill.lifecycle", lambda data: seen.append(dict(data)))
    api._publish_registrations()
    real_bus.publish("skill.lifecycle", {"probe": "before-unload"})
    assert [row.get("probe") for row in seen] == ["before-unload"]

    extension_loader.unload_extension(skill_name)
    real_bus.publish("skill.lifecycle", {"probe": "after-unload"})
    assert [row.get("probe") for row in seen] == ["before-unload"], (
        "a publish STARTED after the unload's unsubscribe must not deliver"
    )


def test_conflict_refused_publication_has_zero_external_effects(
    tmp_path, monkeypatch, _background_loop
):
    """Ф3.1 fix-round pin (validate -> swap -> attach): a conflict that arises
    AFTER staging but BEFORE publication refuses the publication WITHOUT any
    externally visible effect — the supervised factory never starts and the
    event bus is never touched, because the definitive validation runs
    before the swap and before any deferred side effect attaches."""
    skill_name = "conflprobe"
    factory_ran = threading.Event()
    bus_calls: list = []

    class _Bus:
        def subscribe(self, *args, **kwargs):
            bus_calls.append(("subscribe", args, kwargs))
            return "sub-should-never-exist"

        def unsubscribe(self, sub_id):
            bus_calls.append(("unsubscribe", sub_id))

    bus = _Bus()
    bus._loop = _background_loop
    monkeypatch.setattr(extension_plugin_api, "get_global_event_bus", lambda: bus)
    monkeypatch.setattr(extension_plugin_api, "is_server_process", lambda: True)

    def _api() -> extension_plugin_api.PluginAPIImpl:
        return extension_plugin_api.PluginAPIImpl(_PluginAPIConfig(
            skill_name=skill_name,
            permissions=["tool", "supervised_task", "subscribe_event"],
            env_allowlist=[],
            state_dir=tmp_path,
            settings_reader=lambda: {},
            subscribe_events=["skill.lifecycle"],
        ))

    loser = _api()
    loser.register_tool("t1", lambda **kw: "ok", description="d", schema={})
    loser.register_supervised_task("bg", lambda: factory_ran.set())
    loser.subscribe_event("skill.lifecycle", lambda data: None)

    winner = _api()
    winner.register_tool("t1", lambda **kw: "ok", description="d", schema={})
    winner._publish_registrations()  # the surface goes live between stage and publish

    with pytest.raises(ExtensionRegistrationError, match="publication refused"):
        loser._publish_registrations()

    assert bus_calls == [], "a refused publication touched the event bus"
    assert not factory_ran.wait(0.5), (
        "supervised task factory ran although the conflicting publication was refused"
    )
    with extension_loader._lock:
        bundle = extension_loader._extensions[skill_name]
        assert bundle.supervised_futures == []
        assert bundle.event_subscriptions == []


def test_event_published_before_publication_never_invokes_the_handler(tmp_path):
    """Ф3.1 fix-round pin (pre-publication invisibility, not eventual cleanup):
    subscriptions are STAGED — an event published before the snapshot swap
    must not invoke the handler and must not appear on the bus; the sub_id
    returned by subscribe_event is the id the bus attaches at publication."""
    from ouroboros.event_bus import get_global_event_bus

    seen: list = []
    api = extension_plugin_api.PluginAPIImpl(_PluginAPIConfig(
        skill_name="stagesub",
        permissions=["subscribe_event"],
        env_allowlist=[],
        state_dir=tmp_path,
        settings_reader=lambda: {},
        subscribe_events=["skill.lifecycle"],
    ))
    sub_id = api.subscribe_event("skill.lifecycle", lambda data: seen.append(dict(data)))
    assert sub_id

    bus = get_global_event_bus()
    bus.publish("skill.lifecycle", {"probe": "early"})
    assert seen == [], "an event published before publication invoked a staged handler"
    assert all(
        sub.get("skill_name") != "stagesub" for sub in bus.snapshot().values()
    ), "a staged subscription was visible on the bus before publication"

    api._publish_registrations()
    assert sub_id in bus.snapshot(), "publication must attach the pre-minted sub_id"
    bus.publish("skill.lifecycle", {"probe": "late"})
    assert [row.get("probe") for row in seen] == ["late"]


def test_concurrent_publish_in_the_validate_to_attach_window_never_invokes_the_handler(
    tmp_path, monkeypatch
):
    """Ф3.1 fix-round-2 pin (a REAL race, not a post-factum check): a
    concurrent ``EventBus.publish()`` landing inside the publication critical
    section — after the definitive validation, before the bus attach — must
    not invoke the staged handler. The ordering that closes the window is
    attach-strictly-AFTER-swap: at the moment the publication thread first
    reaches for the bus, the bundle is already published (generation digest
    minted) while the handler is still invisible to the bus, so the racing
    publish (which takes only the bus's own lock and does NOT block on the
    registry lock) sees no subscription at all."""
    from ouroboros.event_bus import EventBus

    skill_name = "racesub"
    seen: list = []
    during_window: dict = {}
    in_window = threading.Event()
    racing_publish_done = threading.Event()
    real_bus = EventBus()

    def hooked_get_bus():
        # Called by _publish_registrations at the START of the attach phase
        # (registry lock held). Record whether the swap already happened,
        # then freeze this publication thread until the racing publisher
        # has interleaved a publish into the window.
        with extension_loader._lock:
            bundle = extension_loader._extensions.get(skill_name)
            during_window["published"] = bundle is not None and bool(
                bundle.generation_digest
            )
        in_window.set()
        assert racing_publish_done.wait(5.0), "racing publisher never ran"
        during_window["handler_calls"] = len(seen)
        return real_bus

    monkeypatch.setattr(extension_plugin_api, "get_global_event_bus", hooked_get_bus)

    api = extension_plugin_api.PluginAPIImpl(_PluginAPIConfig(
        skill_name=skill_name,
        permissions=["subscribe_event"],
        env_allowlist=[],
        state_dir=tmp_path,
        settings_reader=lambda: {},
        subscribe_events=["skill.lifecycle"],
    ))
    sub_id = api.subscribe_event("skill.lifecycle", lambda data: seen.append(dict(data)))

    def racing_publisher():
        if not in_window.wait(5.0):
            racing_publish_done.set()
            return
        real_bus.publish("skill.lifecycle", {"probe": "mid-publication"})
        racing_publish_done.set()

    publisher = threading.Thread(target=racing_publisher, daemon=True)
    publisher.start()
    try:
        api._publish_registrations()
    finally:
        publisher.join(timeout=5.0)

    assert during_window["published"] is True, (
        "the bus attach ran before the snapshot swap — the ABI-9 order is "
        "validate -> swap -> attach"
    )
    assert during_window["handler_calls"] == 0 and seen == [], (
        "a publish racing the publication window invoked a not-yet-attached "
        "handler"
    )
    assert sub_id in real_bus.snapshot(), "publication must attach the pre-minted sub_id"
    real_bus.publish("skill.lifecycle", {"probe": "post-publication"})
    assert [row.get("probe") for row in seen] == ["post-publication"]


def test_supervised_effect_starts_only_after_the_swap(tmp_path, monkeypatch):
    """Ф3.1 fix-round-2 pin: the supervised-runner side effect attaches
    strictly AFTER the registry swap — at the moment the runner would be
    scheduled, the bundle is already published under a minted generation
    digest."""
    skill_name = "swapfirst"
    observed: list = []

    def probing_is_server_process() -> bool:
        with extension_loader._lock:
            bundle = extension_loader._extensions.get(skill_name)
            observed.append(bundle is not None and bool(bundle.generation_digest))
        return False  # probe only: never schedule a real runner

    monkeypatch.setattr(
        extension_plugin_api, "is_server_process", probing_is_server_process
    )

    api = extension_plugin_api.PluginAPIImpl(_PluginAPIConfig(
        skill_name=skill_name,
        permissions=["supervised_task"],
        env_allowlist=[],
        state_dir=tmp_path,
        settings_reader=lambda: {},
    ))
    api.register_supervised_task("bg", lambda: None)
    api._publish_registrations()
    assert observed == [True], (
        "the supervised effect started before the snapshot swap published "
        "the bundle"
    )


def test_post_swap_attach_failure_disposes_through_the_standard_unload_path(
    tmp_path, monkeypatch
):
    """Ф3.1 fix-round-2 pin: an effect that fails AFTER the swap (the snapshot
    is already published) orphans nothing — the raise routes load_extension
    into the standard dispose+unload path: the load reports the error, the
    registries and the bus end empty, and the extension's own on_unload
    callback ran (the disclosure that the published bundle was disposed, not
    leaked)."""
    from ouroboros.event_bus import get_global_event_bus

    def exploding_start(self, spec):
        raise RuntimeError("supervised runner refused post-swap")

    monkeypatch.setattr(
        extension_plugin_api.PluginAPIImpl, "_start_supervised_task", exploding_start
    )
    marker = tmp_path / "unload_ran.marker"
    loaded, _repo, drive_root = _prepare_extension(
        tmp_path,
        "postswapfail",
        plugin_body=(
            "import pathlib\n"
            f"MARKER = pathlib.Path({str(marker)!r})\n"
            "def register(api):\n"
            "    api.register_tool('t1', lambda **kw: 'ok', description='d', schema={})\n"
            "    api.subscribe_event('skill.lifecycle', lambda data: None)\n"
            "    api.register_supervised_task('bg', lambda: None)\n"
            "    api.on_unload(lambda: MARKER.write_text('disposed', encoding='utf-8'))\n"
        ),
        permissions=["tool", "subscribe_event", "supervised_task"],
        extra_frontmatter="subscribe_events: [skill.lifecycle]\n",
    )
    err = extension_loader.load_extension(
        loaded, lambda: {}, drive_root=drive_root, _force_in_process=True
    )
    assert err is not None and "supervised runner refused post-swap" in err
    snap = extension_loader.snapshot()
    assert snap["extensions"] == []
    assert snap["tools"] == []
    with extension_loader._lock:
        assert "postswapfail" not in extension_loader._extensions
    listing = get_global_event_bus().snapshot()
    assert all(sub.get("skill_name") != "postswapfail" for sub in listing.values()), (
        "the post-swap failure left a live bus subscription behind"
    )
    assert marker.is_file() and marker.read_text(encoding="utf-8") == "disposed", (
        "the standard unload path must run the published bundle's on_unload"
    )


def test_disposers_stay_out_of_the_plugin_api_surface():
    """The disposers list is loader-internal (ABI-9): the PluginAPI contract
    must not grow a disposer/staging method an extension could call."""
    from ouroboros.contracts.plugin_api import PluginAPI

    public = {m for m in dir(PluginAPI) if not m.startswith("_")}
    assert not any("disposer" in name or "staged" in name or "publish" in name for name in public)
    impl_public = {m for m in dir(extension_plugin_api.PluginAPIImpl) if not m.startswith("_")}
    assert not any("disposer" in name or "staged" in name or "publish" in name for name in impl_public)
