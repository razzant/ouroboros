"""Automatic call-skill replacement waits; an explicit disable still cancels work."""
from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
from types import SimpleNamespace

import pytest

from ouroboros import extension_registry_state as registry


@pytest.fixture(params=["realtime_providers", "conversation_features"])
def publication(monkeypatch, request, tmp_path):
    name = "synthetic-call-skill"
    bundle = registry._ExtensionRegistrations(
        content_hash="old", skill_dir=str(tmp_path), generation_digest="generation-one")
    getattr(bundle, request.param).append(f"{name}:call")
    monkeypatch.setitem(registry._extensions, name, bundle)
    return name, bundle, {"skill": name, "extension_generation": bundle.generation_digest}


class Work:
    cancelled = False

    def cancel(self):
        self.cancelled = True


def test_replacement_refusal_lasts_only_while_call_work_is_live(publication):
    name, bundle, spec = publication
    background = Work()
    bundle.supervised_futures.append(background)
    assert registry.realtime_load_refusal(name) == ""
    work = Work()
    with registry.extension_work_scope(spec, work):
        assert registry.realtime_load_refusal(name) == "live_call_active"
        assert work in bundle.supervised_futures and not work.cancelled
    assert registry.realtime_load_refusal(name) == ""
    assert bundle.supervised_futures == [background]


def test_reconcile_retries_replacement_and_disable_ends_call(publication, monkeypatch, tmp_path):
    from ouroboros import extension_loader as loader

    name, bundle, spec = publication
    subject = SimpleNamespace(name=name)
    current = {"skill": name, "enabled": True, "desired_live": True,
               "loaded_present": True, "live_loaded": False, "reason": "ready"}
    monkeypatch.setattr(loader, "runtime_state_for_skill_name", lambda *a, **kw: dict(current))
    monkeypatch.setattr(loader, "_finalize_extension_reconcile", lambda *a, **kw: None)
    loaded = []
    monkeypatch.setattr(loader, "load_extension", lambda *a, **kw: loaded.append(a[0]) or None)
    reconcile = lambda: loader.reconcile_extension(name, tmp_path, lambda: {}, skills=[subject], repo_path="")
    work = Work()
    with registry.extension_work_scope(spec, work):
        refused = reconcile()
        assert refused["reason"] == "live_call_active"
        assert refused["action"] == "extension_reload_pending"
        assert refused["loaded_present"] and not loaded and not work.cancelled
        # An edited payload can lose current review authority; automatic reconcile
        # still leaves the running publication alone until the call ends.
        current.update(desired_live=False, reason="review_stale")
        assert reconcile()["reason"] == "live_call_active"
        current.update(desired_live=True, reason="ready")
    assert reconcile()["action"] == "extension_loaded"
    assert loaded == [subject]

    monkeypatch.setitem(registry._extensions, name, bundle)
    with registry.extension_work_scope(spec, work):
        current.update(enabled=False, desired_live=False, reason="disabled")
        assert reconcile()["action"] == "extension_unloaded"
        assert work.cancelled and name not in registry._extensions
    assert registry.realtime_load_refusal(name) == ""


def test_new_work_cannot_attach_to_a_replaced_publication(publication):
    from ouroboros.contracts.plugin_api import ExtensionRegistrationError

    _name, bundle, spec = publication
    bundle.generation_digest = "generation-two"
    with pytest.raises(ExtensionRegistrationError, match="generation changed"):
        with registry.extension_work_scope(spec, Work()):
            pytest.fail("stale call work was admitted")
    with registry.extension_work_scope({**spec, "extension_generation": "generation-two"}, Work()):
        assert registry.realtime_load_refusal(spec["skill"]) == "live_call_active"


def test_ordinary_work_is_cancellable_but_never_a_call(publication):
    from ouroboros import extension_loader as loader
    from ouroboros.contracts.plugin_api import ExtensionRegistrationError

    name, bundle, spec = publication
    work, call = Work(), Work()
    with registry.extension_work_scope(spec, work, call_work=False):
        assert work in bundle.supervised_futures and bundle.active_call_work == 0
        assert registry.realtime_load_refusal(name) == ""
        # A real call beside it still owns the fence, and only for its own life.
        with registry.extension_work_scope(spec, call):
            assert registry.realtime_load_refusal(name) == "live_call_active"
        assert registry.realtime_load_refusal(name) == "" and call not in bundle.supervised_futures
        loader.unload_extension(name)
        assert work.cancelled and name not in registry._extensions
    assert bundle.supervised_futures == [] and bundle.active_call_work == 0
    with pytest.raises(ExtensionRegistrationError, match="generation changed"):
        with registry.extension_work_scope(spec, Work(), call_work=False):
            pytest.fail("stale ordinary work was admitted")


def test_only_call_admission_waits_for_a_running_load(publication):
    from concurrent.futures import ThreadPoolExecutor

    name, _bundle, spec = publication

    def enter(**kind):
        with registry.extension_work_scope(spec, Work(), **kind):
            return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        # The load holds this for its whole duration; a held loop would stop every request.
        with registry._lifecycle_lock_for(name):
            assert pool.submit(enter, call_work=False).result(timeout=5)
            admission = pool.submit(enter)
            with pytest.raises(FutureTimeout):
                admission.result(timeout=0.1)
        assert admission.result(timeout=5)


def test_an_isolated_route_response_is_ordinary_work(publication):
    from contextlib import nullcontext
    from ouroboros.extension_route_stream import RouteStreamResponse

    name, bundle, spec = publication
    response = RouteStreamResponse(spec, nullcontext)

    async def serve():
        async with response._child_scope():
            assert response in bundle.supervised_futures
            assert registry.realtime_load_refusal(name) == ""

    asyncio.run(serve())
    assert response not in bundle.supervised_futures


def test_direct_load_refuses_only_live_calls(publication, monkeypatch, tmp_path):
    from ouroboros import extension_loader as loader

    name, bundle, spec = publication
    loaded = []
    monkeypatch.setattr(loader, "_load_extension_locked", lambda *a, **kw: loaded.append(a[0]) or None)
    subject = SimpleNamespace(name=name)
    work = Work()
    with registry.extension_work_scope(spec, work):
        assert loader.load_extension(subject, lambda: {}, drive_root=tmp_path) == "live_call_active"
        assert registry._extensions[name] is bundle and not work.cancelled and not loaded
    assert loader.load_extension(subject, lambda: {}, drive_root=tmp_path) is None
    assert loaded == [subject]


def test_reconcile_marker_waits_for_call_then_retries(publication, monkeypatch, tmp_path):
    from ouroboros import extension_loader as loader
    from ouroboros.extension_reconcile_queue import process_extension_reconcile_requests, request_extension_reconcile

    name, _bundle, spec = publication
    companions = []
    monkeypatch.setattr(loader, "reconcile_extension", lambda *a, **kw: {
        "action": "extension_reload_pending" if registry.realtime_load_refusal(name) else "extension_loaded",
        "reason": registry.realtime_load_refusal(name) or "ready"})
    monkeypatch.setattr(loader, "ensure_companions_running", lambda *a, **kw: companions.append(name))
    marker = request_extension_reconcile(tmp_path, name, reason="payload_changed")
    with registry.extension_work_scope(spec, Work()):
        [result] = process_extension_reconcile_requests(tmp_path, lambda: {})
        assert result["reason"] == "live_call_active" and not result["marker_removed"]
        assert marker.exists() and not companions
    [result] = process_extension_reconcile_requests(tmp_path, lambda: {})
    assert result["action"] == "extension_loaded" and result["marker_removed"]
    assert not marker.exists() and companions == [name]


def test_source_unload_and_settlement_refuse_only_live_calls(publication, monkeypatch, tmp_path):
    from ouroboros import skill_source_install as installer

    name, bundle, spec = publication
    settled = []
    monkeypatch.setattr(installer, "settle_pending_update", lambda *a, **kw: settled.append(name) or {"status": "none"})
    work = Work()
    with registry.extension_work_scope(spec, work):
        for operation in (lambda: installer._unload(name), lambda: installer.settle_before_replacement(tmp_path, name)):
            with pytest.raises(installer.SkillInstallError) as exc:
                operation()
            assert exc.value.code == "live_call_active"
            assert registry._extensions[name] is bundle and not work.cancelled and not settled
    installer.settle_before_replacement(tmp_path, name)
    installer._unload(name)
    assert settled == [name] and name not in registry._extensions


@pytest.mark.parametrize("admission", ["before_update", "before_unload", "idle"])
def test_marketplace_replacement_checks_both_mutation_boundaries(publication, monkeypatch, tmp_path, admission):
    from contextlib import ExitStack
    from ouroboros import extension_loader, skill_install_record, skill_loader
    from ouroboros.marketplace import install

    name, bundle, spec = publication
    work, settled, installed = Work(), [], []
    monkeypatch.setattr(install, "_clawhub_skills_root", lambda root: root / "skills")
    monkeypatch.setattr(skill_loader, "_skill_location_conflict_error", lambda *a, **kw: None)
    monkeypatch.setattr(install, "read_provenance", lambda *a: {"slug": name})
    monkeypatch.setattr(skill_install_record, "read_install_record", lambda *a: {})
    monkeypatch.setattr(skill_install_record, "settle_pending_update",
                        lambda *a, **kw: settled.append(name) or {"status": "none"})
    monkeypatch.setattr(extension_loader, "is_extension_live", lambda *a: True)
    monkeypatch.setattr(extension_loader, "reconcile_extension", lambda *a, **kw: None)
    monkeypatch.setattr(install, "install_skill", lambda *a, **kw: installed.append(name) or install.InstallResult(False, name))
    with ExitStack() as stack:
        if admission == "before_update":
            stack.enter_context(registry.extension_work_scope(spec, work))
        def basis(*a, **kw):
            if admission == "before_unload":
                stack.enter_context(registry.extension_work_scope(spec, work))
            return {}
        monkeypatch.setattr(skill_install_record, "install_basis_fields", basis)
        result = install.update_skill(tmp_path, tmp_path, sanitized_name=name, auto_review=False)
        if admission == "idle":
            assert installed == settled == [name] and name not in registry._extensions
        else:
            assert result.error_code == "live_call_active" and result.error_status == 409
            assert registry._extensions[name] is bundle and not installed and not work.cancelled
            assert settled == ([] if admission == "before_update" else [name])


def test_explicit_source_uninstall_cancels_live_work(publication, monkeypatch, tmp_path):
    from ouroboros import skill_loader, skill_source_install as installer, skill_uninstall_state

    name, _bundle, spec = publication
    payload = tmp_path / "synthetic-payload"
    payload.mkdir()
    monkeypatch.setattr(skill_loader, "find_skill", lambda *a, **kw: SimpleNamespace(
        skill_dir=payload, location="native"))
    monkeypatch.setattr(skill_uninstall_state, "removal_action", lambda *a: "uninstall")
    work = Work()
    with registry.extension_work_scope(spec, work):
        assert installer.uninstall_installed(tmp_path, name)["ok"]
        assert work.cancelled and name not in registry._extensions and not payload.exists()
