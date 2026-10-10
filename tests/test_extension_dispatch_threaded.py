from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace


def _request(tmp_path):
    return SimpleNamespace(
        path_params={"skill": "alpha", "rest": "hello"},
        method="GET",
        app=SimpleNamespace(
            state=SimpleNamespace(drive_root=tmp_path / "drive", repo_dir=tmp_path / "repo")
        ),
    )


def _skill_request(tmp_path, skill="alpha"):
    return SimpleNamespace(
        path_params={"skill": skill},
        method="GET",
        app=SimpleNamespace(
            state=SimpleNamespace(drive_root=tmp_path / "drive", repo_dir=tmp_path / "repo")
        ),
    )


def _decode_response(response):
    return json.loads(response.body.decode("utf-8"))


def test_sync_extension_route_handler_runs_on_worker_thread(tmp_path, monkeypatch):
    import ouroboros.extension_loader as extension_loader
    import ouroboros.gateway.extensions as extensions_api

    (tmp_path / "drive").mkdir()
    (tmp_path / "repo").mkdir()
    loop_thread = threading.get_ident()
    handler_thread = {}
    state_threads = []

    def handler(_request):
        handler_thread["id"] = threading.get_ident()
        return {"ok": True}

    def routes():
        return {
            "/api/extensions/alpha/hello": {
                "skill": "alpha",
                "methods": ("GET",),
                "handler": handler,
            }
        }
    monkeypatch.setattr(extension_loader, "list_routes", routes)
    monkeypatch.setattr(extensions_api, "list_routes", routes)

    def runtime_state(*_a, **_kw):
        state_threads.append(threading.get_ident())
        return {"desired_live": True, "live_loaded": True}

    monkeypatch.setattr(extension_loader, "runtime_state_for_skill_name", runtime_state)

    response = asyncio.run(extensions_api.api_extension_dispatch(_request(tmp_path)))

    assert _decode_response(response) == {"ok": True}
    assert state_threads and all(thread_id != loop_thread for thread_id in state_threads)
    assert handler_thread["id"] != loop_thread


def test_async_extension_route_handler_is_awaited_on_event_loop_thread(tmp_path, monkeypatch):
    import ouroboros.extension_loader as extension_loader
    import ouroboros.gateway.extensions as extensions_api

    (tmp_path / "drive").mkdir()
    (tmp_path / "repo").mkdir()
    handler_thread = {}

    async def handler(_request):
        handler_thread["id"] = threading.get_ident()
        return {"ok": True}

    async def run_case():
        loop_thread = threading.get_ident()
        response = await extensions_api.api_extension_dispatch(_request(tmp_path))
        return loop_thread, response

    def routes():
        return {
            "/api/extensions/alpha/hello": {
                "skill": "alpha",
                "methods": ("GET",),
                "handler": handler,
            }
        }
    monkeypatch.setattr(extension_loader, "list_routes", routes)
    monkeypatch.setattr(extensions_api, "list_routes", routes)
    monkeypatch.setattr(
        extension_loader,
        "runtime_state_for_skill_name",
        lambda *_a, **_kw: {"desired_live": True, "live_loaded": True},
    )

    loop_thread, response = asyncio.run(run_case())

    assert _decode_response(response) == {"ok": True}
    assert handler_thread["id"] == loop_thread


def test_extension_manifest_state_scans_run_on_worker_thread(tmp_path, monkeypatch):
    import ouroboros.extension_loader as extension_loader
    import ouroboros.gateway.extensions as extensions_api
    from ouroboros.skill_loader import SkillReviewState

    (tmp_path / "drive").mkdir()
    (tmp_path / "repo").mkdir()
    loop_thread = threading.get_ident()
    find_threads = []
    state_threads = []

    loaded = SimpleNamespace(
        name="alpha",
        manifest=SimpleNamespace(
            name="alpha",
            description="test",
            version="0.1.0",
            type="extension",
            entry="plugin.py",
            permissions=[],
            env_from_settings=[],
            ui_tab={},
        ),
        enabled=True,
        review=SkillReviewState(status="clean", content_hash="hash"),
        content_hash="hash",
        load_error="",
    )

    def find_skill(*_args, **_kwargs):
        find_threads.append(threading.get_ident())
        return loaded

    def runtime_state(*_args, **_kwargs):
        state_threads.append(threading.get_ident())
        return {"load_error": ""}

    monkeypatch.setattr(extensions_api, "find_skill", find_skill)
    monkeypatch.setattr(extension_loader, "runtime_state_for_skill_name", runtime_state)

    response = asyncio.run(extensions_api.api_extension_manifest(_skill_request(tmp_path)))

    payload = _decode_response(response)
    assert payload["name"] == "alpha"
    assert payload["manifest"]["conflicts"] == []
    assert find_threads and all(thread_id != loop_thread for thread_id in find_threads)
    assert state_threads and all(thread_id != loop_thread for thread_id in state_threads)


def test_skill_reconcile_runs_on_worker_thread(tmp_path, monkeypatch):
    import ouroboros.extension_loader as extension_loader
    import ouroboros.gateway.extensions as extensions_api

    (tmp_path / "drive").mkdir()
    (tmp_path / "repo").mkdir()
    loop_thread = threading.get_ident()
    reconcile_threads = []

    def reconcile_extension(*_args, **_kwargs):
        reconcile_threads.append(threading.get_ident())
        return {"action": "extension_loaded", "reason": "ok", "live_loaded": True, "load_error": ""}

    monkeypatch.setattr(extension_loader, "reconcile_extension", reconcile_extension)

    response = asyncio.run(extensions_api.api_skill_reconcile(_skill_request(tmp_path)))

    assert _decode_response(response)["extension_action"] == "extension_loaded"
    assert reconcile_threads and all(thread_id != loop_thread for thread_id in reconcile_threads)


def test_lifecycle_queue_endpoint_does_not_reconcile_review_jobs(tmp_path, monkeypatch):
    import ouroboros.gateway.extensions as extensions_api
    import ouroboros.skill_review_runner as runner

    (tmp_path / "drive").mkdir()
    (tmp_path / "repo").mkdir()
    reconcile_calls = []

    def reconcile_stale_review_jobs(*_args, **_kwargs):
        reconcile_calls.append(threading.get_ident())

    monkeypatch.setattr(runner, "reconcile_stale_review_jobs", reconcile_stale_review_jobs)

    response = asyncio.run(extensions_api.api_skill_lifecycle_queue(_skill_request(tmp_path)))

    assert "active" in _decode_response(response)
    assert reconcile_calls == []


def test_skill_reconcile_schedule_resync_runs_on_worker_thread(tmp_path, monkeypatch):
    import ouroboros.extension_loader as extension_loader
    import ouroboros.gateway.extensions as extensions_api
    import supervisor.queue as supervisor_queue

    (tmp_path / "drive").mkdir()
    (tmp_path / "repo").mkdir()
    loop_thread = threading.get_ident()
    resync_threads = []

    def reconcile_extension(*_args, **_kwargs):
        return {"action": "extension_loaded", "reason": "ok", "live_loaded": True, "load_error": ""}

    def resync_skill_schedules(*_args, **_kwargs):
        resync_threads.append(threading.get_ident())

    monkeypatch.setattr(extension_loader, "reconcile_extension", reconcile_extension)
    monkeypatch.setattr(supervisor_queue, "resync_skill_schedules", resync_skill_schedules)

    response = asyncio.run(extensions_api.api_skill_reconcile(_skill_request(tmp_path)))

    assert _decode_response(response)["extension_action"] == "extension_loaded"
    assert resync_threads and all(thread_id != loop_thread for thread_id in resync_threads)


def test_dispatch_cost_disclosure_runs_on_worker_thread_and_still_refuses_with_502(tmp_path, monkeypatch):
    import ouroboros.extension_loader as extension_loader
    import ouroboros.extension_process_runner as extension_process_runner
    import ouroboros.gateway.extensions as extensions_api

    (tmp_path / "drive").mkdir()
    (tmp_path / "repo").mkdir()
    loop_thread = threading.get_ident()
    handler_calls = []
    disclosure_threads = []

    def handler(_request):
        handler_calls.append(threading.get_ident())
        return {"ok": True}

    def routes():
        return {
            "/api/extensions/alpha/hello": {
                "skill": "alpha",
                "methods": ("GET",),
                "handler": handler,
            }
        }

    def disclose(*_args, **_kwargs):
        disclosure_threads.append(threading.get_ident())
        return ""

    monkeypatch.setattr(extension_loader, "list_routes", routes)
    monkeypatch.setattr(extensions_api, "list_routes", routes)
    monkeypatch.setattr(
        extension_loader, "runtime_state_for_skill_name",
        lambda *_a, **_kw: {"desired_live": True, "live_loaded": True},
    )
    monkeypatch.setattr(extension_process_runner, "disclose_inprocess_extension_dispatch", disclose)

    response = asyncio.run(extensions_api.api_extension_dispatch(_request(tmp_path)))

    assert _decode_response(response) == {"ok": True}
    assert disclosure_threads and disclosure_threads[0] != loop_thread
    assert len(handler_calls) == 1

    def boom(*_args, **_kwargs):
        raise RuntimeError("ledger closed")

    monkeypatch.setattr(extension_process_runner, "disclose_inprocess_extension_dispatch", boom)
    handler_calls.clear()

    response = asyncio.run(extensions_api.api_extension_dispatch(_request(tmp_path)))

    assert response.status_code == 502
    assert "model-cost disclosure failed" in _decode_response(response)["error"]
    assert handler_calls == []
