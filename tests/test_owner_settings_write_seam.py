"""The owner settings WRITE seam: the lock is a precondition, and "saved" is a fact.

Two failure classes, both proved against a REAL settings file rather than a mock:

* the settings lock timing out used to be IGNORED — ``_acquire_settings_lock``
  answers ``None`` and the write ran anyway, so a contended save was the one save
  that skipped the precondition it advertises and raced another writer;
* a failure AFTER the bytes landed used to be reported as a failed save (``400``
  from the generic endpoint, ``saved=False`` from onboarding), sending the owner
  to re-do a save that is already on disk.

The onboarding endpoint's own coverage lives in
``test_onboarding_complete_endpoint.py``; these cover the SHARED seam and the
generic ``POST /api/settings`` that reaches it.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings_path = data_dir / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)
    cfg.reset_runtime_mode_baseline_for_tests()
    yield settings_path
    cfg.reset_runtime_mode_baseline_for_tests()


@contextlib.contextmanager
def _foreign_lock(settings_path: pathlib.Path):
    """Hold the settings lock the way another PROCESS would: a real O_EXCL fd,
    released exactly as ``_release_settings_lock`` does (close + unlink)."""
    lock_path = pathlib.Path(str(settings_path) + ".lock")
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        yield lock_path
    finally:
        os.close(fd)
        lock_path.unlink(missing_ok=True)


def _settings_app(monkeypatch, settings_path):
    from ouroboros.gateway import settings as settings_mod

    monkeypatch.setattr(settings_mod, "apply_runtime_provider_defaults", lambda s: (s, False, []))
    monkeypatch.setattr(settings_mod, "_start_supervisor_if_needed_for_request",
                        lambda *_a, **_k: False)
    monkeypatch.setattr(settings_mod, "_apply_settings_to_env", lambda *_a, **_k: None)
    monkeypatch.setattr(settings_mod, "_apply_settings_save_side_effects", lambda *_a, **_k: None)
    app = Starlette(routes=[
        Route("/api/settings", endpoint=settings_mod.api_settings_post, methods=["POST"])])
    app.state.drive_root = settings_path.parent
    app.state.repo_dir = settings_path.parent
    return app


def test_a_contended_lock_aborts_before_the_precondition_and_the_write(isolated_settings):
    from ouroboros.gateway.owner_settings import SettingsLockUnavailable, _owner_write_settings

    checked: list = []
    with _foreign_lock(isolated_settings) as lock_path:
        with pytest.raises(SettingsLockUnavailable):
            _owner_write_settings({"TOTAL_BUDGET": 10.0},
                                  precondition=lambda: checked.append("ran") or "")
        assert lock_path.exists(), "the holder's lock was taken or removed"

    assert checked == [], "the precondition ran without the lock it is supposed to hold"
    assert not isolated_settings.exists(), "a contended write still touched settings.json"


def test_the_commit_boundary_is_only_marked_by_a_real_write(isolated_settings):
    from ouroboros.gateway.owner_settings import (
        CommitBoundary,
        SettingsLockUnavailable,
        SettingsPreconditionFailed,
        _owner_write_settings,
    )

    refused = CommitBoundary()
    with pytest.raises(SettingsPreconditionFailed):
        _owner_write_settings({"TOTAL_BUDGET": 10.0}, precondition=lambda: "no",
                              boundary=refused)
    assert refused.committed is False

    locked = CommitBoundary()
    with _foreign_lock(isolated_settings):
        with pytest.raises(SettingsLockUnavailable):
            _owner_write_settings({"TOTAL_BUDGET": 10.0}, boundary=locked)
    assert locked.committed is False

    landed = CommitBoundary()
    _owner_write_settings({"TOTAL_BUDGET": 10.0}, boundary=landed)
    assert landed.committed is True
    assert json.loads(isolated_settings.read_text(encoding="utf-8"))["TOTAL_BUDGET"] == 10.0


def test_generic_settings_post_reports_a_contended_lock_as_unsaved(monkeypatch,
                                                                   isolated_settings):
    app = _settings_app(monkeypatch, isolated_settings)
    with _foreign_lock(isolated_settings):
        resp = TestClient(app).post("/api/settings", json={"TOTAL_BUDGET": "25"})

    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["code"] == "settings_locked"
    assert body["saved"] is False
    assert not isolated_settings.exists()


def test_generic_settings_post_says_saved_when_a_post_commit_step_fails(monkeypatch,
                                                                       isolated_settings):
    """FINDING 4 at the second site: the write lands at settings.py's commit,
    and every step after it (env projection, supervisor start, hot-reload) is
    post-commit. The broad ``except`` answered all three with ``400``."""
    from ouroboros.gateway import settings as settings_mod

    app = _settings_app(monkeypatch, isolated_settings)

    def _boom(*_a, **_k):
        raise RuntimeError("hot reload exploded")

    monkeypatch.setattr(settings_mod, "_apply_settings_save_side_effects", _boom)
    resp = TestClient(app).post("/api/settings", json={"TOTAL_BUDGET": "25"})

    assert resp.status_code == 500, resp.text
    body = resp.json()
    assert body["saved"] is True
    assert body["status"] == "saved_with_post_commit_error"
    assert body["post_commit_failed"] == "hot-reload"
    assert "hot reload exploded" in body["error"]
    # The claim is checked against the file, not against the handler's opinion.
    assert json.loads(isolated_settings.read_text(encoding="utf-8"))["TOTAL_BUDGET"] == 25.0


def test_a_pre_commit_failure_is_still_reported_as_unsaved(monkeypatch, isolated_settings):
    """The other side of the boundary must not drift: a failure BEFORE the write
    keeps its old, correct answer — and SAYS so.

    The earlier version of this test checked only that the file was absent, which
    is the one thing the CLIENT cannot see. Once a post-commit failure started
    answering ``saved=true``, an envelope that merely omits ``saved`` stopped
    being readable: nothing-was-written and an old/truncated response look
    identical. So the assertion is on the FIELD, not on the disk."""
    from ouroboros.gateway import settings as settings_mod

    app = _settings_app(monkeypatch, isolated_settings)
    monkeypatch.setattr(settings_mod, "_classify_settings_changes",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("before the write")))
    resp = TestClient(app).post("/api/settings", json={"TOTAL_BUDGET": "25"})

    assert resp.status_code == 400, resp.text
    assert "before the write" in resp.json()["error"]
    assert resp.json()["saved"] is False, resp.text
    assert not isolated_settings.exists()


@pytest.mark.parametrize("payload", [
    {"TOTAL_BUDGET": "25", "OUROBOROS_UPDATE_CHANNEL": "not-a-channel"},
    {"TOTAL_BUDGET": "25", "OUROBOROS_POST_TASK_EVOLUTION_CADENCE": "every_n:0"},
    {"TOTAL_BUDGET": "not a number"},
    {"MINIMAX_REGION": "atlantis"},
])
def test_every_generic_validation_refusal_says_saved_false(monkeypatch, isolated_settings,
                                                           payload):
    """Malformed input is a PRE-commit refusal like any other. Each of these
    answered a bare ``{"error": ...}``, so the wizard and any API client had to
    infer "nothing was saved" from the status code."""
    app = _settings_app(monkeypatch, isolated_settings)
    resp = TestClient(app).post("/api/settings", json=payload)

    assert resp.status_code == 400, resp.text
    assert resp.json()["saved"] is False, resp.text
    assert not isolated_settings.exists()


def test_a_malformed_body_says_saved_false(monkeypatch, isolated_settings):
    app = _settings_app(monkeypatch, isolated_settings)
    resp = TestClient(app).post("/api/settings", json=["not", "an", "object"])

    assert resp.status_code == 400, resp.text
    assert resp.json()["saved"] is False, resp.text


# The FIVE single-decision owner endpoints — membership is "calls
# `_owner_write_settings`", not "wears the decorator". Each entry is the route,
# the handler name and a payload its own validation accepts.
_OWNER_SETTINGS_WRITERS = [
    ("/api/owner/runtime-mode", "api_owner_runtime_mode", {"mode": "pro"}),
    ("/api/owner/auto-grant", "api_owner_auto_grant", {"enabled": False}),
    ("/api/owner/context-mode", "api_owner_context_mode", {"mode": "low"}),
    ("/api/owner/scope-review-floor", "api_owner_scope_review_floor", {"floor": "advisory"}),
    ("/api/owner/safety-mode", "api_owner_safety_mode", {"mode": "light"}),
]


def _owner_app(handler_name, route, isolated_settings):
    from ouroboros.gateway import settings as settings_mod

    app = Starlette(routes=[
        Route(route, endpoint=getattr(settings_mod, handler_name), methods=["POST"])])
    app.state.drive_root = isolated_settings.parent
    return app


@pytest.mark.parametrize("route,handler_name,payload", _OWNER_SETTINGS_WRITERS)
def test_owner_endpoints_map_a_contended_lock_to_a_typed_refusal(monkeypatch, isolated_settings,
                                                                 route, handler_name, payload):
    """Every single-decision owner endpoint shares the seam, and none of them had
    a handler at all: the refusal reached Starlette as an opaque 500 that said
    nothing about whether the file changed. Parametrised over ALL FIVE, so the
    guard cannot quietly go missing from four of them."""
    from ouroboros.gateway import settings as settings_mod

    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(settings_mod, "_has_running_agent_tasks", lambda: False, raising=False)
    app = _owner_app(handler_name, route, isolated_settings)

    with _foreign_lock(isolated_settings):
        resp = TestClient(app).post(route, json=payload)

    assert resp.status_code == 503, resp.text
    assert resp.json()["code"] == "settings_locked"
    assert resp.json()["saved"] is False
    assert not isolated_settings.exists()


@pytest.mark.parametrize("route,handler_name,_payload", _OWNER_SETTINGS_WRITERS)
def test_owner_endpoint_validation_refusals_say_saved_false(monkeypatch, isolated_settings,
                                                            route, handler_name, _payload):
    """The same field on the same endpoints' PRE-commit validation path."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    app = _owner_app(handler_name, route, isolated_settings)
    resp = TestClient(app).post(route, json={"mode": "?", "enabled": "?", "floor": "?"})

    assert resp.status_code == 400, resp.text
    assert resp.json()["saved"] is False, resp.text
    assert not isolated_settings.exists()


def test_the_capability_ack_is_not_a_settings_writer(monkeypatch, isolated_settings, tmp_path):
    """FINDING 3, decided the smaller way. `api_acknowledge_capability` wore
    `owner_write_guard` but never calls `_owner_write_settings`: it writes its own
    route-fingerprinted evidence file. The decorator translated exceptions that
    cannot be raised while implying the endpoint was lock-guarded — under a
    genuinely held lock the five writers above refuse 503 and this one records
    its acknowledgement and answers 200, which is CORRECT and now unclaimed.

    Widening the settings lock to cover an unrelated ledger would have made the
    decorator true at the price of coupling a capability ack to whether some
    settings save is in flight. The decorator went instead; this test pins both
    halves so the count of guarded settings writers stays five."""
    from ouroboros.gateway import settings as settings_mod

    assert not hasattr(settings_mod.api_acknowledge_capability, "__wrapped__"), (
        "the capability ack is wearing owner_write_guard again; it writes no settings"
    )

    app = Starlette(routes=[Route("/api/owner/capability-ack",
                                  endpoint=settings_mod.api_acknowledge_capability,
                                  methods=["POST"])])
    app.state.drive_root = tmp_path / "drive"
    with _foreign_lock(isolated_settings):
        resp = TestClient(app).post("/api/owner/capability-ack", json={
            "provider": "openai", "model": "gpt-5.6-luna", "window_tokens": 1_000_000})

    assert resp.status_code == 200, resp.text
    assert resp.json()["ok"] is True
    assert resp.json()["ack"]["window_tokens"] == 1_000_000
    assert not isolated_settings.exists(), "a capability ack touched settings.json"


def test_the_environment_cannot_author_an_install_time_fact_through_a_generic_save(
        monkeypatch, isolated_settings):
    """FINDING 1 at the generic endpoint. The merge skip-list blocks the request
    BODY, so the reviewer's probe never sent one: it put the preset marker in the
    ENVIRONMENT. `load_settings` overlaid it, the save carried it through, and an
    environment-only marker landed on disk as a durable install-time fact."""
    from ouroboros import config as cfg

    monkeypatch.setenv("OUROBOROS_SUBSCRIPTION_PRESET_VERSION", "1")
    monkeypatch.setenv("OUROBOROS_ONBOARDING_COMPLETED_AT", "2020-01-01T00:00:00Z")
    assert cfg.load_settings()["OUROBOROS_SUBSCRIPTION_PRESET_VERSION"] == "", (
        "the loader read an install-time fact out of the environment")

    app = _settings_app(monkeypatch, isolated_settings)
    resp = TestClient(app).post("/api/settings", json={"TOTAL_BUDGET": "25"})

    assert resp.status_code == 200, resp.text
    saved = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert not saved.get("OUROBOROS_SUBSCRIPTION_PRESET_VERSION")
    assert not saved.get("OUROBOROS_ONBOARDING_COMPLETED_AT")


def test_install_time_facts_are_never_projected_back_into_the_environment(monkeypatch,
                                                                          isolated_settings):
    """The other direction of the same set: exported, they would be read back by
    the next process that loads settings from a bare environment."""
    from ouroboros import config as cfg

    for key in cfg.ENDPOINT_AUTHORED_SETTINGS:
        monkeypatch.delenv(key, raising=False)
        assert key not in cfg.settings_env_keys()

    cfg.apply_settings_to_env({
        "OUROBOROS_SUBSCRIPTION_PRESET_VERSION": "1",
        "OUROBOROS_ONBOARDING_COMPLETED_AT": "2026-08-09T00:00:00Z",
    })

    for key in cfg.ENDPOINT_AUTHORED_SETTINGS:
        assert key not in os.environ, f"{key} was projected into the environment"


def test_a_lock_held_read_does_not_wait_for_the_lock_it_holds(isolated_settings):
    """FINDING 5, at the config seam: ``load_settings`` takes the lock, so a
    precondition running INSIDE it must use the lock-held read or spend the full
    2s timeout re-taking a lock it already owns."""
    import time

    from ouroboros import config as cfg

    isolated_settings.write_text(json.dumps({"TOTAL_BUDGET": 42.0}), encoding="utf-8")
    with _foreign_lock(isolated_settings):
        started = time.monotonic()
        settings = cfg.load_settings_lock_held()
        elapsed = time.monotonic() - started

    assert settings["TOTAL_BUDGET"] == 42.0
    assert elapsed < 0.5, f"the lock-held read waited {elapsed:.2f}s for a lock it holds"


def test_settings_save_body_runs_off_the_event_loop():
    """The save body is synchronous work — validation, the disk write, env
    projection, hot-reload side effects, and (when review keys changed)
    NETWORK evidence fetches for the warning surface. On the event loop it
    froze every other request and WebSocket for the whole save."""
    import ast
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "ouroboros" / "gateway" / "settings.py"
    ).read_text(encoding="utf-8")
    endpoint_text = ""
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "api_settings_post":
            endpoint_text = ast.unparse(node)
    assert "asyncio.to_thread(_api_settings_post_sync" in endpoint_text

    # Worker threads do not inherit the event loop's free serialization: a
    # writer interleaving read-merge-write with another would silently drop
    # keys. EVERY settings.py writer holds the seam-wide document lock — the
    # generic save's threaded body and all five single-decision endpoints.
    sync_text = ""
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "_api_settings_post_sync":
            sync_text = ast.unparse(node)
    assert "settings_document_mutation" in sync_text
    # Named per writer, not a substring count: a seventh writer added without
    # the lock must FAIL this pin, and a refactor must not satisfy it by
    # accident.
    writers = {
        "_api_settings_post_sync",
        "_api_owner_runtime_mode_sync",
        "_api_owner_auto_grant_sync",
        "_api_owner_context_mode_sync",
        "_api_owner_scope_review_floor_sync",
        "_api_owner_safety_mode_sync",
    }
    # The loop must NEVER hold the document lock: every async settings writer
    # delegates its locked body to a worker thread.
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.AsyncFunctionDef) and node.name in {
            "api_owner_runtime_mode", "api_owner_auto_grant", "api_owner_context_mode",
            "api_owner_scope_review_floor", "api_owner_safety_mode",
        }:
            assert "asyncio.to_thread" in ast.unparse(node), (
                f"{node.name} must run its locked body off the event loop"
            )
    seen = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in writers:
            seen[node.name] = ast.unparse(node)
    assert set(seen) == writers, f"missing writer functions: {writers - set(seen)}"
    for name, text in seen.items():
        assert "settings_document_mutation" in text, (
            f"{name} must hold the document lock across its read-merge-write"
        )
    # The onboarding transaction lives in its own module and holds the lock
    # from its write through env projection and hot-reload (a fingerprint
    # precondition refuses ITS stale merge, not a later writer's).
    onboarding_src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "ouroboros" / "gateway" / "onboarding.py"
    ).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(onboarding_src)):
        if not isinstance(node, ast.With):
            continue
        header = "".join(ast.unparse(item.context_expr) for item in node.items)
        if "settings_document_mutation" not in header:
            continue
        locked_body = "".join(ast.unparse(stmt) for stmt in node.body)
        assert "_owner_write_settings(" in locked_body, (
            "the onboarding write must sit inside the document lock"
        )
        assert "apply_settings_to_env(" in locked_body, (
            "the lock must cover the environment projection, not just the write"
        )
        break
    else:
        raise AssertionError("onboarding.py holds no settings_document_mutation block")

    # And any FUTURE writer: every _owner_write_settings call site in this
    # module must live inside one of the locked writers above. The locked body
    # itself is the one exemption — its caller _api_settings_post_sync holds
    # the lock for it.
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in writers or node.name == "_api_settings_post_locked":
                continue
            if "_owner_write_settings(" in ast.unparse(node):
                assert "settings_document_mutation" in ast.unparse(node), (
                    f"new settings writer {node.name} does not hold the document lock"
                )


def test_reviewer_slots_reload_does_not_await_the_status_probe():
    """The shared Claudexor status read can wake a cold daemon and walk model
    discovery; awaiting it inside reloadReviewerSlots held the Save button
    (loadSettings awaits that function) hostage for the whole probe. The
    status surface binding repaints the rows when the snapshot lands."""
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "web" / "modules" / "reviewer_slots.js"
    ).read_text(encoding="utf-8")
    assert "await boundedStatusRefresh(state.store);" in source
    assert "await state.store.refresh" not in source
    # loadSettings awaits BOTH sections via Promise.all: one unbounded sibling
    # would keep the Save button hostage to the same probe.
    subagents = (
        pathlib.Path(__file__).resolve().parents[1]
        / "web" / "modules" / "subagents_settings.js"
    ).read_text(encoding="utf-8")
    assert "await boundedStatusRefresh(state.store);" in subagents
    assert "await state.store.refresh" not in subagents
    store = (
        pathlib.Path(__file__).resolve().parents[1]
        / "web" / "modules" / "claudexor_status_store.js"
    ).read_text(encoding="utf-8")
    assert "Promise.race([refresh, beat]).finally(() => { if (timer) clearTimeout(timer); });" in store, (
        "the bounded beat must race the refresh (never cancel it) and clear "
        "the losing timer"
    )


def test_a_second_gateway_writer_cannot_erase_the_first_writers_committed_key(
    monkeypatch, isolated_settings,
):
    """Two gateway writers, one document: the second must not erase the first's key.

    The generic save is the fragile direction. It does NOT re-merge onto a fresh
    document — it reads, merges the incoming keys into a FULL snapshot, and persists
    that snapshot (the shape both parents of this sync shipped; the transform it now
    goes through reads the fresh document under the file lock and deliberately keeps
    the caller's snapshot). Nothing about the write itself prevents a lost update; the
    ONLY thing that does is that its read and its write are one transaction under the
    in-process document lock, which every other gateway writer also holds.

    So the pin is the transaction, proved by racing it: park the generic save inside
    the lock with its snapshot already built, start a dedicated single-decision writer
    concurrently, and show that the second writer cannot land while the first holds the
    lock — which is exactly why the first's snapshot cannot have gone stale — and that
    once both have run, each other's key survives.

    Membership of that closed set is pinned separately, by
    ``test_settings_save_body_runs_off_the_event_loop``: every ``_owner_write_settings``
    call site in ``gateway/settings.py`` must sit inside a locked writer. Writers that
    take only the FILE lock (``tools/control_runtime.py::_set_tool_timeout``, the
    launcher's ``_save_settings``, the ``config`` helpers) are outside this seam and can
    still lose an update; that residual pre-dates this sync on both parents, is
    disclosed in SYNC2_LOG.md, and needs a settings-transaction contract the in-process
    lock cannot provide (the launcher is a different process entirely).
    """
    import threading

    from ouroboros.gateway import settings as settings_mod

    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false")
    isolated_settings.write_text(json.dumps({"TOTAL_BUDGET": 10.0}), encoding="utf-8")

    app = _settings_app(monkeypatch, isolated_settings)
    app.routes.append(Route(
        "/api/owner/auto-grant", endpoint=settings_mod.api_owner_auto_grant, methods=["POST"],
    ))

    at_write = threading.Event()
    release = threading.Event()
    real_write = settings_mod._owner_write_settings

    def parked_write(settings, **kwargs):
        # Inside the generic save's document-lock transaction, snapshot already
        # built, bytes not yet on disk: the exact instant a lost update would be
        # decided.
        at_write.set()
        assert release.wait(timeout=10), "the parked generic save was never released"
        return real_write(settings, **kwargs)

    monkeypatch.setattr(settings_mod, "_owner_write_settings", parked_write)

    generic: dict = {}
    dedicated: dict = {}

    def run_generic():
        generic["response"] = TestClient(app).post("/api/settings", json={"TOTAL_BUDGET": "77"})

    def run_dedicated():
        dedicated["response"] = TestClient(app).post(
            "/api/owner/auto-grant", json={"enabled": True})

    saver = threading.Thread(target=run_generic, name="generic-save")
    saver.start()
    assert at_write.wait(timeout=10), "the generic save never reached its write"

    toggler = threading.Thread(target=run_dedicated, name="auto-grant")
    toggler.start()
    toggler.join(timeout=1.0)
    assert toggler.is_alive(), (
        "the dedicated writer ran while the generic save held the document lock; "
        "the generic save's snapshot can therefore go stale between its read and "
        "its write, and whichever writes second erases the other's key"
    )
    assert "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS" not in json.loads(
        isolated_settings.read_text(encoding="utf-8")
    ), "the blocked writer persisted its key anyway"

    release.set()
    saver.join(timeout=10)
    toggler.join(timeout=10)
    assert not saver.is_alive() and not toggler.is_alive()

    assert generic["response"].status_code == 200, generic["response"].text
    assert dedicated["response"].status_code == 200, dedicated["response"].text
    stored = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert stored["TOTAL_BUDGET"] == 77.0, "the second writer erased the first's key"
    assert stored["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"] == "true"
