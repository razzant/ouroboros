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

from ouroboros.reviewer_slot_config import review_pool_state
from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS


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


@pytest.fixture
def _clean_subagent_env(monkeypatch):
    for key in (
        "OUROBOROS_SUBAGENTS",
        "OUROBOROS_SUBAGENT_HARNESS",
        "OUROBOROS_SUBAGENT_PROFILE",
    ):
        monkeypatch.delenv(key, raising=False)


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


@pytest.fixture
def cyber_settings(isolated_settings, monkeypatch):
    from ouroboros import config as cfg

    for key in cfg.SETTINGS_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    initial = {
        "OUROBOROS_RUNTIME_MODE": "cyber_pro", "OUROBOROS_SAFETY_MODE": "full",
        "OUROBOROS_CONTEXT_MODE": "max", "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "false",
        "OUROBOROS_REVIEW_ENFORCEMENT": "blocking",
        **{key: "original-install-fact" for key in cfg.ENDPOINT_AUTHORED_SETTINGS},
    }
    isolated_settings.write_text(json.dumps(initial), encoding="utf-8")
    cfg.apply_settings_to_env(cfg.load_settings())
    cfg.initialize_runtime_mode_baseline("cyber_pro")
    return isolated_settings


def test_cyber_save_settings_can_configure_supervisor_and_keys(cyber_settings):
    from ouroboros import config as cfg

    chosen = {
        "OUROBOROS_SAFETY_MODE": "off", "OUROBOROS_RUNTIME_MODE": "pro",
        "OUROBOROS_MODEL": "openai/test-model", "SERVICE_API_KEY": "owner-test-key",
    }
    cfg.save_settings({**cfg.load_settings(), **chosen})

    stored = json.loads(cyber_settings.read_text(encoding="utf-8"))
    assert {key: stored[key] for key in chosen} == chosen
    assert stored["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"
    assert cfg.get_runtime_mode() == "cyber_pro", "saved access is restart-bound"


@pytest.mark.parametrize("key,value", [
    ("OUROBOROS_SAFETY_MODE", "off"), ("OUROBOROS_CONTEXT_MODE", "low"), ("OUROBOROS_CONTEXT_MODE", "nano"),
])
def test_pro_lowering_ratchets_use_effective_boot_mode(cyber_settings, monkeypatch, key, value):
    from ouroboros import config as cfg

    cfg.reset_runtime_mode_baseline_for_tests()
    cfg.initialize_runtime_mode_baseline("pro")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    before = cyber_settings.read_bytes()
    with pytest.raises(PermissionError, match="lowering refused"):
        cfg.save_settings({**cfg.load_settings(), key: value})
    assert cyber_settings.read_bytes() == before


def test_cyber_generic_post_saves_controls_and_preserves_fact_provenance(cyber_settings, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.gateway import settings as settings_mod

    app = _settings_app(monkeypatch, cyber_settings)
    monkeypatch.setattr(settings_mod, "_apply_settings_to_env", cfg.apply_settings_to_env)
    chosen = {
        "OUROBOROS_RUNTIME_MODE": "pro", "OUROBOROS_SAFETY_MODE": "off",
        "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "true", "OUROBOROS_MODEL": "openai/test-model",
        "OUROBOROS_CONTEXT_MODE": "low", "OUROBOROS_REVIEW_ENFORCEMENT": "advisory",
        "SERVICE_API_KEY": "owner-test-key",
    }
    response = TestClient(app).post("/api/settings", json={
        **chosen, "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "true", "OUROBOROS_CONTEXT_MODE": "low",
        **{key: "forged-fact" for key in cfg.ENDPOINT_AUTHORED_SETTINGS},
    })
    assert response.status_code == 200, response.text
    stored = json.loads(cyber_settings.read_text(encoding="utf-8"))
    assert {key: stored[key] for key in chosen} == chosen
    assert stored["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"
    assert all(stored[key] == "original-install-fact" for key in cfg.ENDPOINT_AUTHORED_SETTINGS)
    assert response.json()["restart_required"] is True
    assert "OUROBOROS_RUNTIME_MODE" in response.json()["restart_keys"]
    assert response.json()["next_task_changed"] is True
    assert cfg.get_runtime_mode() == os.environ["OUROBOROS_RUNTIME_MODE"] == "cyber_pro"
    # An unrelated later save keeps the pending next-boot value and does not author facts.
    later = TestClient(app).post("/api/settings", json={"TOTAL_BUDGET": 25})
    assert later.status_code == 200, later.text
    assert json.loads(cyber_settings.read_text())["OUROBOROS_RUNTIME_MODE"] == "pro"
    events = [json.loads(line) for line in (cyber_settings.parent / "logs/events.jsonl").read_text().splitlines()]
    change = next(event for event in events if event.get("action") == "settings_controls")
    assert change["changes"]["OUROBOROS_SAFETY_MODE"] == {"old": "full", "new": "off"}
    assert change["changes"]["OUROBOROS_CONTEXT_MODE"] == {"old": "max", "new": "low"}
    assert change["changes"]["OUROBOROS_REVIEW_ENFORCEMENT"] == {"old": "blocking", "new": "advisory"}
    assert "owner-test-key" not in json.dumps(change)


@pytest.mark.parametrize("mode,status", [("cyber_pro", 200), ("pro", 409)])
@pytest.mark.parametrize("context_mode", ["low", "nano"])
def test_context_owner_endpoint_allows_cyber_during_work(cyber_settings, monkeypatch, mode, status, context_mode):
    from ouroboros import config as cfg
    from ouroboros.gateway import settings as settings_mod
    from supervisor.active_activity import get_direct_activity_registry

    cfg.reset_runtime_mode_baseline_for_tests()
    cfg.initialize_runtime_mode_baseline(mode)
    app = Starlette(routes=[Route(
        "/api/owner/context-mode", endpoint=settings_mod.api_owner_context_mode, methods=["POST"])])
    app.state.drive_root = cyber_settings.parent
    registry = get_direct_activity_registry()
    registry.register("settings-author", 1)
    try:
        assert settings_mod._has_running_agent_tasks()
        response = TestClient(app).post("/api/owner/context-mode", json={"mode": context_mode})
    finally:
        registry.unregister("settings-author")
    assert response.status_code == status, response.text
    assert json.loads(cyber_settings.read_text())["OUROBOROS_CONTEXT_MODE"] == (context_mode if status == 200 else "max")


@pytest.mark.parametrize("has_context", [True, False])
@pytest.mark.parametrize("context_mode", ["low", "nano"])
def test_cyber_can_self_lower_context_and_author_its_marker(cyber_settings, has_context, context_mode):
    from ouroboros import config as cfg

    if not has_context:
        raw = json.loads(cyber_settings.read_text())
        raw.pop("OUROBOROS_CONTEXT_MODE")
        raw.pop("OUROBOROS_CONTEXT_MODE_AUTO_LOW")
        cyber_settings.write_text(json.dumps(raw))
    cfg.save_settings({**cfg.load_settings(), "OUROBOROS_CONTEXT_MODE": context_mode})
    stored = json.loads(cyber_settings.read_text())
    assert stored["OUROBOROS_CONTEXT_MODE"] == context_mode
    assert cfg.load_settings()["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"
    assert cfg.load_settings()["OUROBOROS_CONTEXT_MODE"] == context_mode


def test_cyber_context_save_retains_current_task_snapshot(cyber_settings, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.gateway import settings as settings_mod
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot

    app = _settings_app(monkeypatch, cyber_settings)
    monkeypatch.setattr(settings_mod, "_apply_settings_to_env", cfg.apply_settings_to_env)
    snapshot = task_settings_snapshot(cfg.load_settings(), dict(os.environ))
    with task_settings_scope(snapshot):
        response = TestClient(app).post("/api/settings", json={
            "OUROBOROS_CONTEXT_MODE": "low", "OUROBOROS_REVIEW_ENFORCEMENT": "advisory",
        })
        assert response.status_code == 200, response.text
        assert cfg.get_owner_context_mode() == "max"
        assert cfg.get_review_enforcement() == "blocking"
    assert cfg.get_owner_context_mode() == "low"
    assert cfg.get_review_enforcement() == "advisory"
    assert snapshot.settings["OUROBOROS_CONTEXT_MODE"] == "max"


def test_cyber_still_cannot_write_a_pinned_benchmark_snapshot(cyber_settings, monkeypatch):
    import hashlib
    from ouroboros import config as cfg

    before = cyber_settings.read_bytes()
    monkeypatch.setenv(cfg.SETTINGS_INTEGRITY_ENV, hashlib.sha256(before).hexdigest())
    with pytest.raises(cfg.SettingsIntegrityError, match="immutable"):
        cfg.save_settings({**cfg.load_settings(), "OUROBOROS_SAFETY_MODE": "off"})
    assert cyber_settings.read_bytes() == before


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
    monkeypatch.setattr(settings_mod, "_merge_settings_payload",
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


def test_generic_settings_save_validates_and_canonicalizes_available_subagents(
    monkeypatch, isolated_settings,
):
    from ouroboros.configured_subagents import SUBAGENTS_SETTING, parse_configured_subagents

    app = _settings_app(monkeypatch, isolated_settings)
    payload = {
        "enabled": True,
        "items": [{
            "subagent_id": "owner-row",
            "name": "",
            "recommended_use": "Use for owner-selected work.",
            "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-sol"},
        }],
    }
    # No row is marked Reviewer, so the owner confirms saving an empty review pool.
    response = TestClient(app).post(
        "/api/settings", json={SUBAGENTS_SETTING: payload, "allow_empty_review_pool": True})

    assert response.status_code == 200, response.text
    saved = json.loads(isolated_settings.read_text(encoding="utf-8"))
    canonical = saved[SUBAGENTS_SETTING]
    assert isinstance(canonical, str)
    parsed = parse_configured_subagents(canonical)
    assert parsed.items[0].name == ""
    assert '"name"' not in canonical
    assert not {key for key in saved if key.lower() == "allow_empty_review_pool"}, "a request flag, never a setting"


_OWNER_CATALOG = {"enabled": True, "items": [{
    "subagent_id": "owner-row", "recommended_use": "Use for owner-selected work.",
    "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-sol"},
}]}


def _spy_on_the_pool_judge(monkeypatch, record):
    """Package A's real ``review_pool_save_error`` behind a call log: the tests pin WHEN the
    gateway asks the judge; what it answers is A's own rule."""
    from ouroboros import reviewer_slot_config

    real = reviewer_slot_config.review_pool_save_error

    def judge(raw, *, allow_empty):
        record(raw, allow_empty)
        return real(raw, allow_empty=allow_empty)

    monkeypatch.setattr(reviewer_slot_config, "review_pool_save_error", judge)
    return real


def test_a_changed_catalog_meets_the_empty_pool_rule_and_the_owner_flag_confirms_it(
    monkeypatch, isolated_settings,
):
    """Package A's empty-pool rule judges a catalog save as THIS save produces it: its
    text is a typed ``empty_review_pool`` 400 that writes nothing; ``allow_empty_review_pool``
    is the owner's confirmation; a save that re-posts the stored catalog is not judged."""
    from ouroboros.configured_subagents import SUBAGENTS_SETTING

    judged = []
    real = _spy_on_the_pool_judge(
        monkeypatch, lambda raw, allow_empty: judged.append((json.loads(raw)["items"][0]["recommended_use"], allow_empty)))
    client = TestClient(_settings_app(monkeypatch, isolated_settings))

    refused = client.post("/api/settings", json={SUBAGENTS_SETTING: _OWNER_CATALOG})
    assert refused.status_code == 400, refused.text
    assert refused.json()["code"] == "empty_review_pool" and refused.json()["saved"] is False
    assert refused.json()["error"] == real(json.dumps(_OWNER_CATALOG), allow_empty=False) != ""
    assert not isolated_settings.exists()
    assert judged == [("Use for owner-selected work.", False)]

    confirmed = client.post("/api/settings", json={SUBAGENTS_SETTING: _OWNER_CATALOG, "allow_empty_review_pool": True})
    assert confirmed.status_code == 200, confirmed.text
    stored = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert json.loads(stored[SUBAGENTS_SETTING])["items"][0]["subagent_id"] == "owner-row"
    assert len(judged) == 1, "a confirmed save is not judged"

    again = client.post("/api/settings", json={SUBAGENTS_SETTING: _OWNER_CATALOG, "TOTAL_BUDGET": "25"})
    assert again.status_code == 200, again.text
    assert len(judged) == 1, "re-posting the stored catalog is not a catalog change"

    edited = {**_OWNER_CATALOG, "items": [{**_OWNER_CATALOG["items"][0], "recommended_use": "Edited use.",
                                           "review_eligible": True}]}
    accepted = client.post("/api/settings", json={SUBAGENTS_SETTING: edited})
    assert accepted.status_code == 200, accepted.text
    assert judged[-1] == ("Edited use.", False)
    assert json.loads(json.loads(isolated_settings.read_text(encoding="utf-8"))[SUBAGENTS_SETTING])[
        "items"][0]["recommended_use"] == "Edited use."


def test_re_posting_the_catalog_a_read_showed_is_not_a_catalog_change(
    monkeypatch, isolated_settings, _clean_subagent_env,
):
    """With no catalog stored, the read seam shows the factory reviewer rows (a never-configured
    install) and every save re-posts them: an unrelated save is not refused by the empty-pool rule
    and persists the shown catalog, while an edit that unmarks every reviewer is judged."""
    from ouroboros.configured_subagents import SUBAGENTS_SETTING
    from ouroboros.gateway import settings as settings_mod

    judged = []
    _spy_on_the_pool_judge(monkeypatch, lambda raw, allow_empty: judged.append(len(json.loads(raw)["items"])))
    isolated_settings.write_text(json.dumps({
        "OPENROUTER_API_KEY": "configured", "OUROBOROS_MODEL": "openai/gpt-5.6-sol",
        "OUROBOROS_MODEL_LIGHT": "openai/gpt-5.6-luna",
    }), encoding="utf-8")
    app = _settings_app(monkeypatch, isolated_settings)
    app.router.routes.append(Route("/api/settings", endpoint=settings_mod.api_settings_get, methods=["GET"]))
    client = TestClient(app)
    shown = client.get("/api/settings").json()["_meta"]["available_subagents"]
    assert shown["source"] == "configured" and [r["minted_from"] for r in shown["candidate"]["items"]
                                                 if r["review_eligible"]] == ["factory_default"] * 3
    edited = {**shown["candidate"], "items": [{k: v for k, v in row.items() if k != "review_eligible"}
                                              for row in shown["candidate"]["items"]]}
    refused = client.post("/api/settings", json={SUBAGENTS_SETTING: edited})
    assert refused.status_code == 400 and refused.json()["code"] == "empty_review_pool", refused.text
    assert judged == [3] and SUBAGENTS_SETTING not in json.loads(isolated_settings.read_text(encoding="utf-8"))

    saved = client.post("/api/settings", json={SUBAGENTS_SETTING: shown["candidate"], "TOTAL_BUDGET": "25"})
    assert saved.status_code == 200, saved.text
    assert judged == [3], "re-posting the shown catalog is no catalog change"
    stored = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert [row["route"]["target_id"] for row in json.loads(stored[SUBAGENTS_SETTING])["items"]] == list(
        OPENROUTER_REVIEW_DEFAULTS["triad"])


def test_a_catalog_save_retires_the_stored_review_lanes(monkeypatch, isolated_settings):
    """The pool replaces the former review lanes. The read seam migrates readable lanes into
    catalog marks itself; ``OUROBOROS_REVIEWER_SLOTS`` is still in the loaded document only
    when it cannot read them. There, even a re-posted catalog is judged (retiring the lanes
    never empties review silently), and the save that writes a catalog drops the key."""
    from ouroboros.configured_subagents import SUBAGENTS_SETTING, normalize_configured_subagents

    _rows, unmarked = normalize_configured_subagents(_OWNER_CATALOG)
    unreadable_lanes = json.dumps({"triad": [{"slot_id": "t1"}]})
    isolated_settings.write_text(json.dumps({SUBAGENTS_SETTING: unmarked, "OUROBOROS_REVIEWER_SLOTS": unreadable_lanes}),
                                 encoding="utf-8")
    judged = []
    _spy_on_the_pool_judge(monkeypatch, lambda raw, allow_empty: judged.append(raw))
    client = TestClient(_settings_app(monkeypatch, isolated_settings))

    refused = client.post("/api/settings", json={SUBAGENTS_SETTING: _OWNER_CATALOG})

    assert refused.status_code == 400, refused.text
    assert refused.json()["code"] == "empty_review_pool"
    assert judged == [unmarked]
    assert json.loads(isolated_settings.read_text(encoding="utf-8"))["OUROBOROS_REVIEWER_SLOTS"] == unreadable_lanes

    marked = {**_OWNER_CATALOG, "items": [{**_OWNER_CATALOG["items"][0], "review_eligible": True}]}
    _rows, canonical = normalize_configured_subagents(marked)
    accepted = client.post("/api/settings", json={SUBAGENTS_SETTING: marked})

    assert accepted.status_code == 200, accepted.text
    assert judged == [unmarked, canonical]
    stored = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert "OUROBOROS_REVIEWER_SLOTS" not in stored and stored[SUBAGENTS_SETTING] == canonical


def _pool_world(monkeypatch, *, records=None):
    """The endpoint's durable-state seams (the last-run file, package C's migration records,
    the tariff lookup) bound in memory; the pool itself is package A's real reading."""
    from ouroboros import reviewer_slot_config, server_maintenance
    from ouroboros.gateway import settings as settings_mod

    monkeypatch.setattr(reviewer_slot_config, "reviewer_slot_last_executions", lambda: {
        "api-critic": {"surface": "commit_gate", "observed_model": "openai/gpt-5.6-luna", "record_id": "rec-1"}})
    monkeypatch.setattr(server_maintenance, "review_pool_migration_records", lambda: records or {})
    monkeypatch.setattr(settings_mod, "_review_pool_costs", lambda items, env: {
        "api-critic": {"usd_per_review": 0.12, "basis": "route_tariff"},
        "session-critic": {"usd_per_review": None, "basis": "subscription_seat"},
        "helper": {"usd_per_review": None, "basis": "unknown"}})
    return settings_mod


_POOL_CATALOG = {"enabled": False, "items": [
    {"subagent_id": "api-critic", "recommended_use": "Reviews diffs.", "review_eligible": True,
     "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-luna"}, "delivery": "packet",
     "minted_from": "review_lane"},
    {"subagent_id": "helper", "recommended_use": "Helps.", "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-sol"}},
    {"subagent_id": "session-critic", "recommended_use": "Reads the repository.", "review_eligible": True,
     "route": {"kind": "agent_session", "target_id": "cursor=gpt-5.6-sol-xhigh"}},
    {"subagent_id": "paused", "recommended_use": "Switched off.", "review_eligible": True, "enabled": False,
     "route": {"kind": "api_model", "target_id": "google/gemini-3.8-flash"}},
    {"subagent_id": "bare-critic", "recommended_use": "Reads the work itself.", "review_eligible": True,
     "route": {"kind": "api_model", "target_id": "deepseek/deepseek-v4-pro"}, "effort": "low"},
]}


def test_review_pool_endpoint_reports_the_pool_in_catalog_order(monkeypatch):
    """``GET /api/review-pool`` (contract §1.4): package A's pool slots in catalog order,
    each joined with its row's stored facts, price and last run; a marked row that is
    switched off is excluded by name; the catalog switch does not empty the pool."""
    from ouroboros.configured_subagents import MAX_CONFIGURED_SUBAGENTS

    settings_mod = _pool_world(monkeypatch)
    body = settings_mod.review_pool_payload({"OUROBOROS_SUBAGENTS": json.dumps(_POOL_CATALOG)})

    assert body["limits"] == {"rows": MAX_CONFIGURED_SUBAGENTS}
    assert body["catalog"] == {"present": True, "enabled": False, "rows": 5, "eligible": 4}
    assert [row["subagent_id"] for row in body["pool"]] == ["api-critic", "session-critic", "bare-critic"]
    api, session, bare = body["pool"]
    assert api["route"] == {"kind": "api_model", "target_id": "openai/gpt-5.6-luna", "credential_profile_id": ""}
    assert (api["effort"], api["effort_source"], api["delivery"], api["minted_from"]) == ("high", "auto", "packet", "review_lane")
    assert api["cost"] == {"usd_per_review": 0.12, "basis": "route_tariff"}
    assert api["last_execution"]["record_id"] == "rec-1" and api["handle"]
    assert (session["effort"], session["effort_source"], session["delivery"], session["access"]) == (
        "xhigh", "model_name", "session", "full")
    assert session["cost"] == {"usd_per_review": None, "basis": "subscription_seat"}
    assert session["last_execution"] is None and session["minted_from"] == ""
    assert (bare["effort"], bare["effort_source"], bare["delivery"]) == ("low", "pin", "native")
    assert all(row["review_eligible"] is True and row["enabled"] is True for row in body["pool"])
    assert body["excluded"] == [{"subagent_id": "paused", "reason": "row_disabled"}]
    assert body["row_costs"]["helper"] == {"usd_per_review": None, "basis": "unknown"}
    assert body["config_error"] == "" and body["migration"] is None


def test_review_pool_endpoint_types_a_bad_catalog_and_names_the_migration_snapshot(monkeypatch):
    """A catalog package A cannot read is a typed ``config_error`` with an empty pool,
    never a 500; the newest lanes-to-pool record (package C) names its snapshot."""
    import asyncio

    records = {
        "older": {"ts": "20261001T000000Z", "snapshot": "state/review_migrations/20261001T000000Z-slots-to-pool.json",
                  "reported": "2026-10-01T00:00:05Z"},
        "newer": {"ts": "20261007T214000Z", "snapshot": "state/review_migrations/20261007T214000Z-slots-to-pool.json",
                  "reported": None},
    }
    settings_mod = _pool_world(monkeypatch, records=records)
    unreadable = {**_POOL_CATALOG, "items": [{**_POOL_CATALOG["items"][0], "delivery": "x"}, *_POOL_CATALOG["items"][1:]]}
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", json.dumps(unreadable))
    response = asyncio.run(settings_mod.api_review_pool(None))

    assert response.status_code == 200
    body = json.loads(response.body)
    expected = review_pool_state(json.dumps(unreadable))["error"]
    assert expected and "delivery" in expected and body["config_error"] == expected
    assert body["pool"] == [] and body["excluded"] == [] and body["row_costs"] == {}
    assert body["catalog"]["eligible"] == 4
    # Records without their snapshot files decide no document: the newest is reported as history.
    assert body["migration"] == {"snapshot": records["newer"]["snapshot"], "reported": False,
                                 "trigger": "", "outcome": "", "error": "", "source": "history"}


# --- the migration outcome and the credential fact in the payload (VD3-06, VD3-08) ----------


@pytest.fixture
def pool_root(tmp_path, monkeypatch):
    """A data root with supervisor state bound: the receipts' home (``persist_receipts`` writes
    the snapshot and the ``state.json`` record there, as the saving process does)."""
    from ouroboros import config as cfg
    from ouroboros import review_pool_migration as rpm
    from ouroboros import server_maintenance
    from supervisor import state

    state.init(tmp_path)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "locks").mkdir(parents=True, exist_ok=True)
    state.save_state({})
    monkeypatch.setattr(server_maintenance, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    rpm._MIGRATIONS_SEEN.clear()
    yield tmp_path
    rpm._MIGRATIONS_SEEN.clear()


_AUTHORED_LANES = json.dumps({"triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-luna"}}],
                              "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-luna"}}]})
_BROKEN_LANES = json.dumps({"triad": [{"model": "openai/gpt-5.6-luna"}]})  # no slot_id/route: the strict parser refuses


def _served(settings_mod, document):
    """What ``GET /api/review-pool`` shows for ``document`` once the saving process wrote the receipts."""
    from ouroboros import config as cfg
    from ouroboros import review_pool_receipts, server_maintenance

    loaded = cfg.normalize_settings_raw(dict(document))
    review_pool_receipts.persist_receipts(server_maintenance.DATA_DIR)
    return loaded, settings_mod.review_pool_payload(loaded)


def test_review_pool_payload_carries_the_migration_outcome_that_decides_the_document(pool_root, monkeypatch):
    """VD3-06: ``migration`` is the full receipt — ``{snapshot, reported, trigger, outcome, error,
    source}`` — for the document the payload shows: authored lanes converted (source ``document``),
    a never-configured document's factory rows (``factory``)."""
    from ouroboros import reviewer_slot_config
    from ouroboros.gateway import settings as settings_mod

    monkeypatch.setattr(reviewer_slot_config, "reviewer_slot_last_executions", lambda: {})
    monkeypatch.setattr(settings_mod, "_review_pool_costs", lambda items, env: {})

    loaded, body = _served(settings_mod, {"OUROBOROS_REVIEWER_SLOTS": _AUTHORED_LANES, "OPENROUTER_API_KEY": "present"})
    (snapshot,) = sorted((pool_root / "state" / "review_migrations").glob("*-slots-to-pool.json"))
    assert body["config_error"] == "" and [row["subagent_id"] for row in body["pool"]] == ["review-1", "review-2"]
    assert body["migration"] == {"snapshot": f"state/review_migrations/{snapshot.name}", "reported": False,
                                 "trigger": "lanes_key", "outcome": "converted", "error": "", "source": "document"}

    _loaded, factory = _served(settings_mod, {"OPENROUTER_API_KEY": "present"})
    assert len(factory["pool"]) == 3
    assert (factory["migration"]["trigger"], factory["migration"]["outcome"], factory["migration"]["source"]) == (
        "never_configured", "factory", "document")
    assert factory["migration"]["snapshot"] != body["migration"]["snapshot"], "each document its own receipt"


def test_review_pool_payload_names_the_reason_a_broken_lanes_key_was_retained(pool_root, monkeypatch):
    """VD3-06: a lane value the migration refused stays in the document (no partial migration)
    and the payload says WHY — ``outcome: error`` with the reason and ``source: error`` — while
    the catalog, untouched, still serves whatever pool it had (none here)."""
    from ouroboros import reviewer_slot_config
    from ouroboros.gateway import settings as settings_mod

    monkeypatch.setattr(reviewer_slot_config, "reviewer_slot_last_executions", lambda: {})
    monkeypatch.setattr(settings_mod, "_review_pool_costs", lambda items, env: {})

    loaded, body = _served(settings_mod, {"OUROBOROS_REVIEWER_SLOTS": _BROKEN_LANES, "OPENROUTER_API_KEY": "present"})
    assert loaded["OUROBOROS_REVIEWER_SLOTS"] == _BROKEN_LANES, "retained for the owner's catalog save"
    (snapshot,) = sorted((pool_root / "state" / "review_migrations").glob("*-slots-to-pool.json"))
    migration = body["migration"]
    assert (migration["snapshot"], migration["reported"]) == (f"state/review_migrations/{snapshot.name}", False)
    assert (migration["trigger"], migration["outcome"], migration["source"]) == ("lanes_key", "error", "error")
    assert migration["error"].startswith("OUROBOROS_REVIEWER_SLOTS: ") and "unknown keys" in migration["error"]
    assert body["pool"] == [] and body["config_error"] == ""


@pytest.mark.parametrize("refused", [False, True], ids=["converted_control", "retained_lanes_error"])
def test_the_live_handler_judges_the_receipt_by_the_document_on_disk_not_the_process_projection(pool_root, monkeypatch,
                                                                                               refused):
    """VD3-06, the real path: the server reads the document, the boot receipts it, then
    ``config.apply_settings_to_env`` projects the LIVE settings keys into the process environment
    and ``GET /api/review-pool`` is served from that projection (``runtime_environ``). A lane value
    a refused migration retained lives in the document, not in the projection — so judged against
    the projection the current ``error`` receipt read as ``history`` and the Settings note hid it.
    The handler judges the receipt against the document on disk, the same view the direct payload
    of the loaded document gives; the retained lanes never enter the runtime configuration."""
    import asyncio
    import os

    from ouroboros import config as cfg
    from ouroboros import server_maintenance
    from ouroboros.gateway import settings as settings_mod
    from tests.test_review_pool_migration import N1_DOC, _served_from_disk

    monkeypatch.setattr(settings_mod, "_review_pool_costs", lambda items, env: {})
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    loaded = _served_from_disk(pool_root, monkeypatch, {"OUROBOROS_REVIEWER_SLOTS": _BROKEN_LANES} if refused
                               else dict(N1_DOC))
    server_maintenance._startup_review_pool_notice(loaded)  # the boot's receipt (no owner chat: it waits, unreported)
    cfg.apply_settings_to_env(loaded)  # the server's projection the handler reads
    assert ("OUROBOROS_REVIEWER_SLOTS" in loaded) is refused and "OUROBOROS_REVIEWER_SLOTS" not in os.environ

    response = asyncio.run(settings_mod.api_review_pool(None))
    assert response.status_code == 200
    served = json.loads(response.body)["migration"]
    assert served == settings_mod.review_pool_payload(loaded)["migration"], "the handler and the document view agree"
    assert (served["source"], served["outcome"]) == (("error", "error") if refused else ("document", "factory"))
    assert (served["error"].startswith("OUROBOROS_REVIEWER_SLOTS: ")) is refused and not served["reported"]


def test_review_pool_payload_states_which_pool_rows_have_no_credentials(monkeypatch):
    """VD3-08: ``pool_without_credentials`` lists every pool row whose model this install holds
    no credentials for — all of them is the loud fact the Settings note needs; a subscription
    seat logs in itself and is never listed. The pool (the pinning) is unchanged either way."""
    settings_mod = _pool_world(monkeypatch)
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    catalog = json.dumps(_POOL_CATALOG)

    bare = settings_mod.review_pool_payload({"OUROBOROS_SUBAGENTS": catalog})
    assert [row["subagent_id"] for row in bare["pool"]] == ["api-critic", "session-critic", "bare-critic"]
    assert bare["pool_without_credentials"] == ["api-critic", "bare-critic"]

    funded = settings_mod.review_pool_payload({"OUROBOROS_SUBAGENTS": catalog, "OPENROUTER_API_KEY": "present"})
    assert [row["subagent_id"] for row in funded["pool"]] == [row["subagent_id"] for row in bare["pool"]]
    assert funded["pool_without_credentials"] == []

    unreadable = {**_POOL_CATALOG, "items": [{**_POOL_CATALOG["items"][0], "delivery": "x"}, *_POOL_CATALOG["items"][1:]]}
    broken = settings_mod.review_pool_payload({"OUROBOROS_SUBAGENTS": json.dumps(unreadable)})
    assert broken["config_error"] and broken["pool_without_credentials"] == []


def test_generic_settings_save_projects_model_role_objects_as_json(
        monkeypatch, isolated_settings,
):
    from ouroboros import config as cfg
    from ouroboros.gateway import settings as settings_mod

    app = _settings_app(monkeypatch, isolated_settings)
    monkeypatch.setattr(settings_mod, "_apply_settings_to_env", cfg.apply_settings_to_env)
    response = TestClient(app).post("/api/settings", json={
        "OUROBOROS_MODEL_ACCOUNTS": {"main": "", "fallback": ["work", ""]},
        "OUROBOROS_MODEL_CONTEXT_WINDOWS": {"main": 131072, "fallback": [0, 65536]},
    })

    assert response.status_code == 200, response.text
    stored = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert json.loads(stored["OUROBOROS_MODEL_ACCOUNTS"]) == {
        "fallback": ["work", ""], "main": "",
    }
    assert json.loads(os.environ["OUROBOROS_MODEL_ACCOUNTS"])["main"] == ""
    assert json.loads(os.environ["OUROBOROS_MODEL_CONTEXT_WINDOWS"])["main"] == 131072


def test_generic_settings_save_rejects_malformed_available_subagents_without_write(
    monkeypatch, isolated_settings,
):
    from ouroboros.configured_subagents import SUBAGENTS_SETTING

    app = _settings_app(monkeypatch, isolated_settings)
    response = TestClient(app).post("/api/settings", json={SUBAGENTS_SETTING: {
        "enabled": True,
        "items": [{
            "subagent_id": "bad",
            "name": "Bad",
            "recommended_use": "Bad",
            "route": {"kind": "api_model", "target_id": "x", "extra": True},
        }],
    }})

    assert response.status_code == 400, response.text
    assert response.json()["saved"] is False
    assert not isolated_settings.exists()


def test_settings_get_reads_a_legacy_actor_through_the_seam_without_materializing_it(
    monkeypatch, isolated_settings, _clean_subagent_env,
):
    """A legacy single-harness document reads as the catalog the migration seeds (its session row first)."""
    from ouroboros import config as cfg
    from ouroboros.gateway import settings as settings_mod

    # Other endpoint tests exercise server.py's legacy compatibility wrapper,
    # which intentionally rebinds this gateway module in-process. This direct
    # gateway test pins the real reader it is meant to exercise.
    monkeypatch.setattr(settings_mod, "load_settings", cfg.load_settings)
    original = {
        "OUROBOROS_SUBAGENT_HARNESS": "codex=gpt-5.6-sol:high",
        "OUROBOROS_SUBAGENT_PROFILE": "owner-profile",
    }
    isolated_settings.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(settings_mod, "apply_runtime_provider_defaults", lambda s: (s, False, []))
    app = Starlette(routes=[
        Route("/api/settings", endpoint=settings_mod.api_settings_get, methods=["GET"]),
    ])
    app.state.drive_root = isolated_settings.parent
    app.state.repo_dir = isolated_settings.parent
    response = TestClient(app).get("/api/settings")

    assert response.status_code == 200, response.text
    projection = response.json()["_meta"]["available_subagents"]
    items = projection["candidate"]["items"]
    assert projection["source"] == "configured" and [r["minted_from"] for r in items if r.get("review_eligible")] == ["factory_default"] * 3
    assert items[0]["route"]["credential_profile_id"] == "owner-profile" and not items[0].get("review_eligible")
    assert json.loads(isolated_settings.read_text(encoding="utf-8")) == original


def test_settings_get_reads_the_factory_reviewers_in_place_of_an_unsaved_api_candidate(
    monkeypatch, isolated_settings, _clean_subagent_env,
):
    from ouroboros import config as cfg
    from ouroboros.gateway import settings as settings_mod
    from ouroboros.server_runtime import apply_runtime_provider_defaults

    monkeypatch.setattr(settings_mod, "load_settings", cfg.load_settings)
    monkeypatch.setattr(
        settings_mod, "apply_runtime_provider_defaults", apply_runtime_provider_defaults,
    )
    original = {
        "OPENROUTER_API_KEY": "configured",
        "OUROBOROS_MODEL": "openai/gpt-5.6-sol",
        "OUROBOROS_MODEL_LIGHT": "openai/gpt-5.6-luna",
        "OUROBOROS_SUBAGENTS": "",
        "OUROBOROS_SUBAGENT_HARNESS": "",
    }
    isolated_settings.write_text(json.dumps(original), encoding="utf-8")
    app = Starlette(routes=[
        Route("/api/settings", endpoint=settings_mod.api_settings_get, methods=["GET"]),
    ])
    app.state.drive_root = isolated_settings.parent
    app.state.repo_dir = isolated_settings.parent

    response = TestClient(app).get("/api/settings")

    assert response.status_code == 200, response.text
    projection = response.json()["_meta"]["available_subagents"]
    assert projection["source"] == "configured"
    assert [row["route"]["target_id"] for row in projection["candidate"]["items"]] == list(
        OPENROUTER_REVIEW_DEFAULTS["triad"])
    assert json.loads(isolated_settings.read_text(encoding="utf-8")) == original


def test_a_malformed_body_says_saved_false(monkeypatch, isolated_settings):
    app = _settings_app(monkeypatch, isolated_settings)
    resp = TestClient(app).post("/api/settings", json=["not", "an", "object"])

    assert resp.status_code == 400, resp.text
    assert resp.json()["saved"] is False, resp.text


# The FOUR single-decision owner endpoints — membership is "calls
# `_owner_update_settings`" (directly, or through `_owner_write_settings`), not
# "wears the decorator". Each entry is the route, the handler name and a payload
# its own validation accepts.
_OWNER_SETTINGS_WRITERS = [
    ("/api/owner/runtime-mode", "api_owner_runtime_mode", {"mode": "pro"}),
    ("/api/owner/auto-grant", "api_owner_auto_grant", {"enabled": False}),
    ("/api/owner/context-mode", "api_owner_context_mode", {"mode": "low"}),
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
    endpoint_text = writer_seam_text = ""
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "api_settings_post":
            endpoint_text = ast.unparse(node)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_settings_writer":
            writer_seam_text = ast.unparse(node)
    # The endpoint hands its body to the ONE writer seam, and that seam is
    # what runs it off the loop (and maps the bounded document lock's typed
    # refusal to 503 settings_busy for every writer).
    assert "_run_settings_writer(_api_settings_post_sync, request, body)" in endpoint_text
    assert "asyncio.to_thread(fn, context, body)" in writer_seam_text
    assert "SettingsDocumentBusy" in writer_seam_text

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
        "_api_owner_safety_mode_sync",
    }
    # The loop must NEVER hold the document lock: every async settings writer
    # delegates its locked body to a worker thread.
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.AsyncFunctionDef) and node.name in {
            "api_owner_runtime_mode", "api_owner_auto_grant", "api_owner_context_mode",
            "api_owner_safety_mode",
        }:
            # Off the loop through the ONE writer seam (itself pinned above to
            # asyncio.to_thread + the typed busy mapping), never inline.
            assert "_run_settings_writer(" in ast.unparse(node), (
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
    # And the onboarding endpoint runs that locked body through the SAME bounded
    # writer seam as the settings.py writers (audit point 3): a bare
    # ``to_thread`` was the one settings write with no initiating-writer cap.
    for node in ast.walk(ast.parse(onboarding_src)):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "api_onboarding_complete":
            text = ast.unparse(node)
            assert "_run_settings_writer(" in text and "to_thread(" not in text, (
                "api_onboarding_complete must persist through settings._run_settings_writer"
            )
            break
    else:
        raise AssertionError("onboarding.py defines no api_onboarding_complete")

    # And any FUTURE writer: every _owner_write_settings / _owner_update_settings
    # call site in this module must live inside one of the locked writers above.
    # The locked body itself is the one exemption — its caller
    # _api_settings_post_sync holds the lock for it.
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in writers or node.name == "_api_settings_post_locked":
                continue
            text = ast.unparse(node)
            if "_owner_write_settings(" in text or "_owner_update_settings(" in text:
                assert "settings_document_mutation" in text, (
                    f"new settings writer {node.name} does not hold the document lock"
                )


def test_settings_enrichment_does_not_await_the_status_probe():
    """The shared Claudexor status read can wake a cold daemon and walk model
    discovery; a section reload awaiting it unbounded would hold the Save button
    (loadSettings awaits its enrichment) hostage for the whole probe. The status
    surface binding repaints the rows when the snapshot lands, and the review-pool
    read starts only after the confirmed document is already editable."""
    import pathlib

    modules = pathlib.Path(__file__).resolve().parents[1] / "web" / "modules"
    subagents = (modules / "subagents_settings.js").read_text(encoding="utf-8")
    assert "await boundedStatusRefresh(store);" in subagents
    assert "await store.refresh" not in subagents
    host = (modules / "settings.js").read_text(encoding="utf-8")
    load = host[host.index("async function loadSettings"):host.index("async function reloadSettingsWithFeedback")]
    assert load.index("settingsLoaded = true;") < load.index("reloadReviewPool({ isCurrent })")
    store = (
        pathlib.Path(__file__).resolve().parents[1]
        / "web" / "modules" / "claudexor_status_store.js"
    ).read_text(encoding="utf-8")
    assert "Promise.race([refresh, beat]).finally(() => { if (timer) clearTimeout(timer); });" in store, (
        "the bounded beat must race the refresh (never cancel it) and clear "
        "the losing timer"
    )
