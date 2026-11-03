"""Preflight and the real restart decision agree over exact pins; no daemon starts."""

import asyncio
import json
import pathlib
import platform
import sqlite3
import threading

import pytest

from ouroboros import claudexor_daemon, server_restart, update_impact, update_qualification
from ouroboros.gateway import control
from supervisor import git_ops, update_merge
from tests.test_git_lock_owners import _commit, _repo


@pytest.fixture
def scene(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    pin = json.loads((pathlib.Path(__file__).resolve().parents[1] / "ouroboros/claudexor_runtime_pin.json").read_text(encoding="utf-8"))
    pin["release"].update(version="4.6.0", build_sha="a" * 40)
    target = _commit(repo, "update", {
        "ouroboros/claudexor_runtime_pin.json": json.dumps(pin),
        "README.md": "Desktop downloads bundle CPython 3.10.22 and SQLite 3.53.1.\n",
    })
    plan = {"target_sha": target, "merge_commit": target, "base_sha": "b" * 40, "kind": "clean"}
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(server_restart, "_OWNER_STOP_BEGAN", threading.Event())
    monkeypatch.setattr(claudexor_daemon, "owned_daemon_provisioned", lambda: True)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda **kw: pytest.fail("preflight cannot start an engine"))
    monkeypatch.setattr(update_merge, "plan_managed_update_merge", lambda **kw: plan)
    monkeypatch.setattr(update_qualification, "preflight_description",
                        lambda p, **kw: {"letter": None, "qualification": None})
    monkeypatch.setattr(server_restart, "_next_generation_pin", lambda: update_impact._pin_at(plan["merge_commit"]))
    stops = []
    monkeypatch.setattr(server_restart, "_stop_owned_daemon", lambda label: stops.append(label))
    return repo, plan, stops


class Gateway:
    def __init__(self, serving, active=None):
        self.engine_version, self.engine_build_sha = serving
        self.active = {"model": 2, "non_model": 3} if active is None else active
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.closed = True

    def daemon_status(self, **_kw):
        if isinstance(self.active, Exception):
            raise self.active
        return {"admission": {"active": self.active}}


@pytest.mark.parametrize("serving,action,direction", [
    (("4.6.0", "a" * 40), "keep", "unchanged"),
    (("4.5.0", "b" * 40), "replace", "forward"),
    (("4.6.0", "b" * 40), "replace", "same_release"),
    (("4.7.0", "b" * 40), "keep", "backward"),
    (("4.6.1-rc.1", "b" * 40), "replace", "unordered"),
    (None, "unknown", "unknown"),
])
def test_preflight_and_restart_pin_decision_matrix(scene, monkeypatch, serving, action, direction):
    _repo, plan, stops = scene
    gateway = Gateway(serving) if serving else None

    def attach():
        if gateway is None:
            raise RuntimeError("engine unreachable")
        return gateway

    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", attach)
    response = asyncio.run(control.api_update_preflight(None))
    assert response.status_code == 200
    effects = json.loads(response.body)["effects"]
    assert effects["target_sha"] == plan["target_sha"]
    assert effects["engine"]["action"] == action and effects["engine"]["direction"] == direction
    assert effects["engine"]["data_direction"] == "forward_only"
    assert effects["active_work"]["total"] == (5 if serving else None)
    assert effects["runtime"]["installed"]["python"] == platform.python_version()
    assert effects["runtime"]["installed"]["sqlite"] == sqlite3.sqlite_version
    assert effects["runtime"]["bundle"]["sqlite"] == "3.53.1"
    assert effects["runtime"]["changed_by_apply"] is False
    assert stops == [] and (gateway is None or gateway.closed)
    server_restart._stop_owned_daemon_for_new_pin()
    assert bool(stops) is (action == "replace")


@pytest.mark.parametrize("active", [RuntimeError("cannot read"), {}, {"model": 1}, {"model": True, "non_model": 0}])
def test_unreadable_work_stays_unknown_without_hiding_the_pin(scene, monkeypatch, active):
    _repo, plan, _stops = scene
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda: Gateway(("4.5.0", "b" * 40), active))
    effects = update_impact.update_effects(plan)
    assert effects["engine"]["action"] == "replace"
    assert effects["active_work"]["status"] == "unknown" and effects["active_work"]["total"] is None


def test_confirmed_empty_work_is_zero(scene, monkeypatch):
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda: Gateway(("4.5.0", "b" * 40), {"model": 0, "non_model": 0}))
    effects = update_impact.update_effects(scene[1])
    assert effects["active_work"] == {"status": "known", "total": 0, "model": 0, "non_model": 0}


@pytest.mark.parametrize("pin", ['{"schema_version":1,"release":null}', "not JSON"])
def test_unknown_target_pin_preserves_the_restart_handoff(scene, monkeypatch, pin):
    repo, plan, stops = scene
    plan["merge_commit"] = plan["target_sha"] = _commit(repo, "unknown pin", {"ouroboros/claudexor_runtime_pin.json": pin})
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda: Gateway(("4.5.0", "b" * 40)))
    assert update_impact.update_effects(plan)["engine"]["action"] == "unknown"
    server_restart._stop_owned_daemon_for_new_pin()
    assert stops == []


def test_merge_and_replacement_disclose_their_own_pin(scene, monkeypatch):
    repo, plan, _stops = scene
    pin = json.loads((repo / "ouroboros/claudexor_runtime_pin.json").read_text(encoding="utf-8"))
    pin["release"]["version"] = "4.4.0"
    plan["merge_commit"] = _commit(repo, "local pin", {"ouroboros/claudexor_runtime_pin.json": json.dumps(pin)})
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda: Gateway(("4.5.0", "b" * 40)))
    effects = update_impact.update_effects(plan)
    assert effects["engine"]["action"] == "keep" and effects["replacement_engine"]["action"] == "replace"


def test_bundle_unknown_is_not_filled_from_the_running_install(scene):
    repo, plan, _stops = scene
    target = _commit(repo, "no runtime claim", {"README.md": "No bundle metadata.\n"})
    assert update_impact._bundle_runtime(target)["sqlite"] is None
    assert update_impact._bundle_runtime(plan["target_sha"])["sqlite"] == "3.53.1"


def test_preflight_carries_its_exact_range_description_and_survives_its_failure(scene, monkeypatch):
    _repo, plan, _stops = scene
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda: Gateway(("4.6.0", "a" * 40)))
    seen = []

    def described(p, **_kw):
        seen.append((p["base_sha"], p["target_sha"]))
        return {"letter": {"text": "range letter"}, "qualification": {"base_sha": p["base_sha"], "target_sha": p["target_sha"]}}

    monkeypatch.setattr(update_qualification, "preflight_description", described)
    body = json.loads(asyncio.run(control.api_update_preflight(None)).body)
    assert seen == [(plan["base_sha"], plan["target_sha"])]
    assert body["qualification"] == {"base_sha": plan["base_sha"], "target_sha": plan["target_sha"]}
    assert body["letter"] == {"text": "range letter"} and body["effects"]["target_sha"] == plan["target_sha"]

    def unavailable(_p, **_kw):
        raise OSError("letter store unreadable")

    monkeypatch.setattr(update_qualification, "preflight_description", unavailable)
    response = asyncio.run(control.api_update_preflight(None))
    body = json.loads(response.body)
    assert response.status_code == 200 and body["merge_plan"] == plan
    assert body["letter"] is None and body["qualification"] is None, "unknown, never a refusal or a stale range"
