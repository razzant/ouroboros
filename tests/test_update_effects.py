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


def _pin(version, build):
    """A valid pin, one field per line, its version and build apart: a version edit and a rebuild merge cleanly."""
    pin = json.loads((pathlib.Path(__file__).resolve().parents[1] / "ouroboros/claudexor_runtime_pin.json").read_text(encoding="utf-8"))
    release = {key: value for key, value in pin["release"].items() if key != "build_sha"}
    pin["release"] = {**release, "version": version, "build_sha": build}
    return json.dumps(pin, indent=2) + "\n"


class _Body:
    """A request whose JSON body is ``body``."""

    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


@pytest.mark.serial  # real preparation in a linked worktree and a real authorization of its switch
@pytest.mark.parametrize("edited", ["4.6.10", "4.6.1"], ids=["another_size", "same_size_in_the_index_second"])
def test_restart_to_update_discloses_the_pin_its_restart_selects(tmp_path, monkeypatch, edited):
    """The composition, neither the official target nor the owner's edit alone: the target rebuilds 4.6.0 and the
    owner's uncommitted edit pins ``edited``, so W′ pins it of the new build. The candidate's effects read it there,
    and the real restart, authorized and bound as Restart to update binds it, selects the same pin and stop. 4.6.10
    is another size, never a stat match; 4.6.1 keeps the committed pin's size and is staged, then edited, inside the
    second Git wrote the index (LABELLED CONTROLLED TIMING, ``stage_then_edit_in_one_second``): Git itself checks it
    by content, and so do preparation and the base its activation compares."""
    from ouroboros import body_adoption
    from supervisor import update_prepare
    from tests.body_candidate_support import git, stage_then_edit_in_one_second
    from tests.test_update_activation import Scene

    scene = Scene(tmp_path, monkeypatch)
    pin = scene.serving / "ouroboros" / "claudexor_runtime_pin.json"
    pin.write_text(_pin("4.6.0", "b" * 40), encoding="utf-8")
    git(scene.serving, "add", "-A")
    git(scene.serving, "commit", "-q", "-m", "engine pin")
    scene.official({"ouroboros/claudexor_runtime_pin.json": _pin("4.6.0", "a" * 40)})
    if len(edited) != len("4.6.0"):
        pin.write_text(_pin(edited, "b" * 40), encoding="utf-8")
    elif not stage_then_edit_in_one_second(scene.serving, "ouroboros/claudexor_runtime_pin.json",
                                           _pin("4.6.0", "b" * 40), _pin(edited, "b" * 40)):
        pytest.skip("the stage and the edit never fit one second here: the race was not set")
    row = scene.prepare()
    assert row["state"] == "ready", row.get("reason")
    based = git(scene.serving, "show", f"{row['base']['files_tree']}:ouroboros/claudexor_runtime_pin.json")
    assert json.loads(based)["release"]["version"] == edited, "the activation's base holds the owner's files"
    assert update_prepare.activation_check(row) == "", "the unchanged checkout reads as the base it was prepared on"
    monkeypatch.setattr(claudexor_daemon, "owned_daemon_provisioned", lambda: True)
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda: Gateway(("4.6.0", "a" * 40)))
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda **kw: pytest.fail("preflight cannot start an engine"))
    monkeypatch.setattr(update_merge, "plan_managed_update_merge", lambda **kw: pytest.fail("a candidate is not planned again"))

    response = asyncio.run(control.api_update_preflight(_Body({"candidate_id": row["candidate_id"]})))
    assert response.status_code == 200
    body = json.loads(response.body)
    effects = body["effects"]
    assert body["merge_plan"] == {"base_sha": row["base_sha"], "target_sha": scene.target, "target_ref": "official",
                                  "update_channel": "development"}
    assert effects["candidate_id"] == row["candidate_id"] and effects["target_sha"] == scene.target
    assert effects["engine"]["source_sha"] == row["layers"]["new_files"]
    assert effects["engine"]["target"] == {"version": edited, "build_sha": "a" * 40}
    assert [effects["engine"][key] for key in ("action", "direction")] == ["replace", "forward"]
    assert effects["replacement_engine"]["action"] == "keep", "the official target alone would keep this engine"
    assert effects["provisional"] is False and effects["active_work"]["total"] == 5
    assert effects["runtime"]["selected"]["python"] == row["environment"]["candidate"]["version"]
    assert effects["runtime"]["changed_by_apply"] is False, "no new app: this interpreter runs the new start"
    # The real restart of that candidate reads the switched file of W′, so it stops the serving engine.
    body_adoption.authorize_update(scene.data, row)
    assert body_adoption.bind_update_restart(scene.data)
    monkeypatch.setattr(server_restart, "DATA_DIR", scene.data)
    monkeypatch.setattr(server_restart, "_OWNER_STOP_BEGAN", threading.Event())
    stops = []
    monkeypatch.setattr(server_restart, "_stop_owned_daemon", lambda label: stops.append(label))
    assert server_restart._next_generation_pin().version == edited
    server_restart._stop_owned_daemon_for_new_pin()
    assert stops == ["Planned restart"]


def test_a_candidate_read_names_its_candidate_before_anything_is_observed(scene, monkeypatch):
    from supervisor import update_prepare

    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda: pytest.fail("nothing is observed for another one"))
    for live in (None, {"candidate_id": "update_b", "state": "ready"}):
        monkeypatch.setattr(update_prepare, "current", lambda live=live: live)
        response = asyncio.run(control.api_update_preflight(_Body({"candidate_id": "update_a"})))
        assert response.status_code == 409 and json.loads(response.body)["reason_code"] == "candidate_changed"
    assert asyncio.run(control.api_update_preflight(_Body({"candidate_id": 7}))).status_code == 400
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda: Gateway(("4.6.0", "a" * 40)))
    legacy = asyncio.run(control.api_update_preflight(_Body({})))
    assert legacy.status_code == 200 and json.loads(legacy.body)["merge_plan"] == scene[1], "an empty body plans"


def test_a_candidate_without_its_composition_or_runtime_stays_unknown(scene, monkeypatch):
    """Never the target's pin for a missing composition, never this interpreter for an unrecorded selection."""
    _repo, plan, _stops = scene
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda: Gateway(("4.5.0", "b" * 40)))
    bare = update_impact.candidate_effects({"candidate_id": "update_a", "target": {"target_sha": plan["target_sha"]}})
    assert bare["engine"]["action"] == "unknown" and bare["engine"]["source_sha"] == ""
    assert bare["replacement_engine"]["action"] == "replace", "the target's pin was readable and is not used"
    assert bare["runtime"]["selected"] == {"python": None, "sqlite": None, "basis": "candidate_preparation"}
    assert bare["runtime"]["changed_by_apply"] is None and bare["candidate_id"] == "update_a"
    tree = git_ops.git_capture(["git", "rev-parse", f"{plan['target_sha']}^{{tree}}"])[1]
    row = {"candidate_id": "update_a", "target": {"target_sha": plan["target_sha"]}, "layers": {"new_files": tree}}
    for selected, changes in (({"version": "3.99.1", "sqlite": "9.9.9"}, True),
                              ({"version": platform.python_version(), "sqlite": sqlite3.sqlite_version}, False)):
        effects = update_impact.candidate_effects({**row, "environment": {"candidate": selected}})
        assert effects["engine"]["action"] == "replace" and effects["engine"]["source_sha"] == tree
        assert effects["runtime"]["changed_by_apply"] is changes
        assert effects["runtime"]["selected"]["python"] == selected["version"]
