"""Restart Claudexor (Accounts): the preview says what ends and whose, the restart acts only on it.

The engine is a stub (its operation catalog, run list and status) and the data root is the
test's own; the owned-daemon stop and start are recorded doubles. No engine process exists.
Each guard asserts the case that fires and the case that stays quiet.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from ouroboros import claudexor_daemon, claudexor_restart as restart, claudexor_runtime, delegate_recovery
from ouroboros import delegate_custody as custody
from ouroboros import review_run_isolation
from ouroboros.gateways import claudexor_live_runs as live_runs
from ouroboros.gateways.claudexor import ClaudexorUnavailable

STATE_FILTER = {"method": "GET", "path": "/v2/runs", "parameters": [
    {"name": "state", "location": "query", "enum": ["queued", "running", "succeeded", "failed", "cancelled", "interrupted"]}]}


def run(run_id, state="running", **facts):
    return {"runId": run_id, "jobId": f"job-{run_id}", "state": state, "primaryHarness": "codex",
            "startedAt": "2026-11-06T10:00:00Z", **facts}


class Engine:
    """A stub engine: ``runs`` are its live runs; ``page`` rows per list page."""

    def __init__(self, runs=(), *, serving=("5.0.1", "b" * 40), lists=True, page=1000):
        self.runs, self.lists, self.page, self.requests = list(runs), lists, page, []
        self.engine_version, self.engine_build_sha = serving

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def operations(self, **_kw):
        return [STATE_FILTER] if self.lists else [{"method": "GET", "path": "/v2/runs", "parameters": []}]

    def _request(self, method, path, **_kw):
        self.requests.append(path)
        query = parse_qs(urlsplit(path).query)
        rows = [row for row in self.runs if row["state"] == query["state"][0]]
        start = int(query.get("cursor", ["0"])[0])
        end = start + self.page
        return {"runs": rows[start:end], "hasMore": end < len(rows), "nextCursor": str(end) if end < len(rows) else None}

    def daemon_status(self, **_kw):
        return {"admission": {"active": {"model": 2, "non_model": len(self.runs)},
                              "queued": {"model": 1, "non_model": 0}}}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """One installation: a data root with its own custody, a pinned engine and a daemon double."""
    import ouroboros.config as config

    scene = SimpleNamespace(root=tmp_path, engine=Engine(), pin=SimpleNamespace(version="5.0.1", build_sha="b" * 40),
                            provisioned=True, answers=True, calls=[], stop="stopped", start_error=None)

    def read_owned_gateway(**_kw):
        if not scene.answers:
            raise ClaudexorUnavailable("daemon_unreachable", "no answer on the control port")
        return scene.engine

    def ensure_owned_gateway(**_kw):
        scene.calls.append("start")
        if scene.start_error is not None:
            raise scene.start_error
        return SimpleNamespace(close=lambda: None)

    def stop_outcome():
        scene.calls.append("stop")
        if isinstance(scene.stop, Exception):
            raise scene.stop
        return scene.stop

    daemon = SimpleNamespace(stop_outcome=stop_outcome,
                             clear_start_failure_latch=lambda *, cleared_by: scene.calls.append(cleared_by))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(claudexor_daemon, "owned_daemon_provisioned", lambda: scene.provisioned)
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", read_owned_gateway)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", ensure_owned_gateway)
    monkeypatch.setattr(claudexor_daemon, "get_owned_daemon", lambda **_kw: daemon)
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: SimpleNamespace(pin=scene.pin))
    monkeypatch.setattr(review_run_isolation, "attach_home", lambda: None)
    monkeypatch.setattr(delegate_recovery, "owner_restart_in_progress", lambda root: False)
    monkeypatch.delenv("OUROBOROS_CLAUDEXOR_BIN", raising=False)
    return scene


def delegate(root, run_id, task_id="task-1", category="subagent"):
    entry = custody.RunCustody(run_id=run_id, task_id=task_id, invocation_id=f"inv-{run_id}", route_id="codex",
                               model="test", category=category, ledger_root=str(root))
    assert custody.record_started(root, entry)


def rows(root, kind):
    """The supervisor rows of one type (the store writes its own rows beside them)."""
    path = root / "logs" / "supervisor.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [row for row in map(json.loads, lines) if row.get("type") == kind]


# ---- the engine's own run list -------------------------------------------------------------

def test_the_live_run_list_reads_every_page_of_both_live_states_or_refuses():
    engine = Engine([run("q1", "queued"), run("r1"), run("r2"), run("r3"), run("done", "succeeded")], page=2)
    listed = live_runs.read_live_runs(engine)
    assert sorted(row["runId"] for row in listed) == ["q1", "r1", "r2", "r3"], "terminal runs are not live"
    assert [parse_qs(urlsplit(path).query)["state"][0] for path in engine.requests] == ["queued", "running", "running"]
    assert "cursor=2" in engine.requests[-1], "the second page is the engine's own cursor"

    silent = Engine([run("r1")], lists=False)
    with pytest.raises(ClaudexorUnavailable) as refused:
        live_runs.read_live_runs(silent)
    assert refused.value.code == "run_list_unsupported" and silent.requests == [], "an unfiltered page is never read"

    broken = Engine([run("r1")])
    broken._request = lambda *a, **k: {"runs": "none"}
    with pytest.raises(ClaudexorUnavailable) as malformed:
        live_runs.read_live_runs(broken)
    assert malformed.value.code == "malformed_response"


# ---- the preview ---------------------------------------------------------------------------

def test_the_preview_attributes_each_live_run_by_custody_records_never_by_its_path(world):
    delegate(world.root, "run-own")
    delegate(world.root, "run-review", task_id="task-2", category="review")
    world.engine.runs = [
        run("run-own", project={"projectName": "ouroboros", "root": "/work/ouroboros"}),
        run("run-child", "queued", delegatedFromRunId="run-own"),
        run("run-review"),
        # Another program working inside this installation's own tree: the path proves nothing.
        run("run-foreign", project={"projectName": "repo", "root": str(world.root / "repo")}),
    ]
    preview = restart.restart_preview()
    assert preview["runs"] | {"items": None} == {"status": "known", "total": 4, "ouroboros": 3, "other": 1,
                                                  "error": None, "items": None, "items_omitted": 0}
    owners = {item["run_id"]: (item["owner"], item["task_id"], item["kind"]) for item in preview["runs"]["items"]}
    assert owners == {"run-own": ("ouroboros", "task-1", "subagent"), "run-child": ("ouroboros", "task-1", "subagent"),
                      "run-review": ("ouroboros", "task-2", "review"), "run-foreign": ("other", "", "")}
    assert preview["engine"] == {"state": "serving", "serving": {"version": "5.0.1", "build_sha": "b" * 40},
                                 "pin": {"version": "5.0.1", "build_sha": "b" * 40}, "direction": "unchanged"}
    assert preview["model_calls"] == {"status": "known", "active": 2, "queued": 1}

    world.engine.runs = []
    quiet = restart.restart_preview()
    assert quiet["runs"]["status"] == "known" and quiet["runs"]["total"] == 0 and quiet["runs"]["items"] == []
    assert quiet["token"] != preview["token"], "the token names the runs that end"


@pytest.mark.parametrize("pin,serving,direction,warns", [
    ("4.9.0", "5.0.1", "backward", True),
    ("5.0.1", "5.0.1", "unchanged", False),
    ("5.1.0", "5.0.1", "forward", False),
    (None, "5.0.1", "unknown", False),
])
def test_the_preview_warns_only_when_the_pin_is_below_the_serving_engine(world, pin, serving, direction, warns):
    world.pin = None if pin is None else SimpleNamespace(version=pin, build_sha="b" * 40)
    world.engine = Engine(serving=(serving, "b" * 40))
    preview = restart.restart_preview()
    assert preview["engine"]["direction"] == direction
    assert [warning["code"] for warning in preview["warnings"]] == (["engine_pin_below_serving"] if warns else [])
    if warns:
        warning = preview["warnings"][0]
        assert (warning["pin_version"], warning["serving_version"]) == (pin, serving)
        assert pin in warning["message"] and serving in warning["message"] and "does not start" in warning["message"]


def test_an_engine_binary_override_starts_no_pinned_release_so_it_cannot_warn(world, monkeypatch):
    world.pin = SimpleNamespace(version="4.9.0", build_sha="b" * 40)
    assert restart.restart_preview()["warnings"], "the lower pin warns while it selects the next start"
    monkeypatch.setenv("OUROBOROS_CLAUDEXOR_BIN", "/opt/claudexord")
    preview = restart.restart_preview()
    assert preview["warnings"] == [] and preview["engine"]["pin"] is None


def test_an_uncounted_engine_is_unknown_never_zero(world):
    delegate(world.root, "run-own")
    world.engine.runs = [run("run-own"), run("run-foreign")]
    world.answers = False
    silent = restart.restart_preview()
    assert silent["engine"]["state"] == "not_answering" and silent["engine"]["serving"] is None
    assert silent["runs"] | {"error": None} == {"status": "unknown", "total": None, "ouroboros": 1, "other": None,
                                                "error": None, "items": [], "items_omitted": 0}
    assert silent["runs"]["error"].startswith("daemon_unreachable:")
    assert silent["model_calls"] == {"status": "unknown", "active": None, "queued": None}

    world.answers, world.engine.lists = True, False
    unlisted = restart.restart_preview()
    assert unlisted["engine"]["state"] == "serving" and unlisted["runs"]["status"] == "unknown"
    assert unlisted["runs"]["total"] is None and unlisted["runs"]["error"].startswith("run_list_unsupported:")

    world.provisioned = False  # never started: nothing can end, and nothing is asked
    world.engine.requests.clear()
    fresh = restart.restart_preview()
    assert fresh["engine"]["state"] == "not_started" and fresh["runs"]["status"] == "known"
    assert fresh["runs"]["total"] == 0 and world.engine.requests == []


def test_unreadable_own_records_leave_the_total_and_withhold_the_owners(world, monkeypatch):
    world.engine.runs = [run("r1"), run("r2")]
    assert restart.restart_preview()["runs"]["other"] == 2, "readable records attribute every run"

    def unreadable(_ids):
        raise OSError("store is locked")

    monkeypatch.setattr(restart, "_custody_of", unreadable)
    preview = restart.restart_preview()
    assert preview["runs"] | {"error": None} == {"status": "unattributed", "total": 2, "ouroboros": None, "other": None,
                                                 "error": None, "items": [], "items_omitted": 0}
    assert "store is locked" in preview["runs"]["error"]


# ---- the restart ---------------------------------------------------------------------------

def test_a_stale_preview_stops_nothing_and_the_confirmed_one_restarts(world):
    delegate(world.root, "run-own")
    world.engine.runs = [run("run-own")]
    seen = restart.restart_preview()
    world.engine.runs.append(run("run-foreign"))  # another program started work after the owner looked

    with pytest.raises(restart.RestartRefused) as stale:
        restart.restart_engine(seen["token"])
    assert stale.value.code == "restart_preview_stale" and world.calls == [], "nothing stopped, nothing started"
    fresh = stale.value.preview
    assert (fresh["runs"]["total"], fresh["runs"]["ouroboros"], fresh["runs"]["other"]) == (2, 1, 1)
    assert rows(world.root, "claudexor_owner_restart") == []

    receipt = restart.restart_engine(fresh["token"])
    assert world.calls == ["stop", "owner_restart_claudexor", "start"], "stop, release the latch, then the ordinary start"
    assert receipt == {"stop": "stopped", "serving": True, "error": None,
                       "runs": {"status": "known", "total": 2, "ouroboros": 1, "other": 1}}
    (row,) = rows(world.root, "claudexor_owner_restart")
    assert row["requested_by"] == "owner" and row["stop"] == "stopped" and row["serving"] is True
    assert row["purpose"] == claudexor_daemon.CUSTODY_PURPOSE and row["runs"]["other"] == 1


@pytest.mark.parametrize("stop,start_error,expected", [
    ("stopped", None, {"stop": "stopped", "serving": True, "error": None}),
    ("nothing_to_stop", None, {"stop": "nothing_to_stop", "serving": True, "error": None}),
    ("unconfirmed", None, {"stop": "unconfirmed", "serving": True, "error": None}),
    ("stopped", ClaudexorUnavailable("daemon_spawn_failed", "engine refused the store"),
     {"stop": "stopped", "serving": False, "error": "daemon_spawn_failed: engine refused the store"}),
    (RuntimeError("ledger unreadable"), None, {"stop": "unconfirmed", "serving": True, "error": None}),
])
def test_the_receipt_keeps_the_stop_and_the_next_start_apart(world, stop, start_error, expected):
    world.stop, world.start_error = stop, start_error
    receipt = restart.restart_engine(restart.restart_preview()["token"])
    assert {key: receipt[key] for key in expected} == expected
    assert world.calls == ["stop", "owner_restart_claudexor", "start"], "an unconfirmed stop is no veto"
    disclosed = rows(world.root, "process_stop_unconfirmed")
    # A raising stop is rowed here; an unconfirmed outcome was already rowed by the daemon manager.
    assert [row["reason"] for row in disclosed] == (["stop raised RuntimeError"] if isinstance(stop, Exception) else [])


def test_a_restart_that_could_not_start_again_or_is_already_running_stops_nothing(world, monkeypatch):
    token = restart.restart_preview()["token"]
    monkeypatch.setattr(delegate_recovery, "owner_restart_in_progress", lambda root: True)
    with pytest.raises(ClaudexorUnavailable) as restarting:
        restart.restart_engine(token)
    monkeypatch.setattr(delegate_recovery, "owner_restart_in_progress", lambda root: False)
    monkeypatch.setattr(review_run_isolation, "attach_home", lambda: world.root)
    with pytest.raises(ClaudexorUnavailable) as attached:
        restart.restart_engine(token)
    monkeypatch.setattr(review_run_isolation, "attach_home", lambda: None)
    assert restart._RESTART_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(restart.RestartRefused) as busy:
            restart.restart_engine(token)
    finally:
        restart._RESTART_LOCK.release()
    assert (restarting.value.code, attached.value.code, busy.value.code) == (
        "owner_restart", "attach_only_engine", "restart_in_progress")
    assert world.calls == [], "no refusal reached the stop"

    assert restart.restart_engine(token)["stop"] == "stopped" and world.calls[0] == "stop"


# ---- the route -----------------------------------------------------------------------------

def request(method, body=None):
    async def read_json():
        if body is None:
            raise ValueError("no body")
        return body

    async def read_body():
        return b"" if body is None else json.dumps(body).encode("utf-8")

    return SimpleNamespace(method=method, json=read_json, body=read_body)


def test_the_route_answers_the_receipt_beside_the_status_read_after_it(world, monkeypatch):
    from ouroboros.gateway import claudexor_accounts as accounts

    def status(include_models, **_kw):
        world.calls.append("read")
        return {"daemon": {"state": "running"}, "reads": {"catalog": "ok", "accounts": "ok", "quota": "ok"}}

    monkeypatch.setattr(accounts, "_status_payload", status)
    world.engine.runs = [run("run-foreign")]
    preview = asyncio.run(accounts.api_claudexor_restart(request("GET")))
    assert preview.status_code == 200 and world.calls == [], "the preview stops, starts and reads nothing"
    token = json.loads(preview.body)["token"]

    tokenless = asyncio.run(accounts.api_claudexor_restart(request("POST", {})))
    stale = asyncio.run(accounts.api_claudexor_restart(request("POST", {"token": "what the owner never saw"})))
    assert tokenless.status_code == 400 and stale.status_code == 409 and world.calls == []
    refusal = json.loads(stale.body)
    assert refusal["code"] == "restart_preview_stale" and refusal["preview"]["token"] == token

    done = asyncio.run(accounts.api_claudexor_restart(request("POST", {"token": token})))
    assert done.status_code == 200
    assert world.calls == ["stop", "owner_restart_claudexor", "start", "read"], "the status is read after the start"
    body = json.loads(done.body)
    assert set(body) == {"status", "restart"} and body["status"]["daemon"]["state"] == "running"
    assert body["restart"]["stop"] == "stopped" and body["restart"]["runs"]["other"] == 1

    def unreadable(include_models, **_kw):
        raise ClaudexorUnavailable("daemon_unreachable", "the status read lost the engine")

    monkeypatch.setattr(accounts, "_status_payload", unreadable)
    token = json.loads(asyncio.run(accounts.api_claudexor_restart(request("GET"))).body)["token"]
    lost = asyncio.run(accounts.api_claudexor_restart(request("POST", {"token": token})))
    assert lost.status_code == 200, "a restart that happened is never answered as a refusal"
    body = json.loads(lost.body)
    assert body["restart"]["stop"] == "stopped" and "Claudexor status failed" in body["status"]["error"]
