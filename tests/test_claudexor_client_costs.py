"""Claudexor C2, the Ouroboros side: the static live-input facet is read once per engine build,
and a 404 run read names its caller.

``_leaf_live_input`` asked ``GET /v2/agent-capabilities`` — the engine's full harness doctor,
p50 3.1 s on the 01.11 daemon, 3,123 calls — at every supervised wait for one static bit of one
route. ``declared_live_input`` keeps that bit per exact engine identity (version AND build sha:
the gateway's own handshake, else one this process remembers for the same daemon generation) and
asks again only for a route the catalog did not list; it never spends a handshake. ``GET
/v2/runs/:id`` answered 404 95,697 times from a poller nobody could name: the refusal now carries
``caller_site`` and each distinct site is logged once — diagnostic only, the error is unchanged.
"""
from __future__ import annotations

import logging

import httpx
import pytest

from ouroboros.gateways import claudexor as gateway_module
from ouroboros.gateways import claudexor_metadata, claudexor_run_events


class _Engine:
    def __init__(self, version="5.0.1", sha="build-a", routes=("codex",)):
        self.engine_version, self.engine_build_sha, self.reads = version, sha, 0
        self.routes = routes

    def handshake(self, **_kwargs):
        raise AssertionError("no handshake is spent on a capability facet")

    def agent_capabilities(self, *, timeout_sec=None):
        self.reads += 1
        return {"harnesses": [{"id": route, "liveInput": "mid_turn"} for route in self.routes]
                + [{"id": "plain"}]}


@pytest.fixture(autouse=True)
def _fresh_memo(monkeypatch):
    monkeypatch.setattr(claudexor_run_events, "_LIVE_INPUT", {})


def test_the_live_input_facet_is_read_once_per_exact_engine_build():
    engine = _Engine()
    assert claudexor_run_events.declared_live_input(engine, "codex", timeout_sec=3) == "mid_turn"
    assert claudexor_run_events.declared_live_input(engine, "codex", timeout_sec=3) == "mid_turn"
    assert claudexor_run_events.declared_live_input(engine, "plain", timeout_sec=3) == "none"
    assert engine.reads == 1, "one catalog read serves every later question of this build"

    rebuilt = _Engine(sha="build-b")
    assert claudexor_run_events.declared_live_input(rebuilt, "codex", timeout_sec=3) == "mid_turn"
    assert rebuilt.reads == 1, "another build is asked again"


def test_an_unlisted_route_and_an_unknown_identity_are_asked_again_every_time(monkeypatch):
    engine = _Engine()
    for _ in range(2):
        assert claudexor_run_events.declared_live_input(engine, "absent-harness", timeout_sec=3) is None
    assert engine.reads == 2, "a harness's availability is not static: never remembered as absent"

    monkeypatch.setattr(claudexor_metadata, "remembered_engine", lambda _gateway: None)
    unproven = _Engine(version="", sha="")
    for _ in range(2):
        assert claudexor_run_events.declared_live_input(unproven, "codex", timeout_sec=3) == "mid_turn"
    assert unproven.reads == 2, "without an exact identity nothing is remembered (and no handshake spent)"


def test_a_not_yet_handshaken_gateway_uses_the_handshake_its_process_remembers(monkeypatch):
    claudexor_run_events.declared_live_input(_Engine(), "codex", timeout_sec=3)
    fresh = _Engine(version="", sha="")
    monkeypatch.setattr(claudexor_metadata, "remembered_engine", lambda gateway: ("5.0.1", "build-a"))
    assert claudexor_run_events.declared_live_input(fresh, "codex", timeout_sec=3) == "mid_turn"
    assert fresh.reads == 0


def _gateway(handler):
    gateway = gateway_module.ClaudexorGateway(gateway_module.DaemonEndpoint("127.0.0.1", 1, "fixture"))
    gateway._client.close()
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(handler))
    return gateway


def test_a_404_run_read_names_its_caller_once_and_stays_the_same_refusal(monkeypatch, caplog):
    from ouroboros.gateways import claudexor_observation

    monkeypatch.setattr(claudexor_observation, "_ABSENT_RUN_SITES", set())
    gateway = _gateway(lambda request: httpx.Response(404, json={"code": "run_not_found", "message": "no run"}))

    def poller():
        return gateway.get_run("run-gone")

    errors = []
    with caplog.at_level(logging.INFO, logger=claudexor_observation.log.name):
        for _ in range(3):
            with pytest.raises(gateway_module.ClaudexorUnavailable) as caught:
                poller()
            errors.append(caught.value)
    assert {(error.code, error.status_code) for error in errors} == {("run_not_found", 404)}
    assert errors[0].caller_site.startswith(f"{__name__}:poller:"), errors[0].caller_site
    assert sum("GET /v2/runs/:id" in record.getMessage() for record in caplog.records) == 1, "once per site"


def test_other_refusals_and_other_run_routes_carry_no_tag():
    gone = _gateway(lambda request: httpx.Response(404, json={"code": "not_found", "message": "no"}))
    with pytest.raises(gateway_module.ClaudexorUnavailable) as caught:
        gone._request("GET", "/v2/runs/run-x/artifacts/out.txt")
    assert not hasattr(caught.value, "caller_site")
    busy = _gateway(lambda request: httpx.Response(503, json={"code": "daemon_busy", "message": "busy"}))
    with pytest.raises(gateway_module.ClaudexorUnavailable) as caught:
        busy.get_run("run-x")
    assert not hasattr(caught.value, "caller_site")
