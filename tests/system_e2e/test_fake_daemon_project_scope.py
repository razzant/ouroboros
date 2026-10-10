"""Default lane: project scope on the wire, real tool and real gateway against the fake engine.

The engine refuses a run on an unregistered project root, accepts ``scope.ephemeral``
without any registration, and answers ``POST /v2/projects`` with ``created``. A root
the host minted therefore starts with no ``/v2/projects`` request at all, and a user
root registers with one POST once the engine's answer carries ``created``.
"""

from __future__ import annotations

import json

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_registration_policy as policy
from ouroboros.gateways import claudexor as cx
from tests._delegated_transport_shared import (  # noqa: F401 - autouse transport binding
    _delegating_ctx,
    _owned_gateway_uses_each_test_transport,
)
from tests.system_e2e.interfaces import FakeClaudexorDaemon


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "_CREATED_REPORTED", {}, raising=False)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    with FakeClaudexorDaemon(runs_dir=tmp_path / "runs", reports_created=True,
                             require_registration=True) as daemon:
        daemon.install(tmp_path / "cx")
        real = cx.ClaudexorGateway

        def handshaken(*_args, **_kwargs):
            gateway = real(cx.discover_daemon_at(tmp_path / "cx"))
            gateway.handshake()
            return gateway

        monkeypatch.setattr(cx, "ClaudexorGateway", handshaken)
        monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", f"{daemon.harness_id}=mock-model")
        yield daemon


def _start(ctx):
    from ouroboros.tools import delegate

    payload = json.loads(delegate._delegate_start(ctx, "survey the tree").text)
    assert payload["status"] == "started", payload
    return custody.replay(custody.custody_root(ctx))[payload["run_id"]]


def test_the_fake_engine_refuses_an_unregistered_root_unless_it_is_ephemeral(engine, tmp_path):
    request = {"prompt": "p", "instructions": "i", "authPreference": "subscription", "mode": "ask",
               "access": "readonly", "harnesses": [engine.harness_id], "primaryHarness": engine.harness_id}
    with cx.ClaudexorGateway() as gateway:
        with pytest.raises(cx.ClaudexorUnavailable) as refused:
            gateway.start_run({**request, "scope": {"kind": "project", "root": str(tmp_path)}},
                              idempotency_key="k-plain")
        assert (refused.value.status_code, refused.value.code) == (404, "project_not_registered")
        handle = gateway.start_run({**request, "scope": {"kind": "project", "root": str(tmp_path),
                                                         "ephemeral": True}}, idempotency_key="k-ephemeral")
    assert handle["runId"] and engine.calls("POST", "/v2/projects") == []


def test_a_registry_checkout_starts_ephemeral_with_no_project_request(engine, tmp_path):
    from ouroboros.subagent_worktrees import _registry_path

    ctx = _delegating_ctx(tmp_path, acting=False, task_id="t-minted")
    _registry_path().parent.mkdir(parents=True, exist_ok=True)
    _registry_path().write_text(json.dumps({"worktrees": [{"path": str(ctx.repo_dir)}]}), encoding="utf-8")
    row = _start(ctx)
    [start] = engine.run_start_posts()
    assert start["body"]["scope"] == {"kind": "project", "root": str(ctx.repo_dir), "ephemeral": True}
    assert engine.calls(path_prefix="/v2/projects") == []
    assert (row.project_id, row.project_owned, row.project_persistent) == ("", False, False)


def test_a_user_root_registers_with_the_engine_created_answer(engine, tmp_path):
    first = _start(_delegating_ctx(tmp_path, acting=False, task_id="t-user-1"))
    lists = engine.calls("GET", "/v2/projects")
    posts = engine.calls("POST", "/v2/projects")
    # The first start on an unobserved engine runs the previous flow and learns the field.
    assert len(lists) == 1 and len(posts) == 1 and first.project_owned and first.project_id
    second = _start(_delegating_ctx(tmp_path, acting=False, task_id="t-user-2"))
    assert len(engine.calls("GET", "/v2/projects")) == 1, "no project list once created is known"
    assert len(engine.calls("POST", "/v2/projects")) == 2
    assert (second.project_id, second.project_owned) == (first.project_id, False)
    for start in engine.run_start_posts():
        assert "ephemeral" not in start["body"]["scope"], "a user root keeps its registration"
