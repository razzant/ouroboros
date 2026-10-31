"""An omitted ``access`` on ``continue_from`` keeps the continued run's own access.

A parent that lowered a run to ``readonly`` (once ``workspace_write``) and later continued it
without naming access got the row's captured ``full``, and the continuation floor refused its own
start (``continuation_authority_widened``); every such task spent a round repeating the call with
the access spelled out. Omission now inherits the continued run's recorded access, never wider than
today's captured profile; an explicit wider choice is still refused and ``retry_of`` still replays
its recorded body unchanged.
"""

from __future__ import annotations

import json

import pytest

from ouroboros import delegate_continuation as continuation
from ouroboros import delegate_custody as custody
from tests._delegated_transport_shared import (  # noqa: F401 -- autouse transport fixture
    _delegating_ctx,
    _owned_gateway_uses_each_test_transport,
    _transport_snapshot,
)

TASK = "t-nanny"


class _Engine:
    """A serving engine that continues runs and offers every access profile."""

    engine_version = "3.22.0"

    def __init__(self, run_id, calls):
        self.run_id, self.calls = run_id, calls

    def handshake(self, **_kw): return {}
    def agent_capabilities(self, **_kw):
        return {"runControlKeys": ["continueFrom"], "harnesses": [{
            "id": "some-route", "enabled": True, "status": "ok",
            "accessProfilesSupported": ["readonly", "workspace_write", "full"]}]}
    def quota_snapshots(self): return []
    def find_project_id(self, root): return "prj-existing"
    def register_project(self, root): return "prj-new"
    def remove_project(self, project_id): return {}
    def ensure_full_access(self, root): return {}
    def start_run(self, request, *, idempotency_key=""):
        self.calls.append(request)
        return {"runId": self.run_id, "runDir": f"/tmp/{self.run_id}"}
    def close(self): pass


def _start(tmp_path, monkeypatch, *, run_id, captured, access=None, **spec):
    """One ``exact_start`` the way the tool reaches it: the selected row captured ``captured``,
    ``access`` passed only when the caller names it. Returns (request sent or None, payload)."""
    import ouroboros.tools.delegate as delegate
    from ouroboros import subagent_runtime, subagents
    from ouroboros.gateways import claudexor as gw

    calls = []
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snap_root"))
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Engine(run_id, calls))
    monkeypatch.setattr(delegate, "prepare_delegate_start_actor", subagent_runtime.prepare_delegate_start_actor)
    delegate._CUSTODY.clear()
    ctx = _delegating_ctx(tmp_path, acting=True, task_id=TASK)
    spec["snapshot"] = _transport_snapshot(subagents.get_subagent_harness(), captured)
    if access is not None:
        spec["access"] = access
    payload = json.loads(subagent_runtime.exact_start(ctx, "finish the work", spec).text)
    delegate._CUSTODY.clear()
    return (calls[-1] if calls else None), payload


def _settle(tmp_path, run_id):
    assert custody.emit(tmp_path, custody.SETTLED, {"run_id": run_id, "task_id": TASK, "route": "some-route",
                                                    "state": "failed", "outcome_reason": "subscription_window_exhausted"})
    custody._CUSTODY.clear()


def test_a_read_only_run_continues_read_only_without_naming_access(tmp_path, monkeypatch):
    first, payload = _start(tmp_path, monkeypatch, run_id="run-read", captured="full", access="readonly")
    assert payload["status"] == "started" and first["access"] == "readonly"
    _settle(tmp_path, "run-read")

    # The caller's own wider choice is still refused, typed and before anything is sent.
    request, refused = _start(tmp_path, monkeypatch, run_id="run-wide", captured="full", access="workspace_write",
                              continue_from="run-read")
    assert request is None and refused["reason"] == continuation.REFUSAL_AUTHORITY_WIDENED
    assert refused["definitely_unrun"] is True and refused["detail"].endswith("omit access to keep the run's own.")
    # Omission keeps the run's own access: the start that used to refuse itself now runs.
    request, payload = _start(tmp_path, monkeypatch, run_id="run-next", captured="full", continue_from="run-read")
    assert payload["status"] == "started", payload
    assert request["continueFrom"] == "run-read" and request["access"] == "readonly" and request["mode"] == "ask"
    assert payload["access"] == "readonly" and payload["continuation"]["prior_access"] == "readonly"
    assert custody.replay(tmp_path)["run-next"].access == "readonly"


def test_a_lowered_writer_stays_lowered_in_its_own_snapshot(tmp_path, monkeypatch):
    first, _payload = _start(tmp_path, monkeypatch, run_id="run-write", captured="full", access="workspace_write")
    assert first["access"] == "workspace_write"
    _settle(tmp_path, "run-write")

    request, payload = _start(tmp_path, monkeypatch, run_id="run-next", captured="full", continue_from="run-write")
    assert payload["status"] == "started", payload
    assert request["access"] == "workspace_write" and payload["continuation"]["workspace"] == "same_snapshot"
    assert request["execution"]["workspaceRoot"] == first["execution"]["workspaceRoot"]


@pytest.mark.parametrize("today", ["workspace_write", "readonly"])
def test_authority_lowered_since_the_run_is_never_widened_back(tmp_path, monkeypatch, today):
    first, _payload = _start(tmp_path, monkeypatch, run_id="run-full", captured="full")
    assert first["access"] == "full"
    _settle(tmp_path, "run-full")

    # The owner has since lowered the configured row: today's narrower profile wins.
    request, payload = _start(tmp_path, monkeypatch, run_id="run-next", captured=today, continue_from="run-full")
    assert payload["status"] == "started", payload
    assert request["access"] == today and payload["continuation"]["prior_access"] == "full"


def test_a_configured_session_continuation_keeps_its_runs_access_too(tmp_path, monkeypatch):
    """The actor-first path: the frozen snapshot is the row captured at dispatch."""
    from ouroboros import subagent_runtime, subagents

    _start(tmp_path, monkeypatch, run_id="run-read", captured="full", access="readonly")
    _settle(tmp_path, "run-read")
    calls = []
    from ouroboros.gateways import claudexor as gw

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Engine("run-next", calls))
    ctx = _delegating_ctx(tmp_path, acting=True, task_id=TASK)
    ctx._configured_actor_bootstrap = {
        "selected_subagent_id": "transport-fixture",
        "snapshot": _transport_snapshot(subagents.get_subagent_harness(), "full"),
        "canonical_work_order": "THE CANONICAL WORK ORDER", "work_order_fingerprint": "wo-1"}
    payload = json.loads(subagent_runtime.delegate_start_entry(
        ctx, "the coordination note", continue_from="run-read").text)
    assert payload["status"] == "started", payload
    assert calls[-1]["access"] == "readonly" and calls[-1]["prompt"].endswith("the coordination note")


def test_retry_of_replays_its_recorded_body_without_consulting_the_continued_run(tmp_path, monkeypatch):
    from ouroboros import subagent_runtime

    def unexpected(*_a, **_k):
        raise AssertionError("a retry replays its recorded access")

    monkeypatch.setattr(continuation, "inherited_access", unexpected)
    ctx = _delegating_ctx(tmp_path, acting=True, task_id=TASK)
    payload = json.loads(subagent_runtime.exact_start(ctx, "", {"retry_of": "no-such-invocation"}).text)
    assert payload["status"] == "refused" and payload["reason"] == "unknown_invocation"


def test_an_unrecorded_access_inherits_nothing_and_is_not_told_to_omit_it(tmp_path):
    from tests.test_delegate_continuation import _gate, _seed

    _seed(tmp_path, "run-legacy", access="", mode="")
    ctx = _delegating_ctx(tmp_path, acting=True, task_id="t-a")
    assert continuation.inherited_access(ctx, "run-legacy") is None
    _facts, code, detail = _gate(tmp_path, "run-legacy")
    assert code == continuation.REFUSAL_AUTHORITY_WIDENED and "omit access" not in detail


def test_unknown_custody_inherits_nothing_and_the_floor_still_answers(tmp_path, monkeypatch):
    request, payload = _start(tmp_path, monkeypatch, run_id="run-next", captured="full", continue_from="run-ghost")
    assert request is None and payload["reason"] == continuation.REFUSAL_SOURCE_UNKNOWN
    assert continuation.inherited_access(_delegating_ctx(tmp_path, acting=True, task_id=TASK), "run-ghost") is None
