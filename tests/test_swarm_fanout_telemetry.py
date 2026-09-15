"""WS8: swarm_fanout telemetry shape + reject-meta marker."""
from __future__ import annotations

import hashlib
import json
import types

from ouroboros.tools.control import _emit_swarm_fanout, maybe_emit_delegated_run_fanout
from supervisor.events import _subagent_rejection_meta, _subagent_scheduled_meta


def test_swarm_fanout_event_shape(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    ctx = types.SimpleNamespace(drive_logs=lambda: logs, _last_fanout_ts=0.0)
    _emit_swarm_fanout(
        ctx,
        parent_task_id="p1",
        root_task_id="r1",
        depth=2,
        task_group_id="subagents-x",
        task_ids=["a", "b"],
        role="researcher",
        requested_model_lane="auto",
        objective="o" * 300,
        emitted_live=True,
    )
    lines = (logs / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    evt = json.loads(lines[0])
    assert evt["type"] == "swarm_fanout"
    # Must not be foldable into a grouped-task lane or rendered as a subagent card.
    assert not evt["type"].startswith(("task_", "llm_", "tool_"))
    assert "delegation_role" not in evt and "subagent_task_id" not in evt
    assert evt["requested_count"] == 2 and evt["task_ids"] == ["a", "b"]
    assert evt["slot_count"] == 2
    # A wave event is written BEFORE any child starts, so it names the lane that was
    # asked for and never one that was resolved (v6.87.28).
    assert evt["requested_model_lane"] == "auto"
    assert "effective_model_lanes" not in evt
    assert len(evt["objective_preview"]) == 200
    assert evt["fanout_interval_sec"] is None  # first wave (prev ts was 0)


def test_swarm_fanout_interval_on_second_fanout(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    ctx = types.SimpleNamespace(drive_logs=lambda: logs, _last_fanout_ts=0.0)
    for _ in range(2):
        _emit_swarm_fanout(
            ctx, parent_task_id="p", root_task_id="r", depth=1,
            task_group_id="", task_ids=["x"], role="r",
            requested_model_lane="auto", objective="o", emitted_live=False,
        )
    evts = [json.loads(line) for line in (logs / "events.jsonl").read_text().splitlines()]
    assert len(evts) == 2
    assert evts[0]["fanout_interval_sec"] is None
    assert isinstance(evts[1]["fanout_interval_sec"], float)


# --- delegated harness runs fold into swarm telemetry ONLY under Swarm intent ---


def _delegating_host_ctx(tmp_path, metadata):
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    return types.SimpleNamespace(
        drive_logs=lambda: logs, _last_fanout_ts=0.0,
        task_id="t-host", task_depth=2, task_metadata=metadata,
    )


def _fanout_events(tmp_path):
    path = tmp_path / "logs" / "events.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("type") == "swarm_fanout"]


def test_delegated_run_fanout_emits_exact_wave_shape_under_swarm_intent(tmp_path):
    ctx = _delegating_host_ctx(
        tmp_path, {"force_plan_source": "swarm", "root_task_id": "r-root"},
    )
    maybe_emit_delegated_run_fanout(
        ctx, run_id="run-1", route_id="codex-route", objective="o" * 300, durable=True,
    )
    evts = _fanout_events(tmp_path)
    assert len(evts) == 1
    evt = evts[0]
    assert evt["type"] == "swarm_fanout"
    assert evt["parent_task_id"] == "t-host" and evt["task_id"] == "t-host"
    assert evt["root_task_id"] == "r-root"
    assert evt["depth"] == 3
    assert evt["requested_count"] == 1 and evt["slot_count"] == 1
    assert evt["task_ids"] == ["run-1"]
    assert evt["role"] == "delegated_run"
    # The REQUESTED lane is the selected session route; nothing has resolved yet.
    assert evt["requested_model_lane"] == "codex-route"
    assert len(evt["objective_preview"]) == 200
    assert evt["emitted_live"] is True
    # Same non-foldable envelope as every wave event (no phantom child card).
    assert "delegation_role" not in evt and "subagent_task_id" not in evt


def test_delegated_run_fanout_defaults_root_to_host_task(tmp_path):
    ctx = _delegating_host_ctx(tmp_path, {"force_plan_source": "swarm"})
    maybe_emit_delegated_run_fanout(
        ctx, run_id="run-2", route_id="r", objective="o", durable=True,
    )
    assert _fanout_events(tmp_path)[0]["root_task_id"] == "t-host"


def test_delegated_run_fanout_silent_without_swarm_intent(tmp_path):
    # An ordinary delegate_start on a task that never asked for a swarm emits
    # nothing — including under the non-swarm force_plan variants.
    for metadata in ({}, {"force_plan_source": "operator"}, {"force_plan": True}):
        ctx = _delegating_host_ctx(tmp_path, metadata)
        maybe_emit_delegated_run_fanout(
            ctx, run_id="run-x", route_id="r", objective="o", durable=True,
        )
    assert _fanout_events(tmp_path) == []


def test_delegated_run_fanout_silent_when_start_uncustodied(tmp_path):
    # started_uncustodied: the custody write failed, so the started fact is not
    # attested — no telemetry.
    ctx = _delegating_host_ctx(tmp_path, {"force_plan_source": "swarm"})
    maybe_emit_delegated_run_fanout(
        ctx, run_id="run-y", route_id="r", objective="o", durable=False,
    )
    assert _fanout_events(tmp_path) == []


def _real_delegate_start(
    tmp_path, monkeypatch, *, metadata, custody_durable=True,
    actor_first_work_order="", coordination_context="",
):
    """Run the REAL _delegate_start against a stubbed gateway with an explicit
    exact-start selection (the test_delegated_subagent_transport idiom) and
    return its parsed payload."""
    import ouroboros.tools.delegate as delegate
    from ouroboros import claudexor_daemon, delegate_custody, subagent_runtime, subagents
    from ouroboros.config import CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools.registry import ToolContext

    class _Stub:
        engine_version = CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION

        def handshake(self, **_kw):
            return {}

        def agent_capabilities(self):
            return {"harnesses": [{
                "id": "some-route", "enabled": True, "status": "ok",
                "accessProfilesSupported": ["readonly", "workspace_write"],
            }]}

        def quota_snapshots(self):
            return []

        def find_project_id(self, root):
            return "prj-existing"

        def register_project(self, root):
            raise AssertionError("must reuse the registration")

        def start_run(self, request, *, idempotency_key=""):
            return {"runId": "run-real", "runDir": "/tmp/run-real"}

        def close(self):
            pass

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: _Stub())
    original_actor = delegate.prepare_delegate_start_actor

    def explicit_transport_actor(ctx, drive_root, **kwargs):
        route = subagents.get_subagent_harness()
        target = route.route_id + (f"={route.model}" if route.model else "")
        token = subagent_runtime._EXACT_START_SELECTION.set({
            "snapshot": {
                "schema": 1, "selected_subagent_id": "transport-fixture",
                "config_fingerprint": "transport-fixture-v1",
                "route": {"kind": "agent_session", "target_id": target,
                          "credential_profile_id": route.profile_id},
                "effort": route.effort,
            },
        })
        try:
            return original_actor(ctx, drive_root, **kwargs)
        finally:
            subagent_runtime._EXACT_START_SELECTION.reset(token)

    if not actor_first_work_order:
        monkeypatch.setattr(delegate, "prepare_delegate_start_actor", explicit_transport_actor)
    if not custody_durable:
        monkeypatch.setattr(delegate, "record_started_custody", lambda *a, **k: False)
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    ctx = ToolContext(
        repo_dir=repo, drive_root=tmp_path,
        task_constraint=TaskConstraint(mode="local_readonly_subagent"),
    )
    ctx.task_id = "t-nanny"
    ctx.task_depth = 1
    ctx.task_metadata = metadata
    delegate_custody._CUSTODY.clear()
    if actor_first_work_order:
        target = "some-route=weak-model:low"
        snapshot = {
            "schema": 1,
            "selected_subagent_id": "transport-fixture",
            "config_fingerprint": "transport-fixture-v1",
            "route": {
                "kind": "agent_session",
                "target_id": target,
                "credential_profile_id": "",
            },
            "effort": "low",
        }
        ctx._configured_actor_bootstrap = {
            "snapshot": snapshot,
            "selected_subagent_id": "transport-fixture",
            "config_fingerprint": "transport-fixture-v1",
            "canonical_work_order": actor_first_work_order,
            "work_order_fingerprint": hashlib.sha256(
                actor_first_work_order.encode("utf-8")
            ).hexdigest(),
            "source_request": {},
            "source_channel": {},
            "exact_start_pending": True,
            "physical_started": False,
            "route_available": True,
        }
        payload = json.loads(
            subagent_runtime.delegate_start_entry(ctx, coordination_context).text
        )
    else:
        payload = json.loads(delegate._delegate_start(ctx, "edit the README").text)
    delegate_custody._CUSTODY.clear()
    return payload


def test_delegate_start_emits_fanout_after_custody_under_swarm_intent(tmp_path, monkeypatch):
    payload = _real_delegate_start(tmp_path, monkeypatch, metadata={
        "root_task_id": "t-root", "parent_task_id": "t-root",
        "force_plan_source": "swarm",
    })
    assert payload["status"] == "started" and payload["custody_durable"] is True
    evts = _fanout_events(tmp_path)
    assert len(evts) == 1
    evt = evts[0]
    assert evt["task_ids"] == [payload["run_id"]]
    assert evt["parent_task_id"] == "t-nanny"
    assert evt["root_task_id"] == "t-root"
    assert evt["requested_count"] == 1
    assert evt["role"] == "delegated_run"
    assert evt["requested_model_lane"] == payload["route"]


def test_delegate_start_without_swarm_intent_stays_out_of_swarm_telemetry(tmp_path, monkeypatch):
    payload = _real_delegate_start(tmp_path, monkeypatch, metadata={
        "root_task_id": "t-root", "parent_task_id": "t-root",
    })
    assert payload["status"] == "started"
    assert _fanout_events(tmp_path) == []


def test_delegate_start_uncustodied_emits_no_fanout_even_under_swarm_intent(tmp_path, monkeypatch):
    payload = _real_delegate_start(
        tmp_path, monkeypatch, custody_durable=False, metadata={
            "root_task_id": "t-root", "parent_task_id": "t-root",
            "force_plan_source": "swarm",
        },
    )
    assert payload["status"] == "started_uncustodied"
    assert _fanout_events(tmp_path) == []


def test_actor_first_swarm_fanout_uses_frozen_work_order_after_custody(
    tmp_path, monkeypatch,
):
    canonical = "CANONICAL_WORK_ORDER:" + ("source-owned " * 30)
    appendix = "COORDINATION_APPENDIX_ONLY"
    payload = _real_delegate_start(
        tmp_path,
        monkeypatch,
        metadata={
            "root_task_id": "t-root",
            "parent_task_id": "t-root",
            "force_plan_source": "swarm",
        },
        actor_first_work_order=canonical,
        coordination_context=appendix,
    )

    assert payload["status"] == "started" and payload["custody_durable"] is True
    events = _fanout_events(tmp_path)
    assert len(events) == 1
    assert events[0]["objective_preview"] == canonical[:200]
    assert appendix not in events[0]["objective_preview"]


def test_actor_first_uncustodied_start_stays_out_of_swarm_telemetry(
    tmp_path, monkeypatch,
):
    payload = _real_delegate_start(
        tmp_path,
        monkeypatch,
        custody_durable=False,
        metadata={"force_plan_source": "swarm"},
        actor_first_work_order="CANONICAL_WORK_ORDER",
        coordination_context="coordination only",
    )

    assert payload["status"] == "started_uncustodied"
    assert _fanout_events(tmp_path) == []


def test_reject_meta_marks_not_accepted():
    meta = _subagent_rejection_meta(
        "t1", root_task_id="r1", parent_id="p1", role="x", status="failed", error="e",
    )
    assert meta.get("accepted") is False


def test_scheduled_meta_marks_accepted():
    meta = _subagent_scheduled_meta(
        tid="t1",
        role="researcher",
        task_constraint={"surface": "external_workspace"},
        task_group_id="g1",
        requested_model_lane="auto",
        active_subagent_count=2,
        max_active_subagents=6,
    )
    assert meta["accepted"] is True
    # An ACCEPTANCE card cannot carry an effective lane or a model: the child has not
    # been dispatched, so nothing has resolved them (v6.87.28).
    assert "effective_model_lane" not in meta and "model" not in meta
    assert meta["active_subagent_count"] == 2
    assert meta["max_active_subagents"] == 6
    assert meta["subagent_event"] == "scheduled"
    assert meta["write_surface"] == "external_workspace"
