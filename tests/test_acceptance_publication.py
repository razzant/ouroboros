"""Full applied acceptance custody and monotonically published read models."""

import copy
import hashlib
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import artifacts, loop, review_projection
from ouroboros.gateway.tasks import api_task_artifact
from ouroboros.review_substrate import ReviewRunResult
from ouroboros.task_results import load_task_result, write_task_result


def _context(root):
    return SimpleNamespace(task_id="applied", task_attempt=1, drive_root=root,
                           task_metadata={}, event_queue=queue.Queue(), current_chat_id=1)


def _run(task_attempt=1):
    findings = [{"id": f"f{i}", "severity": "low", "item": f"full finding {i}",
                 "evidence": "complete context", "recommendation": "consider later"} for i in range(80)]
    return {"request": {"surface": "task_acceptance", "task_id": "applied"},
            "panel_id": "panel_exact", "authority": "host_root", "task_attempt": task_attempt, "aggregate_signal": "PASS",
            "actors": [{"slot_id": "s1", "signal": "PASS", "status": "ok",
                        "parsed": {"verdict": "PASS", "outcome_tier": "solved", "findings": findings},
                        "criteria_refs_unresolved": [{"criterion": "complete", "supported_evidence_resolves": True}]}],
            "parsed_findings": findings, "enforcement_impact": "allows_completion",
            "dialogue": {"status": "inconclusive"}}


def _source(root, panel):
    ref = panel["applied_source_ref"]
    path = artifacts.task_artifact_dir_path(root, "applied") / ref["path"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
    assert len(raw) == ref["bytes"]
    return json.loads(raw)


def test_full_applied_source_downloads_while_task_is_running(tmp_path):
    ctx = _context(tmp_path)
    write_task_result(tmp_path, "applied", "running", result="working", independent={"keep": True})
    trace = {"review_runs": [_run()]}
    loop._set_acceptance_decision(trace, {"status": "accepted", "reason": "clean_pass"})
    review_projection.publish_acceptance_checkpoint(ctx, trace)
    stored = load_task_result(tmp_path, "applied")
    panel = stored["review_projection"]["panels"][0]
    assert stored["status"] == "running"
    assert stored["result"] == "working" and stored["independent"] == {"keep": True}
    assert "artifacts" not in stored, "the live publisher must not replace another writer's artifact list"
    full = _source(tmp_path, panel)
    assert full["applied_decision"]["status"] == "accepted"
    assert full["enforcement_impact"] == "allows_completion"
    assert full["actors"][0]["criteria_refs_unresolved"][0]["supported_evidence_resolves"] is True
    assert len(full["actors"][0]["parsed"]["findings"]) == 80
    assert len(panel["actors"][0]["findings"]) < 80
    # Real endpoint path and its normal effective-task materialization. Nothing
    # ran terminal collection, and no path-only bypass grants artifact access.
    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", api_task_artifact)])
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        response = client.get(f"/api/tasks/applied/artifacts/{panel['applied_source_ref']['path']}")
        assert response.status_code == 200
        assert response.json() == full
        assert client.get("/api/tasks/applied/artifacts/missing.json").status_code == 404
    envelope = ctx.event_queue.get_nowait()
    assert envelope["type"] == "log_event"
    event = envelope["data"]
    assert event["type"] == "review_reference" and event["surface"] == "task_acceptance"
    assert event["presentation_owner_task_id"] == "applied"
    assert len(event["state_revision"]) == 64


def test_delayed_publication_cannot_replace_supersession_or_terminal_fields(tmp_path, monkeypatch):
    ctx = _context(tmp_path)
    trace = {"review_runs": [_run()]}
    reached, release = threading.Event(), threading.Event()
    actual_store = artifacts.store_task_artifact_bytes
    first_thread = []

    def delayed_store(*args, **kwargs):
        if not first_thread:
            first_thread.append(threading.get_ident())
            reached.set()
            assert release.wait(10)
        return actual_store(*args, **kwargs)

    monkeypatch.setattr(artifacts, "store_task_artifact_bytes", delayed_store)
    with ThreadPoolExecutor(max_workers=1) as executor:
        older = executor.submit(review_projection.publish_acceptance_checkpoint, ctx, trace)
        try:
            assert reached.wait(10)
            trace["review_runs"][0].update(superseded_by_revision=True, superseded_reason="owner_followup",
                                           enforcement_impact="requires_revision")
            loop._set_acceptance_decision(trace, {"status": "revision_requested", "reason": "owner_followup"})
            write_task_result(tmp_path, "applied", "completed", result="delivered", independent={"keep": True},
                              accounted_upper_bound_usd=7.25, cost_final=True)
            review_projection.publish_acceptance_checkpoint(ctx, trace)
        finally:
            release.set()
        older.result(timeout=10)
    stored = load_task_result(tmp_path, "applied")
    assert stored["status"] == "completed" and stored["result"] == "delivered"
    assert stored["independent"] == {"keep": True} and stored["accounted_upper_bound_usd"] == 7.25
    panels = stored["review_projection"]["panels"]
    assert len(panels) == 1 and panels[0]["publication_revision"] == 2
    assert panels[0]["superseded"] is True
    assert panels[0]["applied_source_ref"] == trace["review_runs"][0]["applied_source_ref"]
    assert _source(tmp_path, panels[0])["applied_decision"]["reason"] == "owner_followup"


def test_stale_child_read_and_copyback_keep_newest_canonical_panel(tmp_path):
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_status import load_effective_task_result

    canonical, child = tmp_path / "canonical", tmp_path / "child"
    old = {"surface": "task_acceptance", "panel_id": "p", "task_attempt": 1,
           "publication_revision": 1, "superseded": False}
    new = {**old, "publication_revision": 2, "superseded": True}
    write_task_result(canonical, "applied", "completed", child_drive_root=str(child),
                      root_phase_checkpoint={"post_task_synthesis": "completed"},
                      review_projection={"panels": [new]}, accounted_upper_bound_usd=7, cost_final=True)
    write_task_result(child, "applied", "completed", review_projection={"panels": [old]},
                      accounted_upper_bound_usd=2, cost_final=False, result="replica result")
    read = load_effective_task_result(canonical, "applied", materialize_artifacts=False)
    assert read["review_projection"]["panels"] == [new]
    copied = copy_child_task_result(canonical, {"id": "applied", "drive_root": str(child)})
    assert copied["review_projection"]["panels"] == [new]
    assert copied["accounted_upper_bound_usd"] == 7 and copied["cost_final"] is True
    assert copied["result"] == "replica result"


def test_new_task_attempt_is_not_deduplicated_with_previous_attempt(tmp_path):
    ctx = _context(tmp_path)
    first = {"review_runs": [_run()]}
    review_projection.publish_acceptance_checkpoint(ctx, first)
    old = copy.deepcopy(load_task_result(tmp_path, "applied")["review_projection"])
    ctx.task_attempt = 2
    review_projection.publish_acceptance_checkpoint(ctx, {"review_runs": [_run(task_attempt=2)]})
    write_task_result(tmp_path, "applied", "running", review_projection=old)
    rows = load_task_result(tmp_path, "applied")["review_projection"]["panels"]
    assert [p["task_attempt"] for p in rows] == [1, 2]


def test_publishing_legacy_run_does_not_invent_its_task_attempt(tmp_path):
    ctx, run = _context(tmp_path), _run()
    run.pop("task_attempt")
    ctx.task_attempt = 2
    review_projection.publish_acceptance_checkpoint(ctx, {"review_runs": [run]})
    panel = load_task_result(tmp_path, "applied")["review_projection"]["panels"][0]
    assert "task_attempt" not in panel


def test_source_failure_discloses_unavailable_without_changing_verdict(tmp_path, monkeypatch):
    ctx, trace = _context(tmp_path), {"review_runs": [_run()]}
    monkeypatch.setattr(artifacts, "store_task_artifact_bytes", lambda *a, **k: (_ for _ in ()).throw(OSError("disk unavailable")))
    review_projection.publish_acceptance_checkpoint(ctx, trace)
    panel = load_task_result(tmp_path, "applied")["review_projection"]["panels"][0]
    assert panel["aggregate_signal"] == "PASS"
    assert panel["applied_source_status"] == "unavailable" and "applied_source_ref" not in panel


@pytest.mark.parametrize("apply_failure", [False, True])
def test_host_application_publishes_full_decision_before_finalization(tmp_path, monkeypatch, apply_failure):
    from ouroboros.contracts.task_contract import build_task_contract
    import ouroboros.review_substrate as substrate

    ctx = _context(tmp_path)
    ctx.task_contract = build_task_contract({"id": "applied", "root_task_id": "applied"})
    ctx._task_acceptance_reviewed = False
    ctx.is_direct_chat = False
    write_task_result(tmp_path, "applied", "running", task_contract=ctx.task_contract)
    monkeypatch.setattr(loop, "get_task_review_mode", lambda: "required")
    monkeypatch.setattr(substrate, "triad_delivery_slots", lambda **kw: [])
    parsed = {"verdict": "PASS", "outcome_tier": "solved", "completion_coach": "ship",
              "criteria_used": [{"criterion": "requested file exists", "status": "supported",
                                 "evidence_refs": ["artifacts"]}]}
    result = ReviewRunResult(request={"surface": "task_acceptance"}, aggregate_signal="PASS",
                             actors=[{"signal": "PASS", "slot_id": "s1", "status": "ok", "parsed": parsed}], parsed_findings=[])
    monkeypatch.setattr(loop, "_execute_task_acceptance_panel", lambda context: result)
    if apply_failure:
        def fail_apply(*args, **kwargs):
            raise RuntimeError("host application failed after receiving the panel")
        monkeypatch.setattr("ouroboros.loop_acceptance_review._apply_task_acceptance_result", fail_apply)
    trace = {"tool_calls": []}
    again = loop._run_task_acceptance_review_once(
        tools=SimpleNamespace(_ctx=ctx), content="The requested file is ready.", task_id="applied", task_type="task",
        llm_trace=trace, drive_root=tmp_path, messages=[], emit_progress=lambda *_a, **_k: None,
    )
    assert again is False
    assert trace["acceptance_decision"]["status"] == ("finalized_unaccepted" if apply_failure else "accepted")
    # Publishing again must preserve both the returned panel and its separate
    # host application failure row, despite their shared binding-derived id.
    review_projection.publish_acceptance_checkpoint(ctx, trace)
    saved = load_task_result(tmp_path, "applied")
    assert saved["status"] == "running"
    panels = saved["review_projection"]["panels"]
    assert len(panels) == (2 if apply_failure else 1)
    full = _source(tmp_path, panels[-1])
    assert full["applied_decision"] == trace["acceptance_decision"]
    assert full["enforcement_impact"] == ("degrades_completion" if apply_failure else "allows_completion")
    assert full["task_attempt"] == ctx.task_attempt
    if apply_failure:
        assert panels[0]["panel_id"] == panels[1]["panel_id"]
        assert [row["panel_index"] for row in panels] == [0, 1]
        assert _source(tmp_path, panels[0])["actors"][0]["parsed"] == parsed
        assert "host application failed" in full["degraded_reasons"][0]
