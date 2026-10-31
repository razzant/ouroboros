"""Old published acceptance panels keep their newest canonical form on every read.

Nothing publishes a task-acceptance panel any more; the result review replaced that
machine. Panels stored before the fold still render, so their read side stays: a
stale child read or copyback never replaces the newer canonical panel, and stored
panels of two task attempts are never deduplicated into one. ``_context``, ``_run``
and ``_source`` describe such an old record for the suites that read one.
"""

import hashlib
import json
import queue
from types import SimpleNamespace

from ouroboros import artifacts
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
    assert len(raw) == ref["size"]
    return json.loads(raw)


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
    """Each attempt's stored panel stays its own row, also when a delayed snapshot
    of the first attempt is written again after the second attempt's panel."""
    panel = {"surface": "task_acceptance", "panel_id": "panel_exact", "authority": "host_root",
             "aggregate_signal": "PASS", "publication_revision": 1}
    old = {"panels": [{**panel, "task_attempt": 1}], "publication_revision": 1, "task_attempt": 1}
    write_task_result(tmp_path, "applied", "running", review_projection=old)
    write_task_result(tmp_path, "applied", "running", review_projection={
        "panels": [{**panel, "task_attempt": 2}], "publication_revision": 1, "task_attempt": 2})
    write_task_result(tmp_path, "applied", "running", review_projection=old)
    rows = load_task_result(tmp_path, "applied")["review_projection"]["panels"]
    assert [p["task_attempt"] for p in rows] == [1, 2]
