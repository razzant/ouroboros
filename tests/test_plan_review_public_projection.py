"""Public task results expose the canonical read-only Plan Review v1 projection."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros.gateway.tasks import api_task_get
from ouroboros.outcomes import public_task_result
from ouroboros.utils import atomic_write_json, read_json_dict


def _legacy(kind: str) -> dict:
    fingerprint = "a" * 64
    state = {
        "schema_version": 1,
        "current_attempt": {},
        "latest_review_fingerprint": "",
        "waves": [],
    }
    if kind == "pending":
        state["waves"] = [{"request_fingerprint": fingerprint, "phase": "collected"}]
        return state
    state["current_attempt"] = {
        "fingerprint": fingerprint,
        "status": "rail_degraded" if kind == "rail_degraded" else (
            "unavailable" if kind == "unavailable" else "open"
        ),
        "reason": "deadline" if kind == "rail_degraded" else (
            "reviewer unavailable" if kind == "unavailable" else ""
        ),
    }
    state["latest_review_fingerprint"] = fingerprint
    if kind != "unavailable":
        closed = kind == "closed"
        state["waves"] = [{
            "request_fingerprint": fingerprint,
            "phase": "reviewed",
            "review_evidence_status": "integrated",
            "review": {
                "aggregate_signal": "GREEN" if closed else "REVIEW_REQUIRED",
                "closed": closed,
            },
        }]
    return state


@pytest.mark.parametrize(
    ("kind", "status", "outcome", "closed"),
    [
        ("open", "open", "REVIEW_REQUIRED", False),
        ("pending", "pending", "", False),
        ("unavailable", "open", "", False),
        ("rail_degraded", "rail_degraded", "REVIEW_REQUIRED", False),
        ("closed", "closed", "GREEN", True),
    ],
)
def test_public_result_adds_legacy_projection_without_mutating_authority(
    kind, status, outcome, closed,
):
    raw = _legacy(kind)
    before = copy.deepcopy(raw)
    public = public_task_result({"task_id": "legacy", "plan_review_state": raw})
    assert raw == before
    assert public["plan_review_state"]["schema_version"] == 1
    projection = public["plan_review_state"]["legacy_v1_projection"]
    assert (projection["status"], projection["outcome"], projection["closed"]) == (
        status, outcome, closed,
    )


def test_task_detail_projects_legacy_state_without_rewriting_disk(tmp_path):
    task_id = "legacy-detail"
    path = tmp_path / "task_results" / f"{task_id}.json"
    path.parent.mkdir(parents=True)
    stored = {"_schema_version": 1, "task_id": task_id, "status": "running", "plan_review_state": _legacy("open")}
    atomic_write_json(path, stored)
    request = SimpleNamespace(
        path_params={"task_id": task_id},
        app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)),
    )
    payload = json.loads(asyncio.run(api_task_get(request)).body.decode("utf-8"))
    assert payload["plan_review_state"]["legacy_v1_projection"]["outcome"] == "REVIEW_REQUIRED"
    assert read_json_dict(path) == stored
