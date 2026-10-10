"""Terminal root accounting retains one scoped physical-ledger observation."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from ouroboros import usage_accounting as ua
from ouroboros.gateway.tasks import _task_get_response
from ouroboros.headless import copy_child_task_result
from ouroboros.post_task_checkpoint import (
    project_replica_task_result_fields,
    project_root_post_task_checkpoint_fields,
    set_root_post_task_checkpoint,
)
from ouroboros.task_results import STATUS_COMPLETED, load_task_result, write_task_result
from ouroboros.task_status import load_effective_task_result


@pytest.fixture
def root(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    for key, value in {
        "OUROBOROS_APP_ROOT": tmp_path,
        "OUROBOROS_REPO_DIR": tmp_path / "repo",
        "OUROBOROS_DATA_DIR": data,
        "OUROBOROS_SETTINGS_PATH": data / "settings.json",
    }.items():
        monkeypatch.setenv(key, str(value))
    return data


def _reserve(root, *, task_id="root", root_id="root", bound=0.25, unknown=False):
    return ua.reserve_attempt(ua.AttemptRequest(
        model="openai/test", provider="openai", reservation_usd=bound,
        force_unknown_reservation=unknown, drive_root=root,
        task_id=task_id, root_task_id=root_id, global_limit_usd=100.0,
        root_limit_usd=100.0,
    ))


def _settle(root, cost, *, task_id="root", final=True):
    reservation = _reserve(root, task_id=task_id)
    ua.mark_dispatched(reservation)
    ua.settle_attempt(reservation, {}, cost_usd=cost, cost_final=final)


def _checkpoint(root, status="completed", *, task=None):
    task = task or {"id": "root", "root_task_id": "root", "budget_drive_root": str(root)}
    if load_task_result(root, task["id"]) is None:
        write_task_result(root, task["id"], STATUS_COMPLETED,
                          root_task_id=task["root_task_id"],
                          root_phase_checkpoint={"post_task_synthesis": "running"})
    return set_root_post_task_checkpoint(
        SimpleNamespace(drive_root=root, repo_dir=root.parent / "repo"), task, status,
    )


def test_root_snapshot_reaches_saved_and_gateway_results_without_flattening_own_cost(root):
    _settle(root, 1.0)
    reservation = _reserve(root, task_id="child")
    ua.mark_dispatched(reservation)
    ua.mark_unresolved(reservation, "provider_outcome_unknown")
    stored = _checkpoint(root)
    expected = {
        "schema": "ouroboros.root_cost_snapshot.v1", "scope": "root_tree",
        "root_task_id": "root", "cost_accounting_status": "available",
        "accounted_upper_bound_usd": 1.25, "unresolved_upper_bound_usd": 0.25,
        "reserved_usd": 0.0, "non_final_rows": 1,
        "attempt_counts": {"unresolved": 1}, "unknown_unmetered": 0,
        "ledger_integrity_degraded": False,
    }
    assert stored["root_phase_checkpoint"]["accounting"] == expected
    assert stored["accounted_upper_bound_usd"] == 1.0
    assert stored["non_final_rows"] == 0  # Own-task openness is not tree openness.
    assert stored["cost_final"] is False
    assert "cost_estimated" not in stored
    saved = json.loads((root / "task_results" / "root.json").read_text(encoding="utf-8"))
    request = Request({
        "type": "http", "method": "GET", "path": "/api/tasks/root",
        "path_params": {"task_id": "root"}, "query_string": b"", "headers": [],
        "app": SimpleNamespace(state=SimpleNamespace(drive_root=root)),
    })
    response = _task_get_response(request)
    assert response.status_code == 200
    public = json.loads(response.body)
    assert public["root_phase_checkpoint"]["accounting"] == expected
    assert saved["root_phase_checkpoint"]["accounting"] == expected


@pytest.mark.parametrize("state", ["reserved", "dispatched", "estimated", "unknown"])
def test_zero_dollar_or_unpriced_rows_remain_distinguishable_from_unresolved(root, state):
    _settle(root, 1.0)
    reservation = _reserve(root, task_id="child", bound=0.0)
    if state != "reserved":
        ua.mark_dispatched(reservation)
    if state in {"estimated", "unknown"}:
        ua.settle_attempt(reservation, {}, cost_usd=0.0 if state == "estimated" else None,
                          cost_final=False)
    snapshot = _checkpoint(root)["root_phase_checkpoint"]["accounting"]
    assert snapshot["non_final_rows"] == 1
    assert snapshot["attempt_counts"] == {"unresolved": 0}
    assert snapshot["unknown_unmetered"] == int(state == "unknown")
    assert snapshot["reserved_usd"] == snapshot["unresolved_upper_bound_usd"] == 0.0


def test_unavailable_refresh_replaces_old_tree_proof(root, monkeypatch):
    _settle(root, 1.0)
    before = _checkpoint(root)
    assert before["root_phase_checkpoint"]["accounting"]["cost_accounting_status"] == "available"

    def unavailable(*args, **kwargs):
        raise ua.UsageAccountingError("unavailable test ledger")

    monkeypatch.setattr(ua, "usage_breakdown", unavailable)
    after = _checkpoint(root, "refresh")
    snapshot = after["root_phase_checkpoint"]["accounting"]
    assert after["root_phase_checkpoint"]["post_task_synthesis"] == "completed"
    assert snapshot["cost_accounting_status"] == "unavailable"
    assert snapshot["ledger_integrity_degraded"] is True
    for key in ("accounted_upper_bound_usd", "unresolved_upper_bound_usd", "reserved_usd",
                "non_final_rows", "attempt_counts", "unknown_unmetered"):
        assert snapshot[key] is None


def test_running_checkpoint_emits_no_terminal_accounting_snapshot(root):
    _settle(root, 1.0)
    assert "accounting" not in _checkpoint(root, "running")["root_phase_checkpoint"]


@pytest.mark.parametrize("stale_phase", ["pending_once", "running", "degraded", "completed"])
def test_canonical_snapshot_survives_stale_replica_reads_and_copyback(root, stale_phase):
    _settle(root, 1.0)
    canonical = _checkpoint(root)
    child = root.parent / "child"
    replica = deepcopy(canonical)
    replica["root_phase_checkpoint"].update({
        "post_task_synthesis": stale_phase, "accounting": {"wrong": "replica"},
    })
    write_task_result(child, "root", STATUS_COMPLETED,
                      root_phase_checkpoint=replica["root_phase_checkpoint"], result="kept answer")
    write_task_result(root, "root", STATUS_COMPLETED, child_drive_root=str(child))
    effective = load_effective_task_result(root, "root", materialize_artifacts=False)
    assert effective["root_phase_checkpoint"]["accounting"] == canonical["root_phase_checkpoint"]["accounting"]
    copied = copy_child_task_result(root, {"id": "root", "drive_root": str(child)})
    assert copied["root_phase_checkpoint"]["accounting"] == canonical["root_phase_checkpoint"]["accounting"]


def test_legacy_canonical_absence_cannot_be_filled_by_replica():
    canonical = {"root_phase_checkpoint": {"post_task_synthesis": "completed"}}
    replica = {"root_phase_checkpoint": {
        "post_task_synthesis": "completed", "accounting": {"wrong": "replica"},
    }}
    projected = project_replica_task_result_fields(canonical, replica)
    assert "accounting" not in projected["root_phase_checkpoint"]
    assert "accounting" in replica["root_phase_checkpoint"]


def test_checkpoint_refresh_updates_snapshot_but_stale_phase_patch_does_not(root):
    _settle(root, 1.0)
    before = _checkpoint(root)
    _settle(root, 0.5)
    after = _checkpoint(root, "refresh")
    assert after["root_phase_checkpoint"]["accounting"]["accounted_upper_bound_usd"] == 1.5
    for phase in ("running", "degraded"):
        patch = {"root_phase_checkpoint": {
            "post_task_synthesis": phase, "accounting": before["root_phase_checkpoint"]["accounting"],
        }}
        projected = project_root_post_task_checkpoint_fields(after, patch)
        assert projected["root_phase_checkpoint"] == after["root_phase_checkpoint"]


def test_an_imported_weighted_aggregate_keeps_checkpoint_counts(root, tmp_path):
    from tests._usage_store_testing import write_compacted_journal

    for _ in range(6):
        reservation = _reserve(root)
        ua.mark_dispatched(reservation)
        ua.mark_unresolved(reservation, "provider_outcome_unknown")
    for _ in range(4):
        _settle(root, 0.5)
    live = _checkpoint(root)["root_phase_checkpoint"]["accounting"]
    assert live["attempt_counts"]["unresolved"] == 6 and live["non_final_rows"] == 6

    # The same money as the retired compaction left it: the four settled
    # attempts folded into one aggregate (weight 4), the open ones retained.
    imported = tmp_path / "imported"
    imported.mkdir()
    common = dict(model="openai/test", provider="openai", task_id="root", root_task_id="root",
                  root_limit_usd=100.0, pricing_known=True)
    chains = []
    for index in range(6):
        row = {**common, "kind": "attempt", "attempt_id": f"open-{index}", "reservation_upper_bound_usd": 0.25}
        chains += [{**row, "state": "reserved"}, {**row, "state": "dispatched"},
                   {**row, "state": "unresolved", "reason": "provider_outcome_unknown"}]
    write_compacted_journal(imported, [{**common, "attempt_id": "fold-root", "folded_attempt_count": 4,
                                        "cost_usd": 2.0, "cost_final": True,
                                        "reservation_upper_bound_usd": 1.0}], chains)
    assert _checkpoint(imported)["root_phase_checkpoint"]["accounting"] == live


def test_snapshot_keeps_logical_root_scope_on_retry(root):
    reservation = _reserve(root, task_id="first", root_id="logical-root")
    ua.mark_dispatched(reservation)
    ua.mark_unresolved(reservation, "provider_outcome_unknown")
    unrelated = _reserve(root, task_id="other", root_id="other")
    ua.mark_dispatched(unrelated)
    task = {
        "id": "retry", "root_task_id": "logical-root", "parent_task_id": "",
        "delegation_role": "root", "original_task_id": "first",
        "timeout_retry_from": "first", "budget_drive_root": str(root),
    }
    snapshot = _checkpoint(root, task=task)["root_phase_checkpoint"]["accounting"]
    assert snapshot["root_task_id"] == "logical-root"
    assert snapshot["non_final_rows"] == 1
    assert snapshot["attempt_counts"] == {"unresolved": 1}
    assert snapshot["accounted_upper_bound_usd"] == 0.25


def test_snapshot_exposes_unknown_and_integrity_without_claiming_free_work(root):
    reservation = _reserve(root, bound=None, unknown=True)
    ua.mark_dispatched(reservation)
    ua.mark_unresolved(reservation, "provider_outcome_unknown")
    (root / ua.QUARANTINE_REL).write_text("{}\n", encoding="utf-8")
    snapshot = _checkpoint(root)["root_phase_checkpoint"]["accounting"]
    assert snapshot["cost_accounting_status"] == "available"
    assert snapshot["accounted_upper_bound_usd"] is None
    assert snapshot["unknown_unmetered"] == 1
    assert snapshot["ledger_integrity_degraded"] is True
    assert snapshot["non_final_rows"] == 1
    assert snapshot["attempt_counts"] == {"unresolved": 1}
