"""CyberGym reconcile-launcher tests: durability ordering, dedup, refusals.

Split from the protocol suite along the reconcile-arm seam: these tests drive
``reconcile_main`` against real on-disk run roots with a fake executor — no
Docker daemon, upstream package, or provider credential is used.
"""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from devtools.benchmarks.cybergym import cybergym_reconcile
from devtools.benchmarks.cybergym.cybergym_adapter import (
    BudgetLedger,
    append_cybergym_result,
    campaign_execution_lock,
    task_slug,
)
from devtools.benchmarks.cybergym.cybergym_reconcile import reconcile_main
from tests.test_cybergym_protocol import OFFICIAL_MODEL, _reconcile_args

_TERMINAL_OUTCOME = {
    "status": "infra_failed",
    "lifecycle": "gateway_terminal",
    "infra_reason": "failed",
    "runtime_result": {"task_id": "gateway-1", "status": "failed"},
    "cost_usd": 0.0,
    "cost_estimated": False,
    "cost_final": True,
    "cost_status": "known_no_dispatch",
    "artifact_refs": {},
    "error": "worker crashed",
}


def _write_run(run_dir, task_ids, *, rows=(), checkpoints=(), state_layout=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "harness": {"model": OFFICIAL_MODEL, "ouroboros_url": "http://127.0.0.1:8765"},
        "requested_task_ids": list(task_ids),
        "extra": {},
    }
    if state_layout is not None:
        manifest["extra"]["state_layout"] = state_layout
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if rows:
        with (run_dir / "result_index.jsonl").open("w", encoding="utf-8") as handle:
            for task_id in rows:
                handle.write(json.dumps({"task_id": task_id}) + "\n")
    claims = []
    for task_id, attempt_id in checkpoints:
        claims.append(
            {
                "schema": "ouroboros.benchmark.cybergym.claims.v1",
                "event": "claim",
                "task_id": task_id,
                "attempt_id": attempt_id,
                "reserved_usd": 1.0,
                "ts_unix": 1.0,
            }
        )
        attempt_dir = run_dir / "checkpoints" / task_slug(task_id) / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "gateway_checkpoint.json").write_text(
            json.dumps({"gateway_task_id": f"gateway-{attempt_id}", "status": "failed"}),
            encoding="utf-8",
        )
    if claims:
        (run_dir / "claims.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in claims),
            encoding="utf-8",
        )
    return run_dir


def _read_manifest(run_dir):
    return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))


def _read_rows(run_dir):
    index = run_dir / "result_index.jsonl"
    if not index.exists():
        return []
    return [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]


class _FakeExecutor:
    def __init__(self, outcome, events=None):
        self.outcome = outcome
        self.events = events
        self.reconciled = []
        self.released = []
        self.adopted_workspaces = []

    def adopt_campaign(self):
        return {"status": "adopted", "ok": True}

    def reconcile_task(self, spec, task_dir, attempt_id, checkpoint):
        self.reconciled.append((spec.task_id, attempt_id))
        return dict(self.outcome)

    def release_reconciled_workspace(self, spec, attempt_id):
        self.released.append((spec.task_id, attempt_id))
        if self.events is not None:
            self.events.append("cleanup")
        return None

    def adopt_reconciled_workspace(self, spec, attempt_id):
        self.adopted_workspaces.append((spec.task_id, attempt_id))
        return "container-id"

    def finalize_adopted_campaign(self):
        return {"status": "verified", "ok": True}

    def close(self):
        return {"status": "detached"}


def _install_fake_executor(monkeypatch, fake):
    monkeypatch.setattr(
        "devtools.benchmarks.cybergym.run_cybergym._build_default_executor",
        lambda *args, **kwargs: SimpleNamespace(executor=fake),
    )


def test_reconcile_releases_workspace_only_after_row_and_settle(tmp_path, monkeypatch):
    """Durability order: row append first, claim settle next, cleanup last."""
    run_dir = _write_run(tmp_path / "run", ["arvo:1"], checkpoints=[("arvo:1", "attempt-a01")])
    events = []
    fake = _FakeExecutor(_TERMINAL_OUTCOME, events=events)
    _install_fake_executor(monkeypatch, fake)
    real_append_pair = cybergym_reconcile._append_result_pair
    real_settle = cybergym_reconcile.settle_finished_attempt

    def recording_append(root, row):
        events.append("append")
        return real_append_pair(root, row)

    def recording_settle(ledger, attempt_id, outcome):
        events.append("settle")
        return real_settle(ledger, attempt_id, outcome)

    monkeypatch.setattr(cybergym_reconcile, "_append_result_pair", recording_append)
    monkeypatch.setattr(cybergym_reconcile, "settle_finished_attempt", recording_settle)

    assert reconcile_main(_reconcile_args(run_dir)) == 0
    assert events == ["append", "settle", "cleanup"]
    assert fake.released == [("arvo:1", "attempt-a01")]
    rows = _read_rows(run_dir)
    assert [row["task_id"] for row in rows] == ["arvo:1"]
    report = _read_manifest(run_dir)["extra"]["reconcile_passes"][-1]
    assert report["status"] == "completed"
    assert [entry["attempt_id"] for entry in report["delivered"]] == ["attempt-a01"]


def test_terminal_sparse_cost_row_adopts_workspace_for_post_durable_cleanup(
    tmp_path, monkeypatch
):
    run_dir = _write_run(
        tmp_path / "run",
        ["arvo:1"],
        checkpoints=[("arvo:1", "attempt-a01")],
    )
    outcome = {
        "status": "infra_failed",
        "lifecycle": "terminal_cost_unverifiable",
        "infra_reason": "terminal_cost_unverifiable",
        "reconcile_disposition": "delivery_failed",
        "runtime_result": {
            "task_id": "gateway-task-1",
            "status": "completed",
            "cost_usd": 1.25,
            "cost_final": False,
            "unresolved_upper_bound_usd": 0.2,
        },
    }
    fake = _FakeExecutor(outcome)
    _install_fake_executor(monkeypatch, fake)

    assert reconcile_main(_reconcile_args(run_dir)) == 2
    assert fake.adopted_workspaces == [("arvo:1", "attempt-a01")]
    assert fake.released == [("arvo:1", "attempt-a01")]
    assert len(_read_rows(run_dir)) == 1


def test_reconcile_crash_before_append_keeps_workspace_and_writes_no_row(tmp_path, monkeypatch):
    """A crash before the append must strand neither the container nor the row."""
    run_dir = _write_run(tmp_path / "run", ["arvo:1"], checkpoints=[("arvo:1", "attempt-a01")])
    fake = _FakeExecutor(_TERMINAL_OUTCOME)
    _install_fake_executor(monkeypatch, fake)

    def crashing_append(root, row):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(cybergym_reconcile, "_append_result_pair", crashing_append)
    with pytest.raises(RuntimeError, match="simulated crash"):
        reconcile_main(_reconcile_args(run_dir))
    assert fake.released == []
    assert _read_rows(run_dir) == []


def test_reconcile_crash_between_append_and_settle_keeps_workspace(tmp_path, monkeypatch):
    """The row is durable first; a settle crash must not cost the container."""
    run_dir = _write_run(tmp_path / "run", ["arvo:1"], checkpoints=[("arvo:1", "attempt-a01")])
    fake = _FakeExecutor(_TERMINAL_OUTCOME)
    _install_fake_executor(monkeypatch, fake)
    real_settle = cybergym_reconcile.settle_finished_attempt

    def crashing_settle(ledger, attempt_id, outcome):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(cybergym_reconcile, "settle_finished_attempt", crashing_settle)
    with pytest.raises(RuntimeError, match="simulated crash"):
        reconcile_main(_reconcile_args(run_dir))
    assert fake.released == []
    assert [row["task_id"] for row in _read_rows(run_dir)] == ["arvo:1"]

    monkeypatch.setattr(
        cybergym_reconcile,
        "settle_finished_attempt",
        real_settle,
    )
    assert reconcile_main(_reconcile_args(run_dir)) == 0
    assert fake.reconciled == [("arvo:1", "attempt-a01")]
    assert fake.adopted_workspaces == [("arvo:1", "attempt-a01")]
    assert fake.released == [("arvo:1", "attempt-a01")]


def test_reconcile_processes_each_retry_attempt_independently(tmp_path, monkeypatch):
    attempts = ("attempt-a01", "attempt-a02")
    run_dir = _write_run(
        tmp_path / "run",
        ["arvo:1"],
        checkpoints=[("arvo:1", attempt) for attempt in attempts],
    )
    expected_dirs = []
    for attempt in attempts:
        retry_dir = run_dir / task_slug("arvo:1") / attempt
        retry_dir.mkdir(parents=True)
        expected_dirs.append(retry_dir)

    class RetryDirExecutor(_FakeExecutor):
        def __init__(self, outcome):
            super().__init__(outcome)
            self.task_dirs = []

        def reconcile_task(self, spec, task_dir, attempt_id, checkpoint):
            self.task_dirs.append(task_dir)
            return super().reconcile_task(spec, task_dir, attempt_id, checkpoint)

    fake = RetryDirExecutor(_TERMINAL_OUTCOME)
    _install_fake_executor(monkeypatch, fake)

    assert reconcile_main(_reconcile_args(run_dir)) == 0
    assert fake.reconciled == [
        ("arvo:1", "attempt-a01"),
        ("arvo:1", "attempt-a02"),
    ]
    assert fake.task_dirs == expected_dirs
    assert len(_read_rows(run_dir)) == 2
    report = _read_manifest(run_dir)["extra"]["reconcile_passes"][-1]
    assert [entry["attempt_id"] for entry in report["delivered"]] == [
        "attempt-a01", "attempt-a02",
    ]
    assert report["skipped_recorded"] == []
    assert report["undeliverable"] == []


def test_reconcile_drops_row_when_task_was_recorded_mid_delivery(tmp_path, monkeypatch):
    """A row appearing under the delivery window wins; the duplicate is dropped."""
    run_dir = _write_run(tmp_path / "run", ["arvo:1"], checkpoints=[("arvo:1", "attempt-a01")])

    class RacingExecutor(_FakeExecutor):
        def reconcile_task(self, spec, task_dir, attempt_id, checkpoint):
            outcome = super().reconcile_task(spec, task_dir, attempt_id, checkpoint)
            append_cybergym_result(
                run_dir,
                {"task_id": spec.task_id, "attempt_id": attempt_id, "status": "infra_failed"},
            )
            return outcome

    fake = RacingExecutor(_TERMINAL_OUTCOME)
    _install_fake_executor(monkeypatch, fake)

    assert reconcile_main(_reconcile_args(run_dir)) == 2
    assert len(_read_rows(run_dir)) == 1
    report = _read_manifest(run_dir)["extra"]["reconcile_passes"][-1]
    assert report["delivered"] == []
    assert report["skipped_recorded"] == []
    assert [entry["disposition"] for entry in report["undeliverable"]] == [
        "recorded_elsewhere"
    ]
    assert fake.released == []


def test_result_pair_append_repairs_torn_task_local_row(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    row = {"task_id": "arvo:1", "attempt_id": "attempt-a01", "status": "completed"}
    (run_dir / "result_index.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    append_cybergym_result(run_dir, row)

    assert _read_rows(run_dir) == [row]
    assert _read_rows(run_dir / task_slug("arvo:1")) == [row]


def test_reconcile_uses_exact_recorded_row_for_each_retry_attempt(tmp_path, monkeypatch):
    attempts = ("attempt-a01", "attempt-a02")
    run_dir = _write_run(
        tmp_path / "run",
        ["arvo:1"],
        checkpoints=[("arvo:1", attempt) for attempt in attempts],
    )
    rows = [
        {
            "task_id": "arvo:1",
            "attempt_id": attempt,
            "status": "infra_failed",
            "cost_usd": 0.1,
            "cost_estimated": False,
            "cost_final": True,
        }
        for attempt in attempts
    ]
    (run_dir / "result_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    fake = _FakeExecutor(_TERMINAL_OUTCOME)
    _install_fake_executor(monkeypatch, fake)

    assert reconcile_main(_reconcile_args(run_dir)) == 0
    assert fake.released == [("arvo:1", attempt) for attempt in attempts]
    assert BudgetLedger(run_dir / "claims.jsonl", cap_usd=3500).projection().active_attempt_ids == ()
    assert _read_rows(run_dir / task_slug("arvo:1")) == rows


def test_corrupt_torn_task_local_row_refuses_campaign_finalization(tmp_path, monkeypatch):
    run_dir = _write_run(
        tmp_path / "run",
        ["arvo:1"],
        rows=["arvo:1"],
        checkpoints=[("arvo:1", "attempt-a01")],
    )
    task_dir = run_dir / task_slug("arvo:1")
    task_dir.mkdir()
    (task_dir / "result_index.jsonl").write_text('{"task_id":', encoding="utf-8")

    class TrackingExecutor(_FakeExecutor):
        finalized = 0
        detached = 0

        def finalize_adopted_campaign(self):
            self.finalized += 1
            return super().finalize_adopted_campaign()

        def close(self):
            self.detached += 1
            return super().close()

    fake = TrackingExecutor(_TERMINAL_OUTCOME)
    _install_fake_executor(monkeypatch, fake)

    assert reconcile_main(_reconcile_args(run_dir)) == 2
    assert fake.finalized == 0
    assert fake.detached == 1
    assert fake.released == []
    report = _read_manifest(run_dir)["extra"]["reconcile_passes"][-1]
    assert report["status"] == "partial"
    assert report["undeliverable"][0]["disposition"] == "row_refused"


def test_second_concurrent_reconcile_process_is_refused(tmp_path, capsys, monkeypatch):
    pytest.importorskip("fcntl")
    run_dir = _write_run(tmp_path / "run", ["arvo:1"], rows=["arvo:1"])
    manifest_path = run_dir / "run_manifest.json"
    manifest_reads = []
    real_read_text = pathlib.Path.read_text

    def track_manifest_read(path, *args, **kwargs):
        if path == manifest_path:
            manifest_reads.append(path)
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", track_manifest_read)
    with campaign_execution_lock(run_dir, blocking=False) as lock_held:
        assert lock_held is True
        assert reconcile_main(_reconcile_args(run_dir)) == 2
    assert manifest_reads == []
    assert "another launcher or --reconcile process" in capsys.readouterr().err


def test_reconcile_state_dir_override_must_match_manifest(tmp_path, capsys):
    recorded = tmp_path / "recorded-state"
    run_dir = _write_run(
        tmp_path / "run",
        ["arvo:1"],
        rows=["arvo:1"],
        state_layout={
            "mode": "external_state_dir",
            "state_dir": str(recorded),
            "data_root": str((recorded / "ouroboros-data").resolve()),
        },
    )
    args = _reconcile_args(run_dir, state_dir=str(tmp_path / "other-state"))
    assert reconcile_main(args) == 2
    assert "state root" in capsys.readouterr().err


def test_reconcile_state_dir_override_matching_manifest_is_accepted(tmp_path):
    recorded = tmp_path / "recorded-state"
    run_dir = _write_run(
        tmp_path / "run",
        ["arvo:1"],
        rows=["arvo:1"],
        state_layout={
            "mode": "external_state_dir",
            "state_dir": str(recorded),
            "data_root": str((recorded / "ouroboros-data").resolve()),
        },
    )
    assert reconcile_main(_reconcile_args(run_dir, state_dir=str(recorded))) == 0
    report = _read_manifest(run_dir)["extra"]["reconcile_passes"][-1]
    assert report["status"] == "nothing_pending"


def test_reconcile_state_dir_override_without_recorded_layout_is_accepted(tmp_path):
    run_dir = _write_run(tmp_path / "run", ["arvo:1"], rows=["arvo:1"])
    args = _reconcile_args(run_dir, state_dir=str(tmp_path / "anywhere"))
    assert reconcile_main(args) == 0


def test_reconcile_without_rows_or_checkpoints_is_incomplete(tmp_path, capsys):
    """An empty run is an incomplete recovery, never a silent success."""
    run_dir = _write_run(tmp_path / "run", ["arvo:1", "arvo:2"])
    assert reconcile_main(_reconcile_args(run_dir)) == 2
    manifest = _read_manifest(run_dir)
    report = manifest["extra"]["reconcile_passes"][-1]
    assert report["status"] == "incomplete"
    assert report["unaccounted"] == ["arvo:1", "arvo:2"]
    assert manifest["extra"]["outcome"] == "reconcile_incomplete"
    assert manifest["extra"]["exit_code"] == 2
    assert "reconcile incomplete" in capsys.readouterr().err


def test_reconcile_partially_recorded_without_checkpoints_is_incomplete(tmp_path):
    run_dir = _write_run(tmp_path / "run", ["arvo:1", "arvo:2"], rows=["arvo:1"])
    assert reconcile_main(_reconcile_args(run_dir)) == 2
    report = _read_manifest(run_dir)["extra"]["reconcile_passes"][-1]
    assert report["status"] == "incomplete"
    assert report["unaccounted"] == ["arvo:2"]
    assert report["already_recorded"] == ["arvo:1"]


def test_reconcile_passes_are_append_only(tmp_path):
    """A second pass appends its report; the first pass record survives."""
    run_dir = _write_run(tmp_path / "run", ["arvo:1"], rows=["arvo:1"])
    assert reconcile_main(_reconcile_args(run_dir)) == 0
    assert reconcile_main(_reconcile_args(run_dir)) == 0
    extra = _read_manifest(run_dir)["extra"]
    passes = extra["reconcile_passes"]
    assert len(passes) == 2
    assert all(report["status"] == "nothing_pending" for report in passes)
    assert all("pass_unix_ts" in report for report in passes)
    assert "reconcile" not in extra
