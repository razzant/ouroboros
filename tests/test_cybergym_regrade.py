from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from devtools.benchmarks.common.result_index import read_result_index
from devtools.benchmarks.cybergym.cybergym_regrade import (
    execute_regrade_inventory,
    select_latest_regrade_candidates,
    write_regrade_inventory,
)

MODEL = "deepseek/deepseek-v4-flash-0731"


def _row(task_id, task_dir, *, ts, status="completed", result=False, digest=""):
    return {
        "task_id": task_id,
        "status": status,
        "final_submission_status": "known_success" if result else "known_failure",
        "final_submission_success": result,
        "observed_model": MODEL,
        "observed_effort": "high",
        "cost_final": True,
        "final_poc_hash": digest,
        "artifact_refs": {"task_dir": str(task_dir)},
        "ts_unix": ts,
    }


def test_selects_latest_eligible_outcome_without_preferring_success(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        root.mkdir()
    old_dir = first / "arvo__1"
    new_dir = second / "arvo__1"
    old_dir.mkdir()
    new_dir.mkdir()
    (old_dir / "final.poc").write_bytes(b"old")
    (new_dir / "final.poc").write_bytes(b"new")
    old_hash = hashlib.sha256(b"old").hexdigest()
    new_hash = hashlib.sha256(b"new").hexdigest()
    (first / "result_index.jsonl").write_text(
        json.dumps(_row("arvo:1", old_dir, ts=1, result=True, digest=old_hash)) + "\n"
    )
    (second / "result_index.jsonl").write_text(
        json.dumps(_row("arvo:1", new_dir, ts=2, result=False, digest=new_hash)) + "\n"
        + json.dumps(_row("arvo:2", second / "arvo__2", ts=2, status="failed")) + "\n"
    )

    selected = select_latest_regrade_candidates([first, second])

    assert [item["task_id"] for item in selected] == ["arvo:1", "arvo:2"]
    assert selected[0]["historical_final_submission_success"] is False
    assert selected[0]["source_final_poc_hash"] == new_hash
    assert selected[0]["regrade_status"] == "ready"
    assert selected[1]["regrade_status"] == "no_final_poc"


def test_inventory_is_append_only_and_counts_candidates(tmp_path):
    root = tmp_path / "run"
    task_dir = root / "arvo__1"
    task_dir.mkdir(parents=True)
    (task_dir / "final.poc").write_bytes(b"poc")
    digest = hashlib.sha256(b"poc").hexdigest()
    (root / "result_index.jsonl").write_text(
        json.dumps(_row("arvo:1", task_dir, ts=1, result=True, digest=digest)) + "\n"
    )

    output = tmp_path / "inventory.json"
    payload = write_regrade_inventory(output, [root])

    assert payload["counts"] == {"selected": 1, "ready": 1, "no_final_poc": 0, "hash_mismatch": 0}
    assert json.loads(output.read_text())["candidates"][0]["task_id"] == "arvo:1"


class _FakeRegradeExecutor:
    def __init__(self, config):
        self.config = config
        self.regraded: list[str] = []
        self.cleaned: list[str] = []
        self.closed = False

    def regrade_final_poc(self, task, source_marker, task_dir, *, attempt_id=""):
        self.regraded.append(task.task_id)
        digest = hashlib.sha256(b"poc-" + task.task_id.encode()).hexdigest()
        return {
            "task_id": task.task_id,
            "attempt_id": attempt_id,
            "source_final_poc_sha256": digest,
            "classification": {"official_success": task.task_id != "arvo:2"},
            "record": {"vul_exit_code": 1, "fix_exit_code": 0, "poc_id": "poc-1", "poc_hash": digest},
        }

    def _cleanup_workspace_container(self, container_name, task_id, attempt_id, report_path):
        self.cleaned.append(task_id)

    def close(self):
        self.closed = True
        return {"status": "verified", "ok": True}


def _inventory_payload(root):
    ready_dir = root / "src" / "arvo__1"
    ready_dir.mkdir(parents=True)
    (ready_dir / "final.poc").write_bytes(b"poc-arvo:1")
    ready_hash = hashlib.sha256(b"poc-arvo:1").hexdigest()
    fail_dir = root / "src" / "arvo__2"
    fail_dir.mkdir(parents=True)
    (fail_dir / "final.poc").write_bytes(b"poc-arvo:2")
    fail_hash = hashlib.sha256(b"poc-arvo:2").hexdigest()
    return {
        "schema": "ouroboros.benchmark.cybergym.regrade_inventory.v1",
        "model": MODEL,
        "candidates": [
            {
                "task_id": "arvo:1",
                "source_run_root": str(root / "src"),
                "source_line": 1,
                "source_result_sha256": "a" * 64,
                "historical_final_submission_success": True,
                "regrade_status": "ready",
                "source_final_poc_path": str(ready_dir / "final.poc"),
                "source_final_poc_hash": ready_hash,
            },
            {
                "task_id": "arvo:2",
                "source_run_root": str(root / "src"),
                "source_line": 2,
                "source_result_sha256": "b" * 64,
                "historical_final_submission_success": False,
                "regrade_status": "ready",
                "source_final_poc_path": str(fail_dir / "final.poc"),
                "source_final_poc_hash": fail_hash,
            },
            {
                "task_id": "arvo:3",
                "source_run_root": str(root / "src"),
                "source_line": 3,
                "source_result_sha256": "c" * 64,
                "historical_final_submission_success": False,
                "regrade_status": "no_final_poc",
            },
        ],
    }


def test_execute_regrade_records_every_candidate_and_resumes(tmp_path):
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(_inventory_payload(tmp_path)))
    config = SimpleNamespace(provider_probe=False, campaign_id="regrade-test")
    run_root = tmp_path / "regrade-run"
    executor = _FakeRegradeExecutor(config)

    summary = execute_regrade_inventory(
        inventory,
        config=config,
        run_root=run_root,
        workers=2,
        executor_factory=lambda _config: executor,
    )

    assert summary["counts"] == {
        "rows": 3,
        "regraded": 2,
        "official_success": 1,
        "official_failure": 1,
        "no_final_poc": 1,
        "hash_mismatch": 0,
        "error": 0,
    }
    assert summary["historical_success_among_regraded"] == 1
    assert sorted(executor.regraded) == ["arvo:1", "arvo:2"]
    assert sorted(executor.cleaned) == ["arvo:1", "arvo:2"]
    assert executor.closed is True
    rows = read_result_index(run_root)
    assert len(rows) == 3
    by_task = {row["task_id"]: row for row in rows}
    assert by_task["arvo:1"]["official_success"] is True
    assert by_task["arvo:2"]["official_success"] is False
    assert by_task["arvo:3"]["regraded"] is False

    second = _FakeRegradeExecutor(config)
    resumed = execute_regrade_inventory(
        inventory,
        config=config,
        run_root=run_root,
        workers=2,
        executor_factory=lambda _config: second,
    )

    assert resumed["counts"]["rows"] == 3
    assert second.regraded == []
    assert len(read_result_index(run_root)) == 3


def test_execute_regrade_records_executor_failures_without_stopping(tmp_path):
    payload = _inventory_payload(tmp_path)
    payload["candidates"] = payload["candidates"][:2]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload))
    config = SimpleNamespace(provider_probe=False, campaign_id="regrade-test")

    class FailingExecutor(_FakeRegradeExecutor):
        def regrade_final_poc(self, task, source_marker, task_dir, *, attempt_id=""):
            raise RuntimeError("verifier exploded")

    summary = execute_regrade_inventory(
        inventory,
        config=config,
        run_root=tmp_path / "regrade-run",
        workers=1,
        executor_factory=FailingExecutor,
    )

    assert summary["counts"]["error"] == 2
    assert summary["counts"]["regraded"] == 0
    rows = read_result_index(tmp_path / "regrade-run")
    assert {row["error_type"] for row in rows} == {"RuntimeError"}
