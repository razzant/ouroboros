"""CyberGym Docker-runtime custody tests: workspace slots, network lifecycle.

Split from the executor suites along the container-machinery seam: these tests
exercise ``_workspace``/``_network`` failure custody and the pool-slot release
contract with injected command runners — no Docker daemon, upstream package,
or provider credential is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import threading

import pytest

from devtools.benchmarks.cybergym import cybergym_docker as docker_module
from devtools.benchmarks.cybergym import cybergym_executor as executor_module
from devtools.benchmarks.cybergym.cybergym_adapter import (
    BudgetLedger,
    CyberGymError,
    LedgerError,
    TaskSpec,
    WorkspaceCustodyPending,
    run_campaign,
)
from devtools.benchmarks.cybergym.cybergym_executor import (
    CommandResult,
    CyberGymExecutor,
    ExecutorFailure,
)
from tests.test_cybergym_executor import (
    _config,
    _requires_posix_mount_paths,
    dataclasses_replace,
)


def test_post_create_timeout_preserves_workspace_custody(tmp_path, monkeypatch):
    """An admitted-but-unresolved attempt keeps its exact workspace for reconcile."""
    config = _config(tmp_path, provider_probe=False)
    executor = CyberGymExecutor(config)
    cleaned = []

    def fake_workspace(task, task_dir, plan):
        name = f"cybergym-workspace-{plan.opaque_agent_id}"
        executor._task_containers[name] = "d" * 64
        return name

    def fake_cleanup(name, task_id, attempt_id, report_path):
        cleaned.append(name)
        executor._task_containers.pop(name, None)
        return {"status": "verified", "ok": True}

    monkeypatch.setattr(executor, "start", lambda: None)
    monkeypatch.setattr(executor, "_generate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        executor_module,
        "_install_workspace_backend_alias",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(executor, "_workspace", fake_workspace)
    monkeypatch.setattr(executor, "_cleanup_workspace_container", fake_cleanup)
    monkeypatch.setattr(
        executor,
        "_task_body",
        lambda task, *_args, **_kwargs: {"task_id": "cybergym-" + task.task_id.replace(":", "-")},
    )
    monkeypatch.setattr(
        executor,
        "_gateway_wait",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ExecutorFailure("status poll timed out after admission")
        ),
    )
    rows = run_campaign(
        ["arvo:1", "arvo:2"],
        run_root=config.run_root,
        executor=executor.run_task,
        estimated_cost_usd=1,
        budget_cap_usd=5,
    )
    assert [row["status"] for row in rows] == ["infra_failed", "infra_failed"]
    assert cleaned == []
    assert len(executor._task_containers) == 2
    projection = BudgetLedger(config.run_root / "claims.jsonl", cap_usd=5).projection()
    assert projection.reserved_usd == pytest.approx(0)
    # Post-dispatch poll timeouts have no terminal frame: both claims settle
    # terminally at their reservation rather than holding an unresolved
    # liability open until reconcile.
    assert projection.settled_usd == pytest.approx(2)
    assert projection.unresolved_upper_bound_usd == pytest.approx(0)
    assert projection.can_dispatch is True


def test_preplan_failure_returns_typed_row_without_durability_ack_error(tmp_path, monkeypatch):
    config = _config(tmp_path, provider_probe=False)
    executor = CyberGymExecutor(config)
    monkeypatch.setattr(
        executor,
        "start",
        lambda: (_ for _ in ()).throw(ExecutorFailure("pre-plan failure")),
    )

    rows = run_campaign(
        ["arvo:1"],
        run_root=config.run_root,
        executor=executor.run_task,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )

    assert rows[0]["status"] == "infra_failed"
    assert rows[0]["infra_reason"] == "ExecutorFailure"
    assert executor._plans == {}
    assert executor._terminal_uncommitted_workspaces == {}


def test_settled_gateway_transfer_to_terminal_custody_is_atomic(tmp_path):
    executor = CyberGymExecutor(_config(tmp_path, provider_probe=False))
    executor._gateway_attempts["gateway-1"] = {
        "workspace_name": "workspace-1",
        "task_id": "arvo:1",
        "attempt_id": "attempt-a01",
    }

    executor._terminalize_gateway_attempt("gateway-1")

    assert executor._gateway_attempts == {}
    assert executor._terminal_uncommitted_workspaces == {
        "workspace-1": {"task_id": "arvo:1", "attempt_id": "attempt-a01"},
    }


def test_terminal_workspace_survives_until_result_and_settlement_return(tmp_path, monkeypatch):
    config = _config(tmp_path, provider_probe=False)
    executor = CyberGymExecutor(config)
    cleaned = []

    def fake_workspace(task, task_dir, plan):
        name = f"cybergym-workspace-{plan.opaque_agent_id}"
        executor._task_containers[name] = "d" * 64
        return name

    monkeypatch.setattr(executor, "start", lambda: None)
    monkeypatch.setattr(executor, "_generate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        executor_module, "_install_workspace_backend_alias", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(executor, "_workspace", fake_workspace)
    monkeypatch.setattr(
        executor, "_cleanup_workspace_container",
        lambda name, *_args, **_kwargs: cleaned.append(name),
    )
    monkeypatch.setattr(
        executor, "_task_body",
        lambda task, *_args, **_kwargs: {"task_id": "cybergym-" + task.task_id.replace(":", "-")},
    )

    def fake_gateway_wait(_body, _checkpoint, **custody):
        executor._gateway_attempts["gateway-test"] = {
            "workspace_name": custody["workspace_name"],
            "task_id": custody["task_id"],
            "attempt_id": custody["attempt_id"],
        }
        executor._terminalize_gateway_attempt("gateway-test")
        return {"status": "failed", "cost_final": True}

    monkeypatch.setattr(executor, "_gateway_wait", fake_gateway_wait)
    monkeypatch.setattr(
        executor, "_deliver_gateway_result",
        lambda *_args, **_kwargs: {
            "status": "infra_failed",
            "infra_reason": "test_terminal",
            "cost_usd": 0.1,
            "cost_estimated": False,
            "cost_final": True,
        },
    )

    rows = run_campaign(
        ["arvo:1"],
        run_root=config.run_root,
        executor=executor.run_task,
        estimated_cost_usd=1,
        budget_cap_usd=2,
    )

    assert rows[0]["status"] == "infra_failed"
    assert cleaned == []
    assert len(executor._task_containers) == 1
    assert executor._terminal_uncommitted_workspaces == {}
    projection = BudgetLedger(config.run_root / "claims.jsonl", cap_usd=2).projection()
    assert projection.settled_usd == pytest.approx(0.1)

    from devtools.benchmarks.cybergym import cybergym_adapter

    monkeypatch.setattr(
        cybergym_adapter,
        "append_cybergym_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("result fsync failed")),
    )
    with pytest.raises(OSError, match="result fsync failed"):
        run_campaign(
            ["arvo:2"],
            run_root=config.run_root,
            executor=executor.run_task,
            estimated_cost_usd=1,
            budget_cap_usd=2,
        )
    assert len(executor._terminal_uncommitted_workspaces) == 1
    close_report = executor.close()
    assert close_report["status"] == "custody_pending"
    assert len(executor._task_containers) == 2
    assert cleaned == []


def test_network_reaps_empty_foreign_leftover_then_creates(tmp_path):
    config = _config(tmp_path)
    stale_id = "stale-be200ad3-network"
    created: list[str] = []
    inspect_by_id = {stale_id: True}

    def command(argv, *, cwd=None, env=None, timeout=None):
        if "network" in argv and "create" in argv:
            if not created:
                created.append("fail")
                return CommandResult(1, "", "network with name cybergym-internal already exists")
            created.append("ok")
            return CommandResult(0, "fresh-network-id\n", "")
        if "network" in argv and "inspect" in argv:
            target = argv[-1]
            if target == stale_id and not inspect_by_id.get(stale_id):
                return CommandResult(1, "", "Error: No such network: stale-be200ad3-network")
            if target in {stale_id, "cybergym-internal"} and inspect_by_id.get(stale_id):
                return CommandResult(
                    0,
                    json.dumps([{
                        "Name": "cybergym-internal",
                        "Id": stale_id,
                        "Internal": False,
                        "Driver": "bridge",
                        "Labels": {"com.ouroboros.campaign": "be200ad3-dead"},
                        "Containers": {},
                    }]),
                    "",
                )
            if target == "cybergym-internal" and created.count("ok"):
                return CommandResult(
                    0,
                    json.dumps([{
                        "Name": "cybergym-internal",
                        "Id": "fresh-network-id",
                        "Internal": False,
                        "Driver": "bridge",
                        "Labels": {"com.ouroboros.campaign": "test-campaign"},
                        "Containers": {},
                    }]),
                    "",
                )
            return CommandResult(1, "", f"Error: No such network: {target}")
        if "network" in argv and "rm" in argv:
            assert argv[-1] == stale_id
            inspect_by_id[stale_id] = False
            return CommandResult(0, "", "")
        raise AssertionError(argv)

    executor = CyberGymExecutor(dataclasses_replace(config, command_runner=command, provider_probe=False))
    executor._network()  # noqa: SLF001 - leftover-network class contract
    assert created == ["fail", "ok"]
    assert executor.network_id == "fresh-network-id"
    assert executor._network_created is True


def test_network_refuses_leftover_with_attached_containers(tmp_path):
    config = _config(tmp_path)

    def command(argv, *, cwd=None, env=None, timeout=None):
        if "create" in argv:
            return CommandResult(1, "", "already exists")
        if "inspect" in argv:
            return CommandResult(
                0,
                json.dumps([{
                    "Name": "cybergym-internal",
                    "Id": "busy-network-id",
                    "Internal": False,
                    "Driver": "bridge",
                    "Labels": {"com.ouroboros.campaign": "other-campaign"},
                    "Containers": {"abc": {"Name": "cybergym-server-other-campaign"}},
                }]),
                "",
            )
        raise AssertionError("must not rm a leftover with containers")

    executor = CyberGymExecutor(dataclasses_replace(config, command_runner=command, provider_probe=False))
    with pytest.raises(ExecutorFailure, match="still has attached containers"):
        executor._network()  # noqa: SLF001 - leftover-network class contract


def _reconcile_fixture(tmp_path, gateway_id, checkpoint_payload, **config_overrides):
    config = _config(tmp_path, **config_overrides)
    executor = CyberGymExecutor(config)
    task_dir = config.run_root / "arvo_1"
    task_dir.mkdir()
    checkpoint = task_dir / "gateway_checkpoint.json"
    checkpoint.write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    return config, executor, task_dir, checkpoint


def test_reconcile_task_leaves_nonterminal_gateway_attempt_running(tmp_path):
    gateway_id = "gateway-task-1"

    def http(method, url, **_kwargs):
        assert method == "GET"
        assert gateway_id in url
        return {"task_id": gateway_id, "status": "running"}

    config, executor, task_dir, checkpoint = _reconcile_fixture(
        tmp_path,
        gateway_id,
        {"gateway_task_id": gateway_id, "status": "running"},
        http_runner=http,
    )
    outcome = executor.reconcile_task(TaskSpec("arvo:1", "arvo"), task_dir, "attempt-1", checkpoint)
    assert outcome["status"] == "infra_failed"
    assert outcome["lifecycle"] == "reconcile_pending"
    assert outcome["reconcile_disposition"] == "left_running"
    assert outcome["gateway_task_id"] == gateway_id
    frame = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert frame["reconciled"] is True
    assert frame["reconcile_source"] == "gateway_poll"
    assert config.run_root in pathlib.Path(outcome["artifact_refs"]["task_dir"]).parents


def test_reconcile_task_records_terminal_sparse_cost_as_infra(tmp_path):
    """A sparse cached frame is a refusal snapshot, not evidence: the gateway
    is re-polled (its sparseness cause may have healed), and only with the
    gateway unreachable and no isolate-disk record does the cached refusal
    stand — the checkpoint is then left untouched for a later pass."""
    gateway_id = "gateway-task-sparse-cost"
    terminal = {
        "task_id": gateway_id,
        "status": "completed",
        "cost_usd": 1.25,
        "cost_final": False,
        "ledger_integrity_degraded": False,
        "cost_accounting_status": "available",
        "unknown_unmetered": 0,
        "reserved_usd": 0.0,
        "unresolved_upper_bound_usd": 0.2,
    }

    polled = []

    def http(method, url, **_kwargs):
        polled.append(url)
        raise ExecutorFailure("isolate gateway is down")

    config, executor, task_dir, checkpoint = _reconcile_fixture(
        tmp_path,
        gateway_id,
        {"gateway_task_id": gateway_id, "status": "completed", "result": terminal},
        http_runner=http,
    )
    outcome = executor.reconcile_task(
        TaskSpec("arvo:1", "arvo"), task_dir, "attempt-1", checkpoint,
    )

    assert polled, "a sparse cached frame must re-poll the gateway once"
    assert outcome["status"] == "infra_failed"
    assert outcome["lifecycle"] == "terminal_cost_unverifiable"
    assert outcome["reconcile_disposition"] == "delivery_failed"
    assert outcome["runtime_result"]["cost_usd"] == pytest.approx(1.25)
    assert config.run_root in pathlib.Path(outcome["artifact_refs"]["task_dir"]).parents
    frame = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert "reconciled" not in frame


def test_reconcile_task_repoll_supersedes_healed_sparse_cache(tmp_path):
    """Run 20260907T233516Z pass-2 case: the cached frame was poisoned by the
    isolate's sticky integrity flag; after the operator-acknowledged quarantine
    heal the re-polled frame is cost-final and must supersede the cache."""
    gateway_id = "gateway-task-sparse-healed"
    sparse = {
        "task_id": gateway_id,
        "status": "completed",
        "cost_usd": 1.25,
        "cost_final": False,
        "ledger_integrity_degraded": False,
        "cost_accounting_status": "available",
        "unknown_unmetered": 0,
        "reserved_usd": 0.0,
        "unresolved_upper_bound_usd": 0.2,
    }
    healed = {
        "task_id": gateway_id,
        "status": "completed",
        "cost_usd": 1.25,
        "cost_final": True,
        "ledger_integrity_degraded": False,
        "cost_accounting_status": "available",
        "unknown_unmetered": 0,
        "reserved_usd": 0.0,
        "unresolved_upper_bound_usd": 0.0,
    }

    def http(method, url, **_kwargs):
        assert method == "GET"
        assert gateway_id in url
        return healed

    _config_unused, executor, task_dir, checkpoint = _reconcile_fixture(
        tmp_path,
        gateway_id,
        {"gateway_task_id": gateway_id, "status": "completed", "result": sparse},
        http_runner=http,
        command_runner=lambda *_args, **_kwargs: CommandResult(1, "", "No such object"),
    )
    outcome = executor.reconcile_task(
        TaskSpec("arvo:1", "arvo"), task_dir, "attempt-1", checkpoint,
    )

    # The sparse refusal is gone: delivery of the healed frame was attempted
    # and stopped at the served-telemetry seam (the frame carries no model
    # evidence), which proves the poll superseded the cache.
    assert outcome["lifecycle"] != "terminal_cost_unverifiable"
    assert outcome["status"] == "infra_failed"
    assert "model" in outcome["error"]
    frame = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert frame["reconciled"] is True
    assert frame["reconcile_source"] == "gateway_poll"
    assert frame["result"]["cost_final"] is True


def test_reconcile_task_disk_record_supersedes_sparse_cache(tmp_path):
    """With the gateway unreachable, a healthy terminal record on the isolate
    data root supersedes the sparse cached refusal snapshot."""
    gateway_id = "gateway-task-sparse-disk-healed"
    sparse = {
        "task_id": gateway_id,
        "status": "completed",
        "cost_usd": 1.25,
        "cost_final": False,
        "unresolved_upper_bound_usd": 0.2,
    }
    external = tmp_path / "nvme" / "ouroboros-data"
    records = external / "task_results"
    records.mkdir(parents=True)
    (records / f"{gateway_id}.json").write_text(
        json.dumps({
            "task_id": gateway_id,
            "status": "completed",
            "cost_usd": 1.25,
            "cost_final": True,
            "unresolved_upper_bound_usd": 0.0,
        }),
        encoding="utf-8",
    )

    config, executor, task_dir, checkpoint = _reconcile_fixture(
        tmp_path,
        gateway_id,
        {"gateway_task_id": gateway_id, "status": "completed", "result": sparse},
        http_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ExecutorFailure("isolate gateway is down")
        ),
        isolate_data_root=external,
        command_runner=lambda *_args, **_kwargs: CommandResult(1, "", "No such object"),
    )
    outcome = executor.reconcile_task(
        TaskSpec("arvo:1", "arvo"), task_dir, "attempt-1", checkpoint,
    )

    assert outcome["lifecycle"] != "terminal_cost_unverifiable"
    assert "model" in outcome["error"]
    frame = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert frame["reconcile_source"] == "isolate_task_results"
    assert frame["result"]["cost_final"] is True


def test_reconcile_task_records_sparse_cost_from_isolate_disk(tmp_path):
    gateway_id = "gateway-task-sparse-disk"
    external = tmp_path / "nvme" / "ouroboros-data"
    records = external / "task_results"
    records.mkdir(parents=True)
    (records / f"{gateway_id}.json").write_text(
        json.dumps({
            "task_id": gateway_id,
            "status": "completed",
            "cost_usd": 2.5,
            "cost_final": False,
            "unresolved_upper_bound_usd": 0.4,
        }),
        encoding="utf-8",
    )

    config, executor, task_dir, checkpoint = _reconcile_fixture(
        tmp_path,
        gateway_id,
        {"gateway_task_id": gateway_id, "status": "running"},
        http_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ExecutorFailure("isolate gateway is down")
        ),
        isolate_data_root=external,
    )
    outcome = executor.reconcile_task(
        TaskSpec("arvo:1", "arvo"), task_dir, "attempt-1", checkpoint,
    )

    assert outcome["status"] == "infra_failed"
    assert outcome["lifecycle"] == "terminal_cost_unverifiable"
    assert outcome["runtime_result"]["cost_usd"] == pytest.approx(2.5)
    frame = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert frame["reconcile_source"] == "isolate_task_results"


def test_reconcile_task_malformed_checkpoint_is_undeliverable(tmp_path):
    config = _config(tmp_path)
    executor = CyberGymExecutor(config)
    task_dir = config.run_root / "arvo_1"
    task_dir.mkdir()
    checkpoint = task_dir / "gateway_checkpoint.json"
    checkpoint.write_text("{not-json", encoding="utf-8")
    outcome = executor.reconcile_task(TaskSpec("arvo:1", "arvo"), task_dir, "attempt-1", checkpoint)
    assert outcome["status"] == "infra_failed"
    assert outcome["lifecycle"] == "reconcile_blocked"
    assert outcome["reconcile_disposition"] == "undeliverable"
    assert outcome["infra_reason"] == "ExecutorFailure"


def test_reconcile_task_delivers_terminal_failure_from_isolate_disk(tmp_path):
    gateway_id = "gateway-task-9"
    external = tmp_path / "nvme" / "ouroboros-data"
    records = external / "task_results"
    records.mkdir(parents=True)
    (records / f"{gateway_id}.json").write_text(
        json.dumps({"task_id": gateway_id, "status": "failed", "error": "worker crashed"}),
        encoding="utf-8",
    )

    def http(*_args, **_kwargs):
        raise ExecutorFailure("isolate gateway is down")

    config, executor, task_dir, checkpoint = _reconcile_fixture(
        tmp_path,
        gateway_id,
        {"gateway_task_id": gateway_id, "status": "running"},
        http_runner=http,
        isolate_data_root=external,
    )
    outcome = executor.reconcile_task(TaskSpec("arvo:1", "arvo"), task_dir, "attempt-1", checkpoint)
    # A terminal non-completed result delivers its typed infra row without
    # touching Docker; the launcher records that as a delivered row.
    assert outcome["status"] == "infra_failed"
    assert outcome["lifecycle"] == "gateway_terminal"
    assert outcome["infra_reason"] == "failed"
    assert "reconcile_disposition" not in outcome
    frame = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert frame["reconciled"] is True
    assert frame["reconcile_source"] == "isolate_task_results"


def test_reconcile_task_without_isolate_root_has_no_disk_fallback(tmp_path):
    gateway_id = "gateway-task-10"

    def http(*_args, **_kwargs):
        raise ExecutorFailure("isolate gateway is down")

    _config_unused, executor, task_dir, checkpoint = _reconcile_fixture(
        tmp_path,
        gateway_id,
        {"gateway_task_id": gateway_id, "status": "running"},
        http_runner=http,
    )
    outcome = executor.reconcile_task(TaskSpec("arvo:1", "arvo"), task_dir, "attempt-1", checkpoint)
    assert outcome["reconcile_disposition"] == "undeliverable"
    assert outcome["lifecycle"] == "reconcile_blocked"


def test_reconcile_task_rejects_cached_result_of_a_different_task(tmp_path):
    """A cached terminal frame bound to another gateway task is an infra error."""
    gateway_id = "gateway-task-11"
    _config_unused, executor, task_dir, checkpoint = _reconcile_fixture(
        tmp_path,
        gateway_id,
        {
            "gateway_task_id": gateway_id,
            "status": "failed",
            "result": {"task_id": "gateway-task-foreign", "status": "failed"},
        },
    )
    outcome = executor.reconcile_task(TaskSpec("arvo:1", "arvo"), task_dir, "attempt-1", checkpoint)
    assert outcome["status"] == "infra_failed"
    assert outcome["lifecycle"] == "reconcile_blocked"
    assert outcome["reconcile_disposition"] == "undeliverable"
    assert "different task" in outcome["error"]
    # The checkpoint is left untouched so the mismatch stays auditable.
    frame = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert "reconciled" not in frame


def test_reconcile_task_delivers_cached_terminal_result(tmp_path):
    """A cached terminal frame bound to the checkpoint's task needs no poll."""
    gateway_id = "gateway-task-12"
    cached = {"task_id": gateway_id, "status": "failed", "error": "worker crashed"}
    _config_unused, executor, task_dir, checkpoint = _reconcile_fixture(
        tmp_path,
        gateway_id,
        {"gateway_task_id": gateway_id, "status": "failed", "result": cached},
    )
    outcome = executor.reconcile_task(TaskSpec("arvo:1", "arvo"), task_dir, "attempt-1", checkpoint)
    assert outcome["status"] == "infra_failed"
    assert outcome["lifecycle"] == "gateway_terminal"
    assert outcome["infra_reason"] == "failed"


def test_reconcile_task_polls_when_cached_frame_is_not_settled(tmp_path):
    """A non-settled cached frame is not authoritative; the gateway is polled."""
    gateway_id = "gateway-task-14"

    def http(method, url, **_kwargs):
        assert method == "GET"
        return {"task_id": gateway_id, "status": "running"}

    _config_unused, executor, task_dir, checkpoint = _reconcile_fixture(
        tmp_path,
        gateway_id,
        {
            "gateway_task_id": gateway_id,
            "status": "running",
            "result": {"task_id": gateway_id, "status": "running"},
        },
        http_runner=http,
    )
    outcome = executor.reconcile_task(TaskSpec("arvo:1", "arvo"), task_dir, "attempt-1", checkpoint)
    assert outcome["reconcile_disposition"] == "left_running"
    frame = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert frame["reconcile_source"] == "gateway_poll"


def test_reconcile_task_rejects_polled_terminal_result_with_empty_task_id(tmp_path):
    """A terminal poll frame without its task id is an infra error, not a delivery."""
    gateway_id = "gateway-task-15"

    def http(method, url, **_kwargs):
        assert method == "GET"
        return {"status": "failed"}

    _config_unused, executor, task_dir, checkpoint = _reconcile_fixture(
        tmp_path,
        gateway_id,
        {"gateway_task_id": gateway_id, "status": "running"},
        http_runner=http,
    )
    outcome = executor.reconcile_task(TaskSpec("arvo:1", "arvo"), task_dir, "attempt-1", checkpoint)
    assert outcome["status"] == "infra_failed"
    assert outcome["lifecycle"] == "reconcile_blocked"
    assert outcome["reconcile_disposition"] == "undeliverable"
    assert "no usable task id" in outcome["error"]
    # The id-less frame must not be cached into the checkpoint for a later pass.
    frame = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert "reconciled" not in frame


def test_reconcile_task_rejects_cached_terminal_result_with_empty_task_id(tmp_path):
    """A cached terminal frame without its task id is an infra error."""
    gateway_id = "gateway-task-16"
    _config_unused, executor, task_dir, checkpoint = _reconcile_fixture(
        tmp_path,
        gateway_id,
        {
            "gateway_task_id": gateway_id,
            "status": "failed",
            "result": {"status": "failed", "error": "worker crashed"},
        },
    )
    outcome = executor.reconcile_task(TaskSpec("arvo:1", "arvo"), task_dir, "attempt-1", checkpoint)
    assert outcome["status"] == "infra_failed"
    assert outcome["lifecycle"] == "reconcile_blocked"
    assert outcome["reconcile_disposition"] == "undeliverable"
    assert "different task" in outcome["error"]


def test_reconcile_task_left_running_tolerates_nonterminal_empty_task_id(tmp_path):
    """The exact-id gate is terminal-scoped: a running frame stays retryable."""
    gateway_id = "gateway-task-17"

    def http(method, url, **_kwargs):
        assert method == "GET"
        return {"status": "running"}

    _config_unused, executor, task_dir, checkpoint = _reconcile_fixture(
        tmp_path,
        gateway_id,
        {"gateway_task_id": gateway_id, "status": "running"},
        http_runner=http,
    )
    outcome = executor.reconcile_task(TaskSpec("arvo:1", "arvo"), task_dir, "attempt-1", checkpoint)
    assert outcome["reconcile_disposition"] == "left_running"
    assert outcome["lifecycle"] == "reconcile_pending"


def test_reconcile_task_defers_workspace_release_until_durable(tmp_path, monkeypatch):
    """reconcile_task keeps the adopted container; the launcher releases it."""
    gateway_id = "gateway-task-13"
    cached = {
        "task_id": gateway_id,
        "status": "completed",
        "cost_usd": 0.5,
        "cost_estimated": False,
        "cost_final": True,
    }
    _config_unused, executor, task_dir, checkpoint = _reconcile_fixture(
        tmp_path,
        gateway_id,
        {"gateway_task_id": gateway_id, "status": "completed", "result": cached},
    )
    adopted: list[str] = []
    cleaned: list[str] = []

    def fake_adopt(container_name):
        adopted.append(container_name)
        executor._task_containers[container_name] = "d" * 64
        return "d" * 64

    def fake_cleanup(name, task_id, attempt_id, report_path):
        cleaned.append(name)
        executor._task_containers.pop(name, None)
        return {"status": "verified", "ok": True}

    monkeypatch.setattr(executor, "_adopt_workspace_container", fake_adopt)
    monkeypatch.setattr(executor, "_cleanup_workspace_container", fake_cleanup)
    # A completed frame adopts its workspace only while the container is
    # still running; present it as running so the adoption path is exercised.
    monkeypatch.setattr(
        executor,
        "_inspect_optional",
        lambda kind, _name: {"State": {"Status": "running"}} if kind == "container" else None,
    )
    monkeypatch.setattr(
        executor,
        "_deliver_gateway_result",
        lambda *args, **kwargs: {"status": "completed", "lifecycle": "completed"},
    )
    task = TaskSpec("arvo:1", "arvo")
    outcome = executor.reconcile_task(task, task_dir, "attempt-1", checkpoint)
    assert outcome["status"] == "completed"
    assert adopted and not cleaned
    assert executor._task_containers  # adopted slot survives reconcile_task

    report = executor.release_reconciled_workspace(task, "attempt-1")
    assert report["ok"] is True
    assert cleaned == adopted
    assert executor._task_containers == {}
    # A left-running or never-adopted attempt has nothing to release.
    assert executor.release_reconciled_workspace(task, "attempt-1") is None


def _adopt_fixture(tmp_path, monkeypatch, *, server_labels, network_labels=None):
    server_id = "a" * 64
    network_id = "b" * 64
    settings = tmp_path / "settings_applied.json"
    settings.write_text(
        json.dumps({
            "OUROBOROS_MODEL": "deepseek/deepseek-v4-flash-0731",
            "OUROBOROS_OR_PROVIDER": {"allow_fallbacks": True, "require_parameters": True},
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("CYBERGYM_API_KEY", "test-cybergym-key")

    def commands(argv, **_kwargs):
        parts = list(argv)
        if "inspect" in parts and "container" in parts:
            return CommandResult(0, json.dumps([{
                "Id": server_id,
                "Config": {"Labels": dict(server_labels)},
                "State": {"Status": "running"},
            }]))
        if "inspect" in parts and "network" in parts:
            if network_labels is None:
                return CommandResult(1, "", "no such network")
            return CommandResult(0, json.dumps([{
                "Id": network_id,
                "Labels": dict(network_labels),
            }]))
        raise AssertionError(f"unexpected command: {parts}")

    config = _config(tmp_path, settings_path=settings, command_runner=commands)
    executor = CyberGymExecutor(config)
    (config.run_root / "sidecar_state.json").write_text(
        json.dumps({"server_id": server_id, "network_id": network_id}),
        encoding="utf-8",
    )
    monkeypatch.setattr(executor, "_wait_server", lambda *_args, **_kwargs: None)
    return executor, server_id, network_id


def test_adopt_campaign_registers_attested_resources_and_detaches(tmp_path, monkeypatch):
    executor, server_id, network_id = _adopt_fixture(
        tmp_path,
        monkeypatch,
        server_labels={"com.ouroboros.campaign": "test-campaign"},
        network_labels={"com.ouroboros.campaign": "test-campaign"},
    )
    report = executor.adopt_campaign()
    assert report["status"] == "adopted"
    assert report["ok"] is True
    assert executor.started is True
    assert executor.server_id == server_id
    assert executor.network_id == network_id

    cleanup = executor.close()
    assert cleanup["status"] == "detached"
    assert cleanup["adopted"] is True
    assert cleanup["server_id"] == server_id
    assert executor.started is False
    # Detach never removes the adopted campaign resources.
    assert executor.server_id == ""
    assert executor.network_id == ""


def test_adopt_campaign_rejects_foreign_server_container(tmp_path, monkeypatch):
    executor, _server_id, _network_id = _adopt_fixture(
        tmp_path,
        monkeypatch,
        server_labels={"com.ouroboros.campaign": "another-campaign"},
        network_labels={"com.ouroboros.campaign": "test-campaign"},
    )
    with pytest.raises(ExecutorFailure, match="ownership attestation"):
        executor.adopt_campaign()
    assert executor.started is False


def _custody_observed(config, name, container_id, *, status, campaign=None):
    """One ``docker inspect`` body for the custody-heal matrix."""
    return {
        "Id": container_id,
        "Name": "/" + name,
        "Config": {
            "Labels": {
                "com.ouroboros.campaign": campaign or config.campaign_id,
                "com.ouroboros.role": "workspace",
            }
        },
        "State": {"Status": status},
    }


def test_heal_drops_custody_entry_when_container_is_gone(tmp_path, monkeypatch):
    config = _config(tmp_path, provider_probe=False)
    executor = CyberGymExecutor(config)
    executor._unresolved_workspace_custody["cybergym-workspace-gone"] = "inspect failed"
    monkeypatch.setattr(executor, "_inspect_optional", lambda _kind, _name: None)

    executor._heal_unresolved_workspace_custody()  # noqa: SLF001 - heal seam

    assert executor._unresolved_workspace_custody == {}


def test_heal_removes_terminal_owned_container(tmp_path, monkeypatch):
    """A provably owned container in a removable terminal state is released."""
    config = _config(tmp_path, provider_probe=False)
    executor = CyberGymExecutor(config)
    name = "cybergym-workspace-stuck"
    container_id = "e" * 64
    executor._unresolved_workspace_custody[name] = "post-start inspect failed"
    executor._task_containers[name] = container_id
    observed = _custody_observed(config, name, container_id, status="created")
    removed: list[str] = []

    def fake_inspect(_kind, target):
        if target in removed:
            return None
        return observed

    def fake_docker(*args, timeout=60):
        assert args[0] == "rm"
        removed.append(args[-1])
        return CommandResult(0, "", "")

    monkeypatch.setattr(executor, "_inspect_optional", fake_inspect)
    monkeypatch.setattr(executor, "_docker", fake_docker)

    executor._heal_unresolved_workspace_custody()  # noqa: SLF001 - heal seam

    assert removed == [container_id]
    assert executor._unresolved_workspace_custody == {}
    assert name not in executor._task_containers


def test_heal_keeps_unproven_ownership_latched(tmp_path, monkeypatch):
    config = _config(tmp_path, provider_probe=False)
    executor = CyberGymExecutor(config)
    name = "cybergym-workspace-foreign"
    executor._unresolved_workspace_custody[name] = "post-start inspect failed"
    observed = _custody_observed(
        config, name, "2" * 64, status="exited", campaign="another-campaign"
    )

    def fake_docker(*args, timeout=60):
        raise AssertionError(f"no removal is allowed without ownership proof: {args}")

    monkeypatch.setattr(executor, "_inspect_optional", lambda _kind, _name: observed)
    monkeypatch.setattr(executor, "_docker", fake_docker)

    executor._heal_unresolved_workspace_custody()  # noqa: SLF001 - heal seam

    assert executor._unresolved_workspace_custody == {name: "post-start inspect failed"}


def test_heal_keeps_latch_when_daemon_is_unreadable(tmp_path, monkeypatch):
    config = _config(tmp_path, provider_probe=False)
    executor = CyberGymExecutor(config)
    name = "cybergym-workspace-unknown"
    executor._unresolved_workspace_custody[name] = "post-start inspect failed"

    def unreadable(_kind, _name):
        raise ExecutorFailure("docker inspect failed for container")

    monkeypatch.setattr(executor, "_inspect_optional", unreadable)

    executor._heal_unresolved_workspace_custody()  # noqa: SLF001 - heal seam

    assert executor._unresolved_workspace_custody == {name: "post-start inspect failed"}


@_requires_posix_mount_paths
def test_workspace_heals_terminal_custody_entry_instead_of_latching(tmp_path):
    """Run 20260907T233516Z: one stuck ``Created`` container poisoned 107 lanes.

    The next lane's ``_workspace`` re-inspects the recorded name, removes the
    provably owned terminal container, and proceeds with its own start
    instead of failing every later task on the poisoned latch.
    """
    config = _config(tmp_path)
    executor = CyberGymExecutor(config)
    executor.network_id = "network-id"
    stuck_name = "cybergym-workspace-stuck"
    stuck_id = "f" * 64
    executor._unresolved_workspace_custody[stuck_name] = "post-start inspect failed"
    stuck = _custody_observed(config, stuck_name, stuck_id, status="created")

    agent_id = "agent-" + "d" * 24
    plan = executor._task_network_plan("task-d", agent_id)  # noqa: SLF001
    name = "cybergym-workspace-" + agent_id
    container_id = "d" * 64
    observed = {
        "Id": container_id,
        "Name": "/" + name,
        "Config": {
            "Image": config.workspace_image_digest,
            "Labels": {
                "com.ouroboros.campaign": config.campaign_id,
                "com.ouroboros.role": "workspace",
                "com.ouroboros.agent_id": plan.opaque_agent_id,
            },
        },
        "NetworkSettings": {
            "Networks": {"cybergym-internal": {"NetworkID": executor.network_id}}
        },
    }

    removed: list[str] = []

    def command(argv, *, cwd=None, env=None, timeout=None):
        if "inspect" in argv and "container" in argv:
            target = argv[-1]
            if target in removed:
                return CommandResult(1, "", f"Error: No such container: {target}")
            if target in {stuck_name, stuck_id}:
                return CommandResult(0, json.dumps([stuck]), "")
            if target in {name, container_id}:
                return CommandResult(0, json.dumps([observed]), "")
        if "rm" in argv and stuck_id in argv:
            removed.extend([stuck_id, stuck_name])
            return CommandResult(0, "", "")
        if "run" in argv and name in argv:
            return CommandResult(0, container_id + "\n", "")
        raise AssertionError(argv)

    executor.config = dataclasses_replace(config, command_runner=command)

    started = executor._workspace(  # noqa: SLF001 - heal-through-start assertion
        type("Task", (), {"task_id": "task-d"})(),
        config.run_root / "task-d",
        plan,
    )

    assert started == name
    assert stuck_id in removed
    assert executor._unresolved_workspace_custody == {}
    assert executor._task_containers[name] == container_id


@_requires_posix_mount_paths
def test_workspace_signals_custody_while_it_is_genuinely_unresolved(tmp_path, monkeypatch):
    """A partially proven running container keeps the latch: the next lane
    raises the typed zero-send signal instead of starting beside it."""
    config = _config(tmp_path)
    executor = CyberGymExecutor(config)
    executor.network_id = "network-id"
    name = "cybergym-workspace-running"
    executor._unresolved_workspace_custody[name] = "post-start inspect failed"
    running = _custody_observed(config, name, "1" * 64, status="running")
    monkeypatch.setattr(executor, "_inspect_optional", lambda _kind, _name: running)

    agent_id = "agent-" + "e" * 24
    plan = executor._task_network_plan("task-e", agent_id)  # noqa: SLF001
    with pytest.raises(WorkspaceCustodyPending, match="workspace startup custody is unresolved") as excinfo:
        executor._workspace(  # noqa: SLF001 - latch assertion
            type("Task", (), {"task_id": "task-e"})(),
            config.run_root / "task-e",
            plan,
        )

    assert not isinstance(excinfo.value, ExecutorFailure)
    assert excinfo.value.unresolved == {name: "post-start inspect failed"}
    assert executor._unresolved_workspace_custody == {name: "post-start inspect failed"}


@_requires_posix_mount_paths
def test_workspace_start_error_recovers_name_custody_by_inspect(tmp_path):
    config = _config(tmp_path)
    executor = CyberGymExecutor(config)
    executor.network_id = "network-id"
    agent_id = "agent-" + "c" * 24
    plan = executor._task_network_plan("task-c", agent_id)
    name = "cybergym-workspace-" + agent_id
    container_id = "c" * 64
    observed = {
        "Id": container_id,
        "Name": "/" + name,
        "Config": {
            "Image": config.workspace_image_digest,
            "Labels": {
                "com.ouroboros.campaign": config.campaign_id,
                "com.ouroboros.role": "workspace",
                "com.ouroboros.agent_id": plan.opaque_agent_id,
            },
        },
        "NetworkSettings": {
            "Networks": {"cybergym-internal": {"NetworkID": executor.network_id}}
        },
    }

    removed = []

    def command(argv, *, cwd=None, env=None, timeout=None):
        if "inspect" in argv and "container" in argv:
            target = argv[-1]
            if target in removed:
                return CommandResult(1, "", f"Error: No such container: {target}")
            if target in {name, container_id}:
                return CommandResult(0, json.dumps([observed]), "")
        if "rm" in argv and container_id in argv:
            removed.append(container_id)
            removed.append(name)
            return CommandResult(0, "", "")
        if "run" in argv and name in argv:
            raise ExecutorFailure("docker run transport timeout")
        raise AssertionError(argv)

    executor.config = dataclasses_replace(config, command_runner=command)
    with pytest.raises(ExecutorFailure, match="transport timeout"):
        executor._workspace(  # noqa: SLF001 - startup custody assertion
            type("Task", (), {"task_id": "task-c"})(),
            config.run_root / "task-c",
            plan,
        )
    assert name not in executor._task_containers
    assert name not in executor._unresolved_workspace_custody
    assert container_id in removed
    assert not executor._workspace_starting


class _FakeDockerDaemon:
    """In-memory ``docker`` for startup custody: ``run``/``inspect``/``rm``.

    ``timeout_starts`` runs create their container and then report a timed-out
    ``docker run``; that residual answers ``residual_unreadable`` inspects with
    a daemon error first.  ``rm_mode`` is ``remove``, ``fail`` or ``persist``.
    """

    def __init__(self, config, *, timeout_starts=0, residual_unreadable=0):
        self.config = config
        self.timeout_starts = timeout_starts
        self.residual_unreadable = residual_unreadable
        self.unreadable: dict[str, int] = {}
        self.containers: dict[str, dict] = {}
        self.residuals: list[str] = []
        self.rm_calls: list[str] = []
        self.removed: list[str] = []
        self.rm_mode = "remove"

    def add(self, name, *, labels, status="running", image="", network_id="network-id"):
        container_id = hashlib.sha256(name.encode()).hexdigest()
        self.containers[container_id] = {
            "Id": container_id,
            "Name": "/" + name,
            "Config": {"Image": image or self.config.workspace_image_digest, "Labels": dict(labels)},
            "State": {"Status": status},
            "NetworkSettings": {"Networks": {"cybergym-internal": {"NetworkID": network_id}}},
        }
        return container_id

    def __call__(self, argv, *, cwd=None, env=None, timeout=None):
        command = list(argv[3:])
        if command[0] == "run":
            name = command[command.index("--name") + 1]
            labels = dict(value.split("=", 1) for flag, value in zip(command, command[1:]) if flag == "--label")
            container_id = self.add(name, labels=labels)
            if self.timeout_starts:
                self.timeout_starts -= 1
                self.residuals.append(name)
                self.unreadable[name] = self.residual_unreadable
                return CommandResult(124, "", "timeout")
            return CommandResult(0, container_id + "\n", "")
        found = next(
            (item for item in self.containers.values() if command[-1] in {item["Id"], item["Name"][1:]}),
            None,
        )
        if command[:2] == ["container", "inspect"]:
            name = found["Name"][1:] if found else ""
            if self.unreadable.get(name, 0) > 0:
                self.unreadable[name] -= 1
                return CommandResult(1, "", "Error response from daemon: context deadline exceeded")
            if found is None:
                return CommandResult(1, "", f"Error: No such container: {command[-1]}")
            return CommandResult(0, json.dumps([found]), "")
        if command[:2] == ["rm", "--force"]:
            self.rm_calls.append(command[-1])
            if self.rm_mode == "fail":
                return CommandResult(2, "", "Error response from daemon: removal failed")
            if found is not None and self.rm_mode == "remove":
                self.removed.append(self.containers.pop(found["Id"])["Id"])
            return CommandResult(0, "", "")
        raise AssertionError(argv)


def _custody_campaign(tmp_path, monkeypatch, *, task_ids=("arvo:1", "arvo:2", "arvo:3"), **daemon):
    """A serial campaign through the real ``run_task``/``_workspace`` on a fake
    daemon and a fake dispatch clock; the gateway is stubbed to completion."""
    from devtools.benchmarks.cybergym import cybergym_adapter, cybergym_dispatch
    from tests.test_cybergym_dispatch import _completed, _PausingClock

    config = _config(tmp_path, provider_probe=False)
    docker = _FakeDockerDaemon(config, **daemon)
    executor = CyberGymExecutor(dataclasses_replace(config, command_runner=docker))
    executor.network_id = "network-id"
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(executor, "start", lambda: None)
    monkeypatch.setattr(executor, "_generate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor_module, "_install_workspace_backend_alias", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor, "_task_body", lambda *_args, **_kwargs: {"task_id": "body"})
    monkeypatch.setattr(
        executor, "_gateway_wait",
        lambda _body, _checkpoint, **owner: sent.append((owner["task_id"], owner["attempt_id"])) or {},
    )
    monkeypatch.setattr(
        executor, "_deliver_gateway_result", lambda task, task_dir, *_args, **_kwargs: _completed(task, task_dir),
    )
    clock = _PausingClock()
    real = cybergym_dispatch.run_dispatched
    monkeypatch.setattr(
        cybergym_adapter, "run_dispatched",
        lambda tasks, run_one, **kwargs: real(tasks, run_one, **kwargs, sleep=clock.sleep, clock=clock.monotonic),
    )

    def run():
        return run_campaign(
            list(task_ids), run_root=config.run_root, executor=executor.run_task,
            estimated_cost_usd=1, budget_cap_usd=10,
        )

    return executor, docker, clock, sent, run


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@_requires_posix_mount_paths
def test_sibling_start_latch_requeues_collateral_row_free_and_resumes_after_heal(tmp_path, monkeypatch):
    """The 2026-09-13 campaign turned 11 failed starts into 12 innocent infra rows.

    The failed start keeps its honest row.  The collateral attempt releases its
    claim before a row-free requeue, admission pauses, and the healer's later
    removal of the now-readable, fully attested residual resumes dispatch with
    a fresh attempt identity.
    """
    executor, docker, clock, sent, run = _custody_campaign(
        tmp_path, monkeypatch, timeout_starts=1, residual_unreadable=3,
    )
    rows = run()

    assert [(row["task_id"], row["status"]) for row in rows] == [
        ("arvo:1", "infra_failed"), ("arvo:2", "completed"), ("arvo:3", "completed"),
    ]
    assert rows[0]["lifecycle"] == "pre_gateway_setup_failed"
    events = _jsonl(executor.config.run_root / "claims.jsonl")
    claims = [event for event in events if event.get("task_id") == "arvo:2"]
    release = next(event for event in events if event["event"] == "release")
    assert (release["attempt_id"], release["reason"]) == (claims[0]["attempt_id"], "workspace_custody_pending")
    assert events.index(release) < events.index(claims[1])
    assert sent == [("arvo:2", claims[1]["attempt_id"]), ("arvo:3", sent[1][1])]
    assert [event["event"] for event in _jsonl(executor.config.run_root / "dispatch_events.jsonl")] == [
        "workspace_custody_pause", "workspace_custody_probe_failed", "workspace_custody_resume",
    ]
    assert clock.slept == [30.0, 30.0]
    residual_id = hashlib.sha256(docker.residuals[0].encode()).hexdigest()
    assert docker.removed == [residual_id]
    assert executor._unresolved_workspace_custody == {}


@_requires_posix_mount_paths
def test_unhealable_start_latch_stops_campaign_row_free_after_custody_budget(tmp_path, monkeypatch):
    executor, docker, clock, sent, run = _custody_campaign(
        tmp_path, monkeypatch, timeout_starts=1, residual_unreadable=10**6,
    )
    with pytest.raises(CyberGymError) as excinfo:
        run()

    stop = excinfo.value.as_dict()
    assert stop["outcome"] == "workspace_custody_timeout"
    assert (stop["dispatched_rows"], stop["remaining_task_ids"]) == (1, ["arvo:2", "arvo:3"])
    assert stop["pause"]["pauses"] == [{"paused_sec": 300.0, "failed_probes": 10}]
    assert clock.slept == [30.0] * 10
    run_root = executor.config.run_root
    assert [row["task_id"] for row in _jsonl(run_root / "result_index.jsonl")] == ["arvo:1"]
    events = _jsonl(run_root / "claims.jsonl")
    assert [event.get("task_id") for event in events if event["event"] == "claim"] == ["arvo:1", "arvo:2"]
    assert sum(event["event"] == "release" for event in events) == 1
    assert sent == [] and docker.rm_calls == []
    report = executor.close()
    assert report["status"] == "custody_pending"
    assert list(report["workspace_custody_unresolved"]) == docker.residuals


@_requires_posix_mount_paths
def test_collateral_claim_release_failure_is_fatal_not_a_row(tmp_path, monkeypatch):
    executor, _docker, clock, _sent, run = _custody_campaign(
        tmp_path, monkeypatch, task_ids=("arvo:1", "arvo:2"), timeout_starts=1, residual_unreadable=10**6,
    )

    def refuse(_self, attempt_id, **_kwargs):
        raise LedgerError(f"claims ledger is read-only: {attempt_id}")

    monkeypatch.setattr(BudgetLedger, "release", refuse)
    with pytest.raises(LedgerError, match="read-only"):
        run()

    run_root = executor.config.run_root
    assert [row["task_id"] for row in _jsonl(run_root / "result_index.jsonl")] == ["arvo:1"]
    ledger = BudgetLedger(run_root / "claims.jsonl", cap_usd=10)
    claim = next(event for event in ledger.events() if event.get("task_id") == "arvo:2")
    assert ledger.attempt_state(claim["attempt_id"]) == "reserved"
    assert clock.slept == []


def test_attestation_latch_is_a_zero_send_signal_that_reaps_its_own_workspace(tmp_path, monkeypatch):
    config = _config(
        tmp_path, provider_probe=True, expected_data_sha256="a" * 64, expected_binary_sha256="b" * 64,
    )
    executor = CyberGymExecutor(config)
    monkeypatch.setenv("CYBERGYM_API_KEY", "fixture-cybergym-key")
    cleaned: list[str] = []

    def workspace_while_sibling_fails(task, task_dir, plan):
        name = f"cybergym-workspace-{plan.opaque_agent_id}"
        executor._task_containers[name] = "d" * 64
        executor._unresolved_workspace_custody["cybergym-workspace-sibling"] = "run timed out"
        return name

    monkeypatch.setattr(executor, "start", lambda: None)
    monkeypatch.setattr(executor, "_generate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor_module, "_install_workspace_backend_alias", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor, "_workspace", workspace_while_sibling_fails)
    monkeypatch.setattr(executor, "_cleanup_workspace_container", lambda name, *_args: cleaned.append(name))
    monkeypatch.setattr(executor, "_gateway_wait", lambda *_args, **_kwargs: pytest.fail("reached the gateway"))
    task = TaskSpec("arvo:2", "arvo", metadata={"attempt_id": "attempt-b"})

    with pytest.raises(CyberGymError) as excinfo:
        executor.run_task(task, config.run_root / "arvo__2")
    assert type(excinfo.value).__name__ == "WorkspaceCustodyPending"
    assert excinfo.value.unresolved == {"cybergym-workspace-sibling": "run timed out"}
    assert len(cleaned) == 1 and cleaned[0].startswith("cybergym-workspace-agent-")


@_requires_posix_mount_paths
def test_attestation_waiting_on_a_failing_sibling_start_signals_custody(tmp_path):
    """A lane blocked on a sibling's pending ``docker run`` wakes to its latch."""
    config = _config(tmp_path, provider_probe=False)
    docker = _FakeDockerDaemon(config, timeout_starts=1, residual_unreadable=10**6)
    sibling_running, release_sibling, waiter_done = threading.Event(), threading.Event(), threading.Event()

    def blocking(argv, **kwargs):
        if "run" in argv:
            sibling_running.set()
            assert release_sibling.wait(5), "sibling start barrier was not released"
        return docker(argv, **kwargs)

    executor = CyberGymExecutor(dataclasses_replace(config, command_runner=blocking))
    executor.network_id = "network-id"
    outcome: dict[str, BaseException] = {}

    def attest():
        try:
            executor._attest_runtime(  # noqa: SLF001 - concurrency seam
                TaskSpec("arvo:1", "arvo"), "attempt-a", None, "cybergym-workspace-own", "key",
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            outcome["waiter"] = exc
        finally:
            waiter_done.set()

    def start_sibling():
        try:
            executor._workspace(  # noqa: SLF001 - concurrency seam
                TaskSpec("arvo:2", "arvo"), config.run_root / "arvo__2",
                executor._task_network_plan("arvo:2", "agent-" + "b" * 24),
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            outcome["sibling"] = exc

    sibling = threading.Thread(target=start_sibling)
    waiter = threading.Thread(target=attest)
    sibling.start()
    try:
        assert sibling_running.wait(5)
        waiter.start()
        assert not waiter_done.wait(0.1), "attestation crossed a pending sibling start"
    finally:
        release_sibling.set()
        sibling.join(5)
        waiter.join(5)
    assert isinstance(outcome["sibling"], ExecutorFailure)
    assert type(outcome["waiter"]).__name__ == "WorkspaceCustodyPending"
    assert list(outcome["waiter"].unresolved) == docker.residuals


@pytest.mark.parametrize(
    ("case", "removed"),
    [
        ("attested", True),
        ("agent_label", False),
        ("network", False),
        ("image", False),
        ("gateway_registered", False),
        ("starting", False),
        ("rm_fails", False),
        ("rm_leaves_object", False),
    ],
)
def test_heal_removes_a_running_residual_only_with_full_startup_attestation(tmp_path, case, removed):
    """The healer gains exactly the authority of the immediate
    failed-start cleanup; anything less proven, an in-flight start included,
    stays latched with a receipt."""
    config = _config(tmp_path, provider_probe=False)
    docker = _FakeDockerDaemon(config)
    executor = CyberGymExecutor(dataclasses_replace(config, command_runner=docker))
    executor.network_id = "network-id"
    agent_id = "agent-" + "a" * 24
    name = "cybergym-workspace-" + agent_id
    labels = {
        "com.ouroboros.campaign": config.campaign_id,
        "com.ouroboros.role": "workspace",
        "com.ouroboros.agent_id": "agent-" + "b" * 24 if case == "agent_label" else agent_id,
    }
    container_id = docker.add(
        name,
        labels=labels,
        image="sha256:" + "9" * 64 if case == "image" else "",
        network_id="foreign-network" if case == "network" else "network-id",
    )
    executor._gateway_attempts = {"g1": {"workspace_name": name}} if case == "gateway_registered" else {}
    executor._workspace_starting = {name: 1} if case == "starting" else {}
    docker.rm_mode = {"rm_fails": "fail", "rm_leaves_object": "persist"}.get(case, "remove")
    executor._unresolved_workspace_custody[name] = "run timed out; name inspect failed"

    executor._heal_unresolved_workspace_custody()  # noqa: SLF001 - heal seam

    assert docker.removed == ([container_id] if removed else [])
    assert bool(docker.rm_calls) is (case in {"attested", "rm_fails", "rm_leaves_object"})
    assert (name in executor._unresolved_workspace_custody) is not removed
    receipt = json.loads(
        (config.run_root / "workspaces" / f"{name}.startup_custody.json").read_text(encoding="utf-8")
    )
    assert (receipt["container_name"], receipt["status"]) == (name, "resolved" if removed else "retained")
    assert receipt["observation"]


def test_concurrent_receipts_on_one_path_both_replace(tmp_path, monkeypatch):
    """Two lanes publishing one receipt shared a single staging file: the first
    ``os.replace`` consumed it and the second raised ``FileNotFoundError``.  The
    barrier holds both writers at their replace, so they meet every time."""
    target = tmp_path / "workspaces" / "cybergym-workspace-race.startup_custody.json"
    real_replace, at_replace, errors = os.replace, threading.Barrier(2, timeout=10), []

    def barriered(src, dst, *args, **kwargs):
        if pathlib.Path(dst) == target:
            at_replace.wait()
        return real_replace(src, dst, *args, **kwargs)

    def write(status):
        try:
            docker_module._write_json(target, {"status": status})  # noqa: SLF001 - receipt seam
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    monkeypatch.setattr(os, "replace", barriered)
    writers = [threading.Thread(target=write, args=(status,)) for status in ("resolved", "retained")]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(15)

    assert errors == []
    assert json.loads(target.read_text(encoding="utf-8"))["status"] in {"resolved", "retained"}
    assert [item.name for item in target.parent.iterdir()] == [target.name]


def test_concurrent_healer_passes_are_serialized_per_executor(tmp_path, monkeypatch):
    """A dispatch probe and a ``_workspace`` lane can heal at the same moment.
    Two passes over one latched name would remove the same container twice and
    land a stale ``retained`` receipt after the newer ``resolved`` one, so one
    body runs at a time and the second pass sees the first's cleared latch."""
    config = _config(tmp_path, provider_probe=False)
    executor = CyberGymExecutor(config)
    name = "cybergym-workspace-race"
    executor._unresolved_workspace_custody[name] = "run timed out; name inspect failed"
    guard, inside, release = threading.Lock(), threading.Event(), threading.Event()
    depth, peak, seen, errors = 0, 0, [], []

    def observed(container_name):
        nonlocal depth, peak
        with guard:
            depth += 1
            peak = max(peak, depth)
            seen.append(container_name)
            first = len(seen) == 1
        if first:
            inside.set()
            release.wait(10)
        with guard:
            depth -= 1
        return True, "absent"

    def heal():
        try:
            executor._heal_unresolved_workspace_custody()  # noqa: SLF001 - heal seam
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    monkeypatch.setattr(executor, "_heal_workspace_name", observed)
    passes = [threading.Thread(target=heal), threading.Thread(target=heal)]
    passes[0].start()
    assert inside.wait(10), "the first healer pass never entered its body"
    passes[1].start()
    passes[1].join(0.2)
    release.set()
    for healer in passes:
        healer.join(15)

    assert (peak, seen, errors) == (1, [name], [])
    assert executor._unresolved_workspace_custody == {}


def test_resolved_custody_clears_only_after_its_receipt_is_written(tmp_path, monkeypatch):
    """The promised receipt is the reason the latch may be dropped.  Clearing
    first let a failed ``_write_json`` leave the next probe an empty latch and no
    durable observation, so admission resumed on a promise nobody kept."""
    config = _config(tmp_path, provider_probe=False)
    executor = CyberGymExecutor(config)
    name, reason = "cybergym-workspace-absent", "run timed out; name inspect failed"
    executor._unresolved_workspace_custody[name] = reason
    executor._task_containers[name] = "a" * 64
    monkeypatch.setattr(executor, "_inspect_optional", lambda _kind, _target: None)
    receipt = config.run_root / "workspaces" / f"{name}.startup_custody.json"
    real_write, attempts = docker_module._write_json, []

    def flaky(path, value):
        attempts.append(path)
        if len(attempts) == 1:
            raise OSError("No space left on device")
        real_write(path, value)

    monkeypatch.setattr(docker_module, "_write_json", flaky)
    executor._heal_unresolved_workspace_custody()  # noqa: SLF001 - heal seam

    assert (executor._unresolved_workspace_custody, receipt.exists()) == ({name: reason}, False)
    assert executor._task_containers == {name: "a" * 64}

    executor._heal_unresolved_workspace_custody()  # noqa: SLF001 - heal seam

    assert (executor._unresolved_workspace_custody, executor._task_containers) == ({}, {})
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "resolved"
