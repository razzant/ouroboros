"""CyberGym Docker-runtime custody tests: workspace slots, network lifecycle.

Split from the executor suites along the container-machinery seam: these tests
exercise ``_workspace``/``_network`` failure custody and the pool-slot release
contract with injected command runners — no Docker daemon, upstream package,
or provider credential is used.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from devtools.benchmarks.cybergym import cybergym_executor as executor_module
from devtools.benchmarks.cybergym.cybergym_adapter import (
    BudgetLedger,
    TaskSpec,
    run_campaign,
)
from devtools.benchmarks.cybergym.cybergym_executor import (
    CommandResult,
    CyberGymExecutor,
    ExecutorFailure,
)
from tests.test_cybergym_executor import _config, dataclasses_replace


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
    assert projection.unresolved_upper_bound_usd == pytest.approx(2)
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
