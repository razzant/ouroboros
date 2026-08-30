"""CyberGym Docker-runtime custody tests: workspace slots, network lifecycle.

Split from the executor suites along the container-machinery seam: these tests
exercise ``_workspace``/``_network`` failure custody and the pool-slot release
contract with injected command runners — no Docker daemon, upstream package,
or provider credential is used.
"""

from __future__ import annotations

import json

import pytest

from devtools.benchmarks.cybergym import cybergym_executor as executor_module
from devtools.benchmarks.cybergym.cybergym_adapter import (
    BudgetLedger,
    run_campaign,
)
from devtools.benchmarks.cybergym.cybergym_executor import (
    CommandResult,
    CyberGymExecutor,
    ExecutorFailure,
)
from tests.test_cybergym_executor import _config, dataclasses_replace


def test_post_create_timeout_releases_workspace_slot(tmp_path, monkeypatch):
    """A post-create timeout must rm the workspace so the next row can claim."""
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
    assert len(cleaned) == 2
    assert executor._task_containers == {}
    projection = BudgetLedger(config.run_root / "claims.jsonl", cap_usd=5).projection()
    assert projection.reserved_usd == pytest.approx(0)
    assert projection.unresolved_upper_bound_usd == pytest.approx(0)
    assert projection.can_dispatch is True


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
