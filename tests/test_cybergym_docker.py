"""CyberGym Docker-runtime custody tests: workspace slots, network lifecycle.

Split from the executor suites along the container-machinery seam: these tests
exercise ``_workspace``/``_network`` failure custody and the pool-slot release
contract with injected command runners — no Docker daemon, upstream package,
or provider credential is used.
"""

from __future__ import annotations

import pytest

from devtools.benchmarks.cybergym import cybergym_executor as executor_module
from devtools.benchmarks.cybergym.cybergym_adapter import (
    BudgetLedger,
    run_campaign,
)
from devtools.benchmarks.cybergym.cybergym_executor import (
    CyberGymExecutor,
    ExecutorFailure,
)
from tests.test_cybergym_executor import _config


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
