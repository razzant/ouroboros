"""Continue keeps the same work's recorded limits, grants and acceptance conditions.

``_successor_task`` rebuilt the successor without the interrupted root's withheld tools,
protected artifacts, resource map, acceptance claims and workspace executor, so a Continue
dropped constraints (a tool the work may not call became callable, commands left their
executor) as well as its criteria. Each test drives the real queue admission, the worker's
assignment and terminal handling, ``admit_continuation`` and the registry's own dispatch
readers. Nothing is added, the predecessor's identity (lineage, objective, source) is not
copied, and no owner source is created: the opening rules stay ``owner_sources``'s.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_continue_execution_budget import _continue, _settle_by_task_done, _start

WITHHELD = ["web_search"]
POLICY = {"protected_artifacts": [{"id": "golden", "role": "black_box_reference", "paths": ["golden/"]}]}
RESOURCES = {"network": False}
CLAIMS = [{"id": "fixed", "claim": "The crash no longer reproduces", "surface": "", "support": "",
           "priority": "must"}]
EXECUTOR = {"type": "docker", "id": "box", "network": "none", "workspace_host_path": "/work",
            "workspace_backend_path": "/app", "container_name": "box",
            "path_mappings": [{"host_path": "/work", "backend_path": "/app"}]}


@pytest.fixture(autouse=True)
def _isolated_projects_root(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path_factory.mktemp("projects_root")))


@pytest.fixture
def pool(tmp_path, monkeypatch):
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda task: True)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **_k: True)
    return SimpleNamespace(root=tmp_path, queue=queue, state=state, workers=workers, mp=monkeypatch)


def _admit_root(pool, task_id="limits-1", *, limited=True):
    """A root admitted as the API door admits one: the policies top-level and in metadata,
    the executor in metadata; the queue builds its contract."""
    from ouroboros.task_results import write_task_result

    origin = {"origin_message_text": "Find the crash without the web",
              "origin_message_ref": {"chat_id": 7, "client_message_id": f"m-{task_id}"}}
    limits = {"disabled_tools": WITHHELD, "resource_policy": POLICY, "allowed_resources": RESOURCES,
              "acceptance_claims": CLAIMS} if limited else {}
    metadata = {**limits, **({"executor_ref": EXECUTOR} if limited else {})}
    write_task_result(pool.root, task_id, "scheduled", chat_id=7, root_task_id=task_id, title="Find the crash",
                      metadata=dict(metadata), **origin)
    admitted = pool.queue.enqueue_task({"id": task_id, "type": "task", "chat_id": 7, "root_task_id": task_id,
                                        "text": "Find the crash", "metadata": dict(metadata), **limits, **origin})
    assert not admitted.get("_admission_blocked"), admitted
    return admitted


def _registry_view(row):
    """The registry's own readers over the task metadata the worker builds from this row."""
    from ouroboros.tools.registry_guards import _disabled_tools, _resource_allowed

    ctx = SimpleNamespace(task_metadata=dict(row["metadata"]))
    return _disabled_tools(ctx), _resource_allowed(ctx, "network")


def _assert_same_limits(row, successor_id):
    contract = row["task_contract"]
    assert set(WITHHELD) <= set(contract["disabled_tools"])
    assert contract["resource_policy"] == POLICY and contract["allowed_resources"] == RESOURCES
    assert contract["acceptance_claims"] == CLAIMS
    assert row["metadata"]["executor_ref"] == EXECUTOR
    # The successor's own contract: its lineage and objective, never the predecessor's identity.
    assert contract["lineage"]["root_task_id"] == successor_id and contract["objective"].startswith("[CONTINUE]")
    withheld, network = _registry_view(row)
    assert "web_search" in withheld and network is False


def test_continue_keeps_the_recorded_limits_through_admission_replay_and_a_never_started_successor(pool):
    from ouroboros.task_results import load_task_result
    from tests.test_restart_retention import _pool_events

    _admit_root(pool)
    _start(pool, "limits-1", executed=60)
    _settle_by_task_done(pool, "limits-1")

    first, _binding = _continue(pool, "limits-1")

    _assert_same_limits(first, first["id"])
    stored = load_task_result(pool.root, first["id"])
    assert stored["metadata"]["executor_ref"] == EXECUTOR
    assert stored["metadata"]["task_contract"]["acceptance_claims"] == CLAIMS
    # The same nonce replays the recorded admission; nothing is admitted twice.
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE

    replay = admit_continuation("limits-1", action_nonce=NONCE)
    assert replay["ok"] is True and replay["replay"] is True and replay["successor_task_id"] == first["id"]
    # The successor never starts: its scheduled result alone hands the same limits on.
    _pool_events(pool.workers, pool.mp)
    pool.mp.setattr(pool.workers, "WORKERS", {})
    assert pool.workers.kill_workers(terminal_status="cancelled", result_reason="server shutdown",
                                     stop_source="server_shutdown")

    second, _binding2 = _continue(pool, first["id"])

    _assert_same_limits(second, second["id"])


def test_the_successors_worker_row_carries_them_to_the_registry(pool):
    _admit_root(pool)
    _start(pool, "limits-1", executed=60)
    _settle_by_task_done(pool, "limits-1")
    row, _binding = _continue(pool, "limits-1")

    sent, _meta = _start(pool, row["id"], executed=0)

    _assert_same_limits(sent, row["id"])


def test_a_root_without_limits_gains_none(pool):
    _admit_root(pool, "plain-1", limited=False)
    _start(pool, "plain-1", executed=10)
    _settle_by_task_done(pool, "plain-1")

    row, _binding = _continue(pool, "plain-1")

    contract = row["task_contract"]
    assert contract["disabled_tools"] == [] and contract["resource_policy"] == {}
    assert contract["allowed_resources"] == {} and contract["acceptance_claims"] == []
    assert contract["delegation_budget"]["may_delegate"] is True and contract["budget_profile"]["cost_hard_stop_pct"] is None
    assert "executor_ref" not in row["metadata"] and "executor_ref" not in row
    withheld, network = _registry_view(row)
    assert "web_search" not in withheld and network is True


def test_a_wake_successor_keeps_its_recorded_list_and_still_derives_its_level(tmp_path, monkeypatch):
    """The carried list rides the contract, so the level's own derivation onto metadata still
    runs: the registry unions both, never fewer withheld tools than either."""
    from ouroboros.consciousness_authority import disabled_tools_for
    from supervisor.continuation_admission import admit_continuation
    from tests.test_continue_wake_opening import _successor, _wake_root
    from tests.test_owner_continue import NONCE

    workers, metadata = _wake_root(tmp_path, monkeypatch)

    row = _successor(workers, admit_continuation("wake-1", action_nonce=NONCE))

    assert row["metadata"]["disabled_tools"] == disabled_tools_for(metadata["consciousness_autonomy"])
    assert set(metadata["disabled_tools"]) <= set(row["task_contract"]["disabled_tools"])
    withheld, _network = _registry_view(row)
    assert set(metadata["disabled_tools"]) <= withheld
