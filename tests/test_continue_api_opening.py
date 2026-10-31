"""Owner 30 Oct (A): Continue of a root the task API or the CLI opened, from the request it saved.

The settled card offered Continue on such an interrupted root, and the admission then refused it
as ``owner_source_missing`` (``original_owner_message_text``): the API door records no owner
message, it records its own receipt (``api_admission``) beside the request. Each test opens the
root through the real door (``POST /api/tasks``, the CLI's ``run`` through it), then drives the
real assignment, the terminal writers and ``admit_continuation``. The saved request is the
caller's words under the channel it declared: never owner words, an owner ingress or a new check
of who called. The same work keeps its limits, criteria, answer protocol and closed execution,
once along a chain; Stop, a spent budget and an unreadable request still refuse.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_continue_execution_budget import _close, _continue, _settle_by_task_done, _start
from tests.test_continue_wake_opening import _no_owner_words
from tests.test_root_effort_ingress import _pool_ready

DESCRIPTION = "Find why the nightly import drops rows"
CONTEXT = "The importer lives in tools/import.py"
EXPECTED = "A patch and the failing input"
CONSTRAINTS = "Do not touch the schema"
WITHHELD = ["web_search"]
RESOURCES = {"network": False}
CLAIMS = [{"id": "fixed", "claim": "The dropped rows are imported", "surface": "", "support": "",
           "priority": "must"}]
BUDGET = 900


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


def _door(pool):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.tasks import api_tasks_create

    app = Starlette(routes=[Route("/api/tasks", api_tasks_create, methods=["POST"])])
    app.state.drive_root, app.state.repo_dir = pool.root, pool.root / "repo"
    (pool.root / "repo").mkdir(exist_ok=True)
    return TestClient(app)


def _api_root(pool, *, cli=False, budget=BUDGET, **extra):
    """A root the door admits: the request's words, its policies and an explicit execution budget (``extra``: more
    request fields)."""
    _pool_ready(pool.mp, pool.workers)
    client = _door(pool)
    if cli:
        from ouroboros import cli as cli_module

        created = []

        class GatewayClient:
            def request(self, method, path, body=None):
                response = client.post(path, json=body)
                assert (method, response.status_code) == ("POST", 200), response.text
                created.append(response.json())
                return response.json()

        pool.mp.setattr(cli_module, "_client", lambda *_a, **_k: GatewayClient())
        assert cli_module.main(["run", "--detach", "--execution-timeout", str(budget), "--disable-tools",
                                ",".join(WITHHELD), DESCRIPTION]) == 0
        task_id = created[0]["task_id"]
    else:
        response = client.post("/api/tasks", json={
            "description": DESCRIPTION, "context": CONTEXT, "expected_output": EXPECTED,
            "constraints": CONSTRAINTS, "disabled_tools": WITHHELD, "allowed_resources": RESOURCES,
            "acceptance_claims": CLAIMS, "answer_protocol": "final_answer_line", "execution_timeout_sec": budget,
            **extra})
        assert response.status_code == 200, response.text
        task_id = response.json()["task_id"]
    pool.mp.setattr(pool.workers, "WORKERS", {})  # the door's ready slot; ``_start`` installs the real one
    return task_id


def _interrupted(pool, *, cli=False, executed=700, budget=BUDGET, **extra):
    task_id = _api_root(pool, cli=cli, budget=budget, **extra)
    _start(pool, task_id, executed=executed)
    _settle_by_task_done(pool, task_id)
    return task_id


def _opening(row):
    return row["metadata"]["continuation"]["owner_sources"]["api"]


def test_an_api_root_continues_from_its_saved_request_under_the_same_contract(pool):
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_results import load_task_result

    task_id = _interrupted(pool)
    recorded = load_task_result(pool.root, task_id)
    assert recorded["api_admission"]["status"] == "accepted"
    assert continuation_offer(recorded, task_id)["eligible"] is True
    closed = _close(pool, task_id)

    row, binding = _continue(pool, task_id)

    assert _opening(row) == {"source": "api_request", "task_id": task_id, "channel": "api_task",
                             "description": DESCRIPTION, "context": CONTEXT, "expected_output": EXPECTED,
                             "constraints": CONSTRAINTS}
    for words in (DESCRIPTION, CONTEXT, EXPECTED, CONSTRAINTS):
        assert words in row["text"]
    assert "the caller's words, not owner words" in row["text"]
    _no_owner_words(row)
    # The same work: its withheld tool, resources, criteria and answer protocol, its closed execution.
    contract = row["task_contract"]
    assert set(WITHHELD) <= set(contract["disabled_tools"]) and contract["allowed_resources"] == RESOURCES
    assert contract["acceptance_claims"] == CLAIMS and contract["answer_protocol"] == "final_answer_line"
    assert binding["execution_timeout_sec"] == BUDGET and binding["execution_consumed_sec"] == closed["sec"]
    assert row["_execution_handoff_sec"] == closed["sec"] == pytest.approx(700, abs=30)


def test_a_cli_root_names_its_declared_channel_and_a_never_started_successor_hands_the_facts_on_once(pool):
    from tests.test_restart_retention import _pool_events

    task_id = _interrupted(pool, cli=True)
    closed = _close(pool, task_id)
    first, _binding = _continue(pool, task_id)
    assert _opening(first) == {"source": "api_request", "task_id": task_id, "channel": "cli",
                               "description": DESCRIPTION}
    kept = copy.deepcopy(_opening(first))
    # The successor never starts: its scheduled result alone hands the same facts on.
    _pool_events(pool.workers, pool.mp)
    pool.mp.setattr(pool.workers, "WORKERS", {})
    assert pool.workers.kill_workers(terminal_status="cancelled", result_reason="server shutdown",
                                     stop_source="server_shutdown")

    second, binding = _continue(pool, first["id"])

    assert _opening(second) == kept, "the saved request rides by value, once, never re-derived"
    assert second["text"].count(DESCRIPTION) == 1
    _no_owner_words(second)
    assert set(WITHHELD) <= set(second["task_contract"]["disabled_tools"])
    assert binding["execution_timeout_sec"] == BUDGET and binding["execution_consumed_sec"] == closed["sec"]


RIGHTS = {"forbidden": {"delegation_budget": {"may_delegate": False}, "budget_profile": {"cost_hard_stop_pct": 20}},
          "allowed": {"delegation_budget": {"may_delegate": True, "max_children": 2}}}


@pytest.mark.parametrize("rights", list(RIGHTS))
def test_continue_keeps_the_delegation_rights_and_cost_stop_at_their_consumers(pool, rights):
    """Astra R1: the door admitted may_delegate=false and a 20% cost stop, the root was assigned, its provider was
    unavailable, and Continue handed the successor may_delegate=true and no stop, so the existing delegation gate
    admitted a child the work forbids. Through the admission, its replay, a successor that never starts and the
    next Continue, the running successor's own contract keeps both as recorded; the supervisor's gate and the
    in-task cost stop read them, beside the same resources and criteria. Work that may delegate keeps that right
    and its child cap (the allowed control)."""
    from types import SimpleNamespace as Ctx

    from ouroboros.task_pacing import resolve_task_cost_ceiling
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control_delegation import check_delegation_admission
    from supervisor.continuation_admission import admit_continuation
    from supervisor.events import _parent_delegation_budget
    from tests.test_continue_work_limits import _registry_view
    from tests.test_owner_continue import NONCE
    from tests.test_restart_retention import _pool_events

    task_id = _interrupted(pool, executed=60, metadata=copy.deepcopy(RIGHTS[rights]))
    admitted = load_task_result(pool.root, task_id)["task_contract"]
    assert admitted["delegation_budget"]["may_delegate"] is (rights == "allowed"), "the door's own admission"
    first, _binding = _continue(pool, task_id)
    assert admit_continuation(task_id, action_nonce=NONCE)["replay"] is True
    _pool_events(pool.workers, pool.mp)
    pool.mp.setattr(pool.workers, "WORKERS", {})
    assert pool.workers.kill_workers(terminal_status="cancelled", result_reason="server shutdown",
                                     stop_source="server_shutdown")

    second, _binding2 = _continue(pool, first["id"])
    sent, _meta = _start(pool, second["id"], executed=0)

    contract = sent["task_contract"]
    gate = check_delegation_admission(_parent_delegation_budget(Ctx(RUNNING=pool.workers.RUNNING, DRIVE_ROOT=pool.root),
                                                                second["id"], pool.root), direct_child_count=0)
    ceiling = resolve_task_cost_ceiling(Ctx(task_contract=contract, task_metadata=sent["metadata"]), 5.0)
    if rights == "forbidden":
        assert (gate.ok, gate.reason_code) == (False, "delegation_rights_may_delegate")
        assert (ceiling.state, ceiling.ceiling_usd) == ("active", pytest.approx(1.0)), "20% of the starting $5"
    else:
        assert gate.ok and contract["delegation_budget"]["max_children"] == 2
        assert ceiling.state == "disabled" and contract["budget_profile"]["cost_hard_stop_pct"] is None
    assert contract["acceptance_claims"] == CLAIMS and contract["allowed_resources"] == RESOURCES
    withheld, network = _registry_view(sent)
    assert "web_search" in withheld and network is False


@pytest.mark.parametrize("depth", [3, 0], ids=["at-the-cap", "root-control"])
def test_continue_keeps_the_api_depth_at_the_delegation_consumer(pool, depth):
    """R4: the door admitted a root at its public depth 3 (the configured cap) and its Continue started the
    successor at depth 0, so work with no nesting room left could delegate again. The admission row records the
    depth; through the Continue, its replay, a successor that never starts and the next Continue, the running
    successor starts at the same depth and the real schedule_subagent refuses the child the cap forbids. A door
    root at depth 0 still delegates (the control)."""
    import pathlib

    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext
    from supervisor.continuation_admission import admit_continuation
    from tests._shared import configure_test_subagent
    from tests.test_owner_continue import NONCE
    from tests.test_restart_retention import _pool_events

    subagent_id = configure_test_subagent(pool.mp)
    pool.mp.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "3")
    task_id = _interrupted(pool, executed=60, depth=depth)
    assert load_task_result(pool.root, task_id)["depth"] == depth, "the door's admission records the position"
    first, _binding = _continue(pool, task_id)
    assert first["depth"] == depth and load_task_result(pool.root, first["id"]).get("depth", 0) == depth
    assert admit_continuation(task_id, action_nonce=NONCE)["replay"] is True
    _pool_events(pool.workers, pool.mp)
    pool.mp.setattr(pool.workers, "WORKERS", {})
    assert pool.workers.kill_workers(terminal_status="cancelled", result_reason="server shutdown",
                                     stop_source="server_shutdown")

    second, _binding2 = _continue(pool, first["id"])
    sent, _meta = _start(pool, second["id"], executed=0)

    assert sent["depth"] == depth
    ctx = ToolContext(repo_dir=pool.root / "repo", drive_root=pathlib.Path(sent.get("drive_root") or pool.root),
                      task_id=second["id"], task_depth=int(sent.get("depth", 0)),  # as the agent builds it
                      task_contract=sent["task_contract"], task_metadata=dict(sent["metadata"]),
                      budget_drive_root=str(sent.get("budget_drive_root") or ""))
    result = _schedule_task(ctx, subagent_id=subagent_id, objective="One more level of the same work",
                            expected_output="child id", memory_mode="empty")
    if depth:
        assert "depth limit (3) exceeded by attempted depth 4" in result and not ctx.pending_events
    else:
        assert "subtask_depth_limit" not in result and ctx.pending_events


def test_a_spent_budget_still_refuses_and_admits_nothing(pool):
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE

    task_id = _interrupted(pool, executed=960)

    ack = admit_continuation(task_id, action_nonce=NONCE)

    assert ack == {"ok": False, "error": "hard_limit_reached", "cause": "execution_deadline"}
    assert pool.workers.PENDING == []


def test_a_root_that_never_started_continues_from_a_known_zero_after_a_shutdown(pool):
    """R2: the door admits a budgeted root that is still queued when the server stops. Its admission recorded the
    close of attempt 0 at 0 seconds and no start stamped an attempt since, so its Continue keeps the same 900 seconds
    from a known zero instead of refusing them as ``execution_budget_unknown``."""
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_results import load_task_result
    from tests.test_restart_retention import _pool_events

    task_id = _api_root(pool)
    _pool_events(pool.workers, pool.mp)
    assert pool.workers.kill_workers(terminal_status="cancelled", result_reason="server shutdown",
                                     stop_source="server_shutdown")
    stored = load_task_result(pool.root, task_id)
    assert stored["cancel_origin"]["source"] == "server_shutdown" and "task_attempt" not in stored
    assert stored["execution_consumed"] == {"sec": 0.0, "attempt": 0}
    assert continuation_offer(stored, task_id) == {"eligible": True, "cause": "server_shutdown", "refusal": ""}

    row, binding = _continue(pool, task_id)

    assert (binding["execution_timeout_sec"], binding["execution_consumed_sec"]) == (BUDGET, 0.0)
    assert row["execution_timeout_sec"] == BUDGET and row["_execution_handoff_sec"] == 0.0


def test_a_root_that_started_and_never_closed_stays_unknown(pool):
    """The negative of the known zero: the same door's root was assigned and its worker recorded its start
    (attempt 1), then the process died before any close. The opening of attempt 0 never stands in for it."""
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE

    task_id = _api_root(pool)
    _start(pool, task_id, executed=300)
    write_task_result(pool.root, task_id, "failed", reason_code="provider_unavailable",
                      result="The provider was unavailable.")  # the worker's own terminal: no release, no close
    stored = load_task_result(pool.root, task_id)
    assert stored["task_attempt"] == 1 and stored["execution_consumed"] == {"sec": 0.0, "attempt": 0}

    assert continuation_offer(stored, task_id) == {
        "eligible": False, "cause": "provider_unavailable", "refusal": "execution_budget_unknown"}
    assert admit_continuation(task_id, action_nonce=NONCE) == {
        "ok": False, "error": "execution_budget_unknown", "cause": "provider_unavailable"}


def test_an_owner_stop_still_refuses(pool):
    from ouroboros.owner_continue import continuation_eligibility
    from ouroboros.task_results import load_task_result, write_task_result

    task_id = _api_root(pool)
    write_task_result(pool.root, task_id, "cancelled", cancel_origin={"source": "http_single"},
                      result="Stopped by the owner.")

    verdict = continuation_eligibility(load_task_result(pool.root, task_id), task_id)

    assert verdict == {"eligible": False, "cause": "http_single", "refusal": "stopped_by_owner"}


@pytest.mark.parametrize("damage", [
    {"api_admission": {"token": "t", "status": "pending"}},
    {"api_admission": "accepted"},
    {"description": "   "},
])
def test_a_request_without_its_accepted_receipt_or_words_is_a_gap_never_owner_text(pool, damage):
    from ouroboros.owner_continue import owner_sources
    from ouroboros.task_results import load_task_result

    task_id = _interrupted(pool)
    result = {**load_task_result(pool.root, task_id), **damage}

    sources = owner_sources(pool.root, result, task_id)

    assert sources["api"] is None and "api_request_unreadable" in sources["gaps"]
    assert sources["original"] is None and sources["later"] == []


def test_a_malformed_chained_request_is_a_gap(pool):
    from ouroboros.owner_continue import owner_sources
    from ouroboros.task_results import load_task_result

    task_id = _interrupted(pool)
    first, _binding = _continue(pool, task_id)
    stored = load_task_result(pool.root, first["id"])
    stored["metadata"]["continuation"]["owner_sources"]["api"]["channel"] = ""

    sources = owner_sources(pool.root, stored, first["id"])

    assert sources["api"] is None and "api_source_malformed" in sources["gaps"]
