"""E1-E12 — end-to-end owner-control scenarios (cancel / cascade / graceful stop / hurry).

WHAT THIS FILE IS. The cancellation protocol is densely unit-pinned already; what was
never exercised is the whole thing running as ONE system: a real ``server.py`` process, a
real supervisor with real workers, real HTTP ingress over the SAME surface the web UI
posts to, and the durable artifacts the owner and the watchdog actually read back
(``state/cancel_intents.json``, the ``cancel_intent`` forensics in
``logs/supervisor.jsonl``, ``task_results/<id>.json``, ``state/queue_snapshot.json``,
``state/terminal_deliveries.json``, the ``owner_hurry`` projection). Every scenario asserts
those artifacts — never an HTTP 200 on its own, and never a harness exit code (AGENTS.md:
the exit code is not the run status).

THREE LANES.

1. ``mock`` — a REAL isolated server driven by a LOCAL stub model. The stub is an
   OpenAI-compatible ``/v1/chat/completions`` endpoint on loopback that keeps answering
   with a harmless read-only tool call, so the agent loop stays alive as long as the
   scenario needs and terminates on command. No external host is contacted and no money is
   spent: every model slot is pinned to ``openai-compatible::mock-model`` and no other
   provider credential exists in the isolated settings, so a mis-pinned slot can only
   fail — it can never silently reach a paid provider. Opt in with
   ``OUROBOROS_E2E_CANCEL=mock``.
2. ``paid`` — scenarios whose subject is a REAL external lane (the delegated-run
   transport) or real cost accounting, which a stub cannot stand in for. Opt in with
   ``OUROBOROS_E2E_CANCEL=paid``, which also runs the ``mock`` lane.
3. default — no server at all: the driver's wire contract, and the real gateway handlers'
   acceptance of exactly the bodies that driver sends, asserted in process. These run in
   every ordinary pytest pass.

Both server lanes are ``serial``: they bind real ports and spawn real process trees.

WHAT THE PAID PASS NEEDS (operator, later, under controlled keys):

- ``OUROBOROS_E2E_CANCEL=paid``; the mock lane needs nothing but disk.
- ONE provider credential in the isolated settings the harness builds, passed BY NAME:
  export ``OUROBOROS_E2E_PAID_KEY_ENV=<name of the env var holding the key>`` and the
  harness reads that variable. The key value IS persisted — into the isolated server's own
  ``settings.json``, which ``write_settings_file`` creates at mode 0600 before the key bytes
  land — because the server can only read credentials from its settings file. The suite never
  prints a key value and never touches ``~/ouro/data/settings.json``. The workspace key pool
  (``~/ouro/file1.txt``) is the operator's source for the value; ``hope*`` buckets last.
- ``OUROBOROS_E2E_PAID_MODEL`` — the exact slug every slot is pinned to, e.g.
  ``openrouter::<cheap-slug>``. A cheap model is right: the scenarios need tool-calling and
  a terminating final turn, not reasoning quality.
- For E1-E3 additionally a working delegated-run harness (Claudexor lane) reachable from
  the isolated server; ``scripts/claudexor_platform_smoke.py`` is the existing precedent
  for proving one real delegated run before spending on the suite.
- Spend order of magnitude: single-digit US dollars for the whole paid lane (each scenario
  is a handful of short tool-calling turns on a cheap slug). If one scenario burns more
  than about a dollar by itself, stop and look — that is a runaway loop, not the scenario.

THE MOCK LANE IS THE PROOF THAT EXISTS TODAY. E4, E5, E6, E7, E9, E10, E11 and E12 were
developed and run green against the stub (re-proven on the v7next tip form). The paid
tests (E1-E3, E13) were written from the protocol and the S5 inventory but have never
been EXECUTED: treat a first-run failure there as "the scenario needs adjusting" until
the artifacts say otherwise. E13 supersedes the retired E8 (F6 disposition: pause is
the one live semantics of a budget-drained queued task).
"""

from __future__ import annotations

import json
import os
import pathlib
import time
import uuid

import pytest

from devtools.benchmarks.common.server_runner import (
    IsolatedServer,
    _api_status,
    supervisor_state_is_ready,
)
from tests.fixtures_e2e_cancellation import (
    LANE_MOCK,
    LANE_PAID,
    SCENARIOS,
    RecordingEndpoint,
    StubModelServer,
    chat_bytes,
    clone_repo,
    driver_at,
    events,
    forensics,
    intents,
    isolated_settings,
    queue_snapshot,
    require_lane,
    stall_forensics,
    start_server,
    submit_running,
    task_result,
    task_result_bytes,
    wait_until,
    write_settings_file,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def e2e_clone(tmp_path_factory):
    """One throwaway clone of the checkout under test, shared by every scenario server."""
    require_lane(LANE_MOCK)
    return clone_repo(tmp_path_factory.mktemp("e2e_clone"))


@pytest.fixture(scope="module")
def mock_stack(tmp_path_factory, request):
    """One stub model + one isolated server shared by the scenarios that can share it.

    Each scenario submits its OWN task, so sharing the server costs nothing in isolation
    and saves a process start per scenario.
    """
    require_lane(LANE_MOCK)
    clone = request.getfixturevalue("e2e_clone")
    root = tmp_path_factory.mktemp("e2e_mock")
    with StubModelServer() as stub:
        server = start_server(clone, root, isolated_settings(stub=stub))
        try:
            yield stub, server
        finally:
            server.stop()


# ===========================================================================
# Default lane: the driver's wire contract, and the real gateway's acceptance
# of exactly the bodies the driver sends. No server, no model, no egress.
# ===========================================================================

def test_scenario_manifest_is_covered():
    """Every E-id in the S5 inventory still has at least one test in this module."""
    import sys

    names = [name for name in dir(sys.modules[__name__]) if name.startswith("test_")]
    for scenario_id, (title, _lane) in SCENARIOS.items():
        prefix = f"test_{scenario_id.lower()}_"
        assert any(name.startswith(prefix) for name in names), (
            f"scenario {scenario_id} ({title}) has no {prefix}* test"
        )


def test_driver_cancel_wire_contract_matches_the_ui_client():
    """cancel_task builds the SAME request web/modules/api_client.js::cancelTask builds.

    The two axes are independent, and an options-free call must stay the legacy empty-body
    single-task request every existing benchmark caller already sends — a driver that
    silently started posting a policy would change what those runs mean.
    """
    with RecordingEndpoint() as recorder:
        driver = driver_at(recorder)
        driver.cancel_task("task-1")
        driver.cancel_task("task-2", cascade=True)
        driver.cancel_task("task-3", stop_policy="finalize_then_cancel")
        driver.cancel_task("task-4", cascade=True, stop_policy="finalize_then_cancel")
        driver.cancel_task("task-5", stop_policy="immediate")
    assert [row["path"] for row in recorder.requests] == [
        "/api/tasks/task-1/cancel", "/api/tasks/task-2/cancel", "/api/tasks/task-3/cancel",
        "/api/tasks/task-4/cancel", "/api/tasks/task-5/cancel",
    ]
    assert [row["body"] for row in recorder.requests] == [
        {},
        {"cascade": True},
        {"stop_policy": "finalize_then_cancel"},
        {"cascade": True, "stop_policy": "finalize_then_cancel"},
        {},  # explicit "immediate" IS the absent policy, exactly as the UI encodes it
    ]


def test_driver_hurry_wire_contract_is_text_free_and_id_stable():
    """hurry_task posts ONLY {"request_id": ...} and reuses a stable id per task."""
    with RecordingEndpoint() as recorder:
        driver = driver_at(recorder)
        driver.hurry_task("task-1")
        driver.hurry_task("task-1")
        driver.hurry_task("task-1", request_id="explicit-id")
        driver.hurry_task("task-2")
    bodies = [row["body"] for row in recorder.requests]
    assert [row["path"] for row in recorder.requests] == (
        ["/api/tasks/task-1/hurry"] * 3 + ["/api/tasks/task-2/hurry"]
    )
    assert all(set(body) == {"request_id"} for body in bodies), bodies
    assert bodies[0]["request_id"] == bodies[1]["request_id"], "a retry must reuse the id"
    assert bodies[2]["request_id"] == "explicit-id"
    assert bodies[3]["request_id"] != bodies[0]["request_id"], "per-task ids are distinct"


def test_driver_reports_typed_refusals_instead_of_raising():
    """A 409/404/503 refusal is part of the contract under test, so the driver has to be
    able to SEE it: urllib turns every non-2xx into an exception, and the pre-existing
    ``_api`` helper would surface a typed refusal as a bare raise."""
    with RecordingEndpoint(status=409, payload={"error": "hurry refused: cancel_pending",
                                                "reason_code": "cancel_pending"}) as recorder:
        answer = driver_at(recorder).hurry_task("task-1", request_id="rid")
    assert answer["status"] == 409
    assert answer["body"]["reason_code"] == "cancel_pending"


def test_driver_reports_transport_failure_as_status_zero():
    """A dead server stays distinguishable from a refusal (status 0, never an exception)."""
    driver = IsolatedServer(pathlib.Path("/nonexistent-clone"),
                            pathlib.Path("/nonexistent-data"),
                            pathlib.Path("/nonexistent-settings.json"))
    # Port 1 is privileged and unbindable by this user, so the refusal is deterministic —
    # a just-freed ephemeral port could be claimed by a parallel worker between the bind
    # and the request, and this test would then assert against somebody else's server.
    driver.base_url = "http://127.0.0.1:1"
    answer = driver.cancel_task("task-1", timeout=5)
    assert answer["status"] == 0 and answer["body"] == {}
    assert answer.get("error")


def _gateway_client(tmp_path, routes):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    app = Starlette(routes=[Route(path, endpoint, methods=["POST"]) for path, endpoint in routes])
    app.state.drive_root = tmp_path
    return TestClient(app)


def _isolate_queue(monkeypatch, tmp_path, *, pending=(), running=None):
    from supervisor import queue as q
    from supervisor import workers

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "PENDING", [dict(row) for row in pending])
    monkeypatch.setattr(q, "RUNNING", dict(running or {}))
    monkeypatch.setattr(q, "ACCEPTANCE_FENCES", {}, raising=False)
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    return q


def test_driver_cancel_bodies_are_accepted_by_the_real_endpoint(tmp_path, monkeypatch):
    """The bodies the driver sends are fed to the REAL handler, and the durable intent it
    mints carries the policy the driver asked for.

    This is the join that makes the wire-contract test meaningful: a driver agreeing with
    itself proves nothing, and the endpoint validates bodies strictly (a non-boolean
    cascade or an unknown policy is a 400).
    """
    from ouroboros.cancel_intents import STOP_POLICY_FINALIZE
    from ouroboros.gateway.tasks import api_task_cancel

    with RecordingEndpoint() as recorder:
        driver = driver_at(recorder)
        driver.cancel_task("root-1", stop_policy="finalize_then_cancel")
        driver.cancel_task("root-2")
    graceful_body, immediate_body = (row["body"] for row in recorder.requests)

    task = {"id": "root-1", "chat_id": 0, "root_task_id": "root-1", "_attempt": 1}
    _isolate_queue(monkeypatch, tmp_path, running={"root-1": {"task": task, "attempt": 1}})
    routes = [("/api/tasks/{task_id}/cancel", api_task_cancel)]
    with _gateway_client(tmp_path, routes) as client:
        graceful = client.post("/api/tasks/root-1/cancel", json=graceful_body)
    assert graceful.status_code == 202
    assert graceful.json()["cancel_state"] == "pending"
    assert graceful.json()["stop_policy"] == STOP_POLICY_FINALIZE
    intent = intents(tmp_path).get("root-1") or {}
    assert intent.get("stop_policy") == STOP_POLICY_FINALIZE
    assert intent.get("state") == "requested", "the graceful intent stays OPEN"

    # The options-free body keeps the legacy single-task envelope.
    other = {"id": "root-2", "chat_id": 0, "root_task_id": "root-2"}
    _isolate_queue(monkeypatch, tmp_path, pending=[other])
    with _gateway_client(tmp_path, routes) as client:
        immediate = client.post("/api/tasks/root-2/cancel", json=immediate_body)
    assert immediate.status_code == 200
    assert immediate.json() == {"ok": True, "task_id": "root-2"}


def test_driver_hurry_body_is_accepted_by_the_real_endpoint(tmp_path, monkeypatch):
    """The driver's hurry body is accepted, and a body carrying anything else is refused —
    so the driver cannot drift into the text-carrying shape the contract forbids."""
    from ouroboros.gateway.tasks import api_task_hurry

    with RecordingEndpoint() as recorder:
        driver_at(recorder).hurry_task("root-1", request_id="rid-1")
    body = recorder.requests[0]["body"]

    task = {"id": "root-1", "chat_id": 0, "root_task_id": "root-1", "_attempt": 1}
    _isolate_queue(monkeypatch, tmp_path, running={"root-1": {"task": task, "attempt": 1}})
    routes = [("/api/tasks/{task_id}/hurry", api_task_hurry)]
    with _gateway_client(tmp_path, routes) as client:
        accepted = client.post("/api/tasks/root-1/hurry", json=body)
        smuggled = client.post("/api/tasks/root-1/hurry", json={**body, "text": "hurry up"})
    assert accepted.status_code == 200
    assert accepted.json()["duplicate"] is False
    assert task_result(tmp_path, "root-1")["owner_hurry"]["request_id"] == "rid-1"
    assert smuggled.status_code == 400
    assert smuggled.json()["reason_code"] == "unexpected_fields"


def test_supervisor_readiness_contract_is_the_one_the_driver_polls():
    """A guard on the harness itself: ``supervisor_ready`` alone is not readiness — a
    server with zero workers accepts a task nothing will pick up, and every scenario would
    then time out for a reason that looks like a protocol bug."""
    assert supervisor_state_is_ready({"supervisor_ready": True, "workers_total": 1})
    assert not supervisor_state_is_ready({"supervisor_ready": True, "workers_total": 0})
    assert not supervisor_state_is_ready({"supervisor_ready": False, "workers_total": 4})


def test_api_status_helper_never_raises_on_an_error_status():
    """``_api_status`` is the only reason the scenarios can assert refusals at all; pin
    that it degrades to a typed envelope even for a non-JSON error body."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib callback name
            body = b"<html>gateway error</html>"
            self.send_response(503)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        answer = _api_status(f"http://127.0.0.1:{server.server_address[1]}", "POST", "/x", {})
    finally:
        server.shutdown()
        server.server_close()
    assert answer["status"] == 503
    assert answer["body"] == {}


def test_stall_forensics_separates_the_three_stall_hypotheses(tmp_path):
    """The diagnostic a red ``wait_task`` carries must itself never fail: on a
    task RUNNING on the slot a cancel respawned, it names the worker, whether that
    worker's child ever confirmed ready after the settle, and the pool's readiness
    rows -- and it degrades to plain facts on an empty root."""
    from tests.fixtures_e2e_cancellation import stall_forensics

    assert stall_forensics(tmp_path, "t1")["task"] == {"state": "absent"}

    (tmp_path / "state").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "state" / "queue_snapshot.json").write_text(json.dumps({
        "pending_count": 0, "running_count": 1, "reaping_count": 0, "worker_total": 4,
        "assignable_idle_workers": 3,
        "pending": [], "running": [{"id": "t1", "worker_id": 0, "runtime_sec": 241.0, "heartbeat_lag_sec": 241.0}],
    }), encoding="utf-8")
    (tmp_path / "logs" / "supervisor.jsonl").write_text("\n".join([
        json.dumps({"ts": "2026-09-05T13:28:27.5+00:00", "type": "cancel_intent", "event": "settled", "task_id": "t0"}),
        json.dumps({"ts": "2026-09-05T13:28:29.0+00:00", "type": "worker_sha_verify", "worker_id": 0, "worker_pid": 20}),
        json.dumps({"ts": "2026-09-05T13:28:30.0+00:00", "type": "task_received", "task_id": "t1"}),
    ]) + "\n", encoding="utf-8")
    (tmp_path / "logs" / "events.jsonl").write_text("\n".join([
        json.dumps({"ts": "2026-09-05T13:28:21.0+00:00", "type": "worker_ready", "worker_id": 0, "pid": 10}),
        json.dumps({"ts": "2026-09-05T13:28:27.6+00:00", "type": "worker_boot", "pid": 20}),
        json.dumps({"ts": "2026-09-05T13:28:29.0+00:00", "type": "worker_ready", "worker_id": 0, "pid": 20}),
        json.dumps({"ts": "2026-09-05T13:28:31.0+00:00", "type": "llm_usage", "task_id": "t1"}),
        "not json",
    ]) + "\n", encoding="utf-8")

    facts = stall_forensics(tmp_path, "t1")
    assert facts["task"] == {"state": "running", "worker_id": 0, "runtime_sec": 241.0, "heartbeat_lag_sec": 241.0}
    assert facts["cancel_settled_ts"] == "2026-09-05T13:28:27.5+00:00"
    assert [(r["type"], r["pid"]) for r in facts["worker_rows_after_settle"]] == [("worker_boot", 20), ("worker_ready", 20)]
    assert [r["type"] for r in facts["readiness_rows_after_settle"]] == ["worker_sha_verify"]
    assert facts["worker_ready_for_assigned_worker"] is True
    assert facts["task_event_types"] == ["llm_usage"]
    assert facts["snapshot_counts"]["assignable_idle_workers"] == 3


# ===========================================================================
# Mock lane: a real isolated server driven by the local stub model.
# ===========================================================================

@pytest.mark.serial
def test_e4_cancel_single_settles_the_intent_and_writes_the_terminal(mock_stack):
    """E4 — the single cancel: intent requested -> claimed -> settled, the projection is
    empty again, and the task has a durable ``cancelled`` result."""
    _stub, server = mock_stack
    data_root = server.data_root
    task_id = submit_running(server, "List the repository files and keep watching them.")

    answer = server.cancel_task(task_id)
    assert answer["status"] == 200
    assert answer["body"] == {"ok": True, "task_id": task_id}

    assert server.wait_task(task_id, timeout=180).get("status") == "cancelled"
    assert task_result(data_root, task_id).get("status") == "cancelled"

    rows = forensics(data_root, task_id=task_id)
    assert [row.get("event") for row in rows][:3] == ["requested", "claimed", "settled"], rows
    assert rows[0].get("source") == "http_single"
    assert rows[0].get("scope") == "single"
    settled = [row for row in rows if row.get("event") == "settled"]
    assert settled and settled[0].get("outcome") == "cancelled"
    # Custody released it: no open intent may be left behind for the watchdog to re-feed.
    wait_until(lambda: task_id not in intents(data_root), 60)
    assert task_id not in intents(data_root)


@pytest.mark.serial
def test_e5_cancel_cascade_settles_descendants_then_the_root(e2e_clone, tmp_path_factory):
    """E5 — the cascade: the root intent is minted with ``scope=cascade`` AT THE INGRESS,
    each captured descendant gets its own intent, and the ROOT settles only on the cascade
    postcondition — never on its own worker's death."""
    require_lane(LANE_MOCK)
    root_dir = tmp_path_factory.mktemp("e2e_cascade")
    with StubModelServer(mode="spawn") as stub:
        server = start_server(e2e_clone, root_dir, isolated_settings(stub=stub))
        try:
            data_root = server.data_root
            task_id = submit_running(server, "Delegate a read-only survey to a subagent, then wait.")
            # The child must actually be live: a cascade over an empty subtree would pass
            # the assertions below for the wrong reason.
            assert wait_until(
                lambda: int(queue_snapshot(data_root).get("running_count") or 0) >= 2, 180,
            ), "the subagent never reached the RUNNING set"

            answer = server.cancel_task(task_id, cascade=True)
            assert answer["status"] == 200
            assert answer["body"] == {"ok": True, "task_id": task_id, "cascade": True}
            assert server.wait_task(task_id, timeout=180).get("status") == "cancelled"

            root_rows = forensics(data_root, task_id=task_id)
            requested = [row for row in root_rows if row.get("event") == "requested"]
            assert requested and requested[0].get("scope") == "cascade"
            assert requested[0].get("source") == "http_cascade"
            descendants = [
                row for row in forensics(data_root, event="requested")
                if row.get("source") == "cascade_descendant"
            ]
            assert descendants, "no per-descendant cancel intent was minted"
            assert all(row.get("requested_by") == task_id for row in descendants)
            root_settled = [row for row in root_rows if row.get("event") == "settled"]
            assert root_settled, root_rows
            assert "cascade postcondition" in str(root_settled[-1].get("detail") or ""), root_settled
        finally:
            server.stop()


@pytest.mark.serial
def test_e6_cancel_after_settlement_is_a_404_and_preserves_the_result(mock_stack):
    """E6 — the cancel x completion race, on its deterministic side: once a task settles on
    its own, a cancel must neither resurrect it nor rewrite its stored result.

    The interleaving where the kill lands DURING finalization is inherently racy and is
    unit-pinned; what an E2E can prove without flaking is the postcondition."""
    stub, server = mock_stack
    data_root = server.data_root
    previous_mode = stub.mode
    stub.mode = "finish"
    try:
        task_id = server.submit("Answer immediately and stop.")
        final = server.wait_task(task_id, timeout=240)
    finally:
        stub.mode = previous_mode
    if final.get("status") != "completed":
        # A bare {'status': 'timeout'} cannot tell a child wedged during boot from one wedged at
        # its first lazy import from a task stuck in assignment: the queue row, the post-cancel
        # worker_boot/worker_ready pids and the pool's readiness rows separate them in one red run.
        final = dict(final, stall=stall_forensics(data_root, task_id))
    assert final.get("status") == "completed", final

    # A `completed` STATUS is not yet "settled and dead". A settled result whose worker is
    # still winding down post-task cognition keeps LIVE OWNERSHIP, and a cancel then
    # legitimately answers 200 and mints an intent so custody can kill the spending
    # worker (GR6-1). Waiting for ownership to end is what makes this assert the
    # settled-AND-dead contract it names instead of racing the other lane.
    assert wait_until(
        lambda: not any(
            str(row.get("id") or "") == task_id
            for row in (queue_snapshot(data_root).get("running") or [])
        ),
        180,
    ), "the completed task never released its worker"

    def _settled_bytes():
        first = task_result_bytes(data_root, task_id)
        time.sleep(2)
        return first if first == task_result_bytes(data_root, task_id) else None

    # Snapshot only once the terminal writers have finished: an unrelated post-terminal
    # projection landing between the two reads would be misread as the cancel rewriting
    # the result.
    before = wait_until(_settled_bytes, 60)
    assert before, "the stored result never stopped changing"

    answer = server.cancel_task(task_id)
    assert answer["status"] == 404
    assert answer["body"].get("error") == "task not found or not active"
    assert task_result_bytes(data_root, task_id) == before, \
        "a cancel rewrote a settled task's result"
    assert task_id not in intents(data_root)


@pytest.mark.serial
def test_e7_terminal_delivery_is_owed_then_recorded_exactly_once(mock_stack):
    """E7 — the owed outbox, on the side a stub can drive: a terminal answer registers a
    delivery id in ``state/terminal_deliveries.json`` and lands in ``delivered`` exactly
    once, which is the dedupe key the crash replay reuses.

    The SIGKILL-between-owe-and-send half belongs to the operator pass: the replay is only
    due after ``_REPLAY_MIN_AGE_SEC``, so a "nothing happened yet" read is
    indistinguishable from a loss unless the run waits past it (S5 §5 note on E7)."""
    stub, server = mock_stack
    data_root = server.data_root
    previous_mode = stub.mode
    stub.mode = "finish"
    try:
        task_id = server.submit("Answer immediately and stop.")
        assert server.wait_task(task_id, timeout=240).get("status") == "completed"
    finally:
        stub.mode = previous_mode

    ledger_path = pathlib.Path(data_root) / "state" / "terminal_deliveries.json"

    def _read_ledger() -> dict:
        if not ledger_path.exists():
            return {}
        try:
            return json.loads(ledger_path.read_text(encoding="utf-8"))
        except ValueError:  # a concurrent writer mid-replace
            return {}

    # Wait for THIS task's row, not merely for the file: the ledger already exists from an
    # earlier scenario on the shared server, and the settle -> owe -> send hop lands after
    # the task result does. Waiting on the file would read a ledger that is simply not
    # there yet and call an absent row a lost delivery.
    delivered = wait_until(
        lambda: [row for row in (_read_ledger().get("delivered") or []) if task_id in str(row)],
        120,
    )
    assert delivered, f"the terminal answer never reached the delivery ledger: {_read_ledger()}"
    assert len(delivered) == 1, delivered
    assert str(delivered[0]).startswith("final:"), delivered
    assert not any(task_id in str(key) for key in (_read_ledger().get("pending") or {})), \
        _read_ledger()


@pytest.mark.serial
def test_e9_boot_migration_adopts_a_legacy_cancel_requested_latch(e2e_clone, tmp_path_factory):
    """E9 — a pre-redesign ``cancel_requested`` result file becomes an ordinary intent at
    boot (``source=boot_migration``), and a SECOND boot migrates it zero more times."""
    require_lane(LANE_MOCK)
    root_dir = tmp_path_factory.mktemp("e2e_boot")
    data_root = root_dir / "data"
    (data_root / "task_results").mkdir(parents=True, exist_ok=True)
    legacy_id = uuid.uuid4().hex[:16]
    (data_root / "task_results" / f"{legacy_id}.json").write_text(json.dumps({
        "task_id": legacy_id,
        "status": "cancel_requested",
        "description": "a task wedged in the pre-redesign cancel latch",
        "created_at": "2020-01-01T00:00:00+00:00",
    }), encoding="utf-8")

    def _migrations():
        return [row for row in forensics(data_root, task_id=legacy_id, event="requested")
                if row.get("source") == "boot_migration"]

    with StubModelServer() as stub:
        settings = isolated_settings(stub=stub)
        server = start_server(e2e_clone, root_dir, settings)
        try:
            migrated = wait_until(_migrations, 90)
            assert migrated, "boot did not adopt the legacy cancel_requested latch"
            assert len(migrated) == 1, migrated
        finally:
            server.stop()

        # Second boot on the SAME data root: custody has settled the latch, so the
        # migration must not mint a second intent for the same task.
        server = start_server(e2e_clone, root_dir, settings)
        try:
            time.sleep(10)
            assert len(_migrations()) == 1, _migrations()
        finally:
            server.stop()


@pytest.mark.serial
def test_e10_graceful_stop_keeps_the_intent_open_and_finalizes(e2e_clone, tmp_path_factory):
    """E10 — ``finalize_then_cancel``: a 202 pending acknowledgement, an intent that stays
    OPEN carrying the policy, and a terminal reason of ``owner_requested_finalization``
    (never an acceptance-deadline bypass)."""
    require_lane(LANE_MOCK)
    root_dir = tmp_path_factory.mktemp("e2e_graceful")
    with StubModelServer() as stub:
        server = start_server(e2e_clone, root_dir, isolated_settings(stub=stub))
        try:
            data_root = server.data_root
            task_id = submit_running(server, "List the repository files and keep watching them.")

            answer = server.cancel_task(task_id, stop_policy="finalize_then_cancel")
            assert answer["status"] == 202
            assert answer["body"]["cancel_state"] == "pending"
            assert answer["body"]["stop_policy"] == "finalize_then_cancel"
            # The intent is the whole owner will and must be durable BEFORE the episode.
            intent = intents(data_root).get(task_id) or {}
            assert intent.get("stop_policy") == "finalize_then_cancel", intents(data_root)
            assert intent.get("state") == "requested"
            assert intent.get("source") == "http_graceful"

            final = server.wait_task(task_id, timeout=300)
            assert final.get("status") in {"completed", "cancelled"}, final
            # The terminal reason is stamped by the finalization rail, which lands after
            # the status does — read it with a bound rather than in the same instant.
            reason = wait_until(lambda: task_result(data_root, task_id).get("reason_code"), 60)
            stored = task_result(data_root, task_id)
            assert reason == "owner_requested_finalization", {
                "status": stored.get("status"), "reason_code": stored.get("reason_code"),
            }
        finally:
            server.stop()


@pytest.mark.serial
def test_e11_stop_now_hardens_the_same_intent_without_minting_a_second(e2e_clone, tmp_path_factory):
    """E11 — Stop-now during the graceful wait HARDENS the pending intent: same
    ``request_id``, policy flips to immediate, exactly one ``stop_policy_hardened``
    forensic row, and no second intent for the task."""
    require_lane(LANE_MOCK)
    root_dir = tmp_path_factory.mktemp("e2e_harden")
    # The stub answers slowly so the graceful episode is still in flight when Stop-now
    # arrives — otherwise the task finalizes first and the hardening has nothing to harden.
    with StubModelServer(latency_sec=8.0) as stub:
        server = start_server(e2e_clone, root_dir, isolated_settings(stub=stub))
        try:
            data_root = server.data_root
            task_id = submit_running(server, "List the repository files and keep watching them.")
            assert server.cancel_task(task_id, stop_policy="finalize_then_cancel")["status"] == 202
            first = dict(intents(data_root).get(task_id) or {})
            assert first.get("request_id"), intents(data_root)

            server.cancel_task(task_id)  # Stop-now: 200 while live, 404 if it just settled
            hardened = wait_until(
                lambda: forensics(data_root, task_id=task_id, event="stop_policy_hardened"), 60,
            )
            assert hardened, "no stop_policy_hardened forensic row"
            assert len(hardened) == 1, hardened
            assert hardened[0].get("stop_policy") == "immediate"
            assert hardened[0].get("request_id") == first["request_id"], "a SECOND intent was minted"
            requested = forensics(data_root, task_id=task_id, event="requested")
            assert {row.get("request_id") for row in requested} == {first["request_id"]}, requested
            assert server.wait_task(task_id, timeout=300).get("status") in {"cancelled", "completed"}
        finally:
            server.stop()


@pytest.mark.serial
def test_e12_owner_hurry_is_idempotent_text_free_and_loses_to_a_pending_stop(
    e2e_clone, tmp_path_factory,
):
    """E12 — hurry: one typed control, an ``owner_hurry`` projection written WITHOUT
    touching status, exactly one non-chat ``owner_hurry`` event, zero chat rows, and a 409
    ``cancel_pending`` once a stop is pending (a stop always owns the terminal reason)."""
    require_lane(LANE_MOCK)
    root_dir = tmp_path_factory.mktemp("e2e_hurry")
    with StubModelServer() as stub:
        server = start_server(e2e_clone, root_dir, isolated_settings(stub=stub))
        try:
            data_root = server.data_root
            chat_before = chat_bytes(data_root)
            task_id = submit_running(server, "List the repository files and keep watching them.")

            first = server.hurry_task(task_id)
            assert first["status"] == 200, first
            assert first["body"]["state"] == "requested"
            assert first["body"]["duplicate"] is False
            request_id = first["body"]["request_id"]

            retry = server.hurry_task(task_id)  # the same stable id
            assert retry["status"] == 200
            assert retry["body"]["request_id"] == request_id
            assert retry["body"]["duplicate"] is True, "a retry minted a second control"

            block = (task_result(data_root, task_id) or {}).get("owner_hurry") or {}
            assert block.get("request_id") == request_id, block
            assert block.get("reason") == "owner_hurry"
            assert task_result(data_root, task_id).get("status") != "cancelled"

            hurry_events = [row for row in events(data_root, "owner_hurry")
                            if row.get("task_id") == task_id]
            assert len(hurry_events) == 1, hurry_events
            assert hurry_events[0].get("phase") == "requested"
            assert hurry_events[0].get("is_progress") is False
            assert chat_bytes(data_root) == chat_before, "hurry produced a chat row"

            assert server.cancel_task(task_id, stop_policy="finalize_then_cancel")["status"] == 202
            refused = server.hurry_task(task_id, request_id=f"after-stop-{uuid.uuid4().hex[:8]}")
            assert refused["status"] == 409, refused
            assert refused["body"].get("reason_code") == "cancel_pending"
            server.wait_task(task_id, timeout=300)
        finally:
            server.stop()


# ===========================================================================
# Paid lane: scenarios a stub cannot stand in for. NEVER EXECUTED by this
# lane's author — see the module docstring.
# ===========================================================================

@pytest.mark.serial
def test_e1_delegated_run_lifecycle_emits_the_four_verb_families(e2e_clone, tmp_path_factory):
    """E1 — start -> wait -> answer -> cancel against a REAL delegated lane: the durable
    event families exist and containment recorded no fault."""
    require_lane(LANE_PAID)
    root_dir = tmp_path_factory.mktemp("e2e_delegate")
    server = start_server(e2e_clone, root_dir, isolated_settings(stub=None, paid=True))
    try:
        data_root = server.data_root
        task_id = server.submit(
            "Use delegate_start(subagent_id=\"delegated-leaf\") to open one delegated run that asks a trivial question, "
            "delegate_wait for it, delegate_answer any interaction it raises, then "
            "delegate_cancel the run and finish."
        )
        server.wait_task(task_id, timeout=1800)
        assert events(data_root, "delegate_run_start_requested"), "no delegated run was requested"
        assert events(data_root, "delegate_run_started"), "the delegated run never started"
        assert events(data_root, "delegate_run_cancel_outcome"), "no cancel outcome was recorded"
        from ouroboros.delegate_custody import open_containment_faults
        # The faults log is a LEDGER: a resolution row follows its fault, and a clean run
        # writes only resolutions (first paid execution, 2026-09-03: `settled_terminal`,
        # `verified_terminal`). Open faults are the fault, not the file's existence.
        assert open_containment_faults(data_root) == [], \
            "the delegated lane left an OPEN containment fault"
    finally:
        server.stop()


@pytest.mark.serial
def test_e2_delegated_patch_integration_disposes_the_snapshot(e2e_clone, tmp_path_factory):
    """E2 — a clean ``integrate_delegated_patch``: capture -> apply -> disposed=applied."""
    require_lane(LANE_PAID)
    root_dir = tmp_path_factory.mktemp("e2e_integrate")
    server = start_server(e2e_clone, root_dir, isolated_settings(stub=None, paid=True))
    try:
        data_root = server.data_root
        task_id = server.submit(
            "Open a MUTATING delegated run with delegate_start(subagent_id=\"delegated-leaf\") that adds one new file with a single line of "
            "text, then integrate its patch with integrate_delegated_patch and finish."
        )
        server.wait_task(task_id, timeout=1800)
        assert events(data_root, "delegate_run_patch_captured"), "no patch was captured"
        disposed = events(data_root, "delegate_run_patch_disposed")
        assert disposed, "the captured patch was never disposed"
        assert any(str(row.get("disposition") or row.get("outcome") or "") == "applied"
                   for row in disposed), disposed
    finally:
        server.stop()


@pytest.mark.serial
def test_e3_conflicting_delegated_patch_preserves_the_snapshot(e2e_clone, tmp_path_factory):
    """E3 — a conflicting integrate: the patch is NOT disposed and the snapshot survives,
    so the owner can retry instead of losing the delegated work."""
    require_lane(LANE_PAID)
    root_dir = tmp_path_factory.mktemp("e2e_conflict")
    server = start_server(e2e_clone, root_dir, isolated_settings(stub=None, paid=True))
    try:
        data_root = server.data_root
        task_id = server.submit(
            "Open a MUTATING delegated run with delegate_start(subagent_id=\"delegated-leaf\") that edits README.md, then — before "
            "integrating — change the same lines of README.md yourself, then attempt "
            "integrate_delegated_patch and report what happened."
        )
        server.wait_task(task_id, timeout=1800)
        assert events(data_root, "delegate_run_patch_captured"), "no patch was captured"
        applied = [row for row in events(data_root, "delegate_run_patch_disposed")
                   if str(row.get("disposition") or row.get("outcome") or "") == "applied"]
        assert not applied, "a conflicting patch was disposed as applied"
        registry = pathlib.Path(data_root) / "state" / "subagent_worktrees.json"
        assert registry.exists() and registry.read_text(encoding="utf-8").strip(), \
            "the snapshot registry was discarded on a conflict"
    finally:
        server.stop()


@pytest.mark.serial
def test_e13_budget_drain_pauses_queued_tasks_and_leaves_no_intents(e2e_clone, tmp_path_factory):
    """E13 — the budget drain under the LIVE pause semantics (supersedes E8, F6
    disposition): a task whose root budget is exhausted BEFORE its first model
    dispatch is paused — durable ``scheduled`` result with
    ``reason_code=budget_exhausted`` and a typed ``budget_scope_paused`` event —
    and nothing is left holding an open cancel intent. Explicit resume or cancel
    is the owner's move; nothing is silently failed.

    Paid-only by construction: the drain is driven by real cost accounting, and a
    stub model has no tariff, so its usage never reaches the ceiling
    (``estimate_cost_optional`` preserves an unknown model's cost as None)."""
    require_lane(LANE_PAID)
    root_dir = tmp_path_factory.mktemp("e2e_budget")
    # The budget must arrive through the settings FILE: an env budget silently does not
    # reach an isolated server (AGENTS.md), and a post-mortem bump is never read back.
    settings = isolated_settings(stub=None, paid=True,
                                 TOTAL_BUDGET=0.02, OUROBOROS_PER_TASK_COST_USD=0.02)
    server = start_server(e2e_clone, root_dir, settings)
    try:
        data_root = server.data_root
        task_ids = [server.submit(f"Describe the repository layout, pass {n}.") for n in range(4)]
        paused = wait_until(lambda: events(data_root, "budget_scope_paused"), 900)
        assert paused, "no budget_scope_paused event was ever recorded"
        paused_ids = {str(row.get("task_id") or "") for row in paused}
        assert paused_ids & set(task_ids), (paused_ids, task_ids)
        for tid in sorted(paused_ids & set(task_ids)):
            stored = task_result(data_root, tid)
            assert stored.get("status") == "scheduled", stored
            assert stored.get("reason_code") == "budget_exhausted", stored
        assert not [tid for tid in task_ids if tid in intents(data_root)], intents(data_root)
    finally:
        server.stop()


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits are meaningless on Windows")
def test_settings_file_is_created_secret_safe(tmp_path):
    """No lane marker: this contract must hold in the ordinary battery. The paid
    lane's settings file carries a live API key on a shared host, so the file
    must never exist with group/other-readable bits."""
    settings_path = tmp_path / "settings.json"
    write_settings_file(settings_path, {"OPENROUTER_API_KEY": "not-a-credential"})
    assert (settings_path.stat().st_mode & 0o777) == 0o600
    # Re-writing an existing (possibly wider) file must also end at 0600.
    settings_path.chmod(0o664)
    write_settings_file(settings_path, {"OPENROUTER_API_KEY": "not-a-credential"})
    assert (settings_path.stat().st_mode & 0o777) == 0o600
