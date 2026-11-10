"""GATE ROUND-4 (v6.98.0 phase A) — final micro-fix regression tests.

GR4-1  the owed-registration-failure rule is UNIFORM: cascade summary and the
       miss/already-settled fast paths release the claim and leave the intent
       OPEN (watchdog retries, loudly) — never a settle over an unowed answer;
GR4-2  the cascade summary's delivery id is deterministic per INTENT
       (``cascade:<root_tid>:<request_id>``): a replay of the same intent
       dedups, a later separate cancel request delivers its own;
GR4-3  the synthetic lifecycle-fault terminal fires the assisted-update and
       cooperative-checkpoint hooks exactly as the normal dispatch does;
GR4-4  the cascade digest reports each child's CURRENT durable status;
GR4-5  chat.js: a failed post-failure detail fetch proves nothing — the card
       keeps the pending presentation and the disabled Cancel (runs as behaviour
       in web/tests/cancel_pending_stop.test.js);
GR4-6  /evolve start clears ``evolution_owner_stopped`` BEFORE the campaign
       mint (both ingresses), and the owner-stop backstop defers while any
       other evolution task is live;
GR4-7  the cascade-settle auto-release requires a generation fence —
       ``request_id`` alone must not release a different owner's live claim;
GR4-8  corrupt-projection cancel refusals carry a distinct typed reason and
       honest (non-"retry") wording.
"""

from __future__ import annotations

import json
import pathlib
import types

import pytest

from ouroboros import cancel_intents as ci
from ouroboros.task_results import (
    STATUS_CANCELLED,
    STATUS_RUNNING,
    load_task_result,
    write_task_result,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class _CaptureQueue:
    def __init__(self):
        self.events = []

    def put(self, evt):
        self.events.append(evt)


def _event_rows(drive, log_name="events.jsonl"):
    path = pathlib.Path(drive) / "logs" / log_name
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture()
def qenv(tmp_path, monkeypatch):
    import supervisor.queue as q
    from supervisor import task_lifecycle, workers

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr(q, "RUNNING", {}, raising=False)
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: None, raising=False)
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(task_lifecycle, "CANCELLED_ROOT_FENCES", {}, raising=False)
    monkeypatch.setattr(task_lifecycle, "_ACTIVE_CASCADE_FENCES", {}, raising=False)
    return types.SimpleNamespace(q=q, tl=task_lifecycle, workers=workers, drive=tmp_path)


def _corrupt_delivery_registry(drive) -> pathlib.Path:
    store = pathlib.Path(drive) / "state" / "terminal_deliveries.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("[1, 2, 3]", encoding="utf-8")  # non-object JSON
    return store


# --------------------------------------------------------------------------
# GR4-1 — the registration-failure rule is uniform across every settle path
# --------------------------------------------------------------------------


def test_gr4_1_cascade_summary_registration_failure_keeps_the_intent(qenv, monkeypatch):
    """A cascade whose summary could NOT be durably owed must leave the root's
    cascade intent OPEN (watchdog re-runs it, loudly via the typed event) —
    settling would replay the incident to silence. A healed registry converges."""
    from supervisor import workers

    queue = _CaptureQueue()
    monkeypatch.setattr(workers, "get_event_q", lambda *, create=True: queue, raising=False)
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *a, **kw: None)
    write_task_result(qenv.drive, "cs1", STATUS_CANCELLED, chat_id=6, result="killed")
    ci.request_cancel(qenv.drive, "cs1", scope=ci.SCOPE_CASCADE, allow_settled_target=True)
    store = _corrupt_delivery_registry(qenv.drive)

    assert qenv.tl.cancel_task_by_id("cs1") is True, (
        "the teardown itself is not gated on the registry"
    )
    row = ci.active_intent(qenv.drive, "cs1")
    assert row is not None and row["scope"] == ci.SCOPE_CASCADE, (
        "GR4-1: unowed summary → the cascade intent stays OPEN"
    )
    assert any(
        r.get("type") == "terminal_delivery_unregistered" for r in _event_rows(qenv.drive)
    ), "the durability gap is a loud typed event"

    # Healed registry: the watchdog replay owes the summary and settles.
    store.unlink()
    assert qenv.tl.cancel_task_by_id("cs1") is True
    assert ci.active_intent(qenv.drive, "cs1") is None, "healed registry converges"


# --------------------------------------------------------------------------
# GR4-2 — deterministic per-intent cascade-summary delivery id
# --------------------------------------------------------------------------


def test_gr4_2_same_intent_replay_dedups_by_the_per_intent_delivery_id(qenv, monkeypatch):
    """Crash-after-owed: the watchdog replay of the SAME cascade intent rebuilds
    the summary (content may differ) but mints the SAME ``cascade:<tid>:<rid>``
    id, so the delivered registry suppresses a second copy. A later separate
    cancel request (new request_id) legitimately delivers its own."""
    from supervisor import terminal_delivery as td
    from supervisor import workers

    queue = _CaptureQueue()
    monkeypatch.setattr(workers, "get_event_q", lambda *, create=True: queue, raising=False)
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *a, **kw: None)
    write_task_result(qenv.drive, "cr2", STATUS_CANCELLED, chat_id=6, result="killed")
    minted = ci.request_cancel(
        qenv.drive, "cr2", scope=ci.SCOPE_CASCADE, allow_settled_target=True,
    )
    did = f"cascade:cr2:{minted['request_id']}"

    # Run 1 crashes between the owed summary and the postcondition settle.
    real_settle = ci.settle_intent
    armed = {"on": True}

    def _crashy(root, tid, **kw):
        if kw.get("allow_cascade_scope") and armed["on"]:
            armed["on"] = False
            raise OSError("crash before the settle")
        return real_settle(root, tid, **kw)

    monkeypatch.setattr("ouroboros.cancel_intents.settle_intent", _crashy)

    assert qenv.tl.cancel_task_by_id("cr2") is True
    sends = [e for e in queue.events if e.get("type") == "send_message"]
    assert [e["delivery_id"] for e in sends] == [did]
    assert ci.active_intent(qenv.drive, "cr2") is not None, "crash shape: intent open"

    # Confirmed send → the same-intent replay delivers NOTHING new and settles.
    td.register_delivery(qenv.drive, did)
    queue.events.clear()
    assert qenv.tl.cancel_task_by_id("cr2") is True
    assert [e for e in queue.events if e.get("type") == "send_message"] == []
    assert ci.active_intent(qenv.drive, "cr2") is None

    # A LATER separate cancel request mints a new intent → its own summary.
    second = ci.request_cancel(
        qenv.drive, "cr2", scope=ci.SCOPE_CASCADE, allow_settled_target=True,
    )
    assert second["request_id"] != minted["request_id"]
    queue.events.clear()
    assert qenv.tl.cancel_task_by_id("cr2") is True
    later = [e for e in queue.events if e.get("type") == "send_message"]
    assert [e["delivery_id"] for e in later] == [f"cascade:cr2:{second['request_id']}"]


# --------------------------------------------------------------------------
# GR4-3 — synthetic terminal fires the assisted-update + coop hooks
# --------------------------------------------------------------------------


def _fault_ctx(tmp_path, running):
    from ouroboros.utils import append_jsonl

    return types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING=running,
        WORKERS={3: types.SimpleNamespace(busy_task_id=next(iter(running)), reaping=False)},
        append_jsonl=append_jsonl,
        persist_queue_snapshot=lambda reason="": None,
        bridge=types.SimpleNamespace(push_log=lambda evt: None),
        send_with_budget=lambda *a, **kw: None,
    )


def _fault_probes(monkeypatch):
    import supervisor.events as events_mod
    import supervisor.update_merge as um

    probes = {"abort": 0, "gate": 0, "root_done": 0, "quiesce": 0}
    monkeypatch.setattr(
        um, "abort_orphaned_assisted_tx",
        lambda tid, meta: probes.__setitem__("abort", probes["abort"] + 1),
    )
    monkeypatch.setattr(
        um, "release_assisted_writer_gate_after_task",
        lambda meta: probes.__setitem__("gate", probes["gate"] + 1),
    )
    monkeypatch.setattr(
        events_mod, "_checkpoint_coop_roots_on_root_done",
        lambda ctx, task, tid: probes.__setitem__("root_done", probes["root_done"] + 1),
    )
    monkeypatch.setattr(
        events_mod, "_maybe_checkpoint_coop_on_tree_quiescence",
        lambda ctx, task, tid: probes.__setitem__("quiesce", probes["quiesce"] + 1),
    )
    return probes


def test_gr4_3_lifecycle_fault_fires_the_quiescence_hook_for_a_subagent(tmp_path, monkeypatch):
    import supervisor.queue as q
    from supervisor.events import _handle_task_done

    monkeypatch.setattr(q, "clear_acceptance_fence_for_root", lambda tid: None)
    probes = _fault_probes(monkeypatch)
    running = {"g3s": {"task": {
        "id": "g3s", "chat_id": 0, "delegation_role": "subagent",
        "root_task_id": "g3root", "parent_task_id": "g3root",
    }}}
    ctx = _fault_ctx(tmp_path, running)
    write_task_result(tmp_path, "g3s", STATUS_RUNNING, result="working")

    _handle_task_done({"task_id": "g3s", "status": "running", "worker_id": 3}, ctx)

    assert load_task_result(tmp_path, "g3s")["status"] == "failed"
    assert probes["abort"] == 1 and probes["gate"] == 1
    assert probes["root_done"] == 0
    assert probes["quiesce"] == 1, (
        "GR4-3: a faulted LAST subagent still triggers the tree-quiescence checkpoint"
    )


# --------------------------------------------------------------------------
# GR4-4 — the digest reports each child's CURRENT durable status
# --------------------------------------------------------------------------
# GR4-6 — evolution start/backstop race
# --------------------------------------------------------------------------


def test_gr4_6_toggle_evolution_clears_owner_stop_before_the_campaign_mint(
    tmp_path, monkeypatch,
):
    import supervisor.state as state
    from supervisor import events as events_mod
    from supervisor import evolution_lifecycle as el

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    state.update_state(lambda live: live.update(
        owner_chat_id=7, evolution_owner_stopped=True,
    ))
    seen: list = []
    monkeypatch.setattr(el, "evolution_block_reason", lambda: "")
    monkeypatch.setattr(
        el, "start_evolution_campaign",
        lambda objective, source="", **kwargs: seen.append(
            bool(state.load_state().get("evolution_owner_stopped"))
        ) or {"status": "active"},
    )
    ctx = types.SimpleNamespace(
        load_state=state.load_state, send_with_budget=lambda *a, **kw: None,
    )

    # An OWNER start (В12: only owner provenance clears the sticky stop).
    events_mod._handle_toggle_evolution(
        {"enabled": True, "objective": "x", "source": "owner_chat"}, ctx,
    )

    assert seen == [False], (
        "GR4-6: the owner-stop flag is cleared BEFORE the campaign is minted — "
        "the backstop firing in that window must find flag=False"
    )
    assert bool(state.load_state().get("evolution_owner_stopped")) is False


def test_gr4_6_server_evolve_start_clears_the_flag_before_the_campaign():
    """The owner-chat `/evolve` ingress and the owner-sourced toggle share ONE start
    transaction (#1307); its clear-before-mint ordering is pinned at the source."""
    import inspect

    from supervisor.events_runtime_controls import owner_evolution_start

    src = (REPO_ROOT / "server.py").read_text(encoding="utf-8")
    assert 'owner_evolution_start(objective, source="owner_chat")' in src
    body = inspect.getsource(owner_evolution_start)
    assert body.index("update_state(_clear_owner_stop") < body.index("start_evolution_campaign("), (
        "GR4-6: the owner-stop flag is cleared BEFORE the campaign is minted"
    )

def test_gr4_6_backstop_defers_while_another_evolution_task_is_live(tmp_path, monkeypatch):
    import supervisor.queue as q
    import supervisor.state as state
    from supervisor import events as events_mod
    from supervisor import evolution_lifecycle as el

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    q.init(tmp_path)
    assert el.start_evolution_campaign("Improve", source="test").get("status") == "active"
    state.update_state(lambda live: live.update(evolution_owner_stopped=True))
    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr(q, "RUNNING", {
        "evo-a": {"task": {"id": "evo-a", "type": "evolution"}},
        "evo-b": {"task": {"id": "evo-b", "type": "evolution"}},
    }, raising=False)

    # evo-a is the settling task (its RUNNING row pops only after dispatch);
    # evo-b is STILL LIVE — the multi-live incomplete-stop shape.
    events_mod._close_campaign_after_owner_stop(exclude_task_id="evo-a")
    assert el._read_evolution_campaign().get("status") == "active", (
        "GR4-6: the backstop must not close over another live evolution task"
    )

    # Once only the settling task remains, the deferred close becomes honest.
    q.RUNNING.pop("evo-b")
    events_mod._close_campaign_after_owner_stop(exclude_task_id="evo-a")
    closed = el._read_evolution_campaign()
    assert closed.get("status") == "stopped"


# --------------------------------------------------------------------------
# GR4-7 — the cascade auto-release requires a generation fence
# --------------------------------------------------------------------------
# GR4-8 — corrupt-projection refusals are typed and honest
# --------------------------------------------------------------------------


def test_gr4_8_corrupt_projection_cancel_refusal_is_typed_and_honest(tmp_path, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.tasks import api_task_cancel

    from supervisor import queue, workers

    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", [{"id": "root", "chat_id": 0, "root_task_id": "root"}])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    custody_calls: list = []
    monkeypatch.setattr(
        queue, "cancel_task_custody",
        lambda tid, **_kw: custody_calls.append(tid) or queue.CANCEL_CANCELLED,
    )
    monkeypatch.setattr(
        queue, "cancel_task_by_id",
        lambda tid, **_kw: custody_calls.append(tid) or True,
    )
    monkeypatch.setattr(
        "ouroboros.cancel_intents.request_cancel",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            ci.CancelIntentProjectionCorrupt("not an object"),
        ),
    )
    app = Starlette(routes=[
        Route("/api/tasks/{task_id}/cancel", api_task_cancel, methods=["POST"]),
    ])
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        plain = client.post("/api/tasks/root/cancel")
        cascade = client.post("/api/tasks/root/cancel", json={"cascade": True})

    for response in (plain, cascade):
        assert response.status_code == 503
        body = response.json()
        assert body["reason_code"] == "cancel_intent_projection_corrupt", (
            "GR4-8: the corrupt-projection refusal is a DISTINCT typed reason"
        )
        assert "cannot succeed until the file is repaired" in body["error"]
        assert "preserved" in body["error"]
        assert "projection_corrupt_refused" in body["error"]
    assert custody_calls == [], "no teardown without the durable intent"


def test_gr4_8_transient_intent_write_failure_keeps_the_retry_wording(tmp_path, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.tasks import api_task_cancel

    from supervisor import queue, workers

    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", [{"id": "root", "chat_id": 0, "root_task_id": "root"}])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(
        "ouroboros.cancel_intents.request_cancel",
        lambda *_a, **_kw: (_ for _ in ()).throw(OSError("intent store io")),
    )
    app = Starlette(routes=[
        Route("/api/tasks/{task_id}/cancel", api_task_cancel, methods=["POST"]),
    ])
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        response = client.post("/api/tasks/root/cancel")
    assert response.status_code == 503
    body = response.json()
    assert body["reason_code"] == "cancel_intent_write_failed"
    assert "retry" in body["error"]


def test_gr4_8_agent_tool_names_the_corrupt_projection(tmp_path, monkeypatch):
    """The agent `cancel_task` tool refuses the same class with the distinct
    marker and honest wording (never a false 'Retry')."""
    from ouroboros.tools import join_ledger as jl

    monkeypatch.setattr(jl, "_status_drive_root", lambda ctx: tmp_path)
    monkeypatch.setattr(jl, "_is_own_child", lambda ctx, root, tid: False)
    monkeypatch.setattr(
        "ouroboros.cancel_intents.request_cancel",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            ci.CancelIntentProjectionCorrupt("not an object"),
        ),
    )
    ctx = types.SimpleNamespace(task_id="parent")

    out = jl._cancel_task(ctx, "someid1")

    assert "CANCEL_INTENT_PROJECTION_CORRUPT" in out
    assert "cannot succeed until the file is repaired" in out
    assert "projection_corrupt_refused" in out
    assert "Retry, or report" not in out
