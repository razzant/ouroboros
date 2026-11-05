"""GATE ROUND-5 (v6.98.0 phase A) — owner-audit-driven micro-fix regression tests.

GR5-1  a FAILED ``/evolve start`` / ``toggle_evolution(True)`` restores the
       CAPTURED prior ``evolution_owner_stopped`` value (both ingresses) —
       the GR4-6 pre-mint clear must not outlive a start that never happened,
       or the post-task promotion pipeline can re-arm evolution the owner
       believes is off;
GR5-2  the timeout reaper reconciles the killed task's open DELEGATED runs
       (same custody seam as the cancel kill path) and discloses what stayed
       open on the reap outcome;
GR5-3  the fast already-settled and finalize-on-miss cancel paths run the
       same open-run audit and thread ``unreconciled_runs`` into the
       miss-lane delivery — a cancel over a dead task with live delegated
       runs must not read as a clean completion;
GR5-4  cascade digest MEMBERSHIP comes from the durable tree merged with the
       sweep outcomes — a watchdog replay after the children terminalized
       (empty outcomes) still lists them;
GR5-5  project deletion cascades only the lineage ROOTS of the live set —
       descendants fall with their trees (one cascade, one summary per
       tree); an orphan child without a live root still gets its own;
GR5-6  nested registry strictness, cheap form: a present-but-non-dict
       ``pending`` / ``intents`` under a valid top-level dict refuses the
       mutation exactly like top-level corruption (no overwrite), and the
       read paths distinguish "file absent" (empty, silent) from
       "unreadable/malformed" (loud typed disclosure, then empty).
"""

from __future__ import annotations

import json
import logging
import pathlib
import types



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


# --------------------------------------------------------------------------
# GR5-1 — a failed evolution start restores the captured owner-stop flag
# --------------------------------------------------------------------------


def _toggle_ctx(state, sent):
    return types.SimpleNamespace(
        load_state=state.load_state,
        send_with_budget=lambda cid, text, **kw: sent.append(text),
    )


def test_gr5_1_toggle_start_failure_restores_a_prior_owner_stop(tmp_path, monkeypatch):
    """Prior flag True + failed start → the flag is True again afterwards, so
    ``apply_pending_request`` keeps refusing the autonomous re-arm."""
    import supervisor.state as state
    from supervisor import events as events_mod
    from supervisor import evolution_lifecycle as el

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    state.update_state(lambda live: live.update(
        owner_chat_id=7, evolution_owner_stopped=True,
    ))
    monkeypatch.setattr(el, "evolution_block_reason", lambda: "")
    monkeypatch.setattr(
        el, "start_evolution_campaign",
        lambda objective, source="": (_ for _ in ()).throw(OSError("campaign store io")),
    )
    sent: list = []

    # An OWNER start reaches the mint (В12: the agent tool is refused outright
    # while the flag stands — pinned in test_post_task_evolution).
    events_mod._handle_toggle_evolution(
        {"enabled": True, "objective": "x", "source": "owner_chat"}, _toggle_ctx(state, sent),
    )

    assert bool(state.load_state().get("evolution_owner_stopped")) is True, (
        "GR5-1: a failed start must restore the captured owner-stop flag"
    )
    assert any("stayed OFF" in text for text in sent)


def test_gr5_1_toggle_start_failure_with_no_prior_stop_stays_false(tmp_path, monkeypatch):
    """Prior flag False + refused start ("Evolution stayed OFF" branch) → the
    flag stays False: an unconditional True would invent an owner stop."""
    import supervisor.state as state
    from supervisor import events as events_mod
    from supervisor import evolution_lifecycle as el

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    state.update_state(lambda live: live.update(
        owner_chat_id=7, evolution_owner_stopped=False,
    ))
    monkeypatch.setattr(el, "evolution_block_reason", lambda: "")
    monkeypatch.setattr(el, "start_evolution_campaign", lambda objective, source="": {})
    sent: list = []

    events_mod._handle_toggle_evolution({"enabled": True, "objective": "x"}, _toggle_ctx(state, sent))

    assert bool(state.load_state().get("evolution_owner_stopped")) is False, (
        "GR5-1: no owner stop preceded — the restore must not fabricate one"
    )
    assert any("stayed OFF" in text for text in sent)


def test_gr5_1_server_evolve_start_failure_restores_the_captured_flag():
    """The owner-chat `/evolve` ingress runs the shared owner start transaction; its
    capture -> clear -> (failure) restore ordering is pinned at the source, and the
    restore puts back an UNKNOWN prior as unknown (#1307)."""
    import inspect

    from supervisor.events_runtime_controls import owner_evolution_start

    src = (REPO_ROOT / "server.py").read_text(encoding="utf-8")
    assert 'owner_evolution_start(objective, source="owner_chat")' in src
    body = inspect.getsource(owner_evolution_start)
    capture_at = body.index('prior["known"], prior["value"] = control_value(live, "evolution_owner_stopped")')
    clear_at = body.index("update_state(_clear_owner_stop")
    start_at = body.index("start_evolution_campaign(")
    restore_at = body.index("update_state(_restore)")
    stayed_off_at = body.index("Evolution stayed OFF: campaign state could not be created.")
    assert capture_at < clear_at < start_at < restore_at < stayed_off_at
    assert 'mark_unconfirmed(live, "evolution_owner_stopped")' in body


# --------------------------------------------------------------------------
# GR5-3 — miss/fast cancel paths audit + disclose delegated runs
# --------------------------------------------------------------------------
# GR5-4 — replay-shape digest membership comes from the durable tree
# --------------------------------------------------------------------------
# GR5-5 — project deletion cascades lineage roots only
# --------------------------------------------------------------------------


def test_gr5_5_mid_tree_live_ancestor_covers_its_branch(tmp_path, monkeypatch):
    """The recorded root already settled, but a mid-tree parent is live: the
    parent-chain walk must still cover the grandchild (one cascade for the
    live branch, rooted at the mid-tree ancestor)."""
    import supervisor.queue as q
    from supervisor import queue_transitions as qt

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "PENDING", [
        {"id": "mid", "parent_task_id": "settled-root", "root_task_id": "settled-root",
         "project_id": "p5m"},
        {"id": "leaf", "parent_task_id": "mid", "root_task_id": "settled-root"},
    ])
    monkeypatch.setattr(q, "RUNNING", {}, raising=False)

    assert set(qt._live_project_task_ids(tmp_path, "p5m", roots_only=True)) == {"mid"}


# --------------------------------------------------------------------------
# GR5-6 — nested registry strictness (cheap form)
# --------------------------------------------------------------------------


def test_gr5_6_nested_corrupt_pending_refuses_the_mutation(tmp_path):
    """A valid top-level dict whose ``pending`` is not an object used to be
    coerced to {} — the next mutation overwrote every owed row silently. The
    mutators now refuse it exactly like top-level corruption: no overwrite,
    typed disclosure."""
    from supervisor import terminal_delivery as td

    store = tmp_path / "state" / "terminal_deliveries.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(
        json.dumps({"schema_version": 2, "delivered": ["final:old:aa"], "pending": [1, 2]}),
        encoding="utf-8",
    )
    before = store.read_text(encoding="utf-8")

    assert td.register_pending_delivery(tmp_path, {
        "type": "send_message", "chat_id": 1, "task_id": "n6",
        "text": "x", "delivery_id": "final:n6:abc",
    }) is False
    assert store.read_text(encoding="utf-8") == before, "no overwrite"
    assert td.register_delivery(tmp_path, "final:n6:abc") is True, (
        "register_delivery keeps its fail-open contract on a refused mutation"
    )
    assert store.read_text(encoding="utf-8") == before, "no overwrite"
    corrupt_events = [
        r for r in _event_rows(tmp_path)
        if r.get("type") == "terminal_delivery_registry_corrupt"
    ]
    assert len(corrupt_events) >= 2, "each refused mutation is a loud typed event"


def test_gr5_6_corrupt_outbox_read_is_loud_and_empty(tmp_path, caplog):
    from supervisor import terminal_delivery as td

    store = tmp_path / "state" / "terminal_deliveries.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("[1, 2, 3]", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="supervisor.terminal_delivery"):
        assert td.pending_deliveries(tmp_path) == []
    assert any(
        "unreadable/malformed" in record.message for record in caplog.records
    )

    # The replay (watchdog) read discloses through the existing typed event.
    assert td.replay_pending_deliveries(tmp_path, event_queue=_CaptureQueue()) == []
    assert any(
        r.get("type") == "terminal_delivery_registry_corrupt"
        and r.get("op") == "pending_deliveries_read"
        for r in _event_rows(tmp_path)
    )
