"""The alarm clock of Background Consciousness (redesign P2): ``tick`` decides WHEN an
ordinary Main turn starts on its own; the wake's RUNNING row under the conversation lane (or a
disclosed inline executor's registered actor) is the only liveness truth; nothing but the
schedule moves the next wake (#1603); a wake's own finish never re-arms it. The clock admits
with a placeholder and the facts only its process holds; the carrier's first step renders and
binds the observation, and the accepted boundary reaches the clock with the wake's pooled
terminal (owner decision 1A). The lane that runs the turn is pinned in
``test_consciousness_wake_lane.py``.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from ouroboros import consciousness as clock_module
from ouroboros import dialogue_store
from ouroboros.consciousness import (
    ARCHIVED_INBOX_REL,
    INTERVAL_STATE_KEY,
    LAST_SKIP_STATE_KEY,
    LAST_WAKE_STATE_KEY,
    LEGACY_INBOX_REL,
    NEXT_WAKE_STATE_KEY,
    BackgroundConsciousness,
)
from supervisor.active_activity import get_direct_activity_registry
from tests._store_shared import seed_result

T0 = 1_800_000_000.0


def _iso(ts):
    return clock_module._iso(ts)
FLOOR, CEILING, DEFAULT = 900, 14400, 3300
AVAILABLE = {"status": "available", "limit_usd": 20.0, "settled_usd": 2.5, "accounted_usd": 2.5,
             "remaining_usd": 17.5, "resets_at": "", "unknown_unmetered": 0}


@pytest.fixture
def clock(monkeypatch, tmp_path):
    """A constructed alarm clock over an in-memory runtime state and a stubbed lane.

    The stub admits as ``worker_chat_lane.handle_wake_direct`` does: the clock's binder runs once the
    wake has its id, the RUNNING row under the conversation lane is the wake's custody
    (``consciousness_carriers.queued_wakes``), and the server renders nothing. The carrier's first
    step (``consciousness_wake.prepare_wake`` over the clock's ``_wake_prepare`` facts) then runs in
    place, so ``launch["text"]`` is the rendered observation and ``launch["task"]`` the prepared task,
    while ``launch["admitted_text"]`` is the clock's placeholder. ``clock.finish(task_id, ok)`` is the
    wake's pooled terminal as the task_done handler delivers it (``finish_pooled_wake`` over the
    carrier's terminal row, then the row's release); the lane calls no finish callback.
    """
    from supervisor import queue, state, workers
    from supervisor.consciousness_carriers import finish_pooled_wake
    from ouroboros.consciousness_wake import prepare_wake
    from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION
    from ouroboros.task_results import write_task_result
    from tests._store_shared import read_result, result_exists

    store = {"bg_consciousness_enabled": True, "owner_chat_id": 7}
    monkeypatch.setattr(state, "load_state", lambda: dict(store))

    def update_state(mutator, **_kwargs):
        mutator(store)
        return dict(store)

    monkeypatch.setattr(state, "update_state", update_state)
    monkeypatch.setenv("OUROBOROS_BG_WAKEUP_MIN", str(FLOOR))
    monkeypatch.setenv("OUROBOROS_BG_WAKEUP_MAX", str(CEILING))
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "act")
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "50")
    monkeypatch.delenv("OUROBOROS_CONSCIOUSNESS_MAX_TASKS", raising=False)
    (tmp_path / "logs").mkdir()
    (tmp_path / "repo" / "prompts").mkdir(parents=True)
    (tmp_path / "repo" / "prompts" / "CONSCIOUSNESS.md").write_text(
        "[Wake-up · {reason}] since {last_wake_ago}: {events} level={level} withheld={withheld_tools} "
        "spent={spent_usd}/{daily_usd} running={running}/{max_tasks} interval={interval} line={level_line}",
        encoding="utf-8")
    running: dict = {}
    pending: list = []
    for module in (workers, queue):  # the pool the clock reads its own wake from
        monkeypatch.setattr(module, "RUNNING", running)
        monkeypatch.setattr(module, "PENDING", pending)
    launches: list = []
    receipt = {"admitted": True, "task_id": "wake0001", "reason": ""}

    def handle_wake_direct(chat_id, text, task_metadata, on_finished=None, bind_input=None):
        launch = {"chat_id": chat_id, "text": text, "admitted_text": text, "metadata": task_metadata,
                  "on_finished": on_finished}
        launches.append(launch)
        if not receipt.get("admitted"):
            return dict(receipt)
        task = {"id": receipt["task_id"], "type": "task", "chat_id": chat_id, "text": text, "_is_direct_chat": True,
                "metadata": dict(task_metadata)}
        if bind_input is not None:  # the real lane binds once the wake has its id, before its row exists
            bind_input(task)
        task["_execution_owner"] = {"kind": "direct", "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER}
        running[task["id"]] = {"task": dict(task), "worker_id": 90, "attempt": 1, "lane": LANE_CONVERSATION,
                               "started_at": time.time(), "last_heartbeat_at": time.time()}
        launch["task"], launch["prepare"] = task, dict(task.get("_wake_prepare") or {})
        try:
            prepare_wake(task, tmp_path)  # the carrier's first step (worker_chat_lane.prepare_conversation_task)
        except Exception:
            task.pop("_wake_prepare", None)  # the admitted text stays the wake's only input
        launch["text"] = task["text"]
        return dict(receipt)

    def finish(task_id, ok, *, alarm=None, **fields):
        """The pooled terminal: the carrier's row (unless the test wrote one), the clock's settlement, the release."""
        launch = next((entry for entry in launches if (entry.get("task") or {}).get("id") == task_id), None)
        task = (launch or {}).get("task") or {}
        if launch is not None and not result_exists(tmp_path, task_id):
            write_task_result(tmp_path, task_id, "completed" if ok else "failed", chat_id=7, _is_direct_chat=True,
                              metadata=task.get("metadata") or {}, execution_owner=task.get("_execution_owner"),
                              **fields)
        row = read_result(tmp_path, task_id) or {}
        finish_pooled_wake(SimpleNamespace(consciousness=alarm or clock), task_id, row, dict(task))
        running.pop(task_id, None)

    monkeypatch.setattr(workers, "handle_wake_direct", handle_wake_direct)
    monkeypatch.setattr(clock_module, "allowance_window", lambda root, now=None, **_display_read: dict(AVAILABLE))
    monkeypatch.setattr("supervisor.queue.live_consciousness_roots", lambda: [
        {"task_id": "root", "title": "Root", "consciousness_cause": "started"}])
    get_direct_activity_registry().clear()
    clock = BackgroundConsciousness(tmp_path, tmp_path / "repo", lambda: store.get("owner_chat_id"), now=T0)
    yield SimpleNamespace(clock=clock, store=store, launches=launches, receipt=receipt, root=tmp_path,
                          running=running, finish=finish)
    get_direct_activity_registry().clear()


def _events(root):
    path = root / "logs" / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def _register(task_id, *, initiator=""):
    actor = SimpleNamespace(_busy=True, _current_task_id=task_id, _accepting_owner_messages=True,
                            _current_task_metadata={"initiator": initiator, "wake_cause": "own_clock"} if initiator else {},
                            _current_chat_id=7, _current_task_text="x", _task_started_ts=T0)
    return get_direct_activity_registry().register(task_id, 7, actor=actor)


# --- boot -----------------------------------------------------------------------


def test_headline_while_the_launched_own_wake_runs(clock, monkeypatch):
    from ouroboros import consciousness_view as view

    due = clock.clock.next_wake_at
    assert clock.clock.tick(now=due + 1) == "launched"
    assert set(clock.running) == {"wake0001"}  # the lane's RUNNING row is the wake's liveness
    monkeypatch.setattr(view, "time", SimpleNamespace(time=lambda: due + 120))
    snapshot = clock.clock.status_snapshot()
    assert snapshot["live_wake_task_id"] == "wake0001"
    assert snapshot["next_wake_at"] == _iso(due)
    described = view.describe_consciousness_state(snapshot, True)
    assert described["status"] == "thinking"
    assert described["detail"] == "Wake-up wake0001 is running."


def test_status_details_and_launch_use_only_stored_places(clock, monkeypatch):
    from ouroboros import consciousness_view as view, places

    place = places.connect_place(clock.root, connector="bridge", provider="chat", label="Research room")
    places.edit_place(clock.root, place.place_id, "add_note", author="owner", text="Watch releases.")
    reads = []

    def forbidden(*args, **kwargs):
        pytest.fail("status and launch must not discover connectors or reread the allowance")

    for target in ("ouroboros.skill_loader.discover_skills", "ouroboros.place_projection.place_declarations",
                   "ouroboros.place_arrivals.account_candidates", "ouroboros.place_projection.migration_problems",
                   "ouroboros.consciousness_allowance.allowance_window", "ouroboros.place_projection.place_activity"):
        monkeypatch.setattr(target, forbidden)
    # One index pass per place: open turns and the newest arrival come from the room index.
    monkeypatch.setattr(places, "place_room_index", lambda root, place_id: [
        {"room_chat_id": 123, "unanswered_count": 1, "last_at": "2027-01-15T11:00:00Z"},
        {"room_chat_id": 124, "unanswered_count": 2, "last_at": "2027-01-15T12:00:00Z"}])
    original = view.place_views

    def project(*args, **kwargs):
        result = original(*args, **kwargs)
        reads.append(result)
        return result

    monkeypatch.setattr(view, "place_views", project)
    window = {**AVAILABLE, "spend_by_cause": [{"cause": "place:" + place.place_id, "usd": 1.25, "unmetered_calls": 0}]}
    monkeypatch.setattr(clock_module, "allowance_window", lambda *a, **kw: window)
    snapshot = clock.clock.status_snapshot()
    assert snapshot["places_brief"] == [{"place_id": place.place_id, "label": "Research room", "state": "active", "unanswered": 3}]
    assert len(reads) == 1 and reads[0][0]["spent_24h_usd"] == 1.25
    assert reads[0][0]["notes"][0]["note_id"]
    assert reads[0][0]["last_activity_at"] == "2027-01-15T12:00:00Z"
    assert not {"icon", "readiness", "account_candidates"} & reads[0][0].keys()
    reads.clear()
    panel = clock.clock.details_snapshot(True)
    assert len(reads) == 1 and "places" not in panel
    clock.root.joinpath("repo/prompts/CONSCIOUSNESS.md").write_text(
        "Places: {places}", encoding="utf-8")
    assert clock.clock.tick(clock.clock.next_wake_at) == "launched"
    text = clock.launches[-1]["text"]
    assert f"Research room [{place.place_id}]: active; open turns: 3; rooms: 2" in text
    assert "note (owner): Watch releases." in text and "a place's rooms: configure_presence(action='inspect'," in text


@pytest.mark.parametrize("captured", [True, False])
def test_wake_identity_uses_its_own_root_not_configured_root(clock, monkeypatch, captured):
    from ouroboros import places
    from tests.test_place_arrivals import seed_identity_arrival

    other_root = clock.root / "other-installation"
    for root, label in [(clock.root, "Wake mailbox"), (other_root, "Other mailbox")]:
        place = places.connect_place(root, place_id="mail", connector="bridge", provider="mail", account_id="account")
        if root == other_root or captured:
            seed_identity_arrival(root, place, {"sends_as": {"id": "writer", "label": label, "kind": "user"}})
    monkeypatch.setattr("ouroboros.config.DATA_DIR", other_root)
    clock.root.joinpath("repo/prompts/CONSCIOUSNESS.md").write_text("Places: {places}", encoding="utf-8")

    assert clock.clock.tick(clock.clock.next_wake_at) == "launched"
    launch = clock.launches[-1]
    for text in (launch["text"], launch["task"]["text"]):
        assert ("writes as Wake mailbox" in text) is captured
        assert "Other mailbox" not in text
        assert ("writes as" in text) is captured


def test_boot_floor_never_wakes_in_the_first_second(clock):
    """An overdue persisted next_wake_at is pushed past now + floor; a later one is kept."""
    assert clock.clock.next_wake_at == T0 + FLOOR
    clock.store[NEXT_WAKE_STATE_KEY] = T0 + 5000
    later = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7, now=T0)
    assert later.next_wake_at == T0 + 5000
    assert later.enabled is True


def test_boot_restores_the_previous_wake_boundary_for_context_since_window(clock):
    clock.store[LAST_WAKE_STATE_KEY] = T0 - 3600
    later = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7, now=T0)
    assert later.status_snapshot()["last_wake_at"].startswith("2027-")
    assert later._last_wake_at == T0 - 3600


def test_boot_discloses_invalid_wake_boundary_and_uses_process_start(caplog, clock):
    for raw in ("not-a-timestamp", "NaN", "Infinity", T0 + 3600):
        clock.store[LAST_WAKE_STATE_KEY] = raw
        with caplog.at_level("WARNING"):
            later = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7, now=T0)
        assert later._last_wake_at == 0.0
    assert any("invalid persisted last wake boundary" in record.message for record in caplog.records)


def _chat_row(**row):
    """A chat row the dialogue writers would have stored, in the canonical store."""
    return dialogue_store.append(row.pop("root"), "chat", row)


def _observed(clock):
    """The last launched wake's observation as its bound source stores it: every event line whole (its text
    counts my human's words, which my view holds)."""
    from ouroboros.artifacts import read_actor_source_bytes

    source = clock.launches[-1]["task"]["metadata"]["wake_observation"]["source"]
    return read_actor_source_bytes(clock.root, source["task_id"], source).decode("utf-8")


def test_restored_boundary_reaches_the_launched_wake_text(clock):
    """Without an accepted observation boundary yet, the first window starts at the last wake."""
    clock.store[LAST_WAKE_STATE_KEY] = T0 - 3600
    (clock.root / "task_results").mkdir()
    seed_result(clock.root, "settled", {
        "task_id": "settled", "status": "completed", "updated_at": _iso(T0 - 1800),
        "ts": _iso(T0 - 1800), "description": "settled before restart", "_schema_version": 1,
    })
    _chat_row(root=clock.root, ts=_iso(T0 - 7200), direction="system", type="task_summary", task_id="older",
              status="completed")
    _chat_row(root=clock.root, ts=_iso(T0 - 1800), direction="system", type="task_summary", task_id="settled",
              status="completed")
    later = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7, now=T0)
    assert later.tick(T0 + FLOOR + 1) == "launched"
    assert "- task settled Done" in clock.launches[-1]["text"]
    assert "older" not in clock.launches[-1]["text"]
    assert "no wake since this process started" not in clock.launches[-1]["text"]


def test_an_accepted_wake_advances_the_observation_boundary_and_a_refused_one_does_not(clock):
    from ouroboros.consciousness import OBSERVATION_STATE_KEY

    _chat_row(root=clock.root, ts=_iso(T0 + 5), direction="in", chat_id=1, source="web", text="first")
    clock.receipt.update({"admitted": False, "task_id": "", "reason": "budget_exhausted"})
    assert clock.clock.tick(T0 + FLOOR + 1) == "rejected:budget_exhausted"
    assert OBSERVATION_STATE_KEY not in clock.store  # a refused launch consumed nothing
    clock.receipt.update({"admitted": True, "task_id": "wake0001", "reason": ""})
    clock.clock._next_wake_at = T0 + FLOOR + 2
    assert clock.clock.tick(T0 + FLOOR + 2) == "launched"
    assert clock.launches[-1]["admitted_text"] == "[Wake-up · heartbeat]"  # the clock admits a placeholder
    assert '] first' in _observed(clock)  # the refused window is observed again, whole, by the carrier
    assert "1 owner message (counted, not listed: my view holds the words)" in clock.launches[-1]["text"]
    # The carrier bound the observation to the admitted wake before its turn could run, and accepted
    # the boundary on the task; the clock persists nothing until the wake's terminal carries it.
    bound = clock.launches[-1]["task"]["metadata"]["wake_observation"]
    assert bound["composition"] == {"owner_message": 1} and bound["source"]["sha256"]
    accepted = bound["accepted_boundary"]
    # The boundary is the chain coordinate the next row takes (the file size, when the chain was a file).
    assert accepted["task_id"] == "wake0001" and accepted["upper"] == dialogue_store.stream_end(clock.root, "chat")
    assert OBSERVATION_STATE_KEY not in clock.store
    # A line appended while that wake runs — stamped even before its capture — reaches the next wake.
    _chat_row(root=clock.root, ts=_iso(T0 + 10), direction="in", chat_id=1, source="web", text="second")
    clock.finish("wake0001", True)
    assert clock.store[OBSERVATION_STATE_KEY] == accepted  # persisted at the terminal, from the terminal row
    clock.receipt["task_id"] = "wake0002"
    clock.clock._next_wake_at = T0 + 2 * FLOOR
    assert clock.clock.tick(T0 + 2 * FLOOR) == "launched"
    observed = _observed(clock)
    assert '] second' in observed and '] first' not in observed


def test_a_refused_wake_consumes_no_transition_and_the_admitted_one_persists_observed_identities(clock):
    """The accepted boundary carries the task-result transition state beside the chat position:
    a child that settles with no chat row is reported by the next ADMITTED wake, not lost to a
    refused one. The complete inventory lives in the accepted transition chain."""
    from ouroboros.consciousness import OBSERVATION_STATE_KEY
    from ouroboros.consciousness_transitions import read_chain

    (clock.root / "task_results").mkdir()
    row = {"task_id": "kid", "status": "running", "ts": _iso(T0 - 100), "updated_at": _iso(T0 - 100),
           "_schema_version": 1, "metadata": {}, "description": "child work"}
    seed_result(clock.root, "kid", row)
    assert clock.clock.tick(T0 + FLOOR + 1) == "launched"
    accepted = clock.launches[-1]["task"]["metadata"]["wake_observation"]["accepted_boundary"]["transition_chain"]
    inventory = read_chain(clock.root, accepted).state
    assert inventory["inventory"] == {"kid": []} and inventory["scan_at"] == _iso(T0 + FLOOR + 1)
    seed_result(clock.root, "kid", {**row, "status": "failed", "updated_at": _iso(T0 + FLOOR + 50)})  # the orphan sweep
    clock.finish("wake0001", True)
    assert clock.store[OBSERVATION_STATE_KEY]["transition_chain"] == accepted  # the terminal carried the accepted identities
    clock.receipt.update({"admitted": False, "task_id": "", "reason": "budget_exhausted"})
    clock.clock._next_wake_at = T0 + 2 * FLOOR
    assert clock.clock.tick(T0 + 2 * FLOOR) == "rejected:budget_exhausted"
    assert clock.store[OBSERVATION_STATE_KEY]["transition_chain"] == accepted
    clock.receipt.update({"admitted": True, "task_id": "wake0002", "reason": ""})
    clock.clock._next_wake_at = T0 + 3 * FLOOR
    assert clock.clock.tick(T0 + 3 * FLOOR) == "launched"
    assert "- task kid Failed, completion time not recorded" in clock.launches[-1]["text"]
    clock.finish("wake0002", True)
    # The carried wake's own terminal row is a transition like any task's.
    chain = read_chain(clock.root, clock.store[OBSERVATION_STATE_KEY]["transition_chain"])
    assert set(chain.state["inventory"]) == {"kid", "wake0001"}


@pytest.mark.parametrize('failure', ['write', 'corrupt_previous', 'missing_previous'])
def test_inventory_source_failure_never_advances_accepted_boundary(clock, monkeypatch, failure):
    from ouroboros import artifacts, chat_chain
    from ouroboros.consciousness import OBSERVATION_STATE_KEY
    from ouroboros.consciousness_transitions import CHAIN_REL

    _chat_row(root=clock.root, ts=_iso(T0 + 1), direction='in', chat_id=1, source='web', text='first')
    assert clock.clock.tick(T0 + FLOOR + 1) == 'launched'
    clock.finish('wake0001', True)
    accepted = clock.store[OBSERVATION_STATE_KEY]
    path = clock.root / CHAIN_REL / f"{accepted['transition_chain']['link']}.json"
    original = path.read_bytes()
    _chat_row(root=clock.root, ts=_iso(T0 - 100), direction='in', chat_id=1, source='web', text='late old row')
    clock.receipt['task_id'] = 'wake0002'
    with monkeypatch.context() as fault:
        if failure == 'corrupt_previous':
            path.write_bytes(b'corrupt')
        elif failure == 'missing_previous':
            path.unlink()
        elif failure == 'write':
            fault.setattr(chat_chain, 'retain_memory_source', lambda *a, **kw: (_ for _ in ()).throw(OSError('write failed')))
        else:
            read = artifacts.read_actor_source_bytes
            fault.setattr(artifacts, 'read_actor_source_bytes', lambda root, task, ref: read(root, task, ref)
                          if task == 'wake0001' else b'bad readback')
        clock.clock._next_wake_at = T0 + 2 * FLOOR
        assert clock.clock.tick(T0 + 2 * FLOOR) == 'launched'
        if failure in {'corrupt_previous', 'missing_previous'}:
            assert 'unreadable_transition_source' in clock.launches[-1]['text']
        assert 'accepted_boundary' not in clock.launches[-1]['task']['metadata']['wake_observation']
        clock.finish('wake0002', True)  # its terminal carries no accepted boundary: nothing advances
        assert clock.store[OBSERVATION_STATE_KEY] == accepted
    path.write_bytes(original)
    clock.receipt['task_id'] = 'wake0003'
    clock.clock._next_wake_at = T0 + 3 * FLOOR
    assert clock.clock.tick(T0 + 3 * FLOOR) == 'launched'
    assert 'late old row' in _observed(clock)
    clock.finish('wake0003', True)
    assert clock.store[OBSERVATION_STATE_KEY] != accepted


@pytest.mark.parametrize('version', [1, 2])
def test_inline_inventory_migrates_to_a_base_of_the_transition_chain(clock, version):
    """An inline inventory (v1, a late-review wake of a closed task left with the acceptance
    machine; or v2) migrates to a base of the transition chain the boundary names; the migration stays."""
    from ouroboros import consciousness_wake as wake
    from ouroboros.consciousness import OBSERVATION_STATE_KEY
    from ouroboros.consciousness_transitions import read_chain
    from tests.test_consciousness_wake import _write

    _write(clock.root, 'old', status='completed', ts=_iso(T0 - 86400))
    prior = wake.observe_wake(clock.root, boundary=None, since=T0 - 10, now=T0).boundary
    if version == 1:
        prior['transitions'] = {'version': 1, 'inventory': {}, 'observed': [], 'scan_at': _iso(T0)}
    clock.store[OBSERVATION_STATE_KEY] = prior
    assert clock.clock.tick(T0 + FLOOR + 1) == 'launched'
    assert ('task old Done' in clock.launches[-1]['text']) is (version == 1)
    clock.finish('wake0001', True)
    accepted = clock.store[OBSERVATION_STATE_KEY]
    chain = read_chain(clock.root, accepted['transition_chain'])
    assert 'transitions' not in accepted and accepted['transition_chain']['version'] == 3
    assert len(chain.names) == 1 and set(chain.state['inventory']) == {'old'}  # one base, the whole state


@pytest.mark.parametrize(("restart", "failure"), [(False, 'readback'), (True, 'write'), (True, 'bootstrap_write')])
def test_first_failed_source_preserves_intervening_events(clock, monkeypatch, failure, restart):
    from ouroboros import artifacts, chat_chain
    from ouroboros.consciousness import OBSERVATION_BOOTSTRAP_STATE_KEY, OBSERVATION_STATE_KEY
    from tests.test_consciousness_wake import _write

    _chat_row(root=clock.root, ts=_iso(T0 - 1), direction='in', chat_id=1, source='web', text='before bootstrap')
    with monkeypatch.context() as fault:
        if failure in {'write', 'bootstrap_write'}:
            fault.setattr(chat_chain, 'retain_memory_source', lambda *a, **kw: (_ for _ in ()).throw(OSError('write failed')))
        else:
            fault.setattr(artifacts, 'read_actor_source_bytes', lambda *a, **kw: b'bad readback')
        if failure == 'bootstrap_write':
            from supervisor import state

            update = state.update_state

            def fail_bootstrap(mutator):
                staged = dict(clock.store)
                mutator(staged)
                if OBSERVATION_BOOTSTRAP_STATE_KEY in staged:
                    raise OSError('bootstrap write failed')
                return update(mutator)

            fault.setattr(state, 'update_state', fail_bootstrap)
        assert clock.clock.tick(T0 + FLOOR) == 'launched'
        assert not clock.store.get(OBSERVATION_STATE_KEY)
    event_at, finish_at = T0 + FLOOR + 1, T0 + FLOOR + 2
    _chat_row(root=clock.root, ts=_iso(event_at), direction='in', chat_id=1, source='web', text='during first wake')
    _write(clock.root, 'intervening', status='completed', ts=_iso(event_at), updated_at=_iso(event_at))
    monkeypatch.setattr(clock_module.time, 'time', lambda: finish_at)
    clock.finish('wake0001', not restart)
    assert not clock.store.get(OBSERVATION_STATE_KEY)  # the failed source rode no accepted boundary to the clock
    assert clock.store[OBSERVATION_BOOTSTRAP_STATE_KEY] == T0  # nothing accepted: the bootstrap stays
    alarm = BackgroundConsciousness(clock.root, clock.root / 'repo', lambda: 7, now=finish_at + 1) if restart else clock.clock
    clock.receipt['task_id'] = 'wake0002'
    assert alarm.tick(alarm.next_wake_at) == 'launched'
    assert 'during first wake' in _observed(clock)
    assert 'task intervening Done' in clock.launches[-1]['text']
    assert 'before bootstrap' not in _observed(clock)
    clock.finish('wake0002', True, alarm=alarm)
    assert clock.store[OBSERVATION_STATE_KEY]
    # An accepted boundary moves the bootstrap to its capture: a body without the chain reader starts there.
    captured = clock_module.parse_deadline_ts(clock.store[OBSERVATION_STATE_KEY]['captured_at']).timestamp()
    assert clock.store[OBSERVATION_BOOTSTRAP_STATE_KEY] == captured == alarm._observation_bootstrap_since > T0
    clock.receipt['task_id'] = 'wake0003'
    assert alarm.tick(alarm.next_wake_at + DEFAULT) == 'launched'
    assert 'during first wake' not in _observed(clock)
    assert 'task intervening Done' not in clock.launches[-1]['text']


def test_full_inventory_moves_out_of_hot_state_without_forgetting_closed_tasks(clock):
    from ouroboros.consciousness import OBSERVATION_STATE_KEY
    from ouroboros.consciousness_transitions import read_chain
    from ouroboros.task_results import write_task_result

    for index in range(160):
        write_task_result(clock.root, f'closed-{index}', 'completed', ts=_iso(T0 - 86400))
    assert clock.clock.tick(T0 + FLOOR + 1) == 'launched'
    clock.finish('wake0001', True)
    accepted = clock.store[OBSERVATION_STATE_KEY]
    assert len(json.dumps(accepted)) < 1800
    assert len(read_chain(clock.root, accepted['transition_chain']).state['inventory']) == 160
    write_task_result(clock.root, 'new-old', 'completed', ts=_iso(T0 - 86400))
    clock.receipt['task_id'] = 'wake0002'
    clock.clock._next_wake_at = T0 + 2 * FLOOR
    assert clock.clock.tick(T0 + 2 * FLOOR) == 'launched'
    assert 'new-old Done' in clock.launches[-1]['text']
    assert 'closed-0 Done' not in clock.launches[-1]['text']
    clock.finish('wake0002', True)
    clock.receipt['task_id'] = 'wake0003'
    clock.clock._next_wake_at = T0 + 3 * FLOOR
    assert clock.clock.tick(T0 + 3 * FLOOR) == 'launched'
    assert 'new-old Done' not in clock.launches[-1]['text']


def test_legacy_inbox_is_archived_once_without_being_read(clock):
    inbox = clock.root / LEGACY_INBOX_REL
    inbox.parent.mkdir(parents=True, exist_ok=True)
    inbox.write_bytes(b'{"op":"enqueue"\n not json at all\n')
    BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7, now=T0)
    assert not inbox.exists()
    assert (clock.root / ARCHIVED_INBOX_REL).read_bytes() == b'{"op":"enqueue"\n not json at all\n'
    # A second legacy file (an older archive already there) gets its own name, nothing is overwritten.
    inbox.write_bytes(b"second")
    BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7, now=T0 + 1)
    assert (clock.root / ARCHIVED_INBOX_REL).read_bytes().startswith(b'{"op":"enqueue"')
    assert (clock.root / "archive" / f"consciousness_observations_{int(T0 + 1)}.jsonl").read_bytes() == b"second"


# --- tick branches ----------------------------------------------------------------


def test_disabled_clock_does_nothing(clock):
    clock.clock.stop()
    assert clock.clock.tick(T0 + FLOOR + 1) == "disabled"
    assert clock.launches == []


def test_not_due_before_next_wake_at(clock):
    assert clock.clock.tick(T0 + FLOOR - 1) == "not_due"
    assert clock.launches == []


@pytest.mark.parametrize("carrier", ["lane_row", "inline_actor"])
def test_a_live_own_wake_defers_the_wake(clock, carrier):
    if carrier == "lane_row":  # the pool carries it: a RUNNING row under the conversation lane
        clock.running["wakeXYZ"] = {"task": {"id": "wakeXYZ", "chat_id": 7, "metadata": {
            "initiator": "consciousness", "wake_cause": "own_clock"}}, "worker_id": 90, "lane": "conversation"}
    else:  # a disclosed inline executor's registered actor
        _register("wakeXYZ", initiator="consciousness")
    assert clock.clock.tick(T0 + FLOOR + 1) == "wake_live"
    assert clock.clock.status_snapshot()["live_wake_task_id"] == "wakeXYZ"
    assert clock.launches == []


def test_allowance_unknown_skips_with_a_typed_status_and_the_floor(clock, monkeypatch):
    monkeypatch.setattr(clock_module, "allowance_window",
                        lambda root, now=None, **_display_read: {"status": "allowance_unknown", "error": "OSError: ledger"})
    now = T0 + FLOOR + 1
    assert clock.clock.tick(now) == "skipped:allowance_unknown"
    assert clock.clock.next_wake_at == now + FLOOR
    assert clock.store[NEXT_WAKE_STATE_KEY] == now + FLOOR
    snapshot = clock.clock.status_snapshot()
    assert snapshot["last_wake_outcome"] == "skipped:allowance_unknown" and "OSError" in snapshot["last_error"]
    assert [row["reason"] for row in _events(clock.root) if row["type"] == "consciousness_wake_skipped"] == ["allowance_unknown"]
    assert clock.launches == []


def test_allowance_exhausted_skips_until_the_window_frees(clock, monkeypatch):
    from ouroboros.deadline_utils import parse_deadline_ts

    resets_at = "2027-02-01T00:00:00+00:00"
    monkeypatch.setattr(clock_module, "allowance_window", lambda root, now=None, **_display_read: {
        **AVAILABLE, "status": "exhausted", "accounted_usd": 21.0, "remaining_usd": 0.0, "resets_at": resets_at})
    assert clock.clock.tick(T0 + FLOOR + 1) == "skipped:allowance_exhausted"
    assert clock.clock.next_wake_at == parse_deadline_ts(resets_at).timestamp()
    assert clock.clock.status_snapshot()["last_wake_outcome"] == "skipped:allowance_exhausted"
    # A reset instant already in the past (or none: DAILY_USD=0) still waits at least the floor.
    monkeypatch.setattr(clock_module, "allowance_window", lambda root, now=None, **_display_read: {
        **AVAILABLE, "status": "exhausted", "resets_at": ""})
    clock.clock._next_wake_at = T0
    assert clock.clock.tick(T0 + 5) == "skipped:allowance_exhausted"
    assert clock.clock.next_wake_at == T0 + 5 + FLOOR


def test_a_skipped_wake_records_the_allowance_window_it_met(clock, monkeypatch):
    """A46: "did it stop for money?" is answered by the skip itself. Its event and the one overwritten
    ``consciousness_last_skip`` key, written with the next wake time, carry the projection of the window a
    wake's preparation carries (``allowance_fact``); unknown stays null, never 0, and a later ordinary wake
    leaves the newest skip as it was."""
    resets_at = "2027-02-01T00:00:00+00:00"
    causes = [{"cause": "own", "label": "Its own wakes", "usd": 20.4, "unmetered_calls": 2}]
    exhausted = {**AVAILABLE, "status": "exhausted", "settled_usd": 20.4, "remaining_usd": 0.0,
                 "resets_at": resets_at, "unknown_unmetered": 2, "spend_by_cause": causes}
    monkeypatch.setattr(clock_module, "allowance_window", lambda root, now=None, **_display_read: dict(exhausted))
    now = T0 + FLOOR + 1
    assert clock.clock.tick(now) == "skipped:allowance_exhausted"
    met = {"settled_usd": 20.4, "unknown_unmetered": 2, "limit_usd": 20.0, "spend_by_cause": causes, "resets_at": resets_at}
    [event] = [row for row in _events(clock.root) if row["type"] == "consciousness_wake_skipped"]
    assert {key: event[key] for key in ("reason", "next_wake_at", "error", "allowance")} == {
        "reason": "allowance_exhausted", "next_wake_at": resets_at, "error": "", "allowance": met}
    assert clock.store[LAST_SKIP_STATE_KEY] == {"at": _iso(now), "reason": "allowance_exhausted",
                                                "next_wake_at": resets_at, "error": "", "allowance": met}
    assert clock.store[NEXT_WAKE_STATE_KEY] == clock.clock.next_wake_at
    # An unreadable ledger: the configured limit is known, the spend is not.
    monkeypatch.setattr(clock_module, "allowance_window", lambda root, now=None, **_display_read: {
        "status": "allowance_unknown", "error": "OSError: ledger", "limit_usd": 20.0, "settled_usd": None,
        "accounted_usd": None, "remaining_usd": None, "resets_at": "", "spend_by_cause": None})
    later = now + 10
    clock.clock._next_wake_at = later
    assert clock.clock.tick(later) == "skipped:allowance_unknown"
    skip = clock.store[LAST_SKIP_STATE_KEY]
    assert skip["reason"] == "allowance_unknown" and skip["error"] == "OSError: ledger" and skip["at"] == _iso(later)
    assert skip["allowance"] == {"settled_usd": None, "unknown_unmetered": None, "limit_usd": 20.0,
                                 "spend_by_cause": None, "resets_at": None}
    # The wake that is admitted later is prepared with the same projection, and the skip stays the newest skip.
    monkeypatch.setattr(clock_module, "allowance_window", lambda root, now=None, **_display_read: dict(AVAILABLE))
    clock.clock._next_wake_at = later + 20
    assert clock.clock.tick(later + 20) == "launched"
    assert clock.launches[-1]["prepare"]["window"] == clock_module.allowance_fact(AVAILABLE)
    assert set(clock.launches[-1]["prepare"]["window"]) == set(skip["allowance"])
    assert clock.store[LAST_SKIP_STATE_KEY] == skip


def _spent_window(resets_at, settled=20.4):
    """The ledger as ``allowance_window`` reads it, against whatever daily limit the setting holds now. ``reads``
    are the wake admission's exact reads (the tick's), ``shown`` the status view's display reads."""
    from ouroboros.config import get_consciousness_daily_usd

    def window(root, now=None, **display_read):
        limit = float(get_consciousness_daily_usd())
        (window.shown if display_read.get("forecast") else window.reads).append(now)
        return {**AVAILABLE, "limit_usd": limit, "settled_usd": settled, "accounted_usd": settled,
                "remaining_usd": max(0.0, limit - settled), "resets_at": resets_at,
                "status": "exhausted" if settled >= limit else "available"}

    window.reads, window.shown = [], []
    return window


def test_a_changed_daily_limit_retries_a_money_skip_after_min(clock, monkeypatch):
    """A46 (R2_fable_arch §1 №20): an exhausted allowance waits for its window to free, a time chosen under
    the limit it met; raising the limit in Settings left the clock asleep until that reset. A changed limit
    turns that wait into any skip's retry: the same window is re-checked no sooner than MIN after the skip,
    and admits or skips again under the new limit. The same limit stays not due, and a not-due tick reads
    only the settings (no ledger, no window)."""
    from ouroboros.deadline_utils import parse_deadline_ts

    resets_at = "2027-02-01T00:00:00+00:00"
    window = _spent_window(resets_at)
    monkeypatch.setattr(clock_module, "allowance_window", window)
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "20")
    now = T0 + FLOOR + 1
    assert clock.clock.tick(now) == "skipped:allowance_exhausted"
    assert clock.clock.next_wake_at == parse_deadline_ts(resets_at).timestamp()
    assert clock.clock.tick(now + 60) == "not_due" and len(window.reads) == 1
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "19")  # lowered: still spent, skipped again under it
    assert clock.clock.tick(now + FLOOR - 1) == "not_due" and len(window.reads) == 1
    assert clock.clock.tick(now + FLOOR) == "skipped:allowance_exhausted"
    assert clock.store[LAST_SKIP_STATE_KEY]["allowance"]["limit_usd"] == 19.0
    assert clock.clock.tick(now + FLOOR + 60) == "not_due" and len(window.reads) == 2
    # the status and its line show the time the clock will act on: the reset while the limit stands ...
    from ouroboros import consciousness_view as view

    monkeypatch.setattr(view, "time", SimpleNamespace(time=lambda: now + FLOOR + 60))
    assert clock.clock.status_snapshot()["next_wake_at"] == clock_module._iso(parse_deadline_ts(resets_at).timestamp())
    line = view.describe_consciousness_state(clock.clock.status_snapshot(), True)["detail"]
    assert line.endswith(f"next check {view._when(resets_at)}."), line
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "50")  # raised: MIN after the repeated skip, not the reset
    # ... and MIN after the skip once the limit changed
    assert clock.clock.status_snapshot()["next_wake_at"] == clock_module._iso(now + 2 * FLOOR)
    line = view.describe_consciousness_state(clock.clock.status_snapshot(), True)["detail"]
    assert line.endswith(f"next check {view._when(clock_module._iso(now + 2 * FLOOR))}."), line
    assert clock.clock.tick(now + 2 * FLOOR - 1) == "not_due" and len(window.reads) == 2
    # The status view read its own display windows, a changed limit included; a tick never read for it.
    assert len(window.shown) >= 2
    assert clock.clock.tick(now + 2 * FLOOR) == "launched"
    assert clock.launches[-1]["metadata"]["root_cost_ceiling_usd"] == pytest.approx(29.6)


def test_a_money_skip_is_rechecked_after_a_restart_from_its_key(clock, monkeypatch):
    """The skip's time and limit are restored from ``consciousness_last_skip`` when that skip still sets the
    next wake; a limit changed while the app was off still waits for MIN after boot (the boot floor)."""
    resets_at = "2027-02-01T00:00:00+00:00"
    monkeypatch.setattr(clock_module, "allowance_window", _spent_window(resets_at))
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "20")
    now = T0 + FLOOR + 1
    assert clock.clock.tick(now) == "skipped:allowance_exhausted"
    restarted = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: clock.store.get("owner_chat_id"),
                                        now=now + 10)
    assert restarted.tick(now + 20 + FLOOR) == "not_due"  # the same limit waits for the reset
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "50")  # changed while the app is off
    booted = now + 30 + FLOOR
    rebooted = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: clock.store.get("owner_chat_id"),
                                       now=booted)
    assert rebooted.tick(booted + 1) == "not_due"
    assert rebooted.tick(booted + FLOOR - 1) == "not_due"
    assert rebooted.tick(booted + FLOOR) == "launched"


@pytest.mark.parametrize("stored", [
    "not a record", {"reason": "allowance_unknown", "next_wake_at": "2027-02-01T00:00:00+00:00"},
    {"reason": "allowance_exhausted", "next_wake_at": "2027-02-01T00:00:00+00:00", "allowance": {"limit_usd": None}},
])
def test_a_restart_restores_no_recheck_from_any_other_record(clock, monkeypatch, stored):
    """Only an exhausted-allowance skip with a known time and limit that set the stored next wake is restored."""
    from ouroboros.deadline_utils import parse_deadline_ts

    clock.store.update({NEXT_WAKE_STATE_KEY: parse_deadline_ts("2027-02-01T00:00:00+00:00").timestamp(),
                        LAST_SKIP_STATE_KEY: stored})
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "50")
    restarted = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7, now=T0)
    assert restarted.tick(T0 + FLOOR + 1) == "not_due"


@pytest.mark.parametrize("before", ["allowance_unknown", "waiting_for_first_conversation", "a finished wake"])
def test_a_changed_limit_rechecks_only_a_money_skip(clock, monkeypatch, before):
    """The other side: a skip for another reason and an ordinary wake keep their time whatever the limit."""
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "20")
    now = T0 + FLOOR + 1
    if before == "allowance_unknown":
        monkeypatch.setattr(clock_module, "allowance_window", lambda root, now=None, **_display_read: {
            "status": "allowance_unknown", "error": "OSError: ledger", "limit_usd": 20.0})
        assert clock.clock.tick(now) == "skipped:allowance_unknown"
    elif before == "waiting_for_first_conversation":
        clock.store["owner_chat_id"] = None
        assert clock.clock.tick(now) == "skipped:waiting_for_first_conversation"
    else:
        assert clock.clock.tick(now) == "launched"
        clock.finish("wake0001", True)
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "50")
    assert clock.clock.tick(clock.clock.next_wake_at - 1) == "not_due"


def test_no_owner_chat_waits_for_the_first_conversation(clock):
    clock.store["owner_chat_id"] = None
    assert clock.clock.tick(T0 + FLOOR + 1) == "skipped:waiting_for_first_conversation"
    assert clock.clock.next_wake_at == T0 + FLOOR + 1 + FLOOR
    assert clock.launches == []


# --- launch -----------------------------------------------------------------------


def test_launch_starts_an_ordinary_main_turn_with_the_wake_envelope(clock):
    clock.store[INTERVAL_STATE_KEY] = 1200
    now = T0 + FLOOR + 1
    assert clock.clock.tick(now) == "launched"
    [launch] = clock.launches
    assert launch["chat_id"] == 7
    meta = launch["metadata"]
    assert meta["initiator"] == "consciousness" and meta["usage_category"] == "consciousness"
    assert meta["wake_reason"] == "heartbeat" and meta["consciousness_autonomy"] == "act"
    assert meta["model_role"] == "consciousness" and meta["runtime_mode_cap"] == "light"
    assert "toggle_evolution" in meta["disabled_tools"] and "tell_task" not in meta["disabled_tools"]
    # В26=A: the wake tree's GRACEFUL ceiling = min(per-task cap 50, remaining 17.5); the
    # ledger fence keeps the per-task cap (a fence narrowed below one Main attempt's
    # reservation refused every wake of a nearly spent day before its first call).
    assert meta["root_cost_ceiling_usd"] == 17.5
    assert "root_limit_usd" not in meta
    # The clock admits a placeholder with the facts only its process holds; the carrier's first
    # step renders the wake from them (``consciousness_wake.prepare_wake``), never this thread.
    assert launch["admitted_text"] == "[Wake-up · heartbeat]"
    prepare = launch["prepare"]
    assert prepare["reason"] == "heartbeat" and prepare["level"] == "act" and prepare["interval"] == 1200
    assert prepare["window"]["limit_usd"] == 20.0 and prepare["window"]["settled_usd"] == 2.5
    assert prepare["counted_roots"] == [{"task_id": "root", "title": "Root", "consciousness_cause": "started"}]
    assert prepare["disabled_tools"] == meta["disabled_tools"] and prepare["repo_dir"] == str(clock.root / "repo")
    text = launch["text"]
    assert text.startswith("[Wake-up · heartbeat]") and "level=Acts" in text and "spent=2.50/20.00" in text
    assert "running=1/2" in text and "interval=1200" in text and "toggle_evolution" in text
    assert "no wake since this process started" in text
    assert launch["on_finished"] is None  # no callback: the wake's end is its pooled terminal
    assert set(clock.running) == {"wake0001"}
    snapshot = clock.clock.status_snapshot()
    assert snapshot["last_wake_task_id"] == "wake0001" and snapshot["last_wake_outcome"] == "running"
    started = [row for row in _events(clock.root) if row["type"] == "consciousness_wake_started"]
    assert started and started[0]["task_id"] == "wake0001" and started[0]["wake_reason"] == "heartbeat"


def test_launch_text_names_the_schedule_as_the_only_cause(clock):
    """#1603: no event pulls a wake, so every wake's cause is its scheduled time."""
    assert clock.clock.tick(T0 + FLOOR + 1) == "launched"
    assert "- wake cause: the scheduled wake time; events never move it" in clock.launches[-1]["text"]
    assert clock.launches[-1]["metadata"]["wake_reason"] == "heartbeat"


def test_launch_carries_the_main_lane_routing_facts_an_owner_turn_gets(clock):
    """P3c: a wake is an ordinary Main turn, so it is handed the host's routing manifest —
    without it every predecessor it names is refused as not addressable and it cannot
    continue prior work. The wake's own markers win the merge."""
    asked: list = []
    facts = {"main_routing_manifest": {"final_results": [{"task_id": "root-9"}]},
             "current_chat": {"chat_id": 7}, "initiator": "owner", "model_role": "main"}

    def routing_metadata_fn(chat_id):
        asked.append(chat_id)
        return dict(facts)

    alarm = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7,
                                    routing_metadata_fn=routing_metadata_fn, now=T0)
    assert alarm.tick(T0 + FLOOR + 1) == "launched"
    meta = clock.launches[-1]["metadata"]
    assert asked == [7]
    assert meta["main_routing_manifest"] == facts["main_routing_manifest"]
    assert meta["current_chat"] == {"chat_id": 7}
    assert meta["initiator"] == "consciousness" and meta["model_role"] == "consciousness"


def test_a_failing_routing_seam_is_disclosed_and_the_wake_still_starts(clock, caplog):
    """The facts are a courtesy, not a gate: the wake can always start fresh work."""

    def broken(_chat_id):
        raise RuntimeError("routing facts unreadable")

    alarm = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7,
                                    routing_metadata_fn=broken, now=T0)
    with caplog.at_level("WARNING"):
        assert alarm.tick(T0 + FLOOR + 1) == "launched"
    meta = clock.launches[-1]["metadata"]
    assert "main_routing_manifest" not in meta and meta["initiator"] == "consciousness"
    assert any("Main routing facts unavailable" in record.message for record in caplog.records)


def test_launch_without_a_routing_seam_keeps_the_bare_wake_envelope(clock):
    assert clock.clock.tick(T0 + FLOOR + 1) == "launched"
    assert "main_routing_manifest" not in clock.launches[-1]["metadata"]


def test_launch_cap_is_the_remaining_allowance_when_no_per_task_cap(clock, monkeypatch):
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "0")
    assert clock.clock.tick(T0 + FLOOR + 1) == "launched"
    assert clock.launches[0]["metadata"]["root_cost_ceiling_usd"] == 17.5
    # A disabled per-task cap leaves the allowance alone to set it.
    assert clock.launches[0]["metadata"]["root_cost_ceiling_source"] == "allowance"


@pytest.mark.parametrize("per_task,ceiling,source", [("50", 17.5, "allowance"), ("10", 10.0, "per_task_cap"),
                                                     ("17.5", 17.5, "both")])
def test_launch_stamps_which_limit_set_the_ceiling(clock, monkeypatch, per_task, ceiling, source):
    """Observed at launch, where min(per-task cap, allowance left) is taken (17.5 left of the day here): the views
    that explain a pause read it; no stop, fence or Resume does."""
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", per_task)
    assert clock.clock.tick(T0 + FLOOR + 1) == "launched"
    metadata = clock.launches[0]["metadata"]
    assert (metadata["root_cost_ceiling_usd"], metadata["root_cost_ceiling_source"]) == (ceiling, source)
    started = [row for row in _events(clock.root) if row["type"] == "consciousness_wake_started"]
    assert (started[0]["root_cost_ceiling_usd"], started[0]["root_cost_ceiling_source"]) == (ceiling, source)
    # The durable row keeps it as the queue's park and a restart's restore copy metadata (queue_snapshot).
    clock.finish("wake0001", True)
    from tests._store_shared import read_result
    assert read_result(clock.root, "wake0001")["metadata"]["root_cost_ceiling_source"] == source


def test_a_thin_allowance_still_wakes_and_only_known_exhaustion_skips(clock, monkeypatch):
    """Owner 2026-10-07: no extra $3 margin skip. $0.50 of known allowance left is a
    wake under a $0.50 producer ceiling; only known spend at the limit skips."""
    thin = dict(AVAILABLE, remaining_usd=0.5, settled_usd=19.5, accounted_usd=60.0)
    monkeypatch.setattr(clock_module, "allowance_window", lambda root, now=None, **_display_read: dict(thin))
    assert clock.clock.tick(T0 + FLOOR + 1) == "launched"
    assert clock.launches[-1]["metadata"]["root_cost_ceiling_usd"] == 0.5
    clock.finish("wake0001", True)
    clock.launches.clear()
    clock.clock._last_wake_task_id, clock.clock._next_wake_at = "", 0.0
    clock.live = None
    spent = dict(AVAILABLE, status="exhausted", remaining_usd=0.0, settled_usd=20.0, accounted_usd=20.0)
    monkeypatch.setattr(clock_module, "allowance_window", lambda root, now=None, **_display_read: dict(spent))
    assert clock.clock.tick(T0 + FLOOR + 1) == "skipped:allowance_exhausted"
    assert clock.launches == []
    assert clock.clock.next_wake_at == T0 + FLOOR + 1 + FLOOR  # the skip's own retry, nothing sooner


def test_launch_cap_rides_the_started_event_too(clock):
    """The durable row says what the wake was allowed to spend, under the same key
    the scope binds — a reader must not have to know a second name for the cap."""
    assert clock.clock.tick(T0 + FLOOR + 1) == "launched"
    started = [row for row in _events(clock.root) if row["type"] == "consciousness_wake_started"]
    assert started and started[0]["root_cost_ceiling_usd"] == 17.5
    assert "root_limit_usd" not in started[0]


def test_the_clock_has_no_early_wake_entry(clock):
    """#1603 (owner 12 Oct: no accelerated wake, only the scheduled time): no notify, no pending
    reason; every wake is the heartbeat, at the scheduled time."""
    assert not hasattr(clock.clock, "notify") and not hasattr(clock.clock, "pending_reason")
    assert not hasattr(clock.clock, "_pending_reason") and not hasattr(clock.clock, "_last_skip_at")
    assert clock.clock.tick(T0 + FLOOR + 1) == "launched"
    assert clock.launches[0]["metadata"]["wake_reason"] == "heartbeat"
    assert clock.launches[0]["text"].startswith("[Wake-up · heartbeat]")


def test_rejected_wake_is_typed_and_retried_on_its_schedule(clock):
    clock.receipt.update({"admitted": False, "task_id": "", "reason": "budget_exhausted"})
    now = T0 + FLOOR + 1
    assert clock.clock.tick(now) == "rejected:budget_exhausted"
    assert clock.clock.next_wake_at == now + DEFAULT  # the owner's budget is out: quietly, at the interval
    assert clock.clock.status_snapshot()["last_wake_outcome"] == "rejected:budget_exhausted"
    assert [row["reason"] for row in _events(clock.root) if row["type"] == "consciousness_wake_rejected"] == ["budget_exhausted"]
    clock.receipt["reason"], clock.clock._next_wake_at = "repo_writer_gate_closed", now
    assert clock.clock.tick(now) == "rejected:repo_writer_gate_closed"
    assert clock.clock.next_wake_at == now + FLOOR  # a transient door: the floor
    # The lane could not admit the turn and already reported the error in the chat: back
    # off like a failed wake, so a broken install is not told so every 15 minutes forever.
    clock.receipt["reason"], clock.clock._next_wake_at = "admission_failed", now
    assert clock.clock.tick(now) == "rejected:admission_failed"
    assert clock.clock.next_wake_at == now + DEFAULT * 2
    clock.clock._next_wake_at = now
    assert clock.clock.tick(now) == "rejected:admission_failed"
    assert clock.clock.next_wake_at == now + DEFAULT * 4


# --- finish -----------------------------------------------------------------------


def _launched(clock, now=T0 + FLOOR + 1):
    """Launch one wake; the way its end reaches the clock (the pooled terminal: ``clock.finish``)."""
    assert clock.clock.tick(now) == "launched"
    assert clock.launches[-1]["on_finished"] is None  # the lane calls nothing back
    return clock.finish


@pytest.mark.parametrize(("interval", "expected"), [(100, FLOOR), (10 ** 6, CEILING), (None, DEFAULT)])
def test_finish_schedules_the_chosen_interval_clamped(clock, monkeypatch, interval, expected):
    from ouroboros.task_results import write_task_result
    write_task_result(clock.root, "wake0001", "completed", result="Finished")
    finished = _launched(clock)
    monkeypatch.setattr(clock_module.time, "time", lambda: T0 + 5000)
    if interval is not None:
        clock.store[INTERVAL_STATE_KEY] = interval
    finished("wake0001", True)
    assert clock.clock.next_wake_at == T0 + 5000 + expected
    assert clock.store[LAST_WAKE_STATE_KEY] == T0 + 5000
    snapshot = clock.clock.status_snapshot()
    assert snapshot["last_wake_outcome"] == "done" and snapshot["last_error"] == ""
    assert snapshot["last_wake_at"].startswith("2027-")


def test_events_during_a_wake_never_shorten_the_next_interval(clock, monkeypatch):
    """The old finish-while-pending pull (``floor if pending_reason``) is gone: a task or
    project event during a wake waits for the chosen interval as an observed fact."""
    finished = _launched(clock)
    monkeypatch.setattr(clock_module.time, "time", lambda: T0 + 5000)
    finished("wake0001", True)
    assert clock.clock.next_wake_at == T0 + 5000 + DEFAULT


def test_runner_failure_backs_off_by_doubling_until_a_wake_succeeds(clock, monkeypatch):
    finished = _launched(clock)
    monkeypatch.setattr(clock_module.time, "time", lambda: T0 + 5000)
    finished("wake0001", False)
    assert clock.clock.next_wake_at == T0 + 5000 + DEFAULT * 2
    snapshot = clock.clock.status_snapshot()
    assert snapshot["last_wake_outcome"] == "failed" and "wake0001" in snapshot["last_error"]
    # Each failure belongs to a distinct admitted wake; duplicate callbacks do nothing.
    for index, ok, delay in [(2, False, DEFAULT * 4), (3, False, CEILING), (4, True, DEFAULT)]:
        clock.receipt["task_id"] = f"wake000{index}"
        finished = _launched(clock, now=clock.clock.next_wake_at)
        finished(clock.receipt["task_id"], ok)
        assert clock.clock.next_wake_at == T0 + 5000 + delay


# --- events never move the wake (#1603) --------------------------------------------


def test_a_root_finishing_and_a_project_digest_never_reach_the_clock(monkeypatch, tmp_path):
    """The producers of the old early wake are gone: a root's task_done and a project
    digest leave the alarm alone (the digest still touches its project), and the
    healed-orphan sweep still heals but has no callback left to report a count through."""
    import inspect

    import supervisor.events_task_done as events_task_done
    from ouroboros import server_maintenance
    from supervisor.events_project_routing import _handle_project_digest

    assert not hasattr(events_task_done, "_notify_consciousness_of_root_done")
    touched: list = []

    class _Clock:
        def __getattr__(self, name):
            raise AssertionError(f"the alarm clock was touched: {name}")

    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, consciousness=_Clock())
    monkeypatch.setattr("ouroboros.projects_registry.touch_project", lambda root, pid: touched.append(pid))
    _handle_project_digest({"project_id": "p9", "task_id": "t9"}, ctx)
    _handle_project_digest({"project_id": "p9", "task_id": "t10", "initiator": "consciousness"}, ctx)
    assert touched == ["p9", "p9"]
    monkeypatch.setattr("ouroboros.skill_review_runner.reconcile_stale_review_jobs", lambda root: None)
    healed: list = []
    monkeypatch.setattr("ouroboros.task_status.reconcile_orphaned_running_tasks",
                        lambda root, **kw: healed.append(root) or 2)
    monkeypatch.setattr("ouroboros.projects_registry.reconcile_projects", lambda root: None)
    monkeypatch.setattr(server_maintenance, "_resume_interrupted_project_deletions", lambda: None)
    server_maintenance._periodic_zombie_reconcile()
    assert healed == [server_maintenance.DATA_DIR]
    for name in ("_periodic_supervisor_maintenance", "_run_periodic_reconcile_sweep", "_periodic_zombie_reconcile"):
        assert "on_orphans_healed" not in inspect.signature(getattr(server_maintenance, name)).parameters, name


# --- owner controls ---------------------------------------------------------------


def test_start_and_stop_flip_the_flag_and_stop_arms_a_graceful_stop_of_a_live_wake(clock, monkeypatch):
    import threading

    assert clock.clock.start() == "Background consciousness is already enabled."
    assert clock.clock.stop() == "Consciousness off: no own wakes."  # no place, nothing live: nothing else to say
    assert clock.clock.enabled is False and clock.clock.tick(T0 + FLOOR + 1) == "disabled"
    assert clock.clock.stop() == "Background consciousness is already disabled."
    assert clock.clock.start().startswith("Background consciousness enabled; next wake-up at ")
    assert clock.clock.enabled is True
    _launched(clock)
    stopped: list = []
    done = threading.Event()

    def stop_pooled_wake(task_id):  # the lane's wake is a RUNNING row: pooled custody stops it
        stopped.append(task_id)
        done.set()

    monkeypatch.setattr("supervisor.consciousness_carriers.stop_pooled_wake", stop_pooled_wake)
    monkeypatch.setattr("supervisor.worker_chat_lane.stop_direct_chat_turn",
                        lambda *a, **k: pytest.fail("a carried wake has no in-process actor to stop"))
    message = clock.clock.stop()
    assert message == "Consciousness off: no own wakes. Stopping now: its wake-up."
    assert done.wait(5) and stopped == ["wake0001"]


@pytest.mark.parametrize("stored,said", [
    ((("active", "active"), ("silent", "silent"), ("paused", "paused")),
     "Consciousness off: no own wakes, and 2 places only hear until Start."),
    ((("silent", "silent"),), "Consciousness off: no own wakes, and 1 place only hears until Start."),
    ((("paused", "paused"),), "Consciousness off: no own wakes."),
    (None, "Consciousness off: no own wakes, and its places only hear until Start."),
])
def test_off_says_how_many_places_only_hear_until_start(clock, monkeypatch, stored, said):
    """Owner 06.11: he pressed Off to stop one place's replies, and nothing said that every place that is not Paused
    then only hears until Start. The Main row (and the Telegram /bg reply) says it; an unreadable place store changes
    only the words, never the switch."""
    from ouroboros import places
    if stored is None:
        monkeypatch.setattr(places, "load_places", lambda _root: (_ for _ in ()).throw(OSError("places unreadable")))
    for place_id, state in stored or ():
        places.connect_place(clock.root, place_id=place_id, connector="bridge", provider="chat",
                             conversation_id=place_id, label=place_id, state=state)
    assert clock.clock.stop() == said
    assert clock.clock.enabled is False, "an unreadable place store never holds the switch"
    assert clock.clock.stop() == "Background consciousness is already disabled."


# --- who stopped it (owner 11.11, 5A): the owner's and Panic's stops are the owner's to lift -----------------------


@pytest.fixture
def held(monkeypatch, tmp_path):
    """A clock over the REAL state store (locked writes, recovery, unavailability), consciousness on, nothing live."""
    from supervisor import queue, state, workers

    prior = state.DRIVE_ROOT
    state.init(tmp_path)
    state.save_state({"owner_chat_id": 7, "bg_consciousness_enabled": True})
    for module in (workers, queue):
        monkeypatch.setattr(module, "RUNNING", {})
        monkeypatch.setattr(module, "PENDING", [])
    get_direct_activity_registry().clear()

    def saved():
        projection = state.read_state(tmp_path).projection()
        return (state.control_value(projection, "bg_consciousness_enabled"),
                state.control_value(projection, clock_module.STOP_SOURCE_KEY))

    yield SimpleNamespace(clock=BackgroundConsciousness(tmp_path, tmp_path, lambda: 7, now=T0), root=tmp_path,
                          saved=saved, state=state)
    state.init(prior)


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])  # cyber_pro: no mode makes the owner's stop the mind's
def test_the_owners_stop_holds_against_the_minds_start_until_the_owners_own_start(held, monkeypatch, mode):
    from ouroboros.consciousness import AGENT_STOP, OWNER_STOP, held_stop_text

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    held.clock.stop(source=OWNER_STOP)
    assert held.saved() == ((True, False), (True, "owner"))
    assert held.clock.start(source=AGENT_STOP) == held_stop_text("owner") and held.clock.enabled is False
    restarted = BackgroundConsciousness(held.root, held.root, lambda: 7, now=T0)  # a restart reads the saved stop
    assert restarted.start(source=AGENT_STOP).startswith("Background consciousness stayed off: the owner turned it off")
    assert held.saved() == ((True, False), (True, "owner")) and restarted.enabled is False
    assert restarted.start().startswith("Background consciousness enabled; next wake-up at ")  # /bg start, any door
    assert held.saved() == ((True, True), (True, None)) and restarted.enabled is True


def test_the_minds_own_stop_stays_its_own_and_never_relabels_another(held):
    from ouroboros.consciousness import AGENT_STOP, OWNER_STOP, held_stop_text

    held.clock.stop(source=AGENT_STOP)
    assert held.saved() == ((True, False), (True, "agent_tool"))
    assert held.clock.start(source=AGENT_STOP).startswith("Background consciousness enabled")
    assert held.saved() == ((True, True), (True, None))
    held.clock.stop(source=OWNER_STOP)
    held.clock.stop(source=AGENT_STOP)  # a later stop of its own never makes the owner's stop the mind's
    assert held.saved() == ((True, False), (True, "owner"))
    assert held.clock.start(source=AGENT_STOP) == held_stop_text("owner")


def test_panic_and_an_unrecorded_stop_need_the_owners_start(held):
    from ouroboros import server_control
    from ouroboros.consciousness import AGENT_STOP, held_stop_text

    held.state.update_state(server_control._panic_controls, confirm=server_control.PANIC_CONTROL_KEYS)
    clock = BackgroundConsciousness(held.root, held.root, lambda: 7, now=T0)
    clock.stop(source=AGENT_STOP)
    assert held.saved() == ((True, False), (True, "panic"))
    assert clock.start(source=AGENT_STOP) == held_stop_text("panic") and "Panic turned it off" in held_stop_text("panic")
    # Migration: off with no recorded source (an earlier body's stop) is unknown until the owner starts it once.
    held.state.update_state(lambda st: st.pop(clock_module.STOP_SOURCE_KEY))
    assert clock.start(source=AGENT_STOP) == held_stop_text("")
    assert "who turned it off is not recorded" in held_stop_text("") and clock.enabled is False
    assert clock.start().startswith("Background consciousness enabled") and held.saved()[1] == (True, None)


def test_unknown_or_unwritable_state_refuses_the_mind_and_the_owners_stop_still_holds_here(held, monkeypatch):
    from ouroboros.consciousness import AGENT_STOP, OWNER_STOP, held_stop_text
    from supervisor.state import StateUnavailable, mark_unconfirmed

    def recovered(st):  # a backup recovery: the mind's own earlier stop is no longer proven
        st.update(bg_consciousness_enabled=False, bg_consciousness_stop_source="agent_tool")
        mark_unconfirmed(st, "bg_consciousness_enabled")
        mark_unconfirmed(st, clock_module.STOP_SOURCE_KEY)

    held.state.update_state(recovered)
    clock = BackgroundConsciousness(held.root, held.root, lambda: 7, now=T0)
    assert clock.start(source=AGENT_STOP) == held_stop_text("") and clock.enabled is False
    assert clock.start().startswith("Background consciousness enabled")  # the owner's start confirms both again
    real = held.state.update_state

    def unwritable(*_args, **_kwargs):
        raise StateUnavailable("lock_timeout", "update")

    monkeypatch.setattr(held.state, "update_state", unwritable)
    said = clock.stop(source=OWNER_STOP)
    assert said.startswith("Consciousness off") and clock.enabled is False
    assert " Not saved (runtime state unavailable, lock_timeout): the stop holds in this process only." in said
    assert clock.start(source=AGENT_STOP) == held_stop_text("owner")
    stayed = "Background consciousness stayed off: runtime state is unavailable (lock_timeout); nothing was started."
    assert clock.start() == stayed and clock.enabled is False, "a failed enable never turns the clock on first"
    monkeypatch.setattr(held.state, "update_state", real)
    assert held.saved()[0] == (True, True), "the stale saved choice: only this process holds the owner's stop"
    assert clock.start(source=AGENT_STOP) == held_stop_text("owner") and clock.enabled is False
    assert held.saved()[0] == (True, True), "a refused start writes nothing"
    clock.stop(source=AGENT_STOP)  # the mind's next stop saves the stop this process holds as the owner's
    assert held.saved() == ((True, False), (True, "owner"))
    assert clock.start(source=AGENT_STOP) == held_stop_text("owner")


def test_a_backup_only_failure_says_what_was_saved_and_the_next_tick_agrees(held, monkeypatch):
    """The primary copy holds the change and only its backup failed (``primary_written``): the reply says it was
    saved, never "nothing was started" or "this process only", and the clock, then and on its next tick, is what the
    saved state says; a restart reads the same."""
    from ouroboros.consciousness import AGENT_STOP, OWNER_STOP

    real = held.state.atomic_write_text

    def no_backup(path, content):
        if path == held.state.STATE_LAST_GOOD_PATH:
            raise OSError("backup unwritable")
        return real(path, content)

    note = " Saved, but the state's backup copy was not written (backup_write_failed)."
    held.clock.stop(source=AGENT_STOP)
    clock = BackgroundConsciousness(held.root, held.root, lambda: 7, now=T0)  # a fresh process over the mind's own stop
    monkeypatch.setattr(held.state, "atomic_write_text", no_backup)
    said = clock.start(source=AGENT_STOP)
    assert said.startswith("Background consciousness enabled; next wake-up at ") and said.endswith(note), said
    assert held.saved() == ((True, True), (True, None)) and clock.enabled is True
    clock.tick(now=T0)
    assert clock.enabled is True, "the next tick reads the enable the reply named"
    said = clock.stop(source=OWNER_STOP)
    assert note in said and "this process only" not in said, said
    assert held.saved() == ((True, False), (True, "owner"))
    clock.tick(now=T0)
    restarted = BackgroundConsciousness(held.root, held.root, lambda: 7, now=T0)
    assert clock.enabled is False and restarted.enabled is False, "the stop the reply named is the saved one"


def test_a_queued_tool_start_meets_the_stop_the_owner_placed_after_it(held):
    """The tool only queues; the clock decides when the supervisor runs the event, against the stop standing then."""
    from ouroboros.consciousness import OWNER_STOP, held_stop_text
    from ouroboros.tools import control
    from supervisor import events_runtime_controls

    tool = SimpleNamespace(task_id="t1", pending_events=[], drive_root=held.root, task_metadata={})
    assert control._toggle_consciousness(tool, "start") == "OK: consciousness 'start' requested."
    held.clock.stop(source=OWNER_STOP)  # /bg stop from the phone lands before the event is drained
    sent: list = []
    ctx = SimpleNamespace(consciousness=held.clock, load_state=held.state.load_state,
                          send_with_budget=lambda chat, text, **kw: sent.append((chat, text, kw)))
    events_runtime_controls._handle_toggle_consciousness(tool.pending_events[-1], ctx)
    assert sent == [(7, f"🧠 {held_stop_text('owner')}", {"role": "system", "system_type": "consciousness_notice"})]
    assert held.clock.enabled is False and held.saved() == ((True, False), (True, "owner"))


def test_status_snapshot_carries_the_alarm_facts(clock):
    snapshot = clock.clock.status_snapshot()
    assert set(snapshot) == {
        "enabled", "level", "next_wake_at", "last_wake_at", "last_wake_task_id",
        "last_wake_outcome", "last_error", "spent_24h_usd", "daily_usd", "allowance_resets_at",
        "tasks_running", "max_tasks", "live_wake_task_id", "unknown_unmetered", "integrity_degraded",
        "spend_by_cause", "counted_roots", "places_brief", "appointment_at", "live_wake_held",
        "live_wake_hold", "allowance_status", "allowance_open_rows", "allowance_room_after", "model_route",
    }
    assert snapshot["enabled"] is True and snapshot["level"] == "act"
    assert snapshot["unknown_unmetered"] == 0 and snapshot["integrity_degraded"] is False
    assert snapshot["next_wake_at"].startswith("2027-") and snapshot["last_wake_at"] == ""
    assert snapshot["spent_24h_usd"] == 2.5 and snapshot["daily_usd"] == 20.0
    assert snapshot["tasks_running"] == 1 and snapshot["max_tasks"] == 2 and snapshot["live_wake_task_id"] == ""
    # The current window's own verdict; no held wake, so no hold; an absent count stays absent, never 0.
    assert snapshot["allowance_status"] == "available" and snapshot["live_wake_hold"] is None
    assert snapshot["allowance_open_rows"] is None and snapshot["allowance_room_after"] == ""


def test_status_snapshot_names_the_model_its_next_wake_thinks_with(clock, monkeypatch):
    """``model_route``: the route the next wake takes, with its local flag; ``layer`` ``consciousness`` when its own
    slot decides it (``route_choice.model_role_slot_override``), else ``main``; a route that cannot be read is null,
    never a guess."""
    from ouroboros import route_choice

    monkeypatch.setenv("OUROBOROS_MODEL", "openai/gpt-6-astra")
    monkeypatch.setenv("USE_LOCAL_MAIN", "false")
    monkeypatch.setenv("OUROBOROS_MODEL_CONSCIOUSNESS", "anthropic/claude-opus-5-5")
    monkeypatch.setenv("USE_LOCAL_CONSCIOUSNESS", "")
    route = lambda: clock.clock.status_snapshot()["model_route"]  # noqa: E731
    assert route() == {"value": "anthropic/claude-opus-5-5", "layer": "consciousness", "local": False}
    monkeypatch.setenv("OUROBOROS_MODEL_CONSCIOUSNESS", "")
    assert route() == {"value": "openai/gpt-6-astra", "layer": "main", "local": False}
    # An empty slot's own box, checked under a cloud Main, decides the route: Main's model, run locally.
    monkeypatch.setenv("USE_LOCAL_CONSCIOUSNESS", "true")
    assert route() == {"value": "openai/gpt-6-astra", "layer": "consciousness", "local": True}
    monkeypatch.setenv("USE_LOCAL_MAIN", "true")  # equal flags: Main's own route, local
    assert route() == {"value": "openai/gpt-6-astra", "layer": "main", "local": True}
    monkeypatch.setenv("USE_LOCAL_CONSCIOUSNESS", "false")  # an unchecked box follows a local Main (owner, #3272)
    assert route() == {"value": "openai/gpt-6-astra", "layer": "main", "local": True}
    monkeypatch.setattr(route_choice, "following_route", lambda *_a, **_k: (_ for _ in ()).throw(OSError("unreadable")))
    assert route() is None


def test_the_status_view_may_ride_a_snapshot_and_a_wake_admission_never_does(clock, monkeypatch):
    """The status view shows money, so it may lag behind a contended ledger lock, and asks for the display's
    forecast; the wake ADMISSION spends it, so it reads exactly and computes none. One reader, two callers."""
    reads: list = []

    def window(root, now=None, **display_read):
        reads.append(dict(display_read))
        return dict(AVAILABLE)

    monkeypatch.setattr(clock_module, "allowance_window", window)
    clock.clock._allowance = (0.0, {}, False)  # nothing cached: the view must read
    clock.clock.status_snapshot()
    assert reads == [{"allow_stale": True, "forecast": True}]
    clock.clock._allowance_view(T0, fresh=True)
    assert reads[-1] == {"allow_stale": False}


def test_a_cached_view_is_reused_only_while_it_is_still_this_window(clock, monkeypatch):
    """Inside the TTL a display read reuses its own cached window, but not a wake's forecast-less one, not under a
    changed limit and not after the first counted row left (the forecast would be stale)."""
    from ouroboros.config import get_consciousness_daily_usd

    reads: list = []
    exhausted = dict(AVAILABLE, status="exhausted", settled_usd=20.0, remaining_usd=0.0, room_after=_iso(T0 + 600))

    def window(root, now=None, **display_read):  # as allowance_window: the limit it read, its rows' next exit
        reads.append(dict(display_read))
        first_exit = T0 + 300 if now < T0 + 300 else T0 + 900
        return dict(exhausted, limit_usd=float(get_consciousness_daily_usd()), window_changes_at=first_exit)

    monkeypatch.setattr(clock_module, "allowance_window", window)
    view = clock.clock._allowance_view
    clock.clock._allowance = (0.0, {}, False)
    view(T0)
    view(T0 + 10)
    assert len(reads) == 1, "a display read reuses its own window"
    view(T0 + 20, fresh=True)  # a wake's admission read replaces the cache without a forecast
    view(T0 + 25)
    assert reads[-1] == {"allow_stale": True, "forecast": True} and len(reads) == 3, "never served the admission's"
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "40")
    view(T0 + 30)
    assert len(reads) == 4, "a changed limit is another window"
    view(T0 + 301)
    assert len(reads) == 5, "once a counted row left, the cached forecast is stale"
    view(T0 + 302)
    assert len(reads) == 5


def _held_wake(clock, monkeypatch, *, metadata=None, pause=None, members=()):
    """My own wake parked in the pool as the queue parks it, its durable pause row beside it."""
    from ouroboros.consciousness_wake import wake_task_metadata
    from ouroboros.task_results import write_task_result
    from supervisor import queue

    metadata = metadata if metadata is not None else wake_task_metadata(
        "act", "heartbeat", root_cost_ceiling_usd=2.99, root_cost_ceiling_source="allowance")
    pause = pause if pause is not None else {
        "pause_id": "p1", "state": "paused", "reason": "budget", "rail": "graceful_ceiling", "scope": "root",
        "task_attempt": 1, "cost_ceiling": {"state": "active", "ceiling_usd": 2.99, "root_cap_usd": 50.0,
                                            "planning_margin_usd": None, "basis": "producer_allowance"}}
    write_task_result(clock.root, "wake-held", "scheduled", metadata=metadata, budget_pause=pause)
    rows = [{"id": "wake-held", "metadata": metadata, "root_task_id": "wake-held",
             "_budget_pause": {"rail": pause.get("rail"), "reason": pause.get("reason"), "root_task_id": "wake-held"}}]
    for member_id, member_pause in members:
        write_task_result(clock.root, member_id, "scheduled", root_task_id="wake-held", parent_task_id="wake-held",
                          budget_pause=member_pause)
        rows.append({"id": member_id, "root_task_id": "wake-held", "metadata": {},
                     "_budget_pause": {"rail": member_pause["rail"], "reason": "budget", "root_task_id": "wake-held"}})
    queue.PENDING[:] = rows
    clock.clock._own_wake = {"task_id": "wake-held", "wake_cause": "own_clock", "initiator": "consciousness",
                             "finished": False}


def test_a_wake_held_at_the_allowance_it_started_with_is_said_so(clock, monkeypatch):
    from ouroboros.consciousness_view import describe_consciousness_state

    _held_wake(clock, monkeypatch)
    snapshot = clock.clock.status_snapshot()
    assert snapshot["live_wake_held"] is True and snapshot["live_wake_task_id"] == "wake-held"
    assert snapshot["live_wake_hold"] == {"cause": "allowance", "ceiling_usd": 2.99}
    assert snapshot["allowance_status"] == "available", "the current window is its own, separate fact"
    described = describe_consciousness_state(snapshot, True)
    assert described["status"] == "wake_paused"
    # The fact is the headline; what follows from it is said beneath it.
    assert described["detail"] == "Wake-up wake-held stopped at the allowance it had left when it started ($2.99)."
    assert described["detail_note"] == ("Resume keeps that limit, so it pauses again while its known spend is at it, "
                                        "and a higher allowance does not raise it. Stop now on its card ends it and "
                                        "frees the clock for its next check.")


def test_a_wake_held_at_a_per_task_cap_raised_before_its_resume_is_not_the_allowance(clock, monkeypatch):
    """The cap bound it at launch; the owner raised the cap and pressed Resume, and it paused again at the same
    number: the saved cap is now above its ceiling, as an allowance hold's would be (the old inference's trap)."""
    from ouroboros.consciousness_view import describe_consciousness_state
    from ouroboros.consciousness_wake import wake_task_metadata

    pause = {"pause_id": "p2", "state": "paused", "reason": "budget", "rail": "graceful_ceiling", "scope": "root",
             "task_attempt": 1, "cost_ceiling": {"state": "active", "ceiling_usd": 10.0, "root_cap_usd": 80.0,
                                                 "planning_margin_usd": None, "basis": "producer_allowance"}}
    _held_wake(clock, monkeypatch, pause=pause, metadata=wake_task_metadata(
        "act", "heartbeat", root_cost_ceiling_usd=10.0, root_cost_ceiling_source="per_task_cap"))
    snapshot = clock.clock.status_snapshot()
    assert snapshot["live_wake_hold"]["cause"] == "per_task_cap"
    described = describe_consciousness_state(snapshot, True)
    assert described["detail"] == ("Wake-up wake-held is paused at the per-task cost cap it started with; its card "
                                   "shows why. The next wake-up follows it."), "never the allowance's words"
    assert "detail_note" not in described


@pytest.mark.parametrize("next_attempt", [T0 - 100, T0 + 6000])
def test_start_after_a_long_off_period_never_announces_a_past_wake(clock, monkeypatch, next_attempt):
    clock.clock.stop()
    clock.clock._next_wake_at = next_attempt  # the clock did not advance while disabled
    monkeypatch.setattr(clock_module.time, "time", lambda: T0 + 5000)
    message = clock.clock.start()
    expected = max(next_attempt, T0 + 5000)
    assert clock.clock.enabled and clock.clock.next_wake_at == expected
    assert "next wake-up at" in message
    assert clock.store[NEXT_WAKE_STATE_KEY] == expected
    assert clock.store[clock_module.APPOINTMENT_STATE_KEY] == expected


@pytest.mark.parametrize("pause_state", ["paused", "pausing"])
def test_finish_parked_wake_keeps_slot_and_runner_backoff(clock, monkeypatch, pause_state):
    from ouroboros.task_results import write_task_result
    write_task_result(clock.root, "wake0001", "scheduled",
                      budget_pause={"state": pause_state, "reason": "budget"})
    _launched(clock)
    clock.clock._backoff = 8
    clock.clock._wake_finished("wake0001", True, pooled=True)  # the pooled carrier's end as the clock reads it
    snapshot = clock.clock.status_snapshot()
    assert snapshot["last_wake_outcome"] == pause_state
    assert snapshot["tasks_running"] == 1
    assert clock.clock._backoff == 1


def test_finish_unreadable_wake_never_reports_done(clock, monkeypatch):
    import ouroboros.task_results as results
    _launched(clock)
    monkeypatch.setattr(results, "load_task_result", lambda *a, **k: (_ for _ in ()).throw(OSError("read failed")))
    clock.clock._wake_finished("wake0001", True, pooled=True)
    assert clock.clock.status_snapshot()["last_wake_outcome"] == "unknown"
    assert clock.clock._backoff == 1


def test_finish_before_dispatch_budget_pause_is_not_done(clock):
    from ouroboros.task_results import write_task_result
    write_task_result(clock.root, "wake0001", "scheduled", reason_code="budget_exhausted",
                      resource_limit={"replay_safe": True, "physical_calls": 0})
    _launched(clock)
    clock.clock._wake_finished("wake0001", True, pooled=True)
    assert clock.clock.status_snapshot()["last_wake_outcome"] == "paused"
    assert clock.clock._backoff == 1


def test_off_names_the_roots_it_leaves_running_on_my_humans_cited_words(clock, monkeypatch):
    """Owner 06.11 (3A): the switch stops autonomous work, not work started on a verified cite of my human's words,
    and its one Main row says which roots keep running and on which words; without such roots it adds nothing."""
    import supervisor.consciousness_carriers as carriers
    from supervisor import queue as q

    cited = {"source": "place_owner", "content": "Проверь экономику", "basis": "cited", "carried": True}
    pending = [{"id": "cited-root", "metadata": {"work_kind": "owner", "owner_corpus": [cited]}},
               {"id": "owner-door-root", "metadata": {"work_kind": "owner", "owner_corpus": [{"source": "initial_user", "content": "x"}]}},
               {"id": "own-root", "metadata": {"initiator": "consciousness", "work_kind": "consciousness", "owner_corpus": [cited]}}]
    monkeypatch.setattr(q, "PENDING", pending)
    span = {**cited, "content": "Проверь экономику и напиши мне", "cited": "Проверь экономику"}  # the whole message, the span named
    monkeypatch.setattr(q, "RUNNING", {"cited-running": {"task": {"id": "cited-running", "metadata": {
        "work_kind": "owner", "owner_corpus": [{"source": "owner_mailbox", "content": "y"}, span]}}}})
    assert carriers.cited_owner_roots() == [("cited-root", "Проверь экономику"), ("cited-running", "Проверь экономику")]
    schedules = [{"id": "sched-1", "enabled": True, "writer_kind": "owner", "task": {"metadata": {"owner_corpus": [span]}}},
                 {"id": "sched-off", "enabled": False, "writer_kind": "owner", "task": {"metadata": {"owner_corpus": [span]}}},
                 {"id": "sched-own", "enabled": True, "writer_kind": "consciousness", "task": {"metadata": {"owner_corpus": [span]}}}]
    monkeypatch.setattr(q, "list_scheduled_tasks", lambda _root=None: {"tasks": schedules})
    assert carriers.cited_owner_schedules(None) == [("sched-1", "Проверь экономику")]
    stopped = []
    monkeypatch.setattr(carriers, "stop_pooled_wake", stopped.append)
    assert clock.clock.stop() == ("Consciousness off: no own wakes. Stopping now: «own-root»."
                                  "\nKept running on your words (task cited-root): «Проверь экономику»"
                                  "\nKept running on your words (task cited-running): «Проверь экономику»"
                                  "\nKept firing on your words (schedule sched-1): «Проверь экономику»")
    assert set(stopped) <= {"own-root"}, "only the autonomous root is stopped; the cited ones keep running"
    monkeypatch.setattr(q, "PENDING", []); monkeypatch.setattr(q, "RUNNING", {}); schedules.clear()
    assert clock.clock.stop() == "Background consciousness is already disabled."


def test_a_wake_whose_child_paused_first_at_the_inherited_ceiling_is_held_by_its_allowance(clock, monkeypatch):
    """Child first: the member holding the wake's number paused at it and latched the tree, so the wake's own next
    send was refused at the root; the member's row proves the cause, read only for that refusal."""
    refused = {"pause_id": "p3", "state": "paused", "reason": "budget", "rail": "dispatch_refused", "scope": "root",
               "task_attempt": 1}
    child = {"pause_id": "c1", "state": "paused", "reason": "budget", "rail": "graceful_ceiling", "scope": "root",
             "task_attempt": 1, "cost_ceiling": {"state": "active", "ceiling_usd": 2.99, "root_cap_usd": 50.0,
                                                 "planning_margin_usd": None, "basis": "root_resolved_ceiling"}}
    _held_wake(clock, monkeypatch, pause=refused, members=[("wake-child", child)])
    assert clock.clock.status_snapshot()["live_wake_hold"] == {"cause": "allowance", "ceiling_usd": 2.99}
    # Without that member the same refusal is the per-task fence, never the allowance.
    _held_wake(clock, monkeypatch, pause=refused)
    assert clock.clock.status_snapshot()["live_wake_hold"]["cause"] == "per_task_fence"
