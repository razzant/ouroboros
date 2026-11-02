"""The alarm clock of Background Consciousness (redesign P2): ``tick`` decides WHEN an
ordinary Main turn starts on its own; the direct-activity census is the only liveness
truth; nothing but the schedule moves the next wake (#1603); a wake's own finish never
re-arms it. The lane that runs the turn is pinned in ``test_consciousness_wake_lane.py``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import consciousness as clock_module
from ouroboros import dialogue_store
from ouroboros.consciousness import (
    ARCHIVED_INBOX_REL,
    INTERVAL_STATE_KEY,
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
    """A constructed alarm clock over an in-memory runtime state and a stubbed lane."""
    from supervisor import state, workers

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
    launches: list = []
    receipt = {"admitted": True, "task_id": "wake0001", "reason": ""}

    def handle_wake_direct(chat_id, text, task_metadata, on_finished=None, bind_input=None):
        launch = {"chat_id": chat_id, "text": text, "metadata": task_metadata, "on_finished": on_finished}
        if receipt.get("admitted") and bind_input is not None:
            # The real lane binds after registration, before the turn's thread starts.
            launch["task"] = {"id": receipt["task_id"], "text": text, "metadata": dict(task_metadata)}
            bind_input(launch["task"])
        launches.append(launch)
        return dict(receipt)

    monkeypatch.setattr(workers, "handle_wake_direct", handle_wake_direct)
    monkeypatch.setattr(clock_module, "allowance_window", lambda root, now=None, **_display_read: dict(AVAILABLE))
    monkeypatch.setattr("supervisor.queue.live_consciousness_roots", lambda: [
        {"task_id": "root", "title": "Root", "consciousness_cause": "started"}])
    get_direct_activity_registry().clear()
    clock = BackgroundConsciousness(tmp_path, tmp_path / "repo", lambda: store.get("owner_chat_id"), now=T0)
    yield SimpleNamespace(clock=clock, store=store, launches=launches, receipt=receipt, root=tmp_path)
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
    from supervisor import workers

    due = clock.clock.next_wake_at
    assert clock.clock.tick(now=due + 1) == "launched"
    _register("wake0001", initiator="consciousness")
    monkeypatch.setattr(workers, "direct_chat_turn", lambda task_id: {
        "metadata": {"initiator": "consciousness", "wake_cause": "own_clock"},
    } if task_id == "wake0001" else None)
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
    assert f"Research room [{place.place_id}]: active; open turns: 3; chat_history(chat_id=123)" in text
    assert "note (owner): Watch releases." in text


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
    assert "- task settled completed" in clock.launches[-1]["text"]
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
    assert '"first"' in clock.launches[-1]["text"]  # the refused window is observed again, whole
    accepted = clock.store[OBSERVATION_STATE_KEY]
    # The boundary is the chain coordinate the next row takes (the file size, when the chain was a file).
    assert accepted["task_id"] == "wake0001" and accepted["upper"] == dialogue_store.stream_end(clock.root, "chat")
    # The observation was bound to the registered wake before its turn could run.
    bound = clock.launches[-1]["task"]["metadata"]["wake_observation"]
    assert bound["composition"] == {"owner_message": 1} and bound["source"]["sha256"]
    # A line appended while that wake runs — stamped even before its capture — reaches the next wake.
    _chat_row(root=clock.root, ts=_iso(T0 + 10), direction="in", chat_id=1, source="web", text="second")
    clock.clock._wake_finished("wake0001", True)
    clock.clock._next_wake_at = T0 + 2 * FLOOR
    get_direct_activity_registry().clear()
    assert clock.clock.tick(T0 + 2 * FLOOR) == "launched"
    text = clock.launches[-1]["text"]
    assert '"second"' in text and '"first"' not in text


def test_a_refused_wake_consumes_no_transition_and_the_admitted_one_persists_observed_identities(clock):
    """The accepted boundary carries the task-result transition state beside the chat position:
    a child that settles with no chat row is reported by the next ADMITTED wake, not lost to a
    refused one. The complete inventory lives in the exact wake source."""
    from ouroboros.consciousness import OBSERVATION_STATE_KEY

    (clock.root / "task_results").mkdir()
    row = {"task_id": "kid", "status": "running", "ts": _iso(T0 - 100), "updated_at": _iso(T0 - 100),
           "_schema_version": 1, "metadata": {}, "description": "child work"}
    seed_result(clock.root, "kid", row)
    assert clock.clock.tick(T0 + FLOOR + 1) == "launched"
    accepted = clock.store[OBSERVATION_STATE_KEY]["transitions"]
    from ouroboros.artifacts import read_actor_source_bytes
    source = accepted['source_ref']
    header = json.loads(read_actor_source_bytes(clock.root, source['task_id'], source).splitlines()[0])
    inventory = header['boundary']['transitions']
    assert inventory["inventory"] == {"kid": []} and inventory["scan_at"] == _iso(T0 + FLOOR + 1)
    seed_result(clock.root, "kid", {**row, "status": "failed", "updated_at": _iso(T0 + FLOOR + 50)})  # the orphan sweep
    clock.clock._wake_finished("wake0001", True)
    get_direct_activity_registry().clear()
    clock.receipt.update({"admitted": False, "task_id": "", "reason": "budget_exhausted"})
    clock.clock._next_wake_at = T0 + 2 * FLOOR
    assert clock.clock.tick(T0 + 2 * FLOOR) == "rejected:budget_exhausted"
    assert clock.store[OBSERVATION_STATE_KEY]["transitions"] == accepted
    clock.receipt.update({"admitted": True, "task_id": "wake0002", "reason": ""})
    clock.clock._next_wake_at = T0 + 3 * FLOOR
    assert clock.clock.tick(T0 + 3 * FLOOR) == "launched"
    assert "- task kid failed, completion time not recorded" in clock.launches[-1]["text"]
    source = clock.store[OBSERVATION_STATE_KEY]["transitions"]['source_ref']
    header = json.loads(read_actor_source_bytes(clock.root, source['task_id'], source).splitlines()[0])
    assert set(header['boundary']['transitions']["inventory"]) == {"kid"}


@pytest.mark.parametrize('failure', ['write', 'readback', 'corrupt_previous', 'missing_previous'])
def test_inventory_source_failure_never_advances_accepted_boundary(clock, monkeypatch, failure):
    from ouroboros import artifacts, chat_chain
    from ouroboros.consciousness import OBSERVATION_STATE_KEY

    _chat_row(root=clock.root, ts=_iso(T0 + 1), direction='in', chat_id=1, source='web', text='first')
    assert clock.clock.tick(T0 + FLOOR + 1) == 'launched'
    accepted = clock.store[OBSERVATION_STATE_KEY]
    source = accepted['transitions']['source_ref']
    path = artifacts.task_artifact_dir_path(clock.root, source['task_id']) / source['path']
    original = path.read_bytes()
    _chat_row(root=clock.root, ts=_iso(T0 - 100), direction='in', chat_id=1, source='web', text='late old row')
    clock.clock._wake_finished('wake0001', True)
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
        assert clock.store[OBSERVATION_STATE_KEY] == accepted
        if failure in {'corrupt_previous', 'missing_previous'}:
            assert 'unreadable_transition_source' in clock.launches[-1]['text']
    path.write_bytes(original)
    clock.launches[-1]['on_finished']('wake0002', True)
    clock.receipt['task_id'] = 'wake0003'
    clock.clock._next_wake_at = T0 + 3 * FLOOR
    assert clock.clock.tick(T0 + 3 * FLOOR) == 'launched'
    assert 'late old row' in clock.launches[-1]['text']
    assert clock.store[OBSERVATION_STATE_KEY] != accepted


@pytest.mark.parametrize('version', [1, 2])
def test_inline_inventory_migrates_to_the_source_bound_transitions(clock, version):
    """A v1 inline inventory migrates to the v2 source-bound record (a late-review wake of a
    closed task left with the acceptance machine; the migration stays)."""
    from ouroboros import consciousness_wake as wake
    from ouroboros.consciousness import OBSERVATION_STATE_KEY
    from tests.test_consciousness_wake import _write

    _write(clock.root, 'old', status='completed', ts=_iso(T0 - 86400))
    prior = wake.observe_wake(clock.root, boundary=None, since=T0 - 10, now=T0).boundary
    if version == 1:
        prior['transitions'] = {'version': 1, 'inventory': {}, 'observed': [], 'scan_at': _iso(T0)}
    clock.store[OBSERVATION_STATE_KEY] = prior
    assert clock.clock.tick(T0 + FLOOR + 1) == 'launched'
    assert ('task old completed' in clock.launches[-1]['text']) is (version == 1)
    accepted = clock.store[OBSERVATION_STATE_KEY]['transitions']
    assert set(accepted) == {'version', 'source_ref'} and accepted['version'] == 2


@pytest.mark.parametrize('failure', ['write', 'readback', 'bootstrap_write'])
@pytest.mark.parametrize('restart', [False, True])
def test_first_failed_source_preserves_intervening_events(clock, monkeypatch, failure, restart):
    from ouroboros import artifacts, chat_chain
    from ouroboros.consciousness import OBSERVATION_STATE_KEY
    from tests.test_consciousness_wake import _write

    _chat_row(root=clock.root, ts=_iso(T0 - 1), direction='in', chat_id=1, source='web', text='before bootstrap')
    with monkeypatch.context() as fault:
        if failure in {'write', 'bootstrap_write'}:
            fault.setattr(chat_chain, 'retain_memory_source', lambda *a, **kw: (_ for _ in ()).throw(OSError('write failed')))
        else:
            fault.setattr(artifacts, 'read_actor_source_bytes', lambda *a, **kw: b'bad readback')
        if failure == 'bootstrap_write':
            from supervisor import state
            from ouroboros.consciousness import OBSERVATION_BOOTSTRAP_STATE_KEY

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
    clock.launches[-1]['on_finished']('wake0001', not restart)
    alarm = BackgroundConsciousness(clock.root, clock.root / 'repo', lambda: 7, now=finish_at + 1) if restart else clock.clock
    clock.clock._wake_finished('wake0001', True)
    clock.receipt['task_id'] = 'wake0002'
    assert alarm.tick(alarm.next_wake_at) == 'launched'
    assert 'during first wake' in clock.launches[-1]['text']
    assert 'task intervening completed' in clock.launches[-1]['text']
    assert 'before bootstrap' not in clock.launches[-1]['text']
    assert clock.store[OBSERVATION_STATE_KEY]
    clock.launches[-1]['on_finished']('wake0002', True)
    clock.receipt['task_id'] = 'wake0003'
    assert alarm.tick(alarm.next_wake_at + DEFAULT) == 'launched'
    assert 'during first wake' not in clock.launches[-1]['text']
    assert 'task intervening completed' not in clock.launches[-1]['text']


def test_full_inventory_moves_out_of_hot_state_without_forgetting_closed_tasks(clock):
    from ouroboros import artifacts
    from ouroboros.consciousness import OBSERVATION_STATE_KEY
    from ouroboros.task_results import write_task_result

    for index in range(160):
        write_task_result(clock.root, f'closed-{index}', 'completed', ts=_iso(T0 - 86400))
    assert clock.clock.tick(T0 + FLOOR + 1) == 'launched'
    accepted = clock.store[OBSERVATION_STATE_KEY]
    assert len(json.dumps(accepted)) < 1800
    source = accepted['transitions']['source_ref']
    header = json.loads(artifacts.read_actor_source_bytes(clock.root, source['task_id'], source).splitlines()[0])
    assert len(header['boundary']['transitions']['inventory']) == 160
    write_task_result(clock.root, 'new-old', 'completed', ts=_iso(T0 - 86400))
    clock.clock._wake_finished('wake0001', True)
    clock.receipt['task_id'] = 'wake0002'
    clock.clock._next_wake_at = T0 + 2 * FLOOR
    assert clock.clock.tick(T0 + 2 * FLOOR) == 'launched'
    assert 'new-old completed' in clock.launches[-1]['text']
    assert 'closed-0 completed' not in clock.launches[-1]['text']
    clock.launches[-1]['on_finished']('wake0002', True)
    clock.receipt['task_id'] = 'wake0003'
    clock.clock._next_wake_at = T0 + 3 * FLOOR
    assert clock.clock.tick(T0 + 3 * FLOOR) == 'launched'
    assert 'new-old completed' not in clock.launches[-1]['text']


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


def test_a_live_own_wake_defers_the_wake(clock):
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
    text = launch["text"]
    assert text.startswith("[Wake-up · heartbeat]") and "level=Acts" in text and "spent=2.50/20.00" in text
    assert "running=1/2" in text and "interval=1200" in text and "toggle_evolution" in text
    assert "no wake since this process started" in text
    assert launch["on_finished"] == clock.clock._wake_finished
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


def test_a_thin_allowance_still_wakes_and_only_known_exhaustion_skips(clock, monkeypatch):
    """Owner 2026-10-07: no extra $3 margin skip. $0.50 of known allowance left is a
    wake under a $0.50 producer ceiling; only known spend at the limit skips."""
    thin = dict(AVAILABLE, remaining_usd=0.5, settled_usd=19.5, accounted_usd=60.0)
    monkeypatch.setattr(clock_module, "allowance_window", lambda root, now=None, **_display_read: dict(thin))
    assert clock.clock.tick(T0 + FLOOR + 1) == "launched"
    assert clock.launches[-1]["metadata"]["root_cost_ceiling_usd"] == 0.5
    clock.clock._wake_finished("wake0001", True)
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
    assert clock.clock.tick(now) == "launched"
    return clock.launches[-1]["on_finished"]


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
    assert clock.clock.stop() == "Background consciousness disabled."
    assert clock.clock.enabled is False and clock.clock.tick(T0 + FLOOR + 1) == "disabled"
    assert clock.clock.stop() == "Background consciousness is already disabled."
    assert clock.clock.start().startswith("Background consciousness enabled; next wake-up at ")
    assert clock.clock.enabled is True
    _launched(clock)
    _register("wake0001", initiator="consciousness")
    stopped: list = []
    done = threading.Event()

    def stop_direct_chat_turn(task_id, turn, **_kw):
        stopped.append((task_id, turn["id"]))
        done.set()
        return "ended"

    monkeypatch.setattr("supervisor.worker_chat_lane.stop_direct_chat_turn", stop_direct_chat_turn)
    message = clock.clock.stop()
    assert message == "Background consciousness disabled; wake-up wake0001 ends at its next step."
    assert done.wait(5) and stopped == [("wake0001", "wake0001")]


def test_status_snapshot_carries_the_alarm_facts(clock):
    snapshot = clock.clock.status_snapshot()
    assert set(snapshot) == {
        "enabled", "level", "next_wake_at", "last_wake_at", "last_wake_task_id",
        "last_wake_outcome", "last_error", "spent_24h_usd", "daily_usd", "allowance_resets_at",
        "tasks_running", "max_tasks", "live_wake_task_id", "unknown_unmetered", "integrity_degraded",
        "spend_by_cause", "counted_roots", "places_brief", "appointment_at", "live_wake_held",
    }
    assert snapshot["enabled"] is True and snapshot["level"] == "act"
    assert snapshot["unknown_unmetered"] == 0 and snapshot["integrity_degraded"] is False
    assert snapshot["next_wake_at"].startswith("2027-") and snapshot["last_wake_at"] == ""
    assert snapshot["spent_24h_usd"] == 2.5 and snapshot["daily_usd"] == 20.0
    assert snapshot["tasks_running"] == 1 and snapshot["max_tasks"] == 2 and snapshot["live_wake_task_id"] == ""


def test_the_status_view_may_ride_a_snapshot_and_a_wake_admission_never_does(clock, monkeypatch):
    """The status view shows money, so it may lag behind a contended ledger lock; the
    wake ADMISSION spends it, so it reads exactly. One reader, two callers."""
    reads: list = []

    def window(root, now=None, **display_read):
        reads.append(dict(display_read))
        return dict(AVAILABLE)

    monkeypatch.setattr(clock_module, "allowance_window", window)
    clock.clock._allowance = (0.0, {})  # nothing cached: the view must read
    clock.clock.status_snapshot()
    assert reads == [{"allow_stale": True}]
    clock.clock._allowance_view(T0, fresh=True)
    assert reads[-1] == {"allow_stale": False}


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
    clock.clock._wake_finished("wake0001", True)
    snapshot = clock.clock.status_snapshot()
    assert snapshot["last_wake_outcome"] == pause_state
    assert snapshot["tasks_running"] == 1
    assert clock.clock._backoff == 1


def test_finish_unreadable_wake_never_reports_done(clock, monkeypatch):
    import ouroboros.task_results as results
    _launched(clock)
    monkeypatch.setattr(results, "load_task_result", lambda *a, **k: (_ for _ in ()).throw(OSError("read failed")))
    clock.clock._wake_finished("wake0001", True)
    assert clock.clock.status_snapshot()["last_wake_outcome"] == "unknown"
    assert clock.clock._backoff == 1


def test_finish_before_dispatch_budget_pause_is_not_done(clock):
    from ouroboros.task_results import write_task_result
    write_task_result(clock.root, "wake0001", "scheduled", reason_code="budget_exhausted",
                      resource_limit={"replay_safe": True, "physical_calls": 0})
    _launched(clock)
    clock.clock._wake_finished("wake0001", True)
    assert clock.clock.status_snapshot()["last_wake_outcome"] == "paused"
    assert clock.clock._backoff == 1
