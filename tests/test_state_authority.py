"""#1307: an unreadable, corrupt or lost state.json is never a fresh install.

Isolated roots only (``state.init(tmp_path)``); every test restores the module paths.
"""
from __future__ import annotations

import errno
import json
import os
import pathlib
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros.upgrade_notices import BORN_SETTLED_NOTICE_KEYS, JUDGED_BY_FACTS_NOTICE_KEYS
from supervisor import state, state_initialization
from tests._source_scan import parse
from tests._store_shared import remove_store_tree


@pytest.fixture
def root(tmp_path, monkeypatch):
    from supervisor import queue
    from supervisor import evolution_lifecycle as lifecycle

    prior_state, prior_queue = state.DRIVE_ROOT, queue.DRIVE_ROOT
    state.init(tmp_path)
    queue.init(tmp_path)
    monkeypatch.setattr(state, "check_openrouter_ground_truth", lambda: None)
    monkeypatch.setitem(lifecycle._STOP_LATCH, "stopped", False)
    yield tmp_path
    state.init(prior_state)
    queue.init(prior_queue)


def _write(path: pathlib.Path, payload) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    path.write_bytes(raw)
    return raw


def _prior(**extra):
    return {"created_at": "2026-07-30T00:00:00+00:00", "session_id": "old-session", "owner_id": 7,
            "owner_chat_id": 7, "owner_external_id": 99, "owner_external_chat_id": 99,
            "evolution_mode_enabled": True, "evolution_owner_stopped": False, "evolution_cycle": 6,
            "bg_consciousness_enabled": True, "usage_ledger_high_water_seq": [1, 50], **extra}


def _born_settled(values):
    return all(values.get(key) for key in BORN_SETTLED_NOTICE_KEYS) and not any(
        key in values for key in JUDGED_BY_FACTS_NOTICE_KEYS)


def _files(root):
    out = {}
    for p in (root / "state").glob("*"):
        if p.is_file():
            with open(p, "rb") as handle:  # not Path.read_bytes: tests patch it
                out[p.name] = handle.read()
    return out


@pytest.mark.parametrize("failure", ["enfile", "eacces"])
def test_an_unreadable_primary_is_never_written_and_its_controls_are_unknown(root, monkeypatch, failure):
    _write(state.STATE_PATH, _prior())
    _write(state.STATE_LAST_GOOD_PATH, _prior(evolution_owner_stopped=False))
    before = _files(root)
    real = pathlib.Path.read_bytes

    def flaky(self):
        if self == state.STATE_PATH:
            raise OSError(errno.ENFILE if failure == "enfile" else errno.EACCES, "denied")
        return real(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", flaky)
    read = state.read_state()
    assert (read.quality, read.source) == ("recovered_transient", "backup")
    shown = state.load_state()
    assert shown["evolution_cycle"] == 6  # display values survive
    assert state.control_value(shown, "evolution_owner_stopped") == (False, None)
    assert not state.control_is(shown, "evolution_mode_enabled", True)
    with pytest.raises(state.StateUnavailable) as refused:
        state.update_state(lambda live: live.__setitem__("session_id", "new"))
    assert refused.value.reason == "primary_unreadable"
    assert state.init_state().quality == "recovered_transient"
    assert _files(root) == before  # nothing minted, nothing overwritten, no witness


def test_a_corrupt_primary_recovers_from_backup_without_laundering_authority(root):
    corrupt = _write(state.STATE_PATH, b'{"evolution_owner_stopped": tru')
    _write(state.STATE_LAST_GOOD_PATH, _prior())
    # Routine bookkeeping triggers the durable recovery write…
    state.update_state(lambda live: live.__setitem__("last_owner_message_at", "now"))
    [copy] = list((root / "state").glob("state.corrupt-*.json"))
    assert copy.read_bytes() == corrupt
    saved = json.loads(state.STATE_PATH.read_bytes())
    assert saved["created_at"] == "2026-07-30T00:00:00+00:00" and saved["session_id"] == "old-session"
    assert "usage_ledger_high_water_seq" not in saved  # #1144: money re-derives from the ledger
    assert set(saved[state.RECOVERY_KEY]["unconfirmed"]) == set(state.CONTROL_KEYS)
    # …and more bookkeeping, then a cold boot, still cannot clear it.
    state.update_state(lambda live: live.pop(state.RECOVERY_KEY, None))
    state.update_state(lambda live: live.__setitem__("evolution_owner_stopped", False))
    assert state.init_state().quality == "recovered"
    cold = state.load_state()
    assert state.control_value(cold, "evolution_owner_stopped") == (False, None)
    assert state.control_value(cold, "owner_external_id") == (False, None)
    # A real decision confirms exactly the control it sets.
    state.update_state(lambda live: live.__setitem__("bg_consciousness_enabled", False),
                       confirm=("bg_consciousness_enabled",))
    after = state.load_state()
    assert state.control_value(after, "bg_consciousness_enabled") == (True, False)
    assert state.control_value(after, "evolution_mode_enabled") == (False, None)


def test_evolution_after_a_recovery_waits_for_the_owner_then_resumes(root, monkeypatch):
    from supervisor import evolution_lifecycle as lifecycle
    from supervisor import queue
    from supervisor.events_runtime_controls import owner_evolution_start

    state.save_state(_prior())
    _write(state.STATE_PATH, b'{"evolution_mode_enabled": tru')
    assert lifecycle.start_evolution_campaign("improve", source="owner")
    state.update_state(lambda live: live.__setitem__("last_owner_message_at", "now"))  # the recovery write
    pending = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.0)
    monkeypatch.setattr(queue, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "evolution_block_reason", lambda: "")
    queue.enqueue_evolution_task_if_needed()
    assert pending == []  # a recovered "enabled" is not the owner's current decision
    assert owner_evolution_start("resume", source="owner_chat") == {}
    queue.enqueue_evolution_task_if_needed()
    # The witnessed write-once binding survives; Start confirms evolution only.
    assert [task["chat_id"] for task in pending] == [7]
    assert state.control_value(state.load_state(), "owner_chat_id") == (True, 7)


def test_writers_never_proceed_unlocked(root, monkeypatch):
    _write(state.STATE_PATH, _prior())
    before = state.STATE_PATH.read_bytes()
    monkeypatch.setattr(state, "acquire_file_lock", lambda *a, **k: None)
    with pytest.raises(state.StateUnavailable) as refused:
        state.update_state(lambda live: live.__setitem__("session_id", "x"))
    assert refused.value.reason == "lock_timeout"
    assert state.update_budget_from_usage({}) is False
    assert state.STATE_PATH.read_bytes() == before


def test_reads_never_create_state(root):
    assert state.load_state()[state.STATE_READ_KEY]["quality"] == "uninitialized"
    with pytest.raises(state.StateUnavailable):
        state.update_state(lambda live: None)
    assert not state.STATE_PATH.exists() and not state.STATE_LAST_GOOD_PATH.exists()
    assert not state_initialization.witness_path(root).exists()


@pytest.mark.parametrize("scaffold", ["empty_dirs", "settings_first", "benchmark_seed"])
def test_a_positive_fresh_bootstrap_creates_one_identity(root, scaffold):
    for rel in ("state", "logs", "memory", "task_results", "locks"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    if scaffold in {"settings_first", "benchmark_seed"}:
        (root / "settings.json").write_text("{}", encoding="utf-8")
    if scaffold == "benchmark_seed":
        (root / state.ISOLATED_BENCHMARK_SENTINEL).write_text("1", encoding="utf-8")
        (root / "logs" / "chat.jsonl").write_text("", encoding="utf-8")  # empty is not history
    read = state.init_state()
    assert read.quality == "current"
    status, witness = state_initialization.read_witness(root)
    assert status == "ok" and witness["phase"] == "complete" and witness["origin"] == "first_boot"
    assert read.values["initialization_id"] == witness["initialization_id"]
    assert state.control_value(state.load_state(), "owner_id") == (True, None)
    assert _born_settled(read.values)  # a fresh install owes no news of an earlier release


@pytest.mark.parametrize("state_written", [False, True], ids=["intent_only", "state_written"])
def test_an_interrupted_initialization_finishes_the_same_identity(root, state_written):
    from ouroboros import projects_registry

    (root / "state").mkdir(exist_ok=True)
    (root / "state" / "state.initialized.json").write_text(json.dumps(
        {"initialization_id": "first-id", "phase": "pending", "origin": "first_boot"}), encoding="utf-8")
    if state_written:
        _write(state.STATE_PATH, state.ensure_state_defaults({"initialization_id": "first-id"}))
    assert state.init_state().values["initialization_id"] == "first-id"
    assert state_initialization.read_witness(root)[1]["phase"] == "complete"
    assert json.loads((root / "state/projects.json").read_bytes())["projects"] == []
    assert projects_registry.create_project(root, "first")["created"] is True


@pytest.mark.parametrize("history", ["witness_complete", "bindings", "evidence_unknown"])
def test_lost_state_on_an_initialized_or_historical_root_is_never_reminted(root, monkeypatch, history):
    (root / "state").mkdir(exist_ok=True)
    if history == "witness_complete":
        (root / "state" / "state.initialized.json").write_text(json.dumps(
            {"initialization_id": "i", "phase": "complete", "completed_at": "t"}), encoding="utf-8")
    elif history == "witness_corrupt":
        (root / "state" / "state.initialized.json").write_text("{", encoding="utf-8")
    elif history == "queue_snapshot":
        (root / "state" / "queue_snapshot.json").write_text("{}", encoding="utf-8")
    elif history == "task_result":
        _write(root / "task_results" / "t1.json", {"task_id": "t1"})
    elif history == "chat":
        _write(root / "logs" / "chat.jsonl", b'{"text": "hi"}\n')
    elif history == "campaign":
        _write(root / "state" / "evolution_campaign.json", {"id": "c"})
    elif history == "bindings":
        _write(root / "state" / "project_task_bindings.json", {})
    else:
        real = os.scandir

        def blind(path):
            raise PermissionError(errno.EACCES, "denied") if str(path).endswith("task_results") else real(path)

        (root / "task_results").mkdir()
        monkeypatch.setattr(os, "scandir", blind)
    read = state.init_state()
    assert read.quality in {"uninitialized", "unavailable"} and read.values == {}
    assert not state.STATE_PATH.exists() and not state.STATE_LAST_GOOD_PATH.exists()
    assert not state.control_is(state.load_state(), "evolution_owner_stopped", False)


def test_a_legacy_readable_state_is_adopted_with_its_values(root):
    raw = _write(state.STATE_PATH, _prior())
    _write(state.STATE_LAST_GOOD_PATH, raw)
    (root / "state" / "queue_snapshot.json").write_text("{}", encoding="utf-8")
    read = state.init_state()
    assert read.quality == "current" and read.values["session_id"] == "old-session"
    assert read.values["created_at"] == "2026-07-30T00:00:00+00:00" and read.values["evolution_cycle"] == 6
    assert not any(key in read.values for key in BORN_SETTLED_NOTICE_KEYS)  # an upgraded install keeps them owed
    status, witness = state_initialization.read_witness(root)
    assert (status, witness["phase"], witness["origin"]) == ("ok", "complete", "legacy_adopted")


def test_legacy_reset_intent_with_a_registry_is_not_a_cleanup_receipt(root):
    from ouroboros import projects_registry
    projects_registry.create_project(root, "previous")
    _write(root / "task_results" / "leftover.json", {"task_id": "leftover"})
    primary = projects_registry._registry_path(root).read_bytes()
    state_initialization.mark_pending(root, origin="owner_reset")
    assert state.init_state().quality == "unavailable"
    assert state_initialization.reset_status(root)["status"] == "interrupted"
    with pytest.raises(FileNotFoundError):
        projects_registry.create_project(root, "fresh")
    assert projects_registry._registry_path(root).read_bytes() == primary


@pytest.mark.parametrize("published", [False, True], ids=["before_publish", "after_publish"])
def test_a_completed_cleanup_repeat_finishes_only_publication(root, monkeypatch, published):
    from ouroboros import project_registry_recovery as registry
    from ouroboros import projects_registry

    assert state.init_state().quality == "current"
    projects_registry.create_project(root, "example", name="Example")
    _write(root / "memory/identity.md", b"invented previous memory")
    note = root / "projects/example/knowledge/note.md"
    _write(note, b"invented project note")
    identity = state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    assert _hand_off(root) == "reset_requested"

    original = registry.initialize_new_root

    def interrupted_publication(*args, **kwargs):
        if published:
            original(*args, **kwargs)
        raise OSError("test: publication interrupted")

    with monkeypatch.context() as scope:
        scope.setattr(registry, "initialize_new_root", interrupted_publication)
        state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    status, witness = state_initialization.read_witness(root)
    assert status == "resetting"
    receipt_data = witness[state_initialization._CLEANUP_DONE_KEY]
    assert receipt_data["initialization_id"] == identity and "memory" in receipt_data["deleted"]
    assert not (root / "memory/identity.md").exists()
    assert registry.registry_path(root).exists() == published
    if published:
        prior = registry.registry_path(root).stat()

    monkeypatch.setattr(state_initialization, "_clear_runtime_data", lambda *_: pytest.fail("repeated cleanup"))
    assert state_initialization.request_runtime_reset(root, launch_protection=lambda: "local") == identity
    assert _hand_off(root) == "reset_requested"
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    status, saved = state_initialization.read_witness(root)
    assert status == "ok"
    assert saved["phase"] == "pending"
    assert saved["reset_receipt"] == receipt_data and state_initialization._CLEANUP_DONE_KEY not in saved
    assert registry.read_bytes(registry.registry_path(root).read_bytes())["projects"] == []
    if published:
        assert registry.registry_path(root).stat().st_ino == prior.st_ino
    assert (root / receipt_data["projects_set_aside"] / "example/knowledge/note.md").read_bytes() == (
        b"invented project note")


def _hand_off(root):
    from ouroboros import store_switch
    stopped = {"worker_exits": {"unconfirmed": []}, "live_children": [],
               "owned_stop": {"state": "completed", "unconfirmed": []}}
    return store_switch.switch_at_restart(root, None, stopped, restart_drained=True)["status"]


def _reset_request(root, *, confirmed=True, **app_state):
    """The versioned POST the Settings page sends; ``confirmed=False`` is the bare POST of a client written for the
    earlier answer. The started restart coordinator is named separately by each positive test: a request without
    one answers the missing-door refusal, never a hand-off."""
    async def body():
        return json.dumps({"confirm": f"{state_initialization.RESET_CONFIRMATION}:v1"}).encode() if confirmed else b""

    async def parsed():
        return json.loads(await body())  # an empty body is no JSON, as Starlette answers it

    app_state.setdefault("reset_launch_protection", lambda _root: "local")
    return SimpleNamespace(method="POST", url=SimpleNamespace(path="/api/reset/v1"), json=parsed, body=body,
                           app=SimpleNamespace(state=SimpleNamespace(drive_root=root, **app_state)))


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_reading_reset_status_leaves_the_profile_alone(root, monkeypatch, method):
    import asyncio
    from ouroboros.gateway import control

    monkeypatch.setattr(control, "_acquire_repo_mutation_lock", lambda: pytest.fail("a status read needs no writer"))
    request = SimpleNamespace(method=method, app=SimpleNamespace(state=SimpleNamespace(drive_root=root)))
    before = _files(root)
    assert json.loads(asyncio.run(control.api_reset(request)).body) == {"status": "none"}
    assert _files(root) == before


@pytest.mark.parametrize("ingress_closed,unconfirmed,error", [
    (True, [1], "worker_exits_unconfirmed"), (False, [], "shutdown_unconfirmed"),
])
def test_reset_requires_a_restart_door_and_confirmed_writers(root, monkeypatch, ingress_closed, unconfirmed, error):
    import asyncio
    from ouroboros import store_switch
    from ouroboros.gateway import control

    state.init_state()
    before = _files(root)
    assert asyncio.run(control.api_reset(_reset_request(root))).status_code == 409
    assert _files(root) == before
    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    assert state_initialization.reset_status(root)["status"] == "requested"
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")  # there is no hand-off yet
    stop = {"worker_exits": {"unconfirmed": unconfirmed}, "live_children": [],
            "owned_stop": {"state": "completed", "unconfirmed": []}}
    outcome = store_switch.switch_at_restart(root, None, stop, restart_drained=ingress_closed)
    assert outcome["status"] == "reset_interrupted"
    monkeypatch.setattr(state_initialization, "_clear_runtime_data", lambda *_: pytest.fail("replayed Reset"))
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    assert state_initialization.reset_status(root) == {
        "status": "interrupted", "error": error,
        "reset_id": state_initialization.read_witness(root)[1]["initialization_id"]}
    assert state.STATE_PATH.read_bytes() == before["state.json"]


@pytest.mark.parametrize("confirmed,status", [(False, 400), (True, 200)])
def test_only_a_post_that_names_the_confirmation_requests_reset(root, monkeypatch, confirmed, status):
    import asyncio
    from ouroboros.gateway import control

    # A page written for the earlier answer posts no body and would read ``requested`` as a failure while the
    # Reset went on: its request is refused before anything is recorded, locked or restarted.
    state.init_state()
    before, witness, restarts = _files(root), state_initialization.read_witness(root), []
    monkeypatch.setattr(state_initialization, "_requested_reset", None)  # this test hands nothing off
    monkeypatch.setattr(control, "_acquire_repo_mutation_lock",
                        lambda: (None, None) if confirmed else pytest.fail("a refused request takes no lock"))
    request = _reset_request(root, confirmed=confirmed, request_restart=lambda *, owner: restarts.append(owner))
    if confirmed:  # the started restart coordinator, named separately, is what accepts the hand-off
        request.app.state.restart_consumer = SimpleNamespace(is_alive=lambda: True)
    response = asyncio.run(control.api_reset(request))
    answer = json.loads(response.body)
    assert response.status_code == status
    if confirmed:
        # The acknowledgement names a restart request and no deletion, and carries this Reset's own identity;
        # the profile is whole until the hand-off.
        assert answer == {"status": "requested", "restarting": True,
                          "reset_id": state_initialization.read_witness(root)[1]["initialization_id"]}
        assert restarts == [True]
        assert state_initialization.reset_status(root) == {"status": "requested",
                                                          "reset_id": answer["reset_id"]}
        after = _files(root)  # only the witness changed: a request removes nothing
        assert all(after[name] == raw for name, raw in before.items() if name != "state.initialized.json")
    else:
        assert answer["reason"] == "reset_confirmation_required" and "Reload" in answer["error"]
        assert restarts == [] and state_initialization.read_witness(root) == witness
        assert state_initialization.reset_status(root) == {"status": "none"} and _files(root) == before


@pytest.mark.parametrize("phase,stop", [
    ("request", "latch"), ("handoff", "latch"), ("handoff", "input"),
    ("admission", "input"), ("admission", "pending_sweep"), ("admission", "unknown"),
])
def test_reset_leaves_panic_and_uncertain_stop_facts_to_their_owner(root, monkeypatch, phase, stop):
    from ouroboros import server_control
    from ouroboros.process_scope import panic_sweep_pending_path

    state.init_state()
    before, witness = state.STATE_PATH.read_bytes(), state_initialization.read_witness(root)
    if phase != "request":
        state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    if phase == "admission":
        assert _hand_off(root) == "reset_requested"
    marker = root / "state/panic_stop.flag"
    sweep = panic_sweep_pending_path(root)
    if stop == "latch":
        monkeypatch.setattr(server_control, "panic_termination_requested", lambda: True)
    elif stop == "input":
        _write(marker, b"panic")  # a Panic that arrived after the owner confirmed
    elif stop == "pending_sweep":
        _write(sweep, {"synthetic": True})
    else:
        def unreadable(_root):
            raise PermissionError("test double cannot read the stop input")

        monkeypatch.setattr(state_initialization, "panic_input", unreadable)
    monkeypatch.setattr(state_initialization, "_clear_runtime_data", lambda *_: pytest.fail("Reset proceeded"))
    if phase == "request":
        with pytest.raises(OSError, match="Panic"):
            state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
        assert state_initialization.read_witness(root) == witness
    elif phase == "handoff":
        assert _hand_off(root) == "reset_interrupted"
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")  # the start after that Panic clears nothing
    assert state.STATE_PATH.read_bytes() == before
    if stop == "input":
        # That Reset had not begun clearing, so the Panic withdraws it: the replaced witness is back, the input is
        # untouched, and this is the ordinary start after Panic, whose supervisor acknowledges it in these controls.
        assert state_initialization.read_witness(root) == witness and marker.read_bytes() == b"panic"
        assert state_initialization.reset_status(root) == {"status": "none"}
        assert state.init_state().quality == "current"
    elif phase != "request":
        outcome = state_initialization.reset_status(root)
        assert outcome["status"] == "interrupted" and "Panic" in outcome["error"]
    if stop == "pending_sweep":
        assert json.loads(sweep.read_bytes()) == {"synthetic": True}


def test_a_panic_after_clearing_began_leaves_the_reset_for_its_explicit_repeat(root, monkeypatch):
    import shutil
    from ouroboros.installation_stop import installation_panic, panic_input

    # Once something was removed there is no whole profile to return to. The Reset stays interrupted, a start
    # clears nothing more, and the owner's explicit repeat finishes around the input nobody but Panic's owner touches.
    assert state.init_state().quality == "current"
    _write(root / "memory/identity.md", b"an invented earlier memory")
    archive, remove = root / "archive", shutil.rmtree
    _write(archive / "chat_20260101.jsonl", b'{"chat_id":1,"text":"old profile"}\n')

    def withheld(path, *args, **kwargs):
        if path == archive:
            raise PermissionError("archive in use")
        return remove(path, *args, **kwargs)

    with monkeypatch.context() as cut:
        cut.setattr(shutil, "rmtree", withheld)
        state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
        assert _hand_off(root) == "reset_requested"
        state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    marker = root / "state/panic_stop.flag"
    _write(marker, b"panic")  # a Panic after the first pass had removed the memory
    known = panic_input(root)
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    held = state_initialization.reset_status(root)
    assert held["status"] == "interrupted" and "memory" in held["deleted"]
    assert archive.is_dir() and state.STATE_PATH.is_file() and not (root / "memory").exists()
    done = _reset(root)
    assert done["status"] == "complete" and {"memory", "archive"} <= set(done["deleted"])
    assert marker.read_bytes() == b"panic" and panic_input(root) == known and installation_panic(root)


def test_a_panic_during_clearing_starts_no_further_removal(root, monkeypatch):
    import shutil

    assert state.init_state().quality == "current"
    _write(root / "memory/identity.md", b"an invented earlier memory")
    archive, marker, remove = root / "archive", root / "state/panic_stop.flag", shutil.rmtree
    _write(archive / "chat_20260101.jsonl", b'{"chat_id":1,"text":"old profile"}\n')

    def panic_lands(path, *args, **kwargs):
        remove(path, *args, **kwargs)
        if path == root / "memory":
            _write(marker, b"panic")  # a double of Panic's own input, written while this removal ran

    with monkeypatch.context() as cut:
        cut.setattr(shutil, "rmtree", panic_lands)
        state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
        assert _hand_off(root) == "reset_requested"
        state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    held = state_initialization.reset_status(root)
    assert held["status"] == "interrupted" and held["error"] == "Panic interrupted Reset"
    assert "memory" in held["deleted"] and archive.is_dir() and state.STATE_PATH.is_file()
    # The start after that Panic neither returns a profile that lost its memory nor removes anything more.
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    assert state_initialization.reset_status(root) == held and archive.is_dir() and marker.read_bytes() == b"panic"


def test_a_pass_cut_inside_its_first_removal_is_never_read_as_one_that_changed_nothing(root, monkeypatch):
    assert state.init_state().quality == "current"
    before = state_initialization.read_witness(root)
    _write(root / "logs/events.jsonl", b"{}\n")
    clear = state_initialization._clear_reset_directory

    def cut_inside(path, **kwargs):
        if path == root / "logs":
            (path / "events.jsonl").unlink()  # part of the step happened; no completed step names it
            raise OSError("the invented cut inside the first removal")
        return clear(path, **kwargs)

    monkeypatch.setattr(state_initialization, "_clear_reset_directory", cut_inside)
    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    assert _hand_off(root) == "reset_requested"
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    cut = state_initialization.reset_status(root)
    assert cut == {"status": "interrupted", "error": "the invented cut inside the first removal",
                   "reset_id": state_initialization.read_witness(root)[1]["initialization_id"]}
    # The boundary was saved before that removal began, so a Panic afterwards withdraws nothing.
    assert state_initialization.read_witness(root)[1]["reset_removing"] is True
    _write(root / "state/panic_stop.flag", b"panic")
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    assert state_initialization.read_witness(root) != before and state_initialization.reset_status(root) == cut


def test_a_pass_cut_between_a_removal_and_its_receipt_stays_reconcilable(root, monkeypatch):
    class _generation_cut(BaseException):
        """The process ends mid-pass: no write of the interrupted Reset follows the removal."""

    assert state.init_state().quality == "current"
    _write(root / "memory/identity.md", b"an invented earlier memory")
    archived = _write(root / "archive/chat_20260101.jsonl", b'{"chat_id":1,"text":"old profile"}\n')
    remove = state_initialization._remove_tree_for_reset

    def severed(path):
        result = remove(path)
        if path == root / "memory":
            raise _generation_cut()  # between the completed removal and any later receipt
        return result

    with monkeypatch.context() as cut:
        cut.setattr(state_initialization, "_remove_tree_for_reset", severed)
        state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
        assert _hand_off(root) == "reset_requested"
        with pytest.raises(_generation_cut):
            state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    # The interrupted status still knows the cut pass's removal: memory was recorded pending before
    # its effect and is proven gone, while the removal the pass never reached is claimed nowhere.
    interrupted = state_initialization.reset_status(root)
    assert interrupted["status"] == "interrupted" and "memory" in interrupted["deleted"]
    assert "archive" not in interrupted["deleted"] and not (root / "memory").exists()
    assert (root / "archive/chat_20260101.jsonl").read_bytes() == archived
    # The explicit repeat finishes the same Reset, and its one receipt covers the cut pass's removal too.
    done = _reset(root)
    assert done["status"] == "complete" and {"memory", "archive"} <= set(done["deleted"])
    assert not (root / "archive").exists() and not state.STATE_PATH.exists()


def test_a_panic_input_the_owner_confirmed_over_stays_for_the_new_profile(root):
    from ouroboros.installation_stop import installation_panic, panic_input

    # A Panic input is already there when the owner confirms (no supervisor has acknowledged it yet). Reset
    # finishes around it and never acknowledges, rewrites or removes it: the first supervisor start of the new
    # profile does, exactly as after any Panic.
    state.init_state()
    marker = root / "state/panic_stop.flag"
    _write(marker, b"panic")
    known = panic_input(root)
    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    assert state_initialization.read_witness(root)[1]["panic_input"] == known
    assert _hand_off(root) == "reset_requested"
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    done = state_initialization.read_witness(root)[1]
    assert state_initialization.reset_status(root)["status"] == "complete" and "panic_input" not in done
    assert not state.STATE_PATH.exists()
    assert marker.read_bytes() == b"panic" and panic_input(root) == known and installation_panic(root)


def _reset(root):
    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    assert _hand_off(root) == "reset_requested"
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    return state_initialization.reset_status(root)


def _tree(path):
    """Every entry below ``path``: its bytes, a link's target, or ``None`` for a directory."""
    return {entry.relative_to(path).as_posix():
            os.readlink(entry) if entry.is_symlink() else entry.read_bytes() if entry.is_file() else None
            for entry in sorted(path.rglob("*"))}


def _set_aside(root, outcome):
    """The folder a finished or interrupted Reset names, checked against both spellings it reports."""
    name = outcome["projects_set_aside"]
    assert name.startswith("projects/.before-reset-") and outcome["projects_set_aside_path"] == str(root / name)
    return root / name


@pytest.mark.parametrize("phase,target,can_switch", [
    ("complete", b"an older body without the Reset hand-off", True),
    ("complete", b"def accept_runtime_reset(", True),
    ("resetting", b"an older body without the Reset hand-off", False),
    ("resetting", b"def accept_runtime_reset(", True),
])
def test_only_an_unfinished_reset_restricts_a_version_switch(root, monkeypatch, phase, target, can_switch):
    from ouroboros import projects_registry as projects, store_switch

    assert state.init_state().quality == "current"
    projects.create_project(root, "previous")
    kept = _write(root / "projects/previous/knowledge/note.md", b"invented earlier fact")
    _write(root / "projects/previous/.project.json", {"id": "previous", "name": "Previous"})
    if phase == "complete":
        container = _set_aside(root, _reset(root))
        assert (container / "previous/knowledge/note.md").read_bytes() == kept
        assert [entry.name for entry in (root / "projects").iterdir()] == [container.name]
    else:
        state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    monkeypatch.setattr(store_switch, "target_file_bytes", lambda *_: target)
    monkeypatch.setattr(store_switch, "switch_needs", lambda *_: {})
    monkeypatch.setattr(store_switch, "lacks_wake_chain_reader", lambda *_: False)
    monkeypatch.setattr(store_switch, "_hand_off_stores", lambda *_: "")
    owners = []

    def handoff_owner():
        owners.append("called")
        return True, "accepted"

    ok, message, deferred = store_switch.switch_or_reset(
        root, None, target_sha="synthetic", target_ref="test", door="ui_rollback", reason="test",
        reset=handoff_owner)
    assert (ok, deferred, owners) == (can_switch, False, ["called"] if can_switch else [])
    assert can_switch or message == "The target version cannot preserve the unfinished Reset."
    if phase == "complete":
        # Neither this body nor the store readers of a rollback target see a room in the dot folder.
        assert projects.reconcile_projects(root) == 0 and projects.list_projects(root) == []
        assert (container / "previous/knowledge/note.md").read_bytes() == kept
    else:
        assert (root / "projects/previous/knowledge/note.md").read_bytes() == kept


@pytest.mark.parametrize("keep_diagnostics,active_logs", [(False, False), (True, False), (True, True)],
                         ids=["no_diagnostics", "closed_logs", "windows_shared_files"])
def test_reset_sets_project_notes_aside_and_reopens_project_creation(
        root, monkeypatch, keep_diagnostics, active_logs, tmp_path_factory):
    import asyncio
    from contextlib import ExitStack
    from ouroboros import config, context, dialogue_store, project_facts, project_registry_recovery
    from ouroboros import projects_registry as registry, server_entrypoint
    from ouroboros.gateway import control, projects
    from ouroboros.installation_stop import PANIC_RECEIPT
    from ouroboros.server_boot import reconcile_projects
    from supervisor import message_bus

    monkeypatch.setattr(config, "DATA_DIR", root)  # the journal and workpad readers address the data root
    prior = state.init_state()
    assert prior.quality == "current"
    work = tmp_path_factory.mktemp("working-folder")  # the owner's own folder, outside the data root
    _write(work / "src/main.py", b"print('invented working file')\n")
    registry.create_project(root, "retained", name="Retained", working_dir=str(work))
    registry.create_project(root, "fileless")
    bare = project_facts.resolve_project_id({"id": "unbound", "workspace_root": str(tmp_path_factory.mktemp("bare"))})
    assert bare.startswith("proj_")  # a workspace task without a registered Project keeps a shelf by folder hash
    note = root / "projects/retained/knowledge/note.md"
    contents = _write(note, b"synthetic retained fact")
    for pid in ("retained", bare):
        _write(root / "projects" / pid / "knowledge/earlier.md", f"# EARLIER-KNOWLEDGE-{pid}\n\nAn invented fact.\n".encode())  # the index shows a note's title
        _write(root / "projects" / pid / "workpad.md", f"EARLIER-WORKPAD-{pid}\n".encode())
        _write(root / "projects" / pid / "workpad_history.jsonl", b'{"revision":"invented"}\n')
        _write(root / "projects" / pid / "journal.jsonl",
               json.dumps({"ts": "2026-01-01T00:00:00+00:00", "kind": "done", "text": f"EARLIER-JOURNAL-{pid}",
                           "task_id": "invented"}).encode() + b"\n")
    _write(root / "projects/.DS_Store", b"a dot entry stays where it is")
    env = SimpleNamespace(drive_path=lambda rel: root / rel, repo_path=lambda rel: root / rel, drive_root=root)

    def notes_shown(pid):  # what a mind in this Project would be given
        return "\n".join(str(section) for section in context.build_knowledge_sections(env, project_id=pid))

    for pid in ("retained", bare):
        shown = notes_shown(pid)
        assert all(f"EARLIER-{part}-{pid}" in shown for part in ("KNOWLEDGE", "WORKPAD", "JOURNAL")), shown
    shelves, working = _tree(root / "projects"), _tree(work)
    del shelves[".DS_Store"]
    dialogue_store.append(root, "chat", {"chat_id": 1, "direction": "in", "text": "prior store row"})
    _write(root / "archive/chat_20260101.jsonl", b'{"chat_id":1,"text":"archived prior row"}\n')
    _write(root / "logs/chat.jsonl", b'{"chat_id":1,"text":"prior legacy row"}\n')
    admission = root / "state" / server_entrypoint.SERVER_LOCK_NAME
    update_lock = root / "locks/managed_update.lock"
    # A historical receipt alone is not a pending Panic or permission to clear this profile.
    state.update_state(lambda values: values.update({PANIC_RECEIPT: {"inode": 17}}))
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    assert state.STATE_PATH.is_file() and note.read_bytes() == contents
    _write(update_lock, b"held")
    diagnostics = {root / "logs" / name for name in ("agent_stdout.log", "launcher.log", "server.log")}
    if keep_diagnostics:
        for path in diagnostics:
            _write(path, b"launcher diagnostics")
    unlink = pathlib.Path.unlink

    def unlink_with_windows_sharing(path, *args, **kwargs):
        if active_logs and os.name != "nt" and path in diagnostics:
            raise PermissionError(errno.EACCES, "file is in use", str(path))
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", unlink_with_windows_sharing)
    # The current owner's emergency control remains available until the generation exits.
    old_bridge = SimpleNamespace(panic=SimpleNamespace(invalidate_owner=lambda: pytest.fail("owner revoked before exit")))
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: old_bridge)
    monkeypatch.setattr(control, "_acquire_repo_mutation_lock", lambda: (None, None))
    restart_requests = []
    request = _reset_request(root, repo_dir=root, request_restart=lambda *, owner: restart_requests.append(owner),
                             restart_consumer=SimpleNamespace(is_alive=lambda: True))
    _write(admission, b"synthetic admission inode")
    lock_stat = admission.stat()
    with ExitStack() as handles:
        if active_logs:  # actual open append handles on Windows; simulate its sharing rule elsewhere
            for path in diagnostics:
                handles.enter_context(path.open("ab"))
        response = asyncio.run(control.api_reset(request))
        assert response.status_code == 200 and json.loads(response.body) == {
            "status": "requested", "restarting": True,
            "reset_id": state_initialization.read_witness(root)[1]["initialization_id"]}
        assert restart_requests == [True] and state.STATE_PATH.is_file()
        assert _hand_off(root) == "reset_requested"
        # Ordinary final writes of the departing process, after its worker census.
        _write(root / "state/queue_snapshot.json", {"pending": ["previous-task"]})
        _write(root / "logs/progress.jsonl", b'{"task_id":"previous-task"}\n')
        # This root has ordinary exclusive create, but cannot publish via link/exclusive rename.
        def unsupported_publication(_temporary, _path):
            raise OSError(errno.ENOTSUP, "exclusive publication is unavailable")

        monkeypatch.setattr(project_registry_recovery, "_publish_new_file", unsupported_publication)
        # Enter through the successor's production admission on this disposable inode.
        # No server, listener, child process or older implementation is started.
        monkeypatch.setattr(server_entrypoint, "_admission", {})
        handles.callback(server_entrypoint.release_admission_for_tests)
        server_entrypoint.admit_data_root(root, wait_sec=0, reset_launch=SimpleNamespace(at_boot=lambda _root: "local"))
        assert os.path.samestat(lock_stat, admission.stat())
    request.method = "GET"
    result = json.loads(asyncio.run(control.api_reset(request)).body)
    assert result["status"] == "complete"
    # Every earlier shelf is the same bytes under one dot folder; nothing of it stays addressable by id.
    container = _set_aside(root, result)
    assert _tree(container) == shelves and _tree(work) == working
    assert sorted(entry.name for entry in (root / "projects").iterdir()) == [".DS_Store", container.name]
    assert state_initialization.SET_ASIDE_KEY not in state_initialization.read_witness(root)[1]
    assert not (root / "state/queue_snapshot.json").exists() and not (root / "logs/progress.jsonl").exists()
    # All three diagnostic names may be open in a surviving launcher. Keep them on every OS,
    # even when this fixture's handles are closed; with no survivors, report removal of logs/.
    assert ("logs" in result["deleted"]) is not keep_diagnostics
    assert result["retained_logs"] == (sorted(path.name for path in diagnostics) if keep_diagnostics else [])
    assert "archive" in result["deleted"] and "state" not in result["deleted"]
    assert all(path.is_file() is keep_diagnostics for path in diagnostics)
    if keep_diagnostics:
        assert all(path.read_bytes() == b"launcher diagnostics" for path in diagnostics)
    assert not (root / "logs/chat.jsonl").exists() and not (root / "archive").exists()
    assert not dialogue_store.path(root).exists()
    assert update_lock.read_bytes() == b"held"
    pending = state_initialization.read_witness(root)[1]
    assert pending["phase"] == "pending" and pending["origin"] == "owner_reset"

    reconcile_projects(root)  # providerless restart: supervisor init_state has not run
    assert registry._registry_path(root).is_file()
    assert registry.reserved_project_chat_ids(root, strict=True) == set()
    assert registry.reconcile_projects(root) == 0
    assert not registry.list_projects(root)
    async def create_body():
        return {"name": "New"}

    request.json = create_body
    monkeypatch.setattr(projects, "_broadcast_projects_changed", lambda *_: None)
    first = json.loads(asyncio.run(projects.api_projects_create(request)).body)["project"]
    again = json.loads(asyncio.run(projects.api_projects_create(request)).body)["project"]
    assert first["created"] is True and again["created"] is False
    assert first["routing_incarnation"] == again["routing_incarnation"]
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")  # another admission preserves this new room
    read = state.init_state()
    assert read.quality == "current" and _born_settled(read.values)  # a Reset is a fresh start: nothing to hear
    assert state_initialization.read_witness(root)[1]["origin"] == "owner_reset"
    assert read.values["initialization_id"] == pending["initialization_id"] != prior.values["initialization_id"]
    assert registry.reconcile_projects(root) == 0
    assert {row["id"] for row in registry.list_projects(root)} == {"new"}
    # The same name and the same working folder derive the same ids, and both start with no notes.
    again = registry.create_project(root, project_facts.project_id_from_display_name("Retained"), name="Retained",
                                    working_dir=str(work))
    assert again["created"] and again["id"] == "retained"
    assert project_facts.resolve_project_id({"id": "later", "workspace_root": str(work)}) == "retained"
    for pid in ("retained", bare):
        shown = notes_shown(pid)
        assert "EARLIER-" not in shown, shown
        for name in ("knowledge/earlier.md", "workpad.md", "workpad_history.jsonl", "journal.jsonl"):
            assert not (root / "projects" / pid / name).exists()
    assert registry.reconcile_projects(root) == 0
    assert {row["id"] for row in registry.list_projects(root)} == {"new", "retained"}
    assert _tree(container) == shelves and _tree(work) == working
    assert (container / "retained/knowledge/note.md").read_bytes() == contents and not note.exists()


@pytest.mark.parametrize("kind", ["move", "remove"])
def test_a_registered_working_folder_inside_the_reset_footprint_stops_the_reset_before_any_effect(
        root, monkeypatch, kind):
    """The pre-effect guard: a Project's registered folder that this Reset would move aside or remove
    whole stops the Reset while the profile is still whole."""
    from ouroboros import projects_registry as registry

    assert state.init_state().quality == "current"
    inside = (root / "projects/inside") if kind == "move" else (root / "uploads/project-folder")
    kept = _write(inside / "src/main.py", b"print('invented working file')\n")
    memory = _write(root / "memory/identity.md", b"an invented earlier memory")
    registry.create_project(root, "inside", name="Inside", working_dir=str(inside))
    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    assert _hand_off(root) == "reset_requested"
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    status = state_initialization.reset_status(root)
    assert status["status"] == "interrupted" and "working folder" in status["error"]
    assert str(inside) in status["error"]  # the folder this Reset would have moved or removed is named
    # Nothing moved and nothing was removed: the profile is whole and the interruption is kept for the owner.
    assert (inside / "src/main.py").read_bytes() == kept
    assert (root / "memory/identity.md").read_bytes() == memory
    assert state_initialization.read_witness(root)[1]["phase"] == "resetting"


def test_a_registered_working_folder_outside_the_reset_footprint_stays_where_it_is(root, monkeypatch,
                                                                                    tmp_path_factory):
    """The same guard lets the ordinary Reset through: a Project's folder outside the footprint is
    neither moved nor removed."""
    from ouroboros import projects_registry as registry

    assert state.init_state().quality == "current"
    work = tmp_path_factory.mktemp("working-folder")
    kept = _write(work / "src/main.py", b"print('invented working file')\n")
    _write(root / "memory/identity.md", b"an invented earlier memory")  # this Reset's footprint is not empty
    registry.create_project(root, "retained", name="Retained", working_dir=str(work))
    done = _reset(root)
    assert done["status"] == "complete"
    assert (work / "src/main.py").read_bytes() == kept


def test_a_missing_prior_registry_never_reads_as_no_projects(root, monkeypatch):
    """An absent registry is not proof that this profile never had a Project: while the registry's own
    witness still stands, the Reset stops before it removes anything."""
    from ouroboros import projects_registry as registry

    assert state.init_state().quality == "current"
    registry.create_project(root, "retained", name="Retained")
    memory = _write(root / "memory/identity.md", b"an invented earlier memory")
    registry_file = root / "state/projects.json"
    assert registry_file.is_file()  # this profile had a Project registry
    registry_file.unlink()  # and it is lost: the prior profile's folders are now unknown
    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    assert _hand_off(root) == "reset_requested"
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    status = state_initialization.reset_status(root)
    assert status["status"] == "interrupted" and "registry is missing" in status["error"]
    assert (root / "memory/identity.md").read_bytes() == memory


def _shelves(root, *names):
    for name in names:
        _write(root / "projects" / name / "knowledge/fact.md", f"an invented fact of {name}".encode())
        _write(root / "projects" / name / "journal.jsonl", b'{"kind":"done","text":"invented"}\n')
    return {name: _tree(root / "projects" / name) for name in names}


def test_an_empty_profile_sets_nothing_aside(root):
    assert state.init_state().quality == "current"
    _write(root / "projects/.DS_Store", b"a dot entry is no Project shelf")
    outcome = _reset(root)
    assert (outcome["status"], outcome["projects_set_aside"], outcome["projects_set_aside_path"]) == ("complete", None, None)
    assert state_initialization.read_witness(root)[1]["reset_receipt"]["projects_set_aside"] is None
    assert [entry.name for entry in (root / "projects").iterdir()] == [".DS_Store"]


@pytest.mark.parametrize("failure,code,renames", [
    ("prepare", errno.EACCES, []),
    ("move", errno.EXDEV, ["out alpha", "out beta", "out gamma", "back beta", "back alpha"]),
    ("save", errno.EIO, ["out alpha", "out beta", "out gamma", "back gamma", "back beta", "back alpha"]),
])
def test_a_set_aside_that_fails_puts_every_shelf_back_and_names_no_folder(root, monkeypatch, failure, code, renames):
    from ouroboros import platform_file_rename

    assert state.init_state().quality == "current"
    before = state.STATE_PATH.read_bytes()
    archived = _write(root / "archive/chat_20260101.jsonl", b'{"chat_id":1,"text":"old profile"}\n')
    shelves = _shelves(root, "alpha", "beta", "gamma")
    _write(root / "projects/.DS_Store", b"a dot entry is no Project shelf")
    whole = _tree(root / "projects")
    rename, mkdir, sync, seen = platform_file_rename.rename_exclusive, pathlib.Path.mkdir, state_initialization.fsync_folder, []

    def refuse(path):
        raise OSError(code, os.strerror(code), str(path))

    def another_volume(source, target, **kwargs):  # what a shelf that is its own mount answers
        seen.append(("out " if source.parent.name == "projects" else "back ") + source.name)
        if failure == "move" and source.name == "gamma":
            refuse(target)
        return rename(source, target, **kwargs)

    def no_folder(path, *args, **kwargs):
        return refuse(path) if failure == "prepare" and path.name.startswith(".before-reset-") else mkdir(path, *args, **kwargs)

    def unsaved(path):
        return refuse(path) if failure == "save" and path.name.startswith(".before-reset-") else sync(path)

    with monkeypatch.context() as cut:
        cut.setattr(platform_file_rename, "rename_exclusive", another_volume)
        cut.setattr(pathlib.Path, "mkdir", no_folder)
        cut.setattr(state_initialization, "fsync_folder", unsaved)
        first = _reset(root)
    # The profile that keeps running has every note where it was, byte for byte; nothing was deleted either.
    assert set(first) == {"status", "error"} and first["status"] == "interrupted", first
    assert _tree(root / "projects") == whole and seen == renames
    assert state_initialization.SET_ASIDE_KEY not in state_initialization.read_witness(root)[1]
    assert state.STATE_PATH.read_bytes() == before and (root / "archive/chat_20260101.jsonl").read_bytes() == archived
    said = {"prepare": "could not prepare " + str(root / "projects/.before-reset-"),
            "move": f"could not move {root / 'projects/gamma'} to {root / 'projects/.before-reset-'}",
            "save": "could not save the moves into " + str(root / "projects/.before-reset-")}[failure]
    assert first["error"].startswith("Reset moved no Project notes: " + said) and f"[Errno {code}]" in first["error"]

    done = _reset(root)  # the explicit repeat, with nothing in its way
    container = _set_aside(root, done)
    assert done["status"] == "complete" and {name: _tree(container / name) for name in shelves} == shelves
    assert sorted(entry.name for entry in (root / "projects").iterdir()) == [".DS_Store", container.name]
    assert not (root / "archive").exists() and not state.STATE_PATH.exists()


def test_a_move_back_that_fails_keeps_the_folder_named_and_a_repeat_never_merges(root, monkeypatch):
    import shutil
    from ouroboros import platform_file_rename

    assert state.init_state().quality == "current"
    before = state.STATE_PATH.read_bytes()
    archived = _write(root / "archive/chat_20260101.jsonl", b'{"chat_id":1,"text":"old profile"}\n')
    shelves = _shelves(root, "alpha", "beta", "gamma")
    rename = platform_file_rename.rename_exclusive

    def stuck(source, target, **kwargs):
        if source.name == "gamma":
            raise OSError(errno.EXDEV, os.strerror(errno.EXDEV), str(target))
        if source.name == "beta" and target.parent.name == "projects":  # the way back
            raise OSError(errno.EBUSY, os.strerror(errno.EBUSY), str(target))
        return rename(source, target, **kwargs)

    def kept_whole():  # nothing is deleted while a shelf waits
        return state.STATE_PATH.read_bytes() == before and (root / "archive/chat_20260101.jsonl").read_bytes() == archived

    def in_folder(*names):
        return {name: _tree(container / name) for name in names} == {name: shelves[name] for name in names}

    with monkeypatch.context() as cut:
        cut.setattr(platform_file_rename, "rename_exclusive", stuck)
        first = _reset(root)
    container = _set_aside(root, first)  # the folder still holds notes, so the interrupted Reset names it
    assert first["status"] == "interrupted" and kept_whole()
    for part in (f"could not move {root / 'projects/gamma'} to {container / 'gamma'}: [Errno {errno.EXDEV}]",
                 f"then could not move {container / 'beta'} back to {root / 'projects/beta'}: [Errno {errno.EBUSY}]",
                 f"Project notes moved so far stay in {container}"):
        assert part in first["error"], first["error"]
    # Undoing stopped at beta, so alpha was not tried; gamma never left.
    assert in_folder("alpha", "beta") and _tree(root / "projects/gamma") == shelves["gamma"]
    assert sorted(entry.name for entry in (root / "projects").iterdir()) == [container.name, "gamma"]
    assert state_initialization.read_witness(root)[1]["projects_set_aside"] == first["projects_set_aside"]

    # The old profile is still the running one and writes a note for alpha again. Two shelves of one id are
    # never merged, whatever the day of the repeated request, and that attempt moves nothing else.
    recreated = _write(root / "projects/alpha/workpad.md", b"written after the interruption")
    monkeypatch.setattr(state_initialization, "utc_now_iso", lambda *_: "2031-02-03T04:05:06+00:00")
    second = _reset(root)
    assert second["status"] == "interrupted" and _set_aside(root, second) == container and kept_whole()
    assert second["error"] == (f"Reset found both {root / 'projects/alpha'} and {container / 'alpha'} and merged nothing: "
                               "move or remove one of them, then confirm Reset again")
    assert (root / "projects/alpha/workpad.md").read_bytes() == recreated
    assert in_folder("alpha", "beta") and _tree(root / "projects/gamma") == shelves["gamma"]

    shutil.rmtree(root / "projects/alpha")  # the owner settles the invented conflict by hand
    done = _reset(root)
    assert done["status"] == "complete" and _set_aside(root, done) == container and in_folder(*shelves)
    assert [entry.name for entry in (root / "projects").iterdir()] == [container.name]
    assert not (root / "archive").exists() and not state.STATE_PATH.exists()


def test_a_volume_without_exclusive_rename_still_sets_shelves_aside_and_never_replaces(root, monkeypatch):
    from ouroboros import platform_file_rename

    assert state.init_state().quality == "current"
    shelves = _shelves(root, "alpha")
    plain = []

    def unavailable(source, target, **kwargs):
        raise OSError(errno.ENOTSUP, "this volume has no rename without replacement")

    def plain_rename(source, target):
        assert not os.path.lexists(target), "the plain rename runs only to a name proven absent"
        plain.append(pathlib.Path(source).name)
        return os.replace(source, target)

    monkeypatch.setattr(platform_file_rename, "rename_exclusive", unavailable)
    monkeypatch.setattr(state_initialization.os, "rename", plain_rename)
    done = _reset(root)
    assert done["status"] == "complete" and plain == ["alpha"]
    assert _tree(_set_aside(root, done) / "alpha") == shelves["alpha"]


def test_each_independent_reset_keeps_its_own_folder(root):
    assert state.init_state().quality == "current"
    first_shelves = _shelves(root, "alpha")
    first = _set_aside(root, _reset(root))
    assert state.init_state().quality == "current"  # the new profile starts, then is reset in its turn
    second_shelves = _shelves(root, "alpha", "gamma")
    second = _set_aside(root, _reset(root))
    assert first != second and sorted(entry.name for entry in (root / "projects").iterdir()) == sorted(
        [first.name, second.name])
    assert {name: _tree(first / name) for name in first_shelves} == first_shelves
    assert sorted(entry.name for entry in first.iterdir()) == ["alpha"]
    assert {name: _tree(second / name) for name in second_shelves} == second_shelves


def test_failed_reset_cleanup_keeps_survivors_until_another_explicit_request(root, monkeypatch):
    import shutil

    assert state.init_state().quality == "current"
    before = state.STATE_PATH.read_bytes()
    archived = root / "archive" / "chat_20260101.jsonl"
    _write(archived, b'{"chat_id":1,"text":"old profile"}\n')
    _write(root / "memory/identity.md", b"an invented earlier memory")
    _write(root / "uploads/invented.bin", b"an invented upload")
    remove = shutil.rmtree

    def withheld(path, *args, **kwargs):
        if path == archived.parent:
            raise PermissionError("archive in use")
        return remove(path, *args, **kwargs)

    with monkeypatch.context() as cut:
        cut.setattr(shutil, "rmtree", withheld)
        state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
        assert _hand_off(root) == "reset_requested"
        state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    # The interrupted pass says what it had already removed, and claims nothing else.
    interrupted = state_initialization.reset_status(root)
    # The shared permission-granting removal swallows the withheld error; the absence check then proves
    # the folder is still there, and the interruption says so instead of claiming a removal.
    assert interrupted["status"] == "interrupted" and "present after failed read" in interrupted["error"]
    assert "memory" in interrupted["deleted"] and not {"archive", "uploads"} & set(interrupted["deleted"])
    # The witness names the withheld archive as a pending removal — presence observed, effect unfinished —
    # and the interrupted status above claims neither it nor the removal the pass never reached.
    assert "archive" in state_initialization.read_witness(root)[1]["reset_removed"]
    assert "uploads" not in state_initialization.read_witness(root)[1]["reset_removed"]
    assert not (root / "memory").exists() and (root / "uploads/invented.bin").is_file()
    assert state.init_state().quality == "unavailable"
    assert state.STATE_PATH.read_bytes() == before and archived.is_file()
    assert state_initialization.read_witness(root)[0] == "resetting"
    # The explicit repeat finishes, and its receipt covers the whole Reset: the first pass's removal too.
    done = _reset(root)
    assert done["status"] == "complete" and {"memory", "archive", "uploads"} <= set(done["deleted"])
    assert not archived.parent.exists() and not (root / "uploads").exists() and not state.STATE_PATH.exists()
    assert "reset_removed" not in state_initialization.read_witness(root)[1]


@pytest.mark.parametrize("holds", [False, True], ids=["empty_folder", "retained_shelf"])
def test_the_set_aside_folder_is_named_only_while_it_holds_notes(root, monkeypatch, holds):
    # The folder's name is recorded before the first shelf moves, so the name alone says nothing was set aside.
    assert state.init_state().quality == "current"
    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    monkeypatch.setattr(state_initialization, "_requested_reset", None)  # the generation that asked has ended
    name = "projects/.before-reset-20260101-0123abcd"
    state_initialization._write(root, {**state_initialization.read_witness(root)[1],
                                       state_initialization.SET_ASIDE_KEY: name})
    unnamed = {"status": "interrupted", "error": state_initialization._NO_REASON,
               "reset_id": state_initialization.read_witness(root)[1]["initialization_id"]}
    assert state_initialization.reset_status(root) == unnamed  # no such folder
    (root / name).mkdir(parents=True)
    assert state_initialization.reset_status(root) == unnamed  # an empty one holds no notes
    if holds:
        kept = _write(root / name / "alpha/knowledge/fact.md", b"an invented fact")
        assert _set_aside(root, state_initialization.reset_status(root)) == root / name
    done = _reset(root)
    assert done["status"] == "complete"
    if holds:
        assert _set_aside(root, done) == root / name
        assert (root / name / "alpha/knowledge/fact.md").read_bytes() == kept
    else:
        assert (done["projects_set_aside"], done["projects_set_aside_path"]) == (None, None)


def test_reset_keeps_what_belongs_to_the_installation_and_to_processes_that_outlive_a_profile(root):
    from ouroboros import install_identity, owned_shutdown, process_custody, process_scope, workspace_executor
    from ouroboros.server_entrypoint import SERVER_LOCK_NAME

    # Each kept name is its owner's: the shared engine runs from its runtime, the ledger and the ownership set
    # name that daemon and a restart successor, and Panic's census finds them by the installation marker.
    owners = {install_identity.IDENTITY_PATH.name, process_scope.INSTALLATION_SCOPE_FILE,
              process_scope.PANIC_SWEEP_PENDING_FILE, process_scope.task_scopes_dir(root).name,
              process_custody.LEDGER_FILENAME, owned_shutdown.OWNED_PROCESSES_FILENAME,
              owned_shutdown.PENDING_DIRNAME, workspace_executor._PROCESS_STATE_DIR}
    assert owners <= set(state_initialization._STATE_KEPT)
    assert state.init_state().quality == "current"
    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    assert _hand_off(root) == "reset_requested"
    # Present at the successor's admission. No identity record is invented: a readable one would name another install.
    kept = {}
    for name in ("cx", "python-userbase", "body_adoption", "task_scopes", "workspace_executor_processes",
                 "owned_processes.pending"):
        kept[f"{name}/invented.bin"] = _write(root / "state" / name / "invented.bin", f"invented {name}".encode())
    for name in ("installation_scope.json", "process_ledger.jsonl", "owned_processes.json",
                 "extension_companions.json", "server_process.json", SERVER_LOCK_NAME):
        kept[name] = _write(root / "state" / name, f"invented {name}".encode())
    cleared = ("queue_snapshot.json", "migrations.json", "auth_secret.key", "skills/demo/enabled.json",
               "headless_tasks/invented/data/note.txt", "delegate_recovery/invented.json")
    for name in cleared:
        _write(root / "state" / name, b"the previous profile's")
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    assert state_initialization.reset_status(root)["status"] == "complete"
    for name, raw in kept.items():
        assert (root / "state" / name).read_bytes() == raw, name
    for name in cleared:
        assert not (root / "state" / name).exists(), name
    assert not state.STATE_PATH.exists()


def test_the_hand_off_waits_for_the_custody_owners_records_not_for_an_empty_census(root, monkeypatch):
    from ouroboros import delegate_custody_reconcile as runs, launch_custody, process_custody

    # A generation that started none of the profile's work has an empty census of its own. What decides is what
    # the custody owners still record: here passive doubles of a pending delegated start the engine had accepted,
    # the run it became, and ledgered processes. No engine, process or signal is real.
    state.init_state()
    before = state.STATE_PATH.read_bytes()
    _write(root / "state/current.sqlite", b"")  # a custody store exists; its readers are the doubles below
    run = SimpleNamespace(task_id="invented-task", root_task_id="invented-root", run_id="invented-run")
    starts, open_now, stops, sweeps, stoppable = [{"invocation_id": "invented-start"}], [], [], [], []

    def attach_only():
        pytest.fail("no engine is asked directly once no pending start is left")

    def sweep(drive_root, live, *, gateway_factory, observe_pending):
        sweeps.append((live, gateway_factory, observe_pending()))
        starts.clear()  # what the engine had accepted is bound: it is an open run now
        open_now.append(run)
        return []

    def stop_tree(drive_root, tree, *, gateway_factory):
        stops.append((tree, gateway_factory))
        if stoppable:
            open_now.clear()
        return []

    monkeypatch.setattr(runs, "attach_only_gateway", attach_only)
    monkeypatch.setattr(runs, "pending_invocations", lambda drive_root: list(starts))
    monkeypatch.setattr(runs, "open_runs", lambda drive_root: list(open_now))
    monkeypatch.setattr(runs, "reconcile_orphaned_runs", sweep)
    monkeypatch.setattr(runs, "stop_tree_runs", stop_tree)
    worker = {"pid": 4242, "scope": "session", "purpose": "worker:1"}
    engine = {"pid": 4343, "scope": "daemon", "purpose": "claudexor_daemon"}
    companion = {"pid": 4444, "scope": "daemon", "purpose": "companion:demo:bridge"}
    alive, judged = {id(worker), id(companion)}, []
    custody = launch_custody.LaunchCustody(True, [worker, engine, companion], {}, set(), "", b"", [])

    def unsettled(drive_root, entries, read):
        judged.append([entry["purpose"] for entry in entries])
        return [(entry, "live") for entry in entries if id(entry) in alive]

    monkeypatch.setattr(launch_custody, "read_custody", lambda drive_root, **kw: custody)
    monkeypatch.setattr(launch_custody, "unsettled_selection", unsettled)
    # The installation's daemon is uniquely verified here: live, fingerprint-matching, retained purpose.
    monkeypatch.setattr(process_custody, "spared_daemon_roots", lambda drive_root: {4343})
    monkeypatch.setattr(state_initialization, "_clear_runtime_data", lambda *_: pytest.fail("Reset proceeded"))

    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    assert _hand_off(root) == "reset_interrupted"
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")  # the next boot clears nothing
    assert state_initialization.reset_status(root) == {
        "status": "interrupted", "error": "delegated_runs_unconfirmed:1, processes_unconfirmed:2",
        "reset_id": state_initialization.read_witness(root)[1]["initialization_id"]}
    assert state.STATE_PATH.read_bytes() == before

    # Once the owner's Stop ended the run and the processes are gone, the explicit repeat is handed off.
    stoppable.append(True)
    alive.clear()
    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    assert _hand_off(root) == "reset_requested"
    assert state_initialization.read_witness(root)[1]["reset_handed_off"] is True
    # Every engine contact was attach-only, with no task counted as running; the installation's own daemon was
    # never judged as this profile's work.
    assert sweeps == [(set(), attach_only, True)] and stops == [("invented-root", attach_only)] * 2
    assert judged == [["worker:1", "companion:demo:bridge"]] * 2 and state.STATE_PATH.read_bytes() == before


@pytest.mark.parametrize("cleared,expected", [(False, ["delegated_runs_unknown:OSError"]), (True, [])])
def test_an_unreadable_custody_store_blocks_until_this_reset_began_clearing_it(root, monkeypatch, cleared, expected):
    from ouroboros import delegate_custody_reconcile as runs

    def unreadable(_root):
        raise OSError("the invented custody store cannot be read")

    from ouroboros import store as current_store

    store = root / "state/current.sqlite"
    _write(store, b"")
    monkeypatch.setattr(runs, "pending_invocations", unreadable)
    assert state_initialization._delegated_run_blockers(root, cleared=cleared) == expected
    store.unlink()  # a root that never had the store recorded no delegated work
    assert state_initialization._delegated_run_blockers(root, cleared=False) == []
    # A store that was published and is gone took its custody with it, unless this Reset's own pass removed it.
    monkeypatch.setattr(current_store, "generation_stamped", lambda _root: True)
    assert state_initialization._delegated_run_blockers(root, cleared=cleared) == (
        [] if cleared else ["delegated_custody_lost"])


def test_a_torn_generation_record_is_unknown_delegated_work_never_a_never_used_store(root):
    assert not (root / "state/current.sqlite").exists()
    torn = _write(root / "state/migrations.json", b'{"current_store_generation": ')  # unreadable JSON
    # The retained chain folds positively empty and the store is missing: only the generation record can
    # say "never used", and a torn one says unknown — never a clean "this root recorded no work".
    assert state_initialization._delegated_run_blockers(root, cleared=False) == ["delegated_generation_unreadable"]
    assert (root / "state/migrations.json").read_bytes() == torn
    # Once this Reset's own pass began clearing ``state/``, that folder's record settles nothing either way.
    assert state_initialization._delegated_run_blockers(root, cleared=True) == []


@pytest.mark.parametrize("cleared", [False, True])
def test_a_delegated_start_its_own_custody_has_not_settled_holds_the_reset(root, monkeypatch, cleared):
    from ouroboros import delegate_custody_reconcile as runs

    # Passive doubles: one pending start the custody owner's read could not bind (no answer for this operation
    # from its own engine and store). No answer about anything else, and no earlier pass of this Reset, makes it
    # a start that never ran: a readable store's pending start holds the Reset, and nothing is sent for it.
    _write(root / "state/current.sqlite", b"")
    start, sweeps = {"invocation_id": "invented-start", "task_id": "invented-task"}, []

    def attach_only():
        pytest.fail("an engine that merely answers is no evidence about this start")

    monkeypatch.setattr(runs, "attach_only_gateway", attach_only)
    monkeypatch.setattr(runs, "pending_invocations", lambda drive_root: [start])
    monkeypatch.setattr(runs, "open_runs", lambda drive_root: [])
    monkeypatch.setattr(runs, "stop_tree_runs", lambda *a, **kw: pytest.fail("no run of this profile is open"))
    monkeypatch.setattr(runs, "reconcile_orphaned_runs", lambda drive_root, live, **how: sweeps.append(
        (live, how["gateway_factory"], how["observe_pending"](), sorted(how))) or [])
    assert state_initialization._delegated_run_blockers(root, cleared=cleared) == ["delegated_starts_unknown:1"]
    # The one read was the custody owner's, attach-only, with reads admitted and no replay admitted.
    assert sweeps == [(set(), attach_only, True, ["gateway_factory", "observe_pending"])]


def test_a_review_start_the_reset_read_bound_gets_the_normal_owner_stop(root, monkeypatch):
    from ouroboros import delegate_custody_reconcile as runs

    # The one unfinished review case (#3294), on a made-up profile with mocked custody: a REVIEW
    # surface's pending start — the class the sweep leaves to its panel (#1006), and this Reset
    # stopped the generation whose seat would have rejoined it. The engine's recorded answer for
    # exactly that invocation is a found command: the read-only observer binds it once, and the
    # normal owner Stop ends the bound run behind the Reset's verdict. No engine is real.
    _write(root / "state/current.sqlite", b"")
    start = {"invocation_id": "invented-review-start", "task_id": "invented-reviewed-task",
             "root_task_id": "invented-reviewed-root", "source": "review_substrate:seat"}
    run = SimpleNamespace(task_id="invented-reviewed-task", root_task_id="invented-reviewed-root",
                          run_id="invented-review-run")
    state_now, reads, stops, sweeps = {"pending": [start], "open": []}, [], [], []

    def attach_only():
        return SimpleNamespace(handshake=lambda: None, close=lambda: None)

    def sweep(drive_root, live, **how):  # the sweep leaves the review start to its panel
        sweeps.append((live, how["gateway_factory"], how["observe_pending"]()))
        return []

    def observe(drive_root, gateway, record):  # the engine's accepted answer for this one invocation
        reads.append((record["invocation_id"], record.get("source")))
        state_now.update(pending=[], open=[run])  # bound once: an open run now
        return {"invocation_id": record["invocation_id"], "task_id": record["task_id"],
                "action": "accepted_start_bound", "run_id": run.run_id}

    def stop_tree(drive_root, tree, *, gateway_factory):
        stops.append((tree, gateway_factory))
        state_now.update(open=[])  # the owner's Stop ended the run
        return []

    monkeypatch.setattr(runs, "attach_only_gateway", attach_only)
    monkeypatch.setattr(runs, "pending_invocations", lambda drive_root: list(state_now["pending"]))
    monkeypatch.setattr(runs, "open_runs", lambda drive_root: list(state_now["open"]))
    monkeypatch.setattr(runs, "reconcile_orphaned_runs", sweep)
    monkeypatch.setattr(runs, "stop_tree_runs", stop_tree)
    monkeypatch.setattr(runs, "_observe_pending_invocation", observe)
    assert state_initialization._delegated_run_blockers(root, cleared=False) == []
    # The read reached the review start through the same attach-only seam, the sweep kept out of
    # it, and the bound run left through the normal owner Stop with the Reset's attach-only transport.
    assert reads == [("invented-review-start", "review_substrate:seat")]
    assert sweeps == [(set(), attach_only, True)]
    assert stops == [("invented-reviewed-root", attach_only)]

    # Incomplete evidence preserves the interrupted Reset: a review start the read could not bind
    # stays pending — no answer about it ever reads as absence of work.
    def miss(drive_root, gateway, record):
        return {"invocation_id": record["invocation_id"], "task_id": record["task_id"],
                "action": "invocation_retained", "reason": "accepted_start_not_found"}

    monkeypatch.setattr(runs, "_observe_pending_invocation", miss)
    state_now.update(pending=[start], open=[])
    assert state_initialization._delegated_run_blockers(root, cleared=False) == ["delegated_starts_unknown:1"]


def test_a_missing_store_does_not_settle_legacy_delegated_custody(root):
    # No current store: the retained event chain's owner folds it, and an open run in it blocks the Reset.
    assert not (root / "state/current.sqlite").exists()
    started = {"ts": "2026-11-21T00:00:00Z", "type": "delegate_run_started", "event_id": "e2",
               "custody_ordinal": 2, "run_id": "legacy-run", "task_id": "legacy-task",
               "root_task_id": "legacy-root"}
    _write(root / "logs/events.jsonl", (json.dumps(started) + "\n").encode())
    assert state_initialization._delegated_run_blockers(root, cleared=False) == ["delegated_runs_unconfirmed:1"]
    # A settled run is closed; a still-pending start in the same chain blocks as unknown work.
    requested = {"ts": "2026-11-21T00:00:30Z", "type": "delegate_run_start_requested", "event_id": "e1",
                 "custody_ordinal": 1, "invocation_id": "legacy-start", "task_id": "legacy-task",
                 "operation_id": "legacy-op"}
    settled = {"ts": "2026-11-21T00:01:00Z", "type": "delegate_run_settled", "event_id": "e3",
               "custody_ordinal": 3, "run_id": "legacy-run", "task_id": "legacy-task"}
    with (root / "logs/events.jsonl").open("ab") as handle:
        for row in (requested, settled):
            handle.write((json.dumps(row) + "\n").encode())
    assert state_initialization._delegated_run_blockers(root, cleared=False) == ["delegated_starts_unknown:1"]


def test_an_unreadable_retained_chain_blocks_and_never_reads_as_empty(root, monkeypatch):
    from ouroboros import delegate_custody

    monkeypatch.setattr(delegate_custody, "custody_chain_unreadable", lambda drive_root: True)
    _write(root / "logs/events.jsonl", b'{"type":"delegate_run_started","run_id":"legacy-run"}\n')
    assert state_initialization._retained_chain_blockers(root) == ["delegated_chain_unreadable"]
    assert state_initialization._delegated_run_blockers(root, cleared=False) == ["delegated_chain_unreadable"]


def test_only_a_uniquely_verified_installation_daemon_is_not_profile_work(root, monkeypatch):
    from ouroboros import launch_custody, process_custody
    from ouroboros.claudexor_daemon import CUSTODY_PURPOSE

    worker = {"pid": 4242, "scope": "session", "purpose": "worker:1"}
    engine = {"pid": 4343, "scope": "daemon", "purpose": "claudexor_daemon"}
    legacy = {"pid": 4545, "scope": "session", "purpose": CUSTODY_PURPOSE}  # a pre-scope daemon row
    companion = {"pid": 4646, "scope": "daemon", "purpose": "companion:1"}
    custody = launch_custody.LaunchCustody(True, [worker, engine, legacy, companion], {}, set(), "", b"", [])
    monkeypatch.setattr(launch_custody, "read_custody", lambda drive_root, **kw: custody)
    monkeypatch.setattr(launch_custody, "unsettled_selection",
                        lambda drive_root, entries, read: [(entry, "live") for entry in entries])
    # Uniquely verified (live, fingerprint-matching, retained purpose) — the daemon row and the legacy row
    # under the same retained purpose — are not this profile's writers; a worker or companion row is never spared.
    monkeypatch.setattr(process_custody, "spared_daemon_roots", lambda drive_root: {4343, 4545})
    assert state_initialization._process_blockers(root) == ["processes_unconfirmed:2"]
    # An unverified legacy root stays held with the rest; unknown or empty answers spare nothing at all.
    monkeypatch.setattr(process_custody, "spared_daemon_roots", lambda drive_root: {4343})
    assert state_initialization._process_blockers(root) == ["processes_unconfirmed:3"]
    monkeypatch.setattr(process_custody, "spared_daemon_roots", lambda drive_root: None)
    assert state_initialization._process_blockers(root) == ["processes_unconfirmed:4"]
    monkeypatch.setattr(process_custody, "spared_daemon_roots", lambda drive_root: set())
    assert state_initialization._process_blockers(root) == ["processes_unconfirmed:4"]


def test_a_daemon_row_without_a_verified_identity_is_held_not_spared(root, monkeypatch):
    from ouroboros import launch_custody, process_custody

    worker = {"pid": 4242, "scope": "session", "purpose": "worker:1"}
    engine = {"pid": 4343, "scope": "daemon", "purpose": "claudexor_daemon"}
    nameless = {"pid": None, "scope": "daemon", "purpose": "claudexor_daemon"}
    custody = launch_custody.LaunchCustody(True, [worker, engine, nameless], {}, set(), "", b"", [])
    monkeypatch.setattr(launch_custody, "read_custody", lambda drive_root, **kw: custody)
    monkeypatch.setattr(launch_custody, "unsettled_selection",
                        lambda drive_root, entries, read: [(entry, "live") for entry in entries])
    # The uniquely verified live daemon root is the one row no longer this profile's writer; a row with
    # no pid of its own is never spared by it, and an unknown answer spares nothing at all.
    monkeypatch.setattr(process_custody, "spared_daemon_roots", lambda drive_root: {4343})
    assert state_initialization._process_blockers(root) == ["processes_unconfirmed:2"]
    monkeypatch.setattr(process_custody, "spared_daemon_roots", lambda drive_root: None)
    assert state_initialization._process_blockers(root) == ["processes_unconfirmed:3"]


def test_a_projects_root_that_is_not_a_plain_directory_is_never_traversed(root):
    assert state.init_state().quality == "current"
    stray = _write(root / "projects", b"an ordinary file where the projects directory belongs")
    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    assert _hand_off(root) == "reset_requested"
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    outcome = state_initialization.reset_status(root)
    assert outcome["status"] == "interrupted" and "ordinary directory" in outcome["error"]
    assert (root / "projects").read_bytes() == stray and not (root / "projects").is_dir()
    assert state.STATE_PATH.is_file()  # nothing of the profile was cleared, and no folder was named
    assert state_initialization.SET_ASIDE_KEY not in state_initialization.read_witness(root)[1]


def test_a_panic_before_the_first_mutation_starts_no_set_aside_move(root, monkeypatch):
    assert state.init_state().quality == "current"
    _write(root / "projects/alpha/knowledge/fact.md", b"kept")
    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    assert _hand_off(root) == "reset_requested"
    _write(root / "state/panic_stop.flag", b"panic")  # after the hand-off, before the admission's first move
    monkeypatch.setattr(state_initialization, "_withdrawn_by_panic", lambda *args: False)
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    outcome = state_initialization.reset_status(root)
    assert outcome["status"] == "interrupted" and outcome["error"] == "Panic interrupted Reset"
    assert (root / "projects/alpha/knowledge/fact.md").read_bytes() == b"kept"
    assert not list((root / "projects").glob(".before-reset-*"))


def test_a_panic_between_two_moves_starts_no_further_set_aside_move_and_puts_them_back(root, monkeypatch):
    from ouroboros import platform_file_rename

    assert state.init_state().quality == "current"
    _shelves(root, "alpha", "beta", "gamma")
    whole = _tree(root / "projects")
    marker = root / "state/panic_stop.flag"
    rename = platform_file_rename.rename_exclusive
    seen = []

    def panic_lands(source, target, **kwargs):
        seen.append(("out " if source.parent.name == "projects" else "back ") + source.name)
        moved = rename(source, target, **kwargs)
        if source.name == "alpha" and source.parent.name == "projects":
            _write(marker, b"panic")  # Panic's own input, observed after the first move only
        return moved

    with monkeypatch.context() as cut:
        cut.setattr(platform_file_rename, "rename_exclusive", panic_lands)
        state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
        assert _hand_off(root) == "reset_requested"
        state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    outcome = state_initialization.reset_status(root)
    assert outcome["status"] == "interrupted" and outcome["error"].endswith("Panic interrupted Reset")
    assert f"could not move {root / 'projects/beta'}" in outcome["error"]
    assert seen == ["out alpha", "back alpha"]  # beta never moved: no new move once Panic was observed
    assert _tree(root / "projects") == whole and state.STATE_PATH.is_file()
    assert state_initialization.SET_ASIDE_KEY not in state_initialization.read_witness(root)[1]
    # Nothing was removed yet, so the Panic withdraws the whole request and the profile keeps serving.
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    assert state_initialization.reset_status(root) == {"status": "none"}
    assert _tree(root / "projects") == whole and state.init_state().quality == "current"


def test_reset_removes_read_only_files_through_the_shared_removal(root):
    assert state.init_state().quality == "current"
    _write(root / "memory/locked.bin", b"read-only bytes")
    os.chmod(root / "memory/locked.bin", 0o444)
    done = _reset(root)
    assert done["status"] == "complete" and "memory" in done["deleted"] and not (root / "memory").exists()


def test_a_legacy_completed_reset_without_a_detailed_receipt_reads_complete_not_unknown(root):
    state_initialization.mark_pending(root, origin="owner_reset")
    witness = {**state_initialization.read_witness(root)[1], "phase": "complete"}
    state_initialization._write(root, witness)
    assert state_initialization.read_witness(root) == ("ok", witness)
    assert state_initialization.reset_status(root) == {"status": "complete",
                                                       "reset_id": witness["initialization_id"]}
    # An incoherent witness (resetting without its basis) stays unknown, never complete — and names no identity,
    # as the receipt it cannot validate could not have matched one.
    state_initialization._write(root, {**witness, "phase": "resetting"})
    assert state_initialization.reset_status(root) == {"status": "unknown"}


def test_the_reset_receipt_names_that_resets_identity_and_no_reset_names_none(root):
    state_initialization.mark_pending(root, origin="owner_reset")
    identity = state_initialization.read_witness(root)[1]["initialization_id"]
    state_initialization._write(root, {**state_initialization.read_witness(root)[1], "phase": "complete"})
    assert state_initialization.reset_status(root) == {"status": "complete", "reset_id": identity}
    # A witness that belongs to no owner Reset names no identity: the receipt never invents one.
    state_initialization.mark_pending(root, origin="first_boot")
    assert state_initialization.reset_status(root) == {"status": "none"}


def test_a_recorded_receipt_that_does_not_validate_is_never_certified_complete(root):
    state_initialization.mark_pending(root, origin="owner_reset")
    witness = {**state_initialization.read_witness(root)[1], "phase": "complete"}
    identity = witness["initialization_id"]
    # Another Reset's identity, and a wrong shape: both receipts are recorded, neither validates.
    wrong_identity = {"initialization_id": "another-reset", "deleted": ["memory"], "retained_logs": []}
    state_initialization._write(root, {**witness, "reset_receipt": wrong_identity})
    assert state_initialization.reset_status(root) == {"status": "unknown"}
    wrong_shape = {"initialization_id": identity, "deleted": "memory", "retained_logs": []}
    state_initialization._write(root, {**witness, "reset_receipt": wrong_shape})
    assert state_initialization.reset_status(root) == {"status": "unknown"}
    # Truly absent details on a completed Reset stay complete without them.
    state_initialization._write(root, witness)
    assert state_initialization.reset_status(root) == {"status": "complete", "reset_id": identity}


def test_unconfirmed_reset_cleanup_survives_boot_without_running_again(root, monkeypatch):
    from ouroboros import launch_roots, projects_registry
    from ouroboros.server_boot import reconcile_projects

    # The erased test root survives a failed completion write without another
    # destructive pass or a new profile. All files and calls here are synthetic.
    state_initialization.mark_pending(root, origin="first_boot", work_root_basis=launch_roots.HOME_BASIS)
    assert state.init_state().quality == "current"
    shelf = root / "projects/retained/knowledge/kept.md"
    _write(shelf, b"kept")
    write = state_initialization._write

    def deny_completion(drive_root, record):
        if record["phase"] == "pending":
            raise OSError("cleanup receipt not saved")
        write(drive_root, record)

    with monkeypatch.context() as cut:
        cut.setattr(state_initialization, "_write", deny_completion)
        state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
        assert _hand_off(root) == "reset_requested"
        state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")

    resetting, saved = state_initialization.read_witness(root)
    assert resetting == "resetting"
    assert launch_roots.durable_basis(root) == (launch_roots.HOME_BASIS, "witness")
    assert not state.STATE_PATH.exists() and not state.STATE_LAST_GOOD_PATH.exists()
    reconcile_projects(root)
    assert projects_registry._registry_path(root).exists()  # publishing the empty map preceded the cut
    with pytest.raises(FileNotFoundError):
        projects_registry.create_project(root, "premature")
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    assert state.init_state().quality == "unavailable"
    assert state_initialization.read_witness(root) == (resetting, saved)
    # The interrupted pass had already set the shelf aside; its folder is named until the repeat finishes.
    interrupted = state_initialization.reset_status(root)
    moved = _set_aside(root, interrupted) / "retained/knowledge/kept.md"
    assert interrupted["status"] == "interrupted" and moved.read_bytes() == b"kept" and not shelf.exists()

    # The repeat finds nothing left to move and still reports the folder its first pass filled.
    assert _set_aside(root, _reset(root)) == moved.parents[2]
    assert state.init_state().values["initialization_id"] == saved["initialization_id"]
    assert projects_registry.create_project(root, "new")["created"]
    assert moved.read_bytes() == b"kept" and not shelf.exists()


def test_stop_during_an_outage_outlives_the_primary_coming_back(root, monkeypatch):
    from supervisor import evolution_lifecycle as lifecycle
    from supervisor.events_runtime_controls import owner_evolution_start, owner_evolution_stop_controls

    state.save_state(_prior())
    assert lifecycle.start_evolution_campaign("improve", source="owner")
    real = pathlib.Path.read_bytes
    outage = {"on": True}
    monkeypatch.setattr(pathlib.Path, "read_bytes", lambda self: (_ for _ in ()).throw(
        OSError(errno.EIO, "io")) if outage["on"] and self == state.STATE_PATH else real(self))
    disclosure = owner_evolution_stop_controls("owner /evolve off")
    assert "runtime state" in disclosure and "campaign" not in disclosure
    outage["on"] = False
    assert state.control_is(state.load_state(), "evolution_mode_enabled", True)  # the stale primary returns
    assert lifecycle.evolution_stop_reason() == "stop_latched"
    lifecycle._STOP_LATCH["stopped"] = False  # a restart loses the latch; the durable intent stays
    assert lifecycle.evolution_stop_reason() == "stop_intent"
    assert not lifecycle.start_evolution_campaign("agent", source="agent_tool")
    assert lifecycle._read_evolution_campaign()["status"] == "active"  # no invented terminal status
    assert owner_evolution_start("resume", source="owner_chat") == {}
    assert lifecycle.evolution_stop_reason() == "" and "stop_intent" not in lifecycle._read_evolution_campaign()
    assert state.control_is(state.load_state(), "evolution_mode_enabled", True)


def test_panic_requests_every_physical_stop_before_any_persistence_and_never_waits(root, monkeypatch):
    import ouroboros.server_control as control
    from ouroboros import workspace_executor
    from ouroboros.tools import services, shell_process
    from supervisor import evolution_lifecycle as lifecycle

    # Production exits permanently; this in-process exit stub must not leave
    # its terminal admission latches set for subsequent consumer tests, nor kill
    # this worker's own multiprocessing children (the shared supervisor event bus).
    for owner in (workspace_executor, services, shell_process):
        monkeypatch.setattr(owner, "_panic_requested", False)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    _write(state.STATE_PATH, _prior())
    order: list = []
    release = threading.Event()
    monkeypatch.setattr(state, "update_state", lambda *a, **k: order.append("state") or release.wait(30))
    monkeypatch.setattr(lifecycle, "record_evolution_stop_intent", lambda *a, **k: order.append("intent"))
    monkeypatch.setattr(control, "_write_panic_flag", lambda data_dir: (
        order.append("flag"), control_write(data_dir)))
    control_write = lambda data_dir: (data_dir / "state" / "panic_stop.flag").write_text("panic")  # noqa: E731
    monkeypatch.setattr(control.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    import ouroboros.platform_layer as platform_layer

    monkeypatch.setattr(platform_layer, "kill_process_on_port", lambda port: order.append(("port", port)))
    # The real panic kills every multiprocessing child of THIS process; in a test that
    # would include another test's process-global event bus manager, leaving a dead
    # proxy that breaks every later emission. Like the other in-process panic tests.
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    started = time.monotonic()
    with pytest.raises(SystemExit):
        control.execute_panic_stop(  # a stalled log handler and a held clock lock block nothing either
            SimpleNamespace(stop=lambda: release.wait(30)),
            lambda **kw: order.append("workers"), data_dir=root, panic_exit_code=99,
            log=SimpleNamespace(critical=lambda *a, **k: release.wait(30)))
    release.set()
    assert time.monotonic() - started < 15  # every write bounded, never awaited
    assert "workers" not in order  # lifelines own pooled child requests
    assert not [step for step in order if isinstance(step, tuple) and step[0] == "port"]  # no port sweep
    assert order.index("flag") < order.index("state")
    assert (root / "state" / "panic_stop.flag").read_text() == "panic"


def test_an_owner_restart_never_replaces_a_panic_flag_still_owed_its_controls(root, monkeypatch):
    import ouroboros.server_restart as restart
    from supervisor import worker_chat_lane

    flag = root / "state" / "panic_stop.flag"
    _write(flag, b"panic")  # boot kept it: the disabled controls were not durable yet
    monkeypatch.setattr(restart, "DATA_DIR", root)
    monkeypatch.setattr(restart, "_safe_restart_serialized", lambda *a, **k: (True, ""))
    monkeypatch.setattr(restart, "_stop_owned_work", lambda ctx: [])
    monkeypatch.setattr(restart, "_request_restart_exit", lambda owner=False: None)
    assert restart._perform_owner_restart(SimpleNamespace(safe_restart=None)) == (True, "")
    from ouroboros.delegate_recovery import active_restart_transaction
    assert flag.read_text() == "panic" and active_restart_transaction(root)["door"] == "owner_restart"
    monkeypatch.setattr(worker_chat_lane, "_pool", lambda: SimpleNamespace(DRIVE_ROOT=root, load_state=state.load_state))
    state.save_state(_prior())
    worker_chat_lane.auto_resume_after_restart()  # the owner-restart boot still settles the Panic debt
    assert not flag.exists()
    assert state.control_value(state.load_state(), "bg_consciousness_enabled") == (True, False)


def test_boot_consumes_the_panic_flag_only_after_its_controls_are_durable(root, monkeypatch):
    from supervisor import worker_chat_lane

    monkeypatch.setattr(worker_chat_lane, "_pool", lambda: SimpleNamespace(DRIVE_ROOT=root, load_state=state.load_state))
    state.save_state(_prior())
    primary, backup = state.STATE_PATH.read_bytes(), state.STATE_LAST_GOOD_PATH.read_bytes()
    state.STATE_PATH.unlink()
    state.STATE_LAST_GOOD_PATH.unlink()
    flag = root / "state" / "panic_stop.flag"
    _write(flag, b"panic")
    worker_chat_lane.auto_resume_after_restart()  # initialized state unavailable: controls not durable
    assert flag.exists()
    # The retained queue is now prior history: repair known state, never mint defaults.
    state.STATE_PATH.write_bytes(primary)
    state.STATE_LAST_GOOD_PATH.write_bytes(backup)
    worker_chat_lane.auto_resume_after_restart()
    assert not flag.exists()
    live = state.load_state()
    for key, value in (("evolution_mode_enabled", False), ("bg_consciousness_enabled", False),
                       ("evolution_owner_stopped", True)):
        assert state.control_value(live, key) == (True, value)


def _bridge(text: str, *, source: str = "telegram", user: int = 99, chat: int = 99):
    class Bridge:
        def get_updates(self, offset=0, timeout=1):
            return [{"update_id": offset, "message": {"chat": {"id": chat}, "from": {"id": user}, "text": text,
                                                       "source": source}}]

    return Bridge()


def _ingress_ctx(root, replies, panics):
    return SimpleNamespace(
        RUNNING={}, consciousness=None, DRIVE_ROOT=root,
        load_state=state.load_state, update_state=state.update_state, save_state=state.save_state,
        send_with_budget=lambda _chat, text, **_kw: replies.append(text),
        kill_workers=lambda **kw: panics.append("kill"),
    )


@pytest.mark.parametrize("readable", [True, False])
def test_external_commands_need_a_positively_bound_owner(root, monkeypatch, readable):
    import server
    import supervisor.message_bus as message_bus

    monkeypatch.setattr(message_bus, "log_chat", lambda *a, **k: None)
    monkeypatch.setattr(message_bus, "record_inbound_message", lambda *a, **k: {}, raising=False)
    panics: list = []
    monkeypatch.setattr(server, "_execute_panic_stop", lambda *a: panics.append("panic"))
    backup = _prior(owner_external_id=None, owner_external_chat_id=None)
    _write(state.STATE_LAST_GOOD_PATH, backup)
    if readable:
        _write(state.STATE_PATH, _prior())
        state.init_state()
    replies: list = []
    ctx = _ingress_ctx(root, replies, panics)
    server._process_bridge_updates(_bridge("/panic", user=5, chat=5), 0, ctx)  # a stranger
    assert panics == []
    if readable:
        server._process_bridge_updates(_bridge("/panic"), 0, ctx)  # the bound owner
        assert panics == ["panic"]
    else:
        # Unknown binding (only a backup is readable): refused and NEVER registered —
        # neither as the external owner nor as the global owner.
        assert any("unknown" in reply for reply in replies)
        live = state.load_state()
        assert state.control_value(live, "owner_external_id") == (False, None)
        assert state.control_value(live, "owner_id") == (False, None)


@pytest.mark.parametrize("primary", ["corrupt", "unreadable", "other_identity"])
def test_a_set_owner_binding_of_the_same_identity_survives_a_recovery(root, monkeypatch, primary):
    import server
    import supervisor.message_bus as message_bus

    monkeypatch.setattr(message_bus, "log_chat", lambda *a, **k: None)
    monkeypatch.setattr(message_bus, "record_inbound_message", lambda *a, **k: {}, raising=False)
    panics: list = []
    monkeypatch.setattr(server, "_execute_panic_stop", lambda *a: panics.append("panic"))
    assert state.init_state().quality == "current"  # a first boot: the witness identity
    identity = json.loads(state.STATE_PATH.read_bytes())["initialization_id"]
    backup = _prior(initialization_id="someone-else" if primary == "other_identity" else identity,
                    owner_id=None)
    _write(state.STATE_LAST_GOOD_PATH, backup)
    _write(state.STATE_PATH, b'{"owner_external_id": 5')
    if primary == "unreadable":
        real = pathlib.Path.read_bytes
        monkeypatch.setattr(pathlib.Path, "read_bytes", lambda self: (_ for _ in ()).throw(
            OSError(errno.EIO, "io")) if self == state.STATE_PATH else real(self))
    replies: list = []
    server._process_bridge_updates(_bridge("/panic", user=5, chat=5), 0, _ingress_ctx(root, replies, panics))
    server._process_bridge_updates(_bridge("/panic"), 0, _ingress_ctx(root, replies, panics))
    shown = state.load_state()
    if primary == "other_identity":  # a copy of another install proves nothing
        assert panics == [] and state.control_value(shown, "owner_external_id") == (False, None)
        return
    assert panics == ["panic"]  # the bound owner only, never the stranger
    assert state.control_value(shown, "owner_external_id") == (True, 99)
    assert state.control_value(shown, "owner_id") == (False, None)  # an empty slot may have been filled
    assert state.control_value(shown, "evolution_owner_stopped") == (False, None)


def test_the_local_panic_door_reads_no_state(root, monkeypatch):
    import server

    panics: list = []
    monkeypatch.setattr(server, "_execute_panic_stop", lambda *a: panics.append("panic"))
    ctx = _ingress_ctx(root, [], panics)
    ctx.load_state = lambda: pytest.fail("the local /panic must not read state first")
    server._process_bridge_updates(_bridge("/panic", source="web", user=1, chat=1), 0, ctx)
    assert panics == ["panic"]


def test_a_file_where_the_state_directory_belongs_is_unknown_not_absent(root):
    import shutil

    shutil.rmtree(root / "state", ignore_errors=True)
    (root / "state").write_text("not a directory", encoding="utf-8")
    read = state.init_state()  # typed unavailable, never a crash and never a mint
    assert read.quality == "unavailable" and read.values == {}
    assert state_initialization.supervisor_evidence(root)[0] in {"none", "unknown"}
    (root / "state").unlink()
    (root / "task_results").write_text("a file", encoding="utf-8")
    assert state_initialization.supervisor_evidence(root)[0] == "unknown"


def _windows_shaped_not_found(monkeypatch):
    """Where POSIX raises ENOTDIR because a PARENT component is a file, Windows raises
    ERROR_PATH_NOT_FOUND: the same FileNotFoundError as a missing parent. Only the readers'
    own operation errors are reshaped; the absence proof's ``os.stat`` stays real."""
    def reshaped(real):
        def call(path, *args, **kwargs):
            try:
                return real(path, *args, **kwargs)
            except NotADirectoryError as exc:
                if os.path.isdir(os.path.dirname(os.fspath(path))):
                    raise  # the path itself is the file: Windows says ENOTDIR too (WinError 267)
                raise FileNotFoundError(errno.ENOENT, "[WinError 3] path not found", os.fspath(path)) from exc
        return call

    monkeypatch.setattr(pathlib.Path, "read_bytes", reshaped(pathlib.Path.read_bytes))
    monkeypatch.setattr(pathlib.Path, "lstat", reshaped(pathlib.Path.lstat))
    monkeypatch.setattr(os, "lstat", reshaped(os.lstat))
    monkeypatch.setattr(os, "scandir", reshaped(os.scandir))


@pytest.mark.parametrize("errors", ["native", "windows"])
def test_only_a_directory_ancestor_proves_the_state_absent(root, monkeypatch, errors):
    import shutil

    if errors == "windows":
        _windows_shaped_not_found(monkeypatch)
    shutil.rmtree(root / "state", ignore_errors=True)
    (root / "state").write_text("not a directory", encoding="utf-8")
    assert state.read_state_copy(state.STATE_PATH)[0] == "unreadable"
    assert state_initialization.read_witness(root)[0] == "unreadable"
    assert state_initialization.supervisor_evidence(root)[0] == "unknown"
    assert state.init_state().quality == "unavailable"
    assert (root / "state").read_bytes() == b"not a directory"  # nothing minted over it
    (root / "state").unlink()  # truly absent below a real directory: a first boot
    assert state.read_state_copy(state.STATE_PATH)[0] == "missing"
    assert state_initialization.supervisor_evidence(root) == ("none", "")
    assert state.init_state().quality == "current" == state.read_state().quality
    assert state_initialization.read_witness(root)[1]["phase"] == "complete"
    remove_store_tree(root / "state")  # the boot above opened its stores
    real_stat = os.stat
    monkeypatch.setattr(os, "stat", lambda path, *a, **k: (_ for _ in ()).throw(PermissionError(
        errno.EACCES, "denied")) if pathlib.Path(path) == root / "state" else real_stat(path, *a, **k))
    assert state.read_state_copy(state.STATE_PATH)[0] == "unreadable"  # an unexaminable ancestor is unknown


def test_a_first_state_whose_witness_cannot_complete_admits_nothing_this_boot(root, monkeypatch):
    real_complete = state_initialization.complete
    monkeypatch.setattr(state_initialization, "complete", lambda *a, **k: False)
    read = state.init_state()
    assert read.quality == "unavailable" and "witness" in read.reason
    with pytest.raises(state.StateUnavailable):
        state.update_state(lambda live: live.update(owner_id=5))  # no registration on that identity
    assert not state.control_value(state.load_state(), "owner_id")[0]
    pending = state_initialization.read_witness(root)[1]
    assert pending["phase"] == "pending"
    monkeypatch.setattr(state_initialization, "complete", real_complete)
    state.init(root)  # the next boot adopts and completes the SAME identity
    assert state.init_state().values["initialization_id"] == pending["initialization_id"]
    assert state_initialization.read_witness(root)[1]["phase"] == "complete"


def test_a_whole_state_write_mints_only_where_initialization_would(root):
    state.save_state({"owner_chat_id": 1})  # a fresh root: a first state with a completed witness
    witness = state_initialization.read_witness(root)[1]
    assert witness["phase"] == "complete" and state.load_state()["initialization_id"] == witness["initialization_id"]
    state.save_state({"owner_chat_id": 2})
    assert state.load_state()["initialization_id"] == witness["initialization_id"]  # identity kept
    state.STATE_PATH.unlink()
    with pytest.raises(state.StateUnavailable):  # a recoverable backup is not overwritten
        state.save_state({})
    state.STATE_LAST_GOOD_PATH.unlink()
    with pytest.raises(state.StateUnavailable):  # a lost initialized state is never re-minted
        state.save_state({})
    assert not state.STATE_PATH.exists()


def test_a_writer_racing_an_owner_reset_never_resurrects_the_old_identity(root):
    raw = _write(state.STATE_PATH, _prior(initialization_id="old-id"))
    _write(state.STATE_LAST_GOOD_PATH, raw)
    state_initialization.mark_pending(root, origin="owner_reset")
    read = state.init_state()
    assert read.quality == "unavailable"
    assert state.STATE_PATH.read_bytes() == raw == state.STATE_LAST_GOOD_PATH.read_bytes()


def test_owner_bindings_have_only_known_empty_slot_writers():
    """The proof a backup's SET binding relies on (``_backup_unconfirmed``): every
    production write of an owner binding key sits behind a known-empty-slot check."""
    import ast

    keys = {"owner_id", "owner_chat_id", "owner_external_id", "owner_external_chat_id"}
    repo = pathlib.Path(__file__).resolve().parents[1]
    writers = []
    for path in [repo / "server.py", *repo.joinpath("ouroboros").rglob("*.py"), *repo.joinpath("supervisor").rglob("*.py")]:
        tree = parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
                node, ast.AugAssign) else []
            for target in targets:
                if (isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                        and target.slice.value in keys):
                    writers.append((path.name, target.slice.value))
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"update", "pop"} and any(
                        (isinstance(arg, ast.Constant) and arg.value in keys) for arg in node.args)
                    or isinstance(node, ast.Call) and any(kw.arg in keys for kw in node.keywords)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "update"):
                writers.append((path.name, "update/pop"))
    assert sorted(writers) == sorted([("server.py", "owner_id"), ("server.py", "owner_chat_id"),
                                      ("server.py", "owner_external_id"), ("server.py", "owner_external_chat_id")])
    source = (repo / "server.py").read_text(encoding="utf-8")
    assert source.count('control_value(live, "owner_id") == (True, None)') == 1
    assert source.count('control_value(live, "owner_external_id") == (True, None)') == 1


def test_an_initialization_write_failure_is_typed_never_fatal(root, monkeypatch):
    def refuse(*_a, **_k):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(state_initialization, "_write", refuse)
    read = state.init_state()  # the pending witness cannot be written: unavailable, supervisor keeps serving
    assert read.quality in {"uninitialized", "unavailable"} and not state.STATE_PATH.exists()


@pytest.mark.parametrize('failure', ['complete', 'backup'])
def test_interrupted_initialization_never_grants_on_repeated_boot(root, monkeypatch, failure):
    real_complete, real_write = state_initialization.complete, state.atomic_write_text
    if failure == 'complete':
        monkeypatch.setattr(state_initialization, 'complete', lambda *a, **k: False)
    else:
        def write(path, body):
            if path == state.STATE_LAST_GOOD_PATH:
                raise OSError(errno.ENOSPC, 'backup full')
            return real_write(path, body)
        monkeypatch.setattr(state, 'atomic_write_text', write)
    for _ in range(2):
        state.init(root)  # no process-local latch is relied on
        assert state.init_state().quality == 'unavailable'
        assert state.control_value(state.load_state(), 'owner_id') == (False, None)
        assert state.control_in_copy(state.STATE_PATH, 'owner_id') == (False, None)
    identity = json.loads(state.STATE_PATH.read_text())['initialization_id']
    monkeypatch.setattr(state_initialization, 'complete', real_complete)
    monkeypatch.setattr(state, 'atomic_write_text', real_write)
    assert state.init_state().quality == 'current'
    assert state.read_state().values['initialization_id'] == identity


def test_complete_witness_does_not_adopt_mismatched_primary(root):
    state.init_state()
    prior = json.loads(state.STATE_PATH.read_text())
    _write(state.STATE_PATH, {**prior, 'initialization_id': 'foreign'})
    assert state.read_state().reason == 'initialization_identity_mismatch'
    assert state.init_state().quality == 'unavailable'
    assert state_initialization.read_witness(root)[1]['initialization_id'] == prior['initialization_id']
    with pytest.raises(state.StateUnavailable):
        state.update_state(lambda st: st.update(owner_id=99))


def test_partial_state_write_is_typed_and_stop_without_campaign_is_durable(root, monkeypatch):
    from supervisor import evolution_lifecycle as lifecycle
    from supervisor.events_runtime_controls import owner_evolution_stop_controls

    state.save_state(_prior())
    real = state.atomic_write_text
    def write(path, body):
        if path == state.STATE_LAST_GOOD_PATH:
            raise OSError(errno.EIO, 'backup unavailable')
        return real(path, body)
    monkeypatch.setattr(state, 'atomic_write_text', write)
    with pytest.raises(state.StateUnavailable) as err:
        state.update_state(lambda st: st.update(last_owner_message_at='now'))
    assert err.value.reason == 'backup_write_failed' and err.value.primary_written
    assert 'runtime state' in owner_evolution_stop_controls('owner off')
    campaign = lifecycle._read_evolution_campaign()
    assert campaign['stop_intent']['reason'] == 'owner off'
    assert 'status' not in campaign and 'id' not in campaign
    lifecycle._STOP_LATCH['stopped'] = False
    assert lifecycle.evolution_stop_reason() == 'stop_intent'


def test_panic_flag_blocks_constructor_tick_and_public_wake_after_failed_disable(root, monkeypatch):
    from ouroboros.consciousness import BackgroundConsciousness
    from supervisor import worker_chat_lane

    state.save_state(_prior())
    _write(root / 'state/panic_stop.flag', b'panic')
    monkeypatch.setattr(worker_chat_lane, '_pool', lambda: SimpleNamespace(DRIVE_ROOT=root))
    monkeypatch.setattr(state, 'update_state', lambda *a, **k: (_ for _ in ()).throw(
        state.StateUnavailable('backup_write_failed', primary_written=True)))
    worker_chat_lane.auto_resume_after_restart()
    clock = BackgroundConsciousness(root, root, lambda: 7, now=0)
    assert not clock.enabled
    assert clock.tick(now=99999) == 'panic_stop'
    monkeypatch.setattr(worker_chat_lane, 'wake_gate_refusal', lambda: '')
    assert worker_chat_lane.handle_wake_direct(7, 'wake', {})['reason'] == 'consciousness_disabled_or_unknown'
    assert (root / 'state/panic_stop.flag').exists()


def test_recovery_uses_existing_owner_stop_evidence_without_reset(root):
    from supervisor import evolution_lifecycle as lifecycle

    state.save_state(_prior())
    assert lifecycle.record_evolution_stop_intent("owner", "stop during outage")
    _write(state.STATE_PATH, b"broken")
    state.init_state()
    restored = state.load_state()
    assert state.control_value(restored, "evolution_owner_stopped") == (True, True)
    assert state.control_value(restored, "evolution_mode_enabled") == (True, False)
    assert state.control_value(restored, "owner_chat_id") == (True, _prior()["owner_chat_id"])


def test_raw_campaign_and_state_errors_do_not_skip_actual_stop_cancellation(root, monkeypatch):
    import ouroboros.server_owner_routing as routing
    from supervisor import evolution_lifecycle, queue

    attempts = []
    def fail_campaign(*args, **kwargs):
        attempts.append("campaign")
        raise OSError(28, "full")
    def fail_state(*args, **kwargs):
        attempts.append("state")
        raise OSError(5, "io")
    monkeypatch.setattr(evolution_lifecycle, "record_evolution_stop_intent", fail_campaign)
    monkeypatch.setattr(state, "update_state", fail_state)
    monkeypatch.setattr(queue, "stop_evolution_tasks", lambda *_: attempts.append("cancel") or {})
    monkeypatch.setattr(queue, "evolution_stop_report", lambda _: ([], True))
    ctx = SimpleNamespace(DRIVE_ROOT=root, sort_pending=lambda: None, persist_queue_snapshot=lambda **_: None)
    routing._owner_evolution_stop(ctx, 1)
    assert attempts[:3] == ["campaign", "state", "cancel"]
    assert evolution_lifecycle._STOP_LATCH["stopped"] is True


def test_the_saved_working_folder_proof_guides_a_repeat_after_the_clear(root, monkeypatch, tmp_path_factory):
    """The guard saves the prior working folders before any effect; a continuation of the same
    Reset reads the saved proof instead of a registry the first pass already cleared."""
    from ouroboros import projects_registry as registry

    assert state.init_state().quality == "current"
    work = tmp_path_factory.mktemp("working-folder")
    kept = _write(work / "src/main.py", b"invented working file\n")
    registry.create_project(root, "retained", name="Retained", working_dir=str(work))
    _write(root / "memory/identity.md", b"an invented earlier memory")
    write = state_initialization._write

    def deny_completion(drive_root, record):
        if record["phase"] == "pending":
            raise OSError("receipt not saved")
        write(drive_root, record)

    with monkeypatch.context() as cut:
        cut.setattr(state_initialization, "_write", deny_completion)
        state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
        assert _hand_off(root) == "reset_requested"
        state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    saved = state_initialization.read_witness(root)[1]
    assert saved.get("reset_working_folders") == [str(work)]
    assert saved.get("reset_clearing_state") is True
    done = _reset(root)
    assert done["status"] == "complete"
    assert (work / "src/main.py").read_bytes() == kept


def test_an_undecided_work_root_default_blocks_the_reset(root, monkeypatch):
    """WorkRootUnavailable means the default is undecided (basis ambiguous), not that no root
    exists: the guard refuses rather than silently proceeding without a possible work folder."""
    from ouroboros import config
    from ouroboros.launch_roots import WorkRootUnavailable

    assert state.init_state().quality == "current"
    _write(root / "memory/identity.md", b"an invented earlier memory")
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    monkeypatch.setattr(config, "get_subagent_worktree_root",
                        lambda: (_ for _ in ()).throw(WorkRootUnavailable("invented ambiguity")))
    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    assert _hand_off(root) == "reset_requested"
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    status = state_initialization.reset_status(root)
    assert status["status"] == "interrupted" and "undecided" in status["error"]
    assert (root / "memory/identity.md").read_bytes() == b"an invented earlier memory"


def test_split_settings_keys_from_the_root_document_count_in_the_guard(root, monkeypatch):
    """When the configured settings live elsewhere, the root's own settings.json carries its
    work-root keys alongside; the guard checks both before any effect."""
    from ouroboros import config

    assert state.init_state().quality == "current"
    work = root / "uploads" / "working-folder"
    _write(work / "src/main.py", b"invented\n")
    _write(root / "settings.json", {"OUROBOROS_SUBAGENT_WORKTREE_ROOT": str(work)})
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "state" / "elsewhere.json")
    _write(root / "state" / "elsewhere.json", {})
    _write(root / "memory/identity.md", b"an invented earlier memory")
    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    assert _hand_off(root) == "reset_requested"
    state_initialization.accept_runtime_reset(root, boot_protection=lambda: "local")
    status = state_initialization.reset_status(root)
    assert status["status"] == "interrupted" and "working folder" in status["error"]
    assert (work / "src/main.py").read_bytes() == b"invented\n"
