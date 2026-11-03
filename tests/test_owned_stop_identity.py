"""A measured stranger ends stale custody; an unknown or live owner does not."""

import json
import time

import pytest

from ouroboros import owned_shutdown as owned, platform_layer, process_custody as custody


@pytest.fixture
def member(tmp_path, monkeypatch):
    row = {"pid": 600, "pgid": 600, "purpose": "service:fixture", "scope": "task",
           "owner_task": "task", "session_id": "previous-generation",
           "fingerprint": {"start_time": "recorded-birth", "cmd_sha256": custody._cmd_sha256("service")}}
    state = {"alive": True, "birth": "recorded-birth", "command": row["fingerprint"]["cmd_sha256"]}
    signals = []
    monkeypatch.setattr(custody, "IS_WINDOWS", False)
    monkeypatch.setattr(custody, "pid_is_alive", lambda _: state["alive"])
    monkeypatch.setattr(custody, "pid_is_zombie", lambda _: False)
    monkeypatch.setattr(custody, "process_start_time", lambda _: state["birth"])
    monkeypatch.setattr(custody, "_live_cmd_sha256", lambda _: state["command"])
    monkeypatch.setattr(custody, "process_group_has_live_members", lambda _: state["alive"])
    monkeypatch.setattr(owned, "_exit_proven", lambda _: not state["alive"])
    monkeypatch.setattr(platform_layer, "kill_process_group_id", lambda pid, **_: signals.append(pid))
    monkeypatch.setattr(platform_layer, "kill_pid_tree", lambda pid, **_: signals.append(pid))
    monkeypatch.setattr(custody, "kill_process_group_id", lambda pid, **_: signals.append(pid))
    assert owned.record_ledgered_process(tmp_path, row)
    owned._stamp(tmp_path, "stop_requested_at", owned._is_stop_target)
    return row, state, signals


def _stop(root, *, expired=False):
    [entry] = owned.owned_records(root)
    return owned._stop_ledgered(root, entry, time.monotonic() + (-1 if expired else 30), set())


def _outcomes(root):
    path = root / "logs" / "supervisor.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


@pytest.mark.parametrize("dimension", ["birth", "command"])
def test_foreign_leader_is_retired_without_signals_or_a_wait(tmp_path, monkeypatch, member, dimension):
    row, state, signals = member
    state[dimension] = "another-process"
    monkeypatch.setattr(owned.time, "sleep", lambda _: pytest.fail("waiting for a stranger"))
    assert not custody._service_group_survives_leader(row)
    assert _stop(tmp_path)
    assert signals == [] and owned.owned_records(tmp_path) == []
    [event] = _outcomes(tmp_path)
    assert event["outcome"] == "not_ours" and event["record_id"] == "pid-600"
    field = "start_time" if dimension == "birth" else "cmd_sha256"
    assert event["mismatches"] == [field]
    assert event["recorded"][field] != event["measured"][field]
    assert owned.finish_unconfirmed_stops(tmp_path) == {"retried": 0, "confirmed": 0, "unconfirmed": 0}

    # The same identity now matches: retain the ordinary signal/exit path.
    state.update(birth="recorded-birth", command=row["fingerprint"]["cmd_sha256"])
    assert owned.record_ledgered_process(tmp_path, row)
    monkeypatch.setattr(platform_layer, "kill_process_group_id",
                        lambda pid, **_: (signals.append(pid), state.update(alive=False)))
    assert _stop(tmp_path)
    assert signals == [600] and owned.owned_records(tmp_path) == []
    assert _outcomes(tmp_path) == [event]


@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("pgid", [0, 600])
def test_failed_signal_or_unreadable_identity_never_expires_custody(tmp_path, monkeypatch, member, unknown, pgid):
    row, state, signals = member
    row["pgid"] = pgid
    assert owned.record_ledgered_process(tmp_path, row)
    if unknown:
        state.update(birth="", command="")
    assert custody.measured_identity_mismatch(row) == {}
    assert not _stop(tmp_path, expired=True)  # EPERM/no-op cannot prove an exit
    assert len(owned.owned_records(tmp_path)) == 1 and _outcomes(tmp_path) == []
    assert signals == ([] if unknown and not pgid else [600])
    # A missing leader and missing group do prove exit, independently of the deadline.
    state["alive"] = False
    assert _stop(tmp_path)
    assert owned.owned_records(tmp_path) == []


def test_reused_nonleader_does_not_discard_its_surviving_group(tmp_path, member):
    row, state, signals = member
    row["pgid"] = 601
    assert owned.record_ledgered_process(tmp_path, row)
    state["birth"] = "another-process"
    assert custody._service_group_survives_leader(row)
    assert not _stop(tmp_path, expired=True)
    assert signals == [601] and len(owned.owned_records(tmp_path)) == 1


def test_no_legacy_measurement_is_not_a_birth_mismatch(tmp_path, monkeypatch, member):
    row, state, _ = member
    row["pgid"] = 0
    assert owned.record_ledgered_process(tmp_path, row)
    state["birth"] = "4242.currentboot"
    monkeypatch.setattr(custody, "process_start_time_legacy", lambda _: "")
    assert custody.measured_identity_mismatch(row) == {}
    assert not _stop(tmp_path, expired=True)
    assert owned.owned_records(tmp_path)
    monkeypatch.setattr(custody, "process_start_time_legacy", lambda _: "recorded-birth")
    assert custody.measured_identity_mismatch(row) == {}
    monkeypatch.setattr(custody, "process_start_time_legacy", lambda _: "foreign-birth")
    assert custody.measured_identity_mismatch(row)["mismatches"] == ["start_time"]


@pytest.mark.parametrize("operation", ["stop", "reap", "quiesce"])
def test_refutation_must_be_recorded_before_retirement(tmp_path, monkeypatch, member, operation):
    row, state, signals = member
    assert custody.append_jsonl(custody.ledger_path(tmp_path), row)
    state["birth"] = "another-process"

    def attempt():
        if operation == "stop":
            return _stop(tmp_path)
        if operation == "reap":
            assert custody.reap_orphaned_processes(tmp_path) == []
        else:
            custody.quiesce_custodied_services(tmp_path, timeout_sec=0)

    with monkeypatch.context() as patch:
        patch.setattr(custody, "record_not_ours", lambda *_, **__: False)
        attempt()
    assert owned.owned_records(tmp_path) and custody._read_ledger(tmp_path) == [row]
    assert _outcomes(tmp_path) == [] and signals == []
    attempt()
    assert owned.owned_records(tmp_path) == [] and signals == []
    assert _outcomes(tmp_path)[0]["outcome"] == "not_ours"
    if operation != "stop":
        assert custody._read_ledger(tmp_path) == []


def test_retirement_cannot_forget_a_concurrently_registered_successor(tmp_path, monkeypatch, member):
    row, state, signals = member
    state["birth"] = "another-process"
    successor = {**row, "fingerprint": {**row["fingerprint"], "start_time": "another-process"}}
    record = custody.record_not_ours

    def record_and_register(*args, **kwargs):
        assert record(*args, **kwargs)
        assert owned.record_ledgered_process(tmp_path, successor)
        return True

    monkeypatch.setattr(custody, "record_not_ours", record_and_register)
    _stop(tmp_path)
    assert owned.owned_records(tmp_path)[0]["ledger_entry"] == successor
    assert signals == []
