"""D8 process-log spool: exhaustive reducer transitions plus the real-process case.

The reducer half is driven with no processes and no files at all — every state
and every failure path has its own named case.  The I/O half is driven against
tmp_path, and one case runs a REAL process that overruns a tiny quota so the
"terminate, then seal every accepted byte" invariant is proven end to end.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import threading

import pytest

import ouroboros.execd_spool as spool_module
from ouroboros.execd_spool import (
    QUOTA_SCOPE_HOST,
    QUOTA_SCOPE_STREAM,
    QUOTA_SCOPE_TASK,
    SPOOL_HOST_QUOTA_BYTES,
    SPOOL_MIN_SEAL_BYTES,
    SPOOL_STREAM_QUOTA_BYTES,
    SPOOL_TASK_QUOTA_BYTES,
    SPOOL_STATES,
    SPOOL_TERMINAL_FAILURE_STATES,
    SPOOL_TERMINAL_STATES,
    STATE_ACKNOWLEDGED,
    STATE_DISK_FULL,
    STATE_EXPIRED,
    STATE_HASH_FAILED,
    STATE_OPEN,
    STATE_SEALED,
    STATE_SEALING,
    STATE_STATE_CORRUPT,
    STATE_TERMINATING_ON_QUOTA,
    DeleteSegments,
    ExecdError,
    HomeAcknowledged,
    OfferBytes,
    ProcessEnded,
    ProcessLogSpool,
    QuotaGranted,
    RegisterArtifact,
    RejectBytes,
    ReleaseQuota,
    RetentionExpired,
    SealComputed,
    SealHashFailed,
    SealSpool,
    SpoolQuotaLedger,
    SpoolStream,
    SpoolWriteFailed,
    StateCorrupted,
    TerminateProcessGroup,
    WriteSegment,
    apply_spool_event,
    binding_quota_scope,
)
from ouroboros.workspace_native import _run_process

# ── plan-frozen quota values ────────────────────────────────────────────


def test_plan_quota_values_are_frozen():
    assert SPOOL_STREAM_QUOTA_BYTES == 512 * 1024 * 1024
    assert SPOOL_TASK_QUOTA_BYTES == 2 * 1024 * 1024 * 1024
    assert SPOOL_HOST_QUOTA_BYTES == 8 * 1024 * 1024 * 1024


def test_declared_state_space_matches_the_states_the_reducer_can_reach():
    """A new state must be declared and classified, not merely returned."""

    events = [
        QuotaGranted(64),
        OfferBytes(4096),
        ProcessEnded(),
        SealComputed("e" * 64, "e" * 64),
        HomeAcknowledged(),
        RetentionExpired(),
        SpoolWriteFailed("x"),
        SealHashFailed("x"),
        StateCorrupted("x"),
    ]
    reached: set[str] = set()
    frontier = [_open(stream_limit=4096)]
    seen: set[str] = set()
    while frontier:
        stream = frontier.pop()
        if stream.state in seen:
            continue
        seen.add(stream.state)
        for event in events:
            result = apply_spool_event(stream, event)
            reached.add(result.stream.state)
            frontier.append(result.stream)
    assert reached == set(SPOOL_STATES)
    assert SPOOL_TERMINAL_FAILURE_STATES < SPOOL_TERMINAL_STATES < SPOOL_STATES
    assert STATE_OPEN not in SPOOL_TERMINAL_STATES
    assert STATE_SEALED not in SPOOL_TERMINAL_STATES


# ── reducer helpers ─────────────────────────────────────────────────────


def _open(**kwargs) -> SpoolStream:
    base = {"stream": "stdout", "task_id": "task-1", "operation_id": "op-1"}
    base.update(kwargs)
    return SpoolStream(**base)


def _granted(limit: int = 1000, granted: int = 1000, scope: str = "") -> SpoolStream:
    return _open(stream_limit=limit, granted_bytes=granted, grant_scope=scope)


def _kinds(transition) -> list[type]:
    return [type(effect) for effect in transition.effects]


# ── open ────────────────────────────────────────────────────────────────


def test_open_quota_granted_full_grows_the_ceiling_with_no_effects():
    result = apply_spool_event(_open(), QuotaGranted(4096))
    assert result.stream.state == STATE_OPEN
    assert result.stream.granted_bytes == 4096
    assert result.stream.grant_scope == ""
    assert result.effects == ()


def test_open_quota_granted_partial_records_the_binding_scope():
    result = apply_spool_event(_open(), QuotaGranted(100, QUOTA_SCOPE_HOST))
    assert result.stream.granted_bytes == 100
    assert result.stream.grant_scope == QUOTA_SCOPE_HOST
    assert result.effects == ()


def test_open_offer_that_fits_writes_a_segment_and_stays_open():
    result = apply_spool_event(_granted(), OfferBytes(300))
    assert result.stream.state == STATE_OPEN
    assert result.stream.accepted_bytes == 300
    assert result.stream.segments == 1
    assert result.effects == (WriteSegment(300),)


def test_open_offer_past_the_stream_quota_accepts_the_prefix_then_terminates():
    result = apply_spool_event(_granted(limit=500, granted=500), OfferBytes(800))
    assert result.stream.state == STATE_TERMINATING_ON_QUOTA
    assert result.stream.accepted_bytes == 500
    assert result.stream.rejected_bytes == 300
    assert result.stream.quota_scope == QUOTA_SCOPE_STREAM
    assert result.effects == (
        WriteSegment(500),
        TerminateProcessGroup(QUOTA_SCOPE_STREAM),
        RejectBytes(300, QUOTA_SCOPE_STREAM),
    )


def test_open_offer_that_exactly_fills_the_quota_still_terminates():
    result = apply_spool_event(_granted(limit=500, granted=500), OfferBytes(500))
    assert result.stream.state == STATE_TERMINATING_ON_QUOTA
    assert result.stream.accepted_bytes == 500
    assert result.stream.rejected_bytes == 0
    assert result.effects == (
        WriteSegment(500),
        TerminateProcessGroup(QUOTA_SCOPE_STREAM),
    )


def test_open_offer_with_no_headroom_rejects_without_writing():
    stream = _granted(limit=1000, granted=200, scope=QUOTA_SCOPE_TASK)
    stream = apply_spool_event(stream, OfferBytes(200)).stream
    result = apply_spool_event(stream, OfferBytes(64))
    assert result.stream.state == STATE_TERMINATING_ON_QUOTA
    assert _kinds(result) == [RejectBytes]


def test_open_offer_bounded_by_the_task_grant_names_the_task_scope():
    result = apply_spool_event(
        _granted(limit=10_000, granted=400, scope=QUOTA_SCOPE_TASK), OfferBytes(900)
    )
    assert result.stream.quota_scope == QUOTA_SCOPE_TASK
    assert result.stream.terminate_reason == QUOTA_SCOPE_TASK
    assert TerminateProcessGroup(QUOTA_SCOPE_TASK) in result.effects


def test_open_offer_bounded_by_the_host_grant_names_the_host_scope():
    result = apply_spool_event(
        _granted(limit=10_000, granted=400, scope=QUOTA_SCOPE_HOST), OfferBytes(900)
    )
    assert result.stream.quota_scope == QUOTA_SCOPE_HOST


def test_open_process_ended_seals():
    result = apply_spool_event(_granted(), ProcessEnded())
    assert result.stream.state == STATE_SEALING
    assert result.effects == (SealSpool(),)


def test_open_write_failure_is_disk_full_and_terminates_the_group():
    result = apply_spool_event(_granted(), SpoolWriteFailed("ENOSPC"))
    assert result.stream.state == STATE_DISK_FULL
    assert result.stream.failure == "ENOSPC"
    assert result.stream.granted_bytes == 0
    assert _kinds(result) == [TerminateProcessGroup, DeleteSegments, ReleaseQuota]


def test_open_state_corruption_terminates_and_releases():
    result = apply_spool_event(_granted(), StateCorrupted("ledger unreadable"))
    assert result.stream.state == STATE_STATE_CORRUPT
    assert result.stream.failure == "ledger unreadable"
    assert _kinds(result) == [TerminateProcessGroup, DeleteSegments, ReleaseQuota]


@pytest.mark.parametrize(
    "event",
    [
        SealComputed("a" * 64, "a" * 64),
        SealHashFailed("x"),
        HomeAcknowledged(),
        RetentionExpired(),
    ],
)
def test_open_rejects_out_of_order_events_as_state_corrupt(event):
    result = apply_spool_event(_granted(), event)
    assert result.stream.state == STATE_STATE_CORRUPT
    assert STATE_OPEN in result.stream.failure
    assert type(event).__name__ in result.stream.failure


# ── terminating_on_quota ────────────────────────────────────────────────


def _terminating() -> SpoolStream:
    stream = apply_spool_event(_granted(limit=500, granted=500), OfferBytes(800)).stream
    assert stream.state == STATE_TERMINATING_ON_QUOTA
    return stream


def test_terminating_rejects_late_pipe_bytes_without_re_terminating():
    result = apply_spool_event(_terminating(), OfferBytes(120))
    assert result.stream.state == STATE_TERMINATING_ON_QUOTA
    assert result.stream.accepted_bytes == 500
    assert result.stream.rejected_bytes == 420
    assert result.effects == (RejectBytes(120, QUOTA_SCOPE_STREAM),)


def test_terminating_ignores_a_late_quota_grant():
    result = apply_spool_event(_terminating(), QuotaGranted(8192))
    assert result.stream == _terminating()
    assert result.effects == ()


def test_terminating_process_ended_seals_the_accepted_bytes():
    result = apply_spool_event(_terminating(), ProcessEnded())
    assert result.stream.state == STATE_SEALING
    assert result.stream.accepted_bytes == 500
    assert result.effects == (SealSpool(),)


def test_terminating_write_failure_is_disk_full():
    result = apply_spool_event(_terminating(), SpoolWriteFailed("EIO"))
    assert result.stream.state == STATE_DISK_FULL
    assert _kinds(result) == [TerminateProcessGroup, DeleteSegments, ReleaseQuota]


def test_terminating_state_corruption_is_terminal():
    result = apply_spool_event(_terminating(), StateCorrupted("torn record"))
    assert result.stream.state == STATE_STATE_CORRUPT


def test_terminating_rejects_a_seal_result_it_never_asked_for():
    result = apply_spool_event(_terminating(), SealComputed("b" * 64, "b" * 64))
    assert result.stream.state == STATE_STATE_CORRUPT
    assert STATE_TERMINATING_ON_QUOTA in result.stream.failure


# ── sealing ─────────────────────────────────────────────────────────────


def _sealing(accepted: int = 500, granted: int = 900) -> SpoolStream:
    stream = _open(stream_limit=1000, granted_bytes=granted, accepted_bytes=accepted)
    return apply_spool_event(stream, ProcessEnded()).stream


def test_sealing_seal_computed_registers_the_artifact_and_frees_the_unused_grant():
    digest = "c" * 64
    result = apply_spool_event(_sealing(), SealComputed(digest, digest))
    assert result.stream.state == STATE_SEALED
    assert result.stream.blob_id == digest
    assert result.stream.sha256 == digest
    # The ledger now holds exactly the bytes that are really on disk.
    assert result.stream.granted_bytes == 500
    assert result.effects == (RegisterArtifact(digest, digest, 500), ReleaseQuota(400))


def test_sealing_an_empty_stream_registers_no_artifact():
    result = apply_spool_event(
        _sealing(accepted=0, granted=0), SealComputed(hashlib.sha256(b"").hexdigest(), "")
    )
    assert result.stream.state == STATE_SEALED
    assert result.effects == ()


def test_sealing_hash_failure_is_terminal_and_does_not_terminate_a_dead_group():
    result = apply_spool_event(_sealing(), SealHashFailed("OSError: EIO"))
    assert result.stream.state == STATE_HASH_FAILED
    assert result.stream.failure == "OSError: EIO"
    assert _kinds(result) == [DeleteSegments, ReleaseQuota]


def test_sealing_write_failure_is_disk_full_without_a_terminate():
    result = apply_spool_event(_sealing(), SpoolWriteFailed("ENOSPC"))
    assert result.stream.state == STATE_DISK_FULL
    assert _kinds(result) == [DeleteSegments, ReleaseQuota]


def test_sealing_state_corruption_does_not_terminate_a_dead_group():
    result = apply_spool_event(_sealing(), StateCorrupted("truncated spool"))
    assert result.stream.state == STATE_STATE_CORRUPT
    assert _kinds(result) == [DeleteSegments, ReleaseQuota]


@pytest.mark.parametrize("event", [OfferBytes(10), ProcessEnded(), HomeAcknowledged()])
def test_sealing_rejects_out_of_order_events(event):
    result = apply_spool_event(_sealing(), event)
    assert result.stream.state == STATE_STATE_CORRUPT
    assert STATE_SEALING in result.stream.failure


# ── sealed / acknowledged / expired ─────────────────────────────────────


def _sealed() -> SpoolStream:
    digest = "d" * 64
    return apply_spool_event(_sealing(), SealComputed(digest, digest)).stream


def test_sealed_home_acknowledgement_drops_the_bytes_and_the_reservation():
    result = apply_spool_event(_sealed(), HomeAcknowledged())
    assert result.stream.state == STATE_ACKNOWLEDGED
    assert result.stream.granted_bytes == 0
    assert result.effects == (DeleteSegments(), ReleaseQuota(500))


def test_sealed_retention_expiry_drops_the_bytes_and_the_reservation():
    result = apply_spool_event(_sealed(), RetentionExpired())
    assert result.stream.state == STATE_EXPIRED
    assert result.effects == (DeleteSegments(), ReleaseQuota(500))


def test_sealed_state_corruption_is_terminal():
    result = apply_spool_event(_sealed(), StateCorrupted("blob vanished"))
    assert result.stream.state == STATE_STATE_CORRUPT


@pytest.mark.parametrize("event", [OfferBytes(1), ProcessEnded(), SealHashFailed("x")])
def test_sealed_rejects_out_of_order_events(event):
    result = apply_spool_event(_sealed(), event)
    assert result.stream.state == STATE_STATE_CORRUPT
    assert STATE_SEALED in result.stream.failure


def test_acknowledged_acknowledgement_is_idempotent_for_home_retries():
    acknowledged = apply_spool_event(_sealed(), HomeAcknowledged()).stream
    result = apply_spool_event(acknowledged, HomeAcknowledged())
    assert result.stream == acknowledged
    assert result.effects == ()


def test_expired_expiry_is_idempotent_for_retention_resweeps():
    expired = apply_spool_event(_sealed(), RetentionExpired()).stream
    result = apply_spool_event(expired, RetentionExpired())
    assert result.stream == expired
    assert result.effects == ()


def test_acknowledged_cannot_be_expired():
    acknowledged = apply_spool_event(_sealed(), HomeAcknowledged()).stream
    result = apply_spool_event(acknowledged, RetentionExpired())
    assert result.stream.state == STATE_STATE_CORRUPT
    assert STATE_ACKNOWLEDGED in result.stream.failure


def test_expired_cannot_be_acknowledged():
    expired = apply_spool_event(_sealed(), RetentionExpired()).stream
    result = apply_spool_event(expired, HomeAcknowledged())
    assert result.stream.state == STATE_STATE_CORRUPT
    assert STATE_EXPIRED in result.stream.failure


# ── terminal failure states ─────────────────────────────────────────────


@pytest.mark.parametrize("failure_state", sorted(SPOOL_TERMINAL_FAILURE_STATES))
@pytest.mark.parametrize(
    "event", [OfferBytes(5), ProcessEnded(), HomeAcknowledged(), RetentionExpired()]
)
def test_terminal_failure_states_accept_no_further_transition(failure_state, event):
    result = apply_spool_event(_open(state=failure_state), event)
    assert result.stream.state == STATE_STATE_CORRUPT


def test_state_corrupt_absorbs_repeated_corruption_reports():
    corrupt = _open(state=STATE_STATE_CORRUPT, failure="first")
    result = apply_spool_event(corrupt, StateCorrupted("second"))
    assert result.stream == corrupt
    assert result.effects == ()


def test_binding_quota_scope_prefers_the_stream_limit():
    stream = _granted(limit=100, granted=100, scope=QUOTA_SCOPE_HOST)
    assert binding_quota_scope(stream, 100) == QUOTA_SCOPE_STREAM
    assert binding_quota_scope(stream, 99) == ""


# ── quota ledger ────────────────────────────────────────────────────────


def test_ledger_reserves_and_releases_under_the_limits(tmp_path):
    ledger = SpoolQuotaLedger(tmp_path, task_limit=1000, host_limit=4000)
    assert ledger.reserve("t1", 400) == (400, "")
    assert ledger.usage()["host_bytes"] == 400
    ledger.release("t1", 400)
    assert ledger.usage()["host_bytes"] == 0
    assert ledger.usage()["tasks"] == {}


def test_ledger_task_limit_binds_before_the_host_limit(tmp_path):
    ledger = SpoolQuotaLedger(tmp_path, task_limit=500, host_limit=4000)
    assert ledger.reserve("t1", 900) == (500, QUOTA_SCOPE_TASK)
    assert ledger.reserve("t1", 100) == (0, QUOTA_SCOPE_TASK)
    assert ledger.reserve("t2", 100) == (100, "")


def test_ledger_host_limit_binds_across_tasks(tmp_path):
    ledger = SpoolQuotaLedger(tmp_path, task_limit=4000, host_limit=600)
    assert ledger.reserve("t1", 500) == (500, "")
    assert ledger.reserve("t2", 500) == (100, QUOTA_SCOPE_HOST)


def test_ledger_release_never_goes_negative(tmp_path):
    ledger = SpoolQuotaLedger(tmp_path, task_limit=1000, host_limit=1000)
    ledger.reserve("t1", 100)
    ledger.release("t1", 900)
    assert ledger.usage()["host_bytes"] == 0


def test_ledger_rejects_a_corrupt_record(tmp_path):
    ledger = SpoolQuotaLedger(tmp_path, task_limit=1000, host_limit=1000)
    ledger.reserve("t1", 10)
    ledger.path.write_text(json.dumps({"host_bytes": 5}), encoding="utf-8")
    with pytest.raises(ExecdError) as excinfo:
        ledger.reserve("t1", 10)
    assert excinfo.value.code == "spool_quota_corrupt"


# ── sink I/O ────────────────────────────────────────────────────────────


def _spool(root: pathlib.Path, **kwargs) -> ProcessLogSpool:
    options = {
        "stream_limit": 1 << 20,
        "task_limit": 1 << 20,
        "host_limit": 1 << 20,
        "grant_bytes": 4096,
        "min_seal_bytes": 0,
    }
    options.update(kwargs)
    return ProcessLogSpool(root, **options)


def test_sink_seals_the_exact_accepted_bytes_under_their_own_digest(tmp_path):
    spool = _spool(tmp_path)
    sink = spool.bind(task_id="t", operation_id="op").open_stream(stream="stdout")
    payload = b"line\n" * 2000
    assert sink.write(payload) == len(payload)
    row = sink.seal()
    assert sink.record.state == STATE_SEALED
    assert row is not None
    digest = hashlib.sha256(payload).hexdigest()
    assert row["blob_id"] == digest
    assert row["sha256"] == digest
    assert row["size"] == len(payload)
    assert row["full_log"] is True
    assert row["truncated"] is False
    # No `fetchable`/`spool_state`: Home never read them and materializes
    # eagerly, so advertising an on-demand action would be a false capability.
    assert "fetchable" not in row
    assert "spool_state" not in row
    assert spool.sealed_path(digest).read_bytes() == payload
    assert spool.read_sealed(digest, max_bytes=len(payload)) == payload
    # Only the real on-disk bytes remain reserved after sealing.
    assert spool.ledger.usage()["host_bytes"] == len(payload)


def test_sink_acknowledgement_deletes_the_blob_and_frees_the_quota(tmp_path):
    spool = _spool(tmp_path)
    sink = spool.bind(task_id="t", operation_id="op").open_stream(stream="stdout")
    sink.write(b"z" * 5000)
    row = sink.seal()
    assert row is not None
    sink.acknowledge()
    assert sink.record.state == STATE_ACKNOWLEDGED
    assert not spool.sealed_path(row["blob_id"]).exists()
    assert spool.ledger.usage()["host_bytes"] == 0


def test_sink_below_the_inline_preview_bound_keeps_no_spool_bytes(tmp_path):
    spool = _spool(tmp_path, min_seal_bytes=4096)
    sink = spool.bind(task_id="t", operation_id="op").open_stream(stream="stderr")
    sink.write(b"tiny output\n")
    assert sink.seal() is None
    assert sink.record.state == STATE_EXPIRED
    assert spool.ledger.usage()["host_bytes"] == 0
    assert list(spool.sealed_root.iterdir()) == []


def test_sink_quota_hit_terminates_the_group_and_keeps_every_accepted_byte(tmp_path):
    spool = _spool(tmp_path, stream_limit=8192, grant_bytes=4096)
    reasons: list[str] = []
    sink = spool.open_stream(
        task_id="t", operation_id="op", stream="stdout", terminate=reasons.append
    )
    accepted = sum(sink.write(b"x" * 4096) for _ in range(4))
    assert accepted == 8192
    assert sink.record.state == STATE_TERMINATING_ON_QUOTA
    assert sink.record.rejected_bytes == 8192
    assert reasons == [QUOTA_SCOPE_STREAM]
    row = sink.seal()
    assert row is not None
    assert row["size"] == 8192
    assert row["truncated"] is True
    assert row["full_log"] is False
    assert spool.sealed_path(row["blob_id"]).read_bytes() == b"x" * 8192
    assert sink.trace()["quota_scope"] == QUOTA_SCOPE_STREAM


def test_sink_write_after_seal_is_a_closed_door_not_a_corruption(tmp_path):
    spool = _spool(tmp_path)
    sink = spool.bind(task_id="t", operation_id="op").open_stream(stream="stdout")
    sink.write(b"a" * 100)
    sink.seal()
    assert sink.write(b"late") == 0
    assert sink.record.state == STATE_SEALED


def test_sink_write_failure_becomes_disk_full_and_frees_the_quota(tmp_path, monkeypatch):
    spool = _spool(tmp_path)
    sink = spool.bind(task_id="t", operation_id="op").open_stream(stream="stdout")

    def _boom(_data: bytes) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(sink, "_append", _boom)
    assert sink.write(b"y" * 100) == 100
    assert sink.record.state == STATE_DISK_FULL
    assert "No space left" in sink.record.failure
    assert spool.ledger.usage()["host_bytes"] == 0


def test_sink_unusable_quota_ledger_is_state_corrupt(tmp_path):
    spool = _spool(tmp_path)
    sink = spool.bind(task_id="t", operation_id="op").open_stream(stream="stdout")
    spool.ledger.path.write_bytes(b"{not json")
    assert sink.write(b"q" * 10) == 0
    assert sink.record.state == STATE_STATE_CORRUPT
    assert "quota ledger unusable" in sink.record.failure


def test_sealed_path_rejects_a_non_digest_blob_id(tmp_path):
    spool = _spool(tmp_path)
    with pytest.raises(ExecdError) as excinfo:
        spool.sealed_path("../escape")
    assert excinfo.value.code == "spool_blob_invalid"


def test_read_sealed_refuses_to_exceed_the_requested_bound(tmp_path):
    spool = _spool(tmp_path)
    sink = spool.bind(task_id="t", operation_id="op").open_stream(stream="stdout")
    sink.write(b"w" * 500)
    row = sink.seal()
    assert row is not None
    with pytest.raises(ExecdError) as excinfo:
        spool.read_sealed(row["blob_id"], max_bytes=100)
    assert excinfo.value.code == "spool_blob_too_large"


def test_concurrent_stream_writers_cannot_oversubscribe_the_host_quota(tmp_path):
    host_limit = 96 * 1024
    spool = _spool(
        tmp_path,
        stream_limit=1 << 20,
        task_limit=1 << 20,
        host_limit=host_limit,
        grant_bytes=16 * 1024,
    )
    bound = spool.bind(task_id="t", operation_id="op")
    sinks = [bound.open_stream(stream=name) for name in ("stdout", "stderr")]
    accepted = [0, 0]

    def _pump(index: int) -> None:
        accepted[index] = sum(sinks[index].write(b"z" * 8192) for _ in range(64))

    threads = [threading.Thread(target=_pump, args=(index,)) for index in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert sum(accepted) == host_limit
    assert spool.ledger.usage()["host_bytes"] == host_limit
    assert {sink.record.state for sink in sinks} == {STATE_TERMINATING_ON_QUOTA}
    assert QUOTA_SCOPE_HOST in {sink.record.quota_scope for sink in sinks}


# ── real-process integration through the native kernel ──────────────────


class _FakeControl:
    """Minimal NativeExecutionControl carrying an operation-bound spool."""

    def __init__(self, process_spool: object) -> None:
        self.process_spool = process_spool
        self.registered: list[int] = []

    def cancelled(self) -> bool:
        return False

    def register_process(self, *, pgid: int, **_kwargs: object) -> None:
        self.registered.append(pgid)

    def release_process(self, *, pgid: int, **_kwargs: object) -> None:
        return None

    def recover_service(self, **_kwargs: object) -> None:
        return None

    def stop_service(self, **_kwargs: object) -> bool:
        return False


_FLOOD_SOURCE = (
    "import sys, time\n"
    "block = b'x' * 65536\n"
    "deadline = time.time() + 10\n"
    "while time.time() < deadline:\n"
    "    sys.stdout.buffer.write(block)\n"
)


# Serial: this launches a REAL flooding subprocess and asserts it outran the quota
# (`rejected_bytes > 0`, `returncode < 0`). Under a loaded `-n auto` pass the flooder
# gets less CPU relative to the spool reader, the terminate lands before any byte is
# rejected, and the assertion fails on scheduling rather than on behaviour — observed
# exactly once in a full parallel run. The source scan in
# tests/test_serial_lane_contract.py cannot see this: the subprocess is spawned by the
# code under test, not by a `Popen` in the test body.
@pytest.mark.serial
def test_real_process_overrunning_the_spool_quota_is_terminated_and_fully_sealed(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    stream_limit = 128 * 1024
    spool = ProcessLogSpool(
        tmp_path / "spool",
        stream_limit=stream_limit,
        task_limit=8 << 20,
        host_limit=8 << 20,
        grant_bytes=64 * 1024,
    )
    result = _run_process(
        workspace,
        {"cwd": ".", "timeout_sec": 30},
        cmd=[sys.executable, "-c", _FLOOD_SOURCE],
        control=None,
        process_spool=spool.bind(task_id="task-flood", operation_id="op-flood"),
        backend="local_native",
    )

    process = result.envelope.process
    assert process is not None
    # The quota reached => the process GROUP was signalled, not the bytes dropped.
    assert process.returncode < 0
    facts = process.backend_trace["output_capture"]["stdout"]["spool"]
    assert facts["state"] == STATE_SEALED
    assert facts["quota_scope"] == QUOTA_SCOPE_STREAM
    assert facts["accepted_bytes"] == stream_limit
    assert facts["rejected_bytes"] > 0
    assert facts["segments"] >= 2
    assert "PROCESS_LOG_QUOTA" in result.envelope.text

    rows = [row for row in result.envelope.artifacts if row["name"] == "stdout.txt"]
    assert len(rows) == 1
    row = rows[0]
    assert row["blob_id"] == facts["blob_id"]
    assert row["size"] == stream_limit
    assert "fetchable" not in row
    assert "spool_state" not in row
    # The blob stays on the host and is NOT inlined in this envelope; Home
    # materializes it eagerly at import time, not through a spool-state action.
    assert row["blob_id"] not in result.blobs
    sealed = spool.sealed_path(row["blob_id"]).read_bytes()
    assert sealed == b"x" * stream_limit
    assert hashlib.sha256(sealed).hexdigest() == row["sha256"]
    assert process.backend_trace["output_capture"]["stdout"]["full_log_available"] is False


def test_native_kernel_reads_the_spool_off_the_control_object(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    spool = ProcessLogSpool(tmp_path / "spool", min_seal_bytes=0)
    control = _FakeControl(spool.bind(task_id="task-1", operation_id="op-1"))
    result = _run_process(
        workspace,
        {"cwd": ".", "timeout_sec": 30},
        cmd=[sys.executable, "-c", "print('hello spool')"],
        control=control,
        backend="local_native",
    )
    facts = result.envelope.process.backend_trace["output_capture"]["stdout"]["spool"]
    assert facts["state"] == STATE_SEALED
    assert facts["accepted_bytes"] == len(b"hello spool\n")
    assert spool.sealed_path(facts["blob_id"]).read_bytes() == b"hello spool\n"
    assert control.registered


def test_without_a_spool_the_capture_behavior_is_unchanged(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    result = _run_process(
        workspace,
        {"cwd": ".", "timeout_sec": 30},
        cmd=[sys.executable, "-c", "print('plain')"],
        control=None,
        backend="local_native",
    )
    capture = result.envelope.process.backend_trace["output_capture"]["stdout"]
    assert "spool" not in capture
    assert capture["full_log_available"] is True
    assert result.envelope.artifacts == ()
    assert result.blobs == {}


# ── D8 retention: the quota has to come BACK ─────────────────────────────
#
# The sealed log was written, quotad and never released. `SpoolStreamSink.
# acknowledge`/`expire` are the only producers of the release, and they live on a
# per-operation object that dies with the operation — so a task that ran two large
# commands left its whole reservation behind for good, and the HOST quota (8 GiB) was
# a one-way ratchet: it fills once, and then EVERY later remote process on that host
# refuses to spool, with no single event to blame. Materializing a sealed blob on
# Home's demand is still a deferred phase, by owner decision; freeing the quota is not.


def _seal_one(spool: ProcessLogSpool, *, task_id: str, payload: bytes, stream: str = "stdout"):
    sink = spool.open_stream(task_id=task_id, operation_id=f"op-{stream}", stream=stream)
    sink.write(payload)
    return sink.seal()


def _big(marker: bytes) -> bytes:
    """Past `min_seal_bytes`, so `seal()` does NOT expire it as fully-inline."""

    return marker + b"x" * (SPOOL_MIN_SEAL_BYTES + 64)


def test_a_sealed_log_holds_quota_until_its_task_is_terminal(tmp_path):
    spool = ProcessLogSpool(tmp_path / "logs")
    row = _seal_one(spool, task_id="task-1", payload=_big(b"a"))
    assert row is not None
    blob = spool.sealed_path(row["blob_id"])
    assert blob.is_file()
    held = spool.ledger.usage()["host_bytes"]
    assert held > 0, "a sealed log that reserved nothing would make this test vacuous"

    freed = spool.release_task("task-1")

    assert freed["quota_released"] == held
    assert freed["blobs_removed"] == 1
    assert not blob.exists()
    assert spool.ledger.usage() == {
        "_schema_version": 1, "host_bytes": 0, "tasks": {},
    }


def test_the_host_quota_fills_and_then_RECOVERS_at_the_terminal(tmp_path):
    """The failure mode itself, in miniature: fill the host limit, then free it.

    A limit small enough to reach is the only honest way to test a ratchet — with the
    real 8 GiB the test would assert arithmetic instead of behaviour.
    """

    host_limit = SPOOL_MIN_SEAL_BYTES * 4
    spool = ProcessLogSpool(
        tmp_path / "logs", host_limit=host_limit, grant_bytes=SPOOL_MIN_SEAL_BYTES,
    )
    for index in range(3):
        _seal_one(spool, task_id="task-1", payload=_big(bytes([65 + index])), stream=f"s{index}")

    # The ratchet, stated as the thing that actually breaks: a LATER task can no longer
    # get the host reservation it needs, and the ledger names HOST as the limit that
    # bound. (A refusal rather than an exact byte total, because a stream that reaches
    # its ceiling hands the unused part of its grant back — the total is arithmetic
    # about grant sizes, not the property under test.)
    partial, scope = spool.ledger.reserve("task-2", host_limit)
    assert scope == QUOTA_SCOPE_HOST and partial < host_limit
    spool.ledger.release("task-2", partial)

    spool.release_task("task-1")

    assert spool.ledger.reserve("task-2", host_limit) == (host_limit, "")


def test_a_blob_two_tasks_share_survives_the_first_terminal(tmp_path):
    """Sealed blobs are CONTENT-addressed, so retention needs owners, not a walk.

    Two tasks whose stdout was byte-identical hash to ONE file. Deleting "task-1's
    blobs" by listing the directory would take task-2's evidence with it — which is
    why the index carries an owner SET and the file is unlinked only when it empties.
    """

    spool = ProcessLogSpool(tmp_path / "logs")
    payload = _big(b"shared")
    first = _seal_one(spool, task_id="task-1", payload=payload)
    second = _seal_one(spool, task_id="task-2", payload=payload)
    assert first is not None and second is not None
    assert first["blob_id"] == second["blob_id"], "the premise of this test is dedup"
    blob = spool.sealed_path(first["blob_id"])

    freed = spool.release_task("task-1")

    assert freed["blobs_removed"] == 0
    assert blob.is_file(), "task-2's evidence must survive task-1's terminal"
    assert spool.release_task("task-2")["blobs_removed"] == 1
    assert not blob.exists()


def test_retention_expires_a_blob_whose_home_never_came_back(tmp_path):
    """The AGE backstop, for the cases the terminal half cannot see.

    A Home that died mid-task sends no cancel, so nothing declares the task terminal.
    Without the age sweep the host quota is still a one-way ratchet — just a slower one.
    """

    spool = ProcessLogSpool(tmp_path / "logs")
    row = _seal_one(spool, task_id="task-1", payload=_big(b"orphan"))
    assert row is not None
    blob = spool.sealed_path(row["blob_id"])
    held = spool.ledger.usage()["host_bytes"]

    # Inside the window: nothing is touched, because a live task's evidence is not
    # garbage and a sweep that cannot tell the difference is worse than none.
    assert spool.expire_retained() == {"blobs_removed": 0, "quota_released": 0}
    assert blob.is_file()

    aged = spool.expire_retained(ttl_ms=0)

    assert aged == {"blobs_removed": 1, "quota_released": held}
    assert not blob.exists()
    assert spool.ledger.usage()["host_bytes"] == 0


def test_a_corrupt_retention_index_is_repaired_rather_than_fatal(tmp_path):
    """The index holds no evidence of its own, so it must never block a release.

    A spool that refuses to free anything because its bookkeeping is unreadable is
    exactly the failure retention exists to prevent, so a corrupt index is rebuilt
    empty and the ledger row — which IS durable — is still handed back.
    """

    spool = ProcessLogSpool(tmp_path / "logs")
    _seal_one(spool, task_id="task-1", payload=_big(b"c"))
    held = spool.ledger.usage()["host_bytes"]
    spool.retention_path.write_text("{not json", encoding="utf-8")

    freed = spool.release_task("task-1")

    assert freed["quota_released"] == held
    assert spool.ledger.usage()["host_bytes"] == 0


def test_release_task_is_idempotent_and_never_goes_negative(tmp_path):
    spool = ProcessLogSpool(tmp_path / "logs")
    _seal_one(spool, task_id="task-1", payload=_big(b"d"))
    spool.release_task("task-1")
    assert spool.release_task("task-1") == {"blobs_removed": 0, "quota_released": 0}
    assert spool.release_task("never-existed") == {"blobs_removed": 0, "quota_released": 0}
    assert spool.ledger.usage()["host_bytes"] == 0


def test_expiry_frees_only_the_AGED_blob_of_a_long_running_task(tmp_path):
    """Per-blob, not per task — the direction a quota must never round wrongly.

    A task that outlives the retention window can hold one aged blob and one fresh one.
    Dropping the whole task ROW for the aged one would leave the fresh one's bytes
    unaccounted, so the host would look EMPTIER than it is — the one direction a quota
    error must not take, because it lets the next reservation oversubscribe the disk.
    """

    spool = ProcessLogSpool(tmp_path / "logs")
    old_row = _seal_one(spool, task_id="task-1", payload=_big(b"old"), stream="s-old")
    new_row = _seal_one(spool, task_id="task-1", payload=_big(b"new"), stream="s-new")
    assert old_row is not None and new_row is not None
    total = spool.ledger.usage()["host_bytes"]

    # Age ONLY the first blob, by rewriting its seal stamp far into the past.
    index = json.loads(spool.retention_path.read_text(encoding="utf-8"))
    index["blobs"][old_row["blob_id"]]["sealed_at_ms"] = 1
    spool.retention_path.write_text(json.dumps(index), encoding="utf-8")

    swept = spool.expire_retained()

    assert swept["blobs_removed"] == 1
    assert swept["quota_released"] == old_row["size"]
    assert not spool.sealed_path(old_row["blob_id"]).exists()
    assert spool.sealed_path(new_row["blob_id"]).is_file()
    # The fresh blob is STILL accounted for, to the byte.
    assert spool.ledger.usage()["host_bytes"] == total - old_row["size"]
    assert spool.ledger.usage()["tasks"] == {"task-1": total - old_row["size"]}


# ── D8 retention: reserved and released must be the SAME number ──────────
#
# The paid review's C9: a retention row held ONE `size` and a SET of owners, so the
# sweep could only ever release `size` once per owner id. Every case below is a place
# where the count of reservations and the count of releases diverged, and each one was
# recoverable only through `release_task` — the backstop that exists to cover what the
# accounting cannot see, not to be the accounting.


def test_two_streams_of_ONE_task_that_hash_alike_free_BOTH_reservations(tmp_path):
    """The exact shape the old row could not represent.

    stdout and stderr of one task carrying identical bytes hash to ONE blob. Each sink
    reserved its own `size`; the row recorded one `size` and one owner id, so the age
    sweep handed back half the reservation and the other half survived until the task
    went terminal. Per-owner byte counts make the two numbers the same number.
    """

    spool = ProcessLogSpool(tmp_path / "logs")
    payload = _big(b"identical")
    first = _seal_one(spool, task_id="task-1", payload=payload, stream="stdout")
    second = _seal_one(spool, task_id="task-1", payload=payload, stream="stderr")
    assert first is not None and second is not None
    assert first["blob_id"] == second["blob_id"], "the premise of this test is dedup"
    held = spool.ledger.usage()["host_bytes"]
    assert held == first["size"] + second["size"], "two sinks, two reservations"

    swept = spool.expire_retained(ttl_ms=0)

    assert swept == {"blobs_removed": 1, "quota_released": held}
    assert spool.ledger.usage() == {
        "_schema_version": 1, "host_bytes": 0, "tasks": {},
    }


def test_a_fully_inline_log_leaves_no_row_for_the_sweep_to_free_twice(tmp_path):
    """A log small enough to ride inline in the preview is dropped at `seal()`.

    That path released the quota AND left its retention row behind, so the age sweep
    later released the same bytes a second time — out of whatever the SAME task had
    reserved since. `release` clamps, so nothing goes negative; what it does instead is
    make the host look emptier than it is, which is the one direction that lets the next
    reservation oversubscribe the disk.
    """

    spool = ProcessLogSpool(tmp_path / "logs")
    assert _seal_one(spool, task_id="task-1", payload=b"tiny") is None
    assert json.loads(spool.retention_path.read_text(encoding="utf-8"))["blobs"] == {}
    assert list(spool.sealed_root.iterdir()) == []
    assert spool.ledger.usage()["host_bytes"] == 0

    live, _scope = spool.ledger.reserve("task-1", 4096)
    assert live == 4096

    assert spool.expire_retained(ttl_ms=0) == {"blobs_removed": 0, "quota_released": 0}
    assert spool.ledger.usage()["tasks"] == {"task-1": live}


def test_home_acknowledging_one_stream_keeps_the_blob_another_task_shares(tmp_path):
    """A sink must not unlink its own sealed path: the path is the CONTENT digest.

    `acknowledge` deleted `self._sealed_path` directly, which for a byte-identical
    stream of another task is that task's only evidence. Withdrawing the claim is what
    unlinks the file now, and only when the claim was the last one.
    """

    spool = ProcessLogSpool(tmp_path / "logs")
    payload = _big(b"shared")
    sink = spool.open_stream(task_id="task-1", operation_id="op-1", stream="stdout")
    sink.write(payload)
    first = sink.seal()
    second = _seal_one(spool, task_id="task-2", payload=payload)
    assert first is not None and second is not None
    assert first["blob_id"] == second["blob_id"], "the premise of this test is dedup"
    blob = spool.sealed_path(first["blob_id"])

    sink.acknowledge()

    assert blob.is_file(), "task-2's evidence must survive task-1's acknowledgement"
    assert spool.ledger.usage()["tasks"] == {"task-2": second["size"]}
    assert spool.expire_retained(ttl_ms=0) == {
        "blobs_removed": 1, "quota_released": second["size"],
    }
    assert not blob.exists()


# ── D8 retention: the durable row and the file, in ONE order ─────────────
#
# The paid review's C10. The index write was committed on the way out of `_retention`
# and the blobs unlinked after it, so a crash in between left a file the index no longer
# named — and `_unlink_blobs` removes only ids it is handed, so no later sweep could
# reach it. The docstring's "the next sweep fixes it" was false. Both halves are fixed:
# the file goes first, and an unindexed file is now genuinely swept.


def test_the_blob_goes_before_the_index_row_so_a_crash_orphans_nothing(tmp_path, monkeypatch):
    spool = ProcessLogSpool(tmp_path / "logs")
    row = _seal_one(spool, task_id="task-1", payload=_big(b"crash"))
    assert row is not None
    blob = spool.sealed_path(row["blob_id"])

    def _power_cut(path, payload):
        raise OSError("the index write never reached the platter")

    monkeypatch.setattr(spool_module, "durable_json", _power_cut)
    with pytest.raises(OSError):
        spool.release_task("task-1")
    monkeypatch.undo()

    # The FILE is gone and the ROW survives — the recoverable direction. A row naming a
    # missing file costs one wasted sweep; a file naming no row is unreachable for good.
    assert not blob.exists()
    assert json.loads(spool.retention_path.read_text(encoding="utf-8"))["blobs"], (
        "the surviving row is the whole point: it is what still owes the quota back"
    )
    assert spool.expire_retained(ttl_ms=0) == {
        "blobs_removed": 1, "quota_released": row["size"],
    }
    assert spool.ledger.usage()["host_bytes"] == 0


def test_an_orphan_blob_no_row_names_is_really_swept(tmp_path):
    """The promise the old docstring made and could not keep.

    A directory walk must never decide to delete an INDEXED blob — the filename is the
    content digest and says nothing about who owns it. A file the index does not name at
    all has no owner to protect, so the walk is safe exactly there, and that is the only
    thing in this module that can see such a file.
    """

    spool = ProcessLogSpool(tmp_path / "logs")
    orphan = spool.sealed_path("a" * 64)
    orphan.write_bytes(b"evidence nothing owns")

    # Fresh: never race a sink that has just published a blob.
    assert spool.expire_retained() == {"blobs_removed": 0, "quota_released": 0}
    assert orphan.is_file()

    assert spool.expire_retained(ttl_ms=0) == {"blobs_removed": 1, "quota_released": 0}
    assert not orphan.exists()

    # A file whose name is not a content digest was not written by this spool.
    stranger = spool.sealed_root / "notes.log"
    stranger.write_bytes(b"")
    assert spool.expire_retained(ttl_ms=0)["blobs_removed"] == 0
    assert stranger.is_file()


def test_the_sweep_leaves_an_indexed_blob_alone_even_when_it_is_aged(tmp_path):
    """The walk's safety condition, stated as a test rather than as a comment."""

    spool = ProcessLogSpool(tmp_path / "logs")
    row = _seal_one(spool, task_id="task-1", payload=_big(b"owned"))
    assert row is not None
    index = json.loads(spool.retention_path.read_text(encoding="utf-8"))
    assert row["blob_id"] in index["blobs"]

    # Inside the window: the row protects the file even though the FILE is old enough.
    assert spool.expire_retained(now_ms=0) == {"blobs_removed": 0, "quota_released": 0}
    assert spool.sealed_path(row["blob_id"]).is_file()
    assert spool.ledger.usage()["host_bytes"] == row["size"]
