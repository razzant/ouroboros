"""Only observed publication probes/waits spend the publication budget, not other contention."""
import asyncio
import contextlib
from types import SimpleNamespace

import pytest

from ouroboros import _usage_wait, config, model_wait, queue_observation, store, usage_accounting as ua
from ouroboros.llm_attempt import PhysicalDispatchInterrupted
from ouroboros.usage_ledger import ControlAuthorityUnavailable, UsageLockUnavailable
from tests.test_admission_contention import database_reader, publish, root as root

pytestmark = pytest.mark.serial


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    owner = SimpleNamespace(task={}, control_reason=lambda: "")
    monkeypatch.setattr(_usage_wait, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(queue_observation, "_monotonic", lambda: now[0])
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: owner)
    monkeypatch.setattr(model_wait, "dispatch_deadline_remaining_sec", lambda: None)
    monkeypatch.setattr(_usage_wait, "_check", lambda: None)
    monkeypatch.setattr(_usage_wait, "_hold", lambda *_args: None)
    monkeypatch.setattr(queue_observation, "DISPATCH_PUBLICATION_WAIT_SEC", 2.0)
    monkeypatch.setattr(queue_observation, "DISPATCH_PUBLICATION_POLL_SEC", 0.5)
    return now, owner


def _sleep(monkeypatch, callback):
    monkeypatch.setattr(queue_observation, "_sleep", callback)

    async def sleeping(seconds):
        callback(seconds)

    monkeypatch.setattr(_usage_wait.asyncio, "sleep", sleeping)


def _retry(mode, callback):
    if mode == "async":
        return asyncio.run(_usage_wait.retry_presend_async(callback, check=False))
    return _usage_wait.retry_presend(callback, check=False)


def _publication_gap(root, monkeypatch, now, probe_sec):
    (root / "state").mkdir(exist_ok=True)
    publish(root)
    (root / "state/queue.alive").unlink()
    read = queue_observation.load_queue_document

    def timed_read(*args, **kwargs):
        now[0] += probe_sec  # Cost belongs to the actual publication reader.
        return read(*args, **kwargs)

    monkeypatch.setattr(queue_observation, "load_queue_document", timed_read)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_real_store_contention_between_publication_gaps_does_not_spend_their_budget(root, monkeypatch, clock, mode):
    now, _owner = clock
    _publication_gap(root, monkeypatch, now, 0.2)
    probes, waits, busy = [], [], []
    with database_reader(root, exclusive=True) as release:
        def probe():
            probes.append(len(probes) + 1)
            if len(probes) == 2:
                now[0] += 10.0  # A slow acquisition failed for storage, not missing publication.
                try:
                    with store.open_store(root, write=False, wait_sec=0):
                        pytest.fail("the independent SQLite writer was not exclusive")
                except store.StoreBusy as error:
                    assert error.reason == "contention"
                    busy.append(error)
                    raise
            return queue_observation.dispatch_probe(root)

        def sleep(seconds):
            waits.append(seconds)
            now[0] += seconds
            if len(waits) == 2:
                release()
            elif len(waits) == 3:
                publish(root, generation="recovered")

        _sleep(monkeypatch, sleep)
        assert _retry(mode, probe)["generation"] == "recovered"
    assert probes == [1, 2, 3, 4] and len(busy) == 1
    assert waits == [0.5, _usage_wait.ACQUISITION_SLICE_SEC, 0.5]


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_intervening_contention_does_not_renew_spent_publication_budget(tmp_path, monkeypatch, clock, mode):
    now, _owner = clock
    _publication_gap(tmp_path, monkeypatch, now, 0.6)
    calls, sleeps = [], []

    def probe():
        calls.append(len(calls) + 1)
        if len(calls) == 2:
            now[0] += 10
            raise store.StoreBusy("busy", reason="contention")
        return queue_observation.dispatch_probe(tmp_path)

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    _sleep(monkeypatch, sleep)
    with pytest.raises(ControlAuthorityUnavailable) as caught:
        _retry(mode, probe)
    assert caught.value.cause == "dispatch_publication_unconfirmed"
    assert calls == [1, 2, 3, 4]
    assert sleeps == pytest.approx([0.5, _usage_wait.ACQUISITION_SLICE_SEC, 0.3])


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("slow", ["probe", "sleep"])
def test_real_elapsed_publication_work_keeps_the_existing_bound(tmp_path, monkeypatch, clock, mode, slow):
    now, _owner = clock
    _publication_gap(tmp_path, monkeypatch, now, 1.1 if slow == "probe" else 0.1)
    calls, sleeps = [], []

    def probe():
        calls.append(True)
        return queue_observation.dispatch_probe(tmp_path)

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += 2.1 if slow == "sleep" else seconds

    _sleep(monkeypatch, sleep)
    with pytest.raises(ControlAuthorityUnavailable) as caught:
        _retry(mode, probe)
    assert caught.value.cause == "dispatch_publication_unconfirmed"
    assert len(calls) == 2 and sleeps == [0.5], "actual probe/sleep elapsed must count, not only requested delays"


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("bound", ["interactive", "unowned"])
def test_other_real_elapsed_bounds_still_apply_during_storage_contention(monkeypatch, clock, mode, bound):
    now, owner = clock
    if bound == "interactive":
        owner.task["_is_direct_chat"] = True
        monkeypatch.setattr(config, "get_task_idle_timeout_sec", lambda: 2.0)
    else:
        monkeypatch.setattr(model_wait, "current_model_wait", lambda: None)
        monkeypatch.setattr(_usage_wait, "USAGE_LOCK_TIMEOUT_SEC", 2.0)
    error = store.StoreBusy("same storage contention", reason="contention")

    def probe():
        now[0] += 3
        raise error

    _sleep(monkeypatch, lambda _: pytest.fail("the existing elapsed bound must end this wait"))
    if bound == "interactive":
        with pytest.raises(PhysicalDispatchInterrupted, match="accounting_wait_expired"):
            _retry(mode, probe)
    else:
        with pytest.raises(store.StoreBusy) as caught:
            _retry(mode, probe)
        assert caught.value is error


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("control", ["stop", "deadline"])
def test_controls_win_between_publication_and_busy_retries(monkeypatch, clock, mode, control):
    now, _owner = clock
    calls, sleeps = [], []

    def check():
        if len(sleeps) >= 2:
            raise PhysicalDispatchInterrupted(control)

    def probe():
        calls.append(True)
        if len(calls) == 1:
            raise queue_observation.DispatchPublicationPending("publication in flight")
        raise store.StoreBusy("storage busy", reason="contention")

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(_usage_wait, "_check", check)
    _sleep(monkeypatch, sleep)
    with pytest.raises(PhysicalDispatchInterrupted, match=control):
        _retry(mode, probe)
    assert len(calls) == 2, "control must prevent another unsent callback invocation"


@pytest.mark.parametrize("storage_elapsed", [0.0, 10.0], ids=["no-contention", "inner-usage-contention"])
def test_real_transition_excludes_inner_usage_acquisition_time(tmp_path, monkeypatch, clock, storage_elapsed):
    now, _owner = clock
    (tmp_path / "state").mkdir(exist_ok=True)
    publish(tmp_path)
    probes, locks, sleeps, observed_attempts = [], [], [], []
    blocked_slices = int(storage_elapsed / _usage_wait.ACQUISITION_SLICE_SEC)
    pending_probes = {1, 3 + blocked_slices}
    read = queue_observation.load_queue_document

    def publication(root, name):
        probes.append(len(probes) + 1)
        if len(probes) in pending_probes:
            (root / "state/queue.alive").unlink(missing_ok=True)
        else:
            publish(root)
        now[0] += 0.01
        return read(root, name)  # The real dispatch_probe produces the timed exception.

    current = {"state": "reserved", "seq": 1}

    def attempt(attempt_id):
        observed_attempts.append(attempt_id)
        return current

    @contextlib.contextmanager
    def locked(_root, *, timeout_sec=None):
        locks.append(timeout_sec)
        assert timeout_sec == _usage_wait.ACQUISITION_SLICE_SEC
        if len(locks) <= blocked_slices:
            now[0] += timeout_sec  # Ordinary bounded slices accumulate INSIDE the outer callback.
            raise UsageLockUnavailable("contention slice", reason="contention")
        yield SimpleNamespace(attempt=attempt)

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    monkeypatch.setattr(queue_observation, "load_queue_document", publication)
    monkeypatch.setattr(queue_observation, "_sleep", sleep)
    monkeypatch.setattr(ua, "_locked", locked)
    reservation = SimpleNamespace(scope=ua.UsageScope(drive_root=tmp_path, task_id="root", root_task_id="root"),
                                  drive_root=tmp_path, attempt_id="same-attempt")
    # The real expected-sequence return observes the same reservation before any
    # mutation/physical claim. The fake view has no writer or provider to invoke.
    assert ua._transition(reservation, "dispatched", _expected_seq=0) == current
    assert observed_attempts == ["same-attempt"] and current == {"state": "reserved", "seq": 1}
    assert len(locks) == blocked_slices + 1 and len(probes) == blocked_slices + 5
    assert sleeps == [0.5, 0.5]
