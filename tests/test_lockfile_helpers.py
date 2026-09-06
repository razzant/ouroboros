import contextlib
import errno
import inspect
import os
import pathlib
import re
import threading
import time

import pytest

from ouroboros import platform_layer
from ouroboros.platform_layer import (
    acquire_exclusive_file_lock,
    refresh_exclusive_file_lock,
    release_exclusive_file_lock,
    unlink_lockfile,
)


def _steal(lock_path, text="pid=1 ts=stolen\n"):
    """Replace the lock file the way an evictor + a second acquirer would."""
    lock_path.unlink()
    lock_path.write_text(text, encoding="utf-8")


def test_release_without_fd_does_not_unlink_existing_lock(tmp_path):
    lock_path = tmp_path / "state.lock"
    lock_path.write_text("owned elsewhere", encoding="utf-8")

    release_exclusive_file_lock(lock_path, None)

    assert lock_path.read_text(encoding="utf-8") == "owned elsewhere"


def test_release_with_fd_unlinks_owned_lock(tmp_path):
    lock_path = tmp_path / "state.lock"
    fd = acquire_exclusive_file_lock(lock_path, metadata="owned\n")
    assert fd is not None

    release_exclusive_file_lock(lock_path, fd)

    assert not lock_path.exists()


def test_path_only_git_lock_cleanup_remains_available(tmp_path):
    lock_path = tmp_path / "git.lock"
    fd = acquire_exclusive_file_lock(lock_path, metadata="owned\n")
    assert fd is not None
    os.close(fd)

    unlink_lockfile(lock_path)

    assert not lock_path.exists()


# --- Ownership: a lock is only ever OURS to renew or remove -------------------


@pytest.mark.skipif(platform_layer.IS_WINDOWS, reason="the test itself unlinks or rewrites a lock file its owner holds open, which Windows refuses; the protocol it pins runs on both tiers of both OSes")
def test_release_never_unlinks_a_lock_that_was_stolen(tmp_path):
    """A hold evicted as stale must not delete the new owner's lock on exit."""
    lock_path = tmp_path / "state.lock"
    fd = acquire_exclusive_file_lock(lock_path, metadata="pid=old ts=0\n")
    assert fd is not None
    _steal(lock_path)

    release_exclusive_file_lock(lock_path, fd)

    assert lock_path.read_text(encoding="utf-8") == "pid=1 ts=stolen\n"


@pytest.mark.skipif(platform_layer.IS_WINDOWS, reason="the test itself unlinks or rewrites a lock file its owner holds open, which Windows refuses; the protocol it pins runs on both tiers of both OSes")
def test_stale_eviction_never_removes_a_lock_re_created_under_it(tmp_path, monkeypatch):
    """Judge one file, unlink that same file — or none at all.

    Between the staleness judgement and the unlink the real owner may release
    and a third writer take the lock; removing THAT file would put two writers
    on one authority.
    """
    lock_path = tmp_path / "state.lock"
    lock_path.write_text("pid=424242 ts=0\n", encoding="utf-8")
    os.utime(lock_path, (0.0, 0.0))  # ancient: judged abandoned

    def racing_alive(pid):
        _steal(lock_path, "pid=999 ts=fresh\n")
        return False  # the judged owner really is gone

    monkeypatch.setattr(platform_layer, "pid_is_alive", racing_alive)
    fd = acquire_exclusive_file_lock(
        lock_path, timeout_sec=0.3, stale_sec=1.0, poll_sec=0.02,
        owner_aware_stale=True,
    )

    assert fd is None  # the lock was not ours to take
    assert lock_path.read_text(encoding="utf-8") == "pid=999 ts=fresh\n"


def test_stale_eviction_still_reclaims_a_genuinely_abandoned_lock(tmp_path, monkeypatch):
    lock_path = tmp_path / "state.lock"
    lock_path.write_text("pid=424242 ts=0\n", encoding="utf-8")
    os.utime(lock_path, (0.0, 0.0))
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: False)

    fd = acquire_exclusive_file_lock(
        lock_path, timeout_sec=1.0, stale_sec=1.0, poll_sec=0.02,
        owner_aware_stale=True,
    )

    assert fd is not None
    release_exclusive_file_lock(lock_path, fd)


@pytest.mark.skipif(platform_layer.IS_WINDOWS, reason="kill(pid, 0) is the POSIX liveness probe")
def test_a_pid_that_refuses_our_signal_is_alive_and_its_lock_is_not_reclaimed(tmp_path, monkeypatch):
    """EPERM from ``kill(pid, 0)`` means the process EXISTS and merely refuses
    our signal — another user's process on a shared host, or a pid recycled
    onto one. Reading it as dead let the owner-aware stale rule evict such a
    lock by age; it is alive, so the lock stays (the disclosed recycled-pid
    wedge), and only ESRCH is a process provably gone."""
    def answering(code):
        def kill(pid, sig):
            raise OSError(code, "kill refused")
        return kill

    monkeypatch.setattr(os, "kill", answering(errno.EPERM))
    assert platform_layer.pid_is_alive(4242) is True and platform_layer.pid_provably_gone(4242) is False
    lock_path = tmp_path / "state.lock"
    lock_path.write_text("pid=4242 ts=0\n", encoding="utf-8")
    os.utime(lock_path, (0.0, 0.0))  # aged past any staleness window
    assert acquire_exclusive_file_lock(
        lock_path, timeout_sec=0.3, stale_sec=1.0, poll_sec=0.02, owner_aware_stale=True,
    ) is None
    assert lock_path.read_text(encoding="utf-8") == "pid=4242 ts=0\n"
    monkeypatch.setattr(os, "kill", answering(errno.ESRCH))
    assert platform_layer.pid_is_alive(4242) is False and platform_layer.pid_provably_gone(4242) is True


@pytest.mark.skipif(platform_layer.IS_WINDOWS, reason="the test itself unlinks or rewrites a lock file its owner holds open, which Windows refuses; the protocol it pins runs on both tiers of both OSes")
def test_heartbeat_reports_lost_ownership_instead_of_renewing(tmp_path):
    """The renewal is an OWNERSHIP statement: a stolen lock renews nothing."""
    lock_path = tmp_path / "state.lock"
    fd = acquire_exclusive_file_lock(lock_path, metadata="pid=old ts=0\n")
    assert fd is not None
    assert refresh_exclusive_file_lock(lock_path, fd) is True
    _steal(lock_path)

    assert refresh_exclusive_file_lock(lock_path, fd) is False

    os.close(fd)


@pytest.mark.skipif(platform_layer.IS_WINDOWS, reason="the test itself unlinks or rewrites a lock file its owner holds open, which Windows refuses; the protocol it pins runs on both tiers of both OSes")
def test_heartbeat_on_a_deleted_lock_reports_lost_ownership(tmp_path):
    lock_path = tmp_path / "state.lock"
    fd = acquire_exclusive_file_lock(lock_path, metadata="pid=old ts=0\n")
    assert fd is not None
    lock_path.unlink()

    assert refresh_exclusive_file_lock(lock_path, fd) is False

    os.close(fd)


@pytest.mark.skipif(
    platform_layer.IS_WINDOWS,
    reason="the swap mechanism is POSIX: Windows cannot replace an open, "
    "LockFileEx-held lock file",
)
def test_heartbeat_after_an_atomic_swap_of_the_lock_reports_false(tmp_path):
    """A thief that REPLACES the lock file atomically never leaves the path
    absent, not even briefly — so any verdict weaker than an identity
    comparison (an existence check, a successful utime) would renew a hold
    that is no longer ours."""
    lock_path = tmp_path / "state.lock"
    fd = acquire_exclusive_file_lock(lock_path, metadata="pid=old ts=0\n")
    assert fd is not None
    assert refresh_exclusive_file_lock(lock_path, fd) is True
    imposter = tmp_path / "imposter.lock"
    imposter.write_text("pid=1 ts=swapped\n", encoding="utf-8")
    os.replace(imposter, lock_path)

    assert refresh_exclusive_file_lock(lock_path, fd) is False

    os.close(fd)
    assert lock_path.read_text(encoding="utf-8") == "pid=1 ts=swapped\n"


@pytest.mark.skipif(
    platform_layer.IS_WINDOWS,
    reason="the eviction ORDER under test is POSIX: it unlinks under the held "
    "flock, while Windows must close its probe first and leans on the new owner's "
    "open handle across that gap (test_windows_evicts_a_stale_lock_only_under_the_probe_hold)",
)
def test_two_racing_reclaimers_never_yield_two_holders(tmp_path, monkeypatch):
    """Kernel-enforced eviction: judge, re-check and unlink happen under a
    held flock on the very fd that was judged, so of two reclaimers racing
    over one abandoned lock at most ONE may evict — the other either fails
    the non-blocking flock or fails the inode re-check.  Without the kernel
    lock, a pause between the inode re-check and the unlink lets the second
    reclaimer remove the first one's freshly won lock: two writers on one
    monetary authority."""
    lock_path = tmp_path / "state.lock"
    lock_path.write_text("pid=424242 ts=0\n", encoding="utf-8")
    os.utime(lock_path, (0.0, 0.0))  # ancient: judged abandoned by age
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: False)
    stale_ident = platform_layer._lock_identity(lock_path)
    barrier = threading.Barrier(2)
    arrival_lock = threading.Lock()
    arrivals: list = []
    real_identity = platform_layer._lock_identity

    def pausing_identity(target):
        # The pause lands between the inode re-check (this read of the PATH's
        # identity, still naming the judged stale file) and the unlink that
        # trusts it.  Both reclaimers are herded into the window together;
        # the first to arrive then yields, so the second evicts and acquires
        # while the first still believes its own, now stale, re-check.
        result = real_identity(target)
        if not isinstance(target, int) and result and result[:2] == stale_ident[:2]:
            with contextlib.suppress(threading.BrokenBarrierError):
                barrier.wait(timeout=0.4)
            with arrival_lock:
                first = not arrivals
                arrivals.append(1)
            if first:
                time.sleep(0.15)
        return result

    monkeypatch.setattr(platform_layer, "_lock_identity", pausing_identity)
    results: list = [None, None]

    def reclaim(slot):
        results[slot] = acquire_exclusive_file_lock(
            lock_path, timeout_sec=2.0, stale_sec=1.0, poll_sec=0.01,
            owner_aware_stale=True,
        )

    threads = [threading.Thread(target=reclaim, args=(slot,)) for slot in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    holders = [fd for fd in results if fd is not None]
    assert len(holders) == 1, "two writers hold one monetary lock"
    assert lock_path.exists()
    release_exclusive_file_lock(lock_path, holders[0])


@pytest.mark.skipif(
    platform_layer.IS_WINDOWS,
    reason="flock-held eviction is POSIX; Windows cannot unlink a creator's open file",
)
def test_a_creator_evicted_while_lock_less_never_returns_a_descriptor(tmp_path, monkeypatch):
    """Between its O_EXCL create and its kernel lock a creator holds nothing an
    evictor must respect: stalled there past ``stale_sec`` (SIGSTOP, a suspend,
    clock skew) its fresh file is judged abandoned and evicted, and the lock it
    then takes lands on an inode the path no longer names. That is not a hold:
    the acquisition proves the path still names its descriptor and re-contends.
    Belt: the owner pid is written BEFORE the lock, so an owner-aware reclaimer
    never judges the live creator's file empty."""
    lock_path = tmp_path / "state.lock"
    monkeypatch.setattr(platform_layer, "_KERNEL_LOCK_TIER", {})
    assert platform_layer.kernel_file_locks_enforced(lock_path) is True  # decided before the hook
    real_flock = platform_layer.file_lock_exclusive_nb
    seen: dict = {}

    def stalled_creator_flock(fd):
        if not seen:  # the creator's own first kernel lock: it stalls here
            seen["metadata"] = lock_path.read_text(encoding="utf-8")
            seen["reclaimer"] = None
            os.utime(lock_path, (0.0, 0.0))  # the stall aged its lock-less file
            seen["reclaimer"] = acquire_exclusive_file_lock(  # an age-only reclaimer
                lock_path, timeout_sec=2.0, stale_sec=1.0, poll_sec=0.01,
            )
        return real_flock(fd)

    monkeypatch.setattr(platform_layer, "file_lock_exclusive_nb", stalled_creator_flock)
    creator = acquire_exclusive_file_lock(lock_path, timeout_sec=0.5, stale_sec=1.0, poll_sec=0.01)

    holders = [fd for fd in (creator, seen["reclaimer"]) if fd is not None]
    assert len(holders) == 1, "two descriptors believed to be one lock"
    assert refresh_exclusive_file_lock(lock_path, holders[0]) is True
    assert f"pid={os.getpid()}" in seen["metadata"]  # known to any owner-aware evictor
    release_exclusive_file_lock(lock_path, holders[0])


# --- Tiers: kernel-enforced or name-only, chosen by predicate, never by a refusal


def _tier(monkeypatch, enforced):
    monkeypatch.setattr(
        platform_layer, "kernel_file_locks_enforced", lambda path: enforced, raising=False,
    )


def _refusing(code):
    def refuse(fd):
        raise OSError(code, "injected kernel refusal")
    return refuse


def test_a_kernel_refusal_that_is_not_contention_fails_closed(tmp_path, monkeypatch):
    """On the enforced tier a descriptor the kernel would not lock is not a
    hold: the acquisition answers None promptly (not after the timeout) and
    removes the file it created, instead of degrading to the name protocol —
    where the round-3 reclaimer race lives again."""
    lock_path = tmp_path / "state.lock"
    _tier(monkeypatch, True)
    monkeypatch.setattr(platform_layer, "file_lock_exclusive_nb", _refusing(errno.ENOLCK))
    started = time.time()

    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=5.0, poll_sec=0.01)

    assert fd is None
    assert time.time() - started < 2.0
    assert not lock_path.exists()


def test_a_stale_lock_is_never_evicted_without_the_kernel_hold(tmp_path, monkeypatch):
    """Eviction happens only under a held kernel lock on the judged fd: a
    refusal of that hold that is not contention leaves the stale file where
    it is and fails the acquisition closed — never unlink-by-name instead."""
    lock_path = tmp_path / "state.lock"
    lock_path.write_text("pid=424242 ts=0\n", encoding="utf-8")
    os.utime(lock_path, (0.0, 0.0))
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: False)
    _tier(monkeypatch, True)
    monkeypatch.setattr(platform_layer, "file_lock_exclusive_nb", _refusing(errno.EIO))

    fd = acquire_exclusive_file_lock(
        lock_path, timeout_sec=1.0, stale_sec=1.0, poll_sec=0.01, owner_aware_stale=True,
    )

    assert fd is None
    assert lock_path.read_text(encoding="utf-8") == "pid=424242 ts=0\n"


def test_the_name_tier_is_chosen_by_the_predicate_not_by_a_refusal(tmp_path, monkeypatch):
    """Where the predicate says the filesystem takes no kernel locks, the name
    protocol runs alone and NO kernel call is attempted — so a refusal can
    never be what decides the tier. Abandoned locks are still reclaimed there
    by the disclosed re-check-then-unlink shape."""
    lock_path = tmp_path / "state.lock"
    _tier(monkeypatch, False)
    kernel_calls: list = []
    monkeypatch.setattr(platform_layer, "file_lock_exclusive_nb", kernel_calls.append)

    fd = acquire_exclusive_file_lock(lock_path, metadata="pid=old ts=0\n")
    assert fd is not None and kernel_calls == []
    assert refresh_exclusive_file_lock(lock_path, fd) is True
    release_exclusive_file_lock(lock_path, fd)
    assert not lock_path.exists()

    lock_path.write_text("pid=424242 ts=0\n", encoding="utf-8")
    os.utime(lock_path, (0.0, 0.0))
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: False)
    fd = acquire_exclusive_file_lock(
        lock_path, timeout_sec=1.0, stale_sec=1.0, poll_sec=0.01, owner_aware_stale=True,
    )
    assert fd is not None and kernel_calls == []
    release_exclusive_file_lock(lock_path, fd)


def test_the_capability_probe_decides_once_and_leaves_no_residue(tmp_path, monkeypatch):
    """Only the kernel's own "this filesystem cannot" answer selects the name
    tier; any other refusal keeps the enforced tier (where a live acquisition
    fails closed). The verdict is memoized per directory and the probe file
    is gone afterwards."""
    monkeypatch.setattr(platform_layer, "_KERNEL_LOCK_TIER", {})
    assert platform_layer.kernel_file_locks_enforced(tmp_path / "real.lock") is True
    lockless = tmp_path / "lockless"
    lockless.mkdir()
    answers: list = []

    def probing(fd):
        answers.append(fd)
        raise OSError(errno.EOPNOTSUPP, "operation not supported")

    monkeypatch.setattr(platform_layer, "file_lock_exclusive_nb", probing)
    assert platform_layer.kernel_file_locks_enforced(lockless / "a.lock") is False
    assert platform_layer.kernel_file_locks_enforced(lockless / "b.lock") is False
    assert len(answers) == 1  # one probe per directory
    assert list(lockless.iterdir()) == []

    # EIO is no capability answer: the enforced tier. ENOLCK — "no locks available",
    # a lockd-less NFS or an exhausted lock table — is the name tier for ordinary
    # locks, RECORDED beside the verdict so a caller may refuse it (the monetary lock does).
    for code, enforced in ((errno.EIO, True), (errno.ENOLCK, False)):
        refusing = tmp_path / f"refusing-{code}"
        refusing.mkdir()
        monkeypatch.setattr(platform_layer, "file_lock_exclusive_nb", _refusing(code))
        assert platform_layer.kernel_file_locks_enforced(refusing / "a.lock") is enforced, code
        assert platform_layer._KERNEL_LOCK_TIER[os.path.realpath(str(refusing))] == (enforced, code)
        assert list(refusing.iterdir()) == []


def test_enolck_is_the_name_tier_for_ordinary_locks_and_a_typed_refusal_for_money(tmp_path, monkeypatch):
    """A filesystem without a lock daemon answers ENOLCK to every kernel lock; so
    does an exhausted lock table. Neither is "held", so the probe records it as the
    name tier: ordinary locks (state singletons, task results, custody) keep the
    O_EXCL name protocol they always ran there — the shared primitive fails no
    one else closed — while the MONETARY lock names ENOLCK in
    ``refuse_name_tier_errnos`` and fails closed at once (no descriptor, no file,
    the name protocol never run for money): every monetary writer refuses typed
    and the compaction pass never enters on that tier."""
    from ouroboros import usage_ledger

    lock_path = tmp_path / "state" / "ordinary.lock"
    lock_path.parent.mkdir()  # an unprobeable (absent) directory answers enforced-uncached; probe the real one
    monkeypatch.setattr(platform_layer, "_KERNEL_LOCK_TIER", {})
    monkeypatch.setattr(platform_layer, "file_lock_exclusive_nb", _refusing(errno.ENOLCK))
    assert platform_layer.kernel_file_locks_enforced(lock_path) is False
    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=5.0, poll_sec=0.01)
    assert fd is not None and lock_path.exists()  # the name protocol, as before round 5.4
    platform_layer.release_exclusive_file_lock(lock_path, fd)
    started = time.time()
    assert acquire_exclusive_file_lock(
        lock_path, timeout_sec=5.0, poll_sec=0.01, refuse_name_tier_errnos=frozenset({errno.ENOLCK}),
    ) is None
    assert time.time() - started < 2.0 and not lock_path.exists()
    with pytest.raises(usage_ledger.UsageAccountingError, match="lock unavailable"):
        with usage_ledger._named_lock(tmp_path, "usage_attempts.lock", timeout_sec=1.0, stale_sec=90.0):
            raise AssertionError("the monetary lock was taken on the ENOLCK tier")
    assert not (tmp_path / "state" / "usage_attempts.lock").exists()


def test_two_threads_racing_the_first_probe_run_one_probe_and_read_one_tier(tmp_path, monkeypatch):
    """The tier cache is read and written under one lock: two threads asking
    about a directory nobody has probed run ONE probe and read one verdict —
    never two probes whose answers could disagree (a lock-less answer on one,
    a transient refusal on the other) and leave one thread on each tier of the
    same directory, the compactor trusting a name-tier descriptor as enforced."""
    monkeypatch.setattr(platform_layer, "_KERNEL_LOCK_TIER", {})
    barrier = threading.Barrier(2)
    probes: list = []

    def probing(fd):
        probes.append(fd)
        with contextlib.suppress(threading.BrokenBarrierError):
            barrier.wait(timeout=0.4)  # two probes can only meet here without the cache lock
        raise OSError(errno.EOPNOTSUPP, "operation not supported")

    monkeypatch.setattr(platform_layer, "file_lock_exclusive_nb", probing)
    answers: list = [None, None]

    def ask(slot):
        answers[slot] = platform_layer.kernel_file_locks_enforced(tmp_path / "a.lock")

    threads = [threading.Thread(target=ask, args=(slot,)) for slot in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert answers == [False, False] and len(probes) == 1


def test_lockfileex_refusals_classify_by_the_win32_error():
    """Only ERROR_LOCK_VIOLATION (33) means held by someone: it reads as the
    busy errno and re-contends. Access denied (5) and sharing violation (32)
    land on EACCES beside it on Windows, so EACCES cannot be in the busy set
    (POSIX flock never answers it). ERROR_INVALID_FUNCTION (1) and
    ERROR_NOT_SUPPORTED (50) — what a redirector answers when the volume takes
    no byte-range locks at all — must reach the UNSUPPORTED set, or the name
    tier is unreachable on Windows and a lock-less volume fails every monetary
    append closed instead of degrading to the disclosed name protocol. Every
    other error is in neither set: it fails the acquisition closed at once, not
    after the timeout. The classified codes carry their errno themselves — the
    4-argument OSError form derives errno FROM the winerror on Windows and
    ignores the one passed, and answers 0 here — so this arithmetic is the same
    on both platforms."""
    assert errno.EACCES not in platform_layer._LOCK_HELD_ERRNOS
    busy = platform_layer._win32_lock_error(33)
    assert busy.errno in platform_layer._LOCK_HELD_ERRNOS and busy.winerror == 33
    for err in (1, 50):
        unsupported = platform_layer._win32_lock_error(err)
        assert unsupported.errno in platform_layer._LOCK_UNSUPPORTED_ERRNOS and unsupported.winerror == err
    for err in (5, 32, 6):
        answered = platform_layer._win32_lock_error(err).errno
        assert answered not in platform_layer._LOCK_HELD_ERRNOS
        assert answered not in platform_layer._LOCK_UNSUPPORTED_ERRNOS


def test_the_design_note_names_the_exact_kernel_refusal_sets():
    """Round 5.2 corrected the busy set in the code, in its pin and in the
    review packet, and left the RATIFIED design note saying EACCES means
    contention — the negation of what that same pin asserts. A reader who
    implements the note re-opens the finding: a genuine access-denied would
    re-contend for the whole 45 s monetary timeout instead of failing closed.
    So the note names both sets and this compares them, member for member, by
    the numbers (EWOULDBLOCK and ENOTSUP are aliases on Linux, not everywhere)."""
    note = pathlib.Path(__file__).resolve().parents[1] / "docs" / "v7next" / "DESIGN_USAGE_COMPACTION.md"
    spelled = re.findall(r"are exactly ((?:`[A-Z]+`/)+`[A-Z]+`)", note.read_text(encoding="utf-8"))
    assert len(spelled) == 2, spelled
    unsupported, held = ({getattr(errno, name.strip("`")) for name in group.split("/")} for group in spelled)
    assert held == set(platform_layer._LOCK_HELD_ERRNOS)
    assert unsupported == set(platform_layer._LOCK_UNSUPPORTED_ERRNOS)


@pytest.mark.skipif(
    not platform_layer.IS_WINDOWS,
    reason="LockFileEx error classification is Windows mechanics",
)
def test_windows_lockfileex_contention_reads_as_busy(tmp_path):  # pragma: no cover - Windows only
    """A refused LockFileEx carries the errno the acquisition classifies: a
    lock violation is contention (stand down, re-contend); anything else
    fails closed. An errno-less OSError used to fall into the degrade."""
    path = tmp_path / "held.lock"
    first = os.open(str(path), os.O_CREAT | os.O_RDWR)
    second = os.open(str(path), os.O_RDWR)
    try:
        platform_layer._win32_lock(first, exclusive=True, blocking=False)
        with pytest.raises(OSError) as refused:
            platform_layer._win32_lock(second, exclusive=True, blocking=False)
        assert refused.value.errno in platform_layer._LOCK_HELD_ERRNOS
        assert refused.value.winerror == 33  # ERROR_LOCK_VIOLATION
    finally:
        platform_layer._win32_unlock(first)
        os.close(second)
        os.close(first)


@pytest.mark.skipif(
    platform_layer.IS_WINDOWS,
    reason="the unlink-under-the-creator shape is POSIX; Windows cannot unlink an open file",
)
def test_a_lock_whose_identity_cannot_be_read_is_never_a_hold(tmp_path, monkeypatch):
    """``_lock_identity`` answers ``()`` for a descriptor it cannot ``fstat`` —
    ESTALE/EIO on exactly the network filesystems this tier exists for. Two
    unreadable sides are not a match: comparing them raw makes ``() == ()``
    vacuously true and hands back a descriptor for an inode the path no longer
    names (a reclaimer unlinked it inside its own re-contend window), i.e. a
    second holder of one monetary lock. Unprovable is not owned: the
    acquisition answers None — and the file it stamped with its own LIVE pid
    goes with it, or no owner-aware reclaimer may ever remove it again."""
    lock_path = tmp_path / "state.lock"
    monkeypatch.setattr(platform_layer, "_KERNEL_LOCK_TIER", {})
    assert platform_layer.kernel_file_locks_enforced(lock_path) is True  # decided before the hooks
    real_flock = platform_layer.file_lock_exclusive_nb
    real_identity = platform_layer._lock_identity

    def blind_descriptor(target):  # our own fd answers nothing; the path still answers
        return () if isinstance(target, int) else real_identity(target)

    def evicting_flock(fd):  # a reclaimer removed our file between the create and the lock
        if lock_path.exists():
            os.unlink(str(lock_path))
        return real_flock(fd)

    monkeypatch.setattr(platform_layer, "_lock_identity", blind_descriptor)
    monkeypatch.setattr(platform_layer, "file_lock_exclusive_nb", evicting_flock)
    assert acquire_exclusive_file_lock(lock_path, timeout_sec=0.3, poll_sec=0.01) is None

    monkeypatch.setattr(platform_layer, "file_lock_exclusive_nb", real_flock)
    assert acquire_exclusive_file_lock(lock_path, timeout_sec=0.3, poll_sec=0.01) is None
    assert not lock_path.exists(), "a live pid was stamped on a lock nobody may reclaim"

    # The heartbeat has the same blind spots. A renewal is an ownership verdict:
    # a renewal the kernel refuses is no renewal, our own identity unreadable
    # with the path absent is not a match of two empty answers, and a stranger's
    # file at the path is not ours either.
    monkeypatch.setattr(platform_layer, "_lock_identity", real_identity)
    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=0.3, poll_sec=0.01)
    assert fd is not None and refresh_exclusive_file_lock(lock_path, fd) is True
    real_utime = os.utime
    monkeypatch.setattr(os, "utime", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EIO, "utime refused")))
    assert refresh_exclusive_file_lock(lock_path, fd) is False
    monkeypatch.setattr(os, "utime", real_utime)
    monkeypatch.setattr(platform_layer, "_lock_identity", blind_descriptor)
    os.unlink(str(lock_path))
    assert refresh_exclusive_file_lock(lock_path, fd) is False
    lock_path.write_text("pid=1 ts=stranger\n", encoding="utf-8")
    assert refresh_exclusive_file_lock(lock_path, fd) is False
    os.close(fd)


def test_pid_is_signalable_is_the_kill_question():
    """Liveness and ownership are different questions: a pid another user owns
    (pid 1 on a non-root POSIX host) is ALIVE for a lock owner but NOT signalable,
    so a cleanup must not try to kill it; our own pid is both; a vanished pid is neither."""
    assert platform_layer.pid_is_signalable(os.getpid()) is True
    assert platform_layer.pid_is_signalable(0) is False
    if not platform_layer.IS_WINDOWS and getattr(os, "geteuid", lambda: 1)() != 0:
        assert platform_layer.pid_is_alive(1) is True
        assert platform_layer.pid_is_signalable(1) is False


def _windows_kernel_tier(monkeypatch):
    """Drive the WINDOWS lock shapes on any host: ``IS_WINDOWS`` plus a stand-in for the
    two LockFileEx wrappers (whose ``ctypes.wintypes`` exists on Windows only).  The
    stand-in grants exactly what the real one asks the kernel for — an exclusive hold on
    the single byte at ``_WIN32_LOCK_OFFSET``, refused with ERROR_LOCK_VIOLATION while
    another descriptor holds it — and RECORDS every range requested, so a pin can compare
    it against the stamp bytes a contender must still be able to read.  Returns that log."""
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    monkeypatch.setattr(platform_layer, "_KERNEL_LOCK_TIER", {})
    ranges: list = []
    held: dict = {}

    def win32_lock(fd, *, exclusive=True, blocking=True):
        ranges.append((platform_layer._WIN32_LOCK_OFFSET, platform_layer._WIN32_LOCK_LENGTH))
        info = os.fstat(fd)
        if held.get((info.st_dev, info.st_ino), fd) != fd:
            raise platform_layer._win32_lock_error(33)  # ERROR_LOCK_VIOLATION: held by someone
        held[(info.st_dev, info.st_ino)] = fd

    def win32_unlock(fd):
        for key, owner in list(held.items()):
            if owner == fd:
                del held[key]

    monkeypatch.setattr(platform_layer, "_win32_lock", win32_lock)
    monkeypatch.setattr(platform_layer, "_win32_unlock", win32_unlock)
    return ranges


def test_the_windows_lock_range_lies_beyond_every_owner_stamp():
    """The bf8b6549 matrix died of a MANDATORY whole-file LockFileEx: a contender
    could not READ the stamp it must read to judge the hold, so every wait ran to
    its timeout (eight monetary writers refused, a chat append lost).  The hold is
    therefore one byte at an offset the stamp — one short line, read 512 bytes at a
    time — can never reach, and the whole-file range is gone from both wrappers."""
    assert platform_layer._WIN32_LOCK_LENGTH == 1
    assert platform_layer._WIN32_LOCK_OFFSET > 512 * 1024 ** 3  # unreachable by any lock file
    for wrapper in (platform_layer._win32_lock, platform_layer._win32_unlock):
        source = inspect.getsource(wrapper)
        assert "_WIN32_LOCK_OFFSET" in source and "_WIN32_LOCK_LENGTH" in source
        assert "0xFFFFFFFF, 0xFFFFFFFF" not in source, "the whole-file range is back"


def test_windows_contenders_read_the_owner_stamp_while_the_kernel_hold_stands(tmp_path, monkeypatch):
    """With the range beyond the stamp the Windows tier behaves like POSIX: the
    predicate answers enforced, the hold is real (a second acquirer is refused), and
    the contender still READS the owner stamp on every poll — so it judges a live
    hold and stands down instead of timing out blind, which is what makes the tier
    shippable at all.  The stamp bytes are never inside a requested range."""
    ranges = _windows_kernel_tier(monkeypatch)
    lock_path = tmp_path / "state.lock"
    assert platform_layer.kernel_file_locks_enforced(lock_path) is True
    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=1.0, poll_sec=0.01)
    assert fd is not None

    with open(str(lock_path), "rb") as contender:  # the read a mandatory hold refused
        assert f"pid={os.getpid()}".encode() in contender.read(512)
    assert acquire_exclusive_file_lock(lock_path, timeout_sec=0.3, poll_sec=0.01) is None
    assert ranges and all(offset >= 512 for offset, _length in ranges)

    release_exclusive_file_lock(lock_path, fd)
    assert not lock_path.exists()
    again = acquire_exclusive_file_lock(lock_path, timeout_sec=0.5, poll_sec=0.01)
    assert again is not None
    release_exclusive_file_lock(lock_path, again)


def test_windows_evicts_a_stale_lock_only_under_the_probe_hold(tmp_path, monkeypatch):
    """On the enforced tier Windows now takes the SAME kernel hold on the judged
    descriptor before it may evict — it just unlinks after closing the probe,
    because it deletes no open file.  A stale-looking lock somebody actually holds
    is therefore never evicted (the hold refuses the probe); an abandoned one is."""
    _windows_kernel_tier(monkeypatch)
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: False)
    lock_path = tmp_path / "state.lock"
    lock_path.write_text("pid=424242 ts=0\n", encoding="utf-8")
    os.utime(lock_path, (0.0, 0.0))  # ancient: judged abandoned by age
    holder = os.open(str(lock_path), os.O_RDWR)
    platform_layer.file_lock_exclusive_nb(holder)  # ... but a live kernel hold stands

    assert acquire_exclusive_file_lock(
        lock_path, timeout_sec=0.4, stale_sec=1.0, poll_sec=0.01, owner_aware_stale=True,
    ) is None
    assert lock_path.read_text(encoding="utf-8") == "pid=424242 ts=0\n"

    platform_layer.file_unlock(holder)
    os.close(holder)
    reclaimed = acquire_exclusive_file_lock(
        lock_path, timeout_sec=1.0, stale_sec=1.0, poll_sec=0.01, owner_aware_stale=True,
    )
    assert reclaimed is not None
    release_exclusive_file_lock(lock_path, reclaimed)


def test_windows_release_unlocks_before_the_close_and_unlinks_after_it(tmp_path, monkeypatch):
    """Order, on Windows: release the kernel hold (a handle closed with an
    outstanding lock leaves the release undefined), then close, then unlink — the
    file being undeletable while our own handle is open.  Read by the fd's own
    liveness: it must still be open at the unlock and gone by the unlink."""
    _windows_kernel_tier(monkeypatch)
    lock_path = tmp_path / "state.lock"
    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=1.0, poll_sec=0.01)
    assert fd is not None
    order: list = []
    real_unlock, real_unlink = platform_layer.file_unlock, platform_layer._unlink_lock_path

    def unlock(target):
        order.append(("unlock", _fd_open(target)))
        return real_unlock(target)

    def unlink(path, held):
        order.append(("unlink", _fd_open(fd)))
        return real_unlink(path, held)

    monkeypatch.setattr(platform_layer, "file_unlock", unlock)
    monkeypatch.setattr(platform_layer, "_unlink_lock_path", unlink)

    release_exclusive_file_lock(lock_path, fd)

    assert order == [("unlock", True), ("unlink", False)]
    assert not lock_path.exists()


def _fd_open(fd):
    try:
        os.fstat(fd)
    except OSError:
        return False
    return True


def _refusing_unlink(monkeypatch, lock_path, *, refusals):
    """Windows delete semantics on any host: ``os.unlink`` of ``lock_path`` answers
    a sharing violation ``refusals`` times (a contender's probe handle still open —
    ``None``: forever), then behaves.  Returns the attempt counter."""
    real_unlink, attempts = os.unlink, []

    def unlink(path, *args, **kwargs):
        if pathlib.Path(path) == lock_path:
            attempts.append(time.monotonic())
            if refusals is None or len(attempts) <= refusals:
                raise PermissionError(errno.EACCES, "[WinError 32] sharing violation", str(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(platform_layer.os, "unlink", unlink)
    return attempts


def test_windows_release_retries_a_contenders_transient_sharing_refusal(tmp_path, monkeypatch):
    """The name protocol's contenders open the lock on every poll; on Windows that
    handle makes the owner's unlink fail with a sharing violation for a moment.
    Swallowing it orphaned the lock with the owner's LIVE pid inside (the Windows
    matrices after the C6 merge: every later monetary writer refused, chat appends
    fell to the unlocked lane), so the release retries until the handle is gone
    and the next acquirer wins at once instead of waiting out its timeout."""
    _windows_kernel_tier(monkeypatch)
    lock_path = tmp_path / "state.lock"
    fd = acquire_exclusive_file_lock(lock_path)
    assert fd is not None
    attempts = _refusing_unlink(monkeypatch, lock_path, refusals=3)

    release_exclusive_file_lock(lock_path, fd)

    assert len(attempts) == 4 and not lock_path.exists()
    fd2 = acquire_exclusive_file_lock(lock_path, timeout_sec=0.2, owner_aware_stale=True)
    assert fd2 is not None
    release_exclusive_file_lock(lock_path, fd2)
    assert not lock_path.exists()


def test_windows_release_gives_up_a_refusal_that_never_clears(tmp_path, monkeypatch):
    """The retry is bounded: a handle that never closes (an indexer, a foreign
    reader) cannot pin the releasing writer forever — it logs and returns
    within the window, leaving the file it could not remove."""
    _windows_kernel_tier(monkeypatch)
    lock_path = tmp_path / "state.lock"
    fd = acquire_exclusive_file_lock(lock_path)
    assert fd is not None
    attempts = _refusing_unlink(monkeypatch, lock_path, refusals=None)
    started = time.monotonic()

    release_exclusive_file_lock(lock_path, fd)

    assert 1.5 < time.monotonic() - started < 6.0 and len(attempts) > 10
    assert lock_path.exists()


@pytest.mark.skipif(platform_layer.IS_WINDOWS, reason="POSIX release order under test: with IS_WINDOWS patched False the enforced-tier probe imports fcntl, which Windows lacks")
def test_posix_release_does_not_retry_a_permission_refusal(tmp_path, monkeypatch):
    """POSIX never refuses an unlink for a reader's open handle: a
    PermissionError there is the directory's mode, permanent — retrying would
    only delay the writer by the whole window."""
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", False)
    lock_path = tmp_path / "state.lock"
    fd = acquire_exclusive_file_lock(lock_path)
    assert fd is not None
    attempts = _refusing_unlink(monkeypatch, lock_path, refusals=None)

    release_exclusive_file_lock(lock_path, fd)

    assert len(attempts) == 1 and lock_path.exists()
    monkeypatch.undo()  # 3.11+ pathlib.unlink calls os.unlink live: the refuser must be gone first
    lock_path.unlink()
