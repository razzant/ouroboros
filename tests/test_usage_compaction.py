"""CPL4-C6 pins: the seq-preserving compaction pass over the monetary ledger.

Design contract: docs/v7next/DESIGN_USAGE_COMPACTION.md. The invariants
pinned here are monetary-authority invariants (owner sanction 1A):

1. decimal-exact money before/after; the production projections render EQUAL;
2. in-flight (unsettled) rows never fold and stay transitionable;
3. crash between archive write and ledger swap leaves the ledger byte-identical;
4. budget enforcement sees the same numbers across compaction;
6. idempotent kinds (subscription/external/legacy) never fold, so their replay dedup keeps working;
7. trigger policy: config SSOT threshold, thrash guard, verify-abort = no-op.

The reader side of the same organ — invariant 5 (the CPL-5 join across
chained compactions) and invariant 8 (baseline rows are legal only as the
leading block) — lives in ``tests/test_usage_compaction_archive.py``; the
fixtures both modules share live in ``tests/fixtures_usage_compaction.py``.
"""

from __future__ import annotations

import base64
import decimal
import errno
import json
import logging
import os
import pathlib
import stat
import time
from decimal import Decimal

import pytest

from ouroboros import platform_layer
from ouroboros import usage_accounting as ua
from ouroboros import usage_compaction as uc
from ouroboros.agent_startup_checks import _hot_store_thresholds
from ouroboros.usage_ledger import UsageLedgerCorrupt, _validate_records
from tests import fixtures_usage_compaction as _fixtures
from tests.fixtures_usage_compaction import (
    _append_raw_row,
    _compact,
    _ledger_lines,
    _ledger_rows,
    _raced_row,
    _request,
    _seed_mixed_ledger,
    _settle,
)

# The three fixtures are re-exported through the module object rather than
# imported by name: a test's ``data_root`` parameter would shadow a bare
# import of the same name. pytest resolves them from this module either way.
compacted = _fixtures.compacted
data_root = _fixtures.data_root
data_root_any_tier = _fixtures.data_root_any_tier


def _decimal_money(rows):
    """Exact decimal (cost, bound) totals over FINAL rows, strings included.

    Summed under an INDEPENDENT wide context — wider than the compactor's own
    — so this oracle stays an oracle: a helper that rounds the same way the
    code under test rounds cannot see the code under test round."""
    finals = {}
    for row in rows:
        finals[str(row.get("attempt_id"))] = row
    with decimal.localcontext() as context:
        context.prec = 200
        cost = Decimal(0)
        bound = Decimal(0)
        for row in finals.values():
            if str(row.get("kind") or "") == "usage_baseline":
                continue
            value = row.get("cost_usd")
            if value is not None and str(row.get("state") or "") == "settled":
                cost += Decimal(str(value))
            upper = row.get("reservation_upper_bound_usd")
            if upper is not None:
                bound += Decimal(str(upper))
    return cost, bound


def _lock_path(data_root):
    return data_root / "state" / "usage_attempts.lock"


def _charge_survived(data_root, injected, before_money):
    """The raced charge is in the live ledger, no baseline landed, no temp
    residue, and money = before + that charge: the swap was refused whole."""
    rows = _ledger_rows(data_root)
    assert injected["attempt_id"] in {str(row.get("attempt_id")) for row in rows}
    assert not any(row.get("kind") == "usage_baseline" for row in rows)
    _validate_records(rows)
    ledger_path = data_root / ua.LEDGER_REL
    assert not list(ledger_path.parent.glob(f".{ledger_path.name}.tmp.*"))
    assert _decimal_money(rows) == (before_money[0] + Decimal("0.25"), before_money[1])


def _snapshot_looks(monkeypatch, on_look=lambda looks: None):
    """Record every snapshot verdict the pass asks for, in order; ``on_look``
    runs right after each answer — the instant an intrusion would land on it."""
    looks: list = []
    real = uc._snapshot_intact

    def counting(path, raw):
        looks.append(real(path, raw))
        on_look(looks)
        return looks[-1]

    monkeypatch.setattr(uc, "_snapshot_intact", counting)
    return looks


def _projection_snapshot(data_root):
    return (
        ua.usage_projection(data_root),
        ua.usage_projection(data_root, root_task_id="root"),
        ua.usage_projection(data_root, root_task_id="root2"),
        ua.usage_breakdown(data_root),
        ua.usage_breakdown(data_root, root_task_id="root"),
        ua.usage_breakdown(data_root, task_id="t2"),
    )


# --- 1 + 4: monetary exactness and budget equality ---------------------------

def test_compaction_preserves_money_and_projections_exactly(data_root):
    _seed_mixed_ledger(data_root)
    before_rows = _ledger_rows(data_root)
    before_money = _decimal_money(before_rows)
    before_projection = _projection_snapshot(data_root)
    before_review = ua.skill_review_usage(
        data_root, review_skill="skill-x", review_wave_id="w1")

    receipt = _compact(data_root)
    assert receipt is not None
    assert receipt["folded_row_count"] > 0

    after_rows = _ledger_rows(data_root)
    assert len(after_rows) < len(before_rows)
    assert _decimal_money(after_rows) == before_money
    assert _projection_snapshot(data_root) == before_projection
    # Review-attributed attempts are retained: the per-attempt wave projection
    # is unchanged, attempt ids included.
    after_review = ua.skill_review_usage(
        data_root, review_skill="skill-x", review_wave_id="w1")
    assert after_review == before_review
    assert after_review["attempt_ids"]

    # Baseline block structure: one header first, groups after, dense seq.
    header = after_rows[0]
    assert header["kind"] == "usage_baseline"
    assert header["seq"] == 1
    assert header["source_sha256"]
    assert [row["seq"] for row in after_rows] == list(range(1, len(after_rows) + 1))
    group_kinds = {row["kind"] for row in after_rows[1:]}
    assert "usage_baseline" not in group_kinds
    # Money on group rows is carried as exact-decimal strings.
    groups = [row for row in after_rows if row["kind"] == "usage_baseline_group"]
    assert groups
    assert all(isinstance(row.get("cost_usd"), str)
               for row in groups if row.get("cost_usd") is not None)


def test_group_sums_survive_beyond_the_default_decimal_precision(data_root, monkeypatch):
    """10**28 + 1 is 29 digits: the ambient 28-digit context loses the 1."""
    monkeypatch.setenv("TOTAL_BUDGET", "1e40")
    exact = Decimal("10000000000000000000000000001")
    _settle(data_root, cost=1e28, cost_final=True)
    _settle(data_root, cost=1.0, cost_final=True)
    before = _decimal_money(_ledger_rows(data_root))
    assert before[0] == exact
    assert _compact(data_root) is not None
    rows = _ledger_rows(data_root)
    groups = [row for row in rows if row["kind"] == "usage_baseline_group"]
    assert len(groups) == 1
    assert Decimal(groups[0]["cost_usd"]) == exact
    assert _decimal_money(rows) == before


def test_budget_enforcement_sees_identical_numbers(data_root, monkeypatch):
    monkeypatch.setenv("TOTAL_BUDGET", "10")
    _settle(data_root, cost=4.0, cost_final=True, reservation_usd=4.0)
    _settle(data_root, cost=4.0, cost_final=True, reservation_usd=4.0)
    before = ua.usage_projection(data_root)
    assert _compact(data_root) is not None
    assert ua.usage_projection(data_root) == before
    # Remaining ≈ 2: a 1.5 reservation fits, a 3.0 reservation does not —
    # exactly as before compaction.
    reservation = ua.reserve_attempt(_request(data_root, reservation_usd=1.5))
    ua.release_attempt(reservation, "probe")
    with pytest.raises(ua.BudgetExceeded):
        ua.reserve_attempt(_request(data_root, reservation_usd=3.0))


def test_root_budget_enforcement_survives_compaction(data_root):
    _settle(data_root, cost=8.0, cost_final=True, reservation_usd=8.0,
            task_id="rt", root_task_id="rooted", root_limit_usd=10.0)
    assert _compact(data_root) is not None
    with pytest.raises(ua.BudgetExceeded) as excinfo:
        ua.reserve_attempt(_request(data_root, reservation_usd=5.0, task_id="rt2",
                                    root_task_id="rooted", root_limit_usd=10.0))
    assert excinfo.value.limit_scope == "root"
    projection = ua.usage_projection(data_root, root_task_id="rooted")
    assert projection["limit_usd"] == 10.0
    assert projection["settled_usd"] == 8.0


# --- 2: in-flight rows never fold -------------------------------------------

def test_unsettled_rows_survive_and_stay_transitionable(data_root):
    reserved, dispatched = _seed_mixed_ledger(data_root)
    assert _compact(data_root) is not None
    states = {
        str(row.get("attempt_id")): str(row.get("state"))
        for row in _ledger_rows(data_root)
    }
    assert states[reserved.attempt_id] == "reserved"
    assert states[dispatched.attempt_id] == "dispatched"
    # Their lifecycle continues over the compacted file.
    ua.settle_attempt(dispatched, {"prompt_tokens": 3, "completion_tokens": 1},
                      cost_usd=0.25, cost_final=True)
    ua.release_attempt(reserved, "not_dispatched")
    finals = {
        str(row.get("attempt_id")): str(row.get("state"))
        for row in _ledger_rows(data_root)
    }
    assert finals[dispatched.attempt_id] == "settled"
    assert finals[reserved.attempt_id] == "released"


# --- 3: crash-safety ---------------------------------------------------------

def test_crash_at_the_ledger_rename_leaves_ledger_intact(data_root, monkeypatch):
    _seed_mixed_ledger(data_root)
    ledger_path = data_root / ua.LEDGER_REL
    archive_dir = data_root / "archive" / "usage_ledger"
    before_bytes = ledger_path.read_bytes()
    real_replace = os.replace
    observed: dict = {}

    def crashing_replace(src, dst):
        if pathlib.Path(dst) != ledger_path:
            return real_replace(src, dst)
        # The power cut lands ON the rename itself. By contract the archive
        # segment is already durable at this instant and holds the exact source
        # bytes: a swap that happened first would find no segment here.
        observed["segments"] = [
            path.read_bytes() for path in sorted(archive_dir.glob("segment_*.jsonl"))
        ]
        raise OSError(errno.EIO, "injected power loss at the ledger rename")

    monkeypatch.setattr(os, "replace", crashing_replace)
    with ua._locked(data_root) as heartbeat:
        with pytest.raises(OSError):
            uc.compact_usage_ledger_locked(data_root, heartbeat=heartbeat)
    assert observed["segments"] == [before_bytes]
    assert ledger_path.read_bytes() == before_bytes
    _validate_records(_ledger_rows(data_root))
    assert not list(ledger_path.parent.glob(f".{ledger_path.name}.tmp.*"))
    # The orphaned archive segment is tolerated; the ledger keeps working and a
    # retry (without the injection) compacts.
    monkeypatch.setattr(os, "replace", real_replace)
    _settle(data_root, cost=0.5, cost_final=True)
    assert _compact(data_root) is not None


@pytest.mark.skipif(platform_layer.IS_WINDOWS, reason="directory fsync and inode identity are POSIX (the ARCHITECTURE row discloses the Windows no-op)")
def test_archive_directory_chain_is_durable_before_the_swap(data_root, monkeypatch):
    _seed_mixed_ledger(data_root)
    archive_dir = data_root / "archive" / "usage_ledger"
    real_fsync = os.fsync
    synced: list = []
    swapped: list = []

    def recording_fsync(fd):
        info = os.fstat(fd)  # a descriptor being fsync'd is open by construction
        synced.append((info.st_dev, info.st_ino))
        return real_fsync(fd)

    real_swap = uc._swap_ledger_fsync

    def watched_swap(*args):
        swapped.append(len(synced))
        return real_swap(*args)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(uc, "_swap_ledger_fsync", watched_swap)
    assert _compact(data_root) is not None
    assert swapped, "the swap never ran"
    before_swap = set(synced[: swapped[0]])
    # Every directory whose entry the archive chain created must be durable
    # BEFORE the live ledger is replaced — not just the segment's own parent.
    for directory in (archive_dir, archive_dir.parent, data_root):
        info = directory.stat()
        assert (info.st_dev, info.st_ino) in before_swap, directory


def test_posix_directory_fsync_failure_aborts_before_the_swap(data_root, monkeypatch):
    if platform_layer.IS_WINDOWS:  # pragma: no cover - platform predicate
        pytest.skip("directory fsync is a disclosed no-op on Windows")
    _seed_mixed_ledger(data_root)
    ledger_path = data_root / ua.LEDGER_REL
    before_bytes = ledger_path.read_bytes()
    real_fsync = os.fsync

    def failing_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "injected directory fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with ua._locked(data_root) as heartbeat:
        with pytest.raises(OSError):
            uc.compact_usage_ledger_locked(data_root, heartbeat=heartbeat)
    assert ledger_path.read_bytes() == before_bytes


def test_the_directory_chain_is_re_synced_on_the_retry_after_a_failed_pass(data_root, monkeypatch):
    """A pass that died on the directory fsync leaves the directories PRESENT
    but not durable; the retry must sync them again rather than skip them for
    existing, or a crash after its swap loses the archive the swap relies on."""
    real_fsync = os.fsync
    test_posix_directory_fsync_failure_aborts_before_the_swap(data_root, monkeypatch)
    assert (data_root / "archive" / "usage_ledger").is_dir()  # present, durability unknown
    monkeypatch.setattr(os, "fsync", real_fsync)
    # The retry is the first-pass proof itself, run over the directories the
    # failed pass left behind: every level is fsync'd again before the swap.
    test_archive_directory_chain_is_durable_before_the_swap(data_root, monkeypatch)


@pytest.mark.skipif(platform_layer.IS_WINDOWS, reason="directory fsync and inode identity are POSIX (the ARCHITECTURE row discloses the Windows no-op)")
def test_the_swap_fsyncs_the_candidate_before_the_rename_and_its_directory_after(data_root, monkeypatch):
    """"A crash during the swap leaves the old file or the new one, both valid"
    rests on two calls: the candidate temp fsync'd BEFORE the replace (without
    it the renamed inode can hold unwritten data — neither the old ledger nor
    the approved new one) and the ledger's own directory after it, without
    which the rename may not survive the power cut. The archive half has three pins."""
    _seed_mixed_ledger(data_root)
    ledger_path = data_root / ua.LEDGER_REL
    real_fsync, real_replace = os.fsync, os.replace
    synced: list = []
    swap: dict = {}

    def recording_fsync(fd):
        synced.append(os.fstat(fd).st_ino)
        return real_fsync(fd)

    def watched_replace(src, dst):
        if pathlib.Path(dst) == ledger_path:
            swap.update(candidate=os.stat(src).st_ino, before=len(synced))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", watched_replace)
    assert _compact(data_root) is not None
    assert swap["candidate"] in synced[: swap["before"]]  # the bytes, before the rename
    assert ledger_path.parent.stat().st_ino in synced[swap["before"]:]  # the directory, after


# --- 1b: the lock the pass runs under ----------------------------------------

def test_monetary_lock_is_owner_aware_and_the_pass_heartbeats_it(data_root, monkeypatch):
    _seed_mixed_ledger(data_root)
    requested: dict = {}
    real_acquire = platform_layer.acquire_exclusive_file_lock

    def spy(path, **kwargs):
        requested.update(kwargs)
        return real_acquire(path, **kwargs)

    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", spy)
    lock_path = _lock_path(data_root)
    with ua._locked(data_root) as heartbeat:
        # A LIVE owner is never evicted on elapsed time alone: a stolen monetary
        # lock means two writers rewriting the same authority.
        assert requested.get("owner_aware_stale") is True
        # A pass that outlives the staleness window keeps its lockfile young
        # for acquirers that judge by age only.
        os.utime(lock_path, (0.0, 0.0))
        assert uc.compact_usage_ledger_locked(data_root, heartbeat=heartbeat) is not None
        assert lock_path.stat().st_mtime > time.time() - 60


def test_append_between_snapshot_and_swap_aborts_instead_of_erasing_it(data_root, monkeypatch):
    _seed_mixed_ledger(data_root)
    before_money = _decimal_money(_ledger_rows(data_root))
    original_write = uc._write_new_file_fsync
    injected: dict = {}

    def racing_write(path, payload, root):
        # A writer that got the lock (age-broken lock, foreign repair) lands a
        # settled charge AFTER the compactor snapshotted the file.
        injected.update(_append_raw_row(data_root, _raced_row("sess-raced")))
        original_write(path, payload, root)

    monkeypatch.setattr(uc, "_write_new_file_fsync", racing_write)
    assert _compact(data_root) is None  # refused the swap
    _charge_survived(data_root, injected, before_money)


def test_a_lost_lock_aborts_the_pass_instead_of_swapping(data_root):
    """A heartbeat is an OWNERSHIP verdict: losing it abandons the pass.

    A pass that keeps building after the lock left it would swap a snapshot
    over whatever the new owner appended in the meantime — and a pass handed
    no heartbeat at all has nothing to prove ownership with: refused, not run."""
    _seed_mixed_ledger(data_root)
    ledger_path = data_root / ua.LEDGER_REL
    before = ledger_path.read_bytes()
    for heartbeat in (
        lambda: False,                       # evicted / re-created under us
        lambda: (_ for _ in ()).throw(OSError("lock unreadable")),
        None,                                # the wire dropped: a caller defect, refused
    ):
        assert uc.compact_usage_ledger_locked(data_root, heartbeat=heartbeat) is None
        assert ledger_path.read_bytes() == before
    assert not (data_root / "archive" / "usage_ledger").exists()  # not even an orphan


def test_the_long_build_and_verification_section_beats_the_lock(data_root, monkeypatch):
    """The build/verify span is the long one; it must renew the hold WHILE it
    runs, not only at its edges."""
    _seed_mixed_ledger(data_root)
    beats: list = []
    seen: dict = {}
    real_build = uc._build_candidate

    def watched_build(*args, **kwargs):
        seen["entry"] = len(beats)
        result = real_build(*args, **kwargs)
        seen["exit"] = len(beats)
        return result

    monkeypatch.setattr(uc, "_build_candidate", watched_build)
    with ua._locked(data_root) as heartbeat:
        def counting():
            beats.append(1)
            return heartbeat()

        assert uc.compact_usage_ledger_locked(data_root, heartbeat=counting) is not None
    assert seen["exit"] > seen["entry"]   # the build itself renewed the hold
    assert len(beats) > seen["exit"]      # and so did the verification after it


def test_no_writer_can_append_between_the_snapshot_check_and_the_swap(data_root, monkeypatch):
    """The compare->replace window is closed by the lock, not by luck."""
    _seed_mixed_ledger(data_root)
    ledger_path = data_root / ua.LEDGER_REL
    real_replace = os.replace
    observed: dict = {}

    def probing_replace(src, dst):
        if pathlib.Path(dst) == ledger_path:
            # The legitimate append path IS this acquisition; at the instant of
            # the swap it must find the monetary lock held.
            observed["free"] = platform_layer.acquire_exclusive_file_lock(
                _lock_path(data_root), timeout_sec=0.2, stale_sec=3600.0,
                poll_sec=0.02, owner_aware_stale=True,
            )
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", probing_replace)
    assert _compact(data_root) is not None
    assert observed["free"] is None


def test_every_ledger_writer_refuses_when_the_lock_cannot_be_taken(data_root_any_tier, monkeypatch):
    """No unlocked fallback: a lock that cannot be taken is a typed refusal.

    An append that gave up on the lock and wrote anyway is exactly the second
    writer the compaction snapshot cannot see."""
    _settle(data_root_any_tier, cost=1.0, cost_final=True)
    reserved = ua.reserve_attempt(_request(data_root_any_tier, task_id="probe"))
    ledger_path = data_root_any_tier / ua.LEDGER_REL
    before = ledger_path.read_bytes()
    monkeypatch.setattr(
        platform_layer, "acquire_exclusive_file_lock", lambda *a, **k: None)
    for write in (
        lambda: ua.reserve_attempt(_request(data_root_any_tier)),
        lambda: ua.mark_dispatched(reserved),
        lambda: ua.record_unmetered_external_dispatch(
            "ext-locked", drive_root=data_root_any_tier, model="m", task_id="t"),
    ):
        with pytest.raises(ua.UsageAccountingError):
            write()
    assert ledger_path.read_bytes() == before


@pytest.mark.parametrize("intrusion", ("written_over", "erased"))
def test_a_swap_that_did_not_land_is_a_typed_failure_not_a_receipt(data_root, monkeypatch, intrusion):
    """Post-verify: the receipt describes the bytes that are actually there — and
    a charge landed on the OLD inode inside the one syscall between the last proof
    and the rename (an out-of-protocol holder) is erased by that rename, with
    nothing left at the path to show it. The old inode, held open across the swap
    (POSIX), still shows it: those bytes are quarantined, integrity is flagged, and
    the pass raises instead of returning a receipt over an erased charge."""
    if intrusion == "erased" and platform_layer.IS_WINDOWS:  # pragma: no cover - platform predicate
        pytest.skip("Windows cannot hold the destination open across os.replace: the loss stays silent, disclosed")
    _seed_mixed_ledger(data_root)
    ledger_path = data_root / ua.LEDGER_REL
    real_replace = os.replace
    landed: dict = {}

    def lying_replace(src, dst):
        if pathlib.Path(dst) == ledger_path and intrusion == "erased":
            landed.update(_append_raw_row(data_root, _raced_row("sess-in-the-syscall")))  # on the old inode
        real_replace(src, dst)
        if pathlib.Path(dst) == ledger_path and intrusion == "written_over":
            with open(dst, "ab") as handle:
                handle.write(b'{"kind":"attempt","attempt_id":"late"}\n')

    monkeypatch.setattr(os, "replace", lying_replace)
    with ua._locked(data_root) as heartbeat:
        with pytest.raises(UsageLedgerCorrupt):
            uc.compact_usage_ledger_locked(data_root, heartbeat=heartbeat)
    if intrusion == "erased":
        row = json.loads((data_root / ua.QUARANTINE_REL).read_text(encoding="utf-8").splitlines()[-1])
        assert json.loads(base64.b64decode(row["raw_base64"])) == landed  # the erased charge, byte for byte
        assert ua.usage_projection(data_root)["integrity_degraded"] is True


def test_an_append_between_the_recheck_and_the_replace_aborts_without_loss(
    data_root, monkeypatch
):
    """The pre-swap re-check is not the last look: the live ledger is proven
    unchanged again INSIDE the swap, after the candidate bytes are durable and
    immediately before the rename — the last instant the replace can still be
    refused. A row that lands after the outer re-check therefore survives."""
    _seed_mixed_ledger(data_root)
    before_money = _decimal_money(_ledger_rows(data_root))
    injected: dict = {}

    def land_after_the_recheck(looks):
        if len(looks) == 2 and looks[-1]:
            # The charge lands AFTER the outer pre-swap re-check answered
            # "intact" and BEFORE the rename trusts that answer.
            injected.update(_append_raw_row(data_root, _raced_row("sess-last-instant")))

    _snapshot_looks(monkeypatch, land_after_the_recheck)
    assert _compact(data_root) is None  # the replace was refused
    _charge_survived(data_root, injected, before_money)


def test_a_same_size_rewrite_between_the_recheck_and_the_replace_also_refuses(data_root, monkeypatch):
    """The in-swap look proves the live file is still the snapshot by size AND
    by bytes. Every intrusion the other pins inject is an append, so the size
    half alone would satisfy all of them — and a foreign repair that rewrites a
    row in place, changing no length, is exactly the swap that must not land."""
    _seed_mixed_ledger(data_root)
    ledger_path = data_root / ua.LEDGER_REL
    rewritten: dict = {}

    def rewrite_after_the_recheck(looks):
        if len(looks) == 2 and looks[-1] and not rewritten:
            raw = ledger_path.read_bytes()
            spot = raw.rindex(b'"attempt_id"') + 20  # inside the id, one byte for one byte
            rewritten["bytes"] = raw[:spot] + bytes([raw[spot] ^ 1]) + raw[spot + 1:]
            ledger_path.write_bytes(rewritten["bytes"])

    _snapshot_looks(monkeypatch, rewrite_after_the_recheck)
    assert _compact(data_root) is None  # the replace was refused
    assert ledger_path.read_bytes() == rewritten["bytes"]  # and nothing was erased


def test_a_hold_lost_at_the_archive_is_seen_before_the_snapshot_is_trusted(
    data_root, monkeypatch
):
    """The pass proves ownership immediately BEFORE each snapshot look: a
    re-check answered while the lock already belongs to someone else is a
    meaningless answer, so the loss must abort before it is even asked."""
    _seed_mixed_ledger(data_root)
    ledger_path = data_root / ua.LEDGER_REL
    before = ledger_path.read_bytes()
    archive_dir = data_root / "archive" / "usage_ledger"
    looks = _snapshot_looks(monkeypatch)

    def heartbeat():
        # Ownership dies at the exact moment the archive segment lands.
        return not list(archive_dir.glob("segment_*.jsonl"))

    with ua._locked(data_root):
        assert uc.compact_usage_ledger_locked(data_root, heartbeat=heartbeat) is None
    assert ledger_path.read_bytes() == before
    assert len(looks) == 1  # the post-archive re-check never ran
    assert list(archive_dir.glob("segment_*.jsonl"))  # the orphan stays, disclosed

def test_a_hold_lost_before_the_first_commit_look_writes_no_orphan(data_root, monkeypatch):
    """The FIRST commit beat — after the archive bound check, before the
    pre-archive snapshot look — is load-bearing too: a hold already lost there
    must abort before that look is asked and before any segment is written,
    or the pass writes an orphan (and trusts an answer it may not use) on
    behalf of a lock that is no longer its own."""
    _seed_mixed_ledger(data_root)
    ledger_path = data_root / ua.LEDGER_REL
    before = ledger_path.read_bytes()
    lost = {"hold": False}
    real_bound = uc._archive_dir_bounded

    def losing_bound(root):
        lost["hold"] = True  # ownership dies as the commit section is entered
        return real_bound(root)

    looks = _snapshot_looks(monkeypatch)
    monkeypatch.setattr(uc, "_archive_dir_bounded", losing_bound)
    with ua._locked(data_root):
        assert uc.compact_usage_ledger_locked(
            data_root, heartbeat=lambda: not lost["hold"]) is None
    assert ledger_path.read_bytes() == before
    assert looks == []  # the pre-archive look was never asked
    assert not (data_root / "archive").exists()  # and no orphan was written


def test_a_hold_lost_after_the_recheck_aborts_before_the_swap(data_root, monkeypatch):
    """Ownership is proven once more between the re-check and the rename: a
    verdict that arrived while ours cannot license a replace that happens
    after the hold left us."""
    _seed_mixed_ledger(data_root)
    ledger_path = data_root / ua.LEDGER_REL
    before = ledger_path.read_bytes()
    looks = _snapshot_looks(monkeypatch)

    def heartbeat():
        # Ownership dies the moment the pre-swap re-check has answered.
        return len(looks) < 2

    with ua._locked(data_root):
        assert uc.compact_usage_ledger_locked(data_root, heartbeat=heartbeat) is None
    assert ledger_path.read_bytes() == before  # no swap: byte-identical
    assert not any(row.get("kind") == "usage_baseline" for row in _ledger_rows(data_root))
    assert len(looks) == 2  # the in-swap look is never asked once the hold is gone


def test_a_hold_lost_after_the_last_snapshot_look_refuses_the_rename(data_root, monkeypatch):
    """The in-swap look is not the last proof: ownership is proven again AFTER
    it answered, immediately before the rename, so the only interval a robbery
    can slip through is the rename syscall itself — not the milliseconds of a
    full-file compare. A hold lost the moment that look answered True, the new
    holder's charge landing right behind it, refuses the replace."""
    _seed_mixed_ledger(data_root)
    before_money = _decimal_money(_ledger_rows(data_root))
    injected: dict = {}
    looks = _snapshot_looks(monkeypatch)

    def heartbeat():
        if len(looks) < 3:
            return True
        if not injected:  # lands after the last look answered, before the rename
            injected.update(_append_raw_row(data_root, _raced_row("sess-after-last-look")))
        return False

    with ua._locked(data_root):
        assert uc.compact_usage_ledger_locked(data_root, heartbeat=heartbeat) is None
    _charge_survived(data_root, injected, before_money)


def test_a_hold_lost_while_the_temp_is_written_refuses_the_replace(data_root):
    """Writing and fsyncing the candidate temp can take arbitrarily long, so
    an ownership proof taken before the swap began is stale by the rename.
    The proof therefore lives INSIDE the atomic writer — once the temp bytes
    are durable, immediately before the snapshot look and the rename — and it
    runs FIRST: a hold lost while the temp was written refuses the replace,
    and a charge the new holder appended survives byte-for-byte."""
    _seed_mixed_ledger(data_root)
    before_money = _decimal_money(_ledger_rows(data_root))
    ledger_path = data_root / ua.LEDGER_REL
    injected: dict = {}

    def heartbeat():
        # Ownership dies while the candidate temp is being written: the first
        # proof asked with the temp on disk answers False — and the new
        # holder's charge lands with it, exactly the row a rename would erase.
        if not list(ledger_path.parent.glob(f".{ledger_path.name}.tmp.*")):
            return True
        if not injected:
            injected.update(_append_raw_row(data_root, _raced_row("sess-new-holder")))
        return False

    with ua._locked(data_root):
        assert uc.compact_usage_ledger_locked(data_root, heartbeat=heartbeat) is None
    _charge_survived(data_root, injected, before_money)


@pytest.mark.parametrize("intrusion", ("append", "hold_lost"))
def test_a_refused_rename_re_proves_the_hold_and_the_snapshot_before_retrying(
    data_root, monkeypatch, intrusion
):
    """The atomic replace retries a refused rename (a Windows sharing
    violation), and a proof taken before the refused attempt is stale by the
    next one: between the attempts a charge can land or the hold can leave us.
    Ownership and the snapshot are re-proven before EVERY attempt, so a
    refusal followed by either intrusion never reaches a second rename, and
    the landed row survives byte-for-byte."""
    _seed_mixed_ledger(data_root)
    before_money = _decimal_money(_ledger_rows(data_root))
    ledger_path = data_root / ua.LEDGER_REL
    before = ledger_path.read_bytes()
    real_replace = os.replace
    attempts: list = []
    state = {"owned": True}
    injected: dict = {}

    def refusing_replace(src, dst):
        if pathlib.Path(dst) != ledger_path:
            return real_replace(src, dst)
        attempts.append(1)
        if len(attempts) == 1:
            if intrusion == "append":
                injected.update(_append_raw_row(data_root, _raced_row("sess-between-attempts")))
            else:
                state["owned"] = False
            raise PermissionError(errno.EACCES, "injected sharing violation")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", refusing_replace)
    with ua._locked(data_root) as heartbeat:
        assert uc.compact_usage_ledger_locked(
            data_root, heartbeat=lambda: state["owned"] and heartbeat()) is None
    assert attempts == [1]  # the second rename never happened
    if intrusion == "append":
        _charge_survived(data_root, injected, before_money)
    else:
        assert ledger_path.read_bytes() == before
        assert not list(ledger_path.parent.glob(f".{ledger_path.name}.tmp.*"))


def test_the_pass_refuses_on_the_name_tier_while_appends_continue(data_root_any_tier, tmp_path, monkeypatch, caplog):
    """Where the lock directory takes no kernel locks the monetary lock is a name protocol
    only — exclusion the pass cannot prove — so compaction refuses (disclosed reason, ledger
    byte-identical) while appends continue under the name protocol; the refusal is durable
    and typed — ONE usage_ledger_compaction_refused event per process per data root."""
    _seed_mixed_ledger(data_root_any_tier)
    ledger_path = data_root_any_tier / ua.LEDGER_REL
    before = ledger_path.read_bytes()
    monkeypatch.setattr(
        platform_layer, "kernel_file_locks_enforced", lambda path: False, raising=False)
    # The FIRST refusals cannot write their event (a full disk, an unwritable
    # logs/) — append_jsonl reports exhausted retries as False, not only as an
    # exception. Neither is "already told": marking the root before the row lands
    # downgrades the durable event to a log line for the life of the process.
    landing_append = uc.append_jsonl
    monkeypatch.setattr(uc, "append_jsonl", lambda *a, **k: (_ for _ in ()).throw(OSError("no space left")))
    with caplog.at_level(logging.WARNING, logger="ouroboros.usage_compaction"):
        assert _compact(data_root_any_tier) is None
    monkeypatch.setattr(uc, "append_jsonl", lambda *a, **k: False)
    assert _compact(data_root_any_tier) is None
    monkeypatch.setattr(uc, "append_jsonl", landing_append)
    assert ledger_path.read_bytes() == before
    assert not (data_root_any_tier / "archive").exists()
    assert any("name tier" in record.getMessage() for record in caplog.records)
    # The reserve-path trigger refuses the same way, and the append it serves lands.
    monkeypatch.setattr("ouroboros.config.USAGE_LEDGER_COMPACT_BYTES", 1)
    uc._COMPACT_ATTEMPTS.clear()
    _settle(data_root_any_tier, cost=0.5, cost_final=True)
    rows = _ledger_rows(data_root_any_tier)
    assert not any(row.get("kind") == "usage_baseline" for row in rows)
    assert len(rows) == len(before.splitlines()) + 3  # reserved, dispatched, settled
    # Two refusals in this process, ONE durable typed row: an operator (and the
    # 20 MB tripwire) can tell the tier apart from "nothing foldable".
    def refusals():
        events = (data_root_any_tier / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        return [row for row in map(json.loads, events) if row.get("type") == "usage_ledger_compaction_refused"]

    assert [row["reason"] for row in refusals()] == ["name_tier"]  # the one that could land
    spelling = tmp_path / "linked-data"  # ~/Ouroboros -> ~/ouro: one root, two names
    spelling.symlink_to(data_root_any_tier)
    assert _compact(spelling) is None
    assert len(refusals()) == 1  # ONE data root, however it is spelled
    ledger_note = next(note for rel, _, note in _hot_store_thresholds() if rel == "state/usage_attempts.jsonl")
    assert "name tier" in ledger_note and "usage_ledger_compaction_refused" in ledger_note


# --- 6: idempotent kinds never fold ------------------------------------------

def test_subscription_replay_still_dedups_after_compaction(data_root, compacted):
    before = len(_ledger_rows(data_root))
    attempt_id = ua.record_subscription_session(
        "sess-1", drive_root=data_root, route="claudexor:claude", model="fable",
        task_id="t5", root_task_id="root", spend_usd=0.5, reset_at="2026-09-02T00:00:00Z")
    assert attempt_id.startswith("session-")
    assert len(_ledger_rows(data_root)) == before  # no duplicate row
    with pytest.raises(ua.UsageAccountingError):
        ua.record_subscription_session("sess-1", drive_root=data_root, route="other-route", model="fable",
                                       task_id="t5", root_task_id="root", spend_usd=0.5)
    before = len(_ledger_rows(data_root))
    ua.record_unmetered_external_dispatch("ext-1", drive_root=data_root, model="ext-model", task_id="t6",
                                          prompt_tokens=7, completion_tokens=3)
    assert len(_ledger_rows(data_root)) == before


def test_legacy_import_rows_are_retained_and_reimport_dedups(data_root):
    events = data_root / "logs" / "events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(json.dumps({
        "type": "llm_usage", "model": "m", "provider": "openai", "cost": 0.7,
        "prompt_tokens": 5, "completion_tokens": 2, "task_id": "lt",
    }) + "\n", encoding="utf-8")
    (data_root / ua.IMPORT_REL).unlink()
    ua.ensure_legacy_imported(data_root)
    legacy_ids = {
        str(row["attempt_id"]) for row in _ledger_rows(data_root)
        if str(row.get("kind", "")).startswith("legacy_")
    }
    assert legacy_ids
    _settle(data_root, cost=0.9, cost_final=True)
    assert _compact(data_root) is not None
    live_ids = {str(row["attempt_id"]) for row in _ledger_rows(data_root)}
    assert legacy_ids <= live_ids  # never folded
    # Watermark loss: the resumable import replays against the LIVE ledger and
    # appends nothing new.
    before = len(_ledger_rows(data_root))
    (data_root / ua.IMPORT_REL).unlink()
    result = ua.ensure_legacy_imported(data_root)
    assert result["rows_appended"] == 0
    assert len(_ledger_rows(data_root)) == before


# --- 7: trigger policy -------------------------------------------------------

def test_reserve_path_compacts_only_past_config_threshold(data_root, monkeypatch):
    _seed_mixed_ledger(data_root)
    # The pass rewrites the whole monetary authority, so it is correct ONLY
    # while the ledger lock is held: prove the hold at the moment of the call
    # rather than trusting where the call site sits.
    holds: list = []
    renewed: list = []
    original = uc.compact_usage_ledger_locked

    def observing(root, **kwargs):
        probe = platform_layer.acquire_exclusive_file_lock(  # None: held by someone else
            _lock_path(data_root), timeout_sec=0.05, stale_sec=3600.0, poll_sec=0.01,
            owner_aware_stale=True,
        )
        # ... and the hold is WIRED THROUGH: without the heartbeat every ownership proof inside the
        # pass is a no-op and the swap runs unproven; pinned at the only production caller, to THIS lock.
        os.utime(_lock_path(data_root), (0.0, 0.0))
        renewed.append(kwargs["heartbeat"]() is True and _lock_path(data_root).stat().st_mtime > time.time() - 60)
        holds.append(probe is None)
        if probe is not None:
            platform_layer.release_exclusive_file_lock(_lock_path(data_root), probe)
        return original(root, **kwargs)

    monkeypatch.setattr(uc, "compact_usage_ledger_locked", observing)
    size = (data_root / ua.LEDGER_REL).stat().st_size
    monkeypatch.setattr("ouroboros.config.USAGE_LEDGER_COMPACT_BYTES", size * 10)
    uc._COMPACT_ATTEMPTS.clear()
    _settle(data_root, cost=0.1, cost_final=True)
    assert not any(row.get("kind") == "usage_baseline" for row in _ledger_rows(data_root))
    assert holds == []  # below threshold: the stat fast-path never enters the pass
    monkeypatch.setattr("ouroboros.config.USAGE_LEDGER_COMPACT_BYTES", 1)
    _settle(data_root, cost=0.1, cost_final=True)
    assert renewed == [True], "a stub, not the held lock's heartbeat"
    assert holds == [True]
    assert any(row.get("kind") == "usage_baseline" for row in _ledger_rows(data_root))


def test_unprofitable_pass_is_throttled(data_root, monkeypatch):
    # Only in-flight rows: nothing foldable, compaction must not thrash.
    for task in ("a", "b"):
        reservation = ua.reserve_attempt(_request(data_root, task_id=task))
        ua.mark_dispatched(reservation)
    monkeypatch.setattr("ouroboros.config.USAGE_LEDGER_COMPACT_BYTES", 1)
    monkeypatch.setattr(
        "ouroboros.config.USAGE_LEDGER_COMPACT_RETRY_GROWTH_BYTES", 10_000_000)
    uc._COMPACT_ATTEMPTS.clear()
    calls = []
    original = uc.compact_usage_ledger_locked

    def counting(root, **kwargs):
        calls.append(1)
        return original(root, **kwargs)

    monkeypatch.setattr(uc, "compact_usage_ledger_locked", counting)
    before = _ledger_lines(data_root)
    with ua._locked(data_root) as heartbeat:
        assert uc.maybe_compact_usage_ledger_locked(data_root, heartbeat=heartbeat) is False
    assert _ledger_lines(data_root) == before  # nothing foldable -> no-op
    with ua._locked(data_root) as heartbeat:
        assert uc.maybe_compact_usage_ledger_locked(data_root, heartbeat=heartbeat) is False
    assert len(calls) == 1  # second call throttled by the growth guard


def test_verify_abort_on_foreign_noncanonical_literal(data_root):
    _settle(data_root, cost=1.0, cost_final=True)
    ua.record_subscription_session("sess-nc", drive_root=data_root, route="claudexor:claude", model="fable",
                                   task_id="t", root_task_id="root", spend_usd=0.5)
    path = data_root / ua.LEDGER_REL
    lines = _ledger_lines(data_root)
    # A RETAINED row (subscription kind never folds) whose monetary literal is
    # NOT double-round-trippable: a foreign writer's long literal. Its value
    # survives as a double, but the exact decimal cannot be re-serialized, so
    # the pass must abort and leave the ledger untouched. (Folded rows are
    # immune by construction: their decimals are carried as exact strings.)
    doctored = lines[-1].replace('"cost_usd":0.5', '"cost_usd":0.50000000000000002775557561563')
    assert doctored != lines[-1]
    path.write_text("\n".join(lines[:-1] + [doctored]) + "\n", encoding="utf-8")
    before_bytes = path.read_bytes()
    assert _compact(data_root) is None
    assert path.read_bytes() == before_bytes
