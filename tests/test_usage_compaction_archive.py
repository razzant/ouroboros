"""CPL4-C6 pins: the archive reader of the compacted monetary ledger.

Design contract: docs/v7next/DESIGN_USAGE_COMPACTION.md. The invariants
pinned here are the reader-side half of the same monetary-authority set
(owner sanction 1A):

5. every pre-compaction attempt_id stays resolvable (live ∪ archive; the CPL-5 join) across chained compactions, tamper-evident;
8. baseline rows are legal only as the leading block.

The pass side — invariants 1, 2, 3, 4, 6 and 7 — lives in
``tests/test_usage_compaction.py``; the fixtures both modules share live in
``tests/fixtures_usage_compaction.py``.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import pathlib
import shutil
import signal
import time

import pytest

from ouroboros import platform_layer
from ouroboros import usage_accounting as ua
from ouroboros import usage_compaction as uc
from ouroboros.usage_ledger import UsageLedgerCorrupt, _validate_records
from tests import fixtures_usage_compaction as _fixtures
from tests.fixtures_usage_compaction import (
    _append_raw_row,
    _compact,
    _ledger_rows,
    _raced_row,
    _seed_mixed_ledger,
    _settle,
)

# The three fixtures are re-exported through the module object rather than
# imported by name: a test's ``data_root`` parameter would shadow a bare
# import of the same name. pytest resolves them from this module either way.
compacted = _fixtures.compacted
data_root = _fixtures.data_root
data_root_any_tier = _fixtures.data_root_any_tier


def _rewrite_header(data_root, header):
    """Replace the live ledger's leading row (tamper simulation)."""
    path = data_root / ua.LEDGER_REL
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps(header, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    uc._SEGMENT_CACHE.clear()
    uc._CHAIN_UNION_CACHE.clear()


# --- 5: CPL-5 join surface ---------------------------------------------------

def test_every_attempt_id_stays_resolvable_across_chained_compactions(data_root):
    _seed_mixed_ledger(data_root)
    first_ids = {str(row["attempt_id"]) for row in _ledger_rows(data_root)}
    assert _compact(data_root) is not None
    # Second generation of traffic, second compaction.
    _settle(data_root, cost=0.75, cost_final=True, task_id="gen2")
    second_ids = {str(row["attempt_id"]) for row in _ledger_rows(data_root)}
    assert _compact(data_root) is not None

    live_ids = {str(row["attempt_id"]) for row in _ledger_rows(data_root)}
    archived = uc.archived_attempt_ids(data_root)
    for attempt_id in first_ids | second_ids:
        assert attempt_id in live_ids or attempt_id in archived
        assert uc.usage_attempt_recorded(data_root, attempt_id)
    assert not uc.usage_attempt_recorded(data_root, "never-recorded")


def test_tampered_archive_segment_is_detected(data_root, compacted):
    _, segment = compacted
    payload = segment.read_bytes()
    segment.write_bytes(payload.replace(b'"settled"', b'"sett1ed"', 1))
    with pytest.raises(UsageLedgerCorrupt):
        uc.archived_attempt_ids(data_root)


def test_rehashed_segment_still_fails_the_ledger_structure(data_root, compacted):
    """A tamperer who also repairs the hash still has to produce a LEDGER."""
    header, segment = compacted
    forged = segment.read_bytes().replace(b'"seq":2', b'"seq":9', 1)
    assert len(forged) == header["source_size_bytes"]
    segment.write_bytes(forged)
    _rewrite_header(data_root, {
        **header, "source_sha256": hashlib.sha256(forged).hexdigest(),
    })
    with pytest.raises(UsageLedgerCorrupt):
        uc.archived_attempt_ids(data_root)


def test_warm_segment_cache_revalidates_the_file_it_cached(data_root, compacted):
    _, segment = compacted
    assert uc.archived_attempt_ids(data_root)  # warms the per-segment cache
    segment.unlink()
    with pytest.raises(UsageLedgerCorrupt):
        uc.archived_attempt_ids(data_root)
    segment.write_bytes(b'{"kind":"attempt"}\n')
    with pytest.raises(UsageLedgerCorrupt):
        uc.archived_attempt_ids(data_root)
    segment.unlink()
    segment.mkdir()  # a directory where the header names a segment: typed, never a bare IsADirectoryError
    with pytest.raises(UsageLedgerCorrupt, match="not a regular file|could not complete|cannot be inspected|segment unreadable"):  # POSIX types the fstat; Windows the refused open (rc.2 matrix 33680647341: `segment unreadable`)
        uc.archived_attempt_ids(data_root)


def _rewrite_segment_in_place(segment, marker):
    """Same size, same inode, same mtime: the fingerprint a rewrite can keep."""
    payload = segment.read_bytes()
    forged = payload.replace(b'"settled"', marker, 1)
    assert forged != payload and len(forged) == len(payload)
    info = os.stat(segment)
    with open(segment, "r+b") as handle:
        handle.write(forged)
    os.utime(segment, ns=(info.st_atime_ns, info.st_mtime_ns))


def test_a_rewrite_inside_the_timestamp_window_is_re_hashed_not_recalled(data_root, compacted):
    """Filesystem timestamps have granularity; a file touched about NOW cannot
    prove it is the file that was read, however well the fingerprint matches."""
    _, segment = compacted
    assert uc.archived_attempt_ids(data_root)  # warms the per-segment cache
    _rewrite_segment_in_place(segment, b'"sett1ed"')
    # Deterministically place the file INSIDE the settle window while keeping
    # the cache's fingerprint exact, so only the window can decide this read.
    entry = uc._SEGMENT_CACHE[str(segment)]
    os.utime(segment)
    info = os.stat(segment)
    uc._SEGMENT_CACHE[str(segment)] = entry[:3] + (
        (info.st_ino, info.st_dev, info.st_size, info.st_mtime_ns),
    ) + entry[4:]

    with pytest.raises(UsageLedgerCorrupt):
        uc.archived_attempt_ids(data_root)


def test_a_same_size_rewrite_is_caught_once_the_cache_entry_expires(data_root, compacted):
    """A verified read is evidence with a shelf life, not a standing answer."""
    _, segment = compacted
    settled = time.time() - 3600
    os.utime(segment, (settled, settled))  # well outside the settle window
    assert uc.archived_attempt_ids(data_root)
    _rewrite_segment_in_place(segment, b'"sett1ed"')
    entry = uc._SEGMENT_CACHE[str(segment)]
    uc._SEGMENT_CACHE[str(segment)] = entry[:4] + (entry[4] - 3600,)

    with pytest.raises(UsageLedgerCorrupt):
        uc.archived_attempt_ids(data_root)


# Everything the forger would carry over from the older genuine stamp — the
# MUTABLE epoch included, because a rollback that leaves the epoch behind is
# caught by the chain-step rule alone and proves nothing about this one.
_SOURCE_PROVENANCE_KEYS = (
    "archive_rel", "source_sha256", "source_size_bytes", "source_row_count",
    "source_first_seq", "source_last_seq", "folded_row_count", "retained_row_count",
    "compaction_epoch",
)


def _embedded_header(data_root, header):
    """The previous epoch's header, as embedded in the segment ``header`` names."""
    first = (data_root / header["archive_rel"]).read_text(
        encoding="utf-8").splitlines()[0]
    return json.loads(first)


def test_repointing_the_header_at_an_older_segment_is_corrupt(data_root, compacted):
    """Dropping the epochs between is a shortened chain, not a shorter history."""
    for generation in ("gen2", "gen3"):
        _settle(data_root, cost=0.75, cost_final=True, task_id=generation)
        assert _compact(data_root) is not None
    header3 = _ledger_rows(data_root)[0]
    assert header3["compaction_epoch"] == 3
    everything = uc.archived_attempt_ids(data_root)
    # Each older header is embedded, verbatim and correctly hashed, as the
    # first row of the segment the newer one names — genuine references, all.
    header2 = _embedded_header(data_root, header3)
    header1 = _embedded_header(data_root, header2)
    assert (header2["compaction_epoch"], header1["compaction_epoch"]) == (2, 1)
    for skipped_to in (header2, header1):
        forged = {**header3, **{key: skipped_to[key] for key in _SOURCE_PROVENANCE_KEYS}}
        _rewrite_header(data_root, forged)
        _validate_records(_ledger_rows(data_root))  # structurally impeccable
        with pytest.raises(UsageLedgerCorrupt):
            uc.archived_attempt_ids(data_root)
    assert everything  # what the forgeries were trying to make disappear


def test_an_orphan_segment_of_the_live_generation_is_not_a_rollback(data_root, monkeypatch, compacted):
    """A pass that lost the snapshot race leaves a segment for an epoch that
    never committed. It holds THIS generation's bytes, so the archive still
    anchors the live stamp and the history stays readable."""
    known = uc.archived_attempt_ids(data_root)
    original_write = uc._write_new_file_fsync

    def racing_write(path, payload, root):
        _append_raw_row(data_root, _raced_row("sess-orphaned"))
        original_write(path, payload, root)

    monkeypatch.setattr(uc, "_write_new_file_fsync", racing_write)
    _settle(data_root, cost=0.75, cost_final=True, task_id="gen2")
    assert _compact(data_root) is None  # refused the swap; the segment stays
    segments = sorted((data_root / "archive" / "usage_ledger").glob("*.jsonl"))
    assert len(segments) == 2  # one referenced, one orphan of THIS generation
    uc._SEGMENT_CACHE.clear()
    uc._CHAIN_UNION_CACHE.clear()
    assert uc.archived_attempt_ids(data_root) == known


@pytest.mark.parametrize("restored", ("stamped", "unstamped"))
def test_a_restored_previous_generation_is_out_anchored_not_taken_for_an_orphan(data_root, monkeypatch, restored):
    """A ledger restored from a backup taken before the last compaction leaves
    that pass's segment on disk holding every attempt it folded — ids that
    exist nowhere else — and its leading row IS the restored file's, so
    matching that row admitted it as an orphan and the join answered a
    strictly smaller set: silent absence, the one verdict this surface may
    never reach. An orphan is the pre-swap COPY of the live file, which only
    grows behind it: its bytes stay a PREFIX, a restored generation's do not —
    compared from the descriptor that was classified, never from a re-open of
    the name. No stamp at all is the same question with the floor at zero: the
    archive contradicts the missing stamp instead of never being read."""
    _seed_mixed_ledger(data_root)
    generations = [(data_root / ua.LEDGER_REL).read_bytes()]
    assert _compact(data_root) is not None
    generations.append((data_root / ua.LEDGER_REL).read_bytes())
    _settle(data_root, cost=0.75, cost_final=True, task_id="gen2")
    folded_by_the_second_pass = {row["attempt_id"] for row in _ledger_rows(data_root) if row.get("attempt_id")}
    assert _compact(data_root) is not None
    assert folded_by_the_second_pass <= uc.archived_attempt_ids(data_root)  # resolvable while intact

    (data_root / ua.LEDGER_REL).write_bytes(generations[restored == "stamped"])
    uc._SEGMENT_CACHE.clear()
    uc._CHAIN_UNION_CACHE.clear()
    opened: set = set()  # a name opened AGAIN answers an empty file: the entry swapped in between
    real_open = uc._open_archive_entry
    monkeypatch.setattr(uc, "_open_archive_entry", lambda path, dir_fd: os.open(os.devnull, os.O_RDONLY)
                        if path.name in opened else (opened.add(path.name) or real_open(path, dir_fd)))
    with pytest.raises(UsageLedgerCorrupt, match="generation newer"):
        uc.archived_attempt_ids(data_root)


@pytest.mark.skipif(
    platform_layer.IS_WINDOWS,
    reason="the held dir-fd scan is POSIX; Windows keeps the path-based scan, fail-closed",
)
def test_the_epoch_anchor_scans_the_directory_the_chain_was_walked_in(data_root, monkeypatch, compacted):
    """The chain walk and the anchor scan must look at the SAME directory: a
    directory swapped between them (renamed away, a look-alike put in its
    place) hides the newer generation from a path-based scan and admits a
    forged rollback. Listing through the held dir-fd is only half of it: the
    entries must be OPENED relative to that handle too, or a look-alike that
    carries the epoch-3 NAME with the forged live header as its leading row
    is admitted by the orphan exemption. The renamed directory is still the
    one the handle names, so the real epoch-3 segment is what gets read."""
    for generation in ("gen2", "gen3"):
        _settle(data_root, cost=0.75, cost_final=True, task_id=generation)
        assert _compact(data_root) is not None
    header3 = _ledger_rows(data_root)[0]
    header2 = _embedded_header(data_root, header3)
    forged = {**header3, **{key: header2[key] for key in _SOURCE_PROVENANCE_KEYS}}
    _rewrite_header(data_root, forged)
    archive_dir = data_root / "archive" / "usage_ledger"
    hidden = archive_dir.with_name("usage_ledger.hidden")
    seg3_name = pathlib.PurePosixPath(header3["archive_rel"]).name
    real_anchor = uc._no_newer_archived_epoch

    def swapping_anchor(*args, **kwargs):
        # After the walk, before the anchor: the real directory goes away and a
        # look-alike takes its name — the older generations copied, and under
        # the epoch-3 NAME a segment whose leading row is the forged live
        # header, which the orphan exemption admits if the scan opens by path.
        archive_dir.rename(hidden)
        archive_dir.mkdir()
        for segment in hidden.glob("segment_ep000[12]_*.jsonl"):
            shutil.copy2(segment, archive_dir / segment.name)
        body = (hidden / seg3_name).read_bytes().split(b"\n", 1)[1]
        (archive_dir / seg3_name).write_bytes(json.dumps(forged, sort_keys=True).encode() + b"\n" + body)
        return real_anchor(*args, **kwargs)

    monkeypatch.setattr(uc, "_no_newer_archived_epoch", swapping_anchor)
    with pytest.raises(UsageLedgerCorrupt, match="generation newer"):
        uc.archived_attempt_ids(data_root)


@pytest.mark.skipif(
    platform_layer.IS_WINDOWS,
    reason="a dangling link, a FIFO and O_NONBLOCK are POSIX shapes; Windows keeps the path-based scan",
)
@pytest.mark.parametrize("held_dir_fd", (True, False))
def test_an_archive_entry_the_anchor_cannot_open_is_typed_corruption(data_root, monkeypatch, held_dir_fd):
    """An entry the anchor cannot open or read is not "no evidence": the scan
    did not complete, so the question is UNKNOWN (typed), never an answer built
    on the part of the archive that could be read. A directory or a writer-less
    FIFO is no segment (segments are regular files by construction): classified
    and skipped without hanging the open, on BOTH scan shapes — without the held
    dir-fd the entry is classified BEFORE the open, the step a directory refuses
    on Windows and a FIFO blocks on. The data root's own handle is typed the
    same way: a bare OSError from THAT open would escape the sweep's UNKNOWN mapping."""
    monkeypatch.setattr(uc, "_dir_fd_capable", lambda: held_dir_fd)
    _seed_mixed_ledger(data_root)
    assert _compact(data_root) is not None
    known = uc.archived_attempt_ids(data_root)
    archive_dir = data_root / "archive" / "usage_ledger"
    (archive_dir / "backup").mkdir()
    os.mkfifo(archive_dir / "segment_ep0009_fifo.jsonl")
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("FIFO open blocked")))
    signal.alarm(5)
    try:
        assert uc.archived_attempt_ids(data_root) == known
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    if held_dir_fd:  # the root handle is opened only on the dir-fd shape
        data_root.chmod(0o111)  # traversable, unreadable: fd exhaustion reads the same
        try:
            with pytest.raises(UsageLedgerCorrupt):
                uc.archived_attempt_ids(data_root)
        finally:
            data_root.chmod(0o755)
    planted = archive_dir / "segment_ep0009_planted.jsonl"
    planted.symlink_to(data_root / "nowhere.jsonl")  # dangling: unopenable either way
    with pytest.raises(UsageLedgerCorrupt, match="could not complete"):
        uc.archived_attempt_ids(data_root)


def test_pre_compaction_seq_must_name_a_row_the_named_source_held(data_root, compacted):
    """The claim is provenance about an archived range, not a free number."""
    rows = _ledger_rows(data_root)
    header = rows[0]
    carriers = [index for index, row in enumerate(rows) if "pre_compaction_seq" in row]
    assert carriers
    _validate_records([dict(row) for row in rows])  # control: the honest file
    forged = [dict(row) for row in rows]
    # Strictly increasing, so only the source range itself refuses it.
    forged[carriers[-1]]["pre_compaction_seq"] = header["source_last_seq"] + 1
    with pytest.raises(UsageLedgerCorrupt):
        _validate_records(forged)


def test_archive_reference_is_bounded_to_the_archive_directory(data_root, compacted):
    rows = _ledger_rows(data_root)
    header = rows[0]
    for archive_rel in (
        "../../other.jsonl",
        "/etc/passwd",
        "archive/usage_ledger/../../other.jsonl",
        "archive/other/segment.jsonl",
        "archive\\usage_ledger\\segment.jsonl",
        "",
    ):
        with pytest.raises(UsageLedgerCorrupt):
            _validate_records([{**header, "archive_rel": archive_rel}, *rows[1:]])
    # And the reader refuses it even when the named file exists and hashes right.
    outside = data_root / "outside.jsonl"
    outside.write_bytes((data_root / header["archive_rel"]).read_bytes())
    _rewrite_header(data_root, {**header, "archive_rel": "../outside.jsonl"})
    with pytest.raises(UsageLedgerCorrupt):
        uc.archived_attempt_ids(data_root)


@pytest.mark.parametrize("level", ("archive", "archive/usage_ledger"))
def test_a_symlinked_archive_path_is_refused_by_writer_and_reader(data_root, tmp_path, level, compacted):
    """The archive directory must BE inside the data root, not a door to
    somewhere else: with a link at either level the segment and the directory
    resolve THROUGH the same link, so "next to the archive" proves nothing."""
    ledger_path = data_root / ua.LEDGER_REL
    target = data_root / level
    elsewhere = tmp_path / "elsewhere"
    shutil.move(str(target), str(elsewhere))  # same segments, same hashes
    target.symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(UsageLedgerCorrupt):
        uc.archived_attempt_ids(data_root)

    _settle(data_root, cost=0.5, cost_final=True, task_id="gen2")
    before = ledger_path.read_bytes()
    assert _compact(data_root) is None  # the writer refuses to feed it
    assert ledger_path.read_bytes() == before


@pytest.mark.skipif(platform_layer.IS_WINDOWS, reason="dir-fd anchoring is POSIX; Windows is a disclosed best effort")
def test_a_link_planted_after_the_writer_bound_check_cannot_receive_history(
    data_root, tmp_path, monkeypatch, compacted
):
    """The bound check and the write are not one instant: on POSIX the writer
    creates the segment through O_NOFOLLOW dir-fd handles, so a link swapped
    in AFTER the check passed still cannot receive monetary history."""
    _settle(data_root, cost=0.5, cost_final=True, task_id="gen2")
    ledger_path = data_root / ua.LEDGER_REL
    archive = data_root / "archive"
    elsewhere = tmp_path / "elsewhere"
    real_bound = uc._archive_dir_bounded

    def racing_bound(root):
        result = real_bound(root)
        if not archive.is_symlink():
            shutil.move(str(archive), str(elsewhere))
            archive.symlink_to(elsewhere, target_is_directory=True)
        return result

    monkeypatch.setattr(uc, "_archive_dir_bounded", racing_bound)
    before = ledger_path.read_bytes()

    assert _compact(data_root) is None  # the pass aborts at the write itself
    assert ledger_path.read_bytes() == before
    linked = list((elsewhere / "usage_ledger").glob("segment_*.jsonl"))
    assert len(linked) == 1  # only the pre-existing segment: nothing crossed the link


@pytest.mark.skipif(platform_layer.IS_WINDOWS, reason="dir-fd anchoring is POSIX; Windows is a disclosed best effort")
def test_a_link_planted_after_the_reader_bound_check_is_refused(
    data_root, tmp_path, monkeypatch, compacted
):
    """A byte-identical copy behind a planted link hashes perfectly — the
    only defense is that the read itself refuses to traverse a link, which
    the O_NOFOLLOW dir-fd open enforces at the open, not at an earlier look."""
    _, segment = compacted
    copy = tmp_path / "copy.jsonl"
    copy.write_bytes(segment.read_bytes())  # identical bytes: the hash cannot object
    real_path = uc._segment_path

    def racing_path(root, rel):
        result = real_path(root, rel)
        if not segment.is_symlink():
            segment.unlink()
            segment.symlink_to(copy)
        return result

    monkeypatch.setattr(uc, "_segment_path", racing_path)

    with pytest.raises(UsageLedgerCorrupt):
        uc.archived_attempt_ids(data_root)


@pytest.mark.skipif(platform_layer.IS_WINDOWS or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="permission shapes are POSIX and need a non-root euid")
def test_a_stamp_less_ledger_still_inspects_its_archive_fail_closed(data_root):
    """Before any compaction the question ends early ONLY on the kernel's exact "no
    archive directory": a regular file standing where the directory belongs or a
    directory that cannot be inspected is UNKNOWN — typed, never silent or bare."""
    _seed_mixed_ledger(data_root)
    assert uc.archived_attempt_ids(data_root) == frozenset()  # control: no archive at all
    archive_dir = data_root / "archive" / "usage_ledger"
    archive_dir.parent.mkdir(exist_ok=True)
    archive_dir.write_bytes(b"")  # a regular file where the archive directory belongs
    with pytest.raises(UsageLedgerCorrupt, match="not our own directory"):
        uc.archived_attempt_ids(data_root)
    archive_dir.unlink()
    archive_dir.mkdir()
    archive_dir.parent.chmod(0o000)  # present, but no inspection allowed
    try:
        with pytest.raises(UsageLedgerCorrupt, match="cannot be inspected"):
            uc.archived_attempt_ids(data_root)
    finally:
        archive_dir.parent.chmod(0o755)


def test_a_path_inspection_the_reader_cannot_make_is_typed_corruption(data_root, compacted, monkeypatch):
    """pathlib re-raises every OSError but ENOENT/ENOTDIR/EBADF/ELOOP and turns a
    symlink LOOP into RuntimeError, so the reader's bounds — both archive levels, the
    named segment, its resolution — must type EACCES/EIO/a loop themselves or a bare
    error escapes the CPL-5 sweep's UNKNOWN mapping. Real shape first: a segment directory readable but not searchable."""
    _, segment = compacted
    if not (platform_layer.IS_WINDOWS or getattr(os, "geteuid", lambda: 1)() == 0):
        segment.parent.chmod(0o600)
        try:
            with pytest.raises(UsageLedgerCorrupt, match="cannot be inspected"):
                uc.archived_attempt_ids(data_root)
        finally:
            segment.parent.chmod(0o755)
    loop = segment.with_name(segment.name[:-14] + "deadbeef.jsonl")  # a self-loop where a segment name is legal:
    loop.symlink_to(loop.name)  # Path.resolve(strict=False) raised RuntimeError("Symlink loop") through the reader
    _rewrite_header(data_root, {**_ledger_rows(data_root)[0], "archive_rel": loop.relative_to(data_root).as_posix()})
    with pytest.raises(UsageLedgerCorrupt):
        uc.archived_attempt_ids(data_root)
    monkeypatch.setattr(pathlib.Path, "is_symlink", lambda self: (_ for _ in ()).throw(PermissionError(errno.EACCES, "refused")))
    with pytest.raises(UsageLedgerCorrupt, match="cannot be inspected"):
        uc.archived_attempt_ids(data_root)


def test_unreadable_leading_row_is_typed_corruption_not_absence(data_root, compacted):
    folded = sorted(uc.archived_attempt_ids(data_root))
    assert folded
    path = data_root / ua.LEDGER_REL
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(['{"kind": "usage_baseline"'] + lines[1:]) + "\n",
                    encoding="utf-8")
    with pytest.raises(UsageLedgerCorrupt):
        uc.archived_attempt_ids(data_root)
    # The CPL-5 join must reach UNKNOWN, never "no attempt row" (orphan seal).
    with pytest.raises(UsageLedgerCorrupt):
        uc.usage_attempt_recorded(data_root, folded[0], live_ids=set())


def test_archived_id_union_is_built_once_per_chain(data_root, monkeypatch, compacted):
    _settle(data_root, cost=0.25, cost_final=True, task_id="gen2")
    assert _compact(data_root) is not None
    folded = sorted(uc.archived_attempt_ids(data_root))
    assert len(folded) > 4
    builds: list = []
    original = uc._union_segment_ids

    def counting(segment_ids):
        builds.append(len(segment_ids))
        return original(segment_ids)

    monkeypatch.setattr(uc, "_union_segment_ids", counting)
    uc._CHAIN_UNION_CACHE.clear()
    # A reverse sweep asks once per seal; the union over the chain is chain
    # work, not per-question work.
    for attempt_id in folded:
        assert uc.usage_attempt_recorded(data_root, attempt_id, live_ids=set())
    assert builds == [2]
    # The key IS the chain, so it changes at every compaction: unbounded, a
    # long-lived process would keep one archived-id set per epoch it ever saw.
    monkeypatch.setattr(uc, "_CHAIN_UNION_CACHE_MAX", 1)
    _settle(data_root, cost=0.25, cost_final=True, task_id="gen3")
    assert _compact(data_root) is not None
    assert uc.archived_attempt_ids(data_root)
    assert len(uc._CHAIN_UNION_CACHE) == 1  # only the newest chain can be asked again


# --- 8: structural validation ------------------------------------------------

def test_baseline_rows_are_rejected_outside_the_leading_block(data_root_any_tier):
    _settle(data_root_any_tier, cost=1.0, cost_final=True)
    rows = _ledger_rows(data_root_any_tier)
    smuggled = {
        "kind": "usage_baseline_group", "attempt_id": "baseline-x-g0001",
        "state": "settled", "seq": len(rows) + 1, "ts": "2026-09-01T00:00:00Z",
        "baseline_id": "x", "folded_attempt_count": 2, "model": "m",
        "provider": "p", "category": "task", "source": "llm", "task_id": "t",
        "root_task_id": "t", "parent_task_id": "", "cost_usd": "1.0",
        "cost_final": True,
    }
    with pytest.raises(UsageLedgerCorrupt):
        _validate_records([*rows, smuggled])


def test_a_group_row_cannot_rejoin_the_block_after_it_closed(data_root, compacted):
    """The money-injection shape: a real group row, real ``baseline_id``, later."""
    rows = _ledger_rows(data_root)
    template = next(row for row in rows if row.get("kind") == "usage_baseline_group")
    smuggled = {
        **template,
        "attempt_id": f"{template['attempt_id']}-dup",
        "seq": len(rows) + 1,
    }
    with pytest.raises(UsageLedgerCorrupt):
        _validate_records([*rows, smuggled])


def test_baseline_header_is_rejected_by_POSITION_not_by_shape(data_root, compacted):
    rows = _ledger_rows(data_root)
    header = rows[0]
    groups = [row for row in rows if row.get("kind") == "usage_baseline_group"]
    assert groups
    # Control: this exact, unmodified block IS legal at the head of a file.
    _validate_records([header, *groups])
    # The SAME rows, changed in nothing but where they sit, are corrupt.
    displaced = [
        {"kind": "attempt", "attempt_id": "displacer", "state": "reserved",
         "seq": 1, "ts": header["ts"]},
        {**header, "seq": 2},
        *[{**group, "seq": 3 + offset} for offset, group in enumerate(groups)],
    ]
    with pytest.raises(UsageLedgerCorrupt):
        _validate_records(displaced)


def test_baseline_header_counts_must_close(data_root, compacted):
    rows = _ledger_rows(data_root)
    header = rows[0]
    _validate_records(rows)
    for field, value in (
        ("compaction_epoch", 0),
        ("source_first_seq", 2),
        ("source_last_seq", header["source_last_seq"] + 1),
        ("folded_row_count", header["folded_row_count"] + 1),
        ("retained_row_count", header["retained_row_count"] + 1),
        ("group_count", header["group_count"] + 1),
        ("folded_attempt_count", header["folded_attempt_count"] + 1),
        ("source_sha256", "not-a-digest"),
        ("source_size_bytes", 0),
    ):
        with pytest.raises(UsageLedgerCorrupt):
            _validate_records([{**header, field: value}, *rows[1:]])


def test_pre_compaction_seq_is_a_checked_provenance_claim(data_root, compacted):
    rows = _ledger_rows(data_root)
    carriers = [index for index, row in enumerate(rows) if "pre_compaction_seq" in row]
    assert len(carriers) >= 2
    duplicated = [dict(row) for row in rows]
    duplicated[carriers[1]]["pre_compaction_seq"] = rows[carriers[0]]["pre_compaction_seq"]
    with pytest.raises(UsageLedgerCorrupt):
        _validate_records(duplicated)
    # Claiming a folded epoch on a file that carries no baseline stamp is a
    # forged provenance; the same rows WITHOUT the claim are ordinary history.
    orphaned = [
        dict(row) for row in rows
        if row.get("kind") not in {"usage_baseline", "usage_baseline_group"}
    ]
    for index, row in enumerate(orphaned, start=1):
        row["seq"] = index
    with pytest.raises(UsageLedgerCorrupt):
        _validate_records(orphaned)
    _validate_records([
        {key: value for key, value in row.items() if key != "pre_compaction_seq"}
        for row in orphaned
    ])


def test_group_rows_require_a_leading_header(data_root_any_tier):
    group = {
        "kind": "usage_baseline_group", "attempt_id": "baseline-x-g0001",
        "state": "settled", "seq": 1, "baseline_id": "x",
        "folded_attempt_count": 1, "cost_usd": "1.0", "cost_final": True,
    }
    with pytest.raises(UsageLedgerCorrupt):
        _validate_records([group])


def test_compacted_ledger_revalidates_and_quarantine_semantics_hold(data_root, compacted):
    rows = _ledger_rows(data_root)
    _validate_records(rows)  # full-file validation accepts the baseline block
    # Torn tail on the compacted file still quarantines exactly as before.
    path = data_root / ua.LEDGER_REL
    with path.open("ab") as handle:
        handle.write(b'{"torn": ')
    projection = ua.usage_projection(data_root)
    assert projection["integrity_degraded"] is True
    assert (data_root / ua.QUARANTINE_REL).is_file()


def test_archive_segment_holds_exact_source_bytes(data_root):
    _seed_mixed_ledger(data_root)
    source = (data_root / ua.LEDGER_REL).read_bytes()
    receipt = _compact(data_root)
    segment = data_root / receipt["archive_rel"]
    payload = segment.read_bytes()
    assert payload == source
    assert hashlib.sha256(payload).hexdigest() == receipt["source_sha256"]
    header = _ledger_rows(data_root)[0]
    assert header["archive_rel"] == receipt["archive_rel"]
    assert header["source_sha256"] == receipt["source_sha256"]
    # Retained rows carry their pre-compaction seq for provenance.
    retained = [row for row in _ledger_rows(data_root)
                if row["kind"] not in {"usage_baseline", "usage_baseline_group"}]
    assert retained
    assert all(isinstance(row.get("pre_compaction_seq"), int) for row in retained)
