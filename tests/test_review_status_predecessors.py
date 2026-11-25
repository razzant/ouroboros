"""``review_status`` over a recorded Continue's predecessor attempts: order, an unreadable read, and what stays listed."""
from __future__ import annotations

import json

import pytest

from ouroboros.owner_continue import binding_sha
from ouroboros.review_state import CommitAttemptRecord, _STATE_RELPATH, make_repo_key, update_state
from ouroboros.store import StoreBusy
from ouroboros.task_results import write_task_result
from ouroboros.task_store import patch_document
from ouroboros.tools.preflight_review import _handle_review_status
from tests._git_review_pipeline_shared import _make_ctx

PRED, SUCC = "synthetic-pred", "synthetic-succ"


def _row(key, task, attempt, ts, message, status="failed"):
    return CommitAttemptRecord(ts=ts, started_ts=ts, finished_ts=ts, commit_message=message, status=status, phase="review_only",
                               repo_key=key, tool_name="commit_reviewed", task_id=task, root_task_id=task, attempt=attempt)


@pytest.fixture
def chain(tmp_path):
    """A predecessor on its own drive, a successor on its fork, and the recorded Continue between them."""
    ctx = _make_ctx(tmp_path)
    canonical = tmp_path / "canonical"
    pred_root, fork = (canonical / "state" / "headless_tasks" / task / "data" for task in (PRED, SUCC))
    for root in (canonical, pred_root, fork):
        root.mkdir(parents=True, exist_ok=True)
    write_task_result(canonical, PRED, "completed", chat_id=7)
    write_task_result(canonical, SUCC, "running", chat_id=7)
    binding = {"successor_task_id": SUCC}
    patch_document(canonical, SUCC, lambda row: {**row, "metadata": {**(row.get("metadata") or {}),
                                                                      "continuation": {"predecessor_task_id": PRED}}})
    patch_document(canonical, PRED, lambda row: {**row, "continued_by": {
        "successor_task_id": SUCC, "binding": binding, "binding_sha256": binding_sha(binding), "state": "admitted"}})
    ctx.task_id, ctx.budget_drive_root, ctx.drive_root = SUCC, canonical, fork
    key = make_repo_key(ctx.repo_dir)

    def put(root, *rows):
        update_state(root, lambda state: [state.record_attempt(row) for row in rows])

    return ctx, key, pred_root, fork, put


def _status(ctx, **kwargs):
    return json.loads(_handle_review_status(ctx, **kwargs))


def _tasks(status):
    return [(row["task_id"], row["attempt"]) for row in reversed(status["attempts"])]  # oldest first


def test_a_predecessor_row_takes_its_place_by_time_not_in_front_of_the_local_ones(chain):
    ctx, key, pred_root, fork, put = chain
    put(pred_root, _row(key, PRED, 1, "2026-11-20T10:00:00+00:00", "pred"))
    put(fork, _row(key, "unrelated", 1, "2026-11-19T10:00:00+00:00", "older local"),
        _row(key, SUCC, 1, "2026-11-21T10:00:00+00:00", "own"))
    status = _status(ctx)
    assert _tasks(status) == [("unrelated", 1), (PRED, 1), (SUCC, 1)]
    assert status["selected_commit_attempt"]["commit_message"] == "own" and not status.get("predecessor_attempts_gap")
    put(fork, _row(key, "unrelated", 2, "2026-11-19T11:00:00+00:00", "local, still older"))
    assert _status(ctx)["selected_commit_attempt"]["commit_message"] == "own"
    assert _status(ctx, task_id=PRED)["selected_commit_attempt"]["commit_message"] == "pred"


def _stamps(status):
    return sorted(row["ts"] for row in status["attempts"])


def test_a_verified_predecessors_copy_on_the_fork_is_no_source(chain):
    ctx, key, pred_root, fork, put = chain
    put(pred_root, _row(key, PRED, 1, "2026-11-20T10:00:00+00:00", "on its own drive"))
    put(fork, _row(key, PRED, 1, "2026-11-20T12:00:00+00:00", "copy on the fork"), _row(key, PRED, 2, "2026-11-20T13:00:00+00:00", "fork only"))
    assert _stamps(_status(ctx, task_id=PRED)) == ["2026-11-20T10:00:00+00:00"]


def test_a_busy_store_names_the_gap_and_leaves_the_local_status(chain, monkeypatch):
    ctx, key, pred_root, fork, put = chain
    put(pred_root, _row(key, PRED, 1, "2026-11-20T10:00:00+00:00", "pred", status="reviewed"))
    put(fork, _row(key, SUCC, 1, "2026-11-21T10:00:00+00:00", "own"),
        _row(key, PRED, 1, "2026-11-20T12:00:00+00:00", "fork copy", status="reviewed"))
    assert "review_reference" in _status(ctx, task_id=PRED), "the authorized predecessor row offers its reference"
    import ouroboros.task_results as task_results

    def busy(*args, **kwargs):
        raise StoreBusy("current store busy", reason="contention")

    monkeypatch.setattr(task_results, "load_task_result", busy)
    for filters in ({}, {"tool_name": "commit_reviewed"}):
        status = _status(ctx, **filters)
        assert "StoreBusy" in status["predecessor_attempts_gap"] and "not all listed" in status["status_summary"], status
        assert status["selected_commit_attempt"]["commit_message"] == "own"
    own_only = _status(ctx, task_id=SUCC)
    assert not own_only.get("predecessor_attempts_gap") and own_only["selected_commit_attempt"]["commit_message"] == "own"
    # Nothing was read: the fork's copy stays listed as a historical local row, never as the predecessor's evidence.
    assert _stamps(_status(ctx)) == ["2026-11-20T12:00:00+00:00", "2026-11-21T10:00:00+00:00"]
    copy = _status(ctx, task_id=PRED)
    assert copy["selected_commit_attempt"]["commit_message"] == "fork copy" and "review_reference" not in copy


def test_current_task_filter_skips_predecessor_reads_entirely(chain, monkeypatch):
    import ouroboros.owner_continue as owner_continue
    import ouroboros.source_retention as source_retention
    import ouroboros.task_results as task_results

    ctx, key, pred_root, fork, put = chain
    put(pred_root, _row(key, PRED, 1, "2026-11-20T10:00:00+00:00", "pred"))
    put(fork, _row(key, SUCC, 1, "2026-11-21T10:00:00+00:00", "own"),
        _row(key, PRED, 1, "2026-11-20T12:00:00+00:00", "fork copy"))
    (pred_root / _STATE_RELPATH).write_text("{not json", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError(f"predecessor lookup should not run for task_id={SUCC}")

    monkeypatch.setattr(task_results, "load_task_result", forbidden)
    monkeypatch.setattr(owner_continue, "continuation_chain", forbidden)
    monkeypatch.setattr(source_retention, "retained_task_roots", forbidden)
    status = _status(ctx, task_id=SUCC)
    assert _stamps(status) == ["2026-11-21T10:00:00+00:00"] and not status.get("predecessor_attempts_gap")
    assert status["selected_commit_attempt"]["commit_message"] == "own"


def test_unreadable_lower_priority_fallback_keeps_proven_row_and_names_gap(chain):
    ctx, key, pred_root, fork, put = chain
    canonical = ctx.budget_drive_root
    put(pred_root, _row(key, PRED, 1, "2026-11-20T10:00:00+00:00", "pred", status="reviewed"))
    put(fork, _row(key, SUCC, 1, "2026-11-21T10:00:00+00:00", "own"),
        _row(key, PRED, 1, "2026-11-20T12:00:00+00:00", "fork copy", status="reviewed"))
    (canonical / _STATE_RELPATH).parent.mkdir(parents=True, exist_ok=True)
    (canonical / _STATE_RELPATH).write_text("{not json", encoding="utf-8")

    status = _status(ctx)
    assert _stamps(status) == ["2026-11-20T10:00:00+00:00", "2026-11-21T10:00:00+00:00"], status
    assert str(canonical) in status["predecessor_attempts_gap"] and "could not be read" in status["predecessor_attempts_gap"]
    pred_only = _status(ctx, task_id=PRED)
    assert pred_only["selected_commit_attempt"]["commit_message"] == "pred"
    assert "review_reference" in pred_only and str(canonical) in pred_only["predecessor_attempts_gap"]

    # Reverse priority: an unreadable higher-priority drive stops before reading a staler canonical fallback.
    (canonical / _STATE_RELPATH).unlink()
    put(canonical, _row(key, PRED, 1, "2026-11-20T09:00:00+00:00", "stale canonical", status="reviewed"))
    (pred_root / _STATE_RELPATH).write_text("{not json", encoding="utf-8")
    blocked = _status(ctx)
    assert _stamps(blocked) == ["2026-11-21T10:00:00+00:00"] and str(pred_root) in blocked["predecessor_attempts_gap"]


def test_gap_deduplicates_shared_drive_and_link_errors_and_task_filter_scopes_drives(tmp_path, monkeypatch):
    import ouroboros.task_results as task_results

    first, middle, last = "zz-first", "aa-middle", "mm-last"
    ctx = _make_ctx(tmp_path)
    canonical = tmp_path / "canonical"
    roots = {task: canonical / "state" / "headless_tasks" / task / "data" for task in (first, middle, last)}
    for root in (canonical, *roots.values()):
        root.mkdir(parents=True, exist_ok=True)
    for task, state in ((first, "completed"), (middle, "completed"), (last, "running")):
        write_task_result(canonical, task, state, chat_id=7)
    for task, successor in ((first, middle), (middle, last)):
        binding = {"successor_task_id": successor}
        patch_document(canonical, task, lambda row, b=binding, s=successor: {**row, "continued_by": {
            "successor_task_id": s, "binding": b, "binding_sha256": binding_sha(b), "state": "admitted"}})
        patch_document(canonical, successor, lambda row, p=task: {**row, "metadata": {**(row.get("metadata") or {}),
                                                                                         "continuation": {"predecessor_task_id": p}}})
    ctx.task_id, ctx.budget_drive_root, ctx.drive_root = last, canonical, roots[last]
    key = make_repo_key(ctx.repo_dir)
    update_state(roots[first], lambda state: state.record_attempt(_row(key, first, 1, "2026-11-20T10:00:00+00:00", "first", status="reviewed")))
    update_state(roots[middle], lambda state: state.record_attempt(_row(key, middle, 1, "2026-11-20T11:00:00+00:00", "middle", status="reviewed")))
    update_state(roots[last], lambda state: [state.record_attempt(_row(key, middle, 1, "2026-11-20T08:00:00+00:00", "stale middle")),
                                             state.record_attempt(_row(key, last, 1, "2026-11-20T12:00:00+00:00", "own"))])
    (canonical / _STATE_RELPATH).parent.mkdir(parents=True, exist_ok=True)
    (canonical / _STATE_RELPATH).write_text("{not json", encoding="utf-8")

    shared_gap = _status(ctx)
    assert _stamps(shared_gap) == ["2026-11-20T10:00:00+00:00", "2026-11-20T11:00:00+00:00", "2026-11-20T12:00:00+00:00"]
    assert shared_gap["predecessor_attempts_gap"].count(str(canonical)) == 1, shared_gap["predecessor_attempts_gap"]

    # Filtering to first opens only first's drives; an unreadable middle drive does not pollute first's status.
    (canonical / _STATE_RELPATH).unlink()
    (roots[middle] / _STATE_RELPATH).write_text("{not json", encoding="utf-8")
    first_only = _status(ctx, task_id=first)
    assert _stamps(first_only) == ["2026-11-20T10:00:00+00:00"] and not first_only.get("predecessor_attempts_gap")

    # An unreadable Continue link names the unreadable task once (not again as unverified) and drops its fork copy.
    real_load = task_results.load_task_result

    def middle_busy(root, task_id, *args, **kwargs):
        if task_id == middle:
            raise StoreBusy("middle store busy", reason="contention")
        return real_load(root, task_id, *args, **kwargs)

    monkeypatch.setattr(task_results, "load_task_result", middle_busy)
    link_gap = _status(ctx)
    assert _stamps(link_gap) == ["2026-11-20T12:00:00+00:00"]
    assert "StoreBusy" in link_gap["predecessor_attempts_gap"] and link_gap["predecessor_attempts_gap"].count(middle) == 1, link_gap


def test_an_unreadable_predecessor_drive_is_named_and_its_fork_copy_is_omitted(chain):
    ctx, key, pred_root, fork, put = chain
    put(pred_root, _row(key, PRED, 1, "2026-11-20T10:00:00+00:00", "pred"))
    put(fork, _row(key, SUCC, 1, "2026-11-21T10:00:00+00:00", "own"), _row(key, PRED, 1, "2026-11-20T12:00:00+00:00", "fork copy"))
    (pred_root / _STATE_RELPATH).write_text("{not json", encoding="utf-8")
    status = _status(ctx)
    assert "could not be read" in status["predecessor_attempts_gap"]
    assert _stamps(status) == ["2026-11-21T10:00:00+00:00"], "a known predecessor's stale copy is not offered as its history"


def test_an_unverified_link_is_a_gap_and_opens_no_drive(chain):
    ctx, key, pred_root, fork, put = chain
    put(pred_root, _row(key, PRED, 1, "2026-11-20T10:00:00+00:00", "pred"))
    put(fork, _row(key, SUCC, 1, "2026-11-21T10:00:00+00:00", "own"))
    put(fork, _row(key, PRED, 1, "2026-11-20T12:00:00+00:00", "fork copy"))
    patch_document(ctx.budget_drive_root, PRED, lambda row: {**row, "continued_by": {**row["continued_by"], "binding_sha256": "tampered"}})
    status = _status(ctx)
    assert _stamps(status) == ["2026-11-21T10:00:00+00:00"] and PRED in status["predecessor_attempts_gap"]


def test_each_predecessor_is_read_from_its_own_drives_never_from_a_later_ones_fork(tmp_path):
    """Two valid Continues FIRST -> MIDDLE -> LAST. MIDDLE sorts before FIRST, and its fork carries a stale copy of
    FIRST's row: the flattened read took the first drive that held the identity, so the stale copy won."""
    first, middle, last = "zz-first", "aa-middle", "mm-last"
    ctx = _make_ctx(tmp_path)
    canonical = tmp_path / "canonical"
    roots = {task: canonical / "state" / "headless_tasks" / task / "data" for task in (first, middle, last)}
    for root in (canonical, *roots.values()):
        root.mkdir(parents=True, exist_ok=True)
    for task, state in ((first, "completed"), (middle, "completed"), (last, "running")):
        write_task_result(canonical, task, state, chat_id=7)
    for task, successor, prior in ((first, middle, None), (middle, last, first)):
        binding = {"successor_task_id": successor}
        patch_document(canonical, task, lambda row, b=binding, s=successor: {**row, "continued_by": {
            "successor_task_id": s, "binding": b, "binding_sha256": binding_sha(b), "state": "admitted"}})
        patch_document(canonical, successor, lambda row, p=task: {**row, "metadata": {**(row.get("metadata") or {}),
                                                                                         "continuation": {"predecessor_task_id": p}}})
    ctx.task_id, ctx.budget_drive_root, ctx.drive_root = last, canonical, roots[last]
    key = make_repo_key(ctx.repo_dir)
    update_state(roots[first], lambda state: state.record_attempt(_row(key, first, 1, "2026-11-20T10:00:00+00:00", "true first")))
    update_state(roots[middle], lambda state: [state.record_attempt(_row(key, first, 1, "2026-11-20T09:00:00+00:00", "stale first")),
                                               state.record_attempt(_row(key, middle, 1, "2026-11-20T11:00:00+00:00", "middle"))])
    update_state(roots[last], lambda state: state.record_attempt(_row(key, first, 1, "2026-11-20T08:00:00+00:00", "inherited first")))
    status = _status(ctx)
    assert _stamps(status) == ["2026-11-20T10:00:00+00:00", "2026-11-20T11:00:00+00:00"] and not status.get("predecessor_attempts_gap"), status
    assert _stamps(_status(ctx, task_id=first)) == ["2026-11-20T10:00:00+00:00"]


def _drive_probes(monkeypatch):
    """Record, without changing them, every listing of a task's drives and every path put to the placement check."""
    import ouroboros.source_retention as source_retention
    from ouroboros.tools import review_collection

    probes = {"listed": [], "asked": []}

    def watch(module, name, bucket, subject):
        real = getattr(module, name)

        def watched(*args, **kwargs):
            probes[bucket].append(subject(*args))
            return real(*args, **kwargs)

        monkeypatch.setattr(module, name, watched)

    watch(review_collection, "commit_attempt_roots", "listed", lambda _ctx, task: task)
    watch(source_retention, "retained_task_roots", "listed", lambda _canonical, task: task)
    watch(source_retention, "retained_child_root", "asked",
          lambda _canonical, child, _task="": child.resolve(strict=False))
    return probes


@pytest.mark.parametrize("layout", ["canonical", "predecessor_drive", "fork"])
def test_another_checkouts_filter_keeps_a_shared_drives_rows_and_opens_no_other_drive(
        chain, tmp_path, monkeypatch, layout):
    """#2901: another checkout's filter lists no drive of a predecessor and opens none, so nothing replaces a row there.
    Only the drive this call reads is asked whether it is the predecessor's own source (the shared canonical drive, or
    its retained drive by the placement check): a row there stays listed though the Continue is now known, as history
    that offers no continuation reference, while the fork's copy of the predecessor's row stays omitted."""
    ctx, key, pred_root, fork, put = chain
    other = tmp_path / "other-checkout"
    other.mkdir()
    other_key = make_repo_key(other)
    here = ctx.drive_root = {"canonical": ctx.budget_drive_root, "predecessor_drive": pred_root, "fork": fork}[layout]
    put(here, _row(other_key, PRED, 1, "2026-11-20T10:00:00+00:00", "previous task's other checkout", status="reviewed"),
        _row(other_key, SUCC, 1, "2026-11-21T10:00:00+00:00", "own", status="reviewed"))
    unreadable = {pred_root, fork} - {here}
    for root in unreadable:  # a drive opened would be named in the gap
        (root / _STATE_RELPATH).parent.mkdir(parents=True, exist_ok=True)
        (root / _STATE_RELPATH).write_text("{not json", encoding="utf-8")
    if pred_root in unreadable:  # the same call for this checkout does read the predecessor's drive
        assert str(pred_root) in _status(ctx)["predecessor_attempts_gap"]
    probes = _drive_probes(monkeypatch)
    _status(ctx)  # and lists its drives: the listing the probes must also see were another checkout's filter to make it
    assert set(probes["listed"]) == {PRED}, probes
    probes["listed"].clear()
    probes["asked"].clear()
    expected = [(SUCC, 1)] if layout == "fork" else [(PRED, 1), (SUCC, 1)]
    for filters in ({"repo_key": other_key}, {"repo_key": other_key, "tool_name": "commit_reviewed"}):
        status = _status(ctx, **filters)
        assert _tasks(status) == expected and not status.get("predecessor_attempts_gap"), (layout, status)
    assert "review_reference" in _status(ctx, repo_key=other_key), "this task's own row still offers its reference"
    pred_only = _status(ctx, repo_key=other_key, task_id=PRED)
    assert _tasks(pred_only) == ([] if layout == "fork" else [(PRED, 1)]), (layout, pred_only)
    assert "review_reference" not in pred_only, "a row no read proved is history, never a continuation reference"
    assert not probes["listed"], f"another checkout's filter listed the drives of {probes['listed']}"
    assert set(probes["asked"]) <= {here.resolve(strict=False)}, "only this call's own drive meets the placement check"


@pytest.mark.parametrize("tool", ["review_change", "review_result", "review_skill"])
def test_a_tool_filter_no_commit_attempt_answers_walks_no_chain_and_no_drive(chain, monkeypatch, tool):
    """A tool filter no commit attempt answers reads neither the Continue chain nor a drive; the commit filter beside it
    still names the unreadable predecessor drive."""
    import dataclasses

    import ouroboros.owner_continue as owner_continue
    import ouroboros.source_retention as source_retention
    import ouroboros.task_results as task_results

    ctx, key, pred_root, fork, put = chain
    put(pred_root, _row(key, PRED, 1, "2026-11-20T10:00:00+00:00", "pred"))
    put(fork, _row(key, PRED, 1, "2026-11-20T12:00:00+00:00", "fork copy"),
        dataclasses.replace(_row(key, SUCC, 1, "2026-11-21T10:00:00+00:00", "own review"), tool_name=tool))
    (pred_root / _STATE_RELPATH).write_text("{not json", encoding="utf-8")
    assert "could not be read" in _status(ctx, tool_name="commit_reviewed")["predecessor_attempts_gap"]

    def forbidden(*args, **kwargs):
        raise AssertionError(f"no predecessor is walked for tool_name={tool}")

    monkeypatch.setattr(task_results, "load_task_result", forbidden)
    monkeypatch.setattr(owner_continue, "continuation_chain", forbidden)
    monkeypatch.setattr(source_retention, "retained_task_roots", forbidden)
    status = _status(ctx, tool_name=tool)
    assert _stamps(status) == ["2026-11-21T10:00:00+00:00"] and not status.get("predecessor_attempts_gap")
    assert status["selected_commit_attempt"]["commit_message"] == "own review"


def test_unreadable_retry_placement_blocks_stale_canonical_and_preserves_proven_rows(chain, tmp_path, monkeypatch):
    """Trigger 2 (#2901): an unreadable retry-occupant placement never silently falls through to a stale canonical row;
    any higher-priority proven rows stay visible in status, the gap names the unreadable placement, confirmed nonmatch
    still falls through, and canonical task reads stay bounded."""
    import shutil
    from ouroboros import source_retention, task_store

    ctx, key, pred_root, fork, put = chain
    canonical = ctx.budget_drive_root
    origin_drive = canonical / "task_drives" / "retry-origin"
    origin_drive.mkdir(parents=True, exist_ok=True)
    patch_document(canonical, PRED, lambda row: {
        **row, "child_drive_root": str(origin_drive), "metadata": {**(row.get("metadata") or {}), "drive_root": str(origin_drive)},
    })
    put(pred_root, _row(key, PRED, 1, "2026-11-20T10:00:00+00:00", "proven own row", status="reviewed"))
    put(canonical, _row(key, PRED, 2, "2026-11-20T09:00:00+00:00", "stale canonical row", status="reviewed"))
    put(fork, _row(key, SUCC, 1, "2026-11-21T10:00:00+00:00", "own", status="reviewed"))

    real_read = task_store.read_document
    canonical_reads = []

    def counting_busy_origin(root, task_id, *args, **kwargs):
        resolved = root.resolve(strict=False)
        if task_id == PRED and resolved == canonical.resolve(strict=False):
            canonical_reads.append(task_id)
        if task_id == PRED and resolved == origin_drive.resolve(strict=False):
            raise StoreBusy("retry drive busy", reason="contention")
        return real_read(root, task_id, *args, **kwargs)

    # 1. Higher-priority own drive IS readable -> proven row stays visible, stale canonical is NOT selected, gap named.
    with monkeypatch.context() as mp:
        mp.setattr(task_store, "read_document", counting_busy_origin)
        status = _status(ctx)
    assert _stamps(status) == ["2026-11-20T10:00:00+00:00", "2026-11-21T10:00:00+00:00"], status
    assert _tasks(status) == [(PRED, 1), (SUCC, 1)], status
    assert "StoreBusy" in status["predecessor_attempts_gap"] and str(origin_drive.resolve()) in status["predecessor_attempts_gap"]

    # 2. Bounded cost: retained_task_roots reads canonical task doc at most once even with duplicate candidate fields.
    canonical_reads.clear()
    with monkeypatch.context() as mp:
        mp.setattr(task_store, "read_document", lambda root, tid, *a, **k: (
            canonical_reads.append((root.resolve(strict=False), tid)), real_read(root, tid, *a, **k))[1])
        source_retention.retained_task_roots(canonical, PRED, strict=True)
    assert sum(1 for r, t in canonical_reads if r == canonical.resolve(strict=False) and t == PRED) == 1

    # 3. No own drive + unreadable retry drive -> stale canonical is NOT selected and gap names the retry drive.
    shutil.rmtree(pred_root)
    with monkeypatch.context() as mp:
        mp.setattr(task_store, "read_document", counting_busy_origin)
        blocked = _status(ctx)
    assert _stamps(blocked) == ["2026-11-21T10:00:00+00:00"], blocked
    assert "StoreBusy" in blocked["predecessor_attempts_gap"] and str(origin_drive.resolve()) in blocked["predecessor_attempts_gap"]

    # 4. Confirmed nonmatch (origin store readable without PRED's row) falls through to canonical with no gap.
    fell_through = _status(ctx)
    assert _stamps(fell_through) == ["2026-11-20T09:00:00+00:00", "2026-11-21T10:00:00+00:00"], fell_through
    assert not fell_through.get("predecessor_attempts_gap")

    # 5. Another checkout's filter on an unreadable retry drive records a gap instead of silently failing.
    other = tmp_path / "other-repo"
    other.mkdir()
    other_key = make_repo_key(other)
    put(origin_drive, _row(other_key, PRED, 1, "2026-11-20T10:00:00+00:00", "pred other", status="reviewed"),
        _row(other_key, SUCC, 1, "2026-11-21T10:00:00+00:00", "succ other", status="reviewed"))
    ctx.drive_root = origin_drive
    with monkeypatch.context() as mp:
        mp.setattr(task_store, "read_document", counting_busy_origin)
        other_status = _status(ctx, repo_key=other_key)
    assert "StoreBusy" in other_status["predecessor_attempts_gap"] and str(origin_drive.resolve()) in other_status["predecessor_attempts_gap"]
