"""In-process validated-rows memo for display projections (perf sprint P1).

``usage_breakdown``/``usage_projection`` serve their results by running the
UNCHANGED aggregation code over an incrementally-resumed in-memory copy of the
validated ledger rows. These tests pin the three properties that make that
safe: (a) bit-exact equivalence with a fresh from-scratch replay after every
kind of append, (b) zero full-ledger reads on warm repeat calls, and (c) every
disclosed invalidation trigger (torn tail/quarantine, sequence discontinuity,
inode change, size shrink, cross-process append) forces a refold instead of
serving stale data. The write paths' own in-lock warm read cache
(razzant/ouroboros#129) and the append newline guard (#138) are pinned at the
end of this file; the monetary write SEMANTICS stay pinned by
tests/test_usage_accounting.py.
"""
from __future__ import annotations

import json
import os
import pathlib
import random

import pytest

from ouroboros import usage_accounting as ua


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    (root / "state").mkdir(parents=True)
    return root


def _request(data_root, **overrides):
    values = {
        "model": "openai/gpt-5.2",
        "provider": "openai",
        "reservation_usd": 0.05,
        "drive_root": data_root,
        "task_id": "child",
        "root_task_id": "root",
        "source": "test",
    }
    values.update(overrides)
    return ua.AttemptRequest(**values)


def _memo_key(root) -> str:
    return str(pathlib.Path(root).resolve(strict=False))


def _clear_memo(root) -> None:
    with ua._ROWS_MEMO_LOCK:
        ua._ROWS_MEMO.pop(_memo_key(root), None)


def _fresh_result(root, fn, *args, **kwargs):
    """Run ``fn`` as a fresh process would (no memo), then RESTORE the warm memo
    so the incremental chain under test keeps advancing instead of being
    replaced by the refold this comparison performs."""
    key = _memo_key(root)
    with ua._ROWS_MEMO_LOCK:
        saved = ua._ROWS_MEMO.pop(key, None)
    try:
        return fn(*args, **kwargs)
    finally:
        with ua._ROWS_MEMO_LOCK:
            if saved is not None:
                ua._ROWS_MEMO[key] = saved
            else:
                ua._ROWS_MEMO.pop(key, None)


def _assert_memo_matches_fresh(root):
    """Full nested-dict equivalence of every display read, warm vs from-scratch."""
    for fn, kwargs in (
        (ua.usage_projection, {}),
        (ua.usage_projection, {"root_task_id": "root-a"}),
        (ua.usage_projection, {"global_limit_usd": 50.0}),
        (ua.usage_breakdown, {}),
        (ua.usage_breakdown, {"root_task_id": "root-a"}),
        (ua.usage_breakdown, {"task_id": "task-a-0"}),
    ):
        warm = fn(root, **kwargs)
        fresh = _fresh_result(root, fn, root, **kwargs)
        assert warm == fresh, f"{fn.__name__}({kwargs}) diverged from a fresh replay"


def test_memoized_reads_equal_fresh_replay_after_every_append(data_root):
    """Property test: random real-API interleaving, checked after EVERY append."""
    rng = random.Random(20260808)
    reserved: list = []
    dispatched: list = []
    counter = {"n": 0}

    def new_reservation():
        counter["n"] += 1
        index = counter["n"]
        lane = rng.choice(("a", "b"))
        reservation = ua.reserve_attempt(_request(
            data_root,
            task_id=f"task-{lane}-{index % 3}",
            root_task_id=f"root-{lane}",
            category=rng.choice(("task", "review")),
            root_limit_usd=5.0 if lane == "a" else None,
            reservation_usd=rng.choice((0.01, 0.05, None)),
            prompt_tokens_estimate=rng.choice((0, 400)),
        ))
        reserved.append(reservation)

    def dispatch():
        reservation = reserved.pop(rng.randrange(len(reserved)))
        ua.mark_dispatched(reservation)
        dispatched.append(reservation)

    def release():
        ua.release_attempt(reserved.pop(rng.randrange(len(reserved))), "test_release")

    def settle():
        reservation = dispatched.pop(rng.randrange(len(dispatched)))
        variant = rng.randrange(3)
        if variant == 0:
            ua.settle_attempt(
                reservation,
                {"prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 5,
                 "prompt_cache_ttl": "1h"},
                cost_usd=0.02,
                cost_final=True,
            )
        elif variant == 1:
            ua.settle_attempt(reservation, {"prompt_tokens": 40}, cost_usd=0.01, cost_final=False)
        else:
            ua.settle_attempt(reservation, {}, cost_usd=None, cost_final=False)

    def unresolve():
        ua.mark_unresolved(dispatched.pop(rng.randrange(len(dispatched))), "provider unknown")

    def session():
        counter["n"] += 1
        ua.record_subscription_session(
            f"sess-{counter['n']}",
            drive_root=data_root,
            route=rng.choice(("codex", "claude")),
            task_id="task-a-0",
            root_task_id="root-a",
            reset_at=f"2026-08-08T0{rng.randrange(10)}:00:00Z",
            spend_usd=rng.choice((0.0, 0.03, None)),
            spend_estimated=rng.choice((False, True)),
        )

    def external():
        counter["n"] += 1
        ua.record_unmetered_external_dispatch(
            f"ext-{counter['n']}",
            drive_root=data_root,
            provider="external-skill",
            task_id="task-b-1",
            root_task_id="root-b",
            prompt_tokens=7,
        )

    for _ in range(40):
        actions = [new_reservation, session, external]
        if reserved:
            actions.extend((dispatch, release))
        if dispatched:
            actions.extend((settle, unresolve))
        rng.choice(actions)()
        _assert_memo_matches_fresh(data_root)

    # Terminalize an abandoned dispatch too (supervisor recovery path).
    if dispatched:
        ua.terminalize_abandoned_attempt(
            dispatched.pop(), reason="child died",
            usage={"prompt_tokens": 11, "completion_tokens": 2},
        )
        _assert_memo_matches_fresh(data_root)


def _install_full_read_counter(monkeypatch):
    calls = {"full": 0}
    real = ua._read_records_locked

    def counting(root, *args, **kwargs):
        calls["full"] += 1
        return real(root, *args, **kwargs)

    monkeypatch.setattr(ua, "_read_records_locked", counting)
    return calls


def test_warm_display_reads_do_zero_full_replays_cold_exactly_one(data_root, monkeypatch):
    reservation = ua.reserve_attempt(_request(data_root))
    ua.mark_dispatched(reservation)
    ua.settle_attempt(reservation, {"prompt_tokens": 5}, cost_usd=0.01, cost_final=True)

    calls = _install_full_read_counter(monkeypatch)
    _clear_memo(data_root)

    ua.usage_projection(data_root)
    assert calls["full"] == 1, "cold read must replay the ledger exactly once"
    for _ in range(5):
        ua.usage_projection(data_root)
        ua.usage_breakdown(data_root)
        ua.usage_breakdown(data_root, task_id="child")
        ua.usage_projection(data_root, root_task_id="root")
    assert calls["full"] == 1, "warm repeat display reads must do zero full replays"

    # A same-process append advances the memo incrementally on the next read,
    # and the write path itself reads through its warm in-lock cache (#129):
    # neither the write nor the display read after it adds a full replay.
    ua.release_attempt(ua.reserve_attempt(_request(data_root, task_id="next")))
    after_write = calls["full"]
    assert after_write == 1, "warm write-path reads must not full-replay the ledger"
    projection = ua.usage_projection(data_root)
    assert calls["full"] == after_write, "post-append display read must resume, not replay"
    assert projection["attempt_counts"]["released"] == 1


def test_torn_tail_forces_refold_and_quarantine_still_works(data_root, monkeypatch):
    reservation = ua.reserve_attempt(_request(data_root))
    ua.release_attempt(reservation)
    assert ua.usage_projection(data_root)["attempt_counts"] == {"released": 1}  # warm memo

    ledger = data_root / ua.LEDGER_REL
    with ledger.open("ab") as handle:
        handle.write(b'{"seq":')

    calls = _install_full_read_counter(monkeypatch)
    projection = ua.usage_projection(data_root)
    assert calls["full"] == 1, "a torn tail must route the read through the full refold"
    assert projection["attempt_counts"] == {"released": 1}
    assert projection["integrity_degraded"] is True
    assert (data_root / ua.QUARANTINE_REL).is_file()
    assert b'{"seq":' not in ledger.read_bytes(), "quarantine (owned by the full reader) still repairs"
    assert projection == _fresh_result(data_root, ua.usage_projection, data_root)


def test_sequence_discontinuity_in_tail_forces_refold(data_root):
    reservation = ua.reserve_attempt(_request(data_root))
    ua.release_attempt(reservation)
    warm = ua.usage_projection(data_root)
    assert warm["attempt_counts"] == {"released": 1}

    # A JSON-valid appended row whose seq does NOT continue the memoized count:
    # the incremental path must reject it (refold), and the full reader then
    # quarantines the structurally-invalid final row.
    ledger = data_root / ua.LEDGER_REL
    first_row = json.loads(ledger.read_text().splitlines()[0])
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**first_row, "seq": 1}) + "\n")

    projection = ua.usage_projection(data_root)
    assert projection["attempt_counts"] == {"released": 1}
    assert projection["integrity_degraded"] is True
    assert projection == _fresh_result(data_root, ua.usage_projection, data_root)


def test_inode_change_forces_refold_to_the_replacement_content(data_root):
    reservation = ua.reserve_attempt(_request(data_root))
    ua.release_attempt(reservation)
    assert ua.usage_projection(data_root)["attempt_counts"] == {"released": 1}  # warm

    # Replace the ledger wholesale (new inode) with a longer, DIFFERENT valid
    # history — a stale memo would keep reporting the old released attempt.
    ledger = data_root / ua.LEDGER_REL
    replacement = data_root / "state" / "replacement.jsonl"
    rows = [
        {"seq": 1, "ts": "2026-08-08T00:00:00Z", "kind": "attempt", "attempt_id": "n1",
         "state": "reserved", "model": "m", "provider": "openai",
         "reservation_upper_bound_usd": 0.5, "task_id": "t", "root_task_id": "r"},
        {"seq": 2, "ts": "2026-08-08T00:00:01Z", "kind": "attempt", "attempt_id": "n1",
         "state": "dispatched", "model": "m", "provider": "openai",
         "reservation_upper_bound_usd": 0.5, "task_id": "t", "root_task_id": "r"},
        {"seq": 3, "ts": "2026-08-08T00:00:02Z", "kind": "attempt", "attempt_id": "n1",
         "state": "settled", "model": "m", "provider": "openai", "cost_usd": 0.25,
         "cost_final": True, "task_id": "t", "root_task_id": "r"},
    ]
    replacement.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    os.replace(replacement, ledger)

    projection = ua.usage_projection(data_root)
    assert projection["attempt_counts"] == {"settled": 1}
    assert projection["settled_usd"] == 0.25
    assert projection == _fresh_result(data_root, ua.usage_projection, data_root)


def test_size_shrink_forces_refold_not_stale_rows(data_root):
    reservation = ua.reserve_attempt(_request(data_root))
    ua.mark_dispatched(reservation)
    ua.settle_attempt(reservation, {"prompt_tokens": 5}, cost_usd=0.02, cost_final=True)
    assert ua.usage_projection(data_root)["attempt_counts"] == {"settled": 1}  # warm

    ledger = data_root / ua.LEDGER_REL
    first_line = ledger.read_bytes().split(b"\n")[0] + b"\n"
    with ledger.open("r+b") as handle:
        handle.truncate(len(first_line))

    projection = ua.usage_projection(data_root)
    assert projection["attempt_counts"] == {"reserved": 1}, "shrink must refold, never serve stale rows"
    assert projection["settled_usd"] == 0.0
    assert projection == _fresh_result(data_root, ua.usage_projection, data_root)


def test_newline_less_crash_tail_never_lets_warm_reads_diverge(data_root):
    """Review-wave regression (GPT probe): a crash-torn final line that is still
    VALID JSON parses in the full read, but its end is not a row boundary. The
    resume fingerprint must refuse a non-row-aligned tail, keeping reads full
    replays until the tail is repaired. (Since the razzant/ouroboros#138 append
    guard, the next real append repairs the boundary with a leading newline
    instead of gluing; the fingerprint refusal still protects every read that
    happens before any append does.)
    """
    reservation = ua.reserve_attempt(_request(data_root))
    ua.release_attempt(reservation)
    ledger = data_root / ua.LEDGER_REL
    raw = ledger.read_bytes()
    assert raw.endswith(b"\n")
    ledger.write_bytes(raw[:-1])  # crash-torn: final row valid JSON, no newline

    _clear_memo(data_root)
    warm = ua.usage_projection(data_root)
    assert warm == _fresh_result(data_root, ua.usage_projection, data_root)

    # A real-API append lands after a repaired boundary (#138 guard): the torn
    # row survives as a valid row and warm reads must still match a fresh replay.
    ua.reserve_attempt(_request(data_root, task_id="glued"))

    warm = ua.usage_projection(data_root)
    fresh = _fresh_result(data_root, ua.usage_projection, data_root)
    assert warm == fresh, "warm read after an append onto the torn tail must match a fresh replay"
    breakdown = ua.usage_breakdown(data_root)
    assert breakdown == _fresh_result(data_root, ua.usage_breakdown, data_root)


def test_cross_process_append_is_seen_by_the_next_read(data_root):
    reservation = ua.reserve_attempt(_request(data_root))
    ua.release_attempt(reservation)
    assert ua.usage_projection(data_root)["attempt_counts"] == {"released": 1}  # warm

    # Simulate ANOTHER PROCESS: a direct substrate append that never touches
    # this process's memo. The next display read must see the row via the
    # stat + seq-continuity authority, not via any in-process invalidation.
    records = ua._read_records_locked(data_root)
    ua._append_rows_locked(data_root, records, [{
        "kind": "external_unmetered",
        "attempt_id": "external-foreign-process",
        "state": "settled",
        "model": "", "provider": "external",
        "cost_usd": None, "cost_final": False,
        "reservation_upper_bound_usd": None,
        "prompt_tokens": 3, "completion_tokens": 1,
        "task_id": "foreign", "root_task_id": "foreign",
        "parent_task_id": "", "category": "external", "source": "test",
    }])

    projection = ua.usage_projection(data_root)
    assert projection["attempt_counts"] == {"released": 1, "settled": 1}
    assert projection["unknown_unmetered"] == 1
    assert projection == _fresh_result(data_root, ua.usage_projection, data_root)
    breakdown = ua.usage_breakdown(data_root)
    assert breakdown["by_task"]["foreign"]["physical_calls"] == 1


def _clear_ledger_read_cache() -> None:
    from ouroboros import _usage_rows_memo as memo

    with memo._LEDGER_READ_CACHE_LOCK:
        memo._LEDGER_READ_CACHE.clear()


def test_torn_newline_less_tail_costs_at_most_itself_on_the_next_append(data_root):
    """razzant/ouroboros#138: a crashed writer can leave a JSON-complete row
    with no trailing newline — the one shape the full read ACCEPTS as a row.
    Before the append-boundary guard, the next append glued its first row onto
    that tail and the following read quarantined BOTH rows, destroying a
    previously validated row and orphaning the caller's live reservation. The
    guard prepends the missing newline, so the torn row and every following
    append survive with no quarantine at all.
    """
    r1 = ua.reserve_attempt(_request(data_root))
    ua.release_attempt(r1)
    ledger = data_root / ua.LEDGER_REL
    raw = ledger.read_bytes()
    assert raw.endswith(b"\n")
    ledger.write_bytes(raw[:-1])  # crash-torn: final row valid JSON, no newline

    r2 = ua.reserve_attempt(_request(data_root))
    ua.mark_dispatched(r2)
    ua.settle_attempt(r2, usage={"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.01})

    assert not (data_root / ua.QUARANTINE_REL).exists()
    projection = _fresh_result(data_root, ua.usage_projection, data_root)
    assert projection["attempt_counts"] == {"released": 1, "settled": 1}
    assert projection["integrity_degraded"] is False


def test_ledger_read_cache_matches_the_full_read_cold_and_warm(data_root):
    """razzant/ouroboros#129: the in-lock write-path read is served from a warm
    incremental cache. It must return exactly what the full validated read
    returns — cold, and after appends grow the file."""
    _clear_ledger_read_cache()
    r1 = ua.reserve_attempt(_request(data_root))
    ua.release_attempt(r1)
    with ua._locked(data_root):
        cold = ua._read_records_locked_cached(data_root)
        raw = ua._read_records_locked(data_root)
    assert cold == raw

    # Grow the ledger; the next cached read must pick up the delta incrementally.
    r2 = ua.reserve_attempt(_request(data_root))
    ua.mark_dispatched(r2)
    ua.settle_attempt(r2, usage={"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.01})
    with ua._locked(data_root):
        warm = ua._read_records_locked_cached(data_root)
        raw2 = ua._read_records_locked(data_root)
    assert warm == raw2
    assert len(warm) > len(cold)

    # A projection computed with the cache warm equals one from a cold process.
    warm_projection = ua.usage_projection(data_root)
    _clear_ledger_read_cache()
    assert ua.usage_projection(data_root) == warm_projection


def test_ledger_read_cache_falls_back_when_the_file_is_rewritten(data_root):
    """An atomic replace of the ledger (new inode — restore, future compaction)
    must force a full re-read. The rewrite REMOVES rows, so serving the stale
    cached state would be a visible wrong answer, not a lucky equality."""
    _clear_ledger_read_cache()
    r1 = ua.reserve_attempt(_request(data_root))
    ua.release_attempt(r1)
    r2 = ua.reserve_attempt(_request(data_root))
    ua.release_attempt(r2)
    with ua._locked(data_root):
        seeded = ua._read_records_locked_cached(data_root)
    assert len(seeded) == 4

    ledger = data_root / ua.LEDGER_REL
    lines = ledger.read_bytes().splitlines(keepends=True)
    tmp = ledger.with_suffix(".rewrite")
    tmp.write_bytes(b"".join(lines[:2]))  # keep only r1's reserve+release
    tmp.replace(ledger)

    with ua._locked(data_root):
        after = ua._read_records_locked_cached(data_root)
        raw = ua._read_records_locked(data_root)
    assert after == raw
    assert len(after) == 2  # the cached four-row state was not served


# CyberGym r8 (2026-09-04): every append re-validated the WHOLE ledger under the
# monetary lock (0.6 s at 43K rows), saturating the lock at ~3 transitions/s and
# starving callers past the 45 s timeout — tasks died as task_exception:
# UsageLockUnavailable. Appends validate only their tail against the prefix's
# per-attempt last state, preferring the cached validated read's state map.


def test_append_validates_only_the_appended_tail(data_root, monkeypatch):
    for _ in range(3):
        reservation = ua.reserve_attempt(_request(data_root))
        ua.mark_dispatched(reservation)
        ua.settle_attempt(reservation, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=0.001, cost_final=True)
    from ouroboros import usage_ledger

    seen: list[tuple[int, int]] = []
    real = usage_ledger._validate_records

    def spy(records, *, start_seq=1, states=None):
        seen.append((len(records), start_seq))
        return real(records, start_seq=start_seq, states=states)

    monkeypatch.setattr(usage_ledger, "_validate_records", spy)
    with ua._locked(data_root):
        records = ua._read_records_locked_cached(data_root)
        assert len(records) == 9
        ua._append_rows_locked(data_root, records, [{
            "kind": "external_unmetered", "attempt_id": "ext-1", "state": "settled",
            "model": "", "provider": "external", "cost_usd": None, "cost_final": False,
            "reservation_upper_bound_usd": None, "prompt_tokens": 1, "completion_tokens": 1,
            "task_id": "t", "root_task_id": "t", "parent_task_id": "", "category": "external",
            "source": "test",
        }])
    # The in-lock read validates only its resumed tail, and the append validates
    # only the one new row at seq 10 — the 9-row prefix is never replayed.
    assert seen[-1] == (1, 10)
    assert all(count < 9 for count, _ in seen)


def test_append_still_rejects_an_invalid_transition_against_the_prefix(data_root):
    reservation = ua.reserve_attempt(_request(data_root))
    ua.mark_dispatched(reservation)
    ua.settle_attempt(reservation, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=0.001, cost_final=True)
    with ua._locked(data_root):
        records = ua._read_records_locked_cached(data_root)
        final = dict(records[-1])
        final.pop("seq")
        with pytest.raises(ua.UsageLedgerCorrupt, match="changed after terminal state"):
            ua._append_rows_locked(data_root, records, [final])
        fresh = {**records[0], "attempt_id": "brand-new", "state": "dispatched"}
        fresh.pop("seq")
        with pytest.raises(ua.UsageLedgerCorrupt, match="did not begin reserved"):
            ua._append_rows_locked(data_root, records, [fresh])
    # Nothing was written by the rejected appends.
    assert len(ua._read_records_locked(data_root)) == 3


def test_append_state_map_matches_between_cached_and_derived_paths(data_root):
    from ouroboros import _usage_rows_memo as memo

    reservations = [ua.reserve_attempt(_request(data_root)) for _ in range(4)]
    ua.mark_dispatched(reservations[0])
    ua.release_attempt(reservations[1])
    with ua._locked(data_root):
        records = ua._read_records_locked_cached(data_root)
        cached = memo._cached_attempt_states(data_root, len(records))
    assert cached is not None
    derived = {str(row["attempt_id"]): str(row["state"]) for row in records}
    assert cached == derived
    # Extent mismatch => not usable; the derived map is the fallback.
    assert memo._cached_attempt_states(data_root, len(records) - 1) is None
    _clear_ledger_read_cache()
    assert memo._cached_attempt_states(data_root, len(records)) is None
    # And the derived path appends just as well after the cache was dropped.
    ua.mark_dispatched(reservations[2])
    assert ua.usage_projection(data_root)["attempt_counts"] == {"dispatched": 2, "released": 1, "reserved": 1}


def test_last_row_for_is_the_final_row_of_the_attempt(data_root):
    reservation = ua.reserve_attempt(_request(data_root))
    ua.mark_dispatched(reservation)
    other = ua.reserve_attempt(_request(data_root))
    records = ua._read_records_locked(data_root)
    assert ua._last_row_for(records, reservation.attempt_id) is ua._final_rows(records)[reservation.attempt_id]
    assert ua._last_row_for(records, reservation.attempt_id)["state"] == "dispatched"
    assert ua._last_row_for(records, other.attempt_id)["state"] == "reserved"
    assert ua._last_row_for(records, "nobody") is None
