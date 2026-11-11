"""The rolling timeline the body's window view records, and the tail a wake's read shows.

Split out of ``tests/test_delegated_subagent_transport.py`` by theme. An engine without the run's
journal keeps the window view (``delegate_feed.Feed.window``): ``WindowObservations.record`` must
lose no row the daemon added, and say what a bounded batch does not show. A wake's one read of a
live run (``delegate_progress.present_payload``) carries the run's bounded timeline tail, which
must fit the result whole.
"""

from __future__ import annotations

import json

from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _owned_gateway_uses_each_test_transport,
)


def test_a_verbose_timeline_tail_fits_a_wakes_payload_whole():
    """`timeline_tail` carries harness-authored text — three fields, each bounded at 300 chars, twelve rows — beside
    a known question set; the whole payload still fits the result, so the generic truncator never cuts the JSON
    mid-structure and the model gets something it can parse."""
    from ouroboros.delegate_progress import present_payload
    from ouroboros.loop_tool_execution import _truncate_tool_result
    from ouroboros.tool_capabilities import tool_result_limit

    limit = tool_result_limit("delegate_wait")
    verbose = [{"title": "T" * 400, "type": "Y" * 400, "severity": "S" * 400} for _ in range(12)]
    pending = [{"interaction_id": f"int-{i}", "questions": [{"question_id": "q", "question": "Q" * 2_000}]}
               for i in range(8)]
    payload = present_payload(
        run_id="run-1", state="running", last_seq=601, moved=True, elapsed_seconds=1800, max_seconds=1800,
        waiting_on_user=True, detail={"timeline": verbose}, budget=limit, pending_interactions=pending)

    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    assert len(raw) <= limit, len(raw)
    assert _truncate_tool_result(raw, "delegate_wait", {}) == raw, \
        "the generic truncator had to cut a payload the sizing claimed would fit"
    assert json.loads(raw) == payload, "the model received unparseable JSON"
    assert len(payload["timeline_tail"]) == 12, "the tail keeps its full bounded self; the questions yield room"
    assert len(payload.get("pending_interactions") or []) + payload["interactions_omitted"] == 8


def test_a_batch_bigger_than_the_display_tail_says_how_much_it_is_not_showing(tmp_path):
    """`record` sizes each batch with the display tail written for the standing timeline —
    the last twelve rows. Reproduced against a harness emitting sixteen rows per cursor
    step: 48 rows published, 36 delivered, and E17-E20 / E33-E36 / E49-E52 gone from the
    live stream with nothing naming them. A batch may still be bounded — a busy daemon can
    publish hundreds of rows between two reads; what it may not do is cut without saying
    so: each advance counts what it does not show, and the live line says it."""
    from ouroboros.delegate_progress import WindowObservations
    from ouroboros.delegate_activity import window_observations, progress_text
    from types import SimpleNamespace

    seen = WindowObservations()
    timeline, per_step = [], 16
    for step in range(1, 4):
        timeline.extend({"type": "tool", "title": f"E{step * per_step - per_step + i}"}
                        for i in range(per_step))
        advance = seen.record({"timeline": list(timeline)}, step)
        assert len(advance.events) + advance.events_omitted == per_step, advance
        assert advance.events_omitted == per_step - 12, advance
        # The human's stream is the surface that had no marker whatsoever.
        ctx = SimpleNamespace(task_id="child", drive_root=tmp_path)
        (record, commit), = window_observations(ctx, "run-1", advance, after_seq=step - 1)
        assert "+4 earlier timeline rows not shown" in progress_text(record)
        commit()


def _timeline(*titles):
    """A daemon `get_run` detail carrying exactly these timeline rows, in this order."""
    return {"timeline": [{"type": "tool", "title": title, "severity": "info"}
                         for title in titles]}


def test_a_bounded_rolling_timeline_records_the_whole_batch_that_arrived():
    """A real daemon timeline is BOUNDED: past some depth its LENGTH stops growing while
    the cursor keeps moving, so the number of new rows can no longer be read from the
    length. The batch was then taken as a SINGLE tail row, and everything else that
    arrived between two polls vanished — from the live stream the human is watching AND
    from the record, with nothing disclosing the loss. Silent loss of an
    advance is the one failure this whole path exists to prevent.

    Nor does the CURSOR carry the count — that was the first answer here and it was wrong
    in both directions (see the `batch=4` and cursor-overshoot shapes below). The batch is
    read off the DATA: the longest overlap between the END of the previous tail and the
    START of this one is what survived the roll. The review's own shape is first — a full
    twelve-row tail that rolls by two, which is simply two events arriving inside one poll
    interval.
    """
    from ouroboros.delegate_progress import WindowObservations

    a = [f"A{i}" for i in range(1, 13)]
    seen = WindowObservations()

    first = seen.record(_timeline(*a), 12)
    assert [row["title"] for row in first.events] == a

    # THE defect: A1 and A2 fell off the front, B1 and B2 arrived, and the LENGTH is
    # unchanged. Both belong to this advance, in order.
    second = seen.record(_timeline(*a[2:], "B1", "B2"), 14)
    assert [row["title"] for row in second.events] == ["B1", "B2"]

    # A three-event roll from the same rolled state — the batch is a delta, not a
    # constant, and it is read against the PREVIOUS tail rather than from zero.
    third = seen.record(_timeline(*a[5:], "B1", "B2", "B3", "B4", "B5"), 17)
    assert [row["title"] for row in third.events] == ["B3", "B4", "B5"]

    # NOTHING changed on the timeline while the cursor moved (a journal entry that never
    # became a timeline row). There is no news, and inventing some would re-announce rows
    # the human was already shown.
    quiet = seen.record(_timeline(*a[5:], "B1", "B2", "B3", "B4", "B5"), 18)
    assert [row["title"] for row in quiet.events] == []

    # A tail with NOTHING in common with the previous one — a replaced timeline, not a
    # roll. All of it is new; it must not raise and must not over-slice into rows that
    # are not there.
    c = [f"C{i}" for i in range(1, 13)]
    fourth = seen.record(_timeline(*c), 35)
    assert [row["title"] for row in fourth.events] == c

    # Every observation is exactly one advance at its own cursor, whatever the batch was.
    assert [advance.seq for advance in (first, second, third, quiet, fourth)] == [12, 14, 17, 18, 35]


def test_a_daemon_that_adds_more_rows_than_cursor_steps_loses_none_of_them():
    """The cursor is not a row count, and this repo's own `_StreamingStub(batch=4)` is the
    proof: it publishes FOUR timeline rows for every single `lastSeq` step, which is what a
    harness whose journal entry carries several session events looks like. Reading the
    batch off the cursor took one row per step and three of every four vanished — end to
    end, a daemon that produced E1..E16 delivered E1..E12 and E16, with E13/E14/E15 gone
    from the live stream, and no `events_omitted` anywhere saying so.
    """
    from ouroboros.delegate_progress import WindowObservations

    rows = [f"E{i}" for i in range(1, 17)]
    seen, recorded = WindowObservations(), []
    for step in range(4):
        end = (step + 1) * 4                      # four rows per single cursor step
        window = rows[max(0, end - 12):end]       # ...through a twelve-row rolling tail
        recorded.extend(row["title"] for row in seen.record(_timeline(*window), step + 1).events)

    assert recorded == rows, "a row the daemon published never reached the record"


def test_a_growing_timeline_still_records_exactly_the_rows_that_are_new():
    """The control the rolling shapes above must not cost: while the list is still
    GROWING, the tail comparison has to land on the plain append, even when the cursor
    disagrees. A daemon whose `seq` counts more than the timeline shows — journal
    entries that never become timeline rows — must not inflate the batch beyond the
    rows that actually arrived, and must not re-report rows already recorded."""
    from ouroboros.delegate_progress import WindowObservations

    a = [f"A{i}" for i in range(1, 13)]
    seen = WindowObservations()
    assert [row["title"] for row in seen.record(_timeline(*a), 12).events] == a

    # Three rows appended; the cursor jumped far further than three.
    grown = seen.record(_timeline(*a, "B1", "B2", "B3"), 99)
    assert [row["title"] for row in grown.events] == ["B1", "B2", "B3"]

    # The same overshoot against a ROLLED tail, where the length cannot help either: one
    # row arrived, the cursor moved by three. Reading the batch off the cursor took the
    # last three rows and re-emitted A11 and A12 — rows already reported
    # — as new session events.
    seen = WindowObservations()
    seen.record(_timeline(*a), 12)
    rolled = seen.record(_timeline(*a[1:], "B1"), 15)
    assert [row["title"] for row in rolled.events] == ["B1"]


def test_the_standing_tail_is_adopted_as_history_only_for_a_caught_up_caller():
    """A view that attaches to a run which has been talking for a while must not announce
    the whole standing tail as news to the human's live stream. But the cursor is the
    CALLER's, and a caller BEHIND the daemon is
    asking for exactly those standing rows; adopting them there would drop the very batch
    it called for. Rows carry no cursor of their own, so it is all or nothing, and only
    the caught-up caller can be told nothing."""
    from ouroboros.delegate_progress import WindowObservations

    a = [f"A{i}" for i in range(1, 13)]
    detail = dict(_timeline(*a), lastSeq=12)

    caught_up = WindowObservations()
    caught_up.observe_baseline(detail, 12)
    assert [row["title"] for row in caught_up.record(dict(_timeline(*a, "B1"), lastSeq=13),
                                                     13).events] == ["B1"]

    behind = WindowObservations()
    behind.observe_baseline(detail, 4)
    assert [row["title"] for row in behind.record(detail, 12).events] == a
