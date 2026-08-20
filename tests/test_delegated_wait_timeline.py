"""The advance list and the rolling timeline a delegated wait reports.

Split verbatim out of ``tests/test_delegated_subagent_transport.py`` by theme. This
module owns the per-advance rows the wait records, the budget under which their labels
are shed, and the bounded timeline that must lose no row the daemon added.
"""

from __future__ import annotations

import json



from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _owned_gateway_uses_each_test_transport,
    _StreamingStub,
    _nanny_ctx,
)


def test_the_advance_list_is_a_list_not_a_count(tmp_path, monkeypatch):
    """A count would be cheaper and would also be a lie about what the run did. The list
    keeps one row per advance with its `seq` and `at_sec`; under budget pressure the rows
    shed their event LABELS oldest-first, and every shedding is disclosed ON its row — a
    shed label already reached the live stream, and the row itself is on the run in
    Claudexor's own timeline. (A BATCH-cut row is the one that reached neither, which is
    why it is counted rather than dropped; see the batch test.)"""
    from ouroboros.tool_capabilities import tool_result_limit

    import ouroboros.gateways.claudexor as gw
    import ouroboros.tools.delegate as delegate

    monkeypatch.setattr(gw, "ClaudexorGateway",
                        lambda *a, **k: _StreamingStub(batch=4, title="T" * 20_000))
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        task_id="t-a", route_id="some-route", model="m",
        project_id="prj", project_owned=False,
    )
    raw = delegate._delegate_wait(_nanny_ctx(tmp_path), "run-1", wait_sec=4, since_seq=0)
    delegate._CUSTODY.clear()

    assert len(raw) <= tool_result_limit("delegate_wait")
    payload = json.loads(raw)
    rows = payload["advances"]
    assert len(rows) >= 2 and all("seq" in row and "at_sec" in row for row in rows), rows
    assert rows[-1]["events"], "the newest advance keeps its labels"
    # No SILENT loss, in whichever regime the budget put this payload: a row that gave
    # up its labels says so, and a window whose head was dropped says that instead. The
    # label-shedding regime itself is pinned directly below, where a budget can be named
    # rather than inferred from a stub's byte sizes.
    assert all(row.get("events") or row.get("events_omitted") or "advances_omitted" in row
               for row in rows), rows


def test_a_verbose_timeline_tail_cannot_push_the_whole_payload_over_the_limit():
    """The advance list is sized against what the REST of the payload leaves, not a
    fixed share of the budget. `timeline_tail` carries harness-authored text — three
    fields, each bounded at 300 chars, twelve rows — so it can eat most of the limit on
    its own; a list that fitted its own third then rode on top and the whole result
    overflowed, where the generic truncator cut the JSON mid-structure and the model got
    something it could not parse. That is the same failure the measured bound exists to
    prevent, one level up."""
    from ouroboros.delegate_progress import WindowObservations, window_payload
    from ouroboros.loop_tool_execution import _truncate_tool_result
    from ouroboros.tool_capabilities import tool_result_limit

    limit = tool_result_limit("delegate_wait")
    # TWO long fields per row is what tips it: one alone stays under the limit.
    verbose = [{"title": "T" * 400, "type": "Y" * 400, "severity": "info"}
               for _ in range(12)]
    seen = WindowObservations()
    timeline = []
    for i in range(601):
        timeline = timeline + [{"title": "x" * 80, "type": "e"}]
        seen.record({"timeline": list(timeline)}, i + 1, i * 3)

    payload = window_payload(
        run_id="run-1", state="running", last_seq=601, window=1800,
        elapsed_seconds=1800, max_seconds=1800, waiting_on_user=False,
        detail={"timeline": verbose}, seen=seen, budget=limit)

    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    assert len(raw) <= limit, len(raw)
    assert _truncate_tool_result(raw, "delegate_wait", {}) == raw, \
        "the generic truncator had to cut a payload the sizing claimed would fit"
    assert json.loads(raw) == payload, "the model received unparseable JSON"
    # The tail keeps its full bounded self; it is the ADVANCE list that yields room.
    assert len(payload["timeline_tail"]) == 12
    assert payload["advances"], "the list yielded room without disappearing"


def test_the_advance_list_yields_entirely_rather_than_pushing_the_payload_over():
    """One notch past the sibling above, where the residual cannot hold even the FLOOR.
    The fit loop kept a 400-char floor for the advance list no matter what was left, so
    the floor itself overflowed: twelve tail rows with all three harness fields at 20 000
    chars left ~400 chars once the closing `note` was reserved, the floor shipped 422 on
    top, and the 15 018-char result crossed a 15 000 limit — where the generic truncator
    cut the JSON mid-structure and the model got a payload it could not parse. The
    payload WITHOUT the list measured 14 596, so this is the module's own floor
    overflowing, not harness text that no sizing could have fitted.

    The floor is an ask, not a guarantee: the list drops to its omission marker, and the
    drop is disclosed where the list stood."""
    from ouroboros.delegate_progress import WindowObservations, window_payload
    from ouroboros.loop_tool_execution import _truncate_tool_result
    from ouroboros.tool_capabilities import tool_result_limit

    limit = tool_result_limit("delegate_wait")
    verbose = [{"title": "T" * 20_000, "type": "Y" * 20_000, "severity": "S" * 20_000}
               for _ in range(12)]
    seen = WindowObservations()
    timeline = []
    for i in range(12):
        timeline = timeline + [{"title": "x" * 80, "type": "e"}]
        seen.record({"timeline": list(timeline)}, i + 1, i * 3)

    payload = window_payload(
        run_id="run-1", state="running", last_seq=12, window=1800,
        elapsed_seconds=1800, max_seconds=1800, waiting_on_user=False,
        detail={"timeline": verbose}, seen=seen, budget=limit)
    raw = json.dumps(payload, ensure_ascii=False, indent=2)

    assert len(raw) <= limit, len(raw)
    assert _truncate_tool_result(raw, "delegate_wait", {}) == raw, \
        "the generic truncator had to cut a payload the sizing claimed would fit"
    assert json.loads(raw) == payload, "the model received unparseable JSON"
    # Nothing vanished quietly: the list says it yielded, how much, and through where.
    marker, = payload["advances"]
    assert marker["advances_omitted"] == len(seen.advances) == 12, marker
    assert marker["omitted_through_seq"] == 12, marker
    # ...and the note points at where the omitted rows ACTUALLY are. It used to say they
    # "were streamed live and are in the event log": the first half is untrue of a batch
    # bigger than the display tail (those rows never reached the live line at all) and
    # the second of all of them (Ouroboros persists no timeline; the daemon's own run
    # directory holds it). A recovery instruction that sends the reader to a log that
    # never had the rows is worse than no instruction.
    assert "Claudexor's own timeline" in marker["note"], marker
    assert "event log" not in marker["note"], marker


def test_label_shedding_is_disclosed_on_the_row_that_gave_them_up():
    """The regime `test_the_advance_list_is_a_list_not_a_count` cannot name: a budget
    that forces labels out but keeps every advance. Oldest-first, disclosed per row,
    newest labels kept — a silently emptied `events` list would be a lie about what the
    run did, and this is the only place that lie is cheap to tell."""
    from ouroboros.delegate_progress import WindowObservations

    seen = WindowObservations()
    timeline = []
    for i in range(8):
        timeline = timeline + [{"title": "L" * 200, "type": "harness.event"}]
        seen.record({"timeline": list(timeline)}, i + 1, i)

    rows = seen.rows(1200)          # room for the spine and a couple of label sets

    assert [row["seq"] for row in rows] == list(range(1, 9)), "no advance was dropped"
    assert rows[-1]["events"], "the newest advance keeps its labels"
    shed = [row for row in rows if "events_omitted" in row]
    assert shed, rows
    assert all(row["events"] == [] for row in shed), shed
    assert all(row["events_omitted"] > 0 for row in shed), shed
    assert [row["seq"] for row in shed] == sorted(row["seq"] for row in shed), shed
    assert shed[0]["seq"] == 1, "shedding starts at the OLDEST advance"
    assert len(json.dumps(rows, ensure_ascii=False, indent=2)) <= 1200


def test_a_batch_bigger_than_the_display_tail_says_how_much_it_is_not_showing():
    """The cut that had NO vocabulary at all: not the budget's, the OBSERVATION's.

    `record` sized each batch with the display tail written for the standing timeline —
    the last twelve rows, head dropped in silence. That is right for a timeline whose
    head the model has already seen and wrong for a batch whose head arrived one poll
    ago. Reproduced against a harness emitting sixteen rows per cursor step: 48 rows
    published in-window, 36 delivered, and E17-E20 / E33-E36 / E49-E52 gone from the live
    stream AND from `advances`, with the rows carrying only ['at_sec', 'events', 'seq'].

    A batch may still be bounded — a busy daemon can publish hundreds of rows between two
    three-second polls. What it may not do is cut without saying so, in a module whose
    whole point is that an omission is disclosed."""
    from ouroboros.delegate_progress import WindowObservations, live_line

    seen = WindowObservations()
    timeline, per_step = [], 16
    for step in range(1, 4):
        timeline.extend({"type": "tool", "title": f"E{step * per_step - per_step + i}"}
                        for i in range(per_step))
        seen.record({"timeline": list(timeline)}, step, step)

    rows = seen.rows(100_000)             # a budget that forces no shedding of its own
    assert len(rows) == 3, rows
    for row in rows:
        assert "events_omitted" in row, f"the batch was cut and said nothing: {row}"
        assert len(row["events"]) + row["events_omitted"] == per_step, row
        assert row["events_omitted"] == per_step - 12, row
    # The human's stream is the surface that had no marker whatsoever.
    assert "+4 earlier in this batch" in live_line("run-1", seen.advances[0])
    # And the two cuts ADD rather than one overwriting the other: a row whose batch was
    # already cut, then shed for budget, must report ALL sixteen and not just the twelve
    # this payload dropped.
    tight = seen.rows(400)
    assert [row["seq"] for row in tight if "seq" in row] == [1, 2, 3], tight
    assert [row["events_omitted"] for row in tight if "seq" in row] == [per_step] * 3, tight


def test_a_long_busy_windows_advance_list_is_measured_not_estimated():
    """The sibling of the verbose-harness case, from the other direction: not a handful
    of enormous labels but HIGH CARDINALITY — 601 advances of twelve ordinary titles,
    which is simply what a healthy run looks like when it is watched for a long window.
    It is not an exotic shape either: the 1800s ceiling divided by the 3s poll interval
    is 600, so this is the WORST case the wait can actually produce, not a synthetic one.

    The bound used to be ESTIMATED: a running total decremented by each shed row, and
    then a survivor count of `budget // 40` on the assumption that a bare spine row
    costs about forty characters. Both assumptions ran UNDER the truth (a rendered
    `{"seq": …, "at_sec": …, "events": [], "events_omitted": …}` costs far more than
    forty, and the caller renders with `indent=2`, which the estimate never accounted
    for), so this shape left here already over `tool_result_limit("delegate_wait")`. The
    generic truncator then cut the JSON mid-structure and the model was handed a payload
    it could not parse AT ALL — strictly worse than any amount of shedding, because a
    disclosed omission is still readable and a severed object is not.

    So the size is MEASURED, with the caller's own rendering, and re-measured after every
    shed — including the head shedding, which pays for its own marker row."""
    from ouroboros.loop_tool_execution import _truncate_tool_result
    from ouroboros.tool_capabilities import tool_result_limit

    from ouroboros import delegate_progress as progress

    limit = tool_result_limit("delegate_wait")
    advances, labels = 601, 12
    # The daemon serves the timeline as a growing LIST, not a delta — so drive `record`
    # with that shape rather than hand-building rows, and each poll's batch is fresh.
    timeline = []
    seen = progress.WindowObservations()
    for seq in range(1, advances + 1):
        timeline.extend({"type": "tool", "title": f"advance {seq:04d} · " + "T" * 65,
                         "severity": "info"} for _ in range(labels))
        seen.record({"timeline": timeline}, seq, seq)
    assert len(seen.advances) == advances

    payload = progress.window_payload(
        run_id="run-1", state="running", last_seq=advances, window=600,
        elapsed_seconds=600, max_seconds=1800, waiting_on_user=False,
        detail={"timeline": timeline}, seen=seen, budget=limit)
    # Exactly how `_delegate_wait` renders it, which is the rendering that has to fit.
    raw = json.dumps(payload, ensure_ascii=False, indent=2)

    # THE defect: what the model is handed arrives WHOLE and parses.
    assert len(raw) <= limit, len(raw)
    delivered = _truncate_tool_result(raw, "delegate_wait", {})
    assert delivered == raw, "the generic truncator had to cut this payload"
    assert json.loads(delivered) == payload, "the model received unparseable JSON"
    rows = payload["advances"]
    # The list is sized against what is LEFT of the budget, not a fixed share: a share
    # bounds only itself, and `timeline_tail` (harness-authored text) can eat most of
    # the limit on its own — which is how a sub-block that fitted its third still
    # overflowed the result. So the invariant is the WHOLE payload above, and here that
    # the list did take a real share of it rather than being emptied to make room.
    assert len(json.dumps(rows, ensure_ascii=False, indent=2)) < len(raw)

    # It shed from the HEAD and it SAYS so, with the accounting adding up: a payload that
    # pretended the window started later than it did is the one shape forbidden here.
    marker, kept = rows[0], rows[1:]
    assert kept, rows
    assert marker["advances_omitted"] == advances - len(kept), (marker, len(kept))
    assert marker["omitted_through_seq"] == kept[0]["seq"] - 1, (marker, kept[0])
    # ...and the note points at where the omitted rows ACTUALLY are. It used to say they
    # "were streamed live and are in the event log": the first half is untrue of a batch
    # bigger than the display tail (those rows never reached the live line at all) and
    # the second of all of them (Ouroboros persists no timeline; the daemon's own run
    # directory holds it). A recovery instruction that sends the reader to a log that
    # never had the rows is worse than no instruction.
    assert "Claudexor's own timeline" in marker["note"], marker
    assert "event log" not in marker["note"], marker
    assert [row["seq"] for row in kept] == list(range(kept[0]["seq"], advances + 1))
    assert kept[-1]["seq"] == advances, "the NEWEST advance is never the one shed"


def _timeline(*titles):
    """A daemon `get_run` detail carrying exactly these timeline rows, in this order."""
    return {"timeline": [{"type": "tool", "title": title, "severity": "info"}
                         for title in titles]}


def test_a_bounded_rolling_timeline_records_the_whole_batch_that_arrived():
    """A real daemon timeline is BOUNDED: past some depth its LENGTH stops growing while
    the cursor keeps moving, so the number of new rows can no longer be read from the
    length. The batch was then taken as a SINGLE tail row, and everything else that
    arrived between two polls vanished — from the live stream the human is watching AND
    from `advances`, with nothing on the payload disclosing the loss. Silent loss of an
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

    first = seen.record(_timeline(*a), 12, 0)
    assert [row["title"] for row in first.events] == a

    # THE defect: A1 and A2 fell off the front, B1 and B2 arrived, and the LENGTH is
    # unchanged. Both belong to this advance, in order.
    second = seen.record(_timeline(*a[2:], "B1", "B2"), 14, 3)
    assert [row["title"] for row in second.events] == ["B1", "B2"]

    # A three-event roll from the same rolled state — the batch is a delta, not a
    # constant, and it is read against the PREVIOUS tail rather than from zero.
    third = seen.record(_timeline(*a[5:], "B1", "B2", "B3", "B4", "B5"), 17, 6)
    assert [row["title"] for row in third.events] == ["B3", "B4", "B5"]

    # NOTHING changed on the timeline while the cursor moved (a journal entry that never
    # became a timeline row). There is no news, and inventing some would re-announce rows
    # the human was already shown.
    quiet = seen.record(_timeline(*a[5:], "B1", "B2", "B3", "B4", "B5"), 18, 7)
    assert [row["title"] for row in quiet.events] == []

    # A tail with NOTHING in common with the previous one — a replaced timeline, not a
    # roll. All of it is new; it must not raise and must not over-slice into rows that
    # are not there.
    c = [f"C{i}" for i in range(1, 13)]
    fourth = seen.record(_timeline(*c), 35, 9)
    assert [row["title"] for row in fourth.events] == c

    # Every observation is still exactly one advance, in order, whatever the batch was.
    assert [advance.seq for advance in seen.advances] == [12, 14, 17, 18, 35]
    assert [advance.at_sec for advance in seen.advances] == [0, 3, 6, 7, 9]


def test_a_daemon_that_adds_more_rows_than_cursor_steps_loses_none_of_them():
    """The cursor is not a row count, and this repo's own `_StreamingStub(batch=4)` is the
    proof: it publishes FOUR timeline rows for every single `lastSeq` step, which is what a
    harness whose journal entry carries several session events looks like. Reading the
    batch off the cursor took one row per step and three of every four vanished — end to
    end, a daemon that produced E1..E16 delivered E1..E12 and E16, with E13/E14/E15 gone
    from the live stream and from `advances`, and no `events_omitted` anywhere saying so.
    """
    from ouroboros.delegate_progress import WindowObservations

    rows = [f"E{i}" for i in range(1, 17)]
    seen, recorded = WindowObservations(), []
    for step in range(4):
        end = (step + 1) * 4                      # four rows per single cursor step
        window = rows[max(0, end - 12):end]       # ...through a twelve-row rolling tail
        recorded.extend(row["title"] for row in seen.record(_timeline(*window), step + 1, step).events)

    assert recorded == rows, "a row the daemon published never reached the record"
    assert [advance.seq for advance in seen.advances] == [1, 2, 3, 4]


def test_a_growing_timeline_still_records_exactly_the_rows_that_are_new():
    """The control the rolling shapes above must not cost: while the list is still
    GROWING, the tail comparison has to land on the plain append, even when the cursor
    disagrees. A daemon whose `seq` counts more than the timeline shows — journal
    entries that never become timeline rows — must not inflate the batch beyond the
    rows that actually arrived, and must not re-report rows already recorded."""
    from ouroboros.delegate_progress import WindowObservations

    a = [f"A{i}" for i in range(1, 13)]
    seen = WindowObservations()
    assert [row["title"] for row in seen.record(_timeline(*a), 12, 0).events] == a

    # Three rows appended; the cursor jumped far further than three.
    grown = seen.record(_timeline(*a, "B1", "B2", "B3"), 99, 3)
    assert [row["title"] for row in grown.events] == ["B1", "B2", "B3"]

    # The same overshoot against a ROLLED tail, where the length cannot help either: one
    # row arrived, the cursor moved by three. Reading the batch off the cursor took the
    # last three rows and re-emitted A11 and A12 — rows this window had already reported
    # — as new session events.
    seen = WindowObservations()
    seen.record(_timeline(*a), 12, 0)
    rolled = seen.record(_timeline(*a[1:], "B1"), 15, 3)
    assert [row["title"] for row in rolled.events] == ["B1"]


def test_the_standing_tail_is_adopted_as_history_only_for_a_caught_up_caller():
    """A wait that attaches to a run which has been talking for a while must not announce
    the whole standing tail as this window's news — to the human's live stream or to the
    model. But `since_seq` is the CALLER's cursor, and a caller BEHIND the daemon is
    asking for exactly those standing rows; adopting them there would drop the very batch
    it called for. Rows carry no cursor of their own, so it is all or nothing, and only
    the caught-up caller can be told nothing."""
    from ouroboros.delegate_progress import WindowObservations

    a = [f"A{i}" for i in range(1, 13)]
    detail = dict(_timeline(*a), lastSeq=12)

    caught_up = WindowObservations()
    caught_up.observe_baseline(detail, 12)
    assert [row["title"] for row in caught_up.record(dict(_timeline(*a, "B1"), lastSeq=13),
                                                     13, 1).events] == ["B1"]

    behind = WindowObservations()
    behind.observe_baseline(detail, 4)
    assert [row["title"] for row in behind.record(detail, 12, 1).events] == a
