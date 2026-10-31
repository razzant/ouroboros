"""One engine-keyed utterance from the run journal to every reader that shows it.

The engine names a streamed text block with an opaque ``speech`` key: fragments, then at
most one complete text, or an abandon statement; the typed final answer is never keyed
(Claudexor's speech interface contract, v2). These tests serve such journals through
the real gateway, wait, carrier, delivery and history read, and check what each reader
shows: one utterance per (run, actor, key), equal words under other keys kept, the typed
final answer apart, and every original line retained. Hand-written journals pin the edge
cases; the engine's own normalized journals of four recorded CLI 2.1.311 streams
(``tests/fixtures/claudexor_speech``, provenance beside them) are served verbatim, split
at every poll boundary, through the exact and window paths and the wait's delivery.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import delegate_activity, delegate_progress
from ouroboros.delegate_custody import RunCustody
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.subagent_messages import delegated_activity_meta
from ouroboros.utils import truncate_within_limit
from supervisor import events_chat_delivery, message_bus
from tests import _dialogue as dialogue_test
from tests.test_delegated_activity import Daemon, _agent, _ctx, _gateway, _harness, _records, _run_event

CLAUDE = ("claude", "a01")
# The adapter's key shape (consumers compare it whole and never parse it).
K1, K2, K3 = "claude:ses-1:1:0", "claude:ses-1:2:0", "claude:ses-1:3:0"


def _said(seq, text, key, op, *, actor=CLAUDE):
    event = _harness(seq, "message", text=text, actor=actor, delta=op == "fragment")
    event["payload"]["speech"] = {"key": key, "op": op}
    return event


def _stream(first_seq, pieces, key, *, complete=None, actor=CLAUDE):
    """Fragments of one block, then its complete text (default: the fragments joined)."""
    events = [_said(first_seq + i, piece, key, "fragment", actor=actor) for i, piece in enumerate(pieces)]
    if complete is not False:
        events.append(_said(first_seq + len(pieces), "".join(pieces) if complete is None else complete, key, "complete",
                            actor=actor))
    return events


def _final(seq, text, *, actor=CLAUDE, source="result"):
    event = _harness(seq, "message", text=text, actor=actor)
    event["payload"].update(final=True, payload={"final_source": source})
    return event


def _abandoned(seq, key, *, actor=CLAUDE):
    event = _harness(seq, "status", actor=actor)
    event["payload"].update(speech={"key": key, "op": "abandoned"},
                            payload={"code": "speech_abandoned", "from_block_index": 0})
    return event


def _tool(seq, name="Read"):
    return _harness(seq, "tool_call", tool={"name": name, "target": "blog/rss_import.py"})


def _one(tmp_path, events):
    daemon = Daemon([events])
    daemon.revealed = 1
    record, = _records(_ctx(tmp_path, []), _gateway(daemon), events[-1]["seq"])
    return record


def _read(ctx, daemon, gateway, step, through, after=0):
    """The records of one advance once the journal holds ``step`` steps (no detail poll here)."""
    daemon.revealed = step
    return _records(ctx, gateway, through, after=after)


def _speech_lines(record):
    return [line for line in delegate_activity.progress_text(record).splitlines()[1:]
            if line.startswith("claude") or line.startswith("codex")]


PARAGRAPH = ("Both duplicates come from one feed item: the importer keys entries by title, and the feed "
             "renamed that item. I will key them by GUID and re-run the import tests.")


def _pieces(text, n):
    size = -(-len(text) // n)
    return [text[i:i + size] for i in range(0, len(text), size)]


def test_a_keyed_stream_and_its_complete_text_are_one_utterance_with_every_original_kept(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes

    pieces = _pieces(PARAGRAPH, 12)
    events = [_tool(1), _harness(2, "tool_result", tool={"name": "Read", "status": "ok"}),
              *_stream(3, pieces, K1), _tool(16, "Edit")]
    record = _one(tmp_path, events)
    part, = record["parts"]
    assert part == {"kind": "message", "actor": "claude/a01", "text": PARAGRAPH, "speech": {"key": K1, "op": "complete"},
                    "seq": 3, "last_seq": 15, "fragments": 12, "chars": len(PARAGRAPH)}, "no cuts: the text is the complete one"
    assert record["technical"]["count"] == 3 and record["technical"]["seqs"] == [[1, 2], [16, 16]]
    assert _speech_lines(record) == [f"claude · a01: {PARAGRAPH}"], "the frame text says it once"
    assert delegate_activity.card_text(record, 650).count(PARAGRAPH) == 1
    assert delegated_activity_meta(record, task_id="child") == record
    original = read_actor_source_bytes(tmp_path, "child", record["source"]["ref"]).decode().splitlines()
    assert [json.loads(line) for line in original] == events, "the draft and the complete text both stay in the source"


@pytest.mark.parametrize("pieces,complete,updated", [
    (["teh", " cat"], "the cat", True),           # a correction of equal length
    (["the", " cat"], "the cat", False),
    (["the", " cat"], "the cat.", True),
])
def test_a_complete_text_differing_from_its_stream_is_updated_whatever_its_length(tmp_path, pieces, complete, updated):
    record = _one(tmp_path, _stream(1, pieces, K1, complete=complete))
    part, = record["parts"]
    assert part["text"] == complete and part.get("updated", False) is updated
    assert _speech_lines(record) == [f"claude · a01{' (updated)' if updated else ''}: {complete}"]


def test_equal_words_under_other_keys_actors_or_unkeyed_events_stay_distinct(tmp_path):
    again = "Re-running the import tests."
    events = [*_stream(1, ["Re-running", " the import tests."], K1), _tool(4, "Bash"),
              *_stream(5, ["Re-running", " the import tests."], K2),
              *_stream(8, ["Re-running", " the import tests."], K1, actor=("claude", "a02")),
              _harness(11, "message", text="Re-running", delta=True), _harness(12, "message", text=again)]
    record = _one(tmp_path, events)
    assert [(p["actor"], p.get("speech", {}).get("key"), p["text"]) for p in record["parts"]] == [
        ("claude/a01", K1, again), ("claude/a01", K2, again), ("claude/a02", K1, again),
        ("claude/a01", None, "Re-running"), ("claude/a01", None, again)], \
        "identity is the key within its run and actor; unkeyed text keeps the delta rule"
    assert len(_speech_lines(record)) == 5


def test_a_joined_snapshot_stays_unkeyed_beside_the_drafts_it_repeats(tmp_path):
    # Citations mode: each block's fragments are keyed, no block completes, and the CLI's joined
    # snapshot of both blocks arrives unkeyed after them. Nothing links it to the drafts.
    events = [_said(1, "Part one.", K1, "fragment"), _said(2, " Part two.", K2, "fragment"),
              _harness(3, "message", text="Part one. Part two.")]
    record = _one(tmp_path, events)
    assert [(p.get("speech"), p["text"]) for p in record["parts"]] == [
        ({"key": K1, "op": "fragment"}, "Part one."), ({"key": K2, "op": "fragment"}, " Part two."),
        (None, "Part one. Part two.")], "equal words never stand in for the identity the engine did not give"


def test_technical_frames_between_fragments_neither_split_nor_join_the_utterance(tmp_path):
    events = [_said(1, "Checking the", K1, "fragment"),
              _harness(2, "tool_call", tool={"name": "Task"}),                 # a subagent's frame: unkeyed
              _harness(3, "message", text="sub-agent words", delta=True),
              _harness(4, "status"),
              _said(5, " importer.", K1, "fragment"), _said(6, "Checking the importer.", K1, "complete")]
    record = _one(tmp_path, events)
    assert [(p.get("speech"), p["text"], p.get("seq"), p.get("last_seq")) for p in record["parts"]] == [
        ({"key": K1, "op": "complete"}, "Checking the importer.", 1, 6), (None, "sub-agent words", 3, 3)]
    assert record["technical"]["seqs"] == [[2, 2], [4, 4]]


@pytest.mark.parametrize("complete,updated", [(PARAGRAPH, False), (PARAGRAPH.replace("GUID", "guid"), True)])
def test_a_split_poll_completes_the_draft_the_previous_record_showed(tmp_path, complete, updated):
    pieces = _pieces(PARAGRAPH, 6)
    daemon = Daemon([[_tool(1), *_stream(2, pieces[:3], K1, complete=False)],
                     [*_stream(5, pieces[3:], K1, complete=complete), _tool(9, "Edit")]])
    gateway, ctx = _gateway(daemon), _ctx(tmp_path, [])
    draft, = _read(ctx, daemon, gateway, 1, 4)
    done, = _read(ctx, daemon, gateway, 2, 9, after=4)
    assert draft["parts"][0]["speech"] == {"key": K1, "op": "fragment"} and draft["parts"][0]["delta"] is True
    assert draft["parts"][0]["text"] == "".join(pieces[:3])
    part, = done["parts"]
    assert (part["speech"], part["text"], part["seq"], part.get("updated", False)) == (
        {"key": K1, "op": "complete"}, complete, 5, updated), "the whole draft is compared across records"


@pytest.mark.parametrize("lose", ["restart", "eviction"])
def test_a_draft_begun_before_this_process_never_reads_as_updated(tmp_path, lose):
    pieces = _pieces(PARAGRAPH, 4)
    daemon = Daemon([_stream(1, pieces[:2], K1, complete=False), _stream(3, pieces[2:], K1, complete=False),
                     [_said(5, PARAGRAPH, K1, "complete"), _tool(6),
                      *_stream(7, ["Next", " block."], K2, complete="Next block, revised.")]])
    gateway, ctx = _gateway(daemon), _ctx(tmp_path, [])
    _read(ctx, daemon, gateway, 1, 2)
    _read(ctx, daemon, gateway, 2, 4, after=2)
    if lose == "restart":
        delegate_activity.reset_process_memo()
    else:
        for index in range(delegate_activity._MEMO_LIMIT):
            delegate_activity._memo("child", f"other-{index}")
    record, = _read(ctx, daemon, gateway, 3, 9, after=4)
    assert record["after_seq"] == 2, "it re-reads the last retained range, not the draft's beginning"
    first, second = record["parts"]
    assert first["text"] == PARAGRAPH and not {"updated", "carried"} & first.keys(), \
        "only half the draft was seen here: a difference is unknown, never invented"
    assert _speech_lines(record)[0] == f"claude · a01: {PARAGRAPH}", "unknown keeps every word, claims nothing"
    assert second["updated"] is True, "a later block of an actor this process saw began in its view"


def test_an_abandoned_draft_is_marked_and_a_replay_with_the_same_words_stays_its_own(tmp_path):
    events = [*_stream(1, ["Checking", " the"], K1, complete=False), _abandoned(3, K1), _harness(4, "status"),
              *_stream(5, ["Checking", " the", " file first."], K2), _tool(9)]
    record = _one(tmp_path, events)
    assert [(p["speech"], p["text"]) for p in record["parts"]] == [
        ({"key": K1, "op": "fragment"}, "Checking the"), ({"key": K2, "op": "complete"}, "Checking the file first.")]
    assert record["speech_abandoned"] == [{"actor": "claude/a01", "key": K1, "seq": 3}]
    assert record["technical"]["seqs"] == [[3, 4], [9, 9]], "the statement stays a counted event"
    assert _speech_lines(record) == ["claude · a01 (abandoned): Checking the", "claude · a01: Checking the file first."]
    assert delegated_activity_meta(record, task_id="child") == record


def test_an_abandon_read_after_its_draft_marks_it_from_the_later_record(tmp_path):
    daemon = Daemon([_stream(1, ["Checking", " the"], K1, complete=False), [_abandoned(3, K1), _tool(4)]])
    gateway, ctx = _gateway(daemon), _ctx(tmp_path, [])
    draft, = _read(ctx, daemon, gateway, 1, 2)
    marked, = _read(ctx, daemon, gateway, 2, 4, after=2)
    assert draft["parts"][0]["speech"]["op"] == "fragment" and "speech_abandoned" not in draft
    assert marked["parts"] == [] and marked["speech_abandoned"] == [{"actor": "claude/a01", "key": K1, "seq": 3}]
    assert delegate_activity._memo("child", "run-1").drafts["claude/a01"] == (K1, None, 2, None), "ended, nothing to skip"


ANSWER = "Fixed the duplicate posts; all 15 import tests pass."


@pytest.mark.parametrize("reply,final,lines", [
    pytest.param(_stream(1, ["Fixed the duplicate posts;", " all 15 import tests pass."], K1), _final(4, ANSWER),
                 [f"claude · a01: {ANSWER}", f"claude · a01 (answer): {ANSWER}"], id="streamed"),
    pytest.param([_said(1, ANSWER, K1, "complete")], _final(2, ANSWER),
                 [f"claude · a01: {ANSWER}", f"claude · a01 (answer): {ANSWER}"], id="nonstreamed"),
    pytest.param([_said(1, "Done.", K1, "complete")], _final(2, ANSWER),
                 ["claude · a01: Done.", f"claude · a01 (answer): {ANSWER}"], id="unique-words"),
    pytest.param([], _final(1, ANSWER), [f"claude · a01 (answer): {ANSWER}"], id="result-only"),
    pytest.param([], _final(1, '{"fixed": 3}', source="structured_output"),
                 ['claude · a01 (answer): {"fixed": 3}'], id="structured"),
    pytest.param([_said(1, "Starting.", K1, "complete", actor=("codex", "a02"))], _final(2, ANSWER),
                 ["codex · a02: Starting.", f"claude · a01 (answer): {ANSWER}"], id="another-actor-spoke"),
])
def test_the_typed_final_answer_is_its_own_part_and_keeps_its_words_for_text_readers(tmp_path, reply, final, lines):
    record = _one(tmp_path, [*reply, final])
    answer = record["parts"][-1]
    assert answer["final"] is True and answer["text"] == final["payload"]["text"] and "speech" not in answer, \
        "chosen by type, never by comparing words; never keyed"
    assert _speech_lines(record) == lines, "a text-only reader cannot open the card's folded answer"
    assert final["payload"]["text"] in delegate_activity.card_text(record, 650)


@pytest.mark.parametrize("split", [None, 2, 3])          # 3: the answer alone in a later poll
def test_the_answer_line_is_the_same_whichever_poll_carries_it(tmp_path, split):
    events = [*_stream(1, ["Fixed the duplicate posts;", " all 15 import tests pass."], K1), _final(4, ANSWER)]
    steps = [events] if split is None else [events[:split], events[split:]]
    daemon = Daemon(steps)
    gateway, ctx = _gateway(daemon), _ctx(tmp_path, [])
    records, after = [], 0
    for step in range(1, len(steps) + 1):
        records += _read(ctx, daemon, gateway, step, steps[step - 1][-1]["seq"], after=after)
        after = steps[step - 1][-1]["seq"]
    answers = [line for record in records for line in _speech_lines(record) if "(answer)" in line]
    assert answers == [f"claude · a01 (answer): {ANSWER}"], "the poll boundary never decides whether its words are shown"
    assert sum(delegate_activity.progress_text(record).count(f"(answer): {ANSWER}") for record in records) == 1


def test_window_rows_carry_the_key_whole_and_fold_one_utterance(tmp_path):
    key = "claude:" + "s" * 400 + ":1:0"            # longer than the row label bound: compared whole
    detail = {"lastSeq": 5, "timeline": [
        {"type": "harness.event", "title": "Check", "detail": "Check", "textKind": "message", "textDelta": True,
         "speechKey": key, "speechOp": "fragment", "harnessId": "claude", "attemptId": "a01"},
        {"type": "harness.event", "title": "ing", "detail": "ing", "textKind": "message", "textDelta": True,
         "speechKey": key, "speechOp": "fragment", "harnessId": "claude", "attemptId": "a01"},
        {"type": "harness.event", "title": "Checking.", "detail": "Checking.", "textKind": "message",
         "speechKey": key, "speechOp": "complete", "harnessId": "claude", "attemptId": "a01"},
        {"type": "harness.event", "title": "status", "speechKey": "claude:x:2:0", "speechOp": "abandoned",
         "harnessId": "claude", "attemptId": "a01"},
        {"type": "harness.event", "title": "Checking.", "detail": "Checking.", "textKind": "message", "textFinal": True,
         "harnessId": "claude", "attemptId": "a01"},
    ]}
    advance = delegate_progress.WindowObservations().record(detail, 5, 0)
    assert [row.get("speechKey") for row in advance.events] == [key, key, key, "claude:x:2:0", None]
    assert advance.events[-1]["textFinal"] is True and "textFinal" not in advance.events[0]
    assert delegate_progress.live_line("run-1", advance).endswith(": Checking. · [claude/a01] status · Checking.")
    daemon = Daemon([[_harness(1, "message", text="x")]], stream=False)
    daemon.revealed = 1
    record, = _records(_ctx(tmp_path, []), _gateway(daemon), 5, rows=advance.events)
    assert record["source"]["kind"] == "timeline_window"
    assert [(p.get("speech"), p.get("final"), p["text"]) for p in record["parts"]] == [
        ({"key": key, "op": "complete"}, None, "Checking."), (None, True, "Checking.")]
    assert "updated" not in record["parts"][0], "bounded window labels are never compared"
    assert record["speech_abandoned"] == [{"actor": "claude/a01", "key": "claude:x:2:0"}]


@pytest.mark.parametrize("draft,complete,updated", [
    (["🛰", "️ ok"], "🛰️ ok", False),
    (["a" * 4000, "😀" * 3000], "a" * 4000 + "😀" * 3000, False),
    (["a" * 4000, "😀" * 3000], "a" * 4000 + "😀" * 2999 + "🙂", True),   # differs only past the preview
], ids=["split-cluster", "long-same", "long-differs-past-preview"])  # generated ids overflow Windows' env limit
def test_unicode_and_preview_cuts_never_decide_whether_a_complete_text_differs(tmp_path, draft, complete, updated):
    record = _one(tmp_path, _stream(1, draft, K1, complete=complete))
    part, = record["parts"]
    assert part.get("updated", False) is updated and part["chars"] == len(complete)
    assert bool(part.get("truncated")) is (len(complete) > 6000)


class Lossy(Daemon):
    """``Daemon`` whose stream never serves the ``missing`` seqs and breaks the ``unreadable`` lines."""

    def __init__(self, steps, *, missing=(), unreadable=(), **kwargs):
        super().__init__(steps, **kwargs)
        self.missing, self.unreadable = set(missing), set(unreadable)

    def handler(self, request):
        if request.url.path != f"/v2/runs/{self.run_id}/events":
            return super().handler(request)
        after, body = int(request.headers.get("last-event-id") or 0), ": connected\n\n"
        for event in self.journal:
            if event["seq"] > after and event["seq"] not in self.missing:
                data = '{"seq": ' if event["seq"] in self.unreadable else json.dumps(event)
                body += f"id: {event['seq']}\nevent: {event['type']}\ndata: {data}\n\n"
        end = "event: end\ndata: {}\n\n" if self.revealed == len(self.steps) else ""
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body + end)


@pytest.mark.parametrize("lose", ["missing", "unreadable"])
def test_a_draft_with_a_fragment_that_never_passed_here_is_not_proven_either_way(tmp_path, lose):
    daemon = Lossy([_stream(1, ["the", " big", " cat"], K1),                     # seq 2, " big", is lost
                    _stream(5, ["Next", " block."], K2, complete="Next block, revised."),
                    _stream(8, ["Last", " one."], K3, complete="Last one, revised.")], **{lose: [2]})
    gateway, ctx = _gateway(daemon), _ctx(tmp_path, [])
    lost, = _read(ctx, daemon, gateway, 1, 4)
    part, = lost["parts"]
    assert part["text"] == "the big cat" and not {"updated", "carried"} & part.keys(), \
        "an exact range that lost an event proves no draft whole: unknown, never a difference or an equality"
    assert _speech_lines(lost) == ["claude · a01: the big cat"], "its words stay whole for text readers"
    _read(ctx, daemon, gateway, 2, 7, after=4)
    proven, = _read(ctx, daemon, gateway, 3, 10, after=7)
    assert proven["parts"][0]["updated"] is True, "a later block begun behind one seen whole is proven again"


LOST = [_said(1, "the", K1, "fragment"), _said(2, " big", K1, "fragment"), _said(3, " cat", K1, "fragment"), _tool(4)]


@pytest.mark.parametrize("bounds", [[4], [1, 4]], ids=["one-poll", "across-polls"])
@pytest.mark.parametrize("lose", ["missing", "unreadable"])
def test_words_are_never_joined_across_an_event_this_reader_did_not_get_whole(tmp_path, lose, bounds):
    steps = [[event for event in LOST if low < event["seq"] <= high] for low, high in zip([0, *bounds], bounds)]
    daemon = Lossy(steps, **{lose: [2]})                                # " big" never reaches the reader whole
    gateway, ctx, records = _gateway(daemon), _ctx(tmp_path, []), []
    for step, (after, through) in enumerate(zip([0, *bounds], bounds), start=1):
        records += _read(ctx, daemon, gateway, step, through, after=after)
    assert [part["text"] for record in records for part in record["parts"]] == ["the", " cat"], \
        "the words on either side stay apart, never 'the cat'"
    reason = "not_served" if lose == "missing" else "unreadable"
    assert records[-1]["gaps"] == [{"after_seq": 1, "through_seq": 2, "reason": reason}], "the record names what it lost"
    assert f"not read whole: seq 2–2 ({reason})" in delegate_activity.progress_text(records[-1])
    assert all(delegated_activity_meta(record, task_id="child") == record for record in records)


def test_a_window_that_moved_the_cursor_leaves_no_earlier_draft_proven(tmp_path):
    from ouroboros.gateways import claudexor_run_events

    daemon = Daemon([[_said(1, "the", K1, "fragment")], [_said(2, " big", K1, "fragment"), _said(3, " cat", K1, "fragment")],
                     [_said(4, "the big cat", K1, "complete"), _tool(5)]])
    gateway, ctx = _gateway(daemon), _ctx(tmp_path, [])
    _read(ctx, daemon, gateway, 1, 1)
    # Another engine build without the stream serves the next advance as bounded window rows only.
    daemon.stream, daemon.revealed = False, 2
    claudexor_run_events._SUPPORT.clear()
    rows = [{"type": "harness.event", "title": text, "textKind": "message", "textDelta": True, "speechKey": K1,
             "speechOp": "fragment", "harnessId": "claude", "attemptId": "a01"} for text in (" big", " cat")]
    window, = _records(ctx, gateway, 3, after=1, rows=rows)
    assert (window["source"]["kind"], window["through_seq"]) == ("timeline_window", 3)
    daemon.stream = True
    claudexor_run_events._SUPPORT.clear()
    record, = _read(ctx, daemon, gateway, 3, 5, after=3)
    part, = record["parts"]
    assert (record["after_seq"], part["text"]) == (3, "the big cat")
    assert not {"updated", "carried"} & part.keys(), "the window's words never passed the digest: unknown"


K4 = "claude:ses-1:4:0"
# Two blocks of one actor; the second loses " block." (seq 5) on the way, or the window kept no row for it.
TWO_KEYS = [*_stream(1, ["Checking", " the file."], K1), *_stream(4, ["Next", " block."], K2)]
TWO_KEY_ROWS = [{"type": "harness.event", "title": text, "textKind": "message", "textDelta": op == "fragment",
                 "speechKey": key, "speechOp": op, "harnessId": "claude", "attemptId": "a01"}
                for text, key, op in (("Checking", K1, "fragment"), (" the file.", K1, "fragment"),
                                      ("Checking the file.", K1, "complete"), ("Next", K2, "fragment"),
                                      ("Next block.", K2, "complete"))]
LATER_KEYS = [*_stream(7, ["Last", " one."], K3, complete="Last one, revised."),
              *_stream(10, ["Done", "."], K4, complete="Done, revised.")]


@pytest.mark.parametrize("coverage", ["missing", "unreadable", "window", "provisional"])
def test_no_key_of_a_record_without_exact_coverage_is_proven_either_way(tmp_path, coverage):
    """Every comparison needs the gap-free exact range, not only the record's first key: an earlier block of the
    same actor proves nothing about a later block whose fragment never passed here whole, or that a bounded window
    shows. Its complete text still shows whole, and a later exact read proves its keys again."""
    from ouroboros.gateways import claudexor_run_events

    lossy = coverage in ("missing", "unreadable")
    daemon = (Lossy([TWO_KEYS, LATER_KEYS], **{coverage: [5]}) if lossy
              else Daemon([TWO_KEYS, LATER_KEYS], stream=coverage == "provisional"))
    gateway, ctx = _gateway(daemon), _ctx(tmp_path, [])
    daemon.revealed = 1
    if coverage == "provisional":                        # the exact read fails: the window stands in, the cursor stays
        gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(
            lambda request: httpx.Response(503, json={"code": "busy"}) if request.url.path.endswith("/events")
            else daemon.handler(request)))
    first, = _records(ctx, gateway, 6, rows=[] if lossy else TWO_KEY_ROWS)
    assert first["source"].get("provisional", False) is (coverage == "provisional")
    assert [(p["speech"], p["text"]) for p in first["parts"]] == [
        ({"key": K1, "op": "complete"}, "Checking the file."), ({"key": K2, "op": "complete"}, "Next block.")]
    assert not any({"updated", "carried"} & part.keys() for part in first["parts"]), \
        "neither key's draft passed here whole: unknown, never a difference"
    assert _speech_lines(first)[-1] == "claude · a01: Next block.", "its words stay whole for text readers"
    if coverage == "provisional":
        gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(daemon.handler))
        exact, = _records(ctx, gateway, 6)
        assert exact["after_seq"] == 0 and not any("updated" in part for part in exact["parts"]), \
            "the exact re-read of the same range saw both drafts whole: equal, not updated"
    elif coverage == "window":
        daemon.stream = True
        claudexor_run_events._SUPPORT.clear()
    later, = _read(ctx, daemon, gateway, 2, 12, after=6)
    third, fourth = later["parts"]
    assert ("updated" in third) is (coverage == "provisional"), \
        "after a record that moved the cursor without exact coverage, a block's beginning is unknown again"
    assert fourth["updated"] is True, "a later block begun behind one seen whole is proven again"


@pytest.mark.parametrize("space", [" ", "\n\n"], ids=["space", "paragraph"])
@pytest.mark.parametrize("done", [False, True], ids=["unfinished", "completed"])
def test_a_poll_holding_only_the_space_between_words_keeps_it_in_the_utterance(tmp_path, space, done):
    """The engine streams words and the space between them as fragments, and a poll can hold only the space. The
    record carries that fragment, whose seq it read, so the card joins across it with the space; no text reader
    prints an empty line for it, and the run's latest words stay the last visible ones."""
    steps = [[_said(1, "the", K1, "fragment")], [_said(2, space, K1, "fragment")],
             [_said(3, "cat", K1, "fragment"), *([_said(4, f"the{space}cat", K1, "complete")] if done else []),
              _tool(5 if done else 4)]]
    daemon = Daemon(steps)
    gateway, ctx = _gateway(daemon), _ctx(tmp_path, [])
    first, = _read(ctx, daemon, gateway, 1, 1)
    blank, = _read(ctx, daemon, gateway, 2, 2, after=1)
    assert [(p["text"], p["seq"], p["speech"]) for p in blank["parts"]] == [(space, 2, {"key": K1, "op": "fragment"})]
    assert delegated_activity_meta(blank, task_id="child") == blank
    assert _speech_lines(blank) == [] and "claude · a01:" not in delegate_activity.progress_text(blank)
    assert delegate_activity.card_text(blank, 650).splitlines()[0] == "claude · a01 (earlier): the"
    assert delegate_activity._memo("child", "run-1").latest_message["text"] == "the"
    last, = _read(ctx, daemon, gateway, 3, steps[2][-1]["seq"], after=2)
    if done:
        part, = last["parts"]
        assert (part["text"], part.get("updated"), part.get("carried")) == (f"the{space}cat", None, len("the" + space))
        assert _speech_lines(last) == ["claude · a01 (continued, complete): cat"], "each earlier piece was shown"
    else:
        assert "".join(record["parts"][0]["text"] for record in (first, blank, last)) == f"the{space}cat"


LEFT_OUT = {"actor": "claude/a01", "key": K1}


@pytest.mark.parametrize("others,left_out", [(40, True), (39, False)], ids=["part-count", "control"])
def test_a_keyed_piece_the_preview_leaves_out_is_named_so_no_reader_joins_across_it(tmp_path, others, left_out):
    """The reported case: a poll whose 40 other messages fill the preview's part count leaves the draft's ' big ' out.
    The record still claims every seq it read, so it names that piece by key and seq: a reader joining the words on
    either side across it would show 'thecat'. With one message fewer the piece is in the preview and nothing is named."""
    steps = [[_said(1, "the", K1, "fragment")],
             [*(_harness(2 + i, "message", text=f"Note {i}.") for i in range(others)), _said(2 + others, " big ", K1, "fragment")],
             [_said(3 + others, "cat", K1, "fragment")]]
    daemon = Daemon(steps)
    gateway, ctx, records = _gateway(daemon), _ctx(tmp_path, []), []
    for step, (after, through) in enumerate(((0, 1), (1, 2 + others), (2 + others, 3 + others)), start=1):
        records += _read(ctx, daemon, gateway, step, through, after=after)
    middle = records[1]
    assert delegated_activity_meta(middle, task_id="child") == middle
    if left_out:
        assert (middle["omitted"], middle["speech_omitted"]) == ({"message": 1}, [{**LEFT_OUT, "seq": 42, "last_seq": 42}])
        assert " big " not in [part["text"] for part in middle["parts"]]
    else:
        assert not {"omitted", "speech_omitted"} & middle.keys() and middle["parts"][-1]["text"] == " big "


@pytest.mark.parametrize("bound", ["characters", "carrier"])
def test_a_keyed_piece_left_out_by_a_size_bound_is_named_too(tmp_path, bound):
    """The speech budget (four whole previews spend it) and the carrier bound (three previews whose JSON escapes pass it,
    shed from the end) leave a keyed piece out the same way, and name it the same way."""
    unit, count = ("w", 4) if bound == "characters" else ("\x00", 3)
    filler = [_harness(1 + i, "message", text=unit * 6000) for i in range(count)]
    record = _one(tmp_path, [*filler, _said(count + 1, " big ", K1, "fragment"), _tool(count + 2)])
    assert record["speech_omitted"] == [{**LEFT_OUT, "seq": count + 1, "last_seq": count + 1}]
    assert record["omitted"]["message"] == (1 if bound == "characters" else len(filler) + 1 - len(record["parts"]))
    assert len(json.dumps(record, ensure_ascii=False)) <= delegate_activity._RECORD_CHARS
    assert delegated_activity_meta(record, task_id="child") == record


def test_more_left_out_pieces_than_the_list_holds_say_so(tmp_path):
    """Past its bound the list says it is incomplete, so a reader joins nothing across the record."""
    record = _one(tmp_path, [*(_harness(1 + i, "message", text=f"Note {i}.") for i in range(40)),
                             *(_said(41 + i, "x", f"claude:ses-1:9:{i}", "fragment") for i in range(41))])
    assert (len(record["speech_omitted"]), record["speech_omitted_truncated"], record["omitted"]) == (40, True, {"message": 41})


# -- the frame log: each word once ---------------------------------------------------------

@pytest.mark.parametrize("complete,updated", [(PARAGRAPH, False), (PARAGRAPH.replace("GUID", "guid"), True)])
def test_the_frame_log_prints_a_complete_texts_new_words_after_its_proven_draft(tmp_path, complete, updated):
    pieces = _pieces(PARAGRAPH, 6)
    daemon = Daemon([[_tool(1), *_stream(2, pieces[:3], K1, complete=False)],
                     [*_stream(5, pieces[3:], K1, complete=complete), _tool(9, "Edit")]])
    gateway, ctx = _gateway(daemon), _ctx(tmp_path, [])
    head = "".join(pieces[:3])
    draft, = _read(ctx, daemon, gateway, 1, 4)
    done, = _read(ctx, daemon, gateway, 2, 9, after=4)
    assert _speech_lines(draft) == [f"claude · a01: {head}"]
    assert _speech_lines(done) == ([f"claude · a01 (updated): {complete}"] if updated else
                                   [f"claude · a01 (continued, complete): {PARAGRAPH[len(head):]}"]), \
        "an equal complete adds only its new words; a changed one is printed whole and says so"
    assert done["parts"][0]["text"] == complete, "the record keeps the whole complete text for the card and history"
    assert delegate_activity.card_text(done, 650).count(complete) == 1, "a card shows one record: the words whole"


NOTES = [f"Note {index}." for index in range(40)]                   # with the draft, 41 parts: one is omitted
LONG = ["L" * 550] * 4                                                   # their lines alone pass the frame's 2000


@pytest.mark.parametrize("before,draft,rest,shown", [
    pytest.param([], ["Checking the file."], [], True, id="shown-whole"),
    pytest.param([], ["w" * 700], [], False, id="line-bound"),          # its line keeps 600 code points
    pytest.param(NOTES, ["Checking the file."], [], False, id="part-bound"),
    pytest.param(LONG, ["Checking the file."], [], False, id="frame-bound"),
    pytest.param([], ["😀" * 599], ["é ok"], True, id="astral-whole"),   # code points, not UTF-16 units
    pytest.param([], ["😀" * 601], ["é ok"], False, id="astral-cut"),
])
def test_the_frame_log_skips_only_words_its_earlier_lines_showed_whole(tmp_path, before, draft, rest, shown):
    first = [_harness(index + 1, "message", text=text) for index, text in enumerate(before)]
    first += [_said(len(first) + 1 + index, piece, K1, "fragment") for index, piece in enumerate(draft)]
    complete = "".join(draft + rest)
    second = [_said(len(first) + 1 + index, piece, K1, "fragment") for index, piece in enumerate(rest)]
    second += [_said(len(first) + len(second) + 1, complete, K1, "complete"), _tool(len(first) + len(second) + 2)]
    daemon = Daemon([first, second])
    gateway, ctx = _gateway(daemon), _ctx(tmp_path, [])
    streamed, = _read(ctx, daemon, gateway, 1, len(first))
    done, = _read(ctx, daemon, gateway, 2, second[-1]["seq"], after=len(first))
    assert ("".join(draft) in delegate_activity.progress_text(streamed)) is shown, "the control: did that frame show it?"
    part = next(part for part in done["parts"] if part.get("speech"))
    assert part["text"] == complete and "updated" not in part
    line, = _speech_lines(done)
    if not shown:
        assert "carried" not in part and f"claude · a01: {truncate_within_limit(complete, 600)}" in \
            delegate_activity.progress_text(done), "printed again from the start, inside the line's disclosed bound"
    else:
        assert line == (f"claude · a01 (continued, complete): {''.join(rest)}" if rest
                        else "claude · a01 (complete as streamed above)"), "each word once, none skipped unseen"


def test_a_complete_text_that_adds_nothing_is_a_completion_fact_in_the_frame_log(tmp_path):
    daemon = Daemon([_stream(1, ["Checking", " the file."], K1, complete=False),
                     [_said(3, "Checking the file.", K1, "complete"), _tool(4)]])
    gateway, ctx = _gateway(daemon), _ctx(tmp_path, [])
    _read(ctx, daemon, gateway, 1, 2)
    done, = _read(ctx, daemon, gateway, 2, 4, after=2)
    assert done["parts"][0]["carried"] == len("Checking the file.")
    assert _speech_lines(done) == ["claude · a01 (complete as streamed above)"]
    assert delegate_activity.card_text(done, 650).splitlines()[0] == "claude · a01: Checking the file."


def test_a_later_records_recap_keeps_the_abandon_and_the_answer_it_recalls(tmp_path):
    daemon = Daemon([_stream(1, ["Checking", " the"], K1, complete=False), [_abandoned(3, K1), _tool(4)], [_tool(5)],
                     [_final(6, ANSWER)], [_run_event(7, "run.completed")]])
    gateway, ctx = _gateway(daemon), _ctx(tmp_path, [])
    _read(ctx, daemon, gateway, 1, 2)
    late, = _read(ctx, daemon, gateway, 2, 4, after=2)
    quiet, = _read(ctx, daemon, gateway, 3, 5, after=4)
    for record in (late, quiet):
        assert delegate_activity.card_text(record, 650).splitlines()[0] == "claude · a01 (earlier, abandoned): Checking the"
    _read(ctx, daemon, gateway, 4, 6, after=5)
    ended, = _read(ctx, daemon, gateway, 5, 7, after=6)
    assert delegate_activity.card_text(ended, 650).splitlines()[0] == f"claude · a01 (earlier, answer): {ANSWER}"
    assert all(delegated_activity_meta(record, task_id="child") == record for record in (late, quiet, ended))


def _waited(tmp_path, monkeypatch, daemon):
    """The frames ``delegate_wait`` hands to the progress transport while it watches ``daemon``'s run."""
    from ouroboros.tools import delegate
    from tests._delegated_transport_shared import _nanny_ctx

    ctx = _nanny_ctx(tmp_path, task_id="child")
    ctx.task_attempt = 0
    _agent_obj, events = _agent(ctx)
    monkeypatch.setitem(delegate._CUSTODY, daemon.run_id, RunCustody(
        task_id="child", run_id=daemon.run_id, route_id="claude", model="m", access="readonly"))
    monkeypatch.setattr(delegate, "_capture_terminal_patch", lambda *a, **k: None)
    clock = SimpleNamespace(now=0.0)
    with monkeypatch.context() as timing:
        timing.setattr(delegate.time, "monotonic", lambda: clock.now)
        timing.setattr(delegate.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
        result = json.loads(delegate._delegate_wait(ctx, daemon.run_id, wait_sec=60, since_seq=0,
                                                    gateway=_gateway(daemon)))
    assert result["status"] == "terminal", result
    return [row for row in (events.get_nowait() for _ in range(events.qsize())) if row["type"] == "send_message"]


def _delivered(tmp_path, monkeypatch, frames):
    """The same frames through supervisor delivery: live broadcast, bus, stored progress and history."""
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "logs" / "chat.jsonl").touch()
    live, outbound = [], []
    bridge = message_bus.LocalChatBridge()
    bridge._broadcast_fn = live.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "publish_event", lambda _topic, event: outbound.append(event))
    monkeypatch.setattr(events_chat_delivery, "_bound_project_chat_id", lambda *_: 0)
    task = {"id": "child", "_attempt": 0, "delegation_role": "subagent",
            "root_task_id": "root", "parent_task_id": "root", "model": "coordinator-model"}
    delivery = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={"child": {"task": task}},
                               send_with_budget=message_bus.send_with_budget,
                               append_jsonl=lambda *_: pytest.fail("delivery raised"))
    for frame in frames:
        events_chat_delivery._handle_send_message(frame, delivery)
    stored = dialogue_test.log_rows(tmp_path / "logs" / "progress.jsonl")
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "50"})))
    replay = [row for row in json.loads(response.body)["messages"] if row.get("is_progress")]
    return live, outbound, stored, replay


def _retained(drive_root, activities):
    from ouroboros.artifacts import read_actor_source_bytes

    return b"".join(read_actor_source_bytes(drive_root, "child", activity["source"]["ref"]) for activity in activities)


def test_the_wait_delivers_one_utterance_live_to_history_and_the_text_readers(tmp_path, monkeypatch):
    pieces = _pieces(PARAGRAPH, 6)
    steps = [[_tool(1), *_stream(2, pieces[:3], K1, complete=False)],
             [*_stream(5, pieces[3:], K1, complete=PARAGRAPH), _tool(9, "Edit"),
              *_stream(10, ["Checking", " the"], K2, complete=False),
              _abandoned(12, K2)],
             [_final(13, PARAGRAPH), _run_event(14, "run.completed")]]
    frames = _waited(tmp_path, monkeypatch, Daemon(steps))
    activities = [frame["progress_meta"]["delegated_activity"] for frame in frames]
    assert [[(p.get("speech", {}).get("op"), p.get("final"), p["text"]) for p in a["parts"]] for a in activities] == [
        [("fragment", None, "".join(pieces[:3]))],
        [("complete", None, PARAGRAPH), ("fragment", None, "Checking the")],
        [(None, True, PARAGRAPH)]]
    assert activities[1]["speech_abandoned"] == [{"actor": "claude/a01", "key": K2, "seq": 12}]
    assert [frame["text"].count(PARAGRAPH) for frame in frames] == [0, 0, 1], "the log never prints the paragraph twice"
    head = "".join(pieces[:3])
    assert f"claude · a01: {head}" in frames[0]["text"]
    assert f"claude · a01 (continued, complete): {PARAGRAPH[len(head):]}" in frames[1]["text"]
    assert "claude · a01 (abandoned): Checking the" in frames[1]["text"]
    assert f"claude · a01 (answer): {PARAGRAPH}" in frames[2]["text"]
    for rows in _delivered(tmp_path, monkeypatch, frames):
        assert [row["delegated_activity"] for row in rows] == activities
    retained = [json.loads(line) for line in _retained(tmp_path, activities).decode().splitlines()]
    assert retained == [event for step in steps for event in step], "every original line, both forms of the words"


# -- the engine's own normalized journals --------------------------------------------------

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "claudexor_speech"
SCENARIOS = ("messages", "multiblock", "replay", "joined")


def engine_journal(name):
    """The engine's served lines for ``name``, verbatim, with its timeline rows and answer,
    checked against the export's digests (``provenance.json``)."""
    provenance = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
    digests = next(row["files"] for row in provenance["scenarios"] if row["scenario"] == name)
    raw = {name_: (FIXTURES / name / name_).read_bytes() for name_ in digests}
    assert {name_: hashlib.sha256(data).hexdigest() for name_, data in raw.items()} == digests, "fixture bytes drifted"
    lines = raw["events.ndjson"].decode("utf-8").removesuffix("\n").split("\n")
    return SimpleNamespace(raw=raw["events.ndjson"], lines=lines, events=[json.loads(line) for line in lines],
                           timeline=json.loads(raw["timeline.json"]), answer=json.loads(raw["answer.json"]))


class Verbatim(Daemon):
    """``Daemon`` serving each journal event as the exact line the engine wrote, revealed in steps."""

    def __init__(self, journal, cuts):
        bounds, steps, first = [*cuts, journal.events[-1]["seq"]], [], 0
        for bound in bounds:
            steps.append([event for event in journal.events if first < event["seq"] <= bound])
            first = bound
        super().__init__(steps, run_id=journal.events[0]["run_id"])
        self.bounds = bounds
        self.line = {event["seq"]: line for event, line in zip(journal.events, journal.lines)}

    def handler(self, request):
        if request.url.path != f"/v2/runs/{self.run_id}/events":
            return super().handler(request)
        after = int(request.headers.get("last-event-id") or 0)
        body = ": connected\n\n" + "".join(f"id: {e['seq']}\nevent: {e['type']}\ndata: {self.line[e['seq']]}\n\n"
                                           for e in self.journal if e["seq"] > after)
        end = "event: end\ndata: {}\n\n" if self.revealed == len(self.steps) else ""
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body + end)


def _observed(ctx, gateway, run_id, advance, after):
    """The records of one wait advance, each committed as its emitter would after handing it over."""
    out = []
    for record, commit in delegate_activity.observations(ctx, gateway, run_id, advance, after_seq=after, read_sec=2.0):
        commit()
        out.append(record)
    return out


def _splits(count, *, most=None):
    """No cut, every single cut, and a cut every third event (each part at most ``most`` long)."""
    cuts = [(), *((k,) for k in range(1, count)), tuple(range(3, count, 3))]
    return [c for c in cuts if most is None or all(b - a <= most for a, b in zip((0, *c), (*c, count)))]


def _speech_facts(journal):
    """Per key, the engine's joined fragments and complete text; abandoned keys; unkeyed words."""
    keys, abandoned, unkeyed = {}, [], []
    for event in journal.events:
        payload = event["payload"]
        speech = payload.get("speech") or {}
        if speech.get("op") == "abandoned":
            abandoned.append(speech["key"])
        elif payload.get("type") == "message" and speech:
            entry = keys.setdefault(speech["key"], {"draft": "", "complete": None})
            if speech["op"] == "fragment":
                entry["draft"] += payload["text"]
            else:
                entry["complete"] = payload["text"]
        elif payload.get("type") == "message" and not payload.get("final"):
            unkeyed.append(payload["text"])
    return keys, abandoned, unkeyed


def _logged(records):
    """Each message part with the words its record's frame log prints for it."""
    out = []
    for record in records:
        lines, messages = _speech_lines(record), [part for part in record["parts"] if part["kind"] == "message"]
        assert len(lines) == len(messages), lines
        out += [(part, line.partition(": ")[2]) for part, line in zip(messages, lines)]
    return out


def _one_utterance_per_key(journal, records, *, exact=True):
    """What every reader of these records may rely on, however the polls cut the journal."""
    keys, abandoned, unkeyed = _speech_facts(journal)
    for key, facts in keys.items() if exact else ():
        words = "".join(text for part, text in _logged(records) if (part.get("speech") or {}).get("key") == key)
        assert words == (facts["draft"] if facts["complete"] is None else facts["complete"]), \
            (key, words, "the frame log prints each word of an utterance it saw whole once")
    parts = [part for record in records for part in record["parts"]]
    for record in records:
        named = [part["speech"]["key"] for part in record["parts"] if part.get("speech")]
        assert len(named) == len(set(named)), "a key is one part in a record"
    for key, facts in keys.items():
        mine = [part for part in parts if (part.get("speech") or {}).get("key") == key]
        drafts = "".join(part["text"] for part in mine if part["speech"]["op"] == "fragment")
        completes = [part["text"] for part in mine if part["speech"]["op"] == "complete"]
        assert completes == ([facts["complete"]] if facts["complete"] is not None else []), key
        assert facts["draft"].startswith(drafts) and (facts["complete"] is not None or drafts == facts["draft"]), key
        assert not any(part.get("updated") for part in mine), "the engine's complete texts equal their drafts"
    marks = [mark["key"] for record in records for mark in record.get("speech_abandoned", [])]
    assert marks == abandoned and not any((part.get("speech") or {}).get("op") == "complete"
                                          for part in parts if (part.get("speech") or {}).get("key") in abandoned)
    assert [part["text"] for part in parts if not part.get("speech") and not part.get("final")
            and part["kind"] == "message"] == unkeyed, "unkeyed words stay as they came, linked to nothing"
    finals = [part["text"] for part in parts if part.get("final")]
    assert finals == [journal.answer["text"]], "the typed final is the engine's answer, once"
    answers = [line for record in records for line in delegate_activity.progress_text(record).splitlines()
               if "(answer)" in line]
    assert answers == [f"claude · a01 (answer): {journal.answer['text']}"], "its words reach text readers whole"


@pytest.mark.parametrize("name", SCENARIOS)
def test_the_engines_journals_read_as_one_utterance_per_key_however_the_polls_cut_them(tmp_path, name):
    journal = engine_journal(name)
    assert journal.answer["hasFinal"] is True
    for cuts in _splits(len(journal.events)):
        delegate_activity.reset_process_memo()
        root = tmp_path / ("cut-" + "-".join(map(str, cuts)))
        daemon = Verbatim(journal, cuts)
        gateway, ctx, records, after = _gateway(daemon), _ctx(root, []), [], 0
        for step, bound in enumerate(daemon.bounds, start=1):
            daemon.revealed = step
            records += _observed(ctx, gateway, daemon.run_id, SimpleNamespace(seq=bound, events=[], events_omitted=0), after)
            after = bound
        assert [(r["after_seq"], r["source"]["read_through"]) for r in records] == list(zip((0, *cuts), daemon.bounds))
        assert all(delegated_activity_meta(record, task_id="child") == record for record in records)
        assert _retained(root, records) == journal.raw, "the engine's lines are retained byte for byte"
        _one_utterance_per_key(journal, records)


@pytest.mark.parametrize("name", SCENARIOS)
def test_the_engines_timeline_rows_fold_the_same_way_in_the_window_view(tmp_path, name):
    journal = engine_journal(name)
    rows = journal.timeline
    keys, _abandoned_keys, _unkeyed = _speech_facts(journal)
    for cuts in _splits(len(rows), most=12):        # a window batch shows its last 12 rows
        delegate_activity.reset_process_memo()
        daemon = Daemon([[_harness(1, "message", text="x")]], stream=False, run_id=journal.events[0]["run_id"])
        daemon.revealed = 1
        gateway, ctx = _gateway(daemon), _ctx(tmp_path / ("cut-" + "-".join(map(str, cuts))), [])
        seen, records = delegate_progress.WindowObservations(), []
        for index, bound in enumerate((*cuts, len(rows))):
            advance = seen.record({"timeline": rows[:bound], "lastSeq": bound}, bound, index)
            assert not advance.events_omitted
            line = delegate_progress.live_line(daemon.run_id, advance)
            for key, facts in keys.items():
                batch = [row for row in advance.events if row.get("speechKey") == key]
                if facts["complete"] not in (None, journal.answer["text"]) and {"fragment", "complete"} <= {
                        row["speechOp"] for row in batch}:
                    assert line.count(facts["complete"]) == 1, (line, key)
            records += _observed(ctx, gateway, daemon.run_id, advance, 0)
        assert all(record["source"]["kind"] == "timeline_window" for record in records)
        _one_utterance_per_key(journal, records, exact=False)


@pytest.mark.parametrize("name", SCENARIOS)
def test_the_engines_journal_through_the_wait_bus_and_history(tmp_path, monkeypatch, name):
    journal = engine_journal(name)
    count = len(journal.events)
    daemon = Verbatim(journal, (count // 3, 2 * count // 3))
    frames = _waited(tmp_path, monkeypatch, daemon)
    activities = [frame["progress_meta"]["delegated_activity"] for frame in frames]
    _one_utterance_per_key(journal, activities)
    assert sum(frame["text"].count(f"(answer): {journal.answer['text']}") for frame in frames) == 1
    for rows in _delivered(tmp_path, monkeypatch, frames):
        assert [row["delegated_activity"] for row in rows] == activities
    assert _retained(tmp_path, activities) == journal.raw
