"""Unicode line separators in a delegated run's served lines are text, never framing.

JSON leaves U+0085, U+2028 and U+2029 unescaped (``JSON.stringify`` and
``json.dumps(ensure_ascii=False)`` write them as they are), while ``str.splitlines``
and HTTPX ``iter_lines`` end a line at each. Only CR, LF and CRLF end an SSE line
(``gateways.claudexor_run_events``) and only LF ends a retained JSONL record
(``delegate_activity._persist``). These tests serve such lines through the real
gateway, whole and in one-byte reads, and replay what was retained through the real
artifact store, ``delegate_progress.emit_retained`` and the wait's terminal delivery.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import delegate_activity, delegate_progress
from ouroboros.delegate_custody import RunCustody
from ouroboros.gateways import claudexor as gateway_module
from ouroboros.gateways import claudexor_run_events as run_events
from ouroboros.gateways.claudexor_run_events import RunEvent, read_run_events
from ouroboros.subagent_messages import delegated_activity_meta
from tests.test_delegated_activity import Daemon, _agent, _ctx, _gateway, _harness, _records, _run_event

SEPARATORS = ("\u0085", "\u2028", "\u2029")


def _forged(sep):
    """Words that would read as fields, an end frame and a blank line if ``sep`` framed them."""
    return f"before{sep}event: end{sep}{sep}id: 999{sep}data: {{\"seq\": 999}}{sep}: note{sep}after 🙂"


def _line(event, *, escaped=False):
    return json.dumps(event, ensure_ascii=escaped)


def _wire(events, *, eol="\n", end=True, escaped=()):
    body = ": connected" + eol + eol
    for event in events:
        body += eol.join((f"id: {event['seq']}", f"event: {event['type']}",
                          f"data: {_line(event, escaped=event['seq'] in escaped)}", "", ""))
    return body + (eol.join(("event: end", "data: {}", "", "")) if end else "")


class _Chunks(httpx.SyncByteStream):
    """The body in ``size``-byte reads (a character or a CRLF split between two), then ``fail``."""

    def __init__(self, body, size=None, fail=None):
        self.body, self.size, self.fail, self.closed = body.encode("utf-8"), size, fail, False

    def __iter__(self):
        size = self.size or max(1, len(self.body))
        for start in range(0, len(self.body), size):
            yield self.body[start:start + size]
        if self.fail is not None:
            raise self.fail

    def close(self):
        self.closed = True


def _served(stream):
    gateway = _gateway(Daemon([[]]))
    gateway._client.close()
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(
        lambda _request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)))
    return gateway


def _read(body, *, size=None, fail=None, **bounds):
    bounds = {"through_seq": None, "timeout_sec": 2, **bounds}
    return read_run_events(_served(_Chunks(body, size, fail)), "run-1", after_seq=0, **bounds)


class _LiteralDaemon(Daemon):
    """The same journal, each line served unescaped in one-byte reads."""

    def handler(self, request):
        if not request.url.path.endswith("/events"):
            return super().handler(request)
        self.requests.append((request.method, request.url.path, request.headers.get("last-event-id")))
        after = int(request.headers.get("last-event-id") or 0)
        events = [event for event in self.journal if event["seq"] > after]
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=_Chunks(_wire(events, end=self.revealed == len(self.steps)), 1))


# -- the stream reader ----------------------------------------------------------------------

def test_a_served_line_keeps_its_separators_in_every_line_ending_and_chunking():
    events = [_harness(seq, "message", text=_forged(sep)) for seq, sep in enumerate(SEPARATORS, 1)]
    events += [_harness(4, "tool_call", tool={"name": "Read", "target": "a\u2028b.py"}),
               _harness(5, "message", text=_forged("\u2028"))]
    reads = {(eol, size): _read(_wire(events, eol=eol, escaped={5}), size=size)
             for eol in ("\n", "\r\n", "\r") for size in (None, 1, 7)}
    first = reads["\n", None]
    assert [event.data for event in first.events] == [_line(e, escaped=e["seq"] == 5) for e in events], \
        "each served line is kept verbatim, literal and escaped separators alike"
    assert [event.value for event in first.events] == events
    assert (first.through_seq, first.ended, first.stop) == (5, True, "end"), \
        "only the real end frame ends the read; the forged one is text"
    for key, read in reads.items():
        assert asdict(read) == asdict(first), f"line ending and chunk boundaries change nothing: {key}"


def test_a_malformed_line_stays_one_unreadable_event_the_cursor_passes():
    broken = '{"seq": 1, "text": "a\x0bb\x0cc\x1cd\x1de\x1ef\u2028'  # raw controls, unterminated
    good = _harness(2, "message", text="fine")
    body = f"id: 1\nevent: harness.event\ndata: {broken}\n\nid: 2\nevent: harness.event\ndata: {_line(good)}\n\n"
    read = _read(body, size=3)
    assert [(event.seq, event.data, event.value) for event in read.events] == [(1, broken, None), (2, _line(good), good)]
    assert read.through_seq == 2 and not read.ended
    assert delegate_activity._event_item(1, None)["label"] == "unreadable event line"


@pytest.mark.parametrize("text", ["plain words 🙂", _forged("\u2028")])
@pytest.mark.parametrize("size", [None, 1])
@pytest.mark.parametrize("tail, closed", [("", False), ("\n", False), ("\r", False), ("\r\n", False),
                                          ("\n\n", True), ("\n\r", True), ("\r\r", True), ("\r\n\r\n", True)])
def test_at_eof_only_a_frame_closed_by_its_blank_line_counts(text, size, tail, closed):
    first, last = _harness(1, "message", text=text), _harness(2, "message", text=text)
    body = _wire([first], end=False) + f"id: 2\nevent: harness.event\ndata: {_line(last)}" + tail
    read = _read(body, size=size)
    assert [event.value for event in read.events] == ([first, last] if closed else [first])
    assert (read.through_seq, read.ended, read.stop) == (2 if closed else 1, False, "time")


def test_the_byte_bound_keeps_only_whole_frames_of_separated_text():
    events = [_harness(seq, "message", text="x\u2028" * 200) for seq in range(1, 9)]
    read = _read(_wire(events), size=64, through_seq=8, max_bytes=1500)
    assert read.stop == "bytes" and 0 < len(read.events) < 8
    assert read.through_seq == read.events[-1].seq and [e.value for e in read.events] == events[:len(read.events)]


def test_the_deadline_between_lines_keeps_only_whole_frames(monkeypatch):
    ticks = iter(range(100_000))
    monkeypatch.setattr(run_events, "time", SimpleNamespace(monotonic=lambda: next(ticks) / 10))
    events = [_harness(seq, "message", text=_forged("\u2029")) for seq in range(1, 6)]
    read = _read(_wire(events), size=5, timeout_sec=1.05)
    assert read.stop == "time" and not read.ended and 0 < len(read.events) < len(events)
    assert read.through_seq == read.events[-1].seq and [e.value for e in read.events] == events[:len(read.events)]


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("quiet"), httpx.RemoteProtocolError("peer closed")])
def test_a_stream_that_breaks_mid_frame_keeps_the_whole_frames_it_read(failure):
    first, cut = _harness(1, "message", text=_forged("\u0085")), _harness(2, "message", text=_forged("\u2028"))
    stream = _Chunks(_wire([first], end=False) + f"id: 2\nevent: harness.event\ndata: {_line(cut)[:40]}", 5, failure)
    read = read_run_events(_served(stream), "run-1", after_seq=0, through_seq=5, timeout_sec=2)
    assert [event.value for event in read.events] == [first] and (read.through_seq, read.stop) == (1, "time")
    assert stream.closed


@pytest.mark.parametrize("text", ["plain words 🙂", _forged("\u2028")])
def test_cancellation_inside_the_stream_propagates_and_closes_it(text):
    stream = _Chunks(_wire([_harness(1, "message", text=text)], end=False), 4, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        read_run_events(_served(stream), "run-1", after_seq=0, through_seq=5, timeout_sec=2)
    assert stream.closed


def test_a_transport_failure_before_the_stream_opens_stays_typed():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    gateway = _gateway(Daemon([[]]))
    gateway._client.close()
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(refuse))
    with pytest.raises(gateway_module.ClaudexorUnavailable) as caught:
        read_run_events(gateway, "run-1", after_seq=0, through_seq=1, timeout_sec=1)
    assert caught.value.code == "daemon_unreachable"


# -- consumers: the live frame, retained replay, the wait's terminal delivery ----------------

def test_the_live_frame_and_its_retained_source_keep_the_executors_exact_words(tmp_path):
    words = [_forged(sep) for sep in SEPARATORS]
    events = [_harness(seq, "message", text=text) for seq, text in enumerate(words, 1)]
    daemon = _LiteralDaemon([events])
    daemon.revealed = 1
    ctx = _ctx(tmp_path, [])
    _agent_obj, frames = _agent(ctx)
    delegate_progress.emit(ctx, "run-1", SimpleNamespace(seq=3, events=[], events_omitted=0),
                           entry=SimpleNamespace(task_id="child", run_id="run-1"), gateway=_gateway(daemon), after_seq=0)
    activity = frames.get_nowait()["progress_meta"]["delegated_activity"]
    assert delegated_activity_meta(activity, task_id="child") == activity
    assert [part["text"] for part in activity["parts"]] == words and "technical" not in activity
    source = tmp_path / "task_results" / "artifacts" / "child" / activity["source"]["ref"]["path"]
    assert source.read_bytes() == "".join(_line(event) + "\n" for event in events).encode("utf-8"), \
        "the retained original is the served lines, byte for byte"
    assert delegate_activity._memo("child", "run-1").shown_through == 3


def _retain(ctx, events):
    """Retain served lines through the production writer, as an unemitted live read leaves them."""
    assert delegate_activity._persist(ctx, "child", "run-1",
                                      [RunEvent(e["seq"], e["type"], _line(e), e) for e in events])


def test_a_new_process_replays_retained_separated_ranges_in_order_and_once(tmp_path):
    emitted = []
    ctx = _ctx(tmp_path, emitted)
    first = [_harness(seq, "message", text=_forged(sep)) for seq, sep in enumerate(SEPARATORS, 1)]
    _retain(ctx, first)
    _retain(ctx, [_harness(4, "tool_call", tool={"name": "Read"}), _harness(5, "message", text="plain words")])
    delegate_progress.emit_retained(ctx, "run-1")  # a fresh memo: only the retained sources remain
    activities = [meta["delegated_activity"] for _text, meta in emitted]
    assert [(a["after_seq"], a["through_seq"]) for a in activities] == [(0, 3), (3, 5)], \
        "one record cut at a separator would stop the whole replay, the later range included"
    assert [[part["text"] for part in a["parts"]] for a in activities] == [
        [event["payload"]["text"] for event in first], ["plain words"]]
    assert delegate_activity._memo("child", "run-1").shown_through == 5
    delegate_progress.emit_retained(ctx, "run-1")
    assert len(emitted) == 2, "a committed range is not replayed again"
    daemon = _LiteralDaemon([[_harness(6, "message", text="next\u2028line")]])
    daemon.revealed = 1
    record, = _records(ctx, _gateway(daemon), 6, after=5)
    assert (record["after_seq"], [part["text"] for part in record["parts"]]) == (5, ["next\u2028line"])


def test_a_blank_retained_record_is_malformed_never_skipped(tmp_path):
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.delegate_custody import custody_root

    emitted = []
    ctx = _ctx(tmp_path, emitted)
    one, two = _harness(1, "message", text="one"), _harness(2, "message", text="two")
    store_actor_source_bytes(custody_root(ctx), "child", category=delegate_activity.SOURCE_CATEGORY,
                             source_id=delegate_activity._source_id("run-1", 1, 2),
                             data=f"{_line(one)}\n\n{_line(two)}\n".encode("utf-8"), extension="jsonl")
    delegate_progress.emit_retained(ctx, "run-1")
    assert emitted == [] and delegate_activity._memo("child", "run-1").shown_through is None, \
        "an empty record between records is malformed source, neither dropped nor acknowledged"


def test_a_later_wait_replays_retained_separated_words_through_terminal_delivery(tmp_path, monkeypatch):
    from ouroboros.tools import delegate
    from tests._delegated_transport_shared import _nanny_ctx

    words = [_forged("\u2028"), _forged("\u0085")]
    daemon = _LiteralDaemon([[_harness(1, "message", text=words[0])],
                             [_harness(2, "message", text=words[1]), _run_event(3, "run.completed")]])
    gateway = _gateway(daemon)
    ctx = _nanny_ctx(tmp_path, task_id="child")
    ctx.task_attempt = 0
    _agent_obj, frames = _agent(ctx)
    delivered, channel = ctx.emit_progress_fn, {"open": False}

    def emit(text, **meta):
        if not channel["open"] and "delegated_activity" in meta:
            raise RuntimeError("progress channel closed")
        return delivered(text, **meta)

    ctx.emit_progress_fn = emit
    monkeypatch.setitem(delegate._CUSTODY, "run-1", RunCustody(task_id="child", run_id="run-1", route_id="claude",
                                                                model="m", access="readonly"))
    monkeypatch.setattr(delegate, "_capture_terminal_patch", lambda *a, **k: None)
    clock = SimpleNamespace(now=0.0)
    with monkeypatch.context() as timing:
        timing.setattr(delegate.time, "monotonic", lambda: clock.now)
        timing.setattr(delegate.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
        assert json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=60, since_seq=0,
                                                  gateway=gateway))["status"] == "terminal"
        assert frames.empty()
        channel["open"] = True
        delegate_activity.reset_process_memo()
        requests = list(daemon.requests)
        assert json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=60, gateway=gateway))["status"] == "terminal"
        assert daemon.requests == requests, "the replay reads only the retained source"
    activity, = [frame["progress_meta"]["delegated_activity"]
                 for frame in (frames.get_nowait() for _ in range(frames.qsize())) if frame["type"] == "send_message"]
    assert (activity["after_seq"], activity["through_seq"]) == (0, 3)
    assert [part["text"] for part in activity["parts"]] == words
