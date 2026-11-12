"""A delegated executor's words and technical events stay typed from the run journal to Chat (#1350).

The fake daemon below serves the real ``ClaudexorGateway`` through ``httpx.MockTransport``:
run detail with its ``lastSeq`` fence, the route catalog, and ``GET /v2/runs/:id/events`` as
SSE with durable ``seq`` ids resumed after ``Last-Event-ID`` — the contract of Claudexor
control-api ``run-events-stream.ts`` at 80ef3839 (the engine redacts each line before
serving it). Nothing here reads the frame's text to decide what is speech.
"""

from __future__ import annotations

from tests import _dialogue as dialogue_test

import asyncio
import contextlib
import json
import queue
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import delegate_activity, delegate_progress
from ouroboros.agent import OuroborosAgent
from ouroboros.delegate_custody import RunCustody
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.gateways import claudexor as gateway_module
from ouroboros.gateways.claudexor_run_events import read_run_events, run_events_supported
from ouroboros.subagent_messages import delegated_activity_meta, subagent_message_meta
from supervisor import events_chat_delivery, message_bus

_STREAM_ROUTE = {"method": "GET", "path": "/v2/runs/:id/events"}


def _harness(seq, kind, *, text=None, actor=("claude", "a01"), delta=False, tool=None, error=None, ts="2026-09-27T12:00:00Z"):
    payload = {"harness_id": actor[0], "attempt_id": actor[1], "type": kind, "ts": ts}
    if text is not None:
        payload["text"] = text
    if delta:
        payload["payload"] = {"delta": True}
    if tool is not None:
        payload["tool"] = tool
    if error is not None:
        payload["error"] = error
    return {"seq": seq, "ts": ts, "run_id": "run-1", "task_id": "t", "type": "harness.event", "payload": payload}


def _run_event(seq, kind, **payload):
    return {"seq": seq, "ts": "2026-09-27T12:00:00Z", "run_id": "run-1", "task_id": "t", "type": kind, "payload": payload}


def _sse(events, after, *, end=False):
    body = ": connected\n\n"
    for event in events:
        if event["seq"] > after:
            body += f"id: {event['seq']}\nevent: {event['type']}\ndata: {json.dumps(event)}\n\n"
    return body + ("event: end\ndata: {}\n\n" if end else "")


class Daemon:
    """One run's journal, revealed in steps; ``lastSeq`` is the served fence."""

    def __init__(self, steps, *, stream=True, state_at_end="succeeded", run_id="run-1"):
        self.steps, self.stream, self.state_at_end, self.run_id = steps, stream, state_at_end, run_id
        self.revealed, self.requests = 0, []

    @property
    def journal(self):
        return [event for step in self.steps[:self.revealed] for event in step]

    def handler(self, request):
        path = request.url.path
        self.requests.append((request.method, path, request.headers.get("last-event-id")))
        if path == "/v2/handshake":
            return httpx.Response(200, json={"compatible": True, "protocolMajor": 3,
                                             "engine": {"version": "3.16.0", "sha": "80ef3839"}})
        if path == "/v2/operations":
            return httpx.Response(200, json={"operations": [_STREAM_ROUTE] if self.stream else []})
        if path == f"/v2/runs/{self.run_id}":
            self.revealed = min(len(self.steps), self.revealed + 1)
            journal = self.journal
            done = self.revealed == len(self.steps)
            return httpx.Response(200, json={
                "lastSeq": journal[-1]["seq"] if journal else 0,
                "summary": {"runId": self.run_id, "state": self.state_at_end if done else "running",
                            "effectiveAccess": "readonly"},
                "timeline": [{"type": e["type"], "title": str(e["payload"].get("text") or e["payload"].get("type") or e["type"]),
                              "severity": "info", "harnessId": "claude", "attemptId": "a01"} for e in journal],
                "primaryOutput": {"kind": "answer", "text": "done", "truncated": False},
            })
        if path == f"/v2/runs/{self.run_id}/events":
            after = int(request.headers.get("last-event-id") or 0)
            done = self.revealed == len(self.steps)
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text=_sse(self.journal, after, end=done))
        return httpx.Response(404, json={"error": "not found"})


def _gateway(daemon):
    gateway = gateway_module.ClaudexorGateway(gateway_module.DaemonEndpoint("127.0.0.1", 1, "fixture"))
    gateway._client.close()
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(daemon.handler))
    gateway.handshake()
    return gateway


def _refusal(code, retryable):
    """The engine's own refusal of a run's event stream, as ``run-events-stream.ts`` writes it:
    no ``id``, and the response ends after it."""
    return "event: error\ndata: " + json.dumps({
        "code": code, "message": "Run event stream could not be read.", "retryable": retryable, "fieldErrors": [],
        "requiredActions": ["reconnect"] if retryable else [], "evidenceRefs": [], "context": {}},
        separators=(",", ":")) + "\n\n"


class LoopbackEngine:
    """The engine's routes on a real loopback socket. The event stream is chunked SSE written one
    frame at a time after ``Last-Event-ID``, then ``tail`` (``event: end``, a ``_refusal`` or nothing)
    and the response's last chunk; ``POST /v2/runs/wait-many`` answers the contract probe, else
    ``attention``; ``GET /v2/runs/<run_id>`` serves ``detail``."""

    def __init__(self, journal, *, tail="", run_id="run-1", detail=None, attention=None):
        self.journal, self.tail, self.run_id, self.detail, self.attention = journal, tail, run_id, detail, attention
        self.requests = []

    def stream_reads(self):
        """The ``Last-Event-ID`` of every event stream request, in order."""
        return [cursor for _method, path, cursor in self.requests if path.endswith("/events")]

    @contextlib.contextmanager
    def serving(self):
        from ouroboros.gateways.claudexor_observation import PROBE_RUN_ID

        engine = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def _head(self, content_type, **headers):
                self.send_response(200)
                for name, value in {"Content-Type": content_type, "Connection": "close", **headers}.items():
                    self.send_header(name, value)
                self.end_headers()

            def _json(self, body):
                data = json.dumps(body).encode()
                self._head("application/json", **{"Content-Length": str(len(data))})
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                engine.requests.append(("POST", self.path, body))
                if self.path == "/v2/handshake":
                    return self._json({"compatible": True, "protocolMajor": 3,
                                       "engine": {"version": "5.1.0", "sha": "loopback"}})
                if body.get("targets") == [{"runId": PROBE_RUN_ID, "seen": None}]:
                    return self._json({"wakeReason": "timeout", "attention": [], "summaries": [
                        {"status": None, "pending": None, "token": None, "problem": "not_found"}]})
                return self._json(engine.attention)

            def do_GET(self):
                engine.requests.append(("GET", self.path, self.headers.get("Last-Event-ID")))
                if self.path == "/v2/operations":
                    return self._json({"operations": [_STREAM_ROUTE, {"method": "POST", "path": "/v2/runs/wait-many"}]})
                if self.path == f"/v2/runs/{engine.run_id}":
                    return self._json(engine.detail)
                after = int(self.headers.get("Last-Event-ID") or 0)
                self._head("text/event-stream", **{"Transfer-Encoding": "chunked"})
                for frame in [": connected\n\n", *(f"id: {e['seq']}\nevent: {e['type']}\ndata: {json.dumps(e)}\n\n"
                                                   for e in engine.journal if e["seq"] > after), engine.tail]:
                    if frame:
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(frame.encode()), frame.encode()))
                        self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        gateway = gateway_module.ClaudexorGateway(gateway_module.DaemonEndpoint("127.0.0.1", server.server_port, "fixture"))
        try:
            gateway.handshake()
            yield gateway
        finally:
            gateway.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def _ctx(tmp_path, emitted):
    return SimpleNamespace(task_id="child", task_attempt=0, drive_root=tmp_path, task_metadata={},
                           emit_progress_fn=lambda text, **meta: emitted.append((text, meta)))


def _records(ctx, gateway, through, *, after=0, rows=(), omitted=0, emit=None, run_id="run-1"):
    """One body journal read or timeline window, committed after the fixture's channel accepts it."""
    from ouroboros.gateways import claudexor_run_events as events

    advance = SimpleNamespace(seq=through, events=list(rows), events_omitted=omitted)
    after = delegate_activity.shown_cursor(ctx, ctx.task_id, run_id, after)
    if through <= after:
        frames = delegate_activity.retained_observations(ctx, run_id)
    elif events.run_events_supported(gateway, timeout_sec=2):
        read = events.read_run_events(gateway, run_id, after_seq=after, through_seq=None, timeout_sec=2)
        frames = delegate_activity.stream_observations(ctx, run_id, read, after=after)
    else:
        frames = delegate_activity.window_observations(ctx, run_id, advance, after_seq=after)
    out = []
    for record, commit in frames:
        if emit is not None:
            emit(record)
        commit()
        out.append(record)
    return out


def _observe(ctx, gateway, through, **kwargs):
    records = _records(ctx, gateway, through, **kwargs)
    assert len(records) <= 1, records
    return records[0] if records else None


# -- the bounded reader -------------------------------------------------------------------

def test_reader_stops_at_the_fence_resumes_exactly_and_ends_with_the_stream(tmp_path):
    events = [_harness(seq, "tool_call", tool={"name": "Read"}) for seq in range(1, 6)]
    daemon = Daemon([events])
    gateway = _gateway(daemon)
    daemon.revealed = 1
    first = read_run_events(gateway, "run-1", after_seq=0, through_seq=3, timeout_sec=2)
    assert [e.seq for e in first.events] == [1, 2, 3] and first.stop == "fence" and first.through_seq == 3
    rest = read_run_events(gateway, "run-1", after_seq=first.through_seq, through_seq=None, timeout_sec=2)
    assert [e.seq for e in rest.events] == [4, 5] and rest.ended and rest.stop == "end"
    assert [row[2] for row in daemon.requests if row[1].endswith("/events")] == ["0", "3"]
    assert json.loads(rest.events[0].data)["seq"] == 4, "the served redacted line is kept verbatim"


def test_reader_counts_only_whole_frames_under_a_byte_bound_and_refuses_typed(tmp_path):
    events = [_harness(seq, "message", text="x" * 400) for seq in range(1, 9)]
    daemon = Daemon([events])
    gateway = _gateway(daemon)
    daemon.revealed = 1
    bounded = read_run_events(gateway, "run-1", after_seq=0, through_seq=8, timeout_sec=2, max_bytes=1500)
    assert bounded.stop == "bytes" and 0 < len(bounded.events) < 8
    assert bounded.through_seq == bounded.events[-1].seq, "the cursor names the last whole event read"
    refused = _gateway(Daemon([events]))
    refused._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(
        lambda _request: httpx.Response(410, json={"code": "run_expired", "message": "reclaimed"})))
    with pytest.raises(gateway_module.ClaudexorUnavailable) as caught:
        read_run_events(refused, "run-1", after_seq=0, through_seq=1, timeout_sec=1)
    assert caught.value.code == "run_expired" and caught.value.status_code == 410


def test_an_open_stream_that_goes_quiet_returns_what_it_read():
    class Quiet(httpx.SyncByteStream):
        def __iter__(self):
            yield _sse([_harness(1, "message", text="hello")], 0).encode()
            raise httpx.ReadTimeout("no more events yet")

    gateway = _gateway(Daemon([[]]))
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(
        lambda _request: httpx.Response(200, stream=Quiet())))
    read = read_run_events(gateway, "run-1", after_seq=0, through_seq=5, timeout_sec=1)
    assert [e.seq for e in read.events] == [1] and read.through_seq == 1 and read.stop == "time"


@pytest.mark.parametrize(("tail", "stop", "problem"), [
    (_refusal("run_expired_by_retention", False), "error",
     {"code": "run_expired_by_retention", "retryable": False, "required_actions": []}),
    (_refusal("bulk_executor_failed", True), "error",
     {"code": "bulk_executor_failed", "retryable": True, "required_actions": ["reconnect"]}),
    ("event: error\ndata: worker gone\n\n", "error", {"code": "unknown", "retryable": True, "required_actions": []}),
    ('id: 9\nevent: error\ndata: {"retryable": false}\n\n', "error",
     {"code": "unknown", "retryable": False, "required_actions": []}),
    ("event: end\ndata: {}\n\n", "end", None),
    ("", "time", None),
    (_refusal("store_" + "x" * 120, True), "error",
     {"code": "store_" + "x" * 120, "retryable": True, "required_actions": ["reconnect"]}),
], ids=["retention-for-good", "worker-reconnect", "unreadable-is-retryable", "numbered-is-no-event", "end", "eof",
        "long-code-kept-whole"])
def test_the_engines_refusal_ends_the_read_typed_after_every_whole_frame(tail, stop, problem):
    engine = LoopbackEngine([_harness(1, "message", text="one"), _harness(2, "message", text="two")], tail=tail)
    with engine.serving() as gateway:
        read = read_run_events(gateway, "run-1", after_seq=0, through_seq=None, timeout_sec=5)
    assert [e.seq for e in read.events] == [1, 2] and read.through_seq == 2, "the frames before it stand, the cursor too"
    assert (read.stop, read.ended, read.problem) == (stop, stop == "end", problem)


def test_route_presence_comes_from_the_engine_catalog_once_per_engine_identity():
    daemon = Daemon([[]])
    gateway = _gateway(daemon)
    assert run_events_supported(gateway, timeout_sec=1) and run_events_supported(gateway, timeout_sec=1)
    assert sum(1 for row in daemon.requests if row[1] == "/v2/operations") == 1
    assert not run_events_supported(SimpleNamespace(engine_version="", open_run_events=None), timeout_sec=1)
    assert not run_events_supported(SimpleNamespace(engine_version="3.9.0"), timeout_sec=1), \
        "a transport without the stream seam keeps the window view without a request"


# -- projection ---------------------------------------------------------------------------

def test_mixed_burst_keeps_attributed_words_problems_and_counts_apart(tmp_path):
    burst = [
        _harness(1, "tool_call", tool={"name": "Read", "target": "chat.js"}),
        _harness(2, "tool_result", tool={"name": "Read", "status": "ok"}),
        _harness(3, "message", text="Checking the Telegram consumer next."),
        _harness(4, "thinking", text="maybe the key lives in chat.js"),
        _harness(5, "tool_call", tool={"name": "Bash", "target": "pytest -q"}),
        _harness(6, "tool_result", tool={"name": "Bash", "status": "error", "error_summary": "exit 1: 2 failed"}),
        _harness(7, "message", text="Two tests fail; reading them.", actor=("codex", "a02")),
        _run_event(8, "run.failed", error="budget exhausted"),
    ]
    daemon = Daemon([burst])
    daemon.revealed = 1
    emitted = []
    record = _observe(_ctx(tmp_path, emitted), _gateway(daemon), 8)
    kinds = [(p["kind"], p.get("actor"), p["seq"]) for p in record["parts"]]
    assert kinds == [("message", "claude/a01", 3), ("thinking", "claude/a01", 4), ("problem", "claude/a01", 6),
                     ("message", "codex/a02", 7), ("problem", "", 8)]
    assert record["parts"][2]["text"] == "exit 1: 2 failed" and record["parts"][2]["label"] == "Bash"
    assert record["technical"]["count"] == 3
    assert record["technical"]["labels"] == [["Read", 1], ["tool results", 1], ["Bash", 1]]
    assert record["technical"]["seqs"] == [[1, 2], [5, 5]], "every technical event keeps its own seq identity"
    assert (record["after_seq"], record["through_seq"], record["source"]["events"]) == (0, 8, 8)
    text = delegate_activity.progress_text(record)
    assert text.splitlines()[1] == "claude · a01: Checking the Telegram consumer next."
    assert text.index("codex · a02: Two tests fail") < text.index("⚠ claude · a01 Bash: exit 1")
    assert "maybe the key" not in text and "thinking ×1" in text
    assert delegated_activity_meta(record, task_id="child") == record


def test_only_the_delta_fact_joins_text_and_complete_texts_stay_separate(tmp_path):
    events = [
        _harness(1, "thinking", text="UX and interaction", delta=True, actor=("cursor", "a01")),
        _harness(2, "thinking", text=" ", delta=True, actor=("cursor", "a01")),
        _harness(3, "thinking", text="plan", delta=True, actor=("cursor", "a01")),
        _harness(4, "message", text="Check", delta=True),
        _harness(5, "message", text="ing now", delta=True),
        _harness(6, "message", text="Checking now."),
        _harness(7, "message", text="Checking now."),
        _harness(8, "message", text="other attempt", delta=True, actor=("claude", "a02")),
        _harness(9, "tool_call", tool={"name": "Read"}),
        _harness(10, "message", text=" continues", delta=True, actor=("claude", "a02")),
    ]
    daemon = Daemon([events])
    daemon.revealed = 1
    parts = _observe(_ctx(tmp_path, []), _gateway(daemon), 10)["parts"]
    assert [(p["kind"], p["text"], p["seq"], p["last_seq"]) for p in parts] == [
        ("thinking", "UX and interaction plan", 1, 3),
        ("message", "Checking now", 4, 5),
        ("message", "Checking now.", 6, 6),
        ("message", "Checking now.", 7, 7),
        ("message", "other attempt", 8, 8),
        ("message", " continues", 10, 10),
    ], "equal words at distinct seqs stay distinct utterances; only an adjacent delta joins a stream"
    assert parts[1]["fragments"] == 2 and parts[1]["delta"] is True
    assert parts[0]["cuts"] == [[2, 18], [3, 19]] and parts[1]["cuts"] == [[5, 5]], \
        "each joined fragment keeps its seq and its offset in the text"
    assert "delta" not in parts[2] and "cuts" not in parts[2]


def test_long_text_is_a_bounded_preview_of_a_retained_verifiable_original(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes

    long_text = "word " * 3000
    events = [_harness(1, "message", text=long_text), _harness(2, "tool_call", tool={"name": "Read"})]
    daemon = Daemon([events])
    daemon.revealed = 1
    record = _observe(_ctx(tmp_path, []), _gateway(daemon), 2)
    part = record["parts"][0]
    assert part["truncated"] and part["chars"] == len(long_text) and "OMISSION NOTE" in part["text"]
    ref = record["source"]["ref"]
    assert ref["path"].startswith("source_handles/delegated_activity/run-1-1-2-")
    original = read_actor_source_bytes(tmp_path, "child", ref).decode().splitlines()
    assert [json.loads(line)["payload"].get("text") for line in original] == [long_text, None]


# -- cursor, gaps, window, commit --------------------------------------------------------

def test_the_same_event_reaches_the_human_once_and_an_unread_remainder_is_caught_up(tmp_path, monkeypatch):
    import ouroboros.gateways.claudexor_run_events as run_events

    step1 = [_harness(1, "message", text="first")]
    step2 = [_harness(2, "message", text="second"), _harness(3, "message", text="third")]
    daemon = Daemon([step1, step2])
    gateway = _gateway(daemon)
    ctx = _ctx(tmp_path, [])
    daemon.revealed = 1
    first = _observe(ctx, gateway, 1)
    assert [p["text"] for p in first["parts"]] == ["first"]
    assert _observe(ctx, gateway, 1, after=0) is None, "a re-announced window shows nothing twice"
    daemon.revealed = 2
    real = run_events.read_run_events

    def short(*args, **kwargs):  # the wall-clock bound runs out after one event
        read = real(*args, **kwargs)
        read.events, read.through_seq, read.stop = read.events[:1], read.events[0].seq, "time"
        return read

    with monkeypatch.context() as bounded:
        bounded.setattr(run_events, "read_run_events", short)
        partial_record = _observe(ctx, gateway, 3, after=1)
    assert [p["text"] for p in partial_record["parts"]] == ["second"]
    assert partial_record["source"]["read_through"] == 2 and not partial_record.get("gaps")
    daemon.steps.append([_harness(4, "tool_call", tool={"name": "Read"})])
    daemon.revealed = 3
    caught_up = _observe(ctx, gateway, 4, after=3)
    assert caught_up["after_seq"] == 2 and [p["text"] for p in caught_up["parts"]] == ["third"]


def test_a_range_counts_as_shown_only_after_its_frame_was_handed_over(tmp_path):
    daemon = Daemon([[_harness(1, "message", text="said once")], [_harness(2, "tool_call", tool={"name": "Read"})]])
    gateway = _gateway(daemon)
    ctx = _ctx(tmp_path, [])
    daemon.revealed = 1

    def broken(_record):
        raise RuntimeError("progress channel closed")

    with pytest.raises(RuntimeError):
        _records(ctx, gateway, 1, emit=broken)
    daemon.revealed = 2
    retried = _observe(ctx, gateway, 2, after=1)
    assert (retried["after_seq"], [p["text"] for p in retried["parts"]]) == (0, ["said once"]), \
        "a frame that never left is read again, not acknowledged"


def test_a_process_that_lost_its_cursor_resumes_at_the_last_retained_range(tmp_path):
    steps = [[_harness(1, "message", text="one"), _harness(2, "tool_call", tool={"name": "Read"})],
             [_harness(3, "message", text="three")], [_harness(4, "message", text="four")]]
    daemon = Daemon(steps)
    gateway = _gateway(daemon)
    ctx = _ctx(tmp_path, [])
    daemon.revealed = 1
    _observe(ctx, gateway, 2)
    daemon.revealed = 2
    _observe(ctx, gateway, 3, after=2)
    delegate_activity.reset_process_memo()           # a new worker process
    daemon.revealed = 3
    resumed = _observe(ctx, gateway, 4, after=4)     # its wait starts at the durable journal cursor
    assert (resumed["after_seq"], [p["text"] for p in resumed["parts"]]) == (2, ["three", "four"]), \
        "the last retained range is read again (a reader shows seq 3 once) and nothing after it is skipped"
    assert [p["seq"] for p in resumed["parts"]] == [3, 4]


def test_an_engine_without_the_stream_keeps_a_marked_window_with_its_real_counts(tmp_path):
    daemon = Daemon([[_harness(1, "message", text="hi")]], stream=False)
    daemon.revealed = 1
    rows = [{"type": "harness.event", "title": "Bash", "severity": "info", "harnessId": "claude", "attemptId": "a01"},
            {"type": "harness.event", "title": "Reading chat.js", "textKind": "message", "harnessId": "claude",
             "attemptId": "a01"},
            {"type": "reviewer.failed", "title": "Reviewer setup failed", "severity": "error"}]
    record = _observe(_ctx(tmp_path, []), _gateway(daemon), 40, after=20, rows=rows, omitted=4)
    assert record["source"] == {"kind": "timeline_window", "reason": "stream_not_listed", "rows": 3, "rows_omitted": 4}
    assert [(p["kind"], p["text"]) for p in record["parts"]] == [
        ("message", "Reading chat.js"), ("problem", "Reviewer setup failed")]
    assert all("seq" not in p for p in record["parts"]), "a window row has no identity to invent"
    assert "seqs" not in record["technical"]
    assert "+4 earlier timeline rows not shown" in delegate_activity.progress_text(record)


# -- the terminal tail ----------------------------------------------------------------------

# -- the wait loop, the carrier and replay ------------------------------------------------

def _agent(tool_ctx):
    events = queue.Queue()
    agent = SimpleNamespace(
        _last_progress_ts=None, _event_queue=events, _current_chat_id=1,
        _current_task_id="child", tools=SimpleNamespace(_ctx=tool_ctx),
        _subagent_progress_meta=lambda event: subagent_message_meta({
            "delegation_role": "subagent", "root_task_id": "root", "parent_task_id": "root",
            "subagent_role": "critic", "model": "coordinator-model", "executor_route": "claude",
        }, task_id="child", event=event),
    )
    tool_ctx.emit_progress_fn = partial(OuroborosAgent._emit_progress, agent)
    return agent, events


@pytest.mark.parametrize("unit,needs_omission", [("\x00", True), ("\\", False), ("🙂", False)])
def test_serialized_speech_bound_survives_the_real_progress_carrier(tmp_path, unit, needs_omission):
    text = unit * (6000 // len(unit))
    originals = [_harness(seq, "message", text=text) for seq in range(1, 5)]
    daemon = Daemon([originals])
    daemon.revealed = 1
    ctx = _ctx(tmp_path, [])
    _agent_obj, events = _agent(ctx)
    _followed(ctx, daemon)
    frame = events.get_nowait()
    activity = frame["progress_meta"].get("delegated_activity")
    assert activity, "emission must not drop a record whose cursor is committed"
    assert len(json.dumps(activity, ensure_ascii=False)) <= delegate_activity._RECORD_CHARS
    assert delegated_activity_meta(activity, task_id="child") == activity
    omitted = activity.get("omitted", {}).get("message", 0)
    assert bool(omitted) == needs_omission
    assert activity["parts"] and len(activity["parts"]) + omitted == 4
    ref = activity["source"]["ref"]
    source = tmp_path / "task_results" / "artifacts" / "child" / ref["path"]
    assert [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()] == originals
    assert delegate_activity._memo("child", "run-1").shown_through == 4


def _followed(ctx, daemon):
    """The body's follow of ``daemon``'s run, one beat per revealed step (``delegate_feed.Feed._beat``): each beat
    reads the journal from the shown cursor and publishes that read (``delegate_progress.emit_stream``); the cursor
    moves only once a frame was handed over, so a beat whose frame failed reads the same range again."""
    gateway = _gateway(daemon)
    for step in range(1, len(daemon.steps) + 1):
        daemon.revealed = step
        after = delegate_activity.shown_cursor(ctx, ctx.task_id, daemon.run_id)
        read = read_run_events(gateway, daemon.run_id, after_seq=after, through_seq=None, timeout_sec=2.0)
        delegate_progress.emit_stream(ctx, daemon.run_id, read, after_seq=after)
    return gateway


def test_the_follow_streams_exact_ranges_and_delivery_replays_the_same_parts(tmp_path, monkeypatch):
    steps = [
        [_harness(1, "tool_call", tool={"name": "Read"}), _harness(2, "message", text="Reading the consumer.")],
        [_harness(3, "tool_call", tool={"name": "Bash"}), _harness(4, "message", text="Reading the consumer.")],
        [_harness(5, "message", text="All done; summary follows."), _run_event(6, "run.completed")],
    ]
    ctx = _ctx(tmp_path, [])
    _agent_obj, events = _agent(ctx)
    _followed(ctx, Daemon(steps))
    frames = [row for row in (events.get_nowait() for _ in range(events.qsize())) if row["type"] == "send_message"]
    activities = [frame["progress_meta"]["delegated_activity"] for frame in frames]
    assert [(a["after_seq"], a["through_seq"]) for a in activities] == [(0, 2), (2, 4), (4, 6)]
    assert [[p["text"] for p in a["parts"]] for a in activities] == [
        ["Reading the consumer."], ["Reading the consumer."], ["All done; summary follows."]]
    assert all(frame["progress_meta"]["narration"] is False and frame["role"] == "system" for frame in frames)
    assert frames[-1]["text"].startswith("💬 🛰 delegated run run-1 @seq 6\nclaude · a01: All done")

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
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "10"})))
    replay = [row for row in json.loads(response.body)["messages"] if row.get("is_progress")]
    for rows in (live, outbound, stored, replay):
        assert [row["delegated_activity"] for row in rows] == activities
        assert all(row["narration"] is False for row in rows)


@pytest.mark.parametrize("restart_memo", [False, True])
def test_a_terminal_delivery_catches_up_what_no_follow_frame_carried(tmp_path, monkeypatch, restart_memo):
    from ouroboros.tools import delegate
    from tests._delegated_transport_shared import _nanny_ctx

    steps = [[_harness(1, "message", text="Working on it.")],
             [_harness(2, "message", text="Finished."), _run_event(3, "run.completed")]]
    daemon = Daemon(steps)
    ctx = _nanny_ctx(tmp_path, task_id="child")
    ctx.task_attempt = 0
    _agent_obj, events = _agent(ctx)
    delivered = ctx.emit_progress_fn
    channel = {"open": False}

    def emit(text, **meta):
        if not channel["open"] and "delegated_activity" in meta:
            raise RuntimeError("progress channel closed")
        return delivered(text, **meta)

    ctx.emit_progress_fn = emit
    monkeypatch.setitem(delegate._CUSTODY, "run-1", RunCustody(task_id="child", run_id="run-1", route_id="claude",
                                                                model="m", access="readonly"))
    monkeypatch.setattr(delegate, "_capture_terminal_patch", lambda *a, **k: None)
    gateway = _followed(ctx, daemon)  # every read retained, no frame handed over
    first = json.loads(delegate._read_run(ctx, "run-1", 0, gateway=gateway))
    assert first["status"] == "terminal" and events.empty(), "no frame left, so nothing counts as shown"
    channel["open"] = True
    if restart_memo:
        delegate_activity.reset_process_memo()
    requests_before = list(daemon.requests)
    second = json.loads(delegate._read_run(ctx, "run-1", gateway=gateway))
    assert daemon.requests == requests_before, "retained output and activity never wait on remote I/O"
    assert second["status"] == "terminal"
    frames = [row for row in (events.get_nowait() for _ in range(events.qsize())) if row["type"] == "send_message"]
    activities = [frame["progress_meta"]["delegated_activity"] for frame in frames]
    assert activities and activities[0]["after_seq"] == 0 and activities[-1]["source"]["read_through"] == 3
    assert [p["text"] for activity in activities for p in activity["parts"]] == ["Working on it.", "Finished."]
    assert json.loads(delegate._read_run(ctx, "run-1", gateway=gateway))["status"] == "terminal"
    assert events.empty(), "only the successful frame commits the local activity cursor"


def test_a_wakes_read_over_a_reclaimed_journal_reads_detail_and_delivers_the_same_result(tmp_path, monkeypatch):
    """The wake's one read over the real gateway and socket, for one ended run with its journal kept and then
    reclaimed by retention: the result is the same, read from the detail; the journal is the body's to read
    (``tests/test_body_feed.py``)."""
    from ouroboros.tools import delegate
    from tests._delegated_transport_shared import _nanny_ctx

    detail = {"lastSeq": 2, "summary": {"runId": "run-1", "state": "succeeded", "effectiveAccess": "readonly"},
              "timeline": [], "primaryOutput": {"kind": "answer", "text": "done", "truncated": False}}
    attention = {"wakeReason": "attention", "attention": [0],
                 "summaries": [{"status": "succeeded", "pending": False, "token": "a" * 43, "problem": None}]}
    monkeypatch.setattr(delegate, "_capture_terminal_patch", lambda *a, **k: None)
    seen = {}
    for case, journal, tail in (
            ("kept", [_harness(1, "message", text="Working on it."), _run_event(2, "run.completed")], "event: end\ndata: {}\n\n"),
            ("reclaimed", [], _refusal("run_expired_by_retention", False))):
        delegate_activity.reset_process_memo()
        (tmp_path / case).mkdir()
        ctx = _nanny_ctx(tmp_path / case, task_id="child")
        ctx.task_attempt = 0
        monkeypatch.setitem(delegate._CUSTODY, "run-1", RunCustody(task_id="child", run_id="run-1", route_id="claude",
                                                                    model="m", access="readonly"))
        engine = LoopbackEngine(journal, tail=tail, detail=detail, attention=attention)
        with engine.serving() as gateway:
            result = json.loads(delegate._read_run(ctx, "run-1", 0, gateway=gateway))
        seen[case] = (result, engine)
    (kept, kept_engine), (reclaimed, engine) = seen["kept"], seen["reclaimed"]
    assert reclaimed == kept and kept["status"] == "terminal", "a journal the engine reclaimed never changes the result"
    assert engine.stream_reads() == [] == kept_engine.stream_reads(), "the wake reads no journal: it is the body's"
    assert ("GET", "/v2/runs/run-1", None) in engine.requests, "the wake's one read is the detail"
    assert not [body for method, path, body in engine.requests if path == "/v2/runs/wait-many"
                and body["targets"][0]["runId"] == "run-1"], "and no attention is read for it"


def test_the_binder_delivery_and_replay_drop_a_foreign_or_malformed_record(tmp_path):
    record = {"v": 1, "task_id": "child", "run_id": "run-1", "after_seq": 0, "through_seq": 2,
              "source": {"kind": "run_events"}, "parts": [{"kind": "message", "text": "hi", "actor": "claude/a01"}]}
    assert delegated_activity_meta(record, task_id="child") == record
    for change in ({"task_id": "parent"}, {"after_seq": 2}, {"v": 2}, {"source": {"kind": "prose"}},
                   {"parts": [{"kind": "speech", "text": "hi"}]}, {"parts": [{"kind": "message", "text": "x" * 70_000}]}):
        assert delegated_activity_meta({**record, **change}, task_id="child") == {}
    agent, events = _agent(SimpleNamespace(task_attempt=0))
    OuroborosAgent._emit_progress(agent, "run progress", delegated_activity=record)
    OuroborosAgent._emit_progress(agent, "foreign", delegated_activity={**record, "task_id": "parent"})
    assert events.get_nowait()["progress_meta"]["delegated_activity"] == record
    assert "delegated_activity" not in events.get_nowait()["progress_meta"]
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").touch()
    (logs / "progress.jsonl").write_text(json.dumps({
        "ts": "2026-09-27T00:00:00Z", "task_id": "parent", "content": "note", "delegated_activity": record,
    }) + "\n")
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "10"})))
    row, = json.loads(response.body)["messages"]
    assert row["text"] == "note" and "delegated_activity" not in row


def test_telegram_card_leads_with_the_executors_latest_words_and_keeps_its_counts():
    from skills.telegram.lib.telegram_state import _subagent_card_text

    record = {"v": 1, "task_id": "child", "run_id": "run-1", "after_seq": 0, "through_seq": 9,
              "source": {"kind": "run_events"},
              "parts": [{"kind": "message", "actor": "claude/a01", "text": "Tracing the Telegram consumer."},
                        {"kind": "problem", "actor": "claude/a01", "label": "Bash", "text": "exit 1"}],
              "technical": {"count": 7, "labels": [["Read", 4], ["Bash", 3]]}}
    event = {"subagent_event": "progress", "subagent_task_id": "child-task-1", "subagent_role": "critic",
             "text": "💬 🛰 delegated run run-1 @seq 9\n[claude/a01] Bash · tool_result", "delegated_activity": record}
    card = _subagent_card_text(event, "progress", "en")
    assert card.splitlines()[1:] == ["claude · a01: Tracing the Telegram consumer.", "⚠ claude · a01 Bash: exit 1",
                                     "7 technical events · Read ×4 · Bash ×3", "Full ref: task child-task-1"]
    event.pop("delegated_activity")
    assert "tool_result" in _subagent_card_text(event, "progress", "en"), "a frame without the record is unchanged"


@pytest.mark.parametrize(("problem_count", "recap"), [(1, True), (8, False)])
def test_telegram_long_speech_preserves_problems_gaps_and_full_task_reference(recap, problem_count):
    from skills.telegram.lib.telegram_state import _subagent_card_text

    speech = {"kind": "message", "actor": "claude/a01", "text": "I am reading the consumer. " * 100}
    activity = {"v": 1, "task_id": "child-full-id", "run_id": "run-1", "after_seq": 0, "through_seq": 9,
                "source": {"kind": "run_events", "read_through": 7},
                "parts": [{"kind": "problem", "actor": "claude/a01", "label": "Bash", "text": "exit 1"}] * problem_count
                         + [{"kind": "thinking", "text": "PRIVATE_THINKING"}],
                "gaps": [{"after_seq": 7, "through_seq": 9, "reason": "read_bound:time", "final": True}],
                "technical": {"count": 7, "labels": [["Read", 7]]}}
    if recap:
        activity["latest_message"] = speech
    else:
        activity["parts"].insert(0, speech)
    card = _subagent_card_text({"subagent_task_id": "child-full-id", "delegated_activity": activity}, "progress", "en")
    assert "I am reading the consumer" in card
    assert "OMISSION NOTE" in card
    assert "exit 1" in card and card.index("I am reading") < card.index("exit 1")
    assert "not shown" in card.lower() and "8–9" in card
    assert "thinking" in card and "details" in card and "PRIVATE_THINKING" not in card
    assert "7 technical events" in card
    if problem_count > 1:
        assert f"{problem_count} problems" in card
    assert "Full ref: task child-full-id" in card


def test_permanent_fallback_replays_all_retained_words_after_repeated_emission_failure(tmp_path):
    from ouroboros.gateways.claudexor_run_events import RunEvent, RunEventsRead

    ctx = _ctx(tmp_path, [])
    ctx.emit_progress_fn = lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("closed channel"))
    problem = {"code": "run_expired_by_retention", "retryable": False, "required_actions": []}
    for seq, text in ((1, "one"), (2, "two")):
        value = _harness(seq, "message", text=text)
        read = RunEventsRead(events=[RunEvent(seq, value["type"], json.dumps(value), value)], through_seq=seq,
                             problem=problem if seq == 2 else None)
        delegate_progress.emit_stream(ctx, "run-1", read, after_seq=seq - 1)
    records = _records(ctx, SimpleNamespace(), 3)
    assert [part["text"] for record in records for part in record["parts"]] == ["one", "two"]
    assert records[-1]["gaps"] == [{"after_seq": 2, "through_seq": 3,
                                    "reason": "stream_error:run_expired_by_retention", "retryable": False}]
    assert delegate_activity._memo("child", "run-1").shown_through == 3


def test_permanent_refusal_is_published_even_when_no_unread_tail_remains(tmp_path, monkeypatch):
    from ouroboros.gateways.claudexor_run_events import RunEvent, RunEventsRead

    monkeypatch.setattr(delegate_activity, "_persist", lambda *_a: {})
    value = _harness(1, "message", text="long words " * 2000)
    read = RunEventsRead(events=[RunEvent(1, value["type"], json.dumps(value), value)], through_seq=1,
                         problem={"code": "run_expired_by_retention", "retryable": False, "required_actions": []})
    ctx = _ctx(tmp_path, [])
    records = list(delegate_activity.stream_observations(ctx, "run-1", read, after=0))
    record, commit = records[0]
    commit()
    assert record["source"]["retained"] is False and record["parts"][0]["truncated"] is True
    assert record["source"]["retryable"] is False and not record.get("gaps")
    assert record["source"]["reason"] == "stream_error:run_expired_by_retention"
