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
import json
import queue
from functools import partial
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


def _ctx(tmp_path, emitted):
    return SimpleNamespace(task_id="child", task_attempt=0, drive_root=tmp_path, task_metadata={},
                           emit_progress_fn=lambda text, **meta: emitted.append((text, meta)))


def _records(ctx, gateway, through, *, after=0, rows=(), omitted=0, drain=None, emit=None):
    """Every record of one advance, each committed after ``emit`` (default: accepted)."""
    advance = SimpleNamespace(seq=through, events=list(rows), events_omitted=omitted)
    out = []
    for record, commit in delegate_activity.observations(ctx, gateway, "run-1", advance, after_seq=after,
                                                         read_sec=2.0, drain_sec=drain):
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
    assert partial_record["gaps"] == [{"after_seq": 2, "through_seq": 3, "reason": "read_bound:time"}]
    assert "read at the run's next advance or its end" in delegate_activity.progress_text(partial_record)
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


def test_a_failed_read_is_a_provisional_window_and_the_exact_range_is_read_again(tmp_path):
    daemon = Daemon([[_harness(1, "message", text="hi")], [_harness(2, "message", text="later")]])
    gateway = _gateway(daemon)
    ctx = _ctx(tmp_path, [])
    daemon.revealed = 1
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(
        lambda request: httpx.Response(503, json={"code": "busy"}) if request.url.path.endswith("/events")
        else daemon.handler(request)))
    record = _observe(ctx, gateway, 1, rows=[{"title": "hi", "textKind": "message"}])
    assert record["source"] == {"kind": "timeline_window", "reason": "stream_unavailable:busy", "provisional": True, "rows": 1}
    assert record["gaps"] == [{"after_seq": 0, "through_seq": 1, "reason": "stream_unavailable:busy"}]
    assert "the exact read failed" in delegate_activity.progress_text(record)
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(daemon.handler))
    daemon.revealed = 2
    exact = _observe(ctx, gateway, 2, after=1)
    assert (exact["after_seq"], [p["seq"] for p in exact["parts"]]) == (0, [1, 2]), \
        "the failed range was not skipped: the exact read covers it and supersedes the window"


# -- the terminal tail ----------------------------------------------------------------------

def test_a_terminal_drain_reads_every_bounded_step_to_the_journals_end(tmp_path, monkeypatch):
    import ouroboros.gateways.claudexor_run_events as run_events

    events = [_harness(seq, "message", text=f"line {seq} " + "x" * 300) for seq in range(1, 10)]
    daemon = Daemon([events + [_run_event(10, "run.completed")]])
    daemon.revealed = 1
    gateway = _gateway(daemon)
    real = run_events.read_run_events
    monkeypatch.setattr(run_events, "read_run_events", lambda *a, **k: real(*a, **{**k, "max_bytes": 1200}))
    records = _records(_ctx(tmp_path, []), gateway, 10, drain=30)
    assert len(records) > 2, "the byte bound cut the tail into several reads"
    assert [r["after_seq"] for r in records] == [0] + [r["source"]["read_through"] for r in records[:-1]], \
        "each read resumes exactly where the previous one stopped"
    assert records[-1]["source"]["read_through"] == 10 and records[-1]["source"]["ended"] is True
    assert not any(r.get("gaps") for r in records), "nothing is left for a next observation that will not come"
    assert [p["seq"] for r in records for p in r["parts"]] == list(range(1, 10))


def test_a_terminal_drain_out_of_time_states_a_final_gap_and_where_the_originals_are(tmp_path, monkeypatch):
    import ouroboros.gateways.claudexor_run_events as run_events

    events = [_harness(seq, "message", text=f"line {seq} " + "x" * 300) for seq in range(1, 10)]
    daemon = Daemon([events + [_run_event(10, "run.completed")]])
    daemon.revealed = 1
    gateway = _gateway(daemon)
    ctx = _ctx(tmp_path, [])
    real = run_events.read_run_events
    clock = [0.0]

    def spend_allowance(*args, **kwargs):
        read = real(*args, **{**kwargs, "max_bytes": 1200})
        clock[0] = 1.0
        return read

    with monkeypatch.context() as bounded:
        bounded.setattr(delegate_activity, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        bounded.setattr(run_events, "read_run_events", spend_allowance)
        records = _records(ctx, gateway, 10, drain=1)
    assert len(records) == 1, "only the first read was admitted before the allowance expired"
    gap, = records[0]["gaps"]
    assert gap == {"after_seq": records[0]["source"]["read_through"], "through_seq": 10, "reason": "read_bound:time",
                   "final": True, "where": "run_journal"}
    assert "another delegate_wait on the run reads them" in delegate_activity.progress_text(records[0])
    rest = _records(ctx, gateway, 10, after=10, drain=30)
    assert rest[0]["after_seq"] == gap["after_seq"] and rest[-1]["source"]["ended"] is True, \
        "a later wait on the ended run reads the remainder from the same journal, without a new run"


def test_a_terminal_read_that_keeps_failing_is_a_final_provisional_gap(tmp_path, monkeypatch):
    daemon = Daemon([[_harness(1, "message", text="hi")]])
    daemon.revealed = 1
    gateway = _gateway(daemon)
    calls = []

    def flaky(request):
        if request.url.path.endswith("/events"):
            calls.append(request)
            return httpx.Response(503, json={"code": "busy"})
        return daemon.handler(request)

    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(flaky))
    clock = [0.0]
    monkeypatch.setattr(delegate_activity, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)))
    record, = _records(_ctx(tmp_path, []), gateway, 1, drain=2, rows=[{"title": "hi", "textKind": "message"}])
    assert len(calls) > 1, "the failed terminal read was retried inside its allowance"
    assert record["source"]["provisional"] is True
    assert record["gaps"] == [{"after_seq": 0, "through_seq": 1, "reason": "stream_unavailable:busy",
                               "final": True, "where": "run_journal"}]


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


@pytest.mark.parametrize("unit,needs_omission", [("\x00", True), ("\x00\\", True), ("\\", False), ("🙂", False)])
def test_serialized_speech_bound_survives_the_real_progress_carrier(tmp_path, unit, needs_omission):
    text = unit * (6000 // len(unit))
    originals = [_harness(seq, "message", text=text) for seq in range(1, 5)]
    daemon = Daemon([originals])
    daemon.revealed = 1
    ctx = _ctx(tmp_path, [])
    _agent_obj, events = _agent(ctx)
    delegate_progress.emit(ctx, "run-1", SimpleNamespace(seq=4, events=[], events_omitted=0),
                           entry=SimpleNamespace(task_id="child", run_id="run-1"), gateway=_gateway(daemon), after_seq=0)
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
    assert [json.loads(line) for line in source.read_text().splitlines()] == originals
    assert delegate_activity._memo("child", "run-1").shown_through == 4


@pytest.mark.parametrize("drain", [None, 2])
def test_end_before_the_observed_fence_discloses_a_gap_and_remains_readable(tmp_path, drain):
    # Conditional inconsistent source: this fixture does not assert the engine
    # emits an early end in production. End alone cannot prove fence coverage.
    daemon = Daemon([[_harness(1, "message", text="first")]])
    daemon.revealed = 1
    gateway, ctx = _gateway(daemon), _ctx(tmp_path, [])
    record, = _records(ctx, gateway, 3, drain=drain)
    assert record["source"]["ended"] and record["source"]["read_through"] == 1
    assert record.get("gaps") == [{"after_seq": 1, "through_seq": 3, "reason": "stream_end_before_fence",
                                   **({"final": True, "where": "run_journal"} if drain is not None else {})}]
    assert delegate_activity._memo("child", "run-1").shown_through == 1
    assert "coverage unresolved" in delegate_activity.progress_text(record)
    daemon.steps[0].extend([_harness(2, "message", text="second"), _run_event(3, "run.completed")])
    resumed, = _records(ctx, gateway, 3, drain=drain)
    assert resumed["after_seq"] == 1 and resumed["source"]["read_through"] == 3
    assert not resumed.get("gaps")
    assert [part["text"] for part in resumed["parts"]] == ["second"]


def test_wait_streams_exact_ranges_drains_the_terminal_tail_and_replays_the_same_parts(tmp_path, monkeypatch):
    from ouroboros.tools import delegate
    from tests._delegated_transport_shared import _nanny_ctx

    steps = [
        [_harness(1, "tool_call", tool={"name": "Read"}), _harness(2, "message", text="Reading the consumer.")],
        [_harness(3, "tool_call", tool={"name": "Bash"}), _harness(4, "message", text="Reading the consumer.")],
        [_harness(5, "message", text="All done; summary follows."), _run_event(6, "run.completed")],
    ]
    daemon = Daemon(steps)
    gateway = _gateway(daemon)
    ctx = _nanny_ctx(tmp_path, task_id="child")
    ctx.task_attempt = 0
    _agent_obj, events = _agent(ctx)
    monkeypatch.setitem(delegate._CUSTODY, "run-1", RunCustody(task_id="child", run_id="run-1", route_id="claude",
                                                                model="m", access="readonly"))
    monkeypatch.setattr(delegate, "_capture_terminal_patch", lambda *a, **k: None)
    clock = SimpleNamespace(now=0.0)
    with monkeypatch.context() as timing:
        timing.setattr(delegate.time, "monotonic", lambda: clock.now)
        timing.setattr(delegate.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
        result = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=60, since_seq=0, gateway=gateway))
    assert result["status"] == "terminal", result
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
def test_a_later_wait_on_an_ended_run_catches_up_what_no_frame_carried(tmp_path, monkeypatch, restart_memo):
    from ouroboros.tools import delegate
    from tests._delegated_transport_shared import _nanny_ctx

    steps = [[_harness(1, "message", text="Working on it.")],
             [_harness(2, "message", text="Finished."), _run_event(3, "run.completed")]]
    daemon = Daemon(steps)
    gateway = _gateway(daemon)
    handshaken = len(daemon.requests)
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
    clock = SimpleNamespace(now=0.0)
    with monkeypatch.context() as timing:
        timing.setattr(delegate.time, "monotonic", lambda: clock.now)
        timing.setattr(delegate.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
        first = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=60, since_seq=0, gateway=gateway))
        assert first["status"] == "terminal" and events.empty(), "no frame left, so nothing counts as shown"
        channel["open"] = True
        if restart_memo:
            from ouroboros.delegate_activity import reset_process_memo
            reset_process_memo()
        requests_before = list(daemon.requests)
        second = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=60, gateway=gateway))
        assert daemon.requests == requests_before, "retained output and activity never wait on remote I/O"
    assert second["status"] == "terminal"
    frames = [row for row in (events.get_nowait() for _ in range(events.qsize())) if row["type"] == "send_message"]
    activity, = [frame["progress_meta"]["delegated_activity"] for frame in frames]
    assert (activity["after_seq"], activity["source"]["read_through"]) == (0, 3)
    if restart_memo:
        assert "ended" not in activity["source"], "the raw source did not retain the stream's end-frame receipt"
    else:
        assert activity["source"]["ended"] is True
    assert [p["text"] for p in activity["parts"]] == ["Working on it.", "Finished."]
    assert json.loads(delegate._delegate_wait(ctx, "run-1", gateway=gateway))["status"] == "terminal"
    assert events.empty(), "only the successful frame commits the local activity cursor"
    assert {method for method, _path, _cursor in daemon.requests[handshaken:]} == {"GET"}, \
        "catching up reads the ended run's journal: no new run, answer or control request"


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


def test_plain_emit_without_the_waits_transport_keeps_the_one_argument_contract():
    advance = delegate_progress.WindowObservations().record({"timeline": [{"title": "read"}]}, 3, 0)
    plain = []
    delegate_progress.emit(SimpleNamespace(emit_progress_fn=plain.append, task_id="child"), "run-1", advance)
    assert plain == [delegate_progress.live_line("run-1", advance)]


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


@pytest.mark.parametrize("recap", [False, True])
@pytest.mark.parametrize("problem_count", [1, 8])
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


def test_a_terminal_drain_yields_to_the_owners_stop_between_reads(tmp_path, monkeypatch):
    import ouroboros.gateways.claudexor_run_events as run_events
    from ouroboros import cancel_intents

    events = [_harness(seq, "message", text=f"line {seq} " + "x" * 300) for seq in range(1, 10)]
    daemon = Daemon([events + [_run_event(10, "run.completed")]])
    daemon.revealed = 1
    gateway = _gateway(daemon)
    real = run_events.read_run_events
    stopped = [False]

    def stop_in_flight(*args, **kwargs):
        read = real(*args, **{**kwargs, "max_bytes": 1200})
        stopped[0] = True
        return read

    monkeypatch.setattr(run_events, "read_run_events", stop_in_flight)
    monkeypatch.setattr(cancel_intents, "cancel_pending", lambda root, task_id, **_: stopped[0] and task_id == "child")
    records = _records(_ctx(tmp_path, []), gateway, 10, drain=30)
    assert len(records) == 1, "a pending Stop ends the drain after the read in flight"
    assert records[0]["gaps"][0]["reason"] == "owner_cancelled" and records[0]["gaps"][0]["final"] is True
    assert records[0]["source"]["ref"], "bytes read before Stop remain retained"
    assert len([r for r in daemon.requests if r[1].endswith("/events")]) == 1
    (tmp_path / "state").mkdir(exist_ok=True)
    from tests._work_stop_helpers import live_restart
    live_restart(tmp_path, monkeypatch)
    flaky = _gateway(Daemon([[_harness(1, "message", text="hi")]]))
    flaky._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(
        lambda request: httpx.Response(503, json={"code": "busy"}) if request.url.path.endswith("/events")
        else daemon.handler(request)))
    monkeypatch.setattr(cancel_intents, "cancel_pending", lambda *_a, **_k: False)
    delegate_activity.reset_process_memo()
    before = len(daemon.requests)
    other = SimpleNamespace(task_id="other", task_attempt=0, drive_root=tmp_path, task_metadata={})
    record, = _records(other, flaky, 1, drain=30)
    assert len(daemon.requests) == before, "Restart admits no catalog or stream request"
    assert record["gaps"][0]["reason"] == "owner_owner_restart"
