"""Every history read is counted with why its client made it and what it cost (``gateway/history_attribution.py``).

The route counts each GET in memory under its cause: the client's tags (``reason``, ``mode``, ``decision``,
``detail``) as values of their closed sets, the status and the server's own refusal; each cause keeps its reads, bytes
and time (total, longest), and the window its totals and a fixed time histogram. The window rides the next
``supervisor_load`` row as ``history_reads`` and starts again. The tags are telemetry: the answer is byte for byte the
same with or without them, an absent tag is ``unspecified`` and one outside its set ``invalid``, nothing a client sent
is kept raw, and a read writes nothing (DEVELOPMENT "Passive GET").
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store as ds
from ouroboros.gateway import history_attribution, history_paging
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.gateway.history_attribution import DETAILS, GROUPS_MAX, drain, recorded

TS = "2026-10-16T10:00:{:02d}Z"
LOGGER = "ouroboros.gateway.history_attribution"
COST = ("n", "bytes", "total_ms", "max_ms")


@pytest.fixture(autouse=True)
def window():
    drain()  # the window is the process's: each case counts only its own reads
    yield
    drain()


def _say(root, *texts, start=0):
    for index, text in enumerate(texts):
        ds.append(root, "chat", {"ts": TS.format(start + index), "direction": "in", "chat_id": 1, "text": text})


def _read(endpoint, **params):
    return asyncio.run(endpoint(SimpleNamespace(method="GET", query_params=params)))


def _cause(group):
    return {key: value for key, value in group.items() if key not in COST}


def _tally(section, word):
    """Reads per value of one counted word over the window's causes (a cause without the word adds none)."""
    tally = {}
    for group in section["groups"]:
        if word in group:
            tally[group[word]] = tally.get(group[word], 0) + group["n"]
    return tally


def test_a_read_is_counted_with_its_tags_and_answers_the_same_without_them(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    _say(tmp_path, "first", "second")
    read = recorded(make_chat_history_endpoint(tmp_path))
    bare = _read(read, chat_id="1")
    tagged = _read(read, chat_id="1", reason="settled", mode="newest", decision="after_refused", detail="task_settled")
    assert bare.status_code == tagged.status_code == 200 and bare.body == tagged.body  # telemetry, never a selector
    section = drain()
    assert (section["n"], section["bytes"]) == (2, 2 * len(bare.body)) and section["max_ms"] <= section["total_ms"]
    assert sum(section["seconds"].values()) == 2 and list(section["seconds"]) == ["<0.5", "<1", "<2", "<5", "<10", ">=10"]
    assert [_cause(group) for group in section["groups"]] == [
        {"reason": "unspecified", "mode": "newest", "status": "200"},  # a read no gate chose names no decision
        {"reason": "settled", "mode": "newest", "decision": "after_refused", "detail": "task_settled", "status": "200"}]
    assert [group["bytes"] for group in section["groups"]] == [len(bare.body)] * 2
    assert not any("room" in part for part in (section, *section["groups"])), "no room is counted"
    assert drain() == {}, "a drained window starts empty"

    # A cursor's kind is the one the server decoded, whatever its client said or left out; an `around` read and the
    # newest page are the request's own. Values outside their sets are `invalid`, never kept as sent; an HTTP status
    # is a detail.
    after = json.loads(bare.body)["after_cursor"]
    older = json.loads(_read(read, chat_id="1", n_human="1").body)["next_cursor"]
    drain()
    _read(read, chat_id="1", cursor=after, reason="<b>open</b>", mode="sideways", decision="x" * 300, detail="http_503")
    _read(read, chat_id="1", cursor=after)
    _read(read, chat_id="1", cursor=older, mode="after")
    _read(read, chat_id="1", around="task:nothing-here", mode="older")
    section = drain()
    assert _tally(section, "reason") == {"invalid": 1, "unspecified": 3}
    assert _tally(section, "decision") == {"invalid": 1} and _tally(section, "detail") == {"http_503": 1}
    assert _tally(section, "mode") == {"after": 2, "older": 1, "around": 1}
    kept = json.dumps(section)
    assert "<b>" not in kept and "sideways" not in kept and after not in kept and older not in kept

    # A refusal is counted with the server's code, cause and stream; the client's earlier leg with its own detail.
    monkeypatch.setattr(history_paging, "_DELTA_CEILING_BYTES", 8)
    _say(tmp_path, "third", "fourth", start=2)
    refused = _read(read, chat_id="1", cursor=after, reason="settled", mode="after")
    assert refused.status_code == 409
    section = drain()
    assert [_cause(group) for group in section["groups"]] == [
        {"reason": "settled", "mode": "after", "status": "409", "code": "history_delta_reset",
         "refusal": "delta_byte_limit", "stream": "chat"}]
    _read(read, chat_id="1", cursor="not-a-cursor", mode="after")
    section = drain()
    assert [_cause(group) for group in section["groups"]] == [
        {"reason": "unspecified", "mode": "after", "status": "400", "code": "history_cursor_invalid"}]
    assert _tally(section, "mode") == {"after": 1}, "no kind decoded: the client's word"
    assert "not-a-cursor" not in json.dumps(section)
    assert history_paging.DELTA_RESET_DETAILS <= DETAILS  # a client may echo every refusal the server names
    assert not [record for record in caplog.records if record.name == LOGGER], "a read writes nothing, not even a line"


def test_the_route_counts_each_get_once_and_never_a_head(tmp_path):
    from starlette.applications import Starlette
    from starlette.testclient import TestClient
    from ouroboros.gateway.router import _room_reading_routes

    _say(tmp_path, "hello")
    route, = (route for route in _room_reading_routes(tmp_path) if route.path == "/api/chat/history")
    with TestClient(Starlette(routes=[route])) as client:
        got = client.get("/api/chat/history?chat_id=1&reason=open")
        head = client.head("/api/chat/history?chat_id=1&reason=refresh")
    assert got.status_code == head.status_code == 200
    section = drain()
    assert section["n"] == 1 and _tally(section, "reason") == {"open": 1}, "a HEAD reads nothing the client keeps"


def test_an_answer_that_is_no_error_page_still_counts_and_a_failed_count_fails_no_read(tmp_path, monkeypatch):
    from starlette.responses import PlainTextResponse

    async def plain(_request):
        return PlainTextResponse("upstream trouble", status_code=502)

    assert _read(recorded(plain), chat_id="1", reason="retry").status_code == 502
    section = drain()
    assert section["n"] == 1 and [_cause(group) for group in section["groups"]] == [
        {"reason": "retry", "mode": "newest", "status": "502"}]  # no code: the body named none

    _say(tmp_path, "hello")
    read = recorded(make_chat_history_endpoint(tmp_path))
    answer = _read(read, chat_id="1", reason="open")

    def broken(*_args):
        raise RuntimeError("the count could not be made")

    monkeypatch.setattr(history_attribution, "_fields", broken)
    again = _read(read, chat_id="1", reason="open")
    assert again.status_code == 200 and again.body == answer.body and drain()["n"] == 1


def test_a_reads_time_lands_in_its_bucket(tmp_path, monkeypatch):
    async def instant(_request):
        return SimpleNamespace(status_code=200, body=b"{}")

    clock = iter([0.0, 0.2, 10.0, 11.5, 20.0, 31.0])  # 200 ms, 1.5 s, 11 s
    monkeypatch.setattr(history_attribution, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    for _ in range(3):
        _read(recorded(instant), chat_id="1")
    section = drain()
    assert section["seconds"] == {"<0.5": 1, "<1": 0, "<2": 1, "<5": 0, "<10": 0, ">=10": 1}
    assert (section["total_ms"], section["max_ms"]) == (12700, 11000)


def test_each_cause_keeps_its_own_cost(monkeypatch):
    """Two windows of the same two reads with their causes swapped (a 3 MB, 11 s ``open`` beside a 100-byte, 17 ms
    ``settled``, then the reverse) have the same totals and histogram, and still read differently: each read's
    bytes and time stay with the cause that made it, which counts per word beside one total could not tell apart."""
    sizes = {}

    async def answer(request):
        return SimpleNamespace(status_code=200, body=b"x" * sizes[request.query_params["reason"]])

    def window(costly, cheap):
        sizes.update({costly: 3_000_000, cheap: 100})
        clock = iter([0.0, 11.0, 20.0, 20.017])  # the first read takes 11 s, the second 17 ms
        monkeypatch.setattr(history_attribution, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
        _read(recorded(answer), chat_id="1", reason=costly)
        _read(recorded(answer), chat_id="1", reason=cheap)
        return drain()

    first, second = window("open", "settled"), window("settled", "open")
    assert first != second, "the same reads with their causes swapped read the same"
    assert {key: first[key] for key in (*COST, "seconds")} == {key: second[key] for key in (*COST, "seconds")}
    for section, costly, cheap in ((first, "open", "settled"), (second, "settled", "open")):
        cost = {group["reason"]: {key: group[key] for key in COST} for group in section["groups"]}
        assert cost == {costly: {"n": 1, "bytes": 3_000_000, "total_ms": 11_000, "max_ms": 11_000},
                        cheap: {"n": 1, "bytes": 100, "total_ms": 17, "max_ms": 17}}, section


def test_causes_beyond_the_bound_share_one_overflow_group_and_a_drain_starts_afresh():
    """A window keeps at most ``GROUPS_MAX`` causes (a row-size bound a busy minute never meets); each further one
    is counted in one ``overflow`` group with its reads and cost, never dropped, so the groups add up to the window;
    a cause already kept keeps counting in its own group, and the next window starts with none."""
    async def answer(_request):
        return SimpleNamespace(status_code=200, body=b"{}")

    read = recorded(answer)
    details = [f"http_{status}" for status in range(500, 500 + GROUPS_MAX + 6)]  # six causes past the bound
    for detail in details:
        _read(read, chat_id="1", detail=detail)
    _read(read, chat_id="1", detail=details[0])  # a kept cause
    _read(read, chat_id="1", detail=details[-1])  # a cause past the bound
    section = drain()
    groups = section["groups"]
    assert [group.get("detail") for group in groups] == [*details[:GROUPS_MAX], None]
    assert _cause(groups[-1]) == {"overflow": True} and (groups[0]["n"], groups[-1]["n"]) == (2, 7)
    assert section["n"] == len(details) + 2
    for key in ("n", "bytes", "total_ms"):
        assert sum(group[key] for group in groups) == section[key], key
    assert max(group["max_ms"] for group in groups) == section["max_ms"]

    _read(read, chat_id="1", detail=details[-1])
    assert [_cause(group) for group in drain()["groups"]] == [
        {"reason": "unspecified", "mode": "newest", "detail": details[-1], "status": "200"}]


def test_reads_counted_while_the_supervisor_drains_land_whole_in_one_window_each():
    """The route counts on the event loop with the cursor kind its read's thread decoded; the supervisor drains on
    its own thread. Every GET lands in exactly one drained window, whole (its window's groups add up to its totals),
    with the decoded kind; a HEAD in none."""
    import sys
    import threading

    async def answer(_request):
        def select():  # as ``select_history_page`` publishes a decoded cursor's kind (``asyncio.to_thread`` context)
            history_paging.DECODED_CURSOR.get()["kind"] = "older"
        await asyncio.to_thread(select)
        return SimpleNamespace(status_code=200, body=b"abc")

    read, windows, done = recorded(answer), [], threading.Event()

    def supervisor():
        while not done.is_set():
            if window := drain():  # an empty window is kept by nobody
                windows.append(window)

    async def reads():
        for index in range(400):
            method = "HEAD" if index % 4 == 3 else "GET"
            await read(SimpleNamespace(method=method, query_params={"chat_id": "1", "cursor": "c", "mode": "after"}))

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads as often as the interpreter allows
    drainer = threading.Thread(target=supervisor)
    try:
        drainer.start()
        asyncio.run(reads())
    finally:
        done.set()
        drainer.join()
        sys.setswitchinterval(interval)
    windows = [window for window in (*windows, drain()) if window]  # the last reads may wait for this drain
    for window in windows:
        assert [sum(group[key] for group in window["groups"]) for key in COST[:3]] == [window[key] for key in COST[:3]]
        assert sum(window["seconds"].values()) == window["n"]
    assert sum(window["n"] for window in windows) == 300 and sum(window["bytes"] for window in windows) == 900
    assert all(_tally(window, "mode") == {"older": window["n"]} for window in windows), "the decoded kind, not 'after'"
    assert drain() == {}


def test_the_window_rides_the_next_supervisor_load_row(monkeypatch, tmp_path):
    from ouroboros import server_liveness
    from supervisor import loop_load

    assert server_liveness._load_sections()["history_reads"] is drain
    _say(tmp_path, "hello")
    _read(recorded(make_chat_history_endpoint(tmp_path)), chat_id="1", reason="open")
    monkeypatch.setattr(loop_load, "CONSUMER", loop_load.ConsumerWindow())
    monkeypatch.setattr(server_liveness, "_load_sections", lambda: {"history_reads": drain})
    monkeypatch.setattr(server_liveness.BOOT, "_load", None)
    monkeypatch.setattr(server_liveness, "SUPERVISOR_LOAD_WINDOW_SEC", 0.0)
    loop_load.CONSUMER.note("task_heartbeat", "telemetry", 0.001, None)
    row = server_liveness.publish_supervisor_load(SimpleNamespace(), tmp_path)
    assert row["history_reads"]["n"] == 1 and _tally(row["history_reads"], "reason") == {"open": 1}
    written = json.loads((tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert written["history_reads"] == row["history_reads"] and drain() == {}, "the row took the window"
