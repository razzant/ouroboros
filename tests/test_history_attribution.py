"""Every history read is counted with why its client made it (``gateway/history_attribution.py``).

The route counts each GET in memory: the client's tags (``reason``, ``mode``, ``decision``, ``detail``) as values of
their closed sets, the status, the server's own refusal, the answer's bytes and its time (total, longest and a fixed
histogram); the window rides the next ``supervisor_load`` row as ``history_reads`` and starts again. The tags are
telemetry: the answer is byte for byte the same with or without them, an absent tag is ``unspecified`` and one
outside its set ``invalid``, nothing a client sent is kept raw, and a read writes nothing (DEVELOPMENT "Passive GET").
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
from ouroboros.gateway.history_attribution import DETAILS, drain, recorded

TS = "2026-10-16T10:00:{:02d}Z"
LOGGER = "ouroboros.gateway.history_attribution"


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
    assert section["reason"] == {"unspecified": 1, "settled": 1} and section["mode"] == {"newest": 2}
    assert (section["decision"], section["detail"], section["status"]) == (
        {"after_refused": 1}, {"task_settled": 1}, {"200": 2})  # a read no gate chose names no decision
    assert "room" not in section and drain() == {}, "no room is counted; a drained window starts empty"

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
    assert section["reason"] == {"invalid": 1, "unspecified": 3} and section["decision"] == {"invalid": 1}
    assert section["mode"] == {"after": 2, "older": 1, "around": 1} and section["detail"] == {"http_503": 1}
    kept = json.dumps(section)
    assert "<b>" not in kept and "sideways" not in kept and after not in kept and older not in kept

    # A refusal is counted with the server's code, cause and stream; the client's earlier leg with its own detail.
    monkeypatch.setattr(history_paging, "_DELTA_CEILING_BYTES", 8)
    _say(tmp_path, "third", "fourth", start=2)
    refused = _read(read, chat_id="1", cursor=after, reason="settled", mode="after")
    assert refused.status_code == 409
    section = drain()
    assert {key: section.get(key) for key in ("mode", "status", "code", "refusal", "stream")} == {
        "mode": {"after": 1}, "status": {"409": 1}, "code": {"history_delta_reset": 1},
        "refusal": {"delta_byte_limit": 1}, "stream": {"chat": 1}}
    _read(read, chat_id="1", cursor="not-a-cursor", mode="after")
    section = drain()
    assert (section["status"], section["code"], "refusal" in section) == ({"400": 1}, {"history_cursor_invalid": 1}, False)
    assert section["mode"] == {"after": 1}, "no kind decoded: the client's word"
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
    assert section["n"] == 1 and section["reason"] == {"open": 1}, "a HEAD reads nothing the client keeps"


def test_an_answer_that_is_no_error_page_still_counts_and_a_failed_count_fails_no_read(tmp_path, monkeypatch):
    from starlette.responses import PlainTextResponse

    async def plain(_request):
        return PlainTextResponse("upstream trouble", status_code=502)

    assert _read(recorded(plain), chat_id="1", reason="retry").status_code == 502
    section = drain()
    assert (section["n"], section["status"], section["reason"], "code" in section) == (1, {"502": 1}, {"retry": 1}, False)

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
    assert row["history_reads"]["n"] == 1 and row["history_reads"]["reason"] == {"open": 1}
    written = json.loads((tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert written["history_reads"] == row["history_reads"] and drain() == {}, "the row took the window"
