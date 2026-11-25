"""Every history read is recorded with why its client made it (``gateway/history_attribution.py``).

One ``history_read`` line per request at the route: the room, the client's tags (``reason``, ``mode``,
``decision``, ``detail``) as values of their closed sets, the status, the server's own refusal, the answer's bytes
and its time. The tags are telemetry: the answer is byte for byte the same with or without them, an absent tag is
``unspecified`` and one outside its set ``invalid``, and nothing a client sent is written back raw.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

from ouroboros import dialogue_store as ds
from ouroboros.gateway import history_paging
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.gateway.history_attribution import DETAILS, recorded

TS = "2026-10-16T10:00:{:02d}Z"
LOGGER = "ouroboros.gateway.history_attribution"


def _say(root, *texts, start=0):
    for index, text in enumerate(texts):
        ds.append(root, "chat", {"ts": TS.format(start + index), "direction": "in", "chat_id": 1, "text": text})


def _read(endpoint, **params):
    return asyncio.run(endpoint(SimpleNamespace(query_params=params)))


def _records(caplog):
    return [dict(item.split("=", 1) for item in record.getMessage().split()[1:])
            for record in caplog.records if record.name == LOGGER and record.getMessage().startswith("history_read ")]


def test_a_read_is_recorded_with_its_tags_and_answers_the_same_without_them(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger=LOGGER)
    _say(tmp_path, "first", "second")
    read = recorded(make_chat_history_endpoint(tmp_path))
    bare = _read(read, chat_id="1")
    tagged = _read(read, chat_id="1", reason="settled", mode="newest", decision="after_refused", detail="task_settled")
    assert bare.status_code == tagged.status_code == 200 and bare.body == tagged.body  # telemetry, never a selector
    untagged, told = _records(caplog)
    assert untagged == {"room": "1", "reason": "unspecified", "mode": "newest", "status": "200",
                        "bytes": str(len(bare.body)), "ms": untagged["ms"]} and untagged["ms"].isdigit()
    assert [told.get(key) for key in ("reason", "mode", "decision", "detail")] == [
        "settled", "newest", "after_refused", "task_settled"]

    # A cursor is opaque to the record: its client names its kind; an `around` read and the newest page are the
    # request's own. Values outside their sets are `invalid`, never written as sent; an HTTP status is a detail.
    after = json.loads(bare.body)["after_cursor"]
    _read(read, chat_id="1", cursor=after, reason="<b>open</b>", mode="sideways", decision="x" * 300, detail="http_503")
    _read(read, chat_id="1", cursor=after)
    _read(read, chat_id="1", around="task:nothing-here", mode="older")
    strange, unnamed, around = _records(caplog)[2:]
    assert [strange[key] for key in ("reason", "mode", "decision", "detail")] == ["invalid", "invalid", "invalid", "http_503"]
    assert "<b>" not in caplog.text and "sideways" not in caplog.text and after not in caplog.text
    assert (unnamed["reason"], unnamed["mode"], "decision" in unnamed) == ("unspecified", "unspecified", False)
    assert around["mode"] == "around"

    # A refusal is recorded with the server's code, cause and stream; the client's earlier leg with its own detail.
    monkeypatch.setattr(history_paging, "_DELTA_CEILING_BYTES", 8)
    _say(tmp_path, "third", "fourth", start=2)
    refused = _read(read, chat_id="1", cursor=after, reason="settled", mode="after")
    assert refused.status_code == 409
    assert {key: _records(caplog)[-1].get(key) for key in ("reason", "mode", "status", "code", "refusal", "stream")} == {
        "reason": "settled", "mode": "after", "status": "409", "code": "history_delta_reset",
        "refusal": "delta_byte_limit", "stream": "chat"}
    _read(read, chat_id="1", cursor="not-a-cursor", mode="older")
    invalid = _records(caplog)[-1]
    assert (invalid["status"], invalid["code"], "refusal" in invalid) == ("400", "history_cursor_invalid", False)
    assert "not-a-cursor" not in caplog.text
    assert history_paging.DELTA_RESET_DETAILS <= DETAILS  # a client may echo every refusal the server names


def test_the_history_route_records_every_request_and_a_failed_record_fails_no_read(tmp_path, caplog, monkeypatch):
    from ouroboros.gateway import history_attribution
    from ouroboros.gateway.router import _room_reading_routes

    caplog.set_level(logging.INFO, logger=LOGGER)
    _say(tmp_path, "hello")
    route, = (route for route in _room_reading_routes(tmp_path) if route.path == "/api/chat/history")
    answer = _read(route.endpoint, chat_id="1", reason="open", mode="newest")
    assert answer.status_code == 200 and _records(caplog)[-1]["reason"] == "open"

    def broken(*_args):
        raise RuntimeError("the record could not be made")

    monkeypatch.setattr(history_attribution, "_fields", broken)
    again = _read(route.endpoint, chat_id="1", reason="open")
    assert again.status_code == 200 and again.body == answer.body and len(_records(caplog)) == 1
