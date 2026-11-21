"""``GET /api/recall``: the neighbour door over the conversation (``ouroboros/gateway/recall.py``, K9).

The door projects the words reader (``dialogue_recall.collect``) outward and reopens one row through the shared row
resolver: admitted dialogue rows of every room and nothing else, each a closed ``RecallRow``; every printed reader is
the door's own; no model turn; one content-free event per admitted call. Who may call (the raw loopback peer) is
pinned in ``tests/test_browser_boundary.py``.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import urllib.parse

import pytest

from ouroboros import dialogue_recall, dialogue_store as ds
from ouroboros.gateway import recall
from ouroboros.gateway.history_contracts import RecallRowResponse, RecallSearchResponse
from ouroboros.gateway.schema import validate_ingress

REPO = pathlib.Path(__file__).resolve().parents[1]
MARKERS = ("SECRETVALUEMARK", "SECRETLABELMARK", "SECRETQUESTIONMARK", "ATTACHMARK", "NOTEMARK", "SETTINGSMARK",
           "SEALPAGEMARK", "A2AMARK", "TRANSPORTRAWMARK")


def _say(root, ts, text, *, direction="in", chat_id=1, **extra):
    ds.append(root, "chat", {"ts": ts, "direction": direction, "chat_id": chat_id, "text": text, **extra})


def _ok(root, *items):
    status, body, mode = recall.answer(root, list(items))
    assert status == 200, body
    shape = RecallRowResponse if mode == "row" else RecallSearchResponse
    assert not validate_ingress(body, shape), validate_ingress(body, shape)
    return body


def _search(root, *words, **options):
    return _ok(root, *[("words", word) for word in words], *[(key, str(value)) for key, value in options.items()])


def _refused(root, *items):
    status, body, _mode = recall.answer(root, list(items))
    assert status != 200 and set(body) == {"error", "code"}, body
    return status, body


def _run(reader):
    """Run a printed reader as a caller would: its path through ``answer``, and its CLI form through the parser."""
    from ouroboros import cli

    parsed = urllib.parse.urlsplit(reader["path"])
    assert parsed.path == "/api/recall" and not parsed.scheme and not parsed.netloc
    args = cli.build_parser().parse_args(reader["cli"])
    assert args.func is cli._recall_command
    return urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)


def test_search_reads_every_admitted_room_in_the_callers_order_never_a2a(tmp_path):
    _say(tmp_path, "2026-11-01T10:00:00Z", "the archive decision, first said")
    _say(tmp_path, "2026-11-02T10:00:00Z", "archive decision restated in a project", chat_id=7)
    _say(tmp_path, "2026-11-03T10:00:00Z", "archive decision A2AMARK", chat_id=-3)
    newest = _search(tmp_path, "archive decision")
    oldest = _search(tmp_path, "archive decision", order="oldest")
    assert newest["total"] == 2 and newest["observed"] == 2 and newest["gaps"] == []
    assert [row["room_id"] for row in newest["rows"]] == ["7", "1"]
    assert [row["address"] for row in oldest["rows"]] == [row["address"] for row in reversed(newest["rows"])]
    assert "A2AMARK" not in json.dumps(newest)
    one_room = _search(tmp_path, "archive decision", room_id=7)
    assert [row["room_id"] for row in one_room["rows"]] == ["7"]
    first = oldest["rows"][0]
    assert first["address"].startswith("row:1@2026-11-01T10:00:00Z#") and len(first["address"].rsplit("#")[1]) == 64
    assert first["matched"] == ["text"] and first["fields"] == [
        {"name": "text", "speaker": "User", "text": "the archive decision, first said"}]


def test_a_row_reopens_whole_through_the_reader_its_search_line_printed(tmp_path):
    long_text = "archive decision " + "padding words " * 40 + "the very end"
    _say(tmp_path, "2026-11-01T10:00:00Z", long_text)
    hit = _search(tmp_path, "archive")["rows"][0]
    assert hit["fields"][0]["text"].endswith(f"[{len(long_text)} chars in full]")
    assert _run(hit["reopen"]) == [("row", hit["address"])] and hit["reopen"]["cli"] == ["recall", "--row", hit["address"]]
    reopened = _ok(tmp_path, *_run(hit["reopen"]))["row"]
    assert reopened["fields"] == [{"name": "text", "speaker": "User", "text": long_text}]
    assert reopened["address"] == hit["address"] and reopened["reopen"] == hit["reopen"]
    short = hit["address"][: hit["address"].index("#") + 13]  # the 12-hex form a mind's line prints
    assert _ok(tmp_path, ("row", short))["row"]["address"] == hit["address"]


def test_an_unadmitted_row_is_not_found_even_by_its_known_address(tmp_path):
    _say(tmp_path, "2026-11-01T10:00:00Z", "archive A2AMARK", chat_id=-3)
    _say(tmp_path, "2026-11-01T10:00:00Z", "archive admitted")
    synthetic = next(entry.row for entry in ds.iter_rows(tmp_path, "chat") if entry.row["chat_id"] == -3)
    admitted = _search(tmp_path, "archive")["rows"][0]["address"]
    assert _ok(tmp_path, ("row", admitted))["row"]["fields"][0]["text"] == "archive admitted"
    hidden = _refused(tmp_path, ("row", recall._full_address(synthetic)))
    missing = _refused(tmp_path, ("row", "row:1@2026-11-01T10:00:00Z#" + "0" * 64))
    assert hidden == missing == (404, {"error": recall._NOT_FOUND, "code": "recall_row_not_found"})


def test_no_protected_bytes_or_internal_readers_in_search_reopen_or_errors(tmp_path, monkeypatch):
    (tmp_path / "settings.json").write_text(json.dumps({"OPENROUTER_API_KEY": "SETTINGSMARK"}), encoding="utf-8")
    (tmp_path / "memory" / "knowledge").mkdir(parents=True)
    (tmp_path / "memory" / "knowledge" / "plan.md").write_text("archive plan NOTEMARK", encoding="utf-8")
    _say(tmp_path, "2026-11-01T10:00:00Z", "archive plan with a file",
         attachments=[{"name": "a.txt", "content": "ATTACHMARK", "upload_id": "u1"}],
         transport={"provider": "telegram", "raw": {"payload": "TRANSPORTRAWMARK"}})
    ds.append(tmp_path, "chat", {
        "ts": "2026-11-01T11:00:00Z", "type": "quiz_answer", "direction": "system", "chat_id": 1, "task_id": "t1",
        "text": "archive plan secret SECRETVALUEMARK",
        "quiz": {"quiz_id": "q1", "question": "archive plan SECRETQUESTIONMARK", "options": ["archive SECRETLABELMARK"],
                 "answered_index": 0, "comment": "archive SECRETVALUEMARK",
                 "secret_request": {"plugin_name": "telegram", "fields": [{"id": "bot_token", "label": "SECRETLABELMARK"}]}}})
    monkeypatch.setattr(dialogue_recall, "_seals", lambda root, wanted: ({key: "SEALPAGEMARK" for key in wanted}, ""))
    from ouroboros.memory import Memory

    mind = dialogue_recall.words_history(Memory(drive_root=tmp_path), dialogue_recall.parse(["archive plan"]),
                                         room=None, filters={})
    assert "SEALPAGEMARK" in mind and "memory_read(" in mind  # the mind's line names its seal and its own reader
    found = _search(tmp_path, "archive plan")
    assert found["total"] == 1 and found["rows"][0]["fields"][0]["text"] == "archive plan with a file"
    receipt = _search(tmp_path, "secret request")["rows"]
    assert [field["name"] for field in receipt[0]["fields"]] == ["receipt"]
    answers = [found, receipt, _ok(tmp_path, ("row", found["rows"][0]["address"])),
               _ok(tmp_path, ("row", receipt[0]["address"]))]
    stale = _refused(tmp_path, ("words", "archive"), ("snapshot", "f" * 64))
    assert stale[0] == 409
    answers += [stale, _refused(tmp_path, ("row", "bogus")), _refused(tmp_path, ("row", "row:1@x#" + "a" * 12)),
                _refused(tmp_path, ("words", "archive"), ("room_id", "Main")), _refused(tmp_path, ("note", "plan"))]
    text = json.dumps(answers, ensure_ascii=False)
    assert not [marker for marker in MARKERS if marker in text]
    assert not [word for word in ("memory_read", "node_id", "knowledge_read", "sealed", "chat_history") if word in text]


def test_every_printed_reader_is_the_doors_own_and_continues_the_same_capture(tmp_path):
    for minute in range(5):
        _say(tmp_path, f"2026-11-01T10:0{minute}:00Z", f"archive note {minute} with a #hash and «кавычки»")
    first = _search(tmp_path, "archive", "«кавычки»", count=2, order="oldest")
    assert first["more"] == 3 and first["snapshot"]
    following = _run(first["next"])
    assert ("words", "archive") in following and ("words", "«кавычки»") in following
    assert dict(following)["snapshot"] == first["snapshot"] and dict(following)["offset"] == "2"
    second = _ok(tmp_path, *following)
    assert [row["ts"] for row in second["rows"]] == ["2026-11-01T10:02:00Z", "2026-11-01T10:03:00Z"]
    _say(tmp_path, "2026-11-01T10:09:00Z", "archive later")
    assert _refused(tmp_path, *following) == (409, {"code": "recall_snapshot_changed", "error": (
        "the conversation or the query changed since this snapshot; no mixed page was returned: "
        "start again without snapshot and offset")})


@pytest.mark.parametrize("items, said", [
    ([], "give words"), ([("words", "a"), ("row", "row:1@x#" + "a" * 12)], "give words"),
    ([("row", "row:1@x#" + "a" * 12), ("count", "3")], "one row address alone"),
    ([("words", "a"), ("voice", "mine")], "voice is owner"), ([("words", "a"), ("order", "best")], "order is newest"),
    ([("words", "a"), ("count", "500")], "count is a whole number from 1 to 100"),
    ([("words", "a"), ("count", "1"), ("count", "2")], "given twice: count"),
    ([("words", '"open')], "not closed"), ([("words", "a"), ("date_from", "yesterday")], "ISO-8601"),
])
def test_unusable_requests_are_typed_400_and_read_nothing(tmp_path, items, said):
    status, body = _refused(tmp_path, *items)
    assert status == 400 and body["code"] == "recall_bad_request" and said in body["error"]


def _events(root):
    path = root / "logs" / "supervisor.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == "dialogue_recall"]


def _call(root, query: str):
    import asyncio
    from types import SimpleNamespace

    from starlette.datastructures import QueryParams
    from ouroboros.server_auth import RECALL_PEER

    request = SimpleNamespace(scope={RECALL_PEER: True}, query_params=QueryParams(query))
    return asyncio.run(recall.make_recall_endpoint(root)(request))


def test_each_admitted_call_is_counted_once_without_content_and_a_failed_count_never_fails_it(tmp_path, monkeypatch):
    _say(tmp_path, "2026-11-01T10:00:00Z", "archive words")
    address = _search(tmp_path, "archive")["rows"][0]["address"]
    assert _events(tmp_path) == []  # ``answer`` alone counts nothing: the endpoint counts
    statuses = [_call(tmp_path, query).status_code for query in (
        "words=archive", "row=" + urllib.parse.quote(address), "words=archive&row=x", "row=" + "x" * 3)]
    assert statuses == [200, 200, 400, 400]
    events = _events(tmp_path)
    assert [(event["mode"], event["outcome"]) for event in events] == [
        ("search", "ok"), ("row", "ok"), ("invalid", "recall_bad_request"), ("row", "recall_bad_request")]
    assert all(set(event) == {"ts", "type", "mode", "outcome"} for event in events)
    assert "archive" not in json.dumps(events) and address not in json.dumps(events)
    import ouroboros.utils

    def broken(*_args, **_kwargs):
        raise OSError("log volume gone")

    monkeypatch.setattr(ouroboros.utils, "append_jsonl", broken)
    response = _call(tmp_path, "words=archive")
    assert response.status_code == 200 and json.loads(response.body)["total"] == 1
    assert response.headers["cache-control"] == "no-store"


def test_a_request_without_the_raw_peer_admission_is_closed_before_any_read(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from ouroboros.server_auth import RECALL_CLOSED

    def touched(*_args, **_kwargs):
        raise AssertionError("the door read before admission")

    monkeypatch.setattr(recall, "answer", touched)
    monkeypatch.setattr(recall, "_count", touched)
    for scope in ({}, {"ouroboros.recall_peer": False}, {"ouroboros.recall_peer": "yes"}):
        request = SimpleNamespace(scope=scope, query_params=None)
        response = asyncio.run(recall.make_recall_endpoint(tmp_path)(request))
        assert response.status_code == 403 and json.loads(response.body) == RECALL_CLOSED
    assert not (tmp_path / "logs").exists()


_NO_TURN = r"""
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
from ouroboros import dialogue_store as ds
ds.append(root, "chat", {"ts": "2026-11-01T10:00:00Z", "direction": "in", "chat_id": 1, "text": "archive decision"})
rows = len(list(ds.iter_rows(root, "chat")))
from ouroboros.gateway import recall
status, body, _ = recall.answer(root, [("words", "archive")])
again, _, _ = recall.answer(root, [("row", body["rows"][0]["address"])])
print(json.dumps({"statuses": [status, again], "rows": [rows, len(list(ds.iter_rows(root, "chat")))],
                  "modules": sorted(m for m in sys.modules if m.startswith(("supervisor", "ouroboros.loop",
                                    "ouroboros.llm", "ouroboros.tools", "ouroboros.agent", "ouroboros.owner_continue")))}))
"""


def test_the_door_starts_no_model_turn_dispatches_nothing_and_writes_no_row(tmp_path):
    """In a fresh interpreter a search and a reopen import no turn, model, tool, queue or supervisor module (code
    that is not loaded cannot run) and append no dialogue row."""
    env = {**os.environ, "OUROBOROS_DATA_DIR": str(tmp_path), "HOME": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": str(REPO)}
    done = subprocess.run([sys.executable, "-c", _NO_TURN, str(tmp_path)], env=env, cwd=REPO, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    facts = json.loads(done.stdout.strip().splitlines()[-1])
    assert facts == {"statuses": [200, 200], "rows": [1, 1], "modules": []}
