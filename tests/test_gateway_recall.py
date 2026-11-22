"""``GET /api/recall``: the neighbour door over the conversation (``ouroboros/gateway/recall.py``).

The door projects the words reader (``dialogue_recall.search``) outward and reopens one row through the shared row
resolver: dialogue rows of the rooms a person reads (Main, Projects, places) and nothing else, each a closed ``RecallRow``; every printed reader is
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
    from ouroboros.chat_chain import row_address

    address = row_address(synthetic)
    hidden = _refused(tmp_path, ("row", f"row:{address['chat_id']}@{address['ts']}#{address['row_sha256']}"))
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


def _answered_card(root, ts="2026-11-01T12:00:00Z"):
    ds.append(root, "chat", {
        "ts": ts, "type": "quiz_answer", "direction": "system", "chat_id": 1, "task_id": "asker-1",
        "text": "[answer q] chose (2) Archive now",
        "quiz": {"quiz_id": "q-card", "question": "When do we archive?", "options": ["Later", "Archive now"],
                 "answered_index": 1, "comment": "archive it now"}})


def test_outward_field_speakers_name_the_owner_never_my_human(tmp_path):
    _answered_card(tmp_path)
    from ouroboros.memory import Memory

    mind = dialogue_recall.words_history(Memory(drive_root=tmp_path), dialogue_recall.parse(["archive it now"]),
                                         room=None, filters={})
    assert "comment (my human): archive it now" in mind  # the mind's own reader speaks in the first person
    row = _search(tmp_path, "archive it now")["rows"][0]
    reopened = _ok(tmp_path, ("row", row["address"]))["row"]
    for view in (row, reopened):
        speakers = {field["name"]: field["speaker"] for field in view["fields"]}
        assert speakers == {"question": "task asker-1", "option 1": "task asker-1", "option 2": "task asker-1",
                            "press": "owner", "comment": "owner"}
    assert "my human" not in json.dumps([row, reopened])


def test_no_runnable_next_while_the_capture_is_unstable_and_dash_values_round_trip(tmp_path, monkeypatch):
    from ouroboros import cli

    for minute in range(3):
        _say(tmp_path, f"2026-11-01T10:0{minute}:00Z", f"-rf flag {minute}")
    stable = _search(tmp_path, "-rf", count=1, room_id=1)
    assert "--words=-rf" in stable["next"]["cli"] and stable["notes"] == []
    assert cli.build_parser().parse_args(stable["next"]["cli"]).words == ["-rf"]
    with pytest.raises(SystemExit):  # the plain two-argument form would be read as an option
        cli.build_parser().parse_args(["recall", "--words", "-rf"])
    capture = dialogue_recall.search

    def unstable(*args, **kwargs):
        return {**capture(*args, **kwargs), "snapshot": None}  # the reader's answer when the capture is unstable

    monkeypatch.setattr(dialogue_recall, "search", unstable)
    shaky = _search(tmp_path, "-rf", count=1)
    assert shaky["next"] is None and shaky["snapshot"] is None and shaky["more"] == 2 and shaky["observed"] == 3
    assert shaky["notes"] == [recall.UNSTABLE_READ]
    assert recall.UNSTABLE_READ == "The conversation changed during this read. Start again without snapshot or offset."


@pytest.mark.parametrize("items", [
    [("words", "a"), ("count", "9" * 5000)], [("words", "a"), ("offset", "1" * 25)],
    [("words", "a"), ("room_id", str(2**70))], [("words", "a"), ("room_id", str(-2**63 - 1))],
    [("row", f"row:{2**70}@2026-11-01T10:00:00Z#" + "a" * 12)],
])
def test_every_bad_number_is_a_typed_400_never_an_escaping_exception(tmp_path, items):
    _say(tmp_path, "2026-11-01T10:00:00Z", "a")
    status, body = _refused(tmp_path, *items)
    assert status == 400 and body["code"] == "recall_bad_request"


def test_the_largest_room_id_the_store_holds_is_still_a_search(tmp_path):
    assert _search(tmp_path, "a", room_id=2**63 - 1)["total"] == 0


def test_a_row_from_before_the_lineage_epoch_reopens_with_the_author_its_search_names(tmp_path):
    from ouroboros.tools.chronicle import _memory_read
    from ouroboros.tools.registry import ToolContext

    _say(tmp_path, "2026-10-01T00:00:01+00:00", "Review the patch.")
    _say(tmp_path, "2026-10-01T00:00:02+00:00", "Child review: the patch is unsafe.", direction="out",
         task_id="kid00001")
    _memory_read(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root0001", current_chat_id=1),
                 rows=True)  # activates: the lineage epoch stands after these rows
    found = _search(tmp_path, "unsafe")["rows"][0]
    reopened = _ok(tmp_path, *_run(found["reopen"]))["row"]
    assert "outgoing, author not recorded" in found["speaker"]
    assert reopened["speaker"] == found["speaker"] and reopened["fields"] == found["fields"]


def test_unreadable_owner_accounts_are_noted_on_search_and_reopen(tmp_path, monkeypatch):
    _say(tmp_path, "2026-11-01T10:00:00Z", "archive words")
    address = _search(tmp_path, "archive")["rows"][0]["address"]
    assert _search(tmp_path, "archive")["notes"] == [] and _ok(tmp_path, ("row", address))["notes"] == []
    import ouroboros.places

    def broken(_root):
        raise OSError("places registry unreadable")

    monkeypatch.setattr(ouroboros.places, "place_owner_accounts", broken)
    unread = "owner accounts unreadable: no place row is labelled as the owner's"
    assert _search(tmp_path, "archive")["notes"] == [unread] and _ok(tmp_path, ("row", address))["notes"] == [unread]


def test_the_door_serves_the_rooms_not_hidden_work_and_the_mind_still_reads_everything(tmp_path):
    """The world of ``tests/test_chat_history_source_recovery.py``: Main, a Project's own row, a Main task bound to
    the Project, hidden work (chat 0) alone and bound, and synthetic A2A. The door answers what a room shows."""
    from ouroboros.memory import Memory
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "room", name="Room")
    bind_task_to_project(tmp_path, "bound", project["id"], origin={"absent": "system"})
    bind_task_to_project(tmp_path, "hidden-bound", project["id"], origin={"absent": "system"})
    for minute, (text, chat, task) in enumerate([
            ("archive main", 1, ""), ("archive direct", project["chat_id"], ""), ("archive bound", 1, "bound"),
            ("archive hidden", 0, "hidden"), ("archive hidden bound", 0, "hidden-bound"), ("archive a2a", -100, "")]):
        _say(tmp_path, f"2026-11-01T10:0{minute}:00Z", text, chat_id=chat, direction="out",
             **({"task_id": task} if task else {}))
    found = _search(tmp_path, "archive", order="oldest")
    texts = [row["fields"][0]["text"] for row in found["rows"]]
    assert texts == ["archive main", "archive direct", "archive bound", "archive hidden bound"]
    assert found["total"] == 4 and found["observed"] == 4
    by_text = {row["fields"][0]["text"]: row for row in found["rows"]}
    assert by_text["archive bound"]["rooms"] == [str(project["chat_id"])]  # it shows in the Project now
    assert by_text["archive hidden bound"]["room_id"] == str(project["chat_id"])
    assert all("0" not in row["rooms"] for row in found["rows"])
    for row in found["rows"]:
        assert _ok(tmp_path, *_run(row["reopen"]))["row"]["address"] == row["address"]
    hidden = next(entry.row for entry in ds.iter_rows(tmp_path, "chat") if entry.row["text"] == "archive hidden")
    from ouroboros.chat_chain import row_address

    address = row_address(hidden)
    full = f"row:{address['chat_id']}@{address['ts']}#{address['row_sha256']}"
    assert _refused(tmp_path, ("row", full)) == _refused(tmp_path, ("row", "row:0@2026-11-01T10:03:00Z#" + "0" * 64))
    mind = dialogue_recall.words_history(Memory(drive_root=tmp_path), dialogue_recall.parse(["archive hidden"]),
                                         room=None, filters={})
    assert "archive hidden" in mind and "total=2" in mind  # the mind's own reader keeps every row


def test_voice_owner_at_the_door_finds_a_card_the_owner_answered_by_its_question(tmp_path):
    _answered_card(tmp_path)
    _say(tmp_path, "2026-11-01T12:01:00Z", "When do we archive? asked by Ouroboros", direction="out")
    owner = _search(tmp_path, "when do we archive", voice="owner")
    assert [row["matched"] for row in owner["rows"]] == [["question"]]
    assert _search(tmp_path, "when do we archive")["total"] == 2


def test_a_malformed_snapshot_is_a_typed_400_and_an_unstable_one_a_409_with_its_sentence(tmp_path, monkeypatch):
    for minute in range(3):
        _say(tmp_path, f"2026-11-01T10:0{minute}:00Z", f"archive {minute}")
    first = _search(tmp_path, "archive", count=1)
    token = first["snapshot"]
    for bad in ("nothex", token + ":x", token + ":5", token[:-1], "   "):  # a suffix or spaces are never a snapshot
        assert _refused(tmp_path, ("words", "archive"), ("snapshot", bad)) == (400, {
            "code": "recall_bad_snapshot",
            "error": "Invalid snapshot. Use the snapshot from a search response or start again without it."})
    assert _ok(tmp_path, ("words", "archive"), ("snapshot", token), ("offset", "1"))["rows"]
    revisions = iter([1, 2])
    monkeypatch.setattr(dialogue_recall, "_text_revision", lambda _root: next(revisions, 2))
    assert _refused(tmp_path, ("words", "archive"), ("snapshot", token), ("offset", "1")) == (409, {
        "code": "recall_snapshot_changed", "error": recall.UNSTABLE_READ})


def test_the_door_never_shows_the_time_or_address_of_a_row_it_does_not_serve(tmp_path, monkeypatch):
    from ouroboros import dialogue_word_index as dwi

    _say(tmp_path, "2026-11-01T10:00:00Z", "archive visible")
    _say(tmp_path, "2026-11-02T22:22:22Z", "archive hidden", chat_id=0)
    dwi.run(tmp_path, lambda: False)
    for bound in (50_000, 1):  # through the postings, and handed to the scan past the bound
        monkeypatch.setattr(dialogue_recall, "RECHECK_BOUND", bound)
        for order in ("newest", "oldest"):
            found = _search(tmp_path, "archive", order=order)
            assert "22:22:22" not in json.dumps(found) and "archive hidden" not in json.dumps(found)
            assert [row["ts"] for row in found["rows"]] == ["2026-11-01T10:00:00Z"] and found["total"] == 1


def test_a_projects_founding_message_is_found_and_reopened_in_both_its_rooms(tmp_path):
    from tests import _memory_inventory_shared as shared

    rooms = shared.world(tmp_path, activate=False)
    alpha = str(rooms["alpha"])
    found = _search(tmp_path, "inventory work")
    row, = found["rows"]
    assert set(row["rooms"]) == {"1", alpha}
    assert _search(tmp_path, "inventory work", room_id=alpha)["rows"][0]["address"] == row["address"]
    reopened = _ok(tmp_path, *_run(row["reopen"]))
    assert set(reopened["row"]["rooms"]) == {"1", alpha}


def _bound_world(root):
    """A Project founded by an owner message whose address names only its room and checksum, and a task bound to it
    that spoke in the hidden partition: both rows are the Project's by the room rule's sources alone."""
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import create_project

    project = create_project(root, "alpha", name="Alpha")
    founding = {"ts": "2026-11-01T09:00:00Z", "direction": "in", "chat_id": 1, "client_message_id": "",
                "text": "archive founding words"}
    ds.append(root, "chat", founding)
    ref = build_owner_message_ref(chat_id=1, client_message_id="", ts="", text=founding["text"])
    (root / "state").mkdir(exist_ok=True)
    (root / "state" / "project_task_bindings.json").write_text(json.dumps({"bindings": {
        "bound": {"project_id": project["id"], "project_chat_id": project["chat_id"], "chat_id": project["chat_id"],
                  "source_ref": ref, "source_text": founding["text"]}}}), encoding="utf-8")
    _say(root, "2026-11-01T10:00:00Z", "archive bound work", chat_id=0, direction="out", task_id="bound")
    _say(root, "2026-11-01T11:00:00Z", "archive loner reply", direction="out", task_id="loner")  # its room needs lineage
    return project


def _break(root, monkeypatch, source):
    from ouroboros import project_dialogue, projects_registry

    def broken(*_args, **_kwargs):
        raise OSError(f"{source} unreadable")

    if source == "reservations":
        real = projects_registry.list_reserved_projects

        def strict_broken(root, *args, strict=False, **kwargs):
            if strict:
                raise OSError("reservations unreadable")
            return real(root, *args, strict=strict, **kwargs)

        monkeypatch.setattr(projects_registry, "list_reserved_projects", strict_broken)
    elif source == "bindings":
        (root / "state" / "project_task_bindings.json").write_text("{not json", encoding="utf-8")
    elif source == "lineage":
        monkeypatch.setattr(projects_registry, "_read_lineage", lambda *_a, **_kw: None)
    else:
        monkeypatch.setattr(project_dialogue, "physical_origin_refs", broken)


@pytest.mark.parametrize("source, words", [
    ("reservations", "archive bound"), ("bindings", "archive bound"), ("lineage", "archive loner"),
    ("founding", "archive founding"),
])
def test_every_unreadable_source_of_the_room_rule_is_one_gap_and_an_undecidable_reopen(tmp_path, monkeypatch,
                                                                                         source, words):
    project = _bound_world(tmp_path)
    readable = _search(tmp_path, words)
    row, = readable["rows"]
    assert readable["total"] == 1 and (source == "lineage") is (str(project["chat_id"]) not in row["rooms"])
    assert _ok(tmp_path, *_run(row["reopen"]))["row"]["address"] == row["address"]
    _break(tmp_path, monkeypatch, source)
    shaky = _search(tmp_path, words)
    assert shaky["total"] is None and sum("rooms_unreadable" in gap for gap in shaky["gaps"]) == 1, shaky
    assert not any("rooms_unreadable" in note for note in shaky["notes"])
    status, body = _refused(tmp_path, *_run(row["reopen"]))
    assert (status, body["code"]) == (503, "recall_source_unreadable")


def test_the_door_tells_an_import_still_running_from_an_observed_edit(tmp_path, monkeypatch):
    for minute in range(3):
        _say(tmp_path, f"2026-11-01T10:0{minute}:00Z", f"archive {minute}")
    token = _search(tmp_path, "archive", count=1)["snapshot"]
    assert recall.IMPORT_INCOMPLETE == ("History import is incomplete. Start again without snapshot or offset after "
                                        "it completes.")
    real = ds.range_coverage
    monkeypatch.setattr(ds, "range_coverage", lambda *a, **kw: {**real(*a, **kw), "imported": False, "pending": 1})
    importing = _search(tmp_path, "archive", count=1)
    assert importing["snapshot"] is None and importing["notes"] == [recall.IMPORT_INCOMPLETE]
    assert _refused(tmp_path, ("words", "archive"), ("snapshot", token), ("offset", "1")) == (409, {
        "code": "recall_snapshot_changed", "error": recall.IMPORT_INCOMPLETE})
    assert recall.IMPORT_FAILED == ("A history segment could not be imported. Retry without snapshot or offset after "
                                    "the import is repaired.")
    monkeypatch.setattr(ds, "range_coverage",
                        lambda *a, **kw: {**real(*a, **kw), "imported": False, "pending": 0, "mismatch": 1})
    failed = _search(tmp_path, "archive", count=1)
    assert failed["snapshot"] is None and failed["notes"] == [recall.IMPORT_FAILED]
    assert _refused(tmp_path, ("words", "archive"), ("snapshot", token), ("offset", "1")) == (409, {
        "code": "recall_snapshot_changed", "error": recall.IMPORT_FAILED})
    monkeypatch.setattr(ds, "range_coverage", real)
    revisions = iter([1, 2])
    monkeypatch.setattr(dialogue_recall, "_text_revision", lambda _root: next(revisions, 2))
    edited = _search(tmp_path, "archive", count=1)
    assert edited["snapshot"] is None and edited["notes"] == [recall.UNSTABLE_READ]



@pytest.mark.parametrize("damage", ["registry removed beside its witness", "registry malformed", "bindings malformed"])
def test_a_read_scoped_to_a_room_never_certifies_an_absence_it_could_not_check(tmp_path, damage):
    from ouroboros import room_appearances
    from ouroboros.tools.control_runtime import _chat_history
    from types import SimpleNamespace

    project = _bound_world(tmp_path)
    room_appearances.run(tmp_path, lambda: False)
    alpha = str(project["chat_id"])
    healthy = _search(tmp_path, "archive bound", room_id=alpha)
    assert healthy["total"] == 1 and len(healthy["rows"]) == 1
    registry = tmp_path / "state" / "projects.json"
    assert registry.exists() and registry.with_name("projects.json.committed").exists()
    if damage == "registry removed beside its witness":
        registry.unlink()
    elif damage == "registry malformed":
        registry.write_text("{not json", encoding="utf-8")
    else:
        (tmp_path / "state" / "project_task_bindings.json").write_text("{not json", encoding="utf-8")
    door = _search(tmp_path, "archive bound", room_id=alpha)  # typed: never an escaping exception
    assert door["total"] is None and door["rows"] == [] and sum("rooms_unreadable" in gap for gap in door["gaps"]) == 1
    mind = str(_chat_history(SimpleNamespace(drive_root=tmp_path, task_metadata={}, pending_events=[], task_id="t"),
                             words=["archive bound"], room_id=alpha))
    assert "observed=0; total_unknown" in mind and "rooms_unreadable" in mind and "total=0" not in mind



def test_one_strict_registry_read_however_many_bound_tasks_the_hits_hold(tmp_path, monkeypatch):
    from ouroboros import projects_registry
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "alpha", name="Alpha")
    reads = {}
    for many in (2, 8):
        bindings = {f"task{n}": {"project_id": project["id"], "project_chat_id": project["chat_id"],
                                 "chat_id": project["chat_id"]} for n in range(many)}
        (tmp_path / "state" / "project_task_bindings.json").write_text(json.dumps({"bindings": bindings}),
                                                                         encoding="utf-8")
        for n in range(many):
            _say(tmp_path, f"2026-11-0{many}T10:0{n}:00Z", f"archive{many} bound {n}", chat_id=0, direction="out",
                 task_id=f"task{n}")
        real, counted = projects_registry._load, [0]

        def load(root, *args, strict=False, **kwargs):
            counted[0] += bool(strict)
            return real(root, *args, strict=strict, **kwargs)

        monkeypatch.setattr(projects_registry, "_load", load)
        found = _search(tmp_path, f"archive{many}", count=100)
        monkeypatch.setattr(projects_registry, "_load", real)
        assert found["total"] == many
        reads[many] = counted[0]
    assert reads[8] == reads[2], reads  # never one registry read per bound task
