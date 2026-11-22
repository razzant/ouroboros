"""``chat_history(words=…)``: the words reader over the conversation (``ouroboros/dialogue_recall.py``)."""
from __future__ import annotations

import json
import re
import sqlite3
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_recall, dialogue_store as ds
from ouroboros.memory import Memory
from ouroboros.tools.control_runtime import _chat_history


def _ctx(root):
    return SimpleNamespace(drive_root=root, task_metadata={}, pending_events=[], task_id="words-test")


def _read(root, **kwargs) -> str:
    return str(_chat_history(_ctx(root), **kwargs))


def _say(root, ts, text, *, direction="in", chat_id=1, **extra):
    return ds.append(root, "chat", {"ts": ts, "direction": direction, "chat_id": chat_id, "text": text, **extra})


def _card(root, ts, *, question, options, index, comment, task="asker-1", **quiz):
    ds.append(root, "chat", {
        "ts": ts, "type": "quiz_answer", "direction": "system", "chat_id": 1, "task_id": task,
        "text": f"[answer q] chose ({index + 1}) {options[index]}",
        "quiz": {"quiz_id": "q-" + ts[-9:-1], "question": question, "options": options, "answered_index": index,
                 "comment": comment, **quiz}})


def _lines(text: str) -> str:
    return text.split("\n\n", 1)[1] if "\n\n" in text else ""


def _addresses(text: str):
    return re.findall(r"^(row:\S+)", text, flags=re.M)


def test_words_are_prefixes_in_any_order_within_one_field(tmp_path):
    _say(tmp_path, "2026-11-01T10:00:00Z", "Решили: таймаут сокета 30 секунд")
    _say(tmp_path, "2026-11-01T11:00:00Z", "only the socket here")
    _card(tmp_path, "2026-11-01T12:00:00Z", question="Socket fix now?", options=["Wait", "Go"], index=1,
          comment="timeout after the exam")
    answer = _read(tmp_path, words=["сокет тайм"])
    assert "total=1" in answer and "Решили: таймаут сокета" in answer
    # "socket" lives in the card's question, "timeout" in its comment: one field never joins another
    assert "total=0" in _read(tmp_path, words=["socket timeout"])
    assert "total=1" in _read(tmp_path, words=["timeout exam"])


def test_words_and_search_exclude_each_other_and_search_keeps_its_contract(tmp_path):
    _say(tmp_path, "2026-11-01T10:00:00Z", "match one")
    refused = _read(tmp_path, words=["match"], search="match")
    assert refused.startswith("⚠️ TOOL_ARG_ERROR (chat_history): words and search exclude each other")
    legacy = _read(tmp_path, search="MATCH")
    assert legacy == Memory(tmp_path).chat_history(search="MATCH")
    assert legacy.startswith("Showing 1 of 1 messages")
    assert "total=1" in _read(tmp_path, words=["match"])


@pytest.mark.parametrize("given", [{"voice": "owner"}, {"order": "oldest"}, {"room_id": "1"}])
def test_words_parameters_never_reach_the_legacy_reader(tmp_path, given):
    _say(tmp_path, "2026-11-01T10:00:00Z", "match one")
    assert "goes with words" in _read(tmp_path, **given)
    assert "TOOL_ARG_ERROR" not in _read(tmp_path, words=["match"], **given)


@pytest.mark.parametrize("given, said", [
    ({"words": []}, "words is empty"), ({"words": ['"open']}, "not closed"),
    ({"words": ["x"], "voice": "mine"}, "voice is owner"), ({"words": ["x"], "order": "best"}, "order is newest"),
    ({"words": ["x"], "room_id": "Main"}, "not a room address"),
    ({"words": ["x"], "date_from": "yesterday"}, "date_from must be an ISO-8601"),
])
def test_unusable_words_calls_are_argument_errors(tmp_path, given, said):
    _say(tmp_path, "2026-11-01T10:00:00Z", "x")
    answer = _read(tmp_path, **given)
    assert answer.startswith("⚠️ TOOL_ARG_ERROR (chat_history)") and said in answer and "Nothing was read." in answer


def test_order_is_the_callers_direction_of_time(tmp_path):
    for day, text in ((1, "decision original"), (2, "decision restated"), (3, "decision restated again")):
        _say(tmp_path, f"2026-11-0{day}T10:00:00Z", text)
    newest, oldest = _read(tmp_path, words=["decision"]), _read(tmp_path, words=["decision"], order="oldest")
    assert "newest first" in newest and "oldest first" in oldest
    assert _addresses(newest) == list(reversed(_addresses(oldest)))
    assert "2026-11-01" in _addresses(oldest)[0] and "2026-11-03" in _addresses(newest)[0]


def test_voice_owner_selects_my_humans_rows_and_matches_every_field_of_them(tmp_path):
    _say(tmp_path, "2026-11-01T10:00:00Z", "merge plan from me")
    _say(tmp_path, "2026-11-01T10:01:00Z", "merge plan from Ouroboros", direction="out")
    _card(tmp_path, "2026-11-01T10:02:00Z", question="Merge plan B?", options=["Yes", "No"], index=0, comment="")
    # a card asked but never answered: Ouroboros's row, not my human's
    ds.append(tmp_path, "chat", {"ts": "2026-11-01T10:03:00Z", "type": "quiz", "direction": "out", "chat_id": 1,
                                 "text": "Merge plan C?", "quiz": {"quiz_id": "qc", "question": "Merge plan C?",
                                                                   "options": ["Yes"]}})
    assert "total=4" in _read(tmp_path, words=["merge plan"])
    mine = _read(tmp_path, words=["merge plan"], voice="owner")
    assert "total=2" in mine and "merge plan from me" in mine
    assert "from Ouroboros" not in mine and "Merge plan C" not in mine
    # the answered card is found by its question's words, and the line says whose words they are
    assert "matched question" in mine
    assert "question (task asker-1): Merge plan B?" in mine and "press (my human): (1) Yes" in mine


def test_the_api_door_stamp_is_named_and_an_unstamped_row_claims_nothing(tmp_path):
    _say(tmp_path, "2026-11-01T10:00:00Z", "stamped words", client_surface={"channel": "api_command"})
    _say(tmp_path, "2026-11-01T10:01:00Z", "plain words")
    stamped = _read(tmp_path, words=["stamped"])
    plain = _read(tmp_path, words=["plain"])
    assert dialogue_recall.API_DOOR_NOTE in stamped
    assert dialogue_recall.API_DOOR_NOTE not in plain and "API" not in plain.split("\n\n", 1)[1]


def test_total_is_exact_on_a_complete_set_and_unknown_with_a_named_gap(tmp_path):
    complete = tmp_path / "complete"
    _say(complete, "2026-11-01T10:00:00Z", "match valid")
    answer = _read(complete, words=["match"])
    assert "total=1;" in answer and "total_unknown" not in answer and "Gaps:" not in answer
    gapped = tmp_path / "gapped"
    path = gapped / "logs" / "chat.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"ts":"2026-08-21T09:00:00Z","direction":"in","text":"match valid"}\n'
                     b'{"direction":"in","text":"match hidden"\n')
    answer = _read(gapped, words=["match"])
    assert "observed=1; total_unknown" in answer and "total=" not in answer
    assert "Gaps: jsonl_malformed" in answer


def test_the_scan_rechecks_every_row_so_a_default_call_starts_at_the_newest(tmp_path, monkeypatch):
    """No postings here: the bound of the postings path does not apply, the scan reads every row as the
    legacy search does, so the newest match leads even when the candidates outnumber the bound."""
    for minute in range(4):
        _say(tmp_path, f"2026-11-01T10:0{minute}:00Z", f"match {minute}")
    monkeypatch.setattr(dialogue_recall, "RECHECK_BOUND", 2)
    answer = _read(tmp_path, words=["match"], count=1)
    assert "total=4" in answer and "recheck_bound" not in answer and "Read by scan of every row" in answer
    assert "match 3" in answer and "match 0" not in answer
    assert "match 0" in _read(tmp_path, words=["match"], count=1, order="oldest")


def test_order_follows_recorded_time_not_storage(tmp_path):
    _say(tmp_path, "2026-11-05T10:00:00Z", "match later said")  # stored first
    _say(tmp_path, "2026-11-01T10:00:00Z", "match earlier said")  # a backdated arrival stored after it
    _say(tmp_path, "", "match without a time")
    newest = _lines(_read(tmp_path, words=["match"]))
    assert newest.index("later said") < newest.index("earlier said") < newest.index("without a time")
    oldest = _lines(_read(tmp_path, words=["match"], order="oldest"))
    assert oldest.index("without a time") < oldest.index("earlier said") < oldest.index("later said")


def test_a_source_row_is_counted_once_across_fields_and_copies(tmp_path):
    row = {"ts": "2026-11-01T10:00:00Z", "direction": "in", "chat_id": 1, "text": "twice stored socket"}
    ds.append(tmp_path, "chat", dict(row))
    ds.append(tmp_path, "chat", dict(row))  # a crash-retry copy of the same source row
    _card(tmp_path, "2026-11-01T11:00:00Z", question="socket now?", options=["socket fix"], index=0,
          comment="socket yes")
    answer = _read(tmp_path, words=["socket"])
    assert "total=2" in answer and len(_addresses(answer)) == 2
    assert "matched question, option 1, comment" in answer


def test_a_secret_request_card_shows_only_its_receipt(tmp_path):
    _card(tmp_path, "2026-11-01T10:00:00Z", question="Enter the bot token", options=["x"], index=0, comment="",
          secret_request={"plugin_name": "tg", "fields": [{"id": "botToken", "setting_key": "TG_SECRET_KEY"}]},
          secret_outcome={"event": "secret_filled", "pluginName": "tg", "fields": ["botToken"]})
    answer = _read(tmp_path, words=["botToken"])
    assert "receipt (host): secret request" in answer and "secret_filled: tg, botToken" in answer
    assert "Enter the bot token" not in answer and "TG_SECRET_KEY" not in answer
    assert "total=0" in _read(tmp_path, words=["bot token"])  # its question is not a field


def test_words_pages_keep_their_snapshot_and_refuse_a_shifted_page(tmp_path):
    for minute in range(3):
        _say(tmp_path, f"2026-11-01T10:0{minute}:00Z", f"match {minute}")
    first = _read(tmp_path, words=["match"], count=2)
    snapshot = re.search(r"snapshot=([0-9a-f]{64})", first).group(1)
    second = _read(tmp_path, words=["match"], count=2, offset=2, snapshot=snapshot)
    assert "match 0" in second and "match 2" not in second
    assert _read(tmp_path, words=["other"], offset=2, snapshot=snapshot).startswith("CHAT_HISTORY_SNAPSHOT_CHANGED")
    _say(tmp_path, "2026-11-01T10:09:00Z", "match 9")
    assert _read(tmp_path, words=["match"], offset=2, snapshot=snapshot).startswith("CHAT_HISTORY_SNAPSHOT_CHANGED")


def test_room_id_reads_one_room(tmp_path):
    _say(tmp_path, "2026-11-01T10:00:00Z", "match in Main")
    _say(tmp_path, "2026-11-01T10:01:00Z", "match in seven", chat_id=7)
    answer = _read(tmp_path, words=["match"], room_id="7")
    assert "in room 7" in answer and "total=1" in answer and "match in seven" in answer
    assert "total=2" in _read(tmp_path, words=["match"])


def _chronicle_index(root, *, journal: bytes = b"", seals=()):
    from ouroboros import chronicle_store

    store = chronicle_store.ChronicleStore(root)
    store.directory.mkdir(parents=True, exist_ok=True)
    store.log_path.write_bytes(journal)
    db = sqlite3.connect(store.index_path)
    db.executescript(chronicle_store._SCHEMA)
    db.execute(f"PRAGMA user_version={chronicle_store.SCHEMA_VERSION}")
    db.execute("INSERT INTO state VALUES ('offset', '0')")
    db.executemany("INSERT INTO page_rows VALUES (?,?,?)", seals)
    db.commit()
    db.close()
    return store


def test_the_acting_seal_is_read_passively_with_its_own_address(tmp_path):
    from ouroboros.chat_chain import source_row_id

    _say(tmp_path, "2026-11-01T10:00:00Z", "sealed match")
    _say(tmp_path, "2026-11-01T10:01:00Z", "open match")
    sealed = next(entry.row for entry in ds.iter_rows(tmp_path, "chat") if entry.row["text"] == "sealed match")
    store = _chronicle_index(tmp_path, seals=[("1", "page-abc", source_row_id(sealed))])
    answer = _read(tmp_path, words=["match"])
    lines = {line.split(" · ")[0]: line for line in answer.splitlines() if line.startswith("row:")}
    assert sum("sealed by page page-abc in room 1: memory_read(node_id=\"page-abc\")" in line for line in lines.values()) == 1
    assert "Seals:" not in answer
    # behind its journal, the index is not caught up by this read: the gap is named and the file is untouched
    store.log_path.write_bytes(b'{"kind":"page"}\n')
    before = store.index_path.read_bytes()
    behind = _read(tmp_path, words=["match"])
    assert "Seals: page seals were not read on this path" in behind and "memory_read of the row's room" in behind
    assert "sealed by page" not in behind and store.index_path.read_bytes() == before


def test_without_a_chronicle_the_answer_says_so_and_creates_nothing(tmp_path):
    _say(tmp_path, "2026-11-01T10:00:00Z", "match")
    answer = _read(tmp_path, words=["match"])
    assert "Seals: chronicle not activated" in answer
    assert not (tmp_path / "memory" / "chronicle").exists()


def test_chat_history_words_projects_whole_on_the_custom_route():
    from ouroboros.openai_chat_custom import (MAX_CUSTOM_DESCRIPTION_CHARS, _SCHEMA_CLAUSE,
                                              project_function_tools_to_openai_custom)
    from ouroboros.tools.control import get_tools

    entry = next(entry for entry in get_tools() if entry.name == "chat_history")
    for name in ("words", "room_id", "voice", "order"):
        assert entry.schema["parameters"]["properties"][name]["description"]
    assert entry.schema["parameters"]["properties"]["order"]["description"] == (
        "With words: newest (default) or oldest first; oldest shows an original before its later restatements.")
    projected = project_function_tools_to_openai_custom([{"type": "function", "function": entry.schema}]).wire_tools()
    description = projected[0]["custom"]["description"]
    lead = " ".join(entry.schema["description"].split())
    assert len(description) <= MAX_CUSTOM_DESCRIPTION_CHARS
    assert description.startswith(lead + "\n" + _SCHEMA_CLAUSE + json.dumps(
        entry.schema["parameters"], ensure_ascii=False, sort_keys=True, separators=(",", ":")))


# ---- the reader's shape: showing rooms, authors, snapshot, the structured result -------------------------

def _follow(root, line):
    """The printed reader of a hit's line, called as printed."""
    from ouroboros.tools.chronicle import _memory_read
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=root, drive_root=root, task_id="reader0001", current_chat_id=1)
    rows = re.search(r'read: memory_read\(rows=true, room_id="(-?\d+)", from="([^"]+)", to="([^"]+)"\)', line)
    if rows:
        return _memory_read(ctx, rows=True, room_id=rows[1], **{"from": rows[2], "to": rows[3]})
    source = re.search(r"read: memory_read\(source_ref=(\{.*\})\)", line)
    return _memory_read(ctx, source_ref=json.loads(source[1]))


@pytest.mark.parametrize("words, text, room", [
    (["hello back"], "hello back", "1"),  # an ordinary row of Main
    (["bound work"], "bound work", "alpha"),  # Main speech of a task later bound to a Project: shown there only
    (["hidden note"], "hidden note", "0"),  # the hidden partition
])
def test_every_printed_reader_returns_its_row(tmp_path, words, text, room):
    from tests import _memory_inventory_shared as shared

    rooms = shared.world(tmp_path, activate=False)
    answer = _read(tmp_path, words=words)
    assert re.search(r"(total|observed)=1;", answer), answer  # the world holds a torn line: observed
    line = next(line for line in answer.splitlines() if line.startswith("row:"))
    expected = str(rooms.get(room, room))
    assert f'room_id="{expected}"' in line
    assert text in _follow(tmp_path, line)


def test_a_row_stored_without_a_chat_id_reads_and_seals_in_main(tmp_path):
    from ouroboros.chat_chain import source_row_id

    ds.append(tmp_path, "chat", {"ts": "2026-11-01T10:00:00Z", "direction": "in", "chat_id": None, "text": "nullroom match"})
    row = next(entry.row for entry in ds.iter_rows(tmp_path, "chat"))
    _chronicle_index(tmp_path, seals=[("1", "page-null", source_row_id(row))])
    answer = _read(tmp_path, words=["nullroom"])
    line = next(line for line in answer.splitlines() if line.startswith("row:"))
    assert 'room_id="None"' not in line and 'room_id="1"' in line
    assert 'sealed by page page-null in room 1' in line


def test_an_in_place_edit_between_pages_refuses_the_shifted_page(tmp_path):
    receipts = [_say(tmp_path, f"2026-11-01T10:0{n}:00Z", f"match {n}") for n in range(3)]
    first = _read(tmp_path, words=["match"], count=1)
    token = re.search(r"snapshot=([0-9a-f]{64})", first).group(1)
    assert "match 1" in _read(tmp_path, words=["match"], count=1, offset=1, snapshot=token)  # unchanged: pages
    ds.revise_text(tmp_path, "chat", receipts[2].seq, "no longer here")
    shifted = _read(tmp_path, words=["match"], count=1, offset=1, snapshot=token)
    assert shifted.startswith("CHAT_HISTORY_SNAPSHOT_CHANGED") and "edited row" in shifted


def test_a_row_from_before_the_lineage_epoch_is_named_as_the_row_reader_names_it(tmp_path):
    from ouroboros.tools.chronicle import _memory_read
    from ouroboros.tools.registry import ToolContext

    for row in ({"chat_id": 1, "direction": "in", "ts": "2026-10-01T00:00:01+00:00", "text": "Review the patch."},
                {"chat_id": 1, "direction": "out", "ts": "2026-10-01T00:00:02+00:00", "task_id": "kid00001",
                 "text": "Child review: the patch is unsafe."}):
        ds.append(tmp_path, "chat", row)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root0001", current_chat_id=1)
    rows = _memory_read(ctx, rows=True)  # activates: the lineage epoch stands after these rows
    assert "outgoing, author not recorded" in rows
    answer = _read(tmp_path, words=["unsafe"])
    assert "outgoing, author not recorded" in answer and "· Ouroboros" not in answer


def test_the_line_names_a_reply_target_and_the_room_it_was_read_in(tmp_path):
    target = _say(tmp_path, "2026-11-01T10:00:00Z", "the original")
    from ouroboros.chat_chain import format_address, row_address

    address = format_address(row_address(target.row))
    _say(tmp_path, "2026-11-01T10:01:00Z", "a reply match", reply_to={"address": address, "kind": "message",
                                                                     "snippet": "the original"})
    _say(tmp_path, "2026-11-01T10:02:00Z", "plain match")
    answer = _read(tmp_path, words=["match"])
    reply_line = next(line for line in answer.splitlines() if line.startswith("row:") and "10:01:00Z" in line)
    plain_line = next(line for line in answer.splitlines() if line.startswith("row:") and "10:02:00Z" in line)
    assert f"replies to {address}" in reply_line and "replies to" not in plain_line
    seven = _say(tmp_path, "2026-11-01T10:03:00Z", "seven match", chat_id=7)
    in_seven = _read(tmp_path, words=["seven"], room_id="7")
    assert "in room 7" in in_seven and 'room_id="7"' in in_seven and format_address(row_address(seven.row)) in in_seven


def test_unreadable_owner_accounts_are_said_and_counted_only_under_voice(tmp_path, monkeypatch):
    from ouroboros import places

    _say(tmp_path, "2026-11-01T10:00:00Z", "match")
    assert "owner accounts unreadable" not in _read(tmp_path, words=["match"])

    def broken(_root):
        raise places.PlaceError("registry unreadable")

    monkeypatch.setattr(places, "place_owner_accounts", broken)
    plain = _read(tmp_path, words=["match"])
    assert "Note: owner accounts unreadable" in plain and "total=1" in plain
    owner = _read(tmp_path, words=["match"], voice="owner")
    assert "owner_accounts_unreadable" in owner and "total_unknown" in owner and "my human's rows only" in owner


def test_whitespace_search_is_still_a_search_and_excludes_words(tmp_path):
    _say(tmp_path, "2026-11-01T10:00:00Z", "two  spaces")
    assert "exclude each other" in _read(tmp_path, words=["two"], search="  ")
    assert _read(tmp_path, search="  ").startswith("Showing 1 of 1 messages")


def test_two_rows_sharing_a_short_address_are_counted_apart(tmp_path, monkeypatch):
    from ouroboros import chat_chain

    for side in ("left match", "right match"):
        _say(tmp_path, "2026-11-01T10:00:00Z", side)
    real = chat_chain.source_row_id
    monkeypatch.setattr(chat_chain, "source_row_id", lambda row: "abcdef012345" + real(row)[12:])
    assert "total=2" in _read(tmp_path, words=["match"])


def test_the_seal_lookup_chunks_its_variables(tmp_path):
    _say(tmp_path, "2026-11-01T10:00:00Z", "match")
    _chronicle_index(tmp_path, seals=[("1", "page-x", "f" * 64)])
    found, gap = dialogue_recall._seals(tmp_path, [(1, f"{n:064x}") for n in range(40_000)] + [(1, "f" * 64)])
    assert gap == "" and found == {(1, "f" * 64): "page-x"}


def test_a_store_error_reads_like_the_old_readers_sentence(tmp_path, monkeypatch):
    _say(tmp_path, "2026-11-01T10:00:00Z", "match")

    def broken(*_args, **_kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(ds, "stream_end", broken)
    assert _read(tmp_path, words=["match"]) == "(error reading history: disk I/O error)"


def test_the_structured_result_is_what_the_text_renders(tmp_path):
    from ouroboros.dialogue_words import parse
    from ouroboros.memory import Memory

    _card(tmp_path, "2026-11-01T10:00:00Z", question="Merge plan B?", options=["Yes", "No"], index=0, comment="fine")
    found = dialogue_recall.search(Memory(tmp_path), parse(["merge plan"]), room=None, voice="owner", filters={})
    hit, = found["hits"]
    assert found["total"] == 1 and found["refused"] is None and found["snapshot"]
    assert hit["rooms"] == [1] and hit["reader_room"] == 1 and hit["reader"].startswith("memory_read(rows=true")
    assert len(hit["sha256"]) == 64 and hit["matched"] == ["question"]
    speakers = {field["name"]: (field["speaker"], field["label"]) for field in hit["fields"]}
    assert speakers["question"] == ("asker", "task asker-1") and speakers["press"] == ("owner", "my human")


def test_a_projects_founding_message_lists_both_its_rooms(tmp_path):
    from tests import _memory_inventory_shared as shared

    rooms = shared.world(tmp_path, activate=False)
    alpha = rooms["alpha"]
    for scope in ({}, {"room_id": str(alpha)}):
        answer = _read(tmp_path, words=["inventory work"], **scope)
        line = next(line for line in answer.splitlines() if line.startswith("row:"))
        assert "Main" in line and f"[chat_id={alpha}]" in line, line
        assert shared.ORIGIN in _follow(tmp_path, line)


def test_a_read_that_overlaps_an_edit_claims_no_exact_count_and_no_continuation(tmp_path, monkeypatch):
    for n in range(3):
        _say(tmp_path, f"2026-11-01T10:0{n}:00Z", f"match {n}")
    stable = _read(tmp_path, words=["match"], count=1)
    token = re.search(r"snapshot=([0-9a-f]{64})", stable).group(1)
    revisions = iter([0, 1])
    monkeypatch.setattr(dialogue_recall, "_text_revision", lambda _root: next(revisions, 1))
    moved = _read(tmp_path, words=["match"], count=1)
    assert "observed=3; total_unknown" in moved and "capture_unstable" in moved
    assert "Snapshot unavailable" in moved and "Continue with" not in moved
    assert _read(tmp_path, words=["match"], count=1, offset=1, snapshot=token).startswith("CHAT_HISTORY_SNAPSHOT_CHANGED")


@pytest.mark.parametrize("bad", ["nothex", "f" * 63, "f" * 64 + ":x", "f" * 64 + ":0", "{valid}:3"])
def test_every_malformed_continuation_is_a_typed_refusal(tmp_path, bad):
    _say(tmp_path, "2026-11-01T10:00:00Z", "match")
    valid = re.search(r"snapshot=([0-9a-f]{64})", _read(tmp_path, words=["match"])).group(1)
    answer = _read(tmp_path, words=["match"], snapshot=bad.format(valid=valid))
    assert answer.startswith("CHAT_HISTORY_SNAPSHOT_INVALID") or answer.startswith("CHAT_HISTORY_SNAPSHOT_CHANGED")
    if bad.startswith("{valid}"):  # a window suffix on a scan, which issues none
        assert answer.startswith("CHAT_HISTORY_SNAPSHOT_INVALID")
