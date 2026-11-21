"""``chat_history(words=…)``: the words reader over the conversation (``ouroboros/dialogue_recall.py``, K4)."""
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
    ds.append(root, "chat", {"ts": ts, "direction": direction, "chat_id": chat_id, "text": text, **extra})


def _card(root, ts, *, question, options, index, comment, task="asker-1", **quiz):
    ds.append(root, "chat", {
        "ts": ts, "type": "quiz_answer", "direction": "system", "chat_id": 1, "task_id": task,
        "text": f"[answer q] chose ({index + 1}) {options[index]}",
        "quiz": {"quiz_id": "q-" + ts[-9:-1], "question": question, "options": options, "answered_index": index,
                 "comment": comment, **quiz}})


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


def test_a_candidate_set_past_the_recheck_bound_returns_the_examined_range(tmp_path, monkeypatch):
    for minute in range(4):
        _say(tmp_path, f"2026-11-01T10:0{minute}:00Z", f"match {minute}")
    assert "total=4" in _read(tmp_path, words=["match"])
    monkeypatch.setattr(dialogue_recall, "RECHECK_BOUND", 4)  # met exactly, nothing left unexamined
    assert "total=4" in _read(tmp_path, words=["match"])
    monkeypatch.setattr(dialogue_recall, "RECHECK_BOUND", 2)
    answer = _read(tmp_path, words=["match"])
    assert "observed=2; total_unknown" in answer
    assert "recheck_bound: 2 candidate rows re-checked in stored order, through row:1@2026-11-01T10:01:00Z" in answer
    assert "match 1" in answer and "match 3" not in answer  # an answer, never a block


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
    assert "Seals: page seals not read: the chronicle index is behind its journal" in behind
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
    projected = project_function_tools_to_openai_custom([{"type": "function", "function": entry.schema}]).wire_tools()
    description = projected[0]["custom"]["description"]
    lead = " ".join(entry.schema["description"].split())
    assert len(description) <= MAX_CUSTOM_DESCRIPTION_CHARS
    assert description.startswith(lead + "\n" + _SCHEMA_CLAUSE + json.dumps(
        entry.schema["parameters"], ensure_ascii=False, sort_keys=True, separators=(",", ":")))
