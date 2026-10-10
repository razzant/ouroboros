"""The chat-generation chain: moved helpers, index-free row addresses and the row stream.

The behavior cases pin the outputs the helpers produced in ``consolidator`` on the
same inputs (archive ordering, cursor-segment resolution in every branch, the
A2A-free row stream, signatures, immutable source retention). The structural
cases pin the move itself: one definition in ``chat_chain``, the legacy writer
binding the same objects, and ``retain_memory_source`` reached as a module
attribute so a single substitution reaches every caller. The address cases pin
``{chat_id, ts, row_sha256}`` resolution (hint, then the rotation-time search)
and its typed refusals; the stream cases pin positions to the legacy cursor and
room rows to ``dialogue_evidence.read_room_source``.
"""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import shutil
from types import SimpleNamespace

import pytest

from ouroboros import chat_chain as cc
from ouroboros.utils import jsonl_generation_signature

REPO = pathlib.Path(__file__).resolve().parents[1]
MOVED = ("retain_memory_source", "_ordered_chat_generation_paths", "_resolve_generation_segments",
         "_chat_log_signature", "_read_chat_entries")


def _layout(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    logs, archive = tmp_path / "logs", tmp_path / "archive"
    logs.mkdir()
    archive.mkdir()
    return logs / "chat.jsonl", archive


def _rows(tag: str, count: int = 2) -> str:
    return "".join(json.dumps({"chat_id": 1, "direction": "in", "text": f"{tag}-{i}"}) + "\n" for i in range(count))


def _first_sha(path: pathlib.Path) -> str:
    return jsonl_generation_signature(path)["first_line_sha256"]


# --- behavior on fixed inputs ------------------------------------------------------------------

def test_ordered_chain_is_sorted_archives_then_live(tmp_path):
    live, archive = _layout(tmp_path)
    for name in ("chat_20260902T000000.jsonl", "chat_20260901T000000_1.jsonl", "chat_20260901T000000.jsonl"):
        (archive / name).write_text(_rows(name), encoding="utf-8")
    (archive / "chat_20260903T000000.txt").write_text("x", encoding="utf-8")
    (archive / "other_20260901.jsonl").write_text("x", encoding="utf-8")
    chain = cc._ordered_chat_generation_paths(live)
    assert [p.name for p in chain] == ["chat_20260901T000000.jsonl", "chat_20260901T000000_1.jsonl",
                                       "chat_20260902T000000.jsonl", "chat.jsonl"]
    assert chain[-1] == live  # the live file closes the chain even before it exists


def test_ordered_chain_without_archive_dir_is_live_only(tmp_path):
    live = tmp_path / "logs" / "chat.jsonl"
    assert cc._ordered_chat_generation_paths(live) == [live]


def test_uninitialized_cursor_takes_every_archive_but_legacy_offset_stays_live_only(tmp_path):
    live, archive = _layout(tmp_path)
    live.write_text(_rows("live"), encoding="utf-8")
    assert cc._resolve_generation_segments({}, live) == ([live], 0, False)
    old = archive / "chat_20260901T000000.jsonl"
    old.write_text(_rows("old"), encoding="utf-8")
    assert cc._resolve_generation_segments({}, live) == ([old, live], 0, False)
    # A nonzero offset without a signature is the pre-signature legacy shape: live only.
    assert cc._resolve_generation_segments({"last_consolidated_offset": 7}, live) == ([live], 7, False)
    # A signature that is not a mapping counts as absent.
    assert cc._resolve_generation_segments({"chat_log_signature": "junk"}, live) == ([old, live], 0, False)


def test_stored_signature_locates_its_generation_or_reports_a_gap(tmp_path):
    live, archive = _layout(tmp_path)
    gens = []
    for stamp in ("20260901T000000", "20260902T000000", "20260903T000000"):
        path = archive / f"chat_{stamp}.jsonl"
        path.write_text(_rows(stamp), encoding="utf-8")
        gens.append(path)
    live.write_text(_rows("live"), encoding="utf-8")

    def meta(path, offset=5):
        return {"last_consolidated_offset": offset, "chat_log_signature": {"first_line_sha256": _first_sha(path)}}

    assert cc._resolve_generation_segments(meta(live), live) == ([live], 5, False)
    assert cc._resolve_generation_segments(meta(gens[1]), live) == ([gens[1], gens[2], live], 5, False)
    assert cc._resolve_generation_segments(meta(gens[0], 0), live) == ([*gens, live], 0, False)
    missing = {"last_consolidated_offset": 5, "chat_log_signature": {"first_line_sha256": "f" * 64}}
    assert cc._resolve_generation_segments(missing, live) == ([live], 0, True)


def test_row_stream_skips_blank_broken_and_a2a_rows_only(tmp_path):
    live, _archive = _layout(tmp_path)
    assert cc._read_chat_entries(live) == []
    rows = [{"chat_id": 1, "text": "main"}, {"chat_id": 1001, "text": "project"},
            {"chat_id": -1, "text": "a2a"}, {"chat_id": "-3999", "text": "a2a-text-id"},
            {"text": "no chat id"}, {"chat_id": "main", "text": "non-numeric id"}]
    body = "\n".join(json.dumps(row) for row in rows[:3]) + "\n\n   \n{broken json\n" + \
        "\n".join(json.dumps(row) for row in rows[3:]) + "\n"
    live.write_text(body, encoding="utf-8")
    assert [row["text"] for row in cc._read_chat_entries(live)] == [
        "main", "project", "no chat id", "non-numeric id"]


def test_generation_signature_is_the_shared_utils_signature(tmp_path):
    live, _archive = _layout(tmp_path)
    assert cc._chat_log_signature is jsonl_generation_signature
    assert cc._chat_log_signature(live) == {}
    live.write_text("\n  \n" + _rows("first", 1) + _rows("second", 1), encoding="utf-8")
    first = json.dumps({"chat_id": 1, "direction": "in", "text": "first-0"})
    assert cc._chat_log_signature(live) == {
        "first_line_sha256": hashlib.sha256(first.encode("utf-8")).hexdigest(), "size": live.stat().st_size}


def test_retained_source_is_exact_and_readable_after_the_task(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes

    data = "exact bytes ✓\n".encode("utf-8")
    ref = cc.retain_memory_source(SimpleNamespace(drive_root=tmp_path, task_id="task-1"), "probe", data, "jsonl")
    assert ref["task_id"] == "task-1" and ref["canonical_root"] == str(tmp_path.resolve())
    assert ref["path"].endswith(".jsonl")
    arguments = ref["read"]["arguments"]
    assert ref["read"]["tool"] == "read_file" and arguments["root"] == "runtime_data"
    assert arguments["start_line"] == 1
    assert (tmp_path.resolve() / arguments["path"]).read_bytes() == data
    assert read_actor_source_bytes(tmp_path, "task-1", ref) == data
    default = cc.retain_memory_source(SimpleNamespace(drive_root=tmp_path, task_id=None), "probe", b"x")
    assert default["task_id"] == "consolidation" and default["path"].endswith(".md")


# --- the move itself ---------------------------------------------------------------------------

def _module_ast(relative: str) -> ast.Module:
    return ast.parse((REPO / relative).read_text(encoding="utf-8"))


def _top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.ImportFrom) and node.module == "ouroboros.utils":
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def test_moved_helpers_are_defined_once_in_chat_chain():
    assert set(MOVED) <= _top_level_names(_module_ast("ouroboros/chat_chain.py"))
    assert not set(MOVED) & _top_level_names(_module_ast("ouroboros/consolidator.py"))


def test_consolidator_keeps_no_chain_helper_after_the_writer_is_retired():
    """The old dialogue writer bound three chain helpers; it is gone, so consolidator binds none
    of them (no facade left behind) and reaches the chain only as the ``chat_chain`` module."""
    import ouroboros.consolidator as cons

    imported = [node for node in _module_ast("ouroboros/consolidator.py").body
                if isinstance(node, ast.ImportFrom) and node.module == "ouroboros.chat_chain"]
    assert imported == []
    for name in ("_chat_log_signature", "_read_chat_entries", "_resolve_generation_segments",
                 "retain_memory_source", "_ordered_chat_generation_paths"):
        assert not hasattr(cons, name) and hasattr(cc, name)
    assert cons.chat_chain is cc


def _retain_bindings(relative: str) -> tuple[list[int], list[int], int]:
    """(module-level imports of the name, bare-name calls, ``chat_chain.retain_memory_source`` calls)."""
    tree = _module_ast(relative)
    top = [node.lineno for node in tree.body if isinstance(node, ast.ImportFrom)
           and any(alias.name == "retain_memory_source" for alias in node.names)]
    bare = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "retain_memory_source"]
    attribute = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "retain_memory_source"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "chat_chain")
    return top, bare, attribute


def test_retain_memory_source_is_reached_as_a_module_attribute_everywhere():
    """A module-level ``from ... import retain_memory_source`` freezes a copy that a
    substitution of ``chat_chain.retain_memory_source`` would miss; function-level
    imports and attribute calls read the module at call time."""
    top, bare, attribute = _retain_bindings("ouroboros/consolidator.py")
    assert top == [] and bare == [] and attribute > 0
    importers = []
    for path in sorted((REPO / "ouroboros").rglob("*.py")):
        relative = path.relative_to(REPO).as_posix()
        text = path.read_text(encoding="utf-8")
        if relative == "ouroboros/chat_chain.py" or "retain_memory_source" not in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(a.name == "retain_memory_source" for a in node.names):
                importers.append((relative, node.module))
        top, _bare, _attr = _retain_bindings(relative)
        assert top == [], f"{relative} binds retain_memory_source at module level: lines {top}"
    assert importers and all(module == "ouroboros.chat_chain" for _path, module in importers), importers



# --- index-free row addresses ------------------------------------------------------------------

def _append(path: pathlib.Path, *rows) -> None:
    """Append rows (dicts) or raw physical lines (strings, newline included)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row if isinstance(row, str) else json.dumps(row, ensure_ascii=False) + "\n")


def _msg(ts: str, text: str, chat_id: int = 1, **extra) -> dict:
    return {"ts": ts, "chat_id": chat_id, "direction": "in", "text": text, **extra}


def _ok(root, address):
    row, found = cc.resolve_row(root, address)
    assert found["status"] == "ok", found
    return row, found


def _chain(tmp_path: pathlib.Path) -> pathlib.Path:
    """Two archives and the live file, with blank, broken and A2A lines between rows."""
    live, archive = _layout(tmp_path)
    _append(archive / "chat_20260901T000000.jsonl", _msg("2026-08-31T10:00:00+00:00", "first"), "\n",
            _msg("2026-08-31T11:00:00+00:00", "a2a", chat_id=-7), "{torn\n",
            _msg("2026-08-31T12:00:00+00:00", "second", chat_id=1001, task_id="t1"))
    _append(archive / "chat_20260902T000000.jsonl", _msg("2026-09-01T10:00:00+00:00", "third"),
            _msg("2026-09-01T11:00:00+00:00", "fourth", task_id="t2"), "   \n",
            _msg("2026-09-01T12:00:00+00:00", "fifth"))
    _append(live, _msg("2026-09-02T10:00:00+00:00", "sixth"), _msg("2026-09-02T11:00:00+00:00", "a2a-2", chat_id=-1),
            _msg("2026-09-02T12:00:00+00:00", "seventh", client_message_id="c7"))
    return live


def test_source_row_id_hashes_the_raw_row_not_its_projection(tmp_path):
    from ouroboros.dialogue_evidence import _row_projection

    raw = _msg("2026-09-01T00:00:00+00:00", "Привет", task_id="t")
    canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert cc.source_row_id(raw) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert cc.source_row_id(dict(reversed(list(raw.items())))) == cc.source_row_id(raw)
    assert cc.source_row_id(_row_projection(raw, "chat", 1)) != cc.source_row_id(raw)
    live, _archive = _layout(tmp_path)
    _append(live, json.dumps(dict(reversed(list(raw.items()))), ensure_ascii=True) + "\n")
    [(address, row, _pos)] = list(cc.iter_rows(tmp_path))
    assert row == raw and address["row_sha256"] == cc.source_row_id(raw)


def test_text_form_round_trips_and_malformed_addresses_are_refused():
    address = cc.row_address(_msg("2026-10-01T20:17:46.902747+00:00", "x", task_id="t", type="quiz"), gen="g", line=3)
    assert address["hint"] == {"task_id": "t", "type": "quiz", "gen": "g", "line": 3}
    text = cc.format_address(address)
    assert text == f"row:1@2026-10-01T20:17:46.902747+00:00#{address['row_sha256'][:12]}"
    parsed = cc.parse_address(text)
    assert (parsed["chat_id"], parsed["ts"], parsed["row_sha256"], parsed["hint"]) == (
        1, "2026-10-01T20:17:46.902747+00:00", address["row_sha256"][:12], {})
    for bad in ("row:x@2026#abcdefabcdef", "row:1@2026#abcdef", "1@2026#abcdefabcdef", "row:1@2026#ABCDEFABCDEZ"):
        with pytest.raises(ValueError):
            cc.parse_address(bad)
    with pytest.raises(ValueError):
        cc.resolve_row(".", {"chat_id": 1, "ts": "x"})


def test_every_row_resolves_with_its_hint_and_by_search_alone(tmp_path):
    _chain(tmp_path)
    streamed = list(cc.iter_rows(tmp_path))
    assert [row["text"] for _a, row, _p in streamed] == [
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
    for address, row, _pos in streamed:
        hinted, found = _ok(tmp_path, address)
        assert hinted == row and found["address"] == address
        for bare in (cc.format_address(address), {**address, "hint": {}}):
            searched, again = _ok(tmp_path, bare)
            assert searched == row and (again["path"], again["line"]) == (found["path"], found["line"])


def test_a_wrong_hint_is_ignored_and_the_search_decides(tmp_path):
    _chain(tmp_path)
    streamed = list(cc.iter_rows(tmp_path))
    address, row, _pos = streamed[3]
    other = streamed[2][0]["hint"]
    for hint in ({**address["hint"], "line": other["line"]}, {**address["hint"], "gen": "f" * 64},
                 {**address["hint"], "line": 999}, {"gen": other["gen"], "line": other["line"]}):
        found_row, found = _ok(tmp_path, {**address, "hint": hint})
        assert found_row == row and found["line"] == address["hint"]["line"]


def test_the_hint_is_used_and_a_row_stamped_during_rotation_resolves_from_the_earlier_archive(tmp_path):
    live, archive = _layout(tmp_path)
    # Rotation names the archive from the clock, truncated to the second, BEFORE it waits for
    # the append lock: a row appended during that wait carries a later second than its archive.
    _append(archive / "chat_20261001T120000.jsonl", _msg("2026-10-01T12:00:00.900000+00:00", "same second"),
            _msg("2026-10-01T12:00:01.200000+00:00", "appended while the rotation waited"))
    _append(archive / "chat_20261001T130000.jsonl", _msg("2026-10-01T12:30:00+00:00", "next archive"))
    _append(live, _msg("2026-10-02T00:00:00+00:00", "live"))
    (same, _row, _pos), (late, late_row, late_pos), _next, _live = list(cc.iter_rows(tmp_path))
    _ok(tmp_path, cc.format_address(same))
    assert cc.resolve_row(tmp_path, late)[0] == late_row  # the hint
    text = cc.format_address(late)  # every address a tool prints is this text form, without a hint
    found_row, found = _ok(tmp_path, text)
    assert found_row == late_row and found["path"] == "archive/chat_20261001T120000.jsonl" and found["line"] == 2
    assert [row["text"] for _a, row, _p in cc.iter_rows(tmp_path, from_addr=text)][:2] == [
        "appended while the rotation waited", "next archive"]
    assert cc.stream_position_of(tmp_path, text) == late_pos == 1
    # The wider search still answers a typed refusal for a row no generation holds.
    never = cc.row_address(_msg("2026-10-01T12:00:01.300000+00:00", "never written"))
    assert cc.resolve_row(tmp_path, cc.format_address(never)) == (None, {"status": "row_missing"})
    forged = cc.row_address({**late_row, "text": "never said"})
    assert cc.resolve_row(tmp_path, cc.format_address(forged))[1]["status"] == "row_mismatch"


def test_address_survives_rotation_and_a_copied_data_directory(tmp_path):
    from supervisor.state import rotate_jsonl_log_if_needed

    root = tmp_path / "data"
    live = root / "logs" / "chat.jsonl"
    _append(live, _msg("2026-09-01T10:00:00.250000+00:00", "before rotation", client_message_id="c1"),
            _msg("2026-09-01T10:00:01+00:00", "also before"))
    (address, row, pos), _second = list(cc.iter_rows(root))
    rotate_jsonl_log_if_needed(root, "chat.jsonl", "chat", max_bytes=1)
    assert live.read_text(encoding="utf-8") == "" and len(list((root / "archive").glob("chat_*.jsonl"))) == 1
    _append(live, _msg("2026-09-02T00:00:00+00:00", "after rotation"))
    copy = tmp_path / "elsewhere" / "restored"
    shutil.copytree(root, copy)
    for where in (root, copy):
        for form in (address, cc.format_address(address), {**address, "hint": {}}):
            found_row, found = _ok(where, form)
            assert found_row == row and found["path"].startswith("archive/chat_") and found["line"] == 1
        assert cc.stream_position_of(where, cc.format_address(address)) == pos == 0


def test_rows_sharing_chat_ts_client_id_and_task_differ_by_hash(tmp_path):
    live, _archive = _layout(tmp_path)
    twin = {"client_message_id": "c", "task_id": "t"}
    _append(live, _msg("2026-09-10T17:10:14+00:00", "one", **twin), _msg("2026-09-10T17:10:14+00:00", "two", **twin))
    (first, first_row, _p1), (second, second_row, _p2) = list(cc.iter_rows(tmp_path))
    assert first["row_sha256"] != second["row_sha256"]
    assert {k: first[k] for k in ("chat_id", "ts")} == {k: second[k] for k in ("chat_id", "ts")}
    assert first["hint"]["client_message_id"] == second["hint"]["client_message_id"] == "c"
    assert _ok(tmp_path, cc.format_address(first))[0] == first_row
    assert _ok(tmp_path, cc.format_address(second))[0] == second_row


def test_substituted_row_is_a_mismatch_and_an_absent_row_is_missing(tmp_path):
    live, _archive = _layout(tmp_path)
    _append(live, _msg("2026-09-01T00:00:00+00:00", "what was said"), _msg("2026-09-01T00:00:01+00:00", "next"))
    (address, _row, _pos), _next = list(cc.iter_rows(tmp_path))
    _ok(tmp_path, address)
    live.write_text(live.read_text(encoding="utf-8").replace("what was said", "what was never said"), encoding="utf-8")
    for form in (address, cc.format_address(address)):
        row, found = cc.resolve_row(tmp_path, form)
        assert row is None and found["status"] == "row_mismatch"
        assert found["candidates"][0]["row_sha256"] != address["row_sha256"]
    absent = cc.row_address(_msg("2026-09-01T00:00:02+00:00", "never written"))
    assert cc.resolve_row(tmp_path, absent) == (None, {"status": "row_missing"})
    assert cc.stream_position_of(tmp_path, absent) is None


def test_a_shared_twelve_hex_prefix_is_ambiguous_and_names_both_full_forms(tmp_path, monkeypatch):
    live, _archive = _layout(tmp_path)
    ts = "2026-09-01T00:00:00+00:00"
    _append(live, _msg(ts, "left"), _msg(ts, "right"))
    plain = [address for address, _row, _pos in cc.iter_rows(tmp_path)]
    _ok(tmp_path, cc.format_address(plain[0]))
    real = cc.source_row_id
    monkeypatch.setattr(cc, "source_row_id", lambda row: "abcdef012345" + real(row)[12:])
    collided = [address for address, _row, _pos in cc.iter_rows(tmp_path)]
    text = cc.format_address(collided[0])
    assert text == cc.format_address(collided[1])
    row, found = cc.resolve_row(tmp_path, text)
    assert row is None and found["status"] == "row_ambiguous"
    assert sorted(c["row_sha256"] for c in found["candidates"]) == sorted(a["row_sha256"] for a in collided)
    assert all(len(c["row_sha256"]) == 64 for c in found["candidates"])
    assert _ok(tmp_path, {**collided[1], "hint": {}})[0]["text"] == "right"


def test_a_torn_line_holding_the_timestamp_is_unreadable_not_missing(tmp_path):
    live, _archive = _layout(tmp_path)
    ts = "2026-09-01T00:00:05+00:00"
    address = cc.row_address(_msg(ts, "lost words"))
    _append(live, _msg("2026-09-01T00:00:00+00:00", "intact"), '{"ts": "%s", "chat_id": 1, "text": "lost wo\n' % ts)
    row, found = cc.resolve_row(tmp_path, address)
    assert row is None and found == {"status": "row_unreadable", "unreadable": [{"path": "logs/chat.jsonl", "line": 2}]}
    live.write_text(json.dumps(_msg("2026-09-01T00:00:00+00:00", "intact")) + "\n", encoding="utf-8")
    assert cc.resolve_row(tmp_path, address) == (None, {"status": "row_missing"})


def test_chat_chain_keeps_no_index_and_no_physical_locator():
    source = (REPO / "ouroboros/chat_chain.py").read_text(encoding="utf-8")
    for banned in ("sqlite3", "st_ino", "st_birthtime", "st_dev"):
        assert banned not in source


# --- the row stream ----------------------------------------------------------------------------

def test_stream_positions_are_the_legacy_cursor_positions(tmp_path):
    live = _chain(tmp_path)
    paths = cc._ordered_chat_generation_paths(live)
    entries = [entry for path in paths for entry in cc._read_chat_entries(path)]
    streamed = list(cc.iter_rows(tmp_path))
    assert [row for _a, row, _p in streamed] == entries
    assert [pos for _a, _r, pos in streamed] == list(range(len(entries)))
    # The legacy cursor names a generation by its first line plus an offset into
    # the rows from there on; its absolute position is the rows before plus offset.
    for generation in range(len(paths)):
        for offset in range(len(cc._read_chat_entries(paths[generation]))):
            meta = {"chat_log_signature": jsonl_generation_signature(paths[generation]),
                    "last_consolidated_offset": offset}
            segments, cursor, gap = cc._resolve_generation_segments(meta, live)
            pending = [entry for path in segments for entry in cc._read_chat_entries(path)]
            absolute = len(entries) - len(pending) + cursor
            assert not gap and streamed[absolute][1] == pending[cursor]
    for address, _row, pos in streamed:
        assert cc.stream_position_of(tmp_path, address) == pos
        assert cc.stream_position_of(tmp_path, cc.format_address(address)) == pos


def test_rows_start_at_the_addressed_row_and_an_unresolved_bound_is_typed(tmp_path):
    _chain(tmp_path)
    streamed = list(cc.iter_rows(tmp_path))
    tail = list(cc.iter_rows(tmp_path, from_addr=cc.format_address(streamed[3][0])))
    assert tail == streamed[3:]
    absent = cc.row_address(_msg("2026-09-02T10:30:00+00:00", "never written"))
    with pytest.raises(cc.RowAddressError) as refused:
        cc.iter_rows(tmp_path, from_addr=absent)
    assert refused.value.resolution == {"status": "row_missing"}
    with pytest.raises(cc.RowAddressError):
        cc.iter_room_rows(tmp_path, "1", to_addr=absent)


def _room_fixture(root: pathlib.Path) -> int:
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(root, "room", name="Room")
    chat = project["chat_id"]
    origin_ts = "2026-09-01T00:00:00+00:00"
    ref = build_owner_message_ref(chat_id=1, client_message_id="origin", ts=origin_ts, text="Start the room")
    bind_task_to_project(root, "bound", project["id"], origin={"ref": ref, "text": "Start the room"})
    _append(root / "archive" / "chat_20260902T000000.jsonl",
            _msg(origin_ts, "Start the room", client_message_id="origin"),
            _msg("2026-09-01T00:01:00+00:00", "Bound work", task_id="bound", direction="out"),
            _msg("2026-09-01T00:02:00+00:00", "Child of bound", task_id="kid", parent_task_id="bound",
                 root_task_id="bound", direction="out"),
            _msg("2026-09-01T00:03:00+00:00", "Agent traffic", chat_id=-5), "{torn\n")
    _append(root / "logs" / "chat.jsonl",
            _msg("2026-09-02T00:00:00+00:00", "Main question", client_message_id="m1", task_id="t1"),
            _msg("2026-09-02T00:01:00+00:00", "Main answer", task_id="t1", direction="out"),
            _msg("2026-09-02T00:02:00+00:00", "Sub answer", task_id="t2", parent_task_id="t1",
                 root_task_id="t1", direction="out"),
            _msg("2026-09-03T00:00:00+00:00", "Hidden partition", chat_id=0, direction="out"),
            _msg("2026-09-03T00:01:00+00:00", "Other transport", chat_id=777),
            _msg("2026-09-03T00:02:00+00:00", "Project message", chat_id=chat, client_message_id="p1"),
            _msg("2026-09-03T00:03:00+00:00", "Room started", type="project_started", direction="system"),
            {"chat_id": 1, "direction": "out", "text": "No timestamp"})
    return chat


def _projected(rows, root) -> set:
    from ouroboros.dialogue_evidence import _row_projection

    out = set()
    for row in rows:
        view = row if "stream" in row else _row_projection(row, "chat", 0, root)
        out.add(json.dumps({k: v for k, v in view.items() if k not in {"stream", "source_ordinal"}},
                           sort_keys=True, ensure_ascii=False))
    return out


def test_room_rows_are_the_room_evidence_rows_by_room_task_and_period(tmp_path):
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.dialogue_evidence import read_room_source

    project_chat = _room_fixture(tmp_path)
    for room in (1, project_chat, 0, 777):
        evidence = [row for row in read_room_source(tmp_path, room)["rows"] if row["stream"] == "chat"]
        ours = [row for _a, row, _p in cc.iter_room_rows(tmp_path, str(room))]
        assert ours and _projected(ours, tmp_path) == _projected(evidence, tmp_path), room
        tasks = [row for _a, row, _p in cc.iter_room_rows(tmp_path, str(room), task_ids=["t1", "bound"])]
        assert _projected(tasks, tmp_path) == _projected(
            [r for r in evidence if {"t1", "bound"} & {r.get(f) for f in ("task_id", "parent_task_id", "root_task_id")}],
            tmp_path), room
        period = {"from": "2026-09-01T00:00:30+00:00", "to": "2026-09-02T00:01:00+00:00"}
        lower, upper = parse_deadline_ts(period["from"]), parse_deadline_ts(period["to"])
        dated = [row for _a, row, _p in cc.iter_room_rows(tmp_path, str(room), period=period)]
        assert _projected(dated, tmp_path) == _projected(
            [r for r in evidence if r.get("ts") and lower <= parse_deadline_ts(r["ts"]) <= upper], tmp_path), room
    main = [row["text"] for _a, row, _p in cc.iter_room_rows(tmp_path, "1")]
    assert "Start the room" in main and "Bound work" not in main and "Agent traffic" not in main
    assert [row["text"] for _a, row, _p in cc.iter_room_rows(tmp_path, str(project_chat))] == [
        "Start the room", "Bound work", "Child of bound", "Project message"]
    assert list(cc.iter_room_rows(tmp_path, "1", task_ids=[])) == []


def test_room_rows_between_two_addresses_keep_stream_positions(tmp_path):
    _room_fixture(tmp_path)
    room = list(cc.iter_room_rows(tmp_path, "1"))
    every = {pos: address for address, _row, pos in cc.iter_rows(tmp_path)}
    assert all(every[pos] == address for address, _row, pos in room)
    bounded = list(cc.iter_room_rows(tmp_path, "1", from_addr=room[1][0], to_addr=cc.format_address(room[3][0])))
    assert bounded == room[1:4]
    assert list(cc.iter_room_rows(tmp_path, "1", from_addr=room[3][0], to_addr=room[1][0])) == []


def test_a_redelivered_message_stays_two_addressable_rows(tmp_path):
    from ouroboros.dialogue_evidence import read_room_source

    live, _archive = _layout(tmp_path)
    _append(live, _msg("2026-09-01T00:00:00+00:00", "same words", client_message_id="c"),
            _msg("2026-09-01T00:00:09+00:00", "same words", client_message_id="c"))
    ours = list(cc.iter_room_rows(tmp_path, "1"))
    assert [pos for _a, _r, pos in ours] == [0, 1] and ours[0][0]["row_sha256"] != ours[1][0]["row_sha256"]
    # The evidence view folds a repeated client id into its first copy; the stream does not.
    assert [row["text"] for row in read_room_source(tmp_path, 1)["rows"]] == ["same words"]
