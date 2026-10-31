"""SQL dialogue identities and the imported cursor/position contract."""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import pathlib
import shutil
from types import SimpleNamespace

import pytest

from ouroboros import chat_chain as cc, dialogue_store as ds
from tests._dialogue import import_all
from ouroboros.utils import jsonl_generation_signature

REPO = pathlib.Path(__file__).resolve().parents[1]
MOVED = ("retain_memory_source", "generation_cursor", "iter_rows", "resolve_row")


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

def test_imported_generation_cursors_keep_their_absolute_positions(tmp_path):
    live, archive = _layout(tmp_path)
    old = archive / "chat_20260901T000000.jsonl"
    old.write_text(_rows("old"), encoding="utf-8")
    live.write_text(_rows("live"), encoding="utf-8")
    old_gen, live_gen = _first_sha(old), _first_sha(live)
    import_all(tmp_path)
    assert cc.generation_cursor(tmp_path, {}) == (0, False)
    assert cc.generation_cursor(tmp_path, {"last_consolidated_offset": 7}) == (0, True)
    assert cc.generation_cursor(tmp_path, {"chat_log_signature": "junk"}) == (0, False)
    for gen, base in ((old_gen, 0), (live_gen, 2)):
        assert cc.generation_cursor(tmp_path, {"chat_log_signature": {"first_line_sha256": gen},
                                             "last_consolidated_offset": 1}) == (base + 1, False)
    assert cc.generation_cursor(tmp_path, {"chat_log_signature": {"first_line_sha256": "f" * 64}}) == (0, True)
    assert [r["text"] for _a, r, _p in cc.iter_rows(tmp_path)] == ["old-0", "old-1", "live-0", "live-1"]


def test_row_stream_skips_blank_broken_and_a2a_rows_only(tmp_path):
    live, _archive = _layout(tmp_path)
    rows = [{"chat_id": 1, "text": "main"}, {"chat_id": 1001, "text": "project"},
            {"chat_id": -1, "text": "a2a"}, {"chat_id": "-3999", "text": "a2a-text-id"},
            {"text": "no chat id"}, {"chat_id": "main", "text": "non-numeric id"}]
    live.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n{broken json\n", encoding="utf-8")
    assert [row["text"] for _a, row, _p in cc.iter_rows(tmp_path)] == [
        "main", "project", "no chat id", "non-numeric id"]


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
        assert not hasattr(cons, name)
        assert hasattr(cc, name) == (name == "retain_memory_source")
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
    if ds.present(path.parent.parent):
        for row in rows:
            ds.append(path.parent.parent, "chat", row)
        return
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
    import_all(tmp_path)
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
    import_all(tmp_path)
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
    address = cc.row_address({**_row, "text": "what was never said"})
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
    assert row is None and found["status"] == "row_unreadable"
    assert found["unreadable"][0]["path"] == "state/dialog.sqlite"
    assert found["unreadable"][0]["length"] > 0
    live.write_text(json.dumps(_msg("2026-09-01T00:00:00+00:00", "intact")) + "\n", encoding="utf-8")
    assert cc.resolve_row(tmp_path, address)[1] == found  # frozen files cannot erase an imported gap


def test_a_torn_line_elsewhere_keeps_a_substituted_rows_mismatch_candidates(tmp_path):
    # Generic readers (chronicle marks, iter_rows bounds) name the same-time rows a stale address
    # meant; navigation applies its own gap rule (chat_references.resolve_history_row).
    live, _archive = _layout(tmp_path)
    said = _msg("2026-09-01T00:00:00+00:00", "what was said")
    _append(live, said, '{"ts": "2026-09-01T00:00:09+00:00", "chat_id": 1, "text": "torn\n')
    stale = cc.row_address({**said, "text": "what was never said"})
    row, found = cc.resolve_row(tmp_path, stale)
    assert row is None and found["status"] == "row_mismatch", found
    assert [candidate["row_sha256"] for candidate in found["candidates"]] == [
        address["row_sha256"] for address, _row, _pos in cc.iter_rows(tmp_path)]
    with pytest.raises(cc.RowAddressError) as refused:
        list(cc.iter_rows(tmp_path, from_addr=stale))
    assert refused.value.resolution["status"] == "row_mismatch"


def test_chat_chain_keeps_no_index_and_no_physical_locator():
    source = (REPO / "ouroboros/chat_chain.py").read_text(encoding="utf-8")
    for banned in ("sqlite3", "st_ino", "st_birthtime", "st_dev"):
        assert banned not in source


# --- the row stream ----------------------------------------------------------------------------

def test_stream_positions_are_the_legacy_cursor_positions(tmp_path):
    _chain(tmp_path)
    streamed = list(cc.iter_rows(tmp_path))
    assert [pos for _a, _r, pos in streamed] == list(range(7))
    for unit in ds.generation_metadata(tmp_path, "chat"):
        for offset in range(unit["stream_rows"] or 0):
            absolute, gap = cc.generation_cursor(tmp_path, {
                "chat_log_signature": {"first_line_sha256": unit["gen"]},
                "last_consolidated_offset": offset})
            assert not gap and absolute == unit["pos_base"] + offset
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
    assert refused.value.resolution["status"] == "row_unreadable"
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
    import_all(root)
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


def test_a_projects_task_reach_reaches_python_in_one_step_whatever_its_size():
    """The tasks under a project's bound tasks, by parent or by root, as the recursive query names them row by row,
    handed to Python in one step (each ``sqlite3_step`` releases the GIL)."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE tasks (task_id TEXT PRIMARY KEY, parent_task_id TEXT, root_task_id TEXT)")
    tree = [(f"c{i}", f"c{i - 1}" if i else "bound", "bound" if i % 3 else None) for i in range(300)]
    conn.executemany("INSERT INTO tasks VALUES (?,?,?)", tree + [("x1", "elsewhere", "elsewhere"), ("x2", None, None)])
    rows_by_row = {str(row[0]) for row in conn.execute(cc._REACH.replace("json_group_array(id)", "id"), ('["bound"]',))}
    stepped = []

    class Rows(list):
        def fetchone(self):
            return self[0] if self else None

        def fetchall(self):
            return list(self)

    class Stepped:  # counts the result rows each statement hands to Python: one ``sqlite3_step`` each
        def execute(self, *args):
            rows = Rows(conn.execute(*args).fetchall())
            stepped.append(len(rows))
            return rows

    assert cc._reached(Stepped(), {"bound"}) == rows_by_row == {"bound", *(f"c{i}" for i in range(300))}
    assert stepped == [1]


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


def _paged_chain(tmp_path: pathlib.Path) -> pathlib.Path:
    """1,300 chat rows of three kinds (an archive and the live file imported with their legacy
    hints, rows appended to the store after the import) and the A2A rows the stream skips."""
    live, archive = _layout(tmp_path)
    lines = [json.dumps({"chat_id": 1 if i % 3 else 1001, "direction": "in", "text": f"old-{i}"}) + "\n"
             for i in range(700)]
    lines[350:350] = [json.dumps({"chat_id": -1, "text": "a2a"}) + "\n"]
    (archive / "chat_20260901T000000.jsonl").write_text("".join(lines), encoding="utf-8")
    live.write_text(_rows("live", 300), encoding="utf-8")
    import_all(tmp_path)
    _append(live, *(_msg(f"2026-09-02T00:{i // 60:02d}:{i % 60:02d}+00:00", f"new-{i}") for i in range(300)))
    return live


def _one_by_one(root: pathlib.Path) -> list:
    from ouroboros import dialogue_positions as dp
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id

    return [(cc.stored_address(root, entry), entry.row, dp.ordinal_at(root, "chat", entry.seq).pos)
            for entry in ds.iter_rows(root, "chat") if not is_a2a_chat_id(entry.row.get("chat_id", 1))]


def test_addressed_reads_look_rows_up_a_page_at_a_time(tmp_path, monkeypatch):
    """The stream and a room answer each row's address (its legacy hint or its SQL coordinate) and
    position exactly as the one-row lookups do, from a few reads per page of rows: on 24.10 two read
    transactions per open row of 241 rooms kept a server core busy for a quarter hour."""
    _paged_chain(tmp_path)
    expected = _one_by_one(tmp_path)
    kinds = {"legacy" if address["hint"].get("gen") else "sql" for address, _row, _pos in expected}
    assert len(expected) == 1_300 and kinds == {"legacy", "sql"}
    opened, real = [], ds.transaction

    def counted(*args, **kwargs):
        opened.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(ds, "transaction", counted)
    assert list(cc.iter_rows(tmp_path)) == expected
    pages = -(-len(expected) // cc._ADDRESS_PAGE)
    assert len(opened) <= 3 * pages + 3, len(opened)  # the reader's pages, two lookups a page, the stream's end and import check
    opened.clear()
    room = list(cc.iter_room_rows(tmp_path, "1001"))
    assert room == [entry for entry in expected if entry[1].get("chat_id") == 1001] and len(room) == 234
    assert len(opened) <= 3 * pages + 3, len(opened)
    opened.clear()  # the rows behind the addresses: the coordinates in one read, the legacy hints in another
    assert cc.rows_by_address(tmp_path, [entry[0] for entry in expected]) == {a["row_sha256"]: r for a, r, _p in expected}
    assert len(opened) == 2, len(opened)


def test_an_unsettled_unit_still_answers_the_rows_before_it_and_a_busy_store_stops_the_page(tmp_path, monkeypatch):
    from ouroboros import dialogue_positions as dp
    from ouroboros.store import StoreBusy, StoreMigrating

    _chain(tmp_path)
    streamed = list(cc.iter_rows(tmp_path))
    real, unsettled = dp.ordinals_of, [entry.seq for entry in ds.iter_rows(tmp_path, "chat")][4]

    def counts(root, stream, seqs):
        seqs = list(seqs)
        if len(seqs) > 1 or unsettled in seqs:
            raise StoreMigrating("Dialogue history is still importing")
        return real(root, stream, seqs)

    monkeypatch.setattr(dp, "ordinals_of", counts)
    answered = []
    with pytest.raises(StoreMigrating):
        for item in cc.iter_rows(tmp_path):
            answered.append(item)
    assert answered == streamed[:3]  # the A2A row before the unsettled one is no stream row

    asked = []
    monkeypatch.setattr(dp, "ordinals_of", lambda *_a, **_k: (_ for _ in ()).throw(StoreBusy("held")))
    monkeypatch.setattr(dp, "ordinal_at", lambda *args: asked.append(args))
    with pytest.raises(StoreBusy):
        list(cc.iter_rows(tmp_path))
    assert asked == []  # contention never turns a page into one wait per row


def test_a_source_that_fails_mid_page_still_answers_the_rows_it_read_first(tmp_path, monkeypatch):
    """A busy store between the source's own pages ends the reader where the one-row lookups ended it:
    after every row read before the failure, with the same failure. A room's candidate union pages
    unaligned with the reader's pages, so the failure can land inside one."""
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
    from ouroboros.store import StoreBusy

    _paged_chain(tmp_path)
    expected, real = _one_by_one(tmp_path), ds.iter_rows
    read_first = sum(1 for entry in itertools.islice(real(tmp_path, "chat"), 620)
                     if not is_a2a_chat_id(entry.row.get("chat_id", 1)))

    def failing(*args, **kwargs):
        for count, entry in enumerate(real(*args, **kwargs)):
            if count == 620:  # inside the reader's second page
                raise StoreBusy("held")
            yield entry

    monkeypatch.setattr(ds, "iter_rows", failing)
    answered = []
    with pytest.raises(StoreBusy):
        for item in cc.iter_rows(tmp_path):
            answered.append(item)
    assert answered == expected[:read_first] and read_first > cc._ADDRESS_PAGE
