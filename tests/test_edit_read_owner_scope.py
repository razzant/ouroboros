"""Registry regressions for the owner-approved file editing and reading contract."""
from __future__ import annotations

import pathlib

import pytest

from ouroboros.tool_access import resource_root_path
from ouroboros.tools.core_file_tools import delivered_source_prefix
from ouroboros.tools.registry import ToolContext, ToolRegistry

pytestmark = pytest.mark.serial


@pytest.fixture
def file_tools(tmp_path, monkeypatch):
    home, system, workspace, data = (tmp_path / name for name in ("home", "repo", "workspace", "data"))
    for directory in (home, system, workspace, data):
        directory.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    ctx = ToolContext(repo_dir=system, drive_root=data, workspace_root=workspace,
                      workspace_mode="external", task_id="edit-owner-scope")
    registry = ToolRegistry(repo_dir=system, drive_root=data)
    registry.set_context(ctx)
    payload = data / "skills" / "external" / "demo"
    payload.mkdir(parents=True)
    (payload / "SKILL.md").write_text("# Demo\n")
    return registry, ctx, home, system, workspace, data, payload


def _target(file_tools, root):
    _, ctx, _, _, _, _, payload = file_tools
    return payload if root == "skill_payload" else resource_root_path(ctx, root)


def _selectors(root):
    return {"bucket": "external", "skill_name": "demo"} if root == "skill_payload" else {}


@pytest.mark.parametrize("root", ["active_workspace", "system_repo", "runtime_data",
                                  "task_drive", "artifact_store", "user_files", "skill_payload"])
def test_batch_and_patch_use_each_writable_root_through_registry(file_tools, root):
    registry, _, *_ = file_tools
    base = _target(file_tools, root)
    base.mkdir(parents=True, exist_ok=True)
    target = base / "sample.txt"
    target.write_text("alpha\nbeta\n")
    batch = registry.execute("edit_batch", {"root": root, **_selectors(root), "edits": [
        {"path": "sample.txt", "old_str": "alpha", "new_str": "ALPHA"},
        {"path": "sample.txt", "old_str": "beta", "new_str": "BETA"},
    ]})
    assert batch.startswith("✅"), batch
    assert "| ALPHA" in batch and "| BETA" in batch
    patch = registry.execute("apply_patch", {"root": root, **_selectors(root), "patch":
        "*** Update File: sample.txt\n-ALPHA\n+one\n-BETA\n+two\n"})
    assert patch.startswith("✅"), patch
    assert target.read_text() == "one\ntwo\n"
    assert "| one" in patch


def test_edit_text_list_is_atomic_and_singleton_reports_old_site(file_tools):
    registry, _, _, _, workspace, *_ = file_tools
    target = workspace / "notes.txt"
    target.write_text("new\nold\nend\n")
    result = registry.execute("edit_text", {"path": "notes.txt", "old_str": "old", "new_str": "new"})
    assert "line 2" in result and "     2| new" in result, result
    failed = registry.execute("edit_text", {"path": "notes.txt", "edits": [
        {"old_str": "end", "new_str": "done"}, {"old_str": "missing", "new_str": "x"},
    ]})
    assert "NOTHING was written" in failed and target.read_text() == "new\nnew\nend\n"
    success = registry.execute("edit_text", {"path": "notes.txt", "edits": [
        {"old_str": "end", "new_str": "done"}, {"old_str": "done", "new_str": "finish"},
    ]})
    assert success.startswith("✅") and target.read_text() == "new\nnew\nfinish\n"
    assert "     3| finish" in success
    mixed = registry.execute("edit_text", {"path": "notes.txt", "old_str": "new", "new_str": "x",
                                            "edits": [{"old_str": "finish", "new_str": "end"}]})
    assert "not both" in mixed and target.read_text() == "new\nnew\nfinish\n"


def test_omitted_root_uses_absolute_home_target_without_widening_explicit_root(file_tools):
    registry, ctx, home, _, workspace, *_ = file_tools
    target = home / "outside.txt"
    target.write_text("start\n")
    first = registry.execute("edit_text", {"path": str(target), "old_str": "start", "new_str": "middle"})
    assert first.startswith("OK: edited"), first
    second = registry.execute("edit_batch", {"edits": [
        {"path": str(target), "old_str": "middle", "new_str": "next"}]})
    assert second.startswith("✅"), second
    third = registry.execute("apply_patch", {"patch": f"*** Update File: {target}\n-next\n+done\n"})
    assert third.startswith("✅"), third
    assert target.read_text() == "done\n"
    explicit = registry.execute("edit_text", {"root": "active_workspace", "path": str(target),
                                              "old_str": "done", "new_str": "wrong"})
    assert "outside selected root" in explicit and target.read_text() == "done\n"
    direct = __import__("ouroboros.tools.core", fromlist=["_edit_text"])._edit_text
    assert direct(ctx, str(target), "done", "direct").startswith("OK: edited")
    assert target.read_text() == "direct\n"
    assert not (workspace / target.relative_to(home)).exists()


def test_patch_collects_bad_hunks_and_avoids_all_writes(file_tools):
    registry, _, _, _, workspace, *_ = file_tools
    (workspace / "a.txt").write_text("first\nsecond\n")
    (workspace / "b.txt").write_text("valid\n")
    result = registry.execute("apply_patch", {"patch":
        "*** Update File: a.txt\n-missing one\n+x\n@@ absent\n-missing two\n+y\n"
        "*** Update File: b.txt\n-valid\n+changed\n"})
    assert "hunk 1" in result and "hunk 2" in result
    assert (workspace / "a.txt").read_text() == "first\nsecond\n"
    assert (workspace / "b.txt").read_text() == "valid\n"


def test_numbered_read_keeps_source_offsets_and_prefix_cuts(file_tools):
    registry, ctx, _, _, workspace, *_ = file_tools
    (workspace / "view.txt").write_text("a\u2028long αβγ\nlast", encoding="utf-8")
    result = registry.execute("read_file", {"path": "view.txt", "start_line": 2, "max_lines": 1,
                                            "start_char": 5})
    view = ctx.last_read_view
    assert "     2\tαβγ" in result
    assert delivered_source_prefix(view, result, view["body_start"] + 2) == ""
    assert delivered_source_prefix(view, result, len(result)) == "αβγ\n"
    assert view["source_end_char"] - view["source_start_char"] == 4


@pytest.mark.parametrize("root", ["active_workspace", "system_repo", "runtime_data",
                                  "task_drive", "artifact_store", "user_files", "skill_payload"])
def test_patch_add_delete_authority_and_data_recovery(file_tools, root):
    registry, _, _, _, _, data, _ = file_tools
    base = _target(file_tools, root)
    base.mkdir(parents=True, exist_ok=True)
    file_path = str(base / "extra.txt") if root == "user_files" else "extra.txt"
    added = registry.execute("apply_patch", {"root": root, **_selectors(root), "patch":
        f"*** Add File: {file_path}\n+created\n"})
    assert added.startswith("✅"), added
    target = base / "extra.txt"
    assert target.read_text() == "created\n"
    if root not in {"active_workspace", "system_repo"}:
        refused = registry.execute("apply_patch", {"root": root, **_selectors(root), "patch":
            f"*** Delete File: {file_path}\n"})
        assert "force=true" in refused and target.exists()
    deleted = registry.execute("apply_patch", {"root": root, **_selectors(root), "force": True,
                                               "patch": f"*** Delete File: {file_path}\n"})
    assert deleted.startswith("✅") and not target.exists(), deleted
    if root not in {"active_workspace", "system_repo"}:
        from ouroboros.artifacts import registered_task_artifact

        assert "Recovery copies:" in deleted
        name = deleted.split("artifact_store:")[-1].splitlines()[0]
        record = registered_task_artifact(data, "edit-owner-scope", name)
        assert record and pathlib.Path(record["path"]).read_bytes() == b"created\n"


def test_repo_protection_survives_aliases_and_add_delete(file_tools):
    registry, _, _, system, _, _, _ = file_tools
    bible = system / "BIBLE.md"
    bible.write_text("constitution\n")
    alias = system / "alias.md"
    alias.symlink_to(bible)
    for path in ("BIBLE.md", "repo/BIBLE.md", str(bible), "alias.md"):
        batch = registry.execute("edit_batch", {"root": "system_repo", "edits": [
            {"path": path, "old_str": "constitution", "new_str": "erased"}]})
        patch = registry.execute("apply_patch", {"root": "system_repo", "patch":
            f"*** Delete File: {path}\n"})
        assert "BLOCKED" in batch.upper() or "protected" in batch.lower(), (path, batch)
        assert "BLOCKED" in patch.upper() or "protected" in patch.lower(), (path, patch)
        assert bible.read_text() == "constitution\n"
    added = registry.execute("apply_patch", {"root": "system_repo", "patch":
        "*** Add File: BIBLE.md\n+replacement\n"})
    assert "BLOCKED" in added.upper() or "protected" in added.lower()


def test_patch_out_of_order_indent_and_ambiguous_anchor():
    from ouroboros.tools.edit_ops import _apply_hunks_to_text, _parse_patch

    patch = ("*** Update File: f.txt\n"
             "-last\n+LAST\n"
             "@@ def first\n"
             "-    old\n+    new\n")
    ops, error = _parse_patch(patch)
    assert not error
    changed, notes, failure = _apply_hunks_to_text("def first\n        old\nlast\n", ops[0].hunks, "f.txt")
    assert not failure and changed == "def first\n        new\nLAST\n"
    assert any("indentation" in note for note in notes)
    ambiguous, _, failure = _apply_hunks_to_text("marker\nmarker\n", _parse_patch(
        "*** Update File: f.txt\n@@ marker\n+insert\n")[0][0].hunks, "f.txt")
    assert ambiguous is None and "ambiguous" in failure


def test_patch_original_spans_overlap_and_uniform_indent_failure():
    from ouroboros.tools.edit_ops import _apply_hunks_to_text, _parse_patch

    overlapping = ("*** Update File: f.txt\n@@\n-a\n+A\n"
                   "@@\n a\n-b\n+B\n")
    hunks = _parse_patch(overlapping)[0][0].hunks
    changed, _, failure = _apply_hunks_to_text("a\nb\n", hunks, "f.txt")
    assert changed is None and "overlap" in failure
    nonuniform = ("*** Update File: f.txt\n"
                  "-  first\n+  FIRST\n"
                  "-    second\n+    SECOND\n")
    hunks = _parse_patch(nonuniform)[0][0].hunks
    changed, _, failure = _apply_hunks_to_text("    first\n       second\n", hunks, "f.txt")
    assert changed is None and "not uniform" in failure


@pytest.mark.parametrize("root", ["active_workspace", "system_repo", "runtime_data",
                                  "task_drive", "artifact_store", "user_files", "skill_payload"])
def test_patch_combines_uniform_indent_and_trailing_space(file_tools, root):
    registry, _, *_ = file_tools
    base = _target(file_tools, root)
    base.mkdir(parents=True, exist_ok=True)
    target = base / "combined.txt"
    target.write_text("    anchor   \n    old   \n")
    result = registry.execute("apply_patch", {"root": root, **_selectors(root), "patch":
        "*** Update File: combined.txt\n   anchor\n-  old\n+  new\n"})
    assert result.startswith("✅"), result
    assert target.read_text() == "    anchor   \n    new\n"
    assert "indentation+trailing" in result and "shifted by +2" in result


def test_combined_whitespace_keeps_ambiguity_and_content_refusals():
    from ouroboros.tools.edit_ops import _apply_hunks_to_text, _parse_patch

    hunk = _parse_patch("*** Update File: f.txt\n-  old\n+  new\n")[0][0].hunks
    changed, _, failure = _apply_hunks_to_text("    old   \n      old \n", hunk, "f.txt")
    assert changed is None and "ambiguous" in failure
    changed, _, failure = _apply_hunks_to_text("    OLD   \n", hunk, "f.txt")
    assert changed is None and "context not found" in failure
    mixed = _parse_patch("*** Update File: f.txt\n   anchor\n-  old\n+  new\n")[0][0].hunks
    changed, _, failure = _apply_hunks_to_text("    anchor   \n     old   \n", mixed, "f.txt")
    assert changed is None and "not uniform" in failure


@pytest.mark.parametrize("tool", ["write_file", "edit_text", "edit_batch", "apply_patch"])
def test_explicit_wrong_user_root_redirects_each_editor_without_writes(file_tools, tool):
    registry, _, _, _, workspace, *_ = file_tools
    target = workspace / "explicit.txt"
    target.write_text("old\n")
    if tool == "write_file":
        args = {"path": str(target), "content": "new\n"}
    elif tool == "edit_text":
        args = {"path": str(target), "old_str": "old", "new_str": "new"}
    elif tool == "edit_batch":
        args = {"edits": [{"path": str(target), "old_str": "old", "new_str": "new"}]}
    else:
        args = {"patch": f"*** Update File: {target}\n-old\n+new\n"}
    result = registry.execute(tool, {"root": "user_files", **args})
    assert "ROOT_REQUIRED_ACTIVE_WORKSPACE" in result, result
    assert target.read_text() == "old\n"
    retry = registry.execute(tool, {"root": "active_workspace", **args})
    assert retry.startswith(("✅", "OK:")), retry
    assert target.read_text() == "new\n"


def test_patch_eof_final_newline_and_internal_spacing_are_exact():
    from ouroboros.tools.edit_ops import _apply_hunks_to_text, _parse_patch

    update = _parse_patch("*** Update File: f.txt\n-last\n+LAST\n")[0][0].hunks
    assert _apply_hunks_to_text("top\nlast", update, "f.txt")[0] == "top\nLAST"
    insertion = _parse_patch("*** Update File: f.txt\n@@ last\n+tail\n")[0][0].hunks
    assert _apply_hunks_to_text("top\nlast\n", insertion, "f.txt")[0] == "top\nlast\ntail\n"
    spacing = _parse_patch("*** Update File: f.txt\n-foo bar\n+new\n")[0][0].hunks
    changed, _, failure = _apply_hunks_to_text("foo  bar\n", spacing, "f.txt")
    assert changed is None and "context not found" in failure


def test_data_delete_failed_recovery_keeps_source(file_tools, monkeypatch):
    registry, _, _, _, _, data, _ = file_tools
    source = data / "task_results" / "artifacts" / "edit-owner-scope" / "saved.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("keep\n")
    from ouroboros import artifacts

    def fail_capture(*_args, **_kwargs):
        raise OSError("capture unavailable")

    monkeypatch.setattr(artifacts, "store_task_artifact_bytes", fail_capture)
    result = registry.execute("apply_patch", {"root": "artifact_store", "force": True,
        "patch": "*** Delete File: saved.txt\n"})
    assert "recovery capture failed" in result and source.read_text() == "keep\n"


def test_crlf_notice_and_earlier_replacement_sites(file_tools):
    registry, _, _, _, workspace, *_ = file_tools
    target = workspace / "sequence.txt"
    target.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    result = registry.execute("edit_batch", {"edits": [
        {"path": "sequence.txt", "old_str": "gamma", "new_str": "last"},
        {"path": "sequence.txt", "old_str": "alpha\nbeta\nlast", "new_str": "done"},
    ]})
    assert result.startswith("✅") and "CRLF" in result
    assert "     1| done" in result
    assert target.read_bytes() == b"done\n"


def _call(call_id, name, arguments):
    import json

    return {"id": call_id, "type": "function", "function": {"name": name,
            "arguments": json.dumps(arguments)}}


def test_editor_parallel_footprints_and_model_delivery(file_tools):
    import threading
    from ouroboros.loop_tool_execution import StatefulToolExecutor, handle_tool_calls, tool_calls_can_run_parallel
    from ouroboros.tool_capabilities import FOREGROUND_MUTATIVE_TOOLS
    from ouroboros.tools.core import _edit_text

    registry, ctx, _, _, workspace, data, _ = file_tools
    (workspace / "a.txt").write_text("one\n")
    (workspace / "b.txt").write_text("two\n")
    barrier = threading.Barrier(2)

    def held_editor(call_ctx, _resolved_binding=None, **kwargs):
        barrier.wait(timeout=10)
        return _edit_text(call_ctx, _resolved_binding=_resolved_binding, **kwargs)

    registry.override_handler("edit_text", held_editor)
    calls = [_call("one", "edit_text", {"path": "a.txt", "old_str": "one", "new_str": "ONE"}),
             _call("two", "edit_text", {"path": "b.txt", "old_str": "two", "new_str": "TWO"})]
    assert tool_calls_can_run_parallel(calls, ctx)
    messages, trace = [], {"tool_calls": []}
    logs = data / "logs"
    logs.mkdir()
    stateful = StatefulToolExecutor()
    try:
        errors = handle_tool_calls(calls, registry, logs, ctx.task_id, stateful,
                                   messages, trace, lambda _text: None)
    finally:
        stateful.shutdown()
    assert errors == 0 and [m["tool_call_id"] for m in messages] == ["one", "two"], messages
    assert "| ONE" in messages[0]["content"] and "| TWO" in messages[1]["content"]
    assert (workspace / "a.txt").read_text() == "ONE\n"
    assert (workspace / "b.txt").read_text() == "TWO\n"
    assert {"write_file", "edit_text", "edit_batch", "apply_patch"} <= FOREGROUND_MUTATIVE_TOOLS
    alias = workspace / "alias.txt"
    alias.symlink_to(workspace / "a.txt")
    overlapping = [_call("x", "edit_text", {"path": "a.txt", "old_str": "ONE", "new_str": "A"}),
                   _call("y", "edit_text", {"path": "alias.txt", "old_str": "A", "new_str": "B"})]
    assert not tool_calls_can_run_parallel(overlapping, ctx)
    same_name = [_call("x", "edit_text", {"path": "x/a.txt", "old_str": "x", "new_str": "X"}),
                 _call("y", "edit_text", {"path": "y/a.txt", "old_str": "y", "new_str": "Y"})]
    assert not tool_calls_can_run_parallel(same_name, ctx)
    mixed = [_call("r", "read_file", {"path": "a.txt"}), overlapping[0]]
    assert not tool_calls_can_run_parallel(mixed, ctx)
    registry.override_handler("edit_text", _edit_text)
    ordered = [_call("before", "read_file", {"path": "a.txt"}),
               _call("change", "edit_text", {"path": "a.txt", "old_str": "ONE", "new_str": "AFTER"}),
               _call("after", "read_file", {"path": "a.txt"})]
    messages, trace = [], {"tool_calls": []}
    stateful = StatefulToolExecutor()
    try:
        assert handle_tool_calls(ordered, registry, logs, ctx.task_id, stateful,
                                 messages, trace, lambda _text: None) == 0
    finally:
        stateful.shutdown()
    assert [row["tool_call_id"] for row in messages] == ["before", "change", "after"]
    assert "\tONE" in messages[0]["content"] and "\tAFTER" in messages[2]["content"]
    assert (workspace / "a.txt").read_text() == "AFTER\n"


def test_shared_skill_revision_and_user_artifact_namespace_serialize(file_tools):
    from ouroboros.loop_tool_execution import tool_calls_can_run_parallel

    _, ctx, home, _, _, _, payload = file_tools
    (payload / "one.txt").write_text("a")
    (payload / "two.txt").write_text("b")
    selectors = {"root": "skill_payload", "bucket": "external", "skill_name": "demo"}
    skill_calls = [_call("one", "edit_text", {**selectors, "path": "one.txt", "old_str": "a", "new_str": "A"}),
                   _call("two", "edit_text", {**selectors, "path": "two.txt", "old_str": "b", "new_str": "B"})]
    assert not tool_calls_can_run_parallel(skill_calls, ctx)
    (home / "one.txt").write_text("a")
    (home / "two.txt").write_text("b")
    user_calls = [_call("one", "edit_text", {"root": "user_files", "path": str(home / "one.txt"),
                                            "old_str": "a", "new_str": "A"}),
                  _call("two", "edit_text", {"root": "user_files", "path": str(home / "two.txt"),
                                            "old_str": "b", "new_str": "B"})]
    assert not tool_calls_can_run_parallel(user_calls, ctx)


def test_module_preview_matches_lines_not_whole_file_characters(file_tools, monkeypatch):
    from ouroboros.tools import edit_ops

    registry, _, _, _, workspace, *_ = file_tools
    source = ''.join(f'def function_{i}(value):\n    return value + {i}\n\n' for i in range(1400))
    target = workspace / 'module.txt'
    target.write_text(source)
    original_matcher = edit_ops.difflib.SequenceMatcher
    matched_sizes = []

    def line_matcher(*args, **kwargs):
        assert isinstance(kwargs['a'], list) and isinstance(kwargs['b'], list)
        matched_sizes.append((len(kwargs['a']), len(kwargs['b'])))
        return original_matcher(*args, **kwargs)

    monkeypatch.setattr(edit_ops.difflib, 'SequenceMatcher', line_matcher)
    result = registry.execute('apply_patch', {'patch':
        '*** Update File: module.txt\n-    return value + 700\n+    return value + 7010\n'})
    assert result.startswith('✅'), result
    assert target.read_text() == source.replace('    return value + 700\n', '    return value + 7010\n')
    assert matched_sizes == [(4200, 4200)]
    assert '2102|     return value + 7010' in result


def test_footprint_failure_falls_back_to_actual_dispatch(file_tools, monkeypatch):
    from ouroboros.tools import tool_resolution
    from ouroboros.loop_tool_execution import tool_calls_can_run_parallel

    _, ctx, *_ = file_tools
    calls = [_call('one', 'edit_text', {'path': 'a.txt', 'old_str': 'a', 'new_str': 'A'}),
             _call('two', 'edit_text', {'path': 'b.txt', 'old_str': 'b', 'new_str': 'B'})]

    class FootprintUnavailable(Exception):
        pass

    def unavailable(*args, **kwargs):
        raise FootprintUnavailable('the registry will report the real refusal')

    monkeypatch.setattr(tool_resolution, '_build_builtin_target_binding', unavailable)
    assert not tool_calls_can_run_parallel(calls, ctx)


def test_user_output_registration_is_visible_and_recoverable(file_tools):
    from ouroboros.artifacts import registered_task_artifact

    registry, _, home, _, _, data, _ = file_tools
    source = home / 'report.txt'
    source.write_text('alpha\nbeta\n')
    for tool, arguments in [
        ('edit_batch', {'edits': [{'path': str(source), 'old_str': 'alpha', 'new_str': 'ALPHA'}]}),
        ('apply_patch', {'patch': f'*** Update File: {source}\n-beta\n+BETA\n'}),
    ]:
        result = registry.execute(tool, {'root': 'user_files', **arguments})
        assert result.startswith('✅') and 'ARTIFACT_OUTPUTS: registered user file' in result
        name = result.split('artifact_store:')[-1].splitlines()[0]
        record = registered_task_artifact(data, 'edit-owner-scope', name)
        assert record and pathlib.Path(record['path']).read_bytes() == source.read_bytes()


def test_omitted_root_write_file_registry_and_direct_handler(file_tools):
    from ouroboros.tools.core import _write_file

    registry, ctx, home, *_ = file_tools
    source = home / 'new-report.txt'
    assert registry.execute('write_file', {'path': str(source), 'content': 'first\n'}).startswith('OK: wrote user_files:')
    assert _write_file(ctx, path=str(source), content='second\n').startswith('OK: wrote user_files:')
    assert source.read_text() == 'second\n'


def test_indentation_shift_is_named_in_result():
    from ouroboros.tools.edit_ops import _parse_patch, _apply_hunks_to_text

    hunks = _parse_patch('*** Update File: f.txt\n-  before\n+  after\n')[0][0].hunks
    result, notes, error = _apply_hunks_to_text('    before\n', hunks, 'f.txt')
    assert not error and result == '    after\n'
    assert any('+2 leading characters' in note for note in notes)


def test_payload_reached_by_repo_label_is_not_git_recoverable(file_tools):
    from dataclasses import replace
    from ouroboros.tool_access import build_resolved_resource_binding
    from ouroboros.tools.edit_ops import _repo_edit_binding

    _, ctx, *_ = file_tools
    binding = build_resolved_resource_binding(ctx, root='skill_payload', operation='write',
                                             path='notes.txt', bucket='external', skill_name='demo')
    assert not _repo_edit_binding(binding)
    assert not _repo_edit_binding(replace(binding, root='active_workspace'))


@pytest.mark.parametrize('root', ['active_workspace', 'system_repo', 'runtime_data',
                                  'task_drive', 'artifact_store', 'user_files', 'skill_payload'])
@pytest.mark.parametrize('tool', ['edit_batch', 'apply_patch'])
def test_light_root_parity_and_readonly_ceiling(file_tools, monkeypatch, root, tool):
    registry, ctx, *_ = file_tools
    base = _target(file_tools, root)
    base.mkdir(parents=True, exist_ok=True)
    source = base / 'light.txt'
    source.write_text('before\n')
    path = str(source) if root == 'user_files' else 'light.txt'
    arguments = {'root': root, **_selectors(root)}
    if tool == 'edit_batch':
        arguments['edits'] = [{'path': path, 'old_str': 'before', 'new_str': 'after'}]
    else:
        arguments['patch'] = f'*** Update File: {path}\n-before\n+after\n'
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'light')
    result = registry.execute(tool, arguments)
    if root in {'system_repo', 'runtime_data'}:
        assert 'BLOCKED' in result.upper() or 'cognitive' in result.lower(), result
        assert source.read_text() == 'before\n'
    else:
        assert result.startswith('✅') and source.read_text() == 'after\n', result
    source.write_text('before\n')
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'pro')
    monkeypatch.setattr(ctx, 'task_constraint', {'mode': 'local_readonly_subagent'})
    denied = registry.execute(tool, arguments)
    assert not denied.startswith('✅') and source.read_text() == 'before\n', denied


def test_add_then_update_retains_existing_explicit_chaining(file_tools):
    registry, _, _, _, workspace, *_ = file_tools
    result = registry.execute('apply_patch', {'patch':
        '*** Add File: chain.py\n+x = INVALID\n'
        '*** Update File: chain.py\n-x = INVALID\n+x = 1\n'})
    assert result.startswith('✅') and (workspace / 'chain.py').read_text() == 'x = 1\n', result


@pytest.mark.parametrize("root", ["active_workspace", "system_repo", "runtime_data",
                                  "task_drive", "artifact_store", "user_files", "skill_payload"])
def test_unicode_edit_sites_agree_with_numbered_reader(file_tools, root):
    registry, _, *_ = file_tools
    base = _target(file_tools, root)
    base.mkdir(parents=True, exist_ok=True)
    target = base / "unicode.txt"
    path = str(target) if root == "user_files" else target.name
    args = {"path": path, "root": root, **_selectors(root)}
    target.write_text("first\u2028old\n", encoding="utf-8")
    read = registry.execute("read_file", args)
    assert "     2\told" in read
    singleton = registry.execute("edit_text", {**args, "old_str": "old", "new_str": "new"})
    assert "     2| new" in singleton, singleton
    if root in {"active_workspace", "system_repo", "skill_payload"}:
        assert "line 2" in singleton
    listed = registry.execute("edit_text", {**args, "edits": [{"old_str": "new", "new_str": "next"}]})
    assert "     2| next" in listed, listed
    batched = registry.execute("edit_batch", {"root": root, **_selectors(root), "edits": [
        {"path": path, "old_str": "next", "new_str": "last"}]})
    assert "     2| last" in batched, batched
    # Patch grammar uses LF-delimited context. Exercise its numbered receipt
    # after an unchanged Unicode boundary, without changing that grammar.
    target.write_text("first\u2028middle\nlast\n", encoding="utf-8")
    patched = registry.execute("apply_patch", {"root": root, **_selectors(root), "patch":
        f"*** Update File: {path}\n-last\n+final\n"})
    assert "     3| final" in patched, patched


@pytest.mark.parametrize("root", ["active_workspace", "system_repo", "runtime_data",
                                  "task_drive", "artifact_store", "user_files", "skill_payload"])
@pytest.mark.parametrize("denied", ["delete", "read_bytes", "copy"])
def test_delete_checks_its_own_policy_and_recovery_authority(file_tools, root, denied):
    registry, ctx, _, _, _, data, _ = file_tools
    if root in {"active_workspace", "system_repo"} and denied != "delete":
        pytest.skip("Repo deletion does not read bytes for a data recovery copy")
    base = _target(file_tools, root)
    base.mkdir(parents=True, exist_ok=True)
    target = base / "protected.txt"
    target.write_text("preserved\n")
    ctx.task_metadata = {"task_contract": {"resource_policy": {"protected_artifacts": [
        {"paths": [str(target)], "deny": [denied]}]}}}
    path = str(target) if root == "user_files" else target.name
    result = registry.execute("apply_patch", {"root": root, **_selectors(root), "force": True,
        "patch": f"*** Add File: {base / 'untouched.txt' if root == 'user_files' else 'untouched.txt'}\n+must not land\n*** Delete File: {path}\n"})
    assert "NOTHING was written" in result, result
    assert target.read_text() == "preserved\n" and not (base / "untouched.txt").exists()
    output = data / "task_results" / "artifacts" / "edit-owner-scope"
    assert not list(output.glob("deleted-*.bak"))


def test_public_root_schema_does_not_invent_an_explicit_default(file_tools):
    from ouroboros.tools.core import get_tools as core_tools
    from ouroboros.tools.edit_ops import get_tools as edit_tools

    for entry in core_tools() + edit_tools():
        schema = entry.schema["parameters"]["properties"]
        if "root" in schema:
            assert "default" not in schema["root"], entry.name
            assert "omitted" in schema["root"]["description"].lower(), entry.name


@pytest.mark.parametrize("root", ["active_workspace", "system_repo", "runtime_data",
                                  "task_drive", "artifact_store", "user_files", "skill_payload"])
def test_terminal_line_deletion_keeps_old_match_number(file_tools, root):
    registry, _, *_ = file_tools
    base = _target(file_tools, root)
    base.mkdir(parents=True, exist_ok=True)
    target = base / "terminal.txt"
    target.write_text("first\nlast\n")
    path = str(target) if root == "user_files" else target.name
    result = registry.execute("edit_text", {"root": root, **_selectors(root), "path": path,
                                            "old_str": "last\n", "new_str": "", "force": True})
    assert "line 2" in result, result
    assert "     1| first" in result and "nearest surviving" in result
    assert target.read_text() == "first\n"


def test_empty_post_edit_preview_does_not_invent_a_source_line():
    from ouroboros.tools.edit_ops import numbered_edit_preview

    assert numbered_edit_preview("", [0]) == "(empty file after edit; no surviving source lines)"


def test_editor_results_reach_the_actual_keyless_http_model_send(file_tools, tmp_path, monkeypatch):
    import json
    from ouroboros.llm import LLMClient
    from ouroboros.loop_tool_execution import StatefulToolExecutor, handle_tool_calls
    from ouroboros import usage_accounting
    from tests.test_first_input_selection_wire import WireModel
    from tests.system_e2e.harness import keyless_settings, MOCK_SLUG

    registry, ctx, _, _, workspace, data, _ = file_tools
    (workspace / "wire-a.txt").write_text("alpha\n")
    (workspace / "wire-b.txt").write_text("beta\n")
    calls = [_call("wire-a", "edit_text", {"path": "wire-a.txt", "old_str": "alpha", "new_str": "ALPHA"}),
             _call("wire-b", "edit_batch", {"edits": [
                 {"path": "wire-b.txt", "old_str": "beta", "new_str": "BETA"}]})]
    messages = [{"role": "system", "content": "Synthetic file-editor wire check."},
                {"role": "user", "content": "Apply the two declared edits."},
                {"role": "assistant", "content": None, "tool_calls": calls}]
    logs = data / "logs"
    logs.mkdir()
    stateful = StatefulToolExecutor()
    try:
        assert handle_tool_calls(calls, registry, logs, ctx.task_id, stateful,
                                 messages, {"tool_calls": []}, lambda _: None) == 0
    finally:
        stateful.shutdown()
    evidence = tmp_path / "wire"
    evidence.mkdir()
    with WireModel(evidence, retry=False) as model:
        for key, value in keyless_settings(model).items():
            if isinstance(value, (str, int, float)):
                monkeypatch.setenv(key, str(value))
        monkeypatch.setattr(usage_accounting, "estimate_cost_optional", lambda *_a, **_k: 0.0)
        client = LLMClient()
        try:
            answer, _ = client.chat(messages, model=MOCK_SLUG, max_tokens=256,
                                     no_proxy=True, timeout=30, wait_for_resources=False)
        finally:
            for remote in client._remote_clients.values():
                remote.close()
        assert answer["content"] == "Cooperation complete."
        assert len(model.received) == 1 and not model.errors
        wire = json.loads(pathlib.Path(model.received[0]["path"]).read_bytes())
        delivered = [row for row in wire["messages"] if row["role"] == "tool"]
        assert [row["tool_call_id"] for row in delivered] == ["wire-a", "wire-b"]
        assert "     1| ALPHA" in delivered[0]["content"]
        assert "     1| BETA" in delivered[1]["content"]


@pytest.mark.parametrize("root", ["runtime_data", "task_drive", "artifact_store", "user_files", "skill_payload"])
@pytest.mark.parametrize("tool", ["edit_text", "edit_batch", "apply_patch"])
def test_data_edit_keeps_existing_non_json_text_capability(file_tools, root, tool):
    registry, _, *_ = file_tools
    base = _target(file_tools, root)
    base.mkdir(parents=True, exist_ok=True)
    target = base / "config.json"
    target.write_text("// comment\nold configuration\n")
    arguments = {"root": root, **_selectors(root)}
    if tool == "edit_text":
        arguments.update(path="config.json", old_str="old", new_str="new")
    elif tool == "edit_batch":
        arguments["edits"] = [{"path": "config.json", "old_str": "old", "new_str": "new"}]
    else:
        arguments["patch"] = "*** Update File: config.json\n-old configuration\n+new configuration\n"
    result = registry.execute(tool, arguments)
    assert result.startswith(("OK: edited", "✅")), result
    assert target.read_text() == "// comment\nnew configuration\n"
    assert "SYNTAX_GUARD_BYPASSED" not in result


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("tool", ["edit_text", "edit_batch", "apply_patch"])
def test_repo_syntax_guard_refusal_and_explicit_bypass(file_tools, tool, force):
    registry, _, _, _, workspace, *_ = file_tools
    target = workspace / "config.json"
    target.write_text('{"value": 1}\n')
    if tool == "edit_text":
        arguments = {"path": "config.json", "old_str": "1", "new_str": "invalid"}
    elif tool == "edit_batch":
        arguments = {"edits": [{"path": "config.json", "old_str": "1", "new_str": "invalid"}]}
    else:
        arguments = {"patch": '*** Update File: config.json\n-{"value": 1}\n+{"value": invalid}\n'}
    arguments["force"] = force
    result = registry.execute(tool, arguments)
    assert "SYNTAX" in result, result
    if force:
        assert "SYNTAX_GUARD_BYPASSED" in result, result
        assert target.read_text() == '{"value": invalid}\n'
    else:
        assert "force=true" in result, result
        assert target.read_text() == '{"value": 1}\n'


def test_duplicate_occurrence_lines_use_reader_unicode_boundaries():
    from ouroboros.tools.core import _str_match_replace

    updated, error = _str_match_replace("first\u2028same\u2029third\nsame\n", "same", "new", "sample.txt", "EDIT_TEXT_ERROR")
    assert updated is None
    assert "Occurrences at: line 2, line 4" in error
