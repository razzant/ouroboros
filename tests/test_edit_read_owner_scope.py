"""Registry regressions for the owner-approved file editing and reading contract."""
from __future__ import annotations

import pathlib

import pytest

from ouroboros.tool_access import resource_root_path
from ouroboros.tools.core_file_tools import delivered_source_prefix
from ouroboros.tools.registry import ToolContext, ToolRegistry

@pytest.fixture
def file_tools(tmp_path, monkeypatch):
    home, system, workspace, data = (tmp_path / name for name in ("home", "repo", "workspace", "data"))
    for directory in (home, system, workspace, data):
        directory.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
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
    assert "\tALPHA" in batch and "\tBETA" in batch
    patch = registry.execute("apply_patch", {"root": root, **_selectors(root), "patch":
        "*** Update File: sample.txt\n-ALPHA\n+one\n-BETA\n+two\n"})
    assert patch.startswith("✅"), patch
    assert target.read_text() == "one\ntwo\n"
    assert "\tone" in patch


def test_edit_text_list_is_atomic_and_singleton_reports_old_site(file_tools):
    registry, _, _, _, workspace, *_ = file_tools
    target = workspace / "notes.txt"
    target.write_text("new\nold\nend\n")
    result = registry.execute("edit_text", {"path": "notes.txt", "old_str": "old", "new_str": "new"})
    assert "line 2" in result and "     2\tnew" in result, result
    failed = registry.execute("edit_text", {"path": "notes.txt", "edits": [
        {"old_str": "end", "new_str": "done"}, {"old_str": "missing", "new_str": "x"},
    ]})
    assert "NOTHING was written" in failed and target.read_text() == "new\nnew\nend\n"
    success = registry.execute("edit_text", {"path": "notes.txt", "edits": [
        {"old_str": "end", "new_str": "done"}, {"old_str": "done", "new_str": "finish"},
    ]})
    assert success.startswith("✅") and target.read_text() == "new\nnew\nfinish\n"
    assert "     3\tfinish" in success
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


@pytest.mark.parametrize("root", ["active_workspace", "system_repo", "runtime_data",
                                  "task_drive", "artifact_store", "user_files", "skill_payload"])
def test_patch_preview_tracks_placed_change_with_repeated_lines(file_tools, root):
    registry, _, *_ = file_tools
    base = _target(file_tools, root)
    base.mkdir(parents=True, exist_ok=True)
    target = base / "repeated.txt"
    target.write_text("c\nc\na\nc\na\na\nc\na\nc\nb\n")
    result = registry.execute("apply_patch", {"root": root, **_selectors(root), "patch":
        "*** Update File: repeated.txt\n c\n-a\n+c\n a\n"})
    assert result.startswith("✅"), result
    assert target.read_text().splitlines()[4] == "c"
    assert "     5\tc" in result, result


@pytest.mark.parametrize("tool", ["edit_text", "edit_batch", "apply_patch"])
def test_light_runtime_payload_editor_parity(file_tools, monkeypatch, tool):
    registry, _, _, _, _, _, payload = file_tools
    target = payload / "notes.txt"
    target.write_text("old\n")
    path = "skills/external/demo/notes.txt"
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    args = {"root": "runtime_data"}
    if tool == "edit_text":
        args.update(path=path, old_str="old", new_str="new")
    elif tool == "edit_batch":
        args["edits"] = [{"path": path, "old_str": "old", "new_str": "new"}]
    else:
        args["patch"] = f"*** Update File: {path}\n-old\n+new\n"
    result = registry.execute(tool, args)
    assert result.startswith(("✅", "OK:")), result
    assert target.read_text() == "new\n"


def test_patch_previews_remap_chained_sites_and_long_context(file_tools):
    registry, _, _, _, workspace, *_ = file_tools
    target = workspace / "long.txt"
    before = "\n".join(f"line-{i}" for i in range(30)) + "\n"
    target.write_text(before)
    context = "".join(f" line-{i}\n" for i in range(20))
    result = registry.execute("apply_patch", {"patch":
        f"*** Update File: long.txt\n{context}-line-20\n+changed\n"
        "*** Update File: long.txt\n@@ line-0\n+inserted\n"})
    assert result.startswith("✅"), result
    assert target.read_text().splitlines()[21] == "changed"
    assert "    22\tchanged" in result and "     2\tinserted" in result, result
    deleted = registry.execute("apply_patch", {"patch":
        "*** Update File: long.txt\n-changed\n-line-21\n"})
    assert deleted.startswith("✅"), deleted
    assert "    22\tline-22" in deleted, deleted


def test_patch_chaining_preserves_earlier_site_inside_unchanged_context(file_tools):
    registry, _, _, _, workspace, *_ = file_tools
    target = workspace / "context.txt"
    target.write_text("".join(f"line-{i}\n" for i in range(30)))
    context = "".join(" " + ("changed" if i == 20 else f"line-{i}") + "\n" for i in range(25))
    result = registry.execute("apply_patch", {"patch":
        "*** Update File: context.txt\n-line-20\n+changed\n"
        f"*** Update File: context.txt\n{context}-line-25\n+second\n"})
    assert result.startswith("✅"), result
    assert target.read_text().splitlines()[20] == "changed"
    assert "    21\tchanged" in result and "    26\tsecond" in result, result


def test_light_payload_and_control_data_mix_does_not_widen_authority(file_tools, monkeypatch):
    registry, _, _, _, _, data, payload = file_tools
    (payload / "notes.txt").write_text("old\n")
    (data / "ordinary.txt").write_text("old\n")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    result = registry.execute("edit_batch", {"root": "runtime_data", "edits": [
        {"path": "skills/external/demo/notes.txt", "old_str": "old", "new_str": "new"},
        {"path": "ordinary.txt", "old_str": "old", "new_str": "new"}]})
    assert not result.startswith("✅"), result
    assert (payload / "notes.txt").read_text() == (data / "ordinary.txt").read_text() == "old\n"


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
    target.write_bytes(b"created\r\nuntracked edit\n")  # bytes no repository could restore
    deleted = registry.execute("apply_patch", {"root": root, **_selectors(root), "force": True,
                                               "patch": f"*** Delete File: {file_path}\n"})
    assert deleted.startswith("✅") and not target.exists(), deleted
    from ouroboros.artifacts import registered_task_artifact

    assert "Recovery copies:" in deleted
    name = deleted.split("artifact_store:")[-1].split(" (sha256", 1)[0]
    record = registered_task_artifact(data, "edit-owner-scope", name)
    assert record and record["immutable"] and pathlib.Path(record["path"]).read_bytes() == b"created\r\nuntracked edit\n"


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
             "-        old\n+        new\n")
    ops, error = _parse_patch(patch)
    assert not error
    changed, notes, failure = _apply_hunks_to_text("def first\n        old\nlast\n", ops[0].hunks, "f.txt")
    assert not failure and changed == "def first\n        new\nLAST\n" and not notes
    shifted = _parse_patch("*** Update File: f.txt\n@@ def first\n-    old\n+    new\n")[0][0].hunks
    changed, _, failure = _apply_hunks_to_text("def first\n        old\nlast\n", shifted, "f.txt")
    assert changed is None and "context not found" in failure and "indentation differs" in failure
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
    shifted = ("*** Update File: f.txt\n"
               "-  first\n+  FIRST\n")
    hunks = _parse_patch(shifted)[0][0].hunks
    changed, _, failure = _apply_hunks_to_text("    first\n", hunks, "f.txt")
    assert changed is None and "context not found" in failure


@pytest.mark.parametrize("root", ["active_workspace", "system_repo", "runtime_data",
                                  "task_drive", "artifact_store", "user_files", "skill_payload"])
def test_patch_trailing_space_keeps_file_bytes_and_never_shifts_indentation(file_tools, root):
    registry, _, *_ = file_tools
    base = _target(file_tools, root)
    base.mkdir(parents=True, exist_ok=True)
    target = base / "combined.txt"
    target.write_text("    anchor   \n    old   \n")
    path = str(target) if root == "user_files" else target.name
    shifted = registry.execute("apply_patch", {"root": root, **_selectors(root), "patch":
        f"*** Update File: {path}\n   anchor\n-  old\n+  new\n"})
    assert "context not found" in shifted and "indentation differs" in shifted, shifted
    assert target.read_text() == "    anchor   \n    old   \n"
    result = registry.execute("apply_patch", {"root": root, **_selectors(root), "patch":
        f"*** Update File: {path}\n     anchor\n-    old\n+    new\n"})
    assert result.startswith("✅"), result
    assert target.read_text() == "    anchor   \n    new\n"
    assert "ignoring trailing whitespace" in result and "shifted" not in result


def test_trailing_whitespace_keeps_ambiguity_and_content_refusals():
    from ouroboros.tools.edit_ops import _apply_hunks_to_text, _parse_patch

    hunk = _parse_patch("*** Update File: f.txt\n-old\n+new\n")[0][0].hunks
    changed, _, failure = _apply_hunks_to_text("old   \nold \n", hunk, "f.txt")
    assert changed is None and "ambiguous" in failure
    changed, _, failure = _apply_hunks_to_text("OLD   \n", hunk, "f.txt")
    assert changed is None and "context not found" in failure
    shifted = _parse_patch("*** Update File: f.txt\n   anchor\n-  old\n+  new\n")[0][0].hunks
    changed, _, failure = _apply_hunks_to_text("    anchor   \n    old   \n", shifted, "f.txt")
    assert changed is None and "context not found" in failure


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

    monkeypatch.setattr(artifacts, "copy_file_to_task_artifacts", fail_capture)
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
    assert "     1\tdone" in result
    assert target.read_bytes() == b"done\n"


def _call(call_id, name, arguments):
    import json

    return {"id": call_id, "type": "function", "function": {"name": name,
            "arguments": json.dumps(arguments)}}


def test_editor_round_runs_in_call_order_and_each_call_sees_the_previous(file_tools):
    from ouroboros.loop_tool_execution import StatefulToolExecutor, handle_tool_calls, tool_calls_can_run_parallel
    from ouroboros.tool_capabilities import FOREGROUND_MUTATIVE_TOOLS

    registry, ctx, _, _, workspace, data, _ = file_tools
    (workspace / "a.txt").write_text("one\n")
    (workspace / "b.txt").write_text("two\n")
    calls = [_call("before", "read_file", {"path": "a.txt"}),
             _call("first", "edit_text", {"path": "a.txt", "old_str": "one", "new_str": "ONE"}),
             _call("other", "edit_batch", {"edits": [{"path": "b.txt", "old_str": "two", "new_str": "TWO"}]}),
             _call("second", "edit_text", {"path": "a.txt", "old_str": "ONE", "new_str": "AFTER"}),
             _call("after", "read_file", {"path": "a.txt"})]
    disjoint = [calls[1], calls[2]]
    assert not tool_calls_can_run_parallel(disjoint) and not tool_calls_can_run_parallel(calls)
    assert {"write_file", "edit_text", "edit_batch", "apply_patch"} <= FOREGROUND_MUTATIVE_TOOLS
    messages, trace = [], {"tool_calls": []}
    logs = data / "logs"
    logs.mkdir()
    stateful = StatefulToolExecutor()
    try:
        assert handle_tool_calls(calls, registry, logs, ctx.task_id, stateful,
                                 messages, trace, lambda _text: None) == 0
    finally:
        stateful.shutdown()
    assert [row["tool_call_id"] for row in messages] == ["before", "first", "other", "second", "after"]
    assert "\tone" in messages[0]["content"] and "\tAFTER" in messages[4]["content"]
    assert (workspace / "a.txt").read_text() == "AFTER\n" and (workspace / "b.txt").read_text() == "TWO\n"


def test_module_preview_uses_placed_sites_without_diff_matching(file_tools, monkeypatch):
    from ouroboros.tools import edit_ops

    registry, _, _, _, workspace, *_ = file_tools
    source = ''.join(f'def function_{i}(value):\n    return value + {i}\n\n' for i in range(1400))
    target = workspace / 'module.txt'
    target.write_text(source)
    def no_matcher(*args, **kwargs):
        raise AssertionError('a placed edit needs no diff alignment for its preview')

    monkeypatch.setattr(edit_ops.difflib, 'SequenceMatcher', no_matcher)
    result = registry.execute('apply_patch', {'patch':
        '*** Update File: module.txt\n-    return value + 700\n+    return value + 7010\n'})
    assert result.startswith('✅'), result
    assert target.read_text() == source.replace('    return value + 700\n', '    return value + 7010\n')
    assert '2102\t    return value + 7010' in result


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


def test_indentation_difference_shows_the_region_instead_of_shifting():
    from ouroboros.tools.edit_ops import _parse_patch, _apply_hunks_to_text

    hunks = _parse_patch('*** Update File: f.txt\n-  before\n+  after\n')[0][0].hunks
    result, notes, error = _apply_hunks_to_text('    before\n', hunks, 'f.txt')
    assert result is None and not notes
    assert 'indentation differs: file 4 leading chars, needle 2' in error
    assert "1\t    before" in error and 'read_file start_line=1 max_lines=1' in error


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
    assert "     2\tnew" in singleton, singleton
    if root in {"active_workspace", "system_repo", "skill_payload"}:
        assert "line 2" in singleton
    listed = registry.execute("edit_text", {**args, "edits": [{"old_str": "new", "new_str": "next"}]})
    assert "     2\tnext" in listed, listed
    batched = registry.execute("edit_batch", {"root": root, **_selectors(root), "edits": [
        {"path": path, "old_str": "next", "new_str": "last"}]})
    assert "     2\tlast" in batched, batched
    # Patch grammar uses LF-delimited context. Exercise its numbered receipt
    # after an unchanged Unicode boundary, without changing that grammar.
    target.write_text("first\u2028middle\nlast\n", encoding="utf-8")
    patched = registry.execute("apply_patch", {"root": root, **_selectors(root), "patch":
        f"*** Update File: {path}\n-last\n+final\n"})
    assert "     3\tfinal" in patched, patched


@pytest.mark.parametrize("root", ["active_workspace", "system_repo", "runtime_data",
                                  "task_drive", "artifact_store", "user_files", "skill_payload"])
@pytest.mark.parametrize("denied", ["delete", "read_bytes", "copy"])
def test_delete_checks_its_own_policy_and_recovery_authority(file_tools, root, denied):
    registry, ctx, _, _, _, data, _ = file_tools
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
    assert not list(output.glob("protected-*.txt"))


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
    assert "     1\tfirst" in result and "nearest surviving" in result
    assert target.read_text() == "first\n"


def test_empty_post_edit_preview_does_not_invent_a_source_line():
    from ouroboros.tools.edit_ops import numbered_edit_preview

    assert numbered_edit_preview("", [0]) == "(empty file after edit; no surviving source lines)"


def test_post_edit_preview_discloses_long_rows_and_the_whole_cap():
    from ouroboros.tools.edit_ops import numbered_edit_preview

    # Short previews are byte-for-byte what they were before the disclosed bounds.
    assert numbered_edit_preview("a\nb\n", [4]) == (
        "     1\ta\n     2\tb\nEdit site is at end of file; nearest surviving source lines shown.")
    assert numbered_edit_preview("a\nb\nc\n", [0, 2], max_sites=1) == (
        "     1\ta\n     2\tb\n     3\tc\n… 1 more edit site(s)")
    row = 5_003  # chars per source line, newline excluded
    text = "".join("x" * 5_000 + f"{i:03d}\n" for i in range(600))
    preview = numbered_edit_preview(text, [site * 50 * (row + 1) for site in range(10)])
    tail = "… 2 more edit site(s)"
    assert len(preview) <= 4_000 and len(preview.splitlines()) < 100
    assert preview.startswith("     1\t" + "x" * 100) and preview.endswith("\n" + tail)
    assert f"OMISSION NOTE: truncated at 240 chars; original length {row}" in preview
    assert f"OMISSION NOTE: truncated at {4_000 - len(tail) - 1} chars;" in preview


def test_apply_patch_schema_and_add_file_warn_on_syntax_after_applying(file_tools):
    from ouroboros.tools.edit_ops import get_tools

    registry, _, _, _, workspace, *_ = file_tools
    description = next(entry for entry in get_tools() if entry.name == "apply_patch").schema["description"]
    assert "(Add File included) is reported as SYNTAX_WARNING after it is applied, never refused" in description
    result = registry.execute("apply_patch", {"patch": "*** Add File: broken.py\n+def broken(:\n"})
    assert result.startswith("✅") and "SYNTAX_WARNING" in result and "the edit WAS applied" in result, result
    assert (workspace / "broken.py").read_text() == "def broken(:\n"


def _artifact_writes(name):
    return {"write_file": {"root": "artifact_store", "path": name, "content": "forged\n", "force": True},
            "edit_text": {"root": "artifact_store", "path": name, "old_str": "real", "new_str": "forged"},
            "edit_batch": {"root": "artifact_store", "edits": [{"path": name, "old_str": "real", "new_str": "forged"}]},
            "apply_patch": {"root": "artifact_store", "patch": f"*** Update File: {name}\n-real\n+forged\n"}}


@pytest.mark.parametrize("mode", ["advanced", "pro", "cyber_pro"])
def test_artifact_reservation_is_one_rule_for_every_writer(file_tools, monkeypatch, mode):
    registry, ctx, *_ = file_tools
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    store = resource_root_path(ctx, "artifact_store")
    store.mkdir(parents=True, exist_ok=True)
    for name in (".artifact_manifest.json", "verification_receipts.jsonl", "report.txt"):
        landed = {}
        for tool, arguments in _artifact_writes(name).items():
            (store / name).write_bytes(b"real\n")
            result = registry.execute(tool, arguments)
            landed[tool] = (store / name).read_bytes() != b"real\n"
            assert landed[tool] or "artifact_store path blocked" in result, (tool, result)
        # Ordinary artifacts stay editable; the reserved authority files follow write_file's
        # own shared rule in every mode, including Cyber's existing advisory agency.
        assert set(landed.values()) == {name == "report.txt" or mode == "cyber_pro"}, (name, landed)


@pytest.mark.serial  # a real local HTTP model stand-in
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
        assert "     1\tALPHA" in delivered[0]["content"]
        assert "     1\tBETA" in delivered[1]["content"]


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
    assert "SYNTAX_WARNING" not in result


@pytest.mark.parametrize("form", ["edit_text", "edit_text_list", "edit_batch", "apply_patch"])
def test_repo_syntax_failure_is_reported_after_the_edit_lands(file_tools, form):
    registry, _, _, _, workspace, *_ = file_tools
    target = workspace / "config.json"
    target.write_text('{"value": 1}\n')
    tool = "edit_text" if form.startswith("edit_text") else form
    if form == "edit_text":
        arguments = {"path": "config.json", "old_str": "1", "new_str": "invalid"}
    elif form == "edit_text_list":
        arguments = {"path": "config.json", "edits": [{"old_str": "1", "new_str": "invalid"}]}
    elif form == "edit_batch":
        arguments = {"edits": [{"path": "config.json", "old_str": "1", "new_str": "invalid"}]}
    else:
        arguments = {"patch": '*** Update File: config.json\n-{"value": 1}\n+{"value": invalid}\n'}
    result = registry.execute(tool, arguments)
    assert result.startswith(("✅", "OK")), result
    assert "SYNTAX_WARNING" in result and "the edit WAS applied" in result, result
    assert "WRITE_BLOCKED" not in result and "force" not in result
    assert target.read_text() == '{"value": invalid}\n'


def test_duplicate_occurrence_lines_use_reader_unicode_boundaries():
    from ouroboros.tools.core import _str_match_replace

    updated, error = _str_match_replace("first\u2028same\u2029third\nsame\n", "same", "new", "sample.txt", "EDIT_TEXT_ERROR")
    assert updated is None
    assert "Occurrences at: line 2, line 4" in error


@pytest.mark.parametrize("own_body", [True, False])
@pytest.mark.parametrize("tool", ["write_file", "edit_text", "edit_batch", "apply_patch"])
def test_admitted_body_copy_keeps_protected_paths_for_every_writer(tmp_path, monkeypatch, tool, own_body):
    system, copy, data = (tmp_path / name for name in ("repo", "copy", "data"))
    for directory in (system, copy, data):
        directory.mkdir()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    bible = copy / "BIBLE.md"
    bible.write_text("constitution\n")
    ctx = ToolContext(repo_dir=system, drive_root=data, workspace_root=str(copy),
                      workspace_mode="self_worktree", task_id="body-copy-child")
    ctx.task_constraint = {"mode": "acting_subagent", "surface": "self_worktree", "write_root": str(copy)}
    ctx.task_metadata = {"delegation_role": "subagent", "workspace_copy": {
        "execution_root": str(copy), "source_root": str(system), "source_is_system_repo": own_body}}
    registry = ToolRegistry(repo_dir=system, drive_root=data)
    registry.set_context(ctx)
    arguments = {"write_file": {"path": "BIBLE.md", "content": "erased\n", "force": True},
                 "edit_text": {"path": "BIBLE.md", "old_str": "constitution", "new_str": "erased"},
                 "edit_batch": {"edits": [{"path": "BIBLE.md", "old_str": "constitution", "new_str": "erased"}]},
                 "apply_patch": {"patch": "*** Update File: BIBLE.md\n-constitution\n+erased\n"}}[tool]
    result = registry.execute(tool, arguments)
    if own_body:  # the admitted copy of the body keeps the body's protected set
        assert "BLOCKED" in result.upper() and bible.read_text() == "constitution\n", result
    else:  # a foreign project's BIBLE.md is that project's ordinary file
        assert result.startswith(("✅", "OK")) and bible.read_text() == "erased\n", result


def test_handler_policy_protects_an_admitted_body_copy_without_the_registry(tmp_path, monkeypatch):
    from ouroboros.tools.core import _edit_text

    system, copy, data = (tmp_path / name for name in ("repo", "copy", "data"))
    for directory in (system, copy, data):
        directory.mkdir()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    (copy / "BIBLE.md").write_text("constitution\n")
    ctx = ToolContext(repo_dir=system, drive_root=data, workspace_root=str(copy),
                      workspace_mode="self_worktree", task_id="body-copy-direct")
    ctx.task_metadata = {"workspace_copy": {"execution_root": str(copy), "source_root": str(system),
                                            "source_is_system_repo": True}}
    result = _edit_text(ctx, "BIBLE.md", "constitution", "erased", root="active_workspace")
    assert "BLOCKED" in result.upper() and (copy / "BIBLE.md").read_text() == "constitution\n", result


@pytest.mark.parametrize("tool", ["edit_text", "edit_batch", "apply_patch"])
def test_misses_cite_reader_lines_across_unicode_line_boundaries(file_tools, tool):
    registry, _, _, _, workspace, *_ = file_tools
    (workspace / "u.txt").write_text("first\u2028second\nthird\n    target here\n", encoding="utf-8")
    assert "     4\t    target here" in registry.execute("read_file", {"path": "u.txt"})
    if tool == "edit_text":
        arguments = {"path": "u.txt", "old_str": "target  here", "new_str": "x"}
    elif tool == "edit_batch":
        arguments = {"edits": [{"path": "u.txt", "old_str": "target  here", "new_str": "x"}]}
    else:
        arguments = {"patch": "*** Update File: u.txt\n-  target here\n+x\n"}
    miss = registry.execute(tool, arguments)
    assert "matches line 4 ignoring whitespace" in miss and "read_file start_line=4 max_lines=1" in miss, miss
    assert "4\t    target here" in miss and (workspace / "u.txt").read_text().endswith("    target here\n")


def test_numbered_read_maps_head_and_tail_without_crediting_prefixes_or_the_gap():
    import json
    from ouroboros.tools.core_file_tools import (
        _render_line_slice, delivered_source_prefix, display_intervals_to_source, display_source_segments)

    content = "".join(f"row {i:04d}\n" for i in range(2000))
    view: dict = {}
    rendered = _render_line_slice("big.txt", content, max_lines=2000, extent=view)
    # The display descriptor stays a few bytes for 2,000 numbered rows.
    assert len(json.dumps(view["display_map"])) < 100
    body = view["body_start"]
    head_end = body + len("     1\t") + 4           # the first row's prefix plus "row "
    tail_start = len(rendered) - len("row 1999\n")   # exactly the last row's source
    assert display_intervals_to_source(view, rendered, [(0, head_end), (tail_start, len(rendered))]) == [
        (0, 4), (len(content) - 9, len(content))]
    # A tail that also shows the last prefix credits the same source; prefixes alone credit nothing.
    assert display_intervals_to_source(view, rendered, [(tail_start - 7, len(rendered))]) == [
        (len(content) - 9, len(content))]
    assert display_intervals_to_source(view, rendered, [(0, body + 6)]) == []
    assert delivered_source_prefix(view, rendered, head_end) == "row "
    assert display_source_segments(view, rendered.replace("\t", " ", 1)) is None


def test_user_file_edit_reports_a_landed_write_when_registration_fails(file_tools, monkeypatch):
    from ouroboros.tools import core

    registry, _, home, *_ = file_tools
    source = home / "notes.txt"
    source.write_text("alpha\n")

    def failed(*_args, **_kwargs):
        raise OSError("artifact store unavailable")

    monkeypatch.setattr(core, "copy_file_to_task_artifacts", failed)
    result = registry.execute("edit_text", {"path": str(source), "old_str": "alpha", "new_str": "beta"})
    assert "EDIT_OPS_PARTIAL_WRITE_FAILED" in result and "WAS written" in result, result
    assert "Do not repeat this call" in result and source.read_text() == "beta\n"


def test_deleting_a_registered_immutable_artifact_never_counts_it_as_its_own_copy(file_tools):
    from ouroboros.artifacts import store_task_artifact_bytes

    registry, _, _, _, _, data, _ = file_tools
    receipt = store_task_artifact_bytes(data, "edit-owner-scope", "kept.txt", b"only copy\n")
    target = data / "task_results" / "artifacts" / "edit-owner-scope" / receipt["path"]
    result = registry.execute("apply_patch", {"root": "artifact_store", "force": True,
                                              "patch": "*** Delete File: kept.txt\n"})
    assert "recovery capture failed" in result and "Nothing was deleted" in result, result
    assert target.read_bytes() == b"only copy\n"


def _body_copy_registry(tmp_path, monkeypatch, *, mode="advanced", grant=False):
    system, copy, data = (tmp_path / name for name in ("repo", "copy", "data"))
    for directory in (system, copy, data):
        directory.mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    (copy / "BIBLE.md").write_text("constitution\n")
    (copy / "alias.md").symlink_to(copy / "BIBLE.md")
    ctx = ToolContext(repo_dir=system, drive_root=data, workspace_root=str(copy),
                      workspace_mode="self_worktree", task_id="body-copy-grants")
    ctx.task_constraint = {"mode": "acting_subagent", "surface": "self_worktree", "write_root": str(copy),
                           **({"protected_paths_grant": True} if grant else {})}
    ctx.task_metadata = {"delegation_role": "subagent", "workspace_copy": {
        "execution_root": str(copy), "source_root": str(system), "source_is_system_repo": True}}
    registry = ToolRegistry(repo_dir=system, drive_root=data)
    registry.set_context(ctx)
    return registry, copy


def test_admitted_body_copy_protection_covers_aliases_add_delete_and_keeps_grants(tmp_path, monkeypatch):
    registry, copy = _body_copy_registry(tmp_path / "advanced", monkeypatch)
    for tool, arguments in (
        ("edit_batch", {"edits": [{"path": "alias.md", "old_str": "constitution", "new_str": "erased"}]}),
        ("apply_patch", {"patch": "*** Delete File: BIBLE.md\n"}),
        ("apply_patch", {"patch": "*** Add File: ouroboros/safety.py\n+UNSAFE = True\n"}),
    ):
        result = registry.execute(tool, arguments)
        assert "BLOCKED" in result.upper(), (tool, result)
    assert (copy / "BIBLE.md").read_text() == "constitution\n" and not (copy / "ouroboros").exists()
    # Pro keeps the existing rule for an acting child: the protected grant is required.
    registry, copy = _body_copy_registry(tmp_path / "pro", monkeypatch, mode="pro")
    refused = registry.execute("edit_text", {"path": "BIBLE.md", "old_str": "constitution", "new_str": "x"})
    assert "BLOCKED" in refused.upper() and (copy / "BIBLE.md").read_text() == "constitution\n"
    registry, copy = _body_copy_registry(tmp_path / "granted", monkeypatch, mode="pro", grant=True)
    granted = registry.execute("edit_text", {"path": "BIBLE.md", "old_str": "constitution", "new_str": "amended"})
    assert granted.startswith(("✅", "OK")) and (copy / "BIBLE.md").read_text() == "amended\n", granted


@pytest.mark.parametrize("tool", ["edit_text", "edit_batch", "apply_patch"])
def test_copied_read_numbers_get_a_hint_and_are_never_stripped(file_tools, tool):
    registry, _, _, _, workspace, *_ = file_tools
    target = workspace / "n.txt"
    target.write_text("first\nold value\nnext\n")
    copied = "     2\told value\n     3\tnext"
    if tool == "edit_text":
        arguments = {"path": "n.txt", "old_str": copied, "new_str": "x"}
    elif tool == "edit_batch":
        arguments = {"edits": [{"path": "n.txt", "old_str": copied, "new_str": "x"}]}
    else:
        arguments = {"patch": "*** Update File: n.txt\n-     2\told value\n+x\n"}
    result = registry.execute(tool, arguments)
    assert "carries read_file line numbers" in result and "matches at line 2" in result, result
    assert target.read_text() == "first\nold value\nnext\n"
    # Genuine tab-separated text that merely starts with digits stays an ordinary exact edit.
    (workspace / "t.tsv").write_text("12\tvalue\n13\tother\n")
    exact = registry.execute("edit_text", {"path": "t.tsv", "old_str": "12\tvalue", "new_str": "12\tnew"})
    assert exact.startswith("OK") or exact.startswith("✅"), exact


def test_repeated_anchor_is_a_lower_bound_and_repeated_context_is_refused():
    from ouroboros.tools.edit_ops import _apply_hunks_to_text, _parse_patch

    text = "def f():\n    a = 1\ndef f():\n    b = 2\n"
    unique = _parse_patch("*** Update File: f.py\n@@ def f():\n-    b = 2\n+    b = 3\n")[0][0].hunks
    changed, _notes, failure = _apply_hunks_to_text(text, unique, "f.py")
    assert not failure and changed == "def f():\n    a = 1\ndef f():\n    b = 3\n"
    repeated = "def f():\n    a = 1\ndef f():\n    a = 1\n"
    same = _parse_patch("*** Update File: f.py\n@@ def f():\n-    a = 1\n+    a = 9\n")[0][0].hunks
    changed, _notes, failure = _apply_hunks_to_text(repeated, same, "f.py")
    assert changed is None and "ambiguous" in failure and "2 placements" in failure
    assert "line 2: before 'def f():'" in failure and "line 4: before 'def f():'" in failure


def test_repo_lane_delete_keeps_exact_dirty_tracked_and_binary_bytes(file_tools):
    import subprocess
    from ouroboros.artifacts import registered_task_artifact

    registry, _, _, _, workspace, data, _ = file_tools
    def git(*args):
        subprocess.run(["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args],
                       cwd=workspace, check=True, capture_output=True)
    git("init", "-q")
    (workspace / "tracked.txt").write_bytes(b"committed\n")
    git("add", "tracked.txt")
    git("commit", "-q", "-m", "base")
    (workspace / "tracked.txt").write_bytes(b"committed\nuncommitted work\r\n")  # dirty: Git cannot restore it
    (workspace / "blob.bin").write_bytes(bytes(range(256)) * 4)
    for name in ("tracked.txt", "blob.bin"):
        before = (workspace / name).read_bytes()
        result = registry.execute("apply_patch", {"patch": f"*** Delete File: {name}\n"})
        assert result.startswith("✅") and not (workspace / name).exists(), result
        copy = result.split("artifact_store:")[-1].split(" (sha256", 1)[0]
        record = registered_task_artifact(data, "edit-owner-scope", copy)
        assert record and record["immutable"] and pathlib.Path(record["path"]).read_bytes() == before
