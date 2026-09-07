"""The core tool producers publish their own result code, with unchanged text.

Two things are pinned per site, because either one alone would let the cutover
change what the loop records:

* the EXACT text the producer returned before it published anything — the string
  ABI the model sees is unchanged;
* that the published code is the code the single adapter already assigns to that
  text — the outcome bucket and ``is_error`` are therefore the same answer the
  host gave for the same bytes, so nativisation carries no owner semantics.

The second assertion is computed, not restated: a site that drifts away from the
adapter fails here rather than in a differential run over a regenerated golden.

v7next F3.1 adaptation, disclosed: the reference also typed the four IN-PLACE
core.py producers (_data_write/_write_file/_edit_text/_forward_to_worker,
MIGRATION rows 2083-2086 — same-file rows outside the D05/D10 relocation
sets this lane was sanctioned to cut over). Their pins are NOT carried here;
they return with those rows (ledger correction entry for this lane).
"""

from __future__ import annotations

import os
import pathlib
import types

import pytest

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools import core_artifacts, core_file_tools
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.tool_result import (
    LegacyTextResultAdapter,
    ToolResult,
    _install_tool_result_sidecar,
    _published_tool_result,
    _restore_tool_result_sidecar,
)


def _published(ctx, tool: str, call, *, owner_delta: str = "") -> ToolResult:
    """Run one producer under the registry's own result-consumption rule.

    ``registry_core`` installs a per-invocation sentinel and accepts the published
    result only when its text is exactly the string the handler returned; a helper
    called outside a dispatch must therefore still return that same text.

    Adapter equality is the default contract. ``owner_delta`` names the owner item
    that authorised a producer to answer something the adapter would not, and it
    asserts the OPPOSITE — the divergence has to be real, so a site cannot claim an
    approved delta it no longer has.
    """
    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        text = call()
        published = _published_tool_result(ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)
    assert isinstance(published, ToolResult), f"{tool}: producer published no typed result"
    assert published.text == text, f"{tool}: published text is not the returned text"
    adapter_code = LegacyTextResultAdapter.from_text(tool, text).code
    if owner_delta:
        assert published.code != adapter_code, (
            f"{tool}: {owner_delta} claims a divergence from the adapter that is not there"
        )
    else:
        assert published.code == adapter_code, (
            f"{tool}: published code diverges from the adapter answer for the same text"
        )
    return published


def _tree(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    repo = tmp_path / "repo"
    drive = tmp_path / "drive"
    (repo / "nested").mkdir(parents=True)
    drive.mkdir()
    (repo / "sample.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    (repo / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (drive / "settings.json").write_text("{}\n", encoding="utf-8")
    return repo, drive


def _readonly_ctx(repo: pathlib.Path, drive: pathlib.Path) -> ToolContext:
    return ToolContext(
        repo_dir=repo,
        drive_root=drive,
        task_constraint=TaskConstraint(mode="local_readonly_subagent"),
        task_metadata={},
    )


@pytest.mark.parametrize(
    ("tool", "args", "code", "text"),
    [
        (
            "read_file",
            {"path": "missing.txt"},
            "LEGACY_WARNING",
            "⚠️ NOT_FOUND: file does not exist: {repo}{sep}missing.txt",
        ),
        (
            "read_file",
            {"path": "identity.md"},
            "LEGACY_WARNING",
            "⚠️ NOT_FOUND: 'identity.md' is not at the repo root.\n\n"
            "This file lives at `data_root/memory/identity.md`, not in the "
            "git repo. Some memory artifacts are already summarized in "
            "context as `## Identity`, but raw memory state must be read "
            "from the data root. If you need the raw file, call "
            "`read_file(root='runtime_data', path='memory/identity.md')`.",
        ),
        (
            "read_file",
            {"path": "memory/none.md", "root": "runtime_data"},
            "LEGACY_WARNING",
            "⚠️ DATA_NOT_YET_CREATED: memory/none.md\n\n"
            "Memory artifacts under memory/ are created lazily on first "
            "write. Treat this as an empty/absent state and proceed with "
            "initialization if that is the task. Use list_files with "
            "root=runtime_data to confirm what currently exists.",
        ),
        (
            "list_files",
            {"path": "nope"},
            "LEGACY_TOOL_ERROR",
            "⚠️ LIST_FILES_ERROR: Directory not found: nope",
        ),
        (
            "list_files",
            {"path": "sample.txt"},
            "LEGACY_TOOL_ERROR",
            "⚠️ LIST_FILES_ERROR: Not a directory: sample.txt",
        ),
        (
            "list_files",
            {"path": "nope", "root": "runtime_data"},
            "LEGACY_TOOL_ERROR",
            "⚠️ LIST_FILES_ERROR: Directory not found: nope",
        ),
        # Owner item A.20, and the only approved TEXT change in the lane: this refusal
        # shipped without the warning marker, so the adapter answered ok and the model
        # received a policy denial in the position of file content. The marker is now
        # present and the producer publishes the code the marker implies.
        (
            "read_file",
            {"path": "state/skills/demo/grants.json", "root": "runtime_data"},
            "DATA_BLOCKED",
            "⚠️ DATA_READ_BLOCKED: skill owner state is not readable through generic data tools.",
        ),
    ],
)
def test_read_and_list_terminals_are_native_through_the_registry(
    tmp_path, tool, args, code, text
):
    repo, drive = _tree(tmp_path)
    tools = ToolRegistry(repo_dir=repo, drive_root=drive)
    expected = text.format(repo=repo, sep=os.sep)

    result = tools.execute_result(tool, dict(args))

    assert result.text == expected
    assert result.code == code
    assert result.code == LegacyTextResultAdapter.from_text(tool, expected).code
    # The string ABI is the same projection, byte for byte.
    assert tools.execute(tool, dict(args)) == expected


def test_root_guard_publishes_its_two_refusals(tmp_path):
    repo, drive = _tree(tmp_path)
    plain = ToolContext(repo_dir=repo, drive_root=drive)
    readonly = _readonly_ctx(repo, drive)

    bad_root = _published(
        plain, "read_file", lambda: core_file_tools._access_or_block(plain, "nope_root", "read")[1]
    )
    assert bad_root.code == "TOOL_ARG_ERROR"
    assert bad_root.text.startswith("⚠️ TOOL_ARG_ERROR: unknown root 'nope_root'; expected one of ")
    assert bad_root.text.endswith(" Roots your profile can read: active_workspace, artifact_store, "
                                  "deliverables, runtime_data, skill_payload, subagent_projects, "
                                  "system_repo, task_drive, user_files.")

    denied = _published(
        readonly,
        "write_file",
        lambda: core_file_tools._access_or_block(readonly, "system_repo", "write")[1],
    )
    assert denied.code == "ACCESS_BLOCKED"
    assert denied.text == (
        "⚠️ TOOL_ACCESS_BLOCKED: profile=local_readonly_subagent cannot write "
        "root=system_repo. Roots your profile can write: (none)."
    )


@pytest.mark.parametrize(
    ("label", "tool", "code", "text"),
    [
        (
            "repo_read",
            "read_file",
            "LEGACY_BLOCKED",
            "⚠️ REPO_READ_BLOCKED: this subagent cannot read repo secret or control files.",
        ),
        (
            "repo_list",
            "list_files",
            "LEGACY_BLOCKED",
            "⚠️ REPO_LIST_BLOCKED: this subagent cannot list repo secret or control paths.",
        ),
        (
            "data_read",
            "read_file",
            "DATA_BLOCKED",
            "⚠️ DATA_READ_BLOCKED: this subagent cannot read secret or owner-control data files.",
        ),
        (
            "data_list",
            "list_files",
            "DATA_BLOCKED",
            "⚠️ DATA_LIST_BLOCKED: this subagent cannot list secret or owner-control data paths.",
        ),
        (
            "resource_block",
            "read_file",
            "LEGACY_BLOCKED",
            "⚠️ READ_FILE_BLOCKED: this subagent cannot access secret or owner-control data files.",
        ),
    ],
)
def test_restricted_subagent_refusals_publish_their_adapter_code(tmp_path, label, tool, code, text):
    repo, drive = _tree(tmp_path)
    ctx = _readonly_ctx(repo, drive)
    calls = {
        "repo_read": lambda: core_file_tools._repo_read(ctx, ".env"),
        "repo_list": lambda: core_file_tools._repo_list(ctx, ".git"),
        "data_read": lambda: core_file_tools._data_read(ctx, "settings.json"),
        "data_list": lambda: core_file_tools._data_list(ctx, "secrets"),
        "resource_block": lambda: core_file_tools._read_file(ctx, ".env", root="system_repo"),
    }

    published = _published(ctx, tool, calls[label])

    assert published.code == code
    assert published.text == text


def test_user_files_path_refusal_stays_a_policy_denial(tmp_path, monkeypatch):
    repo, drive = _tree(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    outside = repo / "sample.txt"

    read = _published(ctx, "read_file", lambda: core_file_tools._read_file(ctx, str(outside), root="user_files"))
    listed = _published(ctx, "list_files", lambda: core_file_tools._list_files(ctx, str(repo), root="user_files"))

    for published, target in ((read, outside), (listed, repo)):
        assert published.code == "USER_FILES_PATH_BLOCKED"
        # {str(target)!r} mirrors the producer's `{raw_text!r}`: on Windows the
        # repr of the path string doubles the backslashes, on POSIX it is just quoting.
        assert published.text == (
            f"⚠️ USER_FILES_PATH_BLOCKED: user_files path blocked: absolute path {str(target)!r} "
            f"is outside the user_files home ({home}). Use root='active_workspace' for "
            "workspace paths, or a home-relative path (e.g. 'Desktop/file.txt') for user files."
        )


def _media_ctx(chat_id=123):
    return types.SimpleNamespace(
        current_chat_id=chat_id,
        pending_events=[],
        browser_state=types.SimpleNamespace(last_screenshot_b64=""),
    )


def _png(tmp_path: pathlib.Path) -> pathlib.Path:
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
    return image


def _mp4(tmp_path: pathlib.Path) -> pathlib.Path:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200)
    return video


@pytest.mark.parametrize(
    ("label", "tool", "code", "text"),
    [
        ("photo_no_chat", "send_photo", "LEGACY_UNAVAILABLE", "⚠️ No active chat — cannot send photo."),
        ("photo_no_source", "send_photo", "LEGACY_TOOL_ERROR", "⚠️ Provide either file_path or image_base64."),
        ("photo_short", "send_photo", "LEGACY_TOOL_ERROR", "⚠️ Image data is empty or too short."),
        ("photo_no_screenshot", "send_photo", "LEGACY_TOOL_ERROR",
         "⚠️ No screenshot stored. Take one first with browse_page(output='screenshot')."),
        ("video_no_chat", "send_video", "LEGACY_UNAVAILABLE", "⚠️ No active chat — cannot send video."),
        ("video_no_path", "send_video", "LEGACY_TOOL_ERROR", "⚠️ Provide a file_path."),
        ("video_missing", "send_video", "LEGACY_TOOL_ERROR", "⚠️ File not found: /nonexistent/clip.mp4"),
        ("file_no_chat", "send_file", "LEGACY_UNAVAILABLE", "⚠️ No active chat — cannot send file."),
        ("file_no_path", "send_file", "LEGACY_TOOL_ERROR", "⚠️ Provide a file_path."),
        ("file_missing", "send_file", "LEGACY_TOOL_ERROR", "⚠️ File not found: /nonexistent/report.md"),
        ("photo_ok", "send_photo", "OK", "OK: photo queued for delivery to owner."),
        ("video_ok", "send_video", "OK", "OK: video queued for delivery to owner."),
        ("file_ok", "send_file", "OK", "OK: file 'shot.png' queued for delivery to owner."),
    ],
)
def test_owner_chat_delivery_terminals_are_native(tmp_path, label, tool, code, text):
    """Every media terminal, including the queued-for-delivery success.

    Owner item A.20: these refusals used to report `ok`, because their sentences
    carry no uppercase identifier for the adapter to key on — a send that queued
    nothing looked like a send that worked. Absence of an owner chat is now the
    `unavailable` surface it describes; everything else that prevented a delivery
    is an `error`. The text is unchanged, so only the code moved.
    """
    chatty = _media_ctx()
    chatless = _media_ctx(chat_id=None)
    calls = {
        "photo_no_chat": (chatless, lambda: core_artifacts._send_photo(chatless, file_path=str(_png(tmp_path)))),
        "photo_no_source": (chatty, lambda: core_artifacts._send_photo(chatty)),
        "photo_short": (chatty, lambda: core_artifacts._send_photo(chatty, image_base64="tiny")),
        "photo_no_screenshot": (chatty, lambda: core_artifacts._send_photo(chatty, image_base64="__last_screenshot__")),
        "video_no_chat": (chatless, lambda: core_artifacts._send_video(chatless, file_path=str(_mp4(tmp_path)))),
        "video_no_path": (chatty, lambda: core_artifacts._send_video(chatty)),
        "video_missing": (chatty, lambda: core_artifacts._send_video(chatty, file_path="/nonexistent/clip.mp4")),
        "file_no_chat": (chatless, lambda: core_artifacts._send_file(chatless, file_path=str(_png(tmp_path)))),
        "file_no_path": (chatty, lambda: core_artifacts._send_file(chatty)),
        "file_missing": (chatty, lambda: core_artifacts._send_file(chatty, file_path="/nonexistent/report.md")),
        "photo_ok": (chatty, lambda: core_artifacts._send_photo(chatty, file_path=str(_png(tmp_path)))),
        "video_ok": (chatty, lambda: core_artifacts._send_video(chatty, file_path=str(_mp4(tmp_path)))),
        "file_ok": (chatty, lambda: core_artifacts._send_file(chatty, file_path=str(_png(tmp_path)))),
    }
    ctx, call = calls[label]

    published = _published(ctx, tool, call, owner_delta="" if code == "OK" else "A.20")

    assert published.code == code
    assert published.text == text
    # A refused delivery queues nothing; a published success queues exactly one event.
    assert len(ctx.pending_events) == (1 if code == "OK" else 0)


@pytest.mark.parametrize(
    ("tool", "args", "code", "text"),
    [
        (
            "write_file",
            {"path": "notes.txt", "root": "runtime_data", "content": "x"},
            "LEGACY_BLOCKED",
            "⚠️ WRITE_BLOCKED: new content for 'notes.txt' is 9% of original "
            "(11 -> 1 chars). This looks like accidental truncation. "
            "Use edit_text for surgical edits, or pass force=true to confirm an "
            "intentional rewrite.",
        ),
        (
            "edit_text",
            {"path": "gone.txt", "root": "runtime_data", "old_str": "a", "new_str": "b"},
            "EDIT_TEXT_BLOCKED",
            "⚠️ EDIT_TEXT_ERROR: file not found: runtime_data:gone.txt",
        ),
        (
            "edit_text",
            {"path": "notes.txt", "root": "runtime_data", "old_str": "zeta", "new_str": "q"},
            "EDIT_TEXT_BLOCKED",
            "⚠️ EDIT_TEXT_ERROR: old_str not found in runtime_data:notes.txt.\n"
            "File preview (first 2000 chars):\nalpha\nbeta\n",
        ),
        (
            "edit_text",
            {"path": "notes.txt", "root": "runtime_data", "old_str": "alpha\nbeta\n", "new_str": "x"},
            "LEGACY_BLOCKED",
            "⚠️ WRITE_BLOCKED: new content for 'notes.txt' is 9% of original "
            "(11 -> 1 chars). This looks like accidental truncation. "
            "Use edit_text for surgical edits, or pass force=true to confirm an "
            "intentional rewrite.",
        ),
        ("search_code", {"query": ""}, "LEGACY_TOOL_ERROR", "⚠️ SEARCH_ERROR: query is required."),
        (
            "search_code",
            {"query": "x", "path": "nope"},
            "LEGACY_TOOL_ERROR",
            "⚠️ SEARCH_ERROR: path not found: active_workspace:nope",
        ),
        (
            "search_code",
            {"query": "[", "regex": True},
            "LEGACY_TOOL_ERROR",
            "⚠️ SEARCH_ERROR: invalid regex: unterminated character set at position 0",
        ),
        (
            "forward_to_worker",
            {"task_id": "not a task id!", "message": "m"},
            "TOOL_ARG_ERROR",
            "⚠️ TOOL_ARG_ERROR (forward_to_worker): task_id must match "
            "[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
        ),
    ],
)
def test_write_edit_search_and_forward_terminals_are_native(tmp_path, tool, args, code, text):
    repo, drive = _tree(tmp_path)
    (drive / "notes.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    tools = ToolRegistry(repo_dir=repo, drive_root=drive)

    result = tools.execute_result(tool, dict(args))

    assert result.text == text
    assert result.code == code
    adapter_code = LegacyTextResultAdapter.from_text(tool, text).code
    if code == "LEGACY_UNAVAILABLE":
        # Owner item A.20: this one row is a deliberate divergence from the adapter.
        assert adapter_code == "LEGACY_WARNING"
    else:
        assert result.code == adapter_code
