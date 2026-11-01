"""A relative local-image path means the calling task's active workspace.

``view_image``, ``vlm_query`` and the host's auto-attach share one loader. In every
scenario the serving checkout, which is also the process working directory, holds a
different image under the same relative name: only the task's own pixels can pass, and
an image absent from the workspace stays a miss. Absolute and ``~`` addresses keep their
own identity, and the loader's admission guards judge the workspace-resolved path.
"""
from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import types

import pytest
from PIL import Image

from ouroboros import body_candidate
from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.loop_tool_execution import _maybe_auto_attach_image
from ouroboros.tools import vision
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send
from tests.body_candidate_support import isolate, make_ctx, make_serving
from tests.test_discovery_miss_warnings import _protected, _symlink_or_skip

MODEL = "google/gemini-3.5-flash"
RELATIVE = "shots/frame.png"


def _png(color) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 16), color).save(buffer, format="PNG")
    return buffer.getvalue()


TASK_PIXELS = _png((23, 84, 140))
DECOY_PIXELS = _png((200, 30, 30))
OTHER_PIXELS = _png((30, 160, 60))


def _put(path: pathlib.Path, raw: bytes = TASK_PIXELS) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


@pytest.fixture
def world(tmp_path, monkeypatch):
    """The installed layout ``~/Ouroboros/{repo,data}``; the serving checkout is the process directory and holds the decoy."""
    home = tmp_path / "home"
    root = home / "Ouroboros"
    root.mkdir(parents=True)
    serving = make_serving(root)
    data = isolate(monkeypatch, root)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr("ouroboros.vision_routing.supports_vision", lambda *_a, **_k: True)
    _put(serving / RELATIVE, DECOY_PIXELS)
    monkeypatch.chdir(serving)
    return types.SimpleNamespace(home=home, root=root, serving=serving, data=data, outside=tmp_path / "outside",
                                 chdir=monkeypatch.chdir)


def _unbound(world):
    """No binding: the body root answers, so only a process directory elsewhere can differ from it."""
    (world.serving / RELATIVE).unlink()
    elsewhere = world.home / "elsewhere"
    _put(elsewhere / RELATIVE, DECOY_PIXELS)
    world.chdir(elsewhere)
    return ToolContext(repo_dir=world.serving, drive_root=world.data), world.serving


def _external_project(world):
    project = world.home / "project"
    project.mkdir()
    return ToolContext(repo_dir=world.serving, drive_root=world.data,
                       workspace_root=project, workspace_mode="external"), project


def _project_room_chat(world):
    room = world.home / "room"
    room.mkdir()
    return ToolContext(repo_dir=world.serving, drive_root=world.data, is_direct_chat=True, project_id="room",
                       task_metadata={"_project_room_dir": str(room)}), room


def _folderless_project(world):
    ctx = ToolContext(repo_dir=world.serving, drive_root=world.data, task_id="scratch-task", project_id="room")
    return ctx, world.data / "task_drives" / "scratch-task"


def _own_body_candidate(world):
    ctx = make_ctx(world.serving, world.data, "body-root")
    candidate = pathlib.Path(body_candidate.prepare(ctx)["path"])
    assert body_candidate.is_bound(ctx) and not ctx.is_workspace_mode()
    return ctx, candidate


def _external_beside_a_candidate(world):
    ctx, candidate = _own_body_candidate(world)
    _put(candidate / RELATIVE, OTHER_PIXELS)  # the bound body is one more wrong answer here
    project = world.home / "chosen-project"
    project.mkdir()
    ctx.workspace_root, ctx.workspace_mode = project, "external"
    return ctx, project


def _self_worktree_child(world):
    copy = world.root / "subagent_worktrees" / "copies.noindex" / "child"
    copy.mkdir(parents=True)
    return ToolContext(repo_dir=world.serving, drive_root=world.data, task_id="child",
                       workspace_root=copy, workspace_mode="self_worktree",
                       task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree", write_root=str(copy)),
                       task_metadata={"delegation_role": "subagent", "root_task_id": "root"}), copy


def _read_only_child(world):
    folder = world.home / "selected-folder"
    folder.mkdir()
    return ToolContext(repo_dir=world.serving, drive_root=world.data, task_id="reader",
                       workspace_root=folder, workspace_mode="read_only",
                       task_constraint=TaskConstraint(mode="local_readonly_subagent"),
                       task_metadata={"delegation_role": "subagent", "root_task_id": "root"}), folder


SHAPES = (_unbound, _external_project, _project_room_chat, _folderless_project, _own_body_candidate,
          _external_beside_a_candidate, _self_worktree_child, _read_only_child)


def _bound(world, shape):
    ctx, workspace = shape(world)
    registry = ToolRegistry(repo_dir=world.serving, drive_root=world.data)
    registry.set_context(ctx)
    ctx.messages = []
    return registry, workspace


def _named(images) -> list[str]:
    """Whose file these exact bytes are, so a failure reads as the fact it is."""
    owners = {TASK_PIXELS: "task", DECOY_PIXELS: "decoy", OTHER_PIXELS: "other"}
    return [owners.get(raw, "unknown") for raw in images]


def _sent_images(registry) -> list[str]:
    """The images the send path would carry for this transcript."""
    sent = prepare_messages_for_send(registry._ctx.messages, routing=VisionRoutingContext(MODEL, object(), {}))
    return _named(base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
                  for message in sent if isinstance(message.get("content"), list)
                  for block in message["content"] if block.get("type") == "image_url")


def _view(registry, path):
    """A successful registered view_image; a typed miss is also ``status=ok``, so the code decides."""
    viewed = registry.execute_result("view_image", {"path": path})
    assert (viewed.status, viewed.code) == ("ok", "OK"), viewed.text
    return viewed


def _ask_vlm(registry, monkeypatch, path):
    """The registered vlm_query over a fake provider; returns the result and the images it was handed."""
    seen = []
    monkeypatch.setattr(vision, "_get_llm_client", lambda: object())
    monkeypatch.setattr(vision, "_vision_query_with_timeout",
                        lambda _client, **kwargs: (seen.append(kwargs) or "described", {}))
    result = registry.execute_result("vlm_query", {"prompt": "inspect", "file_path": path, "model": MODEL})
    return result, _named(base64.b64decode(call["images"][0]["base64"]) for call in seen)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: shape.__name__.strip("_"))
def test_both_tools_deliver_the_workspace_pixels_never_the_process_directory_file(world, shape, monkeypatch):
    registry, workspace = _bound(world, shape)
    _put(workspace / RELATIVE)

    viewed = registry.execute_result("view_image", {"path": RELATIVE})
    asked, handed = _ask_vlm(registry, monkeypatch, RELATIVE)

    assert {"view_image": (viewed.code, _sent_images(registry)), "vlm_query": (asked.text, handed)} == {
        "view_image": ("OK", ["task"]), "vlm_query": ("described", ["task"])}, viewed.text


@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: shape.__name__.strip("_"))
def test_absence_from_the_workspace_never_selects_the_process_directory_file(world, shape, monkeypatch):
    registry, workspace = _bound(world, shape)
    workspace.mkdir(parents=True, exist_ok=True)

    viewed = registry.execute_result("view_image", {"path": RELATIVE})
    asked, handed = _ask_vlm(registry, monkeypatch, RELATIVE)

    miss = ("ok", "LEGACY_WARNING",
            f"⚠️ FILE_NOT_FOUND: image file not found: {RELATIVE} (resolved: {(workspace / RELATIVE).resolve()}).", [])
    assert {"view_image": (viewed.status, viewed.code, viewed.text, _sent_images(registry)),
            "vlm_query": (asked.status, asked.code, asked.text, handed)} == {"view_image": miss, "vlm_query": miss}


def test_the_host_auto_attach_uses_the_same_workspace_meaning(world):
    registry, project = _bound(world, _external_project)
    _put(project / RELATIVE)
    row = {"fn_name": "ext_demo_screenshot", "is_error": False,
           "result": json.dumps({"ok": True, "auto_attach_image": RELATIVE})}

    observation = _maybe_auto_attach_image(row, registry)

    assert observation["status"] == "attached", observation
    assert pathlib.Path(observation["source_path"]).read_bytes() == TASK_PIXELS
    assert _sent_images(registry) == ["task"]


# --- absolute and home addresses keep their identity ------------------------------------


def test_absolute_addresses_are_not_rebased_onto_the_workspace(world, monkeypatch):
    registry, project = _bound(world, _external_project)
    _put(project / RELATIVE)  # the wrong answer for every address below
    task_id = "absolute-task"
    registry._ctx.task_id = task_id
    addresses = {
        "upload": _put(world.data / "uploads" / RELATIVE, OTHER_PIXELS),
        "skill screenshot": _put(world.data / "state" / "skills" / "demo" / "jobs" / "j" / "output" / "frame.png", OTHER_PIXELS),
        "task artifact": _put(task_artifact_dir_path(world.data, task_id, create=True) / "video_frames" / "frame.png", OTHER_PIXELS),
        "task drive": _put(world.data / "task_drives" / task_id / RELATIVE, OTHER_PIXELS),
    }
    for label, path in addresses.items():
        registry._ctx.messages = []
        _view(registry, str(path))
        assert _sent_images(registry) == ["other"], label
        asked, handed = _ask_vlm(registry, monkeypatch, str(path))
        assert asked.status == "ok" and handed == ["other"], (label, asked.text)

    # An absolute workspace address and an absolute address nothing admits.
    registry._ctx.messages = []
    _view(registry, str((project / RELATIVE).resolve()))
    assert _sent_images(registry) == ["task"]
    outside = _put(world.outside / "secret.png", OTHER_PIXELS)
    refused = registry.execute_result("view_image", {"path": str(outside)})
    assert (refused.status, refused.code) == ("error", "TOOL_ARG_ERROR"), refused


def test_a_home_address_means_the_home(world, monkeypatch):
    registry, project = _bound(world, _external_project)
    _put(project / RELATIVE)
    owner = world.home / "owner"  # a HOME above the fixture's data root would make that root look live
    _put(owner / RELATIVE, OTHER_PIXELS)
    monkeypatch.setenv("HOME", str(owner))
    monkeypatch.setenv("USERPROFILE", str(owner))

    _view(registry, f"~/{RELATIVE}")
    assert _sent_images(registry) == ["other"]


# --- every guard judges the path the workspace resolved ---------------------------------


def _external_outside_every_root(world):
    """A workspace whose own parent no root admits, so ``..`` genuinely leaves."""
    project = world.outside / "project"
    project.mkdir(parents=True)
    return ToolContext(repo_dir=world.serving, drive_root=world.data,
                       workspace_root=project, workspace_mode="external"), project


def test_traversal_is_judged_from_the_workspace_not_the_process_directory(world):
    registry, _project = _bound(world, _external_outside_every_root)
    _put(world.outside / "secret.png", OTHER_PIXELS)
    _put(world.root / "secret.png", DECOY_PIXELS)  # what ``..`` names from the process directory

    refused = registry.execute_result("view_image", {"path": "../secret.png"})

    assert (refused.status, refused.code) == ("error", "TOOL_ARG_ERROR"), refused
    assert str((world.outside / "secret.png").resolve()) in refused.text
    assert registry._ctx.messages == []


@pytest.mark.parametrize("target", ["real.png", "gone.png"], ids=["escape", "dangling escape"])
def test_a_relative_symlink_escape_is_refused_before_any_miss(world, target):
    registry, project = _bound(world, _external_project)
    _put(world.outside / "real.png", OTHER_PIXELS)
    _symlink_or_skip(project / "link.png", world.outside / target)

    refused = registry.execute_result("view_image", {"path": "link.png"})

    assert (refused.status, refused.code) == ("error", "TOOL_ARG_ERROR"), refused
    assert registry._ctx.messages == []


def test_a_symlinked_workspace_root_reads_its_real_folder(world):
    registry, project = _bound(world, _external_project)
    _put(project / RELATIVE)
    alias = world.home / "project-alias"
    _symlink_or_skip(alias, project, directory=True)
    registry._ctx.workspace_root = alias

    _view(registry, RELATIVE)
    assert _sent_images(registry) == ["task"]


@pytest.mark.parametrize("present", [True, False], ids=["existing", "missing"])
def test_a_protected_workspace_image_stays_blocked_under_its_relative_name(world, present):
    registry, project = _bound(world, _external_project)
    if present:
        _put(project / "sealed.png")
    contract = _protected([project / "sealed.png"], ["read_bytes"])
    registry._ctx.task_contract, registry._ctx.task_metadata = contract, {"task_contract": contract}
    _put(world.serving / "sealed.png", DECOY_PIXELS)

    blocked = registry.execute_result("view_image", {"path": "sealed.png"})

    assert blocked.status == "blocked" and "RESOURCE_POLICY_BLOCKED" in blocked.text, blocked
    assert registry._ctx.messages == []


def test_size_and_format_refusals_read_the_workspace_file(world, monkeypatch):
    registry, project = _bound(world, _external_project)
    (project / "notes.png").write_bytes(b"plain text, not an image")
    (project / "folder.png").mkdir()
    _put(project / "large.png", DECOY_PIXELS + b"\0" * 16)
    for name in ("notes.png", "folder.png", "large.png"):  # readable images only in the process directory
        _put(world.serving / name, DECOY_PIXELS)
    monkeypatch.setattr(vision, "_VLM_MAX_FILE_BYTES", len(DECOY_PIXELS))

    for name in ("notes.png", "folder.png", "large.png"):
        refused = registry.execute_result("view_image", {"path": name})
        assert (refused.status, refused.code) == ("error", "TOOL_ARG_ERROR"), (name, refused)
    assert registry._ctx.messages == []


# --- a workspace that cannot answer never falls back to the process directory ------------


@pytest.mark.parametrize("field", ["workspace_root", "_project_room_dir"])
def test_a_deleted_folder_stays_the_address(world, field):
    gone = world.home / "gone"
    if field == "workspace_root":
        ctx = ToolContext(repo_dir=world.serving, drive_root=world.data, workspace_root=gone, workspace_mode="external")
    else:
        ctx = ToolContext(repo_dir=world.serving, drive_root=world.data, is_direct_chat=True, project_id="room",
                          task_metadata={"_project_room_dir": str(gone)})
    registry = ToolRegistry(repo_dir=world.serving, drive_root=world.data)
    registry.set_context(ctx)
    ctx.messages = []

    miss = registry.execute_result("view_image", {"path": RELATIVE})

    assert (miss.status, miss.code) == ("ok", "LEGACY_WARNING"), miss
    assert f"(resolved: {(gone / RELATIVE).resolve()})" in miss.text and ctx.messages == []


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs POSIX permissions enforced for this user")
def test_an_unreadable_workspace_fails_without_reading_the_process_directory(world, monkeypatch):
    registry, project = _bound(world, _external_project)
    _put(project / RELATIVE)
    project.chmod(0)
    try:
        viewed = registry.execute_result("view_image", {"path": RELATIVE})
        asked, handed = _ask_vlm(registry, monkeypatch, RELATIVE)
    finally:
        project.chmod(0o700)

    # Python decides whether a denied stat is an error or an absent file; either way nothing else answers.
    assert registry._ctx.messages == [] and handed == [], (viewed, asked)
    assert str(world.serving.resolve()) not in viewed.text + asked.text


def test_a_broken_room_refuses_a_relative_path_and_keeps_absolute_ones(world, monkeypatch):
    """A set-but-broken project folder: the room's note stays loud, never a repository fallback."""
    ctx = ToolContext(repo_dir=world.serving, drive_root=world.data, is_direct_chat=True, project_id="room",
                      task_metadata={"_project_room_note": "This room's working folder is unavailable."})
    registry = ToolRegistry(repo_dir=world.serving, drive_root=world.data)
    registry.set_context(ctx)
    ctx.messages = []
    upload = _put(world.data / "uploads" / "frame.png", OTHER_PIXELS)

    viewed = registry.execute_result("view_image", {"path": RELATIVE})
    monkeypatch.setattr(vision, "_get_llm_client", lambda: pytest.fail("no workspace, no vision call"))
    asked = registry.execute_result("vlm_query", {"prompt": "inspect", "file_path": RELATIVE})

    for result in (viewed, asked):
        assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR"), result
        assert "working folder is unavailable" in result.text
    assert ctx.messages == []
    _view(registry, str(upload))
    assert _sent_images(registry) == ["other"]


def test_a_context_without_any_workspace_fact_refuses_a_relative_path(world, monkeypatch):
    ctx = types.SimpleNamespace(messages=[], drive_root=str(world.data))
    _put(world.data / "uploads" / RELATIVE, DECOY_PIXELS)
    monkeypatch.chdir(world.data / "uploads")  # the loader's own admitted root must not answer either

    ok, message = vision.attach_local_image_to_context(ctx, RELATIVE)

    assert ok is False and "absolute path" in message
    assert ctx.messages == []
