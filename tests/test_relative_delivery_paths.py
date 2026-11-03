"""A relative path given to the owner-delivery verbs or the media readers means the calling task's active workspace.

``send_photo``, ``send_video`` and ``send_file`` used to resolve a relative ``file_path``
against the process working directory, and ``extract_pdf_text``/``ocr_pdf`` and
``extract_video_frames`` searched every admitted root with uploads first (#2577; the image
readers were #2440, ``tests/test_relative_image_paths.py``). In each scenario the process
directory and the uploads folder hold a different file under the same relative name: only the
task's own bytes are delivered or read, and a file absent from the workspace stays a miss.
Absolute and ``~`` addresses keep their identity. The media readers keep one compatibility
address: a staged ``attachments/<name>`` also names this task's own artifact store, after the
workspace.
"""
from __future__ import annotations

import base64
import pathlib
import types

import pytest

from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.tools import core_artifacts, media
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.body_candidate_support import isolate, make_serving
from tests.test_discovery_miss_warnings import _protected, _symlink_or_skip
from tests.test_relative_image_paths import (
    DECOY_PIXELS,
    OTHER_PIXELS,
    RELATIVE,
    SHAPES,
    TASK_PIXELS,
    _bound,
    _external_outside_every_root,
    _external_project,
    _put,
    _read_only_child,
    _self_worktree_child,
    _unbound,
)
from tests.test_tool_usability_slice import _write_pdf

PHOTO = RELATIVE
VIDEO = "shots/clip.mp4"
DOC = "shots/report.txt"
PDF = "shots/doc.pdf"
CHAT_ID = 123


def _clip(label: str) -> bytes:
    return b"\x00\x00\x00\x18ftypmp42" + label.encode()


TASK = {PHOTO: TASK_PIXELS, VIDEO: _clip("task"), DOC: b"task report"}
DECOY = {PHOTO: DECOY_PIXELS, VIDEO: _clip("decoy"), DOC: b"decoy report"}
OTHER = {PHOTO: OTHER_PIXELS, VIDEO: _clip("other"), DOC: b"other report"}
SENDS = (("send_photo", PHOTO, "image_base64"), ("send_video", VIDEO, "video_base64"), ("send_file", DOC, "file_base64"))
PDF_TEXT = {"task": "TASKPDF workspace copy", "decoy": "DECOYPDF process directory copy",
            "upload": "UPLOADPDF uploads copy", "store": "STOREPDF artifact store copy",
            "parent": "PARENTPDF parent store copy", "other": "OTHERPDF elsewhere"}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """The installed layout ``~/Ouroboros/{repo,data}``; the serving checkout is the process directory."""
    home = tmp_path / "home"
    root = home / "Ouroboros"
    root.mkdir(parents=True)
    serving = make_serving(root)
    data = isolate(monkeypatch, root)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    _put(serving / PHOTO, DECOY_PIXELS)  # the image shapes expect their decoy here
    monkeypatch.chdir(serving)
    return types.SimpleNamespace(home=home, root=root, serving=serving, data=data, outside=tmp_path / "outside",
                                 chdir=monkeypatch.chdir, events=[], monkeypatch=monkeypatch)


def _decoys(world) -> None:
    """The same names in the process directory and in uploads, each holding someone else's bytes."""
    for base in (pathlib.Path.cwd(), world.data / "uploads"):
        for name, raw in DECOY.items():
            _put(base / name, raw)
        _write_pdf(_dir(base / PDF), [PDF_TEXT["decoy" if base != world.data / "uploads" else "upload"]])


def _dir(path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _registry(world, shape):
    """A registered task in ``shape`` whose owner deliveries are captured, never sent."""
    if shape is _self_worktree_child:  # only Cyber Pro gives a child the owner-delivery verbs
        world.monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    registry, workspace = _bound(world, shape)
    registry._ctx.current_chat_id = CHAT_ID
    world.monkeypatch.setattr("ouroboros.tools.owner_delivery.deliver_owner_event",
                              lambda _ctx, event: (world.events.append(event) or "live"))
    _decoys(world)
    return registry, workspace


def _owner(raw: bytes) -> str:
    for label, files in (("task", TASK), ("decoy", DECOY), ("other", OTHER)):
        if raw in files.values():
            return label
    return "unknown"


def _send_each(world, registry, path_for=lambda name: name) -> dict:
    """Run every registered sender; report its outcome and whose bytes it delivered."""
    seen = {}
    for tool, name, field in SENDS:
        before = len(world.events)
        result = registry.execute_result(tool, {"file_path": path_for(name)})
        seen[tool] = (result.status, [_owner(base64.b64decode(e[field])) for e in world.events[before:]])
    return seen


def _pdf_owner(text: str) -> list[str]:
    return [label for label, marker in PDF_TEXT.items() if marker in text]


# --- owner delivery ----------------------------------------------------------------------

SENDER_SHAPES = tuple(shape for shape in SHAPES if shape is not _read_only_child)  # no delivery verbs


@pytest.mark.parametrize("shape", SENDER_SHAPES, ids=lambda shape: shape.__name__.strip("_"))
def test_each_sender_delivers_the_workspace_file_never_the_process_directory_one(world, shape):
    registry, workspace = _registry(world, shape)
    for name, raw in TASK.items():
        _put(workspace / name, raw)

    assert _send_each(world, registry) == {tool: ("ok", ["task"]) for tool, _name, _field in SENDS}
    assert world.events[-1]["filename"] == "report.txt"


@pytest.mark.parametrize("shape", (_unbound, _external_project, _self_worktree_child),
                         ids=lambda shape: shape.__name__.strip("_"))
def test_a_file_absent_from_the_workspace_is_a_miss_that_names_where_it_looked(world, shape):
    registry, workspace = _registry(world, shape)

    for tool, name, _field in SENDS:
        result = registry.execute_result(tool, {"file_path": name})
        assert (result.status, result.code, result.text) == (
            "error", "LEGACY_TOOL_ERROR", f"⚠️ File not found: {name} (resolved: {(workspace / name).resolve()})")
    assert world.events == []


def test_absolute_and_home_addresses_keep_their_meaning(world):
    registry, project = _registry(world, _external_project)
    for name, raw in TASK.items():
        _put(project / name, raw)  # the wrong answer for the outside and home addresses
        _put(world.outside / name, OTHER[name])
    owner = world.home / "owner"
    for name, raw in OTHER.items():
        _put(owner / name, raw)
    world.monkeypatch.setenv("HOME", str(owner))
    world.monkeypatch.setenv("USERPROFILE", str(owner))

    # Delivery has never been root-confined: an absolute path outside every root still sends.
    others = {tool: ("ok", ["other"]) for tool, _name, _field in SENDS}
    assert _send_each(world, registry, lambda name: str(world.outside / name)) == others
    assert _send_each(world, registry, lambda name: f"~/{name}") == others
    assert _send_each(world, registry, lambda name: str(project / name)) == {
        tool: ("ok", ["task"]) for tool, _name, _field in SENDS}
    for tool, _name, _field in SENDS:  # an absolute miss keeps its exact historical text
        missing = str(world.outside / "missing.bin")
        assert registry.execute_result(tool, {"file_path": missing}).text == f"⚠️ File not found: {missing}"


def test_a_broken_room_refuses_a_relative_path_and_keeps_absolute_ones(world):
    ctx = ToolContext(repo_dir=world.serving, drive_root=world.data, is_direct_chat=True, project_id="room",
                      current_chat_id=CHAT_ID,
                      task_metadata={"_project_room_note": "This room's working folder is unavailable."})
    registry = ToolRegistry(repo_dir=world.serving, drive_root=world.data)
    registry.set_context(ctx)
    world.monkeypatch.setattr("ouroboros.tools.owner_delivery.deliver_owner_event",
                              lambda _ctx, event: (world.events.append(event) or "live"))
    _decoys(world)

    for tool, name, _field in SENDS:
        refused = registry.execute_result(tool, {"file_path": name})
        assert (refused.status, refused.code) == ("error", "TOOL_ARG_ERROR"), refused
        assert "working folder is unavailable" in refused.text and "absolute path" in refused.text
    assert world.events == []
    uploads = world.data / "uploads"
    assert _send_each(world, registry, lambda name: str(uploads / name)) == {
        tool: ("ok", ["decoy"]) for tool, _name, _field in SENDS}


def test_a_deleted_workspace_stays_the_address(world):
    gone = world.home / "gone"
    ctx = ToolContext(repo_dir=world.serving, drive_root=world.data, workspace_root=gone,
                      workspace_mode="external", current_chat_id=CHAT_ID)
    registry = ToolRegistry(repo_dir=world.serving, drive_root=world.data)
    registry.set_context(ctx)
    world.monkeypatch.setattr("ouroboros.tools.owner_delivery.deliver_owner_event",
                              lambda _ctx, event: (world.events.append(event) or "live"))
    _decoys(world)

    for tool, name, _field in SENDS:
        miss = registry.execute_result(tool, {"file_path": name})
        assert miss.text == f"⚠️ File not found: {name} (resolved: {(gone / name).resolve()})", miss
    assert world.events == []


def test_a_context_without_any_workspace_fact_refuses_a_relative_path(world):
    ctx = types.SimpleNamespace(current_chat_id=CHAT_ID, browser_state=types.SimpleNamespace(last_screenshot_b64=""))
    world.monkeypatch.setattr("ouroboros.tools.owner_delivery.deliver_owner_event",
                              lambda _ctx, event: (world.events.append(event) or "live"))
    _decoys(world)
    senders = {"send_photo": core_artifacts._send_photo, "send_video": core_artifacts._send_video,
               "send_file": core_artifacts._send_file}

    for tool, name, _field in SENDS:
        text = senders[tool](ctx, file_path=name)
        assert text.startswith(f"⚠️ Relative file path {name!r} has no active workspace") and "absolute path" in text
    assert world.events == []


# --- media readers -----------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: shape.__name__.strip("_"))
def test_extract_pdf_text_reads_the_workspace_pdf_never_the_upload_or_process_directory_one(world, shape):
    pytest.importorskip("pypdf")
    registry, workspace = _registry(world, shape)
    _write_pdf(_dir(workspace / PDF), [PDF_TEXT["task"]])

    result = registry.execute_result("extract_pdf_text", {"path": PDF})

    assert (result.status, _pdf_owner(result.text)) == ("ok", ["task"]), result.text


def test_a_pdf_only_in_uploads_is_a_workspace_miss_and_stays_readable_by_its_absolute_path(world):
    pytest.importorskip("pypdf")
    registry, project = _registry(world, _external_project)
    project.mkdir(exist_ok=True)

    miss = registry.execute_result("extract_pdf_text", {"path": PDF})
    upload = registry.execute_result("extract_pdf_text", {"path": str(world.data / "uploads" / PDF)})

    assert miss.text == f"⚠️ FILE_NOT_FOUND: {PDF} (resolved: {(project / PDF).resolve()})", miss
    assert _pdf_owner(upload.text) == ["upload"], upload.text


def test_absolute_and_home_pdf_addresses_keep_their_meaning(world):
    pytest.importorskip("pypdf")
    registry, project = _registry(world, _external_project)
    _write_pdf(_dir(project / PDF), [PDF_TEXT["task"]])
    owner = world.home / "owner"
    _write_pdf(_dir(owner / PDF), [PDF_TEXT["other"]])
    world.monkeypatch.setenv("HOME", str(owner))
    world.monkeypatch.setenv("USERPROFILE", str(owner))

    readings = {path: _pdf_owner(registry.execute_result("extract_pdf_text", {"path": path}).text)
                for path in (str(project / PDF), f"~/{PDF}", str(world.serving / PDF))}

    assert readings == {str(project / PDF): ["task"], f"~/{PDF}": ["other"], str(world.serving / PDF): ["decoy"]}


def test_extract_video_frames_hands_ffmpeg_the_workspace_clip(world, monkeypatch):
    registry, project = _registry(world, _external_project)
    _put(project / VIDEO, TASK[VIDEO])
    commands = []

    def fake_ffmpeg(cmd, **_kwargs):
        commands.append(cmd)
        pathlib.Path(cmd[-1]).write_bytes(b"jpg")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(media, "_resolve_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr("ouroboros.tools.shell._tracked_subprocess_run", fake_ffmpeg)
    result = registry.execute_result("extract_video_frames", {"path": VIDEO, "timestamps": "0"})

    assert "Extracted video frame" in result.text, result.text
    assert [cmd[cmd.index("-i") + 1] for cmd in commands] == [str((project / VIDEO).resolve())]


def test_a_staged_attachment_resolves_in_this_tasks_store_after_the_workspace(world):
    pytest.importorskip("pypdf")
    registry, project = _registry(world, _external_project)
    registry._ctx.task_id = "attach-task"
    store = task_artifact_dir_path(world.data, "attach-task", create=True)
    _write_pdf(_dir(store / "attachments" / "doc.pdf"), [PDF_TEXT["store"]])
    _write_pdf(_dir(store / "attachments" / "nested" / "doc.pdf"), [PDF_TEXT["store"]])
    _write_pdf(_dir(store / "notes.pdf"), [PDF_TEXT["store"]])
    for name in ("attachments/doc.pdf", "attachments/nested/doc.pdf", "notes.pdf"):
        _write_pdf(_dir(world.data / "uploads" / name), [PDF_TEXT["upload"]])

    def read(path):
        return registry.execute_result("extract_pdf_text", {"path": path}).text

    assert _pdf_owner(read("attachments/doc.pdf")) == ["store"]
    for name in ("attachments/nested/doc.pdf", "notes.pdf"):  # only the manifest's own two-part shape
        assert read(name).startswith(f"⚠️ FILE_NOT_FOUND: {name} (resolved: {(project / name).resolve()})")
    _write_pdf(_dir(project / "attachments" / "doc.pdf"), [PDF_TEXT["task"]])
    assert _pdf_owner(read("attachments/doc.pdf")) == ["task"]  # a workspace file of that name wins


def test_a_child_reads_the_attachment_copy_in_its_own_store(world):
    """``attachments/<name>`` means the caller's own store, as ``read_file(root='artifact_store')`` does."""
    pytest.importorskip("pypdf")
    registry, _copy = _registry(world, _self_worktree_child)
    _write_pdf(_dir(task_artifact_dir_path(world.data, "root", create=True) / "attachments" / "doc.pdf"),
               [PDF_TEXT["parent"]])

    missing = registry.execute_result("extract_pdf_text", {"path": "attachments/doc.pdf"})
    _write_pdf(_dir(task_artifact_dir_path(world.data, "child", create=True) / "attachments" / "doc.pdf"),
               [PDF_TEXT["store"]])
    own = registry.execute_result("extract_pdf_text", {"path": "attachments/doc.pdf"})

    assert missing.text.startswith("⚠️ FILE_NOT_FOUND: attachments/doc.pdf"), missing.text
    assert _pdf_owner(own.text) == ["store"], own.text


def test_without_a_workspace_media_refuses_an_ordinary_relative_path_but_finds_a_staged_attachment(world):
    pytest.importorskip("pypdf")
    ctx = ToolContext(repo_dir=world.serving, drive_root=world.data, is_direct_chat=True, project_id="room",
                      task_id="room-task",
                      task_metadata={"_project_room_note": "This room's working folder is unavailable."})
    registry = ToolRegistry(repo_dir=world.serving, drive_root=world.data)
    registry.set_context(ctx)
    _decoys(world)
    _write_pdf(_dir(task_artifact_dir_path(world.data, "room-task", create=True) / "attachments" / "doc.pdf"),
               [PDF_TEXT["store"]])

    refused = registry.execute_result("extract_pdf_text", {"path": PDF})
    attachment = registry.execute_result("extract_pdf_text", {"path": "attachments/doc.pdf"})
    upload = registry.execute_result("extract_pdf_text", {"path": str(world.data / "uploads" / PDF)})

    assert (refused.status, refused.code) == ("error", "TOOL_ARG_ERROR"), refused
    assert "working folder is unavailable" in refused.text and "absolute path" in refused.text
    assert _pdf_owner(attachment.text) == ["store"] and _pdf_owner(upload.text) == ["upload"]


def test_media_guards_judge_the_workspace_resolved_path(world):
    pytest.importorskip("pypdf")
    registry, project = _registry(world, _external_project)
    _write_pdf(_dir(project / "sealed.pdf"), [PDF_TEXT["task"]])
    contract = _protected([project / "sealed.pdf"], ["read_bytes"])
    registry._ctx.task_contract, registry._ctx.task_metadata = contract, {"task_contract": contract}
    _write_pdf(_dir(world.serving / "sealed.pdf"), [PDF_TEXT["decoy"]])
    _write_pdf(_dir(world.data / "uploads" / "sealed.pdf"), [PDF_TEXT["upload"]])

    sealed = registry.execute_result("extract_pdf_text", {"path": "sealed.pdf"})
    assert "PATH_BLOCKED" in sealed.text and _pdf_owner(sealed.text) == [], sealed

    _write_pdf(_dir(world.outside / "real.pdf"), [PDF_TEXT["other"]])
    _symlink_or_skip(project / "link.pdf", world.outside / "real.pdf")
    escaped = registry.execute_result("extract_pdf_text", {"path": "link.pdf"})
    assert escaped.text.startswith("⚠️ PATH_BLOCKED: link.pdf is outside the allowed file roots"), escaped


def test_traversal_is_judged_from_the_workspace(world):
    registry, _project = _registry(world, _external_outside_every_root)
    _write_pdf(_dir(world.outside / "secret.pdf"), [PDF_TEXT["other"]])
    _write_pdf(_dir(world.root / "secret.pdf"), [PDF_TEXT["decoy"]])  # what ``..`` names from the process directory

    refused = registry.execute_result("extract_pdf_text", {"path": "../secret.pdf"})

    assert refused.text.startswith("⚠️ PATH_BLOCKED: ../secret.pdf is outside the allowed file roots"), refused
