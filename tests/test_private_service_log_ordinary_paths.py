"""#1868 R5: the ordinary file boundary requires no service-specific context.

Home synthetic checks (a source deliverable; run isolated on GitHub Actions, never
on the owner's Mac). Only ordinary effects are exercised: a regular file stays
deliverable and readable through contexts that carry no drive, a synthetic private
source stays out of every structured answer, and a confined ordinary read keeps its
historical bare call shape while the service-log keywords stay opt-in. No real
logs, no services, no containers, no product or tool dispatch.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import artifacts, confined_files
from ouroboros.tool_access_paths import is_private_service_log_path
from ouroboros.tools.core_artifacts import _delivery_path, _send_photo, _send_video


def _png(tmp_path, name="frame.png"):
    path = tmp_path / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 120)
    return path


def _chat_ctx(**extra):
    """The historical ordinary delivery caller: a chat context without any service data."""
    return SimpleNamespace(current_chat_id=1, browser_state=None, pending_events=[], **extra)


def _synthetic_private_sources(drive):
    """Both recognized private-store layouts under a bound root, with synthetic bytes only."""
    modern = drive / "private_service_logs" / "t1"
    modern.mkdir(parents=True, exist_ok=True)
    launch = modern / "svc@launch-0000000000000000.log"
    launch.write_bytes(b"synthetic private diagnostic")
    legacy = drive / "services" / "web"
    legacy.mkdir(parents=True, exist_ok=True)
    old = legacy / "out.log"
    old.write_bytes(b"synthetic legacy diagnostic")
    return launch, old


def test_ordinary_names_outside_a_bound_root_are_not_private(tmp_path):
    """The classifier judges an addressed store, never a name found in an ordinary workspace."""
    lookalike = tmp_path / "workspace" / "services" / "web" / "out.log"
    lookalike.parent.mkdir(parents=True)
    lookalike.write_bytes(b"a workspace file that only looks the part")
    assert not is_private_service_log_path(lookalike, tmp_path / "data")
    drive = tmp_path / "data"
    launch, old = _synthetic_private_sources(drive)
    assert is_private_service_log_path(launch, drive)
    assert is_private_service_log_path(old, drive)


def test_photo_and_video_delivery_need_no_drive_root(tmp_path):
    png = _png(tmp_path)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 120)
    ctx = _chat_ctx()  # no drive_root, no task metadata: an ordinary caller
    assert "OK" in _send_photo(ctx, file_path=str(png))
    assert "OK" in _send_video(ctx, file_path=str(video))
    assert [event["type"] for event in ctx.pending_events] == ["send_photo", "send_video"]


def test_delivery_with_a_drive_still_reads_ordinary_files(tmp_path):
    drive = tmp_path / "data"
    drive.mkdir()
    ctx = _chat_ctx(drive_root=drive, task_id="t1", task_metadata={})
    assert "OK" in _send_photo(ctx, file_path=str(_png(tmp_path)))
    assert ctx.pending_events[-1]["type"] == "send_photo"


def test_delivery_blocks_a_synthetic_private_source(tmp_path):
    drive = tmp_path / "data"
    launch, old = _synthetic_private_sources(drive)
    for source in (launch, old):
        ctx = _chat_ctx(drive_root=drive, task_id="t1", task_metadata={})
        fp, refusal = _delivery_path(ctx, str(source))
        assert fp is None and "Private service diagnostics" in refusal
        assert "Private service diagnostics" in _send_photo(ctx, file_path=str(source))
        assert ctx.pending_events == []  # nothing structured carried the private source


def test_staging_and_capture_refuse_a_synthetic_private_source(tmp_path):
    drive = tmp_path / "data"
    launch, _old = _synthetic_private_sources(drive)
    rows = artifacts.stage_task_attachments(drive, "t1", [{"path": str(launch)}])
    assert rows[0]["status"] == "rejected" and rows[0]["reason"] == "private_service_log"
    ctx = SimpleNamespace(drive_root=drive, task_id="t1")
    assert artifacts.copy_file_to_task_artifacts(ctx, launch) is None


@pytest.mark.parametrize("root", ["runtime_data", "task_drive"])
def test_single_level_listing_uses_the_bound_drive_not_config_root(tmp_path, monkeypatch, root):
    from ouroboros import config
    from ouroboros.tools.core_file_tools import _list_files, _read_file
    from ouroboros.tools.registry import ToolContext

    drive = tmp_path / "alternate"
    actual = drive if root == "runtime_data" else drive / "task_drives" / "task-1"
    private, legacy = _synthetic_private_sources(actual)
    public = legacy.with_name("readme.txt")
    public.write_text("not a service log", encoding="utf-8")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "other-installation")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=drive, task_id="task-1")
    assert "private_service_logs/" not in json.loads(_list_files(ctx, root=root))
    rows = json.loads(_list_files(ctx, root=root, path="services/web"))
    assert rows == ["services/web/readme.txt"]
    for source in private, legacy:
        assert "private service diagnostics" in _read_file(ctx, root=root, path=source.relative_to(actual).as_posix())


def test_confined_ordinary_open_keeps_the_bare_call_shape(tmp_path, monkeypatch):
    """The seam the failing CI suites wrap: an ordinary ``open_confined_file`` call
    reaches ``ConfinedDirectory.open`` positionally, and only a quiescent read adds
    the service-log keyword."""
    payload = tmp_path / "notes.txt"
    payload.write_bytes(b"ordinary bytes")
    real_open = confined_files.ConfinedDirectory.open
    calls = []

    def recording(self, name, **kwargs):
        calls.append(kwargs)
        return real_open(self, name, **kwargs)

    monkeypatch.setattr(confined_files.ConfinedDirectory, "open", recording)
    handle, _observed = confined_files.open_confined_file(tmp_path, ("notes.txt",))
    with handle:
        assert handle.read() == b"ordinary bytes"
    handle, _observed = confined_files.open_confined_file(tmp_path, ("notes.txt",), quiescent=True)
    with handle:
        pass
    assert calls == [{}, {"quiescent": True}]

    def narrow(self, name):  # the historical wrapper shape of the failing suites
        return real_open(self, name)

    monkeypatch.setattr(confined_files.ConfinedDirectory, "open", narrow)
    handle, _observed = confined_files.open_confined_file(tmp_path, ("notes.txt",))
    with handle:
        assert handle.read() == b"ordinary bytes"
