"""CPL4-C21 pins (owner batch №8, 6A): agent media ages out, owner uploads never.

``uploads/screenshots/`` and ``uploads/views/`` are agent-generated re-view
copies and follow GC retention; owner attachments in the ``uploads/`` root
are owner-explicit-delete only and must never be touched by any sweep.
"""

from __future__ import annotations

import os
import time

from ouroboros.server_maintenance import prune_agent_media_uploads

_OLD = time.time() - 400 * 86400


def test_old_agent_media_pruned_owner_uploads_untouched(tmp_path):
    shots = tmp_path / "uploads" / "screenshots"
    views = tmp_path / "uploads" / "views"
    shots.mkdir(parents=True)
    views.mkdir(parents=True)
    old_shot = shots / "20260101T000000.png"
    old_shot.write_bytes(b"png")
    os.utime(old_shot, (_OLD, _OLD))
    fresh_shot = shots / "20260901T000000.png"
    fresh_shot.write_bytes(b"png")
    old_view = views / "20260101T000000_diagram.png"
    old_view.write_bytes(b"png")
    os.utime(old_view, (_OLD, _OLD))
    owner_upload = tmp_path / "uploads" / "abc123_report.pdf"
    owner_upload.write_bytes(b"pdf")
    os.utime(owner_upload, (_OLD, _OLD))

    report = prune_agent_media_uploads(tmp_path)

    assert not old_shot.exists() and not old_view.exists()
    assert fresh_shot.exists()
    assert owner_upload.exists()  # owner authority: never swept, any age
    assert report == {"removed": 2, "kept": 1, "skipped": 0, "errors": 0}


def test_missing_media_dirs_are_a_noop(tmp_path):
    assert prune_agent_media_uploads(tmp_path) == {
        "removed": 0, "kept": 0, "skipped": 0, "errors": 0,
    }


def test_the_sweep_never_follows_a_symlink_out_of_the_drive(tmp_path):
    """Audit #15-13: ``is_file()``/``stat()`` FOLLOW symlinks, so an age sweep
    of the drive would unlink old files anywhere the link pointed. Both shapes
    — a symlinked family directory and a single symlink inside a real one —
    must be skipped, not followed."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    outside_file = outside / "someone_elses.png"
    outside_file.write_bytes(b"png")
    os.utime(outside_file, (_OLD, _OLD))

    drive = tmp_path / "drive"
    shots = drive / "uploads" / "screenshots"
    shots.mkdir(parents=True)
    (drive / "uploads" / "views").symlink_to(outside, target_is_directory=True)
    escaping = shots / "20260101T000000.png"
    escaping.symlink_to(outside_file)
    real_old = shots / "20260101T000001.png"
    real_old.write_bytes(b"png")
    os.utime(real_old, (_OLD, _OLD))

    report = prune_agent_media_uploads(drive)

    assert outside_file.exists()  # the sweep never reached outside the drive
    assert escaping.is_symlink()  # the link itself is not an age-sweep target
    assert not real_old.exists()  # contained files still age out normally
    assert report == {"removed": 1, "kept": 0, "skipped": 2, "errors": 0}


def test_startup_prune_sweeps_run_the_media_prune():
    import inspect

    import ouroboros.server_maintenance as sm

    assert "prune_agent_media_uploads" in inspect.getsource(sm._startup_prune_sweeps)
