"""A second launch of another copy names the running one from that instance's own report (``launcher_background``).

The launcher shows this one native line before it activates the running window (``tests/test_launcher_background.py``).
``ps`` joins a live command line's argv without quoting, so copies whose paths hold spaces are the hard case.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import types

import launcher  # before per-test root rebinding, as tests/test_launcher_background.py does
from ouroboros import launcher_background as lb
from ouroboros import platform_layer

HEALTH = json.dumps({"status": "ok", "app_version": "7.9.0"}).encode()


def _copy(parent, name):
    """A packaged copy's bundle directory: it carries the bundle manifest."""
    bundle = parent / name / "Contents" / "Frameworks"
    bundle.mkdir(parents=True)
    (bundle / "repo_bundle_manifest.json").write_text("{}", encoding="utf-8")
    return bundle


def _running(tmp_path, monkeypatch, *argv):
    """The running instance: its PID in the lock and its command line as process_command spells it."""
    lock, port = tmp_path / "ouroboros.pid", tmp_path / "port"
    lock.write_text("4242", encoding="utf-8")
    port.write_text("8765", encoding="utf-8")
    command = subprocess.list2cmdline(argv) if os.name == "nt" else " ".join(argv)  # ps keeps no quotes
    monkeypatch.setattr(platform_layer, "process_command", lambda pid: command if pid == 4242 else "")
    monkeypatch.setattr(lb.urllib.request, "urlopen", lambda url, timeout: contextlib.nullcontext(io.BytesIO(HEALTH)))
    for name in ("APPIMAGE", "APPDIR"):
        monkeypatch.delenv(name, raising=False)
    return lock, port


def _handed_off(bundle):
    """The source handoff's argv: the copy's shell, launcher.py, its options, ``--seed-bundle <bundle>`` last."""
    return str(bundle.parent / "MacOS" / "ouroboros-shell"), "/repo/launcher.py", "--launch-intent", "owner", \
        "--seed-bundle", str(bundle)


def test_the_second_copy_line_names_the_running_instance_by_its_own_report(tmp_path, monkeypatch):
    """Its process names its copy and its server its version; the same copy, an unreadable lock or this app's
    own AppImage (whose per-launch mount names no copy) say nothing; another AppImage app's inherited pair does
    not count."""
    old, new = _copy(tmp_path / "Desktop", "Ouroboros.app"), _copy(tmp_path / "Applications", "Ouroboros.app")
    lock, port = _running(tmp_path, monkeypatch, *_handed_off(old))
    assert lb.other_copy_line(lock, port, new) == (
        f"Ouroboros 7.9.0 is already running from {old.parents[1]}. Quit it, then open this copy.")
    assert lb.other_copy_line(lock, port, old) == "", "the same copy: unchanged"
    lock.write_text("not a pid", encoding="utf-8")
    assert lb.other_copy_line(lock, port, new) == ""
    lock.write_text("4242", encoding="utf-8")
    monkeypatch.setenv("APPIMAGE", str(tmp_path / "Ouroboros.AppImage"))
    monkeypatch.setenv("APPDIR", str(new.parents[1]))  # this copy's own AppImage mount
    assert lb.other_copy_line(lock, port, new) == ""
    monkeypatch.setenv("APPDIR", str(tmp_path / ".mount_Editor"))  # another AppImage app's, from its terminal
    assert lb.other_copy_line(lock, port, new).startswith(f"Ouroboros 7.9.0 is already running from {old.parents[1]}.")


def test_copies_whose_paths_hold_spaces_are_told_apart_by_their_whole_paths(tmp_path, monkeypatch):
    """Finder's "Keep Both" copy or a second disk image's ``/Volumes/Ouroboros 1``, a deleted copy included."""
    first, second = _copy(tmp_path / "My Apps", "Ouroboros.app"), _copy(tmp_path / "My Apps", "Ouroboros 2.app")
    lock, port = _running(tmp_path, monkeypatch, *_handed_off(second))
    assert lb.other_copy_line(lock, port, second) == "", "the same copy got a false line naming a truncated path"
    line = f"Ouroboros 7.9.0 is already running from {second.parents[1]}. Quit it, then open this copy."
    assert lb.other_copy_line(lock, port, first) == line
    (second / "repo_bundle_manifest.json").unlink()
    second.rmdir()  # the running copy was moved to the Trash: its own report still names it
    assert lb.other_copy_line(lock, port, first) == line


def test_a_source_run_names_no_other_copy(tmp_path, monkeypatch):
    """Without a bundle manifest this launch is no packaged copy, so no running instance is another one."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    lock, port = _running(tmp_path, monkeypatch, "/usr/bin/python3", str(checkout / "launcher.py"))
    assert lb.other_copy_line(lock, port, checkout) == ""


def test_a_copy_whose_path_ends_like_this_ones_is_another_copy(tmp_path, monkeypatch):
    """This copy in /Applications, the running one in ~/Applications: only a whole path (at the start of the command,
    after a space or a quote) makes the same copy, never a path that merely ends like this one."""
    this = _copy(tmp_path / "Applications", "Ouroboros.app")
    other = pathlib.Path(f"/Users/me{this}")  # this copy's path below another folder
    lock, port = _running(tmp_path, monkeypatch, *_handed_off(other))
    assert lb.other_copy_line(lock, port, this) == (
        f"Ouroboros 7.9.0 is already running from {other.parents[1]}. Quit it, then open this copy.")


def test_the_notice_wraps_a_long_unbroken_path(monkeypatch):
    """The 440-wide notice holds a full path, and an AppTranslocation path is one unbroken ~110-character word."""
    pages = []
    webview = types.ModuleType("webview")
    webview.create_window = lambda *args, **kwargs: pages.append(kwargs["html"])
    webview.start = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "webview", webview)
    monkeypatch.setattr(launcher, "paint_room", lambda window: window)
    launcher._already_running_window("Ouroboros 7.9.0 is already running from /private/var/folders/x/AppTranslocation/"
                                     "0123456789ABCDEF/d/Ouroboros.app. Quit it, then open this copy.")
    rule = re.search(r"\.window p \{([^}]*)\}", pages[0])
    assert rule and "overflow-wrap: anywhere" in rule.group(1), "a long path ran out of the window"
