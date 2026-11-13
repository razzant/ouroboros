"""My map of my world (`memory/WORLD.md`): the host refreshes only its measured block, the
mind publishes the rest through ``update_world`` on the canonical root, and the context
presents the file as mine, not as a disposable profile."""
from __future__ import annotations

import hashlib
import json
import pathlib

from ouroboros import world_profiler as wprof


def _measured(stamp: str) -> str:
    return (f"{wprof.MEASURED_START}\n## System (measured at startup, {stamp})\n"
            f"- **OS**: test\n{wprof.MEASURED_END}")


def _fake_measured(monkeypatch, stamp: str) -> None:
    monkeypatch.setattr(wprof, "measured_block", lambda repo_dir="", data_dir="": _measured(stamp))


def test_refresh_replaces_only_the_measured_block_and_keeps_my_text(tmp_path, monkeypatch):
    _fake_measured(monkeypatch, "second")
    path = tmp_path / "WORLD.md"
    mine = "## What lives here\n- a Claude Code chat named X, inbox /tmp/cc-socks/1.sock (seen 2026-10-22)\n"
    path.write_text(f"{wprof.TITLE}\n\nintro\n\n{_measured('first')}\n\n{mine}", encoding="utf-8")

    assert wprof.refresh_world_profile(str(path)) is True
    text = path.read_text(encoding="utf-8")
    assert "measured at startup, second" in text and "first" not in text
    assert mine in text and text.count(wprof.MEASURED_START) == 1
    # Idempotent: an unchanged block writes nothing.
    assert wprof.refresh_world_profile(str(path)) is False


def test_refresh_fences_a_legacy_first_boot_profile_once_and_keeps_later_notes(tmp_path, monkeypatch):
    _fake_measured(monkeypatch, "now")
    path = tmp_path / "WORLD.md"
    path.write_text(
        "# WORLD.md — Environment Profile\n\nThis is where I currently exist. It defines my hardware, OS, and local "
        "constraints.\n\n## System\n- **OS**: old\n\n## Available Tools\nThe following binaries are available in my "
        "PATH:\n`git`\n\n## File System Rules\n- `/r` contains my codebase.\n\n"
        "*(Generated automatically on first boot)*\n\n## My notes\n- kept\n", encoding="utf-8")

    assert wprof.refresh_world_profile(str(path)) is True
    text = path.read_text(encoding="utf-8")
    assert text.startswith(wprof.TITLE) and text.count(wprof.MEASURED_START) == 1
    assert "- **OS**: old" not in text and "Generated automatically" not in text
    assert "## My notes\n- kept" in text
    assert wprof.refresh_world_profile(str(path)) is False


def test_refresh_keeps_unrecognized_text_whole_and_tolerates_crlf(tmp_path, monkeypatch):
    _fake_measured(monkeypatch, "now")
    path = tmp_path / "WORLD.md"
    path.write_bytes(b"# Mine\r\n\r\n- note one\r\n- note two\r\n")

    assert wprof.refresh_world_profile(str(path)) is True
    text = path.read_bytes().decode("utf-8")
    assert text.startswith("# Mine") and wprof.MEASURED_START in text
    assert "- note one\r\n- note two\r\n" in text

    # A fenced block whose inside carries CRLF is still one block, and the separators around it stay CRLF.
    path.write_bytes(b"# Mine\r\n\r\n" + _measured("old").replace("\n", "\r\n").encode() + b"\r\n\r\n- note\r\n")
    assert wprof.refresh_world_profile(str(path)) is True
    text = path.read_bytes().decode("utf-8")
    assert text.count(wprof.MEASURED_START) == 1 and "old" not in text
    assert text.startswith("# Mine\r\n\r\n") and text.endswith(wprof.MEASURED_END + "\r\n\r\n- note\r\n")


def test_a_stock_profile_with_one_mind_line_is_kept_whole(tmp_path, monkeypatch):
    _fake_measured(monkeypatch, "now")
    stock = ("# WORLD.md — Environment Profile\n\n## System\n- **OS**: old\n\n## Available Tools\n"
             "The following binaries are available in my PATH:\n`git, python3`\n\n## File System Rules\n"
             "- `/r` contains my codebase.\n- `/d` contains my memory, state, and logs.\n")
    footer = "\n*(Generated automatically on first boot)*\n"
    # Lines that only LOOK like generated ones are the mind's: a root-shaped bullet with other words, a
    # lone backticked command, a bullet that reads like a tool line, a heading that extends a stock one.
    for mind_line in ("- `/data/agents` contains my owner's other agent sessions; use their documented inbox.",
                      "`peer-review send --socket /tmp/peer.sock --as Ouroboros`",
                      "- `/tmp/inbox.sock` is the channel I found",
                      "## System notes\n- the Pi answers on ssh"):
        path = tmp_path / "WORLD.md"
        path.write_text(stock + mind_line + "\n" + footer, encoding="utf-8")
        assert wprof.refresh_world_profile(str(path)) is True
        text = path.read_text(encoding="utf-8")
        assert mind_line in text and "- **OS**: old" in text, mind_line
    # The untouched stock profile itself is replaced.
    path.write_text(stock + footer, encoding="utf-8")
    assert wprof.refresh_world_profile(str(path)) is True
    assert "- **OS**: old" not in path.read_text(encoding="utf-8")


def test_the_first_fencing_replaces_nothing_when_its_pre_image_cannot_be_journaled(tmp_path, monkeypatch):
    _fake_measured(monkeypatch, "now")
    import ouroboros.utils as utils_module

    monkeypatch.setattr(utils_module, "append_jsonl", lambda *_a, **_k: False)
    path = tmp_path / "WORLD.md"
    legacy = "# WORLD.md — Environment Profile\n\n## System\n- **OS**: old\n\n*(Generated automatically on first boot)*\n"
    path.write_text(legacy, encoding="utf-8")
    assert wprof.refresh_world_profile(str(path)) is False
    assert path.read_text(encoding="utf-8") == legacy


def test_refresh_generates_a_missing_file_with_an_empty_map(tmp_path, monkeypatch):
    _fake_measured(monkeypatch, "now")
    path = tmp_path / "WORLD.md"
    assert wprof.refresh_world_profile(str(path)) is True
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# WORLD.md") and "## What lives here" in text and wprof.MEASURED_END in text


def test_generate_names_the_roots_and_calls_the_probe_a_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(wprof, "_tools_found", lambda: ["git", "python3"])
    monkeypatch.setattr(wprof, "get_cpu_info", lambda: "test-cpu")
    monkeypatch.setattr(wprof, "get_system_memory", lambda: "1.0 GB")
    out = tmp_path / "WORLD.md"
    wprof.generate_world_profile(str(out), "/repo root", "/data root")
    text = out.read_text(encoding="utf-8")
    assert "`/repo root` contains my codebase" in text and "`/data root` contains my memory" in text
    assert "A probe of common binaries, not a census" in text and "`git, python3`" in text
    assert text.index(wprof.MEASURED_START) < text.index("## What lives here")


def test_with_measured_block_keeps_the_hosts_block_whatever_the_mind_sends():
    old = f"{wprof.TITLE}\n\n{_measured('host')}\n\n## What lives here\n- old note\n"
    # The mind dropped the fence: the host's block comes back after the title.
    rendered = wprof.with_measured_block(f"{wprof.TITLE}\n\n## What lives here\n- new note\n", old)
    assert rendered.startswith(wprof.TITLE) and "measured at startup, host" in rendered and "- new note" in rendered
    # The mind edited inside the fence: the host's block wins.
    rendered = wprof.with_measured_block(f"{wprof.TITLE}\n\n{_measured('forged')}\n\n- new note\n", old)
    assert "forged" not in rendered and "measured at startup, host" in rendered and rendered.count(wprof.MEASURED_START) == 1


def test_context_presents_the_map_as_mine(tmp_path):
    from ouroboros.context import build_memory_sections
    from ouroboros.memory import Memory

    (tmp_path / "memory").mkdir(parents=True)
    (tmp_path / "memory" / "WORLD.md").write_text("world-map-data", encoding="utf-8")
    combined = "\n\n".join(build_memory_sections(Memory(drive_root=tmp_path)))
    assert "world-map-data" in combined
    assert "update_world" in combined and "delete WORLD.md" not in combined


def test_delegate_start_discloses_that_a_run_sees_no_owner_sessions():
    from ouroboros.tools.delegate import get_tools

    schema = next(entry.schema for entry in get_tools() if entry.name == "delegate_start")
    assert "engine-scoped native state" in schema["description"] and "no courier" in schema["description"]
    assert "owner's own sessions" in schema["description"]


def test_write_tools_redirect_the_map_to_update_world():
    from ouroboros.tool_access import light_cognitive_or_root_redirect

    message = light_cognitive_or_root_redirect("edit_text", {"root": "runtime_data", "path": "memory/WORLD.md"})
    assert message and "update_world" in message
    assert light_cognitive_or_root_redirect("edit_text", {"root": "runtime_data", "path": "memory/notes.md"}) is None


def _world_ctx(canonical: pathlib.Path, drive: pathlib.Path):
    from ouroboros.tools.registry import ToolContext

    # A forked execution drive with the canonical budget root: the shape of a child task.
    return ToolContext(repo_dir=canonical / "repo", drive_root=drive, task_id="t1", current_chat_id=444,
                       task_metadata={"budget_drive_root": str(canonical)})


def test_update_world_publishes_on_the_canonical_root_with_revision_and_the_hosts_block(tmp_path, monkeypatch):
    from ouroboros.tools.control_runtime import _update_world

    canonical, drive = tmp_path / "canonical", tmp_path / "fork"
    (canonical / "memory").mkdir(parents=True)
    (drive / "memory").mkdir(parents=True)
    world = canonical / "memory" / "WORLD.md"
    world.write_text(f"{wprof.TITLE}\n\n{_measured('host')}\n\n## What lives here\n- old\n", encoding="utf-8")
    ctx = _world_ctx(canonical, drive)

    # An existing map needs its revision; a stale one is refused with the current revision.
    refused = _update_world(ctx, "# WORLD.md\n\n## What lives here\n- new\n")
    assert "revision_required" in refused
    import hashlib
    revision = hashlib.sha256(world.read_bytes()).hexdigest()
    stale = _update_world(ctx, "# WORLD.md\n\n## What lives here\n- new\n", expected_revision="0" * 64)
    assert "revision_conflict" in stale and revision in stale

    result = _update_world(ctx, "# WORLD.md\n\n## What lives here\n- new\n", expected_revision=revision)
    assert result.startswith("OK: world map updated") and "revision=" in result
    text = world.read_bytes().decode("utf-8")  # the journal keeps the file's bytes, line ends included
    assert "- new" in text and "- old" not in text and "measured at startup, host" in text
    # The forked drive stayed untouched: the map lives on the canonical root the context reads.
    assert not (drive / "memory" / "WORLD.md").exists()
    rows = [json.loads(line) for line in (canonical / "memory" / "world_journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and rows[-1]["new_content"] == text and rows[-1]["task_id"] == "t1"


def test_update_world_warns_above_the_context_budget_without_refusing(tmp_path, monkeypatch):
    from ouroboros.context_budget import WORLD_SECTION_BUDGET_CHARS
    from ouroboros.tools.control_runtime import _update_world

    _fake_measured(monkeypatch, "host")  # a missing file takes the host's block, never a live probe here
    canonical = tmp_path / "canonical"
    (canonical / "memory").mkdir(parents=True)
    ctx = _world_ctx(canonical, canonical)
    big = f"{wprof.TITLE}\n\n{_measured('host')}\n\n" + "x" * (WORLD_SECTION_BUDGET_CHARS + 10)
    result = _update_world(ctx, big)
    assert result.startswith("OK:") and "WORLD_SIZE_NOTICE" in result
    assert _update_world(ctx, "   ").startswith("⚠️ TOOL_ARG_ERROR: content is empty or whitespace only")
    # Edits inside the measured block are dropped, and the result says so.
    revision = hashlib.sha256((canonical / "memory" / "WORLD.md").read_bytes()).hexdigest()
    edited = _update_world(ctx, f"{wprof.TITLE}\n\n{_measured('forged')}\n\n- note\n", expected_revision=revision)
    assert edited.startswith("OK:") and "edits inside it were not applied" in edited
    assert "forged" not in (canonical / "memory" / "WORLD.md").read_text(encoding="utf-8")


def test_refresh_keeps_a_legacy_profile_the_mind_edited_and_journals_the_pre_image(tmp_path, monkeypatch):
    _fake_measured(monkeypatch, "now")
    path = tmp_path / "WORLD.md"
    legacy = ("# WORLD.md — Environment Profile\n\n## Channels\n- Claude chat X: socket /tmp/x\n\n## System\n"
              "- **OS**: old\n\n*(Generated automatically on first boot)*\n")
    path.write_text(legacy, encoding="utf-8")
    legacy_raw = path.read_bytes()  # the pre-image as stored (CRLF on Windows)

    assert wprof.refresh_world_profile(str(path)) is True
    text = path.read_text(encoding="utf-8")
    # Nothing the mind wrote inside the old generated span is dropped; the block is added.
    assert "- Claude chat X: socket /tmp/x" in text and "- **OS**: old" in text
    assert text.count(wprof.MEASURED_START) == 1 and text.startswith("# WORLD.md")
    rows = [json.loads(line) for line in (tmp_path / "world_journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["source_type"] == "legacy_wrap" and rows[-1]["old_content"].encode("utf-8") == legacy_raw


def test_insertion_keeps_an_indented_first_line(tmp_path, monkeypatch):
    _fake_measured(monkeypatch, "now")
    path = tmp_path / "WORLD.md"
    path.write_text("# Mine\n\n    code line\n", encoding="utf-8")
    assert wprof.refresh_world_profile(str(path)) is True
    assert "\n\n    code line\n" in path.read_text(encoding="utf-8")


def test_a_publication_during_measurement_survives_the_refresh(tmp_path, monkeypatch):
    from ouroboros.memory_publication import publish_memory_file

    path = tmp_path / "WORLD.md"
    path.write_text(f"{wprof.TITLE}\n\n{_measured('first')}\n\n- old note\n", encoding="utf-8")

    def measure(repo_dir="", data_dir=""):
        # A focus publishes while the host is still measuring: the refresh must splice what it reads under the lock.
        revision = hashlib.sha256(path.read_bytes()).hexdigest()
        published = publish_memory_file(path, revision, lambda old: old + "- late note\n",
                                        journal=tmp_path / "world_journal.jsonl",
                                        author={"task_id": "t", "focus": {}, "route": "r"}, source_type="task")
        assert published.ok
        return _measured("second")

    monkeypatch.setattr(wprof, "measured_block", measure)
    assert wprof.refresh_world_profile(str(path)) is True
    text = path.read_text(encoding="utf-8")
    assert "- late note" in text and "- old note" in text and "measured at startup, second" in text


def test_update_world_mirrors_the_fork_copy_and_names_the_map_on_a_stale_revision(tmp_path):
    from ouroboros.tools.control_runtime import _update_world

    canonical, drive = tmp_path / "canonical", tmp_path / "fork"
    for root in (canonical, drive):
        (root / "memory").mkdir(parents=True)
    initial = f"{wprof.TITLE}\n\n{_measured('host')}\n\n- old\n"
    (canonical / "memory" / "WORLD.md").write_text(initial, encoding="utf-8")
    (drive / "memory" / "WORLD.md").write_text(initial, encoding="utf-8")
    ctx = _world_ctx(canonical, drive)
    revision = hashlib.sha256((canonical / "memory" / "WORLD.md").read_bytes()).hexdigest()  # of the stored bytes

    assert _update_world(ctx, f"{wprof.TITLE}\n\n- new\n", expected_revision=revision).startswith("OK")
    canonical_text = (canonical / "memory" / "WORLD.md").read_text(encoding="utf-8")
    assert "- new" in canonical_text
    assert (drive / "memory" / "WORLD.md").read_text(encoding="utf-8") == canonical_text
    stale = _update_world(ctx, f"{wprof.TITLE}\n\n- newer\n", expected_revision="0" * 64)
    assert "revision_conflict" in stale and str(canonical / "memory" / "WORLD.md") in stale


def test_world_read_returns_the_revision_of_the_same_bytes(tmp_path):
    from ouroboros.tools.control_runtime import _update_world
    from ouroboros.tools.core_file_tools import _read_file

    (tmp_path / "memory").mkdir()
    raw = f"{wprof.TITLE}\n\n{_measured('host')}\n\n- note\n".encode()
    (tmp_path / "memory" / "WORLD.md").write_bytes(raw)
    ctx = _world_ctx(tmp_path, tmp_path)
    result = _read_file(ctx, "memory/WORLD.md", root="runtime_data")
    revision = hashlib.sha256(raw).hexdigest()
    assert result.startswith(f"revision={revision}\n")
    assert _update_world(ctx, f"{wprof.TITLE}\n\n- changed\n", revision).startswith("OK")


def test_measured_block_falls_back_to_username(monkeypatch):
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.setenv("USERNAME", "winuser")
    monkeypatch.setattr(wprof, "_tools_found", lambda: [])
    monkeypatch.setattr(wprof, "get_cpu_info", lambda: "c")
    monkeypatch.setattr(wprof, "get_system_memory", lambda: "m")
    assert "- **User**: winuser" in wprof.measured_block("/r", "/d")



def test_the_measured_block_names_the_app_copy_and_the_body_it_runs(tmp_path, monkeypatch):
    """The app is a replaceable host: a moved, translocated or newer app shows at the next start. The line names
    the body's version only: its commit rides the runtime section, and a commit must not rewrite this cached block."""
    import platform
    import subprocess
    import sys

    for name, value in (("_tools_found", lambda: []), ("get_cpu_info", lambda: "c"), ("get_system_memory", lambda: "m")):
        monkeypatch.setattr(wprof, name, value)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "VERSION").write_text("7.9.0\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x"],
                   cwd=repo, check=True)
    body, python = "body 7.9.0", platform.python_version()

    def host(block):
        return next(line for line in block.splitlines() if line.startswith("- **Host**: "))

    for name in ("OUROBOROS_BUNDLE_DIR", "APPIMAGE", "APPDIR"):
        monkeypatch.delenv(name, raising=False)
    assert host(wprof.measured_block(str(repo), "/d")) == f"- **Host**: source runtime (Python {python} at {sys.executable}); {body}"
    import os
    import types

    real_statvfs = getattr(os, "statvfs", None)
    for app, read_only, label in ((tmp_path / "Volumes" / "Disk" / "Ouroboros.app", False, ""),  # a writable volume
                                  (tmp_path / "AppTranslocation" / "X" / "d" / "Ouroboros.app", False, " (translocated)"),
                                  (tmp_path / "Volumes" / "Ouroboros 1" / "Ouroboros.app", True, " (read-only mount)")):
        bundle = app / "Contents" / "Frameworks"
        bundle.mkdir(parents=True)
        (bundle / "repo_bundle_manifest.json").write_text("{}", encoding="utf-8")
        monkeypatch.setenv("OUROBOROS_BUNDLE_DIR", str(bundle))
        monkeypatch.setenv("OUROBOROS_APP_VERSION", "7.10.0")
        flag = getattr(os, "ST_RDONLY", 1) if read_only else 0
        monkeypatch.setattr(os, "statvfs", (lambda _path, flag=flag: types.SimpleNamespace(f_flag=flag)) if real_statvfs
                            else (lambda _path: (_ for _ in ()).throw(OSError("unmeasured"))), raising=False)
        expected = label if real_statvfs or not read_only else ""  # an unmeasured mount gets no label
        assert host(wprof.measured_block(str(repo), "/d")) == (
            f"- **Host**: App 7.10.0 at {app}{expected} (bundled Python {python}); {body}")
    appimage = tmp_path / "Ouroboros.AppImage"
    monkeypatch.setenv("APPIMAGE", str(appimage))
    monkeypatch.setenv("APPDIR", str(tmp_path / ".mount_Editor"))  # another AppImage app's, from its terminal
    assert host(wprof.measured_block(str(repo), "/d")).startswith(f"- **Host**: App 7.10.0 at {app}")
    monkeypatch.setenv("APPDIR", str(app))  # this app's own mount
    assert host(wprof.measured_block(str(repo), "/d")).startswith(f"- **Host**: App 7.10.0 at {appimage}")

def test_startup_verification_refreshes_the_measured_block_in_place(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks as checks_module

    class _Env:
        def __init__(self, root):
            self.repo_dir = self.drive_root = root

        def drive_path(self, name):
            path = self.repo_dir / name
            path.mkdir(parents=True, exist_ok=True)
            return path

    env = _Env(tmp_path)
    for name in ("check_uncommitted_changes", "check_version_sync", "check_budget", "check_extension_health",
                 "check_stray_server_processes", "check_review_continuations"):
        monkeypatch.setattr(checks_module, name, lambda _env: ({"status": "ok"}, 0))
    monkeypatch.setattr(checks_module, "_reconcile_review_attempts_on_startup",
                        lambda _env: {"reconciled": [], "expired": []})
    monkeypatch.setattr(checks_module, "runtime_setting", lambda key, default="": "model-x")
    _fake_measured(monkeypatch, "fresh")
    memory_dir = env.drive_path("memory")
    (memory_dir / "identity.md").write_text("who I am", encoding="utf-8")
    (memory_dir / "scratchpad.md").write_text("", encoding="utf-8")
    world = memory_dir / "WORLD.md"
    world.write_text(f"{wprof.TITLE}\n\n{_measured('stale')}\n\n- my note\n", encoding="utf-8")

    checks_module.verify_system_state(env, "a" * 40)
    text = world.read_text(encoding="utf-8")
    assert "measured at startup, fresh" in text and "stale" not in text and "- my note" in text


def _map_with_block(tmp_path, stamp="host", tail="## What lives here\n- old note\n"):
    canonical = tmp_path / "canonical"
    (canonical / "memory").mkdir(parents=True)
    world = canonical / "memory" / "WORLD.md"
    world.write_text(f"{wprof.TITLE}\n\n{_measured(stamp)}\n\n{tail}", encoding="utf-8")
    return canonical, world


def _rev(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_an_edit_of_the_map_outside_the_block_keeps_the_block_byte_for_byte(tmp_path, monkeypatch):
    from ouroboros.tools.control_runtime import _update_world

    _fake_measured(monkeypatch, "fresh")  # a probe here would mean the host's block was regenerated
    canonical, world = _map_with_block(tmp_path)
    ctx = _world_ctx(canonical, canonical)
    block = _measured("host")
    for edits in ([{"old_str": "- old note", "new_str": "- a service on :8080 (seen today)"}],  # outside the block
                  [{"old_str": f"{wprof.MEASURED_END}\n\n## What lives here",  # the unchanged block as anchor context
                    "new_str": f"{wprof.MEASURED_END}\n\n## What lives here (mine)"}],
                  [{"old_str": "(seen today)", "new_str": f"(its notes mention the {wprof.MEASURED_START} fence inline)"}]):
        reply = _update_world(ctx, mode="edit", edits=edits, expected_revision=_rev(world))
        assert reply.startswith("OK: world map updated (edit"), reply
        assert world.read_text(encoding="utf-8").count(block) == 1
    assert "## What lives here (mine)\n- a service on :8080 (its notes mention" in world.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in (canonical / "memory" / "world_journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["mode"] for row in rows] == ["edit"] * 3 and all(row["old_content"] and row["new_content"] for row in rows)


def test_an_edit_that_touches_the_hosts_block_or_brings_a_marker_is_refused_whole(tmp_path, monkeypatch):
    from ouroboros.tools.control_runtime import _update_world

    _fake_measured(monkeypatch, "fresh")
    canonical, world = _map_with_block(tmp_path)
    ctx = _world_ctx(canonical, canonical)
    before = world.read_bytes()
    for edits, said in (
            ([{"old_str": "- **OS**: test", "new_str": "- **OS**: mine"}], "item 1 overlaps it"),  # inside the block
            ([{"old_str": "- old note", "new_str": "- fine"},  # cuts the closing marker
              {"old_str": f"{wprof.MEASURED_END}\n\n##", "new_str": "##"}], "item 2 overlaps it"),
            ([{"old_str": "- old note", "new_str": f"- old note\n{wprof.MEASURED_START}\n- hijack\n{wprof.MEASURED_END}"}],
             "item 1's new_str brings a marker; the result has a marker line outside it at line 10, 12"),
            ([{"old_str": "- old note", "new_str": f"- old note\n{wprof.MEASURED_END}"}],  # a lone marker: no complete pair
             "the result has a marker line outside it at line 10")):
        reply = _update_world(ctx, mode="edit", edits=edits, expected_revision=_rev(world))
        assert reply.startswith("⚠️ TOOL_ERROR: host_block: the host's measured block (lines 3-6 of the current map)")
        assert said in reply, reply
        assert world.read_bytes() == before
    assert not (canonical / "memory" / "world_journal.jsonl").exists()


def test_an_edit_of_an_unfenced_legacy_map_gets_the_block_as_an_overwrite_does(tmp_path, monkeypatch):
    from ouroboros.tools.control_runtime import _update_world

    _fake_measured(monkeypatch, "fresh")
    canonical = tmp_path / "canonical"
    (canonical / "memory").mkdir(parents=True)
    world = canonical / "memory" / "WORLD.md"
    world.write_text("# My map\n\n- a note\n", encoding="utf-8")
    ctx = _world_ctx(canonical, canonical)
    reply = _update_world(ctx, mode="edit", old_str="- a note", content="- a revised note", expected_revision=_rev(world))
    assert reply.startswith("OK:"), reply
    assert world.read_text(encoding="utf-8") == f"# My map\n\n{_measured('fresh')}\n\n- a revised note\n"
    marker = _update_world(ctx, mode="edit", old_str="- a revised note", expected_revision=_rev(world),
                           content=f"{wprof.MEASURED_START}\n- mine\n{wprof.MEASURED_END}")
    assert "host_block" in marker and "marker line outside it" in marker


def test_revert_takes_the_versions_own_part_with_the_hosts_current_block(tmp_path, monkeypatch):
    from ouroboros.tools.control_runtime import _update_world

    canonical, world = _map_with_block(tmp_path, "monday")
    ctx = _world_ctx(canonical, tmp_path / "fork")
    (tmp_path / "fork" / "memory").mkdir(parents=True)
    (tmp_path / "fork" / "memory" / "WORLD.md").write_text("an older copy", encoding="utf-8")
    first = _rev(world)
    assert _update_world(ctx, mode="edit", old_str="- old note", content="- replaced note",
                         expected_revision=first).startswith("OK:")
    _fake_measured(monkeypatch, "tuesday")  # the host refreshes its block at the next start: no row
    assert wprof.refresh_world_profile(str(world)) is True
    refreshed = _rev(world)
    reply = _update_world(ctx, mode="revert", version=first, expected_revision=refreshed, reason="the note was right")
    text = world.read_text(encoding="utf-8")
    assert text == f"{wprof.TITLE}\n\n{_measured('tuesday')}\n\n## What lives here\n- old note\n"
    assert _rev(world) != first and "so the new revision is not that version's" in reply, reply
    assert f"Replaced {refreshed}: no history row wrote it (a change outside the writer, such as the host's refresh" in reply
    row = json.loads((canonical / "memory" / "world_journal.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert (row["mode"], row["reverted_from"], row["new_sha256"]) == ("revert", first, _rev(world))
    assert (tmp_path / "fork" / "memory" / "WORLD.md").read_text(encoding="utf-8") == text  # the fork copy follows
    again = _update_world(ctx, mode="revert", version=first, expected_revision=_rev(world))
    assert again.startswith("OK: world map unchanged (revert)"), again  # judged after the splice


def test_revert_to_a_stock_legacy_profile_drops_its_generated_lines(tmp_path, monkeypatch):
    from ouroboros.tools.control_runtime import _update_world

    stock = ("# WORLD.md — Environment Profile\n\n## System\n- **OS**: old\n\n"
             "*(Generated automatically on first boot)*\n\n## My notes\n- kept\n")
    canonical, world = _map_with_block(tmp_path)
    ctx = _world_ctx(canonical, canonical)
    journal = canonical / "memory" / "world_journal.jsonl"
    journal.write_text(json.dumps({"ts": "2026-01-01T00:00:00Z", "focus": {"role": "host"}, "old_sha256": "",
                                   "new_sha256": hashlib.sha256(stock.encode()).hexdigest(), "new_content": stock,
                                   "old_content": ""}) + "\n", encoding="utf-8")
    reply = _update_world(ctx, mode="revert", version=hashlib.sha256(stock.encode()).hexdigest(),
                          expected_revision=_rev(world))
    text = world.read_text(encoding="utf-8")
    assert reply.startswith("OK:") and "- **OS**: old" not in text and "Generated automatically" not in text, reply
    assert text.count(wprof.MEASURED_START) == 1 and _measured("host") in text and "## My notes\n- kept" in text
