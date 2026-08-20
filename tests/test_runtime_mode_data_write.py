"""The data-tool fence: what ``_data_write`` and ``_data_read`` may touch under the drive.

Split verbatim out of ``tests/test_runtime_mode_elevation.py`` by theme. This module
owns the refusal of writes that resolve onto ``SETTINGS_PATH`` — including symlink,
env-override and case-variant spellings — the skill owner-state and self-authored
marker fences, and the reads those fences still allow.

Hermetic — no network, no supervisor boot. Uses temp dirs for ``DATA_DIR`` /
``SETTINGS_PATH`` overrides via monkeypatching ``ouroboros.config`` module-level
constants.
"""

from __future__ import annotations

import json

import pytest


from tests._runtime_mode_elevation_shared import (
    _make_drive_ctx,
)


# ---------------------------------------------------------------------------
# 2. _data_write block on settings.json
# ---------------------------------------------------------------------------


def test_data_write_blocks_settings_json(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.core import _data_write

    drive_root = tmp_path / "data"
    drive_root.mkdir()
    settings_path = drive_root / "settings.json"
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)

    ctx = _make_drive_ctx(tmp_path)
    result = _data_write(ctx, "settings.json", json.dumps({"OUROBOROS_RUNTIME_MODE": "pro"}))
    assert "DATA_WRITE_BLOCKED" in result
    assert "settings.json" in result
    # File must NOT have been written.
    assert not settings_path.exists()


def test_data_write_blocks_skill_grants_json(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.core import _data_write

    drive_root = tmp_path / "data"
    drive_root.mkdir()
    monkeypatch.setattr(cfg, "DATA_DIR", drive_root, raising=True)

    ctx = _make_drive_ctx(tmp_path)
    result = _data_write(
        ctx,
        "state/skills/weather/grants.json",
        json.dumps({"granted_keys": ["OPENROUTER_API_KEY"]}),
    )
    assert "DATA_WRITE_BLOCKED" in result
    assert "skill review" in result
    assert not (drive_root / "state" / "skills" / "weather" / "grants.json").exists()


def test_data_read_supports_line_ranges(tmp_path):
    from ouroboros.tools.core_file_tools import _data_read

    ctx = _make_drive_ctx(tmp_path)
    target = ctx.drive_root / "skills" / "external" / "demo" / "notes.txt"
    target.parent.mkdir(parents=True)
    target.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    result = _data_read(ctx, "skills/external/demo/notes.txt", start_line=2, max_lines=2)

    assert "lines 2–3 of 4" in result
    assert "two\nthree\n" in result
    assert "one" not in result


def test_data_read_does_not_slice_memory_by_default(tmp_path):
    from ouroboros.tools.core_file_tools import _data_read

    ctx = _make_drive_ctx(tmp_path)
    target = ctx.drive_root / "memory" / "identity.md"
    target.parent.mkdir(parents=True)
    body = "\n".join(f"line-{idx}" for idx in range(2105)) + "\n"
    target.write_text(body, encoding="utf-8")

    result = _data_read(ctx, "memory/identity.md")

    assert result == body
    assert "lines 1–2000" not in result


def test_data_read_cognitive_bad_line_args_are_tolerant(tmp_path):
    from ouroboros.tools.core_file_tools import _data_read

    ctx = _make_drive_ctx(tmp_path)
    target = ctx.drive_root / "memory" / "identity.md"
    target.parent.mkdir(parents=True)
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    result = _data_read(ctx, "memory/identity.md", start_line="abc", max_lines="bad")

    assert result == "alpha\nbeta\n"


def test_data_write_marks_new_external_skill_self_authored(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.core import _data_write

    drive_root = tmp_path / "data"
    drive_root.mkdir()
    monkeypatch.setattr(cfg, "DATA_DIR", drive_root, raising=True)
    ctx = _make_drive_ctx(tmp_path)
    ctx.current_chat_id = 123
    ctx.task_id = "task-1"

    result = _data_write(
        ctx,
        "skills/external/demo/SKILL.md",
        "---\nname: demo\ntype: instruction\n---\nbody\n",
    )

    assert result.startswith("OK:")
    marker = drive_root / "skills" / "external" / "demo" / ".self_authored.json"
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data["origin"] == "self_authored"
    assert data["chat_id"] == 123
    assert data["task_id"] == "task-1"
    state_marker = drive_root / "state" / "skills" / "demo" / "self_authored.json"
    assert json.loads(state_marker.read_text(encoding="utf-8"))["task_id"] == "task-1"


def test_malformed_self_authored_marker_is_not_trusted(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.skill_loader import is_self_authored_skill_dir

    drive_root = tmp_path / "data"
    skill_dir = drive_root / "skills" / "external" / "demo"
    state_dir = drive_root / "state" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    (skill_dir / ".self_authored.json").write_text('{"schema_version":"x","origin":"self_authored"}', encoding="utf-8")
    (state_dir / "self_authored.json").write_text('{"schema_version":1,"origin":"self_authored"}', encoding="utf-8")
    monkeypatch.setattr(cfg, "DATA_DIR", drive_root, raising=True)

    assert is_self_authored_skill_dir(skill_dir, drive_root=drive_root) is False


def test_data_write_blocks_self_authored_state_marker(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.core import _data_write

    drive_root = tmp_path / "data"
    drive_root.mkdir()
    monkeypatch.setattr(cfg, "DATA_DIR", drive_root, raising=True)
    ctx = _make_drive_ctx(tmp_path)

    result = _data_write(ctx, "state/skills/demo/self_authored.json", '{"origin":"self_authored"}')

    assert "DATA_WRITE_BLOCKED" in result
    assert not (drive_root / "state" / "skills" / "demo" / "self_authored.json").exists()


def test_data_write_blocks_unseeded_native_payload(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.core import _data_write

    drive_root = tmp_path / "data"
    drive_root.mkdir()
    monkeypatch.setattr(cfg, "DATA_DIR", drive_root, raising=True)
    ctx = _make_drive_ctx(tmp_path)

    result = _data_write(
        ctx,
        "skills/native/demo/SKILL.md",
        "---\nname: demo\ntype: instruction\n---\nbody\n",
    )

    assert "DATA_WRITE_BLOCKED" in result
    assert "data/skills/native" in result
    assert not (drive_root / "skills" / "native" / "demo" / "SKILL.md").exists()


def test_data_write_blocks_serialized_content_object(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.core import _data_write

    drive_root = tmp_path / "data"
    drive_root.mkdir()
    monkeypatch.setattr(cfg, "DATA_DIR", drive_root, raising=True)
    ctx = _make_drive_ctx(tmp_path)

    result = _data_write(ctx, "skills/external/demo/plugin.py", "{'content': 'print(1)\\n'}")

    assert "DATA_WRITE_BLOCKED" in result
    assert "serialized tool result" in result


def test_str_replace_blocks_self_authored_marker(tmp_path, monkeypatch):
    from ouroboros.tools.git import _str_replace_editor

    ctx = _make_drive_ctx(tmp_path)
    marker = ctx.drive_root / "skills" / "external" / "demo" / ".self_authored.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"origin":"self_authored"}\n', encoding="utf-8")

    result = _str_replace_editor(
        ctx,
        "skills/external/demo/.self_authored.json",
        "self_authored",
        "evil",
    )

    assert "STR_REPLACE_BLOCKED" in result
    assert "self_authored" in marker.read_text(encoding="utf-8")


@pytest.mark.parametrize("filename", [
    "review.json", "review_history.jsonl", "accepted_rebuttals.json", "enabled.json", "clawhub.json",
])
def test_data_write_blocks_skill_trust_state_json(filename, tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.core import _data_write

    drive_root = tmp_path / "data"
    drive_root.mkdir()
    monkeypatch.setattr(cfg, "DATA_DIR", drive_root, raising=True)

    ctx = _make_drive_ctx(tmp_path)
    result = _data_write(
        ctx,
        f"state/skills/weather/{filename}",
        json.dumps({"status": "pass", "enabled": True}),
    )
    assert "DATA_WRITE_BLOCKED" in result
    assert not (drive_root / "state" / "skills" / "weather" / filename).exists()


def test_data_read_allows_skill_review_json(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.core_file_tools import _data_read

    drive_root = tmp_path / "data"
    review_path = drive_root / "state" / "skills" / "weather" / "review.json"
    review_path.parent.mkdir(parents=True)
    review_path.write_text(json.dumps({"status": "pass", "findings": []}), encoding="utf-8")
    monkeypatch.setattr(cfg, "DATA_DIR", drive_root, raising=True)

    ctx = _make_drive_ctx(tmp_path)
    result = _data_read(ctx, "state/skills/weather/review.json")

    assert "DATA_READ_BLOCKED" not in result
    assert '"status": "pass"' in result


def test_data_write_blocks_skill_grants_case_variants(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.core import _data_write

    drive_root = tmp_path / "data"
    drive_root.mkdir()
    monkeypatch.setattr(cfg, "DATA_DIR", drive_root, raising=True)

    ctx = _make_drive_ctx(tmp_path)
    result = _data_write(
        ctx,
        "State/Skills/weather/grants.json",
        json.dumps({"granted_keys": ["OPENROUTER_API_KEY"]}),
    )
    assert "DATA_WRITE_BLOCKED" in result
    assert not (drive_root / "State" / "Skills" / "weather" / "grants.json").exists()


def test_data_write_blocks_skill_trust_state_under_symlinked_skill_dir(tmp_path, monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools.core import _data_write

    drive_root = tmp_path / "data"
    link_target = drive_root / "memory" / "linkstate"
    link_target.mkdir(parents=True)
    skills_root = drive_root / "state" / "skills"
    skills_root.mkdir(parents=True)
    try:
        (skills_root / "weather").symlink_to(link_target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this filesystem")
    monkeypatch.setattr(cfg, "DATA_DIR", drive_root, raising=True)

    ctx = _make_drive_ctx(tmp_path)
    result = _data_write(ctx, "state/skills/weather/review.json", json.dumps({"status": "pass"}))
    assert "DATA_WRITE_BLOCKED" in result
    assert not (link_target / "review.json").exists()

    backing_result = _data_write(ctx, "memory/linkstate/enabled.json", json.dumps({"enabled": True}))
    assert "DATA_WRITE_BLOCKED" in backing_result
    assert not (link_target / "enabled.json").exists()


def test_data_write_allows_other_data_files(tmp_path, monkeypatch):
    """Defense doesn't break legitimate data writes."""
    from ouroboros import config as cfg
    from ouroboros.tools.core import _data_write

    drive_root = tmp_path / "data"
    drive_root.mkdir()
    monkeypatch.setattr(cfg, "SETTINGS_PATH", drive_root / "settings.json", raising=True)

    ctx = _make_drive_ctx(tmp_path)
    result = _data_write(ctx, "memory/scratchpad.md", "hello world")
    assert "DATA_WRITE_BLOCKED" not in result
    assert (drive_root / "memory" / "scratchpad.md").read_text(encoding="utf-8") == "hello world"


def test_data_write_blocks_settings_via_symlink(tmp_path, monkeypatch):
    """Symlink obfuscation: agent writes to ``alias.json`` which points to settings.json."""
    from ouroboros import config as cfg
    from ouroboros.tools.core import _data_write

    drive_root = tmp_path / "data"
    drive_root.mkdir()
    settings_path = drive_root / "settings.json"
    settings_path.write_text("{}", encoding="utf-8")  # exist so symlink resolves
    alias_path = drive_root / "alias.json"
    try:
        alias_path.symlink_to(settings_path)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this filesystem (Windows non-admin?)")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)

    ctx = _make_drive_ctx(tmp_path)
    result = _data_write(ctx, "alias.json", json.dumps({"OUROBOROS_RUNTIME_MODE": "pro"}))
    assert "DATA_WRITE_BLOCKED" in result


def test_data_write_blocks_settings_via_env_override(tmp_path, monkeypatch):
    """OUROBOROS_SETTINGS_PATH override: SETTINGS_PATH is computed at module
    load, so monkeypatch the live constant directly."""
    from ouroboros import config as cfg
    from ouroboros.tools.core import _data_write

    drive_root = tmp_path / "data"
    drive_root.mkdir()
    relocated = drive_root / "deep" / "alt-settings.json"
    relocated.parent.mkdir(parents=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", relocated, raising=True)

    ctx = _make_drive_ctx(tmp_path)
    result = _data_write(ctx, "deep/alt-settings.json", "{}")
    assert "DATA_WRITE_BLOCKED" in result


# ---------------------------------------------------------------------------
# 6. macOS APFS / Windows NTFS case-insensitive filesystem bypass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant",
    [
        "Settings.json",
        "SETTINGS.JSON",
        "settings.JSON",
        "SettiNgs.json",
    ],
)
def test_data_write_blocks_settings_case_variants(variant, tmp_path, monkeypatch):
    """Adversarial-review iteration 1 (Gemini/GPT, verified empirically): on
    case-insensitive filesystems (APFS, NTFS) ``os.path.normcase`` is a
    no-op on darwin, so the previous string-equality compare let
    ``data_write("Settings.json", ...)`` route around the chokepoint
    even though the filesystem wrote to the same inode. The
    ``Path.samefile`` + case-insensitive name-compare fallback closes
    this. Parametrize over multiple case variants so a future regression
    that touches only one branch is caught."""
    from ouroboros import config as cfg
    from ouroboros.tools.core import _data_write

    drive_root = tmp_path / "data"
    drive_root.mkdir()
    settings_path = drive_root / "settings.json"
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)

    ctx = _make_drive_ctx(tmp_path)
    result = _data_write(ctx, variant, json.dumps({"OUROBOROS_RUNTIME_MODE": "pro"}))
    assert "DATA_WRITE_BLOCKED" in result, (
        f"Case variant {variant!r} bypassed the chokepoint. "
        "macOS APFS / Windows NTFS treat these as the same file; the "
        "block must too."
    )
    # On case-insensitive FS the file may exist (write went through
    # rejection path before opening). Ensure the actual on-disk
    # ``settings.json`` has not been written.
    if settings_path.exists():
        # We didn't seed it; if the chokepoint correctly refused the write,
        # this branch should be empty.
        assert "OUROBOROS_RUNTIME_MODE" not in settings_path.read_text()
