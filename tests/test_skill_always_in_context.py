"""An instruction skill may place its reviewed body in every task's context.

Pins: the manifest field parses and refuses with a repair hint; the body reaches the
cached semi-stable system block of roots, shared-input children and every context
mode, and session leaves' host references; disabled, unreviewed, stale or opted-out
skills stay read-on-demand; declared-input children keep their explicit composition.
"""

import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.contracts.skill_manifest import (
    MAX_ALWAYS_IN_CONTEXT_BODY_BYTES,
    SkillManifestError,
    parse_skill_manifest_text,
)
from ouroboros.skill_catalogue import always_in_context_section, discover_context_skills, session_skill_index
from ouroboros.skill_loader import compute_content_hash, save_enabled
from tests.test_doc_context import _make_env_and_memory

BODY = "ALWAYS_BODY_SENTINEL: avoid stock contrasts by default.\n\nSee references/signals.md."


def _manifest(always="true", *, kind="instruction", body=BODY) -> str:
    field = "" if always is None else f"always_in_context: {always}\n"
    return (f"---\nname: guide\ndescription: d\nversion: '1'\ntype: {kind}\n{field}---\n{body}\n")


def _install(drive: pathlib.Path, name: str, *, always: bool = True, reviewed: bool = True) -> pathlib.Path:
    skill_dir = drive / "skills" / "external" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: writing guide\nversion: 0.2.0\ntype: instruction\n"
        f"when_to_use: writing prose\n{'always_in_context: true' if always else ''}\n---\n"
        f"{BODY.replace('ALWAYS_BODY', name.upper())}\n",
        encoding="utf-8")
    state = drive / "state" / "skills" / name
    state.mkdir(parents=True)
    save_enabled(drive, name, True, actor="owner_ui")
    if reviewed:
        (state / "review.json").write_text(json.dumps(
            {"status": "pass", "content_hash": compute_content_hash(skill_dir)}), encoding="utf-8")
    return skill_dir


# --- manifest contract -------------------------------------------------------


def test_field_parses_and_defaults_off():
    manifest = parse_skill_manifest_text(_manifest())
    assert manifest.always_in_context is True and "always_in_context" not in manifest.raw_extra
    for absent in (None, "false"):
        assert parse_skill_manifest_text(_manifest(absent)).always_in_context is False
    exact = "é" * (MAX_ALWAYS_IN_CONTEXT_BODY_BYTES // 2)
    assert parse_skill_manifest_text(_manifest(body=exact)).always_in_context is True


@pytest.mark.parametrize("text, problem", [
    (_manifest("'yes'"), "must be true or false"),
    (_manifest("1"), "must be true or false"),
    # A present null (blank or `null`) is a malformed declaration; only an absent key opts out silently.
    (_manifest(""), "must be true or false"),
    (_manifest("null"), "must be true or false"),
    (_manifest(kind="extension"), "instruction skills only"),
    (_manifest(kind="script"), "instruction skills only"),
    (_manifest(body=""), "non-empty SKILL.md body"),
    (json.dumps({"name": "g", "description": "d", "version": "1", "type": "instruction",
                 "always_in_context": True}), "non-empty SKILL.md body"),
    # The bound counts UTF-8 bytes, not characters.
    (_manifest(body="é" * (MAX_ALWAYS_IN_CONTEXT_BODY_BYTES // 2 + 1)), "bytes"),
], ids=["quoted-bool", "integer-bool", "blank", "null", "extension", "script",
        "empty-markdown", "empty-json", "oversized-utf8"])
def test_refusals_teach_the_repair(text, problem):
    with pytest.raises(SkillManifestError) as excinfo:
        parse_skill_manifest_text(text)
    assert problem in excinfo.value.problem
    assert excinfo.value.fix_hint and excinfo.value.fix_hint in str(excinfo.value)


@pytest.mark.parametrize("declared", [True, False])
def test_clawhub_adaptation_keeps_the_declaration(tmp_path, declared):
    from ouroboros.marketplace.adapter import adapt_openclaw_skill

    (tmp_path / "SKILL.md").write_text(_manifest("true" if declared else None), encoding="utf-8")
    result = adapt_openclaw_skill(tmp_path, slug="guide")
    assert result.ok, result.blockers
    assert result.manifest.always_in_context is declared
    rendered = (tmp_path / "SKILL.md").read_text(encoding="utf-8")
    assert ("always_in_context: true" in rendered) is declared
    assert parse_skill_manifest_text(rendered).always_in_context is declared
    assert not any("always_in_context is dropped" in warning for warning in result.warnings)


def test_clawhub_adaptation_names_a_dropped_declaration_when_the_package_becomes_a_script(tmp_path):
    from ouroboros.marketplace.adapter import adapt_openclaw_skill

    (tmp_path / "SKILL.md").write_text(_manifest(), encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.py").write_text("print('fixture')\n", encoding="utf-8")
    result = adapt_openclaw_skill(tmp_path, slug="guide")
    assert result.ok, result.blockers
    assert result.manifest.type == "script" and result.manifest.always_in_context is False
    assert any("always_in_context is dropped" in warning for warning in result.warnings)


# --- selection and rendering ----------------------------------------------


def test_only_enabled_reviewed_current_opted_in_bodies_ride_whole(tmp_path):
    _install(tmp_path, "kept")
    _install(tmp_path, "optout", always=False)
    _install(tmp_path, "off")
    save_enabled(tmp_path, "off", False, actor="owner_ui")
    _install(tmp_path, "unreviewed", reviewed=False)
    stale = _install(tmp_path, "stale")
    (stale / "SKILL.md").write_text((stale / "SKILL.md").read_text() + "edited after review\n", encoding="utf-8")
    section = always_in_context_section(discover_context_skills(tmp_path))
    assert section.startswith("## Always-in-context skills\n\n")
    assert "never outranks BIBLE.md" in section
    assert "### kept (version 0.2.0)" in section
    assert "KEPT_SENTINEL: avoid stock contrasts by default.\n\nSee references/signals.md." in section
    assert "read_file(root='skill_payload', bucket='external', skill_name='kept'" in section
    for name in ("OPTOUT", "OFF", "UNREVIEWED", "STALE"):
        assert f"{name}_SENTINEL" not in section
    assert always_in_context_section(discover_context_skills(tmp_path / "empty")) == ""


def test_catalog_marks_the_always_on_skill_from_the_same_discovery(tmp_path):
    from ouroboros.context import _build_installed_skills_section

    _install(tmp_path, "kept")
    _install(tmp_path, "optout", always=False)
    section = _build_installed_skills_section(SimpleNamespace(drive_root=tmp_path))
    entries = {block.split(" ", 1)[0]: block for block in section.split("\n- ")[1:]}
    assert "Context: always; its text is in the Always-in-context skills section." in entries["kept"]
    assert "Context: always" not in entries["optout"]
    assert "KEPT_SENTINEL" not in section  # the catalog stays metadata
    assert _build_installed_skills_section(SimpleNamespace(drive_root=tmp_path), skills=[]) == ""


# --- delivery into real context builds ------------------------------------


@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_body_rides_in_the_cached_stable_block_in_every_mode(tmp_path, monkeypatch, mode):
    from ouroboros.context import build_llm_messages

    env, memory = _make_env_and_memory(tmp_path)
    _install(env.drive_root, "kept")
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", mode)
    messages, _ = build_llm_messages(env=env, memory=memory, task={"id": "t", "type": "task", "text": "hi"})
    blocks = [block for block in messages[0]["content"] if isinstance(block, dict)]
    holding = [block for block in blocks if "KEPT_SENTINEL" in block.get("text", "")]
    assert len(holding) == 1 and holding[0].get("cache_control") == {"type": "ephemeral"}
    assert "Context: always" in blocks[-1]["text"] and "KEPT_SENTINEL" not in blocks[-1]["text"]


@pytest.mark.parametrize("selection, delivered", [(None, True), ("shared", True), ("declared", False)])
def test_children_get_it_unless_their_inputs_are_declared(tmp_path, selection, delivered):
    from ouroboros import context

    env, memory = _make_env_and_memory(tmp_path)
    _install(env.drive_root, "kept")
    child = {"id": "child", "type": "task", "delegation_role": "subagent", "text": "Q",
             "configured_subagent": {"route": {"kind": "api_model"}}, "task_contract": {}}
    if selection:
        child["task_contract"]["input_sources"] = selection
    core = context._capture_context_core(env, memory, child, None, None)
    assert ("KEPT_SENTINEL" in core.semi_stable_text) is delivered
    assert "KEPT_SENTINEL" not in core.dynamic_text + core.dynamic_head_text


def test_session_leaves_receive_the_body_and_others_stay_references(tmp_path):
    _install(tmp_path, "kept")
    _install(tmp_path, "optout", always=False)
    index = session_skill_index(SimpleNamespace(drive_root=tmp_path))
    assert "- kept — writing guide — Trigger: writing prose — read: " in index
    assert "- optout — writing guide — Trigger: writing prose — read: " in index
    guidance = index.split("ALWAYS-IN-CONTEXT HOST GUIDANCE\n", 1)[1]
    root = (tmp_path / "skills" / "external" / "kept").resolve()
    assert f"### kept (files it names are relative to {root})\nKEPT_SENTINEL" in guidance
    assert "OPTOUT_SENTINEL" not in index
    save_enabled(tmp_path, "kept", False, actor="owner_ui")
    assert "ALWAYS-IN-CONTEXT HOST GUIDANCE" not in session_skill_index(SimpleNamespace(drive_root=tmp_path))
