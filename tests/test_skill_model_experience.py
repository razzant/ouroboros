"""CPL-7 Model Experience: manifest section + model-visible rendering + teaching refusals.

Pins (plan §7 item 7): a manifest WITH the section parses and reaches the
model-visible surfaces (summarize_skills → list_skills JSON, installed-skills
context section); a manifest WITHOUT it keeps the exact prior behavior; a
registration refusal explains how to fix the manifest (typed ``fix_hint``).
"""

import json
import pathlib

import pytest

from ouroboros.contracts.skill_manifest import (
    MODEL_EXPERIENCE_KEYS,
    SkillManifestError,
    parse_skill_manifest_text,
)

REPO = pathlib.Path(__file__).resolve().parent.parent


# --- manifest schema -------------------------------------------------------


def test_manifest_parses_string_form_as_prose():
    manifest = parse_skill_manifest_text(json.dumps({
        "name": "x", "description": "d", "version": "1", "type": "instruction",
        "model_experience": "adds a short cheat-sheet to context",
    }))
    assert manifest.model_experience == {"what_model_sees": "adds a short cheat-sheet to context"}


def test_manifest_parses_mapping_form_trimmed():
    manifest = parse_skill_manifest_text(
        "---\n"
        "name: y\n"
        "description: d\n"
        "version: '1'\n"
        "type: instruction\n"
        "model_experience:\n"
        "  what_model_sees: '  prose here  '\n"
        "  token_effect: 'small fixed cost'\n"
        "---\nbody\n"
    )
    assert manifest.model_experience == {
        "what_model_sees": "prose here",
        "token_effect": "small fixed cost",
    }


def test_manifest_without_section_is_none_and_not_extra():
    manifest = parse_skill_manifest_text(json.dumps({
        "name": "z", "description": "d", "version": "1", "type": "instruction",
    }))
    assert manifest.model_experience is None
    assert "model_experience" not in manifest.raw_extra


@pytest.mark.parametrize("bad_section, expected_problem_bit", [
    ({"wat": "x"}, "unknown keys"),
    ({"what_model_sees": ["not", "prose"]}, "must be a prose string"),
    ({}, "at least one non-empty prose field"),
    ({"what_model_sees": "   "}, "at least one non-empty prose field"),
    # Same teaching error for the STRING form: a section the author filled in
    # with nothing but whitespace was accepted and stored as empty prose, so
    # every model-visible surface rendered a "Model experience:" label with no
    # sentence after it. The mapping form has always refused it.
    ("   ", "at least one non-empty prose field"),
    ("\n\t \n", "at least one non-empty prose field"),
    (42, "prose string or a mapping"),
])
def test_model_experience_refusals_teach(bad_section, expected_problem_bit):
    with pytest.raises(SkillManifestError) as excinfo:
        parse_skill_manifest_text(json.dumps({
            "name": "w", "description": "d", "version": "1", "type": "instruction",
            "model_experience": bad_section,
        }))
    err = excinfo.value
    assert expected_problem_bit in err.problem
    assert err.fix_hint  # every refusal of the new section teaches the repair
    assert err.fix_hint in str(err)
    assert "fix:" in str(err)


def test_legacy_refusal_sites_teach_the_repair():
    """Representative pre-existing refusals now carry a non-empty fix_hint."""
    cases = [
        {"name": "a", "description": "d", "version": "1", "type": "script",
         "scripts": "not-a-list"},
        {"name": "b", "description": "d", "version": "1", "type": "instruction",
         "schema_version": 99},
        {"name": "c", "description": "d", "version": "1", "type": "extension",
         "entry": "plugin.py", "plugin_api": {"version": "2.0", "surprise": True}},
    ]
    for payload in cases:
        with pytest.raises(SkillManifestError) as excinfo:
            parse_skill_manifest_text(json.dumps(payload))
        assert excinfo.value.fix_hint, f"refusal for {payload} does not teach"
        assert excinfo.value.fix_hint in str(excinfo.value)


def test_model_experience_keys_are_the_contract():
    assert MODEL_EXPERIENCE_KEYS == ("what_model_sees", "token_effect")


# --- model-visible surfaces ------------------------------------------------


def _install_skill(drive: pathlib.Path, name: str, *, model_experience: str = "") -> None:
    from ouroboros.skill_loader import compute_content_hash

    skill_dir = drive / "skills" / "external" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: does a thing\n"
        "version: 1.0.0\n"
        "type: instruction\n"
        "when_to_use: user asks for the thing\n"
        + model_experience
        + "---\nbody\n",
        encoding="utf-8",
    )
    digest = compute_content_hash(skill_dir)
    state = drive / "state" / "skills" / name
    state.mkdir(parents=True)
    (state / "enabled.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    (state / "review.json").write_text(
        json.dumps({"status": "pass", "content_hash": digest}), encoding="utf-8"
    )


_SECTION_YAML = (
    "model_experience:\n"
    "  what_model_sees: one bounded cheat-sheet block\n"
    "  token_effect: about two hundred tokens per round\n"
)


def test_summarize_skills_carries_model_experience(tmp_path):
    from ouroboros.skill_loader import summarize_skills

    _install_skill(tmp_path, "with_exp", model_experience=_SECTION_YAML)
    _install_skill(tmp_path, "without_exp")
    rows = {row["name"]: row for row in summarize_skills(tmp_path)["skills"]}
    assert rows["with_exp"]["model_experience"] == {
        "what_model_sees": "one bounded cheat-sheet block",
        "token_effect": "about two hundred tokens per round",
    }
    assert rows["without_exp"]["model_experience"] is None


def test_context_section_renders_model_experience(tmp_path):
    from ouroboros.context import _build_installed_skills_section

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    _install_skill(drive, "with_exp", model_experience=_SECTION_YAML)

    class Env:
        repo_dir = repo
        drive_root = drive

    section = _build_installed_skills_section(Env())
    assert "with_exp" in section
    assert "Model experience: one bounded cheat-sheet block" in section
    assert "Token effect: about two hundred tokens per round" in section


def test_context_section_without_section_unchanged(tmp_path):
    from ouroboros.context import _build_installed_skills_section

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    _install_skill(drive, "plain")

    class Env:
        repo_dir = repo
        drive_root = drive

    section = _build_installed_skills_section(Env())
    assert "plain" in section
    assert "Model experience" not in section
    assert "Token effect" not in section


def test_bundled_skills_declare_model_experience():
    for rel in ("skills/telegram/SKILL.md", "skills/unix_computer_use/SKILL.md"):
        manifest = parse_skill_manifest_text((REPO / rel).read_text(encoding="utf-8"))
        section = manifest.model_experience
        assert section, f"{rel} lacks the Model Experience section"
        assert section.get("what_model_sees"), rel
        assert section.get("token_effect"), rel


def test_adapter_frontmatter_preserves_model_experience():
    from ouroboros.marketplace.adapter import _manifest_frontmatter_dict, _render_frontmatter

    manifest = parse_skill_manifest_text(json.dumps({
        "name": "hub_skill", "description": "d", "version": "1",
        "type": "instruction",
        "model_experience": {"what_model_sees": "prose survives adaptation"},
    }))
    front = _manifest_frontmatter_dict(manifest)
    assert front["model_experience"] == {"what_model_sees": "prose survives adaptation"}
    rendered = _render_frontmatter(front)
    assert "model_experience" in rendered
    assert "prose survives adaptation" in rendered
