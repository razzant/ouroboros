"""Bundled starter buttons exercise the same manifest catalog as user skills."""
from pathlib import Path

from ouroboros.contracts.skill_manifest import parse_skill_manifest_text


def test_templates_are_instruction_contributions_that_project_to_drafts():
    from ouroboros.extension_contributions import project_contributions
    from ouroboros.extension_ui_validation import validate_contributions

    directory = Path(__file__).resolve().parents[1] / "skills" / "templates"
    manifest = parse_skill_manifest_text((directory / "SKILL.md").read_text(encoding="utf-8"))
    assert manifest.validate() == []
    assert manifest.type == "instruction"
    entries, problems = validate_contributions(manifest, skill_dir=directory)
    assert problems == []
    rows = project_contributions(directory.name, entries, "test-revision", manifest.name)
    assert [(row["label"], row["action"]["text"]) for row in rows] == [
        ("Explain simpler", "Explain your last answer more simply, with an example."),
        ("Summarize this chat", "Summarize what we decided in this chat and what is still open."),
        ("Check my text", "Check the text below for mistakes and unclear places, then show a corrected version:\n\n"),
    ]
    assert [row["order"] for row in rows] == [1, 2, 3], "the \"+\" menu keeps the manifest's order"
    for row in rows:
        assert row["key"] == f"templates:{row['id']}" and row["skill_label"] == "Templates"
        assert row["placement"] == "composer.plus"
        assert row["action"]["verb"] == "draft.insert"
        assert row["render"]["kind"] == "declarative"
        assert row["revision"] == "test-revision"
