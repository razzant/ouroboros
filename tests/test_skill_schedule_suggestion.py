"""Skill declarations are bounded suggestions with a path to their full source."""

import json
from types import SimpleNamespace

import pytest

from ouroboros.context import _build_installed_skills_section
from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_enabled, save_review_state, summarize_skills
from tests.test_skill_catalogue import _read, _seed, registry as registry


def _suggestions(count=1):
    return [{"name": f"review-{i}", "cron": "0 9 * * 1", "timezone": "Europe/Paris",
             "description": "Review the project using this skill"} for i in range(count)]


def _review(registry, folder, *, enabled=True):
    root = registry._ctx.drive_root
    save_review_state(root, folder.name, SkillReviewState(status="pass", content_hash=compute_content_hash(folder)))
    save_enabled(root, folder.name, enabled)


def test_bounded_suggestions_lead_to_every_full_declaration(registry):
    specs = _suggestions(10)
    specs[0]["description"] = "Opening\n" + "d" * 1000 + "\nDESCRIPTION_END"
    folder = _seed(registry, "review-guide", scheduled_tasks=specs, permissions=["supervised_task"])
    _review(registry, folder)
    root = registry._ctx.drive_root
    [row] = summarize_skills(root)["skills"]
    assert len(row["suggested_schedules"]) == 8 and row["suggested_schedules_omitted_count"] == 2
    preview = row["suggested_schedules"][0]
    assert preview["cron"] == specs[0]["cron"] and preview["timezone"] == specs[0]["timezone"]
    assert len(preview["description"]) <= 260 and "OMISSION" in preview["description"]
    section = _build_installed_skills_section(SimpleNamespace(drive_root=root))
    assert section.count("Suggests schedule ") == 8
    assert "2 more suggestions omitted" in section
    assert "list_skills(name='review-guide') supplies manifest.read" in section
    named = json.loads(registry.execute("list_skills", {"name": "review-guide"}))
    source = _read(registry, named["manifest"]["read"])
    assert source == (folder / "SKILL.md").read_text(encoding="utf-8")
    assert "DESCRIPTION_END" in source and "review-9" in source
    assert not (root / "state" / "scheduled_tasks.json").exists()


@pytest.mark.parametrize("state", ["ready", "disabled", "pending", "stale"])
def test_context_suggestions_require_current_review_and_enablement(registry, state, monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    folder = _seed(registry, "review-guide", scheduled_tasks=_suggestions(), permissions=["supervised_task"])
    root = registry._ctx.drive_root
    if state == "pending":
        save_enabled(root, folder.name, True)
    else:
        _review(registry, folder, enabled=state != "disabled")
    if state == "stale":
        (folder / "playbook.md").write_text("Changed procedure", encoding="utf-8")
    section = _build_installed_skills_section(SimpleNamespace(drive_root=root))
    line = ("Suggests schedule review-0: cron 0 9 * * 1 (Europe/Paris) — "
            "Review the project using this skill. Not scheduled unless adopted.")
    assert (line in section) is (state == "ready")
    if state == "ready":  # nothing omitted, so no omission note
        assert "supplies manifest.read." in section and "more suggestions omitted" not in section
    assert not (root / "state" / "scheduled_tasks.json").exists()


def test_skills_without_suggestions_keep_an_empty_projection(registry):
    folder = _seed(registry, "review-guide")
    _review(registry, folder)
    root = registry._ctx.drive_root
    [row] = summarize_skills(root)["skills"]
    assert row["suggested_schedules"] == [] and row["suggested_schedules_omitted_count"] == 0
    section = _build_installed_skills_section(SimpleNamespace(drive_root=root))
    assert "review-guide" in section and "Suggests schedule" not in section and "Schedule source" not in section
