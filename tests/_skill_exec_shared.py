"""Skill builders, context factories and review-state helpers shared by the skill_exec suites.

Split out of ``tests/test_skill_exec.py`` when that module was divided by theme; every
definition is verbatim, so each sibling suite keeps the exact payload, manifest and review
state it was written against. ``_clean_extension_runtime`` is autouse, so importing it into
a test module re-applies it there — every sibling suite imports it.
"""

from __future__ import annotations

import pathlib

import pytest

from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_enabled, save_review_state
from ouroboros.tools.registry import ToolContext
from ouroboros.contracts.task_constraint import TaskConstraint

from tests._shared import clean_extension_runtime_state


@pytest.fixture(autouse=True)
def _clean_extension_runtime():
    clean_extension_runtime_state()
    yield
    clean_extension_runtime_state()


def _valid_script_manifest(
    name: str = "weather",
    *,
    runtime: str = "python3",
    timeout_sec: int = 30,
    scripts_only: bool = True,
) -> str:
    return (
        "---\n"
        f"name: {name}\n"
        "description: Simple greeter.\n"
        "version: 0.1.0\n"
        f"type: {'script' if scripts_only else 'extension'}\n"
        f"runtime: {runtime}\n"
        f"timeout_sec: {timeout_sec}\n"
        "scripts:\n"
        "  - name: hello.py\n"
        "    description: Print hello.\n"
        "---\n"
        "# body\n"
    )


def _build_skill(
    skills_root: pathlib.Path,
    name: str,
    *,
    script_body: str = "print('hello from skill')\n",
    manifest: str | None = None,
) -> pathlib.Path:
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(manifest or _valid_script_manifest(name), encoding="utf-8")
    scripts = skill_dir / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "hello.py").write_text(script_body, encoding="utf-8")
    return skill_dir


def _make_ctx(tmp_path: pathlib.Path) -> ToolContext:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    return ToolContext(repo_dir=repo_dir, drive_root=drive_root)


def _set_skill_repair(ctx: ToolContext, name: str = "alpha", payload_root: str = "skills/external/alpha") -> None:
    ctx.task_constraint = TaskConstraint(mode="skill_repair", skill_name=name, payload_root=payload_root, allow_enable=False, allow_review=True)


def _admit_repair(ctx: ToolContext, name: str, payload_root: str) -> None:
    """Bind the repair to the payload state it is admitted against (X3/F8).

    A repair TASK now writes only under its admission record: the promote seam
    records it for every real repair, and a task without one is typed STALE
    rather than silently unverified. These heal-mode tests drive the constraint
    directly, so they mint the same binding the promote seam would.
    """
    from ouroboros.skill_repair_admission import record_repair_admission

    ctx.task_id = ctx.task_id or "repair-heal-test"
    record_repair_admission(
        ctx.drive_root, name, task_id=ctx.task_id,
        base_content_hash=compute_content_hash(ctx.drive_root / payload_root),
    )


def _mark_reviewed_and_enabled(drive_root: pathlib.Path, skill_dir: pathlib.Path, name: str):
    content_hash = compute_content_hash(skill_dir)
    save_enabled(drive_root, name, True)
    save_review_state(
        drive_root,
        name,
        SkillReviewState(status="pass", content_hash=content_hash),
    )


def _mark_reviewed(drive_root: pathlib.Path, skill_dir: pathlib.Path, name: str):
    save_review_state(
        drive_root,
        name,
        SkillReviewState(status="pass", content_hash=compute_content_hash(skill_dir)),
    )
