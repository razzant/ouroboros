"""The skill writer and the valid script manifest shared by the skill-loader suites.

Split out of ``tests/test_skill_loader.py`` when that module was divided by theme; both are
verbatim, so every sibling suite keeps the exact payload layout and manifest it was written
against.
"""

from __future__ import annotations

import pathlib


def _write_skill(
    repo_root: pathlib.Path,
    name: str,
    *,
    manifest: str,
    scripts: dict[str, str] | None = None,
    manifest_name: str = "SKILL.md",
) -> pathlib.Path:
    skill_dir = repo_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / manifest_name).write_text(manifest, encoding="utf-8")
    if scripts:
        (skill_dir / "scripts").mkdir(exist_ok=True)
        for filename, body in scripts.items():
            (skill_dir / "scripts" / filename).write_text(body, encoding="utf-8")
    return skill_dir


def _valid_script_manifest(name: str = "weather") -> str:
    return (
        "---\n"
        f"name: {name}\n"
        "description: Check the weather.\n"
        "version: 0.1.0\n"
        "type: script\n"
        "runtime: python3\n"
        "timeout_sec: 30\n"
        "permissions: [net]\n"
        "scripts:\n"
        "  - name: fetch.py\n"
        "    description: Fetch current weather.\n"
        "---\n"
        "# Weather skill\n\nCall fetch.py with a city.\n"
    )
