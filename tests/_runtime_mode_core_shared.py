"""The registry, git-repo and skill-payload builders shared by the runtime-mode suites.

Split out of ``tests/test_runtime_mode_core.py`` when that module was divided by
theme; every builder is verbatim, so each sibling suite keeps the exact
registry wiring, repository layout and skill-payload tree it was written against.
"""

from __future__ import annotations

import pathlib
import subprocess


from ouroboros.tools.registry import ToolRegistry


def _registry(tmp_path):
    return ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)


def _git_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("ok\n", encoding="utf-8")
    (repo / "BIBLE.md").write_text("constitution\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    return repo


def _make_skill_payload(tmp_path, bucket, name):
    """Create data/skills/<bucket>/<name>/plugin.py so resolve_skill_payload_target
    sees an existing payload root."""
    payload = tmp_path / "skills" / bucket / name
    payload.mkdir(parents=True)
    (payload / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test\nversion: 1.0.0\ntype: skill\n---\n",
        encoding="utf-8",
    )
    (payload / "plugin.py").write_text("def register(api):\n    pass\n", encoding="utf-8")
    return payload
