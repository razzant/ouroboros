"""Network-free skill source fixtures: real local git repositories (a hub, an author repo, a plain
skills repository) that the source transport fetches exactly as it fetches a remote one."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
from typing import Any, Dict, List, Optional

_GIT_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"}


def git(*args: str, cwd: pathlib.Path) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main", *args],
        cwd=cwd, env={**os.environ, **_GIT_ENV}, capture_output=True, text=True, timeout=60, check=True,
    )
    return result.stdout.strip()


def skill_md(name: str, version: str = "1.0.0", *, description: str = "", body: str = "About it.",
             extra: str = "") -> str:
    return (f"---\nname: {name}\ndescription: {description or name + ' does things'}\nversion: {version}\n"
            f"type: instruction\n{extra}---\n{body}\n")


def write_skill(root: pathlib.Path, name: str, version: str = "1.0.0", **kwargs: Any) -> pathlib.Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(skill_md(name, version, **kwargs), encoding="utf-8")
    return folder


def commit_all(work: pathlib.Path, message: str = "change") -> str:
    git("add", "-A", cwd=work)
    git("commit", "-q", "-m", message, cwd=work)
    return git("rev-parse", "HEAD", cwd=work)


def publish(work: pathlib.Path, bare: pathlib.Path) -> None:
    """Push ``work``'s main into ``bare`` (created on first use)."""
    if not bare.exists():
        git("init", "-q", "--bare", str(bare), cwd=work.parent)
    git("push", "-q", "--force", str(bare), "HEAD:refs/heads/main", cwd=work)


def entry(name: str, digest: str, *, version: str = "1.0.0", category: Optional[List[str]] = None,
          source: Optional[Dict[str, Any]] = None, withdrawn: Any = None) -> Dict[str, Any]:
    return {"schema_version": 2, "name": name, "description": f"{name} card text", "category": category or ["city"],
            "owner": "author", "version": version, "source": source or {"kind": "hosted", "path": f"skills/{name}"},
            "digest": digest, "withdrawn": withdrawn}


class HubFixture:
    """A hub repository with hosted entries, an Official endorsement and an optional pointer entry."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.work = root / "hub-work"
        self.bare = root / "hub.git"
        self.work.mkdir(parents=True)
        git("init", "-q", cwd=self.work)
        (self.work / "hub.json").write_text(json.dumps({"schema_version": 2, "name": "TestHub", "maintainer": "m"}),
                                            encoding="utf-8")
        (self.work / "categories.json").write_text(json.dumps({"schema_version": 1, "categories": [
            {"id": "city", "label": "City & services"}, {"id": "writing", "label": "Writing & review"}]}),
            encoding="utf-8")
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.official: List[Dict[str, Any]] = []

    def add_hosted(self, name: str, version: str = "1.0.0", *, official: bool = False, recommended: bool = False,
                   category: Optional[List[str]] = None, withdrawn: Any = None, **skill: Any) -> str:
        from ouroboros.skill_source_catalog import tree_release_digest

        folder = write_skill(self.work / "skills", name, version, **skill)
        digest = tree_release_digest(folder)
        self.entries[name] = entry(name, digest, version=version, category=category, withdrawn=withdrawn)
        self.official = [item for item in self.official if item["name"] != name]
        if official:
            self.official.append({"name": name, "digest": digest, "recommended": recommended})
        return digest

    def add_pointer(self, name: str, repo: str, commit: str, digest: str, path: str = "") -> None:
        self.entries[name] = entry(name, digest, source={"kind": "git", "repo": repo, "path": path, "commit": commit})

    def remove(self, name: str) -> None:
        self.entries.pop(name, None)
        (self.work / "entries" / f"{name}.json").unlink(missing_ok=True)

    def commit(self, message: str = "hub change") -> str:
        entries = self.work / "entries"
        entries.mkdir(exist_ok=True)
        for name, value in self.entries.items():
            (entries / f"{name}.json").write_text(json.dumps(value, indent=1), encoding="utf-8")
        (self.work / "official.json").write_text(json.dumps(
            {"schema_version": 1, "maintainer": "m", "endorsements": self.official}), encoding="utf-8")
        commit = commit_all(self.work, message)
        publish(self.work, self.bare)
        return commit

    def publish_signals(self, skills: Dict[str, Any]) -> None:
        signals = self.root / "signals-work"
        if not signals.exists():
            signals.mkdir()
            git("init", "-q", cwd=signals)
        (signals / "signals.json").write_text(json.dumps({"schema_version": 1, "generated_at": "now",
                                                          "skills": skills, "author_repos": {}}), encoding="utf-8")
        commit_all(signals, "signals")
        git("push", "-q", "--force", str(self.bare), "HEAD:refs/heads/signals", cwd=signals)

    def rows(self, *, official_trust: str = "trusted", community_trust: str = "review") -> List[Dict[str, Any]]:
        url = str(self.bare)
        return [
            {"id": "builtin", "name": "Built in", "kind": "builtin", "trust": "trusted", "confirmed": True},
            {"id": "ouroboros", "name": "Ouroboros", "kind": "hub", "url": url, "ref": "main", "part": "official",
             "trust": official_trust, "confirmed": True},
            {"id": "ouroboros-community", "name": "Ouroboros Community", "kind": "hub", "url": url, "ref": "main",
             "part": "community", "trust": community_trust, "confirmed": True},
        ]


def author_repo(root: pathlib.Path, name: str, version: str = "2.0.0") -> tuple[pathlib.Path, str, str]:
    """An author repository holding one skill at its root; returns (bare path, commit, digest)."""
    from ouroboros.skill_source_catalog import tree_release_digest

    work = root / f"{name}-author"
    work.mkdir(parents=True)
    git("init", "-q", cwd=work)
    (work / "SKILL.md").write_text(skill_md(name, version), encoding="utf-8")
    commit = commit_all(work, "author release")
    bare = root / f"{name}-author.git"
    publish(work, bare)
    return bare, commit, tree_release_digest(work)


def join_source_refreshes(timeout: float = 60.0) -> None:
    """Wait for the background source refreshes a test started (catalog reads start them by design)."""
    from ouroboros import skill_source_git

    with skill_source_git._THREADS_LOCK:
        threads = list(skill_source_git._THREADS.values())
    for thread in threads:
        thread.join(timeout)


def builtin_repo(root: pathlib.Path) -> pathlib.Path:
    """A fake app checkout whose ``skills/`` is the built-in source."""
    repo = root / "app"
    write_skill(repo / "skills", "telegram", "1.3.2", description="Owner bridge")
    write_skill(repo / "skills", "unix_computer_use", "0.4.3", description="Desktop tools")
    (repo / "skills" / "builtin.json").write_text(json.dumps({"schema_version": 1, "skills": [
        {"name": "telegram", "category": ["messaging"], "recommended": True},
        {"name": "unix_computer_use", "category": ["computer"], "recommended": False}]}), encoding="utf-8")
    return repo
