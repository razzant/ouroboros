"""Shared real-Git fixtures for the body-candidate and body-adoption consumer tests.

A miniature body: the REAL package-init hook and switch helper around a tiny
``server.py`` that reports which generation of each file a process imported.
Everything lives under the caller's temporary directory.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

SERVER_PY = '''"""Toy entry: parses whole, then reaches its first body import (the hook)."""
import json
import os
import sys

import ouroboros  # the package-init hook runs here, before any other body module
from ouroboros import mod_a, mod_b

if __name__ == "__main__":
    out = os.environ.get("TOY_REPORT")
    report = {"server": "GEN_OLD", "a": mod_a.GEN, "b": mod_b.GEN, "pid": os.getpid(), "argv": sys.argv[1:]}
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(report) + "\\n")
    print(json.dumps(report))
'''


CLI_PY = '''"""Toy module CLI: ``python -m ouroboros.cli`` reaches the hook through the package import."""
import json
import os
import sys

from ouroboros import mod_a, mod_b

if __name__ == "__main__":
    report = {"entry": "cli", "a": mod_a.GEN, "b": mod_b.GEN, "argv": sys.argv[1:]}
    out = os.environ.get("TOY_REPORT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(report) + "\\n")
    print(json.dumps(report))
'''

LAUNCHER_PY = '''"""Toy source launcher: imports the body at its own cold start, then supervises server.py."""
import os
import subprocess
import sys

from ouroboros import mod_a  # the hook runs in the launcher process itself

if __name__ == "__main__":
    codes = []
    for _ in range(4):
        code = subprocess.run([sys.executable, "server.py", *sys.argv[1:]],
                              env=dict(os.environ, OUROBOROS_MANAGED_BY_LAUNCHER="1")).returncode
        codes.append(code)
        if code != 42:
            break
    print("LAUNCHER_GEN=%s codes=%s" % (mod_a.GEN, codes))
    sys.exit(codes[-1])
'''


def git(repo, *args, check=True, env=None):
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True,
                          env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
                               **(env or {})})
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout.rstrip("\n")


def make_serving(root: pathlib.Path) -> pathlib.Path:
    """A committed miniature body on branch ``ouroboros`` carrying the real hook and helper."""
    repo = root / "repo"
    (repo / "ouroboros").mkdir(parents=True)
    files = {
        "ouroboros/__init__.py": (REPO / "ouroboros" / "__init__.py").read_text(encoding="utf-8"),
        "ouroboros/version.py": "def get_version():\n    return '0.0.0'\n",
        "ouroboros/body_switch.py": (REPO / "ouroboros" / "body_switch.py").read_text(encoding="utf-8"),
        "ouroboros/mod_a.py": "GEN = 'GEN_OLD'\n", "ouroboros/mod_b.py": "GEN = 'GEN_OLD'\n",
        "ouroboros/gone.py": "GEN = 'GEN_OLD'\n", "server.py": SERVER_PY,
        "launcher.py": LAUNCHER_PY, "ouroboros/cli.py": CLI_PY, "VERSION": "1.0.0\n",
        "BIBLE.md": "# Constitution\n", "requirements-runtime.lock": "toydep==1.0\n",
        ".gitignore": "__pycache__/\n*.pyc\nlocal-notes/\n",
    }
    for relative, text in files.items():
        (repo / relative).write_bytes(text.encode("utf-8"))
    git(repo, "init", "-q", "-b", "ouroboros")
    # Candidate/source byte identities must not depend on the runner's Git EOL policy.
    git(repo, "config", "core.autocrlf", "false")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    return repo


def isolate(monkeypatch, root: pathlib.Path) -> pathlib.Path:
    """Point the worktree registry and data root at ``root``; returns the data dir."""
    data = root / "data"
    (data / "state").mkdir(parents=True, exist_ok=True)
    (data / "logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(root / "worktrees"))
    return data


def make_ctx(serving: pathlib.Path, data: pathlib.Path, task_id: str, **overrides):
    from ouroboros.tools.registry import ToolContext

    fields = dict(repo_dir=serving, drive_root=data, system_repo_dir=serving, task_id=task_id,
                  branch_dev="ouroboros", task_metadata={"root_task_id": task_id})
    fields.update(overrides)
    return ToolContext(**fields)


def candidate_commit(path, message="candidate change", *, files=None, delete=()):
    """Commit a change in a candidate checkout and return its SHA."""
    for rel, text in (files or {"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"}).items():
        target = pathlib.Path(path) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))
    for rel in delete:
        (pathlib.Path(path) / rel).unlink()
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", message)
    return git(path, "rev-parse", "HEAD")


def run_entry(serving: pathlib.Path, *argv, env=None, entry="server.py"):
    """Start a real cold entry process in the serving checkout."""
    command = [sys.executable, *( [entry] if entry.endswith(".py") else ["-m", entry]), *argv]
    return subprocess.run(command, cwd=str(serving), capture_output=True, text=True, timeout=120,
                          env={**os.environ, **(env or {})})


RICH_CHANGE = {
    "ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n",
    "ouroboros/mod_b.py": "GEN = 'GEN_CAND'\n",
    "ouroboros/added.py": "GEN = 'GEN_CAND'\n",
    "web/sentinel.txt": "candidate ui\n",
}


def rich_commit(candidate):
    """A candidate commit that modifies, adds and deletes body files and rewrites the entry."""
    server = (pathlib.Path(candidate) / "server.py").read_text(encoding="utf-8")
    return candidate_commit(candidate, files={**RICH_CHANGE, "server.py": server.replace(
        '"server": "GEN_OLD"', '"server": "GEN_CAND"')}, delete=("ouroboros/gone.py",))


def restart_receipt(data, expected_sha, reason, branch="ouroboros"):
    """The exact restart receipt ``request_restart`` writes (the existing verify marker)."""
    from supervisor.evolution_lifecycle import write_pending_restart_marker

    return write_pending_restart_marker(data, expected_sha=expected_sha, expected_branch=branch, reason=reason)
