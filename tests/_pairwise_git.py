"""Source checks against the event's target commit."""
from __future__ import annotations

import pathlib
import subprocess

BASE_REF_ENV = "OURO_SIZE_RATCHET_BASE_REF"


def git_read(repo: pathlib.Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, check=check, capture_output=True)


def growth_base(repo: pathlib.Path, ref: str | None) -> str:
    """The event base, or an empty string when this clone cannot measure the change."""
    ref = (ref or "").strip()
    if not ref:
        return ""
    resolved = git_read(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
    return resolved.stdout.decode("ascii").strip() if resolved.returncode == 0 else ""
