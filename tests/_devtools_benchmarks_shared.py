"""Repo helpers shared by the benchmark devtools suites.

Split out of ``tests/test_devtools_benchmarks.py`` when that module was divided by theme;
the definitions are verbatim, so every sibling suite gets the same throwaway git repo, the
same bench-runs isolation and the same repo root it was written against.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest



REPO_ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def _isolate_bench_runs_root(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_BENCH_RUNS_ROOT", str(tmp_path / "bench_runs"))
    # Command-construction tests inspect the raw solver argv; the GAIA bwrap
    # answer-cache isolation (default-on at runtime) would prepend a `bwrap … --`
    # prefix and SystemExit where bwrap is absent (CI). Disable by default; the
    # dedicated bwrap test re-enables it explicitly.
    monkeypatch.setenv("GAIA_BWRAP_ISOLATE", "0")

def _git_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "app.py").write_text("print('base')\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()

def _git_commit_all(repo: Path) -> None:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t.t", "-c", "user.name=t", "commit", "-qm", "seed"],
        check=True,
        capture_output=True,
    )
