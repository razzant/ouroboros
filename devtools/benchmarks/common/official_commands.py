"""Declarative official benchmark command builders."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys


def resolve_programbench_python() -> str:
    """Return a Python executable that can import programbench (needs typing.Self, 3.11+)."""
    override = os.environ.get("PROGRAMBENCH_PYTHON", "").strip()
    if override:
        return override
    candidates: list[str] = []
    for name in (sys.executable, "python3.12", "python3.11", "python3"):
        path = shutil.which(name) if name != sys.executable else sys.executable
        if path and path not in candidates:
            candidates.append(path)
    for python in candidates:
        if _python_supports_programbench(python):
            return python
    raise RuntimeError(
        "programbench eval requires Python 3.11+ (typing.Self). "
        "Install programbench on python3.11+ or set PROGRAMBENCH_PYTHON."
    )


def _python_supports_programbench(python: str) -> bool:
    try:
        proc = subprocess.run(
            [python, "-c", "from typing import Self; import programbench"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def resolve_programbench_cli() -> list[str]:
    """Argv prefix for the official programbench console script."""
    override = os.environ.get("PROGRAMBENCH_CLI", "").strip()
    if override:
        return [override]
    python = resolve_programbench_python()
    try:
        proc = subprocess.run(
            [python, "-c", "import sysconfig; print(sysconfig.get_path('scripts'))"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=True,
        )
        cli = pathlib.Path((proc.stdout or "").strip()) / "programbench"
        if cli.is_file():
            return [str(cli)]
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    found = shutil.which("programbench")
    if found:
        return [found]
    raise RuntimeError(
        f"programbench CLI not found for {python}. "
        "Install with: python3.11 -m pip install programbench"
    )


def programbench_base_cmd() -> list[str]:
    return resolve_programbench_cli()


def programbench_eval_cmd(run_root: pathlib.Path, *, docker_cpus: int | None = None) -> list[str]:
    cpus = docker_cpus if docker_cpus is not None else int(os.environ.get("PROGRAMBENCH_DOCKER_CPUS", "4"))
    return [*programbench_base_cmd(), "eval", str(run_root), "--docker-cpus", str(cpus)]


def programbench_info_cmd(run_root: pathlib.Path) -> list[str]:
    return [*programbench_base_cmd(), "info", str(run_root)]


def programbench_command_for_manifest(run_root: pathlib.Path, *, eval_requested: bool) -> list[str]:
    """Best-effort official argv for run manifests.

    Failure sidecars must not require a 3.11+ programbench install before
    cleanroom preflight errors surface; fall back to a generic argv shape.
    """
    try:
        return programbench_eval_cmd(run_root) if eval_requested else programbench_info_cmd(run_root)
    except RuntimeError:
        verb = "eval" if eval_requested else "info"
        return ["programbench", verb, str(run_root)]


def swebench_eval_cmd(dataset_name: str, predictions_path: pathlib.Path, run_id: str, workers: int = 1) -> list[str]:
    return [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(int(workers)),
        "--run_id",
        run_id,
    ]
