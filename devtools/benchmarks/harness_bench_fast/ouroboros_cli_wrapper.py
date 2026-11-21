#!/usr/bin/env python3
"""Thin CLI adapter used by harness-bench-fast ``run-cli``.

``harness_bench run-cli`` invokes the configured command with:

    cwd=<fresh task workspace>
    argv=[...configured command..., <official task prompt>]

This wrapper preserves that contract and delegates to ``ouroboros run`` with the
current working directory as the external workspace root. The wrapper writes
per-task logs outside the transient benchmark workspace so they survive cleanup.
Downstream the prompt is handed to ``ouroboros run`` via ``--prompt-file`` (the
durable ``prompt.txt`` in the log dir), never as an argv tail (E2BIG hygiene).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any


DEFAULT_REPO = pathlib.Path(__file__).resolve().parents[3]
# Invoked as a SCRIPT by `harness_bench run-cli`, so the repo root is not implicitly on the
# path the way it is for the package-imported adapters.
if str(DEFAULT_REPO) not in sys.path:
    sys.path.insert(0, str(DEFAULT_REPO))

from devtools.benchmarks.common.model_slots import (  # noqa: E402
    fixed_model_actor_snapshot,
)
from devtools.benchmarks.common.result_index import runtime_terminal_disclosure  # noqa: E402

DEFAULT_DATA = DEFAULT_REPO.parent / "data"
DEFAULT_SETTINGS = DEFAULT_DATA / "settings.json"
DEFAULT_OUROBOROS_BIN = DEFAULT_REPO.parent / ".venv" / "bin" / "ouroboros"


def _safe_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return text[:96] or "task"


def _task_id_from_workspace(workspace: pathlib.Path) -> str:
    name = workspace.name
    if name.startswith("hb_cli_"):
        stem = name.removeprefix("hb_cli_")
        parts = stem.split("_")
        if len(parts) > 1:
            return "_".join(parts[:-1]) or stem
    if name.startswith("hb_cli_prom_"):
        stem = name.removeprefix("hb_cli_prom_")
        parts = stem.split("_")
        if len(parts) > 1:
            return "_".join(parts[:-1]) or stem
    return name


def _read_json(path: pathlib.Path) -> Any:
    """Best-effort read; an absent/corrupt result file must not fail the wrapper."""
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - disclosure is best-effort, the run's exit code is not
        return None


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ensure_workspace_git_root(workspace: pathlib.Path) -> dict[str, Any]:
    """Make harness-bench's temp workspace acceptable as Ouroboros external workspace."""

    probe = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=workspace,
        text=True,
        capture_output=True,
    )
    if probe.returncode == 0 and pathlib.Path(probe.stdout.strip()).resolve(strict=False) == workspace:
        return {"created": False, "status": "already_git_root"}
    if probe.returncode == 0:
        return {"created": False, "status": "nested_git_root", "root": probe.stdout.strip()}

    init = subprocess.run(["git", "init"], cwd=workspace, text=True, capture_output=True)
    config_email = subprocess.run(
        ["git", "config", "user.email", "ouroboros-harness-bench@example.invalid"],
        cwd=workspace,
        text=True,
        capture_output=True,
    )
    config_name = subprocess.run(
        ["git", "config", "user.name", "Ouroboros Harness Bench"],
        cwd=workspace,
        text=True,
        capture_output=True,
    )
    return {
        "created": init.returncode == 0,
        "status": "initialized" if init.returncode == 0 else "init_failed",
        "init_returncode": init.returncode,
        "init_stderr": init.stderr[-500:],
        "config_email_returncode": config_email.returncode,
        "config_name_returncode": config_name.returncode,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default=str(DEFAULT_REPO))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--settings-path", default=str(DEFAULT_SETTINGS))
    parser.add_argument("--ouroboros-bin", default=os.environ.get("OUROBOROS_BIN") or str(DEFAULT_OUROBOROS_BIN))
    parser.add_argument("--model", required=True)
    parser.add_argument("--log-root", default="")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--memory-mode", default="empty", choices=["empty", "forked", "shared"])
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("prompt", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    prompt_parts = list(args.prompt or [])
    if prompt_parts and prompt_parts[0] == "--":
        prompt_parts = prompt_parts[1:]
    prompt = " ".join(prompt_parts).strip()
    if args.prompt_file:
        prompt = pathlib.Path(args.prompt_file).read_text(encoding="utf-8")
    if not prompt:
        print("harness-bench-fast wrapper: missing prompt", file=sys.stderr)
        return 2

    workspace = pathlib.Path.cwd().resolve(strict=False)
    git_workspace = _ensure_workspace_git_root(workspace)
    repo_dir = pathlib.Path(args.repo_dir).expanduser().resolve(strict=False)
    data_dir = pathlib.Path(args.data_dir).expanduser().resolve(strict=False)
    settings_path = pathlib.Path(args.settings_path).expanduser().resolve(strict=False)
    task_id = _task_id_from_workspace(workspace)
    log_root = pathlib.Path(args.log_root).expanduser().resolve(strict=False) if args.log_root else data_dir / "bench_logs" / "harness_bench_fast"
    run_id = f"{_safe_slug(task_id)}_{int(time.time())}_{os.getpid()}"
    log_dir = log_root / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = log_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    summary_path = log_dir / "summary.json"
    result_json = log_dir / "ouroboros-task-result.json"
    started = time.time()

    env = os.environ.copy()
    fixed_actor = fixed_model_actor_snapshot(args.model, target=env)
    env.update(
        {
            "OUROBOROS_REPO_DIR": str(repo_dir),
            "OUROBOROS_DATA_DIR": str(data_dir),
            "OUROBOROS_SETTINGS_PATH": str(settings_path),
            "OUROBOROS_RUNTIME_MODE": "pro",
            "PYTHONUNBUFFERED": "1",
        }
    )
    cmd = [
        str(pathlib.Path(args.ouroboros_bin).expanduser()),
        "run",
        "--workspace",
        str(workspace),
        "--memory-mode",
        args.memory_mode,
        "--quiet",
        "--timeout",
        str(max(0, int(args.timeout))),
        "--result-json-out",
        str(result_json),
        "--actor-id",
        "harness-bench-fast",
    ]
    if not args.no_start_server:
        cmd.append("--start")
    # E2BIG hygiene (C5): the prompt already lives on disk above; it travels to
    # `ouroboros run` as a file, never as an argv tail.
    cmd.extend(["--prompt-file", str(prompt_path)])

    def _run_once() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            cwd=str(repo_dir),
            env=env,
            text=True,
            capture_output=True,
            timeout=max(1, int(args.timeout)) + 90,
        )

    # The shared server can return HTTP 503 "supervisor is still starting" on a
    # cold start (or mid-run supervisor restart). That is an infra startup race,
    # not the task's own time budget, so retry a few times instead of scoring
    # the task as an instant failure.
    completed = _run_once()
    startup_retries = 0
    while (
        completed.returncode != 0
        and startup_retries < 18
        and "supervisor is still starting" in (completed.stderr or "")
    ):
        time.sleep(10)
        startup_retries += 1
        completed = _run_once()

    (log_dir / "stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (log_dir / "stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    _write_json(
        summary_path,
        {
            "schema": "ouroboros.harness_bench_fast.wrapper.v1",
            "task_id_guess": task_id,
            "workspace": str(workspace),
            "workspace_git": git_workspace,
            "repo_dir": str(repo_dir),
            "data_dir": str(data_dir),
            "settings_path": str(settings_path),
            "model": args.model,
            "model_slots": fixed_actor["model_slots"],
            "available_subagents": fixed_actor["available_subagents"],
            "fixed_model_actor": fixed_actor,
            "memory_mode": args.memory_mode,
            "returncode": completed.returncode,
            "startup_retries": startup_retries,
            "elapsed_sec": round(time.time() - started, 3),
            "stdout_chars": len(completed.stdout or ""),
            "stderr_chars": len(completed.stderr or ""),
            "result_json": str(result_json),
            # The runtime task result is written to `result_json` by `--result-json-out`, but
            # the summary only ever POINTED at it, so a task the runtime truncated (per-task
            # USD rail, round cap) was indistinguishable from an honest verifier failure in
            # every artefact a reader of this summary sees.
            "runtime_outcome": runtime_terminal_disclosure(_read_json(result_json)),
        },
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
