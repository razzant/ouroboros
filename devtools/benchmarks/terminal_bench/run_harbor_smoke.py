#!/usr/bin/env python3
"""Build and optionally run a Harbor Terminal-Bench smoke command."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import secrets
import shlex
import subprocess
import sys
from datetime import datetime
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from devtools.benchmarks.common.manifests import (
    admit_benchmark_run,
    finalize_run_manifest,
    write_json,
)
from devtools.benchmarks.common.model_slots import (
    fixed_model_actor_snapshot,
)
from devtools.benchmarks.common.result_index import task_result_row, write_result_index
from devtools.benchmarks.common.run_roots import (
    default_settings_path,
    assert_file_output_outside_repo,
    assert_outside_repo,
    repo_root_from_devtools,
    run_root as default_run_root,
    safe_join_under,
)


AGENT_IMPORT = "devtools.benchmarks.terminal_bench.harbor_installed_agent:OuroborosTerminalBenchAgent"


def generated_job_name() -> str:
    """Invocation-unique Harbor job name in Harbor's timestamp shape plus a random suffix."""
    return f"{datetime.now():%Y-%m-%d__%H-%M-%S}__{secrets.token_hex(4)}"


def add_force_build_option(parser: argparse.ArgumentParser) -> None:
    """Harbor's own image flag pair, shared by both launchers; independent of `--execute`."""
    parser.add_argument(
        "--force-build",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="pass Harbor's --force-build or --no-force-build; default: neither, so Harbor's "
             "job configuration decides (its default prefers a task's published image)",
    )


def force_build_flag(force_build: bool | None) -> list[str]:
    """Harbor's flag for an explicit choice; nothing when the choice is unset."""
    if force_build is None:
        return []
    return ["--force-build" if force_build else "--no-force-build"]


def agent_environment_force_build(
    force_build: bool | None, job_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Configured `environment.force_build` and its source (CLI > job config > Harbor's false).

    Only a literal boolean is resolved; any other job-config value is recorded as None and
    left for Harbor to validate. A configured policy, not a record of what Harbor pulled or built."""
    if force_build is not None:
        return {"force_build": bool(force_build), "source": "cli"}
    environment = (job_config or {}).get("environment", {})
    if not isinstance(environment, Mapping):
        return {"force_build": None, "source": "base_job_config"}
    if "force_build" in environment:
        value = environment["force_build"]
        return {"force_build": value if isinstance(value, bool) else None, "source": "base_job_config"}
    return {"force_build": False, "source": "harbor_default"}


def harbor_command(
    *,
    task_names: list[str],
    model: str,
    run_root: pathlib.Path,
    dataset: str = "terminal-bench/terminal-bench-2-1",
    harbor_bin: str = "harbor",
    n_tasks: int = 1,
    n_concurrent: int = 1,
    k: int = 1,
    agent_setup_timeout_multiplier: float = 1.0,
    environment_build_timeout_multiplier: float = 1.0,
    light_model: str = "",
    options: dict[str, Any] | None = None,
) -> list[str]:
    opts = dict(options or {})
    host_settings_path = str(opts.get("host_settings_path") or "")
    # Keep the light model pinned to the main model by default; v6.27.0 otherwise
    # defaults it to google/gemini-3.5-flash, which would diverge from a pure-model run.
    effective_light_model = light_model or model
    cmd = [
        harbor_bin,
        "run",
        "--dataset",
        dataset,
        "--agent-import-path",
        AGENT_IMPORT,
        "--model",
        f"ouroboros-{model.replace('/', '-')}",
        "--agent-kwarg",
        f"ouroboros_model={model}",
        "--agent-kwarg",
        f"ouroboros_light_model={effective_light_model}",
        "--agent-kwarg",
        "install_timeout_sec=1200",
        "--agent-kwarg",
        "server_start_timeout_sec=240",
        # Dataset identity for the adapter's per-task cache lookup (org is not a constant).
        "--agent-kwarg",
        f"dataset={dataset}",
    ]
    if host_settings_path:
        cmd.extend(["--agent-kwarg", f"host_settings_path={host_settings_path}"])
    cmd.extend(
        [
            "--agent-setup-timeout-multiplier",
            str(float(agent_setup_timeout_multiplier)),
            "--environment-build-timeout-multiplier",
            str(float(environment_build_timeout_multiplier)),
            "-k",
            str(int(k)),
            "--n-concurrent",
            str(int(n_concurrent)),
            "--n-tasks",
            str(int(n_tasks)),
            "--jobs-dir",
            str(run_root),
            "--yes",
        ]
    )
    job_name = str(opts.get("job_name") or "").strip()
    if job_name:
        cmd.extend(["--job-name", job_name])
    for task_name in task_names:
        cmd.extend(["--include-task-name", task_name])
    cmd.extend(force_build_flag(opts.get("force_build")))
    return cmd


def _own_harbor_result(run_root: pathlib.Path, job_name: str) -> pathlib.Path:
    """Expected `<run_root>/<job_name>/result.json` for this launch's own Harbor job."""
    job_dir = safe_join_under(run_root, job_name)
    result_path = safe_join_under(job_dir, "result.json")
    if not result_path.is_file():
        raise RuntimeError(f"expected Harbor result.json for job {job_name!r}, not found at {result_path}")
    return result_path


def _harbor_task_outcomes(result_path: pathlib.Path) -> list[dict[str, object]]:
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    stats = data.get("stats") if isinstance(data.get("stats"), dict) else {}
    evals = stats.get("evals") if isinstance(stats.get("evals"), dict) else {}
    outcomes: list[dict[str, object]] = []
    seen: set[str] = set()
    for eval_summary in evals.values():
        if not isinstance(eval_summary, dict):
            continue
        reward_stats = eval_summary.get("reward_stats") if isinstance(eval_summary.get("reward_stats"), dict) else {}
        rewards = reward_stats.get("reward") if isinstance(reward_stats.get("reward"), dict) else {}
        for reward_text, task_ids in rewards.items():
            if not isinstance(task_ids, list):
                continue
            try:
                reward_value = float(str(reward_text))
            except ValueError:
                reward_value = None
            for task_id in task_ids:
                instance_id = str(task_id or "").strip()
                if not instance_id or instance_id in seen:
                    continue
                seen.add(instance_id)
                outcomes.append({"instance_id": instance_id, "reward": reward_value})
    return sorted(outcomes, key=lambda item: str(item["instance_id"]))


def _harbor_child_env(repo_root: pathlib.Path,
                      pinned_env: dict[str, str]) -> dict[str, str]:
    env = dict(pinned_env)
    existing = env.get("PYTHONPATH", "")
    entries = [str(repo_root)]
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", default=[], help="task name; repeat for a deterministic subset")
    parser.add_argument("--model", default="google/gemini-3.5-flash")
    parser.add_argument("--dataset", default="terminal-bench/terminal-bench-2-1")
    parser.add_argument("--harbor-bin", default="harbor")
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("-k", "--k", type=int, default=1, dest="k", help="trials per task (harbor -k); default 1")
    parser.add_argument("--agent-setup-timeout-multiplier", type=float, default=1.0, help="harbor agent-setup timeout multiplier; default 1.0 (official)")
    parser.add_argument("--environment-build-timeout-multiplier", type=float, default=1.0, help="harbor environment-build timeout multiplier; default 1.0 (official)")
    parser.add_argument("--ouroboros-light-model", default="", help="light model kwarg; default = main --model (avoids v6.27.0 gemini-3.5-flash default)")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--settings-path", default="")
    parser.add_argument("--isolated-data-root", default="")
    parser.add_argument("--ledger-output", default="", help="denominator-preserving JSONL result ledger")
    parser.add_argument(
        "--allow-dirty-seed",
        action="store_true",
        help="record and proceed with an unclean/unidentifiable seed checkout instead of refusing",
    )
    parser.add_argument("--execute", action="store_true")
    add_force_build_option(parser)
    args = parser.parse_args()

    repo_root = repo_root_from_devtools()
    effective_light = str(args.ouroboros_light_model or "").strip() or args.model
    child_env = dict(os.environ)
    fixed_actor = fixed_model_actor_snapshot(
        args.model,
        light_model=effective_light,
        target=child_env,
    )
    actor_slots = fixed_actor["model_slots"]
    settings_path = pathlib.Path(args.settings_path).expanduser() if args.settings_path else default_settings_path()
    run_root = assert_outside_repo(
        pathlib.Path(args.run_root).expanduser() if args.run_root else default_run_root("terminal_bench"),
        repo_root,
    )
    # The shared seed gate below (benchmark_run_manifest, fail-closed since v6.75.0) is now the
    # single authority on source cleanliness; the old opt-in --require-clean-source check was a
    # weaker duplicate of it.
    actual_include_filters = args.task or []
    effective_n_tasks = len(actual_include_filters) if actual_include_filters else int(args.n_tasks)
    task_names = actual_include_filters or [f"selection-slot-{idx + 1}" for idx in range(effective_n_tasks)]
    job_name = generated_job_name()
    cmd = harbor_command(
        task_names=actual_include_filters,
        model=actor_slots["OUROBOROS_MODEL"],
        run_root=run_root,
        dataset=args.dataset,
        harbor_bin=args.harbor_bin,
        n_tasks=effective_n_tasks,
        n_concurrent=args.n_concurrent,
        k=args.k,
        agent_setup_timeout_multiplier=args.agent_setup_timeout_multiplier,
        environment_build_timeout_multiplier=args.environment_build_timeout_multiplier,
        light_model=actor_slots["OUROBOROS_MODEL_LIGHT"],
        options={"force_build": args.force_build, "host_settings_path": str(settings_path), "job_name": job_name},
    )
    ledger_output = (
        assert_file_output_outside_repo(pathlib.Path(args.ledger_output), repo_root)
        if args.ledger_output
        else run_root / "result_index.jsonl"
    )
    manifest_path = run_root / "run_manifest.json"
    # Admission is the outermost REFUSAL point, not the outermost side effect: `ensure_outside_repo`
    # above already created `run_root` (it mkdirs). So a refused run leaves that directory holding
    # nothing but the persisted refusal — no harbor invocation, no ledger, no settings render.
    manifest = admit_benchmark_run(
        manifest_path,
        benchmark="terminal_bench",
        run_root=run_root,
        repo_dir=repo_root,
        requested_task_ids=list(actual_include_filters),
        require_clean=not args.allow_dirty_seed,
        argv=sys.argv,
        output_paths={"harbor_output_dir": str(run_root)},
        dataset=args.dataset,
        harness={"agent_import": AGENT_IMPORT, "harbor_bin": args.harbor_bin},
        official_command=cmd,
        timeout_sec=None,
        isolated_data_root=args.isolated_data_root,
        settings_path=settings_path,
        extra={
            "outcome": "started",
            "n_tasks": effective_n_tasks,
            "requested_n_tasks_arg": int(args.n_tasks),
            "n_concurrent": int(args.n_concurrent),
            "source_dirty_allowed": bool(args.allow_dirty_seed),
            "agent_environment_force_build": agent_environment_force_build(args.force_build),
            "selection": {
                "mode": "explicit_task_ids" if actual_include_filters else "deterministic_first_n",
                "requested_slots": task_names,
                "include_filters": actual_include_filters,
            },
        },
    )
    manifest["model_slots"] = {
        key: actor_slots[key]
        for key in (
            "OUROBOROS_MODEL",
            "OUROBOROS_MODEL_LIGHT",
            "OUROBOROS_MODEL_FALLBACKS",
        )
    }
    manifest["available_subagents"] = fixed_actor["available_subagents"]
    manifest["harness"]["fixed_model_actor"] = fixed_actor
    # Durable before Harbor can spend: no ambient settings model/Heavy can survive.
    write_json(manifest_path, manifest)
    if not actual_include_filters:
        manifest["requested_count"] = effective_n_tasks
    with finalize_run_manifest(manifest_path, manifest) as final:
        (run_root / "harbor_command.json").write_text(json.dumps({"run_root": str(run_root), "cmd": cmd, "agent_import": AGENT_IMPORT}, indent=2), encoding="utf-8")
        status = "planned"
        reason = "command_generated"
        official_eval_status = "not_run"
        print(shlex.join(cmd))
        returncode = 0
        harbor_result: pathlib.Path | None = None
        harbor_result_error = ""
        observed_outcomes: list[dict[str, object]] = []
        if args.execute:
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=repo_root,
                    env=_harbor_child_env(repo_root, child_env),
                )
            except Exception as exc:
                harbor_result_error = f"{type(exc).__name__}: {exc}"
                status = "harness_failed"
                reason = "harbor_invocation_failed"
                official_eval_status = "failed"
                returncode = 2
            else:
                returncode = completed.returncode
                status = "harness_completed" if returncode == 0 else "harness_failed"
                reason = "harbor_returncode_0" if returncode == 0 else "harbor_returncode_nonzero"
                official_eval_status = "completed" if returncode == 0 else "failed"
                try:
                    harbor_result = _own_harbor_result(run_root, job_name)
                except Exception as exc:
                    harbor_result_error = str(exc)
                    status = "harness_failed"
                    reason = "harbor_result_unresolved"
                    official_eval_status = "failed"
                    returncode = returncode or 2
            if harbor_result is not None:
                observed_outcomes = _harbor_task_outcomes(harbor_result)
                observed_ids = {str(item.get("instance_id") or "") for item in observed_outcomes}
                expected_ids = set(actual_include_filters)
                if not observed_outcomes and effective_n_tasks > 0:
                    harbor_result_error = "Harbor result contained no parseable task outcomes"
                elif expected_ids and not observed_ids.issubset(expected_ids):
                    extras = sorted(observed_ids - expected_ids)
                    harbor_result_error = f"Harbor result included unexpected task ids: {extras}"
                elif expected_ids:
                    missing = sorted(expected_ids - observed_ids)
                    if missing:
                        harbor_result_error = f"Harbor result omitted requested task ids: {missing}"
                elif len(observed_outcomes) != effective_n_tasks:
                    harbor_result_error = (
                        f"Harbor result completed {len(observed_outcomes)} outcomes, expected {effective_n_tasks}"
                    )
                if harbor_result_error:
                    status = "harness_failed"
                    reason = "harbor_result_unresolved"
                    official_eval_status = "failed"
                    returncode = returncode or 2
                # The finalization seam rewrites the manifest on EVERY exit path, so these
                # observed-result fields need no separate write here.
                manifest["output_paths"]["harbor_result"] = str(harbor_result)
                manifest["observed_task_ids"] = [str(item["instance_id"]) for item in observed_outcomes]
                manifest["official_result_summary"] = {
                    "result_path": str(harbor_result),
                    "completed_outcomes": len(observed_outcomes),
                }
        ledger_tasks: list[dict[str, object]] = []
        if observed_outcomes and not harbor_result_error:
            ledger_tasks.extend(observed_outcomes)
            recorded_ids = {str(item.get("instance_id") or "") for item in ledger_tasks}
            if actual_include_filters:
                for task in task_names:
                    if task not in recorded_ids:
                        ledger_tasks.append({"instance_id": task, "reward": None})
            while not actual_include_filters and len(ledger_tasks) < effective_n_tasks:
                ledger_tasks.append({"instance_id": f"selection-slot-missing-{len(ledger_tasks) + 1}", "reward": None})
        else:
            ledger_tasks.extend({"instance_id": task, "reward": None} for task in task_names)
        write_result_index(
            ledger_output,
            [
                task_result_row(
                    benchmark="terminal_bench",
                    instance_id=str(task["instance_id"]),
                    status=status,
                    metadata={
                        "reason_code": reason,
                        "official_eval_status": official_eval_status,
                        "output_paths": {
                            "harbor_output_dir": str(run_root),
                            "manifest": str(run_root / "run_manifest.json"),
                            "harbor_result": str(harbor_result) if harbor_result is not None else "",
                        },
                        "error": harbor_result_error,
                        "details": {
                            "returncode": returncode,
                            "include_filters": actual_include_filters,
                            "n_tasks": effective_n_tasks,
                            "official_reward": task.get("reward"),
                        },
                    },
                )
                for task in ledger_tasks
            ],
        )
        if not args.execute:
            final.update({"outcome": "command_generated", "exit_code": 0})
            return 0
        final.update({"outcome": status, "exit_code": int(returncode)})
        return returncode


if __name__ == "__main__":
    raise SystemExit(main())
