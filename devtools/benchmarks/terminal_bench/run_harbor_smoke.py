#!/usr/bin/env python3
"""Build and optionally run a Harbor Terminal-Bench smoke command."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import secrets
import shlex
import subprocess
import sys
from collections.abc import Iterator
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


def direct_trial_results(job_dir: pathlib.Path) -> Iterator[tuple[str, bytes | None]]:
    """Yield `(trial, bytes of its result.json)` one direct child at a time, in name order; None
    when the trial or its result cannot be read or resolves outside the job. Skips children without
    a result.json; nothing deeper is read. Raises OSError when `job_dir` cannot be listed."""
    with os.scandir(job_dir) as entries:
        names = sorted(entry.name for entry in entries)
    for trial in names:
        try:
            raw: bytes | None = safe_join_under(job_dir, trial, "result.json").read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            continue
        except (OSError, ValueError):
            raw = None
        yield trial, raw


def _own_harbor_result(run_root: pathlib.Path, job_name: str) -> pathlib.Path:
    """Expected `<run_root>/<job_name>/result.json` for this launch's own Harbor job."""
    job_dir = safe_join_under(run_root, job_name)
    result_path = safe_join_under(job_dir, "result.json")
    if not result_path.is_file():
        raise RuntimeError(f"expected Harbor result.json for job {job_name!r}, not found at {result_path}")
    return result_path


def _parse_job_reward_stats(data: Mapping[str, Any]) -> tuple[dict[str, float], list[str]]:
    """Parse `stats.evals.*.reward_stats.reward` (`trial_name -> float`) from Harbor `result.json`."""
    stats = data.get("stats")
    if not isinstance(stats, dict):
        return {}, ["Harbor result.json missing 'stats' mapping"]
    evals = stats.get("evals")
    if evals is None:
        return {}, []
    if not isinstance(evals, dict):
        return {}, ["Harbor result.json has invalid 'stats.evals'"]
    by_trial: dict[str, float] = {}
    errors: list[str] = []
    for eval_key, eval_summary in evals.items():
        if not isinstance(eval_summary, dict):
            errors.append(f"Harbor result.json has invalid eval summary {eval_key!r}")
            continue
        reward_stats = eval_summary.get("reward_stats")
        if reward_stats is None:
            continue
        if not isinstance(reward_stats, dict):
            errors.append(f"Harbor result.json has invalid reward_stats in {eval_key!r}")
            continue
        rewards = reward_stats.get("reward")
        if rewards is None:
            continue
        if not isinstance(rewards, dict):
            errors.append(f"Harbor result.json has invalid reward_stats.reward in {eval_key!r}")
            continue
        for reward_text, trial_names in rewards.items():
            try:
                reward_value = float(str(reward_text))
            except ValueError:
                errors.append(f"Harbor result.json has non-numeric reward bucket {reward_text!r}")
                continue
            if not math.isfinite(reward_value):
                errors.append(f"Harbor result.json has non-finite reward bucket {reward_text!r}")
                continue
            if not isinstance(trial_names, list):
                errors.append(f"Harbor result.json reward bucket {reward_text!r} is not a list")
                continue
            for raw_name in trial_names:
                if not isinstance(raw_name, str) or not raw_name.strip():
                    errors.append(f"Harbor result.json reward bucket {reward_text!r} has invalid trial name")
                    continue
                trial_name = raw_name.strip()
                if trial_name in by_trial:
                    errors.append(f"Harbor result.json lists trial {trial_name!r} more than once")
                    continue
                by_trial[trial_name] = reward_value
    return by_trial, errors


def _parse_trial_reward(data: Mapping[str, Any], trial_name: str) -> tuple[float | None, str]:
    """Extract numeric `verifier_result.rewards.reward` from a Harbor TrialResult."""
    verifier_result = data.get("verifier_result")
    if verifier_result is None:
        return None, ""
    if not isinstance(verifier_result, dict):
        return None, f"trial {trial_name!r} has invalid verifier_result"
    rewards = verifier_result.get("rewards")
    if rewards is None or rewards == {}:
        return None, ""
    if not isinstance(rewards, dict):
        return None, f"trial {trial_name!r} has invalid verifier_result.rewards"
    if "reward" not in rewards:
        return None, f"trial {trial_name!r} verifier_result.rewards missing 'reward' key: {sorted(rewards)}"
    raw_reward = rewards["reward"]
    if raw_reward is None:
        return None, ""
    if isinstance(raw_reward, bool) or not isinstance(raw_reward, (int, float)):
        return None, f"trial {trial_name!r} has non-numeric reward {raw_reward!r}"
    value = float(raw_reward)
    if not math.isfinite(value):
        return None, f"trial {trial_name!r} has non-finite reward {raw_reward!r}"
    return value, ""


def _read_harbor_job_outcomes(result_path: pathlib.Path) -> tuple[list[dict[str, object]], list[str]]:
    """Read per-task outcomes from this job's direct `<trial>/result.json` outputs, cross-checked
    against `<job>/result.json`.

    Harbor 0.18 writes `trial_result.trial_name` (truncated + random suffix), not `task_name`, into
    `stats.evals.*.reward_stats.reward`. Authoritative `task_name` lives in each direct trial
    `result.json`; we never split a trial-name suffix."""
    try:
        job_data = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [f"Harbor result.json unreadable or malformed: {exc}"]
    if not isinstance(job_data, dict):
        return [], ["Harbor result.json must be a JSON object"]
    job_rewards, errors = _parse_job_reward_stats(job_data)

    job_dir = result_path.parent
    direct_trials: dict[str, dict[str, Any]] = {}
    try:
        trial_entries = list(direct_trial_results(job_dir))
    except OSError as exc:
        return [], [*errors, f"cannot read Harbor job directory {job_dir}: {exc}"]
    for trial_dir, raw in trial_entries:
        if raw is None:
            errors.append(f"unreadable or out-of-job trial result: {trial_dir}")
            continue
        try:
            trial_data = json.loads(raw)
        except ValueError:
            errors.append(f"malformed trial result JSON: {trial_dir}")
            continue
        if not isinstance(trial_data, dict):
            errors.append(f"trial result is not a JSON object: {trial_dir}")
            continue
        raw_task = trial_data.get("task_name")
        if not isinstance(raw_task, str) or not raw_task.strip():
            errors.append(f"trial {trial_dir!r} missing authoritative task_name")
            continue
        raw_trial = trial_data.get("trial_name")
        if raw_trial is not None and (not isinstance(raw_trial, str) or not raw_trial.strip()):
            errors.append(f"trial {trial_dir!r} has invalid trial_name")
            continue
        trial_name = raw_trial.strip() if isinstance(raw_trial, str) else trial_dir
        if trial_name != trial_dir:
            errors.append(f"trial directory {trial_dir!r} disagrees with trial_name {trial_name!r}")
            continue
        if trial_name in direct_trials:
            errors.append(f"ambiguous duplicate trial_name {trial_name!r}")
            continue
        reward, reward_error = _parse_trial_reward(trial_data, trial_name)
        if reward_error:
            errors.append(reward_error)
            continue
        exc_info = trial_data.get("exception_info")
        exc_type = (
            exc_info.get("exception_type")
            if isinstance(exc_info, dict) and isinstance(exc_info.get("exception_type"), str)
            else None
        )
        direct_trials[trial_name] = {
            "task_name": raw_task.strip(),
            "trial_name": trial_name,
            "reward": reward,
            "exception_type": exc_type,
        }

    missing_direct = sorted(set(job_rewards) - set(direct_trials))
    if missing_direct:
        errors.append(f"Harbor result.json references trial(s) missing direct trial outputs: {missing_direct}")
    for trial_name, trial in sorted(direct_trials.items()):
        if trial_name in job_rewards:
            if trial["reward"] is None or trial["reward"] != job_rewards[trial_name]:
                errors.append(
                    f"reward mismatch for trial {trial_name!r}: "
                    f"job result.json={job_rewards[trial_name]!r}, trial result={trial['reward']!r}"
                )
        elif trial["reward"] is not None:
            errors.append(f"scored direct trial {trial_name!r} missing from Harbor result.json reward_stats")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for trial in sorted(direct_trials.values(), key=lambda item: item["trial_name"]):
        grouped.setdefault(trial["task_name"], []).append(trial)
    outcomes: list[dict[str, object]] = []
    for task_name, task_trials in sorted(grouped.items()):
        scored = [item["reward"] for item in task_trials if item["reward"] is not None]
        all_scored = len(scored) == len(task_trials) and bool(task_trials)
        agg_reward = (sum(scored) / len(scored)) if all_scored else None
        attempts = [
            {
                "trial_name": item["trial_name"],
                "reward": item["reward"],
                **({"exception_type": item["exception_type"]} if item["exception_type"] else {}),
            }
            for item in task_trials
        ]
        outcomes.append({
            "instance_id": task_name,
            "reward": agg_reward,
            "trial_count": len(task_trials),
            "scored_trials": len(scored),
            "trials": attempts,
        })
    return outcomes, errors


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
    if args.k < 1:
        parser.error("-k/--k must be >= 1")
    if not args.task and args.n_tasks < 0:
        parser.error("--n-tasks must be >= 0")

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
    actual_include_filters = list(dict.fromkeys(args.task or []))
    effective_n_tasks = len(actual_include_filters) if actual_include_filters else int(args.n_tasks)
    task_names = actual_include_filters or [f"selection-slot-{idx + 1}" for idx in range(effective_n_tasks)]
    job_name = generated_job_name()
    invocation_manifest_path = safe_join_under(run_root, f"run_manifest.{job_name}.json")
    own_job_dir = safe_join_under(run_root, job_name)
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
    # Admission is the outermost REFUSAL point and the first side effect: `assert_outside_repo`
    # above is pure, so a refused run leaves `run_root` holding only the persisted refusal — no
    # per-invocation manifest, no harbor invocation, no ledger, no command file.
    manifest = admit_benchmark_run(
        manifest_path,
        benchmark="terminal_bench",
        run_root=run_root,
        repo_dir=repo_root,
        requested_task_ids=list(actual_include_filters),
        require_clean=not args.allow_dirty_seed,
        argv=sys.argv,
        output_paths={
            "harbor_output_dir": str(run_root),
            "harbor_job_dir": str(own_job_dir),
            "manifest": str(manifest_path),
        },
        dataset=args.dataset,
        harness={"agent_import": AGENT_IMPORT, "harbor_bin": args.harbor_bin},
        official_command=cmd,
        timeout_sec=None,
        isolated_data_root=args.isolated_data_root,
        settings_path=settings_path,
        extra={
            "outcome": "started",
            "job_name": job_name,
            "n_tasks": effective_n_tasks,
            "k": int(args.k),
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
    if not actual_include_filters:
        manifest["requested_count"] = effective_n_tasks
    manifest["output_paths"]["manifest"] = str(invocation_manifest_path)
    # Durable before Harbor can spend: no ambient settings model/Heavy can survive.
    write_json(manifest_path, manifest)
    write_json(invocation_manifest_path, manifest)
    try:
        with finalize_run_manifest(invocation_manifest_path, manifest) as final:
            (run_root / "harbor_command.json").write_text(json.dumps({"run_root": str(run_root), "cmd": cmd, "agent_import": AGENT_IMPORT}, indent=2), encoding="utf-8")
            status = "planned"
            reason = "command_generated"
            official_eval_status = "not_run"
            print(shlex.join(cmd))
            returncode = 0
            harbor_result: pathlib.Path | None = None
            harbor_result_error = ""
            parse_errors: list[str] = []
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
                    observed_outcomes, parse_errors = _read_harbor_job_outcomes(harbor_result)
                    observed_ids = {str(item.get("instance_id") or "") for item in observed_outcomes}
                    expected_ids = set(actual_include_filters)
                    expected_k = int(args.k)
                    if parse_errors:
                        harbor_result_error = "; ".join(parse_errors)
                    elif not observed_outcomes and effective_n_tasks > 0:
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
                    if not harbor_result_error and observed_outcomes:
                        bad_counts = [
                            f"{item['instance_id']} ({item['trial_count']}/{expected_k})"
                            for item in observed_outcomes
                            if int(item["trial_count"]) != expected_k
                        ]
                        unscored = [
                            f"{item['instance_id']} ({item['scored_trials']}/{item['trial_count']} scored)"
                            for item in observed_outcomes
                            if int(item["scored_trials"]) != int(item["trial_count"])
                        ]
                        if bad_counts:
                            harbor_result_error = (
                                f"Harbor result attempt count mismatch (expected k={expected_k}): {bad_counts}"
                            )
                        elif unscored:
                            harbor_result_error = f"Harbor result has unscored trial(s): {unscored}"
                    if harbor_result_error:
                        status = "harness_failed"
                        reason = "harbor_result_unresolved"
                        official_eval_status = "failed"
                        returncode = returncode or 2
                    # The finalization seam rewrites the manifest on EVERY exit path, so these
                    # observed-result fields need no separate write here.
                    manifest["output_paths"]["harbor_result"] = str(harbor_result)
                    manifest["observed_task_ids"] = (
                        [str(item["instance_id"]) for item in observed_outcomes] if not parse_errors else []
                    )
                    manifest["official_result_summary"] = {
                        "result_path": str(harbor_result),
                        "completed_outcomes": len(observed_outcomes) if not parse_errors else 0,
                        "completed_trials": (
                            sum(int(item["trial_count"]) for item in observed_outcomes) if not parse_errors else 0
                        ),
                        "scored_trials": (
                            sum(int(item["scored_trials"]) for item in observed_outcomes) if not parse_errors else 0
                        ),
                    }
            ledger_tasks: list[dict[str, object]] = []
            if observed_outcomes and not harbor_result_error:
                ledger_tasks.extend(observed_outcomes)
            else:
                outcome_by_task = (
                    {str(item["instance_id"]): item for item in observed_outcomes} if not parse_errors else {}
                )
                for task in task_names:
                    observed = outcome_by_task.get(task)
                    ledger_tasks.append({
                        "instance_id": task,
                        "reward": None,
                        "trial_count": int(observed["trial_count"]) if observed else 0,
                        "scored_trials": int(observed["scored_trials"]) if observed else 0,
                        "trials": list(observed["trials"]) if observed else [],
                    })
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
                                "manifest": str(invocation_manifest_path),
                                "harbor_result": str(harbor_result) if harbor_result is not None else "",
                            },
                            "error": harbor_result_error,
                            "details": {
                                "returncode": returncode,
                                "include_filters": actual_include_filters,
                                "n_tasks": effective_n_tasks,
                                "k": int(args.k),
                                "official_reward": task.get("reward"),
                                "trial_count": int(task.get("trial_count") or 0),
                                "scored_trials": int(task.get("scored_trials") or 0),
                                "trials": list(task.get("trials") or []),
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
    finally:
        # Mirror the finalized per-invocation manifest to the shared latest compatibility path.
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
