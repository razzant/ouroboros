#!/usr/bin/env python3
"""End-to-end ProgramBench runner for Ouroboros.

Orchestrates docker cleanroom setup, Ouroboros task submission via the gateway API,
submission packaging, and optional official ``programbench eval``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import traceback
import urllib.parse
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from devtools.benchmarks.common.manifests import MODEL_SLOT_KEYS, benchmark_run_manifest, write_json
from devtools.benchmarks.common.official_commands import programbench_command_for_manifest
from devtools.benchmarks.common.result_index import task_result_row, write_result_index
from devtools.benchmarks.common.run_roots import (
    default_settings_path,
    ensure_outside_repo,
    run_root,
    safe_join_under,
)
from devtools.benchmarks.programbench.programbench_adapter import (
    _checkpoint_task_id,
    build_instruction,
    build_ouroboros_task_body,
    classify_infra_failure,
    container_name_for_instance,
    create_submission_tarball,
    default_protected_backend_paths,
    ouroboros_api_request,
    pull_cleanroom_image,
    run_official_eval,
    seed_workspace_from_image,
    start_cleanroom_container,
    stop_cleanroom_container,
    submit_and_wait,
    terminal_task_status,
)
from devtools.benchmarks.programbench.schemas import PROGRAMBENCH_TIMEOUT_SEC
from ouroboros.provider_models import migrate_model_value

# Model-carrying slots only; the OUROBOROS_EFFORT_* entries in MODEL_SLOT_KEYS are
# effort levels, not model ids.
_MODEL_ID_SLOT_KEYS = tuple(key for key in MODEL_SLOT_KEYS if not key.startswith("OUROBOROS_EFFORT_"))

TASK_CHECKPOINT_BASENAME = "ouroboros_task_checkpoint.json"

# A resume-skipped instance (prior submission already on disk) is successful
# prior work, not a failure — the run exit code and failed_count agree on this.
_SUCCESS_STATUSES = ("completed", "skipped")


def _row_successful(row: dict[str, Any]) -> bool:
    return row.get("status") in _SUCCESS_STATUSES


def _ensure_docker_host() -> None:
    if os.environ.get("DOCKER_HOST"):
        return
    colima_sock = pathlib.Path.home() / ".colima" / "default" / "docker.sock"
    if colima_sock.exists():
        os.environ["DOCKER_HOST"] = f"unix://{colima_sock}"


def _setting_or_env(settings: dict[str, Any], key: str) -> str:
    """Settings value with env fallback — the server treats settings.json as SSOT
    and falls back to the launch environment for keys it does not carry."""
    raw = settings.get(key)
    if raw is None or str(raw).strip() == "":
        raw = os.environ.get(key, "")
    return str(raw or "").strip()


def _active_direct_provider(settings: dict[str, Any]) -> str:
    """Mirror config._exclusive_direct_remote_provider_env over settings+env.

    '' means the OpenRouter-style route (or an ambiguous key set), where
    ``provider/model`` ids are canonical and must NOT be rewritten.
    """
    if any(
        _setting_or_env(settings, key)
        for key in ("OPENROUTER_API_KEY", "OPENAI_BASE_URL", "OPENAI_COMPATIBLE_BASE_URL")
    ):
        return ""
    direct = [
        provider
        for provider, key in (
            ("openai", "OPENAI_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("cloudru", "CLOUDRU_FOUNDATION_MODELS_API_KEY"),
        )
        if _setting_or_env(settings, key)
    ]
    if _setting_or_env(settings, "GIGACHAT_CREDENTIALS") or (
        _setting_or_env(settings, "GIGACHAT_USER") and _setting_or_env(settings, "GIGACHAT_PASSWORD")
    ):
        direct.append("gigachat")
    return direct[0] if len(direct) == 1 else ""


def preflight_model_slots(settings_path: pathlib.Path, *, solve_model: str = "") -> dict[str, str]:
    """Validate configured model ids the way the runtime routes them.

    Direct-provider routes reject legacy ``provider/model`` ids (they need
    ``provider::model``); a prior debug run burned 5/5 tasks on exactly that
    400. The runner cannot patch the externally started server's environment, so
    a legacy-form id (or a --solve-model that disagrees with settings) is a
    launch blocker, not a warning. Returns the normalized slot snapshot.
    """
    settings: dict[str, Any] = {}
    try:
        loaded = json.loads(pathlib.Path(settings_path).read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            settings = loaded
    except (OSError, json.JSONDecodeError):
        settings = {}
    provider = _active_direct_provider(settings)
    problems: list[str] = []
    slots: dict[str, str] = {}
    for key in _MODEL_ID_SLOT_KEYS:
        raw = _setting_or_env(settings, key)
        if not raw:
            continue
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        migrated = [migrate_model_value(provider, part) if provider else part for part in parts]
        if migrated != parts:
            problems.append(f"{key}: {raw!r} -> {','.join(migrated)!r}")
        slots[key] = ",".join(migrated)
    expected_solve = ""
    if solve_model:
        expected_solve = migrate_model_value(provider, solve_model) if provider else solve_model
        if expected_solve != solve_model:
            problems.append(f"--solve-model: {solve_model!r} -> {expected_solve!r}")
        configured = slots.get("OUROBOROS_MODEL", "")
        if configured and configured != expected_solve:
            problems.append(
                f"--solve-model {expected_solve!r} does not match settings OUROBOROS_MODEL "
                f"{configured!r}; the server would solve on the wrong model"
            )
    if problems:
        raise SystemExit(
            "model slot preflight failed for the "
            + (f"direct {provider!r}" if provider else "configured")
            + " route — fix the settings the server was started with (see "
            f"{settings_path}):\n  " + "\n  ".join(problems)
        )
    return slots


def _load_instances(
    *,
    difficulty: str,
    filter_spec: str,
    slice_spec: str,
    instance_id: str,
    shuffle: bool,
) -> list[dict[str, Any]]:
    from programbench.utils.instance_filters import filter_instances
    from programbench.utils.load_data import load_all_instances

    instances = load_all_instances(include_tests=False)
    if difficulty:
        instances = [item for item in instances if str(item.get("difficulty") or "") == difficulty]
    if instance_id:
        instances = [item for item in instances if str(item.get("instance_id") or "") == instance_id]
    return filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)


@dataclasses.dataclass(frozen=True)
class InstanceRunConfig:
    """Per-run knobs shared by every instance (keeps `_process_instance` within
    the DEVELOPMENT.md parameter budget)."""

    out_root: pathlib.Path
    ouroboros_url: str
    timeout_sec: float
    cpus: str
    memory: str
    protected_paths: list[str]
    dry_run: bool
    skip_pull: bool
    redo_existing: bool


def _process_instance(instance: dict[str, Any], cfg: InstanceRunConfig) -> dict[str, Any]:
    instance_id = str(instance["instance_id"])
    image_name = str(instance["image_name"])
    instance_dir = safe_join_under(cfg.out_root, instance_id)
    workspace = instance_dir / "workspace"
    container_name = container_name_for_instance(instance_id)
    instruction = build_instruction(instance)
    instruction_path = instance_dir / "instruction.txt"
    instance_dir.mkdir(parents=True, exist_ok=True)
    instruction_path.write_text(instruction, encoding="utf-8")
    checkpoint_path = instance_dir / TASK_CHECKPOINT_BASENAME
    if cfg.redo_existing:
        # An explicit redo means a fresh solve; a surviving checkpoint would
        # re-attach to the previous task instead. Cancel a still-live server-side
        # task FIRST — otherwise it keeps running and its docker-exec calls land
        # in the NEW same-named container, contaminating the fresh solve
        # (adversarial review r1).
        stale_task_id = _checkpoint_task_id(checkpoint_path)
        if stale_task_id:
            try:
                ouroboros_api_request(
                    cfg.ouroboros_url, "POST",
                    f"/api/tasks/{urllib.parse.quote(stale_task_id)}/cancel", timeout=30,
                )
            except Exception:
                pass  # best-effort: server may not know it / be unreachable
        checkpoint_path.unlink(missing_ok=True)

    harness: dict[str, Any] = {"container_name": container_name, "image_name": image_name}
    output_paths: dict[str, str] = {"instruction": str(instruction_path)}
    keep_container_for_reattach = False
    # Reattach honor-check BEFORE any cleanroom work: a fresh seed/start would
    # reset the workspace and (start_cleanroom_container stops any namesake
    # first) kill the very container a prior client_poll_timeout_reattachable
    # row deliberately left running. If the checkpoint names a task this server
    # still knows, we preserve the container/workspace and let submit_and_wait
    # reattach (a settled task returns immediately and is packaged as-is); only
    # a missing/stale checkpoint (or --redo-existing, which unlinked it above)
    # gets a fresh cleanroom.
    reattach_task_id = _checkpoint_task_id(checkpoint_path)
    if reattach_task_id:
        try:
            _reattach_payload = ouroboros_api_request(
                cfg.ouroboros_url, "GET",
                f"/api/tasks/{urllib.parse.quote(reattach_task_id)}", timeout=60,
            )
        except Exception:
            # ANY failure (RuntimeError from an HTTPError, or a connection-level
            # URLError/OSError the adapter does not map) means we cannot confirm
            # the task is live — fall to a fresh cleanroom rather than aborting
            # the whole run before the top-level ledger is written (r1 #14).
            reattach_task_id = ""
        else:
            # Reattach ONLY to an in-flight or COMPLETED task. A task that already
            # SETTLED as a failure/cancellation must NOT be reattached — replaying
            # it is zero work and defeats the resume-RETRY contract (r1 #12: a
            # failed instance gets NO tarball precisely so a resume run re-solves
            # it). Drop the stale checkpoint and fall to a fresh cleanroom solve
            # (adversarial review r2 #5). An in-flight task (status not settled)
            # returns "" from terminal_task_status and is reattached as a genuine
            # crash-resume; a settled "completed" is reattached and packaged as-is.
            _settled = terminal_task_status(_reattach_payload)
            if _settled and _settled != "completed":
                reattach_task_id = ""
                checkpoint_path.unlink(missing_ok=True)
    harness["reattached_task_id"] = reattach_task_id
    # While a reattachable task is live, NOTHING may stop its container — not a
    # dry-run early return, not a mid-poll transport error. Only a settled
    # terminal result (below) re-arms the normal teardown.
    keep_container_for_reattach = bool(reattach_task_id)
    try:
        if not reattach_task_id:
            if not cfg.skip_pull:
                harness["image"] = pull_cleanroom_image(image_name)
            harness["seed"] = seed_workspace_from_image(image_name, workspace)
            harness["container"] = start_cleanroom_container(
                container_name,
                image_name,
                workspace,
                cpus=cfg.cpus,
                memory=cfg.memory,
            )
        body = build_ouroboros_task_body(
            instruction=instruction,
            workspace_host_path=workspace,
            container_name=container_name,
            protected_backend_paths=cfg.protected_paths,
        )
        body.setdefault("metadata", {})["programbench_instance_id"] = instance_id
        body.setdefault("metadata", {})["cleanroom_preflight"] = (harness.get("container") or {}).get("preflight") or {}
        body["timeout_sec"] = float(cfg.timeout_sec)
        write_json(instance_dir / "ouroboros_task_body.json", body)
        output_paths["task_body"] = str(instance_dir / "ouroboros_task_body.json")
        output_paths["workspace"] = str(workspace)

        if cfg.dry_run:
            return task_result_row(
                benchmark="programbench",
                instance_id=instance_id,
                status="completed",
                reason_code="dry_run",
                official_eval_status="not_run",
                output_paths=output_paths,
                details=harness,
            )

        output_paths["task_checkpoint"] = str(checkpoint_path)
        try:
            task_result = submit_and_wait(
                cfg.ouroboros_url,
                body,
                timeout_sec=cfg.timeout_sec,
                checkpoint_path=checkpoint_path,
            )
        except Exception as exc:
            # A client-side poll failure (timeout OR a transient mid-poll transport
            # error) after a task was submitted leaves it RUNNING server-side with
            # its task_id in the checkpoint. Tearing the container down here would
            # make the next run reattach to a task whose Docker backend is gone —
            # keep the executor alive for reattach (r1 #10, generalized from the
            # timeout-only guard). If NO task_id was ever recorded (creation itself
            # failed), there is nothing to reattach: re-raise to the normal
            # teardown+failed-row path.
            if not _checkpoint_task_id(checkpoint_path):
                raise
            keep_container_for_reattach = True
            reason = "client_poll_timeout_reattachable" if isinstance(exc, TimeoutError) else "client_poll_error_reattachable"
            return task_result_row(
                benchmark="programbench",
                instance_id=instance_id,
                status="failed",
                reason_code=reason,
                official_eval_status="not_run",
                output_paths=output_paths,
                error=str(exc),
                details={"harness": harness, "container_left_running": True,
                         "reattach_checkpoint": str(checkpoint_path)},
            )
        keep_container_for_reattach = False  # settled: normal teardown applies
        write_json(instance_dir / "ouroboros_task_result.json", task_result)
        output_paths["task_result"] = str(instance_dir / "ouroboros_task_result.json")

        # Classification comes from the payload's EXPLICIT terminal status; a
        # stale provider reason_code on a completed task must never demote it.
        status = terminal_task_status(task_result)
        infra_failed = classify_infra_failure(task_result)
        if status != "completed":
            # A non-completed settled task produces NO submission tarball — else the
            # resume gate would see submission.tar.gz and skip a FAILED instance as a
            # skipped-success, never retrying it (r1 #12). Denominator stays honest
            # via the failed row below.
            return task_result_row(
                benchmark="programbench",
                instance_id=instance_id,
                status="failed",
                reason_code=str(task_result.get("reason_code") or "") or "task_not_completed",
                prediction_written=False,
                official_eval_status="not_run",
                output_paths=output_paths,
                error=str(task_result.get("result") or task_result.get("error") or status),
                details={"harness": harness, "task_status": status, "infra_failed": infra_failed},
            )
        # Only a COMPLETED task produces the source submission — its presence is
        # what the resume gate keys on, so it must mean "solved", not "attempted".
        submission = create_submission_tarball(
            workspace,
            instance_dir / "submission.tar.gz",
            protected_paths=cfg.protected_paths,
        )
        output_paths["submission"] = str(submission)
        return task_result_row(
            benchmark="programbench",
            instance_id=instance_id,
            status="completed",
            reason_code="submission_prepared",
            prediction_written=True,
            official_eval_status="not_run",
            output_paths=output_paths,
            details={"harness": harness, "task_status": status, "infra_failed": infra_failed},
        )
    except Exception as exc:
        return task_result_row(
            benchmark="programbench",
            instance_id=instance_id,
            status="failed",
            reason_code=type(exc).__name__,
            official_eval_status="not_run",
            output_paths=output_paths,
            error=str(exc),
            details={"harness": harness, "traceback": traceback.format_exc()},
        )
    finally:
        if not keep_container_for_reattach:
            stop_cleanroom_container(container_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ProgramBench instances through Ouroboros")
    parser.add_argument("--repo-dir", default=str(pathlib.Path(__file__).resolve().parents[3]))
    parser.add_argument("--ouroboros-url", default="http://127.0.0.1:8770", help="Ouroboros source server URL (desktop sandbox on 8765 cannot see bench_runs)")
    parser.add_argument("--solve-model", default="", help="expected OUROBOROS_MODEL; validated/normalized against --settings-path before any task burns")
    parser.add_argument("--difficulty", default="", help="filter by task.yaml difficulty (default: all)")
    parser.add_argument("--filter", default="", dest="filter_spec", help="regex filter on instance_id")
    parser.add_argument("--slice", default="", dest="slice_spec", help="slice spec, e.g. 0:3 (default: all)")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle instances (programbench uses seed 42 after sorting by instance_id)",
    )
    parser.add_argument("--instance-id", default="", help="run a single instance id")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--timeout-sec", type=float, default=PROGRAMBENCH_TIMEOUT_SEC)
    parser.add_argument("--cpus", default="4")
    parser.add_argument("--memory", default="16g")
    parser.add_argument("--protected-path", action="append", default=[])
    parser.add_argument("--settings-path", default="")
    parser.add_argument("--redo-existing", action="store_true")
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="docker setup + task body only")
    parser.add_argument("--eval", action="store_true", help="run official programbench eval/info after all instances")
    parser.add_argument("--no-eval", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.no_eval:
        args.eval = False

    _ensure_docker_host()
    repo_dir = pathlib.Path(args.repo_dir).expanduser().resolve(strict=False)
    settings_path = pathlib.Path(args.settings_path).expanduser() if args.settings_path else default_settings_path()
    model_slots = preflight_model_slots(settings_path, solve_model=str(args.solve_model or ""))
    out_root = ensure_outside_repo(run_root("programbench", args.run_id), repo_dir)
    protected_paths = args.protected_path or default_protected_backend_paths()

    instances = _load_instances(
        difficulty=str(args.difficulty or ""),
        filter_spec=str(args.filter_spec or ""),
        slice_spec=str(args.slice_spec or ""),
        instance_id=str(args.instance_id or ""),
        shuffle=bool(args.shuffle),
    )
    if not instances:
        raise SystemExit("no ProgramBench instances matched the selection")

    write_json(
        out_root / "instance_order.json",
        {
            "count": len(instances),
            "shuffle": bool(args.shuffle),
            "shuffle_seed": 42 if args.shuffle else None,
            "instance_ids": [str(item["instance_id"]) for item in instances],
        },
    )

    selected_ids = [str(item["instance_id"]) for item in instances]
    skipped_rows: list[dict[str, Any]] = []
    if not args.redo_existing:
        # Resume narrows the WORK, never the ledger denominator: instances with a
        # prior submission are recorded as explicit skipped rows, not dropped.
        pending = []
        for instance in instances:
            submission = safe_join_under(out_root, str(instance["instance_id"])) / "submission.tar.gz"
            if submission.is_file() and submission.stat().st_size > 0:
                skipped_rows.append(task_result_row(
                    benchmark="programbench",
                    instance_id=str(instance["instance_id"]),
                    status="skipped",
                    reason_code="skipped_existing_submission",
                    prediction_written=True,
                    official_eval_status="not_run",
                    output_paths={"submission": str(submission)},
                ))
                continue
            pending.append(instance)
        instances = pending

    rows: list[dict[str, Any]] = list(skipped_rows)
    cfg = InstanceRunConfig(
        out_root=out_root,
        ouroboros_url=str(args.ouroboros_url),
        timeout_sec=float(args.timeout_sec),
        cpus=str(args.cpus),
        memory=str(args.memory),
        protected_paths=protected_paths,
        dry_run=bool(args.dry_run),
        skip_pull=bool(args.skip_pull),
        redo_existing=bool(args.redo_existing),
    )
    for instance in instances:
        row = _process_instance(instance, cfg)
        rows.append(row)
        write_result_index(safe_join_under(out_root, str(instance["instance_id"])) / "result_index.jsonl", [row])

    ledger_path = out_root / "result_index.jsonl"
    write_result_index(ledger_path, rows)

    eval_result = None
    official_eval_status = "not_run"
    if args.eval and not args.dry_run:
        try:
            eval_result = run_official_eval(out_root)
            # returncode != 0 = the eval RAN and reported test failures (a valid
            # partial-pass benchmark result, not a run error).
            official_eval_status = "completed" if eval_result.get("eval", {}).get("returncode") == 0 else "failed"
        except Exception as exc:
            # The eval could not RUN at all (harness/infra error) — distinct from a
            # ran-but-some-tests-failed result; this is a run error (r2 fable #3).
            official_eval_status = "error"
            write_json(
                out_root / "programbench_eval_result.json",
                {"error": str(exc), "traceback": traceback.format_exc()},
            )

    write_json(
        out_root / "run_manifest.json",
        benchmark_run_manifest(
            benchmark="programbench",
            run_root=out_root,
            repo_dir=repo_dir,
            requested_task_ids=selected_ids,
            argv=sys.argv,
            output_paths={
                "run_root": str(out_root),
                "ledger": str(ledger_path),
                "manifest": str(out_root / "run_manifest.json"),
            },
            dataset="programbench",
            harness={
                "ouroboros_url": str(args.ouroboros_url),
                "difficulty": str(args.difficulty or ""),
                "dry_run": bool(args.dry_run),
                "solve_model": str(args.solve_model or ""),
                "model_slots_normalized": model_slots,
            },
            official_command=programbench_command_for_manifest(out_root, eval_requested=bool(args.eval)),
            isolated_data_root="",
            settings_path=settings_path,
            extra={
                "eval_requested": bool(args.eval),
                "official_eval_status": official_eval_status,
                "protected_paths": protected_paths,
                "shuffle": bool(args.shuffle),
                "shuffle_seed": 42 if args.shuffle else None,
                "instance_order": str(out_root / "instance_order.json"),
                "completed_count": sum(1 for row in rows if row.get("status") == "completed"),
                "failed_count": sum(1 for row in rows if not _row_successful(row)),
            },
        ),
    )
    print(out_root)
    rows_ok = all(_row_successful(row) for row in rows)
    # A REQUESTED official eval that could NOT run (harness error) is a run failure
    # even when every solve completed — otherwise `run_...e2e --eval && ...` reads a
    # broken eval as success. A "failed" eval that merely reported test failures is
    # a valid partial-pass result and does NOT fail the run (r2 fable #3).
    eval_errored = bool(args.eval) and not args.dry_run and official_eval_status == "error"
    return 0 if (rows_ok and not eval_errored) else 1


if __name__ == "__main__":
    raise SystemExit(main())
