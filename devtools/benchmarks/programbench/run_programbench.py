#!/usr/bin/env python3
"""ProgramBench adapter entrypoint.

This script intentionally stops before reinventing ProgramBench orchestration.
It prepares task bodies/submissions for official cleanroom runs and delegates
evaluation to the official `programbench` CLI.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from devtools.benchmarks.common.manifests import benchmark_run_manifest, write_json
from devtools.benchmarks.common.official_commands import programbench_command_for_manifest
from devtools.benchmarks.common.result_index import task_result_row, write_result_index
from devtools.benchmarks.common.run_roots import (
    default_settings_path,
    ensure_file_output_outside_repo,
    ensure_outside_repo,
    run_root,
    safe_join_under,
)
from devtools.benchmarks.programbench.programbench_adapter import (
    build_ouroboros_task_body,
    create_submission_tarball,
    prepare_seeded_workspace,
    default_protected_backend_paths,
    preflight_cleanroom_container,
    run_official_eval,
)


def _write_failure_sidecars(
    context: dict[str, object],
    *,
    status: str,
    reason_code: str,
    official_eval_status: str,
    output_paths: dict[str, str],
    error: str,
) -> None:
    ledger_output = pathlib.Path(str(context["ledger_output"]))
    manifest_output = pathlib.Path(str(context["manifest_output"]))
    out_root = pathlib.Path(str(context["out_root"]))
    repo_dir = pathlib.Path(str(context["repo_dir"]))
    instance_dir = pathlib.Path(str(context["instance_dir"]))
    instance_id = str(context["instance_id"])
    argv = list(context["argv"]) if isinstance(context.get("argv"), list) else []
    container_name = str(context["container_name"])
    protected_paths = list(context.get("protected_paths") or [])
    preflight = dict(context.get("preflight") or {})
    eval_requested = bool(context.get("eval_requested"))
    settings_path = pathlib.Path(str(context["settings_path"]))
    isolated_data_root = str(context.get("isolated_data_root") or "")
    paths = {
        "instance_dir": str(instance_dir),
        "ledger": str(ledger_output),
        "manifest": str(manifest_output),
        **output_paths,
    }
    write_result_index(
        ledger_output,
        [
            task_result_row(
                benchmark="programbench",
                instance_id=instance_id,
                status=status,
                reason_code=reason_code,
                official_eval_status=official_eval_status,
                output_paths=paths,
                error=error,
                details={"container_name": container_name, "cleanroom_preflight": preflight, "protected_paths": protected_paths},
            )
        ],
    )
    write_json(
        manifest_output,
        benchmark_run_manifest(
            benchmark="programbench",
            run_root=out_root,
            repo_dir=repo_dir,
            requested_task_ids=[instance_id],
            argv=argv,
            output_paths=paths,
            dataset="programbench",
            harness={"container_name": container_name, "cleanroom_preflight": preflight},
            official_command=programbench_command_for_manifest(out_root, eval_requested=eval_requested),
            isolated_data_root=isolated_data_root,
            settings_path=settings_path,
            extra={
                "eval_requested": eval_requested,
                "protected_paths": protected_paths,
                "failure_reason_code": reason_code,
                "failure_error": error,
            },
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default=str(pathlib.Path(__file__).resolve().parents[3]))
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--instruction-file", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--protected-path", action="append", default=[], help="protected reference path inside cleanroom; repeatable")
    parser.add_argument("--ledger-output", default="", help="denominator-preserving JSONL result ledger")
    parser.add_argument("--manifest-output", default="", help="run manifest JSON")
    parser.add_argument("--settings-path", default="")
    parser.add_argument("--isolated-data-root", default="", help="isolated Ouroboros data root used for this run")
    parser.add_argument("--eval", action="store_true", help="run official programbench eval/info after writing submission")
    args = parser.parse_args()

    repo_dir = pathlib.Path(args.repo_dir).expanduser().resolve(strict=False)
    settings_path = pathlib.Path(args.settings_path).expanduser() if args.settings_path else default_settings_path()
    out_root = ensure_outside_repo(run_root("programbench", args.run_id), repo_dir)
    instance_dir = safe_join_under(out_root, args.instance_id)
    ledger_output = (
        ensure_file_output_outside_repo(pathlib.Path(args.ledger_output), repo_dir)
        if args.ledger_output
        else instance_dir / "result_index.jsonl"
    )
    manifest_output = (
        ensure_file_output_outside_repo(pathlib.Path(args.manifest_output), repo_dir)
        if args.manifest_output
        else instance_dir / "run_manifest.json"
    )
    protected_paths = args.protected_path or default_protected_backend_paths()
    preflight: dict[str, object] = {}
    sidecar_context: dict[str, object] = {
        "ledger_output": str(ledger_output),
        "manifest_output": str(manifest_output),
        "out_root": str(out_root),
        "repo_dir": str(repo_dir),
        "instance_dir": str(instance_dir),
        "instance_id": args.instance_id,
        "argv": sys.argv,
        "container_name": args.container_name,
        "protected_paths": protected_paths,
        "preflight": preflight,
        "eval_requested": bool(args.eval),
        "settings_path": str(settings_path),
        "isolated_data_root": args.isolated_data_root,
    }
    try:
        preflight = preflight_cleanroom_container(args.container_name)
        sidecar_context["preflight"] = preflight
    except Exception as exc:
        _write_failure_sidecars(
            sidecar_context,
            status="blocked",
            reason_code="cleanroom_preflight_failed",
            official_eval_status="not_run",
            output_paths={},
            error=str(exc),
        )
        raise
    try:
        # Normalize the workspace exactly like the e2e seed path: the reference
        # binary must live at reference_executable (protected) BEFORE the task
        # body advertises it and before anything is packaged — a raw cleanroom
        # workspace otherwise leaves the real reference at ./executable,
        # unprotected and inside the submission.
        sidecar_context["reference_layout"] = prepare_seeded_workspace(pathlib.Path(args.workspace))
    except Exception as exc:
        _write_failure_sidecars(
            sidecar_context,
            status="blocked",
            reason_code="workspace_prepare_failed",
            official_eval_status="not_run",
            output_paths={},
            error=str(exc),
        )
        raise
    body = build_ouroboros_task_body(
        instruction=pathlib.Path(args.instruction_file).read_text(encoding="utf-8"),
        workspace_host_path=pathlib.Path(args.workspace),
        container_name=args.container_name,
        protected_backend_paths=protected_paths,
        task_id=args.instance_id,
    )
    body.setdefault("metadata", {})["cleanroom_preflight"] = preflight
    write_json(instance_dir / "ouroboros_task_body.json", body)
    task_body_path = instance_dir / "ouroboros_task_body.json"
    try:
        submission = create_submission_tarball(
            pathlib.Path(args.workspace),
            instance_dir / "submission.tar.gz",
            protected_paths=protected_paths,
        )
    except Exception as exc:
        _write_failure_sidecars(
            sidecar_context,
            status="failed",
            reason_code="submission_failed",
            official_eval_status="not_run",
            output_paths={"task_body": str(task_body_path)},
            error=str(exc),
        )
        raise
    eval_result = None
    if args.eval:
        try:
            eval_result = run_official_eval(out_root)
        except Exception as exc:
            _write_failure_sidecars(
                sidecar_context,
                status="failed",
                reason_code="official_eval_failed",
                official_eval_status="failed",
                output_paths={"task_body": str(task_body_path), "submission": str(submission)},
                error=str(exc),
            )
            raise
    official_eval_status = "not_run"
    if eval_result is not None:
        official_eval_status = "completed" if eval_result.get("eval", {}).get("returncode") == 0 else "failed"
    output_paths = {
        "task_body": str(instance_dir / "ouroboros_task_body.json"),
        "submission": str(submission),
    }
    if eval_result is not None:
        output_paths["official_eval"] = str(out_root / "programbench_eval_result.json")
    write_result_index(
        ledger_output,
        [
            task_result_row(
                benchmark="programbench",
                instance_id=args.instance_id,
                status="completed",
                reason_code="submission_prepared",
                prediction_written=True,
                official_eval_status=official_eval_status,
                output_paths=output_paths,
                details={"cleanroom_preflight": preflight, "protected_paths": protected_paths},
            )
        ],
    )
    write_json(
        manifest_output,
        benchmark_run_manifest(
            benchmark="programbench",
            run_root=out_root,
            repo_dir=repo_dir,
            requested_task_ids=[args.instance_id],
            argv=sys.argv,
            output_paths={
                "instance_dir": str(instance_dir),
                "ledger": str(ledger_output),
                "manifest": str(manifest_output),
                "submission": str(submission),
            },
            dataset="programbench",
            harness={"container_name": args.container_name, "cleanroom_preflight": preflight},
            official_command=programbench_command_for_manifest(out_root, eval_requested=bool(args.eval)),
            isolated_data_root=args.isolated_data_root,
            settings_path=settings_path,
            extra={"eval_requested": bool(args.eval), "protected_paths": protected_paths},
        ),
    )
    print(instance_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
