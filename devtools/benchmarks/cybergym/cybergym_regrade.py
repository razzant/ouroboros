"""Prepare immutable, outcome-neutral inputs for CyberGym final-PoC regrading.

This module deliberately does not run an agent, start a gateway, or alter a
historical run.  It chooses the latest complete outcome per task from explicit
append-only result indexes and writes a new inventory whose entries bind the
old final marker by hash.  A later verifier arm consumes that inventory with a
fresh sidecar and records its own results separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from devtools.benchmarks.cybergym.cybergym_adapter import TaskSpec, final_poc_record
from devtools.benchmarks.cybergym.cybergym_protocol import OFFICIAL_MODEL, safe_task_id

REGRADABLE_STATUSES = frozenset({"known_success", "known_failure"})
SCHEMA = "ouroboros.benchmark.cybergym.regrade_inventory.v1"
RESULT_SCHEMA = "ouroboros.benchmark.cybergym.regrade_result.v1"
SUMMARY_SCHEMA = "ouroboros.benchmark.cybergym.regrade_summary.v1"


def _read_rows(root: pathlib.Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    path = root / "result_index.jsonl"
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} is not a result mapping")
        yield line_number, value


def _eligible(row: Mapping[str, Any], *, model: str) -> bool:
    return (
        str(row.get("status") or "") in {"completed", "failed"}
        and str(row.get("final_submission_status") or "") in REGRADABLE_STATUSES
        and str(row.get("observed_model") or "") == model
        and str(row.get("observed_effort") or "") == "high"
        and row.get("cost_final") is True
    )


def select_latest_regrade_candidates(
    run_roots: Sequence[pathlib.Path | str], *, model: str = OFFICIAL_MODEL
) -> list[dict[str, Any]]:
    """Choose the latest eligible historical outcome for each task neutrally.

    A complete task result can be a pass or a failure.  The selection key is
    timestamp followed by the explicit source-root order and JSONL line, never
    the official result.  Missing final markers remain represented as valid
    historical zeroes but are explicitly not regradable.
    """
    selected: dict[str, tuple[tuple[float, int, int], dict[str, Any]]] = {}
    for root_index, raw_root in enumerate(run_roots):
        root = pathlib.Path(raw_root).expanduser().resolve(strict=True)
        for line_number, row in _read_rows(root):
            if not _eligible(row, model=model):
                continue
            task_id = safe_task_id(str(row.get("task_id") or row.get("instance_id") or ""))
            ts = float(row.get("ts_unix") or 0.0)
            key = (ts, root_index, line_number)
            candidate = {
                "task_id": task_id,
                "source_run_root": str(root),
                "source_line": line_number,
                "source_timestamp": ts,
                "source_result_sha256": hashlib.sha256(
                    json.dumps(dict(row), sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "historical_final_submission_success": row.get("final_submission_success"),
                "historical_final_submission_status": row.get("final_submission_status"),
                "source_final_poc_hash": str(row.get("final_poc_hash") or ""),
                "regrade_status": "not_regradable",
            }
            source_marker = pathlib.Path(
                str((row.get("artifact_refs") or {}).get("task_dir") or "")
            ) / "final.poc"
            if source_marker.is_file():
                marker = final_poc_record(source_marker)
                if marker.sha256 != candidate["source_final_poc_hash"]:
                    candidate["regrade_status"] = "hash_mismatch"
                else:
                    candidate.update(
                        regrade_status="ready",
                        source_final_poc_path=str(source_marker),
                        source_final_poc_size=marker.size,
                    )
            elif candidate["historical_final_submission_status"] == "known_failure":
                candidate["regrade_status"] = "no_final_poc"
            current = selected.get(task_id)
            if current is None or key > current[0]:
                selected[task_id] = (key, candidate)
    return [selected[task_id][1] for task_id in sorted(selected)]


def write_regrade_inventory(
    output_path: pathlib.Path | str,
    run_roots: Sequence[pathlib.Path | str],
    *,
    model: str = OFFICIAL_MODEL,
) -> dict[str, Any]:
    """Write a new inventory, refusing to overwrite an existing artifact."""
    output = pathlib.Path(output_path).expanduser().resolve(strict=False)
    if output.exists():
        raise ValueError(f"regrade inventory already exists: {output}")
    candidates = select_latest_regrade_candidates(run_roots, model=model)
    payload = {
        "schema": SCHEMA,
        "model": model,
        "source_run_roots": [str(pathlib.Path(root).expanduser().resolve(strict=True)) for root in run_roots],
        "candidates": candidates,
        "counts": {
            "selected": len(candidates),
            "ready": sum(item["regrade_status"] == "ready" for item in candidates),
            "no_final_poc": sum(item["regrade_status"] == "no_final_poc" for item in candidates),
            "hash_mismatch": sum(item["regrade_status"] == "hash_mismatch" for item in candidates),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def _regrade_attempt_id(task_id: str, source_result_sha256: str) -> str:
    """Deterministic per-task regrade attempt id; makes reruns resumable."""

    digest = hashlib.sha256(
        f"cybergym-regrade\0{task_id}\0{source_result_sha256}".encode("utf-8")
    ).hexdigest()[:24]
    return f"rg{digest}"


def load_regrade_inventory(path: pathlib.Path | str) -> dict[str, Any]:
    inventory_path = pathlib.Path(path).expanduser().resolve(strict=True)
    value = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError(f"{inventory_path} is not a {SCHEMA} inventory")
    candidates = value.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("regrade inventory carries no candidates list")
    return value


def _passthrough_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "task_id": candidate["task_id"],
        "regrade_status": candidate["regrade_status"],
        "regraded": False,
        "historical_final_submission_success": candidate["historical_final_submission_success"],
        "source_run_root": candidate["source_run_root"],
        "source_line": candidate["source_line"],
        "source_result_sha256": candidate["source_result_sha256"],
        "ts_unix": time.time(),
    }


def execute_regrade_inventory(
    inventory_path: pathlib.Path | str,
    *,
    config: Any,
    run_root: pathlib.Path | str,
    workers: int = 8,
    limit: int = 0,
    executor_factory: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Re-run the official verifier for every ready inventory candidate.

    No model or gateway call is possible here: the executor arm only generates
    the official task scaffold, copies the hash-bound historical ``final.poc``
    and queries the fresh verifier.  Results land in an append-only
    ``result_index.jsonl`` below ``run_root``; already-recorded task ids are
    skipped so an interrupted regrade resumes without repeating verifier work.
    Historical zeroes without a final marker are carried as passthrough rows.
    """

    if config.provider_probe:
        raise ValueError("regrade executor config must disable provider probing")
    if executor_factory is None:
        from devtools.benchmarks.cybergym.cybergym_executor import CyberGymExecutor

        executor_factory = CyberGymExecutor
    from devtools.benchmarks.common.result_index import append_result_index, read_result_index
    from devtools.benchmarks.cybergym.cybergym_sidecar import make_opaque_agent_id

    root = pathlib.Path(run_root).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    inventory = load_regrade_inventory(inventory_path)
    candidates = [item for item in inventory["candidates"] if isinstance(item, Mapping)]
    if limit > 0:
        candidates = candidates[:limit]

    ledger_lock = threading.Lock()
    done: set[str] = {
        str(row.get("task_id") or "")
        for row in read_result_index(root)
        if row.get("schema") == RESULT_SCHEMA
    }

    def record(row: dict[str, Any]) -> None:
        with ledger_lock:
            if row["task_id"] in done:
                return
            append_result_index(root, row)
            done.add(row["task_id"])

    pending_ready: list[Mapping[str, Any]] = []
    for candidate in candidates:
        task_id = str(candidate.get("task_id") or "")
        if not task_id or task_id in done:
            continue
        if candidate.get("regrade_status") != "ready":
            record(_passthrough_row(candidate))
            continue
        pending_ready.append(candidate)

    executor = executor_factory(config)
    errors: list[str] = []

    def regrade_one(candidate: Mapping[str, Any]) -> None:
        task_id = str(candidate["task_id"])
        project = task_id.split(":", 1)[0] if ":" in task_id else task_id
        attempt_id = _regrade_attempt_id(task_id, str(candidate["source_result_sha256"]))
        task_dir = root / "tasks" / safe_task_id(task_id)
        container_name = ""
        try:
            outcome = executor.regrade_final_poc(
                TaskSpec(task_id, project),
                str(candidate["source_final_poc_path"]),
                task_dir,
                attempt_id=attempt_id,
            )
            agent_id = make_opaque_agent_id(config.campaign_id, task_id, "regrade-" + attempt_id)
            container_name = f"cybergym-workspace-{agent_id}"
            classification = dict(outcome.get("classification") or {})
            verify_record = dict(outcome.get("record") or {})
            row = {
                "schema": RESULT_SCHEMA,
                "task_id": task_id,
                "attempt_id": attempt_id,
                "regrade_status": "regraded",
                "regraded": True,
                "official_success": classification.get("official_success"),
                "vul_exit_code": verify_record.get("vul_exit_code", verify_record.get("vul_exit")),
                "fix_exit_code": verify_record.get("fix_exit_code", verify_record.get("fix_exit")),
                "poc_id": verify_record.get("poc_id"),
                "poc_hash": verify_record.get("poc_hash"),
                "source_final_poc_sha256": outcome.get("source_final_poc_sha256"),
                "historical_final_submission_success": candidate["historical_final_submission_success"],
                "source_run_root": candidate["source_run_root"],
                "source_line": candidate["source_line"],
                "source_result_sha256": candidate["source_result_sha256"],
                "ts_unix": time.time(),
            }
        except Exception as exc:  # noqa: BLE001 - every failure must land in the ledger
            errors.append(f"{task_id}: {type(exc).__name__}")
            row = {
                "schema": RESULT_SCHEMA,
                "task_id": task_id,
                "attempt_id": attempt_id,
                "regrade_status": "error",
                "regraded": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
                "historical_final_submission_success": candidate["historical_final_submission_success"],
                "source_run_root": candidate["source_run_root"],
                "source_line": candidate["source_line"],
                "source_result_sha256": candidate["source_result_sha256"],
                "ts_unix": time.time(),
            }
        finally:
            if container_name:
                try:
                    executor._cleanup_workspace_container(
                        container_name,
                        task_id,
                        attempt_id,
                        task_dir / "workspace_cleanup.json",
                    )
                except Exception as cleanup_exc:  # noqa: BLE001 - surfaced in the summary
                    errors.append(f"{task_id}: cleanup {type(cleanup_exc).__name__}")
        record(row)

    try:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {pool.submit(regrade_one, candidate): candidate for candidate in pending_ready}
            remaining = set(futures)
            while remaining:
                finished, remaining = wait(remaining, return_when=FIRST_COMPLETED)
                for future in finished:
                    future.result()
    finally:
        close_report = executor.close()

    rows = [row for row in read_result_index(root) if row.get("schema") == RESULT_SCHEMA]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "inventory": str(pathlib.Path(inventory_path).expanduser().resolve(strict=True)),
        "run_root": str(root),
        "model": inventory.get("model"),
        "counts": {
            "rows": len(rows),
            "regraded": sum(1 for row in rows if row.get("regraded") is True),
            "official_success": sum(1 for row in rows if row.get("official_success") is True),
            "official_failure": sum(1 for row in rows if row.get("official_success") is False),
            "no_final_poc": sum(1 for row in rows if row.get("regrade_status") == "no_final_poc"),
            "hash_mismatch": sum(1 for row in rows if row.get("regrade_status") == "hash_mismatch"),
            "error": sum(1 for row in rows if row.get("regrade_status") == "error"),
        },
        "historical_success_among_regraded": sum(
            1
            for row in rows
            if row.get("regraded") is True and row.get("historical_final_submission_success") is True
        ),
        "errors": errors[:100],
        "cleanup": dict(close_report) if isinstance(close_report, Mapping) else close_report,
        "ts_unix": time.time(),
    }
    summary_path = root / "regrade_summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def _build_regrade_config(args: argparse.Namespace) -> Any:
    from devtools.benchmarks.cybergym.cybergym_executor import ExecutorConfig

    python_executable = shutil.which(str(args.cybergym_python))
    if not python_executable:
        raise ValueError("the --cybergym-python executable is not available on PATH")
    return ExecutorConfig(
        campaign_id=str(args.campaign_id),
        source_root=pathlib.Path(args.source_root),
        data_root=pathlib.Path(args.data_root),
        mask_map=pathlib.Path(args.mask_map),
        run_root=pathlib.Path(args.regrade_run_root),
        server_root=pathlib.Path(args.server_root),
        server_image=str(args.server_image),
        server_image_digest=str(args.server_image_digest),
        workspace_image=str(args.workspace_image),
        workspace_image_digest=str(args.workspace_image_digest),
        ouroboros_url="http://regrade-unused.local",
        docker_host=str(args.docker_host),
        provider_probe=False,
        provider_inventory_probe=False,
        binary_dir=pathlib.Path(args.binary_dir),
        expected_data_sha256=str(args.expected_data_sha256 or ""),
        expected_binary_sha256=str(args.expected_binary_sha256 or ""),
        python_executable=python_executable,
        settings_path=None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CyberGym final-PoC regrade tooling")
    parser.add_argument("--source-run-root", action="append", default=[])
    parser.add_argument("--output", default="")
    parser.add_argument("--model", default=OFFICIAL_MODEL)
    parser.add_argument("--execute-inventory", default="", help="inventory JSON to re-verify")
    parser.add_argument("--regrade-run-root", default="", help="new append-only regrade output root")
    parser.add_argument("--campaign-id", default="")
    parser.add_argument("--source-root", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--mask-map", default="")
    parser.add_argument("--server-root", default="", help="fresh server root for the regrade verifier DB/logs")
    parser.add_argument("--binary-dir", default="", help="pinned binary tree, mounted read-only")
    parser.add_argument("--server-image", default="")
    parser.add_argument("--server-image-digest", default="")
    parser.add_argument("--workspace-image", default="")
    parser.add_argument("--workspace-image-digest", default="")
    parser.add_argument("--docker-host", default="")
    parser.add_argument("--cybergym-python", default="")
    parser.add_argument("--expected-data-sha256", default="")
    parser.add_argument("--expected-binary-sha256", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="cap candidates for a smoke regrade")
    args = parser.parse_args(argv)

    if not args.execute_inventory:
        if not args.source_run_root or not args.output:
            print("[cybergym] regrade inventory requires --source-run-root and --output")
            return 2
        try:
            payload = write_regrade_inventory(args.output, args.source_run_root, model=args.model)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[cybergym] regrade inventory refusal: {exc}")
            return 2
        print(json.dumps(payload["counts"], sort_keys=True))
        return 0

    missing = [
        name
        for name in (
            "regrade_run_root", "source_root", "data_root", "mask_map", "server_root",
            "binary_dir", "server_image", "server_image_digest", "workspace_image",
            "workspace_image_digest", "docker_host", "cybergym_python",
        )
        if not getattr(args, name)
    ]
    if missing:
        print(f"[cybergym] regrade execute missing: {', '.join('--' + name.replace('_', '-') for name in missing)}")
        return 2
    run_root = pathlib.Path(args.regrade_run_root).expanduser().resolve(strict=False)
    campaign_id = args.campaign_id or ("regrade-" + run_root.name)
    args.campaign_id = campaign_id
    try:
        config = _build_regrade_config(args)
        summary = execute_regrade_inventory(
            args.execute_inventory,
            config=config,
            run_root=run_root,
            workers=args.workers,
            limit=args.limit,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[cybergym] regrade refusal: {exc}")
        return 2
    print(json.dumps(summary["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
