#!/usr/bin/env python3
"""CyberGym launcher with a durable admission boundary.

The launcher owns only protocol bookkeeping.  Upstream task generation and
the private server/agent sidecar are injected after admission (or supplied by
the companion ``cybergym_sidecar`` module).  Running without an injected
executor therefore produces explicit blocked rows instead of silently falling
back to a host shell, Docker default network, or a different model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import pathlib
import re
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from devtools.benchmarks.common.manifests import (
    BenchmarkAdmissionRefused,
    admit_benchmark_run,
    finalize_run_manifest,
)
from devtools.benchmarks.common.model_slots import (
    disabled_subagents_setting,
    single_model_reviewer_slots_setting,
)
from devtools.benchmarks.common.run_roots import (
    assert_file_output_outside_repo,
    assert_outside_repo,
    run_root,
)
from devtools.benchmarks.cybergym.cybergym_adapter import (
    BENCHMARK_NAME,
    DEFAULT_BUDGET_CAP_USD,
    DEFAULT_FINAL_POC_PATH,
    DEFAULT_LEVEL,
    MAX_CROSS_TASK_WORKERS,
    MAX_TASK_TIMEOUT_SEC,
    OFFICIAL_DATA_REVISION,
    OFFICIAL_MODEL,
    OFFICIAL_SOURCE_PIN,
    OFFICIAL_TASKS_SHA256,
    BudgetLedger,
    CyberGymError,
    CyberGymIntegrationUnavailable,
    TaskSpec,
    append_cybergym_result,
    build_generate_task_argv,
    build_task_result_row,
    derive_disabled_tools,
    load_task_catalog,
    mask_task_id,
    output_root_freshness,
    pre_admission_report,
    run_campaign,
    safe_task_id,
    source_tree_digest,
    task_contract_metadata,
    validate_model_pin,
    validate_positive_finite,
    validate_positive_integral,
    verify_mask_map,
    verify_source_checkout,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_TIMEOUT_SEC = 2 * 60 * 60
DEFAULT_MAX_ROUNDS = 400
# Runtime tree cap for each measured task.  This is deliberately separate from
# ``--per-task-estimate-usd``: the latter is the campaign reservation, while
# this value is consumed by the isolated Ouroboros server's UsageScope.
DEFAULT_PER_TASK_COST_USD = 20.0


def _row_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Summarize terminal rows without treating planned rows as completed."""
    return {
        "rows_written": len(rows),
        "completed_count": sum(1 for row in rows if row.get("status") == "completed"),
        "genuine_failure_count": sum(
            1
            for row in rows
            if row.get("status") not in {"infra_failed", "blocked", "planned"}
            and row.get("final_submission_success") is False
        ),
        "planned_count": sum(1 for row in rows if row.get("status") == "planned"),
        "infra_count": sum(
            1 for row in rows if row.get("status") in {"infra_failed", "blocked"}
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse only scalar/argv intent; no filesystem or network probes occur here."""
    parser = argparse.ArgumentParser(description="Run the Level-1 CyberGym adapter")
    parser.add_argument("--repo-dir", default=str(REPO_ROOT), help="clean Ouroboros execution seed")
    parser.add_argument("--source-root", default="", help="pinned CyberGym checkout")
    parser.add_argument("--data-root", default="", help="CyberGym data directory")
    parser.add_argument("--tasks-file", default="", help="pinned tasks.json catalog")
    parser.add_argument("--task-id", action="append", default=[], help="task id (repeatable, e.g. arvo:47101)")
    parser.add_argument("--server", default="http://cybergym-internal:8666", help="private CyberGym submit server URL")
    parser.add_argument(
        "--ouroboros-url",
        default="",
        help="external gateway URL for an explicitly injected --executor (concrete path owns one)",
    )
    parser.add_argument("--docker-host", default="", help="explicit rootless Docker unix socket")
    parser.add_argument("--server-image", default="", help="pinned CyberGym server image")
    parser.add_argument("--server-image-digest", default="", help="resolved sha256 digest for server image")
    parser.add_argument("--workspace-image", default="", help="pinned Ouroboros workspace image")
    parser.add_argument("--workspace-image-digest", default="", help="resolved sha256 digest for workspace image")
    parser.add_argument("--server-root", default="", help="host root mounted at the same absolute path in the server sidecar")
    parser.add_argument("--binary-dir", default="", help="pinned CyberGym binary directory inside server-root")
    parser.add_argument(
        "--cybergym-python",
        default="",
        help="Python executable for the pinned CyberGym package (required for paid runs)",
    )
    parser.add_argument("--cybergym-api-key-env", default="CYBERGYM_API_KEY", help="host env name for the private verifier key")
    parser.add_argument("--mask-map", default="", help="task mask-map JSON")
    parser.add_argument("--difficulty", default=DEFAULT_LEVEL)
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash-0731")
    parser.add_argument(
        "--settings-path",
        default=str(pathlib.Path(__file__).with_name("settings_base.json")),
        help="settings template (never the live Ouroboros settings file)",
    )
    parser.add_argument("--out-dir", default="", help="append-only benchmark output root")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--state-dir", default="", help="external local-disk directory for isolated-server mutable state")
    parser.add_argument("--allow-network-state-dir", action="store_true", help="accept a network filesystem for --state-dir (logs a loud warning)")
    parser.add_argument("--reconcile", default="", help="adopt an interrupted run root and deliver its terminal gateway results")
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_CAP_USD)
    parser.add_argument("--per-task-estimate-usd", type=float, default=None,
                        help="finite reservation required for a paid injected executor")
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--max-rounds", type=float, default=DEFAULT_MAX_ROUNDS, help="per-task Ouroboros round ceiling (recorded in applied settings)")
    parser.add_argument(
        "--per-task-cost-usd",
        type=float,
        default=None,
        help="runtime per-task Ouroboros tree cost cap (separate from the ledger estimate)",
    )
    parser.add_argument("--workers", type=int, default=1,
                        help="bounded cross-task lanes; freeze only after pilot validation")
    parser.add_argument("--executor", default="", help="post-admission module:function callback")
    parser.add_argument("--dry-run", action="store_true", help="write a protocol plan without invoking an executor")
    parser.add_argument("--allow-dirty-seed", action="store_true",
                        help="record and proceed with a dirty seed (not submittable)")
    parser.add_argument("--expected-source-sha256", default="")
    parser.add_argument(
        "--expected-data-sha256", default="",
        help="exact SHA-256 of the immutable CyberGym data tree",
    )
    parser.add_argument(
        "--expected-binary-sha256", default="",
        help="exact SHA-256 of the immutable CyberGym binary tree",
    )
    parser.add_argument(
        "--reuse-input-attestation",
        default="",
        help=(
            "prior append-only run_manifest.json whose verified data/binary "
            "observations may be reused without rereading the immutable trees"
        ),
    )
    parser.add_argument("--expected-tasks-sha256", default=OFFICIAL_TASKS_SHA256)
    parser.add_argument("--expected-mask-sha256", default="")
    parser.add_argument("--provider-only", action="append", default=[], help="OpenRouter provider id to allow (repeatable/comma-separated)")
    parser.add_argument("--provider-order", action="append", default=[], help="OpenRouter provider order (repeatable/comma-separated)")
    return parser.parse_args(argv)


def _csv_values(values: Sequence[str] | str | None) -> tuple[str, ...]:
    """Normalize repeatable/comma-separated provider ids without I/O."""
    raw = [values] if isinstance(values, str) else list(values or ())
    result: list[str] = []
    for item in raw:
        result.extend(part.strip() for part in str(item).split(",") if part.strip())
    return tuple(dict.fromkeys(result))


def _load_reused_input_observations(
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load a small prior manifest as the authority for already-paid digests.

    The operator still supplies the exact expected hashes. This fast path only
    avoids rereading the immutable payload after an earlier append-only run
    recorded matching path-bound observations.
    """

    raw_path = str(getattr(args, "reuse_input_attestation", "") or "").strip()
    if not raw_path:
        return None, None
    source = pathlib.Path(raw_path).expanduser()
    if not source.is_absolute():
        raise CyberGymIntegrationUnavailable(
            "--reuse-input-attestation must be an absolute manifest path"
        )
    try:
        source = source.resolve(strict=True)
        payload = source.read_bytes()
        manifest = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise CyberGymIntegrationUnavailable(
            "reused input attestation is unreadable"
        ) from exc
    if source.name != "run_manifest.json" or not isinstance(manifest, Mapping):
        raise CyberGymIntegrationUnavailable(
            "reused input attestation must be a CyberGym run_manifest.json"
        )
    extra = manifest.get("extra")
    if not isinstance(extra, Mapping):
        raise CyberGymIntegrationUnavailable(
            "reused input attestation has no benchmark observations"
        )

    source_sha256 = hashlib.sha256(payload).hexdigest()
    created_at = manifest.get("created_at_unix")
    specifications = (
        (
            "cybergym_data",
            pathlib.Path(args.data_root),
            str(args.expected_data_sha256 or "").strip().lower(),
        ),
        (
            "cybergym_binary",
            pathlib.Path(args.binary_dir),
            str(args.expected_binary_sha256 or "").strip().lower(),
        ),
    )
    observations: list[dict[str, Any]] = []
    for field, expected_path, expected_sha256 in specifications:
        value = extra.get(field)
        if not isinstance(value, Mapping):
            raise CyberGymIntegrationUnavailable(
                f"reused input attestation omitted {field}"
            )
        try:
            observed_path = pathlib.Path(str(value.get("path") or "")).resolve(
                strict=True
            )
            resolved_expected_path = expected_path.resolve(strict=True)
        except OSError as exc:
            raise CyberGymIntegrationUnavailable(
                f"reused input attestation path is unavailable for {field}"
            ) from exc
        observed_sha256 = str(value.get("sha256") or "").strip().lower()
        observed_expected = str(value.get("expected_sha256") or "").strip().lower()
        if observed_path != resolved_expected_path:
            raise CyberGymIntegrationUnavailable(
                f"reused input attestation path does not match {field}"
            )
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or observed_sha256 != expected_sha256
            or observed_expected != expected_sha256
        ):
            raise CyberGymIntegrationUnavailable(
                f"reused input attestation digest does not match {field}"
            )
        files = value.get("files")
        size = value.get("bytes")
        if (
            not isinstance(files, int)
            or isinstance(files, bool)
            or files <= 0
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
        ):
            raise CyberGymIntegrationUnavailable(
                f"reused input attestation counts are invalid for {field}"
            )
        observations.append(
            {
                **dict(value),
                "attestation_mode": "reused_manifest_observation",
                "attestation_source_manifest": str(source),
                "attestation_source_sha256": source_sha256,
                "attestation_created_at_unix": created_at,
            }
        )
    return observations[0], observations[1]


def _sha256_file(path: pathlib.Path | str) -> str:
    return hashlib.sha256(pathlib.Path(path).expanduser().resolve(strict=False).read_bytes()).hexdigest()


def _validate_launcher_values(args: argparse.Namespace) -> None:
    """Normalize scalar launch values and enforce the campaign safety rails.

    This function is deliberately filesystem-free.  Paid input hashes are
    required here so an invalid declaration cannot reach server startup; the
    bytes themselves are attested by the concrete executor after admission.
    """
    args.model = validate_model_pin(args.model, expected=OFFICIAL_MODEL)
    args.budget_usd = validate_positive_finite(args.budget_usd, field="budget_usd")
    if args.budget_usd > DEFAULT_BUDGET_CAP_USD:
        raise ValueError(
            f"budget_usd may not exceed the CyberGym hard cap of {DEFAULT_BUDGET_CAP_USD:.2f}"
        )
    args.timeout_sec = validate_positive_integral(args.timeout_sec, field="timeout_sec")
    if args.timeout_sec > MAX_TASK_TIMEOUT_SEC:
        raise ValueError(
            f"timeout_sec may not exceed the CyberGym task cap of {MAX_TASK_TIMEOUT_SEC}"
        )
    args.max_rounds = validate_positive_integral(
        getattr(args, "max_rounds", DEFAULT_MAX_ROUNDS), field="max_rounds"
    )
    dry_run = getattr(args, "dry_run", False)
    raw_per_task_cost = getattr(args, "per_task_cost_usd", None)
    if raw_per_task_cost is None:
        if not dry_run:
            raise ValueError(
                "--per-task-cost-usd is required for paid CyberGym execution"
            )
        raw_per_task_cost = DEFAULT_PER_TASK_COST_USD
    args.per_task_cost_usd = validate_positive_finite(
        raw_per_task_cost, field="per_task_cost_usd"
    )
    if args.per_task_cost_usd > args.budget_usd:
        raise ValueError("per_task_cost_usd may not exceed budget_usd")
    args.workers = validate_positive_integral(args.workers, field="workers")
    if args.workers > MAX_CROSS_TASK_WORKERS:
        raise ValueError(
            f"workers may not exceed the CyberGym cross-task cap of {MAX_CROSS_TASK_WORKERS}"
        )
    if args.per_task_estimate_usd is not None:
        args.per_task_estimate_usd = validate_positive_finite(
            args.per_task_estimate_usd, field="per_task_estimate_usd"
        )
        if args.per_task_estimate_usd > args.budget_usd:
            raise ValueError("per_task_estimate_usd may not exceed budget_usd")

    allow_dirty = getattr(args, "allow_dirty_seed", False)
    if not isinstance(dry_run, bool) or not isinstance(allow_dirty, bool):
        raise ValueError("dry_run and allow_dirty_seed must be booleans")
    if allow_dirty and not dry_run:
        raise ValueError("--allow-dirty-seed is not permitted for paid CyberGym execution")

    if not dry_run:
        python_executable = str(getattr(args, "cybergym_python", "") or "").strip()
        if not python_executable:
            raise ValueError(
                "--cybergym-python is required for paid CyberGym execution"
            )
        for field, label in (
            ("expected_data_sha256", "expected-data-sha256"),
            ("expected_binary_sha256", "expected-binary-sha256"),
        ):
            value = str(getattr(args, field, "") or "").strip()
            if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
                raise ValueError(
                    f"--{label} is required for paid CyberGym execution and must be a SHA-256"
                )
        if str(getattr(args, "executor", "") or "").strip() and not str(
            getattr(args, "ouroboros_url", "") or ""
        ).strip():
            raise ValueError("paid --executor requires an explicit --ouroboros-url")


def _declared_task_ids(args: argparse.Namespace) -> list[str]:
    """Normalize explicit ids without touching the task catalog."""
    result: list[str] = []
    seen: set[str] = set()
    for raw in list(args.task_id or []):
        task_id = safe_task_id(str(raw))
        if task_id in seen:
            raise ValueError(f"duplicate task id: {task_id}")
        seen.add(task_id)
        result.append(task_id)
    return result


def _generator_template(
    args: argparse.Namespace, *, server: str | None = None
) -> list[str]:
    """A manifest-safe command shape, using a placeholder task until catalog load."""
    task = "arvo:0"
    return build_generate_task_argv(
        task,
        out_dir="<task-output>",
        data_dir=str(args.data_root or "<cybergym-data>"),
        server=str(server or args.server or "<private-server>"),
        mask_map=str(args.mask_map or "") or None,
        difficulty=str(args.difficulty or DEFAULT_LEVEL),
        python=str(getattr(args, "cybergym_python", "") or "") or None,
    )


def _apply_server_provenance(
    manifest: dict[str, Any], args: argparse.Namespace, applied_url: str
) -> None:
    """Replace a requested server placeholder with the URL actually applied."""
    applied = str(applied_url or "").strip()
    if not applied:
        return
    harness = manifest.setdefault("harness", {})
    harness.setdefault("requested_server", str(args.server))
    harness["server"] = applied
    harness["applied_server"] = applied
    manifest["official_command"] = _generator_template(args, server=applied)


def _load_executor(spec: str) -> Callable[[TaskSpec, pathlib.Path], Mapping[str, Any]]:
    """Resolve an explicitly requested callback only after durable admission."""
    text = str(spec or "").strip()
    if not text or ":" not in text:
        raise CyberGymIntegrationUnavailable(
            "no CyberGym executor supplied; pass --executor module:function or use --dry-run"
        )
    module_name, function_name = text.rsplit(":", 1)
    if not module_name or not function_name:
        raise CyberGymIntegrationUnavailable("--executor must be module:function")
    try:
        module = importlib.import_module(module_name)
        callback = getattr(module, function_name)
    except (ImportError, AttributeError) as exc:
        raise CyberGymIntegrationUnavailable(f"CyberGym executor could not be loaded: {text}") from exc
    if not callable(callback):
        raise CyberGymIntegrationUnavailable(f"CyberGym executor is not callable: {text}")
    return callback


def _build_default_executor(
    args: argparse.Namespace,
    out_root: pathlib.Path,
    *,
    ouroboros_url: str | None = None,
    isolate_data_root: pathlib.Path | None = None,
) -> Callable[[TaskSpec, pathlib.Path], Mapping[str, Any]]:
    """Construct the concrete sidecar executor after admission.

    An explicit ``--executor`` remains useful for a pre-started server or a
    laboratory harness.  The normal paid path must provide all image/socket
    pins; a missing value is a typed refusal, never a host-shell fallback.
    """
    effective_ouroboros_url = str(ouroboros_url or getattr(args, "ouroboros_url", "") or "").strip()
    required = {
        "--docker-host": args.docker_host,
        "--server-image": args.server_image,
        "--server-image-digest": args.server_image_digest,
        "--workspace-image": args.workspace_image,
        "--workspace-image-digest": args.workspace_image_digest,
        "--server-root": args.server_root,
        "--binary-dir": args.binary_dir,
        "--cybergym-python": getattr(args, "cybergym_python", ""),
        "--expected-data-sha256": getattr(args, "expected_data_sha256", ""),
        "--expected-binary-sha256": getattr(args, "expected_binary_sha256", ""),
        "managed Ouroboros URL": effective_ouroboros_url,
    }
    missing = [name for name, value in required.items() if not str(value or "").strip()]
    if missing:
        raise CyberGymIntegrationUnavailable(
            "concrete CyberGym executor requires " + ", ".join(missing)
        )
    python_executable = str(getattr(args, "cybergym_python", "") or "").strip()
    resolved_python = shutil.which(python_executable)
    if not resolved_python:
        raise CyberGymIntegrationUnavailable(
            "the --cybergym-python executable is not available on PATH"
        )
    from devtools.benchmarks.cybergym.cybergym_executor import ExecutorConfig, build_executor

    disabled_tools = derive_disabled_tools()
    data_attestation, binary_attestation = _load_reused_input_observations(args)
    config = ExecutorConfig(
        campaign_id=out_root.name,
        source_root=pathlib.Path(args.source_root),
        data_root=pathlib.Path(args.data_root),
        mask_map=pathlib.Path(args.mask_map),
        run_root=out_root,
        server_root=pathlib.Path(args.server_root),
        binary_dir=pathlib.Path(args.binary_dir),
        expected_data_sha256=str(getattr(args, "expected_data_sha256", "") or ""),
        expected_binary_sha256=str(getattr(args, "expected_binary_sha256", "") or ""),
        preverified_data_observation=data_attestation,
        preverified_binary_observation=binary_attestation,
        server_image=str(args.server_image),
        server_image_digest=str(args.server_image_digest),
        workspace_image=str(args.workspace_image),
        workspace_image_digest=str(args.workspace_image_digest),
        ouroboros_url=effective_ouroboros_url,
        docker_host=str(args.docker_host),
        model=str(args.model),
        settings_path=out_root / "settings_applied.json",
        difficulty=str(args.difficulty or DEFAULT_LEVEL),
        task_timeout_sec=int(args.timeout_sec),
        api_key_env=str(args.cybergym_api_key_env),
        provider_only=_csv_values(getattr(args, "provider_only", ())),
        provider_order=_csv_values(getattr(args, "provider_order", ())),
        disabled_tools=disabled_tools,
        python_executable=resolved_python,
        isolate_data_root=isolate_data_root,
    )
    executor = build_executor(config)

    def callback(task: TaskSpec, task_dir: pathlib.Path) -> Mapping[str, Any]:
        return executor.run_task(task, task_dir)

    # The launcher finalizer owns terminal bookkeeping; this closure's object
    # is kept alive by the callback.  ``main`` invokes close in its finally
    # path through the optional attribute below.
    setattr(callback, "close", executor.close)
    setattr(callback, "prepare", executor.start)
    setattr(callback, "executor", executor)
    return callback


def _paid_prepare_failure_text(exc: BaseException) -> str:
    """Keep the typed refuse line, plus a short secret-free cause."""
    detail = " ".join(str(exc).split())
    suffix = type(exc).__name__
    if detail and detail != suffix:
        if len(detail) > 240:
            detail = detail[:237] + "..."
        suffix = f"{suffix}: {detail}"
    return "paid executor preparation failed: " + suffix


def _prepared_observation(
    executor: Any,
    prepared: Any,
    name: str,
) -> dict[str, Any]:
    """Read one structured readiness observation from a paid executor.

    The concrete executor stores observations on its object; an injected
    callback may return them from ``prepare`` or expose the same attributes.
    No fallback hash or synthetic provider charge is accepted.
    """
    candidates: list[Any] = []
    if isinstance(prepared, Mapping):
        nested = prepared.get(name)
        if isinstance(nested, Mapping):
            candidates.append(nested)
        if name == "provider_observation" and "cost_usd" in prepared:
            candidates.append(prepared)
    elif prepared is not None:
        value = getattr(prepared, name, None)
        if isinstance(value, Mapping):
            candidates.append(value)
    nested_executor = getattr(executor, "executor", None)
    for owner in (nested_executor, executor):
        value = getattr(owner, name, None)
        if isinstance(value, Mapping):
            candidates.append(value)
    if not candidates:
        raise CyberGymIntegrationUnavailable(
            f"paid executor did not expose an exact {name} observation"
        )
    return dict(candidates[0])


def _validate_paid_observations(
    executor: Any,
    prepared: Any,
    *,
    model: str,
    expected_data_sha256: str,
    expected_binary_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float]:
    """Validate provider and immutable-input evidence returned by ``prepare``."""
    provider = _prepared_observation(executor, prepared, "provider_observation")
    if str(provider.get("status") or "").strip().lower() != "passed":
        raise CyberGymIntegrationUnavailable("provider probe did not pass")
    if provider.get("cost_estimated") is not False:
        raise CyberGymIntegrationUnavailable(
            "provider probe cost is unknown or estimated"
        )
    observed_model = str(
        provider.get("observed_model") or provider.get("model") or ""
    ).strip()
    if observed_model != model:
        raise CyberGymIntegrationUnavailable(
            "provider probe served a model different from the pinned request"
        )
    if not str(provider.get("provider") or "").strip() or not str(
        provider.get("response_id") or ""
    ).strip():
        raise CyberGymIntegrationUnavailable(
            "provider probe omitted authoritative provider or response id"
        )
    raw_cost = provider.get("cost_usd")
    if isinstance(raw_cost, bool):
        raise CyberGymIntegrationUnavailable("provider probe cost is not numeric")
    try:
        cost = float(raw_cost)
    except (TypeError, ValueError) as exc:
        raise CyberGymIntegrationUnavailable("provider probe cost is unknown") from exc
    if not math.isfinite(cost) or cost < 0:
        raise CyberGymIntegrationUnavailable("provider probe cost is not finite")

    data = _prepared_observation(executor, prepared, "data_observation")
    binary = _prepared_observation(executor, prepared, "binary_observation")
    expected_data = str(expected_data_sha256 or "").strip().lower()
    expected_binary = str(expected_binary_sha256 or "").strip().lower()
    if str(data.get("sha256") or "").strip().lower() != expected_data:
        raise CyberGymIntegrationUnavailable("CyberGym data observation does not match its pin")
    if str(binary.get("sha256") or "").strip().lower() != expected_binary:
        raise CyberGymIntegrationUnavailable("CyberGym binary observation does not match its pin")
    return provider, data, binary, cost


def _redacted_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only non-secret provider/input attestation fields in the manifest."""
    allowed = {
        "status",
        "ts_unix",
        "requested_model",
        "observed_model",
        "model",
        "provider",
        "provider_pool_membership",
        "provider_policy",
        "inventory",
        "response_id",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "cost_usd",
        "cost_estimated",
        "label",
        "path",
        "sha256",
        "expected_sha256",
        "files",
        "bytes",
        "attestation_mode",
        "attestation_source_manifest",
        "attestation_source_sha256",
        "attestation_created_at_unix",
    }
    return {str(key): value for key, value in observation.items() if str(key) in allowed}


def _record_provider_probe_cost(
    out_root: pathlib.Path,
    budget_usd: float,
    cost_usd: float,
) -> dict[str, Any]:
    """Charge the exact provider readiness request before task claims."""
    ledger = BudgetLedger(out_root / "claims.jsonl", cap_usd=float(budget_usd))
    return ledger.record_campaign_cost(cost_usd, label="provider_probe")


def _start_isolated_ouroboros_server(
    args: argparse.Namespace,
    out_root: pathlib.Path,
    applied_settings: pathlib.Path,
    expected_commit: str,
    expected_settings_sha256: str = "",
) -> Any:
    """Start the campaign-owned Ouroboros gateway on the selected Docker daemon.

    The wrapper is imported and started only after admission and settings
    rendering; the live operator server and settings are never reused.
    """
    from devtools.benchmarks.cybergym.cybergym_server import CyberGymIsolatedServer
    provider_key = str(os.environ.get("OPENROUTER_API_KEY", "") or "").strip()
    if not provider_key:
        raise CyberGymIntegrationUnavailable(
            "OPENROUTER_API_KEY must be injected in the host environment for the isolated server"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_settings_sha256 or "").strip().lower()):
        raise CyberGymIntegrationUnavailable(
            "paid CyberGym execution requires a producer settings SHA-256 digest"
        )

    server: Any | None = None
    try:
        server = CyberGymIsolatedServer(
            pathlib.Path(args.repo_dir),
            out_root,
            applied_settings,
            str(args.docker_host),
            expected_commit=str(expected_commit or ""),
            provider_key=provider_key,
            expected_settings_sha256=str(expected_settings_sha256 or ""),
            state_dir=pathlib.Path(args.state_dir) if str(getattr(args, "state_dir", "") or "").strip() else None,
            allow_network_state_dir=bool(getattr(args, "allow_network_state_dir", False)),
        )
        return server.start(ready_timeout=180)
    except Exception as exc:
        # ``CyberGymIsolatedServer.start`` already cleans up its own partial
        # state, but keep this boundary safe for injected/fake implementations
        # and for failures in construction before the wrapper can do so.
        if server is not None:
            try:
                server.close()
            except Exception:
                pass
        raise CyberGymIntegrationUnavailable(
            "isolated Ouroboros server preparation failed: "
            + type(exc).__name__
        ) from exc


def _cleanup_execution_resources(
    executor: Any | None,
    isolated_server: Any | None,
    manifest: dict[str, Any],
) -> None:
    """Close owned resources while retaining a typed custody report.

    A callback may return ``{"ok": False}`` when a late gateway result still
    owns live workers; the server then stays alive for reattachment, and both
    decisions persist through the manifest finalizer.  Cleanup errors are
    recorded without copying endpoint or credential data from messages."""
    extra = manifest.setdefault("extra", {})
    executor_cleanup: dict[str, Any] = {
        "attempted": False,
        "status": "not_available",
        "ok": None,
    }
    server_close_allowed = True
    try:
        if executor is not None:
            close = getattr(executor, "close", None)
            if callable(close):
                executor_cleanup["attempted"] = True
                close_report = close()
                if isinstance(close_report, Mapping):
                    # Keep the executor's structured fields at the top level
                    # so the manifest remains compatible with existing cleanup
                    # attestations while adding only launcher-owned metadata.
                    executor_cleanup = dict(close_report)
                    executor_cleanup["attempted"] = True
                    executor_cleanup.setdefault("status", "reported")
                    executor_cleanup.setdefault("ok", None)
                    if executor_cleanup.get("ok") is False:
                        # An unresolved gateway attempt owns live worker/server
                        # custody.  Keep the isolated server alive for reattach.
                        server_close_allowed = False
                else:
                    executor_cleanup.update({"status": "not_reported", "ok": None})
    except BaseException as exc:
        executor_cleanup.update(
            {"status": "error", "ok": False, "error_type": type(exc).__name__}
        )
        extra["executor_cleanup"] = executor_cleanup
        server_close_allowed = False
        raise
    else:
        extra["executor_cleanup"] = executor_cleanup
    finally:
        server_cleanup: dict[str, Any] = {
            "attempted": isolated_server is not None,
            "close_skipped": isolated_server is not None and not server_close_allowed,
            "status": (
                "not_available"
                if isolated_server is None
                else "skipped_custody"
                if not server_close_allowed
                else "pending"
            ),
        }
        if isolated_server is not None and server_close_allowed:
            try:
                isolated_server.close()
            except BaseException as exc:
                server_cleanup.update(
                    {"status": "error", "error_type": type(exc).__name__}
                )
                extra["server_cleanup"] = server_cleanup
                extra["close_skipped"] = False
                raise
            else:
                server_cleanup["status"] = "closed"
                state_export = getattr(isolated_server, "state_export", None)
                if isinstance(state_export, Mapping) and state_export: extra["state_export"] = dict(state_export)
        extra["server_cleanup"] = server_cleanup
        extra["close_skipped"] = bool(server_cleanup["close_skipped"])


def _task_specs(
    task_ids: Sequence[str],
    *,
    level: str = DEFAULT_LEVEL,
    contract: Mapping[str, Any] | None = None,
) -> list[TaskSpec]:
    """Build task values carrying the immutable contract into the executor seam."""
    base_contract = dict(contract or task_contract_metadata(level=level))
    specs: list[TaskSpec] = []
    for raw_task_id in task_ids:
        task_id = safe_task_id(raw_task_id)
        task_contract = {**base_contract, "task_id": task_id}
        specs.append(
            TaskSpec(
                task_id,
                task_id.split(":", 1)[0],
                level,
                {"task_contract": task_contract},
            )
        )
    return specs


def _write_planned_rows(
    out_root: pathlib.Path,
    task_ids: Sequence[str],
    *,
    level: str,
    contract: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        row = build_task_result_row(
            task_id,
            status="planned",
            lifecycle="dry_run",
            level=level,
            masked_id=mask_task_id(task_id),
            infra_reason="dry_run",
            artifact_refs={"run_root": str(out_root)},
            task_contract=contract,
        )
        row["masked_id_source"] = "local_digest_diagnostic_not_upstream_mask"
        append_cybergym_result(out_root, row)
        rows.append(row)
    return rows


def _prepare_applied_settings(
    template_path: pathlib.Path, out_root: pathlib.Path, args: argparse.Namespace
) -> tuple[pathlib.Path, dict[str, Any]]:
    """Derive a sanitized settings snapshot after admission.

    The template is read only after ``admit_benchmark_run`` has persisted the
    manifest.  ``build_isolated_settings`` filters credentials and legacy keys;
    explicit benchmark overrides become the applied, auditable settings for an
    injected server — an ordinary run artifact, never the live settings path.
    """
    try:
        template = json.loads(template_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CyberGymIntegrationUnavailable(
            f"settings template is unreadable or invalid: {template_path}"
        ) from exc
    if not isinstance(template, dict):
        raise CyberGymIntegrationUnavailable("settings template must contain a JSON object")
    # The template is intentionally credential-free.  ``build_isolated_settings``
    # derives grants from declared slots, and its general-purpose provider planner
    # retains a dormant Claude default for compatibility.  Accepting a custom
    # template that already contains a provider secret would therefore make a
    # single-OpenRouter CyberGym run reach a second provider.  Refuse before the
    # artifact is written; never copy or print the value.
    from ouroboros.provider_models import ALL_PROVIDER_CREDENTIAL_KEYS

    provider_fields = set(ALL_PROVIDER_CREDENTIAL_KEYS) | {"GIGACHAT_PROFANITY_CHECK"}
    supplied_provider_fields = sorted(
        key for key in provider_fields
        if str(template.get(key) or "").strip()
    )
    if supplied_provider_fields:
        raise CyberGymIntegrationUnavailable(
            "CyberGym settings template must not carry provider credentials/fields: "
            + ", ".join(supplied_provider_fields)
        )
    from devtools.benchmarks.common.manifests import (
        model_slot_snapshot,
        provider_credential_disclosure,
        write_json,
    )
    from devtools.benchmarks.common.server_runner import build_isolated_settings

    model = validate_model_pin(args.model, expected=OFFICIAL_MODEL)
    budget_usd = validate_positive_finite(args.budget_usd, field="budget_usd")
    timeout_sec = validate_positive_integral(args.timeout_sec, field="timeout_sec")
    max_rounds = validate_positive_integral(
        getattr(args, "max_rounds", DEFAULT_MAX_ROUNDS), field="max_rounds"
    )
    per_task_cost_usd = validate_positive_finite(
        getattr(args, "per_task_cost_usd", DEFAULT_PER_TASK_COST_USD),
        field="per_task_cost_usd",
    )
    workers = validate_positive_integral(
        getattr(args, "workers", 1), field="workers"
    )
    if workers > MAX_CROSS_TASK_WORKERS:
        raise ValueError(
            f"workers may not exceed the CyberGym cross-task cap of {MAX_CROSS_TASK_WORKERS}"
        )
    overrides: dict[str, Any] = {
        "OUROBOROS_MODEL": model,
        "OUROBOROS_MODEL_LIGHT": model,
        "OUROBOROS_MODEL_VISION": model,
        "OUROBOROS_MODEL_CONSCIOUSNESS": model,
        "OUROBOROS_MODEL_FALLBACKS": model,
        "OUROBOROS_MODEL_DEEP_SELF_REVIEW": model,
        "OUROBOROS_WEBSEARCH_MODEL": model,
        "OUROBOROS_SCOPE_REVIEW_MODELS": model,
        "OUROBOROS_SCOPE_REVIEW_MODEL": model,
        # One routed reviewer is the explicit single-model campaign contract.
        # Keeping the legacy projection in sync prevents a stale three-row
        # value from shadowing the structured panel in older consumers.
        "OUROBOROS_REVIEW_MODELS": model,
        "OUROBOROS_MAX_SUBAGENT_DEPTH": 0,
        "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS": "false",
        "OUROBOROS_RUNTIME_MODE": "pro",
        "OUROBOROS_SAFETY_MODE": "off",
        "OUROBOROS_CONTEXT_MODE": "max",
        "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "false",
        "OUROBOROS_MAX_WORKERS": workers,
        "OUROBOROS_MAX_ROUNDS": max_rounds,
        "OUROBOROS_TASK_IDLE_TIMEOUT_SEC": 900,
        "OUROBOROS_PER_TASK_COST_USD": per_task_cost_usd,
        "TOTAL_BUDGET": budget_usd,
        "OUROBOROS_TASK_ABS_CEILING_SEC": timeout_sec,
        # Task review runs in automatic mode (the host reviewer selects
        # substantive attempts); its verdict is advisory and is bounded to two
        # paid review cycles as requested for this cohort.
        "OUROBOROS_TASK_REVIEW_MODE": "auto",
        "OUROBOROS_REVIEW_ENFORCEMENT": "advisory",
        "OUROBOROS_REVIEW_MAX_CYCLES": "2",
        "OUROBOROS_POST_TASK_EVOLUTION": "false",
        "OUROBOROS_POST_TASK_EVOLUTION_CADENCE": "off",
        # Keep internet access explicit and auditable.  Exact-model OpenRouter
        # server search is model-discretionary and did not execute in live
        # forced probes; the explicit web_search tool therefore uses the
        # query-visible DDGS retrieval path instead.
        "OUROBOROS_MAIN_WEB_SEARCH": "off",
        "OUROBOROS_MAIN_WEB_SEARCH_ENGINE": "auto",
        "OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS": 0,
        "OUROBOROS_WEBSEARCH_BACKEND": "ddgs",
        "OUROBOROS_IMAGE_INPUT_MODE": "auto",
        "OUROBOROS_RETURN_REASONING": True,
        "OUROBOROS_REASONING_SUMMARY": "auto",
        "MCP_ENABLED": False,
        "MCP_SERVERS": [],
        "OUROBOROS_EFFORT_TASK": "high",
        "OUROBOROS_EFFORT_EVOLUTION": "high",
        # The benchmark task wire is validated as literal ``high``; the
        # broader review surfaces support the stronger ``max`` tier.
        "OUROBOROS_EFFORT_REVIEW": "max",
        "OUROBOROS_EFFORT_SCOPE_REVIEW": "max",
        "OUROBOROS_EFFORT_DEEP_SELF_REVIEW": "max",
        "OUROBOROS_EFFORT_CONSCIOUSNESS": "high",
    }
    # Structured no-swarm/reviewer declarations are explicit overrides rather
    # than values accidentally inherited from the live settings file.  Keep a
    # disabled actor row for provenance, but make the execution authority
    # explicit: the runtime must refuse delegated children.
    overrides["OUROBOROS_SUBAGENTS"] = disabled_subagents_setting(model)
    overrides["OUROBOROS_REVIEWER_SLOTS"] = single_model_reviewer_slots_setting(
        model,
        review_slots=1,
        scope_slots=1,
        review_effort="max",
        scope_effort="max",
    )
    # Keep automatic routing explicit and parameter-compatible. Optional
    # only/order flags remain an auditable laboratory override.
    provider = {"allow_fallbacks": True, "require_parameters": True}
    provider_only = _csv_values(getattr(args, "provider_only", ()))
    provider_order = _csv_values(getattr(args, "provider_order", ()))
    if provider_only and provider_order and not set(provider_only).issubset(provider_order):
        raise CyberGymIntegrationUnavailable(
            "provider-only entries must be included in provider-order"
        )
    if provider_only:
        provider["only"] = list(provider_only)
    if provider_order:
        provider["order"] = list(provider_order)
    overrides["OUROBOROS_OR_PROVIDER"] = json.dumps(provider, separators=(",", ":"))
    # The structured panel explicitly disables the optional Claude SDK advisory
    # route. Preserve the generic planner's default-fallback semantics elsewhere,
    # but do not let an empty transport field resurrect an Anthropic grant here.
    applied = build_isolated_settings(
        template,
        include_claude_sdk_defaults=False,
        **overrides,
    )
    output_path = out_root / "settings_applied.json"
    write_json(output_path, applied)
    # Hash the exact producer serialization, then verify the atomic writer
    # left those bytes in place. The expected digest is passed downstream and
    # compared against file bytes again before the isolated copy, so it must
    # bind the on-disk form; a replacement after this point is rejected
    # instead of becoming a new baseline during isolated-server setup.
    # ``write_json`` persists through the atomic text writer, which applies
    # platform newline semantics — the on-disk bytes carry ``os.linesep``
    # (CRLF on Windows). JSON escapes newlines inside strings, so only the
    # structural indent newlines are translated.
    serialized_settings = (
        json.dumps(applied, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    if os.linesep != "\n":
        serialized_settings = serialized_settings.replace(
            b"\n", os.linesep.encode("ascii")
        )
    try:
        if output_path.read_bytes() != serialized_settings:
            raise CyberGymIntegrationUnavailable(
                "applied settings changed during producer write"
            )
    except OSError as exc:
        raise CyberGymIntegrationUnavailable(
            "applied settings cannot be verified after producer write"
        ) from exc
    settings_sha256 = hashlib.sha256(serialized_settings).hexdigest()
    model_slots = model_slot_snapshot(output_path, env_overrides=False)
    applied_model = str(model_slots.get("OUROBOROS_MODEL") or "").strip()
    if applied_model != model:
        raise CyberGymIntegrationUnavailable(
            "applied settings changed the pinned model: "
            f"expected {model!r}, got {applied_model!r}"
        )
    model_mismatches = []
    for key, value in model_slots.items():
        if "MODEL" not in key or not value:
            continue
        configured = [item.strip() for item in str(value).split(",") if item.strip()]
        if any(item != model for item in configured):
            model_mismatches.append(key)
    if model_mismatches:
        raise CyberGymIntegrationUnavailable(
            "applied settings contain non-pinned model slots: " + ", ".join(model_mismatches)
        )
    return output_path, {
        "path": str(output_path),
        "sha256": settings_sha256,
        "template_path": str(template_path),
        "requested_model": model,
        "model": applied_model,
        "max_rounds": max_rounds,
        "per_task_cost_usd": per_task_cost_usd,
        "workers": workers,
        "model_slots": model_slots,
        "budget_usd": budget_usd,
        "task_abs_ceiling_sec": timeout_sec,
        "provider_policy": provider,
        "provider_probe_required": True,
        "provider_policy_complete": True,
        "provider_routing_mode": "pinned_pool" if provider_only or provider_order else "automatic",
        "provider_credentials": provider_credential_disclosure(
            output_path,
            runtime_credentials={
                "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", ""),
            },
            include_claude_sdk_defaults=False,
        ),
        "effective_overrides": {
            key: overrides[key]
            for key in sorted(overrides)
            if key != "OUROBOROS_OR_PROVIDER"
        },
        "keys": sorted(str(key) for key in applied if isinstance(key, str)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if str(getattr(args, "reconcile", "") or "").strip():
        from devtools.benchmarks.cybergym.cybergym_reconcile import reconcile_main
        return reconcile_main(args)
    # Everything through this point is argument/path arithmetic.  In particular,
    # do not read tasks.json, inspect Docker, import the optional upstream package,
    # or create a directory before the shared admission manifest exists.
    try:
        _validate_launcher_values(args)
    except ValueError as exc:
        print(f"[cybergym] pre-admission refusal: {exc}", file=sys.stderr)
        return 2
    try:
        declared_ids = _declared_task_ids(args)
    except ValueError as exc:
        print(f"[cybergym] pre-admission refusal: {exc}", file=sys.stderr)
        return 2

    repo_dir = pathlib.Path(args.repo_dir).expanduser().resolve(strict=False)
    out_root = pathlib.Path(args.out_dir).expanduser().resolve(strict=False) if args.out_dir else run_root(BENCHMARK_NAME, args.run_id)
    manifest_path = out_root / "run_manifest.json"
    ledger_path = out_root / "result_index.jsonl"
    settings_path = pathlib.Path(args.settings_path).expanduser().resolve(strict=False) if args.settings_path else None

    # Pure confinement is intentionally separate from ensure_* helpers: the
    # latter mkdir and are forbidden before admission by launcher_audit.
    try:
        # An explicit run directory must be a new append-only root.  Perform
        # this small read-only probe before the shared manifest writer so a
        # stale directory can never be overwritten.
        freshness = output_root_freshness(out_root)
        if not freshness.get("ok"):
            print(
                "[cybergym] pre-admission path refusal: "
                + str(freshness.get("reason") or "output root is not fresh"),
                file=sys.stderr,
            )
            return 2
        out_root = pathlib.Path(str(freshness["path"]))
        manifest_path = out_root / "run_manifest.json"
        ledger_path = out_root / "result_index.jsonl"
        assert_outside_repo(out_root, repo_dir)
        assert_file_output_outside_repo(manifest_path, repo_dir)
        assert_file_output_outside_repo(ledger_path, repo_dir)
        if args.tasks_file:
            assert_file_output_outside_repo(pathlib.Path(args.tasks_file), repo_dir)
        if args.source_root:
            assert_outside_repo(pathlib.Path(args.source_root), repo_dir)
        if args.data_root:
            assert_outside_repo(pathlib.Path(args.data_root), repo_dir)
        if args.mask_map:
            assert_file_output_outside_repo(pathlib.Path(args.mask_map), repo_dir)
        if args.server_root:
            assert_outside_repo(pathlib.Path(args.server_root), repo_dir)
        if args.binary_dir:
            assert_outside_repo(pathlib.Path(args.binary_dir), repo_dir)
        if str(getattr(args, "state_dir", "") or "").strip(): assert_outside_repo(pathlib.Path(args.state_dir), repo_dir)
    except (CyberGymError, ValueError, OSError) as exc:
        print(f"[cybergym] pre-admission path refusal: {exc}", file=sys.stderr)
        return 2

    report = pre_admission_report(
        task_ids=declared_ids,
        output_root=out_root,
        repo_dir=repo_dir,
        source_root=args.source_root,
        data_root=args.data_root,
        server_url=args.server,
        difficulty=str(args.difficulty or DEFAULT_LEVEL),
        model=str(args.model or ""),
        api_key=None,
        require_api_key=False,
        settings_path=settings_path,
        require_settings=True,
        require_inputs=not bool(args.dry_run),
        network_mode="cybergym-internal",
        mask_map=args.mask_map,
        server_root=args.server_root,
        binary_dir=args.binary_dir,
    )
    if not report["ok"]:
        print("[cybergym] pre-admission refusal: " + "; ".join(report["reasons"]), file=sys.stderr)
        return 2

    try:
        manifest = admit_benchmark_run(
            manifest_path,
            benchmark=BENCHMARK_NAME,
            run_root=out_root,
            repo_dir=repo_dir,
            requested_task_ids=declared_ids,
            # Paid runs have already rejected ``--allow-dirty-seed`` above;
            # keep the explicit dry-run escape only for diagnostic planning.
            require_clean=(
                True
                if not bool(getattr(args, "dry_run", False))
                else not bool(getattr(args, "allow_dirty_seed", False))
            ),
            argv=list(sys.argv if argv is None else [sys.argv[0], *argv]),
            dataset="sunblaze-ucb/cybergym",
            harness={
                "model": str(args.model),
                "difficulty": str(args.difficulty),
                "server": str(args.server),
                "requested_server": str(args.server),
                "timeout_sec": int(args.timeout_sec),
                "max_rounds": int(getattr(args, "max_rounds", DEFAULT_MAX_ROUNDS)),
                "per_task_cost_usd": float(
                    getattr(args, "per_task_cost_usd", DEFAULT_PER_TASK_COST_USD)
                ),
                "executor": str(args.executor or "concrete_sidecar"),
                "workers": int(args.workers),
                "provider_only": list(_csv_values(getattr(args, "provider_only", ()))),
                "provider_order": list(_csv_values(getattr(args, "provider_order", ()))),
                "executor_mode": "injected" if args.executor else "concrete_sidecar",
                "expected_data_sha256": str(
                    getattr(args, "expected_data_sha256", "") or ""
                ),
                "expected_binary_sha256": str(
                    getattr(args, "expected_binary_sha256", "") or ""
                ),
            },
            official_command=_generator_template(args),
            isolated_data_root=str(args.data_root or ""),
            settings_path=settings_path,
            # The applied CyberGym profile explicitly disables both Claude SDK
            # transports; keep even the pre-admission/refusal disclosure truthful.
            include_claude_sdk_defaults=False,
            # The sidecar receives a settings file and a fresh environment;
            # refusal provenance must use that same file as its authority.
            settings_authoritative_env=True,
            output_paths={
                "run_root": str(out_root),
                "manifest": str(manifest_path),
                "ledger": str(ledger_path),
            },
            extra={
                "source_pin": OFFICIAL_SOURCE_PIN,
                "data_revision": OFFICIAL_DATA_REVISION,
                "tasks_sha256_expected": str(args.expected_tasks_sha256 or ""),
                "data_sha256_expected": str(
                    getattr(args, "expected_data_sha256", "") or ""
                ),
                "binary_sha256_expected": str(
                    getattr(args, "expected_binary_sha256", "") or ""
                ),
                "metric_name": "final_submission",
                "any_of_projection": "diagnostic_only",
                "network_contract": "custom_bridge_unrestricted_outbound_private_sidecar",
                "network_name": "cybergym-internal",
                "docker_network_internal": False,
                "server_host_publish": False,
                "trajectory_audit": {
                    "required": True,
                    "status": "pending",
                    "promotion_gate": True,
                },
                "final_poc_basename": "final.poc",
                "budget_cap_usd": float(args.budget_usd),
                "max_rounds": int(getattr(args, "max_rounds", DEFAULT_MAX_ROUNDS)),
                "per_task_cost_usd": float(
                    getattr(args, "per_task_cost_usd", DEFAULT_PER_TASK_COST_USD)
                ),
            },
        )
    except BenchmarkAdmissionRefused as exc:
        print(f"[cybergym] admission refused: {exc}", file=sys.stderr)
        return 2

    with finalize_run_manifest(manifest_path, manifest, outcome="completed") as final:
        try:
            task_ids = list(declared_ids)
            catalog: dict[str, Any] | None = None
            if not args.dry_run:
                if not args.source_root or not args.data_root or not args.tasks_file or not args.mask_map:
                    raise CyberGymError(
                        "paid CyberGym execution requires source-root, data-root, tasks-file, and mask-map"
                    )
                source_provenance = verify_source_checkout(
                    args.source_root,
                    expected_commit=OFFICIAL_SOURCE_PIN,
                    require_clean=True,
                )
                manifest["extra"]["cybergym_source"] = source_provenance
                observed_source_digest = source_tree_digest(args.source_root)
                expected_source_digest = str(args.expected_source_sha256 or "").strip().lower()
                if expected_source_digest and observed_source_digest != expected_source_digest:
                    raise CyberGymError(
                        "source tree SHA-256 mismatch: "
                        f"expected {expected_source_digest}, got {observed_source_digest}"
                    )
                manifest["extra"]["cybergym_source"]["tree_sha256"] = observed_source_digest
                expected_mask = str(args.expected_mask_sha256 or "").strip().lower()
                if not expected_mask:
                    raise CyberGymError("paid CyberGym execution requires --expected-mask-sha256")
                mask_info = verify_mask_map(args.mask_map, declared_ids, expected_sha256=expected_mask)
                manifest["extra"]["mask_map"] = mask_info
            if args.tasks_file:
                catalog = load_task_catalog(
                    args.tasks_file,
                    expected_sha256=str(args.expected_tasks_sha256 or ""),
                    level=DEFAULT_LEVEL,
                )
                if task_ids:
                    allowed = set(catalog["task_ids"])
                    missing = [task_id for task_id in task_ids if task_id not in allowed]
                    if missing:
                        raise CyberGymError("requested task ids are absent from pinned catalog: " + ", ".join(missing))
                else:
                    task_ids = list(catalog["task_ids"])
                manifest["extra"]["task_catalog"] = catalog
                if not args.dry_run and catalog is not None:
                    # When no explicit subset was supplied, validate coverage
                    # against the complete pinned order as well; a partial map
                    # must never silently become a different denominator.
                    if not declared_ids:
                        manifest["extra"]["mask_map"] = verify_mask_map(
                            args.mask_map, catalog["task_ids"], expected_sha256=str(args.expected_mask_sha256).lower()
                        )
            if not task_ids:
                final.update({
                    "outcome": "refused",
                    "exit_code": 2,
                    "refusal": {"stage": "task_selection", "reason": "no_task_ids", "exit_code": 2},
                })
                print("[cybergym] no tasks selected", file=sys.stderr)
                return 2

            manifest["requested_task_ids"] = task_ids
            manifest["requested_count"] = len(task_ids)

            contract = task_contract_metadata(
                model=args.model,
                level=DEFAULT_LEVEL,
                source_pin=OFFICIAL_SOURCE_PIN,
                data_revision=OFFICIAL_DATA_REVISION,
                tasks_sha256=str(args.expected_tasks_sha256 or OFFICIAL_TASKS_SHA256),
                final_poc_path=DEFAULT_FINAL_POC_PATH,
                disabled_tools=derive_disabled_tools(),
            )
            if not args.dry_run:
                contract["data_sha256"] = str(
                    getattr(args, "expected_data_sha256", "") or ""
                ).lower()
                contract["binary_sha256"] = str(
                    getattr(args, "expected_binary_sha256", "") or ""
                ).lower()
            if isinstance(manifest.get("extra", {}).get("mask_map"), Mapping):
                contract["mask_map_sha256"] = str(
                    manifest["extra"]["mask_map"].get("sha256") or ""
                )
            manifest.setdefault("extra", {})["task_contract"] = contract

            applied_path, applied_metadata = _prepare_applied_settings(settings_path, out_root, args)
            manifest.setdefault("extra", {})["settings_snapshot"] = applied_metadata
            manifest.setdefault("output_paths", {})["settings_applied"] = str(applied_path)
            manifest["model_slots"] = dict(applied_metadata.get("model_slots") or {})
            manifest["provider_credentials"] = dict(applied_metadata.get("provider_credentials") or {})
            manifest.setdefault("harness", {})["applied_model"] = str(
                applied_metadata.get("model") or ""
            )

            if args.dry_run:
                rows = _write_planned_rows(
                    out_root, task_ids, level=DEFAULT_LEVEL, contract=contract
                )
            else:
                if args.per_task_estimate_usd is None:
                    raise CyberGymIntegrationUnavailable(
                        "paid CyberGym execution requires --per-task-estimate-usd; no price is invented"
                    )
                executor: Callable[[TaskSpec, pathlib.Path], Mapping[str, Any]] | None = None
                isolated_server: Any | None = None
                # Keep server startup, executor construction, and execution in
                # one ownership scope.  If construction or provider
                # preparation fails after the server starts, the server still
                # receives the same close path as a normally completed run.
                try:
                    if args.executor:
                        # An injected callback owns its own gateway contract.
                        # It is intentionally an explicit lab seam, never a
                        # silent replacement for the campaign-owned server.
                        executor = _load_executor(args.executor)
                        manifest.setdefault("harness", {})["ouroboros_url"] = str(
                            args.ouroboros_url
                        )
                    else:
                        if str(args.ouroboros_url or "").strip():
                            raise CyberGymIntegrationUnavailable(
                                "--ouroboros-url is only allowed with an injected --executor; "
                                "the concrete path starts its campaign-owned server"
                            )
                        seed_source = manifest.get("source")
                        expected_seed_commit = (
                            str(seed_source.get("head") or "").strip()
                            if isinstance(seed_source, Mapping)
                            else ""
                        )
                        if not expected_seed_commit:
                            raise CyberGymIntegrationUnavailable(
                                "admission did not provide a committed Ouroboros seed identity"
                            )
                        isolated_server = _start_isolated_ouroboros_server(
                            args,
                            out_root,
                            applied_path,
                            expected_seed_commit,
                            str(applied_metadata.get("sha256") or ""),
                        )
                        manifest.setdefault("extra", {})["ouroboros_server"] = dict(
                            getattr(isolated_server, "attestation", {}) or {}
                        )
                        from devtools.benchmarks.cybergym.cybergym_server import state_layout_manifest
                        manifest["extra"]["state_layout"] = state_layout_manifest(isolated_server)
                        manifest.setdefault("harness", {})["ouroboros_url"] = str(
                            isolated_server.base_url
                        )
                        executor = _build_default_executor(
                            args, out_root, ouroboros_url=isolated_server.base_url,
                            isolate_data_root=getattr(isolated_server, "data_root", None),
                        )
                    prepare = getattr(executor, "prepare", None)
                    if not callable(prepare):
                        raise CyberGymIntegrationUnavailable(
                            "paid executor must expose a prepare() provider probe"
                        )
                    # Provider/network readiness is established before the first
                    # budget claim, so a failed probe cannot consume a task
                    # reservation or masquerade as a model result.
                    try:
                        prepared = prepare()
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as exc:
                        raise CyberGymIntegrationUnavailable(
                            _paid_prepare_failure_text(exc)
                        ) from exc
                    provider_observation, data_observation, binary_observation, probe_cost = (
                        _validate_paid_observations(
                            executor,
                            prepared,
                            model=str(args.model),
                            expected_data_sha256=str(
                                getattr(args, "expected_data_sha256", "") or ""
                            ),
                            expected_binary_sha256=str(
                                getattr(args, "expected_binary_sha256", "") or ""
                            ),
                        )
                    )
                    manifest["extra"]["provider_probe"] = _redacted_observation(
                        provider_observation
                    )
                    manifest["extra"]["cybergym_data"] = _redacted_observation(
                        data_observation
                    )
                    manifest["extra"]["cybergym_binary"] = _redacted_observation(
                        binary_observation
                    )
                    overhead_event = _record_provider_probe_cost(
                        out_root,
                        float(args.budget_usd),
                        probe_cost,
                    )
                    manifest["extra"]["provider_probe_cost"] = {
                        "label": "provider_probe",
                        "cost_usd": probe_cost,
                        "ledger_event": _redacted_observation(overhead_event),
                    }
                    executor_obj = getattr(executor, "executor", None)
                    if executor_obj is not None:
                        applied_server_url = str(getattr(executor_obj, "server_url", "") or "")
                        if applied_server_url:
                            _apply_server_provenance(
                                manifest, args, applied_server_url
                            )
                    rows = run_campaign(
                        _task_specs(task_ids, contract=contract),
                        run_root=out_root,
                        executor=executor,
                        estimated_cost_usd=float(args.per_task_estimate_usd),
                        budget_cap_usd=float(args.budget_usd),
                        max_workers=int(args.workers),
                    )
                finally:
                    _cleanup_execution_resources(executor, isolated_server, manifest)

            projection_path = out_root / "claims.jsonl"
            try:
                budget = BudgetLedger(projection_path, cap_usd=float(args.budget_usd)).projection()
                manifest["extra"]["budget_projection"] = budget.as_dict()
            except CyberGymError as exc:
                manifest["extra"]["budget_projection"] = {"available": False, "error": str(exc)}
            manifest["extra"].update(_row_counts(rows))
            custody_pending = bool(manifest.get("extra", {}).get("close_skipped"))
            code = (
                2
                if custody_pending
                else 0
                if args.dry_run or all(row.get("status") == "completed" for row in rows)
                else 2
            )
            if custody_pending:
                final.update({"outcome": "custody_pending", "exit_code": code})
            elif code:
                final.update({"outcome": "integration_or_task_failure", "exit_code": code})
            else:
                final.update({"outcome": "completed", "exit_code": 0})
            print(out_root)
            return code
        except CyberGymError as exc:
            final.update({
                "outcome": "refused",
                "exit_code": 2,
                "refusal": {"stage": "integration", "reason": type(exc).__name__, "exit_code": 2},
            })
            print(f"[cybergym] refused: {exc}", file=sys.stderr)
            return 2
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            final.update({
                "outcome": "refused",
                "exit_code": 2,
                "refusal": {"stage": "post_admission_preflight", "reason": type(exc).__name__, "exit_code": 2},
            })
            print(f"[cybergym] failed: {exc}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
