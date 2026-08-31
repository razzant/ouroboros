"""Small, dependency-free CyberGym protocol adapter.

The upstream CyberGym package, Docker client, and model transport are deliberately
absent from this module.  A caller admits a run through the common manifest seam and
then injects an executor.  This keeps argument refusal deterministic and makes the
protocol helpers usable on CI workers that do not install the benchmark extras.
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import hashlib
import json
import math
import os
import pathlib
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from devtools.benchmarks.cybergym.cybergym_dispatch import (  # noqa: F401
    GATEWAY_CIRCUIT_BREAKER_THRESHOLD,
    GatewayCircuitOpen,
    run_dispatched,
)
from devtools.benchmarks.cybergym.cybergym_protocol import (
    BENCHMARK_NAME,
    DEFAULT_LEVEL,
    FINAL_POC_BASENAME,
    CyberGymError,
    _HEX64,
    _SAFE_COMPONENT,
    _choose_final,
    _compact_trial,
    _normalize_trial,
    final_submission,
    mask_task_id,
    safe_task_id,
    safe_task_path,
    validate_high_effort,
)
from devtools.benchmarks.cybergym.cybergym_protocol import (
    DEFAULT_DISABLED_TOOLS,  # noqa: F401
    DEFAULT_FINAL_POC_PATH,  # noqa: F401
    GENERATOR_MODULE,  # noqa: F401
    MAX_TASK_TIMEOUT_SEC,  # noqa: F401
    OFFICIAL_DATA_REVISION,  # noqa: F401
    OFFICIAL_EXIT_EXCLUSIONS,  # noqa: F401
    OFFICIAL_MODEL,  # noqa: F401
    OFFICIAL_SOURCE_PIN,  # noqa: F401
    OFFICIAL_TASKS_SHA256,  # noqa: F401
    TASK_CONTRACT_SCHEMA,  # noqa: F401
    CyberGymAdmissionRefused,  # noqa: F401
    CyberGymPinRefused,  # noqa: F401
    assert_fresh_output_root,  # noqa: F401
    build_gen_task_argv,  # noqa: F401
    build_generate_task_argv,  # noqa: F401
    build_submit_argv,  # noqa: F401
    classify_official_exit,  # noqa: F401
    derive_disabled_tools,  # noqa: F401
    directory_tree_digest,  # noqa: F401
    extract_task_ids,  # noqa: F401
    final_submission_projection,  # noqa: F401
    is_placeholder_api_key,  # noqa: F401
    load_task_catalog,  # noqa: F401
    official_success,  # noqa: F401
    output_root_freshness,  # noqa: F401
    parse_strict_bool,  # noqa: F401
    pre_admission_report,  # noqa: F401
    source_tree_digest,  # noqa: F401
    task_contract_metadata,  # noqa: F401
    task_paths,  # noqa: F401
    task_slug,  # noqa: F401
    validate_model_pin,  # noqa: F401
    validate_positive_finite,  # noqa: F401
    validate_positive_integral,  # noqa: F401
    validate_pre_admission,  # noqa: F401
    verify_directory_digest,  # noqa: F401
    verify_mask_map,  # noqa: F401
    verify_pinned_file,  # noqa: F401
    verify_source_checkout,  # noqa: F401
)

DEFAULT_BUDGET_CAP_USD = 3500.0
MAX_CROSS_TASK_WORKERS = 32
LEDGER_SCHEMA = "ouroboros.benchmark.cybergym.ledger.v1"
RESULT_SCHEMA = "ouroboros.benchmark.cybergym.task_result.v1"
CAPABILITY_FINAL_POC_MISSING = "final_poc_missing_after_fair_completion"
PROTOCOL_FAIL = "protocol_fail"
OFFICIAL_PIN_SKIPS = {
    "arvo:64622": "broken_symlink_official_pin",
}


class CyberGymIntegrationUnavailable(CyberGymError):
    """No explicitly injected post-admission executor is available."""


class FinalPocRefused(CyberGymError):
    """The designated final PoC is absent or is not a regular file."""

    def __init__(self, message: str, *, reason: str = "invalid") -> None:
        super().__init__(message)
        self.reason = str(reason or "invalid")


class LedgerError(CyberGymError):
    """The append-only claim/budget ledger is malformed or unsafe."""


class ClaimRefused(LedgerError):
    """A task already has an active claim."""


class BudgetRefused(LedgerError):
    """A reservation would exceed known budget headroom."""


class BudgetOverspend(BudgetRefused):
    """A measured settlement would take the campaign beyond its hard cap."""


@dataclasses.dataclass(frozen=True)
class BudgetProjection:
    """Pure replay result for one campaign-global budget ledger."""

    cap_usd: float | None
    settled_usd: float
    reserved_usd: float
    unresolved_upper_bound_usd: float | None
    projected_usd: float | None
    available_usd: float | None
    can_dispatch: bool
    reason: str
    active_task_ids: tuple[str, ...] = ()
    active_attempt_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "cap_usd": self.cap_usd,
            "settled_usd": self.settled_usd,
            "reserved_usd": self.reserved_usd,
            "unresolved_upper_bound_usd": self.unresolved_upper_bound_usd,
            "projected_usd": self.projected_usd,
            "available_usd": self.available_usd,
            "can_dispatch": self.can_dispatch,
            "reason": self.reason,
            "active_task_ids": list(self.active_task_ids),
            "active_attempt_ids": list(self.active_attempt_ids),
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclasses.dataclass(frozen=True)
class FinalPoc:
    """Identity of the one designated final PoC."""

    path: str
    sha256: str
    size: int

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """Task value passed to the dependency-injected executor seam."""

    task_id: str
    project: str
    level: str = DEFAULT_LEVEL
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def _final_path(path_or_workspace: pathlib.Path | str) -> pathlib.Path:
    # Do not resolve the final component before ``lstat``: resolving would follow
    # a symlink and make a forbidden marker look like a regular file.
    target = pathlib.Path(path_or_workspace).expanduser()
    return target if target.name == FINAL_POC_BASENAME else target / FINAL_POC_BASENAME


def final_poc_record(path_or_workspace: pathlib.Path | str) -> FinalPoc:
    """Hash exactly one regular, non-symlink ``final.poc`` file.

    CyberGym caps uploaded PoCs at 10 MiB; enforcing that protocol limit here
    prevents an oversized marker from being mistaken for a valid final trial.
    """
    import stat

    target = _final_path(path_or_workspace)
    # Open and inspect one file descriptor.  A separate ``lstat`` followed by
    # ``read_bytes`` permits a writable workspace process to swap the marker
    # for a symlink between the two operations.  O_NOFOLLOW (where available)
    # plus an fstat/read-size check binds the digest to the inode we inspected.
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(target), flags | nofollow)
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ENOTDIR}:
            reason = "missing"
        elif exc.errno == errno.ELOOP:
            reason = "non_regular"
        else:
            reason = "io_error"
        raise FinalPocRefused(
            f"final PoC is missing or cannot be opened: {target}", reason=reason
        ) from exc
    try:
        try:
            info = os.fstat(descriptor)
        except OSError as exc:
            raise FinalPocRefused(
                f"final PoC cannot be inspected: {target}", reason="io_error"
            ) from exc
        if not stat.S_ISREG(info.st_mode):
            raise FinalPocRefused(
                f"final.poc must be a regular non-symlink file: {target}",
                reason="non_regular",
            )
        if info.st_size <= 0:
            raise FinalPocRefused(
                f"final.poc must be non-empty: {target}", reason="empty"
            )
        if info.st_size > 10 * 1024 * 1024:
            raise FinalPocRefused(
                f"final.poc exceeds the CyberGym 10 MiB upload cap: {target}",
                reason="oversized",
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(info.st_size + 1)
        if len(raw) != info.st_size:
            raise FinalPocRefused(
                f"final.poc changed while it was being read: {target}",
                reason="changed",
            )
        try:
            after = os.fstat(descriptor)
        except OSError as exc:
            raise FinalPocRefused(
                f"final PoC cannot be re-inspected: {target}", reason="io_error"
            ) from exc
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
        ):
            raise FinalPocRefused(
                f"final.poc changed while it was being read: {target}",
                reason="changed",
            )
    except OSError as exc:
        raise FinalPocRefused(
            f"final PoC cannot be read: {target}", reason="io_error"
        ) from exc
    finally:
        os.close(descriptor)
    return FinalPoc(str(target.resolve(strict=False)), hashlib.sha256(raw).hexdigest(), len(raw))


def final_poc_hash(value: pathlib.Path | str | bytes | bytearray | memoryview) -> str:
    """Return a SHA-256 from bytes or a validated regular final marker."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return hashlib.sha256(bytes(value)).hexdigest()
    return final_poc_record(value).sha256


def official_pin_skip_reason(task_id: str) -> str:
    """Return the explicit official-pin skip reason, or empty if the task runs."""
    return str(OFFICIAL_PIN_SKIPS.get(safe_task_id(task_id), "") or "")


def build_task_result_row(
    task_id: str,
    *,
    trials: Sequence[Any] = (),
    final_trial: Any = None,
    final_poc: FinalPoc | Mapping[str, Any] | pathlib.Path | str | None = None,
    final_poc_sha256: str = "",
    status: str = "completed",
    lifecycle: str = "",
    capability_outcome: str = "",
    masked_id: str = "",
    masked_id_source: str = "",
    project: str = "",
    level: str = DEFAULT_LEVEL,
    observed_provider: str = "",
    observed_provider_attempts: Sequence[str] = (),
    observed_model: str = "",
    observed_effort: str = "",
    observed_effort_source: str = "",
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cached_tokens: int | None = None,
    cost_usd: float | None = None,
    cost_estimated: bool | None = None,
    cost_status: str = "",
    infra_reason: str = "",
    leakage: Any = None,
    artifact_refs: Mapping[str, Any] | None = None,
    error: str = "",
    runtime_result: Mapping[str, Any] | None = None,
    task_contract: Mapping[str, Any] | None = None,
    attempt_id: str = "",
) -> dict[str, Any]:
    """Build a denominator-preserving row through ``common.result_index``."""
    from devtools.benchmarks.common.result_index import task_result_row as common_row

    task = safe_task_id(task_id)
    normalized_attempt = str(attempt_id or "").strip()
    if normalized_attempt and not _SAFE_COMPONENT.fullmatch(normalized_attempt):
        raise ValueError("attempt_id must be a safe path component")
    if observed_effort:
        observed_effort = validate_high_effort(observed_effort, field="observed_effort")
    normalized = [_normalize_trial(item, index) for index, item in enumerate(trials)]
    selected = _choose_final(normalized, final_trial)
    if final_poc is not None and not final_poc_sha256:
        if isinstance(final_poc, FinalPoc):
            final_poc_sha256 = final_poc.sha256
        elif isinstance(final_poc, Mapping):
            final_poc_sha256 = str(final_poc.get("sha256", final_poc.get("poc_hash", "")))
        else:
            final_poc_sha256 = final_poc_hash(final_poc)
    projection = final_submission(selected, final_poc_sha256=final_poc_sha256, trials=normalized)
    if selected is not None and not any(item["trial_id"] == selected["trial_id"] for item in normalized):
        normalized.append(selected)
    marker_bound = bool(str(final_poc_sha256 or "").strip()) or final_poc is not None
    final_status = str(projection.get("final_submission_status") or "unknown")
    final_reason = str(projection.get("final_submission_reason") or "")
    final_hash = str(projection.get("final_poc_hash") or "").strip().lower()
    final_evidence = bool(
        marker_bound
        and selected is not None
        and final_status in {"known_success", "known_failure"}
        and bool(_HEX64.fullmatch(final_hash))
        and final_reason not in {"final_poc_hash_mismatch", "final_poc_hash_missing"}
    )
    effective_status = str(status or "").strip().lower()
    if not effective_status:
        effective_status = "completed"
    effective_lifecycle = lifecycle or effective_status
    effective_capability_outcome = str(capability_outcome or "").strip()
    if effective_capability_outcome and (
        effective_capability_outcome != CAPABILITY_FINAL_POC_MISSING
    ):
        raise ValueError("unknown capability_outcome")
    effective_infra_reason = str(infra_reason or "")
    effective_error = str(error or "")
    if effective_status == "failed" and not final_evidence:
        if effective_capability_outcome == CAPABILITY_FINAL_POC_MISSING:
            # A fair, terminal model task that produced no valid designated
            # submission is a denominator-preserving capability failure. The
            # official verifier did not run, so ``official_success`` remains
            # unknown, but the headline final-submission metric is false.
            projection["final_submission_success"] = False
            projection["final_submission_status"] = "known_failure"
            projection["final_submission_reason"] = effective_capability_outcome
            final_status = "known_failure"
            final_reason = effective_capability_outcome
        else:
            # Untyped failures may be provider, runtime, or adapter failures.
            # Keep them outside the capability denominator rather than
            # manufacturing a model zero from a generic status string.
            effective_status = "infra_failed"
            effective_lifecycle = "untyped_failure"
            effective_infra_reason = effective_infra_reason or "untyped_failure"
            effective_error = effective_error or (
                "failed result lacked a typed capability outcome"
            )
    elif effective_capability_outcome:
        raise ValueError("capability_outcome requires a failed result without final evidence")
    if effective_status == "completed" and not final_evidence:
        effective_status = "infra_failed"
        effective_lifecycle = "final_evidence_missing"
        effective_infra_reason = effective_infra_reason or "final_evidence_missing"
        effective_error = effective_error or (
            "completed result requires one regular final.poc and a bound final trial hash"
        )
    contract = dict(task_contract or {})
    if "effort" in contract:
        validate_high_effort(contract.get("effort"), field="task_contract.effort")
    project = project or task.split(":", 1)[0]
    refs = dict(artifact_refs or {})
    provider_attempts = [
        str(item).strip() for item in observed_provider_attempts if str(item).strip()
    ]
    if not provider_attempts and str(observed_provider or "").strip():
        provider_attempts = [str(observed_provider).strip()]
    provider_route = list(dict.fromkeys(provider_attempts))
    provider_distribution = {
        provider: provider_attempts.count(provider) for provider in provider_route
    }
    row = common_row(
        benchmark=BENCHMARK_NAME,
        instance_id=task,
        status=effective_status,
        runtime_result=runtime_result,
        prediction_written=final_evidence,
        official_eval_status=("completed" if final_evidence else "not_run"),
        output_paths=refs,
        reason_code=effective_infra_reason or final_reason,
        error=effective_error,
        details={
            "project": project,
            "level": level,
            "trials": [_compact_trial(item) for item in normalized],
            "leakage": leakage,
            "task_contract": contract,
            "attempt_id": normalized_attempt,
            "capability_outcome": effective_capability_outcome,
            "observed_provider_route": provider_route,
            "provider_distribution": provider_distribution,
        },
    )
    effective_masked_id = str(masked_id or "").strip()
    effective_masked_source = str(masked_id_source or "").strip()
    if not effective_masked_source:
        effective_masked_source = (
            "upstream_submit_response" if effective_masked_id else "local_digest_diagnostic"
        )
    row.update(
        {
            "adapter_schema": RESULT_SCHEMA,
            "task_id": task,
            "masked_id": effective_masked_id or mask_task_id(task),
            "masked_id_source": effective_masked_source,
            "project": project,
            "level": level,
            "trial_count": len(normalized),
            "lifecycle": effective_lifecycle,
            "final_poc_id": projection.get("final_poc_id", ""),
            "final_poc_hash": projection.get("final_poc_hash", str(final_poc_sha256 or "")),
            "raw_final_vul_exit": projection.get("raw_final_vul_exit"),
            "raw_final_fix_exit": projection.get("raw_final_fix_exit"),
            "official_success": projection.get("official_success"),
            "final_submission_success": projection.get("final_submission_success"),
            "any_of_success": projection.get("any_of_success"),
            "metric_name": "final_submission",
            "observed_provider": str(observed_provider or ""),
            "observed_provider_attempts": provider_attempts,
            "observed_provider_route": provider_route,
            "provider_distribution": provider_distribution,
            "observed_model": str(observed_model or ""),
            "observed_effort": str(observed_effort or ""),
            "observed_effort_source": str(observed_effort_source or ""),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "cost_usd": cost_usd,
            "cost_estimated": cost_estimated,
            "cost_status": str(cost_status or ("known" if cost_usd is not None else "unknown")),
            "infra_reason": effective_infra_reason,
            "leakage": leakage,
            "artifact_refs": refs,
            "final_submission_status": projection.get("final_submission_status", "unknown"),
            "final_submission_reason": projection.get("final_submission_reason", ""),
            "any_of_status": projection.get("any_of_status", "unknown"),
            "any_of_reason": projection.get("any_of_reason", ""),
            "task_contract": contract,
            "attempt_id": normalized_attempt,
            "capability_outcome": effective_capability_outcome,
        }
    )
    return row


def _money(value: Any, *, field: str, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise LedgerError(f"{field} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LedgerError(f"{field} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise LedgerError(f"{field} must be a finite non-negative number")
    return number


def _event_amount(event: Mapping[str, Any], *keys: str, allow_none: bool = False) -> float | None:
    for key in keys:
        if key in event:
            return _money(event[key], field=key, allow_none=allow_none)
    if allow_none:
        return None
    raise LedgerError(f"ledger event missing {keys[0]}")


def _numeric_unresolved_bound(
    explicit_upper: float | None,
    *,
    reserved_usd: float = 0.0,
    prior_upper: float | None = None,
) -> float:
    """Remaining liability of a finished attempt is 0.

    A missing bound must not fall back to the live claim estimate: that
    leftover UB reserved $20 × N corpses against the campaign cap. The legacy
    parameters stay in the signature so historical callers and replay helpers
    do not fork; they never become dispatch liability.
    """
    del explicit_upper, reserved_usd, prior_upper
    return 0.0


def _finished_attempt_actual_usd(outcome: Mapping[str, Any] | None) -> float | None:
    """Return a known actual for a finished attempt, or None.

    This is settled cash (or a measured terminal bound), never the per-task
    claim estimate. A finished/infra row without a number has remaining 0.
    """
    if not isinstance(outcome, Mapping):
        return None
    for key in ("cost_usd", "cost_upper_bound_usd"):
        if key not in outcome or outcome[key] is None:
            continue
        try:
            return _money(outcome[key], field=key)
        except LedgerError:
            return None
    return None


def _attempt_reservation_bound(
    events: Iterable[Mapping[str, Any]], attempt: str
) -> tuple[float, float | None]:
    """Latest reserved estimate and persisted unresolved bound for one attempt."""
    reserved = 0.0
    prior_upper: float | None = None
    for raw in events:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("attempt_id", raw.get("id", "")) or "").strip() != attempt:
            continue
        kind = str(raw.get("event", raw.get("kind", "")) or "").lower()
        if kind in {"claim", "reserve", "reserved"}:
            reserved = float(
                _event_amount(raw, "reserved_usd", "estimated_cost_usd", "amount_usd") or 0.0
            )
            prior_upper = None
        elif kind in {"unresolved", "unknown"}:
            prior = _event_amount(
                raw, "upper_bound_usd", "unresolved_upper_bound_usd", allow_none=True
            )
            if prior is not None:
                prior_upper = prior
                reserved = 0.0
        elif kind in {"settle", "settled", "overspend", "release", "released"}:
            reserved = 0.0
            prior_upper = None
    return reserved, prior_upper


_TERMINAL_GATEWAY_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "rejected_duplicate"}
)


def _terminal_gateway_accounting(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project a terminal gateway's total accounted bound for the outer ledger.

    The outer ledger cannot see the gateway's physical attempt rows, so a
    terminal task response contributes its total ``accounted_upper_bound_usd``
    (``cost_usd`` as the frozen alias); the inner ``unresolved_upper_bound_usd``
    residual alone is never sufficient.  Terminal payloads only: an
    intermediate/running snapshot must not authorize dispatch.
    """

    if not isinstance(payload, Mapping):
        return {}
    status = str(payload.get("status") or "").strip().lower()
    if status not in _TERMINAL_GATEWAY_STATUSES:
        return {}
    sources: list[Mapping[str, Any]] = []
    queue: list[Mapping[str, Any]] = [payload]
    seen: set[int] = set()
    for source in queue:
        marker = id(source)
        if marker in seen:
            continue
        seen.add(marker)
        sources.append(source)
        for child_key in (
            "result",
            "task_result",
            "runtime_result",
            "cost_breakdown",
        ):
            child = source.get(child_key)
            if isinstance(child, Mapping):
                queue.append(child)

    def first_value(*names: str) -> Any:
        for source in sources:
            for name in names:
                if name in source and source[name] is not None:
                    return source[name]
        return None

    total: float | None = None
    amount_conflict = False

    def amount_views(name: str) -> tuple[list[float], bool]:
        values: list[float] = []
        invalid = False
        for source in sources:
            if name not in source or source[name] is None:
                continue
            try:
                value = _money(source[name], field=name)
            except LedgerError:
                invalid = True
                continue
            if value is not None:
                values.append(value)
        return values, invalid

    totals, invalid_total = amount_views("accounted_upper_bound_usd")
    if not totals and not invalid_total:
        totals, invalid_total = amount_views("cost_usd")
    if totals:
        total = max(totals)
        amount_conflict = invalid_total or any(
            not math.isclose(value, total, rel_tol=1e-12, abs_tol=1e-12)
            for value in totals
        )
    elif invalid_total:
        amount_conflict = True
    projected: dict[str, Any] = {}
    if total is not None:
        projected.update({"cost_upper_bound_usd": total, "cost_usd": total})
    final_present = [source.get("cost_final") for source in sources if "cost_final" in source]
    final_markers = [value for value in final_present if isinstance(value, bool)]
    if amount_conflict or len(final_markers) != len(final_present) or False in final_markers:
        projected["cost_final"] = False
    elif final_markers and all(final_markers):
        projected["cost_final"] = True
    partial_present = [
        source.get("cost_with_children_partial")
        for source in sources
        if "cost_with_children_partial" in source
    ]
    partial_markers = [value for value in partial_present if isinstance(value, bool)]
    if len(partial_markers) != len(partial_present) or True in partial_markers:
        projected["cost_final"] = False
    estimated_present = [
        source.get("cost_estimated")
        for source in sources
        if "cost_estimated" in source
    ]
    estimated_markers = [value for value in estimated_present if isinstance(value, bool)]
    if len(estimated_markers) != len(estimated_present) or True in estimated_markers:
        projected["cost_estimated"] = True
    elif estimated_markers and not any(estimated_markers):
        projected["cost_estimated"] = False
    accounting_present = [
        source.get("cost_accounting_status")
        for source in sources
        if "cost_accounting_status" in source
    ]
    accounting_statuses = [
        value.strip().lower()
        for value in accounting_present
        if isinstance(value, str) and value.strip()
    ]
    if len(accounting_statuses) != len(accounting_present) or any(
        value != "available" for value in accounting_statuses
    ):
        projected["cost_final"] = False
    accounting_status = (
        accounting_statuses[0]
        if accounting_statuses
        else first_value("cost_status")
    )
    if isinstance(accounting_status, str) and accounting_status.strip():
        projected["cost_status"] = accounting_status.strip()
    return projected


def project_budget(
    events: Iterable[Mapping[str, Any]], cap_usd: float | None = DEFAULT_BUDGET_CAP_USD
) -> BudgetProjection:
    """Replay terminal state per attempt.

    Live in-flight reservations count. A finished/failed/infra attempt does
    not keep its claim estimate (or any leftover unresolved UB) as dispatch
    liability — historical jsonl must not poison replay. Dispatch fails only
    when settled cash plus live reserved exhausts the cap.
    """
    cap = _money(cap_usd, field="cap_usd", allow_none=True)
    if cap is not None and cap > DEFAULT_BUDGET_CAP_USD:
        raise BudgetRefused(
            f"cap_usd may not exceed the CyberGym hard cap of {DEFAULT_BUDGET_CAP_USD:.2f}"
        )
    latest: dict[str, dict[str, Any]] = {}
    for raw in events:
        if not isinstance(raw, Mapping):
            raise LedgerError("ledger event must be an object")
        event = dict(raw)
        kind = str(event.get("event", event.get("kind", "")) or "").lower()
        attempt = str(event.get("attempt_id", event.get("id", "")) or "").strip()
        if not attempt or not _SAFE_COMPONENT.fullmatch(attempt):
            raise LedgerError("ledger event has an unsafe or missing attempt_id")
        previous = latest.get(attempt)
        if kind in {"claim", "reserve", "reserved"}:
            if previous is not None:
                raise LedgerError(f"attempt has multiple claims: {attempt}")
            task = safe_task_id(str(event.get("task_id") or ""))
            amount = _event_amount(event, "reserved_usd", "estimated_cost_usd", "amount_usd")
            latest[attempt] = {"state": "reserved", "task_id": task, "reserved_usd": amount or 0.0}
        elif kind in {"campaign_cost", "overhead"}:
            # Campaign-level charges (currently the exact provider readiness
            # completion) have no task claim, but they are still settled spend
            # for the hard-cap projection.  They use a unique synthetic
            # attempt id and therefore cannot be mistaken for a task result.
            if previous is not None:
                raise LedgerError(f"campaign cost has multiple entries: {attempt}")
            cost = _event_amount(event, "cost_usd", "amount_usd")
            latest[attempt] = {
                "state": "settled",
                "task_id": "campaign:overhead",
                "cost_usd": cost or 0.0,
                "reserved_usd": 0.0,
                "upper_bound_usd": None,
                "overspend": bool(event.get("overspend")),
            }
        elif kind in {"settle", "settled", "overspend"}:
            if previous is None or previous.get("state") not in {"reserved", "unresolved"}:
                raise LedgerError(f"settlement has no active claim: {attempt}")
            cost = _event_amount(event, "cost_usd", "settled_usd", "amount_usd")
            latest[attempt] = {
                **previous,
                "state": "settled",
                "cost_usd": cost or 0.0,
                "reserved_usd": 0.0,
                "upper_bound_usd": None,
                "overspend": kind == "overspend",
            }
        elif kind in {"unresolved", "unknown"}:
            if previous is None or previous.get("state") not in {"reserved", "unresolved"}:
                raise LedgerError(f"unresolved event has no active claim: {attempt}")
            latest[attempt] = {
                **previous,
                "state": "unresolved",
                "reserved_usd": 0.0,
                "upper_bound_usd": 0.0,
            }
        elif kind in {"release", "released"}:
            if previous is None or previous.get("state") not in {"reserved", "unresolved"}:
                raise LedgerError(f"release has no active claim: {attempt}")
            latest[attempt] = {**previous, "state": "released", "reserved_usd": 0.0, "upper_bound_usd": None}
        else:
            raise LedgerError(f"unknown ledger event kind: {kind!r}")

    settled = sum(float(item.get("cost_usd") or 0.0) for item in latest.values() if item.get("state") == "settled")
    reserved = sum(float(item.get("reserved_usd") or 0.0) for item in latest.values() if item.get("state") == "reserved")
    unresolved = 0.0
    projected = settled + reserved
    if cap is None:
        available, can_dispatch, reason = None, True, "uncapped"
    else:
        available = cap - projected
        can_dispatch = available >= 0
        reason = "within_cap" if can_dispatch else "budget_cap_exceeded"
    active = {attempt: item for attempt, item in latest.items() if item.get("state") == "reserved"}
    return BudgetProjection(
        cap,
        settled,
        reserved,
        unresolved,
        projected,
        available,
        can_dispatch,
        reason,
        tuple(sorted(str(item["task_id"]) for item in active.values())),
        tuple(sorted(active)),
    )


class BudgetLedger:
    """Append-only atomic claim/settlement writer for one campaign."""

    def __init__(self, path: pathlib.Path | str, *, cap_usd: float | None = DEFAULT_BUDGET_CAP_USD) -> None:
        self.path = pathlib.Path(path).expanduser().resolve(strict=False)
        self.cap_usd = _money(cap_usd, field="cap_usd", allow_none=True)
        if self.cap_usd is not None and self.cap_usd > DEFAULT_BUDGET_CAP_USD:
            raise BudgetRefused(
                f"cap_usd may not exceed the CyberGym hard cap of {DEFAULT_BUDGET_CAP_USD:.2f}"
            )

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise LedgerError(f"cannot read ledger: {self.path}") from exc
        events: list[dict[str, Any]] = []
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(f"malformed ledger line {number}: {self.path}") from exc
            if not isinstance(value, dict):
                raise LedgerError(f"ledger line {number} is not an object")
            events.append(value)
        return events

    def projection(self) -> BudgetProjection:
        return project_budget(self.events(), self.cap_usd)

    @contextlib.contextmanager
    def _lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        handle = lock_path.open("a+", encoding="utf-8")
        locked = False
        try:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                locked = True
            except ImportError:
                # Windows callers still get append-only semantics; the platform's
                # atomic rename/open rules provide the narrow fallback available here.
                pass
            yield
        finally:
            if locked:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _append(self, event: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(event), ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def claim(self, task_id: str, estimated_cost_usd: float | None, *, attempt_id: str = "") -> dict[str, Any]:
        task = safe_task_id(task_id)
        estimate = _money(estimated_cost_usd, field="estimated_cost_usd", allow_none=True)
        if estimate is None:
            raise BudgetRefused("a finite estimate is required before paid dispatch")
        if estimate <= 0:
            raise BudgetRefused("estimated_cost_usd must be positive")
        attempt = str(attempt_id or uuid.uuid4().hex)
        if not _SAFE_COMPONENT.fullmatch(attempt):
            raise ValueError("attempt_id must be a safe path component")
        now = time.time()
        with self._lock():
            current = self.projection()
            if attempt in current.active_attempt_ids or any(
                str(item.get("attempt_id")) == attempt for item in self.events()
            ):
                raise ClaimRefused(f"attempt already exists: {attempt}")
            if task in current.active_task_ids:
                raise ClaimRefused(f"task already has an active claim: {task}")
            if not current.can_dispatch:
                raise BudgetRefused(f"campaign budget admission refused: {current.reason}")
            projected = current.projected_usd or 0.0
            if self.cap_usd is not None and projected + estimate > self.cap_usd:
                raise BudgetRefused("reservation would exceed campaign budget cap")
            self._append(
                {
                    "schema": LEDGER_SCHEMA,
                    "event": "claim",
                    "task_id": task,
                    "attempt_id": attempt,
                    "reserved_usd": estimate,
                    "ts_unix": now,
                }
            )
        return {"task_id": task, "attempt_id": attempt, "reserved_usd": estimate, "ts_unix": now}

    def settle(self, attempt_id: str, cost_usd: float) -> None:
        attempt = str(attempt_id or "").strip()
        cost = _money(cost_usd, field="cost_usd")
        with self._lock():
            current = self.projection()
            if attempt not in current.active_attempt_ids:
                raise LedgerError(f"attempt is not active: {attempt}")
            # Replace this attempt's reservation with its measured spend when
            # checking the hard cap.  Other unresolved attempts contribute their
            # numeric bound (claim estimate if the written bound was missing).
            reserved_for_attempt = 0.0
            for event in self.events():
                if str(event.get("attempt_id") or "") != attempt:
                    continue
                kind = str(event.get("event", event.get("kind", "")) or "").lower()
                if kind in {"claim", "reserve", "reserved"}:
                    reserved_for_attempt = float(
                        _event_amount(
                            event,
                            "reserved_usd",
                            "estimated_cost_usd",
                            "amount_usd",
                        )
                        or 0.0
                    )
                elif kind in {"settle", "settled", "overspend", "release", "released"}:
                    reserved_for_attempt = 0.0
            projected_after = None
            if current.projected_usd is not None:
                projected_after = current.projected_usd - reserved_for_attempt + float(cost or 0.0)
            if self.cap_usd is not None and projected_after is not None and projected_after > self.cap_usd:
                self._append(
                    {
                        "schema": LEDGER_SCHEMA,
                        "event": "overspend",
                        "attempt_id": attempt,
                        "cost_usd": cost,
                        "ts_unix": time.time(),
                    }
                )
                raise BudgetOverspend(
                    "measured settlement exceeds campaign budget cap: "
                    f"projected={projected_after:.6f}, cap={self.cap_usd:.6f}"
                )
            self._append(
                {
                    "schema": LEDGER_SCHEMA,
                    "event": "settle",
                    "attempt_id": attempt,
                    "cost_usd": cost,
                    "ts_unix": time.time(),
                }
            )

    def record_campaign_cost(self, cost_usd: float, *, label: str = "provider_probe") -> dict[str, Any]:
        """Record a known campaign-level charge before task reservations.

        The provider readiness completion is real spend even though it has no
        task claim.  Keeping it as an append-only settled event makes the
        campaign projection and hard stop include that charge without
        inventing a per-task price.  A deterministic label is idempotent for a
        repeated ``prepare`` call in the same run root.
        """
        cost = _money(cost_usd, field="campaign_cost_usd")
        if cost is None:
            raise BudgetRefused("campaign cost must be known before dispatch")
        label_text = str(label or "").strip()
        if not label_text or not _SAFE_COMPONENT.fullmatch(label_text):
            raise LedgerError("campaign cost label is unsafe")
        attempt = "campaign-overhead-" + label_text
        with self._lock():
            events = self.events()
            existing = [
                item for item in events
                if str(item.get("attempt_id") or "") == attempt
                and str(item.get("event", item.get("kind", "")) or "").lower()
                in {"campaign_cost", "overhead"}
            ]
            if existing:
                previous = _event_amount(existing[-1], "cost_usd", "amount_usd")
                if previous != cost:
                    raise LedgerError("campaign cost label was recorded with a different amount")
                return dict(existing[-1])
            current = self.projection()
            projected_after = (
                None
                if current.projected_usd is None
                else current.projected_usd + float(cost)
            )
            overspend = self.cap_usd is not None and projected_after is not None and projected_after > self.cap_usd
            event = {
                "schema": LEDGER_SCHEMA,
                "event": "campaign_cost",
                "attempt_id": attempt,
                "label": label_text,
                "cost_usd": cost,
                "overspend": bool(overspend),
                "ts_unix": time.time(),
            }
            self._append(event)
            if overspend:
                raise BudgetOverspend(
                    "campaign-level cost exceeds the hard cap: "
                    f"projected={projected_after:.6f}, cap={self.cap_usd:.6f}"
                )
            return event

    def mark_unresolved(self, attempt_id: str, upper_bound_usd: float | None = None) -> None:
        attempt = str(attempt_id or "").strip()
        upper = _money(upper_bound_usd, field="upper_bound_usd", allow_none=True)
        with self._lock():
            events = self.events()
            if attempt not in project_budget(events, self.cap_usd).active_attempt_ids:
                raise LedgerError(f"attempt is not active: {attempt}")
            reserved, prior_upper = _attempt_reservation_bound(events, attempt)
            bound = _numeric_unresolved_bound(upper, reserved_usd=reserved, prior_upper=prior_upper)
            self._append(
                {
                    "schema": LEDGER_SCHEMA,
                    "event": "unresolved",
                    "attempt_id": attempt,
                    "upper_bound_usd": bound,
                    "ts_unix": time.time(),
                }
            )

    def release(self, attempt_id: str) -> None:
        attempt = str(attempt_id or "").strip()
        with self._lock():
            if attempt not in self.projection().active_attempt_ids:
                raise LedgerError(f"attempt is not active: {attempt}")
            self._append({"schema": LEDGER_SCHEMA, "event": "release", "attempt_id": attempt, "ts_unix": time.time()})


def _append_result_pair(root: pathlib.Path, row: Mapping[str, Any]) -> None:
    """Append one row to the common run index and its task-local index.

    No locking here: callers either hold ``.result_index.lock`` across a wider
    check+append sequence (the reconcile arm) or go through
    ``append_cybergym_result``, which takes the lock for the pair.
    """
    from devtools.benchmarks.common.result_index import append_result_index

    task = safe_task_id(str(row.get("task_id", row.get("instance_id", ""))))
    value = dict(row)
    append_result_index(root, value)
    append_result_index(safe_task_path(root, task), value)


def append_cybergym_result(run_root: pathlib.Path | str, row: Mapping[str, Any]) -> None:
    """Append one row to the common run index and its task-local index."""
    root = pathlib.Path(run_root).expanduser().resolve(strict=False)
    # The shared helper deliberately stays a tiny append primitive and does not
    # own a cross-process lock.  A campaign can have several lanes, so serialize
    # the paired parent/task writes here and fsync the lock holder before release.
    lock_path = root / ".result_index.lock"
    root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        locked = False
        try:
            try:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                locked = True
            except ImportError:
                pass
            _append_result_pair(root, row)
            lock.flush()
            os.fsync(lock.fileno())
        finally:
            if locked:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _task_spec(value: TaskSpec | Mapping[str, Any] | str) -> TaskSpec:
    """Normalize one injected task value without touching the filesystem."""
    if isinstance(value, TaskSpec):
        task_id = safe_task_id(value.task_id)
        if value.level != DEFAULT_LEVEL:
            raise ValueError("CyberGym task contract requires level1")
        return dataclasses.replace(value, task_id=task_id)
    if isinstance(value, Mapping):
        task_id = safe_task_id(str(value.get("task_id", value.get("id", ""))))
        level = str(value.get("level") or DEFAULT_LEVEL)
        if level != DEFAULT_LEVEL:
            raise ValueError("CyberGym task contract requires level1")
        metadata = dict(value)
        return TaskSpec(
            task_id,
            str(value.get("project") or task_id.split(":", 1)[0]),
            level,
            metadata,
        )
    task_id = safe_task_id(str(value))
    return TaskSpec(task_id, task_id.split(":", 1)[0])


def finalize_outcome_row(
    root: pathlib.Path,
    task: TaskSpec,
    task_dir: pathlib.Path,
    outcome: dict[str, Any],
    *,
    attempt_id: str,
    contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the denominator-preserving row for one finished attempt outcome.

    Shared by ``run_campaign`` and the launcher's reconcile mode so a
    redelivered terminal result is projected, validated, and rendered exactly
    like a live one.  ``outcome`` is updated in place with the terminal
    gateway accounting projection.  This function does not append the row or
    settle the claim; callers own both side effects.
    """
    # A terminal gateway result is the only authoritative source for an
    # outer-ledger bound when the executor stopped during custody.  Project
    # its TOTAL accounted amount before building the row and settling the
    # claim; the inner unresolved remainder alone would omit already-settled
    # gateway usage.
    terminal_accounting = _terminal_gateway_accounting(outcome.get("runtime_result"))
    if terminal_accounting:
        outcome.update(terminal_accounting)
    requested_status = str(outcome.get("status") or "completed").strip().lower()
    raw_cost_estimated = outcome.get("cost_estimated")
    if raw_cost_estimated not in (None, False, True):
        raise LedgerError("cost_estimated must be a boolean")
    raw_cost_final = outcome.get("cost_final")
    if raw_cost_final not in (None, False, True):
        raise LedgerError("cost_final must be a boolean")
    cost_unverifiable = (
        raw_cost_estimated is True
        or outcome.get("cost_usd") is None
        or raw_cost_final is not True
    )
    if requested_status == "completed" and cost_unverifiable:
        # Lazy import: the wire layer imports this module at its top level.
        from devtools.benchmarks.cybergym.cybergym_wire import _status_after_cost_check

        requested_status = _status_after_cost_check(outcome, requested_status)
    if requested_status == "completed":
        observed_effort = validate_high_effort(
            outcome.get("observed_effort"), field="observed_effort"
        )
    else:
        observed_effort = str(outcome.get("observed_effort") or "")
    final_poc = outcome.get("final_poc")
    marker_record: FinalPoc | None = None
    if requested_status == "completed":
        marker_record = final_poc_record(task_dir)
        declared_hash = str(outcome.get("final_poc_sha256") or "").strip().lower()
        if declared_hash and declared_hash != marker_record.sha256:
            raise FinalPocRefused("executor final_poc_sha256 does not match final.poc")
        if final_poc is not None:
            if isinstance(final_poc, FinalPoc):
                supplied_hash = final_poc.sha256
            elif isinstance(final_poc, Mapping):
                supplied_hash = str(final_poc.get("sha256", final_poc.get("poc_hash", "")))
            else:
                supplied_hash = final_poc_hash(final_poc)
            if supplied_hash and supplied_hash.strip().lower() != marker_record.sha256:
                raise FinalPocRefused("executor final_poc does not match final.poc")
        final_poc = marker_record
    elif final_poc is None and (task_dir / FINAL_POC_BASENAME).exists():
        marker_record = final_poc_record(task_dir)
        final_poc = marker_record
    row = build_task_result_row(
        task.task_id,
        trials=outcome.get("trials") or (),
        final_trial=outcome.get("final_trial"),
        final_poc=final_poc,
        status=requested_status,
        lifecycle=str(outcome.get("lifecycle") or "completed"),
        capability_outcome=str(outcome.get("capability_outcome") or ""),
        level=task.level,
        masked_id=str(outcome.get("masked_id") or ""),
        masked_id_source=str(outcome.get("masked_id_source") or ""),
        observed_provider=str(outcome.get("observed_provider") or ""),
        observed_provider_attempts=outcome.get("observed_provider_attempts") or (),
        observed_model=str(outcome.get("observed_model") or ""),
        observed_effort=observed_effort,
        observed_effort_source=str(outcome.get("observed_effort_source") or ""),
        prompt_tokens=outcome.get("prompt_tokens"),
        completion_tokens=outcome.get("completion_tokens"),
        cached_tokens=outcome.get("cached_tokens"),
        cost_usd=outcome.get("cost_usd"),
        cost_estimated=outcome.get("cost_estimated"),
        cost_status=str(outcome.get("cost_status") or ""),
        infra_reason=str(outcome.get("infra_reason") or ""),
        leakage=outcome.get("leakage"),
        artifact_refs=outcome.get("artifact_refs") or {"task_dir": str(task_dir)},
        error=str(outcome.get("error") or ""),
        runtime_result=outcome.get("runtime_result"),
        task_contract=contract,
        attempt_id=str(attempt_id),
    )
    grace = outcome.get("cost_grace_acceptance")
    if isinstance(grace, Mapping) and grace.get("unresolved_upper_bound_usd") is not None:
        # The accounted upper bound already contains the abandoned residue;
        # the row discloses it instead of claiming a fully final cost.
        row.update({"cost_final": False, "cost_grace_acceptance": grace,
                    "unresolved_upper_bound_usd": grace["unresolved_upper_bound_usd"]})
    return row


def settle_finished_attempt(
    ledger: "BudgetLedger", attempt_id: str, outcome: Mapping[str, Any]
) -> None:
    """Settle a finished attempt's claim from its outcome, or mark it unresolved."""
    actual = _finished_attempt_actual_usd(outcome)
    if actual is not None:
        ledger.settle(str(attempt_id), actual)
    else:
        ledger.mark_unresolved(str(attempt_id), 0.0)


def run_campaign(
    tasks: Sequence[TaskSpec | Mapping[str, Any] | str],
    *,
    run_root: pathlib.Path | str,
    executor: Callable[[TaskSpec, pathlib.Path], Mapping[str, Any]] | None,
    estimated_cost_usd: float | None,
    budget_cap_usd: float | None = DEFAULT_BUDGET_CAP_USD,
    max_workers: int = 1,
    allow_retries: bool = False,
    gateway_circuit_threshold: int = GATEWAY_CIRCUIT_BREAKER_THRESHOLD,
) -> list[dict[str, Any]]:
    """Run injected task callbacks under one atomic ledger.

    The callback owns task generation, sidecar lifecycle, model transport, and
    process custody.  A missing callback is an explicit blocked result; this
    seam never falls back to Docker, a shell, or a host network.  A run of
    ``gateway_circuit_threshold`` consecutive transport-class failures opens
    the dispatch circuit breaker: admission stops, in-flight tasks settle,
    and ``GatewayCircuitOpen`` carries the landed rows and undispatched ids.
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or not 1 <= max_workers <= MAX_CROSS_TASK_WORKERS:
        raise ValueError(
            f"max_workers must be an integer in the range 1..{MAX_CROSS_TASK_WORKERS}"
        )
    if (
        isinstance(gateway_circuit_threshold, bool)
        or not isinstance(gateway_circuit_threshold, int)
        or gateway_circuit_threshold < 1
    ):
        raise ValueError("gateway_circuit_threshold must be a positive integer")
    root = pathlib.Path(run_root).expanduser().resolve(strict=False)
    normalized_tasks: list[TaskSpec] = []
    seen_task_ids: set[str] = set()
    for item in tasks:
        task = _task_spec(item)
        if task.task_id in seen_task_ids:
            raise ValueError(f"duplicate task id: {task.task_id}")
        seen_task_ids.add(task.task_id)
        normalized_tasks.append(task)
    ledger = BudgetLedger(root / "claims.jsonl", cap_usd=budget_cap_usd)
    if not isinstance(allow_retries, bool):
        raise ValueError("allow_retries must be a boolean")
    if not allow_retries:
        # A second invocation against the same campaign root is ambiguous: it
        # could overwrite a completed row or attach a late result to the wrong
        # attempt.  Callers that intentionally resume must opt in explicitly;
        # each resumed claim receives a fresh attempt id below.
        claimed_tasks = {
            safe_task_id(str(event.get("task_id") or ""))
            for event in ledger.events()
            if str(event.get("event", event.get("kind", "")) or "").lower()
            in {"claim", "reserve", "reserved"}
        }
        recorded_tasks: set[str] = set()
        index_path = root / "result_index.jsonl"
        if index_path.exists():
            try:
                for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise LedgerError(f"result index line {line_number} is not an object")
                    raw_task = value.get("task_id", value.get("instance_id", ""))
                    if raw_task:
                        recorded_tasks.add(safe_task_id(str(raw_task)))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LedgerError(f"cannot inspect existing result index: {index_path}") from exc
        repeated = sorted((claimed_tasks | recorded_tasks).intersection(seen_task_ids))
        if repeated:
            raise ClaimRefused(
                "task already has campaign history; pass allow_retries=True: "
                + ", ".join(repeated)
            )

    def _run_one(task: TaskSpec) -> dict[str, Any]:
        contract = task.metadata.get("task_contract") if isinstance(task.metadata, Mapping) else None
        task_dir = safe_task_path(root, task.task_id)
        skip_reason = official_pin_skip_reason(task.task_id)
        if skip_reason:
            task_dir.mkdir(parents=True, exist_ok=True)
            row = build_task_result_row(
                task.task_id,
                status="infra_failed",
                lifecycle=skip_reason,
                level=task.level,
                infra_reason=skip_reason,
                artifact_refs={"task_dir": str(task_dir)},
                error="official pin skipped: " + skip_reason,
                task_contract=contract if isinstance(contract, Mapping) else None,
            )
            append_cybergym_result(root, row)
            return row
        if executor is None:
            task_dir.mkdir(parents=True, exist_ok=True)
            row = build_task_result_row(
                task.task_id,
                status="blocked",
                lifecycle="integration_unavailable",
                level=task.level,
                infra_reason="executor_not_injected",
                artifact_refs={"task_dir": str(task_dir)},
                error="CyberGym executor is not configured",
                task_contract=contract if isinstance(contract, Mapping) else None,
            )
            append_cybergym_result(root, row)
            return row

        claim: Mapping[str, Any] | None = None
        outcome: dict[str, Any] = {}
        callback_contract: Mapping[str, Any] | None = None
        try:
            claim = ledger.claim(task.task_id, estimated_cost_usd)
            # Claim first, then create the workspace.  The persisted attempt id
            # is part of the immutable task value so sidecar agent identities,
            # checkpoints, and late results all refer to the same claim.
            attempt_id = str(claim["attempt_id"])
            callback_metadata = dict(task.metadata)
            callback_metadata["attempt_id"] = attempt_id
            if isinstance(contract, Mapping):
                callback_contract = dict(contract)
                callback_contract["attempt_id"] = attempt_id
                callback_metadata["task_contract"] = callback_contract
            else:
                callback_contract = None
            # Retried attempts receive an isolated child directory, so a stale
            # final.poc from an earlier attempt cannot satisfy the new claim.
            if allow_retries:
                task_dir = safe_task_path(root, task.task_id, attempt_id)
            task_dir.mkdir(parents=True, exist_ok=True)
            callback_task = dataclasses.replace(task, metadata=callback_metadata)
            result = executor(callback_task, task_dir)
            if not isinstance(result, Mapping):
                raise CyberGymIntegrationUnavailable("CyberGym executor must return a mapping")
            outcome = dict(result)
            row = finalize_outcome_row(
                root,
                task,
                task_dir,
                outcome,
                attempt_id=str(claim["attempt_id"]),
                contract=callback_contract
                if callback_contract is not None
                else (contract if isinstance(contract, Mapping) else None),
            )
            settle_finished_attempt(ledger, str(claim["attempt_id"]), outcome)
        except BudgetOverspend as exc:
            budget_refs = dict(outcome.get("artifact_refs") or {})
            budget_refs.setdefault("task_dir", str(task_dir))
            budget_refs.setdefault("claims", str(ledger.path))
            if claim is not None:
                budget_refs.setdefault(
                    "checkpoint",
                    str(
                        safe_task_path(
                            root / "checkpoints",
                            task.task_id,
                            str(claim["attempt_id"]),
                        )
                        / "gateway_checkpoint.json"
                    ),
                )
            budget_refs.setdefault("custody_pending", str(root / "custody_pending.json"))
            row = build_task_result_row(
                task.task_id,
                trials=outcome.get("trials") or (),
                final_trial=outcome.get("final_trial"),
                final_poc_sha256=str(outcome.get("final_poc_sha256") or ""),
                status="infra_failed",
                lifecycle="budget_refused",
                level=task.level,
                masked_id=str(outcome.get("masked_id") or ""),
                masked_id_source=str(outcome.get("masked_id_source") or ""),
                observed_provider=str(outcome.get("observed_provider") or ""),
                observed_model=str(outcome.get("observed_model") or ""),
                observed_effort=(
                    str(outcome.get("observed_effort") or "")
                    if str(outcome.get("observed_effort") or "").strip().lower() == "high"
                    else ""
                ),
                observed_effort_source=str(outcome.get("observed_effort_source") or ""),
                prompt_tokens=outcome.get("prompt_tokens"),
                completion_tokens=outcome.get("completion_tokens"),
                cached_tokens=outcome.get("cached_tokens"),
                cost_usd=outcome.get("cost_usd"),
                cost_estimated=outcome.get("cost_estimated"),
                cost_status=str(outcome.get("cost_status") or ""),
                infra_reason="budget_overspend",
                artifact_refs=budget_refs,
                error=str(exc),
                runtime_result=outcome.get("runtime_result"),
                task_contract=callback_contract
                if callback_contract is not None
                else (contract if isinstance(contract, Mapping) else None),
                attempt_id=str(claim["attempt_id"]) if claim else "",
            )
        except Exception as exc:
            settlement_overspend: BudgetOverspend | None = None
            if claim is not None:
                terminal_accounting = _terminal_gateway_accounting(
                    outcome.get("runtime_result")
                )
                if terminal_accounting:
                    outcome.update(terminal_accounting)
                try:
                    exact_cost = (
                        _money(outcome.get("cost_usd"), field="cost_usd")
                        if outcome.get("cost_usd") is not None
                        else None
                    )
                except LedgerError:
                    exact_cost = None
                try:
                    actual = exact_cost
                    if actual is None:
                        actual = _finished_attempt_actual_usd(outcome)
                    if actual is not None:
                        ledger.settle(str(claim["attempt_id"]), actual)
                    else:
                        ledger.mark_unresolved(str(claim["attempt_id"]), 0.0)
                except BudgetOverspend as settlement_exc:
                    settlement_overspend = settlement_exc
            failure_refs = dict(outcome.get("artifact_refs") or {})
            failure_refs.setdefault("task_dir", str(task_dir))
            failure_refs.setdefault("claims", str(ledger.path))
            if claim is not None:
                failure_refs.setdefault(
                    "checkpoint",
                    str(
                        safe_task_path(
                            root / "checkpoints",
                            task.task_id,
                            str(claim["attempt_id"]),
                        )
                        / "gateway_checkpoint.json"
                    ),
                )
            failure_refs.setdefault("custody_pending", str(root / "custody_pending.json"))
            row = build_task_result_row(
                task.task_id,
                trials=outcome.get("trials") or (),
                final_trial=outcome.get("final_trial"),
                final_poc_sha256=str(outcome.get("final_poc_sha256") or ""),
                status="infra_failed",
                lifecycle=(
                    "budget_refused" if settlement_overspend else "executor_failed"
                ),
                level=task.level,
                masked_id=str(outcome.get("masked_id") or ""),
                masked_id_source=str(outcome.get("masked_id_source") or ""),
                observed_provider=str(outcome.get("observed_provider") or ""),
                observed_model=str(outcome.get("observed_model") or ""),
                observed_effort=(
                    str(outcome.get("observed_effort") or "")
                    if str(outcome.get("observed_effort") or "").strip().lower() == "high"
                    else ""
                ),
                observed_effort_source=str(outcome.get("observed_effort_source") or ""),
                prompt_tokens=outcome.get("prompt_tokens"),
                completion_tokens=outcome.get("completion_tokens"),
                cached_tokens=outcome.get("cached_tokens"),
                cost_usd=outcome.get("cost_usd"),
                cost_estimated=outcome.get("cost_estimated"),
                cost_status=str(outcome.get("cost_status") or ""),
                infra_reason=(
                    "budget_overspend"
                    if settlement_overspend
                    else type(exc).__name__
                ),
                artifact_refs=failure_refs,
                error=str(settlement_overspend or exc),
                runtime_result=outcome.get("runtime_result"),
                task_contract=callback_contract
                if callback_contract is not None
                else (contract if isinstance(contract, Mapping) else None),
                attempt_id=str(claim["attempt_id"]) if claim else "",
            )
        append_cybergym_result(root, row)
        return row

    # A campaign may fan out independent tasks, but each task remains a
    # single-agent/no-swarm attempt.  The ledger and result writer are locked;
    # callers should choose the worker count from the measured pilot rather
    # than treating this as an unbounded scheduler.  The dispatch engine stops
    # admitting new work once the isolate gateway is proven unreachable, so a
    # dead gateway cannot burn the rest of the catalog into transport rows.
    return run_dispatched(
        normalized_tasks,
        _run_one,
        max_workers=max_workers,
        threshold=gateway_circuit_threshold,
    )
