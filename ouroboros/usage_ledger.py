"""Money substrate vocabulary: errors, row rules, the drive root and the name lock.

The monetary authority is the usage store (``ouroboros/usage_store.py``,
``state/usage.sqlite``; docs/USAGE_STORE.md). This leaf keeps what every money
module shares and nothing that reads history: the typed errors, the per-row
structural and transition rules (applied by the store on every write and by
the one-time journal import, ``usage_journal``), the drive-root resolver, and
the name-protocol lock that serializes every store access on installations
without kernel file locks (the store's ``name`` lock tier). It has no opinion
about reservations, budgets, pricing or projections; those modules import FROM
here and are never imported BY here.
"""

from __future__ import annotations

import contextlib
import errno
import math
import os
import pathlib
import re
from typing import Any, Callable, Dict, Iterator, Optional, Sequence

from ouroboros._usage_money import amount

LEDGER_REL = pathlib.Path("state/usage_attempts.jsonl")  # the retired journal (import/export only)
QUARANTINE_REL = pathlib.Path("state/usage_attempts.quarantine.jsonl")
LOCK_REL = pathlib.Path("state/usage_attempts.lock")  # the name-tier money lock
# The ONE directory a retained journal baseline header may name.
ARCHIVE_SEGMENT_DIR_REL = pathlib.Path("archive/usage_ledger")
_ARCHIVE_SEGMENT_PREFIX = ARCHIVE_SEGMENT_DIR_REL.as_posix() + "/"
_TERMINAL = frozenset({"settled", "unresolved", "released"})
ONE_SHOT_KINDS = frozenset({"external_unmetered", "subscription_session"})

__all__ = (
    "LEDGER_REL", "LOCK_REL", "QUARANTINE_REL", "UsageAccountingError", "UsageLedgerCorrupt",
    "UsageLockUnavailable", "UsageNonFiniteMoney", "is_abandoned_settlement",
)


class UsageAccountingError(RuntimeError):
    """Base error for fail-closed accounting operations."""


class UsageLedgerCorrupt(UsageAccountingError):
    """Raised when durable history is structurally invalid."""


class UsageNonFiniteMoney(UsageAccountingError):
    """Nonfinite monetary evidence: preserve bytes, never quarantine or zero it."""


class UsageLockUnavailable(UsageAccountingError):
    """A named monetary lock was not acquired: the caller's timeout ran out, or
    the platform refused the lock outright.

    Distinct from corruption and validation failures: a display read reports
    the fact unavailable after its short wait; platform refusal and unknown
    failures propagate; every monetary caller fails closed.
    """

    def __init__(self, message: str, *, reason: str = "unknown", error_number: int | None = None):
        super().__init__(message)
        self.reason = reason
        self.error_number = error_number


def is_abandoned_settlement(row: Dict[str, Any]) -> bool:
    """An administratively closed attempt whose actual price is still unknown."""
    return (
        str(row.get("kind") or "attempt") == "attempt"
        and row.get("state") == "settled"
        and row.get("settle_reason") == "abandoned"
        and row.get("cost_usd") is None
        and row.get("cost_final") is False
    )


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_IDENTITY_FIELDS = (
    "candidate_raw_sha256", "candidate_raw_size_bytes",
    "candidate_context_sha256", "candidate_context_size_bytes",
)


def _validate_candidate_facts(row: Dict[str, Any], sequence: int) -> None:
    if "candidate_payload" in row:
        raise UsageLedgerCorrupt(f"mutable candidate payload in usage row seq={sequence}")
    present = "candidate_measurement_kind" in row or any(key in row for key in _CANDIDATE_IDENTITY_FIELDS)
    if not present:
        return  # pre-feature/legacy rows
    kind = row.get("candidate_measurement_kind")
    if kind not in {"canonical_json_v1", "opaque"}:
        raise UsageLedgerCorrupt(f"invalid candidate_measurement_kind in usage row seq={sequence}")
    if kind == "opaque":
        if any(row.get(key) is not None for key in _CANDIDATE_IDENTITY_FIELDS):
            raise UsageLedgerCorrupt(f"opaque candidate claims identity in usage row seq={sequence}")
    else:
        for key in ("candidate_raw_sha256", "candidate_context_sha256"):
            if not isinstance(row.get(key), str) or not _SHA256_RE.fullmatch(row[key]):
                raise UsageLedgerCorrupt(f"invalid {key} in usage row seq={sequence}")
        for key in ("candidate_raw_size_bytes", "candidate_context_size_bytes"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise UsageLedgerCorrupt(f"invalid {key} in usage row seq={sequence}")
    context = row.get("physical_context")
    if context is not None:
        if not isinstance(context, dict):
            raise UsageLedgerCorrupt(f"invalid physical_context in usage row seq={sequence}")
        if context.get("profile") not in {"owner_max", "owner_low", "owner_nano", "task_local_low", "task_local_nano"}:
            raise UsageLedgerCorrupt(f"invalid physical_context profile in usage row seq={sequence}")
        if context.get("rendered_mode") not in {"max", "low", "nano"} or context.get("measurement_basis") not in {
            "fresh_route_usage", "fresh_model_usage", "cold_estimate",
        }:
            raise UsageLedgerCorrupt(f"invalid physical_context mode/basis in usage row seq={sequence}")
        for key in ("target_total_tokens", "capacity_total_tokens"):
            value = context.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise UsageLedgerCorrupt(f"invalid physical_context {key} in usage row seq={sequence}")
        if not all(isinstance(context.get(key), bool) for key in ("context_target_miss", "automatic_pass_used")):
            raise UsageLedgerCorrupt(f"invalid physical_context flags in usage row seq={sequence}")
        if not all(isinstance(context.get(key), str) for key in ("route_fp", "round_id")):
            raise UsageLedgerCorrupt(f"invalid physical_context identity in usage row seq={sequence}")
        density = context.get("measurement_density")  # absent on rows written before the field existed
        if density is not None and (isinstance(density, bool) or not isinstance(density, (int, float))
                                    or not math.isfinite(density) or density <= 0):
            raise UsageLedgerCorrupt(f"invalid physical_context measurement_density in usage row seq={sequence}")
    manifest_ref = row.get("candidate_manifest_ref")
    if manifest_ref is not None and (
        not isinstance(manifest_ref, dict)
        or manifest_ref.get("call_id") != row.get("attempt_id")
        or not _SHA256_RE.fullmatch(str(manifest_ref.get("sha256") or ""))
    ):
        raise UsageLedgerCorrupt(f"invalid candidate_manifest_ref in usage row seq={sequence}")


def valid_archive_rel(value: Any) -> bool:
    """Whether ``archive_rel`` names a segment INSIDE the archive directory.

    A baseline header is the only ledger row that points at bytes outside its
    own file, so the reference is bounded HERE, once, instead of at each
    reader: relative, forward-slash, exactly ``archive/usage_ledger/<name>``,
    no traversal, no drive letter, no separator inside the name. An absolute
    path or a ``..`` hop cannot satisfy the prefix, so a tampered header can
    never point a reader at a file elsewhere on the host and have its
    ``attempt_id``s counted as archived history.
    """
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    if not value.startswith(_ARCHIVE_SEGMENT_PREFIX):
        return False
    name = value[len(_ARCHIVE_SEGMENT_PREFIX):]
    return bool(name) and name not in {".", ".."} and "/" not in name

def _drive_root(value: pathlib.Path | str | None = None) -> pathlib.Path:
    if value is not None:
        if not isinstance(value, (str, pathlib.Path)):
            raise UsageAccountingError(f"invalid usage accounting drive root type: {type(value).__name__}")
        resolved = pathlib.Path(value)
        if not resolved.is_absolute():
            raise UsageAccountingError(f"usage accounting drive root must be absolute: {resolved}")
        return resolved
    configured = str(os.environ.get("OUROBOROS_DATA_DIR") or "").strip()
    if configured:
        resolved = pathlib.Path(configured)
        if not resolved.is_absolute():
            raise UsageAccountingError(f"OUROBOROS_DATA_DIR must be absolute for usage accounting: {resolved}")
        return resolved
    from ouroboros.config import DATA_DIR

    return pathlib.Path(DATA_DIR)


@contextlib.contextmanager
def _named_lock(
    root: pathlib.Path,
    filename: str,
    *,
    timeout_sec: float,
    stale_sec: float,
) -> Iterator[Callable[[], bool]]:
    """Hold a named monetary lock; yields a heartbeat that renews its age.

    Acquisition is OWNER-AWARE: elapsed time alone never evicts a lock whose
    writing process is still alive, because a stolen monetary lock means two
    writers rewriting the same authority.  The yielded callable additionally
    keeps the lockfile young for acquirers that are not owner-aware (an older
    build, a foreign helper), and is a no-op cost for holds that never
    approach ``stale_sec``.
    """
    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock,
        refresh_exclusive_file_lock,
        release_exclusive_file_lock,
    )

    path = root / "state" / filename
    outcome: dict = {}
    fd = acquire_exclusive_file_lock(  # ENOLCK is the name tier for ordinary locks; money never runs there
        path, timeout_sec=timeout_sec, stale_sec=stale_sec, owner_aware_stale=True,
        refuse_name_tier_errnos=frozenset({errno.ENOLCK}), outcome=outcome,
    )
    if fd is None:
        raise UsageLockUnavailable(f"usage accounting lock unavailable: {path}",
                                   reason=outcome.get("reason", "unknown"), error_number=outcome.get("errno"))
    try:
        yield lambda: refresh_exclusive_file_lock(path, fd)
    finally:
        release_exclusive_file_lock(path, fd)


USAGE_LOCK_TIMEOUT_SEC = 45.0


@contextlib.contextmanager
def _locked(root: pathlib.Path, *, timeout_sec: float = USAGE_LOCK_TIMEOUT_SEC) -> Iterator[Callable[[], bool]]:
    # Bounded maintenance and post-response custody. Task-owned pre-send policy
    # supplies short acquisition slices; it never retries the transaction body.
    # Waits at most USAGE_LOCK_TIMEOUT_SEC; a lock with a recorded live owner is
    # never evicted by age, and the 90 s stale age applies to ownerless locks.
    with _named_lock(root, LOCK_REL.name, timeout_sec=timeout_sec, stale_sec=90.0) as heartbeat:
        yield heartbeat



def is_one_shot_kind(kind: str) -> bool:
    """Single-row kinds: the first and only row of their attempt id."""
    return kind.startswith("legacy_") or kind in ONE_SHOT_KINDS


def late_receipt_eligible(row: Optional[Dict[str, Any]]) -> bool:
    """An attempt whose current row still accepts ONE late receipt."""
    return bool(row) and str(row.get("kind") or "attempt") == "attempt" and (
        row.get("state") == "unresolved" or is_abandoned_settlement(row))


def provider_price_refinable(row: Optional[Dict[str, Any]]) -> bool:
    """Exact attempt price refinement is independent of physical ownership."""
    return bool(row) and row.get("kind", "attempt") == "attempt" and (
        row.get("cost_final") is not True
        and row.get("state") in {"dispatched", "unresolved", "settled"})


def _provider_price_transition(row: dict, previous: Optional[dict]) -> bool:
    if not provider_price_refinable(previous) or row.get("cost_final") is not True or row.get("cost_usd") is None:
        return False
    receipt, binding = row.get("provider_price_receipt"), previous.get("provider_receipt_binding")
    if not isinstance(receipt, dict) or not isinstance(binding, dict) or not binding:
        return False
    if (receipt.get("attempt_id") != previous.get("attempt_id") or receipt.get("provider") != previous.get("provider")
            or not receipt.get("evidence_ref") or amount(receipt.get("cost_usd")) != amount(row.get("cost_usd"))
            or receipt.get("binding") != binding):
        return False
    changed = {"seq", "ts", "revision", "pre_compaction_seq", "cost_usd", "cost_final", "settle_reason", "provider_price_receipt"}
    return all(row.get(key) == value for key, value in previous.items() if key not in changed)


def validate_row_fields(row: Dict[str, Any], sequence: int) -> None:
    """Structural rules of one usage row, independent of its history."""
    attempt_id = str(row.get("attempt_id") or "")
    state = str(row.get("state") or "")
    if not attempt_id or state not in {"reserved", "dispatched", *_TERMINAL}:
        raise UsageLedgerCorrupt(f"invalid usage ledger row seq={row.get('seq')}")
    if row.get("settle_reason") == "abandoned" and not is_abandoned_settlement(row):
        raise UsageLedgerCorrupt(f"invalid abandoned settlement in usage row seq={sequence}")
    _validate_candidate_facts(row, sequence)
    for numeric_field in (
        "cost_usd", "reservation_upper_bound_usd", "reservation_usd",
        "max_budget_usd", "global_limit_usd", "root_limit_usd", "billing_group_limit_usd",
    ):
        # Nonfinite money is not a torn row. Its distinct error is never
        # quarantined and never stored.
        amount(row.get(numeric_field))
        if row.get(numeric_field) is not None and _number(row.get(numeric_field)) is None:
            raise UsageLedgerCorrupt(f"invalid {numeric_field} in usage row seq={sequence}")
    for token_field in (
        "prompt_tokens", "completion_tokens", "cached_tokens",
        "cache_write_tokens", "ambiguous_call_count",
    ):
        if row.get(token_field) is None:
            continue
        try:
            value = int(row.get(token_field))
        except (TypeError, ValueError, OverflowError) as exc:
            raise UsageLedgerCorrupt(
                f"invalid {token_field} in usage row seq={sequence}"
            ) from exc
        if value < 0 or isinstance(row.get(token_field), bool):
            raise UsageLedgerCorrupt(
                f"invalid {token_field} in usage row seq={sequence}"
            )


def validate_transition(row: Dict[str, Any], previous: Optional[str], late_receipt: bool,
                        sequence: int, *, previous_row: Optional[Dict[str, Any]] = None) -> None:
    """The per-attempt transition table: ``previous`` is the attempt's current
    state (``None`` for a new attempt) and ``late_receipt`` whether that current
    row still accepts one late receipt (``late_receipt_eligible``)."""
    attempt_id = str(row.get("attempt_id") or "")
    state = str(row.get("state") or "")
    kind = str(row.get("kind") or "attempt")
    if is_one_shot_kind(kind):
        if previous is not None or state not in {"settled", "unresolved"}:
            raise UsageLedgerCorrupt(f"invalid legacy usage row seq={row.get('seq')}")
    elif previous is None:
        if state != "reserved":
            raise UsageLedgerCorrupt(f"attempt {attempt_id} did not begin reserved")
    elif previous == "reserved":
        if state not in {"dispatched", "released"}:
            raise UsageLedgerCorrupt(f"invalid transition {previous}->{state}")
    elif previous == "dispatched":
        if state not in {"settled", "unresolved", "released"}:
            raise UsageLedgerCorrupt(f"invalid transition {previous}->{state}")
        if state == "released" and not str(row.get("reason") or "").startswith(
            "before_dispatch_failed:"
        ):
            raise UsageLedgerCorrupt(
                f"dispatched->released requires a typed pre-dispatch reason at seq={row.get('seq')}"
            )
    elif (previous == "settled" and state == "settled" and row.get("settle_reason") == "late_receipt"
          and _provider_price_transition(row, previous_row)):
        pass  # Nonfinal successful price only; no release or abandonment right.
    elif kind == "attempt" and late_receipt and (
        (state == "settled" and (
            row.get("settle_reason") == "late_receipt"
            or (previous == "unresolved" and is_abandoned_settlement(row))
        ))
        or (state == "released" and str(row.get("reason") or "").startswith("before_dispatch_failed:"))
    ):
        pass  # One late receipt replaces uncertainty, never another actual settlement.
    else:
        raise UsageLedgerCorrupt(f"attempt {attempt_id} changed after terminal state")


def _number(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 and parsed == parsed else None


def _final_rows(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row["attempt_id"]): row for row in records}
