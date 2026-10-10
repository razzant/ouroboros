"""Frozen copy of the RETIRED usage-ledger JSONL validator (base 84febbdd3).

Test fixture only. The downgrade export (scripts/export_usage_journal.py) must
produce a journal that an OLDER release accepts, so the export tests validate
with this verbatim copy of ``usage_ledger._validate_records`` and its helpers
as they shipped before the usage store, never with whatever the package holds
now. Do not edit to make a test pass: a change here changes the oracle.
"""
from __future__ import annotations

import decimal
import json
import re
from decimal import Decimal
from typing import Any, Dict, Optional, Sequence

ARCHIVE_SEGMENT_DIR_REL = "archive/usage_ledger"
_ARCHIVE_SEGMENT_PREFIX = ARCHIVE_SEGMENT_DIR_REL + "/"
_TERMINAL = frozenset({"settled", "unresolved", "released"})


class UsageLedgerCorrupt(RuntimeError):
    """Raised when durable history is structurally invalid."""


class UsageNonFiniteMoney(RuntimeError):
    """Nonfinite monetary evidence."""


class LiteralFloat(float):
    """Ordinary JSON row shape with the original numeric literal retained.

Only arithmetic consumes ``literal``. No private field is added to durable
rows or public projections. A copied monetary field is serialized as its exact
decimal string if binary float serialization would change the literal.
    """

    def __new__(cls, literal: str):
        value = super().__new__(cls, literal)
        value.literal = literal
        return value


def decimal_of(value: Any) -> Decimal:
    if isinstance(value, bool):
        # The ledger's historical numeric grammar accepts JSON booleans as 0/1.
        return Decimal(int(value))
    return value if isinstance(value, Decimal) else Decimal(
        value.literal if isinstance(value, LiteralFloat) else str(value))


def amount(value: Any) -> Decimal | None:
    """Decode cash; nonfinite evidence is an integrity failure, never unknown."""
    if value is None:
        return None
    try:
        parsed = decimal_of(value)
    except (decimal.InvalidOperation, ValueError):
        return None
    if not parsed.is_finite():
        # The substrate imports this arithmetic leaf. Resolve its error lazily
        # so every reader fails typed without making it quarantinable corruption.

        raise UsageNonFiniteMoney("non-finite monetary value; source data requires repair")
    return parsed if parsed >= 0 else None


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


def _validate_baseline_header(row: Dict[str, Any], sequence: int) -> None:
    """Provenance checks on the compaction stamp.

    The header claims a summary of bytes that are no longer in this file, so
    its claim must be checkable WITHOUT reading them: a bounded archive path,
    a well-formed source hash, a positive epoch, and counts that actually add
    up to the source row range it names. A header whose numbers do not close
    cannot be an honest fold of anything, whatever the archive holds.
    """

    def _count(key: str, minimum: int) -> int:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise UsageLedgerCorrupt(f"invalid usage baseline {key} seq={sequence}")
        return value

    _count("compaction_epoch", 1)
    if not valid_archive_rel(row.get("archive_rel")):
        raise UsageLedgerCorrupt(f"invalid usage baseline archive_rel seq={sequence}")
    if not _SHA256_RE.fullmatch(str(row.get("source_sha256") or "")):
        raise UsageLedgerCorrupt(f"invalid usage baseline source_sha256 seq={sequence}")
    _count("source_size_bytes", 1)
    _count("folded_attempt_count", 1)
    _count("group_count", 1)
    source_rows = _count("source_row_count", 1)
    folded_rows = _count("folded_row_count", 1)
    retained_rows = _count("retained_row_count", 0)
    if _count("source_first_seq", 1) != 1 or _count("source_last_seq", 1) != source_rows:
        raise UsageLedgerCorrupt(f"usage baseline source range mismatch seq={sequence}")
    if folded_rows + retained_rows != source_rows:
        raise UsageLedgerCorrupt(f"usage baseline row counts do not sum seq={sequence}")


def _validate_records(
    records: Sequence[Dict[str, Any]],
    *,
    start_seq: int = 1,
    states: Optional[Dict[str, str]] = None,
    late_receipt_ids: Optional[set[str]] = None,
) -> None:
    """Validate row structure, dense sequence, and per-attempt transitions.

    ``start_seq``/``states`` are the ADDITIVE resume seam for incremental tail
    validation: a caller that already validated a prefix passes the next
    expected sequence number, per-attempt last states and ids still awaiting a
    late receipt. Both collections are mutated as the tail validates. The latter
    preserves the distinction between an abandoned settlement and an ordinary
    immutable settlement across the incremental boundary.

    Baseline rows (docs/USAGE_COMPACTION.md)
    are legal ONLY as the leading block of a from-scratch validation: exactly
    one ``usage_baseline`` header at seq 1, ``usage_baseline_group`` rows
    joined to it by ``baseline_id``. The compactor rewrites the whole file
    atomically and never appends, so a baseline row in an incremental tail (or
    after any non-baseline row) is corruption. The header's own provenance
    (epoch, bounded archive reference, source hash, closing counts) and the
    block's agreement with it are checked here too, so a forged stamp fails at
    the substrate rather than at whichever reader happens to trust it first.
    """
    baseline_allowed = int(start_seq) == 1 and not states
    baseline_id: Optional[str] = None
    baseline_header: Optional[Dict[str, Any]] = None
    baseline_groups = 0
    baseline_attempts = 0
    baseline_closed = False
    pre_compaction_seq = 0
    pre_compaction_closed = False

    def _close_baseline_block() -> None:
        """Reconcile the header's declared totals with the block that follows."""
        nonlocal baseline_closed
        if baseline_closed or baseline_header is None:
            return
        baseline_closed = True
        if baseline_groups != int(baseline_header.get("group_count") or 0):
            raise UsageLedgerCorrupt("usage baseline group_count does not match the block")
        if baseline_attempts != int(baseline_header.get("folded_attempt_count") or 0):
            raise UsageLedgerCorrupt(
                "usage baseline folded_attempt_count does not match the block"
            )

    states = {} if states is None else states
    late_receipt_ids = set() if late_receipt_ids is None else late_receipt_ids
    expected = int(start_seq)
    for row in records:
        try:
            sequence = int(row.get("seq") or 0) if isinstance(row, dict) else 0
        except (TypeError, ValueError, OverflowError) as exc:
            raise UsageLedgerCorrupt(f"invalid usage ledger sequence at {expected}") from exc
        if not isinstance(row, dict) or sequence != expected:
            raise UsageLedgerCorrupt(f"usage ledger sequence mismatch at {expected}")
        expected += 1
        attempt_id = str(row.get("attempt_id") or "")
        state = str(row.get("state") or "")
        kind = str(row.get("kind") or "attempt")
        if not attempt_id or state not in {"reserved", "dispatched", *_TERMINAL}:
            raise UsageLedgerCorrupt(f"invalid usage ledger row seq={row.get('seq')}")
        if row.get("settle_reason") == "abandoned" and not is_abandoned_settlement(row):
            raise UsageLedgerCorrupt(f"invalid abandoned settlement in usage row seq={sequence}")
        _validate_candidate_facts(row, sequence)
        for numeric_field in (
            "cost_usd", "reservation_upper_bound_usd", "reservation_usd",
            "max_budget_usd", "global_limit_usd", "root_limit_usd", "billing_group_limit_usd",
        ):
            # Nonfinite money is not a torn row. Its distinct error bypasses
            # tail quarantine, including incremental fallback and compaction.
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
        previous = states.get(attempt_id)
        if kind in {"usage_baseline", "usage_baseline_group"}:
            if not baseline_allowed or previous is not None:
                raise UsageLedgerCorrupt(
                    f"baseline row outside the leading block at seq={sequence}"
                )
            if kind == "usage_baseline":
                identity = row.get("baseline_id")
                if baseline_id is not None or sequence != 1 or state != "settled" or not (
                    isinstance(identity, str) and identity
                ):
                    raise UsageLedgerCorrupt(f"invalid usage baseline header seq={sequence}")
                _validate_baseline_header(row, sequence)
                baseline_id = identity
                baseline_header = row
            else:
                count = row.get("folded_attempt_count")
                if (
                    baseline_id is None
                    or row.get("baseline_id") != baseline_id
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 1
                    or state not in _TERMINAL
                ):
                    raise UsageLedgerCorrupt(f"invalid usage baseline group seq={sequence}")
                baseline_groups += 1
                baseline_attempts += count
            states[attempt_id] = state
            continue
        baseline_allowed = False
        _close_baseline_block()
        # ``pre_compaction_seq`` is a provenance claim about an epoch that only
        # a leading header proves happened, and the compactor emits it on the
        # retained rows in one strictly increasing run before any later append.
        # The claim also has to name a row the archived source ACTUALLY held:
        # the header declares that range (``source_first_seq``..
        # ``source_last_seq``), and a retained row claiming an origin outside
        # it claims to come from bytes nobody archived.
        carried = row.get("pre_compaction_seq")
        if carried is not None:
            if (
                baseline_header is None
                or pre_compaction_closed
                or isinstance(carried, bool)
                or not isinstance(carried, int)
                or carried <= pre_compaction_seq
                or carried < int(baseline_header.get("source_first_seq") or 0)
                or carried > int(baseline_header.get("source_last_seq") or 0)
            ):
                raise UsageLedgerCorrupt(f"invalid pre_compaction_seq in usage row seq={sequence}")
            pre_compaction_seq = carried
        else:
            pre_compaction_closed = True
        if kind.startswith("legacy_") or kind in {"external_unmetered", "subscription_session"}:
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
        elif kind == "attempt" and attempt_id in late_receipt_ids and (
            (state == "settled" and (
                row.get("settle_reason") == "late_receipt"
                or (previous == "unresolved" and is_abandoned_settlement(row))
            ))
            or (state == "released" and str(row.get("reason") or "").startswith("before_dispatch_failed:"))
        ):
            pass  # One late receipt replaces uncertainty, never another actual settlement.
        else:
            raise UsageLedgerCorrupt(f"attempt {attempt_id} changed after terminal state")
        states[attempt_id] = state
        if kind == "attempt" and (state == "unresolved" or is_abandoned_settlement(row)):
            late_receipt_ids.add(attempt_id)
        else:
            late_receipt_ids.discard(attempt_id)
    _close_baseline_block()


def _decode_record(chunk: bytes) -> Optional[Dict[str, Any]]:
    """One JSONL grammar for full, incremental and unlocked prepared reads.

    Only empty CR/LF lines are empty records. UTF-8 (without BOM), object
    shape and literal money spelling are identical in every cache state.
    """
    raw = chunk.rstrip(b"\r\n")
    if not raw:
        return None
    row = json.loads(raw.decode("utf-8"), parse_float=LiteralFloat)
    if not isinstance(row, dict):
        raise ValueError("row is not an object")
    return row


def _number(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 and parsed == parsed else None

