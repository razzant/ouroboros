"""The runtime invariant ``model-visible ⟺ logged`` for ``model_send``.

The invariant binds exactly one object — the physical candidate payload at the
last host-controlled pre-transport seam (``llm_attempt._candidate_before_dispatch``)
— per the design note ``docs/MODEL_SEND_OBSERVABILITY.md``:

- **Forward** (``sent ⟹ logged``): every physical attempt persists a sealed
  durable record of its exact send copy before dispatch (the ``model_send_seal``
  block of the existing physical-candidate manifest). Explicit audit compares
  its digest to the attempt and reads back CAS bytes. A supplied original
  candidate also permits the exclusion-aware byte comparison
  (:func:`verify_sealed_candidate`); dispatch does no disk read-back.
- **Reverse** (``logged ⟹ sent``, ``model_send`` records ONLY): a bounded
  fail-soft reconciliation sweep (:func:`reconcile_model_send_seals`) joins every
  seal to exactly one accounting attempt and every seam-dispatched attempt back
  to its seal. Orphans on either side become typed facts, never a repair.
- Everything else leaves the byte domain only through the CLOSED exclusion enum
  below; an undisclosed transformation is by definition a violation. Delegated
  and opaque-SDK lanes never hold the final wire bytes, so they carry the
  disclosed :data:`MODEL_SEND_SEAL_UNOBSERVED` limit instead of a fake seal.

Reuse-first: the seal reuses the existing ``canonical_json_v1`` digests
(``llm_attempt``), the existing CAS/redaction pipeline
(``observability.persist_call``) and the existing attempt ledger — no parallel
serializer, plane, or scheduler is minted here.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


SEAL_VERSION = 1

# Serializer identity (design note §3.1/§5.4): the ONLY equality basis. Any
# serializer change is a NEW basis string; a reader never re-interprets bytes
# under a different basis.
CANONICAL_BASIS = "canonical_json_v1"

# Closed exclusion enum (design note §4). Anything not named here is IN the
# byte domain by construction.
EXCLUSION_SECRET_REDACTION = "secret_redaction"
EXCLUSION_PROVIDER_NATIVE_CUSTODY = "provider_native_custody"
EXCLUSION_TRANSPORT_ENVELOPE = "transport_envelope"
EXCLUSION_PROVIDER_SIDE_TRANSFORM = "provider_side_transform"
EXCLUSION_CLASSES = frozenset({
    EXCLUSION_SECRET_REDACTION,
    EXCLUSION_PROVIDER_NATIVE_CUSTODY,
    EXCLUSION_TRANSPORT_ENVELOPE,
    EXCLUSION_PROVIDER_SIDE_TRANSFORM,
})

# Lane-level disclosed limit for delegated/harness model calls (agent_session
# executor lanes): the host never holds the final wire bytes there, so their
# accounting rows carry this marker instead of a fabricated seal (same honesty
# pattern as the scope brief's ``read_provenance_expected`` fact).
MODEL_SEND_SEAL_UNOBSERVED = "unobserved"

VIOLATION_EVENT_TYPE = "model_send_invariant_violation"
# One record shape for all failure surfaces (design note §3.4).
VIOLATION_KINDS = (
    "content_divergence", "reconstruction_divergence", "orphan_seal", "unlogged_attempt",
)


def build_model_send_seal(
    *,
    attempt_id: str,
    candidate: Dict[str, Any],
    projected: Any,
    candidate_facts: Dict[str, Any],
    redaction: Any,
    custody_projected: Optional[bool] = None,
) -> Dict[str, Any]:
    """The ``model_send_seal`` block for one physical-candidate manifest.

    ``projected`` is the physical custody projection persist already computed;
    ``redaction`` is the :class:`observability.RedactionResult` of the CAS write
    (built by ``persist_call`` — the seal discloses the exact redaction
    instances that fired, it never re-runs them). ``custody_projected`` lets the
    caller pass the ``projected != candidate`` fact it already computed for its
    own disclosure flag instead of paying a second tree compare.
    """
    exclusions: List[Dict[str, Any]] = [
        {"class": EXCLUSION_SECRET_REDACTION, "path": str(record.path)}
        for record in getattr(redaction, "records", [])
    ]
    if custody_projected is None:
        custody_projected = projected != candidate
    if custody_projected:
        exclusions.extend(_custody_exclusion_rows(candidate, projected, "$"))
    # Class-level rows (design note §4): the transport adds envelope fields below
    # the seam on every lane, and provider-side transforms are never
    # host-observable — the seam is the LAST host-controlled point, not the last
    # point.
    exclusions.append({"class": EXCLUSION_TRANSPORT_ENVELOPE})
    exclusions.append({"class": EXCLUSION_PROVIDER_SIDE_TRANSFORM})
    return {
        "seal_version": SEAL_VERSION,
        "canonical_basis": str(
            candidate_facts.get("candidate_measurement_kind") or CANONICAL_BASIS
        ),
        "pre_redaction_sha256": str(candidate_facts.get("candidate_raw_sha256") or ""),
        "size_bytes": int(candidate_facts.get("candidate_raw_size_bytes") or 0),
        "attempt_id": str(attempt_id),
        "exclusions": exclusions,
    }


def persist_physical_candidate(
    drive_root: pathlib.Path,
    *,
    task_id: str,
    attempt_id: str,
    candidate: Dict[str, Any],
    candidate_facts: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist one inspectable post-transform candidate under its attempt id.

    ``candidate_facts`` describe the pre-redaction canonical object. The normal
    ``persist_call`` refs describe the redacted-by-default CAS blob; the two
    digest domains are deliberately labelled rather than equated.

    The manifest carries the ``model_send_seal`` block (additive key under
    the existing SCHEMA_VERSION object; readers ignore unknown keys), and the
    returned ``manifest_ref`` is stamped ``model_send_seal_version`` so the
    accounting row it lands on names its attempt as seam-sealed — the join key
    the reverse reconciliation sweep enforces. ``observability.py`` keeps the
    compatibility export so existing callers use this same implementation.
    """
    from ouroboros.anthropic_native_custody import physical_custody_projection
    from ouroboros.observability import persist_call

    projected = physical_custody_projection(candidate)
    custody_projected = projected != candidate
    persisted = persist_call(
        drive_root,
        task_id=task_id,
        call_id=attempt_id,
        call_type="physical_llm_candidate",
        payload=projected,
        keep_raw=False,
        manifest={
            "candidate_manifest_kind": "physical_llm_candidate",
            "candidate_raw_digest_basis": "canonical_json_v1_pre_redaction",
            "redacted_projection_digest_basis": "observability_json_v1_post_default_redaction_cas",
            "anthropic_native_custody_projected": custody_projected,
            **dict(candidate_facts),
        },
        finalize_manifest=lambda redacted: {
            "model_send_seal": build_model_send_seal(
                attempt_id=attempt_id,
                candidate=candidate,
                projected=projected,
                candidate_facts=dict(candidate_facts),
                redaction=redacted,
                custody_projected=custody_projected,
            ),
        },
    )
    persisted["manifest_ref"] = {
        **persisted["manifest_ref"], "model_send_seal_version": SEAL_VERSION,
    }
    return persisted


def _custody_exclusion_rows(
    original: Any, projected: Any, path: str,
) -> List[Dict[str, Any]]:
    """Per-instance ``provider_native_custody`` rows from a structural diff.

    The custody projector (``anthropic_native_custody``) stays the SSOT of what
    leaves the byte domain; this walk only NAMES where it acted, so the two can
    never disagree about the set of excluded sites.
    """
    rows: List[Dict[str, Any]] = []
    if original == projected:
        return rows
    if isinstance(original, dict) and isinstance(projected, dict):
        for key, value in original.items():
            child_path = f"{path}.{key}"
            if key not in projected:
                # A dropped private custody key (replay receipt / consumed list).
                row = {"class": EXCLUSION_PROVIDER_NATIVE_CUSTODY, "path": child_path}
                digest = value.get("content_sha256") if isinstance(value, dict) else None
                if digest:
                    row["opaque_sha256"] = str(digest)
                rows.append(row)
            else:
                rows.extend(_custody_exclusion_rows(value, projected[key], child_path))
        return rows
    if (
        isinstance(original, (list, tuple))
        and isinstance(projected, list)
        and len(original) == len(projected)
    ):
        for idx, (item, twin) in enumerate(zip(original, projected)):
            rows.extend(_custody_exclusion_rows(item, twin, f"{path}[{idx}]"))
        return rows
    # A replaced node: the projection substituted opaque provider content with
    # digest metadata; disclose each digest it minted, or the site alone.
    digests: List[str] = []
    if isinstance(projected, dict):
        from ouroboros.anthropic_native_custody import ANTHROPIC_OPAQUE_PROJECTION_KEY

        for item in projected.get(ANTHROPIC_OPAQUE_PROJECTION_KEY) or []:
            if isinstance(item, dict) and item.get("sha256"):
                digests.append(str(item["sha256"]))
    if digests:
        rows.extend(
            {"class": EXCLUSION_PROVIDER_NATIVE_CUSTODY, "path": path, "opaque_sha256": digest}
            for digest in digests
        )
    else:
        rows.append({"class": EXCLUSION_PROVIDER_NATIVE_CUSTODY, "path": path})
    return rows


def _reconstruct_blob_bytes(candidate: Dict[str, Any]) -> bytes:
    """Apply the SAME exclusion map the persist pipeline applies and serialize
    on the CAS basis (design note §3.2: redaction and custody projection are not
    invertible, so the comparable domain is ``project(W, exclusions)``).

    This composition mirrors ``persist_physical_candidate`` → ``persist_call``
    exactly; the normal-call equality pin in the test suite is the drift
    detector between the two.
    """
    from ouroboros.anthropic_native_custody import (
        observability_custody_projection,
        physical_custody_projection,
    )
    from ouroboros.observability import _json_bytes, redact_projection

    return _json_bytes(
        redact_projection(
            observability_custody_projection(physical_custody_projection(candidate))
        ).value
    )


def _read_blob_bytes(manifest: Any) -> Tuple[Optional[bytes], bool]:
    """(raw CAS bytes, cas_digest_intact) read back from the durable record."""
    ref = manifest.get("full_payload_ref") if isinstance(manifest, dict) else None
    try:
        path = pathlib.Path(str((ref or {}).get("path") or ""))
        with gzip.open(path, "rb") as handle:
            raw = handle.read()
    except Exception:
        return None, False
    intact = hashlib.sha256(raw).hexdigest() == str((ref or {}).get("sha256") or "")
    return raw, intact


def _first_divergent_offset(left: bytes, right: bytes) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def violation_fact_path(
    drive_root: pathlib.Path | str, task_id: str, attempt_id: str,
) -> pathlib.Path:
    """The durable twin's beside-the-seal location (survives log rotation with
    its subject). Sanitization mirrors ``observability.write_call_manifest``."""
    from ouroboros.observability import _observability_root

    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id or "unknown")).strip("_") or "unknown"
    safe_attempt = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(attempt_id or "unknown")).strip("_") or "unknown"
    return (
        _observability_root(pathlib.Path(drive_root))
        / "calls" / safe_task / f"{safe_attempt}.model_send_violation.json"
    )


def record_model_send_violation(
    drive_root: pathlib.Path | str, fact: Dict[str, Any],
) -> bool:
    """Write one typed violation fact durably: beside the seal AND appended to
    ``events.jsonl`` (design note §3.4). The beside-file is a write-once latch,
    so a repeated sweep can never flood the events plane with the same fact.
    Digests and offsets only — no secret bytes ever enter the fact.
    """
    from ouroboros.utils import append_jsonl, atomic_write_json, utc_now_iso

    path = violation_fact_path(
        drive_root, str(fact.get("task_id") or ""), str(fact.get("attempt_id") or ""),
    )
    if path.exists():
        return False
    payload = {"ts": utc_now_iso(), **fact}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload, trailing_newline=True)
    append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", payload)
    return True


def verify_sealed_candidate(
    drive_root: pathlib.Path | str,
    *,
    task_id: str,
    attempt_id: str,
    candidate: Optional[Dict[str, Any]],
    manifest_ref: Any,
    raw_sha256: str,
    raw_size_bytes: int,
    record: Any = None,
) -> Optional[Dict[str, Any]]:
    """Explicit forward audit; never called from the send path.

    Reads the durable sealed record FROM DISK; with an original candidate applies the same
    exclusion map to the wire-bound candidate, and compares byte-for-byte —
    so a bug between "what we persisted" and "what we believe we persisted" is
    caught, not assumed away. ``raw_sha256``/``raw_size_bytes`` are the fresh
    seam digests the caller already computed for its identity re-check: the raw
    candidate is deliberately NOT serialized a second time here; the one added
    projection+serialization is the note's budgeted cost.

    Returns the typed violation fact (already durably recorded) on mismatch,
    else ``None``. Without an original candidate, compare attempt digest and CAS
    integrity only: excluded secret/native bytes cannot be reconstructed. Never raises: this invariant is
    observability — the in-memory identity refusal above the call keeps being
    the only blocking authority.
    """
    try:
        observed = {
            "basis": CANONICAL_BASIS,
            "sha256": str(raw_sha256 or ""),
            "size": int(raw_size_bytes or 0),
        }
        expected = {"basis": "", "sha256": "", "size": 0}
        kind = ""
        divergence_class = ""
        first_offset: Optional[int] = None
        try:
            manifest_path = pathlib.Path(str((manifest_ref or {}).get("path") or ""))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = None
        seal = manifest.get("model_send_seal") if isinstance(manifest, dict) else None
        if not isinstance(seal, dict):
            kind, divergence_class = "reconstruction_divergence", "seal_unreadable"
        else:
            expected = {
                "basis": str(seal.get("canonical_basis") or ""),
                "sha256": str(seal.get("pre_redaction_sha256") or ""),
                "size": int(seal.get("size_bytes") or 0),
            }
            undisclosed = [
                row for row in (seal.get("exclusions") or [])
                if str((row or {}).get("class") if isinstance(row, dict) else "") not in EXCLUSION_CLASSES
            ]
            if expected["basis"] != CANONICAL_BASIS:
                # §5.4: never re-interpret bytes under a different basis.
                kind, divergence_class = "reconstruction_divergence", "serializer_basis"
            elif undisclosed:
                # Closed enum: an exclusion class outside §4 is undisclosed by
                # definition, whatever its row says.
                kind, divergence_class = "reconstruction_divergence", "undisclosed_exclusion"
            elif (
                expected["sha256"] != observed["sha256"]
                or expected["size"] != observed["size"]
            ):
                # The durable claim and the wire bytes disagree about the raw
                # candidate itself (§5.5 in-place mutation class).
                kind, divergence_class = "content_divergence", "sdk_mutation"
            else:
                blob_bytes, cas_intact = _read_blob_bytes(manifest)
                if blob_bytes is None:
                    kind, divergence_class = "reconstruction_divergence", "record_unreadable"
                else:
                    reconstructed = _reconstruct_blob_bytes(candidate) if candidate is not None else blob_bytes
                    if reconstructed != blob_bytes or not cas_intact:
                        first_offset = _first_divergent_offset(reconstructed, blob_bytes)
                        # CAS-intact blob differing from a fresh projection of a
                        # digest-matching candidate means the exclusion map
                        # itself moved between persist and dispatch (§5.1
                        # redaction divergence); a broken CAS digest is plain
                        # record corruption.
                        kind = "reconstruction_divergence"
                        divergence_class = "redaction" if cas_intact else "record_corrupt"
        if not kind:
            return None
        fact: Dict[str, Any] = {
            "type": VIOLATION_EVENT_TYPE,
            "kind": kind,
            "attempt_id": str(attempt_id),
            "task_id": str(task_id or ""),
            "expected": expected,
            "observed": observed,
            "divergence_class": divergence_class,
        }
        if first_offset is not None:
            fact["first_divergent_offset"] = int(first_offset)
        (record(fact) if record is not None else record_model_send_violation(drive_root, fact))
        return fact
    except Exception:
        # Fail-soft by contract: the observability invariant must never take
        # down the call it observes; the failure is still visible here.
        log.warning("model_send invariant verification failed soft", exc_info=True)
        return None


def _seal_manifest_paths(drive_root: pathlib.Path, limit: int) -> List[pathlib.Path]:
    from ouroboros.observability import _observability_root

    def _mtime(path: pathlib.Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    # Read-only glob below the existing calls plane (no new path construction:
    # the persistence inventory pins the writer-path population).
    paths = [
        path for path in _observability_root(drive_root).glob("calls/*/*.json")
        if not path.name.endswith(".model_send_violation.json")
    ]
    paths.sort(key=_mtime, reverse=True)
    return paths[: max(0, int(limit))]


def reconcile_model_send_seals(
    drive_root: pathlib.Path | str,
    *,
    max_manifests: int = 2000,
    max_facts: int = 50,
) -> Dict[str, Any]:
    """Reverse direction of the invariant (design note §3.3, ``model_send`` only).

    Bounded reconciliation in the explicit lifecycle audit: every
    ``model_send`` seal must join exactly one accounting attempt (any terminal
    state, including refused-before-dispatch), and every attempt DISPATCHED
    THROUGH THE SEALING SEAM (its ``candidate_manifest_ref`` carries
    ``model_send_seal_version``) must still resolve to its durable seal. An
    orphan on either side is a typed durable fact — the sweep deletes no seals
    and fabricates no attempts. "No attempt row" is asked of the usage store
    UNION the retained journal evidence (the imported journal and the old
    compaction archive: an attempt folded before the store existed is absent
    from the store by design). Fail-soft: an unreadable store is UNKNOWN
    accounting state and skips every conclusion, and so is retained evidence
    that cannot be read. This is the explicit audit's history read (the whole
    store via ``read_usage_records``), never an ordinary path. Manifests promoted from a
    child drive (``promoted_call_manifest``) are excluded — their attempt rows
    legitimately live in the child's ledger, not this one.
    """
    report: Dict[str, Any] = {
        "status": "completed", "manifests_checked": 0,
        "seals": 0, "sealed_attempts": 0,
        "orphan_seals": 0, "unlogged_attempts": 0,
        "facts_written": 0, "truncated": False,
    }
    try:
        root = pathlib.Path(drive_root)
        from ouroboros.usage_accounting import read_usage_records

        # Reservation precedes seal persistence. Select this pass's manifests
        # before its live snapshot so a newly created seal cannot be mistaken
        # for an orphan merely because its reservation arrived after the read.
        manifest_paths = _seal_manifest_paths(root, max_manifests)
        finals = {str(row["attempt_id"]): row for row in read_usage_records(root)}
    except Exception:
        log.debug("model_send reconciliation skipped: ledger state unknown", exc_info=True)
        report["status"] = "unknown"
        return report

    def _write(fact: Dict[str, Any]) -> None:
        if report["facts_written"] >= max_facts:
            report["truncated"] = True
            return
        try:
            if record_model_send_violation(root, fact):
                report["facts_written"] += 1
        except Exception:
            log.debug("model_send reconciliation fact write failed", exc_info=True)
            report["status"] = "unknown"

    try:
        _reconcile_seal_directions(root, finals, _write, report, manifest_paths)
    except Exception:
        log.debug("model_send reconciliation failed soft", exc_info=True)
        report["status"] = "unknown"
    return report


def _reconcile_seal_directions(
    root: pathlib.Path,
    finals: Dict[str, Dict[str, Any]],
    _write: Any,
    report: Dict[str, Any],
    manifest_paths: List[pathlib.Path],
) -> None:
    live_ids = set(finals)
    seals: List[Tuple[Dict[str, Any], Dict[str, Any], str]] = []
    for manifest_path in manifest_paths:
        report["manifests_checked"] = report.get("manifests_checked", 0) + 1
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            report["status"] = "unknown"
            continue
        if not isinstance(manifest, dict) or manifest.get("promoted_call_manifest"):
            continue
        seal = manifest.get("model_send_seal")
        if not isinstance(seal, dict):
            continue
        report["seals"] += 1
        aid = str(seal.get("attempt_id") or manifest.get("call_id") or "")
        attempt = finals.get(aid, {})
        if attempt.get("candidate_raw_sha256"):
            fact = verify_sealed_candidate(root, task_id=str(manifest.get("task_id") or ""),
                attempt_id=aid, candidate=None, manifest_ref={"path": str(manifest_path)},
                raw_sha256=attempt["candidate_raw_sha256"],
                raw_size_bytes=int(attempt.get("candidate_raw_size_bytes") or 0), record=_write)
            if fact:
                report["forward_divergences"] = report.get("forward_divergences", 0) + 1
        seals.append((manifest, seal, str(seal.get("attempt_id") or manifest.get("call_id") or "")))
    # One retained-evidence scan per pass, for the ids the store does not hold.
    retained = _retained_attempt_ids(root, {attempt_id for _m, _s, attempt_id in seals
                                            if attempt_id and attempt_id not in live_ids})
    if retained is None:  # UNKNOWN skips reverse accusations; never a failed read as absence
        report["status"] = "unknown"
    for manifest, seal, attempt_id in seals:
        if attempt_id and retained is not None and attempt_id not in live_ids and attempt_id not in retained:
            report["orphan_seals"] += 1
            _write({
                "type": VIOLATION_EVENT_TYPE,
                "kind": "orphan_seal",
                "attempt_id": attempt_id,
                "task_id": str(manifest.get("task_id") or ""),
                "expected": {
                    "basis": str(seal.get("canonical_basis") or ""),
                    "sha256": str(seal.get("pre_redaction_sha256") or ""),
                    "size": int(seal.get("size_bytes") or 0),
                },
                "observed": {"basis": "", "sha256": "", "size": 0},
                "divergence_class": "missing_attempt_row",
            })

    for attempt_id, row in finals.items():
        ref = row.get("candidate_manifest_ref")
        if not isinstance(ref, dict) or not ref.get("model_send_seal_version"):
            continue
        if str(row.get("state") or "") not in {"dispatched", "settled", "unresolved"}:
            continue
        report["sealed_attempts"] += 1
        seal = None
        try:
            from ouroboros.observability import read_call_manifest_ref
            manifest = read_call_manifest_ref(root, ref, task_id=str(row.get("task_id") or ""))
            if isinstance(manifest, dict):
                seal = manifest.get("model_send_seal")
        except Exception:
            seal = None
        if not isinstance(seal, dict):
            report["unlogged_attempts"] += 1
            _write({
                "type": VIOLATION_EVENT_TYPE,
                "kind": "unlogged_attempt",
                "attempt_id": str(attempt_id),
                "task_id": str(row.get("task_id") or ""),
                "expected": {
                    "basis": str(row.get("candidate_measurement_kind") or ""),
                    "sha256": str(row.get("candidate_raw_sha256") or ""),
                    "size": int(row.get("candidate_raw_size_bytes") or 0),
                },
                "observed": {"basis": "", "sha256": "", "size": 0},
                "divergence_class": "missing_seal_record",
            })


def _retained_attempt_ids(root: pathlib.Path, wanted: set) -> Optional[set]:
    """The ids of ``wanted`` that retained money evidence holds: the retired
    journal files (``state/usage_attempts.jsonl*``: the imported journal) and
    the old compaction archive segments (``archive/usage_ledger/``), newest
    first, stopping once every id is found. A plain id scan for the explicit
    audit: no chain verification, no hashes. ``None`` when a retained file
    cannot be read (unknown, never absence)."""
    if not wanted:
        return set()

    def newest_first(paths):
        def mtime(path):
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0
        return sorted((path for path in paths if path.is_file()), key=mtime, reverse=True)

    files = newest_first((root / "state").glob("usage_attempts.jsonl*"))
    archive = root / "archive" / "usage_ledger"
    if archive.is_dir():
        files += newest_first(archive.iterdir())
    needles = {attempt_id.encode("utf-8"): attempt_id for attempt_id in wanted}
    found: set = set()
    try:
        for path in files:
            with open(path, "rb") as handle:
                for line in handle:
                    if not any(needle in line for needle in needles):
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    attempt_id = str(row.get("attempt_id") or "") if isinstance(row, dict) else ""
                    if attempt_id in wanted:
                        found.add(attempt_id)
                        if len(found) == len(wanted):
                            return found
    except OSError:
        log.debug("model_send reconciliation: retained evidence unreadable", exc_info=True)
        return None
    return found
