"""Transport-side result verification and operation reconciliation (RWS v2 §3.2).

This is the TRANSPORT half of the donor's ``remote_finalization``: everything
that decides whether returned bytes are the bytes that were declared, and
everything that walks a reopened session's operation ledger to a durable,
acknowledged conclusion.

Two rules shape the module:

* **Nothing is trusted because it arrived.**  Every blob is fetched only when a
  declaration names it, bounded before allocation, and accepted only when its
  size and SHA-256 match that declaration; the externalized envelope is parsed
  with duplicate keys, floats, oversized integers, depth and item counts all
  rejected.  A verification failure prevents the ACK, so the remote side keeps
  its evidence.
* **No Home authority is imported.**  The Home import itself happens behind the
  injected ``HomeImporter`` protocol (``transport.home_importer``).  The Home
  half proper — writing artifacts, observability rows and the public record —
  belongs to the transfer service; from here it is one call.

Reconciliation is deliberately not a retry engine.  ``completed`` imports and
ACKs; ``completed`` with an unavailable stored result becomes durable terminal
evidence, never permission to repeat the mutation; only a proven
``not_started`` drops the intent.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import time
from collections.abc import Mapping
from hashlib import sha256
from typing import Any, Callable

from ouroboros.remote_pending_operations import (
    pending_scope_root,
    remove_pending_operation,
    terminal_evidence_identity,
)
from ouroboros.remote_protocol import MAX_REMOTE_EXTERNAL_ENVELOPE_BYTES
from ouroboros.utils import atomic_write_json

_REMOTE_PROCESS_STREAM_ORDER = ("stdout.txt", "stderr.txt")
_REMOTE_PROCESS_PREVIEW_BYTES = 64_000
_REMOTE_PROCESS_BLOB_MAX_BYTES = 16_000_000
_REMOTE_DECLARED_OUTPUT_MAX_BYTES = 32 * 1024 * 1024
_REMOTE_RESULT_IMPORT_MAX_BYTES = (
    2 * _REMOTE_PROCESS_BLOB_MAX_BYTES
    + _REMOTE_DECLARED_OUTPUT_MAX_BYTES
    + MAX_REMOTE_EXTERNAL_ENVELOPE_BYTES
)
_REMOTE_IMPORT_JSON_MAX_DEPTH = 64
_REMOTE_IMPORT_JSON_MAX_ITEMS = 100_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_RESULT_UNAVAILABLE_TEXT = (
    "The remote operation completed, but its retained result is unavailable."
)
_RESULT_UNAVAILABLE_REASON = (
    "The operation will not be repeated because remote completion is already "
    "durable."
)


# ── declaration verification ────────────────────────────────────────────


def _remote_blob_ref(
    raw: Any,
    *,
    label: str,
    expected_mime: str,
    max_bytes: int,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"{label} is not an object")
    blob_id = str(raw.get("blob_id") or "")
    digest = str(raw.get("sha256") or "")
    size_raw = raw.get("size")
    if (
        not _SHA256_RE.fullmatch(blob_id)
        or digest != blob_id
        or not isinstance(size_raw, int)
        or isinstance(size_raw, bool)
        or size_raw < 0
        or size_raw > max_bytes
        or str(raw.get("mime") or "") != expected_mime
    ):
        raise RuntimeError(f"{label} declaration is invalid")
    return {
        "name": str(raw.get("name") or ""),
        "blob_id": blob_id,
        "sha256": digest,
        "size": size_raw,
        "mime": expected_mime,
        "truncated": bool(raw.get("truncated")),
    }


def _strict_remote_envelope(payload: bytes) -> dict[str, Any]:
    def _reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        decoded = payload.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise RuntimeError(
            f"externalized operation envelope is invalid: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError("externalized operation envelope is not an object")
    stack: list[tuple[Any, int]] = [(value, 0)]
    item_count = 0
    while stack:
        current, depth = stack.pop()
        if depth > _REMOTE_IMPORT_JSON_MAX_DEPTH:
            raise RuntimeError("externalized operation envelope exceeds depth limit")
        if isinstance(current, dict):
            item_count += len(current)
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            item_count += len(current)
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, float):
            raise RuntimeError("externalized operation envelope contains a float")
        elif isinstance(current, int) and not (
            -(1 << 63) <= current <= (1 << 63) - 1
        ):
            raise RuntimeError(
                "externalized operation envelope contains an oversized integer"
            )
        elif not isinstance(current, (str, int, bool, type(None))):
            raise RuntimeError(
                "externalized operation envelope contains an invalid value"
            )
        if item_count > _REMOTE_IMPORT_JSON_MAX_ITEMS:
            raise RuntimeError("externalized operation envelope exceeds item limit")
    return value


def _externalized_envelope_ref(
    envelope: Mapping[str, Any],
) -> dict[str, Any] | None:
    trace = envelope.get("trace")
    raw_ref = trace.get("externalized_result") if isinstance(trace, Mapping) else None
    if raw_ref is None:
        return None
    ref = _remote_blob_ref(
        raw_ref,
        label="externalized operation envelope",
        expected_mime="application/json",
        max_bytes=MAX_REMOTE_EXTERNAL_ENVELOPE_BYTES,
    )
    if ref["name"] != "operation-envelope.json" or ref["size"] <= 0:
        raise RuntimeError("externalized operation envelope identity is invalid")
    artifacts = envelope.get("artifacts")
    rows = list(artifacts) if isinstance(artifacts, list) else []
    matching = [
        item
        for item in rows
        if isinstance(item, Mapping)
        and str(item.get("name") or "") == "operation-envelope.json"
    ]
    if len(matching) != 1 or _remote_blob_ref(
        matching[0],
        label="externalized operation envelope artifact",
        expected_mime="application/json",
        max_bytes=MAX_REMOTE_EXTERNAL_ENVELOPE_BYTES,
    ) != ref:
        raise RuntimeError("externalized operation envelope declarations disagree")
    return ref


def _process_blob_refs(envelope: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(envelope.get("process"), Mapping):
        return []
    artifacts = envelope.get("artifacts")
    rows = list(artifacts) if isinstance(artifacts, list) else []
    refs: list[dict[str, Any]] = []
    for name in _REMOTE_PROCESS_STREAM_ORDER:
        matches = [
            item
            for item in rows
            if isinstance(item, Mapping) and str(item.get("name") or "") == name
        ]
        if len(matches) > 1:
            raise RuntimeError(
                f"remote process returned duplicate {name} declarations"
            )
        if not matches:
            continue
        ref = _remote_blob_ref(
            matches[0],
            label=f"remote process {name}",
            expected_mime="text/plain",
            max_bytes=_REMOTE_PROCESS_BLOB_MAX_BYTES,
        )
        if ref["size"] <= _REMOTE_PROCESS_PREVIEW_BYTES:
            raise RuntimeError(
                f"remote process {name} blob is below externalization threshold"
            )
        refs.append(ref)
    return refs


def _declared_output_refs(envelope: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = envelope.get("artifacts")
    rows = list(artifacts) if isinstance(artifacts, list) else []
    refs: list[dict[str, Any]] = []
    total = 0
    for index, item in enumerate(rows):
        if not isinstance(item, Mapping) or item.get("kind") != "declared_output":
            continue
        ref = _remote_blob_ref(
            item,
            label=f"remote declared output {index}",
            expected_mime="application/octet-stream",
            max_bytes=_REMOTE_DECLARED_OUTPUT_MAX_BYTES,
        )
        total += ref["size"]
        if total > _REMOTE_DECLARED_OUTPUT_MAX_BYTES:
            raise RuntimeError("remote declared outputs exceed aggregate limit")
        ref.update({
            "declared_as": str(item.get("declared_as") or ""),
            "member_path": str(item.get("member_path") or ""),
        })
        refs.append(ref)
    return refs


def prefetch_remote_result_import(
    result: Mapping[str, Any],
    fetch_blob: Callable[[str, int], bytes],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch only declared process/envelope blobs, verify them, and stay bounded."""

    raw_envelope = result.get("envelope")
    if not isinstance(raw_envelope, Mapping):
        raise RuntimeError("execd result omitted its operation envelope")
    envelope = dict(raw_envelope)
    imported_bytes = 0
    external_payload = b""
    external_ref = _externalized_envelope_ref(envelope)
    source_envelope = envelope
    if external_ref is not None:
        external_payload = bytes(
            fetch_blob(external_ref["blob_id"], external_ref["size"])
        )
        if (
            len(external_payload) != external_ref["size"]
            or sha256(external_payload).hexdigest() != external_ref["sha256"]
        ):
            raise RuntimeError(
                "externalized operation envelope failed integrity verification"
            )
        imported_bytes += len(external_payload)
        source_envelope = _strict_remote_envelope(external_payload)

    output_projection_present = "output_blobs" in result
    declared_outputs = result.get("output_blobs")
    if output_projection_present and not isinstance(declared_outputs, Mapping):
        raise RuntimeError("remote output blob projection is invalid")
    output_blobs = declared_outputs if isinstance(declared_outputs, Mapping) else {}
    process_refs = _process_blob_refs(source_envelope)
    declared_output_refs = _declared_output_refs(source_envelope)
    payloads: dict[str, bytes] = {}
    for ref in [*process_refs, *declared_output_refs]:
        blob_id = ref["blob_id"]
        if (
            output_projection_present
            and str(output_blobs.get(blob_id) or "") != blob_id
        ):
            raise RuntimeError(
                f"remote result {ref['name']} is not a declared output blob"
            )
        if blob_id in payloads:
            continue
        if imported_bytes + ref["size"] > _REMOTE_RESULT_IMPORT_MAX_BYTES:
            raise RuntimeError("remote result import exceeds aggregate byte limit")
        payload = bytes(fetch_blob(blob_id, ref["size"]))
        if len(payload) != ref["size"] or sha256(payload).hexdigest() != blob_id:
            raise RuntimeError(
                f"remote process {ref['name']} failed integrity verification"
            )
        imported_bytes += len(payload)
        payloads[blob_id] = payload
    return envelope, {
        "externalized_envelope": external_payload,
        "process_blobs": payloads,
    }


# ── the Home seam ───────────────────────────────────────────────────────


def home_importer_for(transport: Any) -> Any:
    """Return the injected Home importer, or fail loudly if none was given.

    The transport is constructed with this protocol; a missing importer is a
    wiring bug, never a reason to import a Home authority from here.
    """

    importer = getattr(transport, "home_importer", None)
    if importer is None:
        raise RuntimeError("Home completion importer is unavailable")
    return importer


def complete_transport_import(
    transport: Any,
    context: Mapping[str, Any],
    wire_result: Mapping[str, Any],
    envelope: Mapping[str, Any],
    fetched: Mapping[str, Any],
) -> dict[str, Any]:
    """Hand one verified result to Home through the injected importer."""

    validator = context.get("validator")
    kind = str(context.get("import_kind") or "")
    if kind:
        imported = home_importer_for(transport).complete_import(
            kind=kind,
            context=context,
            wire_result=wire_result,
            envelope=envelope,
            fetched=fetched,
        )
    elif callable(validator):
        imported = validator(wire_result, envelope, fetched)
    else:
        raise RuntimeError("Home completion importer is unavailable")
    if not isinstance(imported, dict):
        raise RuntimeError("Home completion importer returned a non-object")
    return imported


def remove_transport_pending(context: Mapping[str, Any]) -> bool:
    """Drop the durable intent; ``False`` keeps tracking instead of guessing."""

    pending = context.get("pending_record")
    if not isinstance(pending, Mapping):
        return True
    try:
        remove_pending_operation(pending)
    except Exception:
        return False
    return True


# ── reconciliation ──────────────────────────────────────────────────────


def _result_unavailable_envelope(request_id: str, operation_id: str) -> dict[str, Any]:
    return {
        "text": _RESULT_UNAVAILABLE_TEXT,
        "diagnostic": {
            "domain": "protocol",
            "code": "remote_result_unavailable",
            "message": _RESULT_UNAVAILABLE_REASON,
            "phase": "finalize",
            "request_id": request_id,
            "operation_id": operation_id,
            "completion": "completed",
            "retryable": False,
            "details": {},
        },
        "process": None,
        "artifacts": [],
        "trace": {"reconciled": True},
    }


def _write_terminal_evidence(
    transport: Any,
    context: Mapping[str, Any],
    *,
    request_id: str,
    operation_id: str,
    prepared_hash: str,
    envelope: Mapping[str, Any],
    retention_cap: int,
) -> str:
    """Durably record a completed-but-unavailable result and prune old evidence.

    Pruning only ever touches retained evidence files: the ``*.pending.json``
    intents in the same directory are live claims and are never candidates.
    """

    request = transport.request
    root = pending_scope_root(
        pathlib.Path(request.drive_root),
        str(request.connection["id"]),
        str(request.project_id),
        str(request.workspace_id),
    )
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    identity = terminal_evidence_identity(request_id, operation_id, prepared_hash)
    path = root / f"{identity}.json"
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "recorded_at_ms": int(time.time() * 1000),
            "connection_id": str(request.connection["id"]),
            "project_id": str(request.project_id),
            "workspace_id": str(request.workspace_id),
            "task_id": str(context.get("task_id") or ""),
            "request_id": request_id,
            "operation_id": operation_id,
            "prepared_hash": prepared_hash,
            "completion": "completed",
            "envelope": dict(envelope),
        },
        fsync=True,
        mode=0o600,
        fsync_directory=True,
    )
    os.chmod(path, 0o600)
    retained = sorted(
        (
            item
            for item in root.glob("*.json")
            if not item.name.endswith(".pending.json")
        ),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    for stale in retained[retention_cap:]:
        try:
            stale.unlink()
        except OSError:
            pass
    return str(path.relative_to(request.drive_root))


def _import_completed_result(
    transport: Any,
    context: Mapping[str, Any],
    stored: Mapping[str, Any],
    row: dict[str, Any],
) -> bool:
    try:
        envelope, fetched = prefetch_remote_result_import(
            stored,
            transport.fetch_blob,
        )
        imported = complete_transport_import(
            transport,
            context,
            stored,
            envelope,
            fetched,
        )
    except Exception as exc:
        row.update(imported=False, import_error=type(exc).__name__)
        return False
    row.update(imported=True, envelope=imported)
    return True


def _import_unavailable_result(
    transport: Any,
    context: Mapping[str, Any],
    row: dict[str, Any],
    *,
    request_id: str,
    operation_id: str,
    prepared_hash: str,
    retention_cap: int,
) -> bool:
    terminal = _result_unavailable_envelope(request_id, operation_id)
    try:
        if context.get("import_kind") != "attachment_stage_v1":
            terminal = complete_transport_import(
                transport,
                context,
                {
                    "completion": "completed",
                    "prepared_hash": prepared_hash,
                    "envelope": terminal,
                    "output_blobs": {},
                },
                terminal,
                {"externalized_envelope": b"", "process_blobs": {}},
            )
        else:
            import_context = context.get("import_context")
            expected = (
                import_context.get("expected_manifest")
                if isinstance(import_context, Mapping)
                else None
            )
            if not isinstance(expected, list):
                raise RuntimeError("attachment import context is unavailable")
        evidence_ref = _write_terminal_evidence(
            transport,
            context,
            request_id=request_id,
            operation_id=operation_id,
            prepared_hash=prepared_hash,
            envelope=terminal,
            retention_cap=retention_cap,
        )
    except Exception as exc:
        row.update(imported=False, import_error=type(exc).__name__)
        return False
    row.update(
        result_unavailable=True,
        imported=True,
        envelope=terminal,
        evidence_ref=evidence_ref,
    )
    return True


def _await_ack(
    transport: Any,
    *,
    request_id: str,
    operation_id: str,
    prepared_hash: str,
    response_seq: int,
    ack_timeout_sec: float,
) -> bool:
    """Send the ACK and report whether the remote side confirmed it."""

    sequence = transport._send(
        "ack",
        ack_seq=response_seq,
        request_id=request_id,
        operation_id=operation_id,
        optional={"prepared_hash": prepared_hash},
    )
    try:
        ack = transport._wait_control(
            lambda item: (
                item.get("kind") in {"ack", "diagnostic"}
                and (
                    item.get("ack_seq") == sequence
                    or (
                        item.get("request_id") == request_id
                        and item.get("operation_id") == operation_id
                    )
                )
            ),
            timeout_sec=ack_timeout_sec,
        )
    except Exception:
        return False
    return ack.get("kind") == "ack"


def reconcile_remote_operations(
    transport: Any,
    *,
    ack_timeout_sec: float,
    retention_cap: int,
) -> list[dict[str, Any]]:
    """Import, durably fix and ACK the transport's bounded operation ledger."""

    rows: list[dict[str, Any]] = []
    contexts = getattr(transport, "_operation_contexts", None)
    if contexts is None:
        contexts = {}
        transport._operation_contexts = contexts
    for (request_id, operation_id), prepared_hash in list(
        transport._known_operations.items()
    ):
        transport._send(
            "reconcile",
            request_id=request_id,
            operation_id=operation_id,
            prepared_hash=prepared_hash,
        )
        response = transport._wait_control(
            lambda item: (
                item.get("kind") == "reconcile_result"
                and item.get("request_id") == request_id
                and item.get("operation_id") == operation_id
            )
        )
        row = dict(response)
        reconciled = response.get("result")
        reconciled = reconciled if isinstance(reconciled, dict) else {}
        completion = str(
            reconciled.get("completion") or response.get("completion") or ""
        )
        key = (request_id, operation_id)
        context = contexts.get(key, {})
        row.update(completion=completion, task_id=str(context.get("task_id") or ""))
        should_ack = False
        if completion == "completed":
            stored = reconciled.get("result")
            if isinstance(stored, dict):
                should_ack = _import_completed_result(
                    transport, context, stored, row
                )
            elif bool(reconciled.get("result_unavailable")):
                should_ack = _import_unavailable_result(
                    transport,
                    context,
                    row,
                    request_id=request_id,
                    operation_id=operation_id,
                    prepared_hash=prepared_hash,
                    retention_cap=retention_cap,
                )
        elif completion == "not_started":
            if remove_transport_pending(context):
                transport._known_operations.pop(key, None)
                contexts.pop(key, None)
            else:
                row["cleanup_pending"] = True
        if should_ack and _await_ack(
            transport,
            request_id=request_id,
            operation_id=operation_id,
            prepared_hash=prepared_hash,
            response_seq=int(response["seq"]),
            ack_timeout_sec=ack_timeout_sec,
        ):
            if remove_transport_pending(context):
                transport._known_operations.pop(key, None)
                contexts.pop(key, None)
            else:
                row["cleanup_pending"] = True
        rows.append(row)
    return rows
