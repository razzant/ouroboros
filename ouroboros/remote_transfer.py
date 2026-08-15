"""Home transfer-service seam for remote workspace bytes (RWS v2 §3.2).

ONE Home import executor for every blob kind that crosses the remote boundary.
This is the HOME half of the ``remote_finalization`` split: the transport decides
whether returned bytes are the bytes that were declared, and everything past that
point happens here, because none of it may happen on the transport side.

The pipeline is two steps on purpose, and the split is what makes a crash
recoverable rather than ambiguous:

1. the service stops at a verified Home temp file plus a typed PRIVATE
   :class:`ImportReceipt` (RWSB-03) — proven bytes, written-down provenance, and
   nothing public yet;
2. canonical publication goes through the EXISTING artifact authority via
   ``artifacts.publish_verified_task_artifact`` (RWSB2-02), which is deterministic
   per ``{task_id, import_id, canonical_name}``, streams without rereading, and
   returns the existing record on a same-hash replay.

No parallel artifact identity or lifecycle authority exists here (D9): the receipt
keeps the target-native provenance privately, and the published record is the sole
public identity the model, the CLI and the review evidence ever see. Redaction
happens BEFORE publication, so the record's hash binds the bytes that are actually
in it rather than the source digest of content nobody stored.

The transport depends on the narrow :class:`HomeImporter` / :class:`PendingJournal`
protocols, injected at broker construction — it never imports ``artifacts``,
``observability`` or this module. The default injection happens at the TOP of the
graph (``RemoteSessionBroker.__init__``) rather than being reached for from below,
because a function-local import of this module inside the transport is exactly the
upward arrow that forced the donor to mirror Home guards in its remote path.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import pathlib
import shutil
import tempfile
from hashlib import sha256
from typing import Any, Mapping, Protocol

from ouroboros import artifacts
from ouroboros.export_policy_contract import EXPORT_REASONS
from ouroboros.remote_contracts import refuse_unknown_members
from ouroboros.remote_protocol import IMPORT_CHANNELS
from ouroboros.remote_refusal_actions import ACTION_RETRY, REFUSAL_ACTIONS
from ouroboros.task_results import validate_task_id
# The PRODUCER's caps, so Home's acceptance ceilings below cannot drift from what a
# well-behaved target actually emits. Home reading a native module is the permitted
# direction; the reverse is what the import-closure gate forbids.
from ouroboros.workspace_snapshot_native import (
    MAX_SNAPSHOT_BYTES as TARGET_MAX_SNAPSHOT_BYTES,
    MAX_SNAPSHOT_FILES as TARGET_MAX_SNAPSHOT_FILES,
)

# Typed unavailability code for the not-yet-landed transport branches.

_PROCESS_STREAM_NAMES = frozenset({"stdout.txt", "stderr.txt"})
# Home decides its own bounds. A remote envelope must not be able to set the size
# of a Home record, so the trace, the artifact list and the model preview are all
# capped here rather than trusted from the wire.
_MODEL_PREVIEW_CHARS = 64_000
_HOME_ARTIFACT_LIMIT = 128
_HOME_TRACE_KEYS = 128
_HOME_TRACE_VALUE_BYTES = 256 * 1024


@dataclasses.dataclass(frozen=True)
class ImportReceipt:
    """The typed PRIVATE receipt emitted once per imported object (§3.2).

    Never exposed publicly: ``source_path`` (target-native provenance) and the
    transport identities live here and only here — the published artifact
    record carries none of them (D9).
    """

    import_id: str          # stable identity of this import operation
    task_id: str            # owning task
    kind: str               # closed channel/blob kind (Appendix C-1 registry)
    connection_id: str      # source connection identity
    workspace_id: str       # source workspace identity
    source_path: str        # target-NATIVE source path (private provenance)
    sha256: str             # verified content hash of the imported bytes
    size: int               # verified byte size
    excluded: tuple[str, ...]   # disclosed policy-exclusion manifest (bounded)
    excluded_count: int     # EXACT number of policy-omitted entries
    transport_op_id: str    # transport / pending-operation identity
    artifact_name: str = ""  # canonical imported-artifact name (set at publish)
    home_ref: str = ""       # final Home artifact path (set at publish)


class HomeImporter(Protocol):
    """Home-side sink the transport hands verified bytes to.

    The transport verifies size/hash against the wire envelope, writes the
    payload to a Home temp file, and calls this — it never publishes bytes
    itself and never imports Home authorities.
    """

    def stage_verified_blob(
        self, receipt: ImportReceipt, verified_tmp_path: pathlib.Path
    ) -> ImportReceipt: ...

    def complete_import(
        self,
        *,
        kind: str,
        context: Mapping[str, Any],
        wire_result: Mapping[str, Any],
        envelope: Mapping[str, Any],
        fetched: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Complete ONE closed import contract for an already-verified result.

        `kind` is a member of the closed channel registry (Appendix C-1); the
        transport has already proved every declared blob's size and hash.  This
        is the single call through which a verified remote result becomes Home
        state, and the only reason the transport needs a Home object at all.
        """
        ...


class PendingJournal(Protocol):
    """Durable journal of in-flight transfer operations for crash reconciliation.

    Exactly the two calls the transfer service makes. Enumeration of what is still
    pending is deliberately NOT a member: recovery reads the durable ``*.pending.json``
    evidence directly (``remote_reconciliation.recover_pending_scopes``), because after
    a crash the process holding an in-memory journal is the thing that died. A third
    ``pending_operation_ids`` member existed here with no production caller, which
    made the protocol describe a recovery route nobody took.
    """

    def record_pending(self, op_id: str, payload: Mapping[str, Any]) -> None: ...

    def resolve_pending(self, op_id: str) -> None: ...


class RemoteTransferService:
    """The single Home import/export executor (§3.2).

    `complete_import` is the one entry every returned blob kind routes through:
    the transport has already proved each declared blob's size and hash, and this
    side does what must NOT happen on the transport side — redaction, the returned
    manifest's policy check, publication through the artifact authority, and the
    private receipt that keeps remote provenance out of the public record (D9).
    `stage_verified_blob` and `publish_import` are the two halves of that
    publication, split so a crash between them is recoverable rather than ambiguous.
    """

    def __init__(self, journal: PendingJournal | None = None):
        self._journal = journal

    def complete_import(
        self,
        *,
        kind: str,
        context: Mapping[str, Any],
        wire_result: Mapping[str, Any],
        envelope: Mapping[str, Any],
        fetched: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Turn one verified remote result into Home state (Home import half).

        The transport has already proved every declared blob's size and hash; what
        happens here is the part that must NOT happen on the transport side —
        redaction, publication through the artifact authority, and the private
        receipt that keeps the remote provenance out of the public record (D9).

        `kind` is a member of the closed channel registry. An unknown kind is a
        refusal, not a best-effort import: a blob whose channel nobody declared has
        no policy attached to it.
        """

        channel = str(kind or "").strip()
        if channel not in IMPORT_CHANNELS:
            # The closed import-channel registry is a Home↔execd contract like the
            # export document: an unknown kind is a blob one half declared and the
            # other has never heard of. Same typed refusal, same owner action, and
            # it still fails closed — an undeclared channel has no policy attached.
            refuse_unknown_members(
                "import_channel",
                unknown=[channel],
                understood=IMPORT_CHANNELS,
                member="import channels",
            )
        # Judged as the RAW value, before `pathlib.Path` can swallow it. This read
        # `if not str(drive_root)` over an already-constructed Path, and
        # `pathlib.Path("")` is `PosixPath('.')` whose `str()` is `"."` — truthy — so
        # the guard could not fire for the one input it named, and an absent
        # `drive_root` would have staged the import into the CURRENT WORKING DIRECTORY
        # instead of refusing. Not remotely reachable (the context is built from
        # Home's own drive root) and that is exactly why it was invisible: a vacuous
        # guard looks identical from the outside to one that works.
        raw_drive_root = str(context.get("drive_root") or "").strip()
        task_id = str(context.get("task_id") or "")
        if not raw_drive_root or not task_id:
            raise ValueError("Home import requires a drive_root and a task_id")
        drive_root = pathlib.Path(raw_drive_root)
        del wire_result
        import_context = context.get("import_context")
        if channel == "attachment_stage_v1":
            # The OUTGOING channel's result. Nothing is imported: the bytes went the
            # other way and the answer is a manifest, so what this owes Home is proof
            # that the target staged exactly the authorized set — not a publication.
            # Routing it through the result importer would have published an envelope
            # as a task artifact and called the staging verified because it parsed.
            return _completed_attachment_stage(envelope, fetched, import_context)
        return _import_remote_result(
            self,
            drive_root,
            task_id,
            operation_id=str(context.get("operation_id") or ""),
            connection_id=str(context.get("connection_id") or ""),
            workspace_id=str(context.get("workspace_id") or ""),
            channel=channel,
            envelope=envelope,
            fetched=fetched,
            export_policy=(
                import_context.get("export_policy")
                if isinstance(import_context, Mapping)
                else None
            ),
        )

    def export_operation(
        self,
        executor: Any,
        operation: str,
        args: Mapping[str, Any],
        *,
        blobs: Mapping[str, bytes] | None = None,
        task_id: str = "",
        import_kind: str = "",
        import_context: Mapping[str, Any] | None = None,
        echo_args: bool = True,
        timeout_sec: float | None = None,
    ) -> Any:
        """Run ONE Home-initiated native operation, carrying Home bytes OUTWARD.

        The export half of the service, and the reason it lives here rather than in
        each channel module: a `blobs=` call site outside this service is a channel
        that reaches around the boundary, which is the postmortem's "one policy × N
        doors" shape and is what the closed-registry grep-proof exists to prevent.
        Channels decide WHAT may leave (policy, manifest, disclosure); this moves it.

        ``echo_args`` asserts the target prepared the arguments Home sent, and aborts
        the prepared operation when it did not: an operation whose token binds a
        payload Home never composed must be withdrawn, not executed, and leaving it
        prepared would hold the target's reserved blobs until expiry.
        """

        from ouroboros.remote_protocol import canonical_json
        from ouroboros.workspace_diagnostics import RemoteWorkspaceError
        from ouroboros.workspace_executor import (
            abort_prepared_operation,
            execute_prepared,
            prepare_native_operation,
        )

        payload = dict(args)
        prepared = prepare_native_operation(
            executor,
            str(operation),
            args=payload,
            blobs=dict(blobs or {}),
            task_id=str(task_id or ""),
        )
        if echo_args and canonical_json(
            getattr(prepared, "execution_args", {}) or {}
        ) != canonical_json(payload):
            abort_prepared_operation(
                executor, prepared, task_id=task_id, reason="export_arguments_changed"
            )
            raise RemoteWorkspaceError(
                "export_arguments_changed",
                f"The target prepared different arguments for {operation!r} than Home sent.",
                phase="authorize",
            )
        return execute_prepared(
            executor,
            prepared,
            task_id=str(task_id or ""),
            timeout_sec=timeout_sec,
            import_kind=str(import_kind or ""),
            import_context=dict(import_context or {}),
        )

    def publish_import(
        self,
        drive_root: pathlib.Path | str,
        receipt: ImportReceipt,
        verified_tmp_path: pathlib.Path | str,
        *,
        canonical_name: str,
    ) -> ImportReceipt:
        """Publish a VERIFIED temp file through the artifact authority.

        Fully local and live: delegates to
        ``artifacts.publish_verified_task_artifact`` (deterministic
        destination, atomic streaming publish, idempotent replay, loud hash
        conflict) and returns the receipt completed with the public artifact
        identity. The receipt itself stays private.
        """

        record = artifacts.publish_verified_task_artifact(
            drive_root,
            receipt.task_id,
            receipt.import_id,
            canonical_name,
            verified_tmp_path,
            size=receipt.size,
            sha256=receipt.sha256,
        )
        if self._journal is not None and receipt.transport_op_id:
            self._journal.resolve_pending(receipt.transport_op_id)
        return dataclasses.replace(
            receipt,
            artifact_name=str(record.get("name") or ""),
            home_ref=str(record.get("path") or ""),
        )

    def stage_verified_blob(
        self, receipt: ImportReceipt, verified_tmp_path: pathlib.Path
    ) -> ImportReceipt:
        """Record the private receipt for bytes that are verified but not yet public.

        The service STOPS here on purpose (RWSB-03): the bytes are on Home and
        proven, the provenance is written down, and NOTHING public exists yet.
        Publication is a separate, idempotent step, which is what makes a crash
        between the two recoverable rather than ambiguous.
        """

        del verified_tmp_path
        if self._journal is not None and receipt.transport_op_id:
            self._journal.record_pending(
                receipt.transport_op_id,
                {"import_id": receipt.import_id, "task_id": receipt.task_id, "kind": receipt.kind},
            )
        return receipt


def _completed_attachment_stage(
    envelope: Mapping[str, Any],
    fetched: Mapping[str, Any],
    import_context: Any,
) -> dict[str, Any]:
    """Verify one attachment-staging result against the manifest Home authorized.

    Refuses any returned bytes: on this channel there is nothing for an externalized
    result or an output blob to be, so their presence means the target answered a
    different question. The verified target-side manifest replaces the trace's copy,
    which is what makes the entries downstream the ones Home checked rather than the
    ones the wire happened to carry.
    """

    from ouroboros.remote_task_files import validate_staged_attachment_envelope

    if fetched.get("externalized_envelope") or fetched.get("process_blobs"):
        raise RuntimeError(
            "attachment staging returned bytes; this channel only carries a manifest"
        )
    expected = (
        import_context.get("expected_manifest") if isinstance(import_context, Mapping) else None
    )
    if not isinstance(expected, list):
        raise RuntimeError("attachment import context is unavailable")
    staged = validate_staged_attachment_envelope(expected, dict(envelope))
    result = dict(envelope)
    result["trace"] = {
        **(dict(envelope["trace"]) if isinstance(envelope.get("trace"), Mapping) else {}),
        "attachment_manifest": staged,
    }
    return result


# ── the Home half of the pending-operation recovery split ────────────────────


def _scope_row(group: Mapping[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    return {
        "connection_id": str(group["connection_id"]),
        "project_id": str(group["project_id"]),
        "workspace_id": str(group["workspace_id"]),
        "pending_count": len(group["records"]),
        "status": status,
        **extra,
    }


def recover_pending_scopes(broker: Any) -> list[dict[str, Any]]:
    """Reopen each durable pending scope and walk it to a conclusion.

    The Home half of the recovery split. It is a HOOK rather than something the
    broker does itself because deciding whether a scope still means what it meant
    requires Home authorities — the connection store and the project registry —
    which the broker must not import.

    The rules are deliberately conservative, because the alternative to caution
    here is repeating a mutation:

    * a connection that is gone or retired is NOT reopened. Its records stay on
      disk and are reported, because the owner retired the connection and silently
      reconciling against a host they revoked is worse than an outstanding claim.
    * reopening is admission, so it goes through the same broker door as ordinary
      work; a target whose identity changed refuses there, as it should.
    * reconciliation itself is the transport's, and it is not a retry engine: a
      proven `not_started` drops the intent, `completed` imports and ACKs, and
      `completed` with an unavailable result becomes durable terminal evidence.
    """

    from ouroboros.connection_store import get_connection
    from ouroboros.remote_pending_operations import pending_operation_groups

    rows: list[dict[str, Any]] = []
    for group in pending_operation_groups(pathlib.Path(broker.drive_root)):
        connection = get_connection(str(group["connection_id"]), include_retired=True)
        if connection is None or str(connection.get("lifecycle") or "") != "active":
            rows.append(_scope_row(
                group,
                "scope_retired",
                error={
                    "code": "connection_retired",
                    "message": (
                        "Durable remote operations belong to a connection the owner "
                        "retired; they are retained, not reconciled."
                    ),
                    "phase": "finalize",
                    "completion": "unknown",
                    "retryable": False,
                    # The recovery report is a REFUSAL like any other, so it names
                    # the one action that removes it, from the same register. It
                    # used to name none, and a retired connection is precisely the
                    # case where an owner staring at outstanding operations has no
                    # way to guess that only pointing the project at an ACTIVE
                    # connection will ever clear them.
                    "action": REFUSAL_ACTIONS["connection_retired"],
                    "details": {"action": REFUSAL_ACTIONS["connection_retired"]},
                },
            ))
            continue
        try:
            broker.admit_workspace(
                connection,
                remote_root=str(group["remote_root"]),
                project_id=str(group["project_id"]),
                workspace_id=str(group["workspace_id"]),
            )
            reconciled = broker.recover_scope(
                connection_id=str(group["connection_id"]),
                project_id=str(group["project_id"]),
                workspace_id=str(group["workspace_id"]),
            )
        except Exception as exc:
            code = str(getattr(exc, "code", "") or type(exc).__name__)
            # The refusal's OWN action first, then the register for its code, then
            # `retry` — the same ladder `gateway/connections._public_live_fields`
            # walks, spelled once here instead of being omitted. A typed refusal
            # already derived this in its constructor; this row used to drop it.
            action = str(
                getattr(exc, "action", "")
                or REFUSAL_ACTIONS.get(code.strip().lower())
                or ACTION_RETRY
            )
            rows.append(_scope_row(
                group,
                "reconcile_failed",
                error={
                    "code": code,
                    "message": str(exc),
                    "phase": str(getattr(exc, "phase", "finalize") or "finalize"),
                    "completion": str(getattr(exc, "completion", "unknown") or "unknown"),
                    "retryable": bool(getattr(exc, "retryable", True)),
                    "action": action,
                    "details": {"action": action},
                },
            ))
            continue
        rows.append(_scope_row(group, "reconciled", operations=list(reconciled)))
    return rows


# ── the Home import half of the `remote_finalization` split ──────────────────


def _redacted_text_artifact(
    service: RemoteTransferService,
    drive_root: pathlib.Path,
    task_id: str,
    receipt: ImportReceipt,
    text: str,
    *,
    canonical_name: str,
) -> dict[str, Any]:
    """Publish already-redacted TEXT through the artifact authority.

    The published bytes are the REDACTED ones, so the hash that binds the publication
    is the REDACTED hash — anything else would either publish unredacted bytes or bind
    a record to a digest of content that is not in it. `dataclasses.replace` therefore
    overwrites the caller's `sha256`/`size`, and the receipt describes what was
    STORED, matching its own field comment ("verified content hash of the imported
    bytes").

    This used to claim in the same breath that "the source hash survives only in the
    private receipt", which was FALSE: there is one receipt and the replace above is
    what overwrote it, so the source digest survived in no receipt at all. It survives
    where it is actually proven — the transport verified every blob against the
    declared `blob_id`, which IS the source digest, before any of this ran
    (`remote_reconciliation.prefetch_remote_result_import`). Integrity is not weaker
    for it; the sentence was.

    Returns only the public artifact row. It used to return `(receipt, row)` and both
    call sites discarded the receipt.
    """

    payload = text.encode("utf-8")
    digest = sha256(payload).hexdigest()
    staged = dataclasses.replace(receipt, sha256=digest, size=len(payload))
    with tempfile.NamedTemporaryFile(
        dir=str(_import_tmp_dir(drive_root, task_id)), delete=False, suffix=".tmp"
    ) as handle:
        handle.write(payload)
        tmp_path = pathlib.Path(handle.name)
    try:
        service.stage_verified_blob(staged, tmp_path)
        published = service.publish_import(
            drive_root, staged, tmp_path, canonical_name=canonical_name
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    return {
        "name": canonical_name,
        "sha256": published.sha256,
        "size": published.size,
        # D9: the SOLE public identity is the Home artifact. The target-native
        # source path stays in the private receipt and never reaches this dict.
        "home_ref": {"root": "artifact_store", "path": published.artifact_name},
    }


def discard_task_import_staging(drive_root: Any, task_id: str) -> bool:
    """Drop ONE task's ephemeral import staging directory.

    Called when a task's remote lease ends. Staging holds only pre-publication
    temp files — every published artifact already lives in the task artifact store,
    so this can never remove evidence. Scoped to one task on purpose: a broader
    sweep would race a sibling task that is mid-import on the same drive.
    """

    root = str(drive_root or "").strip()
    # ONE authority for what a task id may be, imported rather than restated: the
    # local `"/" in task or ... task in {".", ".."}` spelling agreed with
    # `task_results.validate_task_id` only by coincidence, and the CREATE side
    # (`_import_tmp_dir`) had no check at all — so the same value was refused here and
    # accepted there. A cleanup answers `False` where the creator raises, which is a
    # difference in the ANSWER SHAPE, not in the rule.
    try:
        task = validate_task_id(task_id)
    except ValueError:
        return False
    if not root:
        return False
    target = pathlib.Path(root) / "remote_imports" / task
    if not target.is_dir():
        return False
    shutil.rmtree(target, ignore_errors=True)
    return not target.exists()


def _import_tmp_dir(drive_root: pathlib.Path, task_id: str) -> pathlib.Path:
    """A per-task staging directory ON the drive, so publication is a rename-scale
    copy on one filesystem rather than a cross-device move.

    The task id is validated through the SAME authority the discard side uses
    (`task_results.validate_task_id`), because this is the half that builds a PATH out
    of it and it had no check at all — a `/` or a `..` in the id would have made a
    staging directory somewhere else on the drive. Not remotely reachable today (the
    ids are Home's own), which is why nothing noticed; the refusal is the point, not
    the current reachability.
    """

    staging = pathlib.Path(drive_root) / "remote_imports" / validate_task_id(task_id)
    staging.mkdir(parents=True, exist_ok=True, mode=0o700)
    return staging


def _bounded_trace(raw: Any, *, full_envelope_ref: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep the trace usable without letting a remote envelope define Home's size."""

    from ouroboros.remote_protocol import canonical_json

    if not isinstance(raw, Mapping):
        return {}
    rows = list(raw.items())
    bounded: dict[str, Any] = {}
    omitted: list[str] = []
    for key, value in rows[:_HOME_TRACE_KEYS]:
        try:
            encoded = canonical_json(value)
        except (TypeError, ValueError):
            encoded = b""
        if encoded and len(encoded) <= _HOME_TRACE_VALUE_BYTES:
            bounded[str(key)] = value
        else:
            omitted.append(str(key))
    if len(rows) > _HOME_TRACE_KEYS:
        bounded["externalized_trace_keys_omitted"] = len(rows) - _HOME_TRACE_KEYS
    if omitted:
        bounded["externalized_trace_values_omitted"] = omitted
    if (omitted or len(rows) > _HOME_TRACE_KEYS) and full_envelope_ref:
        bounded["externalized_trace_full_ref"] = dict(full_envelope_ref)
    return bounded


def _validated_export_disclosure(
    export_policy: Mapping[str, Any] | None,
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the returned manifests against Home's policy; refuse on mismatch.

    Two asymmetric failures, both loud:

    * a trace that claims a `policy_hash` while Home bound no policy means bytes
      were filtered by rules Home never authored — nothing can say what is missing;
    * a manifest whose hash or exported paths disagree with Home's document means
      the source did not apply the policy, which
      :func:`remote_export_policy.validate_returned_manifest` raises on.

    An operation with neither is the ordinary "nothing policy-bearing here" case
    and yields the empty disclosure.
    """

    from ouroboros.remote_export_policy import (
        ExportPolicy,
        ExportPolicyViolation,
        export_policy_hash,
        merge_export_disclosures,
        normalize_export_policy,
        validate_operation_trace,
    )

    from ouroboros.export_policy_contract import MANIFEST_TRACE_KEYS

    claims_policy = any(
        isinstance(trace.get(key), Mapping) and trace[key].get("policy_hash")
        for key in MANIFEST_TRACE_KEYS
    )
    if not export_policy:
        if claims_policy:
            raise ExportPolicyViolation(
                "the target filtered this export under a policy Home never issued"
            )
        return merge_export_disclosures(())
    document = normalize_export_policy(export_policy)
    policy = ExportPolicy(
        channel=str(document.get("channel") or ""),
        document=document,
        policy_hash=export_policy_hash(document),
    )
    return validate_operation_trace(policy, trace)


def _import_remote_result(
    service: RemoteTransferService,
    drive_root: pathlib.Path,
    task_id: str,
    *,
    operation_id: str,
    connection_id: str,
    workspace_id: str,
    channel: str,
    envelope: Mapping[str, Any],
    fetched: Mapping[str, Any],
    export_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Redact the verified bytes, publish them, and return the PUBLIC result.

    Three properties the caller depends on:

    * the returned dict carries no target-native path — the provenance lives in the
      private receipts (D9), so the model, the CLI and the review evidence all see
      one artifact identity;
    * process output is published as its own artifact rather than inlined, and the
      model-facing text is bounded with a pointer to it;
    * publication is idempotent per `{task_id, operation_id, canonical_name}`, so a
      replay after a crash between staging and publishing returns the existing
      record instead of a second one.
    """

    from ouroboros.observability import redact_projection, write_call_manifest
    from ouroboros.remote_reconciliation import (
        _declared_output_refs,
        _process_blob_refs,
        _strict_remote_envelope,
    )

    external = fetched.get("externalized_envelope")
    source_redaction = redact_projection(
        _strict_remote_envelope(bytes(external))
        if isinstance(external, (bytes, bytearray)) and external
        else dict(envelope)
    )
    source = dict(source_redaction.value)
    blobs = fetched.get("process_blobs")
    blobs = blobs if isinstance(blobs, Mapping) else {}
    # BEFORE anything is published: re-evaluate Home's own policy over the paths
    # that came back. A returned entry the policy excludes means source-side
    # filtering did not run, and refusing here is the point — publishing first and
    # filtering after would be exactly the leak the policy exists to prevent.
    disclosure = _validated_export_disclosure(
        export_policy, source.get("trace") if isinstance(source.get("trace"), Mapping) else {}
    )

    def _receipt(source_path: str, source_sha256: str, source_size: int) -> ImportReceipt:
        return ImportReceipt(
            import_id=operation_id,
            task_id=task_id,
            kind=channel,
            connection_id=connection_id,
            workspace_id=workspace_id,
            source_path=source_path,
            sha256=source_sha256,
            size=source_size,
            excluded=tuple(
                str(row.get("path") or "")
                for row in disclosure.get("excluded") or []
                if isinstance(row, Mapping)
            ),
            excluded_count=int(disclosure.get("excluded_count") or 0),
            transport_op_id=operation_id,
        )

    imported: list[dict[str, Any]] = []
    full_envelope_ref: dict[str, Any] | None = None
    if isinstance(external, (bytes, bytearray)) and external:
        full_envelope_ref = _redacted_text_artifact(
            service,
            drive_root,
            task_id,
            _receipt("operation-envelope.json", sha256(bytes(external)).hexdigest(), len(external)),
            json.dumps(source, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")),
            canonical_name=f"remote-{operation_id[:16]}-operation-envelope.json",
        )
        imported.append(full_envelope_ref)

    for ref in _process_blob_refs(source):
        raw = blobs.get(ref["blob_id"])
        if not isinstance(raw, bytes):
            raise RuntimeError(f"Home import omitted remote process {ref['name']}")
        if len(raw) != ref["size"] or sha256(raw).hexdigest() != ref["blob_id"]:
            raise RuntimeError(f"Home import could not reverify remote process {ref['name']}")
        stream = str(ref["name"]).removesuffix(".txt")
        row = _redacted_text_artifact(
            service,
            drive_root,
            task_id,
            _receipt(str(ref["name"]), str(ref["sha256"]), int(ref["size"])),
            str(redact_projection(raw.decode("utf-8", errors="replace")).value),
            canonical_name=f"remote-{operation_id[:16]}-{stream}.txt",
        )
        imported.append({**row, "name": str(ref["name"]), "truncated": bool(ref["truncated"])})

    for index, ref in enumerate(_declared_output_refs(source)):
        raw = blobs.get(ref["blob_id"])
        if not isinstance(raw, bytes):
            raise RuntimeError(f"Home import omitted remote output {ref['name']}")
        if len(raw) != ref["size"] or sha256(raw).hexdigest() != ref["blob_id"]:
            raise RuntimeError(f"Home import could not reverify remote output {ref['name']}")
        suffix = pathlib.PurePosixPath(str(ref["name"])).suffix[:20]
        receipt = _receipt(str(ref["name"]), str(ref["sha256"]), int(ref["size"]))
        canonical = f"remote-{operation_id[:16]}-output-{index}{suffix}"
        with tempfile.NamedTemporaryFile(
            dir=str(_import_tmp_dir(drive_root, task_id)), delete=False, suffix=".tmp"
        ) as handle:
            handle.write(raw)
            tmp_path = pathlib.Path(handle.name)
        try:
            service.stage_verified_blob(receipt, tmp_path)
            published = service.publish_import(
                drive_root, receipt, tmp_path, canonical_name=canonical
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        imported.append({
            "name": str(ref["name"]),
            "sha256": published.sha256,
            "size": published.size,
            "declared_as": str(ref["declared_as"]),
            "member_path": str(ref["member_path"]),
            "home_ref": {"root": "artifact_store", "path": published.artifact_name},
        })

    text = str(source.get("text") or "")
    if len(text) > _MODEL_PREVIEW_CHARS:
        text = (
            text[:_MODEL_PREVIEW_CHARS]
            + "\n… remote result preview bounded; full redacted output is in task artifacts"
        )
    trace = _bounded_trace(source.get("trace"), full_envelope_ref=full_envelope_ref)
    # These two keys name BLOB IDs on the target: they are transport bookkeeping and
    # would be a remote identity leaking into the public record.
    trace.pop("externalized_result", None)
    trace.pop("output_blobs", None)
    # The bound is applied AFTER the eligibility filter, so the number we disclose is
    # the number of artifacts a reader would otherwise never learn existed. Slicing the
    # RAW list first made the drop both invisible and undercounted, because filtered
    # rows silently consumed budget (BIBLE P1: no silent elision on a model surface).
    eligible = [
        dict(item)
        for item in list(source.get("artifacts") or [])
        if isinstance(item, Mapping)
        and str(item.get("name") or "") not in _PROCESS_STREAM_NAMES | {"operation-envelope.json"}
        and item.get("kind") != "declared_output"
    ]
    kept = eligible[: max(0, _HOME_ARTIFACT_LIMIT - len(imported))]
    undisclosed_artifacts = len(eligible) - len(kept)
    if undisclosed_artifacts:
        text = (
            f"{text}\n\n⚠️ OMISSION NOTE: {undisclosed_artifacts} of {len(eligible)} "
            f"returned artifact(s) are not listed — this result's artifact list is "
            f"bounded at {_HOME_ARTIFACT_LIMIT} entries."
        ).strip()
        trace["artifacts_undisclosed_count"] = undisclosed_artifacts
    if int(disclosure.get("excluded_count") or 0):
        # The model is told, in words, that this export was filtered — the same fact
        # the ledger and the CLI get. A partial that only shows up in a trace key is
        # a partial the model will reason past.
        from ouroboros.remote_export_policy import disclosure_summary_line

        text = f"{text}\n\n{disclosure_summary_line(disclosure)}".strip()
    result = {
        "text": text,
        "diagnostic": dict(source["diagnostic"]) if isinstance(source.get("diagnostic"), Mapping) else None,
        "process": dict(source["process"]) if isinstance(source.get("process"), Mapping) else None,
        "artifacts": kept + imported,
        "trace": trace,
        # Additive (D7): the disclosed omission travels as its own key so
        # `artifact_bundle`, the verification ledger, task-acceptance evidence and
        # the CLI can all read ONE block. No terminal status is touched.
        "remote_export": disclosure,
    }
    manifest_ref = write_call_manifest(
        drive_root,
        task_id=task_id,
        call_id=f"remote_result_{operation_id}",
        manifest={
            "call_type": "remote_result_import",
            "operation_id": operation_id,
            "channel": channel,
            "full_payload_redacted": True,
            "artifacts": imported,
            "result": result,
        },
    )
    trace["observability_ref"] = {
        "call_id": manifest_ref["call_id"],
        "sha256": manifest_ref["sha256"],
    }
    return result


# ── snapshot materialization: the Home mirror of a remote workspace ──────────
#
# Home materialization of a remote workspace snapshot (RWS v2 §3.2, D7).
#
# Three Home faculties are unavoidably filesystem-shaped: the Claude Agent SDK edits
# files, plan review reads them, and the subagent-patch integration has to know what a
# patch would DO before it is allowed to touch the target. None of them can be taught
# to speak a wire protocol, and none of them may be handed a Home path that pretends to
# be a target path. So the target's tree is materialized here, once, into a temporary
# mirror, and the faculties work on the mirror.
#
# What makes it a mirror rather than a copy is verification. The manifest arrives first
# and every fetch is authorized by an entry in it; each blob is accepted only against
# its own declared size and SHA-256; and when the walk finishes, two fingerprints are
# recomputed from the bytes that actually landed and compared with the manifest's own.
# A mirror that cannot prove it is the target is not a mirror, and reviewing it would be
# reviewing a guess.
#
# D7 is the reason `complete=False` is NOT an error here. A policy exclusion is
# disclosed work: the mirror is missing the excluded paths, it says exactly which and
# why, and every consumer carries that disclosure forward — a `.env` in a remote repo
# must not take plan review, the file bridge and the edit bridge down with it, which is
# precisely what the donor's fail-closed conflation did. An INTEGRITY failure is the
# opposite and stays fail-closed: an unstable observation, a walk error or a partial
# read means nobody can say what the tree was, and a mirror built on that would be a
# confident answer about a state that never existed.
#
# The lifecycle is explicit and never lives on a context object. `materialize` returns a
# `RemoteWorkspaceSnapshot` context manager the CALLER owns; the donor cached it on
# `ctx` as `_remote_plan_review_snapshot`, which made "is there a snapshot" a property of
# a mutable attribute two unrelated modules read and one deleted.


# Home's ACCEPTANCE ceilings, DERIVED from the target's production caps rather than
# restated beside them. They used to be independent numbers under the very same names
# the producer uses (`MAX_SNAPSHOT_FILES`/`MAX_SNAPSHOT_BYTES` in
# `workspace_snapshot_native`), and they disagreed in both directions: 20_000 files
# against the target's 25_000 — so a CLEAN snapshot of a 22k-file workspace was refused
# here as "exceeds the Home file limit", with nothing about the target to fix — and
# 512 MiB against the target's 256 MiB, a bound that could never bind. Accepting
# exactly what a well-behaved target can produce is the only relationship that is not
# either a false refusal or a dead ceiling; a target that exceeds its OWN cap records a
# `*_limit_exceeded` failure, which `_validated_manifest` refuses on its own terms.
MAX_ACCEPTED_SNAPSHOT_FILES = TARGET_MAX_SNAPSHOT_FILES
MAX_ACCEPTED_SNAPSHOT_BYTES = TARGET_MAX_SNAPSHOT_BYTES
SNAPSHOT_EXPORT_OPERATION = "snapshot_manifest_and_blob_export"
SNAPSHOT_EXPORT_CHANNEL = "workspace_snapshot"
# The reasons that mean "the owner's policy omitted this", as opposed to an integrity
# failure: a consumer DISCLOSES these rather than refusing over them.
#
# IMPORTED, not restated. The hand-written set held three of the four — no
# `sensitive_component` — so such a row would have been dropped from `exclusions()`,
# and with it the owner's omission note, the plan-review "names a withheld path"
# refusal and the patch bridge's `blocked` check, all while `partial` stayed True.
# Unreachable today (that reason is deliverable-profiled and both snapshot channels are
# tree-profiled), which is exactly how a restated list survives: the comment above it
# had already drifted to "the two that a consumer must disclose" over a three-member
# set, and nothing noticed.
_POLICY_REASONS = frozenset(EXPORT_REASONS)


class RemoteSnapshotError(RuntimeError):
    """A materialization that cannot be proven to be the target's tree."""

    code = "REMOTE_SNAPSHOT_UNVERIFIED"


@dataclasses.dataclass
class RemoteWorkspaceSnapshot(contextlib.AbstractContextManager):
    """One verified mirror of a remote workspace, owned by its caller."""

    root: pathlib.Path
    manifest: dict[str, Any]
    _cleanup_root: pathlib.Path
    # The EXACT policy document this mirror was taken under. Carried rather than
    # recomputed because a consumer that re-walks the mirror (the patch bridge must,
    # to learn what an edit did) has to apply the same rules the target applied — a
    # second document would make Home and the target disagree about the tree.
    policy_document: dict[str, Any] = dataclasses.field(default_factory=dict)
    closed: bool = False

    @property
    def partial(self) -> bool:
        """D7's derived state: filtered by policy but whole in itself."""

        return not bool(self.manifest.get("complete", True)) and bool(
            self.manifest.get("integrity_complete", True)
        )

    def exclusions(self) -> list[dict[str, str]]:
        """The disclosed policy omissions, as ``{"path", "reason"}`` rows."""

        rows = self.manifest.get("policy_exclusions") or self.manifest.get("exclusions") or []
        return [
            {"path": str(row.get("path") or ""), "reason": str(row.get("reason") or "")}
            for row in rows
            if isinstance(row, Mapping) and str(row.get("reason") or "") in _POLICY_REASONS
        ]

    def omission_note(self) -> str:
        """One honest sentence naming what the mirror does NOT contain, or ``""``.

        Every consumer renders this rather than inventing its own wording, because a
        reviewer reading three differently-phrased partiality notices has to work out
        whether they mean the same thing.
        """

        rows = self.exclusions()
        if not rows:
            return ""
        named = ", ".join(f"{row['path']} ({row['reason']})" for row in rows[:20])
        more = f"; +{len(rows) - 20} more" if len(rows) > 20 else ""
        return (
            f"NOTICE: this remote snapshot is POLICY-FILTERED — {len(rows)} path(s) were "
            f"excluded by the owner's export policy and are NOT present in the mirror: "
            f"{named}{more}. Everything else transferred and was verified byte-for-byte."
        )

    def close(self) -> None:
        shutil.rmtree(self._cleanup_root, ignore_errors=True)
        self.closed = True

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def materialize_remote_snapshot(
    ctx: Any,
    *,
    channel: str = SNAPSHOT_EXPORT_CHANNEL,
    max_files: int = MAX_ACCEPTED_SNAPSHOT_FILES,
    max_bytes: int = MAX_ACCEPTED_SNAPSHOT_BYTES,
) -> RemoteWorkspaceSnapshot:
    """Export, fetch, write and VERIFY one remote workspace mirror on Home.

    ``channel`` names the PURPOSE this mirror serves, and it is an argument because a
    manifest's fingerprint includes the hash of the policy that produced it: a mirror
    taken under one channel and a patch applied under another describe different
    filterings, so the target would refuse a correct patch with a fingerprint
    mismatch. One purpose, one document, from the export to the apply.
    """

    from ouroboros.remote_export_policy import build_export_policy, validate_returned_manifest
    from ouroboros.workspace_executor import executor_ref_from_ctx, fetch_native_blob
    from ouroboros.workspace_ref import workspace_ref_for

    ref = workspace_ref_for(ctx)
    if getattr(ref, "kind", "") != "ssh":
        raise RemoteSnapshotError("a remote snapshot requires a sealed ssh placement")
    policy = build_export_policy(ctx, channel=channel, workspace_root=ref.remote_root)
    task_id = str(getattr(ctx, "task_id", "") or "")
    envelope = RemoteTransferService().export_operation(
        executor_ref_from_ctx(ctx),
        SNAPSHOT_EXPORT_OPERATION,
        policy.arg_payload(),
        task_id=task_id,
        # The snapshot operation takes no arguments of its own, so there is nothing
        # for the target to echo: `execution_args` is `{}` while what Home sent is the
        # policy, which prepare strips by convention (underscore-prefixed).
        echo_args=False,
    )
    manifest = _validated_manifest(envelope, max_files=max_files, max_bytes=max_bytes)
    # Home's own policy, re-evaluated over what came back, BEFORE a byte is written:
    # a returned entry the policy excludes proves source-side filtering did not run.
    validate_returned_manifest(policy, manifest)
    executor = executor_ref_from_ctx(ctx)
    temp_root = pathlib.Path(tempfile.mkdtemp(prefix="ouroboros-remote-snapshot-"))
    try:
        mirror = temp_root / "workspace"
        mirror.mkdir(mode=0o700)
        written = _materialize_entries(
            mirror,
            manifest,
            max_bytes=max_bytes,
            fetch=lambda blob_id, size: fetch_native_blob(
                executor, blob_id, max_bytes=size, task_id=task_id
            ),
        )
        _verify_materialized(manifest, written)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return RemoteWorkspaceSnapshot(
        root=mirror,
        manifest=manifest,
        _cleanup_root=temp_root,
        policy_document=dict(policy.document),
    )


def _validated_manifest(
    envelope: Any, *, max_files: int, max_bytes: int
) -> dict[str, Any]:
    """The manifest, or a typed refusal — integrity fail-closed, policy disclosed."""

    trace = getattr(envelope, "trace", None)
    raw = trace.get("snapshot") if isinstance(trace, Mapping) else None
    if not isinstance(raw, Mapping):
        raise RemoteSnapshotError("the target returned no snapshot manifest")
    manifest = dict(raw)
    if int(manifest.get("schema_version") or 0) != 3:
        raise RemoteSnapshotError(
            f"unsupported remote snapshot schema {manifest.get('schema_version')!r}"
        )
    # Integrity, not completeness. `complete=False` with `integrity_complete=True` is
    # D7's disclosed policy filtering and is materialized on purpose; everything below
    # means the target could not say what its tree WAS.
    for field in ("integrity_complete", "materializable"):
        if manifest.get(field) is not True:
            raise RemoteSnapshotError(
                f"remote snapshot is not {field}: nothing can be reviewed against a "
                "tree the target could not observe"
            )
    if manifest.get("unstable") is not False:
        raise RemoteSnapshotError("remote snapshot observed an unstable tree")
    failures = manifest.get("failures")
    if not isinstance(failures, list) or failures:
        raise RemoteSnapshotError(f"remote snapshot reported read failures: {failures}")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) > max(1, int(max_files)):
        raise RemoteSnapshotError("remote snapshot exceeds the Home file limit")
    total = manifest.get("total_bytes")
    if not isinstance(total, int) or isinstance(total, bool) or total < 0 or total > max(1, int(max_bytes)):
        raise RemoteSnapshotError("remote snapshot exceeds the Home byte limit")
    return manifest


def _snapshot_relative(value: Any) -> tuple[str, ...]:
    """A canonical, non-escaping relative path, or a typed refusal."""

    text = str(value or "").replace("\\", "/")
    parts = tuple(part for part in text.split("/") if part)
    if not text or text != "/".join(parts) or any(part in {".", ".."} for part in parts):
        raise RemoteSnapshotError(f"remote snapshot path is not canonical: {value!r}")
    return parts


def _materialize_entries(
    mirror: pathlib.Path,
    manifest: Mapping[str, Any],
    *,
    max_bytes: int,
    fetch: Any,
) -> list[dict[str, Any]]:
    """Write every declared entry, accepting bytes only against their own hash."""

    consumed = 0
    written: list[dict[str, Any]] = []
    for raw in manifest.get("entries") or []:
        if not isinstance(raw, Mapping):
            raise RemoteSnapshotError("remote snapshot entry is not an object")
        parts = _snapshot_relative(raw.get("path"))
        digest = str(raw.get("sha256") or "")
        size = raw.get("size")
        if len(digest) != 64 or not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RemoteSnapshotError(f"remote snapshot entry is invalid: {'/'.join(parts)}")
        consumed += size
        if consumed > max(1, int(max_bytes)):
            raise RemoteSnapshotError("remote snapshot exceeds the Home byte limit")
        target = mirror.joinpath(*parts)
        # Containment is checked on the RESOLVED parent, because an earlier symlink
        # entry could otherwise redirect a later write outside the mirror.
        resolved_parent = target.parent.resolve(strict=False)
        if resolved_parent != mirror.resolve(strict=False) and mirror.resolve(strict=False) not in resolved_parent.parents:
            raise RemoteSnapshotError(f"remote snapshot entry escapes the mirror: {'/'.join(parts)}")
        payload = bytes(fetch(digest, size))
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise RemoteSnapshotError(
                f"remote snapshot blob failed verification: {'/'.join(parts)}"
            )
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        kind = str(raw.get("kind") or "file")
        mode = int(raw.get("mode") or 0o600) & 0o777
        if kind == "symlink":
            link = payload.decode("utf-8", errors="surrogateescape")
            link_parts = tuple(part for part in link.replace("\\", "/").split("/") if part)
            if link.startswith("/") or any(part == ".." for part in link_parts):
                raise RemoteSnapshotError(f"remote snapshot symlink escapes the mirror: {link!r}")
            os.symlink(link, target)
        elif kind == "file":
            target.write_bytes(payload)
            with contextlib.suppress(OSError):
                os.chmod(target, mode)
        else:
            raise RemoteSnapshotError(f"remote snapshot entry kind is invalid: {kind!r}")
        written.append({
            "path": "/".join(parts), "kind": kind, "sha256": digest,
            "size": size, "mode": mode,
        })
    return written


def _verify_materialized(
    manifest: Mapping[str, Any], written: list[dict[str, Any]]
) -> None:
    """Recompute the fingerprints from what LANDED and compare with the manifest.

    This is the step that makes the mirror evidence rather than a copy: the entry
    hashes were checked one at a time, and this proves the SET is the same set — no
    entry silently skipped, none added, and the same git state the target reported.
    """

    declared_total = int(manifest.get("total_bytes") or 0)
    landed_total = sum(int(row["size"]) for row in written)
    if landed_total != declared_total:
        raise RemoteSnapshotError(
            f"materialized {landed_total} bytes against a declared {declared_total}"
        )
    content = hashlib.sha256(
        json.dumps(written, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if content != str(manifest.get("content_fingerprint") or ""):
        raise RemoteSnapshotError("materialized content fingerprint does not match the manifest")
    # The row shape comes from the CONTRACT, not from a second spelling here: this
    # reconstructed two keys while the source wrote three, so adding `judged` turned a
    # legitimately filtered mirror into "fingerprint does not match the manifest".
    from ouroboros.export_policy_contract import MANIFEST_EXCLUSION_ROW_FIELDS

    exclusions = [
        {
            field: str(row.get(field) or (row.get("path") if field == "judged" else ""))
            for field in MANIFEST_EXCLUSION_ROW_FIELDS
        }
        for row in manifest.get("policy_exclusions") or []
        if isinstance(row, Mapping)
    ]
    overall = hashlib.sha256(
        json.dumps(
            {
                "entries": written,
                "git": manifest.get("git"),
                "policy_exclusions": exclusions,
                # The hash of the policy that produced this manifest is PART of its
                # identity: two manifests with identical entries under different rules
                # describe different filterings, and a fingerprint that ignored the
                # policy would call them the same tree.
                "policy_hash": str(manifest.get("policy_hash") or ""),
                "protected_paths": [
                    str(item) for item in manifest.get("protected_paths") or []
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if overall != str(manifest.get("fingerprint") or ""):
        raise RemoteSnapshotError("materialized snapshot fingerprint does not match the manifest")


__all__ = [
    "MAX_ACCEPTED_SNAPSHOT_BYTES",
    "MAX_ACCEPTED_SNAPSHOT_FILES",
    "SNAPSHOT_EXPORT_CHANNEL",
    "SNAPSHOT_EXPORT_OPERATION",
    "HomeImporter",
    "ImportReceipt",
    "PendingJournal",
    "RemoteSnapshotError",
    "RemoteTransferService",
    "RemoteWorkspaceSnapshot",
    "discard_task_import_staging",
    "materialize_remote_snapshot",
    "recover_pending_scopes",
]
