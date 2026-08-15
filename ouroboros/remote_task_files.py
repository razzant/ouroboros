"""The HOME half of the two task-FILE channels (RWS v2 §3.2, D12).

Two directions, one module, because they are the two ends of the same contract
(`execd_task_files`) and the same private cache on the target:

* `attachment_stage` (Home → target): the owner's attached files, so a remote
  `run_command` can open them;
* `media_frames` (target → Home): one file the vision faculties need as BYTES,
  because the brain is on Home (D1) and a target path is not something a vision
  model can look at.

── attachment_stage ─────────────────────────────────────────────────────────

The owner creates a task in a remote project with files attached. Those bytes are
on Home, the work is on the target, and until this module existed the two facts
never met: `execd_task_files` had the whole receiving contract — a content-addressed
manifest, an all-or-nothing private cache, an idempotent replay — and nothing on
Home ever spoke it. An attachment on a remote task was silently a Home-only file.

Three things happen here, in this order, and the order is the design:

1. **Policy, before any byte leaves.** This is an OUTGOING channel, so Home is the
   SOURCE and the single export policy (`remote_export_policy`) applies at exactly
   the same place it applies on the target: before the blob is constructed. The
   channel is `attachment_stage`, whose profile is `deliverable` — the owner named
   these paths, so credential-shaped ones are not deliverables. Filtering after the
   upload would not be filtering; the bytes would already be on another host.
2. **Upload as content-addressed blobs**, through the transport primitives that
   already exist (`prepare(message, blobs)` chunks, hashes and ACKs each one). The
   blob id IS the digest, so the target admits only the set Home declared.
3. **Verify what came back.** The target answers with the same manifest plus the
   ONE path it chose (`execution_path` — the caller cannot nominate it). Home
   compares the returned entries field-by-field against the set it authorized: a
   changed manifest is a refusal, because the alternative is a task whose prompt
   advertises a path pointing at bytes nobody proved.

Disclosure is not optional. An excluded attachment is DISCLOSED work (D7), never a
silent absence — the owner attached a file and is owed the sentence saying why it
did not travel. Two doors can drop an attachment (Home staging's credential-source
skip, and this export policy), and both feed ONE omission list so the story the
owner reads is complete regardless of which door dropped what.

── media_frames ─────────────────────────────────────────────────────────────

The other direction, and the asymmetry is the point: here the TARGET is the source,
so the policy is applied there (`RemoteTaskFileCache.export_media`) before the file
is read, and a single named source the policy excludes REFUSES rather than
discloses — once the one source is out there is nothing left to deliver, and an
empty success would read as "the file had no content".

What arrives on Home is an ordinary task artifact: the export is declared as a
declared output, so the transport prefetches and verifies it and the transfer
service publishes it through `artifacts.publish_verified_task_artifact`. The model
then sees a Home path (D9) — `remote_media_predispatch` rewrites the tool's path
argument before any guard judges it, so the guards judge the file that will actually
be opened. The rewrite lives here rather than in each of the vision handlers because
there are several of them plus the auto-attachment path, and a per-handler branch is
the "one policy × N doors" shape the postmortem is about.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

from ouroboros.execd_task_files import (
    ATTACHMENT_STAGE_OPERATION,
    ATTACHMENT_WIRE_FIELDS,
    MAX_ATTACHMENT_BYTES,
    MAX_MEDIA_EXPORT_BYTES,
    MEDIA_EXPORT_OPERATION,
    RemoteTaskFileError,
    attachment_blob_map,
)
from ouroboros.export_policy_contract import ExportPolicyExcludedError
from ouroboros.remote_protocol import canonical_json

# The wire keys of one canonical attachment entry. Anything else Home records for
# its own use (the display label's source spelling, the Home absolute path) stays
# on Home: the target has no use for it and an unknown field fails its contract.
#
# DERIVED from the contract's own set rather than restated, because two halves of one
# field list are exactly the pair that drifts — and this half is the PRODUCER, so a key
# it forgot to project would simply have been missing, while a key it invented is now
# refused by name on the far side.
_WIRE_FIELDS = tuple(sorted(ATTACHMENT_WIRE_FIELDS))
# The channels' own names in the closed registry (`export_policy_contract`).
ATTACHMENT_EXPORT_CHANNEL = "attachment_stage"
MEDIA_EXPORT_CHANNEL = "media_frames"
# The declared import channel of the staging result (`remote_protocol.IMPORT_CHANNELS`).
ATTACHMENT_IMPORT_KIND = "attachment_stage_v1"


def _policy_judged_spellings(entry: Mapping[str, Any]) -> tuple[str, ...]:
    """Every spelling of one attachment the export policy must judge.

    Two, because staging SANITIZES the name: `_safe_attachment_name` rewrites a
    leading dot, so an attached `.env` is stored as `attachments/_.env` and the
    credential-name rules — which are exactly the rules that exist to catch a
    `.env` — would read the stored spelling as an ordinary file. Judging the
    original basename as well closes that, and costs one extra evaluation of a
    pure predicate.
    """

    relpath = str(entry.get("relpath") or "")
    source = str(entry.get("source_name") or "").strip()
    spellings = [relpath]
    if source:
        parent = str(pathlib.PurePosixPath(relpath).parent)
        spellings.append(f"{parent}/{source}" if parent not in {"", "."} else source)
    return tuple(dict.fromkeys(spelling for spelling in spellings if spelling))


def filter_attachments_for_export(
    manifest: Sequence[Mapping[str, Any]],
    policy: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Split a staged manifest into what may leave Home and what may not.

    Returns ``(admitted_wire_entries, excluded_rows)``. The excluded rows carry the
    policy's own reason code and its owner-facing sentence, so the disclosure the
    owner reads is the disclosure the policy actually produced rather than a second
    description of it.
    """

    from ouroboros.export_policy_contract import unaliased_exclusion

    admitted: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for raw in manifest:
        if not isinstance(raw, Mapping):
            continue
        # The reason AND its sentence from ONE evaluation. They used to be two calls —
        # `exclusion_reason` for the verdict, `describe_exclusion` for the text — which is
        # the shape that let a door hold a verdict from one rule group and a sentence from
        # another. The pair cannot disagree when it arrives as a pair.
        reason = sentence = ""
        judged = str(raw.get("relpath") or "")
        for spelling in _policy_judged_spellings(raw):
            reason, sentence = unaliased_exclusion(spelling, policy.document)
            if reason:
                judged = spelling
                break
        if reason:
            excluded.append({
                "path": judged,
                "reason": reason,
                "disclosure": sentence,
            })
            continue
        admitted.append({key: raw[key] for key in _WIRE_FIELDS if key in raw})
    return admitted, excluded


def read_attachment_blobs(
    drive_root: pathlib.Path | str,
    task_id: str,
    manifest: Sequence[Mapping[str, Any]],
) -> dict[str, bytes]:
    """Read the admitted attachments from the task artifact store, confined.

    Every path is re-derived from the artifact-store root and re-checked with
    ``relative_to``: the manifest is Home's own, but a manifest is data, and a
    `relpath` that escaped the store would upload a file the export policy never
    saw. The blob key is the digest recorded at staging, and the transport re-hashes
    every payload, so a file that changed between staging and upload fails the
    content-address check rather than travelling as something else.
    """

    from ouroboros.artifacts import task_artifact_dir_path

    root = task_artifact_dir_path(pathlib.Path(drive_root), task_id, create=False).resolve(strict=False)
    blobs: dict[str, bytes] = {}
    for entry in manifest:
        relpath = str(entry.get("relpath") or "")
        parts = [part for part in relpath.split("/") if part not in {"", "."}]
        if not parts or ".." in parts:
            raise RemoteTaskFileError(
                "attachment_home_stage_escape",
                f"staged attachment path is not store-relative: {relpath!r}",
            )
        candidate = root.joinpath(*parts).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RemoteTaskFileError(
                "attachment_home_stage_escape",
                f"staged attachment escapes the task artifact store: {relpath!r}",
            ) from exc
        if not candidate.is_file():
            raise RemoteTaskFileError(
                "attachment_home_stage_missing",
                f"staged attachment is missing on Home: {relpath!r}",
            )
        size = candidate.stat().st_size
        if size > MAX_ATTACHMENT_BYTES:
            raise RemoteTaskFileError(
                "attachment_manifest_invalid",
                f"staged attachment exceeds the admission limit: {relpath!r}",
            )
        blobs[str(entry.get("sha256") or "")] = candidate.read_bytes()
    return blobs


def validate_staged_attachment_envelope(
    manifest: Sequence[Mapping[str, Any]],
    envelope: Any,
) -> list[dict[str, Any]]:
    """Prove the target staged EXACTLY the authorized set, and learn its one path.

    The staging result carries no bytes back — the whole point of the operation is
    that the bytes went the other way — so this refuses any envelope that brings
    an externalized result or output blobs: on this channel there is nothing for
    them to be. What it does accept is one added fact per entry, the target-chosen
    `execution_path`, which becomes the address a remote `run_command` opens.
    """

    from ouroboros.workspace_diagnostics import RemoteWorkspaceError

    expected = [{key: entry[key] for key in _WIRE_FIELDS if key in entry} for entry in manifest]

    def _field(source: Any, name: str) -> Any:
        # The envelope is a dataclass at the executor seam and a plain mapping at the
        # transport import seam; both call this, so it reads either rather than
        # forcing one of the two callers to convert (and forcing the Home policy
        # module to import a transport helper to do it).
        return source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)

    diagnostic = _field(envelope, "diagnostic")
    if diagnostic is not None and str(_field(diagnostic, "code") or ""):
        raise RemoteWorkspaceError(
            str(_field(diagnostic, "code")),
            str(_field(diagnostic, "message") or "remote attachment staging failed"),
            phase=str(_field(diagnostic, "phase") or "execute"),
        )
    trace = _field(envelope, "trace") or {}
    returned = trace.get("attachment_manifest")
    if not isinstance(returned, list) or len(returned) != len(expected):
        raise RemoteWorkspaceError(
            "attachment_manifest_changed",
            "The target did not report the authorized attachment set.",
            phase="import",
        )
    staged: list[dict[str, Any]] = []
    for authorized, remote in zip(expected, returned):
        if not isinstance(remote, Mapping):
            raise RemoteWorkspaceError(
                "attachment_manifest_changed",
                "The target reported a malformed attachment entry.",
                phase="import",
            )
        projected = {key: remote.get(key) for key in authorized}
        if canonical_json(projected) != canonical_json(authorized):
            raise RemoteWorkspaceError(
                "attachment_manifest_changed",
                f"The target staged a different attachment than Home authorized: "
                f"{authorized.get('relpath')!r}.",
                phase="import",
            )
        execution_path = str(remote.get("execution_path") or "")
        if (
            not execution_path.startswith("/")
            or len(execution_path) > 4096
            or any(character < " " for character in execution_path)
        ):
            raise RemoteWorkspaceError(
                "attachment_execution_path_invalid",
                "The target reported an unusable attachment execution path.",
                phase="import",
            )
        staged.append({**authorized, "execution_path": execution_path, "abs_path": execution_path})
    return staged


def stage_attachments_on_target(
    executor: Any,
    task_id: str,
    manifest: Sequence[Mapping[str, Any]],
    blobs: Mapping[str, bytes],
) -> list[dict[str, Any]]:
    """Upload one verified attachment set and return its target-side manifest.

    The byte movement goes through the transfer service (the ONE place `blobs=`
    crosses the boundary); what this owns is the channel's contract — the
    content-addressed set, its declared import kind, and the verification of what
    came back.
    """

    from ouroboros.remote_transfer import RemoteTransferService

    canonical, verified = attachment_blob_map(list(manifest), dict(blobs))
    envelope = RemoteTransferService().export_operation(
        executor,
        ATTACHMENT_STAGE_OPERATION,
        {"manifest": canonical},
        blobs=verified,
        task_id=str(task_id),
        import_kind=ATTACHMENT_IMPORT_KIND,
        import_context={"expected_manifest": canonical},
    )
    return validate_staged_attachment_envelope(canonical, envelope)


def export_task_attachments(
    workspace_ref: Any,
    *,
    drive_root: pathlib.Path | str,
    task_id: str,
    manifest: Sequence[Mapping[str, Any]],
    protected_paths: Sequence[str] = (),
    staging_omissions: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """The ONE Home entry point: policy, upload, verification, disclosure.

    Returns ``{"attachments", "excluded", "excluded_count", "partial", "note"}``.
    ``attachments`` is the target-side manifest (empty when nothing survived the
    policy), and ``note`` is the single owner-facing sentence — empty when nothing
    was omitted, so the presence of the sentence always means something really was.

    A transport failure is NOT swallowed: it raises, because a task whose prompt
    promises attachments the target does not have is a task built on a false
    premise, and admission is the right place to say so.
    """

    from ouroboros.export_policy_contract import build_policy_document, export_policy_hash
    from ouroboros.remote_export_policy import ExportPolicy
    from ouroboros.workspace_executor import executor_ref_from_workspace_ref

    document = build_policy_document(
        channel=ATTACHMENT_EXPORT_CHANNEL, protected_paths=list(protected_paths)
    )
    policy = ExportPolicy(
        channel=ATTACHMENT_EXPORT_CHANNEL,
        document=document,
        policy_hash=export_policy_hash(document),
    )
    admitted, excluded = filter_attachments_for_export(list(manifest), policy)
    rows = staging_omission_rows(staging_omissions) + excluded
    staged: list[dict[str, Any]] = []
    if admitted:
        staged = stage_attachments_on_target(
            executor_ref_from_workspace_ref(workspace_ref),
            str(task_id),
            admitted,
            read_attachment_blobs(drive_root, task_id, admitted),
        )
    return {
        "attachments": staged,
        "excluded": rows,
        "excluded_count": len(rows),
        "partial": bool(rows),
        "policy_hash": policy.policy_hash,
        "note": attachment_omission_note(rows),
    }


# ── media import: the target has the bytes, the vision model is on Home ──────


class RemoteMediaUnavailable(RuntimeError):
    """One media source that cannot become a Home artifact, with the real reason."""

    code = "REMOTE_MEDIA_UNAVAILABLE"


def import_media_from_target(
    ctx: Any,
    reference: str,
    *,
    attachment_id: str = "",
    max_bytes: int = MAX_MEDIA_EXPORT_BYTES,
) -> dict[str, Any]:
    """Pull ONE target file to Home and publish it as a task artifact (D9).

    The model's brain is on Home (D1), so vision needs BYTES here — a target path is
    not something a vision model can look at. What comes back is therefore an
    ordinary Home artifact: the transport verifies the declared blob's size and hash,
    the transfer service publishes it through `artifacts.publish_verified_task_artifact`,
    and what this returns is the Home path plus the media type. The target-native
    source path stays in the private import receipt (D9) — the model, the CLI and the
    review evidence all see one identity, the Home one.

    The export policy is applied ON THE TARGET, before the file is read: a single
    named source that the policy excludes REFUSES rather than returning an empty
    success, because there is nothing left to deliver once the one source is out.
    """

    from ouroboros.remote_export_policy import build_export_policy
    from ouroboros.remote_transfer import RemoteTransferService
    from ouroboros.workspace_diagnostics import RemoteWorkspaceError
    from ouroboros.workspace_executor import executor_ref_from_ctx
    from ouroboros.workspace_ref import workspace_ref_for

    ref = workspace_ref_for(ctx)
    if getattr(ref, "kind", "") != "ssh":
        raise RemoteMediaUnavailable("media import requires a sealed ssh placement")
    policy = build_export_policy(
        ctx, channel=MEDIA_EXPORT_CHANNEL, workspace_root=ref.remote_root
    )
    args: dict[str, Any] = {"max_bytes": int(max_bytes), **policy.arg_payload()}
    if attachment_id:
        args["attachment_id"] = attachment_id
    else:
        args["path"] = str(reference)
    try:
        envelope = RemoteTransferService().export_operation(
            executor_ref_from_ctx(ctx),
            MEDIA_EXPORT_OPERATION,
            args,
            task_id=str(getattr(ctx, "task_id", "") or ""),
            # The target NORMALIZES the path it was handed (and reports the resolved
            # relative spelling), so echoing the arguments back would compare a
            # request against its own resolution. Nothing of Home's is at risk in
            # this direction: the blob's own hash is what binds the bytes.
            echo_args=False,
        )
    except RemoteWorkspaceError as exc:
        raise RemoteMediaUnavailable(f"{exc.code}: {exc}") from exc
    except (ExportPolicyExcludedError, RemoteTaskFileError) as exc:
        # The target's OWN typed refusals, seen directly when execd runs in-process
        # (tests, a local execd). Over the wire they arrive already wrapped as a
        # `RemoteWorkspaceError`; catching both shapes keeps the model-facing answer
        # the same either way instead of making it depend on the transport.
        raise RemoteMediaUnavailable(str(exc)) from exc
    return _published_media(ctx, envelope, reference)


def _published_media(ctx: Any, envelope: Any, reference: str) -> dict[str, Any]:
    """Read the Home artifact identity out of an imported media envelope.

    The absolute Home path is DERIVED from the artifact-store root plus the published
    name rather than carried in the result: the public record deliberately holds only
    the artifact identity, and adding a host path to it would be a second address for
    one file. The derived path lands inside the task's own artifact store, which the
    media trust boundary already admits — so the imported file is readable by exactly
    the rules a locally produced screenshot is, and no new surface appears.
    """

    from ouroboros.tool_access import resource_root_path

    trace = getattr(envelope, "trace", None)
    trace = trace if isinstance(trace, Mapping) else {}
    facts = trace.get("remote_media")
    facts = facts if isinstance(facts, Mapping) else {}
    for row in getattr(envelope, "artifacts", None) or []:
        home_ref = row.get("home_ref") if isinstance(row, Mapping) else None
        if not isinstance(home_ref, Mapping) or not home_ref.get("path"):
            continue
        artifact_name = str(home_ref["path"])
        home_path = pathlib.Path(resource_root_path(ctx, "artifact_store")) / artifact_name
        if not home_path.is_file():
            raise RemoteMediaUnavailable(
                f"imported media artifact {artifact_name!r} is not on Home after publication"
            )
        return {
            "home_path": str(home_path),
            "artifact_name": artifact_name,
            "sha256": str(row.get("sha256") or ""),
            "size": int(row.get("size") or 0),
            "mime": str(facts.get("mime") or "application/octet-stream"),
            "source_label": str(facts.get("name") or reference),
        }
    raise RemoteMediaUnavailable(
        f"the target returned no importable bytes for {reference!r}; nothing can be shown"
    )


# Which argument of each vision/media tool names the file the model needs to SEE.
# `extract_video_frames` is absent on purpose: it is already routed as a native
# operation (the frames are cut ON the target, where the video is), so what it needs
# from Home is the import of its RESULT, not of its input.
MEDIA_PATH_ARGS: dict[str, str] = {
    "view_image": "path",
    "ocr_pdf": "path",
    "vlm_query": "file_path",
}


def remote_media_predispatch(ctx: Any, tool: str, args: dict[str, Any]) -> str:
    """Import a remote media source to Home and REWRITE the arg to its Home path.

    Called before the Home handler of a vision tool on an ssh placement. Returns ``""``
    when the handler may proceed (the argument now names a Home artifact) and a
    model-facing refusal otherwise. Rationale for rewriting rather than teaching each
    handler about placement: there are four of them plus the auto-attachment path, and
    a per-handler branch is the "one policy × N doors" shape — the handler should keep
    seeing exactly one kind of path.
    """

    key = MEDIA_PATH_ARGS.get(str(tool or ""))
    if not key:
        return ""
    reference = str(args.get(key) or "").strip()
    if not reference:
        return ""
    attachment_id = _attachment_id_for_reference(ctx, reference)
    try:
        source = reference if attachment_id else _media_source_on_target(ctx, reference)
        if not source and not attachment_id:
            # A Home root owns this path, so the bytes are already here: the handler
            # opens the same file it would on a local task.
            return ""
        imported = import_media_from_target(
            ctx, source, attachment_id=attachment_id
        )
    except RemoteMediaUnavailable as exc:
        return (
            f"⚠️ {RemoteMediaUnavailable.code}: {tool} needs the FILE on Home (vision runs "
            f"here), and {reference!r} could not be imported from the task's remote "
            f"workspace: {exc}"
        )
    args[key] = imported["home_path"]
    return ""


def _media_source_on_target(ctx: Any, reference: str) -> str:
    """The target-relative spelling to import, or ``""`` when Home owns the bytes.

    A remote task's media argument is not one kind of path. Its workspace lives on the
    target, but its ARTIFACTS, its task scratch and the owner's own uploads live on Home
    (the root-placement matrix, `workspace_ref.SSH_NATIVE_ROOTS`: `active_workspace` is
    the only target-native root), and asking the target for a Home artifact fetched the
    wrong file or nothing at all. So the root decides, and it is decided by root FACTS —
    `vision.home_file_root_for`, the same containment the local handler enforces — never
    by the shape of the string.

    A RELATIVE name goes to the target unless Home actually HOLDS it: only a file that
    exists under a Home root can be served from one, so a workspace-relative spelling
    Home has nothing for still crosses the wire, while `attachments/doc.pdf` — the
    spelling the artifact-store manifest advertises — is read where it lives instead of
    being asked of a target that never had it.

    A path belonging to no admitted root is refused HERE. It could be handed to the
    target, which refuses an absolute spelling of its own accord, but that spends a round
    trip to be told something Home already knew and answers "could not be imported from
    the remote workspace" for a path that was never about the remote workspace.
    """

    from ouroboros.tools.vision import home_file_root_for
    from ouroboros.workspace_ref import (
        normalize_remote_root_relative,
        workspace_ref_for,
    )

    if home_file_root_for(ctx, reference) is not None:
        return ""
    ref = workspace_ref_for(ctx)
    # Absolute target spellings are folded to workspace-relative exactly as every
    # root-labelled tool's path argument is; the target admits nothing else.
    relative = normalize_remote_root_relative(
        getattr(ref, "remote_root", ""), reference
    )
    if (
        not relative
        or relative.startswith("/")
        or relative == ".."
        or relative.startswith("../")
    ):
        raise RemoteMediaUnavailable(
            f"{reference!r} is inside no root this task may read: not the target's "
            f"workspace, and not the task's artifact store, task drive, uploads or "
            f"skill state on Home"
        )
    return relative


def _attachment_id_for_reference(ctx: Any, reference: str) -> str:
    """Match a reference against the task's staged attachments, or ``""``.

    A remote task's attachment lives in the target's private cache, not in its
    workspace, so an `execution_path` cannot be read as a workspace-relative path. The
    manifest is the only thing that can tell the two apart, which is why it is
    consulted here rather than guessed at from the shape of the string.
    """

    metadata = getattr(ctx, "task_metadata", None)
    rows = metadata.get("_remote_attachment_manifest") if isinstance(metadata, Mapping) else None
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        if reference in {
            str(row.get("execution_path") or ""),
            str(row.get("abs_path") or ""),
            str(row.get("relpath") or ""),
        }:
            return str(row.get("attachment_id") or "")
    return ""


def staging_omission_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Project Home staging skips into the shared disclosure row shape.

    Home staging and the export policy are two different doors that can drop the
    same attachment; folding both into one row shape is what lets the owner read ONE
    omission list instead of having to know which door was involved.
    """

    return [
        {
            "path": str(row.get("name") or ""),
            "reason": str(row.get("reason") or ""),
            "disclosure": f"{row.get('name')}: not staged for this task ({row.get('reason')})",
        }
        for row in rows
        if isinstance(row, Mapping)
    ]


def attachment_omission_note(rows: Sequence[Mapping[str, str]]) -> str:
    """One honest sentence about attachments that did not travel, or ``""``."""

    if not rows:
        return ""
    named = "; ".join(str(row.get("disclosure") or row.get("path") or "") for row in rows[:5])
    more = f" (+{len(rows) - 5} more)" if len(rows) > 5 else ""
    return (
        f"⚠️ ATTACHMENTS_OMITTED: {len(rows)} attached file(s) were NOT made available "
        f"to this task: {named}{more}. Everything else was staged and verified."
    )


__all__ = [
    "ATTACHMENT_EXPORT_CHANNEL",
    "ATTACHMENT_IMPORT_KIND",
    "attachment_omission_note",
    "export_task_attachments",
    "filter_attachments_for_export",
    "read_attachment_blobs",
    "stage_attachments_on_target",
    "staging_omission_rows",
    "validate_staged_attachment_envelope",
]
