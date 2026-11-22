"""Selected task documents for a delegated start, attached as engine-owned immutable refs.

``attachment_paths`` names retained task inputs by exact manifest relpath
(``attachments/<name>``) on ``delegate_start`` and on an agent-session
``schedule_subagent``, whose child's first start carries it. Omitted keeps the
work order's input paths and readonly copies exactly as before; ``[]`` adds no engine
document. Each selected staged, non-image input is imported into the engine's store
(``POST /v2/uploads/imports``) and the canonical run body carries
``attachments: [{resourceId}]``, so ``retry_of`` replays the same refs and never
reselects files. A continuation or replacement inherits no refs: it selects again.

One current import intent exists per input, bound to the engine store it was minted in:
a host-minted random key, written to this task's custody marks (the ordinary custody
event log) with that store id, the input's manifest identity and the exact request before
any network call. Only that fresh intent sends the full declaration, after a durable
possibly-sent mark; every later attempt (a crash, a lost response, another run) replays
lookup-only first, before any source read. A default start never changes store: an
intent or inherited receipt bound to another store refuses typed before any source read
or import call. A 404 is a retained-binding gap and a cancelled or conflicting key is
finished: typed refusals that never mint a key on their own. ``attachment_import='new'``
is the explicit decision to begin another import in the serving store: one new random
key per input for that invocation; earlier records stay on the marks row and in the chain.
A scheduled child's first start keeps the one intent saved with its task, and that intent
begins only the first import of an input the task holds no record of, so neither a restarted
bootstrap nor an omitted repeat of that start mints again, even after the child's own new.
An answer too malformed to compare leaves the outcome unknown, recovered under the same key.
The source is the retained staged copy, read only for a fresh intent or while the engine
reports ``open``, and verified against the recorded digest, so a changed or missing
original never switches the version. Observed receipts, finished and gapped too, ride inherited manifest
rows (``engine_imports``), so a child replays the same key instead of minting another import.

Managed lifetime exists only where the engine's catalog lists the exact release row
(``claudexor_uploads.RESOURCE_RELEASE_OPERATION``): there every fresh intent declares
``lifetime: "managed"``, bound into its key, so every replay of it (lookup-only included)
resends the same body; inherited receipts and earlier requests replay unchanged. The task
that minted a managed intent owes its release: the intent's own REQUESTED fact opens it in
``managed_imports`` on that task's marks row (``opens_release``), never capped like earlier
records, and only a settlement closes it; the same custody transaction keeps a retirement pass
that never ran yet from pinning past it (``maintenance_retirement.release_opened``). A child,
a successor or any other holder of the carried receipt (``owner_task_id`` names the owner)
replays it and owes nothing. ``release_owed_imports``, run by ``runtime_retirement`` over
finished tasks, releases only after nothing in the whole work can name the resource in a new
operation (``review_thread_retention.tree_hold`` with ``every_offer``): the owner's tree, each
root its owner's Continue admitted after it and their trees, and any Continue still offered
or undecided, an author's stop and an unrecorded interruption included. A key whose resource
was never observed is asked lookup-only first; an unfinished open upload is cancelled.
After waits, closure is checked again and release binds under the existing Keep custody
lock before HTTP. Earlier Keep prevents release; later Keep discloses the bound uncertainty.
Unknown outcomes retain their exact identity. A ``released`` key is finished: only an
explicit ``new`` begins another import.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import logging
import pathlib
import re
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

log = logging.getLogger(__name__)

# Task-scoped custody facts, reduced onto the task's marks row by ``fold_document_import``.
DOCUMENT_IMPORT_REQUESTED = "delegate_run_document_import_requested"
DOCUMENT_IMPORT_SENDING = "delegate_run_document_import_sending"
DOCUMENT_IMPORT_OBSERVED = "delegate_run_document_import_observed"
DOCUMENT_RELEASE_BOUND = "delegate_run_document_release_bound"
DOCUMENT_RELEASE_SETTLED = "delegate_run_document_release_settled"
DOCUMENT_IMPORT_TYPES = (DOCUMENT_IMPORT_REQUESTED, DOCUMENT_IMPORT_SENDING, DOCUMENT_IMPORT_OBSERVED,
                         DOCUMENT_RELEASE_BOUND, DOCUMENT_RELEASE_SETTLED)
# Observed states that finish a key: only an explicit new intent could follow them.
FINISHED_STATES = frozenset({"cancelled", "conflict", "rejected", "released"})
_RESOURCE_FIELDS = ("resourceId", "sha256", "sizeBytes", "mime", "name")
# An owed release whose key holds no observed resource and needs no engine answer: what its state settles as.
_UNASKED = {"minted": "unsent", "cancelled": "cancelled", "rejected": "rejected", "conflict": "conflict",
            "gap": "gap", "damaged": "damaged"}
# The lookup's typed finishes (``_post`` already observed them on the record): what each settles as.
_LOOKUP_FINISHED = {"document_import_binding_gap": "gap", "document_import_binding_invalid": "damaged",
                    "document_import_conflict": "conflict", "document_import_rejected": "rejected"}
# Release refusals that end the obligation without effect: this resource is not one this task may release here.
_RELEASE_REFUSED = frozenset({"resource_not_found", "resource_purpose_mismatch", "resource_digest_mismatch",
                              "resource_lifetime_indefinite"})
# Unrun refusals a later start of the same task settles without a second key: the host's first
# start wakes its child with them instead of ending it (``subagent_bootstrap``).
RECOVERABLE_REFUSALS = frozenset({"document_import_outcome_unknown", "document_import_in_flight",
                                  "document_import_busy"})
MAX_ATTACHMENT_PATHS = 32
MAX_PREVIOUS_RECORDS = 8
_SHA256 = re.compile(r"[0-9a-f]{64}")
# Definite no-effect answers to a first declaration: nothing was bound, so it stays fresh.
_UNSENT_CODES = frozenset({"store_mismatch", "upload_import_unavailable"})
# Engine answers that finish a key (INTERFACE "Errors"): recorded, never retried under it.
_FINISHING_CODES = {"idempotency_conflict": "conflict", "upload_import_content_conflict": "conflict",
                    "sensitive_resource_rejected": "rejected"}
_PATHS_CONTRACT = (
    "Retained task inputs to attach as engine documents, by exact manifest relpath ('attachments/<name>'; "
    "staged, non-image rows only). Omit to keep today's inputs (work-order paths and readonly copies); [] "
    "attaches none. An engine without keyed document import, or a route whose attachmentInputs do not admit "
    "a file, refuses typed instead of dropping it. Images keep their existing path.")


def attachment_parameters(scope: str) -> Dict[str, Any]:
    """The ``attachment_paths``/``attachment_import`` contract delegate_start and schedule_subagent share."""
    return {
        "attachment_paths": {"type": "array", "items": {"type": "string"}, "description": f"{_PATHS_CONTRACT} {scope}"},
        "attachment_import": {"type": "string", "enum": ["reuse", "new"], "description": (
            "reuse (default) replays this task's recorded import of each selected input, so no second engine "
            "resource exists; it never changes engine store. new explicitly begins another import of every "
            "selected input in the serving store (after a cancelled, gapped, conflicting or other-store one) and "
            "may create another resource; needs attachment_paths. Recover an unknown outcome with reuse, never new.")},
    }


def selection_options(paths: Any, mode: Any) -> bool:
    """Validate the pair's shape; True when it asks for a new import intent."""
    if mode not in (None, "reuse", "new"):
        raise DocumentSelectionError("attachment_import_invalid", "attachment_import is 'reuse' or 'new'.")
    if paths is not None and (not isinstance(paths, list) or any(not isinstance(path, str) for path in paths)):
        raise DocumentSelectionError("attachment_paths_invalid", "attachment_paths is a list of manifest relpaths.")
    if mode == "new" and not paths:
        raise DocumentSelectionError("attachment_import_requires_selection",
                                     "attachment_import='new' needs a nonempty attachment_paths.")
    return mode == "new"


def first_start_selection(saved: Dict[str, Any]) -> Dict[str, Any]:
    """A scheduled child's saved selection as the start options of its work order's first start."""
    return {**{key: copy.deepcopy(saved[key]) for key in ("attachment_paths", "attachment_import") if key in saved},
            **({"_document_intent": saved["document_import_intent"]} if saved.get("document_import_intent") else {})}


class DocumentSelectionError(ValueError):
    """A typed refusal before any engine call: nothing was sent to the engine."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code, self.detail = code, detail


class DocumentAttachments(NamedTuple):
    request: Dict[str, Any]      # merged into the canonical run body; {} keeps it byte-identical
    custody: Dict[str, Any]      # START_REQUESTED provenance
    facts: List[Dict[str, Any]]  # the started receipt's view


NO_DOCUMENTS = DocumentAttachments({}, {}, [])


# -- selection (local, no engine) ----------------------------------------------

def task_contract(ctx: Any) -> Dict[str, Any]:
    """The task authority, with the same precedence the scheduler uses (emptiness decides)."""
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    contract = metadata.get("task_contract") if isinstance(metadata.get("task_contract"), dict) else {}
    if not contract and isinstance(getattr(ctx, "task_contract", None), dict):
        contract = ctx.task_contract
    return contract


def retained_inputs(drive_root: Any, task_id: str, contract: Any) -> List[Dict[str, Any]]:
    """This task's retained inputs: its manifest plus owner-mailbox rows, from its own store.

    The staged path is rebuilt from the manifest relpath under this task's artifact root;
    a serialized host path never redirects the reader.
    """
    from ouroboros.artifacts import resolve_attachment_manifest, task_artifact_dir_path
    from ouroboros.owner_mailbox import owner_attachment_manifest

    drive = pathlib.Path(drive_root)
    rows = [*resolve_attachment_manifest(drive, task_id, contract if isinstance(contract, dict) else {}),
            *owner_attachment_manifest(drive, task_id)]
    root = task_artifact_dir_path(drive, task_id).resolve(strict=False)
    for row in rows:
        parts = pathlib.PurePosixPath(str(row.get("relpath") or "")).parts
        row.pop("abs_path", None)
        if row.get("status") == "staged" and len(parts) == 2 and parts[0] == "attachments":
            path = root.joinpath(*parts)
            if not path.is_symlink() and path.resolve(strict=False).is_relative_to(root):
                row["abs_path"] = str(path)
    return rows


def select_inputs(rows: List[Dict[str, Any]], paths: Any) -> List[Dict[str, Any]]:
    """The staged, non-image rows PATHS name, in order: strict identity, never normalized."""
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise DocumentSelectionError("attachment_paths_invalid", "attachment_paths is a list of manifest relpaths.")
    if len(paths) > MAX_ATTACHMENT_PATHS or len(set(paths)) != len(paths):
        raise DocumentSelectionError("attachment_paths_invalid",
                                     f"attachment_paths holds at most {MAX_ATTACHMENT_PATHS} distinct relpaths.")
    selected = []
    for path in paths:
        parts = pathlib.PurePosixPath(path).parts
        if path != "/".join(parts) or len(parts) != 2 or parts[0] != "attachments" or "\\" in path:
            raise DocumentSelectionError("attachment_path_invalid",
                                         f"{path!r} is not an exact 'attachments/<name>' manifest relpath.")
        matches = [row for row in rows if row.get("status") == "staged" and row.get("relpath") == path]
        if not matches or not all(row.get("abs_path") for row in matches):
            raise DocumentSelectionError("attachment_path_unknown",
                                         f"{path!r} names no staged retained input of this task.")
        if any(row.get("is_image") for row in matches):
            raise DocumentSelectionError("attachment_image_unsupported",
                                         f"{path!r} is an image; images keep their existing path.")
        if len({(row.get("sha256"), row.get("size"), row.get("mime")) for row in matches}) != 1:
            raise DocumentSelectionError("attachment_path_ambiguous",
                                         f"{path!r} names inputs with different recorded bytes or MIME.")
        row = matches[0]
        if (type(row.get("size")) is not int or not _SHA256.fullmatch(str(row.get("sha256") or ""))
                or not str(row.get("mime") or "")):
            raise DocumentSelectionError("attachment_identity_missing",
                                         f"{path!r} has no captured size, sha256 and MIME.")
        selected.append(row)
    return selected


def task_inputs(ctx: Any, paths: Any) -> List[Dict[str, Any]]:
    """The rows PATHS select from this task's retained inputs; raises ``DocumentSelectionError``."""
    try:
        rows = retained_inputs(ctx.drive_root, str(getattr(ctx, "task_id", "") or ""), task_contract(ctx))
    except (OSError, ValueError, TypeError) as exc:
        raise DocumentSelectionError("attachment_manifest_unavailable",
                                     f"This task's retained inputs could not be resolved: {exc}") from exc
    return select_inputs(rows, paths)


# -- custody records -----------------------------------------------------------

def material_id(row: Dict[str, Any]) -> str:
    """One retained input's identity: manifest relpath, recorded bytes and MIME. The store is
    a fact of the input's import record, never part of this key, so a store change finds it."""
    identity = [row.get("relpath"), row.get("sha256"), row.get("size"), row.get("mime")]
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode("utf-8")).hexdigest()[:32]


def fold_document_import(mark: Dict[str, Any], event: Dict[str, Any]) -> None:
    """Reduce one import fact onto its task's marks row (live publish and chain fold alike): the input's current
    record (``document_imports``) and every release this task still owes (``managed_imports``, by key)."""
    material, key, kind = str(event.get("material") or ""), str(event.get("import_key") or ""), event.get("type")
    if not material or not key:
        return
    imports = dict(mark.get("document_imports") or {})
    record = dict(imports.get(material) or {})
    owed = dict(mark.get("managed_imports") or {})
    if kind == DOCUMENT_IMPORT_REQUESTED:
        previous = list(record.pop("previous", []))
        if key in {record.get("import_key"), *(item.get("import_key") for item in previous), *owed}:
            return  # a replayed row
        if record:  # an explicit new intent supersedes; the earlier record stays visible
            previous.append(record)
        record = {name: copy.deepcopy(event[name]) for name in
                  ("store_id", "import_key", "request", "input", "origin", "intent", "owner_task_id") if name in event}
        record["state"] = str(event.get("state") or "send_possible") if event.get("origin") == "inherited" else "minted"
        if previous:
            record["previous"] = previous[-MAX_PREVIOUS_RECORDS:]
    elif kind == DOCUMENT_RELEASE_SETTLED:
        if key in owed:
            # Keep retains every observed disposition, independently of retry debt and the
            # capped replay history. Neither a gap nor refusal proves deletion or preservation.
            outcome = str(event.get("outcome") or "")
            view = {"released": "future_reuse_released", "cancelled": "upload_cancelled"}.get(outcome)
            if outcome in {"gap", "damaged", "conflict", "rejected"}:
                view = "binding_unavailable:" + outcome
            elif outcome == "refused":
                view = "release_refused:" + str(event.get("code") or "unknown")
            if view:
                mark.setdefault("managed_import_releases", {})[key] = view
        record = _with_release(record, key, event) if event.get("outcome") == "released" else {}
    elif kind == DOCUMENT_RELEASE_BOUND:
        record = {}  # release intent belongs to the owed entry, not the import's observed state
    elif record.get("import_key") != key:
        record = {}  # a superseded intent's fact: in the chain, and on its release obligation if one is owed
    elif kind == DOCUMENT_IMPORT_SENDING:
        record["state"] = "send_possible" if record.get("state") == "minted" else record.get("state")
    elif kind == DOCUMENT_IMPORT_OBSERVED:
        record["state"] = str(event.get("state") or "")
        record.pop("code", None)
        if event.get("code"):
            record["code"] = str(event["code"])
        if isinstance(event.get("resource"), dict):
            record["resource"] = dict(event["resource"])
        if event.get("released_at"):
            record["released_at"] = str(event["released_at"])
    else:
        return
    _fold_owed(owed, material, key, kind, event)
    if record:
        imports[material] = record
        mark["document_imports"] = imports
    if owed:
        mark["managed_imports"] = owed
    else:
        mark.pop("managed_imports", None)


def opens_release(event: Dict[str, Any]) -> bool:
    """Is EVENT the one transition that opens a release obligation: the REQUESTED fact of a managed intent this task
    minted itself (an inherited receipt's never does)? Its custody transaction also keeps the retirement pass's first
    read at or before this point (``maintenance_retirement.release_opened``)."""
    request = event.get("request")
    return (event.get("type") == DOCUMENT_IMPORT_REQUESTED and event.get("origin") == "imported"
            and isinstance(request, dict) and request.get("lifetime") == "managed")


def _fold_owed(owed: Dict[str, Any], material: str, key: str, kind: Any, event: Dict[str, Any]) -> None:
    """One release obligation: opened by this task's own managed intent (``opens_release``), its sent and observed
    facts applied whether or not its record is still current, closed only by its settlement."""
    if kind == DOCUMENT_IMPORT_REQUESTED:
        if opens_release(event):
            owed[key] = {"store_id": str(event.get("store_id") or ""), "import_key": key, "material": material,
                         "request": copy.deepcopy(event["request"]), "state": "minted"}
        return
    entry = dict(owed.get(key) or {})
    if not entry:
        return
    if kind == DOCUMENT_RELEASE_SETTLED:
        owed.pop(key)
        return
    if kind == DOCUMENT_IMPORT_SENDING:
        entry["state"] = "send_possible" if entry.get("state") == "minted" else entry.get("state")
    elif kind == DOCUMENT_RELEASE_BOUND:
        entry["release_state"] = "bound"
    else:
        entry["state"] = str(event.get("state") or "")
        if isinstance(event.get("resource"), dict):
            entry["resource"] = dict(event["resource"])
        if event.get("released_at"):
            entry["released_at"] = str(event["released_at"])
    owed[key] = entry


def _with_release(record: Dict[str, Any], key: str, event: Dict[str, Any]) -> Dict[str, Any]:
    """RECORD with the intent under KEY (it, or an earlier intent it keeps) marked released; ``{}`` when it holds
    no such intent (an earlier one past ``MAX_PREVIOUS_RECORDS``: its settlement stays in the chain)."""
    stamp = {"state": "released", **({"released_at": str(event["released_at"])} if event.get("released_at") else {})}
    if record.get("import_key") == key:
        return {**record, **stamp}
    previous = record.get("previous") or []
    if not any(item.get("import_key") == key for item in previous):
        return {}
    return {**record, "previous": [{**item, **stamp} if item.get("import_key") == key else item for item in previous]}


def task_document_imports(drive_root: Any, task_id: str) -> Dict[str, Dict[str, Any]]:
    """This task's import records by material (the custody marks row)."""
    from ouroboros.delegate_custody_current import marks

    return dict(marks(drive_root, task_id).get("document_imports") or {})


def carried_receipts(row: Dict[str, Any], records: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The row's own receipts plus its task's observed imports, one per store.

    A child copy of the row carries them (``engine_imports``), so its starts replay the
    same key and reuse the same resourceId instead of importing equal bytes again.
    """
    from ouroboros.contracts.task_contract import normalize_engine_import_receipts

    receipts = []
    current = records.get(material_id(row)) or {}
    for record in [current, *reversed(current.get("previous") or [])]:  # newest observed intent, even if finished
        resource = record.get("resource") if isinstance(record.get("resource"), dict) else {}
        if record and record.get("state") != "minted":
            receipts.append({"store_id": record["store_id"], "import_key": record["import_key"],
                             "request": copy.deepcopy(record["request"]),
                             **({"resource_id": resource["resourceId"]} if resource.get("resourceId") else {}),
                             **({"state": record["state"]} if record.get("state") != "ready" else {}),
                             **({"owner_task_id": record["owner_task_id"]} if record.get("owner_task_id") else {})})
    return normalize_engine_import_receipts([*receipts, *(row.get("engine_imports") or [])], row)


def carry_owner_receipts(custody_drive: Any, task_id: str, rows: List[Dict[str, Any]]) -> None:
    """Attach the owner's observed imports before copying staged document rows to a child, including finished keys.

    Every copy of a staged non-image row reads the owner's marks once, selected or not, so a
    later selection by the child reuses the resource; images never become documents and read
    nothing. Unreadable custody refuses typed before the child exists: a copy without the
    owner's import identity would let the child mint a fresh import and hide that gap.
    """
    from ouroboros.store import StoreError

    staged = [row for row in rows if row.get("status") == "staged" and not row.get("is_image")]
    if not staged:
        return
    try:
        records = task_document_imports(custody_drive, task_id)
    except StoreError as exc:
        raise DocumentSelectionError("document_import_custody_unavailable", (
            f"The document import records of {task_id} could not be read ({type(exc).__name__}), so its inputs "
            "cannot be copied with their engine identities; nothing was scheduled.")) from exc
    for row in staged:
        receipts = carried_receipts(row, records)
        if receipts:
            row["engine_imports"] = receipts


# -- the engine ----------------------------------------------------------------

def _refusal(code: str, detail: str, status_code: int = 0) -> Exception:
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    return ClaudexorUnavailable(code, detail, status_code=status_code)


def _admitted_store(gateway: Any, route_id: str, rows: List[Dict[str, Any]]) -> Tuple[str, bool]:
    """The store this start imports into, once the engine's own facts admit every row, and whether that engine
    admits managed lifetime (its catalog lists the release row)."""
    from ouroboros.gateways.claudexor_uploads import resource_release_supported, upload_import_supported

    operations = gateway.operations()
    if not upload_import_supported(operations):
        raise _refusal("document_import_unsupported", (
            "This Claudexor engine's catalog has no keyed document import (POST /v2/uploads/imports), so the "
            "selected documents cannot be attached and nothing was started. Omit attachment_paths to keep the "
            "work order's input paths and readonly copies."))
    store = str(getattr(gateway, "store_identity", "") or "")
    if not store:
        raise _refusal("document_store_identity_unknown",
                       "The engine handshake named no store identity, so no import can be bound to a store.")
    harness = next((item for item in gateway.agent_capabilities().get("harnesses") or []
                    if isinstance(item, dict) and item.get("id") == route_id), None)
    classes = [item for item in (harness or {}).get("attachmentInputs") or []
               if isinstance(item, dict) and item.get("kind") == "file"]
    used: Dict[int, int] = {}
    for row in rows:
        match = next((index for index, item in enumerate(classes)
                      if row["mime"] in (item.get("mime_types") or [])
                      and type(item.get("max_bytes")) is int and row["size"] <= item["max_bytes"]), None)
        if match is None:
            raise _refusal("document_route_unsupported", (
                f"Route {route_id} declares no attachment input for {row['relpath']} ({row['mime']}, "
                f"{row['size']} bytes); nothing was imported or started."))
        used[match] = used.get(match, 0) + 1
    for index, count in used.items():
        limit = classes[index].get("max_count")
        if type(limit) is not int or count > limit:
            raise _refusal("document_route_unsupported",
                           f"Route {route_id} admits at most {limit} such documents per run; {count} were selected.")
    return store, resource_release_supported(operations)


@contextmanager
def _import_lock(drive: pathlib.Path, task_id: str, material: str, timeout_sec: float = 30.0) -> Iterator[None]:
    """Serialize one input's decide/send/settle (and its owner's release of it), so a parallel start never reads an
    undelivered first declaration as a gap. The custody actor lock stays network-free."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    digest = hashlib.sha256(f"{task_id}\0{material}".encode("utf-8")).hexdigest()[:24]
    path = drive / "state" / "delegate_document_imports" / f"{digest}.lock"
    fd = acquire_exclusive_file_lock(path, timeout_sec=timeout_sec, stale_sec=600.0, owner_aware_stale=True)
    if fd is None:
        raise _refusal("document_import_busy", "Another start is importing this document now; start again shortly.")
    try:
        yield
    finally:
        release_exclusive_file_lock(path, fd)


def _emit(drive: pathlib.Path, kind: str, payload: Dict[str, Any], *, required: bool) -> None:
    from ouroboros import delegate_custody as custody

    if custody.emit(drive, kind, payload):
        return
    if required:
        raise _refusal("document_import_custody_unwritable",
                       "The import intent could not be recorded, so nothing was sent to the engine.")
    log.warning("document import fact %s was not recorded; the engine binding stays authoritative", kind)


def _retained_bytes(row: Dict[str, Any]) -> bytes:
    """The retained staged copy, verified against the recorded size and sha256."""
    from ouroboros.artifacts import stream_artifact_file

    sink = io.BytesIO()
    try:
        stream_artifact_file(pathlib.Path(row["abs_path"]), sink, expected={"size": row["size"], "sha256": row["sha256"]})
    except (OSError, ValueError) as exc:
        raise _refusal("document_source_unavailable", (
            f"The retained copy of {row['relpath']} is missing or differs from its recorded sha256/size; "
            "no other version is sent.")) from exc
    return sink.getvalue()


def _request_for(task_id: str, store: str, row: Dict[str, Any], managed: bool) -> Dict[str, Any]:
    """A fresh intent's exact declaration; ``lifetime`` only where the engine admits it (absent keeps a body
    byte-identical to one minted before managed lifetime existed)."""
    return {"expectedStoreId": store, "kind": "file", "mime": row["mime"],
            "name": pathlib.PurePosixPath(row["relpath"]).name,
            "source": f"ouroboros-task-input:{task_id}/{row['relpath']}",
            **({"lifetime": "managed"} if managed else {}),
            "content": {"sizeBytes": row["size"], "sha256": f"sha256:{row['sha256']}"}}


def _input_identity(row: Dict[str, Any]) -> Dict[str, Any]:
    return {name: row[name] for name in ("relpath", "ordinal", "label", "sha256", "size", "mime") if name in row}


def _outcome_unknown(key: str, cause: str, status_code: int = 0) -> Exception:
    return _refusal("document_import_outcome_unknown", (
        f"Import {key} has an unknown outcome ({cause}); a later start with attachment_import='reuse' replays it "
        "lookup-only before any source read, while 'new' would begin another import and may create another "
        "resource. Nothing was started."), status_code)


def _checked(answer: Dict[str, Any], record: Dict[str, Any], drive: pathlib.Path,
             facts: Dict[str, Any]) -> Dict[str, Any]:
    """The engine's answer for exactly this store and these bytes, or a typed refusal.

    An answer naming another store or other bytes is evidence and refuses as such; one too
    malformed to compare leaves the outcome unknown, so a later start replays the same key.
    A ``ready`` or ``released`` resource must also carry exactly the requested lifetime."""
    request, key = record["request"], record["import_key"]
    store, state, resource = answer.get("storeId"), answer.get("state"), answer.get("resource")
    if isinstance(store, str) and store and store != record["store_id"]:
        raise _refusal("document_import_store_mismatch", f"Import {key} was answered by another engine store.")
    upload_id = answer.get("uploadId")
    resource_id = resource.get("resourceId") if isinstance(resource, dict) else None
    held = state in ("ready", "released")
    if (store != record["store_id"] or not isinstance(state, str)
            or state not in {"open", "uploading", "ready", "cancelled", "released"}
            or not isinstance(upload_id, str) or not upload_id or not isinstance(answer.get("content"), dict)
            or (held and (not isinstance(resource_id, str) or not resource_id))
            or (state == "released" and not (isinstance(answer.get("releasedAt"), str) and answer["releasedAt"]))):
        raise _outcome_unknown(key, "unusable answer")
    if answer["content"] != request["content"] or (held and (
            resource.get("purpose") is not None or resource.get("lifetime") != request.get("lifetime")
            or any(resource.get(name) != request.get(name) for name in ("kind", "mime", "name"))
            or resource.get("sha256") != request["content"]["sha256"]
            or resource.get("sizeBytes") != request["content"]["sizeBytes"])):
        _emit(drive, DOCUMENT_IMPORT_OBSERVED, {**facts, "state": "conflict", "code": "content_mismatch"}, required=False)
        raise _refusal("document_import_conflict",
                       f"Import {key} is bound to other bytes or another kind, MIME or name than this input.")
    return answer


def _post(gateway: Any, drive: pathlib.Path, facts: Dict[str, Any], record: Dict[str, Any],
          fresh: bool) -> Dict[str, Any]:
    """One import call: the full declaration for a fresh intent, else lookup-only (no content)."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    key = record["import_key"]
    body = dict(record["request"]) if fresh else {k: v for k, v in record["request"].items() if k != "content"}
    try:
        answer = gateway.import_upload(body, idempotency_key=key)
    except ClaudexorUnavailable as exc:
        code, status = exc.code, exc.status_code
        if code in ("upload_import_not_found", "resource_not_found") and not fresh:
            _emit(drive, DOCUMENT_IMPORT_OBSERVED, {**facts, "state": "gap", "code": code}, required=False)
            raise _refusal("document_import_binding_gap", (
                f"The engine store no longer retains what import {key} named ({code}). That is a retained-binding "
                "gap, not proof it was never accepted, so no new import was minted and nothing was started."),
                status) from exc
        if code == "upload_import_binding_invalid":  # engine store damage: observed, never repaired here
            _emit(drive, DOCUMENT_IMPORT_OBSERVED, {**facts, "state": "damaged", "code": code}, required=False)
            raise _refusal("document_import_binding_invalid", (
                f"The engine store holds a damaged binding for import {key}; no new import was minted and "
                "nothing was started."), status) from exc
        finished = _FINISHING_CODES.get(code)
        if finished:
            _emit(drive, DOCUMENT_IMPORT_OBSERVED, {**facts, "state": finished, "code": code}, required=False)
            raise _refusal(f"document_import_{finished}", f"Import {key} is finished: {code}: {exc}", status) from exc
        if code in _UNSENT_CODES or status == 400:
            if fresh:  # answered before any binding: this intent was never sent
                _emit(drive, DOCUMENT_IMPORT_OBSERVED, {**facts, "state": "minted", "code": code}, required=False)
            raise _refusal("document_import_refused", f"Import {key} was refused before any effect: {code}: {exc}",
                           status) from exc
        if 400 <= status < 500:
            raise _refusal("document_import_refused", f"Import {key} was refused: {code}: {exc}", status) from exc
        raise _outcome_unknown(key, code, status) from exc
    return _checked(answer, record, drive, facts)


def _put(gateway: Any, upload_id: str, data: bytes, row: Dict[str, Any]) -> None:
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        gateway.put_upload_bytes(upload_id, data)
    except ClaudexorUnavailable as exc:
        unknown = exc.status_code == 0 or exc.status_code >= 500
        raise _refusal("document_import_outcome_unknown" if unknown else "document_import_bytes_refused",
                       f"Sending the bytes of {row['relpath']} failed ({exc.code}); nothing was started.",
                       exc.status_code) from exc


def _resource(answer: Dict[str, Any]) -> Dict[str, Any]:
    return {name: answer["resource"][name] for name in _RESOURCE_FIELDS if name in answer["resource"]}


def _import(gateway: Any, drive: pathlib.Path, task_id: str, store: str, row: Dict[str, Any],
            new_intent: str, first_start: bool = False, managed: bool = False) -> Dict[str, Any]:
    from ouroboros.contracts.task_contract import normalize_engine_import_receipts
    from ouroboros.store import StoreError

    material = material_id(row)
    with _import_lock(drive, task_id, material):
        try:
            record = task_document_imports(drive, task_id).get(material)
        except StoreError as exc:
            raise _refusal("document_import_custody_unavailable",
                           f"This task's import records could not be read: {type(exc).__name__}") from exc
        receipts = normalize_engine_import_receipts(row.get("engine_imports"), row)
        data, origin, superseded, intent = None, "replayed", "", None
        # A saved first-start intent begins only an input's first import here; an input with a record (from
        # that intent or any of the task's own starts) keeps its key, and only an explicit new supersedes it.
        if (new_intent and (record is None or (not first_start and record.get("intent") != new_intent))) \
                or (record is None and not receipts):
            # A first import, or the explicit new intent in the serving store (the earlier record stays visible).
            data, origin = _retained_bytes(row), "imported"  # verified before any effect
            superseded = str((record or {}).get("import_key") or "")
            # A managed intent's REQUESTED fact is also this task's release obligation (``managed_imports``).
            intent = {"store_id": store, "import_key": f"ouroboros-document-{uuid.uuid4().hex}",
                      "request": _request_for(task_id, store, row, managed),
                      **({"owner_task_id": task_id} if managed else {}),
                      **({"intent": new_intent} if new_intent else {})}
        elif record is None:  # an inherited receipt; one bound to another store is kept, then refused below
            carried = next((item for item in receipts if item["store_id"] == store), receipts[0])
            origin, intent = "inherited", {"store_id": carried["store_id"], "import_key": carried["import_key"],
                                           "request": carried["request"],
                                           "state": carried.get("state", "send_possible"),
                                           **({"owner_task_id": carried["owner_task_id"]} if "owner_task_id" in carried
                                              else {})}
        if intent is not None:
            _emit(drive, DOCUMENT_IMPORT_REQUESTED, {"task_id": task_id, "material": material, **intent,
                                                     "input": _input_identity(row), "origin": origin}, required=True)
            record = {**intent, "state": intent.get("state", "minted" if origin == "imported" else "send_possible")}
        facts = {"task_id": task_id, "material": material, "import_key": record["import_key"]}
        if record.get("store_id") != store:
            raise _refusal("document_import_store_mismatch", (
                f"Import {record['import_key']} for {row['relpath']} is bound to engine store "
                f"{record.get('store_id')}, not the serving store {store}. A default start never changes store, so "
                "nothing was read, sent or started; attachment_import='new' explicitly begins an import in this "
                "store and may create another resource."))
        if record.get("state") in FINISHED_STATES:
            raise _refusal("document_import_finished", (
                f"Import {record['import_key']} for {row['relpath']} is {record['state']} "
                f"({record.get('code') or 'engine'}); no new import is minted automatically."))
        if record["request"].get("lifetime") == "managed" and not managed:
            raise _refusal("document_import_unsupported", (
                f"Import {record['import_key']} requires managed lifetime, which this engine does not advertise. "
                "Its exact key, body and store are retained; no unsupported field or replacement import is sent."))
        fresh = record.get("state") == "minted"
        if fresh:
            data = data if data is not None else _retained_bytes(row)
            _emit(drive, DOCUMENT_IMPORT_SENDING, facts, required=True)
        answer = _post(gateway, drive, facts, record, fresh)
        if answer["state"] == "open":
            _put(gateway, answer["uploadId"], data if data is not None else _retained_bytes(row), row)
            # The binding and PUT were accepted: refusal of this later lookup cannot undo
            # their effects or turn the release obligation back into an unsent intent.
            answer = _post(gateway, drive, facts, record, False)
        if answer["state"] == "released":  # a finished key: never masked by a quiet second import
            _emit(drive, DOCUMENT_IMPORT_OBSERVED, {**facts, "state": "released", "resource": _resource(answer),
                                                    "released_at": answer["releasedAt"]}, required=False)
            raise _refusal("document_import_finished", (
                f"Import {record['import_key']} for {row['relpath']} was released at {answer['releasedAt']} by the "
                f"task that owed it ({record.get('owner_task_id') or 'its owner'}): its resource admits no new "
                "reference and nothing is uploaded again. attachment_import='new' begins another import."))
        if answer["state"] == "cancelled":
            _emit(drive, DOCUMENT_IMPORT_OBSERVED, {**facts, "state": "cancelled"}, required=False)
            raise _refusal("document_import_finished",
                           f"Import {record['import_key']} for {row['relpath']} was cancelled by the engine.")
        if answer["state"] != "ready":
            raise _refusal("document_import_in_flight", (
                f"Import {record['import_key']} for {row['relpath']} is still {answer['state']}; "
                "start again shortly."))
        resource = _resource(answer)
        _emit(drive, DOCUMENT_IMPORT_OBSERVED, {**facts, "state": "ready", "resource": resource}, required=False)
    return {"path": row["relpath"], "label": str(row.get("label") or ""), "mime": row["mime"], "size": row["size"],
            "sha256": row["sha256"], "store_id": store, "import_key": record["import_key"],
            "resource_id": resource["resourceId"], "origin": origin, **({"superseded": superseded} if superseded else {})}


def attach_selected_documents(ctx: Any, gateway: Any, route_id: str, paths: Any,
                              new_intent: str = "", first_start: bool = False) -> DocumentAttachments:
    """Import the selected inputs for one start; raises ``ClaudexorUnavailable`` refusals.

    ``new_intent`` (``attachment_import='new'``) names this invocation's one explicit new
    import intent; ``first_start`` marks it as the intent saved with a scheduled child's task,
    which begins only the first import of an input this task holds no record of. Every
    refusal precedes the run's START_REQUESTED row and POST, so the start is definitely
    unrun; an import with an unknown outcome is recovered by the next ``reuse`` start.
    """
    if not paths:
        return NO_DOCUMENTS
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_evidence import record_start_blocked
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    task_id = str(getattr(ctx, "task_id", "") or "")
    try:
        try:
            rows = task_inputs(ctx, paths)
        except DocumentSelectionError as exc:
            raise _refusal(exc.code, exc.detail) from exc
        store, managed = _admitted_store(gateway, route_id, rows)
        drive = pathlib.Path(custody.custody_root(ctx))
        facts = [_import(gateway, drive, task_id, store, row, new_intent, first_start, managed) for row in rows]
    except ClaudexorUnavailable as exc:
        record_start_blocked(ctx, task_id, exc.code)  # the attempt stays a durable fact (D5)
        raise
    view = [{key: fact[key] for key in ("path", "resource_id", "origin", "superseded") if key in fact} for fact in facts]
    return DocumentAttachments({"attachments": [{"resourceId": fact["resource_id"]} for fact in facts]},
                               {"document_attachments": facts, **({"document_import": "new"} if new_intent else {})},
                               view)


# -- the release of what this task minted managed ------------------------------------------------

def release_owed_imports(root: Any, task_id: str, *, engine: Callable[[], Any], live: Callable[[str], Any],
                         census: Callable[[], Optional[set]], hold: Callable[[str], None], stop: Callable[[], bool],
                         observed: Callable[[Dict[str, Any]], None]) -> bool:
    """Release each managed import TASK_ID still owes once nothing can name it in a new operation; True while one is
    still owed (the task waits for a later pass). Nothing is asked while the owner keeps the task or the whole work
    holds (``review_thread_retention.tree_hold`` with ``every_offer``). ``engine(store_id)`` opens the engine of
    that store on first need (``None``: none), ``hold`` counts why a release waits, ``stop`` ends the pass between two releases and
    ``observed`` receives each settlement. Never called inside the keep gate or a custody lock."""
    from ouroboros.delegate_custody_current import marks
    from ouroboros.owner_keep import is_kept
    from ouroboros.review_thread_retention import tree_hold
    from ouroboros.task_store import read_document

    def held(why: str) -> bool:
        hold("document_" + why.removeprefix("document_"))  # one family beside the thread holds in the pass report
        return True

    owed = marks(root, task_id).get("managed_imports") or {}
    if not owed:
        return False
    if is_kept(root, task_id):
        return held("owner_keep")

    def allowed() -> str:
        if stop():
            return "stopped"
        return tree_hold(root, task_id, read_document(root, task_id) or {}, live=live, census=census(),
                         every_offer=True)[0]

    why = allowed()
    if why:
        return held(why)
    waiting = False
    for done, entry in enumerate(sorted(owed.values(), key=lambda item: str(item.get("import_key") or ""))):
        if done and stop():
            return True
        why = _release_one(pathlib.Path(root), task_id, entry, engine, observed, allowed)
        waiting = held(why) if why else waiting
    return waiting


def _release_one(drive: pathlib.Path, task_id: str, entry: Dict[str, Any], engine: Callable[[], Any],
                 observed: Callable[[Dict[str, Any]], None], allowed: Callable[[], str]) -> str:
    """Settle one owed release: ``""`` once settled, else why it still waits."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    def settle(outcome: str, *, code: str = "", resource: Optional[Dict[str, Any]] = None,
               released_at: Any = "") -> str:
        fact = {"task_id": task_id, "material": entry["material"], "import_key": entry["import_key"],
                "store_id": entry["store_id"], "outcome": outcome, **({"code": code} if code else {}),
                **({"resource_id": resource["resourceId"]} if (resource or {}).get("resourceId") else {}),
                **({"released_at": str(released_at)} if released_at else {})}
        if not _emit_settled(drive, fact):
            return "settlement_unwritable"  # the same release is asked again: a repeat answers the same
        observed(fact)
        return ""

    state = str(entry.get("state") or "")
    resource = dict(entry["resource"]) if isinstance(entry.get("resource"), dict) else {}
    if state == "released":  # the engine already answered so (a start's replay, or a settlement not yet written)
        return settle("released", resource=resource, released_at=entry.get("released_at"))
    if not resource.get("resourceId") and state in _UNASKED:
        return settle(_UNASKED[state])  # never sent, or a key no answer can name a resource for
    gateway = engine(entry["store_id"])  # the engine whose store holds the import (``execution_domain``)
    why = _release_unusable(gateway, entry["store_id"])
    if why:
        return why
    try:
        with _import_lock(drive, task_id, entry["material"], timeout_sec=0.0):
            # Discovery and lookup can wait. Recheck closure after each, outside the Keep gate
            # (tree_hold may take the queue lock), then bind the effect under the existing custody lock.
            def begin() -> str:
                why = allowed()
                return why or _bind_release(drive, task_id, entry)

            if not resource.get("resourceId"):
                why, resource = _lookup(gateway, drive, task_id, entry, settle, begin)
                if why or not resource:
                    return why
            why = begin()
            if why:
                return "" if why == "already_settled" else why
            return _release(gateway, entry, resource, settle)
    except ClaudexorUnavailable as exc:  # ``document_import_busy``: a start holds this input's import lock
        return exc.code


def _emit_settled(drive: pathlib.Path, fact: Dict[str, Any]) -> bool:
    from ouroboros import delegate_custody as custody

    return custody.emit(drive, DOCUMENT_RELEASE_SETTLED, fact)


def _bind_release(drive: pathlib.Path, task_id: str, entry: Dict[str, Any]) -> str:
    """Keep and release serialize at one recorded intent, never across HTTP. A later Keep
    names this intent release_unconfirmed; an earlier Keep prevents the send."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import marks
    from ouroboros.owner_keep import gate

    with gate(drive, task_id, timeout_sec=0) as why:
        if why:
            return why
        current = (marks(drive, task_id).get("managed_imports") or {}).get(entry["import_key"])
        if current is None:
            return "already_settled"
        if any(current.get(key) != entry.get(key) for key in ("material", "store_id", "request")):
            return "binding_changed"
        if current.get("release_state") == "bound":
            return ""  # an unresolved send reuses the same resource/body or cancellation ID
        fact = {"task_id": task_id, "material": entry["material"], "import_key": entry["import_key"]}
        return "" if custody.emit(drive, DOCUMENT_RELEASE_BOUND, fact) else "release_unwritable"


def managed_keep_places(root: Any, task_id: str) -> Dict[str, str]:
    """Current imports and closed reuse remain Keep facts after their local inputs go.
    Read inside the caller's custody lock for the final Keep answer; no engine request."""
    from ouroboros.delegate_custody_current import marks

    mark = marks(root, task_id)
    places = {f"managed_import:{key}": state for key, state in (mark.get("managed_import_releases") or {}).items()}
    places.update({f"managed_import:{key}": ("release_unconfirmed" if entry.get("release_state") == "bound" else "held")
                   for key, entry in (mark.get("managed_imports") or {}).items()})
    return places


def _release_unusable(gateway: Any, store_id: str) -> str:
    """Why this engine cannot be asked about the release now (``""``: it can)."""
    from ouroboros.gateways.claudexor_uploads import resource_release_supported

    if gateway is None:
        return "engine_unavailable"
    try:
        if not resource_release_supported(gateway.operations()):
            return "engine_release_unsupported"
    except Exception:  # an unread catalog proves no route either way
        return "engine_unavailable"
    return "" if str(getattr(gateway, "store_identity", "") or "") == store_id else "store_mismatch"


def _lookup(gateway: Any, drive: pathlib.Path, task_id: str, entry: Dict[str, Any],
            settle: Callable[..., str], begin: Callable[[], str]) -> Tuple[str, Dict[str, Any]]:
    """A key with no observed resource, asked by a lookup-only replay of its exact recorded body (``lifetime``
    included): ``("", resource)`` to release, ``("", {})`` once settled here, or ``(why it waits, {})``."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    facts = {"task_id": task_id, "material": entry["material"], "import_key": entry["import_key"]}
    try:
        answer = _post(gateway, drive, facts, entry, False)
    except ClaudexorUnavailable as exc:  # ``_post`` observed a typed finish itself; anything else stays unknown
        finished = _LOOKUP_FINISHED.get(exc.code)
        return (settle(finished, code=exc.code), {}) if finished else (exc.code or "lookup_unknown", {})
    state = answer["state"]
    if state == "uploading":
        return "import_uploading", {}
    if state == "open":  # no resource exists, and no start of this ended work sends the bytes
        why = begin()
        if why:
            return ("" if why == "already_settled" else why), {}
        try:
            cancelled = gateway.cancel_upload(answer["uploadId"])
        except ClaudexorUnavailable as exc:
            return exc.code or "cancel_unknown", {}
        if cancelled.get("uploadId") != answer["uploadId"] or cancelled.get("state") != "cancelled":
            return "cancel_unknown", {}
        _emit(drive, DOCUMENT_IMPORT_OBSERVED, {**facts, "state": "cancelled"}, required=False)
        return settle("cancelled"), {}
    if state == "cancelled":
        _emit(drive, DOCUMENT_IMPORT_OBSERVED, {**facts, "state": "cancelled"}, required=False)
        return settle("cancelled"), {}
    resource = _resource(answer)
    if state == "released":
        _emit(drive, DOCUMENT_IMPORT_OBSERVED, {**facts, "state": "released", "resource": resource,
                                                "released_at": answer["releasedAt"]}, required=False)
        return settle("released", resource=resource, released_at=answer["releasedAt"]), {}
    _emit(drive, DOCUMENT_IMPORT_OBSERVED, {**facts, "state": "ready", "resource": resource}, required=False)
    return "", resource


def _release(gateway: Any, entry: Dict[str, Any], resource: Dict[str, Any], settle: Callable[..., str]) -> str:
    """One release of exactly the recorded resource and digest in the recorded store; an unknown outcome waits
    and is sent again unchanged (a repeat answers the identical body)."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    body = {"expectedStoreId": entry["store_id"], "sha256": entry["request"]["content"]["sha256"]}
    try:
        answer = gateway.release_resource(resource["resourceId"], body)
    except ClaudexorUnavailable as exc:
        if exc.code in _RELEASE_REFUSED:
            return settle("refused", code=exc.code, resource=resource)
        return exc.code or "release_unknown"
    if (answer.get("state"), answer.get("storeId"), answer.get("resourceId"), answer.get("sha256")) != (
            "released", entry["store_id"], resource["resourceId"], body["sha256"]) or not (
            isinstance(answer.get("releasedAt"), str) and answer["releasedAt"]):
        return "release_unknown"
    return settle("released", resource=resource, released_at=answer["releasedAt"])


__all__ = [
    "DOCUMENT_IMPORT_TYPES", "DOCUMENT_RELEASE_SETTLED", "DocumentAttachments", "DocumentSelectionError",
    "NO_DOCUMENTS", "RECOVERABLE_REFUSALS", "attach_selected_documents", "attachment_parameters", "carried_receipts",
    "carry_owner_receipts", "first_start_selection", "fold_document_import", "managed_keep_places", "opens_release",
    "selection_options",
    "material_id", "release_owed_imports", "retained_inputs", "select_inputs", "task_document_imports",
    "task_inputs",
]
