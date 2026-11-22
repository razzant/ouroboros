"""Propose exact files of a finished task and carry out the owner's choice in Storage.

A proposal names each file's role and loss and the report, method and result that stay.
The host can prove file identity and protect known record carriers. The author must
recognize irreplaceable history copied into arbitrary evidence. Selected bytes are hashed
before locking. The inode, size and mtime bind the choice across rename, which changes
ctime. Native name witnesses preserve the directory's lookup rules after the move.

Only Storage Free's manual selection with accept_loss reaches release. Keep and whole-tree
closure are checked before and under the admission barrier. A Continue claim already
recorded holds the release. A later enqueue waits for the barrier and sees the receipts.
Holding queue, then the task's custody (Keep's lock) and each line row's (a Continue claim's),
release rechecks the line's signed reads and the offer and durably records its
slots before moving any source. The released/ staging shell separates these files from
scratch ownership markers. Recovery accounts only for observed moves and hands proved
slots to custody trash. Unexplained bytes stay staged, and unmoved files need a new choice.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import os
import pathlib
import re
import stat
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.confined_files import confined_directory
from ouroboros.payload_release_state import (
    KEY, PublicationReleases, ReleaseUnavailable, file_identity, name_witness, plain_path, regular_file,
    release_record, released_files, staging_dir, store_of, trash_dir,
)
from ouroboros.utils import utc_now_iso

__all__ = ["FLAG", "MARK", "candidates", "free_group", "holds", "offered", "propose", "protected", "recover",
           "release", "settle"]

log = logging.getLogger(__name__)

FLAG = "payload_offers"  # the store's derived flag over it (``store.FLAG_COLUMNS``: ``has_payload_offer``)
MARK = "+payload-"  # staging and trash entry names: '<task>+payload-<offer>-<op>'; '+' is in no task id
RETAINED_ROLES = ("report", "method", "result")
TASK_RESULT_REF = "task_result"  # the task's own answer as the retained result
MAX_FILES = 200
# Always protected, whatever an author says: model inputs and outputs, history, review, memory, manifests,
# summaries and patches, chat media and attachments; every ``source_handles`` subtree but raw QA
# (``test-evidence``, its run records and source recipe excluded) and a proven result-material copy.
_PROTECTED_TOP = frozenset({"chat_media", "attachments"})
_PROTECTED_NAMES = frozenset({".artifact_manifest.json", ".artifact_manifest.json.lock", ".scratch_manifest.json",
                              "verification_receipts.jsonl", "verification_ledger.json", "memory_export.json",
                              "workspace_patch.json", "summary.json", "conversation-wire.jsonl"})
_RESULT_MATERIAL = re.compile(r"result-material-([0-9a-f]{16})-([0-9a-f]{64})\.[A-Za-z0-9]{1,12}")


def _test_evidence_records() -> frozenset:
    """A test run's own records (``test_evidence``, ``safe_test_supervisor``) and its source recipe packs."""
    from ouroboros import test_evidence as te

    return frozenset({"result.json", "source.json", "owner.json", "base.pack", "index.pack", te.EVIDENCE_RUN_MARKER,
                      te.PUBLISHED_DIAGNOSTICS_MARKER, te.EVIDENCE_POINTER, te.DURATIONS_POINTER, te.DURATIONS_FILE})


def protected(rel: str) -> str:
    """Why a store-relative path may never be released (``""``: it may be proposed). A carrier protected by its
    canonical name is protected in every subtree. Result-material names also need a verified immutable digest."""
    parts = pathlib.PurePosixPath(rel).parts
    name = parts[-1] if parts else ""
    if (name in _PROTECTED_NAMES or name.endswith("manifest.json") or name.endswith(".patch")
            or (name.startswith("subagent_patch_verdict_") and name.endswith(".json"))):
        return "protected_record"
    if parts and parts[0] in _PROTECTED_TOP:
        return "protected_input"
    if parts and parts[0] == "source_handles":
        if len(parts) > 2 and parts[1] == "test-evidence":
            return "protected_record" if name in _test_evidence_records() else ""
        if len(parts) == 3 and parts[1] == "tool_results" and _RESULT_MATERIAL.fullmatch(name):
            return ""
        return "protected_source"
    return ""


def _protected_object(store: pathlib.Path, rel: str) -> str:
    """A reserved spelling can address a real entry stored with different case. Ask its own parent
    which object that spelling opens. Distinct case-sensitive siblings remain independently selectable."""
    parts = rel.split("/")
    if reason := protected(rel):
        return reason
    def spelling(index, candidates):
        part = parts[index]
        parent = store.joinpath(*parts[:index])
        for candidate in candidates:
            if part != candidate and part.casefold() == candidate.casefold():
                try:
                    if os.path.samestat(os.lstat(parent / part), os.lstat(parent / candidate)):
                        return candidate
                except FileNotFoundError:
                    pass
        return part

    try:
        parts[0] = spelling(0, (*_PROTECTED_TOP, "source_handles"))
        if parts[0] == "source_handles" and len(parts) > 1:
            parts[1] = spelling(1, ("test-evidence", "tool_results"))
        name = parts[-1]
        candidates = set(_PROTECTED_NAMES)
        if parts[:2] == ["source_handles", "test-evidence"]:
            candidates.update(_test_evidence_records())
        lower = name.casefold()
        for suffix in ("manifest.json", ".patch"):
            if lower.endswith(suffix):
                candidates.add(name[:-len(suffix)] + suffix)
        prefix = "subagent_patch_verdict_"
        if lower.startswith(prefix) and lower.endswith(".json"):
            candidates.add(prefix + name[len(prefix):-5] + ".json")
        parts[-1] = spelling(len(parts) - 1, candidates)
        canonical = "/".join(parts)
        return protected(canonical) or ("noncanonical_path" if canonical != rel else "")
    except OSError:
        return "unreadable"


def _material_proven(path, info, digest):
    if not path.startswith("source_handles/tool_results/"):
        return True
    match = _RESULT_MATERIAL.fullmatch(path.rsplit("/", 1)[-1])
    return bool(match and not (info.st_mode & 0o222) and match[1] == match[2][:16] and match[2] == digest)


def _listing_row(store: pathlib.Path, rel: str) -> Dict[str, Any]:
    """The identity fields the store listing gives this file (``artifacts.collect_task_artifact_records``),
    so the view can show it released with the same identity a parent's disposition hashed."""
    if rel.startswith("source_handles/"):
        return {}  # the store walk excludes this subtree even when it holds a real regular file
    row: Dict[str, Any] = {"kind": "task_artifact", "name": rel.rsplit("/", 1)[-1]}
    if "/" in rel:
        return {**row, "relpath": rel}
    from ouroboros.confined_files import read_confined_json

    manifest = read_confined_json(store.parents[2], (*store.parts[-3:], ".artifact_manifest.json"), strict=True) or {}
    registered = (manifest.get("artifacts") or {}).get(rel) or {}
    row["kind"] = str(registered.get("kind") or "task_artifact")
    if registered.get("immutable"):
        row.update({key: registered.get(key) for key in ("immutable", "size", "sha256")})
    return row


def _words(value: Any) -> str:
    return str(value or "").strip()  # authored consequences stay whole, including deliberate line breaks


def _answer(document: Any) -> Optional[str]:
    """The SHA-256 of the task's own answer, None when it has none."""
    text = str((document or {}).get("result") or "") if isinstance(document, dict) else ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text.strip() else None


def _fingerprint(offer: Dict[str, Any]) -> str:
    basis = {"files": offer.get("files"), "retained": offer.get("retained"), "id": offer.get("id")}
    return hashlib.sha256(json.dumps(basis, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _positions(rows: Any, store=None) -> set:
    """Both the stored spelling and the filesystem object. Old offers may have used another spelling."""
    positions = set()
    for item in rows if isinstance(rows, list) else ():
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            if item.get("role") == "result" and item["path"] == TASK_RESULT_REF and "identity" not in item:
                continue  # the retained answer is a document field, outside the file namespace
            positions.add(item["path"])
            stamp = item.get("identity")
            if isinstance(stamp, list) and len(stamp) == 4 and all(isinstance(n, int) for n in stamp):
                positions.add(tuple(stamp[:2]))
            if store is not None:
                info, why = regular_file(store, item["path"])
                if info is not None:
                    positions.add((info.st_dev, info.st_ino))
                elif why != "missing":
                    raise OSError("stored position cannot be identified")
    return positions


def _conflict(store, offer: dict, record: dict, *, replace: str = "") -> str:
    files, kept = _positions(offer["files"]), _positions(offer["retained"])
    try:
        state = PublicationReleases({KEY: record}, store)
        selected = {state.address(row["path"]) for row in offer["files"]}
        retained = {state.address(ref["path"]) for ref in offer["retained"] if ref.get("identity")}
        if (selected | retained) & state.pending.keys():
            return "release_pending"
        if files & kept or retained & state.released.keys():
            return "retained_is_released"
        if selected & state.released.keys():
            return "path_already_offered_or_released"
        for pid, other in record["offers"].items():
            if pid == offer["id"] or not isinstance(other, dict):
                continue
            if files & _positions(other.get("retained"), store):
                return "retained_by_other_offer"
            if pid != replace and other.get("state") == "open":
                selected = _positions(other.get("files"), store)
                if files & selected:
                    return "path_already_offered_or_released"
                if kept & selected:
                    return "retained_offered_for_release"
    except (ValueError, OSError):
        return "release_address_unproven"
    return ""


def _write(root: Any, task_id: str, change: Callable[[Dict[str, Any], Dict[str, Any]], Optional[str]]) -> str:
    """One write of the task's record: ``change(document, record)`` mutates the record or returns why not;
    the store's writer derives the flag from the written document. ``""`` once written."""
    from ouroboros.task_store import patch_document

    refused: List[str] = []

    def mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        record = release_record(current)
        why = change(current, record) if current else "task_result_unreadable"
        if why:
            refused.append(why)
            return None
        return {**current, KEY: record}

    patch_document(root, task_id, mutate)
    return refused[0] if refused else ""


# ---- the proposal ---------------------------------------------------------------------------------------


def _proposed_files(store: pathlib.Path, files: List[Any], refused: List[Dict[str, str]], names: dict) -> List[Dict[str, Any]]:
    from ouroboros.artifacts import stream_artifact_file

    rows: List[Dict[str, Any]] = []
    for item in files:
        item = item if isinstance(item, dict) else {}
        rel, role, loss = plain_path(item.get("path")), _words(item.get("role")), _words(item.get("loss"))
        why = "bad_path" if not rel else "duplicate_path" if any(row["path"] == rel for row in rows) else (
            "role_and_loss_required" if not (role and loss) else protected(rel))
        if not why:
            info, why = regular_file(store, rel, names=names)
        if not why:
            why = _protected_object(store, rel)
        if not why:
            parts = rel.split("/")
            try:
                with confined_directory(store.parents[2], (*store.parts[-3:], *parts[:-1])) as directory:
                    handle, opened = directory.open_for_read(parts[-1])
                    with handle:
                        if file_identity(opened) != file_identity(info):
                            raise OSError("payload changed before hashing")
                        measured = stream_artifact_file(handle)
                after, why = regular_file(store, rel)
                if not why and (file_identity(after) != file_identity(info) or measured["size"] != info.st_size):
                    why = "payload_changed"
                if not _material_proven(rel, info, measured["sha256"]):
                    why = "result_material_unproven"
                listing, witness = _listing_row(store, rel), name_witness(store / rel)
            except (OSError, ValueError, AttributeError):
                why = "unreadable"
        if why:
            refused.append({"path": str(item.get("path") or ""), "reason": why})
        else:
            rows.append({"path": rel, "role": role, "loss": loss, "identity": file_identity(info),
                         "row": listing, "sha256": measured["sha256"], "name_witness": witness})
    return rows


def _retained_refs(store: pathlib.Path, document: Dict[str, Any], retained: Any, released: set,
                   refused: List[Dict[str, str]], names: dict) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    refs = retained if isinstance(retained, dict) else {}
    for role in RETAINED_ROLES:
        named = refs.get(role)
        named = [named] if isinstance(named, str) else named if isinstance(named, list) else []
        if not named:
            refused.append({"path": f"retained.{role}", "reason": "retained_ref_required"})
        for raw in named:
            if role == "result" and raw == TASK_RESULT_REF:
                answer = _answer(document)
                if answer:
                    kept.append({"role": role, "path": TASK_RESULT_REF, "sha256": answer})
                else:
                    refused.append({"path": raw, "reason": "task_result_empty"})
                continue
            rel = plain_path(raw)
            info, why = regular_file(store, rel, names=names) if rel else (None, "bad_path")
            why = "retained_is_released" if rel in released else why
            if why:
                refused.append({"path": f"retained.{role}:{raw}", "reason": why})
            else:
                kept.append({"role": role, "path": rel, "identity": file_identity(info)})
    return kept


def propose(root: Any, task_id: str, files: Any, retained: Any, consequence: Any, *, proposer: str = "",
            replace_offer: str = "", withdraw_offer: str = "") -> Dict[str, Any]:
    """Record one proposal, or refuse it whole with each path's reason: ``{"ok", "offer", "files", "refused"}``.
    The caller's right to read this task is the tool's to check (``tools.control_storage``)."""
    from ouroboros.owner_keep import gate
    from ouroboros.task_status import FINAL_STATUSES
    from ouroboros.task_store import read_document

    try:
        store = store_of(root, task_id)
    except ValueError:
        return {"ok": False, "error": "BAD_TASK_ID"}
    document = read_document(root, task_id)
    if not isinstance(document, dict) or not document:
        return {"ok": False, "error": "TASK_RESULT_UNREADABLE"}
    if str(document.get("status") or "") not in FINAL_STATUSES:
        return {"ok": False, "error": "TASK_NOT_FINISHED"}
    if replace_offer and withdraw_offer:
        return {"ok": False, "error": "SELECTION_CONFLICT"}
    if withdraw_offer:
        def withdraw(_current, record):
            old = record["offers"].get(withdraw_offer)
            if not isinstance(old, dict) or old.get("state") != "open":
                return "offer_not_open"
            record["offers"][withdraw_offer] = {**old, "state": "withdrawn", "closed_at": utc_now_iso(),
                                                "withdrawn_by": str(proposer or "")}
            return ""
        with gate(root, task_id, timeout_sec=2.0) as why:
            if why == "busy":
                return {"ok": False, "error": "TASK_BUSY"}
            refusal = _write(root, task_id, withdraw)
        return {"ok": not refusal, **({"error": refusal.upper()} if refusal else {"withdrawn": withdraw_offer})}
    if not isinstance(files, list) or not files or len(files) > MAX_FILES:
        return {"ok": False, "error": "FILES_REQUIRED", "limit": MAX_FILES}
    refused: List[Dict[str, str]] = []
    names = {}
    rows = _proposed_files(store, files, refused, names)
    kept = _retained_refs(store, document, retained, _positions(rows) | set(released_files(document)), refused, names)
    summary = _words(consequence)
    if not summary:
        refused.append({"path": "consequence", "reason": "consequence_required"})
    if refused:
        return {"ok": False, "error": "PROPOSAL_REFUSED", "refused": refused}
    offer = {"id": uuid.uuid4().hex[:12], "state": "open", "proposed_at": utc_now_iso(), "proposed_by": str(proposer or ""),
             "consequence": summary, "files": rows, "retained": kept}

    def add(current: Dict[str, Any], record: Dict[str, Any]) -> str:
        old = record["offers"].get(replace_offer) if replace_offer else None
        if replace_offer and (not isinstance(old, dict) or old.get("state") != "open"):
            return "offer_not_open"
        if any(ref["role"] == "result" and ref["path"] == TASK_RESULT_REF and ref["sha256"] != _answer(current) for ref in kept):
            return "retained_changed"
        if replace_offer:
            record["offers"][replace_offer] = {**old, "state": "superseded", "replaced_by": offer["id"],
                                               "closed_at": utc_now_iso()}
            offer["replaces"] = replace_offer
        record["offers"][offer["id"]] = offer
        return ""

    with gate(root, task_id, timeout_sec=2.0) as why:
        if why == "busy":
            return {"ok": False, "error": "TASK_BUSY"}
        # The custody gate serializes proposals and moves. Resolve native file identities outside the SQL writer.
        conflict = _conflict(store, offer, release_record(read_document(root, task_id)), replace=replace_offer)
        refusal = conflict or _write(root, task_id, add)
    if refusal:
        return {"ok": False, "error": refusal.upper()}
    return {"ok": True, "offer": offer["id"], "owner_keep": why == "owner_keep",
            "files": [{"path": row["path"], "bytes": row["identity"][2]} for row in rows]}


# ---- what the Storage preview offers --------------------------------------------------------------------


def offered(root: Any) -> Dict[str, Dict[str, Any]]:
    """The flagged tasks (``store.FLAG_COLUMNS``), one indexed read; ``{}`` when the store cannot be read."""
    from ouroboros import obligations

    try:
        return obligations.members(root, FLAG)
    except Exception:
        log.debug("payload offers unreadable", exc_info=True)
        return {}


def _verify(store: pathlib.Path, document: Dict[str, Any], record: Dict[str, Any], pid: str, offer: Dict[str, Any]) -> str:
    """Each file and retained ref still the one proposed, none protected, none another offer's to keep, the
    retained answer the same text. This compares directory metadata and the answer's hash."""
    if (not isinstance(offer.get("files"), list) or not offer["files"]
            or not isinstance(offer.get("retained"), list) or not offer["retained"]):
        return "proposal_unreadable"
    for row in [*offer["files"], *offer["retained"]]:
        if not isinstance(row, dict) or not plain_path(row.get("path")):
            return "proposal_unreadable"
        if not (row.get("role") == "result" and row["path"] == TASK_RESULT_REF and "sha256" in row):
            identity = row.get("identity")
            if not isinstance(identity, list) or len(identity) != 4 or not all(isinstance(n, int) for n in identity):
                return "proposal_unreadable"
    if any(not row.get("role") or not row.get("loss") for row in offer["files"]):
        return "proposal_unreadable"
    if not set(RETAINED_ROLES) <= {ref.get("role") for ref in offer["retained"]}:
        return "proposal_unreadable"
    conflict = _conflict(store, {**offer, "id": pid}, record)
    if conflict:
        return conflict
    names = {}
    selected_objects = set()
    for row in offer.get("files") or ():
        path = str(row.get("path") or "")
        info, why = regular_file(store, path, names=names)
        if why or _protected_object(store, path):
            return f"payload_{why or 'protected'}"
        if not _material_proven(path, info, row.get("sha256")):
            return "result_material_unproven"
        object_id = (info.st_dev, info.st_ino)
        if object_id in selected_objects:
            return "duplicate_path"
        selected_objects.add(object_id)
        if file_identity(info) != row.get("identity"):
            return "payload_changed"
        try:
            if row.get("name_witness") and name_witness(store / path) != row["name_witness"]:
                return "payload_changed"
        except OSError:
            return "payload_unreadable"
    for ref in offer.get("retained") or ():
        if ref.get("role") == "result" and ref["path"] == TASK_RESULT_REF:
            if _answer(document) != ref.get("sha256"):
                return "retained_changed"
            continue
        info, why = regular_file(store, ref["path"], names=names)
        if why or file_identity(info) != ref.get("identity"):
            return "retained_changed"
    return ""


def holds(root: Any, task_id: str, document: Any, live: Callable[[str], Any], *, continuation_reads: Any = None) -> str:
    """Why this task's files cannot go now: its record, the owner's keep, its own terminal proof, then the
    whole work's closure (tree, line, Continue successors, reviews, custody). Never inside a custody lock."""
    from ouroboros.owner_keep import OWNER_KEEP, is_kept
    from ouroboros.review_thread_retention import _queue_census, tree_hold
    from ouroboros.subagent_worktrees import _task_proven_over
    from ouroboros.task_status import FINAL_STATUSES

    if not isinstance(document, dict) or not document:
        return "task_result_unreadable"
    if str(document.get("status") or "") not in FINAL_STATUSES:
        return "task_not_terminal"
    if is_kept(root, task_id):
        return OWNER_KEEP
    # The locked supervisor probe consumes its pre-interlock preparation once per task.
    # Both readers share this observation only within this invocation of holds().
    live = functools.cache(live)
    if not _task_proven_over(task_id, root, live):
        return "not_proven_over"
    return tree_hold(root, task_id, document, live=live, census=_queue_census(root), every_offer=True,
                     continuation_reads=continuation_reads)[0]


def candidates(root: Any, live: Callable[[str], Any]) -> List[Dict[str, Any]]:
    """Every open proposal, with what holds it (``hold``) or what the owner would choose (``manual``), and every
    staging an operation could not explain (held, never offered)."""
    from ouroboros.task_store import read_document

    found: List[Dict[str, Any]] = []
    for task_id in sorted(offered(root)):
        try:
            store, document = store_of(root, task_id), read_document(root, task_id)
        except Exception:
            store, document = None, None
        record = release_record(document)
        try:
            hold = holds(root, task_id, document, live) if store is not None else "task_result_unreadable"
        except Exception:  # a proof that cannot be read holds, shown with its reason
            log.debug("payload release holds of %s unreadable", task_id, exc_info=True)
            hold = "record_unreadable"
        for pid, offer in sorted(record["offers"].items()):
            if not isinstance(offer, dict) or offer.get("state") != "open":
                continue
            try:
                why = hold or _verify(store, document, record, pid, offer)
                sizes = [int(row["identity"][2]) for row in offer["files"]]
            except (KeyError, TypeError, ValueError, IndexError, OSError):
                why, sizes = "proposal_unreadable", []
            found.append({
                "key": f"payload:{task_id}{MARK}{pid}", "task_id": task_id, "path": str(store), "bytes": sum(sizes), "hold": why,
                "row": {"task": task_id, "offer": pid, "fingerprint": _fingerprint(offer)},
                "manual": None if why else {
                    "signature": None, "unsaved_count": 0, "unsaved_bytes": 0, "unsaved": [], "keep_branch": False,
                    "payload_release": True, "consequence": offer.get("consequence", ""), "proposed_at": offer.get("proposed_at"),
                    "proposed_by": offer.get("proposed_by", ""), "root_task_id": str(document.get("root_task_id") or task_id),
                    "files": [{"path": row["path"], "bytes": size, "role": row["role"], "loss": row["loss"]}
                              for row, size in zip(offer["files"], sizes)],
                    "retained": [{"role": ref["role"], "path": ref["path"],
                                  "bytes": ref["identity"][2] if ref.get("identity") else None} for ref in offer["retained"]]},
            })
        for op, operation in sorted(record["operations"].items()):
            held = operation.get("staging_held") if isinstance(operation, dict) else None
            if held:
                found.append({"key": f"held:{operation.get('staging') or op}", "task_id": task_id, "hold": "release_unexplained",
                              "path": str(staging_dir(root) / str(operation.get("staging") or "")), "manual": None, "row": {},
                              "bytes": sum(int(row.get("bytes") or 0) for row in held)})
    return found


def free_group(root: Any, live: Callable[[str], Any], group: Dict[str, Any], make: Callable[..., Any]) -> Dict[str, Any]:
    """Storage -> Free's group of proposals: each one object of the manual form (``make(key, path, task)`` builds
    the preview's candidate), never in the pool, a saved group or the mind's request; a held one is named with
    its reason and offers nothing. Sized here from the proposal's own identities (no walk)."""
    from ouroboros.storage_free import _KEPT_REASONS, _reason

    for found in candidates(root, live):
        item = make(found["key"], pathlib.Path(found["path"]), found["task_id"])
        item.kind, item.ordinary, item.bytes, item.row = "payload", False, found["bytes"], found["row"]
        item.cause = found["hold"] or None
        hold = found["hold"]
        item.reason = "payload_offer" if not hold else hold if hold in _KEPT_REASONS else _reason(hold)
        item.manual = found["manual"]
        if hold:
            group["unreadable" if hold == "task_result_unreadable" else "in_use"] += 1
        group["excluded"].append(item)
    return group


# ---- the effect -------------------------------------------------------------------------------------------


def _hold_line(root: Any, task_id: str, reads: Any, held: contextlib.ExitStack) -> str:
    """Hold, beside the task's own custody (Keep's lock, ``gate``), the custody of every other row its line proof
    read before the queue (``tree_hold``'s signed reads: root, retries, Continue successors), the lock
    ``owner_continue.claim_on_predecessor`` takes, each without waiting; then those reads must be unchanged.
    An unsigned or unreadable row proves nothing. A refusal records and moves nothing."""
    from ouroboros.task_custody import task_custody_lock
    from supervisor.task_ownership import BUSY, UNCHANGED

    rows = set(reads.rows)
    if (reads.errors or task_id not in rows or not rows <= set(reads.stamps)
            or any(reads.stamps[member] is None for member in rows)):
        return "continuation_unverified"
    try:
        for member in sorted(rows - {task_id}):
            if not held.enter_context(task_custody_lock(root, member, timeout_sec=0)):
                return "custody_busy"
    except ValueError:
        return "continuation_unverified"
    except OSError:  # CustodyBusy: a platform refusal; a later choice retries
        return "custody_busy"
    state = reads.recheck(tuple(sorted(rows)))
    return "" if state == UNCHANGED else "store_busy" if state == BUSY else "changed"


def release(root: Any, row: Dict[str, Any], live: Callable[[str], Any], guard: Callable[[], Any], *,
            actor: str = "owner") -> Tuple[str, int]:
    """The owner's exact choice of one previewed proposal: ``(why it stays, or "", bytes really moved)``.
    ``release_partial`` when some moved: the rest is never deleted later without a new choice."""
    from ouroboros import maintenance
    from ouroboros.owner_keep import gate
    from ouroboros.store import StoreBusy, acquisition_budget
    from ouroboros.task_custody import _INTERLOCK_LOCK_TIMEOUT_SEC
    from ouroboros.task_status import FINAL_STATUSES
    from ouroboros.task_store import read_document
    from supervisor.task_ownership import TaskOwnershipRead

    if actor != "owner":
        return "not_owner", 0  # the mind's request runs the ordinary pool; this form is the owner's alone
    task_id, pid = str(row.get("task") or ""), str(row.get("offer") or "")
    try:
        store = store_of(root, task_id)
    except ValueError:
        return "task_result_unreadable", 0
    continuation_reads = TaskOwnershipRead(root)
    why = holds(root, task_id, read_document(root, task_id), live, continuation_reads=continuation_reads)
    if why:
        return why, 0
    op = uuid.uuid4().hex[:12]
    staging = staging_dir(root) / f"{task_id}{MARK}{pid}-{op}"
    refusal, accounted = "", None
    with guard() as allowed:  # admission's barrier: no Continue, retry or assignment enters the tree meanwhile
        if not allowed:
            return "generation_closed", 0
        try:
            with acquisition_budget(0):
                why = holds(root, task_id, read_document(root, task_id), live, continuation_reads=continuation_reads)
        except StoreBusy:
            return "store_busy", 0
        if why:
            return why, 0
        with gate(root, task_id, timeout_sec=_INTERLOCK_LOCK_TIMEOUT_SEC) as kept, \
                contextlib.ExitStack() as line:  # queue -> custody: the task's (Keep), then its line's (Continue)
            if kept:
                return kept, 0
            why = _hold_line(root, task_id, continuation_reads, line)
            if why:
                return why, 0  # nothing recorded or moved: each source keeps its name and readers
            document = read_document(root, task_id)
            record = release_record(document)
            offer = record["offers"].get(pid)
            if str((document or {}).get("status") or "") not in FINAL_STATUSES:
                return "task_not_terminal", 0
            if not isinstance(offer, dict) or offer.get("state") != "open" or _fingerprint(offer) != row.get("fingerprint"):
                return "changed", 0
            why = _verify(store, document, record, pid, offer)
            if why:
                return why, 0
            try:
                slots = [{"path": item["path"], "identity": item["identity"], "slot": str(index),
                          "name_witness": item.get("name_witness") or name_witness(store / item["path"])}
                         for index, item in enumerate(offer["files"])]
            except OSError:
                return "payload_unreadable", 0

            def decide(_current: Dict[str, Any], held: Dict[str, Any]) -> str:
                now = held["offers"].get(pid)
                if not isinstance(now, dict) or now.get("state") != "open" or _fingerprint(now) != row.get("fingerprint"):
                    return "changed"
                held["offers"][pid] = {**now, "state": "closed", "closed_by": op, "closed_at": utc_now_iso()}
                held["operations"][op] = {"offer": pid, "state": "moving", "actor": "owner", "decided_at": utc_now_iso(),
                                          "staging": staging.name, "files": slots}
                return ""

            try:
                why = _write(root, task_id, decide)  # the durable decision, every identity, before any rename
            except Exception as exc:  # nothing moved: the choice is not repeated, a new preview asks again
                return ("store_busy" if type(exc).__name__ == "StoreBusy" else "record_unreadable"), 0
            if why:
                return why, 0
            row["operation"] = op  # links the existing Free outcome to this exact durable decision
            try:
                (staging / "released").mkdir(parents=True)
                for item in slots:
                    info, why = regular_file(store, item["path"])
                    if why or file_identity(info) != item["identity"]:
                        refusal = f"payload_{why}" if why else "payload_changed"
                        break  # nothing after a refusal moves, now or later
                    _move_selected(store / item["path"], staging / "released" / item["slot"], store.parents[2], item["identity"])
            except OSError as exc:
                refusal = f"release_failed:{exc.errno or type(exc).__name__}"
            row["release_cause"] = refusal  # the same Free outcome keeps the failed syscall's diagnostic
            accounted = _account(root, task_id, op, staging)
            try:
                if accounted is not None:
                    _publish(root, staging, accounted["slots"])
            except OSError:  # recorded released, its bytes still staged: the next custody pass hands them over
                log.warning("Released files of %s stay staged until the next custody pass", task_id, exc_info=True)
                accounted = None
    if accounted is None:
        return "release_pending", 0  # moved into its staging; the next custody pass records and hands it over
    if accounted["slots"]:
        maintenance.child.request()  # the trash entry's deletion is the maintenance child's job
    moved = int(accounted["bytes"])
    if not refusal and not accounted["held"]:
        return "", moved
    return ("release_partial" if moved else "release_unexplained" if accounted["held"] else
            refusal or "other"), moved


def _move_selected(source, target, root, stamp):
    with confined_directory(root, source.parent.relative_to(root).parts) as before, \
         confined_directory(root, target.parent.relative_to(root).parts) as after:
        before.move_selected(source.name, after, target.name, stamp)


def settle(root: Any, item: Any, live: Callable[[str], Any], guard: Callable[[], Any], *, owner: bool) -> Tuple[str, int]:
    """Storage -> Free's run of one chosen proposal (its preview ``row``); a failure is ``other``, never raised."""
    try:
        return release(root, item.row, live, guard, actor="owner" if owner else "mind")
    except Exception:
        log.warning("Stored files of %s were not released", getattr(item, "key", ""), exc_info=True)
        return "other", 0


def _account(root: Any, task_id: str, op: str, staging: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Record what the staging of OP actually holds: a slot whose staged file keeps the decided identity is
    released; a slot that holds other bytes, or a name the decision never made, is ``staging_held`` (never
    published); a slot it does not hold is ``not_moved`` (its source still there) or ``unknown_missing``. One
    write, idempotent: even under Keep this only records an already performed move, never deletes or moves
    another source. ``slots`` maps the released slots to their decided identities; None is unrecorded."""
    shell = staging / "released"
    try:
        present = set(os.listdir(shell))
    except FileNotFoundError:
        present = set()
    except OSError:
        return None
    try:
        beside = set(os.listdir(staging)) - {"released"}  # never a slot: named held, never published
    except FileNotFoundError:
        beside = set()
    except OSError:
        return None
    out: Dict[str, Any] = {}

    def account(_document: Dict[str, Any], record: Dict[str, Any]) -> str:
        operation = record["operations"].get(op)
        if not isinstance(operation, dict):
            return "operation_unknown"
        if operation.get("state") != "moving":
            moved = set(operation.get("moved") or ())
            slots = {str(item["slot"]): item["identity"] for item in operation.get("files") or () if item["path"] in moved}
            out.update(slots=slots, bytes=sum(int(identity[2]) for identity in slots.values()),
                       held=list(operation.get("staging_held") or ()))
            return "accounted_already"
        offer = record["offers"].get(str(operation.get("offer") or "")) or {}
        meaning = {row.get("path"): row for row in offer.get("files") or () if isinstance(row, dict)}
        done, missing, held, slots, size = [], [], [], {}, 0
        for item in operation.get("files") or ():
            path, slot = str(item.get("path") or ""), str(item.get("slot"))
            try:
                info = os.lstat(shell / slot)
            except FileNotFoundError:
                info = None
            except OSError:
                return "staging_unreadable"  # no accounting write: a later recovery retries this slot
            if info is not None and stat.S_ISREG(info.st_mode) and file_identity(info) == item.get("identity"):
                done.append(path)
                slots[slot] = item["identity"]
                size += int(item["identity"][2])
                source = meaning.get(path) or {}
                record["released"][path] = {"operation": op, "offer": operation.get("offer"), "at": utc_now_iso(),
                                            "bytes": item["identity"][2], "role": source.get("role", ""),
                                            "loss": source.get("loss", ""), "sha256": source.get("sha256"),
                                            "row": source.get("row") or {}, "name_witness": item.get("name_witness")}
            elif info is not None:
                held.append({"path": path, "slot": slot, "reason": "staged_changed", "bytes": info.st_size})
            else:
                source, why = regular_file(store_of(root, task_id), path)
                if why == "unreadable":
                    return "source_unreadable"
                reason = ("not_moved" if source is not None and file_identity(source) == item.get("identity")
                          else "unknown_missing" if why == "missing" else "source_changed")
                missing.append({"path": path, "reason": reason})
        decided = {str(item.get("slot")) for item in operation.get("files") or ()}
        held += [{"slot": name, "reason": "staging_unknown", "bytes": _size(shell / name)} for name in sorted(present - decided)]
        held += [{"entry": name, "reason": "staging_unknown", "bytes": _size(staging / name)} for name in sorted(beside)]
        record["operations"][op] = {**operation, "state": "accounted", "accounted_at": utc_now_iso(), "moved": done,
                                    "not_moved": missing, **({"staging_held": held} if held else {})}
        out.update(slots=slots, bytes=size, held=held)
        return ""

    try:
        why = _write(root, task_id, account)
    except Exception:
        log.warning("Payload release %s of %s not accounted yet", op, task_id, exc_info=True)
        return None
    if why not in ("", "accounted_already"):
        log.warning("Payload release %s of %s not accounted: %s", op, task_id, why)
        return None
    return out


def _size(path: pathlib.Path) -> int:
    with contextlib.suppress(OSError):
        return os.lstat(path).st_size
    return 0


def _publish(root: Any, staging: pathlib.Path, slots: Dict[str, List[int]]) -> None:
    """Exactly the accounted SLOTS into the custody trash entry of the same name (its remover deletes them off
    the serving process); anything else in the staging stays there, held. An empty staging goes."""
    shell = staging / "released"
    try:
        present, beside = set(os.listdir(shell)), set(os.listdir(staging)) - {"released"}
    except FileNotFoundError:
        return
    wanted, refused = [], []
    for slot, identity in slots.items():  # decided identity observed now; no proof keeps the slot staged
        if slot not in present:
            continue
        try:
            info = os.lstat(shell / slot)
        except OSError:
            refused.append(slot)
            continue
        (wanted if stat.S_ISREG(info.st_mode) and file_identity(info) == identity else refused).append(slot)
    if present and not beside and not refused and present <= slots.keys():
        trash_dir(root).mkdir(parents=True, exist_ok=True)
        if not os.path.lexists(trash_dir(root) / staging.name):
            os.replace(staging, trash_dir(root) / staging.name)  # the whole staging, nothing unexplained in it
            return
    if wanted:
        target = trash_dir(root) / staging.name / "released"
        target.mkdir(parents=True, exist_ok=True)
        for slot in wanted:
            if os.path.lexists(target / slot):  # preserve an existing slot, never replace it
                refused.append(slot)
            else:
                os.rename(shell / slot, target / slot)
    with contextlib.suppress(OSError):
        if not os.listdir(shell):
            shell.rmdir()
            staging.rmdir()  # nothing held: no staging left behind
    if refused:
        raise ReleaseUnavailable("staged_unproven")  # keep it staged for the next custody pass


def recover(root: Any) -> Dict[str, Any]:
    """Finish what an interrupted release left: account each staging of this kind from its contents and move its
    accounted slots to the trash (a keep set since holds them in staging), and account a decision whose staging
    never appeared. Never renames a source and never publishes what the decision cannot explain. Runs on the
    custody pass before the scratch release (``workspace_preflight.release_finished_scratch``)."""
    from ouroboros import maintenance
    from ouroboros.owner_keep import gate
    from ouroboros.task_custody import CustodyBusy
    from ouroboros.task_store import read_document

    report: Dict[str, Any] = {"accounted": [], "held": [], "errors": []}
    base = staging_dir(root)
    seen: set = set()
    with contextlib.suppress(OSError):
        for staged in sorted(base.glob(f"*{MARK}*")) if base.is_dir() and not base.is_symlink() else ():
            task_id, _, rest = staged.name.partition(MARK)
            op = rest.rsplit("-", 1)[-1]
            seen.add((task_id, op))
            try:
                with gate(root, task_id, timeout_sec=0) as why:
                    if why == "busy":
                        continue  # a release in flight holds this very lock
                    accounted = _account(root, task_id, op, staged)
                    if accounted is None:
                        continue
                    if not why:
                        _publish(root, staged, accounted["slots"])
                        if accounted["slots"]:
                            maintenance.child.request()
                report["accounted"].append(staged.name)
                if accounted["held"]:
                    report["held"].append(staged.name)
            except Exception as exc:
                report["errors"].append(f"{staged.name}: {type(exc).__name__}")
    for task_id in offered(root):
        try:
            operations = release_record(read_document(root, task_id))["operations"]
        except Exception:
            continue
        for op, operation in operations.items():
            if (task_id, op) in seen or not isinstance(operation, dict) or operation.get("state") != "moving":
                continue
            try:
                with gate(root, task_id, timeout_sec=0) as why:
                    if why != "busy" and _account(root, task_id, op, base / str(operation.get("staging") or "")) is not None:
                        report["accounted"].append(f"{task_id}:{op}")
            except CustodyBusy as exc:  # the lock names a per-task I/O failure, not an overall failed sweep
                report["errors"].append(f"{task_id}:{op}: {type(exc).__name__}")
    return report
