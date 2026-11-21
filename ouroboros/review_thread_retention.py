"""Keep a finished reviewer thread's complete engine export, then let the engine retire the thread.

A plan-review seat's sticky Claudexor thread (``review_thread_continuity``) outlives its task in the
engine (#1180). A pass of the maintenance child's ``runtime_retirement`` (``maintenance_retirement``) takes
it up only when the owning task and its root are proven over and the whole work has ended (``tree_hold``):
the cascade's own durable-plus-queue membership walk, each member settled or ended through the retry its
effective result follows, none with open review work, the tree's custody settled, the line not continuable,
and the same closure for the tree of every root the owner's Continue admitted after it (its verified chain).
Its threads are those its plan seats' replies named and, for a seat whose reply named none (it stopped, stopped again
after an explicit continuation), the one its own START recorded (``seat_threads``). Only a complete, retirable
manifest (Claudexor wire seam v2/v3) of the store the gateway negotiated, naming every run this host started in the
thread, is copied: each file item streams into the owning task's
``source_handles/delegated_activity`` (``artifacts.stream_actor_source``; checked against the transfer's
``Content-Length``/``ETag`` and the manifest's size and sha256 where it gives them), beside the exact manifest
bytes and a receipt listing every item and both digests recomputed from what arrived. The receipt is
referenced from the task document and from each review-ledger row that named the thread
(``historical_source_refs``, role ``thread_export``), then the tree is read again at that release boundary,
outside the keep gate. An item's relative POSIX path (``/``-separated segments of any character but NUL, never
empty, ``.`` or ``..``; a backslash is an ordinary character) is receipt metadata only, never a storage name. An
inventory hold (a malformed, unsupported or partial manifest, a closure missing a host run) keeps the exact
manifest bytes and a receipt naming the hold as the thread's durable held record, examined again only on the
task's next terminal write and only once the engine's manifest bytes changed; a host hold keeps the verified
items for the next pass. The store's typed answer to the export read before any release (``OBSERVED``) is saved on
the same record once, when it changes, scoped to that store and with no operation: a purge is final, an absence is
asked again on the task's next terminal write and says nothing of a purge or of any other store.

The release is one operation in the task document's ``review_thread_releases``, bound under the
owner's keep gate (the task custody lock) before any POST and never re-derived: ``bound`` until the
engine settles it (a 200 naming it, a 409/410 marked ``settled``, or a read of that operation that is
not null), then ``accepted``, ``refused`` or ``abandoned``, written back under the same lock. HTTP runs
outside the lock. An unknown outcome is read again on a later pass, then the same POST is repeated, or
abandoned when the owner's Keep now stands (``owner_keep`` meanwhile answers that thread
``release_unconfirmed``, never kept: a retire already delivered can still win). A settled refusal spends the
operation; only a changed export starts another. Acceptance is the boundary after which the thread's lane is
the engine's to delete, and only a cleanup result the engine already returned is reported removed. The archive
is on disk before any binding (files synced as written, then their folders) and stays readable: each ref names
its task and a ``read_file`` path another task or a fork resolves, and ``read_thread_export`` and
``stream_retained_item`` (which never reads an item whole) read it here.

Candidates are the tasks written terminal since the pass last looked (``task_store.terminal_writes_since``),
each with the timeout-retried predecessors its reciprocal retry links name, and the tasks the step still
waits on, kept on the ``runtime_retirement`` job row. The first pass pins its position and time; a managed import's
prior ``floor`` (``maintenance_retirement.release_opened``) makes it process that older interval in the same pass.
A thread whose host runs started before activation is a candidate only of the one historical walk
(``historical_retirement``), never of this pass. No timer, poller or retry count: a waiting task is asked again whenever the maintenance child runs
(``owed``), never a reason to start one. A pass its caller stops between tasks or between a task's operations (a
delegate capture arrived) leaves the rest waiting for that next run. Each candidate's managed document imports go
through the same pass (``delegate_documents.release_owed_imports``, holds named ``document_<why>``); a task waits
while either still does. Why the engine could not be asked at all
(``engine_hold``) stands on the job row beside the tasks it leaves waiting, written when it changes. A pass reports what
it newly saved of the engine's answers as ``observed``, which its caller logs. Each candidate also owes the engine one
``taken`` decision per finished delegated result it holds whole, under the same tree proof (``delegate_result_taken``);
the task waits while either duty does. Every other settled task gets it once from the ``walk``; leftovers wait here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger(__name__)

RELEASES = "review_thread_releases"  # task-document key: thread id -> its one current release record
JOB_FIELD = "review_threads"  # the runtime_retirement job row's field: activation, cursor, waiting tasks
CATEGORY = "delegated_activity"
ROLE = "thread_export"
SETTLED = ("accepted", "refused", "abandoned")
# Holds only the engine's inventory answers: the same export, asked again, answers the same.
INVENTORY_HOLDS = ("export_partial", "closure_identity_mismatch", "export_manifest_unreadable",
                   "export_schema_unsupported", "store_mismatch", "export_identity_mismatch", "export_item_malformed",
                   "export_symlink_mismatch")
# The export read's typed answers before any release (wire seam 410, 404) -> the state each leaves on the record.
OBSERVED = {"thread_purged": "purged", "thread_not_found": "not_found"}
_PLANES = frozenset({"sql", "run", "native", "lane", "resource", "projection"})
_KINDS = frozenset({"file", "symlink", "special", "credential_store"})
_DIGESTED = ("file", "symlink")
WAIT, DONE = "wait", "done"


class ExportHeld(ValueError):
    """The export cannot be retained whole; the message is the hold's code, ``detail`` where it names an item and
    field. Nothing is released; the exact manifest bytes and this hold are kept as the thread's held record."""

    def __init__(self, code: str, detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(code)
        self.detail = detail


def export_digest(lines: Iterable[Tuple[str, str, int]]) -> str:
    """The seam's digest: ``"<itemId> <sha256hex> <size>\\n"`` lines sorted by byte order, hashed together."""
    body = sorted(f"{item} {sha} {size}\n".encode("utf-8") for item, sha, size in lines)
    return "sha256:" + hashlib.sha256(b"".join(body)).hexdigest()


def _hex(value: Any) -> str:
    text = str(value or "").removeprefix("sha256:")
    return text if len(text) == 64 and all(char in "0123456789abcdef" for char in text) else ""


def _relative_posix(value: Any) -> bool:
    """A relative POSIX path as the seam promises it: no NUL, not absolute, no empty, ``.`` or ``..`` segment. Every
    other character, a backslash or a newline included, is an ordinary filename byte. The path is receipt METADATA
    only: an item is stored under the name ``artifacts.actor_source_reference`` makes from its item id and suffix,
    never at this path."""
    text = value if isinstance(value, str) else ""
    return bool(text) and "\x00" not in text and not text.startswith("/") and all(
        part not in ("", ".", "..") for part in text.split("/"))


def _size(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def checked_items(manifest: Any, *, thread_id: str, store_id: str) -> List[Dict[str, Any]]:
    """The manifest's items once its identity, its closure's run list and every item's shape match the seam;
    else ``ExportHeld``. A value of another JSON type where the seam names a string or a list is malformed too."""
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1:
        raise ExportHeld("export_schema_unsupported")
    if not store_id or manifest.get("storeId") != store_id:
        raise ExportHeld("store_mismatch")
    if (manifest.get("threadId") != thread_id or not isinstance(manifest.get("exportId"), str)
            or not manifest["exportId"] or not isinstance(manifest.get("closureDigest"), str)
            or not isinstance(manifest.get("items"), list)):
        raise ExportHeld("export_identity_mismatch")
    closure = manifest.get("closure")
    if not isinstance(closure, dict) or not isinstance(closure.get("runs"), list):
        raise ExportHeld("export_identity_mismatch", {"item_id": None, "field": "closure"})
    seen = set()
    for item in manifest["items"]:
        ident = item.get("itemId") if isinstance(item, dict) else None
        field = ("item" if not isinstance(item, dict)
                 else "itemId" if not isinstance(ident, str) or not ident or ident in seen or any(c in ident for c in " \n/")
                 else "plane" if not isinstance(item.get("plane"), str) or item["plane"] not in _PLANES
                 else "kind" if not isinstance(item.get("kind"), str) or item["kind"] not in _KINDS
                 else "path" if not _relative_posix(item.get("path"))
                 else "sha256" if item.get("sha256") is not None and not _hex(item["sha256"])
                 else "sizeBytes" if item.get("sizeBytes") is not None and not _size(item["sizeBytes"]) else "")
        if field:
            raise ExportHeld("export_item_malformed", {"item_id": ident if isinstance(ident, str) else None,
                                                       "field": field})
        seen.add(ident)
        if item["kind"] == "symlink":
            target = item.get("linkTarget")
            data = target.encode("utf-8") if isinstance(target, str) else None
            if data is None or _hex(item.get("sha256")) != hashlib.sha256(data).hexdigest() or item.get("sizeBytes") != len(data):
                raise ExportHeld("export_symlink_mismatch")
    return manifest["items"]


def _same_export(prior: Dict[str, Any], raw: bytes) -> bool:
    """Has the engine's answer stayed what PRIOR was recorded against? Its exact manifest bytes where the record kept
    their digest (a held export, a malformed one included); else, for a spent operation, its export identity."""
    if prior.get("manifest_sha256"):
        return hashlib.sha256(raw).hexdigest() == prior["manifest_sha256"]
    try:
        manifest = json.loads(raw)
    except ValueError:
        return False
    return isinstance(manifest, dict) and manifest.get("exportId") == prior.get("export_id")


def _closure_names(manifest: Dict[str, Any]) -> set:
    """The run and job ids the closure names; its run list's shape is ``checked_items``'s to prove first."""
    return {str(row.get(key) or "") for row in manifest["closure"]["runs"] if isinstance(row, dict)
            for key in ("runId", "jobId")} - {""}


def _extension(path: str) -> str:
    """A suffix hint only: ``actor_source_reference`` keeps its ASCII letters and digits for the stored name."""
    return pathlib.PurePosixPath(path).suffix.lstrip(".") or "bin"


def _transfer(root: Any, task_id: str, engine: Any, thread_id: str, export_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """Stream one file item into the task's sources; published only when the engine's own ETag and
    Content-Length and the manifest's size/sha256 all describe the bytes that arrived."""
    from ouroboros.artifacts import stream_actor_source

    def fill(sink: Any) -> None:
        got = engine.stream_thread_export_item(thread_id, export_id, item["itemId"], sink)
        etag = str(got.get("etag") or "").strip()
        if etag.startswith("W/"):
            etag = ""  # a weak validator names no exact bytes
        if etag.strip('"') != f"sha256:{got.get('sha256')}" or got.get("content_length") not in (None, got.get("size")):
            raise ExportHeld("export_item_transfer_mismatch")

    return stream_actor_source(root, task_id, category=CATEGORY, source_id=f"thread-{thread_id}-{item['itemId']}",
                               extension=_extension(item["path"]), fill=fill,
                               expected={"size": item.get("sizeBytes"), "sha256": _hex(item.get("sha256")) or None})


def _reused(root: Any, task_id: str, reuse: Optional[Dict[str, Any]], export_id: str, item: Dict[str, Any]):
    """The ref an earlier receipt of this same export verified for this item, still intact on disk."""
    if not reuse or reuse.get("export_id") != export_id:
        return None  # another inventory: its items are other bytes until fetched
    row = next((row for row in reuse.get("items") or [] if row.get("item_id") == item["itemId"] and row.get("ref")), None)
    expected = {"size": item.get("sizeBytes"), "sha256": _hex(item.get("sha256")) or None}
    if row is None or any(expected[key] not in (None, row["ref"].get(key)) for key in expected):
        return None
    try:
        stream_retained_item(root, task_id, reuse, item["itemId"], None)
        return row["ref"]
    except (OSError, ValueError, LookupError, TypeError):  # not intact: the item is fetched again
        return None


def _addressed(root: Any, task_id: str, ref: Dict[str, Any]) -> Dict[str, Any]:
    """``ref`` as any task or fork reads it: its owning ``task_id`` and a ``read_file`` pointer at its absolute
    path under the canonical data root (an ``artifact_store`` path would name the reader's own task files). Its
    ``artifact_store`` identity stays for ``artifacts.read_actor_source_bytes``; dispatch keeps every read rule."""
    from ouroboros.artifacts import task_artifact_dir_path

    path = task_artifact_dir_path(pathlib.Path(root).resolve(strict=False), task_id).joinpath(
        *pathlib.PurePosixPath(ref["path"]).parts)
    return {**ref, "task_id": task_id, "read": {"tool": "read_file",
                                                 "arguments": {"root": "runtime_data", "path": str(path)}}}


def _flush_archive(root: Any, task_id: str, ref: Dict[str, Any]) -> None:
    """Flush the archive's entries through the configured data root before a release can be bound. That root
    already holds this task's record; the archive may create every directory below it, including its task folder.
    Each file (and compressed replacement) syncs its bytes before publication. This destructive caller cannot use
    the store's best-effort directory helper: a POSIX open/sync failure keeps the original via ``archive_unavailable``.
    Windows cannot sync directories: its existing file-only durability limit remains explicit, not a power-loss proof."""
    import os

    from ouroboros.artifacts import task_artifact_dir_path

    if os.name == "nt":
        return
    base = pathlib.Path(root)
    folder = task_artifact_dir_path(base, task_id).joinpath(*pathlib.PurePosixPath(ref["path"]).parts).parent
    for directory in (folder, *folder.parents[:len(folder.relative_to(base).parts)]):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def retain_export(root: Any, task_id: str, thread_id: str, engine: Any, *, host_runs: Iterable[str] = (),
                  reuse: Optional[Dict[str, Any]] = None, raw: Optional[bytes] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Read, copy and verify one thread's export: ``(receipt, receipt_ref)``. The exact manifest bytes are kept
    first, whatever they say (``raw``: the body this pass already read). ``receipt["verified"]`` only when the
    manifest is well formed, complete, retirable, of this store and thread, names every run in ``host_runs``,
    and every file item arrived exactly; ``holds`` (and ``detail`` for a malformed item) say why not. An item an
    earlier receipt (``reuse``) of the same export verified is read again from disk, never fetched again. Every
    file is synced as it is written and the folders after the receipt (``_flush_archive``); every ref is
    ``_addressed``. Raises ``ClaudexorUnavailable`` when the engine does not answer the export read."""
    from ouroboros.artifacts import store_actor_source_bytes

    store_id = str(getattr(engine, "store_identity", "") or "")
    raw = engine.thread_export(thread_id, expected_store_id=store_id) if raw is None else raw
    manifest_ref = _addressed(root, task_id, store_actor_source_bytes(
        root, task_id, category=CATEGORY, data=raw, extension="json", source_id=f"thread-{thread_id}-manifest",
        fsync=True))
    manifest: Any = None
    holds: List[str] = []
    detail: Optional[Dict[str, Any]] = None
    try:
        try:
            manifest = json.loads(raw)
        except ValueError as exc:
            raise ExportHeld("export_manifest_unreadable") from exc
        items = checked_items(manifest, thread_id=thread_id, store_id=store_id)
    except ExportHeld as held:  # kept as evidence with its named hold; nothing is transferred or released
        items, holds, detail = [], [str(held)], held.detail
    facts = manifest if isinstance(manifest, dict) else {}
    export_id = facts.get("exportId") if isinstance(facts.get("exportId"), str) else None
    if not holds and not (facts.get("coverage") == "complete" and facts.get("gaps") == [] and facts.get("retirable") is True):
        holds.append("export_partial")
    if not holds and not set(host_runs) <= _closure_names(facts):
        holds.append("closure_identity_mismatch")
    rows: List[Dict[str, Any]] = []
    for item in items:
        row = {"item_id": item["itemId"], "plane": item["plane"], "path": item["path"], "kind": item["kind"],
               "size": item.get("sizeBytes"), "sha256": _hex(item.get("sha256")) or None,
               "projection_of": item.get("projectionOf"), "link_target": item.get("linkTarget"), "ref": None}
        if item["kind"] == "file" and not holds:
            try:
                ref = _addressed(root, task_id, _reused(root, task_id, reuse, export_id, item)
                                 or _transfer(root, task_id, engine, thread_id, export_id, item))
                row.update(ref=ref, size=ref["size"], sha256=ref["sha256"])
            except Exception as exc:  # the engine's typed refusal, a short body, a full volume: nothing published
                row["error"] = f"{type(exc).__name__}: {getattr(exc, 'code', '') or exc}"[:300]
                holds.append("export_item_unverified")
        rows.append(row)
    verified = not holds
    digested = [row for row in rows if row["kind"] in _DIGESTED]  # credential stores and specials are in neither
    receipt = {
        "schema_version": 1, "kind": "review_thread_export", "task_id": task_id, "thread_id": thread_id,
        "store_id": store_id, "export_id": export_id,
        "closure_digest": facts.get("closureDigest") if isinstance(facts.get("closureDigest"), str) else None,
        "thread_state": facts.get("threadState"), "coverage": facts.get("coverage"),
        "retirable": facts.get("retirable") is True,
        "gaps": facts.get("gaps") if isinstance(facts.get("gaps"), list) else [],
        "disclosures": facts.get("disclosures") if isinstance(facts.get("disclosures"), list) else [],
        "manifest": manifest_ref, "manifest_sha256": manifest_ref["sha256"], "items": rows,
        "verified": verified, "holds": sorted(set(holds)), "detail": detail,
        "content_digest": export_digest((r["item_id"], r["sha256"], r["size"]) for r in digested) if verified else None,
        "lane_digest": export_digest((r["item_id"], r["sha256"], r["size"]) for r in digested
                                     if r["plane"] == "lane") if verified else None,
    }
    data = json.dumps(receipt, ensure_ascii=False, sort_keys=True).encode("utf-8")
    receipt_ref = _addressed(root, task_id, store_actor_source_bytes(
        root, task_id, category=CATEGORY, data=data, extension="json", source_id=f"thread-{thread_id}-receipt",
        fsync=True))
    _flush_archive(root, task_id, receipt_ref)
    return receipt, receipt_ref


def read_thread_export(root: Any, task_id: str, receipt_ref: Dict[str, Any]) -> Dict[str, Any]:
    """A retained receipt, its exact bytes verified; nothing of the engine is read."""
    from ouroboros.artifacts import read_actor_source_bytes

    return json.loads(read_actor_source_bytes(root, task_id, receipt_ref))


def stream_retained_item(root: Any, task_id: str, receipt: Dict[str, Any], item_id: str, sink: Any) -> Dict[str, Any]:
    """Stream one retained item's exact bytes into ``sink`` (``None``: only verify them, as a later pass reusing the
    item does) from the archive alone, hashed as they pass and checked against the receipt
    (``artifacts.stream_artifact_file``). Items live only at their canonical name in the task's
    ``source_handles/delegated_activity``: the evidence packer packs context checkpoints and call blobs, never this
    category, so a missing item is reported missing (``FileNotFoundError``), never read whole from elsewhere."""
    from ouroboros.artifacts import stream_artifact_file, task_artifact_dir_path

    row = next((row for row in receipt.get("items") or [] if row.get("item_id") == item_id and row.get("ref")), None)
    if row is None:
        raise LookupError(f"item {item_id} was not retained")
    ref = row["ref"]
    parts = pathlib.PurePosixPath(str(ref.get("path") or "")).parts
    if len(parts) != 3 or parts[:2] != ("source_handles", CATEGORY) or ".." in parts:
        raise ValueError("retained item ref leaves its source category")
    return stream_artifact_file(task_artifact_dir_path(root, task_id).joinpath(*parts), sink,
                                expected={"size": int(ref["size"]), "sha256": str(ref["sha256"])})


# ---- task-side facts ---------------------------------------------------------------------------------


def review_threads(document: Dict[str, Any]) -> Dict[str, List[Tuple[str, str]]]:
    """Thread id -> the ``(review record id, seat id)`` pairs whose plan participant named it."""
    record = document.get("plan_record") if isinstance(document.get("plan_record"), dict) else {}
    threads: Dict[str, List[Tuple[str, str]]] = {}
    for ask in record.get("asks") or []:
        for participant in (ask.get("participants") or []) if isinstance(ask, dict) else ():
            thread = str(participant.get("review_thread_id") or "") if isinstance(participant, dict) else ""
            if thread:
                threads.setdefault(thread, []).append((str(ask.get("record_id") or ""), str(participant.get("seat_id") or "")))
    return threads


def seat_threads(root: Any, document: Dict[str, Any], *, settled_only: bool = False
                 ) -> Tuple[Dict[str, List[Tuple[str, str]]], bool]:
    """``review_threads`` and every thread a plan seat started in, and whether each start could be read. A session
    seat whose reply named no thread is looked up by its own operations (one keyed custody read each, its START holding
    the thread since before the POST), never by a scan. This names the task's material for this module and
    ``owner_keep`` only: ``review_thread_id`` stays the reply's word that the thread answered, and only that word
    continues into a next ask (``plan_delivery.last_answer_row``). An unreadable start is ``False``, never an absent
    thread. ``settled_only`` leaves out a seat still pending: no pass can release its thread, and its settlement's
    terminal write names the task again, so a seat whose custody was lost keeps nobody waiting."""
    from ouroboros.delegate_invocation import LEGACY_GAP, operation_record

    threads, complete = review_threads(document), True
    record = document.get("plan_record") if isinstance(document.get("plan_record"), dict) else {}
    for ask in record.get("asks") or []:
        for participant in (ask.get("participants") or []) if isinstance(ask, dict) else ():
            if (not isinstance(participant, dict) or participant.get("review_thread_id")
                    or participant.get("route") != "agent_session"
                    or (settled_only and participant.get("status") == "pending")):
                continue
            seat = str(participant.get("seat_id") or "")
            for operation in (participant.get("operation_id"), *(participant.get("superseded_operation_ids") or [])):
                started = operation_record(root, str(operation), seat) if operation else None
                if started is None:
                    continue  # no start was recorded for it: that operation ran in no thread
                request = started.get("request")
                if started.get("state") == LEGACY_GAP or not isinstance(request, dict):
                    complete = False
                elif request.get("_thread_id"):
                    seats = threads.setdefault(str(request["_thread_id"]), [])
                    if (str(ask.get("record_id") or ""), seat) not in seats:
                        seats.append((str(ask.get("record_id") or ""), seat))
    return threads, complete


def _export_owed(record: Any) -> bool:
    """False once nothing here exports the thread again: a release in flight or accepted, or a store's word of purge."""
    record = record if isinstance(record, dict) else {}
    engine = record.get("engine") if isinstance(record.get("engine"), dict) else {}
    return record.get("state") not in ("bound", "accepted") and engine.get("thread_state") != "purged"


def _host_runs(root: Any, runs: Dict[str, Any], task_id: str) -> Dict[str, List[Any]]:
    """Thread id -> this task's custody runs in it: requests naming the thread, and the runs continuing them."""
    from ouroboros.delegate_custody import invocation_record

    mine = {run_id: entry for run_id, entry in runs.items() if entry.task_id == task_id}
    named: Dict[str, str] = {}
    for run_id, entry in mine.items():
        request = (invocation_record(root, entry.invocation_id) or {}).get("request") if entry.invocation_id else None
        thread = str(request.get("_thread_id") or "") if isinstance(request, dict) else ""
        if thread:
            named[run_id] = thread
    grew = True
    while grew:  # a standalone continueFrom of a thread turn belongs to the same closure
        grew = False
        for run_id, entry in mine.items():
            if run_id not in named and entry.continuation_of in named:
                named[run_id], grew = named[entry.continuation_of], True
    threads: Dict[str, List[Any]] = {}
    for run_id, thread in named.items():
        threads.setdefault(thread, []).append(mine[run_id])
    return threads


def _after(started: Any, since: str) -> bool:
    try:
        return datetime.fromisoformat(str(started)) >= datetime.fromisoformat(since)
    except (TypeError, ValueError):
        return False  # an unknown start is not after activation


def retry_line(root: Any, task_id: str) -> List[str]:
    """The task, then the timeout-retried predecessors its reciprocal retry links name (``source_retention``'s own
    walk): a predecessor's plan record keeps its threads while only its retry is ever written terminal."""
    from ouroboros.source_retention import _source_locations

    try:
        return list(dict.fromkeys([task_id, *(owner for _root, owner in _source_locations(root, task_id))]))
    except Exception:
        return [task_id]


def _queue_census(root: Any) -> Optional[set]:
    """Task ids the queue snapshot holds running or pending; ``None`` when it proves nothing (missing, stale)."""
    from ouroboros.task_status import _load_queue_snapshot, _snapshot_is_stale

    try:
        snapshot = _load_queue_snapshot(pathlib.Path(root))
        if snapshot.get("_snapshot_missing") or snapshot.get("_snapshot_invalid") or _snapshot_is_stale(snapshot):
            return None
        return {str(row.get("id") or (row.get("task") or {}).get("id") or "") for lane in ("running", "pending")
                for row in snapshot.get(lane) or [] if isinstance(row, dict)} - {""}
    except Exception:
        return None


def _reviews_open(document: Dict[str, Any]) -> bool:
    """One member's own open review work: an operation pointer, or a plan participant pending or unresolved."""
    from ouroboros.obligations import review_collection_pending
    from ouroboros.plan_record import collection_pending, unknown_outcome_asks

    record = document.get("plan_record")
    return bool(review_collection_pending(document) or collection_pending(record)
                or (isinstance(record, dict) and unknown_outcome_asks(record)))


def _members_hold(root: Any, tree: str, task_id: str, document: Dict[str, Any]) -> str:
    """The whole tree, from its durable rows and the published queue (the cascade's own membership walk,
    strict): every member ended, a timeout-retried one through the retry its effective result follows, and
    none with open review work. Open review work is each row's derived collection facts, read for the whole
    tree at once (``task_store.collection_flags``); only a row still underived is answered by its body. A row
    that cannot be placed or read holds."""
    from ouroboros.task_status import SETTLED_STATUSES, load_effective_task_result
    from ouroboros.task_store import collection_flags, read_document
    from supervisor.terminal_delivery import _cascade_descendant_rows

    gaps: List[str] = []
    try:
        statuses = _cascade_descendant_rows(pathlib.Path(root), tree, strict=True, gaps=gaps)
        members = list(dict.fromkeys((tree, task_id, *statuses)))
        flags = collection_flags(root, members)
    except Exception:
        return "tree_unreadable"
    if gaps:
        return "tree_unresolved"
    for member in members:
        if member not in flags:  # no durable row: a queue-only member is not known to have ended
            return "tree_unresolved"
        if member in statuses and statuses[member] not in SETTLED_STATUSES and str(
                (load_effective_task_result(root, member, materialize_artifacts=False) or {}).get("status") or ""
        ) not in SETTLED_STATUSES:
            return "tree_unsettled"
        if any(flags[member]):
            return "review_open"
        if None in flags[member]:  # underived: the body answers, as the derivation would
            own = document if member == task_id else read_document(root, member)
            if not isinstance(own, dict) or not own:
                return "tree_unresolved"
            if _reviews_open(own):
                return "review_open"
    return ""


def _tree_open(root: Any, tree: str, task_id: str, document: Dict[str, Any], live: Callable[[str], Any]) -> str:
    """Why one tree's work is not over (``""`` when it is): the task and its root proven over, no live queue row
    in the tree, every member ended with no open review work (``_members_hold``)."""
    from ouroboros.subagent_worktrees import _task_proven_over
    from ouroboros.task_status import task_subtree_has_live_queue_rows

    if any(not _task_proven_over(member, root, live) for member in dict.fromkeys((task_id, tree))):
        return "task_not_proven_over"
    if task_subtree_has_live_queue_rows(root, tree):
        return "tree_live"
    return _members_hold(root, tree, task_id, document)


def _custody_hold(root: Any, tree: str, task_id: str = "") -> Tuple[str, Dict[str, Any]]:
    """``("", task_id's custody runs in the tree)`` once no invocation of the tree is pending and none of its runs
    is unsettled or settled without a terminal state, else why not; one read (``delegate_custody_current.tree_closure``)."""
    from ouroboros import delegate_custody_current as current
    from ouroboros.delegate_custody import TERMINAL_STATES

    try:
        if current.unreadable(root):
            return "custody_unreadable", {}
        snapshot = current.tree_closure(root, tree, task_id)
    except Exception:
        return "custody_unreadable", {}
    if snapshot["pending"] or any(not entry.settled or entry.terminal_state not in TERMINAL_STATES
                                  for entry in snapshot["unfinished"].values()):
        return "custody_open", {}
    return "", snapshot["runs"]


def _continued_roots(root: Any, tree: str) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
    """Each root the owner's Continue admitted after ``tree``, in chain order, with its document: the claim each
    root records (only a root can be continued, ``owner_continue.continuation_eligibility``), each verified by
    ``owner_continue.recorded_continuation``. ``None`` when one cannot be read or verified."""
    from ouroboros.owner_continue import recorded_continuation
    from ouroboros.task_store import read_document

    roots: List[Tuple[str, Dict[str, Any]]] = []
    current = tree
    try:
        document = read_document(root, tree) or {}
        while document.get("continued_by"):
            claim = document["continued_by"]
            successor = str(claim.get("successor_task_id") or "") if isinstance(claim, dict) else ""
            if (not successor or successor in (tree, *(name for name, _ in roots))
                    or not recorded_continuation(root, current, successor)):
                return None
            document = read_document(root, successor) or {}
            roots.append((successor, document))
            current = successor
    except Exception:
        return None
    return roots


def tree_hold(root: Any, task_id: str, document: Dict[str, Any], *, live: Callable[[str], Any],
              census: Optional[set], every_offer: bool = False) -> Tuple[str, Dict[str, Any]]:
    """Why the task's threads cannot go yet (``""`` when nothing holds them) and its own custody runs in its tree.
    The whole work holds them: the task's tree, the line while it is continuable, then the tree of each root
    the owner's Continue admitted after it, whose own members, reviews and custody must end too (that hold
    is named ``continued_<why>``). Every answer that cannot be read holds. ``every_offer`` (a managed document's
    release, ``delegate_documents``) also holds for an author's stop and an offer not yet decidable
    (``delegate_continuation.still_continuable``). Never called inside the keep gate (the continuation reader may
    take the supervisor's queue lock in the server process)."""
    from ouroboros.delegate_continuation import still_continuable

    tree = str(document.get("root_task_id") or task_id)
    why = _tree_open(root, tree, task_id, document, live)
    if why:
        return why, {}
    if census is None:
        return "census_unknown", {}
    why, runs = _custody_hold(root, tree, task_id)
    if why:
        return why, {}
    line = SimpleNamespace(review_owned=False, run_id=f"review-threads:{task_id}", task_id=task_id,
                           root_task_id=tree, invocation_id="")
    if still_continuable(root, line, census, every_offer=every_offer):
        return "continuable", {}
    successors = _continued_roots(root, tree)
    if successors is None:
        return "continuation_unverified", {}
    for successor, own in successors:  # an ended, uncontinuable successor may still have its tree's work open
        why = _tree_open(root, successor, successor, own, live) or _custody_hold(root, successor)[0]
        if why:
            return f"continued_{why}", {}
    return "", runs


# ---- the release operation --------------------------------------------------------------------------


def _patch(root: Any, task_id: str, thread_id: str, change: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]):
    """Replace this thread's release record by ``change(record)`` (``None``: write nothing); the record after."""
    from ouroboros.task_store import patch_document

    def mutate(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        releases = dict(current.get(RELEASES) or {}) if isinstance(current.get(RELEASES), dict) else {}
        record = releases.get(thread_id) if isinstance(releases.get(thread_id), dict) else {}
        changed = change(dict(record))
        if changed is None:
            return None
        return {**current, RELEASES: {**releases, thread_id: changed}}

    document = patch_document(root, task_id, mutate)
    return (document.get(RELEASES) or {}).get(thread_id) or {}


def _bind(root: Any, task_id: str, receipt: Dict, receipt_ref: Dict, *, historical: bool = False) -> Tuple[str, Dict]:
    """Durably bind one release under the keep gate (``historical``: the one historical walk's own): ``("", record)``, or why not."""
    from ouroboros.owner_keep import gate
    from ouroboros.utils import utc_now_iso

    operation = uuid.uuid4().hex
    with gate(root, task_id, timeout_sec=0) as why:
        if why:
            return why, {}
        record = _patch(root, task_id, receipt["thread_id"], lambda prior: None if prior.get("state") in (
            "bound", "accepted") else {
            "state": "bound", "operation_id": operation, "thread_id": receipt["thread_id"],
            "store_id": receipt["store_id"], "export_id": receipt["export_id"], "bound_at": utc_now_iso(),
            "closure_digest": receipt["closure_digest"], "content_digest": receipt["content_digest"],
            "lane_digest": receipt["lane_digest"], "receipt": receipt_ref, **({"historical": True} if historical else {})})
    return ("", record) if record.get("operation_id") == operation else ("release_exists", record)


def _settle(root: Any, task_id: str, record: Dict[str, Any], outcome: Dict[str, Any]) -> bool:
    """Write the engine's settled outcome of exactly this bound operation, under the task custody lock."""
    from ouroboros.task_custody import task_custody_lock
    from ouroboros.utils import utc_now_iso

    operation = record["operation_id"]
    with task_custody_lock(root, task_id, timeout_sec=10.0) as locked:
        if not locked:
            return False
        _patch(root, task_id, record["thread_id"], lambda prior: {**prior, **outcome, "settled_at": utc_now_iso()}
               if prior.get("operation_id") == operation and prior.get("state") == "bound" else None)
    return True


def settled_outcome(answer: Any, operation_id: str) -> Optional[Dict[str, Any]]:
    """A ``ThreadRetirement`` body's bound outcome of OUR operation; ``None`` settles nothing."""
    operation = answer.get("operation") if isinstance(answer, dict) else None
    if not (isinstance(operation, dict) and operation.get("operationId") == operation_id
            and operation.get("outcome") in SETTLED):
        return None
    cleanup = answer.get("cleanup") if isinstance(answer.get("cleanup"), dict) else {}
    return {"state": operation["outcome"], "code": str(operation.get("code") or ""),
            "message": str(operation.get("message") or "")[:500],
            "engine": {"status": str(answer.get("status") or ""), "thread_state": str(answer.get("threadState") or ""),
                       "cleanup_reason": cleanup.get("reason")}}


def problem_outcome(exc: BaseException) -> Optional[Dict[str, Any]]:
    """A 409/410 the engine bound to the operation (``context.settled``); anything else proves nothing."""
    problem = getattr(exc, "problem", None)
    context = problem.get("context") if isinstance(problem, dict) else None
    if int(getattr(exc, "status_code", 0) or 0) not in (409, 410) or not (
            isinstance(context, dict) and context.get("settled") is True):
        return None
    code = str(getattr(exc, "code", "") or "")
    return {"state": "abandoned" if code == "thread_retirement_abandoned" else "refused", "code": code,
            "message": str(problem.get("message") or "")[:500],
            "engine": {"thread_state": "purged"} if code == "thread_purged" else {}}


def _ask(call: Callable[[], Any], operation_id: str) -> Optional[Dict[str, Any]]:
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    try:
        return settled_outcome(call(), operation_id)
    except ClaudexorUnavailable as exc:
        return problem_outcome(exc)


def drive_release(root: Any, task_id: str, record: Dict[str, Any], engine: Any, *, may_send: bool) -> str:
    """Send or reconcile one bound operation, outside any lock: its settled state, or ``"bound"`` (unknown).
    The operation's own outcome is read first. Unsettled, a standing Keep abandons it (the engine answers
    ``accepted`` if it already was); otherwise the same body is POSTed again, only while ``may_send``."""
    from ouroboros.owner_keep import is_kept

    thread, operation = record["thread_id"], record["operation_id"]
    if str(getattr(engine, "store_identity", "") or "") != record.get("store_id"):
        return "bound"  # another store cannot answer for this operation
    outcome = _ask(lambda: engine.thread_retirement(thread, operation_id=operation,
                                                    expected_store_id=record["store_id"]), operation)
    if outcome is None and is_kept(root, task_id):
        outcome = _ask(lambda: engine.abandon_thread_retirement(thread, operation), operation)
    elif outcome is None and may_send:
        body = {"operationId": operation, "exportId": record["export_id"], "closureDigest": record["closure_digest"],
                "contentDigest": record["content_digest"], "laneDigest": record["lane_digest"]}
        outcome = _ask(lambda: engine.retire_thread(thread, body), operation)
    if outcome is None or not _settle(root, task_id, record, outcome):
        return "bound"
    return outcome["state"]


def _link_ledger(root: Any, task_id: str, seats: List[Tuple[str, str]], receipt: Dict[str, Any],
                 receipt_ref: Dict[str, Any]) -> bool:
    """Name the retained receipt in each review-ledger row that named the thread; False when one could not be written."""
    from ouroboros import review_ledger

    entry = {"role": ROLE, "part": receipt["export_id"], "chars": int(receipt_ref["size"]), "status": "retained",
             "ref": receipt_ref, "task_id": task_id}
    by_record: Dict[str, set] = {}
    for record_id, seat_id in seats:
        if record_id and seat_id:
            by_record.setdefault(record_id, set()).add(seat_id)
    for record_id, seat_ids in sorted(by_record.items()):
        def add(payload: Dict[str, Any], seat_ids: set = seat_ids) -> Dict[str, Any]:
            for row in payload.get("rows") or []:
                if isinstance(row, dict) and row.get("seat_id") in seat_ids:
                    named = {**entry, "seat_id": row["seat_id"]}
                    history = row.setdefault("historical_source_refs", [])
                    if named not in history:
                        history.append(named)
            return payload

        try:
            current = review_ledger.load_record(root, record_id)
            if current is None:
                continue  # the plan ask's record lives elsewhere or was never written: the task document names it
            missing = [row for row in current.get("rows") or [] if isinstance(row, dict) and row.get("seat_id") in seat_ids
                       and {**entry, "seat_id": row["seat_id"]} not in (row.get("historical_source_refs") or [])]
            if missing:
                review_ledger.revise_record(root, record_id, add)
        except (OSError, ValueError):
            log.warning("review ledger %s does not name thread export %s yet", record_id, receipt["export_id"],
                        exc_info=True)
            return False
    return True


# ---- the pass ---------------------------------------------------------------------------------------


class _Pass:
    """One pass: engine opened at most once, only when a candidate needs it."""

    def __init__(self, root: Any, activated_at: str, *, engine: Any, live: Callable[[str], Any],
                 census: Callable[[], Optional[set]], stop: Callable[[], bool]) -> None:
        self.root, self.activated_at, self._engine, self.live, self.census = root, activated_at, engine, live, census
        self._opened, self._owned, self._supported = engine is not None, False, None
        from ouroboros.execution_domain import EngineClients
        self.work = EngineClients(self.engine)  # each engine-held object through its own engine (``execution_domain``)
        self._work_supported: Dict[Any, str] = {}
        self.stop, self.stopped, self.walk_states = stop, False, {}  # walk_states: what an ask of the walk settled
        self.report: Dict[str, Any] = {"tasks": 0, "retained": 0, "accepted": 0, "refused": 0, "abandoned": 0,
                                       "unknown": 0, "holds": {}, "errors": [], "observed": []}

    def hold(self, reason: str) -> None:
        self.report["holds"][reason] = self.report["holds"].get(reason, 0) + 1

    def engine(self) -> Any:
        if not self._opened:
            self._opened = True
            try:
                from ouroboros.claudexor_daemon import read_owned_gateway

                self._engine, self._owned = read_owned_gateway(timeout_sec=10.0), True
            except Exception as exc:  # no engine now: everything stays, the task waits for a later pass
                log.info("Reviewer thread retention found no engine: %s", exc)
                self._engine = None
        return self._engine

    def usable(self, engine: Any = None) -> str:
        """Why this pass cannot ask an engine anything about thread retirement, or ``""``: the owned engine (asked once)
        by default, else ``engine``, the work engine holding an object (once per engine; ``execution_domain``)."""
        if engine is not None and engine is not self._engine:
            from ouroboros.execution_domain import engine_key

            key = engine_key(engine)
            if key not in self._work_supported or key is None:  # an unreachable stand-in is asked each time
                try:
                    verdict = "" if engine.thread_retirement_supported() else "engine_export_unsupported"
                except Exception:
                    verdict = "engine_unavailable"
                self._work_supported[key] = verdict
            return self._work_supported[key]
        if self._supported is None:
            engine = self.engine()
            try:
                self._supported = "" if engine is not None and engine.thread_retirement_supported() else (
                    "engine_unavailable" if engine is None else "engine_export_unsupported")
            except Exception:  # an unread catalog proves no route either way
                self._supported = "engine_unavailable"
        return self._supported

    def close(self) -> None:
        if self._owned and self._engine is not None:  # only the attachment this pass opened
            try:
                self._engine.close()
            except Exception:
                log.debug("engine close failed", exc_info=True)
        self.work.drop()

    def task(self, task_id: str, historical: bool = False) -> str:
        """``WAIT`` keeps the task for a later pass; ``DONE`` lets it go until its next terminal write."""
        from ouroboros.owner_keep import is_kept
        from ouroboros.task_store import read_document

        document = read_document(self.root, task_id) or {}
        threads, complete = seat_threads(self.root, document, settled_only=True)
        if not complete:
            self.hold("thread_start_unreadable")  # unnamed, so nothing can release it; ``owner_keep`` answers unknown
        releases = document.get(RELEASES) if isinstance(document.get(RELEASES), dict) else {}
        bound = [record for record in releases.values() if isinstance(record, dict) and record.get("state") == "bound"]
        fresh = [thread for thread in sorted(threads) if _export_owed(releases.get(thread))]
        if not (bound or fresh):
            return DONE
        from ouroboros.execution_domain import current

        if current() is None and self.usable():  # unmanaged: the owned engine answers for every release and thread
            self.hold(self.usable())  # nothing can be read, sent or reconciled now: the task is asked again later
            return WAIT
        kept = is_kept(self.root, task_id)
        why, runs = ("owner_keep", {}) if kept else tree_hold(self.root, task_id, document, live=self.live,
                                                               census=self.census())
        verdict, done = DONE, 0
        for record in bound:  # an unknown outcome is re-sent only while nothing holds the tree
            if self.between(done):
                return WAIT
            verdict, done = self.reconcile(task_id, record, may_send=not why) or verdict, done + 1
        if not fresh:
            return verdict
        if why:
            self.hold(why)
            return verdict if kept else WAIT
        host = _host_runs(self.root, runs, task_id)
        for thread_id in fresh:
            if self.between(done):
                return WAIT
            verdict, done = self.thread(task_id, thread_id, threads[thread_id], host.get(thread_id) or [],
                                        releases.get(thread_id) or {}, historical) or verdict, done + 1
        return verdict

    def documents(self, task_id: str) -> bool:
        """Whether the task's managed document imports still owe a release after this pass asked
        (``delegate_documents.release_owed_imports``: its own whole-work hold, this pass's engine, stop and report)."""
        from ouroboros.delegate_documents import release_owed_imports

        return release_owed_imports(self.root, task_id, engine=self.work.for_store, live=self.live, census=self.census,
                                    hold=self.hold, stop=self.stop, observed=self.report["observed"].append)

    def between(self, done: int) -> bool:
        """Whether the caller's stop ends the pass before this task's next operation (a bound release to reconcile,
        then a fresh thread): only after one (``done``), which kept its own; the task waits for the rest."""
        if done and self.stop():
            self.stopped = True
            return True
        return False

    def reconcile(self, task_id: str, record: Dict[str, Any], *, may_send: bool) -> Optional[str]:
        engine = self.work.for_store(record.get("store_id"))  # the engine whose store holds the release
        if self.usable(engine):  # that engine cannot be asked now: the release stays bound
            self.hold(self.usable(engine))
            return WAIT
        state = drive_release(self.root, task_id, record, engine, may_send=may_send)
        if record.get("historical"):
            self.walk_states[record["operation_id"]] = state  # the walk's count must not need a later read of the task
        self.report["unknown" if state == "bound" else state] += 1
        return WAIT if state == "bound" else None

    def held(self, task_id: str, thread_id: str, receipt: Dict[str, Any], receipt_ref: Dict[str, Any],
             holds: List[str]) -> None:
        """Record why a retained export is not released, with its receipt (``owner_keep`` reads it as retained)."""
        from ouroboros.utils import utc_now_iso

        for code in holds:
            self.hold(code)
        _patch(self.root, task_id, thread_id, lambda record: None if record.get("state") in ("bound", "accepted") or (
            record.get("state") == "held" and record.get("export_id") == receipt["export_id"]
            and record.get("holds") == holds and record.get("receipt") == receipt_ref) else {
            "state": "held", "thread_id": thread_id, "export_id": receipt["export_id"], "holds": holds,
            "detail": receipt.get("detail"), "manifest_sha256": receipt.get("manifest_sha256"),
            "gaps": [str((gap or {}).get("code") or "") for gap in receipt["gaps"] if isinstance(gap, dict)],
            "receipt": receipt_ref, "at": utc_now_iso()})

    def observed(self, task_id: str, thread_id: str, code: str, store_id: str, prior: Dict[str, Any]) -> Optional[str]:
        """Save the store's typed answer before any release (``OBSERVED``), only when it changed: no operation or
        acceptance is invented and what an earlier pass kept stays named. An answer no store can be named for proves
        nothing, so nothing is saved and a later pass asks again."""
        from ouroboros.utils import utc_now_iso

        if not store_id:
            self.hold("store_unknown")
            return WAIT
        self.hold(code)
        state = OBSERVED[code]
        if (prior.get("state"), prior.get("store_id")) != (state, store_id):
            _patch(self.root, task_id, thread_id, lambda record: None if record.get("state") in ("bound", "accepted")
                   else {**record, "state": state, "thread_id": thread_id, "store_id": store_id, "code": code,
                         "engine": {"thread_state": "purged"} if state == "purged" else {}, "at": utc_now_iso()})
            self.report["observed"].append({"task_id": task_id, "thread_id": thread_id, "state": state})
        return None

    def thread(self, task_id: str, thread_id: str, seats: List[Tuple[str, str]], runs: List[Any],
               prior: Dict[str, Any], historical: bool = False) -> Optional[str]:
        from ouroboros.gateways.claudexor import ClaudexorUnavailable
        from ouroboros.task_store import read_document

        if not runs or not (historical or all(_after(entry.started_ts, self.activated_at) for entry in runs)):
            self.hold("predates_activation" if runs else "no_host_runs")  # only the historical walk's to release
            return None
        engine = self.work.reach(runs[0])  # the thread's own engine, which its host runs were started on
        if self.usable(engine):  # that engine cannot be asked now: the thread waits for a later pass
            self.hold(self.usable(engine))
            return WAIT
        store_id = str(getattr(engine, "store_identity", "") or "")
        # Only the engine's inventory answers an inventory hold: the same export is not tried again as it was. A hold
        # of this host (an unverified item, the tree at the release boundary) reuses the verified items it kept.
        spent = prior.get("state") == "refused" or prior.get("state") == "held" and bool(
            set(prior.get("holds") or []) & set(INVENTORY_HOLDS))
        try:
            raw = engine.thread_export(thread_id, expected_store_id=store_id) if spent else None
            if spent and _same_export(prior, raw):
                self.hold(f"{prior['state']}_unchanged")
                return None
            reuse = read_thread_export(self.root, task_id, prior["receipt"]) if prior.get("state") == "held" and not spent \
                and prior.get("receipt") else None
            receipt, receipt_ref = retain_export(self.root, task_id, thread_id, engine, reuse=reuse, raw=raw,
                                                 host_runs=[entry.run_id for entry in runs if entry.run_id])
        except (ClaudexorUnavailable, OSError, ValueError) as exc:
            code = str(getattr(exc, "code", "") or "")
            if code in OBSERVED:  # nothing to export from this store: its typed answer
                return self.observed(task_id, thread_id, code, store_id, prior)
            engine_side = isinstance(exc, ClaudexorUnavailable)  # else this archive could not be written or read
            self.hold("export_unavailable" if engine_side else "archive_unavailable")
            log.log(logging.INFO if engine_side else logging.WARNING, "Thread %s export not retained: %s", thread_id,
                    exc, exc_info=not engine_side)
            return WAIT
        if not receipt["verified"]:
            self.held(task_id, thread_id, receipt, receipt_ref, receipt["holds"])
            return None if set(receipt["holds"]) & set(INVENTORY_HOLDS) else WAIT
        self.report["retained"] += 1
        if not _link_ledger(self.root, task_id, seats, receipt, receipt_ref):
            self.hold("ledger_unwritable")
            return WAIT
        # The release boundary: the bulk copy took time, so the tree is read again right before binding, outside the
        # keep gate (its liveness and continuation readers must never run inside it); the engine re-checks the closure.
        why, _runs = tree_hold(self.root, task_id, read_document(self.root, task_id) or {}, live=self.live,
                               census=self.census())
        if why:
            self.held(task_id, thread_id, receipt, receipt_ref, [why])
            return WAIT
        why, record = _bind(self.root, task_id, receipt, receipt_ref, historical=historical)
        if why:
            self.hold(why)
            return WAIT if why == "busy" else None
        return self.reconcile(task_id, record, may_send=True)


def owed(row: Dict[str, Any]) -> bool:
    """Whether a pass left tasks waiting, from the job ROW alone. A task written terminal since is no such debt: a row
    does not name a review thread. A managed-import terminal writer separately schedules retirement."""
    state = row.get(JOB_FIELD) if isinstance(row.get(JOB_FIELD), dict) else {}
    return bool(state.get("activated_at") and state.get("waiting"))


def retire_finished_review_threads(root: Any, *, engine: Any = None, live: Optional[Callable[[str], Any]] = None,
                                   census: Any = "queue", should_stop: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """One pass (module docstring): ``runtime_retirement``'s step. ``engine``/``live``/``census`` replace the
    attached engine, the queue snapshot's liveness probe and its running/pending census (tests, a caller
    that already holds them). ``should_stop`` is asked between tasks and between a task's operations; a pass it ends
    keeps the rest waiting (reported ``stopped``), which the maintenance child's next run takes up (``owed``). A pass
    that changed neither the cursor, the waiting tasks, the engine's hold nor the walk's record writes nothing."""
    from ouroboros import historical_retirement, maintenance_jobs, store
    from ouroboros.delegate_result_taken import empty_walk, tell_engine, walk
    from ouroboros.maintenance_retirement import JOB
    from ouroboros.task_status import task_has_live_queue_ownership
    from ouroboros.task_store import terminal_writes_since
    from ouroboros.utils import utc_now_iso

    stop = should_stop or (lambda: False)
    with store.open_store(pathlib.Path(root), write=False) as txn:
        counter = int(txn.meta(store.TERMINAL_WRITES) or 0)
        row = txn.meta(maintenance_jobs.PREFIX + JOB) or {}
    state = row.get(JOB_FIELD) if isinstance(row.get(JOB_FIELD), dict) else {}
    if not state.get("activated_at"):  # the first pass: what was written before it, or before a floor, is never read
        floor = state.get("floor")  # a managed document import owed since (``maintenance_retirement.release_opened``)
        pin = min(counter, floor) if type(floor) is int else counter
        state = {"activated_at": utc_now_iso(), "counter": pin, "waiting": state.get("waiting") or []}
        over = {**empty_walk(root), **historical_retirement.empty_walk(root)}  # nothing delegated: both walks are over
        maintenance_jobs.record(root, JOB, **{JOB_FIELD: state}, **over)
        if pin == counter and not state["waiting"] and over:
            return {"activated": True}
        # Process the preserved import interval, or walk, now: pinning a cursor is not scheduling a second pass.
    written = terminal_writes_since(root, int(state.get("counter") or 0))
    tasks = list(dict.fromkeys([*(state.get("waiting") or []), *(owner for task in written for owner in retry_line(root, task))]))
    run = _Pass(root, str(state["activated_at"]), engine=engine,
                live=live or (lambda task_id: task_has_live_queue_ownership(pathlib.Path(root), task_id)),
                census=(lambda: _queue_census(root)) if census == "queue" else (lambda: census), stop=stop)
    waiting: List[str] = []
    try:
        for index, task_id in enumerate(tasks):
            if stop():
                waiting.extend(tasks[index:])
                run.stopped = True
                break
            run.report["tasks"] += 1
            wait = False  # the task waits while its threads, managed documents or engine decisions do, each asked on its own
            for step in (lambda: run.task(task_id) == WAIT, lambda: run.documents(task_id),
                         lambda: tell_engine(run, task_id) == WAIT):
                try:
                    wait = run.stopped or step() or wait
                except Exception as exc:  # one unreadable task never stops the pass; it is asked again
                    wait = True
                    run.report["errors"].append(f"{task_id}: {type(exc).__name__}: {exc}"[:300])
                    log.warning("Runtime retirement failed for task %s", task_id, exc_info=True)
            if wait:
                waiting.append(task_id)
        walked = {**walk(run, tasks, row, waiting), **historical_retirement.walk(run, row)}  # the one-time walks (each module)
    finally:
        run.close()
    after = {"activated_at": state["activated_at"], "counter": counter, "waiting": sorted(set(waiting))}
    # Why the engine could not be asked, as last answered: a pass that never asked it learned nothing new.
    hold = str(state.get("engine_hold") or "") if run._supported is None else run._supported
    if hold:
        after["engine_hold"] = hold
    if hold != str(state.get("engine_hold") or ""):
        run.report["observed"].append({"engine_hold": hold})
    if after != state or walked:  # an unchanged pass commits nothing: the child's next look over the same facts rests
        maintenance_jobs.record(root, JOB, **{JOB_FIELD: after}, **walked)  # the walk's record lands with ``waiting``
    first = {} if isinstance(row.get(JOB_FIELD), dict) and row[JOB_FIELD].get("activated_at") else {"activated": True}
    return {**run.report, "waiting": len(after["waiting"]), **first, **({"stopped": True} if run.stopped else {})}
