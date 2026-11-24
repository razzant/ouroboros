"""First-use and saved-history evidence read over the dialogue store (#3294).

The store module owns writing and ordinary reads; this module owns the three evidence
reads that guard a missing Project registry: ``saved_history_snapshot`` (one whole-request
read of the existing cutover or a confirmed empty source, for the read-only room archive),
``could_lack_project_history`` (consult retained Project facts without causing their import,
distinguishing a pending input from a loss) and the strict listing of chain files still to
import, including the importer's planned-archive rule (``dialogue_import._plan``). Their
``ProjectHistoryImportPending`` distinguishes pending evidence from loss for registry recovery.
The web-ingress proof and its source-keyed cache also stay with this reader. Nothing here
writes, starts a cutover or reconstructs a registry.
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

from ouroboros import dialogue_store as ds, store, store_empty, utils
from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
from ouroboros.installation_stop import confirm_absent
from ouroboros.store import StoreError, StoreLost, StoreMigrating, TIER_NAME


class ProjectHistoryImportPending(FileNotFoundError):
    """First-use evidence is not ready; writers still see an unavailable registry."""


def _pending_dialogue_files(root: pathlib.Path, *, streams=ds.STREAMS,
                            planned_archive=None,
                            imported: Optional[Callable[[str, str], Optional[int]]] = None) -> bool:
    """Whether live rows or archive segments await inspection; the first segment found answers.

    While the archive keeps ``planned_archive``, the stamp the importer's last plan recorded, nothing
    is listed. ``imported(stream, name)`` is the size the import ledger records for a segment it
    imported (else ``None``): then a regular file still of that size is accounted for, the
    importer's own rule (``dialogue_import._plan``). An unreadable source raises: strict evidence,
    never a silent absence."""
    for stream in streams:
        source = root / ds.LIVE_REL[stream]
        try:
            if source.stat().st_size:
                return True
        except FileNotFoundError:
            confirm_absent(source)
        except OSError:
            raise
    try:
        if planned_archive is not None:
            archive = (root / "archive").stat()
            if [archive.st_ino, archive.st_mtime_ns] == planned_archive:
                return False  # the importer already accounted for this directory; a new live tail was checked above
        with os.scandir(root / "archive") as entries:
            for entry in entries:
                stream = next((name for name in streams if entry.name.startswith(name + "_")), None)
                if stream is None or not entry.name.endswith(".jsonl"):
                    continue
                if imported is None:
                    return True
                try:  # the importer plans only a regular file, at the size it stats (``segment_plan``)
                    if entry.is_file() and imported(stream, entry.name) != os.stat(entry.path).st_size:
                        return True
                except FileNotFoundError:
                    return True  # changing under the listing: not accounted for
        return False
    except FileNotFoundError:
        confirm_absent(root / "archive")
    except OSError:
        raise
    return False


@contextmanager
def saved_history_snapshot(root: pathlib.Path) -> Iterator[sqlite3.Connection]:
    """Read the existing cutover, or one confirmed empty source, for the whole request."""
    identity = store.file_identity(ds.path(root))
    if identity is None:
        confirm_absent(ds.path(root))
        if _pending_dialogue_files(pathlib.Path(root)):
            raise StoreMigrating("the dialogue store is not prepared yet")
        confirm_absent(ds.path(root))  # cutover could have moved the files during their inspection
        with store_empty.empty_store(ds._SCHEMA) as conn, store.transaction(conn, write=False):
            yield conn
        return
    with ds.transaction(root, write=False, existing_only=True) as conn:
        if ds._meta(conn, "cutover") is None:  # read the current file's plan, including after Reset
            raise StoreMigrating("the dialogue store's import plan is not committed yet")
        yield conn
        if store.file_identity(ds.path(root)) != identity:
            raise StoreLost("the dialogue store changed during the archive read")


def could_lack_project_history(root: Any, *, initializing: bool = False) -> bool:
    """Consult retained Project facts without causing their import; distinguish a pending input from a loss."""
    root = pathlib.Path(root)

    def no_pending_files(stamp=None, imported=None):
        if _pending_dialogue_files(root, streams=("chat",), planned_archive=stamp, imported=imported):
            raise ProjectHistoryImportPending("Dialogue history is still importing")
        return True

    target = ds.path(root)
    identity = store.file_identity(target)
    if identity is None:
        confirm_absent(target)
        return no_pending_files()
    tier = ds.header_tier(root, identity)
    if tier not in ds._APPLICATION_ID:
        return False
    utils.assert_test_data_path(target)
    proof = ingress_proof_key(root)  # before the snapshot opens: a commit landing mid-scan re-keys it
    try:
        with contextlib.ExitStack() as stack:
            # Name-tier readers take the dialogue lock. WAL readers need SQLite's
            # live snapshot, including its shared memory even on the first open after close.
            if initializing or tier == TIER_NAME or store.wal_file(target):
                conn = stack.enter_context(ds.transaction(root, write=False, existing_only=True, wait_sec=0))
            else:
                uri = target.absolute().as_uri() + "?mode=ro&readonly_shm=1"
                conn = stack.enter_context(contextlib.closing(store.counted_connect(uri, target, uri=True, timeout=0)))
                conn.row_factory = sqlite3.Row
                conn.execute("BEGIN")
            if (conn.execute("PRAGMA application_id").fetchone()[0] != ds._APPLICATION_ID[tier]
                    or conn.execute("PRAGMA user_version").fetchone()[0] not in (ds.SCHEMA_VERSION, ds.SHARED_VERSION)):
                return False
            # The producers of these notices require a registered Project. Ordinary task
            # project_id metadata can instead refer to an external workspace's facts shelf.
            if conn.execute("SELECT 1 FROM dialogue INDEXED BY dialogue_kind WHERE stream='chat' AND kind IN ("
                            "'project_started', 'project_handoff', 'project_completion_summary', 'project_closure') "
                            "LIMIT 1").fetchone():
                return False
            # appearance_keys writes room:<chat> only for a room reserved by the registry. This
            # table may be absent on old stores; reading it never starts its derivation.
            if (conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='room_appearances'").fetchone()
                    and conn.execute("SELECT 1 FROM room_appearances WHERE key GLOB 'room:*' LIMIT 1").fetchone()):
                return False
            # A pending segment still becomes rows: the proof waits for its importer. A mismatched one is
            # distinct conflicted evidence -- its bytes moved after the ledger recorded them, so what it
            # holds is unknown. A skipped line is neither: permanently unreadable, it never becomes a row
            # at all, can never reach Main, and so proves no Project history -- an established root with a
            # torn line keeps its ordinary first use.
            unfinished = conn.execute("SELECT state FROM import_ledger INDEXED BY ledger_state WHERE stream='chat' "
                                      "AND state IN ('mismatch', 'pending') ORDER BY state LIMIT 1").fetchone()
            if unfinished:
                if unfinished[0] == "pending":
                    raise ProjectHistoryImportPending("Dialogue history is still importing")
                return False
            if could_have_web_project_ingress(conn, root, proof):
                return False
            # Within this snapshot, the live tail before the archive (a rotation between them lands in one): a
            # segment is imported only by the ledger whose rows the checks above read.
            no_pending_files(ds._meta(conn, "archive_stamp"), lambda stream, name: (conn.execute(
                "SELECT size FROM import_ledger WHERE stream=? AND name=? AND state='done'", (stream, name),
            ).fetchone() or [None])[0])
    except StoreMigrating as exc:  # an input still being prepared is pending, never a lost registry
        raise ProjectHistoryImportPending("Dialogue history is still importing") from exc
    except (sqlite3.Error, StoreError, ValueError):
        return False  # unreadable, changing, or under repair
    return store.file_identity(target) == identity


_INGRESS_PROOFS: Dict[tuple, bool] = {}  # one web-ingress answer per identity of everything the scan reads
_INGRESS_PROOFS_LIMIT = 64


def ingress_proof_key(root) -> Optional[tuple]:
    """Identity of everything the web-ingress scan reads: the dialogue store and its WAL (rows move only
    with those), the owner bindings' copy and the initialization witness that vouches for it
    (``message_origin._owner_chat``). Captured BEFORE the reader's snapshot opens: a commit landing during
    the scan moves those files, so a later reader re-keys and re-reads instead of reusing an answer another
    snapshot produced. ``None`` when a stat is unavailable -- every scan then re-reads."""
    from ouroboros.launch_roots import witness_path

    target = ds.path(root)
    key = [str(target)]
    for source in (target, target.with_name(target.name + "-wal"),
                   Path(root) / "state" / "state.json", witness_path(root)):
        try:
            key.append(utils.file_identity(os.stat(source)))
        except FileNotFoundError:
            key.append(None)
        except OSError:
            return None
    return tuple(key)


def could_have_web_project_ingress(conn: Any, root, proof: Optional[tuple] = None) -> bool:
    """Whether non-Main web ingress may have been Project traffic (``_scan_web_project_ingress``).

    While the registry is absent every strict read and live-bus frame re-proves first use, and that scan
    reads every non-Main inbound row: ``proof`` (``ingress_proof_key``) holds one answer per identity of
    everything it reads, so hot re-proofs do not infer over growing history. Stored rows are never
    retracted, so a held answer stands while its key reads the same."""
    # Re-key before trusting an earlier answer and after a new scan. A source moving
    # between those reads leaves only ambiguity, never a proven absence on mixed facts.
    if proof is not None and ingress_proof_key(root) != proof:
        return True
    cached = _INGRESS_PROOFS.get(proof) if proof is not None else None
    if cached is not None:
        return cached
    answer = _scan_web_project_ingress(conn, root)
    if proof is not None and ingress_proof_key(root) != proof:
        return True
    return _record_ingress_proof(proof, answer)


def _record_ingress_proof(proof: Optional[tuple], ambiguous: bool) -> bool:
    if proof is not None:
        if len(_INGRESS_PROOFS) >= _INGRESS_PROOFS_LIMIT:
            _INGRESS_PROOFS.clear()  # a memory bound, not a policy: the next scan re-reads
        _INGRESS_PROOFS[proof] = ambiguous
    return ambiguous


def _scan_web_project_ingress(conn: Any, root) -> bool:
    """Non-Main web ingress is ambiguous except a host-delivered decision in a bound owner chat: a late
    answer or a press on an ended author's card, accepted typed (``late_answer``/``card_press``) in the
    card's own chat (``task_decision.deliver_as_owner_message``), where the owner bindings prove it
    (``message_origin._owner_chat``)."""
    from ouroboros.card_store import late_press_ref
    from ouroboros.owner_quiz import late_answer_ref
    from supervisor.message_origin import _owner_chat

    # The decision producer's `source` says where the owner pressed, not where the card
    # was published. Read only the typed references (message_ingress), never the answer,
    # the pressed values, the question, or an id spelling.
    rows = conn.execute(
        "SELECT chat_id, json_extract(body, '$.late_answer.task_id'), json_extract(body, '$.late_answer.quiz_id'), "
        "json_extract(body, '$.card_press.task_id'), json_extract(body, '$.card_press.card_id'), "
        "json_extract(body, '$.card_press.request_id') "
        "FROM dialogue INDEXED BY dialogue_inbound WHERE stream='chat' AND direction='in' AND chat_id>? "
        "AND json_extract(body, '$.source')='web'", (WEB_UI_CHAT_ID,))
    owner_chats: Dict[Any, bool] = {}
    for chat, task_id, quiz_id, press_task, card_id, request_id in rows:
        answer = (isinstance(task_id, str) and isinstance(quiz_id, str)
                  and late_answer_ref({"task_id": task_id, "quiz_id": quiz_id}))
        press = (all(isinstance(value, str) for value in (press_task, card_id, request_id))
                 and late_press_ref({"task_id": press_task, "card_id": card_id, "request_id": request_id}))
        if not (answer or press):
            return True
        if chat not in owner_chats:
            owner_chats[chat] = _owner_chat(root, chat)
        if not owner_chats[chat]:
            return True
    return False
