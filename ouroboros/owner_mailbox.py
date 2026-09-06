"""Per-task owner-message mailboxes for running worker tasks."""
import json
import logging
import pathlib
import uuid
from typing import Any, Dict, List, Optional

from ouroboros.task_results import validate_task_id
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

_MAILBOX_DIR = "memory/owner_mailbox"

# Typed mailbox entry kinds. KIND_OWNER_TEXT entries are injected verbatim as
# owner dialogue; control kinds carry supervisor->worker protocol signals and
# are routed structurally (never shown as user prose).
KIND_OWNER_TEXT = "owner_text"
KIND_TASK_MESSAGE = "task_message"
KIND_FINALIZE_NOW = "finalize_now"
# Owner "hurry" control (HQ1, 2026-08-15): a task-local typed acceleration
# directive — NEVER owner dialogue and NEVER revoked after drain (restart
# re-drain must restore the attempt latch; only terminal cleanup removes it).
# Its ``text`` is the parser-required internal reason ("owner_hurry"), not prose.
KIND_HURRY = "hurry"
# Owner's verbatim quiz answer (#Q-2b): a typed control whose ``text`` is the
# chosen option label (plus an optional owner comment) — delivered inside a
# structural frame, never as forged free-form owner dialogue.
KIND_QUIZ_ANSWER = "quiz_answer"
# The mailbox is append-only, so a sender that changes its mind cannot delete the
# control it already wrote — it appends this retraction naming the target msg_id.
# Revocations are resolved by the READER over the whole mailbox, so a control that
# was revoked before anyone drained it is never delivered at all. Its payload IS
# the revoked msg_id (same convention as finalize_now, whose payload is a reason).
KIND_CONTROL_REVOKED = "control_revoked"


def _mailbox_path(drive_root: pathlib.Path, task_id: str) -> pathlib.Path:
    return pathlib.Path(drive_root) / _MAILBOX_DIR / f"{validate_task_id(task_id)}.jsonl"


def _ack_path(drive_root: pathlib.Path, task_id: str) -> pathlib.Path:
    return pathlib.Path(drive_root) / _MAILBOX_DIR / f"{validate_task_id(task_id)}.acks.jsonl"


def acknowledged_task_message_ids(
    drive_root: pathlib.Path,
    task_id: str,
    *,
    attempt_key: Any = None,
    _read_status: Optional[Dict[str, bool]] = None,
) -> set[str]:
    """Read acknowledgements effective for one physical attempt.

    Legacy task-message acks (no attempt field) remain globally effective.
    A legacy ack for durable owner text cannot prove which physical attempt
    incorporated it, so fresh attempts replay that exact directive until
    terminal cleanup. New owner-text acks are scoped by ``attempt_key``.
    Internal peek callers may also request complete-read evidence; a fail-soft
    empty set alone is not proof that every required source was readable.
    """

    if _read_status is not None:
        _read_status["complete"] = False
    complete = True
    path = _ack_path(drive_root, task_id)
    if not path.exists():
        if _read_status is not None:
            _read_status["complete"] = True
        return set()
    legacy_owner_ids: set[str] = set()
    if attempt_key is not None:
        try:
            content = _mailbox_path(drive_root, task_id).read_text(encoding="utf-8")
            complete = not content or content.endswith("\n")
            for line in content.splitlines():
                try:
                    entry = json.loads(line)
                except (TypeError, ValueError):
                    complete = False
                    continue
                if (
                    isinstance(entry, dict)
                    and str(entry.get("kind") or KIND_OWNER_TEXT) == KIND_OWNER_TEXT
                    and str(entry.get("msg_id") or "")
                ):
                    legacy_owner_ids.add(str(entry["msg_id"]))
        except FileNotFoundError:
            complete = False
        except OSError:
            complete = False
            log.warning(
                "Failed to classify legacy owner acknowledgements for %s",
                task_id, exc_info=True,
            )
    found: set[str] = set()
    try:
        content = path.read_text(encoding="utf-8")
        complete = complete and (not content or content.endswith("\n"))
        for line in content.splitlines():
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                complete = False
                continue
            if not isinstance(row, dict) or not str(row.get("msg_id") or ""):
                continue
            row_attempt = row.get("attempt_key")
            if (
                attempt_key is None
                or str(row_attempt) == str(attempt_key)
                or (row_attempt is None and str(row["msg_id"]) not in legacy_owner_ids)
            ):
                found.add(str(row["msg_id"]))
    except OSError:
        complete = False
        log.warning("Failed to read task-message acknowledgements for %s", task_id, exc_info=True)
    if _read_status is not None:
        _read_status["complete"] = complete
    return found


def acknowledge_task_messages(
    drive_root: pathlib.Path,
    task_id: str,
    msg_ids: List[str],
    *,
    wake_id: str,
    attempt_key: Any = None,
) -> bool:
    """Acknowledge messages only after their full content entered a transcript."""

    path = _ack_path(drive_root, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = acknowledged_task_message_ids(
        drive_root, task_id, attempt_key=attempt_key,
    )
    for msg_id in [str(item) for item in msg_ids if str(item) and str(item) not in existing]:
        row = {
            "ts": utc_now_iso(), "type": "task_message_acknowledged",
            "task_id": str(task_id), "msg_id": msg_id, "wake_id": str(wake_id or ""),
        }
        if attempt_key is not None:
            row["attempt_key"] = str(attempt_key)
            row["settled"] = False
        if not append_jsonl(path, row):
            return False
    return True


def acknowledge_transcript_entry(
    drive_root: pathlib.Path,
    task_id: str,
    entry: Dict[str, Any],
    *,
    wake_id: str = "loop_delivery",
    attempt_key: Any = None,
) -> None:
    """Durably acknowledge one mailbox entry after transcript injection."""
    if attempt_key is None:
        attempt_key = entry.get("_owner_attempt_key")
    msg_id = str(entry.get("msg_id") or "")
    if msg_id and not acknowledge_task_messages(
        drive_root, task_id, [msg_id], wake_id=wake_id, attempt_key=attempt_key,
    ):
        log.warning("Mailbox delivery acknowledgement failed for %s", msg_id)


def write_owner_message(
    drive_root: pathlib.Path,
    text: str,
    task_id: str,
    msg_id: Optional[str] = None,
    kind: str = KIND_OWNER_TEXT,
    client_surface: Optional[Dict[str, Any]] = None,
    attachment_manifest: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Write an owner message or typed control entry to a task's mailbox."""
    path = _mailbox_path(drive_root, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "msg_id": msg_id or uuid.uuid4().hex,
        "ts": utc_now_iso(),
        "text": text,
        "kind": str(kind or KIND_OWNER_TEXT),
    }
    if isinstance(client_surface, dict) and client_surface:
        # Owner Surface Fact (additive, like ``ts``): which client surface sent
        # this follow-up, so the loop can note a mid-task device change.
        entry["client_surface"] = dict(client_surface)
    if isinstance(attachment_manifest, list):
        entry["attachment_manifest"] = [
            dict(item) for item in attachment_manifest if isinstance(item, dict)
        ]
    try:
        if not append_jsonl(path, entry):
            log.warning("Failed to durably append owner message for task %s", task_id)
            return False
        return True
    except Exception:
        log.warning("Failed to write owner message for task %s", task_id, exc_info=True)
        return False


def write_task_message(
    drive_root: pathlib.Path,
    text: str,
    task_id: str,
    *,
    source_task_id: str,
    provenance: str = "ancestor_task",
    relayed_from_task_id: str = "",
    msg_id: Optional[str] = None,
) -> bool:
    """Write an addressed task-tree message without forging owner provenance."""

    if provenance not in {"ancestor_task", "peer_via_ancestor", "system", "descendant_task"}:
        return False
    path = _mailbox_path(drive_root, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "msg_id": msg_id or uuid.uuid4().hex,
        "ts": utc_now_iso(),
        "text": str(text or ""),
        "kind": KIND_TASK_MESSAGE,
        "provenance": provenance,
        "source_task_id": str(source_task_id or ""),
    }
    if relayed_from_task_id:
        entry["relayed_from_task_id"] = str(relayed_from_task_id)
    try:
        return bool(append_jsonl(path, entry))
    except Exception:
        log.warning("Failed to write task message for task %s", task_id, exc_info=True)
        return False


def owner_attachment_manifest(drive_root: pathlib.Path, task_id: str) -> List[Dict[str, Any]]:
    """Return every durable owner-text attachment row, including acknowledged mail."""

    path = _mailbox_path(drive_root, task_id)
    if not path.exists():
        return []
    manifests: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(entry, dict) or str(entry.get("kind") or KIND_OWNER_TEXT) != KIND_OWNER_TEXT:
                continue
            msg_id = str(entry.get("msg_id") or "")
            if msg_id and msg_id in seen_ids:
                continue
            if msg_id:
                seen_ids.add(msg_id)
            manifest = entry.get("attachment_manifest")
            if isinstance(manifest, list):
                manifests.extend(dict(item) for item in manifest if isinstance(item, dict))
    except OSError:
        log.warning("Failed to read owner attachment manifest for %s", task_id, exc_info=True)
    return manifests


def deliver_task_message(
    entry: Dict[str, Any], task_id: str, event_queue: Any, append_message: Any,
) -> None:
    """Render typed provenance and publish the corresponding worker event."""

    provenance = str(entry.get("provenance") or "ancestor_task")
    source = str(entry.get("source_task_id") or "unknown")
    relayed = str(entry.get("relayed_from_task_id") or "")
    if provenance == "peer_via_ancestor" and relayed:
        prefix = f"[Message from task {relayed}, relayed by ancestor {source}]"
    elif provenance == "system":
        prefix = "[System task message]"
    elif provenance == "descendant_task":
        # Escalation direction is upward: signing it "ancestor" would invert
        # the sender's place in the tree (decision 31 hierarchy).
        prefix = f"[Escalation from descendant task {source}]"
    else:
        prefix = f"[Message from ancestor task {source}]"
    append_message(f"{prefix}\n{entry.get('text') or ''}")
    if event_queue is not None:
        try:
            event_queue.put_nowait({
                "type": "task_message_injected", "task_id": task_id,
                "source_task_id": source, "provenance": provenance,
            })
        except Exception:
            pass


def deliver_quiz_answer(
    entry: Dict[str, Any], task_id: str, event_queue: Any, append_message: Any,
) -> None:
    """Inject a typed quiz answer (#Q-2b).

    The entry's ``text`` is the complete host-authored frame (structural
    header + the owner's VERBATIM chosen label and optional comment, composed
    at ingress time where the projection block is in hand). The model judges
    freshness itself from the asked/answered timestamps in the frame — no
    host staleness verdict (owner decision 30=A)."""
    append_message(str(entry.get("text") or ""))
    if event_queue is not None:
        try:
            event_queue.put_nowait({
                "type": "quiz_answer_injected", "task_id": task_id,
                "msg_id": str(entry.get("msg_id") or ""),
            })
        except Exception:
            pass


def revoke_owner_control(
    drive_root: pathlib.Path, task_id: str, control_msg_id: str,
) -> bool:
    """Retract an already-written control entry by its msg_id.

    The only way to un-send a mailbox control: the file is append-only, so this
    appends a revocation that ``drain_owner_entries`` resolves for every reader.
    Returns False when there is nothing to revoke or the append was not durable —
    the caller must then treat the control as STILL LIVE.
    """
    control_msg_id = str(control_msg_id or "").strip()
    if not control_msg_id:
        return False
    return write_owner_message(
        drive_root, control_msg_id, task_id, kind=KIND_CONTROL_REVOKED,
    )


def reset_attempt_controls_for_retry(
    drive_root: pathlib.Path,
    task_id: str,
) -> int:
    """Revoke live attempt-local controls without deleting durable owner text.

    A same-id retry is a new execution attempt, so ``hurry`` and
    ``finalize_now`` must not arm it.  Owner dialogue and its acknowledgement
    ledger are task authority, however, and survive until terminal cleanup.
    Reuse the mailbox's existing append-only revocation protocol so repeated
    retry resets are idempotent and no second delivery mechanism is introduced.
    """

    path = _mailbox_path(drive_root, task_id)
    if not path.exists():
        return 0
    try:
        rows: List[dict] = []
        revoked: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(row, dict):
                continue
            if str(row.get("kind") or KIND_OWNER_TEXT) == KIND_CONTROL_REVOKED:
                revoked.add(str(row.get("text") or ""))
            rows.append(row)
        reset = 0
        for row in rows:
            kind = str(row.get("kind") or KIND_OWNER_TEXT)
            msg_id = str(row.get("msg_id") or "")
            if kind not in {KIND_HURRY, KIND_FINALIZE_NOW} or not msg_id or msg_id in revoked:
                continue
            if revoke_owner_control(drive_root, task_id, msg_id):
                revoked.add(msg_id)
                reset += 1
        return reset
    except OSError:
        log.warning("Failed to reset owner controls for retry of %s", task_id, exc_info=True)
        return 0


def copy_owner_mailbox_for_retry(
    drive_root: pathlib.Path,
    task_id: str,
    retry_task_id: str,
    *,
    path_replacements: Optional[Dict[str, str]] = None,
) -> bool:
    """Carry the existing mailbox protocol across a new-id physical retry.

    Timeout retries of root tasks receive a fresh task id.  Copy the append-only
    mailbox and acknowledgement rows by value so the retry keeps the same
    durable message ids while later steering can continue on its new task id.
    Repeated calls merge idempotently instead of replacing any newer rows.
    """

    task_id = validate_task_id(task_id)
    retry_task_id = validate_task_id(retry_task_id)
    if task_id == retry_task_id:
        return True

    for source, target in (
        (_mailbox_path(drive_root, task_id), _mailbox_path(drive_root, retry_task_id)),
        (_ack_path(drive_root, task_id), _ack_path(drive_root, retry_task_id)),
    ):
        if not source.exists():
            continue
        try:
            source_rows = [
                json.loads(line)
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            target_rows = (
                [
                    json.loads(line)
                    for line in target.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if target.exists() else []
            )
        except (OSError, TypeError, ValueError):
            log.warning(
                "Failed to read owner mailbox retry handoff %s -> %s",
                task_id, retry_task_id, exc_info=True,
            )
            return False

        def _rebase(value: Any) -> Any:
            if isinstance(value, str):
                for old, new in (path_replacements or {}).items():
                    value = value.replace(str(old), str(new))
                return value
            if isinstance(value, list):
                return [_rebase(item) for item in value]
            if isinstance(value, dict):
                return {key: _rebase(item) for key, item in value.items()}
            return value

        normalized: List[dict] = []
        for row in source_rows:
            if not isinstance(row, dict):
                continue
            copied = dict(row)
            if "task_id" in copied:
                copied["task_id"] = retry_task_id
            normalized.append(_rebase(copied))
        fingerprints = {
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for row in target_rows if isinstance(row, dict)
        }
        for row in normalized:
            fingerprint = json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
            if fingerprint in fingerprints:
                continue
            if not append_jsonl(target, row):
                return False
            fingerprints.add(fingerprint)
    return True


def drain_owner_entries(
    drive_root: pathlib.Path,
    task_id: str,
    seen_ids: Optional[set] = None,
    attempt_key: Any = None,
    *,
    _read_status: Optional[Dict[str, bool]] = None,
) -> List[dict]:
    """Read unseen mailbox entries without mutating the append-only mailbox.

    Revocations are resolved over the WHOLE mailbox before anything is yielded,
    so a control retracted by a later line is never delivered even if the reader
    had not drained it yet; the revocation lines themselves are protocol and are
    never returned as content.
    ``_read_status`` distinguishes successful emptiness from failed/torn reads
    for wait-local peeks without changing the normal delivery projection.
    """
    if _read_status is not None:
        _read_status["complete"] = False
    path = _mailbox_path(drive_root, task_id)
    if not path.exists():
        if _read_status is not None:
            _read_status["complete"] = True
        return []
    if seen_ids is None:
        seen_ids = set()
    ack_status: Dict[str, bool] = {}
    seen_ids.update(
        acknowledged_task_message_ids(
            drive_root, task_id, attempt_key=attempt_key,
            **({"_read_status": ack_status} if _read_status is not None else {}),
        )
    )
    try:
        content = path.read_text(encoding="utf-8")
        complete = ack_status.get("complete", False) and (not content or content.endswith("\n"))
        content = content.strip()
        if not content:
            if _read_status is not None:
                _read_status["complete"] = complete
            return []
        parsed: List[dict] = []
        revoked: set = set()
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                complete = False
                log.debug("Malformed mailbox line for task %s", task_id, exc_info=True)
                continue
            if not isinstance(entry, dict):
                complete = False
                continue
            if str(entry.get("kind") or KIND_OWNER_TEXT) == KIND_CONTROL_REVOKED:
                revoked.add(str(entry.get("text") or ""))
            parsed.append(entry)
        entries = []
        for entry in parsed:
            mid = entry.get("msg_id", "")
            if mid and mid in seen_ids:
                continue
            if mid:
                seen_ids.add(mid)
            kind = str(entry.get("kind") or KIND_OWNER_TEXT)
            if kind == KIND_CONTROL_REVOKED or (mid and mid in revoked):
                continue
            text = entry.get("text", "")
            if text:
                # ``ts`` is ADDITIVE (2026-08-15 Fable pin): typed controls such
                # as ``hurry`` carry their request time into the drained entry so
                # the attempt latch can preserve when the owner actually asked.
                drained = {
                    "msg_id": mid, "text": text, "kind": kind,
                    "ts": str(entry.get("ts") or ""),
                }
                # Owner Surface Fact: the drain projection must carry the field
                # explicitly or a written fact is never delivered (the exact
                # dead-wire class this sprint closes).
                if isinstance(entry.get("client_surface"), dict) and entry.get("client_surface"):
                    drained["client_surface"] = dict(entry["client_surface"])
                if isinstance(entry.get("attachment_manifest"), list):
                    drained["attachment_manifest"] = [
                        dict(item) for item in entry["attachment_manifest"]
                        if isinstance(item, dict)
                    ]
                if attempt_key is not None and kind == KIND_OWNER_TEXT:
                    drained["_owner_attempt_key"] = attempt_key
                if kind == KIND_TASK_MESSAGE:
                    drained["provenance"] = str(entry.get("provenance") or "ancestor_task")
                    drained["source_task_id"] = str(entry.get("source_task_id") or "")
                    drained["relayed_from_task_id"] = str(entry.get("relayed_from_task_id") or "")
                entries.append(drained)
        if _read_status is not None:
            _read_status["complete"] = complete
        return entries
    except Exception:
        log.debug("Failed to read mailbox for task %s", task_id, exc_info=True)
        return []


class OwnerMailboxPeek:
    """One wait's proven-empty mailbox snapshot; never delivery or ACK authority."""

    def __init__(self) -> None:
        self._empty_key: Any = None

    @staticmethod
    def _fingerprint(root: pathlib.Path, task_id: str, attempt: Any, seen: set) -> tuple:
        files = []
        for path in (_mailbox_path(root, task_id), _ack_path(root, task_id)):
            try:
                stat = path.stat()
                files.append((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
            except FileNotFoundError:
                files.append(None)
        return (str(root.resolve()), task_id, None if attempt is None else str(attempt), frozenset(seen), tuple(files))

    def pending(self, root: pathlib.Path, task_id: str, seen: set, attempt: Any) -> bool:
        try:
            before = self._fingerprint(root, task_id, attempt, seen)
        except OSError:
            before = None
        if before is not None and before == self._empty_key:
            return False
        self._empty_key = None
        status: Dict[str, bool] = {}
        # Drain changes only this private set; normal loop delivery owns the real one.
        entries = drain_owner_entries(root, task_id, set(seen), attempt, _read_status=status)
        if not entries and before is not None and status.get("complete"):
            try:
                if before == self._fingerprint(root, task_id, attempt, seen):
                    self._empty_key = before
            except OSError:
                pass  # uncertainty never becomes a remembered empty mailbox
        return bool(entries)


def drain_owner_messages(
    drive_root: pathlib.Path,
    task_id: str,
    seen_ids: Optional[set] = None,
) -> List[str]:
    """Read unseen owner-dialogue message texts (legacy text-only view)."""
    return [
        entry["text"]
        for entry in drain_owner_entries(drive_root, task_id, seen_ids=seen_ids)
        if entry.get("kind", KIND_OWNER_TEXT) == KIND_OWNER_TEXT
    ]


def cleanup_task_mailbox(drive_root: pathlib.Path, task_id: str) -> None:
    """Remove a task's mailbox file after task completes."""
    for path in (_mailbox_path(drive_root, task_id), _ack_path(drive_root, task_id)):
        try:
            if path.exists():
                path.unlink()
        except Exception:
            log.debug("Failed to cleanup mailbox for task %s", task_id, exc_info=True)


def sweep_settled_owner_mailboxes(drive_root: pathlib.Path) -> Dict[str, Any]:
    """Startup sweep of mailboxes whose task died off the terminal paths (CPL4-C18).

    The only regular unlink is the task_done dispatch; a task that never
    reached it (crash, lost event, hard kill) leaked its mailbox forever.
    A mailbox goes ONLY when the task's durable result is SETTLED — no result
    or a non-terminal result keeps it (fail-closed: an undelivered owner
    directive must survive any ambiguity). Lock sidecars are untouched
    (self-healing by staleness).
    """
    report: Dict[str, Any] = {"removed": [], "kept": 0}
    mailbox_dir = pathlib.Path(drive_root) / _MAILBOX_DIR
    try:
        entries = sorted(p for p in mailbox_dir.glob("*.jsonl") if p.is_file())
    except OSError:
        return report
    for path in entries:
        stem = path.name[: -len(".jsonl")]
        if stem.endswith(".acks"):
            continue  # swept with its mailbox
        try:
            task_id = validate_task_id(stem)
        except Exception:
            report["kept"] += 1
            continue  # not a task mailbox we can reason about: keep
        try:
            from ouroboros.task_results import load_task_result
            from ouroboros.task_status import SETTLED_STATUSES

            result = load_task_result(pathlib.Path(drive_root), task_id) or {}
            settled = str(result.get("status") or "") in SETTLED_STATUSES
        except Exception:
            settled = False
        if not settled:
            report["kept"] += 1
            continue
        cleanup_task_mailbox(pathlib.Path(drive_root), task_id)
        if path.exists():
            report["kept"] += 1  # unlink refused: still owned by the mailbox
        else:
            report["removed"].append(task_id)
    return report
