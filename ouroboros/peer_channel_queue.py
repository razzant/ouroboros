"""Durable tool-message intents and one server delivery loop per peer link."""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import pathlib
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from ouroboros.utils import atomic_write_json, in_worker_process, utc_now_iso

log = logging.getLogger(__name__)

_TRANSPORT_CHANNEL = "channel"
_FLUSHING_LOCK = threading.Lock()
_FLUSHING_LINKS: dict[tuple[str, str], dict[str, Any]] = {}
_UNPERSISTED_RETRY: dict[tuple[str, str, str], tuple[float, float, int]] = {}
_WORKER_WAKE_QUEUE: Any = None
_SWITCH_RECHECK_SEC = 5.0
_OUTCOME_WAIT_SEC = 90.0  # leave room in the 120s message_ouroboros tool for admission and result delivery


def bind_worker_event_queue(queue: Any) -> None:
    """Bind the worker's supervisor event queue so peer-link wakes route to the server coordinator."""
    global _WORKER_WAKE_QUEUE
    _WORKER_WAKE_QUEUE = queue


def _emit_worker_link_wake(root: Any, link_id: str = "") -> bool:
    """Ask the server's per-link coordinator to flush ``link_id`` (or all queued links when empty)."""
    q = _WORKER_WAKE_QUEUE
    if q is None:
        return False
    try:
        q.put({"type": "peer_link_wake", "drive_root": str(pathlib.Path(root)), "link_id": str(link_id or "")})
        return True
    except Exception:  # noqa: BLE001 - best-effort control wake never fails the caller
        log.debug("worker peer_link_wake emit failed for %s", link_id, exc_info=True)
        return False

# Transient task-level launch holds: stay queued without POST and without timers until Resume / store recovery.
_TASK_PAUSE_HOLD_CODES = frozenset({
    "owner_pause", "owner_pause_before_review_launch", "model_sleep",
    "owner_launch_authority_unavailable", "owner_pause_authority_unreadable",
    "owner_pause_authority_missing", "model_sleep_authority_unreadable",
    "operation_already_returned", "task_control_unreadable",
})


def _canonical_root(root: Any) -> str:
    return os.path.realpath(str(pathlib.Path(root)))


def _queued_dir(root: Any, link_id: str) -> pathlib.Path:
    from ouroboros import peer_links

    return peer_links.token_path(root, link_id).parent / "queued"


def _queued_path(root: Any, link_id: str, letter_id: str) -> pathlib.Path:
    return _queued_dir(root, link_id) / f"{hashlib.sha256(str(letter_id).encode('utf-8')).hexdigest()}.json"


def record_queued_message(
    root: Any, link_id: str, letter_id: str, *, place_id: str, account_id: str, text: str, reply_to: str,
    origin: Mapping[str, Any], approved_basis: str, coverage_kind: str, root_task_id: str = "",
    wake: bool = True,
    retry_post: bool = False, control_required: bool = False, fresh_approval: bool = True,
) -> dict[str, Any]:
    """Persist one intent before waking its server. A fresh approval changes only its speech facts, never its age."""
    from ouroboros import peer_links
    from ouroboros.utils import json_update_lock

    path, now = _queued_path(root, link_id, letter_id), utc_now_iso()
    try:
        with json_update_lock(peer_links.registry_path(root)):
            row = peer_links.read_registry(root)["links"].get(link_id)
            if (not isinstance(row, dict) or row.get("revoked_at")
                    or str(row.get("peer_install_id") or "") != account_id):
                raise peer_links.PeerLinkError("peer_link_unavailable", "the link changed; nothing was sent", 409)
            old = queued_message(root, link_id, letter_id)
            channel = str(row.get("transport") or "url") == _TRANSPORT_CHANNEL
            if old is not None:
                if (old.get("account_id"), old.get("place_id"), old.get("text"), old.get("reply_to")) != (
                        account_id, place_id, text, reply_to):
                    raise peer_links.PeerLinkError("queued_identity_mismatch", "different queued letter identity", 409)
            if old is not None and (channel or old.get("pending", True)):
                record = old
                if fresh_approval:
                    record = {**old, "approved_basis": approved_basis, "coverage_kind": coverage_kind, "at": now,
                              "approval_at": now, "last_outcome": None}
                    atomic_write_json(path, record, fsync=True)
            elif old is not None and not retry_post:
                record = old
            else:
                old = old or {}
                sent = peer_links.sending_record(root, link_id, letter_id) or {} if retry_post and not channel else {}
                record = {
                    "letter_id": letter_id, "link_id": link_id, "place_id": place_id, "account_id": account_id,
                    "text": text, "reply_to": reply_to, "origin": dict(origin), "root_task_id": root_task_id,
                    "approved_basis": approved_basis, "coverage_kind": coverage_kind, "control_required": control_required,
                    "transport": "channel" if channel else "url", "pending": True,
                    "retry_pass": int(sent.get("passes") or 0) if retry_post and not channel else None,
                    "queued_at": old.get("queued_at") or now, "at": now,
                    "approval_at": now if fresh_approval else old.get("approval_at") or old.get("queued_at") or now,
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(path, record, fsync=True)
                path.with_suffix(".parked").unlink(missing_ok=True)
    except OSError as exc:
        raise peer_links.PeerLinkError("peer_links_unreadable", f"the queued message could not be saved: {exc}", 503) from exc
    if wake:
        trigger_link_flush(root, link_id)
    return record


def _write_queue_outcome(root: Any, link_id: str, letter_id: str, seen: tuple, outcome: dict, *,
                         observed_retry: int, park: bool = False) -> bool:
    from ouroboros import peer_links
    from ouroboros.utils import json_update_lock

    with json_update_lock(peer_links.registry_path(root)):
        if queued_version(root, link_id, letter_id) != seen:
            return False  # a new approval or revoke won
        path = _queued_path(root, link_id, letter_id)
        existing = queued_message(root, link_id, letter_id)
        updated = {**existing, "pending": not park, "last_outcome": outcome, "at": utc_now_iso(),
                   "observed_retry": observed_retry}
        if park:
            # A parked name dominates a shadowed .json after a crash in either transition.
            atomic_write_json(path.with_suffix(".parked"), updated, fsync=True)
            path.unlink(missing_ok=True)
        else:
            atomic_write_json(path, updated, fsync=True)
    if park:
        from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

        task_id = str((updated.get("origin") or {}).get("task_id") or "")
        row = load_task_result(root, task_id) if task_id else None
        if isinstance(row, dict) and row.get("status") in _TRULY_TERMINAL_STATUSES:
            # The producer can finish while the server is still recording the URL pass.
            _retire_parked(root, link_id, path.with_suffix(".parked"), task_id)
    return True


def queued_message(root: Any, link_id: str, letter_id: str) -> dict[str, Any] | None:
    from ouroboros import peer_links

    active = _queued_path(root, link_id, letter_id)
    for path in (active.with_suffix(".parked"), active, active.with_suffix(".parked")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            raise peer_links.PeerLinkError(
                "peer_links_unreadable", f"the queued message {path.name} cannot be read now: {exc}", 503,
            ) from exc
        if not isinstance(row, dict) or row.get("letter_id") != letter_id:
            raise peer_links.PeerLinkError("peer_links_unreadable", f"the queued message {path.name} needs repair", 503)
        return row
    return None


def queued_version(root: Any, link_id: str, letter_id: str) -> tuple | None:
    """Exact file version of a queued message (inode, mtime_ns, size, sha256) for compare-and-delete."""
    from ouroboros import peer_links

    path = _queued_path(root, link_id, letter_id)
    try:
        if path.with_suffix(".parked").exists():
            return None
        with path.open("rb") as reader:
            st, data = os.fstat(reader.fileno()), reader.read()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise peer_links.PeerLinkError(
            "peer_links_unreadable", f"the queued message {path.name} cannot be read now: {exc}", 503,
        ) from exc
    return st.st_ino, st.st_mtime_ns, st.st_size, hashlib.sha256(data).hexdigest()


def settle_queued_message(root: Any, link_id: str, letter_id: str, *, seen: tuple | None = None,
                          wire_terminal: bool = False) -> bool:
    """Compare and retire, without forgetting an unsettled letter's wait."""
    from ouroboros import peer_links
    from ouroboros.utils import json_update_lock

    with json_update_lock(peer_links.registry_path(root)):
        if seen is not None and queued_version(root, link_id, letter_id) != seen:
            return False
        _queued_path(root, link_id, letter_id).unlink(missing_ok=True)
    with _FLUSHING_LOCK:
        key = (_canonical_root(root), str(link_id), str(letter_id))
        if wire_terminal or not _UNPERSISTED_RETRY.get(key, (0, 0, 0))[2]:
            _UNPERSISTED_RETRY.pop(key, None)  # without a durable write, an unsettled reply must still wait
    return True


def queued_messages(root: Any, link_id: str = "") -> list[tuple[str, dict[str, Any]]]:
    """``(link_id, record)`` of every queued tool message waiting for a channel, oldest first."""
    from ouroboros import peer_links

    found: list[tuple[str, str, dict[str, Any]]] = []
    links = peer_links.read_registry(root)["links"]
    target_ids = ([link_id] if link_id in links else []) if link_id else list(links)
    for lid in target_ids:
        folder = _queued_dir(root, lid)
        for entry in (folder.iterdir() if folder.is_dir() else ()):
            if not _active_queue_entry(entry):
                continue
            try:
                row = json.loads(entry.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(row, dict) and row.get("letter_id") and row.get("pending", True):
                found.append((str(row.get("queued_at") or ""), lid, row))
    return [(lid, row) for _at, lid, row in sorted(found, key=lambda item: item[0])]


def _check_task_stop(root: pathlib.Path, task_id: str, root_task_id: str) -> tuple[str, str] | None:
    """Strictly check whether the originating task (or an ancestor via cascade Stop) was stopped."""
    from ouroboros import cancel_intents, owner_pause
    from ouroboros.task_results import load_task_result

    if not task_id:
        return None
    tree_root = root_task_id or task_id
    try:
        if cancel_intents.immediate_stop_pending(root, task_id, tree_root_id=tree_root):
            return "task_stopped", "the originating task or its tree was stopped; nothing was sent"
        row = load_task_result(root, task_id, strict=True)
        if isinstance(row, dict) and str(row.get("status") or "") == "cancelled":
            return "task_stopped", "the originating task was stopped; nothing was sent"
        seen = {task_id}
        anc = str(row.get("parent_task_id") or "") if isinstance(row, dict) else ""
        candidates: list[str] = []
        if anc and anc not in seen:
            candidates.append(anc)
        if tree_root and tree_root not in seen and tree_root not in candidates:
            candidates.append(tree_root)
        while candidates:
            cur_id = candidates.pop(0)
            if cur_id in seen:
                continue
            seen.add(cur_id)
            anc_row = load_task_result(root, cur_id, strict=True)
            if isinstance(anc_row, dict):
                if owner_pause.owner_stopped(anc_row):
                    origin = anc_row.get("cancel_origin")
                    scope = str(origin.get("scope") or "") if isinstance(origin, dict) else ""
                    if scope == cancel_intents.SCOPE_CASCADE:
                        return "task_stopped", "the originating task tree was stopped; nothing was sent"
                parent = str(anc_row.get("parent_task_id") or "")
                if parent and parent not in seen:
                    candidates.append(parent)
    except Exception:  # noqa: BLE001 - unreadable control/lineage state stays safely queued, never sends or drops
        return "task_control_unreadable", "the originating task's stop state cannot be read now; nothing was sent"
    return None


def _queued_task_gate(root: pathlib.Path, task_id: str, root_task_id: str, *,
                      control_required: bool = False) -> Callable[[Any], Any] | None:
    """Launch gate for a queued ``message_ouroboros`` send: honors originating task Stop and Pause fences."""
    if not task_id:
        return None
    tree_root = root_task_id or task_id
    source = SimpleNamespace(
        task_id=task_id, root_task_id=tree_root, drive_root=root, budget_drive_root=root,
        task_lifecycle_bound=control_required,
        task_metadata={"root_task_id": tree_root, "budget_drive_root": str(root)},
    )

    def _gate(attempt: Any) -> Any:
        from ouroboros import owner_pause, peer_replies

        if stop_hold := _check_task_stop(root, task_id, tree_root):
            raise peer_replies.AttemptHeld(*stop_hold)

        def submit() -> Any:
            if stop_hold := _check_task_stop(root, task_id, tree_root):
                raise peer_replies.AttemptHeld(*stop_hold)
            return attempt()

        try:
            return owner_pause.run_operation(source, submit)
        except owner_pause.OwnerPauseRefused as exc:
            code = str(exc) or "owner_pause"
            if code not in _TASK_PAUSE_HOLD_CODES:
                code = "owner_pause" if getattr(exc, "held", False) else "task_control_unreadable"
            raise peer_replies.AttemptHeld(code, f"{exc or code}; nothing was sent") from exc
        except Exception as exc:  # noqa: BLE001 - StoreBusy or unexpected launch-gate error stays transient
            if type(exc).__name__ in {"StoreBusy", "StoreError"}:
                raise peer_replies.AttemptHeld("task_control_unreadable", f"{type(exc).__name__}; nothing was sent") from exc
            raise

    return _gate


def _queued_conversation_key(root: pathlib.Path, *, place_id: str, link_id: str, account_id: str) -> str | None:
    """Resolve the conversation key for a queued message's room; ``None`` if place store is unreadable."""
    from ouroboros import peer_links
    from ouroboros.places import place_records
    from ouroboros.presence_bindings import conversation_key

    try:
        place = place_records(root).places.get(place_id)
    except Exception:  # noqa: BLE001 - unreadable place store stays conservative
        return None
    if place is not None:
        return conversation_key(place.provider, place.account_id, place.conversation_id, "")
    return conversation_key(peer_links.CONNECTOR, account_id, link_id, "")


def _hold_queued_message(
    root: pathlib.Path, *, place_id: str, link_id: str, account_id: str, letter_id: str, text: str,
    reply_to: str, origin: dict, outcome: dict, identity: str, earlier_uncertain: bool, states: set[str],
) -> bool:
    """Record a hold notice in the room (preceded by an ``uncertain`` receipt when needed).

    Returns ``True`` only once the notice is durably written; ``False`` preserves the queued record.
    """
    from ouroboros import peer_replies
    from ouroboros.place_reply import record_notice

    if earlier_uncertain and "uncertain" not in states:
        if not peer_replies.record_receipt(root, link_id, account_id, letter_id, "uncertain",
                                           text=text, reply_to=reply_to, origin=origin):
            return False
    key = _queued_conversation_key(root, place_id=place_id, link_id=link_id, account_id=account_id)
    if key is None:
        log.warning("could not read place %s while holding queued message %s", place_id, letter_id)
        return False
    note = str(outcome.get("message") or outcome.get("code") or "held")
    try:
        return bool(record_notice(root, place_id=place_id, conversation_key=key, task_id=str(origin.get("task_id") or ""),
                                  text=f"Message held: {note}", identity=identity, held_text=text,
                                  transport={"conversation_key": key, "peer_delivery": outcome}))
    except Exception:  # noqa: BLE001 - failed notice write leaves queued record intact for retry
        log.warning("could not record hold notice for queued message %s", letter_id, exc_info=True)
        return False


def retire_link_waits(root: Any, link_id: str) -> None:
    key = (_canonical_root(root), link_id)
    with _FLUSHING_LOCK:
        for mem_key in tuple(_UNPERSISTED_RETRY):
            if mem_key[:2] == key:
                _UNPERSISTED_RETRY.pop(mem_key, None)
        if state := _FLUSHING_LINKS.get(key):
            state["revoked"] = True
            state["wake_event"].set()


def deadline(root: Any, lid: str, letter_id: str, sending: Mapping[str, Any]) -> tuple[bool, float | None, float]:
    """Observable deadline of one letter. An in-process unpersisted deadline never grants a send."""
    from ouroboros import peer_links

    key = (_canonical_root(root), lid, letter_id)
    with _FLUSHING_LOCK:
        memory = _UNPERSISTED_RETRY.get(key)
    at = float(sending.get("not_before") or 0.0)
    if memory is None:
        wait = at - time.time()
    else:
        wall, mono, step = memory
        at = max(at, wall)
        if step and not peer_links.set_sending_deadline(root, lid, letter_id, at, step):
            return False, None, at
        wait = max(at - time.time(), wall - time.time(), mono - time.monotonic())
        with _FLUSHING_LOCK:
            if _UNPERSISTED_RETRY.get(key) is not memory:
                return False, None, at  # revoke or settlement retired this wait during the write
            if wait <= 0:
                _UNPERSISTED_RETRY.pop(key, None)
            elif step:
                _UNPERSISTED_RETRY[key] = (wall, mono, 0)  # retain same-process monotonic floor
    return wait <= 0, wait if wait > 0 else None, at


def delay_letter(root: Any, lid: str, letter_id: str, record: Mapping[str, Any], *,
                 receiver_wait: float, place_id: str, account_id: str, task_id: str, text: str,
                 automatic: bool = True) -> tuple[float | None, float]:
    """Keep one deadline on the addressed envelope; a failed write parks only this letter."""
    from ouroboros import peer_links, peer_replies
    from ouroboros.place_reply import record_notice

    delays, _maximum = peer_replies.follow_up_schedule() if automatic else ((), 0.0)
    step = int(record.get("retry_step") or 0)
    delay = max(float(receiver_wait), delays[min(step, len(delays) - 1)] if delays else 0.0)
    at, mono = time.time() + delay, time.monotonic() + delay
    saved = peer_links.set_sending_deadline(root, lid, letter_id, at, step + 1)
    key = (_canonical_root(root), lid, letter_id)
    with _FLUSHING_LOCK:
        if _FLUSHING_LINKS.get(key[:2], {}).get("revoked"):
            return None, 0.0
        if automatic or not saved:
            _UNPERSISTED_RETRY[key] = (at, mono, 0 if saved else step + 1)
    if saved:
        return (max(0.0, mono - time.monotonic()) if delays else None), at
    room = _queued_conversation_key(pathlib.Path(root), place_id=place_id, link_id=lid, account_id=account_id)
    if room is not None:
        with contextlib.suppress(Exception):
            record_notice(root, place_id=place_id, conversation_key=room, task_id=task_id,
                          text=f"Message delayed: the {delay:g}s retry wait could not be saved; holding this letter",
                          identity=f"{letter_id}:retry_unpersisted", held_text=text)
    return None, at


def _wait_result(at: float, earlier: str) -> dict:
    return {"state": "uncertain" if earlier == "uncertain" else "waiting", "code": "retry_not_before",
            "queued": True, "not_before": at}


def _hold_ids(letter_id: str, approval: Mapping[str, Any]) -> dict[str, str]:
    return {"task_stopped": f"{letter_id}:task_stopped", "held": f"{letter_id}:held",
            "rules_changed": f"{letter_id}:rules_changed:{approval.get('approval_at') or approval.get('queued_at')}"}


def _flush_one_queued(root: pathlib.Path, lid: str, rec: Mapping[str, Any], observed_retry: int = 0,
                      switch_hold: tuple[str, str] | None = None) -> tuple[dict, float | None]:
    from ouroboros import peer_replies

    letter_id, place_id, account_id = (str(rec.get(k) or "") for k in ("letter_id", "place_id", "account_id"))
    seen = queued_version(root, lid, letter_id)
    live = queued_message(root, lid, letter_id)
    if live is None or seen is None:
        return {}, None

    def write_outcome(outcome: dict, *, park: bool = False) -> None:
        _write_queue_outcome(root, lid, letter_id, seen, outcome, observed_retry=observed_retry, park=park)

    rec = live
    text, reply_to = str(rec.get("text") or ""), str(rec.get("reply_to") or "")
    origin = dict(rec.get("origin") or {})
    kind, basis = str(rec.get("coverage_kind") or "consciousness"), str(rec.get("approved_basis") or "")
    task_id = str(origin.get("task_id") or "")
    tree = str(rec.get("root_task_id") or task_id)
    still_waiting = peer_replies.STILL_OWED | _TASK_PAUSE_HOLD_CODES
    channel = rec.get("transport", "channel") == "channel"
    earlier, states, sending, _error = peer_replies.earlier_state(root, lid, account_id, letter_id)
    if earlier in {"delivered", "failed"}:
        sent = peer_replies.reconcile_confirmed(root, lid, account_id, letter_id, earlier=earlier, states=states,
                                                record=sending, text=text, reply_to=reply_to, origin=origin)
        if sent["recorded"]:
            settle_queued_message(root, lid, letter_id, seen=seen, wire_terminal=True)
        else:
            write_outcome(sent)
        return sent, None
    if earlier == "unknown":
        unknown = {"state": "unknown", "code": _error, "message": "earlier attempts cannot be read now"}
        write_outcome(unknown)
        return _pending_outcome(unknown), None
    if not rec.get("pending", True):
        return dict(rec.get("last_outcome") or {}), None
    if (retry_pass := rec.get("retry_pass")) is not None and int(sending.get("passes") or 0) > retry_pass:
        # A previous server generation began this explicit attempt; this intent cannot buy a second one.
        result = peer_replies.kept_outcome("uncertain")
        write_outcome(result, park=True)
        return result, None
    room = _queued_conversation_key(root, place_id=place_id, link_id=lid, account_id=account_id)
    if room is None:
        return {}, None
    hold_ids = _hold_ids(letter_id, rec)
    for identity in hold_ids.values():
        if held := _notice_outcome(root, room, identity):
            settle_queued_message(root, lid, letter_id, seen=seen)
            return held, None

    eligible, wait, at = deadline(root, lid, letter_id, sending)
    if not eligible and switch_hold is None:
        waiting = _wait_result(at, earlier)
        write_outcome(waiting)
        return waiting, wait

    def ready() -> tuple[str, str] | None:
        if stop := _check_task_stop(root, task_id, tree):
            return stop
        if not basis:
            return "rules_unreadable", "the rules these words were approved under are not recorded; nothing was sent"
        return peer_replies.check_speech_basis(root, place_id, lid, kind=kind, approved_basis=basis,
                                               expected_account_id=account_id)

    # A URL retry buys one new POST. Channel transport on ambiguous break stops immediately and awaits a wake.
    if switch_hold is not None:
        sent = peer_replies.prewire_hold(letter_id, earlier, *(_check_task_stop(root, task_id, tree) or switch_hold))
    else:
        sent = peer_replies.send_peer_message(root, place_id, letter_id=letter_id, text=text, reply_to=reply_to,
                                             origin=origin, ready=ready, link_id=lid, account_id=account_id,
                                             gate=_queued_task_gate(root, task_id, tree,
                                                                    control_required=bool(rec.get("control_required"))),
                                             max_attempts=1 if retry_pass is not None else peer_replies._SEND_ATTEMPTS)
    state, code = sent["state"], str(sent.get("held_code") or sent.get("code") or "")
    if state in {"delivered", "failed"}:
        if sent.get("recorded"):
            settle_queued_message(root, lid, letter_id, seen=seen, wire_terminal=True)
        else:
            write_outcome(sent)
        return sent, None
    if sent.get("channel_wait") and queued_message(root, lid, letter_id) is None:
        return {**sent, "channel_wait": False}, None  # a revoke won while the wire outcome was pending
    if state == "uncertain" and sent.get("channel_wait"):
        sent = {**sent, "code": "channel_unconfirmed"}
    if channel and state in {"retry", "uncertain"} and not (
            sent.get("spent") or sent.get("channel_wait") or sent.get("held_code")):
        after, at = delay_letter(root, lid, letter_id, sending,
                                 receiver_wait=float(sent.get("retry_after") or 0.0),
                                 place_id=place_id, account_id=account_id, task_id=task_id, text=text)
        waiting = _wait_result(at, state)
        write_outcome(waiting)
        return waiting, after
    if state in {"unknown", "busy"} or code in still_waiting or (channel and sent.get("channel_wait")):
        outcome = (_wait_result(at, earlier) if not eligible else _pending_outcome(sent))
        write_outcome(outcome)
        if code == "switch_unreadable":
            wait = min(wait, _SWITCH_RECHECK_SEC) if wait is not None else _SWITCH_RECHECK_SEC
        return outcome, wait
    if state == "held" or sent.get("spent") or code == "attempts_spent" or sent.get("held_code"):
        hold_kind = code if (code == "task_stopped" or (code == "rules_changed" and earlier == "none"
                            and int(sending.get("passes") or 0) == 0)) else "held"
        if _hold_queued_message(root, place_id=place_id, link_id=lid, account_id=account_id, letter_id=letter_id,
                                text=text, reply_to=reply_to, origin=origin, outcome=sent, identity=hold_ids[hold_kind],
                                earlier_uncertain=earlier == "uncertain", states=states):
            settle_queued_message(root, lid, letter_id, seen=seen)
        return sent, None
    # The URL transport leaves later attempts to the caller and does not occupy an active queue slot.
    if not channel:
        if sent.get("retry_after") is not None:
            delay_letter(root, lid, letter_id, sending, receiver_wait=float(sent["retry_after"]),
                         place_id=place_id, account_id=account_id, task_id=task_id, text=text, automatic=False)
        write_outcome(sent, park=True)
    return sent, None


def _notice_outcome(root: pathlib.Path, key: str, identity: str) -> dict:
    from ouroboros import dialogue_store
    from ouroboros.places import conversation_chat_id

    rows = dialogue_store.rows_by_client_id(root, f"place-notice:{identity}", chat_id=conversation_chat_id(key), limit=1)
    if not rows:
        return {}
    return (rows[0].row.get("transport") or {}).get("peer_delivery") or {
        "state": "held", "code": "letter_held", "message": "an earlier hold was recorded"}


def _pending_outcome(outcome: dict | None = None) -> dict:
    outcome = outcome or {}
    if outcome and outcome["state"] not in {"delivered", "failed"}:
        # A transient hold is not a refusal to queue. A later wake still owns this letter.
        return {**outcome, "queued": True, "code": outcome.get("held_code") or outcome.get("code")}
    return outcome or {"state": "pending", "code": "delivery_pending",
                       "message": "the server has the letter, but this call has no delivery outcome yet"}


def _observe_terminal(root: Any, link_id: str, account_id: str, letter_id: str) -> dict | None:
    """Observe a server pass after its intent was removed, without writing a receipt from a worker."""
    from ouroboros import peer_replies

    state, states, sending, _ = peer_replies.earlier_state(root, link_id, account_id, letter_id)
    if state == "uncertain":
        return peer_replies.kept_outcome(state)
    if state not in {"delivered", "failed"}:
        return None
    out = {"state": state, "recorded": state in states}
    if state == "failed":
        out.update(peer_replies._refusal_fields(sending))
        with contextlib.suppress(Exception):  # legacy receipts may not carry a code or detail
            receipt = next(row for row in peer_replies._recorded_receipts(root, link_id, account_id, letter_id) if row["state"] == state)
            out.update((receipt.get("message") or {}).get("peer_refusal") or {})
    return out


def _retry_observation(root: Any, lid: str, *, after: int | None = None) -> int:
    """Order retry observations across workers and the server without depending on clock resolution."""
    from ouroboros import peer_links

    def observe(registry: dict) -> int:
        link = registry["links"].get(lid)
        if not isinstance(link, dict) or link.get("revoked_at"):
            return 0
        ordinal = link.get("queued_observation_seq", 0)
        if not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("invalid retry observation ordinal")
        if after is not None:
            ordinal = max(ordinal, after) + 1
            link["queued_observation_seq"] = ordinal
        return ordinal

    try:
        return (observe(peer_links.read_registry(root)) if after is None
                else peer_links.update_registry(root, observe))
    except (ValueError, peer_links.PeerLinkError):
        log.warning("could not coordinate a retry observation for link %s", lid, exc_info=True)
        return 0


def _observe_pass(root: pathlib.Path, lid: str, letter_id: str, initial: dict, *, until: float,
                  requested_retry: int | None = None) -> dict:
    from ouroboros import peer_links

    while True:
        try:
            row = queued_message(root, lid, letter_id)
            if row is None:
                break  # the server removed the intent after recording the outcome
            outcome = row.get("last_outcome")
            # A pass can finish after retry was requested without having seen its ordinal.
            observed = row.get("observed_retry")
            if outcome and (not row.get("pending", True) or requested_retry is None or (
                    requested_retry > 0 and isinstance(observed, int) and observed >= requested_retry)):
                return outcome if not row.get("pending", True) else _pending_outcome(outcome)
            left = until - time.monotonic()
            if left <= 0:
                return _pending_outcome()
        except peer_links.PeerLinkError:
            return {"state": "pending", "code": "delivery_pending",
                    "message": "the server accepted the letter, but its current queue record cannot be read"}
        time.sleep(min(0.05, left))
    place_id, account_id = initial["place_id"], initial["account_id"]
    outcome = _observe_terminal(root, lid, account_id, letter_id)
    if outcome is not None and outcome["state"] != "uncertain":
        return outcome
    key = _queued_conversation_key(root, place_id=place_id, link_id=lid, account_id=account_id)
    if key is not None:
        for identity in _hold_ids(letter_id, initial).values():
            if held := _notice_outcome(root, key, identity):
                return held
    return outcome or {"state": "unknown", "code": "delivery_outcome_unknown",
                       "message": "the queued intent is gone and this call found no recorded outcome"}


def submit_tool_message(root: Any, lid: str, letter_id: str, *, place_id: str, account_id: str, text: str,
                        reply_to: str, origin: Mapping[str, Any], approved_basis: str, coverage_kind: str,
                        root_task_id: str, retry: bool, control_required: bool = False,
                        retry_envelope: Mapping[str, Any] | None = None) -> dict:
    """Enqueue, then observe: a worker can never run the physical submission rail."""
    from ouroboros import peer_links, peer_replies

    until = time.monotonic() + _OUTCOME_WAIT_SEC
    root = pathlib.Path(root)
    earlier, states, sending, error = peer_replies.earlier_state(root, lid, account_id, letter_id)
    if earlier in {"delivered", "failed"}:
        return peer_replies.reconcile_confirmed(root, lid, account_id, letter_id, earlier=earlier, states=states,
                                                record=sending, text=text, reply_to=reply_to, origin=dict(origin))
    try:
        prior = queued_message(root, lid, letter_id)
        if prior is None and retry_envelope and retry_envelope.get("pending", False):
            return _observe_pass(root, lid, letter_id, retry_envelope, until=until)
        if retry and prior is not None and prior.get("pending", True):
            observed = prior.get("observed_retry")
            requested_retry = _retry_observation(root, lid, after=observed if isinstance(observed, int) else 0)
            trigger_link_flush(root, lid)
            return _observe_pass(root, lid, letter_id, prior, until=until if requested_retry else time.monotonic(),
                                 requested_retry=requested_retry)
        if prior is None and earlier == "unknown":
            return {"state": "unknown", "code": error, "message": "earlier attempts cannot be read now"}
        # First-stage compatibility: repeating identical URL words from the same task is the same ID.
        parked_url = prior is not None and prior.get("transport") == "url" and not prior.get("pending", True)
        repeat = parked_url and (prior.get("last_outcome") or {}).get("state") in {"uncertain", "retry"}
        if prior is None or (parked_url and (retry or repeat)):
            _, refusal = peer_replies.resolve_outgoing(root, place_id, lid, expected_account_id=account_id)
            if refusal is not None:
                if earlier == "uncertain" and not parked_url:
                    return peer_replies.kept_outcome(earlier, refusal[1])
                message = peer_replies.kept_outcome(earlier, refusal[1])["message"] if earlier == "uncertain" else refusal[1]
                return {"state": "held", "code": refusal[0], "message": message}
        row = record_queued_message(root, lid, letter_id, place_id=place_id, account_id=account_id, text=text,
                                    reply_to=reply_to, origin=origin, approved_basis=approved_basis,
                                    coverage_kind=coverage_kind, root_task_id=root_task_id, wake=False,
                                    retry_post=retry or repeat, control_required=control_required,
                                    fresh_approval=not retry)
    except peer_links.PeerLinkError as exc:
        kept = peer_links.retain_unsent_envelope(root, lid, letter_id, place_id=place_id, account_id=account_id, text=text,
                                reply_to=reply_to, origin=dict(origin)) if exc.code == "peer_links_unreadable" else False
        return {"state": "queue_failed", "code": exc.code, "message": str(exc), "envelope_retained": kept}
    if not row.get("pending", True):
        return dict(row.get("last_outcome") or _pending_outcome())
    # Even a server-local caller only observes; an inline wire pass can outlive its tool budget.
    trigger_link_flush(root, lid)
    return _observe_pass(root, lid, letter_id, row, until=until)


def matching_queued(root: Any, link_id: str, *, producer: str, text: str, reply_to: str) -> bool:
    """A matching pending tool letter is handled by the queue pass, never reported under a new output id."""
    from ouroboros import peer_links, peer_replies

    if not producer:
        return False
    try:
        row = queued_message(root, link_id, peer_replies.message_letter_id(link_id, producer, reply_to, text))
        return row is not None and bool(row.get("pending", True))
    except (peer_links.PeerLinkError, OSError):
        return True


def is_parked_queue_name(name: str) -> bool:
    from ouroboros import peer_links

    return bool(name.endswith(".parked") and peer_links._RECEIPT_NAME.fullmatch(name[:-7] + ".json"))


def _retire_parked(root: Any, link_id: str, path: pathlib.Path, task_id: str) -> None:
    from ouroboros import peer_links
    from ouroboros.utils import json_update_lock

    try:
        with json_update_lock(peer_links.registry_path(root)):
            row = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(row, dict) and str((row.get("origin") or {}).get("task_id") or "") == task_id:
                path.with_suffix(".json").unlink(missing_ok=True)
                path.unlink()
    except (KeyError, OSError, ValueError, peer_links.PeerLinkError):
        return


def retire_parked_messages(root: Any, task_id: str) -> None:
    """No active pass reads these retry-only rows; retire this producer's at its proved end."""
    from ouroboros import peer_links

    for lid in peer_links.read_registry(root)["links"]:
        for path in _queued_dir(root, lid).glob("*.parked"):
            if is_parked_queue_name(path.name):
                _retire_parked(root, lid, path, task_id)


def _active_queue_entry(path: pathlib.Path) -> bool:
    from ouroboros import peer_links

    return bool(peer_links._RECEIPT_NAME.fullmatch(path.name) and not path.with_suffix(".parked").exists())


def _dir_has_receipt_files(folder: pathlib.Path, *, queue: bool = False, if_unreadable: bool = False) -> bool:
    from ouroboros import peer_links

    try:
        with os.scandir(folder) as entries:
            if queue:
                return any(_active_queue_entry(pathlib.Path(entry.path)) for entry in entries)
            return any(peer_links._RECEIPT_NAME.fullmatch(entry.name) for entry in entries)
    except FileNotFoundError:
        return False
    except OSError:
        return if_unreadable


def _has_pending_files_for_link(root: pathlib.Path, link_id: str, *, if_unreadable: bool = False) -> bool:
    from ouroboros import peer_links

    base = peer_links.token_path(root, link_id).parent
    return (_dir_has_receipt_files(base / "queued", queue=True, if_unreadable=if_unreadable)
            or _dir_has_receipt_files(base / "owed", if_unreadable=if_unreadable))


def wake_for_task_control(root: Any, *, task_id: str = "") -> None:
    """Wake links with queued messages after an applicable task control changes."""
    from ouroboros import peer_links

    if in_worker_process():
        _emit_worker_link_wake(root)
        return
    try:
        for lid in peer_links.read_registry(root)["links"]:
            if _dir_has_receipt_files(_queued_dir(root, lid), queue=True):
                trigger_link_flush(root, lid)
    except (peer_links.PeerLinkError, OSError):
        log.debug("peer wake after control %s failed", task_id, exc_info=True)


def _wait_for_deadline(wake: threading.Event, deadline_at: float) -> None:
    remaining = max(0.0, deadline_at - time.monotonic())
    wake.wait(timeout=min(remaining, threading.TIMEOUT_MAX))


def deliver_until_idle(run_pass: Callable[[], float | None], wake: threading.Event,
                       retire: Callable[[], bool] = lambda: True, *,
                       still_pending: Callable[[], bool] = lambda: True, first_deadline: float | None = None) -> None:
    """An intervening wake cannot discard the link's promised next inspection."""
    wake_at = first_deadline
    while True:
        if wake_at is not None and time.monotonic() >= wake_at:
            wake_at = None  # this pass is the promised inspection
        wake.clear()  # anything already queued at this point is visible to the fresh pass
        if (wait := run_pass()) is not None:
            wake_at = min(wake_at, wait) if wake_at is not None else wait
        elif wake_at is not None and not still_pending():
            wake_at = None
        if wake_at is not None:
            _wait_for_deadline(wake, wake_at)
        elif not wake.is_set() and retire():
            return


def outgoing_switch_hold(root: Any) -> tuple[str, str] | None:
    from ouroboros.place_projection import switch_state

    switch = switch_state(root)
    if switch == "panic":
        return "panic_hold", "a Panic awaits my human's return here"
    if switch not in {"on", "off"}:
        return "switch_unreadable", "the consciousness switch cannot be read now"
    return None


def _delivery_pass(root: pathlib.Path, lid: str, *, results: dict | None = None) -> float | None:
    from ouroboros import peer_links, peer_replies
    from ouroboros.install_identity import acting_as

    with acting_as(root):
        observed_retry = _retry_observation(root, lid)
        hold = outgoing_switch_hold(root)
        queued = queued_messages(root, lid)
        refs = peer_links.owed_refs(root, lid) if hold is None or hold[0] == "switch_unreadable" else ()
        waits = []
        for _, row in queued:
            try:
                letter_id = str(row["letter_id"])
                outcome, wait = _flush_one_queued(root, lid, row, observed_retry=observed_retry, switch_hold=hold)
                if results is not None:
                    results[letter_id] = outcome
                if wait is not None:
                    waits.append(time.monotonic() + wait)
            except Exception:
                log.warning("could not deliver queued peer letter %s", row.get("letter_id"), exc_info=True)
        for ref_link, ref in refs:
            if ref_link != lid:
                continue
            try:
                if hold is not None:
                    waits.append(time.monotonic() + _SWITCH_RECHECK_SEC)
                    break
                _done, wait = peer_replies._one_pass(root, lid, ref)
                if wait is not None:
                    waits.append(time.monotonic() + wait)
            except Exception:
                log.warning("could not deliver owed peer reply %s", ref, exc_info=True)
        return min(waits, default=None)


def _run_claimed_flush(root: pathlib.Path, lid: str, key: tuple[str, str], state: dict[str, Any], *,
                       first_pass_results: dict | None = None) -> None:
    wake = state["wake_event"]

    def retire() -> bool:
        with _FLUSHING_LOCK:
            if wake.is_set():
                return False
            if _FLUSHING_LINKS.get(key) is state:
                _FLUSHING_LINKS.pop(key)
            return True

    def run_pass(*, capture: dict | None = None) -> float | None:
        try:
            return _delivery_pass(root, lid, results=capture)
        except Exception:
            log.warning("peer delivery pass failed for link %s", lid, exc_info=True)
            return None

    def still_pending() -> bool:
        return _has_pending_files_for_link(root, lid, if_unreadable=True)

    if first_pass_results is not None:
        wake.clear()
        wait = run_pass(capture=first_pass_results)
        if wait is None and retire():
            return
        # Only the first pass is inline: a delayed letter cannot block the originating call or other links.
        def resume() -> None:
            if wait is not None and not wake.is_set():
                _wait_for_deadline(wake, wait)
            deliver_until_idle(run_pass, wake, retire, still_pending=still_pending, first_deadline=wait)

        _start_flush_thread(key, state, resume)
    else:
        deliver_until_idle(run_pass, wake, retire, still_pending=still_pending)


def _start_flush_thread(key: tuple[str, str], state: dict[str, Any], call: Callable[[], None]) -> None:
    try:
        threading.Thread(target=call, name=f"peer-reply-link-{key[1][:16]}", daemon=True).start()
    except BaseException:
        with _FLUSHING_LOCK:
            if _FLUSHING_LINKS.get(key) is state:
                _FLUSHING_LINKS.pop(key)
        raise


def _claim_link(root: pathlib.Path, lid: str) -> tuple[tuple[str, str], dict | None]:
    key = (_canonical_root(root), lid)
    with _FLUSHING_LOCK:
        if existing := _FLUSHING_LINKS.get(key):
            existing["wake_event"].set()
            return key, None
    if not _has_pending_files_for_link(root, lid):
        return key, None
    with _FLUSHING_LOCK:
        if existing := _FLUSHING_LINKS.get(key):
            existing["wake_event"].set()
            return key, None
        state = {"wake_event": threading.Event()}
        _FLUSHING_LINKS[key] = state
    return key, state


def _inline_first_pass(root: pathlib.Path, lid: str) -> dict:
    key, state = _claim_link(root, lid)
    results: dict = {}
    if state is not None:
        _run_claimed_flush(root, lid, key, state, first_pass_results=results)
    return results


def flush_link_sync(root: Any, link_id: str) -> None:
    """Run the first pass inline; subsequent passes and waits belong to the same link loop."""
    if in_worker_process():
        _emit_worker_link_wake(root, link_id)
        return
    _inline_first_pass(pathlib.Path(root), link_id)


def flush_queued_messages(root: Any, link_id: str) -> None:
    flush_link_sync(root, link_id)


def trigger_link_flush(root: Any, link_id: str, *, external: bool = True) -> None:
    """Coalesce all producers, announces and control wakes into one server loop."""
    if in_worker_process():
        _emit_worker_link_wake(root, link_id)
        return
    root = pathlib.Path(root)
    key, state = _claim_link(root, link_id)
    if state is not None:
        _start_flush_thread(key, state, lambda: _run_claimed_flush(root, link_id, key, state))


def note_verified_exchange(root: Any, link_id: str) -> None:
    """A confirmed POST is progress, including inside the current pass."""
    if link_id:
        trigger_link_flush(root, link_id)


def accept_verified_cancel(root: Any, engine_ref: Any, receipt: Any, *, require_verdict: bool) -> tuple[bool, str]:
    """A typed peer cancel receipt is a successful exchange; a refusal does not wake delivery."""
    accepted = (receipt or {}).get("accepted")
    if require_verdict and not isinstance(accepted, bool):
        return False, "The engine returned no cancellation verdict."
    if accepted is True and isinstance(engine_ref, Mapping) and engine_ref.get("kind") == "peer":
        note_verified_exchange(root, str(engine_ref.get("link_id") or ""))
    return bool(accepted), ""
