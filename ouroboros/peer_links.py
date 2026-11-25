"""Links between two Ouroboros installations of one human: registry, keys and introduction.

A link is directional at both ends: each side keeps its own secret (``state/peer_links/<link_id>/token.json``,
written owner-only) to sign the letters it sends, and the sha256 of the other side's secret to recognise the
letters it receives. An introduction carries hashes only: the inviting side's in the invitation my human
pastes, the accepting side's in the ``introduce`` call that one-time invitation authorises. The registry
(``state/peer_links.json``) holds no secret. A key written before its link exists belongs to a dated row (an
invitation here, or an acceptance under way), and leaves with that row at its deadline unless a link took it;
one attempt holds an invitation or an acceptance at a time and takes back only what it made. The answers a link
delivered here are receipts beside its key (``replies/``), read by address only and removed with the link.

The other installation is a place (``connector = provider = "ouroboros"``, ``account_id`` its install id,
``conversation_id`` the link id). Its two typed facts, ``mine`` and ``may_start``, are my human's choices at
their own door (``places`` ``set_peer``); a place on pause pauses the link. A new link starts paused with its
six-digit code unchecked: only my human's ``confirm`` on that side, after comparing the codes both installations
show, starts its place, so an invitation seen in the channel that carried it starts no work; declining the check
is an ordinary ``revoke``. The install id moves only with the data directory's path (``install_identity``): a
copy kept at the same path is indistinguishable from its original, which revocation and a new introduction close.
"""
from __future__ import annotations

import base64
import contextlib
import datetime as _dt
import hashlib
import hmac
import ipaddress
import json
import os
import pathlib
import re
import secrets
import urllib.parse
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from ouroboros.utils import atomic_write_json, update_json_locked, utc_now_iso, write_owner_only_bytes

PROTOCOL = 1
CONNECTOR = "ouroboros"
INVITE_PREFIX = "ouroboros-invite:1:"
INVITE_TTL_SEC = 15 * 60
LINK_HEADER, INSTALL_HEADER, PROTOCOL_HEADER = "X-Ouroboros-Link", "X-Ouroboros-Install", "X-Ouroboros-Protocol"
INVITE_HEADER = "X-Ouroboros-Invite"
INTRODUCE_TIMEOUT_SEC = 20.0
_EMPTY_SHA = hashlib.sha256(b"").hexdigest()
_SECTIONS = ("links", "invites", "accepting")  # accepting: this side's acceptances under way, by link id
_RECEIPT_NAME = re.compile(r"[0-9a-f]{64}\.json")


class PeerLinkError(ValueError):
    """A typed refusal: ``code`` is the wire refusal code, ``status`` its HTTP status."""

    def __init__(self, code: str, message: str = "", status: int = 400) -> None:
        super().__init__(message or code)
        self.code, self.status = code, status


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def registry_path(root: Any) -> pathlib.Path:
    return pathlib.Path(root) / "state" / "peer_links.json"


def token_path(root: Any, link_id: str) -> pathlib.Path:
    return pathlib.Path(root) / "state" / "peer_links" / _valid_link_id(link_id) / "token.json"


def _valid_link_id(link_id: Any) -> str:
    text = str(link_id or "")
    if not (8 <= len(text) <= 64) or not all(ch.isalnum() or ch in "-_" for ch in text):
        raise PeerLinkError("link_unknown", "link id is malformed", 401)
    return text


def _document(value: Any) -> dict:
    if value is None or value == {}:  # an absent store (update_json_locked reads it as {})
        return {"schema": 1, "revision": 0, **{key: {} for key in _SECTIONS}}
    if (not isinstance(value, dict) or value.get("schema") != 1
            or not all(isinstance(value.get(key, {}), dict) for key in _SECTIONS)):
        raise PeerLinkError("peer_links_unreadable", "state/peer_links.json needs repair", 503)
    return {"revision": 0, **{key: {} for key in _SECTIONS}, **value}


def read_registry(root: Any) -> dict:
    path = registry_path(root)
    if not path.exists():
        return _document(None)
    try:
        return _document(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise PeerLinkError("peer_links_unreadable", f"state/peer_links.json needs repair: {exc}", 503) from exc


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _deadline(seconds: float) -> str:
    return (_dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=seconds)).isoformat()


def _live(row: Any) -> bool:
    """A dated row (an invitation, an acceptance under way) authorises only before its deadline."""
    return isinstance(row, dict) and str(row.get("expires_at") or "") > _now()


def _retire_expired(root: Any, document: dict) -> None:
    """Invitations and acceptances under way end at their deadline and authorise nothing after it. Their attempt's
    files leave with them (``_drop_attempt_files``) unless a link took the key; a row whose files could not be
    confirmed gone stays, expired and authorising nothing, until a later write of this registry confirms it."""
    for section in ("invites", "accepting"):
        kept = {}
        for key, row in document[section].items():
            if not isinstance(row, dict):
                continue
            link_id, attempt = str(row.get("link_id") or ""), str(row.get("attempt") or "")
            key_sha256 = str(row.get("key_sha256") or row.get("own_key_sha256") or "")
            if _live(row) or (link_id and link_id not in document["links"]
                              and not _drop_attempt_files(root, link_id, attempt, key_sha256)):
                kept[key] = row
        document[section] = kept


def update_registry(root: Any, mutate: Any) -> Any:
    """One locked read-modify-write; ``mutate(document)`` edits in place and returns its result. Expired rows
    leave first (``_retire_expired``). A store that cannot be read or written is ``peer_links_unreadable`` (503)."""
    outcome: list = []

    def apply(current: Any) -> dict:
        document = _document(current)
        _retire_expired(root, document)
        outcome.append(mutate(document))
        document["revision"] = int(document.get("revision") or 0) + 1
        return document

    try:
        update_json_locked(registry_path(root), apply, strict_existing_dict=True)
    except PeerLinkError:
        raise
    except (OSError, ValueError) as exc:  # a malformed store, a held lock (TimeoutError) or a failed write
        raise PeerLinkError("peer_links_unreadable", f"state/peer_links.json could not be updated: {exc}",
                            503) from exc
    return outcome[0] if outcome else None


def _new_secret() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")


def _write_secret(root: Any, link_id: str, secret: str, attempt: str = "") -> None:
    """The attempt's key, landing through ``.token.json.tmp.<attempt>``: its dated row finds exactly that file after
    a stop between the two (``_drop_attempt_files``). Without an attempt the temporary's name is unique per call."""
    write_owner_only_bytes(token_path(root, link_id), json.dumps({"token": secret}).encode("utf-8"), landing=attempt)


def _drop_attempt_files(root: Any, link_id: str, attempt: str, key_sha256: str) -> bool:
    """Remove what one attempt wrote for ``link_id``'s key and nothing else: its landing temporary by its exact
    name, and the key file only while it holds this attempt's key (``key_sha256``); another attempt's files and a
    link's key stay. True when nothing of this attempt is left here; False while a file of it could not be
    removed (or a key it may have written cannot be read), so its row keeps custody."""
    try:
        key = token_path(root, link_id)
    except PeerLinkError:  # a malformed id names no file of ours
        return True
    try:
        if attempt:
            key.with_name(f".token.json.tmp.{attempt}").unlink(missing_ok=True)
        try:
            held = json.loads(key.read_text(encoding="utf-8"))
        except FileNotFoundError:
            held = None
        except ValueError:
            held = None  # not a whole key of anyone's: never this attempt's completed write
        if isinstance(held, dict) and key_sha256 and _sha256(str(held.get("token") or "")) == key_sha256:
            key.unlink()
    except OSError:
        return False
    with contextlib.suppress(OSError):
        key.parent.rmdir()  # only when nothing else is there
    return True


def _read_secret(root: Any, link_id: str) -> str:
    try:
        return str(json.loads(token_path(root, link_id).read_text(encoding="utf-8")).get("token") or "")
    except (OSError, ValueError, AttributeError):
        return ""


def _drop_secret(root: Any, link_id: str) -> None:
    path = token_path(root, link_id)
    path.unlink(missing_ok=True)
    try:
        path.parent.rmdir()
    except OSError:
        pass


def _key_named(name: str) -> bool:
    name = name.lower()
    return name == "token.json" or name.startswith(".token.json.tmp.")


def may_be_secret_file(entry: Any) -> bool:
    """Whether a folder's own entry can be a link's key at all, read from the entry alone: it is named like the key
    or its landing temporary, or it is a link, which may point at one. A listing resolves only such a candidate
    (``is_secret_file``; one the resolver fails on, as a looping link before Python 3.13, stays unclassified and
    unlisted); every other entry is no key under the name its folder gives it."""
    entry = pathlib.Path(entry)
    return _key_named(entry.name) or entry.is_symlink()


def is_secret_file(path: Any, data_root: Any) -> bool:
    """Whether ``path`` is a link's sending key, or the temporary its owner-only write leaves beside it while
    landing (``utils.write_owner_only_bytes``): what Files never lists, reads, serves or writes. The folder is
    compared without case and by identity, as a case-insensitive disk opens ``PEER_LINKS/x/TOKEN.JSON``."""
    target = pathlib.Path(path).resolve(strict=False)
    if not _key_named(target.name):
        return False
    folder, parent = pathlib.Path(data_root).resolve(strict=False) / "state" / "peer_links", target.parent.parent
    if str(parent).lower() == str(folder).lower():
        return True
    try:
        return parent.exists() and folder.exists() and os.path.samefile(parent, folder)
    except OSError:
        return False


def _receipt_path(root: Any, link_id: str, letter_id: str) -> pathlib.Path:
    """Where this link's answer letter ``letter_id`` is recorded: named by its id's digest, so two ids differing
    only in case stay two receipts on a case-insensitive disk."""
    return token_path(root, link_id).parent / "replies" / f"{_sha256(letter_id)}.json"


def reply_receipt(root: Any, link_id: str, letter_id: str) -> dict | None:
    """The answer this link sent under ``letter_id`` (``target_task_id``, ``question_id``, ``content_sha256``,
    ``msg_id``, ``delivered``, ``at``; ``record_reply``), None when it sent none; one that cannot be read is
    ``peer_links_unreadable`` (503), never none."""
    path = _receipt_path(root, link_id, letter_id)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise PeerLinkError("peer_links_unreadable", f"the answer receipt {path.name} needs repair: {exc}",
                            503) from exc
    if not isinstance(value, dict) or value.get("letter_id") != letter_id:
        raise PeerLinkError("peer_links_unreadable", f"the answer receipt {path.name} needs repair", 503)
    return value


def record_reply(root: Any, link_id: str, letter_id: str, *, target_task_id: str, question_id: str,
                 content_sha256: str, msg_id: str, delivered: bool = False) -> None:
    """Record ``letter_id`` on this link as one answer (``gateway.peer_letters_api``): reserved (``delivered``
    false) before anything of it lands, so from here that id is this answer's alone, for a new letter as for another
    answer; then ``delivered`` once its words are proven in the asking root's mailbox as the place event ``msg_id``.
    A reservation proves no delivery. ``at`` is when this record was written. A receipt that cannot be written is
    ``peer_links_unreadable`` (503): the sender sends the same bytes again."""
    path = _receipt_path(root, link_id, letter_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"letter_id": letter_id, "target_task_id": target_task_id,
                                 "question_id": question_id, "content_sha256": content_sha256, "msg_id": msg_id,
                                 "delivered": bool(delivered), "at": utc_now_iso()}, fsync=True)
    except OSError as exc:
        raise PeerLinkError("peer_links_unreadable", f"the answer receipt could not be recorded now: "
                            f"{type(exc).__name__}", 503) from exc


def _message_receipt_path(root: Any, link_id: str, letter_id: str) -> pathlib.Path:
    """Where this link's standalone peer message ``letter_id`` is recorded, beside ``replies``."""
    return token_path(root, link_id).parent / "messages" / f"{_sha256(letter_id)}.json"


def message_receipt(root: Any, link_id: str, letter_id: str) -> dict | None:
    """The standalone message this link received under ``letter_id``, None when none; unreadable is 503."""
    path = _message_receipt_path(root, link_id, letter_id)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise PeerLinkError("peer_links_unreadable", f"the message receipt {path.name} needs repair: {exc}",
                            503) from exc
    if not isinstance(value, dict) or value.get("letter_id") != letter_id:
        raise PeerLinkError("peer_links_unreadable", f"the message receipt {path.name} needs repair", 503)
    return value


def record_message(root: Any, link_id: str, letter_id: str, *, task_id: str, chat_id: int,
                   content_sha256: str) -> None:
    """Record ``letter_id`` on this link as a received standalone message (one id, one outcome across letters)."""
    path = _message_receipt_path(root, link_id, letter_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"letter_id": letter_id, "task_id": task_id, "chat_id": int(chat_id),
                                 "content_sha256": content_sha256, "at": utc_now_iso()}, fsync=True)
    except OSError as exc:
        raise PeerLinkError("peer_links_unreadable", f"the message receipt could not be recorded now: "
                            f"{type(exc).__name__}", 503) from exc


def _owed_path(root: Any, link_id: str, work_ref: str) -> pathlib.Path:
    return token_path(root, link_id).parent / "owed" / f"{_sha256(work_ref)}.json"


def record_owed(root: Any, link_id: str, work_ref: str, *, place_id: str,
                approved: Mapping[str, str] | None = None) -> None:
    """This link is owed what ``work_ref`` answers: a turn a message of the link started, or work such a turn handed
    off. Written before that work can end and settled once its outputs are (``settle_owed``), so a restart resumes
    only these addressed records (``peer_replies.resume_owed``), as an outside connector keeps its work
    refs. ``approved``: the sha256 of words the work's selection let go there, each with the rules they were read
    under (``peer_letters.note_reply_approved``), kept beside what the record already holds; a newer reading of the
    same words replaces its older one. The read and the replacement are one step under the registry's writer lock, so
    a concurrent writer (an address without ``approved``) never drops an approval it did not read. A record that cannot
    be written is ``peer_links_unreadable``."""
    from ouroboros.utils import json_update_lock

    path = _owed_path(root, link_id, work_ref)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with json_update_lock(registry_path(root)):
            old = owed_record(root, link_id, work_ref) or {}
            kept = old.get("approved") if isinstance(old.get("approved"), dict) else {}
            atomic_write_json(path, {"work_ref": work_ref, "place_id": place_id, "at": old.get("at") or utc_now_iso(),
                                     "approved": {**kept, **dict(approved or {})}}, fsync=True)
    except OSError as exc:  # a held lock (TimeoutError) included
        raise PeerLinkError("peer_links_unreadable", f"the owed reply could not be recorded now: "
                            f"{type(exc).__name__}", 503) from exc


def owed_record(root: Any, link_id: str, work_ref: str) -> dict | None:
    """``work_ref``'s owed record on ``link_id`` as written, else None; one that cannot be read is
    ``peer_links_unreadable`` (503): unknown, never an absent approval."""
    path = _owed_path(root, link_id, work_ref)
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise PeerLinkError("peer_links_unreadable", f"the owed record {path.name} cannot be read now: "
                            f"{type(exc).__name__}", 503) from exc
    if not isinstance(row, dict) or row.get("work_ref") != work_ref:
        raise PeerLinkError("peer_links_unreadable", f"the owed record {path.name} needs repair", 503)
    return row


def owed_version(root: Any, link_id: str, work_ref: str) -> tuple | None:
    """The exact written state of ``work_ref``'s owed record (its file's identity and bytes' sha256), None when there
    is none; one that cannot be read is ``peer_links_unreadable`` (503): unknown, never absent. Every write replaces
    the file (``record_owed``), so any rewrite since a reading is a change of this value."""
    path = _owed_path(root, link_id, work_ref)
    try:
        before, data = path.stat(), path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PeerLinkError("peer_links_unreadable", f"the owed record {path.name} cannot be read now: "
                            f"{type(exc).__name__}", 503) from exc
    return before.st_ino, before.st_mtime_ns, before.st_size, hashlib.sha256(data).hexdigest()


def settle_owed(root: Any, link_id: str, work_ref: str, *, seen: tuple | None, unowed: Any = None) -> bool:
    """``work_ref``'s owed record leaves only in the exact state its remover read before deciding (``seen``, an
    ``owed_version``) and while ``unowed()`` still reads true, both under the registry's writer lock ``record_owed``
    writes under: a record a newer attempt or approval rewrote since, even one that ended meanwhile, stays. Whether
    it left."""
    from ouroboros.utils import json_update_lock

    with json_update_lock(registry_path(root)):
        if owed_version(root, link_id, work_ref) != seen or (unowed is not None and not unowed()):
            return False
        _owed_path(root, link_id, work_ref).unlink(missing_ok=True)
        return True


def owed_refs(root: Any, link_id: str = "") -> list[tuple[str, str]]:
    """``(link_id, work_ref)`` of every reply still owed on a link this registry names (only ``link_id``'s folder when
    named), oldest first (an unreadable registry raises: unknown, never none); a record that cannot be read is left
    for a later start."""
    owed = []
    links = read_registry(root)["links"]
    for link_id in ([link_id] if link_id in links else []) if link_id else links:
        folder = token_path(root, link_id).parent / "owed"
        for entry in (folder.iterdir() if folder.is_dir() else ()):
            try:
                row = json.loads(entry.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if _RECEIPT_NAME.fullmatch(entry.name) and isinstance(row, dict) and row.get("work_ref"):
                owed.append((str(row.get("at") or ""), link_id, str(row["work_ref"])))
    return [(link_id, work_ref) for _at, link_id, work_ref in sorted(owed)]


def _sending_path(root: Any, link_id: str, letter_id: str) -> pathlib.Path:
    return token_path(root, link_id).parent / "sent" / f"{_sha256(letter_id)}.json"


def sending_record(root: Any, link_id: str, letter_id: str) -> dict | None:
    """My message ``letter_id`` on ``link_id`` while it is unsettled (``record_sending``), else None; a record that
    cannot be read is ``peer_links_unreadable`` (503): unknown, never "nothing of it left"."""
    path = _sending_path(root, link_id, letter_id)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise PeerLinkError("peer_links_unreadable", f"the sending record {path.name} needs repair: {exc}",
                            503) from exc
    if not isinstance(value, dict) or value.get("letter_id") != letter_id:
        raise PeerLinkError("peer_links_unreadable", f"the sending record {path.name} needs repair", 503)
    return value


def record_sending(root: Any, link_id: str, letter_id: str, *, account_id: str, text: str, reply_to: str,
                   origin: Mapping[str, str], place_id: str = "") -> int:
    """Before a call's first physical send of my message ``letter_id`` (``peer_replies.send_peer_message``):
    from here an attempt of it may arrive, so no later refusal reads as its end. Kept until its delivered or failed
    receipt lands (``settle_sending``), with the words and address a start needs to say an interrupted attempt in its
    room (``resume_owed``) and ``passes``, the calls that sent it; returns that count. A record that cannot be written
    is ``peer_links_unreadable``, and nothing is sent then (read and replaced under the registry's writer lock)."""
    from ouroboros.utils import json_update_lock

    path = _sending_path(root, link_id, letter_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with json_update_lock(registry_path(root)):
            old = sending_record(root, link_id, letter_id) or {}
            if old:
                if old.get("account_id") and str(old["account_id"]) != str(account_id):
                    raise PeerLinkError("sending_identity_mismatch",
                                        "existing sending record belongs to another recipient", 409)
                if "text" in old and str(old.get("text") or "") != str(text):
                    raise PeerLinkError("sending_identity_mismatch",
                                        "existing sending record carries different text", 409)
            passes = int(old.get("passes") or 0) + 1
            kept_account = str(old["account_id"]) if old.get("account_id") else str(account_id)
            kept_text = str(old["text"]) if ("text" in old and old["text"] is not None) else str(text)
            kept_reply_to = str(old["reply_to"]) if ("reply_to" in old and old["reply_to"] is not None) else str(reply_to)
            kept_origin = (dict(old["origin"]) if isinstance(old.get("origin"), Mapping) and old.get("origin")
                           else dict(origin))
            preserved = {k: old[k] for k in ("confirmed_state", "confirmed_at", "confirmed_code", "confirmed_message")
                         if old.get(k)}
            atomic_write_json(path, {"letter_id": letter_id, "account_id": kept_account,
                                     "place_id": str(old.get("place_id") or place_id or ""),
                                     "text": kept_text, "reply_to": kept_reply_to, "origin": kept_origin,
                                     "first_at": old.get("first_at") or utc_now_iso(), "at": utc_now_iso(),
                                     "passes": passes, **preserved}, fsync=True)
    except OSError as exc:  # a held lock (TimeoutError) included
        raise PeerLinkError("peer_links_unreadable", f"the sending record could not be written now: "
                            f"{type(exc).__name__}", 503) from exc
    return passes


def confirm_sending(root: Any, link_id: str, letter_id: str, *, state: str, account_id: str = "", place_id: str = "",
                    text: str = "", reply_to: str = "", origin: Mapping[str, str] | None = None,
                    code: str = "", message: str = "") -> dict:
    """Record a confirmed delivery or definite first-send refusal on ``sent/`` when its canonical receipt could not
    be written yet, so later retries, pauses, revocations or restarts finish recording that confirmed fact instead of
    downgrading it to ``uncertain`` or re-sending. Never downgrades ``delivered`` to ``failed``."""
    from ouroboros.utils import json_update_lock

    if state not in {"delivered", "failed"}:
        raise ValueError("confirm_sending state must be delivered or failed")
    path = _sending_path(root, link_id, letter_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with json_update_lock(registry_path(root)):
            try:
                old = sending_record(root, link_id, letter_id) or {}
            except PeerLinkError:
                old = {}
            kept_state = "delivered" if old.get("confirmed_state") == "delivered" else state
            now = utc_now_iso()
            confirmed_at = (old.get("confirmed_at")
                            if kept_state == old.get("confirmed_state") and old.get("confirmed_at") else now)
            kept_origin = (dict(old["origin"]) if isinstance(old.get("origin"), Mapping) and old.get("origin")
                           else dict(origin) if isinstance(origin, Mapping) and origin else {})
            kept_reply_to = (str(old["reply_to"]) if ("reply_to" in old and old["reply_to"] is not None)
                             else str(reply_to or ""))
            kept_text = str(old["text"]) if ("text" in old and old["text"] is not None) else str(text)
            updated: dict[str, Any] = {
                "letter_id": letter_id,
                "account_id": str(old.get("account_id") or account_id or ""),
                "place_id": str(old.get("place_id") or place_id or ""),
                "text": kept_text,
                "reply_to": kept_reply_to,
                "origin": kept_origin,
                "first_at": old.get("first_at") or now,
                "at": now,
                "passes": max(int(old.get("passes") or 0), 1),
                "confirmed_state": kept_state,
                "confirmed_at": confirmed_at,
            }
            if kept_state == "failed":
                if kept_code := str(code or old.get("confirmed_code") or ""):
                    updated["confirmed_code"] = kept_code
                if kept_msg := str(message or old.get("confirmed_message") or ""):
                    updated["confirmed_message"] = kept_msg
            atomic_write_json(path, updated, fsync=True)
    except OSError as exc:
        raise PeerLinkError("peer_links_unreadable", f"the confirmed sending record could not be written now: "
                            f"{type(exc).__name__}", 503) from exc
    return updated


def rollback_sending(root: Any, link_id: str, letter_id: str, *, expected_passes: int) -> bool:
    """Undo a pass recorded by ``record_sending`` when nothing of that call left on the wire: removes the record if it
    was the first pass (``expected_passes <= 1``), else decrements ``passes`` by 1. Never touches a confirmed record
    or a record whose pass count changed."""
    from ouroboros.utils import json_update_lock

    path = _sending_path(root, link_id, letter_id)
    try:
        with json_update_lock(registry_path(root)):
            old = sending_record(root, link_id, letter_id)
            if old is None or old.get("confirmed_state") or int(old.get("passes") or 0) != int(expected_passes):
                return False
            if int(expected_passes) <= 1:
                path.unlink(missing_ok=True)
                return True
            atomic_write_json(path, {**old, "passes": int(expected_passes) - 1, "at": utc_now_iso()}, fsync=True)
            return True
    except OSError as exc:
        raise PeerLinkError("peer_links_unreadable", f"the sending record could not be rolled back now: "
                            f"{type(exc).__name__}", 503) from exc


def settle_sending(root: Any, link_id: str, letter_id: str) -> None:
    _sending_path(root, link_id, letter_id).unlink(missing_ok=True)


def sending_records(root: Any) -> list[tuple[str, dict, bool]]:
    """``(link_id, record, revoked)`` of every message of mine still unsettled on a link this registry names, revoked
    ones included (an unreadable registry raises: unknown, never none); a record that cannot be read is left for a later
    start."""
    found = []
    for link_id, row in read_registry(root)["links"].items():
        folder = token_path(root, link_id).parent / "sent"
        for entry in (folder.iterdir() if folder.is_dir() else ()):
            try:
                record = json.loads(entry.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if _RECEIPT_NAME.fullmatch(entry.name) and isinstance(record, dict) and record.get("letter_id"):
                found.append((link_id, record, bool(isinstance(row, dict) and row.get("revoked_at"))))
    return found


def _drop_link_files(root: Any, link_id: str) -> None:
    """A revoked link's files here: its key, the receipts of its answers and messages and the replies it was owed. My
    messages to it still unsettled (``sent/``) stay: such a record may be the only sign that an attempt arrived, and it
    leaves only once its room's receipt says so (``peer_replies._reconcile_sending``)."""
    for folder in ("replies", "messages", "owed"):
        directory = token_path(root, link_id).parent / folder
        with contextlib.suppress(FileNotFoundError, NotADirectoryError):
            for entry in list(directory.iterdir()):
                if _RECEIPT_NAME.fullmatch(entry.name):
                    entry.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            directory.rmdir()
    _drop_secret(root, link_id)


def link_code(link_id: str, install_a: str, install_b: str, key_a: str, key_b: str) -> str:
    """The six-digit check both sides show after an introduction (A invited, B accepted); no secret enters it."""
    digest = hashlib.sha256("\0".join(("ouroboros-link-code", link_id, install_a, install_b, key_a, key_b))
                            .encode("utf-8")).hexdigest()
    return str(int(digest, 16))[:6]


def _own_install_id(root: Any) -> str:
    """This data root's install id, minted at its first use (the server's own root mints it at boot)."""
    from ouroboros.install_identity import acting_as, initialize_install

    with acting_as(root):
        return str(initialize_install()["id"])


def _current_install_id(root: Any) -> str:
    from ouroboros.install_identity import recorded_id

    return recorded_id(root)


def _text(value: Any, field: str, *, limit: int = 300, required: bool = True) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()) or len(value) > limit:
        raise PeerLinkError("invalid_request", f"{field} must be a string of at most {limit} characters")
    return value.strip()


def _label(value: Any, field: str) -> str:
    """A name for a linked Ouroboros: it becomes the author label of its letters (``recorded_author``)."""
    from ouroboros.dialogue_provenance import AUTHOR_LABEL_MAX

    text = _text(value, field, limit=AUTHOR_LABEL_MAX, required=False)
    return " ".join(text.split())


def _bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise PeerLinkError("invalid_request", f"{field} must be true or false")
    return value


def loopback_url(url: Any) -> bool:
    """An ``http`` address on this computer: this step's links join two installations of one computer only."""
    parts = urllib.parse.urlsplit(str(url or ""))
    if parts.scheme != "http" or not parts.hostname:
        return False
    try:
        return ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:
        return parts.hostname == "localhost"


def _url(value: Any, field: str) -> str:
    text = _text(value, field, limit=500)
    if not loopback_url(text):
        raise PeerLinkError("peer_url_refused", f"{field} must be an http address on this computer (loopback): "
                            "this step links two installations of one computer")
    return text.rstrip("/")


def _connect_peer_place(root: Any, *, peer_install_id: str, link_id: str, label: str, note: str,
                        mine: bool, may_start: bool) -> str:
    """The other installation as a place with my human's two facts (their acts at their door: the accept request,
    or the invitation they created for this link), Paused until they confirm the link's code (``confirm``). Never
    disconnects or overwrites an already-selected place; only an unlinked Paused provisional place with the exact
    same facts from an earlier attempt of this link is reused as-is."""
    from ouroboros.places import (AUTHOR_OWNER, FACT_DIRECT_MESSAGE, FACT_FROM_BOT, STATE_PAUSED,
                                  connect_place, edit_place, place_records)

    target_id = uuid.uuid5(uuid.NAMESPACE_URL, json.dumps((CONNECTOR, CONNECTOR, peer_install_id, link_id, ""))).hex
    chosen_label, wakes, expected_notes = label or "Ouroboros", (FACT_DIRECT_MESSAGE, FACT_FROM_BOT), ((note,) if note else ())
    records = place_records(root)
    if (existing := records.places.get(target_id)) is not None:
        if (link_of_place(root, target_id) is not None or existing.state != STATE_PAUSED
                or existing.label != chosen_label or records.peer(target_id) != {"mine": mine, "may_start": may_start}
                or tuple(n.text for n in existing.notes) != expected_notes):
            raise PeerLinkError("place_exists", "this link's place is connected already", 409)
        return existing.place_id
    place = connect_place(root, connector=CONNECTOR, provider=CONNECTOR, label=chosen_label,
                          account_id=peer_install_id, conversation_id=link_id, scope="conversation",
                          wakes=wakes, state=STATE_PAUSED, first={"note": note} if note else None, author=AUTHOR_OWNER)
    if (place.revision != place.connected_revision or place.label != chosen_label or place.state != STATE_PAUSED
            or tuple(w.fact for w in place.wakes) != wakes or tuple(n.text for n in place.notes) != expected_notes):
        raise PeerLinkError("place_exists", "this link's place is connected already", 409)
    edit_place(root, place.place_id, "set_peer", author=AUTHOR_OWNER, expected_revision=place.revision,
               mine=mine, may_start=may_start)
    return place.place_id


# Why nothing a linked Ouroboros sends is accepted now (``effective_permissions``), in the letter door's own order.
_REFUSED = {"unlinked": "no link names this place", "revoked": "this link was revoked",
            "code_check_pending": "the link's six-digit code is not confirmed on this side yet",
            "paused": "this place is Paused"}
# Why a letter that assigns work is refused while its messages are accepted, in the door's order after them.
_NO_WORK = {"may_not_start": "may_start is no",
            "silent": "this place is Silent here, so work it starts could not answer",
            "panic": "a Panic stopped this installation, and new work waits for my human's manual return"}
# The letter door's typed answer (refusal code, detail) for each of those facts (``gateway.peer_letters_api``).
DOOR_REFUSALS = {
    "code_check_pending": ("link_paused", "code_check_pending"),  # my human has not yet compared the six digits here
    "paused": ("link_paused", "this link is paused here"),
    "not_mine": ("peer_may_not_start", "this installation does not count the sender as its human's Ouroboros"),
    "may_not_start": ("peer_may_not_start", "this link only answers here; it may not start work"),
    "silent": ("link_silent", "this link's place is Silent here: work it would start could not answer"),
    "panic": ("panic_hold", "a Panic stopped this installation; new work waits for its human's manual return"),
}
# What the two choices do not grant, said with them: "mine" names an agent, and a place's own words decide the rest.
_SCOPE = ("These facts name the writer and what it may start here. By themselves they are neither my human's consent "
          "to share with it what my human keeps private, nor their order for what only my human may ask: my human's "
          "notes for this place, its circle's words and their rules say what they allow.")


def effective_permissions(peer: Any, *, state: str, link: Any, switch: str = "") -> dict[str, Any]:
    """What this side accepts from a linked Ouroboros now, apart from my human's stored choices (``peer``): the one
    reading the letter door enforces at each request (``gateway.peer_letters_api._admission``) and the place's card
    states. ``messages``: a message it writes on its own is saved in its place's conversation; ``assignments``: a
    letter that assigns work starts ordinary work; ``reason``: the first fact that refuses what is refused,
    ``unlinked`` | ``revoked`` | ``code_check_pending`` | ``paused`` | ``not_mine`` (all refuse both), then
    ``may_not_start`` | ``silent`` (its answer could not leave) | ``panic`` (``switch``, the consciousness switch's
    one cause, says a Panic holds until my human's return) for assignments only, else ``""``. ``link`` is the link's
    ``revoked_at`` and ``code_check``, None when no link names the place. A choice a refusing fact overrides stays
    stored as my human set it."""
    link, peer = (link if isinstance(link, Mapping) else None), (peer if isinstance(peer, Mapping) else {})
    reason = ("unlinked" if link is None else "revoked" if link.get("revoked_at")
              else "code_check_pending" if link.get("code_check") != "confirmed"
              else "paused" if state == "paused" else "" if peer.get("mine") is True else "not_mine")
    work = (reason or ("may_not_start" if peer.get("may_start") is not True else "silent" if state != "active"
                       else "panic" if switch == "panic" else ""))
    return {"messages": not reason, "assignments": not work, "reason": work}


def place_permissions(root: Any, place: Any, peer: Any, *, switch: str | None = None) -> dict[str, Any]:
    """``effective_permissions`` of one linked place from the stores as they stand (``switch`` read here when
    absent: ``place_projection.switch_state``). A registry that cannot be read is ``unknown`` (both answers None):
    the door answers "not known now" then, never a refusal or a permission."""
    try:
        found = link_of_place(root, place.place_id)
    except PeerLinkError:
        return {"messages": None, "assignments": None, "reason": "unknown"}
    if switch is None:
        from ouroboros.place_projection import switch_state

        switch = switch_state(root)
    link = None if found is None else {"revoked_at": found[1].get("revoked_at"), "code_check": _code_check(found[1])}
    return effective_permissions(peer, state=place.state, link=link, switch=switch)


def peer_facts_line(peer: Any, effect: Mapping[str, Any]) -> str:
    """A linked Ouroboros's facts as the host states them to the mind, to Safety and on its place's card
    (configure_presence, a letter's context, every check of work it occasioned): my human's stored choices
    (``mine``, ``may_start``) as they set them, then what is in effect now (``effective_permissions``), messages and
    assignments apart, so an unmarked or revoked link never reads as one that may still start work. A refusal is
    attributed to the fact that causes it; ``_SCOPE`` says what the two choices do not grant."""
    known = isinstance(peer, Mapping)
    stored = ("none recorded" if not known else "mine: " + ("yes" if peer.get("mine") is True else "no")
              + "; may_start: " + ("yes" if peer.get("may_start") is True else "no"))
    reason = str(effect.get("reason") or "")
    if reason == "unknown":
        now = "unknown: the link registry could not be read, and until it can its letters are answered 'not known now'"
    elif not effect.get("messages"):
        why = _REFUSED.get(reason) or ("my human has not marked it as their own Ouroboros" if known
                                       else "my human recorded no choices for it")
        now = f"nothing it sends is accepted here, because {why}; the stored choices stay as they are"
    else:
        now = ("a message it writes on its own is saved in this place's conversation and answered under the ordinary "
               "place rules, waking nothing while consciousness is off or its allowance is spent; a letter that "
               "assigns work " + ("starts ordinary work here, also while consciousness is off"
                                  if effect.get("assignments") else f"is refused, because {_NO_WORK.get(reason) or reason}"))
    return ("Host facts of this linked Ouroboros (another installation: an agent, never my human). My human's stored "
            f"choices for it, set at their door: {stored}. In effect now: {now}. {_SCOPE}")


def letter_facts_line(root: Any, letter: Any) -> str:
    """``peer_facts_line`` of the link a letter (``metadata.peer_letter``) or a linked Ouroboros's message came
    through, read now from its place; ``""`` for work that is neither, and for a place that is no linked
    Ouroboros's (its connector is another: nobody's typed choices are stated about it). A place no longer connected
    reads as no link; an unreadable place store is said as such, never guessed."""
    place_id = str(letter.get("place_id") or "") if isinstance(letter, Mapping) else ""
    if not place_id or root is None:
        return ""
    from ouroboros.places import place_records

    try:
        records = place_records(root)
    except Exception:  # noqa: BLE001 - an unreadable store is a fact to state, never a guessed choice
        return "Host facts of this linked Ouroboros could not be read now; its choices stay unknown."
    place = records.places.get(place_id)
    if place is None:  # disconnected since: no link names a place that is gone
        return peer_facts_line(None, effective_permissions(None, state="", link=None))
    if place.connector != CONNECTOR:
        return ""
    peer = records.peer(place_id)
    return peer_facts_line(peer, place_permissions(root, place, peer))


def _encode_invite(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return INVITE_PREFIX + base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def parse_invite(text: Any) -> dict:
    if not isinstance(text, str) or not text.strip().startswith(INVITE_PREFIX):
        raise PeerLinkError("invite_invalid", "this is not an Ouroboros invitation")
    body = text.strip()[len(INVITE_PREFIX):]
    try:
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)).decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise PeerLinkError("invite_invalid", "the invitation is damaged") from exc
    keys = {"link_id", "install_id", "url", "label", "key_sha256", "token", "may_start", "expires_at"}
    if not isinstance(payload, dict) or set(payload) != keys:
        raise PeerLinkError("invite_invalid", "the invitation has an unknown shape")
    _valid_link_id(payload["link_id"])
    if str(payload["expires_at"]) <= _dt.datetime.now(_dt.timezone.utc).isoformat():
        raise PeerLinkError("invite_expired", "the invitation has expired; ask for a new one", 410)
    return payload


def create_invite(root: Any, *, own_url: str, label: str = "", own_label: str = "", note: str = "",
                  may_start: bool) -> dict:
    """My human's invitation at A: A's secret is written, the one-time token kept as a hash for 15 minutes.
    ``label`` names the coming Ouroboros here; ``may_start`` and ``note`` are my human's facts for its place."""
    own_url, may_start = _url(own_url, "own_url"), _bool(may_start, "may_start")
    label, own_label = _label(label, "label"), _label(own_label, "own_label")
    note = _text(note, "note", limit=2000, required=False)
    link_id, secret, token = uuid.uuid4().hex, _new_secret(), secrets.token_urlsafe(32)
    install_id, expires_at, key = _own_install_id(root), _deadline(INVITE_TTL_SEC), uuid.uuid4().hex
    row = {"token_sha256": _sha256(token), "link_id": link_id, "expires_at": expires_at, "attempt": key,
           "own_key_sha256": _sha256(secret), "own_install_id": install_id, "label": label, "note": note,
           "may_start": may_start, "created_at": utc_now_iso()}
    # The row first: a key never lands without a dated row that takes it away (``_retire_expired``), and a row
    # whose key did not land authorises nothing (``introduce`` needs the key).
    update_registry(root, lambda document: document["invites"].__setitem__(key, row))
    try:
        _write_secret(root, link_id, secret, key)
    except Exception:
        if _drop_attempt_files(root, link_id, key, row["own_key_sha256"]):  # else the row keeps custody
            with contextlib.suppress(PeerLinkError):
                update_registry(root, lambda document: document["invites"].pop(key, None))
        raise
    invite = _encode_invite({"link_id": link_id, "install_id": install_id, "url": own_url,
                             "label": own_label or "Ouroboros", "key_sha256": row["own_key_sha256"],
                             "token": token, "may_start": may_start, "expires_at": expires_at})
    return {"invite": invite, "link_id": link_id, "expires_at": expires_at}


def preview_invite(text: Any) -> dict:
    payload = parse_invite(text)
    return {"link_id": payload["link_id"], "peer_label": payload["label"], "peer_url": payload["url"],
            "peer_install_id": payload["install_id"], "remote_may_start": payload["may_start"],
            "expires_at": payload["expires_at"]}


def _post_introduce(url: str, token: str, body: Mapping[str, Any]) -> dict:
    """The one request carrying an invitation token: straight to the address in the invitation (checked on this
    computer), never through an environment proxy or a redirect, which would hand the token to someone else."""
    import httpx

    try:
        with httpx.Client(trust_env=False, follow_redirects=False,
                          timeout=httpx.Timeout(INTRODUCE_TIMEOUT_SEC)) as client:
            response = client.post(url + "/api/peer-links/introduce", json=dict(body),
                                   headers={INVITE_HEADER: token, PROTOCOL_HEADER: str(PROTOCOL)})
    except httpx.HTTPError as exc:
        raise PeerLinkError("introduce_unreachable",
                            f"the other Ouroboros did not answer: {type(exc).__name__}", 502) from exc
    if response.status_code != 200:  # a redirect included: it is an answer, never followed
        try:
            refusal = response.json()
        except ValueError:
            refusal = None
        code = str(refusal.get("refusal_code") or refusal.get("error") or "") if isinstance(refusal, dict) else ""
        raise PeerLinkError("introduce_refused", f"the other Ouroboros refused: {code or response.status_code}", 502)
    try:
        return response.json()
    except ValueError as exc:
        raise PeerLinkError("introduce_unreachable", "the other Ouroboros's answer is not JSON", 502) from exc


def accept_invite(root: Any, text: Any, *, own_url: str, label: str = "", own_label: str = "", note: str = "",
                  may_start: bool) -> dict:
    """My human's paste at B: B's secret and its place for A, then A records B through ``introduce``. The link id is
    reserved first (``accepting``), so a second paste of the same invitation is ``invite_used`` before it writes
    anything; a refused or unanswered introduction takes back only this attempt's exact key files and reservation,
    leaving any provisional place Paused and unlinked."""
    payload = parse_invite(text)
    own_url, may_start, peer_url = _url(own_url, "own_url"), _bool(may_start, "may_start"), _url(payload["url"], "url")
    label, own_label = _label(label, "label"), _label(own_label, "own_label")
    note = _text(note, "note", limit=2000, required=False)
    link_id, install_id = payload["link_id"], _own_install_id(root)
    if payload["install_id"] == install_id:
        raise PeerLinkError("same_install", "this invitation comes from this same installation")
    label = label or _label(payload["label"], "the invitation's label") or "Ouroboros"
    attempt, secret = uuid.uuid4().hex, _new_secret()
    key_sha256 = _sha256(secret)

    def reserve(document: dict) -> bool:
        if link_id in document["links"] or link_id in document["accepting"]:
            return False
        document["accepting"][link_id] = {"link_id": link_id, "attempt": attempt, "key_sha256": key_sha256,
                                          "expires_at": _deadline(INVITE_TTL_SEC)}
        return True
    if not update_registry(root, reserve):
        raise PeerLinkError("invite_used", "this invitation was already accepted here, or is being accepted", 409)
    try:
        _write_secret(root, link_id, secret, attempt)
        place_id = _connect_peer_place(root, peer_install_id=payload["install_id"], link_id=link_id,
                                       label=label, note=note, mine=True, may_start=may_start)
        answer = _post_introduce(peer_url, payload["token"], {
            "link_id": link_id, "install_id": install_id, "url": own_url, "label": own_label or "Ouroboros",
            "key_sha256": key_sha256, "may_start": may_start, "protocol": PROTOCOL})
        if (not isinstance(answer, dict) or answer.get("install_id") != payload["install_id"]
                or answer.get("key_sha256") != payload["key_sha256"] or answer.get("link_id") != link_id):
            raise PeerLinkError("introduce_mismatch", "the answer does not match the invitation", 502)
        code = link_code(link_id, payload["install_id"], install_id, payload["key_sha256"], key_sha256)
        row = {"place_id": place_id, "peer_label": label, "peer_url": peer_url,
               "peer_install_id": payload["install_id"], "own_install_id": install_id,
               "inbound_key_sha256": payload["key_sha256"], "key_set_at": utc_now_iso(), "protocol": PROTOCOL,
               "created_at": utc_now_iso(), "code": code, "remote_may_start": answer.get("may_start")}

        def settle(document: dict) -> bool:
            held = document["accepting"].get(link_id)
            if not _live(held) or held.get("attempt") != attempt:
                return False
            del document["accepting"][link_id]
            document["links"][link_id] = row
            return True
        if not update_registry(root, settle):
            raise PeerLinkError("invite_expired", "this acceptance outlived its deadline; ask for a new invitation",
                                410)
    except Exception:
        _release_acceptance(root, link_id, attempt, key_sha256)
        raise
    return link_view(root, link_id)


def _release_acceptance(root: Any, link_id: str, attempt: str, key_sha256: str) -> None:
    """A failed acceptance drops only this attempt's exact key files and reservation while no link or newer attempt
    holds the id; any provisional place stays Paused and unlinked rather than risking an unproven disconnect."""
    def release(document: dict) -> None:
        held = document["accepting"].get(link_id)
        if link_id in document["links"] or (isinstance(held, dict) and held.get("attempt") != attempt):
            return
        if _drop_attempt_files(root, link_id, attempt, key_sha256) and isinstance(held, dict):
            del document["accepting"][link_id]
    try:
        update_registry(root, release)
    except PeerLinkError:
        _drop_attempt_files(root, link_id, attempt, key_sha256)


def _invite_by_token(document: Mapping[str, Any], digest: str, *, live: bool = True) -> tuple[str, dict] | None:
    """The invitation of this token, only a live one unless ``live`` is False: an expired row kept for its files'
    sake authorises nothing, though presenting its token is a write that retires it (``introduce``)."""
    for key, invite in document["invites"].items():
        if (_live(invite) or (not live and isinstance(invite, dict))) and hmac.compare_digest(
                str(invite.get("token_sha256") or ""), digest):
            return key, invite
    return None


def introduce(root: Any, token: Any, body: Any) -> dict:
    """B's call at A under the one-time invitation token: A records B's key hash and creates B's place with the
    facts my human chose when inviting. A token no invitation here names writes nothing; a mismatched install
    (B naming this installation, or this one moved since inviting) is refused before the invitation is touched.
    One introduction holds the invitation at a time (``claimed_by``), and the link lands in the same write that
    marks the invitation used, so no used invitation is left without its link nor a link without its key."""
    if not isinstance(token, str) or not token:
        raise PeerLinkError("invite_unknown", "an invitation token is required", 401)
    keys = {"link_id", "install_id", "url", "label", "key_sha256", "may_start", "protocol"}
    if not isinstance(body, dict) or set(body) - keys or {"link_id", "install_id", "url", "key_sha256"} - set(body):
        raise PeerLinkError("unexpected_fields", "introduce accepts exactly: " + ", ".join(sorted(keys)))
    if body.get("protocol", PROTOCOL) != PROTOCOL:
        raise PeerLinkError("protocol_unsupported", "protocol 1 is the only one this Ouroboros speaks")
    link_id, peer_install = _valid_link_id(body["link_id"]), _text(body["install_id"], "install_id", limit=64)
    peer_url, label = _url(body["url"], "url"), _label(body.get("label"), "label")
    key = _text(body["key_sha256"], "key_sha256", limit=64)
    remote_may_start = body.get("may_start") if type(body.get("may_start")) is bool else None
    digest, own, attempt = _sha256(token), _current_install_id(root), uuid.uuid4().hex
    if _invite_by_token(read_registry(root), digest, live=False) is None:
        raise PeerLinkError("invite_unknown", "the invitation is unknown, used or expired", 401)

    def claim(document: dict) -> Any:
        found = _invite_by_token(document, digest)
        if found is None:
            return "invite_unknown"
        invite = found[1]
        if (invite.get("used_at") or invite.get("claimed_by") or invite.get("link_id") != link_id
                or link_id in document["links"]):
            return "invite_unknown"
        if not own or invite.get("own_install_id") != own or peer_install == own:
            return "install_mismatch"
        invite["claimed_by"] = attempt
        return found[0], dict(invite)
    claimed = update_registry(root, claim)
    if claimed == "install_mismatch":
        raise PeerLinkError("install_mismatch", "this invitation belongs to another installation", 403)
    if not isinstance(claimed, tuple):
        raise PeerLinkError("invite_unknown", "the invitation is unknown, used or expired", 401)
    invite_key, invite = claimed
    settled = False
    try:
        if not _read_secret(root, link_id):
            raise PeerLinkError("invite_unknown", "this invitation's key is missing here; ask for a new one", 401)
        place_id = _connect_peer_place(root, peer_install_id=peer_install, link_id=link_id,
                                       label=invite.get("label") or label, note=invite.get("note") or "",
                                       mine=True, may_start=bool(invite.get("may_start")))
        code = link_code(link_id, own, peer_install, invite["own_key_sha256"], key)
        row = {"place_id": place_id, "peer_label": invite.get("label") or label or "Ouroboros",
               "peer_url": peer_url, "peer_install_id": peer_install, "own_install_id": own,
               "inbound_key_sha256": key, "key_set_at": utc_now_iso(), "protocol": PROTOCOL,
               "created_at": utc_now_iso(), "code": code, "remote_may_start": remote_may_start}

        def settle(document: dict) -> bool:
            current = document["invites"].get(invite_key)
            if not _live(current) or current.get("claimed_by") != attempt or current.get("used_at"):
                return False
            current.pop("claimed_by", None)
            current["used_at"] = utc_now_iso()
            document["links"][link_id] = row
            return True
        settled = bool(update_registry(root, settle))
        if not settled:
            raise PeerLinkError("invite_unknown", "the invitation expired during the introduction", 401)
    finally:
        if not settled:
            def release(document: dict) -> None:
                current = document["invites"].get(invite_key)
                if isinstance(current, dict) and current.get("claimed_by") == attempt:
                    current.pop("claimed_by", None)
            with contextlib.suppress(PeerLinkError):  # an unwritable store keeps the claim until the deadline
                update_registry(root, release)
    return {"link_id": link_id, "install_id": own, "key_sha256": invite["own_key_sha256"],
            "may_start": bool(invite.get("may_start")), "code": code, "protocol": PROTOCOL}


def revoke(root: Any, link_id: str) -> dict:
    """My human's revoke: the secret, the link's answer and message receipts and the replies owed there leave, the
    inbound key stops matching and the place pauses."""
    from ouroboros.places import AUTHOR_OWNER, STATE_PAUSED, PlaceError, edit_place

    link_id = _valid_link_id(link_id)

    def mark(document: dict) -> dict | None:
        row = document["links"].get(link_id)
        if row is not None and not row.get("revoked_at"):
            row["revoked_at"] = utc_now_iso()
        return dict(row) if row is not None else None
    row = update_registry(root, mark)
    if row is None:
        raise PeerLinkError("link_unknown", "no such link", 401)
    _drop_link_files(root, link_id)
    try:
        edit_place(root, row["place_id"], "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
    except PlaceError as exc:
        if exc.code != "place_not_found":
            raise
    return link_view(root, link_id)


def confirm(root: Any, link_id: str) -> dict:
    """My human's check that this side shows the same six digits as the other: the link's place starts (Active) and
    its letters flow from here on. Confirming again changes nothing (a later pause stays my human's)."""
    from ouroboros.places import AUTHOR_OWNER, STATE_ACTIVE, STATE_PAUSED, PlaceError, edit_place

    link_id = _valid_link_id(link_id)
    row = read_registry(root)["links"].get(link_id)
    if row is None or row.get("revoked_at"):
        raise PeerLinkError("link_unknown", "no such link, or it was revoked", 401)
    if not row.get("code_confirmed_at"):
        # The place starts before the mark lands: a crash between them leaves letters refused until a new confirm.
        edit_place(root, row["place_id"], "set_state", author=AUTHOR_OWNER, state=STATE_ACTIVE)

        def mark(document: dict) -> bool:
            current = document["links"].get(link_id)
            if current is None or current.get("revoked_at"):
                return False
            current.setdefault("code_confirmed_at", utc_now_iso())
            return True
        if not update_registry(root, mark):
            # A revoke landed while this confirm ran, and its pause may have come before the start above: the place
            # of a revoked link ends Paused whichever of the two wrote last.
            with contextlib.suppress(PlaceError):
                edit_place(root, row["place_id"], "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
            raise PeerLinkError("link_unknown", "the link was revoked meanwhile", 401)
    return link_view(root, link_id)


def _code_check(row: Mapping[str, Any]) -> str:
    return "confirmed" if row.get("code_confirmed_at") else "pending"


def link_view(root: Any, link_id: str) -> dict:
    """A link as my human's screens show it: never a secret or a hash of one."""
    row = read_registry(root)["links"].get(link_id)
    if row is None:
        raise PeerLinkError("link_unknown", "no such link", 401)
    from ouroboros.places import place_records

    records = place_records(root)
    place = records.places.get(row["place_id"])
    return {"link_id": link_id, "place_id": row["place_id"], "peer_label": row.get("peer_label") or "",
            "peer_url": row.get("peer_url") or "", "peer_install_id": row.get("peer_install_id") or "",
            "own_install_id": row.get("own_install_id") or "", "code": row.get("code") or "",
            "code_check": _code_check(row), "created_at": row.get("created_at") or "",
            "revoked_at": row.get("revoked_at") or "", "remote_may_start": row.get("remote_may_start"),
            "remote_unknown_at": row.get("remote_unknown_at") or "", "peer": records.peer(row["place_id"]),
            "state": "removed" if row.get("revoked_at") else (place.state if place is not None else "disconnected")}


def link_card(root: Any, place_id: str) -> dict | None:
    """What a linked place's card shows of its link: id, check code, when made or revoked, what the other side
    allows; None when no link names the place or the registry cannot be read (never a key or a hash)."""
    try:
        found = link_of_place(root, place_id)
    except PeerLinkError:
        return None
    if found is None:
        return None
    link_id, row = found
    return {"link_id": link_id, "code": row.get("code") or "", "code_check": _code_check(row),
            "peer_install_id": row.get("peer_install_id") or "", "created_at": row.get("created_at") or "",
            "revoked_at": row.get("revoked_at") or "", "remote_may_start": row.get("remote_may_start"),
            "remote_unknown_at": row.get("remote_unknown_at") or ""}


def mark_remote_unknown(root: Any, link_id: str) -> None:
    """The other side answered ``link_unknown`` to this side's key: it no longer knows the link (revoked there,
    or its data moved). The fact and its first time stay on the link until a new introduction replaces it."""
    def mark(document: dict) -> None:
        row = document["links"].get(link_id)
        if row is not None and not row.get("remote_unknown_at"):
            row["remote_unknown_at"] = utc_now_iso()
    update_registry(root, mark)


def has_outgoing_link(root: Any) -> bool:
    """Whether this installation holds a confirmed, unrevoked link: the letter tool is shown only then. An unreadable
    registry is not proof of none (the tool stays shown and a send refuses typed)."""
    try:
        rows = read_registry(root)["links"].values()
    except PeerLinkError:
        return True
    return any(row.get("code_confirmed_at") and not row.get("revoked_at") for row in rows)


def link_of_place(root: Any, place_id: str) -> tuple[str, dict] | None:
    for link_id, row in read_registry(root)["links"].items():
        if row.get("place_id") == place_id:
            return link_id, row
    return None


def authenticate(root: Any, presented: Any) -> dict:
    """The caller behind ``X-Ouroboros-Link: <link_id>.<secret>``: a known, unrevoked link whose inbound hash
    matches, compared in constant time. A missing, revoked or mismatched key is ``link_unknown`` (a bad key never
    falls through); a registry that cannot be read now stays ``peer_links_unreadable`` (503), never an absent
    link, so the sender asks again rather than taking it for a revocation."""
    link_id, _sep, secret = str(presented or "").strip().partition(".")
    row = read_registry(root)["links"].get(_valid_link_id(link_id)) if secret else None
    expected = str((row or {}).get("inbound_key_sha256") or _EMPTY_SHA)
    matched = hmac.compare_digest(expected, _sha256(secret))
    if row is None or row.get("revoked_at") or not matched:
        raise PeerLinkError("link_unknown", "the link key is unknown or revoked", 401)
    return {"kind": "peer", "link_id": link_id, "place_id": row["place_id"],
            "peer_label": row.get("peer_label") or "Ouroboros", "peer_install_id": row.get("peer_install_id") or "",
            "own_install_id": row.get("own_install_id") or "", "code_confirmed": bool(row.get("code_confirmed_at"))}


@dataclass(frozen=True)
class OutgoingLink:
    """What the sending side needs to write to its linked Ouroboros. The secret stays in its file: only
    ``auth_headers()`` reads it, at the moment of a send, and nothing here prints it."""

    link_id: str
    place_id: str
    peer_url: str
    peer_label: str
    peer_install_id: str
    own_install_id: str
    drive_root: str
    remote_unknown_at: str = ""  # the other side answered link_unknown: nothing is sent until a new introduction

    def auth_headers(self) -> dict[str, str]:
        """The three headers of a letter: this install's current id (a moved data directory is refused there)."""
        secret = _read_secret(self.drive_root, self.link_id)
        if not secret:
            raise PeerLinkError("link_unknown", "this link has no key here", 401)
        return {LINK_HEADER: f"{self.link_id}.{secret}",
                INSTALL_HEADER: _current_install_id(self.drive_root) or self.own_install_id,
                PROTOCOL_HEADER: str(PROTOCOL)}


def outgoing_link(place_id: str, *, drive_root: Any = None) -> OutgoingLink | None:
    """The link behind one of my places, or None: no link, revoked, its code not confirmed here, the place paused or
    gone, or not marked mine. ``drive_root`` defaults to this installation's canonical data root
    (``install_identity.canonical_root``)."""
    return _link_of(place_id, drive_root, paused=False)


def stopping_link(place_id: str, *, drive_root: Any = None) -> OutgoingLink | None:
    """``outgoing_link`` that a Paused place still yields, for one act only: the Stop of a letter this side already
    started there and its read-back (``gateways.peer.PeerGateway.cancel_run``). A Pause stops what my side says, never
    my side's own Stop; nothing else reaches the other side through it."""
    return _link_of(place_id, drive_root, paused=True)


def _link_of(place_id: str, drive_root: Any, *, paused: bool) -> OutgoingLink | None:
    from ouroboros.install_identity import canonical_root
    from ouroboros.places import STATE_PAUSED, place_records

    root = pathlib.Path(drive_root) if drive_root is not None else canonical_root()
    try:
        found = link_of_place(root, str(place_id or ""))
        records = place_records(root)
    except (PeerLinkError, ValueError, OSError):
        return None
    if found is None:
        return None
    link_id, row = found
    place, peer = records.places.get(row["place_id"]), records.peer(row["place_id"])
    if (row.get("revoked_at") or not row.get("code_confirmed_at") or place is None
            or (place.state == STATE_PAUSED and not paused) or not peer or not peer["mine"]
            or not _read_secret(root, link_id)):
        return None
    return OutgoingLink(link_id=link_id, place_id=row["place_id"], peer_url=row.get("peer_url") or "",
                        peer_label=row.get("peer_label") or "Ouroboros",
                        peer_install_id=row.get("peer_install_id") or "",
                        own_install_id=row.get("own_install_id") or "", drive_root=str(root),
                        remote_unknown_at=row.get("remote_unknown_at") or "")
