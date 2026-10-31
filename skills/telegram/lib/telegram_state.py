"""Telegram bridge: settings/state file helpers + subagent-card rendering."""
from __future__ import annotations

import json
import pathlib
import threading
from typing import Any, Dict

import httpx

# This native payload ships in lockstep with core. DEVELOPMENT.md requires this
# shared truncation/JSONL implementation to remain the single source of truth.
from ouroboros.utils import iter_jsonl_objects, truncate_review_artifact

from ..scripts.telegram_settings import load_settings
from .telegram_i18n import Index


_RUNTIME_STATE_TIMEOUT_SEC = 2.0
_MAX_MESSAGE_REFS = 1000
_MESSAGE_REFS_LOCK = threading.RLock()


def _reference_state(api) -> dict:
    data = _read_json_file(_state_file(api, "message_refs.json"))
    return data if isinstance(data, dict) else {}


def _message_refs(api) -> dict:
    """Bounded transport index, never authority for host routing or authorship."""
    return _reference_state(api).get("messages", {})


def _remember_message(api, chat_id: int, message_id: int, record: dict) -> None:
    if not message_id:
        return
    with _MESSAGE_REFS_LOCK:
        state = _reference_state(api)
        data = state.setdefault("messages", {})
        key = f"{chat_id}:{message_id}"
        data.pop(key, None)
        data[key] = dict(record)
        for stale in list(data)[:-_MAX_MESSAGE_REFS]:
            data.pop(stale, None)
        if record.get("direction") == "in":
            latest = state.setdefault("latest_owner", {})
            latest[str(chat_id)] = max(message_id, latest.get(str(chat_id), 0))
            for stale in list(latest)[:-_MAX_MESSAGE_REFS]:
                latest.pop(stale, None)
        path = _state_file(api, "message_refs.json")
        # The map is derived: losing an entry costs one reference, never the
        # owner's message or a confirmed chunk's later siblings.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            api.log("warning", f"Telegram message reference was not saved ({type(exc).__name__}); "
                               "its reply reference may be unavailable or stale.")


def _owner_message_seen(api, chat_id: int, message_id: int) -> None:
    """Even a local command is the latest owner message in this Telegram chat."""
    if not message_id:
        return
    record = _message_ref(api, chat_id, message_id)
    _remember_message(api, chat_id, message_id, {
        **record, "direction": "in", "chat_id": chat_id, "kind": "message",
    })


def _owner_ingress_fields(api, chat_id: int, message: dict, update_id: int) -> dict:
    message_id = int(message.get("message_id") or 0)
    client_id = f"tg:{chat_id}:{message_id or update_id}"
    if message_id:
        _remember_message(api, chat_id, message_id, {
            **_message_ref(api, chat_id, message_id), "direction": "in", "chat_id": chat_id,
            "kind": "message", "client_message_id": client_id,
        })
    reply = message.get("reply_to_message")
    address = _reply_reference(api, chat_id, int(reply.get("message_id") or 0)) if isinstance(reply, dict) else ""
    return {"client_message_id": client_id, **({"reply_to": address} if address else {})}


def _is_routing_notice(event: dict) -> bool:
    """A routing act's owner notice: its typed ACK, or the System row its receipt selected as carrier."""
    return event.get("annotation_type") == "routing_ack" or (
        event.get("role") == "system" and event.get("routing_notice_carrier") == "system")


def _native_reply_id(api, chat_id: int, event: dict) -> int:
    """Join host-derived owner references to exact, same-chat Telegram ingress.

    A reply may arrive while /chat/inject is still awaiting its HTTP receipt.
    Bind pending ids here from the canonical acceptance index, never by text or
    chat order. The index already serves the shared ingress and reads only deltas.
    An answer quotes only an earlier owner message; a routing notice also quotes
    the latest one, since it says what became of that very message.
    """
    reply = event.get("reply_to")
    if not isinstance(reply, dict) or reply.get("role") != "user" or reply.get("unresolved"):
        return 0
    address = str(reply.get("address") or "")
    if not address:
        return 0
    with _MESSAGE_REFS_LOCK:
        owners = [(int(key.split(":", 1)[1]), record) for key, record in _message_refs(api).items()
                  if key.startswith(f"{chat_id}:") and record.get("direction") == "in"]
        latest = _reference_state(api).get("latest_owner", {}).get(str(chat_id), 0)
    for mid, record in owners:
        if not record.get("row_address") and record.get("client_message_id"):
            from supervisor.message_bus import accepted_chat_message
            from ouroboros.chat_chain import format_address, row_address

            try:
                row = accepted_chat_message(_data_dir(api), record["chat_id"], record["client_message_id"])
            except Exception:
                api.log("warning", "Telegram reply source could not be read; sending without a native reply.")
                continue
            if row is not None:
                record = {**record, "row_address": format_address(row_address(row))}
                _remember_message(api, chat_id, mid, record)
        if record.get("row_address") == address and record.get("chat_id") == reply.get("chat_id"):
            return mid if (latest and mid != latest) or _is_routing_notice(event) else 0
    return 0


def _quiet_progress(event: dict) -> bool:
    """A progress note Silent mode mutes and lets the next note of its episode replace: the
    model's own narration, or the host's typed technical progress (``host_progress``). A
    problem the host types as ``task_incident``, and every other System row, keeps its sound
    and is never replaced."""
    if event.get("is_progress") is not True:
        return False
    return event.get("role") != "system" or (
        event.get("system_type") == "host_progress" and not event.get("task_incident"))


def _mirror_options(api, event: dict, chat_id: int) -> dict:
    """A client mirrors one host row; all of its physical parts share this fact."""
    task_id = str(event.get("subagent_task_id") or event.get("task_id") or "")
    kind = str(event.get("kind") or event.get("type") or event.get("system_type") or "message")
    if (event.get("subagent_event") or kind == "task_summary"
            or event.get("role") == "system" and event.get("system_type") == "project_completion_summary"):
        # Telegram shows a status/salvage notice, not Main's completion_answer.
        kind = "work_card"
    elif event.get("quiz_id"):
        kind = "question"
    reply = event.get("reply_to")
    native_reply_id = _native_reply_id(api, chat_id, event)
    record = {"row_address": str(event.get("row_address") or ""), "chat_id": event.get("chat_id"),
              "task_id": task_id, "kind": kind,
              "reply_to": str(reply.get("address") or "") if isinstance(reply, dict) else "",
              "native_reply_id": native_reply_id}
    if _quiet_progress(event) and kind not in ("work_card", "question"):
        record["progress"] = True  # the type fact Silent mode's replacement requires
    return {"on_message": lambda chat, mid: _remember_message(api, chat, mid, record),
            "reply_to_message_id": native_reply_id}


def _replaceable_progress(api, chat_id: int, message_id: int, event: dict) -> bool:
    """Whether Silent mode may replace ``message_id`` with this progress note.

    Only a technical progress note is ever replaced, and only by the next progress
    note of its own episode: its reference must positively carry the progress fact.
    The same task, reply origin or anchor alone never makes a reply, a final answer
    or a reference saved by an older bridge replaceable. Telegram cannot change a
    message's native reply anchor by editing its text, so a changed host reply
    origin, like a Continue episode, starts a fresh bubble.
    """
    old = _message_ref(api, chat_id, message_id)
    reply = event.get("reply_to")
    address = str(reply.get("address") or "") if isinstance(reply, dict) else ""
    return (old.get("progress") is True and event.get("is_progress") is True
            and str(old.get("task_id") or "") == str(event.get("task_id") or "")
            and str(old.get("reply_to") or "") == address
            and int(old.get("native_reply_id") or 0) == _native_reply_id(api, chat_id, event))


def _message_ref(api, chat_id: int, message_id: int) -> dict:
    with _MESSAGE_REFS_LOCK:
        value = _message_refs(api).get(f"{chat_id}:{message_id}")
        return dict(value) if isinstance(value, dict) else {}


def _reply_reference(api, chat_id: int, message_id: int) -> str:
    record = _message_ref(api, chat_id, message_id)
    task_id = str(record.get("task_id") or "")
    if record.get("kind") == "work_card" and task_id:
        return f"task:{task_id}"
    return str(record.get("row_address") or (f"task:{task_id}" if task_id else ""))


def _data_dir(api) -> pathlib.Path:
    """Main Ouroboros data dir (state/logs live here), obtained safely from the PluginAPI."""
    try:
        data_dir = api.get_runtime_info().get("data_dir")
        if data_dir:
            return pathlib.Path(data_dir)
    except Exception:
        pass
    return pathlib.Path(api.get_state_dir()).parent.parent.parent


def _read_json_file(path: pathlib.Path) -> Dict[str, Any]:
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _jsonl_tail(
    path: pathlib.Path,
    *,
    max_entries: int,
    tail_bytes: int,
) -> tuple[list[dict[str, Any]], bool]:
    """A bounded tail; canonical chat/progress come only from the dialogue store.

    The legacy path is an address, not a file fallback. Missing import coverage
    raises a typed outcome before a notifier can mark the window as observed.
    Other telemetry files still use the shared JSONL parser.
    """
    path = pathlib.Path(path)
    if path.parent.name == "logs" and path.name in {"chat.jsonl", "progress.jsonl"}:
        from ouroboros import dialogue_store as ds

        root, stream = path.parent.parent, path.stem
        upper = ds.stream_end(root, stream)
        lower = max(0, upper - tail_bytes)
        entries = [row for row in ds.newest(root, stream, before=upper, limit=max_entries + 1)
                   if row.seq >= lower]
        omitted_prefix = lower > 0 or len(entries) > max_entries
        if len(entries) > max_entries:
            entries = entries[-max_entries:]
            lower = entries[0].seq
        coverage = ds.range_coverage(root, stream, lower=lower, upper=upper)
        if coverage["mismatch"]:
            raise ds.StoreError(f"{stream}: a segment could not be imported")
        if coverage["pending"]:
            raise ds.StoreMigrating(f"{stream}: history is still importing")
        if coverage["gaps"]:
            raise ds.StoreError(f"{stream}: recorded invalid intervals in the requested window")
        return [entry.row for entry in entries], omitted_prefix
    try:
        omitted_prefix = path.stat().st_size > tail_bytes
    except OSError:
        omitted_prefix = False
    try:
        rows = list(
            iter_jsonl_objects(
                path,
                max_entries=max_entries + 1,
                tail_bytes=tail_bytes,
            )
        )
    except OSError:
        rows = []
    if len(rows) > max_entries:
        omitted_prefix = True
        rows = rows[-max_entries:]
    return rows, omitted_prefix


async def _load_runtime_state(api) -> dict[str, Any]:
    """Read the ledger-backed runtime projection from the local gateway."""
    try:
        info = api.get_runtime_info()
        port = int(info.get("server_port") or 0) if isinstance(info, dict) else 0
        if not 1 <= port <= 65_535:
            return {}
        async with httpx.AsyncClient(
            timeout=_RUNTIME_STATE_TIMEOUT_SEC,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = await client.get(f"http://127.0.0.1:{port}/api/state")
        if response.status_code != 200:
            return {}
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _state_file(api, name: str) -> pathlib.Path:
    return pathlib.Path(api.get_state_dir()) / name


def _load_settings(api) -> Dict[str, Any]:
    return load_settings(pathlib.Path(api.get_state_dir()))


def _is_silent_mode_enabled(settings: Dict[str, Any]) -> bool:
    """Silent mode sends replies, the final answer and progress without sound, and a
    progress note replaces the progress note just before it (``_replaceable_progress``);
    every authored message stays its own message. Default: off."""
    raw = str(settings.get("TELEGRAM_SILENT_MODE") or "off").strip().lower()
    return raw in ("on", "true", "1", "yes")


def _load_silent_state(api) -> Dict[str, int]:
    """Load per-chat last outbound message id mapping. Returns {} if missing/corrupt."""
    path = _state_file(api, "silent_state.json")
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                out: Dict[str, int] = {}
                for key, value in data.items():
                    try:
                        out[str(key)] = int(value)
                    except (TypeError, ValueError):
                        continue
                return out
    except Exception:
        pass
    return {}


def _save_silent_state(api, data: Dict[str, int]) -> None:
    path = _state_file(api, "silent_state.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _get_silent_msg(api, chat_id: int) -> int:
    return int(_load_silent_state(api).get(str(chat_id)) or 0)


def _set_silent_msg(api, chat_id: int, message_id: int) -> None:
    state = _load_silent_state(api)
    state[str(chat_id)] = int(message_id)
    _save_silent_state(api, state)


def _clear_silent_msg(api, chat_id: int) -> None:
    state = _load_silent_state(api)
    if str(chat_id) in state:
        state.pop(str(chat_id), None)
        _save_silent_state(api, state)


def _load_subagent_state(api) -> Dict[str, int]:
    """Per-(chat, subagent) Telegram message-id map so each subagent gets ONE
    bubble that is edited in place across its lifecycle instead of spamming."""
    path = _state_file(api, "subagent_state.json")
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                out: Dict[str, int] = {}
                for key, value in data.items():
                    try:
                        out[str(key)] = int(value)
                    except (TypeError, ValueError):
                        continue
                return out
    except Exception:
        pass
    return {}


def _save_subagent_state(api, data: Dict[str, int]) -> None:
    path = _state_file(api, "subagent_state.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _get_subagent_msg(api, chat_id: int, child_id: str) -> int:
    return int(_load_subagent_state(api).get(f"{chat_id}:{child_id}") or 0)


def _set_subagent_msg(api, chat_id: int, child_id: str, message_id: int) -> None:
    state = _load_subagent_state(api)
    state[f"{chat_id}:{child_id}"] = int(message_id)
    # Bound growth: one entry per subagent ever seen would leak unbounded. Keep
    # the most recent 200 (dict preserves insertion order).
    if len(state) > 200:
        for stale in list(state.keys())[:-200]:
            state.pop(stale, None)
    _save_subagent_state(api, state)


def _subagent_cards_enabled(settings: Dict[str, Any]) -> bool:
    """Render each subagent's lifecycle as one updating Telegram message. Default: on."""
    raw = str(settings.get("TELEGRAM_SUBAGENT_CARDS") or "on").strip().lower()
    return raw in ("on", "true", "1", "yes")


def _mirror_progress_enabled(settings: Dict[str, Any]) -> bool:
    """Mirror generic progress telemetry to Telegram. Default: off."""
    raw = str(settings.get("TELEGRAM_MIRROR_PROGRESS") or "off").strip().lower()
    return raw in ("on", "true", "1", "yes")


def _child_row_held_for_root(api, event: Dict[str, Any]) -> bool:
    """Whether a child's card-internal host row stays out of this chat for now.

    The web renders such a row inside the child's card. Here the owner hears
    about a child from its root, so the row is not sent while the root is
    unfinished. Only typed lifecycle facts decide: the row's lineage, its card
    placement or incident type, and the root's durable status. A root whose
    status cannot be read counts as unfinished; a row naming no root is not held.
    """
    if str(event.get("delegation_role") or "").strip().lower() != "subagent":
        return False
    if not event.get("card_row") and str(event.get("system_type") or "") != "terminal_incident":
        return False
    root = str(event.get("root_task_id") or event.get("parent_task_id") or "").strip()
    if not root:
        return False
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import FINAL_STATUSES

    try:
        stored = load_task_result(_data_dir(api), root)
    except Exception:
        return False
    status = str(stored.get("status") or "").strip().lower() if isinstance(stored, dict) else ""
    return status not in FINAL_STATUSES


_SUBAGENT_ICONS = {
    "scheduled": "🔵", "running": "🟡", "update": "🟡", "progress": "🟡",
    "completed": "✅", "completed_warn": "⚠️", "failed": "❌",
    "cancelled": "🚫", "rejected": "♻️", "interrupted": "⏸️",
}
# Terminal states lock the bubble's final look; 'interrupted' is retryable so its
# bubble stays editable for the resumed run.
_SUBAGENT_TERMINAL = {"completed", "completed_warn", "failed", "cancelled", "rejected"}
_SUBAGENT_LABELS_EN = {"scheduled": "queued", "running": "running", "update": "running", "progress": "running",
    "completed": "done", "completed_warn": "done (warn)", "failed": "failed",
    "cancelled": "stopped", "rejected": "rejected (duplicate)",
    "interrupted": "interrupted", "_title": "subagent"}
_SUBAGENT_LABELS = Index("subagent", _SUBAGENT_LABELS_EN, "the status words of a subagent card")


def _subagent_card_text(event: Dict[str, Any], sub_event: str, lang: str) -> str:
    """One subagent's card: a status header plus the latest work note.

    The card is a SINGLE Telegram message edited in place across the subagent's
    lifecycle. The header line carries icon/role/status/cost; the body mirrors
    the live progress commentary shown in the Ouroboros UI (the "I will read
    lines …" notes). Because the note changes every event, the edit always has
    fresh content — so the message updates in place instead of Telegram
    rejecting an identical edit and forcing a brand-new bubble. Sent verbatim
    as plain text (no HTML parse risk)."""
    labels = _SUBAGENT_LABELS[lang]
    icon = _SUBAGENT_ICONS.get(sub_event, "🟡")
    role = str(event.get("subagent_role") or "").strip()
    child_ref = str(event.get("subagent_task_id") or event.get("task_id") or "").strip()
    child = child_ref[:8]
    who = role or labels["_title"]
    header = f"{icon} {who} — {labels.get(sub_event, sub_event)}"
    if child:
        header += f" [{child}]"
    try:
        cost = float(event.get("cost_usd") or 0)
    except (TypeError, ValueError):
        cost = 0.0
    if sub_event in _SUBAGENT_TERMINAL and cost > 0:
        header += f" · ${cost:.2f}"
    # Live work commentary: the in-flight note, or the result/summary on finish. A
    # delegated run's observation carries its typed parts: the card then leads with the
    # executor's attributed latest words (an earlier one when this batch had none), then
    # problems and the technical-event count, instead of the frame's joined labels.
    activity = event.get("delegated_activity")
    note = str(event.get("text") or event.get("result") or event.get("trace_summary") or "").strip()
    if isinstance(activity, dict) and activity.get("parts") is not None:
        from ouroboros.delegate_activity import card_text

        body = card_text(activity, 650).strip()
        if body:
            # The shared renderer already bounds speech around its diagnostic
            # tail. Never cut that tail again or lose the complete-source cue.
            full_ref = child_ref or str(activity.get("task_id") or "").strip()
            return f"{header}\n{body}" + (f"\nFull ref: task {full_ref}" if full_ref else "")
    if note:
        preview = truncate_review_artifact(note, 700)
        if preview != note and child_ref:
            preview += f"\nFull ref: task {child_ref}"
        return f"{header}\n{preview}"
    return header


async def _render_subagent_card(api, client, chat_id: int, event: Dict[str, Any], sub_event: str, lang: str,
                                *, quiet: bool = False) -> None:
    """One Telegram message per subagent, edited in place across its lifecycle; in Silent
    mode (``quiet``) a new bubble is sent without sound, as other progress is."""
    child = str(event.get("subagent_task_id") or event.get("task_id") or "").strip()
    if not child:
        return
    text = _subagent_card_text(event, sub_event, lang)
    msg_id = _get_subagent_msg(api, chat_id, child)
    if msg_id:
        # Edit the subagent's existing bubble (plain text — no HTML parse risk).
        if await client.edit_message_text(chat_id, msg_id, text, parse_mode=""):
            return
        # Edit failed (message too old / deleted) → post a fresh bubble below.
    new_id = await client.send_message(chat_id, text, parse_mode="", **({"disable_notification": True} if quiet else {}))
    if new_id:
        _set_subagent_msg(api, chat_id, child, new_id)
