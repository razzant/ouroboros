"""Telegram bridge: periodic push notifications (task done / budget thresholds)."""
from __future__ import annotations

import asyncio
import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .telegram_api import (
    TELEGRAM_RETRY_INITIAL_SEC,
    TelegramClient,
    TelegramRequestRejected,
    TelegramTransportError,
    next_telegram_retry_delay,
    _LOCALIZED_TEXTS,
)
from .telegram_state import (
    _data_dir,
    _jsonl_tail,
    _load_runtime_state,
    _load_settings,
    _read_json_file,
    _state_file,
    _mirror_options,
)
from ..scripts.telegram_settings import telegram_api_base, telegram_files_dir, telegram_proxy
from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID, is_project_chat_id
from ouroboros import i18n_memory as _memory
from ouroboros.project_dialogue import OUTCOME_PHASE_HEADLINE, routing_option_label

from . import telegram_i18n
from .telegram_i18n import Index


def _notify_enabled(settings: Dict[str, Any], key: str) -> bool:
    return str(settings.get(key) or "off").strip().lower() in ("on", "true", "1", "yes")


def _load_notif_state(api) -> Dict[str, Any]:
    return _read_json_file(_state_file(api, "notif_state.json"))


def _save_notif_state(api, data: Dict[str, Any]) -> None:
    path = _state_file(api, "notif_state.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _pinned_chat_id(settings: Dict[str, Any]) -> int:
    raw = str(settings.get("TELEGRAM_CHAT_ID") or "").strip()
    try:
        return int(raw)
    except ValueError:
        return 0


async def _push_notification(
    api, chat_id: int, text: str, *, trust_env: bool = False, event=None,
) -> Tuple[str, Optional[BaseException]]:
    """Send one notification; never raises a typed Telegram failure.

    Returns ``("sent", None)`` on delivery, ``("transient", exc)`` when the
    failure is worth retrying next cycle, and ``("skipped", exc)`` on a
    permanent rejection: the notification is consumed so one dead send cannot
    replay forever and exhaust the supervised-restart budget.

    Transient failures log at debug only: the notifier loop's degraded/
    recovered transition warning (which carries the exception) is the one
    owner-visible line per episode.
    """
    protected = api.get_settings(["TELEGRAM_BOT_TOKEN"])
    state_dir = Path(api.get_state_dir())
    client = TelegramClient(
        protected.get("TELEGRAM_BOT_TOKEN", ""), trust_env=trust_env,
        proxy=telegram_proxy(state_dir),
        **(_mirror_options(api, event, chat_id) if event is not None else {}),
        api_base_url=telegram_api_base(state_dir),
        files_dir=telegram_files_dir(state_dir),
    )
    try:
        await client.send_message(int(chat_id), text, parse_mode="")
        return "sent", None
    except TelegramTransportError as exc:
        api.log("debug", f"Telegram notify failed: {exc}")
        return "transient", exc
    except TelegramRequestRejected as exc:
        if exc.transient:
            api.log("debug", f"Telegram notify failed: {exc}")
            return "transient", exc
        api.log("error", f"Telegram notification permanently rejected; skipping it: {exc}")
        return "skipped", exc


_BUDGET_THRESHOLDS = (100, 90, 80)  # checked high → low


async def _check_budget_notify(
    api, settings: Dict[str, Any], chat_id: int, state: Dict[str, Any], lang: str,
    *, trust_env: bool = False,
) -> Tuple[Optional[BaseException], bool]:
    """Returns (transient send failure if any, whether a send was delivered)."""
    if not _notify_enabled(settings, "TELEGRAM_NOTIFY_BUDGET"):
        return None, False
    snapshot = await _load_runtime_state(api)
    try:
        spent = float(snapshot["spent_usd"])
        total = float(snapshot["budget_limit"])
        pct = float(snapshot["budget_pct"])
    except (KeyError, TypeError, ValueError):
        return None, False
    if total <= 0:
        return None, False
    crossed = 0
    for thr in _BUDGET_THRESHOLDS:
        if pct >= thr:
            crossed = thr
            break
    notified = int(state.get("budget_threshold") or 0)
    delivered = False
    if crossed > notified:
        msg = _NOTIFY[lang].format("budget", pct=f"{pct:.0f}", spent=f"{spent:.2f}", total=f"{total:.2f}")
        outcome, exc = await _push_notification(api, chat_id, msg, trust_env=trust_env)
        if outcome == "transient":
            return exc, False
        # "sent" delivered it; "skipped" consumes it (permanent rejection) so
        # the same send does not replay every cycle forever.
        delivered = outcome == "sent"
        state["budget_threshold"] = crossed
    elif crossed < notified:
        state["budget_threshold"] = crossed  # budget raised / spend reset → re-arm
    return None, delivered


def _axis_status(axes: Any, axis: str) -> str:
    """One `outcome_axes` axis status, tolerating the legacy bare-string shape.

    Canonical rows carry ``{"lifecycle": {"status": "completed"}}``; pre-axes
    rows carry ``{"lifecycle": "completed"}``. Stringifying the axis itself is
    what once pushed ``· {'status': 'completed'}`` with a permanent warning
    icon, so read the status and never the container.
    """
    value = axes.get(axis) if isinstance(axes, dict) else None
    if isinstance(value, dict):
        value = value.get("status")
    return str(value or "")


def _summary_ids_in_tail(api, limit: int = 200) -> list:
    rows, _omitted = _jsonl_tail(
        _data_dir(api) / "logs" / "chat.jsonl",
        max_entries=limit,
        tail_bytes=256 * 1024,
    )
    summaries = {}
    for e in rows:
        if (str(e.get("type") or "") == "task_summary" and e.get("task_id")
                and e.get("outcome_final") is not False
                and str(e.get("outcome_phase") or "") != "working"):
            summaries[str(e.get("task_id"))] = e
    return list(summaries.items())


# The host's task-phase vocabulary is the SOURCE here, not a copy: this transport
# lowercases the same words the task card prints, so a new phase or a reworded status
# can never mean one thing on the card and another on a phone. ``working`` is excluded
# on purpose — a pre-finalization row is not a finish, and its absence is what makes the
# legacy axes fallback below fire.
_PHASE_WORDS = {phase: word.lower() for phase, word in OUTCOME_PHASE_HEADLINE.items() if phase != "working"}


def _phase_word(phase: str, lang: str) -> str:
    """The lowercase status word in the install language: the host's own headline table
    read by code through the translation memory (the same entry the web card paints)."""
    source = OUTCOME_PHASE_HEADLINE.get(phase) or OUTCOME_PHASE_HEADLINE["done"]
    if telegram_i18n.english(lang):
        return source.lower()
    found = _memory.tr(f"{_memory.CODE_PREFIX}task.headline.{phase}", lang, source, drive_root=telegram_i18n.drive_root())
    return str(found or source).lower()


# The two push lines this transport composes itself; everything else it relays is the
# host's own sentence, sent as the host wrote it.
_NOTIFY = Index("notify", {
    "task_finished": "{icon} Task {id} {word}{tail}",
    "task_finished_details": "Task {id} finished{tail}. Review and verification details are in the app.",
    "project_finished": "{icon} {label} · {word}{tail}",
    "project_finished_details": "{label} · finished{tail}. Review and verification details are in the app.",
    "budget": "⚠️ Budget: {pct}% (${spent} / ${total})",
    "cost_final": "${amount}",
    "cost_up_to": "up to ${amount}",
    "cost_tracked": "Tracked: {cost}",
}, "push notifications about finished tasks and the budget")
_PHASE_ICONS = {"done": "✅", "warn": "⚠️", "error": "❌", "cancelled": "🚫"}

# The owner's own words Ouroboros passes to another work (a relay, one Ouroboros message): each quoted
# message under where he wrote it, verbatim, with the names of the files that came with it, then the note.
_RELAY = Index("relay", {
    "head": "Your message in {place}:",
    "files": "Attached: {names}",
}, "the owner's own words quoted when Ouroboros passes them on to another work")


def _relay_text(relay: Any, note: str, lang: str) -> str:
    """The relay's quoted owner messages above its note; nothing when it quotes none (a note between works
    is written for that work, so it stays in the room and never reaches the phone)."""
    relay = relay if isinstance(relay, dict) else {}
    quotes = relay.get("quotes") if isinstance(relay.get("quotes"), list) and relay.get("quotes") else [relay]
    blocks = []
    for quote in quotes:
        if not isinstance(quote, dict):
            continue
        words = str(quote.get("text") or "").strip()
        names = [str(item.get("name") or item.get("display_name") or item.get("filename") or item.get("label") or "").strip()
                 for item in quote.get("attachments") or [] if isinstance(item, dict)]
        names = [name for name in names if name]
        if not words and not names:
            continue
        place = str((quote.get("source") or {}).get("label") or "Main") if isinstance(quote.get("source"), dict) else "Main"
        lines = [_RELAY[lang].format("head", place=place)]
        lines += [words] if words else []
        lines += [_RELAY[lang].format("files", names=", ".join(names))] if names else []
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks + ([note] if note else [])) if blocks else ""


def _routing_notice_text(event: Dict[str, Any], lang: str) -> Tuple[str, bool]:
    """A routing receipt's owner notice as ONE message, and whether it asks for a choice.

    The refused act's target and cause, as the host wrote them (unless the host chose the
    System row as this notice's carrier); then the picker when the host asks for a target, a
    numbered list the owner answers in plain words (the router reads it from history, P5) or
    on the web picker card; then the files it could not attach. A landed act says nothing,
    and a picker gets a reason line only when the host sent a cause, never an invented one.
    """
    lines = []
    if str(event.get("cause") or "").strip() and event.get("routing_notice_carrier") != "system":
        lines.append(" · ".join(str(event[key]).strip() for key in ("target_label", "cause") if str(event.get(key) or "").strip()))
    options = event.get("options") if isinstance(event.get("options"), list) else []
    labels = [label for label in map(routing_option_label, options) if label][:8] \
        if str(event.get("status") or "") == "needs_manual_target" else []
    texts = _LOCALIZED_TEXTS[lang]  # the bridge's own lines; the host's cause and labels stay as written
    if labels:
        lines.append(texts["routing_choose_target"])
        lines.extend(f"{index}. {label}" for index, label in enumerate(labels, 1))
        if len(options) > len(labels):
            lines.append(texts.format("routing_more_targets", count=len(options) - len(labels)))
        lines.append(texts["routing_reply_with_number"])
    names = [str(row.get("label") or "file") for row in event.get("attachment_manifest") or []
             if isinstance(row, dict) and row.get("status") == "rejected"]
    if names:
        lines.append(texts.format("attachments_not_attached", names=", ".join(names)))
    return "\n".join(lines), bool(labels)

# Completion notices are opt-in for every outcome. The task's evidence stays in
# its details even when the owner chooses no additional phone messages.
# Only a ROOT outcome, and only in a room the owner actually talks in. ``summary_kind``
# is the typed root fact the host stamps, without which a twelve-child swarm pushes
# twelve lines; the chat id keeps the hidden partition (Skill Review and headless
# admissions, HIDDEN_CHAT_ID = 0) and synthetic A2A rooms off the phone.
# A legacy row that predates either field never pushes.
_ROOT_SUMMARY_KIND = "terminal_root_projection"


def _announce_summary(row: Dict[str, Any], *, notify_all: bool) -> bool:
    """The existing toggle controls every extra finish notice, never ordinary replies."""
    if str(row.get("summary_kind") or "") != _ROOT_SUMMARY_KIND:
        return False
    if row.get("quiet_effective") is True:
        return False
    try:
        chat_id = int(row.get("chat_id") or 0)
    except (TypeError, ValueError):
        return False
    if chat_id != WEB_UI_CHAT_ID and not is_project_chat_id(chat_id):
        return False
    return notify_all


def _cost_text(presentation: Any, lang: str) -> str:
    """The task card's own cost words (web ``costPresentationMeta``) for a root's whole-tree
    carrier: a final amount, ``up to`` while costs may still arrive, ``Tracked:`` when some
    step has no price. Nothing known is no money part here, never a zero or a guess; the
    retired ``cost_usd`` alias is not a current price (#971)."""
    if (not isinstance(presentation, dict) or presentation.get("scope") != "root_tree"
            or not presentation.get("has_rows")):
        return ""
    try:
        amount = float(presentation["tracked_amount"])
    except (KeyError, TypeError, ValueError):
        return ""
    if not math.isfinite(amount):
        return ""
    # ``toFixed(2)``: the exact binary value, a tie going up, as the card rounds it.
    money = str(Decimal(amount).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    texts = _NOTIFY[lang]
    cost = texts.format("cost_final" if presentation.get("tracked_final") else "cost_up_to", amount=money)
    return texts.format("cost_tracked", cost=cost) if presentation.get("has_unpriced") else cost


def _completion_text(row: Dict[str, Any], tid: str, lang: str, parts=()) -> str:
    """Render the host's compact view on both existing completion transports."""
    parts = list(parts)
    notice = row.get("completion_notice")
    notice = notice if isinstance(notice, dict) and isinstance(notice.get("neutral"), bool) and isinstance(notice.get("detail"), str) else None
    phase = str((notice or {}).get("phase") or row.get("outcome_phase") or "")
    if phase in _PHASE_WORDS:
        icon, word = _PHASE_ICONS[phase], _phase_word(phase, lang)
    else:
        outcome = _axis_status(row.get("outcome_axes"), "lifecycle")
        degraded = _axis_status(row.get("outcome_axes"), "execution").lower() in ("degraded", "best_effort")
        healthy = outcome in ("", "completed", "done") and not degraded
        icon, word = ("✅" if healthy else "⚠️"), _phase_word("done", lang)
        if outcome and outcome not in ("completed", "done"):
            parts.append(outcome)
    tail = (" · " + " · ".join(parts)) if parts else ""
    label = str(row.get("target_label") or "")
    key = "project_finished" if label else "task_finished"
    key += "_details" if notice is not None and notice["neutral"] else ""
    message = _NOTIFY[lang].format(key, label=label, icon=icon, id=tid[:8], word=word, tail=tail)
    reason = notice["detail"] if notice is not None else str(row.get("reason_detail") or "")
    return "\n".join(part for part in (message, reason, str((notice or {}).get("salvage") or "")) if part)


async def _check_tasks_notify(
    api, settings: Dict[str, Any], chat_id: int, state: Dict[str, Any], lang: str,
    *, trust_env: bool = False,
) -> Tuple[Optional[BaseException], bool]:
    """Returns (last transient send failure if any, whether a send was delivered)."""
    from ouroboros.store import StoreError

    notify_all = _notify_enabled(settings, "TELEGRAM_NOTIFY_TASKS")
    try:
        summaries = _summary_ids_in_tail(api)
    except StoreError as exc:
        return exc, False  # An unknown window cannot initialize or advance the seen ids.
    if "notified_task_ids" not in state:
        # First observation → treat the existing backlog as seen
        # so enabling the toggle doesn't blast a notification for every old task.
        state["notified_task_ids"] = [tid for tid, _ in summaries][-300:]
        return None, False
    transient: Optional[BaseException] = None
    delivered = False
    seen = list(state.get("notified_task_ids") or [])
    seen_set = set(seen)
    for tid, e in summaries:
        if tid in seen_set:
            continue
        if not _announce_summary(e, notify_all=notify_all):
            # Not worth a push, but it IS seen: a later toggle flip must not blast a
            # backlog this lane was never going to announce.
            seen.append(tid)
            seen_set.add(tid)
            continue
        parts = []
        rounds = e.get("rounds")
        if rounds is not None:
            parts.append(f"{rounds}r")
        try:
            from ouroboros.task_results import load_task_result
            tr = load_task_result(_data_dir(api), tid) or {}
        except Exception:
            tr = {}
        if cost := _cost_text(tr.get("cost_presentation"), lang):
            parts.append(cost)
        msg = _completion_text(e, tid, lang, parts)
        from ouroboros.chat_chain import format_address, row_address

        event = {**e, "row_address": format_address(row_address(e))}
        send_outcome, exc = await _push_notification(api, chat_id, msg, trust_env=trust_env, event=event)
        if send_outcome == "transient":
            # Stop the batch on the first transient failure: every further
            # send would burn its full transport timeout against the same dead
            # network (a long backlog could stall one cycle for minutes before
            # the backoff even starts). State is already preserved — the
            # unsent summaries stay unseen and retry next cycle.
            transient = exc
            break
        # "sent" or permanent "skipped": either way this notification is done.
        delivered = delivered or send_outcome == "sent"
        seen.append(tid)
        seen_set.add(tid)
    state["notified_task_ids"] = seen[-300:]
    return transient, delivered


def _notifier_language() -> str:
    """The install's interface language for proactive pushes — the same reader the poller and the
    outbound mirror use, never the bridge's retired private setting."""
    return telegram_i18n.language()


def _make_notifier(api, *, trust_env: bool = False):
    """Periodic, file-based proactive notifications (task done / budget threshold).
    Read-only over durable files; sends only when a pinned chat + toggle are set."""
    async def notifier() -> None:
        retry_delay = TELEGRAM_RETRY_INITIAL_SEC
        degraded_cause = ""
        while True:
            settings = _load_settings(api)
            chat_id = _pinned_chat_id(settings)
            transient: Optional[BaseException] = None
            delivered = False
            # Both lanes self-gate. Keep observing task IDs while OFF so enabling
            # notices later does not replay completions from the disabled period.
            if chat_id:
                lang = _notifier_language()
                state = _load_notif_state(api)
                transient, delivered = await _check_budget_notify(
                    api, settings, chat_id, state, lang, trust_env=trust_env,
                )
                if transient is None:
                    # A transient budget-send failure means the transport is
                    # down right now: skip the tasks batch entirely instead of
                    # burning more transport timeouts, and let the backoff
                    # below pace the retry.
                    transient, tasks_delivered = await _check_tasks_notify(
                        api, settings, chat_id, state, lang, trust_env=trust_env,
                    )
                    delivered = delivered or tasks_delivered
                _save_notif_state(api, state)
            if transient is not None:
                # Transition logging: one warning entering degraded, one info
                # on recovery; per-attempt failures stay at debug while the
                # monotone backoff paces retries.
                if not degraded_cause:
                    degraded_cause = type(transient).__name__
                    api.log("warning", f"Telegram notifier degraded ({degraded_cause}): {transient}")
                await asyncio.sleep(retry_delay)
                retry_delay = next_telegram_retry_delay(retry_delay)
                continue
            if degraded_cause:
                # Declare recovery only on an actual delivery; when the pending
                # work merely evaporated (threshold re-armed, task id rolled out
                # of the tail, toggle flipped) end the episode silently.
                if delivered:
                    api.log("info", f"Telegram notifier recovered after {degraded_cause}.")
                degraded_cause = ""
            retry_delay = TELEGRAM_RETRY_INITIAL_SEC
            await asyncio.sleep(30)
    return notifier
