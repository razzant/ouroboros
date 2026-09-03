"""Telegram bridge: periodic push notifications (task done / budget thresholds)."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, Tuple

from .telegram_api import (
    TELEGRAM_RETRY_INITIAL_SEC,
    TelegramClient,
    TelegramRequestRejected,
    TelegramTransportError,
    next_telegram_retry_delay,
)
from .telegram_state import (
    _data_dir,
    _jsonl_tail,
    _load_runtime_state,
    _load_settings,
    _read_json_file,
    _state_file,
)


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
    api, chat_id: int, text: str, *, trust_env: bool = False,
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
    client = TelegramClient(protected.get("TELEGRAM_BOT_TOKEN", ""), trust_env=trust_env)
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
        msg = (f"⚠️ Бюджет: {pct:.0f}% (${spent:.2f} / ${total:.2f})" if lang == "ru"
               else f"⚠️ Budget: {pct:.0f}% (${spent:.2f} / ${total:.2f})")
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
    ids = []
    for e in rows:
        if str(e.get("type") or "") == "task_summary" and e.get("task_id"):
            ids.append((str(e.get("task_id")), e))
    return ids


async def _check_tasks_notify(
    api, settings: Dict[str, Any], chat_id: int, state: Dict[str, Any], lang: str,
    *, trust_env: bool = False,
) -> Tuple[Optional[BaseException], bool]:
    """Returns (last transient send failure if any, whether a send was delivered)."""
    if not _notify_enabled(settings, "TELEGRAM_NOTIFY_TASKS"):
        return None, False
    summaries = _summary_ids_in_tail(api)
    if "notified_task_ids" not in state:
        # First run with task notifications on → treat the existing backlog as seen
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
        parts = []
        rounds = e.get("rounds")
        if rounds is not None:
            parts.append(f"{rounds}r")
        tr = _read_json_file(_data_dir(api) / "task_results" / f"{tid}.json")
        try:
            if tr.get("cost_usd") is not None:
                parts.append(f"${float(tr.get('cost_usd')):.2f}")
        except (TypeError, ValueError):
            pass
        oa = e.get("outcome_axes")
        outcome = _axis_status(oa, "lifecycle")
        degraded = _axis_status(oa, "execution").lower() in ("degraded", "best_effort")
        if outcome and outcome not in ("completed", "done"):
            parts.append(outcome)
        tail = (" · " + " · ".join(parts)) if parts else ""
        healthy = outcome in ("", "completed", "done") and not degraded
        icon = "✅" if healthy else "⚠️"
        msg = (f"{icon} Задача {tid[:8]} готова{tail}" if lang == "ru" else f"{icon} Task {tid[:8]} done{tail}")
        send_outcome, exc = await _push_notification(api, chat_id, msg, trust_env=trust_env)
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


def _make_notifier(api, *, trust_env: bool = False):
    """Periodic, file-based proactive notifications (task done / budget threshold).
    Read-only over durable files; sends only when a pinned chat + toggle are set."""
    async def notifier() -> None:
        retry_delay = TELEGRAM_RETRY_INITIAL_SEC
        degraded_cause = ""
        while True:
            settings = _load_settings(api)
            chat_id = _pinned_chat_id(settings)
            want = _notify_enabled(settings, "TELEGRAM_NOTIFY_TASKS") or _notify_enabled(
                settings,
                "TELEGRAM_NOTIFY_BUDGET",
            )
            transient: Optional[BaseException] = None
            delivered = False
            if chat_id and want:
                lang = str(settings.get("TELEGRAM_LANGUAGE") or "en").strip().lower()
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
