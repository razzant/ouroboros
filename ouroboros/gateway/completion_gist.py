"""Short completion lines at the browser boundary; stored and model-read gists stay whole."""
from __future__ import annotations

from ouroboros.text_sentences import first_sentence


def short_gist(row: dict) -> dict:
    """Project a task, room row or project portal without changing its source document."""
    if isinstance(row.get("wake_summary"), str) and row["wake_summary"]:
        return {**row, "wake_summary": first_sentence(row["wake_summary"])}
    return row


def short_gist_frame(message: dict) -> dict:
    """Live chat facts sit on the frame; task_done facts sit inside a log frame's data."""
    if message.get("type") == "log" and isinstance(message.get("data"), dict):
        return {**message, "data": short_gist(message["data"])}
    return short_gist(message)


def short_schedule_gists(payload: dict) -> dict:
    """The next scheduled run reads the full cached outcome; only its UI row is shortened. An answer that lists no
    rows (an unavailable reader's) passes as it is."""
    if "tasks" not in payload:
        return payload
    rows = []
    for row in payload.get("tasks", ()):
        for key in ("outcome", "last_outcome"):
            outcome = row.get(key) or {}
            if isinstance(outcome.get("line"), str) and not outcome.get("notice"):
                row = {**row, key: {**outcome, "line": first_sentence(outcome["line"])}}
        rows.append(row)
    return {**payload, "tasks": rows}
