"""Bounded manifest schedule suggestions; the declarations never create schedules."""

from __future__ import annotations

from ouroboros.utils import truncate_within_limit


def suggested_schedule_fields(specs: list[dict]) -> dict:
    """The catalogue carries previews; its manifest.read points to every full declaration."""
    limits = {"name": 96, "cron": 160, "timezone": 80, "description": 260}
    return {
        "suggested_schedules": [
            {key: truncate_within_limit(" ".join(str(spec.get(key) or "").split()), limit)
             for key, limit in limits.items()}
            for spec in specs[:8]
        ],
        "suggested_schedules_omitted_count": max(0, len(specs) - 8),
    }
