"""One-window compatibility for retired persistent context auto-Low state."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from ouroboros.utils import write_text_atomic

log = logging.getLogger(__name__)

VALID_CONTEXT_MODES = ("low", "max")
_MIGRATION_WARNED_PATHS: set[str] = set()


def normalize_context_mode(value: Any) -> str:
    """Clamp caller-supplied context mode to the closed enum."""
    text = str(value or "").strip().lower()
    return text if text in VALID_CONTEXT_MODES else "max"


def owner_declared_low(value: Any) -> bool:
    """Whether the false tombstone proves an owner-authored Low."""
    return str(value if value is not None else "").strip().lower() in {"0", "false", "off"}


def normalize_context_mode_compat(
    raw: dict,
    *,
    settings_path: Path | None = None,
    warn_ambiguous: bool = False,
) -> dict:
    """Normalize the retired auto-Low pair before defaults or coercion."""
    normalized = dict(raw or {})
    mode_key = "OUROBOROS_CONTEXT_MODE"
    marker_key = "OUROBOROS_CONTEXT_MODE_AUTO_LOW"
    if mode_key not in normalized and marker_key not in normalized:
        return normalized
    raw_mode = normalize_context_mode(normalized.get(mode_key))
    marker_is_false = owner_declared_low(normalized.get(marker_key))
    ambiguous_low = raw_mode == "low" and not marker_is_false
    normalized[mode_key] = "low" if raw_mode == "low" and marker_is_false else "max"
    normalized[marker_key] = "false"
    warning_key = str((settings_path or Path("<memory>")).resolve(strict=False))
    if ambiguous_low and warn_ambiguous and warning_key not in _MIGRATION_WARNED_PATHS:
        _MIGRATION_WARNED_PATHS.add(warning_key)
        log.warning(
            "Legacy context settings contained Low without an explicit false owner "
            "provenance marker; normalized Low to Max. Re-select Low in Settings "
            "if Low is the intended owner choice."
        )
    return normalized


def normalize_and_persist_context_mode_compat(
    raw: dict,
    *,
    settings_path: Path,
    lock_held: bool,
    guard_live_write: Callable[[], None],
) -> dict:
    """Normalize raw settings and persist the canonical pair while its lock is held.

    The write is the raw mapping with only the pair changed — never a defaults-merged
    document — in the one on-disk spelling every settings writer produces: the bytes of
    ``config.serialize_settings`` through the byte-exact ``write_text_atomic``."""
    normalized = normalize_context_mode_compat(
        raw, settings_path=settings_path, warn_ambiguous=True,
    )
    missing = object()
    changed = any(
        raw.get(key, missing) != normalized.get(key, missing)
        for key in ("OUROBOROS_CONTEXT_MODE", "OUROBOROS_CONTEXT_MODE_AUTO_LOW")
    )
    if not changed:
        return normalized
    if not lock_held:
        log.warning(
            "Context settings were normalized in memory, but the canonical compatibility "
            "pair was not persisted because the settings lock was unavailable; migration "
            "will retry on the next load."
        )
        return normalized
    from ouroboros.config import serialize_settings  # bound late: config imports this module

    try:
        guard_live_write()
        write_text_atomic(settings_path, serialize_settings(normalized))
    except Exception:
        log.warning(
            "Context settings were normalized in memory, but the canonical compatibility "
            "pair could not be persisted; migration will retry on the next load.",
            exc_info=True,
        )
    return normalized
