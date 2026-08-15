"""Closed settings vocabularies, their orderings, and pure value transforms.

Extracted from ``ouroboros/config.py`` so that module stays what its docstring
claims — paths, defaults, load/save with locking — and stays under the module
size gate. Everything here is PURE: stdlib only, no environment reads, no
settings file access, no import of ``config`` (which imports this). That is the
whole boundary: a closed enum, a rank over it, or a transform of an already-read
value belongs here; anything that resolves a CONFIGURED value belongs in
``config``.

``config`` re-exports these names, so the long-standing ``config.EFFORT_SCALE``
/ ``config.migrate_legacy_slot_keys`` spellings keep working.
"""

from __future__ import annotations

# v6.57.0 — EFFORT_SCALE: ORDERED reasoning-effort SSOT (low→high), the single place a tier
# is defined (settings, llm.py builder, switch_model enum, subagent lanes). xhigh/max extend
# none..high; llm.py clamps a request DOWN to each model's learned ceiling (BIBLE P1: disclosed).
EFFORT_SCALE: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

# Runtime mode and review enforcement are separate axes.
VALID_RUNTIME_MODES = ("light", "advanced", "pro")

# Context mode is an independent, owner-controlled working-context size profile
# (low/max). Unlike runtime mode it is NOT boot-pinned — it is not a privilege
# boundary, so it hot-applies on the next task.
VALID_CONTEXT_MODES = ("low", "max")

VALID_SAFETY_MODES = ("full", "light", "off")

# Lower rank = stricter scope. ``config.save_settings`` refuses agent self-elevation.
RUNTIME_MODE_RANK = {"light": 0, "advanced": 1, "pro": 2}

# ``full -> light -> off`` is a strictly decreasing LLM-safety coverage ladder.
SAFETY_MODE_RANK = {"full": 2, "light": 1, "off": 0}

# v6.39 slot rename-alias migration (same shape as the retention-key rename):
# OUROBOROS_MODEL_CODE -> _HEAVY, USE_LOCAL_CODE -> USE_LOCAL_HEAVY,
# OUROBOROS_MODEL_FALLBACK -> _FALLBACKS.
LEGACY_SLOT_RENAMES = (
    ("OUROBOROS_MODEL_CODE", "OUROBOROS_MODEL_HEAVY"),
    ("OUROBOROS_VISION_MODEL", "OUROBOROS_MODEL_VISION"),
    ("USE_LOCAL_CODE", "USE_LOCAL_HEAVY"),
    ("OUROBOROS_MODEL_FALLBACK", "OUROBOROS_MODEL_FALLBACKS"),
)


def effort_rank(value: str) -> int:
    """Index of an effort in EFFORT_SCALE (−1 if unknown). Strength-ordering SSOT."""
    v = str(value or "").strip().lower()
    return EFFORT_SCALE.index(v) if v in EFFORT_SCALE else -1


def clamp_effort_to(value: str, ceiling: str) -> str:
    """Clamp ``value`` down to ``ceiling`` on EFFORT_SCALE; unknown inputs pass through."""
    vi, ci = effort_rank(value), effort_rank(ceiling)
    return ceiling if (vi >= 0 and ci >= 0 and vi > ci) else str(value or "").strip().lower()


def effort_one_step_down(value: str) -> str:
    """Next-lower effort on EFFORT_SCALE (reject-and-retry walk); floors at `none`."""
    idx = effort_rank(value)
    return EFFORT_SCALE[idx - 1] if idx > 0 else ("none" if idx == 0 else "medium")


def parse_model_list(value: str) -> list[str]:
    """Split a comma-separated model slot into trimmed, non-empty entries."""
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def migrate_legacy_slot_keys(settings: dict) -> dict:
    """In-place settings migration, applied BEFORE defaults are merged.

    Preserves a stored value (never orphans an owner customization), then drops the legacy
    key. Shared SSOT for every settings entry point (load_settings AND the Colab builder).
    Order matters: the singular scope-review pin is promoted HERE, before ``SETTINGS_DEFAULTS``
    supplies the plural that WINS in get_scope_review_models."""
    for _old, _new in LEGACY_SLOT_RENAMES:
        if _new not in settings and _old in settings:
            settings[_new] = settings[_old]
        settings.pop(_old, None)
    _pin = str(settings.get("OUROBOROS_SCOPE_REVIEW_MODEL") or "").strip()
    if _pin and not str(settings.get("OUROBOROS_SCOPE_REVIEW_MODELS") or "").strip():
        settings["OUROBOROS_SCOPE_REVIEW_MODELS"] = _pin
    return settings
