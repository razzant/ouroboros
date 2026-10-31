"""Resolved Safety Supervisor settings authority and one-time upgrade persistence."""

import os


def _safety_values(settings: dict) -> dict:
    from ouroboros.settings_defaults import SETTINGS_DEFAULTS
    from ouroboros.server_runtime import authored_model_slots, propose_unauthored_model_slots, apply_runtime_provider_defaults

    values = {**SETTINGS_DEFAULTS, **settings}
    propose_unauthored_model_slots(values, authored_model_slots(settings))
    return apply_runtime_provider_defaults(values)[0]


def safety_configuration(settings: dict) -> tuple:
    """Resolved checker identity; unrelated model/catalog changes remain ordinary writes."""
    return _resolved_safety_configuration(_safety_values(settings))


def _resolved_safety_configuration(values: dict) -> tuple:
    from ouroboros.configured_subagents import parse_configured_subagents
    from ouroboros.settings_defaults import SETTINGS_DEFAULTS
    from ouroboros.settings_scales import normalize_safety_mode
    from ouroboros.model_slots import (model_role_option, MODEL_ACCOUNTS_KEY,
                                     MODEL_CONTEXT_WINDOWS_KEY, MODEL_PROCESSING_PREFERENCES_KEY)
    slot = "safety" if str(values.get("OUROBOROS_MODEL_SAFETY") or "").strip() else "light"
    if slot == "light" and not str(values.get("OUROBOROS_MODEL_LIGHT") or "").strip():
        slot = "main"
    model = str(values.get({"safety": "OUROBOROS_MODEL_SAFETY", "light": "OUROBOROS_MODEL_LIGHT",
                           "main": "OUROBOROS_MODEL"}[slot]) or SETTINGS_DEFAULTS["OUROBOROS_MODEL"]).strip()
    local = model.endswith(" (local)") or str(values.get(f"USE_LOCAL_{slot.upper()}") or "").lower() in {"true", "1", "yes", "on"}
    model = model.removesuffix(" (local)")
    options = tuple(model_role_option(key, "safety", settings=values)
                    for key in (MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_WINDOWS_KEY, MODEL_PROCESSING_PREFERENCES_KEY))
    raw = values.get("OUROBOROS_SUBAGENTS")
    recheck = ()
    if raw:
        try:
            catalog = parse_configured_subagents(raw)
            recheck = tuple((row.subagent_id, row.route, row.effort, row.processing_preference,
                             row.enabled, catalog.enabled) for row in catalog.items if row.rechecks_flags)
        except ValueError:
            recheck = ("invalid", str(raw))
    return (model, local, options, recheck,
            normalize_safety_mode(values["OUROBOROS_SAFETY_MODE"]),
            normalize_safety_mode(values["OUROBOROS_CONSCIOUSNESS_SAFETY_MODE"]),
            str(values.get("OUROBOROS_SAFETY_OWNER_RULES") or ""))


def safety_configuration_requires_owner(previous: dict, settings: dict, *, before: dict | None = None,
                                        after: dict | None = None) -> bool:
    """Judge a prepared write, with unauthored slots absent from both documents.

    Only new provider proposals for still-unauthored slots may explain a changed
    checker. Keep every other checker input at its previous resolved value, so
    this exception never authorizes an explicit model, route, mark or rules edit.
    The gateway supplies effective snapshots to retain operator environment input;
    the documents still own whether a slot was authored.
    """
    from ouroboros.settings_defaults import SETTINGS_DEFAULTS
    from ouroboros.server_runtime import provider_slot_proposals

    proposed = {**previous, **settings}
    resolved_after = safety_configuration(proposed if after is None else after)
    expected = _safety_values(previous if before is None else before)
    if _resolved_safety_configuration(expected) == resolved_after:
        return False
    proposals = provider_slot_proposals({**SETTINGS_DEFAULTS, **proposed})
    for key in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT"):
        if key not in previous and key not in settings:
            expected[key] = proposals[key]
    return _resolved_safety_configuration(expected) != resolved_after


def _guard_safety_configuration(settings: dict, *, owner_write: bool = False) -> None:
    from ouroboros import config as cfg
    rules = settings.get("OUROBOROS_SAFETY_OWNER_RULES", "")
    if not isinstance(rules, str) or len(rules.encode("utf-8")) > 64_000:
        raise ValueError("Safety owner rules must be text of at most 64,000 UTF-8 bytes")
    if owner_write or cfg.runtime_mode_at_least(cfg.get_runtime_mode(), "cyber_pro"):
        return
    previous = normalize_consciousness_safety(
        cfg._settings_integrity.read_settings_json_verified(cfg.SETTINGS_PATH) or {})
    if safety_configuration_requires_owner(previous, settings):
        raise PermissionError("The resolved Safety Supervisor is owner-controlled; use /api/owner/safety-model, safety-mode or safety-rules.")


def normalize_consciousness_safety(raw: dict) -> dict:
    """Copy saved owner coverage into an absent consciousness field before defaults.

    Pure and idempotent: explicit choices, including full, never follow later edits.
    """
    from ouroboros.settings_scales import normalize_safety_mode

    key = "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE"
    if key in raw or "OUROBOROS_SAFETY_MODE" not in raw:
        return raw
    return {**raw, key: normalize_safety_mode(raw["OUROBOROS_SAFETY_MODE"])}


def _migrate_consciousness_safety(raw: dict, *, lock_held: bool) -> dict:
    """Persist the upgrade once, on an existing document under its settings lock.

    An absent document is fresh; a pinned child snapshot is read-only. Neither
    is rewritten. Unlike raw reads, this upgrade also persists full when both fields
    are absent.
    """
    from ouroboros import config as cfg
    key = "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE"
    if key in raw or not lock_held or os.environ.get("OUROBOROS_SETTINGS_SHA256"):
        return raw
    from ouroboros.settings_scales import normalize_safety_mode
    from ouroboros.utils import append_jsonl, utc_now_iso, write_text_atomic

    cfg._guard_live_settings_write()
    migrated = {**raw, key: normalize_safety_mode(raw.get("OUROBOROS_SAFETY_MODE", "full"))}
    write_text_atomic(cfg.SETTINGS_PATH, cfg.serialize_settings(migrated))
    append_jsonl(cfg.DATA_DIR / "logs" / "events.jsonl", {
        "type": "safety_settings_migration", "actor": "migration", "ts": utc_now_iso(),
        "key": key, "value": migrated[key],
    })
    return migrated


def get_safety_mode(kind: str = "owner") -> str:
    """Return captured Supervisor coverage (full | light | off).

    Ordinary modes use the owner writer; Cyber may configure subsequent work
    through the same settings owner. Coverage is separate from Cyber's advisory
    decision authority and from the facts of a saved physical request.
    """
    from ouroboros import config as cfg
    key = "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE" if kind == "consciousness" else "OUROBOROS_SAFETY_MODE"
    default_val = str(cfg.SETTINGS_DEFAULTS[key])
    return cfg.normalize_safety_mode(cfg.runtime_setting(key, default_val) or default_val)


OWNER_RULES_KEY = "OUROBOROS_SAFETY_OWNER_RULES"
_RULES_ADDRESS = "Settings › Behavior › Safety Supervisor › Your rules"


def owner_rules_section(*, session: bool = False) -> str:
    """The owner's standing rules for an acting mind (``session``: a delegated session's start request).

    The text is the Supervisor's own read (``runtime_setting``: this actor's settings snapshot), so a
    task's context and its checks hold the same bytes; a later save reaches the next task, turn or
    session start. Empty rules read as none; an unreadable setting is a gap, never none."""
    from ouroboros import config as cfg
    head = ("\n\nOWNER RULES DELIVERY (host instructions, separate from canonical work order)\n" if session
            else "## My human's standing rules\n\n")
    try:
        rules = str(cfg.runtime_setting(OWNER_RULES_KEY, "") or "")
    except Exception as exc:
        return head + f"Unreadable now ({type(exc).__name__}); they live in {_RULES_ADDRESS}."
    if not rules.strip():
        return head + f"None set: {_RULES_ADDRESS} is empty."
    frame = (("The owner of your host set these standing rules for Ouroboros and the work it delegates, this "
              f"session included; verbatim from {_RULES_ADDRESS} as this start was prepared. Within their scope "
              "they outrank general defaults, as permission and as restriction; your assignment's explicit "
              "constraints also bind.") if session else
             (f"Verbatim from {_RULES_ADDRESS}, as this task's settings hold them; the Safety Supervisor checks "
              "my calls against the same text. Within their scope they bind in every mode (BIBLE P0) and outrank "
              "general defaults, as permission and as restriction; a place's notes add rules for that place. "
              "A change in Settings reaches the next task or turn."))
    return head + frame + "\n\n" + rules
