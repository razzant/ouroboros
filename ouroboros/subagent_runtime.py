"""Immutable configured-subagent selection for scheduling and exact starts.

This module consumes :mod:`ouroboros.configured_subagents`; it deliberately does
not parse either the new setting or the bounded legacy settings itself.  A task
copies the returned snapshot once and every later dispatch/recovery decision reads
that copy rather than mutable owner settings.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
from dataclasses import dataclass, replace as dataclass_replace
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Mapping, Optional

from ouroboros.configured_subagents import (
    ConfiguredSubagent,
    ConfiguredSubagents,
    ConfiguredSubagentsResolution,
    SESSION_ACCESS_PROFILES,
    SESSION_ACCESS_LOWERING,
    SOURCE_INVALID,
    SOURCE_LEGACY_MIGRATED,
    SOURCE_UNDECIDED,
    configured_subagents_fingerprint,
    resolve_configured_subagents,
    resolve_roster_selector,
    roster_handles,
)
from ouroboros.delegate_shared import delegate_payload
from ouroboros.route_spec import RouteSpec, model_named_effort, route_spec_dict
from ouroboros.runtime_mode_policy import effort_range_binds
from ouroboros.settings_integrity import SETTINGS_ENV_LOCK, TaskSettingsSnapshot, live_effort_range, runtime_setting
from ouroboros.settings_scales import choose_effort, effort_fact, effort_range
from ouroboros.subagent_history import snapshot_handle
from ouroboros.tools.tool_result import ToolResult, _replace_tool_result
from ouroboros.utils import utc_now_iso


# These are the only inputs read by the bounded legacy compiler.  At task start
# provider normalization has already projected the EFFECTIVE values into the
# process environment.  Overlaying exactly these keys prevents a later raw disk
# read from resurrecting a shipped Heavy default which normalization cleared.
_RUNTIME_LEGACY_KEYS = (
    "OUROBOROS_SUBAGENT_HARNESS",
    "OUROBOROS_SUBAGENT_PROFILE",
    "OUROBOROS_MODEL_HEAVY",
    "OUROBOROS_MODEL_LIGHT",
    "OUROBOROS_MODEL",
    "USE_LOCAL_HEAVY",
    "USE_LOCAL_LIGHT",
    "USE_LOCAL_MAIN",
    "LOCAL_MODEL_SOURCE",
)

# Call-stack carrier only.  The selected row itself is durable in the task and
# start records; ContextVar keeps concurrent tool calls from mutating ToolContext.
_EXACT_START_SELECTION: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "delegate_exact_start_selection", default={}
)


@dataclass(frozen=True)
class SubagentSelectionError(ValueError):
    """Typed scheduling/start refusal, safe to project directly to a tool result."""

    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


def effective_runtime_subagent_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay normalized task-start legacy values without rereading raw defaults."""

    effective = dict(settings)
    for key in _RUNTIME_LEGACY_KEYS:
        # Absence is meaningful: apply_settings_to_env removes a normalized-empty
        # setting, so retaining the raw disk value here would undo normalization.
        effective[key] = runtime_setting(key, "")
    return effective


def model_visible_subagent_catalog(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Project saved, dispatchable rows as facts, without probing or ranking them.

    Facts only: how to choose among rows is the mind's own prose
    (``prompts/SYSTEM.md`` §Delegation), never a host-authored sentence here.
    ``subagent_id`` carries the row's handle — the value the tools accept; the
    stored key and the list fingerprint stay host-side in snapshots.
    """

    resolution = resolve_configured_subagents(settings)
    config = resolution.config
    if (
        resolution.source in {SOURCE_INVALID, SOURCE_UNDECIDED}
        or config is None
        or not config.enabled
        or not config.items
    ):
        return {}

    handles = roster_handles(config, settings)
    rows: list[dict[str, Any]] = []
    for row in config.items:
        # An owner-disabled row keeps its saved configuration and stays
        # editable in Settings; it is simply not offered for a NEW selection.
        if not row.enabled:
            continue
        session = row.route.is_session
        handle = handles[row.subagent_id]
        # The row's effort as the mind may act on it: ``auto`` (the range decides, recommended by
        # default), the owner's pin, or the level its model name carries (``effort_source``).
        named = model_named_effort(row.route)
        projected: dict[str, Any] = {
            "subagent_id": handle,
            "route_class": "Agent session" if session else "API model",
            "effort": named or row.effort or "auto",
            **({"effort_source": "model_name"} if named else {}),
        }
        if row.route.target_id != handle:  # a bare handle already IS the target
            projected["requested_target" if session else "requested_model"] = row.route.target_id
        if session:
            projected["mutating_access"] = row.access
            if row.route.credential_profile_id:
                projected["credential_profile_id"] = row.route.credential_profile_id
        # The review pool is a mark on catalog rows: whether THIS row also
        # reviews my changes, and — for an api row — how it receives the subject
        # (``native``: reads with its own tools; ``packet``: the assembled brief).
        projected["review_eligible"] = row.review_eligible
        if row.review_eligible and not session:
            projected["delivery"] = row.delivery or "native"
        # The owner's words, verbatim and last: bounded intent, never a title.
        projected["recommended_use"] = row.recommended_use
        rows.append(projected)
    if not rows:
        return {}
    return {"rows": rows}


def current_model_visible_subagent_catalog() -> dict[str, Any]:
    """Read the current normalized settings and return the stable catalog."""

    from ouroboros.config import runtime_settings

    return model_visible_subagent_catalog(
        effective_runtime_subagent_settings(runtime_settings())
    )


_REVIEW_RULES = {
    "cyber_pro": "Cyber Pro: review informs judgment; no finding, failure or unavailable review prohibits action.",
    "blocking": ("Blocking: critical findings, a failed quorum, a review that was not performed or a review "
                 "infrastructure failure stop the commit."),
    "advisory": ("Advisory: material findings and review failures return to the author as the outcome before any Git "
                 "effect; the author may continue explicitly on that outcome, and nothing is rewritten into PASS."),
}


_REVIEW_SURFACES = {
    "commit_gate": "every pool row outside Cyber Pro; composed from the pool in Cyber Pro (reason recorded)",
    "plan_review": "every pool row",
    "task_acceptance": {"root": "every pool row", "child": "≤1 from the pool; with several, name one as reviewer"},
    "preflight": "one enabled catalog row you name: commit_reviewed(preflight_reviewer=…); none → recorded as not performed",
    "system_review": "/review: one enabled catalog row, default Main",
}

# A session seat is paid in subscription time, not route money (THESIS2 §3).
SESSION_SEAT_COST_HINT = "uses a session seat and time"
COST_UNKNOWN_HINT = "cost unknown"


def review_call_cost_text(usd: Optional[float], *, reading: bool) -> str:
    """An api row's price in words: one full call (``review_helpers.review_row_call_usd``).
    Settings → Agents prints the same words (``subagents_settings.js`` ``reviewCostText``),
    the dollars rounded as JavaScript's ``toFixed`` does."""
    if usd is None:
        return COST_UNKNOWN_HINT
    if not usd:
        return "no API cost per review"
    places = Decimal("0.0001") if usd < 0.01 else Decimal("0.01")
    text = f"≈${Decimal(usd).quantize(places, rounding=ROUND_HALF_UP)} per full call (route tariff)"
    return text + ("; a reading reviewer makes several" if reading else "")


def _api_review_cost_hint(slot: Any) -> str:
    """One api seat's price (:func:`review_call_cost_text`), from the tariff and the window
    already held in this process — context assembly never waits on a provider catalog. A
    model call through a subscription uses a seat, as Settings → Agents says."""
    from ouroboros.provider_models import provider_for_model
    from ouroboros.tools.review_helpers import review_row_call_usd

    if provider_for_model(str(getattr(slot, "model", "") or "")) == "claudexor":
        return SESSION_SEAT_COST_HINT
    return review_call_cost_text(review_row_call_usd(slot, allow_live_fetch=False),
                                 reading=bool(getattr(slot, "native_retrieval", False)))


# The wizard's summary prints the same sentence (``subagents_settings.js`` ``PACKET_ONLY_POOL_WARNING``).
PACKET_ONLY_POOL_WARNING = (
    "Every reviewer is a Packet row, so none reads the repository and the coupling question (how the "
    "change fits the rest of the code) goes unanswered: with Blocking review, every commit to Ouroboros "
    "itself stops as “not performed”. Mark a reviewer that reads the work itself, or choose Advisory.")


def coupling_unanswerable(slots: Any) -> bool:
    """A non-empty pool none of whose seats reads the repository (every row Packet): no seat
    is asked the coupling part, so every commit-gate review ends ``NOT_PERFORMED``
    (``coupling_not_performed``, ``review_ledger.reduce_verdict``) — a block under Blocking."""
    return bool(slots) and not any(getattr(slot, "retrieves", False) for slot in slots)


def review_pool_save_warning(settings: Mapping[str, Any]) -> str:
    """The save-time warning, never a refusal, for a document whose pool cannot answer the
    coupling part while review blocks (:func:`coupling_unanswerable`); ``""`` otherwise."""
    from ouroboros.reviewer_slot_config import review_pool_slots
    from ouroboros.tools.review_helpers import review_enforcement_blocks

    enforcement = str(settings.get("OUROBOROS_REVIEW_ENFORCEMENT") or "").strip().lower()
    try:
        warn = review_enforcement_blocks(enforcement) and coupling_unanswerable(review_pool_slots(dict(settings)))
    except Exception:  # a malformed catalog has its own refusal; a warning never fails a landed save
        import logging

        logging.getLogger(__name__).debug("review pool save warning unavailable", exc_info=True)
        return ""
    return PACKET_ONLY_POOL_WARNING if warn else ""


def _review_migration_facts(settings: Mapping[str, Any]) -> dict[str, str]:
    """The A↔C seam: ``config.review_pool_migrations_seen()`` is the tuple of
    ``review_pool_migration.MigrationOutcome`` records this process's settings reads
    computed, oldest first (a no-op — the catalog was already a pool — leaves no fact).
    The block shows the newest one that decided THIS document — the settings view whose
    pool the block shows (``review_pool_receipts.outcome_decides_document``, the same
    predicate the owner's receipt applies), never another document's: its ``error`` (a
    refused migration keeps the lane keys in the document, so the pool the owner expects
    does not exist until the catalog is saved) and the ``snapshot`` path the supervisor
    boot recorded for that document (``server_maintenance.review_pool_migration_records``),
    when it has written one. Either key is present only when it has a value. The process
    registry itself is history and is never trimmed here."""
    from ouroboros import config as cfg
    from ouroboros.review_pool_receipts import outcome_decides_document

    seen = getattr(cfg, "review_pool_migrations_seen", None)
    outcomes = [outcome for outcome in (seen() if callable(seen) else ())
                if outcome_decides_document(outcome, settings)]
    if not outcomes:
        return {}
    latest = outcomes[-1]
    facts = {"error": str(getattr(latest, "error", "") or ""), "snapshot": ""}
    try:
        from ouroboros.server_maintenance import review_pool_migration_records

        record = review_pool_migration_records().get(str(getattr(latest, "input_sha256", "") or ""))
        facts["snapshot"] = str((record or {}).get("snapshot") or "") if isinstance(record, Mapping) else ""
    except Exception:  # the boot's receipt is a pointer; the block never fails on it
        import logging

        logging.getLogger(__name__).debug("review pool migration records unavailable for the ## Review block",
                                          exc_info=True)
    return {key: value for key, value in facts.items() if value}


def review_facts_block(snapshot: Optional[TaskSettingsSnapshot] = None) -> str:
    """``## Review``: the review POOL this task's settings snapshot serves — the enabled catalog rows the owner
    marked as reviewers, read through the same builder every review surface runs (``review_pool_slots``), never
    live settings or the last execution; ``None`` reads the bound task scope. Each seat is named by its catalog
    handle (the same handle ``## Available subagents`` shows and ``review_change(reviewers=[…])`` accepts); its
    ``seat_id`` is the stored row id the records carry. Above four seats the rows shrink to ``{seat_id, model}``
    and ``omitted`` counts them; ``coupling_unanswerable: true`` (:func:`coupling_unanswerable`) is a block fact,
    so the shrink keeps it. ``source`` is ``structured`` (a non-empty pool), ``empty`` (a readable catalog
    with no marked row — a loud, configured fact, never a default panel) or ``error`` (a malformed catalog or a
    refused migration). Stable for the task (cache-marked prefix); the task's recent ledger records ride
    separately in the changing part (:func:`review_records_block`)."""
    from ouroboros import reviewer_slot_config as rs
    from ouroboros.config import get_review_enforcement, get_runtime_mode, runtime_settings, task_settings_scope
    from ouroboros.configured_subagents import SUBAGENTS_SETTING, parse_configured_subagents
    from ouroboros.provider_models import model_has_credentials_in_settings
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.runtime_mode_policy import runtime_mode_at_least
    from ouroboros.tools.review_helpers import review_enforcement_blocks

    with task_settings_scope(snapshot) if snapshot is not None else contextlib.nullcontext():
        settings = effective_runtime_subagent_settings(runtime_settings())
        state = rs.review_pool_state(settings.get(SUBAGENTS_SETTING))
        source, error = state["state"], state["error"]
        pool: list[dict[str, Any]] = []
        slots: list[Any] = []
        without_credentials: list[str] = []
        if source != "error":
            try:
                raw = settings.get(SUBAGENTS_SETTING)
                roster = parse_configured_subagents(raw) if str(raw or "").strip() else None
                handles = roster_handles(roster, settings) if roster is not None else {}
                slots = rs.review_pool_slots(settings)
            except ValueError as exc:  # e.g. a marked session row without a concrete harness
                source, error, slots = "error", str(exc), []
            for slot in slots:
                session = slot.route is ReviewRouteKind.AGENT_SESSION
                pool.append({
                    "seat_id": slot.slot_id,
                    "subagent_id": handles.get(slot.subagent_id, slot.subagent_id),
                    "model": slot.model, "effort": slot.effort,
                    "delivery": "session" if session else ("native" if slot.native_retrieval else "packet"),
                    "cost_hint": SESSION_SEAT_COST_HINT if session else _api_review_cost_hint(slot),
                })
                # A session seat logs in itself; an api seat answers only with its provider's key.
                if not session and not model_has_credentials_in_settings(slot.model, dict(settings)):
                    without_credentials.append(pool[-1]["subagent_id"])
        migration = _review_migration_facts(settings)
        if migration.get("error") and source != "error":
            source, error = "error", migration["error"]
        unanswerable = source == "structured" and coupling_unanswerable(slots)
        enforcement, mode = get_review_enforcement(), get_runtime_mode()
        blocks = review_enforcement_blocks(enforcement)
    seats = len(pool)
    if seats > 4:
        pool = [{"seat_id": row["seat_id"], "model": row["model"]} for row in pool]
    facts: dict[str, Any] = {
        "source": source, "error": error,
        "enforcement": enforcement, "enforcement_blocks": blocks, "mode": mode,
        # The rule is the EXECUTION rule of the effective authority (PR-1): what a
        # finding or a failed review does to the change. Who reviews is ``pool``
        # and ``surfaces``, never this sentence.
        "rule": _REVIEW_RULES["cyber_pro" if runtime_mode_at_least(mode, "cyber_pro") else enforcement],
        "pool": pool,
        "pool_empty": source == "empty",
        **({"coupling_unanswerable": True} if unanswerable else {}),
        # VD3-08: the seats this install holds no credentials for (the fact Settings shows
        # from the same payload field); every seat of the pool is the loud one.
        **({"pool_without_credentials": without_credentials} if without_credentials else {}),
        **({"no_pool_row_has_credentials": True} if without_credentials and len(without_credentials) == seats else {}),
        "surfaces": _REVIEW_SURFACES,
        "omitted": {"rows": seats if seats > 4 else 0},
        "full_source": {"pool": "GET /api/review-pool", "records": "## Review records"},
    }
    if migration.get("snapshot"):
        facts["migration_snapshot"] = migration["snapshot"]
    return "## Review\n\n" + json.dumps(facts, ensure_ascii=False, separators=(",", ":"))


def review_records_block(*, drive_root: Any, task_id: str) -> str:
    """``## Review records``: this task's newest review-ledger records (:func:`_recent_review_records`),
    a changing fact that belongs in the dynamic context part, never in the cached prefix."""
    records, unseen = _recent_review_records(drive_root, task_id)
    return "## Review records\n\n" + json.dumps({
        "recent_records": records, "omitted": {"records": unseen}, "full_source": "state/review_ledger/",
    }, ensure_ascii=False, separators=(",", ":"))


def _recent_review_records(drive_root: Any, task_id: str) -> tuple[list[dict[str, Any]], Any]:
    """This task's five newest review-ledger records from the bounded hot index, in the reader's
    newest-first order, and whether more exist: ``0`` when the hot index holds nothing else and no
    archived segment exists, ``"1+"`` when it holds more than the five shown, ``"unknown"`` when an
    archived segment exists (older records of this task may live there; a context capture never
    opens the archive) or the ledger is unreadable — never a silent zero. No ledger module means
    none. An empty id reads nothing: the reader's empty selector is every task's records."""
    try:
        from ouroboros.review_ledger import archived_segments_exist, recent_records

        rows = recent_records(drive_root, task_id=task_id, limit=6, hot_only=True) if task_id else []
        shown = [{"record_id": row.get("record_id"), "surface": row.get("surface"),
                  "aggregate": (row.get("verdict") or {}).get("aggregate"), "ts": row.get("ts")} for row in rows[:5]]
        more: Any = "1+" if len(rows) > 5 else ("unknown" if task_id and archived_segments_exist(drive_root) else 0)
    except ModuleNotFoundError as exc:
        return [], 0 if exc.name == "ouroboros.review_ledger" else "unknown"
    except Exception:
        return [], "unknown"
    return shown, more


def apply_task_start_settings() -> TaskSettingsSnapshot:
    """Capture a task's normalized projection before publishing the process view."""
    from ouroboros import config
    from ouroboros.server_runtime import apply_runtime_provider_defaults
    from ouroboros.settings_integrity import task_settings_snapshot

    fd = config._acquire_settings_lock()
    try:
        with SETTINGS_ENV_LOCK:
            effective, _changed, _keys = apply_runtime_provider_defaults(
                config.load_settings_lock_held(_settings_lock_held=fd is not None))
            projected = dict(os.environ)
            config.apply_settings_to_env(effective, environ=projected)
            snapshot = task_settings_snapshot(effective, projected)
            config.apply_settings_to_env(effective)
            return snapshot
    finally:
        config._release_settings_lock(fd)


def apply_task_start_settings_or_disclose(task_id: str, emit_live_log: Any) -> TaskSettingsSnapshot:
    """Task-start settings reload with a LOUD failure path (#285).

    A silent failure breaks the save-time promise "the saved changes apply
    from the next task": the task would run on the previously applied
    configuration with nobody told. The task itself stays runnable
    (fail-open), but the breakage becomes a visible live-log fact.

    The common corruption case is probed explicitly: ``load_settings`` falls
    back to defaults+env on an unreadable or malformed settings.json instead
    of raising, which would keep exactly the silence this wrapper exists to
    break. A MISSING file is legitimate (defaults-only install), not a fault.
    """
    from ouroboros.settings_integrity import task_settings_snapshot

    from ouroboros.config import SETTINGS_DEFAULTS, settings_env_keys

    with SETTINGS_ENV_LOCK:
        previous_env = dict(os.environ)
    previous_settings = dict(SETTINGS_DEFAULTS)
    previous_settings.update({key: previous_env.get(key, "") for key in settings_env_keys()})
    previous = task_settings_snapshot(previous_settings, previous_env)
    try:
        from ouroboros import config as _config

        try:
            raw_settings_text = _config.SETTINGS_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            raw_settings_text = None
        if raw_settings_text is not None:
            json.loads(raw_settings_text)
        return apply_task_start_settings()
    except Exception as exc:
        import logging

        logging.getLogger(__name__).error(
            "Task-start settings reload failed; this task uses the environment from the previously applied configuration; document-only values are unavailable",
            exc_info=True,
        )
        emit_live_log(
            "task_start_settings_reload_failed",
            task_id=task_id,
            error=f"{type(exc).__name__}: {exc}",
            message=("Settings reload failed at task start: this task uses the environment "
                     "from the previously applied configuration; document-only values "
                     "could not be recovered."),
        )
    return previous


def _resolution(
    settings: Mapping[str, Any], *, allow_undecided_legacy: bool,
) -> ConfiguredSubagentsResolution:
    resolution = resolve_configured_subagents(settings)
    if resolution.source == SOURCE_INVALID:
        raise SubagentSelectionError(
            "subagent_configuration_invalid",
            resolution.diagnostic or "Available subagents are not configured.",
        )
    if resolution.source == SOURCE_UNDECIDED and not allow_undecided_legacy:
        raise SubagentSelectionError(
            "subagent_configuration_unsaved",
            "Available subagents have not been saved; choose them in onboarding or Settings first.",
        )
    if resolution.config is None:
        raise SubagentSelectionError(
            "subagent_selection_required",
            "The legacy selector does not identify a migrated configured subagent.",
        )
    if not resolution.config.enabled:
        raise SubagentSelectionError(
            "subagents_disabled", "Available subagents are disabled by owner configuration."
        )
    if not resolution.config.items:
        raise SubagentSelectionError(
            "no_available_subagents", "The enabled Available-subagents list is empty."
        )
    return resolution


def _legacy_matches(
    resolution: ConfiguredSubagentsResolution,
    *,
    model_lane: str,
    executor: str,
) -> list[ConfiguredSubagent]:
    """Return only rows the bounded legacy migration can identify exactly.

    ``auto`` is intentionally no filter.  It never selects by list order or live
    health.  New configured rows are not reverse-engineered into the retired axes;
    the compatibility seam is for the canonical legacy-migration projection only.
    """

    if resolution.source not in {SOURCE_LEGACY_MIGRATED, SOURCE_UNDECIDED}:
        return []
    rows = [row for row in (resolution.config.items if resolution.config else ()) if row.enabled]
    if executor == "harness":
        rows = [row for row in rows if row.route.is_session]
    elif executor == "native":
        rows = [row for row in rows if not row.route.is_session]
    if model_lane == "heavy":
        rows = [row for row in rows if row.subagent_id.startswith("legacy-heavy")]
    elif model_lane == "light":
        rows = [row for row in rows if row.subagent_id.startswith("fast-scout")]
    elif model_lane == "main":
        # The bounded compiler has no Main identity row.  Guessing by model value
        # would recreate mutable slot semantics after the list became authoritative.
        rows = []
    return rows


def resolve_configured_row(
    config: ConfiguredSubagents, selector: str, settings: Mapping[str, Any],
) -> ConfiguredSubagent:
    """The one ``subagent_id`` argument resolver (a handle, else a stored id) for scheduling
    and exact starts; whether the row may take NEW work is asked after resolution."""
    row, code, detail = resolve_roster_selector(config, selector, settings)
    if row is None:
        raise SubagentSelectionError(code, detail)
    return row


def select_subagent_snapshot(
    settings: Mapping[str, Any],
    *,
    subagent_id: str = "",
    legacy_model_lane: Any = None,
    legacy_executor: Any = None,
    legacy_model_lane_supplied: bool = False,
    legacy_executor_supplied: bool = False,
    access: Optional[str] = None,
) -> tuple[dict[str, Any], bool]:
    """Resolve one row and return ``(immutable_snapshot, used_legacy_seam)``.

    The snapshot is JSON-only so the queue/result stores can copy it losslessly.
    Availability is deliberately not embedded here: saved intent is immutable;
    dispatch/start records its dated live observation in a separate task field.
    """
    from ouroboros.model_slots import resolve_processing_preference

    selected_id = str(subagent_id or "").strip()
    has_legacy = bool(legacy_model_lane_supplied or legacy_executor_supplied)
    if selected_id and has_legacy:
        raise SubagentSelectionError(
            "subagent_selector_conflict",
            "subagent_id cannot be combined with legacy model_lane/executor selectors.",
        )
    resolution = _resolution(settings, allow_undecided_legacy=has_legacy)
    config = resolution.config
    assert config is not None
    used_legacy = False
    if selected_id:
        # Resolve FIRST, then ask the row: a switched-off row is refused the same
        # way by its handle and by its stored id. Typed and distinct from the
        # list-level `subagents_disabled` and from a live-availability refusal:
        # the row exists and is fully configured, the owner has switched it off
        # for new work. Never a substitute actor.
        row = resolve_configured_row(config, selected_id, settings)
        if not row.enabled:
            raise SubagentSelectionError(
                "subagent_disabled",
                f"Configured subagent {selected_id!r} is switched off in Available "
                "subagents; its configuration is kept. Choose an enabled subagent_id, "
                "or turn that row back on in Settings.",
            )
    else:
        lane = str(legacy_model_lane or "auto").strip().lower() or "auto"
        executor = str(legacy_executor or "auto").strip().lower() or "auto"
        if not has_legacy or (lane == "auto" and executor == "auto"):
            raise SubagentSelectionError(
                "subagent_selection_required",
                "Choose an explicit subagent_id; omitted/auto legacy selectors are ambiguous.",
            )
        matches = _legacy_matches(resolution, model_lane=lane, executor=executor)
        if len(matches) != 1:
            raise SubagentSelectionError(
                "subagent_selection_required",
                f"Legacy selectors mapped to {len(matches)} configured rows; choose subagent_id explicitly.",
            )
        row = matches[0]
        used_legacy = True

    return validate_subagent_snapshot({
        "schema": 1,
        "selected_subagent_id": row.subagent_id,
        "config_fingerprint": configured_subagents_fingerprint(config),
        "recommended_use": row.recommended_use,
        "source": resolution.source,
        "route": route_spec_dict(
            row.route, api_kind="api_model", pin_key="credential_profile_id"
        ),
        "effort": row.effort,
        "processing_preference": resolve_processing_preference(
            override=row.processing_preference or None, settings=dict(settings)),
        **({"access": row.access} if row.route.is_session else {}),
        "selected_at": utc_now_iso(),
    }, access=access), used_legacy


def validate_subagent_snapshot(raw: Any, *, access: Optional[str] = None) -> dict[str, Any]:
    """Validate captured intent and optionally lower it, never consult live settings."""

    snapshot = dict(raw) if isinstance(raw, dict) else {}
    route = snapshot.get("route") if isinstance(snapshot.get("route"), dict) else {}
    kind = str(route.get("kind") or "")
    target = str(route.get("target_id") or "").strip()
    captured_access = snapshot.get("access", "workspace_write")
    if (
        int(snapshot.get("schema") or 0) != 1
        or not str(snapshot.get("selected_subagent_id") or "").strip()
        or not str(snapshot.get("config_fingerprint") or "").strip()
        or kind not in {"api_model", "agent_session"}
        or not target
        or captured_access not in (*SESSION_ACCESS_PROFILES, "readonly")
        or ("access" in snapshot and kind != "agent_session")
    ):
        raise SubagentSelectionError(
            "subagent_snapshot_invalid", "The task has no complete immutable subagent snapshot."
        )
    if access not in (None, "inherit", *SESSION_ACCESS_LOWERING):
        raise SubagentSelectionError(
            "subagent_access_invalid",
            f"access={access!r} for subagent_id={snapshot_handle(snapshot)!r} "
            f"({kind}) must be inherit, readonly or workspace_write; inherit preserves "
            "the configured session access, and API-model access is controlled by write_surface.")
    # API actors derive authority from write_surface; a populated session-only
    # option must not make that route unreachable for all-fields tool forms.
    if kind == "agent_session" and access in SESSION_ACCESS_LOWERING:
        if access == "readonly" or captured_access == "full":
            snapshot["access"] = access
    from ouroboros.model_slots import normalize_processing_preference

    try:
        # An old durable snapshot captures legacy behavior, never today's global setting.
        snapshot["processing_preference"] = normalize_processing_preference(
            snapshot.get("processing_preference"))
    except ValueError as exc:
        raise SubagentSelectionError("subagent_snapshot_invalid", str(exc)) from exc
    return snapshot


def resolve_configured_actor_dispatch(
    task: Mapping[str, Any], *, task_type: str,
) -> Any:
    """Resolve a frozen row into the existing ``SubagentDispatch`` contract."""

    # Lazy import avoids a module cycle: subagents calls this only after defining
    # the shared dispatch dataclasses and route helpers used below.
    from ouroboros.config import resolve_effort
    from ouroboros.provider_models import model_has_credentials
    from ouroboros.subagents import (
        CapabilityDelta,
        DelegationRoute,
        SubagentDispatch,
        SubagentExecutorResolution,
        SubagentLaneResolution,
        delegated_run_shape,
        derive_capability_reason,
        parse_subagent_harness,
        route_health,
    )
    from ouroboros.tools.control_delegation import profile_from_task_constraint

    snapshot = validate_subagent_snapshot(task.get("configured_subagent"))
    route_spec = snapshot["route"]
    route_kind = str(route_spec.get("kind") or "")
    route_target = str(route_spec.get("target_id") or "").strip()
    # THE effort decision for this child, from the row (the owner's pin, a level in the model
    # name) and the parent's request, inside the range this task started with (``choose_effort``).
    requested_effort = str(task.get("requested_effort") or "")
    derived_effort, effort_source = choose_effort(
        requested_effort, pin=snapshot.get("effort"),
        model_named=model_named_effort(RouteSpec(route_kind, route_target)),
        binds=effort_range_binds(task.get("metadata")), rng=effort_range())
    fact = effort_fact(requested_effort, derived_effort, effort_source)
    constraint = task.get("task_constraint") if isinstance(task.get("task_constraint"), dict) else {}
    profile = profile_from_task_constraint(constraint)
    observed_at = utc_now_iso()
    lane = SubagentLaneResolution(
        requested_lane="auto", effective_lane="main", model="",
        resolved_from="main", provenance="configured_subagent",
    )

    if route_kind == "api_model":
        use_local = route_target.endswith(" (local)")
        model = route_target[:-8].strip() if use_local else route_target
        available = bool(model_has_credentials(route_target))
        unavailable = "" if available else "credentials_unavailable"
        executor = "native" if available else "blocked"
        lane = dataclass_replace(lane, model=model, use_local_model=use_local)
        availability = {
            "observed_at": observed_at,
            "status": "ready" if available else "credentials_unavailable",
            "reason": unavailable,
            "route_kind": route_kind,
            "selected_subagent_id": str(snapshot.get("selected_subagent_id") or ""),
            "alternatives": (
                [] if available else current_subagent_alternatives(
                    str(snapshot.get("selected_subagent_id") or "")
                )
            ),
            "host_fallback": False,
        }
        reasons = []
        if unavailable:
            reasons.append(unavailable)
        delta = CapabilityDelta(
            derived_effort=derived_effort,
            effective_effort=derived_effort,
            requested_executor="native", effective_executor=executor,
            reason=derive_capability_reason(reasons), reduced=bool(reasons),
            reduction_reasons=tuple(reasons),
        )
        return SubagentDispatch(
            lane=lane, effort=derived_effort, executor=executor,
            route=route_target if available else "",
            profile=profile, delta=delta,
            executor_resolution=SubagentExecutorResolution(
                "native", executor, reason=unavailable or "requested_native",
            ),
            availability=availability, effort_fact=fact,
        )

    parsed = parse_subagent_harness(route_target)
    if parsed is None:
        unavailable, reset_at = "configured_session_route_invalid", ""
        exact_route = DelegationRoute(route_id="")
    else:
        # The LEAF runs the decided level; the nanny below keeps the parent's.
        exact_route = dataclass_replace(
            parsed, effort=derived_effort,
            profile_id=str(route_spec.get("credential_profile_id") or ""),
        )
        from ouroboros.contracts.task_constraint import normalize_task_constraint
        from ouroboros.tool_access import predicted_subagent_profile

        normalized = normalize_task_constraint(task.get("task_constraint"))
        surface = str(getattr(normalized, "surface", "") or "")
        shape = delegated_run_shape(
            predicted_subagent_profile(write_surface=surface) == "acting_subagent",
            snapshot.get("access", "workspace_write"),
        )
        gateway = None
        try:
            from ouroboros.claudexor_daemon import ensure_owned_gateway

            gateway = ensure_owned_gateway()
            unavailable, reset_at = route_health(
                gateway, exact_route.route_id, shape, route_model=exact_route.model,
                pinned_profile=exact_route.profile_id,
            )
        except Exception as exc:
            unavailable, reset_at = (
                str(getattr(exc, "code", "") or type(exc).__name__),
                str(getattr(exc, "reset_at", "") or ""),
            )
        finally:
            if gateway is not None:
                gateway.close()

    cognitive = task.get("parent_cognitive_route") if isinstance(task.get("parent_cognitive_route"), dict) else {}
    nanny_model = str(cognitive.get("model") or "").strip()
    nanny_effort = str(cognitive.get("effort") or "").strip().lower()
    if not nanny_model and not unavailable:
        unavailable = "parent_cognitive_route_missing"
    if not nanny_effort:
        nanny_effort = resolve_effort(task_type or str(task.get("type") or "task"))
    lane = dataclass_replace(
        lane, model=nanny_model,
        use_local_model=bool(cognitive.get("use_local_model")),
    )
    executor = "blocked" if unavailable else "harness"
    availability = {
        "observed_at": observed_at,
        "status": "unavailable" if unavailable else "ready",
        "reason": unavailable, "reset_at": reset_at, "route_kind": route_kind,
    }
    reasons = []
    if unavailable:
        reasons.append(unavailable)
    delta = CapabilityDelta(
        derived_effort=nanny_effort,
        effective_effort=nanny_effort,
        requested_executor="harness", effective_executor=executor,
        reason=derive_capability_reason(reasons), reduced=bool(reasons),
        reduction_reasons=tuple(reasons),
    )
    resolution = SubagentExecutorResolution(
        "harness", executor, exact_route if exact_route.route_id else None,
        unavailable or "harness_ready", reset_at,
    )
    return SubagentDispatch(
        lane=lane, effort=nanny_effort, executor=executor,
        route=route_target if executor == "harness" else "", profile=profile,
        delta=delta, executor_resolution=resolution, availability=availability,
        effort_fact=fact,
    )


def current_subagent_alternatives(exclude_id: str = "") -> list[dict[str, Any]]:
    """Project the current saved choices without ranking or probing them."""

    try:
        from ouroboros.config import runtime_settings

        settings = effective_runtime_subagent_settings(runtime_settings())
        resolution = resolve_configured_subagents(settings)
        config = resolution.config
        if config is None or not config.enabled:
            return []
        handles = roster_handles(config, settings)
    except Exception:
        return []
    excluded = str(exclude_id or "")
    return [
        {
            "subagent_id": handles[row.subagent_id],
            "recommended_use": row.recommended_use,
            "route_kind": row.route.kind,
            "target_id": row.route.target_id,
            "effort": row.effort,
            **({"mutating_access": row.access} if row.route.is_session else {}),
            "availability": "check_at_dispatch",
        }
        for row in config.items
        if row.subagent_id != excluded and row.enabled
    ]


def exact_session_binding(raw_snapshot: Any, leaf_effort: Optional[str] = None) -> tuple[dict[str, Any], Any]:
    """Validate one snapshotted session row and construct its exact route; ``leaf_effort``
    is the decided level (``choose_effort``) the run starts at, else the row's own pin."""

    snapshot = validate_subagent_snapshot(raw_snapshot)
    route_spec = snapshot["route"]
    if str(route_spec.get("kind") or "") != "agent_session":
        raise SubagentSelectionError(
            "api_actor_requires_schedule_subagent",
            "The selected row is an API actor. Create it as an ordinary recursive child with "
            "schedule_subagent(subagent_id=...), not as a Claudexor leaf.",
        )
    from ouroboros.subagents import parse_subagent_harness

    route = parse_subagent_harness(route_spec.get("target_id"))
    if route is None:
        raise SubagentSelectionError(
            "configured_session_route_invalid", "The selected session route is invalid."
        )
    return snapshot, dataclass_replace(
        route,
        effort=str(snapshot.get("effort") or route.effort) if leaf_effort is None else str(leaf_effort),
        profile_id=str(route_spec.get("credential_profile_id") or ""),
    )


def prepare_delegate_start_actor(
    ctx: Any,
    drive_root: Any,
    *,
    recovering: bool,
    invocation_id: str,
    work_order_fingerprint: str,
    authority_fingerprint: str,
    continuing: str = "",
) -> tuple[dict[str, Any], Optional["ToolResult"]]:
    """Resolve the exact actor/start fence; ``continuing`` names the run a continuation takes over."""

    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_recovery import unsettled_start_ids
    from ouroboros.delegate_shared import _fail
    selection = dict(_EXACT_START_SELECTION.get() or {})
    selected_snapshot = selection.get("snapshot")
    if recovering:
        if selected_snapshot:
            return {}, _fail(
                "delegate_start", "retry_selector_conflict",
                "retry_of replays its already-bound immutable route and cannot accept a new "
                "subagent selector.",
            )
        invocation = custody.invocation_record(drive_root, invocation_id) or {}
        return {
            "selected_subagent_id": str(invocation.get("selected_subagent_id") or ""),
            "config_fingerprint": str(invocation.get("config_fingerprint") or ""),
            "work_order_fingerprint": str(
                invocation.get("work_order_fingerprint") or work_order_fingerprint
            ),
            "authority_fingerprint": str(
                invocation.get("authority_fingerprint") or authority_fingerprint
            ),
        }, None

    if selected_snapshot is None:
        return {}, _fail(
            "delegate_start", "subagent_selection_required",
            "A fresh delegated start requires an explicit agent_session subagent_id. "
            "Only retry_of may replay a selectorless immutable invocation.",
        )
    fact = selection.get("effort_fact") if isinstance(selection.get("effort_fact"), dict) else {}
    snapshot, route = exact_session_binding(selected_snapshot, leaf_effort=fact.get("applied") if fact else None)
    selected_id = str(snapshot.get("selected_subagent_id") or "")
    config_fingerprint = str(snapshot.get("config_fingerprint") or "")
    if custody.custody_log_unreadable(drive_root):
        return {}, _fail(
            "delegate_start", "replacement_custody_unknown",
            "The custody event log exists but cannot be read, so the host cannot "
            "prove that no physical start/run remains. Repair or reconcile the "
            "canonical custody record before starting a replacement.",
        )
    blockers = unsettled_start_ids(
        drive_root, str(getattr(ctx, "task_id", "") or ""), continuing=str(continuing or "")
    )
    if any(blockers.values()):
        live_ids = [str(v) for key in ("open_run_ids", "pending_invocation_ids",
                                       "undisposed_patch_run_ids")
                    for v in (blockers.get(key) or [])]
        shown = ", ".join(live_ids[:4]) + (
            f" (+{len(live_ids) - 4} more)" if len(live_ids) > 4 else "")
        return {}, _fail(
            "delegate_start", "replacement_requires_settlement",
            "This task still owns an unsettled run/start invocation or an undisposed "
            f"captured patch ({shown}). Wait for or cancel an open run; replay a "
            "pending invocation with retry_of=<invocation id>; dispose a captured "
            "patch explicitly (#364).",
            **blockers,
        )
    return {
        "route": route,
        "access": snapshot.get("access", "workspace_write"),
        "selected_subagent_id": selected_id,
        "processing_preference": str(snapshot.get("processing_preference") or ""),
        "config_fingerprint": config_fingerprint,
        "work_order_fingerprint": work_order_fingerprint,
        "authority_fingerprint": authority_fingerprint,
        "compiled_work_order": bool(selection.get("compiled_work_order")),
        # The row's pin ("" = Auto) stays the receipt identity; the decided level is the fact.
        "row_effort": str(snapshot.get("effort") or ""),
        "effort_fact": dict(fact),
    }, None


def exact_start(ctx: Any, prompt: str, spec: Optional[dict[str, Any]] = None) -> "ToolResult":
    """Shared exact-start primitive for actor-first sessions and root-direct calls."""

    options = dict(spec or {})
    retry_token = str(options.get("retry_of") or "").strip()
    bootstrap = getattr(ctx, "_configured_actor_bootstrap", None)
    recovering = False
    if retry_token and isinstance(bootstrap, dict):
        from ouroboros import delegate_custody as custody

        invocation = custody.invocation_record(custody.custody_root(ctx), retry_token) or {}
        recovering = (
            str(invocation.get("state") or "") == "pending"
            and str(invocation.get("task_id") or "")
            == str(getattr(ctx, "task_id", "") or "")
        )
    if isinstance(bootstrap, dict) and not bool(bootstrap.get("zero_run_receipt_recorded")):
        from ouroboros.subagent_bootstrap import _durable_zero_run_receipt

        zero_run_evidence_gaps: set[str] = set()
        durable_zero_run = _durable_zero_run_receipt(
            ctx, gap_reasons=zero_run_evidence_gaps,
        )
        if durable_zero_run:
            bootstrap.update({
                "zero_run_receipt_recorded": True,
                "zero_run_decision": str(durable_zero_run.get("zero_run_decision") or ""),
                "zero_run_basis": str(durable_zero_run.get("zero_run_basis") or ""),
                "exact_start_pending": False,
            })
            bootstrap.pop("zero_run_evidence_status", None)
            bootstrap.pop("zero_run_evidence_gaps", None)
        elif zero_run_evidence_gaps and not recovering:
            bootstrap.update({
                "zero_run_evidence_status": "unknown",
                "zero_run_evidence_gaps": sorted(zero_run_evidence_gaps),
                "exact_start_pending": False,
            })
    if isinstance(bootstrap, dict) and bool(bootstrap.get("zero_run_receipt_recorded")):
        from ouroboros.delegate_shared import _fail

        return _fail(
            "delegate_start", "zero_run_already_recorded",
            "This actor already recorded a terminal delegation_zero_run decision; "
            "starting a physical leaf now would contradict that durable receipt. "
            "Create a new explicitly bound task/retry after the parent disposes the result.",
            zero_run_decision=str(bootstrap.get("zero_run_decision") or ""),
        )
    if (
        not recovering
        and isinstance(bootstrap, dict)
        and str(bootstrap.get("zero_run_evidence_status") or "") == "unknown"
    ):
        from ouroboros.delegate_shared import _fail

        return _fail(
            "delegate_start", "zero_run_evidence_unavailable",
            "The host cannot prove whether this actor already recorded a terminal "
            "delegation_zero_run decision. Starting another physical leaf would "
            "risk duplicating the same logical invocation. Reconcile the durable "
            "receipt evidence or record a new typed zero-run decision.",
            zero_run_evidence_status="unknown",
            zero_run_evidence_gaps=list(
                bootstrap.get("zero_run_evidence_gaps") or []
            ),
        )
    selected_id = str(options.pop("subagent_id", "") or "").strip()
    selected_snapshot = options.pop("snapshot", None)
    effort_request = options.pop("effort", None)
    effort_choice = options.pop("effort_fact", None)
    compiled_work_order = bool(options.pop("compiled_work_order", False))
    canonical_work_order_fingerprint = str(
        options.pop("work_order_fingerprint", "") or ""
    ).strip()
    coordination_context = str(options.pop("_coordination_context", "") or "")
    work_order_source_request = options.pop("work_order_source_request", None)
    access = options.pop("access", None)
    try:
        if str(options.get("retry_of") or "").strip() and (
            selected_id or selected_snapshot is not None or access is not None
        ):
            raise SubagentSelectionError(
                "retry_selector_conflict",
                "retry_of replays its already-bound immutable route and cannot accept a new "
                "subagent selector or access choice.",
            )
        if selected_id and selected_snapshot is not None:
            raise SubagentSelectionError(
                "subagent_selector_conflict", "subagent_id cannot be combined with a snapshot."
            )
        if not str(options.get("retry_of") or "").strip() and not selected_id and selected_snapshot is None:
            raise SubagentSelectionError(
                "subagent_selection_required",
                "A fresh delegated start requires subagent_id; only retry_of replays without it.",
            )
        if selected_id:
            from ouroboros.config import runtime_settings

            selected_snapshot, _legacy = select_subagent_snapshot(
                effective_runtime_subagent_settings(runtime_settings()),
                subagent_id=selected_id,
            )
        if selected_snapshot is not None:
            selected_snapshot = validate_subagent_snapshot(selected_snapshot, access=access)
        effort_choice = _leaf_effort_choice(ctx, selected_snapshot, effort_request, effort_choice, retry_token)
    except SubagentSelectionError as exc:
        from ouroboros.delegate_shared import _fail

        # Selector validation has not entered the physical-start producer yet.
        return _replace_tool_result(_fail("delegate_start", exc.code, exc.detail),
                                    meta_updates={"operation_outcome": "completed_no_effect"})

    token = _EXACT_START_SELECTION.set({
        "snapshot": selected_snapshot,
        "compiled_work_order": compiled_work_order,
        "effort_fact": effort_choice,
    })
    try:
        from ouroboros.tools.delegate import _delegate_start

        result = _delegate_start(
            ctx, prompt, options.pop("max_seconds", None), options.pop("retry_of", None),
            root=options.pop("root", None), bucket=options.pop("bucket", None),
            skill_name=options.pop("skill_name", None),
            _resolved_binding=options.pop("_resolved_binding", None),
            _canonical_work_order_fingerprint=canonical_work_order_fingerprint,
            _work_order_source_request=work_order_source_request,
            _coordination_context=coordination_context,
            **{key: options.pop(key) for key in (
                "directory_strategy", "scope_paths", "continue_from", "continue_carrier") if key in options},
        )
        # Every configured-session start lands here — the host's pre-start
        # (charter, owner 2026-08-28/29) and any model-issued retry/replacement
        # alike. Mark the physical start from the host-owned typed result,
        # including the uncustodied branch: a run may already be live even when
        # its durable custody row could not be written. Without this marker the
        # episode could mint a false zero-run receipt after a successful start
        # and nanny economics would miss real activity.
        _mark_actor_physical_start(ctx, result)
        payload = delegate_payload(result)
        if isinstance(selected_snapshot, dict):
            # Model-facing name: the snapshot's own handle; custody keeps the stored key.
            payload["selected_subagent_id"] = snapshot_handle(selected_snapshot)
            payload["config_fingerprint"] = str(
                selected_snapshot.get("config_fingerprint") or ""
            )
        if isinstance(work_order_source_request, dict):
            payload["work_order_source_request"] = dict(work_order_source_request)
        return _replace_tool_result(
            result, text=json.dumps(payload, ensure_ascii=False, indent=2))
    except SubagentSelectionError as exc:
        from ouroboros.delegate_shared import _fail

        return _fail("delegate_start", exc.code, exc.detail)
    finally:
        _EXACT_START_SELECTION.reset(token)


def _leaf_effort_choice(ctx: Any, snapshot: Any, request: Any, choice: Any, retry_token: str) -> dict[str, Any]:
    """The delegated LEAF's effort fact for this start, ONE carrier from the request to the
    start body (``prepare_delegate_start_actor`` binds the route to its ``applied`` level).

    A configured session hands over the fact its dispatch decided; a direct start decides
    now (``choose_effort`` over the row's pin, its model-named level and the request, against
    the owner's CURRENT range); a retry replays the stored body — ``auto``/omission is fine,
    another concrete level is a typed conflict. An unknown tier is a typed argument refusal.
    """
    from ouroboros.tools.control_subagent_spec import requested_child_effort

    requested, error = requested_child_effort(request, "delegate_start")
    if error:
        raise SubagentSelectionError("effort_invalid", error.split(": ", 1)[-1])
    if retry_token:
        if not requested:
            return {}
        from ouroboros import delegate_custody as custody

        stored = custody.invocation_record(custody.custody_root(ctx), retry_token) or {}
        body = stored.get("request") if isinstance(stored.get("request"), dict) else {}
        if str(body.get("effort") or "") == requested:
            return {}
        raise SubagentSelectionError(
            "retry_selector_conflict",
            "retry_of replays its recorded effort; omit effort (or pass auto) to retry.")
    if isinstance(choice, dict) and choice and not requested:
        return dict(choice)
    if not isinstance(snapshot, dict):
        return {}
    route = snapshot.get("route") if isinstance(snapshot.get("route"), dict) else {}
    level, source = choose_effort(
        requested, pin=snapshot.get("effort"),
        model_named=model_named_effort(RouteSpec(str(route.get("kind") or ""), str(route.get("target_id") or ""))),
        binds=effort_range_binds(getattr(ctx, "task_metadata", None)), rng=live_effort_range())
    return effort_fact(requested, level, source)


def _mark_actor_physical_start(ctx: Any, result: "ToolResult") -> None:
    """Record a successful actor-first physical start on the private bootstrap fact.

    This is deliberately an internal projection, not a new lifecycle ABI.  The
    durable custody/evidence rows remain authoritative; the projection only keeps
    the current host episode from treating a started (possibly uncustodied) run as
    a zero-run and seeds the existing nanny-activity accounting.
    """
    bootstrap = getattr(ctx, "_configured_actor_bootstrap", None)
    if not isinstance(bootstrap, dict):
        return
    # The DOMAIN status, never the host class: a started run is `started` (or
    # `started_uncustodied`), and a refusal that classified `ok` would otherwise
    # mint a physical-start marker over a run that never began.
    status = str(delegate_payload(result).get("status") or "")
    if status not in {"started", "started_uncustodied"}:
        return
    bootstrap["physical_started"] = True
    bootstrap["exact_start_pending"] = False
    bootstrap["physical_start_status"] = status
    ctx._nanny_physical_activity_seed = True


def _names_bound_actor(selector: str, bootstrap: Mapping[str, Any]) -> bool:
    """Whether a ``subagent_id`` argument names the actor this episode is bound to.

    The bound row is the frozen snapshot, so its stored id and its own handle
    (the one the startup receipt shows) always name it; any other value goes
    through the same resolver as ``schedule_subagent`` against the live roster.
    """
    expected_id = str(bootstrap.get("selected_subagent_id") or "")
    snapshot = bootstrap.get("snapshot") if isinstance(bootstrap.get("snapshot"), dict) else {}
    if selector in {expected_id, snapshot_handle(snapshot)}:
        return True
    try:
        from ouroboros.config import runtime_settings

        settings = effective_runtime_subagent_settings(runtime_settings())
        config = _resolution(settings, allow_undecided_legacy=False).config
        return config is not None and resolve_configured_row(
            config, selector, settings).subagent_id == expected_id
    except SubagentSelectionError:
        return False


def delegate_start_entry(ctx: Any, prompt: str, _resolved_binding: Any = None, **params: Any) -> "ToolResult":
    # Actor-first configured sessions bind every fresh start to the immutable
    # snapshot captured before the episode. The model supplies only an advisory
    # coordination appendix; the canonical work order remains host-owned.
    bootstrap = getattr(ctx, "_configured_actor_bootstrap", None)
    retry_of = str(params.get("retry_of") or "").strip()
    from ouroboros.delegate_shared import _fail

    def _blocked(reason: str) -> None:
        # A pre-custody refusal is still a durable ATTEMPT fact: without the
        # START_BLOCKED row, the evidence read "delegate_start never called"
        # over a call the registry provably saw (D5 evidence collapse). One
        # emitter for the whole refusal class, not per-branch.
        from ouroboros.delegate_evidence import record_start_blocked

        record_start_blocked(ctx, str(getattr(ctx, "task_id", "") or ""), reason)

    if isinstance(bootstrap, dict):
        # A fresh start and a retry alike stay bound to the frozen snapshot: an
        # argument the bound start cannot honor is refused typed, never
        # silently discarded.
        expected_id = str(bootstrap.get("selected_subagent_id") or "")
        requested_id = str(params.get("subagent_id") or "").strip()
        if retry_of and params.get("access") is not None:
            _blocked("retry_selector_conflict")
            return _fail("delegate_start", "retry_selector_conflict",
                         "A retry replays its recorded access; omit access.")
        if requested_id and not _names_bound_actor(requested_id, bootstrap):
            _blocked("configured_actor_route_mismatch")
            return _fail(
                "delegate_start", "configured_actor_route_mismatch",
                "This configured session is bound to its scheduled actor, fresh start and "
                "retry alike; select another actor with schedule_subagent instead.",
                selected_subagent_id=expected_id,
                requested_subagent_id=requested_id,
                host_fallback=False,
            )
        if any(str(params.get(key) or "").strip() for key in ("root", "bucket", "skill_name")):
            _blocked("configured_actor_resource_mismatch")
            return _fail(
                "delegate_start", "configured_actor_resource_mismatch",
                "A configured session starts only its assigned route, fresh start and "
                "retry alike, never a skill-payload resource.",
                selected_subagent_id=expected_id,
                host_fallback=False,
            )
    if isinstance(bootstrap, dict) and not retry_of:
        canonical_work_order = str(bootstrap.get("canonical_work_order") or "")
        if not canonical_work_order:
            _blocked("configured_work_order_unavailable")
            return _fail(
                "delegate_start", "configured_work_order_unavailable",
                "The canonical work order is unavailable; do not start a physical leaf "
                "from a prefix. Recover the task's complete chosen assignment.",
                work_order_fingerprint=str(bootstrap.get("work_order_fingerprint") or ""),
                work_order_chars=int(bootstrap.get("work_order_chars") or 0),
                host_fallback=False,
            )
        bound = dict(params)
        bound.pop("subagent_id", None)
        bound.update({
            "snapshot": dict(bootstrap.get("snapshot") or {}),
            "compiled_work_order": True,
            "work_order_fingerprint": str(bootstrap.get("work_order_fingerprint") or ""),
            "_coordination_context": str(prompt or ""),
            # The dispatch's effort decision reaches the first physical start unchanged.
            "effort_fact": dict(bootstrap.get("effort_fact") or {}),
            **{key: bootstrap[key] for key in ("directory_strategy", "scope_paths")
               if key in bootstrap},
        })
        if _resolved_binding is not None:
            bound["_resolved_binding"] = _resolved_binding
        return exact_start(ctx, canonical_work_order, bound)
    if retry_of and isinstance(bootstrap, dict):
        # Retry replays the stored canonical request byte-for-byte.
        from ouroboros import delegate_custody as custody

        invocation = custody.invocation_record(custody.custody_root(ctx), retry_of) or {}
        request = invocation.get("request") if isinstance(invocation.get("request"), dict) else {}
        canonical_work_order = str(request.get("prompt") or "")
        if not canonical_work_order:
            _blocked("configured_work_order_unavailable")
            return _fail(
                "delegate_start", "configured_work_order_unavailable",
                "The retry has no recorded work order; the coordination prompt "
                "cannot replace the original assignment.",
                work_order_fingerprint=str(bootstrap.get("work_order_fingerprint") or ""),
                work_order_chars=int(bootstrap.get("work_order_chars") or 0),
                host_fallback=False,
            )
        retry_spec = {
            "retry_of": retry_of,
            "_resolved_binding": _resolved_binding,
        }
        return exact_start(ctx, canonical_work_order, retry_spec)
    return exact_start(ctx, prompt, {**params, "_resolved_binding": _resolved_binding})


__all__ = [
    "SubagentSelectionError",
    "apply_task_start_settings",
    "current_model_visible_subagent_catalog",
    "current_subagent_alternatives",
    "delegate_start_entry",
    "effective_runtime_subagent_settings",
    "exact_session_binding",
    "exact_start",
    "model_visible_subagent_catalog",
    "prepare_delegate_start_actor",
    "resolve_configured_actor_dispatch",
    "review_facts_block", "review_records_block",
    "select_subagent_snapshot",
    "validate_subagent_snapshot",
]
