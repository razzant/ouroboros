#!/usr/bin/env python3
"""Shared single-model benchmark helper.

A single-model benchmark run pins every model slot to one model and gives the
review pool ``review_slots`` identical packet-delivery seats on that model
(default 1). The pool is the marked rows of ``OUROBOROS_SUBAGENTS`` — the host
never multiplies a seat, so N seats are N identical catalog rows written here.
Three identical reviewers add latency/cost but no diversity, and a single-model
run cannot achieve reviewer-model diversity anyway; the loud
``single_reviewer_no_diversity`` signal stays on. This is a BENCHMARK
convenience, NOT a claim that review got more reliable.

Delegation has the same purity boundary: the run gets one explicit ``api_model``
Available-subagent row on the measured model. It never inherits install defaults,
an API scout, or a session route. Construction and serialization deliberately use
the runtime's canonical ``OUROBOROS_SUBAGENTS`` encoder rather than a benchmark copy.

Generalized here so the SWE-bench Pro adapter (which builds a settings DICT written
to the container's settings.json) and Terminal-Bench (which mutates ``os.environ``
for a harbor subprocess) can share one definition: pass ``target=<dict>`` for the
former, leave it ``None`` for the latter.
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from ouroboros.configured_subagents import (
    MINTED_FROM_FACTORY_DEFAULT,
    PRIMARY_RECOMMENDATION,
    REVIEW_DELIVERY_PACKET,
    SUBAGENTS_SETTING,
    ConfiguredSubagent,
    ConfiguredSubagents,
    configured_subagents_dict,
    make_configured_subagents,
    normalize_configured_subagents,
    parse_configured_subagents,
    serialize_configured_subagents,
)
from ouroboros.provider_models import provider_for_model
from ouroboros.reviewer_slot_config import ROUTE_KIND_API as REVIEWER_ROUTE_KIND_API
from ouroboros.route_spec import (
    ROUTE_KIND_AGENT_SESSION,
    ROUTE_KIND_API_MODEL,
    RouteSpec,
    route_spec_dict,
)

from devtools.benchmarks.common.manifests import ACTIVE_MODEL_SLOT_KEYS, MODEL_ROUTE_OPTION_KEYS, MODEL_SLOT_KEYS

# Every model slot a single-model run pins. Superset that is correct for both the
# settings.json-profile path (SWE-bench Pro) and the forwarded-env path
# (Terminal-Bench); pinning a slot a given adapter ignores is a harmless no-op.
SINGLE_MODEL_SLOT_KEYS = (
    "OUROBOROS_MODEL",
    "OUROBOROS_MODEL_LIGHT",
    "OUROBOROS_MODEL_FALLBACKS",
    "OUROBOROS_MODEL_CONSCIOUSNESS",
    "OUROBOROS_MODEL_VISION",
    "OUROBOROS_WEBSEARCH_MODEL",
)

BENCHMARK_SUBAGENT_ID = "benchmark-model"
BENCHMARK_REVIEW_ID_PREFIX = "benchmark-review-"
BENCHMARK_REVIEW_RECOMMENDATION = (
    "Fixed-model benchmark reviewer: packet delivery on the measured model"
)
# Preserve the active manifest ordering, but compare only actual model-ID slots.
# Role account/window maps and effort metadata never enter model-list parsing.
_ACTIVE_FIXED_MODEL_KEYS = tuple(
    key for key in ACTIVE_MODEL_SLOT_KEYS if key in SINGLE_MODEL_SLOT_KEYS
)
_ACTIVE_LOCAL_ROUTE_KEYS = (
    "USE_LOCAL_MAIN",
    "USE_LOCAL_LIGHT",
    "USE_LOCAL_FALLBACK",
    "USE_LOCAL_CONSCIOUSNESS",
)


def _benchmark_actor(model: str) -> ConfiguredSubagent:
    """Canonical actor row shared by enabled and explicitly disabled benches."""
    target = str(model or "").strip()
    if not target:
        raise ValueError("single-model benchmark subagent requires a model")
    return ConfiguredSubagent(
        subagent_id=BENCHMARK_SUBAGENT_ID,
        recommended_use=PRIMARY_RECOMMENDATION,
        route=RouteSpec(ROUTE_KIND_API_MODEL, target),
    )


def _benchmark_review_rows(
    model: str, review_slots: int, review_effort: str, review_models: Sequence[str] = (),
) -> tuple[ConfiguredSubagent, ...]:
    """``review_slots`` identical review-pool seats on the measured model — or
    one seat per model of an explicit ``review_models`` panel (a benchmark whose
    methodology declares a different-model panel, e.g. GAIA ``--review-models``).

    Every seat is an ``api_model`` row marked review-eligible with PACKET delivery:
    a retrieving seat (a session, or native tool rounds) is a different delivery
    class even on the measured model — it reads the subject itself, pays for its
    own episode and evidences coverage differently — so it is not the packet
    panel every published number was produced with. The rows are minted by the
    benchmark compiler (``minted_from``), never authored by an owner, which is
    also what lets them share the actor row's engine under the save rule.
    """
    target = str(model or "").strip()
    if not target:
        raise ValueError("single-model benchmark reviewers require a model")
    effort = str(review_effort or "").strip().lower()
    targets = [str(m or "").strip() for m in review_models if str(m or "").strip()]
    if not targets:
        targets = [target] * max(1, int(review_slots))
    return tuple(
        ConfiguredSubagent(
            subagent_id=f"{BENCHMARK_REVIEW_ID_PREFIX}{index}",
            recommended_use=BENCHMARK_REVIEW_RECOMMENDATION,
            route=RouteSpec(ROUTE_KIND_API_MODEL, seat_model),
            effort=effort,
            review_eligible=True,
            delivery=REVIEW_DELIVERY_PACKET,
            minted_from=MINTED_FROM_FACTORY_DEFAULT,
        )
        for index, seat_model in enumerate(targets, start=1)
    )


def single_model_subagents_setting(
    model: str, *, review_slots: int = 1, review_effort: str = "",
    review_models: Sequence[str] = (),
) -> str:
    """Canonical Available-subagents value for a fixed-model run: the one
    measured API actor plus ``review_slots`` identical packet review seats (or
    the explicit ``review_models`` panel, one packet seat per model)."""
    rows = (_benchmark_actor(model),
            *_benchmark_review_rows(model, review_slots, review_effort, review_models))
    return serialize_configured_subagents(make_configured_subagents(rows))


def disabled_subagents_setting(
    model: str = "", *, review_slots: int = 1, review_effort: str = "",
) -> str:
    """Canonical explicit-off value, optionally retaining the measured actor row.

    A disabled list still records which API actor the benchmark measured;
    ``enabled=false`` remains the DELEGATION authority. The review pool ignores
    that switch (a marked row reviews whether or not delegation is on), so the
    measured model's review seats ride the disabled document too. The empty
    form stays available for callers that truly have no measured model rather
    than inventing one.
    """
    rows: tuple[ConfiguredSubagent, ...] = ()
    if str(model or "").strip():
        rows = (_benchmark_actor(model), *_benchmark_review_rows(model, review_slots, review_effort))
    return serialize_configured_subagents(make_configured_subagents(rows, enabled=False))


def fixed_model_roster_mismatches(config: ConfiguredSubagents, model: str) -> list[str]:
    """Why a parsed roster is NOT the exact fixed-model contract for ``model``.

    The actor row must be the canonical one-model API actor; every marked row
    (the review pool) must be an ``api_model`` packet seat on the measured model.
    Empty list means the roster is the contract (any seat count).
    """
    target = str(model or "").strip()
    actor = tuple(row for row in config.items if not row.review_eligible)
    expected = make_configured_subagents((_benchmark_actor(target),))
    mismatches: list[str] = []
    if not config.enabled or configured_subagents_dict(make_configured_subagents(actor)) != configured_subagents_dict(expected):
        mismatches.append(
            f"{SUBAGENTS_SETTING}: runtime does not expose the exact one-model "
            f"API actor for {target!r}"
        )
    pool = [row for row in config.items if row.review_eligible]
    if not pool:
        mismatches.append(f"{SUBAGENTS_SETTING}: runtime lacks the fixed-model review pool (no marked row)")
    for row in pool:
        if row.route.kind != ROUTE_KIND_API_MODEL or row.delivery == "native" or row.route.target_id != target:
            route_kind = (
                "agent_session" if row.route.kind == ROUTE_KIND_AGENT_SESSION
                else "native_tool_rounds" if row.delivery != REVIEW_DELIVERY_PACKET
                else REVIEWER_ROUTE_KIND_API
            )
            mismatches.append(
                f"{SUBAGENTS_SETTING}: review seat {row.subagent_id!r} routes via {route_kind} "
                f"to {row.route.target_id!r}; expected {REVIEWER_ROUTE_KIND_API} packet delivery on {target!r}"
            )
    return mismatches


def container_subagents_setting(model: str, host_raw: Any) -> str:
    """The roster a one-model benchmark CONTAINER runs, derived from the host's.

    A host roster that already IS the fixed-model contract for ``model`` (the
    ``--all-model`` launcher wrote it) is forwarded verbatim, seat count and
    effort included. Otherwise the container gets the canonical actor plus the
    host pool's API seats (an agent-session seat structurally cannot run in a
    task container: no harness CLI/daemon, no harness credentials in the
    forwarded env — see ``host_pool_session_targets``), and one packet seat on
    the measured model when the host configured no API seat at all.
    """
    target = str(model or "").strip()
    raw = str(host_raw or "").strip()
    if raw:
        config = parse_configured_subagents(raw)
        if not fixed_model_roster_mismatches(config, target):
            return serialize_configured_subagents(config)
        api_seats = tuple(
            row for row in config.items
            if row.review_eligible and row.route.kind == ROUTE_KIND_API_MODEL
        )
        if api_seats:
            return serialize_configured_subagents(
                make_configured_subagents((_benchmark_actor(target), *api_seats))
            )
    return single_model_subagents_setting(target)


def host_pool_session_targets(host_raw: Any) -> list[str]:
    """The host review pool's agent-session seats, as their ``harness[=model]``
    targets in row order — the seats a task container cannot run."""
    raw = str(host_raw or "").strip()
    if not raw:
        return []
    config = parse_configured_subagents(raw)
    return [
        row.route.target_id.strip() for row in config.items
        if row.review_eligible and row.route.kind == ROUTE_KIND_AGENT_SESSION
    ]


def single_model_slot_snapshot(
    model: str,
    *,
    review_slots: int = 1,
    review_effort: str = "",
) -> dict[str, str]:
    """Model-slot manifest projection derived only from the measured CLI model."""
    pinned: dict[str, str] = {}
    pin_single_model(
        model,
        review_slots=review_slots,
        review_effort=review_effort,
        target=pinned,
    )
    return {key: pinned[key] for key in SINGLE_MODEL_SLOT_KEYS if pinned.get(key)}


def runtime_actor_snapshot(
    settings: Mapping[str, Any],
    *,
    expected_model: str,
    expected_light_model: str = "",
) -> dict[str, Any]:
    """Normalize and compare the actor exposed by a target runtime's settings.

    The caller owns transport and refusal policy.  This helper owns the one canonical parser
    and comparison so ProgramBench and both OSWorld runners cannot disagree about what an
    exact fixed-model actor means.  The returned payload is non-secret and manifest-safe.
    """
    model = str(expected_model or "").strip()
    if not model:
        raise ValueError("runtime actor comparison requires an expected model")
    light_model = str(expected_light_model or "").strip() or model
    if not isinstance(settings, Mapping):
        raise ValueError("runtime settings must be an object")

    active_model_slots = {
        key: str(settings.get(key) or "").strip()
        for key in _ACTIVE_FIXED_MODEL_KEYS
        if str(settings.get(key) or "").strip()
    }
    actual_model = active_model_slots.get("OUROBOROS_MODEL", "")
    mismatches: list[str] = []
    if actual_model != model:
        mismatches.append(
            f"OUROBOROS_MODEL: runtime={actual_model!r} expected={model!r}"
        )
    # Empty optional slots mean "no alternate actor" and are therefore safe. Every non-empty
    # active slot must resolve only to its compiled actor: Light may be an explicit documented
    # helper override, while fallback, reviewer, vision and every other live role stay on Main.
    # ACTIVE_MODEL_SLOT_KEYS deliberately excludes legacy Heavy, so historical settings remain
    # readable without resurrecting it as an execution authority.
    for key, raw in active_model_slots.items():
        if key == "OUROBOROS_MODEL":
            continue
        expected = light_model if key == "OUROBOROS_MODEL_LIGHT" else model
        configured = [item.strip() for item in raw.split(",") if item.strip()]
        foreign = [item for item in configured if item != expected]
        if foreign:
            mismatches.append(
                f"{key}: runtime={raw!r} expected empty or only {expected!r}"
            )

    expected_local_routes = {
        key: provider_for_model(
            light_model if key == "USE_LOCAL_LIGHT" else model
        ) == "local"
        for key in _ACTIVE_LOCAL_ROUTE_KEYS
    }
    local_routes = {
        key: str(settings.get(key) or "").strip().lower() in {"1", "true", "yes", "on"}
        for key in _ACTIVE_LOCAL_ROUTE_KEYS
    }
    for key, uses_local in local_routes.items():
        expected_local = expected_local_routes[key]
        if uses_local != expected_local:
            routed_model = light_model if key == "USE_LOCAL_LIGHT" else model
            mismatches.append(
                f"{key}: runtime local route={uses_local!r} expected={expected_local!r} "
                f"for routed model {routed_model!r}"
            )

    # The roster is the one SSOT for both the delegation actor and the review
    # pool (its marked rows). Retired comma keys and the lane-era panel are not
    # execution authorities and are not compared.
    projection: dict[str, Any] = {}
    review_pool: list[dict[str, Any]] = []
    parse_error = ""
    try:
        config, _normalized = normalize_configured_subagents(settings.get(SUBAGENTS_SETTING))
    except ValueError as exc:
        parse_error = str(exc)
        mismatches.append(f"{SUBAGENTS_SETTING}: runtime value invalid: {exc}")
    else:
        projection = configured_subagents_dict(config)
        mismatches.extend(fixed_model_roster_mismatches(config, model))
        review_pool = [
            {
                "subagent_id": row.subagent_id,
                "route": route_spec_dict(
                    RouteSpec(row.route.kind, row.route.target_id, row.route.credential_profile_id),
                    api_kind=REVIEWER_ROUTE_KIND_API,
                    pin_key="profile_id",
                ),
                "effort": row.effort,
                "delivery": (
                    "session" if row.route.kind == ROUTE_KIND_AGENT_SESSION
                    else row.delivery or "native"
                ),
            }
            for row in config.items if row.review_eligible
        ]
    return {
        "model": actual_model,
        "model_slots": active_model_slots,
        "model_route_options": {key: settings[key] for key in MODEL_ROUTE_OPTION_KEYS if key in settings},
        "local_routes": local_routes,
        "review_pool": review_pool,
        "available_subagents": projection,
        "mismatches": mismatches,
        **({"parse_error": parse_error} if parse_error else {}),
    }


def configured_subagents_snapshot(
    settings_path: pathlib.Path | None = None,
    *,
    env_overrides: bool = True,
    exact_model: str = "",
) -> dict[str, Any]:
    """Canonical non-secret run-manifest projection of the effective actor list.

    ``exact_model`` is the post-CLI-override authority used by fixed-model launchers.
    Otherwise resolution mirrors ``model_slot_snapshot``: environment first for a
    same-process server, settings only for a fresh container. Invalid present config
    raises rather than letting a benchmark record an invented empty/default list.
    """
    if exact_model:
        raw: Any = single_model_subagents_setting(exact_model)
    else:
        settings: dict[str, Any] = {}
        if settings_path and pathlib.Path(settings_path).exists():
            loaded = json.loads(pathlib.Path(settings_path).read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("benchmark settings must be a JSON object")
            settings = loaded
        raw = os.environ.get(SUBAGENTS_SETTING) if env_overrides else None
        if raw is None:
            raw = settings.get(SUBAGENTS_SETTING)
    if raw in (None, ""):
        return {}
    return configured_subagents_dict(parse_configured_subagents(raw))


def pin_single_model(
    model: str,
    review_slots: int = 1,
    review_effort: str = "",
    target: Optional[MutableMapping[str, str]] = None,
    *,
    light_model: str = "",
) -> MutableMapping[str, str]:
    """Pin every execution route to ``model``, except an explicit Light helper.

    Reviewers, fallback and Available subagents always remain on Main.  ``light_model``
    exists only for launchers whose public methodology already exposes that override.

    ``target=None`` mutates ``os.environ`` (host-subprocess path, e.g. Terminal-Bench);
    pass a settings dict to update it instead (e.g. SWE-bench Pro ``derive_run_settings``).
    ``review_slots`` identical packet review seats ride the roster; ``review_effort``
    (when non-empty) is written only on each seat. Returns
    the mutated mapping. A single configured reviewer is intentionally loud
    (``single_reviewer_no_diversity``); this helper does not suppress that.
    """
    sink: MutableMapping[str, str] = os.environ if target is None else target
    sink.pop("OUROBOROS_MODEL_HEAVY", None)
    sink.pop("USE_LOCAL_HEAVY", None)
    for key in SINGLE_MODEL_SLOT_KEYS:
        sink[key] = model
    effective_light = str(light_model or "").strip() or model
    sink["OUROBOROS_MODEL_LIGHT"] = effective_light
    local_value = "true" if provider_for_model(model) == "local" else "false"
    for key in _ACTIVE_LOCAL_ROUTE_KEYS:
        sink[key] = local_value
    sink["USE_LOCAL_LIGHT"] = (
        "true" if provider_for_model(effective_light) == "local" else "false"
    )
    sink[SUBAGENTS_SETTING] = single_model_subagents_setting(
        model, review_slots=review_slots, review_effort=review_effort,
    )
    # Lane-era carriers a previous pin may have left behind are not execution
    # authorities any more; drop them so the forwarded contract is the roster alone.
    for key in set(MODEL_SLOT_KEYS) - set(ACTIVE_MODEL_SLOT_KEYS):
        if key.startswith("OUROBOROS_"):
            sink.pop(key, None)
    return sink


def fixed_model_actor_snapshot(
    model: str,
    *,
    light_model: str = "",
    review_slots: int = 1,
    review_effort: str = "",
    target: Optional[MutableMapping[str, str]] = None,
) -> dict[str, Any]:
    """Compile one execution mapping and return its complete manifest-safe actor.

    Callers pass the SAME mapping they will hand to the subprocess.  This closes
    the provenance gap where launchers recorded three model strings while ambient
    local/reviewer routes still governed execution.  A fresh mapping is used only
    when a root manifest needs the contract but a child wrapper owns execution.
    """
    sink: MutableMapping[str, str] = {} if target is None else target
    pin_single_model(
        model,
        review_slots=review_slots,
        review_effort=review_effort,
        target=sink,
        light_model=light_model,
    )
    snapshot = runtime_actor_snapshot(
        sink,
        expected_model=model,
        expected_light_model=light_model,
    )
    if snapshot["mismatches"]:
        raise RuntimeError(
            "fixed-model actor compiler produced an inconsistent contract: "
            + "; ".join(str(item) for item in snapshot["mismatches"])
        )
    return snapshot
