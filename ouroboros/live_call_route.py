"""The realtime route of a live call and what the host knows about it before the call opens.

A call's engine route is resolved BEFORE the turn's context is fitted, because the
fit must measure against the realtime window and the realtime reply allowance, not
Main's. Those numbers are not a table the host trusts blindly (BIBLE: external
capabilities are sourced, auditable evidence): a known realtime route's facts are
written into the capability-evidence store as what they are, documented facts with
their source and read date, never as metadata the provider confirmed on the wire,
so the ordinary context fit, the record and a later audit all read the same fact
with the same provenance; an unknown route stays unknown and the fit keeps its own
unknown-window behaviour. The fit against the route (``call_context_plan``) is the
whole fit: a setup the provider still refuses ends the call with the provider's
words. The call's initial Nano tool view and the owner's duration ceiling live here
too, so the agent (context preparation) and the driver (the session) read one answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.call_protocol import CallSpec
from ouroboros.gateways.realtime_engine import ENGINE_GOOGLE, GOOGLE_INPUT_RATE, GOOGLE_OUTPUT_RATE, EngineRoute
from ouroboros.tool_capabilities import COGNITIVE_MEMORY_TOOL_NAMES

log = logging.getLogger(__name__)

# The call's initial Nano view: memory of the mind, the web, and the room's chat (what
# reads better than it sounds — links, code, lists — the voice writes there itself);
# read_file, because every retained host section the call view points at is a
# runtime_data read and a Gemini session reopens to gain a tool; and start_task,
# because long work leaves the call as a task whose result reaches it through live
# awareness. Everything else is one ``enable_tools`` away (growth reaches the engine as ``UpdateTools``).
CALL_INITIAL_TOOL_NAMES: Tuple[str, ...] = tuple(sorted(
    COGNITIVE_MEMORY_TOOL_NAMES | {"web_search", "send_user_message", "read_file", "start_task"}))
REALTIME_EVIDENCE_KIND = "realtime_windows"   # capability_evidence.json namespace of the documented windows
# A speech-sized reply: about five minutes of spoken answer, the physical allowance a call's
# fit leaves for one response. A provider's output maximum is documentation, not this budget.
CALL_REPLY_ALLOWANCE_TOKENS = 8_192


@dataclass(frozen=True)
class RouteFact:
    """A documented fact about a realtime route, with its source and read date.

    ``window_tokens`` is the input the route holds (Gemini Live's input token limit).
    ``max_output_tokens`` is the provider's documented output maximum, recorded as
    evidence and never the fit's reserve; ``reply_allowance_tokens`` is what the call's
    fit leaves for one spoken response.
    """

    window_tokens: int
    max_output_tokens: int
    session_cap_minutes: Optional[int]
    source_url: str
    read_on: str
    reply_allowance_tokens: int = CALL_REPLY_ALLOWANCE_TOKENS
    # How many of the route's tokens one chars/4 estimate token costs until the route's own witness
    # exists (a first call, or a witness past its freshness): measured, conservative, never 1.0 by default.
    cold_density: float = 1.0


REALTIME_ROUTE_FACTS: Dict[Tuple[str, str], RouteFact] = {
    # Window and output maximum from the model page (also models.list); the cold density from Gemini's
    # own TEXT counts of a Russian-and-English memory view: 1.19-1.22 per estimate token (2026-10-16).
    (ENGINE_GOOGLE, "gemini-3.8-live"): RouteFact(
        131_072, 65_536, None, "https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live", "2026-10-16",
        cold_density=1.25),
}
EVIDENCE_SOURCE_DOCUMENTED = "documented_route_fact"   # the evidence store's source label for these rows


@dataclass(frozen=True)
class CallRoute:
    engine: EngineRoute
    fact: Optional[RouteFact]
    evidence_fp: str

    @property
    def window_tokens(self) -> Optional[int]:
        return self.fact.window_tokens if self.fact else None


@dataclass(frozen=True)
class CallContextPlan:
    spec: CallSpec
    route: CallRoute
    tool_schemas: List[Dict[str, Any]]
    fit_kwargs: Dict[str, Any]


def call_spec_of(task: Dict[str, Any]) -> Optional[CallSpec]:
    """The call this task is, or ``None`` for a text turn; malformed call metadata raises."""
    meta = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    raw = meta.get("live_call")
    return CallSpec.from_metadata(raw) if isinstance(raw, dict) and raw else None


def resolve_call_route(spec: CallSpec, drive_root: Any) -> CallRoute:
    """The engine route of the call; a known route's documented window becomes evidence."""
    from ouroboros.capability_evidence import route_fingerprint

    route = EngineRoute(engine=spec.engine, model=spec.model, voice=spec.voice,
                        input_rate=GOOGLE_INPUT_RATE, output_rate=GOOGLE_OUTPUT_RATE)
    fact = REALTIME_ROUTE_FACTS.get((spec.engine, spec.model))
    fp = route_fingerprint(provider=spec.engine, model=spec.model)
    if fact is not None:
        _record_window_evidence(drive_root, route, fact, fp)
    return CallRoute(route, fact, fp)


def _record_window_evidence(drive_root: Any, route: EngineRoute, fact: RouteFact, fp: str) -> None:
    """The shipped fact as evidence the fit can use, labelled as what it is.

    ``asserted`` from a documented source: the provider confirmed nothing on the
    wire, so the record never claims provider metadata; a reader sees the URL and
    the read date and can re-probe.
    """
    from ouroboros.capability_evidence import STATUS_ASSERTED, CapabilityEvidence, _store_evidence, canonical_evidence_root

    observed = f"{fact.read_on}T00:00:00Z"
    detail = f"documented context window; {fact.source_url}; read {fact.read_on}"
    evidence = CapabilityEvidence(
        window_tokens=fact.window_tokens, status=STATUS_ASSERTED, source=EVIDENCE_SOURCE_DOCUMENTED, route_fp=fp,
        model=route.model, provider=route.engine, ts=observed, detail=detail, provenance="provider_documentation",
        response_limit={"max_output_tokens": fact.max_output_tokens, "status": STATUS_ASSERTED,
                        "source": EVIDENCE_SOURCE_DOCUMENTED, "observed_at": observed, "route_fp": fp},
    )
    _store_evidence(drive_root if drive_root is not None else canonical_evidence_root(),
                    REALTIME_EVIDENCE_KIND, fp, evidence.to_json())


def window_evidence(drive_root: Any, spec: CallSpec) -> Optional[Dict[str, Any]]:
    """The recorded window evidence of the call's route (``None`` when unknown)."""
    from ouroboros.capability_evidence import _load, route_fingerprint

    record = _load(drive_root).get(REALTIME_EVIDENCE_KIND, {}).get(route_fingerprint(provider=spec.engine, model=spec.model))
    return dict(record) if isinstance(record, dict) else None


def initial_call_schemas(tools: Any, spec: Optional[CallSpec] = None) -> List[Dict[str, Any]]:
    """The call's initial view of the room's permitted schemas.

    An owner call opens with the Nano path and the call's initial names. A
    Presence call opens with every schema its binding's ceiling permits: the
    binding author chose that set, and the registry already narrowed it.
    """
    from ouroboros.tool_policy import initial_tool_schemas

    if spec is not None and spec.presence_binding_id:
        return initial_tool_schemas(tools, context_mode="max")
    return initial_tool_schemas(tools, context_mode="nano", schema_names=CALL_INITIAL_TOOL_NAMES)


def call_context_plan(task: Dict[str, Any], tools: Any, drive_root: Any) -> Optional[CallContextPlan]:
    """Resolve the route BEFORE the context is fitted; ``None`` for a text turn.

    The call's view (``call_view``) rides into the ordinary preparation with the
    realtime route itself: its identity names the window evidence and the calibration
    witness (``known_route``), its documented window is ``known_window_tokens`` and its
    reply allowance ``known_output_reserve_tokens``, so the fit measures against the
    realtime route, not Main's; an unknown route keeps the unknown-window behaviour.
    """
    spec = call_spec_of(task)
    if spec is None:
        return None
    route = resolve_call_route(spec, drive_root)
    fit_kwargs: Dict[str, Any] = {
        "known_route": {"provider": spec.engine, "model": spec.model, "route_fp": route.evidence_fp,
                        **({"cold_density": route.fact.cold_density} if route.fact else {})},
        "known_output_reserve_tokens": route.fact.reply_allowance_tokens if route.fact else CALL_REPLY_ALLOWANCE_TOKENS,
        "call_view": True,
    }
    if route.fact is not None:
        fit_kwargs["known_window_tokens"] = route.fact.window_tokens
    return CallContextPlan(spec, route, initial_call_schemas(tools, spec), fit_kwargs)


def _call_setting(key: str) -> str:
    from ouroboros.settings_integrity import runtime_setting

    return str(runtime_setting(key, "") or "").strip()


def call_minutes_ceiling(spec: CallSpec) -> Optional[int]:
    """The owner's ceiling for this call: the spec's, else the configured one for its kind."""
    if spec.max_minutes:
        return int(spec.max_minutes)
    raw = _call_setting("OUROBOROS_PRESENCE_CALL_MAX_MINUTES" if spec.presence_binding_id else "OUROBOROS_CALL_MAX_MINUTES")
    return int(raw) if raw.isdigit() and int(raw) > 0 else None


def native_search_enabled(spec: CallSpec) -> bool:
    """Gemini's own search grounding for this call, the owner's choice (on unless switched off)."""
    raw = (_call_setting("OUROBOROS_CALL_NATIVE_SEARCH") or "true").lower()
    return raw in ("1", "true", "yes", "on")
