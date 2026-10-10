"""Shared low-level route shape for configured actors and reviewer rows.

The two settings keep different public wire vocabularies (``api_model`` versus
``api_chat`` and different credential-pin keys).  This module owns only the
common mechanical facts: route kind, target identity, optional account pin,
and the ban on provider ``::`` syntax for agent-session targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

ROUTE_KIND_API_MODEL = "api_model"
ROUTE_KIND_AGENT_SESSION = "agent_session"


@dataclass(frozen=True)
class RouteSpec:
    kind: str  # api_model | agent_session
    target_id: str
    credential_profile_id: str = ""

    @property
    def is_session(self) -> bool:
        return self.kind == ROUTE_KIND_AGENT_SESSION

    @property
    def supports_account_pin(self) -> bool:
        """Managed model transports share the same account field as agent sessions."""
        from ouroboros.provider_models import provider_for_model

        return self.is_session or provider_for_model(self.target_id) == "claudexor"


def parse_route_spec(
    raw: Any,
    *,
    setting: str,
    where: str,
    kind_aliases: Mapping[str, str],
    pin_key: str,
    allow_empty_target: bool = False,
    reject_unknown: bool = False,
    strict_strings: bool = False,
    reject_api_pin: bool = False,
) -> RouteSpec:
    """Parse one route while letting each owning setting retain its wire ABI."""
    if not isinstance(raw, dict):
        raise ValueError(f"{setting}: {where} route must be an object {{kind, target_id}}")
    if reject_unknown:
        unknown = sorted(set(raw) - {"kind", "target_id", pin_key})
        if unknown:
            raise ValueError(f"{setting}: {where} route has unknown keys: {unknown}")

    raw_kind = raw.get("kind")
    if strict_strings and not isinstance(raw_kind, str):
        raise ValueError(f"{setting}: {where} route.kind must be a string")
    kind_text = str(raw_kind or "").strip().lower()
    kind = kind_aliases.get(kind_text, "")
    if not kind:
        valid = ", ".join(kind_aliases)
        raise ValueError(f"{setting}: {where} names an unknown route kind {kind_text!r}; valid: {valid}")

    raw_target = raw.get("target_id")
    if strict_strings and not isinstance(raw_target, str):
        raise ValueError(f"{setting}: {where} route.target_id must be a string")
    target = str(raw_target or "").strip()
    if not target and not allow_empty_target:
        raise ValueError(f"{setting}: {where} route.target_id is empty")
    if kind == ROUTE_KIND_AGENT_SESSION and "::" in target:
        raise ValueError(
            f"{setting}: {where} session target {target!r} uses '::' — a delegated row is spelled harness[=model]"
        )
    if kind == ROUTE_KIND_API_MODEL:
        from ouroboros.provider_models import provider_for_model, parse_claudexor_model

        if provider_for_model(target) == "claudexor":
            parse_claudexor_model(target)  # Validate Auto too, before serialization or execution.

    raw_pin = raw.get(pin_key)
    if strict_strings and raw_pin is not None and not isinstance(raw_pin, str):
        raise ValueError(f"{setting}: {where} route.{pin_key} must be a string")
    pin = str(raw_pin or "").strip()
    route = RouteSpec(kind=kind, target_id=target, credential_profile_id=pin)
    if reject_api_pin and pin and not route.supports_account_pin:
        raise ValueError(f"{setting}: {where} route.{pin_key} is meaningful only for agent_session or a managed model source")
    return route


def route_spec_dict(route: RouteSpec, *, api_kind: str, pin_key: str) -> dict[str, str]:
    """Serialize through the owning setting's public kind/pin spellings."""
    payload = {
        "kind": "agent_session" if route.is_session else api_kind,
        "target_id": route.target_id,
    }
    if route.supports_account_pin:
        payload[pin_key] = route.credential_profile_id
    return payload


# Harnesses whose model slugs END with the reasoning tier (``grok-4.7-xhigh-fast``): the
# level is the model's identity, in a session target and in a Claudexor-managed API model alike.
_SLUG_EFFORT_HARNESSES = frozenset({"cursor", "agy"})


def harness_model_named_effort(harness: str, model: str) -> str:
    """THE detector: the tier a ``harness``'s model slug encodes (a trailing ``-fast``
    stripped), '' for every other harness or slug."""
    if str(harness or "").strip().lower() not in _SLUG_EFFORT_HARNESSES:
        return ""
    from ouroboros.settings_scales import EFFORT_SCALE

    text = str(model or "").strip()
    compound_model = text[:-5] if text.lower().endswith("-fast") else text
    encoded = compound_model.rsplit("-", 1)[-1].lower()
    return encoded if encoded in EFFORT_SCALE else ""


def compound_session_effort(route: RouteSpec) -> str:
    """Effort already encoded in a Cursor/Agy compound session slug, if any."""
    if not route.is_session:
        return ""
    harness, separator, model = route.target_id.partition("=")
    return harness_model_named_effort(harness, model) if separator else ""


def api_model_named_effort(model: str) -> str:
    """Effort encoded in a Claudexor-managed API model name (``claudexor::cursor=<slug>``), if any."""
    from ouroboros.provider_models import parse_claudexor_model, provider_for_model

    if provider_for_model(model) != "claudexor":
        return ""
    try:
        source, native = parse_claudexor_model(model)
    except ValueError:
        return ""
    return harness_model_named_effort(source, native)


def model_named_effort(route: RouteSpec) -> str:
    """The level a route's model name carries — session target or API model — '' when none.
    It wins over every pin, request and range (``settings_scales.choose_effort``); a stored
    API row's contradicting pin is set aside at execution and disclosed, never a parse error."""
    return compound_session_effort(route) if route.is_session else api_model_named_effort(route.target_id)


def validate_compound_session_effort(
    route: RouteSpec,
    effort: str,
    *,
    setting: str,
    where: str,
) -> None:
    """Reject two contradictory effort authorities on Cursor/Agy session routes (the shared
    parser runs on stored rows too, so this rule stays exactly as wide as it was)."""
    if not effort:
        return
    encoded = compound_session_effort(route)
    if encoded and encoded != effort:
        raise ValueError(
            f"{setting}: {where} effort {effort!r} conflicts with compound route effort {encoded!r}"
        )


__all__ = [
    "api_model_named_effort",
    "compound_session_effort",
    "harness_model_named_effort",
    "model_named_effort",
    "ROUTE_KIND_AGENT_SESSION",
    "ROUTE_KIND_API_MODEL",
    "RouteSpec",
    "parse_route_spec",
    "route_spec_dict",
    "validate_compound_session_effort",
]
