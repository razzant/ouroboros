"""Shared low-level route shape for configured actors and reviewer rows.

The two settings keep different public wire vocabularies (``api_model`` versus
``api_chat`` and different credential-pin keys).  This module owns only the
common mechanical facts: route kind, target identity, optional account pin, the
engine a session runs on, and the ban on provider ``::`` syntax for agent-session targets.

A session route's ``engine`` names a remote engine connection (``remote_engines``): where the
session runs, orthogonal to its harness, model and account; "" is the owned engine and is not
written, so a local route's bytes stay as they were. ``workspace`` (with an engine only) names a
remote workspace that connection declares: the session then works in that folder on the host
instead of a staged copy (``remote_workspace``). An ``api_model`` route names its engine in its
target (``claudexor@<engine>::source=model``) and carries neither field.
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
    engine: str = ""  # a session's remote engine connection; "" = the owned engine
    workspace: str = ""  # a declared remote workspace on that engine's host; "" = a staged copy

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
        unknown = sorted(set(raw) - {"kind", "target_id", pin_key, "engine", "workspace"})
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
    engine = _engine(raw.get("engine"), kind, setting=setting, where=where)
    route = RouteSpec(kind=kind, target_id=target, credential_profile_id=pin, engine=engine,
                      workspace=_workspace(raw.get("workspace"), engine, setting=setting, where=where))
    if reject_api_pin and pin and not route.supports_account_pin:
        raise ValueError(f"{setting}: {where} route.{pin_key} is meaningful only for agent_session or a managed model source")
    return route


def _engine(raw: Any, kind: str, *, setting: str, where: str) -> str:
    """A session route's remote engine name (the connection-name grammar), "" when absent."""
    if raw is None or raw == "":
        return ""
    from ouroboros.remote_engines import _RESERVED_NAMES, CONNECTION_NAME

    if kind != ROUTE_KIND_AGENT_SESSION:
        raise ValueError(f"{setting}: {where} route.engine is meaningful only for agent_session; an API route "
                         "names its engine in its target (claudexor@<engine>::source=model)")
    if not isinstance(raw, str) or not CONNECTION_NAME.fullmatch(raw) or raw in _RESERVED_NAMES:
        raise ValueError(f"{setting}: {where} route.engine must name an engine connection")
    return raw


def _workspace(raw: Any, engine: str, *, setting: str, where: str) -> str:
    """A session route's declared remote workspace: an absolute folder on its engine's host, "" when absent."""
    if raw is None or raw == "":
        return ""
    from ouroboros.remote_engines import remote_path_error

    if not engine:
        raise ValueError(f"{setting}: {where} route.workspace needs route.engine: it is a folder on that engine's host")
    if (error := remote_path_error(raw, home_ok=False)):
        raise ValueError(f"{setting}: {where} route.workspace {error}")
    return raw


def session_location(route: Mapping[str, Any]) -> dict[str, str]:
    """``{engine, workspace}`` of a stored session route object, "" for the owned engine and a staged copy."""
    return {"engine": str(route.get("engine") or ""), "workspace": str(route.get("workspace") or "")}


def route_spec_dict(route: RouteSpec, *, api_kind: str, pin_key: str) -> dict[str, str]:
    """Serialize through the owning setting's public kind/pin spellings."""
    payload = {
        "kind": "agent_session" if route.is_session else api_kind,
        "target_id": route.target_id,
    }
    if route.supports_account_pin:
        payload[pin_key] = route.credential_profile_id
    if route.engine:
        payload["engine"] = route.engine
    if route.workspace:
        payload["workspace"] = route.workspace
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
    "session_location",
    "validate_compound_session_effort",
]
