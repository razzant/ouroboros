"""Bind transport-neutral wire actions and parsed semantics to physical attempts."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence, Tuple

from ouroboros.request_wire_attempt import WireUsageDisclosure as WireUsageDisclosure
from ouroboros.request_wire_attempt import (
    validate_normalized_wire_success,
    validate_physical_wire_attempt,
)
from ouroboros.anthropic_native_custody import is_replayed_native_content
from ouroboros.request_wire_contract import (
    ANTHROPIC_REASONING_FIELDS,
    CACHE_HINT_FIELDS,
    NESTED_REASONING_FIELD,
    TOOL_DIALECTS,
    WIRE_REASON_CODES,
    WIRE_SEMANTIC_KINDS,
    EphemeralWireAdjustment,
    PendingWireAction,
    RequestWireProfile,
    _freeze_wire_action,
    _valid_sha256,
    _wire_action_dict,
    build_request_wire_profile,
    canonical_sha256,
    infer_tool_dialect,
    payload_effort,
    physical_candidate_bytes,
    physical_candidate_sha256,
    rebuild_request_wire_profile,
    tool_choice_family,
    tool_choice_names,
    tool_choice_projection_matches,
    tool_choice_requires_call,
    validate_durable_wire_action_for_profile,
    validate_wire_action_for_profile,
    wire_action_identity,
)
from ouroboros.request_wire_custom_validation import CustomArgumentValidationReceipt
from ouroboros.request_wire_resolution import resolve_effort_action_set
from ouroboros.utils import utc_now_iso

if TYPE_CHECKING:
    from ouroboros.openai_chat_custom import ProjectedCustomToolCatalog
    from ouroboros.usage_accounting import PhysicalAttemptCapture

WIRE_ACTION_APPLICATION_SOURCES = frozenset({"durable", "pending", "task_local"})
_REASONING_DROP_FIELDS = frozenset({
    "reasoning_effort",
    NESTED_REASONING_FIELD,
    *ANTHROPIC_REASONING_FIELDS,
})
_TOOL_REQUIRED_CHOICES = frozenset({"required", "any", "named"})
_CANDIDATE_BINDING_TOKEN = object()
_SEMANTIC_BINDING_TOKEN = object()
_RECEIPT_BINDING_TOKEN = object()
REQUESTED_FUNCTION_TO_CUSTOM = "function_to_openai_chat_custom"


def _route_identity(profile: RequestWireProfile) -> Tuple[str, ...]:
    return (
        profile.provider,
        profile.endpoint_sha256,
        profile.api_surface,
        profile.model,
        profile.provider_routing_sha256,
        profile.contract_headers_sha256,
    )


@dataclass(frozen=True)
class WireCandidateSpec:
    tool_dialect: str
    effort: str
    reason_code: str
    task_local: bool = False

    def __post_init__(self) -> None:
        from ouroboros.config import EFFORT_SCALE

        if self.tool_dialect not in TOOL_DIALECTS:
            raise ValueError("unsupported candidate tool dialect")
        if self.effort not in (*EFFORT_SCALE, "provider_default"):
            raise ValueError("unsupported candidate effort")
        if self.reason_code not in WIRE_REASON_CODES:
            raise ValueError("unsupported candidate reason")
        if self.task_local != (self.reason_code == "task_local_availability_fallback"):
            raise ValueError("task-local candidate requires the availability-fallback reason")
        if self.task_local and self.effort != "none":
            raise ValueError("task-local availability candidate must apply explicit none")


def direct_openai_tool_candidate_ladder(
    requested_effort: str,
    *,
    remaining_physical_attempts: int,
) -> Tuple[WireCandidateSpec, ...]:
    """Freeze owner-selected custom→function→task-local-none ordering.

    The helper only truncates.  The accounting rail remains authoritative and
    will refuse any dispatch after its caller-owned limit is exhausted.
    """
    from ouroboros.config import EFFORT_SCALE

    effort = str(requested_effort or "").strip().lower()
    if effort not in EFFORT_SCALE:
        raise ValueError("unsupported requested effort")
    try:
        remaining = max(0, int(remaining_physical_attempts))
    except (TypeError, ValueError):
        remaining = 0
    if effort == "none":
        candidates = (WireCandidateSpec("function", "none", "requested_wire_form"),)
    else:
        candidates = (
            WireCandidateSpec("openai_chat_custom", effort, "requested_wire_form"),
            WireCandidateSpec("function", effort, "provider_rejected_tool_dialect"),
            WireCandidateSpec(
                "function",
                "none",
                "task_local_availability_fallback",
                task_local=True,
            ),
        )
    return candidates[:remaining]


@dataclass(frozen=True)
class WireAppliedAction:
    """One exact action present on a physical candidate and its authority."""

    profile: RequestWireProfile
    action: Mapping[str, Any]
    source: str

    def __post_init__(self) -> None:
        if self.source not in WIRE_ACTION_APPLICATION_SOURCES:
            raise ValueError("unsupported wire-action application source")
        validator = (
            validate_wire_action_for_profile
            if self.source == "task_local"
            else validate_durable_wire_action_for_profile
        )
        normalized = validator(self.profile, self.action)
        if not normalized:
            raise ValueError("invalid applied wire action")
        if self.source == "task_local":
            availability_fallback = (
                normalized.get("kind") == "set_value"
                and normalized.get("to") == "none"
                and normalized.get("reason_code") == "task_local_availability_fallback"
            )
            same_rung_repair = (
                normalized.get("kind") == "drop_field"
                and not set(normalized.get("fields") or ()).intersection(
                    _REASONING_DROP_FIELDS)
                and bool(validate_durable_wire_action_for_profile(self.profile, normalized))
            )
            if not availability_fallback and not same_rung_repair:
                raise ValueError("invalid task-local same-rung repair")
        object.__setattr__(self, "action", _freeze_wire_action(normalized))

    @classmethod
    def pending(cls, item: PendingWireAction) -> "WireAppliedAction":
        return cls(item.profile, item.action, "pending")

    @classmethod
    def reactive(
        cls, item: PendingWireAction, *, task_local: bool = False,
    ) -> "WireAppliedAction":
        """Keep a degraded-rung repair ephemeral instead of learnable pending evidence."""
        return cls(item.profile, item.action, "task_local" if task_local else "pending")

    @classmethod
    def task_local(cls, item: EphemeralWireAdjustment) -> "WireAppliedAction":
        return cls(item.profile, item.action, "task_local")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "profile_fingerprint": self.profile.fingerprint,
            "action": _wire_action_dict(self.action),
        }


@dataclass(frozen=True)
class WireCandidateManifest:
    """Builder-validated description of the candidate presented to accounting."""

    source_profile: RequestWireProfile
    accepted_profile: RequestWireProfile
    physical_model: str
    candidate_spec: WireCandidateSpec
    requested_effort: str
    source_payload_sha256: str
    candidate_sha256: str
    _physical_payload_json: str = field(repr=False)
    ladder_ordinal: int
    tool_choice_names: Tuple[str, ...]
    requires_tool_call: bool
    forbids_tool_call: bool
    _custom_catalog: "ProjectedCustomToolCatalog | None" = field(default=None, repr=False)
    requested_tool_projection: str = ""
    applied_actions: Tuple[WireAppliedAction, ...] = ()
    _binding_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        from ouroboros.config import EFFORT_SCALE

        if self._binding_token is not _CANDIDATE_BINDING_TOKEN:
            raise ValueError("wire candidate must be created by bind_wire_candidate")
        if not self.physical_model.strip():
            raise ValueError("wire candidate requires the physical accounting model")
        if self.requested_effort not in EFFORT_SCALE:
            raise ValueError("wire candidate requires a supported requested effort")
        if not all(_valid_sha256(value) for value in (
            self.source_payload_sha256, self.candidate_sha256
        )):
            raise ValueError("wire candidate requires exact source/final payload digests")
        try:
            physical_payload = json.loads(self._physical_payload_json)
        except (TypeError, ValueError):
            physical_payload = None
        if (
            not isinstance(physical_payload, dict)
            or physical_candidate_sha256(physical_payload) != self.candidate_sha256
            or physical_payload.get("model") != self.accepted_profile.model
        ):
            raise ValueError("wire candidate does not own its exact physical payload")
        if type(self.ladder_ordinal) is not int or self.ladder_ordinal < 1:
            raise ValueError("wire candidate requires a positive ladder ordinal")
        if _route_identity(self.source_profile) != _route_identity(self.accepted_profile):
            raise ValueError("wire candidate cannot change provider, endpoint, API, or model")
        if (
            self.source_profile.tool_choice != self.accepted_profile.tool_choice
            or self.source_profile.tool_choice_sha256
            != self.accepted_profile.tool_choice_sha256
        ):
            raise ValueError("wire candidate cannot change tool-choice semantics")
        if self.candidate_spec.tool_dialect != self.accepted_profile.tool_dialect:
            raise ValueError("wire candidate spec and accepted dialect differ")
        if self.requested_tool_projection not in {"", REQUESTED_FUNCTION_TO_CUSTOM}:
            raise ValueError("unsupported requested tool projection")
        if self.requested_tool_projection:
            if (
                self.source_profile.provider != "openai"
                or self.source_profile.api_surface != "chat.completions"
                or self.source_profile.tool_dialect != "function"
                or self.accepted_profile.tool_dialect != "openai_chat_custom"
                or self.candidate_spec.reason_code != "requested_wire_form"
            ):
                raise ValueError("requested custom projection is not bound to direct OpenAI Chat")
        if tuple(sorted(set(self.tool_choice_names))) != self.tool_choice_names:
            raise ValueError("wire candidate tool-choice names must be unique and sorted")
        if (
            self.accepted_profile.tool_choice in {"named", "allowed_tools"}
            and not self.tool_choice_names
        ):
            raise ValueError("named/allowed tool choice requires exact tool names")
        if self.requires_tool_call and self.forbids_tool_call:
            raise ValueError("wire candidate cannot both require and forbid tool calls")
        if self.accepted_profile.tool_dialect == "openai_chat_custom":
            from ouroboros.openai_chat_custom import ProjectedCustomToolCatalog

            if not isinstance(self._custom_catalog, ProjectedCustomToolCatalog):
                raise ValueError("custom candidate requires its codec-owned catalog")
            exact_names = tuple(sorted(set(self._custom_catalog.tool_names)))
            if (
                not _valid_sha256(self._custom_catalog.catalog_sha256)
                or not exact_names
                or exact_names != self._custom_catalog.tool_names
                or self._custom_catalog.wire_tools() != physical_payload.get("tools")
            ):
                raise ValueError("custom candidate catalog differs from its physical payload")
        elif self._custom_catalog is not None:
            raise ValueError("non-custom candidate cannot own a custom catalog")
        if not isinstance(self.applied_actions, tuple) or any(
            not isinstance(item, WireAppliedAction) for item in self.applied_actions
        ):
            raise ValueError("wire candidate applied actions must be typed")
        if any(
            _route_identity(item.profile) != _route_identity(self.source_profile)
            for item in self.applied_actions
        ):
            raise ValueError("wire action is not bound to the candidate route")
        identities = [wire_action_identity(item.action) for item in self.applied_actions]
        if len(identities) != len(set(identities)):
            raise ValueError("wire candidate contains duplicate action identities")
        task_local = [item for item in self.applied_actions if item.source == "task_local"]
        pending = [item for item in self.applied_actions if item.source == "pending"]
        if bool(task_local) != self.candidate_spec.task_local:
            raise ValueError("candidate task-local status is not action-bound")
        if task_local and pending:
            raise ValueError("degraded success cannot confirm unrelated pending evidence")
        if self.requested_tool_projection:
            if any(item.source == "task_local" for item in self.applied_actions):
                raise ValueError(
                    "requested custom projection cannot carry task-local actions"
                )
            if any(
                item.action.get("kind") == "replace_dialect"
                for item in self.applied_actions
            ):
                raise ValueError("requested custom projection is not a dialect action")
        object.__setattr__(self, "_binding_token", None)

    @property
    def task_local(self) -> bool:
        return self.candidate_spec.task_local

    @property
    def pending(self) -> Tuple[PendingWireAction, ...]:
        return tuple(
            PendingWireAction(item.profile, item.action)
            for item in self.applied_actions
            if item.source == "pending"
        )

    @property
    def custom_catalog(self) -> "ProjectedCustomToolCatalog | None":
        return self._custom_catalog

    @property
    def custom_catalog_sha256(self) -> str:
        return self._custom_catalog.catalog_sha256 if self._custom_catalog else ""

    @property
    def custom_catalog_tool_names(self) -> Tuple[str, ...]:
        return self._custom_catalog.tool_names if self._custom_catalog else ()

    def disclosed_actions(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(item.as_dict() for item in self.applied_actions)

    def physical_payload(self) -> Dict[str, Any]:
        """Return a fresh decode of the exact payload owned by this candidate."""
        value = json.loads(self._physical_payload_json)
        if not isinstance(value, dict):  # guarded by construction
            raise ValueError("wire candidate physical payload is malformed")
        return value


def _payload_tool_names(payload: Mapping[str, Any], dialect: str) -> Tuple[str, ...]:
    names = []
    for tool in payload.get("tools") or []:
        if not isinstance(tool, Mapping):
            raise ValueError("physical tool catalog contains a non-object")
        if dialect == "function":
            function = tool.get("function")
            if isinstance(function, Mapping):
                name = str(function.get("name") or "").strip()
            else:
                name = str(tool.get("name") or "").strip()
        elif dialect == "openai_chat_custom":
            custom = tool.get("custom")
            name = str(custom.get("name") or "").strip() if isinstance(custom, Mapping) else ""
        else:
            name = ""
        if not name:
            raise ValueError("physical tool catalog contains an unnamed tool")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("physical tool catalog contains duplicate names")
    return tuple(sorted(names))


def _validate_physical_tool_choice(
    canonical_choice: Any,
    payload: Mapping[str, Any],
    dialect: str,
) -> Tuple[str, ...]:
    physical_choice = payload.get("tool_choice")
    catalog_names = _payload_tool_names(payload, dialect)
    if not tool_choice_projection_matches(
        canonical_choice,
        physical_choice,
        physical_dialect=dialect,
        physical_catalog_names=catalog_names,
    ):
        raise ValueError("physical tool choice differs from canonical semantics")
    return catalog_names


def cache_hint_holders(payload: Mapping[str, Any], field: str) -> list:
    """The objects carrying the prompt-cache hint ``field`` names (``CACHE_HINT_FIELDS``):
    ``tools`` items, ``system``/``messages`` items, their blocks and a tool_result's nested
    blocks, the send-time finalizer's holder set (replayed native content stays untouched)."""
    holders: list = []
    for key in ("tools",) if field == "tools.cache_control" else ("tools", "system", "messages"):
        part = payload.get(key)
        for item in part if isinstance(part, list) else [part]:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            blocks = [block for block in content if isinstance(block, dict)] if (
                isinstance(content, list) and not is_replayed_native_content(content)) else []
            nested = [inner for block in blocks if isinstance(block.get("content"), list)
                      for inner in block["content"] if isinstance(inner, dict)]
            holders.extend(holder for holder in (item, *blocks, *nested) if "cache_control" in holder and (
                field != "cache_control.ttl"
                or isinstance(holder["cache_control"], dict) and "ttl" in holder["cache_control"]))
    return holders


def _drop_cache_hint(payload: Dict[str, Any], field: str) -> None:
    holders = cache_hint_holders(payload, field)
    if not holders:
        raise ValueError("drop action source field is absent")
    for holder in holders:
        if field == "cache_control.ttl":
            holder["cache_control"].pop("ttl", None)
        else:
            holder.pop("cache_control", None)
    if field != "cache_control":
        return
    # A user/tool row that became a one-part text list to carry its marker reads as the
    # plain string again, the shape this lane sent before it carried markers.
    for message in payload.get("messages") or ():
        content = message.get("content") if isinstance(message, dict) else None
        if (isinstance(content, list) and len(content) == 1 and message.get("role") in {"user", "tool"}
                and isinstance(content[0], dict) and set(content[0]) == {"type", "text"}
                and content[0]["type"] == "text" and isinstance(content[0]["text"], str)):
            message["content"] = content[0]["text"]


def _drop_path(payload: Dict[str, Any], path: str) -> None:
    if path in CACHE_HINT_FIELDS:
        _drop_cache_hint(payload, path)
        return
    current: Any = payload
    parts = path.split(".")
    for part in parts[:-1]:
        current = current.get(part) if isinstance(current, dict) else None
    if not isinstance(current, dict) or parts[-1] not in current:
        raise ValueError("drop action source field is absent")
    current.pop(parts[-1])


def _set_payload_effort(payload: Dict[str, Any], carrier: str, effort: str) -> None:
    if carrier == "reasoning_effort":
        payload["reasoning_effort"] = effort
    elif carrier == NESTED_REASONING_FIELD:
        payload["extra_body"]["reasoning"]["effort"] = effort
    elif carrier == "anthropic.adaptive":
        payload["output_config"]["effort"] = effort
    else:
        raise ValueError("effort action cannot rewrite this reasoning carrier")


def _project_function_payload_to_openai_custom(
    payload: Mapping[str, Any],
    canonical_choice: Any,
) -> Tuple[Dict[str, Any], "ProjectedCustomToolCatalog"]:
    """Run the pure codec once and return its complete physical projection."""
    from ouroboros.openai_chat_custom import (
        project_function_tool_request_to_openai_custom,
        project_messages_for_openai_custom,
    )

    tools = payload.get("tools")
    messages = payload.get("messages")
    if not isinstance(tools, (list, tuple)) or not isinstance(messages, (list, tuple)):
        raise ValueError("function-to-custom projection requires tools and messages")
    projected = project_function_tool_request_to_openai_custom(tools, canonical_choice)
    physical = copy.deepcopy(dict(payload))
    physical["messages"] = project_messages_for_openai_custom(
        messages, projected.catalog
    )
    physical["tools"] = projected.catalog.wire_tools()
    if "tool_choice" in physical:
        physical["tool_choice"] = projected.tool_choice
    return physical, projected.catalog


def _materialize_tool_dialect(
    payload: Mapping[str, Any],
    function_twin: Mapping[str, Any],
    target_dialect: str,
    canonical_choice: Any,
) -> Tuple[Dict[str, Any], "ProjectedCustomToolCatalog | None"]:
    """Materialize either exact codec twin from one canonical function source."""
    current_dialect = infer_tool_dialect(payload)
    if current_dialect == target_dialect:
        return copy.deepcopy(dict(payload)), None
    if current_dialect == "function" and target_dialect == "openai_chat_custom":
        return _project_function_payload_to_openai_custom(payload, canonical_choice)
    if current_dialect == "openai_chat_custom" and target_dialect == "function":
        if infer_tool_dialect(function_twin) != "function":
            raise ValueError("reverse dialect action lacks its canonical function twin")
        return copy.deepcopy(dict(function_twin)), None
    raise ValueError("wire action profile requires an underivable tool dialect")


def _apply_action_step(
    action_item: WireAppliedAction,
    pre_payload: Mapping[str, Any],
    canonical_choice: Any,
) -> Tuple[Dict[str, Any], "ProjectedCustomToolCatalog | None"]:
    action = action_item.action
    kind = action.get("kind")
    if kind == "drop_field":
        physical = copy.deepcopy(dict(pre_payload))
        for field_name in action.get("fields") or []:
            _drop_path(physical, str(field_name))
        return physical, None
    if kind == "set_value":
        current = payload_effort(pre_payload)
        if not current:
            raise ValueError("effort action source has no physical effort")
        resolution = resolve_effort_action_set(current, (action,))
        if resolution.conflict or not resolution.actions:
            raise ValueError("effort action is not applicable to its physical step")
        physical = copy.deepcopy(dict(pre_payload))
        _set_payload_effort(
            physical, action_item.profile.reasoning_carrier, resolution.applied_effort
        )
        return physical, None
    source_dialect = infer_tool_dialect(pre_payload)
    if source_dialect != action.get("from"):
        raise ValueError("dialect action does not match its physical source")
    if action.get("to") == "openai_chat_custom":
        return _project_function_payload_to_openai_custom(pre_payload, canonical_choice)
    raise ValueError("reverse dialect action must use the bound canonical function twin")


def _derive_physical_payload(
    *,
    target: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    applications: Tuple[WireAppliedAction, ...],
    canonical_choice: Any,
    requested_projection: str,
) -> Tuple[Dict[str, Any], "ProjectedCustomToolCatalog | None"]:
    """Walk ordered authority actions while keeping exact function/custom twins."""
    physical_payload = copy.deepcopy(dict(source_payload))
    function_twin = copy.deepcopy(dict(source_payload))
    derived_catalog = None
    for item in applications:
        action = item.action
        desired_dialect = item.profile.tool_dialect
        if infer_tool_dialect(physical_payload) != desired_dialect:
            physical_payload, catalog = _materialize_tool_dialect(
                physical_payload, function_twin, desired_dialect, canonical_choice
            )
            derived_catalog = catalog or derived_catalog
        rebuilt = rebuild_request_wire_profile(
            item.profile,
            target,
            physical_payload,
            tool_choice_override=canonical_choice,
        )
        if rebuilt != item.profile:
            raise ValueError("wire action profile differs from its exact pre-step payload")
        if action.get("kind") == "replace_dialect" and action.get("to") == "function":
            physical_payload, catalog = _materialize_tool_dialect(
                physical_payload, function_twin, "function", canonical_choice
            )
        else:
            physical_payload, catalog = _apply_action_step(
                item, physical_payload, canonical_choice
            )
            if action.get("kind") != "replace_dialect":
                if desired_dialect == "function":
                    function_twin = copy.deepcopy(physical_payload)
                elif desired_dialect == "openai_chat_custom":
                    function_twin, _ = _apply_action_step(
                        item, function_twin, canonical_choice
                    )
        derived_catalog = catalog or derived_catalog
    if requested_projection:
        if infer_tool_dialect(physical_payload) == "function":
            physical_payload, derived_catalog = _materialize_tool_dialect(
                physical_payload,
                function_twin,
                "openai_chat_custom",
                canonical_choice,
            )
        elif infer_tool_dialect(physical_payload) != "openai_chat_custom":
            raise ValueError("requested custom projection source is not function tools")
    return physical_payload, derived_catalog


def _resolved_candidate_effort(
    requested: str,
    applications: Tuple[WireAppliedAction, ...],
) -> Tuple[str, set[str]]:
    effort_actions = [
        item.action for item in applications if item.action.get("kind") == "set_value"
    ]
    dropped = {
        field_name
        for item in applications
        if item.action.get("kind") == "drop_field"
        for field_name in item.action.get("fields") or []
    }
    if dropped.intersection(_REASONING_DROP_FIELDS):
        if effort_actions:
            raise ValueError("a removed reasoning carrier cannot also apply an effort action")
        return "provider_default", dropped
    resolution = resolve_effort_action_set(requested, effort_actions)
    if resolution.conflict:
        raise ValueError("conflicting effort evidence cannot produce a wire candidate")
    selected = tuple(wire_action_identity(action) for action in resolution.actions)
    claimed = tuple(wire_action_identity(action) for action in effort_actions)
    if selected != claimed:
        raise ValueError("wire candidate contains unapplied effort evidence")
    return resolution.applied_effort, dropped


def bind_wire_candidate(
    *,
    target: Mapping[str, Any],
    api_surface: str,
    source_payload: Mapping[str, Any],
    candidate_spec: WireCandidateSpec,
    requested_effort: str,
    ladder_ordinal: int,
    applied_actions: Sequence[WireAppliedAction] = (),
) -> WireCandidateManifest:
    """Derive and bind the one exact physical payload from canonical authority."""
    resolved_model = str(target.get("resolved_model") or "").strip()
    accounting_model = str(target.get("usage_model") or resolved_model).strip()
    if not resolved_model or not accounting_model:
        raise ValueError("wire candidate requires resolved and accounting model identity")
    if source_payload.get("model") != resolved_model:
        raise ValueError("physical payload model differs from resolved route model")
    applications = tuple(
        WireAppliedAction(item.profile, item.action, item.source)
        if isinstance(item, WireAppliedAction) else item
        for item in applied_actions
    )
    if any(not isinstance(item, WireAppliedAction) for item in applications):
        raise ValueError("wire candidate applied actions must be typed")
    canonical_choice = source_payload.get("tool_choice")
    source_profile = build_request_wire_profile(
        target,
        source_payload,
        api_surface=api_surface,
        tool_choice_override=canonical_choice,
    )
    if any(
        _route_identity(item.profile) != _route_identity(source_profile)
        for item in applications
    ):
        raise ValueError("wire action is not bound to the candidate route")
    source_dialect = infer_tool_dialect(source_payload)
    requested_projection = ""
    if (
        source_dialect == "function"
        and candidate_spec.tool_dialect == "openai_chat_custom"
        and candidate_spec.reason_code == "requested_wire_form"
    ):
        requested_projection = REQUESTED_FUNCTION_TO_CUSTOM
        if any(item.source == "task_local" for item in applications):
            raise ValueError(
                "requested custom projection cannot carry task-local actions"
            )
        if any(item.action.get("kind") == "replace_dialect" for item in applications):
            raise ValueError("requested custom projection is not a dialect action")
    physical_payload, derived_catalog = _derive_physical_payload(
        target=target,
        source_payload=source_payload,
        applications=applications,
        canonical_choice=canonical_choice,
        requested_projection=requested_projection,
    )
    accepted_profile = build_request_wire_profile(
        target,
        physical_payload,
        api_surface=api_surface,
        tool_choice_override=canonical_choice,
    )
    requested = str(requested_effort or "").strip().lower()
    if payload_effort(source_payload) != requested:
        raise ValueError("requested effort differs from the source physical payload")
    expected_effort, dropped = _resolved_candidate_effort(requested, applications)
    final_effort = payload_effort(physical_payload)
    if expected_effort == "provider_default":
        if final_effort:
            raise ValueError("reasoning-carrier drop is absent from the final payload")
    elif final_effort != expected_effort:
        raise ValueError("final physical effort is not produced by applied actions")
    if candidate_spec.effort == "none" and requested != "none" and not candidate_spec.task_local:
        raise ValueError(
            "non-requested explicit none requires a bound action with task-local status"
        )
    if candidate_spec.effort != expected_effort:
        raise ValueError("candidate spec effort differs from the physical payload")
    if candidate_spec.tool_dialect != accepted_profile.tool_dialect:
        raise ValueError("candidate spec dialect differs from the physical payload")
    dialect_actions = [
        item.action for item in applications if item.action.get("kind") == "replace_dialect"
    ]
    if not dialect_actions and not requested_projection and (
        source_profile.function_strictness != accepted_profile.function_strictness
    ):
        raise ValueError("physical function strictness changed without a dialect action")
    if dropped.intersection(_REASONING_DROP_FIELDS):
        if accepted_profile.reasoning_carrier or accepted_profile.reasoning_options_sha256:
            raise ValueError("reasoning drop is absent from the accepted profile")
    elif (
        source_profile.reasoning_carrier != accepted_profile.reasoning_carrier
        or source_profile.reasoning_options_sha256
        != accepted_profile.reasoning_options_sha256
    ):
        raise ValueError("reasoning shape changed without a drop action")
    canonical_names = tool_choice_names(canonical_choice)
    catalog_names = _validate_physical_tool_choice(
        canonical_choice, physical_payload, accepted_profile.tool_dialect
    )
    if accepted_profile.tool_dialect == "openai_chat_custom":
        if derived_catalog is None or derived_catalog.tool_names != catalog_names:
            raise ValueError("custom candidate is not derived from the bound codec projection")
    else:
        derived_catalog = None
    physical_payload_json = physical_candidate_bytes(physical_payload).decode("utf-8")
    return WireCandidateManifest(
        source_profile=source_profile,
        accepted_profile=accepted_profile,
        physical_model=accounting_model,
        candidate_spec=candidate_spec,
        requested_effort=requested,
        source_payload_sha256=physical_candidate_sha256(source_payload),
        candidate_sha256=physical_candidate_sha256(physical_payload),
        _physical_payload_json=physical_payload_json,
        ladder_ordinal=ladder_ordinal,
        tool_choice_names=canonical_names,
        requires_tool_call=tool_choice_requires_call(canonical_choice),
        forbids_tool_call=tool_choice_family(canonical_choice) == "none",
        _custom_catalog=derived_catalog,
        requested_tool_projection=requested_projection,
        applied_actions=applications,
        _binding_token=_CANDIDATE_BINDING_TOKEN,
    )


def rebind_processing_candidate(
    candidate: WireCandidateManifest, *, target: Mapping[str, Any],
    source_payload: Mapping[str, Any], field_name: str, standard_value: str,
) -> Tuple[WireCandidateManifest, Dict[str, Any]]:
    """Rebuild a processing-only retry while retaining the canonical tool twin.

    Processing is not an effort/dialect downgrade or learned capability action.
    Reuse the original factory and require the rebuilt send to equal the prior
    physical body except for its one explicit native processing field.
    """
    provider = candidate.accepted_profile.provider
    if (provider, field_name, standard_value) not in {
        ("openai", "service_tier", "default"),
        ("openrouter", "service_tier", "default"),
        ("anthropic", "speed", "standard"),
    }:
        raise ValueError("unsupported processing-only transition")
    source = copy.deepcopy(dict(source_payload))
    source[field_name] = standard_value
    rebuilt = bind_wire_candidate(
        target=target, api_surface=candidate.source_profile.api_surface,
        source_payload=source, candidate_spec=candidate.candidate_spec,
        requested_effort=candidate.requested_effort, ladder_ordinal=candidate.ladder_ordinal,
        applied_actions=candidate.applied_actions,
    )
    expected = candidate.physical_payload()
    expected[field_name] = standard_value
    if rebuilt.physical_payload() != expected:
        raise ValueError("processing retry changed unrelated physical input")
    return rebuilt, source


def wire_semantic_kind_allowed(
    profile: RequestWireProfile,
    semantic_kind: str,
) -> bool:
    """Validate response semantics for one sent provider/API/tool shape."""
    kind = str(semantic_kind or "")
    if kind not in WIRE_SEMANTIC_KINDS:
        return False
    has_tool = False
    if profile.provider == "anthropic" and profile.api_surface == "messages":
        if kind not in {"anthropic_message", "anthropic_tool_use"}:
            return False
        has_tool = kind == "anthropic_tool_use"
    elif profile.api_surface == "chat.completions":
        if profile.tool_dialect == "openai_chat_custom":
            if kind not in {"chat_message", "custom_tool_call_json_object"}:
                return False
            has_tool = kind == "custom_tool_call_json_object"
        elif profile.tool_dialect == "function":
            if kind not in {"chat_message", "function_tool_call"}:
                return False
            has_tool = kind == "function_tool_call"
        elif profile.tool_dialect == "none":
            if kind != "chat_message":
                return False
        else:
            return False
    else:
        return False
    if profile.tool_choice in _TOOL_REQUIRED_CHOICES and not has_tool:
        return False
    if profile.tool_choice == "none" and has_tool:
        return False
    return True


@dataclass(frozen=True)
class ObservedWireToolCall:
    call_id: str
    tool_name: str
    arguments_sha256: str
    _binding_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._binding_token is not _SEMANTIC_BINDING_TOKEN:
            raise ValueError("wire tool observation must be parser-issued")
        if not self.call_id or not self.tool_name or not _valid_sha256(self.arguments_sha256):
            raise ValueError("wire tool observation requires exact call identity")
        object.__setattr__(self, "_binding_token", None)


@dataclass(frozen=True)
class WireSemanticObservation:
    """Parser-issued semantics bound to one canonical normalized response."""

    candidate_sha256: str
    accepted_profile_fingerprint: str
    response_sha256: str
    semantic_kind: str
    tool_calls: Tuple[ObservedWireToolCall, ...] = ()
    custom_receipts: Tuple[CustomArgumentValidationReceipt, ...] = ()
    _binding_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._binding_token is not _SEMANTIC_BINDING_TOKEN:
            raise ValueError("wire semantic observation must be parser-issued")
        if not all(_valid_sha256(value) for value in (
            self.candidate_sha256,
            self.accepted_profile_fingerprint,
            self.response_sha256,
        )):
            raise ValueError("wire semantic observation requires exact bindings")
        if self.semantic_kind not in WIRE_SEMANTIC_KINDS:
            raise ValueError("wire semantic observation has an unknown kind")
        if not isinstance(self.tool_calls, tuple) or any(
            not isinstance(item, ObservedWireToolCall) for item in self.tool_calls
        ):
            raise ValueError("wire semantic tool observations must be typed")
        call_ids = [item.call_id for item in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("wire semantic observation contains duplicate call ids")
        if not isinstance(self.custom_receipts, tuple) or any(
            not isinstance(item, CustomArgumentValidationReceipt)
            for item in self.custom_receipts
        ):
            raise ValueError("custom semantic receipts must be typed")
        object.__setattr__(self, "_binding_token", None)

    @property
    def observed_tool_names(self) -> Tuple[str, ...]:
        return tuple(sorted({item.tool_name for item in self.tool_calls}))


def observe_wire_semantics(
    *,
    candidate: WireCandidateManifest,
    normalized_response: Mapping[str, Any],
    normalized_usage: Mapping[str, Any],
    custom_receipts: Sequence[CustomArgumentValidationReceipt] = (),
) -> WireSemanticObservation:
    """Parse canonical response semantics; callers cannot supply a success label."""
    validate_normalized_wire_success(normalized_response, normalized_usage)
    raw_calls = normalized_response.get("tool_calls")
    if raw_calls is None:
        raw_calls = ()
    if not isinstance(raw_calls, (list, tuple)):
        raise ValueError("normalized wire tool calls must be a list")
    observed = []
    seen_ids = set()
    for call in raw_calls:
        if not isinstance(call, Mapping) or call.get("type") != "function":
            raise ValueError("normalized wire tool call is not canonical function form")
        function = call.get("function")
        call_id_value = call.get("id")
        name_value = function.get("name") if isinstance(function, Mapping) else None
        call_id = call_id_value if isinstance(call_id_value, str) else ""
        name = name_value if isinstance(name_value, str) else ""
        arguments = function.get("arguments") if isinstance(function, Mapping) else None
        if (
            not call_id.strip()
            or call_id != call_id.strip()
            or call_id in seen_ids
            or not name.strip()
            or name != name.strip()
            or not isinstance(arguments, str)
        ):
            raise ValueError("normalized wire tool call lacks exact canonical identity")
        seen_ids.add(call_id)
        observed.append(ObservedWireToolCall(
            call_id=call_id,
            tool_name=name,
            arguments_sha256=canonical_sha256(arguments),
            _binding_token=_SEMANTIC_BINDING_TOKEN,
        ))
    profile = candidate.accepted_profile
    if observed:
        if profile.provider == "anthropic" and profile.api_surface == "messages":
            semantic_kind = "anthropic_tool_use"
        elif profile.tool_dialect == "openai_chat_custom":
            semantic_kind = "custom_tool_call_json_object"
        else:
            semantic_kind = "function_tool_call"
    elif profile.provider == "anthropic" and profile.api_surface == "messages":
        semantic_kind = "anthropic_message"
    else:
        semantic_kind = "chat_message"
    if not observed:
        content = normalized_response.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("wire semantic success requires non-empty response content")
    if not wire_semantic_kind_allowed(profile, semantic_kind):
        raise ValueError("provider response is not semantic success for the wire profile")
    if candidate.requires_tool_call and not observed:
        raise ValueError("wire candidate required a tool call")
    if candidate.forbids_tool_call and observed:
        raise ValueError("wire candidate forbade tool calls")
    if (
        semantic_kind == "chat_message"
        and profile.tool_dialect == "openai_chat_custom"
        and any(
            item.action.get("kind") == "replace_dialect"
            and item.action.get("to") == "openai_chat_custom"
            for item in candidate.pending
        )
    ):
        raise ValueError("pending custom dialect evidence requires a custom tool call")
    sidecars = tuple(custom_receipts)
    if any(not isinstance(item, CustomArgumentValidationReceipt) for item in sidecars):
        raise ValueError("custom semantic sidecars must be typed")
    if profile.tool_dialect == "openai_chat_custom":
        by_id = {item.tool_call_id: item for item in sidecars}
        if len(by_id) != len(sidecars) or len(sidecars) != len(observed):
            raise ValueError("custom success requires one sidecar per exact tool call")
        for call in observed:
            receipt = by_id.get(call.call_id)
            binding = (
                candidate.custom_catalog.schema_binding(call.tool_name)
                if candidate.custom_catalog is not None
                else None
            )
            if not (
                receipt is not None
                and binding is not None
                and receipt.candidate_sha256 == candidate.candidate_sha256
                and receipt.catalog_sha256 == candidate.custom_catalog_sha256
                and receipt.tool_name == call.tool_name
                and receipt.schema_sha256 == binding.schema_sha256
                and receipt.arguments_sha256 == call.arguments_sha256
                and receipt.proves_wire_acceptance
            ):
                raise ValueError("custom sidecar differs from normalized call semantics")
    elif sidecars:
        raise ValueError("non-custom response cannot carry custom sidecars")
    observed_names = {item.tool_name for item in observed}
    if profile.tool_dialect == "openai_chat_custom" and not observed_names.issubset(
        candidate.custom_catalog_tool_names
    ):
        raise ValueError("custom response called a tool outside the physical catalog")
    allowed_names = set(candidate.tool_choice_names)
    if allowed_names and not observed_names.issubset(allowed_names):
        raise ValueError("provider called a tool outside the exact tool-choice contract")
    return WireSemanticObservation(
        candidate_sha256=candidate.candidate_sha256,
        accepted_profile_fingerprint=profile.fingerprint,
        response_sha256=canonical_sha256(normalized_response),
        semantic_kind=semantic_kind,
        tool_calls=tuple(observed),
        custom_receipts=sidecars,
        _binding_token=_SEMANTIC_BINDING_TOKEN,
    )


@dataclass(frozen=True)
class WireCompatibilityReceipt:
    """Terminal semantic success tied to the candidate actually dispatched."""

    candidate: WireCandidateManifest
    physical_attempt: "PhysicalAttemptCapture"
    semantic_observation: WireSemanticObservation
    observed_at: str = field(default_factory=utc_now_iso, init=False)
    _binding_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._binding_token is not _RECEIPT_BINDING_TOKEN:
            raise ValueError("wire receipt must be created by bind_wire_compatibility_receipt")
        validate_physical_wire_attempt(self.candidate, self.physical_attempt)
        if not isinstance(self.semantic_observation, WireSemanticObservation):
            raise ValueError("wire receipt requires a parser-issued semantic observation")
        if (
            self.semantic_observation.candidate_sha256 != self.candidate.candidate_sha256
            or self.semantic_observation.accepted_profile_fingerprint
            != self.candidate.accepted_profile.fingerprint
        ):
            raise ValueError("wire semantic observation belongs to another candidate")
        provider_status = self.physical_attempt.provider_status_code
        if provider_status is not None and (
            type(provider_status) is not int or not 200 <= provider_status < 300
        ):
            raise ValueError("non-successful provider settlement cannot produce a wire receipt")
        object.__setattr__(self, "_binding_token", None)

    @property
    def response_sha256(self) -> str:
        return self.semantic_observation.response_sha256

    @property
    def semantic_kind(self) -> str:
        return self.semantic_observation.semantic_kind

    @property
    def observed_tool_names(self) -> Tuple[str, ...]:
        return self.semantic_observation.observed_tool_names

    @property
    def custom_receipts(self) -> Tuple[CustomArgumentValidationReceipt, ...]:
        return self.semantic_observation.custom_receipts

    @property
    def accepted_profile(self) -> RequestWireProfile:
        return self.candidate.accepted_profile

    @property
    def candidate_sha256(self) -> str:
        return self.candidate.candidate_sha256

    @property
    def attempt_id(self) -> str:
        return self.physical_attempt.attempt_id

    @property
    def pending(self) -> Tuple[PendingWireAction, ...]:
        return self.candidate.pending


def bind_wire_compatibility_receipt(
    *,
    candidate: WireCandidateManifest,
    physical_attempt: "PhysicalAttemptCapture",
    semantic_observation: WireSemanticObservation,
) -> WireCompatibilityReceipt:
    """Bind parser-issued semantic evidence to the settled provider attempt."""
    return WireCompatibilityReceipt(
        candidate=candidate,
        physical_attempt=physical_attempt,
        semantic_observation=semantic_observation,
        _binding_token=_RECEIPT_BINDING_TOKEN,
    )
