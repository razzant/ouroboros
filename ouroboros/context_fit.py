"""Task-local context-fit projections for the ordinary Main model path.

This module is deliberately data-only around the existing context builder and
capability-evidence SSOT.  It does not own routing, provider retries, or global
context-mode state; callers supply the captured context core and exact-route
resolver so ``ouroboros.context`` remains the public compatibility surface.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pathlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from ouroboros.context_layout import reference_doc_sections
from ouroboros.utils import estimate_tokens

log = logging.getLogger(__name__)


def extract_plain_text_from_content(content: Any) -> str:
    """Text of a string or multipart message content, for transcript sealing.

    The one extractor lives in ``loop_messages`` (the retired ``delivery_protocol``
    leaf carried a copy). Read at CALL time: ``loop_messages`` imports
    ``ouroboros.llm`` at module top and the LLM lanes import this module, so a
    top-level import here would be an import cycle.
    """
    from ouroboros.loop_messages import _extract_plain_text_from_content

    return _extract_plain_text_from_content(content)

ContextProfile = Literal["owner_max", "owner_low", "task_local_low"]
MeasurementBasis = Literal["fresh_route_usage", "fresh_model_usage", "cold_estimate"]


@dataclass(frozen=True)
class ContextFitProjection:
    """One deterministic Low/Max rendering of a shared immutable context core."""

    mode: str
    system_content_json: str
    estimated_tokens: int
    calibrated_tokens: int
    calibration_ratio: float
    fits_known_window: Optional[bool]

    def system_message(self) -> Dict[str, Any]:
        return {"role": "system", "content": json.loads(self.system_content_json)}


@dataclass(frozen=True)
class MainFitMeasurement:
    route_fp: str
    round_id: str
    profile: ContextProfile
    rendered_mode: Literal["max", "low"]
    estimated_input_tokens: int
    response_reserve_tokens: int
    target_total_tokens: Optional[int]
    capacity_total_tokens: Optional[int]
    measurement_basis: MeasurementBasis
    measurement_density: float
    target_deficit_tokens: Optional[int]
    capacity_deficit_tokens: Optional[int]
    reclaim_goal_tokens: int


@dataclass(frozen=True)
class MainFitDisposition:
    measurement: MainFitMeasurement
    action: Literal["send", "reclaim_once", "send_target_miss"]
    automatic_pass_used: bool
    predicted_capacity_miss: bool


@dataclass(frozen=True)
class ContextFitPlan:
    """Task-local context fit decision for the ordinary Main agent path.

    The two projections are rendered from the same captured core.  The plan is
    deliberately data-only: it neither owns provider routing nor changes the
    owner-selected global context mode.  P3 review calls do not use this path.
    """

    core_sha256: str
    preferred_mode: str
    initial_mode: str
    model: str
    provider: str
    route_fp: str
    # Named for ``capability_evidence.CapabilityEvidence``, not paraphrased: the plan
    # is an evidence-shaped record, so ``is_known`` applies to it DIRECTLY instead of
    # each reader restating the three-clause freshness boolean (the wire keys stay
    # ``evidence_status`` / ``evidence_stale``).
    status: str
    stale: bool
    window_tokens: int
    output_reserve_tokens: int
    user_content_json: str
    max_projection: ContextFitProjection
    low_projection: ContextFitProjection

    def projection(self, mode: str) -> ContextFitProjection:
        return self.low_projection if str(mode or "").lower() == "low" else self.max_projection

    def messages_for(self, mode: str) -> List[Dict[str, Any]]:
        projection = self.projection(mode)
        return [
            projection.system_message(),
            {"role": "user", "content": json.loads(self.user_content_json)},
        ]

    def reproject_transcript(
        self,
        messages: List[Dict[str, Any]],
        mode: str,
    ) -> List[Dict[str, Any]]:
        """Replace only the captured system view; preserve every dialogue/tool turn."""
        if not messages:
            return self.messages_for(mode)
        rebuilt = list(messages)
        if str(rebuilt[0].get("role") or "") == "system":
            rebuilt[0] = self.projection(mode).system_message()
        else:
            rebuilt.insert(0, self.projection(mode).system_message())
        return rebuilt

    def projected_tokens_with_tools(
        self,
        mode: str,
        tools: Optional[List[Dict[str, Any]]],
        *,
        provider: str = "",
        reasoning_effort: str = "",
    ) -> int:
        """Calibrated physical prompt projection after schemas are available."""
        projection = self.projection(mode)
        if not (
            str(provider or "").strip().lower() == "openai"
            and str(reasoning_effort or "").strip().lower() not in {"", "none"}
            and tools
        ):
            return int(
                (projection.estimated_tokens + tool_schema_tokens(tools))
                * projection.calibration_ratio
            )
        return int(
            estimate_context_prompt_tokens(
                self.messages_for(mode),
                tools,
                provider=provider,
                reasoning_effort=reasoning_effort,
            )
            * projection.calibration_ratio
        )

@dataclass(frozen=True)
class ContextCore:
    """Single captured context source rendered into deterministic projections."""

    base_prompt: str
    bible_md: str
    architecture_md: str
    development_md: str
    semi_stable_text: str
    dynamic_text: str
    user_content_json: str
    docs_need_development: bool


def _render_context_system_content(
    env: Any,
    core: ContextCore,
    *,
    mode: str,
) -> List[Dict[str, Any]]:
    # D-ARCH (owner, 2026-08-08): the reference-doc form follows the RENDERED
    # mode directly — ARCHITECTURE is full in max for every task class and the
    # nav map in low; DEVELOPMENT inclusion is the caller's mode-independent
    # decision carried on the core (the former per-task force_low_docs lever is
    # gone: workspace binding no longer shapes the docs).
    static_parts = [core.base_prompt, "## BIBLE.md\n\n" + core.bible_md]
    static_parts.extend(
        reference_doc_sections(
            env,
            context_mode=mode,
            include_development=core.docs_need_development,
            architecture_text=core.architecture_md,
            development_text=core.development_md,
        )
    )
    # Stable governance/policy is first; mutable task evidence is last.  This is
    # the cache-friendly ordering recommended by both supported cache routes.
    return [
        {
            "type": "text",
            "text": "\n\n".join(static_parts),
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": core.semi_stable_text,
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": core.dynamic_text},
    ]


def seal_task_transcript(
    messages: List[Dict[str, Any]],
    keep_active: int = 5,
    min_prefix_tokens: int = 2048,
) -> None:
    """Mark ONE stable message-side boundary for provider prompt caching.

    Until enough tool results exist to seal a rolling boundary among them, the
    boundary is the task message itself: without it everything after the two
    SYSTEM markers -- the mutable context block and the task contract -- is
    re-sent uncached every round, which is exactly the short-lived nanny and
    leaf shape. The marker MIGRATES to the rolling tool seal in the same call
    that first qualifies, because a request may declare at most four
    breakpoints (tools, two system blocks and this one): leaving both would
    make the rolling seal a fifth marker and silently drop it.
    """
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            # Flatten the old sealed boundary before choosing a new one.
            msg["content"] = extract_plain_text_from_content(content)
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    if isinstance(first_user, dict) and isinstance(first_user.get("content"), list):
        # Drop this function's own previous task-message marker, so exactly one
        # message-side breakpoint survives whichever branch runs below.
        for block in first_user["content"]:
            if isinstance(block, dict):
                block.pop("cache_control", None)

    tool_indices = [
        i for i, m in enumerate(messages)
        if m.get("role") == "tool"
    ]
    if len(tool_indices) <= keep_active:
        _mark_task_message(first_user)
        return

    seal_candidate_idx = tool_indices[-(keep_active + 1)]

    prefix_text_len = sum(
        len(extract_plain_text_from_content(m.get("content", "")))
        for m in messages[: seal_candidate_idx + 1]
    )
    prefix_tokens = prefix_text_len // 4  # rough 4-chars-per-token estimate

    if prefix_tokens < min_prefix_tokens:
        # Not yet a worthwhile rolling boundary: the task message stays the anchor.
        _mark_task_message(first_user)
        return

    candidate = messages[seal_candidate_idx]
    plain_text = str(candidate.get("content", ""))
    if not plain_text.strip():
        # Anthropic 400s on cache_control attached to an empty text block; never seal
        # an empty tool output as the cache anchor (turns the whole task unanswerable).
        plain_text = "(no tool output)"
    candidate["content"] = [
        {
            "type": "text",
            "text": plain_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]

def _mark_task_message(message: Optional[Dict[str, Any]]) -> None:
    """Anchor the prefix on the task message's last non-empty text block.

    Anthropic rejects a marker on an empty text block, so a task message with
    no text at all is left unmarked rather than padded: the same rule the
    rolling tool seal enforces on empty tool output."""
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if isinstance(content, str):
        if not content.strip():
            return
        content = [{"type": "text", "text": content}]
        message["content"] = content
    if not isinstance(content, list):
        return
    for block in reversed(content):
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and str(block.get("text") or "").strip()
        ):
            block["cache_control"] = {"type": "ephemeral"}
            return


def tool_schema_tokens(tools: Optional[List[Dict[str, Any]]]) -> int:
    """chars/4 estimate of the TOOL-SCHEMA segment of a prompt.

    One seam for every consumer that has to account for the schemas: they are sent
    on the wire beside ``messages``, so any measure built from the transcript alone
    silently omits them (~148K chars / ~37K tokens on the submarine traces — enough
    to make an emergency-compaction trigger fire a whole tool envelope late).
    """
    if not tools:
        return 0
    return estimate_tokens(json.dumps(tools, ensure_ascii=False, sort_keys=True, default=str))


def bounded_prompt_tokens_for_payload(prompt_payload: Dict[str, Any], fallback_chars: int) -> int:
    """The density witness's basis: the fit estimator's own token count for a
    request payload (messages + tools, images at the proxy), or ``fallback_chars
    // 4`` when there is no message list. Kept beside the estimator so the two
    can never diverge; ``estimate_message_chars`` dropped tool_call objects and
    made density ~1.4x high on the tool-heavy shape (measure_main_fit multiplies
    THIS quantity)."""
    try:
        messages = prompt_payload.get("messages")
        if isinstance(messages, list):
            return int(estimate_context_prompt_tokens(
                messages, prompt_payload.get("tools") or prompt_payload.get("functions")))
    except Exception:
        pass
    return max(0, int(fallback_chars) // 4)


def messages_carry_native_images(messages: Any) -> bool:
    """Whether a message carries an image part the proxy bounds instead of pricing."""
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(
            isinstance(part, dict) and str(part.get("type") or "") in {"image", "image_url"}
            for part in content
        ):
            return True
    return False


def estimate_context_prompt_tokens(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    *,
    provider: str = "",
    reasoning_effort: str = "",
) -> int:
    """Estimate the complete inspectable context shape with bounded images."""
    from ouroboros.anthropic_native_custody import (
        context_custody_proxy,
        custody_private_key,
    )
    from ouroboros.context_budget import IMAGE_BLOCK_CHAR_EQUIVALENT
    from ouroboros.openai_chat_dispatch import direct_openai_context_projections

    def project(value: Any) -> Any:
        if isinstance(value, dict):
            if any(custody_private_key(key) for key in value):
                value = context_custody_proxy(value)
            if str(value.get("type") or "") in {"image", "image_url"}:
                return {
                    "type": str(value.get("type") or "image"),
                    "image_token_proxy": "#" * IMAGE_BLOCK_CHAR_EQUIVALENT,
                }
            return {
                str(key): project(item)
                for key, item in value.items()
                if str(key) != "_context_capsule"
            }
        if isinstance(value, list):
            return [project(item) for item in value]
        return value

    projections = direct_openai_context_projections(
        messages, tools, provider=provider, reasoning_effort=reasoning_effort,
    )
    return max(
        int(estimate_tokens(json.dumps(
            {"messages": project(projected_messages), "tools": projected_tools},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )))
        for projected_messages, projected_tools in projections
    )


def _route_calibration_ratio(
    drive_root: Optional[pathlib.Path],
    route_fp: str,
    model: str,
) -> float:
    """Fresh exact-route/model witness, never an event-tail or orphan maximum.

    ``None`` reads the canonical host evidence root (one observation store).
    """
    try:
        from ouroboros.capability_evidence import (
            canonical_evidence_root, resolve_main_token_density,
        )

        root = drive_root if drive_root is not None else canonical_evidence_root()
        return float(resolve_main_token_density(root, route_fp, model)[0])
    except Exception:
        log.debug("Fresh route token calibration unavailable", exc_info=True)
        return 1.0


def measure_main_fit(
    plan: ContextFitPlan,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    *,
    drive_root: Optional[pathlib.Path] = None,
    profile: ContextProfile,
    rendered_mode: Literal["max", "low"],
    round_id: str,
    automatic_pass_used: bool = False,
    reasoning_effort: str = "",
) -> MainFitDisposition:
    """Measure one sealed Main candidate against owner target T and route W.

    ``drive_root=None`` reads density from the canonical host evidence root
    (one observation store) — a child task's own drive must not be consulted.
    """
    from ouroboros.capability_evidence import (
        canonical_evidence_root, is_known, resolve_main_token_density,
    )
    from ouroboros.context_budget import OWNER_LOW_TARGET_TOKENS

    if drive_root is None:
        drive_root = canonical_evidence_root()
    density, basis = resolve_main_token_density(drive_root, plan.route_fp, plan.model)
    density = float(density)
    estimated_input = int(math.ceil(estimate_context_prompt_tokens(
        messages,
        tools,
        provider=plan.provider,
        reasoning_effort=reasoning_effort,
    ) * density))
    reserve = int(plan.output_reserve_tokens or 0)
    total = estimated_input + reserve
    target = OWNER_LOW_TARGET_TOKENS if profile == "owner_low" else None
    capacity = int(plan.window_tokens or 0) if is_known(plan, require_fresh=True) else None
    target_deficit = max(0, total - target) if target is not None else None
    capacity_deficit = max(0, total - capacity) if capacity is not None else None
    goal = max(
        [value for value in (target_deficit, capacity_deficit) if value is not None]
        or [0]
    )
    measurement = MainFitMeasurement(
        route_fp=str(plan.route_fp or ""),
        round_id=str(round_id or ""),
        profile=profile,
        rendered_mode=rendered_mode,
        estimated_input_tokens=estimated_input,
        response_reserve_tokens=reserve,
        target_total_tokens=target,
        capacity_total_tokens=capacity,
        measurement_basis=basis,
        measurement_density=density,
        target_deficit_tokens=target_deficit,
        capacity_deficit_tokens=capacity_deficit,
        reclaim_goal_tokens=goal,
    )
    if goal > 0 and not automatic_pass_used:
        action: Literal["send", "reclaim_once", "send_target_miss"] = "reclaim_once"
    elif target_deficit and target_deficit > 0:
        action = "send_target_miss"
    else:
        action = "send"
    return MainFitDisposition(
        measurement=measurement,
        action=action,
        automatic_pass_used=bool(automatic_pass_used),
        predicted_capacity_miss=bool(capacity_deficit and capacity_deficit > 0),
    )


def resolve_context_fit_route(
    task: Dict[str, Any],
    *,
    allow_fetch: bool,
) -> Tuple[Dict[str, Any], Any]:
    """Resolve one exact route through the existing settings/evidence SSOT.

    The document the route is read from is the provider-normalized EFFECTIVE
    settings — the same derivation the task-start projection and the settings
    GET make — never the owner-raw read: the read seam carries the vocabulary
    normalization only, and a direct-provider install with no explicit model
    resolves its main slot through the provider normalization alone, so the
    owner-raw route would name a model the loop never runs."""
    from ouroboros.capability_evidence import probe
    from ouroboros.config import DATA_DIR, load_settings
    from ouroboros.gateway.settings import _active_main_route
    from ouroboros.server_runtime import apply_runtime_provider_defaults

    settings, _changed, _keys = apply_runtime_provider_defaults(load_settings())
    model = str(task.get("model") or "").strip()
    local_override = task.get("use_local_model")
    route = _active_main_route(
        settings,
        model_override=model,
        use_local_override=(
            bool(local_override) if local_override is not None else None
        ),
    )
    evidence = probe(
        DATA_DIR,
        provider=route["provider"],
        model=route["model"],
        base_url=route["base_url"],
        use_local=route["use_local"],
        allow_fetch=allow_fetch,
    )
    return route, evidence


def _failed_route_evidence(task: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    from ouroboros.capability_evidence import route_fingerprint
    from ouroboros.config import load_settings
    from ouroboros.gateway.settings import _active_main_route
    from ouroboros.server_runtime import apply_runtime_provider_defaults

    settings, _changed, _keys = apply_runtime_provider_defaults(load_settings())
    route = _active_main_route(
        settings,
        model_override=str(task.get("model") or ""),
        use_local_override=(
            bool(task.get("use_local_model"))
            if task.get("use_local_model") is not None
            else None
        ),
    )
    evidence = SimpleNamespace(
        route_fp=route_fingerprint(
            provider=route["provider"],
            base_url=route["base_url"],
            model=route["model"],
        ),
        status="failed",
        stale=True,
        window_tokens=0,
    )
    return route, evidence


def build_context_fit_plan(
    env: Any,
    core: ContextCore,
    task: Dict[str, Any],
    *,
    preferred_mode: str,
    route_resolver: Callable[..., Tuple[Dict[str, Any], Any]],
) -> ContextFitPlan:
    """Deterministically project one captured core into ordinary-task Max and Low."""
    preferred = str(preferred_mode or "max").strip().lower()
    if preferred not in {"low", "max"}:
        preferred = "max"

    meta = task.get("task_metadata") if isinstance(task.get("task_metadata"), dict) else {}
    is_subagent = str(
        task.get("delegation_role") or meta.get("delegation_role") or ""
    ).strip().lower() == "subagent"
    try:
        route, evidence = route_resolver(task, allow_fetch=not is_subagent)
    except Exception:
        log.debug("Context-fit route evidence unavailable; preserving Max", exc_info=True)
        route, evidence = _failed_route_evidence(task)

    user_content = json.loads(core.user_content_json)
    # Keep the fit projection tied to the physical dispatch contract instead of
    # duplicating its output reservation.  The lazy import avoids coupling the
    # data-only fit representation to the high-level model loop.
    from ouroboros.capability_evidence import is_known
    from ouroboros.loop_llm_call import MAIN_LOOP_MAX_TOKENS

    output_reserve = MAIN_LOOP_MAX_TOKENS
    # One observation store: witnesses are written at settlement into the
    # canonical host root, so a child task's own drive must not be consulted.
    ratio = _route_calibration_ratio(
        None,
        str(evidence.route_fp or ""),
        str(route["model"] or ""),
    )
    known_window = is_known(evidence, require_fresh=True)

    def _projection(mode: str) -> ContextFitProjection:
        system_content = _render_context_system_content(env, core, mode=mode)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        estimated = estimate_context_prompt_tokens(messages)
        calibrated = int(estimated * ratio)
        fits = (
            calibrated + output_reserve <= int(evidence.window_tokens or 0)
            if known_window
            else None
        )
        return ContextFitProjection(
            mode=mode,
            system_content_json=json.dumps(
                system_content,
                ensure_ascii=False,
                sort_keys=True,
            ),
            estimated_tokens=estimated,
            calibrated_tokens=calibrated,
            calibration_ratio=ratio,
            fits_known_window=fits,
        )

    max_projection = _projection("max")
    low_projection = _projection("low")
    # Prediction may request mutable-history reclaim, but it never changes the
    # owner's document projection. Task-local Low is authorized only after a
    # real provider overflow on this route.
    initial_mode = preferred

    core_payload = json.dumps(
        {
            "base_prompt": core.base_prompt,
            "bible_md": core.bible_md,
            "architecture_md": core.architecture_md,
            "development_md": core.development_md,
            "semi_stable_text": core.semi_stable_text,
            "dynamic_text": core.dynamic_text,
            "user_content": user_content,
            "docs_need_development": core.docs_need_development,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return ContextFitPlan(
        core_sha256=hashlib.sha256(core_payload.encode("utf-8")).hexdigest(),
        preferred_mode=preferred,
        initial_mode=initial_mode,
        model=str(route["model"] or ""),
        provider=str(route["provider"] or ""),
        route_fp=str(evidence.route_fp or ""),
        status=str(evidence.status or ""),
        stale=bool(evidence.stale),
        window_tokens=int(evidence.window_tokens or 0),
        output_reserve_tokens=output_reserve,
        user_content_json=core.user_content_json,
        max_projection=max_projection,
        low_projection=low_projection,
    )
