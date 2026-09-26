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
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple
from copy import deepcopy

from ouroboros.context_layout import reference_doc_sections
from ouroboros.reference_books import ReferenceBook
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

ContextProfile = Literal["owner_max", "owner_low", "owner_nano", "task_local_low"]
MeasurementBasis = Literal["fresh_route_usage", "fresh_model_usage", "cold_estimate"]


@dataclass(frozen=True)
class CallContextFit:
    """Arithmetic for one prepared physical input, never dispatch authority."""

    effective_max_tokens: int
    strict_bound_proven: bool
    fit_status: str
    missing_evidence: Tuple[str, ...]
    input_tokens: int
    bound_tokens: Optional[int]
    free_tokens: Optional[int]


def project_tool_result_batch(
    results: List[Dict[str, Any]], messages: List[Dict[str, Any]], tool_schemas: list,
    *, drive_root: pathlib.Path, task_id: str,
    fit_candidate: Callable[[list, list], Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Retain full results before constructing a fitting multi-result view.

    The caller supplies one completed call batch and its actual prospective
    send measurement. No result is appended to the live transcript here;
    source persistence, projection and callback failures never imply full read.
    """
    from ouroboros.artifacts import store_actor_source_bytes

    rows = deepcopy(results)
    full = [str(row["result"]) for row in rows]

    def fit() -> Dict[str, Any]:
        candidate = [*messages, *({"role": "tool", "tool_call_id": row["tool_call_id"],
                                  "content": str(row["result"])} for row in rows)]
        return dict(fit_candidate(deepcopy(candidate), deepcopy(tool_schemas)))

    initial = fit()
    if initial.get("accepted") is True:
        return rows, {"status": "complete", "fit": initial}
    sources = []
    for row, text in zip(rows, full):
        try:
            source = store_actor_source_bytes(drive_root, task_id, category="tool_results",
                source_id=str(row["tool_call_id"]), data=text.encode("utf-8"), extension="txt")
        except (OSError, ValueError):
            source = {}
        sources.append(source)

    def render(index: int, shown: int) -> None:
        row, text, source = rows[index], full[index], sources[index]
        info = {"source_ref": source, "source_status": "ready" if source else "source_unavailable",
                "complete_chars": len(text), "complete_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "requested_range": [0, len(text)], "delivered_range": [0, shown], "text_chars": shown,
                "text_sha256": hashlib.sha256(text[:shown].encode("utf-8")).hexdigest()}
        row["result"] = text[:shown] + "\n[Tool result source view]\n" + json.dumps(info, ensure_ascii=False, separators=(",", ":"))
        row.update(result_partial=shown < len(text), result_source_ref=source,
                   result_source_status=info["source_status"], result_source_view=info)
        if shown < len(text) and isinstance(row.get("result_meta"), dict):
            if "knowledge_source_complete" in row["result_meta"]:
                row["result_meta"]["knowledge_source_complete"] = False

    # Reserve every call's result envelope and source locator before allocating
    # body text. A later result can never disappear because an earlier one grew.
    for index in range(len(rows)):
        render(index, 0)
    minimum = fit()
    if minimum.get("accepted") is not True:
        return rows, {"status": "minimum_view_unfit", "fit": minimum}
    for index, text in enumerate(full):
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            render(index, mid)
            if fit().get("accepted") is True:
                low = mid
            else:
                high = mid - 1
        render(index, low)
    final = fit()
    return rows, {"status": "projected" if final.get("accepted") is True else "minimum_view_unfit", "fit": final}


def resolve_call_context_fit(
    *, input_tokens: int, input_is_exact: bool, caller_max_tokens: int,
    total_target_tokens: Optional[int], route_capacity_tokens: Optional[int],
    minimum_free_tokens: int, tokenizer_template_provenance: Optional[Mapping[str, Any]],
    output_limit_enforced: bool, reasoning_included_in_limit: Optional[bool],
    route_capacity_confirmed: bool = False,
) -> CallContextFit:
    """Choose one output allowance after actual route/template preparation.

    Positive capacity values are sizing observations; only the adapter's
    separately attested facts establish a strict claim. Unknown evidence never
    prohibits a route. Callers keep their existing recovery/admission policy;
    this function creates no send, retry, reservation, or continuation identity.
    """
    if input_tokens < 0 or caller_max_tokens < 0 or minimum_free_tokens < 0:
        raise ValueError("context measurements and token allowances must be nonnegative")
    bounds = [int(value) for value in (total_target_tokens, route_capacity_tokens)
              if value is not None and value > 0]
    bound = min(bounds) if bounds else None
    free = bound - input_tokens if bound is not None else None
    effective = min(caller_max_tokens, max(0, free)) if free is not None else caller_max_tokens
    missing = []
    if not input_is_exact:
        missing.append("exact_input_measurement")
    if not tokenizer_template_provenance:
        missing.append("tokenizer_template_provenance")
    if not route_capacity_confirmed or not route_capacity_tokens:
        missing.append("confirmed_serving_capacity")
    if not output_limit_enforced:
        missing.append("enforced_output_limit")
    if reasoning_included_in_limit is not True:
        missing.append("reasoning_within_measured_window")
    fit_status = ("unknown_capacity" if free is None else "unfit" if free < 0
                  else "insufficient_headroom" if free < minimum_free_tokens else "fits")
    # An estimate alone cannot prescribe a zero-output physical request. Keep
    # the caller's allowance and disclose that the proposed view needs fitting.
    if not input_is_exact and effective == 0 and caller_max_tokens > 0:
        effective = caller_max_tokens
    return CallContextFit(effective, fit_status == "fits" and not missing,
                          fit_status, tuple(missing), input_tokens, bound, free)


@dataclass(frozen=True)
class ContextFitProjection:
    """One deterministic Low/Max rendering of a shared immutable context core."""

    mode: str
    system_content_json: str
    estimated_tokens: int
    calibrated_tokens: int
    calibration_ratio: float
    fits_known_window: Optional[bool]
    user_content_json: Optional[str] = None

    def system_message(self) -> Dict[str, Any]:
        from ouroboros.llm_messages import STABLE_PREFIX_BLOCKS_KEY

        # Declared for the OpenAI-family and Claudexor send projection (llm_messages.split_leading_system_prefix):
        # block 0 (SYSTEM.md, BIBLE, reference docs) is byte-stable across conversations, while
        # the semi-stable memory block changes with every consolidation (8 of 59 Aika events),
        # so keeping it in the cached unit would lose the whole unit on those events.
        return {"role": "system", "content": json.loads(self.system_content_json), STABLE_PREFIX_BLOCKS_KEY: 1}


@dataclass(frozen=True)
class MainFitMeasurement:
    route_fp: str
    round_id: str
    profile: ContextProfile
    rendered_mode: Literal["max", "low", "nano"]
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
    model_role: str = "main"
    model_route: Dict[str, Any] = field(default_factory=dict)
    evidence_source: str = ""
    nano_projection: Optional[ContextFitProjection] = None

    def projection(self, mode: str) -> ContextFitProjection:
        if str(mode or "").lower() == "nano" and self.nano_projection is not None:
            return self.nano_projection
        return self.low_projection if str(mode or "").lower() == "low" else self.max_projection

    def messages_for(self, mode: str) -> List[Dict[str, Any]]:
        projection = self.projection(mode)
        return [
            projection.system_message(),
            {"role": "user", "content": json.loads(projection.user_content_json or self.user_content_json)},
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
    reference_books: Tuple[ReferenceBook, ...] = ()
    compact_reference_docs: bool = False
    reference_book_errors: Tuple[str, ...] = ()


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
            context_mode="low" if core.compact_reference_docs else mode,
            include_development=core.docs_need_development,
            architecture_text=core.architecture_md,
            development_text=core.development_md,
            books=core.reference_books,
        )
    )
    static_parts.extend(core.reference_book_errors)
    # Stable governance/policy is first; mutable task evidence is last: the
    # cache-friendly ordering for Anthropic-style breakpoints. OpenAI's public API
    # (and the Codex backend) caches the whole leading system section as one unit,
    # so their send copies keep only block 0 there (declared in ``system_message``).
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
    rendered_mode: Literal["max", "low", "nano"],
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
    from ouroboros.context_budget import OWNER_LOW_TARGET_TOKENS, OWNER_NANO_TARGET_TOKENS, NANO_MIN_HEADROOM_TOKENS

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
    reserve = NANO_MIN_HEADROOM_TOKENS if profile == "owner_nano" else int(plan.output_reserve_tokens or 0)
    total = estimated_input + reserve
    target = (OWNER_NANO_TARGET_TOKENS if profile == "owner_nano"
              else OWNER_LOW_TARGET_TOKENS if profile == "owner_low" else None)
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


def _context_route(task: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve the same effective settings and account identity on success or failure."""
    from ouroboros.capability_evidence import model_account_options
    from ouroboros.config import runtime_settings
    from ouroboros.gateway.settings import _active_main_route
    from ouroboros.server_runtime import apply_runtime_provider_defaults

    settings, _changed, _keys = apply_runtime_provider_defaults(runtime_settings())
    local_override = task.get("use_local_model")
    route = _active_main_route(
        settings, model_override=str(task.get("model") or "").strip(),
        use_local_override=(
            bool(local_override) if local_override is not None else None
        ),
    )
    from ouroboros.model_slots import task_model_binding
    role, account = task_model_binding(task)
    route["model_role"] = role
    route["options"] = {}
    if route["provider"] == "claudexor":
        route["options"] = model_account_options(
            route["model"], role=role, settings=settings,
            credential_profile_id=account,
            model_route=task.get("model_route"),
        )
    return route, settings


def resolve_context_fit_route(
    task: Dict[str, Any],
    *,
    allow_fetch: bool,
) -> Tuple[Dict[str, Any], Any]:
    """Resolve capacity from effective settings and exact account evidence.

    Auto discovery is advertised preparation evidence, not proof of the account
    that will serve the next operation. The caller replaces ``model_route`` from
    actual operation receipts when it rebinds. A manual role window is an owner
    sizing assertion only; it neither changes the provider nor writes a scope ack.
    """
    from dataclasses import replace
    from ouroboros.capability_evidence import SOURCE_USER_SETTING, STATUS_ASSERTED, probe
    from ouroboros.config import DATA_DIR
    from ouroboros.model_slots import MODEL_CONTEXT_WINDOWS_KEY, model_role_option

    route, settings = _context_route(task)
    evidence = probe(
        DATA_DIR,
        provider=route["provider"],
        model=route["model"],
        base_url=route["base_url"],
        use_local=route["use_local"],
        allow_fetch=allow_fetch,
        options=route["options"] or None,
    )
    window = int(model_role_option(MODEL_CONTEXT_WINDOWS_KEY, route["model_role"],
                                   settings=settings))
    if window:
        evidence = replace(evidence, window_tokens=window, status=STATUS_ASSERTED,
                           source=SOURCE_USER_SETTING, stale=False,
                           detail=f"Context for {route['model_role']} asserted by user; provider limit unchanged")
    return route, evidence


def _failed_route_evidence(task: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    from ouroboros.capability_evidence import route_fingerprint

    route, _settings = _context_route(task)
    evidence = SimpleNamespace(
        route_fp=route_fingerprint(
            provider=route["provider"],
            base_url=route["base_url"],
            model=route["model"],
            options=route["options"] or None,
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
    if preferred not in {"low", "max", "nano"}:
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
    rendered = {}
    input_source = None

    def _projection(mode: str) -> ContextFitProjection:
        nonlocal input_source
        from ouroboros.context_budget import NANO_MIN_HEADROOM_TOKENS, OWNER_NANO_TARGET_TOKENS

        view_mode = "low" if core.compact_reference_docs or mode == "nano" else mode
        if view_mode not in rendered:
            system_content = _render_context_system_content(env, core, mode=view_mode)
            messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
            rendered[view_mode] = (json.dumps(system_content, ensure_ascii=False, sort_keys=True),
                                   estimate_context_prompt_tokens(messages))
        system_content_json, estimated = rendered[view_mode]
        user_projection = None
        target = OWNER_NANO_TARGET_TOKENS if mode == "nano" else None
        if preferred == "nano" and target is not None and estimated + NANO_MIN_HEADROOM_TOKENS > target:
            # The original owner input stays exact in the captured core and
            # existing source store. Only its initial Nano delivery changes;
            # this is neither the external-assignment compiler nor a summary.
            try:
                from ouroboros.artifacts import store_actor_source_bytes
                if input_source is None:
                    input_source = store_actor_source_bytes(env.drive_root, str(task["id"]),
                        category="context_checkpoints", source_id="task_input",
                        data=core.user_content_json.encode("utf-8"), extension="json")
                source_content = (
                    "[Exact task input source]\nThe complete original user input is stored as JSON at this source. "
                    "Read its full content through the given reader in ranges before substantive decisions. "
                    "This pointer is not a summary or a change to the task.\n"
                    + json.dumps(input_source, ensure_ascii=False, sort_keys=True)
                )
                source_estimate = estimate_context_prompt_tokens([
                    {"role": "system", "content": json.loads(system_content_json)},
                    {"role": "user", "content": source_content}])
                if source_estimate < estimated:
                    user_projection = json.dumps(source_content, ensure_ascii=False)
                    estimated = source_estimate
            except (OSError, ValueError, KeyError):
                log.warning("Exact task input source could not be retained; preserving complete input", exc_info=True)
        calibrated = int(estimated * ratio)
        fits = (
            calibrated + (NANO_MIN_HEADROOM_TOKENS if mode == "nano" else output_reserve)
            <= (min(OWNER_NANO_TARGET_TOKENS, int(evidence.window_tokens)) if mode == "nano"
                else int(evidence.window_tokens or 0))
            if known_window
            else None
        )
        return ContextFitProjection(
            mode=mode,
            system_content_json=system_content_json,
            estimated_tokens=estimated,
            calibrated_tokens=calibrated,
            calibration_ratio=ratio,
            fits_known_window=fits,
            user_content_json=user_projection,
        )

    max_projection = _projection("max")
    low_projection = _projection("low")
    nano_projection = _projection("nano")
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
            "compact_reference_docs": core.compact_reference_docs,
            "reference_book_errors": core.reference_book_errors,
            "reference_sources": [(book.book_id, source.source_path, source.sha256)
                                  for book in core.reference_books for source in (book.entrypoint, *book.chapters)],
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
        nano_projection=nano_projection,
        model_role=str(route.get("model_role") or "main"),
        model_route={
            "source": str(getattr(evidence, "source_id", "") or ""),
            "model": str(route["model"]).partition("=")[2],
            "credentialProfileId": str(getattr(evidence, "credential_profile_id", "") or ""),
            "accountFingerprint": str(getattr(evidence, "account_fingerprint", "") or ""),
        } if route["provider"] == "claudexor" else {},
        evidence_source=str(getattr(evidence, "source", "") or ""),
    )
