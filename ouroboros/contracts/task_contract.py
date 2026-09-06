"""Task contract normalization.

The contract is a durable, LLM-readable description of what this task is trying
to accomplish.  It is not a deterministic success oracle: code records the
declared goal, constraints, resources, and artifacts; LLM review/evaluation
interprets whether the objective was met.
"""

from __future__ import annotations

import copy
import json
from hashlib import sha256
from typing import Any, Dict, Mapping, Optional, Tuple


_BOOLEAN_RESOURCE_NAMES = frozenset({
    "web",
    "allow_web",
    "network",
    "allow_network",
    "internet",
    "external_network",
})


def normalize_allowed_resources(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key, raw in value.items():
        name = str(key or "").strip()
        if not name:
            continue
        if isinstance(raw, bool):
            out[name] = raw
        elif isinstance(raw, (int, float)) and raw in (0, 1):
            out[name] = bool(raw)
        elif isinstance(raw, str):
            text = raw.strip().lower()
            if text in {"1", "true", "yes", "y", "on", "allowed", "allow", "enabled", "enable"}:
                out[name] = True
            elif text in {"0", "false", "no", "n", "off", "denied", "deny", "disabled", "disable", "blocked", "block", "forbidden"}:
                out[name] = False
            elif name in _BOOLEAN_RESOURCE_NAMES:
                out[name] = False
            else:
                out[name] = raw
        elif raw is not None:
            out[name] = raw
    return out


def normalize_disabled_tools(value: Any) -> list[str]:
    """A clean, de-duplicated list of tool names the task is NOT allowed to use.

    This is the declarative tool-policy surface a benchmark adapter (or any
    caller) uses to withhold specific capabilities — e.g. disabling the agent's
    own web-search/browser/VLM tools for a faithful run while leaving shell
    network egress (git/pip) intact. It is independent of ``allowed_resources``
    (which gates resource AXES like web/network), so it never triggers the
    web<->network cross-implication in the registry resource gate.
    """
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return []
    seen: list[str] = []
    for item in items:
        name = str(item or "").strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def normalize_attachment_manifest(value: Any) -> list[Dict[str, Any]]:
    """Copy the additive complete attachment manifest into task authority.

    The staging owner already sanitizes labels and rejection reasons.  This
    normalizer keeps only the closed manifest vocabulary so a caller cannot use
    the frozen task contract as an arbitrary metadata bag.
    """

    if not isinstance(value, list):
        return []
    rows: list[Dict[str, Any]] = []
    allowed = (
        "ordinal", "status", "reason", "label", "root", "relpath",
        "abs_path", "mime", "is_image",
    )
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            continue
        row = {key: item[key] for key in allowed if key in item}
        try:
            row["ordinal"] = max(0, int(row.get("ordinal", index)))
        except (TypeError, ValueError):
            row["ordinal"] = index
        status = str(row.get("status") or "staged")
        row["status"] = status if status in {"staged", "rejected"} else "rejected"
        row["reason"] = str(row.get("reason") or "")
        row["label"] = str(row.get("label") or f"attachment {index + 1}")
        rows.append(row)
    return rows


def _opt_nonneg_int(value: Any) -> Any:
    """A non-negative int, or None when unset/blank (meaning 'use the config cap')."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _normalized_intent_note(value: Any) -> str:
    """Preserve the complete delegation intent in the durable authority.

    Cutting or flattening it here used to make
    the normalized contract and every descendant fingerprint certify a prefix
    as the parent's complete nesting/review constraint.
    """
    return str(value or "").strip()


_DEPTH_PROVENANCE_KEYS = (
    "requested_depth",
    "permitted_depth",
    "attempted_depth",
    "achieved_depth",
)


def normalize_depth_provenance(value: Any) -> Dict[str, Any]:
    """Normalize additive depth facts without inventing a requested depth."""
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key in _DEPTH_PROVENANCE_KEYS:
        if key in value:
            out[key] = _opt_nonneg_int(value.get(key))
    return out


def normalize_delegation_budget(value: Any) -> Dict[str, Any]:
    """The typed delegation-budget block — the SSOT for what delegation a task is
    licensed to do, so a parent's 'you may delegate / mutate / fan out further'
    intent propagates STRUCTURALLY to children instead of being lost in freeform
    objective prose (the cyber-racing failure). Enforcement of depth/active caps
    stays where it already is (config + scheduler); this block carries INTENT and
    the remaining budget the orchestrator decrements per generation. Absent input
    -> conservative defaults: a task may delegate and fan out, but mutation must be
    explicitly granted, and ``depth_remaining``/``max_children`` default to None
    (the configured caps apply)."""
    v = value if isinstance(value, Mapping) else {}
    depth_remaining = _opt_nonneg_int(v.get("depth_remaining"))
    budget = {
        "may_delegate": normalize_bool(v.get("may_delegate", True)),
        "may_mutate": normalize_bool(v.get("may_mutate", False)),
        "may_fan_out": normalize_bool(v.get("may_fan_out", True)),
        "depth_remaining": depth_remaining,
        "max_children": _opt_nonneg_int(v.get("max_children")),
        "intent_note": _normalized_intent_note(v.get("intent_note")),
    }
    if "depth_provenance" in v:
        # Only an explicitly authored projection enters the frozen contract.
        # Inferring one from a legacy depth_remaining field would rewrite old
        # work-order hashes during restart/recovery.
        budget["depth_provenance"] = normalize_depth_provenance(v.get("depth_provenance"))
    return budget


# ABI 7.0 (owner Q10=A): the legacy ``until_deadline`` policy alias and the
# never-consumed ``stall_rounds_threshold`` knob are REMOVED. An unknown
# policy (including the retired spelling) normalizes to "fixed".
VALID_IMPROVEMENT_POLICIES = ("fixed", "adaptive")


def _opt_pct(value: Any) -> Any:
    """A 0-100 percentage, or None when unset/blank (meaning 'use the config default')."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return None


def _opt_cost_hard_stop_pct(value: Any) -> Any:
    """Like ``_opt_pct`` but FAIL-SAFE for the one percentage whose 0 is the
    maximally-permissive setting (0 = NO in-task cost stop). A malformed value
    must NOT silently collapse to 0 and disable the safety stop: a negative
    number, a non-numeric, or a ``0 < v < 1`` fraction (a likely fraction-vs-
    percent mix-up, e.g. 0.5 meaning "half") maps to None — the historical 50%
    default — not to 0. An explicit 0 / 0.0 / "0" is honored verbatim."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num == 0:
        return 0
    if num < 1:
        return None  # negative, or a 0<v<1 fraction — do not silently disable the stop
    return max(0, min(100, int(num)))


def normalize_budget_profile(value: Any) -> Dict[str, Any]:
    """The typed improvement-pacing block (v6.54.4) — how the acceptance-review
    improvement loop spends the task's remaining time budget. Lives INSIDE the
    task contract (no new top-level gateway field); subagents inherit via the
    parent-contract spread. Absent input -> None fields, meaning the config
    defaults apply — which reproduce today's behavior exactly (one bounded
    improvement pass, finalization reserve = the grace window).

    ``improvement_policy``: fixed (default; the configured/max pass cap decides) |
    adaptive (passes stop early when the remaining window can no longer fit a
    review comfortably).

    ``cost_hard_stop_pct`` (v6.56.0, additive): the in-task cost hard-stop as a
    percentage of the budget remaining at task start. None -> the historical
    default (50: the global component of the stop is half the remaining
    budget). 0 -> NO in-task cost stop at all — the deadline/rounds axes and the
    global between-task budget gate remain the only bounds, and cost milestones
    become informational against the start snapshot. The ceiling is resolved in
    ``task_pacing.resolve_cost_ceiling`` (typed; 0 maps to the ``disabled``
    state, never a $0 ceiling; a per-task root cap contributes a second
    min-component). A MALFORMED value (negative / non-numeric / a ``0<v<1``
    fraction) maps to None (the 50% default), NOT to 0 — it must not silently
    disable the stop (see ``_opt_cost_hard_stop_pct``).
    """
    v = value if isinstance(value, Mapping) else {}
    policy = str(v.get("improvement_policy") or "").strip().lower()
    return {
        "improvement_policy": policy if policy in VALID_IMPROVEMENT_POLICIES else "fixed",
        "max_improvement_passes": _opt_nonneg_int(v.get("max_improvement_passes")),
        "reserve_finalization_pct": _opt_pct(v.get("reserve_finalization_pct")),
        "cost_hard_stop_pct": _opt_cost_hard_stop_pct(v.get("cost_hard_stop_pct")),
    }


def _claim_text(value: Any) -> str:
    """Strip edges while preserving the complete claim and internal bytes."""
    return str(value or "").strip()


_ANSWER_PROTOCOLS = ("", "final_answer_line")


def normalize_answer_protocol(value: Any) -> str:
    """v6.60.0 — typed answer-protocol selector (owner quiz 16b, option C+B).

    ``"final_answer_line"``: the caller (a benchmark adapter, an exact-match
    consumer) declares that this task's deliverable is a machine-extractable
    ``FINAL ANSWER: <answer>`` line — the host injects the protocol instruction
    into the TASK context, and the marker nudges/pacing phrases activate.
    ``""`` (default): no marker protocol — ordinary chat/self tasks never see
    'FINAL ANSWER' instructions (the owner's aesthetic ask), while the LATCH and
    EXTRACTOR stay unconditional (harmless when no marker is ever produced, and
    they still capture a spontaneous one). Unknown values normalize to ""
    (fail-open to the no-protocol default, never to an instruction)."""
    text = str(value or "").strip().lower()
    return text if text in _ANSWER_PROTOCOLS else ""


def answer_protocol_active(ctx: Any) -> bool:
    """True when the running ctx's task contract declares
    ``answer_protocol="final_answer_line"``. The ONE gate every marker surface
    (context instruction, loop nudges, pacing phrases, UI chip semantics) reads,
    so the protocol can never half-apply. Accepts a ToolContext-like object
    (reads ``task_contract`` / ``task_metadata``) or a bare contract dict."""
    if isinstance(ctx, Mapping):
        return str(ctx.get("answer_protocol") or "").strip() == "final_answer_line"
    for source in (getattr(ctx, "task_contract", None), getattr(ctx, "task_metadata", None)):
        if isinstance(source, Mapping):
            contract = source.get("task_contract") if isinstance(source.get("task_contract"), Mapping) else source
            if str((contract or {}).get("answer_protocol") or "").strip() == "final_answer_line":
                return True
    return False


def normalize_acceptance_claims(value: Any) -> list[Dict[str, str]]:
    """Normalize LLM-readable acceptance claims.

    These are advisory task-success claims, not a deterministic oracle.  The
    fields deliberately stay general (claim/surface/support/priority) so normal
    user tasks and benchmarks share one vocabulary.
    """
    items = value if isinstance(value, list) else []
    out: list[Dict[str, str]] = []
    seen: set[str] = set()
    for idx, item in enumerate(items, start=1):
        if isinstance(item, Mapping):
            claim = _claim_text(item.get("claim"))
            surface = _claim_text(item.get("surface"))
            support = _claim_text(item.get("support"))
            priority = str(item.get("priority") or "must").strip().lower() or "must"
            raw_id = str(item.get("id") or item.get("criterion_id") or f"claim_{idx}").strip()
        else:
            claim = _claim_text(item)
            surface = ""
            support = ""
            priority = "must"
            raw_id = f"claim_{idx}"
        if not claim:
            continue
        criterion_id = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in raw_id)[:80]
        if not criterion_id:
            criterion_id = f"claim_{idx}"
        base = criterion_id
        suffix = 2
        while criterion_id in seen:
            criterion_id = f"{base}_{suffix}"
            suffix += 1
        seen.add(criterion_id)
        if priority not in {"must", "should", "nice_to_have"}:
            priority = "must"
        out.append({
            "id": criterion_id,
            "claim": claim,
            "surface": surface,
            "support": support,
            "priority": priority,
        })
    return out


def effective_acceptance_claims(
    task: Mapping[str, Any] | None,
    closed_plan_wave: Mapping[str, Any] | None = None,
) -> tuple[list[Dict[str, str]], str]:
    """The claims that bind a task, with provenance — the ONE seam the
    acceptance-evidence builder and the child-contract builder both read (W2).

    Ingress-contract claims win (adapter/gateway/parent-authored — already in the
    built contract); the CLOSED plan wave's frozen claims apply ONLY when ingress
    is empty. PURE: no I/O and no contract mutation — the running task contract is
    never rebuilt mid-task; plan-frozen claims live in ``plan_review_state``
    (``task_results.closed_plan_review_wave``) and are resolved at READ time.
    Returns ``(claims, source)`` with source ``ingress_contract`` |
    ``plan_review`` | ``""`` (no claims anywhere)."""
    task = task if isinstance(task, Mapping) else {}
    contract = (
        task.get("task_contract")
        if isinstance(task.get("task_contract"), Mapping)
        else task
    )
    ingress = normalize_acceptance_claims(contract.get("acceptance_claims"))
    if ingress:
        return ingress, "ingress_contract"
    wave = closed_plan_wave if isinstance(closed_plan_wave, Mapping) else {}
    # v2 waves freeze the whole reviewed SPEC (claims inside it); a v1 wave carried
    # a bare ``acceptance_claims`` list — read as the legacy fallback.
    spec = wave.get("spec") if isinstance(wave.get("spec"), Mapping) else {}
    plan_claims = normalize_acceptance_claims(
        spec.get("acceptance_claims") if spec else wave.get("acceptance_claims")
    )
    if plan_claims:
        return plan_claims, "plan_review"
    return [], ""


def normalize_resource_policy(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, Any] = {}
    protected = value.get("protected_artifacts")
    if isinstance(protected, list):
        records = []
        for item in protected:
            if not isinstance(item, Mapping):
                continue
            paths = item.get("paths")
            if isinstance(paths, (str, bytes)):
                normalized_paths = [str(paths)]
            elif isinstance(paths, list):
                normalized_paths = [str(path).strip() for path in paths if str(path).strip()]
            else:
                normalized_paths = []
            if not normalized_paths:
                continue
            record: Dict[str, Any] = {
                "id": str(item.get("id") or "").strip(),
                "role": str(item.get("role") or "black_box_reference").strip() or "black_box_reference",
                "paths": normalized_paths,
            }
            for key in ("allow", "deny"):
                raw = item.get(key)
                if isinstance(raw, (str, bytes)):
                    values = [str(raw).strip()]
                elif isinstance(raw, list):
                    values = [str(entry).strip() for entry in raw if str(entry).strip()]
                else:
                    values = []
                if values:
                    record[key] = values
            records.append(record)
        if records:
            out["protected_artifacts"] = records
    for key, raw in value.items():
        if key == "protected_artifacts":
            continue
        if raw is not None:
            out[str(key)] = raw
    return out


def build_task_contract(task: Mapping[str, Any] | None) -> Dict[str, Any]:
    task = task or {}
    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    existing = task.get("task_contract") if isinstance(task.get("task_contract"), Mapping) else {}
    existing_meta = metadata.get("task_contract") if isinstance(metadata.get("task_contract"), Mapping) else {}
    merged = {**existing_meta, **existing}

    allowed_resources = normalize_allowed_resources(
        merged.get("allowed_resources")
        or metadata.get("allowed_resources")
        or task.get("allowed_resources")
        or {}
    )
    resource_policy = normalize_resource_policy(
        merged.get("resource_policy")
        or metadata.get("resource_policy")
        or task.get("resource_policy")
        or {}
    )
    objective = str(
        merged.get("objective")
        or task.get("objective")
        or task.get("description")
        or task.get("text")
        or ""
    ).strip()
    expected_output = str(
        merged.get("expected_output")
        or task.get("expected_output")
        or metadata.get("expected_output")
        or ""
    ).strip()
    constraints = str(
        merged.get("constraints")
        or task.get("constraints")
        or metadata.get("constraints")
        or ""
    ).strip()
    context_value = (
        merged.get("context")
        if merged.get("context") is not None
        else (task.get("context") if task.get("context") is not None else metadata.get("context") or "")
    )
    context = "" if context_value is None else str(context_value)
    deadline_at = str(
        merged.get("deadline_at")
        or task.get("deadline_at")
        or metadata.get("deadline_at")
        or ""
    ).strip()
    disabled_tools = normalize_disabled_tools(
        merged.get("disabled_tools")
        if merged.get("disabled_tools") is not None
        else (task.get("disabled_tools") or metadata.get("disabled_tools"))
    )
    normalized_workspace = merged.get("workspace") if isinstance(merged.get("workspace"), Mapping) else {}
    workspace_root = str(
        merged.get("workspace_root")
        or normalized_workspace.get("root")
        or task.get("workspace_root")
        or metadata.get("workspace_root")
        or ""
    ).strip()
    workspace_mode = str(
        merged.get("workspace_mode")
        or normalized_workspace.get("mode")
        or task.get("workspace_mode")
        or metadata.get("workspace_mode")
        or ""
    ).strip()
    task_type = str(merged.get("task_type") or task.get("type") or "task").strip() or "task"
    capability_ceiling = None
    if merged.get("capability_ceiling") is not None:
        from ouroboros.presence_authority import (
            presence_ceiling_from_payload,
            presence_ceiling_payload,
        )

        capability_ceiling = presence_ceiling_payload(
            presence_ceiling_from_payload(merged.get("capability_ceiling"))
        )

    acceptance_claims = normalize_acceptance_claims(
        merged.get("acceptance_claims")
        if merged.get("acceptance_claims") is not None
        else (merged.get("success_criteria") or task.get("acceptance_claims") or metadata.get("acceptance_claims"))
    )

    normalized_lineage = merged.get("lineage") if isinstance(merged.get("lineage"), Mapping) else {}
    contract = {
        "schema_version": 1,
        "status": str(merged.get("status") or "draft"),
        "source": str(merged.get("source") or "host_draft"),
        "task_type": task_type,
        "objective": objective,
        "expected_output": expected_output,
        "constraints": constraints,
        # Exact caller/parent authority.  This used to live only on the task row,
        # outside the contract rendered to the model and inherited by children.
        "context": context,
        # (W2) success_criteria is an INPUT ALIAS: it already feeds
        # normalize_acceptance_claims above when no claims were given, so once
        # acceptance_claims is populated the raw list is NOT double-persisted —
        # one concept, one carrier. Historical records keep their stored shape
        # untouched (no normalizer, v6.78 precedent); readers tolerate both
        # shapes (the eligibility probe checks both keys).
        "success_criteria": []
        if acceptance_claims
        else (
            list(merged.get("success_criteria") or [])
            if isinstance(merged.get("success_criteria"), list)
            else []
        ),
        "acceptance_claims": acceptance_claims,
        "allowed_resources": allowed_resources,
        "resource_policy": resource_policy,
        "disabled_tools": disabled_tools,
        "deadline_at": deadline_at,
        "context_requires_self_body_docs": normalize_bool(
            merged.get("context_requires_self_body_docs")
            if "context_requires_self_body_docs" in merged
            else task.get("context_requires_self_body_docs", metadata.get("context_requires_self_body_docs"))
        ),
        "attachment_manifest": normalize_attachment_manifest(
            merged.get("attachment_manifest")
            if merged.get("attachment_manifest") is not None
            else task.get("attachments")
        ),
        "workspace": {
            "root": workspace_root,
            "mode": workspace_mode,
        },
        "lineage": {
            "parent_task_id": str(task.get("parent_task_id") or metadata.get("parent_task_id") or normalized_lineage.get("parent_task_id") or ""),
            "root_task_id": str(task.get("root_task_id") or metadata.get("root_task_id") or normalized_lineage.get("root_task_id") or task.get("id") or ""),
            "session_id": str(task.get("session_id") or metadata.get("session_id") or normalized_lineage.get("session_id") or ""),
            "delegation_role": str(task.get("delegation_role") or metadata.get("delegation_role") or normalized_lineage.get("delegation_role") or "root"),
        },
        "delegation_budget": normalize_delegation_budget(
            merged.get("delegation_budget")
            if merged.get("delegation_budget") is not None
            else (task.get("delegation_budget") or metadata.get("delegation_budget"))
        ),
        "budget_profile": normalize_budget_profile(
            merged.get("budget_profile")
            if merged.get("budget_profile") is not None
            else (task.get("budget_profile") or metadata.get("budget_profile"))
        ),
        # v6.60.0 additive field: "" (no marker protocol, the default) |
        # "final_answer_line" (adapter-declared machine-extractable answer line).
        # Subagents inherit through the same metadata/task propagation as the rest
        # of the contract fields.
        "answer_protocol": normalize_answer_protocol(
            merged.get("answer_protocol")
            if merged.get("answer_protocol") is not None
            else (task.get("answer_protocol") or metadata.get("answer_protocol"))
        ),
    }
    predecessor_authority = (
        merged.get("predecessor_authority")
        if isinstance(merged.get("predecessor_authority"), Mapping)
        else task.get("predecessor_authority")
        if isinstance(task.get("predecessor_authority"), Mapping)
        else metadata.get("predecessor_authority")
    )
    if isinstance(predecessor_authority, Mapping) and predecessor_authority:
        # The predecessor is additive authority, not prose to merge into the new
        # objective.  Preserve its materialized envelope so the ordinary parent
        # contract spread carries it through direct starts and nested work orders.
        # A legacy FAT body (pre-envelope: full result + nested contract chains)
        # is collapsed here to its bounded reference shape - the in-flight
        # migration point: rebuilding any contract sheds the recursion while
        # the durable task_results (the SSOT bodies) stay untouched.
        contract["predecessor_authority"] = _bounded_predecessor_authority(
            dict(predecessor_authority)
        )
    for key in ("notes", "review_notes"):
        if merged.get(key):
            contract[key] = merged.get(key)
    if capability_ceiling is not None:
        contract["capability_ceiling"] = capability_ceiling
    return contract


def _serialized_chars(value: Any) -> Tuple[str, int]:
    """A value's wire text and size - the SERIALIZED form is what rides.

    Strings are measured serialized too (control characters escape to many
    wire bytes: 15K of NULs serializes to 90K). A body too deep to serialize
    (a recursive legacy chain) is oversized by definition, never a crash.
    """
    if isinstance(value, str):
        try:
            return value, len(json.dumps(value, ensure_ascii=False)) - 2
        except (TypeError, ValueError, RecursionError):
            return value, len(value)
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except RecursionError:
        return "<authority body too deep to serialize>", 2 ** 31
    except (TypeError, ValueError):
        text = repr(value)
    return text, len(text)


def bounded_continuation_envelope(
    mapping: Dict[str, Any], *, digest_semantics: str,
    source_ref: Optional[Dict[str, Any]] = None, salvage: bool = False,
    extra: Optional[Dict[str, Any]] = None, reserve_source: bool = False,
) -> Dict[str, Any]:
    """The ONE bounded-envelope producer (startup binding and legacy collapse).

    Every compact terminal fact inherits by copy - fields this reader predates
    included - so authority cannot silently vanish. Only the growth carriers
    are bounded structurally: the nested ``task_contract`` inherits its
    operative core minus ``predecessor_authority`` (the recursion), and any
    field whose body outgrows the tool-result budget rides as a typed preview,
    measured on its SERIALIZED form - lists, dicts and escape-heavy strings
    all count, previews shrink until they fit the wire. Core contract strings
    stay strings under disclosed truncation. Row fields colliding with
    envelope metadata names ride BOUNDED under ``shadowed_authority_fields``
    instead of being clobbered; a malformed non-mapping ``task_contract``
    rides as a bounded plain field rather than vanishing. The digest names
    when it was observed; the durable task_results body stays the SSOT.
    """
    from ouroboros.tool_capabilities import DEFAULT_TOOL_RESULT_LIMIT
    from ouroboros.utils import truncate_within_limit

    limit = DEFAULT_TOOL_RESULT_LIMIT

    def _fit(raw: str) -> str:
        # Every cut is taken from the ORIGINAL text, so the omission marker
        # always discloses the true original length; the allowance shrinks
        # proportionally to the observed escape factor (a 1-char overshoot
        # costs 1 char, a 6x-escaping body converges near limit/6).
        allowed = limit
        preview = truncate_within_limit(raw, limit=allowed)
        while allowed > 100:
            try:
                over = len(json.dumps(preview, ensure_ascii=False)) - 2 - limit
            except (TypeError, ValueError, RecursionError):
                break
            if over <= 0:
                break
            allowed = max(100, min(allowed - 1, allowed * limit // (limit + over)))
            preview = truncate_within_limit(raw, limit=allowed)
        return preview

    def _bounded(key: str, value: Any, *, field: str, force_pointer: bool = False) -> Any:
        text, chars = _serialized_chars(value)
        if not force_pointer and chars <= limit:
            return copy.deepcopy(value)
        entry: Dict[str, Any] = {
            "kind": "unreviewed_host_salvage" if force_pointer else "bounded_field_preview",
            "preview": _fit(text),
            "full_chars": chars,
        }
        if source_ref:
            entry["source_ref"] = {**copy.deepcopy(source_ref), "field": field}
        return entry

    digest_note = digest_semantics
    try:
        serialized = json.dumps(mapping, ensure_ascii=False, sort_keys=True, default=str)
    except RecursionError:
        # No identity can be computed over a body too deep to serialize -
        # disclose that instead of presenting a placeholder digest as exact.
        serialized = ""
        digest_note = f"{digest_semantics}_unserializable"
    except (TypeError, ValueError):
        serialized = repr(mapping)
    raw_contract = mapping.get("task_contract")
    contract = raw_contract if isinstance(raw_contract, Mapping) else {}
    reserved = {"kind", "authority_sha256", "authority_chars", "digest_semantics",
                "previous_task_id", "collapsed_from", "shadowed_authority_fields"}
    if reserve_source:
        # The startup binding writes the pull pointer under ``source`` LAST; a
        # projected row field with that name must shadow, not clobber or vanish.
        reserved.add("source")
    envelope: Dict[str, Any] = {}
    shadowed: Dict[str, Any] = {}
    for key, value in mapping.items():
        if key == "task_contract" and isinstance(raw_contract, Mapping):
            continue
        if key in reserved:
            shadowed[key] = _bounded(key, value, field=f"authority.{key}")
            continue
        envelope[key] = _bounded(
            key, value, field=f"authority.{key}",
            force_pointer=salvage and key == "result",
        )
    core: Dict[str, Any] = {}
    for key, value in contract.items():
        if key == "predecessor_authority":
            continue
        if isinstance(value, str):
            _text, chars = _serialized_chars(value)
            core[key] = _fit(value) if chars > limit else value
            continue
        core[key] = _bounded(key, value, field=f"authority.task_contract.{key}")
    if core:
        envelope["task_contract"] = core
    if shadowed:
        envelope["shadowed_authority_fields"] = shadowed
    envelope.update({
        "kind": "bounded_continuation_envelope",
        "authority_sha256": sha256(serialized.encode("utf-8")).hexdigest() if serialized else "",
        "authority_chars": len(serialized),
        "digest_semantics": digest_note,
        **(extra or {}),
    })
    return envelope

def _bounded_predecessor_authority(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse a legacy full-body predecessor on a growth carrier.

    A body free of both carriers - no nested recursion, every field within
    the tool-result budget on its serialized form - passes through
    byte-identical: exact strings are authority. Otherwise the shared
    envelope producer rewrites it once, and the minted envelope passes
    rebuilds untouched.
    """
    if str(mapping.get("kind") or "") == "bounded_continuation_envelope":
        return copy.deepcopy(mapping)
    from ouroboros.tool_capabilities import DEFAULT_TOOL_RESULT_LIMIT

    raw_contract = mapping.get("task_contract")
    contract = raw_contract if isinstance(raw_contract, Mapping) else {}
    oversized = (
        any(_serialized_chars(v)[1] > DEFAULT_TOOL_RESULT_LIMIT
            for k, v in mapping.items()
            if k != "task_contract" or not isinstance(raw_contract, Mapping))
        or any(k != "predecessor_authority"
               and _serialized_chars(v)[1] > DEFAULT_TOOL_RESULT_LIMIT
               for k, v in contract.items())
    )
    if "predecessor_authority" not in contract and not oversized:
        return copy.deepcopy(mapping)
    source = mapping.get("source") if isinstance(mapping.get("source"), Mapping) else {}
    # The chain cursor names the hop BEFORE this body's subject - the same rule
    # the startup binding mints - never the subject's own id (a self-loop).
    nested = contract.get("predecessor_authority") if isinstance(contract.get("predecessor_authority"), Mapping) else {}
    nested_source = nested.get("source") if isinstance(nested.get("source"), Mapping) else {}
    return bounded_continuation_envelope(
        mapping,
        digest_semantics="observed_at_collapse",
        source_ref=dict(source) or None,
        extra={
            "collapsed_from": "legacy_full_body",
            "previous_task_id": str(
                nested.get("task_id") or nested.get("previous_task_id")
                or nested_source.get("task_id") or ""
            ),
        },
    )

def attach_task_contract(task: Dict[str, Any]) -> Dict[str, Any]:
    contract = build_task_contract(task)
    task["task_contract"] = contract
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    metadata["task_contract"] = contract
    task["metadata"] = metadata
    return task


__all__ = ["answer_protocol_active", "attach_task_contract", "build_task_contract", "effective_acceptance_claims", "normalize_acceptance_claims", "normalize_allowed_resources", "normalize_answer_protocol", "normalize_attachment_manifest", "normalize_bool", "normalize_budget_profile", "normalize_delegation_budget", "normalize_depth_provenance", "normalize_disabled_tools", "normalize_resource_policy"]
