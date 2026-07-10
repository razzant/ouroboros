"""Task contract normalization.

The contract is a durable, LLM-readable description of what this task is trying
to accomplish.  It is not a deterministic success oracle: code records the
declared goal, constraints, resources, and artifacts; LLM review/evaluation
interprets whether the objective was met.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


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


def _opt_nonneg_int(value: Any) -> Any:
    """A non-negative int, or None when unset/blank (meaning 'use the config cap')."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _bounded_intent_note(value: Any, limit: int = 500) -> str:
    """Bound a delegation intent_note to one line, with a VISIBLE omission marker
    rather than a silent clip of the cognitive hint (BIBLE P1)."""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f" ⚠️(+{len(text) - limit} chars omitted)"


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
    return {
        "may_delegate": normalize_bool(v.get("may_delegate", True)),
        "may_mutate": normalize_bool(v.get("may_mutate", False)),
        "may_fan_out": normalize_bool(v.get("may_fan_out", True)),
        "depth_remaining": _opt_nonneg_int(v.get("depth_remaining")),
        "max_children": _opt_nonneg_int(v.get("max_children")),
        "intent_note": _bounded_intent_note(v.get("intent_note")),
    }


VALID_IMPROVEMENT_POLICIES = ("fixed", "adaptive", "until_deadline")


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
    review comfortably) | until_deadline (passes bounded ONLY by the time gate).

    ``cost_hard_stop_pct`` (v6.56.0, additive): the in-task cost hard-stop as a
    percentage of the budget remaining at task start. None -> the historical
    default (50: force-finalize once the task has spent half the remaining
    budget). 0 -> NO in-task cost stop at all — the deadline/rounds axes and the
    global between-task budget gate remain the only bounds, and cost milestones
    become informational against the start snapshot. The ceiling is resolved in
    ``task_pacing.resolve_cost_ceiling_usd`` (0 maps to no ceiling, never a $0
    ceiling). A MALFORMED value (negative / non-numeric / a ``0<v<1`` fraction)
    maps to None (the 50% default), NOT to 0 — it must not silently disable the
    stop (see ``_opt_cost_hard_stop_pct``).
    """
    v = value if isinstance(value, Mapping) else {}
    policy = str(v.get("improvement_policy") or "").strip().lower()
    return {
        "improvement_policy": policy if policy in VALID_IMPROVEMENT_POLICIES else "fixed",
        "max_improvement_passes": _opt_nonneg_int(v.get("max_improvement_passes")),
        "reserve_finalization_pct": _opt_pct(v.get("reserve_finalization_pct")),
        "stall_rounds_threshold": _opt_nonneg_int(v.get("stall_rounds_threshold")),
        "cost_hard_stop_pct": _opt_cost_hard_stop_pct(v.get("cost_hard_stop_pct")),
    }


def _bounded_claim_text(value: Any, limit: int = 600) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f" ⚠️(+{len(text) - limit} chars omitted)"


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
            claim = _bounded_claim_text(item.get("claim"))
            surface = _bounded_claim_text(item.get("surface"), limit=300)
            support = _bounded_claim_text(item.get("support"), limit=500)
            priority = str(item.get("priority") or "must").strip().lower() or "must"
            raw_id = str(item.get("id") or item.get("criterion_id") or f"claim_{idx}").strip()
        else:
            claim = _bounded_claim_text(item)
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
    workspace_root = str(
        merged.get("workspace_root")
        or task.get("workspace_root")
        or metadata.get("workspace_root")
        or ""
    ).strip()
    workspace_mode = str(
        merged.get("workspace_mode")
        or task.get("workspace_mode")
        or metadata.get("workspace_mode")
        or ""
    ).strip()
    task_type = str(merged.get("task_type") or task.get("type") or "task").strip() or "task"

    acceptance_claims = normalize_acceptance_claims(
        merged.get("acceptance_claims")
        if merged.get("acceptance_claims") is not None
        else (merged.get("success_criteria") or task.get("acceptance_claims") or metadata.get("acceptance_claims"))
    )

    contract = {
        "schema_version": 1,
        "status": str(merged.get("status") or "draft"),
        "source": str(merged.get("source") or "host_draft"),
        "task_type": task_type,
        "objective": objective,
        "expected_output": expected_output,
        "constraints": constraints,
        "success_criteria": list(merged.get("success_criteria") or [])
        if isinstance(merged.get("success_criteria"), list)
        else [],
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
        "workspace": {
            "root": workspace_root,
            "mode": workspace_mode,
        },
        "lineage": {
            "parent_task_id": str(task.get("parent_task_id") or metadata.get("parent_task_id") or ""),
            "root_task_id": str(task.get("root_task_id") or metadata.get("root_task_id") or task.get("id") or ""),
            "session_id": str(task.get("session_id") or metadata.get("session_id") or ""),
            "delegation_role": str(task.get("delegation_role") or metadata.get("delegation_role") or "root"),
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
    for key in ("notes", "review_notes"):
        if merged.get(key):
            contract[key] = merged.get(key)
    return contract


def attach_task_contract(task: Dict[str, Any]) -> Dict[str, Any]:
    contract = build_task_contract(task)
    task["task_contract"] = contract
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    metadata["task_contract"] = contract
    task["metadata"] = metadata
    return task


__all__ = ["answer_protocol_active", "attach_task_contract", "build_task_contract", "normalize_acceptance_claims", "normalize_allowed_resources", "normalize_answer_protocol", "normalize_bool", "normalize_budget_profile", "normalize_delegation_budget", "normalize_disabled_tools", "normalize_resource_policy"]
