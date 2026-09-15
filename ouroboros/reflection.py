"""Generate post-task process-memory reflections for non-trivial/error runs."""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros._outcome_tool_errors import _OK_TOOL_STATUSES
from ouroboros.utils import utc_now_iso, append_jsonl, write_text_atomic


def _truncate_with_notice(text: Any, limit: int) -> str:
    """Canonical disclosed truncation for PROSE fields (delegates to the utils
    SSOT, which also refuses cuts cheaper than the omission marker itself).
    Tiny limits (< 100 chars: identifier-shaped fields like kind/priority/topic,
    or the last chars of an exhausted snippet budget) keep a hard slice: an
    omission marker longer than the remaining budget would be worse damage
    than the cut it discloses. Model-bound surface, so disclosed truncation is
    not owed here (v6.70.0 invariant)."""
    raw = str(text or "")
    if limit < 100:
        return raw[:limit]
    from ouroboros.utils import truncate_review_artifact

    return truncate_review_artifact(raw, limit=limit)

log = logging.getLogger(__name__)

# Reflection triggers for non-trivial clean tasks.
NONTRIVIAL_ROUNDS_THRESHOLD: int = 15
NONTRIVIAL_COST_THRESHOLD: float = 5.0

REFLECTIONS_FILENAME = "task_reflections.jsonl"


def _trace_call_errored(tc: Dict[str, Any]) -> bool:
    """One reading of "this call went wrong" for every reflection trigger.

    Reads the ok-status SSOT rather than a fourth private spelling of it. The
    ``("", "ok")`` tuple this replaces counted two statuses whose spec says ok as
    errors: ``untyped`` — the status a dynamic provider body carries when nothing
    typed it, which a SUCCESSFUL extension call now has — and ``ok_autocorrected``,
    a shell command whose regex the host repaired. Both handed a clean run the
    error-reflection prompt with nothing to reflect on. Every other status keeps
    its meaning here exactly.
    """
    return bool(
        tc.get("is_error")
        or str(tc.get("status") or "").strip().lower() not in _OK_TOOL_STATUSES
    )


# The typed code for a success that still carries a failure: the ordinary
# self-modification commit PRESERVES a revision whose post-commit tests failed
# and reports it as an ok result with a warning appended (tools/git.py publishes
# the fact in the result meta, loop_tool_execution stamps it on the trace row).
POST_COMMIT_TESTS_FAILED = "POST_COMMIT_TESTS_FAILED"


def _trace_call_reported_failure(tc: Dict[str, Any]) -> bool:
    """Did this call report something that went wrong, errored or not?

    The commit above must NOT become an error - it succeeded, and every consumer
    of that distinction is right about it - but the failing tests are exactly the
    class the Pattern Register exists for, so the reflection triggers read the
    producer's typed fact beside the ok status instead of hunting for a word in
    the body.
    """
    return _trace_call_errored(tc) or str(tc.get("post_commit_tests") or "") == "failed"


_REFLECTION_PROMPT_ERROR = """\
You are performing a post-task experience review for Ouroboros, a self-modifying AI agent.
The task had errors or blocking events. Write a concise 150-250 word reflection covering:

1. What was the goal?
2. What specific errors/blocks occurred?
3. What was the root cause (if identifiable)?
4. What should be done differently next time?

Be concrete — cite specific file names, tool names, error messages. No platitudes.
If structured review evidence exists, incorporate the critical/advisory findings and
open obligations into the root-cause analysis. Mention them individually with their
severity and item/tag identity rather than collapsing them into a generic "review failed".\
"""

_REFLECTION_PROMPT_NONTRIVIAL = """\
You are performing a post-task experience review for Ouroboros, a self-modifying AI agent.
The task was non-trivial (high round count or high cost) but completed without hard errors.
Write a concise 150-250 word reflection covering:

1. What was the goal?
2. What took the most rounds/cost? Where was the friction?
3. Were there weak assumptions, unnecessary detours, or suboptimal tool choices?
4. What would make a similar task cheaper or faster next time?

Be concrete — cite specific file names, tool names, decision points. No platitudes.\
"""

# Shared tail with {format} fields.
_REFLECTION_PROMPT_TAIL = """

Then, if this task produced durable, reusable learning worth persisting now — about
my own work, and about the people I worked with (what mattered to them, how we
worked together, an interpretation worth testing) — append a line:
MEMORY_ACTIONS_JSON: [...]
A JSON array of 0-3 objects. Each object must have:
- type: one of "scratchpad_append", "knowledge_write", "identity_update_candidate"
- content: concise, concrete text to persist
Optional field:
- topic: REQUIRED only for knowledge_write (shelf-relative path, e.g. "review_process")
- scope: optional global or project:<exact project id>; omission keeps this task's shelf
Rules for memory actions:
- scratchpad_append: a durable working-memory note useful for near-future tasks.
- knowledge_write: complete revised Markdown understanding under `topic`. Use
  knowledge_read to read the whole CURRENT note before replacing it. Preserve
  evidence, uncertainty and useful links; new topics need no prior read. A repeated
  interpretation is not new independent evidence. No blind append of fragments.
- identity_update_candidate: a PROPOSED identity refinement; it is only recorded as a
  review candidate in the scratchpad, never auto-applied to identity.md (avoid drift).
- Persist only genuinely durable, reusable learning, not task-specific trivia.
- If nothing deserves persisting, output MEMORY_ACTIONS_JSON: []

Then, if there is at least one concrete deferred improvement worth tracking, append a final line:
BACKLOG_CANDIDATES_JSON: [...]
Use a JSON array of 0-3 objects. Each object must have:
- summary
- category
- source
- evidence
Optional fields:
- context
- proposed_next_step
- task_id
- requires_plan_review
- priority (high | med | low — how valuable/urgent this is; default med)
- kind (bug | improvement | capability_idea — use capability_idea for a forward-looking NEW ability worth building, not just a fix; default improvement)
Rules for candidates:
- Only include concrete, evidence-backed follow-ups that are OUT OF SCOPE for the current task.
- Prefer recurring process/tool/review friction over one-off noise.
- Capability ideas are welcome (kind=capability_idea), not only bugs — but keep them concrete and evidence-backed.
- Core/code improvements should become explicit Evolution Campaign candidates
  or backlog items; do not assume they were already executed.
- If nothing deserves backlog tracking, output BACKLOG_CANDIDATES_JSON: []
- Tool arguments in logs may show `<TRUNCATED:key:Nch:sha=...>` placeholders.
  That is logging metadata, not the value passed to the tool.

## Task goal

{goal}

## Execution trace

{trace_summary}

## Tool usage profile

{tool_usage}

(If a capability I own was under-used — e.g. shell grep/cat/sed as a reader/search
instead of search_code/read_file/query_code, or a high search_code:query_code ratio
where structure would have been sharper — note it; a faculty owned but unused is one
I am losing. A concrete forward-looking fix can be a kind=capability_idea backlog item.)

## Error details

{error_details}

## Structured review evidence

{review_evidence}

## Related child/subtask evidence

{child_evidence}

{usage_snapshot}{sealed_final}Write the reflection now. Plain text, no markdown headers except the exact final
MEMORY_ACTIONS_JSON and BACKLOG_CANDIDATES_JSON lines.
"""

_REFLECTION_PROMPT_ERROR_FULL = _REFLECTION_PROMPT_ERROR + _REFLECTION_PROMPT_TAIL
_REFLECTION_PROMPT_NONTRIVIAL_FULL = _REFLECTION_PROMPT_NONTRIVIAL + _REFLECTION_PROMPT_TAIL


def should_generate_reflection(
    llm_trace: Dict[str, Any],
    *,
    task: Optional[Dict[str, Any]] = None,
    rounds: int = 0,
    cost_usd: Optional[float] = None,
    child_failure_classes: Optional[List[str]] = None,
) -> bool:
    """Return True for tool errors/blocking markers or costly many-round tasks.

    ``child_failure_classes`` are the typed failure classes of this root's own
    children, from the caller's single evidence walk. Children do not reflect,
    so a short clean root that delegated the work and got a FAILED child back is
    the only place that failure can be learned from: without this the register's
    own admission rule could never fire for the shape it was written for.
    """
    task = task or {}
    if child_failure_classes:
        return True
    if str(task.get("type") or "") in {"evolution", "deep_self_review"}:
        return True
    if str(task.get("workspace_root") or "").strip() or str(task.get("workspace_mode") or "").strip():
        return True
    if rounds >= NONTRIVIAL_ROUNDS_THRESHOLD:
        return True
    if cost_usd is not None and cost_usd >= NONTRIVIAL_COST_THRESHOLD:
        return True

    for tc in (llm_trace.get("tool_calls") or []):
        if isinstance(tc, dict) and _trace_call_reported_failure(tc):
            return True

    return False


def _collect_error_details(llm_trace: Dict[str, Any], cap: int = 3000) -> str:
    """Extract error tool results from the trace, up to *cap* chars."""
    parts: List[str] = []
    total = 0
    tool_calls = llm_trace.get("tool_calls") or []

    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        if not _trace_call_errored(tc):
            continue
        result_str = str(tc.get("result", ""))
        tool_name = tc.get("tool", "unknown")
        facts = []
        status = str(tc.get("status") or "").strip()
        if status:
            facts.append(f"status={status}")
        if tc.get("exit_code") not in (None, ""):
            facts.append(f"exit_code={tc.get('exit_code')}")
        if tc.get("signal"):
            facts.append(f"signal={tc.get('signal')}")
        fact_prefix = f" ({', '.join(facts)})" if facts else ""
        # Redact BEFORE embedding: post evidence-parity the trace carries raw
        # actor-window results (15k-80k+), so an error dump can contain secrets
        # that would otherwise reach the external reflection model (codex v6.71.1).
        from ouroboros.observability import redact_projection

        safe_result = redact_projection(result_str).value
        # Pre-cap each snippet so one oversized error cannot monopolize the whole
        # budget and hide later distinct errors (breadth over depth).
        snippet = _truncate_with_notice(f"[{tool_name}{fact_prefix}]: {safe_result}", 1000)
        if total + len(snippet) > cap:
            remaining = cap - total
            if remaining > 50:
                parts.append(_truncate_with_notice(snippet, remaining))
            break
        parts.append(snippet)
        total += len(snippet)

    return "\n\n".join(parts) if parts else "(no error details captured)"


def _tool_usage_profile(llm_trace: Dict[str, Any]) -> str:
    """Compact tool-call frequency profile + a shell-as-reader/search signal.

    Gives the reflection LLM the DATA to judge capability under-use (e.g. a high
    search_code:query_code ratio, or grep/cat/sed via run_command instead of the
    first-class read_file/search_code/query_code). The LLM decides whether that is
    a problem and may emit a capability_idea backlog item — no keyword gate here."""
    from collections import Counter

    counts: Counter = Counter()
    shell_reader = 0
    for tc in (llm_trace.get("tool_calls") or []):
        if not isinstance(tc, dict):
            continue
        name = str(tc.get("tool") or "").strip()
        if not name:
            continue
        counts[name] += 1
        if name in ("run_command", "run_script"):
            args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
            cmd = str(args.get("cmd") or args.get("command") or "").lower()
            if any(f"{tok} " in cmd or cmd.startswith(tok) for tok in ("grep", "rg", "cat", "sed", "head", "tail", "find", "awk")):
                shell_reader += 1
    if not counts:
        return "(no tool calls recorded)"
    top = ", ".join(f"{name}×{count}" for name, count in counts.most_common(15))
    note = f"\nshell-as-reader/search via run_command/run_script: {shell_reader} call(s)" if shell_reader else ""
    return top + note


def _detect_markers(llm_trace: Dict[str, Any]) -> List[str]:
    """Return the sorted TYPED codes of the calls that went wrong.

    This used to scan every result body for eight hand-listed words (P5: a
    keyword gate standing in for a fact the record already holds). Every call
    carries ``tool_result_code`` beside its status, so the same question is
    answered from the typed record instead: no bounded view to tune, no doc read
    that mentions a marker mid-body classifying a clean task as errored, and no
    typed failure invisible because nobody added its word to the list. A legacy
    row written before the code existed falls back to its recorded status, kept
    verbatim rather than dressed up as a code it never had."""
    found: set = set()
    for tc in (llm_trace.get("tool_calls") or []):
        if not isinstance(tc, dict):
            continue
        if str(tc.get("post_commit_tests") or "") == "failed":
            # An ok commit that preserved a revision with failing tests: its own
            # code says OK and is right, so the failure needs its own name.
            found.add(POST_COMMIT_TESTS_FAILED)
        if not _trace_call_errored(tc):
            continue
        code = str(tc.get("tool_result_code") or "").strip() or str(tc.get("status") or "").strip()
        if code:
            found.add(code)
    return sorted(found)


_ALLOWED_MEMORY_ACTION_TYPES = frozenset({
    "scratchpad_append",
    "knowledge_write",
    "identity_update_candidate",
})


def _extract_trailing_json(text: str, marker: str) -> tuple[str, Optional[list]]:
    """Peel a ``MARKER: [...]`` block out of *text* regardless of its position.

    Removes only the marker and its JSON array (located via a tolerant
    ``raw_decode``), preserving any other marker line so callers can extract
    multiple markers in any order without silently dropping one. Returns
    ``(remaining_text, parsed_list_or_None)``: a present-but-empty payload is
    ``[]``; a malformed payload is ``None`` so the caller can distinguish
    "no items" from "parse failure".
    """
    idx = text.rfind(marker)
    if idx == -1:
        return text, None
    after = text[idx + len(marker):]
    stripped = after.lstrip()
    lead = len(after) - len(stripped)
    if not stripped:
        return text[:idx].rstrip(), []
    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except Exception:
        log.warning("Reflection %s JSON parse failed", marker, exc_info=True)
        return text[:idx].rstrip(), None
    remainder = (text[:idx] + after[lead + end:]).rstrip()
    return remainder, value if isinstance(value, list) else None


def _validate_memory_actions(raw: Any, task_id: str) -> List[Dict[str, Any]]:
    """Keep only well-formed, allowed-type memory actions (max 3)."""
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw[:10]:
        if len(out) >= 3:
            break
        if not isinstance(item, dict):
            continue
        action_type = str(item.get("type") or "").strip()
        if action_type not in _ALLOWED_MEMORY_ACTION_TYPES:
            continue
        content = (str(item.get("content") or "") if action_type == "knowledge_write"
                   else _truncate_with_notice(item.get("content", ""), 1200)).strip()
        if not content:
            continue
        action: Dict[str, Any] = {"type": action_type, "content": content, "task_id": task_id}
        if action_type == "knowledge_write":
            topic = str(item.get("topic") or "").strip()
            if not topic:
                continue
            action["topic"] = topic
            if item.get("scope") is not None:
                action["scope"] = item["scope"]
        out.append(action)
    return out

def generate_reflection(
    task: Dict[str, Any],
    llm_trace: Dict[str, Any],
    trace_summary: str,
    llm_client: Any,
    usage_dict: Dict[str, Any],
    review_evidence: Optional[Dict[str, Any]] = None,
    child_evidence: str = "",
    usage_snapshot_text: str = "",
    sealed_final_text: str = "",
    child_failure_classes: Optional[List[str]] = None,
    knowledge_context: Any = None,
) -> Dict[str, Any]:
    """Call the light LLM and return a JSONL-ready reflection entry."""
    goal = _truncate_with_notice(task.get("text", ""), 200)
    source_ref = None
    memory_operation_errors: List[Dict[str, Any]] = []
    error_details = _collect_error_details(llm_trace)
    markers = _detect_markers(llm_trace)
    error_count = sum(
        1 for tc in (llm_trace.get("tool_calls") or [])
        if isinstance(tc, dict) and _trace_call_errored(tc)
    )
    panels = None
    try:
        from ouroboros.review_substrate import compact_review_projection
        panels = compact_review_projection(llm_trace.get("review_runs") or []).get("panels")
    except Exception:
        log.debug("Acceptance panel projection unavailable for reflection", exc_info=True)
    try:
        from ouroboros.review_evidence import format_review_evidence_for_prompt
        review_evidence_text = format_review_evidence_for_prompt(review_evidence or {}, max_chars=8000, acceptance_panels=panels)
    except Exception:
        review_evidence_text = "(review evidence unavailable)"

    if child_failure_classes and not (error_count or markers):
        error_details = "Child failure classes: " + ", ".join(child_failure_classes)
    if error_count or markers or child_failure_classes:
        prompt_template = _REFLECTION_PROMPT_ERROR_FULL
    else:
        prompt_template = _REFLECTION_PROMPT_NONTRIVIAL_FULL

    prompt = prompt_template.format(
        goal=str(task.get("text") or "(no goal text)"),
        trace_summary=_truncate_with_notice(trace_summary, 2000),
        tool_usage=_tool_usage_profile(llm_trace),
        error_details=error_details,
        review_evidence=review_evidence_text,
        child_evidence=child_evidence or "(none)",
        usage_snapshot=usage_snapshot_text or "",
        sealed_final=sealed_final_text or "",
    )

    try:
        from ouroboros.consolidator import KnowledgeReadContext, KNOWLEDGE_MAINTENANCE_PROMPT, _call_consolidation_llm
        from ouroboros.tools.registry import ToolContext

        if knowledge_context is None:
            from ouroboros.config import DATA_DIR
            root = pathlib.Path(task.get("budget_drive_root") or task.get("drive_root") or DATA_DIR)
            knowledge_context = ToolContext(repo_dir=root, drive_root=root,
                project_id=str(task.get("project_id") or ""),
                task_id=str(task.get("id") or task.get("task_id") or "reflection"))
        knowledge = KnowledgeReadContext(knowledge_context, "task_reflection")
        from ouroboros.consolidator import retain_memory_source
        complete_prompt = KNOWLEDGE_MAINTENANCE_PROMPT + prompt
        source_ref = retain_memory_source(knowledge_context, "task_input_reflection", complete_prompt.encode("utf-8"))
        raw_reflection_text, refl_usage = _call_consolidation_llm(
            llm_client, complete_prompt, "Task reflection", knowledge=knowledge, source_ref=source_ref)
        raw_reflection_text = raw_reflection_text.strip()
        memory_operation_errors = refl_usage.get("_consolidation_errors") or []
        if not raw_reflection_text and memory_operation_errors:
            raw_reflection_text = "(reflection generation failed: " + str(memory_operation_errors[-1].get("message") or "unknown") + ")"
        task_id_str = str(task.get("id", "") or "")

        # Backlog is the last trailing line; peel it first, then memory actions.
        body_after_backlog, raw_candidates = _extract_trailing_json(
            raw_reflection_text, "BACKLOG_CANDIDATES_JSON:"
        )
        reflection_text, raw_memory_actions = _extract_trailing_json(
            body_after_backlog, "MEMORY_ACTIONS_JSON:"
        )
        reflection_text = reflection_text.strip()

        backlog_candidates: List[Dict[str, Any]] = []
        if isinstance(raw_candidates, list):
            from ouroboros.improvement_backlog import _stable_fingerprint

            for raw in raw_candidates[:3]:
                if not isinstance(raw, dict):
                    continue
                raw_summary = str(raw.get("summary") or "")
                raw_category = str(raw.get("category") or "process")
                raw_source = str(raw.get("source") or "execution_reflection")
                summary = _truncate_with_notice(raw_summary, 260).strip()
                category = _truncate_with_notice(raw_category, 80).strip() or "process"
                source = _truncate_with_notice(raw_source, 80).strip() or "execution_reflection"
                evidence = _truncate_with_notice(raw.get("evidence", ""), 220).strip()
                if not summary or not evidence:
                    continue
                backlog_candidates.append({
                    "fingerprint": _stable_fingerprint(raw_summary, raw_category, raw_source),
                    "summary": summary,
                    "category": category,
                    "source": source,
                    "evidence": evidence,
                    "context": _truncate_with_notice(raw.get("context", ""), 400).strip(),
                    "proposed_next_step": _truncate_with_notice(raw.get("proposed_next_step", ""), 260).strip(),
                    "task_id": _truncate_with_notice(raw.get("task_id", task_id_str), 80).strip() or task_id_str,
                    "requires_plan_review": bool(raw.get("requires_plan_review", True)),
                    "priority": _truncate_with_notice(raw.get("priority", "med"), 10).strip().lower() or "med",
                    "kind": _truncate_with_notice(raw.get("kind", "improvement"), 40).strip() or "improvement",
                })
        memory_actions = _validate_memory_actions(raw_memory_actions, task_id_str)
        memory_actions = [bound for action in memory_actions for bound in (
            knowledge.bind_entries([action]) if action["type"] == "knowledge_write" else [action])]

        # Reflection runs outside the tool-event loop; update budget directly.
        if any(refl_usage.get(key) is not None for key in ("cost", "prompt_tokens", "completion_tokens")) or refl_usage.get("ledger_attempt_ids"):
            try:
                from supervisor.state import update_budget_from_usage
                update_budget_from_usage(refl_usage)
            except Exception:
                pass
    except Exception as e:
        from ouroboros.llm_claudexor import propagate_model_error
        propagate_model_error(e)
        log.warning("Reflection LLM call failed: %s", e)
        reflection_text = f"(reflection generation failed: {e})"
        backlog_candidates = []
        memory_actions = []

    return {
        "ts": utc_now_iso(),
        "task_id": task.get("id", ""),
        "task_type": str(task.get("type", "")),
        # Two goal fields with one owner each: ``goal`` is the bounded DISPLAY
        # field every log/UI reader has always shown, ``goal_exact`` is the
        # request as the owner wrote it. A destructive writer (the Pattern
        # Register replaces its whole document) must decide from the exact text,
        # not from a 200-char display prefix that can end mid-sentence.
        "goal": goal,
        "goal_exact": str(task.get("text") or ""),
        "rounds": None if usage_dict.get("loop_evidence_unavailable") else int(usage_dict.get("rounds", 0)),
        "cost_usd": (
            round(float(usage_dict["cost"]), 4)
            if usage_dict.get("cost") is not None
            else None
        ),
        "error_count": None if llm_trace.get("loop_evidence_unavailable") else error_count,
        "key_markers": markers,
        # The typed execution classes of the children this root collected. A root
        # whose OWN calls all succeeded can still own a failed subtree, and that
        # is the common shape: children do not reflect, so the register would
        # never hear about them otherwise.
        "child_failure_classes": list(child_failure_classes or []),
        "review_evidence": review_evidence or {},
        "reflection": reflection_text,
        "backlog_candidates": backlog_candidates,
        "memory_actions": memory_actions,
        **({"source_ref": source_ref} if source_ref else {}),
        **({"memory_operation_errors": memory_operation_errors} if memory_operation_errors else {}),
    }


def apply_memory_actions(env: Any, actions: List[Dict[str, Any]], *, project_id: str = "") -> int:
    """Apply experience-review memory actions to ``env.drive_root``.

    Routes through the existing provenance-preserving memory/knowledge paths.
    Identity is intentionally conservative: an ``identity_update_candidate`` is
    recorded in the scratchpad for review, never auto-written to identity.md, so
    autonomous learning cannot silently drift the personality.

    For a project-scoped task (``project_id`` set) only KNOWLEDGE actions are
    applied: they default to that project's store through ``ToolContext.project_id``,
    while an explicit ``global`` scope on the action still reaches the shared shelf.
    Scratchpad and identity-candidate actions are skipped there.
    Returns the count of actions applied.
    """
    pid = str(project_id or "").strip()
    applied = 0
    for action in (actions or [])[:3]:
        atype = str(action.get("type") or "")
        content = str(action.get("content") or "").strip()
        if not content:
            continue
        if pid and atype in ("scratchpad_append", "identity_update_candidate"):
            continue
        try:
            if atype == "scratchpad_append":
                from ouroboros.memory import Memory

                Memory(env.drive_root, getattr(env, "repo_dir", None)).append_scratchpad_block(
                    content,
                    source="experience_review",
                    metadata={"task_id": str(action.get("task_id") or "")},
                )
                applied += 1
            elif atype == "knowledge_write":
                topic = str(action.get("topic") or "").strip()
                if not topic:
                    continue
                from ouroboros.consolidator import _write_knowledge_entries
                from ouroboros.tools.registry import ToolContext

                canonical = str(action.get("canonical_root") or getattr(env, "budget_drive_root", "") or "")
                root = pathlib.Path(canonical or env.drive_root)
                ctx = ToolContext(repo_dir=getattr(env, "repo_dir", env.drive_root), drive_root=root,
                                  budget_drive_root=canonical,
                                  project_id=pid, task_id=str(action.get("task_id") or ""))
                outcomes = _write_knowledge_entries(root / "memory" / "knowledge", [action], context=ctx)
                applied += sum(row["ok"] for row in outcomes)
                if any(not row["ok"] for row in outcomes):
                    log.warning("Reflection knowledge update was not published: %s", outcomes)
                    append_jsonl(root / "memory" / "knowledge_history.jsonl", {
                        "ts": utc_now_iso(), "type": "reflection_knowledge_write_incomplete",
                        "task_id": ctx.task_id, "proposal": action, "outcomes": outcomes,
                    })
            elif atype == "identity_update_candidate":
                from ouroboros.memory import Memory

                Memory(env.drive_root, getattr(env, "repo_dir", None)).append_scratchpad_block(
                    "IDENTITY UPDATE CANDIDATE (review before applying to identity.md):\n" + content,
                    source="experience_review_identity_candidate",
                    metadata={"task_id": str(action.get("task_id") or "")},
                )
                applied += 1
        except Exception:
            # A learned lesson that silently fails to land is invisible self-
            # learning erosion; warn so the loss is owner-greppable.
            log.warning("Failed to apply reflection memory action %s", atype, exc_info=True)
    return applied


def _admits_pattern_register(entry: Dict[str, Any]) -> bool:
    """Whether a reflection carries error evidence the Pattern Register must see.

    ONE typed gate for both writers below. ``key_markers`` alone used to decide
    it, and while that field was a substring scan the register was structurally
    blind twice over: a typed failure whose word nobody had listed did not open
    it, and a root whose own calls all succeeded while its CHILDREN failed did
    not either (children do not reflect, ARCHITECTURE Post-task reflection).

    Deliberately NOT "reason_code is non-empty": that opens on every terminal."""
    return bool(
        (entry.get("error_count") or 0) > 0
        or entry.get("key_markers")
        or entry.get("child_failure_classes")
    )


def append_reflection(drive_root: pathlib.Path, entry: Dict[str, Any]) -> None:
    """Persist a reflection entry to the JSONL file."""
    reflections_path = drive_root / "logs" / REFLECTIONS_FILENAME
    try:
        append_jsonl(reflections_path, entry)
        log.info("Execution reflection saved (task=%s, markers=%s)",
                 entry.get("task_id", "?"), entry.get("key_markers", []))
    except Exception:
        log.warning("Failed to save execution reflection", exc_info=True)

    if _admits_pattern_register(entry):
        try:
            _update_patterns(drive_root, entry)
        except Exception as exc:
            # Learning that silently fails to land is invisible erosion: the
            # register simply never hears about this class again (P1).
            log.warning("Pattern register update failed for task %s: %s",
                        entry.get("task_id", "?"), exc, exc_info=True)


def append_reflection_routed(env: Any, task: Dict[str, Any], entry: Dict[str, Any]) -> None:
    """Route the FULL reflection to its durable home (P1: process memory must
    survive drive pruning — ``env.drive_root`` for a headless-mirrored root is a
    prunable mirror, where the saga's root reflections silently died).

    A project-scoped root reflects on the PROJECT drive (the documented
    isolation: project work reflects on the project drive) and leaves a BOUNDED
    pointer row in the canonical ``logs/task_reflections.jsonl`` — never the
    full text, which feeds future global context and would leak project facts
    across projects. A non-project root reflects on the canonical budget drive
    directly. The Pattern Register update stays on the canonical drive in both
    cases and reads the WHOLE reflection plus the exact goal (owner decision
    Q2A: general error patterns are cross-project cognition, and the register
    is the one global consumer of that text)."""
    canonical = pathlib.Path(str(task.get("budget_drive_root") or "").strip() or str(env.drive_root))
    try:
        from ouroboros.project_facts import resolve_project_id

        pid = resolve_project_id(task)
    except Exception:
        pid = ""
    if not pid:
        append_reflection(canonical, entry)
        return
    from ouroboros.project_facts import project_reflections_path

    path = project_reflections_path(pid)
    project_write_failed = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(path, entry)
        log.info("Execution reflection saved to project drive (task=%s, project=%s)",
                 entry.get("task_id", "?"), pid)
    except Exception:
        project_write_failed = True
        log.warning("Failed to save project execution reflection", exc_info=True)
    if _admits_pattern_register(entry):
        try:
            _update_patterns(canonical, entry)
        except Exception as exc:
            log.warning("Pattern register update failed for task %s: %s",
                        entry.get("task_id", "?"), exc, exc_info=True)
    try:
        append_jsonl(canonical / "logs" / REFLECTIONS_FILENAME, {
            "ts": str(entry.get("ts") or utc_now_iso()),
            "task_id": str(entry.get("task_id") or ""),
            "type": "project_reflection_pointer",
            "project_id": pid,
            "reflection_path": str(path),
            # The pointer must never claim a full text that was not written: a
            # failed project append is stamped instead of silently pointing at
            # nothing (P1 — the gap is represented as a gap).
            **({"write_failed": True} if project_write_failed else {}),
        })
    except Exception:
        log.warning("Failed to write canonical reflection pointer", exc_info=True)

_PATTERNS_PROMPT = """\
You maintain a Pattern Register for Ouroboros, a self-modifying AI agent.
Below is the current register and a new error reflection. Update the register.

Rules:
- If this is a NEW error class: add a row.
- If this is a RECURRING class: increment count, update root cause/fix if you have better info.
- Keep the markdown table format.
- Be concrete: cite file names, tool names, error types.
- Max 20 rows. If full, merge least-important entries.

## Current register

{current_patterns}

## New reflection

Task: {goal}
Markers: {markers}
Reflection: {reflection}

Output ONLY the updated markdown table (with header). No extra text.
"""

_PATTERNS_HEADER = (
    "# Pattern Register\n\n"
    "| Error class | Count | Root cause | Structural fix | Status |\n"
    "|-------------|-------|------------|----------------|--------|\n"
)


def _update_patterns(drive_root: pathlib.Path, entry: Dict[str, Any]) -> None:
    """Update the Pattern Register topic via LLM."""
    from ouroboros.config import get_light_model
    from ouroboros.llm import LLMClient

    patterns_path = drive_root / "memory" / "knowledge" / "patterns.md"
    patterns_path.parent.mkdir(parents=True, exist_ok=True)

    if patterns_path.exists():
        current = patterns_path.read_text(encoding="utf-8")
    else:
        current = _PATTERNS_HEADER

    prompt = _PATTERNS_PROMPT.format(
        # This call replaces the whole file, so EVERY decision input must be
        # complete: the current register, the exact goal, and the whole
        # reflection.  Provider overflow/error is handled by the caller as an
        # abstention; a prefix can never authorize the rewrite.  A 500-char clip
        # of the reflection once cut an exculpatory clause mid-word and the
        # register recorded the inverse of what the reflection concluded.
        current_patterns=current,
        goal=str(entry.get("goal_exact") or entry.get("goal") or "?"),
        markers=", ".join(entry.get("key_markers", [])),
        reflection=str(entry.get("reflection") or ""),
    )

    light_model = get_light_model()
    client = LLMClient()
    from ouroboros.llm_observability import chat_observed

    resp_msg, patterns_usage = chat_observed(
        client,
        drive_root=drive_root,
        task_id=str(entry.get("task_id") or entry.get("id") or "patterns"),
        call_type="pattern_register_update",
        model_role="light",
        messages=[{"role": "user", "content": prompt}],
        model=light_model,
        reasoning_effort="low",
        max_tokens=16384,
    )
    # Pattern update also runs outside the tool-event loop.
    if patterns_usage:
        try:
            from supervisor.state import update_budget_from_usage
            update_budget_from_usage(patterns_usage)
        except Exception:
            pass
    updated = (resp_msg.get("content") or "").strip()
    if not updated or "|" not in updated:
        log.warning("Pattern register LLM returned invalid output, skipping update")
        return

    if not updated.startswith("#"):
        updated = "# Pattern Register\n\n" + updated

    # The LLM stays outside the stable knowledge lock.  Final source reread,
    # compare-and-swap, history decision, and atomic replacement are one critical
    # section, so two normal writers cannot both authorize from the same snapshot.
    from ouroboros.tools.knowledge import _knowledge_write_lock

    with _knowledge_write_lock(patterns_path.parent):
        try:
            latest = patterns_path.read_text(encoding="utf-8") if patterns_path.exists() else _PATTERNS_HEADER
        except Exception:
            log.warning("Pattern register source became unavailable; preserving it")
            return
        if latest != current:
            # The newer source wins and this rewrite is abandoned — so the
            # learning it carried is DROPPED, not deferred. Name whose it was:
            # a silent "preserving the newer source" hid which task's lesson the
            # register never recorded. No retry (a second paid call would decide
            # from a register that has moved again).
            log.warning(
                "Pattern register changed during this update; preserving the newer source. "
                "Task %s learning was NOT recorded in the register (no retry).",
                str(entry.get("task_id") or "?"),
            )
            return
        if not append_jsonl(drive_root / "memory" / "knowledge" / "patterns_history.jsonl", {
            "ts": utc_now_iso(),
            "task_id": str(entry.get("task_id") or ""),
            "markers": list(entry.get("key_markers") or []),
            "old_content": current,
            "new_content": updated + "\n",
        }):
            log.warning("Pattern register history unavailable; preserving the register")
            return
        write_text_atomic(patterns_path, updated + "\n")
        try:
            from ouroboros.consolidator import _rebuild_knowledge_index
            _rebuild_knowledge_index(patterns_path.parent, _locked=True)
        except Exception:
            log.debug("Failed to rebuild knowledge index after patterns update", exc_info=True)
    log.info("Pattern register updated (%d chars)", len(updated))
