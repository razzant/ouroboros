"""The native tool-round review delivery (configured-subagent api rows).

A reviewer row bound to an ``api_model`` configured subagent RETRIEVES the
subject through bounded read-only inspection rounds instead of receiving the
assembled packet. Not a third public route kind — ``api_chat`` stays the wire
vocabulary; the slot's actor binding selects this executor at the one transport
seam (``review_execution._review_route_executor``).

ONE episode is ONE logical review attempt. The reviewer can replace its own
working view through the common context materializer while exact sources and
observed reads remain in this operation's existing artifacts. Its full required
source manifest is independent from any one window. There is no round cap;
the provider window, owner deadline and paid ledger retain their own bounds.
Every provider call is its own paid row; format repair reuses the final answer.
Observed source coverage is diagnostic evidence beside that answer, never a
quorum decision or a reason to buy another review.

Context facts are available through ``compact_context(inspect=true)``. Previous
provider input, current window evidence and serialized characters are separate
observations; no estimate stops a send or vetoes the reviewer's working view.
A non-delivering or incomplete end keeps a bounded, redacted
``native_terminal_round`` copy and the ``native_*`` counters on the actor usage
row, errored episodes included (``failure_custody``).
"""

from __future__ import annotations

from ouroboros.model_wait import monotonic_now
from dataclasses import asdict, replace

import bisect
import copy
import contextlib
import functools
import hashlib
import json
import logging
import math
import pathlib
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional

from ouroboros.config import get_finalization_grace_sec
from ouroboros.deadline_utils import caller_deadline_arguments, owner_deadline_exhausted, review_transport_timeout
from ouroboros.event_usage import event_token_counts
from ouroboros.review_dispatch import bind_api_review_paid_stamp, invoke_late_review_paid_stamp
from ouroboros.review_verdict_extraction import canonicalize_session_verdict
from ouroboros.review_native_images import NativeReviewImages, staged_image_registry
from ouroboros.triad_review import request_output_shape
from ouroboros.delegate_custody_usage import observe_failed_review_send
from ouroboros.usage_accounting import (
    POSITIVE_PHYSICAL_ATTEMPT_STATES,
    BillingAuthorityUnavailable,
    BudgetExceeded,
    ControlAuthorityUnavailable,
    physical_attempt_limit,
)
from ouroboros.utils import utc_now_iso

from ouroboros.review_execution import (
    ReviewAssignment,
    ReviewAttemptResult,
    ReviewRouteKind,
    ReviewRouteUnavailable,
    ReviewSlotExecutor,
    _deadline_exhausted_error,
)

log = logging.getLogger("review_native_episode")


# Initial delivery leaves room for retrieval; this fraction never gates a send.
NATIVE_LANDING_FRACTION = 0.8

# Chars per estimated token — the `utils.estimate_tokens` heuristic inverted,
# so the transcript counter (chars) and the reviewer window (tokens) meet on
# the SAME scale the review packet sizer uses.
_CHARS_PER_ESTIMATED_TOKEN = 4

def review_native_transcript_bound(
    model_id: str, *, output_reserve: int, use_local: Optional[bool] = None,
    mandatory_read_chars: int = 0,
    model_role: str = "", credential_profile_id: Optional[str] = None,
    model_route: Optional[dict] = None,
) -> int:
    """Character sizing hint for inline versus sourced initial delivery.

    Reuses packet sizing evidence and its unknown-window assumption. Neither
    that assumption nor density calibration proves a physical context limit.
    The episode never uses this hint as admission or stopping authority.
    ``mandatory_read_chars`` describes a corpus that may span authored views.
    """
    from ouroboros.reviewer_window import reviewer_context_window, window_scaled_reserves
    from ouroboros.tools.review_helpers import calibrated_input_token_limit, REVIEW_PROMPT_TOKEN_BUDGET

    sized = reviewer_context_window(str(model_id or ""), use_local=use_local,
                                    model_role=model_role, credential_profile_id=credential_profile_id,
                                    model_route=model_route)
    window = int(sized)  # ``sized`` also carries the account the window was observed on
    if window <= 0:
        return _CHARS_PER_ESTIMATED_TOKEN * REVIEW_PROMPT_TOKEN_BUDGET
    reserve, margin = window_scaled_reserves(
        sized, output_reserve=int(output_reserve or 0), tokenizer_margin=window // 8,
        model_id=model_id, binding=dict(use_local=use_local, model_role=model_role,
                                       credential_profile_id=credential_profile_id, model_route=model_route))
    capacity = _CHARS_PER_ESTIMATED_TOKEN * max(0, int(calibrated_input_token_limit(
        str(model_id or ""), context_window=sized, output_reserve=reserve,
        tokenizer_margin=margin, budget_cap=window)))
    return max(0, capacity)


def native_mandatory_read_bound(mandatory_read_chars: int) -> int:
    """Legacy single-view sizing estimate, retained for prompt-preview callers.

    This estimate describes when a corpus needs several views; it never grants
    or refuses review authority or a send.
    """
    return math.ceil(
        (int(mandatory_read_chars) + _EPISODE_TOOL_RESULT_CHAR_CAP) / NATIVE_LANDING_FRACTION)


def native_mandatory_read_disclosure(bound: int, mandatory_read_chars: int) -> str:
    """Disclose multiwindow reading; required source size is not a send floor."""
    if int(mandatory_read_chars or 0) <= 0 or int(bound) >= native_mandatory_read_bound(mandatory_read_chars):
        return ""
    return "native_multiple_windows_required"


def native_landing_at(bound: int) -> int:
    """Initial inline share of a sizing hint, leaving retrieval working room."""
    return max(0, int(int(bound) * NATIVE_LANDING_FRACTION))


def native_mandatory_read_chars(request: Any) -> int:
    """The surface's declared mandatory reading on the request policy (chars)."""
    return int((getattr(request, "policy", None) or {}).get("native_mandatory_read_chars") or 0)


def native_episode_transcript_bound(request: Any, slot: Any, *, model_route: Optional[dict] = None) -> int:
    """The initial delivery hint of this assignment, never a runtime ceiling."""
    return review_native_transcript_bound(
        slot.model, output_reserve=int(request.max_tokens or slot.max_tokens),
        use_local=bool(slot.use_local), mandatory_read_chars=native_mandatory_read_chars(request),
        model_role=f"reviewer:{slot.slot_id}", credential_profile_id=slot.session_profile,
        **({"model_route": model_route} if model_route else {}))


def native_mandatory_read_facts(request: Any, bound: int) -> Dict[str, Any]:
    """Declared reading and its estimated multi-view layout, never a refusal."""
    declared = native_mandatory_read_chars(request)
    if not declared:
        return {}
    facts: Dict[str, Any] = {"native_mandatory_read_chars": declared}
    disclosure = native_mandatory_read_disclosure(bound, declared)
    if disclosure:
        facts["native_mandatory_read_disclosure"] = disclosure
    return facts


def native_or_packet_attempt_rail(slot: Any, two_send_surface: bool) -> Any:
    """The physical-send rail for one actor: the historical two-send rail for
    a packet/session actor on the P3 surfaces; NO local send count
    for a native tool-round slot (its bounds are the provider, owner deadline
    and paid ledger — a count would be a round cap by
    another name, and `llm.py` already rails its own recovery sends); no rail
    otherwise."""
    if not two_send_surface or bool(getattr(slot, "native_retrieval", False)):
        return contextlib.nullcontext()
    return physical_attempt_limit(2)

# ---------------------------------------------------------------------------
# The native tool-round route variant: a configured-subagent api row RETRIEVES
# the subject through bounded read-only inspection rounds instead of receiving
# the assembled packet. Not a third public route kind — ``api_chat`` stays the
# wire vocabulary; the slot's actor binding selects this executor internally.
# ---------------------------------------------------------------------------

_INSPECTION_TOOL_NAMES = (
    "read_file", "list_files", "search_code", "query_code", "query_data",
    "vcs_read", "compact_context", "view_image",
)

_NATIVE_REVIEW_INSTRUCTIONS = (
    "You are an independent Ouroboros reviewer running a bounded read-only "
    "inspection episode. Retrieve the evidence yourself with the tools you are "
    "given over the bound repository and admitted retained evidence. You cannot "
    "modify anything and have no shell. Read LARGE files in bounded chunks "
    "(read_file supports start_line/max_lines and start_char for long lines) instead of requesting a whole large "
    "document at once. The full required review surface may need several working views. "
    "compact_context(inspect=true) reports the observed view, complete units, window "
    "evidence and previous provider input; serialized size is not token occupancy. "
    "You may select keep_unit_ids "
    "and write your own working_note to retain findings and unresolved work before "
    "reading more. This continues the same review; it does not erase source or read "
    "coverage. No helper summary is needed for an authored view. Read every source "
    "required by the supplied manifest/checklist. A partial read cannot certify the "
    "whole source, and byte coverage alone does not establish semantic correctness. "
    "Your FINAL message must contain no tool calls and "
    "must follow the output contract in the task EXACTLY; your host parses it "
    "structurally, and prose around the verdict is a non-response."
)
# A tool reply's own guidance on handling evidence (its errors, results and unsupported options are facts).
_FOCUS_GUIDANCE = ("Native review focus uses your own working_note; inspect is optional. Select complete units or use "
                   "keep_last_n with your note.")
_CONTINUE_READING = (". Continue reading the remainder in bounded chunks (read_file supports start_line/max_lines and "
                     "start_char for long lines).")

# Bounded retrieval pages keep exact full sources and continuation cursors.
# This is a per-result projection, never a cumulative reading or send limit.
_EPISODE_TOOL_RESULT_CHAR_CAP = 120_000

# The episode walked away from a tool call that outlived its bound. The worker
# is still loose on the shared inspection context, so nothing it produces may
# be read as this episode's evidence.
_SOURCE_GAP_TOOL_ABANDONED = "native_tool_abandoned"


class _ToolAbandoned(Exception):
    """One inspection call outlived its bound and was left to settle alone."""


def _wire_size(messages: List[Dict[str, Any]], schemas: List[Dict[str, Any]]) -> int:
    """Canonical serialized characters, before the provider's wire adaptation.

    Opaque continuation and JSON encoding are included. This diagnostic is
    neither model input tokens nor the final sealed request's byte size.
    """
    from ouroboros.llm_attempt import _physical_candidate
    visible = _physical_candidate({"messages": messages})["messages"]
    return (len(json.dumps(visible, ensure_ascii=False, default=str))
            + len(json.dumps(schemas, ensure_ascii=False, default=str)))


def native_episode_prompt(surface: str, role_hint: str, task: str, output_contract: str, slot_id: str) -> str:
    """The ONE text of a native episode's work-order (the executor's
    ``episode_prompt`` and the commit gate's admission measure share it)."""
    return "\n".join([
        "You are an independent Ouroboros reviewer slot running a bounded "
        "read-only native inspection episode.",
        f"Surface: {surface}",
        f"Role hint: {role_hint or 'general reviewer'}",
        "",
        task,
        "",
        "OUTPUT CONTRACT (your host parses this structurally):",
        output_contract
        + "\nYour final message must contain the deliverable alone, with no tool calls.",
        f"Slot: {slot_id}",
    ])


def native_first_send_task(prompt: str, required_sources_ref: Any = None) -> str:
    """The opening's user text: the work order, plus the required-manifest reference the surface declared."""
    return prompt if required_sources_ref is None else prompt + (
        "\nRequired source manifest: " + json.dumps(required_sources_ref, ensure_ascii=False)
        + "\nSources marked delivered_inline are already included in full. Read the other required "
        "sources through their exact physical addresses, using multiple working views as needed. "
        "Observed read coverage remains diagnostic evidence beside your findings.")


def native_first_send_messages(prompt: str, required_sources_ref: Any = None) -> List[Dict[str, Any]]:
    """Cache the completed assignment, stable throughout this review episode (``native_first_send_task``;
    ``None``: no manifest). ``_open_episode`` and every pre-dispatch measure build the opening here."""
    from ouroboros.tools.review_helpers import cached_prompt_blocks

    return [{"role": "system", "content": _NATIVE_REVIEW_INSTRUCTIONS},
            {"role": "user", "content": cached_prompt_blocks(native_first_send_task(prompt, required_sources_ref))}]


def native_paged_prompt(pointer: str, output_contract: str) -> str:
    """The opening of an episode whose work order is paged: the pointer, then the contract again."""
    return pointer + "\nOUTPUT CONTRACT:\n" + output_contract


def native_material(surface: str, output_contract: str) -> list:
    """Review-contract material (``commit_gate.commit_review_contract_fingerprint``): a native episode's opening over
    a placeholder task, inline and paged, with and without a required-source manifest, its tool replies' evidence
    guidance and its inspection tool set."""
    from ouroboros.tools.review_brief_coupling import paged_review_text
    from ouroboros.tools.review_multi_model import TRIAD_ROLE_HINT

    pointer = paged_review_text("⟨text⟩", {"path": "⟨path⟩", "sha256": "⟨sha⟩"}, label="review brief", delegated=False)
    return [_NATIVE_REVIEW_INSTRUCTIONS, *(native_first_send_task(prompt, ref) for ref in (None, {"⟨ref⟩": 1}) for prompt in (
        native_episode_prompt(surface, TRIAD_ROLE_HINT, "⟨task⟩", output_contract, "⟨slot⟩"),
        native_paged_prompt(pointer, output_contract))), _FOCUS_GUIDANCE, _CONTINUE_READING, *_INSPECTION_TOOL_NAMES]


def inspection_registry(root: str, drive_root: Any, task_id: str = "") -> tuple[Any, Any, List[Dict[str, Any]]]:
    """A fresh read-only registry with ``root`` as its active subject and the
    installed repository as ``system_repo`` for canonical governance reads.

    Reuses the existing capability machinery wholesale instead of a new
    allowlist mechanism: the ``local_readonly_subagent`` constraint gives
    the read-only operation/root matrix, ``disabled_tools`` trims its
    broader name allowlist down to inspection tools, and the resource
    contract keeps extension/MCP discovery off. ``registry._ctx`` is
    per-instance, so the worker's own registry/context is never touched.
    ``drive_root`` is the data plane the inspection tools may read: an
    empty scratch directory by default, or the surface's opt-in
    ``policy["native_data_root"]`` (a stored paged source or skill index
    resolves there; deep self-review exposes the runtime root while its
    memory whitelist arrives inline) — the read-only constraint applies to
    it exactly as to the repository.
    """
    import pathlib as _pathlib

    from ouroboros.tool_capabilities import LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from ouroboros.tools.review_helpers import REPO_ROOT

    registry = ToolRegistry(repo_dir=_pathlib.Path(root), drive_root=_pathlib.Path(drive_root))
    ctx = ToolContext(
        repo_dir=_pathlib.Path(root),
        system_repo_dir=REPO_ROOT,
        drive_root=_pathlib.Path(drive_root),
        task_id=str(task_id or "") or None,
        task_lifecycle_bound=False,  # a standalone inspection episode is an operation
        task_constraint={"mode": "local_readonly_subagent"},
        task_contract={
            "allowed_resources": {"network": False, "web": False},
            "disabled_tools": sorted(
                set(LOCAL_READONLY_SUBAGENT_TOOL_NAMES) - set(_INSPECTION_TOOL_NAMES)
            ),
        },
    )
    from ouroboros.usage_accounting import current_usage_scope
    scope = current_usage_scope()
    if scope is not None and scope.task_id and not scope.non_task_operation:
        # A paid review under a task keeps that task's canonical tree controls;
        # its scratch drive and reviewer label cannot supply Pause authority.
        ctx.task_id = scope.task_id
        ctx.task_metadata = {"root_task_id": scope.root_task_id or scope.task_id,
                             "budget_drive_root": str(scope.drive_root or drive_root)}
        ctx.task_lifecycle_bound = True
    ctx.task_metadata["native_data_root"] = str(drive_root)
    registry.set_context(ctx)
    schemas = [schema for schema in (registry.get_schema_by_name(name) for name in _INSPECTION_TOOL_NAMES) if schema]
    if not any(s["function"]["name"] == "compact_context" for s in schemas):
        from ouroboros.tools.compact_context import get_tools
        schemas.append({"type": "function", "function": get_tools()[0].schema})
    if not schemas:
        raise ReviewRouteUnavailable(
            "no inspection tool schemas are projectable for the native "
            "tool-round episode", code="native_inspection_unavailable")
    return registry, ctx, schemas


def native_first_send_chars(
    root: str, *, surface: str, role_hint: str, slot_id: str, session_task: str,
    output_contract: str, task_id: str = "", required_sources_ref: Any = None,
) -> int:
    """The wire size of a native episode's FIRST send — the prompt its first
    ``reserve_attempt`` prices (every later round reserves itself, on the
    ledger, against the transcript it has grown to). Built from the same
    work-order, message objects (the declared manifest reference included),
    schemas and measure ``_open_episode`` uses, so an initial layout decision and
    the wave admission price a native seat the way the ledger will; the registry
    is built read-only against ``root`` and discarded."""
    _registry, _ctx, schemas = inspection_registry(root, root, task_id)
    prompt = native_episode_prompt(surface, role_hint, str(session_task or "").strip(), output_contract, slot_id)
    return _wire_size(native_first_send_messages(prompt, required_sources_ref), schemas)

class NativeToolRoundReviewExecutor(ReviewSlotExecutor):
    """Bounded native inspection episode for a configured-subagent api row.

    ONE episode is ONE logical review attempt: ``LLMClient.chat(tools=…)``
    calls against a fresh, instance-local inspection-only ``ToolRegistry``
    until the reviewer answers or a provider, deadline or paid-ledger outcome
    ends the work; never a transcript-size or round-count policy. Every
    provider call is its own ledger row (the ambient usage scope attributes
    them); the coordinator's second actor attempt repairs FORMAT locally over
    the collected answer, exactly like the session executor — there is no
    automatic resume or a separate per-round paid ledger. Authored view changes
    preserve complete sources and reads in the existing task artifact store.
    Exhaustion is a typed refusal for verdict shapes; the report shape delivers
    what was collected, marked INCOMPLETE.
    """

    route = ReviewRouteKind.API_CHAT

    def __init__(self, assignment: ReviewAssignment, *, llm: Any = None):
        from ouroboros.model_wait import current_model_wait
        from ouroboros.review_records import apply_review_model_override

        self._waiter = current_model_wait()
        if self._waiter:
            assignment = replace(assignment, slot=apply_review_model_override(assignment.slot, self._waiter.overrides))
        super().__init__(assignment, llm=llm)
        self._sizing_hint = 0
        self._window: dict = {}
        self._previous_request: Optional[dict] = None
        self._episode_prompt: Optional[str] = None
        self._initial_source: dict = {}
        self._raw_transcript: Optional[str] = None
        self._episode_usage: Dict[str, Any] = {}
        self._tool_receipts: List[Dict[str, Any]] = []
        self._pending_staged_reads: list = []
        self._inspection_ctx: Any = None  # the registry's context: the reader stamps `last_read_view` on it
        self._tool_calls_total = 0
        self._rounds_used = 0
        self._episode_deltas: List[Dict[str, Any]] = []
        self._settled_failure: Optional[BaseException] = None
        self._round_source_refs: list[dict] = []
        self._source_gap = ""
        self._view_changes = 0
        self._view_receipt: dict = {}
        self._round_messages: list[dict] = []
        self._last_persisted_round = 0
        self._tool_abandoned = False  # a loose worker retires this episode's read attribution
        self._images = NativeReviewImages()

    def _source_root(self) -> Optional[pathlib.Path]:
        root = (self.assignment.request.policy or {}).get("native_data_root") or self.assignment.custody_root
        return pathlib.Path(root) if root else None

    def _store_source(self, source_id: str, payload: Any, *, category: str = "context_checkpoints") -> dict:
        """Use the operation's existing actor-readable store; failure remains a fact."""
        from ouroboros.artifacts import store_actor_source_bytes
        root = self._source_root()
        if root is None:
            self._source_gap = "native_source_root_unavailable"
            return {}
        try:
            data = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            task_id = self.assignment.request.task_id or "review"
            ref = store_actor_source_bytes(root, task_id, category=category,
                source_id=f"{self.assignment.call_id}-{source_id}", data=data,
                extension="txt" if isinstance(payload, str) else "json")
            return ref
        except Exception as exc:
            self._source_gap = f"native_source_persistence_failed:{type(exc).__name__}"
            return {}

    def _record_round_source(self) -> None:
        if not self._round_messages or self._rounds_used <= self._last_persisted_round:
            return
        ref = self._store_source(f"round-{self._rounds_used}", self._images.source_projection({
            "round": self._rounds_used, "messages": self._round_messages,
            "read_receipts": self._tool_receipts, "view_receipt": self._view_receipt,
            "sent_view": getattr(self._inspection_ctx, "_last_context_observation", {}),
        }))
        if ref:
            self._round_source_refs.append(ref)
        self._last_persisted_round = self._rounds_used

    def _observe_sent_view(self, messages: list, schemas: list, *, confirmed: bool = True) -> None:
        from ouroboros.tools.compact_context import record_context_view
        from ouroboros.transcript_prefix import count_accepted_send
        count_accepted_send(self._inspection_ctx, messages, schemas, round_idx=self._rounds_used, usage=self._episode_usage)
        record_context_view(self._inspection_ctx, messages, schemas)
        self._images.observe(self._rounds_used, confirmed=confirmed)
        for receipt in self._tool_receipts:
            receipt["delivered"] = True
        from ouroboros.delegate_output import acknowledge_staged_output_delivery
        for execution, projected, shown in self._pending_staged_reads:
            acknowledge_staged_output_delivery(self._inspection_ctx, execution, projected, shown_ranges=shown)
        self._pending_staged_reads.clear()

    def _retire_reasoning(self, messages: list, schemas: list, round_idx: int) -> None:
        from ouroboros.transcript_prefix import append_reasoning_retirement_fact, prepare_reasoning_for_send

        if prepare_reasoning_for_send(self._inspection_ctx, messages, schemas,
                                      round_idx=round_idx, usage=self._episode_usage):
            append_reasoning_retirement_fact(self._episode_usage, round_idx, messages)

    def _refresh_window(self, model_route: Optional[dict] = None) -> None:
        """Window evidence for the current route, never an assumed send limit."""
        from ouroboros.reviewer_window import resolve_reviewer_window

        slot = self.assignment.slot
        self._window = asdict(resolve_reviewer_window(
            slot.model, use_local=slot.use_local, model_role=f"reviewer:{slot.slot_id}",
            credential_profile_id=slot.session_profile, model_route=model_route, allow_fetch=False))
        for key in ("window_tokens", "asserted_window_tokens"):
            self._window[key] = self._window[key] or None
        self._window["status"] = self._window["status"] or "unknown"

    def _context_facts(self, messages: Optional[list] = None, schemas: Optional[list] = None) -> dict:
        """A view's representation and a previous response are different facts."""
        from ouroboros.context_compaction import context_reclaim_transcript_sha256

        observed = getattr(self._inspection_ctx, "_last_context_observation", None) or {}
        messages = observed.get("messages", []) if messages is None else messages
        schemas = observed.get("tool_schemas", []) if schemas is None else schemas
        return {
            "view_revision": context_reclaim_transcript_sha256(messages),
            "serialized_chars": _wire_size(messages, schemas),
            "measurement_basis": "canonical_serialized_characters",
            "window": copy.deepcopy(self._window),
            "previous_request": copy.deepcopy(self._previous_request),
            "output_allowance_tokens": int(self.assignment.request.max_tokens or self.assignment.slot.max_tokens),
            "images": copy.deepcopy(list(self._images.facts.values())),
        }

    def _apply_working_view(self, messages: list, schemas: list, round_idx: int) -> bool:
        from ouroboros.context_budget import ContextReclaimRequest
        from ouroboros.context_compaction import compact_tool_history_llm, context_reclaim_transcript_sha256

        pending = getattr(self._inspection_ctx, "_pending_compaction", None)
        if pending is None:
            return False
        self._inspection_ctx._pending_compaction = None
        if not isinstance(pending, dict):
            self._view_receipt = {"status": "authored_note_required",
                                  "reason": "Inspect the working view and provide your own working_note."}
            return False
        root = self._source_root()
        if root is None:
            self._view_receipt = {"status": "checkpoint_failed", "reason": "native_source_root_unavailable"}
            return False
        observed = pending["observed"]
        request = ContextReclaimRequest(
            route_fp=f"reviewer:{self.assignment.slot.slot_id}", round_id=str(round_idx),
            transcript_sha256=context_reclaim_transcript_sha256(messages),
            measurement_basis="cold_estimate", measurement_density=1.0, reclaim_goal_tokens=0,
            **{key: pending[key] for key in ("working_note", "expected_view_revision", "keep_unit_ids", "restore_unit_refs", "schema_names")})
        def fit(candidate, selected_tools):
            return {"accepted": True, "input_chars": _wire_size(candidate, selected_tools),
                    "output_reserve_tokens": int(self.assignment.request.max_tokens or self.assignment.slot.max_tokens),
                    "measurement_basis": "canonical_serialized_characters", "strict_bound_proven": False}
        candidate, receipt = compact_tool_history_llm(
            messages, request=request, observed_messages=observed["messages"],
            observed_tool_schemas=observed["tool_schemas"], tool_schemas=schemas,
            fit_candidate=fit, drive_root=root, task_id=self.assignment.request.task_id or "review")
        self._view_receipt = asdict(receipt)
        ref = self._store_source(f"view-{round_idx}", self._view_receipt)
        if receipt.status == "applied":
            from ouroboros.transcript_prefix import sanction_rewrite
            messages[:] = candidate
            sanction_rewrite(self._inspection_ctx, "compaction")  # the reviewer's own fold names its change
            self._view_changes += 1
        if receipt.status != "no_op":
            notice = {"role": "user", "content": json.dumps({"context_view_status": receipt.status,
                       "receipt_source": ref, "reclaimed_tokens": receipt.reclaimed_tokens}, ensure_ascii=False)}
            messages.append(notice)
        return receipt.status == "applied"

    def _read_coverage(self) -> dict:
        """Fold exact delivered intervals over the caller's complete required manifest.

        Four states: ``unobserved`` when the surface declared no manifest at all,
        ``declared_empty`` (a complete coverage of nothing) when it declared an
        empty one, and ``complete``/``incomplete`` over a declared manifest.
        """
        from ouroboros.tool_access import resource_root_path
        policy = self.assignment.request.policy or {}
        required = policy.get("native_required_sources")
        if not isinstance(required, list):
            return {"status": "unobserved", "reason": "required_source_manifest_missing", "sources": []}
        if not required:
            return {"status": "complete", "reason": "declared_empty", "sources": [],
                    "required_source_count": 0}
        if self._inspection_ctx is None:
            return {"status": "unobserved", "reason": "native_inspection_unavailable", "sources": required}
        rows = []
        for source in required:
            row = dict(source) if isinstance(source, dict) else {"source": source}
            try:
                total = row["complete_chars"]
                if (type(total) is not int or total < 0 or len(row["source_revision"]) != 64
                        or len(row["complete_sha256"]) != 64 or row["range_basis"] != "unicode_text_universal_newlines"):
                    raise ValueError("invalid required source identity")
                if row.get("coverage_basis") == "delivered_inline":
                    row.update(status="complete", missing_ranges=[], covered_chars=total)
                    rows.append(row)
                    continue
                base = resource_root_path(self._inspection_ctx, row["root"])
                spans = []
                for receipt in self._tool_receipts:
                    if (receipt.get("tool") != "read_file" or receipt.get("outcome") != "executed"
                            or receipt.get("delivered") is not True
                            or receipt.get("source_gap")
                            or receipt.get("source_revision") != row["source_revision"]
                            or receipt.get("complete_sha256") != row["complete_sha256"]
                            or receipt.get("complete_chars") != total or receipt.get("opened_path") != row["path"]
                            or resource_root_path(self._inspection_ctx, receipt["opened_root"]) != base):
                        continue
                    spans.append((receipt["source_start_char"], receipt["source_end_char"]))
                cursor, missing = 0, []
                for lo, hi in sorted(spans):
                    if lo > cursor:
                        missing.append([cursor, lo])
                    cursor = max(cursor, hi)
                if cursor < total:
                    missing.append([cursor, total])
                row.update(status="complete" if spans and not missing else "incomplete", missing_ranges=missing,
                           covered_chars=total - sum(hi - lo for lo, hi in missing))
            except (KeyError, TypeError, ValueError):
                row.update(status="unobserved", reason="required_source_identity_unavailable")
            rows.append(row)
        return {"status": "complete" if all(r["status"] == "complete" for r in rows) else "incomplete",
                "sources": rows, "required_source_count": len(rows)}

    def _episode_source_facts(self) -> dict:
        """Keep full source history separate from the final working projection."""
        self._record_round_source()
        policy = self.assignment.request.policy or {}
        coverage = self._read_coverage()
        if self._source_gap and coverage["status"] == "complete":
            coverage = {**coverage, "status": "incomplete", "reason": self._source_gap}
        history_ref = self._store_source("history", {
            "required_sources": policy.get("native_required_sources"),
            "required_sources_ref": policy.get("native_required_sources_ref"),
            "round_sources": self._round_source_refs, "read_receipts": self._tool_receipts,
            "coverage": coverage, "view_changes": self._view_changes,
        })
        if self._source_gap and coverage["status"] == "complete":
            coverage = {**coverage, "status": "incomplete", "reason": self._source_gap}
        facts = {"native_read_coverage": coverage, "native_history_source": history_ref,
                 "native_view_changes": self._view_changes, "native_source_gap": self._source_gap}
        if self._initial_source:
            facts["native_initial_source"] = self._initial_source
        if coverage["status"] == "incomplete":
            facts["native_incomplete"] = "required_source_coverage_incomplete"
        return facts

    def _complete_tool_round(self, registry, tool_calls, validation_by_id, messages, schemas, round_idx):
        """Complete all tool pairs before fitting an actor-requested working view."""
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tool_message = self._execute_inspection_call(
                registry, tc, validation_by_id, round_idx=round_idx)
            self._round_messages.append(copy.deepcopy(tool_message))
            messages.append(tool_message)
        self._round_messages.extend(copy.deepcopy(self._images.append_after_tools(messages, self._inspection_ctx)))
        self._apply_working_view(messages, schemas, round_idx)
        transcript_chars = _wire_size(messages, schemas)
        self._record_round_source()
        return transcript_chars

    # -- prompt (route-owned; never the api pack) ------------------------------

    def prompt_payload(self) -> Dict[str, Any]:
        return {
            "native_episode_prompt": self.episode_prompt,
            "inspection_tools": list(_INSPECTION_TOOL_NAMES),
        }

    def prompt_chars(self) -> int:
        return len(self.episode_prompt)

    @property
    def episode_prompt(self) -> str:
        """The SAME compact work-order the session route sends (D12) — the slot's
        own (``slot_session_tasks``) or the shared ``session_task``: task,
        criteria and output contract, with only the evidence the surface chose
        to include — the actor retrieves context itself; the api pack is never
        assembled here."""
        if self._episode_prompt is None:
            request, slot = self.assignment.request, self.assignment.slot
            task = str((getattr(request, "slot_session_tasks", None) or {}).get(slot.slot_id)
                       or request.session_task or "").strip()
            if not task:
                raise ReviewRouteUnavailable(
                    "native tool-round slot has no session task: the surface must "
                    "supply the route-owned task text (request.session_task) — the "
                    "assembled api pack is deliberately not sendable to a retrieving "
                    "actor", code="session_task_missing")
            self._episode_prompt = native_episode_prompt(
                request.surface, slot.role_hint, task, self._output_contract(), slot.slot_id)
        return self._episode_prompt

    # -- delivery --------------------------------------------------------------

    def execute(self) -> ReviewAttemptResult:
        if self._raw_transcript is not None:
            # The permitted resend repairs FORMAT locally over the collected
            # final answer; it never runs a second paid episode.
            return self._verdict_result(force_extraction=True)
        if self._settled_failure is not None:
            raise self._settled_failure
        try:
            self._run_episode()
        except BaseException as exc:
            self._settled_failure = exc
            raise
        return self._verdict_result()

    def _inspection_registry(self, root: str, drive_root: Any) -> tuple[Any, List[Dict[str, Any]]]:
        """A fresh instance-local registry pinned to ``root``, read-only.

        Reuses the existing capability machinery wholesale instead of a new
        allowlist mechanism: the ``local_readonly_subagent`` constraint gives
        the read-only operation/root matrix, ``disabled_tools`` trims its
        broader name allowlist down to the inspection six, and the resource
        contract keeps extension/MCP discovery off. ``registry._ctx`` is
        per-instance, so the worker's own registry/context is never touched.
        ``drive_root`` is the data plane the inspection tools may read: an
        empty scratch directory by default, or the surface's opt-in
        ``policy["native_data_root"]`` (a stored paged source or skill index
        resolves there; deep self-review exposes the runtime root while its
        memory whitelist arrives inline) — the read-only constraint applies to
        it exactly as to the repository.
        """
        registry, self._inspection_ctx, schemas = inspection_registry(
            root, drive_root, str(self.assignment.request.task_id or "review"))
        return registry, schemas

    def _reprepare_model_call(self, values: dict, *, messages: list) -> dict:
        """Continue one frozen episode on a new route, with fresh window facts.

        Only routing and provider-private continuation state change. The same
        inspection schemas and completed tool results remain on the call.
        """
        from ouroboros.llm import LLMClient
        from ouroboros.review_records import apply_review_model_override

        slot = self.assignment.slot
        updated = apply_review_model_override(slot, {f"reviewer:{slot.slot_id}": {
            "model": values["model"], "use_local": values.get("use_local", False),
            "model_account_override": values.get("model_account_override", slot.session_profile)}})
        messages[:] = LLMClient.sanitize_reasoning_on_model_switch(messages, slot.model, updated.model)
        self.assignment = replace(self.assignment, slot=updated)
        self._refresh_window(values.pop("_model_observed_route", None))
        self._retire_reasoning(messages, values.get("tools"), self._rounds_used + 1)
        values["messages"] = self._images.prepare(messages, updated)
        return values

    def _run_episode(self) -> None:
        from ouroboros.llm import add_usage
        from ouroboros.openai_chat_dispatch import custom_validation_by_call_id, pop_custom_validation_receipts

        request, slot = self.assignment.request, self.assignment.slot
        root = str(request.session_root or "").strip()
        chat = getattr(self.llm, "chat", None)
        if not callable(chat):
            raise ReviewRouteUnavailable(
                "native tool-round episode needs a synchronous chat transport",
                code="api_chat_unavailable")
        deadline_at = str(getattr(request, "deadline_at", "") or "")
        max_tokens = int(request.max_tokens or slot.max_tokens)
        self._sizing_hint = native_episode_transcript_bound(request, slot)
        self._refresh_window()
        shape = request_output_shape(request)
        # The operation's durable task root keeps continuation sources readable.
        # Explicit native_data_root still wins; only our own scratch is removed.
        data_root = str(self._source_root() or "").strip()
        scratch = tempfile.mkdtemp(prefix="ouro-native-review-")
        registry = None
        total_usage: Dict[str, Any] = {}
        self._episode_usage = total_usage
        final_answer: Optional[str] = None
        last_content = ""  # the reviewer's latest prose — the product of an exhausted report episode
        end_reason = "not_started"  # the true reason is set by whichever end the episode takes
        round_idx = 0
        transcript_chars = last_send_chars = 0
        episode: Dict[str, Any] = {}
        messages: List[Dict[str, Any]] = []
        observed_route: dict = {}

        try:
            end_reason = "registry_unavailable"
            registry, schemas, messages, transcript_chars = self._open_episode(root, data_root or scratch)
            end_reason = "running"
            logical_deadline = getattr(self, "_logical_deadline_monotonic", None)
            while True:
                # Time, money and provider outcomes have their own owners.
                # Approximate context sizing never decides whether to continue.
                if owner_deadline_exhausted(
                    deadline_at=deadline_at, reserve_sec=get_finalization_grace_sec(),
                ) or (logical_deadline is not None and monotonic_now() >= float(logical_deadline)):
                    end_reason = "deadline_exhausted"
                    if shape == "report" and last_content:
                        break  # a report keeps its draft (marked incomplete below)
                    raise _deadline_exhausted_error("owner deadline exhausted mid native review episode")
                round_idx += 1
                self._retire_reasoning(messages, schemas, round_idx)
                chat_kwargs = self._chat_kwargs(messages, schemas, max_tokens)
                transport = review_transport_timeout(
                    slot.model, getattr(slot, "transport_timeout_sec", None), deadline_at,
                )
                if logical_deadline is not None:
                    # Recomputed immediately before dispatch: a window that
                    # expired since the round's admission check takes the
                    # deadline path (a report keeps its draft), and a positive
                    # remainder bounds the send with NO floor.
                    remaining = float(logical_deadline) - monotonic_now()
                    if remaining <= 0:
                        end_reason = "deadline_exhausted"
                        if shape == "report" and last_content:
                            break
                        raise _deadline_exhausted_error("slot logical window exhausted before the native review send")
                    transport = remaining if transport is None else min(float(transport), remaining)
                if transport is not None:
                    chat_kwargs["timeout"] = transport
                chat_kwargs.update(caller_deadline_arguments(deadline_at, logical_deadline,
                                   reserve_sec=get_finalization_grace_sec()))
                preparation_scope = (self._waiter.register_reprepare(f"reviewer:{slot.slot_id}", functools.partial(self._reprepare_model_call, messages=messages))
                                     if self._waiter else contextlib.nullcontext())
                with bind_api_review_paid_stamp(self.assignment.dispatch_stamp), preparation_scope:
                    self._round_messages = []
                    try:
                        msg, usage = chat(**chat_kwargs)
                    except BaseException as exc:
                        # A budget or authority refusal of the NEXT send is not a transport fault.
                        end_reason = ("budget_exhausted" if isinstance(exc, BudgetExceeded) else "billing_authority_unavailable"
                                      if isinstance(exc, BillingAuthorityUnavailable) else "control_authority_unavailable"
                                      if isinstance(exc, ControlAuthorityUnavailable) else "transport_error")
                        capture = getattr(exc, "physical_attempt_capture", None)
                        sent = self._images.sent_messages
                        transcript_chars = _wire_size(sent, schemas)
                        stamp_failed = False  # a strict stamp that failed after this send ends the episode here
                        if str(getattr(capture, "state", "") or "") in POSITIVE_PHYSICAL_ATTEMPT_STATES:
                            # A send that was physically dispatched IS a round
                            # of this episode, even when its response never
                            # came back: its receipt keys and custody must not
                            # read as a zero-send refusal.
                            self._rounds_used, last_send_chars = round_idx, transcript_chars  # a dispatched send IS the last one
                            self._observe_sent_view(messages, schemas, confirmed=False)
                            stamp_failed = invoke_late_review_paid_stamp(self.assignment.dispatch_stamp)
                        observe_failed_review_send(self.usage_observer, exc)
                        if not stamp_failed and self._images.refuse_send(exc, self.assignment.slot, round_idx, transcript_chars):
                            end_reason = "running"
                            continue  # Same episode, explicit gap; no estimate suppresses an image send.
                        if isinstance(exc, (BudgetExceeded, BillingAuthorityUnavailable, ControlAuthorityUnavailable)) and shape == "report" and last_content:
                            break  # nothing was sent; a report keeps its draft
                        raise
                transcript_chars = _wire_size(self._images.sent_messages, schemas)
                self._rounds_used, last_send_chars = round_idx, transcript_chars  # a returned send is the last physical send
                self._observe_sent_view(messages, schemas)
                slot = self.assignment.slot
                raw_calls = msg.get("tool_calls") if isinstance(msg, dict) else None
                # absent = no calls; a list = calls; ANY other value (falsy too) = one malformed entry
                tool_calls = [] if raw_calls is None else (raw_calls if isinstance(raw_calls, list) else [raw_calls])
                usage = dict(usage or {})
                route_fact = (usage.get("claudexor") or {}).get("route") or {}
                if route_fact and route_fact != observed_route:
                    observed_route = route_fact
                    self._refresh_window(route_fact)
                # One returned request, before accumulation. Cached input is
                # already part of normalized prompt_tokens, never added again.
                self._previous_request = {
                    "round": round_idx,
                    "view_revision": self._inspection_ctx._last_context_observation["revision"],
                    "prompt_tokens": event_token_counts(usage)["prompt_tokens"],
                    "model": str(usage.get("resolved_model") or ""),
                    "provider": str(usage.get("provider") or ""),
                    "requested_model": slot.model, "requested_profile": slot.session_profile,
                    "ledger_attempt_ids": list(usage.get("ledger_attempt_ids") or []),
                    "route": copy.deepcopy(usage.get("model_role_route") or route_fact),
                }
                self._observe_usage(usage)
                # Pop the wire-validation sidecar BEFORE accumulation (receipts are per-round facts, not usage).
                wire_validation = pop_custom_validation_receipts(usage, tool_calls)
                validation_by_id = custom_validation_by_call_id(wire_validation)
                add_usage(total_usage, usage)
                # add_usage accumulates only token/cost keys: carry the ledger
                # linkage and provenance facts the substrate's actor records
                # consume, or the episode rollup would echo the REQUESTED model
                # dressed as resolved and lose its physical attempt ids.
                for _attempt_id in (usage.get("ledger_attempt_ids") or []):
                    total_usage.setdefault("ledger_attempt_ids", []).append(_attempt_id)
                for _fact in ("resolved_model", "provider", "claudexor", "model_role_route"):
                    if usage.get(_fact):
                        total_usage[_fact] = usage[_fact]
                content = str(msg.get("content") or "") if isinstance(msg, dict) else ""
                if content:
                    last_content = content
                # The envelope joins the record BEFORE any terminal branch, so
                # the terminal-round fact describes the decision-ending output
                # itself — an empty or malformed final round included. The WHOLE
                # dict joins, including opaque continuation and readable reasoning;
                # canonical custody is independent from outbound projections.
                assistant = dict(msg) if isinstance(msg, dict) else {"content": content}
                assistant.setdefault("role", "assistant")
                messages.append(assistant)
                self._round_messages = [copy.deepcopy(assistant)]
                if not tool_calls:
                    # The reviewer's answer — or an empty round, which is the
                    # episode's honest end: the empty answer rides the ordinary
                    # empty-response rail upstream (a report keeps its draft).
                    final_answer = content
                    end_reason = "final_answer" if content else "empty_answer"
                    break
                well_formed = [
                    tc for tc in tool_calls
                    if isinstance(tc, dict) and isinstance(tc.get("function"), dict)
                    and str(tc["function"].get("name") or "").strip()
                ]
                if not content and not well_formed:
                    # PROGRESS FLOOR (P13: a floor, never a ceiling): a round
                    # that carries tool calls but no well-formed one and no
                    # prose is malformed provider output — it adds nothing to
                    # the transcript and would re-enter the paid send forever.
                    # (An EMPTY answer is different: it is the episode's honest
                    # end and rides the ordinary empty-response rail above.)
                    end_reason = "round_without_progress"
                    break
                transcript_chars = self._complete_tool_round(
                    registry, tool_calls, validation_by_id, messages, schemas, round_idx)
            if shape == "report" and not final_answer and last_content:
                # A report is a product, not a verdict: the collected draft is
                # delivered marked INCOMPLETE rather than discarded (the budget
                # or deadline ended, the round made no progress, or the final round came
                # back empty) — the consumer discloses it; nothing unseen is
                # summarized into it. Decided BEFORE the custody row is written
                # so the durable fact knows the product is partial.
                final_answer = last_content
                episode["native_incomplete"] = end_reason
        finally:
            for key, value in self._episode_source_facts().items():
                episode.setdefault(key, value)
            # Only the host's own scratch is removed; an opted-in data root
            # belongs to the caller and survives a failed episode untouched.
            shutil.rmtree(scratch, ignore_errors=True)
            if (not final_answer or episode.get("native_incomplete")) and any(
                m.get("role") == "assistant" for m in messages
            ):
                # Non-delivering (refused, or an EMPTY final answer riding the
                # empty-response rail) or incomplete: the record keeps the
                # decision-ending envelope itself.
                # The terminal round — the exact assistant envelope and the
                # tool results that led to a budget, deadline or transport end
                # — is not reconstructible from the receipts alone (P1): keep
                # a bounded copy on the episode facts, redacted like every
                # other projected review artifact. Assembled HERE so every
                # non-delivering end carries it, exception ends included.
                try:
                    episode["native_terminal_round"] = self._terminal_round_fact(messages)
                except Exception as exc:  # custody must never mask the episode's own end
                    episode["native_terminal_round"] = json.dumps(
                        {"error": "terminal_round_unavailable", "reason": type(exc).__name__})
            # One typed custody row per episode END — including the ends that
            # leave through an exception (deadline, transport, registry): a
            # refused episode used to leave no trace of how far it got. The
            # actor usage is assembled HERE too, so a refused or errored
            # episode still carries its delivery, rounds, receipts and paid
            # ledger facts (`failure_custody` hands them to the error actor).
            episode.update({
                "native_rounds": self._rounds_used,
                "native_tool_calls": self._tool_calls_total,
                "native_transcript_chars": last_send_chars,  # canonical characters of the last sent view
                "native_context": self._context_facts(),
                **native_mandatory_read_facts(request, self._sizing_hint),
                "native_end_reason": end_reason,
            })
            episode["native_custody_row"] = self._emit_episode_fact(episode)
            self._episode_usage = dict(total_usage)
            self._episode_usage.update({
                "provider": str(total_usage.get("provider") or ""),
                # The slot model stands in for an unreported resolved model ONLY
                # once a provider round actually ran: a pre-send refusal leaves
                # the receipt keys empty, or the public execution wire would
                # mint a native run for an episode that never sent anything.
                "resolved_model": str(
                    total_usage.get("resolved_model") or (slot.model if self._rounds_used else "")),
                **episode,
                "native_tool_receipts": list(self._tool_receipts),
                # Provenance class of this delivery: the host SAW these reads.
                "host_file_read_attestation": "host_observed",
                "delivery": "native_tool_rounds",
            })
        if final_answer is None:
            raise ReviewRouteUnavailable(
                f"native review episode round {self._rounds_used} carried neither "
                "an answer nor a well-formed tool call; the episode fails closed "
                "— a zero-progress round would re-enter the paid send forever",
                code="native_round_without_progress")
        if episode.get("native_incomplete") == "required_source_coverage_incomplete":
            self._episode_deltas.append({
                "kind": "capability_delta", "requested": "complete required-source review coverage",
                "effective": "independent final answer with incomplete or unobserved required source coverage",
                "reason": "native_required_source_coverage_incomplete",
            })
        elif episode.get("native_incomplete"):
            self._episode_deltas.append({
                "kind": "capability_delta",
                "requested": "a finished report from the episode",
                "effective": (
                    f"the reviewer's last draft ({len(final_answer)} chars) — "
                    f"{end_reason} after {self._rounds_used} rounds without a final "
                    f"answer (canonical transcript {transcript_chars} chars)"
                ),
                "reason": f"native_{end_reason}_before_final_answer",
            })
        self._raw_transcript = final_answer

    def _open_episode(self, root: str, drive_root: Any) -> tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """Complete the assignment before declaring its stable cache prefix."""
        if not root:
            raise ReviewRouteUnavailable(
                "native tool-round slot has no session root: the surface must name "
                "the repository root the reviewer episode runs in",
                code="session_root_missing")
        registry, schemas = self._inspection_registry(root, drive_root)
        policy = self.assignment.request.policy or {}
        required = policy.get("native_required_sources")
        ref = None if required is None else (
            policy.get("native_required_sources_ref") or self._store_source("required-sources", required))
        messages = native_first_send_messages(self.episode_prompt, ref)
        # Only initial layout: later working views remain the reviewer's choice.
        # Other surfaces may supply a large work order without the brief builder.
        bound = native_episode_transcript_bound(self.assignment.request, self.assignment.slot)
        if _wire_size(messages, schemas) >= native_landing_at(bound):
            source = self._store_source("initial-brief", self.episode_prompt)
            if source:
                from ouroboros.tools.review_brief_coupling import paged_review_text
                from ouroboros.tools.scope_required_sources import source_text_identity

                self._initial_source = source
                required = [{**row, "coverage_basis": "candidate_blob"}
                            if row.get("coverage_basis") == "delivered_inline" else row
                            for row in (required or [])]
                required.append({"root": "artifact_store", "path": source["path"],
                                 **source_text_identity(self.episode_prompt.encode("utf-8")),
                                 "coverage_basis": "candidate_blob"})
                policy["native_required_sources"] = required
                ref = self._store_source("required-sources", required)
                policy["native_required_sources_ref"] = ref
                messages = native_first_send_messages(native_paged_prompt(paged_review_text(
                    self.episode_prompt, source, label="review brief", delegated=False), self._output_contract()), ref)
        return registry, schemas, messages, _wire_size(messages, schemas)

    def _chat_kwargs(self, messages: List[Dict[str, Any]], schemas: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
        """One round's provider call, shaped from the request and the slot."""
        request, slot = self.assignment.request, self.assignment.slot
        kwargs: Dict[str, Any] = {
            "messages": self._images.prepare(messages, slot),
            "model": slot.model,
            "model_role": f"reviewer:{slot.slot_id}",
            "model_account_override": slot.session_profile,
            "tools": schemas,
            "tool_choice": "auto",
            "reasoning_effort": slot.effort,
            "processing_preference": slot.processing_preference,
            "max_tokens": max_tokens,
            "no_proxy": bool(request.no_proxy),
            "use_local": bool(slot.use_local),
            "cache_affinity": f"{request.surface}:{request.task_id or 'review'}",
        }
        if request.temperature is not None or slot.temperature is not None:
            kwargs["temperature"] = request.temperature if request.temperature is not None else slot.temperature
        default_temperature = getattr(request, "default_temperature", None)
        if default_temperature is None:
            default_temperature = getattr(slot, "default_temperature", None)
        if default_temperature is not None:
            kwargs["default_temperature"] = default_temperature
        return kwargs

    def _execute_bounded(self, registry: Any, name: str, args: Dict[str, Any]) -> Any:
        """Run ONE inspection tool under the SAME timeout policy as any tool call.

        The number is the loop's own per-tool resolution (`_get_tool_timeout`:
        the settings/env ceiling and the tool's own minimum, for THIS registry
        and these arguments), narrowed by whatever calendar or execution bound
        this reviewer already inherited (`dispatch_deadline_remaining_sec`) —
        no new number, and no floor, exactly as the episode bounds its own
        sends. Past the bound the call is ABANDONED on its private worker: the
        episode never reads its value, so a late return can neither answer the
        reviewer nor source a receipt. A tool that raised its OWN timeout has
        settled, and rides the ordinary tool-error rail instead.
        """
        import concurrent.futures

        from ouroboros.loop_tool_execution import _get_tool_timeout, abandoned_on_timeout, tool_timeout_text
        from ouroboros.model_wait import dispatch_deadline_remaining_sec, future_result

        timeout = float(_get_tool_timeout(registry, name, args))
        remaining = dispatch_deadline_remaining_sec()
        if remaining is not None:
            timeout = min(timeout, max(0.0, remaining))
        def execute():
            from ouroboros.delegate_output import staged_read_delivery_scope
            with staged_read_delivery_scope():
                return registry.execute_result(name, args)

        with abandoned_on_timeout(timeout) as submit:
            from ouroboros.tool_custody import submit_tool_future
            from ouroboros.tool_call_log import new_invocation
            context = getattr(registry, "_ctx", None)
            from ouroboros.model_wait import current_model_wait
            owner = current_model_wait()
            attempt = owner.attempt if owner is not None else getattr(context, "task_attempt", 1)
            invocation = new_invocation("", {}, attempt)
            future, release = submit_tool_future(context, invocation["invocation_id"], name, submit, execute)
            try:
                return future_result(future, timeout)
            except (TimeoutError, concurrent.futures.TimeoutError):
                if future.done():
                    raise
                raise _ToolAbandoned(tool_timeout_text(name, timeout)) from None
            finally:
                release()

    def _execute_inspection_call(
        self, registry: Any, tc: Dict[str, Any], validation_by_id: Dict[str, Any],
        *, round_idx: int,
    ) -> Dict[str, Any]:
        """Run ONE inspection tool call of a round and return its tool MESSAGE.

        Owns the tool-policy half of the episode — wire-validation refusal, the
        inspection allowlist, argument parsing, execution, the disclosed result
        page (measured on the serialized message, with an exact source and a
        continuation for its remainder) and the host-observed receipt — while the caller owns
        the loop, the transcript counter and the messages. Provenance comes from
        CONTROL FLOW, never string-sniffing: a refused call must not read as an
        executed one.
        """
        # ONE clock domain for the receipt's pair: the wall clock. `started_at`
        # is its ISO stamp and `duration_sec` the seconds elapsed on that same
        # clock, so a receipt lines up with the task's own event rows instead
        # of pairing a wall stamp with a monotonic number nothing can compare.
        started_at, started = utc_now_iso(), time.time()
        call_id = str(tc.get("id") or "")
        function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = str(function.get("name") or "")
        raw_args = function.get("arguments")
        args: Optional[Dict[str, Any]] = None
        outcome = "executed"
        extent: Dict[str, Any] = {}
        typed_result = None
        image_registry = None
        source_gap = ""
        source_ref = {}
        room = _EPISODE_TOOL_RESULT_CHAR_CAP
        verdict = validation_by_id.get(call_id)
        if verdict is not None and not getattr(verdict, "allows_execution", True):
            outcome = "refused"
            result = f"⚠️ TOOL_ARG_ERROR: {getattr(verdict, 'error', 'invalid arguments')}"
        elif name not in _INSPECTION_TOOL_NAMES:
            outcome = "refused"
            result = f"⚠️ tool {name[:200]!r} is not available in this read-only inspection episode"
        else:
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except (TypeError, ValueError) as exc:
                outcome = "refused"
                args, result = None, f"⚠️ TOOL_ARG_ERROR: {exc}"
            if isinstance(args, dict):
                # A stamp belongs to ONE call: clear the reader's `last_read_view`
                # BEFORE dispatch, so a call the registry refuses before the tool
                # runs (its pre-dispatch binding: a traversal shape, a blocked
                # root) — or any path that returns without rendering — leaves NO
                # stamp for `_read_extent` to inherit from the previous read.
                if self._inspection_ctx is not None:
                    self._inspection_ctx.last_read_view = None
                try:
                    if name == "compact_context":
                        # NOT bounded: authoring the working view is this
                        # thread's own bookkeeping over the transcript, not a
                        # call that can wedge on anything outside it.
                        from ouroboros.tools.compact_context import _compact_context
                        if not args.get("inspect") and args.get("working_note") is None:
                            result = _FOCUS_GUIDANCE
                        else:
                            unsupported = [key for key in ("schema_names", "review_notes", "review_transfers")
                                           if key in args and args[key] is not None]
                            supported = {key: value for key, value in args.items() if key not in unsupported}
                            result = _compact_context(self._inspection_ctx, **supported)
                            if args.get("inspect"):
                                inspected = json.loads(result)
                                inspected["context"] = self._context_facts()
                                inspected["unsupported_native_options"] = unsupported
                                result = json.dumps(inspected, ensure_ascii=False)
                            elif unsupported:
                                result += "\nNot applied by native review: " + ", ".join(unsupported) + ". The supported authored request is unchanged."
                    else:
                        from ouroboros.tools.tool_result import ToolResult
                        image_registry = staged_image_registry(registry) if name == "view_image" else None
                        returned = self._execute_bounded(image_registry or registry, name, args)
                        typed_result = returned if isinstance(returned, ToolResult) else None
                        result = typed_result.text if typed_result is not None else str(returned)
                except _ToolAbandoned as exc:
                    # The worker may still settle; its value is never read. The
                    # receipt names the gap, the reviewer gets the host's own
                    # timeout text, and the episode credits no further read
                    # extent (below) — a late stamp cannot become coverage.
                    outcome = "error"
                    source_gap = self._source_gap = _SOURCE_GAP_TOOL_ABANDONED
                    self._tool_abandoned = True
                    result = str(exc)
                except Exception as exc:  # tool errors feed the model, not the rail
                    outcome = "error"
                    result = f"⚠️ {type(exc).__name__}: {exc}"
                result_cap = room
                # Disclosed bound with a continuation handle — the reader keeps
                # reading in chunks (read_file supports start_line/max_lines and start_char for long lines), so
                # nothing is silently cut. The marker is budgeted INSIDE the
                # cap, and the cut is measured on the SERIALIZED message (JSON
                # escaping inflates real text) until it fits the room — a raw
                # result under the cap can still overshoot the room on the wire.
                marker = _CONTINUE_READING
                full = result
                source_ref = {}
                oversized = (len(full) > result_cap or len(json.dumps({"role": "tool", "tool_call_id": call_id,
                             "content": full}, ensure_ascii=False)) + 2 > room)
                if oversized and outcome == "executed":
                    source_ref = self._store_source(f"tool-{round_idx}-{self._tool_calls_total}", full, category="tool_results")
                    source_gap = "" if source_ref else self._source_gap
                    marker += " Full result source: " + json.dumps(source_ref, ensure_ascii=False) if source_ref else " Full source unavailable."
                shown = sent = len(full) if len(full) <= result_cap else max(0, result_cap - len(marker) - 64)
                for _ in range(5):
                    result = full if shown >= len(full) else (
                        full[:shown] + f"\n⚠️ RESULT TRUNCATED: showed {shown} of {len(full)} chars" + marker)
                    sent = shown  # the `shown` the SENT result was built from (an exhausted fit loop shrinks `shown` once more after its last build)
                    overshoot = len(json.dumps({"role": "tool", "tool_call_id": call_id, "content": result},
                                               ensure_ascii=False, default=str)) + 2 - max(0, room)  # +2: list separator
                    if overshoot <= 0 or shown == 0:
                        break
                    shown = max(0, min(shown, len(full) - 1) - overshoot)
                if name == "read_file" and outcome == "executed" and not self._tool_abandoned:
                    # Measured on `sent`, never on a `shown` the exhausted fit
                    # loop reduced after the last build: the receipt credits
                    # exactly what the reviewer received.
                    extent = self._read_extent(full, sent)
                    if typed_result is not None and typed_result.meta.get("read_view"):
                        self._pending_staged_reads.append(({"result": full, "tool_result": typed_result},
                                                           {"result": result}, [[0, sent]]))
        # Host-observed evidence (bounded): which artifacts THIS episode
        # actually opened — disclosure, never a claim of full-surface coverage.
        self._tool_calls_total += 1
        receipt: Dict[str, Any] = {"round": round_idx, "tool": name, "delivered": False,
                                   "started_at": started_at, "duration_sec": round(time.time() - started, 3)}
        if isinstance(args, dict):
            for key in ("path", "root", "query", "pattern"):
                if args.get(key):
                    receipt[key] = str(args[key])[:300]
        receipt["result_chars"] = len(result)
        receipt["outcome"] = outcome
        if source_gap:
            receipt["source_gap"] = source_gap
        if source_ref:
            receipt['result_source_ref'] = source_ref
        receipt.update(extent)
        if name == "view_image":
            result, receipt["image_attachment"] = self._images.accept(
                call_id, (args or {}).get("path", ""), result,
                image_registry if outcome == "executed" else None, round_idx=round_idx)
            receipt["result_chars"] = len(result)
        self._tool_receipts.append(receipt)
        return {"role": "tool", "tool_call_id": call_id, "content": result,
                **({"_review_image_id": receipt["image_attachment"]["image_id"]} if name == "view_image" else {})}

    def _read_extent(self, full: str, shown: int) -> Dict[str, Any]:
        """The extent an executed ``read_file`` actually DELIVERED, as bounded
        facts on the receipt: ``start_line``/``end_line`` (the COMPLETE lines
        the reviewer received; an empty delivery is an empty range with
        ``end_line < start_line``), ``total_lines`` of the file, ``eof``,
        ``opened_path`` — the root-relative path the reader actually opened —
        and ``opened_root``, the normalized root it used (the receipt's ``path``
        and ``root`` stay the model's spelling; the registry normalizes absolute
        in-repo, whitespace-padded and redundant-root path spellings and padded
        root spellings before the handler runs, so coverage folds on the
        opened path and root, never on the spelling).

        Every fact comes from the reader's own stamp (``ctx.last_read_view``,
        written by the renderer AFTER its sub-line cursor cut: ``first_line`` is
        the first complete line, ``body_start`` where the body begins in the
        returned text, ``line_ends`` the end offsets of the body's complete
        lines on the renderer's own line definition) — nothing is parsed back
        from the header and no newline is recounted here. The stamped offsets
        hold on the returned text because every host note (``_annotate_reread``'s
        re-read hint, the registry's route/safety notes via
        ``_compose_execute_result``) TRAILS the rendered view — a prepended note
        would shift ``body_start``. The stamp's binding
        to THIS call is structural, not a comparison: the reader resets it on
        entry, the caller clears it before every dispatch, the context is
        instance-local, and the caller reads the stamp only for a call that
        RETURNED inside its bound — so a call refused before the tool ran, one
        that returned without rendering, or one the episode ABANDONED (its
        outcome is an error and a loose worker retires the episode's read
        attribution for good) finds no stamp and records no extent. The
        ``last_read_view`` WRITER-SET invariant
        backs this: exactly three writers exist — the reader's entry reset and
        its stamp in ``tools/core_file_tools.py`` and this episode's clear-before-dispatch
        — pinned by a static test, so a fourth writer cannot forge coverage
        silently. When this episode's result bound cut the body, only the
        complete lines whose end lies inside the delivered prefix count.
        Exact character ranges additionally require the opened bytes' revision
        and the reader's unmasked, normalized text extent. They include partial
        lines and exclude the header and trailing host notes. Missing or masked
        source facts retain only the legacy line evidence, never exact-source
        coverage. These per-read facts do not make the bounded receipt summary
        a complete history of the review operation."""
        view = getattr(self._inspection_ctx, "last_read_view", None)
        keys = ("first_line", "end_line", "total_lines", "body_start")
        ends = view.get("line_ends") if isinstance(view, dict) else None
        if (not isinstance(view, dict) or not all(isinstance(view.get(k), int) for k in keys)
                or not isinstance(ends, (list, tuple)) or not all(isinstance(e, int) for e in ends)
                or not isinstance(view.get("opened_path"), str) or not isinstance(view.get("opened_root"), str)):
            return {}
        start, end, total, body_start = (int(view[k]) for k in keys)
        if shown < len(full):
            delivered = bisect.bisect_right(ends, shown - body_start) if shown > body_start else 0
            end = min(end, start + delivered - 1)
        result = {"start_line": start, "end_line": end, "total_lines": total, "eof": end >= start and end >= total,
                  "opened_path": view["opened_path"][:300], "opened_root": view["opened_root"][:64]}
        source_ints = ("source_bytes", "complete_chars", "source_start_char", "source_end_char", "body_chars")
        source_hashes = ("source_revision", "complete_sha256")
        if (not all(type(view.get(k)) is int and view[k] >= 0 for k in source_ints)
                or not all(isinstance(view.get(k), str) and len(view[k]) == 64
                           and all(c in "0123456789abcdef" for c in view[k]) for k in source_hashes)
                or view.get("source_masked") is not False
                or view.get("range_basis") != "unicode_text_universal_newlines"
                or type(shown) is not int or not 0 <= shown <= len(full)
                or type(view["body_start"]) is not int or not 0 <= body_start <= len(full)
                or not view["opened_path"] or not view["opened_root"]
                or not 0 <= view["source_start_char"] <= view["source_end_char"] <= view["complete_chars"]
                or view["source_end_char"] - view["source_start_char"] != view["body_chars"]):
            return result
        from ouroboros.tools.core_file_tools import delivered_source_prefix, display_source_segments

        if display_source_segments(view, full) is None:  # not this read's numbered display
            return result
        delivered = delivered_source_prefix(view, full, shown)
        delivered_chars = len(delivered)
        result.update({k: view[k] for k in (*source_hashes, "source_bytes", "complete_chars", "range_basis")})
        result.update({"source_start_char": view["source_start_char"],
                       "source_end_char": view["source_start_char"] + delivered_chars,
                       "text_chars": delivered_chars, "text_sha256": hashlib.sha256(delivered.encode("utf-8")).hexdigest(),
                       "source_masked": False,
                       # Exact source identity cannot fold two long paths into the same shortened name.
                       "opened_path": view["opened_path"], "opened_root": view["opened_root"]})
        return result

    @staticmethod
    def _terminal_round_fact(messages: List[Dict[str, Any]]) -> str:
        """The terminal round (last assistant envelope + its tool results) as
        ONE bounded, structurally valid JSON document: every field is bounded
        BEFORE serialization with the strict in-budget marker, and only the
        last few tool results are kept (the omitted count is disclosed) — a
        serialized document cut mid-syntax would be no record at all."""
        from ouroboros.observability import redact_projection
        from ouroboros.utils import truncate_within_limit

        def _unfold(value: Any, depth: int = 0, expansions: int = 0) -> Any:
            # JSON carried inside strings — at any nesting — is parsed so the
            # structural key masking sees every secret-named field. Two
            # SEPARATE guards bound the work: container levels (`depth`) and
            # nested string expansions along one path (`expansions`); one
            # shared counter let ordinary traversal spend the expansion budget
            # before the secret-keyed object was reached. A JSON-looking string
            # that cannot be expanded (a guard reached, oversized, too deep to
            # parse) is MASKED, never kept in clear: the text pass cannot see
            # structure it never parsed.
            if isinstance(value, str):
                if value.lstrip()[:1] not in "{[":
                    return value
                if expansions >= 8 or len(value) > 200_000:
                    return f"[unexpanded JSON masked: {len(value)} chars]"
                try:
                    parsed = json.loads(value)
                except (TypeError, ValueError):
                    return value  # JSON-looking but not JSON: plain text
                except RecursionError:
                    return f"[unexpanded JSON masked: {len(value)} chars]"
                if isinstance(parsed, (dict, list)):
                    return {"__json__": _unfold(parsed, depth + 1, expansions + 1)}
                return value
            if isinstance(value, (dict, list)) and depth >= 64:
                return "[container too deep: masked]"
            if isinstance(value, dict):
                return {str(k): _unfold(v, depth + 1, expansions) for k, v in value.items()}
            if isinstance(value, list):
                return [_unfold(v, depth + 1, expansions) for v in value]
            return value

        def _fold(value: Any) -> Any:
            if isinstance(value, dict):
                if set(value) == {"__json__"}:
                    return json.dumps(_fold(value["__json__"]), ensure_ascii=False, default=str)
                return {k: _fold(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_fold(v) for v in value]
            return value

        def _redacted(value: Any) -> Any:
            # Structural key masking works on OBJECTS: unfold every JSON string
            # (nested ones included), mask, fold back; plain text and JSON
            # scalars are redacted as text.
            return _fold(redact_projection(_unfold(value)).value)

        tail: List[Dict[str, Any]] = []
        for msg_item in reversed(messages):
            tail.insert(0, msg_item)
            if msg_item.get("role") == "assistant":
                break
        assistant = tail[0]
        # Only the round's own tool results belong to the record; a trailing
        # host view receipt (a user message) is disclosed, never relabelled.
        results = [m for m in tail[1:] if m.get("role") == "tool"]
        trailing_notice = any(m.get("role") == "user" for m in tail[1:])
        calls = []
        raw_calls = assistant.get("tool_calls")
        # Total over provider output: a non-list container is kept as ONE
        # bounded redacted value, never iterated.
        for tc in (raw_calls if isinstance(raw_calls, list) else ([] if raw_calls is None else [raw_calls])):
            calls.append(_redacted(tc))
        bounded: List[Dict[str, Any]] = [{
            "role": "assistant",
            "content": truncate_within_limit(str(_redacted(str(assistant.get("content") or "")) or ""), 1_200),
            "tool_calls": truncate_within_limit(json.dumps(calls, ensure_ascii=False, default=str), 1_200),
        }]
        kept = results[-4:]
        for item in kept:
            bounded.append({
                "role": "tool",
                "tool_call_id": truncate_within_limit(str(_redacted(str(item.get("tool_call_id") or "")) or ""), 200),
                "content": truncate_within_limit(str(_redacted(str(item.get("content") or "")) or ""), 1_000),
            })
        doc = {
            "messages": bounded, "omitted_tool_results": len(results) - len(kept),
            "trailing_host_notice": trailing_notice,
        }
        # Defense in depth: the whole bounded document passes the projection
        # redactor once more before it becomes a durable custody row.
        return json.dumps(redact_projection(doc).value, ensure_ascii=False, default=str)

    def failure_custody(self) -> Dict[str, Any]:
        """The proven facts of a refused or errored episode for the error actor:
        delivery, rounds, receipts, context observations, end reason and the paid
        ledger facts of the rounds that DID run — so a failed native execution
        stays visible on the public wire instead of vanishing. Empty until the
        episode started."""
        return dict(self._episode_usage)

    def _emit_episode_fact(self, episode: Dict[str, Any]) -> str:
        """One typed custody row per episode end (rounds, context facts and cause).
        Returns the row's fate — ``written``, ``failed``
        or ``no_custody_root`` — so the usage can say whether the durable
        trace exists. Never raises."""
        drive = self.assignment.custody_root
        if not drive:
            return "no_custody_root"
        try:
            from ouroboros.delegate_custody import emit
        except Exception:  # telemetry never masks the episode's own outcome
            return "failed"
        written = emit(drive, "review_native_episode", {
            "surface": str(self.assignment.request.surface or ""),
            "task_id": str(self.assignment.request.task_id or ""),
            "slot_id": str(self.assignment.slot.slot_id or ""),
            "model": str(self.assignment.slot.model or ""),
            **episode,
        })
        return "written" if written else "failed"

    def _verdict_result(self, force_extraction: bool = False) -> ReviewAttemptResult:
        text = self._raw_transcript or ""
        if text.strip() and self.answer_observer:
            self.answer_observer()
        canonical, method, extraction_usage = canonicalize_session_verdict(
            text,
            conformance_passed=False,  # no structured-output channel on this route
            contract=self._output_contract(),
            llm=self.llm,
            deadline_at=getattr(self.assignment.request, "deadline_at", "") or "",
            transport_timeout_sec=getattr(self.assignment.slot, "transport_timeout_sec", None),
            shape=request_output_shape(self.assignment.request),
        )
        usage = dict(self._episode_usage)
        deltas: List[Dict[str, Any]] = list(self._episode_deltas)
        if method == "light_model_extraction":
            usage["extraction"] = extraction_usage
            deltas.append({
                "kind": "capability_delta",
                "requested": "contract-shaped verdict from the episode",
                "effective": "light-model extraction over the collected answer",
                "reason": "extraction_instead_of_contract",
            })
        elif method == "extraction_incomplete":
            deltas.append({
                "kind": "capability_delta",
                "requested": "contract-shaped verdict from the episode",
                "effective": (
                    f"no verdict: answer ({len(text)} chars) exceeds the "
                    "single-send extraction bound"
                ),
                "reason": "extraction_incomplete_transcript_exceeds_bound",
            })
        usage["verdict_method"] = method
        usage["verdict_provenance"] = {
            "raw_transcript_chars": len(text),
            "raw_transcript_sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(),
            "canonical_chars": len(canonical),
            "canonical_sha256": hashlib.sha256(canonical.encode("utf-8", "replace")).hexdigest(),
            "output_conformance": "",
            "conformance_trusted": False,
            "verdict_method": method,
            "raw_transcript_carrier": "message.native_transcript (durable response_ref)",
        }
        if deltas:
            usage["capability_delta"] = deltas
        message = {
            "content": canonical,
            "native_transcript": text,
            "verdict_method": method,
        }
        if usage.get("native_incomplete"):
            # The partial-product fact travels WITH the product, not only in
            # usage: a consumer that reads the text alone must still see it.
            message["native_incomplete"] = usage["native_incomplete"]
        return ReviewAttemptResult(message=message, usage=usage, raw_text=canonical)
