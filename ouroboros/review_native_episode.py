"""The native tool-round review delivery (configured-subagent api rows).

A reviewer row bound to an ``api_model`` configured subagent RETRIEVES the
subject through bounded read-only inspection rounds instead of receiving the
assembled packet. Not a third public route kind — ``api_chat`` stays the wire
vocabulary; the slot's actor binding selects this executor at the one transport
seam (``review_execution._review_route_executor``).

ONE episode is ONE logical review attempt: ``LLMClient.chat(tools=…)`` calls
against a fresh, instance-local inspection-only ``ToolRegistry`` until the
reviewer answers. There is NO round cap (BIBLE P13: the floor is hardcoded,
never the ceiling): the episode's bounds are the transcript bound derived from
the reviewer's own context window (never above the owner ceiling, unless a
surface's declared mandatory reading lifts it — a floor the window caps and
the host discloses typed when it cannot be met), the owner deadline and the
paid ledger. The host announces the bound once at the landing
fraction so the reviewer can finish; exhaustion is a typed refusal for verdict
shapes and a disclosed INCOMPLETE product for the report shape — never
mid-episode compaction or resume. Every provider call is its own ledger row;
the coordinator's second actor attempt repairs FORMAT locally, exactly like
the session executor.
"""

from __future__ import annotations

import bisect
import contextlib
import time
import hashlib
import json
import logging
import math
from typing import Any, Dict, List, Optional

from ouroboros.config import get_finalization_grace_sec
from ouroboros.deadline_utils import owner_deadline_exhausted, review_transport_timeout
from ouroboros.review_dispatch import bind_api_review_paid_stamp, invoke_review_paid_stamp
from ouroboros.review_verdict_extraction import canonicalize_session_verdict
from ouroboros.triad_review import default_output_contract, review_output_shape
from ouroboros.usage_accounting import (
    POSITIVE_PHYSICAL_ATTEMPT_STATES,
    BudgetExceeded,
    physical_attempt_limit,
)

from ouroboros.review_execution import (
    ReviewAssignment,
    ReviewAttemptResult,
    ReviewRouteKind,
    ReviewRouteUnavailable,
    ReviewSlotExecutor,
    _deadline_exhausted_error,
)

log = logging.getLogger("review_native_episode")


def review_native_max_transcript_chars() -> int:
    """Owner CEILING on the episode transcript (chars). The effective bound is
    the reviewer window's calibrated capacity (`review_native_transcript_bound`),
    never above this number — except for a surface-declared mandatory reading,
    a floor the window (not this ceiling) caps (`native_mandatory_read_bound`)."""
    from ouroboros.config import _clamped_number_setting

    return _clamped_number_setting(
        "OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", low=50_000, high=2_000_000, cast=int)


# The landing fraction of the transcript bound: the host announces the bound
# ONCE when the transcript crosses it, so the reviewer answers from what it has
# read instead of discovering the wall on the send that would have exceeded it.
NATIVE_LANDING_FRACTION = 0.8

# Chars per estimated token — the `utils.estimate_tokens` heuristic inverted,
# so the transcript counter (chars) and the reviewer window (tokens) meet on
# the SAME scale the review packet sizer uses.
_CHARS_PER_ESTIMATED_TOKEN = 4

# The typed disclosure when a surface's declared mandatory reading cannot land
# before the landing notice even at the bound the reviewer's window allows: the
# surface's prompt and the episode facts both carry it, so a full-read
# instruction the episode cannot honour is never a silent contradiction.
# Vocabulary sibling of the episode's `native_transcript_cap_exceeded` end.
NATIVE_MANDATORY_READ_EXCEEDS_BOUND = "native_mandatory_read_exceeds_bound"


def review_native_transcript_bound(
    model_id: str, *, output_reserve: int, use_local: Optional[bool] = None,
    mandatory_read_chars: int = 0,
) -> int:
    """The episode's SEND bound in chars, derived from the reviewer's window.

    The same density-calibrated input capacity the packet route sizes its pack
    with (`calibrated_input_token_limit`: window minus the output reserve,
    divided by the freshest exact-model token density, never above the
    absolute-margin form), converted back to chars and capped by the owner
    ceiling `OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS`. A 1M reviewer
    therefore lands on the ceiling; a 200K route gets a bound its own window
    can carry instead of a number written for a different model — the
    previous fixed cap either starved a large window or overflowed a small one.

    ``mandatory_read_chars`` is the surface's declared mandatory reading (the
    task text it hands over plus the wire size of the documents it requires
    read in full): a FLOOR (P13) that lifts the bound past the owner ceiling to
    `native_mandatory_read_bound` — never past what the window itself carries;
    `native_mandatory_read_disclosure` names the shortfall when even that is
    not enough. The ceiling keeps bounding DISCRETIONARY reading.
    """
    from ouroboros.reviewer_window import reviewer_context_window, window_scaled_reserves
    from ouroboros.tools.review_helpers import calibrated_input_token_limit

    ceiling = review_native_max_transcript_chars()
    window = int(reviewer_context_window(str(model_id or ""), use_local=use_local))
    reserve, margin = window_scaled_reserves(
        window, output_reserve=int(output_reserve or 0), tokenizer_margin=window // 8)
    capacity = _CHARS_PER_ESTIMATED_TOKEN * max(0, int(calibrated_input_token_limit(
        str(model_id or ""), context_window=window, output_reserve=reserve,
        tokenizer_margin=margin, budget_cap=window)))
    bound = min(ceiling, capacity)
    if int(mandatory_read_chars or 0) > 0:
        bound = max(bound, min(capacity, native_mandatory_read_bound(mandatory_read_chars)))
    return max(0, bound)


def native_mandatory_read_bound(mandatory_read_chars: int) -> int:
    """The bound at which a declared mandatory reading lands one full result
    cap BEFORE the landing notice: the host's own first-send additions (the
    instructions, the tool schemas, the surface's budget section) and the
    per-read envelopes, which the declaration cannot know, ride in that room."""
    return math.ceil(
        (int(mandatory_read_chars) + _EPISODE_TOOL_RESULT_CHAR_CAP) / NATIVE_LANDING_FRACTION)


def native_mandatory_read_disclosure(bound: int, mandatory_read_chars: int) -> str:
    """``""`` when the declared mandatory reading lands before the landing
    notice under ``bound``, else the typed shortfall code."""
    if int(mandatory_read_chars or 0) <= 0 or int(bound) >= native_mandatory_read_bound(mandatory_read_chars):
        return ""
    return NATIVE_MANDATORY_READ_EXCEEDS_BOUND


def native_landing_at(bound: int) -> int:
    """Where the host posts the landing notice: the landing fraction of the
    bound, never nearer to it than the landing reserve."""
    return min(int(int(bound) * NATIVE_LANDING_FRACTION), max(0, int(bound) - _LANDING_RESERVE_CHARS))


def native_mandatory_read_chars(request: Any) -> int:
    """The surface's declared mandatory reading on the request policy (chars)."""
    return int((getattr(request, "policy", None) or {}).get("native_mandatory_read_chars") or 0)


def native_episode_transcript_bound(request: Any, slot: Any) -> int:
    """THE bound of one episode from its assignment — the one computation the
    episode applies, which a surface may preview before dispatch (the advisory
    names it in its prompt's MANDATORY READ budget)."""
    return review_native_transcript_bound(
        slot.model, output_reserve=int(request.max_tokens or slot.max_tokens),
        use_local=bool(slot.use_local), mandatory_read_chars=native_mandatory_read_chars(request))


def native_mandatory_read_facts(request: Any, bound: int) -> Dict[str, Any]:
    """The episode facts of a surface-declared mandatory reading: the declared
    chars and, when ``bound`` cannot hold it, the typed shortfall code; ``{}``
    when nothing was declared."""
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
    a packet/session actor on the P3/acceptance surfaces; NO local send count
    for a native tool-round slot (its bounds are the transcript bound, the
    owner deadline and the paid ledger — a count would be a round cap by
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
    "read_file", "list_files", "search_code", "query_code",
    "vcs_status", "vcs_diff",
)

_NATIVE_REVIEW_INSTRUCTIONS = (
    "You are an independent Ouroboros reviewer running a bounded read-only "
    "inspection episode. Retrieve the evidence yourself with the tools you are "
    "given inside the repository root — read_file, list_files, search_code, "
    "query_code, vcs_status, vcs_diff; no other tools exist here. You cannot "
    "modify anything and have no shell. Read LARGE files in bounded chunks "
    "(read_file supports start_line/max_lines and start_char for long lines) instead of requesting a whole large "
    "document at once: the episode has a hard transcript budget sized to your "
    "own context window, and an oversized read spends it. There is no round "
    "limit. Your host will tell you once when the budget is nearly spent; "
    "answer from what you have read at that point. Read what the checklist "
    "requires, then answer. Your FINAL message must contain no tool calls and "
    "must follow the output contract in the task EXACTLY; your host parses it "
    "structurally, and prose around the verdict is a non-response."
)

# The host's budget fact (same class as the main loop's [ROUND_LIMIT] notice):
# a typed, once-only user message — never a silent cut of the transcript.
_LANDING_NOTICE = (
    "[EPISODE_BUDGET] Your inspection transcript is at {pct}% of its bound "
    "({used} of {bound} chars); no further reading fits. Your NEXT message "
    "must be the final deliverable in the output contract, with no tool "
    "calls. Mark anything you could not verify as unverified — an honest "
    "bounded answer is the expected outcome here, not a failure."
)

# Per-tool-result bound inside the episode: one greedy full read of a giant
# artifact must not consume the whole transcript budget in a single round.
# Disclosed truncation with a line/character continuation cursor is honest —
# unlike compaction, nothing unseen is summarized into the record. The cap is
# ALSO clamped to the room left below the transcript bound minus the landing
# reserve, so no single read can jump over the landing notice and the bound.
_EPISODE_TOOL_RESULT_CHAR_CAP = 120_000

# Room kept below the bound for the landing notice itself AND the envelopes of
# the withheld calls of the round that crossed the landing line (every tool
# call must be answered, so an empty result still costs its message envelope):
# the notice must always fit under the send bound it announces.
_LANDING_RESERVE_CHARS = 2_048

# Below this much room a tool result could carry nothing but its truncation
# marker: the call is WITHHELD (not executed) instead of read-and-discarded.
_RESULT_ROOM_FLOOR_CHARS = 256


def _wire_size(messages: List[Dict[str, Any]], schemas: List[Dict[str, Any]]) -> int:
    """The ONE measure of a send: the serialized messages list plus the tool
    schemas that ride every call. It is RECOMPUTED from the real list after
    every append and before every bound or room decision — no incremental
    charge (a raw text here, a missing list separator there) can drift from
    what the next send actually carries."""
    return (len(json.dumps(messages, ensure_ascii=False, default=str))
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


def native_first_send_messages(prompt: str) -> List[Dict[str, Any]]:
    """The first send's message objects (system instructions and the task)."""
    return [
        {"role": "system", "content": _NATIVE_REVIEW_INSTRUCTIONS},
        {"role": "user", "content": prompt},
    ]


def inspection_registry(root: str, drive_root: Any, task_id: str = "") -> tuple[Any, Any, List[Dict[str, Any]]]:
    """A fresh registry pinned to ``root``, read-only, with its context and the
    inspection tool schemas that ride every send of an episode.

    Reuses the existing capability machinery wholesale instead of a new
    allowlist mechanism: the ``local_readonly_subagent`` constraint gives
    the read-only operation/root matrix, ``disabled_tools`` trims its
    broader name allowlist down to the inspection six, and the resource
    contract keeps extension/MCP discovery off. ``registry._ctx`` is
    per-instance, so the worker's own registry/context is never touched.
    ``drive_root`` is the data plane the inspection tools may read: an
    empty scratch directory by default, or the surface's opt-in
    ``policy["native_data_root"]`` (task acceptance reads task results and
    receipts there; deep self-review exposes the runtime root while its
    memory whitelist arrives inline) — the read-only constraint applies to
    it exactly as to the repository.
    """
    import pathlib as _pathlib

    from ouroboros.tool_capabilities import LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    registry = ToolRegistry(repo_dir=_pathlib.Path(root), drive_root=_pathlib.Path(drive_root))
    ctx = ToolContext(
        repo_dir=_pathlib.Path(root),
        drive_root=_pathlib.Path(drive_root),
        task_id=str(task_id or "") or None,
        task_constraint={"mode": "local_readonly_subagent"},
        task_contract={
            "allowed_resources": {"network": False, "web": False},
            "disabled_tools": sorted(
                set(LOCAL_READONLY_SUBAGENT_TOOL_NAMES) - set(_INSPECTION_TOOL_NAMES)
            ),
        },
    )
    registry.set_context(ctx)
    schemas = [schema for schema in (registry.get_schema_by_name(name) for name in _INSPECTION_TOOL_NAMES) if schema]
    if not schemas:
        raise ReviewRouteUnavailable(
            "no inspection tool schemas are projectable for the native "
            "tool-round episode", code="native_inspection_unavailable")
    return registry, ctx, schemas


def native_first_send_chars(
    root: str, *, surface: str, role_hint: str, slot_id: str, session_task: str,
    output_contract: str, task_id: str = "",
) -> int:
    """The wire size of a native episode's FIRST send — the prompt its first
    ``reserve_attempt`` prices (every later round reserves itself, on the
    ledger, against the transcript it has grown to). Built from the same
    work-order, message objects, schemas and measure ``_open_episode`` uses,
    so the commit gate's wave admission prices a native seat the way the
    ledger will; the registry is built read-only against ``root`` and
    discarded."""
    _registry, _ctx, schemas = inspection_registry(root, root, task_id)
    prompt = native_episode_prompt(surface, role_hint, str(session_task or "").strip(), output_contract, slot_id)
    return _wire_size(native_first_send_messages(prompt), schemas)

class NativeToolRoundReviewExecutor(ReviewSlotExecutor):
    """Bounded native inspection episode for a configured-subagent api row.

    ONE episode is ONE logical review attempt: ``LLMClient.chat(tools=…)``
    calls against a fresh, instance-local inspection-only ``ToolRegistry``
    until the reviewer answers or a bound lands — the window-derived transcript
    bound, the owner deadline, the paid ledger; never a round count. Every
    provider call is its own ledger row (the ambient usage scope attributes
    them); the coordinator's second actor attempt repairs FORMAT locally over
    the collected answer, exactly like the session executor — there is no
    mid-episode resume, transcript compaction, or per-round durable ledger.
    Exhaustion is a typed refusal for verdict shapes; the report shape delivers
    what was collected, marked INCOMPLETE.
    """

    route = ReviewRouteKind.API_CHAT

    def __init__(self, assignment: ReviewAssignment, *, llm: Any = None):
        super().__init__(assignment, llm=llm)
        self._episode_prompt: Optional[str] = None
        self._raw_transcript: Optional[str] = None
        self._episode_usage: Dict[str, Any] = {}
        self._tool_receipts: List[Dict[str, Any]] = []
        self._inspection_ctx: Any = None  # the registry's context: the reader stamps `last_read_view` on it
        self._tool_calls_total = 0
        self._rounds_used = 0
        self._episode_deltas: List[Dict[str, Any]] = []
        self._settled_failure: Optional[BaseException] = None

    # -- prompt (route-owned; never the api pack) ------------------------------

    def _output_contract(self) -> str:
        contract = str((self.assignment.request.policy or {}).get("output_contract") or "")
        return contract or default_output_contract(review_output_shape(self.assignment.request.surface))

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
        ``policy["native_data_root"]`` (task acceptance reads task results and
        receipts there; deep self-review exposes the runtime root while its
        memory whitelist arrives inline) — the read-only constraint applies to
        it exactly as to the repository.
        """
        registry, self._inspection_ctx, schemas = inspection_registry(
            root, drive_root, str(self.assignment.request.task_id or ""))
        return registry, schemas

    def _run_episode(self) -> None:
        import shutil
        import tempfile

        from ouroboros.llm import add_usage
        from ouroboros.openai_chat_dispatch import (
            custom_validation_by_call_id,
            pop_custom_validation_receipts,
        )

        request, slot = self.assignment.request, self.assignment.slot
        root = str(request.session_root or "").strip()
        if not root:
            raise ReviewRouteUnavailable(
                "native tool-round slot has no session root: the surface must name "
                "the repository root the reviewer episode runs in",
                code="session_root_missing")
        chat = getattr(self.llm, "chat", None)
        if not callable(chat):
            raise ReviewRouteUnavailable(
                "native tool-round episode needs a synchronous chat transport",
                code="api_chat_unavailable")
        deadline_at = str(getattr(request, "deadline_at", "") or "")
        max_tokens = int(request.max_tokens or slot.max_tokens)
        transcript_cap = native_episode_transcript_bound(request, slot)
        mandatory_read_facts = native_mandatory_read_facts(request, transcript_cap)
        landing_at = native_landing_at(transcript_cap)
        shape = review_output_shape(request.surface)
        # The data plane is opt-in per surface (policy["native_data_root"]):
        # the default is an empty scratch directory so a repository review
        # cannot read the host's state; a surface that needs task results or
        # memory names the real root, which is the caller's and is never removed.
        data_root = str((request.policy or {}).get("native_data_root") or "").strip()
        scratch = tempfile.mkdtemp(prefix="ouro-native-review-")
        registry = None
        total_usage: Dict[str, Any] = {}
        final_answer: Optional[str] = None
        last_content = ""  # the reviewer's latest prose — the product of an exhausted report episode
        landed = False
        landing_sent = False  # the notice was posted AND a send carried it
        end_reason = "not_started"  # the true reason is set by whichever end the episode takes
        round_idx = 0
        transcript_chars = last_send_chars = refused_chars = 0  # live list size / last send / refused next send
        episode: Dict[str, Any] = {}
        messages: List[Dict[str, Any]] = []
        try:
            end_reason = "registry_unavailable"
            registry, schemas, messages, transcript_chars = self._open_episode(root, data_root or scratch)
            end_reason = "transcript_bound"
            if transcript_chars >= landing_at:
                # FLOOR: a bound that lands before the first send leaves no
                # room to read anything — a review with zero reads is not a
                # review, and the landing notice must never be the first
                # thing the reviewer hears.
                end_reason = "bound_below_first_send"
                raise ReviewRouteUnavailable(
                    f"native review episode bound ({transcript_cap} chars) leaves no "
                    f"room to read: the first send alone carries {transcript_chars} "
                    "chars; the episode fails closed", code="native_bound_below_first_send")
            logical_deadline = getattr(self, "_logical_deadline_monotonic", None)
            while True:
                # The bound is a SEND bound, enforced BEFORE every provider call
                # (the transcript IS the next send's context); a materialized
                # overflow is resolved FIRST, so its typed end and refused-size
                # fact are never preempted by a clock expiring in the same instant.
                # A final content-only answer past the number is accepted: no
                # further send exists for it to poison.
                if refused_chars or transcript_chars > transcript_cap:
                    refused_chars = refused_chars or transcript_chars  # the next send this bound refused
                    break
                if not landed and transcript_chars >= landing_at:
                    # Once: the host's budget fact, so the reviewer lands on
                    # the next send instead of walking into the bound.
                    landed = True
                    notice = _LANDING_NOTICE.format(
                        pct=int(100 * transcript_chars / max(1, transcript_cap)),
                        used=transcript_chars, bound=transcript_cap)
                    messages.append({"role": "user", "content": notice})
                    transcript_chars = _wire_size(messages, schemas)
                    if transcript_chars > transcript_cap:
                        refused_chars = transcript_chars  # even the notice would not fit: the bound has landed
                        break
                # Two clocks bound an episode that could still send (its landing
                # notice is materialized above, so a landed bound is never misread
                # as a clock end): the owner's deadline and the slot's logical
                # window — past either, a paid round buys an unusable answer.
                if owner_deadline_exhausted(
                    deadline_at=deadline_at, reserve_sec=get_finalization_grace_sec(),
                ) or (logical_deadline is not None and time.monotonic() >= float(logical_deadline)):
                    end_reason = "deadline_exhausted"
                    if shape == "report" and last_content:
                        break  # a report keeps its draft (marked incomplete below)
                    raise _deadline_exhausted_error("owner deadline exhausted mid native review episode")
                round_idx += 1
                chat_kwargs = self._chat_kwargs(messages, schemas, max_tokens)
                transport = review_transport_timeout(
                    slot.model, getattr(slot, "transport_timeout_sec", None), deadline_at,
                )
                if logical_deadline is not None:
                    # Recomputed immediately before dispatch: a window that
                    # expired since the round's admission check takes the
                    # deadline path (a report keeps its draft), and a positive
                    # remainder bounds the send with NO floor.
                    remaining = float(logical_deadline) - time.monotonic()
                    if remaining <= 0:
                        end_reason = "deadline_exhausted"
                        if shape == "report" and last_content:
                            break
                        raise _deadline_exhausted_error("slot logical window exhausted before the native review send")
                    transport = remaining if transport is None else min(float(transport), remaining)
                if transport is not None:
                    chat_kwargs["timeout"] = transport
                with bind_api_review_paid_stamp(self.assignment.dispatch_stamp):
                    try:
                        msg, usage = chat(**chat_kwargs)
                    except BaseException as exc:
                        # The paid ledger refusing the NEXT send is the money
                        # floor landing, not a transport fault: name it.
                        end_reason = "budget_exhausted" if isinstance(exc, BudgetExceeded) else "transport_error"
                        capture = getattr(exc, "physical_attempt_capture", None)
                        if str(getattr(capture, "state", "") or "") in POSITIVE_PHYSICAL_ATTEMPT_STATES:
                            # A send that was physically dispatched IS a round
                            # of this episode, even when its response never
                            # came back: its receipt keys and custody must not
                            # read as a zero-send refusal.
                            self._rounds_used, last_send_chars = round_idx, transcript_chars  # a dispatched send IS the last one
                            landing_sent = landing_sent or landed  # the dispatched send carried the notice
                            invoke_review_paid_stamp(self.assignment.dispatch_stamp)
                        self._observe_failed_send(exc)
                        if isinstance(exc, BudgetExceeded) and shape == "report" and last_content:
                            break  # nothing was sent; a report keeps its draft
                        raise
                self._rounds_used, last_send_chars = round_idx, transcript_chars  # a returned send is the last physical send
                landing_sent = landing_sent or landed  # a returned send carried the notice
                raw_calls = msg.get("tool_calls") if isinstance(msg, dict) else None
                # absent = no calls; a list = calls; ANY other value (falsy too) = one malformed entry
                tool_calls = [] if raw_calls is None else (raw_calls if isinstance(raw_calls, list) else [raw_calls])
                usage = dict(usage or {})
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
                for _fact in ("resolved_model", "provider"):
                    if usage.get(_fact):
                        total_usage[_fact] = usage[_fact]
                content = str(msg.get("content") or "") if isinstance(msg, dict) else ""
                if content:
                    last_content = content
                # The envelope joins the record BEFORE any terminal branch, so
                # the terminal-round fact describes the decision-ending output
                # itself — an empty or malformed final round included. The WHOLE
                # dict joins (a reasoning-echo lane's ``reasoning_content`` too), so
                # the wire-size recompute below charges the replayed thinking tail.
                assistant = dict(msg) if isinstance(msg, dict) else {"content": content}
                assistant.setdefault("role", "assistant")
                messages.append(assistant)
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
                # The size is recomputed from the real list after every append
                # (the whole assistant envelope rides every later send).
                transcript_chars = _wire_size(messages, schemas)
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue  # a non-dict tool_call is malformed provider output, not a crash
                    # The per-result room is what is left below the bound minus
                    # the landing reserve: one read can never jump over the
                    # landing notice and the bound in a single round.
                    # The WHOLE tool message rides the next send — role and the
                    # provider's call id included — so an empty withheld result
                    # still costs its envelope; the call executor fits it to the
                    # room on the SERIALIZED size (JSON escaping inflates real
                    # text) and the recompute below decides the bound.
                    tool_message = self._execute_inspection_call(
                        registry, tc, validation_by_id, round_idx=round_idx,
                        room=transcript_cap - _LANDING_RESERVE_CHARS - transcript_chars,
                    )
                    with_result = _wire_size(messages + [tool_message], schemas)
                    if with_result > transcript_cap:
                        # Even the mandatory envelope (the provider's exact call
                        # id must be echoed) no longer fits under the bound: the
                        # round cannot be answered within it, so the episode
                        # ends HERE, typed — no over-bound send is ever made.
                        end_reason = "transcript_bound"
                        refused_chars = with_result  # disclosed as its own fact, never a fake counter
                        break
                    messages.append(tool_message)
                    transcript_chars = with_result
            if shape == "report" and not final_answer and last_content:
                # A report is a product, not a verdict: the collected draft is
                # delivered marked INCOMPLETE rather than discarded (the bound
                # landed, the round made no progress, or the final round came
                # back empty) — the consumer discloses it; nothing unseen is
                # summarized into it. Decided BEFORE the custody row is written
                # so the durable fact knows the product is partial.
                final_answer = last_content
                episode["native_incomplete"] = end_reason
        finally:
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
                # tool results that led to a bound, deadline or transport end
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
                "native_transcript_chars": last_send_chars,  # the wire size of the LAST physical send
                "native_transcript_bound": transcript_cap,
                **mandatory_read_facts,  # a declared mandatory reading and its typed shortfall
                **({"native_transcript_refused_chars": refused_chars} if refused_chars else {}),
                "native_landing_notified": landed,  # posted to the transcript
                "native_landing_sent": landing_sent,  # a provider send carried it
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
            if end_reason == "round_without_progress":
                raise ReviewRouteUnavailable(
                    f"native review episode round {self._rounds_used} carried neither "
                    "an answer nor a well-formed tool call; the episode fails closed "
                    "— a zero-progress round would re-enter the paid send forever",
                    code="native_round_without_progress")
            raise ReviewRouteUnavailable(
                f"native review episode transcript ({refused_chars or transcript_chars} chars) "
                f"exceeded its bound ({transcript_cap}) before a final answer; "
                "the episode fails closed — compaction would review a "
                "fabricated cut", code="native_transcript_cap_exceeded")
        if episode.get("native_incomplete"):
            self._episode_deltas.append({
                "kind": "capability_delta",
                "requested": "a finished report from the episode",
                "effective": (
                    f"the reviewer's last draft ({len(final_answer)} chars) — "
                    f"{end_reason} after {self._rounds_used} rounds without a final "
                    f"answer (transcript {transcript_chars} of {transcript_cap} chars)"
                ),
                "reason": f"native_{end_reason}_before_final_answer",
            })
        self._raw_transcript = final_answer

    def _open_episode(self, root: str, drive_root: Any) -> tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """The episode's opening: the inspection registry, the first send's
        messages, and that send measured the way EVERY send is measured.

        The size is what a send actually carries — the serialized message
        objects (system instructions and the task) plus the tool schemas that
        ride every provider call — computed by the one `_wire_size` measure
        that every later append recomputes. Counting raw text understated an
        escape-heavy first send; summing bare envelopes drifted from the list.
        Units are CHARS throughout — same as the cap."""
        registry, schemas = self._inspection_registry(root, drive_root)
        messages = native_first_send_messages(self.episode_prompt)
        return registry, schemas, messages, _wire_size(messages, schemas)

    def _chat_kwargs(self, messages: List[Dict[str, Any]], schemas: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
        """One round's provider call, shaped from the request and the slot."""
        request, slot = self.assignment.request, self.assignment.slot
        kwargs: Dict[str, Any] = {
            "messages": messages,
            "model": slot.model,
            "tools": schemas,
            "tool_choice": "auto",
            "reasoning_effort": slot.effort,
            "max_tokens": max_tokens,
            "no_proxy": bool(request.no_proxy),
            "use_local": bool(slot.use_local),
            "cache_affinity": f"{request.surface}:{request.task_id or 'review'}",
        }
        if request.temperature is not None or slot.temperature is not None:
            kwargs["temperature"] = request.temperature if request.temperature is not None else slot.temperature
        return kwargs

    def _execute_inspection_call(
        self, registry: Any, tc: Dict[str, Any], validation_by_id: Dict[str, Any],
        *, round_idx: int, room: int,
    ) -> Dict[str, Any]:
        """Run ONE inspection tool call of a round and return its tool MESSAGE.

        Owns the tool-policy half of the episode — wire-validation refusal, the
        inspection allowlist, argument parsing, execution, the disclosed result
        bound (the fixed per-result cap, clamped to ``room`` and measured on the
        SERIALIZED message the send will carry, so JSON escaping cannot inflate
        it past the room) and the host-observed receipt — while the caller owns
        the loop, the transcript counter and the messages. Provenance comes from
        CONTROL FLOW, never string-sniffing: a refused call must not read as an
        executed one.
        """
        call_id = str(tc.get("id") or "")
        function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = str(function.get("name") or "")
        raw_args = function.get("arguments")
        args: Optional[Dict[str, Any]] = None
        outcome = "executed"
        extent: Dict[str, Any] = {}
        verdict = validation_by_id.get(call_id)
        if room < _RESULT_ROOM_FLOOR_CHARS:
            # The round's earlier calls spent the room below the bound: a read
            # whose result could not be returned is not performed at all. The
            # stub itself is charged against the room and is empty once even a
            # stub would not fit; every call still costs its message envelope
            # (each call must be answered) — the caller measures the complete
            # serialized envelope and ends the episode typed when even that
            # cannot fit, so no over-bound send is ever made.
            outcome = "withheld"
            stub = (
                "⚠️ RESULT WITHHELD: the episode transcript budget is spent; "
                "answer now from what you have read."
            )
            result = stub if room >= len(stub) else ""
        elif verdict is not None and not getattr(verdict, "allows_execution", True):
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
                    result = str(registry.execute(name, args))
                except Exception as exc:  # tool errors feed the model, not the rail
                    outcome = "error"
                    result = f"⚠️ {type(exc).__name__}: {exc}"
                result_cap = max(0, min(_EPISODE_TOOL_RESULT_CHAR_CAP, room))
                # Disclosed bound with a continuation handle — the reader keeps
                # reading in chunks (read_file supports start_line/max_lines and start_char for long lines), so
                # nothing is silently cut. The marker is budgeted INSIDE the
                # cap, and the cut is measured on the SERIALIZED message (JSON
                # escaping inflates real text) until it fits the room — a raw
                # result under the cap can still overshoot the room on the wire.
                marker = (
                    " — the episode transcript budget is nearly spent;"
                    " answer now from what you have read."
                    if room < _EPISODE_TOOL_RESULT_CHAR_CAP else
                    ". Continue reading the remainder in bounded"
                    " chunks (read_file supports start_line/max_lines and start_char for long lines)."
                )
                full = result
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
                if name == "read_file" and outcome == "executed":
                    # Measured on `sent`, never on a `shown` the exhausted fit
                    # loop reduced after the last build: the receipt credits
                    # exactly what the reviewer received.
                    extent = self._read_extent(full, sent)
        # Host-observed evidence (bounded): which artifacts THIS episode
        # actually opened — disclosure, never a claim of full-surface coverage.
        self._tool_calls_total += 1
        if len(self._tool_receipts) < 200:
            receipt: Dict[str, Any] = {"round": round_idx, "tool": name}
            if isinstance(args, dict):
                for key in ("path", "root", "query", "pattern"):
                    if args.get(key):
                        receipt[key] = str(args[key])[:300]
            receipt["result_chars"] = len(result)
            receipt["outcome"] = outcome
            receipt.update(extent)  # read_file only: the DELIVERED extent + the opened path/root (see _read_extent)
            self._tool_receipts.append(receipt)
        return {"role": "tool", "tool_call_id": call_id, "content": result}

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
        instance-local and dispatch is synchronous — so a call refused before
        the tool ran, or one that returned without rendering, finds no stamp
        and records no extent. The ``last_read_view`` WRITER-SET invariant
        backs this: exactly three writers exist — the reader's entry reset and
        its stamp in ``tools/core_file_tools.py`` and this episode's clear-before-dispatch
        — pinned by a static test, so a fourth writer cannot forge coverage
        silently. When this episode's result bound cut the body, only the
        complete lines whose end lies inside the delivered prefix count.
        Fail-safe: a stamp missing any fact records NO extent, which coverage
        reads as ``unobserved``. Extends the receipt contract; the existing
        fields and the outcome vocabulary are unchanged."""
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
        return {"start_line": start, "end_line": end, "total_lines": total, "eof": end >= start and end >= total,
                "opened_path": view["opened_path"][:300], "opened_root": view["opened_root"][:64]}

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
        # host landing notice (a user message) is disclosed, never relabelled.
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
        delivery, rounds, receipts, transcript vs bound, end reason and the paid
        ledger facts of the rounds that DID run — so a failed native execution
        stays visible on the public wire instead of vanishing. Empty until the
        episode started."""
        return dict(self._episode_usage)

    def _emit_episode_fact(self, episode: Dict[str, Any]) -> str:
        """One typed custody row per episode end (rounds, transcript vs bound,
        landing, end reason). Returns the row's fate — ``written``, ``failed``
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
        canonical, method, extraction_usage = canonicalize_session_verdict(
            text,
            conformance_passed=False,  # no structured-output channel on this route
            contract=self._output_contract(),
            llm=self.llm,
            deadline_at=getattr(self.assignment.request, "deadline_at", "") or "",
            transport_timeout_sec=getattr(self.assignment.slot, "transport_timeout_sec", None),
            shape=review_output_shape(self.assignment.request.surface),
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
