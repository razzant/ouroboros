"""Single source of truth for AGENT-context size budgets.

These govern the size of Ouroboros's OWN working context: the main-loop
assembled prompt and the typed context-reclaim request/receipt contract.

They are deliberately SEPARATE from the REVIEW-prompt budget family
(``ouroboros.tools.review_helpers.REVIEW_PROMPT_TOKEN_BUDGET`` and the
``ouroboros.tools.scope_window`` window constants), which sizes reviewer
prompts, not the agent's own context. Merging the two would couple unrelated
concerns and is explicitly avoided.

Constants, frozen context-reclaim records, and pure classification helpers
only. This module must stay import-pure: no imports from ``ouroboros.llm``,
``ouroboros.loop*``, or any other runtime module, so every seam can import
the shared vocabulary without cycles.

Char-based guards assume the ~chars/4 estimate (``ouroboros.utils.estimate_tokens``);
the comments give the approximate token equivalents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

# Owner-selected Low's total-context economy/short-window target. This is an
# elastic target, not a provider admission ceiling: Phase 2 measures the sealed
# Main input plus its unchanged response reserve against T and the selected
# route capacity W, requests at most one useful reclaim pass, then sends best
# effort. Crossing T never creates a task failure. Low 250K and Nano 85K are the
# owner's choice of 2026-09-29.
OWNER_LOW_TARGET_TOKENS = 250_000

# Nano's owner-selected target sizes what goes INTO the request (the memory view
# floor, the book form, the schema selection, the task-input pointer, the one
# deficit-triggered reclaim). It stands in for the route window only when that
# window is unknown. NANO_MIN_HEADROOM_TOKENS is the minimum the input fit leaves
# for the reply and the reply floor R of ``reply_allowance_tokens``; the reply
# itself follows the known window up to the caller's ceiling (owner decisions
# 2026-10-05: Q1=A reply follows the window, Q3=A slack W/8 on approximate counts).
OWNER_NANO_TARGET_TOKENS = 85_000
NANO_MIN_HEADROOM_TOKENS = 8_192

# Low-water sizing of the automatic context-reclaim pass. The TRIGGER is
# unchanged: a positive deficit against the binding boundary (the smaller known
# of the owner target T and the route capacity W), one pass per route+round.
# Only the SIZE of the requested pass changes: goal = deficit +
# ceil(boundary / RECLAIM_LOW_WATER_DIVISOR), and 0 without a deficit. A pass
# sized to the deficit alone lands exactly AT the boundary, so the next round's
# ordinary growth re-arms it (a summarizer pass nearly every round). Sized this
# way it lands about an eighth of the boundary below (~125K tokens on a 1M
# route, ~31K under the 250K Low target), so the next pass needs that much real
# growth. Structural constant, not a setting: 8 (12.5 % of the boundary) is a
# disclosed design choice, not a measured optimum; change it here and only here
# (tests/test_context_budget_ssot.py pins it). Cost: older history is condensed
# sooner and each summarizer pass is larger. The materializer, its receipts and
# the route+round latch are unchanged; the checkpoint event records requested
# margin versus achieved headroom (context_fit.measure_main_fit,
# loop_model_call._run_main_reclaim).
RECLAIM_LOW_WATER_DIVISOR = 8

# Working room the memory view's physical floor leaves under a known route
# window: two low-water levels of the reclaim pass above, ceil(W / divisor)
# each. One is the level an in-task reclaim pass lands on (its goal is the
# deficit plus one level), the other is room for the work between two passes;
# with less, the first tool results of a task would re-arm a paid reclaim. The
# floor turns host facts, room headers, legacy pointers and old pages into
# addresses only when the view does not fit the window minus the reply reserve
# and this room; my own replies and people's words become addresses only when
# the window minus the reply reserve cannot hold them. It never adds memory to
# fill the room. Structural constant, not a setting;
# tests/test_context_budget_ssot.py pins it.
MEMORY_VIEW_WORKING_MARGINS = 2

# Low-water levels left to the work under an owner-selected Low or Nano target
# (the target bounds the view the way a window does, with the full reply
# reserve). One level keeps a Low view at most 250 000 - 65 536 - 31 250 =
# 153 214 estimated input tokens and a Nano view 85 000 - 8 192 - 10 625 =
# 66 183, so an in-task reclaim pass still has a level to land on below the
# target; 0 makes the target a plain frame. The owner chose one level; the
# value changes here and only here (tests/test_context_budget_ssot.py pins it).
MODE_TARGET_WORKING_MARGINS = 1


def context_mode_limits(mode: str, owner_mode: str, output_reserve_tokens: int) -> Tuple[Optional[int], int]:
    """``(owner target, reply reserve)`` of a rendered mode.

    A target binds only the mode the owner selected: task-local Low and a mode the
    window lowered keep the window alone. Nano's reserve is the reply floor R of
    ``reply_allowance_tokens`` (its headroom, never above the caller's ceiling), so
    the memory floor and the mode selection use the same R as the send.
    """
    target = {"low": OWNER_LOW_TARGET_TOKENS, "nano": OWNER_NANO_TARGET_TOKENS}.get(mode) if mode == owner_mode else None
    return target, min(NANO_MIN_HEADROOM_TOKENS, output_reserve_tokens) if mode == "nano" else output_reserve_tokens


def _reply_floor(caller_max_tokens: int, nano: bool) -> Tuple[int, int]:
    """``(C, R)``: the caller's ceiling and the reply floor of the rendered mode."""
    ceiling = max(0, int(caller_max_tokens))
    return ceiling, min(NANO_MIN_HEADROOM_TOKENS, ceiling) if nano else ceiling


def reply_allowance_tokens(
    *, caller_max_tokens: int, nano: bool, owner_nano: bool, input_tokens: int,
    raw_input_tokens: Optional[int] = None, window_tokens: Optional[int] = None, exact: bool = False,
) -> int:
    """The one reply allowance of a prepared Main candidate (the wire ``max_tokens``).

    ``C`` is the caller's ceiling, ``R`` the reply floor (``NANO_MIN_HEADROOM_TOKENS``
    in a rendered Nano, ``C`` in Low and Max, so those always get ``C``). Under a known
    window ``W`` the reply gets what the window leaves after the input: an
    approximate count admits by the larger of the calibrated and the raw estimate
    and leaves a slack of ``W / RECLAIM_LOW_WATER_DIVISOR`` for estimate error; an
    exact count (the local server's tokenizer) needs no slack and may raise an
    over-cautious estimate. Without a window the owner's Nano target stands in for
    it. Continuous in the input, never below ``R``, never above ``C``: a ceiling below
    the floor (a local lane's 2,048 or quarter window) is never raised.
    """
    ceiling, floor = _reply_floor(caller_max_tokens, nano)
    if window_tokens is not None and int(window_tokens) > 0:
        window = int(window_tokens)
        measured = int(input_tokens) if exact else max(int(input_tokens), int(raw_input_tokens or input_tokens))
        slack = 0 if exact else math.ceil(window / RECLAIM_LOW_WATER_DIVISOR)
        return min(ceiling, max(floor, window - measured - slack))
    if owner_nano:
        return min(ceiling, max(floor, OWNER_NANO_TARGET_TOKENS - int(input_tokens)))
    return ceiling


def exact_reply_shortfall(*, caller_max_tokens: int, nano: bool, input_tokens: int, window_tokens: Optional[int]) -> bool:
    """An exactly counted input leaves less than the reply floor under a known window.

    The typed local overflow (``LocalContextTooLargeError``) before any send; an
    estimate never establishes it.
    """
    _ceiling, floor = _reply_floor(caller_max_tokens, nano)
    return window_tokens is not None and int(window_tokens) > 0 and int(window_tokens) - int(input_tokens) < floor


def request_context_budget(
    *, window_tokens: Optional[int], output_reserve_tokens: Optional[int], non_memory_tokens: int,
    target_tokens: Optional[int] = None, calibration_ratio: float = 1.0, margin_count: int = 1,
) -> Dict[str, Any]:
    """One measured frame: the memory allowance with and without the working margins.

    The boundary is the smaller known of an owner target and the route window;
    with neither known every allowance is ``None`` (unknown evidence never
    prohibits a route). Input and returned allowances use estimator tokens;
    boundary, reserve, margin and free space use calibrated tokens. Unknown
    reply space gives an explicitly optimistic upper estimate, never a known
    zero or native-fit proof. A zero margin count measures a physical frame
    without a working reserve.
    """
    known = [int(v) for v in (target_tokens, window_tokens) if v is not None and int(v) > 0]
    boundary = min(known) if known else None
    ratio = max(float(calibration_ratio or 1), 0.01)
    reserve = max(0, int(output_reserve_tokens or 0))
    measured = math.ceil(max(0, int(non_memory_tokens)) * ratio)
    margin = (math.ceil(boundary / RECLAIM_LOW_WATER_DIVISOR) * margin_count
              if boundary is not None else 0)
    free = boundary - reserve - measured if boundary is not None else None

    def allowance(extra: int) -> Optional[int]:
        return (max(0, math.floor((boundary - reserve - extra) / ratio) - non_memory_tokens)
                if boundary is not None else None)

    return {"boundary_tokens": boundary, "target_tokens": target_tokens, "window_tokens": window_tokens,
            "output_reserve_tokens": output_reserve_tokens, "reserve_known": output_reserve_tokens is not None,
            "working_margin_tokens": margin, "non_memory_tokens": non_memory_tokens,
            "calibrated_input_tokens": measured, "calibration_ratio": ratio, "free_tokens": free,
            "with_margin_tokens": allowance(margin), "without_margin_tokens": allowance(0),
            "target_deficit_tokens": (max(0, measured + reserve - target_tokens) if target_tokens else None),
            "capacity_deficit_tokens": (max(0, measured + reserve - window_tokens) if window_tokens else None)}

# One overflow vocabulary for every seam that must recognize a CONTEXT-WINDOW
# overflow (Main provider-code precedence, the local transport, and the
# summarizer split path). A provider code or message shape added here reaches
# all three seams at once; per-module copies drifted independently before.
CONTEXT_OVERFLOW_CODES = frozenset({
    "context_length_exceeded",
    "context_window_exceeded",
    "model_context_window_exceeded",
    "prompt_too_long",
    "input_too_long",
})
CONTEXT_OVERFLOW_MESSAGE_MARKERS = (
    "context_length_exceeded",
    "context length",
    "maximum context",
    "prompt is too long",
    "exceeds the context",
    "exceed context limit",
    "context window",
    "input is too long",
)
# OUTPUT/body-size limits take precedence over overflow markers: a message like
# "max_tokens 65536 exceeds maximum context length 32768" is an output-limit
# rejection, not a window overflow — shrinking the prompt cannot fix it.
OUTPUT_OR_BODY_SIZE_MARKERS = (
    "max_tokens",
    "maximum tokens",
    "output tokens",
    "maximum output",
    "request body too large",
    "body too large",
)


def output_or_body_size_message(text: Any) -> bool:
    """True for output/body limits, excluding the combined input+output window."""
    low = str(text or "").lower().replace("`", "")
    if "input length and max_tokens exceed context limit" in low:
        return False  # shrinking input can make the unchanged output reserve fit
    return any(marker in low for marker in OUTPUT_OR_BODY_SIZE_MARKERS)


def context_overflow_message(text: Any) -> bool:
    """True when a provider error message matches the shared overflow markers.

    Applies the output-size precedence itself so every seam (Main classifier,
    local transport, summarizer split) classifies identically: a message that
    matches an output/body-size marker is NOT a context overflow. Callers check
    structured overflow codes first; a structured code still wins over this
    message-level verdict.
    """
    low = str(text or "").lower()
    if output_or_body_size_message(low):
        return False
    return any(marker in low for marker in CONTEXT_OVERFLOW_MESSAGE_MARKERS)

MeasurementBasis = Literal["fresh_route_usage", "fresh_model_usage", "cold_estimate"]
ReclaimStatus = Literal[
    "applied", "no_eligible", "no_positive_reclaim", "checkpoint_failed",
    "summarizer_failed", "no_measurable_shrink", "binding_mismatch",
    "no_op", "fit_rejected", "source_unavailable",
]


@dataclass(frozen=True)
class ContextReclaimRequest:
    route_fp: str
    round_id: str
    transcript_sha256: str
    measurement_basis: MeasurementBasis
    measurement_density: float
    reclaim_goal_tokens: int
    allow_partial_shrink: bool = True
    working_note: Optional[str] = None
    expected_view_revision: str = ""
    keep_unit_ids: Optional[Tuple[str, ...]] = None
    restore_unit_refs: Tuple[Dict[str, Any], ...] = ()
    schema_names: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class ContextReclaimReceipt:
    status: ReclaimStatus
    before_transcript_sha256: str
    after_transcript_sha256: str
    selection_fingerprint: str
    selected_unit_ids: Tuple[str, ...]
    reclaimed_tokens: int
    goal_reached: bool
    checkpoint_ref: Optional[Dict[str, Any]]
    capsule_refs: Tuple[Dict[str, Any], ...]
    observed_view_revision: str = ""
    view_revision: str = ""
    retained_unit_ids: Tuple[str, ...] = ()
    restored_unit_refs: Tuple[Dict[str, Any], ...] = ()
    source_refs: Tuple[Dict[str, Any], ...] = ()
    schema_names: Optional[Tuple[str, ...]] = None
    fit: Optional[Dict[str, Any]] = None


class SummarizerContextOverflow(RuntimeError):
    """Typed permission to split one summarizer batch or source."""


class LocalContextTooLargeError(RuntimeError):
    """Raised when a local model cannot fit context without silent truncation.

    A pre-dispatch exact shortfall (``exact_reply_shortfall``) raises it with
    ``refused_candidate``: the JSON-safe facts of THE candidate it refused (model,
    provider, the allowance it would have sent, the canonical candidate identity
    and its physical context), which the one strict-shrink retry compares with
    instead of an earlier round's capture. The historical home ``llm_local``
    re-exports the name.
    """

    refused_candidate: Optional[Dict[str, Any]] = None


class _UnsafeVisual(ValueError):
    pass


class _UnitSummaryFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class _AtomicUnit:
    unit_id: str
    start: int
    end: int
    raw_sha256: str
    raw_size_bytes: int
    context_size_tokens: int
    source_text: str
    source_sha256: str
    predicted_reclaim_tokens: int
    generation: int
    lineage_hashes: Tuple[str, ...]
    source_refs: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class _SelectedUnit:
    unit: _AtomicUnit
    summary_budget_tokens: int
    negative_memo_key: str


@dataclass(frozen=True)
class _Selection:
    units: Tuple[_SelectedUnit, ...]
    fingerprint: str
    predicted_reclaim_tokens: int


@dataclass(frozen=True)
class _Part:
    root_id: str
    source_id: str
    start_char: int
    end_char: int
    text: str
    sha256: str

# WARN threshold for a single oversized governance/knowledge context section.
LARGE_CONTEXT_SECTION_CHARS = 200_000

# Main-only predecessor projection bounds.  These are deliberately separate
# from the generic section warning: an oversized terminal result is replaced
# by its authored continuation narrative, not merely logged and sent whole.
PREDECESSOR_RESULT_INLINE_CHARS = 200_000
CONTINUATION_NARRATIVE_LEGACY_GENERATIONS = 3
CONTINUATION_NARRATIVE_LEGACY_TAIL_BYTES = 512 * 1024
CONTINUATION_NARRATIVE_LEGACY_MAX_ROWS = 5_000

# --- Native image blocks (v6.26.0 multimodal chat) ---------------------------
# Char-equivalent for ONE image block in chars/4 token estimates (~1.1K tokens):
# vision models bill per tile, not per base64 char.
IMAGE_BLOCK_CHAR_EQUIVALENT = 4_400
# Live image blocks kept in the transcript (single counter across owner
# uploads, browser screenshots, and transport injections). Older images are
# replaced by a caption placeholder pointing to the re-view path.
# 3 -> 5 (v6.81.1, owner decision 2026-07-29): with tool-result images now
# auto-attached (screenshots arrive every observation round), 3 kept too little
# visual history for compare-two-screens reasoning; 5 costs at most ~2.2K extra
# estimated tokens per request and only when that many images are actually live.
MAX_LIVE_IMAGE_BLOCKS = 5

# --- Scratchpad size thresholds (SSOT; previously scattered literals) -------
# Context-section soft budget for the rendered scratchpad (warn-only).
SCRATCHPAD_SECTION_BUDGET_CHARS = 90_000
# Health-invariant bloat warning ("extract durable insights to knowledge").
SCRATCHPAD_BLOAT_WARN_CHARS = 50_000
# Block-storage consolidation trigger (consolidator compresses oldest blocks).
SCRATCHPAD_CONSOLIDATION_THRESHOLD_CHARS = 30_000
# Source-side content cap for Memory.append_scratchpad_block (ibl-2b09abdadd25).
# SINGLE SOURCE OF TRUTH for the new content cap; obeys the ordering invariant
#   SCRATCHPAD_CONSOLIDATION_THRESHOLD_CHARS (30_000)
#     < SCRATCHPAD_MAX_CONTENT_CHARS (60_000)
#     <= SCRATCHPAD_SECTION_BUDGET_CHARS (90_000) - RENDERING_HEADROOM
# Pinned at 60_000 so the consolidator (trigger at >30_000) still fires before
# this cap kicks in, AND the rendered section's framing (header + per-block
# '### [...]' / '---' / journal-pointer lines) fits under the section budget
# at the 10-block cap. Measured in characters (len(str)), matching the family
# naming convention. The count cap (_SCRATCHPAD_MAX_BLOCKS in ouroboros/memory.py)
# is AND'd with this cap (single-pass eviction), not replaced by it.
SCRATCHPAD_MAX_CONTENT_CHARS = 60_000

# --- Hot-store growth thresholds (health invariant; bytes) -------------------
# Deterministic tripwires for the append-only stores whose interactive readers
# degrade with file size (BIBLE P2: the class was caught by the owner, not by
# any instrument — these thresholds are the instrument). Same family as
# SCRATCHPAD_BLOAT_WARN_CHARS above: a health-invariant WARNING, not a gate.
# The money record (state/usage.sqlite) is not one of them: its readers address
# summary rows and single attempts, so its size does not reach them.
# events/tools/supervisor/task_reflections logs are ROTATION-BOUNDED since the
# CPL4-C1..C4 rotation train (same 800KB rotator and supervisor tick as
# chat/progress). 8MB = 10x the rotation cap: these warnings fire only if
# rotation is broken or missing — deliberate regression tripwires, not size
# preferences (the pre-train 100MB values watched for replay degradation of
# the then-unbounded live files; that duty moved to the archive-chain watch
# below).
EVENTS_LOG_WARN_BYTES = 8_000_000
TOOLS_LOG_WARN_BYTES = 8_000_000
SUPERVISOR_LOG_WARN_BYTES = 8_000_000
TASK_REFLECTIONS_LOG_WARN_BYTES = 8_000_000
# progress.jsonl is expected to be ROTATION-BOUNDED (the chat.jsonl rotation
# pattern, 800KB cap in supervisor/state.py::rotate_chat_log_if_needed,
# generalized to progress by the perf/lifecycle sprint). 8MB = 10x that cap:
# this warning fires only if rotation is broken or missing — a deliberate
# regression tripwire, not a size preference.
PROGRESS_LOG_WARN_BYTES = 8_000_000
# state/scheduled_tasks.json is a whole-document store the scheduler READS AND
# REWRITES on every tick under the queue lock (supervisor/queue.py::
# check_scheduled_tasks), and consumed one-shot follow-ups are RETAINED as
# durable receipts (enabled=False + completed_at) — so it now grows with every
# fired follow-up. 2MB ≈ thousands of ~1KB records: the point where a
# per-tick full parse + atomic rewrite under the lock stops being free.
SCHEDULED_TASKS_WARN_BYTES = 2_000_000
# Compact root-task -> skill review index used by acceptance packet assembly.
SKILL_REVIEW_ROOT_TASKS_WARN_BYTES = 20_000_000
# memory/chronicle/records.jsonl is my memory's only authority and is never rotated
# (records are never rewritten). Every task context decodes each acting page and part
# body from its index (memory_view._story_pages), and an owner's install imports about
# 4.4 MB of legacy memory at activation (measured on a copy, 2026-10-04). 64MB is ~15x that: past
# it, decoding the story on every task context stops being free and a projection of the
# acting records is due. Observability, never a retention gate: nothing is cut.
CHRONICLE_JOURNAL_WARN_BYTES = 64_000_000
# ``chat_history`` can deliberately replay the archive chain, while the memory
# view reads only rows after the retold-memory frontier.  Warn before an
# explicit full-history read becomes seconds-scale; this is observability, not
# a retention gate and never shortens the memory horizon.
CHAT_ARCHIVE_SCAN_WARN_BYTES = 100_000_000
# Review ledger index chain (state/review_ledger/index*.jsonl): the hot index rotates
# itself at review_ledger.INDEX_MAX_BYTES and task context reads only that hot index;
# readers that walk the rotated segments (recent_records without hot_only) replay the
# whole chain, so its total size is enrolled here like the chat archive chain.
REVIEW_LEDGER_INDEX_WARN_BYTES = 64_000_000
# The FIRST custody read of each process folds the WHOLE events chain — live
# file plus archive/events_*.jsonl — into the process-local row memo
# (delegate_custody_memo); later reads fold only appended bytes. Explicit
# forensic and retirement scans still walk the chain. This inherits the
# pre-rotation 100MB replay-degradation signal, now measured over the chain;
# archives stay durable history (never GC'd), so the remediation is a durable
# compact custody projection, never deletion.
EVENTS_ARCHIVE_SCAN_WARN_BYTES = 100_000_000
# Warn before the observed 242-of-253 retained-drive corpus becomes routine;
# count only direct children because startup health is an interactive path.
RETAINED_EXECUTION_DRIVES_WARN_COUNT = 200


def estimate_message_chars(messages: Any) -> int:
    """Message chars with image blocks at the provider-billing proxy.

    Serves the local-context compaction proxy (`llm.py`); the remote fit
    estimator and the density witness measure on `context_fit`'s
    `estimate_context_prompt_tokens` basis instead, which serializes message
    dicts recursively. Image base64 never counts as text here.
    """
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if str(block.get("type") or "") in ("image_url", "image"):
                    total += IMAGE_BLOCK_CHAR_EQUIVALENT
                    continue
                total += len(str(block.get("text", "")))
        else:
            total += len(str(content or ""))
        # Reasoning kept on canonical assistant turns is replayed verbatim on
        # the reasoning-echo lane (DeepSeek), so it is real wire prompt. A
        # mixed transcript sent to a non-echo lane still carries the key here
        # while the wire copy strips it — a conservative over-count, the safe
        # direction for a compaction trigger.
        total += len(str(msg.get("reasoning_content") or ""))
    return total
