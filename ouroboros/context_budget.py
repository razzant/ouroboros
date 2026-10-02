"""Single source of truth for AGENT-context size budgets.

These govern the size of Ouroboros's OWN working context: the main-loop
assembled prompt and the typed context-reclaim request/receipt contract.

They are deliberately SEPARATE from the REVIEW-prompt budget family
(``ouroboros.tools.review_helpers.REVIEW_PROMPT_TOKEN_BUDGET`` and the
``ouroboros.tools.scope_review`` window constants), which sizes reviewer
prompts, not the agent's own context. Merging the two would couple unrelated
concerns and is explicitly avoided.

Constants, frozen context-reclaim records, and pure context-payload helpers
only. This module must stay import-pure: no imports from ``ouroboros.llm``,
``ouroboros.loop*``, or any other runtime module, so every seam can import
the shared vocabulary without cycles.

Char-based guards assume the ~chars/4 estimate (``ouroboros.utils.estimate_tokens``);
the comments give the approximate token equivalents.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Literal, Optional, Tuple


def canonical_context_json(value: Any) -> str:
    """Stable context/source presentation bytes, shared by capture and reclaim."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def context_fit_event_fields(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Project captured context facts into the public attempt-event vocabulary."""
    return {
        "context_memory_view": usage.get("_context_memory_view"),
        "context_route_fp": str(usage.get("_context_route_fp") or ""),
        "estimated_prompt_tokens": int(usage.get("_context_prompt_estimate") or 0),
        "context_fit_mode": str(usage.get("_context_fit_mode") or ""),
        "context_profile": str(usage.get("_context_profile") or ""),
        "context_measurement_basis": str(usage.get("_context_measurement_basis") or ""),
        "context_measurement_density": float(usage.get("_context_measurement_density") or 0.0),
        "context_target_total_tokens": usage.get("_context_target_total_tokens"),
        "context_capacity_total_tokens": usage.get("_context_capacity_total_tokens"),
        "context_target_deficit_tokens": usage.get("_context_target_deficit_tokens"),
        "context_capacity_deficit_tokens": usage.get("_context_capacity_deficit_tokens"),
        "context_reclaim_goal_tokens": int(usage.get("_context_reclaim_goal_tokens") or 0),
        "context_target_miss": bool(usage.get("_context_target_miss")),
        "context_automatic_pass_used": bool(usage.get("_context_automatic_pass_used")),
        "context_predicted_capacity_miss": bool(
            usage.get("_context_predicted_capacity_miss")
        ),
    }


def extract_plain_text_from_content(content: Any) -> str:
    """Extract text from strings or multipart content for transcript sealing."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""


# Host context presentation shared by renderers and physical-attempt accounting.
# These delimit sources; they never select semantic behavior or authorize work.
MEMORY_BEGIN = "\n[Ouroboros memory begins]\n"
MEMORY_END = "\n[Ouroboros memory ends]\n"
MEMORY_FACTS_PREFIX = "\n[Memory view facts] "
HOST_CONTEXT_NOTICE_BEFORE_TASK = (
    "Host context for this turn: memory, knowledge index, runtime facts and recent "
    "activity, rendered by the runtime as a continuation of the system prompt. Not "
    "written by my human and not a message to answer; the message to act on follows next."
)
HOST_CONTEXT_NOTICE_AFTER_TASK = (
    "Host context for this turn: memory, knowledge index, runtime facts and recent "
    "activity, rendered by the runtime as a continuation of the system prompt. Not "
    "written by my human and not a message to answer; the message to act on is the one above."
)

# Owner-selected Low's total-context economy/short-window target. This is an
# elastic target, not a provider admission ceiling: Phase 2 measures the sealed
# Main input plus its unchanged response reserve against T and the selected
# route capacity W, requests at most one useful reclaim pass, then sends best
# effort. Crossing T never creates a task failure.
OWNER_LOW_TARGET_TOKENS = 250_000

# Nano's owner-selected total window and free input headroom. The send boundary
# chooses the largest output allowance up to the caller's existing ceiling;
# the headroom is a minimum, never a fixed generation cap.
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

# Raw recent-dialogue tail shown when no valid consolidation can represent older
# dialogue. The universal temporal renderer remains issue #220; this PR neither
# shortens nor reinterprets that horizon.
MAX_RECENT_CHAT_TAIL = 1000

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
#
# Ledger: retain the historical 20MB growth tripwire (the 2026-07-23 incident).
# Warm writers now validate only the tail; cold parsing runs outside the money
# lock and revalidates its generation under it. Size still affects cold parsing,
# full projections and compaction, not the cost of every reservation. Since
# CPL4-C6, size-triggered compaction (config.USAGE_LEDGER_COMPACT_BYTES) should
# hold the file below this. Growth can reflect a large unfoldable residue,
# compaction that is broken or refused, or a file that has not yet outgrown the
# growth floor its last committed pass stamped into the ledger header (declined
# before the pass, so no typed event). The name tier (no kernel
# locks) emits usage_ledger_compaction_refused once per process per data root;
# a policy abort (_Abort) emits usage_ledger_compaction_skipped once per process
# per (data root, reason). The two snapshot-race exits before archive/swap only
# log warnings, without a typed event.
USAGE_LEDGER_WARN_BYTES = 20_000_000
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
# Chronicle capture loads retained interpretations from its rebuildable index;
# a cold rebuild also folds the immutable journal. Use the existing indexed
# ledger warning scale to expose that growth, not as a retention or summary
# trigger: original memory must survive any future projection optimization.
CHRONICLE_JOURNAL_WARN_BYTES = 20_000_000
# ``chat_history`` can deliberately replay the archive chain, while ordinary
# context reads only the unconsolidated generation suffix.  Warn before an
# explicit full-history read becomes seconds-scale; this is observability, not
# a retention gate and never shortens the memory horizon.
CHAT_ARCHIVE_SCAN_WARN_BYTES = 100_000_000
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
