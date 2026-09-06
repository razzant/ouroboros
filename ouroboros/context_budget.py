"""Single source of truth for AGENT-context size budgets.

These govern the size of Ouroboros's OWN working context: the main-loop
assembled prompt, the typed context-reclaim request/receipt contract, and
the background consciousness context guards.

They are deliberately SEPARATE from the REVIEW-prompt budget family
(``ouroboros.tools.review_helpers.REVIEW_PROMPT_TOKEN_BUDGET`` and the
``ouroboros.tools.scope_review`` window constants), which sizes reviewer
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

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

# Owner-selected Low's total-context economy/short-window target. This is an
# elastic target, not a provider admission ceiling: Phase 2 measures the sealed
# Main input plus its unchanged response reserve against T and the selected
# route capacity W, requests at most one useful reclaim pass, then sends best
# effort. Crossing T never creates a task failure.
OWNER_LOW_TARGET_TOKENS = 200_000

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

# Background-consciousness assembled-context guards. P1: fail fast, never
# silently truncate cognitive artifacts.
BG_CONTEXT_WARN_CHARS = 600_000   # ~150K tokens: warn but proceed
BG_CONTEXT_MAX_CHARS = 1_200_000  # ~300K tokens: skip the wakeup cycle

# Drive-state JSON injection guard inside the consciousness context.
BG_STATE_JSON_WARN_CHARS = 200_000

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

# --- Hot-store growth thresholds (health invariant; bytes) -------------------
# Deterministic tripwires for the append-only stores whose interactive readers
# degrade with file size (BIBLE P2: the class was caught by the owner, not by
# any instrument — these thresholds are the instrument). Same family as
# SCRATCHPAD_BLOAT_WARN_CHARS above: a health-invariant WARNING, not a gate.
#
# Ledger: measured evidence in ouroboros/usage_ledger.py::_locked — a ~20MB
# usage_attempts.jsonl costs ~0.5s per full re-read UNDER THE MONETARY LOCK,
# starving concurrent workers (the 2026-07-23 lock-timeout incident). Warn at
# exactly that measured degradation point. Since CPL4-C6, size-triggered
# compaction (config.USAGE_LEDGER_COMPACT_BYTES, usage_compaction.py) should
# hold the file far below this — like the rotation-log warns, this fires only
# if compaction is broken, the unfoldable residue itself grows this large, or
# the lock directory takes no kernel locks and compaction refuses on the name
# tier (typed usage_ledger_compaction_refused event, once per process).
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
# Background observations are append-only and replayed by the consciousness
# owner on each wake.  This is a warning, not a retention gate: acknowledged
# and unacknowledged rows remain durable until a future owner-approved archive.
BG_OBSERVATIONS_WARN_BYTES = 20_000_000
# Compact root-task -> skill review index used by acceptance packet assembly.
SKILL_REVIEW_ROOT_TASKS_WARN_BYTES = 20_000_000
# ``chat_history`` can deliberately replay the archive chain, while ordinary
# context reads only the unconsolidated generation suffix.  Warn before an
# explicit full-history read becomes seconds-scale; this is observability, not
# a retention gate and never shortens the memory horizon.
CHAT_ARCHIVE_SCAN_WARN_BYTES = 100_000_000
# Custody replay (delegate_custody) walks the WHOLE events chain — live file
# plus archive/events_*.jsonl — on ownership questions. This inherits the
# pre-rotation 100MB replay-degradation signal, now measured over the chain;
# archives stay durable history (never GC'd), so the remediation is chain
# indexing/compaction, never deletion.
EVENTS_ARCHIVE_SCAN_WARN_BYTES = 100_000_000


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
