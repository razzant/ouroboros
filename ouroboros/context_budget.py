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
import re
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

# Canonical host-row classification; physical requests omit this top-level metadata.
HOST_CONTEXT_KIND_KEY = "_host_context_kind"

# Provenance is typed at delivery, never inferred from text (17.10.2026: a one-letter owner
# answer "A" matched every host row as a substring and shielded 94k tokens of obsolete
# state). A row of the owner's or a principal's inbound words carries ``OWNER_WORDS_KIND``;
# Ouroboros's outward words carry ``owner_dialogue``. Such rows are their own rows, never
# merged with other content, and no host or helper rewrite takes them.
OWNER_WORDS_KIND = "owner_words"
OWNER_ROW_KINDS = frozenset({OWNER_WORDS_KIND, "owner_dialogue"})

# A host-produced state row (round facts, clock, roster) is bound to its visible text at
# production. Only a row whose text still matches is retirable host state; a row written
# before this binding, or changed since (an old merge), stays whole for the mind to fold.
HOST_TEXT_SHA_KEY = "_host_text_sha256"


def bound_host_row(kind: str, text: str) -> Dict[str, Any]:
    """A typed host row bound to its exact visible text (``HOST_TEXT_SHA_KEY``)."""
    import hashlib

    return {"role": "user", HOST_CONTEXT_KIND_KEY: kind, "content": text,
            HOST_TEXT_SHA_KEY: hashlib.sha256(text.encode("utf-8")).hexdigest()}


def host_row_intact(row: Any) -> bool:
    """Whether ``row`` is still exactly the host state it was produced as."""
    import hashlib

    content = row.get("content") if isinstance(row, dict) else None
    return (isinstance(content, str) and not row.get("tool_calls")
            and row.get(HOST_TEXT_SHA_KEY) == hashlib.sha256(content.encode("utf-8")).hexdigest())

# Owner-selected Low's total-context economy/short-window target. This is an
# elastic target, not a provider admission ceiling: Phase 2 measures the sealed
# Main input plus its unchanged response reserve against T and the selected
# route capacity W, requests at most one useful reclaim pass, then sends best
# effort. Crossing T never creates a task failure. Low 250K and Nano 85K are the
# owner's choice of 2026-09-29.
OWNER_LOW_TARGET_TOKENS = 250_000

# Nano's owner-selected target sizes what the host adds to the request unasked (the
# memory view floor, the book form, the schema selection, unsolicited tool output, the
# one deficit-triggered reclaim). For the reply alone it stands in for an unknown
# window; a read the mind chose and the owner's task text answer to the known window
# only, never to this target (owner decisions 2026-10-11 1A, 2026-10-13 #1527 A).
# NANO_MIN_HEADROOM_TOKENS is the minimum the input fit leaves for the reply and the
# reply floor R of ``reply_allowance_tokens``; the reply itself follows the known
# window up to the caller's ceiling (owner decisions 2026-10-05: Q1=A reply follows
# the window, Q3=A slack W/8 on approximate counts).
OWNER_NANO_TARGET_TOKENS = 85_000
NANO_MIN_HEADROOM_TOKENS = 8_192

# The owner's return-after-fold slack (L33 2=A): a recovery pass after a size refusal
# sheds the measured excess plus ceil(boundary / RECLAIM_LOW_WATER_DIVISOR), so the
# next round's ordinary growth does not meet the same wall at once
# (loop_model_call._refusal_target, context_fit.reclaim_low_water_margin). Structural
# constant, not a setting: 8 (12.5 % of the boundary) is a disclosed design choice,
# not a measured optimum; change it here and only here
# (tests/test_context_budget_ssot.py pins it).
RECLAIM_LOW_WATER_DIVISOR = 8
#: An unknown reclaim target: the step takes everything its category can give (the refusal passes).
WHOLE_CATEGORY = 2**31 - 1

def start_room_levels(role: str, *, focus_wake: bool = False, direct_chat: bool = False) -> int:
    """Roots and wakes leave two eighths; direct chat turns, other Presence turns and helpers leave one.

    Decided from task facts, never the task type (``memory_view.start_room`` reads them).
    """
    if role == "consciousness" or (role == "presence" and focus_wake):
        return 2
    return 2 if role == "integrator" and not direct_chat else 1


def context_mode_limits(mode: str, owner_mode: str, output_reserve_tokens: int, *,
                        working_reserve_tokens: Optional[int] = None) -> Tuple[Optional[int], int]:
    """``(owner target, reply reserve)`` of a rendered mode.

    A target binds only the mode the owner selected: task-local Low and a mode the
    window lowered keep the window alone. Nano's reserve is the reply floor R of
    ``reply_allowance_tokens`` (its headroom, never above the caller's ceiling), so
    the memory floor and the mode selection use the same R as the send.
    """
    target = {"low": OWNER_LOW_TARGET_TOKENS, "nano": OWNER_NANO_TARGET_TOKENS}.get(mode) if mode == owner_mode else None
    reserve = min(NANO_MIN_HEADROOM_TOKENS, output_reserve_tokens) if mode == "nano" else output_reserve_tokens
    return target, max(reserve, working_reserve_tokens or 0)


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
    target_tokens: Optional[int] = None, calibration_ratio: float = 1.0, margin_count: int = 1, boundary_tokens: Optional[int] = None,
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
    margin = (math.ceil((min(boundary, boundary_tokens) if boundary_tokens else boundary) / RECLAIM_LOW_WATER_DIVISOR) * margin_count
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
# overflow (Main provider-code precedence and the local transport). A provider
# code or message shape added here reaches both seams at once; per-module copies
# drifted independently before.
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


# Limit phrases share their parser with recovery. Generic lengths require token
# units; header/character limits and bare numbers never establish a window.
_CTX_LIMIT_PATTERNS = tuple(re.compile(pattern, re.I) for pattern in (
    r"context (?:window|length)\s*(?:of|is)?\s*\(?\s*([0-9][0-9,]*)\s*tokens?\b",
    r"maximum allowed length\s*\(?\s*([0-9][0-9,]*)\s*tokens?\b",
    r"maximum (?:input |prompt )?(?:length|tokens?)\s*(?:is|of)?\s*([0-9][0-9,]*)\s*tokens?\b",
    r"tokens\s*>\s*([0-9][0-9,]*)\s*maximum",
    r"context limit:\s*[0-9][0-9,]*\s*\+\s*[0-9][0-9,]*\s*>\s*([0-9][0-9,]*)",
    r"maximum number of tokens allowed\s*\(?\s*([0-9][0-9,]*)",
))


def _parse_ctx_limit_number(text: str) -> int:
    """Extract a stated context-token limit, or 0 when its units are unknown."""
    for pattern in _CTX_LIMIT_PATTERNS:
        match = pattern.search(str(text or ""))
        if match:
            return int(match.group(1).replace(",", ""))
    return 0


def context_overflow_message(text: Any) -> bool:
    """True when a provider error message matches the shared overflow markers.

    Applies the output-size precedence itself so every seam (Main classifier,
    local transport) classifies identically: a message that
    matches an output/body-size marker is NOT a context overflow. Callers check
    structured overflow codes first; a structured code still wins over this
    message-level verdict.
    """
    low = str(text or "").lower()
    if output_or_body_size_message(low):
        return False
    return any(marker in low for marker in CONTEXT_OVERFLOW_MESSAGE_MARKERS) or _parse_ctx_limit_number(low) > 0

MeasurementBasis = Literal["fresh_route_usage", "fresh_model_usage", "cold_estimate"]
ReclaimStatus = Literal[
    "applied", "no_eligible", "checkpoint_failed", "no_measurable_shrink", "binding_mismatch",
    "no_op", "fit_rejected", "source_unavailable", "authored_note_required",
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
    """A Light consolidation request its route cannot hold (``consolidator.py``)."""


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
class _Selection:
    units: Tuple[_AtomicUnit, ...]
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

# --- Native image blocks (v6.26.0 multimodal chat) ---------------------------
# Char-equivalent for ONE image block in chars/4 token estimates (~1.1K tokens):
# vision models bill per tile, not per base64 char.
IMAGE_BLOCK_CHAR_EQUIVALENT = 4_400
# Initial image cap and batch-eviction target, across owner uploads, browser
# screenshots and transport injections. Running history grows to the high-water
# mark, then replaces oldest images with caption/re-view placeholders down to 5.
# 3 -> 5 (v6.81.1, owner decision 2026-07-29): with tool-result images now
# auto-attached (screenshots arrive every observation round), 3 kept too little
# visual history for compare-two-screens reasoning; 5 costs at most ~2.2K extra
# estimated tokens per request and only when that many images are actually live.
MAX_LIVE_IMAGE_BLOCKS = 5
# Owner decision 2026-10-21: batch eviction above 10 back to the 5-image floor of 2026-07-29.
LIVE_IMAGE_BLOCKS_HIGH_WATER = 10

# --- Scratchpad size thresholds (SSOT; previously scattered literals) -------
# Context-section soft budget for the rendered scratchpad (warn-only).
SCRATCHPAD_SECTION_BUDGET_CHARS = 90_000
# Context-section soft budget for the mind's world map (warn-only; `update_world` notices above it).
WORLD_SECTION_BUDGET_CHARS = 16_000
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
