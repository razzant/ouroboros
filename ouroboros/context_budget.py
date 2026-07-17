"""Single source of truth for AGENT-context size budgets.

These govern the size of Ouroboros's OWN working context: the main-loop
assembled prompt, tool-history compaction triggers, and the background
consciousness context guards.

They are deliberately SEPARATE from the REVIEW-prompt budget family
(``ouroboros.tools.review_helpers.REVIEW_PROMPT_TOKEN_BUDGET`` and the
``ouroboros.tools.scope_review`` window constants), which sizes reviewer
prompts, not the agent's own context. Merging the two would couple unrelated
concerns and is explicitly avoided.

Constants only (no functions) so this module stays free against the codebase
function-count gate (``ouroboros.review.MAX_TOTAL_FUNCTIONS``). Profile-keyed
resolution (the low/max context modes) is layered on later without renaming
these constants.

Char-based guards assume the ~chars/4 estimate (``ouroboros.utils.estimate_tokens``);
the comments give the approximate token equivalents.
"""

from __future__ import annotations

# Main-loop emergency tool-history compaction trigger (~300K tokens at chars/4).
# Remote routine compaction stays off by design; this is the overflow backstop.
EMERGENCY_COMPACTION_CHARS = 1_200_000

# Proactive compaction trigger (~200K tokens at chars/4). Reserved for
# low-mode or local-context scenarios where proactive shrinking is safe.
# Max mode uses emergency-only compaction at 1.2M chars (BIBLE P1).
PROACTIVE_COMPACTION_CHARS = 800_000

# Background-consciousness assembled-context guards. P1: fail fast, never
# silently truncate cognitive artifacts.
BG_CONTEXT_WARN_CHARS = 600_000   # ~150K tokens: warn but proceed
BG_CONTEXT_MAX_CHARS = 1_200_000  # ~300K tokens: skip the wakeup cycle

# Drive-state JSON injection guard inside the consciousness context.
BG_STATE_JSON_WARN_CHARS = 200_000

# WARN threshold for a single oversized governance/knowledge context section.
LARGE_CONTEXT_SECTION_CHARS = 200_000

# Main-loop assembled-context soft cap (tokens). A no-op recorder
# (P1 no-silent-truncation); the live transcript is bounded by compaction below.
CONTEXT_SOFT_CAP_TOKENS = 200_000

# --- Low-profile (≈200K window / local models) overrides -------------------
# These tighten the live working set in low context mode. They never shorten the
# memory HORIZON: recent dialogue is only coarsened when older dialogue is
# already represented by valid consolidation, and tool-history transcript
# compaction persists a forensic checkpoint before summarizing.

# Raw recent-dialogue tail shown when no valid consolidation can represent older
# dialogue. Low mode keeps this horizon rather than silently shortening it.
MAX_RECENT_CHAT_TAIL = 1000

# Low fires emergency tool-history compaction earlier (~100K tokens at chars/4)
# to fit a ~200K window, and (unlike max) also enables remote routine compaction.
# The owner low/max context MODE is the SSOT for the agent's own operating
# window (v6.33.0 BIBLE P1): low => this 400K trigger + routine compaction; max
# => the 1.2M emergency-only trigger. There is no per-model window table; the
# reactive provider-overflow detector (context.py) is the safety net if a route's
# real window is smaller than the mode assumes.
LOW_EMERGENCY_COMPACTION_CHARS = 400_000

# --- Native image blocks (v6.26.0 multimodal chat) ---------------------------
# Char-equivalent for ONE image block in chars/4 token estimates (~1.1K tokens):
# vision models bill per tile, not per base64 char.
IMAGE_BLOCK_CHAR_EQUIVALENT = 4_400
# Live image blocks kept in the transcript (single counter across owner
# uploads, browser screenshots, and transport injections). Older images are
# replaced by a caption placeholder pointing to the re-view path.
MAX_LIVE_IMAGE_BLOCKS = 3

# --- Scratchpad size thresholds (SSOT; previously scattered literals) -------
# Context-section soft budget for the rendered scratchpad (warn-only).
SCRATCHPAD_SECTION_BUDGET_CHARS = 90_000
# Health-invariant bloat warning ("extract durable insights to knowledge").
SCRATCHPAD_BLOAT_WARN_CHARS = 50_000
# Block-storage consolidation trigger (consolidator compresses oldest blocks).
SCRATCHPAD_CONSOLIDATION_THRESHOLD_CHARS = 30_000
