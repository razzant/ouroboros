"""Scope-review budget math: token limits, reserves, and oversize classification.

Owns the scope reviewer's input-fit arithmetic — the output/margin reserves, the
import-time conservative input cap, the per-call calibrated effective limit, the
configured-reviewer resolution, and the gateway-route oversize classifier.
Extracted from ouroboros/tools/scope_review.py (v7 D06 split, re-derived on the
v7next tip AFTER PR #383 rewrote the pack arithmetic — reference bodies were not
reused); scope_review.py re-exports every name. The private aliases for the
scope_window / review_helpers / reviewer_window / triad_review owners are
rebound here (the budget owner that reads them) and re-exported by the parent,
exactly as before the split: they are import-time-frozen on both sides, so no
patch-visibility changes. ``_SCOPE_REVIEW_SLOT_TIMEOUT_SEC`` keeps the tip's
``None`` (the adaptive-timeout contract retired the reference-era constant).
"""

from __future__ import annotations

import os

# The parent's private aliases move with their budget readers; the canonical
# owners stay scope_window / review_helpers / reviewer_window / triad_review.
from ouroboros.tools.scope_window import (
    SCOPE_FAILCLOSED_WINDOW as _SCOPE_FAILCLOSED_WINDOW,
    SCOPE_MODEL_CONTEXT_WINDOW as _SCOPE_MODEL_CONTEXT_WINDOW,
    SCOPE_MODEL_DEFAULT as _SCOPE_MODEL_DEFAULT,
)
from ouroboros.tools.review_helpers import (
    REVIEW_PROMPT_TOKEN_BUDGET as _SCOPE_BUDGET_TOKEN_LIMIT,
    calibrated_input_token_limit as _calibrated_input_token_limit,
)
from ouroboros.reviewer_window import (
    window_scaled_reserves as _shared_window_scaled_reserves,
)
from ouroboros.triad_review import (
    is_provider_oversize_error as _is_provider_oversize_error,
)


def _sr():
    """The parent scope-review module, read at call time.

    The budget members stay monkeypatch-addressable at their historical
    ``ouroboros.tools.scope_review`` bindings (tests rebind them there), so
    this leaf resolves every such cross-reference through the module at each
    call instead of freezing whatever object a from-import saw at import time.
    """
    from ouroboros.tools import scope_review

    return scope_review


_SCOPE_MAX_TOKENS = 100_000  # 100K output tokens


_SCOPE_REVIEW_SLOT_TIMEOUT_SEC = None


_SCOPE_OUTPUT_MARGIN_TOKENS = 155_000


_SCOPE_INPUT_TOKEN_LIMIT = min(
    _SCOPE_BUDGET_TOKEN_LIMIT,
    _SCOPE_MODEL_CONTEXT_WINDOW - _SCOPE_MAX_TOKENS - _SCOPE_OUTPUT_MARGIN_TOKENS,
)


def _window_scaled_reserves(window: int) -> tuple:
    """(output_reserve, tokenizer_margin) scaled to the reviewer window.

    The absolute 1M-calibrated reserves (100K output + 155K margin) would
    swallow a small window whole (gigachat 131K => input limit 0, bricking the
    slot — Provider Independence). Sub-floor windows scale the reserves to the
    window instead: a quarter for output (floored at 8K so the reviewer can
    still produce the full checklist JSON) and an eighth for tokenizer margin.
    >=1M windows keep the absolute reserves unchanged.
    """
    return _shared_window_scaled_reserves(
        window,
        output_reserve=_SCOPE_MAX_TOKENS,
        tokenizer_margin=_SCOPE_OUTPUT_MARGIN_TOKENS,
    )


def _effective_scope_input_limit(*, scope_model: str = "") -> int:
    """Scope input token cap for the configured reviewer, computed PER CALL.

    Two axes: the model's MEASURED tokenizer density sizes the prompt for its real
    tokenizer, and a KNOWN reviewer window (Capability Evidence, not a static table)
    replaces the assumed 1M so a small-window reviewer gets a fit-sized pack instead
    of a deterministic provider 400. Its blocking authority is checked separately and
    stays fail-closed."""
    model = scope_model or _sr()._get_scope_model()
    window = _sr()._scope_window(model).sizing_window(_SCOPE_FAILCLOSED_WINDOW)
    output_reserve, tokenizer_margin = _window_scaled_reserves(window)
    return max(0, _calibrated_input_token_limit(
        model,
        context_window=window,
        output_reserve=output_reserve,
        tokenizer_margin=tokenizer_margin,
        budget_cap=_SCOPE_BUDGET_TOKEN_LIMIT,
    ))


def _get_scope_model() -> str:
    """Return the configured scope review model (env → settings default)."""
    try:
        from ouroboros.config import get_scope_review_models

        models = get_scope_review_models()
        if models:
            return models[0]
    except Exception:
        pass
    return os.environ.get("OUROBOROS_SCOPE_REVIEW_MODEL", "").strip() or _SCOPE_MODEL_DEFAULT


def _provider_error_is_oversize(usage: dict, prompt_tokens_est: int, scope_model: str) -> bool:
    """Gateway-route oversize detection from ``usage['provider_error']``."""
    pe = usage.get("provider_error") if isinstance(usage, dict) else None
    if not isinstance(pe, dict):
        return False
    try:
        code = int(pe.get("code") or 0)
    except (TypeError, ValueError):
        code = 0
    if code != 400:  # never 429/5xx (already rerouted as transient), never non-400
        return False
    # Non-empty 400 messages must explicitly say oversize; only opaque gateway 400s can
    # use size proximity, so auth/param/policy errors stay fail-closed.
    message = str(pe.get("message") or "").strip()
    if message:
        return _is_provider_oversize_error(message)
    try:
        input_limit = int(_sr()._effective_scope_input_limit(scope_model=scope_model) or 0)
    except Exception:
        input_limit = 0
    return input_limit > 0 and int(prompt_tokens_est or 0) >= int(0.8 * input_limit)
