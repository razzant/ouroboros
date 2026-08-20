"""Scope-review prompt budget: how large a pack may be, and how it is sized.

The reviewer window and the measured tokenizer density decide the per-call input
cap for the assembled scope pack; the same numbers decide the output reserve the
call may ask for and classify a provider size rejection after the fact. The
constitutional >=1M window, the unevidenced-route sub-floor, and the designated
reviewer identity remain owned by `tools/scope_window`.
"""

from __future__ import annotations

import os

from ouroboros.tools.scope_window import scope_window as _scope_window

# Shipped designated scope reviewer (v6.82.0). Window evidence checked 2026-07-29:
# provider docs AND OpenRouter /models both state gpt-5.6-terra context_length
# 1,050,000 — a documented MODEL property, so the >=1M BIBLE P3 floor holds on both
# spellings; the sentinel grants only 1M, a real probe/owner-ack supersedes.
from ouroboros.tools.scope_window import SCOPE_MODEL_DEFAULT as _SCOPE_MODEL_DEFAULT  # noqa: E402
_SCOPE_MAX_TOKENS = 100_000  # 100K output tokens
_SCOPE_REVIEW_SLOT_TIMEOUT_SEC = 900
from ouroboros.tools.review_helpers import REVIEW_PROMPT_TOKEN_BUDGET as _SCOPE_BUDGET_TOKEN_LIMIT

# The shared prompt-size SSOT (920K) governs INPUT only; the reviewer also reserves
# _SCOPE_MAX_TOKENS of OUTPUT inside the same 1M window, and provider tokenizers can
# exceed estimate_tokens on atlas-heavy prompts — so gate assembled INPUT on a
# conservative effective cap and retry once with a compact atlas before applying the
# blocking/advisory scope authority. The 1M constitutional window, unevidenced-route
# sub-floor, and default reviewer identity live in `tools/scope_window` (the SSOT).
from ouroboros.tools.scope_window import (  # noqa: E402
    SCOPE_FAILCLOSED_WINDOW as _SCOPE_FAILCLOSED_WINDOW,
    SCOPE_MODEL_CONTEXT_WINDOW as _SCOPE_MODEL_CONTEXT_WINDOW,
)
_SCOPE_OUTPUT_MARGIN_TOKENS = 155_000
_SCOPE_INPUT_TOKEN_LIMIT = min(
    _SCOPE_BUDGET_TOKEN_LIMIT,
    _SCOPE_MODEL_CONTEXT_WINDOW - _SCOPE_MAX_TOKENS - _SCOPE_OUTPUT_MARGIN_TOKENS,
)

# Tokenizer-density calibration (SSOT: review_helpers.calibrated_input_token_limit +
# capability_evidence ``token_density``). Density is MEASURED per model, so the limit
# is computed PER CALL (an import-time constant froze the pre-measurement value). The
# calibration shrinks the PROMPT — never the reviewer or the >=1M floor (BIBLE P3).
from ouroboros.reviewer_window import window_scaled_reserves as _shared_window_scaled_reserves  # noqa: E402
from ouroboros.tools.review_helpers import (  # noqa: E402
    calibrated_input_token_limit as _calibrated_input_token_limit,
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
    model = scope_model or _get_scope_model()
    window = _scope_window(model).sizing_window(_SCOPE_FAILCLOSED_WINDOW)
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


# Provider-oversize fault classification moved to triad_review (shared review
# primitive); the alias keeps this module's historical name for its two readers.
from ouroboros.triad_review import is_provider_oversize_error as _is_provider_oversize_error  # noqa: E402


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
        input_limit = int(_effective_scope_input_limit(scope_model=scope_model) or 0)
    except Exception:
        input_limit = 0
    return input_limit > 0 and int(prompt_tokens_est or 0) >= int(0.8 * input_limit)
