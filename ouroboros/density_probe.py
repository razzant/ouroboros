"""The one bounded review calibration attempt when no exact-model witness exists."""
from __future__ import annotations

import logging
from typing import Any, Optional

from ouroboros.utils import estimate_tokens

log = logging.getLogger(__name__)

# Cold-start density probe (owner decisions R60/R61 for the packed deep
# self-review; the commit gate runs the same rung since 2026-09-05): when a
# review pack is refused or degraded under the COLD floor, ONE bounded send on
# the exact model sources a real witness for the store above.
# Room for a reasoning model to finish thinking AND answer: a cap that ends the
# call mid-reasoning comes back with no content and no usage — no witness.
DENSITY_PROBE_MAX_TOKENS = 256
DENSITY_PROBE_EFFORT = "low"
DENSITY_PROBE_SYSTEM_PROMPT = "Token-density calibration probe: reply with the single word OK."


def cold_start_density_probe(
    drive_root: Any,
    llm: Any,
    emit_progress: Any,
    model: str,
    sample: str,
    *,
    task_id: str,
    call_type: str,
    source: str,
    model_role: str = "", model_account_override: Optional[str] = None,
) -> str:
    """The cold-start density rung for the commit triad packet.
    Returns a typed outcome:

    ``"warm"`` — a fresh exact-model witness already governs: nothing is sent;
    ``"no_sample"`` — nothing to measure on; ``"failed"`` / ``"no_usage"`` /
    ``"unrecorded"`` — the probe sent but yielded no governing witness (the
    cold cap stands, disclosed on progress); ``"measured"`` — the witness is
    recorded and now governs, so the caller recomputes the cap and rebuilds
    ONCE.

    The calibrated input cap is the density-form bound over a FRESH exact-model
    witness, else the cold floor ``COLD_START_TOKEN_DENSITY`` — which lies
    above every measured density, so a repository whose required set fits warm
    is refused cold, and the refusal happens before any send, so the model
    never records the witness that would have admitted it. This rung breaks
    that loop with ONE bounded send on the exact model (``sample``: a slice of
    the real pack, a few output tokens) through the ordinary observed call,
    under ``physical_attempt_limit(1)`` so the transport ladder's own retries
    (a body-error reroute, an encrypted-reasoning strip) cannot turn the ONE
    send into several paid attempts. It never runs on a warm store and never
    retries. ``BudgetExceeded`` propagates:
    the paid ledger's refusal is budget vocabulary the caller discloses in its
    own terms (the deep review lets it reach the agent's budget rail; the
    commit gate records a typed disclosure and keeps its existing refusal)."""
    from ouroboros import capability_evidence as evidence
    from ouroboros.llm_observability import chat_observed
    from ouroboros.usage_accounting import BudgetExceeded, physical_attempt_limit

    _density, density_source = evidence.resolve_review_token_density(drive_root, model)
    if density_source == "measured":
        return "warm"
    if not sample:
        return "no_sample"
    emit_progress(
        f"No fresh token-density witness for {model}: one bounded probe "
        f"(~{estimate_tokens(sample):,} estimated tokens) calibrates the input cap..."
    )
    try:
        with physical_attempt_limit(1):
            _response, usage = chat_observed(
                llm,
                drive_root=drive_root,
                task_id=task_id,
                call_type=call_type,
                messages=[
                    {"role": "system", "content": DENSITY_PROBE_SYSTEM_PROMPT},
                    {"role": "user", "content": sample},
                ],
                model=model,
                model_role=model_role, model_account_override=model_account_override,
                tools=None,
                reasoning_effort=DENSITY_PROBE_EFFORT,
                max_tokens=DENSITY_PROBE_MAX_TOKENS,
                temperature=None,
                no_proxy=True,
            )
    except BudgetExceeded:
        raise
    except Exception as exc:
        from ouroboros.llm_claudexor import propagate_model_error
        propagate_model_error(exc)
        log.warning("Token-density probe failed (%s): %s", call_type, exc, exc_info=True)
        emit_progress(f"Density probe failed ({type(exc).__name__}); the cold input cap stands.")
        return "failed"
    real = int((usage or {}).get("prompt_tokens") or 0)
    if real <= 0:
        emit_progress("Density probe returned no usage (prompt_tokens=0); the cold input cap stands.")
        return "no_usage"
    actual_model = str(((usage or {}).get("model_role_route") or {}).get("model")
                       or (usage or {}).get("resolved_model") or model)
    evidence.record_token_density(
        drive_root,
        evidence._normalized_density_model(actual_model),
        prompt_chars=len(DENSITY_PROBE_SYSTEM_PROMPT) + len(sample),
        prompt_tokens=real,
        source=source,
    )
    if evidence._normalized_density_model(actual_model) != evidence._normalized_density_model(model):
        emit_progress(f"Density witness belongs to the switched model {actual_model}; the original pack cap is unchanged.")
        return "unrecorded"
    density, density_source = evidence.resolve_review_token_density(drive_root, model)
    emit_progress(f"Token density for {model}: {density:.2f} ({density_source}).")
    # A witness the store refused (too few chars, an insane ratio) leaves the
    # cold cap standing — disclosed above by the unchanged source.
    return "measured" if density_source == "measured" else "unrecorded"
