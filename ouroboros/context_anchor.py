"""This conversation's own window facts: its size anchor, its accepted floor and its refused sizes.

Main measures every candidate on the conversation it belongs to. The last request a model
accepted in this task gives the provider-reported input U, the estimator count B of the bytes
it was sent and the estimate P of the transcript they were prepared from (a captioned image
differs); the next candidate is ``U + (P' - P) * U / B``. Before this task has an
anchor, ``context_fit`` uses fresh exact-route/model/view evidence only at comparable raw input size;
otherwise it stays cold (a 97k helper must not calibrate a 900k task). The largest
input a route accepted here is a lower bound for planning, never evidence of reply room. A size refusal clears predicted free room; only room below an accepted input
remains a lower bound. All live in the task's usage, so a working checkpoint keeps
them; a new execution has no task anchor. Data and arithmetic only; the loop records them.
A refused request's own density sizes only its recovery; dated measurements retain it, but it is not learned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

#: This conversation's own size anchors (model key -> its last accepted request) and accepted-input floors in
#: the task's usage (both survive a working checkpoint; a new execution starts without them). Each model keeps
#: its own anchor: a fallback's acceptance never erases the primary's, so a return to it stays measured.
INPUT_ANCHOR_KEY = "_context_input_anchor"
ACCEPTED_INPUTS_KEY = "_context_accepted_inputs"
REFUSED_INPUTS_KEY = "_context_refused_inputs"  # model key -> the smallest estimate it refused in this task


@dataclass(frozen=True)
class InputAnchor:
    """The last request this model accepted in this conversation.

    ``input_tokens`` (U) is the provider-reported input of that request, ``estimate_tokens`` (B) the fit
    estimator's count of the bytes it was sent (prepared images, clock line and wire form included),
    ``transcript_tokens`` (P) the round's estimate of the transcript they were prepared from (None: B).
    ``U / B`` prices any final candidate; the next transcript measures ``U + (P' - P) * U / B``, a rebind's
    view too (``view_terms``): nothing is borrowed from another conversation, whose composition and size
    differ. ``rewritten`` says the transcript was rewritten since (a fold, a recovery pass): the same
    arithmetic, less sure.
    """

    input_tokens: int
    estimate_tokens: int
    round: int
    rewritten: bool = False
    transcript_tokens: Optional[int] = None

    @property
    def density(self) -> float:
        return self.input_tokens / self.estimate_tokens

    def measure(self, transcript_tokens: int) -> int:
        """``U + (P' - P) * U / B`` rounded up, exactly: U carried with the transcript's growth since."""
        since = (self.transcript_tokens or self.estimate_tokens) - int(transcript_tokens)
        return max(0, self.input_tokens - since * self.input_tokens // self.estimate_tokens)

    def view_terms(self, transcript_tokens: int) -> Tuple[float, int]:
        """Physical density and prepared-input estimate for a rebind's existing measurement seam.

        Preparation contributes the signed offset B - P; each removed transcript token still saves U / B.
        This gives the same affine estimate as ``measure`` for kept and narrowed views, without inventing a
        different tokenizer density when preparation removes an image or shortens its caption.
        """
        prepared = int(transcript_tokens) + self.estimate_tokens - (self.transcript_tokens or self.estimate_tokens)
        return self.density, max(0, prepared)


def _anchor_model_key(plan: Any) -> str:
    from ouroboros.provider_models import normalize_model_identity
    from ouroboros.context_window import observation_scope

    return (f"{str(getattr(plan, 'provider', '') or '').strip().lower()}|"
            f"{normalize_model_identity(str(getattr(plan, 'model', '') or '').strip())}|"
            f"{observation_scope(plan)}")


def record_input_anchor(usage: Dict[str, Any], plan: Any, *, round_idx: int, rewrite_epoch: int = 0,
                        estimate_tokens: Optional[int]) -> None:
    """Make the Main request just accepted this conversation's size anchor and raise its accepted floor.

    U is the adapter-normalized input of the settled response (``_last_round_cache_usage``), B
    (``estimate_tokens``) the fit estimator's count of the bytes that response's own ledger receipt names:
    the candidate the admission measured, so a caption or withheld image counts once. P is the round's
    estimate of the transcript (``_context_raw_input_tokens``). A response another model served, an input
    total whose cache semantics are unknown, a gateway total that counts the cache read twice, a count an
    order of magnitude away from the raw estimate of the same bytes, or a missing count or B
    leaves the previous anchor and floor as they were: unknown stays unknown.
    """
    from ouroboros.capability_evidence import _CACHE_INCLUSIVE_PROMPT_TOKEN_PROVIDERS, gateway_double_counted_cache

    if plan is None or usage.get("_model_substitutions"):
        return
    from ouroboros.context_window import observed_plan
    plan = observed_plan(plan, usage)
    if plan is None or not getattr(plan, "window_witness_bound", True):
        return
    last = usage.get("_last_round_cache_usage") or {}
    real, estimate, transcript = last.get("prompt_tokens"), estimate_tokens, usage.get("_context_raw_input_tokens")
    if not isinstance(real, int) or not isinstance(estimate, int) or real <= 0 or estimate <= 0:
        return
    if not estimate / 10 <= real <= estimate * 10:
        return  # an order of magnitude off the same bytes is no input total of this request (a delta, a stub)
    provider = str(last.get("provider") or getattr(plan, "provider", "") or "").strip().lower()
    cache_bearing = bool(last.get("cached_tokens") or last.get("cache_write_tokens"))
    if cache_bearing and provider not in _CACHE_INCLUSIVE_PROMPT_TOKEN_PROVIDERS:
        return
    if gateway_double_counted_cache(real, int(last.get("cached_tokens") or 0)):
        return  # twice the real input would double every estimate and lift the floor above the window for good
    key = _anchor_model_key(plan)
    anchors = usage.get(INPUT_ANCHOR_KEY)
    anchors = dict(anchors) if isinstance(anchors, dict) and "model" not in anchors else {}
    anchors[key] = {"input_tokens": real, "estimate_tokens": estimate, "round": int(round_idx),
                    "rewrite_epoch": int(rewrite_epoch),
                    **({"transcript_tokens": transcript} if isinstance(transcript, int) and transcript > 0 else {})}
    usage[INPUT_ANCHOR_KEY] = anchors
    floors = usage.setdefault(ACCEPTED_INPUTS_KEY, {})
    floors[key] = max(int(floors.get(key) or 0), real)
    if getattr(plan, "route_fp", "") and getattr(plan, "window_witness_bound", True):
        from ouroboros.capability_evidence import canonical_evidence_root, record_token_density, MAIN_DENSITY_BASIS
        from ouroboros.context_window import observation_scope, record_observation
        from ouroboros.provider_models import normalize_model_identity
        root = canonical_evidence_root()
        record_observation(root, observation_scope(plan), accepted=real, kind="input",
                           reserve=int(usage.get("_context_response_reserve_tokens", plan.output_reserve_tokens) or 0),
                           transmitted=int(usage.get("_context_reply_allowance_tokens") or 0))
        record_token_density(root, normalize_model_identity(plan.model), prompt_chars=estimate * 4,
                             prompt_tokens=real, route_fp=plan.route_fp, basis=MAIN_DENSITY_BASIS,
                             view_kind=getattr(plan, "view_kind", ""))


def main_input_anchor(usage: Optional[Mapping[str, Any]], plan: Any, *,
                      rewrite_epoch: Optional[int] = None) -> Tuple[Optional[InputAnchor], Optional[int]]:
    """``(anchor, accepted floor)`` of ``plan``'s model in this conversation, or ``(None, None)``.

    A different model or provider (a fallback, an owner switch) has no anchor here; an account or server
    rotation has its own route fingerprint and inherits neither anchor nor floor. ``rewrite_epoch`` is the slot's current count of
    sanctioned rewrites (``transcript_prefix.rewrite_epoch``); None treats the view as rewritten.
    """
    if not isinstance(usage, Mapping) or plan is None or not getattr(plan, "window_witness_bound", True):
        return None, None
    key = _anchor_model_key(plan)
    floor = (usage.get(ACCEPTED_INPUTS_KEY) or {}).get(key)
    anchors = usage.get(INPUT_ANCHOR_KEY)
    row = anchors.get(key) if isinstance(anchors, Mapping) else None
    anchor = None
    if isinstance(row, Mapping):
        anchor = InputAnchor(input_tokens=int(row["input_tokens"]), estimate_tokens=int(row["estimate_tokens"]),
                             round=int(row.get("round") or 0),
                             rewritten=rewrite_epoch is None or int(row.get("rewrite_epoch") or 0) != rewrite_epoch,
                             transcript_tokens=int(row.get("transcript_tokens") or row["estimate_tokens"]))
    return anchor, (int(floor) if floor else None)


def record_refused_input(usage: Dict[str, Any], plan: Any, estimate_tokens: int, *, measured: bool = False) -> None:
    """Remember the smallest input estimate ``plan``'s model refused by size in this task.

    ``measured`` (the estimate is anchored on an accepted request) also bounds the refusing account's
    window when the provider stated no limit; a cold estimate stays task-local."""
    if plan is None or int(estimate_tokens or 0) <= 0:
        return
    from ouroboros.context_window import record_refusal

    record_refusal(usage, plan, refused_input=int(estimate_tokens) if measured else 0)
    refused = usage.setdefault(REFUSED_INPUTS_KEY, {})
    key = _anchor_model_key(plan)
    refused[key] = min(int(refused.get(key) or estimate_tokens), int(estimate_tokens))


def refused_input(usage: Optional[Mapping[str, Any]], plan: Any) -> Optional[int]:
    """The smallest estimate ``plan``'s model refused here, or None."""
    if not isinstance(usage, Mapping) or plan is None:
        return None
    value = (usage.get(REFUSED_INPUTS_KEY) or {}).get(_anchor_model_key(plan))
    return int(value) if value else None


def refused_request_density(usage: Optional[Mapping[str, Any]], refused: Any, measurement: Any) -> Optional[float]:
    """Provider tokens per estimator token of the request a provider just refused by size, or None.

    The provider's own stated input of that request (``_last_llm_provider_message``, stamped for this very
    failure) over the fit estimator's count of the bytes the same attempt sent (its capture's
    ``prompt_tokens_bounded_estimate``: the pair the dispatch-usage witness reads for an accepted request).
    A bare total, a refusal stating no count, a pre-dispatch or exact-local candidate, a capture without that
    count, or a route the round's ``measurement`` no longer measures (a rebind) measures nothing. Recovery
    receipts label it as refused input; it becomes neither an accepted anchor nor learned route density.
    """
    from ouroboros.capability_evidence import _parse_ctx_requested

    stated = _parse_ctx_requested(str((usage or {}).get("_last_llm_provider_message") or ""))[0]
    estimate = int(getattr(refused, "prompt_tokens_bounded_estimate", 0) or 0)
    if stated <= 0 or estimate <= 0:
        return None
    sent = getattr(getattr(refused, "physical_context", None), "route_fp", None)
    return stated / estimate if sent and sent == getattr(measurement, "route_fp", None) else None


def measured_units(refused_density: Optional[float], measurement: Any,
                   *tokens: Optional[int]) -> Tuple[Optional[int], ...]:
    """``tokens`` the refused request's own density counts, in the units ``measurement`` counts; None stays unknown."""
    scale = measurement.measurement_density / refused_density if refused_density else 1.0
    return tuple(None if value is None else math.ceil(value * scale) if refused_density else int(value)
                 for value in tokens)
