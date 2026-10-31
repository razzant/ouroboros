"""This conversation's own window facts: its size anchor, its accepted floor and its refused sizes.

Main measures every candidate on the conversation it belongs to. The last request a model
accepted in this task gives the provider-reported input U and the host's raw estimate B of
that same candidate; the next candidate is ``U + (B' - B) * U / B``. Before this task has an
anchor, ``context_fit`` uses fresh exact-route/model/view evidence only at comparable raw input size;
otherwise it stays cold (a 97k helper must not calibrate a 900k task). The largest
input a route accepted here is a lower bound for planning, never evidence of reply room. A size refusal clears predicted free room; only room below an accepted input
remains a lower bound. All live in the task's usage, so a working checkpoint keeps
them; a new execution has no task anchor. Data and arithmetic only; the loop records them.
"""

from __future__ import annotations

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

    ``input_tokens`` (U) is the provider-reported input of that request, ``estimate_tokens`` (B) the
    host's raw estimate of the same candidate. The next candidate measures ``U + (B' - B) * U / B``:
    nothing is borrowed from another conversation, whose composition and size differ. ``rewritten``
    says the transcript was rewritten since (a fold, a recovery pass): the same arithmetic, less sure.
    """

    input_tokens: int
    estimate_tokens: int
    round: int
    rewritten: bool = False

    @property
    def density(self) -> float:
        return self.input_tokens / self.estimate_tokens


def _anchor_model_key(plan: Any) -> str:
    from ouroboros.provider_models import normalize_model_identity
    from ouroboros.context_window import observation_scope

    return (f"{str(getattr(plan, 'provider', '') or '').strip().lower()}|"
            f"{normalize_model_identity(str(getattr(plan, 'model', '') or '').strip())}|"
            f"{observation_scope(plan)}")


def record_input_anchor(usage: Dict[str, Any], plan: Any, *, round_idx: int, rewrite_epoch: int = 0) -> None:
    """Make the Main request just accepted this conversation's size anchor and raise its accepted floor.

    U is the adapter-normalized input of the settled response (``_last_round_cache_usage``), B the raw
    estimate measured for that candidate (``_context_raw_input_tokens``). A response another model
    served, an input total whose cache semantics are unknown, a gateway total that counts the cache read
    twice, a count an order of magnitude away from the raw estimate of the same bytes, or a missing count
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
    real, estimate = last.get("prompt_tokens"), usage.get("_context_raw_input_tokens")
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
                    "rewrite_epoch": int(rewrite_epoch)}
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
                             rewritten=rewrite_epoch is None or int(row.get("rewrite_epoch") or 0) != rewrite_epoch)
    return anchor, (int(floor) if floor else None)


def record_refused_input(usage: Dict[str, Any], plan: Any, estimate_tokens: int) -> None:
    """Remember the smallest input estimate ``plan``'s model refused by size in this task."""
    if plan is None or int(estimate_tokens or 0) <= 0:
        return
    from ouroboros.context_window import record_refusal

    record_refusal(usage, plan)
    refused = usage.setdefault(REFUSED_INPUTS_KEY, {})
    key = _anchor_model_key(plan)
    refused[key] = min(int(refused.get(key) or estimate_tokens), int(estimate_tokens))


def refused_input(usage: Optional[Mapping[str, Any]], plan: Any) -> Optional[int]:
    """The smallest estimate ``plan``'s model refused here, or None."""
    if not isinstance(usage, Mapping) or plan is None:
        return None
    value = (usage.get(REFUSED_INPUTS_KEY) or {}).get(_anchor_model_key(plan))
    return int(value) if value else None
