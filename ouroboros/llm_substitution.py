"""Which account the next request of a route should NOT ask for, and why.

Two facts share this leaf because they answer one question. A per-subject
refusal (auth, quota, an unusable credential) says the account that just failed
should not be preferred again, and is spent by the next matching dispatch. A
served-model mismatch says the account that just answered did so with the WRONG
MODEL, and every redo of that call names no account at all. Neither ranks
accounts: the engine remains the only chooser, and these facts only stop the
transport from asking it for the same account again.

## Refusing a round that ANOTHER model answered

A subscription account can answer a request for one model with a different
model: no error, nothing in its quota, and the same account keeps doing it for
tens of minutes while its siblings serve the requested model normally. The
ENGINE states that as a typed fact on its completed result and deprioritises
that account for a while; this leaf owns what the caller does with the fact.

The answer is never the round. It stays retained and acknowledged like any
other answer, because the horizon a round nearly ran under is evidence, but it
adopts no turn state, runs no tool call, and is replaced by a NEW operation
that names no account, so the engine's own ranking decides where every redo
lands. Some rounds are refused outright instead: a pin names ONE account, an
already admitted candidate is bound to exact bytes a redo would change, and a
caller with no time or no send left cannot pay for another ask. When no redo is
left, the typed `model_substituted` refusal reaches the caller, whose
configured fallback chain owns what happens next.

The budget is its own counter, never the transport's no-start preparation
loop: a repair fixes a request that was never sent, while a redo spends a
fresh generation on an answer that already arrived.

## Asking again after a vendor refused one account's quota

The same question as a redo, for a refusal instead of an answer. The serving
engine (3.14.0) resolves ONE account per model operation before dispatch and
invokes it once, so a quota refusal before dispatch is the engine's own verdict
on its pool or a pin: it is never asked again, and only its ``poolCause`` says
that every compatible account is blocked. An HTTP-level vendor refusal is a
settled fact about the one account it names, refused before any response
stream began, so nothing was generated; the engine turns its reset evidence
into that account's cooldown. The SAME payload, naming no account, then lets
the engine choose again. An account it selects twice ends the rotation without
claiming the pool is spent, and so does every reason a redo is refused. A pin,
an unknown outcome and a stream-level refusal (generation may have begun) are
never asked again.
"""

from __future__ import annotations

import contextvars
import copy
from typing import Any

from ouroboros import config
from ouroboros.utils import append_jsonl, utc_now_iso

_FAILED_PROFILE = contextvars.ContextVar("claudexor_failed_profile", default=())
_PER_SUBJECT_REFUSALS = frozenset({
    "auth_required", "auth_refresh_failed", "credential_unusable", "provider_refused",
    "rate_limited", "subscription_window_exhausted",
})
_NON_PROVIDER_FAILURES = frozenset({
    "model_operation_cancelled", "model_operation_interrupted",
})


def failed_account_preference(target: dict, parameters: dict) -> str:
    """The account this route must not prefer right now, if one is remembered."""
    failed = _FAILED_PROFILE.get()
    key = (parameters.get("cache_affinity"), target["source"], target["resolved_model"])
    return failed[3] if len(failed) == 4 and failed[:3] == key else ""


def take_failed_account_preference(target: dict, parameters: dict) -> str:
    """Read that account and spend the fact: it names ONE next dispatch."""
    profile = failed_account_preference(target, parameters)
    if profile:
        _FAILED_PROFILE.set(())
    return profile


def remember_failed_profile(target: dict, parameters: dict, error: Any) -> None:
    if getattr(error, "stream_rejected", False):
        return  # Local message normalization says nothing about account readiness.
    route = error.route or {}
    key = (parameters.get("cache_affinity"), route.get("source"), route.get("model"))
    if (key == (parameters.get("cache_affinity"), target["source"], target["resolved_model"])
            and key[0] and route.get("credentialProfileId")
            and ((error.status_code == 0 and error.code not in _NON_PROVIDER_FAILURES)
                 or error.code in _PER_SUBJECT_REFUSALS)):
        _FAILED_PROFILE.set((*key, route["credentialProfileId"]))


def same_route_refusal(accumulated_usage: dict) -> bool:
    """Did this round already get a refusal that a repeat of it cannot cure?"""
    return (str(accumulated_usage.get("_last_llm_error_kind") or "") == "model_substituted"
            or str(accumulated_usage.get("_last_llm_provider_code") or "") == "invalid_continuation")


def stamp_substitutions(accumulated_usage: dict, error: Any) -> None:
    """Carry the discarded generations into the round's record for disclosure.

    A refusal names the horizon the round would have run under; an accepted
    answer whose redo recovered carries its own list on the usage row, which
    the disclosure owner reads there. Neither is the transport's business.
    """
    context = (getattr(error, "problem", {}) or {}).get("context") or {}
    if context.get("requested_model"):
        accumulated_usage["_model_substitutions"] = [{
            "requested": str(context.get("requested_model") or ""),
            "observed": str(context.get("observed_model") or ""),
            "account": str((getattr(error, "route", {}) or {}).get("credentialProfileId") or ""),
            "disposition": str(context.get("reason") or ""),
        }]


def substitution_fact(result: dict) -> dict | None:
    """The engine's typed fact that ANOTHER model answered this request.

    Only the engine compares model identity: the host never matches model
    strings of its own, so a dated snapshot, an alias or another transport
    cannot turn into a host-invented mismatch here.
    """
    fact = result.get("modelMismatch")
    if not isinstance(fact, dict):
        return None
    requested, observed = str(fact.get("requested") or ""), str(fact.get("observed") or "")
    return {"requested": requested, "observed": observed} if requested and observed else None


def _redo_allowed(payload: dict) -> str:
    """Empty when the same round may be asked again, else why it may not be.

    A pin names ONE account, so re-asking it is the same account and the same
    answer; the caller's configured model fallback owns that case. An admitted
    candidate is bound to exact bytes, and a redo sends no account preference,
    so re-asking would break that admission instead of honoring it.
    """
    from ouroboros.llm_attempt import physical_attempt_headroom
    from ouroboros.usage_accounting import current_physical_attempt_predicate

    if (payload.get("account") or {}).get("mode") == "pin":
        return "pinned_account"
    if current_physical_attempt_predicate() is not None:
        return "admitted_candidate"
    from ouroboros.model_wait import dispatch_deadline_remaining_sec

    remaining = dispatch_deadline_remaining_sec()
    if remaining is not None and remaining <= 0:
        # The round already spent its window. Another send would be paid for
        # out of time the caller no longer has.
        return "deadline_spent"
    headroom = physical_attempt_headroom()
    if headroom is not None and headroom < 1:
        # A bounded actor (a density probe, a packet review) owns those sends.
        # With none left, asking again would raise that rail's own budget error
        # instead of this refusal, and the caller would read the wrong cause.
        return "send_budget_spent"
    return ""


def _discard(invocation: Any, fact: dict, result: dict,
             disposition: str, redo: int, redos: int) -> dict:
    """Keep the paid generation reconstructible, then refuse it as this round's answer.

    The bytes stay retained and acknowledged exactly as an accepted answer's
    would be, because a discarded answer is still evidence of what the horizon
    was. What it does NOT do is become the round: no turn state is adopted from
    it, no tool call runs, and the caller builds the next request with no
    account preference, so the engine's selection is free to land elsewhere.
    """
    route = result.get("route") or {}
    usage = result.get("usage") or {}
    # A discarded answer is only evidence while its bytes are reachable. Say
    # what custody it actually has instead of assuming retention and the ACK
    # both worked; a durable row that hid a failed one would be a false receipt.
    custody = invocation.acknowledge() if invocation.response_ref else {"state": "absent"}
    recorded = append_jsonl(invocation.root / "logs" / "events.jsonl", {
        "ts": utc_now_iso(), "type": "model_served_mismatch", "task_id": invocation.task_id,
        "model_role": invocation.role, "operation_id": invocation.operation_id,
        "physical_attempt_id": invocation.invocation_id, "disposition": disposition,
        "redo": redo, "redos": redos, **fact, "route": copy.deepcopy(route),
        "discarded_usage": {key: usage.get(key) for key in
                            ("input_tokens", "output_tokens", "cached_input_tokens")},
        "result_custody": copy.deepcopy(custody),
    })
    return {**custody, **({} if recorded else {"event_recorded": False})}


class SubstitutionBudget:
    """One call's budget for re-asking a round that ANOTHER model answered.

    Its own counter, never the preparation loop's: a no-start repair fixes the
    request that was never sent, while this spends a fresh generation on an
    answer that already arrived. Both axes can occur in one call.
    """

    def __init__(self, error: type) -> None:
        self.error = error
        self.redos = config.get_model_substitution_redos()
        self.used = 0
        self.discarded: list[dict] = []

    def disclose(self, answer: tuple[dict, dict]) -> tuple[dict, dict]:
        """Carry the discarded generations into the accepted answer's usage row.

        The owner learns which horizon this round nearly ran under even when the
        redo recovered silently; money and tokens stay the ledger's, per attempt.
        """
        message, usage = answer
        if self.discarded and isinstance(usage.get("claudexor"), dict):
            usage["claudexor"]["substituted"] = copy.deepcopy(self.discarded)
        return message, usage

    def admit(self, invocation: Any, result: dict) -> bool:
        """True once this substituted generation is discarded and the round re-asked.

        False leaves an ordinary answer untouched; a round that may not be
        re-asked raises instead, so the other model's answer never becomes the
        caller's — whatever the reason the redo is unavailable.
        """
        fact = substitution_fact(result)
        if fact is None:
            return False
        reason = _redo_allowed(invocation.payload) or (
            "" if self.used < self.redos else "redos_exhausted")
        self.used += 1
        account = str((result.get("route") or {}).get("credentialProfileId") or "")
        self.discarded.append({**fact, "account": account, "disposition": reason or "redo"})
        custody = _discard(invocation, fact, result, reason or "redo", self.used, self.redos)
        self.discarded[-1]["result_custody"] = custody
        if reason:
            raise _refusal(self, invocation, fact, result, reason, self.used - 1)
        return True


def _refusal(budget: "SubstitutionBudget", invocation: Any, fact: dict,
             result: dict, reason: str, redos: int) -> Exception:
    """The typed refusal a caller sees once no further redo is available."""
    error = budget.error({
        "code": "model_substituted",
        "message": (f"The route answered with {fact['observed']} instead of the requested "
                    f"{fact['requested']}; the answer was not accepted."),
        "context": {"requested_model": fact["requested"], "observed_model": fact["observed"],
                    "reason": reason, "redos": redos},
    }, model_role=invocation.role, operation_id=invocation.operation_id,
        route=result.get("route") or {})
    if invocation.capture is not None:
        error.physical_attempt_capture = invocation.capture
    return error


class AccountRotation:
    """One call's unpinned re-asks of the same payload after a vendor quota refusal.

    A re-ask follows only a newly refused account, so the engine's pool bounds
    it; the ceiling is defensive and, like every other stop, claims nothing
    about the accounts that were not tried.
    """

    CEILING = 8

    def __init__(self) -> None:
        self.refused: list[str] = []

    def refused_call(self, target: dict, parameters: dict, invocation: Any, error: Any) -> bool:
        """Record one refusal; True only when this call may ask the engine again.

        A refusal that reaches the caller leaves its account unpreferred for the next
        matching dispatch; a re-ask names no account, so it remembers nothing.
        """
        from ouroboros.model_wait import model_wait_reason

        if model_wait_reason(error) not in {"quota", "auth_quota"}:
            remember_failed_profile(target, parameters, error)
            return False
        account = str((error.route or {}).get("credentialProfileId") or "")
        repeated = account in self.refused
        if account and not repeated:
            self.refused.append(account)
        state = getattr(getattr(error, "physical_attempt_capture", None), "state", None)
        stop = ("pinned_account" if (invocation.payload.get("account") or {}).get("mode") == "pin"
                else "pool_exhausted" if ((error.problem or {}).get("context") or {}).get("poolCause") in {"quota", "mixed"}
                else "refused_before_dispatch" if state == "released"
                else "outcome_not_settled" if state != "settled"
                else "generation_not_excluded" if not error.status_code
                else "account_unobserved" if not account
                else "engine_reselected_refused_account" if repeated
                else "rotation_ceiling" if len(self.refused) > self.CEILING
                else _redo_allowed(invocation.payload))
        append_jsonl(invocation.root / "logs" / "events.jsonl", {
            "ts": utc_now_iso(), "type": "model_account_rotation", "task_id": invocation.task_id,
            "model_role": invocation.role, "operation_id": invocation.operation_id, "account": account,
            "disposition": stop or "reask", "refused_accounts": list(self.refused)})
        if stop:
            remember_failed_profile(target, parameters, error)
            error.account_rotation = {"refused_accounts": list(self.refused), "stop": stop,
                                      "pool_exhausted": stop == "pool_exhausted"}
        return not stop

    @staticmethod
    def reask(parameters: dict, payload: dict) -> tuple[None, dict, dict]:
        """No preparation to rebind, and no account preference on any later request of this call."""
        return None, {**parameters, "_no_account_preference": True}, {**payload, "account": {"mode": "auto"}}

    def disclose(self, answer: tuple[dict, dict]) -> tuple[dict, dict]:
        """An answer that followed a rotation names the accounts that refused first."""
        message, usage = answer
        if self.refused and isinstance(usage.get("claudexor"), dict):
            usage["claudexor"]["account_rotation"] = {"refused_accounts": list(self.refused)}
        return message, usage
