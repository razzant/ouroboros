"""Price live receipts by modality in the ordinary physical-attempt ledger.

A response is reserved when its start event arrives and is already dispatched.
The caller's UsageScope owns root/group attribution and money limits. Catalog
quotes are estimates, never provider invoices; an uncovered modality keeps the
whole amount unknown, and a receipt the provider never reported settles as
unknown, never as a free response. Abandonment retains physical custody as
unresolved; a response the budget refuses after the provider started it keeps
the provider's charge on the ledger. The catalog is warmed by the call's thread
before its event loop exists (``warm_quote``); every lookup made from inside the
loop reads the warmed snapshot and never fetches, so no network wait ever sits
on the audio path.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Mapping

from ouroboros import usage_accounting as accounting
from ouroboros.contracts.realtime import USAGE_AXES

USAGE_KEYS = tuple(key for key in USAGE_AXES if key not in {"connected_seconds", "thinking_tokens"})
from ouroboros.pricing_catalog import RATE_FIELDS, get_catalog_quote, rate

# A provisional accounting unit, not a generation limit or a predicted utterance.
# Each input/output modality keeps its own tariff; no token is priced as text
# merely because a response's physical size is not yet known at ResponseStarted.
PROVISIONAL_USAGE = {key: 1024 for key in USAGE_KEYS}
PROVISIONAL_USAGE.update(cached_input_tokens=0, thinking_tokens=0)


@dataclass(frozen=True)
class LiveAttempt:
    """Opaque handle retaining the response's reservation and original quote."""

    _reservation: accounting.AttemptReservation
    _quote: dict | None
    _evidence: dict


def _count(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token count must be a nonnegative integer")
    return value


def _raw_body(raw: Mapping) -> Mapping:
    for key in ("usage", "usageMetadata", "usage_metadata"):
        if isinstance(raw.get(key), Mapping):
            return raw[key]
    return raw


def input_token_witness(raw_usage: Mapping) -> int | None:
    """Provider input total, including cache once; neither output nor an estimate.

    An absent modality split cannot calibrate text density, but the provider's
    aggregate input count still witnesses the input it accepted.
    """
    raw = _raw_body(raw_usage)
    totals = [raw[key] for key in ("input_tokens", "promptTokenCount", "prompt_token_count") if key in raw]
    if totals and all(type(value) is int and value > 0 and value == totals[0] for value in totals):
        return totals[0]
    return None


def text_input_witness(usage: Mapping, raw_usage: Mapping) -> int | None:
    """Only an explicit text modality split can calibrate selected text."""
    text = usage.get("text_input_tokens")
    if type(text) is not int or text <= 0:
        return None
    raw = _raw_body(raw_usage)
    for key in ("input_token_details", "input_tokens_details"):
        details = raw.get(key)
        if isinstance(details, Mapping) and type(details.get("text_tokens")) is int and details["text_tokens"] == text:
            return text
    for key in ("promptTokensDetails", "prompt_tokens_details"):
        details = raw.get(key)
        if isinstance(details, list):
            values = [part.get("tokenCount", part.get("token_count")) for part in details
                      if isinstance(part, Mapping) and str(part.get("modality", "")).upper() == "TEXT"]
            if values and all(type(value) is int and value >= 0 for value in values) and sum(values) == text:
                return text
    return None


def _cached_split(counts: dict, raw: Mapping) -> dict:
    """The aggregate cache count needs a modality witness when inputs are mixed."""
    total = counts["cached_input_tokens"]
    modalities = ("text", "audio", "image", "video")
    split = None
    details = raw.get("input_token_details", raw.get("input_tokens_details", {}))
    if isinstance(details, Mapping) and "cached_tokens_details" in details:
        cached = details["cached_tokens_details"]
        if not isinstance(cached, Mapping):
            raise ValueError("malformed cached modality details")
        split = {m: _count(cached.get(f"{m}_tokens", 0)) for m in modalities}
        if any(key not in {f"{m}_tokens" for m in modalities} and value != 0 for key, value in cached.items()):
            raise ValueError("unpriced cached modality")
    for key in ("cacheTokensDetails", "cache_tokens_details"):
        if key in raw:
            if not isinstance(raw[key], list):
                raise ValueError("malformed cached modality details")
            split = {m: 0 for m in modalities}
            for item in raw[key]:
                if not isinstance(item, Mapping):
                    raise ValueError("malformed cached modality details")
                modality = str(item.get("modality", "")).lower()
                number = _count(item.get("tokenCount", item.get("token_count", 0)))
                if modality not in split:
                    if number:
                        raise ValueError("unpriced cached modality")
                else:
                    split[modality] += number
    if split is None:
        active = [m for m in modalities if counts[f"{m}_input_tokens"]]
        if total and len(active) != 1:
            raise ValueError("cached input modality is unknown")
        split = {m: total if active == [m] else 0 for m in modalities}
    if sum(split.values()) != total or any(split[m] > counts[f"{m}_input_tokens"] for m in modalities):
        raise ValueError("cached input split disagrees with reported totals")
    return split


def _unpriced_raw(raw: Mapping) -> list[str]:
    missing = []
    for key in ("unpriced_components", "toolUsePromptTokenCount", "tool_use_prompt_token_count",
                "groundingMetadata", "grounding_metadata", "server_tool_use"):
        if raw.get(key):
            missing.append(key)
    for key, supported in (("promptTokensDetails", {"TEXT", "AUDIO", "IMAGE", "VIDEO"}),
                           ("responseTokensDetails", {"TEXT", "AUDIO"}),
                           ("prompt_tokens_details", {"TEXT", "AUDIO", "IMAGE", "VIDEO"}),
                           ("response_tokens_details", {"TEXT", "AUDIO"})):
        if key not in raw:
            continue
        if not isinstance(raw[key], list):
            missing.append(key)
            continue
        for item in raw[key]:
            if (not isinstance(item, Mapping) or
                    (str(item.get("modality", "")).upper() not in supported
                     and item.get("tokenCount", item.get("token_count")) != 0)):
                missing.append(key)
    return missing


def _raw_count_gaps(counts: dict, raw: Mapping) -> list[str]:
    """An adapter cannot erase a reported modality by omitting its normalized count."""
    missing = []
    for key, direction in (("input_token_details", "input"), ("input_tokens_details", "input"),
                           ("output_token_details", "output"), ("output_tokens_details", "output")):
        details = raw.get(key)
        if not isinstance(details, Mapping):
            continue
        for field, value in details.items():
            if field in {"cached_tokens", "cached_tokens_details", "reasoning_tokens"}:
                continue
            if field.endswith("_tokens"):
                number = _count(value)
                modality = field.removesuffix("_tokens")
                if number != counts.get(f"{modality}_{direction}_tokens", 0):
                    missing.append(f"{key}.{field}")
    for key, direction in (("promptTokensDetails", "input"), ("responseTokensDetails", "output"),
                           ("prompt_tokens_details", "input"), ("response_tokens_details", "output")):
        details = raw.get(key)
        if not isinstance(details, list):
            continue
        reported = {}
        for item in details:
            if isinstance(item, Mapping):
                modality = str(item.get("modality", "")).lower()
                number = _count(item.get("tokenCount", item.get("token_count")))
                reported[modality] = reported.get(modality, 0) + number
        for modality, number in reported.items():
            if number != counts.get(f"{modality}_{direction}_tokens", 0):
                missing.append(f"{key}.{modality}")
    return missing


def _thoughts_tokens(raw: Mapping) -> int:
    """Gemini bills its thinking as output tokens, outside the response modality details."""
    return _count(raw.get("thoughtsTokenCount", raw.get("thoughts_token_count", 0)))


def _price(quote: dict | None, usage: Mapping, raw_usage: Mapping) -> tuple[float | None, dict]:
    """Only a fully covered token receipt has a numeric estimated cost.

    An all-zero normalized receipt with nothing from the provider behind it is an
    unreported receipt (the adapter saw no usage), priced unknown; a provider that
    reports zeros reports a free response, and those zeros price as zero.
    """
    if "connected_seconds" in usage:
        seconds = usage["connected_seconds"]
        price = rate(quote, "input_cost_per_second")
        cost = seconds * price if type(seconds) in (int, float) and seconds >= 0 and price is not None else None
        known = cost is not None and math.isfinite(cost)
        return (cost if known else None), {
            "priced_usage": {"connected_seconds": {"seconds": seconds, "usd_per_second": price}},
            "unpriced": [] if known else ["connected_seconds_tariff_or_count_unknown"]}
    body = _raw_body(raw_usage)
    missing = _unpriced_raw(body)
    missing.extend("unknown_" + key for key in USAGE_KEYS if key not in usage)
    if "thinking_tokens" not in usage and not any(key in body for key in ("thoughtsTokenCount", "thoughts_token_count")):
        missing.append("unknown_thinking_tokens")
    priced = {}
    try:
        counts = {key: _count(usage.get(key, 0)) for key in USAGE_KEYS}
        missing.extend(_raw_count_gaps(counts, body))
        missing.extend(str(key) for key, value in usage.items() if key not in USAGE_AXES and value != 0)
        split = _cached_split(counts, body)
        total_input = sum(counts[f"{m}_input_tokens"] for m in ("text", "audio", "image", "video"))
        for key in RATE_FIELDS:
            if key.startswith("cached_"):
                number = split[key.split("_")[1]]
            else:
                number = counts[key]
                if key.endswith("_input_tokens"):
                    number -= split[key.split("_")[0]]
            if number:
                field = RATE_FIELDS[key]
                if key in ("video_input_tokens", "cached_video_input_tokens"):
                    dedicated = ("input_cost_per_video_token" if key == "video_input_tokens"
                                 else "cache_read_input_video_token_cost")
                    if rate(quote, dedicated, input_tokens=total_input) is not None:
                        field = dedicated
                price = rate(quote, field, input_tokens=total_input)
                priced[key] = {"tokens": number, "usd_per_token": price}
                if price is None:
                    missing.append(field)
        thoughts = _count(usage["thinking_tokens"]) if "thinking_tokens" in usage else _thoughts_tokens(body)
        if thoughts:
            price = rate(quote, RATE_FIELDS["text_output_tokens"], input_tokens=total_input)
            priced["thoughts_output_tokens"] = {"tokens": thoughts, "usd_per_token": price}
            if price is None:
                missing.append(RATE_FIELDS["text_output_tokens"])
        if not usage or (not any(counts.values()) and not body):
            missing.append("usage_unreported")
    except (ValueError, TypeError, KeyError) as exc:
        missing.append(str(exc))
    if quote is None:
        missing.append("catalog_entry_unavailable")
    try:
        cost = None if missing else sum(item["tokens"] * item["usd_per_token"] for item in priced.values())
    except OverflowError:
        cost = math.inf
    if cost is not None and not math.isfinite(cost):
        missing.append("nonfinite_cost")
        cost = None
    return cost, {"priced_usage": priced, "unpriced": sorted(set(missing))}


def warm_quote(*, engine: str, model: str, drive_root) -> dict | None:
    """Fetch the catalog once, off the call's event loop; later lookups read this snapshot."""
    return get_catalog_quote(model, provider=engine, drive_root=drive_root, allow_live_fetch=True)


def check_live_budget(*, engine: str, model: str, task_id: str, drive_root) -> None:
    """Read the ordinary admission axes without reserving or admitting anything.

    Connected time is already spent when reported. Its settlement, including the
    closing tail, never calls this guard; the running host's control loop does.
    """
    from ouroboros import usage_store
    from ouroboros.usage_admission import raise_group_refusal

    request, scope = accounting._merge_scope(accounting.AttemptRequest(
        model=model, provider=engine, task_id=task_id, drive_root=drive_root))
    with usage_store.read(drive_root) as view:
        if view.exceeds_limit(accounting._global_limit(request)):
            raise accounting.BudgetExceeded("global model budget exhausted")
        if (scope.root_task_id and scope.root_limit_usd is not None
                and view.exceeds_limit(scope.root_limit_usd, root_task_id=scope.root_task_id)):
            raise accounting.BudgetExceeded("root model budget exhausted", limit_scope="root",
                                            root_task_id=scope.root_task_id)
        raise_group_refusal(view, scope)


def reserve_live_response(*, engine: str, model: str, call_id: str, response_key: str,
                          task_id: str, rendered_mode: str, drive_root, quote=None, physical_context=None) -> LiveAttempt:
    quote = copy.deepcopy(quote)
    provisional = {"connected_seconds": 1} if rate(quote, "input_cost_per_second") is not None else PROVISIONAL_USAGE
    cost, pricing = _price(quote, provisional, {})
    context = physical_context or accounting.current_physical_attempt_context()
    if context is not None:
        context = replace(context, rendered_mode=rendered_mode)
    evidence = {"call_id": call_id, "response_key": response_key, "rendered_mode": rendered_mode,
                "tariff_quote": quote, "reservation_usage": dict(provisional), "reservation_pricing": pricing}
    reservation = accounting.reserve_attempt(accounting.AttemptRequest(
        model=model, provider=engine, task_id=task_id, drive_root=drive_root, source="live_response",
        reservation_usd=cost, force_unknown_reservation=cost is None, physical_context=context,
        cost_evidence=evidence))
    # The driver calls at ResponseStarted: a later disconnect is not proof of
    # non-dispatch and cannot release this response as a free attempt.
    try:
        accounting.mark_dispatched(reservation)
    except accounting.BudgetExceeded:
        accounting.release_attempt(reservation, "response_admission_refused")
        raise
    return LiveAttempt(reservation, quote, evidence)


def _settlement(quote, evidence, usage, raw_usage, status):
    cost, pricing = _price(quote, usage, raw_usage)
    evidence = {**evidence, "usage": copy.deepcopy(dict(usage)),
                "raw_usage": copy.deepcopy(dict(raw_usage)), "status": status, **pricing}
    normalized = {**dict(usage), "cost_evidence": evidence, "cost_basis": "litellm_modalities",
                  "cost_coverage": "complete" if cost is not None else "partial",
                  "price_source": f"litellm@{quote['fetched_at']}" if quote else "unknown"}
    # Compatibility counters contain text only; all modalities and their cache
    # split remain in evidence, so old readers never see audio as prompt text.
    for source, target in (("text_input_tokens", "prompt_tokens"), ("text_output_tokens", "completion_tokens")):
        value = usage.get(source)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            normalized[target] = value
    return normalized, cost


def settle_live_response(attempt: LiveAttempt, *, usage: Mapping, raw_usage: Mapping, status: str) -> None:
    normalized, cost = _settlement(attempt._quote, attempt._evidence, usage, raw_usage, status)
    accounting.settle_attempt(attempt._reservation, normalized, cost_usd=cost, cost_final=False)


def record_refused_live_response(*, engine: str, model: str, call_id: str, response_key: str, task_id: str,
                                 rendered_mode: str, drive_root, reason: str, usage: Mapping, raw_usage: Mapping,
                                 status: str, quote=None, physical_context=None, pending=False) -> LiveAttempt:
    """A response the budget refused after the provider started it: the charge stays on the ledger.

    Admission cannot veto past provider spend (``record_received_attempt``). With a
    reported receipt the row is priced like any settlement; without one (refused at
    the start event) it carries no counts, unknown cost, and names the refusal.
    """
    quote = copy.deepcopy(quote)
    context = physical_context or accounting.current_physical_attempt_context()
    if context is not None:
        context = replace(context, rendered_mode=rendered_mode)
    evidence = {"call_id": call_id, "response_key": response_key, "rendered_mode": rendered_mode,
                "tariff_quote": quote, "budget_refusal": reason}
    normalized, cost = _settlement(quote, evidence, usage, raw_usage, status)
    reservation = accounting.record_received_attempt(accounting.AttemptRequest(
        model=model, provider=engine, task_id=task_id, drive_root=drive_root, source="live_response_refused",
        physical_context=context, cost_evidence=evidence), normalized, cost_usd=cost, pending=pending)
    return LiveAttempt(reservation, quote, evidence)


def settle_connection_usage(*, engine: str, model: str, call_id: str, connection_key: str, task_id: str,
                            rendered_mode: str, drive_root, usage: Mapping, raw_usage: Mapping,
                            watermark: float, quote=None, physical_context=None) -> None:
    """Record spent connection time through the response settlement owner, without admission.

    A connection sample is not a response: it has its own source and connection
    identity, no response key, reservation or invented provider response event.
    """
    quote = copy.deepcopy(quote)
    evidence = {"call_id": call_id, "connection_key": connection_key, "rendered_mode": rendered_mode,
                "tariff_quote": quote, "connected_seconds_watermark": watermark}
    normalized, cost = _settlement(quote, evidence, usage, raw_usage, "observed")
    accounting.record_received_attempt(accounting.AttemptRequest(
        model=model, provider=engine, task_id=task_id, drive_root=drive_root, source="live_connection_usage",
        physical_context=physical_context, cost_evidence=evidence), normalized, cost_usd=cost)


def abandon_live_response(attempt: LiveAttempt, *, reason: str) -> None:
    accounting.mark_unresolved(attempt._reservation, reason)


def observe_live_usage(host, sample: Mapping) -> None:
    """Settle non-response receipts on the call's existing physical-attempt ledger.

    Input transcription is separately billed under its configured ASR model, never the voice
    model's tariff. The call's warm_quote already fetched the catalog off-loop; an absent/stale
    ASR quote or incomplete receipt stays unknown. These are incurred charges, so admission
    cannot discard them even when a prior receipt exhausted the shared limit.
    """
    usage, raw = sample.get("usage", {}), sample.get("raw_usage", {})
    if "connected_seconds" in usage:
        host._settle_connection_sample(host._connection_growth(usage), raw)
    elif sample.get("component") == "input_transcription":
        # The contract requires the component's model; without one the receipt still lands, priced unknown.
        engine, model = host.turn.route.accounting_provider, str(sample.get("model") or "")
        quote = model and get_catalog_quote(model, provider=engine, drive_root=host.ctx.drive_root,
                                            allow_live_fetch=False) or None
        evidence = {"call_id": host.turn.spec.call_id, "connection_key": str(host.connection_number),
                    "rendered_mode": host.plan.initial_mode, "tariff_quote": quote,
                    **{key: sample.get(key) for key in ("item_id", "content_index", "event_id")}}
        normalized, cost = _settlement(quote, evidence, usage, raw, "completed")
        accounting.record_received_attempt(accounting.AttemptRequest(
            model=model, provider=engine, task_id=host.ctx.task_id, drive_root=host.ctx.drive_root,
            source="live_transcription_usage", physical_context=host._physical_context,
            cost_evidence=evidence), normalized, cost_usd=cost)
