"""Main's route-bound window facts, independent of review capability authority.

Acceptances are lower bounds, never evidence of room for a reply. A numbered
refusal or a measured refusal outside the text Chat start-hint scope may cap its
kind of limit (``record_refusal``). Comparable numberless Chat refusals instead
inform first-send composition, independently of window authority, so other foci
of that route take no cap from them and meet their own refusal (a residual whose
cost is not measured). Both share the capability JSON store; old stores need no
migration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ouroboros.server_window import PHYSICAL_SOURCES as _PHYSICAL_SOURCES, physical_statement


_REPLY_INSIDE = frozenset({"refusal", "local_health", "server_metadata"})  # what proves a total holds the reply: ``WindowRecord.admitted``
UNTAUGHT_BASES = frozenset({"status_413", "number_equals_declared"})


@dataclass(frozen=True)
class WindowLimit:
    tokens: int
    kind: str  # input | total | output
    source: str
    observed_at: str = ""
    acceptance_can_raise: bool = True
    # A refusal's number: ``provider_stated``, ``exact_local_count`` (a local instance counted the request the
    # host refused before sending), ``anchored_estimate`` from a numberless refusal (a measured input or
    # confirmation of an existing total), or ``unrecorded`` (stored before bases were kept); "" for every other source.
    basis: str = ""
    sequence: int = 0  # publication order in this route, allocated under the store lock
    sequence_route: str = ""  # also binds an attempted, unpublished receipt (sequence zero)
    read_id: str = ""  # receipt identity, independent of serving-instance identity
    instance_id: str = ""  # empty when the server supplies no instance identifier


@dataclass(frozen=True)
class WindowRecord:
    route_fp: str
    input: Optional[WindowLimit] = None
    total: Optional[WindowLimit] = None
    output: Optional[WindowLimit] = None
    reserve: int = 0
    transmitted: Optional[int] = None
    accepted: int = 0
    declared: tuple[WindowLimit, ...] = ()
    accepted_at: str = ""
    contradicted_by: Optional[WindowLimit] = None
    retired_ack: dict = field(default_factory=dict)
    floor: Optional[int] = None  # the reply floor ``admitted`` charges an unproven total; None: the whole reserve

    @property
    def usable(self) -> Optional[int]:
        """The input a start is sized to: a total always leaves the whole reply reserve."""
        limits = ([self.input.tokens] if self.input else []) + (
            [max(0, self.total.tokens - self.reserve)] if self.total else [])
        return min(limits) if limits else None

    @property
    def admitted(self) -> Optional[int]:
        """The largest input a request that exists goes out at, each stated number read most favourably: an input
        limit whole; a total less the whole reserve where it is known to hold the reply too (a size refusal said so,
        or a local server stated its one buffer), else less the reply floor. Every question a round asks about its
        own request reads this one line (the pre-send check, the room a tool result is shown in, whether a view is
        kept, what a pass leaves), so nothing is cut on the guess that the reply is inside. A provider that counts
        the reply refuses such a request once, and that refusal proves the total."""
        proven = self.floor is None or (self.total is not None and self.total.source in _REPLY_INSIDE)
        limits = ([self.input.tokens] if self.input else []) + ([max(0, self.total.tokens - (
            self.reserve if proven else min(self.reserve, self.floor)))] if self.total else [])
        return min(limits) if limits else None

    @property
    def boundary(self) -> Optional[int]:
        limits = [limit.tokens for limit in (self.input, self.total) if limit]
        return min(limits) if limits else None

    @property
    def limiting(self) -> Optional[WindowLimit]:
        return min((limit for limit in (self.input, self.total) if limit),
                   key=lambda item: item.tokens - (self.reserve if item.kind == "total" else 0), default=None)

    @property
    def source(self) -> str:
        limit = self.limiting
        if limit is None:
            return "unknown"
        stamp = limit.observed_at.replace("T", " ")[:16]
        return f"{limit.source} {stamp}".strip()

    @property
    def total_equivalent(self) -> int:
        """Compatibility frame for consumers which subtract R; never provider evidence."""
        return self.usable + self.reserve if self.usable is not None else 0

    @property
    def admitted_equivalent(self) -> int:
        """``admitted`` in that same frame: what a round's measurement reports as its capacity."""
        return self.admitted + self.reserve if self.admitted is not None else 0


def record_observation(root, route_fp: str, *, accepted: int = 0, refused: int = 0,
                       kind: str = "total", reserve: int = 0, transmitted: int = 0, basis: str = "") -> None:
    """``basis``: how a refused size was known (``WindowLimit.basis``), kept on its row; rows without one stay as written."""
    from ouroboros import capability_evidence as ce

    if not route_fp or not (accepted > 0 or refused > 0):
        return
    try:
        with ce._store_transaction(root):
            data = ce._load(root)
            record = data.setdefault("main_windows", {}).setdefault(route_fp, {})
            rows = record.setdefault("observations", [])
            prior = list(rows)
            if refused and rows and all(rows[-1].get(key, default) == value for key, value, default in (
                    ("refused", refused, 0), ("kind", kind, "total"), ("basis", basis, ""),
                    ("reserve", reserve, 0), ("transmitted", transmitted, 0))) and (
                    rows[-1].get("sequence", 0) == record.get("sequence", 0)
                    and _window_age(rows[-1]["ts"]) <= ce._CONFIRMED_TTL_SEC):
                return
            # Claim identity/order outlives its authority TTL. Equal cache restores
            # must not republish it after a refusal written during expiry.
            reads = record.get("reads", [])

            def answer(index):
                return next((later for later in range(index + 1, len(rows))
                             if rows[later].get("accepted", 0) > rows[index]["refused"]), None)

            def refusals():
                newest = {}
                for index in reversed(range(len(rows))):
                    if rows[index].get("refused"):
                        newest.setdefault(rows[index].get("kind"), index)
                return list(newest.values())

            def after(read):
                return [i for i, row in enumerate(rows) if row.get("sequence", 0) > read["sequence"]
                        and row.get("accepted", 0) > read["tokens"]
                        and _window_age(row["ts"]) <= ce._CONFIRMED_TTL_SEC]

            # A repeated success adds nothing unless it improves the answer to a
            # serving read, answers a refusal recorded since the previous success, or is
            # owed to a plan captured at an earlier read: no fresh success at least as
            # large follows the newest read (the preservation rule below).
            high = max((row.get("accepted", 0) for row in rows), default=0)
            newest_read = max((read.get("sequence", 0) for read in reads), default=0)
            captured = newest_read and not any(
                row.get("accepted", 0) >= accepted and row.get("sequence", 0) > newest_read
                and _window_age(row["ts"]) <= ce._CONFIRMED_TTL_SEC for row in rows)
            changed = refused or accepted > high or captured or any(
                accepted > rows[i]["refused"] and answer(i) is None for i in refusals()) or any(
                accepted > max([read["tokens"], *[rows[i]["accepted"] for i in after(read)]]) for read in reads)
            if not changed:
                return
            record["sequence"] = int(record.get("sequence", 0)) + 1
            rows.append({"accepted": accepted, "refused": refused, "kind": "input" if accepted else kind,
                         "reserve": reserve, "transmitted": transmitted, "ts": ce.utc_now_iso(),
                         "sequence": record["sequence"], **({"basis": basis} if refused and basis else {})})
            keep = set()
            largest = max(range(len(rows)), key=lambda i: rows[i].get("accepted", 0))
            if rows[largest].get("accepted"):
                keep.add(largest)
            # Older captured plans can still resolve after the current read has
            # changed. A fresh acceptance is redundant for every such capture
            # only if a later acceptance is at least as large.
            later_high = 0
            for index in reversed(range(len(rows))):
                value = rows[index].get("accepted", 0)
                if value > later_high and _window_age(rows[index]["ts"]) <= ce._CONFIRMED_TTL_SEC:
                    keep.add(index)
                    later_high = value
            for read in reads:
                witness = max(after(read), key=lambda i: rows[i]["accepted"], default=None)
                if witness is not None:
                    keep.add(witness)
            for index in refusals():
                keep.update(i for i in (index, answer(index)) if i is not None)
            record["observations"] = [rows[i] for i in sorted(keep)]
            record["reads"] = reads
            if record["observations"] != prior:
                ce._save(root, data)
    except OSError:  # a busy or unwritable store skips this observation; the send it saw stands
        return


def observation_scope(plan) -> str:
    record = getattr(plan, "window_record", None)
    return record.route_fp if record is not None else str(getattr(plan, "route_fp", "") or "")


def route_observation_scope(route, settings, *, target=None) -> str:
    """Main's account scope; capability acknowledgements and density keep their own keys."""
    import hashlib
    import json
    from ouroboros.capability_evidence import route_fingerprint
    from ouroboros.llm_attempt import ConnectionUnavailable
    from ouroboros.response_limits import dispatch_target

    if route["provider"] == "claudexor":
        return route_fingerprint(provider=route["provider"], model=route["model"],
            base_url=route["base_url"], options=route["options"]) if route["options"].get("account_fingerprint") else ""
    try:
        if target is None:
            target = dict(route, keyless=True) if route["use_local"] else dispatch_target(route["model"], settings)
        elif route["use_local"]:
            target = {**target, "keyless": True}
    except ConnectionUnavailable:
        return ""  # a refused connection sends nothing, so it has no observations to scope
    credentials = {key: target.get(key, "") for key in ("api_key", "user", "password", "scope")}
    if not any(credentials.values()) and not target.get("keyless"):
        return ""
    identity = target.get("capability_route") or {}
    payload = {"provider": identity.get("provider", target["provider"]), "model": identity.get("model", route["model"]),
               "base_url": identity.get("dispatch_base_url", target.get("base_url", "")), "credentials": credentials}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


_START_HINTS = "main_first_contact"
_START_ROWS = 16  # per outcome kind and exact scope; _trim_start_rows adds back what the reader derives


def _http_chat_target(target) -> bool:
    from ouroboros.llm_fallback import _wire_surface
    from ouroboros.provider_models import OPENAI_API_SURFACES

    return bool(target and target.get("provider") not in {None, "", "local", "claudexor", "gigachat"}
                and _wire_surface(target) == OPENAI_API_SURFACES["chat_completions"])


def first_contact_scope(target, reply_cap: int, *, payload=None) -> str:
    """Comparable HTTP Chat sends share the exact credentialed route, raw measure and reply allowance.

    A local server, a subscription harness and GigaChat may carry a Chat-shaped body, but their sizes are
    not comparable with an HTTP credential's, so this hint scopes none of them. ``_wire_surface`` itself
    keeps naming the wire for every transport: its other readers price and finalize a send by it.
    """
    from urllib.parse import urlsplit
    if (not _http_chat_target(target) or not target.get("capability_route") or not target.get("api_key")
            or target.get("keyless") or not target.get("resolved_model") or reply_cap <= 0):
        return ""
    try:
        endpoint = urlsplit(target.get("base_url", ""))
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
            return ""
    except ValueError:
        return ""
    route = dict(provider=target["provider"], model=target["resolved_model"], use_local=False)
    scope = route_observation_scope(route, None, target={**target, "capability_route": {
        **target["capability_route"], "model": target["resolved_model"]}})
    if scope and target["provider"] == "openrouter":
        import hashlib
        import json
        from ouroboros.llm_routing import openrouter_chat_routing, sealed_provider_routing

        # A fresh chat has no assistant replay. At send, take the finalized SDK carrier, not current settings.
        routing = openrouter_chat_routing(target, []) if payload is None else sealed_provider_routing(payload)
        if routing is None and payload is not None:
            extra = payload.get("extra_body")
            if "provider" in payload or (isinstance(extra, dict) and "provider" in extra):
                return ""  # null is a submitted value, not omission
        if routing is not None and not isinstance(routing, dict):
            return ""
        try:
            digest = hashlib.sha256(json.dumps(routing or {}, sort_keys=True, allow_nan=False).encode()).hexdigest()
        except (TypeError, ValueError):
            return ""
        scope = f"{scope}:{digest}"  # never matches old observations which omitted the routing carrier
    return f"chat_function_raw_v2:{scope}:{reply_cap}" if scope else ""  # v1 also accepted tool-less replies


def text_chat_rows(rows) -> bool:
    """Only text content has the estimate's comparability; image pricing is a different measure."""
    return bool(rows) and all(isinstance(row, dict) and not row.get("audio") and (
        row.get("content") is None or isinstance(row.get("content"), str)
        or isinstance(row["content"], list) and all(isinstance(part, dict) and part.get("type") == "text"
                                                   for part in row["content"])) for row in rows)


def signed_first_contact(physical, target, payload, reply_cap):
    """Stamp only the final Chat candidate of a fresh direct chat's plan (``ContextFitPlan.first_contact``), with the
    scope its resolved target and reply allowance actually give it; other foci keep the window learning they had.
    Advice compares function-schema Chat bodies only, not tool-less, server-tool or custom bodies."""
    from ouroboros.request_wire_contract import infer_tool_dialect

    if physical is None:
        return None
    planned = (physical.first_contact_plan or ("",))[0]
    compare = (planned and physical.view_kind == "owner_turn" and physical.rendered_mode in {"max", "low"}
               and physical.preferred_mode in {"max", "low"} and physical.round_id)
    actual = first_contact_scope(target, reply_cap, payload=payload) if planned else ""
    scope = actual if (compare and actual and payload.get("model") == target.get("resolved_model")
            and infer_tool_dialect(payload) == "function"
            and isinstance(payload.get("messages"), list) and text_chat_rows(payload["messages"])) else ""
    bound = physical.window_witness_bound
    if planned and bound:  # even when the physical target no longer speaks Chat
        route = dict(provider=target["provider"], model=target.get("resolved_model"), use_local=False)
        bound = bool(actual and payload.get("model") == target["resolved_model"]
                     and actual.rpartition(":")[0] == planned.rpartition(":")[0]
                     and route_observation_scope(route, None, target=target) == physical.window_observation_fp)
    if scope or physical.first_contact_scope or bound != physical.window_witness_bound:
        from dataclasses import replace

        return replace(physical, first_contact_scope=scope, first_contact_dispatch=None, window_witness_bound=bound)
    return physical


def first_contact_rebound(physical) -> bool:
    """A start the hint shaped for one scope is about to go out under another (a key, account, cap or tool dialect
    changed between fit and dispatch): it is sent in its ordinary form instead (``loop_model_call``)."""
    planned, hinted = getattr(physical, "first_contact_plan", None) or ("", False)
    return bool(hinted and planned != physical.first_contact_scope)


def first_contact_route_scope(route) -> tuple[str, int]:
    """``(scope, reply cap)`` of a fit route's resolved target (``first_contact_scope``), ``("", 0)`` for a local or
    unresolved one: what a fresh chat's start fit reads, and what a route rebind must still resolve to for an unsent
    hint to stay armed (``loop_routes.first_contact_route``)."""
    from ouroboros.loop_llm_call import MAIN_LOOP_MAX_TOKENS
    from ouroboros.response_limits import response_limit_for_target

    target = None if route.get("use_local") else route.get("_resolved_target")
    if not target:
        return "", 0
    cap = response_limit_for_target(target).ceiling(MAIN_LOOP_MAX_TOKENS)
    return first_contact_scope(target, cap), cap


def _live_start_rows(rows):
    from ouroboros.capability_evidence import _CONFIRMED_TTL_SEC

    return [row for row in rows if _window_age(row["at"]) <= _CONFIRMED_TTL_SEC]


#: The Max wire: a whole-book send, whole or with memory folded. ``low`` is any other chapter-map send (a known window's
#: Low, the owner's Low, a start the hint already took to maps); ``maps`` only a same-round retry after a Max refusal.
_MAX_FORMS = frozenset({"ordinary", "folded"})
_COVER = "cover"  # the one accepted row standing for the replies the bound folded (``_trim_start_rows``)


def _start_form(physical) -> str:
    origins = (physical.prefix or {}).get("origins", ())
    if physical.rendered_mode == "low":
        return ("maps" if physical.preferred_mode == "max" and physical.automatic_pass_used and "low_retry" in origins
                else "low")
    return "folded" if "memory_coarsened" in origins else "ordinary"


def _pending_starts(rows):
    """`(refused, followed by a maps rung)` for each refusal no later, no smaller acceptance answers, in exact dispatch
    order, not publication or wall time. A refused same-round maps send is one too: which start it may bound is
    ``first_contact_limits``' to say."""
    answered, maps_rounds = 0, set()
    for row in reversed(rows):
        if row["form"] == "maps":
            maps_rounds.add(row["round"])
        if row["accepted"]:
            answered = max(answered, row["raw"])
        elif row["raw"] > answered:
            yield row, row["round"] in maps_rounds


def _maps_proof(rows, refusal):
    """The later same-round row that made this refusal's round a proven maps rung."""
    return next((row for row in rows if row["order"] > refusal["order"] and row["round"] == refusal["round"]
                 and row["form"] == "maps"), None)


def _trim_start_rows(ordered):
    """Bound the dispatch-ordered set without erasing what the reader derives from it.

    Recency alone is not a bound on meaning: ``first_contact_limits`` reads the smallest and the largest
    still-unanswered refusal of each class, so a live 548k witness must not leave the set merely because
    sixteen smaller refusals arrived, and a preserved maps rung keeps the same-round row that proved it.
    Replies past the newest ``_START_ROWS`` fold into one conservative watermark, the ``cover`` row: accepted at
    the largest dispatch order and the largest size any of them had, so it answers every refusal they could, one
    still settling included, and may answer more (a hint lost, never a false one), while a refusal larger than all
    of them keeps its own witness. A refusal neither live nor a maps proof says nothing and leaves, so an answered
    refusal never revives when its answer is folded. Bounded by the newest ``_START_ROWS`` of each outcome, each
    class's two extremes with their proofs and one cover; every other kept row is the very same object, so each
    original ``at``, and with it its own expiry, stays untouched (the cover keeps its newest reply's).
    """
    pending = {row["order"]: maps for row, maps in _pending_starts(ordered)}
    cells, proofs = {}, set()
    for row in ordered:
        if row["order"] in pending:
            cells.setdefault((pending[row["order"]], row["form"]), []).append(row)
    replies = [row for row in ordered if row["accepted"] and row["form"] != _COVER]
    keep = {row["order"] for row in replies[-_START_ROWS:] + [r for r in ordered if not r["accepted"]][-_START_ROWS:]}
    for (maps, _form), cell in cells.items():
        for edge in (min(cell, key=lambda row: row["raw"]), max(cell, key=lambda row: row["raw"])):
            keep.add(edge["order"])
            if maps and (proof := _maps_proof(ordered, edge)):
                keep.add(proof["order"])
                proofs.add(proof["order"])
    folded, said = [row for row in ordered if row["accepted"] and row["order"] not in keep], pending.keys() | proofs
    kept = [row for row in ordered if row["order"] in keep and (row["accepted"] or row["order"] in said)]
    if folded:  # an earlier cover is folded again with them
        newest = max(folded, key=lambda row: row["order"])
        kept.append({"attempt": "", "order": newest["order"], "at": newest["at"], "round": "",
                     "raw": max(row["raw"] for row in folded), "form": _COVER, "accepted": True})
    return sorted(kept, key=lambda row: row["order"])


def first_contact_limits(target, reply_cap: int, *, scope=None) -> tuple[int, int, int]:
    """``(Max wire, before maps, any form)``: the raw refused Max-wire size for ordinary folding (a whole send's,
    else a folded one's), the refused Max size a real same-round maps send followed, and the smallest refused size
    of any form, a refused maps rung included (what a chapter-map start compares with). A chapter-map refusal never
    bounds a Max start."""
    from ouroboros import capability_evidence as ce

    if scope is None:
        scope = first_contact_scope(target, reply_cap)
    if not scope:
        return 0, 0, 0
    rows = (ce._read(ce.canonical_evidence_root()).get(_START_HINTS) or {}).get(scope, [])
    refused = list(_pending_starts(_live_start_rows(rows)))
    wire = [(row, maps) for row, maps in refused if row["form"] in _MAX_FORMS]
    return (min((row["raw"] for row, _ in wire if row["form"] == "ordinary"),
                default=min((row["raw"] for row, _ in wire), default=0)),
            min((row["raw"] for row, maps in wire if maps), default=0),
            min((row["raw"] for row, _ in refused), default=0))


def _answers_for(row, other) -> bool:
    """``row`` answers every refusal ``other`` could, published or still settling: usable, dispatched later, no smaller."""
    return row["accepted"] and row["order"] > other["order"] and row["raw"] >= other["raw"]


def _routine(rows, event) -> bool:
    """An acceptance that cannot change dispatch-ordered cover: a stored reply already answers for it (a maps proof
    always counts). Whether a refusal stands yet is irrelevant: an older one may still be settling."""
    return event["form"] != "maps" and any(_answers_for(row, event) for row in rows)


def _start_event(capture, *, accepted: bool):
    """This send's outcome as plain data, ``[scope, event]`` bound to its dispatch order; None for a replay of an
    expired address, which is no new observation."""
    physical = capture.physical_context
    order, at = physical.first_contact_dispatch
    event = {"attempt": capture.attempt_id, "order": order, "at": at, "round": physical.round_id,
             "raw": capture.prompt_tokens_bounded_estimate, "form": _start_form(physical), "accepted": accepted}
    return [physical.first_contact_scope, event] if _live_start_rows([event]) else None


def publish_start_outcome(outcome) -> None:
    """One receipt (``_start_event``), one locked merge; a routine success (``_routine``) is read, never written."""
    from ouroboros import capability_evidence as ce

    if not outcome:
        return
    scope, event = outcome
    accepted, root = event["accepted"], ce.canonical_evidence_root()
    if accepted and _routine(_live_start_rows((ce._read(root).get(_START_HINTS) or {}).get(scope, [])), event):
        return
    try:
        with ce._store_transaction(root):
            data = ce._load(root)
            routes = data.setdefault(_START_HINTS, {})
            rows = _live_start_rows(routes.get(scope, []))
            if any(row["attempt"] == event["attempt"] or row["order"] == event["order"] for row in rows):
                return
            if accepted and _routine(rows, event):
                return
            # The replies it answers for go (a maps proof stays): what is left is a dispatch-ordered skyline.
            kept = [row for row in rows if not (accepted and row["accepted"] and row["form"] != "maps"
                                                and _answers_for(event, row))]
            # An acceptance can arrive before an older refusal is published: trim by meaning, not by count.
            updated = _trim_start_rows(sorted([*kept, event], key=lambda row: row["order"]))
            if updated != rows:
                data[_START_HINTS] = {key: fresh for key, prior in routes.items() if (fresh := _live_start_rows(prior))}
                data[_START_HINTS][scope] = updated
                ce._save(root, data)
    except OSError:  # optional evidence; the provider's actual outcome still stands
        return


def _current_request_receipt(capture, call, round_id) -> bool:
    physical = getattr(capture, "physical_context", None)
    return bool(physical and round_id and physical.round_id == round_id
                and capture.attempt_id and call.get("physical_attempt_id") == capture.attempt_id
                and capture.state in {"dispatched", "settled", "unresolved"}
                and capture.candidate_measurement_kind == "canonical_json_v1" and capture.candidate_manifest_ref
                and capture.prompt_tokens_bounded_estimate > 0)


def _advice_dispatched(capture) -> bool:
    physical = getattr(capture, "physical_context", None)
    return bool(getattr(physical, "first_contact_scope", "") and physical.first_contact_dispatch)


def start_answer(usage, capture, round_idx: int):
    """This round's usable reply as its own send's receipt (``_start_event``, plain data), else None: the capture is
    the reply's own attempt in this round, with no provider error, body error or substituted model."""
    call = usage.get("_last_llm_call_meta") or {}
    round_id = f"{usage['execution_id']}:round:{round_idx}" if usage.get("execution_id") else ""
    if (_advice_dispatched(capture) and _current_request_receipt(capture, call, round_id) and call.get("round_id") == round_id
            and not capture.provider_error_type and capture.provider_status_code is None
            and not call.get("provider_body_error") and not usage.get("_model_substitutions")):
        return _start_event(capture, accepted=True)
    return None


def record_start_answer(usage, capture, round_idx: int) -> None:
    """Called only after the Main loop accepts a usable result, including replies without usage. A forced wrap-up
    takes its receipt at its call (``start_answer``) and publishes it only for a reply its rail keeps."""
    publish_start_outcome(start_answer(usage, capture, round_idx))


def record_start_refusal(usage, capture, round_id: str) -> bool:
    """The advisory hint takes only a qualified refusal that stated no numbers."""
    fact = _current_chat_refusal(usage, capture, round_id) if _advice_dispatched(capture) else None
    if fact is None or fact.has_numbers:
        return False
    publish_start_outcome(_start_event(capture, accepted=False))
    return True


def _current_chat_refusal(usage, capture, round_id: str):
    """This send's own whole-request refusal, before applying the distinct numeric and advisory policies."""
    from ouroboros.context_budget import CONTEXT_OVERFLOW_CODES, INPUT_STRING_LENGTH_CODES, SIZE_REFUSAL_STATUSES
    from ouroboros.size_refusal import SizeRefusal, stamped

    call, answer = (usage.get(key) or {} for key in ("_last_llm_failure", "_last_llm_provider_answer"))
    if not _current_request_receipt(capture, call, round_id) or call.get("failure_code") != "context_overflow":
        return None
    finish = str(call.get("size_finish") or "").lower()
    codes = {str(call.get("provider_code") or "").lower(), finish,
             *(str(code).lower() for _who, code in answer.get("codes") or ())}
    status, body = capture.provider_status_code, bool(call.get("provider_body_error"))
    if answer.get("basis") in UNTAUGHT_BASES or codes & INPUT_STRING_LENGTH_CODES:
        return None
    if status in SIZE_REFUSAL_STATUSES and not body:
        fact = stamped(usage)
        if status == 413 and not (fact and fact.token_window and answer.get("basis") == "typed"
                                  and codes & CONTEXT_OVERFLOW_CODES):
            return None  # a bare, prose-only or body-size 413 never teaches from an estimate or a cited number
    elif status is None and body and isinstance(answer.get("size"), dict):
        fact = SizeRefusal(**answer["size"])
    elif status is None and not body and not (answer.get("size") or answer.get("sentence")) and finish in CONTEXT_OVERFLOW_CODES:
        fact = SizeRefusal(source="finish", own=True)
    else:
        return None
    return fact if fact is not None and fact.own and fact.source not in {"", "excluded", "status_413"} else None


def _event_time(value: str):
    from datetime import datetime, timezone
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp
    except (ValueError, TypeError, AttributeError):
        return datetime.min.replace(tzinfo=timezone.utc)


def _window_age(stamp: str) -> float:
    """Wall time is TTL only. A future date is fresh, never an ordering hint."""
    from ouroboros import capability_evidence as ce
    return max(0.0, (ce.utc_now() - _event_time(stamp)).total_seconds())


# Live receipt attempts survive a probe-cache reconstruction without another store write.
# Only fresh receipts matter; expired entries are discarded on the next publication.
_READ_ATTEMPTS: Dict[tuple, WindowLimit] = {}


def _publish_read(root, fp: str, limit: WindowLimit, *, publish=True) -> WindowLimit:
    from ouroboros import capability_evidence as ce

    # The same RLock as the writer serializes attempts and their in-process memo.
    # A receipt ID comes from the transport, not from a cached projection.
    with ce._STORE_LOCK:
        for key, held in tuple(_READ_ATTEMPTS.items()):
            if _window_age(held.observed_at) > ce._CONFIRMED_TTL_SEC:
                del _READ_ATTEMPTS[key]
        key = (str(root), fp, limit.read_id)
        held = _READ_ATTEMPTS.get(key) if limit.read_id else None
        if held is not None:
            return held
        result = _publish_read_once(root, fp, limit, publish=publish)
        if publish and fp and limit.read_id:
            _READ_ATTEMPTS[key] = result
        return result


def _publish_read_once(root, fp: str, limit: WindowLimit, *, publish=True) -> WindowLimit:
    """Coalesce the same serving claim, preserving order across cache reconstruction.

    Publication is reserved for send/plan paths. Owner projections read the same
    frontier but only assign a hypothetical next number to an unpublished claim.
    """
    from dataclasses import asdict, replace
    from ouroboros import capability_evidence as ce

    if not fp or limit.sequence_route == fp:
        return limit

    def numbered(record):
        current = next(iter(record.get("reads", [])), None)
        if current and statement_key(WindowLimit(**current)) == statement_key(limit):
            return replace(limit, sequence=current["sequence"], sequence_route=fp), True
        return replace(limit, sequence=int(record.get("sequence", 0)) + 1, sequence_route=fp), False

    record = ce._read(root).get("main_windows", {}).get(fp, {})
    result, same = numbered(record)
    def expired(record):
        current = next(iter(record.get("reads", [])), None)
        return current and _window_age(current["observed_at"]) > ce._CONFIRMED_TTL_SEC >= _window_age(limit.observed_at)

    if not publish or (same and not expired(record)):
        return result
    try:
        with ce._store_transaction(root):
            data = ce._load(root)
            record = data.setdefault("main_windows", {}).setdefault(fp, {})
            result, same = numbered(record)
            if not same or expired(record):
                record["sequence"] = max(int(record.get("sequence", 0)), result.sequence)
                record["reads"] = [asdict(result)]
                if not ce._save(root, data):
                    return replace(limit, sequence=0, sequence_route=fp)
            return result
    except OSError:
        return replace(limit, sequence=0, sequence_route=fp)


def window_authority(events):
    """Pure precedence over route events in publication order (no clocks or IO).

    Owner declarations replace their kind. A changed serving statement replaces
    the physical claim and retires prior refusals/acceptance witnesses; an equal
    statement, including its instance, is a replay. Acceptance disproves every
    smaller raiseable claim preceding it, but never establishes a total window.
    The newest refusal of each kind remains until a changed read or a larger
    acceptance. Owner declarations still cap it; such a total counts the reply.
    """
    owners, refusals, physical, witness = {}, {}, None, None
    for event in events:
        if event.source == 'accepted':
            disproved = False
            for claims in (owners, refusals):
                for kind, claim in list(claims.items()):
                    if claim.acceptance_can_raise and event.tokens > claim.tokens:
                        del claims[kind]
                        disproved = disproved or claims is owners
            if physical and _later_acceptance(event, physical):
                disproved = True
            if disproved or witness:
                witness = max((x for x in (witness, event) if x), key=lambda x: x.tokens)
        elif event.source == 'refusal':
            refusals[event.kind] = event
        elif event.source in _PHYSICAL_SOURCES:
            if physical and statement_key(event) == statement_key(physical):
                continue
            physical, witness, refusals = event, None, {}
        else:
            owners[event.kind] = event
    limits = dict(owners)
    candidates = list(refusals.values())
    if physical and not (witness and _later_acceptance(witness, physical)):
        candidates.append(physical)
    if witness:
        candidates.append(witness)
    inside = False
    for claim in candidates:
        current = limits.get(claim.kind)
        if current is None or claim.tokens <= current.tokens:
            limits[claim.kind] = claim
        elif claim.source == 'refusal' and claim.kind == 'total':
            inside = True
    return limits, inside


def _later_acceptance(accepted, physical):
    """Only a proved publication order can disprove a serving ceiling.

    Sequence zero means publication failed, not that the read predates history.
    The same proof applies to a final witness raised by an owner declaration.
    """
    return 0 < physical.sequence < accepted.sequence and accepted.tokens > physical.tokens


def statement_key(limit):
    """Identity of a serving claim. A network-read UUID is not a server instance."""
    return limit.tokens, limit.kind, limit.source, limit.instance_id



def resolve_window(evidence, reserve: int, *, root=None, bound: bool = True,
                   observation_fp=None, accepted_input_tokens: int = 0, publish: bool = True) -> WindowRecord:
    from ouroboros import capability_evidence as ce

    root = root or ce.canonical_evidence_root()
    fp = observation_scope(evidence) if observation_fp is None else observation_fp
    rows = ((ce._read(root or ce.canonical_evidence_root()).get("main_windows", {}).get(fp) or {})
            .get("observations", [])) if bound and fp else []
    accepted = max((row.get("accepted", 0) for row in rows), default=0)
    accepted_at = next((row["ts"] for row in rows if accepted and row.get("accepted") == accepted), "")
    accepted = max(accepted, int(accepted_input_tokens or 0))
    captured = getattr(evidence, "window_record", None)
    limits = ({limit.kind: limit for limit in captured.declared}
              if captured is not None and ce.is_known(evidence, require_fresh=True) else {})
    if captured is None and ce.is_known(evidence, require_fresh=True):
        kind = str(getattr(evidence, "window_kind", "total") or "total")
        limits[kind] = WindowLimit(int(evidence.window_tokens), kind,
                                  str(getattr(evidence, "source", None) or getattr(evidence, "evidence_source", "metadata")),
                                  str(getattr(evidence, "ts", "")),
                                  (getattr(evidence, "route_facts", {}) or {}).get("acceptance_can_raise",
                                      not bool(getattr(evidence, "route_facts", {}))),
                                  read_id=str(getattr(evidence, "window_read_id", "")),
                                  sequence=int(getattr(evidence, "window_sequence", 0)),
                                  sequence_route=str(getattr(evidence, "window_sequence_route", "")),
                                  instance_id=str(getattr(evidence, "window_instance_id", "")))
    declared = tuple(limits.values())
    server = getattr(evidence, "server_window", {})
    contradiction = WindowLimit(**server) if server else captured.contradicted_by if captured else None
    # A serving buffer is authority only on the custom OpenAI-compatible/local route. A cloud
    # catalog (even one using the same field name) cannot overrule an owner's choice.
    physicals = [limit for limit in (*limits.values(), contradiction) if limit is not None and
        physical_statement({"source": limit.source, "provider": getattr(evidence, "provider", "")})
        and _window_age(limit.observed_at) <= ce._CONFIRMED_TTL_SEC]
    if bound and fp:
        physicals = [_publish_read(root, fp, limit, publish=publish) for limit in physicals]
    physical = max(reversed(physicals), key=lambda limit: limit.sequence, default=None)
    if publish and physical and isinstance(evidence, ce.CapabilityEvidence):
        # Copies of an already-published receipt keep its order after a newer
        # read replaces the store's frontier; never publish an old capture anew.
        from dataclasses import asdict
        evidence.window_sequence = physical.sequence
        evidence.window_sequence_route = physical.sequence_route
        if evidence.server_window:
            evidence.server_window = asdict(physical)
    if contradiction and (_window_age(contradiction.observed_at) > ce._CONFIRMED_TTL_SEC or not any(
            limit.source in {ce.SOURCE_OWNER_ACK, ce.SOURCE_USER_SETTING} and
            (limit.tokens, limit.kind) != (contradiction.tokens, contradiction.kind) for limit in declared)):
        contradiction = None
    if physical:
        limits = {kind: physical if (limit.tokens, limit.kind, limit.source, limit.observed_at, limit.read_id) ==
                  (physical.tokens, physical.kind, physical.source, physical.observed_at, physical.read_id) else limit
                  for kind, limit in limits.items()}
        declared = tuple(limits.values())
        if contradiction and (contradiction.tokens, contradiction.observed_at) == (physical.tokens, physical.observed_at):
            contradiction = physical
    # Everything below is a projection of ordered events. Freshness is filtered
    # here; it never changes publication order in the pure reducer.
    # A serving statement reaches the timeline only as the fresh selected read: an expired one is unknown.
    events = [limit for limit in declared if not physical_statement(
        {"source": limit.source, "provider": getattr(evidence, "provider", "")})]
    timeline = [physical] if physical else []
    for row in rows:
        if row.get("refused") and _window_age(row["ts"]) <= ce._CONFIRMED_TTL_SEC:
            timeline.append(WindowLimit(row["refused"], row["kind"], "refusal", row["ts"],
                basis=str(row.get("basis") or "unrecorded"), sequence=row.get("sequence", 0)))
        elif row.get("accepted") and (not physical or _window_age(row["ts"]) <= ce._CONFIRMED_TTL_SEC):
            timeline.append(WindowLimit(row["accepted"], "input", "accepted", row["ts"], sequence=row.get("sequence", 0)))
    if accepted_input_tokens and not physical:
        timeline.append(WindowLimit(accepted_input_tokens, "input", "accepted", sequence=0))
    limits, inside = window_authority([*events, *sorted(timeline, key=lambda e: e.sequence)])
    output = getattr(evidence, "response_limit", {}) or {}
    maximum = output.get("max_output_tokens", 0)
    from ouroboros.context_budget import NANO_MIN_HEADROOM_TOKENS
    # The least a request leaves for its reply: Nano's own reserve, or a route's larger declared working room.
    working = getattr(evidence, "working_reserve_tokens", None) or route_window_kwargs(evidence)["working_reserve_tokens"] or 0
    return WindowRecord(fp, limits.get("input"), limits.get("total"),
                        WindowLimit(maximum, "output", output.get("source", "metadata"),
                                    output.get("observed_at", "")) if maximum and not output.get("stale")
                        else captured.output if captured else None,
                        floor=None if inside else min(int(reserve), max(NANO_MIN_HEADROOM_TOKENS, int(working))),
                        reserve=reserve, accepted=accepted, declared=declared, accepted_at=accepted_at,
                        contradicted_by=contradiction,
                        retired_ack=getattr(evidence, "retired_ack", {}) or (captured.retired_ack if captured else {}))


def record_refusal(usage, plan, refused_input: int = 0, basis: str = "anchored_estimate", *,
                   candidate=None, round_id: str = "") -> bool:
    """Record a typed size refusal for the refusing account: its stated limit (``provider_stated``), else
    ``refused_input`` under ``basis``: ``anchored_estimate``, or ``exact_local_count`` where a local instance counted it.

    ``refused_input`` is the caller's measured provider-unit size of the refused request (anchored on an
    accepted request of this conversation), passed only when measured and outside the Chat start-hint scope.
    A numberless refusal of a request sent inside the reply reservation of a stated total says that total holds the
    reply too, so it is recorded as the total's own refusal (``WindowRecord.admitted`` then charges the reserve).
    Without a stated limit or a measured size a refusal adds no ceiling, and a refusal of one input string's
    length (``INPUT_STRING_LENGTH_CODES``) adds none at any size: it says nothing of the token window.
    A refusal recognised only by HTTP 413 or by a cited number equal to a declared limit (``UNTAUGHT_BASES``)
    writes nothing, the measured size included: neither states a unit, and the host's estimate is no provider
    limit. A refusal typed only by the Z.ai 1261 pair while stating no token window (#3858, ``SizeRefusal.retry_only``)
    records nothing at any size either. Below an accepted input, a token limit teaches only a single-server route
    (local or custom OpenAI-compatible): that server may have restarted smaller. A gateway scope does not name the
    serving endpoint; its smaller statement sizes only this round (``loop_model_call._refusal_window``).
    False means the refusal did not qualify for a window observation or rebind."""
    from ouroboros import capability_evidence as ce
    from ouroboros.context_budget import INPUT_STRING_LENGTH_CODES

    plan = observed_plan(plan, usage, candidate)
    if (plan is None or not getattr(plan, "window_witness_bound", True)
            or (usage.get("_last_llm_provider_answer") or {}).get("basis") in UNTAUGHT_BASES):
        return False
    from ouroboros.size_refusal import for_usage

    chat = planned_chat(candidate)
    refusal = _current_chat_refusal(usage, candidate, round_id) if chat else for_usage(usage)
    if refusal is None:
        return False
    limit = refusal.token_window
    if not limit and refusal.retry_only:
        return False
    if not limit:
        if (int(refused_input or 0) > 0
                and str(usage.get("_last_llm_provider_code") or "").lower() not in INPUT_STRING_LENGTH_CODES):
            record, size = getattr(plan, "window_record", None), int(refused_input)
            total = record.total.tokens if record is not None and record.total else 0
            confirms = total - (record.reserve if record is not None else 0) < size <= total
            if confirms or not chat:
                record_observation(ce.canonical_evidence_root(), observation_scope(plan), basis=basis,
                                   refused=total if confirms else size, kind="total" if confirms else "input")
                return True
        return False
    rows = (ce._read(ce.canonical_evidence_root()).get("main_windows", {}).get(observation_scope(plan)) or {})
    single_server = getattr(plan, "provider", "") in {"local", "openai-compatible"}
    if not single_server and limit < max((row.get("accepted", 0) for row in rows.get("observations", [])), default=0):
        return False
    record_observation(ce.canonical_evidence_root(), observation_scope(plan), refused=limit,
                       kind=refusal.store_kind, basis="provider_stated")
    return True


def planned_chat(candidate) -> bool:
    """The physical candidate was composed for a Chat route, whether or not it can advise a future send."""
    return bool(getattr(getattr(candidate, "physical_context", None), "first_contact_plan", None))


def observed_plan(plan, usage, candidate=None):
    """Keep a witnessed plan's own window; bind Native to its observed account, never Auto's advertisement."""
    physical = getattr(candidate, "physical_context", None)
    if (candidate is None and getattr(plan, "first_contact", "") and getattr(plan, "initial_mode", "") in {"max", "low"}
            and getattr(plan, "provider", "") not in {"claudexor", "anthropic", "local", "gigachat"}):
        return None  # this Chat plan has no physical receipt to bind
    if planned_chat(candidate) and (not physical.window_witness_bound
                                    or physical.window_observation_fp != observation_scope(plan)):
        return None  # the final Chat target differed from the plan; its own soft outcome can still be read
    if getattr(plan, "provider", "") != "claudexor":
        return plan
    from dataclasses import replace
    from ouroboros.capability_evidence import route_fingerprint
    from ouroboros.provider_models import parse_claudexor_model

    observed = usage.get("_model_route") or {}
    if not observed:  # an answer naming no route: its account is the pin's, and on Auto no account's to write
        return plan if getattr(plan, "window_witness_bound", False) and getattr(plan, "account_pin", "") else None
    source, model = parse_claudexor_model(plan.model)
    if observed.get("source") != source or observed.get("model") != model or not all(
            observed.get(key) for key in ("credentialProfileId", "accountFingerprint")):
        return None
    fp = route_fingerprint(provider=plan.provider, model=plan.model, base_url=getattr(plan, "route_base_url", ""),
                           options={"source_id": source, "credential_profile_id": observed["credentialProfileId"],
                                    "account_fingerprint": observed["accountFingerprint"]})
    record = getattr(plan, "window_record", None)
    return replace(plan, route_fp=fp, window_record=replace(record, route_fp=fp) if record else None,
                   model_route=dict(observed), window_witness_bound=True)


def route_window_kwargs(evidence: Any) -> Dict[str, Any]:
    """The plan fields a declared route window carries (a call's RouteFacts): its working reserve and its units."""
    facts = getattr(evidence, "route_facts", {}) or {}
    return {"working_reserve_tokens": facts.get("working_reserve_tokens"), "window_basis": str(facts.get("basis") or "")}


def estimator_ratio(evidence: Any, ratio: float) -> float:
    """An estimator-based window is fitted on raw estimates (density 1); a provider window keeps the measured density."""
    return 1.0 if route_window_kwargs(evidence)["window_basis"] == "estimator" else ratio
