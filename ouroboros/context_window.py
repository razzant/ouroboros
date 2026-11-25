"""Main's route-bound window facts, independent of review capability authority.

Acceptances are lower bounds, never evidence of room for a reply. A numbered
refusal, or a numberless one at the refused request's measured size, caps (never
raises) its kind of limit until a later larger acceptance or the 24-hour evidence
expiry; a cold numberless refusal adds none (``record_refusal``).
The observations share the capability JSON store; old stores need no migration.
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


def record_refusal(usage, plan, refused_input: int = 0, basis: str = "anchored_estimate") -> None:
    """Record a typed size refusal for the refusing account: its stated limit (``provider_stated``), else
    ``refused_input`` under ``basis``: ``anchored_estimate``, or ``exact_local_count`` where a local instance counted it.

    ``refused_input`` is the caller's measured provider-unit size of the refused request (anchored on an
    accepted request of this conversation), passed only when measured: Codex states no number, and a
    route that never records its refusals keeps planning at a larger acceptance and is refused again.
    A numberless refusal of a request sent inside the reply reservation of a stated total says that total holds the
    reply too, so it is recorded as the total's own refusal (``WindowRecord.admitted`` then charges the reserve).
    Without a stated limit or a measured size a refusal adds no ceiling, and a refusal of one input string's
    length (``INPUT_STRING_LENGTH_CODES``) adds none at any size: it says nothing of the token window.
    A refusal recognised only by HTTP 413 or by a cited number equal to a declared limit (``UNTAUGHT_BASES``)
    writes nothing, the measured size included: neither states a unit, and the host's estimate is no provider
    limit. A refusal typed only by the Z.ai 1261 pair while stating no token window (#3858, ``SizeRefusal.retry_only``)
    records nothing at any size either. Below an accepted input, a token limit teaches only a single-server route
    (local or custom OpenAI-compatible): that server may have restarted smaller. A gateway scope does not name the
    serving endpoint; its smaller statement sizes only this round (``loop_model_call._refusal_window``)."""
    from ouroboros import capability_evidence as ce
    from ouroboros.context_budget import INPUT_STRING_LENGTH_CODES

    plan = observed_plan(plan, usage)
    if (plan is None or not getattr(plan, "window_witness_bound", True)
            or (usage.get("_last_llm_provider_answer") or {}).get("basis") in UNTAUGHT_BASES):
        return
    from ouroboros.size_refusal import for_usage
    refusal = for_usage(usage)
    limit = refusal.token_window
    if not limit and refusal.retry_only:
        return
    if not limit:
        if (int(refused_input or 0) > 0
                and str(usage.get("_last_llm_provider_code") or "").lower() not in INPUT_STRING_LENGTH_CODES):
            record, size = getattr(plan, "window_record", None), int(refused_input)
            total = record.total.tokens if record is not None and record.total else 0
            confirms = total - (record.reserve if record is not None else 0) < size <= total
            record_observation(ce.canonical_evidence_root(), observation_scope(plan), basis=basis,
                               refused=total if confirms else size, kind="total" if confirms else "input")
        return
    rows = (ce._read(ce.canonical_evidence_root()).get("main_windows", {}).get(observation_scope(plan)) or {})
    single_server = getattr(plan, "provider", "") in {"local", "openai-compatible"}
    if not single_server and limit < max((row.get("accepted", 0) for row in rows.get("observations", [])), default=0):
        return
    record_observation(ce.canonical_evidence_root(), observation_scope(plan), refused=limit,
                       kind=refusal.store_kind, basis="provider_stated")


def observed_plan(plan, usage):
    """Bind a settled native request to its observed account, never Auto's advertisement."""
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
