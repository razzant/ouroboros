"""Main's route-bound window facts, independent of review capability authority.

Acceptances are lower bounds, never evidence of room for a reply. A numbered
refusal constrains planning until a later larger acceptance or evidence expiry.
The observations share the capability JSON store; old stores need no migration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class WindowLimit:
    tokens: int
    kind: str  # input | total | output
    source: str
    observed_at: str = ""
    acceptance_can_raise: bool = True


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

    @property
    def usable(self) -> Optional[int]:
        limits = ([self.input.tokens] if self.input else []) + (
            [max(0, self.total.tokens - self.reserve)] if self.total else [])
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


def record_observation(root, route_fp: str, *, accepted: int = 0, refused: int = 0,
                       kind: str = "total", reserve: int = 0, transmitted: int = 0) -> None:
    from ouroboros import capability_evidence as ce

    if not route_fp or not (accepted > 0 or refused > 0):
        return
    with ce._STORE_LOCK:
        data = ce._load(root)
        record = data.setdefault("main_windows", {}).setdefault(route_fp, {})
        rows = record.setdefault("observations", [])
        prior = list(rows)
        # Retain immutable frontier observations, including a later acceptance
        # below a historic maximum that nevertheless supersedes a newer refusal.
        last_refusal = next((row for row in reversed(rows) if row.get("refused")), {})
        high = max((row.get("accepted", 0) for row in rows), default=0)
        supersedes = accepted > last_refusal.get("refused", accepted) and not any(
            row.get("accepted", 0) > last_refusal.get("refused", 0)
            for row in rows[rows.index(last_refusal) + 1:]) if last_refusal else False
        if refused or accepted > high or supersedes:
            rows.append({"accepted": accepted, "refused": refused, "kind": "input" if accepted else kind,
                         "reserve": reserve, "transmitted": transmitted, "ts": ce.utc_now_iso()})
        keep = set()
        largest = max(range(len(rows)), key=lambda i: rows[i].get("accepted", 0), default=None)
        if largest is not None and rows[largest].get("accepted"):
            keep.add(largest)
        newest = next((i for i in reversed(range(len(rows))) if rows[i].get("refused")), None)
        if newest is not None:
            keep.add(newest)
            later = next((i for i in range(newest + 1, len(rows))
                          if rows[i].get("accepted", 0) > rows[newest]["refused"]), None)
            if later is not None:
                keep.add(later)
        record["observations"] = [rows[i] for i in sorted(keep)]
        if record["observations"] != prior:
            ce._save(root, data)


def observation_scope(plan) -> str:
    record = getattr(plan, "window_record", None)
    return record.route_fp if record is not None else str(getattr(plan, "route_fp", "") or "")


def route_observation_scope(route, settings) -> str:
    """Main's account scope; capability acknowledgements and density keep their own keys."""
    import hashlib
    import json
    from ouroboros.capability_evidence import route_fingerprint
    from ouroboros.response_limits import dispatch_target

    if route["provider"] == "claudexor":
        return route_fingerprint(provider=route["provider"], model=route["model"],
            base_url=route["base_url"], options=route["options"]) if route["options"].get("account_fingerprint") else ""
    target = dict(route, keyless=True) if route["use_local"] else dispatch_target(route["model"], settings)
    credentials = {key: target.get(key, "") for key in ("api_key", "user", "password", "scope")}
    if not any(credentials.values()) and not target.get("keyless"):
        return ""
    payload = {"provider": target["provider"], "model": route["model"],
               "base_url": target.get("base_url", ""), "credentials": credentials}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def resolve_window(evidence, reserve: int, *, root=None, bound: bool = True,
                   observation_fp=None, accepted_input_tokens: int = 0) -> WindowRecord:
    from ouroboros import capability_evidence as ce

    fp = observation_scope(evidence) if observation_fp is None else observation_fp
    rows = ((ce._load(root or ce.canonical_evidence_root()).get("main_windows", {}).get(fp) or {})
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
                                      not bool(getattr(evidence, "route_facts", {}))))
    declared = tuple(limits.values())
    server = getattr(evidence, "server_window", {})
    contradiction = WindowLimit(**server) if server else captured.contradicted_by if captured else None
    if contradiction and (ce._age_seconds(contradiction.observed_at) > ce._CONFIRMED_TTL_SEC or not any(
            limit.source in {ce.SOURCE_OWNER_ACK, ce.SOURCE_USER_SETTING} and
            (limit.tokens, limit.kind) != (contradiction.tokens, contradiction.kind) for limit in declared)):
        contradiction = None
    # Only a disproved stated limit yields to an accepted INPUT. An acceptance
    # below a stated total proves no reply room; without a statement capacity stays unknown.
    if any(limit.acceptance_can_raise and accepted > limit.tokens for limit in limits.values()):
        limits = {kind: limit for kind, limit in limits.items() if not limit.acceptance_can_raise}
        limits["input"] = WindowLimit(accepted, "input", "accepted", accepted_at)
    for index in range(len(rows) - 1, -1, -1):
        row = rows[index]
        if row.get("refused"):
            if ce._age_seconds(row["ts"]) <= ce._CONFIRMED_TTL_SEC and not any(
                    later.get("accepted", 0) > row["refused"] for later in rows[index + 1:]):
                current = limits.get(row["kind"])
                if current is None or row["refused"] < current.tokens:
                    limits[row["kind"]] = WindowLimit(row["refused"], row["kind"], "refusal", row["ts"])
            break
    output = getattr(evidence, "response_limit", {}) or {}
    maximum = output.get("max_output_tokens", 0)
    return WindowRecord(fp, limits.get("input"), limits.get("total"),
                        WindowLimit(maximum, "output", output.get("source", "metadata"),
                                    output.get("observed_at", "")) if maximum and not output.get("stale")
                        else captured.output if captured else None,
                        reserve=reserve, accepted=accepted, declared=declared, accepted_at=accepted_at,
                        contradicted_by=contradiction,
                        retired_ack=getattr(evidence, "retired_ack", {}) or (captured.retired_ack if captured else {}))


def record_refusal(usage, plan) -> None:
    from ouroboros import capability_evidence as ce

    plan = observed_plan(plan, usage)
    if plan is None:
        return
    message = str(usage.get("_last_llm_provider_message") or "")
    limit = ce._parse_ctx_limit_number(message)
    if not limit or not getattr(plan, "window_witness_bound", True):
        return
    # Sglang shares its input bound with generation, so its length refusal stays total.
    input_only = any(marker in message.lower() for marker in ("prompt is too long", "maximum input", "input token count"))
    record_observation(ce.canonical_evidence_root(), observation_scope(plan), refused=limit,
                       kind="input" if input_only else "total")


def observed_plan(plan, usage):
    """Bind a settled native request to its observed account, never Auto's advertisement."""
    if getattr(plan, "provider", "") != "claudexor":
        return plan
    from dataclasses import replace
    from ouroboros.capability_evidence import route_fingerprint
    from ouroboros.provider_models import parse_claudexor_model

    observed = usage.get("_model_route") or {}
    if not observed:
        return plan if getattr(plan, "window_witness_bound", False) else None
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
