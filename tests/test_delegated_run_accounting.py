"""What a delegated run costs, and when that cost becomes durable.

Split verbatim out of ``tests/test_delegated_subagent_transport.py`` by theme. This
module owns subscription-session accounting, the agent-facing cost projection and the
ledger row it must agree with, the settlement receipt and its retry, and the start
request whose claimed substrate the durable record later has to match.
"""

from __future__ import annotations

import json
import pytest
from ouroboros import (
    delegate_custody as dcust,
    usage_accounting as ua,
)

from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _owned_gateway_uses_each_test_transport,
    _plain_ctx,
)


def test_a_subscription_session_settles_at_zero_and_keeps_the_projection_final(tmp_path):
    """A DISCLOSED zero is the free-session case: the money was spent when the plan was
    bought, so the row is final at 0.0 and the projection stays final.

    An UNDISCLOSED spend is not the same fact and must not be written as one. The engine's
    default auth preference is subscription-first with fallback to a paid key, and a route
    can bill by construction — settling those at a confident 0.0/final would hide real
    money from every budget fence while asserting the projection was complete.
    """
    from ouroboros.usage_accounting import record_subscription_session, usage_projection

    disclosed = tmp_path / "disclosed"
    record_subscription_session("s-free", drive_root=disclosed, route="r", task_id="t1",
                                root_task_id="t1", spend_usd=0.0)
    rows = [json.loads(l) for l in (disclosed / "state" / "usage_attempts.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r.get("kind") == "subscription_session")
    assert row["cost_usd"] == 0.0 and row["cost_final"] is True
    assert usage_projection(disclosed)["cost_final"] is True

    charged = tmp_path / "charged"
    record_subscription_session("s-billed", drive_root=charged, route="r", task_id="t1",
                                root_task_id="t1", spend_usd=4.10)
    rows = [json.loads(l) for l in (charged / "state" / "usage_attempts.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r.get("kind") == "subscription_session")
    assert row["cost_usd"] == 4.10, "a real charge must ride the ledger as money"
    assert row["cost_final"] is True

    unknown = tmp_path / "unknown"
    record_subscription_session("s-quiet", drive_root=unknown, route="r", task_id="t1",
                                root_task_id="t1")
    rows = [json.loads(l) for l in (unknown / "state" / "usage_attempts.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r.get("kind") == "subscription_session")
    assert row["cost_final"] is False, "an undisclosed spend is not a proven zero"
    assert row["pricing_known"] is False
    # UNKNOWN must be None, not 0.0. A `cost_final=False` row costing 0.0 adds zero to
    # the projection's `estimated` total, and `not 0.0` is True — so the honest per-row
    # disclosure was invisible one layer up, which reported `cost_final: True` anyway.
    assert row["cost_usd"] is None
    projection = usage_projection(unknown)
    assert projection["cost_final"] is False, "an unknown session must drop finality"
    assert projection["unknown_unmetered"] == 1


def test_the_unmetered_external_row_would_have_dropped_cost_final(tmp_path):
    # The exact reason record_unmetered_external_dispatch must NOT be reused: one such
    # row makes the WHOLE projection non-final.
    ua.record_unmetered_external_dispatch("d1", drive_root=tmp_path, task_id="t1", root_task_id="t1")
    assert ua.usage_projection(tmp_path, root_task_id="t1")["cost_final"] is False


def test_a_session_is_not_counted_as_a_physical_provider_call(tmp_path):
    ua.record_subscription_session("run-2", drive_root=tmp_path, route="some-route", task_id="t2", root_task_id="t2")
    breakdown = ua.usage_breakdown(tmp_path, root_task_id="t2")
    assert breakdown["physical_calls"] == 0
    assert breakdown["subscription_sessions"] == 1


def test_the_agent_facing_cost_tells_the_same_story_as_the_ledger(tmp_path, monkeypatch):
    """`_terminal_payload` is what the nanny RELAYS to its parent, so it must not
    contradict the row. It used to hardcode `$0.00 / final` — the exact shape the
    settlement fix exists to eliminate — so a billed run settled honestly in the ledger
    and then told the reasoning path the work was free.

    This drives the real transport: a stubbed gateway returns a terminal detail carrying
    a spend, and the assertion is on what `delegate_wait` actually returned.
    """
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools.registry import ToolContext

    def _wait_with_spend(spend_field):
        class _Stub:
            def handshake(self, **_kw): return {}
            def get_run(self, rid, **_kw):
                return {"lastSeq": 9, "summary": {"state": "succeeded", **spend_field}}
            def close(self): pass

        monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
        delegate._CUSTODY.clear()
        delegate._CUSTODY["run-1"] = delegate._RunCustody(
            task_id="t-a", route_id="r", model="m", project_id="p", project_owned=False)
        ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
        ctx.task_id = "t-a"
        ctx.task_metadata = {"root_task_id": "t-a"}
        out = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
        delegate._CUSTODY.clear()
        return out

    billed = _wait_with_spend({"spendUsd": 4.10})["cost"]
    assert billed["cost_usd"] == 4.10, "a billed run must not be relayed as free"
    assert "BILLED" in billed["note"]

    undisclosed = _wait_with_spend({})["cost"]
    assert undisclosed["cost_usd"] is None and undisclosed["cost_final"] is False

    free = _wait_with_spend({"spendUsd": 0.0})["cost"]
    assert free["cost_usd"] == 0.0 and free["cost_final"] is True


def test_settlement_reads_the_harnesss_own_spend_field(tmp_path, monkeypatch):
    """Drives `_settle` through the real transport instead of calling the recorder.

    The round-1 test for this called `record_subscription_session(spend_usd=4.10)`
    directly — it constructed the very value it asserted and never entered `_settle`, so
    renaming the wire field to `totallyWrongFieldName` left the suite green. This one
    reads the ledger row that a delegated run actually produced.
    """
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools.registry import ToolContext

    retired = []

    class _Stub:
        def handshake(self, **_kw): return {}
        def get_run(self, rid, **_kw):
            return {"lastSeq": 9, "summary": {"state": "succeeded", "spendUsd": 4.10,
                                              "inputTokens": 10, "outputTokens": 5}}
        def remove_project(self, pid): retired.append(pid)
        def close(self): pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    row = delegate._RunCustody(run_id="run-1", task_id="t-a", route_id="r",
        model="m", project_id="prj-ours", project_owned=True)
    dcust.record_started(tmp_path, row)
    delegate._CUSTODY["run-1"] = row
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "t-a"
    ctx.task_metadata = {"root_task_id": "t-a"}

    json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    delegate._CUSTODY.clear()

    rows = [json.loads(line) for line
            in (tmp_path / "state" / "usage_attempts.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r.get("kind") == "subscription_session")
    assert row["cost_usd"] == 4.10, "the harness's reported spend must reach the ledger"
    assert row["cost_final"] is True
    assert retired == ["prj-ours"], "a registration we created is retired on settle"


def test_d29_applied_credential_profile_reaches_the_durable_record(tmp_path, monkeypatch):
    """D29: the APPLIED credential-profile id + access profile the engine's
    authRoute receipt discloses must land in the durable ledger row AND the
    settled event by default — 'which account paid' answered from the record."""
    payload, row, event = _settled_run(tmp_path, monkeypatch, {
        "state": "succeeded", "spendUsd": 2.5,
        "authRoute": {"profileId": "koshak", "requested": "subscription"},
        "effectiveAccess": "readonly",
    })
    assert row["credential_profile_id"] == "koshak"
    assert row["access_profile"] == "readonly"
    assert event["credential_profile_id"] == "koshak"
    assert event["access_profile"] == "readonly"


def test_d29_absent_authroute_records_empty_never_invented(tmp_path, monkeypatch):
    """Telemetry that predates the receipt records an empty applied profile —
    the fact is disclosed as unknown, never fabricated."""
    _payload, row, event = _settled_run(tmp_path, monkeypatch, {
        "state": "succeeded", "spendUsd": 0.0})
    assert row["credential_profile_id"] == ""
    assert event["credential_profile_id"] == ""


@pytest.mark.parametrize("observed", [
    {"harness_id": "actual-route", "observed_model": "actual-model", "profile_id": "actual-profile"},
    {},
])
def test_final_attempt_identity_survives_settlement_and_parent_delivery(tmp_path, monkeypatch, observed):
    from ouroboros.subagents import subagent_last_delegation

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "canonical-data")
    payload, ledger, event = _settled_run(tmp_path, monkeypatch, {
        "state": "succeeded", "spendUsd": 0.0, "model": "request-echo",
        "authRoute": {"profileId": "old-profile"}, "effectiveAccess": "readonly",
    }, observed=observed)
    for row in [payload, ledger, event]:
        assert row["model"] == observed.get("observed_model", "")
    assert ledger["credential_profile_id"] == observed.get("profile_id", "")
    assert event["credential_profile_id"] == observed.get("profile_id", "")
    assert event["observed_attempt"] == payload["observed_attempt"]
    assert payload["observed_attempt"]["attempt_id"] == "a02"
    assert ledger["cost_usd"] == 0.0 and ledger["cost_final"] is True
    record = subagent_last_delegation()
    assert record["requested_model"] == "m"
    assert record["applied_model"] == observed.get("observed_model", "")
    assert record["applied_profile"] == observed.get("profile_id", "")


def test_the_durable_access_profile_is_the_receipt_never_our_own_request(tmp_path, monkeypatch):
    """The daemon computes `access` as `effectiveAccess ?? the client's own parsed
    request`, so it is our ask reflected back, not a witness. Reading it as a fallback
    wrote the REQUEST into a durable column that promises applied facts."""
    _payload, row, event = _settled_run(tmp_path, monkeypatch, {
        "state": "succeeded", "spendUsd": 0.0, "access": "workspace_write"})
    assert row["access_profile"] == ""
    assert event["access_profile"] == ""


def _observed_summary(tmp_path, summary, observed=None):
    """Give existing positive fixtures a separate engine telemetry artifact.

    The summary values used by the old tests describe the intended observation;
    new mismatch/unknown scenarios pass an independent observation explicitly.
    """
    final = tmp_path / "engine-run" / "final"
    final.mkdir(parents=True, exist_ok=True)
    if observed is None:
        observed = {"harness_id": "r", "observed_model": summary.get("model"),
                    "profile_id": (summary.get("authRoute") or {}).get("profileId")}
    (final / "telemetry.yaml").write_text(json.dumps({
        "run_id": "run-1", "final_attempt_id": "a02",
        "attempts": [{"attempt_id": "a01", "harness_id": "old-route",
                      "observed_model": "old-model", "profile_id": "old-profile"},
                     {"attempt_id": "a02", **observed}],
    }), encoding="utf-8")
    return {**summary, "runDir": str(final.parent)}


def _settled_run(tmp_path, monkeypatch, summary, observed=None):
    """Drive a real `_settle` for `summary`; return (agent payload, ledger row, envelope).

    The `delegate_run_settled` envelope is returned too because it RE-DERIVES the row's
    finality instead of being handed it, so the only thing keeping the two from drifting
    is a test that reads both from the same run.
    """
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools.registry import ToolContext

    summary = _observed_summary(tmp_path, summary, observed)
    class _Stub:
        def handshake(self, **_kw): return {}
        def get_run(self, rid, **_kw): return {"lastSeq": 9, "summary": dict(summary)}
        def remove_project(self, pid): pass
        def close(self): pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        task_id="t-a", route_id="r", model="m", project_id="p", project_owned=True)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "t-a"
    ctx.task_metadata = {"root_task_id": "t-a"}
    payload = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    delegate._CUSTODY.clear()
    rows = [json.loads(line) for line
            in (tmp_path / "state" / "usage_attempts.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line
              in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()]
    return (payload,
            next(r for r in rows if r.get("kind") == "subscription_session"),
            next(e for e in events if e.get("type") == "delegate_run_settled"))


def _waited_run(tmp_path, monkeypatch, summary, requested_model="m", observed=None):
    """Drive one terminal `delegate_wait` for `summary`; return the agent payload.

    Same transport walk as `_settled_run`, with the custody row's REQUESTED
    model under test-control — the requested-vs-applied disclosure compares it
    against the engine summary's own `model`."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools.registry import ToolContext

    summary = _observed_summary(tmp_path, summary, observed)
    class _Stub:
        def handshake(self, **_kw): return {}
        def get_run(self, rid, **_kw): return {"lastSeq": 9, "summary": dict(summary)}
        def remove_project(self, pid): pass
        def close(self): pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        task_id="t-a", route_id="r", model=requested_model,
        project_id="p", project_owned=False)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "t-a"
    ctx.task_metadata = {"root_task_id": "t-a"}
    payload = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    delegate._CUSTODY.clear()
    return payload


def test_the_terminal_payload_carries_the_applied_model_and_the_mismatch_delta(tmp_path, monkeypatch):
    """(owner, 2026-08-04, option A) The APPLIED model from the run summary
    reaches the nanny payload, and a requested≠applied pair — both non-empty —
    is an ADVISORY capability_delta in the review lane's own lexicon, never a
    failure: the run completes on what the engine gave, and engine aliases
    ('sonnet' beside 'claude-opus-5') make strict equality advisory-only."""
    # The settle seam also writes the last-delegation projection into the
    # canonical data plane; isolate it per-test (xdist workers share the
    # pytest-global OUROBOROS_DATA_DIR).
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "proj-data")
    mismatched = _waited_run(tmp_path / "mm", monkeypatch,
                             {"state": "succeeded", "spendUsd": 0.0, "model": "claude-opus-5"},
                             requested_model="sonnet")
    assert mismatched["state"] == "succeeded", "disclosed, never failed"
    assert mismatched["model"] == "claude-opus-5"
    assert mismatched["capability_delta"] == [{
        "kind": "capability_delta",
        "requested": "model sonnet",
        "effective": "model claude-opus-5",
        "reason": "session_route_resolves_its_own_model",
    }]

    # Agreement (or aliases matching exactly): no delta is invented.
    agreed = _waited_run(tmp_path / "ok", monkeypatch,
                         {"state": "succeeded", "spendUsd": 0.0, "model": "sonnet"},
                         requested_model="sonnet")
    assert agreed["model"] == "sonnet" and "capability_delta" not in agreed

    # An engine that disclosed no model: absence stays absence — empty model,
    # no delta, the requested value never dressed up as the applied one.
    silent = _waited_run(tmp_path / "sil", monkeypatch,
                         {"state": "succeeded", "spendUsd": 0.0},
                         requested_model="sonnet")
    assert silent["model"] == "" and "capability_delta" not in silent


def test_the_last_delegation_projection_is_written_at_the_settle_seam(tmp_path, monkeypatch):
    """The Subagents section's «last delegated run» receipt: {ts, route,
    requested_model, applied_model, run_id} in the canonical data plane, and
    the gateway status payload serves it back even with the daemon down."""
    from ouroboros.subagents import subagent_last_delegation

    # Isolated data plane: the projection is keyed off config.DATA_DIR, which
    # xdist workers would otherwise share (and the sibling test writes it too).
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "proj-data")
    _waited_run(tmp_path / "proj", monkeypatch,
                {"state": "succeeded", "spendUsd": 0.0, "model": "claude-opus-5"},
                requested_model="sonnet")
    record = subagent_last_delegation()
    assert record["route"] == "r" and record["run_id"] == "run-1"
    assert record["requested_model"] == "sonnet"
    assert record["applied_model"] == "claude-opus-5"
    assert record["ts"]

    # Idempotent per run: re-reading the SAME terminal run (a parent polling an
    # already-settled delegate_wait) must not re-stamp `ts` — the "N ago" line
    # would otherwise call an old run fresh.
    from ouroboros.subagents import record_last_delegation
    record_last_delegation(route="r", requested_model="sonnet",
                           applied_model="claude-opus-5", run_id="run-1")
    assert subagent_last_delegation()["ts"] == record["ts"]

    # The status endpoint's payload carries the projection unconditionally —
    # it is Ouroboros state, not daemon truth (daemon down ≠ receipt gone).
    from ouroboros.gateway.claudexor_accounts import _status_payload

    payload = _status_payload(False)
    assert payload["subagent_last_delegation"]["run_id"] == "run-1"


def test_no_receipt_on_a_failed_settlement_and_one_after_the_successful_retry(tmp_path, monkeypatch):
    """Negative pin (delta gate 2026-08-05): a settlement whose durable
    obligations FAILED must not mint the last-delegation receipt (it would be
    re-minted on every retry with a fresh ts); the receipt appears exactly when
    a retry settles successfully."""
    import ouroboros.delegate_custody as custody_mod
    from ouroboros.subagents import subagent_last_delegation

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "receipt-data")

    real_settle = custody_mod.settle_run
    outcomes = iter([False, True])

    def _flaky_settle(drive_root, gateway, custody, detail):
        ok = next(outcomes)
        result = real_settle(drive_root, gateway, custody, detail)
        if not ok:
            custody.settled = False
            result = dict(result)
            result["settled"] = False
        return result

    monkeypatch.setattr(custody_mod, "settle_run", _flaky_settle)
    monkeypatch.setattr("ouroboros.tools.delegate.custody.settle_run", _flaky_settle, raising=False)

    _waited_run(tmp_path / "w1", monkeypatch,
                {"state": "succeeded", "spendUsd": 0.0, "model": "claude-opus-5"},
                requested_model="sonnet")
    assert subagent_last_delegation() == {}, "receipt minted on a FAILED settlement"

    _waited_run(tmp_path / "w2", monkeypatch,
                {"state": "succeeded", "spendUsd": 0.0, "model": "claude-opus-5"},
                requested_model="sonnet")
    record = subagent_last_delegation()
    assert record.get("run_id") == "run-1"
    assert record.get("applied_model") == "claude-opus-5"


def test_an_estimated_spend_is_not_a_settled_one(tmp_path, monkeypatch):
    """`spendUsd` is half the disclosure; `spendEstimated` is the other half.

    The engine really populates it (`packages/schema/src/control.ts`: "True when settled
    cash is estimated rather than exact"), and 8 of 60 live `/v2/runs` rows carried it —
    all of them as an estimated ZERO, which is the trap: reading the amount alone wrote a
    charge nobody had settled into the ledger as `cost_final=True` and relayed it to the
    agent as an already-paid subscription session, final.

    All three surfaces are asserted, because the defect this replaces was a fix that
    landed on one of them. The estimated-ZERO case in particular is what proves the
    projection: an estimated $0.00 adds nothing to `estimated_usd`, so a finality test
    that sums dollars instead of counting rows keeps reporting `cost_final: True`.

    Both AMOUNTS are asserted, because the harm this commit names is an estimated CHARGE
    written as already-paid. Testing only the estimated zero left the fix scoped to it:
    `if estimated and spend == 0`, `not (spend_estimated and spend_usd == 0.0)` and an
    `estimated_rows` that only counts free rows all passed a zero-only suite, so a build
    that still relayed an estimated $4.10 as a closed book was green on every surface.
    """
    estimated, row, _ = _settled_run(tmp_path / "est", monkeypatch, {
        "state": "succeeded", "spendUsd": 0, "spendEstimated": True,
        "inputTokens": 800318, "outputTokens": 4851})
    assert row["cost_usd"] == 0.0, "the amount is still the best fact anyone has"
    assert row["cost_final"] is False, "an ESTIMATED charge is not a settled one"
    assert estimated["cost"]["cost_final"] is False
    assert "ESTIMAT" in estimated["cost"]["note"].upper()
    assert ua.usage_projection(tmp_path / "est")["cost_final"] is False, \
        "one non-final row means the projection is not final, however little it cost"
    assert ua.usage_projection(tmp_path / "est")["estimated_usd"] == 0.0, \
        "and it is not final BECAUSE of the row, not because of the dollars"

    # The MONEY half of the same defect. An estimate with a real amount must ride the
    # ledger as money and still refuse finality on all three surfaces.
    charged, row, _ = _settled_run(tmp_path / "chg", monkeypatch, {
        "state": "succeeded", "spendUsd": 4.10, "spendEstimated": True,
        "inputTokens": 800318, "outputTokens": 4851})
    assert row["cost_usd"] == 4.10, "an estimate is still the best fact anyone has"
    assert row["cost_final"] is False, "and $4.10 unsettled is not $4.10 paid"
    assert charged["cost"]["cost_usd"] == 4.10
    assert charged["cost"]["cost_final"] is False
    charged_projection = ua.usage_projection(tmp_path / "chg")
    assert charged_projection["estimated_usd"] == 4.10, "it lands in the estimated bucket"
    assert charged_projection["confirmed_usd"] == 0.0, "and never in the confirmed one"
    assert charged_projection["cost_final"] is False

    # The control: the same amount, SETTLED, is the free-session case this row kind was
    # created for and must still leave the projection final.
    settled, row, _ = _settled_run(tmp_path / "set", monkeypatch, {
        "state": "succeeded", "spendUsd": 0, "spendEstimated": False,
        "inputTokens": 800318, "outputTokens": 4851})
    assert row["cost_final"] is True and row["cost_usd"] == 0.0
    assert settled["cost"]["cost_final"] is True
    assert ua.usage_projection(tmp_path / "set")["cost_final"] is True


@pytest.mark.parametrize("summary, cost_usd, final, disclosed, estimated", [
    # UNDISCLOSED: no amount. The envelope must not invent a zero, and the flag beside it
    # must be a definite False rather than whatever silence happened to produce.
    ({"state": "succeeded"}, None, False, False, False),
    ({"state": "succeeded", "spendUsd": 0, "spendEstimated": True}, 0.0, False, True, True),
    ({"state": "succeeded", "spendUsd": 4.10, "spendEstimated": True}, 4.10, False, True, True),
    ({"state": "succeeded", "spendUsd": 0}, 0.0, True, True, False),
    ({"state": "succeeded", "spendUsd": 4.10}, 4.10, True, True, False),
])
def test_the_settled_envelope_tells_the_same_story_as_the_row(
        tmp_path, monkeypatch, summary, cost_usd, final, disclosed, estimated):
    """`delegate_run_settled` RE-DERIVES the finality the recorder just decided.

    Nothing in the tree referenced `delegate_run_settled`, `spend_estimated` or
    `spend_disclosed` — `grep -rn` over `tests/` returned nothing — so re-zeroing an
    undisclosed `cost_usd`, dropping `not estimated` from the envelope's finality, and
    deleting the `spend_estimated` field ALL passed. Two writers of one fact with no
    reader watching is the drift this pins shut: the envelope is asserted against the row
    from the SAME run, in every cash state, so the two cannot part company silently.
    """
    _, row, envelope = _settled_run(tmp_path, monkeypatch, summary)
    assert envelope["cost_usd"] == cost_usd, "the envelope reports the row's own amount"
    assert envelope["cost_final"] is final
    assert envelope["spend_disclosed"] is disclosed
    assert envelope["spend_estimated"] is estimated
    assert envelope["cost_usd"] == row["cost_usd"], "one envelope, one story"
    assert envelope["cost_final"] == row["cost_final"], "and one finality"


def test_an_unreported_token_count_is_unknown_not_zero(tmp_path, monkeypatch):
    """The control schema: "null until a harness reported it — never render null as 0".

    Live `/v2/runs` rows really carry `inputTokens: null`, and `int(x or 0)` made a run
    that reported nothing indistinguishable in the ledger from one that genuinely used
    zero. Same rule v6.87.35 established for cost, one axis over.

    That schema sentence governs THREE fields, and `cachedInputTokens` is the third: 28 of
    60 rows on a live `/v2/runs` page carry it non-null, 27 of them non-zero (one at
    34.8M). Reading only two left the row with no `cached_tokens` key at all, which
    `_breakdown_bucket` renders as 0 beside a six-figure prompt count — exactly the
    render-unknown-as-zero shape its two siblings had just stopped doing.
    """
    _, silent, _ = _settled_run(tmp_path / "silent", monkeypatch, {
        "state": "succeeded", "spendUsd": 0, "inputTokens": None, "outputTokens": None,
        "cachedInputTokens": None})
    assert silent["prompt_tokens"] is None and silent["completion_tokens"] is None, \
        "a run that reported nothing must not be written as a run that used zero"
    assert silent["cached_tokens"] is None, "and the third field obeys the same sentence"

    _, real_zero, _ = _settled_run(tmp_path / "zero", monkeypatch, {
        "state": "succeeded", "spendUsd": 0, "inputTokens": 0, "outputTokens": 0,
        "cachedInputTokens": 0})
    assert real_zero["prompt_tokens"] == 0 and real_zero["completion_tokens"] == 0, \
        "a disclosed zero is a fact and must survive as 0, not become None"
    assert real_zero["cached_tokens"] == 0

    _, counted, _ = _settled_run(tmp_path / "counted", monkeypatch, {
        "state": "succeeded", "spendUsd": 0, "inputTokens": 10, "outputTokens": 5,
        "cachedInputTokens": 34808493})
    assert (counted["prompt_tokens"], counted["completion_tokens"]) == (10, 5)
    assert counted["cached_tokens"] == 34808493, \
        "a reported cache hit is real usage and must reach the ledger, not be dropped"
    # It reaches the reader that renders it, and is NOT folded into the grand total —
    # required, because cached is a SUBSET of input for some harnesses and disjoint for
    # others, so a sum across them means nothing.
    bucket = ua.usage_breakdown(tmp_path / "counted")
    assert bucket["cached_tokens"] == 34808493
    assert bucket["total_tokens"] == 15


def test_the_start_request_asks_for_the_substrate_it_claims(tmp_path, monkeypatch):
    """`authPreference` defaults to `auto` = subscription-first WITH fallback to a paid
    key. Asking explicitly is the difference between claiming a free session and getting
    one. Round 1 asserted this nowhere — `grep authPreference tests/` returned nothing."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    seen = {}

    class _Stub:
        def handshake(self, **_kw): return {}
        def agent_capabilities(self):
            return {"harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                                   "accessProfilesSupported": ["readonly"]}]}
        def quota_snapshots(self): return []
        def find_project_id(self, root): return "prj-existing"
        def start_run(self, request, *, idempotency_key=""):
            seen["request"] = request
            return {"runId": "run-1"}
        def close(self): pass

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    delegate._delegate_start(_plain_ctx(tmp_path), "x")
    delegate._CUSTODY.clear()
    assert seen["request"]["authPreference"] == "subscription"
    # And the configured route is PINNED as the explicit one-element pool:
    # `primaryHarness` alone only fronts the engine's auto-pool, so without
    # this the child could fail over onto a harness the owner never named.
    assert seen["request"]["harnesses"] == ["some-route"]
    assert seen["request"]["primaryHarness"] == "some-route"


def test_a_202_handle_without_a_run_id_is_a_live_run_not_a_failure(tmp_path, monkeypatch):
    """A 202 answers with `jobId` and no `runId` when the run has not bound a run dir
    inside the daemon's start timeout. The run IS enqueued and will execute; discarding
    the handle left it live, unwaitable and uncancellable, and invited a duplicate."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    class _Stub:
        def handshake(self, **_kw): return {}
        def agent_capabilities(self):
            return {"harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                                   "accessProfilesSupported": ["readonly"]}]}
        def quota_snapshots(self): return []
        def find_project_id(self, root): return "prj-existing"
        def start_run(self, request, *, idempotency_key=""): return {"jobId": "job-42"}   # 202: no runId yet
        def close(self): pass

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    out = json.loads(delegate._delegate_start(_plain_ctx(tmp_path), "x"))
    assert out["status"] == "started", out
    assert out["run_id"] == "job-42"
    assert "job-42" in delegate._CUSTODY, "the run must be in custody or nobody can cancel it"
    delegate._CUSTODY.clear()


def test_a_failed_ledger_write_leaves_the_session_retryable(tmp_path, monkeypatch):
    """The ledger lock can time out under worker concurrency. That is a transient, not a
    decision — marking custody settled would burn the only chance to record the row."""
    import ouroboros.tools.delegate as delegate
    import ouroboros.usage_accounting as ua
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools.registry import ToolContext

    retired = []

    class _Stub:
        def handshake(self, **_kw): return {}
        def get_run(self, rid, **_kw):
            return {"lastSeq": 9, "summary": {"state": "succeeded", "spendUsd": 0.0}}
        def remove_project(self, pid): retired.append(pid)
        def close(self): pass

    def _boom(*a, **k):
        raise ua.UsageAccountingError("usage accounting lock unavailable")

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    monkeypatch.setattr(ua, "record_subscription_session", _boom)
    delegate._CUSTODY.clear()
    custody = delegate._RunCustody(run_id="run-1", task_id="t-a", route_id="r",
        model="m", project_id="prj-ours", project_owned=True)
    dcust.record_started(tmp_path, custody)
    delegate._CUSTODY["run-1"] = custody
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "t-a"
    ctx.task_metadata = {"root_task_id": "t-a"}

    json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    assert custody.settled is False, "a lost write must stay retryable"
    # Retirement is INDEPENDENT of the ledger write: the round-2 commit claimed
    # this while the fixture owned no project, so deleting the call stayed
    # green — a leak per failed settle, and a halted run never settles again.
    assert retired == ["prj-ours"], "owned registration retired even on failure"
    delegate._CUSTODY.clear()
