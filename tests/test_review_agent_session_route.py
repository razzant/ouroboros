"""Phase 5 review lanes: the agent_session route's typed verdict and routed slots.

Split by theme out of the original giant of the same name. This module owns the
typed verdict contract (schema conformance as the gate, light extraction, strict
whole-answer parsing) and the routes on slots: typed refusals, quorum shape, the
configured route parsing and the durable canonicalized transcript.
"""

import json
import threading
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody

from ouroboros.review_execution import (
    REVIEW_SESSION_ROUTE_ENV,
    ReviewRouteKind,
    canonicalize_session_verdict,
)
from ouroboros.review_substrate import (
    ReviewSlot,
    reviewer_slots,
    run_review_request,
    scope_reviewer_slots,
)
from ouroboros.triad_review import empty_array_is_verified_clean

from tests._review_session_route_shared import _owned_gateway_uses_each_test_transport as __owned_gateway_uses_each_test_transport
from tests._review_session_route_shared import fake_route as __fake_route

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_owned_gateway_uses_each_test_transport = __owned_gateway_uses_each_test_transport
fake_route = __fake_route

from tests._review_session_route_shared import (
    FakeGateway,
    FakeLLM,
    _agent_request,
    _agent_slot,
    _run_session_directly,
    _terminal_detail,
)

# ---------------------------------------------------------------------------
# 5.4 — the typed verdict
# ---------------------------------------------------------------------------


def test_schema_conformant_clean_verdict_survives(tmp_path, fake_route):
    """The full clean path: schema asked, conformance passed, empty findings —
    the actor's raw text is the bare `[]` the constitutional predicate accepts."""
    llm = FakeLLM()
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=llm)
    actor = result.actors[0]
    assert actor["status"] == "ok"
    assert actor["raw_text"] == "[]"
    assert empty_array_is_verified_clean(actor["raw_text"])
    assert actor["usage"]["verdict_method"] == "schema"
    assert llm.calls == []  # no extraction spent, no api pack sent

    gateway = fake_route.instances[0]
    start = gateway.start_requests[0]
    assert start["authPreference"] == "subscription"
    assert start["access"] == "readonly" and start["mode"] == "ask"
    assert start["primaryHarness"] == "fake-review"
    assert start["model"] == "fake-small" and start["effort"] == "low"
    assert start["maxSeconds"] == 30
    assert "unwrapped substantive deliverable" in start["prompt"]
    # The manifest declared a non-interactive structured-output transport, so
    # the schema was asked on the EFFECTIVE route (D19).
    assert "outputSchema" in start
    assert gateway.start_keys[0]  # the invocation id rode the wire


def test_structured_session_compares_the_parsed_model_not_the_harness_spec(
    tmp_path, fake_route,
):
    fake_route.detail = _terminal_detail('{"findings": []}', conformance="passed", model="anthropic::claude-fable-5")
    result = run_review_request(
        _agent_request(),
        slots=[_agent_slot(
            model="fake-review=anthropic/claude-fable-5",
            session_target="fake-review=anthropic/claude-fable-5",
        )],
        drive_root=tmp_path,
        llm=FakeLLM(),
    )

    reasons = {
        item.get("reason")
        for item in result.actors[0]["usage"].get("capability_delta", [])
    }
    assert "session_route_resolves_its_own_model" not in reasons


def test_schema_is_not_asked_when_the_manifest_does_not_declare_it(tmp_path, fake_route):
    fake_route.manifest_capabilities = {}
    fake_route.detail = _terminal_detail("[]")
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=FakeLLM())
    assert "outputSchema" not in fake_route.instances[0].start_requests[0]
    actor = result.actors[0]
    assert actor["raw_text"] == "[]"
    assert actor["usage"]["verdict_method"] == "strict"
    # Landing below the requested structured verdict is disclosed (D4).
    deltas = actor["usage"]["capability_delta"]
    assert any(d["reason"] == "schema_unavailable_on_effective_route" for d in deltas)


def test_report_session_asks_no_schema_and_records_no_schema_landing(tmp_path, fake_route):
    """A report-shaped surface (deep self-review) asks for NO output schema, so
    neither its absence nor a missing outputConformance is a capability delta
    — the host never requested one — and the prose passes through verbatim."""
    fake_route.manifest_capabilities = {}
    report = "# Deep self-review\n\nCRITICAL: loop.py finalization race.\n"
    fake_route.detail = _terminal_detail(report)
    result = run_review_request(
        _agent_request(surface="deep_self_review", call_type="deep_self_review"),
        slots=[_agent_slot()], drive_root=tmp_path, llm=FakeLLM(),
    )
    assert "outputSchema" not in fake_route.instances[0].start_requests[0]
    actor = result.actors[0]
    assert actor["raw_text"] == report
    assert actor["usage"]["verdict_method"] == "report"
    reasons = [d.get("reason", "") for d in actor["usage"].get("capability_delta", [])]
    assert not [r for r in reasons if str(r).startswith("schema_")], reasons


def test_the_run_request_pins_the_route_as_the_explicit_pool(tmp_path, fake_route):
    """CLAIM-1 regression: `primaryHarness` alone only fronts the engine's
    auto-pool (orchestrator orderPool) — other doctor-OK harnesses stay
    eligible and a plain ask run fails over across them. The honest pinning
    contract is the explicit one-element `harnesses` pool; `n` must stay off
    the wire because the engine refuses it on plain ask-mode runs."""
    run_review_request(_agent_request(), slots=[_agent_slot()],
                       drive_root=tmp_path, llm=FakeLLM())
    start = fake_route.instances[0].start_requests[0]
    assert start["harnesses"] == ["fake-review"]
    assert start["primaryHarness"] == "fake-review"
    assert "n" not in start  # refused by mode/strategy coherence on ask runs


def test_schema_is_not_asked_on_an_interactive_transport_route(tmp_path, fake_route):
    """D19: the manifest's `json_schema_output` describes the ADAPTER, not the
    transport. An interactive-capable lane is refused `outputSchema` outright
    by the engine under a daemon (it always arms an interaction channel), so
    asking would kill the whole run typed. The effective-route decision must
    therefore look at BOTH flags — and the landing below the ask is loud."""
    fake_route.manifest_capabilities = {"json_schema_output": True, "interactive": True}
    fake_route.detail = _terminal_detail("[]")
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=FakeLLM())
    assert "outputSchema" not in fake_route.instances[0].start_requests[0]
    actor = result.actors[0]
    assert actor["raw_text"] == "[]"  # the verdict still lands, via strict parse
    deltas = actor["usage"]["capability_delta"]
    assert any(d["reason"] == "schema_unavailable_on_effective_route" for d in deltas)


def test_a_run_reported_off_the_pinned_route_is_disclosed(tmp_path, fake_route):
    """Belt over the pin: the engine's own receipt of the pool the run used
    must echo the pinned route; drift surfaces as a capability_delta, never as
    a quietly accepted substitute route."""
    fake_route.detail = _terminal_detail('{"findings": []}', conformance="passed")
    fake_route.detail["summary"]["harnesses"] = ["other-route"]
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=FakeLLM())
    deltas = result.actors[0]["usage"]["capability_delta"]
    assert any(d["reason"] == "session_ran_off_pinned_route" for d in deltas)


def test_conformance_gate_is_the_gate_not_run_success(tmp_path, fake_route):
    """A successful run WITHOUT outputConformance == passed is narrative: the
    strict parser judges it, never the run's success."""
    fake_route.detail = _terminal_detail('{"findings": []}')  # no conformance
    llm = FakeLLM(reply="UNEXTRACTABLE")
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=llm)
    actor = result.actors[0]
    # '{"findings": []}' is not the contract's array shape: strict refuses, the
    # extractor was consulted and refused too, so the raw narrative survives
    # for forensics instead of being blessed as clean.
    assert actor["raw_text"] == '{"findings": []}'
    assert not empty_array_is_verified_clean(actor["raw_text"])
    assert actor["usage"]["verdict_method"] == "unparsed"
    # The schema was asked and not conformed: the delta is disclosed.
    deltas = actor["usage"]["capability_delta"]
    assert any(d["reason"] == "schema_not_conformed_on_effective_route" for d in deltas)


def test_narrative_clean_verdict_survives_via_light_extraction(tmp_path, fake_route):
    """D19's whole point: a session that says 'no issues found' in prose now
    yields a recognised CLEAN verdict — via the light model, not via loosening
    the constitutional predicate."""
    fake_route.manifest_capabilities = {}
    fake_route.detail = _terminal_detail(
        "I read the staged diff and the touched files. Everything is consistent; "
        "I found no issues worth reporting."
    )
    llm = FakeLLM(reply="[]")
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=llm)
    actor = result.actors[0]
    assert actor["raw_text"] == "[]"
    assert empty_array_is_verified_clean(actor["raw_text"])
    assert actor["usage"]["verdict_method"] == "light_model_extraction"
    # Exactly one session launch and exactly one extraction call.
    assert len(fake_route.instances[0].start_requests) == 1
    assert len(llm.calls) == 1
    assert llm.calls[0]["reasoning_effort"] == "low"
    # The delta is disclosed on the actor record AND durably.
    deltas = actor["usage"]["capability_delta"]
    assert any(d["reason"] == "extraction_instead_of_schema" for d in deltas)
    rows = [json.loads(line) for line in
            (tmp_path / "logs" / "events.jsonl").read_text().splitlines()]
    assert any(r.get("type") == "review_slot_capability_delta"
               and r.get("verdict_method") == "light_model_extraction" for r in rows)


def test_narrative_findings_pass_through_strict_untouched(tmp_path, fake_route):
    findings = '[{"item": "x", "verdict": "FAIL", "severity": "critical", "reason": "r"}]'
    fake_route.manifest_capabilities = {}
    fake_route.detail = _terminal_detail(findings)
    llm = FakeLLM()
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=llm)
    actor = result.actors[0]
    assert actor["raw_text"] == findings
    assert actor["usage"]["verdict_method"] == "strict"
    assert llm.calls == []


def test_strict_requires_the_whole_answer_never_an_embedded_array(tmp_path, fake_route):
    """A transcript that merely CONTAINS a JSON array is not a verdict.

    ``_strictly_parseable`` used to SCAN with ``extract_json_array``, so a
    refusal that quoted the contract's own example was passed through
    byte-identical as a TRUSTED ``strict`` verdict and the extraction rail never
    ran. Downstream that is not cosmetic: the quoted example carries ``item`` and
    ``verdict`` keys, so the actor projected ``parse_status='valid'`` /
    ``semantic_verdict='PASS'`` — a clean quorum vote from a session that
    reviewed nothing.
    """
    from ouroboros.review_execution import _strictly_parseable

    refusal = (
        'I reviewed NOTHING: the sandbox denied every read. The contract asked '
        'for entries like [{"item": "P1 honesty", "verdict": "PASS", '
        '"severity": "advisory", "reason": "example only"}]. Please re-run.'
    )
    assert not _strictly_parseable(refusal)

    # The bare payload — the shape the contract actually asks for — stays strict.
    assert _strictly_parseable('[{"item": "x", "verdict": "FAIL"}]')
    assert _strictly_parseable("[]")

    # End to end: the refusal reaches extraction instead of being trusted, and
    # UNEXTRACTABLE keeps the raw narrative rather than inventing a verdict.
    fake_route.manifest_capabilities = {}
    fake_route.detail = _terminal_detail(refusal)
    llm = FakeLLM(reply="UNEXTRACTABLE")
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=llm)
    actor = result.actors[0]
    assert actor["usage"]["verdict_method"] == "unparsed"
    assert len(llm.calls) == 1  # extraction was consulted, not short-circuited


def test_extraction_never_fabricates_a_clean_verdict():
    """A refusal canonicalizes to UNEXTRACTABLE, never to []: the light model's
    contract forbids blessing a non-review as clean, and an UNEXTRACTABLE reply
    leaves the raw narrative in place (upstream marks it a parse failure)."""
    llm = FakeLLM(reply="UNEXTRACTABLE")
    text, method, _usage = canonicalize_session_verdict(
        "I cannot review this diff.", conformance_passed=False, llm=llm)
    assert text == "I cannot review this diff."
    assert method == "unparsed"


def test_a_fenced_verdict_stays_unparsed_at_this_layer():
    """Disclosed residual (audit 2026-08-05): a verdict wrapped in a ```json
    fence that only the coordinator's downstream scanner parses is telemetered
    `unparsed` HERE — labeling it would need a duplicate parser (drift) or a
    backward import of the coordinator (the one-way seam ARCHITECTURE pins).
    The refusal-quoting-an-array case also stays `unparsed`."""
    llm = FakeLLM(reply="UNEXTRACTABLE")
    fenced = (
        "Review complete. My findings:\n```json\n"
        '[{"item": "x", "verdict": "FAIL", "severity": "critical", "reason": "r"}]\n'
        "```\nEnd of review."
    )
    text, method, _usage = canonicalize_session_verdict(
        fenced, conformance_passed=False, llm=llm)
    assert method == "unparsed"
    assert text == fenced
    assert len(llm.calls) == 1  # extraction was still consulted first (D19 order)

    quoted = ('I reviewed NOTHING. The contract asked for entries like '
              '[{"item": "a", "verdict": "PASS", "severity": "advisory", "reason": "e"}].')
    _text2, method2, _u2 = canonicalize_session_verdict(
        quoted, conformance_passed=False, llm=FakeLLM(reply="UNEXTRACTABLE"))
    assert method2 == "unparsed"

def test_extraction_runs_under_its_own_physical_rail():
    """The extraction is NOT a review call: it claims no send from the reviewing
    actor's two-physical-send rail (D19)."""
    from ouroboros.usage_accounting import physical_attempt_limit

    class RailProbeLLM(FakeLLM):
        def chat(self, **kwargs):
            from ouroboros.usage_accounting import _claim_physical_dispatch

            _claim_physical_dispatch()  # what a real provider send does
            return super().chat(**kwargs)

    llm = RailProbeLLM(reply="[]")
    with physical_attempt_limit(0):  # the actor rail is EXHAUSTED
        text, method, _usage = canonicalize_session_verdict(
            "narrative: clean.", conformance_passed=False, llm=llm)
    assert method == "light_model_extraction" and text == "[]"


def test_extraction_reads_the_whole_transcript_no_window():
    """CLAIM-2 regression: the extraction rail must see the artifact WHOLE. A
    head+tail window silently dropped everything mid-transcript, so a finding
    reported there never reached the light model — and a faithful "[]" over
    the visible cut fabricated a verified-clean verdict."""
    finding = ("CRITICAL FINDING: the retry path drops the durable ledger row "
               "(reported verbatim, mid-transcript).")
    raw = ("transcript begins\n" + "context line\n" * 500      # > old 4k head
           + finding + "\n"
           + "trailing tool output\n" * 6_000)                 # > old 60k tail
    canonical = ('[{"item": "retry path", "verdict": "FAIL", '
                 '"severity": "critical", "reason": "drops the ledger row"}]')
    llm = FakeLLM(reply=canonical)
    text, method, _usage = canonicalize_session_verdict(
        raw, conformance_passed=False, llm=llm)
    assert method == "light_model_extraction"
    assert text == canonical
    prompt = llm.calls[0]["messages"][0]["content"]
    assert finding in prompt  # the mid-transcript finding reached extraction


def test_oversized_transcript_is_typed_extraction_incomplete_never_clean(tmp_path, fake_route):
    """The single-send rail has one hard bound; past it extraction REFUSES with
    the typed disposition — no send at all, so no windowed read a light model
    could faithfully canonicalize into a clean verdict."""
    from ouroboros.review_execution import _EXTRACT_MAX_CHARS

    big = "narrative that never parses\n" * (_EXTRACT_MAX_CHARS // 20)
    assert len(big) > _EXTRACT_MAX_CHARS
    llm = FakeLLM(reply="[]")  # would fabricate clean IF it were consulted
    text, method, _usage = canonicalize_session_verdict(
        big, conformance_passed=False, llm=llm)
    assert method == "extraction_incomplete"
    assert text == big                       # forensics keep the raw transcript
    assert not empty_array_is_verified_clean(text)
    assert llm.calls == []                   # the light model was never shown a cut

    # End to end: the slot discloses the refusal as a capability_delta and the
    # verdict method rides the actor record — never a silent clean.
    fake_route.manifest_capabilities = {}
    fake_route.detail = _terminal_detail(big)
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=llm)
    actor = result.actors[0]
    assert actor["usage"]["verdict_method"] == "extraction_incomplete"
    assert not empty_array_is_verified_clean(actor["raw_text"])
    deltas = actor["usage"]["capability_delta"]
    assert any(d["reason"] == "extraction_incomplete_transcript_exceeds_bound"
               for d in deltas)
    assert llm.calls == []

# ---------------------------------------------------------------------------
# 5.1/5.3 — routes on slots, typed refusals, quorum shape
# ---------------------------------------------------------------------------


def test_unconfigured_session_route_is_a_typed_refusal(tmp_path, fake_route, monkeypatch):
    monkeypatch.delenv(REVIEW_SESSION_ROUTE_ENV, raising=False)
    monkeypatch.delenv("OUROBOROS_SUBAGENT_HARNESS", raising=False)
    llm = FakeLLM()
    stamps = []
    usage_ctx = SimpleNamespace(_review_paid_stamp=lambda: stamps.append("paid"))
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=llm, usage_ctx=usage_ctx)
    actor = result.actors[0]
    assert actor["status"] == "error"
    assert "no configured session route" in actor["error"]
    assert llm.calls == []  # never a silent fallback onto the api route
    assert stamps == []  # a typed pre-start refusal is a $0 unpaid wave


def test_positive_custody_session_failure_emits_one_unknown_cost_usage_row(tmp_path):
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment

    error = RuntimeError("session failed after start")
    error.delegated_run_started = True
    error.delegated_run_id = "run-paid-failure"
    executor = AgentSessionReviewExecutor(ReviewAssignment(
        request=_agent_request(), slot=_agent_slot(), call_id="c-paid-failure",
        call_type="scope_review", custody_root=tmp_path,
    ), llm=FakeLLM())
    executor._run_session = lambda: (_ for _ in ()).throw(error)
    rows = []
    executor.usage_observer = rows.append

    with pytest.raises(RuntimeError, match="session failed after start"):
        executor.execute()
    with pytest.raises(RuntimeError, match="session failed after start"):
        executor.execute()

    assert len(rows) == 1
    assert rows[0]["provider"] == "claudexor"
    assert rows[0]["resolved_model"] == "api/model-a"
    assert rows[0]["delegated_run_started"] is True
    assert rows[0]["delegated_run_id"] == "run-paid-failure"
    assert rows[0]["cost"] is None


def test_pinned_profile_passes_row_status_through_to_the_engine(tmp_path, fake_route):
    """Phase D1 (owner batch-2 1A/2): a slot carrying a manual credential pin must
    not be refused on the harness-row catalog status — a row with no default
    credential store reads "unavailable" FOREVER by design (agy, engine INV-135)
    while its named profiles work. The request REACHES the engine with the pinned
    credentialProfileId on the wire, and the ENGINE's typed refusal propagates
    typed on this slot: never a silent degrade, never a fallback onto the api route."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    fake_route.catalog_entry["status"] = "unavailable"
    fake_route.catalog_entry["enabled"] = False
    fake_route.start_error = ClaudexorUnavailable(
        "engine_refuses_profile", "engine typed refusal for this profile", status_code=422)
    llm = FakeLLM()
    result = run_review_request(
        _agent_request(),
        slots=[_agent_slot(session_target="fake-review=fake-small",
                           session_profile="acct-pinned")],
        drive_root=tmp_path, llm=llm)
    starts = [r for inst in fake_route.instances for r in inst.start_requests]
    assert len(starts) == 1  # the start attempt was actually posted
    assert starts[0]["credentialProfileId"] == "acct-pinned"
    actor = result.actors[0]
    assert actor["status"] == "error"
    assert "engine typed refusal for this profile" in actor["error"]
    assert llm.calls == []  # never a silent fallback onto the api route


def test_absent_catalog_row_refuses_typed_even_with_a_pinned_profile(tmp_path, fake_route):
    """Phase D1 keeps `route_not_in_capability_catalog`: a pin skips only the row
    STATUS refusal — a route the catalog does not carry at all has no engine row
    to be authoritative about, so it still refuses typed before any POST."""
    import dataclasses

    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.subagents import parse_subagent_harness

    fake_route.catalog_entry = {"id": "some-other-route", "enabled": True, "status": "ok",
                                "accessProfilesSupported": ["readonly"]}
    route = dataclasses.replace(parse_subagent_harness("fake-review=fake-small"),
                                profile_id="acct-pinned")
    with pytest.raises(ReviewRouteUnavailable) as excinfo:
        _run_session_directly(tmp_path, session_route=route)
    assert excinfo.value.code == "route_not_in_capability_catalog"
    assert not any(inst.start_requests for inst in fake_route.instances)


def test_mixed_panel_failed_agent_slot_does_not_shrink_n(tmp_path, fake_route, monkeypatch):
    """5.3: one panel, two deliveries. The agent slot failing typed leaves the
    panel at its configured size — the failed slot is an error actor on its own
    row, never a smaller N."""
    monkeypatch.delenv(REVIEW_SESSION_ROUTE_ENV, raising=False)
    monkeypatch.delenv("OUROBOROS_SUBAGENT_HARNESS", raising=False)

    class ApiFindingsLLM(FakeLLM):
        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return ({"content": '{"verdict": "PASS", "findings": [], "summary": "ok"}'},
                    {"prompt_tokens": 3, "completion_tokens": 2})

    llm = ApiFindingsLLM()
    slots = [
        ReviewSlot(slot_id="scope_slot_1", model="api/model-a", timeout_sec=10),
        _agent_slot(slot_id="scope_slot_2"),
    ]
    result = run_review_request(_agent_request(), slots=slots,
                                drive_root=tmp_path, llm=llm)
    assert len(result.actors) == 2
    by_id = {a["slot_id"]: a for a in result.actors}
    assert by_id["scope_slot_1"]["status"] == "ok"
    assert by_id["scope_slot_2"]["status"] == "error"
    assert len(llm.calls) == 1  # the api row; the agent row never touched chat


def test_retired_route_envs_are_ignored(monkeypatch):
    """ABI-10: the phase-5 per-row route envs are RETIRED and IGNORED.

    A row built from a plain model list is pinned api_chat even when a stale
    environment still exports the retired spellings; delegated delivery is a
    structured-SSOT fact (``OUROBOROS_REVIEWER_SLOTS`` rows) only.
    """
    monkeypatch.setenv("OUROBOROS_REVIEW_ROUTES", "agent_session,agent_session")
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_ROUTES", "agent_session")
    rows = scope_reviewer_slots(["m1", "m2"])
    assert all(row.route is ReviewRouteKind.API_CHAT for row in rows)
    assert rows[0].slot_id == "scope_slot_1" and rows[1].slot_id == "scope_slot_2"
    assert all(row.route is ReviewRouteKind.API_CHAT
               for row in reviewer_slots(["m1", "m2"], role_hint="commit review"))


def test_scope_rows_default_to_the_configured_scope_review_effort(monkeypatch):
    """Regression (v6.89.0): with no structured reviewer slots, the legacy path took
    this function's old literal default ("medium") instead of the owner's configured
    OUROBOROS_EFFORT_SCOPE_REVIEW — the BLOCKING constitutional scope reviewer
    silently ran below its configured reasoning strength on every stock install."""
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_MODELS", "some/model")
    monkeypatch.delenv("OUROBOROS_EFFORT_SCOPE_REVIEW", raising=False)
    assert [row.effort for row in scope_reviewer_slots()] == ["high"]  # config default
    monkeypatch.setenv("OUROBOROS_EFFORT_SCOPE_REVIEW", "xhigh")
    assert [row.effort for row in scope_reviewer_slots()] == ["xhigh"]
    # An explicit effort still wins for callers that rebuild one positional row.
    assert [row.effort for row in scope_reviewer_slots(["m"], effort="low")] == ["low"]


def _persisted_response_payloads(drive_root):
    """Every persisted review response payload under ``drive_root``."""
    import gzip

    payloads = []
    for path in sorted((drive_root / "observability" / "blobs").glob("*.json.gz")):
        try:
            payload = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
        except Exception:  # pragma: no cover - non-json blob
            continue
        if isinstance(payload, dict) and "message" in payload:
            payloads.append(payload)
    return payloads


@pytest.mark.parametrize(
    "transcript, conformance",
    [
        # Schema-conformant: canonicalization legitimately reduces this to "[]".
        ('{"findings": []}', "passed"),
        # Narrative: light extraction replaces the whole thing.
        ("I reviewed the diff at length and found nothing that blocks.\nNO_FINDINGS", ""),
    ],
)
def test_delegated_transcript_survives_canonicalization_durably(
    tmp_path, fake_route, transcript, conformance
):
    """P1: the session's own output is the cognitive artifact, and canonicalization
    destroys it — `{"findings": []}` becomes `[]`, and extraction replaces a narrative
    wholesale. Keeping it only in the executor made the decision unreconstructible
    once the process ended; it must reach the DURABLE record with its provenance."""
    fake_route.detail = _terminal_detail(transcript, conformance=conformance)
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                               drive_root=tmp_path, llm=FakeLLM())
    actor = result.actors[0]

    # The canonical form is still what the gate parses...
    assert actor["raw_text"] == "[]"
    # ...and the RAW transcript is recoverable from the durable response payload.
    carriers = [p for p in _persisted_response_payloads(tmp_path)
                if str((p.get("message") or {}).get("session_transcript") or "")]
    assert carriers, "no persisted payload carried the session transcript"
    assert any((p["message"]["session_transcript"] == transcript) for p in carriers), carriers

    prov = actor["usage"]["verdict_provenance"]
    assert prov["raw_transcript_chars"] == len(transcript)
    assert prov["canonical_chars"] == len("[]")
    assert prov["raw_transcript_sha256"] != prov["canonical_sha256"]
    assert prov["verdict_method"] == actor["usage"]["verdict_method"]
    assert prov["conformance_trusted"] is (conformance == "passed")

def test_acceptance_rows_follow_the_configured_triad_delivery(monkeypatch):
    """Owner R2 (2026-09-01): task acceptance reads the SAME triad rows every other
    triad surface reads — a delegated row included — instead of an api-pinned
    projection of them. Upstream wrote this against the legacy comma-list plus its
    per-row route env; ABI-10 retired BOTH reads, so the configured rows come from
    the structured SSOT, which is the only configuration surface that can carry a
    session row at all. The generic model-list builder keeps its explicit pin for
    callers that pass no route list (a caller's own statement, never a surface
    default), and a stale retired route env still leaks into nothing."""
    from ouroboros.reviewer_slot_config import REVIEWER_SLOTS_ENV, triad_delivery_slots

    monkeypatch.setenv("OUROBOROS_REVIEW_ROUTES", "agent_session,agent_session")
    monkeypatch.setenv(REVIEWER_SLOTS_ENV, json.dumps({
        "triad": [
            {"slot_id": "slot_1", "route": {"kind": "agent_session", "target_id": "codex"}},
            {"slot_id": "slot_2", "route": {"kind": "api_chat", "target_id": "m2"}},
        ],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "m2"}}],
        "advisory": {"enabled": False,
                     "route": {"kind": "agent_session", "target_id": "codex"},
                     "effort": "low"},
    }))
    rows = triad_delivery_slots(role_hint="task acceptance")
    assert [row.route for row in rows] == [ReviewRouteKind.AGENT_SESSION, ReviewRouteKind.API_CHAT]
    assert [row.slot_id for row in rows] == ["slot_1", "slot_2"]
    assert all(row.role_hint == "task acceptance" for row in rows)
    pinned = reviewer_slots(["m1", "m2"], effort="high", role_hint="task acceptance")
    assert all(row.route is ReviewRouteKind.API_CHAT for row in pinned)


def test_agent_slot_without_session_task_refuses_the_api_pack(tmp_path, fake_route):
    """5.2: the giant assembled pack is not sendable to a session. A surface
    that supplied no route-owned session task gets a typed refusal, not a
    silently forwarded api pack."""
    request = _agent_request(session_task="",
                             messages=[{"role": "system", "content": "GIANT PACK"}])
    llm = FakeLLM()
    result = run_review_request(request, slots=[_agent_slot()],
                                drive_root=tmp_path, llm=llm)
    actor = result.actors[0]
    assert actor["status"] == "not_dispatched"
    assert "no session task" in actor["error"]
    assert llm.calls == []
    assert not any(inst.start_requests for inst in fake_route.instances)


def test_pre_dispatch_admission_requires_current_window_evidence(tmp_path, fake_route, monkeypatch):
    """Admission health (route_health, before any POST) knew the window was spent but
    said so in PROSE — the reset instant and the code did not survive to the actor
    record. B1: the EXISTING exhausted class is raised there, carrying reset_at, and
    no session is ever started."""
    from ouroboros.gateways.claudexor import ClaudexorSubscriptionWindowExhausted
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment

    spent = {"subject": {"harness": "fake-review", "subject_id": "acct"},
             "freshness": "fresh",
             "constraints": [{"used_ratio": 1.0, "resets_at": "2030-02-02T00:00:00Z"}]}
    monkeypatch.setattr(FakeGateway, "quota_snapshots", lambda self: [dict(spent)])
    executor = AgentSessionReviewExecutor(
        ReviewAssignment(request=_agent_request(), slot=_agent_slot(),
                         call_id="c-admission", call_type="scope_review",
                         custody_root=tmp_path),
        llm=FakeLLM(),
    )
    with pytest.raises(ClaudexorSubscriptionWindowExhausted) as excinfo:
        executor.execute()
    assert excinfo.value.reset_at == "2030-02-02T00:00:00Z"
    assert sum(len(inst.start_requests) for inst in fake_route.instances) == 0

    # An incomplete ratio is not an admission verdict. A fresh executor asks
    # the selected subscription engine, without another model or API fallback.
    spent["constraints"] = [{"used_ratio": 1.0}]
    custody._CUSTODY.clear()
    executor = AgentSessionReviewExecutor(
        ReviewAssignment(request=_agent_request(), slot=_agent_slot(),
                         call_id="c-admission-undated", call_type="scope_review",
                         custody_root=tmp_path / "b"),
        llm=FakeLLM(),
    )
    assert executor.execute().raw_text == "[]"
    starts = [request for instance in fake_route.instances for request in instance.start_requests]
    assert len(starts) == 1
    assert starts[0]["authPreference"] == "subscription"
    assert starts[0]["model"] == "fake-small"


def test_pre_dispatch_admission_preserves_the_pool_code(tmp_path, fake_route, monkeypatch):
    """Review fix 2 (cross-PR contract): an UNDATED `credential_pool_exhausted`
    reason from route_health raises the SAME exhausted class with the POOL code
    preserved — never flattened to the subscription code; the dated reason-empty
    shape keeps the subscription default and its reset exactly as before."""
    from ouroboros import subagents
    from ouroboros.gateways.claudexor import ClaudexorSubscriptionWindowExhausted
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment

    monkeypatch.setattr(
        subagents, "route_health",
        lambda gateway, route_id, shape, *, route_model="", pinned_profile="": ("credential_pool_exhausted", ""))
    executor = AgentSessionReviewExecutor(
        ReviewAssignment(request=_agent_request(), slot=_agent_slot(),
                         call_id="c-pool", call_type="scope_review",
                         custody_root=tmp_path),
        llm=FakeLLM(),
    )
    with pytest.raises(ClaudexorSubscriptionWindowExhausted) as pool:
        executor.execute()
    assert pool.value.code == "credential_pool_exhausted"
    assert pool.value.reset_at == ""

    # Dated, reason-empty shape (the ordinary spent window): unchanged path.
    monkeypatch.setattr(
        subagents, "route_health",
        lambda gateway, route_id, shape, *, route_model="", pinned_profile="": ("", "2030-03-03T00:00:00Z"))
    custody._CUSTODY.clear()
    executor = AgentSessionReviewExecutor(
        ReviewAssignment(request=_agent_request(), slot=_agent_slot(),
                         call_id="c-dated", call_type="scope_review",
                         custody_root=tmp_path / "b"),
        llm=FakeLLM(),
    )
    with pytest.raises(ClaudexorSubscriptionWindowExhausted) as dated:
        executor.execute()
    assert dated.value.code == "subscription_window_exhausted"
    assert dated.value.reset_at == "2030-03-03T00:00:00Z"
    assert sum(len(inst.start_requests) for inst in fake_route.instances) == 0


def test_an_expired_cooldown_is_history_not_exhaustion():
    """The `_exhausted_window` reader (the admission seam above) treated ANY non-empty
    `cooldown_until` as spent. A cooldown whose instant already PASSED is a stale fact
    the harness has not refreshed, not positive evidence of a spent window; a FUTURE
    one still blocks, and an illegible instant keeps the conservative old reading."""
    from ouroboros.subagents import _exhausted_window

    def _quota(cooldown):
        class _Q:
            def quota_snapshots(self):
                return [{"subject": {"harness": "some-route", "subject_id": "a"},
                         "freshness": "fresh",
                         "constraints": [{"used_ratio": 0.4,
                                          "cooldown_until": cooldown}]}]

            def quota_absences(self):
                return []
        return _Q()

    assert _exhausted_window(_quota("2020-01-01T00:00:00Z"), "some-route") == (False, "")
    assert _exhausted_window(_quota("2099-01-01T00:00:00Z"), "some-route") == (
        True, "2099-01-01T00:00:00Z")
    assert _exhausted_window(_quota("soon-ish"), "some-route") == (True, "soon-ish")


class AccountedFakeLLM(FakeLLM):
    """API-review fake that crosses the real durable physical-attempt seam."""

    def __init__(self, drive_root, reply="[]"):
        super().__init__(reply=reply)
        self.drive_root = drive_root

    def chat(self, **kwargs):
        from ouroboros import usage_accounting as ua

        request = ua.AttemptRequest(
            model="local-review-test", provider="local", reservation_usd=0.0,
            drive_root=self.drive_root, task_id="review", root_task_id="review",
        )
        return ua.execute_physical_attempt(request, lambda: FakeLLM.chat(self, **kwargs))

def test_unset_session_window_uses_task_absolute_ceiling(tmp_path, fake_route, monkeypatch):
    monkeypatch.setattr("ouroboros.config.get_task_abs_ceiling_sec", lambda: 21_600)

    run_review_request(
        _agent_request(), slots=[_agent_slot(timeout_sec=None)],
        drive_root=tmp_path, llm=FakeLLM(),
    )

    assert fake_route.instances[0].start_requests[0]["maxSeconds"] == 21_600

def test_task_metadata_deadline_narrows_session_engine_horizon(
    tmp_path, fake_route, monkeypatch,
):
    from datetime import datetime, timedelta, timezone

    monkeypatch.setattr("ouroboros.config.get_finalization_grace_sec", lambda: 0)
    ctx = SimpleNamespace(
        task_id="t-agent", event_queue=None, pending_events=[],
        task_metadata={
            "deadline_at": (datetime.now(timezone.utc) + timedelta(seconds=300)).isoformat(),
        },
    )
    run_review_request(
        _agent_request(), slots=[_agent_slot(timeout_sec=None)],
        drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx,
    )

    # The horizon is the ceiling of the seconds left; a coarse clock (Windows CI)
    # can leave a sub-second remainder after the 300 s the test just added, so
    # the ceiling may read 301 — one second, never a wider horizon.
    assert 1 <= fake_route.instances[0].start_requests[0]["maxSeconds"] <= 301

def test_light_extraction_transport_is_narrowed_by_the_request_deadline(monkeypatch):
    from datetime import datetime, timedelta, timezone

    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "1")
    llm = FakeLLM(reply="[]")
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
    text, method, _usage = canonicalize_session_verdict(
        "narrative: clean",
        conformance_passed=False,
        llm=llm,
        deadline_at=deadline,
    )
    assert method == "light_model_extraction" and text == "[]"
    assert 0 < llm.calls[0]["timeout"] <= 4.1

def test_spent_owner_deadline_skips_light_model_extraction():
    from ouroboros.review_execution import _extract_verdict_via_light_model

    class NeverCalled:
        def chat(self, **_kwargs):
            raise AssertionError("spent deadline must not dispatch extraction")

    canonical, usage = _extract_verdict_via_light_model(
        "narrative", llm=NeverCalled(), deadline_at="2000-01-01T00:00:00Z",
    )

    assert canonical is None
    assert usage["reason_code"] == "deadline_exhausted"
    assert usage["dispatch"] == "not_dispatched"

def test_reserve_only_owner_window_skips_light_model_extraction(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from ouroboros.review_execution import _extract_verdict_via_light_model

    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")

    class NeverCalled:
        def chat(self, **_kwargs):
            raise AssertionError("reserve-only owner window must not dispatch extraction")

    deadline = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
    canonical, usage = _extract_verdict_via_light_model(
        "narrative", llm=NeverCalled(), deadline_at=deadline,
    )

    assert canonical is None
    assert usage["reason_code"] == "deadline_exhausted"
    assert usage["dispatch"] == "not_dispatched"

def test_multi_model_wrapper_preserves_typed_refusal_into_skill_actor_record():
    from ouroboros.tools.review import _parse_model_response
    from ouroboros.triad_review import parse_model_review_results

    envelope = _parse_model_response("claude-fable-5", {
        "error": "Error: route unavailable",
        "slot_id": "skill-slot-1",
        "failure_code": "agent_session_route_unavailable",
        "reset_at": "2030-01-01T00:00:00Z",
        "http_status": 429,
        "transport_status": "unavailable",
    }, None)
    parsed = parse_model_review_results(
        {"results": [envelope]}, required_items=("manifest_schema",),
    )

    assert envelope["failure_code"] == "agent_session_route_unavailable"
    actor = parsed.actor_records[0].to_dict()
    assert actor["slot_id"] == "skill-slot-1"
    assert actor["failure_code"] == "agent_session_route_unavailable"
    assert actor["reset_at"] == "2030-01-01T00:00:00Z"
    assert actor["http_status"] == 429
    assert actor["transport_status"] == "unavailable"

def test_terminal_read_transport_failure_reuses_started_run(tmp_path, fake_route):
    """A run accepted by Claudexor must not be posted a second time when its
    first terminal read is temporarily unavailable."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    fake_route.poll_error = ClaudexorUnavailable("daemon_unreachable", "boom", status_code=0)
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=FakeLLM())
    assert result.actors[0]["status"] == "ok"
    assert sum(len(inst.start_requests) for inst in fake_route.instances) == 1
    assert sum(len(inst.run_gets) for inst in fake_route.instances) == 2

def test_stale_retry_token_cannot_be_reinterpreted_as_fresh_review(
    tmp_path, fake_route,
):
    """A missing durable invocation is custody loss, even with time left."""
    from ouroboros.review_execution import ReviewRouteUnavailable

    state = {"pending_invocation_id": "stale-token"}
    with pytest.raises(ReviewRouteUnavailable) as raised:
        _run_session_directly(
            tmp_path, retry_state=state, operation_id="op-stale",
        )

    assert raised.value.code == "review_custody_lost"
    gateway = fake_route.instances[-1] if fake_route.instances else None
    if gateway is not None:
        assert gateway.start_requests == []
        assert gateway.registrations == []

def test_failed_pending_invocation_checkpoint_refuses_provider_post(
    tmp_path, fake_route,
):
    from ouroboros.review_execution import ReviewRouteUnavailable

    state = {}

    def _fail(_invocation_id):
        raise OSError("ledger unavailable")

    with pytest.raises(ReviewRouteUnavailable) as raised:
        _run_session_directly(
            tmp_path,
            retry_state=state,
            pending_invocation_checkpoint=_fail,
        )

    assert raised.value.code == "review_custody_checkpoint_unwritable"
    gateway = fake_route.instances[-1]
    assert gateway.start_requests == []
    assert gateway.removals == []
    assert state == {}

def _record_pending_review_invocation(
    drive_root, *, invocation_id: str, task_id: str = "t-b", request=None,
):
    route_id = "fake-review"
    body = request or {
        "prompt": "review this",
        "instructions": "stored review instructions",
        "authPreference": "subscription",
        "mode": "ask",
        "access": "readonly",
        "scope": {"kind": "project", "root": "/tmp/fake-repo"},
        "harnesses": [route_id],
        "primaryHarness": route_id,
        "maxSeconds": 30,
        "model": "fake-small",
        "effort": "low",
    }
    assert custody.record_start_requested(
        drive_root, run_id="", task_id=task_id,
        idempotency_key="stored-logical-key", invocation_id=invocation_id,
        max_seconds=30, request=body, project_id="proj-owned",
        project_owned=True, route=route_id, surface="scope_review",
        slot_id="scope_slot_1",
    )

def test_pending_recovery_refuses_foreign_owner_before_gateway(tmp_path, fake_route):
    from ouroboros.review_execution import ReviewRouteUnavailable

    invocation_id = "inv-foreign-pending"
    _record_pending_review_invocation(
        tmp_path, invocation_id=invocation_id, task_id="another-task",
    )
    state = {"pending_invocation_id": invocation_id}
    before = len(fake_route.instances)

    with pytest.raises(ReviewRouteUnavailable) as excinfo:
        _run_session_directly(tmp_path, task_id="t-b", retry_state=state)

    assert excinfo.value.code == "review_recovery_ownership_unverified"
    assert len(fake_route.instances) == before
    assert state == {"pending_invocation_id": invocation_id}

@pytest.mark.parametrize("invalid_field", ["route_pool", "access"])
def test_pending_recovery_refuses_malformed_stored_request_before_gateway(
    tmp_path, fake_route, invalid_field,
):
    from ouroboros.review_execution import ReviewRouteUnavailable

    invocation_id = f"inv-malformed-{invalid_field}"
    route_id = "fake-review"
    request = {
        "prompt": "review this",
        "instructions": "stored review instructions",
        "authPreference": "subscription",
        "mode": "ask",
        "access": "readonly",
        "scope": {"kind": "project", "root": "/tmp/fake-repo"},
        "harnesses": [route_id],
        "primaryHarness": route_id,
        "maxSeconds": 30,
    }
    if invalid_field == "route_pool":
        request["harnesses"] = ["different-route"]
    else:
        request["access"] = "workspace_write"
    _record_pending_review_invocation(
        tmp_path, invocation_id=invocation_id, request=request,
    )
    state = {"pending_invocation_id": invocation_id}
    before = len(fake_route.instances)

    with pytest.raises(ReviewRouteUnavailable) as excinfo:
        _run_session_directly(tmp_path, retry_state=state)

    assert excinfo.value.code == "review_recovery_request_invalid"
    assert len(fake_route.instances) == before
    assert state == {"pending_invocation_id": invocation_id}

def test_real_api_and_session_dispatch_each_fire_the_captured_stamp_once(
    tmp_path, fake_route,
):
    session_stamps = []
    session_ctx = SimpleNamespace(_review_paid_stamp=lambda: session_stamps.append("session"))
    run_review_request(_agent_request(), slots=[_agent_slot()], drive_root=tmp_path,
                       llm=FakeLLM(), usage_ctx=session_ctx)
    assert session_stamps == ["session"]

    api_stamps = []
    api_ctx = SimpleNamespace(_review_paid_stamp=lambda: api_stamps.append("api"))
    llm = AccountedFakeLLM(tmp_path / "api")
    run_review_request(_agent_request(), slots=[_agent_slot(route=ReviewRouteKind.API_CHAT)],
                       drive_root=tmp_path / "api", llm=llm, usage_ctx=api_ctx)
    assert api_stamps == ["api"] and len(llm.calls) == 1

def test_mixed_panel_shares_one_idempotent_wave_stamp(tmp_path, fake_route):
    from ouroboros.review_dispatch import ReviewPaidStamp

    writes = []
    ctx = SimpleNamespace(_review_paid_stamp=ReviewPaidStamp(lambda: writes.append("paid")))
    llm = AccountedFakeLLM(tmp_path)
    run_review_request(
        _agent_request(),
        slots=[
            _agent_slot(slot_id="session-slot"),
            _agent_slot(slot_id="api-slot", route=ReviewRouteKind.API_CHAT),
        ],
        drive_root=tmp_path, llm=llm, usage_ctx=ctx,
    )
    assert writes == ["paid"]
    assert len(llm.calls) == 1 and len(fake_route.instances[0].start_requests) == 1

def test_missing_api_transport_refuses_before_paid_stamp(tmp_path, fake_route):
    stamps = []
    ctx = SimpleNamespace(_review_paid_stamp=lambda: stamps.append("paid"))
    result = run_review_request(
        _agent_request(), slots=[_agent_slot(route=ReviewRouteKind.API_CHAT)],
        drive_root=tmp_path, llm=SimpleNamespace(), usage_ctx=ctx,
    )
    assert result.actors[0]["status"] == "error"
    assert "api_chat client exposes no callable transport" in result.actors[0]["error"]
    assert stamps == []

def test_callable_api_refusal_before_physical_attempt_stays_unpaid(tmp_path, fake_route):
    stamps = []

    class RefusingLLM:
        def chat(self, **_kwargs):
            raise RuntimeError("route resolution failed before provider send")

    ctx = SimpleNamespace(_review_paid_stamp=lambda: stamps.append("paid"))
    result = run_review_request(
        _agent_request(), slots=[_agent_slot(route=ReviewRouteKind.API_CHAT)],
        drive_root=tmp_path, llm=RefusingLLM(), usage_ctx=ctx,
    )
    assert result.actors[0]["status"] == "error"
    assert "route resolution failed" in result.actors[0]["error"]
    assert stamps == []

def test_api_review_rechecks_owner_deadline_at_physical_boundary(tmp_path, fake_route, monkeypatch):
    from ouroboros.review_execution import (
        ApiChatReviewExecutor, ReviewAssignment, ReviewRouteUnavailable,
    )

    llm = FakeLLM()
    executor = ApiChatReviewExecutor(
        ReviewAssignment(
            request=_agent_request(deadline_at="2000-01-01T00:00:00Z"),
            slot=_agent_slot(route=ReviewRouteKind.API_CHAT),
            custody_root=tmp_path,
        ),
        llm=llm,
    )
    # Stand in for a long prompt/render/persistence phase. The final check is
    # deliberately after `_kwargs()` and immediately before `chat`.
    monkeypatch.setattr(executor, "_kwargs", lambda: {"messages": [], "model": "fake"})

    with pytest.raises(ReviewRouteUnavailable) as caught:
        executor.execute()
    assert caught.value.code == "deadline_exhausted"
    assert llm.calls == []

def test_api_review_does_not_stamp_pre_dispatch_released_capture(
    tmp_path, fake_route,
):
    from ouroboros.usage_accounting import PhysicalAttemptCapture
    from ouroboros.review_execution import (
        ApiChatReviewExecutor, ReviewAssignment,
    )

    stamps = []

    class ReleasedBeforeSend:
        def chat(self, **_kwargs):
            error = RuntimeError("request preparation failed")
            error.physical_attempt_capture = PhysicalAttemptCapture(
                attempt_id="attempt-released", model="fake", provider="test",
                state="released", candidate_measurement_kind="opaque",
            )
            raise error

    executor = ApiChatReviewExecutor(
        ReviewAssignment(
            request=_agent_request(),
            slot=_agent_slot(route=ReviewRouteKind.API_CHAT),
            custody_root=tmp_path,
            dispatch_stamp=lambda: stamps.append("paid"),
        ),
        llm=ReleasedBeforeSend(),
    )

    with pytest.raises(RuntimeError, match="request preparation failed"):
        executor.execute()
    assert stamps == []

def test_api_review_does_not_dispatch_inside_finalization_reserve(
    tmp_path, fake_route, monkeypatch,
):
    from datetime import datetime, timedelta, timezone
    from ouroboros.review_execution import (
        ApiChatReviewExecutor, ReviewAssignment, ReviewRouteUnavailable,
    )

    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
    llm = FakeLLM()
    executor = ApiChatReviewExecutor(
        ReviewAssignment(
            request=_agent_request(deadline_at=deadline),
            slot=_agent_slot(route=ReviewRouteKind.API_CHAT),
            custody_root=tmp_path,
        ),
        llm=llm,
    )
    monkeypatch.setattr(executor, "_kwargs", lambda: {"messages": [], "model": "fake"})

    with pytest.raises(ReviewRouteUnavailable) as caught:
        executor.execute()
    assert caught.value.code == "deadline_exhausted"
    assert llm.calls == []

def test_async_only_api_transport_fires_at_the_same_physical_boundary(tmp_path, fake_route):
    from ouroboros import usage_accounting as ua

    stamps = []

    class AsyncLLM:
        def __init__(self):
            self.calls = []

        async def chat_async(self, **kwargs):
            self.calls.append(kwargs)
            request = ua.AttemptRequest(
                model="local-review-test", provider="local", reservation_usd=0.0,
                drive_root=tmp_path, task_id="review", root_task_id="review",
            )

            async def send():
                return {"content": "[]"}, {"prompt_tokens": 0, "completion_tokens": 0}

            return await ua.execute_physical_attempt_async(request, send)

    llm = AsyncLLM()
    ctx = SimpleNamespace(_review_paid_stamp=lambda: stamps.append("paid"))
    result = run_review_request(
        _agent_request(), slots=[_agent_slot(route=ReviewRouteKind.API_CHAT)],
        drive_root=tmp_path, llm=llm, usage_ctx=ctx,
    )
    assert result.actors[0]["status"] == "ok"
    assert stamps == ["paid"] and len(llm.calls) == 1

def test_session_stamp_precedes_durable_start_request(
    tmp_path, fake_route, monkeypatch,
):
    stamped = threading.Event()
    original = custody.record_start_requested

    def checked_record(*args, **kwargs):
        assert stamped.is_set(), "orphan recovery may POST as soon as this row exists"
        return original(*args, **kwargs)

    monkeypatch.setattr(custody, "record_start_requested", checked_record)
    ctx = SimpleNamespace(_review_paid_stamp=stamped.set)
    run_review_request(_agent_request(), slots=[_agent_slot()], drive_root=tmp_path,
                       llm=FakeLLM(), usage_ctx=ctx)
    assert stamped.is_set()

def test_session_strict_wallet_refusal_blocks_start_request(tmp_path, fake_route):
    from ouroboros.review_dispatch import ReviewPaidStamp

    def refuse():
        raise RuntimeError("acceptance wallet unavailable")

    ctx = SimpleNamespace(
        _review_paid_stamp=ReviewPaidStamp(refuse, fail_closed=True),
    )
    result = run_review_request(
        _agent_request(), slots=[_agent_slot()], drive_root=tmp_path,
        llm=FakeLLM(), usage_ctx=ctx,
    )

    assert result.actors[0]["status"] == "error"
    assert "acceptance wallet unavailable" in result.actors[0]["error"]
    assert fake_route.instances[0].start_requests == []

def test_late_worker_uses_its_captured_stamp_after_caller_restores_context(
    tmp_path, fake_route, monkeypatch,
):
    entered, release = threading.Event(), threading.Event()
    stamped, finished = threading.Event(), threading.Event()
    original = FakeGateway.find_project_id
    original_close = FakeGateway.close

    def delayed_find(self, root):
        entered.set()
        # 10s, not 2s: this gate opens only after the caller's run_review_request
        # returns, and on a loaded CI runner that return itself exceeded 2s — the
        # worker then died here, never stamped, and leaked a live _ACTIVE entry
        # into the next (equal-key) test.  Drain/gate bounds are coarse on purpose.
        assert release.wait(10)
        return original(self, root)

    def observed_close(self):
        original_close(self)
        finished.set()

    monkeypatch.setattr(FakeGateway, "find_project_id", delayed_find)
    monkeypatch.setattr(FakeGateway, "close", observed_close)
    ctx = SimpleNamespace(_review_paid_stamp=stamped.set)
    result = run_review_request(
        _agent_request(), slots=[_agent_slot(timeout_sec=0.02)],
        drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx,
    )
    assert entered.is_set() and result.actors[0]["status"] == "error"
    ctx._review_paid_stamp = None  # mirrors review_skill's finally restoration
    release.set()
    # 10s bounds (were 2s): these are drain waits on the released worker, not
    # discrimination margins — on a loaded CI runner the 2s bound tripped while
    # the worker was healthily mid-flight, and the still-live worker then
    # poisoned the next equal-key test (its run joined this _ACTIVE entry and
    # posted zero starts of its own).
    assert stamped.wait(10), "the late physical start must retain its original wave stamp"
    assert finished.wait(10), "the delayed worker must not leak into the next test"
    # Gateway close precedes the substrate's final actor publication by a few
    # instructions. Wait for process-local custody too, otherwise the following
    # equal-content test can legitimately join this late actor.
    import time

    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    key = _attempt_key(_agent_request(), _agent_slot(timeout_sec=0.02))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            if key not in _ACTIVE:
                break
        time.sleep(0.005)
    else:
        raise AssertionError("the delayed review worker did not settle its custody")

def test_degraded_row_status_reaches_the_engine_whose_refusal_stays_typed(tmp_path, fake_route):
    """cx-delegation sprint (owner 7=A, «статус обманывает»): the doctor's
    aggregate row status is no longer a pre-POST refusal — for review slots
    too, since route_health is deliberately ONE reader. The request reaches
    the engine; the ENGINE's typed refusal rides the slot, and there is still
    never a silent fallback onto the api route."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    fake_route.catalog_entry["status"] = "degraded"
    fake_route.start_error = ClaudexorUnavailable(
        "credential_pool_exhausted", "engine typed refusal: pool exhausted",
        status_code=422)
    llm = FakeLLM()
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=llm)
    starts = [r for inst in fake_route.instances for r in inst.start_requests]
    assert len(starts) == 1  # the start attempt was actually posted
    actor = result.actors[0]
    assert actor["status"] == "error"
    assert "engine typed refusal: pool exhausted" in actor["error"]
    assert llm.calls == []  # never a silent fallback onto the api route

def test_owner_disabled_route_refusal_carries_its_typed_code(tmp_path, fake_route):
    """The remaining pre-POST row refusal is the OWNER's settings toggle
    (`enabled=false` — routing excludes it regardless of doctor status), and it
    still rides ReviewRouteUnavailable with a machine-readable `.code` (the
    rotation sprint's quorum classification keys on failure codes; a bare
    RuntimeError is invisible to it). The aggregate doctor status alone no
    longer refuses (owner 7=A)."""
    from ouroboros.review_execution import ReviewRouteUnavailable

    fake_route.catalog_entry["status"] = "unavailable"
    fake_route.catalog_entry["enabled"] = False
    with pytest.raises(ReviewRouteUnavailable) as excinfo:
        _run_session_directly(tmp_path)
    assert excinfo.value.code == "route_disabled"
    assert not any(inst.start_requests for inst in fake_route.instances)
