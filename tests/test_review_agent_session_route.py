"""Phase 5 review lanes: the agent_session route's typed verdict and routed slots.

Split by theme out of the original giant of the same name. This module owns the
typed verdict contract (schema conformance as the gate, light extraction, strict
whole-answer parsing) and the routes on slots: typed refusals, quorum shape, the
configured route parsing and the durable canonicalized transcript.
"""

import json

import pytest

from ouroboros import delegate_custody as custody

from ouroboros.review_execution import (
    REVIEW_SESSION_ROUTE_ENV,
    SCOPE_REVIEW_ROUTES_ENV,
    TRIAD_REVIEW_ROUTES_ENV,
    ReviewRouteKind,
    canonicalize_session_verdict,
    configured_review_routes,
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
    FakeLLM,
    _agent_request,
    _agent_slot,
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
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=llm)
    actor = result.actors[0]
    assert actor["status"] == "error"
    assert "no configured session route" in actor["error"]
    assert llm.calls == []  # never a silent fallback onto the api route


def test_unhealthy_route_refuses_typed_never_falls_back(tmp_path, fake_route):
    fake_route.catalog_entry["status"] = "degraded"
    llm = FakeLLM()
    result = run_review_request(_agent_request(), slots=[_agent_slot()],
                                drive_root=tmp_path, llm=llm)
    actor = result.actors[0]
    assert actor["status"] == "error"
    assert "route_status_degraded" in actor["error"]
    assert llm.calls == []
    assert not any(inst.start_requests for inst in fake_route.instances)


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


def test_route_status_refusal_carries_its_typed_code(tmp_path, fake_route):
    """Phase D2: the route_health refusal rides ReviewRouteUnavailable with a
    machine-readable `.code` (the rotation sprint's quorum classification keys
    on failure codes; a bare RuntimeError is invisible to it)."""
    from ouroboros.review_execution import ReviewRouteUnavailable
    from tests._review_session_route_shared import _run_session_directly

    fake_route.catalog_entry["status"] = "unavailable"
    fake_route.catalog_entry["enabled"] = False
    with pytest.raises(ReviewRouteUnavailable) as excinfo:
        _run_session_directly(tmp_path)
    assert excinfo.value.code == "route_status_unavailable"
    assert not any(inst.start_requests for inst in fake_route.instances)


def test_absent_catalog_row_refuses_typed_even_with_a_pinned_profile(tmp_path, fake_route):
    """Phase D1 keeps `route_not_in_capability_catalog`: a pin skips only the row
    STATUS refusal — a route the catalog does not carry at all has no engine row
    to be authoritative about, so it still refuses typed before any POST."""
    import dataclasses

    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.subagents import parse_subagent_harness
    from tests._review_session_route_shared import _run_session_directly

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


def test_configured_review_routes_parsing(monkeypatch):
    monkeypatch.setenv(TRIAD_REVIEW_ROUTES_ENV, "api_chat, agent_session")
    routes = configured_review_routes(TRIAD_REVIEW_ROUTES_ENV, 3)
    assert routes == [ReviewRouteKind.API_CHAT, ReviewRouteKind.AGENT_SESSION,
                      ReviewRouteKind.API_CHAT]
    monkeypatch.setenv(TRIAD_REVIEW_ROUTES_ENV, "codex")
    with pytest.raises(ValueError):
        configured_review_routes(TRIAD_REVIEW_ROUTES_ENV, 1)


def test_scope_rows_carry_their_configured_routes(monkeypatch):
    monkeypatch.setenv(SCOPE_REVIEW_ROUTES_ENV, "agent_session")
    rows = scope_reviewer_slots(["m1", "m2"])
    assert rows[0].route is ReviewRouteKind.AGENT_SESSION
    assert rows[1].route is ReviewRouteKind.API_CHAT
    assert rows[0].slot_id == "scope_slot_1" and rows[1].slot_id == "scope_slot_2"


def test_scope_rows_default_to_the_configured_scope_review_effort(monkeypatch):
    """Regression (v6.89.0): with no structured reviewer slots, the legacy path took
    this function's old literal default ("medium") instead of the owner's configured
    OUROBOROS_EFFORT_SCOPE_REVIEW — the BLOCKING constitutional scope reviewer
    silently ran below its configured reasoning strength on every stock install."""
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    monkeypatch.delenv(SCOPE_REVIEW_ROUTES_ENV, raising=False)
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

def test_acceptance_rows_stay_api_even_when_triad_routes_delegate(monkeypatch):
    """D15: task acceptance is pinned to the API (plan review follows each configured
    row's delivery since the spec-gate redesign). The triad's route list must not
    leak into surfaces that pass no route_env_key."""
    monkeypatch.setenv(TRIAD_REVIEW_ROUTES_ENV, "agent_session,agent_session,agent_session")
    rows = reviewer_slots(["m1", "m2"], effort="high", role_hint="task acceptance")
    assert all(row.route is ReviewRouteKind.API_CHAT for row in rows)


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
    assert actor["status"] == "error"
    assert "no session task" in actor["error"]
    assert llm.calls == []
    assert not any(inst.start_requests for inst in fake_route.instances)


def test_pre_dispatch_admission_raises_the_typed_window_class(tmp_path, fake_route, monkeypatch):
    """Admission health (route_health, before any POST) knew the window was spent but
    said so in PROSE — the reset instant and the code did not survive to the actor
    record. B1: the EXISTING exhausted class is raised there, carrying reset_at, and
    no session is ever started."""
    from ouroboros.gateways.claudexor import ClaudexorSubscriptionWindowExhausted
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment
    # Local import: this module must NOT re-export the shared fake (the ledger
    # records the split fixture as facadeless).
    from tests._review_session_route_shared import FakeGateway

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

    # An undated exhaustion (route_health's reason-with-empty-reset shape) is
    # STILL the typed class — spent with an unknown healing instant. A fresh
    # executor: a settled typed failure is memoized per executor by design.
    spent["constraints"] = [{"used_ratio": 1.0}]
    custody._CUSTODY.clear()
    executor = AgentSessionReviewExecutor(
        ReviewAssignment(request=_agent_request(), slot=_agent_slot(),
                         call_id="c-admission-undated", call_type="scope_review",
                         custody_root=tmp_path / "b"),
        llm=FakeLLM(),
    )
    with pytest.raises(ClaudexorSubscriptionWindowExhausted) as undated:
        executor.execute()
    assert undated.value.reset_at == ""
    assert sum(len(inst.start_requests) for inst in fake_route.instances) == 0


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
