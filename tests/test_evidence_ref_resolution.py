"""D-Q5: host resolution of reviewer evidence_refs by EXACT membership against the
packet's enumerable exhibit keys. The resolution feeds ONLY the release-clean bit
(task_acceptance_is_clean) plus disclosure — parse validity, quorum participation,
and verdicts are untouched (the v6.71.1 starvation and v6.64.3 bare-veto classes
stay closed), and rows are absent on historical results (forward-only)."""

import json

from ouroboros.review_evidence import (
    acceptance_evidence_ref_vocabulary,
    annotate_criteria_evidence_resolution,
    resolve_criteria_evidence_refs,
)
from ouroboros.review_substrate import ReviewRunResult, task_acceptance_is_clean


_PACKET = {
    "task_contract": {
        "objective": "ship it",
        "acceptance_claims": [
            {"id": "claim_1", "claim": "game boots"},
            {"id": "claim_2", "claim": "score persists"},
        ],
    },
    # Host support table: claim_1 has a PASSING receipt behind it, claim_2 was
    # only declared. A reviewer may cite claim_1 as evidence; claim_2 names a
    # real claim but no host-attested support.
    "acceptance_support_refs": [
        {"criterion_id": "claim_1", "support_status": "supported",
         "support_refs": [{"ref": "verification_receipts[0]", "status": "pass"}]},
        {"criterion_id": "claim_2", "support_status": "declared_only",
         "support_refs": [{"ref": "verification_receipts[1]", "status": "declared"}]},
    ],
    "verification_summary": {"count": 2, "failed_count": 0},
    # The indexed exhibit rows the receipt-ref vocabulary enumerates (F4): the
    # SAME global indices acceptance_support_refs cites, status visible per row.
    "verification_receipts": [
        {"ref": "verification_receipts[0]", "status": "pass", "matched": True,
         "provenance": "host_attested", "criterion_id": "claim_1", "check": "pytest -q"},
        {"ref": "verification_receipts[1]", "status": "declared", "matched": None,
         "provenance": "host_attested", "criterion_id": "claim_2", "check": ""},
    ],
    "acceptance_obligations": [{"id": "ob-12ab34cd", "item": "cover edge case"}],
    "artifacts": [{"name": "report/summary.md", "size": 10}],
    "repo_diff": "diff --git ...",
    "tool_trajectory": [{"tool": "run_command", "status": "ok", "result": "3 passed"}],
    # The agent's OWN prose sections, exactly as build_task_acceptance_evidence
    # tags them. Present in essentially every real task.
    "reasoning_notes": "I believe the feature works.",
    "candidate_answers": ["the answer"],
    "agent_supplied": {"agent_decision": {"disposition": "accept"}},
    # A section the builder emits with no provenance tag (a counter).
    "tool_trajectory_omitted_leading": 4,
    # The real packet's provenance table — the ONE authority the vocabulary keys on.
    "__provenance__": {
        "task_contract": "host_attested",
        "acceptance_support_refs": "host_attested",
        "verification_summary": "host_attested",
        "verification_receipts": "host_attested",
        "acceptance_obligations": "host_attested",
        "repo_diff": "host_attested",
        "artifacts": "artifact",
        "tool_trajectory": "tool_result",
        "reasoning_notes": "agent_supplied",
        "candidate_answers": "agent_supplied",
        "agent_supplied": "agent_supplied",
    },
}


def _actor(criteria, unresolved_rows=None):
    row = {
        "slot_id": "s0", "signal": "PASS",
        "parsed": {
            "verdict": "PASS", "outcome_tier": "solved", "criteria_used": criteria,
        },
    }
    if unresolved_rows is not None:
        row["criteria_refs_unresolved"] = unresolved_rows
    return row


def _result(actors):
    return ReviewRunResult(
        request={"surface": "task_acceptance", "policy": {"min_successful_slots": 1}},
        actors=actors, parsed_findings=[], aggregate_signal="PASS",
    )


def test_vocabulary_enumerates_packet_exhibits_exactly():
    vocab = acceptance_evidence_ref_vocabulary(_PACKET)
    # A claim id is evidence only through the HOST support table.
    assert vocab["claim_1"] == "claim_id"
    assert vocab["claim_2"] == "claim_id_unsupported"
    assert vocab["ob-12ab34cd"] == "obligation_id"
    assert vocab["report/summary.md"] == "artifact"
    # Receipt refs come from the packet's OWN exhibit rows: the green row
    # resolves, the declared-only row is disclosed but never counts (F4), and an
    # index without a row does not exist — count alone mints nothing.
    assert vocab["verification_receipts[0]"] == "verification_receipt"
    assert vocab["verification_receipts[1]"] == "verification_receipt_not_passing"
    assert "verification_receipts[2]" not in vocab
    assert vocab["verification_receipts"] == "packet_section"  # the exhibit list itself
    assert vocab["repo_diff"] == "packet_section"
    assert vocab["verification_summary"] == "packet_section"
    # The packet discloses four omitted leading rows, so the carried tail is partial.
    assert vocab["tool_trajectory"] == "partial"
    assert "__provenance__" not in vocab
    # Robust to junk input.
    assert acceptance_evidence_ref_vocabulary(None) == {}
    assert acceptance_evidence_ref_vocabulary("junk") == {}


def test_agent_supplied_and_intent_sections_never_resolve():
    """The section rule reads the packet's OWN provenance table: only a
    host-attested exhibit resolves. The agent's own prose (reasoning_notes /
    candidate_answers / agent_supplied) and the declared-intent container
    (task_contract, which HOLDS the claims) are disclosed by name and never count
    — otherwise `evidence_refs: ["reasoning_notes"]` or `["task_contract"]` buys
    exactly the clean PASS a bare unsupported claim id cannot (D-Q5)."""
    vocab = acceptance_evidence_ref_vocabulary(_PACKET)
    assert vocab["reasoning_notes"] == "agent_supplied_section"
    assert vocab["candidate_answers"] == "agent_supplied_section"
    assert vocab["agent_supplied"] == "agent_supplied_section"
    assert vocab["task_contract"] == "declared_intent_section"
    # No provenance tag at all = unknown attestation = fails closed.
    assert vocab["tool_trajectory_omitted_leading"] == "unattested_section"

    for ref in ("reasoning_notes", "candidate_answers", "agent_supplied",
                "task_contract", "tool_trajectory_omitted_leading"):
        row = resolve_criteria_evidence_refs(
            [{"criterion": "c", "status": "supported", "evidence_refs": [ref]}], vocab,
        )[0]
        assert row["supported_evidence_resolves"] is False, ref
        # Still DISCLOSED by name — never a bare "" that hides which entry was cited.
        assert row["refs"][0]["resolved_as"].endswith("_section"), ref

    # ...and it lands on the clean bit only, never the verdict/quorum.
    actor = _actor([{"criterion": "c", "status": "supported",
                     "evidence_refs": ["reasoning_notes"]}])
    annotate_criteria_evidence_resolution([actor], _PACKET)
    result = _result([actor])
    assert task_acceptance_is_clean(result) is False
    assert result.aggregate_signal == "PASS" and actor["parsed"]["verdict"] == "PASS"

    # One real host-attested exhibit alongside the prose is still enough.
    assert resolve_criteria_evidence_refs(
        [{"criterion": "c", "status": "supported",
          "evidence_refs": ["reasoning_notes", "repo_diff"]}], vocab,
    )[0]["supported_evidence_resolves"] is True


def test_section_provenance_is_read_from_the_real_builder_packet(tmp_path):
    """The provenance tags the vocabulary keys on are the ones
    build_task_acceptance_evidence actually writes — pinned against the builder,
    not against a hand-made table."""
    from types import SimpleNamespace

    from ouroboros.review_evidence import build_task_acceptance_evidence

    ctx = SimpleNamespace(
        drive_root=str(tmp_path), task_id="t1", task_metadata={},
        task_contract={"objective": "ship", "acceptance_claims": [{"id": "claim_1", "claim": "boots"}]},
    )
    packet = build_task_acceptance_evidence(
        ctx,
        llm_trace={"reasoning_notes": ["my own notes"], "candidate_answers": ["a"]},
        drive_root=tmp_path,
        task_id="t1",
        canonical_subject="answer",
    )
    vocab = acceptance_evidence_ref_vocabulary(packet)
    assert packet["__provenance__"]["reasoning_notes"] == "agent_supplied"
    assert vocab["reasoning_notes"] == "agent_supplied_section"
    assert vocab["candidate_answers"] == "agent_supplied_section"
    assert vocab["task_contract"] == "declared_intent_section"
    assert vocab["repo_diff"] == "packet_section"
    assert vocab["claim_1"] == "claim_id_unsupported"   # no passing receipt behind it


def test_resolution_is_exact_match_only_no_fuzzy():
    vocab = acceptance_evidence_ref_vocabulary(_PACKET)
    rows = resolve_criteria_evidence_refs(
        [{"criterion": "game boots", "status": "supported",
          "evidence_refs": ["verification_receipt", "the SERVE_OK receipt", "repo_dif"]}],
        vocab,
    )
    # Substrings / near-misses of real keys never resolve (v6.78 lossy-identity lesson).
    assert len(rows) == 1
    assert rows[0]["supported_evidence_resolves"] is False
    assert all(ref["resolved_as"] == "" for ref in rows[0]["refs"])


def test_resolution_discloses_deciding_basis_and_counts_one_resolving_ref():
    vocab = acceptance_evidence_ref_vocabulary(_PACKET)
    rows = resolve_criteria_evidence_refs(
        [{"criterion": "game boots", "status": "supported",
          "evidence_refs": ["verification_receipts[0]", "made-up-ref"]}],
        vocab,
    )
    assert len(rows) == 1
    assert rows[0]["supported_evidence_resolves"] is True  # >=1 ref resolves -> counts
    bases = {ref["ref"]: ref["resolved_as"] for ref in rows[0]["refs"]}
    assert bases["verification_receipts[0]"] == "verification_receipt"
    assert bases["made-up-ref"] == ""
    # Fully-resolving criteria produce NO row (common clean path is unannotated),
    # and non-supported criteria are never resolution's business.
    assert resolve_criteria_evidence_refs(
        [{"criterion": "c", "status": "supported", "evidence_refs": ["claim_1"]},
         {"criterion": "d", "status": "missing", "evidence_refs": ["nonsense"]}],
        vocab,
    ) == []


def test_bare_claim_id_without_a_host_receipt_never_resolves():
    """The fabricated-evidence hole D-Q5 exists to close: a reviewer citing a
    claim the task itself declared, with no passing receipt behind it, must not
    buy a release-clean PASS. The disclosure NAMES the claim (it is a real packet
    entry) instead of pretending the ref was nonsense."""
    vocab = acceptance_evidence_ref_vocabulary(_PACKET)
    rows = resolve_criteria_evidence_refs(
        [{"criterion": "score persists", "status": "supported", "evidence_refs": ["claim_2"]}],
        vocab,
    )
    assert rows[0]["supported_evidence_resolves"] is False
    assert rows[0]["refs"][0]["resolved_as"] == "claim_id_unsupported"
    # ...and it demotes only the clean bit, on the existing rail.
    actor = _actor([{"criterion": "score persists", "status": "supported",
                     "evidence_refs": ["claim_2"]}])
    annotate_criteria_evidence_resolution([actor], _PACKET)
    result = _result([actor])
    assert task_acceptance_is_clean(result) is False
    assert result.aggregate_signal == "PASS" and actor["parsed"]["verdict"] == "PASS"
    # One REAL green exhibit alongside the unsupported claim is still enough.
    assert resolve_criteria_evidence_refs(
        [{"criterion": "score persists", "status": "supported",
          "evidence_refs": ["claim_2", "verification_receipts[0]"]}],
        vocab,
    )[0]["supported_evidence_resolves"] is True


def test_receipt_index_form_cannot_bypass_the_support_rule():
    """F4: `claim_2`'s only receipt is declared-only, so the claim id does not
    resolve — and citing the SAME receipt by its `verification_receipts[1]` index
    form must not resolve either, or the index is a bypass of the host support
    rule. The old vocabulary synthesized the index refs from
    `verification_summary.count` alone, so a red/declared receipt the reviewer
    never saw bought a release-clean PASS."""
    vocab = acceptance_evidence_ref_vocabulary(_PACKET)
    rows = resolve_criteria_evidence_refs(
        [{"criterion": "score persists", "status": "supported",
          "evidence_refs": ["verification_receipts[1]"]}], vocab,
    )
    assert rows[0]["supported_evidence_resolves"] is False
    # Disclosed by NAME (a real packet row), never a bare "".
    assert rows[0]["refs"][0]["resolved_as"] == "verification_receipt_not_passing"
    # ...and it demotes only the clean bit, on the existing rail.
    actor = _actor([{"criterion": "score persists", "status": "supported",
                     "evidence_refs": ["verification_receipts[1]"]}])
    annotate_criteria_evidence_resolution([actor], _PACKET)
    result = _result([actor])
    assert task_acceptance_is_clean(result) is False
    assert result.aggregate_signal == "PASS" and actor["parsed"]["verdict"] == "PASS"

    # A count alone mints NO receipt vocabulary: the rows must be in the packet.
    count_only = {
        "verification_summary": {"count": 3, "failed_count": 3},
        "__provenance__": {"verification_summary": "host_attested"},
    }
    assert not any(
        key.startswith("verification_receipts[")
        for key in acceptance_evidence_ref_vocabulary(count_only)
    )


def test_receipt_vocabulary_is_derived_from_the_builder_rows(tmp_path):
    """Builder → vocabulary pin: build_task_acceptance_evidence carries one
    host-attested `verification_receipts` row per receipt under the SAME global
    index acceptance_support_refs cites; a green row resolves, a red row is
    disclosed by name, and an out-of-packet index does not exist."""
    from types import SimpleNamespace

    from ouroboros.outcomes import append_verification_receipt
    from ouroboros.review_evidence import build_task_acceptance_evidence

    append_verification_receipt(tmp_path, "t1", {
        "status": "fail", "returncode": 1, "check": "pytest -q", "criterion_id": "c_red",
    })
    append_verification_receipt(tmp_path, "t1", {
        "status": "pass", "matched": True, "check": "pytest -q", "criterion_id": "c_green",
    })
    ctx = SimpleNamespace(drive_root=str(tmp_path), task_id="t1", task_metadata={}, task_contract={})
    packet = build_task_acceptance_evidence(
        ctx, drive_root=tmp_path, task_id="t1", canonical_subject="answer",
    )
    rows = packet["verification_receipts"]
    assert [row["status"] for row in rows] == ["fail", "pass"]
    assert [row["ref"] for row in rows] == ["verification_receipts[0]", "verification_receipts[1]"]
    assert packet["__provenance__"]["verification_receipts"] == "host_attested"
    vocab = acceptance_evidence_ref_vocabulary(packet)
    assert vocab["verification_receipts[0]"] == "verification_receipt_not_passing"
    assert vocab["verification_receipts[1]"] == "verification_receipt"
    assert "verification_receipts[2]" not in vocab


def test_claims_without_a_host_support_table_are_never_self_supporting():
    """No `acceptance_support_refs` in the packet means the host attested no
    support at all — claim ids fail CLOSED rather than resolving by default."""
    packet = {"task_contract": {"acceptance_claims": [{"id": "claim_1", "claim": "boots"}]}}
    vocab = acceptance_evidence_ref_vocabulary(packet)
    assert vocab["claim_1"] == "claim_id_unsupported"
    assert resolve_criteria_evidence_refs(
        [{"criterion": "c", "status": "supported", "evidence_refs": ["claim_1"]}], vocab,
    )[0]["supported_evidence_resolves"] is False


def test_annotation_marks_only_actors_with_unresolved_refs():
    actors = [
        _actor([{"criterion": "c", "status": "supported", "evidence_refs": ["claim_1"]}]),
        _actor([{"criterion": "c", "status": "supported", "evidence_refs": ["hallucinated"]}]),
    ]
    annotate_criteria_evidence_resolution(actors, _PACKET)
    assert "criteria_refs_unresolved" not in actors[0]
    rows = actors[1]["criteria_refs_unresolved"]
    assert rows[0]["supported_evidence_resolves"] is False


def test_clean_gate_demotes_only_the_clean_bit_never_the_verdict():
    criteria = [{"criterion": "c", "status": "supported", "evidence_refs": ["hallucinated"]}]
    annotated = _actor(criteria, unresolved_rows=[
        {"criterion": "c", "refs": [{"ref": "hallucinated", "resolved_as": ""}],
         "supported_evidence_resolves": False},
    ])
    result = _result([annotated])
    assert task_acceptance_is_clean(result) is False
    # The verdict, signal and aggregate are untouched: the demotion lands on the
    # EXISTING non-clean rails (finalized_unaccepted / bounded capsule), never
    # FAIL/veto/malformed — this cannot create the v6.71.1 unpassable-loop class.
    assert result.aggregate_signal == "PASS"
    assert annotated["signal"] == "PASS"
    assert annotated["parsed"]["verdict"] == "PASS"


def test_panel_reason_names_the_unresolved_ref_not_a_satisfied_condition():
    """`panel_reason` is the shared "one honest reason line naming the REAL
    blocker". On a D-Q5 demotion every criterion IS supported with refs, so the
    criteria-support line described a condition that was already satisfied while
    the deciding fact (an unresolved ref) had no reason text at all."""
    from ouroboros.review_substrate import panel_reason

    actor = _actor([{"criterion": "c", "status": "supported",
                     "evidence_refs": ["reasoning_notes", "hallucinated"]}])
    annotate_criteria_evidence_resolution([actor], _PACKET)
    reason = panel_reason(_result([actor]))

    assert "does not resolve" in reason
    assert "reasoning_notes (agent_supplied_section)" in reason
    assert "hallucinated (no packet entry)" in reason
    assert "until every criterion is supported" not in reason

    # The panel-wide fail-closed row names itself.
    unavailable = _actor([{"criterion": "c", "status": "supported", "evidence_refs": ["x"]}],
                         unresolved_rows=[{"criterion": "*", "refs": [],
                                           "supported_evidence_resolves": False,
                                           "resolution_status": "host_resolution_unavailable"}])
    assert "host_resolution_unavailable" in panel_reason(_result([unavailable]))

    # A non-D-Q5 demotion (unsupported criteria, no annotation) keeps the old line.
    plain = _actor([{"criterion": "c", "status": "missing", "evidence_refs": []}])
    assert "until every criterion is supported" in panel_reason(_result([plain]))
    # ...and a clean panel is untouched.
    clean = _actor([{"criterion": "c", "status": "supported", "evidence_refs": ["repo_diff"]}])
    annotate_criteria_evidence_resolution([clean], _PACKET)
    assert panel_reason(_result([clean])) == "clean acceptance"


def test_clean_gate_accepts_partial_resolution_and_historical_rows():
    criteria = [{"criterion": "c", "status": "supported",
                 "evidence_refs": ["verification_receipts[0]", "made-up"]}]
    # >=1 ref resolved -> the disclosure row exists but the criterion still counts.
    partially = _actor(criteria, unresolved_rows=[
        {"criterion": "c",
         "refs": [{"ref": "verification_receipts[0]", "resolved_as": "verification_receipt"},
                  {"ref": "made-up", "resolved_as": ""}],
         "supported_evidence_resolves": True},
    ])
    assert task_acceptance_is_clean(_result([partially])) is True
    # Historical rows carry no annotation (forward-only): the gate must not
    # invent a demotion for them.
    historical = _actor([{"criterion": "c", "status": "supported",
                          "evidence_refs": ["free-form old ref"]}])
    assert task_acceptance_is_clean(_result([historical])) is True


def test_resolution_failure_is_fail_closed_never_a_silent_clean_pass():
    """The ABSENCE of an annotation row is what authorizes the clean bit, so a
    resolver that did not run must not read as 'everything resolved'. Any failure
    stamps the panel-wide unavailable row on every actor — landing on the same
    clean-bit rail as an unresolved ref, never on the verdict."""
    import ouroboros.review_evidence as review_evidence

    actors = [_actor([{"criterion": "c", "status": "supported",
                       "evidence_refs": ["verification_receipts[0]"]}])]

    class _Exploding(dict):
        def get(self, *_a, **_kw):
            raise RuntimeError("packet read blew up")

    annotate_criteria_evidence_resolution(actors, _Exploding())
    rows = actors[0]["criteria_refs_unresolved"]
    assert rows[0]["supported_evidence_resolves"] is False
    assert rows[0]["resolution_status"] == "host_resolution_unavailable"
    result = _result(actors)
    assert task_acceptance_is_clean(result) is False
    assert result.aggregate_signal == "PASS"           # verdict untouched
    assert actors[0]["parsed"]["verdict"] == "PASS"

    # A per-actor failure is contained the same way, actor by actor.
    def _boom(_criteria, _vocab):
        raise RuntimeError("resolver blew up")

    healthy = [_actor([{"criterion": "c", "status": "supported",
                        "evidence_refs": ["claim_1"]}])]
    original = review_evidence.resolve_criteria_evidence_refs
    review_evidence.resolve_criteria_evidence_refs = _boom
    try:
        annotate_criteria_evidence_resolution(healthy, _PACKET)
    finally:
        review_evidence.resolve_criteria_evidence_refs = original
    assert healthy[0]["criteria_refs_unresolved"][0]["supported_evidence_resolves"] is False
    assert task_acceptance_is_clean(_result(healthy)) is False


def test_panel_call_site_does_not_swallow_the_resolution_pass():
    """review_substrate calls the annotator UNGUARDED: an `except: pass`-shaped
    guard there could only convert 'the host never checked the refs' into a clean
    PASS, and the annotator is already total + fail-closed."""
    import inspect

    from ouroboros.review_substrate import run_review_request

    source = inspect.getsource(run_review_request)
    call = source.index("annotate_criteria_evidence_resolution(result.actors")
    assert "try:" not in source[:call]


def test_require_criterion_evidence_knob_is_deleted():
    import pathlib

    for rel in ("ouroboros/loop.py", "ouroboros/tools/review.py"):
        source = pathlib.Path(rel).read_text(encoding="utf-8")
        assert '"require_criterion_evidence"' not in source, rel
    # The evidence condition is unconditional: a solved PASS without criteria is
    # not clean regardless of any policy content.
    knobless = _actor(None)
    knobless["parsed"].pop("criteria_used")
    assert task_acceptance_is_clean(_result([knobless])) is False


def test_reviewer_prompt_names_the_ref_vocabulary_once():
    from ouroboros.review_execution import _render_prompt_parts
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    request = ReviewRequest(
        surface="task_acceptance", goal="g", subject="s",
        policy={"classify_outcome_tier": True, "min_successful_slots": 1},
    )
    stable, _task, _dynamic = _render_prompt_parts(request, ReviewSlot("s0", "m"))
    assert "EXACT match" in stable
    assert "verification_receipts[i]" in stable
    assert "task_contract.acceptance_claims" in stable
    # The prompt must not teach a resolution the host no longer performs: the
    # agent's own sections and the intent container are named as NOT evidence.
    assert "HOST-ATTESTED top-level packet section names" in stable
    assert "reasoning_notes" in stable
    # Non-acceptance surfaces carry no vocabulary line.
    generic = ReviewRequest(surface="commit_review", goal="g", subject="s")
    g_stable, _t, _d = _render_prompt_parts(generic, ReviewSlot("s0", "m"))
    assert "verification_receipts[i]" not in g_stable


def test_annotation_rides_the_panel_and_never_blocks_it(tmp_path):
    """End-to-end through run_review_request: the annotation lands on actor rows
    while quorum participation and the aggregate stay untouched."""
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    class _PassWithBadRefsLLM:
        def chat(self, **_kwargs):
            return {"content": json.dumps({
                "verdict": "PASS", "outcome_tier": "solved",
                "completion_coach": "ship",
                "criteria_used": [{"criterion": "c", "status": "supported",
                                   "evidence_refs": ["hallucinated-ref"]}],
                "findings": [], "summary": "ready",
            })}, {}

    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance", goal="g", subject="candidate",
            evidence=dict(_PACKET),
            policy={"classify_outcome_tier": True, "min_successful_slots": 1},
            task_id="dq5-e2e",
        ),
        slots=[ReviewSlot("s0", "m-0")],
        drive_root=tmp_path,
        llm=_PassWithBadRefsLLM(),
    )

    assert result.aggregate_signal == "PASS"          # verdict untouched
    actor = result.actors[0]
    assert actor["quorum_contribution"] is True        # quorum untouched
    rows = actor["criteria_refs_unresolved"]
    assert rows[0]["supported_evidence_resolves"] is False
    assert task_acceptance_is_clean(result) is False   # only the clean bit demotes


def test_an_open_wave_exhibit_is_declared_intent_and_never_resolves(tmp_path):
    """AP5: the exhibit is disclosed BY NAME. Citing it (or one of its claim
    ids) must not certify anything, because an open wave binds nothing."""
    from ouroboros.review_evidence_refs import (
        DECLARED_INTENT_SECTION,
        acceptance_evidence_ref_vocabulary,
        resolve_criteria_evidence_refs,
    )

    packet = {
        "task_contract": {"requirements": "do X"},
        "acceptance_claims_source": "none_open_plan_wave",
        "plan_claims_exhibit": {
            "binding": "not bound: wave open",
            "acceptance_claims": [{"id": "claim_1", "claim": "the widget renders"}],
        },
        "repo_diff": "diff --git a/x b/x",
        "__provenance__": {
            "task_contract": "host_attested",
            "acceptance_claims_source": "host_attested",
            "plan_claims_exhibit": "host_attested",
            "repo_diff": "host_attested",
        },
    }
    vocabulary = acceptance_evidence_ref_vocabulary(packet)
    assert vocabulary["plan_claims_exhibit"] == DECLARED_INTENT_SECTION
    assert vocabulary["repo_diff"] == "packet_section"

    rows = resolve_criteria_evidence_refs(
        [{"criterion": "c", "status": "supported", "evidence_refs": ["plan_claims_exhibit"]}],
        vocabulary,
    )
    assert rows and rows[0]["supported_evidence_resolves"] is False
