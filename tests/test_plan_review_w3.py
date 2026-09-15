"""``plan_task`` — the W3 governance pack and host-attached reviewer evidence (owner decision B,
2026-08-16) plus the durable-state / entry bounds hardened in the focused delta reviews.

Shares the engine harness with ``test_plan_review_engine`` (real ``ToolContext``, real
``plan_spec``/``plan_packet``/``task_results`` v2 code, one fake review substrate).
"""

from __future__ import annotations

import json
import pytest

import ouroboros.tools.plan_review as pr
from tests.test_plan_review_engine import CLEAN, DECK_SPEC, _call, _control, _finding, _slots, _state, _user_text
from tests.test_plan_review_engine import harness as _engine_harness  # the shared fixture, re-exported below

harness = _engine_harness  # noqa: F811 - pytest registers the fixture under this module's namespace


@pytest.mark.parametrize("notice_prefix", ["", "source detail " * 4000], ids=["complete", "budgeted"])
def test_task_locator_preserves_host_notice_and_invalidates_changed_evidence(harness, notice_prefix):
    from ouroboros.task_results import load_task_result, write_task_result

    answer = "The report is complete."
    write_task_result(harness.drive, "previous", "completed", result=answer)
    reader = pr._task_evidence_reader(harness.drive)
    assert "terminal_host_notice" not in json.loads(reader("previous"))
    first_notice = notice_prefix + "First source limitation."
    write_task_result(harness.drive, "previous", "completed", terminal_host_notice=first_notice)
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    spec = {**DECK_SPEC, "evidence": ["task:previous"]}
    request = pr._PlanRequest(goal="Ship the deck", plan="Outline first, then draft each slide.", spec=spec)
    packet = pr.build_plan_review_packet_for_dry_run(ctx, request)
    assert _control(_call(ctx, spec=spec))["closed"] is True
    first = _state(harness)["waves"][-1]
    assert first["request_fingerprint"] == packet["fingerprint"]
    attached = packet["manifest"]["attached"][0]
    assert '"terminal_host_notice":' in attached["text"]
    assert '"terminal_host_notice":' in _user_text(sub.calls[0]["request"].messages[1]["content"])
    assert json.loads(reader("previous"))["result"] == answer
    assert _control(_call(ctx, spec=spec))["closed"] is True and len(sub.calls) == 1

    write_task_result(harness.drive, "previous", "completed",
                      terminal_host_notice=notice_prefix + "Second source limitation.")
    packet = pr.build_plan_review_packet_for_dry_run(ctx, request)
    assert _control(_call(ctx, spec=spec))["closed"] is True
    second = _state(harness)["waves"][-1]
    assert second["request_fingerprint"] == packet["fingerprint"]
    changed = packet["manifest"]["attached"][0]
    assert len(sub.calls) == 2 and _state(harness)["cycles_paid"] == 2
    assert second["request_fingerprint"] != first["request_fingerprint"]
    assert changed["sha256"] != attached["sha256"]
    assert load_task_result(harness.drive, "previous")["result"] == answer
    if notice_prefix:
        assert changed["text"] == attached["text"]  # The changed tail is beyond the view.
        assert any(o["reason"].startswith("truncated_to_") for o in second["evidence_manifest"]["omissions"])
    else:
        assert json.loads(attached["text"])["terminal_host_notice"] == first_notice
        assert json.loads(changed["text"])["terminal_host_notice"] == "Second source limitation."


@pytest.mark.parametrize("decision", ["accept", "reject", "defer"])
def test_closed_notes_allow_voluntary_disposition_without_new_authority(harness, decision):
    notes = json.dumps([_finding("n1", "note"), _finding("n2", "note")])
    sub = harness.install({"s1": notes, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    assert _control(_call(ctx)) == {"outcome": "REVIEW_REQUIRED", "closed": True}
    before = _state(harness)
    wave = before["waves"][-1]
    fingerprint = wave["request_fingerprint"]
    prior_ref = wave["wave_artifact"]
    exact_before = pr._read_plan_review_wave_artifact(harness.drive, "task-1", prior_ref)
    items = [{"finding_id": "s1:n1", "decision": decision, "rationale": "Optional design judgment"}]

    out = pr._handle_plan_task(ctx, review_disposition={"review_fingerprint": fingerprint, "items": items})

    assert _control(out) == {"outcome": "REVIEW_REQUIRED", "closed": True}
    assert "Notes are optional" in out and "neither reopens" in out
    after = _state(harness)
    annotated = after["waves"][-1]
    assert annotated["dispositions"] == items  # n2 is not an obligation
    assert after["cycles_paid"] == before["cycles_paid"] == 1
    assert after["current_attempt"] == before["current_attempt"]
    for key in ("spec", "spec_hash", "findings", "aggregate", "closed", "actors"):
        assert annotated[key] == wave[key]
    exact = pr._read_plan_review_wave_artifact(harness.drive, "task-1", annotated["wave_artifact"])
    assert exact["supersedes_wave_artifact"] == prior_ref
    assert exact["dispositions"] == items
    assert pr._read_plan_review_wave_artifact(harness.drive, "task-1", prior_ref) == exact_before
    assert _control(_call(ctx)) == {"outcome": "REVIEW_REQUIRED", "closed": True}
    assert len(sub.calls) == 1


def test_closed_note_annotation_keeps_input_validation_and_current_wave_binding(harness):
    from ouroboros.task_results import record_plan_review_dispositions

    sub = harness.install({"s1": json.dumps([_finding("n1", "note")]), "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    _call(ctx)
    before = _state(harness)
    fp = before["waves"][-1]["request_fingerprint"]
    unknown = pr._handle_plan_task(ctx, review_disposition={"review_fingerprint": fp, "items": [
        {"finding_id": "missing", "decision": "accept", "rationale": "r"},
    ]})
    assert unknown.startswith("ERROR: PLAN_REVIEW_DISPOSITION_INVALID")
    assert _state(harness) == before
    # Existing duplicate-disposition disclosure remains visible, but optional advice
    # cannot become a new hold just because its annotation is malformed.
    duplicate = pr._handle_plan_task(ctx, review_disposition={"review_fingerprint": fp, "items": [
        {"finding_id": "s1:n1", "decision": "accept", "rationale": "r"},
        {"finding_id": "s1:n1", "decision": "reject", "rationale": "r"},
    ]})
    assert "duplicate_disposition:s1:n1" in duplicate
    assert _control(duplicate)["closed"] is True
    _call(ctx, spec={**DECK_SPEC, "in_scope": ["a 4-slide deck"]})
    current = _state(harness)
    stale = pr._handle_plan_task(ctx, review_disposition={"review_fingerprint": fp, "items": []})
    assert stale.startswith("ERROR: PLAN_REVIEW_DISPOSITION_STALE")
    with pytest.raises(ValueError, match="PLAN_REVIEW_DISPOSITION_STALE"):
        record_plan_review_dispositions(harness.drive, "task-1", fingerprint=fp, dispositions=[], closed=True)
    assert _state(harness) == current
    assert len(sub.calls) == 2


def test_constitutional_packet_without_architecture_is_a_typed_failure(harness):
    """W3/D26: a self-modification packet without ARCHITECTURE.md is an assembly failure the
    agent sees, never a reviewer wave that silently lacks the document."""
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    (harness.system / "docs" / "ARCHITECTURE.md").unlink()
    spec = {**DECK_SPEC, "affected_paths": [str(harness.system / "ouroboros" / "loop.py")]}
    out = _call(harness.make_ctx(), spec=spec)
    assert "ARCHITECTURE.md" in out and "W3" in out
    assert sub.calls == []  # no reviewer was called


def test_reviewer_requested_locator_is_attached_by_the_host_next_cycle(harness):
    """W3: a `need_evidence` locator is attached by the HOST on the next cycle — the agent need
    not re-declare it, and it cannot leave it out; it goes through the same roots/deny policy and
    it changes the wave fingerprint (a resubmitted spec is a NEW paid cycle carrying the file)."""
    (harness.workspace / "budget.md").write_text("budget: 12 EUR\n", encoding="utf-8")
    ask = json.dumps([_finding("f1", "need_evidence", breaks="goal", locator="budget.md",
                               summary="the budget file is not attached")])
    sub = harness.install({"s1": ask, "s2": CLEAN, "s3": CLEAN})
    out1 = _call(harness.make_ctx())
    assert _control(out1) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    wave1 = _state(harness)["waves"][-1]
    first_user = _user_text(sub.calls[0]["request"].messages[1]["content"])
    assert "budget: 12 EUR" not in first_user and "reviewer-requested" not in first_user
    # cycle 2: the SAME spec, no re-declaration — the host attaches budget.md and every slot sees it
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    out2 = _call(harness.make_ctx())
    assert _control(out2) == {"outcome": "GREEN", "closed": True}
    waves = _state(harness)["waves"]
    assert len(waves) == 2 and waves[-1]["request_fingerprint"] != wave1["request_fingerprint"]
    assert waves[-1]["paid"] is True and waves[-1]["evidence_manifest"]["reviewer_requested"] == ["budget.md"]
    for call in sub.calls:
        second_user = _user_text(call["request"].messages[1]["content"])
        assert "budget.md [reviewer-requested]" in second_user and "budget: 12 EUR" in second_user
        assert "attached by the host (W3)" in second_user


def test_reviewer_requested_locator_still_obeys_the_deny_policy(harness):
    """W3 attaches what a reviewer asks for — through the SAME policy: a locator under the runtime
    data plane (the live settings file) is a named `denied_path` omission tagged as requested,
    never attached, never silent. The panel is still dispatched and judges with the absence."""
    denied = str(harness.drive / "settings.json")
    (harness.drive / "settings.json").write_text('{"OPENROUTER_API_KEY": "sk-live-x"}', encoding="utf-8")
    ask = json.dumps([_finding("f1", "need_evidence", breaks="goal", locator=denied,
                               summary="show me the settings")])
    sub = harness.install({"s1": ask, "s2": CLEAN, "s3": CLEAN})
    _call(harness.make_ctx())
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    out = _call(harness.make_ctx())
    assert "denied_path" in out
    assert len(sub.calls) == 1
    sent = _user_text(sub.calls[0]["request"].messages[1]["content"])
    assert "[reviewer-requested]" in sent and "denied_path" in sent
    assert "sk-live-x" not in sent and "sk-live-x" not in out




def test_reviewer_request_for_an_already_declared_locator_is_tagged_and_rehashed(harness):
    """W3 (delta review S-1): a request for a locator the agent ALSO declared still changes what
    the next wave is about — it is tagged `[reviewer-requested]` and enters the manifest hash, so
    the resubmitted spec is a NEW paid cycle, never an idempotent replay of the old fingerprint."""
    (harness.workspace / "notes.md").write_text("deck notes\n", encoding="utf-8")
    spec = {**DECK_SPEC, "evidence": ["notes.md"]}
    ask = json.dumps([_finding("f1", "need_evidence", breaks="goal", locator="notes.md",
                               summary="I need the notes")])
    sub = harness.install({"s1": ask, "s2": CLEAN, "s3": CLEAN})
    _call(harness.make_ctx(), spec=spec)
    fp1 = _state(harness)["waves"][-1]["request_fingerprint"]
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    out2 = _call(harness.make_ctx(), spec=spec)
    assert _control(out2) == {"outcome": "GREEN", "closed": True}
    waves = _state(harness)["waves"]
    assert len(waves) == 2 and waves[-1]["request_fingerprint"] != fp1 and waves[-1]["paid"] is True
    assert waves[-1]["evidence_manifest"]["reviewer_requested"] == ["notes.md"]
    assert waves[-1]["evidence_manifest"]["declared"].count("notes.md") == 1  # resolved once
    user2 = _user_text(sub.calls[-1]["request"].messages[1]["content"])
    assert "notes.md [reviewer-requested]" in user2 and "duplicate_locator" not in user2


def test_reviewer_requests_are_bounded_like_declared_evidence(harness):
    """W3 (delta review S-3): host attachment honours at most MAX_LIST_ITEMS reviewer-requested
    locators per task, each at most MAX_ITEM_CHARS long; what the cap drops is a NAMED
    `reviewer_request_cap` omission, never a silent one."""
    from ouroboros.tools.plan_spec import MAX_ITEM_CHARS, MAX_LIST_ITEMS

    many = [f"asks/f{i:03d}.md" for i in range(MAX_LIST_ITEMS + 3)]
    for rel in many:
        (harness.workspace / rel).parent.mkdir(parents=True, exist_ok=True)
        (harness.workspace / rel).write_text(f"{rel}\n", encoding="utf-8")
    long_loc = "asks/" + "l" * MAX_ITEM_CHARS + ".md"
    findings = [_finding(f"f{i}", "need_evidence", breaks="goal", locator=loc, summary="need it")
                for i, loc in enumerate([long_loc, *many[:16]])]
    findings2 = [_finding(f"g{i}", "need_evidence", breaks="goal", locator=loc, summary="need it")
                 for i, loc in enumerate(many[16:32])]
    findings3 = [_finding(f"h{i}", "need_evidence", breaks="goal", locator=loc, summary="need it")
                 for i, loc in enumerate(many[32:])]
    harness.install({"s1": json.dumps(findings), "s2": json.dumps(findings2), "s3": json.dumps(findings3)})
    _call(harness.make_ctx())
    wave1 = _state(harness)["waves"][-1]
    seen = _state(harness)["need_evidence_seen"]
    # the over-long locator is demoted at validation (never remembered, never attached), disclosed
    assert len(seen) == MAX_LIST_ITEMS + 3 and long_loc not in seen
    s1 = next(a for a in wave1["actors"] if a["slot_id"] == "s1")
    assert "need_evidence_locator_too_long:f0" in s1["disclosures"]
    assert all(len(f["locator"]) <= 2_000 + 200 for f in wave1["findings"])  # stored bounded, marker visible
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    _call(harness.make_ctx())
    manifest = _state(harness)["waves"][-1]["evidence_manifest"]
    assert len(manifest["reviewer_requested"]) == MAX_LIST_ITEMS
    capped = [o for o in manifest["omissions"] if o["reason"] == "reviewer_request_cap"]
    assert len(capped) == 3
    user2 = _user_text(sub.calls[-1]["request"].messages[1]["content"])
    assert "reviewer_request_cap" in user2
    # dropped requests keep their PROVENANCE: tagged as reviewer-requested, never "declared by the agent"
    assert manifest["reviewer_requested_dropped"] and all(
        f"| {loc} [reviewer-requested] | reviewer_request_cap |" in user2 for loc in manifest["reviewer_requested_dropped"])


def test_need_evidence_memory_is_bounded_per_task(harness):
    """W3 (delta review round 6): the per-task request memory (`need_evidence_seen`) is bounded
    (MAX_NEED_EVIDENCE_MEMORY) so the durable review state stays bounded whatever a panel asks
    for; a request past the cap is never remembered, disclosed `need_evidence_memory_full`,
    but still needs a free disposition instead of becoming optional advice;
    within one wave the memory accumulates across slots so the cap is exact."""
    from ouroboros.tools.plan_spec import MAX_NEED_EVIDENCE_MEMORY

    from ouroboros.tools.plan_spec import MAX_FINDINGS_PER_SLOT

    def _answers(cycle):  # every slot asks for MAX_FINDINGS_PER_SLOT distinct locators
        return {sid: json.dumps([
            _finding(f"f{i}", "need_evidence", breaks="goal", locator=f"m/{cycle}/{sid}/{i:04d}.md", summary="x")
            for i in range(MAX_FINDINGS_PER_SLOT)]) for sid in ("s1", "s2", "s3")}

    per_wave = 3 * MAX_FINDINGS_PER_SLOT
    assert per_wave < MAX_NEED_EVIDENCE_MEMORY < 2 * per_wave  # the second wave crosses the cap
    harness.install(_answers(1))
    _call(harness.make_ctx())
    first_seen = _state(harness)["need_evidence_seen"]
    assert len(first_seen) == per_wave
    for locator in first_seen:
        path = harness.workspace / locator
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence\n", encoding="utf-8")
    harness.install(_answers(2))
    _call(harness.make_ctx())  # same spec, new fingerprint (host-attached requests), second paid wave
    state = _state(harness)
    assert len(state["need_evidence_seen"]) == MAX_NEED_EVIDENCE_MEMORY
    wave = state["waves"][-1]
    full = [d for a in wave["actors"] for d in a["disclosures"] if d.startswith("need_evidence_memory_full:")]
    assert len(full) == 2 * per_wave - MAX_NEED_EVIDENCE_MEMORY
    assert all(f["class"] == "need_evidence" for f in wave["findings"])
    assert wave["closed"] is False




def test_unreadable_state_refuses_the_wave_instead_of_dropping_reviewer_requests(harness):
    """W3 (delta review S-4): the reviewers' recorded requests live in the durable state; if it
    cannot be read the wave is REFUSED (typed, no reviewer called) — never paid without them."""
    from ouroboros.task_results import task_result_path

    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    path = task_result_path(harness.drive, "task-1", create=True)
    path.write_text("{not json", encoding="utf-8")
    out = _call(harness.make_ctx())
    assert out.startswith("ERROR: PLAN_REVIEW_STATE_INVALID")
    assert sub.calls == []


def test_oversize_slots_are_typed_rows_and_below_quorum_is_a_typed_refusal(harness, monkeypatch):
    """W3 (delta review S-2): the packet is sized PER SLOT against the review organ's calibrated
    input caps. A slot it cannot fit is a FREE `preflight_oversize` row that still counts in the
    quorum denominator; fewer callable slots than quorum refuses the wave with no reviewer called."""
    from ouroboros.tools import review_synthesis

    caps = {"m/a": 10, "m/b": 10_000_000, "m/c": 10_000_000}
    monkeypatch.setattr(review_synthesis, "per_slot_input_token_limits",
                        lambda models, **kw: {slot.slot_id: caps[str(m)] for m, slot in zip(models, kw["slots"])})
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    out = _call(harness.make_ctx())
    assert _control(out) == {"outcome": "GREEN", "closed": True}
    wave = _state(harness)["waves"][-1]
    rows = {r["slot_id"]: r for r in wave["actors"]}
    assert rows["s1"]["ok"] is False and "preflight_oversize" in rows["s1"]["error"]
    assert rows["s1"]["cost"] == 0.0 and rows["s2"]["ok"] and rows["s3"]["ok"]
    assert len(sub.calls) == 1 and [s.slot_id for s in sub.calls[0]["slots"]] == ["s2", "s3"]
    # two of three below the cap -> fewer than quorum(3)=2 callable -> typed refusal
    caps.update({"m/b": 10})
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx(task_id="task-2")
    out2 = _call(ctx)
    assert "PLAN_REVIEW_DEGRADED_PREFLIGHT_OVERSIZE" in out2 and "m/a<=10" in out2
    assert sub.calls == []


def test_reviewer_requested_text_is_redacted_like_declared_evidence(harness):
    """W3 (delta review G-NIT): a host-attached reviewer-requested file goes through the same
    `redact_prompt_secrets` as declared evidence — a key quoted in an allowed notes file never
    reaches a reviewer, and the wave discloses `secrets_redacted`."""
    (harness.workspace / "asked.md").write_text(
        "notes\nOPENROUTER_API_KEY=sk-or-FAKEFIXTURE-not-a-real-key-ABCDEFGHIJKLMNOP\n", encoding="utf-8")
    ask = json.dumps([_finding("f1", "need_evidence", breaks="goal", locator="asked.md", summary="need it")])
    harness.install({"s1": ask, "s2": CLEAN, "s3": CLEAN})
    _call(harness.make_ctx())
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    _call(harness.make_ctx())
    user2 = _user_text(sub.calls[-1]["request"].messages[1]["content"])
    assert "asked.md [reviewer-requested]" in user2 and "notes" in user2
    assert "sk-or-FAKEFIXTURE" not in user2
    assert "secrets_redacted=true" in user2  # the redaction is DISCLOSED to the reviewers…
    manifest = _state(harness)["waves"][-1]["evidence_manifest"]
    attached = {a["locator"]: a for a in manifest["attached"]}
    assert attached["asked.md"]["secrets_redacted"] is True  # …and in the durable wave
    # host-only locators are NOT reported as agent-declared: `declared` is the agent's own list
    assert manifest["declared"] == [] and manifest["reviewer_requested"] == ["asked.md"]


def test_session_slot_is_never_fit_excluded_and_budget_is_priced_on_callable_slots(harness, monkeypatch):
    """W3 (delta review round 2): a RETRIEVING row's model id is an opaque harness target — it is
    never sized out of the panel (organ convention: session rows retrieve with their own tools);
    and the wave budget gate is priced on the slots that will actually be called, so a $0
    oversize row cannot decline the callable quorum."""
    from ouroboros.tools import review_synthesis
    import ouroboros.tools.plan_review as pr_mod

    harness.state["slots"] = _slots(("s1", "m/a"), ("s2", "m/b"), ("s3", "cursor=grok", "session"))
    caps = {"m/a": 10, "m/b": 10_000_000}
    monkeypatch.setattr(review_synthesis, "per_slot_input_token_limits",
                        lambda models, **kw: {slot.slot_id: caps[str(m)] for m, slot in zip(models, kw["slots"])})
    priced: list = []
    monkeypatch.setattr(pr_mod, "review_wave_budget_gate",
                        lambda ctx, **kw: priced.append(list(kw["models"])) or None)
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    out = _call(harness.make_ctx())
    assert _control(out) == {"outcome": "GREEN", "closed": True}
    assert [s.slot_id for s in sub.calls[0]["slots"]] == ["s2", "s3"]
    assert priced == [["m/b", "cursor=grok"]]
    rows = {r["slot_id"]: r for r in _state(harness)["waves"][-1]["actors"]}
    assert rows["s1"]["ok"] is False and rows["s3"]["ok"] is True


def test_session_task_is_the_compact_form_with_governance_by_mandatory_retrieval(harness):
    """W3 (delta review round 3): the executor's session prompt is COMPACT by contract — a
    retrieving reviewer gets the same stance/rubric/checklist and the redacted evidence, but for a
    self-modification plan the governance documents by MANDATORY full retrieval at their
    resolvable locators, never ~500k chars inline; api rows still get them inline."""
    harness.state["slots"] = _slots(("api1", "m/a"), ("sess1", "cursor=grok", "session"), ("api2", "m/b"))
    sub = harness.install({"api1": CLEAN, "sess1": CLEAN, "api2": CLEAN})
    spec = {**DECK_SPEC, "affected_paths": [str(harness.system / "ouroboros" / "loop.py")]}
    _call(harness.make_ctx(), spec=spec)
    request = sub.calls[0]["request"]
    api_system = request.messages[0]["content"][0]["text"]
    assert "slots and quorum." in api_system and "Principle 3: Immune Integrity\n\nreview." in api_system
    task = request.session_task
    assert "MANDATORY FULL READS" in task
    assert str(harness.system / "BIBLE.md") in task and str(harness.system / "docs" / "ARCHITECTURE.md") in task
    assert "slots and quorum." not in task and "Principle 3: Immune Integrity\n\nreview." not in task
    assert "Plan Review Checklist" in task and "REDACTED snapshot" in task
    # the governance documents are the ONE raw-read exception, even when the agent ALSO declared
    # BIBLE.md as evidence beside changing it (the change target is what makes the plan
    # constitutional — owner 16=A): no contradictory orders
    sub = harness.install({"api1": CLEAN, "sess1": CLEAN, "api2": CLEAN})
    _call(harness.make_ctx(task_id="task-2"), spec={**DECK_SPEC,
          "affected_paths": [str(harness.system / "BIBLE.md")],
          "evidence": [str(harness.system / "BIBLE.md")]})
    task2 = sub.calls[0]["request"].session_task
    assert "MANDATORY FULL READS" in task2 and "even if the agent also declared them as evidence" in task2
    assert f"### {harness.system / 'BIBLE.md'}" in task2  # the redacted snapshot is still there too
    # a NON-constitutional plan: the governance documents are optional pointers for a session too
    sub = harness.install({"api1": CLEAN, "sess1": CLEAN, "api2": CLEAN})
    _call(harness.make_ctx(task_id="task-3"))
    task3 = sub.calls[0]["request"].session_task
    assert "## Governance pack" not in task3 and "MAY read raw" in task3  # no mandatory-read section
    assert "ARCHITECTURE navigation map" in task3 and "on-demand pointer" in task3


def test_state_stays_persistable_at_the_worst_case_request_bounds(tmp_path):
    """W3 (delta review round 7): the maximal panel — 10 slots x MAX_FINDINGS_PER_SLOT requests,
    every locator MAX_ITEM_CHARS four-byte characters, the request memory full — must still
    persist under the 1 MB durable bound so `cycles_paid` advances and the paid panel is never
    re-paid; the last-resort cut of locators / request memory is DISCLOSED."""
    from ouroboros import task_results as tr
    from ouroboros.tools import plan_spec

    wide = "𝕏" * plan_spec.MAX_ITEM_CHARS  # 4-byte UTF-8 each
    n_items = plan_spec.MAX_LIST_ITEMS
    # A large operative spec beside maximally populated reviewer-request memory.
    spec = {
        "goal": "Ship", "acceptance_claims": [f"{wide[:-4]}c{i:03d}" for i in range(n_items)],
        "in_scope": [f"{wide[:-4]}i{i:03d}" for i in range(n_items)],
        "non_goals": [f"{wide[:-4]}n{i:03d}" for i in range(n_items)],
        "invariants": [f"{wide[:-4]}v{i:03d}" for i in range(n_items)],
        "decisions": [{"choice": wide, "why": wide,
                       "rejected": [wide] * 8} for _ in range(n_items)],
        "deferred": [{"what": wide, "why_safe_to_defer": wide} for _ in range(n_items)],
        "affected_paths": [f"{wide[:-4]}a{i:03d}" for i in range(n_items)],
        "affected_resources": [f"{wide[:-4]}r{i:03d}" for i in range(n_items)],
        "evidence": [f"{wide[:-4]}e{i:03d}" for i in range(n_items)],
    }
    normalized, errors = plan_spec.normalize_spec(spec)
    assert errors == []
    memory = [f"{wide[:-4]}{i:04d}" for i in range(plan_spec.MAX_NEED_EVIDENCE_MEMORY)]
    for cycle in (1, 2):
        findings = [
            {"finding_id": f"s{slot}:f{n}", "slot": f"s{slot}", "id": f"f{n}",
             "class": "need_evidence" if n < 16 else "note", "locator": f"{wide[:-6]}{slot:02d}{n:04d}",
             "summary": "x" * plan_spec.MAX_FINDING_TEXT_CHARS,
             "recommendation": "y" * plan_spec.MAX_FINDING_TEXT_CHARS}
            for slot in range(1, 11) for n in range(plan_spec.MAX_FINDINGS_PER_SLOT)
        ]
        # every slot's disclosure list carries full-width repeat locators too
        actors = [{"slot_id": f"s{slot}", "model": "m", "ok": True, "error": None,
                   "disclosures": [f"need_evidence_repeat:{wide[:-6]}{slot:02d}{n:04d}"
                                   for n in range(plan_spec.MAX_FINDINGS_PER_SLOT)]}
                  for slot in range(1, 11)]
        omissions = [{"locator": m, "reason": "reviewer_request_cap"} for m in memory[40:]]
        wave = {
            "cycle_index": cycle, "request_fingerprint": f"{cycle:064x}", "spec": normalized,
            "spec_hash": f"{cycle:064x}",
            "evidence_manifest": {"declared": list(normalized["evidence"]), "attached": [], "omissions": omissions,
                                  "reviewer_requested": memory[:40], "reviewer_requested_dropped": memory[40:]},
            "evidence_manifest_hash": f"{cycle:064x}", "constitutional": False, "findings": findings,
            "actors": actors,
            "aggregate": "REVIEW_REQUIRED", "reasons": [], "closed": False, "dispositions": [], "paid": True,
        }
        tr.record_plan_review_wave(tmp_path, "t1", wave=wave, need_evidence_seen=memory)
        state = tr.load_plan_review_state(tmp_path, "t1")
        assert state["cycles_paid"] == cycle
        assert len(json.dumps(state, ensure_ascii=False).encode("utf-8")) <= 1_000_000
    newest = state["waves"][-1]
    assert newest["findings"] and newest["findings_texts_truncated"] is True
    # R10-2: the fit keeps headroom, so the post-cut stamps can never push the state back over
    assert len(json.dumps(state, ensure_ascii=False).encode("utf-8")) <= 1_000_000 - 256
    assert state.get("request_memory_truncated") is True
    assert all("truncated to fit" in x for x in state["need_evidence_seen"] if len(x) > 80)
    # identity survives the cut: hashes, ids, classes, the goal and the acceptance claims
    assert newest["spec_hash"] == f"{2:064x}" and newest["spec"]["goal"] == "Ship"
    assert newest["spec"]["acceptance_claims"] == normalized["acceptance_claims"]
    assert {f["class"] for f in newest["findings"]} == {"need_evidence", "note"}
    assert all("truncated to fit" not in f["id"] for f in newest["findings"])
    # R9-3: a cut frozen body is STAMPED — its hashes name the original, no false delta is claimed
    assert newest["spec_body_truncated"] is True
    # R9-4: later writers on a near-limit state still persist — a closure and the cap stamp
    tr.record_plan_review_dispositions(
        tmp_path, "t1", fingerprint=f"{2:064x}", closed=False,
        dispositions=[{"finding_id": f"s{slot}:f{n}", "decision": "accept", "rationale": "ok" * 500}
                      for slot in range(1, 11) for n in range(plan_spec.MAX_FINDINGS_PER_SLOT)],
        closure_notes=["x"])
    tr.mark_plan_review_cycles_exhausted(tmp_path, "t1", fingerprint=f"{2:064x}")
    state = tr.load_plan_review_state(tmp_path, "t1")
    assert state["waves"][-1]["cycles_exhausted"] is True and len(state["waves"][-1]["dispositions"]) == 320
    assert len(json.dumps(state, ensure_ascii=False).encode("utf-8")) <= 1_000_000


def test_worst_case_state_successor_receives_the_current_decision_core(tmp_path):
    from types import SimpleNamespace

    from ouroboros.agent_startup_checks import validate_task_authority_sources

    test_state_stays_persistable_at_the_worst_case_request_bounds(tmp_path)
    source = {
        "kind": "task_result", "task_id": "t1", "tool": "get_task_result",
        "arguments": {"task_id": "t1", "include_authority": True},
    }
    task = {
        "id": "successor", "budget_drive_root": str(tmp_path),
        "predecessor_authority_source": source,
    }

    assert validate_task_authority_sources(SimpleNamespace(
        drive_root=tmp_path, budget_drive_root=tmp_path,
    ), task) == {}
    carried = task["predecessor_authority"]["plan_review_state"]
    assert carried.get("kind") != "bounded_field_preview"
    assert set(carried["need_evidence_seen"]) == {"items", "items_omitted", "total"}
    preview = json.dumps(carried, ensure_ascii=False, sort_keys=True, default=str)
    assert len(preview) <= 15_000
    for decision_fact in (
        '"acceptance_claims"', '"findings"', '"dispositions"',
        f'"request_fingerprint": "{2:064x}"',
    ):
        assert decision_fact in preview


def test_below_quorum_blocking_rejection_earns_the_promised_delta_cycle(harness, monkeypatch):
    """R9-5: a REVIEW_REQUIRED wave carrying ONE below-quorum blocking finding cannot be closed
    by disposition (C-08); the closure table promises "the next paid delta cycle" for its
    rejection — so an identical envelope after a valid reject must RUN that paid delta panel,
    not replay the cached wave forever."""
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "5")
    blocking = json.dumps([_finding("f1", "blocking", breaks="claim_1")])
    harness.install({"s1": blocking, "s2": CLEAN, "s3": CLEAN})  # 1 of 3 < quorum(2)
    ctx = harness.make_ctx()
    out = _call(ctx)
    assert _control(out) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    fp = _state(harness)["waves"][-1]["request_fingerprint"]
    replay = _call(ctx)  # nothing rejected yet: idempotent replay
    assert "cached" in replay.lower() and _state(harness)["cycles_paid"] == 1
    closed = pr._handle_plan_task(ctx, review_disposition={"review_fingerprint": fp, "items": [
        {"finding_id": "s1:f1", "decision": "reject", "rationale": "the visa is already granted"},
    ]})
    assert _control(closed) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    delta = _call(ctx)  # same envelope: the rejection now buys the promised delta panel
    assert _control(delta) == {"outcome": "GREEN", "closed": True}
    assert len(sub.calls) == 1 and _state(harness)["cycles_paid"] == 2


def test_disposition_inputs_are_bounded_at_entry(harness):
    """R9-4: a disposition is bounded like the findings it answers — the rationale text and the
    item count — so a $0 closure can always be persisted."""
    from ouroboros.tools import plan_spec

    note = json.dumps([_finding("n1", "need_evidence", locator="notes.md")])
    harness.install({"s1": note, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    _call(ctx)
    fp = _state(harness)["waves"][-1]["request_fingerprint"]
    too_many = pr._handle_plan_task(ctx, review_disposition={"review_fingerprint": fp, "items": [
        {"finding_id": "s1:n1", "decision": "accept", "rationale": "ok"} for _ in range(50)]})
    assert too_many.startswith("ERROR: PLAN_REVIEW_DISPOSITION_INVALID")
    huge = "r" * (plan_spec.MAX_FINDING_TEXT_CHARS * 20)
    out = pr._handle_plan_task(ctx, review_disposition={"review_fingerprint": fp, "items": [
        {"finding_id": "s1:n1", "decision": "accept", "rationale": huge}]})
    assert _control(out) == {"outcome": "REVIEW_REQUIRED", "closed": True}
    stored = _state(harness)["waves"][-1]["dispositions"][0]
    assert len(stored["rationale"]) < plan_spec.MAX_FINDING_TEXT_CHARS + 200 and "truncat" in stored["rationale"].lower()
    # R10-1: `decision` is enum-like and bounded at entry — identity keys are never wide carriers
    huge_decision = pr._handle_plan_task(ctx, review_disposition={"review_fingerprint": fp, "items": [
        {"finding_id": "s1:n1", "decision": "accept" + "x" * 5_000, "rationale": "ok"}]})
    assert "invalid_disposition" in huge_decision or "already_closed" in huge_decision
    for w in _state(harness)["waves"]:
        for d in w.get("dispositions") or []:
            assert len(d.get("decision") or "") <= 40


def test_breaks_is_bounded_like_an_id_and_blank_items_are_one_disclosure():
    """R9-1/R9-2: `breaks` is an id (bounded, whatever the class); a list of blank items is ONE
    bounded disclosure per list, so the disclosure list itself cannot outgrow the state."""
    from ouroboros.tools import plan_spec

    normalized, disclosures, _ = plan_spec.validate_findings(
        [{"id": "n1", "class": "note", "breaks": "b" * 5_000, "summary": "s"}],
        spec_ids={"goal"}, seen_locators=set(), slot="s1")
    assert len(normalized[0]["breaks"]) == plan_spec.MAX_ID_CHARS
    spec, errors = plan_spec.normalize_spec({"goal": "g", "acceptance_claims": ["a"], "in_scope": [""] * 30_000})
    assert errors == []
    blanks = [n for n in spec["normalization_omissions"] if "blank item(s) dropped" in n]
    assert blanks == ["in_scope: 30000 blank item(s) dropped"]


def test_an_unsatisfiable_reviewer_request_never_locks_the_panel(harness):
    """A locator the host will never fetch (a URL) is a named omission, not a permanent
    $0 refusal: the next cycle is dispatched and the panel judges with the absence."""
    ask = json.dumps([_finding("f1", "need_evidence", breaks="goal",
                               locator="https://example.com/spec",
                               summary="read the vendor page")])
    sub = harness.install({"s1": ask, "s2": CLEAN, "s3": CLEAN})
    _call(harness.make_ctx())
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    out = _call(harness.make_ctx())
    assert len(sub.calls) == 1
    sent = _user_text(sub.calls[0]["request"].messages[1]["content"])
    assert "url_not_fetched" in sent and "[reviewer-requested]" in sent
    wave = _state(harness)["waves"][-1]
    assert wave["paid"] is True and wave["actors"]
    assert not any(str(r).startswith("cannot_verify") for r in wave.get("reasons") or [])
    assert "cannot_verify" not in out


def test_a_truncated_requested_document_dispatches_with_the_cut_named(harness):
    """A requested source above the per-item byte bound is attached head-first with the
    cut named; the panel is dispatched, never refused for the truncation."""
    from ouroboros.tools.plan_evidence import EVIDENCE_PER_ITEM_BYTES

    big = harness.workspace / "huge.txt"
    big.write_text("h" * (EVIDENCE_PER_ITEM_BYTES + 5_000), encoding="utf-8")
    ask = json.dumps([_finding("f1", "need_evidence", breaks="goal", locator="huge.txt",
                               summary="read the whole document")])
    sub = harness.install({"s1": ask, "s2": CLEAN, "s3": CLEAN})
    _call(harness.make_ctx())
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    out = _call(harness.make_ctx())
    assert len(sub.calls) == 1
    sent = _user_text(sub.calls[0]["request"].messages[1]["content"])
    assert f"truncated_to_{EVIDENCE_PER_ITEM_BYTES}" in sent
    assert "hhh" in sent  # the head IS attached, not withheld
    assert _state(harness)["waves"][-1]["paid"] is True
    assert "cannot_verify" not in out


def test_a_compacted_paid_predecessor_degrades_to_a_fresh_dispatch(harness, monkeypatch):
    """When the prior exact wave is gone (compacted out of the hot state), the evidence
    continuation is a cache miss: the wave re-dispatches fresh and every slot row
    discloses the typed `prior_exact_wave_missing` cause."""
    (harness.workspace / "notes.md").write_text("deck notes\n", encoding="utf-8")
    ask = json.dumps([_finding("f1", "need_evidence", breaks="goal", locator="notes.md",
                               summary="I need the notes")])
    sub = harness.install({"s1": ask, "s2": CLEAN, "s3": CLEAN})
    _call(harness.make_ctx())
    monkeypatch.setattr(pr, "_last_paid_wave", lambda state: None)
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    out = _call(harness.make_ctx())
    assert len(sub.calls) == 1  # dispatched, not refused
    wave = _state(harness)["waves"][-1]
    assert wave["paid"] is True
    deltas = [d for row in wave["actors"] for d in row.get("capability_delta") or []]
    assert deltas and all(d["kind"] == "capability_delta" for d in deltas)
    assert {d["reason"] for d in deltas} == {"prior_exact_wave_missing"}
    assert "cannot_verify" not in out


def test_first_cycle_and_no_request_delta_cycle_carry_no_continuation_delta(harness):
    """The `reviewer_requested` guard is load-bearing: a cycle with no reviewer request
    has no prior thread to continue, so it must not disclose a missing predecessor."""
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    _call(harness.make_ctx())
    wave = _state(harness)["waves"][-1]
    assert not [d for row in wave["actors"] for d in row.get("capability_delta") or []]

    blocking = json.dumps([_finding("f1", "blocking", breaks="goal", summary="thin plan")])
    sub = harness.install({"s1": blocking, "s2": CLEAN, "s3": CLEAN})
    _call(harness.make_ctx(task_id="task-2"))
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    _call(harness.make_ctx(task_id="task-2"), plan="Outline first, then draft each slide. Revised.")
    wave2 = _state(harness, "task-2")["waves"][-1]
    assert len(sub.calls) == 1
    assert not [d for row in wave2["actors"] for d in row.get("capability_delta") or []]


def test_missing_requested_evidence_reask_keeps_free_disposition_without_new_attachment(harness):
    """A repeated need stays open, without growing request memory or buying another panel."""
    ask = json.dumps([_finding("f1", "need_evidence", breaks="goal", locator="gone.md",
                               summary="read it")])
    sub = harness.install({"s1": ask, "s2": CLEAN, "s3": CLEAN})
    _call(harness.make_ctx())
    sub = harness.install({"s1": ask, "s2": CLEAN, "s3": CLEAN})
    out = _call(harness.make_ctx())
    assert len(sub.calls) == 1
    wave = _state(harness)["waves"][-1]
    assert wave["paid"] is True and wave["closed"] is False
    repeat = [f for f in wave["findings"] if f["locator"] == "gone.md"]
    assert repeat and [f["class"] for f in repeat] == ["need_evidence"]
    assert _control(out)["closed"] is False
    assert _state(harness)["need_evidence_seen"] == ["gone.md"]
    assert _control(_call(harness.make_ctx()))["closed"] is False  # Exact replay, not another panel.
    assert len(sub.calls) == 1 and _state(harness)["cycles_paid"] == 2
    disposed = pr._handle_plan_task(harness.make_ctx(), review_disposition={
        "review_fingerprint": wave["request_fingerprint"],
        "items": [{"finding_id": repeat[0]["finding_id"], "decision": "defer",
                   "rationale": "The source is unavailable; it is not needed to begin this work."}],
    })
    assert _control(disposed) == {"outcome": "REVIEW_REQUIRED", "closed": True}
    assert len(sub.calls) == 1 and _state(harness)["cycles_paid"] == 2


def test_both_reviewer_routes_learn_the_range_selectors(harness):
    """The locator forms live in ONE element schema, so the api system prompt and a session
    slot's compact task both advertise them — neither route is taught a narrower spelling."""
    harness.state["slots"] = _slots(("api1", "m/a"), ("sess1", "cursor=grok", "session"),
                                    ("api2", "m/b"))
    sub = harness.install({"api1": CLEAN, "sess1": CLEAN, "api2": CLEAN})
    _call(harness.make_ctx())
    request = sub.calls[0]["request"]
    system = _user_text(request.messages[0]["content"])
    assert "::lines=A-B" in system and "never fetches" in system
    assert "::lines=A-B" in request.session_task


def test_the_plan_spec_schema_names_affected_paths_as_the_only_resolved_list():
    """The trigger reads ONE list — `affected_paths` — and the schema the agent sees says so:
    required, files only, `[]` when none, and the other two lists explicitly not resolved."""
    assert pr._SPEC_SCHEMA["required"] == ["affected_paths"]
    props = pr._SPEC_SCHEMA["properties"]
    paths = props["affected_paths"]["description"]
    assert "REQUIRED" in paths and "system repository" in paths and "[]" in paths
    resources = props["affected_resources"]["description"]
    assert "never file paths" in resources and "system repository" not in resources
    evidence = props["evidence"]["description"]
    assert "does not make the plan" in evidence and "affected_paths" in evidence


# ------------------------------------------------------------- in-flight honesty (P1-4)


def _actor(slot_id, *, ok=False, failure_code="", error=""):
    return {"slot_id": slot_id, "model": "m", "ok": ok, "failure_code": failure_code, "error": error}


def test_progress_line_dedups_typed_reasons_and_names_the_late_result():
    from ouroboros.tools.plan_review_runtime import plan_wave_progress_line

    counts = {"parseable": 0, "configured": 6, "blocking": 0, "note": 0, "need_evidence": 0}
    same = [_actor(f"s{i}", failure_code="subscription_window_exhausted") for i in range(3)]
    distinct = [_actor("d1", failure_code="credential_pool_exhausted"), _actor("d2", error="transport died"),
                _actor("d3", error="x" * 400), _actor("d4", failure_code="deadline_exhausted")]
    line = plan_wave_progress_line("DEGRADED", counts, cycles_paid=1, cap=2,
                                   wave={"actors": same + distinct, "custody_pending": True})
    assert line.count("subscription_window_exhausted") == 1  # three identical reasons -> one
    assert "credential_pool_exhausted; transport died" in line
    assert "(+1 more in the task result)" in line and "deadline_exhausted" not in line  # first four shown
    assert "OMISSION NOTE" in line and "\n" not in line  # bounded, one line
    assert line.endswith("late result pending (reviewer slots still in flight, not yet collected)")
    # Every other aggregate renders byte-identically to the plain form.
    plain = plan_wave_progress_line("GREEN", {**counts, "parseable": 6}, cycles_paid=1, cap=2)
    assert plain == plan_wave_progress_line("GREEN", {**counts, "parseable": 6}, cycles_paid=1, cap=2,
                                            wave={"actors": same, "custody_pending": False})
    assert plain == "📐 plan_task: GREEN — 0 blocking / 0 note / 0 need_evidence; cycles paid 1/2"


def test_refused_redispatch_emits_a_separate_no_dispatch_line(harness, monkeypatch):
    import ouroboros.review_substrate as review_substrate
    from types import SimpleNamespace

    harness.install({"s1": "", "s2": "", "s3": ""})  # every slot dies at dispatch time: paid, empty epoch
    ctx = harness.make_ctx()
    assert _control(_call(ctx)) == {"outcome": "DEGRADED", "closed": False}
    assert _state(harness)["cycles_paid"] == 1

    def zero_send(request, *, slots, drive_root, llm, usage_ctx=None):
        return SimpleNamespace(actors=[{
            "slot_id": slot.slot_id, "model": slot.model, "status": "not_dispatched", "raw_text": "",
            "error": "agent session slot has no session task", "failure_code": "session_task_missing",
            "usage": {}, "prompt_ref": {}, "response_ref": {}, "operation_id": f"op-{slot.slot_id}",
            "operation_state": "not_dispatched", "late_result_pending": False,
        } for slot in slots])

    monkeypatch.setattr(review_substrate, "run_review_request", zero_send)
    harness.progress.clear()
    _call(ctx)  # stale empty-epoch wave re-dispatches; every row refuses pre-send at $0
    assert _state(harness)["cycles_paid"] == 1
    no_dispatch = [line for line in harness.progress if line.startswith("📐 plan_task: no new reviewer cycle dispatched")]
    assert no_dispatch == ["📐 plan_task: no new reviewer cycle dispatched: session_task_missing"]
    assert harness.progress[-1].startswith("📐 plan_task: DEGRADED") and "session_task_missing" in harness.progress[-1]


def test_gate_projection_carries_custody_pending_before_the_aggregate():
    from ouroboros.task_results import plan_review_gate_projection
    from tests.test_plan_review import _force_plan_gate_state

    state = _force_plan_gate_state("degraded")
    assert plan_review_gate_projection(state, "blocking")["custody_pending"] is False
    state["waves"][0]["custody_pending"] = True
    decision = plan_review_gate_projection(state, "blocking")
    assert decision["custody_pending"] is True and decision["reviewer_slots_degraded"] is True
    assert decision["allow"] is False  # the in-flight aggregate stays DEGRADED and holds


def test_one_free_collection_before_the_blocking_gate(harness, monkeypatch):
    from ouroboros.task_results import plan_review_gate_projection
    from ouroboros.tools.plan_review_collect import collect_before_gate
    from tests.test_plan_review_reconciliation import _install_barrier_substrate

    calls = []
    _install_barrier_substrate(monkeypatch, calls)
    ctx = harness.make_ctx()
    _call(ctx)
    state = _state(harness)
    assert plan_review_gate_projection(state, "blocking")["custody_pending"] is True
    collected = collect_before_gate(ctx, state)
    assert [c["reconcile_only"] for c in calls] == [False, True]  # exactly one $0 collection
    verdict = plan_review_gate_projection(collected, "blocking")
    assert verdict["custody_pending"] is False and verdict["status"] == "closed" and verdict["allow"] is True
    assert collect_before_gate(ctx, collected) is collected  # nothing pending: no second send
    assert len(calls) == 2


# ------------------------------------------------------------- verbatim principal directives (P1-6)


def _dry_run_packet(ctx, spec=None):
    request = pr._PlanRequest(goal="Ship the deck", plan="Outline first, then draft each slide.", spec=spec or DECK_SPEC)
    return pr.build_plan_review_packet_for_dry_run(ctx, request)


def test_packet_uses_full_dialogue_and_keeps_acceptance_directives(harness):
    """Full planning dialogue replaces the old 16K display; acceptance keeps its ledger."""
    from ouroboros.review_evidence import build_task_acceptance_evidence
    from ouroboros.owner_mailbox import write_task_message
    from ouroboros.utils import append_jsonl

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    ctx._owner_directives = [{"source": "initial_user", "content": "Build the deck", "msg_id": "t:1"}]
    append_jsonl(harness.drive / "logs" / "chat.jsonl", {"direction": "in", "chat_id": 1,
                 "text": "Build the deck; key sk-abcdefghijklmnopqrstuvwxyz1234"})
    append_jsonl(harness.drive / "logs" / "chat.jsonl", {"direction": "out", "chat_id": 1,
                 "text": "Option A is faster but less flexible. " + "detail " * 6000})
    write_task_message(harness.drive, "Proposal from sibling: use a different format", task_id=ctx.task_id,
                       source_task_id="parent-9", provenance="peer_via_ancestor", relayed_from_task_id="sibling-7")
    packet = _dry_run_packet(ctx)["user_content"]
    assert "## OWNER REQUIREMENTS AND DECISIONS" not in packet
    assert "Option A is faster but less flexible. " + "detail " * 6000 in packet
    assert "sk-abcdefghijklmnopqrstuvwxyz1234" not in packet
    assert "from task sibling-7, relayed by ancestor parent-9" in packet
    rows = build_task_acceptance_evidence(ctx, llm_trace={"tool_calls": []}, drive_root=harness.drive,
                                          task_id=ctx.task_id)["owner_requirements_and_decisions"]
    assert rows[0]["content"] == "Build the deck"
    assert not (harness.drive / "task_results" / "artifacts").exists()  # dry-run stays read-only


def test_task_objective_carries_the_contract_context_redacted(harness):
    from ouroboros.tools.plan_review_runtime import _task_objective

    ctx = harness.make_ctx()
    assert _task_objective(ctx) == "Deliver the thing"
    ctx.task_contract = {"objective": "Deliver the thing",
                         "context": "Owner said: token sk-abcdefghijklmnopqrstuvwxyz1234; audience is the board"}
    text = _task_objective(ctx)
    assert text == "Deliver the thing\n\nContract context: Owner said: token ***REDACTED***; audience is the board"
    assert "## TASK OBJECTIVE\n\n" + text + "\n" in _dry_run_packet(ctx)["user_content"]


# ------------------------------------------------------------- the reviewer's question to the author (P1-7)


def test_reviewer_question_holds_the_wave_until_a_free_disposition_and_its_answer_rides_the_next_cycle(harness):
    """Owner batch 2, Q4=A: a reviewer returns an open question to the author as
    `need_evidence` with the spec id in `breaks` (no locator); the wave holds until the
    author's $0 disposition (accept = answered, reject, defer = deferred openly); the
    answer reaches the reviewers only on the next PAID cycle, disclosed in the next step."""
    question = json.dumps([_finding("q1", "need_evidence", breaks="claim_1", summary="Why exactly 5 slides?")])
    sub = harness.install({"s1": question, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    first = _call(ctx)
    assert _control(first) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    wave = _state(harness)["waves"][-1]
    [finding] = wave["findings"]
    assert finding["class"] == "need_evidence" and finding["breaks"] == "claim_1" and finding["locator"] == ""
    assert _state(harness).get("need_evidence_seen", []) == []  # a question is not a locator the host attaches
    assert "a question addressed to you by spec id" in first and "defer = deferred openly" in first
    answered = pr._handle_plan_task(ctx, review_disposition={
        "review_fingerprint": wave["request_fingerprint"],
        "items": [{"finding_id": "s1:q1", "decision": "accept", "rationale": "The board asked for five."}]})
    assert _control(answered) == {"outcome": "REVIEW_REQUIRED", "closed": True}
    assert len(sub.calls) == 1 and _state(harness)["cycles_paid"] == 1  # $0: no reviewer call, no cycle
    _call(ctx, spec={**DECK_SPEC, "in_scope": ["a 6-slide deck"]})  # the next PAID cycle carries the answer
    assert len(sub.calls) == 2
    user2 = _user_text(sub.calls[1]["request"].messages[1]["content"])
    assert "The board asked for five." in user2 and "s1:q1" in user2
    assert "summaries bounded to 400 chars" in user2  # the carry-forward cut is named where it applies


def test_escalate_is_available_wherever_planning_runs():
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES, CORE_TOOL_NAMES, LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
    )

    assert all("escalate" in names for names in
               (CORE_TOOL_NAMES, LOCAL_READONLY_SUBAGENT_TOOL_NAMES, ACTING_SUBAGENT_TOOL_NAMES))
