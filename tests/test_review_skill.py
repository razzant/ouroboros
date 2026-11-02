"""``skill_review`` → ``review_change(subject="skill")``: one review machinery for a skill payload (owner 21.10, 1A).

Every seat of a skill wave answers the one skill question in the findings-only form over the payload frozen at
its loader hash; the wave is review_change's own (panel by strictness, custody, the paid stamp, $0 reuse, the
ledger record ``surface="skill"``), and ``state/skills/<name>/review.json`` is the run gate's projection of that
record. The seats are fake and offline (the golden physical seam); no model is ever called.
"""
from __future__ import annotations

import json
import pathlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import ouroboros.review_substrate as substrate
from ouroboros import review_ledger
from ouroboros.skill_loader import load_review_state
from ouroboros.tools.registry import ToolContext
from tests import _contributor_packet_shared as shared
from tests.review_pool_rosters import set_review_pool
from tests.test_skill_exec import _build_skill

CLEAN = "[]\nNO_FINDINGS"
HARD = json.dumps([{"item": "no_repo_mutation", "verdict": "FAIL", "severity": "advisory",
                    "reason": "scripts/hello.py commits to ~/Ouroboros/repo"}])
SOFT = json.dumps([{"item": "error_handling", "verdict": "FAIL", "severity": "critical",
                    "reason": "the script swallows every exception"}])
UNKNOWN = json.dumps([{"item": "invented_item", "verdict": "FAIL", "severity": "critical", "reason": "not a checklist item"}])


def _seam(sends, answers):
    def answer(request, slot, actor, *, retry_state, pending_invocation_checkpoint):
        actor.raw_text = answers.get(slot.slot_id, CLEAN)
        return actor

    return shared.golden_physical_seam(sends, answer=answer)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A reviewable script skill in the drive's external bucket, the golden pool and an offline seam."""
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    monkeypatch.delenv("OUROBOROS_REVIEW_MAX_CYCLES", raising=False)
    set_review_pool(monkeypatch, shared.golden_pool())
    system = Path(shared.init_installed_body(tmp_path / "ouroboros")["repo"])
    drive = tmp_path / "ouroboros" / "data"
    for sub in ("logs", "locks", "state"):
        (drive / sub).mkdir(parents=True)
    skill_dir = _build_skill(drive / "skills" / "external", "weather")
    ctx = ToolContext(repo_dir=system, system_repo_dir=system, drive_root=drive, task_id="task-skill")
    sends, answers = [], {}
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, answers))
    return SimpleNamespace(ctx=ctx, drive=drive, skill_dir=skill_dir, sends=sends, answers=answers, system=system,
                           monkeypatch=monkeypatch)


def _review(harness, **kwargs):
    from ouroboros.skill_review import review_skill

    return review_skill(harness.ctx, "weather", **kwargs)


def _state(harness):
    return load_review_state(harness.drive, "weather", skill_type="script", skill_dir=harness.skill_dir)


def test_advisory_one_reviewer_with_a_reason_reviews_the_skill_and_the_same_bytes_reuse_the_record(harness):
    first = _review(harness, reviewers=["t1"], reason="small payload")
    assert (first.status, first.paid, first.reused, first.pending) == ("clean", True, False, False), first.error
    assert [send["slot_id"] for send in harness.sends] == ["t1"] and first.panel["assigned"] == ["t1"]
    record = review_ledger.load_record(harness.drive, first.record_id)
    assert record["surface"] == "skill" and record["subject"]["kind"] == "skill"
    assert record["subject"]["skill"] == "weather" and record["subject"]["content_hash"] == first.content_hash
    assert set(record["subject"]["files"]) == {"SKILL.md", "scripts/hello.py"}
    assert record["brief"]["parts"] == ["skill"] and record["brief"]["checklist"]["severity_policy"]["no_repo_mutation"] == "critical"
    assert record["brief"]["skill"]["name"] == "weather" and record["verdict"]["per_question"] == {"skill": "PASS"}
    state = _state(harness)
    assert (state.status, state.record_id, state.content_hash) == ("clean", first.record_id, first.content_hash)
    assert state.gate_for(first.content_hash, enforcement="advisory")["executable_review"] is True
    assert state.seats and state.seats[0]["seat_id"] == "t1" and state.governance["packet"]["owed_chars"] > 0

    again = _review(harness, reviewers=["t1"], reason="small payload")
    assert again.reused is True and again.record_id == first.record_id and len(harness.sends) == 1


def test_blocking_widens_to_every_ticked_reviewer_and_the_checklist_rule_decides_what_blocks(harness):
    """Blocking seats every ticked reviewer whatever the author named; a hard trust-boundary item blocks even
    when the seat called it advisory (the severity policy is applied before the verdict is read)."""
    harness.monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    harness.answers["t2"] = HARD
    outcome = _review(harness, reviewers=["t1"], reason="small payload")
    assert outcome.status == "blockers" and sorted(send["slot_id"] for send in harness.sends) == ["s1", "t1", "t2"]
    assert outcome.panel["assigned"] == ["t1", "t2", "s1"] and outcome.panel.get("widened_from") == ["t1"]
    [finding] = [f for f in outcome.findings if not f.get("additional")]
    assert (finding["item"], finding["severity"], finding["seat_id"], finding["part"]) == ("no_repo_mutation", "critical", "t2", "skill")
    record = review_ledger.load_record(harness.drive, outcome.record_id)
    assert record["verdict"]["aggregate"] == "FAIL" and record["verdict"]["per_row"]["t2"] == "FAIL"
    state = _state(harness)
    assert state.status == "blockers" and state.gate_for(outcome.content_hash, enforcement="blocking")["executable_review"] is False


@pytest.mark.parametrize("answer, item", [(SOFT, "error_handling"), (UNKNOWN, "invented_item")])
def test_an_advisory_item_and_an_unknown_item_warn_and_never_block(harness, answer, item):
    """Parity with today's aggregation (owner addendum 21.10 ~01:50): a FAIL on one of the four advisory items or
    on a name outside the checklist is a warning, whatever severity the seat wrote."""
    harness.answers["t1"] = answer
    outcome = _review(harness, reviewers=["t1"], reason="small payload")
    assert outcome.status == "warnings", outcome.error
    [finding] = [f for f in outcome.findings if not f.get("additional")]
    assert (finding["item"], finding["severity"]) == (item, "advisory")
    record = review_ledger.load_record(harness.drive, outcome.record_id)
    assert record["verdict"]["aggregate"] == "PASS" and record["verdict"]["advisory_findings"]
    assert _state(harness).gate_for(outcome.content_hash, enforcement="advisory")["executable_review"] is True


def test_an_unreadable_findings_array_is_no_answer_and_a_readable_entry_beside_junk_still_counts(harness):
    """A seat whose array carries no readable entry (no ``item``, no PASS/FAIL ``verdict``) has not answered:
    the one Advisory reviewer's wave stays pending and non-executable, never clean (grok 1 / astra 4); a
    seat with one readable FAIL beside junk counts for what it said."""
    harness.answers["t1"] = json.dumps([{"item": "host_token_handling", "severity": "critical", "reason": "prints the token"}])
    outcome = _review(harness, reviewers=["t1"], reason="small payload")
    assert (outcome.status, outcome.pending, outcome.paid) == ("pending", False, True), outcome.error
    assert "QUORUM_FAILED" in outcome.error or "quorum" in outcome.error.lower(), outcome.error
    record = review_ledger.load_record(harness.drive, outcome.record_id)
    assert record["verdict"]["aggregate"] == "QUORUM_FAILED" and record["rows"][0]["answers"]["skill"]["status"] == "unanswered"
    assert "none of the 1 entries has an item and a PASS/FAIL verdict" in record["rows"][0]["answers"]["skill"]["error"]
    assert _state(harness).gate_for(outcome.content_hash, enforcement="advisory")["executable_review"] is False

    (harness.skill_dir / "SKILL.md").write_text("# weather\n\nRevised.\n", encoding="utf-8")
    harness.answers["t1"] = json.dumps([{"unexpected": "not a finding"},
                                        {"item": "no_repo_mutation", "verdict": "FAIL", "severity": "advisory", "reason": "commits"}])
    outcome = _review(harness, reviewers=["t1"], reason="small payload")
    assert outcome.status == "blockers" and len(harness.sends) == 2, outcome.error
    [finding] = [f for f in outcome.findings if not f.get("additional")]
    assert (finding["item"], finding["severity"]) == ("no_repo_mutation", "critical")


# --- the brief: change-focused re-review (3A), the packet's size, the governance owed -------------------------


def _capturing_seam(sends, answers, briefs):
    """The golden seam plus what each seat was GIVEN: the packet prompt or the retrieving seat's task."""
    def answer(request, slot, actor, *, retry_state, pending_invocation_checkpoint):
        task = (request.slot_session_tasks or {}).get(slot.slot_id) or request.session_task or ""

        def text(content):  # a message's content: a string, or the provider's list of text/cache blocks
            if isinstance(content, list):
                return "\n".join(str(block.get("text") or "") for block in content if isinstance(block, dict))
            return str(content or "")

        briefs[slot.slot_id] = task or "\n".join(text(m.get("content")) for m in (request.messages or []))
        actor.raw_text = answers.get(slot.slot_id, CLEAN)
        return actor

    return shared.golden_physical_seam(sends, answer=answer)


@pytest.fixture
def briefing(harness):
    briefs: dict = {}
    harness.monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _capturing_seam(harness.sends, harness.answers, briefs))
    harness.briefs = briefs
    return harness


def test_a_re_review_after_an_edit_is_change_focused_in_every_mode_and_judges_the_whole_skill(briefing):
    """3A: the brief says what changed since the last verdict — added, modified and removed files with a unified
    diff from the retained earlier copy — in Advisory and in Blocking alike; the verdict is on the whole skill."""
    h = briefing
    (h.skill_dir / "docs.txt").write_text("first notes\n", encoding="utf-8")
    first = _review(h, reviewers=["t1"], reason="small payload")
    assert first.status == "clean", first.error
    (h.skill_dir / "scripts" / "hello.py").write_text("print('hello from skill, revised')\n", encoding="utf-8")
    (h.skill_dir / "scripts" / "util.py").write_text("VALUE = 1\n", encoding="utf-8")
    (h.skill_dir / "docs.txt").unlink()
    h.briefs.clear()
    h.monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    second = _review(h, reviewers=["t1"], reason="small payload")
    assert second.status == "clean" and second.record_id != first.record_id and second.content_hash != first.content_hash
    for seat in ("t1", "t2", "s1"):
        brief = h.briefs[seat]
        assert f"## Changed since the previous review {first.content_hash[:12]} (clean)" in brief, seat
        assert "- added: scripts/util.py" in brief and "- modified: scripts/hello.py" in brief and "- removed: docs.txt" in brief
        assert "### diff scripts/hello.py" in brief and "+print('hello from skill, revised')" in brief
        assert "the verdict is on the whole current skill" in brief
    # Retrieving seats owe the manifest and the changed files; the unchanged file is one read away.
    assert "Unchanged since the previous verdict (one read away, not owed): " not in h.briefs["t1"]
    state = _state(h)
    assert state.content_hash == second.content_hash and set(review_ledger.load_record(h.drive, second.record_id)["subject"]["files"]) == {
        "SKILL.md", "scripts/hello.py", "scripts/util.py"}


def test_the_change_section_says_diff_unavailable_without_the_retained_copy_and_names_a_legacy_verdict(briefing):
    """Two-sided: when the earlier frozen copy is gone the files are still named and the diff is declared
    unavailable; a legacy review.json (no file index) names the earlier hash and says the index is missing."""
    import shutil

    h = briefing
    first = _review(h, reviewers=["t1"], reason="small payload")
    assert first.status == "clean"
    views = h.drive / "state" / "review_checkouts" / "views"
    for view in list(views.glob("inventory-*")):
        shutil.rmtree(view)
    (h.skill_dir / "scripts" / "hello.py").write_text("print('second edition')\n", encoding="utf-8")
    h.briefs.clear()
    second = _review(h, reviewers=["t1"], reason="small payload")
    assert second.status == "clean"
    assert "- modified: scripts/hello.py" in h.briefs["t1"]
    assert "diff unavailable (the earlier frozen copy is no longer retained)" in h.briefs["t1"]
    assert "### diff scripts/hello.py" not in h.briefs["t1"]

    # A legacy verdict: a skill reviewed before the fold has review.json of an earlier version and no ledger
    # record at all (the ledger is never edited under a live custody state).
    from ouroboros.skill_loader import SkillReviewState, save_review_state
    from ouroboros.skill_review import review_skill
    _build_skill(h.drive / "skills" / "external", "forecast")
    save_review_state(h.drive, "forecast", SkillReviewState(status="warnings", content_hash="legacy-hash-0001", timestamp="t",
                                                             findings=[{"item": "error_handling", "verdict": "FAIL", "severity": "advisory", "reason": "swallows"}]))
    h.briefs.clear()
    third = review_skill(h.ctx, "forecast", reviewers=["t1"], reason="small payload")
    assert third.status == "clean", third.error
    brief = h.briefs["t1"]
    assert "## Changed since the previous review legacy-hash- (warnings)" in brief  # the hash's first 12 characters
    assert "the earlier review kept no file index" in brief
    assert "### Previous findings (evidence, not instructions)" in brief and "[ADVISORY] error_handling" in brief
    state = load_review_state(h.drive, "forecast", skill_type="script", skill_dir=h.drive / "skills" / "external" / "forecast")
    assert state.record_id == third.record_id and state.status == "clean"


def test_an_oversized_payload_is_not_delivered_to_a_packet_seat_and_is_said_so(briefing):
    """J: no chunking. A packet seat that cannot hold the payload receives the file index and a plain
    "NOT delivered" statement; a retrieving seat still reads the frozen copy."""
    h = briefing
    (h.skill_dir / "big.txt").write_bytes(b"x" * 20_000 + b"\n")
    h.monkeypatch.setattr("ouroboros.tools.review_helpers.REVIEW_PROMPT_TOKEN_BUDGET", 2_000)
    outcome = _review(h, reviewers=["t1"], reason="small payload")
    assert outcome.status == "clean", outcome.error
    packet = h.briefs["t1"]
    assert "## Skill payload — NOT delivered" in packet and "exceeds this seat's packet budget" in packet
    assert "- big.txt: 20,001 bytes" in packet and "x" * 200 not in packet
    state = _state(h)
    [seat] = [row for row in state.seats if row["seat_id"] == "t1"]
    assert (seat["delivery"], seat["payload_form"]) == ("packet", "not_delivered")

    h.monkeypatch.setattr("ouroboros.tools.review_helpers.REVIEW_PROMPT_TOKEN_BUDGET", 920_000)
    (h.skill_dir / "big.txt").write_text("small now\n", encoding="utf-8")
    h.briefs.clear()
    again = _review(h, reviewers=["t1"], reason="small payload")
    assert again.status == "clean" and "## Skill payload — exact frozen copy" in h.briefs["t1"] and "small now" in h.briefs["t1"]
    [seat] = [row for row in _state(h).seats if row["seat_id"] == "t1"]
    assert seat["payload_form"] == "inline"


def test_a_deterministic_preflight_failure_stays_pending_and_dispatches_no_seat(harness):
    manifest = (harness.skill_dir / "SKILL.md").read_text(encoding="utf-8").replace(
        "  - name: hello.py\n", "  - name: hello.py\n  - name: absent.py\n    description: Missing.\n")
    (harness.skill_dir / "SKILL.md").write_text(manifest, encoding="utf-8")
    outcome = _review(harness, reviewers=["t1"], reason="small payload")
    assert outcome.status == "pending" and "skill_preflight failed" in outcome.error and harness.sends == []
    state = _state(harness)
    assert state.status == "pending" and state.findings[0]["item"] == "skill_preflight"
    assert state.gate_for(outcome.content_hash, enforcement="advisory")["executable_review"] is False
    assert state.gate_for(outcome.content_hash, enforcement="blocking")["executable_review"] is False


# --- the paid ceiling: the manual lane per content version, the tool lane per root task (owner Q23) ---------


def test_the_manual_lane_meets_the_ceiling_per_content_version(harness):
    """A Skills-page review has no task: the shared ceiling counts paid skill waves of these exact bytes; a
    byte-identical ask reuses the record (no wave bought), a rebuttal buys a new wave until the ceiling, and a
    revised payload starts a fresh count."""
    h = harness
    h.ctx.task_id = ""  # the manual lane (the Skills page, a Hub install)
    h.monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    first = _review(h, reviewers=["t1"], reason="small payload")
    assert first.status == "clean" and first.paid and len(h.sends) == 1
    same = _review(h, reviewers=["t1"], reason="small payload")
    assert same.reused and len(h.sends) == 1
    rebutted = _review(h, reviewers=["t1"], reason="small payload", review_rebuttal="The finding cites a helper this skill does not call.")
    assert rebutted.status == "pending" and rebutted.dispatch_refusal_code == "LEGACY_BLOCKED" and len(h.sends) == 1
    assert "REVIEW_CYCLES_EXHAUSTED" in rebutted.error and "manual lane of skill weather" in rebutted.error
    assert _state(h).status == "clean", "the earlier verdict for these bytes stands"
    (h.skill_dir / "scripts" / "hello.py").write_text("print('revised')\n", encoding="utf-8")
    revised = _review(h, reviewers=["t1"], reason="small payload")
    assert revised.status == "clean" and revised.paid and len(h.sends) == 2


def test_the_tool_lane_shares_one_allowance_per_root_task_across_skills(harness):
    """A task's skill reviews (``task:<root>:<skill>`` groups) spend the ROOT's allowance of paid skill waves,
    whichever skill is reviewed; change and result reviews never use it up (their own tool names)."""
    h = harness
    h.monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    _build_skill(h.drive / "skills" / "external", "forecast")
    h.ctx._skill_review_group_id = "task:task-skill:weather"
    first = _review(h, reviewers=["t1"], reason="small payload")
    assert first.status == "clean" and first.paid
    h.ctx._skill_review_group_id = "task:task-skill:forecast"
    from ouroboros.skill_review import review_skill

    second = review_skill(h.ctx, "forecast", reviewers=["t1"], reason="small payload")
    assert second.dispatch_refusal_code == "LEGACY_BLOCKED" and second.status == "pending" and len(h.sends) == 1
    h.ctx._skill_review_group_id = "task:another-root:forecast"
    third = review_skill(h.ctx, "forecast", reviewers=["t1"], reason="small payload")
    assert third.status == "clean" and third.paid and len(h.sends) == 2


# --- Cyber Pro: the panel composes freely and the review is evidence -----------------------------------------


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])
def test_a_narrowed_panel_without_a_reason_is_refused_outside_cyber_pro(harness, mode):
    from ouroboros import config

    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline(mode)
    try:
        outcome = _review(harness, reviewers=["t1"])
        if mode == "cyber_pro":
            assert outcome.status == "clean" and outcome.panel["assigned"] == ["t1"] and len(harness.sends) == 1
            assert _state(harness).gate_for(outcome.content_hash, enforcement="blocking")["executable_review"] is True
        else:
            assert outcome.dispatch_refusal_code == "TOOL_ARG_ERROR" and harness.sends == [] and outcome.status == "pending"
    finally:
        config.reset_runtime_mode_baseline_for_tests()


# --- the lifecycle surfaces: the Skills page, a Hub install, the owner's Skip, the author's finish -----------


def _lifecycle(harness, **kwargs):
    """The lifecycle runner as the Skills page and the tool call it (the lane job, the composed panel)."""
    from ouroboros.skill_review_runner import run_skill_review_lifecycle_blocking
    from tests.test_skill_review_runner import _reset_queue

    _reset_queue()
    try:
        return run_skill_review_lifecycle_blocking(harness.ctx, "weather", **kwargs)
    finally:
        _reset_queue()


def test_the_skills_page_review_is_the_same_wave_under_the_owner_panel_and_the_card_reads_the_record(harness):
    from ouroboros.skill_review_runner import skill_review_ui_projection

    payload = _lifecycle(harness, source="skills", reviewers=["t1"], reason="small payload")
    assert (payload["status"], payload["executable_review"], payload["paid"]) == ("clean", True, True), payload["error"]
    assert payload["record_id"] and payload["panel"]["chosen_by"] == "owner" and payload["panel"]["assigned"] == ["t1"]
    assert [send["slot_id"] for send in harness.sends] == ["t1"]
    record = review_ledger.load_record(harness.drive, payload["record_id"])
    facts = record["brief"]["skill"]
    assert (facts["source"], facts["job_id"], facts["group_id"]) == ("skills", payload["job_id"], "manual:weather")
    assert _state(harness).record_id == payload["record_id"]
    projection = skill_review_ui_projection(harness.drive, "weather")
    [row] = projection["history"]
    assert (row["record_id"], row["status"], row["job_id"], row["source"]) == (payload["record_id"], "clean", payload["job_id"], "skills")
    assert row["panel"]["assigned"] == ["t1"] and row["executions"]


def test_a_reused_lifecycle_review_names_its_record_in_the_chat_row_and_the_card_reads_the_record(harness):
    """astra R1-5: a second lifecycle review of unchanged bytes reuses the retained record under a FRESH job id.
    The chat row carries both, the history projection passes the record id through, and the detail reader
    resolves it; the second job id alone resolves nothing (which is why the card asks by record)."""
    from ouroboros.gateway.extensions import _skill_review_history_detail_sync
    from ouroboros.gateway.history import _collect_chat_rows

    h = harness
    first = _lifecycle(h, source="skills", reviewers=["t1"], reason="small payload")
    second = _lifecycle(h, source="skills", reviewers=["t1"], reason="small payload")
    assert first["status"] == "clean" and second["reused"] is True and len(h.sends) == 1, second["error"]
    assert second["record_id"] == first["record_id"] and second["job_id"] != first["job_id"]
    rows, _quota = _collect_chat_rows(h.drive / "logs" / "chat.jsonl", h.drive / "archive", 10, lambda _c, _e: True, {})
    reviews = [row for row in rows if row.get("system_type") == "skill_review"]
    assert [(row["job_id"], row["record_id"]) for row in reviews] == [
        (first["job_id"], first["record_id"]), (second["job_id"], first["record_id"])]
    detail = _skill_review_history_detail_sync(h.drive, "weather", reviews[1]["record_id"])
    assert detail.get("status") == "clean" and detail["job_status"] == "completed" and detail["markdown"], detail
    assert _skill_review_history_detail_sync(h.drive, "weather", second["job_id"]) == {
        "error": "no review history for skill", "status_code": 404}


def test_a_hub_install_reviews_in_the_background_and_settles_the_kept_update_after_the_wave(harness):
    """``start_review`` (a Hub install, ``check=review``): the same wave on a background thread under the
    marketplace context; the kept update snapshot is settled only once the wave is over, with the digest
    the install record named."""
    import threading

    from ouroboros import skill_source_install as install

    settled = []
    harness.monkeypatch.setattr(install, "read_install_record", lambda drive, name: {"release": {"digest": "sha256:feed"}})
    harness.monkeypatch.setattr(install, "settle_pending_update",
                                lambda drive, name, **kw: settled.append((name, kw, _state(harness).status)) or {"status": "settled"})
    backup = install.update_backup_path(harness.drive, "weather")
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text("{}", encoding="utf-8")
    from tests.test_skill_review_runner import _reset_queue

    _reset_queue()
    try:
        install.start_review(harness.drive, "weather", repo_dir=harness.system, reviewers=["t1"], reason="small payload")
        [thread] = [t for t in threading.enumerate() if t.name == "skill-install-review-weather"]
        thread.join(120)
        assert not thread.is_alive()
    finally:
        _reset_queue()
    state = _state(harness)
    assert state.status == "clean" and state.record_id and state.panel["chosen_by"] == "owner"
    assert [send["slot_id"] for send in harness.sends] == ["t1"]
    # The lifecycle settles the kept update once the verdict is in (the job's own digest), and the install's
    # finally-clause settles again with the record's digest while the snapshot is still kept: both after the verdict.
    assert settled and all(status == "clean" for _name, _kw, status in settled)
    assert settled[-1] == ("weather", {"expected_digest": "sha256:feed"}, "clean")
    record = review_ledger.load_record(harness.drive, state.record_id)
    assert record["brief"]["skill"]["source"] == "skills" and record["task_id"] == "marketplace_install"


def test_the_owner_skip_attests_the_bytes_without_a_wave_and_publication_still_wants_the_full_review(harness):
    from ouroboros.skill_owner_attestation import review_skill_owner_attest
    from ouroboros.skill_publish_eligibility import submit_hub_eligibility

    payload = _lifecycle(harness, source="skills", owner_attestation=True, review_impl=review_skill_owner_attest)
    assert (payload["status"], payload["review_profile"]) == ("clean", "owner_attested"), payload["error"]
    assert harness.sends == [] and review_ledger.recent_records(harness.drive, "", surface="skill") == []
    state = _state(harness)
    assert state.review_profile == "owner_attested" and state.record_id == ""
    assert (harness.drive / "state" / "skills" / "weather" / "owner_attestation.json").is_file()
    assert state.gate_for(state.content_hash, enforcement="blocking")["executable_review"] is True
    eligibility = submit_hub_eligibility(source="external", review_status=state.status, review_profile=state.review_profile,
                                         github_token_configured=True)
    assert eligibility["reason"] == "Owner-attested skills need a full LLM review before publication"


@pytest.mark.parametrize("enforcement", ["advisory", "blocking"])
def test_an_author_finish_after_an_edit_binds_the_new_bytes_in_advisory_and_is_refused_in_blocking(harness, enforcement):
    """The author answers the reviewer's warning by editing and finishing: Advisory records the acceptance for
    the NEW bytes against the retained reviewer evidence (no second panel); Blocking refuses the finish."""
    from ouroboros.tool_access_types import ResolvedResourceBinding
    from ouroboros.tools import skill_exec

    h = harness
    h.answers["t1"] = SOFT
    first = _review(h, reviewers=["t1"], reason="small payload")
    assert first.status == "warnings"
    (h.skill_dir / "scripts" / "hello.py").write_text("print('handled')\n", encoding="utf-8")
    h.monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    binding = ResolvedResourceBinding(profile="self_modification", root="skill_payload", operation="review",
                                      base_path=h.skill_dir, target_path=h.skill_dir, source="test",
                                      skill_name="weather", state_drive_root=h.drive)
    finished = skill_exec._author_finish_existing_skill_review(
        h.ctx, binding, "weather", disposition="accepted", rationale="The exception is handled now; verified locally.")
    assert len(h.sends) == 1, "a finish never buys a panel"
    state = _state(h)
    if enforcement == "blocking":
        assert "explicit author finish requires advisory enforcement" in finished["error"]
        assert not state.author_disposition and state.content_hash == first.content_hash
        return
    assert "error" not in finished and finished["record_id"] == first.record_id
    assert finished["content_hash"] != first.content_hash and finished["reviewed_content_hash"] == first.content_hash
    assert state.author_disposition["disposition"] == "accepted" and state.author_disposition["subject_hash"] == finished["content_hash"]
    assert [f["item"] for f in state.findings] == ["error_handling"], "the reviewer's evidence stays"
    gate = state.gate_for(finished["content_hash"], enforcement="advisory")
    assert gate["executable_review"] is True and gate["blocking_reason"] == "author_accepted_advisory"


def test_a_wave_without_a_verdict_keeps_the_owner_skip_and_a_substantive_wave_replaces_it(harness):
    """The owner's Skip admits these exact bytes; a later wave that forms no verdict (quorum failed) leaves it
    standing and names the pending record, a PASS/FAIL wave replaces it (opus R1-1, both directions)."""
    from ouroboros.skill_owner_attestation import review_skill_owner_attest

    h = harness
    payload = _lifecycle(h, source="skills", owner_attestation=True, review_impl=review_skill_owner_attest)
    assert (payload["status"], payload["review_profile"]) == ("clean", "owner_attested"), payload["error"]
    h.answers["t1"] = "I cannot review this."
    silent = _review(h, reviewers=["t1"], reason="small payload")
    assert silent.status == "pending" and "the owner's attestation for these bytes stands" in silent.error, silent.error
    assert f"record {silent.record_id} formed no verdict" in silent.error
    state = _state(h)
    assert (state.status, state.review_profile, state.record_id) == ("clean", "owner_attested", "")
    assert state.gate_for(silent.content_hash, enforcement="blocking")["executable_review"] is True
    h.answers["t1"] = HARD
    (h.skill_dir / "SKILL.md").write_text("# weather\n\nRevised.\n", encoding="utf-8")
    judged = _review(h, reviewers=["t1"], reason="small payload")
    assert judged.status == "blockers" and _state(h).review_profile == "" and _state(h).status == "blockers"


def test_a_wave_without_a_verdict_keeps_the_author_acceptance_and_a_substantive_wave_replaces_it(harness):
    """Advisory: the author accepted edited bytes against the reviewer's warning; a wave on those bytes that forms
    no verdict keeps the acceptance (the gate stays executable), a clean wave replaces it with its verdict."""
    from ouroboros.tool_access_types import ResolvedResourceBinding
    from ouroboros.tools import skill_exec

    h = harness
    h.monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "3")  # three paid waves of this root task below
    h.answers["t1"] = SOFT
    first = _review(h, reviewers=["t1"], reason="small payload")
    assert first.status == "warnings"
    (h.skill_dir / "scripts" / "hello.py").write_text("print('handled')\n", encoding="utf-8")
    binding = ResolvedResourceBinding(profile="self_modification", root="skill_payload", operation="review",
                                      base_path=h.skill_dir, target_path=h.skill_dir, source="test",
                                      skill_name="weather", state_drive_root=h.drive)
    finished = skill_exec._author_finish_existing_skill_review(
        h.ctx, binding, "weather", disposition="accepted", rationale="The exception is handled now; verified locally.")
    assert "error" not in finished
    accepted_hash = finished["content_hash"]
    h.answers["t1"] = "I cannot review this."
    silent = _review(h, reviewers=["t1"], reason="small payload")
    assert silent.status == "pending" and "the author's acceptance for these bytes stands" in silent.error, silent.error
    state = _state(h)
    assert state.author_disposition["subject_hash"] == accepted_hash and state.content_hash == first.content_hash
    assert state.gate_for(accepted_hash, enforcement="advisory")["blocking_reason"] == "author_accepted_advisory"
    # The same bytes asked again in the same round would replay the settled seat; a rebuttal opens a new round.
    h.answers["t1"] = CLEAN
    judged = _review(h, reviewers=["t1"], reason="small payload", review_rebuttal="The exception is handled; look again.")
    assert judged.status == "clean" and len(h.sends) == 3, (judged.status, judged.error, len(h.sends))
    state = _state(h)
    assert (state.status, state.content_hash, state.author_disposition) == ("clean", accepted_hash, {})


# --- a wave the author really leaves: a seat blocks until an addressed message ends the author's wait ---------


def _until(predicate, timeout=90.0):
    import time

    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.05)


@pytest.fixture
def leaving(harness):
    """The ``skill_review`` tool's author under a real model-wait scope with a real owner mailbox; seat t1
    answers only once ``gate`` is set, so the author can leave on an addressed message while the seat still
    runs. The lifecycle tail (dependencies, turn-on, extension reload, update settlement) is spied."""
    import queue
    import threading

    from ouroboros import model_wait, review_pause
    from ouroboros import skill_review_runner as runner
    from ouroboros import skill_source_install as install
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import skill_exec
    from tests.test_skill_review_runner import _reset_queue

    h = harness
    h.monkeypatch.setattr(review_pause, "DETACH_POLL_SEC", 0.05)
    h.monkeypatch.setattr(skill_exec, "_skill_tool_preflight", lambda *a, **k: "")
    write_task_result(h.drive, "task-skill", "running", chat_id=7)
    h.ctx.owner_wait_callback = lambda *_a, **_k: "parked"  # an interactive author's continuation owner
    h.ctx.task_attempt, h.ctx.current_task_type, h.ctx._loop_mailbox_seen_ids = 1, "task", set()
    h.ctx.task_lifecycle_bound = True
    gate, sent = threading.Event(), threading.Event()

    def answer(request, slot, actor, *, retry_state, pending_invocation_checkpoint):
        sent.set()
        assert gate.wait(120), "the seat was never released"
        actor.raw_text = h.answers.get(slot.slot_id, CLEAN)
        return actor

    h.monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(h.sends, answer=answer))
    tail = {"deps": [], "turn_on": [], "reload": [], "settle": []}
    h.monkeypatch.setattr("ouroboros.skill_dependencies.ensure_skill_dependencies",
                          lambda *a, **k: tail["deps"].append(a[1].name) or ("not_required", ""))
    h.monkeypatch.setattr("ouroboros.skill_lifecycle_actions.reconcile_turn_on_when_ready",
                          lambda *a, **k: tail["turn_on"].append(a[1]) or {"action": "no_change"})
    h.monkeypatch.setattr(runner, "_reconcile_extension_payload", lambda *a, **k: tail["reload"].append(a[1]) or {})
    h.monkeypatch.setattr(install, "settle_pending_update",
                          lambda drive, name, **kw: tail["settle"].append((name, kw)) or {"status": "accepted"})
    backup = install.update_backup_path(h.drive, "weather")
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text("{}", encoding="utf-8")
    threads = []

    def call(**kwargs):
        box = {}

        def run():
            with model_wait.task_model_wait_scope(task={"id": "task-skill", "chat_id": 7, "_attempt": 1},
                                                  drive_root=h.drive, event_queue=queue.Queue(), worker_slot_held=False):
                try:
                    box["text"] = skill_exec._handle_review_skill(h.ctx, skill="weather", reviewers=["t1"],
                                                                  reason="small payload", **kwargs)
                except BaseException as exc:  # surfaced by the test, never lost in the thread
                    box["error"] = exc

        thread = threading.Thread(target=run, name="skill-review-author")
        threads.append(thread)
        thread.start()
        return thread, box

    def join(thread, box, timeout=120):
        thread.join(timeout)
        assert not thread.is_alive(), "the author call did not finish"
        assert "error" not in box, box.get("error")
        return box["text"]

    _reset_queue()
    try:
        yield SimpleNamespace(h=h, gate=gate, sent=sent, call=call, join=join, tail=tail, backup=backup)
    finally:
        gate.set()
        for thread in threads:
            thread.join(120)
        _reset_queue()


def _physical_sends(h):
    """The seats physically sent (a rejoin's reconcile-only pass through the seam is not a send)."""
    return [send for send in h.sends if not send.get("reconcile_only")]


def _settled_record(h, record_id):
    record = review_ledger.load_record(h.drive, record_id)
    return record if record and record.get("state") == "settled" else None


def _leave(L):
    """Start the tool call, let the seat start, write an addressed owner message: the author returns pending."""
    from ouroboros.owner_mailbox import write_owner_message

    thread, box = L.call()
    assert L.sent.wait(60), "the seat never started"
    write_owner_message(L.h.drive, "How is the review going?", "task-skill")
    text = L.join(thread, box, timeout=60)
    assert "The review is still running" in text and len(_physical_sends(L.h)) == 1, text
    return json.loads(text.rsplit("Review reference: ", 1)[1])


def _read_mail(L):
    from ouroboros.owner_mailbox import drain_owner_entries

    drain_owner_entries(L.h.drive, "task-skill", L.h.ctx._loop_mailbox_seen_ids, 1)


def _settled_notices(L):
    from ouroboros.owner_mailbox import drain_owner_entries

    return [entry for entry in drain_owner_entries(L.h.drive, "task-skill", set(), 1, include_acknowledged=True)
            if str(entry.get("msg_id") or "").startswith("skill-review-settled:")]


def _finish_binding(h):
    from ouroboros.tool_access_types import ResolvedResourceBinding

    return ResolvedResourceBinding(profile="self_modification", root="skill_payload", operation="review",
                                   base_path=h.skill_dir, target_path=h.skill_dir, source="test",
                                   skill_name="weather", state_drive_root=h.drive)


@pytest.mark.parametrize("answer, status", [(CLEAN, "clean"), (HARD, "blockers")])
def test_a_left_wave_finalizes_nothing_and_the_reissue_collects_it_at_zero(leaving, answer, status):
    """The author leaves on an addressed message while the seat runs: the call returns pending with one send,
    the lifecycle job ends as a non-failure and ran no tail (no dependencies, turn-on, reload or update
    settlement; the kept update stays pending), review.json is untouched, and the author finish with that
    reference is refused while the record is open. Released, the wave settles on its own (the record keeps
    the frozen severity policy: a hard item called advisory still blocks) and the notice names the re-issue;
    the re-issued call collects at $0, projects the record, runs the tail once and settles the update."""
    from ouroboros.skill_review_runner import _read_review_job, review_job_state_path
    from ouroboros.tools import skill_exec

    L, h = leaving, leaving.h
    h.answers["t1"] = answer
    reference = _leave(L)
    job = _read_review_job(review_job_state_path(h.drive, "weather"))
    assert (job["status"], job["lifecycle_status"], job["review_status"]) == ("completed", "succeeded", "pending"), job
    assert job["record_id"] == reference["record_id"] and "still running" in job["error"]
    assert review_ledger.load_record(h.drive, reference["record_id"])["state"] == "pending"
    assert _state(h).status == "pending" and _state(h).record_id == "", "nothing projected for a left wave"
    assert L.tail == {"deps": [], "turn_on": [], "reload": [], "settle": []}, "no tail on a left wave"
    assert L.backup.exists(), "the kept update waits for the collection"
    refused = skill_exec._author_finish_existing_skill_review(
        h.ctx, _finish_binding(h), "weather", disposition="accepted", rationale="Looks fine to me.",
        review_reference=reference)
    assert "A review still running is not unavailable" in refused["error"] and not _state(h).author_disposition
    L.gate.set()
    record = _until(lambda: _settled_record(h, reference["record_id"]))
    assert record["verdict"]["aggregate"] == {"clean": "PASS", "blockers": "FAIL"}[status]
    [notice] = _until(lambda: _settled_notices(L))
    assert "[Review settled]" in notice["text"] and "review_change(review_reference=the original reference)" in notice["text"]
    assert _state(h).status == "pending", "the settled record reaches review.json on the next call (D6)"
    _read_mail(L)
    thread, box = L.call()
    text = L.join(thread, box)
    assert "The review is still running" not in text and len(_physical_sends(h)) == 1, text
    state = _state(h)
    assert (state.status, state.record_id) == (status, reference["record_id"])
    assert [len(calls) for calls in L.tail.values()] == [1, 1, 1, 1], L.tail
    assert L.tail["settle"] == [("weather", {"expected_digest": reference["content_hash"], "landing_failed": False})]
    collected = json.loads(text.rsplit("Review reference: ", 1)[1])
    assert collected["record_id"] == reference["record_id"] and collected["job_id"] != reference["job_id"]


def test_a_reissue_while_the_left_wave_runs_rejoins_it_and_reads_the_frozen_severity_policy(leaving):
    """Re-issuing the same call before the wave settles rejoins the live wave (no second send) and waits; the
    rejoin's plan carries the record's severity policy (astra R1-2), so the hard item the seat called
    advisory blocks exactly as on the fresh and the background-settled paths."""
    L, h = leaving, leaving.h
    h.answers["t1"] = HARD
    reference = _leave(L)
    _read_mail(L)
    thread, box = L.call()
    thread.join(1.0)
    assert thread.is_alive(), "the rejoin waits for the running wave instead of returning at once"
    L.gate.set()
    text = L.join(thread, box)
    assert len(_physical_sends(h)) == 1 and "The review is still running" not in text, text
    state = _state(h)
    assert (state.status, state.record_id) == ("blockers", reference["record_id"])
    [finding] = [f for f in state.findings if not f.get("additional")]
    assert (finding["item"], finding["severity"]) == ("no_repo_mutation", "critical")
    record = review_ledger.load_record(h.drive, reference["record_id"])
    assert record["verdict"]["aggregate"] == "FAIL" and record["rows"][0]["answers"]["skill"]["critical"] == 1


def test_a_left_wave_that_settles_without_a_verdict_admits_the_author_finish_only_after_it_settled(leaving):
    """The other direction of the running-is-not-unavailable rule: refused while the record is open, the same
    reference admits an Advisory author finish once the wave settled without a verdict (quorum failed)."""
    from ouroboros.tools import skill_exec

    L, h = leaving, leaving.h
    h.answers["t1"] = "I cannot review this."
    reference = _leave(L)
    refused = skill_exec._author_finish_existing_skill_review(
        h.ctx, _finish_binding(h), "weather", disposition="accepted", rationale="Looks fine to me.",
        review_reference=reference)
    assert "A review still running is not unavailable" in refused["error"]
    L.gate.set()
    record = _until(lambda: _settled_record(h, reference["record_id"]))
    assert record["verdict"]["aggregate"] == "QUORUM_FAILED"
    finished = skill_exec._author_finish_existing_skill_review(
        h.ctx, _finish_binding(h), "weather", disposition="accepted", rationale="No reviewer answered; I checked the script myself.",
        review_reference=reference)
    assert "error" not in finished, finished
    state = _state(h)
    assert state.author_disposition["disposition"] == "accepted" and len(_physical_sends(h)) == 1
    assert state.gate_for(reference["content_hash"], enforcement="advisory")["blocking_reason"] == "author_accepted_advisory"


# --- the ledger readers: this skill's rows, the earlier verdict, what the brief cuts ---------------------------


def test_skill_record_rows_count_this_skills_rows_and_parse_the_index_once_per_stamp(harness):
    """opus R1-3/4, grok 2, subagent 1: every per-skill lookup is served from ONE parse of the ledger index per
    index stamp (N skills → one read; a written row → one more), and ``limit`` counts THIS skill's rows however
    many other skills were reviewed since (never a window over every skill's traffic)."""
    from ouroboros.tools import review_skill as subject

    h = harness
    outcome = _review(h, reviewers=["t1"], reason="small payload")
    index = review_ledger.index_path(h.drive)

    def other(n, record_id):
        return json.dumps({"record_id": record_id, "surface": "skill", "state": "settled", "revision": 0,
                           "ts": f"2026-10-21T13:{n % 60:02d}:00+00:00", "subject": {"skill": f"other-{n}"},
                           "verdict": {"aggregate": "PASS"}}) + "\n"

    with index.open("a", encoding="utf-8") as handle:
        handle.writelines(other(n, f"r{n:04d}") for n in range(60))
    parses = []
    real = review_ledger.surface_records
    h.monkeypatch.setattr(review_ledger, "surface_records", lambda drive, surface: parses.append(surface) or real(drive, surface))
    subject._SKILL_ROWS_MEMO.clear()
    assert [row["record_id"] for row in subject.skill_record_rows(h.drive, "weather", limit=1)] == [outcome.record_id]
    for n in range(60):
        assert [row["record_id"] for row in subject.skill_record_rows(h.drive, f"other-{n}", limit=5)] == [f"r{n:04d}"]
    assert subject.skill_record_rows(h.drive, "nobody") == []
    assert parses == ["skill"], "one index parse serves every skill's lookup"
    with index.open("a", encoding="utf-8") as handle:  # a written row: a new stamp, one more parse
        handle.write(other(0, "r9999"))
    assert [row["record_id"] for row in subject.skill_record_rows(h.drive, "other-0", limit=5)] == ["r9999", "r0000"]
    assert parses == ["skill", "skill"]


def test_the_earlier_verdict_is_the_record_review_json_names_and_no_index_is_claimed_missing_while_it_exists(harness):
    """subagent 1, grok 2, opus 4: the earlier verdict is first the record the projection names (exact, one
    read), so 3A holds however many other skills were reviewed since; "kept no file index" is said only of a
    verdict that has none (a legacy review.json)."""
    from ouroboros.skill_loader import compute_content_hash, find_skill, save_review_state
    from ouroboros.tools import review_skill as subject

    h = harness
    first = _review(h, reviewers=["t1"], reason="small payload")
    h.monkeypatch.setattr(subject, "settled_skill_records", lambda *a, **k: [])  # as if every window lost the row
    skill = find_skill(h.drive, "weather", repo_path="")
    earlier = subject.earlier_verdict(h.drive, skill)
    assert earlier["record_id"] == first.record_id and set(earlier["files"]) == {"SKILL.md", "scripts/hello.py"}
    (h.skill_dir / "SKILL.md").write_text("# weather\n\nRevised.\n", encoding="utf-8")
    skill = find_skill(h.drive, "weather", repo_path="")
    current = subject.skill_subject(h.ctx, skill, compute_content_hash(skill.skill_dir, manifest_entry=skill.manifest.entry,
                                                                        manifest_scripts=skill.manifest.scripts),
                                    drive_root=h.drive, governance_root=h.system)
    change = subject.change_since(h.drive, skill, current)
    assert change["modified"] == ["SKILL.md"] and change["index_missing"] == "" and change["record_id"] == first.record_id
    state = _state(h)
    state.record_id = ""  # a verdict of an earlier version: review.json without a record
    save_review_state(h.drive, "weather", state)
    legacy = subject.change_since(h.drive, find_skill(h.drive, "weather", repo_path=""), current)
    assert legacy["files"] is None and "kept no file index" in legacy["index_missing"] and legacy["record_id"] == ""


def test_the_earlier_verdict_survives_waves_that_formed_no_verdict(harness):
    """opus R2-1: waves that formed no verdict (quorum failed, not dispatched) may sit between two verdicts;
    the earlier verdict is searched over this skill's rows, never a window of its newest two, so 3A's change
    focus does not lapse after two of them."""
    from ouroboros.skill_loader import save_review_state
    from ouroboros.skill_loader import find_skill
    from ouroboros.tools import review_skill as subject

    h = harness
    first = _review(h, reviewers=["t1"], reason="small payload")
    index = review_ledger.index_path(h.drive)
    with index.open("a", encoding="utf-8") as handle:
        for n in range(3):
            handle.write(json.dumps({"record_id": f"no-verdict-{n}", "surface": "skill", "state": "settled",
                                     "revision": 0, "ts": f"2026-10-21T23:5{n}:00+00:00",
                                     "subject": {"skill": "weather"}, "verdict": {"aggregate": "QUORUM_FAILED"}}) + "\n")
    subject._SKILL_ROWS_MEMO.clear()
    assert [r["record_id"] for r in subject.settled_skill_records(h.drive, "weather", limit=1)] == [first.record_id]
    state = _state(h)
    state.record_id = ""  # the projection names no record: the search over this skill's rows must find it
    save_review_state(h.drive, "weather", state)
    earlier = subject.earlier_verdict(h.drive, find_skill(h.drive, "weather", repo_path=""))
    assert earlier["record_id"] == first.record_id and earlier["files"] is not None


def test_the_authors_answer_is_judged_against_the_bytes_never_binding():
    """opus R2-2: the author's answer to the previous review is evidence the seats judge against the current
    bytes (the shared rebuttal rule: valid → drop, not valid → keep and say why); neither the checklist the seats
    receive nor the brief makes the author's own word binding (no review bypass for self-authored skills)."""
    from ouroboros.tools.review_skill import _change_section, skill_rubric

    rubric = skill_rubric()
    assert "judge it against the current" in rubric and "binding guidance" not in rubric
    change = {"hash": "a" * 64, "verdict": "blockers", "same_bytes": True, "added": [], "modified": [], "removed": [],
              "diffs": {}, "diff_unavailable": "", "index_missing": "", "record_id": "rec-1",
              "findings": [{"item": "path_confinement", "severity": "critical", "reason": "writes to ~/Documents",
                            "model": "m"}]}
    text = _change_section(change)
    assert "Judge the author's answer against the files" in text and "keep one the files still earn" in text
    assert "without new evidence" not in text


def test_every_seat_reads_the_declared_tool_effect_rule_in_the_stable_half():
    """#2141 (places): the reviewer compares each manifest ``tools:`` effect with its implementation; after the
    fold that rule rides the brief's stable half (byte-identical across skills and rounds), not a deleted prompt
    builder."""
    from ouroboros.tools import review_skill as subject

    assert "Place connectors must declare every tool" in subject._TOOL_EFFECT_RULE
    source = pathlib.Path(subject.__file__).read_text(encoding="utf-8")
    assert "_PREAMBLE, _CHECKLIST_HEADING, rubric, _TOOL_EFFECT_RULE," in source


def test_the_change_section_names_what_it_cut_and_where_the_whole_is():
    """astra R1-6: the brief's earlier findings keep a bound, and every cut is named (N more findings not shown;
    reason truncated) with the record that keeps the whole; nothing is said when nothing was cut."""
    from ouroboros.tools.review_skill import MAX_PREVIOUS_FINDINGS, MAX_PREVIOUS_REASON_CHARS, _change_section

    findings = [{"item": "bug_hunting", "severity": "advisory", "reason": f"finding {n}", "model": "m"}
                for n in range(MAX_PREVIOUS_FINDINGS + 2)]
    findings[0]["reason"] = "x" * (MAX_PREVIOUS_REASON_CHARS + 100)
    change = {"hash": "a" * 64, "verdict": "warnings", "same_bytes": True, "added": [], "modified": [], "removed": [],
              "diffs": {}, "diff_unavailable": "", "index_missing": "", "record_id": "rec-123", "findings": findings}
    text = _change_section(change)
    assert "(100 more chars; reason truncated)" in text and "- … 2 more findings not shown" in text
    assert "(Cut for size: the full findings are on review record rec-123.)" in text
    assert text.count("- [ADVISORY] bug_hunting (m): ") == MAX_PREVIOUS_FINDINGS
    legacy = _change_section({**change, "record_id": ""})
    assert "the full findings are on the skill's review.json of that verdict" in legacy
    whole = _change_section({**change, "findings": findings[1:3]})
    assert "Cut for size" not in whole and "not shown" not in whole and "truncated" not in whole


def test_a_left_skill_wave_that_settles_after_its_task_is_marked_as_a_late_skill_review():
    """subagent 3: the late mark names the surface it was written for."""
    from ouroboros.late_review_marks import SURFACE_NAMES, _text

    assert SURFACE_NAMES["skill"] == "skill review"
    text = _text({"surface": "skill", "task_id": "task-skill", "settled_at": "2026-10-21T12:00:00+00:00",
                  "subject": {"skill": "weather"}, "lines": ["- t1: PASS"]})
    assert text.startswith("Late skill review of task task-skill, settled 2026-10-21T12:00:00+00:00.")


# --- the receptive author, publication, and the silo between skill and repo review -------------------------


def test_the_tool_author_hears_mail_while_the_wave_runs_and_a_left_wave_is_reissued_by_name(harness):
    """The ``skill_review`` tool binds the receptive author around the lifecycle call (an addressed message
    may end the wait; the wave goes on) and the left-wave watch names the exact re-issue; a caller without a
    continuation owner keeps the synchronous wait."""
    import contextlib

    from ouroboros import review_pause
    from ouroboros.tools import skill_exec
    from tests.test_skill_review_runner import _reset_queue

    h = harness
    h.ctx.owner_wait_callback = lambda *_a, **_k: "parked"
    h.ctx.task_attempt = 1
    bindings, watched = [], []
    real_author = review_pause.receptive_author

    @contextlib.contextmanager
    def recording(binding):
        bindings.append(binding)
        with real_author(binding) as bound:
            yield bound

    h.monkeypatch.setattr(review_pause, "receptive_author", recording)
    h.monkeypatch.setattr(review_pause, "watch_left_review", lambda ctx, **kwargs: watched.append(kwargs))
    h.monkeypatch.setattr(skill_exec, "_skill_tool_preflight", lambda *a, **k: "")
    _reset_queue()
    try:
        text = skill_exec._handle_review_skill(h.ctx, skill="weather", reviewers=["t1"], reason="small payload")
    finally:
        _reset_queue()
    assert [send["slot_id"] for send in h.sends] == ["t1"] and "Review reference: " in text
    reference = json.loads(text.rsplit("Review reference: ", 1)[1])
    assert reference["record_id"] == _state(h).record_id and reference["content_hash"] == _state(h).content_hash
    [binding] = bindings
    assert binding and binding["task_id"] == "task-skill" and binding["drive_root"] == h.drive
    [kwargs] = watched
    assert kwargs["surface"] == "skill" and kwargs["record_id"] == reference["record_id"]
    assert kwargs["reissue"] == "review_change(review_reference=the original reference)"
    assert review_pause._LEFT_TOOLS["skill"] == "review_skill" and review_pause._LEFT_SURFACES["skill"] == "skill"
    assert review_pause.receptive_binding(SimpleNamespace(drive_root=h.drive, task_id="task-skill")) is None


def test_publication_reads_the_projected_verdict_fresh_stale_or_blocked(harness):
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_publish_eligibility import submit_hub_eligibility

    h = harness

    def eligibility():
        skill = find_skill(h.drive, "weather", repo_path="")
        state = _state(h)
        return submit_hub_eligibility(source="external", review_status=state.status, review_profile=state.review_profile,
                                      review_stale=state.is_stale_for(skill.content_hash), github_token_configured=True)

    assert _review(h, reviewers=["t1"], reason="small payload").status == "clean"
    assert eligibility()["reason"] == "Open Publish to check the current bytes before publication"
    (h.skill_dir / "scripts" / "hello.py").write_text("print('edited')\n", encoding="utf-8")
    assert eligibility()["reason"] == "Skill needs a fresh review before publication"
    h.answers["t1"] = HARD
    assert _review(h, reviewers=["t1"], reason="small payload").status == "blockers"
    assert eligibility()["reason"] == "Skill needs review work before publication"


def test_a_skill_record_creates_no_repo_review_debt_and_a_repo_record_no_skill_verdict(harness):
    """CHECKLISTS "Skill review vs. repo review", both directions: the paid skill wave is invisible to every
    repo-side reader (the commit gate's cycles, the change lane, the gate surfaces of the ledger), and a repo
    record is invisible to every skill-side reader (the skill's records, its earlier verdict, the task's skills)."""
    from ouroboros.review_state import load_state
    from ouroboros.skill_readiness import _skill_names_from_review_history
    from ouroboros.tools import commit_gate
    from ouroboros.tools.review_change_custody import review_state_root
    from ouroboros.tools.review_skill import earlier_verdict, skill_record_rows
    from ouroboros.skill_loader import find_skill
    from tests.test_review_ledger import _facts, _raw

    h = harness
    h.ctx._skill_review_group_id = "task:task-skill:weather"
    outcome = _review(h, reviewers=["t1"], reason="small payload")
    assert outcome.paid and outcome.status == "clean"
    for surface in ("commit_gate", "change", "result", "preflight"):
        assert review_ledger.recent_records(h.drive, "task-skill", surface=surface) == [], surface
    attempts = load_state(review_state_root(h.ctx))
    assert [row.tool_name for row in attempts.filter_attempts(tool_name="review_skill") if row.paid] == ["review_skill"]
    for tool in ("commit_reviewed", "review_change", "review_result"):
        assert attempts.filter_attempts(tool_name=tool) == []
        h.ctx._current_review_tool_name = tool
        assert commit_gate.count_paid_review_cycles(h.ctx, root_task_id="task-skill") == 0, tool
    h.ctx._current_review_tool_name = "review_skill"
    assert commit_gate.count_paid_review_cycles(h.ctx, root_task_id="task-skill") == 1

    repo = review_ledger.build_wave_record(_facts([_raw("s1", "openai/gpt-5")], task_id="task-skill"),
                                           surface="commit_gate", drive_root=h.drive)
    review_ledger.write_record(h.drive, repo)
    assert [row["record_id"] for row in review_ledger.recent_records(h.drive, "task-skill", surface="commit_gate")] == [repo.record_id]
    assert [row["record_id"] for row in skill_record_rows(h.drive, "weather")] == [outcome.record_id]
    assert earlier_verdict(h.drive, find_skill(h.drive, "weather", repo_path=""))["record_id"] == outcome.record_id
    assert _skill_names_from_review_history(h.drive, "task-skill")["names"] == ["weather"]
