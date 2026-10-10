"""The change-relative required-source manifest of the two-part brief.

What the producer owes, what identity each row carries (the same one a
``read_file`` receipt stamps), and how the chain reaches the retrieving seat's
request policy, its answer record and the wave reducer.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.runtime_mode_policy import GIT_OPS_FAMILY_PATHS, SAFETY_CRITICAL_PATHS
from ouroboros.tools.scope_required_sources import (
    RANGE_BASIS,
    REQUIRED_SOURCE_ROOT,
    SCOPE_REQUIRED_SOURCES_POLICY,
    coverage_state,
    render_required_sources,
    render_touched_manifest,
    required_sources_ref,
    scope_required_sources,
    staged_touched_paths,
    staged_tree_identity,
    touched_manifest,
    uncovered_sources,
)


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@ouroboros"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    return repo


def _write(repo, rel, text):
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _commit(repo, message="c"):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)


def test_only_protected_contract_and_prompt_paths_are_owed_in_full(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "ouroboros/safety.py", "SAFETY = 1\n")
    _write(repo, "prompts/SYSTEM.md", "system prompt\n")
    _write(repo, "ouroboros/contracts/tool_abi.py", "ABI = 1\n")
    _write(repo, "ouroboros/tools/review_brief_coupling.py", "ordinary = 1\n")
    _write(repo, "web/modules/chat.js", "export const a = 1;\n")
    _commit(repo)

    rows = scope_required_sources(repo, [
        ("M", "ouroboros/safety.py"), ("M", "prompts/SYSTEM.md"),
        ("A", "ouroboros/contracts/tool_abi.py"),
        ("M", "ouroboros/tools/review_brief_coupling.py"), ("M", "web/modules/chat.js"),
    ])
    owed = {row["path"]: row["disposition"] for row in rows}
    assert owed == {
        "ouroboros/safety.py": "modified",
        "prompts/SYSTEM.md": "modified",
        "ouroboros/contracts/tool_abi.py": "added",
        # The contract package declares its cross-language twin.
        "web/modules/api_types.js": "twin",
    }
    # A merely-touched ordinary file is a POINTER, never a required source: its
    # complete change evidence is the inlined diff.
    assert "ouroboros/tools/review_brief_coupling.py" not in owed
    assert "web/modules/chat.js" not in owed


def test_row_identity_matches_the_read_file_receipt_it_will_be_folded_against(tmp_path):
    repo = _repo(tmp_path)
    # CRLF on disk: source_revision names the BYTES, the char ranges the
    # universal-newline text the reader delivers.
    (repo / "prompts").mkdir()
    (repo / "prompts" / "SYSTEM.md").write_bytes(b"first\r\nsecond\r\n")
    _commit(repo)

    row = scope_required_sources(repo, [("M", "prompts/SYSTEM.md")])[0]
    raw = (repo / "prompts" / "SYSTEM.md").read_bytes()
    text = raw.decode().replace("\r\n", "\n")
    assert row["root"] == REQUIRED_SOURCE_ROOT
    assert row["source_revision"] == hashlib.sha256(raw).hexdigest()
    assert row["complete_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert row["complete_chars"] == len(text) == 13
    assert row["range_basis"] == RANGE_BASIS
    assert row["coverage_basis"] == "candidate_blob"


def test_a_touched_family_member_owes_the_whole_declared_family(tmp_path):
    repo = _repo(tmp_path)
    for rel in sorted(GIT_OPS_FAMILY_PATHS):
        _write(repo, rel, f"# {rel}\n")
    _commit(repo)

    rows = scope_required_sources(repo, [("M", "supervisor/git_ops_reset.py")])
    owed = {row["path"]: row["disposition"] for row in rows}
    assert set(owed) == set(GIT_OPS_FAMILY_PATHS)
    assert owed["supervisor/git_ops_reset.py"] == "modified"
    assert owed["supervisor/git_ops.py"] == "family"


def test_the_tool_dispatch_family_is_read_from_the_protection_constant(tmp_path):
    repo = _repo(tmp_path)
    family = {p for p in SAFETY_CRITICAL_PATHS if p.startswith("ouroboros/tools/")}
    for rel in sorted(family):
        _write(repo, rel, f"# {rel}\n")
    _commit(repo)

    rows = scope_required_sources(repo, [("M", "ouroboros/tools/registry.py")])
    assert {row["path"] for row in rows} == family


def test_api_types_touched_owes_its_host_contract_twin(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "web/modules/api_types.js", "export const V = 1;\n")
    _write(repo, "ouroboros/gateway/contracts.py", "V = 1\n")
    _commit(repo)

    rows = scope_required_sources(repo, [("M", "web/modules/api_types.js")])
    owed = {row["path"]: row["disposition"] for row in rows}
    # One contract, two languages: both sides are owed in full.
    assert owed == {
        "web/modules/api_types.js": "modified",
        "ouroboros/gateway/contracts.py": "twin",
    }


def test_a_deleted_required_source_keeps_its_preimage_obligation(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "prompts/SYSTEM.md", "system prompt\n")
    _commit(repo)
    (repo / "prompts" / "SYSTEM.md").unlink()

    rows = scope_required_sources(repo, [("D", "prompts/SYSTEM.md")])
    assert rows == [{
        "root": REQUIRED_SOURCE_ROOT, "path": "prompts/SYSTEM.md",
        "disposition": "deleted", "coverage_basis": "preimage_unavailable",
        "preimage": "HEAD:prompts/SYSTEM.md",
        "reason": rows[0]["reason"],
    }]
    assert "not delivered" in rows[0]["reason"]


def test_an_unreadable_required_source_is_typed_rather_than_dropped(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "ouroboros/safety.py", "SAFETY = 1\n")
    _commit(repo)
    # A family member the candidate tree does not carry at all.
    rows = scope_required_sources(repo, [("M", "supervisor/git_ops.py")])
    bases = {row["path"]: row["coverage_basis"] for row in rows}
    assert set(bases.values()) == {"source_unavailable"}


def test_the_candidate_tree_binds_every_row_when_the_caller_supplies_it(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "prompts/SYSTEM.md", "system prompt\n")
    _commit(repo)

    rows = scope_required_sources(repo, [("M", "prompts/SYSTEM.md")], staged_tree_sha="d" * 40)
    assert rows[0]["candidate_tree"] == "d" * 40


def test_touched_paths_come_from_the_staged_index_or_the_managed_subject(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "keep.py", "a = 1\n")
    _write(repo, "gone.py", "b = 2\n")
    _commit(repo)
    _write(repo, "keep.py", "a = 2\n")
    (repo / "gone.py").unlink()
    _write(repo, "added.py", "c = 3\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)

    assert sorted(staged_touched_paths(repo)) == [
        ("A", "added.py"), ("D", "gone.py"), ("M", "keep.py"),
    ]
    assert len(staged_tree_identity(repo)) == 40

    subject = SimpleNamespace(name_status=(("M", "resolved.py"),),
                              conflict_paths=("anchor.py",), staged_tree="e" * 40)
    assert staged_touched_paths(repo, subject) == [("M", "resolved.py"), ("M", "anchor.py")]
    assert staged_tree_identity(repo, subject) == "e" * 40


def test_a_subject_without_a_declared_path_set_falls_back_to_the_index(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "keep.py", "a = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    assert staged_touched_paths(repo, object()) == [("A", "keep.py")]


def test_a_rename_owes_the_preimage_path_too(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "prompts/SYSTEM.md", "x" * 200 + "\n")
    _commit(repo)
    subprocess.run(["git", "mv", "prompts/SYSTEM.md", "prompts/SYSTEM2.md"], cwd=repo, check=True)
    pairs = {path: status for status, path in staged_touched_paths(repo)}
    assert pairs == {"prompts/SYSTEM2.md": "R", "prompts/SYSTEM.md": "D"}

    owed = {row["path"]: row["disposition"] for row in scope_required_sources(repo)}
    assert owed["prompts/SYSTEM2.md"] == "modified"
    assert owed["prompts/SYSTEM.md"] == "deleted"


def test_the_manifest_ref_is_the_manifest_identity(tmp_path):
    rows = [{"path": "prompts/SYSTEM.md", "complete_chars": 3}]
    ref = required_sources_ref(rows, staged_tree_sha="a" * 40)
    assert ref["policy"] == SCOPE_REQUIRED_SOURCES_POLICY
    assert ref["staged_tree_sha"] == "a" * 40
    assert ref["required_source_count"] == 1
    assert ref["sha256"] == hashlib.sha256(json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    assert required_sources_ref([])["required_source_count"] == 0


def test_the_brief_states_the_manifest_is_a_minimum(tmp_path):
    text = render_required_sources([
        {"path": "prompts/SYSTEM.md", "disposition": "modified", "complete_chars": 1234},
        {"path": "prompts/OLD.md", "disposition": "deleted", "coverage_basis": "preimage_unavailable"},
    ])
    assert "MINIMUM, not a sufficiency claim" in text
    assert "prompts/SYSTEM.md (modified, 1,234 chars)" in text
    assert "prompts/OLD.md (deleted, preimage_unavailable)" in text
    assert "no source is owed in full" in render_required_sources([])


@pytest.mark.parametrize("raw,extent", [(b"a = 1\n", "6 bytes"), (b"a = 1\r\n", "7 bytes")])
def test_the_touched_manifest_carries_dispositions_and_candidate_sizes(tmp_path, raw, extent):
    repo = _repo(tmp_path)
    (repo / "keep.py").write_bytes(raw)
    rows = touched_manifest(repo, [("M", "keep.py"), ("D", "gone.py"), ("M", "keep.py")])
    assert rows == [
        {"path": "gone.py", "disposition": "deleted", "extent": "not in the candidate tree"},
        {"path": "keep.py", "disposition": "modified", "extent": extent},
    ]
    rendered = render_touched_manifest(rows)
    assert "complete change evidence is the staged diff" in rendered
    assert f"- keep.py (modified, {extent})" in rendered
    assert render_touched_manifest([]) == ""


@pytest.mark.parametrize("fact,expected", [
    ({"status": "complete"}, "complete"),
    ({"status": "complete", "reason": "declared_empty"}, "declared_empty"),
    ({"status": "incomplete"}, "incomplete"),
    ({"status": "unobserved", "reason": "required_source_manifest_missing"}, "unobserved"),
    (None, "unobserved"),
    ("not a fact", "unobserved"),
])
def test_the_four_coverage_states(fact, expected):
    assert coverage_state(fact) == expected


def test_uncovered_sources_names_only_the_rows_that_fell_short():
    fact = {"status": "incomplete", "sources": [
        {"path": "a.py", "status": "complete"},
        {"path": "b.py", "status": "incomplete"},
        {"path": "c.py", "status": "unobserved"},
    ]}
    assert uncovered_sources(fact) == ["b.py", "c.py"]
    assert uncovered_sources(None) == []


# ---------------------------------------------------------------------------
# The chain: producer -> request policy -> coverage -> record -> reducer
# ---------------------------------------------------------------------------

from tests.test_review_session_scope_wiring import _coupling_matrix_rows  # noqa: E402

def _staged_protected_repo(tmp_path):
    import subprocess

    repo = tmp_path / "candidate"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@ouroboros"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    (repo / "prompts").mkdir()
    (repo / "prompts" / "SYSTEM.md").write_text("runtime system prompt\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    (repo / "prompts" / "SYSTEM.md").write_text("runtime system prompt, amended\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    return repo


def _one_retrieving_seat(delivery):
    from ouroboros.review_execution import ReviewRouteKind

    route = ReviewRouteKind.AGENT_SESSION if delivery == "session" else ReviewRouteKind.API_CHAT
    return {"models": ["fixture/model"], "routes": [route], "efforts": [""], "session_targets": ["fixture/model"],
            "session_profiles": [""], "subagent_ids": [""], "use_local": [None], "slot_ids": ["slot_1"],
            "retrieves": [True], "parts": [("change", "coupling")]}


@pytest.mark.parametrize("delivery", ["native", "session"])
def test_the_required_source_manifest_reaches_both_retrieving_deliveries(
    tmp_path, monkeypatch, delivery
):
    """The producer's rows travel to the reviewer three ways: the exact
    identities in the request policy (folded by whoever observes the reads),
    the manifest identity, and the human list in the brief itself. The wave's
    assembly (``_prepare_unified_review``) builds them per seat; its dispatch
    hands them to the substrate as the seat's policy."""
    from ouroboros.tools import review as review_mod
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.scope_required_sources import SCOPE_REQUIRED_SOURCES_POLICY
    import ouroboros.reviewer_slot_config as slot_cfg

    repo = _staged_protected_repo(tmp_path)
    captured = {}

    def _capture(request, *, slots, drive_root, llm, usage_ctx=None):
        captured["policy"] = dict(request.policy)
        captured["task"] = request.session_task
        captured["messages"] = list(request.messages)
        return SimpleNamespace(actors=[{
            "slot_id": slots[0].slot_id, "model": slots[0].model, "status": "ok",
            "raw_text": json.dumps({"change": [], "change_clean": True, "coupling": _coupling_matrix_rows()}),
            "usage": {"native_read_coverage": {"status": "complete", "sources": []}},
            "prompt_ref": {}, "response_ref": {},
        }])

    monkeypatch.setattr("ouroboros.review_substrate.run_review_request", _capture)
    monkeypatch.setattr(slot_cfg, "commit_triad_delivery", lambda: _one_retrieving_seat(delivery))
    (tmp_path / "data").mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "data")
    ctx.task_id = "required-sources"
    ctx._review_history, ctx._review_advisory, ctx._coupling_review_history = [], [], {}

    prepared, _early, exited = review_mod._prepare_unified_review(ctx, "amend the system prompt")
    assert not exited
    policy = prepared["row_plan"]["session_policies"][0]
    rows = policy["native_required_sources"]
    assert [row["path"] for row in rows] == ["prompts/SYSTEM.md"]
    assert rows[0]["complete_chars"] == len("runtime system prompt, amended\n")
    assert rows[0]["range_basis"] == "unicode_text_universal_newlines"
    ref = policy["native_required_sources_ref"]
    assert ref["policy"] == SCOPE_REQUIRED_SOURCES_POLICY and ref["required_source_count"] == 1
    brief = prepared["row_plan"]["session_tasks"][0]
    assert "prompts/SYSTEM.md (modified" in brief
    assert "MINIMUM, not a sufficiency claim" in brief

    if delivery == "session":
        return  # the session transport is pinned in tests/test_review_session_scope_wiring.py
    review_mod._dispatch_unified_review(ctx, "amend the system prompt", prepared)
    assert captured["policy"]["native_required_sources"] == rows
    assert captured["policy"]["native_required_sources_ref"] == ref
    # No packet is ever rendered for a retrieving seat; the brief names the source.
    assert captured["messages"] == []
    assert captured["task"] == brief
    outcome = ctx._last_coupling_result
    assert outcome.status == "responded" and outcome.verdict == "PASS"
    assert outcome.seats[0]["coverage"] == "complete"


@pytest.mark.parametrize("fact,expected", [
    ({"status": "complete", "sources": []}, "complete"),
    ({"status": "complete", "reason": "declared_empty", "sources": []}, "declared_empty"),
    ({"status": "incomplete", "sources": [{"path": "prompts/SYSTEM.md", "status": "incomplete"}]},
     "incomplete"),
    (None, "unobserved"),
])
def test_the_seat_record_carries_the_observed_coverage_state(fact, expected):
    """One reader for every delivery: whoever observed the reads reports the
    fact, and an absent fact is `unobserved` rather than a guess."""
    from ouroboros.triad_review import parse_seat_answers

    actor = {"model": "fixture", "slot_id": "scope_slot_1", "verdict": "OK",
             "text": json.dumps({"coupling": _coupling_matrix_rows()})}
    if fact is not None:
        actor["native_read_coverage"] = fact
    parsed = parse_seat_answers({"results": [actor]}, {"scope_slot_1": ("coupling",)})
    record = parsed.actor_records[0]
    assert record.status == "responded" and record.coverage == expected
    if fact is not None:
        assert record.context_manifest["native_read_coverage"] == fact


def _seat(slot_id, *, status="responded", coverage="complete", uncovered=(), critical=()):
    """One wave seat's raw record, as the wave's parser would have left it."""
    from ouroboros.triad_review import parse_seat_answers

    rows = _coupling_matrix_rows()
    for finding in critical:
        rows = [finding if row["item"] == finding["item"] else row for row in rows]
    actor = {"model": f"m/{slot_id}", "slot_id": slot_id, "verdict": "OK" if status == "responded" else "ERROR",
             "text": json.dumps({"change": [], "change_clean": True, "coupling": rows}) if status == "responded" else "",
             **({"error": "Error: transport failed"} if status != "responded" else {})}
    if coverage != "unobserved":
        actor["native_read_coverage"] = {
            "status": "incomplete" if coverage == "incomplete" else "complete",
            **({"reason": "declared_empty"} if coverage == "declared_empty" else {}),
            "sources": [{"path": path, "status": "incomplete"} for path in uncovered]}
    return parse_seat_answers({"results": [actor]}, {slot_id: ("change", "coupling")}).actor_records[0].to_dict()


def _reduce(records):
    """The wave reducer over prepared seat records: the verdict, its coupling
    outcome and the history line the subject keeps."""
    from ouroboros.review_ledger import build_rows, coupling_outcome, reduce_verdict
    from ouroboros.tools.parallel_review import _coupling_history_entry

    rows = build_rows({"triad_raw": records})
    verdict = reduce_verdict(rows)
    outcome = coupling_outcome(verdict, rows)
    return verdict, outcome, _coupling_history_entry(outcome)


def test_unobserved_coverage_counts_toward_the_quorum():
    """A delivery whose reads the host cannot see keeps the P3 exception: its
    verdict counts, and the provenance limit is disclosure, not a shortfall."""
    verdict, outcome, _entry = _reduce([_seat("scope_slot_1", coverage="unobserved"),
                                        _seat("scope_slot_2", coverage="declared_empty")])
    assert verdict["aggregate"] == "PASS" and verdict["quorum"]["responded"] == 2
    assert outcome.blocked is False and outcome.status == "responded"
    assert [s["coverage"] for s in outcome.seats] == ["unobserved", "declared_empty"]


@pytest.mark.parametrize("second_coverage", ["complete", "incomplete"])
def test_read_coverage_is_diagnostic_on_the_record(second_coverage):
    """Read coverage never changes the verdict: it rides the seat record and the
    history line as a diagnostic, under every enforcement (the ledger knows no
    enforcement at all; the gate applies it after the verdict)."""
    verdict, outcome, entry = _reduce([
        _seat("scope_slot_1", coverage="complete"),
        _seat("scope_slot_2", coverage=second_coverage, uncovered=("prompts/SYSTEM.md",))])
    assert verdict["aggregate"] == "PASS" and verdict["quorum"]["responded"] == 2
    assert outcome.blocked is False and not outcome.advisory_findings and not outcome.critical_findings
    assert outcome.seats[1] == {**outcome.seats[1], "slot_id": "scope_slot_2", "status": "responded",
                                "coverage": second_coverage, "matrix": "full"}
    assert f"scope_slot_2: {second_coverage}" in entry["summary"]
    assert "Read coverage (diagnostic):" in entry["summary"]


@pytest.mark.parametrize("count", [1, 3])
def test_every_answer_counts_even_when_all_seats_have_incomplete_coverage(count):
    verdict, outcome, _entry = _reduce([
        _seat(f"scope_slot_{i + 1}", coverage="incomplete", uncovered=("ouroboros/safety.py",)) for i in range(count)])
    assert verdict["aggregate"] == "PASS" and verdict["quorum"]["responded"] == count
    assert outcome.blocked is False and not outcome.advisory_findings
    assert all(s["status"] == "responded" and s["coverage"] == "incomplete" for s in outcome.seats)


def test_incomplete_coverage_never_hides_substantive_critical_findings():
    finding = {"item": "cross_module_bugs", "severity": "critical", "verdict": "FAIL",
               "reason": "The producer and consumer use different units."}
    verdict, outcome, entry = _reduce([
        _seat("scope_slot_1"), _seat("scope_slot_2"),
        _seat("scope_slot_3", coverage="incomplete", uncovered=("prompts/SYSTEM.md",), critical=(finding,))])
    assert verdict["aggregate"] == "FAIL" and verdict["per_question"]["coupling"] == "FAIL"
    assert outcome.blocked is True
    assert [{k: f[k] for k in finding} for f in outcome.critical_findings] == [finding]
    assert outcome.seats[2]["coverage"] == "incomplete"
    assert entry["summary"].startswith("Critical: cross_module_bugs")


def test_coverage_diagnostics_do_not_become_technical_failures():
    from ouroboros.tools.commit_gate import review_failure_is_technical

    assert not review_failure_is_technical({"failure_phase": "coverage_authority"})
    assert not review_failure_is_technical(
        {"failure_phase": "coverage_authority", "operation_state": "in_flight"})
    assert review_failure_is_technical({"failure_phase": "delivery"})


@pytest.mark.parametrize("coverage", ["complete", "incomplete"])
def test_read_diagnostics_do_not_hide_a_missing_reviewer_answer(coverage):
    verdict, outcome, _entry = _reduce([_seat("scope_slot_1", coverage=coverage),
                                        _seat("scope_slot_2", status="error", coverage="unobserved")])
    # Two assigned seats need both; one answer is a quorum failure, never a PASS.
    assert verdict["aggregate"] == "QUORUM_FAILED"
    assert verdict["quorum"] == {**verdict["quorum"], "responded": 1, "assigned": 2, "required": 2}
    # The coupling QUESTION keeps the one answer it got (its own quorum is one
    # seat); the WAVE is what failed, and the gate decides by the aggregate.
    assert outcome.status == "responded" and outcome.verdict == "PASS" and outcome.blocked is False
    assert [s["status"] for s in outcome.seats] == ["responded", "error"]


def test_the_review_contract_fingerprint_binds_the_parts_and_the_contracts(monkeypatch):
    """Recorded free-replay authority must not survive a contract change: the
    parts each seat is asked, the two answer contracts and the manifest policy
    are all hashed into the commit gate's contract identity."""
    import ouroboros.reviewer_slot_config as slot_cfg
    from ouroboros.review_records import ReviewRouteKind
    from ouroboros.tools.commit_gate import commit_review_contract_fingerprint

    def _plan(parts):
        return {"models": ["m"], "routes": [ReviewRouteKind.API_CHAT], "efforts": [""], "session_targets": [""],
                "session_profiles": [""], "subagent_ids": [""], "use_local": [None], "slot_ids": ["slot_1"],
                "retrieves": [True], "parts": [parts]}

    monkeypatch.setattr(slot_cfg, "commit_triad_delivery", lambda: _plan(("change", "coupling")))
    baseline = commit_review_contract_fingerprint()
    assert baseline

    # The SAME seat asked a different question is a different contract.
    monkeypatch.setattr(slot_cfg, "commit_triad_delivery", lambda: _plan(("change",)))
    assert commit_review_contract_fingerprint() != baseline

    monkeypatch.setattr(slot_cfg, "commit_triad_delivery", lambda: _plan(("change", "coupling")))
    assert commit_review_contract_fingerprint() == baseline
    with monkeypatch.context() as patched:
        patched.setattr("ouroboros.tools.scope_required_sources.SCOPE_REQUIRED_SOURCES_POLICY", "v-next")
        assert commit_review_contract_fingerprint() != baseline
    with monkeypatch.context() as patched:
        patched.setattr("ouroboros.triad_review.REVIEW_TWO_PART_OBJECT_CONTRACT", "a different contract B")
        assert commit_review_contract_fingerprint() != baseline
    with monkeypatch.context() as patched:
        patched.setattr("ouroboros.triad_review.REVIEW_JSON_ARRAY_CONTRACT", "a different contract A")
        assert commit_review_contract_fingerprint() != baseline


def test_a_native_retrieving_seat_is_priced_as_its_first_send(tmp_path):
    """Wave admission must price what the seat SENDS: a native retrieving seat
    opens an inspection episode, so its price is the episode's first send (work
    order, its own two-part brief and tool schemas), never a packet message pair
    it never assembles."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_native_episode import native_first_send_chars
    from ouroboros.tools.review_admission import commit_gate_paid_seats
    from ouroboros.tools.review_multi_model import TRIAD_ROLE_HINT
    from ouroboros.triad_review import REVIEW_TWO_PART_OBJECT_CONTRACT

    repo = _repo(tmp_path)
    prepared = {"prompt": "", "stable_prefix_len": 0, "target_repo": str(repo), "models": ["api/model"],
                "routes": [ReviewRouteKind.API_CHAT],
                "row_plan": {"models": ["api/model"], "routes": [ReviewRouteKind.API_CHAT], "slot_ids": ["scope_slot_1"],
                             "retrieves": [True], "parts": [("coupling",)], "session_tasks": ["BRIEF"]}}
    seats = commit_gate_paid_seats(prepared, False)
    assert [s["slot_id"] for s in seats] == ["scope_slot_1"]
    assert seats[0]["prompt_chars"] == native_first_send_chars(
        str(repo), surface="multi_model_review", role_hint=TRIAD_ROLE_HINT,
        slot_id="scope_slot_1", session_task="BRIEF", output_contract=REVIEW_TWO_PART_OBJECT_CONTRACT)
    assert commit_gate_paid_seats(prepared, True) == []
