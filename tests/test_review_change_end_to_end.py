"""One subject, two doors, the same brief (seam 4 of the review operation).

``review_change(root="system_repo", subject="index")`` and the commit gate's
review-only cycle review the same staged index through the same seats, the same
body-layer rules and the same prompt text. The ONLY difference the two doors
may leave in a brief is the narrative line that names the wave: the gate hands
the reviewers the intended commit message, the operation its own wave label
(``review_change: staged index of <root> ...``), both inside the
``## Informational context — commit message`` block the goal section renders.
This test pins that difference to exactly that block and nothing else.
"""

import hashlib
import json
from pathlib import Path

import pytest

import ouroboros.review_substrate as substrate
from ouroboros import review_ledger
from ouroboros.tools import git as git_mod
from ouroboros.tools import review as review_mod
from ouroboros.tools import review_change
from ouroboros.tools.git_review_cycle import _run_non_committing_review_cycle
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.review_change import run_review_change
from ouroboros.tools.review_subject import CHECKOUT_SUBDIR
from tests import _contributor_packet_shared as shared
from tests.review_pool_rosters import set_review_pool

GOAL = "Make the installed body's helper return the proposal's constant."
SCOPE = "ouroboros/helper.py only; the checklist and tests stay as they are."
COMMIT_MESSAGE = "fix: return the proposal's constant\n\nThe narrative body of the intended commit."


def _brief_text(brief: dict) -> str:
    """Every byte a seat was given: the message texts (plain or block-structured)
    and, for a retrieving seat, its session task."""
    parts = []
    for message in brief["messages"]:
        content = message.get("content") or ""
        if isinstance(content, list):
            parts.extend(str(block.get("text") or json.dumps(block, sort_keys=True)) for block in content)
        else:
            parts.append(str(content))
    return "\n".join(parts) + "\n" + brief["session_task"]


@pytest.fixture
def staged_body(tmp_path, monkeypatch):
    fixture = shared.init_installed_body(tmp_path)
    repo = Path(fixture["repo"])
    # A real install carries a .gitignore; without one the gate's `git add -A`
    # door would stage the one it writes and review a different tree.
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    shared.git(repo, "add", ".gitignore")
    shared.git(repo, "commit", "-q", "-m", "ignore caches")
    shared.git(repo, "cherry-pick", "--no-commit", fixture["head_sha"])
    fixture["staged_tree_sha"] = shared.git(repo, "write-tree")
    assert fixture["staged_tree_sha"] != fixture["head_tree_sha"]
    set_review_pool(monkeypatch, shared.golden_pool())
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "1")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")  # the proposal touches a protected surface
    monkeypatch.setattr(git_mod, "_run_review_preflight_tests", shared.passing_test_runner)
    return fixture


def test_review_change_on_the_system_index_is_the_commit_gates_brief(staged_body, tmp_path, monkeypatch):
    repo = Path(staged_body["repo"])
    operation_briefs: list[dict] = []
    monkeypatch.setattr(substrate, "run_review_request", shared.golden_substrate(operation_briefs))
    operation_ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "operation-drive")
    result = run_review_change(operation_ctx, root="system_repo", surface="change", goal=GOAL, scope=SCOPE,
                               subject="index")
    assert result["aggregate"] == "PASS" and result["state"] == "settled", result
    operation_record = review_ledger.load_record(operation_ctx.drive_root, result["record_id"])
    # The operation only reads: the staged index it reviewed is still staged.
    assert shared.git(repo, "write-tree") == staged_body["staged_tree_sha"]

    # The gate's review-only cycle is the second door onto the same index (it
    # unstages the index when it is done, which is why it goes second here).
    gate_briefs: list[dict] = []
    monkeypatch.setattr(substrate, "run_review_request", shared.golden_substrate(gate_briefs))
    gate_ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "gate-drive")
    outcome = _run_non_committing_review_cycle(gate_ctx, COMMIT_MESSAGE, skip_advisory_review=True,
                                               goal=GOAL, scope=SCOPE)
    assert outcome["status"] == "passed", outcome
    gate_record = review_ledger.load_record(gate_ctx.drive_root, outcome["review_record_id"])

    # Same subject, same seats, same rules, same verdict.
    assert operation_record["subject"]["tree_sha"] == gate_record["subject"]["tree_sha"] == staged_body["staged_tree_sha"]
    assert operation_record["subject"]["diff_sha"] == gate_record["subject"]["diff_sha"]
    operation_checklist = dict(operation_record["brief"]["checklist"])
    assert operation_checklist.pop("treat_as_body") is False  # the operation's own argument, recorded
    assert operation_checklist == gate_record["brief"]["checklist"]
    assert (operation_checklist["layer"], operation_checklist["body_fact"], operation_checklist["how"]) == (
        "body", "true", "dir")
    assert operation_checklist["rules_source"]["sha"] and operation_checklist["checklist_hash"]
    assert (operation_record["brief"]["goal"], operation_record["brief"]["scope"]) == (GOAL, SCOPE) == (
        gate_record["brief"]["goal"], gate_record["brief"]["scope"])
    seats = lambda record: [(row["seat_id"], row["requested"]["model"], row["effective"]["model"], row["parts"])  # noqa: E731
                            for row in record["rows"]]
    assert seats(operation_record) == seats(gate_record)
    shared_panel = ("seats", "distinct_models", "observed_unknown_seats", "distinct_engines", "single_model_panel",
                    "chosen_by", "assigned", "additional")
    assert {key: operation_record["panel"][key] for key in shared_panel} == {
        key: gate_record["panel"][key] for key in shared_panel}
    assert operation_record["panel"]["composition"] == "full_pool"
    # The whole pool sat on both surfaces (the operation's extra seat is `additional`),
    # and a full pool owes no reason: `reason_missing` is a fact only about a panel an
    # author narrowed without one (contract §1.6).
    assert (gate_record["panel"]["reason_missing"], operation_record["panel"]["reason_missing"]) == (False, False)
    assert operation_record["verdict"]["aggregate"] == gate_record["verdict"]["aggregate"] == "PASS"
    assert (operation_record["surface"], gate_record["surface"]) == ("change", "commit_gate")

    # Same brief per seat: the texts differ in the narrative wave line alone.
    by_seat = lambda briefs: {brief["slot_id"]: brief for brief in briefs}  # noqa: E731
    gate, operation = by_seat(gate_briefs), by_seat(operation_briefs)
    assert sorted(gate) == sorted(operation) == ["s1", "t1", "t2"]
    label = review_change._wave_label(review_change.parse_request(
        {"root": "system_repo", "surface": "change", "goal": GOAL, "scope": SCOPE, "subject": "index"}), repo)
    for slot_id in ("t1", "t2", "s1"):
        assert gate[slot_id]["model"] == operation[slot_id]["model"]
        gate_text, operation_text = _brief_text(gate[slot_id]), _brief_text(operation[slot_id])
        assert gate_text != operation_text, slot_id
        assert operation_text.count(label) == gate_text.count(COMMIT_MESSAGE) == 1, slot_id
        aligned = operation_text.replace(label, COMMIT_MESSAGE)
        assert hashlib.sha256(aligned.encode()).hexdigest() == hashlib.sha256(gate_text.encode()).hexdigest(), slot_id
        assert "## Informational context — commit message" in gate_text, slot_id


QUESTIONS = ["Does the proposal's constant reach every caller of the helper?", "Which test pins the new value?"]
PRIOR_OBLIGATION = "ob-prior-round-7"


def _seed_open_obligation(drive_root: Path, repo: Path) -> None:
    """One obligation of this checkout left open by an earlier round, in the durable
    advisory state every door reads its history from."""
    from ouroboros.review_state import AdvisoryReviewState, ObligationItem, make_repo_key, save_state

    state = AdvisoryReviewState()
    state.open_obligations.append(ObligationItem(
        obligation_id=PRIOR_OBLIGATION, item="cross_module_bugs", severity="critical",
        reason="the helper's callers were not re-read", source_attempt_ts="2026-10-01T00:00:00+00:00",
        source_attempt_msg="fix: an earlier attempt", status="still_open", repo_key=make_repo_key(repo)))
    save_state(drive_root, state)


def test_the_public_builder_renders_the_brief_each_seat_was_sent(staged_body, tmp_path, monkeypatch):
    """``review_admission.build_two_part_brief`` is the one public builder of a seat's
    brief (step R; D5-004, D5-07): for the frozen subject and the intent of a wave it
    renders, byte for byte, the text the operation handed that seat at the delivery
    boundary — the author's questions as the seat read them and the prior rounds with
    the checkout's open obligations, through the owners the runtime itself renders with.

    The builder takes as ARGUMENTS what a wave reads from its context; these, and only
    these, are where its text may differ from a wave's, and each is passed here as the
    wave had it:
      - ``commit_message``: the operation's wave label (``_wave_label``), the gate's
        intended commit message;
      - ``review_history`` / ``review_rebuttal`` / ``coupling_history``: this task's
        earlier rounds (none in a first round);
      - ``owner_words``: the owner's recorded words for the task as the wave renders
        them (``owner_words.owner_words_text(ctx)``, which says so when none are
        recorded); ``task_evidence_section``: the task's execution evidence (none here);
      - ``task_id`` / ``source_root``: the paging identity of a retrieving seat's sources;
      - a packet seat's governance share is sized for the one seat given, a wave's for
        its packet quorum (one packet seat sits in this pool, so the two coincide).
    """
    from ouroboros.owner_words import owner_words_text
    from ouroboros.review_ledger import PART_CHANGE
    from ouroboros.tools.review_admission import build_two_part_brief
    from ouroboros.tools.review_subject import ReviewSubjectSpec, freeze_subject

    repo = Path(staged_body["repo"])
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "operation-drive")
    _seed_open_obligation(ctx.drive_root, repo)
    sent: list[dict] = []
    monkeypatch.setattr(substrate, "run_review_request", shared.golden_substrate(sent))
    args = {"root": "system_repo", "surface": "change", "goal": GOAL, "scope": SCOPE, "subject": "index",
            "author_questions": QUESTIONS}
    result = run_review_change(ctx, **args)
    assert result["state"] == "settled", result
    record = review_ledger.load_record(ctx.drive_root, result["record_id"])
    rows = {row["seat_id"]: row for row in record["rows"]}
    by_seat = {brief["slot_id"]: brief for brief in sent}
    assert sorted(by_seat) == sorted(rows) == ["s1", "t1", "t2"]

    label = review_change._wave_label(review_change.parse_request(dict(args)), repo)
    frozen = freeze_subject(ctx, ReviewSubjectSpec(root_kind="system_repo", root=str(repo), kind="index",
                                                   governance_root=str(repo), surface="change", layer="body"))
    assert frozen.tree_sha == record["subject"]["tree_sha"] == staged_body["staged_tree_sha"]
    for slot_id, row in rows.items():
        requested = row["requested"]
        seat = {"slot_id": slot_id, "model": requested["model"], "route": requested["route"],
                "retrieves": requested["delivery"] == "retrieving", "session_profile": requested["profile"],
                "subagent_id": row["subagent_id"]}
        brief = build_two_part_brief(frozen, seat, goal=GOAL, scope=SCOPE, author_questions=QUESTIONS,
                                     commit_message=label, owner_words=owner_words_text(ctx),
                                     drive_root=ctx.drive_root, task_id=ctx.task_id)
        assert brief["parts"] == row["parts"], slot_id
        # The builder's claim is not vacuous: the questions and the open obligation are in its text.
        asked = "Author questions (answer each as asked):\n1. " + QUESTIONS[0] + "\n2. " + QUESTIONS[1]
        assert asked in brief["system"] and PRIOR_OBLIGATION in brief["system"], slot_id
        # ... and its text IS the text the seat was sent, byte for byte.
        given = by_seat[slot_id]
        if brief["parts"] == [PART_CHANGE]:  # a packet seat: the system blocks and the one user turn
            [system, user] = given["messages"]
            assert "".join(block["text"] for block in system["content"]) == brief["system"], slot_id
            assert (user["content"], given["session_task"]) == (brief["user"], ""), slot_id
        else:  # a retrieving seat: the two-part brief is its session task
            assert (given["messages"], given["session_task"]) == ([], brief["system"]), slot_id
            assert row["brief_sha"] == brief["sha"]["brief"] == hashlib.sha256(brief["system"].encode()).hexdigest(), slot_id


REASON = "A tooling-only change: one api seat and the scout's second opinion suffice."


def _golden_with_critic(briefs: list, critic: str):
    """``golden_substrate`` plus one unmarked packet row (``critic``) that answers as t1 does
    (the gate sends one seat per substrate call)."""
    import dataclasses

    golden = shared.golden_substrate(briefs)

    def run_review_request(request, *, slots, drive_root, llm=None, usage_ctx=None):
        if [slot.slot_id for slot in slots] != [critic]:
            return golden(request, slots=slots, drive_root=drive_root, llm=llm, usage_ctx=usage_ctx)
        [slot] = slots
        result = golden(request, slots=[dataclasses.replace(slot, slot_id="t1")], drive_root=drive_root, llm=llm,
                        usage_ctx=usage_ctx)
        briefs[-1]["slot_id"] = critic
        reserved = (getattr(usage_ctx, "_review_reserved_operations", None) or {}).get(request.surface) or {}
        result.actors[0] = {**result.actors[0], "slot_id": critic, "operation_id": str(reserved.get(critic) or f"op-{critic}")}
        return result

    return run_review_request


def _gate_with_critic(staged_body, tmp_path, monkeypatch, *, mode: str, **args):
    """The gate's review-only cycle with an unmarked enabled row ``scout`` beside the golden pool."""
    from tests.review_pool_rosters import pool_seat

    set_review_pool(monkeypatch, shared.golden_pool(pool_seat("scout", "openai/gpt-5.6-sol", effort="high", marked=False)))
    monkeypatch.setattr(git_mod, "get_runtime_mode", lambda: mode)
    briefs: list[dict] = []
    monkeypatch.setattr(substrate, "run_review_request", _golden_with_critic(briefs, "scout"))
    ctx = ToolContext(repo_dir=Path(staged_body["repo"]), drive_root=tmp_path / "gate-drive")
    outcome = _run_non_committing_review_cycle(ctx, COMMIT_MESSAGE, skip_advisory_review=True, goal=GOAL, scope=SCOPE, **args)
    assert outcome["status"] == "passed", outcome
    record = review_ledger.load_record(ctx.drive_root, outcome["review_record_id"])
    return record, sorted(brief["slot_id"] for brief in briefs)


def test_in_cyber_pro_commit_reviewed_composes_its_panel_from_the_pool_and_records_why(staged_body, tmp_path, monkeypatch):
    """Decision 1A on the commit gate (D1-01, D5-001, AUDV_D1 V01): `reviewers`/`reason` go
    through the ONE composer `review_change` uses. In Cyber Pro the named pool row IS the
    counted panel, the unmarked row is an added critic, and the reason is in the record."""
    record, sent = _gate_with_critic(staged_body, tmp_path, monkeypatch, mode="cyber_pro",
                                     reviewers=["s1", "scout"], reason=REASON)
    assert sent == ["s1", "scout"]  # the rest of the pool was not paid
    panel = record["panel"]
    assert (panel["composition"], panel["chosen_by"], panel["reason"], panel["reason_missing"]) == (
        "composed", "author", REASON, False)
    assert (panel["assigned"], panel["additional"], panel["seats"], panel["additional_seats"]) == (["s1"], ["scout"], 1, 1)
    assert panel["reviewers_requested"] == ["s1", "scout"]
    assert {row["seat_id"]: row["additional"] for row in record["rows"]} == {"s1": False, "scout": True}
    assert record["verdict"]["aggregate"] == "PASS" and record["mode"] == "cyber_pro"


def test_below_cyber_pro_the_whole_pool_judges_the_commit_and_named_rows_only_add(staged_body, tmp_path, monkeypatch):
    record, sent = _gate_with_critic(staged_body, tmp_path, monkeypatch, mode="pro", reviewers=["t1", "scout"])
    assert sent == ["s1", "scout", "t1", "t2"]
    panel = record["panel"]
    assert (panel["composition"], panel["chosen_by"], panel["reason_missing"], panel["reviewers_subset_ignored"]) == (
        "full_pool", "owner", False, True)
    assert (sorted(panel["assigned"]), panel["additional"], panel["seats"]) == (["s1", "t1", "t2"], ["scout"], 3)
    assert record["verdict"]["aggregate"] == "PASS"


@pytest.mark.parametrize("mode", ["cyber_pro", "pro"])
def test_the_commit_panel_hears_any_enabled_catalog_row_as_an_added_critic(staged_body, tmp_path, monkeypatch, mode):
    """``commit_reviewed`` composes through the one composer: an unmarked api row and an
    unmarked agent-session row (its own effort kept) are added critics beside the counted
    pool seats, in Cyber Pro and below; a switched-off row is refused before anything is
    staged, reviewed or recorded."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools import commit_gate
    from ouroboros.tools.review_change import ReviewChangeArgumentError
    from tests.review_pool_rosters import pool_seat

    set_review_pool(monkeypatch, shared.golden_pool(
        pool_seat("scout", "openai/gpt-5.6-sol", effort="high", marked=False),
        pool_seat("session-critic", "codex=gpt-5.6-sol", kind="agent_session", effort="xhigh", marked=False),
        pool_seat("retired", "openai/retired-model", marked=False, enabled=False)))
    monkeypatch.setattr(git_mod, "get_runtime_mode", lambda: mode)
    ctx = ToolContext(repo_dir=Path(staged_body["repo"]), drive_root=tmp_path / "gate-drive")

    panel = commit_gate.compose_commit_panel(ctx, ["s1", "scout", "session-critic"], REASON)
    seats = {seat.slot.slot_id: seat for seat in panel.seats}
    assert panel.facts["additional"] == ["scout", "session-critic"]
    assert sorted(name for name, seat in seats.items() if not seat.additional) == (
        ["s1"] if mode == "cyber_pro" else ["s1", "t1", "t2"])
    critic = seats["session-critic"].slot
    assert (critic.route, critic.session_target, critic.effort) == (ReviewRouteKind.AGENT_SESSION, "codex=gpt-5.6-sol", "xhigh")
    assert panel.facts["composition"] == ("composed" if mode == "cyber_pro" else "full_pool")

    for names in (["s1", "retired"], ["retired"]):
        with pytest.raises(ReviewChangeArgumentError, match="switched off"):
            commit_gate.compose_commit_panel(ctx, names, REASON)


def test_a_commit_panel_that_names_no_pool_seat_is_refused_before_anything_is_staged(staged_body, tmp_path, monkeypatch):
    from tests.review_pool_rosters import pool_seat

    repo = Path(staged_body["repo"])
    set_review_pool(monkeypatch, shared.golden_pool(pool_seat("scout", "openai/gpt-5.6-sol", effort="high", marked=False)))
    monkeypatch.setattr(git_mod, "get_runtime_mode", lambda: "cyber_pro")
    monkeypatch.setattr(substrate, "run_review_request", lambda *a, **k: pytest.fail("no wave may be paid"))
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "gate-drive")
    shared.git(repo, "reset", "-q", "HEAD")  # the index is the gate's to stage; here nothing may be
    for args, fragment in (({"reviewers": ["scout"], "reason": REASON}, "from the review pool"),
                           ({"reviewers": ["nobody"]}, "not an enabled catalog row")):
        result = git_mod._commit_reviewed(ctx, COMMIT_MESSAGE, **args)
        assert "TOOL_ARG_ERROR" in result and fragment in result and "Nothing was staged" in result, result
        assert shared.git(repo, "diff", "--cached", "--name-only") == ""
    schema = next(entry.schema for entry in git_mod.get_tools() if entry.name == "commit_reviewed")["parameters"]["properties"]
    assert schema["reviewers"]["type"] == "array" and schema["reason"]["type"] == "string"


def test_a_new_round_of_the_same_index_is_a_new_physical_review_not_a_replay(staged_body, tmp_path, monkeypatch):
    """Identities (b) and (c) under the REAL custody layer. The custody layer replays a
    settled attempt to a caller whose attempt key it already holds (same context, same
    retry key, same seats) — so a retry key that ignored the round would hand the
    author's NEW rebuttal the OLD round's answers at $0 of new work. The logical
    round rides the retry key: a new rebuttal, a new goal or new author questions
    buy a new wave; the identical request is the settled record, free."""
    repo = Path(staged_body["repo"])
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "8")  # four paid rounds are bought below
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends))
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "drive", task_id="task-rounds")
    ask = dict(root="system_repo", surface="change", goal=GOAL, scope=SCOPE, subject="index")

    first = run_review_change(ctx, **ask, review_rebuttal="Round one: the finding is stale.")
    second = run_review_change(ctx, **ask, review_rebuttal="Round two: here is the new evidence.")
    assert first["aggregate"] == second["aggregate"] == "PASS", (first, second)
    assert first["reused"] is False and second["reused"] is False
    assert second["record_id"] != first["record_id"]
    # Six physical sends (three seats per round): the second round was SENT, not
    # replayed out of the first round's settled custody — under its own retry key.
    assert sorted(send["slot_id"] for send in sends) == ["s1", "s1", "t1", "t1", "t2", "t2"]
    assert len({send["retry_key"] for send in sends}) == 2 and all(send["retry_key"] for send in sends)
    records = [review_ledger.load_record(ctx.drive_root, result["record_id"]) for result in (first, second)]
    assert records[0]["fingerprints"]["retry_key"] != records[1]["fingerprints"]["retry_key"]
    assert {send["retry_key"] for send in sends} == {record["fingerprints"]["retry_key"] for record in records}
    assert records[0]["fingerprints"]["reuse_key"] != records[1]["fingerprints"]["reuse_key"]
    assert records[0]["subject"] == records[1]["subject"]  # the same bytes, another round
    # The identical round is the settled record — free, and no seat is sent again.
    again = run_review_change(ctx, **ask, review_rebuttal="Round two: here is the new evidence.")
    assert again["reused"] is True and again["record_id"] == second["record_id"] and len(sends) == 6
    assert again["cost"] == {"usd": 0.0, "unknown": False}
    # A changed brief or a question for the reviewers is another round again.
    other_goal = run_review_change(ctx, **{**ask, "goal": "Return the OTHER constant."})
    assert (other_goal["aggregate"], other_goal["reused"], len(sends)) == ("PASS", False, 9), other_goal
    asked = run_review_change(ctx, **ask, author_questions=["Does the helper stay pure?"])
    assert (asked["aggregate"], asked["reused"], len(sends)) == ("PASS", False, 12), asked
    assert len({send["retry_key"] for send in sends}) == 4


def _foreign_project(tmp_path, monkeypatch) -> tuple:
    """The production geometry of a project review: the body and its data under one
    Ouroboros home, the reviewed project elsewhere under the user's files; the project
    has a remote, so its body fact is a recognized foreign root (the core layer)."""
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path))
    set_review_pool(monkeypatch, shared.golden_pool())
    system = Path(shared.init_installed_body(tmp_path / "ouroboros")["repo"])
    project = (tmp_path / "work" / "project").resolve()
    project.mkdir(parents=True)
    shared.git(project, "init", "-q")
    (project / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "lib.py").write_text("LIB = 'base'\n", encoding="utf-8")
    shared.git(project, "add", "-A")
    shared.git(project, "commit", "-q", "-m", "base")
    shared.git(project, "remote", "add", "origin", "https://example.com/third-party/project.git")
    drive = tmp_path / "ouroboros" / "data"
    for sub in ("logs", "locks", "state"):
        (drive / sub).mkdir(parents=True)
    ctx = ToolContext(repo_dir=system, system_repo_dir=system, drive_root=drive, workspace_root=project,
                      workspace_mode="external", task_id="task-project")
    return ctx, project


def _probing_substrate(briefs: list[dict]):
    """The golden paid seam, plus what a retrieving seat would find at call time: the
    staged index of the root it is pointed at (``git diff --cached``) and whether that
    root still exists — the checkout is removed when the wave settles."""
    inner = shared.golden_substrate(briefs)

    def run_review_request(request, **kwargs):
        seen = len(briefs)
        answer = inner(request, **kwargs)
        for brief in briefs[seen:]:
            root = Path(brief["session_root"] or "")
            brief["root_existed"] = bool(brief["session_root"]) and root.is_dir()
            brief["index_at_call"] = shared.git(root, "diff", "--cached") if brief["root_existed"] else ""
        return answer

    return run_review_request


def _frozen_delta(project: Path, parent: str, tree: str) -> str:
    return shared.git(project, "diff", "--no-ext-diff", "--no-textconv", "--no-color", parent, tree)


def test_an_unstaged_worktree_change_is_read_in_its_isolated_checkout_by_every_delivery(tmp_path, monkeypatch):
    """A ``worktree`` subject's change is NOT in any index. Every delivery reads the
    frozen subject: the packet and the scope brief carry the frozen diff, the
    retrieving seats are pointed at an isolated checkout whose index IS the frozen
    patch (the live root's index is empty), and the checkout is gone when the wave
    settles with no custody open."""
    ctx, project = _foreign_project(tmp_path, monkeypatch)
    (project / "app.py").write_text("VALUE = 2  # live edit, never staged\n", encoding="utf-8")
    assert shared.git(project, "diff", "--cached") == ""
    briefs: list[dict] = []
    monkeypatch.setattr(substrate, "run_review_request", _probing_substrate(briefs))

    result = run_review_change(ctx, subject="worktree", goal="Bump the value", scope="app.py only")

    assert result["state"] == "settled" and result["aggregate"] == "PASS", result
    subject = result["subject"]
    checkout = subject["checkout"]
    assert subject["kind"] == "worktree" and subject["root"] == str(project) and checkout
    assert Path(checkout).is_relative_to(ctx.drive_root / "state" / CHECKOUT_SUBDIR)
    frozen_diff = _frozen_delta(project, subject["base"], subject["tree_sha"])
    assert "+VALUE = 2  # live edit, never staged" in frozen_diff
    assert sorted(brief["slot_id"] for brief in briefs) == ["s1", "t1", "t2"]
    for brief in briefs:
        text = _brief_text(brief)
        if brief["slot_id"] == "t2":  # the retrieving triad seat reads the checkout's index
            assert brief["session_root"] == checkout and brief["root_existed"], brief["slot_id"]
            assert brief["index_at_call"].strip() == frozen_diff.strip()
        else:
            assert "+VALUE = 2  # live edit, never staged" in text, brief["slot_id"]
        if brief["slot_id"] == "s1":  # the scope seat retrieves in the checkout too
            assert brief["session_root"] == checkout and brief["index_at_call"].strip() == frozen_diff.strip()
    assert not Path(checkout).exists() and not result["subject"].get("retained_checkout")
    assert shared.git(project, "diff", "--cached") == "" and shared.git(project, "worktree", "list").count("\n") == 0


def test_an_index_against_another_base_delivers_the_frozen_delta_not_the_live_index(tmp_path, monkeypatch):
    """``index`` with ``base`` ≠ HEAD: the subject is parent→index-tree, which also
    carries the commits between the base and HEAD. A recapture of the live index
    (HEAD→index) would lose them; every delivery reads the frozen delta instead."""
    ctx, project = _foreign_project(tmp_path, monkeypatch)
    base = shared.git(project, "rev-parse", "HEAD")
    (project / "lib.py").write_text("LIB = 'committed after the base'\n", encoding="utf-8")
    shared.git(project, "add", "lib.py")
    shared.git(project, "commit", "-q", "-m", "lib")
    (project / "app.py").write_text("VALUE = 3  # staged\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    live_index = shared.git(project, "diff", "--cached")
    assert "committed after the base" not in live_index
    briefs: list[dict] = []
    monkeypatch.setattr(substrate, "run_review_request", _probing_substrate(briefs))

    result = run_review_change(ctx, subject="index", base=base, goal="Both changes", scope="app.py and lib.py")

    assert result["state"] == "settled" and result["aggregate"] == "PASS", result
    subject = result["subject"]
    assert (subject["kind"], subject["base"], subject["tree_sha"]) == ("index", base, shared.git(project, "write-tree"))
    frozen_diff = _frozen_delta(project, base, subject["tree_sha"])
    assert "+LIB = 'committed after the base'" in frozen_diff and "+VALUE = 3  # staged" in frozen_diff
    for brief in briefs:
        text = _brief_text(brief)
        if brief["slot_id"] != "t2":
            assert "+LIB = 'committed after the base'" in text and "+VALUE = 3  # staged" in text, brief["slot_id"]
        if brief["slot_id"] in ("t2", "s1"):
            assert brief["session_root"] == subject["checkout"] and brief["index_at_call"].strip() == frozen_diff.strip()
    assert shared.git(project, "diff", "--cached") == live_index  # the live root is untouched


def test_two_revisions_of_one_tree_are_two_rounds(tmp_path, monkeypatch):
    """The resolved ``base``/``head`` the record names are part of the round: a range
    whose head moved to an empty commit is the same bytes (one ``diff_sha``, one
    ``tree_sha``) but a new wave — the old record is not handed back for revisions
    it never named."""
    ctx, project = _foreign_project(tmp_path, monkeypatch)
    base = shared.git(project, "rev-parse", "HEAD")
    (project / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    shared.git(project, "commit", "-q", "-m", "bump")
    head = shared.git(project, "rev-parse", "HEAD")
    shared.git(project, "commit", "-q", "--allow-empty", "-m", "empty")
    moved = shared.git(project, "rev-parse", "HEAD")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends))

    ask = dict(subject="base..head", base=base, goal="Bump", scope="app.py")
    first = run_review_change(ctx, **ask, head=head)
    second = run_review_change(ctx, **ask, head=moved)
    assert first["aggregate"] == second["aggregate"] == "PASS", (first, second)
    assert (first["subject"]["tree_sha"], first["subject"]["diff_sha"]) == (second["subject"]["tree_sha"], second["subject"]["diff_sha"])
    assert (first["subject"]["head"], second["subject"]["head"]) == (head, moved)
    assert second["reused"] is False and second["record_id"] != first["record_id"] and len(sends) == 6
    assert len({send["retry_key"] for send in sends}) == 2
    assert run_review_change(ctx, **ask, head=moved)["reused"] is True and len(sends) == 6


def _pending_then_answering_seat(seat_id: str, token: str):
    """A delegated seat whose first start outcome is unknown: the executor checkpoints
    the start token and reports the seat in flight (``usage["pending_invocation_id"]``,
    the gate's late-session shape). The rejoin of that exact invocation answers."""
    starts: list[dict] = []

    def answer(request, slot, actor, *, retry_state, pending_invocation_checkpoint):
        if slot.slot_id != seat_id or retry_state.get("pending_invocation_id") == token:
            return actor
        starts.append({"session_root": str(request.session_root or ""), "retry_state": dict(retry_state)})
        pending_invocation_checkpoint(token)
        actor.status, actor.error, actor.raw_text = "error", "delegated start outcome unknown", ""
        actor.usage = {"pending_invocation_id": token}
        return actor

    return answer, starts


def _paid_rows(drive: Path, project: Path) -> list:
    from ouroboros.review_state import load_state, make_repo_key

    return [row for row in load_state(drive).filter_attempts(repo_key=make_repo_key(project), tool_name="review_change")
            if row.paid]


def test_a_rerun_of_a_pending_round_rejoins_its_operation_and_passes_a_reached_ceiling(tmp_path, monkeypatch):
    """S2-2, in one process. A wave whose delegated seat is still running settles
    ``pending`` and keeps its checkout. The identical request then COLLECTS that
    operation: the checkout is the same path (the round's), the custody attempt key
    matches, the settled seats replay, the pending seat is rejoined by its exact
    invocation, the SAME record is revised to its verdict, the one paid attempt row
    closes — and all of it with the per-task cycle ceiling already reached, which
    meets only a NEW paid wave. Nothing is paid for twice."""
    ctx, project = _foreign_project(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")  # the first wave reaches it
    (project / "app.py").write_text("VALUE = 2  # staged\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    sends: list[dict] = []
    answer, starts = _pending_then_answering_seat("t2", "invocation-t2-round-1")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    ask = dict(subject="index", goal="Bump", scope="app.py")

    first = run_review_change(ctx, **ask)
    assert first["state"] == "pending" and first["reused"] is False, first
    checkout = Path(first["subject"]["checkout"])
    assert first["subject"]["retained_checkout"] == str(checkout) and checkout.is_dir()
    assert first["subject"]["retention"].get("seats") == ["t2"], first["subject"]["retention"]
    assert sorted(send["slot_id"] for send in sends) == ["s1", "t1", "t2"] and len(starts) == 1
    rows = _paid_rows(ctx.drive_root, project)
    assert len(rows) == 1 and rows[0].late_result_pending and rows[0].review_record_id == first["record_id"]
    pending_rows = [row for row in rows[0].triad_raw_results if row.get("slot_id") == "t2"]
    assert pending_rows and pending_rows[0].get("pending_invocation_id") == "invocation-t2-round-1"  # checkpointed

    rerun = run_review_change(ctx, **ask)
    assert (rerun["state"], rerun["aggregate"], rerun["reused"]) == ("settled", "PASS", False), rerun
    assert rerun["record_id"] == first["record_id"]
    record = review_ledger.load_record(ctx.drive_root, rerun["record_id"])
    assert record["revision"] == 2 and record["dispatch_refusal"] is None
    assert rerun["subject"]["checkout"] == str(checkout)  # the round's path, not a new one
    # One more physical act only: the exact rejoin of the pending invocation, in the
    # retained checkout; the settled seats were not sent again.
    assert len(sends) == 4 and sends[3]["slot_id"] == "t2" and sends[3]["reconcile_only"] is True
    assert sends[3]["retry_state"] == {"pending_invocation_id": "invocation-t2-round-1"}
    first_t2 = next(send for send in sends[:3] if send["slot_id"] == "t2")
    assert sends[3]["session_root"] == first_t2["session_root"] == str(checkout)
    assert sends[3]["retry_key"] == sends[0]["retry_key"]
    rows = _paid_rows(ctx.drive_root, project)
    assert len(rows) == 1 and not rows[0].late_result_pending and rows[0].review_record_id == rerun["record_id"]
    assert not checkout.exists() and not rerun["subject"].get("retained_checkout")  # custody closed
    # The ceiling stands for a NEW paid wave of this task tree.
    other = run_review_change(ctx, **{**ask, "goal": "Another brief"})
    assert other["aggregate"] == "NOT_DISPATCHED" and other["dispatch_refusal"]["kind"] == "review_cycles_exhausted"
    assert len(sends) == 4


def test_a_rerun_after_a_restart_rejoins_the_pending_round_from_durable_state(tmp_path, monkeypatch):
    """S2-2 after a restart: a FRESH context (no process-local custody) rerunning the
    same round finds the pending attempt row, its checkpointed invocation and the
    settled seats' durable producer outcomes, at the same checkout path — and settles
    the same record. A random checkout path would be a different attempt key: the
    bindings would not match and the paid operation would be lost, not collected."""
    ctx, project = _foreign_project(tmp_path, monkeypatch)
    (project / "app.py").write_text("VALUE = 2  # staged\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    sends: list[dict] = []
    answer, starts = _pending_then_answering_seat("t2", "invocation-t2-restart")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    ask = dict(subject="index", goal="Bump", scope="app.py")

    first = run_review_change(ctx, **ask)
    assert first["state"] == "pending", first
    checkout = Path(first["subject"]["checkout"])
    assert checkout.is_dir() and len(sends) == 3

    restarted = ToolContext(repo_dir=ctx.repo_dir, system_repo_dir=ctx.system_repo_dir, drive_root=ctx.drive_root,
                            workspace_root=project, workspace_mode="external", task_id=ctx.task_id)
    rerun = run_review_change(restarted, **ask)
    assert (rerun["state"], rerun["aggregate"], rerun["reused"]) == ("settled", "PASS", False), rerun
    assert rerun["record_id"] == first["record_id"] and rerun["subject"]["checkout"] == str(checkout)
    assert review_ledger.load_record(ctx.drive_root, rerun["record_id"])["revision"] == 2
    assert len(sends) == 4 and sends[3]["slot_id"] == "t2" and sends[3]["session_root"] == str(checkout)
    assert sends[3]["retry_state"] == {"pending_invocation_id": "invocation-t2-restart"} and len(starts) == 1
    rows = _paid_rows(ctx.drive_root, project)
    assert len(rows) == 1 and not rows[0].late_result_pending
    assert {row["slot_id"]: (row["operation_state"], bool(row.get("late_result_pending")))
            for row in rows[0].triad_raw_results} == {"t1": ("settled", False), "t2": ("settled", False),
                                                        "s1": ("settled", False)}
    assert not checkout.exists()


class _ProcessDied(BaseException):
    """The review process dies mid-wave: nothing settles, no cleanup of ours runs."""


def _dies_right_after_the_paid_stamp(monkeypatch):
    """The process is lost the moment the wave's paid row is durable: the row says
    ``reviewing`` with its seats reserved and tokenless, and no seat ever answers."""
    import ouroboros.tools.review_change as review_change_mod
    from ouroboros.review_dispatch import ReviewPaidStamp

    real = review_change_mod.install_paid_stamp

    def install(ctx, wave):
        holder = real(ctx, wave)
        write = ctx._review_paid_stamp

        def write_then_die() -> None:
            write()
            raise _ProcessDied()

        ctx._review_paid_stamp = ReviewPaidStamp(write_then_die, fail_closed=True)
        return holder

    monkeypatch.setattr(review_change_mod, "install_paid_stamp", install)


def _restart(monkeypatch, *, dead_pid: int):
    """The next server generation: a new custody session, the old process proven dead,
    and the process-local registries of the old one gone with it."""
    import ouroboros.platform_layer as platform_layer
    import ouroboros.process_custody as process_custody
    from ouroboros import review_custody

    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "next-generation")
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: int(pid) != dead_pid)
    with review_custody._ACTIVE_LOCK:
        review_custody._ACTIVE.clear()
        review_custody._NO_RESEND.clear()


def test_a_wave_lost_to_process_death_is_closed_at_startup_and_a_new_wave_may_pay(tmp_path, monkeypatch):
    """D2-02. The paid attempt row of a ``review_change`` wave is bound to the process
    that pays it, as the gate's rows are. When that process dies mid-wave (a tokenless
    seat still reserved), the next generation's startup reconciliation proves the owner
    dead and closes the row as an infra failure — so a rerun of the round pays a NEW
    wave instead of forever collecting an open operation nobody can finish."""
    import os

    from ouroboros.review_owner_custody import reconcile_review_custody_on_process_start

    ctx, project = _foreign_project(tmp_path, monkeypatch)
    (project / "app.py").write_text("VALUE = 2  # staged\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends))
    ask = dict(subject="index", goal="Bump", scope="app.py")
    with monkeypatch.context() as dying:
        _dies_right_after_the_paid_stamp(dying)
        with pytest.raises(_ProcessDied):
            run_review_change(ctx, **ask)
    assert sends == []  # paid, then lost before any seat was sent
    rows = _paid_rows(ctx.drive_root, project)
    assert len(rows) == 1 and rows[0].status == "reviewing", rows
    assert all(not row.get("pending_invocation_id") for row in rows[0].triad_raw_results)  # tokenless
    lost_record = rows[0].review_record_id

    _restart(monkeypatch, dead_pid=os.getpid())
    outcome = reconcile_review_custody_on_process_start(ctx.drive_root)
    assert [row.review_record_id for row in outcome["reconciled"]] == [lost_record], outcome
    rows = _paid_rows(ctx.drive_root, project)
    assert (rows[0].status, rows[0].block_reason, rows[0].late_result_pending) == ("failed", "infra_failure", False)

    restarted = ToolContext(repo_dir=ctx.repo_dir, system_repo_dir=ctx.system_repo_dir, drive_root=ctx.drive_root,
                            workspace_root=project, workspace_mode="external", task_id=ctx.task_id)
    rerun = run_review_change(restarted, **ask)
    assert (rerun["state"], rerun["aggregate"], rerun["reused"]) == ("settled", "PASS", False), rerun
    assert rerun["record_id"] != lost_record
    assert sorted(send["slot_id"] for send in sends) == ["s1", "t1", "t2"]  # a new wave, nothing rejoined
    assert all(not send["reconcile_only"] for send in sends)
    rows = _paid_rows(ctx.drive_root, project)
    assert [row.status for row in rows] == ["failed", "reviewed"]


def test_the_same_restart_keeps_a_tokened_pending_round_for_its_exact_rejoin(tmp_path, monkeypatch):
    """Control for the owner stamp: a round whose delegated seat holds a durable start
    token is NOT closed by the dead owner's reconciliation — the token is the recoverable
    fact — and the restarted process rejoins exactly it."""
    import os

    from ouroboros.review_owner_custody import reconcile_review_custody_on_process_start

    ctx, project = _foreign_project(tmp_path, monkeypatch)
    (project / "app.py").write_text("VALUE = 2  # staged\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    sends: list[dict] = []
    answer, starts = _pending_then_answering_seat("t2", "invocation-t2-owner-died")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    ask = dict(subject="index", goal="Bump", scope="app.py")

    first = run_review_change(ctx, **ask)
    assert first["state"] == "pending" and len(sends) == 3, first
    rows = _paid_rows(ctx.drive_root, project)
    assert rows[0].review_owner_pid == os.getpid() and rows[0].late_result_pending

    _restart(monkeypatch, dead_pid=os.getpid())
    assert reconcile_review_custody_on_process_start(ctx.drive_root)["reconciled"] == []
    rows = _paid_rows(ctx.drive_root, project)
    assert rows[0].late_result_pending and rows[0].review_record_id == first["record_id"]

    restarted = ToolContext(repo_dir=ctx.repo_dir, system_repo_dir=ctx.system_repo_dir, drive_root=ctx.drive_root,
                            workspace_root=project, workspace_mode="external", task_id=ctx.task_id)
    rerun = run_review_change(restarted, **ask)
    assert (rerun["state"], rerun["aggregate"], rerun["record_id"]) == ("settled", "PASS", first["record_id"]), rerun
    assert len(sends) == 4 and sends[3]["slot_id"] == "t2" and sends[3]["reconcile_only"] is True
    assert sends[3]["retry_state"] == {"pending_invocation_id": "invocation-t2-owner-died"} and len(starts) == 1
    rows = _paid_rows(ctx.drive_root, project)
    assert len(rows) == 1 and not rows[0].late_result_pending


def _pending_once_seat(seat_id: str, token: str):
    """The FIRST start of ``seat_id`` goes in flight with an unknown outcome (task A's
    wave); every later new start answers at once (another task's wave of the same
    round), and the exact rejoin of ``token`` answers."""
    starts: list[dict] = []

    def answer(request, slot, actor, *, retry_state, pending_invocation_checkpoint):
        if slot.slot_id != seat_id or retry_state.get("pending_invocation_id") == token or starts:
            return actor
        starts.append({"session_root": str(request.session_root or ""), "retry_state": dict(retry_state)})
        pending_invocation_checkpoint(token)
        actor.status, actor.error, actor.raw_text = "error", "delegated start outcome unknown", ""
        actor.usage = {"pending_invocation_id": token}
        return actor

    return answer, starts


def test_another_task_on_the_same_round_never_removes_a_pending_tasks_checkout(tmp_path, monkeypatch):
    """Pending custody is per task, and so is the checkout. Task A's wave settles
    ``pending`` and keeps its checkout; task B asks the identical round, pays its own
    wave in ITS own checkout, settles and removes only that one. A's checkout stays
    for A's reviewer, and A's rerun collects its own operation there instead of
    taking B's settled record of the same round as a reuse."""
    ctx, project = _foreign_project(tmp_path, monkeypatch)
    (project / "app.py").write_text("VALUE = 2  # staged\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    sends: list[dict] = []
    answer, starts = _pending_once_seat("t2", "invocation-t2-task-a")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    ask = dict(subject="index", goal="Bump", scope="app.py")

    first = run_review_change(ctx, **ask)
    assert first["state"] == "pending", first
    checkout_a = Path(first["subject"]["checkout"])
    assert checkout_a.is_dir() and len(sends) == 3

    task_b = ToolContext(repo_dir=ctx.repo_dir, system_repo_dir=ctx.system_repo_dir, drive_root=ctx.drive_root,
                         workspace_root=project, workspace_mode="external", task_id="task-other")
    other = run_review_change(task_b, **ask)
    assert (other["state"], other["aggregate"], other["reused"]) == ("settled", "PASS", False), other
    checkout_b = Path(other["subject"]["checkout"])
    assert checkout_b != checkout_a and not checkout_b.exists()  # B's own path, removed when B settled
    assert checkout_a.is_dir()  # A's reviewer still reads here
    roots_b = {send["session_root"] for send in sends[3:]} - {""}  # packet seats carry no root
    assert len(sends) == 6 and roots_b == {str(checkout_b)}

    rerun = run_review_change(ctx, **ask)
    assert (rerun["state"], rerun["aggregate"], rerun["reused"]) == ("settled", "PASS", False), rerun
    assert rerun["record_id"] == first["record_id"] != other["record_id"]
    assert len(sends) == 7 and sends[6]["slot_id"] == "t2" and sends[6]["reconcile_only"] is True
    assert sends[6]["session_root"] == str(checkout_a) and len(starts) == 1
    assert sends[6]["retry_state"] == {"pending_invocation_id": "invocation-t2-task-a"}
    assert not checkout_a.exists()


def test_a_failed_custody_read_keeps_the_retained_checkout_and_the_next_rerun_rejoins(tmp_path, monkeypatch):
    """The rerun of a pending round cannot read its custody (the review state lock
    times out): it fails before any dispatch, and the checkout the earlier wave
    retained stays, because its reviewer still reads it. The next rerun rejoins the
    exact invocation there. A FIRST wave failing the same way leaves no checkout."""
    ctx, project = _foreign_project(tmp_path, monkeypatch)
    (project / "app.py").write_text("VALUE = 2  # staged\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    sends: list[dict] = []
    answer, starts = _pending_then_answering_seat("t2", "invocation-t2-unreadable")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    ask = dict(subject="index", goal="Bump", scope="app.py")

    def unreadable(*_args, **_kwargs):
        raise TimeoutError("review state lock timed out")

    first = run_review_change(ctx, **ask)
    assert first["state"] == "pending", first
    checkout = Path(first["subject"]["checkout"])
    with monkeypatch.context() as failing:
        failing.setattr(review_change, "pending_round_attempt", unreadable)
        with pytest.raises(TimeoutError):
            run_review_change(ctx, **ask)
    assert checkout.is_dir() and len(sends) == 3  # nothing dispatched; the reviewer's checkout kept

    rerun = run_review_change(ctx, **ask)
    assert (rerun["state"], rerun["aggregate"]) == ("settled", "PASS"), rerun
    assert rerun["record_id"] == first["record_id"] and len(sends) == 4 and len(starts) == 1
    assert sends[3]["session_root"] == str(checkout)
    assert sends[3]["retry_state"] == {"pending_invocation_id": "invocation-t2-unreadable"}
    assert not checkout.exists()

    checkouts = Path(ctx.drive_root) / "state" / CHECKOUT_SUBDIR
    with monkeypatch.context() as failing:
        failing.setattr(review_change, "pending_round_attempt", unreadable)
        with pytest.raises(TimeoutError):
            run_review_change(ctx, **{**ask, "goal": "Another brief"})
    assert len(sends) == 4 and not [path for path in checkouts.iterdir()]  # a fresh checkout is not kept


SERVING_RULE = "SERVING-CONSTITUTION-MARKER: the rule that is running."
CANDIDATE_RULE = "CANDIDATE-CONSTITUTION-MARKER: the candidate rewrote its own rule."


def test_a_bound_body_candidate_is_the_subject_and_the_serving_body_is_the_governance(tmp_path, monkeypatch):
    """A task authoring its own body writes a bound candidate worktree
    (``body_candidate.bind`` points ``repo_dir``/``system_repo_dir`` at it). The
    candidate is the SUBJECT of ``review_change``; the rules every seat reads —
    constitution, handbook, navigation — come from the SERVING checkout
    (``review_substrate.review_repo_dirs_for``'s rule), never from the candidate's
    own copy, so a candidate cannot be judged by the rule it rewrote."""
    from ouroboros import body_candidate

    fixture = shared.init_installed_body(tmp_path)
    serving = Path(fixture["repo"])
    (serving / "BIBLE.md").write_text(f"# Constitution\n\n{SERVING_RULE}\n", encoding="utf-8")
    shared.git(serving, "add", "BIBLE.md")
    shared.git(serving, "commit", "-q", "-m", "constitution")
    head = shared.git(serving, "rev-parse", "HEAD")
    candidate = (tmp_path / "candidates" / "c1").resolve()
    candidate.parent.mkdir()
    shared.git(serving, "worktree", "add", "-q", "-b", "candidate/c1", str(candidate), head)
    (candidate / "BIBLE.md").write_text(f"# Constitution\n\n{CANDIDATE_RULE}\n", encoding="utf-8")
    shared.git(candidate, "add", "BIBLE.md")
    shared.git(candidate, "commit", "-q", "-m", "the candidate rewrites its rule")
    (candidate / "ouroboros" / "config.py").write_text("FIXTURE = 'installed'\nCANDIDATE_CHANGE = 'reviewed on the candidate'\n",
                                                        encoding="utf-8")
    shared.git(candidate, "add", "ouroboros/config.py")

    set_review_pool(monkeypatch, shared.golden_pool())
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "1")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.setattr(git_mod, "_run_review_preflight_tests", shared.passing_test_runner)
    briefs: list[dict] = []
    monkeypatch.setattr(substrate, "run_review_request", shared.golden_substrate(briefs))
    tiered: list[tuple[str, str]] = []  # (delivery, governance root) the triad tiered its rules from
    real_tiering = review_mod._triad_governance_context

    def observed_tiering(ctx, touched_paths, checklist_section, api_models, api_slots, **kwargs):
        tiered.append((str(kwargs.get("delivery", "packet")), str(kwargs.get("governance_root") or "")))
        return real_tiering(ctx, touched_paths, checklist_section, api_models, api_slots, **kwargs)

    monkeypatch.setattr(review_mod, "_triad_governance_context", observed_tiering)
    ctx = ToolContext(repo_dir=serving, system_repo_dir=serving, drive_root=tmp_path / "drive", task_id="task-candidate")
    body_candidate.bind(ctx, {"candidate_id": "c1", "path": str(candidate), "branch": "candidate/c1",
                              "base_sha": head, "repo_dir": str(serving)})
    assert body_candidate.is_bound(ctx) and Path(ctx.repo_dir) == candidate

    result = run_review_change(ctx, root="system_repo", surface="change", goal=GOAL, scope=SCOPE, subject="index")

    assert result["aggregate"] == "PASS" and result["state"] == "settled", result
    record = review_ledger.load_record(ctx.drive_root, result["record_id"])
    # The candidate is the subject (its staged index, read as the body's own index) ...
    assert (result["subject"]["root_kind"], result["subject"]["root"]) == ("system_repo", str(candidate))
    assert result["subject"]["tree_sha"] == shared.git(candidate, "write-tree")
    assert result["checklist"] == {"layer": "body", "body_fact": "true", "how": "git_common_dir", "treat_as_body": False}
    # ... and the serving body is the governance, on the record and in every delivery.
    assert result["subject"]["governance_root"] == record["subject"]["governance_root"] == str(serving.resolve())
    assert sorted(brief["slot_id"] for brief in briefs) == ["s1", "t1", "t2"]
    # The packet seat's constitutional head is the RUNNING body's by construction: the
    # tiers it selects from are the serving copy; the retrieving seats' briefs inline
    # the serving copy's rules directly (checked below: no candidate rule reaches a seat).
    assert sorted(tiered) == [("packet", str(serving.resolve()))]
    for brief in briefs:
        text = _brief_text(brief)
        assert CANDIDATE_RULE not in text, brief["slot_id"]
        if brief["slot_id"] == "t2":  # the retrieving seat reads the candidate's tree under the serving rules
            assert SERVING_RULE in text and brief["session_root"] == str(candidate)
        else:
            assert "+CANDIDATE_CHANGE" in text, brief["slot_id"]
        if brief["slot_id"] == "s1":
            assert SERVING_RULE in text
    assert shared.git(candidate, "write-tree") == result["subject"]["tree_sha"]  # read only
    assert shared.git(serving, "status", "--porcelain") == ""

    # Unbound, the governance and the subject are the one system repository, as before.
    plain = ToolContext(repo_dir=serving, system_repo_dir=serving, drive_root=tmp_path / "plain-drive", task_id="task-plain")
    spec = review_change.ReviewSubjectSpec(root_kind="system_repo", root=str(serving), kind="index")
    frozen = review_change.freeze_subject(plain, spec)
    assert frozen.spec.governance_root == str(serving.resolve()) and review_change._governance_repo(plain) == serving.resolve()


PROTOCOL_CHAPTER = "docs/development/05-review-and-commit-protocol.md"
SERVING_DEV_RULE = "SERVING-HANDBOOK-MARKER: every commit of the body is reviewed by the whole pool."
CANDIDATE_DEV_RULE = "CANDIDATE-HANDBOOK-MARKER: the candidate relaxed the protocol it is judged by."


def _handbook(repo: Path, rule: str) -> None:
    (repo / "docs" / "development").mkdir(parents=True, exist_ok=True)
    (repo / PROTOCOL_CHAPTER).write_text(f"# Protocol\n\nAn authored introduction.\n\n## Protocol rules\n\n{rule}\n",
                                         encoding="utf-8", newline="\n")
    (repo / "docs" / "DEVELOPMENT.md").write_text(
        "# Development\n\nThe handbook entrypoint.\n\n## Chapters\n\n"
        "- [05-review-and-commit-protocol.md](development/05-review-and-commit-protocol.md)\n",
        encoding="utf-8", newline="\n")


def test_the_commit_gate_judges_a_bound_candidate_by_the_serving_handbook_and_records_that_root(tmp_path, monkeypatch):
    """The same binding under the COMMIT GATE (``commit_reviewed`` without a frozen
    subject): a candidate that rewrites the review-protocol chapter of the handbook is
    judged — on every delivery, packet and retrieving — by the SERVING body's chapter,
    its own rewrite reaching the seats only as the diff under review; and the record
    names the serving body as the governance root, recognized through the predicate
    (``git_common_dir``), not the candidate judged against itself (``dir``)."""
    from ouroboros import body_candidate

    fixture = shared.init_installed_body(tmp_path)
    serving = Path(fixture["repo"])
    (serving / "BIBLE.md").write_text(f"# Constitution\n\n{SERVING_RULE}\n", encoding="utf-8")
    (serving / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")  # as a real install carries one
    _handbook(serving, SERVING_DEV_RULE)
    shared.git(serving, "add", "-A")
    shared.git(serving, "commit", "-q", "-m", "constitution and handbook")
    head = shared.git(serving, "rev-parse", "HEAD")
    candidate = (tmp_path / "candidates" / "c1").resolve()
    candidate.parent.mkdir()
    shared.git(serving, "worktree", "add", "-q", "-b", "candidate/c1", str(candidate), head)
    _handbook(candidate, CANDIDATE_DEV_RULE)
    shared.git(candidate, "add", "-A")

    set_review_pool(monkeypatch, shared.golden_pool())
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "1")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.setattr(git_mod, "_run_review_preflight_tests", shared.passing_test_runner)
    briefs: list[dict] = []
    monkeypatch.setattr(substrate, "run_review_request", shared.golden_substrate(briefs))
    ctx = ToolContext(repo_dir=serving, system_repo_dir=serving, drive_root=tmp_path / "drive", task_id="task-candidate")
    body_candidate.bind(ctx, {"candidate_id": "c1", "path": str(candidate), "branch": "candidate/c1",
                              "base_sha": head, "repo_dir": str(serving)})

    outcome = _run_non_committing_review_cycle(ctx, COMMIT_MESSAGE, skip_advisory_review=True, goal=GOAL, scope=SCOPE)

    assert outcome["status"] == "passed", outcome
    record = review_ledger.load_record(ctx.drive_root, outcome["review_record_id"])
    assert record["subject"]["governance_root"] == str(serving.resolve())
    checklist = record["brief"]["checklist"]
    assert (checklist["layer"], checklist["body_fact"], checklist["how"]) == ("body", "true", "git_common_dir")
    texts = {brief["slot_id"]: _brief_text(brief) for brief in briefs}
    assert sorted(texts) == ["s1", "t1", "t2"]
    for slot_id, text in texts.items():
        # The chapter as the governance tiers deliver it (under its `## <path>` heading) is
        # the serving body's; the candidate's rewrite reaches the seat as the change under
        # review (the diff, and the packet's changed-file context), never as a rule.
        _, heading, delivered = text.partition(f"\n## {PROTOCOL_CHAPTER}\n")
        assert heading and delivered.index(SERVING_DEV_RULE) < delivered.index(CANDIDATE_DEV_RULE), slot_id
        assert "-" + SERVING_DEV_RULE in text and "+" + CANDIDATE_DEV_RULE in text, slot_id
        assert text.count(SERVING_DEV_RULE) == 2, slot_id  # the delivered rule and the diff's removed line
        if slot_id != "t1":  # the retrieving seats inline the serving constitution too
            assert SERVING_RULE in text, slot_id
    assert shared.git(serving, "status", "--porcelain") == ""
