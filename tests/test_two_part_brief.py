"""One brief, two parts (PR-3 B): ``review_admission.build_two_part_brief``.

The pure builder renders the brief ONE seat would receive for a frozen subject,
callable outside the gate (step R). A retrieving seat is asked both questions —
Part 1 the change, Part 2 the coupling — and answers contract B; a packet seat
is asked the change alone and answers contract A; a coupling-only seat answers
the coupling alone. Every brief carries exactly one anti pattern-lock guard and
the author's questions ride the goal section.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.review_ledger import PART_CHANGE, PART_COUPLING, seat_parts
from ouroboros.tools.review_admission import build_two_part_brief
from ouroboros.tools.review_brief_coupling import COUPLING_CHECKLIST_SECTION, answer_format_section
from ouroboros.tools.review_multi_model import TRIAD_USER_TURN
from ouroboros.tools.review_subject import ReviewSubjectSpec, freeze_subject
from ouroboros.triad_review import REVIEW_JSON_ARRAY_CONTRACT, REVIEW_TWO_PART_OBJECT_CONTRACT

GUARD_HEADING = "Before returning, challenge the behavior promised"


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)


def _subject(tmp_path):
    repo = tmp_path / "system"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    (repo / "mod.py").write_text("def f():\n    return 2\n", encoding="utf-8")
    _git(repo, "add", "-A")
    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True)
    ctx = SimpleNamespace(repo_dir=str(repo), drive_root=str(data), task_id="t-brief", task_metadata=None,
                          _review_history=[], _review_advisory=[], _review_iteration_count=0,
                          drive_logs=lambda: data / "logs")
    spec = ReviewSubjectSpec(root_kind="system_repo", root=str(repo), kind="index", surface="commit_gate", layer="body")
    return freeze_subject(ctx, spec), repo


RETRIEVING = {"slot_id": "seat-native", "model": "openai/gpt-x", "route": "api_chat", "retrieves": True}
PACKET = {"slot_id": "seat-packet", "model": "openai/gpt-y", "route": "api_chat", "retrieves": False}


def test_seat_parts_follow_the_one_fact():
    assert seat_parts(RETRIEVING) == (PART_CHANGE, PART_COUPLING)
    assert seat_parts(PACKET) == (PART_CHANGE,)
    assert seat_parts(RETRIEVING, coupling_only=True) == (PART_COUPLING,)
    slot = SimpleNamespace(retrieves=True, route=None, subagent_id="")
    assert seat_parts(slot) == (PART_CHANGE, PART_COUPLING)


def test_seat_parts_without_the_derived_property_use_the_delivery_predicate_not_the_id():
    """A slot-like object that lacks ``retrieves`` is classified by the ONE delivery
    predicate over its route and its own native-delivery fact; its catalog id is not
    a signal (F8) — a packet api row with an id is still asked ``change`` alone."""
    from ouroboros.review_execution import ReviewRouteKind

    def seat(route, native):
        return SimpleNamespace(route=route, native_retrieval=native, subagent_id="api-critic")

    assert seat_parts(seat(ReviewRouteKind.API_CHAT, True)) == (PART_CHANGE, PART_COUPLING)
    assert seat_parts(seat(ReviewRouteKind.API_CHAT, False)) == (PART_CHANGE,)
    assert seat_parts(seat(ReviewRouteKind.API_CHAT, None)) == (PART_CHANGE,)
    assert seat_parts(seat(ReviewRouteKind.AGENT_SESSION, None)) == (PART_CHANGE, PART_COUPLING)
    assert seat_parts(SimpleNamespace(route="api_chat", subagent_id="api-critic")) == (PART_CHANGE,)


def test_retrieving_seat_gets_both_parts_and_contract_b(tmp_path):
    frozen, _repo = _subject(tmp_path)
    brief = build_two_part_brief(frozen, RETRIEVING, goal="raise f", commit_message="bump f",
                                 author_questions=["Is the new value covered by a test?"])

    assert brief["parts"] == [PART_CHANGE, PART_COUPLING] and brief["delivery"] == "retrieving"
    assert brief["user"] == TRIAD_USER_TURN
    system = brief["system"]
    assert "## Part 1 — The change" in system and "## Part 2" in system
    assert "## Staged diff" in system and "return 2" in system
    assert COUPLING_CHECKLIST_SECTION.split(" / ")[0] in system or "coupling" in system.lower()
    # Contract B is the answer format of a seat asked both questions.
    assert REVIEW_TWO_PART_OBJECT_CONTRACT.strip() in system
    # The author's questions ride the goal as the ONE owner renders them for every
    # door (review_change, the gate, this builder): numbered, as asked.
    assert "Author questions (answer each as asked):\n1. Is the new value covered by a test?" in system
    # Exactly ONE anti pattern-lock guard per brief.
    assert system.count(GUARD_HEADING) == 1
    sha = brief["sha"]
    assert sha["brief"] == hashlib.sha256(system.encode("utf-8")).hexdigest()
    assert sha["change_prompt_sha"] and sha["coupling_brief_sha"] and sha["change_prompt_sha"] != sha["coupling_brief_sha"]
    manifest = brief["manifest"]
    assert manifest["parts"] == [PART_CHANGE, PART_COUPLING] and manifest["diff_delivery"] == "inline"
    assert manifest["sha"] == sha and manifest["first_send_bound"] > 0


def test_builder_is_pure_over_the_frozen_subject(tmp_path):
    frozen, _repo = _subject(tmp_path)
    one = build_two_part_brief(frozen, RETRIEVING, goal="raise f", commit_message="bump f")
    two = build_two_part_brief(frozen, RETRIEVING, goal="raise f", commit_message="bump f")
    assert one["sha"] == two["sha"] and one["system"] == two["system"]
    # A different intent is a different brief — the sha is the brief's, not the seat's:
    # Part 1 carries the intent, so its sha moves; Part 2 is the coupling question over
    # the same subject's tree and the same history, so its sha does not.
    other = build_two_part_brief(frozen, RETRIEVING, goal="lower f", commit_message="bump f")
    assert other["sha"]["brief"] != one["sha"]["brief"]
    assert other["sha"]["change_prompt_sha"] != one["sha"]["change_prompt_sha"]
    assert other["sha"]["coupling_brief_sha"] == one["sha"]["coupling_brief_sha"] != ""


def test_packet_seat_gets_the_change_alone_and_contract_a(tmp_path):
    frozen, _repo = _subject(tmp_path)
    brief = build_two_part_brief(frozen, PACKET, goal="raise f", commit_message="bump f")

    assert brief["parts"] == [PART_CHANGE] and brief["delivery"] == "packet"
    assert brief["user"] == TRIAD_USER_TURN and brief["stable_prefix_len"] > 0
    system = brief["system"]
    assert "## Part 2" not in system and REVIEW_TWO_PART_OBJECT_CONTRACT.strip() not in system
    assert REVIEW_JSON_ARRAY_CONTRACT.strip() in system
    assert "## Staged diff" in system and "return 2" in system
    assert system.count(GUARD_HEADING) == 1
    assert brief["sha"]["coupling_brief_sha"] == "" and brief["sha"]["change_prompt_sha"] == brief["sha"]["brief"]


def test_coupling_only_seat_answers_the_coupling_alone(tmp_path):
    frozen, _repo = _subject(tmp_path)
    brief = build_two_part_brief(frozen, RETRIEVING, goal="raise f", commit_message="bump f", coupling_only=True)

    assert brief["parts"] == [PART_COUPLING]
    system = brief["system"]
    assert "## Part 2" in system and system.count(GUARD_HEADING) == 1
    assert answer_format_section((PART_COUPLING,)).strip() in system
    assert "This seat is asked Part 2 ONLY" in answer_format_section((PART_COUPLING,))


def test_answer_format_names_the_parts_asked():
    both = answer_format_section((PART_CHANGE, PART_COUPLING))
    assert REVIEW_TWO_PART_OBJECT_CONTRACT.strip() in both
    change_only = answer_format_section((PART_CHANGE,))
    assert REVIEW_JSON_ARRAY_CONTRACT.strip() in change_only and REVIEW_TWO_PART_OBJECT_CONTRACT.strip() not in change_only


def test_slot_object_seat_is_accepted(tmp_path):
    frozen, _repo = _subject(tmp_path)
    seat = SimpleNamespace(slot_id="seat-obj", model="openai/gpt-x", route=SimpleNamespace(value="api_chat"),
                           retrieves=True, subagent_id="", session_profile="", use_local=None)
    brief = build_two_part_brief(frozen, seat, goal="raise f", commit_message="bump f")
    assert brief["parts"] == [PART_CHANGE, PART_COUPLING]


def test_packet_fit_failure_is_a_typed_error(tmp_path, monkeypatch):
    from ouroboros.tools import review as _facade, review_admission

    # The gate facade binds its own alias of the fit ladder at its first import (``review._fit_triad_prompt``): import
    # it before the replacement, or a first import here would keep the refusing stub for every later test.
    assert _facade._fit_triad_prompt is review_admission.fit_triad_prompt
    frozen, _repo = _subject(tmp_path)
    monkeypatch.setattr(review_admission, "fit_triad_prompt", lambda *a, **k: ("", 0, "pack does not fit"))
    with pytest.raises(ValueError, match="pack does not fit"):
        build_two_part_brief(frozen, PACKET, goal="raise f", commit_message="bump f")


# --- what a tool-less packet seat receives, against the pre-PR target ------------------------------------------


def _installed_and_subject(tmp_path, *, manifest=True):
    """An installed body (the governance root) and a clone of it whose index is the reviewed subject."""
    installed = tmp_path / "installed"
    (installed / "ouroboros").mkdir(parents=True)
    _git(installed, "init", "-q")
    for key, value in (("user.email", "t@example.com"), ("user.name", "t"), ("commit.gpgsign", "false")):
        _git(installed, "config", key, value)
    leads = {"ouroboros/a.py": "Installed lead of a.", "ouroboros/b.py": "Lead of b.", "ouroboros/c.py": "Lead of c."}
    for path, lead in leads.items():
        (installed / path).write_text(f'"""{lead}"""\n', encoding="utf-8")
    if manifest:
        (installed / "ouroboros" / "domains.toml").write_text(
            '[domains]\nD06 = "Review stack"\n\n[references]\nD06 = ["docs/architecture/06-4.md"]\n\n[modules]\n'
            + "".join(f'"{path}" = "D06"\n' for path in leads), encoding="utf-8")
    _git(installed, "add", "-A")
    _git(installed, "commit", "-q", "-m", "installed")
    subject = tmp_path / "subject"
    _git(tmp_path, "clone", "-q", str(installed), str(subject))
    for key, value in (("user.email", "t@example.com"), ("user.name", "t"), ("commit.gpgsign", "false")):
        _git(subject, "config", key, value)
    return installed, subject


def test_the_synchronous_gate_reads_the_same_structured_set_from_the_staged_index(tmp_path):
    """No frozen subject: the staged index against HEAD, NUL-decoded, rename sides kept, names unquoted."""
    from ouroboros.tools.review_subject import subject_changes

    _installed, subject = _installed_and_subject(tmp_path)
    (subject / "ouroboros" / "a.py").write_text('"""Changed."""\n', encoding="utf-8")
    _git(subject, "mv", "ouroboros/c.py", "ouroboros/c\u00e9.py")
    _git(subject, "add", "-A")
    assert subject_changes(str(subject)) == (("M", "ouroboros/a.py", ""),
                                             ("R100", "ouroboros/c\u00e9.py", "ouroboros/c.py"))
    assert subject_changes(str(tmp_path / "not-a-repo")) == ()


def _fixture_js(lines: int, word: str = "fixture") -> bytes:
    """Frozen synthetic bytes this test owns (the shipped web tree may change freely)."""
    return "".join(f"// {word} line {index:05d}: " + "x" * 60 + "\n" for index in range(lines)).encode()


# The delivery witnesses below were first pinned to what the target before razzant/ouroboros#4325 delivered. The
# resident Body Map change then added governance text every body packet inlines, deliberately: BIBLE.md +402
# characters and the change-review checklist +903 (both approved by the owner), chapter 10's Closed roads +1,060,
# the books' entry lines and the Reference Atlas line +242; about 650 tokens. The expectations are re-measured to
# what the packet delivers with that text (28 Nov 2026, against the target at ca7009c149c1, whose own boundary is
# named beside each): one line of the fixture, or one 250-token step of the window, less and the witness fails.

# (comment-touched file lines, added test file lines or 0, window, the delivery held)
MEASURED_DELIVERY = [
    (900, 0, 131_072, "full"),
    (150, 220, 100_000, "diff"),
    (150, 279, 100_000, "diff"),     # the last fitting size (that target: 305): a larger fixed cost refuses it
    (150, 320, 100_000, "refused"),  # refused there and here: the control that keeps the boundary honest
    (285, 0, 100_000, "full"),       # the last whole-source size at this window (that target: 311)
    (312, 0, 100_000, "diff"),
    (900, 0, 256_000, "full"),       # roomy control
]


FIXTURE_SENTINEL = "// fixture line 00007: "  # inside the source, outside the diff's hunk
RANK = {"refused": 0, "diff": 1, "full": 2}


def test_a_tool_less_packet_delivers_at_least_its_measured_delivery(tmp_path, monkeypatch):
    """The expectations in ``MEASURED_DELIVERY`` are what a packet seat is delivered for exactly these bytes ("full" =
    the touched file's whole source, "diff" = the diff with the snapshot omitted, "refused" = no packet), measured
    as the comment above the table says. The subject is frozen synthetic bytes; the books,
    their navigation, tier-1 governance, the builder and the cold sizing are the real ones, which is
    what this guards. A failure means the packet's fixed cost (navigation, tier-1 governance, the selected sections)
    grew past what was measured, so a change now loses its snapshot or its review. A legitimate
    re-measure follows only a deliberate change of that fixed cost, with its cause stated by source."""
    from ouroboros.tools import review as review_tool
    from ouroboros.tools.registry import ToolContext

    root = pathlib.Path(__file__).resolve().parents[1]
    for index, (touched, added, window, delivered) in enumerate(MEASURED_DELIVERY):
        _installed, subject = _installed_and_subject(tmp_path / str(index))
        (subject / "web" / "modules").mkdir(parents=True)
        (subject / "web" / "modules" / "fixture.js").write_bytes(_fixture_js(touched))
        _git(subject, "add", "-A")
        _git(subject, "commit", "-qm", "fixture source")
        with (subject / "web" / "modules" / "fixture.js").open("a", encoding="utf-8") as stream:
            stream.write("// Review one trailing comment.\n")
        if added:
            (subject / "web" / "tests").mkdir(parents=True)
            (subject / "web" / "tests" / "fixture.test.js").write_bytes(_fixture_js(added, "testing"))
        _git(subject, "add", "-A")
        ctx = ToolContext(repo_dir=subject, drive_root=tmp_path / "drive")
        ctx._review_iteration_count = 0
        frozen = freeze_subject(ctx, ReviewSubjectSpec("system_repo", str(subject), "index", surface="commit_gate",
                                                       governance_root=str(root), layer="body"))
        monkeypatch.setattr(review_tool, "reviewer_context_window", lambda *a, **k: window)
        where = f"packet fixed cost grew past the measured delivery: {(touched, added, window)}"
        try:
            text = build_two_part_brief(frozen, PACKET, drive_root=tmp_path / "drive",
                                        goal="Review these edits")["system"]
        except ValueError as refusal:
            assert delivered == "refused" and "REVIEW_BLOCKED" in str(refusal), where
            continue
        omitted = "Full post-change snapshots were omitted" in text
        got = "diff" if omitted else "full"
        assert RANK[got] >= RANK[delivered], where  # strictly better than measured is not a failure
        staged = text.split("## Staged diff\n\n", 1)[1].split("\n\n## Changed files", 1)[0].strip()
        # the delivered diff section is exactly the frozen full or -U0 rendering: headers, order and context included
        assert staged in (frozen.diff_text.strip(), frozen.render_prompt_diff(0).strip()), where
        assert (FIXTURE_SENTINEL in text) == (not omitted), where  # the whole source is really there, or not
        if not omitted:  # a full-source rung carries every promised file body whole
            bodies = [(subject / "web" / "modules" / "fixture.js").read_text(encoding="utf-8")]
            bodies += [(subject / "web" / "tests" / "fixture.test.js").read_text(encoding="utf-8")] if added else []
            assert all(body.rstrip("\n") in text for body in bodies), where


def test_every_gate_variant_reads_one_structured_path_set_for_file_bodies(tmp_path, monkeypatch):
    """Default Git quotes a UTF-8 name in its display output; the gate's touched-file pack reads the NUL-decoded path
    set, so a pure rename to ``web/café.js`` arrives with its body on the synchronous, the read-only and the frozen
    gate, as it does through the public builder."""
    import contextlib

    from ouroboros.tools import git_review_cycle, review as review_tool
    from ouroboros.tools.registry import ToolContext
    from tests.review_pool_rosters import set_review_pool

    set_review_pool(monkeypatch, ["openai/packet-fixture"])
    monkeypatch.setattr(review_tool, "reviewer_context_window", lambda *a, **k: 1_000_000)
    _installed, repo = _installed_and_subject(tmp_path)
    for path, text in (("web/change-me.js", "// Only the renamed file holds THIS_RENAME_BODY.\n"),
                       ("web/gone.js", "// deleted\n")):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "installed web surface")
    _git(repo, "config", "core.quotePath", "true")
    _git(repo, "mv", "web/change-me.js", "web/café.js")
    _git(repo, "rm", "-q", "web/gone.js")
    (repo / "web" / "new.js").write_text("// brand new\n", encoding="utf-8")
    _git(repo, "add", "-A")
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "drive")
    frozen = freeze_subject(ctx, ReviewSubjectSpec("system_repo", str(repo), "index", surface="commit_gate",
                                                   governance_root=str(repo), layer="body"))

    def gate(subject, review_only):
        ctx._review_iteration_count = 0
        with contextlib.ExitStack() as wave:
            assert git_review_cycle._receptive_wave(ctx, wave, review_only=review_only) == (None, "")
            prepared, reason, exited = review_tool._prepare_unified_review(
                ctx, "review changed files", scope="paths", goal="Read the exact change", subject=subject)
        assert not exited and prepared, reason
        return prepared["prompt"]

    for label, prompt in (("synchronous", gate(None, False)), ("read-only", gate(None, True)),
                          ("frozen gate", gate(frozen, True))):
        assert "THIS_RENAME_BODY" in prompt and "web/café.js" in prompt, label



TARGET_ROWS = [  # (touched path, its row in the target's chapter-1 tree, verbatim) — reviewers' samples, rev-pr2h
    ('supervisor/workers.py', 'workers.py — pool: forkserver Linux, spawn elsewhere, never fork (why: its comment)'),
    ('supervisor/worker_owner_wait.py', 'worker_owner_wait.py — required owner wait lends active slot; task stays RUNNING, custodied (§5)'),
    ('supervisor/schedule_notes.py', 'schedule_notes.py — due `kind:"notify"` → signed `reminder` System row (no model, no retry)'),
    ('supervisor/task_reaper.py', 'task_reaper.py — off-loop reaper (timeout never retries, crash, idle, unready; unconfirmed death keeps reaping; mints no cancel intents) (§5)'),
    ('supervisor/steering.py', 'steering.py — host-minted `issuer` mailbox steering (refused during pending cancel; §6)'),
    ('ouroboros/owner_hurry.py', 'owner_hurry.py — task-local `kind=hurry` latch, one line for running work, never a chat message (§5)'),
    ('ouroboros/outcome_receipt_store.py', 'outcome_receipt_store.py — verification-receipt store; zero-run writes `incomplete`/`unknown` only (zero-run "complete" is unverifiable self-report)'),
    ('ouroboros/cost_projection.py', 'cost_projection.py — sole task-cost projection (`accounted_upper_bound_usd`, null ≠ $0.00) (§6 Budget tracking)'),
    ('ouroboros/delegate_start_claims.py', 'delegate_start_claims.py — serializes custody recheck/`START_REQUESTED` outside waits (retries never hold root/store locks)'),
    ('ouroboros/workspace_admission.py', 'workspace_admission.py — disjoint roots, Project binding, never system-repo fallback (CLI / Headless Boundary) (§6 Owner routing verbs; CLI / Headless Boundary below)'),
    ('ouroboros/review_slot_cancel.py', 'review_slot_cancel.py — typed cancel outcomes reporting only proof (`ReviewSessionSucceededResultUnavailable`, never "may still be live")'),
    ('web/modules/settings_language.js', 'i18n.js, settings_language.js — chrome-only translation overlay; chat prose, logs, code and owner names never walked (§3 Settings)'),
    ('web/modules/project_reference.js', 'project_reference.js, task_reference.js — sole Project and Main task doors (DESIGN)'),
    ('web/modules/model_chooser.js', 'model_roles.js, model_chooser.js — the Models editor and editable chooser shared by Settings, onboarding and route editors; catalog arrival never assigns a value (§3 Navigation)'),
    ('supervisor/task_admission.py', 'task_admission.py — duplicate-ingress reservation fences before effects (§5)'),
    ('supervisor/schedule_occurrence.py', 'schedule_time.py, schedule_occurrence.py — cron/tz parsing; at most one root per occurrence'),
    ('supervisor/direct_roots.py', 'direct_roots.py — retired: old file folded/removed at boot (§5)'),
    ('ouroboros/agent_startup_checks.py', 'agent_startup_checks.py — boot health (warning-only) and self-restart host adoption (§2)'),
    ('ouroboros/evolution_checkpoints.py', 'evolution_checkpoints.py, evolution_fingerprint.py — append-only campaign checkpoints; canonical objective repeat fingerprint'),
    ('ouroboros/cancel_intents.py', 'cancel_intents.py — sole ingress `request_cancel`, durable projection, claim fencing, fail-closed reads (§5; §10.14–15)'),
    ('ouroboros/local_custody_repair.py', 'local_custody_repair.py, custody_owed.py — ended claims retire on proof; unwritten facts stay owed (§6 Owner Continue)'),
    ('ouroboros/custody_owed.py', 'local_custody_repair.py, custody_owed.py — ended claims retire on proof; unwritten facts stay owed (§6 Owner Continue)'),
    ('ouroboros/_usage_settlement.py', '_usage_settlement.py — usage/price evidence projected outside ledger tx (§6 Budget tracking)'),
    ('ouroboros/argv_budget.py', 'argv_budget.py — ARG_MAX byte admission (argv + env; checked by `skill_exec`)'),
    ('ouroboros/tools/media.py', 'media.py — `extract_pdf_text` (`ocr_pdf`; text layer, not OCR), `youtube_transcript`, `extract_video_frames`, `read_archive` (`archive_reader.py`; nothing extracted)'),
    ('ouroboros/tools/subagent_integration.py', 'subagent_integration.py — `integrate_subagent_patch` (protected-path gated, genesis refused), `compare_subagent_patches`'),
    ('ouroboros/desktop_media.py', 'desktop_media.py — GTK display probe, own-loopback audio grant (other shells keep native prompts; §8)'),
    ('ouroboros/tools/commit_gate.py', 'commit_gate.py — block classification, paid cycle cap, contract fingerprint'),
    ('ouroboros/force_remove.py', 'force_remove.py, unindexed.py — permission-granting `force_rmtree`; `.noindex` names'),
]


def _single_file_subject(tmp_path, path: str, source: bytes):
    """One repository whose index changes ``path`` by a trailing comment line; no installed tree is built."""
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "subject"
    (repo / path).parent.mkdir(parents=True)
    _git(tmp_path, "init", "-q", str(repo))
    for key, value in (("user.email", "t@example.com"), ("user.name", "t"), ("commit.gpgsign", "false")):
        _git(repo, "config", key, value)
    (repo / path).write_bytes(source)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fixture source")
    with (repo / path).open("a", encoding="utf-8") as stream:
        stream.write("// Review one trailing comment.\n")
    _git(repo, "add", "-A")
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "drive")
    ctx._review_iteration_count = 0
    return ctx, repo


@pytest.mark.parametrize("path, row", TARGET_ROWS)
def test_a_tool_less_reviewer_without_the_snapshot_gets_the_target_row_of_the_touched_file(tmp_path, monkeypatch,
                                                                                         path, row):
    """The target's chapter-1 tree row for a touched file (frozen here, verbatim) reaches a tool-less packet seat whose
    snapshot was omitted, as on the pre-PR target: the selector excerpts the directory's lines naming a touched file.
    A rule edited away in the directory turns this red."""
    from ouroboros.tools import review as review_tool

    root = pathlib.Path(__file__).resolve().parents[1]
    ctx, repo = _single_file_subject(tmp_path, path, _fixture_js(3000))
    frozen = freeze_subject(ctx, ReviewSubjectSpec("system_repo", str(repo), "index", surface="commit_gate",
                                                   governance_root=str(root), layer="body"))
    monkeypatch.setattr(review_tool, "reviewer_context_window", lambda *a, **k: 131_072)
    text = build_two_part_brief(frozen, PACKET, drive_root=tmp_path / "drive", goal="Review these edits")["system"]
    assert "Full post-change snapshots were omitted" in text, path
    assert row in text, (path, row)


def test_the_frozen_builder_reviews_a_conflict_anchor_outside_the_delta(tmp_path, monkeypatch):
    """A managed resolution that keeps a conflict file at its mechanical-merge bytes leaves it out of the Git delta;
    the reviewed set is the delta and the anchors (``ManagedReviewSubject.touched_paths``), as at the gate, so the
    packet's changed-files list names the anchor."""
    from ouroboros.tools.registry import ToolContext
    from tests import test_managed_review_subject as managed_fixture

    repo, simple, tx = managed_fixture._managed_resolution_repo(tmp_path, monkeypatch)
    _git(repo, "checkout", tx["m0_tree"], "--", "conflict.txt")  # the anchor back at its M0 bytes
    _git(repo, "add", "-A")
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "data", task_id=simple.task_id,
                      task_metadata=simple.task_metadata)
    frozen = freeze_subject(ctx, ReviewSubjectSpec("system_repo", str(repo), "index", surface="commit_gate",
                                                   layer="body"))
    assert frozen.managed is not None and "conflict.txt" not in {path for _s, path in frozen.name_status}
    system = build_two_part_brief(frozen, PACKET, goal="g", commit_message="m")["system"]
    for path in ("conflict.txt", "resolver_note.txt"):  # a reviewed path: a file heading, or a changed-files row
        assert f"\n### {path}\n" in system or path in system[system.index("## Changed files"):], path


def test_a_chapter_opening_naming_a_touched_file_is_sent_once(tmp_path, monkeypatch):
    """Chapter 12's first paragraph names ``ouroboros/place_arrivals.py``: the selector picks that paragraph for a
    change to it, and the packet navigation (which carries every introduction) then skips it, so the reviewer reads
    it once and its bytes are not paid twice (the gate assembles through the same governance selector)."""
    from ouroboros.reference_books import _preamble, load_reference_book
    from ouroboros.tools import review as review_tool

    root = pathlib.Path(__file__).resolve().parents[1]
    chapter = next(c for c in load_reference_book(root, "architecture").chapters
                   if c.source_path.endswith("12-host-service-companions-and-chat-ids.md"))
    opening = chapter.text_at(_preamble(chapter)).strip()
    assert "place_arrivals.py" in opening
    ctx, repo = _single_file_subject(tmp_path, "ouroboros/place_arrivals.py", b'"""Fixture."""\n')
    frozen = freeze_subject(ctx, ReviewSubjectSpec("system_repo", str(repo), "index", surface="commit_gate",
                                                   governance_root=str(root), layer="body"))
    monkeypatch.setattr(review_tool, "reviewer_context_window", lambda *a, **k: 1_000_000)
    built = build_two_part_brief(frozen, PACKET, drive_root=tmp_path / "drive", goal="Review these edits")["system"]
    assert built.count(opening) == 1


# Real history replayed by a reviewer (rev-pr2j), frozen as synthetic files of the same names and sizes and the same
# numbers of changed lines: (paths with (added, deleted, bytes before), window, the target's rung, sections it selected).
# The window is the smallest 250-token step at which this packet carries both sections (``MEASURED_DELIVERY``'s
# comment; that target: 100,000 for both).
HISTORY = [
    ({"docs/DESIGN.md": (1, 1, 321_047), "web/ui.css": (0, 2, 120_198)}, 100_250, "diff",
     ("Navigation and shared UI contracts", "opening block 2 (excerpt: its lines naming a touched file)")),
    ({"docs/DESIGN.md": (1, 1, 317_362), "tests/test_phone_keyboard_recovery_browser.py": (5, 5, 81_942),
      "tests/test_ui_coherence_browser.py": (1, 1, 89_734), "web/ui.css": (1, 1, 113_592)}, 101_250, "diff",
     ("Navigation and shared UI contracts", "opening block 2 (excerpt: its lines naming a touched file)")),
]


@pytest.mark.parametrize("files, window, rung, sections", HISTORY)
def test_replayed_history_keeps_the_target_sections_and_directory_rows(tmp_path, monkeypatch, files, window, rung,
                                                                      sections):
    """A chapter opening already carried by the packet navigation is not selected again, so the change-specific share
    goes to the sections and directory rows the pre-PR target selected for the same paths (measured 27 Nov 2026 at
    7981018bb40e with the target's governance code and documents), at the window ``HISTORY`` holds each to."""
    from ouroboros.tools import review as review_tool
    from ouroboros.tools.registry import ToolContext

    def lines(count, word):
        return "".join(f"/* {word} line {index:05d}: " + "x" * 60 + " */\n" for index in range(count))

    root = pathlib.Path(__file__).resolve().parents[1]
    repo = tmp_path / "subject"
    repo.mkdir()
    _git(repo, "init", "-q")
    for key, value in (("user.email", "t@example.com"), ("user.name", "t"), ("commit.gpgsign", "false")):
        _git(repo, "config", key, value)
    for path, (added, deleted, size) in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(lines(max(size // 80, deleted + 1), "base"), encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    for path, (added, deleted, _size) in files.items():
        body = (repo / path).read_text(encoding="utf-8").split("\n")
        body = body[:len(body) - 1 - deleted] + [line for line in lines(added, "new").split("\n") if line] + [""]
        (repo / path).write_text("\n".join(body), encoding="utf-8")
    _git(repo, "add", "-A")
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "drive")
    ctx._review_iteration_count = 0
    frozen = freeze_subject(ctx, ReviewSubjectSpec("system_repo", str(repo), "index", surface="commit_gate",
                                                   governance_root=str(root), layer="body"))
    monkeypatch.setattr(review_tool, "reviewer_context_window", lambda *a, **k: window)
    text = build_two_part_brief(frozen, PACKET, drive_root=tmp_path / "drive", goal="Review")["system"]
    got = "diff" if "Full post-change snapshots were omitted" in text else "full"
    assert RANK[got] >= RANK[rung]
    for section in sections:
        assert f"#{section}\n" in text, section


def test_a_packet_sendable_with_the_resident_opening_selected_stays_sendable(monkeypatch):
    """The fit ladder's last step before refusing switches to the governance fallback (the selection that re-sends a
    resident opening, smaller where skipping it admitted larger sections); without one it refuses."""
    from ouroboros.tools import review as review_tool, review_admission

    monkeypatch.setattr(review_tool, "_quorum_input_token_limit", lambda *a, **k: 1_000)
    monkeypatch.setattr(review_tool, "reviewer_context_window", lambda *a, **k: 1_000_000)
    state = {"governance": "x" * 8_000}

    def assemble(files, diff):
        return state["governance"] + files + diff, 0

    def fallback():
        state["governance"] = "small"
    assemble.fallback = fallback
    prompt, _stable, error = review_admission.fit_triad_prompt(["openai/gpt-y"], assemble, "", "+one line", "a.py",
                                                               ".", compact_diff=lambda: "+one line")
    assert not error and prompt.startswith("small")
    state["governance"] = "x" * 8_000
    _prompt, _stable, error = review_admission.fit_triad_prompt(["openai/gpt-y"], assemble, "", "+one line", "a.py",
                                                                ".", compact_diff=lambda: "+one line")
    assert "REVIEW_BLOCKED" in error


def test_a_many_file_change_gets_every_full_body_at_its_measured_window(tmp_path, monkeypatch):
    """Thirteen synthetic web modules with one trailing comment each: at 263,750, the smallest 250-token step where
    this packet fits them (``MEASURED_DELIVERY``'s comment; the target before #4325 and at ca7009c149c1: 262,750),
    it sends all thirteen full bodies, its changed-files list pointing at the file headings it already carries
    instead of repeating thirteen paths."""
    from ouroboros.tools import review as review_tool
    from ouroboros.tools.review_admission import CHANGED_FILES_ARE_HEADINGS
    from ouroboros.tools.registry import ToolContext

    root = pathlib.Path(__file__).resolve().parents[1]
    paths = [f"web/modules/many_{index:02d}.js" for index in range(13)]
    _installed, subject = _installed_and_subject(tmp_path)
    for path in paths:
        (subject / path).parent.mkdir(parents=True, exist_ok=True)
        (subject / path).write_text("".join(f"// many line {k:05d}: " + "x" * 60 + "\n" for k in range(300)),
                                    encoding="utf-8")
    _git(subject, "add", "-A")
    _git(subject, "commit", "-qm", "base")
    for path in paths:
        with (subject / path).open("a", encoding="utf-8") as stream:
            stream.write("// Review one trailing comment.\n")
    _git(subject, "add", "-A")
    ctx = ToolContext(repo_dir=subject, drive_root=tmp_path / "drive")
    ctx._review_iteration_count = 0
    frozen = freeze_subject(ctx, ReviewSubjectSpec("system_repo", str(subject), "index", surface="commit_gate",
                                                   governance_root=str(root), layer="body"))
    monkeypatch.setattr(review_tool, "reviewer_context_window", lambda *a, **k: 263_750)
    text = build_two_part_brief(frozen, PACKET, drive_root=tmp_path / "drive", goal="Review these edits")["system"]
    assert "Full post-change snapshots were omitted" not in text
    assert all(f"\n### {path}\n" in text for path in paths) and CHANGED_FILES_ARE_HEADINGS in text


def test_the_fallback_step_buys_no_second_density_probe(monkeypatch):
    """A cold overflowing route is probed once: switching to the governance fallback reuses the first pass's
    calibrated limit instead of re-entering the cold-density rung (a provider without usage would be probed again)."""
    from ouroboros.tools import review as review_tool, review_admission

    monkeypatch.setattr(review_tool, "calibrated_input_token_limit", lambda *a, **k: 1_000)
    monkeypatch.setattr(review_tool, "reviewer_context_window", lambda *a, **k: 1_000_000)
    probes = []
    monkeypatch.setattr(review_admission, "density_probe_before_size_refusal",
                        lambda *a, **k: probes.append(a[1]) or "no_sample")
    state = {"governance": "x" * 8_000}

    def assemble(files, diff):
        return state["governance"] + files + diff, 0

    def fallback():
        state["governance"] = "small"
    assemble.fallback = fallback
    prompt, _stable, error = review_admission.fit_triad_prompt(["openai/gpt-y"], assemble, "", "+one line", "a.py",
                                                               ".", compact_diff=lambda: "+one line")
    assert not error and prompt.startswith("small") and probes == ["openai/gpt-y"]
