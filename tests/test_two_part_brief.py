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


# (comment-touched file lines, added test file lines or 0, window, the pre-PR target's delivery), measured against the


PRE_PR_DELIVERY = [
    (900, 0, 131_072, "full"),
    (150, 220, 100_000, "diff"),
    (150, 302, 100_000, "diff"),     # the target's last fitting size: a larger packet fixed cost refuses it
    (150, 320, 100_000, "refused"),  # both refuse: the control that keeps the boundary honest
    (306, 0, 100_000, "full"),       # the target's last whole-source size at this window
    (312, 0, 100_000, "diff"),
    (900, 0, 256_000, "full"),       # roomy control
]


FIXTURE_SENTINEL = "// fixture line 00007: "  # inside the source, outside the diff's hunk


def test_a_tool_less_packet_delivers_at_least_what_the_pre_pr_target_delivered(tmp_path, monkeypatch):
    """The expectations in ``PRE_PR_DELIVERY`` are what the pre-PR target (razzant/ouroboros, with its own
    governance_context, context_layout and governance documents; measured 27 Nov 2026 at bcac9f84126b, ad9e98f10198
    and a4c694fb4f21, as the boundaries move with the target's documents) delivered to a packet seat for exactly
    these bytes ("full" = the touched file's whole source, "diff" = the diff with the snapshot omitted, "refused" = no
    packet). The subject is frozen synthetic bytes; the books,
    their navigation, tier-1 governance, the builder and the cold sizing are the real ones, which is
    what this guards. A failure means the packet's fixed cost (navigation, tier-1 governance, the selected sections)
    grew past what the pre-PR target spent, so a change now loses its snapshot or its review. A legitimate
    re-measure follows only a deliberate change of that fixed cost, against the same target."""
    from ouroboros.tools import review as review_tool
    from ouroboros.tools.registry import ToolContext

    root = pathlib.Path(__file__).resolve().parents[1]
    for index, (touched, added, window, delivered) in enumerate(PRE_PR_DELIVERY):
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
        where = f"packet fixed cost grew past the pre-PR target's: {(touched, added, window)}"
        try:
            text = build_two_part_brief(frozen, PACKET, drive_root=tmp_path / "drive",
                                        goal="Review these edits")["system"]
        except ValueError as refusal:
            assert delivered == "refused" and "REVIEW_BLOCKED" in str(refusal), where
            continue
        assert delivered != "refused", f"the control now fits: re-measure the target for {(touched, added, window)}"
        omitted = "Full post-change snapshots were omitted" in text
        assert ("diff" if omitted else "full") == delivered, where
        assert "// Review one trailing comment." in text, where
        assert (FIXTURE_SENTINEL in text) == (not omitted), where  # the whole source is really there, or not


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



ROW_SAMPLE = [  # reviewers' and the implementer's samples of the target's tree rows (rev-pr2h; fix round 6)
    "supervisor/workers.py", "supervisor/worker_owner_wait.py", "supervisor/schedule_notes.py",
    "supervisor/task_reaper.py", "supervisor/steering.py", "ouroboros/owner_hurry.py",
    "ouroboros/outcome_receipt_store.py", "ouroboros/cost_projection.py", "ouroboros/delegate_start_claims.py",
    "ouroboros/workspace_admission.py", "ouroboros/review_slot_cancel.py", "web/modules/settings_language.js",
    "web/modules/project_reference.js", "web/modules/model_chooser.js", "supervisor/task_admission.py",
    "supervisor/schedule_occurrence.py", "supervisor/direct_roots.py", "ouroboros/agent_startup_checks.py",
    "ouroboros/evolution_checkpoints.py", "ouroboros/cancel_intents.py", "ouroboros/local_custody_repair.py",
    "ouroboros/custody_owed.py", "ouroboros/_usage_settlement.py", "ouroboros/argv_budget.py",
    "ouroboros/tools/media.py", "ouroboros/tools/subagent_integration.py", "ouroboros/desktop_media.py",
    "ouroboros/tools/commit_gate.py", "ouroboros/force_remove.py", "supervisor/active_activity.py",
    "supervisor/telemetry_events.py", "ouroboros/genesis_project.py", "web/modules/delegated_activity.js",
    "web/modules/widgets.js", "web/modules/masonry.js", "supervisor/message_ingress.py",
    "supervisor/schedule_view.py", "supervisor/owner_pause_control.py", "supervisor/update_recovery.py",
    "ouroboros/update_channels.py", "ouroboros/maintenance.py", "ouroboros/maintenance_jobs.py",
    "ouroboros/maintenance_retirement.py", "ouroboros/loop_transport.py", "ouroboros/loop_tool_execution.py",
    "ouroboros/tool_call_log.py", "ouroboros/late_tool_results.py", "ouroboros/task_carrier.py",
    "ouroboros/pricing.py", "ouroboros/llm_helper_window.py", "ouroboros/merge_receipts.py",
    "ouroboros/chat_chain.py", "ouroboros/chat_references.py", "ouroboros/chronicle_store.py",
    "ouroboros/context_input_selection.py", "ouroboros/memory_view_events.py", "ouroboros/coop_checkpoint.py",
    "ouroboros/configured_subagents.py", "ouroboros/subagent_runtime.py", "ouroboros/chat_uploads.py",
    "ouroboros/confined_files.py", "ouroboros/review_execution.py", "ouroboros/review_session_preparation.py",
    "ouroboros/review_body_fact.py", "ouroboros/schedule_contract.py", "ouroboros/pause_notices.py",
    "ouroboros/icon_geometry.py", "ouroboros/places.py", "ouroboros/place_words_history.py",
    "ouroboros/conversation_lease.py", "ouroboros/skill_owner_attestation.py", "ouroboros/live_tool_lane.py",
    "ouroboros/panic_source.py", "ouroboros/platform_peer.py", "ouroboros/owned_shutdown.py",
    "ouroboros/gateways/image_api.py", "ouroboros/gateway/router.py", "ouroboros/gateway/logs.py",
    "ouroboros/gateway/storage.py", "ouroboros/gateway/kept_services.py", "ouroboros/gateway/ui_i18n.py",
    "ouroboros/gateway/peer_contracts.py", "ouroboros/gateway/peer_letters_api.py",
    "ouroboros/tools/shell_guards.py", "ouroboros/tools/task_retention.py", "ouroboros/tools/tool_discovery.py",
    "ouroboros/tools/plan_ask.py", "ouroboros/tools/review_change_custody.py",
    "ouroboros/tools/extension_dispatch.py",
]


@pytest.mark.parametrize("path", ROW_SAMPLE)
def test_a_tool_less_reviewer_without_the_snapshot_gets_the_directory_row_of_the_touched_file(tmp_path, monkeypatch,
                                                                                              path):
    """The module directory's row for a touched file reaches a tool-less packet seat whose snapshot was omitted, as on
    the pre-PR target (the selector excerpts a listing's lines naming a touched file). Frozen synthetic bytes at the
    file's real path; the expected line is the directory's own, byte-identical to the target's tree row."""
    from ouroboros.tools import review as review_tool
    from ouroboros.tools.registry import ToolContext

    root = pathlib.Path(__file__).resolve().parents[1]
    directory = (root / "docs" / "architecture" / "01-module-directory.md").read_text(encoding="utf-8")
    base = path.rsplit("/", 1)[-1]
    rows = [line.strip() for line in directory.splitlines() if base in line and " — " in line]
    assert rows, path
    _installed, subject = _installed_and_subject(tmp_path)
    (subject / path).parent.mkdir(parents=True, exist_ok=True)
    (subject / path).write_bytes(_fixture_js(3000))
    _git(subject, "add", "-A")
    _git(subject, "commit", "-qm", "fixture source")
    with (subject / path).open("a", encoding="utf-8") as stream:
        stream.write("// Review one trailing comment.\n")
    _git(subject, "add", "-A")
    ctx = ToolContext(repo_dir=subject, drive_root=tmp_path / "drive")
    ctx._review_iteration_count = 0
    frozen = freeze_subject(ctx, ReviewSubjectSpec("system_repo", str(subject), "index", surface="commit_gate",
                                                   governance_root=str(root), layer="body"))
    monkeypatch.setattr(review_tool, "reviewer_context_window", lambda *a, **k: 131_072)
    text = build_two_part_brief(frozen, PACKET, drive_root=tmp_path / "drive", goal="Review these edits")["system"]
    assert "Full post-change snapshots were omitted" in text, path
    assert any(row in text for row in rows), (path, rows)
