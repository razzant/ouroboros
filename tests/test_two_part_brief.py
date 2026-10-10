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
    assert "### Staged diff" in system and "return 2" in system
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
    from ouroboros.tools import review_admission

    frozen, _repo = _subject(tmp_path)
    monkeypatch.setattr(review_admission, "fit_triad_prompt", lambda *a, **k: ("", 0, "pack does not fit"))
    with pytest.raises(ValueError, match="pack does not fit"):
        build_two_part_brief(frozen, PACKET, goal="raise f", commit_message="bump f")
