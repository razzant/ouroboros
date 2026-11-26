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
    from ouroboros.tools import review_admission

    frozen, _repo = _subject(tmp_path)
    monkeypatch.setattr(review_admission, "fit_triad_prompt", lambda *a, **k: ("", 0, "pack does not fit"))
    with pytest.raises(ValueError, match="pack does not fit"):
        build_two_part_brief(frozen, PACKET, goal="raise f", commit_message="bump f")


# --- the installed domains of the touched paths, for a packet seat (no module tree in the book) ------------

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


def _packet_brief(tmp_path, installed, subject):
    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(repo_dir=str(subject), drive_root=str(data), task_id="t-rows", task_metadata=None,
                          _review_history=[], _review_advisory=[], _review_iteration_count=0,
                          drive_logs=lambda: data / "logs")
    spec = ReviewSubjectSpec(root_kind="system_repo", root=str(subject), kind="index", surface="commit_gate",
                             layer="body", governance_root=str(installed))
    return build_two_part_brief(freeze_subject(ctx, spec), PACKET, goal="g", commit_message="m")["system"]


def _module_block(system):
    from ouroboros.tools.governance_module_rows import HEADING

    start = system.index(HEADING)
    return system[start:system.index("\n\n", start)]


def test_a_packet_seat_gets_the_installed_domain_and_neighbours_never_a_working_copy_lead(tmp_path):
    installed, subject = _installed_and_subject(tmp_path)
    (subject / "ouroboros" / "a.py").write_text('"""STAGED lead of a."""\n', encoding="utf-8")
    _git(subject, "add", "-A")
    (subject / "ouroboros" / "a.py").write_text('"""WORKTREE lead of a."""\n', encoding="utf-8")  # after staging
    system = _packet_brief(tmp_path, installed, subject)
    block = _module_block(system)
    assert "- ouroboros/domains.toml (D06: Review stack, 3 files; manuals: docs/architecture/06-4.md)" in block
    assert "- ouroboros/a.py (touched in this change)" in block
    assert "- ouroboros/b.py — Lead of b." in system and "- ouroboros/c.py — Lead of c." in system
    # No row carries a working-copy lead as the reviewed tree's (the touched file reaches the seat with the change).
    assert "WORKTREE" not in block and "STAGED" not in block and "STAGED lead of a." in system


def test_a_new_and_a_deleted_module_are_named_from_the_installed_manifest(tmp_path):
    installed, subject = _installed_and_subject(tmp_path)
    (subject / "ouroboros" / "new.py").write_text('"""Proposed new module."""\n', encoding="utf-8")
    manifest = (subject / "ouroboros" / "domains.toml").read_text(encoding="utf-8")
    (subject / "ouroboros" / "domains.toml").write_text(
        manifest.replace('"ouroboros/b.py" = "D06"\n', "") + '"ouroboros/new.py" = "D06"\n', encoding="utf-8")
    (subject / "ouroboros" / "b.py").unlink()
    _git(subject, "add", "-A")
    block = _module_block(_packet_brief(tmp_path, installed, subject))
    assert "- ouroboros/new.py (new in this change; its domain assignment, if any, is in the change)" in block
    assert "- ouroboros/b.py (touched in this change)" in block  # deleted: its installed domain still answers
    assert "(D06: Review stack, 3 files;" in block and "Proposed new module." not in block


def test_a_tight_window_and_a_missing_manifest_are_said_in_counted_words(tmp_path, monkeypatch):
    from ouroboros.tools import review as review_tool
    from ouroboros.tools.commit_gate import commit_review_contract_material
    from ouroboros.tools.governance_module_rows import (DOMAINS_NOT_DELIVERED, HEADING, NEIGHBOURS_NOT_DELIVERED,
                                                        UNAVAILABLE)

    def strings(value):
        if isinstance(value, str):
            return [value]
        items = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else ()
        return [text for item in items for text in strings(item)]

    contract = "\n".join(strings(commit_review_contract_material()))
    installed, subject = _installed_and_subject(tmp_path / "roomy")
    (subject / "ouroboros" / "a.py").write_text('"""Changed."""\n', encoding="utf-8")
    _git(subject, "add", "-A")
    assert HEADING in _packet_brief(tmp_path / "roomy", installed, subject)
    monkeypatch.setattr(review_tool, "_triad_governance_usable_window", lambda *a, **k: 0)
    starved = _packet_brief(tmp_path / "roomy", installed, subject)
    assert HEADING not in starved and DOMAINS_NOT_DELIVERED in starved
    assert DOMAINS_NOT_DELIVERED in contract and NEIGHBOURS_NOT_DELIVERED.format(count="⟨count⟩") in contract
    monkeypatch.undo()
    bare, subject = _installed_and_subject(tmp_path / "bare", manifest=False)
    (subject / "ouroboros" / "a.py").write_text('"""Changed."""\n', encoding="utf-8")
    _git(subject, "add", "-A")
    unavailable = _packet_brief(tmp_path / "bare", bare, subject)
    assert UNAVAILABLE in unavailable and UNAVAILABLE in contract and "Errno" not in unavailable
