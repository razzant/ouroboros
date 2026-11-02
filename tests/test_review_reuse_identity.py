"""Which change buys a new review: operative instructions do, governance text never does.

``review_change``'s reuse identity (identity a) is one rule on every surface (change, preflight, result,
skill): the subject, the panel, the round and the review contract, never governance text. An edited
``docs/CHECKLISTS.md``, the reviewed surface's own section included, or another guide reuses the settled
record, which keeps the rules it was judged by. Every text Ouroboros writes into a seat's request is its contract,
by layer and delivery (``commit_gate.commit_review_contract_material``): the change brief's sections, frames and
rules (``review_admission.change_brief_words``), each layer's governance navigation, the skill's and the result's
own (``skill_contract_words``, ``result_contract_words``) and the wrapper of every seated delivery (agent session
per access, writable checking copy, native episode). Each changed owner is shown in the request the seat's own
executor hands its transport (``_final_request``). Every wave runs through ``skill_review``, ``review_change`` or the
commit gate's review cycle on the golden offline seams or the synthetic session engine; no model is called.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

import ouroboros.review_substrate as substrate
from ouroboros import review as review_core
from ouroboros import (
    review_checking,
    review_execution,
    review_history,
    review_ledger,
    review_native_episode,
    review_session_preparation,
    triad_review,
)
from ouroboros.tools import (
    governance_context,
    review_admission,
    review_brief_coupling,
    review_helpers,
    review_multi_model,
    review_prompt_text,
    review_result,
    review_skill,
    review_synthesis,
)
from ouroboros.tools import review as review_facade
from ouroboros.tools import review_change as review_change_mod
from ouroboros.tools.review_change import run_review_change
from tests import _contributor_packet_shared as shared
from tests._review_session_route_shared import (  # noqa: F401 - its autouse transport fixture rides along
    FakeGateway,
    _owned_gateway_uses_each_test_transport,
    _terminal_detail,
)
from tests._review_session_route_shared import fake_route as _fake_route
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
from tests.test_review_change_end_to_end import COMMIT_MESSAGE, GOAL, SCOPE, _foreign_project
from tests.test_review_change_end_to_end import staged_body as _staged_body
from tests.test_review_skill import _leave, _physical_sends, _read_mail, _review, _settled_record, _state, _until
from tests.test_review_skill import harness as _harness
from tests.test_review_skill import leaving as _leaving

harness, leaving, staged_body, fake_route = _harness, _leaving, _staged_body, _fake_route  # requested by name below
REASON = "small payload"
# Each non-skill surface's identical request over one staged change of a foreign project.
SURFACES = {"change": {"subject": "index", "goal": "Bump", "scope": "app.py"},
            "preflight": {"subject": "worktree", "surface": "preflight", "reviewers": ["s1"]},
            "preflight-session": {"subject": "worktree", "surface": "preflight", "reviewers": ["t2"]},
            "result": {"subject": "result", "answer": "Bumped VALUE to 2."}}


# The modules that own a request's text (a constant or a builder); "nav" names a ``NAVIGATION_WORDS`` entry as
# "layer.delivery".
OWNERS = {"skill": review_skill, "result": review_result, "multi": review_multi_model, "text": review_prompt_text,
          "admission": review_admission, "coupling": review_brief_coupling, "helpers": review_helpers,
          "synthesis": review_synthesis, "history": review_history, "session": review_session_preparation,
          "native": review_native_episode, "checking": review_checking, "execution": review_execution,
          "pulse": review_core}


def _revise(monkeypatch, owner: str, module: str = "skill") -> str:
    """Edit one owner of a request's text as a shipped change would (a builder: ``_revise_builder``); returns the
    added words."""
    words = f"Revised {owner}."
    if module == "nav":
        layer, delivery = owner.split(".")
        entry = governance_context.NAVIGATION_WORDS[layer]
        monkeypatch.setitem(entry, delivery, (f"{entry[delivery][0]} {words}", *entry[delivery][1:]))
        return words
    home = OWNERS[module]
    if owner.endswith("]"):  # one entry of a table of templates
        name, key = owner[:-1].split("[")
        table = getattr(home, name)
        monkeypatch.setitem(table, key, f"{table[key]} {words}")
        return words
    value = getattr(home, owner)
    if callable(value):
        return _revise_builder(monkeypatch, home, owner)
    if isinstance(value, dict):
        key = next(iter(value))
        monkeypatch.setitem(value, key, f"{value[key]} {words}")
    elif isinstance(value, tuple):
        monkeypatch.setattr(home, owner, (f"{value[0]} {words}", *value[1:]))
    else:
        monkeypatch.setattr(home, owner, f"{value} {words}")
        if module == "multi" and hasattr(review_facade, owner):  # the facade's import-time binding of the constant
            monkeypatch.setattr(review_facade, owner, getattr(home, owner))
    return words


def _text(content) -> str:
    if isinstance(content, list):
        return "\n".join(str(block.get("text") or "") for block in content if isinstance(block, dict))
    return str(content or "")


class _Transport:
    """A packet or native seat's transport, offline: ``chat(**kwargs)`` answers from ``script`` and keeps every request's
    messages exactly as the executor sent them."""

    def __init__(self, script):
        self.script, self.sent = list(script), []

    def chat(self, **kwargs):
        self.sent.append(copy.deepcopy(kwargs.get("messages") or []))
        return self.script.pop(0), {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0}


def _tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


# A large tracked file of the subject, and a native seat's offline episode before its answer: compact_context asked for
# nothing (the host's focus guidance replies), then that file read past one result's page (the way on is appended).
BIG_FILE = "big.txt"
_NATIVE_EPISODE = ({"tool_calls": [_tool_call("compact_context", {}, "c1")]},
                   {"tool_calls": [_tool_call("read_file", {"path": BIG_FILE}, "c2")]})


def _commit_big_file(project: Path) -> None:
    (project / BIG_FILE).write_text("".join(f"{index:06d} {'evidence ' * 10}\n" for index in range(2500)), encoding="utf-8")
    shared.git(project, "add", BIG_FILE)
    shared.git(project, "commit", "-q", "-m", "a large file")


def _actual_sends(request, slot, answer: str = "[]", source_root=None) -> list:
    """Every request the seat's own executor (``review_execution._review_route_executor``) hands its transport, as a
    list of message lists. A packet seat's ``chat`` messages and a native episode's opening and later requests (tool
    replies included) are recorded at the transport as sent; a native episode first runs ``_NATIVE_EPISODE``. An agent
    session's work order is the executor's own, beside its answer schema and system instructions (read-only, or a
    writable seat's checking copy as ``review_checking.session_checking`` composes them, its private area by
    placeholder): the engine's actual session start body is checked on the synthetic engine (``_starts``)."""
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
    from ouroboros.triad_review import request_output_shape

    assignment = review_execution.ReviewAssignment(request=request, slot=slot, call_id=f"op-{slot.slot_id}",
                                                   custody_root=source_root)
    transport = _Transport([{"content": answer}])
    executor = review_execution._review_route_executor(assignment, llm=transport)
    if isinstance(executor, review_execution.AgentSessionReviewExecutor):
        schema = review_execution.review_session_output_schema(request.surface, request_output_shape(request))
        instructions = review_execution._REVIEW_SESSION_INSTRUCTIONS
        if review_checking.writable_seat(slot):
            source = request.checking_source or {"kind": "workspace", "root": request.session_root}
            instructions = review_checking.CHECKING_REVIEWER_INSTRUCTIONS + review_checking.checking_instructions(
                "⟨private area⟩", source, str(request.session_root or ""))
        return [[{"role": "system", "content": instructions}, {"role": "user", "content": executor.session_prompt},
                 {"role": "schema", "content": json.dumps(schema, sort_keys=True)}]]
    if isinstance(executor, NativeToolRoundReviewExecutor):
        transport.script[:0] = copy.deepcopy(list(_NATIVE_EPISODE))
    executor.execute()
    return transport.sent


def _final_request(request, slot, answer: str = "[]", source_root=None) -> str:
    """The text of every request a seat was handed (``_actual_sends``), opening first."""
    return "\n\n".join("\n".join(_text(message.get("content")) for message in messages)
                       for messages in _actual_sends(request, slot, answer, source_root))


def _revise_builder(monkeypatch, owner, name: str) -> str:
    """A shipped edit of a builder's own text: every loaded binding of ``owner.name`` renders the added words after
    its text. Returns the words."""
    words, original = f"Revised {name}.", getattr(owner, name)

    def revised(*args, **kwargs):
        text = original(*args, **kwargs)
        return f"{text} {words}" if text else text

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("ouroboros") and getattr(module, name, None) is original:
            monkeypatch.setattr(module, name, revised)
    return words


def _restart(ctx) -> None:
    """A shipped instruction change reaches a worker that imports the new code: it holds none of the old
    process's settled-seat cache (``review_custody``); attempt rows and records stay on disk."""
    for name in ("_review_settled_attempts", "_review_pending_invocations"):
        vars(ctx).pop(name, None)


def _rules_copy(root: Path, monkeypatch) -> Path:
    """The executing install's checklist as an editable copy: ``review_checklist`` reads it under REPO_ROOT."""
    from ouroboros.tools import review_checklist

    (root / "docs").mkdir(parents=True)
    shutil.copyfile(Path(review_checklist.REPO_ROOT) / "docs" / "CHECKLISTS.md", root / "docs" / "CHECKLISTS.md")
    monkeypatch.setattr(review_checklist, "REPO_ROOT", root)
    return root / "docs" / "CHECKLISTS.md"


def _amend_guidance(checklist: Path, governance_root, section: str) -> None:
    """New criteria inside the reviewed surface's own checklist section, a new chapter elsewhere and, given a
    governance root the seats read, an amended constitution there."""
    heading = f"## {section}"
    text = checklist.read_text(encoding="utf-8")
    assert text.count(heading) == 1
    checklist.write_text(text.replace(heading, f"{heading}\n\nA new criterion every reviewer applies.", 1)
                         + "\n## An unrelated chapter\n\nNew text.\n", encoding="utf-8")
    if governance_root is not None:
        (Path(governance_root) / "BIBLE.md").write_text("# BIBLE\n\nAn amended principle.\n", encoding="utf-8")


def _spy_wave_facts(monkeypatch) -> list:
    """Every wave's ledger facts as ``review_change`` hands them to the ledger."""
    captured, real = [], review_change_mod.wave_facts

    def spy(*args, **kwargs):
        facts = real(*args, **kwargs)
        captured.append(facts)
        return facts

    monkeypatch.setattr(review_change_mod, "wave_facts", spy)
    return captured


def _assert_fallback_agrees(facts: dict, record: dict) -> None:
    """The ledger's own digest of the record's fields, without the prepared key, is the key the wave stored."""
    structured = {key: value for key, value in dict(facts.get("structured") or {}).items() if key != "reuse_key"}
    rebuilt = review_ledger.build_wave_record({**facts, "reuse_key": "", "structured": structured},
                                              surface=record["surface"])
    assert record["fingerprints"]["reuse_key"] and rebuilt.fingerprints["reuse_key"] == record["fingerprints"]["reuse_key"]


def _judged(record: dict) -> dict:
    return {key: record["brief"]["checklist"][key] for key in ("rules_source", "checklist_hash")}


@pytest.fixture
def foreign(tmp_path, monkeypatch, surface):
    """A foreign project (the core layer) with one staged change and its task's reply, the golden pool and
    physical seam, every seat's brief kept: a change or preflight seat gives the golden two-part answer, a
    result seat the clean array."""
    ctx, project = _foreign_project(tmp_path, monkeypatch)
    _commit_big_file(project)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")
    sends: list = []
    briefs: dict = {}

    def answer(request, slot, actor, *, retry_state, pending_invocation_checkpoint):
        if surface == "result":
            actor.raw_text = "[]"
        briefs[slot.slot_id] = _final_request(request, slot, actor.raw_text, tmp_path / "native-sources")
        return actor

    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    (project / "app.py").write_text("VALUE = 2  # staged\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    return ctx, sends, briefs


# --- governance text never buys a review, on any surface ---------------------------------------------------------


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_guidance_edits_reuse_the_settled_record_on_every_surface(foreign, tmp_path, monkeypatch, surface):
    """New criteria in the surface's own checklist section, another chapter and an amended constitution: the
    identical request answers from the settled record at $0, which names the rules it was judged by; the
    ledger's fallback digest of that record agrees with the key the wave prepared."""
    from ouroboros.tools.review_result import RESULT_CHECKLIST_SECTION

    ctx, sends, _briefs = foreign
    checklist = _rules_copy(tmp_path / "rules-install", monkeypatch)
    facts = _spy_wave_facts(monkeypatch)
    first = run_review_change(ctx, **SURFACES[surface])
    assert (first["aggregate"], first["reused"]) == ("PASS", False), first
    paid, record = len(sends), review_ledger.load_record(ctx.drive_root, first["record_id"])
    judged = _judged(record)
    _assert_fallback_agrees(facts[0], record)

    section = RESULT_CHECKLIST_SECTION if surface == "result" else "Change Review Checklist"
    _amend_guidance(checklist, Path(ctx.system_repo_dir), section)
    again = run_review_change(ctx, **SURFACES[surface])
    assert (again["reused"], again["record_id"], len(sends)) == (True, first["record_id"], paid), again
    assert _judged(review_ledger.load_record(ctx.drive_root, first["record_id"])) == judged


def test_guidance_edits_reuse_the_settled_skill_record_and_the_gate_keeps_its_verdict(harness, tmp_path):
    """The same rule for a skill: new Skill Review Checklist criteria, another chapter and an amended
    constitution reuse the settled record at $0; the run gate keeps it, nothing is re-reviewed or rewritten."""
    h = harness
    checklist = _rules_copy(tmp_path / "rules-install", h.monkeypatch)
    facts = _spy_wave_facts(h.monkeypatch)
    first = _review(h, reviewers=["t1"], reason=REASON)
    assert first.paid and first.status == "clean", first.error
    record = review_ledger.load_record(h.drive, first.record_id)
    judged = _judged(record)
    assert judged["rules_source"]["sha"] == review_skill.skill_rules()["rules_source"]["sha"]
    _assert_fallback_agrees(facts[0], record)

    _amend_guidance(checklist, h.system, review_skill.SKILL_CHECKLIST_SECTION)
    assert "A new criterion every reviewer applies." in review_skill.skill_rubric()
    assert _judged({"brief": {"checklist": review_skill.skill_rules()}}) != judged
    again = _review(h, reviewers=["t1"], reason=REASON)
    assert (again.reused, again.paid, again.record_id, len(h.sends)) == (True, False, first.record_id, 1), again.error
    assert _judged(review_ledger.load_record(h.drive, first.record_id)) == judged
    assert (_state(h).record_id, _state(h).status) == (first.record_id, "clean")


# --- operative instructions are the contract ---------------------------------------------------------------------


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_an_operative_instruction_change_is_a_new_wave_within_the_allowance(foreign, monkeypatch, surface):
    """The answer contract every seat of these surfaces is held to changes: the identical request is a new
    paid wave, the same request again replays it free, and a further change meets the spent allowance."""
    ctx, sends, _briefs = foreign
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "2")
    first = run_review_change(ctx, **SURFACES[surface])
    paid = len(sends)
    assert first["aggregate"] == "PASS" and paid, first
    monkeypatch.setattr(triad_review, "REVIEW_JSON_ARRAY_CONTRACT", triad_review.REVIEW_JSON_ARRAY_CONTRACT + "\nRevised.")
    _restart(ctx)
    second = run_review_change(ctx, **SURFACES[surface])
    assert (second["aggregate"], second["reused"], len(sends)) == ("PASS", False, 2 * paid), second
    assert run_review_change(ctx, **SURFACES[surface])["record_id"] == second["record_id"] and len(sends) == 2 * paid
    monkeypatch.setattr(triad_review, "REVIEW_TWO_PART_OBJECT_CONTRACT",
                        triad_review.REVIEW_TWO_PART_OBJECT_CONTRACT + "\nRevised.")
    _restart(ctx)
    third = run_review_change(ctx, **SURFACES[surface])
    assert third["aggregate"] == "NOT_DISPATCHED" and third["dispatch_refusal"]["kind"] == "review_cycles_exhausted"
    assert len(sends) == 2 * paid


@pytest.mark.parametrize("surface, module, owner, seat", [
    ("change", "multi", "TRIAD_USER_TURN", "t1"), ("change", "multi", "TRIAD_ROLE_HINT", "t2"),
    ("change", "nav", "core.every", "s1"), ("change", "nav", "core.packet", "t1"),
    ("change", "nav", "core.retrieving", "t2"), ("change", "text", "REVIEW_PREAMBLE_CORE", "t1"),
    ("change", "text", "REPO_ANTI_PATTERN_LOCK_GUARD_CORE", "t2"), ("change", "admission", "PACKET_TEMPLATE_STABLE", "t1"),
    # Sections, frames and wrappers a builder writes around the request's data.
    ("change", "helpers", "build_scope_section", "t1"), ("change", "helpers", "build_goal_section", "s1"),
    ("change", "synthesis", "build_coupling_part", "t2"), ("change", "coupling", "answer_format_section", "s1"),
    ("change", "multi", "triad_stable_head", "t1"), ("change", "session", "render_review_session_prompt", "t2"),
    ("change", "native", "native_episode_prompt", "s1"), ("change", "native", "_NATIVE_REVIEW_INSTRUCTIONS", "s1"),
    # A preflight look seats one retrieving row: what only a packet seat is told is not its contract.
    ("preflight", "multi", "TRIAD_ROLE_HINT", "s1"), ("preflight", "nav", "core.retrieving", "s1"),
    ("preflight", "text", "REVIEW_PREAMBLE_CORE", "s1"), ("preflight", "native", "native_episode_prompt", "s1"),
    ("preflight", "multi", "TRIAD_USER_TURN", None), ("preflight", "multi", "triad_stable_head", None),
    ("preflight", "nav", "core.packet", None), ("preflight", "admission", "PACKET_TEMPLATE_STABLE", None),
    # Its native seat reads no agent session's wrapper.
    ("preflight", "session", "render_review_session_prompt", None),
    # A foreign project's change carries no diff pulse (the body's own growth question).
    ("change", "pulse", "DIFF_PULSE_QUESTION", None),
])
def test_a_changed_change_brief_instruction_is_a_new_wave_where_its_delivery_sits(foreign, monkeypatch, surface,
                                                                                 module, owner, seat):
    """The change brief's own words on a foreign project (the core layer): an identical change or preflight
    request after a shipped edit of a word a seated delivery reads is a new paid wave (the seat reads it), the
    same request then replays free; a word no seated delivery reads leaves the settled record answering."""
    ctx, sends, briefs = foreign
    ask = SURFACES[surface]
    first = run_review_change(ctx, **ask)
    paid = len(sends)
    assert first["aggregate"] == "PASS" and paid, first
    words = _revise(monkeypatch, owner, module)
    _restart(ctx)
    briefs.clear()
    second = run_review_change(ctx, **ask)
    if seat is None:
        assert (second["reused"], second["record_id"], len(sends)) == (True, first["record_id"], paid), second
        return
    assert (second["aggregate"], second["reused"], len(sends)) == ("PASS", False, 2 * paid), second
    assert words in briefs[seat]
    assert run_review_change(ctx, **ask)["record_id"] == second["record_id"] and len(sends) == 2 * paid


@pytest.mark.parametrize("surface", ["change"])
@pytest.mark.parametrize("module, owner, read", [("text", "_HISTORY_VERIFICATION_ONLY_RULE", True),
                                                 ("coupling", "_MANAGED_SUBJECT_AUTHORITY", False)])
def test_a_change_brief_word_only_some_rounds_render_counts_in_every_round(foreign, monkeypatch, surface, module,
                                                                           owner, read):
    """History rules reach a seat once its subject has an earlier round, a managed subject's authority only in a
    managed resolution; the contract counts both in every round, so their edit is a new wave, here one whose
    seats read the history rule (the settled wave is its history) and none of which reads the authority."""
    ctx, sends, briefs = foreign
    first = run_review_change(ctx, **SURFACES[surface])
    paid = len(sends)
    assert first["aggregate"] == "PASS" and paid, first
    words = _revise(monkeypatch, owner, module)
    _restart(ctx)
    briefs.clear()
    second = run_review_change(ctx, **SURFACES[surface])
    assert (second["aggregate"], second["reused"], len(sends)) == ("PASS", False, 2 * paid), second
    assert briefs and any(words in brief for brief in briefs.values()) is read


@pytest.mark.parametrize("surface", ["change"])
@pytest.mark.parametrize("case", ["obligations", "paged"])
def test_a_counted_frame_around_host_records_reaches_its_seats(foreign, monkeypatch, surface, case):
    """Open obligations and a paged brief are host records; the frame they are handed in is the contract. With
    an obligation held (every seat) or a session seat's first send paged (its pointer), an edit of that frame is
    a new wave whose seats read the new words, and the same request then replays free."""
    from ouroboros.review_state_model import AdvisoryReviewState
    from ouroboros.review_state_records import ObligationItem

    ctx, sends, briefs = foreign
    if case == "paged":
        monkeypatch.setattr(review_brief_coupling, "SESSION_INLINE_DIFF_CEILING_CHARS", 1)
    first = run_review_change(ctx, **SURFACES[surface])
    paid = len(sends)
    assert first["aggregate"] == "PASS" and paid, first
    if case == "obligations":
        held = [ObligationItem(obligation_id="obl-1", item="code_quality", severity="critical", reason="still open",
                               source_attempt_ts="", source_attempt_msg="")]
        monkeypatch.setattr(AdvisoryReviewState, "get_open_obligations", lambda self, repo_key=None: held)
    words = _revise(monkeypatch, *(("build_obligations_block", "text") if case == "obligations"
                                   else ("paged_review_text", "coupling")))
    _restart(ctx)
    briefs.clear()
    second = run_review_change(ctx, **SURFACES[surface])
    assert (second["aggregate"], second["reused"], len(sends)) == ("PASS", False, 2 * paid), second
    assert all(words in briefs[seat] for seat in (("t1", "t2", "s1") if case == "obligations" else ("t2",)))
    assert run_review_change(ctx, **SURFACES[surface])["record_id"] == second["record_id"] and len(sends) == 2 * paid


@pytest.mark.parametrize("surface", ["change"])
def test_the_session_answer_schema_is_the_contract_where_a_session_sits(foreign, monkeypatch, surface):
    """An agent session is asked for the two-part schema beside contract B's text: a shipped edit of the schema
    alone is a new wave whose session seat is asked the new schema (a packet seat is asked none)."""
    ctx, sends, briefs = foreign
    first = run_review_change(ctx, **SURFACES[surface])
    paid = len(sends)
    assert first["aggregate"] == "PASS" and paid, first
    monkeypatch.setitem(triad_review.TWO_PART_SESSION_OUTPUT_SCHEMA, "description", "Revised schema.")
    _restart(ctx)
    briefs.clear()
    second = run_review_change(ctx, **SURFACES[surface])
    assert (second["aggregate"], second["reused"], len(sends)) == ("PASS", False, 2 * paid), second
    assert "Revised schema." in briefs["t2"] and "Revised schema." not in briefs["t1"]


@pytest.mark.parametrize("surface", ["change", "preflight-session"])
@pytest.mark.parametrize("owner", ["_FOCUS_GUIDANCE", "_CONTINUE_READING"])
def test_a_native_reply_directive_is_the_contract_where_a_native_seat_sits(foreign, monkeypatch, surface, owner):
    """Inside its episode a native seat is told how to handle its evidence: compact_context asked for nothing answers
    with the focus guidance, a result cut at one page ends with the way on. A shipped edit of either is a new wave
    whose native seat reads the new words in a later request (its opening never carries them), and the same request
    then replays free; a session seat alone is told neither, so its panel keeps its settled record."""
    ctx, sends, briefs = foreign
    sent: dict = {}
    real = _actual_sends

    def recording(request, slot, answer="[]", source_root=None):
        sent[slot.slot_id] = real(request, slot, answer, source_root)
        return sent[slot.slot_id]

    monkeypatch.setattr(sys.modules[__name__], "_actual_sends", recording)
    first = run_review_change(ctx, **SURFACES[surface])
    paid = len(sends)
    assert first["aggregate"] == "PASS" and paid, first
    words = _revise(monkeypatch, owner, "native")
    _restart(ctx)
    sent.clear()
    second = run_review_change(ctx, **SURFACES[surface])
    if surface == "preflight-session":
        assert (second["reused"], second["record_id"], len(sends)) == (True, first["record_id"], paid), second
        return
    assert (second["aggregate"], second["reused"], len(sends)) == ("PASS", False, 2 * paid), second
    texts = ["\n".join(_text(message.get("content")) for message in messages) for messages in sent["s1"]]
    assert words not in texts[0] and any(words in text for text in texts[1:]), texts[1:]
    assert run_review_change(ctx, **SURFACES[surface])["record_id"] == second["record_id"] and len(sends) == 2 * paid


@pytest.mark.parametrize("surface", ["change"])
def test_a_paged_native_opening_is_read_under_the_counted_pointer(foreign, monkeypatch, surface):
    """A native seat whose opening exceeds its landing is handed the stored work order's pointer and the contract
    again; an edit of that opening's composition is a new wave whose native seat's first request carries it."""
    ctx, sends, briefs = foreign
    sent: dict = {}
    real = _actual_sends

    def recording(request, slot, answer="[]", source_root=None):
        sent[slot.slot_id] = real(request, slot, answer, source_root)
        return sent[slot.slot_id]

    monkeypatch.setattr(sys.modules[__name__], "_actual_sends", recording)
    monkeypatch.setattr(review_native_episode, "native_landing_at", lambda bound: 1)
    first = run_review_change(ctx, **SURFACES[surface])
    paid = len(sends)
    assert first["aggregate"] == "PASS" and paid, first
    opening = "\n".join(_text(message.get("content")) for message in sent["s1"][0])
    assert "is not inlined because of the initial delivery size" in opening and "\nOUTPUT CONTRACT:\n" in opening
    words = _revise(monkeypatch, "native_paged_prompt", "native")
    _restart(ctx)
    sent.clear()
    second = run_review_change(ctx, **SURFACES[surface])
    assert (second["aggregate"], second["reused"], len(sends)) == ("PASS", False, 2 * paid), second
    assert words in "\n".join(_text(message.get("content")) for message in sent["s1"][0])


def test_a_cut_skill_diff_tells_the_seat_to_read_the_file(harness):
    """A re-review whose changed file's diff passes the per-file cap is cut with the host's own directive to read the
    file; that directive (``_REREVIEW_RULES['diff_cut']``) is its contract and its revised words reach the seat's
    actual request on the cut branch, the payload itself staying data."""
    h = harness
    requests: dict = {}
    h.monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _final_seam(h, requests))
    first = _review(h, reviewers=["t1"], reason=REASON)
    assert first.paid and first.status == "clean", first.error
    before = review_skill.skill_contract_fingerprint()
    words = _revise(h.monkeypatch, "_REREVIEW_RULES[diff_cut]", "skill")
    assert review_skill.skill_contract_fingerprint() != before
    script = Path(h.skill_dir) / "scripts" / "hello.py"
    script.write_text(script.read_text(encoding="utf-8") + "".join(f"# changed line {index}\n" for index in range(500)),
                      encoding="utf-8")
    _restart(h.ctx)
    second = _review(h, reviewers=["t1"], reason=REASON)
    assert second.paid and not second.reused, second.error
    assert "more diff lines; read the file" in requests["t1"] and words in requests["t1"]
    assert _review(h, reviewers=["t1"], reason=REASON).reused


def _starts() -> list:
    return [body for gateway in FakeGateway.instances for body in gateway.start_requests]


@pytest.fixture
def session_engine(tmp_path, monkeypatch, fake_route, access):
    """One agent-session seat of ``access`` on the synthetic engine (``FakeGateway``) over a foreign project's staged
    change: each start request is the seat's final request (work order, system instructions, answer schema)."""
    ctx, project = _foreign_project(tmp_path, monkeypatch)
    set_review_pool(monkeypatch, pool_roster(pool_seat("independent", "fake-review=fake-small", kind="agent_session",
                                                       access=access)))
    FakeGateway.catalog_entry["accessProfilesSupported"] = ["readonly", "workspace_write", "full"]
    FakeGateway.detail = _terminal_detail(shared.ANSWERS["t2"], conformance="passed")
    start, init = FakeGateway.start_run, FakeGateway.__init__

    def unique_start(self, request, **kwargs):
        return {**start(self, request, **kwargs), "runId": f"run-{len(_starts())}"}

    def writable_engine(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.engine_version, self.trust = "3.22.0", []
        self.ensure_full_access = lambda root: self.trust.append(root) or {"allowFullAccess": True}

    monkeypatch.setattr(FakeGateway, "start_run", unique_start)
    monkeypatch.setattr(FakeGateway, "__init__", writable_engine)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")
    (project / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    return ctx


@pytest.mark.parametrize("access, module, owner", [
    ("readonly", "execution", "_REVIEW_SESSION_INSTRUCTIONS"), ("workspace_write", "checking", "checking_instructions"),
    ("workspace_write", "checking", "CHECKING_REVIEWER_INSTRUCTIONS"),
    ("readonly", "session", "render_review_session_prompt"), ("workspace_write", "session", "render_review_session_prompt"),
    ("workspace_write", "schema", "TWO_PART_SESSION_OUTPUT_SCHEMA"),
])
def test_a_session_seats_system_instructions_are_its_contract(session_engine, monkeypatch, access, module, owner):
    """The engine's start request (``prepare_review_session_request``'s body) is the seat's final request: an identical
    change after a shipped edit of the read-only or the checking copy's system instructions, the session work order or
    the answer schema is a new run whose body carries the new words; the same request then replays free."""
    import dataclasses

    ctx = session_engine
    first = run_review_change(ctx, **SURFACES["change"])
    assert (first["aggregate"], len(_starts()), _starts()[0]["access"]) == ("PASS", 1, access), first
    first_body = json.dumps(_starts()[0])  # as sent: the engine keeps the body's objects by reference
    if module == "schema":  # the answer schema the engine is asked for
        words = "Revised schema."
        monkeypatch.setitem(triad_review.TWO_PART_SESSION_OUTPUT_SCHEMA, "description", words)
    else:
        words = _revise(monkeypatch, owner, module)
    if owner == "_REVIEW_SESSION_INSTRUCTIONS":  # the invocation's default: a shipped edit reaches it on import

        @dataclasses.dataclass(frozen=True)
        class Revised(review_execution.SessionInvocation):
            instructions: str = review_execution._REVIEW_SESSION_INSTRUCTIONS

        monkeypatch.setattr(review_execution, "SessionInvocation", Revised)
    _restart(ctx)
    second = run_review_change(ctx, **SURFACES["change"])
    assert (second["aggregate"], second["reused"], len(_starts())) == ("PASS", False, 2), second
    assert words in json.dumps(_starts()[-1]) and words not in first_body
    assert run_review_change(ctx, **SURFACES["change"])["record_id"] == second["record_id"] and len(_starts()) == 2


def _final_request_substrate(requests: dict):
    """The golden paid seam of a whole wave, keeping each seat's final request by seat id (``_final_request``)."""
    inner = shared.golden_substrate([])

    def run_review_request(request, *, slots, **kwargs):
        for slot in slots:
            requests[slot.slot_id] = _final_request(request, slot)
        return inner(request, slots=slots, **kwargs)

    return run_review_request


@pytest.mark.parametrize("module, owner, seat", [
    ("multi", "TRIAD_USER_TURN", "t1"), ("multi", "_CONSTITUTIONAL_PREAMBLE", "t1"), ("multi", "TRIAD_ROLE_HINT", "s1"),
    ("nav", "body.packet", "t1"), ("nav", "body.retrieving", "t2"), ("text", "REPO_ANTI_PATTERN_LOCK_GUARD", "s1"),
    ("helpers", "build_goal_section", "t1"), ("synthesis", "build_coupling_part", "t2"),
    ("pulse", "DIFF_PULSE_QUESTION", "t1"), ("pulse", "DIFF_PULSE_QUESTION", "s1"),
])
def test_the_commit_gate_and_a_change_review_of_the_body_see_one_contract(staged_body, tmp_path, monkeypatch, module,
                                                                          owner, seat):
    """The body's staged index: a changed guide (the body layer's own checklist section) reuses the settled
    change review; a changed change-brief word is a new wave that its seat reads, and the commit gate's
    review-only cycle over the same index records the very contract that wave recorded."""
    from ouroboros.tools.git_review_cycle import _run_non_committing_review_cycle
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_checklist import BODY_CHECKLIST_SECTION

    repo = Path(staged_body["repo"])
    checklist = _rules_copy(tmp_path / "rules-install", monkeypatch)
    requests: dict = {}
    monkeypatch.setattr(substrate, "run_review_request", _final_request_substrate(requests))
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "operation-drive")
    ask = {"root": "system_repo", "surface": "change", "subject": "index", "goal": GOAL, "scope": SCOPE}
    first = run_review_change(ctx, **ask)
    assert first["aggregate"] == "PASS", first
    _amend_guidance(checklist, None, BODY_CHECKLIST_SECTION)
    assert run_review_change(ctx, **ask)["record_id"] == first["record_id"], "a changed guide is no new wave"

    words = _revise(monkeypatch, owner, module)
    requests.clear()
    second = run_review_change(ctx, **ask)
    assert (second["aggregate"], second["reused"]) == ("PASS", False), second
    assert words in requests[seat]
    contract = review_ledger.load_record(ctx.drive_root, second["record_id"])["fingerprints"]["review_contract"]
    assert contract and contract != review_ledger.load_record(ctx.drive_root, first["record_id"])["fingerprints"][
        "review_contract"]

    gate_ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "gate-drive")
    outcome = _run_non_committing_review_cycle(gate_ctx, COMMIT_MESSAGE, skip_advisory_review=True, goal=GOAL, scope=SCOPE)
    assert outcome["status"] == "passed", outcome
    gate = review_ledger.load_record(gate_ctx.drive_root, outcome["review_record_id"])
    assert gate["fingerprints"]["review_contract"] == contract


@pytest.mark.parametrize("module, owner, rendered", [
    ("skill", "_PREAMBLE", True), ("skill", "_CHECKLIST_HEADING", True), ("skill", "_TOOL_EFFECT_RULE", True),
    ("skill", "_REREVIEW_RULES", True), ("skill", "_PAYLOAD_INLINE", True), ("skill", "_OUTPUT_RULE", True),
    ("skill", "SKILL_USER_TURN", True), ("nav", "skill.packet", True), ("skill", "_packet_prompt", True),
    ("skill", "_change_section", True),
    # Not rendered in this brief (no install record, no goal, a payload that fits), still part of the contract.
    ("skill", "_INSTALL_ORIGIN_RULE", False), ("skill", "_AUTHOR_WORDS", False), ("skill", "_PACKET_NOT_DELIVERED", False),
    ("skill", "_install_origin_text", False), ("skill", "_REREVIEW_RULES[diff_cut]", False),
])
def test_a_changed_skill_instruction_buys_one_new_wave_and_the_new_contract_then_replays_free(harness, module, owner,
                                                                                             rendered):
    """An identical request after a shipped edit of the skill brief's own instructions is a new paid wave once
    (the seat reads the new words); the same request again replays it at $0; the earlier record stays."""
    from ouroboros.tools.commit_gate import commit_review_contract_fingerprint

    h = harness
    briefs: dict = {}
    h.monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _final_seam(h, briefs))
    first = _review(h, reviewers=["t1"], reason=REASON)
    assert (first.status, first.paid, len(h.sends)) == ("clean", True, 1), first.error
    before = review_ledger.load_record(h.drive, first.record_id)
    shared_contract = commit_review_contract_fingerprint()
    words = _revise(h.monkeypatch, owner, module)
    assert commit_review_contract_fingerprint() == shared_contract, "the shared contract of every other wave stays"
    _restart(h.ctx)
    briefs.clear()
    second = _review(h, reviewers=["t1"], reason=REASON)
    assert (second.status, second.paid, second.reused, len(h.sends)) == ("clean", True, False, 2), second.error
    assert second.record_id != first.record_id and (words in briefs["t1"]) is rendered
    third = _review(h, reviewers=["t1"], reason=REASON)
    assert (third.reused, third.paid, third.record_id, len(h.sends)) == (True, False, second.record_id, 2)
    assert review_ledger.load_record(h.drive, first.record_id) == before, "no earlier record is rewritten"
    contracts = [review_ledger.load_record(h.drive, rid)["fingerprints"]["review_contract"]
                 for rid in (first.record_id, second.record_id)]
    assert contracts[0] != contracts[1] and shared_contract not in contracts
    assert (_state(h).record_id, _state(h).status) == (second.record_id, "clean")


def _final_seam(h, requests: dict, *, sends: bool = False):
    """The skill harness's golden seam, keeping each seat's requests (``_actual_sends``; their text unless ``sends``)."""
    from tests.test_review_skill import _capturing_seam

    seam = _capturing_seam(h.sends, h.answers, {})

    def run_slot(self, request, slot, **kwargs):
        requests[slot.slot_id] = (_actual_sends if sends else _final_request)(request, slot)
        return seam(self, request, slot, **kwargs)

    return run_slot


@pytest.mark.parametrize("retrieving_module, retrieving_owner, packet_module, packet_owner", [
    ("skill", "_PAYLOAD_SOURCES", "skill", "_PACKET_NOT_DELIVERED"), ("nav", "skill.retrieving", "nav", "skill.packet"),
    ("skill", "_retrieving_task", "skill", "_packet_prompt"), ("skill", "_NOT_PASTED", "multi", "triad_stable_head"),
    ("session", "render_review_session_prompt", "skill", "SKILL_USER_TURN"),
    ("native", "native_episode_prompt", "skill", "_packet_prompt"),
])
def test_each_delivery_serialization_joins_the_skill_contract_only_while_such_a_seat_sits(
        harness, retrieving_module, retrieving_owner, packet_module, packet_owner):
    """Packet seat ``t1`` and retrieving seats ``t2`` (a session) and ``s1`` (a native api row): what only a
    retrieving seat is told (its payload delivery, its governance reading duty, its session or native wrapper)
    is no packet panel's contract, and what only a packet seat is told no retrieving panel's."""
    h = harness
    h.monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")
    packet = _review(h, reviewers=["t1"], reason=REASON)
    retrieving = _review(h, reviewers=["t2", "s1"], reason=REASON)
    assert packet.paid and retrieving.paid and sorted(s["slot_id"] for s in h.sends) == ["s1", "t1", "t2"]

    _revise(h.monkeypatch, retrieving_owner, retrieving_module)
    _restart(h.ctx)
    assert _review(h, reviewers=["t1"], reason=REASON).record_id == packet.record_id and len(h.sends) == 3
    retrieving = _review(h, reviewers=["t2", "s1"], reason=REASON)
    assert (retrieving.status, retrieving.reused, len(h.sends)) == ("clean", False, 5), retrieving.error

    _revise(h.monkeypatch, packet_owner, packet_module)
    _restart(h.ctx)
    assert _review(h, reviewers=["t2", "s1"], reason=REASON).record_id == retrieving.record_id and len(h.sends) == 5
    repacked = _review(h, reviewers=["t1"], reason=REASON)
    assert (repacked.status, repacked.reused, len(h.sends)) == ("clean", False, 6), repacked.error


def test_the_skill_layers_required_sources_are_part_of_its_contract(harness):
    """Which host documents a skill seat must carry is an operative selection, not a guide's text: dropping a
    required chapter is a new paid wave once (the packet stops inlining it), and the new contract replays free."""
    from tests.test_review_skill import _capturing_seam

    h = harness
    briefs: dict = {}
    h.monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _capturing_seam(h.sends, h.answers, briefs))
    first = _review(h, reviewers=["t1"], reason=REASON)
    dropped = governance_context.SKILL_REQUIRED_CHAPTERS[-1]
    assert first.paid and f"## {dropped}" in briefs["t1"], first.error
    h.monkeypatch.setattr(governance_context, "SKILL_REQUIRED_CHAPTERS", governance_context.SKILL_REQUIRED_CHAPTERS[:-1])
    _restart(h.ctx)
    briefs.clear()
    second = _review(h, reviewers=["t1"], reason=REASON)
    assert (second.paid, second.reused, len(h.sends)) == (True, False, 2) and f"## {dropped}" not in briefs["t1"]
    assert _review(h, reviewers=["t1"], reason=REASON).record_id == second.record_id and len(h.sends) == 2


# A task reply under review with the author's own ask, so every word the result brief owns is rendered.
RESULT_ASK = {"subject": "result", "answer": "Bumped VALUE to 2.", "goal": "Check the bump is what was asked"}


@pytest.fixture
def result_wave(tmp_path, monkeypatch):
    """A foreign project's task reply under the golden pool and physical seam, every seat's brief kept."""
    from types import SimpleNamespace

    ctx, _project = _foreign_project(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")
    sends: list = []
    briefs: dict = {}

    def answer(request, slot, actor, *, retry_state, pending_invocation_checkpoint):
        briefs[slot.slot_id] = _final_request(request, slot)
        actor.raw_text = "[]"
        return actor

    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    return SimpleNamespace(ctx=ctx, sends=sends, briefs=briefs, monkeypatch=monkeypatch)


@pytest.mark.parametrize("owner, seat", [
    ("_AUTHOR_WORDS", "t1"), ("RESULT_USER_TURN", "t1"), ("_READ_SUBJECT", "s1"), ("_READ_IN_PLACE", "t2"),
    ("_packet_prompt", "t1"), ("_retrieving_task", "s1"), ("_brief_head", "t2"),
])
def test_a_changed_result_instruction_buys_one_new_wave_and_the_new_contract_then_replays_free(result_wave, owner,
                                                                                              seat):
    """The result brief's own words (how it frames the author's ask, a packet seat's user turn, what a
    retrieving seat must read in place): an identical request after a shipped edit is a new paid wave once and
    the seat reads the new words; the same request again replays free; the earlier record stays."""
    from ouroboros.tools.commit_gate import commit_review_contract_fingerprint

    w = result_wave
    first = run_review_change(w.ctx, **RESULT_ASK)
    paid = len(w.sends)
    assert (first["aggregate"], first["reused"], sorted(w.briefs)) == ("PASS", False, ["s1", "t1", "t2"]), first
    assert run_review_change(w.ctx, **RESULT_ASK)["record_id"] == first["record_id"] and len(w.sends) == paid
    before = review_ledger.load_record(w.ctx.drive_root, first["record_id"])
    shared_contract = commit_review_contract_fingerprint()
    words = _revise(w.monkeypatch, owner, "result")
    assert commit_review_contract_fingerprint() == shared_contract, "the shared contract of every other wave stays"
    _restart(w.ctx)
    w.briefs.clear()
    second = run_review_change(w.ctx, **RESULT_ASK)
    assert (second["aggregate"], second["reused"], len(w.sends)) == ("PASS", False, 2 * paid), second
    assert words in w.briefs[seat]
    assert run_review_change(w.ctx, **RESULT_ASK)["record_id"] == second["record_id"] and len(w.sends) == 2 * paid
    assert review_ledger.load_record(w.ctx.drive_root, first["record_id"]) == before, "no earlier record is rewritten"


def test_each_result_delivery_counts_only_while_such_a_seat_sits(result_wave):
    """A packet seat's user turn is no retrieving panel's contract, and the retrieving seats' read-in-place
    instruction no packet panel's."""
    w = result_wave
    packet_ask, retrieving_ask = ({**RESULT_ASK, "reviewers": seats, "reason": REASON} for seats in (["t1"], ["t2", "s1"]))
    packet = run_review_change(w.ctx, **packet_ask)
    retrieving = run_review_change(w.ctx, **retrieving_ask)
    assert packet["aggregate"] == retrieving["aggregate"] == "PASS" and len(w.sends) == 3

    _revise(w.monkeypatch, "RESULT_USER_TURN", "result")
    _restart(w.ctx)
    assert run_review_change(w.ctx, **retrieving_ask)["record_id"] == retrieving["record_id"] and len(w.sends) == 3
    packet = run_review_change(w.ctx, **packet_ask)
    assert (packet["reused"], len(w.sends)) == (False, 4), packet

    _revise(w.monkeypatch, "_READ_IN_PLACE", "result")
    _restart(w.ctx)
    assert run_review_change(w.ctx, **packet_ask)["record_id"] == packet["record_id"] and len(w.sends) == 4
    retrieving = run_review_change(w.ctx, **retrieving_ask)
    assert (retrieving["reused"], len(w.sends)) == (False, 6), retrieving


def test_inline_source_reading_guidance_changes_skill_identity(harness):
    """A directive beside a required-source row changes a retrieving skill review, not its saved predecessor."""
    from ouroboros.tools import scope_required_sources

    h, briefs = harness, {}
    h.monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")
    h.monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _final_seam(h, briefs))
    ask = {"reviewers": ["s1"], "reason": REASON}
    old = "; delivered inline in full, no second read needed"
    new = "; delivered inline in full; compare the inline source with the declared manifest before judging"
    first = _review(h, **ask)
    assert first.paid and len(h.sends) == 1 and old in briefs["s1"], first.error
    before = review_ledger.load_record(h.drive, first.record_id)
    original = scope_required_sources.render_required_sources

    def changed(rows, **kwargs):
        return original(rows, **kwargs).replace(old, new)

    h.monkeypatch.setattr(scope_required_sources, "render_required_sources", changed)
    _restart(h.ctx)
    briefs.clear()
    second = _review(h, **ask)
    assert (second.paid, second.reused, len(h.sends)) == (True, False, 2), second.error
    assert second.record_id != first.record_id and new in briefs["s1"]
    after = review_ledger.load_record(h.drive, second.record_id)
    assert after["fingerprints"]["review_contract"] != before["fingerprints"]["review_contract"]
    assert review_ledger.load_record(h.drive, first.record_id) == before
    third = _review(h, **ask)
    assert (third.record_id, third.reused, third.paid, len(h.sends)) == (second.record_id, True, False, 2)


# --- recorded identity, paid history and the material axes ---------------------------------------------------


def test_a_running_wave_is_collected_under_its_recorded_contract_after_the_instructions_changed(leaving):
    """The author left a running wave; the brief's instructions then changed. Re-issuing the call rejoins the
    open operation at $0 and settles it under the identity its dispatch recorded."""
    L, h = leaving, leaving.h
    reference = _leave(L)
    recorded = dict(review_ledger.load_record(h.drive, reference["record_id"])["fingerprints"])
    _revise(h.monkeypatch, "_PREAMBLE")
    _read_mail(L)
    thread, box = L.call()
    thread.join(1.0)
    assert thread.is_alive(), "the re-issue rejoins the running wave"
    L.gate.set()
    text = L.join(thread, box)
    assert len(_physical_sends(h)) == 1 and "The review is still running" not in text, text
    record = review_ledger.load_record(h.drive, reference["record_id"])
    assert record["state"] == "settled" and record["fingerprints"] == recorded
    from ouroboros.review_state import load_state
    from ouroboros.tools.review_change_custody import review_state_root

    attempts = [row for row in load_state(review_state_root(h.ctx)).filter_attempts(tool_name="review_skill") if row.paid]
    assert {row.review_contract_fingerprint for row in attempts} == {recorded["review_contract"]}
    assert _state(h).record_id == reference["record_id"]


def test_a_left_wave_settled_before_the_change_stays_collectable_by_its_reference(leaving):
    """The same left wave settles on its own; then a worker on changed instructions collects it by the
    reference its notice names: the original settled record at $0, no second dispatch."""
    from ouroboros.tools.review_change import _handle_review_change

    L, h = leaving, leaving.h
    reference = _leave(L)
    L.gate.set()
    _until(lambda: _settled_record(h, reference["record_id"]))
    _revise(h.monkeypatch, "_PREAMBLE")
    _restart(h.ctx)
    text = _handle_review_change(h.ctx, review_reference={"record_id": reference["record_id"], "task_id": "task-skill",
                                                          "surface": "skill"})
    assert reference["record_id"] in text and '"reused": true' in text and len(_physical_sends(h)) == 1, text


@pytest.mark.parametrize("lane", ["manual", "task"])
def test_a_contract_change_keeps_every_earlier_paid_skill_wave_counted(harness, lane):
    """Both skill ceilings (per content version outside a task, per root task inside one) count the waves paid
    before the contract changed: the new contract buys a wave only while the allowance lasts, then the typed
    refusal stands and the earlier verdict for these bytes keeps the gate."""
    h = harness
    if lane == "manual":
        h.ctx.task_id = ""
    else:
        h.ctx._skill_review_group_id = "task:task-skill:weather"
    h.monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "2")
    first = _review(h, reviewers=["t1"], reason=REASON)
    _revise(h.monkeypatch, "_PREAMBLE")
    _restart(h.ctx)
    second = _review(h, reviewers=["t1"], reason=REASON)
    assert first.paid and second.paid and second.record_id != first.record_id and len(h.sends) == 2, second.error
    _revise(h.monkeypatch, "_OUTPUT_RULE")
    _restart(h.ctx)
    third = _review(h, reviewers=["t1"], reason=REASON)
    assert (third.status, third.dispatch_refusal_code, len(h.sends)) == ("pending", "LEGACY_BLOCKED", 2), third.error
    assert "REVIEW_CYCLES_EXHAUSTED" in third.error
    assert (_state(h).record_id, _state(h).status) == (second.record_id, "clean")


def test_a_spent_content_version_meets_its_exhaustion_instead_of_replaying_the_old_instructions(harness):
    """Outside a task, these bytes already spent their one paid wave and the identical request replays it free.
    Once the brief's instructions change, that record no longer answers: the request meets the spent allowance
    (a typed $0 refusal), nothing is reset, and the earlier verdict keeps the run gate."""
    h = harness
    h.ctx.task_id = ""
    h.monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    first = _review(h, reviewers=["t1"], reason=REASON)
    assert first.paid and _review(h, reviewers=["t1"], reason=REASON).reused and len(h.sends) == 1
    _revise(h.monkeypatch, "_PREAMBLE")
    _restart(h.ctx)
    refused = _review(h, reviewers=["t1"], reason=REASON)
    assert (refused.status, refused.reused, refused.dispatch_refusal_code, len(h.sends)) == (
        "pending", False, "LEGACY_BLOCKED", 1), refused.error
    assert "REVIEW_CYCLES_EXHAUSTED" in refused.error and "manual lane of skill weather" in refused.error
    assert (_state(h).record_id, _state(h).status) == (first.record_id, "clean")


def test_content_panel_rebuttal_and_the_authors_ask_stay_material(harness):
    """Under the skill contract the bytes, the seated panel, a rebuttal and the author's goal and questions
    still make a new subject or round; only the identical request reuses."""
    from ouroboros.reviewer_slot_config import compose_review_panel
    from ouroboros.skill_loader import find_skill

    h = harness
    h.monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")
    first = _review(h, reviewers=["t1"], reason=REASON)
    assert _review(h, reviewers=["t1"], reason=REASON).record_id == first.record_id and len(h.sends) == 1
    rebutted = _review(h, reviewers=["t1"], reason=REASON, review_rebuttal="The finding cites a helper nobody calls.")
    assert rebutted.paid and len(h.sends) == 2, rebutted.error
    other = _review(h, reviewers=["s1"], reason=REASON)
    assert other.paid and len(h.sends) == 3, other.error

    skill = find_skill(h.drive, "weather", repo_path="")
    subject = review_skill.skill_subject(h.ctx, skill, skill.content_hash, drive_root=h.drive, governance_root=h.system)

    def asked(**ask):
        return run_review_change(h.ctx, subject="skill", reviewers=["t1"], reason=REASON, _frozen_skill=subject,
                                 _panel=compose_review_panel(["t1"], REASON, surface="skill"), **ask)

    goal = asked(goal="Does the helper keep its output bounded?")
    assert goal["reused"] is False and len(h.sends) == 4
    assert asked(goal="Does the helper keep its output bounded?")["record_id"] == goal["record_id"]
    questions = asked(goal="Does the helper keep its output bounded?", author_questions=["Is the timeout honoured?"])
    assert questions["reused"] is False and len(h.sends) == 5

    (h.skill_dir / "scripts" / "hello.py").write_text("print('changed bytes')\n", encoding="utf-8")
    changed = _review(h, reviewers=["t1"], reason=REASON)
    assert changed.paid and changed.content_hash != first.content_hash and len(h.sends) == 6
