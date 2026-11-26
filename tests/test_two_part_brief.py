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


def _packet_brief(tmp_path, installed, subject, *, whole=False):
    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(repo_dir=str(subject), drive_root=str(data), task_id="t-rows", task_metadata=None,
                          _review_history=[], _review_advisory=[], _review_iteration_count=0,
                          drive_logs=lambda: data / "logs")
    spec = ReviewSubjectSpec(root_kind="system_repo", root=str(subject), kind="index", surface="commit_gate",
                             layer="body", governance_root=str(installed))
    brief = build_two_part_brief(freeze_subject(ctx, spec), PACKET, goal="g", commit_message="m")
    return brief if whole else brief["system"]


def _module_block(system):
    from ouroboros.tools.governance_module_rows import HEADING

    start = system.index(HEADING)
    return system[start:system.index("\n\n", start)]


def test_a_packet_seat_gets_one_installed_owner_row_per_path_and_no_other_module(tmp_path):
    installed, subject = _installed_and_subject(tmp_path)
    (subject / "ouroboros" / "a.py").write_text('"""STAGED lead of a."""\n', encoding="utf-8")
    _git(subject, "mv", "ouroboros/c.py", "ouroboros/c\u00e9.py")  # a rename to a non-ASCII name
    _git(subject, "add", "-A")
    (subject / "ouroboros" / "a.py").write_text('"""WORKTREE lead of a."""\n', encoding="utf-8")  # after staging
    system = _packet_brief(tmp_path, installed, subject)
    assert _module_block(system).splitlines()[1:] == [
        "- ouroboros/a.py (modified in this change; domain D06: Review stack; manuals: docs/architecture/06-4.md)",
        "- ouroboros/c\u00e9.py (renamed from ouroboros/c.py in this change; domain D06: Review stack; "
        "manuals: docs/architecture/06-4.md)"]  # owned where its source was
    assert "ouroboros/b.py" not in system and "Lead of b." not in system  # no other module of the domain
    assert "lead of a." not in _module_block(system) and "STAGED lead of a." in system  # the change carries it


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


def test_an_added_and_a_deleted_module_are_labelled_by_their_git_change_on_both_gates(tmp_path):
    """Labels come from the frozen subject's name status, never from presence in the installed listing: on a
    separate installed tree and on the in-place gate, whose installed tree is the author's own checkout."""
    installed, subject = _installed_and_subject(tmp_path)
    (subject / "ouroboros" / "new.py").write_text('"""Proposed new module."""\n', encoding="utf-8")
    manifest = (subject / "ouroboros" / "domains.toml").read_text(encoding="utf-8")
    (subject / "ouroboros" / "domains.toml").write_text(
        manifest.replace('"ouroboros/b.py" = "D06"\n', "") + '"ouroboros/new.py" = "D06"\n', encoding="utf-8")
    (subject / "ouroboros" / "b.py").unlink()
    _git(subject, "add", "-A")
    separate = _module_block(_packet_brief(tmp_path, installed, subject))
    assert "- ouroboros/new.py (added in this change; the installed map has no owner for it)" in separate
    assert "- ouroboros/b.py (deleted in this change; domain D06: Review stack;" in separate  # installed owner
    in_place = _module_block(_packet_brief(tmp_path, subject, subject))
    assert "- ouroboros/new.py (added in this change; domain D06: Review stack;" in in_place
    assert "- ouroboros/b.py (deleted in this change; the installed map has no owner for it)" in in_place


def test_the_rows_give_way_first_in_the_fit_ladder_and_a_missing_manifest_is_said(tmp_path, monkeypatch):
    from ouroboros.tools import review as review_tool, review_admission
    from ouroboros.tools.commit_gate import commit_review_contract_material
    from ouroboros.tools.governance_module_rows import HEADING, MODULE_ROWS_PATH, UNAVAILABLE, YIELD_REASON, YIELDED
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool

    def strings(value):
        if isinstance(value, str):
            return [value]
        items = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else ()
        return [text for item in items for text in strings(item)]

    set_review_pool(monkeypatch, pool_roster(pool_seat("packet-seat", "openai/gpt-y", delivery="packet")))
    contract = "\n".join(strings(commit_review_contract_material()))  # a pool with a packet seat counts the rows
    assert HEADING in contract and UNAVAILABLE in contract and YIELDED.format(count="⟨count⟩") in contract
    set_review_pool(monkeypatch, pool_roster(pool_seat("native-seat", "openai/gpt-y", delivery="native")))
    assert HEADING not in "\n".join(strings(commit_review_contract_material()))  # and only then
    installed, subject = _installed_and_subject(tmp_path / "roomy")
    (subject / "ouroboros" / "a.py").write_text('"""Changed."""\n', encoding="utf-8")
    _git(subject, "add", "-A")
    prompts = []
    real = review_admission.assemble_packet_prompt
    monkeypatch.setattr(review_admission, "assemble_packet_prompt",
                        lambda **kw: prompts.append(real(**kw)) or prompts[-1])
    roomy = _packet_brief(tmp_path / "roomy", installed, subject)
    assert HEADING in roomy and YIELDED.format(count=1) not in roomy
    # One token short of the whole packet: the rows give way, nothing of the change does, and both records say so.
    limit = review_tool.estimate_tokens(prompts[-1][0]) - 1
    monkeypatch.setattr(review_tool, "_quorum_input_token_limit", lambda *a, **k: limit)
    tight = _packet_brief(tmp_path / "roomy", installed, subject, whole=True)
    assert HEADING not in tight["system"] and YIELDED.format(count=1) in tight["system"]
    assert '"""Changed."""' in tight["system"] and "Full post-change snapshots were omitted" not in tight["system"]
    assert [(r["disposition"], r["reason"]) for r in tight["manifest"]["governance_manifest"]
            if r["path"] == MODULE_ROWS_PATH] == [("navigation", YIELD_REASON)]
    monkeypatch.undo()
    bare, subject = _installed_and_subject(tmp_path / "bare", manifest=False)
    (subject / "ouroboros" / "a.py").write_text('"""Changed."""\n', encoding="utf-8")
    _git(subject, "add", "-A")
    unavailable = _packet_brief(tmp_path / "bare", bare, subject)
    assert UNAVAILABLE in unavailable and "Errno" not in unavailable
    (bare / "ouroboros" / "domains.toml").write_text("[graph]\nallowed = 7\n", encoding="utf-8")  # valid TOML, wrong type
    assert UNAVAILABLE in _packet_brief(tmp_path / "bare", bare, subject)


def test_owner_rows_never_cost_a_tight_route_the_change(tmp_path, monkeypatch):
    """A one-line change to a real web module, the installed governance root this checkout, the real builder and
    cold sizing: at every window the packet delivers what the same selector without owner rows (the target's)
    delivers. At 100,000 the changed line arrives (the rows give way, the snapshot follows the target's ladder), at
    131,072 the file's whole source, at 256,000 everything, rows included."""
    import pathlib

    from ouroboros.tools import governance_module_rows as gmr, review as review_tool
    from ouroboros.tools.registry import ToolContext

    root, path = pathlib.Path(__file__).resolve().parents[1], "web/modules/account_resources.js"
    witness = "export function resourceReadGap(action = {}) {"
    _installed, subject = _installed_and_subject(tmp_path)
    (subject / path).parent.mkdir(parents=True, exist_ok=True)
    (subject / path).write_bytes((root / path).read_bytes())
    _git(subject, "add", "-A")
    _git(subject, "commit", "-qm", "identical source")
    with (subject / path).open("a", encoding="utf-8") as stream:
        stream.write("\n// Review one trailing comment.\n")
    _git(subject, "add", "-A")
    ctx = ToolContext(repo_dir=subject, drive_root=tmp_path / "drive")
    ctx._review_iteration_count = 0
    frozen = freeze_subject(ctx, ReviewSubjectSpec("system_repo", str(subject), "index", surface="commit_gate",
                                                   governance_root=str(root), layer="body"))

    def outcome(window, rows):
        monkeypatch.setattr(review_tool, "reviewer_context_window", lambda *a, **k: window)
        monkeypatch.setattr(gmr, "module_rows", real_rows if rows else lambda *a, **k: gmr.ModuleRows())
        text = build_two_part_brief(frozen, PACKET, drive_root=tmp_path / "drive", goal="Review these edits")["system"]
        return (("Full post-change snapshots were omitted" in text, witness in text,
                 "// Review one trailing comment." in text), gmr.HEADING in text, "were NOT delivered: with them" in text)

    real_rows = gmr.module_rows
    for window, expected in ((100_000, (True, False, True)), (131_072, (False, True, True)),
                             (256_000, (False, True, True))):
        delivered, rows_in, yielded = outcome(window, True)
        assert delivered == outcome(window, False)[0] == expected, window
        assert (rows_in, yielded) == ((False, True) if window == 100_000 else (True, False)), window


def _edited_rows(needle, replacement):
    """``governance_module_rows`` with one fixed text edited, as a separate module."""
    import inspect
    import sys
    import types

    from ouroboros.tools import governance_module_rows as gmr

    source = inspect.getsource(gmr)
    assert source.count(needle) == 1, needle
    module = types.ModuleType("ouroboros.tools.edited_owner_rows")
    sys.modules[module.__name__] = module
    exec(compile(source.replace(needle, replacement), str(gmr.__file__), "exec"), module.__dict__)
    return module


EDITS = [('NO_MANUALS = "none named"', 'NO_MANUALS = "none published"'),
         ('change; surface {surface}; manuals', 'change; surface area {surface}; manuals'),
         ('"T": "type-changed"', '"T": "mode-changed"'), ('TOUCHED = "touched"', 'TOUCHED = "named"'),
         ("one row per path of the change;", "a row per path of the change;"),
         ("change; the installed map has no owner for it)", "change; the installed map owns it nowhere)"),
         ("they give way before any part", "they yield before any part"),
         ('"unavailable; the review', '"unreadable; the review')]


def test_every_fixed_rows_text_is_in_the_packet_contract_and_only_there(monkeypatch):
    """Editing any fixed text the rows render changes the commit gate's contract fingerprint for a pool with a
    packet seat; a native-only pool keeps the identical material and hash."""
    from ouroboros.tools import commit_gate, governance_context as gc
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool

    def fingerprints():
        set_review_pool(monkeypatch, pool_roster(pool_seat("packet-seat", "openai/gpt-y", delivery="packet")))
        packet = commit_gate.commit_review_contract_fingerprint()
        set_review_pool(monkeypatch, pool_roster(pool_seat("native-seat", "openai/gpt-y", delivery="native")))
        return packet, commit_gate.commit_review_contract_fingerprint(), commit_gate.commit_review_contract_material()

    packet, native, material = fingerprints()
    assert packet and native and packet != native
    for needle, replacement in EDITS:
        with monkeypatch.context() as patched:
            patched.setattr(gc, "gmr", _edited_rows(needle, replacement))
            edited_packet, edited_native, edited_material = fingerprints()
        assert edited_packet != packet, needle
        assert (edited_native, edited_material) == (native, material), needle


def test_an_edited_owner_row_is_never_reused_as_a_settled_verdict(tmp_path, monkeypatch):
    """Astra's replay: a settled PASS is reused for the identical subject; once a fixed text of the rows the packet
    seat received changes, the same subject is reviewed again and the new text is what the seat is sent."""
    from ouroboros import review_substrate as substrate
    from ouroboros.tools import git as git_mod, governance_context as gc
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_change import run_review_change
    from tests import _contributor_packet_shared as shared
    from tests.review_pool_rosters import set_review_pool
    from tests.test_review_reuse_identity import _actual_sends

    set_review_pool(monkeypatch, shared.golden_pool())
    for name, value in (("OUROBOROS_REVIEW_ENFORCEMENT", "blocking"), ("OUROBOROS_RUNTIME_MODE", "pro"),
                        ("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(git_mod, "_run_review_preflight_tests", shared.passing_test_runner)
    repo = pathlib.Path(shared.init_installed_body(tmp_path / "world")["repo"])
    for path, text in ((".gitignore", "__pycache__/\n"), ("web/app.js", "// An installed web app.\n"),
                       ("ouroboros/domains.toml", '[domains]\nD01 = "Core"\n[references]\n"web/" = []\n[modules]\n')):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8")
    shared.git(repo, "add", "-A")
    shared.git(repo, "commit", "-q", "-m", "installed web app")
    (repo / "web/app.js").write_text("// An installed web app (changed).\n", encoding="utf-8")
    shared.git(repo, "add", "web/app.js")
    ctx, sent, inner = ToolContext(repo_dir=repo, drive_root=tmp_path / "drive"), [], shared.golden_substrate([])
    ask = dict(root="system_repo", surface="change", subject="index", goal="Review the one edit", scope="the web app")

    def run(request, *, slots, **kwargs):
        sent.extend(str(_actual_sends(request, slot, "[]")) for slot in slots)
        return inner(request, slots=slots, **kwargs)
    monkeypatch.setattr(substrate, "run_review_request", run)
    first = run_review_change(ctx, **ask)
    row = "- web/app.js (modified in this change; surface web/; manuals: none named)"
    assert first["aggregate"] == "PASS" and not first["reused"], first
    assert any(row in text for text in sent), [text[text.find("## Governance navigation") - 600:][:900] for text in sent]
    again = run_review_change(ctx, **ask)
    assert again["reused"] and again["record_id"] == first["record_id"]
    for needle, replacement, word in (EDITS[0] + ("none published",), EDITS[1] + ("surface area web/",)):
        count = len(sent)
        with monkeypatch.context() as patched:
            patched.setattr(gc, "gmr", _edited_rows(needle, replacement))
            edited = run_review_change(ctx, **ask)
        assert not edited["reused"] and edited["record_id"] != first["record_id"], needle
        assert len(sent) > count and any(word in text for text in sent[count:]), needle
