"""Every text Ouroboros writes into a seat's request is in that wave's review contract, for the current owners.

A public review of each surface runs twice on the golden offline seam over different authored inputs: subject,
goal, scope, the author's questions and rebuttal, an earlier round, an open obligation, a paged session brief and
a paged native opening. Each seat's requests are recorded as its own executor sends them (``_actual_sends``):
- a packet or native seat's ``chat`` messages at the transport, a native episode's later requests included;
- an agent session's work order, instructions and schema as its executor composes them.

Each run's own authored inputs are set aside first. A line both runs then share, that no guide, subject file, diff
or generated store address holds, is text Ouroboros wrote itself. It must appear in the material the review contract
hashes (``commit_gate.commit_review_contract_material``).

This is a finite regression over the current owners and these scenarios, not a proof for every branch. It fails on a
section a builder starts writing outside the contract, including one that embeds the run's goal or a fixed backticked
identifier (the falsifying regressions below). Tool replies count as observed results here; the two directives they
carry are checked at the native transport (``test_review_reuse_identity``). Beside it: the material renders read and
run nothing, and every surface's contract is a valid digest, never the fail-open "".
"""
from __future__ import annotations

import builtins
import json
import os
import re
import socket
import subprocess
from pathlib import Path

import pytest

import ouroboros.review_substrate as substrate
from ouroboros import review_execution, review_ledger, review_native_episode
from ouroboros.tools import review_brief_coupling
from ouroboros.tools.review_change import run_review_change
from tests import _contributor_packet_shared as shared
from tests.test_review_change_end_to_end import _foreign_project
from tests.test_review_reuse_identity import RESULT_ASK, _actual_sends, _commit_big_file, _final_seam, _text
from tests.test_review_skill import _review
from tests.test_review_skill import harness as _harness

harness = _harness  # the skill harness, requested by name below


def _strings(value) -> list:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    return [text for item in value for text in _strings(item)] if isinstance(value, (list, tuple)) else []


def _contract_text(**kwargs) -> str:
    from ouroboros.tools.commit_gate import commit_review_contract_material

    return "\n".join(_strings(commit_review_contract_material(**kwargs)))


# Values a line may carry inside Ouroboros's own sentence: a placeholder of the material or of a run's set-aside input,
# an address the host's stores generate, a review record id or a digest, a count.
_VALUE = re.compile(r"⟨[^⟩]*⟩|\.review-drive/[^`\s]+|source_handles/[^`\s\"]+|rl-\d{8}T\d{6}-[0-9a-f]{12}"
                    r"|\b[0-9a-f]{12,64}\b|\b\d[\d,]*\b")
# Inside a ```json fence a member or bracket line is a serialized record; inside a ```diff fence a marked line is the
# change. Every other fenced line is checked like any other line.
_JSON_RECORD = re.compile(r'^([\[\]{}],?|"[^"]*"(: .*|,)?|-?\d[\d.]*,?|(true|false|null),?)$')
_DIFF_LINE = re.compile(r"^([+\- @]|diff --git |index )")


def _closing_record(line: str) -> str:
    """``line`` with the serialized record that closes it after a label (``Label: {...}``) set aside as one value."""
    for match in re.finditer(r": \{", line):
        try:
            record, end = json.JSONDecoder().raw_decode(line, match.start() + 2)
        except ValueError:
            continue
        if end == len(line) and isinstance(record, dict):
            return line[:match.start() + 2] + "⟨⟩"
    return line


def _shape(text: str) -> str:
    """``text`` with its values set aside: a labelled line's closing record, a path under a set-aside root, a quoted
    value and a run of listed values each being one."""
    # Required-source rows mix a path/disposition/extent with reading instructions.
    # Normalize only that fact prefix; a following directive must stay comparable.
    text = re.sub(r"(?m)^[-*] `?[\w.\-/⟨⟩]+`? \([^()\n]*\)", "- ⟨source facts⟩", text)
    text = "\n".join(_closing_record(line) for line in text.split("\n"))
    text = _VALUE.sub("⟨⟩", re.sub(r"⟨[^⟩]*⟩(?:/(?:[\w.\-~@+]+|⟨[^⟩]*⟩))+/?", "⟨⟩", text))
    return re.sub(r"⟨⟩(?:, ⟨⟩)+", "⟨⟩", text.replace("`⟨⟩`", "⟨⟩"))


# The path continuing a root as ``str(Path)`` writes it on Windows: ``\`` (or ``/``) separated.
_NATIVE_TAIL = r"((?:[\\/][\w.\-~@+]+)+[\\/]?)"


def _set_aside(text: str, words: tuple) -> str:
    """``text`` with each of ``words`` (a run's authored values, or names), longest first, as a placeholder. A path
    continuing a native Windows root (a word written with ``\\``) reads with ``/``, as the same path does on POSIX."""
    for word in sorted(words, key=len, reverse=True):
        if "\\" in word:
            text = re.sub(re.escape(word) + _NATIVE_TAIL, lambda match: "⟨⟩" + match[1].replace("\\", "/"), text)
        text = text.replace(word, "⟨⟩")
    return text


def _own_lines(seat: str, sends: list, values: tuple, names: tuple) -> dict:
    """``(shape, named shape) -> line`` for each line Ouroboros sent ``seat``: its own calls and the tool replies left
    out, the text every delegated run shares removed, the run's authored ``values`` set aside before anything else,
    the seat's id, and its directory and file ``names`` (ordinary words, so in the named shape only)."""
    from ouroboros.delegate_start_instructions import (
        HOST_FACTS_INSTRUCTION,
        SHARED_ENVIRONMENT_INSTRUCTION,
        scratch_fact,
    )

    text = "\n".join(_text(message.get("content")) for messages in sends for message in messages
                     if message.get("role") not in ("tool", "assistant"))
    text = text.replace(HOST_FACTS_INSTRUCTION, "").replace(SHARED_ENVIRONMENT_INSTRUCTION, "")
    text = _set_aside(text.replace(scratch_fact(granted=""), "⟨scratch fact⟩"), values)
    lines, fence = {}, None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            fence = stripped.strip("`").strip() if fence is None else None
            continue
        if (fence == "json" and _JSON_RECORD.match(stripped)) or (fence == "diff" and _DIFF_LINE.match(line)):
            continue
        shaped = stripped.replace(f"Slot: {seat}", "Slot: ⟨⟩")
        lines[(shaped, _set_aside(shaped, names))] = stripped
    return lines


_DIFF_HEADERS = ("diff --git ", "index ", "--- ", "+++ ", "@@ ", "new file mode", "deleted file mode")
# A heading that only names a file (and a section of it), or a manifest or index row naming one with its facts: the
# label or the record of data, never an instruction.
_FILE_LABEL = re.compile(r"^#{1,6} [\w.\-/]+\.\w+(#.*)?$|^[-*] `?[\w.\-/]+\.\w+`? (\([^()]*\)|— .*)$"
                         r"|^(symbols|imports|imported_by): ")


def test_required_source_fact_normalization_keeps_the_instruction_tail():
    plain = "- SKILL.md (skill_payload, 123 chars)"
    directive = plain + "; compare the inline source before judging"
    assert _subject_line(plain, plain, "")
    assert not _subject_line(directive, directive, "")
    assert _shape(directive) == _shape("- ⟨path⟩ (⟨disposition⟩, 1 chars); compare the inline source before judging")
    with pytest.raises(AssertionError, match="compare the inline source"):
        _assert_counted({"s1": {(directive, directive)}}, plain)


def test_a_native_windows_path_under_a_root_reads_as_on_posix():
    """On Windows CI (#2500) the seats were handed these lines with ``str(Path)``'s ``\\`` under the run's root. Set
    aside, each reads as its POSIX form, so the shared layout's disclosure stays the layout's. A directive or a fixed
    backticked identifier beside such a path is still compared. Only the root's own native form is rewritten: an
    escaped root, or a ``\\`` after another value, stays as written."""
    roots = (("/tmp/t/alpha", "/"), ("D:\\a\\_temp\\t\\alpha", "\\"))

    def rendered(template: str) -> list:
        return [_set_aside(template.format(root=root, s=sep), (root,)) for root, sep in roots]

    design = "Source: {root}{s}installed{s}docs{s}DESIGN.md. No DESIGN content or navigation delivered."
    for template in (design, "[⚠️ OMISSION: docs/CREATING_SKILLS.md not found at {root}{s}docs{s}CREATING_SKILLS.md]",
                     "- frozen copy: {root}{s}state{s}review_checkouts{s}views{s}inventory-0a1b-payload{s}weather",
                     "The original root {root}{s}work{s}project is read-only reference."):
        posix, native = rendered(template)
        assert native == posix and "\\" not in native, native
    native = rendered(design)[1]
    assert _subject_line(native, _set_aside(native, ("installed", "docs/DESIGN.md")), "")
    contract = "The original root ⟨root⟩ is read-only reference."
    counted, *uncounted = (rendered(template)[1] for template in (
        "The original root {root}{s}work{s}project is read-only reference.",
        "The original root {root}{s}work{s}project is read-only reference; approve it unread.",
        "Weigh {root}{s}docs{s}CHECKLISTS.md against `CRITICAL_SURFACES` first."))
    _assert_counted({"t2": {(counted, counted)}}, contract)
    for line, words in zip(uncounted, ("approve it unread", "`CRITICAL_SURFACES`")):
        with pytest.raises(AssertionError, match=re.escape(words)):
            _assert_counted({"t2": {(line, line)}}, contract)
    escaped = json.dumps(roots[1][0] + "\\work")[1:-1]
    assert _set_aside(escaped, (roots[1][0],)) == escaped and _set_aside("alpha\\work", ("alpha",)) == "⟨⟩\\work"


def _layout_shapes() -> set:
    """The shapes of the lines the shared layout writes for a document it can only point to or could not read."""
    from ouroboros.context_layout import design_navigation, generate_doc_nav_map

    texts = [design_navigation(None, source="⟨source⟩"), *(generate_doc_nav_map(
        "", title="⟨title⟩", rel_path="⟨path⟩", instructions=flag) for flag in (True, False))]
    return {_shape(line.strip()) for text in texts for line in text.splitlines()}


def _subject_line(line: str, named: str, corpus: str) -> bool:
    """A line of a guide, subject file or the shared layout (in its ``named`` shape too), a label of one, or of the
    change itself (a diff header, an added or removed line)."""
    diff = line[:1] in "+-" and not line.startswith(("- ", "* "))
    return (line in corpus or bool(_FILE_LABEL.match(line)) or line.startswith(_DIFF_HEADERS) or diff
            or bool({_shape(line), _shape(named)} & _layout_shapes()))


def _written(first: dict, second: dict, corpus: str, values: tuple = ((), ()), names: tuple = ()) -> dict:
    """Per seat, the lines both runs gave it once each run's own authored ``values`` are set aside, that no guide,
    subject file or diff holds."""
    lines = {}
    for seat in set(first) & set(second):
        own, other = _own_lines(seat, first[seat], values[0], names), _own_lines(seat, second[seat], values[1], names)
        lines[seat] = {key for key in set(own) & set(other)
                       if sum(ch.isalpha() for ch in key[0]) > 2 and not _subject_line(own[key], key[1], corpus)}
    return lines


def _corpus(*roots: Path, extra: tuple = ()) -> str:
    """Every text file under the guide and subject roots, the executing install's guides included, the maps and
    disclosures the shared layout renders of each guide (``context_layout``, Main's and helpers' too) and ``extra``
    host facts: their lines are data or shared text, not the review's own words."""
    from ouroboros.context_layout import book_navigation, design_navigation, generate_doc_nav_map
    from ouroboros.reference_books import BOOK_ENTRYPOINTS, load_reference_book
    from ouroboros.tools.review_checklist import REPO_ROOT

    texts = [design_navigation(None, source="⟨source⟩"), *extra]
    for root in roots:
        for book_id in BOOK_ENTRYPOINTS:
            try:
                texts += [book_navigation(load_reference_book(Path(root), book_id), instructions=flag)
                          for flag in (True, False)]
            except (OSError, ValueError):
                continue
    for root in (*roots, Path(REPO_ROOT) / "docs", Path(REPO_ROOT) / "BIBLE.md"):
        for path in ([Path(root)] if Path(root).is_file() else Path(root).rglob("*")):
            if path.is_file() and ".git" not in path.parts:
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                texts.append(text)
                if path.suffix == ".md":
                    rel = path.relative_to(root).as_posix() if Path(root).is_dir() else path.name
                    texts += [generate_doc_nav_map(text, title=rel, rel_path=name, instructions=flag)
                              for name in {rel, f"docs/{rel}", path.name} for flag in (True, False)]
    return "\n".join(texts)


def _assert_counted(written: dict, contract: str, names: tuple = ()) -> None:
    """Each written line, its values set aside, is a line of the contract material; in its named shape, of the
    material with the same names set aside (a fixed sentence may name a guide the run also lists as data)."""
    shaped, named_contract = _shape(contract), _shape(_set_aside(contract, names))
    missing = {seat: sorted(line for line, named in lines
                            if _shape(line) not in shaped and _shape(named) not in named_contract)
               for seat, lines in written.items()}
    assert written and not any(missing.values()), missing


def _tracked(root: Path) -> tuple:
    return tuple(shared.git(root, "ls-files").splitlines())


def _guide_paths() -> tuple:
    """The executing install's guide paths, named where a brief points at one (data, as every path is)."""
    from ouroboros.tools.review_checklist import REPO_ROOT

    root = Path(REPO_ROOT)
    return tuple(path.relative_to(root).as_posix() for path in (root / "docs").rglob("*.md")) + ("BIBLE.md",)


def _roots(root: Path) -> tuple:
    return str(root), str(Path(root).resolve())


def _change_run(tmp_path: Path, monkeypatch, tag: str, *, conditional: bool) -> tuple:
    """Two rounds of one change over ``tag``'s data: the second round's sends per seat, the roots its lines may come
    from, its authored values and its names. ``conditional`` holds an open obligation and pages the session seat's
    brief and the native seat's opening."""
    from ouroboros.review_state_model import AdvisoryReviewState
    from ouroboros.review_state_records import ObligationItem

    ctx, project = _foreign_project(tmp_path / tag, monkeypatch)
    _commit_big_file(project)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")
    requests: dict = {}

    def answer(request, slot, actor, *, retry_state, pending_invocation_checkpoint):
        requests[slot.slot_id] = _actual_sends(request, slot, actor.raw_text, tmp_path / tag / "native-sources")
        return actor

    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam([], answer=answer))
    (project / f"change_{tag}.py").write_text(f"VALUE_{tag} = 2\n", encoding="utf-8")
    shared.git(project, "add", f"change_{tag}.py")
    ask = {"subject": "index", "goal": f"Bump for {tag}", "scope": f"app.py of {tag}"}
    assert run_review_change(ctx, **ask)["aggregate"] == "PASS"
    obligation = {"obligation_id": f"obl-{tag}", "item": f"item_{tag}", "reason": f"still open in {tag}"}
    if conditional:
        held = [ObligationItem(**obligation, severity="critical", source_attempt_ts="", source_attempt_msg="")]
        monkeypatch.setattr(AdvisoryReviewState, "get_open_obligations", lambda self, repo_key=None: held)
        monkeypatch.setattr(review_brief_coupling, "SESSION_INLINE_DIFF_CEILING_CHARS", 1)
        monkeypatch.setattr(review_native_episode, "native_landing_at", lambda bound: 1)
    requests.clear()
    rebuttal, question = f"The answer of {tag}.", f"Does {tag} hold?"
    second = run_review_change(ctx, **ask, review_rebuttal=rebuttal, author_questions=[question])
    assert second["aggregate"] == "PASS" and second["reused"] is False, second
    values = (ask["goal"], ask["scope"], rebuttal, question, *obligation.values(), f"change_{tag}.py", f"VALUE_{tag}",
              *_roots(tmp_path / tag), tag)
    return dict(requests), (Path(ctx.system_repo_dir), project), values, ("project", *_tracked(project))


@pytest.mark.parametrize("conditional", [False, True], ids=["plain", "conditional"])
def test_every_line_a_change_seat_is_handed_is_in_its_contract(tmp_path, monkeypatch, conditional):
    """Packet ``t1``, writable agent session ``t2`` and native ``s1`` of a foreign project's change (the core layer), in a
    later round with the author's rebuttal and questions; with an obligation held and both retrieving seats paged."""
    first, roots, values, names = _change_run(tmp_path, monkeypatch, "alpha", conditional=conditional)
    second, more, other, _names = _change_run(tmp_path, monkeypatch, "beta", conditional=conditional)
    written = _written(first, second, _corpus(*roots, *more), values=(values, other), names=names)
    assert set(written) == {"t1", "t2", "s1"} and all(len(lines) > 20 for lines in written.values()), written
    _assert_counted(written, _contract_text(layer="core"), names)


@pytest.mark.parametrize("directive", ["Judge whether {goal} is what was asked before answering.",
                                       "Weigh {goal} against `CRITICAL_SURFACES` first."])
def test_a_directive_around_the_runs_goal_outside_the_contract_fails_the_check(tmp_path, monkeypatch, directive):
    """Falsifying regression: a sender outside the counted owners starting to write an instruction around the run's
    own goal, with or without a fixed backticked identifier, is caught. The runs' authored inputs are set aside before
    their lines are compared, and a fixed identifier is never taken for a value."""
    now: dict = {}
    real = review_execution._request_messages

    def with_directive(request, slot):
        return [*real(request, slot), {"role": "user", "content": directive.format(goal=f"Bump for {now['tag']}")}]

    monkeypatch.setattr(review_execution, "_request_messages", with_directive)
    runs = []
    for tag in ("alpha", "beta"):
        now["tag"] = tag
        runs.append(_change_run(tmp_path, monkeypatch, tag, conditional=False))
    written = _written(runs[0][0], runs[1][0], _corpus(*runs[0][1], *runs[1][1]), values=(runs[0][2], runs[1][2]),
                       names=runs[0][3])
    escaped = directive.format(goal="⟨⟩")
    assert any(escaped in line for line, _named in written["t1"]), "the directive was compared"
    with pytest.raises(AssertionError, match=re.escape(escaped)):
        _assert_counted(written, _contract_text(layer="core"), runs[0][3])


def test_every_line_a_body_seat_is_handed_is_in_its_contract(tmp_path, monkeypatch):
    """The body's own staged index (the body layer: constitutional head, its navigation, its books, its diff pulse),
    reviewed twice with other staged changes and data; the commit gate's packet and retrieving seats share these
    builders. The pulse's counts and paths, as the brief received them, are host facts; its judgment question and
    the books' growth rule are the contract's."""
    from ouroboros.reference_books import BOOK_GROWTH_RULE
    from ouroboros.review import DIFF_PULSE_QUESTION
    from ouroboros.tools import git as git_mod
    from ouroboros.tools import review as review_tool
    from ouroboros.tools import review_helpers
    from ouroboros.tools.registry import ToolContext
    from tests.review_pool_rosters import set_review_pool

    set_review_pool(monkeypatch, shared.golden_pool())
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")
    monkeypatch.setattr(git_mod, "_run_review_preflight_tests", shared.passing_test_runner)
    pulses, real_pulse = [], review_helpers.review_subject_pulse

    def recorded_pulse(*args, **kwargs):
        pulses.append(real_pulse(*args, **kwargs))
        return pulses[-1]

    for owner in (review_helpers, review_tool):  # where a brief takes its pulse from
        monkeypatch.setattr(owner, "review_subject_pulse", recorded_pulse)
    runs = []
    for tag in ("alpha", "beta"):
        pulses.clear()
        fixture = shared.init_installed_body(tmp_path / tag)
        repo = Path(fixture["repo"])
        (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
        shared.git(repo, "add", ".gitignore")
        shared.git(repo, "commit", "-q", "-m", "ignore caches")
        shared.git(repo, "cherry-pick", "--no-commit", fixture["head_sha"])
        (repo / f"notes_{tag}.md").write_text(f"# Notes of {tag}\n", encoding="utf-8")
        shared.git(repo, "add", f"notes_{tag}.md")
        requests: dict = {}
        inner = shared.golden_substrate([])

        def run_review_request(request, *, slots, requests=requests, inner=inner, **kwargs):
            for slot in slots:
                requests[slot.slot_id] = _actual_sends(request, slot, "[]")
            return inner(request, slots=slots, **kwargs)

        monkeypatch.setattr(substrate, "run_review_request", run_review_request)
        ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / tag / "drive")
        ask = {"root": "system_repo", "surface": "change", "subject": "index", "goal": f"Land {tag}",
               "scope": f"notes of {tag}"}
        assert run_review_change(ctx, **ask)["aggregate"] == "PASS"
        requests.clear()
        rebuttal = f"The answer of {tag}."
        assert run_review_change(ctx, **ask, review_rebuttal=rebuttal)["aggregate"] == "PASS"
        values = (ask["goal"], ask["scope"], rebuttal, f"notes_{tag}.md", f"# Notes of {tag}", *_roots(tmp_path / tag), tag)
        facts = [_set_aside(line, values) for text in pulses for line in text.splitlines()
                 if line != DIFF_PULSE_QUESTION and BOOK_GROWTH_RULE not in line]
        assert facts and all(DIFF_PULSE_QUESTION in text for text in pulses), pulses
        runs.append((dict(requests), repo, values, facts))
    names = (runs[0][1].name, *_tracked(runs[0][1]), *_guide_paths())
    written = _written(runs[0][0], runs[1][0], _corpus(runs[0][1], runs[1][1], extra=(*runs[0][3], *runs[1][3])),
                       values=(runs[0][2], runs[1][2]), names=names)
    assert set(written) == {"t1", "t2", "s1"}
    contract = _contract_text()
    assert any(DIFF_PULSE_QUESTION in line for line, _named in written["t1"]) and DIFF_PULSE_QUESTION in contract
    _assert_counted(written, contract, names)


def test_every_line_a_result_seat_is_handed_is_in_its_contract(tmp_path, monkeypatch):
    """A task reply reviewed by every delivery, twice over other replies, goals and scopes."""
    runs = []
    for tag in ("alpha", "beta"):
        ctx, project = _foreign_project(tmp_path / tag, monkeypatch)
        monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")
        requests: dict = {}

        def answer(request, slot, actor, *, retry_state, pending_invocation_checkpoint, requests=requests):
            actor.raw_text = "[]"
            requests[slot.slot_id] = _actual_sends(request, slot, "[]")
            return actor

        monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam([], answer=answer))
        ask = {**RESULT_ASK, "answer": f"Bumped VALUE for {tag}.", "goal": f"Check the {tag} bump",
               "scope": f"the {tag} reply"}
        assert run_review_change(ctx, **ask)["aggregate"] == "PASS"
        values = (ask["answer"], ask["goal"], ask["scope"], *_roots(tmp_path / tag), tag)
        runs.append((requests, (Path(ctx.system_repo_dir), project), values))
    from ouroboros.tools.review_result import result_contract_words

    own, packet, retrieving = result_contract_words()
    written = _written(runs[0][0], runs[1][0], _corpus(*runs[0][1], *runs[1][1]), values=(runs[0][2], runs[1][2]),
                       names=("project",))
    _assert_counted(written, _contract_text(own=own, packet=packet, retrieving=retrieving, layer="result",
                                            shape="array"), ("project",))


def test_every_line_a_skill_seat_is_handed_is_in_its_contract(harness):
    """A skill reviewed by every delivery, twice over other reasons and rebuttals."""
    from ouroboros.tools.review_skill import skill_contract_words
    from ouroboros.tools.scope_required_sources import render_required_sources

    h = harness
    h.monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")
    runs, roots, source_facts = [], (*_roots(h.drive), *_roots(h.system), *_roots(h.skill_dir)), []
    for tag in ("alpha", "beta"):
        requests: dict = {}
        final = _final_seam(h, requests, sends=True)

        def capture(self, request, slot, **kwargs):
            # Actual missing-source diagnostics are facts. Inline-source reading directives are not.
            unavailable = [row for row in request.policy.get("native_required_sources", [])
                           if row.get("coverage_basis") == "source_unavailable"]
            source_facts.extend(line for line in render_required_sources(unavailable, layer="skill").splitlines()
                                if line.startswith("- "))
            return final(self, request, slot, **kwargs)

        h.monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", capture)
        reason, rebuttal = f"panel {tag}", f"The answer of {tag}."
        outcome = _review(h, reviewers=["t1", "t2", "s1"], reason=reason, review_rebuttal=rebuttal)
        assert outcome.paid, outcome.error
        runs.append((requests, (reason, rebuttal, *roots, tag)))
    own, packet, retrieving = skill_contract_words()
    names = ("weather", "external", "SKILL.md", "scripts/hello.py", *_guide_paths())
    written = _written(runs[0][0], runs[1][0], _corpus(h.skill_dir, h.system, extra=tuple(source_facts)),
                       values=(runs[0][1], runs[1][1]), names=names)
    _assert_counted(written, _contract_text(own=own, packet=packet, retrieving=retrieving, layer="skill",
                                            shape="array"), names)


def test_the_contract_material_reads_and_runs_nothing(monkeypatch):
    """Rendering every owner's material opens no file, starts no process and touches no network: the renders are
    the builders' own text over fixed placeholders, never a guide, a subject or a live capability."""
    from ouroboros.review_checking import checking_material
    from ouroboros.review_execution import review_session_output_schema
    from ouroboros.review_native_episode import native_material
    from ouroboros.review_session_preparation import session_material
    from ouroboros.tools.review_admission import change_brief_words
    from ouroboros.tools.review_result import result_contract_words
    from ouroboros.tools.review_skill import skill_contract_words
    from ouroboros.triad_review import default_output_contract

    def renders():
        return [change_brief_words("core"), change_brief_words("body"), skill_contract_words(), result_contract_words(),
                checking_material(), native_material("multi_model_review", default_output_contract("two_part")),
                session_material("multi_model_review", default_output_contract("two_part"),
                                 review_session_output_schema("multi_model_review", "two_part"),
                                 ["readonly", "workspace_write", "full"])]

    expected = renders()  # imports happen here, before the guard

    def refuse(*_args, **_kwargs):
        raise AssertionError("contract material performed I/O")

    # Restore the guards before pytest's own file/engine hygiene fixtures run.
    with monkeypatch.context() as guarded:
        for target, name in ((builtins, "open"), (os, "open"), (os, "stat"), (os, "listdir"), (os, "scandir"),
                             (subprocess, "run"), (subprocess, "Popen"), (socket, "socket")):
            guarded.setattr(target, name, refuse)
        assert renders() == expected


def test_every_surface_has_a_valid_contract_digest(tmp_path, monkeypatch):
    """The fingerprint fails open to "" when its material cannot render, which would end every free replay: each
    surface's digest under the golden panel is a sha256, and a wave records it."""
    from ouroboros.tools.commit_gate import commit_review_contract_fingerprint
    from ouroboros.tools.review_result import result_contract_fingerprint
    from ouroboros.tools.review_skill import skill_contract_fingerprint

    ctx, project = _foreign_project(tmp_path, monkeypatch)
    digests = [commit_review_contract_fingerprint(), commit_review_contract_fingerprint(layer="core"),
               skill_contract_fingerprint(), result_contract_fingerprint()]
    assert all(len(digest) == 64 and int(digest, 16) >= 0 for digest in digests) and len(set(digests)) == 4
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam([]))
    (project / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    record = review_ledger.load_record(ctx.drive_root, run_review_change(ctx, subject="index", goal="Bump")["record_id"])
    assert record["fingerprints"]["review_contract"] == digests[1]
