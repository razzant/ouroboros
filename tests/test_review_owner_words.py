"""Reviewers see the owner's words that caused the work.

The triad, scope and advisory reviewers read the change against the author's intent;
``build_goal_section`` now adds, right after that intent, the host-attested words of
the owner the work answers (``owner_words.owner_words_text``: a root's own corpus, a
child's inherited words, an absence line with the host marker when there are none).
An empty section keeps every prompt byte-identical to a review without it, and the
words ride the dynamic half, so the cache-marked prefixes of the triad and the scope
brief do not move. The plan review puts them right after its objective; a replay of
the same author request takes the recorded words, so an owner letter between waves
cannot mint a paid wave. Reviewers still get no memory and no story: they stay independent.
Every rule is checked in both directions.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import owner_words as ow
from ouroboros.tools.review_helpers import build_goal_section
from ouroboros.utils import append_jsonl
from tests.test_plan_review_engine import CLEAN, DECK_SPEC, _call, _state, _user_text
from tests.test_plan_review_engine import harness as _plan_review_harness

harness = _plan_review_harness  # the plan-review fixture, under the name its tests take

OWNER = "Fix the login timeout,\nand keep the old cookie name"
CHILD_ASSIGNMENT = "Collect the timeout traces"
ABSENT_CONSCIOUS = "No words of my human are recorded for this work (host marker: initiator=consciousness)."

# ``build_goal_section`` output captured from the base code (before ``owner_words`` existed).
BASE_GOAL_WITH_BODY = (
    "## Intended transformation\n\nSource: goal\n\nShip the fix\n\nUse this to judge whether the change actually "
    "completed the intended work,\nincluding tests, prompts, docs, architecture touchpoints, and adjacent surfaces\n"
    "that may have been forgotten.\n\n\n## Informational context — commit message (narrative, NOT a contract)\n\n"
    "fix: retry\n\nbody line\n\nThe text above is a narrative artifact written for humans reading the\ngit log. "
    "Do NOT audit its wording as a contract against the code — use\nthe staged diff, checklists, and intent above "
    "to judge the change."
)
BASE_SUBJECT_ONLY = (
    "## Intended transformation\n\nSource: commit message (subject)\n\nfix: retry\n\nUse this to judge whether the "
    "change actually completed the intended work,\nincluding tests, prompts, docs, architecture touchpoints, and "
    "adjacent surfaces\nthat may have been forgotten."
)


def _attrs(kind: str) -> dict:
    """The owner-words facts of a run: a root that heard its owner, a child, a consciousness root."""
    if kind == "root":
        return {"task_metadata": {"root_task_id": "root"},
                "_owner_directives": [{"source": "initial_user", "content": OWNER},
                                      {"source": "initial_text", "content": "not the owner's"}]}
    if kind == "child":
        inherited = [{"text": OWNER, "source": "initial_user", "carrier": "ctx", "task_id": "root", "ts": "", "ref": ""}]
        return {"task_metadata": {"root_task_id": "root", "delegation_role": "subagent", ow.FIELD: inherited},
                "_owner_directives": [{"source": "initial_text", "content": CHILD_ASSIGNMENT}]}
    return {"task_metadata": {"initiator": "consciousness"}, "_owner_directives": []}


def _section(kind: str, task_id: str = "root") -> str:
    return ow.owner_words_text(SimpleNamespace(task_id=task_id, **_attrs(kind)))


# --- the goal section --------------------------------------------------------------------------------------

def test_no_owner_words_keep_the_goal_section_byte_identical():
    for owner_words in ("", "  \n"):
        assert build_goal_section("Ship the fix", "", "fix: retry\n\nbody line", owner_words) == BASE_GOAL_WITH_BODY
        assert build_goal_section("", "", "fix: retry", owner_words=owner_words) == BASE_SUBJECT_ONLY
    assert build_goal_section("Ship the fix", "", "fix: retry\n\nbody line") == BASE_GOAL_WITH_BODY


def test_the_owner_words_follow_the_intended_transformation():
    words = _section("root")
    assert words.startswith("## Words of my human that caused this work (verbatim, host-attested)\n")
    text = build_goal_section("Ship the fix", "", "fix: retry\n\nbody line", owner_words=words)
    head, _, tail = BASE_GOAL_WITH_BODY.partition("\n\n\n## Informational context")
    assert text == f"{head}\n\n\n{words}\n\n\n## Informational context{tail}"
    assert text.index("## Intended transformation") < text.index(OWNER) < text.index("## Informational context")
    assert "not the owner's" not in text
    assert build_goal_section("", "", "fix: retry", owner_words=words) == f"{BASE_SUBJECT_ONLY}\n\n\n{words}"
    assert build_goal_section("", "", "fix: retry", owner_words=ABSENT_CONSCIOUS).endswith(f"\n\n\n{ABSENT_CONSCIOUS}")


# --- triad --------------------------------------------------------------------------------------------------

def _triad_prompt(tmp_path, monkeypatch, kind: str) -> tuple[str, int, str]:
    """One real triad assembly (``_run_unified_review``) up to the panel call:
    ``(packet prompt, its stable prefix length, the retrieving rows' session task)``."""
    import ouroboros.tools.review as review
    import ouroboros.tools.review_binary_context as rbc

    captured = {}

    def fake_run_cmd(cmd, cwd=None):
        if cmd == ["git", "diff", "--cached", "--name-status"]:
            return "M\tx.py"
        if cmd == ["git", "diff", "--cached", "--name-only"]:
            return "x.py"
        if cmd[:3] == ["git", "diff", "--cached"]:
            return "diff --git a/x.py b/x.py\n+x = 1"
        return ""

    def capture_review(*_args, **kwargs):
        captured.update(prompt=kwargs["prompt"], stable=kwargs["stable_prefix_len"], task=kwargs["session_task"])
        return json.dumps({"results": []})

    monkeypatch.setattr(review, "run_cmd", fake_run_cmd)
    monkeypatch.setattr(rbc, "capture_staged_diff", lambda _repo, *, unified=3: "diff --git a/x.py b/x.py\n+x = 1")
    monkeypatch.setattr(review, "_preflight_check", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(review, "_load_checklist_section", lambda: "checklist")
    monkeypatch.setattr(review, "load_governance_doc", lambda *_args, **_kwargs: "governance")
    monkeypatch.setattr(review, "build_touched_file_pack", lambda *_args, **_kwargs: ("files", []))
    monkeypatch.setattr(review._cfg, "get_review_models", lambda: ["test/reviewer"])
    monkeypatch.setattr(review._cfg, "get_review_enforcement", lambda: "blocking")
    monkeypatch.setattr(review, "_handle_multi_model_review", capture_review)
    ctx = SimpleNamespace(
        repo_dir=tmp_path, drive_root=tmp_path, task_id="root", _review_iteration_count=0,
        _last_review_block_reason="", _last_triad_models=[], _last_review_critical_findings=[],
        _last_triad_raw_results=[], _review_degraded_reasons=[], _review_history=[], **_attrs(kind))
    review._run_unified_review(ctx, "fix: login timeout", goal="GOAL_SENTINEL", scope="SCOPE_SENTINEL")
    return captured["prompt"], captured["stable"], captured["task"]


@pytest.mark.parametrize("kind", ["root", "child"])
def test_the_triad_packet_reads_the_words_in_its_dynamic_half(tmp_path, monkeypatch, kind):
    monkeypatch.setattr("ouroboros.reviewer_slot_config.DEFAULT_TRIAD_DELIVERY", "")  # pin the packet seat
    prompt, stable, _task = _triad_prompt(tmp_path, monkeypatch, kind)
    conscious, conscious_stable, _task = _triad_prompt(tmp_path, monkeypatch, "conscious")
    words = _section(kind)
    assert words in prompt and prompt.index(words) > stable > 0
    assert prompt.index("GOAL_SENTINEL") < prompt.index(words)
    assert CHILD_ASSIGNMENT not in prompt  # a child hands on the root's words, never its own assignment
    # The cache-marked prefix is the same bytes whatever the owner said.
    assert stable == conscious_stable and prompt[:stable] == conscious[:conscious_stable]
    assert ABSENT_CONSCIOUS in conscious and OWNER not in conscious and ABSENT_CONSCIOUS not in prompt


def test_the_retrieving_triad_reads_the_words_in_its_session_task(tmp_path, monkeypatch):
    """The shipped default triad reads the work itself: its session task carries the same section."""
    prompt, _stable, task = _triad_prompt(tmp_path, monkeypatch, "root")
    assert not prompt and _section("root") in task and task.index("GOAL_SENTINEL") < task.index(OWNER)
    _prompt, _stable, conscious = _triad_prompt(tmp_path, monkeypatch, "conscious")
    assert ABSENT_CONSCIOUS in conscious and OWNER not in conscious


# --- scope --------------------------------------------------------------------------------------------------

def _scope_brief(tmp_path, kind: str) -> tuple[str, str]:
    """One real scope row's brief (``prepare_scope_review``) on a staged repository."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_admission import prepare_scope_review
    from tests.test_review_session_scope_wiring import BRIEF_TASK_ID, _staged_subject

    root = tmp_path / kind
    root.mkdir()
    repo = _staged_subject(root)
    drive = root / "data"
    drive.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id=BRIEF_TASK_ID)
    for name, value in _attrs(kind).items():
        setattr(ctx, name, value)
    prepared, final = prepare_scope_review(ctx, "fix: login timeout", goal="GOAL_SENTINEL", scope_model="fixture/model",
                                           slot_id="scope_slot_1", route=ReviewRouteKind.API_CHAT)
    assert final is None
    return prepared["session_task"], ow.owner_words_text(ctx)


def test_the_scope_brief_carries_the_words_after_its_intent(tmp_path):
    brief, words = _scope_brief(tmp_path, "root")
    assert words == _section("root", task_id="scope-brief-task")
    assert words in brief and brief.index("GOAL_SENTINEL") < brief.index(words) < brief.index("## Staged diff")
    conscious, _words = _scope_brief(tmp_path, "conscious")
    assert ABSENT_CONSCIOUS in conscious and OWNER not in conscious


def test_the_scope_prompt_keeps_its_stable_prefix():
    from ouroboros.tools.review_synthesis import build_scope_review_prompt

    def prompt(owner_words: str) -> tuple[str, int]:
        goal = build_goal_section("GOAL_SENTINEL", "", "fix: retry", owner_words)
        return build_scope_review_prompt(
            "touched", scope_checklist="checklist", canonical_docs="docs", intent_context=f"scope\n\n{goal}",
            history_block="", diff_text="diff", repo_pack_placeholder="pack", critical_calibration="calibration",
            task_evidence_section="")

    (bare, bare_stable), (worded, worded_stable) = prompt(""), prompt(_section("root"))
    assert OWNER not in bare and OWNER in worded and worded.index(OWNER) > worded_stable
    assert bare_stable == worded_stable and bare[:bare_stable] == worded[:worded_stable]


# --- advisory -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("surface", ["repo", "skill"])
def test_both_advisory_goal_sections_carry_the_words(tmp_path, surface):
    from ouroboros.tools import claude_advisory_review as advisory

    repo = tmp_path / "repo"
    repo.mkdir()

    def prompt(**extra) -> str:
        return advisory._build_advisory_prompt(
            repo, "fix: login timeout", goal="GOAL_SENTINEL", scope="PAYLOAD", resolved_paths=[],
            prompt_context={"diff": "(not included)", "changed_files": "(not included)",
                            "review_surface": surface, **extra})

    words = _section("root")
    worded, bare = prompt(owner_words=words), prompt()
    assert words in worded and worded.index("GOAL_SENTINEL") < worded.index(words)
    assert OWNER not in bare and worded.replace(f"\n\n\n{words}", "", 1) == bare


def test_the_advisory_run_hands_the_runs_words_to_the_prompt(tmp_path, monkeypatch):
    from tests.test_advisory_observability import _fake_native_result, _get_advisory_module

    advisory = _get_advisory_module()
    seen = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(advisory, "_run_advisory_native", lambda prompt, repo_dir, ctx_, slot, model, **_: (
        _fake_native_result(success=True, result_text="(no output)"), model))
    monkeypatch.setattr(advisory, "_get_staged_diff", lambda *a, **kw: "diff")
    monkeypatch.setattr(advisory, "_get_changed_file_list", lambda *a, **kw: "M file.py")
    monkeypatch.setattr(advisory, "_build_advisory_prompt",
                        lambda *a, **kw: seen.append(kw["prompt_context"]["owner_words"]) or "prompt")
    for kind in ("root", "conscious"):
        ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="root", pending_events=[],
                              emit_progress_fn=lambda *_: None, **_attrs(kind))
        advisory._run_claude_advisory(tmp_path, "msg", ctx)
    assert seen == [_section("root"), ABSENT_CONSCIOUS]
    assert OWNER in seen[0] and "not the owner's" not in seen[0]


def test_reviewers_get_no_memory_or_story(tmp_path, monkeypatch):
    """The reviewer is independent (subject + contract); the words are the only addition."""
    from tests._memory_view_context import blocks, world

    review_root, mind_root = tmp_path / "review", tmp_path / "mind"
    review_root.mkdir()
    mind_root.mkdir()
    headings = ("## Identity", "## My story", "## This room", "## Working sources", "## Dialogue History")
    for delivery in (None, ""):  # the retrieving rows' session task (the default), then the packet seat
        if delivery is not None:
            monkeypatch.setattr("ouroboros.reviewer_slot_config.DEFAULT_TRIAD_DELIVERY", delivery)
        prompt, _stable, task = _triad_prompt(review_root, monkeypatch, "root")
        assert OWNER in prompt + task
        for heading in headings:
            assert heading not in prompt and heading not in task
    # The headings are the memory's real ones: the integrating mind's own request carries the first three.
    env, memory, _rooms = world(mind_root)
    a, b, c, _cap = blocks(env, memory, {"id": "turn0001", "chat_id": 1})
    assert all(heading in a + b + c for heading in headings[:3])


# --- plan review --------------------------------------------------------------------------------------------

LATER = "Add the charts too"
_PACKET = dict(objective="Deliver the thing", goal="Ship the deck", plan_prose="Outline first", spec=DECK_SPEC,
               prior_cycles=[], dispositions=[], spec_delta=None, root_exploration_log="explored")


def _plan_ctx(harness, kind="root"):
    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    attrs = _attrs(kind)
    ctx.task_metadata = {**ctx.task_metadata, **attrs["task_metadata"]}
    ctx._owner_directives = [dict(row) for row in attrs["_owner_directives"]]
    return ctx


def test_the_plan_packet_puts_the_words_after_the_objective_and_never_trims_them(harness):
    from ouroboros.tools.plan_dialogue import attach_own_dialogue, fit_dialogue_view
    from ouroboros.tools.plan_packet import build_plan_review_user_content, plan_user_stable_len

    ctx = _plan_ctx(harness)
    for text in ("ROW_ONE", "ROW_TWO", "ROW_THREE"):
        append_jsonl(harness.drive / "logs" / "chat.jsonl", {"direction": "in", "chat_id": 1, "text": text})
    manifest = attach_own_dialogue(ctx, harness.drive, {}, "a" * 64, persist=True)
    words = ow.owner_words_text(ctx, audience="plan")
    assert manifest["owner_words"] == words and OWNER in words and "not the owner's" not in words
    assert words.startswith("## Words of my human that caused this work (verbatim, host-attested)\n"
                            "They state what was asked; the objective above is the author's account of this task.")
    packet = build_plan_review_user_content(manifest=manifest, **_PACKET)
    order = ["## TASK OBJECTIVE", words, "## SPEC", "## OWN ROOM DIALOGUE", "## ROOT EXPLORATION LOG"]
    assert [packet.index(part) for part in order] == sorted(packet.index(part) for part in order)
    assert packet.index(words) < plan_user_stable_len(packet)  # the cache-stable half
    bare = build_plan_review_user_content(manifest={k: v for k, v in manifest.items() if k != "owner_words"}, **_PACKET)
    assert OWNER not in bare and bare == packet.replace(f"{words}\n\n", "", 1)
    # The tightest fit trims the conversation to nothing and leaves the words whole.
    fitted, facts = fit_dialogue_view(packet, manifest["own_dialogue"], 1)
    assert facts["conversation_rows"] == 3 and facts["conversation_inline_rows"] == 0 and "ROW_ONE" not in fitted
    assert words in fitted and fitted.index(words) < fitted.index("## SPEC")


def test_a_letter_between_waves_keeps_the_recorded_words_until_the_author_asks_anew(harness, monkeypatch):
    from ouroboros.tools import plan_review_artifacts
    from ouroboros.tools.plan_dialogue import attach_own_dialogue
    from ouroboros.tools.plan_evidence import evidence_manifest_hash

    substrate = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    ctx = _plan_ctx(harness)
    _call(ctx)
    first, words = _state(harness)["waves"][-1], ow.owner_words_text(ctx, audience="plan")
    assert words in _user_text(substrate.calls[0]["request"].messages[1]["content"])
    ctx._owner_directives.append({"source": "owner_mailbox", "content": LATER})
    _call(ctx)  # the same author request after the owner's letter
    replay = _state(harness)["waves"][-1]
    assert replay["request_fingerprint"] == first["request_fingerprint"]
    assert replay["evidence_manifest_hash"] == first["evidence_manifest_hash"]
    assert _state(harness)["cycles_paid"] == 1 and len(substrate.calls) == 1
    recorded = attach_own_dialogue(ctx, harness.drive, {}, first["author_request_fingerprint"])
    assert recorded["owner_words"] == words and LATER not in words
    assert LATER in ow.owner_words_text(ctx, audience="plan")  # the run itself did hear the letter
    # A wave recorded before the words existed replays without a section, with its hash.
    real = plan_review_artifacts.authority_wave

    def recorded_without_words(*args, **kwargs):
        exact = real(*args, **kwargs)
        full = {k: v for k, v in (exact.get("evidence_manifest_full") or {}).items() if k != "owner_words"}
        return {**exact, "evidence_manifest_full": full}

    monkeypatch.setattr(plan_review_artifacts, "authority_wave", recorded_without_words)
    old = attach_own_dialogue(ctx, harness.drive, {}, first["author_request_fingerprint"])
    assert "owner_words" not in old and evidence_manifest_hash(old) == evidence_manifest_hash(recorded)
    monkeypatch.setattr(plan_review_artifacts, "authority_wave", real)
    _call(ctx, plan="Revise the outline to add the charts.")  # a new author request reads the words now
    revised = _user_text(substrate.calls[1]["request"].messages[1]["content"])
    assert ow.owner_words_text(ctx, audience="plan") in revised and LATER in revised
    assert _state(harness)["cycles_paid"] == 2


def test_a_consciousness_plan_names_the_absence(harness):
    substrate = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    _call(_plan_ctx(harness, "conscious"))
    packet = _user_text(substrate.calls[0]["request"].messages[1]["content"])
    assert packet.index("## TASK OBJECTIVE") < packet.index(ABSENT_CONSCIOUS) < packet.index("## SPEC")
    assert "## Words of my human" not in packet
