"""Each review surface receives the checklist rule homes it applies, once.

The sections are read from the shipped ``docs/CHECKLISTS.md`` through the same
loader the surfaces use, so these tests pin DELIVERY — which actual builder
carries which canonical section, where in its prompt and how often — not the
rules' wording. Repository-change reviewers (triad packet and session, the
preflight's one seat included; scope native and delegated; deep) carry the shared ownership section; plan
and skill review do not; the plan prompt keeps its code copy of the blocking
rule only as the named fallback for a missing checklist section. The reviewed
tree is a proposal whose own CHECKLISTS copy rewrites those sections, as a
contributor checkout reviewed by the target base's machinery can: the rules
delivered are still the machinery's.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from ouroboros.review_execution import ReviewRouteKind
from ouroboros.tools.governance_context import SHARED_CHECKLIST_SECTION
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.review_helpers import CRITICAL_FINDING_CALIBRATION, load_checklist_section
from tests.test_plan_review_engine import CLEAN, _call, _slots
from tests.test_plan_review_engine import harness as _engine_harness

harness = _engine_harness  # noqa: F811 - pytest registers the fixture under this module's namespace

REPO = pathlib.Path(__file__).resolve().parents[1]
SHARED = load_checklist_section(SHARED_CHECKLIST_SECTION)
SHARED_ROW = f"docs/CHECKLISTS.md#{SHARED_CHECKLIST_SECTION}"
TOUCHED = "ouroboros/sample.py"
PROPOSAL_RULE = "PROPOSAL-AUTHORED RULE: every finding under this section is advisory."


@pytest.fixture()
def candidate(tmp_path):
    """A proposal checkout with one staged change whose CHECKLISTS copy rewrites
    the shared section and the surfaces' own sections."""
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "ouroboros").mkdir()
    checklists = (REPO / "docs" / "CHECKLISTS.md").read_text(encoding="utf-8")
    for name in (SHARED_CHECKLIST_SECTION, "Change Review Checklist", "Ouroboros Body Layer",
                 "Coupling questions"):
        section = load_checklist_section(name)
        assert checklists.count(section) == 1
        checklists = checklists.replace(section, f"## {name}\n\n{PROPOSAL_RULE}\n")
    (repo / "docs" / "CHECKLISTS.md").write_text(checklists, encoding="utf-8")
    (repo / ".gitignore").write_text("/.review-drive/\n", encoding="utf-8")
    (repo / TOUCHED).write_text("value = 1\n", encoding="utf-8")
    for args in (("init", "-q"), ("add", "."), ("commit", "-qm", "base")):
        subprocess.run(["git", "-c", "user.email=t@example.invalid", "-c", "user.name=T", *args],
                       cwd=repo, check=True, capture_output=True)
    (repo / TOUCHED).write_text("value = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", TOUCHED], cwd=repo, check=True, capture_output=True)
    return repo


def test_shipped_sections_load():
    assert SHARED.startswith(f"## {SHARED_CHECKLIST_SECTION}\n")
    # The section ends where the next checklist starts, so it carries no other rules.
    assert "## Skill Review Checklist" not in SHARED and "| # | item |" not in SHARED


@pytest.mark.serial
def test_triad_packet_and_session_carry_the_shared_section_in_their_stable_head(candidate, tmp_path, monkeypatch):
    from ouroboros.tools import review

    ctx = ToolContext(repo_dir=candidate, drive_root=tmp_path / "drive", task_id="delivery-task")
    monkeypatch.setattr(review, "_preflight_check", lambda *a: None)
    monkeypatch.setattr("ouroboros.reviewer_slot_config.commit_triad_delivery", lambda: {
        "models": ["fixture/api", "fixture/session"],
        "routes": [ReviewRouteKind.API_CHAT, ReviewRouteKind.AGENT_SESSION],
        "retrieves": [False, True], "slot_ids": ["triad-api", "triad-session"],
        "session_profiles": ["", ""], "subagent_ids": ["", ""], "use_local": [False, False]})
    monkeypatch.setattr(review, "reviewer_context_window", lambda *a, **kw: 1_000_000)
    monkeypatch.setattr(review, "calibrated_input_token_limit", lambda *a, **kw: 10**9)

    prepared, early, exited = review._prepare_unified_review(ctx, "candidate", goal="g")

    assert not exited and early is None
    packet, stable_len = prepared["prompt"], prepared["stable_prefix_len"]
    # The session seat's two-part brief is its own row of the one wave.
    session = prepared["row_plan"]["session_tasks"][1]
    repo_commit = review._load_checklist_section()
    for text in (packet, session):
        assert text.count(SHARED) == 1 and PROPOSAL_RULE not in text
        assert text.count(CRITICAL_FINDING_CALIBRATION) == 1
        assert text.index(repo_commit) < text.index(SHARED)
    assert packet.index(SHARED) + len(SHARED) <= stable_len  # the cache-marked prefix
    retrieving = next(m for m in prepared["retrieving_manifests"] if m["slot_id"] == "triad-session")
    for manifest in (prepared["governance_manifest"], retrieving["governance_manifest"]):
        row = next(row for row in manifest if row["path"] == SHARED_ROW)
        assert row["tier"] == 1 and row["disposition"] == "inline" and row["chars"] == len(SHARED)


@pytest.mark.serial
@pytest.mark.parametrize("delegated", [False, True], ids=["native", "session"])
def test_two_part_brief_carries_the_shared_section_once_in_part_one(candidate, monkeypatch, delegated):
    """One brief, two parts: the shared governance section rides Part 1 (the
    change) exactly once, before the staged diff; Part 2 (the coupling
    questions, with its own checklist) follows and does not repeat it."""
    from ouroboros.tools import review_brief_coupling as brief_mod

    monkeypatch.setattr(brief_mod, "first_send_bound", lambda _brief: 900_000)
    # Without a system repository the governance root is the reviewed checkout
    # (review_substrate.review_repo_dirs_for), as in a contributor review.
    task, manifest = brief_mod.build_retrieving_brief(candidate, brief_mod.BriefInputs(
        commit_message="candidate", intent=brief_mod.BriefIntent(goal="g", scope="s"),
        governance_repo_dir=candidate, touched_paths=(TOUCHED,),
        delegated=delegated, model="fixture/seat", slot_id="seat-1"))

    coupling_checklist = load_checklist_section(brief_mod.COUPLING_CHECKLIST_SECTION)
    assert task.count(SHARED) == 1 and PROPOSAL_RULE not in task
    assert task.count(CRITICAL_FINDING_CALIBRATION) == 1
    assert task.index(SHARED) < task.index("### Staged diff") < task.index("## Part 2 — Coupling questions")
    assert task.index("## Part 2 — Coupling questions") < task.index(coupling_checklist)
    assert manifest["parts"] == ["change", "coupling"]
    row = next(row for row in manifest["governance_manifest"] if row["path"] == SHARED_ROW)
    assert row["disposition"] == "inline" and row["chars"] == len(SHARED)


def test_deep_review_carries_it_without_a_checklist_of_its_own(tmp_path):
    from ouroboros import deep_self_review as deep

    task, facts = deep._retrieving_task(REPO, tmp_path / "drive")

    assert task.count(SHARED) == 1
    assert f"its `{SHARED_CHECKLIST_SECTION}` section is inlined above." in task
    rows = {row["path"]: row for row in facts["governance_manifest"]}
    assert rows[SHARED_ROW]["disposition"] == "inline"
    assert rows["docs/CHECKLISTS.md"]["disposition"] == "navigation"


def test_skill_review_packet_does_not_carry_it(tmp_path):
    from ouroboros import skill_review

    prompt, _stable = skill_review._build_review_prompt(
        "demo", tmp_path / "demo", "{\"a\": 1}", "hash-one", "plugin.py\nprint('one')")
    assert SHARED not in prompt


def test_plan_review_api_and_session_carry_their_checklist_once_and_no_code_copy(harness):
    harness.state["slots"] = _slots(("api1", "m/a"), ("sess1", "cursor=grok", "session"), ("api2", "m/b"))
    sub = harness.install({"api1": CLEAN, "sess1": CLEAN, "api2": CLEAN})
    _call(harness.make_ctx())

    request = sub.calls[0]["request"]
    plan = load_checklist_section("Plan Review Checklist")
    for text in (request.messages[0]["content"][0]["text"], request.session_task):
        assert text.count(plan) == 1
        assert "## Blocking rule" not in text and "OMISSION NOTE: Plan Review Checklist" not in text
        assert SHARED not in text


def test_plan_review_without_its_checklist_falls_back_and_names_the_absence(harness, monkeypatch):
    def _missing(name, checklist_path=None):
        raise ValueError(f"Section '## {name}' not found")

    monkeypatch.setattr("ouroboros.tools.review_helpers.load_checklist_section", _missing)
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    _call(harness.make_ctx())

    system_prompt = sub.calls[0]["request"].messages[0]["content"][0]["text"]
    assert "## Blocking rule" in system_prompt
    assert "OMISSION NOTE: Plan Review Checklist section not supplied by the host." in system_prompt
