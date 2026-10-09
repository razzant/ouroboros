"""Author-facing retry coaching and governance text are not review-contract identity.

The shared retry note (``review_prompt_text.REVIEW_REPAIR_JUDGMENT`` and
``build_self_verification_template``) coaches the author; governance-document
CONTENTS are reviewer input deliberately kept outside the contract fingerprint
(docs/development 05). Rewording either must not reprice a recorded review: the
REAL commit gate and skill review keep refusing/replaying the recorded verdict
with its findings, while a new rebuttal or a real reviewer-contract change keeps
its paid path.
"""

from __future__ import annotations

import importlib
import types

from tests.test_review_cycles_gates import (
    _gate_ctx,
    _live_skill_contract_fp,
    _seed_verdict_block,
    _wire_review_skill,
    _write_history,
)

# Every module binding of the author-facing retry coaching and of the governance
# document loaders.
_AUTHOR_COACHING_AND_GOVERNANCE_BINDINGS = (
    ("ouroboros.tools.review_prompt_text",
     ("REVIEW_REPAIR_JUDGMENT", "build_self_verification_template")),
    ("ouroboros.tools.review_helpers",
     ("REVIEW_REPAIR_JUDGMENT", "build_self_verification_template",
      "load_governance_doc", "load_checklist_section")),
    ("ouroboros.tools.review",
     ("build_self_verification_template", "load_governance_doc",
      "_load_checklist_section_precise")),
    ("ouroboros.skill_review", ("build_self_verification_template",)),
    ("ouroboros.skill_review_output", ("build_self_verification_template",)),
    ("ouroboros.skill_review_prompt", ("load_checklist_section",)),
)


def _reword_author_coaching_and_governance(monkeypatch):
    reworded = {
        "REVIEW_REPAIR_JUDGMENT": "Reworded author coaching.",
        "build_self_verification_template": lambda *a, **k: "\n\nReworded retry note.",
        "load_governance_doc": lambda *a, **k: "Reworded governance document.",
        "load_checklist_section": lambda *a, **k: "Reworded checklist section.",
        "_load_checklist_section_precise": lambda *a, **k: "Reworded checklist section.",
    }
    # Import every binder BEFORE patching any: a module first imported mid-patch
    # would bind (and monkeypatch would later "restore") the reworded value.
    modules = [(importlib.import_module(module_name), names)
               for module_name, names in _AUTHOR_COACHING_AND_GOVERNANCE_BINDINGS]
    for module, names in modules:
        for name in names:
            monkeypatch.setattr(module, name, reworded[name])  # raises if the binding is gone


def test_author_retry_coaching_and_governance_text_do_not_reprice_recorded_reviews(
    tmp_path, monkeypatch,
):
    """A verdict recorded before the rewording keeps its free refusal/replay with
    the recorded findings, while on both gates a NEW rebuttal still buys one paid
    rerun and a real reviewer prompt-contract change still lapses the verdict."""
    import ouroboros.skill_review_status as skill_status
    import ouroboros.tools.git as git_mod
    import ouroboros.tools.review_helpers as review_helpers
    from ouroboros import skill_review

    monkeypatch.delenv("OUROBOROS_REVIEW_MAX_CYCLES", raising=False)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setattr(git_mod, "run_cmd", lambda *a, **k: "")
    monkeypatch.setattr(git_mod, "_authorized_managed_update_resolver", lambda ctx: False)

    # Recorded before the rewording, under the live contracts.
    commit_contract = git_mod.commit_review_contract_fingerprint()
    skill_contract = _live_skill_contract_fp()
    assert commit_contract and skill_contract
    _seed_verdict_block(tmp_path, "fp-1", commit_contract)
    _write_history(tmp_path, "demo", [
        {"ts": "t1", "status": "warnings", "content_hash": "h1", "paid": True,
         "group_id": "manual:demo", "review_contract_fingerprint": skill_contract,
         "job_id": "j1"},
    ])

    _reword_author_coaching_and_governance(monkeypatch)
    assert git_mod.commit_review_contract_fingerprint() == commit_contract
    assert _live_skill_contract_fp() == skill_contract

    # Commit: the identical diff is still refused free, quoting the recorded finding;
    ctx = _gate_ctx(tmp_path)
    refused = git_mod._free_cycle_gate(
        ctx, "msg", 0.0, pre_fingerprint={"fingerprint": "fp-1"}, review_rebuttal="",
    )
    assert refused is not None and refused["block_reason"] == "identical_diff_refused"
    assert "bug_y" in refused["message"]
    # a genuinely new rebuttal still buys the one paid rerun;
    assert git_mod._free_cycle_gate(
        ctx, "msg", 0.0, pre_fingerprint={"fingerprint": "fp-1"},
        review_rebuttal="The finding cites a helper this diff does not call.",
    ) is None
    # and a real reviewer prompt-contract change lapses the recorded verdict.
    with monkeypatch.context() as contract_change:
        contract_change.setattr(review_helpers, "REVIEW_PREAMBLE", "A changed reviewer preamble.")
        assert git_mod.commit_review_contract_fingerprint() != commit_contract
        assert git_mod._free_cycle_gate(
            ctx, "msg", 0.0, pre_fingerprint={"fingerprint": "fp-1"}, review_rebuttal="",
        ) is None

    # Skill: the identical snapshot replays the recorded verdict at $0 without dispatch.
    def _must_not_dispatch(*a, **k):
        raise AssertionError("run_skill_review_passes must not be called")

    findings = [{"item": "bug_hunting", "verdict": "FAIL", "severity": "advisory",
                 "reason": "meh"}]
    state = types.SimpleNamespace(content_hash="h1", status="warnings",
                                  findings=findings, reviewer_models=["m1", "m2"])
    skill_ctx = types.SimpleNamespace(task_id="", task_metadata={}, event_queue=None)
    _wire_review_skill(monkeypatch, tmp_path, content_hash="h1",
                       review_state=state, passes=_must_not_dispatch)
    replayed = skill_review.review_skill(skill_ctx, "demo", persist=False)
    assert replayed.status == "warnings" and replayed.paid is False
    assert replayed.replayed_from_ts == "t1" and replayed.findings == findings

    # A new rebuttal, or a real prompt-contract change, reaches the panel instead.
    dispatched = []

    def _dispatch(*a, **k):
        dispatched.append(True)
        return ("prompt", {}, "", "provider exploded")

    _wire_review_skill(monkeypatch, tmp_path, content_hash="h1",
                       review_state=state, passes=_dispatch)
    rebutted = skill_review.review_skill(
        skill_ctx, "demo", persist=False,
        review_rebuttal="The finding cites a helper this skill does not call.",
    )
    assert dispatched == [True] and not rebutted.replayed_from_ts
    with monkeypatch.context() as contract_change:
        contract_change.setattr(skill_status, "WARNINGS_CONVERGENCE_ROUNDS",
                                int(skill_status.WARNINGS_CONVERGENCE_ROUNDS) + 1)
        assert _live_skill_contract_fp() != skill_contract
        lapsed = skill_review.review_skill(skill_ctx, "demo", persist=False)
    assert dispatched == [True, True] and not lapsed.replayed_from_ts
