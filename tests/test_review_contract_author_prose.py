"""Author-facing retry coaching and governance text are not review-contract identity.

The shared retry note (``review_prompt_text.REVIEW_REPAIR_JUDGMENT`` and
``build_self_verification_template``) coaches the author; governance-document
CONTENTS are reviewer input deliberately kept outside the contract fingerprint
(docs/development 05). Rewording either must not reprice a recorded review: the
REAL commit gate keeps refusing/replaying the recorded verdict with its findings,
while a new rebuttal or a real reviewer-contract change keeps its paid path. A
skill wave shares this contract fingerprint and the ledger's reuse key (its rules
identity is the checklist section's fingerprint), so the same holds for it.
"""

from __future__ import annotations

import importlib

from tests.test_review_cycles_gates import _gate_ctx, _seed_verdict_block

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
    ("ouroboros.skill_review_output", ("build_self_verification_template",)),
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
    the recorded findings, while a NEW rebuttal still buys one paid rerun and a
    real reviewer prompt-contract change still lapses the verdict."""
    import ouroboros.tools.git as git_mod
    import ouroboros.tools.review_helpers as review_helpers

    monkeypatch.delenv("OUROBOROS_REVIEW_MAX_CYCLES", raising=False)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setattr(git_mod, "run_cmd", lambda *a, **k: "")
    monkeypatch.setattr(git_mod, "_authorized_managed_update_resolver", lambda ctx: False)

    # Recorded before the rewording, under the live contract.
    commit_contract = git_mod.commit_review_contract_fingerprint()
    assert commit_contract
    _seed_verdict_block(tmp_path, "fp-1", commit_contract)

    _reword_author_coaching_and_governance(monkeypatch)
    assert git_mod.commit_review_contract_fingerprint() == commit_contract

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
