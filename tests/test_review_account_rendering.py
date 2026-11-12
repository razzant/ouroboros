"""Reviewer packets render selected authored accounts as text, not capsule bookkeeping.

``project_review_history`` keeps the exact applied actor capsules in
``actor_account`` (Main's own context and the retained selection use them). The
reviewer-visible ``history.authored_view`` carries every selected visible text in
order, with authorship, source and rule; capsule metadata must not become prose
in plan or change/commit briefs. Literal words that merely look like metadata
keys, in an account or in user content, stay intact.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from ouroboros import review_history_view as view
from ouroboros.artifacts import read_actor_source_bytes
from tests.test_review_decision_notes import first_plan, notes_for
from tests.test_review_view_integration import _apply, _capture, harness as _harness, staged_body as _staged_body

harness, staged_body = _harness, _staged_body
LITERAL = 'I keep the words "_context_capsule" and "visible_sha256": literal text of my account.'
BOOKKEEPING = ("summary_contract_digest", "summary_contract_version", "measurement_basis", "source_unit_sha256",
               "source_length_chars", "route_fp")


def _texts(account):
    return [capsule["content"][0]["text"] for capsule in view.selected_actor_capsules(account)]


def _assert_reviewer_text(text, texts):
    for key in BOOKKEEPING:
        assert key not in text, key
    positions = [text.find(json.dumps(item, ensure_ascii=False)[1:-1].split("\\n")[-1]) for item in texts]
    assert all(at >= 0 for at in positions) and positions == sorted(positions), positions


def test_selected_history_shows_selected_text_while_canonical_capsules_stay_exact(harness):
    from ouroboros.context import _task_authority_projection
    from ouroboros.tools.plan_packet import _json_block
    from types import SimpleNamespace

    ctx = first_plan(harness)
    _, receipt, _ = _apply(ctx, _capture(ctx), note=LITERAL, review_notes=notes_for(ctx))
    assert receipt["status"] == "applied", receipt
    source_ref = receipt["selected_review_history_view"]["source_ref"]
    retained = read_actor_source_bytes(ctx.drive_root, ctx.task_id, source_ref)
    history, operative = view.current_plan_history(ctx)
    canonical_history = deepcopy(history)
    selected = view.selected_review_history(history, drive_root=ctx.drive_root, task_id=ctx.task_id,
                                            operative_subject=operative)
    assert history == canonical_history
    texts = _texts(selected["actor_account"])
    assert len(texts) == 1 and LITERAL in texts[0]
    assert selected["history"]["authored_view"] == {
        "authorship": "actor", "texts": texts, "source_ref": source_ref, "rule": selected["actor_account"]["rule"]}
    for capsule in view.selected_actor_capsules(selected["actor_account"]):  # Main/selection keep exact capsules
        meta = view._actor_capsule(capsule)
        assert meta["visible_sha256"] == hashlib.sha256(capsule["content"][0]["text"].encode()).hexdigest()
        assert meta["summary_contract_digest"] and meta["authorship"] == "actor"
    assert selected["review_notes"] and selected["selection_status"] == "applied"
    assert "authored_view" not in selected["mandatory"]
    block = _json_block(selected["history"])
    _assert_reviewer_text(block, texts)
    assert '\\"_context_capsule\\"' in block  # the account's own literal words survive
    runtime = _task_authority_projection(SimpleNamespace(drive_root=ctx.drive_root, repo_dir=ctx.repo_dir),
                                         {"id": ctx.task_id})
    _, tail = view.capture_review_history_messages(runtime, task_id=ctx.task_id, drive_root=ctx.drive_root)
    assert view.selected_actor_capsules(selected["actor_account"])[0] in tail
    assert read_actor_source_bytes(ctx.drive_root, ctx.task_id, source_ref) == retained


def test_multiple_local_accounts_render_every_text_in_order(harness, monkeypatch):
    from tests.test_review_local_accounts import FIRST, SECOND, two_folds
    from tests.test_review_view_integration import _three

    ctx, _ = _three(harness)
    _, _, _, _, receipt, _ = two_folds(ctx)
    history, operative = view.current_plan_history(ctx)
    selected = view.selected_review_history(history, drive_root=ctx.drive_root, task_id=ctx.task_id,
                                            operative_subject=operative)
    texts = selected["history"]["authored_view"]["texts"]
    assert texts == _texts(selected["actor_account"]) and len(texts) == 2
    assert FIRST in texts[0] and SECOND in texts[1]
    assert "capsules" in selected["actor_account"]  # canonical collection is unchanged


@pytest.mark.parametrize("surface", ["commit", "change"])
def test_commit_and_change_briefs_render_the_account_without_bookkeeping(staged_body, tmp_path, monkeypatch, surface):
    from pathlib import Path
    from ouroboros import capability_evidence, reviewer_window, review_substrate, review_ledger
    from ouroboros.review_history import render_review_history
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.registry import ToolContext
    from tests import _contributor_packet_shared as shared
    from tests.test_review_change_end_to_end import _brief_text
    from tests.test_review_cold_history import _dispatch, _run

    monkeypatch.setattr(capability_evidence, "probe", lambda *a, **k: None)
    monkeypatch.setattr(reviewer_window, "reviewer_context_window", lambda *a, **k: 1_000_000)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "6")
    root, drive = Path(staged_body["repo"]), tmp_path / "account-review"
    ctx = ToolContext(repo_dir=root, drive_root=drive, task_id="account-rendering")
    monkeypatch.setattr(review_substrate, "run_review_request", _dispatch([], "Exact critic account."))
    record = _run(ctx, surface, "Keep the reviewed rationale.")
    review_ledger.note_author_decision(drive, record, {"disposition": "rejected", "rationale": "Bounded files stay."})
    canonical = deepcopy(review_ledger.load_record(drive, record))
    _, receipt, _ = _apply(ctx, _capture(ctx), note=LITERAL)
    assert receipt["status"] == "applied", receipt
    texts = _texts(view.load_review_history_view(receipt["selected_review_history_view"],
                                                 lambda ref: read_actor_source_bytes(drive, ctx.task_id, ref)))
    rendered = render_review_history({}, drive_root=drive, repo_root=root, task_id=ctx.task_id)
    section = rendered.split("### Author's selected account of earlier review sources", 1)[1]
    assert json.loads(section.strip().split("\n", 1)[0])["texts"] == texts
    _assert_reviewer_text(rendered, texts)
    briefs = []
    monkeypatch.setattr(review_substrate, "run_review_request", _dispatch(briefs, "New second version"))
    (root / "ouroboros/tools/review.py").write_text("RULES = 'second authored version'\n", encoding="utf-8")
    shared.git(root, "add", "-A")
    task_row = load_task_result(drive, ctx.task_id, strict=True)
    _run(ToolContext(repo_dir=root, drive_root=drive, task_id=ctx.task_id), surface, "A new argument stays full.")
    assert briefs
    for brief in briefs:
        text = _brief_text(brief)
        assert "literal text of my account" in text, text[:500]
        for key in BOOKKEEPING:
            assert key not in text, key
    assert review_ledger.load_record(drive, record) == canonical
    assert load_task_result(drive, ctx.task_id, strict=True)[view.SELECTED_VIEW_FIELD] == \
        task_row[view.SELECTED_VIEW_FIELD]
