"""A stopped skill-review seat continues on its record's own bytes, never today's, from the tool or the Skills API.

``continue_skill_review`` collects the exact recorded review (the generic collector, after the
continuation choice) and projects that record the way its own wave would: an unchanged payload
gains the verdict, a payload changed since the review gains nothing. A real clean skill wave makes
the record and its frozen payload view; the collection of the continued seat is stubbed here
(``test_skill_review_continuation_api`` drives the real door, job and collector end to end).
"""
from __future__ import annotations

import asyncio
import json

from ouroboros.review_continuation import ContinuationRefused
from ouroboros.review_ledger import ledger_root, load_record
from ouroboros.skill_loader import compute_content_hash, load_review_state
from ouroboros.skill_review import continue_skill_review, review_skill
from ouroboros.tools.registry import ToolContext
from tests.test_skill_review_persist_guard import _build_script_skill, _clean_wave


def _reviewed(tmp_path, monkeypatch):
    repo_dir, drive_root, skills_root = tmp_path / "repo", tmp_path / "drive", tmp_path / "skills"
    repo_dir.mkdir()
    drive_root.mkdir()
    skill_dir = _build_script_skill(skills_root)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = ToolContext(repo_dir=repo_dir, drive_root=drive_root, task_id="skill-author")
    _clean_wave(monkeypatch)
    outcome = review_skill(ctx, "alpha")
    assert outcome.record_id and outcome.status != "pending", outcome
    return ctx, skill_dir, drive_root, outcome


def _collected(monkeypatch, calls):
    from ouroboros.tools import review_collection
    from ouroboros.tools.review_change import review_result

    def collect(ctx, reference, continue_seats=None):
        calls.append((reference, continue_seats))
        return {**review_result(load_record(ledger_root(ctx), reference["record_id"]), reused=False),
                "continuation": {"continued": [{"seat_id": "s", "operation_id": "review_continue_x", "rejoined": False}]}}
    monkeypatch.setattr(review_collection, "collect_change", collect)


def test_the_recorded_bytes_are_projected_and_a_wrong_reference_gains_nothing(tmp_path, monkeypatch):
    ctx, skill_dir, drive_root, outcome = _reviewed(tmp_path, monkeypatch)
    calls = []
    _collected(monkeypatch, calls)
    seats = [{"seat_id": "s", "operation_id": "op-s"}]
    wrong = continue_skill_review(ctx, "alpha", reference={"record_id": outcome.record_id, "content_hash": "other"},
                                  continue_seats=seats)
    assert wrong.status == "pending" and wrong.dispatch_refusal_code == "TOOL_ARG_ERROR" and not calls
    continued = continue_skill_review(ctx, "alpha", reference={"record_id": outcome.record_id,
                                                               "content_hash": outcome.content_hash}, continue_seats=seats)
    assert calls == [({"record_id": outcome.record_id, "task_id": "skill-author", "surface": "skill"}, seats)]
    assert continued.status != "pending" and continued.record_id == outcome.record_id
    assert load_review_state(drive_root, "alpha").record_id == outcome.record_id


def test_a_payload_changed_since_its_review_gains_no_projection(tmp_path, monkeypatch):
    ctx, skill_dir, drive_root, outcome = _reviewed(tmp_path, monkeypatch)
    before = load_review_state(drive_root, "alpha")
    (skill_dir / "scripts" / "run.py").write_text("print('changed after review')\n", encoding="utf-8")
    assert compute_content_hash(skill_dir) != outcome.content_hash
    _collected(monkeypatch, [])
    continued = continue_skill_review(ctx, "alpha", reference={"record_id": outcome.record_id,
                                                               "content_hash": outcome.content_hash},
                                      continue_seats=[{"seat_id": "s", "operation_id": "op-s"}])
    assert continued.status == "pending" and "payload changed" in continued.error
    after = load_review_state(drive_root, "alpha")
    assert (after.content_hash, after.record_id, after.status) == (before.content_hash, before.record_id, before.status)


def test_a_refused_choice_names_the_current_offer(tmp_path, monkeypatch):
    ctx, _skill_dir, _drive_root, outcome = _reviewed(tmp_path, monkeypatch)
    from ouroboros.tools import review_collection

    def refuse(ctx, reference, continue_seats=None):
        raise ContinuationRefused("not_continuable", "answered", [{"seat_id": "s", "operation_id": "op-now"}])
    monkeypatch.setattr(review_collection, "collect_change", refuse)
    refused = continue_skill_review(ctx, "alpha", reference={"record_id": outcome.record_id,
                                                             "content_hash": outcome.content_hash},
                                    continue_seats=[{"seat_id": "s", "operation_id": "op-old"}])
    assert refused.status == "pending" and "not_continuable" in refused.error and "op-now" in refused.error


def test_the_tool_and_the_skills_api_route_an_exact_choice_into_the_exact_path(tmp_path, monkeypatch):
    from ouroboros import skill_review
    from ouroboros.gateway import extensions
    from ouroboros import skill_review_runner
    from ouroboros.tools.skill_exec import _REVIEW_SCHEMA

    seen = []
    monkeypatch.setattr(skill_review, "continue_skill_review", lambda ctx, name, **kw: seen.append((ctx.task_id, name, kw)) or
                        skill_review._pending(name, "collected"))

    async def lifecycle(ctx, skill_name, *, source, review_impl, **_kw):
        review_impl(ctx, skill_name)
        return {"status": "ok"}
    monkeypatch.setattr(skill_review_runner, "run_skill_review_lifecycle", lifecycle)

    class _Request:
        path_params = {"skill": "alpha"}

        async def body(self):
            return json.dumps({"review_reference": {"record_id": "rl-1", "content_hash": "h"},
                               "continue_seats": [{"seat_id": "s", "operation_id": "o"}]}).encode()
    monkeypatch.setattr(extensions, "_request_drive_root", lambda _r: tmp_path)
    monkeypatch.setattr(extensions, "_request_repo_dir", lambda _r: tmp_path)
    response = asyncio.run(extensions.api_skill_review(_Request()))
    assert json.loads(response.body) == {"status": "ok"}
    assert seen == [("api_skill_review", "alpha", {"reference": {"record_id": "rl-1", "content_hash": "h"},
                                                   "continue_seats": [{"seat_id": "s", "operation_id": "o"}]})]
    assert "continue_seats" in _REVIEW_SCHEMA["parameters"]["properties"]
