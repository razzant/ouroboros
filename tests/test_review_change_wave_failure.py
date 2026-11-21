"""review_change keeps the cause of a wave its own machinery stopped (#2516).

A wave can end without any reviewer's verdict because the host stopped it: assembly refused, review
custody could not be reserved, the dispatch crashed or the service returned an error. The producer's own
words then say which step failed. Each case below runs the REAL operation (``run_review_change`` over the
installed body's staged index, the real ``run_parallel_review``, record builder, projection, task result,
source door and card renderer); only ONE producer seam fails, and the paid substrate is counted, never
called for a refusal. A reviewer's own rejection is the control: it gains nothing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import ouroboros.review_substrate as substrate
from ouroboros import review_ledger as rl
from ouroboros.tools import parallel_review
from ouroboros.tools import review as review_mod
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.review_change import run_review_change, wave_failure_note
from tests import _contributor_packet_shared as shared
from tests.review_pool_rosters import set_review_pool

REPO = Path(__file__).resolve().parents[1]
TASK = "wave-failure-task"
GOAL, SCOPE = "Make the installed body's helper return the proposal's constant.", "ouroboros/helper.py only."
STAGE = "WAVE-STAGE-7f3c"  # a recognizable phrase inside the producer's own words
SECRET = "sk-ant-api03-" + "Zq9xW2" * 7  # secret-shaped: every safe surface masks it
TAIL = "TAIL-OF-THE-WHOLE-NOTE"
RENDER = """
import fs from 'node:fs';
import { reviewGroupsFromTaskDetail, renderReviewsSection } from './web/modules/review_presentation.js';
const detail = JSON.parse(fs.readFileSync(0, 'utf8'));
const groups = reviewGroupsFromTaskDetail(detail, detail.task_id);
const ids = groups.flatMap(group => (group.attempts || []).map(attempt => `${group.id}:${attempt.id}`));
console.log(renderReviewsSection(groups, { sectionExpanded: true, expandedGroups: new Set(ids) }));
"""


@pytest.fixture
def body(tmp_path, monkeypatch):
    """The installed body with the proposal staged; three pool seats; the paid seam answers PASS."""
    fixture = shared.init_installed_body(tmp_path)
    repo = Path(fixture["repo"])
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    shared.git(repo, "add", ".gitignore")
    shared.git(repo, "commit", "-q", "-m", "ignore caches")
    shared.git(repo, "cherry-pick", "--no-commit", fixture["head_sha"])
    set_review_pool(monkeypatch, shared.golden_pool())
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    sends: list = []
    monkeypatch.setattr(substrate, "run_review_request", shared.golden_substrate(sends))
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "drive", task_id=TASK)
    return SimpleNamespace(repo=repo, ctx=ctx, drive=tmp_path / "drive", sends=sends, tmp=tmp_path)


def _review(body) -> dict:
    return run_review_change(body.ctx, root="system_repo", surface="change", goal=GOAL, scope=SCOPE, subject="index")


def _enforce(monkeypatch, mode: str) -> None:
    if mode == "cyber_pro":
        monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")


def _text(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _notes(record) -> list:
    return list((record.get("verdict") or {}).get("wave_failures") or [])


def _render(task_id: str, projection: dict) -> str:
    node = shutil.which("node")
    assert node, "node is required for the card renderer"
    result = subprocess.run([node, "--input-type=module", "-e", RENDER], cwd=REPO, capture_output=True, check=True,
                            input=json.dumps({"task_id": task_id, "review_projection": projection}),
                            text=True, encoding="utf-8")
    return result.stdout


def _rows_html(html: str) -> str:
    """What the opened row shows above Record facts."""
    return html[:html.index('class="review-facts"')]


def _stored_projection(body) -> tuple:
    """The parent's reading: the task's terminal write and its task_done projection."""
    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.task_results import load_task_result
    from supervisor.events_task_done import _task_done_review_projection

    env, task = SimpleNamespace(drive_root=body.drive, repo_dir=body.repo), {"id": TASK, "type": "task"}
    _store_task_result(env, task, "done", {"rounds": 1, "cost": 0}, {"tool_calls": []})
    stored = load_task_result(body.drive, TASK) or {}
    return stored, _task_done_review_projection(stored, {})


def _refuse_custody(monkeypatch, words: str) -> None:
    """The unpaid custody reservation fails before any worker starts (the producer's typed refusal)."""
    from ouroboros import review_dispatch

    def refuse(ctx):
        raise OSError(words)
    monkeypatch.setattr(review_dispatch, "reserve_review_roster", refuse)


def test_a_custody_refusal_before_any_send_reaches_record_tool_parent_door_and_card(body, monkeypatch):
    words = f"state write refused at {STAGE}: " + "the attempt row could not be saved; " * 30 + SECRET + " " + TAIL
    _refuse_custody(monkeypatch, words)
    result = _review(body)
    assert body.sends == [], "a refusal before custody sends nothing"
    record = rl.load_record(body.drive, result["record_id"])
    # The loss this repair closes: the record and the tool answer named only `gate_block:infra_failure`.
    assert STAGE in _text(result), "the tool answer names the step that failed"
    notes = _notes(record)
    assert len(notes) == 1 and notes[0]["block_reason"] == "infra_failure"
    cause = notes[0]["reported_cause"]
    assert "durable review custody could not be reserved before dispatch" in cause and STAGE in cause
    assert len(cause) <= 512 and "OMISSION NOTE" in cause, "the display bound is disclosed, never silent"
    # The verdict, quorum, seats and cost stay what the wave actually was.
    verdict = record["verdict"]
    assert verdict["aggregate"] == "NOT_DISPATCHED" and "gate_block:infra_failure" in verdict["degraded_reasons"]
    assert record["dispatch_refusal"] is None
    assert {row["status"] for row in record["rows"]} == {"not_dispatched"}
    assert record["cost"] == {"usd": 0.0, "unknown": True}
    assert any(ref["role"] == "prompt" for row in record["rows"] for ref in row["source_refs"]), "prepared briefs kept"
    # The complete producer words are retained exactly, as the panel's failure note (no reviewer seat invented).
    ref = notes[0]["source_ref"]
    assert (ref["role"], ref["seat_id"], ref["status"]) == ("failure", "panel", "retained")
    exact = rl.read_selected_source(body.drive, TASK, record["record_id"], "panel", role="failure", part="",
                                    sha256=ref["ref"]["sha256"]).decode("utf-8")
    assert STAGE in exact and TAIL in exact and SECRET in exact and "No reviewer was started" in exact
    assert {row["seat_id"] for row in record["rows"]} == {"t1", "t2", "s1"}
    assert result["wave_failures"][0]["reported_cause"] == cause
    # Nothing secret reaches the hot index or the tool answer.
    index = rl.index_path(body.drive).read_text(encoding="utf-8")
    assert SECRET[:16] not in index and SECRET[:16] not in _text(result)

    # The parent's reading: the stored task result, its task_done projection, the card and the door.
    stored, done = _stored_projection(body)
    for projection in (stored["review_projection"], done):
        panel = next(p for p in projection["panels"] if p.get("record_id") == record["record_id"])
        assert panel["wave_failures"][0]["reported_cause"] == cause
        assert panel["reason"] == verdict["reason"], "the summary keeps the verdict's own reason"
        source = panel["wave_failures"][0]["source"]
        assert "path" not in _text(source) and source["sha256"] == ref["ref"]["sha256"]
    html = _render(TASK, stored["review_projection"])
    rows = _rows_html(html)
    assert "rejected" not in html, "an infrastructure stop never reads as a reviewer's rejection"
    assert "Ouroboros could not complete this review; its note says which step failed." in rows
    assert "No cause was recorded." not in rows and rows.count("not sent") >= 3
    assert "REVIEW_BLOCKED" not in rows and STAGE not in rows, "tokens and raw words stay in Record facts"
    facts = html[html.index('class="review-facts"'):]
    assert "<dt>Wave failure</dt>" in facts and STAGE in facts and "infra_failure" in facts
    assert "<dt>Saved wave failure note</dt>" in facts and ref["ref"]["sha256"] in facts
    url = next(part.split('"', 1)[0] for part in rows.split('data-review-read-url="')[1:] if "/panel?" in part)
    assert 'aria-label="Read Ouroboros failure note"' in rows
    from tests.test_task_review_sources import client_for

    with client_for(body.drive) as client:
        response = client.get(url.replace("&amp;", "&"))
        foreign = client.get(url.replace("&amp;", "&").replace(f"/api/tasks/{TASK}/", "/api/tasks/another-task/"))
    assert response.status_code == 200 and foreign.status_code == 404
    assert STAGE in response.text and TAIL in response.text and SECRET[:16] not in response.text
    assert response.headers["x-review-source-sha256"] == ref["ref"]["sha256"]


@pytest.mark.parametrize("mode", ["pro", "cyber_pro"])
def test_an_assembly_refusal_before_any_seat_keeps_its_words_in_every_access(body, monkeypatch, mode):
    _enforce(monkeypatch, mode)

    def missing(layer: str = "body") -> str:
        raise FileNotFoundError(f"checklist unreadable at {STAGE}")
    monkeypatch.setattr(review_mod, "_load_checklist_section", missing)
    result = _review(body)
    assert body.sends == []
    record = rl.load_record(body.drive, result["record_id"])
    assert record["rows"] == [] and record["verdict"]["aggregate"] == "NOT_DISPATCHED"
    notes = _notes(record)
    assert len(notes) == 1 and notes[0]["block_reason"] == "infra_failure", notes
    cause = notes[0]["reported_cause"]
    assert "Cannot load review checklist" in cause and STAGE in cause
    expected = {"pro": "Review checklist failed to load", "cyber_pro": "Cyber Pro"}[mode]
    assert expected in cause and "REVIEW_BLOCKED" in cause, cause  # the original signal follows the prefix
    assert result["wave_failures"] == notes
    stored, _done = _stored_projection(body)
    rows = _rows_html(_render(TASK, stored["review_projection"]))
    assert "Ouroboros could not complete this review" in rows and "No reviewer answer is saved in this record." in rows


def test_a_service_error_after_reservation_keeps_its_cause_and_the_unknown_outcome(body, monkeypatch):
    from ouroboros.tools.preflight_review import _handle_review_status

    monkeypatch.setattr(review_mod, "_handle_multi_model_review",
                        lambda *_a, **_k: json.dumps({"error": f"review service closed the wave at {STAGE}"}))
    result = _review(body)
    assert body.sends == []
    record = rl.load_record(body.drive, result["record_id"])
    notes = _notes(record)
    assert len(notes) == 1 and notes[0]["block_reason"] == "infra_failure"
    assert "Review service returned an error" in notes[0]["reported_cause"] and STAGE in notes[0]["reported_cause"]
    # Custody recorded these reserved seats as lost (outcome unknown, never re-sent): the note rewrites none of them.
    assert {(row["status"], row["operation_state"]) for row in record["rows"]} == {("in_flight", "custody_lost")}
    assert record["state"] == "settled" and record["verdict"]["aggregate"] == "QUORUM_FAILED"
    status = json.loads(_handle_review_status(body.ctx, tool_name="review_change"))
    assert STAGE in _text(status), "review_status names the failed step (the settled attempt's block_details)"
    stored, _done = _stored_projection(body)
    rows = _rows_html(_render(TASK, stored["review_projection"]))
    assert rows.count("outcome unknown") == 3 and "Ouroboros could not complete this review" in rows


def test_a_crash_after_answers_keeps_every_answer_the_sibling_failure_and_the_crash(body, monkeypatch):
    golden = substrate.run_review_request

    def one_refuses(request, *, slots, drive_root, llm=None, usage_ctx=None):
        run = golden(request, slots=slots, drive_root=drive_root, llm=llm, usage_ctx=usage_ctx)
        for actor in run.actors:
            if actor["slot_id"] == "t2":
                actor.update(status="error", raw_text="", error="t2 engine refused before answering: quota window")
        return run

    def crash(*_args, **_kwargs):
        raise RuntimeError(f"coupling history unwritable at {STAGE} {SECRET}")
    monkeypatch.setattr(substrate, "run_review_request", one_refuses)
    monkeypatch.setattr(parallel_review, "_record_coupling_outcome", crash)
    result = _review(body)
    assert len(body.sends) == 3, "the crash buys no extra dispatch"
    record = rl.load_record(body.drive, result["record_id"])
    rows = {row["seat_id"]: row for row in record["rows"]}
    assert {seat: row["status"] for seat, row in rows.items()} == {"t1": "responded", "t2": "error", "s1": "responded"}
    assert rl.row_verdict(rows["t1"]) == rl.row_verdict(rows["s1"]) == "PASS", "received answers stay"
    assert "quota window" in rows["t2"]["reported_cause"]
    notes = _notes(record)
    assert len(notes) == 1 and notes[0]["block_reason"] == "infra_failure"
    assert notes[0]["reported_cause"].startswith("RuntimeError: ") and STAGE in notes[0]["reported_cause"]
    crashed = [item for item in record["verdict"]["degraded_reasons"] if item.startswith("review_change_wave_crashed: ")]
    assert crashed and STAGE in crashed[0]
    assert SECRET[:16] not in _text(record["verdict"]) + _text(result) + rl.index_path(body.drive).read_text(encoding="utf-8")
    stored, _done = _stored_projection(body)
    html = _rows_html(_render(TASK, stored["review_projection"]))
    assert "Ouroboros could not complete this review" in html and html.count("passed it") == 2
    t2 = html[html.index(':t2">'):].split('<div class="review-seat"', 1)[0]
    assert "unavailable" in t2 and "Engine report: " in t2 and "t2 engine refused before answering" in t2, t2


def test_a_reviewer_rejection_gains_no_wave_failure(body, monkeypatch):
    golden = substrate.run_review_request
    fail = json.dumps([{"item": "code_quality", "verdict": "FAIL", "severity": "critical", "reason": "t1 found a defect"}])

    def rejecting(request, *, slots, drive_root, llm=None, usage_ctx=None):
        run = golden(request, slots=slots, drive_root=drive_root, llm=llm, usage_ctx=usage_ctx)
        for actor in run.actors:
            if actor["slot_id"] == "t1":
                actor["raw_text"] = fail
        return run
    monkeypatch.setattr(substrate, "run_review_request", rejecting)
    result = _review(body)
    record = rl.load_record(body.drive, result["record_id"])
    assert record["verdict"]["aggregate"] == "FAIL" and len(body.sends) == 3
    assert "wave_failures" not in record["verdict"] and "wave_failures" not in result
    assert not [item for item in record["verdict"]["degraded_reasons"] if item.startswith("gate_block")]
    stored, _done = _stored_projection(body)
    html = _render(TASK, stored["review_projection"])
    assert "data-review-wave-failure" not in html and "rejected" in html


def test_secrets_are_masked_before_the_display_bound_wherever_they_fall(tmp_path):
    """Every offset across the cut: bounding first would leave a ``sk-ant-api03-Z…`` stub too short for any
    secret pattern (starts 426–434 here); masking first leaves none of the secret anywhere."""
    body = SECRET.removeprefix("sk-ant-api03-")
    record = SimpleNamespace(task_id=TASK, record_id="rl-boundary")
    for start in range(380, 481):
        words = "x" * start + " " + SECRET + " " + "y" * 600 + " " + TAIL
        note = wave_failure_note(tmp_path, record, {"block_reason": "infra_failure", "text": words})
        cause = note["reported_cause"]
        assert len(cause) <= 512 and "OMISSION NOTE" in cause, start
        assert SECRET[:14] not in cause and not any(body[i:i + 6] in cause for i in range(len(body) - 5)), (start, cause[-90:])
        assert rl.read_source(tmp_path, TASK, note["source_ref"]).decode("utf-8") == words, "the note keeps every byte"


def test_a_note_survives_pending_settlement_and_stays_readable(tmp_path):
    from tests.test_review_record_card_projection import CRITICAL, _answer, _gate_record

    record = _gate_record(TASK, opus_pending=True)
    record.verdict = {**record.verdict, "wave_failures": [wave_failure_note(
        tmp_path, record, {"block_reason": "infra_failure", "text": f"dispatch crashed at {STAGE}"})]}
    pending = rl.write_record(tmp_path, record)
    assert pending["state"] == "pending"
    fresh = dict(pending, verdict={**pending["verdict"], "wave_failures": []})
    assert rl.settle_pending_payload(pending, fresh)["verdict"]["wave_failures"] == _notes(pending)
    settled = rl.settle_from_collected(tmp_path, record.record_id, [{
        "slot_id": "opus", "model_id": "anthropic/claude-opus-5", "status": "responded", "parts": ["change"],
        "answers": {"change": _answer("change", "FAIL", [CRITICAL])}, "raw_text": "late answer"}])
    assert settled["state"] == "settled" and _notes(settled) == _notes(pending)
    sha = _notes(settled)[0]["source_ref"]["ref"]["sha256"]
    assert STAGE in rl.read_selected_source(tmp_path, TASK, record.record_id, "panel", role="failure", part="",
                                            sha256=sha).decode("utf-8")
    with pytest.raises(LookupError):
        rl.read_selected_source(tmp_path, TASK, record.record_id, "nobody", role="failure", part="", sha256=sha)


@pytest.mark.parametrize("source_state", ["legacy", "retained", "unavailable"])
def test_wave_note_retention_survives_index_rotation(tmp_path, monkeypatch, source_state):
    """A failed source publication keeps the same diagnostic hot; complete and legacy rows may rotate."""
    import errno
    from ouroboros import artifacts

    record = rl.ReviewLedgerRecord(record_id=f"rl-wave-{source_state}", task_id=TASK, surface="change")
    record.verdict = {"aggregate": "NOT_DISPATCHED"}
    if source_state != "legacy":
        def refuse_source(*_args, **_kwargs):
            raise OSError(errno.ENOSPC, "Source publication has no space left")

        with monkeypatch.context() as source_patch:
            if source_state == "unavailable":
                source_patch.setattr(artifacts, "store_actor_source_bytes", refuse_source)
            note = wave_failure_note(tmp_path, record, {"block_reason": "infra_failure", "text": STAGE})
        assert note["source_ref"]["status"] == source_state
        record.verdict["wave_failures"] = [note]
    payload = rl.write_record(tmp_path, record)
    resolvable = source_state != "unavailable"
    assert rl.record_sources_resolvable(tmp_path, payload) is resolvable
    indexed = json.loads(rl.index_path(tmp_path).read_text(encoding="utf-8"))
    assert indexed["heavy_stripped"] is resolvable
    if not resolvable:
        assert indexed["verdict"]["wave_failures"] == payload["verdict"]["wave_failures"]
    elif source_state == "retained":
        assert rl.read_source(tmp_path, TASK, note["source_ref"]).decode("utf-8") == STAGE

    monkeypatch.setattr(rl, "INDEX_MAX_BYTES", 1)
    rl._rotate_index_if_needed(tmp_path)
    hot = [json.loads(line) for line in rl.index_path(tmp_path).read_text(encoding="utf-8").splitlines()]
    archived = [json.loads(line) for segment in rl._index_segments(tmp_path)
                for line in segment.read_text(encoding="utf-8").splitlines()]
    assert [row["record_id"] for row in hot] == ([] if resolvable else [record.record_id])
    assert [row["record_id"] for row in archived] == ([record.record_id] if resolvable else [])
    if not resolvable:
        assert hot[0]["verdict"]["wave_failures"] == payload["verdict"]["wave_failures"]
