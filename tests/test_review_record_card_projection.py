"""A review-ledger record reaches the task card through the production path.

The record is built by the ledger's own builder (``build_commit_gate_record``) and
written where the gate writes it; the real terminal write (``_store_task_result``), the
chat history row (``_record_task_facts``) and the task_done selection carry the
projection, and the real card formatter (``review_record_card.js``) and the task card's
Reviews section (``review_presentation.js``) print it. No hand-built projection.
"""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from ouroboros import review_ledger as rl
from ouroboros.agent_task_pipeline import _store_task_result
from ouroboros.post_task_synthesis import _record_task_facts
from ouroboros.task_results import load_task_result
from supervisor.events_task_done import _task_done_review_projection

REPO = Path(__file__).resolve().parents[1]
SHOWN = 5  # a card carries the five newest records, as ``## Review records`` lists them
CRITICAL = {"item": "unbounded retry loop", "verdict": "FAIL", "severity": "critical", "reason": "retries forever"}
RENDER = """
import fs from 'node:fs';
import { formatReviewProjection } from './web/modules/review_record_card.js';
import { reviewGroupsFromTaskDetail } from './web/modules/review_presentation.js';
const detail = JSON.parse(fs.readFileSync(0, 'utf8'));
const groups = reviewGroupsFromTaskDetail(detail, detail.task_id);
console.log(JSON.stringify({ card: formatReviewProjection(detail.review_projection), groups }));
"""


def _answer(part, verdict, findings=()):
    findings = [dict(item) for item in findings]
    return {"status": "responded", "verdict": verdict, "findings": findings,
            "critical": sum(1 for item in findings if item.get("severity") == "critical"),
            "coverage": "n/a" if part == "change" else "full"}


def _gate_record(task_id, *, root_task_id="", opus_findings=12):
    """One composed commit-gate wave: ``sol`` retrieves (asked both parts), ``opus`` is a
    packet seat whose change answer FAILs with one critical among ``opus_findings``, and
    ``critic`` was added beside the pool and errored."""
    notes = [{"item": f"note {n}", "verdict": "FAIL", "severity": "advisory"} for n in range(opus_findings - 1)]
    raws = [
        {"slot_id": "sol", "model_id": "openai/gpt-5.6-sol", "status": "responded", "parts": ["change", "coupling"],
         "answers": {"change": _answer("change", "PASS"),
                     "coupling": _answer("coupling", "PASS", [{"item": "doc drift", "verdict": "PASS"}])}},
        {"slot_id": "opus", "model_id": "anthropic/claude-opus-5", "status": "responded", "parts": ["change"],
         "answers": {"change": _answer("change", "FAIL", [CRITICAL, *notes])}},
        {"slot_id": "critic", "model_id": "", "status": "error", "parts": ["change"]},
    ]
    plans = [
        {"slot_id": "sol", "model": "openai/gpt-5.6-sol", "route": "api_chat", "parts": ["change", "coupling"],
         "retrieves": True},
        {"slot_id": "opus", "model": "anthropic/claude-opus-5", "route": "api_chat", "parts": ["change"]},
        {"slot_id": "critic", "model": "google/gemini-3.6-flash", "route": "api_chat", "parts": ["change"],
         "additional": True},
    ]
    return rl.build_commit_gate_record({
        "task_id": task_id, "root_task_id": root_task_id or task_id, "enforcement": "blocking",
        "enforcement_blocks": True, "composition": "composed", "chosen_by": "author",
        "composition_reason": "two engines are enough for a docs fix",
        "triad_raw": raws, "structured": {"rows": plans, "started_ts": "2026-10-08T00:00:00+00:00"},
        "binding": {"tree_sha": "t" * 40, "parents": ["p" * 40], "diff_sha256": "d" * 64},
    })


def _terminal(tmp_path, task_id):
    """The task's terminal write, its task_done projection and the chat history row."""
    env, task = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), {"id": task_id, "type": "task"}
    _store_task_result(env, task, "done", {"rounds": 1, "cost": 0}, {"tool_calls": []})
    stored = load_task_result(tmp_path, task_id) or {}
    _record_task_facts(env, task, {"rounds": 1}, {"tool_calls": []}, tmp_path / "logs")
    rows = [json.loads(line) for line in (tmp_path / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()]
    history = next(row for row in rows if row.get("summary_kind") == "host_task_facts" and row["task_id"] == task_id)
    return stored, _task_done_review_projection(stored, {}), history


def _render(task_id, projection):
    result = subprocess.run(["node", "--input-type=module", "-e", RENDER], cwd=REPO, capture_output=True,
                            input=json.dumps({"task_id": task_id, "review_projection": projection}),
                            text=True, encoding="utf-8", check=True)
    return json.loads(result.stdout)


def test_ledger_record_reaches_the_task_card_with_panel_facts_and_per_part_answers(tmp_path):
    record = rl.write_record(tmp_path, _gate_record("t-rev"))
    stored, event_projection, history = _terminal(tmp_path, "t-rev")
    projection = stored["review_projection"]
    assert event_projection == projection and history["review_projection"] == projection
    (panel,) = projection["panels"]
    assert panel["record_id"] == record["record_id"] and panel["panel_facts"] == record["panel"]
    assert projection["review_records_omitted"] == 0
    # Bounded like every actor projection, and the bound loses no verdict: twelve
    # findings are counted, eight bodies ride, four are said to be omitted.
    opus = next(actor for actor in panel["actors"] if actor["slot_id"] == "opus")
    assert len(opus["findings"]) == 8 and opus["findings_omitted"] == 4
    assert opus["answers"]["change"]["findings"] == 12 and panel["aggregate_signal"] == record["verdict"]["aggregate"] == "FAIL"

    shown = _render("t-rev", projection)
    lines = shown["card"].splitlines()
    assert f"Review panel {record['record_id']}: commit_gate · authority=review_ledger · verdict=FAIL" in lines[0]
    assert "quorum=2/2" in lines[0] and "enforcement=blocking" in lines[0]
    assert "Panel reason: critical_findings" in lines
    composition = next(line for line in lines if line.startswith("Panel composition: "))
    assert composition.startswith("Panel composition: composed · chosen by author · seats=2")
    assert "additional seats=1" in composition and "model not observed on 1 seat(s)" in composition
    assert composition.endswith("reason: two engines are enough for a docs fix")
    assert ("Reviewer sol answers: change=responded PASS (findings 0, critical 0)"
            " · coupling=responded PASS (findings 1, critical 0, coverage full)") in lines
    assert "Reviewer opus answers: change=responded FAIL (findings 12, critical 1)" in lines
    assert "Reviewer opus finding: [critical FAIL] unbounded retry loop — reason: retries forever" in lines
    assert "Reviewer opus findings omitted: 4" in lines
    assert any(line.startswith("Reviewer critic: role=commit_gate additional reviewer")
               and "model=google/gemini-3.6-flash · transport=error · parse=none" in line
               and "quorum=abstains" in line for line in lines)
    assert "Reviewer critic answers: change=error (findings 0, critical 0)" in lines

    # The task card's Reviews section shows the same record under its own group.
    (group,) = [g for g in shown["groups"] if g["surface"] == "review_record"]
    assert group["label"] == "Review records" and group["verdict"] == "FAIL" and group["tone"] == "error"
    assert group["countIsAuthoritative"] is True and group["attemptCount"] == 1
    detail = group["attempts"][0]["detailText"]
    assert composition in detail and "Reviewer opus answers: change=responded FAIL (findings 12, critical 1)" in detail
    assert detail.endswith(f"Full record: state/review_ledger/{record['record_id']}.json")


def test_card_shows_only_this_tasks_records_and_says_when_older_ones_exist(tmp_path):
    # Nothing recorded: no record panel, no composition or answers lines, no group.
    stored, _event, _history = _terminal(tmp_path, "t-none")
    assert not any(panel.get("record_id") for panel in (stored.get("review_projection") or {}).get("panels", []))
    shown = _render("t-none", stored.get("review_projection") or {})
    assert "Panel composition" not in shown["card"] and "answers:" not in shown["card"]
    assert not [g for g in shown["groups"] if g["surface"] == "review_record"]

    # A child's record names the root as its root task; it belongs to the child's card.
    rl.write_record(tmp_path, _gate_record("t-child", root_task_id="t-root"))
    stored, _event, _history = _terminal(tmp_path, "t-root")
    assert not (stored.get("review_projection") or {}).get("panels")

    # More records than a card carries: the newest ones, oldest first, and "1+" more.
    written = [rl.write_record(tmp_path, _gate_record("t-many"))["record_id"] for _ in range(SHOWN + 1)]
    stored, _event, _history = _terminal(tmp_path, "t-many")
    projection = stored["review_projection"]
    assert [panel["record_id"] for panel in projection["panels"]] == written[1:]
    assert projection["review_records_omitted"] == "1+"
    (group,) = [g for g in _render("t-many", projection)["groups"] if g["surface"] == "review_record"]
    assert group["countIsAuthoritative"] is False and group["attemptCount"] == SHOWN
