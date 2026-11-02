"""A review-ledger record reaches the task card through the production path.

The record is built by the ledger's own builder (``build_commit_gate_record``) and
written where the gate writes it; the real terminal write (``_store_task_result``), the
chat history row (``_record_task_facts``) and the task_done selection carry the
projection, and the real card formatter (``review_record_card.js``) and the task card's
Reviews section (``review_presentation.js``) print it. No hand-built projection.
"""

from tests import _dialogue as dialogue_test
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


def _gate_record(task_id, *, root_task_id="", opus_findings=12, opus_pending=False):
    """One composed commit-gate wave: ``sol`` retrieves (asked both parts), ``opus`` is a
    packet seat whose change answer FAILs with one critical among ``opus_findings`` (or, with
    ``opus_pending``, whose operation is still in flight), and ``critic`` was added beside the
    pool and errored."""
    notes = [{"item": f"note {n}", "verdict": "FAIL", "severity": "advisory"} for n in range(opus_findings - 1)]
    opus = ({"slot_id": "opus", "model_id": "anthropic/claude-opus-5", "status": "pending", "parts": ["change"],
             "operation_state": "in_flight", "late_result_pending": True} if opus_pending else
            {"slot_id": "opus", "model_id": "anthropic/claude-opus-5", "status": "responded", "parts": ["change"],
             "answers": {"change": _answer("change", "FAIL", [CRITICAL, *notes])}})
    raws = [
        {"slot_id": "sol", "model_id": "openai/gpt-5.6-sol", "status": "responded", "parts": ["change", "coupling"],
         "answers": {"change": _answer("change", "PASS"),
                     "coupling": _answer("coupling", "PASS", [{"item": "doc drift", "verdict": "PASS"}])}},
        opus,
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
    rows = dialogue_test.log_rows(tmp_path / "logs" / "chat.jsonl")
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
               and "model=not reported · requested=google/gemini-3.6-flash · transport=error · parse=none" in line
               and "quorum=abstains" in line for line in lines)
    assert panel["actors"][-1]["requested_model"] == "google/gemini-3.6-flash"
    assert panel["actors"][-1]["observed_model"] == ""
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


# What the strictness-keyed composer records (IMPL_SPEC §4.6, names as L1 writes them): the pool
# snapshot, strictness, who chose, the seats never reached; per seat its cost. Added here to a
# record the real builder made, so the projection and the card read exactly the stored shape.
SECRET = "sk-or-v1-" + "0123456789abcdef" * 3


def _pool_record(task_id, strictness="advisory", opus_pending=False, **panel):
    record = _gate_record(task_id, opus_pending=opus_pending)
    if strictness in ("advisory", "blocking"):  # the enforcement the wave ran under says the same
        record.enforcement, record.enforcement_blocks = strictness, strictness == "blocking"
    record.panel = {**record.panel, "strictness": strictness, "chosen_by": "ouroboros", "pool": ["sol", "opus", "grok"],
                    "not_reached": [{"seat_id": "critic", "cause": f"engine refused token {SECRET}"}], **panel}
    costs = {"sol": {"cost_usd": 0.4213, "cost_kind": "measured"}, "opus": {"cost_usd": None, "cost_kind": "subscription"},
             "critic": {"cost_usd": None, "cost_kind": "unknown"}}
    for seat in record.rows:
        seat.update(costs[seat["seat_id"]])
    return record


def test_pool_facts_reach_the_card_in_words_while_an_older_record_projects_as_before(tmp_path):
    old = rl.write_record(tmp_path, _gate_record("t-old"))
    rl.write_record(tmp_path, _pool_record("t-pool"))
    (old_panel,) = _terminal(tmp_path, "t-old")[0]["review_projection"]["panels"]
    # The older record: its own panel block, no cost, no seat additions, the card's old lines.
    assert old_panel["panel_facts"] == old["panel"] and "cost" not in old_panel
    assert not any({"cost", "additional"} & set(actor) for actor in old_panel["actors"])
    old_lines = _render("t-old", {"panels": [old_panel]})["card"].splitlines()
    assert any(line.startswith("Panel composition: composed · chosen by author · seats=2") for line in old_lines)
    assert not any(line.startswith(("Panel: ", "Wave total", "Why these reviewers")) for line in old_lines)

    stored, _event, history = _terminal(tmp_path, "t-pool")
    projection = stored["review_projection"]
    assert history["review_projection"] == projection
    (panel,) = projection["panels"]
    facts = panel["panel_facts"]
    assert facts["strictness"] == "advisory" and facts["pool"] == ["sol", "opus", "grok"] and facts["chosen_by"] == "ouroboros"
    assert facts["not_reached"] == [{"seat_id": "critic", "cause": "engine refused token ***REDACTED***"}]
    assert SECRET not in json.dumps(projection)
    actors = {actor["slot_id"]: actor for actor in panel["actors"]}
    assert actors["sol"]["cost"] == {"kind": "usd", "usd": 0.4213} and actors["sol"]["additional"] is False
    assert actors["opus"]["cost"] == {"kind": "subscription", "usd": None}
    assert actors["critic"]["cost"] == {"kind": "unknown", "usd": None} and actors["critic"]["additional"] is True
    assert panel["cost"] == {"usd": 0.4213, "seats": {"usd": 1, "subscription": 1, "unknown": 1}}

    shown = _render("t-pool", projection)
    lines = shown["card"].splitlines()
    assert "Panel: Advisory · chosen by Ouroboros · 2 of 3 ticked reviewers · 1 adviser, not counted" in lines
    assert "Why these reviewers: two engines are enough for a docs fix" in lines
    assert "Not reached: critic (engine refused token ***REDACTED***)" in lines
    assert "Wave total: $0.42 for the priced seats · 1 seat on a subscription · 1 seat cost unknown" in lines
    assert any(line.startswith("Reviewer sol: counted · $0.42 · role=commit_gate reviewer") for line in lines)
    assert any(line.startswith("Reviewer critic: adviser, not counted · cost unknown · role=") for line in lines)
    assert not any(line.startswith("Panel composition:") for line in lines)
    (group,) = [g for g in shown["groups"] if g["surface"] == "review_record"]
    assert "Panel: Advisory · chosen by Ouroboros · 2 of 3 ticked reviewers" in group["attempts"][0]["detailText"]


def test_malformed_pool_facts_are_dropped_never_guessed(tmp_path):
    from ouroboros.review_projection import ledger_record_panel

    # A Blocking wave that widened the requested subset: named seats survive, junk entries do not.
    record = rl.write_record(tmp_path, _pool_record("t-wide", strictness="blocking", chosen_by="owner", not_reached="x",
                                                     widened_from=["sol", {"seat_id": "opus"}, 7, ""]))
    facts = ledger_record_panel(record)["panel_facts"]
    assert facts["widened_from"] == ["sol", "opus"] and "not_reached" not in facts
    # A seat's own cost facts decide (an explicit unknown outranks the older `usd` carrier, a
    # measured kind needs an amount); a seat recorded without them shows the money it reported.
    sol, opus, critic = record["rows"]
    sol.update(cost_kind="unknown", cost_usd=None, usd=0.5)
    opus.pop("cost_kind"), opus.pop("cost_usd"), opus.update(usd=0.25)
    critic.update(cost_kind="measured", cost_usd=None)
    assert {actor["slot_id"]: actor["cost"] for actor in ledger_record_panel(record)["actors"]} == {
        "sol": {"kind": "unknown", "usd": None}, "opus": {"kind": "usd", "usd": 0.25},
        "critic": {"kind": "unknown", "usd": None}}
    lines = _render("t-wide", {"panels": [ledger_record_panel(record)]})["card"].splitlines()
    assert "Why these reviewers: widened to every ticked reviewer (Blocking) from the requested sol, opus · reason given: two engines are enough for a docs fix" in lines
    # Neither a pool nor a strictness the card can read: the record prints as before it existed.
    broken = rl.write_record(tmp_path, _pool_record("t-broken", strictness=" ", pool={"sol": True}))
    projected = ledger_record_panel(broken)
    assert "cost" not in projected and not {"pool", "strictness"} & set(projected["panel_facts"])
    assert "Panel composition: composed" in _render("t-broken", {"panels": [projected]})["card"]


def test_an_open_seat_reads_as_awaiting_and_a_preflight_keeps_its_pre_pool_line(tmp_path):
    from ouroboros.review_projection import ledger_record_panel

    # A wave whose opus seat is still in flight, built by the real ledger builder. Its `not_reached`
    # is whatever the producer lists (before fix round 1 the open seat too); the card reads opus's own
    # status, so opus is awaited and only the critic that failed is not reached.
    open_record = _pool_record("t-open", opus_pending=True)
    open_record.panel["not_reached"] = [entry for entry in rl.panel_not_reached(open_record.rows)
                                        if entry["seat_id"] != "opus"] + [{"seat_id": "opus", "cause": "in_flight"}]
    record = rl.write_record(tmp_path, open_record)
    assert record["state"] == "pending"
    (opus,) = [actor for actor in ledger_record_panel(record)["actors"] if actor["slot_id"] == "opus"]
    assert (opus["transport_status"], opus["operation_state"]) == ("pending", "in_flight")
    lines = _render("t-open", {"panels": [ledger_record_panel(record)]})["card"].splitlines()
    assert "Awaiting an answer: opus" in lines, lines
    assert not any(line.startswith("Not reached: opus") for line in lines)
    assert any(line.startswith("Not reached: critic (") for line in lines)

    # A preflight seats one named row and owes no reason: its card keeps the pre-pool line, also for
    # a record written while the composer still stamped pool facts on preflights, and the projection
    # adds no pool-era cost to it. The same record on the commit gate is a pool panel.
    preflight = _pool_record("t-pre")
    preflight.surface = "preflight"
    projected = ledger_record_panel(rl.write_record(tmp_path, preflight))
    assert "cost" not in projected and not any({"cost", "additional"} & set(actor) for actor in projected["actors"])
    lines = _render("t-pre", {"panels": [projected]})["card"].splitlines()
    assert any(line.startswith("Panel composition: composed · chosen by ouroboros · seats=2") for line in lines), lines
    assert not any(line.startswith(("Panel: ", "Why these reviewers", "Wave total", "Awaiting")) for line in lines)
    gate = ledger_record_panel(rl.write_record(tmp_path, _pool_record("t-gate")))
    assert "cost" in gate and gate["actors"][0]["cost"]["kind"] == "usd"
