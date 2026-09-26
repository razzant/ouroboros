from __future__ import annotations

import json

import pytest

pytest_plugins = ("tests.test_ui_smoke_playwright",)

from tests.test_ui_smoke_playwright import _open_review_checkpoint


@pytest.mark.ui_browser
def test_ui_smoke_review_truth_is_visible_in_chat_and_logs(direct_server_with_data):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    projection = {
        "panels": [{
            "panel_id": "panel_visual_truth",
            "surface": "task_acceptance",
            "authority": "host_root",
            "aggregate_signal": "DEGRADED",
            "transport_status": "partial",
            "parse_status": "malformed",
            "quorum": {"required": 2, "contributed": 1, "configured": 3},
            "enforcement_impact": "degrades_completion",
            "reason": "One reviewer timed out, so the panel did not reach quorum.",
            "candidate_hash": "candidate-visual",
            "evidence_revision": "evidence-visual",
            "fence_hash": "fence-visual-hash",
            "actors": [
                {
                    "slot_id": "fable",
                    "actor_role": "task acceptance",
                    "provider": "anthropic",
                    "model": "anthropic/claude-fable-5",
                    "transport_status": "success",
                    "parse_status": "valid",
                    "semantic_verdict": "DEGRADED",
                    "quorum_contribution": True,
                    "enforcement_impact": "supports_pass",
                    "reason": "The browser evidence is incomplete.",
                    "coverage": {"criteria_total": 3, "findings": 3},
                    "findings": [
                        {
                            "severity": "high",
                            "item": "Browser evidence missing for the checkout flow",
                            "evidence": "no screenshot covers step 3",
                            "recommendation": "Capture the payment page state",
                        },
                        {"severity": "low", "item": "Trace summary is terse"},
                    ],
                    "findings_omitted": 1,
                    "response_ref": {"call_id": "call-fable-1", "sha256": "a" * 64},
                },
                {
                    "slot_id": "sol",
                    "actor_role": "task acceptance",
                    "provider": "openai",
                    "model": "openai/gpt-5.6-sol",
                    "transport_status": "timeout",
                    "parse_status": "malformed",
                    "semantic_verdict": "",
                    "quorum_contribution": False,
                    "enforcement_impact": "abstains",
                    "reason": "Provider request timed out.",
                },
            ],
        }],
    }
    axes = {
        "lifecycle": {"status": "completed"},
        "execution": {"status": "ok"},
        "objective": {"status": "best_effort"},
        "review": {"status": "degraded"},
        "artifacts": {"status": "ready"},
    }
    summary = {
        "ts": "2026-07-15T10:00:00+00:00",
        "direction": "system",
        "type": "task_summary",
        "task_id": "review-ui",
        "chat_id": 1,
        "text": "Task finished with review evidence.",
        "tool_calls": 0,
        "rounds": 1,
        "outcome_axes": axes,
        "review_projection": projection,
    }
    event = {
        "ts": "2026-07-15T10:00:01+00:00",
        "type": "task_done",
        "task_id": "review-ui",
        "task_type": "task",
        "status": "completed",
        "outcome_axes": axes,
        "review_projection": projection,
    }
    ordinary_final = {
        "ts": "2026-07-15T10:00:00.500000+00:00",
        "direction": "out",
        "chat_id": 1,
        "task_id": "review-no-summary",
        "text": "Normal final answer after the terminal progress anchor.",
        "format": "markdown",
    }
    (logs_dir / "chat.jsonl").write_text(
        json.dumps(summary) + "\n" + json.dumps(ordinary_final) + "\n",
        encoding="utf-8",
    )
    (logs_dir / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    (logs_dir / "progress.jsonl").write_text(json.dumps({
        "ts": "2026-07-15T09:59:59+00:00",
        "chat_id": 1,
        "task_id": "review-no-summary",
        "content": "Terminal review must survive without a task summary.",
    }) + "\n", encoding="utf-8")
    task_results = data_dir / "task_results"
    task_results.mkdir(parents=True, exist_ok=True)
    plan_state = {
        "schema_version": 2,
        "current_attempt": {"fingerprint": "f" * 64, "status": "open"},
        "waves": [{
            "request_fingerprint": "f" * 64,
            "cycle_index": 1,
            "aggregate": "REVISE_PLAN",
            "closed": False,
            "paid": True,
            "counts": {"blocking": 1, "note": 0, "need_evidence": 0},
            "findings": [{
                "finding_id": "slot_1:f1",
                "id": "f1",
                "class": "blocking",
                "summary": "The migration step drops the audit ledger",
                "breaks": "invariant_1",
                "locator": "ouroboros/usage_accounting.py",
                "recommendation": "Keep the ledger append-only through the migration",
                "slot": "slot_1",
                "model": "anthropic/claude-fable-5",
            }],
            "dispositions": [{
                "finding_id": "slot_1:f1",
                "decision": "accept",
                "rationale": "will rework the migration",
            }],
            "actors": [
                {"slot_id": "slot_1", "model": "anthropic/claude-fable-5", "ok": True},
            ],
            "reviewed_at": "2026-07-15T09:59:58+00:00",
        }],
        "waves_omitted": 0,
    }
    (task_results / "review-no-summary.json").write_text(json.dumps({
        "_schema_version": 1,
        "task_id": "review-no-summary",
        "status": "completed",
        "reason_code": "acceptance_degraded",
        "outcome_axes": axes,
        "review_projection": projection,
        "plan_review_state": plan_state,
    }) + "\n", encoding="utf-8")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                card = page.locator('.chat-live-card[data-task-id="review-ui"]')
                card.wait_for(state="attached", timeout=30_000)
                assert card.get_attribute("data-expanded") == "0"
                assert "Review panel panel_visual_truth" not in card.inner_text()
                _open_review_checkpoint(card)
                chat_text = card.inner_text()
                assert "Done with warnings" in chat_text
                assert "Notice" not in chat_text
                assert "Review panel panel_visual_truth" in chat_text
                assert "Reviewer fable" in chat_text
                assert "Reviewer sol" in chat_text
                # The reviewer's actual findings are readable, not just counted.
                assert "Browser evidence missing for the checkout flow" in chat_text
                assert "fix: Capture the payment page state" in chat_text
                assert "Reviewer fable findings omitted: 1" in chat_text
                assert "observability call call-fable-1" in chat_text
                no_summary = page.locator('.chat-live-card[data-task-id="review-no-summary"]')
                no_summary.wait_for(state="attached", timeout=30_000)
                assert no_summary.get_attribute("data-expanded") == "0"
                # This card carries BOTH groups; the helper opens the first
                # (Plan review), whose finding text must be readable.
                _open_review_checkpoint(no_summary)
                assert no_summary.locator('[data-live-phase]').first.get_attribute("data-phase") == "warn"
                plan_text = no_summary.inner_text()
                assert "The migration step drops the audit ledger" in plan_text
                assert "breaks invariant_1" in plan_text
                assert "agent: accept — will rework the migration" in plan_text
                acceptance_toggle = no_summary.locator(
                    '[data-review-group-toggle="task_acceptance:review-no-summary"]',
                )
                acceptance_toggle.click()
                no_summary.locator(
                    '[data-review-group="task_acceptance:review-no-summary"]'
                    ' [data-review-attempt-toggle]',
                ).first.click()
                assert "Review panel panel_visual_truth" in no_summary.inner_text()
                page.wait_for_timeout(900)  # cover the routine background history sync
                assert no_summary.locator('.chat-live-line-repeat:not([hidden])').count() == 0
                assert card.locator('.chat-live-line-repeat:not([hidden])').count() == 0
                page.screenshot(path=str(data_dir.parent / "review-truth-chat.png"), full_page=True)

                page.click('[data-nav-page="dashboard"]')
                page.click('[data-dashboard-tab="logs"]')
                log_card = page.locator('.log-task-card[data-task-group="review-ui"]')
                log_card.wait_for(state="attached", timeout=30_000)
                assert log_card.is_visible()
                review = log_card.locator('[data-task-review]')
                assert review.is_visible()
                log_text = review.inner_text()
                assert "Review panel panel_visual_truth" in log_text
                assert "Reviewer fable" in log_text
                assert "Reviewer sol" in log_text
                # The same formatter serves Logs: finding bodies ride along.
                assert "Browser evidence missing for the checkout flow" in log_text
                assert "Reviewer fable findings omitted: 1" in log_text
                assert log_card.locator('[data-task-phase]').inner_text() == "warn"
                review.scroll_into_view_if_needed()
                review.screenshot(path=str(data_dir.parent / "review-truth-logs.png"))
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
@pytest.mark.serial
def test_ui_smoke_action_only_author_finish_keeps_the_reviewer_fail(direct_server_with_data, tmp_path, monkeypatch):
    """TZ-2 C4 in the real review renderer: Main's final response after a FAIL panel is an
    action-only finish (no stance). The record comes from the real acceptance loop (only the
    model and the panel are substituted); the card shows the author's act and rationale beside
    the independent reviewer FAIL and never an invented accepted/partial/solved."""
    import re

    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.review_substrate import compact_review_projection
    from tests.test_acceptance_author_stop import _run_stop_loop

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    (tmp_path / "producer").mkdir()
    run = _run_stop_loop(tmp_path / "producer", monkeypatch, ["The export endpoint ships."] * 3, stop_services=False)
    axes = derive_loop_outcome(run.result, run.usage, run.trace)["outcome_axes"]
    decision = axes["review"]["acceptance_decision"]
    assert decision["reason"] == "author_finish" and decision["author_disposition"]["disposition"] == ""
    projection = compact_review_projection(run.trace["review_runs"])
    data_dir = direct_server_with_data["data_dir"]
    (data_dir / "logs").mkdir(parents=True, exist_ok=True)
    task_id = "c4-action-only"
    (data_dir / "logs" / "chat.jsonl").write_text(json.dumps({
        "ts": "2026-09-26T10:00:00+00:00", "direction": "system", "type": "task_summary", "task_id": task_id,
        "chat_id": 1, "status": "completed", "text": "", "tool_calls": 0, "rounds": 1,
        "outcome_axes": axes, "review_projection": projection}) + "\n", encoding="utf-8")
    (data_dir / "task_results").mkdir(parents=True, exist_ok=True)
    (data_dir / "task_results" / f"{task_id}.json").write_text(json.dumps({
        "_schema_version": 1, "task_id": task_id, "status": "completed", "reason_code": "final_message",
        "result": run.result, "outcome_axes": axes, "review_projection": projection}) + "\n", encoding="utf-8")
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            try:
                page.goto(direct_server_with_data["url"], wait_until="domcontentloaded", timeout=30_000)
                card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
                card.wait_for(state="attached", timeout=30_000)
                _open_review_checkpoint(card)
                author = card.locator("[data-review-author-decision]")
                author.wait_for(state="visible", timeout=10_000)
                author_text = author.inner_text()
                assert "Task author decision" in author_text and "Author finish" in author_text
                assert "reviewer signal=FAIL" in author_text
                assert "Main submitted this complete response for delivery" in author_text
                assert not re.search(r"accepted|partial|solved|Author finish:", author_text, re.I), author_text
                group = card.locator(f'[data-review-group="task_acceptance:{task_id}"]')
                assert "FAIL" in group.locator(".chat-review-group-meta").inner_text()
                assert "FAIL" in group.locator("[data-review-attempt] .chat-review-attempt-meta").first.inner_text()
                assert not re.search(r"\bsolved\b", card.inner_text(), re.I)
                card.scroll_into_view_if_needed()
                card.screenshot(path=str(data_dir.parent / "c4-action-only-author-finish.png"))
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
