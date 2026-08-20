"""What the owner can see and decide about a running task.

Split verbatim out of ``tests/test_ui_smoke_playwright.py`` by theme. This module owns
the review truth the chat and the logs must both show, the skip-review button, the input
dialog that resolves an object result even after it was superseded, and the eligibility
of the cancel button together with the cancelled state it leaves behind.

Every test here launches a real browser and is marked ``ui_browser``, so the default
local run deselects the whole module.
"""

from __future__ import annotations

import json

import pytest


from tests._ui_smoke_shared import direct_server_with_data as _direct_server_with_data

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
direct_server_with_data = _direct_server_with_data


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
    (task_results / "review-no-summary.json").write_text(json.dumps({
        "task_id": "review-no-summary",
        "status": "completed",
        "reason_code": "acceptance_degraded",
        "outcome_axes": axes,
        "review_projection": projection,
    }) + "\n", encoding="utf-8")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                card = page.locator('.chat-live-card[data-task-id="review-ui"]')
                card.wait_for(state="attached", timeout=30_000)
                assert card.is_visible()
                assert card.get_attribute("data-expanded") == "1"
                chat_text = card.inner_text()
                assert "Notice" in chat_text
                assert "Review panel panel_visual_truth" in chat_text
                assert "Reviewer fable" in chat_text
                assert "Reviewer sol" in chat_text
                no_summary = page.locator('.chat-live-card[data-task-id="review-no-summary"]')
                no_summary.wait_for(state="attached", timeout=30_000)
                assert no_summary.is_visible()
                assert no_summary.get_attribute("data-expanded") == "1"
                assert no_summary.locator('[data-live-phase]').first.get_attribute("data-phase") == "warn"
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
def test_ui_smoke_v639_skip_review_button(direct_server_with_data):
    # C1: the owner-only "⚠️ Skip review" action is offered for the owner's OWN (external)
    # skill and hash-verified official-hub payloads that still need review, and NEVER for
    # native/ClawHub/unverified marketplace payloads.
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    data_dir = direct_server_with_data["data_dir"]
    url = direct_server_with_data["url"]
    manifest = ("---\nname: {n}\ntype: instruction\ndescription: smoke skill\n"
                "version: 0.1.0\n---\n# {n}\nDo a thing.\n")
    ext = data_dir / "skills" / "external" / "owntool"
    ext.mkdir(parents=True, exist_ok=True)
    (ext / "SKILL.md").write_text(manifest.format(n="owntool"), encoding="utf-8")
    mk = data_dir / "skills" / "clawhub" / "markettool"
    mk.mkdir(parents=True, exist_ok=True)
    (mk / "SKILL.md").write_text(manifest.format(n="markettool"), encoding="utf-8")
    # A real marketplace skill carries clawhub provenance -> resolves to source=clawhub
    # (without it, an unprovenanced clawhub-bucket payload is treated as owner-own external).
    (mk / ".clawhub.json").write_text(
        json.dumps({"slug": "markettool", "version": "0.1.0"}), encoding="utf-8")
    # An already owner-attested skill: must show the distinct 'owner-attested' badge.
    att = data_dir / "skills" / "external" / "attestedtool"
    att.mkdir(parents=True, exist_ok=True)
    (att / "SKILL.md").write_text(manifest.format(n="attestedtool"), encoding="utf-8")
    att_state = data_dir / "state" / "skills" / "attestedtool"
    att_state.mkdir(parents=True, exist_ok=True)
    (att_state / "review.json").write_text(json.dumps({
        "status": "clean", "content_hash": "seed", "review_profile": "owner_attested",
        "reviewer_models": ["owner_attestation"],
        "findings": [{"item": "owner_attestation", "verdict": "PASS", "severity": "info", "reason": "owner attested"}],
    }), encoding="utf-8")
    (att_state / "owner_attestation.json").write_text(
        json.dumps({"attested_at": "now", "content_hash": "seed"}), encoding="utf-8")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.click('[data-nav-page="skills"]')
                page.wait_for_selector("#page-skills", timeout=30_000)
                page.wait_for_selector('.skills-card[data-skill="owntool"]', timeout=30_000)
                own = page.locator('.skills-card[data-skill="owntool"]').first
                market = page.locator('.skills-card[data-skill="markettool"]').first
                # owner-own external skill that still needs review -> Skip review offered.
                assert own.locator(".skills-attest-review").count() == 1
                assert "Skip review" in (
                    own.locator(".skills-attest-review").first.text_content() or "")
                # ClawHub marketplace skill -> never attestable, no Skip review action.
                assert market.locator(".skills-attest-review").count() == 0
                # owner-attested skill -> distinct 'owner-attested' badge (review_profile surfaced).
                page.wait_for_selector('.skills-card[data-skill="attestedtool"]', timeout=30_000)
                att_card = page.locator('.skills-card[data-skill="attestedtool"]').first
                assert att_card.locator(".skills-badge").filter(
                    has_text="owner-attested").count() >= 1
                # submitHubReady guard: an owner-attested skill must NOT offer an enabled
                # publish (the hub refuses to publish owner-attested skills). Render the card
                # WITH a github token configured (in-page module import — node exec is blocked)
                # and assert Submit-to-OuroborosHub is disabled for the owner-attested reason.
                submit_html = page.evaluate(
                    """async () => {
                        const m = await import('/static/modules/skill_card_renderer.js');
                        return m.renderInstalledSkillCard(
                            { name: 'att', type: 'instruction', version: '0.1.0', source: 'external',
                              is_self_authored: true, review_status: 'clean',
                              review_gate: { executable_review: true }, review_stale: false,
                              review_profile: 'owner_attested', grants: {}, permissions: [],
                              payload_root: 'skills/external/att', enabled: true },
                            new Set(), new Set(), {}, { githubTokenConfigured: true });
                    }"""
                )
                assert 'data-submit-disabled="true"' in submit_html
                assert "owner-attested" in submit_html.lower()
                # Defense-in-depth (mirrors the backend source gate): a marketplace skill
                # mislabeled self-authored must STILL NOT offer Skip review.
                market_self_html = page.evaluate(
                    """async () => {
                        const m = await import('/static/modules/skill_card_renderer.js');
                        return m.renderInstalledSkillCard(
                            { name: 'mk2', type: 'instruction', version: '0.1.0', source: 'clawhub',
                              is_self_authored: true, review_status: 'pending',
                              review_gate: { executable_review: false }, review_stale: false,
                              review_profile: '', grants: {}, permissions: [],
                              payload_root: 'skills/clawhub/mk2', enabled: false },
                            new Set(), new Set(), {}, {});
                    }"""
                )
                assert "skills-attest-review" not in market_self_html
                # Unverified OuroborosHub payloads also stay blocked; only the official_hub
                # profile is a cheap UI hint, and the backend still re-verifies.
                hub_html = page.evaluate(
                    """async () => {
                        const m = await import('/static/modules/skill_card_renderer.js');
                        return {
                          unverified: m.renderInstalledSkillCard(
                            { name: 'hub1', type: 'instruction', version: '0.1.0', source: 'ouroboroshub',
                              is_self_authored: false, review_status: 'pending',
                              review_gate: { executable_review: false }, review_stale: false,
                              review_profile: '', grants: {}, permissions: [],
                              payload_root: 'skills/ouroboroshub/hub1', enabled: false },
                            new Set(), new Set(), {}, {}),
                          verified: m.renderInstalledSkillCard(
                            { name: 'hub2', type: 'instruction', version: '0.1.0', source: 'ouroboroshub',
                              is_self_authored: false, review_status: 'pending',
                              review_gate: { executable_review: false }, review_stale: false,
                              review_profile: '', owner_attestable: true, official_hub_verified: true,
                              grants: {}, permissions: [],
                              payload_root: 'skills/ouroboroshub/hub2', enabled: false },
                            new Set(), new Set(), {}, {}),
                          staleProfile: m.renderInstalledSkillCard(
                            { name: 'hub3', type: 'instruction', version: '0.1.0', source: 'ouroboroshub',
                              is_self_authored: false, review_status: 'pending',
                              review_gate: { executable_review: false }, review_stale: true,
                              review_profile: 'official_hub', owner_attestable: false,
                              official_hub_verified: false, grants: {}, permissions: [],
                              payload_root: 'skills/ouroboroshub/hub3', enabled: false },
                            new Set(), new Set(), {}, {})
                        };
                    }"""
                )
                assert "skills-attest-review" not in hub_html["unverified"]
                assert "skills-attest-review" in hub_html["verified"]
                assert "skills-attest-review" not in hub_html["staleProfile"]
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

@pytest.mark.ui_browser
def test_ui_smoke_superseded_input_dialog_resolves_object_result(direct_server_with_data):
    """v6.90.3 dialog contract: superseding an INPUT dialog with a newer dialog
    resolves the documented {confirmed: false, value: ''} — never a bare false
    the docs do not promise (the supersession close is mode-aware)."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                first_result = page.evaluate(
                    """
                    async () => {
                        const m = await import('/static/modules/confirm_dialog.js');
                        const first = m.openConfirmDialog({
                            title: 'first', body: 'input dialog', input: true,
                        });
                        const second = m.openConfirmDialog({
                            title: 'second', body: 'supersedes the first',
                        });
                        const r1 = await first;
                        document.querySelector('[data-confirm-cancel]')?.click();
                        await second;
                        return r1;
                    }
                    """
                )
                assert first_result == {"confirmed": False, "value": ""}
                assert page.locator(".confirm-dialog").count() == 0
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

@pytest.mark.ui_browser
def test_ui_smoke_cancel_run_button_eligibility_and_cancelled_state(direct_server_with_data):
    """v6.82 P5 / S3 Q2: the stop control renders ONLY on live marker-attested
    root cards (never marker-less direct-turn cards, subagent children, or the
    reusable background slot), opens the dropdown, and a cancelled root
    replays as an honest warn-toned "Cancelled" — never a generic "Done"."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "chat.jsonl").write_text("", encoding="utf-8")
    rows = [
        # Pooled live root: carries the supervisor's host-attested marker.
        {"ts": "2026-07-29T10:00:00+00:00", "chat_id": 1, "task_id": "live-root",
         "content": "Working on the big thing", "cancelable": True},
        # Direct-chat-turn shape: same card shape, NO marker -> no button.
        {"ts": "2026-07-29T10:00:01+00:00", "chat_id": 1, "task_id": "direct-turn",
         "content": "Inline turn narration"},
        # Subagent child of the live root: marker present but child cards never
        # offer the action (the root cascade covers them).
        {"ts": "2026-07-29T10:00:02+00:00", "chat_id": 1, "task_id": "sub-child1",
         "content": "Collecting evidence", "delegation_role": "subagent",
         "subagent_event": "scheduled", "subagent_task_id": "sub-child1",
         "parent_task_id": "live-root", "subagent_role": "researcher",
         "cancelable": True},
        # Reusable background-consciousness slot: never eligible.
        {"ts": "2026-07-29T10:00:03+00:00", "chat_id": 1, "task_id": "bg-consciousness",
         "content": "Background thinking", "cancelable": True},
        # A root that was force-cancelled before this reload.
        {"ts": "2026-07-29T10:00:04+00:00", "chat_id": 1, "task_id": "gone-root",
         "content": "Was working before the cancel", "cancelable": True},
    ]
    (logs_dir / "progress.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    task_results = data_dir / "task_results"
    task_results.mkdir(parents=True, exist_ok=True)
    (task_results / "gone-root.json").write_text(json.dumps({
        "task_id": "gone-root",
        "status": "cancelled",
        "reason_code": "cancelled",
        "outcome_axes": {
            "lifecycle": {"status": "cancelled"},
            "execution": {"status": "cancelled"},
        },
    }) + "\n", encoding="utf-8")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                live = page.locator('.chat-live-card[data-task-id="live-root"]')
                live.wait_for(state="attached", timeout=30_000)
                cancel_btn = live.locator('[data-cancel-run]')
                cancel_btn.wait_for(state="attached", timeout=30_000)
                assert cancel_btn.inner_text().strip() == "Stop…"
                # Marker-less direct-turn shape, subagent child, reusable slot,
                # and the finished cancelled root must NOT offer the action.
                for absent_id in ("direct-turn", "sub-child1", "bg-consciousness", "gone-root"):
                    card = page.locator(f'.chat-live-card[data-task-id="{absent_id}"]')
                    card.wait_for(state="attached", timeout=30_000)
                    assert card.locator('[data-cancel-run]').count() == 0, absent_id
                # The cancelled root replays as an honest Cancelled state.
                gone_phase = page.locator('.chat-live-card[data-task-id="gone-root"] [data-live-phase]')
                page.wait_for_function(
                    "() => document.querySelector('.chat-live-card[data-task-id=\"gone-root\"]"
                    " [data-live-phase]')?.textContent === 'Cancelled'",
                    timeout=30_000,
                )
                assert "cancelled" in (gone_phase.get_attribute("class") or "")
                # Dropdown wiring (S3 Q2): open, then dismiss = keep running.
                cancel_btn.click()
                menu = live.locator('.task-control-menu')
                menu.wait_for(state="visible", timeout=10_000)
                assert "Wrap up" in menu.inner_text()
                page.keyboard.press("Escape")
                menu.wait_for(state="detached", timeout=10_000)
                assert cancel_btn.is_enabled()
                page.screenshot(path=str(data_dir.parent / "cancel-run.png"), full_page=True)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


# The in-flight indicator lifecycle smoke test lives in
# tests/test_ui_smoke_inflight_indicator.py (upstream v6.104.0; size-ratchet byte gate).
