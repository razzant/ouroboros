"""Static UI contracts for page chrome: shared header helper, scroll regions,
and evolution/consciousness wiring.

Consolidated in v5.15.x from three small files that each guarded one slice
of the SPA page-chrome layer:

- ``test_page_header_ui_static.py``        — renderPageHeader / renderTabStrip SSOT
- ``test_settings_and_page_layout_static.py`` — secrets generality + scroll regions
- ``test_evolution_ui_guards.py``          — evolution page + server runtime-state wiring
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Shared page header helper (renderPageHeader / renderTabStrip SSOT)
# ---------------------------------------------------------------------------
# Settings secrets layout + skills/widgets scroll regions
# ---------------------------------------------------------------------------


def test_settings_secrets_are_generic_and_integrations_tab_removed():
    # Stored keys are drawn from data (what installed skills asked for, the owner's own keys), never from a
    # skill's name written into the core UI.
    for name in ("settings_ui.js", "settings.js", "settings_home.js", "settings_accounts.js", "settings_rows.js"):
        text = _read(f"web/modules/{name}")
        assert "Integrations" not in text, name
        assert "TELEGRAM_" not in text, name
    accounts = _read("web/modules/settings_accounts.js")
    assert "requested_keys" in accounts
    assert "custom_secret_keys" in accounts
    # The old section's name still finds the GitHub rows in search.
    assert "'Source Control'" in _read("web/modules/settings_rows.js")


def test_onboarding_compact_access_step_keeps_default_width_two_column():
    # The rendered contract (columns, readable type, hidden ornament, visible
    # notes, choice meanings and live feedback) lives in
    # test_onboarding_catalog_browser.py::
    # test_connection_discovery_feedback_stays_visible_on_compact_screens.
    css = _read("web/onboarding.css")
    assert "@media (max-height: 820px), (max-width: 900px)" in css
    assert "@media (max-width: 760px)" in css


# ---------------------------------------------------------------------------
# Evolution / consciousness UI wiring
# ---------------------------------------------------------------------------


def test_server_navigation_and_chat_static_contracts():
    server_source = _read("server.py")
    state_source = _read("ouroboros/gateway/state.py")
    control_source = _read("ouroboros/gateway/control.py")
    app_source = _read("web/app.js")
    evo_source = _read("web/modules/evolution.js")
    chat_source = _read("web/modules/chat.js")
    css = _read("web/style.css")
    ui = _read("web/modules/settings_ui.js")
    costs = _read("web/modules/costs.js")

    assert "def _describe_bg_consciousness_state(requested_enabled: bool | None) -> dict:" in server_source
    assert '"evolution_state": evolution_state,' in state_source
    assert '"bg_consciousness_state": snap.get("bg_state") or {},' in _read("ouroboros/gateway/state_publication.py")
    assert 'request.query_params.get("force")' in control_source
    assert "window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: pageName } }));" in app_source
    # Evolution names Consciousness once, as a pointer to its own tab (its detail is that tab's).
    assert "openDashboardTab?.('consciousness')" in evo_source and "evo-consciousness-detail" not in evo_source
    # A room's bar says consciousness and a running campaign by its chip (DESIGN "Page header"); no More menu repeats them.
    status_source = _read("web/modules/consciousness_status.js")
    assert "normalizeConsciousness(data.bg_consciousness_state || {}, { enabled: data.bg_consciousness_enabled })" in status_source
    assert "const evolving = data.evolution_enabled === true;" in status_source
    assert "evolution_state" not in chat_source and "bg_consciousness_state" not in chat_source
    # The composer's markup and send path live in composer.js, mounted by chat.js.
    composer_source = _read("web/modules/composer.js")
    assert "${composerMarkup()}" in chat_source
    assert re.search(r'<input[^>]+id="chat-file-input"[^>]+multiple', composer_source)
    # The composer's staged files live in chat_attachments.js (DESIGN "Chat attachments");
    # composer.js uploads through it on Send and names the uploads for the model.
    attachments_source = _read("web/modules/chat_attachments.js")
    assert "createComposerAttachments({" in composer_source
    assert "composerText(text, uploadedAttachments.map(" in composer_source
    assert "attachmentTail(names);" in attachments_source  # every message but the exact /restart command
    assert "list.slice(0, ATTACHMENT_PREVIEW_COUNT)" in attachments_source
    assert "for (const item of [...pending])" in attachments_source
    assert "attachBtn.classList.toggle('uploading', flag)" in attachments_source
    assert "input.readOnly = flag;" in attachments_source
    assert "cleanupUploadedAttachments" in composer_source
    assert "await cleanupUploadedAttachments(e.uploaded || []);" in composer_source
    assert "await cleanupUploadedAttachments(uploadedAttachments);" in composer_source
    assert "ws.send(frame, hasAttachments ? { queue: false } : undefined)" in composer_source
    assert "result?.status !== 'sent'" in composer_source
    assert "data-attachment-remove" in attachments_source
    media_source = _read("web/modules/chat_media.js")
    assert "export async function cleanupUploadedAttachments" in media_source
    assert "method: 'DELETE'" in media_source
    assert "Promise.allSettled" in media_source
    assert '>Loading…</span>' in _read("web/modules/chat_header.js")
    assert "syncHeaderControlState({ accounting: { available: false } });" in chat_source
    assert "budget_text: 'Connecting...'" not in chat_source
    assert "send(msg, options = {})" in _read("web/modules/ws.js")
    assert "options.queue === false" in _read("web/modules/ws.js")
    assert ".chat-attachment-preview" in css and "flex-wrap" in css
    # The two budgets have one editor, the Costs rows: Settings draws no second field for them.
    assert "s-total-budget" not in ui and "s-settings-per-task-cost" not in ui
    # Source qualification belongs to the shared model editor, not duplicate model literals in Settings.
    model_roles = _read("web/modules/model_roles.js")
    assert "composeModelSource" in model_roles
    assert "route_editor_primitives.js" in model_roles
    # Costs' limits are setting rows: their floor comes from the setup contract through the row helper (the one
    # writer of a setting), which tells the other pages a row saved; Costs keeps no bound and no save of its own.
    controls = _read("web/modules/settings_controls.js")
    rows = _read("web/modules/settings_rows.js")
    assert "from: 'setup:totalBudget'" in rows and "from: 'setup:perTaskCostUsd'" in rows
    assert "lastSettings?._meta?.setup_contract" in controls
    assert "settings?._meta?.setup_contract?.budgetFields" in costs  # the note of how the limits count
    assert "const MIN_BUDGET_VALUE" not in costs and "Budget values must be at least 0.01." not in costs
    assert "new CustomEvent('ouro:settings-updated'" in controls and "source: 'settings-row'" in controls
    assert "new CustomEvent('ouro:settings-updated'" not in costs
    assert "source !== 'settings-row' || home !== LIMITS_PAGE" in costs
