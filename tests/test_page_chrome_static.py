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
    ui = _read("web/modules/settings_ui.js")
    settings = _read("web/modules/settings.js")
    assert "Integrations" not in ui
    assert "TELEGRAM_" not in ui
    assert "TELEGRAM_" not in settings
    assert "skill-requested-secrets" in ui
    assert "custom-secrets-list" in ui
    assert "Source Control" in ui


def test_onboarding_compact_access_step_keeps_default_width_two_column():
    # The rendered contract (columns, readable type, hidden explanatory copy
    # and visible live feedback) lives in test_onboarding_catalog_browser.py::
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
    settings = _read("web/modules/settings.js")
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
    assert 'id="s-total-budget"' in ui
    assert 'id="s-settings-per-task-cost"' in ui
    assert "setupContract.budgetFields" in settings
    # Model defaults come from the shared setup contract, not duplicate model
    # literals in Settings. Source qualification belongs to the shared editor.
    assert "setupContract.modelSlots" in settings
    assert "modelRoles.load(s" in settings
    assert "inputId: slot.settingsInputId" in settings
    model_roles = _read("web/modules/model_roles.js")
    assert "composeModelSource" in model_roles
    assert "route_editor_primitives.js" in model_roles
    assert "currentSettings?.[field.settingKey]" in settings
    assert "window.addEventListener('ouro:settings-updated'" in settings
    assert "source: 'settings'" in settings
    assert "Budget values must be at least 0.01." in costs
    assert "COST_BUDGET_INPUTS" in costs
    assert "s?._meta?.setup_contract?.budgetFields" in costs
    assert "const MIN_BUDGET_VALUE" not in costs
    assert "new CustomEvent('ouro:settings-updated'" in costs
    assert "source: 'costs'" in costs
    assert "event.detail?.source === 'costs'" in costs
