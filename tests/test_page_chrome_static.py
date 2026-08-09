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


def test_shared_page_header_helper_has_no_inline_styles():
    source = _read("web/modules/page_header.js")

    assert "export function renderPageHeader" in source
    assert "export function renderTabStrip" in source
    assert "style=" not in source
    assert "app-page-header" in source
    assert "app-tab-strip" in source


def _css_rule_body(css: str, selector_list: str) -> str:
    """Body of the first rule whose selector list matches, whitespace-tolerantly.

    The old version split on a literal ``".app-tab.active,\\n.app-tab.is-active {"``,
    so reformatting the selector list broke the test instead of the design.
    """
    pattern = r"\s*,\s*".join(re.escape(part.strip()) for part in selector_list.split(","))
    match = re.search(pattern + r"\s*\{([^}]*)\}", css)
    assert match, f"no rule found for selector list {selector_list!r}"
    return match.group(1)


def test_page_tabs_are_underline_tabs_not_pills():
    """Design pin (flat redesign, Phase D): the ONE shared tab strip is an
    underline strip — a flat label row over a divider with a 2px accent rule
    under the active tab — and no page stylesheet re-declares tab shape.
    Dashboard, Skills and Settings are its only consumers; Skills passes
    activeClass 'is-active', so both selectors carry the marker.

    What is pinned is the underline GRAMMAR: the divider under the strip, the
    accent rule under the active tab, and the absence of pill chrome. The
    transparent background is inherited from the base `.app-tab` rule, so the
    active rule must NOT be required to restate it — a redundant declaration is
    not a design contract, and pinning it forbade the obvious cleanup.
    """
    css = _read("web/style.css")
    settings_css = _read("web/settings.css")

    base_rule = _css_rule_body(css, ".app-tab")
    assert "background: transparent;" in base_rule
    assert "border-bottom: 2px solid transparent;" in base_rule
    assert "border-radius: 0;" in base_rule

    active_rule = _css_rule_body(css, ".app-tab.active, .app-tab.is-active")
    assert "border-bottom-color: var(--accent);" in active_rule
    # No pill chrome creeps back in through the active state.
    assert "border-radius" not in active_rule
    assert "background:" not in active_rule

    strip = _css_rule_body(css, ".app-tab-strip")
    assert "border-bottom: 1px solid var(--divider);" in strip

    # Settings really is a consumer of the shared control, so the negative sweep
    # below is about something that exists.
    settings_ui = _read("web/modules/settings_ui.js")
    assert "stripClass: 'settings-tabs'" in settings_ui
    assert "tabClass: 'settings-tab'" in settings_ui

    # No per-page tab SHAPE: the settings stylesheet may add its own class names to
    # the shared strip, but must not re-declare the control's geometry. The sweep is
    # scoped to the settings TAB rules, so an unrelated 999px pill elsewhere in the
    # file cannot fail this pin — and cannot pass it by being renamed either.
    # Comments are stripped first (prose naming the classes is not a rule), and
    # `(?![\w-])` keeps `.settings-tabs-bar` — the page-level wrapper DIV, not the
    # strip — out of the sweep.
    settings_rules = re.sub(r"/\*.*?\*/", "", settings_css, flags=re.DOTALL)
    tab_blocks = re.findall(
        r"\.settings-tabs?(?![\w-])[^{};]*\{([^}]*)\}", settings_rules
    )
    for block in tab_blocks:
        for forbidden in ("border-radius", "font-size", "padding", "border-bottom"):
            assert forbidden not in block, (
                f"settings.css re-declares shared tab {forbidden}: {block!r}"
            )


def test_primary_pages_use_shared_header_helper():
    for rel in [
        "web/modules/settings_ui.js",
        "web/modules/dashboard.js",
        "web/modules/skills.js",
        "web/modules/widgets.js",
        "web/modules/files.js",
        "web/modules/chat.js",
    ]:
        source = _read(rel)
        assert "page_header.js" in source
        assert "renderPageHeader" in source


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


def test_settings_scope_review_effort_round_trips():
    """6.3 moved the Review/Scope efforts off the Behavior tab onto the Models
    page as PER-SLOT dropdowns (red on cxi/p6-ui-v2's own head — the branch
    moved the carrier and left this pin behind): the owner-facing carrier is now
    reviewer_slots.js, where an EMPTY slot effort inherits the surface default
    (OUROBOROS_EFFORT_SCOPE_REVIEW backend-side) and the advisory row defaults
    low (D14). The mode-guard filter in settings.js is unchanged."""
    slots_ui = _read("web/modules/reviewer_slots.js")
    settings = _read("web/modules/settings.js")
    assert "scope review effort" in slots_ui   # per-slot surface-default wording
    assert "review effort" in slots_ui
    assert "effort: 'low'" in slots_ui          # the advisory default (D14)
    assert "key !== 'OUROBOROS_RUNTIME_MODE' && key !== 'OUROBOROS_CONTEXT_MODE'" in settings


def test_settings_mutative_subagents_toggle_round_trips():
    """The owner-facing master switch for mutative ("acting") subagents must exist
    as an owner-facing Off/Auto/On control wired through the
    settings load/save path, not only as a settings.json/env key.

    The CARRIER moved: the write permission now sits inside the Subagents section
    (Models tab, beside Reviewer Slots) together with the delegation route, so one
    subagent story is told in one place. Two controls over one setting would have
    carried two drafts, so settings_ui.js must NOT keep a second copy."""
    ui = _read("web/modules/subagents_settings.js")
    settings = _read("web/modules/settings.js")
    assert "s-allow-mutative-subagents" not in _read("web/modules/settings_ui.js")
    assert "s-allow-mutative-subagents" in ui
    assert "renderSubagentsSection" in _read("web/modules/settings_ui.js")
    section_start = ui.index("<h3>Subagents</h3>")
    section = ui[section_start:]
    # Segmented control is generated by the renderSegmentedField SSOT (C7.1): the
    # options live in its `options` array, not inline button markup. The control
    # is tri-state again (SC-4): the v6.22.1 binary Off/On display was truthful
    # only while the unset default was all-or-nothing per mode; the surface-aware
    # light default (unset = external_workspace/genesis ON, self_worktree off)
    # made "Off" a false claim for unset-in-light, so unset there displays as
    # an explicit "Auto" state instead.
    assert "renderSegmentedField" in section
    assert "target: 's-allow-mutative-subagents'" in section
    assert "value: 'auto'" in section
    assert "value: 'off'" in section
    assert "value: 'on'" in section
    # The note must describe the surface-aware light default, not the pre-SC-4
    # "off in Light" claim.
    assert "off in Light" not in section
    assert "OUTSIDE the Ouroboros runtime" in section
    # load + save mapping wired in settings.js (outside VALUE_FIELDS): unset in
    # light displays Auto (never a false "Off"); unset in advanced/pro keeps the
    # truthful effective "On"; an untouched control preserves the empty value.
    assert "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS" in settings
    assert "s-allow-mutative-subagents" in settings
    assert "runtimeMode === 'light' ? 'auto' : 'on'" in settings
    assert "dataset.rawValue = rawMutative" in settings
    assert "mutativeTouched" in settings


def test_onboarding_compact_access_step_keeps_default_width_two_column():
    css = _read("web/onboarding.css")
    assert "@media (max-height: 820px), (max-width: 900px)" in css
    assert ".field-note,\n  .footer-copy" in css
    assert "@media (max-width: 760px)" in css


def test_settings_more_providers_collapse_keeps_inputs_mounted():
    """Rarely used provider cards (Cloud.ru, MiniMax, GigaChat) collapse under a
    "More providers" details wrapper, but their inputs must stay mounted:
    settings.js applyInputValue has no null guard, so a missing input id
    breaks settings load. The wrapper auto-opens when configured."""
    ui = _read("web/modules/settings_ui.js")
    settings = _read("web/modules/settings.js")
    css = _read("web/settings.css")

    assert 'id="settings-more-providers"' in ui
    assert ui.count("advanced: true") == 3
    assert "PROVIDER_CARDS.filter((card) => !card.advanced)" in ui
    assert "PROVIDER_CARDS.filter((card) => card.advanced)" in ui
    assert "syncMoreProvidersDisclosure" in settings
    assert ".settings-more-providers > summary" in css
    # About tab footer stays removed (v6.82 cosmetic pass).
    assert "Joi Lab" not in ui


def test_subagent_write_surface_badge_in_both_card_paths():
    """The write=<surface> badge must render on BOTH the Logs path
    (summarizeLogEvent) and the Chat live-card path (summarizeChatLiveEvent),
    not just one — the surface field is plumbed through history/contracts."""
    log_events = _read("web/modules/log_events.js")
    # both summarize functions reference the badge
    assert log_events.count("write=${evt.write_surface}") >= 2
    api_types = _read("web/modules/api_types.js")
    assert "write_surface" in api_types


def test_skills_and_widgets_use_inner_scroll_regions():
    skills = _read("web/modules/skills.js")
    widgets = _read("web/modules/widgets.js")
    css = _read("web/style.css")
    assert 'class="skills-scroll scroll-fade-y"' in skills
    assert 'class="widgets-scroll scroll-fade-y"' in widgets
    assert ".skills-scroll" in css and "overflow-y: auto" in css
    assert ".widgets-scroll" in css and "overflow-y: auto" in css


# ---------------------------------------------------------------------------
# Evolution / consciousness UI wiring
# ---------------------------------------------------------------------------


def test_evolution_page_supports_refresh_and_runtime_state():
    source = _read("web/modules/evolution.js")

    assert 'id="evo-refresh"' in source
    assert "Runtime Status" in source
    assert "apiFetch(`/api/evolution-data${suffix}`" in source
    assert "ws.on('open', () => {" in source
    assert "window.addEventListener('ouro:page-shown'" in source
    assert "document.addEventListener('visibilitychange'" in source
    assert "renderRuntimeState(runtime, data.generated_at || '');" in source
    assert "evolution_state" in source
    assert "bg_consciousness_state" in source


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

    assert "def _describe_bg_consciousness_state(requested_enabled: bool) -> dict:" in server_source
    assert '"evolution_state": evolution_state,' in state_source
    assert '"bg_consciousness_state": bg_state,' in state_source
    assert 'request.query_params.get("force")' in control_source
    assert "window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: pageName } }));" in app_source
    assert "evo-runtime-detail" in evo_source
    assert "data?.evolution_state?.detail" in chat_source
    assert "data?.bg_consciousness_state?.detail" in chat_source
    assert re.search(r'<input[^>]+id="chat-file-input"[^>]+multiple', chat_source)
    assert "MAX_PENDING_ATTACHMENTS = 10" in chat_source
    assert "MAX_ATTACHMENT_FILE_BYTES = 50 * 1024 * 1024" in chat_source
    assert "MAX_PENDING_ATTACHMENT_BYTES = 100 * 1024 * 1024" in chat_source
    assert "pendingAttachments" in chat_source
    assert "attachmentsUploading" in chat_source
    assert "setAttachmentUploadState" in chat_source
    assert "attachBtn.classList.toggle('uploading', uploading)" in chat_source
    assert "input.disabled = uploading;" in chat_source
    assert "cleanupUploadedAttachments" in chat_source
    assert "method: 'DELETE'" in chat_source
    assert "await cleanupUploadedAttachments(uploaded);" in chat_source
    assert "await cleanupUploadedAttachments(uploadedAttachments);" in chat_source
    assert "ws.send({" in chat_source and "{ queue: false }" in chat_source
    assert "result?.status !== 'sent'" in chat_source
    assert "data-attachment-remove" in chat_source
    assert "Promise.allSettled" in chat_source
    # The budget meter moved to the sidebar bottom, so its pre-first-read label
    # is pinned in the shell markup now (chat.js no longer renders a pill).
    assert '>Loading…</span>' in _read("web/index.html")
    assert "syncHeaderControlState({ accounting: { available: false } });" in chat_source
    assert "budget_text: 'Connecting...'" not in chat_source
    assert "send(msg, options = {})" in _read("web/modules/ws.js")
    assert "options.queue === false" in _read("web/modules/ws.js")
    assert ".chat-attachment-preview" in css and "flex-wrap" in css
    assert 'id="s-total-budget"' in ui
    assert 'id="s-settings-per-task-cost"' in ui
    assert "setupContract.budgetFields" in settings
    assert "'anthropic/claude-sonnet-5'" in settings
    assert "'anthropic::claude-sonnet-5'" in settings
    assert "currentSettings?.[field.settingKey]" in settings
    assert "window.addEventListener('ouro:settings-updated'" in settings
    assert "source: 'settings'" in settings
    # Read-only consumers (e.g. the composer model chip) read the already-fetched
    # snapshot through one getter and refresh on the event, which now fires after
    # BOTH load and save — no second /api/settings fetch, no new /api/state field.
    assert "export function getSettingsSnapshot()" in settings
    assert "lastSettingsSnapshot = data;" in settings
    assert "reason: 'settings loaded'" in settings
    assert "reason: 'settings saved'" in settings
    assert "Budget values must be at least 0.01." in costs
    assert "COST_BUDGET_INPUTS" in costs
    assert "s?._meta?.setup_contract?.budgetFields" in costs
    assert "const MIN_BUDGET_VALUE" not in costs
    assert "new CustomEvent('ouro:settings-updated'" in costs
    assert "source: 'costs'" in costs
    assert "event.detail?.source === 'costs'" in costs
