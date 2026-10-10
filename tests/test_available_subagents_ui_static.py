"""Focused build-free guards for the Phase 1C Available subagents UI."""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULES = ROOT / "web" / "modules"


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def test_available_subagents_is_one_canonical_settings_editor() -> None:
    editor = _read(MODULES / "subagents_settings.js")
    host = _read(MODULES / "settings.js")
    assert "<h3>Available subagents</h3>" in editor
    assert "OUROBOROS_SUBAGENTS" in editor
    assert "collectSubagentsSettings" in host
    assert "OUROBOROS_SUBAGENT_HARNESS" not in editor
    assert "OUROBOROS_SUBAGENT_PROFILE" not in editor
    assert "MAX_AVAILABLE_SUBAGENTS = 26" in editor
    assert 'class="available-subagent-heading"' in editor
    assert 'Subagent ${ordinal}' in editor
    assert 'data-subagent-field="recommended_use"' in editor
    assert 'data-subagent-field="id"' not in editor
    assert 'data-subagent-field="name"' not in editor
    for action in ("data-subagent-add", "data-subagent-duplicate", "data-subagent-remove"):
        assert action in editor


def test_every_list_editor_reveals_its_added_entry_through_the_shared_helper() -> None:
    """docs/DESIGN.md "List editors": a new entry is scrolled into view and takes
    the caret through ONE seam, `ui_helpers.revealNewRow` — a local
    scrollIntoView/focus pair in an add path is the class this pins closed
    (DEVELOPMENT.md § Design System). The class is every Settings list editor,
    not the panel the owner happened to report."""
    helper = _read(MODULES / "ui_helpers.js")
    assert "export function revealNewRow(row, field)" in helper
    assert "scrollIntoView?.({ block: 'nearest' })" in helper
    assert "focus?.({ preventScroll: true })" in helper
    for name in ("subagents_settings.js", "mcp_settings.js", "settings.js"):
        source = _read(MODULES / name)
        assert re.search(r"import \{[^}]*\brevealNewRow\b[^}]*\} from './ui_helpers\.js'", source), name
        assert "revealNewRow(" in source, f"{name} never calls the shared reveal"
    for name in ("subagents_settings.js", "mcp_settings.js"):
        assert "scrollIntoView" not in _read(MODULES / name), f"{name} rolls its own reveal"


def test_a_fresh_subagent_row_invites_and_only_a_save_attempt_makes_it_red() -> None:
    """docs/DESIGN.md "List editors": the section-level line and the row-local
    tint appear only after the owner tried to save; Save and Finish say so
    through `noteSaveAttempt`, which judges the rows that existed then (a row
    added afterwards is fresh again), and `validate()` stays pure."""
    editor = _read(MODULES / "subagents_settings.js")
    primitives = _read(MODULES / "subagent_status_primitives.js")
    assert "noteSaveAttempt" in editor
    assert "validate: validationErrors," in editor
    assert "row._uiAttempted = true" in editor
    assert "Boolean(row._uiAttempted) && " in editor
    assert "row._uiAttempted && errors.length" in primitives
    assert "Choose how this subagent runs: an API model or an agent session." in primitives
    host = _read(MODULES / "settings.js")
    # Every Save click is an attempt — including one another field's validation
    # then aborts — so the stamp precedes the cadence check's early return.
    save = host[host.index("byId('btn-save-settings').addEventListener"):]
    assert save.index("noteSubagentsSaveAttempt();") < save.index("const { messages: errors, subject } = renderValidation();")
    assert "Every-N cadence needs" in host
    assert "agentsStep?.noteSaveAttempt?.();" in _read(MODULES / "onboarding_wizard.js")
    # Errors name the card the way its heading does, never a bare "Row N".
    assert "`Subagent ${index + 1} ${text}`" in editor
    assert "`Row ${index + 1}`" not in editor


def test_the_review_pool_is_a_catalog_mark_and_the_route_primitives_stay_reviewer_free() -> None:
    """The lane editor is gone: a reviewer is a catalog row marked Reviewer, owned by
    the catalog editor, and the shared route primitives know nothing of review."""
    primitive = _read(MODULES / "route_editor_primitives.js")
    editor = _read(MODULES / "subagents_settings.js")
    assert not (MODULES / "reviewer_slots.js").exists()
    assert not (ROOT / "web" / "reviewer_slots.css").exists()
    assert "route_editor_primitives.js" in editor
    assert "credentialField: 'credential_profile_id'" in editor
    assert 'data-subagent-field="review_eligible"' in editor
    for lane_term in ("OUROBOROS_REVIEWER_SLOTS", "buildReviewerSlotsSetting", "review_eligible"):
        assert lane_term not in primitive, lane_term
        assert lane_term == "review_eligible" or lane_term not in editor, lane_term


def test_heavy_card_is_gone_but_provider_test_contract_and_controls_remain() -> None:
    ui = _read(MODULES / "settings_ui.js")
    host = _read(MODULES / "settings.js")
    setup = _read(ROOT / "ouroboros" / "settings_setup_contract.py")
    assert "['Heavy'," not in ui
    assert "s-model-heavy" not in ui
    assert "OUROBOROS_MODEL_HEAVY" not in host
    light_copy = "Fast summaries, lightweight internal work, reflections, and the default Fast scout. Empty uses Main."
    # Role labels and copy come from the shared setup contract, not a second
    # hand-maintained card table in Settings.
    assert "modelRolesHost('settings-model-roles')" in ui
    assert "modelRoles.load(s," in host
    assert "slot.note" in _read(MODULES / "model_roles.js")
    assert light_copy in setup
    assert "all deep subagents" not in ui
    assert "all deep subagents" not in setup
    assert "data-provider-test" in ui
    assert "PROVIDER_TEST_INPUTS" in ui
    assert "apiClient.providerTest({ provider_id: provider, overrides })" in host
    assert "providerTestResultIsCurrent" in host


def test_onboarding_previews_and_commits_the_visible_owner_draft() -> None:
    client = _read(MODULES / "api_client.js")
    step = _read(MODULES / "onboarding_agents_step.js")
    wizard = _read(MODULES / "onboarding_wizard.js")
    assert "'/api/onboarding/subagents/preview'" in client
    assert "response?.available_subagents" in _read(MODULES / "subagents_settings.js")
    assert "refreshSubagentsPreview" in step
    assert "OUROBOROS_SUBAGENTS: catalogDraft()," in wizard
    assert "function catalogDraft() { return agentsStep?.availableSubagents || state.availableSubagents; }" in wizard
    assert "Heavy', trim(state.heavyModel)" not in wizard
    assert "OUROBOROS_MODEL_HEAVY" not in wizard


def test_preview_contract_and_task_only_agy_copy_are_explicit() -> None:
    types = _read(MODULES / "api_types.js")
    step = _read(MODULES / "onboarding_agents_step.js")
    assert "@typedef {Object} OnboardingSubagentsPreviewResponse" in types
    assert "@property {boolean} ok" in types
    assert "@property {AvailableSubagentsSetting} available_subagents" in types
    assert "@property {Object[]} diagnostics" in types
    assert "{ harness: 'agy' }" in step
    assert "familyLabel(family.harness" in step
    assert "{ harness: 'agy', label:" not in step
    assert "task-only and does" in step
    assert "@typedef {Object} SubagentLastDelegation" in types
    assert "@property {SubagentLastDelegation=} subagent_last_delegation" in types


def test_generated_preview_is_background_and_whole_draft_clean_gated() -> None:
    editor = _read(MODULES / "subagents_settings.js")
    host = _read(MODULES / "settings.js")
    wizard = _read(MODULES / "onboarding_wizard.js")
    assert "void maybeRefreshGeneratedPreview({ force: true });" in editor
    assert "isOuterDraftClean: () => !settingsDirty" in host
    assert "if (settingsLoaded && !settingsDirty) setSettingsCleanBaseline();" in host
    assert "void agentsStep.refreshSubagentsPreview({ force: true });" in wizard
    assert "agentsStep?.invalidateGeneratedPreview();" in wizard


def test_status_refresh_and_active_task_copy_keep_the_frozen_semantics() -> None:
    editor = _read(MODULES / "subagents_settings.js")
    host = _read(MODULES / "settings.js")
    assert "await boundedStatusRefresh(store);" in editor
    assert "Saved rows remain unchanged" in editor
    assert "take effect for new child tasks" in host
    assert "the current task keeps its existing routes" in host


def test_new_frontend_modules_stay_within_the_context_target() -> None:
    from ouroboros.size_ratchet_manifest import BAND_PATHS

    for name in (
        "route_editor_primitives.js", "subagent_status_primitives.js", "subagents_settings.js",
    ):
        lines = _read(MODULES / name).count("\n") + 1
        # Past the target only as a 1001-1500 band entry whose reason the manifest records.
        assert lines <= 1000 or BAND_PATHS.get(f"web/modules/{name}"), f"{name} grew to {lines} lines"


def test_effort_choice_mirrors_track_the_python_scale() -> None:
    """The JS effort lists are hand-maintained mirrors of config.EFFORT_SCALE;
    this guard makes the next tier addition fail loudly when a mirror is missed."""
    import re

    from ouroboros.config import EFFORT_SCALE

    primitives = _read(MODULES / "route_editor_primitives.js")
    expected = "export const EFFORT_CHOICES = [" + ", ".join(f"'{tier}'" for tier in EFFORT_SCALE) + "];"
    assert expected in primitives

    # The web's effort vocabulary (the composer's range, the card chip, the Behavior line).
    levels = _read(MODULES / "effort_levels.js")
    assert "export const EFFORT_SCALE = [" + ", ".join(f"'{tier}'" for tier in EFFORT_SCALE) + "];" in levels
    block = re.search(r"export const EFFORT_OPTIONS = \[(.*?)\];", levels, re.DOTALL)
    assert block, "EFFORT_OPTIONS block not found in effort_levels.js"
    values = re.findall(r"value: '([a-z]+)'", block.group(1))
    # `minimal` is deliberately not an owner-facing standing level (see EFFORT_OPTIONS).
    assert values == [tier for tier in EFFORT_SCALE if tier != "minimal"]
    labels = re.search(r"export const EFFORT_LABELS = Object\.freeze\(\{(.*?)\}\);", levels, re.DOTALL)
    assert labels, "EFFORT_LABELS block not found in effort_levels.js"
    assert re.findall(r"(\w+): '", labels.group(1)) == list(EFFORT_SCALE), "every runtime tier has a label"
    # The composer's Reset target and its pre-/api/state value are the server's shipped range.
    from ouroboros.settings_defaults import SETTINGS_DEFAULTS
    shipped = {"min": "MIN", "recommended": "TASK", "max": "MAX"}
    assert ("export const EFFORT_RANGE_DEFAULT = Object.freeze({ " + ", ".join(
        f"{key}: '{SETTINGS_DEFAULTS['OUROBOROS_EFFORT_' + name]}'" for key, name in shipped.items()) + " });") in levels


def test_every_status_tone_the_card_emits_has_a_shared_rule_in_both_documents() -> None:
    # The emitted tone resolves in the one shared source actually loaded by both hosts.
    assert '.settings-inline-status[data-tone="neutral"]' in _read(ROOT / "web" / "ui.css")
    for document in ("index.html", "onboarding_template.html"):
        assert 'href="/static/ui.css"' in _read(ROOT / "web" / document), document


def test_the_per_row_switch_is_one_shared_checkbox_saved_by_the_common_save() -> None:
    """docs/DESIGN.md §3 "Controls and editable choices" + "List editors": the
    per-row owner switch is the shared `.ui-checkbox` primitive with its own
    accessible name and hit target, it leads the card head BEFORE the title, and
    it is held as a draft that the section's ONE Save writes — no instant-save
    request, and no dimming of the card it switches off."""
    editor = _read(MODULES / "subagents_settings.js")
    head = editor[editor.index('class="available-subagent-head"'):
                  editor.index("available-subagent-purpose")]
    assert head.index('data-subagent-field="enabled"') < head.index("available-subagent-heading")
    assert 'class="available-subagent-enable"' in head
    assert 'class="ui-checkbox" type="checkbox" data-subagent-field="enabled"' in head
    assert 'aria-label="Subagent ${ordinal} enabled for new work"' in head
    # One writer: the editor never posts by itself, and `false` is the only
    # value it stores (an enabled row keeps its existing canonical bytes).
    assert "apiClient" not in editor and "fetch(" not in editor
    assert "row?.enabled === false ? { enabled: false } : {}" in editor
    for shell in ("settings.css", "onboarding.css"):
        sheet = _read(ROOT / "web" / shell)
        assert ".available-subagent-enable {" in sheet, shell
        assert "opacity" not in sheet[sheet.index(".available-subagent-enable {"):
                                      sheet.index(".available-subagent-enable {") + 240], shell
