import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_settings_and_chat_expose_nano_context_mode():
    settings = (ROOT / 'web/modules/settings_ui.js').read_text(encoding="utf-8")
    # The context capsule (its three modes, its save and its /api/state read) lives with the
    # composer's other owner-setting control.
    controls = (ROOT / 'web/modules/composer_owner_controls.js').read_text(encoding="utf-8")
    assert "{ value: 'nano', label: 'Nano' }" in settings
    assert "const CONTEXT_LABELS = Object.freeze({ nano: 'Nano', low: 'Low', max: 'Max' });" in controls
    assert "const contextModeOf = (mode) => (['nano', 'low', 'max'].includes(mode) ? mode : 'max');" in controls
    assert "el.dataset.contextMode = contextModeOf(data.context_mode);" in controls


def test_chat_control_ids_match_instance_wiring_and_nano_has_active_style():
    # The composer's markup and wiring live in composer.js; chat.js mounts it and
    # namespaces its ids for a Project panel.
    chat = (ROOT / 'web/modules/chat.js').read_text(encoding="utf-8")
    composer = (ROOT / 'web/modules/composer.js').read_text(encoding="utf-8")
    controls = (ROOT / 'web/modules/composer_owner_controls.js').read_text(encoding="utf-8")
    styles = (ROOT / 'web/style.css').read_text(encoding="utf-8")
    # The markup keeps the chat-* ids that byId() resolves for the main
    # instance (and namespaces for project panels), so handlers cannot drift
    # to a selector for a different control. The Subagents and Context capsules
    # are siblings in the row, Subagents first.
    assert "${composerMarkup()}" in chat
    assert "composer = createComposer({ page, byId," in chat
    assert 'id="chat-subagents-mode"' in composer and 'id="chat-swarm"' not in composer
    assert re.search(r'</div>\s*<div class="chat-context-mode" id="chat-context-mode" data-context-mode="max"></div>', composer)
    assert "const subagentsEl = row && saveSubagentsMode ? byId('subagents-mode') : null;" in controls
    assert "const contextEl = byId('context-mode');" in controls
    assert "ownerControls = createComposerOwnerControls({ row: page.querySelector('.chat-toolbar-row'), box: page, byId, apiFetch, showToast," in composer
    assert '.chat-context-mode[data-context-mode="nano"] {' in styles
    assert 'justify-content: center;' in styles
