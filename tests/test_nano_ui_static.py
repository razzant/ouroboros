from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_settings_and_chat_expose_nano_context_mode():
    settings = (ROOT / 'web/modules/settings_ui.js').read_text(encoding="utf-8")
    chat = (ROOT / 'web/modules/chat.js').read_text(encoding="utf-8")
    # The click handler lives with the composer's other owner-setting control.
    controls = (ROOT / 'web/modules/composer_owner_controls.js').read_text(encoding="utf-8")
    assert "{ value: 'nano', label: 'Nano' }" in settings
    assert 'data-mode="nano">Nano' in chat
    assert "['nano', 'low', 'max'].includes(data.context_mode)" in chat
    assert "['nano', 'low', 'max'].includes(seg.dataset.mode)" in controls


def test_chat_control_ids_match_instance_wiring_and_nano_has_active_style():
    chat = (ROOT / 'web/modules/chat.js').read_text(encoding="utf-8")
    controls = (ROOT / 'web/modules/composer_owner_controls.js').read_text(encoding="utf-8")
    styles = (ROOT / 'web/style.css').read_text(encoding="utf-8")
    # The markup keeps the chat-* ids that byId() resolves for the main
    # instance (and namespaces for project panels), so handlers cannot drift
    # to a selector for a different control.
    assert 'id="chat-swarm"' in chat
    assert 'id="chat-context-mode"' in chat
    assert "const swarmBtn = byId('swarm');" in chat
    assert "const contextModeBtn = byId('context-mode');" in controls
    assert "ownerControls = createComposerOwnerControls({ row: page.querySelector('.chat-toolbar-row'), byId, apiFetch, showToast," in chat
    assert '.chat-context-mode[data-context-mode="nano"] .chat-seg[data-mode="nano"]' in styles
    assert 'justify-content: center;' in styles
