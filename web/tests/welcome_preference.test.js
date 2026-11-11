import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_WELCOME_TEXT, mountEmptyChatWelcome, welcomeText } from '../modules/welcome_preference.js';
import { HAND_WRITTEN, HOME_ROWS, ROWS } from '../modules/settings_rows.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

// Every registered Settings control, including the pages drawn by hand.
const settingsSections = () => [JSON.stringify(ROWS), JSON.stringify(HAND_WRITTEN)].join('');

test('welcome mode resolves exactly without treating custom text as markup', () => {
    assert.equal(welcomeText({ mode: 'default', text: 'ignored' }), DEFAULT_WELCOME_TEXT);
    assert.equal(welcomeText({ mode: 'hidden', text: 'saved' }), null);
    assert.equal(welcomeText({ mode: 'custom', text: '<b>Привет</b>\nhello' }), '<b>Привет</b>\nhello');
    assert.equal(welcomeText({ mode: 'custom', text: '   ' }), null);
    assert.equal(welcomeText(null), null);
});

test('the welcome preference is hidden: Settings renders no control for it', () => {
    const html = settingsSections() + JSON.stringify(HOME_ROWS);
    assert.doesNotMatch(html, /data-welcome|welcome-mode|welcome-text/i);
    assert.doesNotMatch(html, /greeting/i);
});

test('the Main empty state needs a complete successful read and an empty feed', () => {
    const { prior, mount } = installDom();
    const previousObserver = globalThis.MutationObserver;
    let mutated = null;
    globalThis.MutationObserver = class {
        constructor(callback) { mutated = callback; }
        observe() {}
        disconnect() { mutated = null; }
    };
    const doc = globalThis.document;
    const node = (className, dataset = {}) => {
        const element = doc.createElement('div');
        element.className = className;
        Object.assign(element.dataset, dataset);
        return element;
    };
    try {
        const messages = node('');
        mount.appendChild(messages);
        const typing = node('chat-bubble assistant typing-bubble');
        messages.appendChild(typing);
        const welcome = mountEmptyChatWelcome(messages);
        const shown = () => messages.children.find((child) => child.classList.contains('chat-empty-welcome'));
        welcome.historyRead(true);
        assert.equal(shown(), undefined, 'no preference read yet: a hidden choice is never overridden');
        welcome.historyRead(false);
        welcome.setPreference({ mode: 'default', text: '' });
        assert.equal(shown(), undefined, 'a partial or failed read confirms nothing');
        welcome.historyRead(true);
        assert.equal(shown().dataset.welcomeState, 'ready');
        assert.equal(shown().lastElementChild.textContent, DEFAULT_WELCOME_TEXT);
        assert.ok(messages.children.indexOf(shown()) < messages.children.indexOf(typing));
        welcome.setPreference({ mode: 'custom', text: '<b>x</b>\nnext' });
        assert.equal(shown().lastElementChild.innerHTML, '&lt;b&gt;x&lt;/b&gt;\nnext', 'owner copy stays text');
        messages.appendChild(node('chat-bubble system', { ephemeral: '1' })); mutated();
        assert.ok(shown(), 'the reconnect notice is chrome, not conversation');
        welcome.historyPending();
        assert.equal(shown(), undefined, 'a read in flight vouches for nothing, over a reconnect notice too');
        welcome.setPreference({ mode: 'custom', text: '<b>x</b>\nnext' });
        assert.equal(shown(), undefined, 'a preference read does not stand in for the history read');
        welcome.historyRead(true);
        assert.ok(shown());
        const card = node('chat-live-card');
        messages.appendChild(card); mutated();
        assert.equal(shown(), undefined, 'a visible task card is content');
        card.remove(); mutated();
        assert.ok(shown());
        const bubble = node('chat-bubble user');
        messages.appendChild(bubble); mutated();
        assert.equal(shown(), undefined, 'a late message removes the empty state');
        bubble.remove(); welcome.historyRead(false);
        assert.equal(shown(), undefined, 'a later failed read retracts it');
        welcome.historyRead(true);
        welcome.setPreference({ mode: 'hidden', text: 'kept' });
        assert.equal(shown(), undefined);
        welcome.dispose();
        assert.equal(mutated, null);
    } finally {
        globalThis.MutationObserver = previousObserver;
        restoreDom(prior);
    }
});
