// The work card's one disclosure (DESIGN §5; owner 17.10, plan P6; refined A 21.10): "Details", its notes count kept
// in its name, is a real button and the only thing that opens the card; the summary band above it and the body below are reading content, which
// neither answers a click nor paints under the pointer (the half-card tint read as a seam on a whole object).
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createChatInstance } from '../modules/chat.js';
import { liveCardCountBits } from '../modules/task_phase_chip.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const ROOT = 'root-p6';

function fixture() {
    const env = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? { messages: [], window: { complete: true } } : { active_direct_turns: [] } }));
    const handlers = new Map();
    let generation = 0;
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
    });
    const messages = document.byId.get('chat-messages');
    return {
        card: (id) => walkCard(messages, id),
        emit: (row) => handlers.get('chat')({ chat_id: 1, role: 'system', is_progress: true, ts: '2026-10-17T18:40:00Z', ...row }),
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

test('the disclosure count starts at two notes and counts children independently', () => {
    assert.deepEqual(liveCardCountBits(0, 0), []);
    assert.deepEqual(liveCardCountBits(1, 0), []);
    assert.deepEqual(liveCardCountBits(2, 0), ['2 notes']);
    assert.deepEqual(liveCardCountBits(1, 1), ['1 child']);
    assert.deepEqual(liveCardCountBits(3, 2), ['3 notes', '2 children']);
});

test('the summary band is reading content; "Details" is the one disclosure, a real button', () => {
    const f = fixture();
    try {
        f.emit({ role: 'assistant', task_id: ROOT, content: 'Reading the third project', narration: true });
        const card = f.card(ROOT);
        const band = card.querySelector('[data-live-summary-button]');
        const disclosure = card.querySelector('[data-live-disclosure]');
        assert.ok(band, 'the band stays the card\'s timeline anchor');
        assert.equal(band.listeners.get('click'), undefined, 'the band answers no click');
        assert.equal(band.listeners.get('keydown'), undefined, 'nor a key');
        const bandTag = card.innerHTML.match(/<div class="chat-live-summary-button"[^>]*>/)[0];
        assert.doesNotMatch(bandTag, /role=|tabindex=|aria-/, 'no role, tab stop or state on reading content');
        assert.equal(disclosure?.tagName, 'BUTTON');
        const buttonTag = card.innerHTML.match(/<button[^>]*data-live-disclosure[^>]*>/)[0];
        assert.match(buttonTag, /type="button"/);
        assert.match(buttonTag, /aria-controls="chat-live-timeline-root-p6"/, 'it controls the timeline it opens');
        assert.equal(disclosure.getAttribute('aria-expanded'), 'false', 'it states the card\'s state from birth');
        disclosure.click();
        assert.equal(card.dataset.expanded, '1');
        assert.equal(disclosure.getAttribute('aria-expanded'), 'true');
        // The word stays; aria-expanded (and the turned chevron) say open or closed.
        assert.equal(card.querySelector('[data-live-toggle]').textContent, 'Details');
        disclosure.click();
        assert.equal(card.dataset.expanded, '0');
        assert.equal(disclosure.getAttribute('aria-expanded'), 'false');
        assert.equal(card.querySelector('[data-live-toggle]').textContent, 'Details');
    } finally { f.close(); }
});

test('nothing on the work card paints under the pointer but its disclosure, and only for a hovering pointer', () => {
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
    assert.doesNotMatch(css, /\.chat-live-summary-button[^{}]*:hover/, 'the summary band never tints (the seam)');
    // Every hover rule of the disclosure sits inside `@media (hover: hover)`: a tap must not leave a tint behind.
    const hovers = [...css.matchAll(/\.chat-live-summary-side:hover/g)];
    assert.ok(hovers.length > 0, 'the disclosure has its own quiet hover');
    for (const { index } of hovers) {
        const before = css.slice(0, index);
        const media = before.lastIndexOf('@media');
        assert.ok(media >= 0 && /^@media \(hover: hover\)/.test(before.slice(media))
            && !/\n\}/.test(before.slice(media)), 'the disclosure hover is gated by (hover: hover)');
    }
    // A finger gets the shared touch minimum around it, without a taller row: the one ui.css recipe.
    const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    assert.match(chat, /class="chat-live-summary-side ui-hit-area" data-live-disclosure/);
    assert.doesNotMatch(css, /\.chat-live-summary-side::after/, 'no second copy of the hit-area recipe');
});
