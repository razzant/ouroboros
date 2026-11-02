// Built-in chrome carries no decorative emoji (DESIGN "Chat authorship and System rows"):
// labels and banners are words, a pictorial control is a monochrome stroke vector whose
// accessible name follows its state, and an applied owner hurry is a named element, not
// a CSS-only glyph. Authored content is outside this guard.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { createChatInstance } from '../modules/chat.js';
import { restartNoticeText, senderLabel } from '../modules/chat_activity.js';
import { budgetSavePresentation } from '../modules/costs.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';
import { fixture as mediaFixture } from './helpers/media_dom.js';

const web = new URL('../', import.meta.url);
const read = (path) => readFileSync(new URL(path, web), 'utf8');

test('sender labels and the restart notice are words; any other reload says nothing', () => {
    assert.equal(senderLabel('system', false, 'startup_notice'), 'System');
    assert.equal(senderLabel('system', false, 'task_summary'), 'Task Summary');
    assert.equal(senderLabel('system', false, 'skill_review'), 'Skill Review');
    assert.equal(senderLabel('assistant', true), 'Thought');
    assert.equal(restartNoticeText('sha-change'), 'Restart complete');
    for (const reason of ['sha-unknown', 'ui-source', 'socket-disconnect', 'ws-drop', '']) {
        assert.equal(restartNoticeText(reason), '', reason);
    }
});

test('fixed warning decoration is gone while the tone and words stay', () => {
    // Skip review keeps its warning as a stroke glyph beside the words, never an emoji.
    const skills = read('modules/skill_presentation.js');
    assert.match(skills, /class="btn btn-default btn-sm skills-skip-btn"/);
    assert.match(skills, /<span class="skills-skip-glyph" aria-hidden="true">\$\{WARNING_GLYPH\}<\/span>\$\{escapeHtml\(label\)\}<\/button>/);
    assert.match(skills, /export const WARNING_GLYPH = strokeIcon\(WARNING_PATH, 14\);/);
    assert.doesNotMatch(skills, /\u26a0/);
    const warningState = /statusMsg \+= '\. Warnings: ' \+ data\.warnings\.join\(' \| '\);\r?\n\s*statusType = 'warn';/;
    for (const eol of ['\n', '\r\n']) {
        const source = read('modules/settings.js').replace(/\r?\n/g, eol);
        assert.match(source, warningState, `warning words and tone survive ${JSON.stringify(eol)} checkout`);
        const opening = source.match(warningState)[0];
        assert.doesNotMatch(opening.replace("statusType = 'warn'", "statusType = 'ok'"), warningState);
        assert.doesNotMatch(opening.replace('Warnings: ', ''), warningState);
    }
    // Costs' save result says its warnings in words, in the warning tone, and no decoration is prefixed to either.
    const warned = budgetSavePresentation({ immediate_changed: true, warnings: ['Limit is below known spend.', 'Cap lowered.'] });
    assert.equal(warned.text, 'Saved. Took effect immediately. Warnings: Limit is below known spend. | Cap lowered.');
    assert.equal(warned.tone, 'warn');
    assert.doesNotMatch(warned.text, /\p{Extended_Pictographic}/u);
    assert.match(read('style.css'), /\.chat-delegated-problem \{\s*color: var\(--status-warn-fg\);/);
});

test('an applied owner hurry is a visible, named mark on its card', () => {
    const env = installDom();
    const handlers = new Map();
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
    });
    try {
        const ts = '2026-10-10T12:00:00Z';
        handlers.get('chat')({ chat_id: 1, ts, task_id: 'h1', role: 'assistant', is_progress: true,
            text: '💬 Reading the diff', content: '💬 Reading the diff' });
        const card = walkCard(document.byId.get('chat-messages'), 'h1');
        assert.ok(card, 'the task has a live card');
        const mark = card.querySelector('[data-live-hurry]');
        assert.ok(mark, 'the card carries the hurry slot');
        handlers.get('log')({ chat_id: 1, data: { type: 'owner_hurry', task_id: 'h1', phase: 'requested', is_progress: false, ts } });
        assert.equal(card.getAttribute('data-owner-hurry'), '', 'a requested hurry is not an applied one');
        handlers.get('log')({ chat_id: 1, data: { type: 'owner_hurry', task_id: 'h1', phase: 'applied', is_progress: false, ts } });
        assert.equal(card.getAttribute('data-owner-hurry'), '1');
        assert.equal(mark.hidden, false);
        assert.equal(mark.getAttribute('aria-label'), 'Owner hurry applied');
        assert.match(mark.innerHTML, /^<svg [^>]*stroke="currentColor"[^>]*><polygon points="13 19 22 12 13 5 13 19">/);
    } finally {
        instance.destroy();
        restoreDom(env.prior);
    }
    assert.match(read('modules/chat.js'), /<span class="chat-live-hurry" data-live-hurry role="img" hidden><\/span>/);
    assert.doesNotMatch(read('style.css'), /data-owner-hurry\][^{]*::after/);
});

test('player controls are stroke vectors whose icon and name follow mute and play state', async () => {
    const fx = mediaFixture();
    try {
        const bubble = fx.controller.buildMediaBubble({ type: 'video', role: 'assistant', video_base64: 'aGVsbG8=', mime: 'video/mp4' });
        const control = (action) => bubble.querySelector(`[data-media-action="${action}"]`);
        for (const action of ['play', 'mute', 'fullscreen']) {
            assert.ok(control(action).querySelector('svg'), `${action} draws a vector`);
        }
        const video = bubble.querySelector('video');
        await control('mute').click();
        assert.equal(video.muted, true);
        assert.equal(control('mute').attributes.get('aria-label'), 'Unmute');
        assert.match(control('mute').innerHTML, /<line x1="22" x2="16" y1="9" y2="15">/);
        await control('mute').click();
        assert.equal(video.muted, false);
        assert.equal(control('mute').attributes.get('aria-label'), 'Mute');
        assert.doesNotMatch(control('mute').innerHTML, /<line/);
        assert.match(control('mute').innerHTML, /M19\.364 18\.364/);
        await control('play').click();
        assert.equal(video.paused, false);
        assert.equal(control('play').attributes.get('aria-label'), 'Pause');
        assert.match(control('play').innerHTML, /<rect x="14"/);
    } finally {
        fx.controller.destroy();
        fx.restore();
    }
});
