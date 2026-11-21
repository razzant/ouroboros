import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { buildTimelineItemHtml, cardRowSummary } from '../modules/chat_activity.js';
import { applyPayload, tx } from '../modules/i18n.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';
import { PREVIEW_KEY } from '../modules/chat_preview.js';

// The owner's own press (ouroboros/pause_notices.notice_text); rows an earlier body wrote keep their bytes.
const ENGLISH = 'You paused this task. Its work is retained; Resume continues it.';
const TRANSLATED = 'Вы приостановили задачу. Работа сохранена; продолжение доступно по кнопке Resume.';
const LEGACY = "The task paused because of the owner's Pause. Its work is retained. Use Resume on the task when available.";
const LEGACY_TRANSLATED = 'Задача приостановлена владельцем. Работа сохранена. Продолжение доступно в карточке задачи.';
const notice = { role: 'system', system_type: 'task_pause_notice', task_id: 'paused-root',
    card_row: 'timeline', card_row_id: 'pause:paused-root:one', narration: false,
    text: ENGLISH, content: ENGLISH, ts: '2026-10-04T12:00:00Z', chat_id: 1 };
const legacy = { ...notice, card_row_id: 'pause:paused-root:old', text: LEGACY, content: LEGACY };
const russian = () => applyPayload({ language: 'ru', english: false, entries: {
    [ENGLISH]: { text: TRANSLATED, provenance: 'imported' },
    [LEGACY]: { text: LEGACY_TRANSLATED, provenance: 'imported' },
} });

test('typed System pause card rows translate through the installed memory without changing source bytes', () => {
    const env = installDom();
    try {
        russian();
        assert.equal(tx(ENGLISH), TRANSLATED, 'the exact sentence is in the translation memory');
        for (const [row, expected] of [[notice, TRANSLATED], [legacy, LEGACY_TRANSLATED],
            [{ ...notice, role: 'assistant' }, ENGLISH], [{ ...notice, system_type: 'other_notice' }, ENGLISH]]) {
            const original = JSON.stringify(row);
            const summary = cardRowSummary(row, 'warn', row.ts);
            const html = buildTimelineItemHtml({ ...summary, lineKey: 'pause-one' },
                { groupId: row.task_id, expandedLineKeys: new Set() });
            assert.equal(summary.headline, expected);
            assert.ok(html.includes(expected));
            assert.equal(JSON.stringify(row), original, 'neither host nor model evidence is rewritten');
        }
    } finally { applyPayload({ language: 'en', english: true }); restoreDom(env.prior); }
});

for (const history of [false, true]) test(`pause notice without a card translates ${history ? 'on history replay' : 'live'} but preserves model speech and stored English`, async () => {
    const source = { ...notice, history_id: 'chat:pause-one' }, original = JSON.stringify(source);
    const env = installDom(async url => ({ ok: true, json: async () => String(url).startsWith('/api/chat/history')
        ? { messages: history ? [source] : [], page_cursor: 'page:pause', window: { complete: true } } : { active_direct_turns: [] } }));
    let instance;
    try {
        russian();
        const handlers = new Map();
        let generation = 0;
        instance = createChatInstance({ ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
            isConnected: () => true, send() {} },
            state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
            chatId: 1, idPrefix: 'chat', mountEl: env.mount,
            stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
                gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
        });
        await instance.refreshHistory({ revision: 1 });
        if (!history) handlers.get('chat')(source);
        const feed = document.byId.get('chat-messages');
        const bubble = feed.children.find(node => node.classList.contains('system') && node.classList.contains('chat-bubble'));
        assert.ok(bubble, 'a missing card leaves the real System fallback');
        assert.ok(bubble.innerHTML.includes(TRANSLATED));
        assert.equal(bubble.innerHTML.includes(ENGLISH), false);
        assert.equal(feed.children.some(node => node.classList.contains('chat-live-card')), false);
        assert.equal(JSON.stringify(source), original, 'live and canonical input keep their English bytes');
        // The saved preview (chat_preview.js) holds the newest read's stored words, never a live frame or a translation.
        const saved = JSON.parse(localStorage.getItem(PREVIEW_KEY));
        const rows = saved.rooms.flatMap(room => room.rows);
        if (history) assert.equal(rows.find(row => row.id === source.history_id).text, ENGLISH);
        else assert.equal(rows.length, 0, 'a live-only notice does not enter the newest read\'s preview');
        assert.equal(JSON.stringify(saved).includes(TRANSLATED), false, 'the display translation never enters saved text');
        handlers.get('chat')({ ...notice, role: 'assistant', task_id: 'model-root' });
        const speech = feed.children.find(node => node.classList.contains('assistant')
            && node.classList.contains('chat-bubble') && !node.classList.contains('typing-bubble'));
        assert.ok(speech.innerHTML.includes(ENGLISH), 'typed metadata never turns model words into host copy');
        assert.equal(speech.innerHTML.includes(TRANSLATED), false);
    } finally { instance?.destroy(); applyPayload({ language: 'en', english: true }); restoreDom(env.prior); }
});

// A press that recorded words (an agent's for the owner, or Ouroboros's own): the head and the retained clause are the
// host's frame, translated as one template; the words are their author's, filled in unchanged (DESIGN "Language").
const SAID = 'You paused this task: По просьбе Антона: stop the export. Its work is retained; Resume continues it.';
const FRAME = 'You paused this task: {words} Its work is retained; Resume continues it.';
const said = { ...notice, card_row_id: 'pause:paused-root:said', text: SAID, content: SAID };

test('the words a pause recorded stay as written on both paths; only the frame around them is translated', async () => {
    const { hostSentenceText } = await import('../modules/host_notice_text.js');
    const { pendingMisses } = await import('../modules/i18n.js');
    const env = installDom();
    try {
        applyPayload({ language: 'ru', english: false, entries: {
            [FRAME]: { text: 'Вы приостановили задачу: {words} Работа сохранена; Resume продолжит её.', provenance: 'imported' } } });
        const shown = 'Вы приостановили задачу: По просьбе Антона: stop the export. Работа сохранена; Resume продолжит её.';
        assert.equal(hostSentenceText(SAID), shown);
        assert.equal(cardRowSummary(said, 'warn', said.ts).headline, shown);
        assert.equal(hostSentenceText(ENGLISH), ENGLISH, 'a press without words is still one exact sentence');
        // A frame the memory lacks shows the English and reports the frame alone, never the author's words.
        applyPayload({ language: 'ru', english: false, entries: {} });
        assert.equal(hostSentenceText(SAID), SAID);
        assert.ok(pendingMisses().includes(FRAME));
        assert.ok(!pendingMisses().some((key) => key.includes('stop the export')));
        // Ouroboros's own sentence rides the same frame; the English reads exactly as stored.
        applyPayload({ language: 'en', english: true });
        const mind = 'Ouroboros paused this task: Freeing the worker. Its work is retained; Resume continues it.';
        assert.equal(hostSentenceText(mind), mind);
        assert.equal(hostSentenceText(SAID), SAID);
    } finally { applyPayload({ language: 'en', english: true }); restoreDom(env.prior); }
});

test('every pause sentence the server writes is cut at the words its press recorded (pause_notice_parity.json)', async () => {
    const { hostSentenceText } = await import('../modules/host_notice_text.js');
    const { cases } = JSON.parse(readFileSync(new URL('./fixtures/pause_notice_parity.json', import.meta.url), 'utf8'));
    assert.ok(cases.some(({ words }) => words === null) && cases.filter(({ words }) => words !== null).length >= 4);
    try {
        for (const { name, text, words } of cases) {
            // The frame is the sentence with the recorded words as its one parameter; any other is one exact sentence.
            const key = words === null ? text : text.replace(words, '{words}');
            applyPayload({ language: 'ru', english: false,
                entries: { [key]: { text: words === null ? '⟨sentence⟩' : '⟨{words}⟩', provenance: 'imported' } } });
            assert.equal(hostSentenceText(text), words === null ? '⟨sentence⟩' : `⟨${words}⟩`, name);
        }
    } finally { applyPayload({ language: 'en', english: true }); }
});
