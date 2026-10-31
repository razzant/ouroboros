// An ordinary root card's chosen title and latest authored line are whole (owner 30.10, B: no cut of a root's text):
// the client never cuts them at the 240-character preview. The host already sends the complete narration, live and
// on replay (`fullHeadline`). A nested child and a Main portal keep their bounded preview, and the data layer's
// preview (`activityPreview`) stays bounded for every other reader.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import {
    COLLAPSED_ACTIVITY_MAX, authoredLine, authoredTitle, boundActivityPreview, projectCollapsedActivity, wholeActivityText,
} from '../modules/chat_activity.js';
import { summarizeChatLiveEvent } from '../modules/log_events.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const TS = '2026-10-30T12:00:00Z';
// Longer than the preview, with Markdown the line's projection removes, and a tail only the whole text keeps.
const LONG = `**Reading** the \`ledger\` rows\n\n${'Comparing every saved row with the ledger, one source at a time. '.repeat(6)}UNIQUE_ROOT_TAIL`;
const WHOLE = wholeActivityText(LONG);

const card = (record) => ({ isSubagent: false, root: { dataset: {} }, ...record });

test('the whole normalized text is longer than the preview and keeps its tail; the old cap would cut it', () => {
    assert.ok(WHOLE.length > COLLAPSED_ACTIVITY_MAX + 100, WHOLE.length);
    assert.ok(WHOLE.startsWith('Reading the ledger rows Comparing') && WHOLE.endsWith('UNIQUE_ROOT_TAIL'), WHOLE);
    const cut = boundActivityPreview(LONG);
    assert.equal(cut.length, COLLAPSED_ACTIVITY_MAX);
    assert.ok(cut.endsWith('…') && !cut.includes('UNIQUE_ROOT_TAIL'), 'the bounded preview is what the old card showed');
});

test('authored words: an ordinary root keeps them whole, a child and a portal keep the bounded preview', () => {
    const summary = summarizeChatLiveEvent({ type: 'send_message', is_progress: true, task_id: 'r', content: LONG, narration: true });
    assert.ok(summary.activityPreview.length <= COLLAPSED_ACTIVITY_MAX, 'the data layer keeps its bounded preview');
    assert.equal(authoredLine(card(), summary), WHOLE);
    assert.equal(authoredLine(card({ isSubagent: true }), summary), boundActivityPreview(summary.activityPreview));
    assert.equal(authoredLine(card({ root: { dataset: { portal: '1' } } }), summary), boundActivityPreview(summary.activityPreview));
    // A frame with no preview has no line, whatever its complete words.
    assert.equal(authoredLine(card(), { ...summary, activityPreview: '' }), '');
    // An unnamed card's title: the ordinary root's whole line; a portal keeps its bounded headline.
    const named = { lastHumanHeadline: summary.headline, lastHumanHeadlineSource: { preview: WHOLE } };
    assert.equal(authoredTitle(card(named)), WHOLE);
    assert.equal(authoredTitle(card({ ...named, root: { dataset: { portal: '1' } } })), summary.headline);
});

test('the line projection is whole for an ordinary root and bounded for a child and a portal', () => {
    assert.equal(projectCollapsedActivity({ record: card(), suggestedName: 'Data Analysis', headline: LONG }), WHOLE);
    assert.equal(projectCollapsedActivity({ record: card({ isSubagent: true }), body: LONG }), boundActivityPreview(LONG));
    assert.equal(projectCollapsedActivity({ record: card({ root: { dataset: { portal: '1' } } }), suggestedName: 'P', headline: LONG }),
        boundActivityPreview(LONG));
    // An unnamed root's title already says these words: its line stays empty.
    assert.equal(projectCollapsedActivity({ record: card(), suggestedName: WHOLE, headline: WHOLE }), '');
});

function fixture(history = []) {
    const env = installDom(async (url) => ({ ok: true, json: async () => String(url).startsWith('/api/chat/history')
        ? { messages: history, window: { complete: true } } : { active_direct_turns: [] } }));
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
    const text = (id, part) => walkCard(messages, id)?.querySelector(`[data-live-${part}]`)?.textContent ?? null;
    return {
        instance, title: (id) => text(id, 'title'), line: (id) => text(id, 'activity'),
        emit: (row) => handlers.get('chat')({ chat_id: 1, ts: TS, ...row }),
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

const narration = (task, extra = {}) => ({ task_id: task, role: 'assistant', is_progress: true, narration: true,
    content: LONG, text: LONG, ts: TS, ...extra });
const child = { task_id: 'kid-1', subagent_task_id: 'kid-1', parent_task_id: 'named-root', root_task_id: 'named-root',
    delegation_role: 'subagent', subagent_event: 'running', subagent_role: 'Reader', role: 'assistant', is_progress: true,
    narration: true, content: LONG, text: LONG, ts: TS };

test('live: an unnamed root titles itself with the whole words, a named root shows them whole as its line', () => {
    const f = fixture();
    try {
        f.emit(narration('unnamed-root'));
        f.emit(narration('named-root', { suggested_name: 'Data Analysis' }));
        f.emit(child);
        assert.equal(f.title('unnamed-root'), WHOLE);
        assert.equal(f.line('unnamed-root'), '', 'its line would repeat its title');
        assert.equal(f.title('named-root'), 'Data Analysis');
        assert.equal(f.line('named-root'), WHOLE);
        // The child keeps its bounded preview, as before (and its one fitted line).
        const preview = summarizeChatLiveEvent({ type: 'send_message', ...child }).activityPreview;
        assert.ok(preview.length <= COLLAPSED_ACTIVITY_MAX && !preview.includes('UNIQUE_ROOT_TAIL'), preview);
        assert.equal(f.line('kid-1'), boundActivityPreview(preview));
    } finally { f.close(); }
});

test('replay: the same words read back from history stay whole on both roots', async () => {
    const f = fixture([{ ...narration('unnamed-root'), history_id: 'h-1' },
        { ...narration('named-root', { suggested_name: 'Data Analysis' }), history_id: 'h-2' }]);
    try {
        await f.instance.refreshHistory({ revision: 1 });
        assert.equal(f.title('unnamed-root'), WHOLE);
        assert.equal(f.line('unnamed-root'), '');
        assert.equal(f.title('named-root'), 'Data Analysis');
        assert.equal(f.line('named-root'), WHOLE);
    } finally { f.close(); }
});
