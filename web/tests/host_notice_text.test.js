// A known host sentence is painted without the pictograph its producer opens it with;
// the row keeps its bytes. Matching is typed ownership (`system_type`) plus the exact
// source prefix — never a pattern over symbols — so authored rows, unknown types,
// untyped rows and unlisted phrases render exactly as stored, and Copy keeps its input.
import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import test from 'node:test';

import { createChatInstance } from '../modules/chat.js';
import { HOST_NOTICE_PREFIXES, RETIRED_HOST_NOTICE_PREFIXES, hostNoticeDisplayText, hostRowDisplayText } from '../modules/host_notice_text.js';
import { summarizeChatLiveEvent } from '../modules/log_events.js';
import { cardRowSummary } from '../modules/chat_activity.js';
import { loadSkillReviewDetail, renderSkillReviewDisclosure, summarizeSkillReviewMessage } from '../modules/skill_review_card.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

const STARTUP = '♻️ Restored pending queue from snapshot: 2 tasks.';
const TS = '2026-10-10T12:00:00Z';

test('a typed known prefix loses only its leading mark; everything else stays', () => {
    assert.equal(hostNoticeDisplayText(STARTUP, 'startup_notice'), 'Restored pending queue from snapshot: 2 tasks.');
    assert.equal(hostNoticeDisplayText('♻️ Settling 1 task that was still running when the server stopped.', 'startup_notice'),
        'Settling 1 task that was still running when the server stopped.');
    assert.equal(hostNoticeDisplayText('♻️ Restored pending queue: 1 tasks.', 'startup_notice'),
        'Restored pending queue: 1 tasks.');
    // Recorded history keeps its old bytes and display treatment.
    assert.equal(hostNoticeDisplayText('♻️ Cancelling 1 task that was still running when the server stopped.', 'startup_notice'),
        'Cancelling 1 task that was still running when the server stopped.');
    assert.equal(hostNoticeDisplayText('⚠️ Restart skipped: dirty tree', 'restart_notice'), 'Restart skipped: dirty tree');
    // Typed ownership: the same bytes under an authored or unknown type, or untyped, stay.
    for (const type of ['reminder', 'skill_notice', 'project_completion_summary', 'a_future_notice', '', undefined, 'constructor']) {
        assert.equal(hostNoticeDisplayText(STARTUP, type), STARTUP, String(type));
    }
    // Source-specific prefix: an unlisted sentence of a listed type keeps its mark.
    assert.equal(hostNoticeDisplayText('♻️ Something new happened.', 'startup_notice'), '♻️ Something new happened.');
    assert.equal(hostNoticeDisplayText('🎉 Restored pending queue from snapshot: 2 tasks.', 'startup_notice'),
        '🎉 Restored pending queue from snapshot: 2 tasks.');
    // A translated sentence (a `tx` type) loses the same mark; the match reads the stored English.
    assert.equal(hostNoticeDisplayText('🧠 Memory: kept.', 'legacy_memory_notice', '🧠 Память: сохранена.'), 'Память: сохранена.');
    assert.equal(hostNoticeDisplayText('🧠 Memory: kept.', 'legacy_memory_notice', 'Память: сохранена.'), 'Память: сохранена.');
});

// Each entry is one producer's prefix: its mark (variation selector included) and its
// opening words must exist in the backend sources, or the table silently cleans nothing.
test('every table entry names a mark and words a backend producer actually writes', () => {
    const root = new URL('../../', import.meta.url);
    const sources = [];
    const walk = (dir) => {
        for (const name of readdirSync(dir)) {
            const path = new URL(name, dir);
            if (statSync(path).isDirectory()) walk(new URL(`${name}/`, dir));
            else if (name.endsWith('.py')) sources.push(readFileSync(path, 'utf8'));
        }
    };
    walk(new URL('ouroboros/', root));
    walk(new URL('supervisor/', root));
    sources.push(readFileSync(new URL('server.py', root), 'utf8'));
    const backend = sources.join('\n');
    for (const [type, prefixes] of HOST_NOTICE_PREFIXES) {
        assert.ok(backend.includes(`"${type}"`), `system_type ${type} is stamped by a producer`);
        for (const prefix of prefixes) {
            const cut = prefix.indexOf(' ') + 1;
            assert.ok(cut > 1, prefix);
            assert.ok(backend.includes(prefix.slice(0, cut)), `mark of ${JSON.stringify(prefix)}`);
            assert.ok(backend.includes(prefix.slice(cut).trimEnd()), `words of ${JSON.stringify(prefix)}`);
        }
    }
});

test('a retired producer\'s old rows still lose only their mark, and nothing writes its prefix again', () => {
    const root = new URL('../../', import.meta.url);
    const sources = [];
    const walk = (dir) => {
        for (const name of readdirSync(dir)) {
            const path = new URL(name, dir);
            if (statSync(path).isDirectory()) walk(new URL(`${name}/`, dir));
            else if (name.endsWith('.py')) sources.push(readFileSync(path, 'utf8'));
        }
    };
    walk(new URL('ouroboros/', root));
    walk(new URL('supervisor/', root));
    sources.push(readFileSync(new URL('server.py', root), 'utf8'));
    const backend = sources.join('\n');
    assert.ok(RETIRED_HOST_NOTICE_PREFIXES.size > 0);
    for (const [type, prefixes] of RETIRED_HOST_NOTICE_PREFIXES) {
        for (const prefix of prefixes) {
            assert.ok(!backend.includes(prefix.trimEnd()), `a producer writes the retired ${JSON.stringify(prefix)} again`);
            const words = prefix.slice(prefix.indexOf(' ') + 1);
            assert.equal(hostNoticeDisplayText(`${prefix}x`, type), `${words}x`);
            assert.equal(hostNoticeDisplayText(`${prefix}x`, 'another_type'), `${prefix}x`, 'only its own type');
        }
    }
});

function chatFixture(history = []) {
    const env = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history')
            ? { messages: history, window: { complete: true } }
            : { active_direct_turns: [] } }));
    const handlers = new Map();
    let generation = 0;
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
            isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
    });
    const messages = document.byId.get('chat-messages');
    const nodes = (node) => [node, ...(node?.children || []).flatMap(nodes)];
    return {
        instance,
        systemBubbles: () => nodes(messages).filter((n) => n.classList?.contains('chat-bubble') && n.classList.contains('system')),
        emit: (row) => handlers.get('chat')({ chat_id: 1, ts: TS, ...row }),
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

async function copied(bubble) {
    const priorNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
    let written = null;
    Object.defineProperty(globalThis, 'navigator', { configurable: true,
        value: { clipboard: { writeText: async (value) => { written = value; } } } });
    try {
        bubble.children.find((child) => child.classList.contains('chat-message-copy')).click();
        await new Promise((resolve) => setTimeout(resolve, 0));
    } finally {
        if (priorNavigator) Object.defineProperty(globalThis, 'navigator', priorNavigator);
        else delete globalThis.navigator;
    }
    return written;
}

test('an old startup row replayed from history paints without ♻ and copies its stored bytes', async () => {
    const f = chatFixture([{ role: 'system', system_type: 'startup_notice', text: STARTUP, content: STARTUP,
        ts: TS, chat_id: 1 }]);
    try {
        await f.instance.refreshHistory({ revision: 1 });
        const [bubble] = f.systemBubbles();
        assert.ok(bubble, 'the replayed row is a System bubble (the existing amber treatment)');
        assert.match(bubble.innerHTML, /<div class="sender">System<\/div>/);
        assert.match(bubble.innerHTML, /<div class="message">Restored pending queue from snapshot: 2 tasks\.<\/div>/);
        assert.doesNotMatch(bubble.innerHTML, /♻/);
        assert.match(bubble.dataset.messageKey, /♻️ Restored pending queue/, 'the message key keeps the stored text');
        assert.equal(await copied(bubble), STARTUP, 'Copy keeps the stored sentence, mark included');
    } finally { f.close(); }
});

test('a live startup row paints the same; authored, unknown and untyped System rows stay verbatim', async () => {
    const f = chatFixture();
    try {
        await f.instance.refreshHistory({ revision: 1 });
        f.emit({ role: 'system', system_type: 'startup_notice', content: STARTUP, text: STARTUP });
        const reminder = 'Reminder · Ouroboros · written 09:00 · for 12:00\n♻️ Restarting. 🎉 water the plants';
        f.emit({ role: 'system', system_type: 'reminder', content: reminder, text: reminder, ts: '2026-10-10T12:00:01Z' });
        f.emit({ role: 'system', system_type: 'a_future_notice', content: '♻️ Restarting.', text: '♻️ Restarting.',
            ts: '2026-10-10T12:00:02Z' });
        f.emit({ role: 'system', content: '⚠️ Restart skipped: untyped', text: '⚠️ Restart skipped: untyped',
            ts: '2026-10-10T12:00:03Z' });
        const [startup, authored, unknown, untyped] = f.systemBubbles();
        assert.match(startup.innerHTML, /class="message">Restored pending queue from snapshot: 2 tasks\.</);
        assert.equal(await copied(startup), STARTUP);
        assert.match(authored.innerHTML, /♻️ Restarting\. 🎉 water the plants/);
        assert.equal(await copied(authored), reminder);
        assert.match(unknown.innerHTML, /class="message">♻️ Restarting\.</);
        assert.match(untyped.innerHTML, /class="message">⚠️ Restart skipped: untyped</);
        for (const bubble of [startup, authored, unknown, untyped]) {
            assert.match(bubble.innerHTML, /<div class="sender">System<\/div>/);
        }
    } finally { f.close(); }
});

test('a typed host progress line shows words in the card; narration and the dedupe key keep their bytes', () => {
    const host = summarizeChatLiveEvent({ type: 'send_message', is_progress: true, task_id: 't1', role: 'system',
        system_type: 'host_progress', narration: false, text: '💬 🌐 Connection restored — continuing from saved work in a new attempt.' });
    assert.equal(host.headline, 'Connection restored — continuing from saved work in a new attempt.');
    assert.equal(host.fullHeadline, host.headline);
    assert.equal(host.dedupeKey, 'progress:🌐 Connection restored — continuing from saved work in a new attempt.');
    const scheduled = summarizeChatLiveEvent({ type: 'send_message', is_progress: true, task_id: 't1', role: 'system',
        system_type: 'task_scheduled', text: '🗓️ Scheduled subagent c1 (reviewer): read the diff' });
    assert.equal(scheduled.headline, 'Scheduled subagent c1 (reviewer): read the diff');
    const said = summarizeChatLiveEvent({ type: 'send_message', is_progress: true, task_id: 't1', role: 'assistant',
        system_type: 'model_narration', narration: true, text: '💬 🌐 Connection restored, I think' });
    assert.equal(said.headline, '🌐 Connection restored, I think', 'model narration keeps its own pictograph');
});

test('authored progress wins over a stale host system_type', () => {
    const words = '🌐 Connection restored — these are my words.';
    for (const role of ['assistant', 'user', 'system']) {
        const row = summarizeChatLiveEvent({ type: 'send_message', is_progress: true, task_id: 't1',
            role, narration: true, system_type: 'host_progress', text: `💬 ${words}` });
        assert.equal(row.headline, words);
    }
    const row = summarizeChatLiveEvent({ type: 'send_message', is_progress: true, task_id: 't1',
        role: 'assistant', narration: false, system_type: 'host_progress', text: `💬 ${words}` });
    assert.equal(row.headline, words);
});


test('placed host rows keep receipt severity and quoted content without decorating the headline', () => {
    const text = '⚠️ PR #12 merge: merged; review coverage unknown\nQuoted: 🎉 keep my words';
    const row = { role: 'system', narration: false, system_type: 'host_progress', text, card_row_id: 'r12' };
    const shown = cardRowSummary(row, 'warn');
    assert.equal(shown.headline, 'Warning: PR #12 merge: merged; review coverage unknown');
    assert.equal(shown.body, 'Quoted: 🎉 keep my words');
    assert.equal(row.text, text);
    assert.equal(shown.dedupeKey, 'cardrow|r12');
    assert.equal(cardRowSummary({ ...row, narration: true }, 'warn').headline, text.split('\n')[0]);
    assert.equal(cardRowSummary({ ...row, role: 'assistant' }, 'warn').headline, text.split('\n')[0]);
});

test('skill review paint changes only its generated opening, in legacy and lazy detail', async () => {
    const text = '⚠️ Skill review round 1 — snapshot abc (attempt 1): `demo` — status=warnings\n'
        + '## Findings\n- [FAIL advisory] clarity: 🎉 is quoted evidence\n'
        + '```text\n⚠️ Skill review round 9 is a quoted answer\n```';
    const painted = text.slice('⚠️ '.length);
    assert.match(summarizeSkillReviewMessage(text).headline, /^Skill review round 1/);
    let rendered = '';
    renderSkillReviewDisclosure(text, null, { render: (value) => { rendered = value; return value; } });
    assert.equal(rendered, painted);
    const full = { dataset: {}, setAttribute() {}, innerHTML: '' };
    await loadSkillReviewDetail(full, { skill: 'demo', jobId: 'j1' }, {
        fetchImpl: async () => ({ ok: true, json: async () => ({ markdown: text }) }), render: (value) => value,
    });
    assert.equal(full.innerHTML, painted);
    assert.match(full.innerHTML, /\n⚠️ Skill review round 9 is a quoted answer/);
});

// Probe independently selected real producer phrases, rather than enumerating the
// display table (which cannot detect a missing producer entry).
test('command, plan and paused-evolution producer variants are covered', () => {
    const cases = [
        ['server.py', 'command_reply', '🧠 Background consciousness: ', 'off'],
        ['ouroboros/tools/plan_ask.py', 'host_progress', '📐 Plan: ', 'ask a1 on revision 1 — sending to 2 participants.'],
        ['supervisor/evolution_lifecycle.py', 'evolution_notice', '🧬⚠️ Evolution paused: ', '3 consecutive failures.'],
        // An OpenAI attempt and the stream's own search event stay two rows, each without its mark.
        ['ouroboros/tools/search.py', 'host_progress', '🔍 Trying ', 'OpenAI web search: q'],
        ['ouroboros/tools/search.py', 'host_progress', '🔍 Searching: ', 'q'],
    ];
    for (const [path, type, prefix, suffix] of cases) {
        assert.ok(readFileSync(new URL(`../../${path}`, import.meta.url), 'utf8').includes(prefix));
        const raw = prefix + suffix;
        assert.equal(hostNoticeDisplayText(raw, type), raw.slice(raw.indexOf(' ') + 1));
        assert.equal(hostNoticeDisplayText(raw, 'reminder'), raw);
    }
});

test('child host progress shares root presentation without changing its raw dedupe identity', () => {
    const text = '💬 ⚡ Fallback: host route';
    const event = { type: 'send_message', is_progress: true, task_id: 'child', parent_task_id: 'root',
        subagent_task_id: 'child', delegation_role: 'subagent', subagent_event: 'progress',
        role: 'system', system_type: 'host_progress', narration: false, text };
    const host = summarizeChatLiveEvent(event);
    assert.equal(host.body, 'Fallback: host route');
    assert.equal(host.fullBody, 'Fallback: host route');
    assert.equal(host.activityPreview, 'Fallback: host route');
    assert.ok(host.dedupeKey.endsWith('⚡ Fallback: host route'));
    for (const authored of [{ role: 'assistant' }, { narration: true }]) {
        const row = summarizeChatLiveEvent({ ...event, ...authored });
        assert.equal(row.body, '⚡ Fallback: host route');
        assert.equal(row.fullBody, '⚡ Fallback: host route');
        assert.equal(row.dedupeKey, host.dedupeKey);
    }
});

test('new warning words use the installed translation memory without translating quoted text', async () => {
    const { applyPayload, currentPayload, CODE_PREFIX } = await import('../modules/i18n.js');
    const prior = currentPayload();
    try {
        applyPayload({ language: 'ru', english: false, entries: { [CODE_PREFIX + 'task.host.warning']: { text: 'Внимание' } } });
        const raw = '⚠️ PR #12 merge: quoted 🎉 evidence';
        assert.equal(hostNoticeDisplayText(raw, 'host_progress'), 'Внимание: PR #12 merge: quoted 🎉 evidence');
        assert.equal(hostNoticeDisplayText(raw, 'reminder'), raw);
    } finally { applyPayload(prior); }
});

test('a progress wrapper is removed only with a known host opening', () => {
    const text = '💬 ⚡ Fallback: route changed';
    const row = { role: 'system', narration: false, is_progress: true, system_type: 'host_progress', text };
    assert.equal(hostRowDisplayText(row), 'Fallback: route changed');
    for (const other of [{ narration: true }, { role: 'assistant' }, { system_type: 'reminder' }, { is_progress: false }]) {
        assert.equal(hostRowDisplayText({ ...row, ...other }), text);
    }
    const unknown = '💬 🦊 an unlisted host note';
    assert.equal(hostRowDisplayText({ ...row, text: unknown }), unknown);
    assert.equal(hostRowDisplayText({ ...row, text: unknown }, '💬 translated unknown'), '💬 translated unknown');
    assert.equal(row.text, text);
});
