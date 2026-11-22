import assert from 'node:assert/strict';
import test, { after } from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { summarizeLogEvent, taskTerminalSummary } from '../modules/log_events.js';
import { censusTaskPhase, desiredLiveCardPhase } from '../modules/task_phase_chip.js';
import { TASK_CAUSE_PHRASES } from '../modules/task_reason_words.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

// Retain each parsed row's markup, as in chat_activity_block.test.js.
const innerHTML = Object.getOwnPropertyDescriptor(ElementStub.prototype, 'innerHTML');
let observeTimelineMarkup;
after(() => Object.defineProperty(ElementStub.prototype, 'innerHTML', innerHTML));
Object.defineProperty(ElementStub.prototype, 'innerHTML', {
    configurable: true,
    get() { return innerHTML.get.call(this); },
    set(value) {
        innerHTML.set.call(this, value);
        const html = String(value || '');
        observeTimelineMarkup?.(html);
        let cursor = 0;
        for (const child of this.children) {
            const index = html.indexOf(`<${child.tagName.toLowerCase()}`, cursor);
            if (index < 0) continue;
            child._innerHTML = html.slice(index);
            cursor = index + 1;
        }
    },
});

const TS = '2026-10-21T12:00:00Z';
const NOTICE = 'A tool this task used failed';
const RETAINED_REPORT = 'Retained reference report.';
const RETAINED_TRACE = 'Only the task trace says this.';
const cases = [
    ['quiet', { wake_silent: true, quiet_effective: true }, 'Done · no message', ''],
    ['schedule report', { quiet_effective: true, schedule_report_only: true }, 'Done · no message', ''],
    ['quiet with notice', { wake_silent: true, quiet_effective: false,
        reason_code: 'tool_failure', outcome_axes: { execution: { status: 'degraded', reason_code: 'tool_failure' } } },
    'Done with warnings', NOTICE],
    ['ordinary', { wake_silent: false, quiet_effective: false }, 'Done', ''],
    ['legacy', {}, 'Done', ''],
];
const terminal = (flags) => ({ task_id: 'check', status: 'completed', outcome_final: true, ts: TS, ...flags });

for (const [name, flags, headline, notice] of cases) {
    test(`${name}: shared summary, Logs and phase chip agree`, () => {
        const row = terminal(flags);
        const summary = taskTerminalSummary(row);
        assert.equal(summary.headline, headline);
        assert.equal(summary.terminal, true);
        if (notice) assert.ok(summary.body.includes(notice));
        else assert.equal(summary.body, '');
        assert.equal(summarizeLogEvent({ ...row, type: 'task_done' }).headline, headline);
        assert.equal(desiredLiveCardPhase({ finished: true, quietEffective: flags.quiet_effective }, summary.phase).text, headline);
        assert.equal(censusTaskPhase(null, row).text, headline);
    });

    const paths = ['log', 'chat', 'history'].map(via => [via, true]);
    if (name === 'schedule report') paths.push(...paths.map(([via]) => [via, false]));
    else paths.push(['pruned', true]);
    for (const [via, withProgress] of paths) test(`${name}: actual card via ${via}${withProgress ? '' : ' without progress'}`, async () => {
        const progress = { task_id: 'check', role: 'assistant', is_progress: true, narration: true,
            text: 'Checking for changes', ts: '2026-10-21T11:59:00Z' };
        const summaryRow = { ...terminal(flags), role: 'system', system_type: 'task_summary',
            text: headline, history_id: 'terminal-check' };
        if (via === 'pruned') {
            progress.historical_terminal = { ...flags, status: 'completed', phase: notice ? 'warn' : 'done',
                ts: TS, provenance: 'canonical_task_result_after_finalization' };
            delete summaryRow.wake_silent;
            delete summaryRow.quiet_effective;
        }
        const answer = name === 'ordinary' || name === 'legacy'
            ? { task_id: 'check', role: 'assistant', text: 'The published answer.', ts: TS } : null;
        const rows = [...(withProgress ? [progress] : []), ...(answer ? [answer] : []), summaryRow];
        let reportReads = 0;
        const { prior, mount } = installDom(async (url, options) => ({ ok: true, json: async () => {
            if (String(url).startsWith('/api/chat/history')) return { messages: rows };
            if (String(url) === '/api/tasks/check') {
                if (options?.cache === 'no-store') reportReads++;  // separate from the card's shared artifact/detail read
                return { result: RETAINED_REPORT, trace_summary: RETAINED_TRACE };
            }
            return { active_direct_turns: [] };
        } }));
        const handlers = new Map();
        let instance;
        try {
            instance = createChatInstance({
                ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
                    isConnected: () => true, send() {} },
                state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
                stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }),
                    gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
                chatId: 2, idPrefix: 'chat', mountEl: mount, asPanel: true,
            });
            const messages = globalThis.document.byId.get('chat-messages');
            if (via === 'history' || via === 'pruned') await instance.refreshHistory({ revision: 1 });
            else {
                if (withProgress) handlers.get('chat')({ ...progress, chat_id: 2, content: progress.text });
                if (answer) handlers.get('chat')({ ...answer, chat_id: 2, content: answer.text });
                if (via === 'chat') handlers.get('chat')({ ...summaryRow, chat_id: 2, content: summaryRow.text });
                else handlers.get('log')({ chat_id: 2, data: { ...terminal(flags), type: 'task_done' } });
                await new Promise(resolve => setImmediate(resolve));
            }
            const card = walkCard(messages, 'check');
            assert.ok(card, 'the task retains its card');
            assert.equal(card.dataset.finished, '1');
            assert.equal(card.querySelector('[data-live-phase]').textContent, headline);
            assert.equal(reportReads, 0, 'rendering does not open the report');
            card.querySelector('[data-live-disclosure]').listeners.get('click')[0]({ detail: 0 });
            const nodes = (node) => [node, ...(node.children || []).flatMap(nodes)];
            const markup = nodes(card).map(node => node.innerHTML || '').join('\n');
            if (notice) assert.ok(markup.includes(notice), 'the notice still shows inside the card');
            else assert.doesNotMatch(markup, /class="chat-live-line-body/, 'no empty answer box');
            if (name === 'quiet') assert.match(markup, /Done · no message/, 'the terminal timeline shares the quiet headline');
            if (name === 'schedule report') {
                assert.equal(reportReads, 0, 'Details alone does not open the report');
                assert.ok(nodes(messages).every(node => !String(node.innerHTML || '').includes(RETAINED_REPORT)),
                    'the report is absent from the whole transcript before Show full');
                const key = /data-live-line-toggle="([^"]+)"/.exec(markup)?.[1];
                assert.ok(key, 'the saved report is available on demand');
                const timeline = card.querySelector('.chat-live-timeline');
                const line = { dataset: { liveLineKey: key }, contains: () => false };
                const header = { matches: () => true };
                // The stub keeps the original innerHTML string after keyed DOM patches.
                // Observe the production renderer's new row markup at its parse boundary.
                const renders = [];
                observeTimelineMarkup = html => {
                    if (html.includes(`data-live-line-key="${key}"`)) renders.push(html);
                };
                timeline.listeners.get('click')[0]({ stopPropagation() {},
                    target: { closest: selector => selector === '.chat-live-line.expandable' ? line : header } });
                await new Promise(resolve => setImmediate(resolve));
                assert.equal(reportReads, 1);
                assert.match(renders.at(-1) || '', /Retained reference report\./);
                assert.ok(!renders.some(html => html.includes(RETAINED_TRACE)), 'the saved report reader does not render the trace');
                observeTimelineMarkup = undefined;
            }
            assert.equal(messages.children.some(node => node.classList.contains('assistant')
                && !node.classList.contains('typing-bubble')), Boolean(answer), 'only an actual answer gets a bubble');
            // Replaying into the live card cannot erase or duplicate the terminal presentation.
            await instance.refreshHistory({ revision: 2 });
            assert.equal(walkCard(messages, 'check'), card);
            assert.equal(card.querySelector('[data-live-phase]').textContent, headline);
        } finally { observeTimelineMarkup = undefined; instance?.destroy(); restoreDom(prior); }
    });
}

test('silent intent alone keeps legacy rendering and unfinished work stays unfinished', () => {
    assert.equal(taskTerminalSummary(terminal({ wake_silent: true })).headline, 'Done');
    const summary = taskTerminalSummary({ ...terminal({ quiet_effective: true }), task_phase: 'finalizing' });
    assert.equal(summary.terminal, false);
    assert.equal(summary.headline, 'Working');
    assert.equal(desiredLiveCardPhase({ finished: false, quietEffective: true, finalizingHold: true }).text, 'Finalizing…');
    assert.equal(desiredLiveCardPhase({ finished: true,
        historicalTerminal: { quiet_effective: true } }).text, 'Done · no message');
});

test('effective quiet suppresses terminal explanation while a notice-bearing run retains it', () => {
    const facts = { wake_silent: true, reason_code: 'receipt_absent' };
    assert.equal(taskTerminalSummary(terminal({ ...facts, quiet_effective: true })).body, '');
    assert.ok(taskTerminalSummary(terminal({ ...facts, quiet_effective: false })).body
        .includes(TASK_CAUSE_PHRASES.receipt_absent.replace(/\.$/, '')));
});
