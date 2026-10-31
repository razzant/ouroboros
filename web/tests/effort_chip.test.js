// The child card's effort chip (DECISIONS v3 §5): composed from the frame's three scalars and
// painted in the card's META line through the real Chat consumer — live, terminal and sticky
// across frames that carry no effort; an old frame paints nothing.
import assert from 'node:assert/strict';
import test, { after } from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { effortChipMarkup, effortChipText, effortFact } from '../modules/effort_chip.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const originalQuery = ElementStub.prototype.querySelector;
after(() => { ElementStub.prototype.querySelector = originalQuery; });
ElementStub.prototype.querySelector = function (selector) {
    const direct = originalQuery.call(this, selector);
    if (direct) return direct;
    for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
    return null;
};

test('the chip words: applied level, the asked level when it differs, then the source marker', () => {
    assert.equal(effortFact({ status: 'running' }), null, 'no level, no fact');
    assert.equal(effortChipText(null), '');
    assert.equal(effortChipText(effortFact({ effort_level: 'high', effort_requested: 'high', effort_source: 'auto' })), 'Effort High');
    assert.equal(effortChipText(effortFact({ effort_level: 'high', effort_requested: '', effort_source: '' })), 'Effort High', 'an unknown source is still the level');
    assert.equal(effortChipText(effortFact({ effort_level: 'high', effort_requested: 'ultra', effort_source: 'auto' })), 'Effort High (asked Ultra)');
    assert.equal(effortChipText(effortFact({ effort_level: 'high', effort_requested: 'ultra', effort_source: 'pin' })), 'High (asked Ultra)');
    assert.equal(effortChipText(effortFact({ effort_level: 'high', effort_source: 'pin' })), 'High');
    assert.equal(effortChipText(effortFact({ effort_level: 'xhigh', effort_requested: 'xhigh', effort_source: 'model_name' })), 'X-High · model name');
    assert.equal(effortChipText(effortFact({ effort_level: 'xhigh', effort_requested: 'low', effort_source: 'model_name' })), 'X-High (asked Low) · model name');
    assert.equal(effortChipText(effortFact({ effort_level: 'ultra', effort_requested: 'ultra', effort_source: 'cyber' })), 'Effort Ultra (Cyber Pro)');
    assert.equal(effortChipText(effortFact({ effort_level: 'minimal', effort_source: 'auto' })), 'Effort Minimal', 'every runtime tier has a word');
    const pinned = effortChipMarkup(effortFact({ effort_level: 'high', effort_source: 'pin' }));
    assert.match(pinned, /^<span class="chat-live-meta-text chat-live-effort" data-effort-source="pin" aria-label="Pinned effort High"><svg class="chat-live-effort-pin"/);
    assert.match(pinned, />High<\/span>$/);
    const auto = effortChipMarkup(effortFact({ effort_level: 'high', effort_source: 'auto' }));
    assert.equal(auto, '<span class="chat-live-meta-text chat-live-effort" data-effort-source="auto">Effort High</span>');
    assert.equal(effortChipMarkup(null), '');
    assert.match(effortChipMarkup(effortFact({ effort_level: 'high', effort_source: '<x>' })), /data-effort-source="&lt;x&gt;"/);
});

const ROOT = 'root-eff';
const lineage = (kid, role = 'researcher') => ({ subagent_task_id: kid, parent_task_id: ROOT, root_task_id: ROOT,
    delegation_role: 'subagent', subagent_role: role });

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
        meta: (id) => walkCard(messages, id)?.querySelector('[data-live-meta]')?.innerHTML ?? null,
        facts: (id) => walkCard(messages, id)?.querySelector('[data-live-facts]')?.innerHTML ?? null,
        emit: (row, seconds = 0) => handlers.get('chat')({ chat_id: 1, role: 'system', is_progress: true,
            ts: `2026-10-10T12:00:${String(seconds).padStart(2, '0')}Z`, ...row }),
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

// A meta fact ends with its separator when another fact follows (the dot rides with the fact before it).
const tail = String.raw`(?:<span class="chat-live-meta-sep">&nbsp;· <\/span>)?<\/span>`;

test('a child card paints the chip from its frames and keeps it across frames without one', () => {
    const f = fixture();
    try {
        f.emit({ role: 'assistant', task_id: ROOT, content: 'Child queued', subagent_event: 'scheduled', ...lineage('kid-a'),
            effort_level: 'high', effort_requested: 'ultra', effort_source: 'auto', model: 'openai/gpt-5.6-terra' });
        assert.match(f.meta('kid-a'), new RegExp(String.raw`<span class="chat-live-meta-text chat-live-effort" data-effort-source="auto">Effort High \(asked Ultra\)${tail}`));
        assert.match(f.meta('kid-a'), /gpt-5.6-terra<span class="chat-live-meta-sep">&nbsp;· <\/span><\/span><\/span><span class="chat-live-meta-part"><span class="chat-live-meta-text chat-live-effort"/,
            'a child without an executor runs on its own model: the effort follows that model, separated by text');
        f.emit({ role: 'assistant', task_id: 'kid-a', content: 'Reading the tests.', narration: true, subagent_event: 'progress', ...lineage('kid-a') }, 1);
        assert.match(f.meta('kid-a'), /Effort High \(asked Ultra\)/, 'a progress frame without the fact keeps the chip');
        f.emit({ role: 'assistant', task_id: ROOT, content: 'Child done', subagent_event: 'completed', status: 'completed', result: 'ok',
            ...lineage('kid-a'), effort_level: 'high', effort_requested: 'high', effort_source: 'auto' }, 2);
        assert.match(f.meta('kid-a'), new RegExp(String.raw`data-effort-source="auto">Effort High${tail}`), 'the terminal frame restates the fact');
        // Pinned, model-named and Cyber children, each with its own words.
        f.emit({ role: 'assistant', task_id: ROOT, content: 'Child queued', subagent_event: 'scheduled', ...lineage('kid-b', 'reviewer'),
            effort_level: 'high', effort_requested: 'ultra', effort_source: 'pin' }, 3);
        assert.match(f.meta('kid-b'), new RegExp(String.raw`data-effort-source="pin" aria-label="Pinned effort High \(asked Ultra\)"><svg class="chat-live-effort-pin"[^>]*>.*<\/svg>High \(asked Ultra\)${tail}`));
        f.emit({ role: 'assistant', task_id: ROOT, content: 'Child queued', subagent_event: 'scheduled', ...lineage('kid-c', 'second-opinion'),
            effort_level: 'xhigh', effort_requested: 'xhigh', effort_source: 'model_name' }, 4);
        assert.match(f.meta('kid-c'), new RegExp(String.raw`data-effort-source="model_name">X-High · model name${tail}`));
        f.emit({ role: 'assistant', task_id: ROOT, content: 'Child queued', subagent_event: 'scheduled', ...lineage('kid-d', 'deep'),
            effort_level: 'ultra', effort_requested: 'ultra', effort_source: 'cyber' }, 5);
        assert.match(f.meta('kid-d'), new RegExp(String.raw`data-effort-source="cyber">Effort Ultra \(Cyber Pro\)${tail}`));
        // An old frame: no level, no chip — never back-filled from the owner's range.
        f.emit({ role: 'assistant', task_id: ROOT, content: 'Child queued', subagent_event: 'scheduled', ...lineage('kid-e', 'scout') }, 6);
        assert.doesNotMatch(f.meta('kid-e'), /chat-live-effort/);
    } finally { f.close(); }
});

// The Coordinator row of the details states the coordinating model and, when the schedule recorded one,
// its provenance (chosen_by): escaped through the renderer, never inferred from a role or a model name.
test('the coordinator row carries the recorded provenance and escapes a chosen row name', () => {
    const f = fixture();
    try {
        f.emit({ role: 'assistant', task_id: ROOT, content: 'Child done', subagent_event: 'completed', status: 'completed',
            result: 'ok', ...lineage('kid-p', 'auditor'), model: 'claudexor::codex=gpt-6-astra', executor_route: 'claude',
            execution_evidence: { delegated_runs_started: 1, delegated_runs_settled: 1, delegated_runs_succeeded: 1,
                delegated_runs_failed: 0, subscription_cost_usd: 0, harness_models: ['claude-opus-5-5[1m]'] },
            parent_cognitive_route: { model: 'claudexor::codex=gpt-6-astra', chosen_by: 'delegator',
                selected_supervisor_snapshot: { selected_subagent_id: '<chosen&row>' } } });
        const facts = f.facts('kid-p');
        assert.match(facts, /<dt>Coordinator<\/dt><dd>codex=gpt-6-astra \(chosen by the delegator \(&lt;chosen&amp;row&gt;\)\)<\/dd>/);
        assert.doesNotMatch(facts, /<chosen&row>/);
        assert.doesNotMatch(f.meta('kid-p'), /chosen by the delegator/, 'the compact line never carries provenance');
    } finally {
        f.close();
    }
});

// A session child's effort carrier is the LEAF's dispatch decision, not the nanny's reasoning level.
// Keep it visible with the executor identity, using the same chosen/requested/source wording as native children.
test('a delegated child keeps its leaf dispatch effort visible while the coordinator stays in details', () => {
    const f = fixture();
    try {
        f.emit({ role: 'assistant', task_id: ROOT, content: 'Child done', subagent_event: 'completed', status: 'completed',
            result: 'ok', ...lineage('kid-x', 'auditor'), model: 'claudexor::codex=gpt-6-astra', executor_route: 'claude',
            execution_evidence: { delegated_runs_started: 1, delegated_runs_settled: 1, delegated_runs_succeeded: 1,
                delegated_runs_failed: 0, subscription_cost_usd: 0, harness_models: ['claude-opus-5-5[1m]'] },
            effort_level: 'high', effort_requested: 'high', effort_source: 'cyber' });
        const meta = f.meta('kid-x');
        assert.match(meta, /Claude Code/);
        assert.match(meta, />claude-opus-5-5\[1m\]</, 'the executor\'s observed model');
        assert.match(meta, /chat-live-effort/);
        assert.match(meta, /Effort High \(Cyber Pro\)/, 'the leaf dispatch level remains visible');
        assert.doesNotMatch(meta, /gpt-6-astra/, 'the coordinator never stands in for the executor');
        const facts = f.facts('kid-x');
        assert.match(facts, /<dt>Coordinator<\/dt><dd>codex=gpt-6-astra<\/dd>/);
        assert.doesNotMatch(facts, /<dt>Effort<\/dt>/, 'no duplicate effort row');
        // The executor's own sentence is the first row of the same details.
        assert.match(facts, /^<div class="chat-live-fact"><dt>Executor<\/dt><dd>Delegated to your Claude Code account — 1 run settled \(1 ok\), \$0\.00 subscription<\/dd><\/div>/);
    } finally { f.close(); }
});
