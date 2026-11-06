// A root card's Details say where its money and work went (audit 05.11, item 11): the task detail's read-time
// `cost_breakdown` by purpose ("Spent on") and by route ("Ran on"), each line in the card's own money words beside
// the volume the ledger recorded. The compact line keeps its one amount; a child card adds nothing.
import assert from 'node:assert/strict';
import test from 'node:test';
import { installDom, restoreDom } from './chat_dom_fixture.js';

// The card module it borrows its money words from touches the document when it loads.
const prior = installDom();
const { costBreakdownRows, costSliceLine, purposeWord, routeWord, syncCostBreakdown } =
    await import('../modules/cost_breakdown_presentation.js');
test.after(() => restoreDom(prior));

const carrier = (amount, { unpriced = false, final = true } = {}) => ({
    scope: 'root_tree', tracked_amount: amount, has_unpriced: unpriced, tracked_final: final,
    accounting_open: false, has_rows: true,
});
const slice = (key, presentation, extra = {}) => ({ key, cost_presentation: presentation, model_calls: 0, sessions: 0, tokens: null, ...extra });
// Root 719899335c0641b6 of the 05.11 audit, as its detail serves it.
const VIEW = {
    by_purpose: [
        slice('plan_review_review', carrier(107.7055766, { unpriced: true }), { model_calls: 248, sessions: 6, tokens: 65353690 }),
        slice('task', carrier(15.3397031, { unpriced: true }), { model_calls: 456, tokens: 203427137 }),
        slice('subagent', carrier(0, { unpriced: true }), { sessions: 11, tokens: 108477045 }),
    ],
    by_route: [
        slice('openrouter', carrier(123.0452797), { model_calls: 283, tokens: 68068060 }),
        slice('claudexor', carrier(null, { unpriced: true }), { model_calls: 421, tokens: 1840149690 }),
        slice('codex', carrier(null, { unpriced: true }), { sessions: 8 }),
    ],
};

test('each line is its word, the card\'s own money words and the recorded volume; nothing absent is a zero', () => {
    assert.equal(costSliceLine(VIEW.by_purpose[0], 'Plan review'),
        'Plan review · Tracked: $107.71 · 248 model calls · 6 sessions · 65.4M tokens');
    assert.equal(costSliceLine(VIEW.by_route[0], 'OpenRouter'), 'OpenRouter · $123.05 · 283 model calls · 68.1M tokens');
    // No priced row: the card's word for settled unpriced use, never $0; billions read as B; no tokens, no token part.
    assert.equal(costSliceLine(VIEW.by_route[1], 'Claudexor'), 'Claudexor · Cost not priced · 421 model calls · 1.8B tokens');
    assert.equal(costSliceLine(VIEW.by_route[2], 'Codex'), 'Codex · Cost not priced · 8 sessions');
    assert.equal(costSliceLine(slice('task', carrier(1), { model_calls: 1, sessions: 1, tokens: 5 }), 'Work'),
        'Work · $1.00 · 1 model call · 1 session · 5 tokens');
});

test('purposes use the words of the card\'s Reviews doors, routes their product names; the rest reads as its key', () => {
    // A review category is `<surface>_review` (review_substrate.review_usage_category); the commit gate's waves
    // are recorded as `multi_model_review_review` (review_price.REVIEW_USAGE_CATEGORY).
    assert.deepEqual(['task', 'subagent', 'multi_model_review_review', 'advisory_review', 'commit_gate_review',
        'preflight_review', 'task_acceptance_review', 'plan_review_review', 'consciousness_task', ''].map(purposeWord),
    ['Work', 'Helpers', 'Code review', 'Early look', 'Code review', 'Early look', 'Result review', 'Plan review',
        'Consciousness task', 'Other']);
    assert.deepEqual(['openrouter', 'openai', 'codex', 'claude', 'agy', 'claudexor', 'external-extension'].map(routeWord),
        ['OpenRouter', 'OpenAI', 'Codex', 'Claude Code', 'Antigravity', 'Claudexor', 'External extension']);
});

test('a root\'s detail gives two Details rows; an event without the view keeps them, a child card keeps none', () => {
    const rows = costBreakdownRows(VIEW);
    assert.deepEqual(rows.map(([label]) => label), ['Spent on', 'Ran on']);
    assert.equal(rows[0][1].split('<br>').length, 3);
    assert.match(rows[1][1], /^OpenRouter · \$123\.05/);
    assert.deepEqual(costBreakdownRows(null), []);
    const root = {};
    assert.equal(syncCostBreakdown(root, { cost_breakdown: VIEW }), true);
    assert.deepEqual(root.costBreakdownRows, rows);
    assert.equal(syncCostBreakdown(root, { cost_breakdown: VIEW }), false, 'the same detail repaints nothing');
    // Every root log event, summary and replayed record reaches the same sync without the view: the rows stay.
    for (const payload of [{ type: 'llm_round_finished' }, {}, { cost_breakdown: null }]) {
        assert.equal(syncCostBreakdown(root, payload), false);
        assert.deepEqual(root.costBreakdownRows, rows);
    }
    const later = { ...VIEW, by_purpose: VIEW.by_purpose.slice(0, 1) };
    assert.equal(syncCostBreakdown(root, { cost_breakdown: later }), true, 'a later view replaces them');
    assert.deepEqual(root.costBreakdownRows, costBreakdownRows(later));
    const child = { isSubagent: true };
    assert.equal(syncCostBreakdown(child, { cost_breakdown: VIEW }), false);
    assert.equal(child.costBreakdownRows, undefined);
});
