import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { reviewGroupsFromTaskDetail, renderReviewsSection } from '../modules/review_presentation.js';

const executions = [{ kind: 'harness', harness_id: 'codex', model: 'observed/model' }];
const groups = () => reviewGroupsFromTaskDetail({ task_id: 'owner', review_projection: { review_records_omitted: 0, panels: [{
    record_id: 'rec', surface: 'change', aggregate_signal: 'PASS', goal: 'Concrete saved change', actors: [
        { slot_id: 'one', model: 'observed/model', observed_model: 'observed/model', requested_model: 'requested/model', executions,
            semantic_verdict: 'PASS', parse_status: 'valid', transport_status: 'success', reason: 'The saved source keeps the final paragraph.', cost: { kind: 'unknown' } },
        { slot_id: 'two', model: '', requested_model: 'codex=not-observed', semantic_verdict: '', parse_status: 'none',
            reason: 'The route refused this request before execution.', transport_status: 'not_dispatched' },
    ],
}] } });
const open = { sectionExpanded: true, expandedGroups: new Set(['review_record:owner:record:rec']) };
const seats = (html) => [...html.matchAll(/<div class="review-seat" data-review-opinion="[^"]+">[\s\S]*?<\/div><\/div>/g)].map(m => m[0]);

test('ledger opinions retain only explicit per-seat harness identity and quiet shared facts', () => {
    const html = renderReviewsSection(groups(), open);
    const [first, second] = seats(html);
    assert.equal(seats(html).length, 2);
    assert.match(first, /data-harness-identity="codex"/);
    assert.match(first, />Codex</);
    // An unobserved model never borrows another seat's harness, and its request is labelled as one.
    assert.doesNotMatch(second, /data-harness-identity/);
    assert.match(second, /<span class="review-seat-name">codex=not-observed<\/span><span class="review-seat-note">requested · not reported<\/span>/);
    assert.match(second, /data-tone="warn">not sent<\/span>/);
    assert.match(second, /The route refused this request before execution\./);
    for (const opinion of seats(html)) assert.doesNotMatch(opinion, /<details|Record facts/);
    assert.equal((html.match(/data-review-facts=/g) || []).length, 1);
    assert.match(html, /cost unknown/);
    assert.match(html, /<dt>Reviewer one model<\/dt><dd>requested requested\/model · observed observed\/model<\/dd>/);
});

test('plan participant execution stays per-seat and objections remain in shared facts', () => {
    const plan = reviewGroupsFromTaskDetail({ task_id: 'plan-owner', status: 'completed',
        plan_record: { revisions: [{ n: 1, goal: 'Review the chosen design' }], asks: [{ ask_id: 'ask', revision: 1,
            participants: [{ seat_id: 'critic', model: 'observed/model', owner_marked: true, status: 'answered', executions,
                reply: { prose: 'The plan preserves the source.', objections: [{ id: 'one', text: 'Check replay.' }] } }] }] },
        plan_objections: { open_objections: {} },
    });
    const html = renderReviewsSection(plan, { sectionExpanded: true, expandedGroups: new Set(['plan:plan-owner']) });
    assert.match(html, /data-harness-identity="codex"/);
    assert.match(html, /<dt>Objection critic\/one<\/dt><dd>Check replay\.<\/dd>/);
    assert.match(html, /Nobody votes on a plan: an objection stands until the participant who raised it withdraws it\./);
    assert.match(html, /data-tone="ok">answered, no objection</);
    assert.equal((html.match(/data-review-facts=/g) || []).length, 1);
});

test('failed hydration and retry remain visible outside a collapsed Reviews tray', () => {
    for (const content of [[], groups()]) {
        for (const state of [{ hydrateStatus: 'error' }, { hydrateStatus: 'loading', hadHydrateError: true }]) {
            const html = renderReviewsSection(content, { ...state, sectionExpanded: false });
            assert.match(html, /data-review-hydrate-visible="1"/);
            assert.ok(html.indexOf('data-review-hydrate-status') < html.indexOf('class="chat-review-groups'));
            assert.match(html, /class="chat-review-groups review-checks" hidden/);
        }
    }
    assert.match(renderReviewsSection([], { hydrateStatus: 'error' }),
        /role="alert"><span class="review-hydrate-text ui-status" data-tone="warn">Review details failed to refresh — shown data may be incomplete\.<\/span><button type="button" class="review-read ui-hit-area" data-review-hydrate-retry>Retry read<\/button>/);
    const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8');
    assert.match(css, /\.chat-live-reviews\[data-expanded="0"\]:not\(\[data-review-hydrate-visible\]\)/);
    assert.equal(renderReviewsSection([], {}), '');
    assert.equal(renderReviewsSection([], { hydrateStatus: 'loading' }), '');
});
