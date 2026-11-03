// The "copies of one model" offer of the Available-subagents editor (owner Q5=A): a review pool
// whose rows run one engine (`sameEngineAs`) says what the copies cost, and its button removes
// them from the DRAFT only; the owner's Save writes the change. Diverse rows, a single row,
// unmarked or switched-off look-alikes and another effort never offer it.
import assert from 'node:assert/strict';
import test from 'node:test';

import { ROUTE_KIND_AGENT_SESSION, ROUTE_KIND_API_MODEL } from '../modules/route_editor_primitives.js';
import { createAvailableSubagentsEditor, reviewCopiesText, reviewPoolCopies } from '../modules/subagents_settings.js';
import { accessEditorDom } from './helpers/subagents_editor.js';

const MAIN = 'openai-compatible::glm-5.3';
const api = (id, extra = {}) => ({ subagent_id: id, recommended_use: 'Reviews.', effort: 'high',
    route: { kind: ROUTE_KIND_API_MODEL, target_id: MAIN }, review_eligible: true, ...extra });
const session = (id, extra = {}) => ({ subagent_id: id, recommended_use: 'Reviews.', access: 'full', effort: 'medium',
    route: { kind: ROUTE_KIND_AGENT_SESSION, target_id: 'claude=claude-opus-5-5' }, review_eligible: true, ...extra });
const catalog = (items) => ({ enabled: true, items });
const offer = (items) => {
    const setting = catalog(items);
    return reviewCopiesText(reviewPoolCopies(setting), items.filter((row) => row.review_eligible && row.enabled !== false).length);
};

test('copies of one model are grouped under the first pool row and priced as repeated runs', () => {
    const twins = catalog([api('review-1', { minted_from: 'factory_default' }), api('review-2', { minted_from: 'factory_default' }),
        api('review-3', { minted_from: 'factory_default' })]);
    assert.deepEqual(reviewPoolCopies(twins), [{ keep: 0, copies: [1, 2] }]);
    assert.equal(offer(twins.items), 'Subagents 1, 2 and 3 are copies of one model: a review that asks all of them pays for '
        + '3 runs of it and gets no second opinion. The pool has 1 independent seat, with or without the copies.');
    // What the former one-harness subscription wizard saved: the owner's own row marked plus two minted twins.
    assert.deepEqual(reviewPoolCopies(catalog([session('primary-builder'), session('review-claude', { minted_from: 'factory_default' }),
        session('review-claude-2', { minted_from: 'factory_default' })])), [{ keep: 0, copies: [1, 2] }]);
    // An unmarked delegation row on the same engine is no pool row: the first MARKED row is kept.
    const helper = catalog([api('helper', { review_eligible: undefined }), api('review-1'), api('review-2')]);
    assert.deepEqual(reviewPoolCopies(helper), [{ keep: 1, copies: [2] }]);
    // Copies beside a distinct reviewer: keeping one leaves two answers, not one.
    const mixed = [api('a'), api('b'), api('c', { route: { kind: ROUTE_KIND_API_MODEL, target_id: 'anthropic::claude-opus-5' } })];
    assert.equal(offer(mixed), 'Subagents 1 and 2 are copies of one model: a review that asks all of them pays for 2 runs of it '
        + 'and gets no second opinion. The pool has 2 independent seats, with or without the copies.');
    const other = { route: { kind: ROUTE_KIND_API_MODEL, target_id: 'anthropic::claude-opus-5' } };
    const pairs = catalog([api('a'), api('b'), api('c', other), api('d', other)]);
    assert.deepEqual(reviewPoolCopies(pairs), [{ keep: 0, copies: [1] }, { keep: 2, copies: [3] }]);
    assert.equal(offer(pairs.items), 'Subagents 1 and 2; Subagents 3 and 4 are copies of one model per group: a review that asks '
        + 'every copy pays for each one and gets no second opinion. The pool has 2 independent seats, with or without the copies.');
});

test('the offer promises no count of answers per review: how many seats a review asks follows the strictness', () => {
    // Advisory may ask one copy and Cyber Pro none (IMPL_SPEC §2): the line prices asking every copy and counts
    // the pool's independent seats, never that every copy runs or that its answers decide or block each review.
    const other = { route: { kind: ROUTE_KIND_API_MODEL, target_id: 'anthropic::claude-opus-5' } };
    for (const items of [[api('a'), api('b'), api('c')], [api('a'), api('b'), api('c', other)], [api('a'), api('b'), api('c', other), api('d', other)]]) {
        const text = offer(items);
        assert.match(text, /a review that asks (all of them|every copy) pays for/);
        assert.match(text, /The pool has \d independent seats?, with or without the copies\.$/);
        assert.doesNotMatch(text, /each review pays|decides? each review|block or pass|answers decide/);
    }
});

test('diverse rows, one row, unmarked or switched-off look-alikes and another effort are no copies', () => {
    const diverse = catalog(['openai/gpt-6-sol', 'anthropic/claude-opus-5', 'google/gemini-3.8-pro'].map((target, index) =>
        api(`review-${index + 1}`, { route: { kind: ROUTE_KIND_API_MODEL, target_id: target } })));
    for (const setting of [diverse, catalog([api('review-1')]), catalog([api('review-1'), api('helper', { review_eligible: undefined })]),
        catalog([api('review-1'), api('review-2', { enabled: false })]), catalog([api('review-1'), api('review-2', { effort: 'low' })]),
        catalog([])]) {
        assert.deepEqual(reviewPoolCopies(setting), [], JSON.stringify(setting.items.map((row) => row.subagent_id)));
        assert.equal(reviewCopiesText(reviewPoolCopies(setting), setting.items.length), '');
    }
});

test('Keep one per model removes the copies from the draft only, and the offer follows the marks in place', () => {
    const dom = accessEditorDom();
    const changes = [];
    const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null, onChange: (value) => changes.push(value) });
    editor.load(catalog([api('helper', { review_eligible: undefined, effort: 'low' }), api('review-1', { minted_from: 'factory_default' }),
        api('review-2', { minted_from: 'factory_default' }), api('review-3', { minted_from: 'factory_default' })]));
    assert.equal(dom.pool.copies.hidden, false);
    assert.match(dom.pool['copies-text'].textContent, /^Subagents 2, 3 and 4 are copies of one model: a review that asks all of them pays for 3 runs/);
    assert.match(dom.html(), /<button type="button" class="btn btn-default" data-review-pool-keep-one>Keep one per model<\/button>/);
    assert.equal(editor.dirty, false, 'loading shows the offer and changes nothing');

    // An unmarked copy leaves the pool, so the line is patched in place: the cards are not rebuilt.
    const card = dom.row(3);
    card.querySelector('[data-subagent-field="review_eligible"]').toggle(false);
    assert.equal(dom.row(3), card);
    assert.match(dom.pool['copies-text'].textContent, /^Subagents 2 and 3 are copies of one model: a review that asks all of them pays for 2 runs/);
    card.querySelector('[data-subagent-field="review_eligible"]').toggle(true);

    dom.toolbar.keepOne.emit('click');
    assert.deepEqual(editor.setting.items.map((row) => [row.subagent_id, row.review_eligible === true]), [['helper', false], ['review-1', true]]);
    assert.equal(editor.dirty, true, 'an unsaved draft: the owner sees it and saves it');
    assert.deepEqual(changes.at(-1).items.map((row) => row.subagent_id), ['helper', 'review-1']);
    assert.deepEqual(editor.collect().OUROBOROS_SUBAGENTS.items.map((row) => row.subagent_id), ['helper', 'review-1']);
    assert.deepEqual(editor.validate(), []);
    assert.equal(dom.pool.copies.hidden, true, 'nothing is a copy any more');

    editor.load(catalog([api('review-1'), api('review-2', { route: { kind: ROUTE_KIND_API_MODEL, target_id: 'anthropic::claude-opus-5' } })]));
    assert.equal(dom.pool.copies.hidden, true, 'diverse reviewers are no copies');
    editor.load('{broken');
    assert.equal(dom.pool.copies.hidden, true, 'an unread catalog states no pool fact');
    editor.destroy();
});
