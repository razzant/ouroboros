// The empty review pool in the Available-subagents editor: with no row marked Reviewer (or every
// marked row switched off) the main model reviews alone, a catalog with rows but no mark saves only
// with the owner's "Save without reviewers", a list without rows needs no tick, and a catalog the
// editor could not read states no pool fact. The server applies the same judge.
import assert from 'node:assert/strict';
import test from 'node:test';

import {
    ALLOW_EMPTY_REVIEW_POOL,
    createAvailableSubagentsEditor,
    reviewPoolErrors,
    reviewPoolRows,
} from '../modules/subagents_settings.js';
import { accessEditorDom, apiRow, sessionRow, setting } from './helpers/subagents_editor.js';

const EMPTY_POOL_ERROR = 'No row is marked Reviewer. Mark at least one row, or tick “Save without reviewers”.';

test('an edited catalog with rows but no Reviewer is refused until the owner saves without reviewers', () => {
    assert.deepEqual(reviewPoolErrors(setting([apiRow()]), { judged: true }), [EMPTY_POOL_ERROR]);
    assert.deepEqual(reviewPoolErrors(setting([apiRow()]), { judged: false }), [], 'an untouched saved catalog is not judged');
    assert.deepEqual(reviewPoolErrors(setting([]), { judged: true }), [], 'no rows leaves nothing to mark');
    assert.deepEqual(reviewPoolErrors(setting([apiRow()]), { judged: true, allowEmpty: true }), []);
    // A pool of switched-off reviewers reviews nothing either: the server refuses it the same way.
    assert.deepEqual(reviewPoolErrors(setting([apiRow({ review_eligible: true, enabled: false })]), { judged: true }),
        ['Every row marked Reviewer is switched off. Switch one on, or tick “Save without reviewers”.']);
    assert.deepEqual(reviewPoolErrors(setting([
        apiRow({ review_eligible: true, enabled: false }), sessionRow({ review_eligible: true }),
    ]), { judged: true }), [], 'one reviewer switched on is a pool');
    assert.deepEqual(reviewPoolRows(setting([
        apiRow({ review_eligible: true }), sessionRow({ review_eligible: true, enabled: false }), apiRow({ subagent_id: 'plain' }),
    ])).map((row) => row.subagent_id), ['api_scout']);

    const dom = accessEditorDom();
    const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null });
    editor.load(setting([apiRow(), sessionRow()]));
    assert.deepEqual(editor.validate(), [], 'loading is not saving');
    assert.equal(dom.pool.count.textContent, 'Reviewers: 0');
    assert.equal(dom.pool.empty.hidden, false);
    assert.equal(dom.pool['empty-text'].textContent, 'No row is marked Reviewer, so the main model reviews alone: one model, no second opinion.');
    assert.equal(dom.pool.confirm.hidden, false);
    assert.equal(dom.pool.stays.hidden, true);

    dom.row(1).querySelector('[data-subagent-field="recommended_use"]').emit('input', 'Edited.');
    assert.deepEqual(editor.validate(), [EMPTY_POOL_ERROR]);
    assert.equal(editor.allowEmptyReviewPool, false);
    assert.equal(editor.allowEmptyReviewPool, false);

    dom.toolbar.allowEmpty.toggle(true);
    assert.deepEqual(editor.validate(), []);
    assert.equal(editor.allowEmptyReviewPool, true);
    assert.equal(editor.allowEmptyReviewPool, true);

    // A marked row makes the confirmation moot: the flag never rides a non-empty pool.
    dom.row(0).querySelector('[data-subagent-field="review_eligible"]').toggle(true);
    assert.equal(editor.setting.items[0].review_eligible, true);
    assert.equal(editor.allowEmptyReviewPool, false);
    assert.deepEqual(editor.validate(), []);
    assert.equal(dom.pool.count.textContent, 'Reviewers: 1');
    assert.equal(dom.pool.empty.hidden, true);
    dom.toolbar.listEnabled.toggle(false);
    assert.deepEqual([dom.pool.stays.textContent, dom.pool.stays.hidden], ['Delegation is off; rows marked Reviewer still review.', false]);
    assert.ok(dom.row(0).querySelector('[data-subagent-field="delivery"]'), 'every API card carries its delivery, shown while marked');
    dom.row(0).querySelector('[data-subagent-field="delivery"]').emit('change', 'packet');
    assert.equal(editor.setting.items[0].delivery, 'packet');
    dom.row(0).querySelector('[data-subagent-field="delivery"]').emit('change', 'native');
    assert.equal('delivery' in editor.setting.items[0], false);

    // A reload forgets the confirmation: it answered one save.
    editor.load(setting([apiRow(), sessionRow()]));
    assert.equal(editor.allowEmptyReviewPool, false);

    // Switching the only reviewer off empties the pool: the same confirmation is asked.
    editor.load(setting([apiRow({ review_eligible: true }), sessionRow()]));
    assert.equal(dom.pool.confirm.hidden, true);
    dom.row(0).querySelector('[data-subagent-field="enabled"]').toggle(false);
    assert.equal(dom.pool['empty-text'].textContent,
        'Every row marked Reviewer is switched off, so the main model reviews alone: one model, no second opinion.');
    assert.equal(dom.pool.confirm.hidden, false);
    assert.deepEqual(editor.validate(), ['Every row marked Reviewer is switched off. Switch one on, or tick “Save without reviewers”.']);
    dom.toolbar.allowEmpty.toggle(true);
    assert.equal(editor.allowEmptyReviewPool, true);
    editor.destroy();
});

test('a list without rows needs no tick and says the main model reviews alone; an unread catalog says nothing', () => {
    const dom = accessEditorDom();
    const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null });
    editor.load(setting([]));
    assert.equal(dom.pool.count.textContent, 'Reviewers: 0');
    assert.equal(dom.pool.empty.hidden, false);
    assert.equal(dom.pool['empty-text'].textContent, 'No row is marked Reviewer, so the main model reviews alone: one model, no second opinion.');
    assert.equal(dom.pool.confirm.hidden, true, 'no rows: the server asks for no confirmation');
    assert.deepEqual(editor.validate(), []);
    assert.equal(editor.allowEmptyReviewPool, false);
    // A catalog this editor could not read states no pool fact, whatever rows it held before.
    editor.load(setting([apiRow()]));
    editor.load('{broken');
    assert.equal(dom.pool.empty.hidden, true);
    assert.equal(dom.pool['empty-text'].textContent, '');
    editor.destroy();
});
