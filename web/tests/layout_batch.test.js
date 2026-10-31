// One layout per batch (layout_batch.js): inside a batch every deferred read runs after the batch's writes, all reads of
// a round before any of its writes; outside one, nothing is deferred.
import assert from 'node:assert/strict';
import test from 'node:test';
import { batch, measure } from '../modules/layout_batch.js';

test('outside a batch a measure reads and applies at once', () => {
    const seen = [];
    measure(() => { seen.push('read'); return 1; }, (value) => seen.push(`apply ${value}`));
    assert.deepEqual(seen, ['read', 'apply 1']);
});

test('inside a batch every read of a round runs after the writes, before any apply; an apply asks for the next round', () => {
    const seen = [];
    const result = batch(() => {
        measure(() => { seen.push('read a'); return 'a'; }, (value) => {
            seen.push(`apply ${value}`);
            measure(() => { seen.push('read a2'); return 'a2'; }, (next) => seen.push(`apply ${next}`));
        });
        seen.push('write');
        measure(() => { seen.push('read b'); return 'b'; }, (value) => seen.push(`apply ${value}`));
        return 'done';
    });
    assert.equal(result, 'done');
    assert.deepEqual(seen, ['write', 'read a', 'read b', 'apply a', 'apply b', 'read a2', 'apply a2']);
});

test('the settle step runs before each round, and not at all when nothing waits', () => {
    const seen = [];
    batch(() => seen.push('write'), () => seen.push('settle'));
    assert.deepEqual(seen, ['write']);
    seen.length = 0;
    batch(() => {
        measure(() => seen.push('read'), () => measure(() => seen.push('read 2'), () => seen.push('apply 2')));
    }, () => seen.push('settle'));
    assert.deepEqual(seen, ['settle', 'read', 'settle', 'read 2', 'apply 2']);
});

test('a nested batch settles what is queued so far before it returns: early, never late', () => {
    const seen = [];
    batch(() => {
        measure(() => seen.push('outer read'), () => seen.push('outer apply'));
        batch(() => measure(() => seen.push('inner read'), () => seen.push('inner apply')));
        seen.push('after inner');
        measure(() => seen.push('late read'), () => seen.push('late apply'));
    });
    assert.deepEqual(seen, ['outer read', 'inner read', 'outer apply', 'inner apply', 'after inner', 'late read', 'late apply']);
});

test('a write that throws leaves no batch open: the next measure runs at once', () => {
    assert.throws(() => batch(() => { measure(() => 1, () => {}); throw new Error('write failed'); }), /write failed/);
    const seen = [];
    measure(() => seen.push('read'), () => seen.push('apply'));
    assert.deepEqual(seen, ['read', 'apply']);
});
