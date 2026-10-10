// The web's effort vocabulary: the 8-tier mirror, the 7 owner tiers, the server's tolerant
// read of the three stored keys, and the words a card or an aria text reads through tr().
import test from 'node:test';
import assert from 'node:assert/strict';

import {
    EFFORT_OPTIONS, EFFORT_RANGE_DEFAULT, EFFORT_SCALE, OWNER_TIERS, effortLabel, effortText, normalizeEffortRange, ownerLevelIndex,
} from '../modules/effort_levels.js';
import { EFFORT_CHOICES } from '../modules/route_editor_primitives.js';

test('the owner tiers are the runtime scale without minimal, in order', () => {
    assert.deepEqual(EFFORT_SCALE, EFFORT_CHOICES);
    assert.deepEqual(OWNER_TIERS, EFFORT_SCALE.filter((tier) => tier !== 'minimal'));
    assert.deepEqual(EFFORT_OPTIONS.map((option) => option.label), ['None', 'Low', 'Medium', 'High', 'X-High', 'Max', 'Ultra']);
    assert.equal(effortLabel('xhigh'), 'X-High');
    assert.equal(effortLabel('minimal'), 'Minimal');
    assert.equal(effortLabel('bogus'), 'bogus');
    assert.equal(effortText('high'), 'High');
});

test('a stored tier outside the owner tiers is shown at the nearest owner tier', () => {
    assert.equal(ownerLevelIndex('none'), 0);
    assert.equal(ownerLevelIndex('minimal'), 1, 'minimal sits at Low');
    assert.equal(ownerLevelIndex('low'), 1);
    assert.equal(ownerLevelIndex('ultra'), 6);
    assert.equal(ownerLevelIndex('bogus'), -1);
});

test("the tolerant read: unknown values take the key's default, min lowers to rec, max rises to rec", () => {
    assert.deepEqual(normalizeEffortRange({}), EFFORT_RANGE_DEFAULT);
    // The owner's install: TASK=high, no MIN/MAX -> Low · High · High.
    assert.deepEqual(normalizeEffortRange({ recommended: 'high' }), { min: 'low', recommended: 'high', max: 'high' });
    assert.deepEqual(normalizeEffortRange({ min: 'ultra', recommended: 'low', max: 'none' }), { min: 'low', recommended: 'low', max: 'low' });
    assert.deepEqual(normalizeEffortRange({ min: 'minimal', recommended: 'extreme', max: 'MAX' }), { min: 'minimal', recommended: 'medium', max: 'max' });
});
