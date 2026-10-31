import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import { slotWaitSeconds } from '../modules/settings.js';

// The help names the wait `model_concurrency._max_slot_wait_sec` applies, which counts a finite
// negative as 0. Settings GET types this key by its integer default (`config._coerce_setting_value`).
test('a finite negative saved wait shows 0, never the default', () => {
    for (const saved of [-1, -30, '-30', -0.5]) {
        assert.equal(slotWaitSeconds(saved), 0, `saved ${JSON.stringify(saved)}`);
    }
});

test('a finite wait of 0 or more shows as saved', () => {
    for (const [saved, shown] of [[90, 90], ['90', 90], [2.5, 2.5], [0, 0]]) {
        assert.equal(slotWaitSeconds(saved), shown, `saved ${JSON.stringify(saved)}`);
    }
});

test('a missing or non-finite wait shows the 180 s default', () => {
    for (const saved of [undefined, null, '', 'abc', Infinity, NaN]) {
        assert.equal(slotWaitSeconds(saved), 180, `saved ${String(saved)}`);
    }
});

test('the help sentence is written through that reading, never a raw copy of the saved value', () => {
    const source = fs.readFileSync(new URL('../modules/settings.js', import.meta.url), 'utf8');
    const writes = source.match(/byId\('s-model-slot-wait'\)\.textContent = [^;]+;/g);
    assert.deepEqual(writes, ["byId('s-model-slot-wait').textContent = String(slotWaitSeconds(s.OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC));"]);
});
