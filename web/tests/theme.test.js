// The accepted theme names are pinned on both ends of the wire: this mirrors
// ui_preferences.py (`dark` | `light`); everything else is the dark default.
import assert from 'node:assert/strict';
import test from 'node:test';

import { normalizeTheme } from '../modules/theme.js';

test('normalizeTheme accepts only the two literal names', () => {
    assert.equal(normalizeTheme('light'), 'light');
    assert.equal(normalizeTheme('dark'), 'dark');
    for (const value of ['Light', 'auto', '', null, undefined, 1, true]) {
        assert.equal(normalizeTheme(value), 'dark', String(value));
    }
});
