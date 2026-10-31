// The capsules' haptics (owner 19.10, 3A; haptics.js): a finger that reaches another Effort level ticks once per level,
// a press on another level or Context mode ticks once, the same level or mode and the keyboard never do.
import test from 'node:test';
import assert from 'node:assert/strict';

import { mount, mountPair } from './composer_controls_fixture.js';

test('a finger that reaches another level ticks once per level; a press on another level or mode ticks once (haptics.js)', async () => {
    const played = [];
    const prior = globalThis.window;
    // Telegram's bridge, as haptics.js finds it; outside Telegram nothing plays (web/tests/haptics.test.js).
    globalThis.window = { Telegram: { WebApp: { initData: 'q', isVersionAtLeast: () => true,
        HapticFeedback: { selectionChanged: () => played.push('tick'), impactOccurred: () => played.push('impact') } } } };
    try {
        const m = mount();
        m.control.syncState({ effort_range: { min: 'low', recommended: 'medium', max: 'high' } });
        m.headClick(1);
        m.down(100);  // on the pill's own level: no change, no tick
        assert.deepEqual(played, []);
        m.move(130);
        assert.deepEqual(played, ['tick'], 'the pill reached High');
        m.move(135);
        assert.deepEqual(played, ['tick'], 'still High');
        m.move(250);
        assert.deepEqual(played, ['tick', 'tick'], 'one tick for the move that reached Ultra');
        m.up(250);
        await m.settle();
        m.down(10); m.up(10);
        assert.deepEqual(played, ['tick', 'tick', 'tick'], 'a press on another level');
        await m.settle();
        m.down(10); m.up(10);
        assert.equal(played.length, 3, 'a press where the pill already is changes nothing');
        await m.settle();
        m.key('rec', 'ArrowRight');
        assert.equal(played.length, 3, 'the keyboard has no haptic');
        m.control.destroy();
        const pair = mountPair();
        await pair.press(pair.contextHead);
        await pair.choose('low');
        assert.deepEqual(played.slice(3), ['tick'], 'another context mode');
        await pair.choose('low');
        assert.deepEqual(played.slice(3), ['tick'], 'the chosen mode again changes nothing');
        pair.controls.destroy();
    } finally {
        globalThis.window = prior;
    }
});
