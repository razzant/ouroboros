// Telegram's haptic feedback (haptics.js, owner 19.10 3A): one Telegram call per kind, only inside Telegram, only on a
// Telegram new enough to have it; elsewhere nothing, and a host that throws never reaches the page.
import assert from 'node:assert/strict';
import test from 'node:test';
import { haptic } from '../modules/haptics.js';

function host({ stamp = '', initData = 'query_id=1', version = true, feedback = true, throws = false } = {}) {
    const calls = [];
    const play = (name) => (...args) => { if (throws) throw new Error('host gone'); calls.push([name, ...args]); };
    const HapticFeedback = feedback ? {
        impactOccurred: play('impactOccurred'), selectionChanged: play('selectionChanged'), notificationOccurred: play('notificationOccurred'),
    } : undefined;
    const win = { Telegram: { WebApp: { initData, HapticFeedback, isVersionAtLeast: (v) => (version ? true : v === '0') } } };
    const doc = { documentElement: { dataset: stamp ? { ouroborosHost: stamp } : {} } };
    return { calls, options: { win, doc } };
}

test('each kind plays its one Telegram haptic inside Telegram', () => {
    const h = host();
    assert.equal(haptic('snap', h.options), true);
    assert.equal(haptic('tick', h.options), true);
    assert.equal(haptic('press', h.options), true);
    assert.equal(haptic('threshold', h.options), true);
    assert.equal(haptic('stop', h.options), true);
    assert.equal(haptic('error', h.options), true);
    assert.equal(haptic('pickup', h.options), true);
    assert.deepEqual(h.calls, [['impactOccurred', 'light'], ['selectionChanged'], ['impactOccurred', 'medium'], ['impactOccurred', 'medium'],
        ['impactOccurred', 'rigid'], ['notificationOccurred', 'error'], ['impactOccurred', 'medium']]);
    // The sidecar's host stamp is Telegram too, before the launch data arrives.
    const stamped = host({ stamp: 'telegram', initData: '' });
    assert.equal(haptic('tick', stamped.options), true);
    assert.deepEqual(stamped.calls, [['selectionChanged']]);
});

test('outside Telegram, on a Telegram without haptics, for an unknown kind or a throwing host nothing plays', () => {
    const browser = host({ initData: '' });
    assert.equal(haptic('snap', browser.options), false, 'the SDK loaded in a plain browser is not Telegram');
    const old = host({ version: false });
    assert.equal(haptic('snap', old.options), false, 'before Bot API 6.1');
    const none = host({ feedback: false });
    assert.equal(haptic('snap', none.options), false);
    const unknown = host();
    assert.equal(haptic('warning', unknown.options), false, 'no kind for warning: Telegram iOS plays a sound instead');
    const broken = host({ throws: true });
    assert.equal(haptic('snap', broken.options), false);
    assert.deepEqual([...browser.calls, ...old.calls, ...unknown.calls, ...broken.calls], []);
    assert.equal(haptic('snap', { win: {}, doc: {} }), false, 'no Telegram at all');
});
