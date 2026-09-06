import assert from 'node:assert/strict';
import test from 'node:test';

import { extensionActionStatus } from '../modules/extension_status_text.js';

test('a message wins and keeps the success tone', () => {
    assert.deepEqual(extensionActionStatus({ message: 'Rebuilt.' }), { text: 'Rebuilt.', tone: 'ok' });
});

test('a degraded bridge is reported, never "Saved." (#376)', () => {
    const body = {
        bridge: { state: 'degraded', owner_bound: true, poller: 'degraded', reason_code: 'telegram_startup_deferred' },
        mini_app: { state: 'disabled', message: 'Mini App is off.' },
    };
    assert.deepEqual(extensionActionStatus(body), {
        text: 'bridge: degraded (telegram_startup_deferred) · mini_app: disabled (Mini App is off.)',
        tone: 'muted',
    });
});

test('a ready bridge without reasons reads as plain states', () => {
    assert.deepEqual(extensionActionStatus({ bridge: { state: 'ready' }, mini_app: { state: 'ready' } }), {
        text: 'bridge: ready · mini_app: ready',
        tone: 'muted',
    });
});

test('responses without states fall back to the generic label', () => {
    assert.deepEqual(extensionActionStatus({ ok: true }), { text: 'Saved.', tone: 'ok' });
    assert.deepEqual(extensionActionStatus(null), { text: 'Saved.', tone: 'ok' });
    assert.deepEqual(extensionActionStatus([{ state: 'x' }]), { text: 'Saved.', tone: 'ok' });
    assert.deepEqual(extensionActionStatus({ items: [{ state: 'x' }], count: 2 }), { text: 'Saved.', tone: 'ok' });
});
