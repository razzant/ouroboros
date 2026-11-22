// Beautiful Context's frame host (context_report_host.js): one button opens and closes the opaque report frame in
// place below itself. Closing drops the frame and the theme subscription it held, once; a repeated close is a no-op.
import assert from 'node:assert/strict';
import test from 'node:test';

import { bindContextReport } from '../modules/context_report_host.js';
import { installCardDom } from './helpers/card_dom.js';

test('a close unsubscribes the frame from the theme once and keeps nothing of it; a repeated close calls nothing', () => {
    const dom = installCardDom();
    try {
        const removed = [];
        const remove = window.removeEventListener;
        window.removeEventListener = (type, handler) => { removed.push(type); return remove(type, handler); };
        const button = document.createElement('button');
        const mount = document.createElement('section');
        mount.hidden = true;
        const close = bindContextReport(button, mount);

        button.click();
        const frame = mount.querySelector('iframe');
        const told = [];
        frame.contentWindow = { postMessage: (message, origin) => told.push([message.type, origin]) };
        assert.equal(mount.hidden, false);
        assert.equal(button.getAttribute('aria-expanded'), 'true');
        assert.equal(window.listeners('ouro:theme-changed'), 1);
        window.emit('ouro:theme-changed');
        assert.deepEqual(told, [['ouro-context-report-theme', '*']], 'an open frame hears the theme');

        button.click();
        assert.equal(mount.hidden, true);
        assert.equal(mount.children.length, 0);
        assert.equal(button.getAttribute('aria-expanded'), 'false');
        assert.equal(window.listeners('ouro:theme-changed'), 0);
        window.emit('ouro:theme-changed');
        assert.equal(told.length, 1, 'a closed frame hears no theme');
        assert.deepEqual(removed, ['ouro:theme-changed']);

        // Every leave of the page closes it again: the spent disposer is not called a second time.
        close();
        close();
        assert.deepEqual(removed, ['ouro:theme-changed']);
        assert.equal(mount.hidden, true);
        assert.equal(button.getAttribute('aria-expanded'), 'false');

        // It opens again as a new frame with its own subscription, and closes the same way.
        button.click();
        assert.notEqual(mount.querySelector('iframe'), frame);
        assert.equal(window.listeners('ouro:theme-changed'), 1);
        close();
        close();
        assert.equal(window.listeners('ouro:theme-changed'), 0);
        assert.deepEqual(removed, ['ouro:theme-changed', 'ouro:theme-changed']);
    } finally {
        dom.restore();
    }
});
