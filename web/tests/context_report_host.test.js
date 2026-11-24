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
        assert.deepEqual(removed, ['ouro:theme-changed', 'pywebviewready']);

        // Every leave of the page closes it again: the spent disposer is not called a second time.
        close();
        close();
        assert.deepEqual(removed, ['ouro:theme-changed', 'pywebviewready']);
        assert.equal(mount.hidden, true);
        assert.equal(button.getAttribute('aria-expanded'), 'false');

        // It opens again as a new frame with its own subscription, and closes the same way.
        button.click();
        assert.notEqual(mount.querySelector('iframe'), frame);
        assert.equal(window.listeners('ouro:theme-changed'), 1);
        close();
        close();
        assert.equal(window.listeners('ouro:theme-changed'), 0);
        assert.deepEqual(removed, ['ouro:theme-changed', 'pywebviewready', 'ouro:theme-changed', 'pywebviewready']);
    } finally {
        dom.restore();
    }
});

const nextTurn = () => new Promise((resolve) => setImmediate(resolve));

test('native action is offered once, and only shown on a positive shell fact', async () => {
    const dom = installCardDom();
    try {
        const button = document.createElement('button');
        const mount = document.createElement('section');
        mount.hidden = true;
        const close = bindContextReport(button, mount);
        button.click();
        window.emit('pywebviewready');
        assert.equal(mount.querySelector('.ui-action-row'), null, 'a browser has no native action');

        let resolveShell, opens = 0, probes = 0;
        window.pywebview = { api: {
            shell_info: () => { probes++; return new Promise((resolve) => { resolveShell = resolve; }); },
            open_local_context_report: async () => { opens++; return { state: 'opened' }; },
        } };
        window.emit('pywebviewready');
        window.emit('pywebviewready');
        const actions = mount.querySelector('.ui-action-row');
        assert.equal(window.listeners('pywebviewready'), 0);
        assert.equal(mount.querySelectorAll('.ui-action-row').length, 1);
        assert.equal(actions.hidden, true);
        await nextTurn();
        assert.equal(probes, 1);
        resolveShell({ local_context_archive: true });
        await nextTurn();
        assert.equal(actions.hidden, false);
        actions.querySelector('button').click();
        await nextTurn();
        assert.equal(opens, 1);
        assert.equal(actions.querySelector('[role="status"]').textContent, 'Opened in a separate local window.');

        close();
        button.click();
        const newer = mount.querySelector('.ui-action-row');
        assert.equal(window.listeners('pywebviewready'), 0, 'the bridge was ready at opening');
        await nextTurn();
        resolveShell({ local_context_archive: false });
        await nextTurn();
        assert.equal(newer.hidden, true, 'a non-WebView2 shell cannot expose archive reading');
        close();
    } finally {
        dom.restore();
    }
});

test('a late native capability or click cannot act on a closed frame', async () => {
    const dom = installCardDom();
    try {
        const button = document.createElement('button');
        const mount = document.createElement('section');
        mount.hidden = true;
        let resolveShell, opens = 0;
        window.pywebview = { api: {
            shell_info: () => new Promise((resolve) => { resolveShell = resolve; }),
            open_local_context_report: () => { opens++; return { state: 'opened' }; },
        } };
        const close = bindContextReport(button, mount);
        button.click();
        const detached = mount.querySelector('.ui-action-row');
        await nextTurn();
        close();
        resolveShell({ local_context_archive: true });
        detached.querySelector('button').click();
        await nextTurn();
        assert.equal(detached.hidden, true);
        assert.equal(mount.children.length, 0);
        assert.equal(opens, 0);
    } finally {
        dom.restore();
    }
});


test('a native open already in flight cannot update a replacement mount', async () => {
    const dom = installCardDom();
    try {
        const button = document.createElement('button');
        const mount = document.createElement('section');
        mount.hidden = true;
        let finishOpen;
        window.pywebview = { api: {
            shell_info: async () => ({ local_context_archive: true }),
            open_local_context_report: () => new Promise((resolve) => { finishOpen = resolve; }),
        } };
        const close = bindContextReport(button, mount);
        button.click();
        await nextTurn();
        const old = mount.querySelector('.ui-action-row');
        assert.equal(old.hidden, false);
        old.querySelector('button').click();
        await nextTurn();
        assert.equal(typeof finishOpen, 'function');
        const oldStatus = old.querySelector('[role="status"]');
        const previous = oldStatus.textContent;
        close();
        button.click();
        await nextTurn();
        const fresh = mount.querySelector('.ui-action-row');
        assert.notEqual(fresh, old);
        finishOpen({ state: 'opened' });
        await nextTurn();
        assert.equal(oldStatus.textContent, previous);
        assert.equal(fresh.querySelector('[role="status"]').textContent, '');
        close();
    } finally {
        dom.restore();
    }
});
