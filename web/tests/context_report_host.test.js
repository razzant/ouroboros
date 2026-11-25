// Beautiful Context's local requests on Dashboard › Logs (context_report_host.js): hidden everywhere except a
// supported Windows WebView2 shell whose DesktopApi exposes `open_local_context_report` and returns
// `shell_info().local_context_archive === true`. A press answers in the shared action result (ui_primitives.js
// setResult): an acknowledgement ends by itself, an unavailable viewer stays until the next press.
import assert from 'node:assert/strict';
import test from 'node:test';

import { bindLocalContextReport } from '../modules/context_report_host.js';
import { SAVE_ACK_MS } from '../modules/ui_primitives.js';
import { installCardDom } from './helpers/card_dom.js';

const nextTurn = () => new Promise((resolve) => setImmediate(resolve));

test('native action is offered once, and only shown on a positive shell fact', async () => {
    const dom = installCardDom();
    try {
        const mount = document.createElement('div');
        mount.hidden = true;
        const dispose = bindLocalContextReport(mount);
        window.emit('pywebviewready');
        assert.equal(mount.hidden, true, 'a browser keeps the mount hidden');
        assert.equal(mount.querySelector('button'), null, 'a browser renders no native button');

        let resolveShell, opens = 0, probes = 0;
        window.pywebview = { api: {
            shell_info: () => { probes++; return new Promise((resolve) => { resolveShell = resolve; }); },
            open_local_context_report: async () => { opens++; return { state: 'opened' }; },
        } };
        window.emit('pywebviewready');
        window.emit('pywebviewready');
        assert.equal(window.listeners('pywebviewready'), 0, 'listener detaches once the bridge answers');
        assert.equal(mount.querySelectorAll('button').length, 1);
        assert.equal(mount.hidden, true, 'button stays hidden until shell_info confirms local_context_archive');
        await nextTurn();
        assert.equal(probes, 1);
        resolveShell({ local_context_archive: true });
        await nextTurn();
        assert.equal(mount.hidden, false);
        const button = mount.querySelector('button');
        assert.equal(button.textContent, 'Open recent requests');
        assert.ok(button.classList.contains('btn-default'), 'a neutral capsule, as Clear beside it');
        // Its result is the shared result line right after it, read out as it changes.
        const line = mount.querySelector('[role="status"]');
        assert.equal(button.nextSibling, line);
        assert.ok(line.classList.contains('ui-result'));
        assert.equal(line.getAttribute('aria-live'), 'polite');
        button.click();
        await nextTurn();
        assert.equal(opens, 1);
        assert.equal(line.textContent, 'Opened in a separate local window.');
        assert.equal(line.dataset.tone, 'ok');

        dispose();
        assert.equal(mount.hidden, true);
        assert.equal(mount.children.length, 0);
        assert.equal(line.textContent, '', 'leaving ends the result with its timer');

        // Non-WebView2 shell (local_context_archive: false) keeps the affordance hidden.
        const secondMount = document.createElement('div');
        const disposeSecond = bindLocalContextReport(secondMount);
        assert.equal(window.listeners('pywebviewready'), 0, 'bridge was already present');
        await nextTurn();
        resolveShell({ local_context_archive: false });
        await nextTurn();
        assert.equal(secondMount.hidden, true, 'a non-WebView2 shell cannot expose archive reading');
        disposeSecond();
    } finally {
        dom.restore();
    }
});

test('status messages cover already_open and unavailable responses cleanly', async () => {
    const dom = installCardDom();
    try {
        let reply = { state: 'already_open' };
        window.pywebview = { api: {
            shell_info: async () => ({ local_context_archive: true }),
            open_local_context_report: async () => {
                if (reply instanceof Error) throw reply;
                return reply;
            },
        } };
        const mount = document.createElement('div');
        const dispose = bindLocalContextReport(mount);
        await nextTurn();
        assert.equal(mount.hidden, false);
        const line = mount.querySelector('[role="status"]');

        mount.querySelector('button').click();
        await nextTurn();
        assert.equal(line.textContent, 'The local viewer is already open.');
        assert.equal(line.dataset.tone, 'muted');

        reply = { state: 'unavailable' };
        mount.querySelector('button').click();
        await nextTurn();
        assert.equal(line.textContent, 'The local viewer is unavailable in this window.');
        assert.equal(line.dataset.tone, 'warn');

        // A state this page does not know, an inherited name among them, is the viewer being unavailable.
        for (const state of ['constructor', 'opening', undefined]) {
            reply = { state };
            mount.querySelector('button').click();
            await nextTurn();
            assert.equal(line.textContent, 'The local viewer is unavailable in this window.', String(state));
            assert.equal(line.dataset.tone, 'warn');
        }

        reply = new Error('bridge disconnected');
        mount.querySelector('button').click();
        await nextTurn();
        assert.equal(line.textContent, 'The local viewer is unavailable in this window.');
        assert.equal(line.dataset.tone, 'warn');
        assert.equal(mount.querySelector('button').disabled, false);

        dispose();
    } finally {
        dom.restore();
    }
});

test('an acknowledgement ends by itself; an unavailable viewer stays until the next press', async () => {
    const dom = installCardDom();
    const prior = { setTimeout: globalThis.setTimeout, clearTimeout: globalThis.clearTimeout };
    const timers = new Map();
    let seq = 0;
    globalThis.setTimeout = (fn, ms) => { seq += 1; timers.set(seq, { fn, ms }); return seq; };
    globalThis.clearTimeout = (id) => { timers.delete(id); };
    const runTimers = () => { const due = [...timers.values()]; timers.clear(); due.forEach(({ fn }) => fn()); };
    try {
        let reply = { state: 'opened' };
        window.pywebview = { api: {
            shell_info: async () => ({ local_context_archive: true }),
            open_local_context_report: async () => reply,
        } };
        const mount = document.createElement('div');
        const dispose = bindLocalContextReport(mount);
        await nextTurn();
        const button = mount.querySelector('button');
        const line = mount.querySelector('[role="status"]');

        button.click();
        await nextTurn();
        assert.equal(line.textContent, 'Opened in a separate local window.');
        assert.deepEqual([...timers.values()].map(({ ms }) => ms), [SAVE_ACK_MS]);
        runTimers();
        assert.equal(line.textContent, '');
        assert.equal(line.dataset.tone, undefined);

        reply = { state: 'unavailable' };
        button.click();
        await nextTurn();
        assert.equal(timers.size, 0, 'a warning holds no timer');
        assert.equal(line.textContent, 'The local viewer is unavailable in this window.');

        // The next press replaces it; leaving ends a pending acknowledgement and its line.
        reply = { state: 'already_open' };
        button.click();
        await nextTurn();
        assert.equal(line.textContent, 'The local viewer is already open.');
        assert.equal(timers.size, 1);
        dispose();
        assert.equal(timers.size, 0);
        assert.equal(line.textContent, '');
    } finally {
        Object.assign(globalThis, prior);
        dom.restore();
    }
});

test('dispose detaches pywebviewready once and blocks late capability or open callbacks', async () => {
    const dom = installCardDom();
    try {
        const removed = [];
        const remove = window.removeEventListener;
        window.removeEventListener = (type, handler) => { removed.push(type); return remove(type, handler); };

        // Disposing before pywebviewready detaches the listener once; a repeated dispose is a no-op.
        const waitingMount = document.createElement('div');
        const disposeWaiting = bindLocalContextReport(waitingMount);
        assert.equal(window.listeners('pywebviewready'), 1);
        disposeWaiting();
        disposeWaiting();
        assert.equal(window.listeners('pywebviewready'), 0);
        assert.deepEqual(removed, ['pywebviewready']);

        // Late shell_info or detached button click after dispose cannot unhide or invoke native open.
        let resolveShell, finishOpen, opens = 0;
        window.pywebview = { api: {
            shell_info: () => new Promise((resolve) => { resolveShell = resolve; }),
            open_local_context_report: () => { opens++; return new Promise((resolve) => { finishOpen = resolve; }); },
        } };
        const mount = document.createElement('div');
        const dispose = bindLocalContextReport(mount);
        const button = mount.querySelector('button');
        await nextTurn();
        dispose();
        resolveShell({ local_context_archive: true });
        button.click();
        await nextTurn();
        assert.equal(mount.hidden, true);
        assert.equal(mount.children.length, 0);
        assert.equal(opens, 0);

        // An in-flight native open that resolves after dispose cannot touch status or re-enable.
        const second = document.createElement('div');
        const disposeSecond = bindLocalContextReport(second);
        await nextTurn();
        resolveShell({ local_context_archive: true });
        await nextTurn();
        assert.equal(second.hidden, false);
        const secondButton = second.querySelector('button');
        const secondStatus = second.querySelector('[role="status"]');
        secondButton.click();
        await nextTurn();
        assert.equal(typeof finishOpen, 'function');
        disposeSecond();
        finishOpen({ state: 'opened' });
        await nextTurn();
        assert.equal(second.hidden, true);
        assert.equal(secondStatus.textContent, '');
        assert.equal(secondStatus.dataset.tone, undefined);
    } finally {
        dom.restore();
    }
});
