// The native archive door is Logs-only. Test bridge availability and the state-only answer separately from
// layout and real input, which the Windows WebView2 journey measures.
import assert from 'node:assert/strict';
import test from 'node:test';

import { bindLocalContextReport } from '../modules/context_report_host.js';
import { SAVE_ACK_MS } from '../modules/ui_primitives.js';
import { setup } from './dashboard_dom_fixture.js';

const nextTurn = () => new Promise((resolve) => setImmediate(resolve));
const currentToast = (doc) => doc.body.querySelector('.toast');
const toastWords = (doc) => currentToast(doc)?.querySelector('.toast-words').textContent;
const notifyReady = (win) => win.dispatchEvent({ type: 'pywebviewready' });
const readyListeners = (win) => win.events.get('pywebviewready')?.length || 0;

test('native action is offered once, and only shown on a positive shell fact', async (t) => {
    const { mount, window, document } = setup(t);
    const { dispose } = bindLocalContextReport(mount);
    notifyReady(window);
    assert.equal(mount.hidden, true, 'a browser keeps the mount hidden');
    assert.equal(mount.querySelector('button'), null);

    let resolveShell, opens = 0, probes = 0;
    window.pywebview = { api: {
        shell_info: () => { probes++; return new Promise((resolve) => { resolveShell = resolve; }); },
        open_local_context_report: async () => { opens++; return { state: 'opened' }; },
    } };
    notifyReady(window);
    notifyReady(window);
    assert.equal(readyListeners(window), 0);
    assert.equal(mount.querySelectorAll('button').length, 1);
    assert.equal(mount.hidden, true, 'the action stays hidden until the bridge confirms its capability');
    await nextTurn();
    assert.equal(probes, 1);
    resolveShell({ local_context_archive: true });
    await nextTurn();
    assert.equal(mount.hidden, false);
    const button = mount.querySelector('button');
    assert.equal(button.textContent, 'Open recent requests');
    assert.ok(button.classList.contains('btn-default'));
    await button.fire('click');
    assert.equal(opens, 1);
    assert.equal(mount.children.length, 1, 'no answer takes width from the header');
    assert.equal(toastWords(document), 'Opened in a separate local window.');
    assert.equal(currentToast(document).getAttribute('role'), 'status');
    assert.ok(currentToast(document).classList.contains('toast-ok'));

    dispose();
    assert.equal(mount.hidden, true);
    assert.equal(mount.children.length, 0);
    assert.equal(currentToast(document), null);

    const second = document.createElement('div');
    const { dispose: disposeSecond, retry } = bindLocalContextReport(second);
    await nextTurn();
    resolveShell({ local_context_archive: false });
    await nextTurn();
    assert.equal(second.hidden, true, 'a non-WebView2 shell cannot expose archive reading');
    retry();
    await nextTurn();
    assert.equal(probes, 2, 'and its no is final: the next showing of Logs asks nothing more');
    disposeSecond();
});

test('the latest toast reflects the native answer; an unavailable answer has a keyboard close', async (t) => {
    const { mount, window, document } = setup(t);
    let reply = { state: 'already_open' };
    window.pywebview = { api: {
        shell_info: async () => ({ local_context_archive: true }),
        open_local_context_report: async () => {
            if (reply instanceof Error) throw reply;
            return reply;
        },
    } };
    const { dispose } = bindLocalContextReport(mount);
    await nextTurn();
    const button = mount.querySelector('button');
    await button.fire('click');
    assert.equal(toastWords(document), 'The local viewer is already open.');
    assert.ok(currentToast(document).classList.contains('toast-muted'));

    for (const outcome of [{ state: 'unavailable' }, { state: 'constructor' }, { state: 'opening' }, {}, new Error()]) {
        reply = outcome;
        await button.fire('click');
        assert.equal(toastWords(document), 'The local viewer is unavailable in this window.');
        assert.ok(currentToast(document).classList.contains('toast-warn'));
        assert.ok(currentToast(document).querySelector('.toast-dismiss').getAttribute('aria-label'));
        assert.equal(document.body.querySelectorAll('.toast').length, 1, 'replacement does not pile up toasts');
        assert.equal(button.disabled, false);
    }
    dispose();
    assert.equal(currentToast(document), null);
});

test('acknowledgements use the shared timeout and warnings do not start one', async (t) => {
    const { mount, window, document } = setup(t);
    const prior = { setTimeout: globalThis.setTimeout, clearTimeout: globalThis.clearTimeout };
    const timers = new Map();
    let seq = 0;
    globalThis.setTimeout = (fn, ms) => { seq += 1; timers.set(seq, { fn, ms }); return seq; };
    globalThis.clearTimeout = (id) => { timers.delete(id); };
    try {
        let reply = { state: 'opened' };
        window.pywebview = { api: {
            shell_info: async () => ({ local_context_archive: true }),
            open_local_context_report: async () => reply,
        } };
        const { dispose } = bindLocalContextReport(mount);
        await nextTurn();
        const button = mount.querySelector('button');
        await button.fire('click');
        assert.deepEqual([...timers.values()].map(({ ms }) => ms), [SAVE_ACK_MS]);

        reply = { state: 'unavailable' };
        await button.fire('click');
        assert.equal(timers.size, 0);
        assert.ok(currentToast(document));

        reply = { state: 'already_open' };
        await button.fire('click');
        assert.equal(timers.size, 1);
        dispose();
        assert.equal(timers.size, 0);
        assert.equal(currentToast(document), null);
    } finally {
        Object.assign(globalThis, prior);
    }
});

test('dispose detaches readiness and blocks late capability, detached clicks and pending open answers', async (t) => {
    const { mount, window, document } = setup(t);
    const { dispose: disposeWaiting } = bindLocalContextReport(mount);
    assert.equal(readyListeners(window), 1);
    disposeWaiting();
    disposeWaiting();
    assert.equal(readyListeners(window), 0);

    let resolveShell, finishOpen, opens = 0;
    window.pywebview = { api: {
        shell_info: () => new Promise((resolve) => { resolveShell = resolve; }),
        open_local_context_report: () => { opens++; return new Promise((resolve) => { finishOpen = resolve; }); },
    } };
    const { dispose } = bindLocalContextReport(mount);
    const button = mount.querySelector('button');
    await nextTurn();
    dispose();
    resolveShell({ local_context_archive: true });
    await button.fire('click');
    await nextTurn();
    assert.equal(mount.hidden, true);
    assert.equal(mount.children.length, 0);
    assert.equal(opens, 0);

    const second = document.createElement('div');
    const { dispose: disposeSecond } = bindLocalContextReport(second);
    await nextTurn();
    resolveShell({ local_context_archive: true });
    await nextTurn();
    const secondButton = second.querySelector('button');
    assert.equal(second.hidden, false);
    const opened = secondButton.fire('click');
    assert.equal(typeof finishOpen, 'function');
    disposeSecond();
    finishOpen({ state: 'opened' });
    await opened;
    assert.equal(second.hidden, true);
    assert.equal(secondButton.disabled, true, 'the late callback cannot re-enable a detached button');
    assert.equal(currentToast(document), null);
});
