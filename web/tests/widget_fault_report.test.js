// A running module widget's fault (widget_module.js): the card or the living card shows it as before, and the page
// sends it once to the host log (POST /api/widgets/fault) under the mount's own identity, never the frame's words.
// Only the mounted frame with its nonce is heard; a kind+message is sent once and a mount sends at most ten, since
// the widget's code can post with the nonce directly; a failed report changes nothing on the card. A frame in its
// ordered stop still reports; once disposed it is not heard. Real engines are checked in the browser lane.
import assert from 'node:assert/strict';
import test from 'node:test';

import { getLogTaskGroupId, isGroupedTaskEvent } from '../modules/log_events.js';
import { classifyLiveFrame } from '../modules/notifications.js';
import { mountModuleWidget, WIDGET_FAULT_REPORTS_PER_MOUNT } from '../modules/widget_module.js';
import { installCardDom } from './helpers/card_dom.js';
import { joinFrame } from './helpers/fake_channel.js';

const flush = async () => { for (let i = 0; i < 8; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
const TAB = { key: 'clock:face', skill: 'clock', tab_id: 'face', revision: 'rev-1', ws_prefix: '', render: { kind: 'module', entry: 'widget.js' } };

function serve({ fault = () => ({ ok: true, status: 200, json: async () => ({ ok: true, logged: true }) }) } = {}) {
    const reports = [];
    const prior = globalThis.fetch;
    globalThis.fetch = async (url, init = {}) => {
        if (url === '/api/widgets/fault') {
            reports.push(JSON.parse(init.body));
            return fault();
        }
        // The classic mount's route lease (its frame's own media loads), then the reviewed source.
        if (url === '/api/widgets/asset-leases') return { ok: true, status: 200, json: async () => ({ token: 'route-lease', base_url: '/api/widget-resources/route-lease/', expires_in: 120, renew_after: 40 }) };
        return { ok: true, status: 200, text: async () => '// reviewed module', json: async () => ({}) };
    };
    return { reports, restore: () => { globalThis.fetch = prior; } };
}

// A running classic mount renews its route lease; each test stops its mounts (with the frame's acknowledgement)
// before the fake page goes away.
const stops = [];
async function stopMounts() {
    while (stops.length) await stops.pop()();
}

async function mounted({ inline = null } = {}) {
    const card = document.createElement('div');
    card.setAttribute('data-widget-key', TAB.key);
    card.innerHTML = '<div data-widget-status hidden></div><div data-widget-mount></div>';
    const mount = card.querySelector('[data-widget-mount]');
    const dispose = await mountModuleWidget(mount, TAB, TAB.render, null, null, inline);
    const frame = mount.querySelector('iframe');
    frame.contentWindow = { postMessage() {} };
    const joined = joinFrame(frame, (event) => window.emit('message', event));
    const nonce = joined.nonce;
    // The frame speaks on its channel; another window can only post a window message, which is not heard.
    const send = (data, { source = frame.contentWindow, ownNonce = nonce } = {}) => (source === frame.contentWindow
        ? joined.channel.port1.postMessage({ nonce: ownNonce, ...data })
        : window.emit('message', { source, data: { nonce: ownNonce, ...data } }));
    const fault = (fields) => send({ type: 'ouro-widget-error', kind: 'error', message: 'boom', source: 'about:srcdoc', line: 3, ...fields });
    stops.push(async () => { const stopped = dispose(); send({ type: 'ouro-widget-disposed' }); await stopped; });
    return { card, status: card.querySelector('[data-widget-status]'), dispose, send, fault };
}

test('a fault on the board is shown and sent once under the mount\'s identity, not the frame\'s', async () => {
    const dom = installCardDom();
    const server = serve();
    try {
        const { status, fault } = await mounted();
        fault({ skill: 'bank', tab_id: 'vault', revision: 'forged', where: 'message', task_id: 'task-1' });
        assert.equal(status.textContent, 'Widget script error: boom');
        await flush();
        assert.deepEqual(server.reports, [{ skill: 'clock', tab_id: 'face', revision: 'rev-1', where: 'board',
            kind: 'error', message: 'boom', source: 'about:srcdoc', line: 3 }]);
        fault({});
        await flush();
        assert.equal(server.reports.length, 1, 'the same kind+message is sent once');
    } finally {
        await stopMounts();
        server.restore();
        dom.restore();
    }
});

test('another window or another nonce is not heard', async () => {
    const dom = installCardDom();
    const server = serve();
    try {
        const { status, send } = await mounted();
        send({ type: 'ouro-widget-error', kind: 'error', message: 'stranger', source: '', line: 0 }, { source: { postMessage() {} } });
        send({ type: 'ouro-widget-error', kind: 'error', message: 'stale', source: '', line: 0 }, { ownNonce: 'other' });
        await flush();
        assert.equal(status.hidden, true);
        assert.deepEqual(server.reports, []);
    } finally {
        await stopMounts();
        server.restore();
        dom.restore();
    }
});

test('a mount sends at most ten faults, clipped and typed before they are serialized; the card keeps showing them', async () => {
    const dom = installCardDom();
    const server = serve();
    try {
        const { status, fault } = await mounted();
        fault({ kind: 'script', message: 'not a frame kind' });
        assert.equal(status.textContent, 'Widget script error: not a frame kind', 'shown as before, not sent');
        fault({ message: 'm'.repeat(600), source: 's'.repeat(300), line: 1.5 });
        fault({ kind: 'rejection', message: { nested: 'object' }, source: 7, line: -1 });
        for (let i = 0; i < 25; i += 1) fault({ kind: 'csp', message: `burst ${i}`, line: i });
        assert.equal(status.textContent, 'Widget blocked by frame policy: about:srcdoc');
        await flush();
        assert.equal(server.reports.length, WIDGET_FAULT_REPORTS_PER_MOUNT);
        assert.deepEqual(server.reports[0], { skill: 'clock', tab_id: 'face', revision: 'rev-1', where: 'board',
            kind: 'error', message: 'm'.repeat(500), source: 's'.repeat(200), line: 0 });
        assert.deepEqual([server.reports[1].kind, server.reports[1].message, server.reports[1].source, server.reports[1].line],
            ['rejection', '', '', 0]);
        assert.equal(server.reports.at(-1).message, 'burst 7');
    } finally {
        await stopMounts();
        server.restore();
        dom.restore();
    }
});

test('a failed report changes nothing: the card shows every fault and nothing is retried', async () => {
    const dom = installCardDom();
    let refusals = 0;
    const server = serve({ fault: () => { refusals += 1; if (refusals === 1) throw new TypeError('Failed to fetch'); return { ok: false, status: 503, json: async () => ({ error: 'not written' }) }; } });
    const unhandled = [];
    const onUnhandled = (reason) => unhandled.push(reason);
    process.on('unhandledRejection', onUnhandled);
    try {
        const { status, fault } = await mounted();
        fault({ message: 'first' });
        fault({ message: 'second' });
        await flush();
        assert.equal(status.textContent, 'Widget script error: second');
        assert.equal(status.dataset.tone, 'error');
        assert.deepEqual(server.reports.map((report) => report.message), ['first', 'second'], 'one attempt each');
        assert.deepEqual(unhandled, []);
    } finally {
        process.off('unhandledRejection', onUnhandled);
        server.restore();
        dom.restore();
    }
});

test('a living card in a message shows its fault through its hook and reports it as where: message', async () => {
    const dom = installCardDom();
    const server = serve();
    try {
        const shown = [];
        const { status, fault } = await mounted({ inline: { fit() {}, fault: (text) => shown.push(text) } });
        fault({});
        await flush();
        assert.deepEqual(shown, ['Widget script error: boom']);
        assert.equal(status.hidden, true, 'the board card slot is not the message\'s');
        assert.equal(server.reports[0].where, 'message');
    } finally {
        await stopMounts();
        server.restore();
        dom.restore();
    }
});

test('a hidden card\'s frame still reports, through its ordered stop; once disposed it is not heard', async () => {
    const dom = installCardDom();
    const server = serve();
    try {
        const { card, dispose, send, fault } = await mounted();
        card.hidden = true;  // Keep running: the card is retained while Widgets is not shown
        fault({ message: 'while hidden' });
        const stopping = dispose();
        fault({ message: 'during the stop' });
        send({ type: 'ouro-widget-disposed' });
        await stopping;
        fault({ message: 'after the stop' });
        await flush();
        assert.deepEqual(server.reports.map((report) => report.message), ['while hidden', 'during the stop']);
        assert.equal(window.listeners('message'), 0);
    } finally {
        await stopMounts();
        server.restore();
        dom.restore();
    }
});

test('the logged row reaches a page as a task-less log frame: no card and no notification read it', () => {
    // The row's shape as gateway/widgets.py writes it and the server's live sink pushes it (chat_id 0, no task).
    const row = { ts: '2026-10-21T10:00:00+00:00', type: 'widget_fault', reported_by: 'page', skill: 'clock', tab_id: 'face',
        revision: 'rev-1', where: 'board', kind: 'error', message: 'boom', source: 'about:srcdoc', line: 3 };
    assert.equal(getLogTaskGroupId(row) || '', '');
    assert.equal(isGroupedTaskEvent(row), false, 'chat.js mints or updates a work card only for a task event');
    assert.equal(classifyLiveFrame(row, { kind: 'log', isMain: true }), null);
});

test('a fault naming a file or route lease URL shows and reports it redacted', async () => {
    const dom = installCardDom();
    const server = serve();
    try {
        const { status, fault } = await mounted();
        const lease = 'A'.repeat(43);
        // A blocked redirect hop under the classic frame's route lease (the CSP report names the lease URL).
        fault({ kind: 'csp', message: 'img-src', source: `http://local/api/widget-resources/${lease}/go-foreign` });
        assert.equal(status.textContent, 'Widget blocked by frame policy: http://local/api/widget-resources/[redacted]/go-foreign');
        fault({ message: `failed /api/widget-assets/${lease}/app/main.js` });
        await flush();
        assert.equal(JSON.stringify(server.reports).includes(lease), false);
        assert.equal(server.reports[0].source, 'http://local/api/widget-resources/[redacted]/go-foreign');
        assert.equal(server.reports[1].message, 'failed /api/widget-assets/[redacted]/app/main.js');
    } finally {
        await stopMounts();
        server.restore();
        dom.restore();
    }
});
