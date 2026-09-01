import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
    classifyShellUrl,
    copyTextWithToast,
    downloadViaHostBridge,
    installDesktopShellLinkInterceptor,
    openViaHostBridge,
} from '../modules/ui_helpers.js';

const BASE = 'http://127.0.0.1:8765/';

// --- classifier -----------------------------------------------------------

test('classifyShellUrl routes loopback file forms to the bridge (path-only url)', () => {
    assert.deepEqual(
        classifyShellUrl('/api/files/download?path=docs/report.pdf', BASE),
        { kind: 'file', url: '/api/files/download?path=docs/report.pdf' },
    );
    assert.deepEqual(
        classifyShellUrl(`${BASE}api/tasks/t-1/artifacts/chat-media-abc.png`, BASE),
        { kind: 'file', url: '/api/tasks/t-1/artifacts/chat-media-abc.png' },
    );
    assert.deepEqual(
        classifyShellUrl('/api/extensions/skill/asset.png', BASE),
        { kind: 'file', url: '/api/extensions/skill/asset.png' },
    );
});

test('classifyShellUrl routes any other http(s) target to the external opener', () => {
    assert.deepEqual(
        classifyShellUrl('https://example.com/page', BASE),
        { kind: 'external', url: 'https://example.com/page' },
    );
    // Same-origin non-file pages have no tab inside the shell either.
    assert.deepEqual(
        classifyShellUrl('/dashboard', BASE),
        { kind: 'external', url: `${BASE}dashboard` },
    );
    // Different port on the loopback host is NOT the file bridge.
    assert.equal(classifyShellUrl('http://127.0.0.1:9999/api/files/download?path=a', BASE).kind, 'external');
    // mailto: is a real chat-link surface and belongs to the OS default handler.
    assert.deepEqual(
        classifyShellUrl('mailto:owner@example.com', BASE),
        { kind: 'external', url: 'mailto:owner@example.com' },
    );
});

test('classifyShellUrl routes data:/blob: payloads to byte saving', () => {
    assert.equal(classifyShellUrl('data:image/png;base64,AAAA', BASE).kind, 'bytes');
    assert.equal(classifyShellUrl(`blob:${BASE}0-1-2`, BASE).kind, 'bytes');
});

test('classifyShellUrl leaves everything else to the default handler', () => {
    assert.equal(classifyShellUrl('', BASE).kind, 'default');
    assert.equal(classifyShellUrl('javascript:void(0)', BASE).kind, 'default');
    assert.equal(classifyShellUrl('#anchor', '').kind, 'default');
});

// --- harness --------------------------------------------------------------

function makeAnchor(attrs) {
    return {
        getAttribute: (name) => (Object.hasOwn(attrs, name) ? attrs[name] : null),
        hasAttribute: (name) => Object.hasOwn(attrs, name),
    };
}

function makeEvent(anchor) {
    return {
        defaultPrevented: false,
        preventDefault() { this.defaultPrevented = true; },
        target: { closest: (selector) => (selector === 'a[href]' ? anchor : null) },
    };
}

function makeHarness({ api, pywebview = true, parent = null, openFileImpl = null, downloadFileImpl = null } = {}) {
    const calls = { open: [], copied: [], toasts: [], openFile: [], downloadFile: [] };
    const docListeners = {};
    const winListeners = {};
    const doc = {
        addEventListener(type, fn) { (docListeners[type] ||= []).push(fn); },
        createElement: () => ({ setAttribute() {}, select() {}, remove() {}, value: '' }),
        body: { appendChild() {} },
        execCommand: () => true,
    };
    const win = {
        location: { href: BASE },
        navigator: { clipboard: { writeText: async (text) => { calls.copied.push(text); } } },
        addEventListener(type, fn) { (winListeners[type] ||= []).push(fn); },
        open(...args) { calls.open.push(args); return 'native-window'; },
    };
    if (pywebview) win.pywebview = { api: api || {} };
    // A top-level window is its own parent; a framed document gets a fake one.
    win.parent = parent || win;
    installDesktopShellLinkInterceptor({
        win,
        doc,
        toast: (message, tone) => calls.toasts.push({ message, tone }),
        openFile: openFileImpl || (async (url, name) => { calls.openFile.push([url, name]); }),
        downloadFile: downloadFileImpl || (async (url, name) => { calls.downloadFile.push([url, name]); }),
    });
    const click = async (anchor) => {
        const event = makeEvent(anchor);
        for (const fn of docListeners.click || []) fn(event);
        await new Promise((resolve) => setTimeout(resolve, 0));
        return event;
    };
    return { win, doc, calls, docListeners, winListeners, click };
}

const FILE_API = { open_file_with_default_app: async () => ({ ok: true }), download_file_to_downloads: async () => ({ ok: true }) };

// --- install gating -------------------------------------------------------

test('browser mode installs nothing: no click listener, native window.open kept', () => {
    const hx = makeHarness({ pywebview: false });
    assert.equal(hx.docListeners.click, undefined, 'no delegated click listener');
    assert.equal(hx.win.open('https://example.com', '_blank'), 'native-window');
    assert.equal(hx.calls.open.length, 1, 'window.open is the untouched native one');
    assert.equal((hx.winListeners.pywebviewready || []).length, 1, 'armed only for a late bridge');
});

test('a late pywebviewready announcement installs the interceptor', async () => {
    const hx = makeHarness({ pywebview: false });
    const external = [];
    hx.win.pywebview = { api: { open_external_url: async (url) => { external.push(url); return { ok: true }; } } };
    for (const fn of hx.winListeners.pywebviewready) fn();
    assert.equal(hx.docListeners.click.length, 1, 'delegated listener installed on the ready event');
    const event = await hx.click(makeAnchor({ href: 'https://example.com/x', target: '_blank' }));
    assert.equal(event.defaultPrevented, true);
    assert.deepEqual(external, ['https://example.com/x']);
});

// --- framed wizard document (two-document pattern) ------------------------

test('a framed document resolves the bridge from the PARENT window: sign-in link routes external', async () => {
    const external = [];
    const parent = {
        pywebview: { api: { open_external_url: async (url) => { external.push(url); return { ok: true }; } } },
        addEventListener() {},
    };
    const hx = makeHarness({ pywebview: false, parent });
    assert.equal(hx.docListeners.click.length, 1, 'parent bridge present at load: installed immediately');
    // The Agents step's primary action is exactly this anchor shape.
    const event = await hx.click(makeAnchor({ href: 'https://vendor.example/oauth/device', target: '_blank', rel: 'noopener' }));
    assert.equal(event.defaultPrevented, true);
    assert.deepEqual(external, ['https://vendor.example/oauth/device']);
});

test('a late pywebviewready on the PARENT window installs the framed interceptor', async () => {
    const parentListeners = {};
    const parent = { addEventListener(type, fn) { (parentListeners[type] ||= []).push(fn); } };
    const hx = makeHarness({ pywebview: false, parent });
    assert.equal(hx.docListeners.click, undefined, 'nothing installed before the bridge exists');
    assert.equal(parentListeners.pywebviewready.length, 1, 'armed on the top-level window too');
    const external = [];
    parent.pywebview = { api: { open_external_url: async (url) => { external.push(url); return { ok: true }; } } };
    for (const fn of parentListeners.pywebviewready) fn();
    assert.equal(hx.docListeners.click.length, 1);
    const event = await hx.click(makeAnchor({ href: 'https://vendor.example/signin', target: '_blank' }));
    assert.equal(event.defaultPrevented, true);
    assert.deepEqual(external, ['https://vendor.example/signin']);
});

test('the onboarding wizard document installs the shell interceptor (source pin)', () => {
    const source = readFileSync(new URL('../modules/onboarding_wizard.js', import.meta.url), 'utf8');
    assert.match(source, /installDesktopShellLinkInterceptor\(\);/,
        'the wizard is a separate document and must install its own interceptor');
    assert.match(source, /import \{ installAltMenuSuppression, installDesktopShellLinkInterceptor \} from '\.\/ui_helpers\.js';/);
});

// --- click routing --------------------------------------------------------

test('external link rides open_external_url; a settled bridge failure degrades to copy-link', async () => {
    const external = [];
    let hx = makeHarness({ api: { open_external_url: async (url) => { external.push(url); return { ok: true }; } } });
    const event = await hx.click(makeAnchor({ href: 'https://example.com/docs', target: '_blank' }));
    assert.equal(event.defaultPrevented, true);
    assert.deepEqual(external, ['https://example.com/docs']);
    assert.deepEqual(hx.calls.toasts, [], 'success is silent — the browser opening IS the feedback');

    // {ok:false} = the host PROVED no browser could be launched: same honest
    // hand-the-owner-the-link degradation as a launcher without the method.
    hx = makeHarness({ api: { open_external_url: async () => ({ ok: false, error: 'no handler found' }) } });
    await hx.click(makeAnchor({ href: 'https://example.com/x', target: '_blank' }));
    assert.deepEqual(hx.calls.copied, ['https://example.com/x']);
    assert.deepEqual(hx.calls.toasts, [{ message: 'Link copied — open it in your browser.', tone: 'info' }]);
});

test('mailto links route through the external opener like any OS-handled link', async () => {
    const external = [];
    const hx = makeHarness({ api: { open_external_url: async (url) => { external.push(url); return { ok: true }; } } });
    const event = await hx.click(makeAnchor({ href: 'mailto:owner@example.com', target: '_blank' }));
    assert.equal(event.defaultPrevented, true);
    assert.deepEqual(external, ['mailto:owner@example.com']);
});

test('old launcher without open_external_url copies the link with an honest toast', async () => {
    const hx = makeHarness({ api: {} });
    const event = await hx.click(makeAnchor({ href: 'https://example.com/release', target: '_blank' }));
    assert.equal(event.defaultPrevented, true);
    assert.deepEqual(hx.calls.copied, ['https://example.com/release']);
    assert.deepEqual(hx.calls.toasts, [{ message: 'Link copied — open it in your browser.', tone: 'info' }]);
});

test('base64 data: payload rides save_bytes_to_downloads with a mime-derived name', async () => {
    const saved = [];
    const hx = makeHarness({
        api: { save_bytes_to_downloads: async (name, b64) => { saved.push([name, b64]); return { ok: true, path: '/home/o/Downloads/download.png' }; } },
    });
    const event = await hx.click(makeAnchor({ href: 'data:image/png;base64,AAECAw==', download: '' }));
    assert.equal(event.defaultPrevented, true);
    assert.deepEqual(saved, [['download.png', 'AAECAw==']]);
    assert.deepEqual(hx.calls.toasts, [{ message: 'Saved to Downloads: download.png', tone: 'ok' }]);
});

test('a download-named anchor keeps its filename for byte saves', async () => {
    const saved = [];
    const hx = makeHarness({
        api: { save_bytes_to_downloads: async (name, b64) => { saved.push([name, b64]); return { ok: true }; } },
    });
    await hx.click(makeAnchor({ href: 'data:text/plain;base64,aGk=', download: 'notes.txt' }));
    assert.deepEqual(saved, [['notes.txt', 'aGk=']]);
});

test('text/plain data: payload saves with the conventional .txt extension', async () => {
    const saved = [];
    const hx = makeHarness({
        api: { save_bytes_to_downloads: async (name, b64) => { saved.push([name, b64]); return { ok: true } } },
    });
    await hx.click(makeAnchor({ href: 'data:text/plain;base64,aGk=', download: '' }));
    assert.deepEqual(saved, [['download.txt', 'aGk=']]);
});

test('bridge failures toast with the verb of what actually failed', async () => {
    let hx = makeHarness({ api: FILE_API, downloadFileImpl: async () => { throw new Error('guard refused'); } });
    await hx.click(makeAnchor({ href: '/api/files/download?path=a.txt', download: '' }));
    assert.deepEqual(hx.calls.toasts, [{ message: 'Could not download file: guard refused', tone: 'error' }]);

    hx = makeHarness({ api: FILE_API, openFileImpl: async () => { throw new Error('guard refused'); } });
    await hx.click(makeAnchor({ href: '/api/tasks/t-1/artifacts/chat-media-aa.png', target: '_blank' }));
    assert.deepEqual(hx.calls.toasts, [{ message: 'Could not open file: guard refused', tone: 'error' }]);
});

test('old launcher without save_bytes_to_downloads toasts that saving is unavailable', async () => {
    const hx = makeHarness({ api: {} });
    const event = await hx.click(makeAnchor({ href: 'data:image/png;base64,AAAA', download: '' }));
    assert.equal(event.defaultPrevented, true);
    assert.deepEqual(hx.calls.toasts, [{ message: "Saving isn't available in the app — open in a browser.", tone: 'warn' }]);
});

test('loopback artifact link opens via the host bridge; download attr downloads instead', async () => {
    const hx = makeHarness({ api: FILE_API });
    const openEvent = await hx.click(makeAnchor({ href: '/api/tasks/t-9/artifacts/chat-media-ff.png', target: '_blank' }));
    assert.equal(openEvent.defaultPrevented, true);
    assert.deepEqual(hx.calls.openFile, [['/api/tasks/t-9/artifacts/chat-media-ff.png', 'chat-media-ff.png']]);

    await hx.click(makeAnchor({ href: '/api/files/download?path=logs/run.txt', download: '' }));
    assert.deepEqual(hx.calls.downloadFile, [['/api/files/download?path=logs/run.txt', 'run.txt']]);
});

test('without ANY file bridge method the file class degrades to copy-link (no loop, no silence)', async () => {
    const hx = makeHarness({ api: {} });
    const event = await hx.click(makeAnchor({ href: '/api/files/download?path=a.txt', target: '_blank' }));
    assert.equal(event.defaultPrevented, true, 'the dead native default is never left in place');
    assert.deepEqual(hx.calls.openFile, [], 'helpers skipped: they would fall back into the shim');
    assert.deepEqual(hx.calls.downloadFile, []);
    assert.deepEqual(hx.calls.copied, [`${BASE}api/files/download?path=a.txt`], 'copied as an absolute URL');
    assert.deepEqual(hx.calls.toasts, [{ message: 'Link copied — open it in your browser.', tone: 'info' }]);
});

test('ordinary same-tab anchors and non-anchor clicks stay untouched', async () => {
    const hx = makeHarness({ api: { open_external_url: async () => ({ ok: true }) } });
    const plain = await hx.click(makeAnchor({ href: 'https://example.com/inline' }));
    assert.equal(plain.defaultPrevented, false, 'no target=_blank and no download attribute');
    const event = makeEvent(null);
    for (const fn of hx.docListeners.click) fn(event);
    assert.equal(event.defaultPrevented, false);
});

// --- window.open shim -----------------------------------------------------

test('window.open shim routes external URLs over the bridge and returns null', async () => {
    const external = [];
    const hx = makeHarness({ api: { open_external_url: async (url) => { external.push(url); return { ok: true }; } } });
    const result = hx.win.open('https://example.com/photo', '_blank', 'noopener');
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(result, null);
    assert.deepEqual(external, ['https://example.com/photo']);
    assert.deepEqual(hx.calls.open, [], 'native open was not reached');
});

test('window.open shim opens durable media via the file bridge and passes the rest through', async () => {
    const hx = makeHarness({ api: FILE_API });
    assert.equal(hx.win.open('/api/tasks/t-1/artifacts/chat-media-aa.png', '_blank', 'noopener'), null);
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.deepEqual(hx.calls.openFile, [['/api/tasks/t-1/artifacts/chat-media-aa.png', 'chat-media-aa.png']]);
    // Unclassifiable targets keep the native behavior.
    assert.equal(hx.win.open('javascript:void(0)'), 'native-window');
    assert.equal(hx.calls.open.length, 1);
});

test('window.open shim degrades file URLs to copy-link on an ancient launcher', async () => {
    const hx = makeHarness({ api: {} });
    assert.equal(hx.win.open('/api/files/download?path=a.txt', '_blank'), null);
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.deepEqual(hx.calls.open, [], 'the dead native open is not reached');
    assert.deepEqual(hx.calls.openFile, []);
    assert.deepEqual(hx.calls.copied, [`${BASE}api/files/download?path=a.txt`]);
    assert.deepEqual(hx.calls.toasts, [{ message: 'Link copied — open it in your browser.', tone: 'info' }]);
});

test('inside the framed wizard the file helpers reach the PARENT bridge (no window.open loop)', async () => {
    const prior = globalThis.window;
    try {
        const opened = [];
        const downloads = [];
        const win = {
            open: () => { throw new Error('window.open must never be reached with a live parent bridge'); },
            parent: {
                pywebview: { api: {
                    open_file_with_default_app: async (url, name) => { opened.push([url, name]); return { ok: true }; },
                    download_file_to_downloads: async (url, name, ext) => { downloads.push([url, name, ext]); return { ok: true }; },
                } },
            },
        };
        globalThis.window = win;
        const openResult = await openViaHostBridge('/api/tasks/t/artifacts/chat-media-aa.png', 'a.png');
        assert.deepEqual(openResult, { ok: true, native: true });
        assert.deepEqual(opened, [['/api/tasks/t/artifacts/chat-media-aa.png', 'a.png']]);
        const downloadResult = await downloadViaHostBridge('/api/files/download?path=x.txt', 'x.txt');
        assert.deepEqual(downloadResult, { ok: true, native: true });
        assert.deepEqual(downloads, [['/api/files/download?path=x.txt', 'x.txt', false]]);
    } finally {
        globalThis.window = prior;
    }
});

test('the parent pywebviewready listener is released when the framed document unloads', () => {
    const parentListeners = [];
    const parent = { addEventListener(type, fn, options) { parentListeners.push({ type, fn, options }); } };
    const hx = makeHarness({ pywebview: false, parent });
    const armed = parentListeners.find((entry) => entry.type === 'pywebviewready');
    assert.ok(armed, 'parent window armed for the bridge announcement');
    assert.ok(armed.options?.signal, 'parent listener carries an abort signal (disposer rule)');
    const pagehide = (hx.winListeners.pagehide || [])[0];
    assert.ok(pagehide, 'the framed document arms its own unload disposer');
    pagehide();
    assert.equal(armed.options.signal.aborted, true, 'unloading the frame releases the parent listener');
});

// --- bridge refusal degrades to copy-link ---------------------------------

// Minimal DOM the shared toast host needs; the copy path itself only needs
// window.navigator.clipboard here.
function toastDocument() {
    const toasts = [];
    const node = () => ({
        classList: { add() {} },
        setAttribute() {},
        appendChild() {},
        addEventListener() {},
        remove() {},
        set textContent(value) { toasts.push(value); },
        get textContent() { return toasts.at(-1) || ''; },
    });
    return { toasts, doc: { getElementById: () => null, createElement: () => node(), body: { appendChild() {} } } };
}

function bridgeRefusalWindow(api, copied) {
    return {
        location: { href: BASE },
        navigator: { clipboard: { writeText: async (text) => { copied.push(text); } } },
        pywebview: { api },
    };
}

test('a refused open degrades to the copy-link handoff instead of a dead control', async () => {
    const prior = { window: globalThis.window, document: globalThis.document };
    const copied = [];
    try {
        globalThis.window = bridgeRefusalWindow({
            open_file_with_default_app: async () => ({ ok: false, error: 'unsupported path' }),
        }, copied);
        const host = toastDocument();
        globalThis.document = host.doc;
        const result = await openViaHostBridge('/api/tasks/t/artifacts/chat-media-aa.png', 'a.png');
        assert.deepEqual(result, { ok: false, native: false, degraded: 'copy-link' });
        assert.deepEqual(copied, [`${BASE}api/tasks/t/artifacts/chat-media-aa.png`]);
        assert.deepEqual(host.toasts, ['Link copied — open it in your browser.']);
    } finally {
        globalThis.window = prior.window;
        globalThis.document = prior.document;
    }
});

test('a refused download degrades the same way', async () => {
    const prior = { window: globalThis.window, document: globalThis.document };
    const copied = [];
    try {
        globalThis.window = bridgeRefusalWindow({
            download_file_to_downloads: async () => ({ ok: false, error: 'path not allowed' }),
        }, copied);
        const host = toastDocument();
        globalThis.document = host.doc;
        const result = await downloadViaHostBridge('/api/files/download?path=x.txt', 'x.txt');
        assert.deepEqual(result, { ok: false, native: false, degraded: 'copy-link' });
        assert.deepEqual(copied, [`${BASE}api/files/download?path=x.txt`]);
        assert.deepEqual(host.toasts, ['Link copied — open it in your browser.']);
    } finally {
        globalThis.window = prior.window;
        globalThis.document = prior.document;
    }
});

test('a refusal the owner cannot even copy still surfaces as an error to the caller', async () => {
    const prior = { window: globalThis.window, document: globalThis.document };
    try {
        globalThis.window = {
            location: { href: BASE },
            navigator: {},
            pywebview: { api: { open_file_with_default_app: async () => ({ ok: false, error: 'gate refused' }) } },
        };
        globalThis.document = {
            createElement: () => ({ setAttribute() {}, select() {}, remove() {} }),
            body: { appendChild() {} },
        };
        await assert.rejects(
            () => openViaHostBridge('/api/tasks/t/artifacts/chat-media-aa.png', 'a.png'),
            /gate refused/,
        );
    } finally {
        globalThis.window = prior.window;
        globalThis.document = prior.document;
    }
});

// --- shared copy contract --------------------------------------------------

test('copyTextWithToast falls back to the textarea path and always reports', async () => {
    const toasts = [];
    const toast = (message, tone) => toasts.push({ message, tone });
    const selected = [];
    const doc = {
        createElement: () => ({
            value: '',
            setAttribute() {},
            select() { selected.push(this.value); },
            remove() {},
        }),
        body: { appendChild() {} },
        execCommand: () => true,
    };

    // No async clipboard at all (non-secure origin / desktop shell).
    const win = { navigator: {} };
    assert.equal(await copyTextWithToast('device-code-42', { win, doc, toast, okMessage: 'Code copied.' }), true);
    assert.deepEqual(selected, ['device-code-42']);
    assert.deepEqual(toasts, [{ message: 'Code copied.', tone: 'ok' }]);

    // Nothing copyable at all is reported, never a silent no-op.
    toasts.length = 0;
    assert.equal(await copyTextWithToast('', { win, doc, toast }), false);
    assert.deepEqual(toasts, [{ message: 'Nothing to copy.', tone: 'warn' }]);

    // A genuinely failing copy is an honest error, not a fake success.
    toasts.length = 0;
    assert.equal(
        await copyTextWithToast('x', { win, doc: { ...doc, execCommand: () => false }, toast }),
        false,
    );
    assert.equal(toasts[0].tone, 'error');
});

test('the agent login card never writes to the clipboard without a fallback', () => {
    const source = readFileSync(new URL('../modules/harness_login_cards.js', import.meta.url), 'utf8');
    // Copying the sign-in link, the device code, and the attach command are
    // STEPS of signing in; a bare optional-chained clipboard write is a control
    // that dies silently on every non-secure origin.
    assert.equal(source.includes('navigator.clipboard'), false);
    assert.equal((source.match(/copyTextWithToast\(/g) || []).length, 3);
});

test('without a bridge the download fetches the canonical URL, not the compat form', async () => {
    // downloadViaHostBridge reads the real module globals: give it a bridgeless
    // window, a stub DOM, and a recording fetch.
    const priorWindow = globalThis.window;
    const priorDocument = globalThis.document;
    const priorFetch = globalThis.fetch;
    const priorURL = globalThis.URL;
    const fetched = [];
    globalThis.window = { location: { href: BASE } };
    globalThis.document = {
        createElement: () => ({ click() {}, remove() {}, setAttribute() {} }),
        body: { appendChild() {} },
    };
    globalThis.fetch = async (url) => { fetched.push(String(url)); return { ok: true, blob: async () => ({}) }; };
    globalThis.URL = Object.assign(function URLStub(u) { return new priorURL(u, BASE); }, priorURL, {
        createObjectURL: () => 'blob:stub', revokeObjectURL: () => {},
    });
    try {
        await downloadViaHostBridge('/api/files/download?path=x.png', 'x.png', { browserUrl: '/api/tasks/t/artifacts/chat-media-aa.png' });
        assert.equal(fetched.length, 1);
        assert.ok(fetched[0].includes('/api/tasks/t/artifacts/chat-media-aa.png'), fetched[0]);
    } finally {
        globalThis.window = priorWindow;
        globalThis.document = priorDocument;
        globalThis.fetch = priorFetch;
        globalThis.URL = priorURL;
    }
});
