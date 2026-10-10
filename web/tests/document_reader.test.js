import assert from 'node:assert/strict';
import test from 'node:test';

import {
    DOCUMENT_READER_MAX_BYTES,
    DOCUMENT_READER_TIMEOUT_MS,
    DocumentReadError,
    createDocumentReader,
    decodeDocumentText,
    documentReaderKind,
    readDocumentBytes,
} from '../modules/document_reader.js';
import { renderChatMarkdown } from '../modules/chat_markdown.js';
import { createChatMedia } from '../modules/chat_media.js';
import { stampNodeTimestamp } from '../modules/chat_activity.js';
import { uploadView } from './helpers/attachment_views.js';

const encoder = new TextEncoder();
const URL_SOURCE = { url: '/api/tasks/task-1/artifacts/report.md' };

// A real fetch Response over a stream that records how much the reader pulled.
function streamed(bytes, { status = 200, headers = {}, chunk = 1024, stallAfter = Infinity, failAfter = Infinity } = {}) {
    const pulls = { chunks: 0, cancelled: false };
    let offset = 0;
    const body = new ReadableStream({
        pull(controller) {
            if (pulls.chunks >= failAfter) { controller.error(new TypeError('network connection lost')); return; }
            if (pulls.chunks >= stallAfter) return new Promise(() => {});
            if (offset >= bytes.length) { controller.close(); return; }
            controller.enqueue(bytes.subarray(offset, offset + chunk));
            offset += chunk;
            pulls.chunks += 1;
        },
        cancel() { pulls.cancelled = true; },
    }, { highWaterMark: 0 });
    return { response: new Response(body, { status, headers }), pulls };
}

function recordingFetch(response) {
    const calls = [];
    const fetchImpl = async (url, init) => { calls.push({ url, init }); return response; };
    return { calls, fetchImpl };
}

test('the reader opens Markdown and plain text only, by extension first', () => {
    assert.equal(documentReaderKind('report.md', 'application/octet-stream'), 'markdown');
    assert.equal(documentReaderKind('NOTES.MARKDOWN', ''), 'markdown');
    assert.equal(documentReaderKind('log.txt', 'text/plain'), 'text');
    // A named non-text type is never read, whatever MIME it was sent with.
    assert.equal(documentReaderKind('page.html', 'text/plain'), '');
    assert.equal(documentReaderKind('icon.svg', 'image/svg+xml'), '');
    assert.equal(documentReaderKind('report.pdf', 'application/pdf'), '');
    assert.equal(documentReaderKind('README', 'text/markdown; charset=utf-8'), 'markdown');
    assert.equal(documentReaderKind('LICENSE', 'text/plain'), 'text');
    assert.equal(documentReaderKind('blob', 'application/octet-stream'), '');
});

test('inline delivered bytes decode whole or as an explicit bounded prefix', async () => {
    const small = await readDocumentBytes({ base64: Buffer.from('# Title\nbody').toString('base64') });
    assert.equal(new TextDecoder().decode(small.bytes), '# Title\nbody');
    assert.deepEqual([small.total, small.partial], [12, false]);
    const large = Buffer.alloc(100, 'a');
    const prefix = await readDocumentBytes({ base64: large.toString('base64') }, { limit: 10 });
    assert.deepEqual([prefix.bytes.length, prefix.total, prefix.partial], [10, 100, true]);
});

test('a known small file is fetched whole; an unknown size asks only for the bounded prefix', async () => {
    const whole = recordingFetch(new Response('# Hi', { headers: { 'content-length': '4' } }));
    const result = await readDocumentBytes(URL_SOURCE, { knownSize: 4, fetchImpl: whole.fetchImpl });
    assert.deepEqual(whole.calls[0].init.headers, {}, 'no Range for a file known to fit');
    assert.deepEqual([new TextDecoder().decode(result.bytes), result.total, result.partial], ['# Hi', 4, false]);

    const bytes = encoder.encode('x'.repeat(16));
    const ranged = recordingFetch(new Response(bytes, {
        status: 206, headers: { 'content-range': 'bytes 0-15/5000', 'content-length': '16' },
    }));
    const prefix = await readDocumentBytes(URL_SOURCE, { limit: 16, fetchImpl: ranged.fetchImpl });
    assert.equal(ranged.calls[0].url, URL_SOURCE.url);
    assert.deepEqual(ranged.calls[0].init.headers, { Range: 'bytes=0-15' });
    assert.deepEqual([prefix.bytes.length, prefix.total, prefix.partial], [16, 5000, true]);
    assert.equal(DOCUMENT_READER_MAX_BYTES, 1024 * 1024);
});

test('a server that ignores Range is read only to the cap and then cancelled', async () => {
    const { response, pulls } = streamed(new Uint8Array(64 * 1024).fill(97), { chunk: 1024 });
    const result = await readDocumentBytes(URL_SOURCE, { limit: 4096, fetchImpl: async () => response });
    assert.deepEqual([result.bytes.length, result.total, result.partial], [4096, null, true]);
    assert.ok(pulls.chunks <= 6, `pulled ${pulls.chunks} chunks for a 4 KiB cap`);
    assert.equal(pulls.cancelled, true, 'the rest of the body is cancelled, not drained');
});

test('an empty delivered file is read as empty; other 416 answers stay errors', async () => {
    const empty = await readDocumentBytes(URL_SOURCE, {
        fetchImpl: async () => new Response(null, { status: 416, headers: { 'content-range': 'bytes */0' } }),
    });
    assert.deepEqual([empty.bytes.length, empty.total, empty.partial], [0, 0, false]);
    // Only the exact empty-file answer: another size, or any other form ending in /0, is a refusal.
    for (const contentRange of ['bytes */10', 'bytes 0-9/0', 'bytes */00', 'items */0', '']) {
        await assert.rejects(readDocumentBytes(URL_SOURCE, {
            fetchImpl: async () => new Response(null, { status: 416, headers: contentRange ? { 'content-range': contentRange } : {} }),
        }), (error) => error instanceof DocumentReadError && error.status === 416, contentRange);
    }
});

test('a body that cannot be streamed is refused, never read whole', async () => {
    let wholeReads = 0;
    const unstreamed = {
        status: 200, headers: new Headers({ 'content-length': '5' }), body: null,
        arrayBuffer: async () => { wholeReads += 1; return new ArrayBuffer(5); },
        text: async () => { wholeReads += 1; return 'hello'; },
    };
    await assert.rejects(readDocumentBytes(URL_SOURCE, { fetchImpl: async () => unstreamed }),
        (error) => error instanceof DocumentReadError && error.code === 'unsupported');
    // A refusal without a stream keeps its status; its reason is simply unknown.
    await assert.rejects(readDocumentBytes(URL_SOURCE, { fetchImpl: async () => ({ ...unstreamed, status: 503 }) }),
        (error) => error.code === 'http' && error.status === 503 && error.reason === '');
    assert.equal(wholeReads, 0, 'no unbounded arrayBuffer() or text()');
});

test('an overflowing chunk is cut to the bound and the rest of the body cancelled', async () => {
    const { response, pulls } = streamed(new Uint8Array(64 * 1024).fill(98), { chunk: 64 * 1024 });
    const result = await readDocumentBytes(URL_SOURCE, { limit: 4096, fetchImpl: async () => response });
    assert.deepEqual([result.bytes.length, result.partial], [4096, true]);
    assert.equal(result.bytes.buffer.byteLength, 4096, 'the result holds the bound, not a view of the 64 KiB chunk');
    assert.equal(pulls.cancelled, true);
});

test('a refusal is read only as far as its bounded reason prefix', async () => {
    const huge = encoder.encode(JSON.stringify({ reason_code: 'artifact_unavailable', padding: 'p'.repeat(256 * 1024) }));
    const { response, pulls } = streamed(huge, { status: 503, chunk: 1024 });
    await assert.rejects(readDocumentBytes(URL_SOURCE, { fetchImpl: async () => response }),
        (error) => error.code === 'http' && error.status === 503 && error.reason === '');
    assert.ok(pulls.chunks <= 6, `pulled ${pulls.chunks} chunks of a refusal`);
    assert.equal(pulls.cancelled, true, 'the rest of the refusal is cancelled');
});

test('a body that fails part-way is a broken transfer and its stream is released', async () => {
    const { response } = streamed(new Uint8Array(8192).fill(97), { chunk: 1024, failAfter: 2 });
    await assert.rejects(readDocumentBytes(URL_SOURCE, { fetchImpl: async () => response }),
        (error) => error.code === 'incomplete');
    assert.equal(response.bodyUsed, true);
});

// Timers of the reader's own bound, recorded; the bound itself shortened to a few ms.
function boundTimers() {
    const real = { set: globalThis.setTimeout, clear: globalThis.clearTimeout };
    const live = new Set();
    globalThis.setTimeout = (callback, ms, ...args) => {
        if (ms !== DOCUMENT_READER_TIMEOUT_MS) return real.set(callback, ms, ...args);
        const id = real.set(() => { live.delete(id); callback(...args); }, 5);
        live.add(id);
        return id;
    };
    globalThis.clearTimeout = (id) => { live.delete(id); real.clear(id); };
    return { live, restore: () => Object.assign(globalThis, { setTimeout: real.set, clearTimeout: real.clear }) };
}

test('one bound covers the answer and the body; an owner abort stays an abort', async () => {
    const timers = boundTimers();
    try {
        const silent = (_url, init) => new Promise((_resolve, reject) => {
            init.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
        });
        await assert.rejects(readDocumentBytes(URL_SOURCE, { fetchImpl: silent }),
            (error) => error instanceof DocumentReadError && error.code === 'timeout', 'no answer');
        const stalled = streamed(new Uint8Array(8192).fill(97), { chunk: 1024, stallAfter: 2 });
        await assert.rejects(readDocumentBytes(URL_SOURCE, { fetchImpl: async () => stalled.response }),
            (error) => error.code === 'timeout', 'a body that stops arriving');
        assert.equal(stalled.pulls.cancelled, true, 'the stalled body is cancelled');
        const owner = new AbortController();
        const pending = readDocumentBytes(URL_SOURCE, { signal: owner.signal, fetchImpl: silent });
        owner.abort();
        await assert.rejects(pending, (error) => error.name === 'AbortError');
        await readDocumentBytes(URL_SOURCE, { fetchImpl: async () => new Response('ok', { headers: { 'content-length': '2' } }) });
        assert.equal(timers.live.size, 0, 'every read clears its bound, however it ended');
    } finally {
        timers.restore();
    }
});

test('refusals keep their status and reason code and never try another address', async () => {
    for (const [status, reason] of [[404, 'artifact_unverified'], [409, 'artifact_identity_changed'], [503, 'artifact_unavailable']]) {
        const { calls, fetchImpl } = recordingFetch(new Response(JSON.stringify({ error: 'x', reason_code: reason }), { status }));
        await assert.rejects(readDocumentBytes(URL_SOURCE, { fetchImpl }), (error) => (
            error instanceof DocumentReadError && error.code === 'http' && error.status === status && error.reason === reason));
        assert.equal(calls.length, 1, `${status}: exactly one request, to the delivered copy`);
    }
    await assert.rejects(readDocumentBytes({}, {}), (error) => error.code === 'unavailable');
});

test('a body shorter than its declared length is a broken transfer, not a partial document', async () => {
    const { response } = streamed(encoder.encode('hello'), { headers: { 'content-length': '10' } });
    await assert.rejects(readDocumentBytes(URL_SOURCE, { fetchImpl: async () => response }),
        (error) => error.code === 'incomplete');
});

test('network failures are typed while an abort stays an abort', async () => {
    await assert.rejects(readDocumentBytes(URL_SOURCE, { fetchImpl: async () => { throw new TypeError('Failed to fetch'); } }),
        (error) => error.code === 'network');
    const controller = new AbortController();
    const pending = readDocumentBytes(URL_SOURCE, {
        signal: controller.signal,
        fetchImpl: (_url, init) => new Promise((_resolve, reject) => {
            init.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
        }),
    });
    controller.abort();
    await assert.rejects(pending, (error) => error.name === 'AbortError');
});

test('UTF-8 decoding separates a cut trailing character from malformed bytes and binary', () => {
    const text = encoder.encode('Привет, мир');
    const cut = text.subarray(0, 3); // 'П' and the first byte of 'р'
    assert.deepEqual(decodeDocumentText(cut, { partial: true }), { text: 'П', binary: false, malformed: false });
    // The same cut in a complete file is a malformed ending.
    assert.equal(decodeDocumentText(cut, { partial: false }).malformed, true);
    const interior = new Uint8Array([0x61, 0xff, 0x62]);
    assert.deepEqual(decodeDocumentText(interior, { partial: true }), { text: 'a�b', binary: false, malformed: true });
    assert.equal(decodeDocumentText(new Uint8Array([0x61, 0x00, 0x62])).binary, true);
    assert.equal(decodeDocumentText(new Uint8Array([0xef, 0xbb, 0xbf, 0x68, 0x69])).text, 'hi', 'a BOM is not content');
});

test('document Markdown reads a single newline as a soft break while chat keeps it', () => {
    const prior = { document: globalThis.document, marked: globalThis.marked, purify: globalThis.DOMPurify };
    const made = [];
    globalThis.marked = {
        Marked: class {
            constructor(options) { this.options = options; made.push(options); }
            lexer() { return []; }
            parse(source) { return `<p data-breaks="${this.options.breaks}">${source}</p>`; }
        },
    };
    globalThis.DOMPurify = { sanitize: (html) => html };
    globalThis.document = {
        createElement: () => ({
            content: { querySelectorAll: () => [] },
            get innerHTML() { return this.value; },
            set innerHTML(value) { this.value = String(value); },
        }),
    };
    try {
        assert.match(renderChatMarkdown('one\ntwo'), /data-breaks="true"/);
        assert.match(renderChatMarkdown('one\ntwo', { softBreaks: true }), /data-breaks="false"/);
        assert.match(renderChatMarkdown('again'), /data-breaks="true"/, 'chat keeps its own parser');
        assert.deepEqual(made.map((options) => options.breaks), [true, false], 'one parser per reading');
    } finally {
        globalThis.document = prior.document;
        globalThis.marked = prior.marked;
        globalThis.DOMPurify = prior.purify;
    }
});

// A small DOM, enough for the reader's own markup and the shared dialog focus
// helper, so the controller's lifecycle runs for real without a browser.
class FakeElement {
    constructor(doc, tag) {
        Object.assign(this, { ownerDocument: doc, tagName: tag.toUpperCase(), childNodes: [], parentNode: null,
            attrs: new Map(), listeners: new Map(), dataset: {}, hidden: false, open: false, scrollTop: 0, id: '' });
        const names = new Set();
        this.classList = {
            add: (...values) => values.forEach((value) => names.add(value)),
            remove: (...values) => values.forEach((value) => names.delete(value)),
            contains: (value) => names.has(value),
            toggle: (value, force = !names.has(value)) => { if (force) names.add(value); else names.delete(value); return force; },
        };
    }
    set className(value) { String(value).split(/\s+/).filter(Boolean).forEach((name) => this.classList.add(name)); }
    get children() { return this.childNodes.filter((node) => node instanceof FakeElement); }
    get isConnected() { let node = this; while (node.parentNode) node = node.parentNode; return node === this.ownerDocument.root; }
    get textContent() { return this.childNodes.map((node) => (node instanceof FakeElement ? node.textContent : node.text)).join(''); }
    set textContent(value) { this.replaceChildren({ text: String(value) }); }
    // Read back only from a text-only element, as `escapeHtmlText` does.
    get innerHTML() { return this.textContent.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
    set innerHTML(html) {
        this.replaceChildren();
        const stack = [this];
        for (const [, close, tag, attrs, text] of String(html).matchAll(/<(\/?)([a-z0-9-]+)([^>]*)>|([^<]+)/gi)) {
            if (text !== undefined) { if (text.trim()) stack.at(-1).append({ text }); continue; }
            if (close) { stack.pop(); continue; }
            const node = this.ownerDocument.createElement(tag);
            for (const [, name, value = ''] of attrs.matchAll(/([\w-]+)(?:="([^"]*)")?/g)) node.setAttribute(name, value);
            stack.at(-1).append(node);
            if (!['br', 'input', 'img'].includes(tag.toLowerCase())) stack.push(node);
        }
    }
    get tabIndex() { return this.attrs.has('tabindex') ? Number(this.attrs.get('tabindex')) : this.tagName === 'BUTTON' ? 0 : -1; }
    set tabIndex(value) { this.attrs.set('tabindex', String(value)); }
    setAttribute(name, value) {
        if (name === 'class') this.className = value;
        else if (name === 'hidden') this.hidden = true;
        else if (name.startsWith('data-')) this.dataset[name.slice(5).replace(/-([a-z])/g, (_all, char) => char.toUpperCase())] = String(value);
        else this.attrs.set(name, String(value));
    }
    getAttribute(name) { return this.attrs.has(name) ? this.attrs.get(name) : null; }
    removeAttribute(name) { this.attrs.delete(name); if (name === 'open') this.open = false; }
    append(...nodes) { for (const node of nodes) { node.parentNode?.childNodes?.splice(node.parentNode.childNodes.indexOf(node), 1); node.parentNode = this; this.childNodes.push(node); } }
    appendChild(node) { this.append(node); return node; }
    before(node) { node.remove?.(); node.parentNode = this.parentNode; this.parentNode.childNodes.splice(this.parentNode.childNodes.indexOf(this), 0, node); }
    replaceChildren(...nodes) { this.childNodes.forEach((node) => { node.parentNode = null; }); this.childNodes = []; this.append(...nodes); }
    remove() { if (this.parentNode) { this.parentNode.childNodes.splice(this.parentNode.childNodes.indexOf(this), 1); this.parentNode = null; } }
    contains(node) { for (let current = node; current; current = current.parentNode) if (current === this) return true; return false; }
    matches(selector) {
        return selector.split(',').some((part) => {
            const match = part.trim().match(/^([a-z]*)((?:\.[\w-]+|\[[^\]]+\])*)$/i);
            if (!match || (match[1] && match[1].toUpperCase() !== this.tagName)) return false;
            return [...match[2].matchAll(/\.([\w-]+)|\[([\w-]+)(?:="([^"]*)")?\]/g)].every(([, name, attr, value]) => {
                if (name) return this.classList.contains(name);
                const key = attr.startsWith('data-') ? attr.slice(5).replace(/-([a-z])/g, (_all, char) => char.toUpperCase()) : null;
                const actual = attr === 'hidden' ? (this.hidden ? '' : null) : key ? this.dataset[key] ?? null : this.getAttribute(attr);
                return actual !== null && (value === undefined || actual === value);
            });
        });
    }
    closest(selector) { for (let node = this; node instanceof FakeElement; node = node.parentNode) if (node.matches(selector)) return node; return null; }
    querySelectorAll(selector) {
        const found = [];
        const visit = (node) => node.children.forEach((child) => { if (child.matches(selector)) found.push(child); visit(child); });
        visit(this);
        return found;
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    getClientRects() { return [{}]; }
    focus() { this.ownerDocument.activeElement = this; }
    addEventListener(type, handler) { if (!this.listeners.has(type)) this.listeners.set(type, new Set()); this.listeners.get(type).add(handler); }
    removeEventListener(type, handler) { this.listeners.get(type)?.delete(handler); }
    dispatch(type, init = {}) {
        const event = { type, target: this, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, stopPropagation() {}, ...init };
        for (let node = this; node instanceof FakeElement; node = node.parentNode) for (const handler of [...(node.listeners.get(type) || [])]) handler(event);
        return event;
    }
    click() { return this.dispatch('click'); }
    showModal() { this.open = true; }
    close() { this.open = false; this.dispatch('close'); }
}

function fakeDocument() {
    const doc = { activeElement: null, defaultView: { getComputedStyle: () => ({ visibility: 'visible' }) } };
    doc.createElement = (tag) => new FakeElement(doc, tag);
    doc.root = doc.createElement('html');
    doc.body = doc.createElement('body');
    doc.root.append(doc.body);
    doc.activeElement = doc.body;
    return doc;
}

// fetch answers the test releases by hand, recording each request's signal.
function heldFetch() {
    const pending = [];
    const fetchImpl = (url, init) => new Promise((resolve, reject) => {
        init.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
        pending.push({ url, signal: init.signal, answer: (body, options) => resolve(new Response(body, options)) });
    });
    return { pending, fetchImpl };
}

async function settle() { for (let i = 0; i < 5; i += 1) await new Promise((resolve) => setTimeout(resolve, 0)); }

function readerFixture() {
    const doc = fakeDocument();
    const held = heldFetch();
    const priorFetch = globalThis.fetch;
    globalThis.fetch = held.fetchImpl;
    const chat = doc.createElement('div');
    doc.body.append(chat);
    const item = (name) => {
        const node = doc.createElement('div');
        const card = doc.createElement('button');
        node.append(card);
        chat.append(node);
        return { node, card, file: { filename: name, meta: 'TXT · 5 B', kind: 'text', size: 5, canOpen: true,
            reader: { url: `/api/tasks/task-1/artifacts/${name}` } } };
    };
    const reader = createDocumentReader({ doc, actions: { open() {}, download() {} } });
    const dialog = () => doc.body.querySelector('.document-reader');
    const view = () => ({
        title: dialog()?.querySelector('.document-reader-title').textContent ?? null,
        status: dialog()?.querySelector('.document-reader-status')?.textContent ?? null,
        source: dialog()?.querySelector('.document-reader-source')?.textContent ?? null,
    });
    return { doc, held, chat, item, reader, dialog, view, restore: () => { reader.destroy(); globalThis.fetch = priorFetch; } };
}

test('the reader paints only its current file and a closed or superseded read paints nothing', async () => {
    const fx = readerFixture();
    try {
        const a = fx.item('a.txt');
        const b = fx.item('b.txt');
        assert.equal(fx.reader.open(a.file, { owner: a.node, returnFocus: a.card }), true);
        assert.equal(fx.dialog().getAttribute('aria-modal'), 'true', 'the shared focus helper handles keys inside it');
        assert.equal(fx.doc.activeElement, fx.dialog().querySelector('.document-reader-body'));
        assert.equal(fx.view().status, 'Loading the document…');
        // Opening another file replaces the dialog and aborts the first read.
        fx.reader.open(b.file, { owner: b.node, returnFocus: b.card });
        assert.equal(fx.held.pending[0].signal.aborted, true);
        assert.equal(fx.doc.body.querySelectorAll('.document-reader').length, 1);
        fx.held.pending[0].answer('first file', { headers: { 'content-length': '10' } });
        await settle();
        assert.deepEqual(fx.view(), { title: 'b.txt', status: 'Loading the document…', source: null });
        fx.held.pending[1].answer('second', { headers: { 'content-length': '6' } });
        await settle();
        assert.deepEqual(fx.view(), { title: 'b.txt', status: null, source: 'second' });

        // Closing returns focus to the card and leaves nothing behind.
        fx.dialog().querySelector('[data-reader-action="close"]').click();
        assert.equal(fx.dialog(), null);
        assert.equal(fx.doc.activeElement, b.card);
        fx.reader.open(a.file, { owner: a.node, returnFocus: a.card });
        fx.dialog().dispatch('keydown', { key: 'Escape' });
        assert.equal(fx.dialog(), null, 'Escape closes');
        assert.equal(fx.held.pending[2].signal.aborted, true, 'closing aborts the pending read');
        fx.held.pending[2].answer('late', { headers: { 'content-length': '4' } });
        await settle();
        assert.equal(fx.dialog(), null, 'a late answer reopens nothing');
    } finally {
        fx.restore();
    }
});

test('the reader closes with the message that opened it and with its chat', async () => {
    const fx = readerFixture();
    try {
        const a = fx.item('a.txt');
        const other = fx.item('other.txt');
        fx.reader.open(a.file, { owner: a.node, returnFocus: a.card });
        fx.reader.release(other.node);
        assert.ok(fx.dialog(), 'releasing another message keeps the reader');
        fx.reader.release(fx.chat);
        assert.equal(fx.dialog(), null, 'releasing an ancestor of the opening message closes it');
        assert.equal(fx.held.pending[0].signal.aborted, true);
        assert.notEqual(fx.doc.activeElement, a.card, 'a released message is not refocused');

        fx.reader.open(a.file, { owner: a.node, returnFocus: a.card });
        fx.dialog().click(); // the backdrop is the dialog box itself
        assert.equal(fx.dialog(), null, 'a backdrop click closes');
        fx.reader.open(a.file, { owner: a.node, returnFocus: a.card });
        fx.reader.destroy();
        assert.equal(fx.dialog(), null);
        assert.equal(fx.reader.open(a.file, { owner: a.node }), false, 'a destroyed reader opens nothing');
    } finally {
        fx.restore();
    }
});

test('a read that outlasts its bound says so, waits for Retry, and close clears its timer', async () => {
    const timers = boundTimers();
    const fx = readerFixture();
    try {
        const a = fx.item('a.txt');
        fx.reader.open(a.file, { owner: a.node, returnFocus: a.card });
        assert.equal(timers.live.size, 1);
        await new Promise((resolve) => setTimeout(resolve, 30));
        await settle();
        assert.equal(fx.view().status, `The document did not arrive within ${DOCUMENT_READER_TIMEOUT_MS / 1000} seconds, so loading stopped.Retry`);
        assert.equal(fx.held.pending[0].signal.aborted, true, 'the timed-out request is stopped');
        await new Promise((resolve) => setTimeout(resolve, 30));
        assert.equal(fx.held.pending.length, 1, 'nothing retries by itself');
        fx.dialog().querySelector('[data-reader-action="retry"]').click();
        assert.equal(fx.held.pending.length, 2);
        assert.equal(timers.live.size, 1);
        fx.reader.close();
        await settle();
        assert.equal(timers.live.size, 0, 'closing clears the bound of the pending read');
        assert.equal(fx.held.pending[1].signal.aborted, true);
    } finally {
        fx.restore();
        timers.restore();
    }
});

test('a refused copy says why, and Retry reads the same address again', async () => {
    const fx = readerFixture();
    try {
        const a = fx.item('a.txt');
        fx.reader.open(a.file, { owner: a.node, returnFocus: a.card });
        fx.held.pending[0].answer(JSON.stringify({ reason_code: 'artifact_unavailable' }), { status: 503 });
        await settle();
        // The status line, then its Retry button.
        assert.equal(fx.view().status, 'This device cannot read the delivered copy right now.Retry');
        const retry = fx.dialog().querySelector('[data-reader-action="retry"]');
        retry.focus();
        retry.click();
        assert.equal(fx.held.pending[1].url, a.file.reader.url, 'Retry asks the delivered copy, nothing else');
        assert.equal(fx.doc.activeElement, fx.dialog().querySelector('.document-reader-body'),
            'the replaced Retry hands focus to the reading region, not to a detached button');
        fx.held.pending[1].answer('hello', { headers: { 'content-length': '5' } });
        await settle();
        assert.equal(fx.view().source, 'hello');
        fx.reader.close();
        fx.reader.open(a.file, { owner: a.node, returnFocus: a.card });
        fx.held.pending[2].answer(JSON.stringify({ reason_code: 'artifact_unverified' }), { status: 404 });
        await settle();
        assert.match(fx.view().status, /did not pass its integrity check/);
        assert.equal(fx.dialog().querySelector('[data-reader-action="retry"]'), null, 'no Retry where it cannot help');
    } finally {
        fx.restore();
    }
});

// The real chat media controller over the small DOM, with fetch held by hand.
function mediaFixture() {
    const doc = fakeDocument();
    const held = heldFetch();
    const prior = { document: globalThis.document, window: globalThis.window, fetch: globalThis.fetch };
    Object.assign(globalThis, { document: doc, window: { open() {} }, fetch: held.fetchImpl });
    const feed = doc.createElement('div');
    doc.body.append(feed);
    const media = createChatMedia({
        chatSessionId: 'session', durableChatMediaUrl: (value) => String(value || ''), formatMsgTime: () => null,
        insertMessageNode: (node) => feed.append(node), senderLabel: () => 'Ouroboros', stampNodeTimestamp,
    });
    const delivered = (filename, second, taskId = 'task-g') => ({
        type: 'document', role: 'assistant', task_id: taskId, filename, mime: 'text/plain', size_bytes: 5,
        download_url: `/api/tasks/${taskId}/artifacts/${encodeURIComponent(filename)}`, ts: `2026-10-07T00:00:0${second}Z`,
    });
    const restore = () => { media.destroy(); Object.assign(globalThis, prior); };
    return { doc, held, feed, media, delivered, reader: () => doc.body.querySelector('.document-reader'), restore };
}

test('a grouped file evicted with the earlier bubble that holds it closes its reader', async () => {
    const { doc, held, feed, media, delivered, restore } = mediaFixture();
    try {
        const bubbles = [delivered('a.txt', 1), delivered('b.txt', 2)].map((msg) => {
            const bubble = media.buildDocumentBubble(msg);
            assert.equal(media.buildGallery('files', msg, bubble), true);
            return bubble;
        });
        // One task's files share the first bubble's grid; the second bubble is never inserted.
        assert.deepEqual([feed.children.length, bubbles[1].isConnected], [1, false]);
        const [wrapper] = feed.children;
        wrapper.querySelectorAll('.chat-file-card')[1].click();
        const reader = () => doc.body.querySelector('.document-reader');
        assert.equal(reader().querySelector('.document-reader-title').textContent, 'b.txt');
        media.release(bubbles[1]);
        assert.ok(reader(), 'the emptied bubble that never reached the feed owns nothing');
        media.release(wrapper);
        assert.equal(reader(), null, 'evicting the wrapper that holds the file closes its reader');
        assert.equal(held.pending[0].signal.aborted, true);
    } finally {
        restore();
    }
});

test('closeTransient closes the reader and the file dialog and keeps the chat as it was', async () => {
    const fx = mediaFixture();
    try {
        const notes = fx.media.buildDocumentBubble(fx.delivered('notes.txt', 1, 'task-n'));
        const archive = fx.media.buildDocumentBubble({ ...fx.delivered('data.bin', 2, 'task-b'), mime: 'application/octet-stream' });
        fx.feed.append(notes, archive);
        notes.querySelector('.chat-file-card').click();
        assert.ok(fx.reader(), 'the reader is open with its read pending');
        fx.media.closeTransient();
        assert.equal(fx.reader(), null, 'the reader is gone');
        assert.equal(fx.held.pending[0].signal.aborted, true, 'and its read stopped');
        archive.querySelector('.chat-file-card').click();
        const dialog = fx.doc.body.querySelector('.chat-file-dialog');
        assert.equal(dialog.open, true);
        fx.media.closeTransient();
        assert.equal(dialog.open, false, 'the file dialog closes too');
        // Nothing was released: the same cards open again.
        assert.deepEqual([notes.isConnected, archive.isConnected], [true, true]);
        notes.querySelector('.chat-file-card').click();
        assert.equal(fx.reader().querySelector('.document-reader-title').textContent, 'notes.txt');
        fx.held.pending[1].answer('hello', { headers: { 'content-length': '5' } });
        await settle();
        assert.equal(fx.reader().querySelector('.document-reader-source').textContent, 'hello');
    } finally {
        fx.restore();
    }
});

// The owner's attachments and Ouroboros's deliveries share one media controller: an uploaded
// Markdown file keeps the file dialog (Read takes only a delivered copy), a delivered one reads,
// and each modal ends with the message that holds its card, a grouped card's earlier bubble too.
test("owner uploads keep the file dialog beside the delivered reader, and each modal ends with its own card", () => {
    const fx = mediaFixture();
    try {
        const owner = (caption, views) => {
            const bubble = fx.doc.createElement('div');
            bubble.innerHTML = `<div class="sender">You</div><div class="message">${caption}</div>`;
            fx.feed.append(bubble);
            assert.equal(fx.media.mountAttachments(bubble, views, caption), true);
            return bubble;
        };
        const earlier = owner('Earlier', [uploadView('sheet.pdf', 'file')]);
        const own = owner('Files', [uploadView('notes.md', 'file', { mime: 'text/markdown' }), uploadView('plan.pdf', 'file')]);
        // Two tasks each deliver two files; the second card of each joins its first bubble's grid.
        const group = (taskId, messages) => {
            const bubbles = messages.map((msg) => {
                const bubble = fx.media.buildDocumentBubble({ ...fx.delivered(msg.filename, msg.second, taskId), ...msg });
                assert.equal(fx.media.buildGallery('files', { ...fx.delivered(msg.filename, msg.second, taskId), ...msg }, bubble), true);
                return bubble;
            });
            assert.equal(bubbles[1].isConnected, false, `${messages[1].filename} joined ${messages[0].filename}`);
            return { wrapper: fx.feed.children.at(-1), dropped: bubbles[1] };
        };
        const report = group('task-r', [{ filename: 'report.pdf', second: 1, mime: 'application/pdf' }, { filename: 'report.md', second: 2 }]);
        const brief = group('task-b', [{ filename: 'data.pdf', second: 3, mime: 'application/pdf' }, { filename: 'brief.md', second: 4 }]);
        assert.equal(fx.feed.children.length, 4);
        const card = (root, name) => root.querySelectorAll('.chat-file-card')
            .find((node) => node.querySelector('.chat-file-name').textContent === name);
        const dialog = () => fx.doc.body.querySelector('.chat-file-dialog');
        const title = () => (dialog()?.open ? dialog().querySelector('.chat-file-dialog-title').textContent : null);
        assert.deepEqual(['notes.md', 'plan.pdf'].map((name) => Boolean(card(own, name).querySelector('.is-read'))), [false, false]);
        assert.deepEqual(['report.md', 'report.pdf'].map((name) => Boolean(card(report.wrapper, name).querySelector('.is-read'))), [true, false]);

        card(own, 'notes.md').click();
        assert.deepEqual([title(), fx.reader()], ['notes.md', null], "the owner's Markdown upload opens the file dialog");
        fx.media.closeTransient();
        card(report.wrapper, 'report.pdf').click();
        assert.equal(title(), 'report.pdf');
        fx.media.release(report.dropped);
        assert.equal(title(), 'report.pdf', 'the emptied bubble that never reached the feed owns nothing');
        fx.media.release(report.wrapper);
        assert.equal(title(), null, 'releasing the group that holds report.pdf closes its file dialog');

        card(brief.wrapper, 'brief.md').click();
        assert.equal(fx.reader().querySelector('.document-reader-title').textContent, 'brief.md');
        fx.media.release(earlier);
        assert.ok(fx.reader(), "releasing an owner's message leaves the delivered reader open");
        fx.media.release(brief.wrapper);
        assert.equal(fx.reader(), null, 'releasing the group that holds brief.md closes its reader');
        assert.equal(fx.held.pending[0].signal.aborted, true, 'and stops its read');

        card(own, 'plan.pdf').click();
        assert.equal(title(), 'plan.pdf');
        fx.media.release(brief.wrapper);
        assert.equal(title(), 'plan.pdf', "another message's release leaves the owner's file dialog open");
        fx.media.release(own);
        assert.equal(title(), null, "releasing the owner's message closes the dialog its card opened");
        card(own, 'notes.md').click();
        assert.deepEqual([title(), fx.reader()], [null, null], 'a released card opens nothing');
    } finally {
        fx.restore();
    }
});

test('the whole delivered name decides readability; only its display is cut to 200 characters', () => {
    const fx = mediaFixture();
    try {
        const long = `${'a'.repeat(202)}.md`;
        const disguised = `${'b'.repeat(197)}.md.html`;
        assert.deepEqual([long.length, disguised.length], [205, 205]);
        const [markdown, html] = [[long, 1], [disguised, 2]].map(([name, second]) => {
            const bubble = fx.media.buildDocumentBubble(fx.delivered(name, second, `task-${second}`));
            fx.feed.append(bubble);
            return bubble;
        });
        const card = (bubble) => ({
            name: bubble.querySelector('.chat-file-name').textContent,
            meta: bubble.querySelector('.chat-file-meta').textContent,
            read: Boolean(bubble.querySelector('.chat-file-more.is-read')),
        });
        assert.deepEqual(card(markdown), { name: long.slice(0, 200), meta: 'MD · 5 B', read: true });
        // Cut to 200, this name would end in `.md`; it is HTML and is not read.
        assert.equal(disguised.slice(0, 200).endsWith('.md'), true);
        assert.deepEqual(card(html), { name: disguised.slice(0, 200), meta: 'HTML · 5 B', read: false });
        html.querySelector('.chat-file-card').click();
        assert.equal(fx.reader(), null, 'the disguised HTML keeps the file dialog');
        assert.equal(fx.doc.body.querySelector('.chat-file-dialog').open, true);
        fx.media.closeTransient();
        markdown.querySelector('.chat-file-card').click();
        assert.equal(fx.reader().querySelector('.document-reader-title').textContent, long.slice(0, 200));
    } finally {
        fx.restore();
    }
});

test('the document name, size line and text are marked authored; the reader chrome is not', async () => {
    const fx = readerFixture();
    try {
        const a = fx.item('a.txt');
        fx.reader.open(a.file, { owner: a.node, returnFocus: a.card });
        fx.held.pending[0].answer('hello', { headers: { 'content-length': '5' } });
        await settle();
        const authored = (selector) => fx.dialog().querySelector(selector).closest('[data-i18n-authored]') !== null;
        assert.deepEqual(['.document-reader-title', '.document-reader-meta', '.document-reader-source'].map(authored),
            [true, true, true]);
        assert.deepEqual(['.document-reader-body', '[data-reader-action="close"]', '[data-reader-view="source"]'].map(authored),
            [false, false, false]);
    } finally {
        fx.restore();
    }
});
