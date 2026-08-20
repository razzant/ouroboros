// Behavioural characterization of the attachment-staging owner, exercised where
// the code now lives. The factory wires the paperclip, paste and drag/drop
// listeners itself, so the tests drive it exactly like the browser does: fire
// the captured listeners and observe the staged list, the preview strip, the
// upload lock and the cleanup calls.

import assert from 'node:assert/strict';
import test from 'node:test';

import { createChatAttachments } from '../modules/chat_attachments.js';

function makeElement(tag = 'div') {
    const el = {
        tagName: tag.toUpperCase(),
        id: '',
        className: '',
        textContent: '',
        innerHTML: '',
        value: '',
        disabled: false,
        files: [],
        dataset: {},
        attributes: {},
        children: [],
        listeners: {},
        classNames: new Set(),
        stubButtons: [],
        appendChild(child) { el.children.push(child); return child; },
        remove() {},
        setAttribute(name, value) { el.attributes[name] = String(value); },
        addEventListener(type, fn) { (el.listeners[type] ||= []).push(fn); },
        fire(type, event = {}) { (el.listeners[type] || []).forEach((fn) => fn(event)); },
        click() { el.fire('click', {}); },
        querySelectorAll() { return el.stubButtons; },
        classList: {
            add(name) { el.classNames.add(name); },
            remove(name) { el.classNames.delete(name); },
            toggle(name, force) {
                const on = force === undefined ? !el.classNames.has(name) : Boolean(force);
                if (on) el.classNames.add(name); else el.classNames.delete(name);
                return on;
            },
        },
    };
    return el;
}

function attachments({ responses = [] } = {}) {
    const page = makeElement('div');
    const input = makeElement('textarea');
    const inputArea = makeElement('div');
    const attachBtn = makeElement('button');
    const fileInput = makeElement('input');
    const attachmentPreview = makeElement('div');
    const paddingCalls = [];
    const toasts = [];
    const fetchCalls = [];

    const priorDocument = globalThis.document;
    const priorFetch = globalThis.fetch;
    const priorRaf = globalThis.requestAnimationFrame;
    const priorSetTimeout = globalThis.setTimeout;
    globalThis.document = {
        getElementById: () => null,
        createElement: (tag) => {
            const el = makeElement(tag);
            // showToast renders through document.createElement; capture the text.
            Object.defineProperty(el, 'textContent', {
                get() { return el._text || ''; },
                set(v) { el._text = v; toasts.push(v); },
            });
            return el;
        },
        body: makeElement('body'),
    };
    globalThis.fetch = async (url, init = {}) => {
        fetchCalls.push({ url, method: init.method || 'GET', body: init.body || '' });
        const next = responses.shift();
        if (typeof next === 'function') return next();
        return next ?? { ok: true, status: 200, json: async () => ({}) };
    };
    globalThis.requestAnimationFrame = (fn) => { fn(); return 0; };
    globalThis.setTimeout = () => 0; // keep toast auto-dismiss timers off the loop

    const api = createChatAttachments({
        page,
        input,
        inputArea,
        attachBtn,
        fileInput,
        attachmentPreview,
        updateMessagesPadding: (options) => paddingCalls.push(options),
    });

    return {
        ...api,
        page,
        input,
        inputArea,
        attachBtn,
        fileInput,
        attachmentPreview,
        paddingCalls,
        toasts,
        fetchCalls,
        stageViaInput(files) {
            fileInput.files = files;
            fileInput.fire('change', {});
        },
        restore() {
            globalThis.document = priorDocument;
            globalThis.fetch = priorFetch;
            globalThis.requestAnimationFrame = priorRaf;
            globalThis.setTimeout = priorSetTimeout;
        },
    };
}

const fakeFile = (name, size = 10) => ({ name, size, type: 'text/plain' });

test('files staged through the hidden input land in the preview strip', () => {
    const a = attachments();
    a.stageViaInput([fakeFile('a.txt'), fakeFile('b.txt')]);
    assert.equal(a.hasPendingAttachments(), true);
    const items = a.stagedAttachmentItems();
    assert.deepEqual(items.map((item) => item.display_name), ['a.txt', 'b.txt']);
    assert.ok(items.every((item) => item.id), 'every staged item gets an id');
    assert.ok(a.attachmentPreview.classNames.has('visible'));
    assert.match(a.attachmentPreview.innerHTML, /a\.txt/);
    assert.equal(a.fileInput.value, '', 'the input clears so re-picking the same file re-fires change');
    a.restore();
});

test('the caps hold: per-message count, per-file bytes, total bytes', () => {
    const a = attachments();
    a.stageViaInput(Array.from({ length: 11 }, (_, i) => fakeFile(`f${i}.txt`)));
    assert.equal(a.hasPendingAttachments(), false);
    assert.match(a.toasts.at(-1), /Attach up to 10 files/);
    a.stageViaInput([fakeFile('big.bin', 51 * 1024 * 1024)]);
    assert.equal(a.hasPendingAttachments(), false);
    assert.match(a.toasts.at(-1), /50 MB or smaller/);
    a.stageViaInput([fakeFile('x.bin', 49 * 1024 * 1024), fakeFile('y.bin', 49 * 1024 * 1024)]);
    assert.equal(a.stagedAttachmentItems().length, 2);
    a.stageViaInput([fakeFile('z.bin', 10 * 1024 * 1024)]);
    assert.equal(a.stagedAttachmentItems().length, 2, 'the total cap rejects the third file');
    assert.match(a.toasts.at(-1), /100 MB total/);
    a.restore();
});

test('the preview remove button unstages exactly its item', () => {
    const a = attachments();
    a.stageViaInput([fakeFile('keep.txt'), fakeFile('drop.txt')]);
    const dropId = a.stagedAttachmentItems()[1].id;
    const button = makeElement('button');
    button.getAttribute = () => dropId;
    a.attachmentPreview.stubButtons = [button];
    a.updateAttachmentPreview();
    button.click();
    assert.deepEqual(a.stagedAttachmentItems().map((item) => item.display_name), ['keep.txt']);
    a.restore();
});

test('paste stages only image items, with timestamped clipboard names', () => {
    const a = attachments();
    let prevented = false;
    a.input.fire('paste', {
        preventDefault: () => { prevented = true; },
        clipboardData: {
            items: [
                { kind: 'string', type: 'text/plain' },
                { kind: 'file', type: 'image/png', getAsFile: () => ({ type: 'image/png' }) },
            ],
        },
    });
    assert.equal(prevented, true);
    const items = a.stagedAttachmentItems();
    assert.equal(items.length, 1);
    assert.match(items[0].display_name, /^clipboard-\d+\.png$/);
    // A text-only paste stays untouched: no staging, no preventDefault.
    prevented = false;
    a.input.fire('paste', {
        preventDefault: () => { prevented = true; },
        clipboardData: { items: [{ kind: 'string', type: 'text/plain' }] },
    });
    assert.equal(prevented, false);
    assert.equal(a.stagedAttachmentItems().length, 1);
    a.restore();
});

test('drag depth drives the drop-zone highlight and drop stages the files', () => {
    const a = attachments();
    const drag = (files = []) => ({
        preventDefault() {},
        dataTransfer: { types: ['Files'], files },
    });
    a.page.fire('dragenter', drag());
    a.page.fire('dragenter', drag());
    assert.ok(a.inputArea.classNames.has('drag-active'));
    a.page.fire('dragleave', drag());
    assert.ok(a.inputArea.classNames.has('drag-active'), 'one leave of two enters keeps the highlight');
    a.page.fire('dragleave', drag());
    assert.ok(!a.inputArea.classNames.has('drag-active'));
    a.page.fire('drop', drag([fakeFile('dropped.txt')]));
    assert.deepEqual(a.stagedAttachmentItems().map((item) => item.display_name), ['dropped.txt']);
    assert.ok(!a.inputArea.classNames.has('drag-active'));
    // A non-file drag is ignored entirely.
    a.page.fire('drop', { preventDefault() {}, dataTransfer: { types: ['text/plain'], files: [fakeFile('no.txt')] } });
    assert.equal(a.stagedAttachmentItems().length, 1);
    a.restore();
});

test('the upload lock disables the controls and refuses staging changes', () => {
    const a = attachments();
    a.stageViaInput([fakeFile('a.txt')]);
    a.setAttachmentUploadState(true);
    assert.equal(a.isAttachmentUploadBusy(), true);
    assert.equal(a.attachBtn.disabled, true);
    assert.equal(a.fileInput.disabled, true);
    assert.equal(a.input.disabled, true);
    a.stageViaInput([fakeFile('b.txt')]);
    assert.equal(a.stagedAttachmentItems().length, 1, 'staging is refused while uploading');
    assert.match(a.toasts.at(-1), /Wait for the current upload/);
    a.setAttachmentUploadState(false);
    assert.equal(a.isAttachmentUploadBusy(), false);
    assert.equal(a.attachBtn.disabled, false);
    a.restore();
});

test('cleanup deletes every uploaded filename and survives per-file failures', async () => {
    const a = attachments({
        responses: [
            { ok: true, status: 200, json: async () => ({}) },
            { ok: false, status: 500, json: async () => ({}) },
        ],
    });
    await a.cleanupUploadedAttachments([
        { filename: 'u1.txt' },
        { filename: 'u2.txt' },
        { filename: '' },
    ]);
    assert.deepEqual(a.fetchCalls.map((call) => call.method), ['DELETE', 'DELETE']);
    assert.ok(a.fetchCalls.every((call) => call.url === '/api/chat/upload'));
    assert.deepEqual(
        a.fetchCalls.map((call) => JSON.parse(call.body).filename),
        ['u1.txt', 'u2.txt'],
    );
    a.restore();
});

test('clearPendingAttachments empties the staging list for the send path', () => {
    const a = attachments();
    a.stageViaInput([fakeFile('a.txt')]);
    a.clearPendingAttachments();
    assert.equal(a.hasPendingAttachments(), false);
    a.updateAttachmentPreview();
    assert.ok(!a.attachmentPreview.classNames.has('visible'));
    assert.equal(a.attachmentPreview.innerHTML, '');
    a.restore();
});
