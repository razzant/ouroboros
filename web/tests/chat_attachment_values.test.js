import assert from 'node:assert/strict';
import test from 'node:test';
import { attachmentCaption, attachmentTail, attachmentViews, buildAttachmentBlock, composerText } from '../modules/chat_attachments.js';
import { uploadView } from './helpers/attachment_views.js';
import { NodeStub, countPropertyWrites, fixture } from './chat_decision_fixture.js';

const composed = { composed: true };

test('the caption hides only the exact generated tail and text the row marks as the host\'s', () => {
    const views = [uploadView('one.png', 'image'), uploadView('two.pdf', 'file')];
    const tail = attachmentTail(views.map((view) => view.name));
    assert.equal(tail, '[Attached file: one.png]\n[Attached file: two.pdf]');
    assert.equal(attachmentCaption(`Look\n\n${tail}`, views, composed), 'Look');
    assert.equal(attachmentCaption(tail, views, composed), '');
    // Host text is hidden only when the row marks it as the host's (`text_placeholder`).
    assert.equal(attachmentCaption('(image attached)', views, { placeholder: true }), '');
    assert.equal(attachmentCaption('[user attachment: one.png]', views, { placeholder: true }), '',
        'the mark covers the whole text, whichever host label it is');
    assert.equal(attachmentCaption('(image attached)', views, composed), '(image attached)', 'typed by the owner, kept');
    // The owner's own words that merely look like a tail stay exactly as written.
    assert.equal(attachmentCaption('Look [Attached file: one.png]', views, composed), 'Look [Attached file: one.png]');
    assert.equal(attachmentCaption(`Look\n\n${tail}`, [views[1], views[0]], composed), `Look\n\n${tail}`);
    assert.equal(attachmentCaption(`Look\n\n${tail}`, [], composed), `Look\n\n${tail}`);
    const many = Array.from({ length: 27 }, (_, i) => `f${i}.txt`);
    assert.ok(attachmentTail(many).endsWith('[Attached file: f24.txt]\n[2 more attached files]'));
});

test('a caption no composer wrote is kept word for word, even when it ends like a tail', () => {
    // skills/telegram/plugin.py passes the owner's caption unchanged: its row is `skill:telegram`, not `web`.
    const views = [uploadView('plan.pdf', 'file')];
    const caption = 'Keep this note\n\n[Attached file: plan.pdf]';
    assert.equal(attachmentCaption(caption, views), caption, 'no provenance claimed: literal');
    assert.equal(attachmentCaption(caption, views, { composed: false }), caption);
    assert.equal(attachmentCaption('[Attached file: plan.pdf]', views), '[Attached file: plan.pdf]');
    assert.equal(attachmentCaption('(file attached)', views, { placeholder: true }), '', "the host's own placeholder still hides");
    assert.equal(attachmentCaption(caption, views, composed), 'Keep this note', 'the composer hides its own tail');
});

test('the composer adds the tail to every message but keeps the exact Restart command exact', () => {
    assert.equal(composerText('Look', ['a.png', 'b.pdf']), 'Look\n\n[Attached file: a.png]\n[Attached file: b.pdf]');
    assert.equal(composerText('', ['a.png']), '[Attached file: a.png]');
    // ws.py and server_control match only `/restart` (trimmed, any case): its files ride that one row.
    for (const command of ['/restart', '/RESTART', ' /Restart ']) assert.equal(composerText(command, ['a.png']), command);
    // Words around it are a message, never the command.
    for (const words of ['/restart please', '/restart\nnow', 'please /restart', '/restarting']) {
        assert.equal(composerText(words, ['a.png']), `${words}\n\n[Attached file: a.png]`);
    }
});

test('every attachment renders past the text tail, and its exact tail still hides', () => {
    const views = Array.from({ length: 30 }, (_, i) => uploadView(`shot-${i}.png`, 'image'));
    assert.equal(attachmentViews(views).length, 30, 'no silent cap: the tail bound is text-only');
    const tail = attachmentTail(views.map((view) => view.name));
    assert.ok(tail.endsWith('[5 more attached files]'));
    assert.equal(attachmentCaption(`Тридцать\n\n${tail}`, attachmentViews(views), composed), 'Тридцать');
});

test('a name is bounded per code point as the server labels it: a server name stays whole, no pair is split', () => {
    // chat_uploads.safe_upload_name('a' * 180 + '😀' * 10 + '.png'): 193 code points, 202 UTF-16 units.
    const stored = `${'a'.repeat(180)}${'😀'.repeat(9)}.png`;
    const [view] = attachmentViews([uploadView(stored, 'image')]);
    assert.equal(view.name, stored, 'the whole server name, extension included');
    assert.equal(attachmentCaption(`Look\n\n${attachmentTail([stored])}`, [view], composed), 'Look',
        "the composer's tail (the server name) still hides");
    const [cut] = attachmentViews([{ name: '😀'.repeat(201), kind: 'file', available: false }]);
    assert.equal(cut.name, '😀'.repeat(200), '200 code points, never half a surrogate pair');
});

test('attachment views keep their closed shape and refuse a foreign URL', () => {
    const [ok, foreign, weird] = attachmentViews([
        uploadView('a.png', 'image'),
        uploadView('b.png', 'image', { url: 'https://evil.example/a.png' }),
        { name: 'c\nd', kind: 'script', available: true, url: `/api/files/download?upload=${'b'.repeat(32)}_c` },
    ]);
    assert.equal(ok.available && ok.kind === 'image' && ok.url.startsWith('/api/files/download?upload='), true);
    assert.deepEqual([foreign.available, foreign.kind, foreign.url], [false, 'file', '']);
    assert.deepEqual([weird.name, weird.kind], ['c d', 'file']);
});

test('a Telegram file never received is one inert card with its known size and reason', () => {
    const fx = fixture();
    try {
        const raw = [
            { name: 'large.mp4', kind: 'file', available: false, size: 11534336, unavailable: 'too_large', limit_bytes: 10485760 },
            { name: 'unread.pdf', kind: 'file', available: false, unavailable: 'download_failed' },
        ];
        const block = buildAttachmentBlock({ fileExtension: () => 'FILE', humanSize: n => n ? `${n} B` : '' }, raw);
        assert.match(block.innerHTML, /disabled aria-disabled="true"/);
        assert.match(block.innerHTML, /11534336 B · File exceeds the download limit \(10485760 B\)/);
        assert.match(block.innerHTML, /File could not be downloaded/);
        assert.doesNotMatch(block.innerHTML, /•••|download\?upload|<video/);
        assert.equal(attachmentCaption('(attachment unavailable)', raw, { placeholder: true }), '');
    } finally { fx.restore(); }
});

test('a file the host could not attach is named right under the owner\'s words, and only while that is true', () => {
    const fx = fixture();
    try {
        const bubble = new NodeStub('div');
        bubble.dataset.clientMessageId = 'cm-files';
        bubble.append(...['message', 'msg-pending', 'msg-time'].map((name) => Object.assign(new NodeStub('div'), { className: name })));
        const row = (ordinal, label, status) => ({ ordinal, label, status, ...(status === 'rejected' ? { reason: 'source_missing' } : {}) });
        const steered = { action: 'steer_task', status: 'delivered', target: 't1', target_label: 'Fix CI',
            attachment_manifest: [row(0, 'a.png', 'staged'), row(1, 'b.pdf', 'rejected')] };
        assert.equal(fx.decision.renderRoutingDecision(bubble, steered), true);
        const note = bubble.querySelector('.msg-attachment-outcome');
        assert.equal(note.textContent, 'Not attached to the task: b.pdf');
        // One place whatever landed first; its own node, so the Main fold of the receipt line never hides it.
        assert.deepEqual(bubble.children.map((node) => node.className),
            ['message', 'msg-attachment-outcome', 'msg-pending', 'msg-routing-annotation', 'msg-time']);
        const writes = countPropertyWrites(note, 'textContent');
        assert.equal(fx.decision.renderRoutingDecision(bubble, steered), false);
        assert.equal(writes(), 0);
        // Only the manifest changed (the receipt line did not): still repainted.
        assert.equal(fx.decision.renderRoutingDecision(bubble, {
            ...steered, attachment_manifest: [row(0, 'a.png', 'rejected'), row(1, '', 'rejected')] }), true);
        assert.equal(note.textContent, 'Not attached to the task: a.png, #2');
        // Everything attached: nothing to say, as on a clean delivery.
        const clean = { ...steered, attachment_manifest: [row(0, 'a.png', 'staged')] };
        assert.equal(fx.decision.renderRoutingDecision(bubble, clean), true);
        assert.equal(bubble.querySelector('.msg-attachment-outcome'), null);
        assert.equal(fx.decision.renderRoutingDecision(bubble, clean), false);
    } finally { fx.restore(); }
});
