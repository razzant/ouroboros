// Behavioural characterization of the routing-annotation owner, exercised where
// the code now lives. The annotation is a sidecar on an existing user bubble,
// so a small element model is enough to pin every projection and every DOM
// decision: the wording per status, the single note node, removal on an empty
// projection, and the pending-delivery mark.

import assert from 'node:assert/strict';
import test from 'node:test';

import { createMessageAnnotations } from '../modules/chat_message_annotations.js';

function makeElement(tag = 'div') {
    const el = {
        tagName: tag.toUpperCase(),
        className: '',
        textContent: '',
        dataset: {},
        children: [],
        removed: false,
        classList: { removed: [], remove(name) { el.classList.removed.push(name); } },
        append(node) { el.children.push(node); return node; },
        before(node) { el.parentNode?.children.unshift(node); return node; },
        remove() { el.removed = true; },
        querySelector(selector) { return el.stubbedNodes?.[selector] ?? null; },
    };
    return el;
}

function annotations() {
    const messagesDiv = makeElement('div');
    const pendingUserBubbles = new Map();
    const priorDocument = globalThis.document;
    globalThis.document = { createElement: (tag) => makeElement(tag) };
    const api = createMessageAnnotations({ messagesDiv, pendingUserBubbles });
    return {
        ...api,
        messagesDiv,
        pendingUserBubbles,
        restore() { globalThis.document = priorDocument; },
    };
}

test('every routing status has its own honest wording', () => {
    const a = annotations();
    assert.equal(a.routingAnnotationText(null), '');
    assert.equal(a.routingAnnotationText({ status: 'pending' }), 'Choosing the right destination…');
    assert.equal(a.routingAnnotationText({ status: 'project_unavailable' }), 'Project is unavailable');
    assert.equal(
        a.routingAnnotationText({ action: 'mailbox_delivery', status: 'ok', target: 'Refactor' }),
        'Delivered to task · Refactor',
    );
    assert.equal(a.routingAnnotationText({ action: 'steer_task', status: 'ok' }), 'Steered task');
    // An unknown action still reads as words, never as a raw snake_case token.
    assert.equal(a.routingAnnotationText({ action: 'some_new_route', status: '' }), 'some new route');
    a.restore();
});

test('a manual-target ack lists the offered destinations', () => {
    const a = annotations();
    assert.equal(a.routingAnnotationText({
        status: 'needs_manual_target',
        options: [
            { label: 'Existing task' },
            { action: 'new_task_in_project', project_name: 'Refactor' },
            { title: 'Third' },
            null,
        ],
    }), 'Choose a target · Existing task / New task in Refactor / Third');
    // With no usable options it falls back to the target, then to a bare prompt.
    assert.equal(
        a.routingAnnotationText({ status: 'needs_manual_target', target: 'Refactor' }),
        'Choose a target · Refactor',
    );
    assert.equal(a.routingAnnotationText({ status: 'needs_manual_target' }), 'Choose a target');
    a.restore();
});

test('the note is created once, updated in place, and never becomes a second bubble', () => {
    const a = annotations();
    const bubble = makeElement('div');
    const time = makeElement('div');
    time.parentNode = bubble;
    bubble.stubbedNodes = { '.msg-routing-annotation': null, '.msg-time': time };
    assert.equal(a.renderRoutingAnnotation(bubble, { status: 'pending' }), true);
    const note = bubble.children[0];
    assert.equal(note.className, 'msg-routing-annotation');
    assert.equal(note.textContent, 'Choosing the right destination…');
    assert.equal(note.dataset.annotationStatus, 'pending');
    assert.equal(bubble.dataset.chatAnnotationStatus, 'pending');
    // A second ack reuses the same node.
    bubble.stubbedNodes['.msg-routing-annotation'] = note;
    a.renderRoutingAnnotation(bubble, { action: 'steer_task', status: 'ok' });
    assert.equal(bubble.children.length, 1);
    assert.equal(note.textContent, 'Steered task');
    a.restore();
});

test('an empty projection removes the note and the bubble marker', () => {
    const a = annotations();
    const bubble = makeElement('div');
    const note = makeElement('div');
    bubble.stubbedNodes = { '.msg-routing-annotation': note };
    bubble.dataset.chatAnnotationStatus = 'pending';
    assert.equal(a.renderRoutingAnnotation(bubble, null), false);
    assert.equal(note.removed, true);
    assert.equal(bubble.dataset.chatAnnotationStatus, undefined);
    assert.equal(a.renderRoutingAnnotation(null, { status: 'pending' }), false);
    a.restore();
});

test('an update addresses the owner message by client id, and only that one', () => {
    const a = annotations();
    const wanted = makeElement('div');
    wanted.dataset.clientMessageId = 'cid-2';
    wanted.stubbedNodes = { '.msg-routing-annotation': null, '.msg-time': null };
    const other = makeElement('div');
    other.dataset.clientMessageId = 'cid-1';
    a.messagesDiv.stubbedLists = { '.chat-bubble.user[data-client-message-id]': [other, wanted] };
    a.messagesDiv.querySelectorAll = (selector) => a.messagesDiv.stubbedLists[selector] ?? [];
    assert.equal(a.updateMessageAnnotation('cid-2', { status: 'pending' }), true);
    assert.equal(wanted.children.length, 1);
    assert.equal(other.children.length, 0);
    assert.equal(a.updateMessageAnnotation('', { status: 'pending' }), false);
    assert.equal(a.updateMessageAnnotation('cid-9', { status: 'pending' }), false);
    a.restore();
});

test('only the transient pending notes are swept', () => {
    const a = annotations();
    const bubble = makeElement('div');
    bubble.dataset.chatAnnotationStatus = 'pending';
    const note = makeElement('div');
    note.closest = () => bubble;
    a.messagesDiv.querySelectorAll = () => [note];
    a.clearTransientRoutingAnnotations();
    assert.equal(note.removed, true);
    assert.equal(bubble.dataset.chatAnnotationStatus, undefined);
    a.restore();
});

test('delivery drops the queued styling and forgets the bubble', () => {
    const a = annotations();
    const bubble = makeElement('div');
    const pendingNote = makeElement('div');
    bubble.stubbedNodes = { '.msg-pending': pendingNote };
    a.pendingUserBubbles.set('cid-1', bubble);
    a.markPendingDelivered('cid-1');
    assert.deepEqual(bubble.classList.removed, ['pending']);
    assert.equal(pendingNote.removed, true);
    assert.equal(a.pendingUserBubbles.size, 0);
    // An unknown id is a no-op, never a throw on a late outbound_sent frame.
    a.markPendingDelivered('cid-9');
    a.markPendingDelivered(undefined);
    a.restore();
});
