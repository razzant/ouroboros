// Behavioural characterization of the composer owner, exercised where the code
// now lives. Everything here reads element geometry and writes styles, classes
// or CSS custom properties, so a small element model is enough to pin the
// contracts that matter: the one-shot Swarm arm, the send-busy presentation,
// the bounded textarea growth, and the reserve/affordance pair that keeps the
// absolute composer off the messages.

import assert from 'node:assert/strict';
import test from 'node:test';

import { createComposer } from '../modules/chat_composer.js';

function makeElement(tag = 'div') {
    const el = {
        tagName: tag.toUpperCase(),
        value: '',
        title: '',
        textContent: '',
        disabled: false,
        selectionEnd: 0,
        scrollTop: 0,
        scrollHeight: 0,
        offsetHeight: 0,
        dataset: {},
        style: {
            properties: {},
            setProperty(name, value) { el.style.properties[name] = value; },
        },
        classList: {
            toggles: [],
            toggle(name, on) { el.classList.toggles.push([name, on]); },
        },
    };
    return el;
}

function composer({ visible = true, nearBottom = true } = {}) {
    const page = makeElement('div');
    const input = makeElement('textarea');
    const inputArea = makeElement('div');
    const pageHeader = makeElement('header');
    const messagesDiv = makeElement('div');
    const sendBtn = makeElement('button');
    const sendGroup = makeElement('div');
    const swarmBtn = makeElement('button');
    const scrollBottomBtn = makeElement('button');
    const tails = [];
    const api = createComposer({
        page,
        input,
        inputArea,
        pageHeader,
        messagesDiv,
        sendBtn,
        sendGroup,
        swarmBtn,
        scrollBottomBtn,
        isInstanceVisible: () => visible,
        isNearBottom: () => nearBottom,
        scrollToBottomAfterLayout: () => tails.push(true),
    });
    return { ...api, page, input, inputArea, pageHeader, messagesDiv, sendBtn, sendGroup, swarmBtn, scrollBottomBtn, tails };
}

test('Swarm is a one-shot arm read straight off the pill', () => {
    const c = composer();
    assert.equal(c.swarmArmed(), false);
    c.setSwarm(true);
    assert.equal(c.swarmBtn.dataset.armed, 'true');
    assert.equal(c.swarmArmed(), true);
    c.setSwarm(false);
    assert.equal(c.swarmBtn.dataset.armed, 'false');
    assert.equal(c.swarmArmed(), false);
});

test('send-busy shows the reason and restores the plain Send afterwards', () => {
    const c = composer();
    c.setSendBusy(true, 'Uploading files');
    assert.equal(c.sendGroup.dataset.busy, '1');
    assert.equal(c.sendBtn.disabled, true);
    assert.equal(c.sendBtn.textContent, 'Uploading files');
    assert.equal(c.sendBtn.title, 'Uploading files');
    c.setSendBusy(true);
    assert.equal(c.sendBtn.textContent, 'Sending');
    c.setSendBusy(false);
    assert.equal(c.sendGroup.dataset.busy, '0');
    assert.equal(c.sendBtn.disabled, false);
    assert.equal(c.sendBtn.textContent, 'Send');
    assert.equal(c.sendBtn.title, 'Send message');
});

test('the textarea grows to a bounded height and keeps the caret in view', () => {
    const c = composer({ nearBottom: false });
    c.input.value = 'hello';
    c.input.selectionEnd = 5;
    c.input.scrollHeight = 60;
    c.resizeChatInput();
    assert.equal(c.input.style.height, '60px');
    assert.equal(c.input.scrollTop, 60, 'a caret at the end follows the growth');
    c.input.scrollHeight = 400;
    c.input.selectionEnd = 0;
    c.input.scrollTop = 12;
    c.resizeChatInput();
    assert.equal(c.input.style.height, '120px', 'growth stops at the 120px cap');
    assert.equal(c.input.scrollTop, 12, 'a caret in the middle keeps the reader where it was');
    // Resizing always re-reserves the composer's space in the message column.
    assert.equal(c.page.style.properties['--chat-input-reserve'], '92px');
});

test('the reserve has floors so the first message is never hidden behind the chrome', () => {
    const c = composer({ nearBottom: false });
    c.updateMessagesPadding();
    assert.equal(c.page.style.properties['--chat-header-reserve'], '56px');
    assert.equal(c.page.style.properties['--chat-input-reserve'], '92px');
    c.pageHeader.offsetHeight = 104;
    c.inputArea.offsetHeight = 200;
    c.updateMessagesPadding();
    assert.equal(c.page.style.properties['--chat-header-reserve'], '104px');
    assert.equal(c.page.style.properties['--chat-input-reserve'], '216px');
});

test('a reserve change re-pins a sticky thread, and never an unsticky one', () => {
    const sticky = composer({ nearBottom: true });
    sticky.updateMessagesPadding();
    assert.deepEqual(sticky.tails, [true]);
    // preserveStickiness:false is the explicit "do not follow" caller contract.
    sticky.updateMessagesPadding({ preserveStickiness: false });
    assert.equal(sticky.tails.length, 1);
    const scrolledUp = composer({ nearBottom: false });
    scrolledUp.updateMessagesPadding();
    assert.deepEqual(scrolledUp.tails, []);
    // Either way the affordance is refreshed.
    assert.equal(scrolledUp.scrollBottomBtn.classList.toggles.length, 1);
});

test('jump-to-newest appears only for a visible column the reader scrolled up in', () => {
    const away = composer({ visible: true, nearBottom: false });
    away.updateScrollButton();
    assert.deepEqual(away.scrollBottomBtn.classList.toggles, [['visible', true]]);
    const atTail = composer({ visible: true, nearBottom: true });
    atTail.updateScrollButton();
    assert.deepEqual(atTail.scrollBottomBtn.classList.toggles, [['visible', false]]);
    const hidden = composer({ visible: false, nearBottom: false });
    hidden.updateScrollButton();
    assert.deepEqual(hidden.scrollBottomBtn.classList.toggles, [['visible', false]]);
});

test('the pin-to-tail write targets this instance own column', () => {
    const c = composer();
    c.messagesDiv.scrollHeight = 4321;
    c.scrollToBottom();
    assert.equal(c.messagesDiv.scrollTop, 4321);
});
