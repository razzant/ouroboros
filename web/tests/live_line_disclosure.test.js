import assert from 'node:assert/strict';
import test from 'node:test';

import { liveLineRowToggleKey } from '../modules/chat_card_state.js';

// Minimal element stubs: enough of the DOM contract (closest/dataset/contains)
// for the pure guard, without jsdom.
function makeLine(lineKey = 'line-1') {
    const line = {
        dataset: { liveLineKey: lineKey },
        matches: (sel) => sel === '.chat-live-line.expandable',
        contains: (node) => Boolean(node && node._line === line),
    };
    return line;
}

function makeTarget(line, { interactiveSelector = null } = {}) {
    return {
        _line: line,
        closest(selector) {
            if (selector === '.chat-live-line.expandable') return line;
            const tokens = selector.split(',').map((s) => s.trim());
            if (interactiveSelector && tokens.includes(interactiveSelector)) return {};
            return null;
        },
    };
}

test('plain click on the row surface toggles the line', () => {
    const line = makeLine('k1');
    assert.equal(liveLineRowToggleKey(makeTarget(line)), 'k1');
});

test('click outside any expandable line does not toggle', () => {
    const target = { closest: () => null };
    assert.equal(liveLineRowToggleKey(target), '');
});

for (const tag of ['button', 'a', 'input', 'label', 'summary']) {
    test(`click on a nested <${tag}> keeps its own behavior (no toggle)`, () => {
        const line = makeLine('k1');
        assert.equal(liveLineRowToggleKey(makeTarget(line, { interactiveSelector: tag })), '');
    });
}

test('an active text selection inside the line never toggles', () => {
    const line = makeLine('k1');
    const target = makeTarget(line);
    const selection = { isCollapsed: false, anchorNode: { _line: line } };
    assert.equal(liveLineRowToggleKey(target, selection), '');
});

test('a collapsed caret (no selection) still toggles', () => {
    const line = makeLine('k1');
    const selection = { isCollapsed: true, anchorNode: { _line: line } };
    assert.equal(liveLineRowToggleKey(makeTarget(line), selection), 'k1');
});

test('a selection anchored OUTSIDE the line does not block the toggle', () => {
    const line = makeLine('k1');
    const selection = { isCollapsed: false, anchorNode: { _line: null } };
    assert.equal(liveLineRowToggleKey(makeTarget(line), selection), 'k1');
});
