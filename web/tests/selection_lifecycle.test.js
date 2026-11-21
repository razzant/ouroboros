import assert from 'node:assert/strict';
import test from 'node:test';
import { bindDocumentSelectionDismissal, ordinaryPress } from '../modules/selection_lifecycle.js';

function fixture({ target = {}, activeElement = null, rects = [] } = {}) {
    const listeners = new Map();
    let cleared = 0;
    const selection = {
        isCollapsed: false,
        rangeCount: 1,
        getRangeAt: () => ({ getClientRects: () => rects }),
        removeAllRanges: () => { cleared += 1; },
    };
    const doc = {
        activeElement,
        getSelection: () => selection,
        addEventListener: (name, handler, capture) => listeners.set(`${name}:${capture}`, handler),
        removeEventListener: (name, handler, capture) => {
            if (listeners.get(`${name}:${capture}`) === handler) listeners.delete(`${name}:${capture}`);
        },
    };
    return { doc, selection, target, listeners, cleared: () => cleared,
        press: (event = {}) => listeners.get('pointerdown:true')?.({
            button: 0, isPrimary: true, clientX: 500, clientY: 500, target, ...event,
        }) };
}

test('a primary press outside the selection clears it', () => {
    const f = fixture();
    const dispose = bindDocumentSelectionDismissal(f.doc);
    f.press();
    assert.equal(f.cleared(), 1);
    dispose();
});

test('secondary, Shift and editable presses preserve the selection', () => {
    const editable = { closest: (selector) => selector.includes('textarea') ? editable : null };
    const f = fixture({ target: editable, activeElement: editable });
    f.selection.anchorNode = editable;
    f.selection.focusNode = editable;
    const dispose = bindDocumentSelectionDismissal(f.doc);
    f.press({ button: 2 });
    f.press({ shiftKey: true });
    f.press();
    assert.equal(f.cleared(), 0);
    dispose();
});

test('touch presses stay native so scrolling and selection handles are preserved', () => {
    const f = fixture({ rects: [{ left: 0, top: 0, right: 100, bottom: 100 }] });
    const dispose = bindDocumentSelectionDismissal(f.doc);
    f.press({ pointerType: 'touch', clientX: 150, clientY: 150 });
    assert.equal(f.cleared(), 0);
    dispose();
});

test('a control inside the selected range dismisses the selection', () => {
    const button = { nodeType: 1, closest: (selector) => selector.includes('button') ? button : null };
    const f = fixture({ target: button, rects: [{ left: 0, top: 0, right: 1000, bottom: 1000 }] });
    const dispose = bindDocumentSelectionDismissal(f.doc);
    f.press({ clientX: 50, clientY: 50 });
    assert.equal(f.cleared(), 1);
    dispose();
});

test('a press inside the current range preserves the selection', () => {
    const f = fixture({ rects: [{ left: 0, top: 0, right: 100, bottom: 100 }] });
    const dispose = bindDocumentSelectionDismissal(f.doc);
    f.press({ clientX: 50, clientY: 50 });
    assert.equal(f.cleared(), 0);
    dispose();
});

test('the disposer removes the document listener', () => {
    const f = fixture();
    const dispose = bindDocumentSelectionDismissal(f.doc);
    dispose();
    f.press();
    assert.equal(f.cleared(), 0);
});

test('every ordinary press is heard, also one the browser collapses itself', () => {
    const f = fixture({ rects: [{ left: 0, top: 0, right: 100, bottom: 100 }] });
    let heard = 0;
    bindDocumentSelectionDismissal(f.doc, { onPress: () => { heard += 1; } });
    f.press({ pointerType: 'mouse' });
    assert.deepEqual([heard, f.cleared()], [1, 1], 'outside the selection: heard and cleared');
    f.press({ pointerType: 'mouse', clientX: 50, clientY: 50 });
    assert.deepEqual([heard, f.cleared()], [2, 1], 'inside it: heard, the native collapse is left alone');
    f.selection.isCollapsed = true;
    f.press({ pointerType: 'mouse' });
    assert.deepEqual([heard, f.cleared()], [3, 1], 'no selection left: still heard');
    for (const event of [{ pointerType: 'touch' }, { shiftKey: true }, { button: 2 }]) f.press(event);
    assert.equal(heard, 3, 'a touch, a modified or another button press keeps what rooms hold');
});

test('an ordinary press is a plain main-button mouse or fine pen press', () => {
    assert.equal(ordinaryPress({ button: 0, isPrimary: true, pointerType: 'mouse' }, {}), true);
    assert.equal(ordinaryPress({ button: 0, pointerType: 'pen' }, { documentElement: { dataset: { pointer: 'fine' } } }), true);
    assert.equal(ordinaryPress({ button: 0, pointerType: 'pen' }, { documentElement: { dataset: { pointer: 'coarse' } } }), false);
    assert.equal(ordinaryPress({ button: 0, pointerType: 'touch' }, {}), false);
    assert.equal(ordinaryPress({ button: 0, altKey: true }, {}), false);
});
