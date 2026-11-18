// Node renders a failed identity check by inspecting the actual stub. That diagnostic should
// describe the node, not traverse its owner document and unrelated nodes.
import assert from 'node:assert/strict';
import test from 'node:test';
import { installDom, restoreDom } from './chat_dom_fixture.js';
import { domDouble } from './dom_double.js';

test('row-level DOM double keeps a diagnostic within the failed node’s subtree', () => {
    const { doc, el } = domDouble();
    const leaf = el('span');
    doc.body.append(el('aside', { text: 'unrelated-sibling' }), leaf);
    assert.ok(leaf.ownerDocument === doc && leaf.parentElement === doc.body);
    assert.ok(!Object.keys(leaf).includes('ownerDocument') && !Object.keys(leaf).includes('parent'));
    assert.throws(() => assert.equal(leaf, null), error =>
        error.code === 'ERR_ASSERTION' && !error.message.includes('unrelated-sibling'));
    leaf.remove();
    assert.equal(leaf.parentElement, null);
});

test('stub back-references work, stay non-enumerable, and keep a failing comparison bounded', () => {
    const { prior, mount } = installDom();
    try {
        const doc = globalThis.document;
        const sibling = doc.createElement('span');
        sibling.id = 'unrelated-sibling';
        sibling.textContent = 'unrelated-sibling';
        mount.append(sibling);
        const [a, b] = [doc.createElement('div'), doc.createElement('div')];
        a.id = 'leaf-a';
        b.id = 'leaf-b';
        mount.append(a, b);
        const text = doc.createTextNode('words');
        mount.append(text);

        // The references read back exactly, on an element and on a text node alike.
        assert.equal(a.parentNode, mount);
        assert.equal(a.parentElement, mount);
        assert.equal(a.ownerDocument, doc);
        assert.equal(text.parentNode, mount);
        assert.equal(text.parentElement, mount);
        // A parent manipulation keeps them exact: a move re-points them, a removal clears them.
        const moved = doc.createElement('p');
        moved.append(a);
        assert.equal(a.parentNode, moved);
        assert.equal(a.parentElement, moved);
        moved.removeChild(a);
        assert.equal(a.parentNode, null);
        assert.equal(a.parentElement, null);
        mount.append(a);
        assert.equal(a.parentNode, mount);
        // A text node removes itself through the same reference (TextStub.remove).
        text.remove();
        assert.equal(text.parentNode, null);

        // None of them is an enumerable own property, on either stub kind.
        for (const node of [a, text]) {
            const names = node.nodeType === 3 ? ['parentNode', 'parentElement']
                : ['parentNode', 'parentElement', 'ownerDocument'];
            for (const name of names) {
                const descriptor = Object.getOwnPropertyDescriptor(node, name);
                assert.equal(descriptor.enumerable, false, `${name} enumerable`);
                assert.equal(Object.keys(node).includes(name), false, `${name} in Object.keys`);
            }
        }

        // No user-supplied message: this exercises the generated Node assertion diagnostic.
        assert.throws(() => assert.equal(a, null), (error) => {
            assert.equal(error.code, 'ERR_ASSERTION');
            assert.match(error.message, /ElementStub/);
            assert.doesNotMatch(error.message, /unrelated-sibling|byId/);
            return true;
        });
    } finally {
        restoreDom(prior);
    }
});
