// The word-cut fitter (text_fit.js): an ellipsis never follows the punctuation a cut leaves, and an element with
// no layout shows its whole text (it is fitted once it has a width). Layout itself is the stand's to judge.
import assert from 'node:assert/strict';
import test from 'node:test';
import { ellipsize, fitText, fullText, isCut } from '../modules/text_fit.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

test('ellipsize drops the punctuation and space a cut leaves before adding the ellipsis', () => {
    assert.equal(ellipsize('без моего компактного редизайна.'), 'без моего компактного редизайна…');
    assert.equal(ellipsize('или ещё что-то; '), 'или ещё что-то…');
    assert.equal(ellipsize('спрашивай.…'), 'спрашивай…');
    assert.equal(ellipsize('word'), 'word…');
});

test('without layout the whole text shows, is remembered as the full text, and is not reported as cut', () => {
    const env = installDom();
    try {
        const el = document.createElement('div');
        const seen = [];
        fitText(el, 'a long line of words', 2, (cut) => seen.push(cut));
        assert.equal(el.textContent, 'a long line of words');
        assert.equal(fullText(el), 'a long line of words');
        assert.equal(isCut(el), false);
        assert.deepEqual(seen, [false]);
        fitText(el, 'open', 0);
        assert.equal(el.textContent, 'open');
        assert.equal(fullText(el), 'open', 'an unfitted element stands for its own text');
    } finally { restoreDom(env.prior); }
});
