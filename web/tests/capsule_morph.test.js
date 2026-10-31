// A strip that opens alone grows from its capsule (capsule_morph.js, owner 19.10 2A): its glass is clipped from the
// closed capsule to the open one and back, its head rides with the glass's leading end, the controls that step aside
// fade where they stood and the ones that come back fade in; everything it added is gone when it ends. Nothing moves
// beside the others, under reduced motion, without layout or without Web Animations. Real frames are in the browser
// lane (tests/test_composer_capsules_browser.py).
import assert from 'node:assert/strict';
import test from 'node:test';
import { MORPH, beginMorph, stopMorph } from '../modules/capsule_morph.js';
import { domDouble } from './dom_double.js';

function animatable(node, log) {
    node.animate = (keyframes, options) => {
        let resolve;
        let reject;
        const finished = new Promise((res, rej) => { resolve = res; reject = rej; });
        finished.catch(() => {});
        const animation = { node, keyframes, options, playState: 'running', finished,
            cancel() { if (animation.playState !== 'running') return; animation.playState = 'idle'; reject(new Error('AbortError')); },
            finish() { animation.playState = 'finished'; resolve(animation); } };
        log.push(animation);
        return animation;
    };
    return node;
}

function fixture({ reduced = false } = {}) {
    const { doc, el } = domDouble();
    const log = [];
    const create = doc.createElement;
    doc.createElement = (tag) => animatable(create(tag), log);
    const win = { getComputedStyle: () => ({ display: 'flex', direction: 'ltr' }),
        matchMedia: (query) => ({ matches: reduced && query === '(prefers-reduced-motion: reduce)' }) };
    const box = (left, width) => ({ left, top: 700, width, height: 32, right: left + width, bottom: 732 });
    const segs = [0, 1, 2].map((i) => animatable(el('button', { className: 'chat-effort-seg', text: `L${i}` }), log));
    const head = animatable(el('button', { className: 'chat-capsule-head' }), log);
    const body = animatable(el('div', { className: 'chat-capsule-body' }, ...segs), log);
    const capsule = animatable(el('div', { className: 'chat-capsule chat-effort-range' }, head, body), log);
    const menu = animatable(el('button', { className: 'composer-menu' }), log);
    const subagents = animatable(el('div', { className: 'chat-capsule chat-subagents-mode' }), log);
    const context = animatable(el('div', { className: 'chat-capsule chat-context-mode' }), log);
    const row = el('div', { className: 'chat-toolbar-row' }, menu, subagents, context, capsule);
    doc.body.append(row);
    row.rect = box(6, 378);
    menu.rect = box(12, 32);
    subagents.rect = box(50, 70);
    context.rect = box(126, 80);
    const closed = () => {
        capsule.rect = box(212, 90);
        head.rect = box(213, 30);
        delete capsule.dataset.fit;
        capsule.dataset.open = 'false';
        for (const kid of [menu, subagents, context]) kid.hidden = false;
    };
    const openAlone = () => {
        capsule.rect = box(6, 378);
        head.rect = box(7, 30);
        capsule.dataset.fit = 'solo';
        capsule.dataset.open = 'true';
        for (const kid of [menu, subagents, context]) kid.hidden = true;
    };
    closed();
    const surface = () => capsule.children.find((node) => node.classList.contains('chat-capsule-surface')) || null;
    const on = (node) => log.filter((animation) => animation.node === node);
    return { doc, win, log, row, capsule, head, body, segs, menu, subagents, context, closed, openAlone, surface, on };
}

const settle = () => new Promise((resolve) => setImmediate(resolve));

test('opening alone: the glass grows from the closed capsule, the head rides, the others fade where they stood', async () => {
    const f = fixture();
    const morph = beginMorph({ el: f.capsule, row: f.row, win: f.win });
    assert.ok(morph);
    f.openAlone();
    morph.play(true);
    assert.equal(f.capsule.dataset.morph, 'open');
    assert.ok(f.surface(), 'the glass is its own layer while it moves');
    const [glass] = f.on(f.surface());
    const s = MORPH.shadow;
    assert.deepEqual(glass.keyframes.map((frame) => frame.clipPath),
        [`inset(-${s}px 82px -${s}px 206px round 999px)`, `inset(-${s}px 0px -${s}px 0px round 999px)`],
        'from where the closed capsule stood (206 px in, 82 px short of the end) to the whole strip');
    assert.equal(glass.options.duration, MORPH.open);
    assert.deepEqual(f.on(f.head)[0].keyframes, [{ transform: 'translateX(206px)' }, { transform: 'none' }]);
    assert.equal(f.on(f.head)[0].options.duration, MORPH.open, 'the head keeps pace with the glass');
    assert.equal(f.on(f.body).length, 1, 'the strip comes in');
    assert.deepEqual(f.segs.map((seg) => f.on(seg)[0].options.delay), [70, 82, 94], 'the levels one after another');
    // They only fade: each level stands under the selection's pill from the first frame (nothing slides into place).
    for (const seg of f.segs) assert.deepEqual(f.on(seg)[0].keyframes, [{ opacity: 0 }, { opacity: 1 }]);
    // The controls that stepped aside stand where they were, out of the line's flow, and fade.
    assert.equal(f.row.dataset.ownerLeaving, '');
    for (const [kid, left, width] of [[f.menu, 12, 32], [f.subagents, 50, 70], [f.context, 126, 80]]) {
        assert.equal(kid.dataset.capsuleLeaving, '');
        assert.equal(kid.css.get('--leave-x'), `${left - 6}px`);
        assert.equal(kid.css.get('--leave-w'), `${width}px`);
        assert.equal(kid.css.get('--leave-display'), 'flex');
        const [fade] = f.on(kid);
        assert.deepEqual(fade.keyframes, [{ opacity: 1 }, { opacity: 0 }]);
        assert.equal(fade.options.duration, MORPH.leave);
    }
    // A control whose fade has ended leaves the page at once, while the glass still grows; the others still stand.
    f.on(f.subagents)[0].finish();
    await settle();
    assert.equal(f.subagents.dataset.capsuleLeaving, undefined);
    assert.equal(f.subagents.css.has('--leave-x'), false);
    assert.equal(f.menu.dataset.capsuleLeaving, '');
    assert.equal(glass.playState, 'running');
    glass.finish();
    await settle();
    assert.equal(f.surface(), null, 'the capsule paints its own surface again');
    assert.equal(f.capsule.dataset.morph, undefined);
    assert.equal(f.capsule.style.transition, '', 'its colour transition is back');
    assert.equal(f.row.dataset.ownerLeaving, undefined);
    for (const kid of [f.menu, f.subagents, f.context]) {
        assert.equal(kid.dataset.capsuleLeaving, undefined);
        assert.equal(kid.css.has('--leave-x'), false);
    }
    assert.ok(f.log.every((animation) => animation.playState !== 'running'), 'nothing left running');
});

test('closing alone runs back, faster: the glass folds into the closed capsule and the others fade in', async () => {
    const f = fixture();
    f.openAlone();
    const morph = beginMorph({ el: f.capsule, row: f.row, win: f.win });
    f.closed();
    morph.play(false);
    assert.equal(f.capsule.dataset.morph, 'close');
    const surface = f.surface();
    assert.equal(surface.css.get('--morph-left'), '-206px', 'the glass starts as the open strip around the closed capsule');
    assert.equal(surface.css.get('--morph-width'), '378px');
    const [glass] = f.on(surface);
    const s = MORPH.shadow;
    assert.deepEqual(glass.keyframes.map((frame) => frame.clipPath),
        [`inset(-${s}px 0px -${s}px 0px round 999px)`, `inset(-${s}px 82px -${s}px 206px round 999px)`]);
    assert.equal(glass.options.duration, MORPH.close);
    assert.ok(MORPH.close < MORPH.open);
    assert.deepEqual(f.on(f.head)[0].keyframes, [{ transform: 'translateX(-206px)' }, { transform: 'none' }]);
    for (const kid of [f.menu, f.subagents, f.context]) assert.deepEqual(f.on(kid)[0].keyframes, [{ opacity: 0 }, { opacity: 1 }]);
    assert.equal(f.row.dataset.ownerLeaving, undefined, 'nothing leaves on a close');
    glass.finish();
    await settle();
    assert.equal(f.surface(), null);
    assert.equal(f.capsule.dataset.morph, undefined);
});

test('beside the others, under reduced motion, without layout or Web Animations nothing moves', () => {
    const inline = fixture();
    const morph = beginMorph({ el: inline.capsule, row: inline.row, win: inline.win });
    inline.capsule.dataset.fit = 'inline';
    inline.capsule.dataset.open = 'true';
    morph.play(true);
    assert.equal(inline.surface(), null, 'beside the others the width moves (style.css)');
    assert.equal(inline.log.length, 0);
    const closing = beginMorph({ el: inline.capsule, row: inline.row, win: inline.win });
    inline.closed();
    closing.play(false);
    assert.equal(inline.log.length, 0, 'a strip that stood beside the others closes by its width too');

    const reduced = fixture({ reduced: true });
    assert.equal(beginMorph({ el: reduced.capsule, row: reduced.row, win: reduced.win }), null);
    const flat = fixture();
    flat.row.rect = { left: 0, top: 0, width: 0, height: 0, right: 0, bottom: 0 };
    assert.equal(beginMorph({ el: flat.capsule, row: flat.row, win: flat.win }), null, 'a page without layout');
    const old = fixture();
    delete old.capsule.animate;
    assert.equal(beginMorph({ el: old.capsule, row: old.row, win: old.win }), null);
});

test('a morph still running ends where it was going when the next one starts, or when the capsule goes', async () => {
    const f = fixture();
    const opening = beginMorph({ el: f.capsule, row: f.row, win: f.win });
    f.openAlone();
    opening.play(true);
    assert.ok(f.surface() && f.menu.dataset.capsuleLeaving === '');
    // A quick second press: the open morph ends first, so the close measures the line as it stands.
    const closing = beginMorph({ el: f.capsule, row: f.row, win: f.win });
    assert.equal(f.surface(), null);
    assert.equal(f.menu.dataset.capsuleLeaving, undefined);
    assert.equal(f.row.dataset.ownerLeaving, undefined);
    f.closed();
    closing.play(false);
    assert.equal(f.capsule.dataset.morph, 'close');
    stopMorph(f.capsule);
    assert.equal(f.surface(), null);
    assert.equal(f.capsule.dataset.morph, undefined);
    assert.ok(f.log.every((animation) => animation.playState !== 'running'));
    await settle();
});
