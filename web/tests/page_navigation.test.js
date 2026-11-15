import assert from 'node:assert/strict';
import test from 'node:test';
import { createLayerStack } from '../modules/layers.js';
import { createPageNavigation } from '../modules/page_navigation.js';

const settle = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };

test('jumps stack after the veto, section choices retain them, the map clears them, and Back restores each retained place atomically', async () => {
    const history = [], changes = [];
    const layers = createLayerStack({ win: null, doc: null, history: {
        pushState: (...args) => history.push(['push', ...args]), back: () => history.push(['back']),
    } });
    let place = { key: 'settings:tools', label: 'Settings', title: 'Settings › Tools', scroll: 240, focus: 'Skills row' };
    let allow = true;
    const doc = { activeElement: {}, addEventListener() {}, removeEventListener() {} };
    const nav = createPageNavigation({ layers, doc, canLeave: async () => allow, capture: () => {
        const saved = { ...place };
        return { ...saved, restore() { place = saved; layers.push({ id: 'tools', page: true, focus: false }); } };
    } });
    layers.push({ id: 'tools', page: true, focus: false });
    layers.onChange(({ ids }) => changes.push(ids));
    await settle();
    const move = key => () => { layers.remove('tools'); place = { key, label: key, scroll: 0 }; };
    allow = false;
    assert.equal(await nav.go('skills', move('skills')), false);
    assert.deepEqual(layers.ids(), ['tools']);
    assert.equal(place.scroll, 240, 'a veto changes neither the place nor the stack');
    allow = true;
    await nav.go('skills', move('skills'));
    assert.deepEqual(layers.ids(), ['page-return-1']);
    assert.equal(layers.page().back.title, 'Settings › Tools');
    await nav.go('skills:browse', move('skills:browse'), { map: 'section' });
    assert.equal(layers.size(), 1, 'tabs keep the return without making a step');
    place.scroll = 120;
    await nav.go('dashboard:costs', move('dashboard:costs'));
    assert.equal(layers.size(), 2);
    layers.back(); await settle();
    assert.equal(place.key, 'skills:browse');
    assert.equal(place.scroll, 120);
    layers.remove('tools');
    layers.back(); await settle();
    assert.deepEqual(place, { key: 'settings:tools', label: 'Settings', title: 'Settings › Tools', scroll: 240, focus: 'Skills row' });
    assert.ok(changes.every(ids => ids.length), 'one-column jumps and returns never expose an empty stack');
    assert.deepEqual(history, [['push', { ouroLayers: 1 }, '']], 'no consumed sentinel, URL or second history');
    const pending = nav.capture();
    place.scroll = 500;
    await nav.go('skills', move('skills'), { origin: pending });
    assert.equal(layers.returnPage().returnTo.scroll, 240, 'an asynchronous destination keeps the place at the press');
    await nav.go('widgets', move('widgets'), { map: true });
    await settle();
    assert.deepEqual(layers.ids(), []);
    assert.equal(history.at(-1)[0], 'back', 'a map choice consumes the one sentinel');
    layers.destroy(); nav.destroy();
});

test('a vetoed asynchronous Back keeps the same return and a second press does not race it', async () => {
    let answer, restores = 0;
    const layers = createLayerStack({ win: null, doc: null });
    layers.push({ id: 'return', page: true, focus: false,
        onClose: () => new Promise(resolve => { answer = resolve; }).then(allowed => { if (allowed) restores++; return allowed; }) });
    layers.back(); const first = answer;
    layers.back(); assert.equal(answer, first);
    answer(false); await settle();
    assert.equal(layers.size(), 1);
    layers.back(); answer(true); await settle();
    assert.equal(layers.size(), 0); assert.equal(restores, 1);
    layers.destroy();
});
