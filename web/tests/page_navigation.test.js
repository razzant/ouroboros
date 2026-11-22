import assert from 'node:assert/strict';
import test from 'node:test';
import { createLayerStack } from '../modules/layers.js';
import { capturePagePlace, createPageNavigation, canLeavePage, openInitialPage } from '../modules/page_navigation.js';

const settle = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };

test('the pressed opener survives a microtask between click listeners but never outlives the dispatch', async () => {
    let onClick;
    const focused = {}, button = {}, target = { closest: () => button };
    const doc = { activeElement: focused, addEventListener: (_, handler) => { onClick = handler; }, removeEventListener() {} };
    const nav = createPageNavigation({ doc, capture: opener => opener });
    const event = { target, eventPhase: 1 };
    onClick(event);
    await settle(); // WebKit may checkpoint before the destination's listener; a pointer does not focus its button.
    event.eventPhase = 3;
    assert.equal(nav.capture(), button);
    event.eventPhase = 0;
    assert.equal(nav.capture(), focused, 'a later programmatic move cannot borrow that old click');
    nav.destroy();
});

test('a jump made from a key focuses the destination\'s round Back; a pointer\'s jump and a named row do not', async () => {
    const layers = createLayerStack({ win: null, doc: null });
    let onClick;
    const back = { focus() { doc.activeElement = back; } };
    const doc = { activeElement: null, addEventListener: (_, handler) => { onClick = handler; }, removeEventListener() {},
        querySelector: (selector) => (selector === `[data-shell-back][data-page-layer="${layers.returnPage()?.id}"]` ? back : null) };
    const nav = createPageNavigation({ layers, doc, canLeave: () => true,
        capture: () => ({ key: 'chat', label: 'Main', restore: () => {} }) });
    for (const [detail, row, focused] of [[0, '', true], [1, '', false], [0, 'TOTAL_BUDGET', false]]) {
        doc.activeElement = null;
        const event = { target: { closest: () => null }, eventPhase: 1, detail };
        onClick(event);
        await nav.go('skills', () => {}, { row });
        event.eventPhase = 0;
        assert.equal(doc.activeElement === back, focused, `detail ${detail}, row "${row}"`);
        layers.back(); await settle();
    }
    nav.destroy(); layers.destroy();
});

test('a sidebar choice keeps the focus on its row, the opener a Project returns to; a jump lets the page it leaves go', async () => {
    const body = {}, doc = { body, activeElement: body, addEventListener() {}, removeEventListener() {} };
    const control = () => ({ isConnected: true, closest: () => null,
        focus() { doc.activeElement = this; }, blur() { if (doc.activeElement === this) doc.activeElement = body; } });
    const row = control(), link = control(), feed = control(), panel = { contains: node => node === feed };
    const layers = createLayerStack({ win: null, doc });
    const nav = createPageNavigation({ layers, doc, canLeave: () => true,
        capture: () => ({ key: 'chat', label: 'Main', restore: () => () => {} }) });
    row.focus();  // Enter on the sidebar's Skills
    await nav.go('skills', () => {}, { map: true });
    assert.equal(doc.activeElement, row, 'a sidebar choice keeps the focus on the row pressed');
    for (const [pressed, map] of [[row, true], [link, false]]) {
        let seen = null;
        pressed.focus();
        await nav.go('project:one', () => {
            seen = doc.activeElement;
            layers.push({ id: 'project', el: panel, modal: false, focus: () => feed.focus(), onClose: () => layers.remove('project') });
        }, { map });
        assert.equal(seen, map ? row : body, 'the focus the commit\'s layer records as its opener');
        assert.equal(doc.activeElement, feed);
        layers.back('escape', 'project'); await settle();
        assert.equal(doc.activeElement, map ? row : feed, 'Escape returns the focus to the row pressed');
        layers.batch(() => nav.clear());
    }
    nav.destroy(); layers.destroy();
});

test('a jump and its return let the room keep its selection before their blur; a sidebar choice blurs nothing', async () => {
    const calls = [], body = {}, doc = { body, activeElement: body, addEventListener() {}, removeEventListener() {} };
    const field = { blur() { calls.push('blur'); doc.activeElement = body; }, focus() { doc.activeElement = field; } };
    const layers = createLayerStack({ win: null, doc: null });
    const nav = createPageNavigation({ layers, doc, canLeave: () => true, leave: () => calls.push('leave'),
        capture: () => ({ key: 'chat', label: 'Main', restore: () => () => {} }) });
    field.focus();  // a draft in Main, a passage selected beside it (WebKit ends it with the blur)
    await nav.go('project:one', () => {});
    assert.deepEqual(calls, ['leave', 'blur'], 'the jump keeps the room\'s selection first');
    field.focus();
    layers.back('escape'); await settle();
    assert.deepEqual(calls, ['leave', 'blur', 'leave', 'blur'], 'and so does its return');
    calls.length = 0;
    field.focus();
    await nav.go('skills', () => {}, { map: true });
    assert.deepEqual(calls, [], 'a sidebar choice neither blurs nor needs to keep');
    nav.destroy(); layers.destroy();
});

test('only keyboard dismissals ask the returned place to show its focus ring', async () => {
    const layers = createLayerStack({ win: null, doc: null }), restored = [];
    const doc = { addEventListener() {}, removeEventListener() {} };
    const nav = createPageNavigation({ layers, doc, canLeave: () => true,
        capture: () => ({ key: 'chat', label: 'Main', restore: () => input => restored.push(input) }) });
    for (const reason of ['pointer', 'keyboard', 'escape', 'shortcut', 'telegram', 'history']) {
        await nav.go('skills', () => {}); layers.back(reason); await settle();
        assert.deepEqual(restored.at(-1), { keyboard: ['keyboard', 'escape', 'shortcut'].includes(reason) });
    }
    nav.destroy(); layers.destroy();
});

test('only a forward jump without a named row lands at the top; vetoes, map choices and returns keep their place', async () => {
    const layers = createLayerStack({ win: null, doc: null });
    const calls = [], doc = { addEventListener() {}, removeEventListener() {} };
    let allow = true;
    const nav = createPageNavigation({ layers, doc, canLeave: () => allow, land: () => calls.push('top'),
        capture: () => ({ key: 'chat', label: 'Main', restore: () => { calls.push('restore'); } }) });
    const commit = () => calls.push('show');
    await nav.go('skills', commit);
    assert.deepEqual(calls.splice(0), ['show', 'top']);
    layers.back(); await settle();
    assert.deepEqual(calls.splice(0), ['restore']);
    for (const options of [{ map: true }, { map: 'section' }, { row: 'TOTAL_BUDGET' }]) {
        await nav.go('dashboard:costs', commit, options);
        assert.deepEqual(calls.splice(0), ['show']);
    }
    await nav.go('chat', commit);
    assert.deepEqual(calls.splice(0), ['show'], 'reopening the same place is not a forward jump');
    allow = false;
    await nav.go('skills', commit);
    assert.deepEqual(calls, []);
    nav.destroy(); layers.destroy();
});

test('a captured place restores its retained owners before their independent reading snapshots', () => {
    for (const page of ['settings', 'dashboard', 'skills', 'chat']) {
        const restored = [], opener = {}, root = {}, state = { activePage: page, settingsPath: 'tools', dashboardActiveSubtab: 'logs' };
        const owner = { capture: () => ({ label: page === 'settings' ? 'Tools' : 'Logs', title: page === 'settings' ? 'Settings › Tools' : 'Dashboard › Logs', restore: () => restored.push('level') }) };
        const captured = capturePagePlace({ state, navState: {}, settingsControls: owner, dashboardControls: owner,
            skillsControls: { capture: () => ({ key: 'skills:tools/docs', label: 'Docs', title: 'Extensions › MCP servers › Docs',
                retains: el => el === root, release: () => restored.push('release'), restore: () => restored.push('level') }) },
            mainChat: { captureReading: () => () => restored.push('feed') }, logsControls: { captureReading: () => () => restored.push('logs') },
            doc: { getElementById: () => root }, restorePlaces: { capture: (place, pressed) => {
                assert.equal(place, root); assert.equal(pressed, opener); return () => restored.push('scroll/focus');
            } }, commitPage: (name, options) => { assert.equal(name, page); assert.equal(options.reveal, true); options.enter(); },
        }, opener);
        state.activePage = 'widgets'; state.settingsPath = ''; state.dashboardActiveSubtab = 'costs';
        const reading = captured.restore();
        assert.deepEqual(restored, ['settings', 'dashboard', 'skills'].includes(page) ? ['level'] : []);
        reading();
        assert.ok(restored.includes('scroll/focus'));
        assert.equal(restored.includes('feed'), page === 'chat');
        assert.equal(restored.includes('logs'), page === 'dashboard');
        assert.equal(captured.key, ({ settings: 'settings:tools', dashboard: 'dashboard:logs', skills: 'skills:tools/docs', chat: 'chat' })[page]);
        assert.equal(captured.label, ({ settings: 'Tools', dashboard: 'Logs', skills: 'Docs', chat: 'Main' })[page]);
        assert.equal(captured.title, ({ settings: 'Settings › Tools', dashboard: 'Dashboard › Logs', skills: 'Extensions › MCP servers › Docs', chat: 'Main' })[page],
            'a section\'s return names the section, then its page');
        if (page === 'skills') {
            assert.equal(captured.retains(root), true);
            captured.release();
            assert.equal(restored.at(-1), 'release');
        }
    }
});

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

test('boot accepts only a mounted destination and never adds a jump or reopens the current page', () => {
    const calls = [], show = (...args) => calls.push(args);
    const doc = { getElementById: id => ['page-settings', 'page-chat'].includes(id) };
    for (const hash of ['', '#missing', '#chat', '# settings ']) openInitialPage(show, 'chat', { location: { hash } }, doc);
    assert.deepEqual(calls, [['settings', { map: true }]]);
});

test('leave guards preserve section moves, map Projects to Main, and stop at the first refusal', async () => {
    const calls = [], state = { activePage: 'settings' };
    const guard = async place => { calls.push(place); return false; };
    assert.equal(await canLeavePage('settings:tools', state, [guard]), true);
    assert.deepEqual(calls, []);
    assert.equal(await canLeavePage('project:one', state, [guard, () => assert.fail('after refusal')]), false);
    assert.deepEqual(calls, [{ from: 'settings', to: 'chat' }]);
    assert.equal(await canLeavePage('dashboard:costs', state, [() => true]), true);
});


test('a source claim is checked after leave permission and handed over when its destination commits', async () => {
    let answer, current = true, commits = 0;
    const layers = createLayerStack({ win: null, doc: null });
    const doc = { activeElement: {}, addEventListener() {}, removeEventListener() {} };
    const nav = createPageNavigation({ layers, doc, canLeave: () => new Promise(resolve => { answer = resolve; }),
        capture: () => ({ key: 'chat', label: 'Main', restore() {} }) });
    const commit = () => { commits++; current = false; }; // covering Main ends the origin's visible claim
    const cancelled = nav.go('project:one', commit, { current: () => current });
    current = false;
    answer(true);
    assert.equal(await cancelled, false);
    assert.equal(commits, 0);
    assert.equal(layers.size(), 0, 'a superseded read creates neither a page nor a return');
    current = true;
    const accepted = nav.go('project:two', commit, { current: () => current });
    answer(true);
    assert.equal(await accepted, true, 'the commit may cover the origin without cancelling its own landing');
    assert.equal(commits, 1);
    assert.equal(layers.returnPage().destination, 'project:two');
    nav.destroy(); layers.destroy();
});
