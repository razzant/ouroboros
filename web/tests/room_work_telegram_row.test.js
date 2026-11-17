import assert from 'node:assert/strict';
import test from 'node:test';
import { createRoomWork } from '../modules/room_work.js';
import { emptyWorkState, reconcileWorkState } from '../modules/work_status.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';
import { projection, room } from './work_fixture.js';

// Main's bar as the product builds it: the title row (where the status goes), the end group with the consciousness chip.
function mountMain(doc, mount) {
    const node = (className, parent) => { const el = doc.createElement('div'); el.className = className; parent.appendChild(el); return el; };
    const page = node('page', mount);
    const input = doc.createElement('textarea');
    input.id = 'chat-input';
    page.appendChild(input);
    const header = node('chat-page-header', page);
    node('app-page-title-row', node('app-page-title-block', header));
    const group = node('app-page-actions ui-chrome-group', header);
    const chip = doc.createElement('button');
    chip.className = 'cx-chip ui-chrome-control';
    chip.setAttribute('aria-controls', 'cx-shade');
    group.appendChild(chip);
    const badge = doc.createElement('span');
    badge.className = 'status-badge';
    badge.textContent = 'Online';
    Object.assign(badge.dataset, { tone: 'online', mark: '', owner: '', amber: '' });
    group.appendChild(badge);
    const messages = node('chat-messages', page);
    return { page, header, group, chip, badge, messages };
}

test('in Telegram\'s shared row the consciousness chip is a row of the work list; it goes home with the row', () => {
    const { prior, mount } = installDom(async () => ({ ok: true, json: async () => ({}) }));
    let work = null;
    try {
        const doc = globalThis.document;
        const root = doc.documentElement;
        const { page, header, group, chip, badge, messages } = mountMain(doc, mount);
        Object.assign(root.dataset, { telegramFullscreen: 'active', layout: 'stack' });
        work = createRoomWork({ page, statusBadge: badge, messagesDiv: messages, ws: { isConnected: () => true },
            reading: { stick: true, cancel() {}, scroll() {} },
            works: () => reconcileWorkState(emptyWorkState(), { supervisor_ready: true, ...projection(1, [], room('online')) }, 1),
            liveCardRecords: new Map(), pendingSubmissions: new Map() });
        const extras = header.querySelector('.rw-extras');
        assert.equal(chip.parentNode, extras, 'in Telegram\'s row the chip is a row of the work list');
        assert.equal(extras.children[0], chip, 'it comes before Restart and the budget');
        // The room's capsule is then the drawn way to the chip's sheet: a return from a page the sheet opened lands on it.
        const trigger = header.querySelector('.rw-trigger');
        const controls = () => trigger.getAttribute('aria-controls').split(' ');
        assert.deepEqual(controls().slice(1), ['cx-shade'], 'in the row the capsule names the chip\'s sheet too');
        // Telegram leaves fullscreen: the bar is the page's own again and the chip its capsule.
        root.dataset.telegramFullscreen = 'expanded';
        work.update();
        assert.equal(chip.parentNode, group, 'outside the shared row the chip is the bar\'s capsule');
        assert.equal(controls().length, 1, 'and the capsule names its own list alone');
        assert.equal(group.children[0], chip, 'it leads the end group again');
        root.dataset.telegramFullscreen = 'active';
        work.update();
        assert.equal(chip.parentNode, extras);
        // A docked column inside Telegram keeps its own bar: no shared row there.
        root.dataset.layout = 'docked';
        work.update();
        assert.equal(chip.parentNode, group, 'a docked layout keeps the chip in its bar');
        root.dataset.layout = 'stack';
        work.update();
        assert.equal(chip.parentNode, extras);
        // The room leaves the screen: the chip goes back to its bar, so the next room's list never carries it.
        page.hidden = true;
        work.update();
        assert.equal(chip.parentNode, group, 'a hidden room takes its chip home');
    } finally {
        work?.destroy?.();
        restoreDom(prior);
    }
});

test('the bar follows Telegram\'s fullscreen when the host tells it, and a room destroyed in the row gives the chip back', () => {
    const { prior, mount } = installDom(async () => ({ ok: true, json: async () => ({}) }));
    let work = null;
    const heard = new Map();
    const realAdd = globalThis.window.addEventListener;
    try {
        globalThis.window.addEventListener = (type, fn) => { heard.set(type, fn); };
        const doc = globalThis.document;
        const root = doc.documentElement;
        const { page, header, group, chip, badge, messages } = mountMain(doc, mount);
        Object.assign(root.dataset, { telegramFullscreen: 'expanded', layout: 'stack' });
        work = createRoomWork({ page, statusBadge: badge, messagesDiv: messages, ws: { isConnected: () => true },
            reading: { stick: true, cancel() {}, scroll() {} },
            works: () => reconcileWorkState(emptyWorkState(), { supervisor_ready: true, ...projection(1, [], room('online')) }, 1),
            liveCardRecords: new Map(), pendingSubmissions: new Map() });
        const extras = header.querySelector('.rw-extras');
        assert.equal(chip.parentNode, group, 'not in fullscreen: the chip is the bar\'s capsule');
        assert.ok(heard.has('ouro:host-chrome'), 'the room listens to the host');
        root.dataset.telegramFullscreen = 'active';
        heard.get('ouro:host-chrome')();
        assert.equal(chip.parentNode, extras, 'the host said fullscreen: the chip is a row of the work list');
        // A Project's bar outlives its rooms: a room destroyed while the chip is its row gives the chip back first.
        work.destroy();
        work = null;
        assert.equal(chip.parentNode, group, 'the destroyed room\'s list never takes the bar\'s chip with it');
    } finally {
        globalThis.window.addEventListener = realAdd;
        work?.destroy?.();
        restoreDom(prior);
    }
});

test('in the shared row Main\'s title shows whole or not at all; elsewhere it keeps its ellipsis', () => {
    const { prior, mount } = installDom(async () => ({ ok: true, json: async () => ({}) }));
    let work = null;
    const heard = new Map();
    const realAdd = globalThis.window.addEventListener;
    try {
        globalThis.window.addEventListener = (type, fn) => { heard.set(type, fn); };
        const doc = globalThis.document;
        const root = doc.documentElement;
        const { page, header, badge, messages } = mountMain(doc, mount);
        const titleRow = header.querySelector('.app-page-title-row');
        const title = doc.createElement('h1');
        title.className = 'app-page-title';
        title.textContent = 'Main Chat';
        titleRow.appendChild(title);
        // The title's box is narrower than its words: the row would cut it.
        Object.defineProperty(title, 'scrollWidth', { get: () => 75 });
        Object.defineProperty(title, 'clientWidth', { get: () => 40 });
        Object.assign(root.dataset, { telegramFullscreen: 'active', layout: 'stack' });
        work = createRoomWork({ page, statusBadge: badge, messagesDiv: messages, ws: { isConnected: () => true },
            reading: { stick: true, cancel() {}, scroll() {} },
            works: () => reconcileWorkState(emptyWorkState(), { supervisor_ready: true, ...projection(1, [], room('online')) }, 1),
            liveCardRecords: new Map(), pendingSubmissions: new Map() });
        assert.equal(titleRow.dataset.title, 'off', 'a title the shared row would cut is not shown at all');
        root.dataset.telegramFullscreen = 'expanded';
        heard.get('ouro:host-chrome')();
        assert.equal(titleRow.dataset.title, undefined, 'outside the shared row the title keeps its ellipsis');
    } finally {
        globalThis.window.addEventListener = realAdd;
        work?.destroy?.();
        restoreDom(prior);
    }
});
