import test from 'node:test';
import assert from 'node:assert/strict';
import { bindConversationControls, projectRoomOptions, startConversations } from '../modules/conversation_controls.js';

test('conversation mounts receive the room label, reveal, visibility and settled callback from navigation', () => {
    let options, shown = false, active = true, pending = false, destroyed = 0, revealed = 0, released = 0;
    const entry = { impl: { mountRoom(value) { options = value; return {
        retains: () => active, destroy: () => destroyed++,
    }; } } };
    const runtime = { onProvidersChange() {}, clientProviders: () => [entry] };
    startConversations({ hosts: runtime, openSettings() {}, roomOptions: ({ chatId }) => ({
        label: () => `Project ${chatId}`, onScreen: () => shown,
        reveal() { shown = true; revealed++; }, onSettled() { released++; },
    }) });
    const composer = { hasPendingWork: () => pending };
    const dispose = bindConversationControls({ composer, page: { getClientRects: () => [1] }, chatId: 71 });
    try {
        assert.equal(options.label?.(), 'Project 71');
        assert.equal(options.onScreen(), false);
        options.reveal();
        assert.equal(revealed, 1);
        assert.equal(options.onScreen(), true);
        assert.equal(composer.hasPendingWork(), true);
        active = false;
        assert.equal(composer.hasPendingWork(), false);
        pending = true;
        assert.equal(composer.hasPendingWork(), true);
        options.onSettled();
        assert.equal(released, 1);
    } finally { dispose(); }
    assert.equal(destroyed, 1);
    assert.equal(composer.hasPendingWork(), true, 'original pending-work owner restored');
});

test('project room options name Main and a Project, reveal the right page and release a project left behind', async () => {
    const nav = { activeProjectId: 'p7' }, state = { activePage: 'chat' };
    const rows = [{ id: 'p7', chat_id: 7, name: 'Seven' }];
    const opened = [], shown = [], destroyed = [];
    const options = projectRoomOptions({
        rows: () => rows, nav, state, title: { textContent: 'Panel title' },
        freshProjectRow: (project) => rows.find(row => row.id === project.id) || project,
        openProjectPanel: (project) => opened.push(project.id), showPage: (name) => shown.push(name),
        destroyProjectInstance: (id) => destroyed.push(id),
    });
    const main = options({ page: { hidden: false }, chatId: 1 });
    assert.equal(main.label(), 'Main');
    assert.equal(main.onScreen(), false);  // a project is active, so Main is not on screen
    main.reveal();
    assert.deepEqual(shown, ['chat']);
    const seven = options({ page: { hidden: false }, chatId: 7 });
    assert.equal(seven.label(), 'Seven');
    assert.equal(seven.onScreen(), true);
    seven.reveal();
    assert.deepEqual(opened, ['p7']);
    nav.activeProjectId = 'p9';  // the owner opened an unlisted Project directly: its panel is the active one
    const unlisted = options({ page: { hidden: true }, chatId: 9 });  // not in the census
    assert.equal(unlisted.label(), 'Panel title');
    assert.equal(unlisted.onScreen(), false);
    seven.onSettled();
    await new Promise(resolve => queueMicrotask(resolve));
    assert.deepEqual(destroyed, ['p7']);  // the project the owner left is released once the call settled
    main.onSettled();
    await new Promise(resolve => queueMicrotask(resolve));
    assert.deepEqual(destroyed, ['p7']);  // Main never destroys anything
});
