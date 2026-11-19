import test from 'node:test';
import assert from 'node:assert/strict';
import { folderEffects, openProjectFolderDialog } from '../modules/project_folder.js';
import { installDomDouble } from './dom_double.js';

// What the Working folder… dialog says before a change is saved (web/modules/project_folder.js): work already
// running or waiting keeps its folder; only work moved into the project that has not restarted since takes the new
// default, and the sentence names that work and where its next start lands.

const view = (work, folder = '/Users/owner/gone') => ({ folder, work });
const WEEKLY = { task_id: 'weekly', name: 'Weekly defaults review', system_repo: true };
const NOTES = { task_id: 'notes', name: 'Reading notes', system_repo: false };

test('work that keeps its folder and moved work are told apart, with the address the moved work will take', () => {
    const seen = view({ observed: true, kept: 2, moved: [WEEKLY, NOTES], moved_total: 2 });
    assert.deepEqual(folderEffects(seen, 'clear'), [
        'Only the link to this folder is removed. Files, history and results stay where they are.',
        '2 tasks already running or waiting keep their folders. They are not moved.',
        "Moved here and not restarted yet: Weekly defaults review. The next start stays on Ouroboros's own repository.",
        'Moved here and not restarted yet: Reading notes. The next start gets no folder from this project and works in '
            + 'its own temporary area, even if it already ran elsewhere.',
    ]);
    // A chosen folder would take work on Ouroboros itself away from its repository: said apart, with the way to keep it.
    assert.deepEqual(folderEffects(seen, 'choose', '/Users/owner/new'), [
        '2 tasks already running or waiting keep their folders. They are not moved.',
        "Moved here and not restarted yet: Weekly defaults review. The next start would work in /Users/owner/new, not "
            + "on Ouroboros's own repository where it ran. No folder keeps it there.",
        'Moved here and not restarted yet: Reading notes. The next start works in /Users/owner/new, even if it already '
            + 'ran elsewhere.',
    ]);
    // Before a folder is selected the sentence still says where the moved work will go.
    assert.match(folderEffects(seen, 'choose')[2], /works in the folder you choose, even if/);
});

test('one task reads in the singular, and a list cut short says how many more', () => {
    const one = view({ observed: true, kept: 1, moved: [], moved_total: 0 }, '');
    assert.deepEqual(folderEffects(one, 'clear'), ['1 task already running or waiting keeps its folder. It is not moved.']);
    const many = view({ observed: true, kept: 0, moved: [NOTES, { task_id: 't2', name: '', system_repo: false }], moved_total: 5 }, '');
    assert.deepEqual(folderEffects(many, 'choose', '/new'), [
        'Moved here and not restarted yet: Reading notes, t2. The next start works in /new, even if it already ran elsewhere.',
        '3 more moved tasks are not listed; the same holds for them.',
    ]);
    assert.equal(folderEffects({ ...many, work: { ...many.work, moved_total: 3 } }, 'clear').at(-1),
        '1 more moved task is not listed; the same holds for it.');
});

test('nothing queued says nothing, and an unreadable queue is said as unknown, never as no work', () => {
    assert.deepEqual(folderEffects(view({ observed: true, kept: 0, moved: [], moved_total: 0 }, ''), 'clear'), []);
    assert.deepEqual(folderEffects(view({ observed: false, kept: 0, moved: [], moved_total: 0 }, ''), 'choose', '/new'),
        ['Running and waiting tasks could not be read, so it is unknown which of them keep their folder.']);
});

// Lifecycle only; the VM browser cases own real focus, layout, transport and the registry.
const turn = () => new Promise((resolve) => setImmediate(resolve));
const deferred = () => {
    let resolve, reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    return { promise, resolve, reject };
};
const missing = (expected = 'before') => ({ state: 'missing', folder: '/gone', expected, actions: ['clear', 'choose'],
    work: { observed: true, kept: 0, moved: [], moved_total: 0 } });
const usable = { ...missing('after'), state: 'usable', folder: '/new', actions: [] };

function dialogHarness(t) {
    const dom = installDomDouble();
    dom.doc.parseHtml = true;
    const create = dom.doc.createElement;
    dom.doc.createElement = (tag) => {
        const node = create(tag), append = node.append.bind(node);
        node.append = (...items) => append(...items.map((item) => typeof item === 'string' ? dom.doc.createTextNode(item) : item));
        return node;
    };
    t.after(() => dom.restore());
    const post = deferred(), reads = [], writes = [];
    const apiClient = {
        projectFolder: async (id) => { reads.push(id); return { folder: missing() }; },
        projectFolderChange: (id, body) => { writes.push([id, body]); return post.promise; },
        fsDirs: async () => ({ path: '/home', dirs: [] }),
    };
    const open = () => openProjectFolderDialog({ apiClient, project: { id: 'room', name: 'Room' } });
    const dialog = () => dom.doc.body.querySelector('[role="dialog"]');
    const save = () => {
        dialog().querySelectorAll('input').find((input) => input.value === 'clear').emit('change');
        dialog().querySelector('[data-pf-save]').emit('click');
    };
    return { ...dom, open, dialog, apiClient, post, reads, writes, save };
}

for (const dismiss of ['button', 'escape', 'backdrop']) {
    for (const late of ['success', 'failure']) {
        test(`a stalled Save permits ${dismiss}; fresh read survives late ${late} with no replay`, async (t) => {
            const h = dialogHarness(t);
            const answer = h.open();
            await turn();
            h.save();
            assert.equal(h.writes.length, 1);
            assert.match(h.dialog().querySelector('[data-pf-error]').textContent, /closing does not cancel/);
            assert.equal(h.dialog().querySelector('[data-pf-cancel]').textContent, 'Close');
            assert.equal(h.dialog().querySelector('[data-pf-save]').disabled, true);
            if (dismiss === 'escape') h.dialog().emit('keydown', { key: 'Escape' });
            else if (dismiss === 'backdrop') h.dialog().parentElement.emit('click');
            else h.dialog().querySelector('[data-pf-cancel]').emit('click');
            assert.equal(h.dialog(), null, 'closure does not wait for the POST');
            assert.equal(await answer, false);
            assert.match(h.doc.getElementById('toast-stack').textContent, /Save has not been confirmed/);
            const read = deferred();
            h.apiClient.projectFolder = (id) => { h.reads.push(id); return read.promise; };
            const reopened = h.open();
            assert.equal(h.dialog().querySelector('[data-pf-save]').disabled, true);
            read.resolve({ folder: usable });
            await turn();
            const fresh = h.dialog();
            assert.equal(h.reads.length, 2);
            assert.equal(fresh.querySelector('[data-pf-path]').textContent, '/new');
            if (late === 'success') h.post.resolve({});
            else h.post.reject(new Error('late broken response'));
            await turn();
            assert.equal(h.dialog(), fresh, 'a disposed request never closes the replacement dialog');
            assert.equal(fresh.querySelector('[data-pf-error]').textContent, '');
            assert.equal(h.writes.length, 1);
            fresh.querySelector('[data-pf-cancel]').emit('click');
            await reopened;
        });
    }
}

test('a confirmed Save closes once with the original expected word', async (t) => {
    const h = dialogHarness(t);
    const answer = h.open();
    await turn();
    h.save();
    h.post.resolve({});
    assert.equal(await answer, true);
    assert.equal(h.dialog(), null);
    assert.deepEqual(h.writes, [['room', { action: 'clear', expected: 'before' }]]);
    assert.equal(h.doc.getElementById('toast-stack'), null);
});

for (const status of [400, 409, 500, 0]) {
    test(`Save failure ${status} preserves its meaning and reads unknown/stale state without replay`, async (t) => {
        const h = dialogHarness(t);
        const answer = h.open();
        await turn();
        const read = deferred();
        h.apiClient.projectFolder = (id) => { h.reads.push(id); return read.promise; };
        h.save();
        h.post.reject(Object.assign(new Error('Save refused or response lost'), { status }));
        await turn();
        if (status === 400) {
            assert.equal(h.reads.length, 1, 'a definite refusal keeps the editable choice');
            assert.equal(h.dialog().querySelector('[data-pf-save]').disabled, false);
            assert.match(h.dialog().querySelector('[data-pf-error]').textContent, /Save refused/);
        } else {
            assert.equal(h.reads.length, 2);
            assert.equal(h.dialog().querySelector('[data-pf-save]').disabled, true, 'no stale retry before GET');
            read.resolve({ folder: missing('fresh') });
            await turn();
            assert.equal(h.dialog().querySelector('[data-pf-save]').disabled, false);
            assert.match(h.dialog().querySelector('[data-pf-error]').textContent,
                status === 409 ? /changed meanwhile/ : /not confirmed.*may still finish/);
        }
        assert.equal(h.writes.length, 1);
        h.dialog().querySelector('[data-pf-cancel]').emit('click');
        await answer;
    });
}

test('a lost Save followed by a failed read cannot reuse the old expected word', async (t) => {
    const h = dialogHarness(t);
    const answer = h.open();
    await turn();
    h.apiClient.projectFolder = async () => { throw new Error('GET unavailable'); };
    h.save();
    h.post.reject(new Error('response lost'));
    await turn();
    assert.equal(h.dialog().querySelector('[data-pf-save]').disabled, true);
    assert.match(h.dialog().querySelector('[data-pf-error]').textContent, /not confirmed.*could not be read/);
    h.dialog().querySelector('[data-pf-save]').emit('click');
    assert.equal(h.writes.length, 1);
    h.dialog().querySelector('[data-pf-cancel]').emit('click');
    await answer;
});

for (const status of [409, 0]) {
    test(`closing during fresh GET after failure ${status} preserves the known or unknown POST outcome`, async (t) => {
        const h = dialogHarness(t), answer = h.open();
        await turn();
        const read = deferred();
        h.apiClient.projectFolder = () => read.promise;
        h.save();
        h.post.reject(Object.assign(new Error('failed POST'), { status }));
        await turn();
        h.dialog().querySelector('[data-pf-cancel]').emit('click');
        assert.equal(await answer, false);
        assert.equal(h.dialog(), null);
        assert.equal(Boolean(h.doc.getElementById('toast-stack')), status === 0);
        read.resolve({ folder: usable });
        await turn();
        assert.equal(h.dialog(), null, 'the late GET never revives a closed dialog');
        assert.equal(h.writes.length, 1);
    });
}
