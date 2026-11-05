import assert from 'node:assert/strict';
import test from 'node:test';
import { DELETED_ROWS, deletedLists, deletedProjectsHtml } from '../modules/storage_deleted.js';
import { barSegments, initStorage, renderStorage } from '../modules/storage.js';
import { setup } from './dashboard_dom_fixture.js';

// The renderer escapes text through document.createElement, as a browser's innerHTML does.
globalThis.document = globalThis.document || {
    createElement: () => ({
        set textContent(value) {
            this.innerHTML = String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        },
    }),
};

const GB = 1e9;
const NOW = new Date('2026-11-01T18:40:00Z');
const project = (name, overrides = {}) => ({ name, deleted_at: '2026-10-14T11:20:00Z', provenance: 'attached', ...overrides });
const folder = (path, name, bytes, overrides = {}) => ({
    path, state: 'measured', bytes, unreadable: 0, counted_in: null, same_volume: true, projects: [project(name)],
    active_projects: [], parts: [], rest: { count: 0, bytes: 0 }, ...overrides,
});
const placeLabel = (id) => ({ project_folders: 'Project folders', program: 'Ouroboros program' })[id] || id;
const html = (deleted, options = {}) => deletedProjectsHtml({ deleted_projects: deleted }, { placeLabel, now: NOW, ...options });
const counted = (folders, no_folder = 0) => ({ state: 'counted', no_folder, folders });
const words = (markup) => markup.replace(/<wbr>/g, '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();

test('a count saved before the list says it was not counted, never none', () => {
    for (const legacy of [undefined, null, 'odd']) {
        const said = words(html(legacy));
        assert.match(said, /Folders of deleted projects Not counted yet/);
        assert.match(said, /Refresh sizes to list them\./);
        assert.doesNotMatch(said, /None/);
    }
});

test('no finished count draws no fold: Where it is already says so', () => {
    assert.equal(deletedProjectsHtml(null, { placeLabel }), '');
});

test('an unreadable list, nothing kept and only folderless projects are three different answers', () => {
    assert.match(words(html({ state: 'unavailable', no_folder: 0, folders: [] })), /Could not be read The project list could not be read, so some may remain\./);
    assert.match(words(html(counted([]))), /None No deleted project keeps a folder\.$/);
    assert.match(words(html(counted([], 3))), /None No deleted project keeps a folder\. 3 deleted projects never had a folder\.$/);
    assert.match(words(html(counted([], 1))), /1 deleted project never had a folder\.$/);
});

test('folders inside a place say they are already in the total; those outside say they are not; nothing adds up to a new total', () => {
    const said = words(html(counted([
        folder('/p/video', 'Example: video', 61 * GB, { counted_in: 'project_folders' }),
        folder('/x/export', 'Example: export', 12 * GB),
    ])));
    assert.match(said, /Folders of deleted projects 2 folders/);
    assert.match(said, /Kept\. 61 GB is already in Project folders above, 12 GB is outside the total\./);
    assert.match(said, /In Project folders, already in the total · 1 folder · 61 GB/);
    assert.match(said, /Outside the total · 1 folder · 12 GB/);
    assert.doesNotMatch(said, /73 GB/);
    assert.match(said, /not space you would get back/);
});

test('it is no sixth group: no key dot, no group id, no share, no action that deletes', () => {
    const markup = html(counted([folder('/x/export', 'Example: export', 12 * GB)]));
    assert.doesNotMatch(markup, /storage-key/);
    assert.doesNotMatch(markup, /data-storage-group/);
    assert.doesNotMatch(markup, /data-share/);
    assert.doesNotMatch(markup, /btn-danger/);
    assert.doesNotMatch(markup, /<button/, 'a short list has no control at all');
    const place = { id: 'program', group: 'other', roots: [], present: true, same_volume: true, bytes: 10, logical_bytes: 10, files: 1, unreadable: 0, parts: [], rest: { count: 0, bytes: 0 } };
    const withList = { places: [place], deleted_projects: counted([folder('/x/export', 'Example: export', 12 * GB)]) };
    assert.deepEqual(barSegments(withList), barSegments({ places: [place] }));
    const view = renderStorage({ count: { ...withList, finished_at: '2026-11-01T16:12:00Z', bytes: 10, unreadable: 0 }, counting: null, error: null, volume: {}, retention_days: 7 }, { now: NOW });
    assert.equal((view.bar.match(/storage-bar-seg/g) || []).length, 1);
    assert.match(view.places, /data-storage-deleted/);
    assert.match(words(view.summary), /Space used by Ouroboros 10 bytes/);
});

test('a folder inside another listed folder is said and never added twice', () => {
    const said = words(html(counted([
        folder('/p/video', 'Example: video', 61 * GB, { counted_in: 'project_folders' }),
        folder('/p/video/renders', 'Example: renders', 14 * GB, { counted_in: 'project_folders', within: '/p/video' }),
    ])));
    assert.match(said, /In Project folders, already in the total · 2 folders · 61 GB/);
    assert.match(said, /Inside the folder of Example: video, so its size is part of that one\./);
});

test('a folder of several deleted projects is one row counted once, and a project that remains is said', () => {
    const said = words(html(counted([folder('/p/monorepo', 'x', 18.4 * GB, {
        counted_in: 'project_folders',
        projects: [project('Example: mobile', { provenance: 'cloned', deleted_at: '2026-09-21T17:44:00Z' }), project('Example: payments', { deleted_at: '2026-10-11T09:12:00Z' })],
        active_projects: [{ name: 'Example: live', relation: 'same' }, { name: 'Example: part', relation: 'holds' }, { name: 'Example: outer', relation: 'inside' }],
    })])));
    assert.match(said, /Example: mobile · Example: payments 18 GB/);
    assert.match(said, /Cloned folder · shared by 2 deleted projects, counted once · last deleted 11 Oct at/);
    assert.match(said, /Still the folder of project: Example: live\./);
    assert.match(said, /Holds the folder of project: Example: part\./);
    assert.match(said, /Inside the folder of project: Example: outer\./);
});

test('a place root inside a folder is said, by the screen\'s own name of it', () => {
    assert.match(words(html(counted([folder('/w', 'Example: wide', 2 * GB, { holds: ['project_folders', 'program'] })]))),
        /Holds Project folders · Ouroboros program, counted there, not in this size\./);
});

test('partial, unreadable, not found and not measured read differently, and a real zero stays zero', () => {
    const said = words(html(counted([
        folder('/a', 'Example: partial', 7.1 * GB, { state: 'partial', unreadable: 14 }),
        folder('/b', 'Example: locked', null, { state: 'unreadable', reason: 'permission_denied' }),
        folder('/c', 'Example: odd', null, { state: 'unreadable', reason: 'read_error' }),
        folder('/d', 'Example: lost', null, { state: 'absent' }),
        folder('/e', 'Example: waiting', null, { state: 'not_measured' }),
        folder('/f', 'Example: empty', 0),
    ])));
    assert.match(said, /Example: partial 7\.1 GB or more/);
    assert.match(said, /14 items could not be read\./);
    assert.match(said, /Could not be read: permission denied\./);
    assert.match(said, /Could not be read: read error\./);
    assert.match(said, /Not found · 1 folder/);
    assert.match(said, /Moved, removed, or on a disk that is not connected\./);
    assert.match(said, /Example: waiting not measured yet/);
    assert.match(said, /Example: empty 0 bytes/);
    assert.match(said, /3 could not be read in full, 1 not measured yet, 1 not found\./);
    assert.match(said, /Outside the total · 5 folders · 7\.1 GB counted, some sizes unknown/);
});

test('every folder gone from the disk says none on disk, and names what is lost', () => {
    const said = words(html(counted([folder('/d', 'Example: lost', null, { state: 'absent' })])));
    assert.match(said, /Folders of deleted projects None on disk 1 not found\./);
    assert.doesNotMatch(said, /Kept\./);
});

test('unknown sizes come first and the list continues only by asking', () => {
    const rows = Array.from({ length: DELETED_ROWS + 3 }, (_item, index) => folder(`/s/${index}`, `Example: study ${index}`, (index + 1) * GB));
    rows.push(folder('/s/locked', 'Example: locked', null, { state: 'unreadable' }));
    const [list] = deletedLists(counted(rows));
    assert.equal(list.rows[0].path, '/s/locked');
    assert.equal(list.rows[1].bytes, (DELETED_ROWS + 3) * GB);
    const first = html(counted(rows));
    assert.equal((first.match(/data-deleted-folder=/g) || []).length, DELETED_ROWS);
    assert.match(words(first), new RegExp(`Showing ${DELETED_ROWS} of ${DELETED_ROWS + 4}\\.`));
    assert.match(words(first), /Show 4 more/);
    const all = html(counted(rows), { shown: { outside: DELETED_ROWS * 2 } });
    assert.equal((all.match(/data-deleted-folder=/g) || []).length, DELETED_ROWS + 4);
    assert.doesNotMatch(all, /data-deleted-more/);
});

test('names, paths and parts are authored text: escaped, never translated, a long path may break', () => {
    const markup = html(counted([folder('/x/<b>&/long', 'A <b>&', 3 * GB, { parts: [{ name: '<i>', bytes: GB }], rest: { count: 2, bytes: 2 * GB } })]));
    assert.doesNotMatch(markup, /<b>/);
    assert.doesNotMatch(markup, /<i>/);
    assert.match(markup, /data-i18n-authored>A &lt;b&gt;&amp;</);
    assert.match(markup, /\/<wbr>/);
    assert.match(words(markup), /Largest parts/);
    assert.match(words(markup), /and 2 more, 2\.0 GB/);
});

test('the outside walk has its own progress sentence, and the places keep theirs', () => {
    const base = { count: null, error: null, volume: {}, retention_days: 7 };
    const running = { started_at: 'x', place: 'deleted_projects', current: 'Example: nas', places_done: 12, places_total: 12, bytes: 12 * GB, files: 3, done: [] };
    assert.match(words(renderStorage({ ...base, counting: running }, { now: NOW }).status),
        /Measuring folders of deleted projects outside these places · Example: nas · 12 GB so far\. You can leave this tab\./);
    assert.match(words(renderStorage({ ...base, counting: { ...running, place: 'engine', current: 'projects', places_done: 1 } }, { now: NOW }).status),
        /Measuring Agent engine · projects · 12 GB so far · 2 of 12 places\./);
});

// The mounted controller: the real initStorage, its reads and its repaint (`patch` keeps opened disclosures and the focus by
// `data-key`), over the shared dashboard DOM stub.
const volumeNow = { free_bytes: 170 * GB, total_bytes: 4000 * GB, observed_at: '2026-11-01T18:40:00Z' };
const savedCount = (finished_at, folders) => ({
    count: { started_at: '2026-11-01T16:08:00Z', finished_at, duration_seconds: 60, io_priority: 'utility', basis: 'allocated', bytes: 10,
        logical_bytes: 10, files: 1, unreadable: 0, volume: volumeNow, places: [], deleted_projects: counted(folders) },
    counting: null, error: null, volume: volumeNow, retention_days: 7,
});
const answer = (data) => ({ ok: true, status: 200, json: async () => data });
const tick = () => new Promise((resolve) => setImmediate(resolve));
const pathOf = (row) => row.querySelector('.storage-free-path').textContent;

function mounted(t, first) {
    const fixture = setup(t);
    const state = { activePage: 'dashboard', dashboardActiveSubtab: 'storage' };
    fixture.routes.set('/api/storage', answer(first));
    const storage = initStorage({ state, mount: fixture.mount });  // destroyed at each test's end, before the stub's globals go
    const show = (tab) => {
        state.dashboardActiveSubtab = tab;
        fixture.window.dispatchEvent({ type: 'ouro:dashboard-subtab-shown', detail: { tab } });
    };
    const rows = () => fixture.mount.querySelectorAll('[data-deleted-folder]');
    return { ...fixture, storage, show, rows, row: (path) => rows().find((node) => pathOf(node) === path) };
}

test('a later count that reorders the folders keeps the opened parts and the focus on the same folder', async (t) => {
    const parts = [{ name: 'cache', bytes: 2 * GB }];
    const alpha = (bytes) => folder('/x/alpha', 'Example: alpha', bytes, { parts });
    const beta = (bytes) => folder('/x/beta', 'Example: beta', bytes, { parts });
    const first = savedCount('2026-11-01T16:12:00Z', [alpha(9 * GB), beta(5 * GB)]);
    const { mount, routes, show, rows, row, document, storage } = mounted(t, first);
    await tick();
    assert.deepEqual(rows().map(pathOf), ['/x/alpha', '/x/beta']);
    mount.querySelector('[data-key="deleted-projects"]').setAttribute('open', '');
    row('/x/beta').querySelector('.storage-parts').setAttribute('open', '');
    row('/x/beta').focus();
    // The next saved count: beta grew past alpha, so the two change places.
    routes.set(`/api/storage?count=${encodeURIComponent(first.count.finished_at)}`, answer(savedCount('2026-11-01T17:00:00Z', [alpha(9 * GB), beta(12 * GB)])));
    show('logs');
    show('storage');
    await tick();
    assert.deepEqual(rows().map(pathOf), ['/x/beta', '/x/alpha'], 'the repaint drew the new order');
    assert.equal(mount.querySelector('[data-key="deleted-projects"]').hasAttribute('open'), true);
    assert.equal(row('/x/beta').querySelector('.storage-parts').hasAttribute('open'), true, 'the parts opened on beta stay on beta');
    assert.equal(row('/x/alpha').querySelector('.storage-parts').hasAttribute('open'), false, 'and never move to alpha');
    assert.equal(document.activeElement, row('/x/beta'), 'the focus stays on beta');
    storage.destroy();
});

test('Show more draws the next rows and moves the focus to the first of them', async (t) => {
    const many = Array.from({ length: DELETED_ROWS + 2 }, (_item, index) => folder(`/s/study-${String(index).padStart(2, '0')}`,
        `Example: study ${index}`, (DELETED_ROWS + 2 - index) * GB));
    const { mount, rows, document, storage } = mounted(t, savedCount('2026-11-01T16:12:00Z', many));
    await tick();
    const places = mount.querySelector('[data-storage-places]');
    const more = places.querySelector('[data-deleted-more]');
    assert.equal(rows().length, DELETED_ROWS);
    more.focus();
    await places.fire('click', { target: more });
    assert.equal(rows().length, DELETED_ROWS + 2);
    assert.equal(document.activeElement, rows()[DELETED_ROWS], 'the first row Show more drew');
    assert.equal(pathOf(document.activeElement), `/s/study-${DELETED_ROWS}`);
    assert.equal(places.querySelector('[data-deleted-more]'), null, 'nothing left to show');
    storage.destroy();
});

test('a row is keyed by its folder, spelled so a quoted selector keeps it whole', () => {
    const markup = html(counted([folder('C:\\Users\\a"b%', 'Example: odd', GB, { parts: [{ name: 'x', bytes: GB }] })]));
    for (const key of ['deleted-row:', 'deleted-parts:', 'deleted-parts-open:']) assert.match(markup, new RegExp(`data-key="${key}C:%5cUsers%5ca%22b%25"`));
    assert.match(markup, /data-deleted-list="outside"/);
});
