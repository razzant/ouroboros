// The home test of web/modules/update_versions.js, Dashboard › Updates › Recovery's list of earlier versions: the
// latest few restore points in view with "Show all (N)", one quiet Restore each (the crimson waits for the
// confirmation the caller opens), dates in the app's format, and a failed read in owner words.
import assert from 'node:assert/strict';
import test from 'node:test';
import { setup, NodeStub } from './dashboard_dom_fixture.js';
import { RESTORE_ROWS, mountVersionList } from '../modules/update_versions.js';

const commit = (n) => ({ sha: `${String(n).padStart(2, '0')}${'a'.repeat(38)}`, short_sha: `${String(n).padStart(2, '0')}aaaaaaaa`,
    date: `2026-11-05T${String(10 + (n % 10)).padStart(2, '0')}:19:00Z`, message: `Change ${n}` });
const answer = (body, ok = true) => async () => ({ ok, status: ok ? 200 : 500, json: async () => body });

function list(t, read, restored = []) {
    setup(t);
    const [commits, more, releases, branch] = ['div', 'button', 'div', 'p'].map((tag) => new NodeStub(tag));
    more.hidden = true;
    const versions = mountVersionList({ commits, more, releases, branch, read, onRestore: (row) => restored.push(row) });
    return { versions, commits, more, releases, branch, restored };
}

test('the latest restore points stay in view and Show all opens the rest in place', async (t) => {
    const view = list(t, answer({ branch: 'ouroboros', sha: 'e686f019de', commits: Array.from({ length: 12 }, (_, n) => commit(n)),
        tags: [{ tag: 'v7.10.0-alpha.0', sha: commit(1).sha }] }));
    await view.versions.load();
    const rows = view.commits.querySelectorAll('.updates-restore-row');
    assert.equal(rows.length, 12);
    assert.deepEqual(rows.map((row) => row.hidden), rows.map((_, index) => index >= RESTORE_ROWS));
    assert.equal(view.more.hidden, false);
    assert.equal(view.more.textContent, 'Show all (12)');
    await view.more.fire('click');
    assert.ok(rows.every((row) => !row.hidden));
    assert.equal(view.more.textContent, 'Show fewer');
    await view.more.fire('click');
    assert.equal(rows.filter((row) => !row.hidden).length, RESTORE_ROWS);
    assert.equal(view.branch.textContent, 'Branch: ouroboros @ e686f019de');
    // A release names its row; a commit is named by its short form.
    assert.match(rows[1].textContent, /v7\.10\.0-alpha\.0/);
    assert.match(rows[0].textContent, /00aaaaaaaa/);
});

test('each row has one quiet Restore, its date in the app\'s format, and hands its words to the confirmation', async (t) => {
    const view = list(t, answer({ commits: [commit(3)], tags: [] }));
    await view.versions.load();
    const [row] = view.commits.querySelectorAll('.updates-restore-row');
    const button = row.querySelector('button');
    assert.equal(button.className, 'btn btn-ghost btn-sm', 'a quiet small capsule, never the danger tone or btn-xs');
    assert.doesNotMatch(row.textContent, /2026-11-05|T13:19/, 'no ISO stamp on the row');
    assert.match(row.querySelector('.updates-restore-date').textContent, /^5 Nov, \d\d:19$/);
    // Git's own commit date (a space and a bare offset) reads the same, in every engine.
    const git = list(t, answer({ commits: [{ ...commit(3), date: '2026-11-05 18:19:55 +0300' }], tags: [] }));
    await git.versions.load();
    assert.equal(git.commits.querySelector('.updates-restore-date').textContent,
        new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false })
            .format(new Date('2026-11-05T18:19:55+03:00')));
    assert.equal(view.more.hidden, true, 'a short list needs no Show all');
    await button.fire('click');
    assert.deepEqual(view.restored, [{ target: commit(3).sha, label: '03aaaaaaaa',
        when: row.querySelector('.updates-restore-date').textContent }]);
});

test('a failed read says so in owner words, the browser\'s transport text never the sentence', async (t) => {
    const view = list(t, async () => { throw new TypeError('Load failed'); });
    await view.versions.load();
    // (the stub DOM keeps the markup's escaped apostrophe; a browser reads it as one)
    assert.match(view.commits.textContent, /The earlier versions could not be read\. Couldn(?:'|&#39;)t reach Ouroboros\./);
    assert.doesNotMatch(view.commits.textContent, /Load failed/);
    assert.equal(view.branch.textContent, 'Branch: unknown');
    assert.equal(view.more.hidden, true);
    const refused = list(t, answer({}, false));
    await refused.versions.load();
    assert.match(refused.commits.textContent, /could not be read\. Git log API error 500/, 'a server answer keeps its own words');
});

test('official releases are rows without a Restore; none yet says how to load them', (t) => {
    const view = list(t, answer({}));
    view.versions.renderReleases([{ tag: 'v7.9.0', sha: 'f'.repeat(40) }]);
    assert.equal(view.releases.querySelectorAll('.updates-restore-row').length, 1);
    assert.equal(view.releases.querySelector('button'), null);
    view.versions.renderReleases([]);
    assert.match(view.releases.textContent, /Run a check to load official releases\./);
});
