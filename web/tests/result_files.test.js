// V12: a task's nested result tree is ONE folder record on its card (name, file count,
// size, member paths in its tooltip); a root file stays one row. A root file offers
// bytes only while its record still serves them, a nested file is never addressed by
// its bare name, a folder offers its `.zip` only where the host's `artifact_archives`
// confirms it (with the host's count/size), and a stat-only listing says `unverified`.
// Wrap up's saved work (`wrap_up_document`) leads the row past any number of files, uncounted.
import assert from 'node:assert/strict';
import test from 'node:test';
import { projectResultArtifacts, resultFilesItemHtml, syncResultFilesItem } from '../modules/result_files.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

const TASK = 'task-files';
const RECORDS = [
    { kind: 'task_artifact', name: 'report.md', size: 2150, status: 'ready', errors: [] },
    { kind: 'task_artifact', name: 'README.md', relpath: 'repo/README.md', size: 500, status: 'ready', errors: [] },
    { kind: 'task_artifact', name: 'app.py', relpath: 'repo/src/app.py', size: 1548, status: 'missing', errors: [] },
    { kind: 'task_artifact', name: 'json.py', relpath: 'venv/lib/json.py', size: 1000, measured: false },
    { kind: 'task_artifact', name: 'notes.md', size: 10, status: 'missing', errors: [] },
    { kind: 'task_artifact', name: 'listed.txt', size: 5, measured: false },
];

test('root files stay rows; each nested tree is one folder record keeping its relative paths', () => {
    const view = projectResultArtifacts(TASK, RECORDS);
    assert.deepEqual(view.files, [
        { name: 'report.md', bytes: 2150, note: '', url: `/api/tasks/${TASK}/artifacts/report.md` },
        { name: 'notes.md', bytes: 10, note: 'missing', url: '' },
        { name: 'listed.txt', bytes: 5, note: 'unverified', url: `/api/tasks/${TASK}/artifacts/listed.txt` },
    ]);
    assert.deepEqual(view.folders, [
        { name: 'repo', fileCount: 2, bytes: 2048, unavailableCount: 1, relpaths: ['repo/README.md', 'repo/src/app.py'], archive: null },
        { name: 'venv', fileCount: 1, bytes: 1000, unavailableCount: 0, relpaths: ['venv/lib/json.py'], archive: null },
    ]);
    assert.equal(view.fileCount, 6);
    assert.ok(!view.files.some((file) => file.name === 'json.py'), 'a nested file never stands in for a root name');
    assert.equal(projectResultArtifacts(TASK, []), null);
    assert.equal(projectResultArtifacts(TASK, [null, 'x', { size: 3 }]), null);
});

function render(view) {
    const { prior } = installDom();
    try { return resultFilesItemHtml({ phase: 'result', lineKey: 'line-1', resultArtifacts: view }); }
    finally { restoreDom(prior); }
}

const ARCHIVES = {
    repo: { name: 'repo.zip', files: 1, size: 500, excluded: 1, available: true },
    venv: { name: 'venv.zip', files: 0, size: 0, excluded: 1, available: false },
};

test('the Files row renders real links and a folder tooltip, never a nested file address', () => {
    const html = render(projectResultArtifacts(TASK, RECORDS, ARCHIVES));
    assert.match(html, /data-result-files/);
    assert.match(html, /<span class="chat-live-line-time">6 files<\/span>/);
    assert.match(html, /<a class="md-link" href="\/api\/tasks\/task-files\/artifacts\/report\.md" download="report\.md">report\.md<\/a> · 2\.1 KB/);
    assert.match(html, /notes\.md · 10 B · missing/);
    assert.match(html, /listed\.txt<\/a> · 5 B · unverified/);
    assert.match(html, /<span title="repo\/README\.md\nrepo\/src\/app\.py"><a class="md-link" href="\/api\/tasks\/task-files\/artifacts\/repo\.zip\?archive=repo" download="repo\.zip">repo\/<\/a> · folder · 1 file · 500 B · 1 not archived<\/span>/);
    assert.match(html, /<span title="venv\/lib\/json\.py">venv\/ · folder · 1 file · 1000 B · no archive<\/span>/);
    assert.equal((html.match(/href=/g) || []).length, 3);
    assert.doesNotMatch(html, /artifacts\/json\.py|artifacts\/README|relpath=/, 'no bare or nested member address');
});

test('the row bounds its length, escapes names and lists at most twenty tooltip paths', () => {
    const many = Array.from({ length: 12 }, (_, index) => ({ name: `f${index}.txt`, size: 1, status: 'ready' }));
    const tree = Array.from({ length: 25 }, (_, index) => ({ name: `m${index}`, relpath: `tree/m${index}`, size: 1 }));
    const html = render(projectResultArtifacts(TASK, [...many, ...tree]));
    assert.equal((html.match(/class="md-li">•/g) || []).length, 8);
    assert.match(html, /\+29 more files/, 'four root files and the 25-file folder are counted, not dropped');
    assert.match(render(projectResultArtifacts(TASK, tree)), /tree\/m19\n… 5 more"/);
    const hostile = render(projectResultArtifacts(TASK, [
        { name: '<img src=x onerror=1>.md', size: 1, status: 'missing' },
        { name: 'a', relpath: '"><b>/a', size: 1, status: 'ready' },
    ]));
    assert.doesNotMatch(hostile, /<img|<b>/);
    assert.match(hostile, /&lt;img src=x onerror=1&gt;\.md/);
});

test('a card keeps one Files row in step with a settled detail only', () => {
    const record = { groupId: TASK, items: [] };
    assert.equal(syncResultFilesItem(record, { status: 'running', artifacts: RECORDS }), false, 'not settled');
    assert.equal(syncResultFilesItem(record, { status: 'completed' }), false, 'no record list: unchanged');
    assert.equal(syncResultFilesItem(record, { status: 'completed', artifacts: RECORDS, artifact_archives: ARCHIVES }), true);
    assert.equal(record.items.length, 1);
    assert.equal(record.items[0].dedupeKey, `files|${TASK}`);
    assert.equal(syncResultFilesItem(record, { status: 'completed', artifacts: RECORDS, artifact_archives: ARCHIVES }), false);
    assert.equal(syncResultFilesItem(record, { status: 'completed', artifacts: [] }), true);
    assert.deepEqual(record.items, []);
});

const SAVED = { name: 'work-saved-at-wrap-up-task-files.md', request_id: 'ci_1' };

test("Wrap up's saved work leads the Files row, uncounted and unsized, past any number of files", () => {
    const many = Array.from({ length: 12 }, (_, index) => ({ name: `f${index}.txt`, size: 1, status: 'ready' }));
    const view = projectResultArtifacts(TASK, many, null, SAVED);
    assert.deepEqual(view.wrapUp, { name: SAVED.name, url: `/api/tasks/${TASK}/artifacts/${SAVED.name}?wrap_up=ci_1` });
    assert.equal(view.fileCount, 12, 'saved work is no counted file');
    const html = render(view);
    assert.match(html, /<span class="chat-live-line-time">12 files<\/span>/);
    assert.match(html, /<div class="chat-live-line-body"><span class="md-li">• <a class="md-link" href="\/api\/tasks\/task-files\/artifacts\/work-saved-at-wrap-up-task-files\.md\?wrap_up=ci_1" download="work-saved-at-wrap-up-task-files\.md">Work saved at Wrap up<\/a> · every saved text<\/span>/);
    assert.equal((html.match(/class="md-li">•/g) || []).length, 9, 'eight file rows plus the saved work');
    assert.match(html, /\+4 more files/);
    const alone = render(projectResultArtifacts(TASK, [], null, SAVED));
    assert.match(alone, /<span class="chat-live-line-time"><\/span>/, 'no file count is invented');
    assert.equal((alone.match(/href=/g) || []).length, 1);
    for (const refused of [{ name: '../x.md', request_id: 'ci_1' }, { name: SAVED.name }, null]) {
        assert.equal(projectResultArtifacts(TASK, [], null, refused), null, JSON.stringify(refused));
    }
});

test('a settled detail naming only saved work still gets its Files row', () => {
    const record = { groupId: TASK, items: [] };
    assert.equal(syncResultFilesItem(record, { status: 'running', wrap_up_document: SAVED }), false, 'not settled');
    assert.equal(syncResultFilesItem(record, { status: 'failed', wrap_up_document: SAVED }), true, 'no file list needed');
    assert.equal(record.items[0].resultArtifacts.wrapUp.name, SAVED.name);
    assert.equal(syncResultFilesItem(record, { status: 'failed', artifacts: [], wrap_up_document: SAVED }), false, 'the same view');
    assert.equal(syncResultFilesItem(record, { status: 'failed', artifacts: [] }), true, 'gone from the detail: gone');
    assert.deepEqual(record.items, []);
});
