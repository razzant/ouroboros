// V12: a task's nested result tree is ONE folder record on its card (name, file count,
// size, member paths in its tooltip); a root file stays one row. A root file offers
// bytes only while its record still serves them, a nested file is never addressed by
// its bare name, a folder offers its `.zip` only where the host's `artifact_archives`
// confirms it (with the host's count/size), and a stat-only listing says `unverified`.
// Wrap up's saved work (`wrap_up_document`) leads the row past any number of files, uncounted.
import assert from 'node:assert/strict';
import test from 'node:test';
import { projectResultArtifacts, resultFilesItemHtml, syncQuestionMaterials, syncResultFilesItem } from '../modules/result_files.js';
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
    assert.match(html, /notes\.md · 10 bytes · missing/);
    assert.match(html, /listed\.txt<\/a> · 5 bytes · unverified/);
    assert.match(html, /<span title="repo\/README\.md\nrepo\/src\/app\.py"><a class="md-link" href="\/api\/tasks\/task-files\/artifacts\/repo\.zip\?archive=repo" download="repo\.zip">repo\/<\/a> · folder · 1 file · 500 bytes · 1 not archived<\/span>/);
    assert.match(html, /<span title="venv\/lib\/json\.py">venv\/ · folder · 1 file · 1\.0 KB · no archive<\/span>/);
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

test('the Files row and material reader distinguish POSIX name bytes from literal escapes', () => {
    const rawName = 'fixture-\uDCFF.txt';
    const literalName = String.raw`fixture-\xff.txt`;
    const normalName = 'файл🪶.txt';
    const view = projectResultArtifacts(TASK, [
        ...[rawName, literalName, normalName].map((name) => ({ name, status: 'ready' })),
        { name: 'a', relpath: 'folder\uDC80/a', status: 'ready' },
        ...[rawName, literalName].map((name) => ({ name, relpath: `regular/${name}`, status: 'ready' })),
    ], { ['folder\uDC80']: { available: true, files: 1 } });
    assert.deepEqual(view.files.map(({ name }) => name), [rawName, literalName, normalName]);
    // Whitespace is part of a POSIX name, not a hint to select a different file.
    const spaced = ` ${rawName} `;
    assert.equal(projectResultArtifacts(TASK, [{ name: spaced }]).files[0].name, spaced);
    assert.deepEqual(view.files.map(({ url }) => url), ['', '', `/api/tasks/${TASK}/artifacts/${encodeURIComponent(normalName)}`]);
    const html = render(view);
    assert.ok(html.includes(String.raw`fixture-\xff.txt · download unavailable`));
    assert.ok(html.includes(String.raw`fixture-\\xff.txt · download unavailable`));
    assert.ok(html.includes(String.raw`folder\x80/ · folder · 1 file · no archive`));
    assert.ok(html.includes(`title="${String.raw`regular/fixture-\xff.txt`}\n${String.raw`regular/fixture-\\xff.txt`}"`));
    assert.equal((html.match(/href=/g) || []).length, 1);
    assert.match(html, /файл🪶\.txt<\/a>/);
    assert.doesNotMatch(html, /[\uD800-\uDFFF]/u);

    const { prior } = installDom();
    try {
        const card = document.createElement('div');
        // The host retains a material under a safe name and carries its display text in `label`: a raw name byte
        // arrives already spelled as a literal backslash and `udcff` (`_question_materials`) and shows as stored, not
        // quoted again; a label still holding the raw byte (an older row) keeps the file rows' spelling.
        // A name the address authority refuses is not a validated reference, so no row stands for it.
        const hostLabel = String.raw`fixture-\udcff.txt`;
        syncQuestionMaterials(card, [
            ...[hostLabel, rawName, 'R&D — файл🪶.txt'].map((label, index) => ({ task_id: TASK, name: `material-${index}.txt`, label, sha256: 'fixture' })),
            ...[rawName, literalName, normalName].map((name) => ({ task_id: TASK, name, sha256: 'fixture' })),
        ]);
        const materials = card.querySelector('.chat-quiz-materials');
        assert.deepEqual(materials.children.map((child) => [child.tagName, child.textContent, child.href]), [
            ['A', hostLabel, `/api/tasks/${TASK}/artifacts/material-0.txt`],
            ['A', String.raw`fixture-\xff.txt`, `/api/tasks/${TASK}/artifacts/material-1.txt`],
            ['A', 'R&D — файл🪶.txt', `/api/tasks/${TASK}/artifacts/material-2.txt`],
            ['A', normalName, view.files[2].url],
        ]);
        // The retained copy's name stays the type/address identity; the label is display text and
        // `saveAs` the safe save suggestion. The ordinary question download keeps the original filename.
        const seen = [];
        syncQuestionMaterials(document.createElement('div'), [
            { task_id: TASK, name: 'material-0.md', label: 'Design.txt', sha256: 'x' },
            { task_id: TASK, name: 'material-1.txt', label: hostLabel, sha256: 'x' },
            { task_id: TASK, name: 'material-2.txt', label: rawName, sha256: 'x' },
        ], (row, options) => { seen.push([row.filename, row.download_url, options]); return null; });
        assert.deepEqual(seen, [
            ['material-0.md', `/api/tasks/${TASK}/artifacts/material-0.md`, { displayName: 'Design.txt', saveAs: 'Design.txt' }],
            ['material-1.txt', `/api/tasks/${TASK}/artifacts/material-1.txt`, { displayName: hostLabel, saveAs: 'material-1.txt' }],
            ['material-2.txt', `/api/tasks/${TASK}/artifacts/material-2.txt`, { displayName: String.raw`fixture-\xff.txt`, saveAs: 'material-2.txt' }],
        ]);
        const plain = document.createElement('div');
        syncQuestionMaterials(plain, [{ task_id: TASK, name: 'material-0.png', label: 'Design.png', sha256: 'x' }]);
        assert.equal(plain.querySelector('a').download, 'Design.png');
    } finally { restoreDom(prior); }
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

test('a file the owner released in Storage offers no download, at the root or in a folder', () => {
    const released = { at: '2024-02-03T12:00:00Z', operation: 'op1', loss: 'no rerun', status_before: 'ready', bytes_now: 'deleting' };
    const view = projectResultArtifacts(TASK, [
        { kind: 'task_artifact', name: 'snapshot.sqlite', size: 70000, status: 'ready', errors: [], released },
        { kind: 'task_artifact', name: 'db.sqlite', relpath: 'runs/r1/db.sqlite', size: 9000, measured: false, released },
        { kind: 'task_artifact', name: 'REPORT.md', relpath: 'runs/r1/REPORT.md', size: 900, measured: false },
    ]);
    assert.deepEqual(view.files, [{ name: 'snapshot.sqlite', bytes: 70000, note: 'released', url: '' }]);
    assert.equal(view.folders[0].unavailableCount, 1, 'the released member is not offered; the report is');
});
