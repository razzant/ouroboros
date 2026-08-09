import assert from 'node:assert/strict';
import test from 'node:test';

import {
    decodePatchPath,
    fileStatusLetter,
    parsePatch,
    splitRows,
    unifiedRows,
} from '../modules/patch_parse.js';

const MODIFY = [
    'diff --git a/ouroboros/loop.py b/ouroboros/loop.py',
    'index 1111111..2222222 100644',
    '--- a/ouroboros/loop.py',
    '+++ b/ouroboros/loop.py',
    '@@ -17,3 +17,4 @@ class Loop:',
    '     def run(self):',
    '-        window = COOLDOWN_S',
    '+        window = self._window_for(provider)',
    '+        self._emit("cooldown_set", provider, window)',
    '         return window',
    '',
].join('\n');

test('a modification yields real line numbers and +/- counts', () => {
    const { files, added, removed } = parsePatch(MODIFY);
    assert.equal(files.length, 1);
    const file = files[0];
    assert.equal(file.path, 'ouroboros/loop.py');
    assert.equal(file.oldPath, 'ouroboros/loop.py');
    assert.equal(fileStatusLetter(file), 'M');
    assert.equal(file.added, 2);
    assert.equal(file.removed, 1);
    assert.equal(added, 2);
    assert.equal(removed, 1);
    assert.equal(file.hunks.length, 1);
    const hunk = file.hunks[0];
    assert.equal(hunk.oldStart, 17);
    assert.equal(hunk.newStart, 17);
    assert.equal(hunk.heading, 'class Loop:');
    assert.equal(hunk.header, '@@ -17,3 +17,4 @@ class Loop:');
    assert.deepEqual(hunk.lines.map((l) => [l.type, l.oldNumber, l.newNumber]), [
        ['ctx', 17, 17],
        ['del', 18, null],
        ['add', null, 18],
        ['add', null, 19],
        ['ctx', 19, 20],
    ]);
});

test('a new file is A, a deleted file is D, a rename is R', () => {
    const added = parsePatch([
        'diff --git a/tests/test_new.py b/tests/test_new.py',
        'new file mode 100644',
        '--- /dev/null',
        '+++ b/tests/test_new.py',
        '@@ -0,0 +1,2 @@',
        '+import pytest',
        '+',
    ].join('\n')).files[0];
    assert.equal(fileStatusLetter(added), 'A');
    assert.equal(added.path, 'tests/test_new.py');
    assert.equal(added.oldPath, 'tests/test_new.py');
    assert.equal(added.added, 2);

    const deleted = parsePatch([
        'diff --git a/old.py b/old.py',
        'deleted file mode 100644',
        '--- a/old.py',
        '+++ /dev/null',
        '@@ -1,2 +0,0 @@',
        '-gone = 1',
        '-also_gone = 2',
    ].join('\n')).files[0];
    assert.equal(fileStatusLetter(deleted), 'D');
    assert.equal(deleted.path, 'old.py');
    assert.equal(deleted.removed, 2);

    const renamed = parsePatch([
        'diff --git a/a/from.py b/a/to.py',
        'similarity index 92%',
        'rename from a/from.py',
        'rename to a/to.py',
        '--- a/a/from.py',
        '+++ b/a/to.py',
        '@@ -1 +1 @@',
        '-x = 1',
        '+x = 2',
    ].join('\n')).files[0];
    assert.equal(fileStatusLetter(renamed), 'R');
    assert.equal(renamed.renamed, true);
    assert.equal(renamed.oldPath, 'a/from.py');
    assert.equal(renamed.path, 'a/to.py');
});

test('binary files are flagged and keep their notice instead of fake counts', () => {
    const { files } = parsePatch([
        'diff --git a/web/favicon.png b/web/favicon.png',
        'index 3333333..4444444 100644',
        'Binary files a/web/favicon.png and b/web/favicon.png differ',
    ].join('\n'));
    assert.equal(files.length, 1);
    assert.equal(files[0].binary, true);
    assert.equal(files[0].added, 0);
    assert.equal(files[0].removed, 0);
    assert.deepEqual(files[0].hunks, []);
    assert.match(files[0].notes[0], /^Binary files /);

    const gitBinary = parsePatch([
        'diff --git a/blob.bin b/blob.bin',
        'new file mode 100644',
        'GIT binary patch',
        'literal 8',
        'zcmZ?wbhEHb',
    ].join('\n')).files[0];
    assert.equal(gitBinary.binary, true);
    assert.equal(fileStatusLetter(gitBinary), 'A');
});

test('mode-only and copy entries still appear with an honest status', () => {
    const { files } = parsePatch([
        'diff --git a/script.sh b/script.sh',
        'old mode 100644',
        'new mode 100755',
    ].join('\n'));
    assert.equal(files.length, 1);
    assert.equal(files[0].path, 'script.sh');
    assert.equal(fileStatusLetter(files[0]), 'M');
    assert.deepEqual(files[0].hunks, []);
    assert.deepEqual(files[0].notes, ['old mode 100644', 'new mode 100755']);
});

test('no-newline markers annotate the preceding line, not a phantom row', () => {
    const file = parsePatch([
        'diff --git a/tail.txt b/tail.txt',
        '--- a/tail.txt',
        '+++ b/tail.txt',
        '@@ -1 +1 @@',
        '-old tail',
        '\\ No newline at end of file',
        '+new tail',
        '\\ No newline at end of file',
    ].join('\n')).files[0];
    assert.equal(file.hunks[0].lines.length, 2);
    assert.equal(file.hunks[0].lines[0].noNewline, true);
    assert.equal(file.hunks[0].lines[1].noNewline, true);
    assert.equal(file.added, 1);
    assert.equal(file.removed, 1);
});

test('quoted and octal-escaped paths are decoded', () => {
    assert.equal(decodePatchPath('a/plain.py'), 'plain.py');
    assert.equal(decodePatchPath('"a/with space.py"'), 'with space.py');
    assert.equal(decodePatchPath('"b/tab\\there.py"'), 'tab\there.py');
    assert.equal(decodePatchPath('"b/caf\\303\\251.py"'), 'café.py');
    assert.equal(decodePatchPath('/dev/null'), '');
    assert.equal(decodePatchPath(null), '');

    const file = parsePatch([
        'diff --git "a/dir/with space.py" "b/dir/with space.py"',
        '--- "a/dir/with space.py"',
        '+++ "b/dir/with space.py"',
        '@@ -1 +1 @@',
        '-a',
        '+b',
    ].join('\n')).files[0];
    assert.equal(file.path, 'dir/with space.py');
});

test('the untracked --no-index section parses as an addition (REAL git bytes)', () => {
    // Verbatim output of the command the server actually runs for an attributed
    // untracked file:
    //   git -c core.quotepath=off diff --no-ext-diff --no-textconv --no-color \
    //       --no-index -- /dev/null new_file.py
    // Note what git emits: an ordinary `diff --git a/<new> b/<new>` header, NOT a
    // `diff --no-index` line. The parser had a branch for the latter, which no
    // real patch can ever reach — this test is pinned to the real shape instead.
    const { files } = parsePatch([
        'diff --git a/new_file.py b/new_file.py',
        'new file mode 100644',
        'index 0000000..b864e36',
        '--- /dev/null',
        '+++ b/new_file.py',
        '@@ -0,0 +1,2 @@',
        '+fresh = True',
        '+more = 1',
    ].join('\n'));
    assert.equal(files.length, 1);
    assert.equal(files[0].path, 'new_file.py');
    assert.equal(fileStatusLetter(files[0]), 'A');
    assert.equal(files[0].added, 2);
    assert.equal(files[0].removed, 0);
    assert.equal(files[0].hunks[0].lines[0].newNumber, 1);
});

test('a deleted line that looks like a header stays a deletion', () => {
    const { files } = parsePatch([
        'diff --git a/doc.md b/doc.md',
        '--- a/doc.md',
        '+++ b/doc.md',
        '@@ -1,3 +1,2 @@',
        '-- a/list item',
        '--- a/deeper item',
        '-+++ b/tricky',
        '+kept',
    ].join('\n'));
    assert.equal(files.length, 1);
    assert.equal(files[0].removed, 3);
    assert.equal(files[0].added, 1);
    assert.deepEqual(files[0].hunks[0].lines.map((l) => l.text), [
        '- a/list item',
        '-- a/deeper item',
        '+++ b/tricky',
        'kept',
    ]);
});

test('multiple files keep patch order and independent counts', () => {
    const patch = [MODIFY, [
        'diff --git a/README.md b/README.md',
        '--- a/README.md',
        '+++ b/README.md',
        '@@ -1,2 +1,2 @@',
        ' # Ouroboros',
        '-old line',
        '+new line',
    ].join('\n')].join('\n');
    const { files, added, removed } = parsePatch(patch);
    assert.deepEqual(files.map((f) => f.path), ['ouroboros/loop.py', 'README.md']);
    assert.deepEqual(files.map((f) => [f.added, f.removed]), [[2, 1], [1, 1]]);
    assert.equal(added, 3);
    assert.equal(removed, 2);
});

test('plain diff -u output without git headers still splits into files', () => {
    const { files } = parsePatch([
        '--- one.txt',
        '+++ one.txt',
        '@@ -1 +1 @@',
        '-a',
        '+b',
        '--- two.txt',
        '+++ two.txt',
        '@@ -1 +1 @@',
        '-c',
        '+d',
    ].join('\n'));
    assert.deepEqual(files.map((f) => f.path), ['one.txt', 'two.txt']);
    assert.deepEqual(files.map((f) => f.added), [1, 1]);
});

test('malformed and empty input degrade to nothing rather than throwing', () => {
    assert.deepEqual(parsePatch(''), { files: [], added: 0, removed: 0 });
    assert.deepEqual(parsePatch(undefined), { files: [], added: 0, removed: 0 });
    assert.deepEqual(parsePatch('not a patch at all\njust prose\n'), { files: [], added: 0, removed: 0 });

    // Hunk body with no hunk header: the lines are preamble, not silent content.
    assert.deepEqual(parsePatch('+orphan add\n-orphan del\n'), { files: [], added: 0, removed: 0 });

    // A truncated hunk (fewer lines than the header declared) keeps what exists.
    const truncated = parsePatch([
        'diff --git a/x.py b/x.py',
        '--- a/x.py',
        '+++ b/x.py',
        '@@ -1,9 +1,9 @@',
        '-only one line survived',
    ].join('\n')).files[0];
    assert.equal(truncated.removed, 1);
    assert.equal(truncated.hunks[0].lines.length, 1);

    // A hunk header with no counts means exactly one line per side.
    const single = parsePatch([
        'diff --git a/y.py b/y.py',
        '--- a/y.py',
        '+++ b/y.py',
        '@@ -3 +3 @@',
        '-a',
        '+b',
    ].join('\n')).files[0];
    assert.equal(single.hunks[0].oldCount, 1);
    assert.equal(single.hunks[0].lines[0].oldNumber, 3);
});

test('combined-diff style @@@ headers are read without crashing', () => {
    const file = parsePatch([
        'diff --git a/merge.py b/merge.py',
        '--- a/merge.py',
        '+++ b/merge.py',
        '@@@ -1,2 +1,2 @@@ ctx',
        ' kept',
        '+added',
    ].join('\n')).files[0];
    assert.equal(file.hunks.length, 1);
    assert.equal(file.added, 1);
});

test('unifiedRows keeps the +/-/space prefix and one row per hunk header', () => {
    const rows = unifiedRows(parsePatch(MODIFY).files[0]);
    assert.equal(rows[0].kind, 'hunk');
    assert.equal(rows[0].text, '@@ -17,3 +17,4 @@ class Loop:');
    assert.deepEqual(rows.slice(1).map((r) => [r.kind, r.oldNumber, r.newNumber, r.text[0]]), [
        ['ctx', '17', '17', ' '],
        ['del', '18', '', '-'],
        ['add', '', '18', '+'],
        ['add', '', '19', '+'],
        ['ctx', '19', '20', ' '],
    ]);
});

test('splitRows pairs del/add runs and leaves empty counterpart cells', () => {
    const rows = splitRows(parsePatch(MODIFY).files[0]);
    assert.equal(rows[0].kind, 'hunk');
    const body = rows.slice(1);
    assert.deepEqual(body.map((r) => [r.left && r.left.kind, r.right && r.right.kind]), [
        ['ctx', 'ctx'],
        ['del', 'add'],
        [null, 'add'],
        ['ctx', 'ctx'],
    ]);
    assert.equal(body[1].left.number, '18');
    assert.equal(body[1].right.number, '18');
    assert.equal(body[2].left, null);
    assert.equal(body[2].right.number, '19');
});

test('splitRows pairs a longer del run against a shorter add run', () => {
    const rows = splitRows(parsePatch([
        'diff --git a/x.py b/x.py',
        '--- a/x.py',
        '+++ b/x.py',
        '@@ -1,3 +1,1 @@',
        '-one',
        '-two',
        '-three',
        '+merged',
    ].join('\n')).files[0]).slice(1);
    assert.deepEqual(rows.map((r) => [r.left && r.left.text, r.right && r.right.text]), [
        ['one', 'merged'],
        ['two', null],
        ['three', null],
    ]);
});

test('renderers over a binary or hunkless file produce no rows', () => {
    const file = parsePatch([
        'diff --git a/blob.bin b/blob.bin',
        'Binary files a/blob.bin and b/blob.bin differ',
    ].join('\n')).files[0];
    assert.deepEqual(unifiedRows(file), []);
    assert.deepEqual(splitRows(file), []);
    assert.deepEqual(unifiedRows(null), []);
    assert.deepEqual(splitRows(undefined), []);
});
