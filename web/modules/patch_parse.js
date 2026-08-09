/**
 * Pure unified-diff parser.
 *
 * The server sends ONE thing — the raw patch bytes — and this module derives
 * everything the UI shows from exactly those bytes: the file list, each file's
 * status (M / A / D / rename / binary), the per-file added/removed counts, and
 * the hunks with real old/new line numbers. One snapshot is one truth: there is
 * deliberately no second server-side stat source that could disagree with the
 * patch the owner is looking at.
 *
 * It handles what git actually emits: quoted/escaped paths (`"a/with space.py"`,
 * octal escapes), rename and mode-only entries, `Binary files … differ` and
 * `GIT binary patch` notices, and `\ No newline at end of file` markers. Note
 * that `git diff --no-index -- /dev/null <new>` — how the server projects an
 * untracked file — still emits an ordinary `diff --git a/<new> b/<new>` header
 * with `new file mode` and `--- /dev/null`, so there is deliberately NO
 * `diff --no-index` branch here: that string never appears in a real patch, and
 * a branch for it would be untested-by-construction code pretending otherwise.
 * Anything it cannot
 * interpret is preserved rather than guessed at: an unrecognized preamble line
 * is ignored, and a file with no hunks still appears in the list with an honest
 * status, so nothing silently vanishes from the review surface.
 *
 * Pure: no DOM, no fetch, no state.
 */

const HUNK_RE = /^@@+ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@+(.*)$/;

/** Decode one path token from a `diff --git` / `---` / `+++` header. */
export function decodePatchPath(token) {
    let raw = String(token == null ? '' : token).trim();
    if (!raw) return '';
    if (raw.startsWith('"') && raw.endsWith('"') && raw.length >= 2) {
        raw = unquoteGitPath(raw.slice(1, -1));
    }
    if (raw === '/dev/null') return '';
    // Strip the a/ b/ (or i/ w/ c/ o/) prefix git adds to header paths.
    return raw.replace(/^[abciwo]\//, '');
}

/** Reverse git's C-style quoting (`\t`, `\"`, `\\`, octal `\303\251`). */
function unquoteGitPath(body) {
    let out = '';
    const bytes = [];
    const flushBytes = () => {
        if (!bytes.length) return;
        // Octal escapes are UTF-8 BYTES; decode the run as one sequence.
        try {
            out += new TextDecoder('utf-8').decode(new Uint8Array(bytes));
        } catch {
            out += bytes.map((b) => String.fromCharCode(b)).join('');
        }
        bytes.length = 0;
    };
    for (let i = 0; i < body.length; i += 1) {
        if (body[i] !== '\\') {
            flushBytes();
            out += body[i];
            continue;
        }
        const next = body[i + 1] || '';
        if (/[0-7]/.test(next)) {
            const octal = body.slice(i + 1, i + 4).match(/^[0-7]{1,3}/)[0];
            bytes.push(parseInt(octal, 8));
            i += octal.length;
            continue;
        }
        flushBytes();
        const simple = { t: '\t', n: '\n', r: '\r', '"': '"', '\\': '\\' };
        out += Object.prototype.hasOwnProperty.call(simple, next) ? simple[next] : next;
        i += 1;
    }
    flushBytes();
    return out;
}

/**
 * Split a `diff --git <a> <b>` line into its two path tokens.
 * Unquoted paths containing spaces are genuinely ambiguous; git quotes those,
 * so the halving heuristic below is only a last resort for hand-made patches.
 */
function splitDiffHeaderPaths(rest) {
    const quoted = rest.match(/^("(?:\\.|[^"\\])*"|\S+)\s+("(?:\\.|[^"\\])*"|\S+)$/);
    if (quoted) return [quoted[1], quoted[2]];
    const parts = rest.split(' ').filter(Boolean);
    if (parts.length === 2) return parts;
    if (parts.length > 2 && parts.length % 2 === 0) {
        const half = parts.length / 2;
        return [parts.slice(0, half).join(' '), parts.slice(half).join(' ')];
    }
    return [parts[0] || '', parts[parts.length - 1] || ''];
}

function newFile(overrides = {}) {
    return {
        path: '',
        oldPath: '',
        status: 'M',
        binary: false,
        renamed: false,
        added: 0,
        removed: 0,
        hunks: [],
        notes: [],
        ...overrides,
    };
}

/** Final path/status resolution once a file's own lines have all been seen. */
function finalizeFile(file) {
    if (!file.path) file.path = file.oldPath;
    if (!file.oldPath) file.oldPath = file.path;
    if (file.renamed) file.status = 'R';
    delete file.sawNewHeader;
    return file;
}

/**
 * A hunk is finished when it has consumed the line counts its own header
 * declared. This — not "the line looks like a header" — is what lets a DELETED
 * line whose content starts with `--- ` stay a deletion instead of being
 * mistaken for the next file's header.
 */
function hunkIsComplete(hunk) {
    return hunk.oldLine - hunk.oldStart >= hunk.oldCount
        && hunk.newLine - hunk.newStart >= hunk.newCount;
}

const BODY_MARKERS = new Set(['+', '-', ' ', '\\']);

/**
 * Parse a unified diff into an ordered file list.
 *
 * @param {string} patchText raw patch bytes as text
 * @returns {{files: Array<Object>, added: number, removed: number}}
 */
export function parsePatch(patchText) {
    const text = typeof patchText === 'string' ? patchText : '';
    const files = [];
    let file = null;
    let hunk = null;
    const openFile = (next) => {
        if (file) finalizeFile(file);
        file = next;
        hunk = null;
        files.push(file);
    };
    for (const line of text.split('\n')) {
        // 1. An OPEN hunk owns its declared line counts. Consuming the body first
        //    is what keeps `--- a/x` (a deleted `-- a/x` line) a deletion.
        if (hunk && line.startsWith('\\')) {
            // `\ No newline at end of file` annotates the PREVIOUS line and counts
            // toward neither side, so it is accepted even on a complete hunk.
            const last = hunk.lines[hunk.lines.length - 1];
            if (last) last.noNewline = true;
            continue;
        }
        if (hunk && !hunkIsComplete(hunk) && (line === '' || BODY_MARKERS.has(line[0]))) {
            const marker = line[0];
            if (marker === '+') {
                hunk.lines.push({ type: 'add', oldNumber: null, newNumber: hunk.newLine, text: line.slice(1) });
                hunk.newLine += 1;
                file.added += 1;
            } else if (marker === '-') {
                hunk.lines.push({ type: 'del', oldNumber: hunk.oldLine, newNumber: null, text: line.slice(1) });
                hunk.oldLine += 1;
                file.removed += 1;
            } else {
                // A context line whose single trailing space was stripped arrives
                // as the empty string; `''.slice(1)` is '' either way.
                hunk.lines.push({
                    type: 'ctx', oldNumber: hunk.oldLine, newNumber: hunk.newLine, text: line.slice(1),
                });
                hunk.oldLine += 1;
                hunk.newLine += 1;
            }
            continue;
        }
        // 2. Structure.
        if (line.startsWith('diff --git ')) {
            const [left, right] = splitDiffHeaderPaths(line.slice('diff --git '.length));
            openFile(newFile({ oldPath: decodePatchPath(left), path: decodePatchPath(right) }));
            continue;
        }
        if (line.startsWith('--- ')) {
            const decoded = decodePatchPath(line.slice(4));
            // A plain `diff -u` stream has no `diff --git` line, so a `---` after a
            // completed header pair starts the NEXT file rather than re-labelling
            // the current one.
            if (!file || file.sawNewHeader) openFile(newFile());
            hunk = null;
            if (decoded) file.oldPath = decoded;
            else file.status = 'A';
            continue;
        }
        if (!file) continue;
        if (line.startsWith('+++ ')) {
            const decoded = decodePatchPath(line.slice(4));
            file.sawNewHeader = true;
            hunk = null;
            if (decoded) file.path = decoded;
            else file.status = 'D';
            continue;
        }
        const match = HUNK_RE.exec(line);
        if (match) {
            hunk = {
                header: line,
                heading: (match[5] || '').trim(),
                oldStart: Number(match[1]),
                oldCount: match[2] === undefined ? 1 : Number(match[2]),
                newStart: Number(match[3]),
                newCount: match[4] === undefined ? 1 : Number(match[4]),
                lines: [],
            };
            hunk.oldLine = hunk.oldStart;
            hunk.newLine = hunk.newStart;
            file.hunks.push(hunk);
            continue;
        }
        hunk = null;
        if (line.startsWith('new file mode')) { file.status = 'A'; continue; }
        if (line.startsWith('deleted file mode')) { file.status = 'D'; continue; }
        if (line.startsWith('rename from ')) {
            file.renamed = true;
            file.oldPath = decodePatchPath(line.slice('rename from '.length));
            continue;
        }
        if (line.startsWith('rename to ')) {
            file.renamed = true;
            file.path = decodePatchPath(line.slice('rename to '.length));
            continue;
        }
        if (line.startsWith('copy from ') || line.startsWith('copy to ')) {
            file.notes.push(line);
            continue;
        }
        if (line.startsWith('Binary files ') || line.startsWith('GIT binary patch')) {
            file.binary = true;
            file.notes.push(line.trim());
            continue;
        }
        if (line.startsWith('old mode ') || line.startsWith('new mode ')) {
            file.notes.push(line.trim());
            continue;
        }
        // index / similarity / dissimilarity lines carry no owner-facing signal.
    }
    if (file) finalizeFile(file);
    return {
        files,
        added: files.reduce((sum, entry) => sum + entry.added, 0),
        removed: files.reduce((sum, entry) => sum + entry.removed, 0),
    };
}

/**
 * Rows for the unified renderer: one hunk-header row then one row per line.
 * Grid columns are old№ / new№ / text (the text keeps its +/-/space prefix).
 */
export function unifiedRows(file) {
    const rows = [];
    for (const hunk of file?.hunks || []) {
        rows.push({ kind: 'hunk', text: hunk.header, oldNumber: '', newNumber: '' });
        for (const line of hunk.lines) {
            const prefix = line.type === 'add' ? '+' : line.type === 'del' ? '-' : ' ';
            rows.push({
                kind: line.type,
                oldNumber: line.oldNumber == null ? '' : String(line.oldNumber),
                newNumber: line.newNumber == null ? '' : String(line.newNumber),
                text: prefix + line.text,
                noNewline: Boolean(line.noNewline),
            });
        }
    }
    return rows;
}

/**
 * Rows for the split renderer, pairing each del run with the following add run
 * (the prototype's pairing algorithm): index k of a run shows del[k] on the left
 * and add[k] on the right, and the shorter side renders an empty counterpart
 * cell. Context lines occupy both sides.
 */
export function splitRows(file) {
    const rows = [];
    for (const hunk of file?.hunks || []) {
        rows.push({ kind: 'hunk', text: hunk.header, left: null, right: null });
        const lines = hunk.lines;
        let i = 0;
        while (i < lines.length) {
            if (lines[i].type === 'ctx') {
                const line = lines[i];
                rows.push({
                    kind: 'ctx',
                    left: { kind: 'ctx', number: String(line.oldNumber), text: line.text, noNewline: Boolean(line.noNewline) },
                    right: { kind: 'ctx', number: String(line.newNumber), text: line.text, noNewline: Boolean(line.noNewline) },
                });
                i += 1;
                continue;
            }
            const dels = [];
            const adds = [];
            while (i < lines.length && lines[i].type === 'del') { dels.push(lines[i]); i += 1; }
            while (i < lines.length && lines[i].type === 'add') { adds.push(lines[i]); i += 1; }
            if (!dels.length && !adds.length) { i += 1; continue; }
            for (let k = 0; k < Math.max(dels.length, adds.length); k += 1) {
                const del = dels[k];
                const add = adds[k];
                rows.push({
                    kind: 'change',
                    left: del
                        ? { kind: 'del', number: String(del.oldNumber), text: del.text, noNewline: Boolean(del.noNewline) }
                        : null,
                    right: add
                        ? { kind: 'add', number: String(add.newNumber), text: add.text, noNewline: Boolean(add.noNewline) }
                        : null,
                });
            }
        }
    }
    return rows;
}

/** `M` / `A` / `D` / `R` letter for a parsed file row. */
export function fileStatusLetter(file) {
    return String(file?.status || 'M').slice(0, 1).toUpperCase();
}
