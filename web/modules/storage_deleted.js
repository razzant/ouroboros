// Dashboard -> Storage -> Where it is: the working folders deleted projects left behind, one quiet fold under the five
// groups. Read only: one finished count's observations (`count.deleted_projects`), no action but looking. It is no sixth
// group: no key, no share of the bar, nothing added to the headline. A folder inside a counted place is already in that
// place's size and is listed under its name; a folder outside them is listed as outside the total. Unknown stays five
// different things: not counted yet, not measured yet, could not be read, not found, never had a folder.

import { fmt } from './i18n.js';
import { escapeHtmlAttr, escapeHtmlText } from './utils.js';
import { formatBytes, isSize, largestPartsHtml, number, plural, whenLabel } from './storage_format.js';

const text = escapeHtmlText;
const attr = escapeHtmlAttr;

/** Rows one list draws at a time; every list says how many it holds, and Show more draws the rest. */
export const DELETED_ROWS = 10;

const SOURCES = { attached: 'Attached folder', cloned: 'Cloned folder', genesis: 'Managed folder' };
const REASONS = { permission_denied: 'permission denied', not_a_folder: 'not a folder', read_error: 'read error' };
const ACTIVE_SAID = {
    same: 'Still the folder of project: {names}.',
    inside: 'Inside the folder of project: {names}.',
    holds: 'Holds the folder of project: {names}.',
};
const UNREAD = new Set(['partial', 'unreadable']);
const onDisk = (folder) => folder.state !== 'absent';
const foldersWord = (n) => plural(n, '{n} folder', '{count} folders');

/** Sizes known so far of a list: their sum, and whether a size is unknown or incomplete. A real 0 stays 0, and a folder
 *  inside another listed folder is already that folder's: it is listed and said, never added twice. */
function sumOf(rows) {
    const sized = rows.filter((row) => isSize(row.bytes) && !row.within);
    return { bytes: sized.length ? sized.reduce((sum, row) => sum + row.bytes, 0) : null, whole: rows.every((row) => row.state === 'measured' && isSize(row.bytes)) };
}

/** Unknown sizes first (either could be the largest), then the largest: a bounded list leaves out only small folders. */
const byUnknownThenSize = (a, b) => (isSize(a.bytes) ? 1 : 0) - (isSize(b.bytes) ? 1 : 0) || (b.bytes || 0) - (a.bytes || 0);

/** The lists in reading order: each counted place holding folders, then what is outside the total, then what is not there. */
export function deletedLists(deleted) {
    const folders = Array.isArray(deleted?.folders) ? deleted.folders : [];
    const places = [...new Set(folders.filter((row) => onDisk(row) && row.counted_in).map((row) => row.counted_in))];
    return [
        ...places.map((place) => ({ key: `in-${place}`, kind: 'inside', place, rows: folders.filter((row) => onDisk(row) && row.counted_in === place) })),
        { key: 'outside', kind: 'outside', rows: folders.filter((row) => onDisk(row) && !row.counted_in) },
        { key: 'absent', kind: 'absent', rows: folders.filter((row) => !onDisk(row)) },
    ].filter((list) => list.rows.length).map((list) => ({ ...list, rows: [...list.rows].sort(byUnknownThenSize) }));
}

function counted({ bytes, whole }) {
    if (bytes === null) return fmt('size unknown');
    return whole ? formatBytes(bytes) : fmt('{size} counted, some sizes unknown', { size: formatBytes(bytes) });
}

/** The closed fold's one line: what stands where. Sizes in a place are already in the total, so nothing here adds up to a new one. */
function saidLine(lists, placeLabel) {
    const parts = [];
    for (const list of lists) {
        const { bytes } = sumOf(list.rows);
        if (bytes === null || list.kind === 'absent') continue;
        if (list.kind === 'inside') parts.push(fmt('{size} is already in {place} above', { size: formatBytes(bytes), place: placeLabel(list.place) }));
        else parts.push(fmt('{size} is outside the total', { size: formatBytes(bytes) }));
    }
    const rows = lists.flatMap((list) => list.rows);
    const many = (state) => rows.filter(state).length;
    const unread = many((row) => UNREAD.has(row.state));
    const waiting = many((row) => row.state === 'not_measured');
    const absent = many((row) => !onDisk(row));
    const gaps = [
        unread ? fmt('{n} could not be read in full', { n: number(unread) }) : '',
        waiting ? fmt('{n} not measured yet', { n: number(waiting) }) : '',
        absent ? fmt('{n} not found', { n: number(absent) }) : '',
    ].filter(Boolean);
    return [rows.some(onDisk) ? fmt('Kept.') : '', parts.length ? `${parts.join(', ')}.` : '', gaps.length ? `${gaps.join(', ')}.` : ''].filter(Boolean).join(' ');
}

const sizeWords = (folder) => {
    if (folder.state === 'absent') return fmt('not found');
    if (folder.state === 'not_measured') return fmt('not measured yet');
    if (!isSize(folder.bytes)) return fmt('size unknown');
    return folder.state === 'partial' ? fmt('{size} or more', { size: formatBytes(folder.bytes) }) : formatBytes(folder.bytes);
};

/** A row's facts, in the place rows' own grammar. */
function factsLine(folder, now) {
    const projects = folder.projects || [];
    const latest = projects.map((row) => row.deleted_at).filter(Boolean).sort().at(-1);
    const when = whenLabel(latest, now);
    const source = projects.map((row) => SOURCES[row.provenance]).find(Boolean);
    return [
        source ? fmt(source) : '',
        projects.length > 1 ? fmt('shared by {n} deleted projects, counted once', { n: number(projects.length) }) : '',
        when ? fmt(projects.length > 1 ? 'last deleted {when}' : 'deleted {when}', { when }) : '',
        folder.same_volume === false ? fmt('on another disk') : '',
    ].filter(Boolean).join(' · ');
}

const projectNames = (folder) => (folder?.projects || []).map((row) => row.name).join(' · ');

/** What changes how a row reads, in the ink: what the count could not read, how its size stands against the folders and
 *  places around it, and a project that remains and still touches it. */
function noteLines(folder, { byPath, placeLabel }) {
    const active = Array.isArray(folder.active_projects) ? folder.active_projects : [];
    const places = (Array.isArray(folder.holds) ? folder.holds : []).map(placeLabel).join(' · ');
    return [
        folder.state === 'partial' ? plural(folder.unreadable, '{n} item could not be read.', '{count} items could not be read.') : '',
        folder.state === 'unreadable' ? fmt('Could not be read: {reason}.', { reason: fmt(REASONS[folder.reason] || 'reason unknown') }) : '',
        folder.within ? fmt('Inside the folder of {name}, so its size is part of that one.', { name: projectNames(byPath.get(folder.within)) || folder.within }) : '',
        places ? fmt('Holds {places}, counted there, not in this size.', { places }) : '',
        ...Object.entries(ACTIVE_SAID).map(([relation, said]) => {
            const names = active.filter((row) => row.relation === relation).map((row) => row.name);
            return names.length ? fmt(said, { names: names.join(' · ') }) : '';
        }),
    ].filter(Boolean);
}

/** A row's key is its folder's path, which one count lists once, never its place in the list: a later count that reorders the
 *  rows keeps an opened disclosure and the focus (both kept by `data-key`) on the same folder. The key goes into a quoted
 *  selector, so the characters that would end or escape it are spelled out, `%` too, so two paths never share a key. */
const folderKey = (path) => String(path).replace(/[%"\\\n\r\f]/g, (c) => `%${c.charCodeAt(0).toString(16).padStart(2, '0')}`);

/** A long path breaks at its separators first; copying it stays exact. */
const pathHtml = (path) => text(path).replace(/[\\/]/g, '$&<wbr>');

function rowHtml(folder, key, context) {
    const known = isSize(folder.bytes) && folder.state !== 'absent';
    return `<li class="storage-place is-member" data-deleted-folder="${attr(folder.state)}" data-key="deleted-row:${attr(key)}" tabindex="-1">
            <span class="storage-place-label" data-i18n-authored>${text(projectNames(folder))}</span>
            <span class="storage-place-size${known ? '' : ' is-unknown'}${known && folder.state === 'measured' ? '' : ' is-qualified'}" data-i18n-fmt="storage.size">${text(sizeWords(folder))}</span>
            <span class="storage-place-delta"></span>
            <span class="storage-place-why" data-i18n-fmt="storage.deleted.facts">${text(factsLine(folder, context.now))}</span>
            ${noteLines(folder, context).map((line) => `<span class="storage-place-why storage-free-risk" data-i18n-fmt="storage.deleted.note">${text(line)}</span>`).join('')}
            <span class="storage-place-why storage-free-path" data-i18n-authored>${pathHtml(folder.path)}</span>
            ${largestPartsHtml(folder.parts, folder.rest, `deleted-parts:${key}`, `deleted-parts-open:${key}`)}
        </li>`;
}

/** A list's heading: where these folders stand against the total above, how many, and what they were counted at. */
function listHead(list, placeLabel) {
    const count = list.rows.length;
    if (list.kind === 'absent') return `${fmt('Not found')} · ${foldersWord(count)}`;
    const where = list.kind === 'inside' ? fmt('In {place}, already in the total', { place: placeLabel(list.place) }) : fmt('Outside the total');
    return `${where} · ${foldersWord(count)} · ${counted(sumOf(list.rows))}`;
}

function listHtml(list, context) {
    const drawn = list.rows.slice(0, Math.max(DELETED_ROWS, context.shown?.[list.key] || 0));
    const left = list.rows.length - drawn.length;
    return `<h4 class="storage-free-sub" data-i18n-fmt="storage.deleted.list">${text(listHead(list, context.placeLabel))}</h4>
        ${list.kind === 'absent' ? '<p class="storage-meta">Moved, removed, or on a disk that is not connected.</p>' : ''}
        <ul class="storage-group-places" data-deleted-list="${attr(list.key)}">${drawn.map((folder) => rowHtml(folder, folderKey(folder.path), context)).join('')}</ul>
        ${left > 0 ? `<div class="storage-free-picks">
            <p class="storage-meta" data-i18n-fmt="storage.deleted.shown">${text(fmt('Showing {shown} of {total}.', { shown: number(drawn.length), total: number(list.rows.length) }))}</p>
            <button type="button" class="btn btn-ghost btn-sm" data-deleted-more="${attr(list.key)}" data-key="deleted-more-${attr(list.key)}" data-i18n-fmt="storage.deleted.more">${text(fmt('Show {n} more', { n: number(Math.min(left, DELETED_ROWS)) }))}</button>
        </div>` : ''}`;
}

/** The fold with nothing to open: the same place and name, its one fact on the right, its sentence under it. */
function staticHtml(value, said, unknown = false) {
    return `<div class="storage-group storage-deleted" data-storage-deleted>
        <div class="storage-deleted-static">
            <span class="storage-group-head"><span class="storage-group-label storage-deleted-label">Folders of deleted projects</span>
                <span class="storage-place-size${unknown ? ' is-unknown' : ''}" data-i18n-fmt="storage.deleted.value">${text(value)}</span><span class="storage-place-delta"></span></span>
            <span class="storage-group-said" data-i18n-fmt="storage.deleted.said">${text(said)}</span>
        </div>
    </div>`;
}

/** The section for one saved count. `placeLabel(id)` is the screen's own name of a place; `shown` holds how far each list was continued. */
export function deletedProjectsHtml(count, { placeLabel, shown = {}, now = new Date() }) {
    if (!count) return '';  // no finished count, no places: Where it is already says so
    const deleted = count.deleted_projects;
    // A count saved before this list existed knows nothing about it: never read as none.
    if (!deleted || typeof deleted !== 'object') return staticHtml(fmt('Not counted yet'), fmt('Refresh sizes to list them.'), true);
    if (deleted.state === 'unavailable') return staticHtml(fmt('Could not be read'), fmt('The project list could not be read, so some may remain.'), true);
    const never = deleted.no_folder > 0 ? plural(deleted.no_folder, '{n} deleted project never had a folder.', '{count} deleted projects never had a folder.') : '';
    const lists = deletedLists(deleted);
    if (!lists.length) return staticHtml(fmt('None'), [fmt('No deleted project keeps a folder.'), never].filter(Boolean).join(' '));
    const present = lists.filter((list) => list.kind !== 'absent').reduce((sum, list) => sum + list.rows.length, 0);
    const value = present ? foldersWord(present) : fmt('None on disk');
    const byPath = new Map(deleted.folders.map((row) => [row.path, row]));
    return `<div class="storage-group storage-deleted" data-storage-deleted>
        <details class="storage-group-fold" data-key="deleted-projects">
            <summary data-key="deleted-projects-open"><span class="storage-group-head"><span class="storage-group-label storage-deleted-label">Folders of deleted projects</span>
                <span class="storage-place-size" data-i18n-fmt="storage.deleted.value">${text(value)}</span><span class="storage-place-delta"></span></span>
                <span class="storage-group-said" data-i18n-fmt="storage.deleted.said">${text(saidLine(lists, placeLabel))}</span></summary>
            <div class="storage-group-open storage-deleted-open">
                <p class="storage-meta">Deleting a project keeps its folder; nothing on this screen deletes it. Sizes are what each folder held at the last count, not space you would get back.</p>
                ${lists.map((list) => listHtml(list, { placeLabel, shown, now, byPath })).join('')}
                ${never ? `<p class="storage-meta storage-deleted-never" data-i18n-fmt="storage.deleted.never">${text(never)}</p>` : ''}
            </div>
        </details>
    </div>`;
}
