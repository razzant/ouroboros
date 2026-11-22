// Dashboard -> Storage -> Free: the heavy files of finished work the mind proposed to release, the owner's choice
// alone (`ouroboros.payload_release`). One row per proposal of one finished task, nothing chosen to begin with: its
// own words on what can no longer be checked again, every file that goes with its role, loss and size, and every
// ref that stays (report, method, result). The shared confirm dialog is the one acknowledgement; only the exact ids
// chosen here are sent, with `accept_loss`. Pure rendering over `view=manual` rows the controller (storage.js) read.

import { fmt } from './i18n.js';
import { escapeHtmlAttr, escapeHtmlText } from './utils.js';
import { formatBytes, isSize, number, plural, whenLabel } from './storage_format.js';

const text = escapeHtmlText;
const attr = escapeHtmlAttr;

/** A proposal row of `view=manual` (anything else belongs to the copies-and-scratch form). */
export const payloadOf = (item) => item?.manual?.payload_release === true;

/** A proposal the owner may choose: offered by hand, its files and the refs that stay all named. */
export const payloadKnown = (item) => payloadOf(item) && item.availability === 'manual_discard'
    && typeof item.manual.consequence === 'string' && item.manual.consequence.length > 0
    && Array.isArray(item.manual.files) && item.manual.files.length > 0
    && item.manual.files.every((row) => typeof row?.path === 'string' && isSize(row.bytes))
    && Array.isArray(item.manual.retained) && item.manual.retained.length > 0;

const ROLES = { report: 'Report', method: 'Method', result: 'Result' };
const STAYS = 'History, model requests and answers, reviews, memory, manifests, summaries and patches stay.';
const filesWord = (n) => plural(n, '{n} file', '{count} files');

/** What is chosen, over the whole list read: proposals, files and their size. */
export function payloadFacts(list, picked) {
    const rows = (list?.complete ? list.items : []).filter((item) => picked?.has(item.id) && payloadKnown(item));
    const files = rows.reduce((sum, row) => sum + row.manual.files.length, 0);
    return { rows, count: rows.length, files, bytes: rows.reduce((sum, row) => sum + row.manual.files.reduce((s, f) => s + f.bytes, 0), 0) };
}

function filesList(item, index) {
    const files = item.manual.files;
    return `<details class="activity-history storage-largest storage-free-paths" data-key="payload-files-${index}">
        <summary data-key="payload-files-${index}-open" data-i18n-fmt="storage.free.payload.files">${text(plural(files.length, 'Show the file that goes', 'Show the {count} files that go'))}</summary>
        <ul class="storage-largest-list">${files.map((row) => `<li class="storage-largest-item">
            <span class="storage-largest-name storage-free-path" data-i18n-authored>${text(row.path)}</span>
            <span class="storage-largest-size" data-i18n-fmt="storage.size">${text(formatBytes(row.bytes))}</span>
            <span class="storage-free-why" data-i18n-authored>${text(`${row.role} · ${row.loss}`)}</span>
        </li>`).join('')}</ul></details>`;
}

function staysList(item, index) {
    const refs = item.manual.retained;
    return `<details class="activity-history storage-largest storage-free-paths" data-key="payload-stays-${index}">
        <summary data-key="payload-stays-${index}-open" data-i18n-fmt="storage.free.payload.stays">${text(fmt('What stays'))}</summary>
        <ul class="storage-largest-list">${refs.map((ref) => `<li class="storage-largest-item">
            <span class="storage-largest-name storage-free-path" data-i18n-authored>${text(ref.path === 'task_result' ? fmt('the task’s own answer') : ref.path)}</span>
            <span class="storage-largest-size" data-i18n-fmt="storage.free.payload.ref">${text(fmt(ROLES[ref.role] || ref.role))}</span>
        </li>`).join('')}</ul>
        <p class="storage-meta" data-i18n-fmt="storage.free.payload.always">${text(fmt(STAYS))}</p></details>`;
}

/** One proposal: the task's title, its size, the author's consequence, then its files and what stays on request. */
export function payloadRow(item, index, picked, held, sizeCell) {
    const choosable = payloadKnown(item);
    const files = choosable ? item.manual.files.length : 0;
    const when = item.manual?.proposed_at ? whenLabel(item.manual.proposed_at) : '';
    return `<li class="storage-free-group" data-free-payload-item="${attr(item.id)}">
        <input type="checkbox" class="ui-checkbox" id="storage-payload-${index}" data-free-manual-pick="${index}" data-key="payload-${index}"${picked.has(item.id) && choosable ? ' checked' : ''}${held || !choosable ? ' disabled' : ''}>
        <label class="storage-free-label" for="storage-payload-${index}" data-i18n-authored>${text(item.name)}</label>
        ${sizeCell(item)}
        <p class="storage-free-why" data-i18n-fmt="storage.free.payload.item">${text([fmt('Stored files of finished work'), files ? filesWord(files) : '', when ? fmt('proposed {when}', { when }) : ''].filter(Boolean).join(' · '))}</p>
        <p class="storage-free-why storage-free-risk" data-i18n-authored>${text(choosable ? item.manual.consequence : fmt('What it would lose could not be read, so it cannot be chosen here.'))}</p>
        ${choosable ? filesList(item, index) + staysList(item, index) : ''}
    </li>`;
}

/** The by-hand screen for proposals; `parts` are the copies form's own pieces (head, list status, buttons). */
export function payloadHtml(free, ui, parts) {
    const list = ui.manual;
    const picked = ui.payloadChosen || new Set();
    const facts = payloadFacts(list, picked);
    const held = parts.held;
    const rows = list.items.map((item, index) => [item, index]).filter(([item]) => payloadOf(item));
    const sum = facts.count ? `${plural(facts.count, '{n} chosen', '{count} chosen')} · ${formatBytes(facts.bytes)}` : '';
    const body = list.complete ? `<div class="storage-free-picks">
            <p class="storage-meta" data-i18n-fmt="storage.free.picked">${text(fmt('{chosen} of {total} chosen', { chosen: number(facts.count), total: number(rows.length) }))}</p>
            ${facts.count ? parts.button('Choose none', 'data-free-manual-none', 'btn-ghost', '', parts.inert(held)) : ''}
        </div>
        <ul class="storage-free-groups storage-free-list">${rows.map(([item, index]) => payloadRow(item, index, picked, held, parts.sizeCell)).join('')}</ul>` : '';
    const go = parts.go(facts.count ? parts.button(plural(facts.files, 'Release {n} file', 'Release {count} files'), 'data-free-payload-go', 'btn-danger', 'storage.free.payload.go') : '');
    return `${parts.head(fmt('Heavy files of finished work'), sum)}
        ${parts.attempt}
        <p class="storage-tray-text">Files finished tasks kept as evidence, such as database snapshots, disks and run data, that the mind proposed releasing. Nothing here starts chosen.</p>
        <p class="ui-status storage-status" data-tone="warn">Releasing deletes exactly the listed files for good: what each could still prove is said beside it. Their reports, methods, results and every task’s history stay.</p>
        ${parts.status}${ui.outdated || ui.postUnknown ? '' : body}
        <p class="storage-meta">Gone for good. Sizes are the files’ own; space shared with other files is not freed.</p>
        <div class="storage-actions">${parts.back}${go}</div>`;
}

/** The confirm dialog for the chosen proposals: its confirmation is the owner's acceptance of exactly these losses. */
export function payloadConfirmCopy(list, picked) {
    const facts = payloadFacts(list, picked);
    return {
        title: fmt('Release these files?'),
        body: `${fmt('This deletes {files} ({size}) of {tasks} for good; they cannot be checked again.', {
            files: filesWord(facts.files), size: formatBytes(facts.bytes), tasks: plural(facts.count, '{n} finished task', '{count} finished tasks') })} ${fmt('Reports, methods, results and history stay.')}`,
        details: {
            summary: fmt('Show what goes'),
            rows: facts.rows.map((item) => ({ label: item.name, value: `${filesWord(item.manual.files.length)} · ${item.manual.consequence}` })),
        },
        confirmLabel: plural(facts.files, 'Release {n} file', 'Release {count} files'),
        danger: true,
    };
}
