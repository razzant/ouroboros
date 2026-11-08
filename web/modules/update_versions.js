// Dashboard › Updates › Recovery: the restore points of this checkout and the official releases, read from
// `GET /api/git/log` and from a check's `official_tags`. A row is one version: its name (a release, else the short
// commit), when it was made in the app's format, its subject on one line (the CSS ends it) and one quiet Restore; the
// latest few stay in view and "Show all (N)" opens the rest in place. What a Restore press does, its confirmation
// included, is the caller's (updates.js rollback).
import { apiFetch } from './api_client.js';
import { attemptTime, failureWords } from './update_status.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

export const RESTORE_ROWS = 5;
// A commit's date as git prints it ("2026-11-05 18:19:55 +0300") becomes the ISO instant every engine parses (WebKit
// refuses the spaced form); a release's ISO date passes unchanged.
const isoDate = (value) => String(value || '').replace(/^(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d) ([+-]\d\d)(\d\d)$/, '$1T$2$3:$4');

export function mountVersionList({ commits, more, releases, branch, onRestore,
    read = () => apiFetch('/api/git/log', { cache: 'no-store' }) }) {
    let expanded = false;
    const row = ({ label, date, message, target, restorable }) => {
        const node = document.createElement('div');
        node.className = 'updates-restore-row';
        const when = attemptTime(isoDate(date));
        node.innerHTML = `
            <span class="updates-restore-label">${escapeHtml(label)}</span>
            <span class="updates-restore-date">${escapeHtml(when)}</span>
            <span class="updates-restore-msg" title="${escapeHtml(message || '')}">${escapeHtml(message || '')}</span>
            ${restorable ? `<button type="button" class="btn btn-ghost btn-sm" data-target="${escapeHtml(target)}">Restore</button>` : ''}`;
        if (restorable) node.querySelector('button').addEventListener('click', () => onRestore({ target, label, when }));
        return node;
    };
    function showLatest() {
        const rows = commits.querySelectorAll('.updates-restore-row');
        rows.forEach((node, index) => { node.hidden = !expanded && index >= RESTORE_ROWS; });
        more.hidden = rows.length <= RESTORE_ROWS;
        more.textContent = expanded ? 'Show fewer' : `Show all (${rows.length})`;
    }
    more.addEventListener('click', () => { expanded = !expanded; showLatest(); });

    function renderReleases(tags) {
        releases.innerHTML = '';
        (tags || []).forEach((tag) => releases.appendChild(row({
            label: tag.tag || '', date: '', message: (tag.sha || '').slice(0, 12), target: '', restorable: false,
        })));
        if (!tags?.length) releases.innerHTML = '<div class="updates-empty">Run a check to load official releases.</div>';
    }

    async function load() {
        try {
            const resp = await read();
            if (!resp.ok) throw Object.assign(new Error(`Git log API error ${resp.status}`), { status: resp.status });
            const data = await resp.json();
            branch.textContent = `Branch: ${data.branch || '?'} @ ${data.sha || '?'}`;
            // One restore list (owner decision 2026-08-31): tags that point at a listed commit become labels on that
            // commit's row; tags whose target is older than the listed window keep their own row.
            const tagsBySha = new Map();
            (data.tags || []).forEach((tag) => {
                if (tag.sha) tagsBySha.set(tag.sha, [...(tagsBySha.get(tag.sha) || []), tag]);
            });
            commits.innerHTML = '';
            const seenTagShas = new Set();
            (data.commits || []).forEach((commit) => {
                const tagged = tagsBySha.get(commit.sha) || [];
                tagged.forEach((tag) => seenTagShas.add(tag.sha));
                commits.appendChild(row({
                    label: tagged.map((tag) => tag.tag).join(', ') || commit.short_sha || commit.sha?.slice(0, 8) || '?',
                    date: commit.date, message: commit.message, target: commit.sha, restorable: true,
                }));
            });
            (data.tags || []).forEach((tag) => {
                if (tag.sha && seenTagShas.has(tag.sha)) return;
                commits.appendChild(row({ label: tag.tag, date: tag.date, message: tag.message, target: tag.tag, restorable: true }));
            });
            if (!commits.children.length) commits.innerHTML = '<div class="updates-empty">No earlier version found.</div>';
        } catch (err) {
            commits.innerHTML = `<div class="updates-empty updates-empty-error">The earlier versions could not be read. ${escapeHtml(failureWords(err))}</div>`;
            branch.textContent = 'Branch: unknown';
        }
        showLatest();
    }
    return { load, renderReleases };
}
