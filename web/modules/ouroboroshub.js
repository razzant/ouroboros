/**
 * OuroborosHub tab: official catalog joined with the GLOBAL skill listing.
 *
 * Identity-first (plan §2b law 1): the card truth comes from joining the
 * catalog row with /api/extensions by the server-computed canonical name
 * (catalogRow.sanitized_name === skill.name) — never from the bucket-scoped
 * installed endpoint, whose bucket view made an externally-occupied name
 * render as "Not installed" with a deterministically dead Install/Retry.
 * The verdict itself is computed by the shared hub_sync helper (§7.5).
 */

import {
    clearPending,
    getPending,
    setPending,
    startLifecyclePoller,
} from './lifecycle_card.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { hubFactsPending, hubListingRowFor, hubSyncVerdict } from './hub_sync.js';
import {
    emitSkillLifecycle,
    escapeHtmlAttr as escapeHtml,
    fetchJson,
    renderHubCard,
    renderSubmissionHistory,
} from './utils.js';


// What a Hub replacement keeps and what it does not carry over, for both
// Update and Adopt: the state dir (saved data), enablement and the
// append-only review history stay, while review verdicts and grants are
// bound to the content hash, so new bytes need a fresh review and may need
// access granted again. No new copy is promised to be pinned or kept.
const HUB_REPLACEMENT_KEEPS = 'Its saved data, enablement and review history stay; '
    + 'the new files are reviewed again and may need access granted again.';


/**
 * A version as Hub copy shows it: `v<version>`, or "unknown version" when the
 * manifest or catalog row names none — never a bare "v". Display only: no
 * parsing, ordering or eligibility. My skills' Update confirmation uses it too.
 */
export function hubVersionText(version) {
    const text = String(version || '');
    return text ? `v${text}` : 'unknown version';
}


/** The same fact after a button, badge or state label: ` v<version>`, or nothing when unknown. */
function hubVersionSuffix(version) {
    return version ? ` ${hubVersionText(version)}` : '';
}


function hubHasHint(version) {
    return version ? `Hub has ${hubVersionText(version)}.` : 'Hub version unknown.';
}


function receiptWarning(facts) {
    if (facts.receipt_unreadable) return 'Local publish record is unreadable.';
    if (facts.no_receipt) {
        return 'No local publish record for this name - the hub skill may belong to someone else.';
    }
    return '';
}


/**
 * The one Update confirmation, shared by the hub card and the My skills menu.
 * `hubVersion` is the last catalog observation, not a pinned package: the
 * server installs whatever the Hub serves when the update runs.
 */
export function confirmHubUpdate(name, { localVersion = '', hubVersion = '' } = {}) {
    return openConfirmDialog({
        title: `Update ${name}`,
        body: `Replace the local files of ${name}, including any local edits, with the current OuroborosHub copy? `
            + HUB_REPLACEMENT_KEEPS,
        details: {
            summary: 'Show details',
            rows: [
                { label: 'Installed version', value: hubVersionText(localVersion) },
                { label: 'Last seen in Hub', value: hubVersion || 'not checked' },
            ],
        },
        confirmLabel: 'Update',
        danger: true,
    });
}


/** State line (tone/label/hint) for one card, from the verdict or pending job. */
function lifecycleForVerdict(verdict, pending, listingRow, listingOnly) {
    if (pending) {
        if (pending.failed === true) {
            return { tone: pending.tone || 'danger', label: pending.label || 'Failed', hint: pending.message || '' };
        }
        return { tone: pending.tone || 'warn', label: pending.label || 'Working', hint: pending.message || '' };
    }
    if (listingOnly) {
        // A receipt-bearing local skill whose name the whole catalog lacks;
        // without a catalog read that absence is unknown, not a fact.
        return verdict.badges.includes('catalog_unavailable')
            ? { tone: 'warn', label: 'Catalog unavailable', hint: '' }
            : { tone: 'muted', label: 'Not in the Hub catalog', hint: '' };
    }
    const facts = verdict.copy_facts;
    switch (verdict.action) {
        case 'install':
            return { tone: 'muted', label: 'Not installed', hint: 'Install starts security review automatically.' };
        case 'installed':
            return {
                tone: listingRow?.review_stale ? 'warn' : 'ok',
                label: `Installed${hubVersionSuffix(facts.local_version)}`,
                hint: listingRow?.review_stale ? 'Review is stale; re-review from My skills before enabling.' : '',
            };
        case 'update':
            return {
                tone: 'warn',
                label: `Installed${hubVersionSuffix(facts.local_version)}`,
                hint: hubHasHint(facts.catalog_version),
            };
        case 'adopt':
            return {
                tone: 'warn',
                label: `Local copy${hubVersionSuffix(facts.local_version)}`,
                hint: [hubHasHint(facts.catalog_version), receiptWarning(facts)].filter(Boolean).join(' '),
            };
        default: {
            if (verdict.badges.includes('conflict')) {
                return { tone: 'danger', label: 'Catalog entry conflict', hint: 'The catalog holds more than one entry with this name.' };
            }
            if (verdict.badges.includes('listing_unavailable')) {
                return { tone: 'warn', label: 'Hub facts unavailable', hint: '' };
            }
            if (facts.occupying_bucket === 'external') {
                // Only reachable while the catalog is unavailable.
                return { tone: 'warn', label: `Local copy${hubVersionSuffix(facts.local_version)}`, hint: '' };
            }
            if (facts.occupying_bucket) {
                return {
                    tone: 'warn',
                    label: `Name taken by a local skill (${facts.occupying_bucket})`,
                    hint: facts.occupying_bucket === 'clawhub'
                        ? 'Adopting a ClawHub-installed skill is not supported yet.'
                        : '',
                };
            }
            if (facts.local_version || facts.no_receipt || facts.receipt_unreadable) {
                // A listing row exists but its bucket could not be classified
                // (e.g. an empty payload_root): honest neutral copy, never the
                // fetch-failure wording.
                return { tone: 'warn', label: 'Name taken by a local skill', hint: '' };
            }
            return { tone: 'muted', label: 'Hub facts unavailable', hint: '' };
        }
    }
}


function badgesHtmlFor(verdict) {
    const facts = verdict.copy_facts;
    const out = [];
    for (const badge of verdict.badges) {
        if (badge === 'published') {
            out.push(`<span class="skills-badge skills-badge-ok">Published${escapeHtml(hubVersionSuffix(facts.local_version))}</span>`);
        } else if (badge === 'update_available') {
            out.push(`<span class="skills-badge skills-badge-warn">Update${escapeHtml(hubVersionSuffix(facts.catalog_version))}</span>`);
        } else if (badge === 'catalog_unavailable') {
            out.push('<span class="skills-badge skills-badge-warn">Catalog unavailable</span>');
        } else if (badge === 'listing_unavailable') {
            out.push('<span class="skills-badge skills-badge-warn">Hub facts unavailable</span>');
        } else if (badge === 'conflict') {
            out.push('<span class="skills-badge skills-badge-danger">Catalog entry conflict</span>');
        }
    }
    return out.join('');
}


function primaryHtmlFor(slug, verdict, pending) {
    const slugAttr = escapeHtml(slug);
    if (pending) {
        if (pending.failed === true) {
            const retryAction = escapeHtml(pending.retry_action || 'install');
            return `<button class="btn btn-default" data-oh-action="${retryAction}" data-oh-slug="${slugAttr}">${escapeHtml(pending.retry_label || 'Retry')}</button>
                <button class="btn btn-ghost" data-oh-dismiss="${slugAttr}">Dismiss</button>`;
        }
        return `<button class="btn btn-primary" disabled>${escapeHtml(pending.label || 'Working…')}</button>`;
    }
    const facts = verdict.copy_facts;
    switch (verdict.action) {
        case 'install':
            return `<button class="btn btn-primary" data-oh-action="install" data-oh-slug="${slugAttr}">Install</button>`;
        case 'update':
            return `<button class="btn btn-primary" data-oh-action="update" data-oh-slug="${slugAttr}">Update${escapeHtml(hubVersionSuffix(facts.catalog_version))}</button>`;
        case 'adopt':
            return `<button class="btn btn-primary" data-oh-action="adopt" data-oh-slug="${slugAttr}">Use Hub version${escapeHtml(hubVersionSuffix(facts.catalog_version))}</button>`;
        case 'installed':
            return `<button class="btn btn-default" disabled>Installed${escapeHtml(hubVersionSuffix(facts.local_version))}</button>`;
        default:
            return '';
    }
}


/**
 * Quiet submission history under the card: what this installation submitted
 * and the explicit local Clear. Neither gates the primary action.
 */
function historyHtmlFor(slug, verdict, rawSkill, pending) {
    const submission = verdict.copy_facts.submission;
    if (!submission) return '';
    const published = rawSkill?.published && typeof rawSkill.published === 'object' ? rawSkill.published : {};
    // Clearing forgets only the local receipt; a conflicted identity offers no mutation.
    const clear = verdict.badges.includes('conflict') ? '' : `<div class="marketplace-secondary-actions">
            <button class="btn btn-ghost" data-oh-clear-publication="${escapeHtml(rawSkill.name)}"
                    data-oh-receipt="${escapeHtml(JSON.stringify(published))}" ${pending ? 'disabled' : ''}>Clear local submission</button>
            <span class="muted">Local record only; the PR remains on GitHub.</span>
        </div>`;
    return `<details class="skills-details marketplace-card-history" data-oh-history="${escapeHtml(slug)}">
            <summary>Submission history</summary>
            <div class="skills-detail-row">${renderSubmissionHistory(submission)}</div>
            ${clear}
        </details>`;
}


/** Typed lifecycle error text: "<code>: <message>" when the payload carries a code. */
function typedErrorText(err) {
    const body = err && typeof err === 'object' ? (err.body || err.payload) : null;
    const code = body && typeof body === 'object' ? String(body.code || '') : '';
    const message = String((body && typeof body === 'object' && (body.error || body.message)) || err?.message || err || 'request failed');
    return code && !message.startsWith(code) ? `${code}: ${message}` : message;
}

function resultError(data) {
    const error = new Error(String(data?.error || 'request failed'));
    error.body = data;
    return error;
}


function controlsTemplate() {
    return `
        <div class="marketplace-controls">
            <input type="search" id="oh-query" class="marketplace-search ui-control" aria-label="Search OuroborosHub skills"
                   placeholder="Search official Ouroboros skills…" autocomplete="off">
            <button class="btn btn-primary" data-oh-search>Search</button>
        </div>
    `;
}


function template({ includeControls = true } = {}) {
    return `
        <div class="marketplace-shell">
            ${includeControls ? controlsTemplate() : ''}
            <div id="oh-status" class="muted marketplace-status"></div>
            <div id="oh-results" class="marketplace-results"></div>
        </div>
    `;
}


export function initOuroborosHub(pane, controlsHost = null) {
    pane.innerHTML = template({ includeControls: !controlsHost });
    if (controlsHost) {
        controlsHost.innerHTML = controlsTemplate();
    }
    const state = {
        query: '',
        catalog: null,
        results: [],
        listingByName: new Map(),
        listingUnavailable: false,
        catalogLoaded: false,
        catalogUnavailable: false,
    };
    const controlsRoot = controlsHost || pane;
    const queryInput = controlsRoot.querySelector('#oh-query');
    const results = pane.querySelector('#oh-results');
    const status = pane.querySelector('#oh-status');

    const show = (message, tone = '') => {
        status.dataset.tone = tone;
        status.textContent = message;
    };

    function catalogRowFor(item) {
        return {
            slug: String(item.slug || ''),
            sanitized_name: String(item.sanitized_name || item.slug || ''),
            latest_version: String(item.latest_version || ''),
            identity_conflict: item.identity_conflict === true,
        };
    }

    function verdictFor(item) {
        const catalogRow = catalogRowFor(item);
        const rawSkill = state.listingByName.get(catalogRow.sanitized_name) || null;
        const listingRow = rawSkill ? hubListingRowFor(rawSkill) : null;
        const verdict = hubSyncVerdict(
            listingRow,
            // A listing-only synthetic row has NO catalog entry: the verdict
            // must see catalogRow=null (slug absent), never a fabricated
            // catalog fact.
            item.listing_only === true ? null : catalogRow,
            { listingUnavailable: state.listingUnavailable, catalogUnavailable: state.catalogUnavailable },
        );
        return { verdict, rawSkill, listingRow, catalogRow };
    }

    function card(item) {
        const slug = String(item.slug || '');
        const pending = getPending(slug);
        const listingOnly = item.listing_only === true;
        const { verdict, rawSkill, listingRow } = verdictFor(item);
        const lifecycle = lifecycleForVerdict(verdict, pending, listingRow, listingOnly);
        const installed = ['installed', 'update'].includes(verdict.action) ? rawSkill : null;
        return renderHubCard(item, {
            pending,
            installed,
            lifecycle,
            primaryHtml: primaryHtmlFor(slug, verdict, pending),
            secondaryHtml: historyHtmlFor(slug, verdict, rawSkill, pending),
            badgesHtml: badgesHtmlFor(verdict),
            // A listing-only row is a local submission, not a catalog entry.
            official: !listingOnly,
        });
    }

    function renderCards() {
        if (destroyed || !state.catalogLoaded) return;
        // Lifecycle ticks and refreshes re-render every card; an owner-opened
        // history disclosure stays open across them.
        const open = new Set(Array.from(results.querySelectorAll('details[data-oh-history][open]'),
            (node) => node.dataset.ohHistory));
        results.innerHTML = state.results.map((item) => card(item)).join('')
            || '<div class="muted">No official skills found.</div>';
        if (!open.size) return;
        for (const node of results.querySelectorAll('details[data-oh-history]')) {
            if (open.has(node.dataset.ohHistory)) node.open = true;
        }
    }

    let refreshGeneration = 0;
    let destroyed = false;

    /** Filter the loaded catalog and listing by the query and paint; typing never reads the network. */
    function applyQuery() {
        if (destroyed || !Array.isArray(state.catalog)) return;
        const query = state.query.trim().toLowerCase();
        // The server search's fields (slug, name, description), applied to
        // catalog rows and local submissions alike.
        const matches = (...texts) => !query || texts.some((text) => String(text || '').toLowerCase().includes(query));
        const catalogNames = new Set(state.catalog.map((row) => String(row.sanitized_name || row.slug || '')));
        state.results = state.catalog.filter((row) => matches(row.slug, row.display_name, row.description));
        const officialCount = state.results.length;
        // A first-time submission is ABSENT from the catalog until its PR
        // merges: a card row synthesized from the receipt-bearing listing
        // entry keeps its submission history reachable. It is not official
        // and not counted as a catalog skill.
        for (const [name, skill] of state.listingByName) {
            if (catalogNames.has(name)) continue;
            if (!skill.published || typeof skill.published !== 'object') continue;
            if (!matches(name, skill.description)) continue;
            state.results.push({
                slug: name,
                sanitized_name: name,
                display_name: name,
                summary: String(skill.description || ''),
                latest_version: '',
                listing_only: true,
            });
        }
        const localOnly = state.results.length - officialCount;
        renderCards();
        const counts = [
            `${officialCount} official skill${officialCount === 1 ? '' : 's'}`,
            localOnly ? `${localOnly} local submission${localOnly === 1 ? '' : 's'} not in the catalog` : '',
        ].filter(Boolean).join(' · ');
        // Filtering a catalog whose last refresh failed keeps saying so: the rows are the previous results.
        if (state.catalogUnavailable) show(`Hub catalog unavailable. Showing previous results (${counts}). Refresh to retry.`, 'danger');
        else show(state.listingUnavailable
            ? 'Installed skills could not be read. Previous details are retained where available; Refresh to retry.'
            : counts, state.listingUnavailable ? 'warn' : 'muted');
    }

    async function refresh() {
        if (destroyed) return;
        // Stale-response guard: a slow earlier refresh must never overwrite
        // the results of a newer one (e.g. typed query racing initial load).
        const generation = ++refreshGeneration;
        show('Loading OuroborosHub…', 'muted');
        try {
            // The WHOLE catalog, filtered below: a name missing from a search
            // result is not missing from the catalog. Global listing beside it —
            // a listing fetch failure is an honest "Hub facts unavailable"
            // state, never "Not installed".
            let [catalog, listingData] = await Promise.all([
                fetchJson('/api/marketplace/ouroboroshub/catalog').then(data => ({ data }), error => ({ error })),
                fetchJson('/api/extensions').catch(() => null),
            ]);
            // The listing is a local read that never waits for the hub; facts it
            // could not know before this catalog read landed need one re-read.
            if (!catalog.error && hubFactsPending(listingData?.skills)) {
                listingData = await fetchJson('/api/extensions').catch(() => listingData);
            }
            if (destroyed || generation !== refreshGeneration) return;
            state.listingUnavailable = !Array.isArray(listingData?.skills);
            if (!state.listingUnavailable) {
                state.listingByName = new Map();
                for (const skill of listingData.skills) {
                    if (skill?.name && !state.listingByName.has(skill.name)) {
                        state.listingByName.set(skill.name, skill);
                    }
                }
            }
            if (catalog.error) throw catalog.error;
            if (!Array.isArray(catalog.data?.results)) throw new Error('Hub catalog response is unavailable.');
            state.catalog = catalog.data.results;
            state.catalogLoaded = true;
            state.catalogUnavailable = false;
            applyQuery();
        } catch (err) {
            if (destroyed || generation !== refreshGeneration) return;
            state.catalogUnavailable = true;
            show(`Hub catalog unavailable: ${err.message || err}.${state.catalogLoaded ? ' Showing previous results.' : ''} Refresh to retry.`, 'danger');
            renderCards();
        }
    }

    async function confirmAdopt(item, verdict, rawSkill) {
        const facts = verdict.copy_facts;
        const name = String(item.sanitized_name || item.slug || '');
        const bucket = facts.occupying_bucket || 'external';
        const lines = [
            `Replace the local copy (${bucket}, ${hubVersionText(facts.local_version)}), including any local edits, `
            + `with the current OuroborosHub copy? ${HUB_REPLACEMENT_KEEPS}`,
        ];
        const warning = receiptWarning(facts);
        if (warning) lines.push(warning);
        const payloadRoot = String(rawSkill?.payload_root || '');
        const submission = facts.submission;
        const submitted = submission ? [
            hubVersionText(submission.version),
            submission.pr_number !== null ? `PR #${submission.pr_number}` : '',
            submission.local_differs ? 'local files differ from it' : '',
        ].filter(Boolean).join(' · ') : '';
        const rows = [
            { label: 'Occupying bucket', value: bucket },
            ...(payloadRoot ? [{ label: 'Local folder', value: `data/${payloadRoot}/` }] : []),
            { label: 'Local version', value: hubVersionText(facts.local_version) },
            { label: 'Last seen in Hub', value: hubVersionText(facts.catalog_version) },
            ...(submitted ? [{ label: 'Submitted', value: submitted }] : []),
        ];
        return openConfirmDialog({
            title: `Use Hub version of ${name}`,
            body: lines.join('\n'),
            details: { summary: 'Show details', rows },
            confirmLabel: 'Use Hub version',
            danger: true,
        });
    }

    async function clearPublication(name, expected) {
        try {
            const data = await fetchJson(`/api/marketplace/ouroboroshub/publication/${encodeURIComponent(name)}/clear`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ expected_published: expected }),
            });
            if (!data.ok) throw resultError(data);
            emitSkillLifecycle('submission cleared', name, data);
            await refresh();
            show(`${name}: local submission record cleared`, 'ok');
        } catch (err) {
            show(`${name}: ${typedErrorText(err)}`, 'danger');
        }
    }

    async function runAction(slug, action) {
        const item = state.results.find((row) => String(row.slug || '') === slug);
        if (!item) return;
        const { verdict, rawSkill, listingRow } = verdictFor(item);
        const target = String(item.sanitized_name || item.slug || '');
        if (verdict.action !== action) {
            // A stale Retry (or any stale affordance) must never act against a
            // state the fresh verdict forbids — no-action states (conflict,
            // listing_unavailable, catalog absence) included.
            show(`${slug}: local state changed; refresh before retrying`, 'warn');
            return;
        }
        let body = null;
        let pendingLabel = '';
        let pendingMessage = '';
        let doneWord = '';
        if (action === 'adopt') {
            const expected = String(listingRow?.content_hash || '');
            if (!expected) {
                show(`${slug}: local skill facts are unavailable; refresh and retry`, 'danger');
                return;
            }
            const ok = await confirmAdopt(item, verdict, rawSkill);
            if (!ok) return;
            body = { slug, adopt: true, expected_content_hash: expected, auto_review: true };
            pendingLabel = 'Replacing';
            pendingMessage = 'Replacing the local copy with the Hub version…';
            doneWord = 'now uses the Hub version';
        } else if (action === 'update') {
            const facts = verdict.copy_facts;
            const ok = await confirmHubUpdate(target, {
                localVersion: facts.local_version,
                hubVersion: hubVersionText(facts.catalog_version),
            });
            if (!ok) return;
            body = null; // update rides its own endpoint (unload/reload + rollback).
            pendingLabel = 'Updating';
            pendingMessage = 'Updating official skill…';
            doneWord = 'updated';
        } else if (action === 'install') {
            body = { slug, auto_review: true };
            pendingLabel = 'Installing';
            pendingMessage = 'Installing official skill…';
            doneWord = 'installed';
        } else {
            return;
        }
        setPending(slug, { label: pendingLabel, tone: 'warn', message: pendingMessage, target });
        show(`${pendingLabel} ${slug}…`, 'muted');
        let actionError = '';
        try {
            const data = action === 'update'
                ? await fetchJson(`/api/marketplace/ouroboroshub/update/${encodeURIComponent(target)}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({}),
                })
                : await fetchJson('/api/marketplace/ouroboroshub/install', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
            if (!data.ok) throw resultError(data);
            show(
                data.review_status ? `${slug}: ${doneWord}, review ${data.review_status}` : `${slug}: ${doneWord}`,
                'ok',
            );
            emitSkillLifecycle(action === 'adopt' ? 'install' : action, data.sanitized_name || slug, data);
            clearPending(slug);
        } catch (err) {
            const message = typedErrorText(err);
            actionError = message;
            setPending(slug, {
                label: 'Failed',
                tone: 'danger',
                message,
                failed: true,
                retry_action: action,
                retry_label: 'Retry',
                target,
            });
        } finally {
            await refresh();
            if (!destroyed && actionError && !Array.from(results.querySelectorAll('[data-slug]')).some(card => card.dataset.slug === slug)) {
                show(`${slug}: ${actionError}`, 'danger');
            }
        }
    }

    queryInput.addEventListener('input', (event) => {
        state.query = event.target.value || '';
        clearTimeout(pane._ohTimer);
        // Local filter over the loaded catalog; Search and Refresh re-read it.
        pane._ohTimer = setTimeout(applyQuery, 250);
    });
    controlsRoot.querySelector('[data-oh-search]').addEventListener('click', refresh);
    const disposeLifecycle = startLifecyclePoller(() => {
        renderCards();
    });
    pane._ouroboroshubDestroy = () => {
        destroyed = true;
        clearTimeout(pane._ohTimer);
        disposeLifecycle();
    };
    results.addEventListener('click', async (event) => {
        const clearButton = event.target.closest('[data-oh-clear-publication]');
        if (clearButton) {
            clearButton.disabled = true;
            try {
                await clearPublication(clearButton.dataset.ohClearPublication, JSON.parse(clearButton.dataset.ohReceipt));
            } finally {
                clearButton.disabled = false;
            }
            return;
        }
        const dismiss = event.target.closest('[data-oh-dismiss]');
        if (dismiss) {
            clearPending(dismiss.dataset.ohDismiss);
            renderCards();
            return;
        }
        const actionBtn = event.target.closest('[data-oh-action]');
        if (!actionBtn) return;
        const slug = actionBtn.dataset.ohSlug;
        const action = actionBtn.dataset.ohAction;
        if (!slug || !action) return;
        actionBtn.disabled = true;
        try {
            await runAction(slug, action);
        } finally {
            actionBtn.disabled = false;
        }
    });
    pane._ouroboroshubRefresh = refresh;
    return refresh();
}
