// Resources are engine facts, with independent clocks. This leaf presents them
// and one explicit confirmation; the shared Accounts store owns all transport.
import { accountName, accountTargetKey, quotaConstraintFact, resourceCapabilitiesRead,
    resetRecoverable, READ_OK } from './claudexor_status_store.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { formatRelativeAge } from './ui_helpers.js';
import { escapeHtmlAttr as esc } from './utils.js';

export const targetFor = row => ({ harness: row.harness, profile_id: row.profile_id });
export const resourcePanelId = row => `resources-${encodeURIComponent(row.harness)}-${encodeURIComponent(row.profile_id)}`;
const knownNumber = value => typeof value === 'number' && Number.isFinite(value);
const dateText = value => {
    const at = Date.parse(value || '');
    if (!Number.isFinite(at)) return 'Not reported';
    const minutes = Math.ceil((at - Date.now()) / 60000);
    if (minutes <= 0) return formatRelativeAge(at, 'just now');
    if (minutes < 60) return `in ${minutes}m`;
    if (minutes < 1440) return `in ${Math.ceil(minutes / 60)}h`;
    return `on ${new Date(at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}`;
};
const fact = (label, value) => `<div class="resource-fact"><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`;
const note = value => value ? `<div class="resource-age">${esc(value)}</div>` : '';

export function resourceSnapshot(row, payload) {
    return (payload?.resources || []).find(resource => resource.target?.harness === row.harness
        && resource.target?.profile_id === row.profile_id) || null;
}

/** Shift only a provider-reported scale; never round the decimal through Number. */
export function resourceAmount(value, { unit = '', currency = null, decimal_places: places = null } = {}) {
    if (typeof value !== 'string' || !value.trim()) return 'Not reported';
    const decimal = /^(-?)(\d+)(?:\.(\d+))?$/.exec(value);
    if (decimal && Number.isSafeInteger(places) && places >= 0 && places <= 100) {
        const integer = decimal[2].padStart(places + 1, '0');
        const point = integer.length - places;
        const fraction = integer.slice(point) + (decimal[3] || '');
        const amount = `${decimal[1]}${integer.slice(0, point)}${fraction ? `.${fraction}` : ''}`;
        return `${amount}${currency || unit ? ` ${currency || unit}` : ''}`;
    }
    return `${value}${unit ? ` ${unit}` : ''}${currency && currency !== unit ? ` (${currency})` : ''}`;
}

export function resourceReadGap(action = {}) {
    if (action.earlierRequests?.some(resourceReadGap)) return true;
    if (!action.request || action.refreshed || action.receipt?.readback?.state === 'fresh') return false;
    return !action.receipt || action.receipt.state === 'running'
        || ['pending', 'unknown', 'reset', 'already_redeemed', 'already_used'].includes(action.receipt.outcome);
}

export function resourceFacetNote(facet, { known = true, lastKnown = false } = {}) {
    const parts = [];
    if (!facet || facet.value === null || facet.value === undefined) parts.push('Not reported');
    else if (!known || lastKnown || facet.freshness !== 'fresh') parts.push('Last known');
    const at = Date.parse(facet?.observed_at || '');
    if (Number.isFinite(at)) parts.push(`checked ${formatRelativeAge(at, 'just now')}`);
    if (facet?.last_error) parts.push('Last check failed');
    if (facet?.last_attempt_at && facet.last_attempt_at !== facet.observed_at) {
        parts.push(`attempted ${dateText(facet.last_attempt_at)}`);
    }
    return parts.join(' · ');
}

export function resetOutcome(action = {}) {
    if (!action.request) return null;
    const receipt = action.receipt;
    const applied = ['reset', 'already_redeemed'].includes(receipt?.outcome);
    if (applied) return {
        tone: 'ok', title: 'Reset applied',
        body: action.refreshed || receipt.readback?.state === 'fresh'
            ? 'Resource reads are shown below with their own freshness.'
            : 'Usage and remaining resets have not been updated. Refresh reads this account without repeating the reset.',
    };
    if (!receipt || ['pending', 'unknown', 'already_used'].includes(receipt.outcome)) return {
        tone: 'warn', title: receipt?.state === 'running' ? 'Reset is in progress' : 'Reset outcome is unconfirmed',
        body: receipt?.outcome === 'already_used'
            ? 'The provider reports that the reset was already used. This does not confirm that this request applied it.'
            : 'The provider may have applied this request. Check the same request without creating another reset.',
    };
    const labels = { nothing_to_reset: 'Nothing to reset', no_credit: 'No reset available',
        not_eligible: 'Reset is not eligible', cooldown: 'Reset is on cooldown', unavailable: 'Reset is unavailable' };
    return { tone: 'warn', title: labels[receipt.outcome] || 'Reset result unavailable', body: receipt.detail || '' };
}

export function resourceSummary(row, payload, { known = true, action = {} } = {}) {
    if (resourceReadGap(action)) return `${resetOutcome(action)?.title || 'Reset requested'} · usage not updated`;
    const facet = resourceSnapshot(row, payload)?.resets;
    if (!known || (resourceCapabilitiesRead(payload) !== READ_OK && !action.resourceRead)
        || facet?.freshness !== 'fresh') return '';
    return (facet.value || []).filter(offer => knownNumber(offer.available_count) && offer.available_count > 0)
        .map(offer => `${offer.label}: ${offer.available_count} available`).join(' · ');
}

function usageMarkup(row, payload, { known, lastKnown }) {
    const rows = (payload?.quota || []).filter(snap => snap.subject?.harness === row.harness
        && snap.subject?.subject_id === row.profile_id);
    const html = rows.map(snap => {
        const stale = !known || lastKnown || snap.freshness !== 'fresh';
        return (snap.constraints || []).map(window => {
            const used = window.used_ratio;
            const percent = knownNumber(used) ? `${Math.round(used * 100)}% used` : 'Not reported';
            const spent = !stale && quotaConstraintFact(window).exhausted;
            const scope = window.applies_to_models?.length ? `Applies to ${window.applies_to_models.join(', ')}` : '';
            return `<div class="resource-window" data-spent="${spent}">
                <dl>${fact(window.label || 'Included usage', `${stale ? 'Last known · ' : ''}${percent}`)}</dl>
                ${knownNumber(used) ? `<div class="resource-track" aria-hidden="true"><div class="resource-fill" style="--resource-used:${Math.max(0, Math.min(100, used * 100))}%"></div></div>` : ''}
                ${note([scope, window.resets_at ? `Renews ${dateText(window.resets_at)}` : '',
                    window.cooldown_until ? `Cooldown until ${dateText(window.cooldown_until)}` : ''].filter(Boolean).join(' · '))}
                ${note(resourceFacetNote({ ...snap, value: snap.constraints }, { known, lastKnown: stale }))}
            </div>`;
        }).join('');
    }).join('');
    const absences = (payload?.quota_absences || []).filter(item => item.subject?.harness === row.harness
        && item.subject?.subject_id === row.profile_id);
    return (html || note('Usage not reported')) + absences.map(item => note(item.detail || item.reason)).join('');
}

function balanceMarkup(resource, known) {
    const balances = resource?.balances;
    const spending = resource?.spending;
    const values = (balances?.value || []).map(balance => {
        let amount = resourceAmount(balance.amount, balance);
        if (balance.unlimited === true) amount = 'Unlimited';
        else if (balance.amount === null && balance.has_balance !== null) {
            amount = `${balance.has_balance ? 'Balance available' : 'No available balance'} · amount not reported`;
        }
        return fact(balance.label, amount);
    }).join('');
    const spent = (spending?.value || []).map(item => `<dl>${fact(item.label,
        item.enabled === true ? 'Enabled' : item.enabled === false ? 'Disabled' : 'State not reported')}
        ${fact('Spent', resourceAmount(item.used, item))}${fact('Limit', resourceAmount(item.limit, item))}</dl>
        ${note(item.resets_at ? `Renews ${dateText(item.resets_at)}` : '')}${note(readableDetail(item.reason))}`).join('');
    return `<dl>${values}</dl>${note(resourceFacetNote(balances, { known }))}
        ${spent}${note(resourceFacetNote(spending, { known }))}`;
}

export function resetEffect(offer, grant = null) {
    const scope = grant?.description || offer.description
        || (offer.kind === 'session_refill' ? 'Restores the session window only.' : 'Restores eligible included usage limits.');
    return `${scope}${offer.weekly_limit_applies ? ' Weekly limits still apply.' : ''}`;
}

// Native identifiers and scalar diagnostics stay available without becoming UI copy.
const readableDetail = value => typeof value === 'string' && !/^(true|false|[\w:.-]+)$/.test(value) ? value : '';
function provenanceMarkup(row, payload, resource, action) {
    const facets = ['balances', 'spending', 'resets', 'diagnostics'].map(name => [name, resource?.[name]]);
    for (const snap of payload?.quota || []) {
        if (snap.subject?.harness === row.harness && snap.subject?.subject_id === row.profile_id) facets.unshift(['usage', snap]);
    }
    const details = facets.filter(([, facet]) => facet).map(([name, facet]) => fact(name,
        [facet.source, facet.observed_at, facet.last_error].filter(Boolean).join(' · ') || 'Not reported')).join('');
    const diagnostics = (resource?.diagnostics?.value || []).map(item => fact(item.code, item.detail || 'Not reported')).join('');
    const clears = (resource?.resets?.value || []).flatMap(offer => (offer.grants || [])
        .map(grant => fact(grant.label, grant.clears.join(', ') || 'Scope not reported'))).join('');
    const receipt = [action.receipt?.detail, action.receipt?.readback?.detail].filter(Boolean).join(' · ');
    return `<details class="resource-provenance"><summary data-resource-provenance>Source details</summary><dl>${details}${diagnostics}${clears}
        ${receipt ? fact('Reset operation', receipt) : ''}</dl></details>`;
}

function offersMarkup(resource, { known, lastKnown, canReset, action }) {
    const facet = resource?.resets;
    const stale = !known || lastKnown || facet?.freshness !== 'fresh';
    const offers = (facet?.value || []).map(offer => {
        const count = knownNumber(offer.available_count) ? `${offer.available_count} available`
            : offer.usable_now === true ? `${stale ? 'available' : 'Available now'} · count not reported`
                : offer.usable_now === false ? 'Currently unavailable · count not reported' : 'Availability not reported';
        const availability = stale ? `Last reported: ${count} · current availability unknown` : count;
        const grants = offer.grants?.length ? offer.grants : [null];
        const controls = grants.map(grant => {
            const unavailable = !stale && (offer.available_count === 0 || offer.usable_now === false
                || offer.eligible === false || grant?.available_count === 0 || grant?.usable_now === false);
            const reason = unavailable ? readableDetail(offer.reason)
                || (offer.available_count === 0 || grant?.available_count === 0
                    ? 'No resets available. Refresh to check again.' : 'Currently unavailable. Refresh to check again.') : '';
            const detail = grant ? [grant.label,
                knownNumber(grant.available_count) ? `${grant.available_count} available`
                    : grant.usable_now === true ? `${stale ? 'available' : 'Available now'} · count not reported`
                        : grant.usable_now === false ? 'Currently unavailable · count not reported' : 'Count not reported',
                grant.starts_at ? `Starts ${dateText(grant.starts_at)}` : '',
                grant.expires_at ? `Expires ${dateText(grant.expires_at)}` : ''].filter(Boolean).join(' · ') : '';
            const label = resetLabel(offer, action);
            return `<div class="resource-offer">
                <div>${note(detail ? `${stale ? 'Last known · ' : ''}${detail}` : '')}
                    <div class="resource-offer-copy">${esc(resetEffect(offer, grant))}</div>${note(reason)}</div>
                ${canReset && offer.id ? `<button type="button" class="btn btn-default" data-resource-reset="${esc(offer.id)}"${grant ? ` data-resource-grant="${esc(grant.id)}"` : ''} aria-disabled="${Boolean(action.busy || unavailable)}"${unavailable ? ' disabled' : ''}>${esc(label)}</button>` : ''}
            </div>`;
        }).join('');
        return `<div class="resource-reset-offer"><div class="resource-offer-title">${esc(offer.label)}</div>
            ${note(availability)}${note(readableDetail(offer.reason))}${note(offer.resets_at ? `Available again ${dateText(offer.resets_at)}` : '')}
            ${controls}</div>`;
    }).join('');
    return offers + note(resourceFacetNote(facet, { known, lastKnown }));
}

const resetLabel = (offer, prior) => offer.kind === 'session_refill'
    ? prior.request ? 'Refill session again' : 'Refill session'
    : prior.request ? 'Use another reset' : 'Use reset';

function resetMessage(action, busy, prefix = '') {
    const outcome = resetOutcome(action);
    if (!outcome) return '';
    return `<div class="resource-message settings-action-row" data-tone="${outcome.tone}" role="status">
        <div class="settings-action-copy"><strong>${esc(prefix + outcome.title)}</strong><small>${esc(outcome.body)}</small>
            ${note(readableDetail(action.receipt?.detail))}${note(readableDetail(action.receipt?.readback?.detail))}
            ${prefix ? note(action.error) : ''}</div>
        ${resetRecoverable(action) ? `<button type="button" class="btn btn-default" data-recover-reset="${esc(action.key || '')}" aria-disabled="${Boolean(busy)}">${action.busy ? 'Checking…' : 'Check reset status'}</button>` : ''}</div>`;
}

export function accountResourcesMarkup(row, payload, { quotaRead = READ_OK, action = {} } = {}) {
    const resource = resourceSnapshot(row, payload);
    const known = quotaRead === READ_OK || action.resourceRead;
    const capabilitiesKnown = resourceCapabilitiesRead(payload) === READ_OK;
    const resourcesKnown = (known && capabilitiesKnown) || action.resourceRead;
    const lastKnown = resourceReadGap(action) || (action.activity === 'refresh' && Boolean(action.error));
    const capabilities = payload?.resource_capabilities || {};
    const earlier = action.earlierRequests || [];
    return `<div class="account-resource-head"><h5>Resources</h5>
        ${capabilities.refresh && row.profile_id ? `<button type="button" class="btn btn-default" data-refresh-account aria-disabled="${Boolean(action.busy)}" aria-busy="${Boolean(action.busy)}">${action.busy && action.activity === 'refresh' ? 'Refreshing…' : 'Refresh account'}</button>` : ''}</div>
        ${resetMessage(action, action.busy)}
        ${earlier.length ? `<section class="resource-section"><h6>Earlier unresolved requests</h6>
            ${earlier.map((request, index) => resetMessage(request, action.busy, `Request ${index + 1}: `)).join('')}</section>` : ''}
        ${action.error ? `<div class="resource-message" data-tone="warn" role="status">${esc(action.error)}${note('Showing last-known values. Other accounts are unchanged.')}</div>` : ''}
        ${action.refreshDone && !action.error ? note('Refresh completed. Each resource shows its own last successful observation.') : ''}
        ${row.enabled === false ? note('Disabled for automatic routing; account management remains available.') : ''}
        <div class="resource-columns"><section class="resource-section"><h6>Included usage</h6>${usageMarkup(row, payload, { known, lastKnown })}</section>
            <section class="resource-section"><h6>Balance and spending</h6>${balanceMarkup(resource, resourcesKnown)}</section></div>
        <section class="resource-reset-section"><h6>Reset options</h6>${offersMarkup(resource, { known: resourcesKnown, lastKnown, canReset: capabilities.reset, action })}</section>
        ${(resource?.diagnostics?.value || []).map(item => note(readableDetail(item.detail))).join('')}
        ${provenanceMarkup(row, payload, resource, action)}
        ${!capabilitiesKnown ? note('Account resource capabilities could not be checked.')
            : !capabilities.read ? note('This engine does not expose account resources. Existing account actions remain available.') : ''}`;
}

/** One confirmation for a NEW intent; recovery is a separate same-request control. */
export async function confirmAccountReset(row, offer, grant, {
    store, dialogImpl = openConfirmDialog,
} = {}) {
    const target = targetFor(row);
    const prior = store.resourceAction(target);
    if (prior.busy) return false;
    const previous = prior.request ? `${resetOutcome(prior).title}. This starts a separate request and may use an additional reset. ` : '';
    const answer = await dialogImpl({
        title: `${prior.request || offer.kind === 'session_refill' ? resetLabel(offer, prior) : offer.label} for ${accountName(row)}?`,
        body: `${previous}${resetEffect(offer, grant)} This uses the selected reset resource and cannot be undone.`,
        confirmLabel: resetLabel(offer, prior),
        details: { summary: 'Selected account and resource', rows: [
            { label: 'Account', value: `${row.harness} / ${row.profile_id}` },
            { label: 'Reset', value: grant?.label || offer.label },
        ] },
    });
    if (answer !== true) return false;
    await store.resetAccount({ target, offer_id: offer.id, ...(grant ? { grant_id: grant.id } : {}) });
    return true;
}

/** Disclosure state and focus survive the existing account-list repaint. */
export function createAccountResourcesController({ store, render }) {
    const expanded = new Set();
    function focusIdentity(active) {
        const row = active?.closest?.('[data-harness][data-profile]');
        const target = row && { harness: row.dataset.harness, profile_id: row.dataset.profile };
        const attributes = ['BUTTON', 'SUMMARY'].includes(active?.tagName) ? [...active.attributes]
            .filter(attr => attr.name.startsWith('data-') && attr.name !== 'data-enabled')
            .map(attr => [attr.name, attr.value]) : [];
        return { target, attributes };
    }
    function restoreFocus(host, { target, attributes }) {
        if (!target || !attributes.length) return;
        const nextRow = [...host.querySelectorAll('[data-harness][data-profile]')]
            .find(node => node.dataset.harness === target.harness && node.dataset.profile === target.profile_id);
        const next = [...(nextRow?.querySelectorAll('button, summary') || [])].find(button =>
            !button.disabled && attributes.every(([name, value]) => button.getAttribute(name) === value));
        (next || nextRow?.querySelector('[data-refresh-account], [data-resources]'))?.focus({ preventScroll: true });
    }
    function preserve(host, update) {
        const identity = focusIdentity(host.ownerDocument?.activeElement);
        const openDetails = [...host.querySelectorAll('.resource-provenance[open]')]
            .map(details => details.closest('.account-resource-panel').id);
        update();
        for (const details of host.querySelectorAll('.resource-provenance')) {
            details.open = openDetails.includes(details.closest('.account-resource-panel').id);
        }
        restoreFocus(host, identity);
    }
    function mount(host, rows, payload, quotaRead) {
        for (const element of host.querySelectorAll('[data-harness][data-profile]')) {
            const row = rows.find(item => item.harness === element.dataset.harness && item.profile_id === element.dataset.profile);
            if (!row) continue;
            const target = targetFor(row), key = accountTargetKey(target);
            const button = element.querySelector('[data-resources]');
            if (!button) continue;
            const open = expanded.has(key);
            button.setAttribute('aria-expanded', String(open));
            button.textContent = `${open ? '▾' : '▸'} Resources`;
            button.onclick = () => {
                const identity = focusIdentity(button);
                if (open) expanded.delete(key); else expanded.add(key);
                render();
                restoreFocus(host, identity);
            };
            if (!open) continue;
            const panel = host.ownerDocument.createElement('div');
            panel.className = 'account-resource-panel ui-card';
            panel.id = resourcePanelId(row);
            panel.innerHTML = accountResourcesMarkup(row, payload, { quotaRead, action: store.resourceAction(target) });
            element.append(panel);
            panel.onclick = async event => {
                const control = event.target.closest('button');
                if (!control || control.getAttribute('aria-disabled') === 'true') return;
                const identity = focusIdentity(control);
                control.focus({ preventScroll: true });
                if (control.hasAttribute('data-refresh-account')) await store.refreshResources(target);
                else if (control.hasAttribute('data-recover-reset')) {
                    const current = store.resourceAction(target);
                    const request = [current, ...(current.earlierRequests || [])]
                        .find(entry => entry.key === control.dataset.recoverReset);
                    if (request) await store.resetAccount(request.request, { recover: true, key: request.key });
                } else if (control.hasAttribute('data-resource-reset')) {
                    const offer = resourceSnapshot(row, payload)?.resets?.value?.find(item => item.id === control.dataset.resourceReset);
                    const grant = offer?.grants?.find(item => item.id === control.dataset.resourceGrant);
                    if (offer) await confirmAccountReset(row, offer, grant, { store });
                }
                // WebKit can blur a detached click target after its handler repaints;
                // a dialog's trigger can also disappear while the dialog is open.
                if (host.ownerDocument.activeElement === host.ownerDocument.body) restoreFocus(host, identity);
            };
        }
    }
    return { preserve, mount };
}
