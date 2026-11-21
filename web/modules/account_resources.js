// Resources are engine facts, with independent clocks. This leaf presents them, one explicit confirmation,
// and the Accounts list's reading place (the controller below); the shared Accounts store owns all transport.
import { accountCapabilityMarks, accountName, quotaConstraintFact, resourceCapabilitiesRead,
    resetRecoverable, READ_OK } from './claudexor_status_store.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { isRestoringFocus, restoreFocusWithoutReveal } from './scroll_fade.js';
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
const fact = (label, value, key = '') => `<div class="resource-fact"${key}><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`;
const note = (value, key = '') => value ? `<div class="resource-age"${key}>${esc(value)}</div>` : '';
const readingKey = (kind, ...ids) => ids.every(Boolean)
    ? ` data-resource-reading="${kind}:${esc(ids.length === 1 ? ids[0] : JSON.stringify(ids))}"` : '';
// Window ids belong to one quota snapshot; older engines identify it by subject route and source.
const usageScope = snap => snap.snapshot_id || JSON.stringify([snap.subject?.credential_route, snap.source]);

/** One muted line per model this account cannot serve right now; the engine probes again later. */
export function capabilityMarkup(row, payload) {
    return accountCapabilityMarks(payload, row).map(mark => {
        const reason = String(mark.provider_reason || '').trim();
        const brief = reason.length > 160 ? `${reason.slice(0, 159)}\u2026` : reason;
        const retry = Date.parse(mark.expires_at || '') > Date.now()
            ? `may be retried ${dateText(mark.expires_at)}` : 'the next request may recheck it';
        return `<div data-model-capability>${esc(`Cannot serve ${mark.requested_model}${brief ? `: ${brief}` : ''}`
            + ` \u00b7 temporary mark set ${dateText(mark.observed_at)}; ${retry}`)}</div>`;
    }).join('');
}

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
    // A reset in flight or not yet confirmed is a quiet fact (never amber, docs/DESIGN.md "Status marks"); the provider's
    // own "already used" is a warning worth reading.
    if (!receipt || ['pending', 'unknown', 'already_used'].includes(receipt.outcome)) return {
        tone: receipt?.outcome === 'already_used' ? 'warn' : 'neutral',
        title: receipt?.state === 'running' ? 'Reset is in progress' : 'Reset outcome is unconfirmed',
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
            return `<div class="resource-window" data-spent="${spent}"${readingKey('usage', usageScope(snap), window.id)}>
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
    return (html || note('Usage not reported')) + absences.map(item => note(item.detail || item.reason,
        readingKey('absence', item.subject?.credential_route, item.reason))).join('');
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
        return fact(balance.label, amount, readingKey('balance', balance.id));
    }).join('');
    const spent = (spending?.value || []).map(item => `<div${readingKey('spending', item.id)}><dl>${fact(item.label,
        item.enabled === true ? 'Enabled' : item.enabled === false ? 'Disabled' : 'State not reported')}
        ${fact('Spent', resourceAmount(item.used, item))}${fact('Limit', resourceAmount(item.limit, item))}</dl>
        ${note(item.resets_at ? `Renews ${dateText(item.resets_at)}` : '')}${note(readableDetail(item.reason))}</div>`).join('');
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
        [facet.source, facet.observed_at, facet.last_error].filter(Boolean).join(' · ') || 'Not reported',
        name === 'usage' ? readingKey('source', name, usageScope(facet)) : readingKey('source', name))).join('');
    const diagnostics = (resource?.diagnostics?.value || []).map(item => fact(item.code, item.detail || 'Not reported',
        readingKey('source-diagnostic', item.code))).join('');
    const clears = (resource?.resets?.value || []).flatMap(offer => (offer.grants || [])
        .map(grant => fact(grant.label, grant.clears.join(', ') || 'Scope not reported',
            readingKey('source-grant', offer.id, grant.id)))).join('');
    const receipt = [action.receipt?.detail, action.receipt?.readback?.detail].filter(Boolean).join(' · ');
    const plan = row.identity?.plan ? fact('Plan id', String(row.identity.plan)) : '';
    return `<details class="resource-provenance"><summary data-resource-provenance data-resource-reading="source">Source details</summary><dl>${plan}${details}${diagnostics}${clears}
        ${receipt ? fact('Reset operation', receipt, readingKey('receipt', action.key)) : ''}</dl></details>`;
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
            return `<div class="resource-offer"${readingKey('grant', offer.id, grant?.id)}>
                <div>${note(detail ? `${stale ? 'Last known · ' : ''}${detail}` : '')}
                    <div class="resource-offer-copy">${esc(resetEffect(offer, grant))}</div>${note(reason)}</div>
                ${canReset && offer.id ? `<button type="button" class="btn btn-default" data-resource-reset="${esc(offer.id)}"${grant ? ` data-resource-grant="${esc(grant.id)}"` : ''} aria-disabled="${Boolean(action.busy || unavailable)}"${unavailable ? ' disabled' : ''}>${esc(label)}</button>` : ''}
            </div>`;
        }).join('');
        return `<div class="resource-reset-offer"${readingKey('reset', offer.id)}><div class="resource-offer-title">${esc(offer.label)}</div>
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
    return `<div class="resource-message ui-action-row" data-tone="${outcome.tone}" role="status"${readingKey('request', action.key)}>
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
    return `<div class="account-resource-head" data-resource-reading="resources"><h5>Resources</h5>
        ${capabilities.refresh && row.profile_id ? `<button type="button" class="btn btn-default" data-refresh-account aria-disabled="${Boolean(action.busy)}" aria-busy="${Boolean(action.busy)}">${action.busy && action.activity === 'refresh' ? 'Refreshing…' : 'Refresh account'}</button>` : ''}</div>
        ${resetMessage(action, action.busy)}
        ${earlier.length ? `<section class="resource-section"><h6>Earlier unresolved requests</h6>
            ${earlier.map((request, index) => resetMessage(request, action.busy, `Request ${index + 1}: `)).join('')}</section>` : ''}
        ${action.error ? `<div class="resource-message" data-tone="warn" role="status">${esc(action.error)}${note('Showing last-known values. Other accounts are unchanged.')}</div>` : ''}
        ${action.refreshDone && !action.error ? note('Refresh completed. Each resource shows its own last successful observation.') : ''}
        ${row.enabled === false ? note('Disabled for automatic routing; account management remains available.') : ''}
        <div class="resource-columns"><section class="resource-section"><h6 data-resource-reading="included">Included usage</h6>${usageMarkup(row, payload, { known, lastKnown })}</section>
            <section class="resource-section"><h6 data-resource-reading="balance">Balance and spending</h6>${balanceMarkup(resource, resourcesKnown)}</section></div>
        <section class="resource-reset-section"><h6 data-resource-reading="resets">Reset options</h6>${offersMarkup(resource, { known: resourcesKnown, lastKnown, canReset: capabilities.reset, action })}</section>
        ${(resource?.diagnostics?.value || []).map(item => note(readableDetail(item.detail), readingKey('diagnostic', item.code))).join('')}
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

// What the owner reads from: a resource fact, or a family's sign-in card. A repaint finds either again by its key.
const READING = '[data-resource-reading], [data-family-login]';
// What the owner can press where it stands.
const CONTROL = 'label, button, summary';
const ROW = '[data-harness][data-profile]';

/** Disclosure, native activation and reading position survive account-list repaint. */
export function createAccountResourcesController({ store }) {
    const panelMarkup = new WeakMap();
    const room = { start: 0, end: 0 };
    let anchor, press, pointer, lastTop, painting, box, list, pending, movedUnderPress;

    const mounted = () => box?.isConnected && !box.ownerDocument.hidden && box.getClientRects().length;
    function visible(rect, view) {
        return rect.height && rect.bottom > Math.max(0, view.top) &&
            rect.top < Math.min(box.ownerDocument.defaultView.innerHeight, view.bottom);
    }
    function snapshot(node, top = node.getBoundingClientRect().top) {
        const key = node.dataset.resourceReading;
        return { node, top, key, panel: key && node.closest('.account-resource-panel') };
    }
    // Where a pointer, a press or the keyboard lands, one rule for all three. `node` is what stays still for
    // it: the fact or card being read, else the control itself, else its account's row. `control` is what can
    // be pressed there.
    function landing(target) {
        const within = selector => {
            const node = target?.closest?.(selector);
            return list.contains(node) ? node : null;
        };
        return { node: within(READING) || within(CONTROL) || within(ROW), control: within(CONTROL) };
    }
    function point(event) {
        pointer = { x: event.clientX, y: event.clientY, ...landing(event.target) };
        return pointer;
    }
    // The page moved under a resting pointer. What lies there now is where the pointer is, not where the
    // owner went: only a later move onto another control is an arrival.
    function rehit() {
        if (pointer && mounted()) Object.assign(pointer, landing(box.ownerDocument.elementFromPoint(pointer.x, pointer.y)));
    }
    function choose(node) {
        if (!list.contains(node)) { anchor = null; return; }
        const rect = node.getBoundingClientRect();
        // Choosing the result already shown keeps it shown: its growth still brings its actions into view.
        if (visible(rect, box.getBoundingClientRect())) {
            anchor = { ...snapshot(node, rect.top), shown: anchor?.node === node && anchor.shown };
        }
    }
    function onPointerMove(event) {
        // WebKit can deliver pointermove at the stationary cursor after a wheel or repaint.
        if (!mounted() || !event.isPrimary || event.buttons || event.pointerType === 'touch') return;
        if (pointer?.x === event.clientX && pointer.y === event.clientY) return;
        const rested = pointer?.control;
        const { node, control } = point(event);
        // One ranking for every control, never a rule per control. A control the pointer arrives at is about
        // to be pressed, so it stays put and outranks a command's result, pending or shown — even the control
        // a repaint kept as the reading place. A shown result outranks what the pointer merely rests on or
        // crosses: reading matter, a row, the space between rows, or a control the page itself moved under it.
        if (control && control !== rested) pending = null;
        else if (anchor?.shown) return;
        if (node !== anchor?.node) choose(node);
    }
    // A command can move the page under the pointer before a multi-click ends. Its next presses were
    // aimed at what stood there before, so they reach nothing. A new single press acts as usual.
    function onMultiPress(event) {
        if (event.detail < 2) movedUnderPress = false;
        else if (movedUnderPress) {
            event.preventDefault();
            event.stopPropagation();
        }
    }
    function onPointerDown(event) {
        if (!mounted() || !event.isPrimary || event.button !== 0) return;
        const { node, control } = point(event);
        press = node && { node, id: event.pointerId, control: control !== node && control };
        pending = null;
        choose(node);
    }
    function onPointerEnd(event) {
        if (event.pointerId !== press?.id) return;
        if (anchor?.node === press.node && (event.type === 'pointercancel' || !press.node.contains(event.target))) {
            anchor = null;
        }
        press = null;
    }
    // WebKit focuses the containing Settings level between pointerdown and click.
    // Neither that transition nor a restored control is a new reading intent.
    const onFocusIn = event => {
        if (!mounted() || painting || press || isRestoringFocus(event.target)) return;
        noticeScroll();
        pending = null;
        choose(landing(event.target).node);
    };

    function setRoom(side, pixels) {
        if (pixels === room[side]) return;
        room[side] = pixels;
        if (pixels) list.style.setProperty(`--account-${side}-room`, `${pixels}px`);
        else list.style.removeProperty(`--account-${side}-room`);
    }
    function keepEndRoom(wanted, limit = Infinity) {
        if (wanted <= 0 || limit <= 0) { setRoom('end', 0); return; }
        // scrollHeight floors at one viewport. Expose the true content end
        // when even the entire collapsed list fits, before measuring the deficit.
        if (box.scrollHeight === box.clientHeight) setRoom('end', room.end + box.clientHeight);
        const naturalEnd = box.scrollHeight - box.clientHeight - room.end;
        setRoom('end', Math.min(limit, Math.max(0, Math.ceil(wanted - naturalEnd))));
    }
    function place(wanted, prior, endLimit = Infinity) {
        // A newly promoted row can require negative scroll. Borrow only the missing
        // range. Any available positive offset pays that leading space back without
        // a second movement of what the reader sees.
        const start = Math.max(0, Math.ceil(room.start - wanted));
        wanted += start - room.start;
        setRoom('start', start);
        keepEndRoom(wanted, endLimit);
        // Record a real move through the Settings scroll seam, even if layout
        // already clamped to the same edge. Don't write on an unchanged poll.
        if (Math.abs(prior - wanted) >= 1 || Math.abs(box.scrollTop - wanted) >= 1) box.scrollTop = wanted;
        lastTop = box.scrollTop;
    }
    function noticeScroll() {
        // Compare to what the engine actually accepted, not the requested
        // fraction. The asynchronous event for our own write keeps its anchor.
        const top = box.scrollTop;
        if (top === lastTop) return false;
        lastTop = top;
        anchor = pending = null;
        return true;
    }
    function onScroll() {
        if (!mounted() || !noticeScroll()) return;
        if (room.start || room.end) place(lastTop, lastTop, room.end);
    }
    function releaseStart() {
        if (!room.start) return;
        setRoom('start', 0);
    }
    const onWheel = event => {
        if (!mounted() || !event.deltaY || event.ctrlKey) return;
        pointer = { x: event.clientX, y: event.clientY };
        anchor = pending = null;
        if (event.deltaY < 0 && box.scrollTop <= 1) releaseStart();
    };
    const onKeyDown = event => {
        if (!mounted() || event.defaultPrevented) return;
        const { node, control } = landing(event.target);
        if (control && !event.repeat && (event.key === ' ' || event.key === 'Enter')) {
            noticeScroll();
            pending = null;
            choose(node);
        }
        if (event.key === 'Home' && !event.target.closest('input, textarea, [contenteditable]')) {
            anchor = pending = null;
            releaseStart();
        }
    };
    function dispose() {
        box?.removeEventListener('scroll', onScroll);
        box?.removeEventListener('pointermove', onPointerMove);
        box?.removeEventListener('pointerdown', onPointerDown);
        box?.removeEventListener('focusin', onFocusIn);
        box?.removeEventListener('wheel', onWheel);
        box?.removeEventListener('keydown', onKeyDown);
        box?.removeEventListener('mousedown', onMultiPress, true);
        box?.removeEventListener('click', onMultiPress, true);
        box?.ownerDocument.removeEventListener('pointerup', onPointerEnd, true);
        box?.ownerDocument.removeEventListener('pointercancel', onPointerEnd, true);
        setRoom('start', 0);
        setRoom('end', 0);
        anchor = press = pointer = lastTop = box = list = pending = movedUnderPress = null;
    }
    // The page moves on purpose: the shortest way to the whole result, which the owner then reads from.
    function show(node) {
        node.scrollIntoView({ block: 'nearest' });
        const top = box.scrollTop;
        if (top !== lastTop) movedUnderPress = true;
        if (top !== lastTop && (room.start || room.end)) place(top, top, room.end);
        lastTop = box.scrollTop;
        anchor = list.contains(node) ? { ...snapshot(node), shown: true } : null;
    }
    /** The owner commands a result that arrives later. It is shown once, unless they have scrolled or chosen
     *  something else meanwhile; every later answer is an ordinary repaint around what they read. */
    function command() {
        if (mounted()) noticeScroll();
        const mine = pending = {};
        return node => {
            if (!mounted() || !node?.getClientRects().length || noticeScroll() || pending !== mine) return;
            pending = null;
            show(node);
            rehit();
        };
    }
    function focusIdentity(active) {
        const row = active?.closest?.(ROW);
        const target = row && { harness: row.dataset.harness, profile_id: row.dataset.profile };
        const attributes = ['BUTTON', 'SUMMARY'].includes(active?.tagName) ? [...active.attributes]
            .filter(attr => attr.name.startsWith('data-') && attr.name !== 'data-enabled')
            .map(attr => [attr.name, attr.value]) : [];
        return { target, attributes };
    }
    function replacementControl(host, { target, attributes }) {
        if (!target || !attributes.length) return;
        const nextRow = [...host.querySelectorAll(ROW)]
            .find(node => node.dataset.harness === target.harness && node.dataset.profile === target.profile_id);
        const next = [...(nextRow?.querySelectorAll('button, summary') || [])].find(button =>
            !button.disabled && attributes.every(([name, value]) => button.getAttribute(name) === value));
        return next || nextRow?.querySelector('[data-refresh-account], [data-resources]');
    }
    function readingAnchors(host, view) {
        if (!box.scrollTop && !room.start) return [];
        let previous, following, cutoff = view.bottom;
        for (const button of host.querySelectorAll('[data-resources]')) {
            const { top, bottom } = button.getBoundingClientRect();
            if (top >= view.bottom) break;
            if (bottom > view.top) {
                following = button;
                cutoff = top;
                break;
            }
            previous = button;
        }
        // A following More can peek in below the fact at the reading edge.
        // Prefer that fact (keyed within its panel), then the visible row head.
        if (!previous || cutoff <= view.top) return following ? [following] : [];
        // At the natural end, keep the end rows as content above them grows.
        if (following && !room.end && box.scrollTop + box.clientHeight >= box.scrollHeight - 1) return [following];
        const panel = previous.closest('[data-profile]').querySelector('.account-resource-panel');
        const visible = [...(panel?.querySelectorAll('[data-resource-reading]') || [])].map(node => ({
            node, rect: node.getBoundingClientRect(),
        })).filter(({ rect }) => rect.height && rect.bottom > view.top && rect.top < cutoff)
            // If a reset and its grant both cross the reading edge, the later start is the grant.
            .sort((a, b) => a.rect.top <= view.top && b.rect.top <= view.top
                ? b.rect.top - a.rect.top : a.rect.top - b.rect.top);
        return [...visible.map(({ node }) => node), ...(following ? [following] : [])];
    }
    function resolve({ node, key, panel }) {
        if (node.isConnected) return node;
        if (node.dataset.profile) {
            return [...list.querySelectorAll(ROW)]
                .find(row => row.dataset.harness === node.dataset.harness && row.dataset.profile === node.dataset.profile);
        }
        return [...(panel?.querySelectorAll('[data-resource-reading]') || [])]
            .find(candidate => candidate.dataset.resourceReading === key);
    }
    function preserve(host, update) {
        if (list !== host) {
            dispose();
            list = host;
            box = host.closest?.('[data-settings-page]');
            lastTop = box?.scrollTop;
            box?.addEventListener('scroll', onScroll, { passive: true });
            box?.addEventListener('pointermove', onPointerMove, { passive: true });
            box?.addEventListener('pointerdown', onPointerDown);
            box?.addEventListener('focusin', onFocusIn);
            box?.addEventListener('wheel', onWheel, { passive: true });
            box?.addEventListener('keydown', onKeyDown);
            box?.addEventListener('mousedown', onMultiPress, true);
            box?.addEventListener('click', onMultiPress, true);
            box?.ownerDocument.addEventListener('pointerup', onPointerEnd, true);
            box?.ownerDocument.addEventListener('pointercancel', onPointerEnd, true);
        }
        const control = press?.node.isConnected ? press.node : host.ownerDocument?.activeElement;
        if (painting) return update(control);
        if (!mounted()) {
            anchor = press = null;
            update(control);
            return;
        }
        noticeScroll();
        const view = box.getBoundingClientRect();
        if (anchor && !visible(anchor.node.getBoundingClientRect(), view)) anchor = null;
        const anchors = readingAnchors(host, view).filter(node => node !== anchor?.node).map(node => snapshot(node));
        if (anchor) anchors.unshift(anchor);
        // A press still held on a control inside what is being read outranks that reading place: while the same
        // control stands it stays where it is pressed, and the card or fact around it is read again from there.
        const held = anchor?.node === press?.node && press?.control?.isConnected ? snapshot(press.control) : null;
        const before = box.scrollTop;
        painting = true;
        try {
            update(control);
        } finally {
            painting = false;
        }
        if (!mounted()) return;
        restore(anchors, before);
        const slip = held?.node.getClientRects().length ? held.node.getBoundingClientRect().top - held.top : 0;
        if (Math.abs(slip) >= 1) {
            place(box.scrollTop + slip, before);
            if (anchor) anchor.top = anchor.node.getBoundingClientRect().top;
        }
        rehit();
    }
    function restore(anchors, before) {
        for (const saved of anchors) {
            const next = resolve(saved);
            if (!next?.isConnected || !next.getClientRects().length) continue;
            place(box.scrollTop + next.getBoundingClientRect().top - saved.top, before);
            // The DOM node may be new; its selected screen coordinate is not.
            anchor = { ...saved, node: next };
            // A shown result that grew keeps its actions in view.
            if (saved.shown) show(next);
            return;
        }
        anchor = null;
        setRoom('start', 0);
        setRoom('end', 0);
        lastTop = box.scrollTop;
    }
    function mount(host, rows, payload, quotaRead) {
        for (const element of host.querySelectorAll(ROW)) {
            const row = rows.find(item => item.harness === element.dataset.harness && item.profile_id === element.dataset.profile);
            if (!row) continue;
            const target = targetFor(row);
            const button = element.querySelector('[data-resources]');
            if (!button) continue;
            button.onclick = () => {
                preserve(host, () => {
                    const panel = element.querySelector('.account-resource-panel');
                    button.setAttribute('aria-expanded', String(!panel));
                    if (panel) panel.remove();
                    else {
                        const opened = host.ownerDocument.createElement('div');
                        opened.className = 'account-resource-panel ui-card';
                        opened.id = resourcePanelId(row);
                        element.append(opened);
                        paintPanel(opened, row, target);
                    }
                });
            };
            const panel = element.querySelector('.account-resource-panel');
            if (panel) paintPanel(panel, row, target);
        }

        function paintPanel(panel, row, target) {
            const markup = accountResourcesMarkup(row, payload, { quotaRead, action: store.resourceAction(target) });
            if (panelMarkup.get(panel) !== markup) {
                const active = host.ownerDocument.activeElement;
                const identity = panel.contains(active) && focusIdentity(active);
                const open = panel.querySelector('.resource-provenance')?.open;
                panel.innerHTML = markup;
                panelMarkup.set(panel, markup);
                if (open) panel.querySelector('.resource-provenance').open = true;
                if (identity) restoreFocusWithoutReveal(replacementControl(host, identity));
            }
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
                if (host.ownerDocument.activeElement === host.ownerDocument.body) {
                    restoreFocusWithoutReveal(replacementControl(host, identity));
                }
            };
        }
    }
    return { preserve, mount, dispose, command };
}
