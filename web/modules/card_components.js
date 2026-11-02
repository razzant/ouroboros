// Component cards (docs/DESIGN.md "Living cards"; docs/CLIENT_CONTRACT.md §5 "Living cards"): the declarative tree
// Ouroboros put into a living card, drawn by widget_render.js in a CARD context — no skill route, the card's `data` as
// what path fields read, only inline `data:` media — in the chat feed (chat_cards.js) and, once kept, on the Widgets
// page (`mountPinnedCard`). Local components (tabs, details) never call the server. A press on an `action` or a `form`
// is the owner's act: it goes to `POST /api/decisions` as `card:<card_id>:<submit.id>` with the card `revision` it was
// drawn from (`createCardPresses`) and comes back to Ouroboros as owner input; a card that changed since answers
// `card_stale`, so a kept copy never presses whatever control now has that id.
import { apiFetch } from './api_client.js';
import { applyChartTheme, onThemeChange } from './theme_palette.js';
import { collectSafeFieldValues } from './ui_helpers.js';
import { indexComponentTree, patchWidgetChildren, renderComponent, syncWidgetCharts } from './widget_render.js';

/** Where a card's markup lives: no skill owns it, so there is no route and no extension URL from data. */
export const CARD_RENDER_CONTEXT = Object.freeze({ routePath: () => '', routePrefix: '' });

const COMMIT_TYPES = new Set(['action', 'form']);
const SENT = { status: 'success', message: 'Sent ✓' };
const WAITING = { status: 'success', message: 'Saved · the task reads it when it continues' };
const OWED = { status: 'warning', message: 'Saved, not delivered yet. Press again to retry.' };
const STALE = { status: 'warning', message: 'This card changed since it was drawn. Look at the new version first.' };
const KEPT_STALE = { status: 'warning', message: 'This kept copy is out of date. In the chat, remove it and keep it again.' };

const newRequestId = () => globalThis.crypto?.randomUUID?.() || `card-${Date.now()}-${Math.random().toString(36).slice(2)}`;
const sameValues = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/** One press to the decision ingress: `{status, body}`, status 0 when no answer came back. */
export async function postCardDecision(body) {
    try {
        const response = await apiFetch('/api/decisions', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });
        return { status: response.status, body: await response.json().catch(() => ({})) };
    } catch (error) {
        return { status: 0, body: { error: error?.message || String(error) } };
    }
}

// A recorded press: owed, waiting in its paused author's mailbox until it reads it, or sent.
const recordedFeedback = (entry) => (entry.delivery === 'owed' ? OWED : entry.reason_code === 'awaiting_recipient_read' ? WAITING : SENT);

// What a press's answer means for its control: the feedback it shows, whether the same request id is retried by the
// next press (no answer, a server failure, delivery still owed), whether the control locks (gone, or a `once`
// control that already counted) and whether the card changed since it was drawn (`stale`: its current revision).
function pressOutcome(status, body = {}) {
    if (status === 200 && body.ok !== false) return { feedback: body.reason_code === 'awaiting_recipient_read' ? WAITING : SENT, sent: true };
    if (status === 202) return { feedback: OWED, retry: true };
    if (status === 409 && body.reason_code === 'card_stale') return { feedback: STALE, stale: Number(body.revision) || 1 };
    if (status === 409 && body.reason_code === 'card_action_once') {
        // Another press of this once control won; while it is still owed, the next press retries that one.
        const winner = body.winner || {};
        if (winner.delivery === 'owed' && winner.request_id) return { feedback: OWED, retryWith: String(winner.request_id) };
        return { feedback: { status: 'success', message: 'Already sent ✓' }, lock: true };
    }
    if (status === 404) {
        const message = body.reason_code === 'card_not_found' ? 'This card is gone.' : 'This control is no longer on the card.';
        return { feedback: { status: 'error', message }, lock: true };
    }
    if (status >= 400 && status < 500) {
        return { feedback: { status: 'error', message: `Not sent: ${body.error || `HTTP ${status}`}` } };
    }
    const reason = body.error ? `: ${body.error}` : (status ? ` (HTTP ${status})` : '');
    return { feedback: { status: 'error', message: `Not sent${reason}. Press again to retry.` }, retry: true };
}

/**
 * The presses of one card, by `submit.id`. Each press is its own submission with a fresh `request_id`; a press that
 * got no answer, failed on the server or is still owed is retried by the next press with the SAME id (when its values
 * are the same), so the host records it once. A press the card records as still owed (`adopt`, after a reload too) is
 * retried under its recorded id when it is a `once` control's (the only press left, whatever the form now holds) or an
 * action's; a form's next press with values is a new press. A `once` control's press that lost to another still owed
 * (409 with the winner) retries the winner. A `once` control locks once its press was delivered or forwarded — here,
 * or in the card's recorded presses.
 * `press(actionId, {values, once, revision})` resolves to the control's feedback (with `stale`, the card's current
 * revision, when it changed since it was drawn), or null while that control's earlier press is still in flight.
 */
export function createCardPresses({ cardId, post = postCardDecision, newId = newRequestId }) {
    const retry = new Map();
    const locked = new Map();
    const inFlight = new Set();
    return {
        locked: (actionId) => locked.has(actionId),
        lockedFeedback: (actionId) => locked.get(actionId) || null,
        /** The card's `submissions_summary`: each control's latest press decides — owed is retried, a counted `once` locks. */
        adopt(summary, onceIds) {
            const latest = new Map((Array.isArray(summary) ? summary : []).filter((entry) => entry?.action_id)
                .map((entry) => [entry.action_id, entry]));
            for (const [actionId, entry] of latest) {
                const once = onceIds.has(actionId);
                if (entry.delivery !== 'owed') {
                    if (once && (!locked.has(actionId) || locked.get(actionId) === WAITING)) locked.set(actionId, recordedFeedback(entry));
                } else if (entry.request_id && !inFlight.has(actionId) && (once || !retry.has(actionId))) {
                    retry.set(actionId, { requestId: entry.request_id, values: null, anyValues: once });
                }
            }
        },
        async press(actionId, { values = null, once = false, revision } = {}) {
            if (locked.has(actionId)) return locked.get(actionId);
            if (inFlight.has(actionId)) return null;
            const prior = retry.get(actionId);
            // A once control has one press: whatever the form holds now, its next press is that press again.
            const reuse = prior && (prior.anyValues || once || sameValues(prior.values, values));
            const requestId = reuse ? prior.requestId : newId();
            retry.set(actionId, { requestId, values, anyValues: Boolean(reuse && prior.anyValues) });
            inFlight.add(actionId);
            const answer = await post({
                request_id: requestId, decision_id: `card:${cardId}:${actionId}`,
                ...(Number.isInteger(revision) ? { revision } : {}), ...(values ? { values } : {}),
            });
            inFlight.delete(actionId);
            const outcome = pressOutcome(answer?.status ?? 0, answer?.body || {});
            if (outcome.retryWith) retry.set(actionId, { requestId: outcome.retryWith, values: null, anyValues: true });
            else if (!outcome.retry) retry.delete(actionId);
            if (outcome.lock || (once && outcome.sent)) locked.set(actionId, outcome.feedback);
            return outcome.stale ? { ...outcome.feedback, stale: outcome.stale } : outcome.feedback;
        },
    };
}

// The tree as drawn now: a commit control says "Sending…" while its press is in flight (the renderer's busy label)
// and is disabled once locked. A copy: the card's own declaration is never edited.
function drawnTree(components, presses) {
    const visit = (component) => {
        if (!component || typeof component !== 'object') return component;
        const next = { ...component };
        if (COMMIT_TYPES.has(next.type)) {
            next.busy_label = next.busy_label || 'Sending…';
            if (presses.locked(next.submit?.id)) next.disabled = true;
        }
        if (Array.isArray(next.components)) next.components = next.components.map(visit);
        if (Array.isArray(next.tabs)) next.tabs = next.tabs.map((tab) => ({ ...tab, components: Array.isArray(tab?.components) ? tab.components.map(visit) : [] }));
        return next;
    };
    return (Array.isArray(components) ? components : []).map(visit);
}

/**
 * Mount one component card: `card` carries the `CardResponse` fields `{card_id, revision, components, data?,
 * submissions_summary?}`. `presses` is the card's ledger when the caller keeps it across mounts (a feed row released
 * and drawn again keeps its retry ids); `write(fn)` is the place's own DOM write (the feed keeps its reading position
 * through it); `onStale(revision)` reads a card that changed since it was drawn (none: a kept copy says it is out of
 * date). Returns `{update(card), dispose()}`; `update` draws a new revision in place.
 */
export function mountCardComponents(mount, card, { presses = createCardPresses({ cardId: card.card_id }), write = (fn) => fn(), onStale = null } = {}) {
    const componentState = {};
    const formValues = {};
    const pendingActions = new Set();
    const charts = new Map();
    const chartShapes = new Map();
    const interactions = new AbortController();
    let current = card;
    let index = indexComponentTree(card.components);
    let disposed = false;
    const unsubscribeTheme = onThemeChange(() => charts.forEach((chart) => applyChartTheme(chart)));

    const adoptSubmissions = () => {
        const commits = index.entries.filter(({ component }) => COMMIT_TYPES.has(component.type) && component.submit?.id);
        presses.adopt(current.submissions_summary, new Set(commits.filter(({ component }) => component.submit.once === true)
            .map(({ component }) => component.submit.id)));
        const latest = new Map((Array.isArray(current.submissions_summary) ? current.submissions_summary : [])
            .map((entry) => [entry?.action_id, entry]));
        for (const { component, key } of commits) {
            const recorded = latest.get(component.submit.id);
            const lockedFeedback = presses.lockedFeedback(component.submit.id);
            if (lockedFeedback) componentState[`feedback:${key}`] = lockedFeedback;
            else if (recorded && (!componentState[`feedback:${key}`] || componentState[`feedback:${key}`] === WAITING)) {
                componentState[`feedback:${key}`] = recordedFeedback(recorded);  // a wait ends once its author read the press
            }
        }
    };
    const render = () => {
        if (disposed) return;
        const view = { state: { result: current.data || {} }, status: {}, componentState, formValues, pendingActions, visibleKeys: new Set() };
        const desired = document.createElement('template');
        desired.innerHTML = drawnTree(current.components, presses)
            .map((component, idx) => renderComponent(CARD_RENDER_CONTEXT, component, view, `components.${idx}`)).join('');
        patchWidgetChildren(mount, desired.content);
        syncWidgetCharts(mount, charts, chartShapes);
    };
    const send = async (key, values) => {
        const spec = index.byKey.get(key)?.component;
        const actionId = spec?.submit?.id;
        if (disposed || !actionId || pendingActions.has(key) || presses.locked(actionId)) return;
        pendingActions.add(key);
        delete componentState[`feedback:${key}`];
        write(render);
        const feedback = await presses.press(actionId, { values, once: spec.submit.once === true, revision: Number(current.revision) || undefined });
        if (disposed) return;
        pendingActions.delete(key);
        if (feedback) componentState[`feedback:${key}`] = feedback.stale && !onStale ? KEPT_STALE : feedback;
        write(render);
        if (feedback?.stale && onStale) onStale(feedback.stale);
    };
    const listen = (type, handler) => mount.addEventListener(type, handler, { signal: interactions.signal });
    listen('submit', (event) => {
        const form = event.target.closest('[data-widget-form]');
        if (!form) return;
        event.preventDefault();
        const key = form.dataset.widgetForm || '';
        const spec = index.byKey.get(key)?.component;
        if (spec) send(key, collectSafeFieldValues(form, spec.fields || []));
    });
    listen('click', (event) => {
        const button = event.target.closest('button');
        if (!button || button.disabled) return;
        if (button.dataset.widgetAction) send(button.dataset.widgetAction, null);
        else if (button.dataset.widgetTabKey) {
            componentState[button.dataset.widgetTabKey] = Number(button.dataset.widgetTabIdx || 0);
            write(render);
        }
    });
    mount.replaceChildren();
    adoptSubmissions();
    render();
    return {
        update(next) {
            if (disposed) return;
            current = next;
            index = indexComponentTree(next.components);
            adoptSubmissions();
            render();
        },
        dispose() {
            if (disposed) return;
            disposed = true;
            interactions.abort();
            unsubscribeTheme();
            charts.forEach((chart) => chart.destroy());
            charts.clear();
        },
    };
}

/** Keep a component card on the Widgets page (`POST`) or take it off (`DELETE`): `{ok, error?}`. */
export async function setCardPinned(cardId, pinned) {
    try {
        const response = await apiFetch(`/api/cards/${encodeURIComponent(cardId)}/pin`, { method: pinned ? 'POST' : 'DELETE' });
        const body = await response.json().catch(() => ({}));
        return response.ok ? { ok: true, ...body } : { ok: false, error: body.error || `HTTP ${response.status}`, reason: body.reason_code || '' };
    } catch (error) {
        return { ok: false, error: error?.message || String(error) };
    }
}

/** A kept card on the Widgets page (a `pinned_card` tab of skill `_cards`): the same host, presses to `card:` with the
 *  revision it was kept at (a card replaced since says the copy is out of date). */
export function mountPinnedCard(mount, tab) {
    const render = tab.render || {};
    // A kept copy hears no `card_state`, so it never shows a wait whose end it could not see: `Sent ✓`, as before.
    const post = async (body) => {
        const answer = await postCardDecision(body);
        return answer?.body?.reason_code === 'awaiting_recipient_read' ? { ...answer, body: { ...answer.body, reason_code: '' } } : answer;
    };
    const mounted = mountCardComponents(mount, { card_id: tab.tab_id, revision: Number(tab.revision) || 1,
        components: render.components, data: render.data }, { presses: createCardPresses({ cardId: tab.tab_id, post }) });
    return () => mounted.dispose();
}
