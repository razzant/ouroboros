// Living cards in the chat feed (docs/DESIGN.md "Living cards"; docs/CLIENT_CONTRACT.md §5 "Living cards"). A card
// row is built one way live (the `card` frame) and on replay (a `msg_type: "card"` history row): both carry only the
// card's identity `{card_id, kind, title, revision}` and its plain `fallback_text`. The body is read from
// `GET /api/cards/{id}` (cached by revision); while it loads, or when it cannot be read, the card shows the fallback
// text — never an empty box. A `card_state` frame re-reads a card whose revision grew and takes new presses in place.
//   components → card_components.js (presses go to the card's `card:` decisions);
//   widget     → the live skill's tab from `GET /api/widgets`, inline: declarative through the Widgets page's own
//                declarative host, module/iframe in a frame under the message's size policy; never registered on
//                the Widgets page, so leaving that page disposes nothing here: the feed disposes it with its row;
//   app        → card_frames.js, an opaque frame that cannot fetch or load; its submit fills the owner's draft.
// Frames mount lazily near the viewport, at most three live at once; a far one becomes a static face with Open. A
// running frame has Close; a closed card, and one whose last start the page did not outlive (the start ledger,
// holds.js), rests as its face until Open, on this device.
import { apiClient, apiFetch, fetchJson } from './api_client.js';
import { createCardPresses, mountCardComponents, setCardPinned } from './card_components.js';
import { createInlineFit, mountCardApp } from './card_frames.js';
import { getHolds, HELD_LINES, settleAfterFrameLoad, START_SETTLE_MS } from './holds.js';
import { escapeHtmlText as escapeHtml } from './utils.js';
import { widgetPublication } from './widget_list.js';
import { frameHeight, mountModuleWidget, mountRouteIframeWidget } from './widget_module.js';
import { mountDeclarativeWidget } from './widgets.js';

export const MAX_LIVE_CARD_FRAMES = 3;
const NEAR_MARGIN = '240px 0px';
const MAKE_SKILL = (title) => `Turn the card «${title}» into a skill so I can keep it on Widgets.`;

// Shared by every room on the page: a card's current body by id, an app's bytes by hash, the live widget list, and
// each card's press ledger (a row drawn again keeps the retry ids of presses still owed).
const bodies = new Map();
const appTexts = new Map();
const presses = new Map();
let widgetTabs = null;

/** The current body of a card, read again only when `revision` is newer than the one held. */
export function readCardBody(cardId, revision = 0, read = (url) => fetchJson(url, { cache: 'no-store' })) {
    const held = bodies.get(cardId);
    if (held && (held.failed ? false : held.revision >= revision)) return held.promise;
    // A read that a newer one replaced while it was in flight leaves the newer entry alone.
    const promise = read(`/api/cards/${encodeURIComponent(cardId)}`).then((body) => {
        if (bodies.get(cardId)?.promise === promise) bodies.set(cardId, { revision: Number(body?.revision) || revision, promise });
        return body;
    }, (error) => {
        if (bodies.get(cardId)?.promise === promise) bodies.set(cardId, { revision, promise, failed: true });
        throw error;
    });
    bodies.set(cardId, { revision, promise });
    return promise;
}

function readAppText(url, sha256) {
    if (!appTexts.has(sha256)) {
        appTexts.set(sha256, apiFetch(url, { cache: 'no-store' }).then((response) => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.text();
        }).catch((error) => { appTexts.delete(sha256); throw error; }));
    }
    return appTexts.get(sha256);
}

function liveWidgetTabs() {
    widgetTabs ||= apiClient.widgets().then((data) => (Array.isArray(data?.ui_tabs) ? data.ui_tabs : []))
        .catch((error) => { widgetTabs = null; throw error; });
    return widgetTabs;
}

/** What a skill's widget card draws from; another value draws it again (the board's rule, widget_list.js). */
function chatWidgetSignature(tab) {
    return JSON.stringify([tab.revision, widgetPublication(tab), tab.render]);
}

const pressLedger = (cardId) => {
    if (!presses.has(cardId)) presses.set(cardId, createCardPresses({ cardId }));
    return presses.get(cardId);
};

/**
 * At most `max` live frames: `near(record, isNear)` follows the viewport (a frame that drifts far unmounts, a near
 * one mounts while there is room and `canStart(record)` allows it), `open(record)` mounts one at the owner's request
 * (the oldest other live frame makes room), `forget(record)` drops a released row.
 */
export function createFrameBudget({ max = MAX_LIVE_CARD_FRAMES, mount, unmount, canStart = () => true }) {
    const live = new Set();
    const nearby = new Set();
    const start = (record) => {
        if (live.has(record)) return;
        live.add(record);
        mount(record);
    };
    const stop = (record) => {
        if (!live.delete(record)) return;
        unmount(record);
    };
    const fill = () => {
        for (const record of nearby) if (live.size < max && canStart(record)) start(record);
    };
    return {
        live: () => [...live],
        near(record, isNear) {
            if (isNear) nearby.add(record);
            else { nearby.delete(record); stop(record); }
            fill();
        },
        open(record) {
            if (!live.has(record) && live.size >= max) stop([...live].find((other) => !nearby.has(other)) || [...live][0]);
            start(record);
        },
        forget(record) { nearby.delete(record); stop(record); fill(); },
    };
}

const button = (label, data) => `<button type="button" class="btn btn-default btn-sm chat-card-action" ${data}>${escapeHtml(label)}</button>`;

/**
 * One room's card rows. `frameNode(msg, node)` wraps a node in an Ouroboros bubble (sender, time, row address);
 * `decorate(bubble, msg, text)` gives it Copy (the fallback text) and Reply; `composer()` is the room's composer, whose
 * draft an app's submit and "Make it a skill" fill; `onDomWrite(write)` keeps the reading position; `holds` is the
 * start ledger and `settleMs` its settle second (a test seam).
 */
export function createChatCards({ frameNode, decorate = () => {}, composer = () => null, onDomWrite = (write) => write(),
    holds = getHolds(), settleMs = START_SETTLE_MS }) {
    const records = new Map();
    const byBubble = new WeakMap();
    const messageHandlers = new Set();
    let destroyed = false;
    // The owner's Close is this card's; a start hold belongs to the code (an app's bytes, a skill widget's revision).
    const closeKey = (record) => `card:${record.id}`;
    const codeHold = (record) => (record.framedKind === 'app'
        ? { key: `app:${record.id}`, rev: String(record.data?.app?.sha256 || '') }
        : { key: `widget:${record.tab?.skill}:${record.tab?.tab_id}`, rev: String(record.tab?.revision ?? '') });
    const heldReason = (record) => {
        if (holds.reason(closeKey(record)) === 'owner') return 'owner';
        const code = codeHold(record);
        return holds.reason(code.key, code.rev);
    };
    const budget = createFrameBudget({
        mount: (record) => { record.framed = startFrame(record); write(() => paintActions(record)); },
        unmount: (record) => { record.framed?.(); record.framed = null; showFace(record); write(() => paintActions(record)); },
        canStart: (record) => !heldReason(record),
    });
    const observer = typeof IntersectionObserver === 'function'
        ? new IntersectionObserver((entries) => entries.forEach((entry) => {
            const record = byBubble.get(entry.target);
            if (record?.framedKind && records.get(record.id) === record) budget.near(record, entry.isIntersecting);
        }), { rootMargin: NEAR_MARGIN })
        : null;

    // Every change to a card's height goes through the room's viewport write, so a room reading its newest message
    // keeps following it (a late growth outside it would leave that message under the composer).
    const write = (fn) => (destroyed ? undefined : onDomWrite(fn));
    const insertLine = (line) => {
        const room = composer();
        if (!room?.input) return false;
        const at = room.input.selectionStart ?? room.input.value.length;
        const before = room.input.value.slice(0, at);
        room.insertAtCursor(`${before && !before.endsWith('\n') ? '\n' : ''}${line}`);
        return true;
    };
    const setStatus = (record, text, tone = 'muted') => {
        record.status.hidden = !text;
        record.status.textContent = text;
        record.status.dataset.tone = tone;
    };
    const showFallback = (record, note = '', tone = 'muted', retry = false) => write(() => {
        record.body.innerHTML = `<p class="chat-card-fallback">${escapeHtml(record.fallback)}</p>`;
        setStatus(record, note, tone);
        if (retry) record.status.insertAdjacentHTML('beforeend', ` ${button('Retry', 'data-card-retry')}`);
    });
    // Every disposal starts a new draw: a draw still resolving (the widget list, a widget's first paint) finds its
    // generation passed and mounts nothing, so overlapping draws never both mount and a later draw always wins.
    const disposeBody = (record) => {
        record.draw += 1;
        observer?.unobserve(record.bubble);
        record.mounted?.dispose?.();
        record.mounted = null;
        record.fit?.dispose();
        record.fit = null;
        if (record.framedKind) budget.forget(record);
        record.framedKind = '';
    };

    // The foot: "Ouroboros made this", then what the owner can do with it — keep a component card on Widgets (and
    // take it off again), or ask Ouroboros to make an app into a skill. A widget card is a skill's: nothing to keep.
    function paintActions(record, justPinned = false) {
        const body = record.data;
        let html = '';
        if (body?.kind === 'components') {
            html = body.pinned
                ? `<span class="ui-status" data-tone="ok">Kept on Widgets</span> ${button(justPinned ? 'Undo' : 'Remove', 'data-card-unpin')}`
                : button('Keep on Widgets', 'data-card-pin');
        } else if (body?.kind === 'app') html = button('Make it a skill', 'data-card-make-skill');
        if (record.framed) html = `${html} ${button('Close', 'data-card-close')}`.trim();
        record.actions.innerHTML = html;
    }

    async function setPinned(record, pinned) {
        record.actions.querySelectorAll('button').forEach((node) => { node.disabled = true; });
        const result = await setCardPinned(record.id, pinned);
        if (destroyed || !records.has(record.id)) return;
        if (result.ok && record.data) record.data = { ...record.data, pinned };
        write(() => {
            paintActions(record, pinned && result.ok);
            if (!result.ok) {
                record.actions.insertAdjacentHTML('afterbegin', `<span class="ui-status" data-tone="danger">${escapeHtml(
                    `${pinned ? 'Not kept' : 'Not removed'}: ${result.error}`)}</span> `);
            }
        });
    }

    function showFace(record) {
        const line = HELD_LINES[heldReason(record)] || '';
        write(() => {
            record.body.innerHTML = `<div class="chat-card-face"><p class="chat-card-fallback">${escapeHtml(record.fallback)}</p>${button('Open', 'data-card-open')}</div>`;
            record.toggle.hidden = true;
            setStatus(record, line);
        });
    }

    // One live frame (an app, or a framed widget): mounts into the body and returns its disposer.
    function startFrame(record) {
        // The host reserves the frame's height from the start, so the frame arriving later moves nothing.
        const host = document.createElement('div');
        host.className = 'chat-card-frame-host';
        const controller = new AbortController();
        let sizing = null;
        write(() => { record.body.replaceChildren(host); setStatus(record, ''); sizing = createInlineFit(host, record.toggle); });
        // This frame's own size policy and fault line. A stopped frame reaches neither the frame that replaced it nor
        // the shared Expand: a module widget answers until its ordered stop ends, and a mount can resolve late.
        const untilStopped = (fn) => (value) => { if (!controller.signal.aborted) fn(value); };
        const fit = { fit: untilStopped((height) => write(() => sizing.fit(height))), dispose: () => sizing?.dispose() };
        record.fit = fit;
        let dispose = null;
        // The ledger entry stands before the frame's first byte runs; the frame's load and one host second clear it.
        const code = codeHold(record);
        const settle = holds.starting(code.key, code.rev);
        let cancelSettle = () => {};
        const done = (fn) => {
            if (controller.signal.aborted) { fn?.(); return; }
            dispose = fn;
            cancelSettle = settleAfterFrameLoad(host, settle, { delay: settleMs });
        };
        const fail = (text) => { settle(); if (!controller.signal.aborted) showFallback(record, text, 'danger', true); };
        const fault = untilStopped((text) => write(() => setStatus(record, text, 'danger')));
        const body = record.data;
        const fresh = record.fresh === true;
        record.fresh = false;
        if (record.framedKind === 'app') {
            readAppText(body.app.url, body.app.sha256).then((html) => done(write(() => mountCardApp(host, {
                cardId: record.id, title: body.title, sha256: body.app.sha256, html, fit, fresh, onFault: fault,
                onSubmit: (line) => insertLine(line),
                onLeave: () => {
                    budget.forget(record);
                    write(() => setStatus(record, 'The app tried to open a page and was stopped.', 'danger'));
                },
            }))), (error) => fail(`The app could not be read: ${error.message || error}`));
        } else {
            const { tab } = record;
            const hooks = { fit: fit.fit, fault };
            Promise.resolve().then(() => (tab.render.kind === 'module'
                ? mountModuleWidget(host, tab, tab.render, controller.signal, messageHandlers, hooks)
                : mountRouteIframeWidget(host, tab, tab.render, controller.signal, hooks, messageHandlers))).then((fn) => {
                if (tab.render.kind !== 'module') {
                    host.querySelector('iframe')?.style.removeProperty('--widget-frame-height');
                    fit.fit(frameHeight(tab.render));
                }
                done(fn);
            }, (error) => fail(`The widget could not start: ${error.message || error}`));
        }
        return () => {
            controller.abort();
            cancelSettle();
            settle();
            const stopping = dispose?.();
            if (stopping?.then) stopping.catch(() => {});
            fit.dispose();
            if (record.fit === fit) record.fit = null;
        };
    }

    async function mountWidget(record, ref) {
        const draw = record.draw;
        const current = () => !destroyed && records.get(record.id) === record && record.draw === draw;
        let tab = null;
        try {
            tab = (await liveWidgetTabs()).find((item) => item.skill === ref?.skill && item.tab_id === ref?.tab_id && !item.pinned_card) || null;
        } catch (error) {
            if (current()) showFallback(record, `The widget list could not be read: ${error.message || error}`, 'danger', true);
            return;
        }
        if (!current()) return;
        if (!tab) { showFallback(record, 'This skill is off.'); return; }
        record.tab = tab;
        record.widgetSignature = chatWidgetSignature(tab);
        if (tab.render?.kind === 'declarative') {
            // Under the message's size policy like a frame: the content is measured, its box is capped, Expand
            // lifts the cap. The declarative host paints its first frame synchronously: inside this write.
            const host = document.createElement('div');
            host.className = 'chat-card-widget';
            const content = document.createElement('div');
            host.append(content);
            let inline = null;
            const mounting = write(() => {
                record.body.replaceChildren(host);
                setStatus(record, '');
                inline = createInlineFit(host, record.toggle);
                return mountDeclarativeWidget(content, { ...tab, key: `card:${record.id}` }, tab.render, { input: ref.input });
            });
            const resized = typeof ResizeObserver === 'function' ? new ResizeObserver(() => record.fit?.fit(content.offsetHeight)) : null;
            record.fit = {
                fit: (height) => write(() => inline.fit(height)),
                dispose: () => { resized?.disconnect(); inline.dispose(); },
            };
            resized?.observe(content);
            record.fit.fit(content.offsetHeight);
            const dispose = await mounting;
            if (!current()) dispose?.();
            else record.mounted = { dispose };
            return;
        }
        if (!['module', 'iframe'].includes(tab.render?.kind)) { showFallback(record, 'This widget cannot be shown in a message.'); return; }
        record.framedKind = 'widget';
        showFace(record);
        if (observer) observer.observe(record.bubble);
    }

    function drawBody(record) {
        const body = record.data;
        disposeBody(record);
        write(() => paintActions(record));
        if (body.kind === 'components') {
            write(() => {
                const host = document.createElement('div');
                host.className = 'chat-card-components';
                record.body.replaceChildren(host);
                setStatus(record, '');
                record.mounted = mountCardComponents(host, body, { presses: pressLedger(record.id), write, onStale: (revision) => load(record, revision) });
            });
        } else if (body.kind === 'widget') {
            mountWidget(record, body.widget);
        } else if (body.kind === 'app' && body.app?.url) {
            record.framedKind = 'app';
            showFace(record);
            if (observer) observer.observe(record.bubble);
        } else {
            showFallback(record, 'This card cannot be shown here.');
        }
    }

    // A replaced card (`show_card replace`) brings its own title and words: the heading, the card's label, the words
    // Copy, Reply and message actions take, and every fallback face follow them.
    function retitle(record, body) {
        const title = String(body?.title ?? record.title);
        const fallback = String(body?.fallback_text ?? record.fallback);
        if (title === record.title && fallback === record.fallback) return;
        record.title = title;
        record.fallback = fallback;
        write(() => {
            record.heading.textContent = title;
            record.node.setAttribute('aria-label', title || 'Card');
            record.bubble.dataset.rowWords = fallback;
        });
    }

    // `quiet`: a re-read that fails keeps the body already drawn (a reconnect's check is not the card failing).
    function load(record, revision, { quiet = false } = {}) {
        readCardBody(record.id, revision).then((body) => {
            if (destroyed || records.get(record.id) !== record) return;
            // An earlier read that settles after a newer one never draws the older card back.
            if (record.data && (Number(body?.revision) || 0) < (Number(record.data.revision) || 0)) return;
            retitle(record, body);
            const sameBody = record.data && record.data.revision === body.revision && record.data.kind === body.kind;
            record.data = body;
            // The same body drawn already: only its presses and pin can have changed; a running frame keeps running.
            if (sameBody) write(() => { paintActions(record); if (body.kind === 'components') record.mounted?.update(body); });
            else drawBody(record);
        }, (error) => {
            if (destroyed || records.get(record.id) !== record || (quiet && record.data)) return;
            if (error?.status === 404) showFallback(record, 'This card is no longer available.');
            else showFallback(record, `The card could not be loaded: ${error?.message || error}`, 'danger', true);
        });
    }

    function onClick(record, event) {
        const target = event.target.closest?.('button');
        if (!target || target.disabled) return;
        if (target.hasAttribute('data-card-retry')) {
            bodies.delete(record.id);
            widgetTabs = null;
            record.data = null;
            disposeBody(record);
            showFallback(record, 'Loading the card…');
            load(record, record.revision);
        } else if (target.hasAttribute('data-card-open')) {
            // A card held after a start the page did not outlive starts without the state it saved: that may be the cause.
            const reason = heldReason(record);
            holds.release(closeKey(record));
            holds.release(codeHold(record).key);
            record.fresh = reason === 'start' || reason === 'all';
            budget.open(record);
        } else if (target.hasAttribute('data-card-close')) {
            holds.stop(closeKey(record));
            budget.forget(record);
        } else if (target.hasAttribute('data-card-pin')) setPinned(record, true);
        else if (target.hasAttribute('data-card-unpin')) setPinned(record, false);
        else if (target.hasAttribute('data-card-make-skill')) insertLine(MAKE_SKILL(record.title));
    }

    /** The bubble for a card row (live frame or history row), or null without a card id. */
    function buildCard(msg) {
        const card = msg?.card || {};
        const id = String(card.card_id || '');
        if (destroyed || !id) return null;
        const title = String(card.title || '');
        const fallback = String(msg.fallback_text ?? msg.text ?? '');
        const node = document.createElement('section');
        node.className = 'chat-card';
        node.dataset.cardId = id;
        node.dataset.cardKind = String(card.kind || '');
        node.setAttribute('aria-label', title || 'Card');
        node.innerHTML = `<div class="chat-card-title" role="heading" aria-level="3">${escapeHtml(title)}</div>
            <div class="chat-card-body" data-card-body><p class="chat-card-fallback">${escapeHtml(fallback)}</p></div>
            <div class="chat-card-status ui-status" data-tone="muted" role="status" aria-live="polite">Loading the card…</div>
            <button type="button" class="btn btn-ghost btn-sm chat-card-expand" aria-expanded="false" hidden>Expand</button>
            <div class="chat-card-foot"><span class="chat-card-made">Ouroboros made this</span><span class="chat-card-actions" data-card-actions></span></div>`;
        const bubble = frameNode(msg, node);
        bubble.classList.add('chat-card-bubble');
        bubble.dataset.cardId = id;
        // A card has no `.message`: its plain words are what Reply quotes and a message action reads.
        bubble.dataset.rowWords = fallback;
        const prior = records.get(id);
        if (prior) release(prior.bubble);
        const record = {
            id, title, fallback, node, bubble, revision: Number(card.revision) || 0, heading: node.querySelector('.chat-card-title'),
            body: node.querySelector('[data-card-body]'), status: node.querySelector('.chat-card-status'),
            toggle: node.querySelector('.chat-card-expand'), actions: node.querySelector('[data-card-actions]'),
            data: null, mounted: null, framed: null, framedKind: '', fit: null, tab: null, draw: 0,
        };
        node.addEventListener('click', (event) => onClick(record, event));
        records.set(id, record);
        byBubble.set(bubble, record);
        decorate(bubble, msg, fallback);
        load(record, record.revision);
        return bubble;
    }

    /** A `card_state` frame: a newer revision is read again; recorded presses update the drawn card in place. */
    function applyCardState(msg) {
        const record = records.get(String(msg?.card?.card_id || ''));
        if (!record) return false;
        const revision = Number(msg.card.revision) || 0;
        record.revision = Math.max(record.revision, revision);
        if (!record.data || revision > record.data.revision) {
            load(record, revision);
            return true;
        }
        record.data = { ...record.data, submissions_summary: msg.submissions_summary || [] };
        bodies.set(record.id, { revision: record.data.revision, promise: Promise.resolve(record.data) });
        write(() => record.mounted?.update?.(record.data));
        return true;
    }

    /** `extension_lifecycle`: a kept card's pin, or a skill's widget that changed or left. */
    function lifecycle(msg) {
        if (msg?.skill === '_cards') {
            const record = records.get(String(msg.data?.card_id || ''));
            const pinned = msg.action === 'card_pinned';
            // The echo of this room's own Keep / Undo changes nothing; another window's does.
            if (record?.data?.kind === 'components' && record.data.pinned !== pinned) {
                record.data = { ...record.data, pinned };
                bodies.delete(record.id);
                write(() => paintActions(record, false));
            }
            return;
        }
        widgetTabs = null;
        const shown = [...records.values()].filter((record) => record.data?.kind === 'widget');
        if (!shown.length) return;
        liveWidgetTabs().then((tabs) => shown.forEach((record) => {
            const ref = record.data?.widget;
            const tab = tabs.find((item) => item.skill === ref?.skill && item.tab_id === ref?.tab_id && !item.pinned_card);
            if (records.get(record.id) !== record || (tab && chatWidgetSignature(tab) === record.widgetSignature)) return;
            drawBody(record);
        }), () => {});
    }

    /** The socket opened again: a change missed meanwhile is read now (a card drawn already stays as it is). */
    function reconnect() {
        for (const record of records.values()) {
            bodies.delete(record.id);
            load(record, record.revision, { quiet: true });
        }
    }

    /** The rows inside a released subtree give back everything they hold. */
    function release(root) {
        for (const record of [...records.values()]) {
            if (root !== record.bubble && !root?.contains?.(record.bubble)) continue;
            disposeBody(record);
            records.delete(record.id);
        }
    }

    return {
        buildCard,
        applyCardState,
        lifecycle,
        reconnect,
        /** Every socket message, for module widgets that listen to their skill's events. */
        relay(msg) { messageHandlers.forEach((handler) => handler(msg)); },
        release,
        destroy() {
            if (destroyed) return;
            for (const record of records.values()) disposeBody(record);
            records.clear();
            observer?.disconnect();
            messageHandlers.clear();
            destroyed = true;
        },
    };
}
