// Your Ouroboroses (DESIGN "Circles and Add"): the owner's own Ouroboroses on other computers, a card of their own on
// the Circles page above the circles. Each is a place of its own kind
// (peer_links: connector `ouroboros`, marked mine) with one row that opens in place, as a circle does, in the order
// they were introduced. Closed, the row says who it is (its face, its name, the time of its last captured event, as
// every place row) and the link's state with who may give whom work, in words; open, the owner's rule for it in their
// own voice with Edit, requests for work from it as this side's one switch, requests to it as the fact the other side
// gave at the introduction, the link's Active / Paused strip with its code, Open conversation (the place sheet, where
// every place's conversation opens; before anything has arrived, at the room this side wrote to it in) and Remove link
// through the app's confirmation. Requests are work: the switch
// never stops its other words. A link waiting for its code check, removed, or no longer known on the other
// side says that first, with its one next step. The facts are the place's (`/api/places`); the writes are the place's
// own (`updatePlace`) and the link doors (peer_links_api.js); each result is written beside its control (.ui-result).
import { escapeHtmlAttr as escapeHtml, errorWords, SAVE_ACK_MS, setResult } from './ui_primitives.js';
import { fmt, tr } from './i18n.js';
import { relativeAge } from './skill_presentation.js';
import { updatePlace } from './consciousness_api.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { renderSegmentedField } from './page_header.js';
import { faceHtml } from './ouroboros_face.js';
import { isPeerPlace, revokeLink } from './peer_links_api.js';
import { taskPresentation } from './log_events.js';
import { PLACE_STATES } from './consciousness_words.js';

/** This installation's own name in its interface (its card in the sidebar). */
export const HERE = 'Ouroboros';
const SEP = ' · ';
const STATES = {
    active: { word: 'Connected', tone: 'ok' },
    check: { word: 'Waiting for your check', tone: 'warn' },
    paused: { word: 'Paused', tone: 'muted' },
    silent: { word: 'Silent', tone: 'muted' },
    removed: { word: 'Link removed', tone: 'muted' },
    lost: { word: 'Needs a new introduction', tone: 'warn' },
    incomplete: { word: 'Introduction incomplete', tone: 'warn' },
    unknown: { word: 'State unknown', tone: 'muted' },
};
const pad = (n) => String(n).padStart(2, '0');
const dateOf = (iso) => { const date = new Date(iso); return Number.isFinite(date.getTime()) ? date : null; };

/** «14:40» today, «yesterday 14:40», else «Nov 21 14:40»; '' for no time. */
export function whenWords(iso, now = new Date()) {
    const date = dateOf(iso);
    if (!date) return '';
    const clock = `${pad(date.getHours())}:${pad(date.getMinutes())}`;
    const day = (value) => new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime();
    const days = Math.round((day(now) - day(date)) / 86_400_000);
    if (days === 0) return fmt('today {clock}', { clock });
    if (days === 1) return fmt('yesterday {clock}', { clock });
    return `${date.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${clock}`;
}

/** The name a linked place goes by here: its label, else «Ouroboros». */
export const peerName = (place) => place?.label || 'Ouroboros';

/** The link's state on this side: removed, no longer known on the other side (`lost`), waiting for its code check,
 *  paused or active ('unknown' when no link names the place or its registry could not be read). */
export function linkStateOf(place) {
    const link = place?.link;
    if (!link) return place?.peerEffective?.reason === 'unlinked' ? 'incomplete' : 'unknown';
    if (link.revokedAt) return 'removed';
    if (link.remoteUnknownAt) return 'lost';
    if (link.codeCheck !== 'confirmed') return 'check';
    return ['paused', 'silent', 'active'].includes(place.state) ? place.state : 'unknown';
}

/** The block's order: as they were introduced, the oldest first (a later one never jumps above an earlier one); the
 *  place id keeps two introduced at once in one order, and one whose link could not be read comes last. */
export function ownOrder(places) {
    const at = (place) => place.link?.createdAt || '￿';
    return [...places].sort((a, b) => at(a).localeCompare(at(b)) || String(a.id).localeCompare(String(b.id)));
}

/** Whether its letters that assign work start work here now, as the letter door reads it (`peerEffective`,
 *  peer_links.effective_permissions): true | false | null (not known now). The switch keeps the owner's stored choice;
 *  these words never derive what the host did not say. */
export const startsWorkHere = (place) => {
    const effect = place?.peerEffective;
    return effect ? effect.assignments : null;
};

/** Current incoming work is independent of speech; the outgoing permission is reported separately. */
export function directionWords(place) {
    const from = startsWorkHere(place);
    return from === true ? tr('peer.direction.in.yes', 'Can ask for work here')
        : from === false ? tr('peer.direction.in.no', 'Cannot start work here')
            : tr('peer.direction.in.unknown', 'Permission to start work here is unknown');
}

/** The room this side wrote to it in while nothing has arrived from it (`peerRoom`, named by the host), else null. */
export const wroteRoom = (place) => (!place?.rooms?.length && place?.peerRoom?.roomChatId ? place.peerRoom : null);

/** The row's second line: when the link was made or removed while that is its news, else the time of its last captured
 *  event, as every place row says it (never «no activity»). */
export function rowLine(place, state = linkStateOf(place)) {
    const link = place.link || {};
    if (state === 'check') return fmt('introduced {when}', { when: whenWords(link.createdAt) });
    if (state === 'removed') return fmt('removed {when}', { when: whenWords(link.revokedAt) });
    if (place.lastActivityAt) return relativeAge(place.lastActivityAt) || tr('places.event.unknown', 'last event unknown');
    // With nothing captured, its news is what this side wrote to it (its time), else the work its letter started here:
    // who sent it (its record keeps no date).
    const wrote = wroteRoom(place), work = place.peerWork?.taskId ? place.peerWork : null;
    if (wrote?.wroteAt) return fmt('{here} wrote {when}', { here: HERE, when: relativeAge(wrote.wroteAt) || whenWords(wrote.wroteAt) });
    if (work) return fmt('work from {name}', { name: work.from || peerName(place) });
    return tr('places.no_event', 'no captured event');
}

/** The owner's rule for it: the newest of their notes on its place (the introduction wrote the first). */
export const ruleNote = (place) => [...(place.notes || [])].reverse().find((note) => ['owner', 'migration'].includes(note.author)) || null;

/** «418 207»: the six digits as both installations show them. */
export const codeWords = (code) => String(code || '').replace(/^(\d{3})(\d{3})$/, '$1 $2');

/** The owner's words, read whole in their voice (the circle's words' part: .place-note.is-owner). */
export const ownerWordsHtml = (text) => `<div class="place-note circle-words-text is-owner"><p data-i18n-authored>${escapeHtml(text)}</p></div>`;

const act = (name, place, label, tone = 'btn-default', attrs = '') => `<button type="button" class="btn ${tone} btn-sm" data-own-act="${name}" data-place="${escapeHtml(place.id)}"${attrs}>${escapeHtml(label)}</button>`;
/** The line an action's result is written in after the paint (setResult); in a row, a line of its own under it. */
const resultHtml = (key, row = false) => `<span class="ui-result${row ? ' settings-row-result' : ''}" data-own-result="${escapeHtml(key)}" role="status" aria-live="polite"></span>`;
/** While its write is out, a control shows the value it is writing (never the old one), held: disabled, aria-busy. */
const writing = (place, ui, field) => ui.busy === place.id && ui.pending?.field === field;
const shown = (place, ui, field, stored) => (writing(place, ui, field) ? ui.pending.value : stored);
const HELD = ' disabled aria-busy="true"';

function ruleHtml(place, ui, editable) {
    const name = peerName(place), note = ruleNote(place), label = fmt('Your rule for {name}', { name });
    if (editable && ui.editing === place.id) {
        const draft = ui.drafts[place.id];
        const value = draft?.value ?? note?.text ?? '';
        const concurrent = draft?.conflict
            ? `<p class="peer-field-label">${escapeHtml(tr('peer.rule.latest', 'Current rule'))}</p>${draft.base.text !== null
                ? ownerWordsHtml(draft.base.text) : `<p class="own-text">${escapeHtml(tr('peer.rule.removed', 'There is no saved rule now.'))}</p>`}` : '';
        return `<div class="own-rule"><label class="ui-field"><span class="peer-field-label" data-i18n-skip>${escapeHtml(label)}</span>
            <textarea class="ui-control consciousness-owner-words" rows="4" data-own-draft="${escapeHtml(place.id)}" data-i18n-authored spellcheck="false">${escapeHtml(value)}</textarea></label>
            <div class="ui-action-row">${act('save-rule', place, tr('peer.rule.save', 'Save'), 'btn-default', writing(place, ui, 'rule') ? HELD : '')}${act('cancel-rule', place, tr('peer.rule.cancel', 'Cancel'), 'btn-ghost')}${resultHtml(`${place.id}:rule`)}</div>${concurrent}</div>`;
    }
    const head = `<div class="own-rule-head"><p class="peer-field-label" data-i18n-skip>${escapeHtml(label)}</p>${editable
        ? act('edit-rule', place, tr('peer.rule.edit', 'Edit'), 'btn-ghost') : ''}</div>`;
    const words = note ? ownerWordsHtml(note.text) : `<p class="own-text">${escapeHtml(tr('peer.rule.none', 'No rule yet. Edit says what it may ask for and what it may be told.'))}</p>`;
    return `<div class="own-rule">${head}${words}${editable ? resultHtml(`${place.id}:rule`, true) : ''}</div>`;
}

function switchHtml(place, ui, locked, label) {
    const busy = writing(place, ui, 'mayStart'), on = shown(place, ui, 'mayStart', place.peer?.mayStart === true);
    return `<label class="ui-switch"><input type="checkbox" role="switch" data-own-switch="${escapeHtml(place.id)}" data-i18n-skip aria-label="${escapeHtml(label)}" aria-checked="${on}"${on ? ' checked' : ''}${busy ? HELD : locked ? ' disabled' : ''}><span class="ui-switch-track" aria-hidden="true"><span class="ui-switch-thumb"></span></span></label>`;
}

const rowHtml = (title, meta, actions = '', result = '') => `<div class="ui-row settings-row own-row"><p class="ui-row-title" data-i18n-skip>${escapeHtml(title)}</p>`
    + `<div class="ui-row-meta"><span data-i18n-skip>${escapeHtml(meta)}</span></div>${actions ? `<div class="ui-row-actions">${actions}</div>` : ''}${result}</div>`;

function toWords(place, link) {
    if (link === 'silent' || link === 'paused') return tr('peer.outgoing.held', 'This side sends nothing while the link is Silent or Paused.');
    const to = place.link?.remoteMayStart;
    if (to === true) return tr('peer.outgoing.reported.yes', 'Allowed at introduction. The current choice there may differ.');
    if (to === false) return tr('peer.outgoing.reported.no', 'Not allowed at introduction. The current choice there may differ.');
    return tr('peer.outgoing.unreported', 'No permission was reported by the other side.');
}

/** The latest accepted work and its room: title and sender are authored; only the state word is translated. */
export function workRowHtml(place) {
    const work = place?.peerWork;
    if (!work) return '';
    if (work.unknown) return rowHtml(fmt('Work from {name}', { name: peerName(place) }), tr('peer.work.unread', 'Its accepted work could not be read now.'));
    const state = work.phase === 'paused' ? tr('peer.work.paused', 'Paused')
        : work.phase ? taskPresentation(work.phase).headline : '';
    const door = work.roomChatId ? act('open-work', place, tr('peer.work.open', 'Open work')) : '';
    return rowHtml(fmt('Work from {name}', { name: work.from || peerName(place) }),
        [work.title, state || tr('peer.work.state.unknown', 'State unknown')].filter(Boolean).join(SEP), door);
}

/** What the letter door does now with its requests for work (its `peerEffective`), in words: the first fact that
 *  refuses them, as the host names it, else that they start work here. */
export function fromWords(place) {
    const name = peerName(place), effect = place?.peerEffective || null, names = { name, here: tr('peer.here', 'this Ouroboros') };
    if (!effect || effect.assignments === null) return fmt('Whether {name} can start work here is not known now.', { name });
    if (effect.assignments) return fmt('{name} can ask {here} for work.', names);
    return {
        paused: fmt('While the link is paused, requests from {name} are refused.', { name }),
        not_mine: fmt('{name} is not marked as yours here: its requests are refused.', { name }),
        silent: fmt('{name} is Silent here, so work it would start could not answer: its requests are refused.', { name }),
        panic: fmt('A Panic stopped {here}: requests from {name} are refused until you return.', names),
    }[effect.reason] || fmt('{name} cannot start work here.', { name });
}

function directionRowsHtml(place, ui) {
    const name = peerName(place), link = shown(place, ui, 'state', linkStateOf(place));
    const paused = linkStateOf(place) === 'paused';
    const title = fmt('Requests from {name}', { name });
    // The place's own three states (the place sheet's strip): Silent keeps the link and its messages, sends nothing.
    const strip = renderSegmentedField({ target: `own-state-${place.id}`, modifier: 'data-own-state', chosen: link,
        label: tr('peer.link', 'Link'), busy: writing(place, ui, 'state'),
        options: PLACE_STATES.map(({ value, label }) => ({ value, label: tr(`peer.link.${value}`, label) })) });
    const code = place.link?.code ? fmt('Code {code}', { code: codeWords(place.link.code) }) : '';
    const introduced = place.link?.createdAt ? fmt('introduced {when}', { when: whenWords(place.link.createdAt) }) : '';
    // While a write is out, the words stay the host's last reading; the switch holds the value it writes.
    const said = fromWords(place);
    const messages = place.peerEffective?.messages;
    const message = messages === true ? tr('peer.messages.yes', 'Messages it writes are kept in this conversation.')
        : messages === false ? tr('peer.messages.no', 'Messages it writes are not accepted here.')
            : tr('peer.messages.unknown', 'Whether messages are accepted here is not known now.');
    return `<div class="own-rows">${rowHtml(title, said, switchHtml(place, ui, paused || !place.peer || !Number.isInteger(place.revision), title), resultHtml(`${place.id}:from`, true))}`
        + rowHtml(tr('peer.messages', 'Messages from it'), message)
        + workRowHtml(place)
        + rowHtml(fmt('Requests to {name}', { name }), toWords(place, linkStateOf(place)))
        + rowHtml(tr('peer.link', 'Link'), paused ? tr('peer.link.paused.note', 'Nothing goes either way until you choose Active. The keys stay.')
            : linkStateOf(place) === 'silent' ? fmt('Silent: what {name} writes is saved here, nothing is sent to it. The keys stay.', { name })
                : [code, introduced].filter(Boolean).join(SEP), strip, resultHtml(`${place.id}:state`, true))
        + '</div>';
}

function stateTextHtml(place, state) {
    const name = peerName(place), words = {
        check: fmt('The link with {name} starts after you check the code on both sides.', { name }),
        removed: fmt('The link is removed on this side: its keys are deleted and nothing goes either way. {name} may still list {here}: remove the link in {name}\'s Circles too.', { name, here: HERE }),
        lost: fmt('{name} no longer knows this link: it was removed there, or a data folder moved. Nothing is sent to {name} until you introduce it again.', { name }),
        incomplete: tr('peer.link.incomplete', 'The introduction did not complete. Introduce it again.'),
        unknown: tr('peer.link.unread', 'Its link could not be read now.'),
    }[state];
    return words ? `<p class="own-text" data-i18n-skip>${escapeHtml(words)}</p>` : '';
}

function bodyHtml(place, ui, state) {
    const talk = (tone) => act('conversation', place, tr('places.rooms.open', 'Open conversation'), tone);
    const remove = act('remove', place, tr('peer.link.remove', 'Remove link'), 'btn-ghost place-danger', writing(place, ui, 'link') ? HELD : '');
    const again = act('introduce', place, tr('peer.link.again', 'Introduce again'));
    const actions = {
        check: act('check', place, tr('peer.code.check', 'Check the code')) + remove,
        removed: again + talk('btn-ghost'),
        lost: again + talk('btn-ghost') + remove,
        incomplete: again,
        unknown: talk('btn-ghost'),
    }[state] ?? talk('btn-default') + remove;
    const quiet = place.quiet?.note ? `<p class="consciousness-foot" data-i18n-skip>${escapeHtml(tr(place.quiet.note, place.quiet.note))}</p>` : '';
    const row = `<div class="ui-action-row">${actions}${resultHtml(`${place.id}:links`)}</div>`;
    const open = state === 'active' || state === 'paused' || state === 'silent';
    const parts = open ? [ruleHtml(place, ui, true), directionRowsHtml(place, ui), quiet, row] : [stateTextHtml(place, state), row, ruleHtml(place, ui, false)];
    return `<div class="own-body">${parts.join('')}</div>`;
}

/** One row of the block: its head is the one control that opens it. */
export function ownRowHtml(place, ui = { open: '', editing: '', drafts: {}, busy: '', pending: null }) {
    const state = linkStateOf(place), view = STATES[state], open = ui.open === place.id, name = peerName(place);
    const direction = state === 'active' ? directionWords(place) : '';
    const line = rowLine(place, state);
    const word = state === 'active' && place.quiet?.word ? tr(place.quiet.word, place.quiet.word) : tr(`peer.state.${state}`, view.word);
    const label = fmt('{name}: {state}', { name, state: [word, direction, line].filter(Boolean).join('; ') });
    // The app's disclosure (as a circle opens): its summary is the one control; one row is open at a time.
    return `<li><details class="ui-disclosure own-link" data-own-row="${escapeHtml(place.id)}" data-state="${state}"${open ? ' open' : ''}>
        <summary class="own-head" data-i18n-skip aria-label="${escapeHtml(label)}">
            <span class="own-pic">${faceHtml(place.accountId || place.link?.peerInstallId || place.id, 36)}</span>
            <span class="circle-head-main"><span class="own-name" data-i18n-authored>${escapeHtml(name)}</span><span class="own-line">${escapeHtml(line)}</span></span>
            <span class="own-state"><span class="ui-status" data-tone="${state === 'active' && place.quiet ? 'muted' : view.tone}">${escapeHtml(word)}</span>${direction ? `<span class="own-dir">${escapeHtml(direction)}</span>` : ''}</span>
        </summary>${open ? bodyHtml(place, ui, state) : ''}</details></li>`;
}

/** The block: its title, Introduce an Ouroboros, and a row for each Ouroboros of the owner's (or what it is for). */
export function ownBlockHtml(places, ui) {
    const rows = ownOrder((places || []).filter(isPeerPlace));
    const list = rows.length ? `<ul class="place-rows">${rows.map((place) => ownRowHtml(place, ui)).join('')}</ul>`
        : `<p class="consciousness-copy">${escapeHtml(tr('peer.none', 'None yet. Another Ouroboros of yours can write here and, if allowed, ask for work.'))}</p>`;
    return `<section class="ui-card consciousness-card" data-own-block aria-label="${escapeHtml(tr('peer.block', 'Your Ouroboroses'))}">
        <div class="consciousness-card-head"><h3 class="dashboard-card-title">${escapeHtml(tr('peer.block', 'Your Ouroboroses'))}</h3>
        <button type="button" class="btn btn-default btn-sm" data-own-act="introduce">${escapeHtml(tr('peer.introduce', 'Introduce an Ouroboros'))}</button></div>
        ${list}</section>`;
}

const rowKey = (id) => `[data-own-row="${CSS.escape(id)}"] > summary`;
/** What the owner can press or focus: a press or focus on one of these elsewhere is their own choice of place. */
const CONTROL = 'a[href], button, input, select, textarea, summary, [tabindex], [contenteditable="true"]';

/** The control a repaint gives focus back to, by its stable attributes ('' when it is not the block's). */
export function ownFocusKey(node) {
    const draft = node?.closest?.('[data-own-draft]');
    if (draft) return `[data-own-draft="${CSS.escape(draft.dataset.ownDraft)}"]`;
    const control = node?.closest?.('[data-own-switch]');
    if (control) return `[data-own-switch="${CSS.escape(control.dataset.ownSwitch)}"]`;
    const segment = node?.closest?.('[data-own-state] [data-effort-value]');
    const segmentRow = segment?.closest('[data-own-row]');
    if (segmentRow) return `[data-own-row="${CSS.escape(segmentRow.dataset.ownRow)}"] [data-own-state] [data-effort-value="${CSS.escape(segment.dataset.effortValue)}"]`;
    const row = node?.closest?.('[data-own-row] > summary');
    if (row) return rowKey(row.parentElement.dataset.ownRow);
    const button = node?.closest?.('[data-own-act]');
    if (!button) return '';
    return `[data-own-act="${button.dataset.ownAct}"]${button.dataset.place ? `[data-place="${CSS.escape(button.dataset.place)}"]` : ':not([data-place])'}`;
}

/**
 * The block's controller on the Circles page: `html(places)` is drawn inside the page's own paint and `painted()` runs
 * after it (the results, and focus a finished sheet or write sends to a row); `paint()` asks the page to repaint,
 * `refresh()` to read again. Its writes are the place's own and the link doors'; the sheets it opens are injected
 * (`openIntro`, `openCodeCheck`, `openConversation`: the place sheet) so a test can drive it without a server.
 */
export function createOwnOuroboroses(body, { places, paint, refresh, openIntro, openCodeCheck, openConversation, openWork = openConversation,
    askConfirm = openConfirmDialog, writes = { updatePlace, revokeLink } } = {}) {
    const ui = { open: '', editing: '', drafts: {}, busy: '', pending: null, results: {}, focusAfter: '', focusBack: '' };
    const find = (id) => (places() || []).find((place) => place.id === id) || null;
    const say = (key, result) => { ui.results[key] = { ...result, at: Date.now() }; };
    // A sheet gives focus back at close by name, for the block repaints meanwhile: where a finished check or removal
    // sends it (the row), else the control that opened it, else its row.
    const current = (selector, id) => () => {
        const after = ui.focusAfter && body.querySelector(ui.focusAfter);
        if (after) { ui.focusAfter = ''; return after; }
        return (selector && body.querySelector(selector)) || (id ? body.querySelector(rowKey(id)) : null);
    };

    // A held control is disabled, so the repaint that holds it drops its focus. The write gives focus back once the place
    // is read again: to where it sends it (`focusAfter`), else to the control that had it; never when the owner pressed
    // or focused another control meanwhile, or left the page (DESIGN principle 6: focus stays where the reader left it).
    let hold = null;
    function holdFocus() {
        const doc = globalThis.document, key = ownFocusKey(doc?.activeElement);
        const mine = { key, moved: false };
        const elsewhere = (event) => {
            const target = event.target?.closest?.(CONTROL);
            if (target && ownFocusKey(target) !== key) mine.moved = true;
        };
        doc?.addEventListener?.('focusin', elsewhere, true);
        doc?.addEventListener?.('pointerdown', elsewhere, true);
        mine.release = () => { doc?.removeEventListener?.('focusin', elsewhere, true); doc?.removeEventListener?.('pointerdown', elsewhere, true); };
        return mine;
    }
    const shown = () => body.isConnected !== false && !(typeof body.getClientRects === 'function' && !body.getClientRects().length);

    // One write at a time; its control holds the value it writes until the place is read again.
    async function write(place, field, value, slot, run, done) {
        if (ui.busy) return;
        hold = holdFocus();
        ui.busy = place.id; ui.pending = { field, value }; delete ui.results[slot];
        paint();
        try {
            await run();
            say(slot, { text: done, tone: 'ok', ends: true });
        } catch (error) {
            say(slot, error?.status === 409 ? { text: tr('places.conflict', 'Changed elsewhere; your text is kept.'), tone: 'warn' } : errorWords(error));
        }
        try {
            await refresh();
        } finally {
            const mine = hold; hold = null; mine?.release();
            ui.busy = ''; ui.pending = null;
            // Where this write gives focus back, for the next paint alone (`painted`); nowhere once the owner moved on.
            ui.focusBack = mine && !mine.moved && shown() ? ui.focusAfter || mine.key : '';
            ui.focusAfter = '';
            paint();
        }
    }

    async function remove(place, opener) {
        const back = current(ownFocusKey(opener), place.id);
        const ok = await askConfirm({ title: tr('peer.remove.confirm.title', 'Remove this link?'),
            body: tr('peer.remove.confirm.body', 'Nothing goes over this link after removal. The other side still lists it; remove it in Circles there too.'),
            confirmLabel: tr('peer.link.remove', 'Remove link'), danger: true });
        if (!ok) { back()?.focus({ preventScroll: true }); return; }
        ui.focusAfter = rowKey(place.id);
        await write(place, 'link', true, `${place.id}:links`, async () => {
            const outcome = await writes.revokeLink(place.link.linkId);
            if (!outcome?.revokedAt) throw new Error(tr('peer.link.remove.unconfirmed', 'Removal was not confirmed.'));
        }, tr('peer.link.removed', 'Link removed.'));
    }

    const editKey = (id) => `[data-own-act="edit-rule"][data-place="${CSS.escape(id)}"]`;
    const closeEditor = (id) => { ui.editing = ''; delete ui.drafts[id]; ui.focusAfter = editKey(id); };
    const ruleBase = (place) => {
        const note = ruleNote(place);
        return { noteId: note?.id ?? '', text: note?.text ?? null, revision: place.revision };
    };
    function saveRule(place) {
        const id = place.id, draft = ui.drafts[id];
        if (ui.editing !== id || !draft || ui.busy) return;
        const base = draft.base, submitted = draft.value, text = submitted.trim(), slot = `${id}:rule`;
        if (!Number.isInteger(base.revision)) {
            say(slot, { text: tr('peer.rule.revision.unread', 'The rule’s saved version could not be read.'), tone: 'warn' }); paint(); return;
        }
        if (text === (base.text ?? '')) { closeEditor(id); paint(); return; }
        const change = { ...(base.noteId ? { remove_note: base.noteId } : {}), ...(text ? { add_note: text } : {}),
            expected_revision: base.revision };
        void write(place, 'rule', text, slot, async () => {
            let saved;
            try {
                saved = await writes.updatePlace(id, change);
            } catch (error) {
                const current = error?.status === 409 && error.body?.place;
                if (ui.drafts[id] === draft && current && Number.isInteger(current.revision)) {
                    const note = [...(current.notes || [])].reverse().find((item) => ['owner', 'migration'].includes(item.author));
                    draft.base = { noteId: note?.note_id ?? '', text: note?.text ?? null, revision: current.revision };
                    draft.conflict = true;
                }
                throw error;
            }
            if (ui.drafts[id] !== draft) return;
            if (ui.editing === id && draft.value === submitted) closeEditor(id);
            else if (saved && Number.isInteger(saved.revision)) { draft.base = ruleBase(saved); draft.conflict = false; }
        }, tr('peer.rule.saved', 'Saved.'));
    }

    const actions = {
        introduce: (place, button) => openIntro({ returnFocus: current(ownFocusKey(button), place?.id), again: place }),
        check: (place, button) => openCodeCheck(place, { returnFocus: current(ownFocusKey(button), place.id) }),
        conversation: (place, button) => openConversation(place, { returnFocus: current(ownFocusKey(button), place.id) }),
        'open-work': (place, button) => openWork(place, { returnFocus: current(ownFocusKey(button), place.id) }),
        remove: (place, button) => remove(place, button),
        'edit-rule': (place) => {
            ui.editing = place.id;
            const base = ruleBase(place);
            ui.drafts[place.id] ??= { base, value: base.text ?? '' }; delete ui.results[`${place.id}:rule`]; paint();
            const field = body.querySelector(`[data-own-draft="${CSS.escape(place.id)}"]`);
            field?.focus({ preventScroll: true }); field?.setSelectionRange?.(field.value.length, field.value.length);
        },
        'cancel-rule': (place) => { closeEditor(place.id); paint(); },
        'save-rule': (place) => saveRule(place),
    };

    const onClick = (event) => {
        const button = event.target.closest?.('[data-own-act]');
        if (button && body.contains(button) && !button.disabled) {
            const place = button.dataset.place ? find(button.dataset.place) : null;
            if (button.dataset.place && !place) return;
            void actions[button.dataset.ownAct]?.(place, button);
            return;
        }
        const segment = event.target.closest?.('[data-own-state] [data-effort-value]');
        if (!segment || !body.contains(segment) || segment.disabled || ui.busy) return;
        const place = find(segment.closest('[data-own-row]')?.dataset.ownRow);
        const state = segment.dataset.effortValue;
        if (place && state !== place.state) {
            void write(place, 'state', state, `${place.id}:state`, () => writes.updatePlace(place.id, { state }), {
                paused: tr('peer.link.paused.done', 'Link paused.'), silent: tr('peer.link.silent.done', 'Link silent.'),
            }[state] || tr('peer.link.active.done', 'Link active.'));
        }
    };
    const onChange = (event) => {
        const control = event.target.closest?.('[data-own-switch]');
        if (!control || !body.contains(control)) return;
        const place = find(control.dataset.ownSwitch);
        if (!place?.peer || !Number.isInteger(place.revision) || ui.busy) {  // pressed while another write is out: the switch goes back to what is stored
            control.checked = place ? place.peer?.mayStart === true : !control.checked;
            control.setAttribute('aria-checked', String(control.checked));
            return;
        }
        // The ack names the choice saved; what the letter door does with it is the row's words after the next read.
        const mayStart = control.checked;
        void write(place, 'mayStart', mayStart, `${place.id}:from`,
            () => writes.updatePlace(place.id, { peer: { mine: place.peer.mine, may_start: mayStart }, expected_revision: place.revision }), tr('peer.rule.saved', 'Saved.'));
    };
    const onInput = (event) => {
        const field = event.target.closest?.('[data-own-draft]');
        if (!field) return;
        const id = field.dataset.ownDraft;
        if (ui.drafts[id]) ui.drafts[id].value = field.value;
        if (ui.results[`${id}:rule`]) { delete ui.results[`${id}:rule`]; setResult(body.querySelector(`[data-own-result="${CSS.escape(`${id}:rule`)}"]`), {}); }
    };
    // One row open at a time: opening one closes the other and draws its body; a fold inserted open echoes its own toggle.
    const onToggle = (event) => {
        const fold = event.target.closest?.('[data-own-row]');
        if (!fold || !body.contains(fold)) return;
        const id = fold.dataset.ownRow;
        if (fold.open && ui.open !== id) { ui.open = id; ui.editing = ''; ui.results = {}; paint(); } else if (!fold.open && ui.open === id) { ui.open = ''; ui.editing = ''; }
    };
    body.addEventListener('toggle', onToggle, true);
    body.addEventListener('click', onClick);
    body.addEventListener('change', onChange);
    body.addEventListener('input', onInput);
    return {
        html: (list) => ownBlockHtml(list, ui),
        /** After each paint of the page: the results the writes keep (an acknowledgement ends by itself), then focus
         *  where a finished sheet or write sent it, unless the owner has put it somewhere since. A write's return is
         *  this paint's alone, never a later one's. */
        painted() {
            const now = Date.now();
            for (const [key, result] of Object.entries(ui.results)) {
                if (result.ends && now - result.at >= SAVE_ACK_MS) { delete ui.results[key]; continue; }
                setResult(body.querySelector(`[data-own-result="${CSS.escape(key)}"]`), result);
            }
            const back = ui.focusBack; ui.focusBack = '';
            const target = (back || ui.focusAfter) && body.querySelector(back || ui.focusAfter);
            if (!target) return;
            if (!back) ui.focusAfter = '';
            const active = document.activeElement;
            if (!active || active === document.body) target.focus({ preventScroll: true });
        },
        /** After an introduction or its check: its row opens, and focus comes back to it. */
        reveal(placeId) { if (placeId) { ui.open = placeId; ui.results = {}; ui.focusAfter = rowKey(placeId); } },
        dispose() {
            hold?.release(); hold = null; ui.focusBack = '';
            body.removeEventListener('toggle', onToggle, true);
            body.removeEventListener('click', onClick);
            body.removeEventListener('change', onChange);
            body.removeEventListener('input', onInput);
        },
    };
}

/** The circles' own places: every place but the owner's Ouroboroses, which have their block. */
export const circlePlaces = (places) => (places || []).filter((place) => !isPeerPlace(place));
