// A letter at its sender (DESIGN "Replies", the note between works): the card the room of the work that wrote it shows,
// dressed as the app's relay note: the address «Ouroboros → Velm» with both faces and the time, the letter's words whole
// (never cut to the relay's three lines), one status line (the app's status mark and a word, then what follows from it)
// and one line of what comes next. Its facts are the host's typed `peer_letter_card` (ouroboros/place_projection.letter_card_fact)
// on the work's own progress line, read the same way live and on replay: the first fact of a letter mounts its card
// where that fact stands in the room, a later fact rewrites the card in place, and a done fact whose reply the receiver
// stated is this Ouroboros's (`result_available`, never the words alone) mounts that answer once,
// as the note «Velm → Ouroboros»; a question the receiver asked about it and this side's answer to that question are
// notes of their own the same way, once each («Question about the letter» and «Answer to the question»);
// a fact older than the one shown never repaints the card. Its facts come only from its work's own reads, so once that
// work has ended (its card settled, `endLetters`) a letter still open says so instead of promising an answer. The card
// derives nothing from tool text and has no button: a letter stops with its work's own Stop, and nothing here sends a
// letter again.
/**
 * @typedef {Object} LetterCardFact
 * @property {string} letter_id
 * @property {string} to                the receiving Ouroboros's name here (its place's label)
 * @property {string=} to_seed          its install id: the seed of its face
 * @property {string|null} text         the letter as recorded at its start (null: too long to carry)
 * @property {string=} at               when it was written (ISO), on the first fact
 * @property {'accepted_running'|'needs_input'|'waiting_owner'|'done'|'failed'|'cancelled'|'refused'|'not_sent'|'unconfirmed'|'unknown'} state
 * @property {'queued'|'running'=} phase
 * @property {string=} refusal_code     the receiver's typed refusal, or why nothing left here (not_sent)
 * @property {number|null=} spent_usd   what is known of its cost at the receiver (null: unknown, never 0)
 * @property {boolean=} spent_final     the receiver's cost is final; otherwise spent_usd is so far, never the total
 * @property {'given'|'none'|'held'|'unknown'=} reply  done: a reply for this Ouroboros by the receiver's explicit word
 *                                      (`held`: the receiver holds one now, its code in `held`)
 * @property {string=} held             a held reply's code (`rules_changed`: it does not come by itself)
 * @property {number|null=} ceiling_usd the letter's ceiling
 * @property {string=} error            a failure's short reason
 * @property {string|null=} answer      a given reply, whole (null: too long to carry; the work holds it)
 * @property {{question_id: string, text: string|null, held?: string}=} question  the receiver's question a read names
 *                                      (`held`: the code it holds its words under now, no words)
 * @property {{question_id: string, text: string|null}=} answered  this side's answer delivered to that question
 */
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { fmt, tr } from './i18n.js';
import { formatMsgTime, formatUsdKnown } from './utils.js';
import { apiFetch, taskLetterWordsUrl } from './api_client.js';
import { stampHistoryNode } from './chat_history_replay.js';
import { faceHtml } from './ouroboros_face.js';
import { statusMarkHtml } from './status_mark.js';
import { ARROW } from './chat_relay_deck.js';
import { HERE } from './own_ouroboroses.js';

/** `$5`, `$1.20`; '' for an unknown amount. */
export function usd(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '';
    return formatUsdKnown(value);
}

/** Where a refusal is put right, in the sender's words (its reason, then the next step at the receiver). */
export function refusalWords(code, to) {
    const names = { to, here: HERE };
    if (code === 'link_paused') return fmt('The link is paused at {to}, or its code is not checked there yet. Turn it on in {to}\'s Circles.', names);
    if (code === 'peer_may_not_start') return fmt('{to} does not take requests from {here}. Turn them on in {to}\'s Circles.', names);
    if (code === 'no_budget') return fmt('{to} has no budget left. Raise it in {to}\'s Costs.', names);
    if (['protocol_unsupported', 'unexpected_fields', 'invalid_letter', 'attachments_unsupported'].includes(code)) {
        return fmt('{to}\'s version cannot read this letter. Update {to}.', names);
    }
    if (code === 'install_mismatch' || code === 'link_unknown') return fmt('{to} does not recognise this link any more. Introduce {to} again in Circles.', names);
    return fmt('{to} refused the letter ({code}).', { to, code: code || 'no reason given' });
}

/** What a letter spent at the receiver: the total only when that side says it is final, else what is known so far. */
function spentWords(fact, limit, to) {
    const spent = usd(fact.spent_usd);
    if (!spent) return tr('peer.letter.spend_unknown', 'spend unknown');
    if (!fact.spent_final) return fmt('spent {spent} so far from {to}\'s budget', { spent, to });
    return limit ? fmt('spent {spent} of {limit} from {to}\'s budget', { spent, limit, to }) : fmt('spent {spent} from {to}\'s budget', { spent, to });
}

const OPEN = new Set(['accepted_running', 'needs_input', 'waiting_owner', 'unknown']);

/** The card's one status line and one next line for a letter's state; `ended`: the work that wrote it has ended, so
 *  an open letter is no longer read here and its last state is only what was last read; `answered`: this side's answer
 *  to the very question the read names was delivered (its receipt mounted the note), so the card no longer asks. */
export function letterCardView(fact, { ended = false, answered = false } = {}) {
    const to = fact.to || 'Ouroboros', limit = usd(fact.ceiling_usd);
    const budget = limit ? fmt('limit {limit} of {to}\'s budget', { limit, to }) : '';
    const going = tr('peer.letter.next', 'The answer will come here.');
    if (ended && OPEN.has(fact.state)) {
        const last = letterCardView(fact).word;
        return { mark: { shape: 'dot', hue: 'neutral' }, word: tr('peer.letter.ended', 'This work ended'),
            rest: fmt('last read: {state}', { state: last }), next: tr('peer.letter.ended.next', 'No answer will come here.') };
    }
    switch (fact.state) {
    case 'accepted_running':
        return fact.phase === 'queued'
            ? { mark: { shape: 'still', hue: 'ink' }, word: fmt('Queued at {to}', { to }), rest: budget, next: going }
            : { mark: { shape: 'moving', hue: 'ink' }, word: fmt('{to} is working on it', { to }), rest: budget, next: going };
    case 'needs_input':
        if (answered) return { mark: { shape: 'moving', hue: 'ink' }, word: fmt('Answered {to}\'s question', { to }), rest: budget, next: going };
        // Words the receiver holds now: nothing moves here until they come.
        if (fact.question?.held) return { mark: { shape: 'still', hue: 'ink' }, word: fmt('{to} asked about the letter', { to }), rest: budget,
            next: fact.question.held === 'rules_changed'
                ? fmt('{to}\'s rules for this link changed since it asked, so its words are not shown here.', { to })
                : fmt('{to} has not released the question\'s words yet, so {here} cannot answer it.', { to, here: HERE }) };
        return { mark: { shape: 'moving', hue: 'ink' }, word: fmt('{to} asked about the letter', { to }), rest: budget,
            next: (fact.question?.delivery_state === 'reserved'
                ? fmt('{to} reserved an answer to it; its delivery is not confirmed yet. ', { to }) : '')
                + fmt('{here} answers it in this work; the answer goes to {to} as a letter.', { to, here: HERE }) };
    case 'waiting_owner':
        return { mark: { shape: 'dot', hue: 'owner' }, word: fmt('Waiting for you at {to}', { to }), rest: '',
            next: fmt('{to} asks you about its own rights or budget, or its work there is paused. Answer in {to}\'s chat: such questions go only to you, not to {here}.', { to, here: HERE }) };
    case 'unconfirmed':
        return { mark: { shape: 'ring', hue: 'ink' }, word: tr('peer.letter.unconfirmed', 'Not confirmed'), rest: '',
            next: fmt('{to} did not confirm that it got the letter. Nothing is sent again on its own.', { to }) };
    case 'refused':
        return { mark: { shape: 'dot', hue: 'failed' }, word: tr('peer.letter.refused', 'Refused'), rest: tr('peer.letter.refused.rest', 'no work was started'),
            next: refusalWords(fact.refusal_code, to) };
    case 'not_sent':  // refused here, after it was written and before it left: the other side never saw it
        return { mark: { shape: 'dot', hue: 'failed' }, word: tr('peer.letter.not_sent', 'Not sent'), rest: tr('peer.letter.not_sent.rest', 'nothing left this computer'),
            next: fact.refusal_code === 'place_paused' ? fmt('The link with {to} is paused here. Turn it on in Circles.', { to })
                : fact.refusal_code === 'place_silent' ? fmt('{to} is Silent here: nothing is sent there while Silent.', { to })
                    : fmt('Nothing was sent ({code}).', { code: fact.refusal_code || 'no reason given' }) };
    case 'done': {
        // A reply is shown only on the receiver's explicit word that it is this Ouroboros's; its words alone are none.
        const spent = spentWords(fact, limit, to);
        if (fact.reply === 'given') {
            return { mark: { shape: 'dot', hue: 'ok' }, word: tr('peer.letter.done', 'Answered'), rest: spent,
                next: fact.answer === null ? tr('peer.letter.answer.long', 'The answer is too long for this card; this work has it whole.')
                    : fact.answer === '' ? tr('peer.letter.answer.empty', 'The reply is empty.') : '' };
        }
        const finished = { mark: { shape: 'dot', hue: 'neutral' }, word: tr('peer.letter.finished', 'Finished') };
        if (fact.reply === 'held') return { ...finished, rest: spent, next: fact.held === 'rules_changed'
            ? fmt('{to}\'s rules for this link changed since, so its reply does not come here.', { to })
            : fmt('{to} has not released its reply yet.', { to }) };
        return fact.reply === 'none'
            ? { ...finished, rest: spent, next: tr('peer.letter.reply.none', 'Work finished without a reply for this Ouroboros.') }
            : { ...finished, rest: [tr('peer.letter.reply.unknown', 'reply unknown'), spent].join(' · '), next: '' };
    }
    case 'failed':
        return { mark: { shape: 'dot', hue: 'failed' }, word: fmt('Failed at {to}', { to }),
            rest: [fact.error || '', fact.error_chars ? tr('peer.letter.error.cut', 'reason cut to its first part') : ''].filter(Boolean).join(' · '), next: '' };
    case 'cancelled':
        return { mark: { shape: 'dot', hue: 'neutral' }, word: tr('peer.letter.cancelled', 'Cancelled'), rest: '', next: '' };
    default:
        return { mark: { shape: 'none', hue: 'ink' }, word: tr('peer.letter.unknown', 'State unknown'), rest: '', next: '' };
    }
}

/** A note's address, sender → receiver · time: the letter goes from here to `to`, its answer comes back. */
export function letterHeadHtml(fact, { answer = false, ts = '' } = {}) {
    const to = fact.to || 'Ouroboros';
    const here = `${faceHtml('', 20, { own: true })}<span>${escapeHtml(HERE)}</span>`;
    const there = `${faceHtml(fact.to_seed || to, 20)}<span data-i18n-authored>${escapeHtml(to)}</span>`;
    const time = formatMsgTime(ts);
    return `<div class="chat-relay-card-head"><span class="chat-relay-from peer-who">${answer ? there : here}</span>
        <span class="chat-relay-dest"><span class="chat-relay-arrow" aria-hidden="true">${ARROW}</span><span class="chat-relay-to peer-who">${answer ? here : there}</span></span>
        <span class="chat-relay-side"><time class="chat-relay-time"${time ? ` title="${escapeHtml(time.full)}"` : ''}>${escapeHtml(time?.short || '')}</time></span></div>`;
}

/** The status line and the next line of a letter's card. */
export function letterStatusHtml(fact, options = {}) {
    const view = letterCardView(fact, options), tone = view.mark.hue === 'failed' ? ' data-tone="error"' : view.mark.hue === 'owner' ? ' data-tone="owner"' : '';
    const stop = fact.stop ? tr(`peer.letter.stop.${fact.stop}`, { requested: 'stop requested', refused: 'stop refused', unverified: 'stop not verified' }[fact.stop]) : '';
    return `<p class="peer-letter-status">${statusMarkHtml('', view.mark)}<span class="peer-letter-word"${tone}>${escapeHtml(view.word)}</span>${view.rest
        ? `<span class="peer-letter-rest">${escapeHtml(view.rest)}</span>` : ''}${stop ? `<span class="peer-letter-rest">${escapeHtml(stop)}</span>` : ''}</p>${view.next ? `<p class="peer-letter-next">${escapeHtml(view.next)}</p>` : ''}`;
}

const label = (fact, options = {}) => {
    const view = letterCardView(fact, options);
    return fmt('Letter from {here} to {to}: {state}', { here: HERE, to: fact.to || 'Ouroboros', state: [view.word, view.rest].filter(Boolean).join(', ') });
};

/** One part of a letter in this room: its card (`letter`) or its answer (`answer`), when mounted. */
function partOf(feed, id, part) {
    return [...feed.querySelectorAll('.peer-letter-card')].find((node) => node.dataset.letterId === id && node.dataset.letterPart === part) || null;
}

/** A bubble the room drew for a letter's part, dressed as a relay note: the face, the address and the words whole. */
// The notes beside the card: which way each goes, the line naming it, and its accessible name.
const NOTES = {
    answer: { back: true, kind: '', name: 'Answer from {to} to {here}' },
    question: { back: true, kind: 'Question about the letter', name: 'Question from {to} to {here} about the letter' },
    answered: { back: false, kind: 'Answer to the question', name: 'Answer from {here} to {to} to its question' },
};

/** The words a question note shows while the receiver holds them (its question fact's `held` code). */
export const heldWords = (to, code = '') => (code === 'rules_changed'
    ? fmt('Not shown under {to}\'s current rules for this link.', { to: to || 'Ouroboros' })
    : fmt('{to} has not released these words yet.', { to: to || 'Ouroboros' }));

function dress(bubble, fact, part, words, ts, held = '') {
    bubble.classList.add('chat-relay-note', 'peer-letter-card');
    Object.assign(bubble.dataset, { kind: 'main', raw: '0', letterId: fact.letter_id, letterPart: part });
    if (held) { bubble.dataset.held = '1'; bubble.dataset.heldCode = held; }
    bubble.setAttribute('role', 'group');
    const note = NOTES[part.split(':')[0]], names = { to: fact.to || 'Ouroboros', here: HERE };
    const long = part === 'letter' ? tr('peer.letter.text.long', 'The letter is too long for this card; this work has it whole.')
        : tr('peer.letter.words.long', 'These words are too long for this card; this work has them whole.');
    const shown = held ? heldWords(fact.to, held) : words === null ? long
        : words === '' && part === 'answer' ? tr('peer.letter.answer.empty', 'The reply is empty.') : words;
    const source = !held && words === null && (part === 'letter' || part === 'answer') ? part : '';
    bubble.insertAdjacentHTML('afterbegin', `<span class="chat-relay-face" aria-hidden="true"></span>${letterHeadHtml(fact, { answer: Boolean(note?.back), ts })}
        ${note?.kind ? `<p class="peer-letter-kind">${escapeHtml(tr(`peer.letter.${part.split(':')[0]}`, note.kind))}</p>` : ''}
        <div class="chat-relay-excerpt"><p class="peer-letter-text"${held || words === null ? '' : ' data-i18n-authored'}>${escapeHtml(shown)}</p>${source
            ? `<button type="button" class="btn btn-default btn-sm peer-letter-read" data-letter-read="${source}">${escapeHtml(source === 'letter'
                ? tr('peer.letter.read_letter', 'Read letter') : tr('peer.letter.read_reply', 'Read full reply'))}</button>` : ''}</div>
        ${part === 'letter' ? '<div class="peer-letter-state"></div>' : ''}`);
    if (note) bubble.setAttribute('aria-label', fmt(note.name, names));
    if (source) wireRead(bubble, fact, source);
}

/** The whole words of a part the card carries none of, read on demand from this work's own records (never another
 *  installation): the exact text in place of the long line, or the typed refusal in place of the button. */
function wireRead(bubble, fact, part) {
    const button = bubble.querySelector('[data-letter-read]');
    button?.addEventListener('click', async () => {
        const url = taskLetterWordsUrl(bubble.dataset.letterTask || '', fact.letter_id, part);
        const text = bubble.querySelector('.peer-letter-text');
        if (!url) { button.remove(); return; }
        button.disabled = true;
        try {
            const response = await apiFetch(url);
            if (!response.ok) throw new Error((await response.json().catch(() => ({})))?.reason_code || 'letter_words_unavailable');
            const words = await response.text();
            if (text) { text.textContent = words; text.setAttribute('data-i18n-authored', ''); }
            button.remove();
        } catch (err) {
            button.disabled = false;
            if (text) text.textContent = fmt('The whole words could not be read now ({code}).', { code: String(err?.message || err) });
        }
    });
}

/** A held question's note once a later fact carries its words: the words, whole, in place of the held line. */
function release(note, words) {
    delete note.dataset.held;
    const text = note.querySelector('.peer-letter-text');
    if (!text) return;
    text.textContent = words;
    text.setAttribute('data-i18n-authored', '');
}

/** A fact of a mounted card: its state lines and name, written only when they change and never by a fact older than
 *  the one shown (an earlier page loading above the card replays its earlier states). */
/** Whether this side's answer to the question a needs_input fact names was delivered: the fact is that answer's own
 *  (published only on a delivered answer, observe_letter_answer), or the answer's note is already in the room. */
function answeredHere(feed, fact) {
    if (fact.state !== 'needs_input') return false;
    const asked = fact.question?.question_id, sent = fact.answered?.question_id;
    if (sent) return !asked || asked === sent;
    return Boolean(asked && feed && partOf(feed, String(fact.letter_id), `answered:${asked}`));
}

function paint(card, fact, at, feed) {
    const task = card.dataset.letterTask || '';
    const ended = Boolean(task) && globalThis.document?.querySelector?.(`.chat-live-card[data-task-id="${CSS.escape(task)}"]`)
        ?.dataset.finished === '1';
    const options = { ended, answered: answeredHere(feed, fact) };
    const html = letterStatusHtml(fact, options), host = card.querySelector('.peer-letter-state');
    if (!host || (at && card.dataset.letterAt && at < card.dataset.letterAt)) return false;
    if (at) card.dataset.letterAt = at;
    card.letterFact = fact;  // the fact shown: its work's end repaints it (endLetters)
    if (card.dataset.letterState === `${fact.state}:${html}`) return false;
    host.innerHTML = html;
    card.dataset.letterState = `${fact.state}:${html}`;
    card.setAttribute('aria-label', label(fact, options));
    return true;
}

/** The work `taskId` has ended (its card settled): each letter it wrote that is still open says so. */
export function endLetters(feed, taskId) {
    if (!feed || !taskId) return;
    for (const card of feed.querySelectorAll('.peer-letter-card[data-letter-part="letter"]')) {
        if (card.dataset.letterTask === String(taskId) && card.letterFact) paint(card, card.letterFact, '', feed);
    }
}

/**
 * One row of a room as a letter's, live or replayed: `row.peer_letter_card` is the fact; `add` is the room's own
 * `addMessage(text, role, markdown, ts, isProgress, opts)`, so a part takes its place, identity and reading anchors as
 * any message does. Returns whether the row was a letter's (the room then draws nothing else for it).
 */
export function letterRow(row, feed, add) {
    // A card fact always states its state; anything else under this name is no card and is drawn as its row.
    const fact = row?.peer_letter_card;
    if (!fact || typeof fact !== 'object' || Array.isArray(fact) || typeof fact.state !== 'string') return false;
    const id = String(fact.letter_id || '');
    if (!id || !feed) return true;
    const retain = (bubble) => {
        // A live part is adopted by its first saved page. Later state replays must not move it to another page.
        if (bubble && !bubble.dataset.historyId) stampHistoryNode(bubble, row.history_id, row.history_position);
        return bubble;
    };
    const mount = (part, words, ts, held = '') => {
        const peer = NOTES[part.split(':')[0]]?.back;
        const bubble = add(String(words ?? ''), 'assistant', false, ts || row.ts || null, false, { systemType: 'peer_letter',
            taskId: row.task_id || '', reply: null, letterPartKey: `${id}:${part}`, senderLabel: peer ? (fact.to || 'Ouroboros') : HERE,
            ...(part === 'letter' && row.history_id ? { historyId: row.history_id, historyPosition: row.history_position } : {}) });
        if (!bubble?.classList) return retain(partOf(feed, id, part));
        dress(bubble, fact, part, words, ts || row.ts, held);
        if (row.task_id) bubble.dataset.letterTask = String(row.task_id);
        return retain(bubble);
    };
    const card = retain(partOf(feed, id, 'letter')) || mount('letter', fact.text, fact.at);
    if (card) paint(card, fact, String(row.ts || ''), feed);
    for (const said of ['question', 'answered']) {  // a question and its answer: a note each, once
        const words = fact[said], part = words?.question_id ? `${said}:${words.question_id}` : '';
        if (!part) continue;
        const note = retain(partOf(feed, id, part)), held = typeof words.text !== 'string' && words.held ? String(words.held) : '';
        if (!note) mount(part, words.text, undefined, held);
        else if (note.dataset.held === '1') {
            if (!held) release(note, typeof words.text === 'string' ? words.text
                : tr('peer.letter.words.long', 'These words are too long for this card; this work has them whole.'));
            else if (note.dataset.heldCode !== held) {
                note.dataset.heldCode = held;
                const text = note.querySelector('.peer-letter-text');
                if (text) text.textContent = heldWords(fact.to, held);
            }
        }
    }
    if (fact.state === 'done' && fact.reply === 'given' && !retain(partOf(feed, id, 'answer'))) mount('answer', fact.answer);
    return true;
}
