// Where a work card's work came from (DESIGN §5 "Conversation activity block"): the schedule that started it and,
// for work a place called, the host's `place_origin`: the place, the checked kinds that matched and whose event it
// was, as the admitted focus recorded them and as the works that focus starts inherit them. Nothing here infers an
// origin, and an absent fact adds no word. A root card and a Main portal say it in one meta line under their name,
// without the place while the name already is it; a nested child's origin is its parent's. Details keep it whole.
import { tr } from './i18n.js';
import { escapeHtmlText as escapeHtml } from './utils.js';
import { META_PART_SEPARATOR } from './harness_presentation.js';
import { placeLabelOnce } from './chat_media.js';
import { capitalized, heardFactWord, listWords, sameWords, wakeFactWord } from './consciousness_words.js';

/** @typedef {import('./place_contract_types.js').PlaceOrigin} PlaceOrigin */

const text = (value) => (typeof value === 'string' ? value.trim() : '');
// The card's meta separator as a part's own tail: the dot holds on to the part's last word, so a wrapped line may
// end with a dot and never starts with one; a copied line reads as it looks.
const SEP = '\u00a0\u00b7 ';

/** The host's `place_origin` as the card reads it, or null when it names no place. A flag counts only when true; an
 *  actor with neither a name nor an id is not identified, never a person and never "not a bot". */
function placeOriginOf(raw) {
    if (!raw || typeof raw !== 'object' || !text(raw.place_id)) return null;
    const actor = raw.actor && typeof raw.actor === 'object' ? raw.actor : {};
    const wakes = (Array.isArray(raw.wakes) ? raw.wakes : []).map(text).filter(Boolean);
    const words = raw.labels && typeof raw.labels === 'object' ? raw.labels : {};
    return {
        placeId: text(raw.place_id), label: placeLabelOnce(text(raw.place_label)), provider: text(raw.provider),
        wakes, labels: Object.fromEntries(wakes.filter((fact) => text(words[fact])).map((fact) => [fact, text(words[fact])])),
        routed: raw.routed === true,
        actor: { id: text(actor.id), label: text(actor.label), owner: actor.owner === true, bot: actor.bot === true },
    };
}

/** The line's parts for one card: the place unless `title` already is it, what called the work in the words a heard
 *  message carries (`every message` names no kind, so it adds none), and whose event it was. `authored` marks words
 *  someone outside wrote (a place's label, a kind's declared words, an account's name), which are never translated;
 *  the host's own words (the reason, `you`) arrive translated. */
export function placeOriginParts(origin, title = '') {
    if (!origin) return [];
    const said = origin.wakes.filter((fact) => heardFactWord(fact, origin.labels));
    const why = listWords(said.map((fact) => heardFactWord(fact, origin.labels)))
        || (origin.routed && !origin.wakes.length ? tr('task.origin.routed', 'ongoing conversation') : '');
    const { owner, label } = origin.actor;
    return [{ text: sameWords(placeLabelOnce(title), origin.label) ? '' : origin.label, authored: true },
        { text: why, authored: said.some((fact) => origin.labels[fact]) },
        { text: owner ? tr('places.account.you', 'you') : label, authored: !owner }].filter((part) => part.text);
}

/** The same origin whole, as rows of the card's Details: the place (its service's id beside a label that does not
 *  name it), each matched kind by its name in the place's "What calls it here" (its connector's own words kept as
 *  written), and the event's author with what the admission recorded about them. A gap is said as a gap. Authored
 *  words keep their boundary exactly as the DOM serializes it (`data-i18n-authored=""`), so an unchanged list equals
 *  its own markup and is never rebuilt under a reader's selection. */
function placeOriginRows(origin) {
    const authored = (value) => `<span data-i18n-authored="">${escapeHtml(value)}</span>`;
    const joined = (parts) => parts.filter(Boolean).join(META_PART_SEPARATOR);
    const place = origin.label || origin.placeId;
    const named = place.toLowerCase().includes(origin.provider.toLowerCase());
    const kinds = listWords(origin.wakes.map((fact) => wakeFactWord(fact, origin.labels)));
    const called = (origin.labels[origin.wakes[0]] ? kinds : capitalized(kinds))
        || (origin.routed ? tr('task.origin.routed.detail', 'An ongoing conversation of its work; no checked kind matched') : '');
    const { actor } = origin;
    const who = actor.label || actor.id;
    return [
        [tr('task.fact.place', 'Place'), joined([authored(place), named ? '' : authored(origin.provider)])],
        [tr('task.fact.called_by', 'What called it'), origin.wakes.some((fact) => origin.labels[fact]) ? authored(called) : escapeHtml(called)],
        [tr('task.fact.from', 'From'), who ? joined([authored(who), actor.owner ? escapeHtml(tr('places.account.you', 'you')) : '',
            actor.bot ? escapeHtml(tr('task.origin.bot', 'bot')) : '', actor.label && actor.id && actor.id !== actor.label ? authored(actor.id) : ''])
            : escapeHtml(tr('places.author_unknown', 'Author not recorded'))],
    ];
}

/** The card's origin line, written only on change; its node is made when there is first something to say and
 *  stands under the card's name, before its episodes and its latest line. Each part is one span that moves to the
 *  next line whole, as the card's meta line does, and an authored part carries its own `data-i18n-authored`. True
 *  when the DOM changed. */
export function paintOrigin(record) {
    const anchor = record?.episodesEl;
    if (!anchor?.ownerDocument || !anchor.parentNode) return false;
    const parts = record.isSubagent ? [] : placeOriginParts(record.placeOrigin, record.titleEl?.textContent || '');
    const words = parts.map((part) => part.text).join(SEP);
    let el = record.originEl;
    if (!el && !words) return false;
    if (!el) {
        el = record.originEl = anchor.ownerDocument.createElement('div');
        el.className = 'chat-live-origin';
        el.dataset.liveOrigin = '';
        anchor.parentNode.insertBefore(el, anchor);
    }
    if (record.originWords === words && el.hidden === !words) return false;
    record.originWords = words;
    el.replaceChildren(...parts.map((part, index) => {
        const span = anchor.ownerDocument.createElement('span');
        span.textContent = index < parts.length - 1 ? part.text + SEP : part.text;
        if (part.authored) span.dataset.i18nAuthored = '';
        return span;
    }));
    el.hidden = !words;
    return true;
}

/** Fold a carrier's origin facts into its card's record: a schedule's name and the host's `place_origin` (a row, a
 *  detail or a terminal projection as `detail`, a frame's projected one as `place`). The card's Details rows
 *  (`record.originRows`, read by the meta renderer) follow the record, and its line meets the card's name here as
 *  well: a name another writer gave it since (a start row on a portal) never leaves the place said twice. True when
 *  a fact or the line changed. */
export function noteOrigin(record, detail, place = detail?.place_origin) {
    if (!record) return false;
    const named = Boolean(detail?.schedule_name) && record.scheduleName !== detail.schedule_name;
    if (named) record.scheduleName = detail.schedule_name;
    const origin = record.isSubagent ? null : placeOriginOf(place);
    const placed = Boolean(origin) && JSON.stringify(origin) !== JSON.stringify(record.placeOrigin || null);
    if (placed) {
        record.placeOrigin = origin;
        record.originRows = placeOriginRows(origin);
    }
    const painted = Boolean(record.placeOrigin) && paintOrigin(record);
    return named || placed || painted;
}
