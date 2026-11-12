import { tr, fmt } from './i18n.js';
import { shortName } from './chat_model_routes.js';
import { taskPresentation } from './log_events.js';
import { taskCostMeta } from './chat_activity.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { formatUsd } from './storage_format.js';

/** The same account stamp on wake and delivery rows; account names remain authored. */
export function sendsAsHtml(sender) {
    if (!sender?.label) return '';
    return sender.owner ? escapeHtml(tr('places.identity.as_you', 'as you'))
        : fmt('as {label}', { label: `<span data-i18n-authored>${escapeHtml(sender.label)}</span>` });
}

// The owner's words for consciousness, places and safety (owner's final names, 21.10), in one place.
// Stored values never change: only these words do, so a relabel is one line here. Pure data and pure
// helpers over host facts; nothing here reads the network or the page.

export { AUTONOMY_LEVELS, autonomyOf } from './consciousness_autonomy.js';

/** Place state: stored active / silent / paused. `word` is the row's status word. */
export const PLACE_STATES = Object.freeze([
    { value: 'active', label: 'Active', word: 'active', note: 'Lives here: listens, wakes when its list says so, speaks and acts.' },
    { value: 'silent', label: 'Silent', word: 'silent', note: 'Hears everything here and changes nothing: no messages, reactions, edits or moderation.' },
    { value: 'paused', label: 'Paused', word: 'paused', note: 'Away from this place: no wakes, no actions. Still connected; reported events are saved.' },
]);

/** What Silent enforces, said by the host (verbatim, owner 21.10). */
export const SILENT_CARD = "Writes through this connector's declared tools and delivery queue are blocked. Scripts, browser actions and unintegrated code may still write.";

/** Safety levels, each kind its own row, named by what the Supervisor reads before it runs (owner, 11 Nov 2026:
 *  Off / Outgoing / + Commands); stored off / light / full. The Settings rows restate these words as data
 *  (settings_rows.js); a test holds the two together. */
export const SAFETY_LEVELS = Object.freeze([
    { value: 'off', label: 'Off', note: 'No model checks. Fixed rules still guard protected files.' },
    { value: 'light', label: 'Outgoing', note: 'Messages, posts, skills, connectors and starting work are read first.' },
    { value: 'full', label: '+ Commands', note: 'Shell commands and scripts are read first too. File edits are not.' },
]);

/** The typed facts of `places.WAKE_FACTS`: wake choices and observations about one message. */
const FACT_WORDS = Object.freeze({
    direct_message: { wake: 'direct messages', heard: 'addressed to it' },
    group_dm: { wake: 'group messages', heard: 'group message' },
    mention: { wake: 'mentions', heard: 'mentions it' },
    reply_to_me: { wake: 'replies', heard: 'reply to it' },
    my_thread: { wake: 'replies in its threads', heard: 'in its thread' },
    new_issue: { wake: 'new issues', heard: 'new issue' },
    new_pull_request: { wake: 'new pull requests', heard: 'new pull request' },
    review_requested: { wake: 'review requests addressed to me', heard: 'review requested' },
    assigned: { wake: 'assignments addressed to me', heard: 'assigned to it' },
    any_message: { wake: 'every message', heard: '' },
    from_bot: { wake: 'also wakes on bot messages', heard: 'from a bot' },
    legacy_addressed: { wake: 'messages addressed to it', heard: 'addressed to it' },
});
export const WAKE_FACT_WORDS = Object.freeze(Object.fromEntries(Object.entries(FACT_WORDS)
    .map(([fact, words]) => [fact, words.wake])));

/** Who added a wake item or a note. */
export const AUTHOR_WORDS = Object.freeze({
    connector: 'added by connector',
    owner: 'added by you',
    ouroboros: 'added by Ouroboros',
    migration: 'carried over',
});

/** A recent wake's safety facts, as chips: a warning worth reading takes the warn pair, a plain fact stays neutral. */
export const CHECK_CHIPS = Object.freeze({
    checked: { label: 'checked', tone: 'neutral' },
    warned: { label: 'warned', tone: 'warn' },
    sent_despite: { label: 'let through despite the check', tone: 'warn' },
    held: { label: 'held', tone: 'warn' },
    not_checked: { label: 'not checked', tone: 'neutral' },
});

export const placeStateOf = (value) => PLACE_STATES.find((state) => state.value === value) || null;
export const safetyLevelOf = (value) => SAFETY_LEVELS.find((level) => level.value === value) || null;

/** «a», «a and b», «a, b and c». */
export function listWords(words) {
    const list = words.filter(Boolean);
    if (list.length < 2) return list.join('');
    return fmt('{head} and {last}', { head: list.slice(0, -1).join(', '), last: list[list.length - 1] });
}

export const capitalized = (text) => {
    const value = String(text || '');
    return value ? value[0].toUpperCase() + value.slice(1) : value;
};

/** One wake fact in words; a declared label wins; an unknown kind keeps its spelling. */
export const wakeFactWord = (fact, labels = null) => labels?.[fact] || (WAKE_FACT_WORDS[fact]
    ? tr(`places.wake.${fact}`, WAKE_FACT_WORDS[fact]) : String(fact || '').replace(/_/g, ' '));

/** A heard fact names this message; the universal any_message adds no chip, even when labelled. */
export function heardFactWord(fact, labels = null) {
    const word = FACT_WORDS[fact]?.heard;
    return word === '' ? '' : labels?.[fact] || (word
        ? tr(`places.heard.${fact}`, word) : String(fact || '').replace(/_/g, ' '));
}

/** `{id: label}` for declared facts, including signals (`facts` entries that are objects). */
export const signalLabels = (facts) => Object.fromEntries((Array.isArray(facts) ? facts : [])
    .filter((item) => item && typeof item === 'object' && item.id && item.label).map((item) => [item.id, item.label]));

export const authorWord = (author) => AUTHOR_WORDS[author] || (author ? `added by ${author}` : '');

/** The row's status word for a stored state; an unknown state says so instead of guessing. */
export const placeStateWord = (state) => placeStateOf(state)?.word || tr('places.state.unknown', 'state unknown');

/** Who wrote a circle's words or a description, and who placed a place in its circle (the stored author). */
export const writtenByWord = (author) => (author === 'owner' || author === 'migration'
    ? tr('places.written.owner', 'written by you') : tr('places.written.ouroboros', 'written by Ouroboros'));
export const placedByWord = (author) => (author === 'owner' || author === 'migration'
    ? tr('places.placed.owner', 'placed here by you') : tr('places.placed.ouroboros', 'placed here by Ouroboros'));

const TITLE_SEP = ' · ';
/** Two labels that read the same, case and spacing aside: «FRONT/OFFICE» and «Front / Office». */
export const sameWords = (a, b) => String(a || '').toLowerCase().replace(/\s+/g, '') === String(b || '').toLowerCase().replace(/\s+/g, '');
/** The service a place is on, as its row and its sheet name it quietly. */
export const placeService = (place) => String(place.serviceLabel || place.provider || '');

// A model a place or a circle is given, as a row says it: the two aliases by their role's name, a route by the
// model's own short name (`z-ai/glm-5.3` reads `glm-5.3`).
const MODEL_ALIASES = Object.freeze({ main: ['places.model.main', 'Main'], light: ['places.model.light', 'Light'] });
export const isModelAlias = (value) => Object.hasOwn(MODEL_ALIASES, String(value || ''));
export const placeModelWord = (value) => (isModelAlias(value) ? tr(...MODEL_ALIASES[value]) : shortName({ model: String(value || '') }));
/** What an empty Model field of a place uses, from its `model_route` and its circle as last read (a circle's model
 *  edit does not revise its places' rows): `{circle, model}`, `circle` empty for the consciousness model and `model`
 *  empty when this read does not say which one that is. */
export function inheritedModel(place, circle = null) {
    const route = place.modelRoute;
    if (circle?.model) {
        // An alias is named by what it resolves to now, which only a place that inherits it carries.
        const resolved = isModelAlias(circle.model) && route?.layer === 'circle' && route.value ? route.value : circle.model;
        return { circle: circle.name, model: placeModelWord(resolved) };
    }
    return { circle: '', model: route?.layer === 'consciousness' && route.value ? placeModelWord(route.value) : '' };
}
/** One title for a place's row and its sheet: its label without the service it is on and without its circle's own
 *  name; a whole-account place with nothing left takes the account's name from its connector's status; the service
 *  itself is the last resort. */
export function placeTitle(place, { circle = null, status = null } = {}) {
    const service = placeService(place);
    let parts = String(place.label || '').split(TITLE_SEP).map((part) => part.trim()).filter(Boolean);
    if (parts.length > 1 && sameWords(parts[0], service)) parts = parts.slice(1);
    if (circle && parts.length > 1 && sameWords(parts[parts.length - 1], circle.name)) parts = parts.slice(0, -1);
    if (circle && parts.length > 1 && sameWords(parts[0], circle.name)) parts = parts.slice(1);
    let title = parts.join(TITLE_SEP);
    if (!title || sameWords(title, service)) title = String(status?.workspace_name || status?.team_name || status?.account_label || '');
    return title || service;
}
/** The instance a place is bound to (its conversation, else its account): what tells two same-titled rows apart. */
export const placeInstance = (place) => String(place.conversationId || place.accountId || '');

// The identifier a service knows its people by, for «Add mine» (from the place's provider).
const ACCOUNT_ID_WORDS = Object.freeze({ slack: 'your member ID (U…)', telegram: 'your numeric user ID' });
const EMAIL_PROVIDERS = new Set(['email', 'e-mail', 'mail', 'gmail', 'imap', 'smtp']);

/** «your member ID (U…)» on Slack, «your numeric user ID» on Telegram, «your email address» for mail; else the plain words. */
export function accountIdText(provider) {
    const key = String(provider || '').trim().toLowerCase();
    if (EMAIL_PROVIDERS.has(key)) return 'your email address';
    return ACCOUNT_ID_WORDS[key] || 'the account ID this service uses';
}

/** How much of the service a place covers, from its record (a connector's own scope word wins). */
export function placeScopeText(place) {
    if (place?.scope === 'conversation' || place?.conversationId) {
        return place?.conversationId ? `one conversation · ${place.conversationId}` : 'one conversation';
    }
    if (place?.accountId) return `whole account · ${place.accountId}`;
    return 'everything this connector serves';
}

/** A dollar amount as Costs writes it (grouped); an absent amount is the dash, never $0.00. */
export const usdText = formatUsd;

/** The money line of the chip's Shade and the Dashboard's headline card: «$1.50 of $20.00 in the last 24 hours»; a
 *  floor (≥) while some calls have no price; '' when either amount is unknown. */
export function allowanceText(snapshot) {
    const { spentUsd, dailyUsd } = snapshot;
    if (spentUsd === null || dailyUsd === null) return '';
    const floor = snapshot.unknownUnmetered > 0 ? '≥ ' : '';
    // Open accounting keeps "so far" (DESIGN): a call still open may add more, so even $0.00 is no final total.
    const sofar = !floor && snapshot.allowanceOpenRows > 0 ? ' so far' : '';
    const degraded = snapshot.integrityDegraded ? ' (ledger integrity degraded)' : '';
    return `${floor}${formatUsd(spentUsd)}${sofar} of ${formatUsd(dailyUsd)} in the last 24 hours${degraded}`;
}

// The owner's move (DESIGN "Status marks", amber): consciousness stopped by its spending limit (new starts refused, or
// its own wake held at the allowance it started with) or waiting for a first conversation. A failure is an outcome's
// red dot, an unconfirmed outcome a ring, any other paused wake two bars, a due wake the still mark.
const OWNER_MOVE = new Set(['allowance_exhausted', 'waiting_for_first_conversation']);
const FAILED = new Set(['wake_failed', 'wake_rejected', 'stopped']);
const UNCONFIRMED = new Set(['wake_outcome_unknown', 'allowance_unknown']);
const ALLOWANCE_HOLDS = new Set(['allowance', 'allowance_and_cap']);

/** Whether its own wake is held, proven, at the allowance it started with (consciousness_view.hold_cause): the
 *  status word stays `wake_paused`; the cause is its own fact. */
export const allowanceHold = (snapshot) => snapshot.enabled === true && snapshot.status === 'wake_paused'
    && snapshot.liveWakeHeld === true && ALLOWANCE_HOLDS.has(snapshot.holdCause);

/** Whether the next step is the allowance, edited in Costs: new starts refused by it, or a wake held at it. */
export const allowanceMove = (snapshot) => snapshot.enabled === true
    && (snapshot.status === 'allowance_exhausted' || allowanceHold(snapshot));

/** The one status mark of a consciousness state, for the bar's chip and the Dashboard's headline alike; `owner` adds
 *  an owner's move the caller knows of (an Evolution campaign that waits for the owner). */
export function consciousnessMark(snapshot, { owner = false } = {}) {
    const off = snapshot.enabled === false;
    const unknown = snapshot.enabled === null;
    const status = snapshot.status;
    if (owner || (!off && !unknown && (OWNER_MOVE.has(status) || allowanceHold(snapshot)))) return { shape: 'dot', hue: 'owner' };
    if (status === 'thinking') return { shape: 'moving', hue: 'ink' };
    if (off) return { shape: 'none', hue: 'ink' };
    if (FAILED.has(status)) return { shape: 'dot', hue: 'failed' };
    if (unknown || UNCONFIRMED.has(status)) return { shape: 'ring', hue: 'ink' };
    if (status === 'wake_paused') return { shape: 'pause', hue: 'ink' };
    return status === 'due' ? { shape: 'still', hue: 'ink' } : { shape: 'none', hue: 'ink' };
}

/** «12:40» in the viewer's clock, or '' for an absent or unreadable instant. */
export function clockText(value) {
    const time = Date.parse(String(value || ''));
    if (!Number.isFinite(time)) return '';
    return new Date(time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
}

/** The chip's clock in the viewer's zone; only an unexplained due wake names its appointment. */
export function deadlineWords(snapshot, now = Date.now()) {
    if (snapshot.liveWakeTaskId || snapshot.liveWakeHeld || snapshot.enabled === false || snapshot.enabled === null) {
        return '';
    }
    // An exhausted allowance is checked at its reset, or sooner once a changed daily limit moved the next attempt.
    const wake = Date.parse(snapshot.nextWakeAt);
    const nextAt = snapshot.status === 'allowance_exhausted' && snapshot.allowanceResetsAt
        && !(wake > now && wake < Date.parse(snapshot.allowanceResetsAt)) ? snapshot.allowanceResetsAt : snapshot.nextWakeAt;
    const next = Date.parse(nextAt);
    if (!Number.isFinite(next)) return '';
    const due = next <= now;
    if (due && snapshot.status !== 'due') return '';
    const value = due ? snapshot.appointmentAt : nextAt;
    const at = new Date(value);
    if (!Number.isFinite(at.getTime())) return due ? tr('consciousness.due', 'due') : '';
    const date = at.toDateString() === new Date(now).toDateString() ? ''
        : `${at.toLocaleDateString([], { month: 'short', day: 'numeric' })} `;
    const time = date + clockText(value);
    return due ? fmt('since {time}', { time }) : time;
}

/** Own wakes and all place wakes keep their order within their separate lists. */
export function splitWakes(wakes) {
    const own = [], places = [];
    for (const wake of Array.isArray(wakes) ? wakes : []) (wake.cause === 'focus' ? places : own).push(wake);
    return { own, places };
}

/** A duration between two instants in words: «40 s», «3 min», «2 h»; '' when either is unknown. */
export function durationText(from, to) {
    const start = Date.parse(String(from || ''));
    const end = Date.parse(String(to || ''));
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return '';
    const seconds = Math.round((end - start) / 1000);
    if (seconds < 60) return fmt('{n} s', { n: seconds });
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return fmt('{n} min', { n: minutes });
    return fmt('{n} h', { n: Math.round(minutes / 60) });
}

/** The card's terminal family and the server's distinct active phases. */
export function wakePhaseText(wake) {
    const active = { waiting: 'Waiting', queued: 'Queued', paused: 'Paused', unknown: 'State unknown' };
    if (wake.phase in active) return tr(`task.phase.${wake.phase}`, active[wake.phase]);
    if (['running', 'working', 'done', 'warn', 'error', 'cancelled'].includes(wake.phase)) {
        return taskPresentation(wake.phase === 'running' ? 'working' : wake.phase, !wake.live && wake.quiet).headline;
    }
    return tr('task.phase.unknown', 'State unknown');
}

export function wakeFactsText(wake) {
    return [
        wake.toolCalls === null ? '' : fmt(wake.toolCalls === 1 ? '{n} tool call' : '{n} tool calls', { n: wake.toolCalls }),
        wake.rounds === null ? '' : fmt(wake.rounds === 1 ? '{n} round' : '{n} rounds', { n: wake.rounds }),
    ].filter(Boolean).join(' · ');
}

export function wakeCheckText(check) {
    const words = {
        checked: 'Safety Supervisor checked {tool}', warned: 'Safety Supervisor warned about {tool}',
        held: 'Safety Supervisor held {tool}', not_checked: 'not checked ({tool})',
        sent_despite: check.tool === 'place_reply' ? '{tool} sent despite the check' : '{tool} let through despite the check',
    };
    if (check.chip === 'not_checked' && !check.tool) return tr('consciousness.check.not_checked', 'Not checked');
    const named = { place_reply: tr('consciousness.tool.reply', 'the reply'), send_user_message: tr('consciousness.tool.message', 'the message') }[check.tool];
    const text = words[check.chip] ? fmt(words[check.chip], { tool: named || check.tool }) : '';
    // Sentence case for the chip, except when it starts with a raw tool id.
    return !named && check.tool && text.startsWith(check.tool) ? text : capitalized(text);
}

/** A warning chip on the wake row says what the Safety Supervisor did in words; a tool it names in words stays, a raw
 *  tool id waits in the row's Details (wakeCheckText). */
export function wakeCheckChip(check) {
    if (!check.tool || ['place_reply', 'send_user_message'].includes(check.tool)) return wakeCheckText(check);
    const words = { checked: 'Safety Supervisor checked a step', warned: 'Safety Supervisor warned about a step',
        held: 'Safety Supervisor held a step', sent_despite: 'A step was let through despite the check', not_checked: 'Not checked' };
    return words[check.chip] ? tr(`consciousness.chip.${check.chip}`, words[check.chip]) : '';
}

/** Keep the card's scoped cost projection and its words; openness is said, and an unknown cost leaves the row for its
 *  Details ('' here, never a false zero). */
export function wakeCostText(wake) {
    const meta = taskCostMeta(wake.cost)[0];
    if (!meta || meta === 'Cost unknown' || meta === 'Cost pending' || meta === 'Cost not priced' || meta === 'cost unavailable') return '';
    // These are the card's closed presentation templates, never authored text.
    const words = meta.startsWith('Tracked: up to ') ? fmt('Tracked: up to {amount}', { amount: meta.slice(15) })
        : meta.startsWith('Tracked: ') ? fmt('Tracked: {amount}', { amount: meta.slice(9) })
            : meta.startsWith('up to ') ? fmt('up to {amount}', { amount: meta.slice(6) }) : meta;
    return wake.live || wake.cost.cost_presentation?.accounting_open || wake.cost.cost_presentation?.tracked_final === false
        ? fmt('{cost} so far', { cost: words }) : words;
}
