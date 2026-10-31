import { tr, fmt } from './i18n.js';
import { taskPresentation } from './log_events.js';
import { taskCostMeta } from './chat_activity.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';

/** The relay and wake note lead: the author's whole text; the surface alone fits its first words. */
export const noteLeadText = (text) => String(text ?? '').trim();

/** The same account stamp on wake and delivery rows; account names remain authored. */
export function sendsAsHtml(sender) {
    if (!sender?.label) return '';
    return sender.owner ? escapeHtml(tr('places.identity.as_you', 'as you'))
        : fmt('as {label}', { label: `<span data-i18n-authored>${escapeHtml(sender.label)}</span>` });
}

// The owner's words for consciousness, places and safety (owner's final names, 21.10), in one place.
// Stored values never change: only these words do, so a relabel is one line here. Pure data and pure
// helpers over host facts; nothing here reads the network or the page.

/** Consciousness autonomy: stored observe / act / full. */
export const AUTONOMY_LEVELS = Object.freeze([
    { value: 'observe', label: 'Advises', note: 'On its own it studies, keeps its memory and tells you; it changes nothing outside.' },
    { value: 'act', label: 'Acts', note: 'Also acts on its own: starts work, runs commands, speaks and acts in places. It does not change itself.' },
    { value: 'full', label: 'Evolves', note: 'Also changes itself: its own code, its settings, connecting new places.' },
]);

/** Place state: stored active / silent / paused. `word` is the row's status word. */
export const PLACE_STATES = Object.freeze([
    { value: 'active', label: 'Active', word: 'active', note: 'Lives here: listens, wakes when its list says so, speaks and acts.' },
    { value: 'silent', label: 'Silent', word: 'silent', note: 'Hears everything here and changes nothing: no messages, reactions, edits or moderation.' },
    { value: 'paused', label: 'Paused', word: 'paused', note: 'Away from this place: no wakes, no actions. Still connected; reported events are saved.' },
]);

/** What Silent enforces, said by the host (verbatim, owner 21.10). */
export const SILENT_CARD = "Writes through this connector's declared tools and delivery queue are blocked. Scripts, browser actions and unintegrated code may still write.";

/** Safety levels, each kind its own row; stored off / light / full. */
export const SAFETY_LEVELS = Object.freeze([
    { value: 'off', label: 'Off', note: 'No model checks. Fixed rules still guard protected files.' },
    { value: 'light', label: 'Light', note: 'A second model checks steps that reach other people and services: messages, posts, skills, connectors, starting work. Not shell commands.' },
    { value: 'full', label: 'Full', note: 'Also shell commands and scripts. File edits are not model-checked.' },
]);
export const SAFETY_RANK = Object.freeze({ off: 0, light: 1, full: 2 });
export const SAFETY_KINDS = Object.freeze({ your_work: 'Your work', consciousness: 'Consciousness' });

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
    sent_despite: { label: 'sent despite the check', tone: 'warn' },
    held: { label: 'held', tone: 'warn' },
    not_checked: { label: 'not checked', tone: 'neutral' },
});

export const placeStateOf = (value) => PLACE_STATES.find((state) => state.value === value) || null;
export const autonomyOf = (value) => AUTONOMY_LEVELS.find((level) => level.value === value) || null;
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

/** Lower an initial capital only in an otherwise lowercase first word; keep GitHub and PR intact. */
export const sentenceLabel = (label) => /^\p{Lu}\p{Ll}+(?!\p{L})/u.test(label)
    ? label[0].toLowerCase() + label.slice(1) : label;

/** «wakes for direct messages, mentions and replies», or '' when nothing wakes it here. */
export function wakesForText(wakes = [], declared = null, labels = null) {
    const facts = [...new Set(wakes.map((item) => item?.fact).filter(Boolean))];
    const kinds = facts.filter((fact) => fact !== 'from_bot');
    const named = { ...signalLabels(declared), ...labels };
    const word = (fact) => named[fact] ? sentenceLabel(named[fact]) : wakeFactWord(fact);
    return [kinds.length ? fmt('wakes for {kinds}', { kinds: listWords(kinds.map(word)) }) : '',
        facts.includes('from_bot') ? word('from_bot') : ''].filter(Boolean).join(' · ');
}

export const authorWord = (author) => AUTHOR_WORDS[author] || (author ? `added by ${author}` : '');

/** The row's status word for a stored state; an unknown state says so instead of guessing. */
export const placeStateWord = (state) => placeStateOf(state)?.word || tr('places.state.unknown', 'state unknown');

/**
 * A place row's one status sentence: «active · wakes for direct messages, mentions and replies». A Silent place
 * keeps its list (it still wakes there to read and think; what it would say or change is held); a Paused place
 * says only its state. The count is open focus turns, not owed replies.
 */
export function placeStatusText(place) {
    const parts = [placeStateWord(place?.state)];
    if (place?.state === 'active' || place?.state === 'silent') parts.push(wakesForText(place.wakes, place.facts, place.labels));
    if (Number.isInteger(place?.unanswered) && place.unanswered > 0) parts.push(fmt('{n} open turns', { n: place.unanswered }));
    return parts.filter(Boolean).join(' · ');
}

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

/** A dollar amount as Costs writes it; an absent amount is the dash, never $0.00. */
export function usdText(value) {
    if (value === null || value === undefined || value === '') return '—';
    const number = Number(value);
    if (!Number.isFinite(number)) return '—';
    return `$${number < 0.01 && number > 0 ? number.toFixed(4) : number.toFixed(2)}`;
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
    const nextAt = snapshot.status === 'allowance_exhausted'
        ? snapshot.allowanceResetsAt || snapshot.nextWakeAt : snapshot.nextWakeAt;
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
        held: 'Safety Supervisor held {tool}', sent_despite: '{tool} sent despite the check',
        not_checked: 'not checked ({tool})',
    };
    if (check.chip === 'not_checked' && !check.tool) return tr('consciousness.check.not_checked', 'Not checked');
    const named = { place_reply: tr('consciousness.tool.reply', 'the reply'), send_user_message: tr('consciousness.tool.message', 'the message') }[check.tool];
    const text = words[check.chip] ? fmt(words[check.chip], { tool: named || check.tool }) : '';
    // Sentence case for the chip, except when it starts with a raw tool id.
    return !named && check.tool && text.startsWith(check.tool) ? text : capitalized(text);
}

/** Keep the card's scoped cost projection; word openness and unknown without a false zero. */
export function wakeCostText(wake) {
    const meta = taskCostMeta(wake.cost)[0];
    if (!meta || meta === 'Cost unknown' || meta === 'cost unavailable') return tr('consciousness.spend_unknown', 'spend unknown');
    // These are the card's closed presentation templates, never authored text.
    const words = meta.startsWith('Tracked: up to ') ? fmt('Tracked: up to {amount}', { amount: meta.slice(15) })
        : meta.startsWith('Tracked: ') ? fmt('Tracked: {amount}', { amount: meta.slice(9) })
            : meta.startsWith('up to ') ? fmt('up to {amount}', { amount: meta.slice(6) }) : meta;
    return wake.live || wake.cost.cost_presentation?.accounting_open || wake.cost.cost_presentation?.tracked_final === false
        ? fmt('{cost} so far', { cost: words }) : words;
}
