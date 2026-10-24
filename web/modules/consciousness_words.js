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
    { value: 'paused', label: 'Paused', word: 'paused', note: 'Away from this place: no wakes, no actions. Still connected.' },
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

/** What wakes it, as words after "wakes for …" (the typed facts of `places.WAKE_FACTS`). */
export const WAKE_FACT_WORDS = Object.freeze({
    direct_message: 'direct messages',
    group_dm: 'group messages',
    mention: 'mentions',
    reply_to_me: 'replies',
    my_thread: 'replies in its threads',
    new_issue: 'new issues',
    new_pull_request: 'new pull requests',
    review_requested: 'review requests addressed to me',
    assigned: 'assignments addressed to me',
    any_message: 'every message',
    from_bot: 'messages from bots',
    legacy_addressed: 'messages addressed to it',
});

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
    return `${list.slice(0, -1).join(', ')} and ${list[list.length - 1]}`;
}

export const capitalized = (text) => {
    const value = String(text || '');
    return value ? value[0].toUpperCase() + value.slice(1) : value;
};

/** One wake fact in words; an unknown kind keeps its own spelling rather than a guess. */
export const wakeFactWord = (fact) => WAKE_FACT_WORDS[fact] || String(fact || '').replace(/_/g, ' ');

/** «wakes for direct messages, mentions and replies», or '' when nothing wakes it here. */
export function wakesForText(wakes = [], attentionNames = []) {
    const facts = [...new Set(wakes.map((item) => item?.fact).filter(Boolean))];
    const words = facts.map(wakeFactWord);
    if (attentionNames.length) words.push('names and aliases');
    return words.length ? `wakes for ${listWords(words)}` : '';
}

export const authorWord = (author) => AUTHOR_WORDS[author] || (author ? `added by ${author}` : '');

/** The row's status word for a stored state; an unknown state says so instead of guessing. */
export const placeStateWord = (state) => placeStateOf(state)?.word || 'state unknown';

/**
 * A place row's one status sentence: «active · wakes for direct messages, mentions and replies». A Silent place
 * keeps its list (it still wakes there to read and think; what it would say or change is held); a Paused place
 * says only its state. Unanswered messages are a count.
 */
export function placeStatusText(place) {
    const parts = [placeStateWord(place?.state)];
    if (place?.state === 'active' || place?.state === 'silent') parts.push(wakesForText(place.wakes, place.attentionNames));
    if (Number.isInteger(place?.unanswered) && place.unanswered > 0) parts.push(`${place.unanswered} unanswered`);
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

/** A duration between two instants in words: «40 s», «3 min», «2 h»; '' when either is unknown. */
export function durationText(from, to) {
    const start = Date.parse(String(from || ''));
    const end = Date.parse(String(to || ''));
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return '';
    const seconds = Math.round((end - start) / 1000);
    if (seconds < 60) return `${seconds} s`;
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes} min`;
    return `${Math.round(minutes / 60)} h`;
}
