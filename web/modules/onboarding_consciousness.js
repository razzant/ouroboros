// The first-run wizard's Consciousness step (owner decision 2026-11-20): one two-position switch between Choose access
// and Review limits, on by default. The server offers it only while the next boot mints a first state and setup has
// never completed here (`bootstrap.consciousnessOffer`; `stepOrder` omits it otherwise): where any state exists the
// step is absent and Dashboard › Consciousness changes it. It asks what Ouroboros should do while the owner is away,
// each position answering under its name, and shows the limits consciousness runs under now, in the host's words
// (consciousness_view.first_run_words). The position rides the one completion request as `consciousness` and the first
// state is born with it: no command is sent and Main shows no row.
import { escapeHtmlAttr as escapeHtml, formatUsd2 } from './utils.js';
import { autonomyOf } from './consciousness_autonomy.js';

/** The bootstrap's offer as the step reads it; an older server's bootstrap without one offers nothing. */
export function normalizeOffer(offer) {
    const value = offer && typeof offer === 'object' ? offer : {};
    const number = (raw) => (raw === null || raw === undefined || raw === '' || !Number.isFinite(Number(raw)) ? null : Number(raw));
    const words = value.words && typeof value.words === 'object' ? value.words : {};
    return {
        offered: value.offered === true,
        autonomy: String(value.autonomy || ''),
        dailyUsd: number(value.dailyUsd),
        maxTasks: number(value.maxTasks),
        // The host's words (owner 15.11, 5A); an older host names no positions, and the step then shows none.
        words: Object.fromEntries(['gloss', 'off', 'on', 'lead', 'offAway', 'onAway', 'meaning', 'subscriptionCost', 'door',
            'summaryLead'].map((key) => [key, String(words[key] || '')])),
    };
}

/** The limits consciousness runs under now, one phrase each, in the words Settings and Costs use. A subscription's
 * quota is said once, beside the allowance, by the step's subscription sentence where a subscription alone pays
 * (`subscriptionOnly`). */
export function limitPhrases(facts, { subscriptionOnly = false } = {}) {
    const level = autonomyOf(facts.autonomy);
    const autonomy = level ? `Autonomy: ${level.label}. ${level.note}` : 'Autonomy unknown.';
    const allowance = facts.dailyUsd === null ? 'Daily allowance unknown.'
        : facts.dailyUsd === 0 ? 'Daily allowance $0.00: it does not wake on its own.'
            : `Up to ${formatUsd2(facts.dailyUsd)} a day in recorded spending.${subscriptionOnly ? '' : ' Subscription quota is separate.'}`;
    const tasks = facts.maxTasks === null ? 'Its own-task limit is unknown.'
        : facts.maxTasks === 0 ? 'It starts no tasks of its own.'
            : `At most ${facts.maxTasks} of its own tasks at once.`;
    return [autonomy, allowance, tasks];
}

const note = (text) => (text ? `<p class="consciousness-step-note">${escapeHtml(text)}</p>` : '');

/** The step: its title, its question as the lead copy, the two positions each saying what it does while the owner is
 * away, what holds while it is on, and where to change it later. `subscriptionOnly`: the draft reaches models through a
 * subscription alone, so the wakes spend its quota (said with the allowance; a $0.00 allowance has no wakes to spend). */
export function stepHtml(facts, on, { title = 'Consciousness', subscriptionOnly = false } = {}) {
    const { words } = facts;
    const position = (value, label, word, away) => `
                <button type="button" class="wizard-choice ${on === value ? 'active' : ''}" data-consciousness-choice="${value ? 'on' : 'off'}" aria-pressed="${on === value}">
                    <span class="tone">${word}</span><h3>${escapeHtml(label)}</h3>${away ? `<p>${escapeHtml(away)}</p>` : ''}
                </button>`;
    const lead = words.lead || words.meaning;  // an older host says the positions' meaning as one paragraph
    const [autonomy, allowance, tasks] = limitPhrases(facts, { subscriptionOnly });
    const quota = subscriptionOnly && facts.dailyUsd !== 0 ? words.subscriptionCost : '';
    return `
            <div class="step-header"><div>
                <h2 class="step-title">${escapeHtml(title)}</h2>
                ${lead ? `<p class="step-copy">${escapeHtml(lead)}</p>` : ''}
            </div></div>
            <div class="panel-card consciousness-step" data-consciousness-step>
                <div class="wizard-choice-grid two" role="group" aria-label="${escapeHtml(words.gloss || title)}">${position(false, words.off, 'Off', words.offAway)}${position(true, words.on, 'On', words.onAway)}</div>
                <div class="consciousness-step-limits">
                    <p class="consciousness-step-label">While it's on</p>
                    ${note(autonomy)}${note([allowance, quota].filter(Boolean).join(' '))}${note(tasks)}
                </div>
                ${note(words.door)}
            </div>`;
}

/** The step's state across renders: the switch's position, its Summary row and its field in the completion request. */
export function createConsciousnessStep(offer) {
    const facts = normalizeOffer(offer);
    let on = true;  // offered only before a first state exists: on unless the owner turns it off
    const step = {
        offered: () => facts.offered,
        on: () => on,
        html: (options) => stepHtml(facts, on, options),
        bind(root) {
            const buttons = [...root.querySelectorAll('[data-consciousness-choice]')];
            buttons.forEach((button) => button.addEventListener('click', () => {
                on = button.getAttribute('data-consciousness-choice') === 'on';
                buttons.forEach((choice) => {
                    const chosen = (choice.getAttribute('data-consciousness-choice') === 'on') === on;
                    choice.classList.toggle('active', chosen);
                    choice.setAttribute('aria-pressed', String(chosen));
                });
            }));
        },
        /** Summary's read-only row; none when the step is not offered. */
        summaryRows: () => (!facts.offered ? [] : [['Consciousness',
            [on ? 'On' : 'Off', on ? facts.words.on : facts.words.off].filter(Boolean).join(' · ')]]),
        /** Summary's lead where a first state is still to come: there Start also begins the first conversation. */
        summaryLead: () => (facts.offered ? facts.words.summaryLead : ''),
        /** The completion request's field, only from an offered step (absent changes nothing). */
        payload: () => (facts.offered ? { consciousness: on } : {}),
    };
    return step;
}
