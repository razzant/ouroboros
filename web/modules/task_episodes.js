/**
 * Continue episodes of one work (owner QB = A, FINAL_DESIGN D10): a Continue is the next episode of
 * the SAME card. The current episode owns the status chip; every episode is one meta line under the
 * title — with its own name, time and past outcome —
 * never a column of chips.
 *
 * The chain is the host's explicit fact only: a settled task's `continued_by` or `continuation_offer`
 * naming its successor once it says `admitted`. A `bound` claim is still pending (its card offers
 * Retry Continue) and is no episode yet. Names, projects or an origin message shared by two tasks are
 * not a chain.
 */
import { escapeHtmlAttr, escapeHtmlText as escapeHtml } from './utils.js';
import { reasonCodesNote, taskDoneIsTerminal, taskPresentation, taskReasonCodes, taskReasonDetail, taskTerminalPhase } from './log_events.js';
import { fmt } from './i18n.js';
import { outcomeTone } from './task_phase_chip.js';
import { projectCollapsedActivity, renderCollapsedActivity } from './chat_activity.js';

export function admittedSuccessor(row) {
    const admitted = (link) => (link && typeof link === 'object' && link.state === 'admitted' ? link.successor_task_id : '');
    return String(admitted(row?.continued_by) || admitted(row?.continuation_offer) || '').trim();
}

/** Fold one row's facts into a task's episode: when its attempt started, when and how it ended. */
export function noteEpisodeFacts(previous = {}, row = {}) {
    const next = { ...previous };
    const name = String(row.suggested_name || row.task_name || row.title || '').trim();
    if (name) next.name = name;
    const time = row?.terminal_time && typeof row.terminal_time === 'object' ? row.terminal_time : null;
    next.started ||= String(time?.attempt?.started_at || row?.started_at || '');
    const terminal = row?.outcome_final === true || row?.system_type === 'task_summary' || taskDoneIsTerminal(row);
    if (terminal && (row?.outcome_axes || row?.task_terminal_status || row?.status)) {
        next.ended = String(time?.occurred_at || next.ended || row?.ts || '');
        next.phase = taskTerminalPhase(row);
        // A code no sentence describes yet rides beside its neutral cause, for the expanded card (owner 2A).
        const cause = taskReasonDetail(row);
        if (cause) Object.assign(next, { cause, codes: taskReasonCodes(row) });
        else next.cause ||= '';
    } else if (time?.occurred_at && !next.ended) {
        next.ended = String(time.occurred_at);
    }
    return next;
}

const pad = (n) => String(n).padStart(2, '0');
const toDate = (raw) => {
    const value = typeof raw === 'number' ? raw : Date.parse(String(raw || ''));
    return Number.isFinite(value) ? new Date(value) : null;
};
const clock = (d) => `${pad(d.getHours())}:${pad(d.getMinutes())}`;
const dayKey = (d) => `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;

/** `17:33–18:15`, `19:42–02:59 next day`, or with dates when an episode spans more. */
export function episodeSpan(started, ended) {
    const a = toDate(started);
    const b = toDate(ended);
    if (!a && !b) return '';
    if (!b) return a ? fmt('since {time}', { time: clock(a) }) : '';
    if (!a) return fmt('until {time}', { time: clock(b) });
    const days = Math.round((new Date(b.getFullYear(), b.getMonth(), b.getDate())
        - new Date(a.getFullYear(), a.getMonth(), a.getDate())) / 86400000);
    if (dayKey(a) === dayKey(b)) return `${clock(a)}–${clock(b)}`;
    if (days === 1) return fmt('{start}–{end} next day', { start: clock(a), end: clock(b) });
    const date = (d) => d.toLocaleDateString('en', { month: 'short', day: 'numeric' });
    return `${date(a)}, ${clock(a)} – ${date(b)}, ${clock(b)}`;
}

/**
 * Paint the episode lines. One episode is not a chain: the block stays empty and hidden.
 * `current` is the last episode: its chip is the card's, so its line carries its own name and time.
 */
export function renderEpisodes(el, episodes = []) {
    if (!el) return false;
    const lines = episodes.length > 1 ? episodes.map((episode, index) => {
        const current = index === episodes.length - 1;
        const span = episodeSpan(episode.started, current && !episode.ended ? '' : episode.ended);
        const outcome = !current && episode.phase ? taskPresentation(episode.phase).headline : '';
        const tone = outcomeTone(episode.phase) || 'neutral';  // the chips' outcome tones
        const parts = [`<span class="chat-live-episode-label">${escapeHtml(fmt('Episode {n}', { n: index + 1 }))}</span>`];
        if (episode.name) parts.push(`<span class="chat-live-episode-name">${escapeHtml(episode.name)}</span>`);
        if (span) parts.push(`<span class="chat-live-episode-span">${escapeHtml(span)}</span>`);
        if (outcome) {
            const cause = String(episode.cause || '').trim();
            parts.push(`<span class="chat-live-episode-outcome" data-tone="${escapeHtmlAttr(tone)}">${escapeHtml(outcome)}</span>`
                + (cause ? `<span class="chat-live-episode-cause">: ${escapeHtml(cause)}</span>` : ''));
        }
        return `<div class="chat-live-episode" data-episode="${index + 1}"${current ? ' data-current="1"' : ''}>${parts.join(' · ')}</div>`;
    }) : [];
    const html = lines.join('');
    const changed = el.innerHTML !== html || el.hidden !== !lines.length;
    if (el.innerHTML !== html) el.innerHTML = html;
    if (el.hidden !== !lines.length) el.hidden = !lines.length;
    return changed;
}

/**
 * The episode chains of one chat instance. `records` is its card registry; `reanchor(record, ts)` moves a
 * card to where its work began (earliest wins); `activity(record)` is the census row its controls read.
 * The host's admitted successor carries the SAME work, so its card takes the predecessor's place and
 * name, and every episode is one meta line on it. Earlier content is disclosed using its original nodes;
 * the hidden physical shell/record stays to take late rows. A successor with no card here runs where
 * this chat sees none of its frames (a Main
 * portal's room): `follow(successorId, predecessor)` may mint its card from the facts it holds.
 * `renamed(record)` hears each title this module writes (a portal's Project reference names itself again).
 */
export function createTaskEpisodes({ records, reanchor, activity, follow = () => false, expand = () => {}, renamed = () => {} }) {
    const predecessors = new Map();  // successor task id -> predecessor task id

    function chainOf(taskId) {
        const chain = [];
        const seen = new Set([taskId]);
        for (let id = predecessors.get(taskId); id && !seen.has(id); id = predecessors.get(id)) {
            seen.add(id);
            chain.unshift(id);
        }
        return chain;
    }

    // Rehome the ORIGINAL content nodes, not snapshots of their DOM or lifecycle. Their physical
    // record/controller still receives late files, reviews and history. Old controls stay in its
    // hidden shell; only the current card offers work controls. The original Reviews door
    // is a reading control and follows its original host/controller into the disclosure.
    function disclose(record, chain) {
        const host = record.episodeHistoryEl;
        if (!host) return;
        for (const [index, previous] of chain.entries()) {
            let disclosure = previous.episodeDisclosure;
            if (!disclosure) {
                disclosure = document.createElement('details');
                disclosure.className = 'chat-episode-history';
                disclosure.dataset.episodeTask = previous.groupId;
                const summary = document.createElement('summary');
                const codes = document.createElement('p');
                codes.className = 'chat-episode-codes';
                disclosure.append(summary, codes);
                const onToggle = () => { if (disclosure.open) expand(previous); };
                disclosure.addEventListener('toggle', onToggle);
                previous.episodeDisclosure = disclosure;
                previous.episodeDisclosureDispose = () => disclosure.removeEventListener('toggle', onToggle);
            }
            const facts = previous.episode || {};
            const label = [fmt('Episode {n}', { n: index + 1 }), facts.name,
                episodeSpan(facts.started, facts.ended), facts.phase && taskPresentation(facts.phase).headline,
                facts.cause].filter(Boolean).join(' · ');
            if (disclosure.firstElementChild.textContent !== label) disclosure.firstElementChild.textContent = label;
            const codes = disclosure.querySelector('.chat-episode-codes');
            codes.textContent = reasonCodesNote(facts.codes);
            codes.hidden = !codes.textContent;
            for (const content of [previous.timelineEl, previous.reviewSummaryEl, previous.reviewsHostEl, previous.subagentsEl]) {
                if (content && content.parentNode !== disclosure) disclosure.appendChild(content);
            }
            const position = host.children[index] || null;
            if (position !== disclosure) host.insertBefore(disclosure, position);
        }
        host.hidden = false;
    }

    function sync(record) {
        if (!record || record.isSubagent || !record.root) return false;
        // Its own first row, remembered before the card moves to where the work began.
        if (record.ownStartedAt === undefined && record.root.dataset.ts) record.ownStartedAt = Number(record.root.dataset.ts);
        record.episode ||= {};
        // Its own name only: once in a chain, suggestedName carries the work's borrowed one.
        if (!record.episodeName) record.episode.name ||= record.suggestedName || '';
        const successorId = [...predecessors].find(([, predecessor]) => predecessor === record.groupId)?.[0];
        if (successorId && records.get(successorId)) return sync(records.get(successorId));
        const chain = chainOf(record.groupId).map((id) => records.get(id)).filter(Boolean);
        if (!chain.length) return false;
        let changed = false;
        // The work keeps its first name (a successor's own "Continue: …" title is the mechanism's).
        const name = chain[0].episode?.name || chain[0].suggestedName || chain[0].titleEl?.textContent || '';
        if (name && record.episodeName !== name) {
            record.episodeName = name;
            record.suggestedName = name;
            changed = true;
        }
        if (name && record.titleEl.textContent !== name) {
            record.titleEl.textContent = name;
            renamed(record);
            // An unnamed successor's own latest line stood in for its title; under the work's name it is the line again.
            renderCollapsedActivity(record, projectCollapsedActivity({ record, suggestedName: name, headline: record.collapsedActivity,
                previous: record.collapsedActivity }));
            changed = true;
        }
        for (const predecessor of chain) {
            if (!predecessor.root.hidden) { predecessor.root.hidden = true; changed = true; }
            predecessor.root.dataset.episodeOf = record.groupId;
        }
        disclose(record, chain);
        const startedAt = Number(chain[0].root.dataset.ts);
        if (Number.isFinite(startedAt)) changed = reanchor(record, startedAt) || changed;
        const current = { ...(record.episode || {}) };
        if (!current.started) {
            const startedSec = Number(activity(record)?.started_at);
            current.started = Number.isFinite(startedSec) && startedSec > 0 ? startedSec * 1000 : record.ownStartedAt || '';
        }
        if (!record.finished) current.ended = '';
        return renderEpisodes(record.episodesEl, [...chain.map((r) => r.episode || {}), current]) || changed;
    }

    return {
        /** Naming belongs with the chain: physical names and the work heading differ. */
        rename(record, name) {
            if (!record || record.isSubagent) return false;
            record.episode = { ...record.episode, name };
            if (record.episodeName) return sync(record);
            const titleChanged = Boolean(record.titleEl && record.titleEl.textContent !== name);
            let changed = record.suggestedName !== name || titleChanged;
            record.suggestedName = name;
            if (titleChanged) { record.titleEl.textContent = name; renamed(record); }
            // Restore the deferred collapsed activity after naming.
            changed = renderCollapsedActivity(record, projectCollapsedActivity({
                record, suggestedName: name, headline: record.collapsedActivity,
                previous: record.collapsedActivity,
            })) || changed;
            changed = sync(record) || changed;
            return Boolean(changed && record.root?.isConnected);
        },
        /** A row or task detail of this card landed: its episode facts and the host's chain fact, if any. */
        note(record, row) {
            if (!record || record.isSubagent || !row || typeof row !== 'object') return false;
            record.episode = noteEpisodeFacts(record.episode, row);
            const successor = admittedSuccessor(row);
            const linked = Boolean(successor) && successor !== record.groupId;
            if (linked) predecessors.set(successor, record.groupId);
            const changed = sync(record);
            return Boolean((linked && !records.get(successor)?.root?.isConnected && follow(successor, record)) || changed);
        },
        /** A card that became visible: if it belongs to a known chain, the chain's card takes over. */
        sync: (record) => (record && (predecessors.has(record.groupId) || record.root?.dataset?.episodeOf) ? sync(record) : false),
    };
}
