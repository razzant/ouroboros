/**
 * Skill Review chat cards (node-tested).
 *
 * Since a776639f the producer writes ONE compact reference line per review
 * terminal into chat.jsonl; the full evidence lives in the skill's durable
 * review_history.jsonl. Rows carrying the exact-job reference (skill + job_id)
 * lazily fetch the server-rendered block from
 * GET /api/skills/{skill}/review-history/{job_id}; rows without it (legacy
 * full-text rows) keep local expansion of their own text.
 */

import { apiFetch } from './api_client.js';
import { escapeHtmlAttr, renderMarkdown } from './utils.js';

const writeDirectly = (mutate) => mutate();

// escapeHtmlAttr (pure string, node-safe) is used for text content too:
// over-escaping quotes renders identically in the browser.

export function summarizeSkillReviewMessage(text) {
    const raw = String(text || '');
    const lines = raw.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    const headline = lines[0] || 'Skill review';
    const hashLine = lines.find((line) => line.startsWith('content_hash=')) || '';
    const reviewersLine = lines.find((line) => line.startsWith('Reviewers:')) || '';
    const findingsLine = lines.find((line) => /^##\s+Findings/.test(line)) || '';
    const meta = [hashLine, reviewersLine.replace(/^Reviewers:\s*/, ''), findingsLine.replace(/^##\s*/, '')]
        .filter(Boolean)
        .map((line) => escapeHtmlAttr(line.length > 140 ? `${line.slice(0, 137)}...` : line))
        .join(' · ');
    return {
        headline: escapeHtmlAttr(headline.replace(/^#+\s*/, '')),
        meta,
    };
}

export function renderSkillReviewDisclosure(text, ref = null, deps = {}) {
    const render = deps.render || renderMarkdown;
    const summary = summarizeSkillReviewMessage(text);
    const jobRef = ref && ref.skill && ref.jobId ? ref : null;
    const refAttrs = jobRef
        ? ` data-skill-review-skill="${escapeHtmlAttr(jobRef.skill)}" data-skill-review-job="${escapeHtmlAttr(jobRef.jobId)}"`
        : '';
    return `
        <div class="skill-review-disclosure" data-skill-review-disclosure data-expanded="0"${refAttrs}>
            <button type="button" class="skill-review-summary-button" data-skill-review-toggle aria-expanded="false">
                <span class="skill-review-summary-main">${summary.headline}</span>
                <span class="skill-review-summary-side">
                    <span class="skill-review-meta">${summary.meta}</span>
                    <span class="skill-review-toggle-label">Show review</span>
                </span>
            </button>
            <div class="skill-review-full" data-skill-review-full data-chat-markdown-enhanced="1" hidden>${jobRef ? '' : render(text)}</div>
        </div>
    `;
}

/**
 * Fill a reference row's detail container from
 * GET /api/skills/{skill}/review-history/{job_id}. States ride
 * `full.dataset.state`: '' (idle) → 'loading' → 'loaded' | 'error'. Nested
 * Reviews may provide a keyed per-instance store so an immutable exact-job
 * read, including its in-flight promise, survives DOM replacement. Retry
 * explicitly replaces an error entry. Errors stay honest and inline.
 */
function detailStoreKey(ref) {
    return `${ref.skill}\u0000${ref.jobId}`;
}

async function responseJson(resp) {
    try {
        return await resp.json();
    } catch {
        return null;
    }
}

function renderDetailState(full, entry, render, onDomWrite = writeDirectly) {
    if (!full || !entry) return false;
    return onDomWrite(() => {
        full.dataset.state = entry.state;
        // Every state this renderer writes is HTML (a status div or rendered
        // markdown), never authored plain text: opt the node out of the chat
        // bubble's pre-wrap so block markup does not gain a blank line per
        // source newline. Host-owned marker, same attribute the chat bubble uses.
        full.dataset.chatMarkdownEnhanced = '1';
        full.setAttribute?.('aria-busy', entry.state === 'loading' ? 'true' : 'false');
        if (entry.state === 'loading') {
            full.innerHTML = '<div class="skill-review-loading" role="status" aria-live="polite">Loading review details…</div>';
        } else if (entry.state === 'loaded') {
            full.innerHTML = render(entry.markdown);
        } else if (entry.state === 'error') {
            full.innerHTML = `<div class="skill-review-error" role="status" aria-live="polite">Review details unavailable (${escapeHtmlAttr(entry.error)}). `
                + (entry.retryable
                    ? '<button type="button" class="skill-review-retry" data-skill-review-retry>Retry</button>'
                    : '')
                + '</div><div class="skill-review-cost-unavailable">Cost unavailable</div>';
        }
        return true;
    });
}

function renderDetailStateIfChanged(full, entry, render, onDomWrite = writeDirectly) {
    if (!full || !entry || full.dataset.state === entry.state) return false;
    return renderDetailState(full, entry, render, onDomWrite);
}

// The per-instance store keeps heavy rendered markdown per exact job, so it
// is trimmed FIFO past this many entries (issue #135).
export const SKILL_REVIEW_DETAIL_CAP = 200;
export async function loadSkillReviewDetail(full, ref, deps = {}) {
    const fetchImpl = deps.fetchImpl || apiFetch;
    const render = deps.render || renderMarkdown;
    const onDomWrite = deps.onDomWrite || writeDirectly;
    const store = deps.store instanceof Map ? deps.store : null;
    if (!full || !ref || !ref.skill || !ref.jobId) return '';
    const cacheKey = detailStoreKey(ref);
    let entry = store?.get(cacheKey) || null;
    if (deps.retry === true && entry?.state === 'error' && entry.retryable !== false) {
        entry = null;
        store?.delete(cacheKey);
    }
    if (entry) {
        // A keyed Reviews reconcile deliberately keeps a live detail node,
        // including its markdown descendants, selection and scroll position.
        // Repainting a cache hit would replace that user-owned DOM on every
        // unrelated review update. Paint only when this node has not reached
        // the cached state yet; an explicit retry remains the only forced
        // rewrite path.
        renderDetailStateIfChanged(full, entry, render, onDomWrite);
        if (entry.state === 'loading' && entry.promise) {
            await entry.promise;
            renderDetailStateIfChanged(full, entry, render, onDomWrite);
        }
        return entry.state;
    }
    if (full.dataset.state === 'loading' || full.dataset.state === 'loaded') {
        return full.dataset.state;
    }
    entry = {
        state: 'loading', markdown: '', error: '', promise: null,
        retryable: true,
    };
    store?.set(cacheKey, entry);
    while (store?.size > SKILL_REVIEW_DETAIL_CAP) store.delete(store.keys().next().value);
    renderDetailState(full, entry, render, onDomWrite);
    entry.promise = (async () => {
        try {
            const url = `/api/skills/${encodeURIComponent(ref.skill)}/review-history/${encodeURIComponent(ref.jobId)}`;
            const resp = await fetchImpl(url, { cache: 'no-store' });
            const data = await responseJson(resp);
            if (!resp.ok) {
                const detail = String(data?.error || data?.detail || '').trim();
                const error = new Error(`HTTP ${resp.status}${detail ? `: ${detail}` : ''}`);
                error.retryable = Number(resp.status) !== 404;
                throw error;
            }
            const markdown = String(data?.markdown || '');
            if (!markdown) throw new Error('empty review detail');
            entry.markdown = markdown;
            entry.state = 'loaded';
        } catch (err) {
            entry.error = String(err?.message || err);
            entry.retryable = err?.retryable !== false;
            entry.state = 'error';
        }
    })();
    await entry.promise;
    renderDetailStateIfChanged(full, entry, render, onDomWrite);
    return entry.state;
}

/** Exact immutable job reference carried by a nested Reviews attempt. */
export function nestedSkillReviewRef(detail) {
    const skill = detail?.dataset?.skillReviewSkill || '';
    const jobId = detail?.dataset?.skillReviewJob || '';
    return skill && jobId ? { skill, jobId } : null;
}

/**
 * Wire one rendered skill-review bubble: expand/collapse toggle, first-expand
 * lazy fetch for reference rows, and the error-state Retry. `bubble` is the
 * chat bubble element containing a `renderSkillReviewDisclosure` result.
 */
export function wireSkillReviewDisclosure(bubble, deps = {}) {
    const onDomWrite = deps.onDomWrite || writeDirectly;
    const toggle = bubble.querySelector('[data-skill-review-toggle]');
    if (!toggle) return false;
    const disclosure = bubble.querySelector('[data-skill-review-disclosure]');
    const full = bubble.querySelector('[data-skill-review-full]');
    const reviewRef = disclosure?.dataset.skillReviewSkill && disclosure?.dataset.skillReviewJob
        ? { skill: disclosure.dataset.skillReviewSkill, jobId: disclosure.dataset.skillReviewJob }
        : null;
    toggle.addEventListener('click', () => {
        const label = bubble.querySelector('.skill-review-toggle-label');
        const expanded = disclosure?.dataset.expanded === '1';
        if (!disclosure || !full) return;
        onDomWrite(() => {
            disclosure.dataset.expanded = expanded ? '0' : '1';
            full.hidden = expanded;
            toggle.setAttribute('aria-expanded', expanded ? 'false' : 'true');
            if (label) label.textContent = expanded ? 'Show review' : 'Hide review';
            return true;
        });
        // Reference rows fetch the rendered review once on first expand;
        // collapse/re-expand keeps whatever state the container reached.
        if (!expanded && reviewRef && !full.dataset.state) {
            void loadSkillReviewDetail(full, reviewRef, deps);
        }
    });
    if (reviewRef && full) {
        full.addEventListener('click', (ev) => {
            if (!ev.target?.closest?.('[data-skill-review-retry]')) return;
            full.dataset.state = '';
            void loadSkillReviewDetail(full, reviewRef, { ...deps, retry: true });
        });
    }
    return true;
}
