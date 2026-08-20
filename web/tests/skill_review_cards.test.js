import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
    loadSkillReviewDetail,
    renderSkillReviewDisclosure,
    summarizeSkillReviewMessage,
    wireSkillReviewDisclosure,
} from '../modules/skill_review_card.js';

const REFERENCE_TEXT = 'Skill review round 2 — snapshot abc123def456 (attempt 1): `alpha` — status=clean, source=skills';

function makeFull() {
    return { dataset: {}, innerHTML: '' };
}

function okResponse(markdown) {
    return { ok: true, status: 200, json: async () => ({ markdown }) };
}

test('legacy skill review rows (no job reference) keep local text expansion', () => {
    const html = renderSkillReviewDisclosure('## Findings\nlegacy full body', null, {
        render: (text) => `<rendered>${text}</rendered>`,
    });
    assert.match(html, /<rendered>## Findings\nlegacy full body<\/rendered>/);
    assert.doesNotMatch(html, /data-skill-review-skill/);
    assert.doesNotMatch(html, /data-skill-review-job/);
});

test('reference rows render an empty lazy container carrying the job reference', () => {
    const html = renderSkillReviewDisclosure(REFERENCE_TEXT, { skill: 'alpha', jobId: 'skill-job-1' }, {
        render: () => { throw new Error('reference rows must not render local text'); },
    });
    assert.match(html, /data-skill-review-skill="alpha"/);
    assert.match(html, /data-skill-review-job="skill-job-1"/);
    assert.match(html, /<div class="skill-review-full" data-skill-review-full hidden><\/div>/);
    // Collapsed layout is unchanged: same summary button + toggle label.
    assert.match(html, /data-skill-review-toggle/);
    assert.match(html, /Show review/);
});

test('reference attribute values are HTML-escaped', () => {
    const html = renderSkillReviewDisclosure(REFERENCE_TEXT, { skill: 'a"l<pha', jobId: 'job"1' });
    assert.doesNotMatch(html, /data-skill-review-skill="a"l/);
    assert.match(html, /data-skill-review-skill="a&quot;l&lt;pha"/);
    assert.match(html, /data-skill-review-job="job&quot;1"/);
});

test('summarize keeps the compact headline for reference rows', () => {
    const summary = summarizeSkillReviewMessage(REFERENCE_TEXT);
    assert.match(summary.headline, /Skill review round 2/);
});

test('loadSkillReviewDetail success renders fetched markdown into the container', async () => {
    const full = makeFull();
    const calls = [];
    const state = await loadSkillReviewDetail(full, { skill: 'alpha', jobId: 'skill-job-1' }, {
        fetchImpl: async (url) => { calls.push(url); return okResponse('## Findings\n- [PASS] manifest_schema: ok'); },
        render: (markdown) => `<rendered>${markdown}</rendered>`,
    });
    assert.equal(state, 'loaded');
    assert.equal(full.dataset.state, 'loaded');
    assert.deepEqual(calls, ['/api/skills/alpha/review-history/skill-job-1']);
    assert.match(full.innerHTML, /<rendered>## Findings/);
});

test('loadSkillReviewDetail encodes skill and job id in the route', async () => {
    const full = makeFull();
    const calls = [];
    await loadSkillReviewDetail(full, { skill: 'a b', jobId: 'j/1' }, {
        fetchImpl: async (url) => { calls.push(url); return okResponse('x'); },
        render: (markdown) => markdown,
    });
    assert.deepEqual(calls, ['/api/skills/a%20b/review-history/j%2F1']);
});

test('loadSkillReviewDetail failure degrades honestly with a Retry control', async () => {
    const full = makeFull();
    const state = await loadSkillReviewDetail(full, { skill: 'alpha', jobId: 'skill-job-1' }, {
        fetchImpl: async () => ({ ok: false, status: 404, json: async () => ({ error: 'review record not found' }) }),
        render: () => { throw new Error('must not render on failure'); },
    });
    assert.equal(state, 'error');
    assert.equal(full.dataset.state, 'error');
    assert.match(full.innerHTML, /Review details unavailable/);
    assert.match(full.innerHTML, /HTTP 404/);
    assert.match(full.innerHTML, /data-skill-review-retry/);
});

test('retry after failure refetches once the state is cleared', async () => {
    const full = makeFull();
    let attempts = 0;
    const deps = {
        fetchImpl: async () => {
            attempts += 1;
            if (attempts === 1) throw new Error('network down');
            return okResponse('recovered body');
        },
        render: (markdown) => markdown,
    };
    assert.equal(await loadSkillReviewDetail(full, { skill: 'alpha', jobId: 'j1' }, deps), 'error');
    // The retry handler clears the state before re-invoking the loader.
    full.dataset.state = '';
    assert.equal(await loadSkillReviewDetail(full, { skill: 'alpha', jobId: 'j1' }, deps), 'loaded');
    assert.equal(attempts, 2);
    assert.match(full.innerHTML, /recovered body/);
});

test('loaded and in-flight containers are never refetched', async () => {
    const full = makeFull();
    let attempts = 0;
    const deps = {
        fetchImpl: async () => { attempts += 1; return okResponse('body'); },
        render: (markdown) => markdown,
    };
    await loadSkillReviewDetail(full, { skill: 'alpha', jobId: 'j1' }, deps);
    assert.equal(await loadSkillReviewDetail(full, { skill: 'alpha', jobId: 'j1' }, deps), 'loaded');
    assert.equal(attempts, 1);
    full.dataset.state = 'loading';
    assert.equal(await loadSkillReviewDetail(full, { skill: 'alpha', jobId: 'j1' }, deps), 'loading');
    assert.equal(attempts, 1);
});

test('empty markdown from the route is treated as an error, not a blank card', async () => {
    const full = makeFull();
    const state = await loadSkillReviewDetail(full, { skill: 'alpha', jobId: 'j1' }, {
        fetchImpl: async () => okResponse(''),
        render: (markdown) => markdown,
    });
    assert.equal(state, 'error');
    assert.match(full.innerHTML, /Review details unavailable/);
});

test('a missing reference is a no-op (defensive)', async () => {
    const full = makeFull();
    assert.equal(await loadSkillReviewDetail(full, null, {
        fetchImpl: async () => { throw new Error('must not fetch'); },
    }), '');
    assert.equal(full.innerHTML, '');
});

// ---- wiring (the real bubble hookup, DOM stubbed) ---------------------------

function makeBubble({ skill = '', jobId = '' } = {}) {
    const listeners = new Map();
    const disclosure = { dataset: {
        expanded: '0',
        ...(skill ? { skillReviewSkill: skill } : {}),
        ...(jobId ? { skillReviewJob: jobId } : {}),
    } };
    const full = {
        dataset: {},
        innerHTML: 'LOCAL-BODY',
        hidden: true,
        addEventListener(type, fn) { listeners.set(`full:${type}`, fn); },
    };
    const label = { textContent: 'Show review' };
    const toggle = {
        attrs: {},
        addEventListener(type, fn) { listeners.set(`toggle:${type}`, fn); },
        setAttribute(key, value) { this.attrs[key] = value; },
    };
    const bubble = {
        querySelector(selector) {
            if (selector === '[data-skill-review-toggle]') return toggle;
            if (selector === '[data-skill-review-disclosure]') return disclosure;
            if (selector === '[data-skill-review-full]') return full;
            if (selector === '.skill-review-toggle-label') return label;
            return null;
        },
    };
    return { bubble, disclosure, full, toggle, label, listeners };
}

const flushAsync = () => new Promise((resolve) => setTimeout(resolve, 0));

test('wiring: expanding a reference row fetches once; re-expand reuses the load', async () => {
    const stub = makeBubble({ skill: 'alpha', jobId: 'j1' });
    let fetches = 0;
    let repaints = 0;
    assert.equal(wireSkillReviewDisclosure(stub.bubble, () => { repaints += 1; }, {
        fetchImpl: async () => { fetches += 1; return okResponse('full body'); },
        render: (markdown) => markdown,
    }), true);
    const click = stub.listeners.get('toggle:click');
    click();               // expand → lazy fetch
    await flushAsync();
    assert.equal(fetches, 1);
    assert.equal(stub.full.dataset.state, 'loaded');
    assert.match(stub.full.innerHTML, /full body/);
    assert.equal(stub.full.hidden, false);
    assert.equal(stub.label.textContent, 'Hide review');
    click();               // collapse
    click();               // re-expand: no refetch
    await flushAsync();
    assert.equal(fetches, 1);
    assert.ok(repaints >= 3);
});

test('wiring: legacy rows toggle locally and never fetch', async () => {
    const stub = makeBubble();
    let fetches = 0;
    wireSkillReviewDisclosure(stub.bubble, () => {}, {
        fetchImpl: async () => { fetches += 1; return okResponse('x'); },
    });
    stub.listeners.get('toggle:click')();
    await flushAsync();
    assert.equal(fetches, 0);
    assert.equal(stub.full.innerHTML, 'LOCAL-BODY');
    assert.equal(stub.full.hidden, false);
    // No retry delegation is installed on legacy rows.
    assert.equal(stub.listeners.has('full:click'), false);
});

test('wiring: Retry click clears the error state and refetches', async () => {
    const stub = makeBubble({ skill: 'alpha', jobId: 'j1' });
    let attempts = 0;
    wireSkillReviewDisclosure(stub.bubble, () => {}, {
        fetchImpl: async () => {
            attempts += 1;
            if (attempts === 1) throw new Error('network down');
            return okResponse('recovered');
        },
        render: (markdown) => markdown,
    });
    stub.listeners.get('toggle:click')();
    await flushAsync();
    assert.equal(stub.full.dataset.state, 'error');
    const retryClick = stub.listeners.get('full:click');
    retryClick({ target: { closest: (sel) => (sel === '[data-skill-review-retry]' ? {} : null) } });
    await flushAsync();
    assert.equal(attempts, 2);
    assert.equal(stub.full.dataset.state, 'loaded');
    assert.match(stub.full.innerHTML, /recovered/);
});

test('wiring: non-review bubbles are left untouched', () => {
    const bubble = { querySelector: () => null };
    assert.equal(wireSkillReviewDisclosure(bubble, () => {}), false);
});

test('chat layout callback ignores late skill-review completion after destroy', () => {
    // addMessage moved with the feed/history owner (W3 wave D).
    const source = readFileSync(new URL('../modules/chat_history_sync.js', import.meta.url), 'utf8');
    assert.match(
        source,
        /wireSkillReviewDisclosure\(bubble,[\s\S]*?requestAnimationFrame\(\s*\(\) => !destroyed && updateMessagesPadding/,
    );
});
