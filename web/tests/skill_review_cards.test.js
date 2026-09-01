import fs from 'node:fs';
import assert from 'node:assert/strict';
import test from 'node:test';

import {
    loadSkillReviewDetail,
    nestedSkillReviewRef,
    renderSkillReviewDisclosure,
    summarizeSkillReviewMessage,
    wireSkillReviewDisclosure,
} from '../modules/skill_review_card.js';

const REFERENCE_TEXT = 'Skill review round 2 — snapshot abc123def456 (attempt 1): `alpha` — status=clean, source=skills';

function makeFull() {
    return {
        dataset: {}, innerHTML: '', attrs: {},
        setAttribute(key, value) { this.attrs[key] = value; },
    };
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
    assert.match(html, /<div class="skill-review-full" data-skill-review-full data-chat-markdown-enhanced="1" hidden><\/div>/);
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

test('permanent bounded 404 parses the server error and offers no fake Retry', async () => {
    const full = makeFull();
    const state = await loadSkillReviewDetail(full, { skill: 'alpha', jobId: 'skill-job-1' }, {
        fetchImpl: async () => ({
            ok: false, status: 404,
            json: async () => ({ error: 'review record unavailable outside the bounded history window' }),
        }),
        render: () => { throw new Error('must not render on failure'); },
    });
    assert.equal(state, 'error');
    assert.equal(full.dataset.state, 'error');
    assert.match(full.innerHTML, /Review details unavailable/);
    assert.match(full.innerHTML, /HTTP 404/);
    assert.match(full.innerHTML, /unavailable outside the bounded history window/);
    assert.match(full.innerHTML, /Cost unavailable/);
    assert.doesNotMatch(full.innerHTML, /data-skill-review-retry/);
});

test('transient non-OK detail parses JSON and retains Retry', async () => {
    const full = makeFull();
    const state = await loadSkillReviewDetail(full, { skill: 'alpha', jobId: 'skill-job-1' }, {
        fetchImpl: async () => ({
            ok: false, status: 503,
            json: async () => ({ error: 'temporary ledger read failure' }),
        }),
    });
    assert.equal(state, 'error');
    assert.match(full.innerHTML, /HTTP 503: temporary ledger read failure/);
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

test('nested attempts reuse the instance exact-job store after DOM rebuild', async () => {
    const store = new Map();
    let fetches = 0;
    const deps = {
        store,
        fetchImpl: async () => { fetches += 1; return okResponse('cached exact detail'); },
        render: (markdown) => markdown,
    };
    await loadSkillReviewDetail(makeFull(), { skill: 'alpha', jobId: 'nested-job' }, deps);
    const rebuilt = makeFull();
    await loadSkillReviewDetail(rebuilt, { skill: 'alpha', jobId: 'nested-job' }, deps);
    assert.equal(fetches, 1);
    assert.match(rebuilt.innerHTML, /cached exact detail/);
    assert.deepEqual(nestedSkillReviewRef({ dataset: {
        skillReviewSkill: 'alpha', skillReviewJob: 'nested-job',
    } }), { skill: 'alpha', jobId: 'nested-job' });
});

test('cached loaded detail does not repaint an already-loaded live node', async () => {
    const store = new Map();
    const full = makeFull();
    let fetches = 0;
    let renders = 0;
    const deps = {
        store,
        fetchImpl: async () => {
            fetches += 1;
            return okResponse('cached body');
        },
        render: (markdown) => {
            renders += 1;
            return '<p>' + markdown + '</p>';
        },
    };
    await loadSkillReviewDetail(full, { skill: 'alpha', jobId: 'reader-job' }, deps);
    const readerOwnedMarkup = '<p>cached body</p><span data-selection-anchor>keep</span>';
    full.innerHTML = readerOwnedMarkup;
    await loadSkillReviewDetail(full, { skill: 'alpha', jobId: 'reader-job' }, deps);
    assert.equal(fetches, 1);
    assert.equal(renders, 1);
    assert.equal(full.innerHTML, readerOwnedMarkup);
});

test('same exact detail shares one in-flight read across a DOM rebuild', async () => {
    const store = new Map();
    let resolve;
    let fetches = 0;
    const gate = new Promise((done) => { resolve = done; });
    const deps = {
        store,
        fetchImpl: async () => { fetches += 1; await gate; return okResponse('settled detail'); },
        render: (markdown) => markdown,
    };
    const first = makeFull();
    const second = makeFull();
    const one = loadSkillReviewDetail(first, { skill: 'alpha', jobId: 'shared-job' }, deps);
    await Promise.resolve();
    const two = loadSkillReviewDetail(second, { skill: 'alpha', jobId: 'shared-job' }, deps);
    assert.equal(fetches, 1);
    assert.equal(second.dataset.state, 'loading');
    assert.equal(second.attrs['aria-busy'], 'true');
    assert.match(second.innerHTML, /role="status" aria-live="polite"/);
    resolve();
    await Promise.all([one, two]);
    assert.equal(second.dataset.state, 'loaded');
    assert.equal(second.attrs['aria-busy'], 'false');
    assert.match(second.innerHTML, /settled detail/);
});

test('retry resets the shared error and exposes accessible busy state', async () => {
    const store = new Map();
    const full = makeFull();
    let attempts = 0;
    const deps = {
        store,
        fetchImpl: async () => {
            attempts += 1;
            if (attempts === 1) throw new Error('offline');
            return okResponse('recovered');
        },
        render: (markdown) => markdown,
    };
    await loadSkillReviewDetail(full, { skill: 'alpha', jobId: 'retry-job' }, deps);
    assert.match(full.innerHTML, /role="status" aria-live="polite"/);
    await loadSkillReviewDetail(full, { skill: 'alpha', jobId: 'retry-job' }, { ...deps, retry: true });
    assert.equal(attempts, 2);
    assert.equal(full.attrs['aria-busy'], 'false');
    assert.match(full.innerHTML, /recovered/);
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

test('exact server accounting renders without a client-authored cost fallback', async () => {
    const full = makeFull();
    await loadSkillReviewDetail(full, { skill: 'alpha', jobId: 'j1' }, {
        fetchImpl: async () => okResponse('## Findings\n\n### Review accounting\n\n- Cash: settled $0.10.'),
        render: (markdown) => `<rendered>${markdown}</rendered>`,
    });
    assert.match(full.innerHTML, /Review accounting/);
    assert.doesNotMatch(full.innerHTML, /skill-review-cost-unavailable/);
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
    let writes = 0;
    assert.equal(wireSkillReviewDisclosure(stub.bubble, {
        fetchImpl: async () => { fetches += 1; return okResponse('full body'); },
        render: (markdown) => markdown,
        onDomWrite(mutate) { writes += 1; return mutate(); },
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
    assert.ok(writes >= 3, 'expand plus loading/final detail writes use the injected boundary');
});

test('wiring: legacy rows toggle locally and never fetch', async () => {
    const stub = makeBubble();
    let fetches = 0;
    wireSkillReviewDisclosure(stub.bubble, {
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
    wireSkillReviewDisclosure(stub.bubble, {
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
    assert.equal(wireSkillReviewDisclosure(bubble), false);
});

test('the inline review body opts out of the chat pre-wrap in the template itself', () => {
    const source = fs.readFileSync(new URL('../modules/skill_review_card.js', import.meta.url), 'utf8');
    assert.match(source, /class="skill-review-full" data-skill-review-full data-chat-markdown-enhanced="1"/);
});
