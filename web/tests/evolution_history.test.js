import assert from 'node:assert/strict';
import test from 'node:test';
import { installDomDouble } from './dom_double.js';
import { campaignView, chartSvg, detailHtml, formatDelta, formatMeasure, historyModel, initEvolution, milestoneLabels, timeAxis, tipHtml,
    versionWords } from '../modules/evolution.js';
import { WS } from '../modules/ws.js';

const flush = async () => { for (let i = 0; i < 8; i++) await new Promise(setImmediate); };
const DAY = 86_400_000;
const at = (iso) => Date.parse(iso);

function version(tag, date, extra = {}) {
    const v = tag.replace(/^v/, '');
    return { tag, aliases: [], version: v, date, commit: `c-${v}`, level: 'patch', milestone: false,
        code_lines: 100, bible_bytes: 2048, system_bytes: 1024, release: { source: '', title: '', kind: '', heading: '', body: '' }, ...extra };
}

test('the model keeps unknown as unknown and puts the installation’s journals beside each version, not into it', () => {
    const model = historyModel({
        saved_at: '2026-10-21T08:20:00Z', refresh: { state: 'idle', unmeasured: 0 },
        versions: [version('v1.0.0', '2026-02-16T10:00:00Z', { level: 'major', milestone: true }),
            version('v1.0.1', '2026-02-17T10:00:00Z', { code_lines: null }),
            version('v1.1.0', 'not a date')],
        install: { identity: { at_versions: [null, 25901, 26000], first_at: '2026-02-16T12:00:00Z', latest: { at: '2026-10-20T09:00:00Z', chars: 26500 } },
            scratchpad: { at_versions: [null, null, null], first_at: '', latest: null } },
    });
    assert.deepEqual(model.versions.map((v) => [v.tag, v.i]), [['v1.0.0', 0], ['v1.0.1', 1]], 'a version without a date is left out');
    assert.deepEqual(model.versions.map((v) => v.values.code_lines), [100, null]);
    assert.deepEqual(model.versions.map((v) => v.values.identity), [null, 25901]);
    assert.deepEqual(model.milestones.map((v) => v.tag), ['v1.0.0']);
    assert.deepEqual(model.observed.identity, { t: at('2026-10-20T09:00:00Z'), v: 26500, at: '2026-10-20T09:00:00Z' });
    assert.equal(model.observed.scratchpad, null);
});

test('the time axis breaks at a long quiet stretch and keeps a burst of early tags readable', () => {
    const burst = [at('2026-02-16T10:00:00Z'), at('2026-02-16T20:00:00Z'), at('2026-02-18T14:00:00Z')];
    const later = ['2026-05-30', '2026-06-15', '2026-07-01', '2026-08-01', '2026-09-01', '2026-10-01', '2026-10-21']
        .map((day) => at(`${day}T09:00:00Z`));
    const axis = timeAxis([...later, ...burst], 900);
    assert.equal(axis.segments.length, 2);
    const [left, right] = axis.segments;
    assert.ok(left.x1 >= 110 && left.x1 < 200, `three days still get a readable width: ${left.x1}`);
    assert.equal(right.x0 - left.x1, 54, 'the gap is a gap');
    assert.ok(axis.x(burst[0]) < axis.x(burst[2]) && axis.x(burst[2]) < left.x1);
    assert.ok(axis.x(later[0]) > right.x0 && axis.x(later.at(-1)) < 900);
    assert.equal(axis.x(at('2026-04-01T00:00:00Z')), left.x1 + 27, 'a time inside the gap sits in its middle');
    const even = timeAxis([at('2026-06-01T00:00:00Z'), at('2026-07-01T00:00:00Z'), at('2026-08-01T00:00:00Z')], 600);
    assert.equal(even.segments.length, 1);
    assert.equal(even.gap, null);
    assert.ok(even.x(at('2026-06-01T00:00:00Z')) > 0 && even.x(at('2026-08-01T00:00:00Z')) < 600);
    assert.equal(timeAxis([], 600), null);
});

test('each measure speaks its own unit, and a missing value is said, never drawn as zero', () => {
    assert.equal(formatMeasure('code_lines', 1388331), '1.39M lines');
    assert.equal(formatMeasure('code_lines', 1388331, { exact: true }), '1,388,331 lines');
    assert.equal(formatMeasure('bible_bytes', 50739), '49.5 KiB');
    assert.equal(formatMeasure('identity', 25901), '25.9K chars');
    assert.equal(formatMeasure('scratchpad', 25901, { exact: true }), '25,901 characters');
    assert.equal(formatMeasure('code_lines', null), 'Not measured');
    assert.equal(formatMeasure('identity', null), 'Not recorded');
    assert.equal(formatDelta('code_lines', 1112820, 985000), '+13%');
    assert.equal(formatDelta('system_bytes', 29342, 25349), '+3.9 KiB');
    assert.equal(formatDelta('system_bytes', 25000, 70000), '−43.9 KiB');
    assert.equal(formatDelta('identity', 26000, 25000), '+1K chars');
    assert.equal(formatDelta('code_lines', 100, 100), 'no change');
    assert.equal(formatDelta('code_lines', 100, null), '');
});

test('a version is titled by its author’s short title, else its row heading, and its source kind is kept', () => {
    const titled = versionWords(version('v7.6.0', '2026-10-05T20:05:25Z', { level: 'minor', release: { source: 'readme',
        title: 'Memory as one ladder', kind: 'feat', heading: 'memory as one ladder, one interface language.', body: 'Memory keeps…' } }));
    assert.deepEqual(titled, { title: 'Memory as one ladder', lead: 'Memory as one ladder, one interface language', body: 'Memory keeps…',
        kind: 'Feature', authored: true });
    const legacy = versionWords(version('v7.7.1', '2026-10-21T08:31:27Z', { release: { source: 'readme', title: '', kind: 'fix(ci/ui)',
        heading: 'everything merged since 7.7.0.', body: '' } }));
    assert.deepEqual([legacy.title, legacy.lead, legacy.kind], ['Everything merged since 7.7.0', '', 'Fix']);
    assert.equal(versionWords(version('v6.9.0-rc.1', '2026-05-31T08:43:50Z', { level: 'pre', release: { source: 'readme', kind: 'rc(evolution)', heading: 'x' } })).kind, 'Pre-release');
    assert.equal(versionWords(version('v7.0.0', '2026-08-21T01:53:42Z', { level: 'major', release: { source: 'readme', kind: 'v7', heading: 'a modular runtime' } })).kind, 'Major release');
    const unnamed = versionWords(version('v6.18.0', '2026-06-04T10:00:00Z'));
    assert.deepEqual([unnamed.title, unnamed.authored], ['Version 6.18.0', false], 'the number is the page’s own words');
});

test('the chart marks a release author’s words as content and leaves the page’s own words to the interface language', () => {
    const model = historyModel({ versions: [
        version('v1.0.0', '2026-06-01T09:00:00Z', { level: 'major', milestone: true, code_lines: 100 }),
        version('v1.1.0', '2026-06-20T09:00:00Z', { level: 'minor', milestone: true, code_lines: 400,
            release: { source: 'tag', title: '', kind: '', heading: 'Web Presence + Budget Categorization', body: '- Landing page\n- /status' } }),
        version('v1.2.0', '2026-07-10T09:00:00Z', { level: 'minor', milestone: true, code_lines: 900,
            release: { source: 'readme', title: 'Memory as one ladder', kind: 'feat', heading: 'memory keeps pages.', body: '' } }),
    ] });
    const texts = (svg, cls) => [...svg.matchAll(new RegExp(`<text class="${cls}"([^>]*)>([^<]*)</text>`, 'g'))]
        .map(([, attrs, text]) => [text, attrs.includes('data-i18n-authored')]);
    const chosen = chartSvg(model, { key: 'code_lines', version: model.versions[1] }, 900);
    assert.deepEqual(texts(chosen.svg, 'evo-ch-label'), [['Web Presence + Budget Categorization', true]]);
    const labels = texts(chosen.svg, 'evo-ch-mlabel');
    assert.ok(labels.length >= 1, 'the chart names milestones where there is room');
    for (const [text, authored] of labels) assert.equal(authored, !text.startsWith('Version '), text);
    const unnamed = chartSvg(model, { key: 'code_lines', version: model.versions[0] }, 900);
    assert.deepEqual(texts(unnamed.svg, 'evo-ch-label'), [['Version 1.0.0', false]]);
});

test('the campaign object says one honest sentence per state', () => {
    assert.equal(campaignView(null), null);
    const off = campaignView({ runtime_mode: 'advanced', evolution_enabled: false, bg_consciousness_enabled: true,
        bg_consciousness_state: { status: 'sleeping', next_wake_at: '2026-10-21T08:40:00Z' } }, new Date('2026-10-21T08:00:00Z'));
    assert.deepEqual([off.chip.text, off.chip.tone, off.note, off.lines], ['Off', 'neutral', 'Improvement cycles run only during a campaign.', []]);
    assert.match(off.consciousness.text, /^Consciousness sleeping until \d\d:\d\d$/);
    assert.equal(off.consciousness.tone, 'ok');
    const light = campaignView({ runtime_mode: 'light', evolution_state: { status: 'disabled', enabled: false } });
    assert.equal(light.lightMode, true);
    assert.match(light.note, /Advanced or Pro/);
    const owner = campaignView({ evolution_state: { status: 'budget_stopped', enabled: false, detail: 'Evolution auto-stopped because only $1.00 remains.' } });
    assert.deepEqual([owner.chip.text, owner.chip.tone, owner.note], ['Budget', 'warn', 'Evolution auto-stopped because only $1.00 remains.']);
});

test('the campaign paints from the state while the history is still being read, and nothing survives disposal', async () => {
    const env = installDomDouble();
    const savedFetch = globalThis.fetch;
    const asked = [];
    globalThis.fetch = (url) => {
        asked.push(String(url));
        if (String(url).includes('evolution-data')) return new Promise(() => {});  // the history never answers here
        return Promise.resolve(Response.json({ runtime_mode: 'advanced', evolution_state: { status: 'disabled', enabled: false } }));
    };
    try {
        const mount = env.doc.createElement('div');
        env.doc.body.append(mount);
        const ws = new WS('ws://unused');
        const { dispose } = initEvolution({ ws, mount, state: { activePage: 'dashboard', dashboardActiveSubtab: 'evolution' } });
        ws.emit('open');
        await flush();
        assert.equal(env.doc.getElementById('evo-campaign-chip').textContent, 'Off');
        assert.equal(env.doc.getElementById('evo-start').disabled, false);
        assert.match(env.doc.body.querySelector('.evo-story-body').innerHTML, /Reading the saved history/);
        assert.deepEqual(asked, ['/api/state', '/api/evolution-data']);
        dispose();
        dispose();
        assert.equal(ws.listeners.open.size, 0);
        ws.emit('open');
        env.doc.defaultView.emit('ouro:dashboard-subtab-shown', { detail: { tab: 'evolution' } });
        await flush();
        assert.equal(asked.length, 2, 'a disposed panel asks nothing');
    } finally {
        globalThis.fetch = savedFetch;
        env.restore();
    }
});

test('the history is polled only while a refresh runs, and the poll stops with the panel', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const env = installDomDouble();
    const savedFetch = globalThis.fetch;
    let state = 'running';
    let reads = 0;
    globalThis.fetch = async (url) => {
        if (!String(url).includes('evolution-data')) return Response.json({});
        reads += 1;
        return Response.json({ versions: [], saved_at: '', refresh: { state }, install: {} });
    };
    try {
        const mount = env.doc.createElement('div');
        env.doc.body.append(mount);
        const ws = new WS('ws://unused');
        const { dispose } = initEvolution({ ws, mount, state: { activePage: 'dashboard', dashboardActiveSubtab: 'evolution' } });
        ws.emit('open');
        await flush();
        assert.equal(reads, 1);
        assert.match(env.doc.body.querySelector('.evo-story-body').innerHTML, /first measurement on this install/);
        t.mock.timers.tick(2000);
        await flush();
        assert.equal(reads, 2, 'a running refresh is asked about again');
        state = 'idle';
        t.mock.timers.tick(2000);
        await flush();
        assert.equal(reads, 3);
        t.mock.timers.tick(10_000);
        await flush();
        assert.equal(reads, 3, 'an idle answer stops the poll');
        state = 'running';
        ws.emit('open');
        await flush();
        dispose();
        t.mock.timers.tick(10_000);
        await flush();
        assert.equal(reads, 4, 'disposal stops a pending poll');
    } finally {
        globalThis.fetch = savedFetch;
        env.restore();
    }
});

test('a day of release tags stays one labelled segment', () => {
    const day = timeAxis([at('2026-02-16T10:00:00Z'), at('2026-02-16T22:00:00Z')], 400);
    assert.equal(day.segments.length, 1);
    assert.ok(day.segments[0].t1 - day.segments[0].t0 >= DAY);
});

test('the drawn line breaks at the axis gap and only the dashed bridge crosses it', () => {
    const burst = ['2026-02-16T10:00:00Z', '2026-02-16T22:00:00Z', '2026-02-18T14:00:00Z'];
    const later = ['2026-05-30', '2026-06-15', '2026-07-01', '2026-08-01', '2026-09-01', '2026-10-01'].map((d) => `${d}T09:00:00Z`);
    const model = historyModel({ versions: [...burst, ...later].map((date, i) => version(`v1.${i}.0`, date, { code_lines: 1000 + 100 * i })) });
    const chart = chartSvg(model, { key: 'code_lines', version: model.latest }, 900);
    const [gapLeft, gapRight] = chart.axis.gap;
    const path = (cls) => chart.svg.match(new RegExp(`class="${cls}" d="([^"]*)"`))?.[1] || '';
    const runs = path('evo-ch-line').split('M').filter(Boolean).map((run) => run.split('L').map((pair) => Number(pair.split(',')[0])));
    assert.equal(runs.length, 2, 'one run each side of the break');
    for (const xs of runs) assert.ok(xs.every((x) => x <= gapLeft) || xs.every((x) => x >= gapRight), `a run spans the gap: ${xs}`);
    const areas = path('evo-ch-area').split('M').filter(Boolean).map((run) => run.replace(/Z$/, '').split('L').map((pair) => Number(pair.split(',')[0])));
    assert.equal(areas.length, 2, 'each side fills its own area');
    for (const xs of areas) assert.ok(xs.every((x) => x <= gapLeft) || xs.every((x) => x >= gapRight), `an area spans the gap: ${xs}`);
    const bridge = path('evo-ch-bridge').split(/[ML]/).filter(Boolean).map((pair) => Number(pair.split(',')[0]));
    assert.ok(bridge.length === 2 && bridge[0] <= gapLeft && bridge[1] >= gapRight, 'the dashed bridge is what crosses');
});

test('a partly measured history says how many versions failed and offers Try again', async () => {
    const env = installDomDouble();
    const savedFetch = globalThis.fetch;
    globalThis.fetch = async (url) => Response.json(String(url).includes('evolution-data')
        ? { versions: [version('v1.0.0', '2026-02-16T10:00:00Z')], saved_at: '2026-10-21T08:20:00Z', install: {},
            refresh: { state: 'idle', unmeasured: 2, unmeasured_error: 'v7.7.1: git ls-tree fc8cd8146 failed (rc=128)' } }
        : { runtime_mode: 'advanced', evolution_state: { status: 'disabled', enabled: false } });
    try {
        const mount = env.doc.createElement('div');
        env.doc.body.append(mount);
        const ws = new WS('ws://unused');
        const { dispose } = initEvolution({ ws, mount, state: { activePage: 'dashboard', dashboardActiveSubtab: 'evolution' } });
        ws.emit('open');
        await flush();
        const status = env.doc.body.querySelector('.evo-story-status').innerHTML;
        assert.match(status, /data-tone="warn">2 versions couldn’t be measured: v7\.7\.1: git ls-tree fc8cd8146 failed \(rc=128\)</);
        assert.match(status, /data-retry>Try again</);
        dispose();
    } finally {
        globalThis.fetch = savedFetch;
        env.restore();
    }
});

test('milestone labels go where there is room: authored titles first, then MAJOR, then MINOR, newest first', () => {
    const pin = (v, level, x, y, t, title = '') => ({ x, y, version: { version: v, level, t, i: t, milestone: true,
        release: { title, heading: `release ${v} heading` } } });
    const pins = [
        pin('7.6.0', 'minor', 300, 60, 6, 'Memory as one ladder'),   // older, but its author named it
        pin('7.7.0', 'minor', 310, 58, 7),                           // newer, same place: loses to the title
        pin('7.0.0', 'major', 150, 90, 5),
        pin('6.10.0', 'minor', 40, 120, 2),                          // near the left edge: labelled to its right
        pin('6.11.0', 'minor', 600, 40, 3),                          // no room on the right: labelled to its left
    ];
    const taken = [{ x0: 140, x1: 260, y0: 60, y1: 80 }];             // e.g. the end value
    const placed = milestoneLabels(pins, taken, 640);
    const names = placed.map((label) => label.version.version);
    assert.equal(names[0], '7.6.0', 'the authored title is placed first');
    assert.ok(!names.includes('7.7.0'), 'a label that would touch another is not drawn; its dot stays');
    assert.ok(!names.includes('7.0.0'), 'a blocked area is respected');
    assert.equal(placed.find((label) => label.version.version === '7.6.0').text, 'Memory as one ladder');
    assert.equal(placed.find((label) => label.version.version === '6.10.0').anchor, 'start');
    assert.equal(placed.find((label) => label.version.version === '6.11.0').anchor, 'end', 'labelled on the side with room');
    const boxes = taken.slice(1);
    for (let a = 0; a < boxes.length; a += 1) {
        for (let b = a + 1; b < boxes.length; b += 1) {
            const [p, q] = [boxes[a], boxes[b]];
            assert.ok(!(p.x0 < q.x1 && q.x0 < p.x1 && p.y0 < q.y1 && q.y0 < p.y1), 'placed labels never overlap');
        }
    }
    assert.equal(pins.length, 5, 'placement never removes a milestone');
});

test('a milestone label stays off the drawn line: left of a rising line, right of a falling one, none in a valley', () => {
    const pin = { x: 300, y: 100, version: { version: '7.6.0', level: 'minor', t: 6, i: 6, milestone: true,
        release: { title: 'A named change', heading: 'release heading' } } };
    const line = (y0, y2) => [[{ x: 100, y: y0 }, { x: 300, y: 100 }], [{ x: 300, y: 100 }, { x: 500, y: y2 }]];
    assert.equal(milestoneLabels([pin], [], 640)[0].anchor, 'start', 'with nothing drawn the right is tried first');
    assert.equal(milestoneLabels([pin], [], 640, { segments: line(150, 40) })[0].anchor, 'end', 'rising: the right is crossed');
    assert.equal(milestoneLabels([pin], [], 640, { segments: line(40, 150) })[0].anchor, 'start', 'falling: the right is clear');
    assert.deepEqual(milestoneLabels([pin], [], 640, { segments: line(40, 40) }), [], 'both sides crossed: no label, the dot stays');
    const rule = [[{ x: 360, y: 60 }, { x: 360, y: 153 }]];                // the chosen version's rule, right of the dot
    assert.equal(milestoneLabels([pin], [], 640, { segments: rule })[0].anchor, 'end', 'a label never covers the chosen rule');
});

test('tag names and release words reach the page as text, never as markup', () => {
    // Rewritten from the static pin on the retired tags table (escapeHtmlText(p.tag)): the same rule, now as behaviour.
    const evil = '<img src=x onerror="window.__evo=1">';
    const model = historyModel({ versions: [
        version('v1.0.0', '2026-06-01T09:00:00Z', { level: 'major', milestone: true, code_lines: 100 }),
        version(`v1.1.0${evil}`, '2026-06-20T09:00:00Z', { level: 'minor', milestone: true, code_lines: 400,
            aliases: [`remote-v1.1.0${evil}`], release: { source: 'tag', title: `Named ${evil}`, kind: '', heading: `Heading ${evil}`,
                body: `Body ${evil}\n- & "quoted" 'too' \`ticks\`` } }),
        version('v1.2.0', '2026-07-10T09:00:00Z', { level: 'minor', milestone: true, code_lines: 900,
            release: { source: 'readme', title: '', kind: 'feat', heading: `the row ${evil}.`, body: '' } }),
    ] });
    const [plain, hostile, row] = model.versions;
    const html = [detailHtml(model, hostile, true), detailHtml(model, row, false), tipHtml(model, hostile, 'code_lines'),
        ...[plain, hostile, row].map((chosen) => chartSvg(model, { key: 'code_lines', version: chosen }, 900).svg)].join('\n');
    assert.doesNotMatch(html, /<img|onerror="/i, 'nothing the data says becomes markup');
    const text = '&lt;img src=x onerror=&quot;window.__evo=1&quot;&gt;';
    for (const said of [`v1.1.0${text}`, `remote-v1.1.0${text}`, `Named ${text}`, `Heading ${text}`, `Body ${text}`,
        '&amp; &quot;quoted&quot; &#39;too&#39; &#96;ticks&#96;', `The row ${text}`]) assert.ok(html.includes(said), said);
});
