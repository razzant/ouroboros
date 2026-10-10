// Review findings are most of the installed-list payload and were rendered into a
// hidden <details> on every full render. The card now renders that block collapsed
// with its summary only; the list is built on the first open from the row already
// in memory, and reads exactly as the eager renderer wrote it.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

import { renderInstalledSkillCard, renderReviewFindingsList } from '../modules/skill_card_renderer.js';

const demo = {
    name: 'demo', source: 'external', payload_root: 'skills/external/demo', content_hash: 'a'.repeat(64),
    version: '1.0.0', review_status: 'warnings', review_gate: { executable_review: true },
    permissions: [], grants: { all_granted: true },
    review_findings: [
        { item: 'exec', verdict: 'warn', reason: 'Spawns a shell' },
        { check: 'net', severity: 'info', message: 'Opens <sockets>' },
        { item: 'skill_preflight', verdict: 'fail', reason: JSON.stringify({ files: [{ path: 'main.py', ok: false, stderr: 'SyntaxError: bad token\nmore' }] }) },
    ],
};

test('a card renders its findings block collapsed, with the summary only', () => {
    const html = renderInstalledSkillCard(demo);
    assert.match(html, /<details class="skills-review-findings ui-rich-content" data-skill-findings="demo"><summary class="muted">3 review findings<\/summary><\/details>/);
    assert.doesNotMatch(html, /Spawns a shell|Opens|Preflight failed|<li>/, 'no finding text before the block is opened');
    assert.match(renderInstalledSkillCard({ ...demo, review_findings: demo.review_findings.slice(0, 1) }), /1 review finding</);
    assert.equal(renderInstalledSkillCard({ ...demo, review_findings: [] }).includes('skills-review-findings'), false);
});

test('the list built on open is what the eager renderer produced; a row gone from the list says so', () => {
    assert.equal(renderReviewFindingsList(demo), '<ul>'
        + '<li><strong>warn</strong> exec: Spawns a shell</li>'
        + '<li><strong>info</strong> net: Opens &lt;sockets&gt;</li>'
        + '<li><strong>fail</strong> skill_preflight: Preflight failed — main.py: SyntaxError: bad token</li>'
        + '</ul>');
    assert.match(renderReviewFindingsList(undefined), /not in the current list/);
    assert.match(renderReviewFindingsList({ ...demo, review_findings: [] }), /not in the current list/);
});

test('opening the block builds its list once from the row in memory; closing or reopening builds nothing', () => {
    const text = readFileSync(new URL('../modules/skills.js', import.meta.url), 'utf8');
    const start = text.indexOf('function attachActionHandlers('), end = text.indexOf('\nfunction activateTab', start);
    assert.ok(start >= 0 && end > start);
    const container = { handlers: {}, addEventListener(name, handler, capture) { this.handlers[name] = { handler, capture }; } };
    const context = vm.createContext({ skillsSnapshot: { rawSkills: [demo] }, renderReviewFindingsList });
    vm.runInContext(text.slice(start, end), context);
    context.attachActionHandlers(container, () => {}, new Set(), new Set());
    assert.equal(container.handlers.toggle.capture, true, 'toggle does not bubble: the container listens in the capture phase');
    const toggle = container.handlers.toggle.handler;
    const details = (open, built = false) => ({
        open, dataset: { skillFindings: 'demo' }, childElementCount: built ? 2 : 1, inserted: [],
        insertAdjacentHTML(where, markup) { this.inserted.push([where, markup]); },
    });
    const closed = details(false);
    toggle({ target: closed });
    assert.deepEqual(closed.inserted, []);
    const opened = details(true);
    toggle({ target: opened });
    assert.deepEqual(opened.inserted, [['beforeend', renderReviewFindingsList(demo)]]);
    const again = details(true, true);
    toggle({ target: again });
    assert.deepEqual(again.inserted, [], 'a built list survives re-opening and card patches untouched');
    const gone = { ...details(true), dataset: { skillFindings: 'vanished' } };
    toggle({ target: gone });
    assert.match(gone.inserted[0][1], /not in the current list/);
    toggle({ target: { open: true, dataset: {} } }); // other <details> blocks are not ours
});
