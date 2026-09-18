import assert from 'node:assert/strict';
import test from 'node:test';

import {
    COLLAPSED_ACTIVITY_MAX,
    boundActivityPreview,
    clearStickyCardState,
    projectCollapsedActivity,
} from '../modules/chat.js';
import { setReasoningVisible, summarizeChatLiveEvent } from '../modules/log_events.js';
import { bindContentButton, isLiveLineExpandable, plainActivityText, selectionInside, subagentIdentityTitle, subagentTwin } from '../modules/chat_activity.js';

test('named root card shows the latest activity headline under the coined title', () => {
    assert.equal(projectCollapsedActivity({
        suggestedName: 'Data Analysis',
        headline: 'Analyzing the dataset',
    }), 'Analyzing the dataset');
});

test('root suppresses duplicated title but can show distinct previous activity before naming)', () => {
    assert.equal(projectCollapsedActivity({
        suggestedName: '',
        headline: 'Analyzing the dataset',
    }), '');
    // No title has replaced the previous activity, so useful existing text remains.
    assert.equal(projectCollapsedActivity({
        suggestedName: '',
        headline: '',
        previous: 'Earlier step',
    }), 'Earlier step');
});

test('subagent card always feeds the line from the routed progress body', () => {
    assert.equal(projectCollapsedActivity({
        isSubagent: true,
        body: 'Running the migration script',
    }), 'Running the migration script');
    // No coined name is required — the subagent title keeps role·model·id.
    assert.equal(projectCollapsedActivity({
        isSubagent: true,
        suggestedName: '',
        body: 'Collecting evidence',
    }), 'Collecting evidence');
});

test('a frame without new activity keeps the previous text (Done never blanks it)', () => {
    assert.equal(projectCollapsedActivity({
        suggestedName: 'Data Analysis',
        headline: '',
        previous: 'Analyzing the dataset',
    }), 'Analyzing the dataset');
    assert.equal(projectCollapsedActivity({
        isSubagent: true,
        body: '',
        previous: 'Running the migration script',
    }), 'Running the migration script');
});

test('whitespace-only frames fall back to the previous activity', () => {
    assert.equal(projectCollapsedActivity({
        suggestedName: 'X',
        headline: '   ',
        previous: 'Real step',
    }), 'Real step');
});

test('clearStickyCardState resets the recycled record activity + cost (reusable slots)', () => {
    const record = {
        collapsedActivity: 'Old cycle activity',
        costMeta: { meta: ['$1.00'], ts: 1, final: true },
        executorChip: { harness: 'codex', icon: '◇', label: 'codex · no run yet' },
        // Models the real element closely enough for attribute handling.
        activityEl: {
            textContent: 'Old cycle activity',
            title: '',
            removeAttribute(name) { if (name === 'title') this.title = ''; },
        },
    };
    record.latestActivityTs = '12:00:00';
    record.activityEl.title = 'Old cycle activity';
    clearStickyCardState(record);
    assert.equal(record.collapsedActivity, '');
    assert.equal(record.costMeta, null);
    // The sticky executor chip is cycle state: a recycled slot must not claim
    // the previous cycle's delegated route as its own.
    assert.equal(record.executorChip, null);
    assert.equal(record.activityEl.textContent, '');
    // The activity clock is cycle state too.
    assert.equal(record.latestActivityTs, '');
    assert.equal(record.activityEl.title, '');
});

test('a terminal subagent keeps its last narration as collapsed activity (replay)', () => {
    // On history replay terminal children are pre-marked before pass 1, so the
    // card is never re-driven through a working frame; the projection must
    // still return the remembered narration for the collapsed line.
    assert.equal(projectCollapsedActivity({
        isSubagent: true, body: 'Collecting evidence', previous: 'Collecting evidence',
    }), 'Collecting evidence');
    // A later empty frame does not blank it.
    assert.equal(projectCollapsedActivity({
        isSubagent: true, body: '', previous: 'Collecting evidence',
    }), 'Collecting evidence');
});

test('the collapsed projection is whitespace-normalized, bounded and explicit', () => {
    const long = 'x'.repeat(COLLAPSED_ACTIVITY_MAX + 250);
    const out = boundActivityPreview(long);
    assert.equal(out.length, COLLAPSED_ACTIVITY_MAX);
    assert.ok(out.endsWith('…'), 'the cut is visible, never silent');
    assert.equal(boundActivityPreview('  Reading\n  the   ledger  '), 'Reading the ledger');
    assert.equal(projectCollapsedActivity({ suggestedName: 'X', headline: long }), out);
});

test('root progress keeps a bounded activity preview and a complete timeline companion', () => {
    const full = `Inspecting evidence\n${'long detail '.repeat(60)}UNIQUE_ROOT_TAIL`;
    const summary = summarizeChatLiveEvent({
        type: 'send_message', is_progress: true, task_id: 'root-1', content: full,
    });
    assert.ok(summary.activityPreview.length <= COLLAPSED_ACTIVITY_MAX);
    assert.match(summary.activityPreview, /^Inspecting evidence/);
    assert.equal(summary.fullHeadline, full);
    assert.match(summary.fullHeadline, /UNIQUE_ROOT_TAIL$/);
});

test('subagent projection keeps identity, compact facts and complete disclosure', () => {
    const full = `${'Collecting evidence '.repeat(30)}UNIQUE_CHILD_TAIL`;
    const summary = summarizeChatLiveEvent({
        type: 'send_message',
        is_progress: true,
        delegation_role: 'subagent',
        subagent_task_id: 'child123456',
        parent_task_id: 'parent1',
        root_task_id: 'root1',
        subagent_role: 'researcher',
        model: 'anthropic/claude-fable-5',
        subagent_event: 'running',
        content: full,
        write_surface: 'workspace',
        status: 'running',
    });
    // Identity only: the status lives in the chip and the id shows only for twins.
    assert.equal(summary.headline, 'researcher');
    // A roleless child is `Subagent · model`: the id is a render-time twin tag, never identity.
    assert.equal(summarizeChatLiveEvent({
        type: 'send_message', is_progress: true, delegation_role: 'subagent',
        subagent_task_id: 'child987654', parent_task_id: 'parent1', model: 'openai/gpt-5.6-sol',
        subagent_event: 'running', content: 'x', status: 'running',
    }).headline, 'Subagent');
    // chat.js writes the child title from the lineage map; it must read like the reducer's headline.
    assert.equal(subagentIdentityTitle({ role: 'researcher', model: 'anthropic/claude-fable-5' }), 'researcher');
    assert.equal(subagentIdentityTitle({ role: '', model: 'openai::gpt-5.6-sol' }), 'Subagent');
    assert.equal(subagentIdentityTitle({ role: 'planner', model: '' }), 'planner');
    assert.ok(summary.activityPreview.length <= COLLAPSED_ACTIVITY_MAX);
    assert.match(summary.fullBody, /UNIQUE_CHILD_TAIL$/);
    assert.deepEqual(summary.meta || [], []);
});

test('the collapsed activity line is plain text: the renderer\'s markdown inventory', () => {
    // The expanded timeline renders the same headline through renderMarkdown, so
    // the compact line strips that marker inventory (line by line, over-strip
    // preferred to a leaked marker).
    assert.equal(plainActivityText('**Planning a network update** I need `git fetch`'),
        'Planning a network update I need git fetch');
    // A heading keeps ' — ' before the text under it (any number of blank lines);
    // a heading with nothing under it is just its text.
    assert.equal(plainActivityText('### Title\n- one\n- two [link](http://x)'), 'Title —\none\ntwo link');
    assert.equal(plainActivityText('## summary\n\nCursor Grok confirmed'), 'summary —\n\nCursor Grok confirmed');
    assert.equal(plainActivityText('## Only'), 'Only');
    assert.equal(plainActivityText('## Trailing   \nnext'), 'Trailing —\nnext');
    // Continuation may be indented, come after whitespace-only lines, or use CRLF.
    assert.equal(plainActivityText('## summary\r\ncontinued'), 'summary —\ncontinued');
    assert.equal(plainActivityText('## summary\n   continued'), 'summary —\n   continued');
    assert.equal(plainActivityText('## A\n   \nB'), 'A —\n\nB');
    // A marker-led line the renderer treats as prose (over MARKDOWN_HEADING_MAX_CHARS)
    // loses its markers but gets no separator.
    const prose = 'x'.repeat(85);
    assert.equal(plainActivityText(`## ${prose}\nnext`), `${prose}\nnext`);
    // A `#` line inside a fence is code: no separator, and its markers stay.
    assert.equal(plainActivityText('```md\n## not a heading\n```'), '## not a heading');
    assert.equal(plainActivityText('```sh\n# comment\nls -la\n```\nafter'), '# comment\nls -la\n\nafter');
    // A heading that already ends in a dash or colon gets no second separator.
    assert.equal(plainActivityText('## summary —\nnext'), 'summary —\nnext');
    assert.equal(plainActivityText('## Steps:\nnext'), 'Steps:\nnext');
    assert.equal(plainActivityText('## **Steps:**\nnext'), 'Steps:\nnext');
    const styled = summarizeChatLiveEvent({ is_progress: true, content: '## **Steps:**\nnext', task_id: 'p3', chat_id: 1 });
    assert.equal(boundActivityPreview(styled.headline), 'Steps: next');
    // Typed shell text is not markdown: a `# comment` in a failed command's compact
    // diagnostic keeps its `#` and gets no separator.
    const failed = summarizeChatLiveEvent({ type: 'tool_call_finished', is_error: true, status: 'non_zero_exit', exit_code: 1,
        args: { cmd: '# explain\nfalse' }, task_id: 'p4', chat_id: 1 });
    assert.ok(failed.body.includes('Command: # explain false'), failed.body);
    assert.ok(!failed.body.includes(' —'), failed.body);
    // Linear on hostile whitespace: a 40k-space run inside a marker line.
    const started = Date.now();
    plainActivityText(`# a${' '.repeat(40000)}b\nnext`);
    assert.ok(Date.now() - started < 500, 'heading projection must stay linear');
    assert.equal(boundActivityPreview('## summary\n\nCursor Grok confirmed the skew'), 'summary — Cursor Grok confirmed the skew');
    // The live preview path: the reducer collapses newlines in describeText, so the
    // heading rule must have run before that — a non-leading heading keeps no marker.
    const live = summarizeChatLiveEvent({ is_progress: true, content: 'text\n## Second\nbeta', task_id: 'p1', chat_id: 1 });
    assert.equal(live.headline, 'text Second — beta');
    assert.equal(boundActivityPreview(live.headline), 'text Second — beta');
    // A fenced `##` line — with text inside and after the fence — stays code on the
    // live path too: markers kept, no separator (the fence itself collapses to the
    // renderer's code-span remnant, which is not this rule's contract).
    // A heading right before an UNCLOSED opener is followed by text (the opener is text).
    const before = summarizeChatLiveEvent({ is_progress: true, content: '## Title\n```md\nmore', task_id: 'p5', chat_id: 1 });
    assert.equal(boundActivityPreview(before.headline), 'Title — `md more');
    for (const content of ['```md\n## code\n```\nafter', '```md\n## not a heading\nnext\n```', '   ```md\n## code\n```\nafter', '  ```md\n## code\n  ```\nafter', 'prefix ```md\n## code\n```\nafter']) {
        const fenced = summarizeChatLiveEvent({ is_progress: true, content, task_id: 'p2', chat_id: 1 });
        for (const text of [fenced.headline, boundActivityPreview(fenced.headline), plainActivityText(content)]) {
            assert.ok(text.includes('## code') || text.includes('## not a heading'), text);
            assert.ok(!text.includes(' —'), text);
        }
    }
    assert.equal(plainActivityText('~~old~~ *new*'), 'old new');
    assert.equal(plainActivityText('#### Deep\n``x → y`` tail'), 'Deep —\nx → y tail');
    assert.equal(plainActivityText('```js\nlet a = 1;\n```'), 'let a = 1;');
    assert.equal(boundActivityPreview('| a | b |\n|---|---|\n| 1 | 2 |'), 'a b 1 2');
    // Markers-only text keeps its source: an empty projection would hide the
    // card's activity band.
    assert.equal(plainActivityText('---'), '---');
    // Whitespace-only narration projects to nothing so :empty hides the band.
    assert.equal(boundActivityPreview(' \n\t '), '');
    assert.equal(plainActivityText(''), '');
    // Composition: the bound preview is built on the plain projection.
    assert.equal(boundActivityPreview('  **Reading**\n  the   ledger  '), 'Reading the ledger');
});

test('twins share one parent and displayed role while model stays metadata', () => {
    const children = new Map([
        ['a', { parentId: 'p', role: 'scout', model: 'gemini-3.6-flash' }],
        ['b', { parentId: 'p', role: 'scout', model: 'openai/gpt-5.6-sol' }],
        ['c', { parentId: 'p', role: 'reviewer', model: 'gemini-3.6-flash' }],
        ['d', { parentId: 'q', role: 'scout', model: 'gemini-3.6-flash' }],
    ]);
    assert.equal(subagentTwin(children, 'a'), true);
    assert.equal(subagentTwin(children, 'b'), true);
    assert.equal(subagentTwin(children, 'c'), false);
    assert.equal(subagentTwin(children, 'd'), false);
    assert.equal(subagentTwin(children, 'missing'), false);
    // The collision key is the displayed headline; a roleless child and an
    // explicitly named Subagent therefore collide even with different models.
    const spelled = new Map([
        ['e', { parentId: 'p', role: 'scout', model: 'openai/gpt-5.6-sol' }],
        ['f', { parentId: 'p', role: 'scout', model: 'openai::gpt-5.6-sol' }],
        ['g', { parentId: 'p', role: '', model: 'gpt-5.6-sol' }],
        ['h', { parentId: 'p', role: 'Subagent', model: 'other-model' }],
    ]);
    assert.equal(subagentTwin(spelled, 'e'), true);
    assert.equal(subagentTwin(spelled, 'f'), true);
    assert.equal(subagentTwin(spelled, 'g'), true);
    assert.equal(subagentTwin(spelled, 'h'), true);
});

test('a selection inside the surface means the reader is copying, not clicking', () => {
    const inside = {}; const outside = {};
    const el = { contains: (node) => node === inside };
    assert.equal(selectionInside(el, { isCollapsed: false, anchorNode: inside }), true);
    assert.equal(selectionInside(el, { isCollapsed: true, anchorNode: inside }), false);
    assert.equal(selectionInside(el, { isCollapsed: false, anchorNode: outside }), false);
    assert.equal(selectionInside(el, null), false);
});

test('a content button toggles on click and keyboard, but not on a selecting drag', () => {
    const handlers = {}; let clicks = 0; let activated = 0;
    const inside = {};
    const el = {
        addEventListener: (type, fn) => { handlers[type] = fn; },
        click: () => { clicks += 1; handlers.click({ detail: 0 }); },
        contains: (node) => node === inside,
    };
    bindContentButton(el, () => { activated += 1; });
    const saved = globalThis.getSelection;
    globalThis.getSelection = () => ({ isCollapsed: false, anchorNode: inside });
    handlers.click({ detail: 1 });
    assert.equal(activated, 0, 'a pointer click after a selecting drag is a copy, not a toggle');
    globalThis.getSelection = () => ({ isCollapsed: true, anchorNode: inside });
    handlers.click({ detail: 1 });
    assert.equal(activated, 1);
    let prevented = 0;
    handlers.keydown({ key: 'Enter', preventDefault: () => { prevented += 1; } });
    handlers.keydown({ key: ' ', preventDefault: () => { prevented += 1; } });
    handlers.keydown({ key: 'a', preventDefault: () => { prevented += 1; } });
    assert.deepEqual([clicks, activated, prevented], [2, 3, 2]);
    globalThis.getSelection = saved;
});


test('a reasoning-stamped progress frame is its own collapsed Thinking line', () => {
    const reasoning = 'Weighing the two migration paths before touching the schema. '.repeat(6).trim();
    const frame = { type: 'send_message', is_progress: true, task_id: 't1', ts: '2026-09-11T10:00:00Z', content: `💬 ${reasoning}` };
    // The display preference is off by default; this is the shown state.
    setReasoningVisible(true);
    const view = summarizeChatLiveEvent({ ...frame, reasoning: true });
    assert.equal(view.phase, 'thinking');
    assert.equal(view.headline, 'Thinking');
    assert.equal(view.fullBody, reasoning);
    assert.ok(view.body.length < reasoning.length && view.body.endsWith('...'));
    // The collapsed card summary keeps the last action; reasoning never promotes.
    assert.equal(view.activityPreview, '');
    assert.equal(view.human, false);
    assert.equal(view.promote, false);
    assert.equal(view.visible, true);
    assert.equal(view.dedupeKey, `reasoning:2026-09-11T10:00:00Z:${reasoning}`);
    assert.equal(isLiveLineExpandable(view), true);
    // The same frame without the stamp renders exactly as before.
    const plain = summarizeChatLiveEvent(frame);
    assert.equal(plain.phase, 'working');
    assert.equal(plain.human, true);
    assert.equal(plain.headline, view.body);
    setReasoningVisible(false);
});

test('reasoning hidden: the stamped frame renders no chat timeline line', () => {
    setReasoningVisible(false);
    const reasoning = 'Weighing the two migration paths before touching the schema.';
    const frame = { type: 'send_message', is_progress: true, task_id: 't1', ts: '2026-09-11T10:00:00Z', content: `💬 ${reasoning}` };
    const hidden = summarizeChatLiveEvent({ ...frame, reasoning: true });
    assert.equal(hidden.visible, false);
    // The body still travels, so flipping the preference on reveals the same row.
    assert.equal(hidden.fullBody, reasoning);
    // An unstamped frame on the same surface stays visible.
    assert.equal(summarizeChatLiveEvent(frame).visible, true);
    setReasoningVisible(true);
    assert.equal(summarizeChatLiveEvent({ ...frame, reasoning: true }).visible, true);
    setReasoningVisible(false);
    assert.equal(summarizeChatLiveEvent({ ...frame, reasoning: true }).visible, false);
});
