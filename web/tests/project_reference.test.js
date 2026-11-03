import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { PAGE_ICONS } from '../modules/page_icons.js';
import { nameProjectReference, projectReference } from '../modules/project_reference.js';

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// Minimal element stub: the reference is built with createElement/append/textContent,
// so a flat stub proves the DOM contract without a browser.
class NodeStub {
    constructor(tag) {
        this.tagName = tag.toUpperCase();
        this.children = [];
        this.attributes = {};
        this.dataset = {};
        this.listeners = new Map();
        this.className = '';
        this.innerHTML = '';
        this.title = '';
        this._text = '';
    }
    set textContent(value) { this._text = String(value ?? ''); }
    get textContent() { return this._text; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    append(...nodes) { this.children.push(...nodes); }
    addEventListener(type, fn) { this.listeners.set(type, fn); }
    querySelector(selector) { return this.children.find((node) => `.${node.className}` === selector) || null; }
}

function withDom(fn) {
    const prior = { document: globalThis.document, window: globalThis.window, CustomEvent: globalThis.CustomEvent };
    const sent = [];
    globalThis.document = { createElement: (tag) => new NodeStub(tag) };
    globalThis.CustomEvent = class { constructor(type, init) { this.type = type; this.detail = init?.detail; } };
    globalThis.window = { dispatchEvent: (event) => { sent.push({ type: event.type, detail: event.detail }); } };
    try { return fn(sent); } finally { Object.assign(globalThis, prior); }
}

const shape = (node) => ({
    tag: node.tagName, type: node.type, intent: node.dataset.intent,
    parts: node.children.map((child) => child.className),
    icon: node.children[0].innerHTML, iconHidden: node.children[0].attributes['aria-hidden'],
    arrowHidden: node.children[2].attributes['aria-hidden'],
});

test('one intent, one control: both forms are the same node with the same name and the same press', () => {
    withDom((sent) => {
        const project = { id: 'launch', name: 'OpenClaw 2.0 <b>x</b>', chat_id: 42 };
        const inline = projectReference(project);
        const footer = projectReference(project, { layout: 'footer' });
        // The callers pick one of two forms; the classes are the door's own business.
        // The capsule keeps a finger's target wherever its row makes it shorter (a question's voice line, a work
        // portal's action row).
        assert.equal(inline.className, 'chat-live-project-card-btn chat-quiz-project ui-hit-area');
        assert.equal(footer.className, 'chat-live-project-card-btn chat-live-bound-pointer ui-hit-area');
        assert.deepEqual(shape(inline), shape(footer));
        assert.deepEqual(shape(inline), {
            tag: 'BUTTON', type: 'button', intent: 'open-project',
            parts: ['chat-live-project-icon', 'chat-live-project-name', 'chat-live-project-status'],
            icon: PAGE_ICONS.projects, iconHidden: 'true', arrowHidden: 'true',
        });
        // The name is text: a Project name can never inject markup.
        for (const node of [inline, footer]) assert.equal(node.children[1].textContent, 'OpenClaw 2.0 <b>x</b>');
        // The arrow alone says "opens", in both forms.
        for (const node of [inline, footer]) assert.equal(node.children[2].textContent, '↗');
        // Phones and the Telegram shell have no hover: the accessible name says everything the pixels do.
        for (const node of [inline, footer]) {
            assert.equal(node.attributes['aria-label'], 'Open project OpenClaw 2.0 <b>x</b>');
            assert.equal(node.title, node.attributes['aria-label']);
        }
        // A key's click has no detail; a pointer's counts its press, and the room it opens stays quiet.
        inline.listeners.get('click')({ detail: 0 });
        footer.listeners.get('click')({ detail: 1 });
        assert.deepEqual(sent, [false, true].map((pointer) => ({
            type: 'ouro:open-project', detail: { project, task_id: '', quiz_id: '', pointer },
        })));
    });
});

test('a question reference opens that exact question; any other layout or a state word is the plain inline capsule', () => {
    withDom((sent) => {
        const node = projectReference({ id: 'launch', name: 'Launch', chat_id: 7 }, { taskId: 't1', quizId: 'q1' });
        assert.equal(node.attributes['aria-label'], 'Open this question in Launch');
        node.listeners.get('click')({ detail: 1 });
        assert.deepEqual(sent[0].detail, { project: { id: 'launch', name: 'Launch', chat_id: 7 }, task_id: 't1', quiz_id: 'q1',
            pointer: true });
        // Two forms only: the retired `bar` layout and `background` words, or any other, read as the inline capsule.
        for (const [layout, state] of [['bar', 'background'], ['sidebar', 'paused']]) {
            const odd = projectReference({ id: 'launch', name: 'Launch' }, { layout, state });
            assert.equal(odd.className, 'chat-live-project-card-btn chat-quiz-project ui-hit-area');
            assert.equal(odd.children[2].textContent, '↗');
            assert.equal(odd.attributes['aria-label'], 'Open project Launch');
            assert.equal(odd.dataset.state, undefined);
        }
    });
});

test('the host and the reference share one never-named rule (project_display_name_parity.json)', () => {
    const { cases } = JSON.parse(fs.readFileSync(path.join(WEB, 'tests/fixtures/project_display_name_parity.json'), 'utf8'));
    withDom(() => {
        for (const { id, name, expected } of cases) {
            assert.equal(projectReference({ id, name }).children[1].textContent, expected || 'Project', `${id}/${name}`);
        }
    });
});

test('a Project that was never named reads Project, and a name that arrives later reaches text, speech and the press', () => {
    withDom((sent) => {
        // The registry names an unnamed row by its minted id; that shape alone is a non-name.
        const unnamed = projectReference({ id: 'proj_5e8bbf266c89', name: 'proj_5e8bbf266c89' });
        assert.equal(unnamed.children[1].textContent, 'Project');
        assert.equal(unnamed.attributes['aria-label'], 'Open project Project');
        // An owner's own lowercase name derives the same id, and it IS the name.
        for (const own of ['blog', 'my-app', 'proj_notes']) {
            assert.equal(projectReference({ id: own, name: own }).children[1].textContent, own);
        }
        const late = projectReference({ id: 'launch', chat_id: 3 }, { quizId: 'q', taskId: 't' });
        assert.equal(late.children[1].textContent, 'Project');
        nameProjectReference(late, { id: 'launch', name: 'Launch' });
        assert.equal(late.children[1].textContent, 'Launch');
        assert.equal(late.title, 'Open this question in Launch');
        assert.equal(late.attributes['aria-label'], 'Open this question in Launch');
        late.listeners.get('click')({ detail: 0 });
        assert.deepEqual(sent[0].detail.project, { id: 'launch', chat_id: 3, name: 'Launch' });
    });
});

test('beside a title that already says the Project\'s name the capsule shows Project; the title\'s owner names it again (owner 26.10, B)', () => {
    withDom(() => {
        const made = [];
        const prior = globalThis.MutationObserver;
        globalThis.MutationObserver = class { constructor() { made.push(this); } observe() {} disconnect() {} };
        try {
            const title = new NodeStub('span');
            title.textContent = 'Audit  the projects';
            const node = projectReference({ id: 'p-1', name: 'audit the Projects' }, { layout: 'footer', taskId: 't-1', beside: title });
            assert.equal(node.children[1].textContent, 'Project', 'the same words, case and spacing aside, are not said twice');
            // What the glyph and the arrow say, words say in full: the tooltip and the accessible name keep the name.
            assert.equal(node.attributes['aria-label'], 'Open project audit the Projects');
            assert.equal(node.title, node.attributes['aria-label']);
            // No watch on the title, so nothing to dispose (DEVELOPMENT "UI resources carry a disposer"): whoever renames
            // the title names the reference again, from the reference's own target.
            assert.equal(made.length, 0, 'the reference acquires no observer');
            title.textContent = 'Audit the projects again';
            nameProjectReference(node);
            assert.equal(node.children[1].textContent, 'audit the Projects', 'a title that says something else leaves the name');
            assert.equal(node.attributes['aria-label'], 'Open project audit the Projects');
            title.textContent = 'Audit the Projects';
            nameProjectReference(node);
            assert.equal(node.children[1].textContent, 'Project');
            // A Project named apart from its work, and any reference with no title beside it, shows its name.
            assert.equal(projectReference({ id: 'p-2', name: 'Audit room' }, { layout: 'footer', beside: title }).children[1].textContent,
                'Audit room');
            assert.equal(projectReference({ id: 'p-1', name: 'Audit the Projects' }).children[1].textContent, 'Audit the Projects');
            assert.equal(made.length, 0);
        } finally { globalThis.MutationObserver = prior; }
    });
});

test('the door is the only place that raises the event or builds the control', () => {
    const sources = [path.join(WEB, 'app.js'), ...fs.readdirSync(path.join(WEB, 'modules'))
        .filter((name) => name.endsWith('.js')).map((name) => path.join(WEB, 'modules', name))];
    const raised = /CustomEvent\(\s*['"`]ouro:open-project['"`]/;
    // Code only: a comment may name the event. An event name held in a constant is out of this scan's reach.
    const code = (file) => fs.readFileSync(file, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
    const raisers = sources.filter((file) => raised.test(code(file)));
    assert.deepEqual(raisers.map((file) => path.basename(file)), ['project_reference.js']);
    // The other half of the wire stays where it was: one listener, in the shell.
    const app = fs.readFileSync(path.join(WEB, 'app.js'), 'utf8');
    assert.equal((app.match(/addEventListener\('ouro:open-project'/g) || []).length, 1);
    // The superseded factories are gone, so no caller can pass a label or a class name again.
    const helpers = fs.readFileSync(path.join(WEB, 'modules', 'ui_helpers.js'), 'utf8');
    assert.doesNotMatch(helpers, /export function (renderProjectChip|createSystemMessageAction)\b/);
    assert.match(helpers, /export function createSystemMessageActions\(/);
});

test('one recipe in two forms: no outline, hairline, ring or tile, no width of its own, hover only for a hovering pointer', () => {
    // Owner 17.10 (Q2 = C): the receipt keeps its capsule, and the capsule reads as one chip.
    const css = fs.readFileSync(path.join(WEB, 'style.css'), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
    const REFERENCE = /chat-quiz-project|chat-live-project-|chat-live-bound-pointer/;
    const rules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map(([, sel, body]) => ({ sel: sel.trim(), body }))
        .filter(({ sel }) => REFERENCE.test(sel));
    assert.ok(rules.length >= 8, 'the recipe is found');
    for (const { sel, body } of rules) {
        assert.doesNotMatch(body, /border(-top)?\s*:\s*1px/, `${sel}: no outline, hairline or ring`);
        assert.doesNotMatch(body, /max-width\s*:[^;]*ch\b/, `${sel}: no width chosen by its surroundings`);
        assert.doesNotMatch(body, /text-decoration\s*:\s*underline/, `${sel}: no underline-only state`);
        if (sel.includes('chat-live-project-icon')) assert.doesNotMatch(body, /background/, `${sel}: one fill, no tile`);
    }
    // Only the recipe's own selectors size the capsule; the one placement that does is a question's voice line,
    // whose 28px row and meta size it takes beside `Ouroboros` (owner 19.10, A). No other ancestor restyles it.
    assert.deepEqual([...new Set(rules.map(({ sel }) => sel).filter((sel) => /^\.(system-message-actions|msg-routing-annotation|chat-bubble)/.test(sel)))],
        ['.chat-bubble-voice > .chat-quiz-project.chat-live-project-card-btn', '.chat-bubble-voice > .chat-quiz-project .chat-live-project-name']);
    for (const { index } of css.matchAll(/\.(chat-live-project-card-btn|chat-quiz-project|chat-live-bound-pointer)[^{},]*:hover/g)) {
        const before = css.slice(0, index);
        const media = before.lastIndexOf('@media');
        assert.ok(media >= 0 && /^@media \(hover: hover\)/.test(before.slice(media)) && !/\n\}/.test(before.slice(media)),
            'a reference hover is gated by (hover: hover)');
    }
});
