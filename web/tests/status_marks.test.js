// Status marks (docs/DESIGN.md "Status marks"): one mapping from facts to a mark, one painter.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import * as acorn from './vendor/acorn.mjs';

import { MARK_HUES, MARK_SHAPES, markView, paintStatusMark, statusMarkHtml, syncMarkPhases } from '../modules/status_mark.js';
import { summarizeRoom } from '../modules/project_activity.js';
import { roomStatusView } from '../modules/work_status.js';
import { room } from './work_fixture.js';

test('the owner move outranks everything and stands alone as the amber dot', () => {
    for (const mark of ['moving', 'still', 'pause', 'ring', 'none']) {
        assert.deepEqual(markView({ mark, owner: true, unread: true, outcome: 'done' }), { shape: 'dot', hue: 'owner' });
    }
});

test('new messages tint the work shape, and are a crimson dot where nothing works', () => {
    assert.deepEqual(markView({ mark: 'moving', unread: true }), { shape: 'moving', hue: 'unread' });
    assert.deepEqual(markView({ mark: 'pause', unread: true }), { shape: 'pause', hue: 'unread' });
    assert.deepEqual(markView({ mark: 'still', unread: true }), { shape: 'still', hue: 'unread' });
    assert.deepEqual(markView({ mark: 'ring', unread: true }), { shape: 'ring', hue: 'unread' });
    assert.deepEqual(markView({ mark: 'none', unread: true }), { shape: 'dot', hue: 'unread' });
});

test('without anything for the owner the shape takes the surface ink, and rest draws nothing', () => {
    for (const mark of ['moving', 'still', 'pause', 'ring']) assert.deepEqual(markView({ mark }), { shape: mark, hue: 'ink' });
    assert.deepEqual(markView({ mark: 'none' }), { shape: 'none', hue: 'ink' });
    assert.deepEqual(markView({}), { shape: 'none', hue: 'ink' });
    // An unknown or legacy token is never drawn as something it is not.
    assert.deepEqual(markView({ mark: 'dot' }), { shape: 'none', hue: 'ink' });
    assert.deepEqual(markView({ mark: 'sparkle' }), { shape: 'none', hue: 'ink' });
});

test('an outcome beside its word is a dot in its tone: Done with warnings succeeds and Stopped is neutral', () => {
    assert.deepEqual(markView({ outcome: 'done' }), { shape: 'dot', hue: 'ok' });
    assert.deepEqual(markView({ outcome: 'warn' }), { shape: 'dot', hue: 'ok' });
    assert.deepEqual(markView({ outcome: 'cancelled' }), { shape: 'dot', hue: 'neutral' });
    for (const outcome of ['error', 'timeout', 'lifecycle_error']) assert.deepEqual(markView({ outcome }), { shape: 'dot', hue: 'failed' });
});

test('every view the mapping returns stays inside the grammar', () => {
    for (const mark of [...MARK_SHAPES, 'x']) for (const owner of [false, true]) for (const unread of [false, true]) {
        for (const outcome of ['', 'done', 'cancelled', 'error', 'x']) {
            const view = markView({ mark, owner, unread, outcome });
            assert.ok(MARK_SHAPES.includes(view.shape) && MARK_HUES.includes(view.hue), JSON.stringify({ mark, view }));
        }
    }
});

test('the painter writes only a change and hides the mark at rest', () => {
    const node = { dataset: { shape: 'none', hue: 'ink' }, hidden: true };
    assert.equal(paintStatusMark(node, { shape: 'moving', hue: 'unread' }), true);
    assert.deepEqual([node.dataset.shape, node.dataset.hue, node.hidden], ['moving', 'unread', false]);
    assert.equal(paintStatusMark(node, { shape: 'moving', hue: 'unread' }), false);
    assert.equal(paintStatusMark(node, { shape: 'none' }), true);
    assert.equal(node.hidden, true);
    assert.equal(paintStatusMark(null, { shape: 'dot' }), false);
});

test('the markup form is the same decorative anatomy', () => {
    const html = statusMarkHtml('nav-mark');
    assert.match(html, /^<span class="status-mark nav-mark" aria-hidden="true" data-shape="none" data-hue="ink" hidden>/);
    assert.equal((html.match(/<i><\/i>/g) || []).length, 3);
    // A template that knows its mark paints it at once; a view outside the grammar draws nothing.
    assert.match(statusMarkHtml('', { shape: 'moving', hue: 'ink' }),
        /^<span class="status-mark" aria-hidden="true" data-shape="moving" data-hue="ink"><i>/);
    assert.match(statusMarkHtml('', { shape: 'sparkle', hue: 'gold' }), /data-shape="none" data-hue="ink" hidden>/);
});

test('the owner move a server room lists reads the same on the bar and the sidebar, an unconfirmed room included', () => {
    const turn = [{ kind: 'question', quiz_id: 'q-1', task_id: 't-1' }];
    for (const activity of ['working', 'waiting', 'paused', 'unknown']) {
        const listed = room(activity, { turn });
        const bar = roomStatusView({ room: listed, works: [] });
        const row = summarizeRoom(listed);
        assert.deepEqual(markView({ mark: bar.mark, owner: bar.owner }), { shape: 'dot', hue: 'owner' }, activity);
        assert.deepEqual(markView({ mark: row.mark, owner: row.owner }), { shape: 'dot', hue: 'owner' }, activity);
        const quiet = room(activity);
        assert.equal(roomStatusView({ room: quiet, works: [] }).owner, false, activity);
        assert.equal(summarizeRoom(quiet).owner, false, activity);
    }
});

// One owner: only status_mark.js builds a mark's anatomy and writes its shape and hue; a stylesheet other than the
// recipe in ui.css may place a mark and set --mark-ink, never paint it; and the painters it replaced stay gone.
const WEB = new URL('../', import.meta.url);
const RETIRED = [
    /\bnav-unread-dot\b/, /\bnav-activity-marker\b/, /\brw-(?:mark|dots|lead|row-dots)\b/, /\bchat-live-typing\b/,
    /\btyping-dots\b/, /\bskills-dots\b/, /\blive-dot-breathe\b/, /\bchat-quiz-dot\b/, /\bsetLiveCardTypingVisible\b/,
    /\[data-(?:amber|wait)(?![-\w])/,
];
const PAINTS = [
    /\bdataset\.(?:shape|hue)\s*=(?!=)/, /setAttribute\(\s*['"`]data-(?:shape|hue)['"`]/,
    /class=\\?["']status-mark\b/, /className\s*=\s*['"`]status-mark\b/, /classList\.add\(\s*['"`]status-mark['"`]/,
];
const MARK_PAINT = /(?:^|;|\{)\s*(color|background(?:-color|-image)?|border-color|fill|stroke|opacity|animation(?:-name)?|box-shadow)\s*:/;

/** The source without its comments, which may name what was retired and why. */
function code(source) {
    let text = source;
    acorn.parse(source, { ecmaVersion: 'latest', sourceType: 'module', onComment: (_block, value, start, end) => {
        text = text.slice(0, start) + ' '.repeat(end - start) + text.slice(end);
    } });
    return text;
}
const css = (source) => source.replace(/\/\*[\s\S]*?\*\//g, (comment) => ' '.repeat(comment.length));

function sources() {
    const scripts = ['app.js', 'theme.js', ...readdirSync(new URL('modules/', WEB)).filter((name) => name.endsWith('.js'))
        .map((name) => `modules/${name}`)];
    const sheets = readdirSync(WEB).filter((name) => name.endsWith('.css'));
    return {
        scripts: scripts.map((name) => [name, code(readFileSync(new URL(name, WEB), 'utf8'))]),
        sheets: sheets.map((name) => [name, css(readFileSync(new URL(name, WEB), 'utf8'))]),
        pages: ['index.html', 'onboarding_template.html'].map((name) => [name, readFileSync(new URL(name, WEB), 'utf8')]),
    };
}

function violations({ scripts, sheets, pages }) {
    const found = [];
    for (const [name, text] of [...scripts, ...sheets, ...pages]) {
        for (const pattern of RETIRED) if (pattern.test(text)) found.push(`${name}: retired ${pattern}`);
    }
    for (const [name, text] of scripts) {
        if (name === 'modules/status_mark.js') continue;
        for (const pattern of PAINTS) if (pattern.test(text)) found.push(`${name}: paints a mark ${pattern}`);
    }
    // A class a script gives its marks (createStatusMark(doc, 'x'), statusMarkHtml('x')) names the same mark in a sheet.
    const aliases = new Set(['status-mark']);
    for (const [, text] of scripts) {
        for (const m of text.matchAll(/(?:createStatusMark\([^,)]*,\s*|statusMarkHtml\(\s*)['"`]([\w-]+)['"`]/g)) aliases.add(m[1]);
    }
    const marked = new RegExp(`\\.(?:${[...aliases].join('|')})(?![\\w-])`);
    for (const [name, text] of sheets) {
        if (name === 'ui.css') continue;
        for (const rule of text.matchAll(/([^{}]*)\{([^{}]*)\}/g)) {
            if (!marked.test(rule[1])) continue;
            const property = rule[2].match(MARK_PAINT);
            if (property) found.push(`${name}: "${rule[1].trim()}" sets ${property[1]} on a mark`);
        }
    }
    return found;
}

test('only status_mark.js builds and paints a mark, sheets only place it, and the old painters stay gone', () => {
    assert.deepEqual(violations(sources()), []);
});

test('the ownership check sees a second painter, a painted mark and a retired name, and lets placement and reads be', () => {
    const fixture = {
        scripts: [
            ['modules/rogue.js', code("node.dataset.shape = 'moving'; // dataset.hue = 'x' in a comment is fine\n"
                + "const html = '<span class=\"status-mark\"></span>';")],
            ['modules/reader.js', code("if (mark.dataset.shape === 'moving') row.querySelector('.status-mark');")],
            ['modules/status_mark.js', code("node.dataset.hue = hue;")],
            ['modules/quiz.js', code("head.append(createStatusMark(document, 'quiz-mark'));")],
        ],
        sheets: [
            ['room.css', css('.rw-trigger .status-mark { margin-left: 4px; --mark-ink: red; }\n.x .status-mark i { background: red }')],
            ['old.css', css('/* .chat-live-typing was here */ .nav-unread-dot { color: red }')],
            ['quiz.css', css('.card[data-busy] .quiz-mark { color: gray } .card .quiz-mark { --mark-ink: gray }')],
        ],
        pages: [],
    };
    assert.deepEqual(violations(fixture), [
        'old.css: retired /\\bnav-unread-dot\\b/',
        'modules/rogue.js: paints a mark /\\bdataset\\.(?:shape|hue)\\s*=(?!=)/',
        'modules/rogue.js: paints a mark /class=\\\\?["\']status-mark\\b/',
        'room.css: ".x .status-mark i" sets background on a mark',
        'quiz.css: ".card[data-busy] .quiz-mark" sets color on a mark',
    ]);
});

test('every moving mark rides one wave: its dots start on the document timeline, other animations are left alone', () => {
    const wave = (startTime) => ({ animationName: 'status-mark-bounce', startTime });
    const other = { animationName: 'chat-quote-target', startTime: 900 };
    const mark = { children: [0, 1, 2].map(() => ({ animations: [] })) };
    for (const dot of mark.children) dot.getAnimations = () => dot.animations;
    const already = wave(120);  // a mark that was moving before the hook was installed
    const listeners = new Map();
    const doc = {
        getAnimations: () => [already, other],
        addEventListener: (type, fn, capture) => listeners.set(type, { fn, capture }),
        removeEventListener: (type, fn) => { if (listeners.get(type)?.fn === fn) listeners.delete(type); },
    };
    const dispose = syncMarkPhases(doc);
    assert.equal(already.startTime, 0, 'what moves already joins the wave');
    assert.equal(other.startTime, 900, 'another animation keeps its own clock');
    assert.equal(listeners.get('animationstart').capture, true);
    // A mark painted 500 ms later: its first dot starts; the other two are still in their delay.
    mark.children[0].animations = [wave(500)];
    mark.children[1].animations = [wave(500)];
    mark.children[2].animations = [wave(500), other];
    listeners.get('animationstart').fn({ animationName: 'status-mark-bounce', target: { parentElement: mark } });
    assert.deepEqual(mark.children.map((dot) => dot.animations[0].startTime), [0, 0, 0], 'all three dots, delays kept');
    assert.equal(other.startTime, 900);
    listeners.get('animationstart').fn({ animationName: 'thinking-pulse', target: { parentElement: mark } });
    dispose();
    assert.equal(listeners.has('animationstart'), false, 'the disposer removes the hook');
    assert.equal(typeof syncMarkPhases(null), 'function', 'no document: nothing to align');
});
