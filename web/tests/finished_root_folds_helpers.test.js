// Helpers' outcomes are their root's Details (owner 09.11, A): under a finished, collapsed root a FINISHED helper card
// folds away; a helper still running stays in view; an open root shows every helper. The rule is one stylesheet
// selector keyed on the attributes chat.js already writes (`data-finished`, `data-expanded`), without `:has()`
// (WebKit does not re-match `:has()` on a later child change), so this test pins the selector's shape and the
// attributes it relies on in the card template.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8').replace(/\r\n?/g, '\n');
const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');

const RULE = /\.chat-live-card\[data-finished="1"\]:not\(\[data-expanded="1"\]\) > \.chat-subagents > \.chat-live-card\.subagent\[data-finished="1"\] \{\s*display: none;\s*\}/;

test('a finished helper under a finished, collapsed root is hidden by one selector on the cards\' own attributes', () => {
    assert.match(css, RULE);
    const start = css.search(RULE);
    const block = css.slice(Math.max(0, start - 700), start);
    assert.doesNotMatch(css.slice(start, start + 200), /:has\(/, 'no :has(): WebKit invalidation pitfall');
    assert.match(block, /accessible name still counts "N children"/, 'the comment states what the collapsed root still names');
});

test('the selector stands on attributes chat.js writes: data-finished flips at settle, children sit in .chat-subagents', () => {
    // The card template starts unfinished and collapsed unless a saved choice says otherwise; settling writes '1'.
    assert.match(chat, /root\.dataset\.finished = '0';/);
    assert.match(chat, /record\.root\.dataset\.finished = '1';/);
    assert.match(chat, /root\.dataset\.expanded = initialExpanded \? '1' : '0';/);
    assert.match(chat, /container\.className = 'chat-subagents';/);
    assert.match(chat, /root\.classList\.add\('subagent'\);/);
});

test('a running helper is not covered by the fold: only data-finished="1" children match', () => {
    // The selector names the child's own finished state; an unfinished child ("0") is outside it by construction.
    const [selector] = css.match(RULE);
    assert.ok(selector.includes('.chat-live-card.subagent[data-finished="1"]'));
    assert.ok(!selector.includes('.chat-subagents {'), 'the container itself is never hidden by this rule');
});
