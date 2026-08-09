import assert from 'node:assert/strict';
import test from 'node:test';

import {
    composerObjectiveText,
    modelChipPresentation,
    recallFieldForEntry,
    renderUserContent,
} from '../modules/chat.js';
import { MAX_CHIP_LINES, makeChipPart, serializeParts } from '../modules/composer_parts.js';
import { escapeHtmlAttr } from '../modules/utils.js';

const rangedChip = (over = {}) => makeChipPart({
    path: 'ouroboros/loop.py', lineStart: 10, lineEnd: 12, ...over,
});

// --- renderUserContent: markers become chips, everything else stays text -----

test('a captured context marker renders as a chip beside the typed text', () => {
    const parts = [
        rangedChip({ content: 'for attempt in range(3):\n    sleep(backoff)\n    backoff *= 2' }),
        { type: 'text', text: 'make this retry honest' },
    ];
    // EXACTLY the string sendMessage hands to addMessage (live echo).
    const raw = serializeParts(parts);
    const html = renderUserContent(raw);

    assert.match(html, /class="chat-user-parts"/);
    assert.match(html, /class="chat-context-chip"/);
    assert.match(html, /loop\.py · 3 lines/);
    assert.match(html, /make this retry honest/);
    // The full referent is reachable on hover, not only the basename.
    assert.match(html, /title="ouroboros\/loop\.py"/);
    // The captured BYTES ride the payload the agent reads; the UI shows the
    // referent instead of pasting the code back at the owner.
    assert.doesNotMatch(html, /backoff \*= 2/);
});

test('server replay of the same raw string renders the same chip', () => {
    // /api/chat/history returns the serialized string, never a parts array: the
    // raw text IS the identity, and the renderer is the only thing that re-reads it.
    // The capture is CONSISTENT (L10-L12 carrying exactly three lines) — that is
    // what earns it a chip; see the mismatch test below.
    const raw = serializeParts([rangedChip({ content: 'x = 1\ny = 2\nz = 3' })]);
    assert.equal(renderUserContent(raw), renderUserContent(raw));
    assert.match(renderUserContent(raw), /class="chat-context-chip"/);
    // A whole-file marker (no range, no bytes) replays as a chip too.
    const whole = renderUserContent('[context: docs/ARCHITECTURE.md]');
    assert.match(whole, /class="chat-context-chip"/);
    assert.match(whole, />ARCHITECTURE\.md</);
});

test('chips and comments render in the ORDER the owner composed them', () => {
    const raw = serializeParts([
        { type: 'text', text: 'compare' },
        makeChipPart({ path: 'a.py', lineStart: 1, lineEnd: 1, content: 'a' }),
        { type: 'text', text: 'against' },
        makeChipPart({ path: 'b.py', lineStart: 2, lineEnd: 2, content: 'b' }),
    ]);
    const html = renderUserContent(raw);
    const at = (needle) => {
        const index = html.indexOf(needle);
        assert.ok(index >= 0, `${needle} missing from ${html}`);
        return index;
    };
    assert.ok(at('compare') < at('a.py · 1 line'), html);
    assert.ok(at('a.py · 1 line') < at('against'), html);
    assert.ok(at('against') < at('b.py · 1 line'), html);
});

test('a chip is earned by a PROVABLE line count, never claimed by the marker', () => {
    // A chip folds its fenced bytes away behind "N lines", and N is counted off the
    // BYTES, so it can never conceal payload the agent still receives. A five-line
    // fence under an `L10-L12` marker — which this codec never emits, but arbitrary
    // owner text can contain — is announced as the five lines it is, not as the
    // three the marker claims.
    const lookalike = '[context: ouroboros/loop.py L10-L12]\n```\none\ntwo\nthree\nfour\nfive\n```';
    const html = renderUserContent(lookalike);
    assert.match(html, /class="chat-context-chip"/, html);
    assert.match(html, /loop\.py · 5 lines/, html);
    assert.doesNotMatch(html, /3 lines/, html);

    // The other direction cannot fold at all: a marker naming MORE lines than its
    // fence can show loses the bytes in the codec (they would under-fill the range),
    // so the round-trip is lossy and the whole raw string is shown instead.
    const overclaim = '[context: ouroboros/loop.py L10-L20]\n```\none\ntwo\n```';
    const rawHtml = renderUserContent(overclaim);
    assert.doesNotMatch(rawHtml, /chat-context-chip/, rawHtml);
    assert.equal(rawHtml, escapeHtmlAttr(overclaim));

    // A genuine capture — bytes exactly filling the range — still renders as a chip
    // with the bytes held back.
    const genuine = serializeParts([rangedChip({ content: 'one\ntwo\nthree' })]);
    const good = renderUserContent(genuine);
    assert.match(good, /class="chat-context-chip"/);
    assert.match(good, /loop\.py · 3 lines/);
    assert.doesNotMatch(good, /two/);

    // A one-line range with a one-line fence is the boundary case, and passes.
    const single = serializeParts([makeChipPart({
        path: 'a.py', lineStart: 7, lineEnd: 7, content: 'x = 1',
    })]);
    assert.match(renderUserContent(single), /class="chat-context-chip"/);
});

// A marker over a fence LONGER than the inline cap: the serializer legitimately
// drops those bytes, so the parts projection is shorter than the message. The
// only honest rendering is the owner's exact bytes.
const OVER_CAP_LINES = MAX_CHIP_LINES + 53;
const overCapBody = Array.from({ length: OVER_CAP_LINES }, (_, i) => `line ${i + 1}`).join('\n');

test('a projection that is not byte-exact renders the WHOLE raw string instead', () => {
    // 253 fenced lines under a 3-line claim. Both guards would once have folded
    // this away: the chip label hides the fence, and the per-part "show the
    // grammar" fallback re-serializes to a BARE marker over the cap — 255 lines of
    // payload the agent receives, rendered as one line the owner can read.
    const raw = `[context: ouroboros/loop.py L10-L12]\n\`\`\`\n${overCapBody}\n\`\`\``;
    const html = renderUserContent(raw);
    assert.doesNotMatch(html, /chat-context-chip/, html);
    assert.doesNotMatch(html, /chat-user-parts/, html);
    // Nothing is dropped and nothing is added: the escaped raw string, exactly
    // (fence backticks are escaped, which renders as the same characters).
    assert.equal(html, escapeHtmlAttr(raw));
    assert.equal(html.split('\n').length, raw.split('\n').length);
    for (const needle of ['[context: ouroboros/loop.py L10-L12]', 'line 1', `line ${OVER_CAP_LINES}`]) {
        assert.ok(html.includes(needle), `${needle} missing`);
    }
});

test('an over-cap capture whose range AGREES with its fence is also shown in full', () => {
    // The consistent version of the same message: the range matches the fence, so
    // the per-part provability check passes — and the projection is still lossy,
    // because the serializer drops bytes over the cap. Consistency does not earn a
    // chip when the round-trip is not byte-exact.
    const raw = `[context: ouroboros/loop.py L1-L${OVER_CAP_LINES}]\n\`\`\`\n${overCapBody}\n\`\`\``;
    const html = renderUserContent(raw);
    assert.doesNotMatch(html, /chat-context-chip/, html);
    assert.equal(html, escapeHtmlAttr(raw));
});

test('a capture AT the inline cap still chips — the guard is not a blanket refusal', () => {
    const atCap = Array.from({ length: MAX_CHIP_LINES }, (_, i) => `line ${i + 1}`).join('\n');
    const raw = serializeParts([makeChipPart({
        path: 'ouroboros/loop.py', lineStart: 1, lineEnd: MAX_CHIP_LINES, content: atCap,
    })]);
    const html = renderUserContent(raw);
    assert.match(html, /class="chat-context-chip"/, html);
    assert.match(html, new RegExp(`loop\\.py · ${MAX_CHIP_LINES} lines`), html);
    assert.doesNotMatch(html, /line 137/, html);
});

test('the newlines BETWEEN parts survive the projection', () => {
    // The renderer is the inverse of serializeParts, which joins parts with '\n'.
    // Dropping those separators would silently reflow the owner's message.
    const raw = serializeParts([
        { type: 'text', text: 'before' },
        makeChipPart({ path: 'a.py', lineStart: 1, lineEnd: 1, content: 'a' }),
        { type: 'text', text: 'after' },
    ]);
    const html = renderUserContent(raw);
    assert.match(html, /<\/span>\n<span/, html);
    // Exactly one separator per gap: two gaps for three parts.
    assert.equal(html.split('\n').length - 1, 2, html);

    // A single-text message never enters the parts path at all — it stays the one
    // escaped string it always was (no wrapper, no per-segment spans).
    const plain = renderUserContent('line one\nline two\n\nline four');
    assert.equal(plain, 'line one\nline two\n\nline four');
    assert.doesNotMatch(plain, /chat-user-parts/);
});

test('malformed marker-shaped lines stay plain text, never a chip', () => {
    const cases = [
        '[context:no-space.py]',
        ' [context: leading-space.py]',
        '[context: trailing.py] and more prose',
        '[context: two.py][context: markers.py]',
        '[context: ]',
        'CONTEXT: not-a-marker.py',
        '[context: has]bracket.py]',
    ];
    for (const raw of cases) {
        const html = renderUserContent(raw);
        assert.doesNotMatch(html, /chat-context-chip/, raw);
        // ...and not one character of what the owner wrote is dropped.
        assert.ok(html.includes(raw.replaceAll('&', '&amp;')), raw);
    }
});

test('prose ABOUT the marker grammar is not silently eaten', () => {
    const raw = 'the codec writes [context: path L1-L2] and then a fenced block';
    const html = renderUserContent(raw);
    assert.doesNotMatch(html, /chat-context-chip/);
    assert.ok(html.includes(raw));
});

test('markup can never escape the projection — text, chip label, or chip path', () => {
    // Plain text payload.
    const plain = renderUserContent('<img src=x onerror="alert(1)">');
    assert.doesNotMatch(plain, /<img/);
    assert.match(plain, /&lt;img/);

    // A path is allowed to contain `<` and `"`, so the label AND the title
    // attribute must both be escaped — otherwise the chip itself is the injection
    // point. Pinned as EXACT output: every markup character is neutralised and the
    // quote cannot close the title attribute early.
    //
    // The label is `b&gt;.py` and that is not a truncation: `chipLabel` shows the
    // BASENAME, and this path's last `/` is the one inside `</b>`, so the segment
    // after it is literally `b>.py`. The full path stays in `title`. A whole-file
    // marker carries no range either, so there is no "· N lines" suffix.
    assert.equal(
        renderUserContent('[context: <b onclick="x">evil</b>.py]'),
        '<span class="chat-user-parts">'
        + '<span class="chat-context-chip"'
        + ' title="&lt;b onclick=&quot;x&quot;&gt;evil&lt;/b&gt;.py">b&gt;.py</span>'
        + '</span>',
    );

    // An HTML payload inside a fenced capture is never rendered at all.
    const fenced = renderUserContent(serializeParts([
        makeChipPart({
            path: 'page.html', lineStart: 1, lineEnd: 1,
            content: '<script>alert(1)</script>',
        }),
    ]));
    assert.doesNotMatch(fenced, /<script/);
    assert.doesNotMatch(fenced, /alert\(1\)/);
});

test('an ordinary message renders as one escaped string, exactly as before', () => {
    assert.equal(renderUserContent('hello there'), 'hello there');
    assert.equal(renderUserContent('a < b && c > d'), 'a &lt; b &amp;&amp; c &gt; d');
    assert.equal(renderUserContent(''), '');
    assert.equal(renderUserContent(null), '');
    // Newlines survive (the .message surface renders them via white-space).
    assert.equal(renderUserContent('one\ntwo'), 'one\ntwo');
});

// --- recall: the field the entry refills, byte for byte ----------------------

// What the composer will actually send back: committed parts plus the live input,
// joined by the same '\n' `composerParts.serialize()` uses.
const fieldBytes = (field) => serializeParts(
    field.input ? [...field.parts, { type: 'text', text: field.input }] : field.parts,
);

test('recalling an entry the codec cannot round-trip is byte-exact on re-send', () => {
    // A marker plus a 201-line fence: parsing it back and re-serializing DROPS the
    // fenced bytes (over the inline cap), so committing it as parts would make
    // ArrowUp-then-Enter send a message the owner never wrote.
    const body = Array.from({ length: MAX_CHIP_LINES + 1 }, (_, i) => `line ${i + 1}`).join('\n');
    const entry = `[context: ouroboros/loop.py L1-L${MAX_CHIP_LINES + 1}]\n\`\`\`\n${body}\n\`\`\``;

    const field = recallFieldForEntry(entry);
    assert.deepEqual(field.parts, [], 'nothing may be committed as parts');
    assert.equal(field.input, entry, 'the whole entry lands in the live input');
    // The property that matters: re-sending transmits the identical string.
    assert.equal(fieldBytes(field), entry);
});

test('a genuine capture still recalls as CHIPS, with the words left editable', () => {
    const chip = rangedChip({ content: 'one\ntwo\nthree' });
    const entry = serializeParts([chip, { type: 'text', text: 'make this retry honest' }]);

    const field = recallFieldForEntry(entry);
    assert.deepEqual(field.parts, [chip], 'the capture comes back as a removable chip');
    // The trailing words stay in the live input so the caret lands in them.
    assert.equal(field.input, 'make this retry honest');
    assert.equal(fieldBytes(field), entry);

    // A chips-only entry recalls as chips with an empty input.
    const chipsOnly = serializeParts([chip, makeChipPart({ path: 'README.md' })]);
    const only = recallFieldForEntry(chipsOnly);
    assert.deepEqual(only.parts.map((p) => p.type), ['chip', 'chip']);
    assert.equal(only.input, '');
    assert.equal(fieldBytes(only), chipsOnly);

    // Plain prose recalls as plain prose in the input, unchanged.
    const prose = recallFieldForEntry('just words\nover two lines');
    assert.deepEqual(prose.parts, []);
    assert.equal(prose.input, 'just words\nover two lines');
    assert.equal(fieldBytes(prose), 'just words\nover two lines');

    // Nothing to recall is an empty field, not a thrown error.
    for (const empty of ['', null, undefined]) {
        assert.deepEqual(recallFieldForEntry(empty), { parts: [], input: '' });
    }
});

// --- _pendingCardObjective: human words only ---------------------------------

test('the project objective comes from the human TEXT parts only', () => {
    const chip = rangedChip({ content: 'x = 1' });
    assert.equal(
        composerObjectiveText([chip, { type: 'text', text: 'make it retry' }], ''),
        'make it retry',
    );
    // The live typed draft counts, and comes last.
    assert.equal(composerObjectiveText([chip], 'and here is why'), 'and here is why');
    assert.equal(
        composerObjectiveText(
            [{ type: 'text', text: 'first' }, chip, { type: 'text', text: 'second' }],
            'draft',
        ),
        'first\nsecond\ndraft',
    );
    // A chips-only message yields NO objective: a serialized marker is machine
    // grammar, and naming a project after it would be a fabricated title.
    assert.equal(composerObjectiveText([chip], ''), '');
    assert.equal(composerObjectiveText([chip], '   '), '');
    assert.equal(composerObjectiveText([], ''), '');
    assert.equal(composerObjectiveText(), '');
    assert.equal(composerObjectiveText(null, null), '');
});

// --- read-only model chip ----------------------------------------------------

test('the model chip stays hidden until the settings snapshot names a model', () => {
    // Before the Settings module's first /api/settings load there is nothing
    // honest to show, so the chip renders nothing rather than a guessed default.
    for (const snapshot of [null, undefined, {}, { OUROBOROS_MODEL: '' }, { OUROBOROS_MODEL: '  ' }]) {
        assert.deepEqual(
            modelChipPresentation(snapshot),
            { visible: false, label: '', title: '' },
        );
    }
});

test('the model chip names the main model compactly and points at Settings', () => {
    const view = modelChipPresentation({ OUROBOROS_MODEL: 'anthropic/claude-fable-5' });
    assert.equal(view.visible, true);
    assert.equal(view.label, 'claude-fable-5');
    // The COMPLETE id stays reachable on hover, and the chip says where to change it.
    assert.match(view.title, /anthropic\/claude-fable-5/);
    assert.match(view.title, /Settings/);
    // A bare id has no provider prefix to strip.
    assert.equal(modelChipPresentation({ OUROBOROS_MODEL: 'gpt-5.6-sol' }).label, 'gpt-5.6-sol');
    // A local route is disclosed as local (shared compactModel projection).
    assert.match(modelChipPresentation({ OUROBOROS_MODEL: 'local::qwen3' }).label, /\(local\)/);
});
