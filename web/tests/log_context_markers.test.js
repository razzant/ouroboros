import assert from 'node:assert/strict';
import test from 'node:test';

import { compactContextMarkers, summarizeChatLiveEvent, summarizeLogEvent } from '../modules/log_events.js';
import { makeChipPart, serializeParts } from '../modules/composer_parts.js';

// The exact bytes the composer sends when the owner captures a range and types a
// question about it. The summarizers read this string back off the event.
const CAPTURE = serializeParts([
    makeChipPart({
        path: 'ouroboros/loop.py',
        lineStart: 18,
        lineEnd: 20,
        // Three lines for a three-line range, with the ONE trailing newline a real
        // file slice carries — the codec strips it and the count still holds.
        content: '    window = COOLDOWN_S\n    window += jitter\n    return window\n',
    }),
    { type: 'text', text: 'why is this constant?' },
]);

test('a captured range becomes its chip label, not a torn-off fence', () => {
    // Precondition: the raw message really does carry the marker AND the bytes.
    assert.match(CAPTURE, /^\[context: ouroboros\/loop\.py L18-L20\]\n```\n/);

    // The label is the BASENAME plus the count — the same projection the composer
    // chip and the transcript chip show, so the three surfaces agree.
    const compact = compactContextMarkers(CAPTURE);
    assert.equal(compact, '[loop.py · 3 lines]\nwhy is this constant?');
    // The fenced bytes are gone from the PRESENTATION only; nothing here mutates
    // the message, which is why the raw string is still what the agent received.
    assert.doesNotMatch(compact, /```/);
    assert.doesNotMatch(compact, /COOLDOWN_S/);
});

test('the label counts the bytes the capture carries', () => {
    // Same rule as the composer chip and the transcript chip: N is provable from
    // the fence, so a 3-line range carrying 3 lines says 3.
    const three = serializeParts([makeChipPart({
        path: 'a/b/deep.py', lineStart: 5, lineEnd: 7, content: 'one\ntwo\nthree',
    })]);
    assert.equal(compactContextMarkers(three), '[deep.py · 3 lines]');

    // A whole-file capture has no count to make up.
    assert.equal(compactContextMarkers('[context: ouroboros/loop.py]'), '[loop.py]');

    // An over-cap capture serializes bare; the range is then all there is to show.
    const bare = '[context: ouroboros/loop.py L1-L400]';
    assert.equal(compactContextMarkers(bare), '[loop.py · 400 lines]');
});

test('text that is not a capture is returned byte for byte', () => {
    const cases = [
        'just a question about the loop',
        '',
        'talk about [context:no-space.py] please',       // malformed: not a marker
        ' [context: leading-space.py]',                   // malformed: indented
        '[context: a.py] and then trailing prose',        // malformed: trailing text
        'a fence with no marker\n```\nx\n```',
        // A marker naming MORE lines than its fence shows: the codec drops those
        // bytes, so the parse is lossy and the owner's exact string stands.
        '[context: a.py L10-L20]\n```\none\ntwo\n```',
    ];
    for (const raw of cases) {
        assert.equal(compactContextMarkers(raw), raw, JSON.stringify(raw));
    }
});

test('a Logs headline shows the capture compactly', () => {
    const view = summarizeLogEvent({
        type: 'progress',
        is_progress: true,
        task_id: 'abc12345',
        content: `\u{1F4AC} ${CAPTURE}`,
    });
    assert.match(view.headline, /\[loop\.py · 3 lines\]/);
    assert.doesNotMatch(view.headline, /```/);
    assert.doesNotMatch(view.headline, /\[context:/);
});

test('a live-card title shows the capture compactly', () => {
    const view = summarizeChatLiveEvent({
        type: 'progress',
        is_progress: true,
        task_id: 'abc12345',
        content: `\u{1F4AC} ${CAPTURE}`,
    });
    assert.match(view.headline, /\[loop\.py · 3 lines\]/);
    assert.doesNotMatch(view.headline, /```/);
    assert.doesNotMatch(view.headline, /\[context:/);
    // The full text a row can expand to is compacted the same way — otherwise the
    // disclosure would contradict the headline it belongs to.
    assert.doesNotMatch(String(view.fullHeadline || ''), /\[context:/);
});
