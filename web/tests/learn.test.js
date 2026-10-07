import test from 'node:test';
import assert from 'node:assert/strict';
import { appendToDraft, shouldRepaintTemplate } from '../modules/learn.js';

test('task template leaves the existing draft intact and separates the new request', () => {
    assert.equal(appendToDraft('My first question  ', 'New task'), 'My first question  \n\nNew task');
    assert.equal(appendToDraft('', 'New task'), 'New task');
    assert.equal(appendToDraft('  ', 'New task'), '  \n\nNew task');
    assert.equal(appendToDraft('```\ncode\n  ', 'New task'), '```\ncode\n  \n\nNew task');
});

test('template repaint happens only while the textarea still holds what we last wrote', () => {
    // Never painted yet: repaint allowed.
    assert.equal(shouldRepaintTemplate(undefined, ''), true);
    // Still our own last translation: repaint allowed (language switch updates it).
    assert.equal(shouldRepaintTemplate('My name is <name>.', 'My name is <name>.'), true);
    // The owner edited the draft: it is theirs now, a repaint would erase their words.
    assert.equal(shouldRepaintTemplate('My name is <name>.', 'My name is Yegor.'), false);
    // An empty textarea we painted empty keeps repainting (nothing to lose).
    assert.equal(shouldRepaintTemplate('', ''), true);
});
