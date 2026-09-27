import test from 'node:test';
import assert from 'node:assert/strict';
import { appendToDraft } from '../modules/learn.js';

test('task template leaves the existing draft intact and separates the new request', () => {
    assert.equal(appendToDraft('Мой первый вопрос  ', 'Новая задача'), 'Мой первый вопрос  \n\nНовая задача');
    assert.equal(appendToDraft('', 'Новая задача'), 'Новая задача');
    assert.equal(appendToDraft('  ', 'Новая задача'), '  \n\nНовая задача');
    assert.equal(appendToDraft('```\ncode\n  ', 'Новая задача'), '```\ncode\n  \n\nНовая задача');
});
