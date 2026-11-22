// The shared confirm dialog itself (confirm_dialog.js): the real dialog on the dashboard test DOM. Its callers'
// decisions are tested with injected dialogs (web_dialogs.test.js); this file holds what only the dialog can prove:
// its caller's content may rename the confirming button while the dialog is open, and nothing else of it.
import assert from 'node:assert/strict';
import test from 'node:test';
import { openConfirmDialog } from '../modules/confirm_dialog.js';
import { setup } from './dashboard_dom_fixture.js';

const opened = () => document.body.querySelector('[role="dialog"]');
const press = (dialog, selector) => dialog.parentElement.fire('click', { target: dialog.querySelector(selector) });

test('content may rename the confirming button while the dialog is open: plain words, or a filled template', async (t) => {
    setup(t);
    let sheet = null;
    const answer = openConfirmDialog({ title: 'Change Main', body: 'A window set by hand is cleared.', confirmLabel: 'Change Main',
        content: { label: 'Main model', mount(container, dialog) { sheet = dialog; container.textContent = 'What Main becomes'; } } });
    const dialog = opened();
    const ok = dialog.querySelector('[data-confirm-ok]');
    assert.equal(ok.textContent, 'Change Main', 'as the caller named it');
    assert.equal(typeof sheet?.setConfirmLabel, 'function', 'the content is handed the seam');

    sheet.setConfirmLabel('Change Main and move {n} more', { n: 2 });
    assert.equal(ok.textContent, 'Change Main and move 2 more');
    assert.equal(ok.dataset.i18nFmt, 'Change Main and move {n} more', 'a filled template is not translated a second time');
    sheet.setConfirmLabel('Change Main');
    assert.deepEqual([ok.textContent, ok.dataset.i18nFmt], ['Change Main', undefined], 'plain words again are the overlay\'s');
    sheet.setConfirmLabel('');
    assert.equal(ok.textContent, 'Change Main', 'no words: the label it was opened with');
    assert.equal(dialog.querySelector('[data-confirm-cancel]').textContent, 'Cancel', 'the seam renames the confirming button only');
    assert.equal(dialog.querySelector('h3').textContent, 'Change Main');

    await press(dialog, '[data-confirm-ok]');
    assert.equal(await answer, true, 'a renamed button still confirms');
    assert.equal(opened(), null);
});

test('a dialog without content has no seam and keeps the label it was opened with', async (t) => {
    setup(t);
    const answer = openConfirmDialog({ title: 'Delete the note?', body: 'This cannot be undone.', confirmLabel: 'Delete', danger: true });
    const dialog = opened();
    assert.equal(dialog.querySelector('[data-confirm-ok]').textContent, 'Delete');
    assert.equal(dialog.querySelector('[data-confirm-content]'), null);
    await press(dialog, '[data-confirm-cancel]');
    assert.equal(await answer, false);
});

test('content that never renames the button changes nothing, and its disposer still runs when the dialog closes', async (t) => {
    setup(t);
    const seen = [];
    const answer = openConfirmDialog({ title: 'Apply the update?', body: 'The app restarts.', confirmLabel: 'Apply',
        content: { mount(container) { seen.push('mounted'); container.textContent = 'What changed'; return () => seen.push('disposed'); } } });
    const dialog = opened();
    assert.equal(dialog.querySelector('[data-confirm-ok]').textContent, 'Apply');
    await press(dialog, '[data-confirm-cancel]');
    assert.equal(await answer, false);
    assert.deepEqual(seen, ['mounted', 'disposed']);
});
