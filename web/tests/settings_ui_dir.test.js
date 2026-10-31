// The main-interface choice in Settings: one restart-applied field, saved only
// when the owner edits it, beside what the core actually serves right now.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { servedUiStatus, uiDirChange } from '../modules/settings.js';
import { renderSettingsPage } from '../modules/settings_ui.js';

const panel = (html, name) => {
    const start = html.indexOf(`data-settings-panel="${name}"`);
    assert.ok(start >= 0, `${name} panel`);
    return html.slice(start, html.indexOf('</section>', start));
};

test('the choice is one Advanced field with its help and a live applied-state line', () => {
    const html = renderSettingsPage();
    assert.equal(html.match(/id="s-ui-dir"/g)?.length, 1);
    const advanced = panel(html, 'advanced');
    assert.match(advanced, /<input id="s-ui-dir"[^>]*aria-describedby="s-ui-dir-help s-ui-dir-applied"/);
    assert.match(advanced, /id="s-ui-dir-applied" role="status" aria-live="polite" hidden/);
    // Deployment plumbing set once: after the trust knobs, before Cleanup and the Danger Zone.
    assert.ok(advanced.indexOf('Extra CA Certificates') < advanced.indexOf('Main Interface'));
    assert.ok(advanced.indexOf('Main Interface') < advanced.indexOf('<h3>Cleanup</h3>'));
    for (const other of ['providers', 'models', 'agents', 'behavior', 'appearance']) {
        assert.doesNotMatch(panel(html, other), /s-ui-dir/);
    }
});

test('the help is truthful about restart, refresh ownership and the way back', () => {
    const help = renderSettingsPage().match(/id="s-ui-dir-help">([\s\S]*?)<\/div>/)[1];
    assert.match(help, /Empty: the built-in interface/);
    assert.match(help, /<code>none<\/code>: no main page/);
    assert.match(help, /absolute or <code>~<\/code>-prefixed path/);
    assert.match(help, /keeps the built-in interface and says why below/);
    assert.match(help, /the built-in page reloads into the new choice; a page from a directory or another host refreshes itself and only reconnects/);
    assert.match(help, /To return to the built-in interface, clear this field .* restart, and reopen the Ouroboros address/);
    assert.match(help, /a saved choice, empty included, wins/);
    assert.match(renderSettingsPage(), /Applies at the next restart\./);
});

test('an untouched field saves nothing; an edit, a cleared blank included, saves exactly the trimmed value', () => {
    assert.deepEqual(uiDirChange('', ''), {});
    assert.deepEqual(uiDirChange('', undefined), {}, 'a core without the key: nothing to author');
    assert.deepEqual(uiDirChange('none', 'none'), {}, 'an environment value shown untouched stays the environment\'s');
    assert.deepEqual(uiDirChange('  ~/ui  ', '~/ui'), {});
    assert.deepEqual(uiDirChange('', 'none'), { OUROBOROS_UI_DIR: '' }, 'clearing saves the blank that selects built-in');
    assert.deepEqual(uiDirChange(' /srv/ui ', ''), { OUROBOROS_UI_DIR: '/srv/ui' });
    assert.deepEqual(uiDirChange('none', '/srv/ui'), { OUROBOROS_UI_DIR: 'none' });
});

test('Settings applies the loaded value without a fallback and collects through the edited-only rule', () => {
    const source = readFileSync(new URL('../modules/settings.js', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
    assert.match(source, /applyInputValue\('s-ui-dir', s\.OUROBOROS_UI_DIR\);/);
    assert.match(source, /Object\.assign\(body, uiDirChange\(fieldValue\('s-ui-dir'\), currentSettings\?\.OUROBOROS_UI_DIR\)\);/);
    // Never an always-sent INPUT_FIELDS row: that would author the key on every save.
    const fields = source.slice(source.indexOf('const INPUT_FIELDS = ['), source.indexOf('const VALUE_FIELDS = ['));
    assert.doesNotMatch(fields, /OUROBOROS_UI_DIR/);
    // Read beside each loaded document and after every reconnect (a restart applies the choice).
    assert.match(source, /_renderNetworkHint\(data\._meta\);\n\s+void refreshServedUi\(\);/);
    assert.match(source, /ws\?\.on\?\.\('open', \(\) => \{\n\s+refreshRestartState\(\);\n\s+void refreshServedUi\(\);/);
});

test('the applied-state line names what is served now, and a refused choice with its reason', () => {
    assert.deepEqual(servedUiStatus({ source: 'builtin' }), { text: 'Serving now: the built-in interface.', tone: 'muted' });
    assert.deepEqual(servedUiStatus({ source: 'directory' }), { text: 'Serving now: the interface directory.', tone: 'muted' });
    assert.deepEqual(servedUiStatus({ source: 'none' }), { text: 'Serving now: no main page.', tone: 'muted' });
    const refused = "OUROBOROS_UI_DIR '/srv/ui' was not applied: /srv/ui has no index.html";
    assert.deepEqual(servedUiStatus({ source: 'builtin', error: refused }), {
        text: `Serving now: the built-in interface. ${refused}`, tone: 'warn',
    });
    // A core older than the choice, or an unreadable fact, shows nothing.
    for (const value of [undefined, null, {}, { source: 'other' }, 'builtin']) {
        assert.equal(servedUiStatus(value), null);
    }
});
