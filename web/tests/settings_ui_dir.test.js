// The main-interface choice in Settings: one restart-applied field beside what the core actually serves right now.
// Its section is unmounted in settings_ui.js until the page that holds it is rebuilt.
import assert from 'node:assert/strict';
import test from 'node:test';

import {
    accountsPanel, advancedPanel, appearancePanel, behaviorPanel, modelsPanel, secretSettingsSection, servedUiStatus,
} from '../modules/settings_ui.js';

test('the choice is one Advanced field with its help and a live applied-state line', () => {
    const others = [accountsPanel(), secretSettingsSection(), modelsPanel(), behaviorPanel(), appearancePanel()];
    const advanced = advancedPanel();
    assert.equal(advanced.match(/id="s-ui-dir"/g)?.length, 1);
    assert.match(advanced, /<input id="s-ui-dir"[^>]*aria-describedby="s-ui-dir-help s-ui-dir-applied"/);
    assert.match(advanced, /id="s-ui-dir-applied" role="status" aria-live="polite" hidden/);
    // Deployment plumbing set once: after the trust knobs, before Cleanup and the Danger Zone.
    assert.ok(advanced.indexOf('Extra CA Certificates') < advanced.indexOf('Main Interface'));
    assert.ok(advanced.indexOf('Main Interface') < advanced.indexOf('<h3>Cleanup</h3>'));
    for (const other of others) assert.doesNotMatch(other, /s-ui-dir/);
});

test('the help is truthful about restart, refresh ownership and the way back', () => {
    const help = advancedPanel().match(/id="s-ui-dir-help">([\s\S]*?)<\/div>/)[1];
    assert.match(help, /Empty: the built-in interface/);
    assert.match(help, /<code>none<\/code>: no main page/);
    assert.match(help, /absolute or <code>~<\/code>-prefixed path/);
    assert.match(help, /keeps the built-in interface and says why below/);
    assert.match(help, /the built-in page reloads into the new choice; a page from a directory or another host refreshes itself and only reconnects/);
    assert.match(help, /To return to the built-in interface, clear this field .* restart, and reopen the Ouroboros address/);
    assert.match(help, /a saved choice, empty included, wins/);
    assert.match(advancedPanel(), /Applies at the next restart\./);
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
