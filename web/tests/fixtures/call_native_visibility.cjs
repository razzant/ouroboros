// Playwright forces every automated page active, which masks native tab visibility.
// Only the native-background call test preloads this adapter into its disposable
// driver. Leave the browser's real focus/visibility alone; never patch page JS or
// installed Playwright files. Fail visibly if the driver's initialization changes.
const Module = require('node:module');
const fs = require('node:fs');
const load = Module._extensions['.js'];
Module._extensions['.js'] = function(module, filename) {
    if (!filename.endsWith('/playwright/driver/package/lib/coreBundle.js')) return load(module, filename);
    let source = fs.readFileSync(filename, 'utf8');
    for (const call of [
        'this._client.send("Emulation.setFocusEmulationEnabled", { enabled: true })',
        'this._pageProxySession.send("Emulation.setActiveAndFocused", { active: true })',
    ]) {
        if (source.split(call).length !== 2) throw new Error('Native visibility fixture needs the current Playwright focus initializer');
        source = source.replace(call, 'Promise.resolve()');
    }
    module._compile(source, filename);
};
