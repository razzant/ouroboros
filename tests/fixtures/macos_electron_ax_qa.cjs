// Optional cold Electron receiver, launched only by an authorized native trial.
// electron macos_electron_ax_qa.cjs <log.jsonl> <disposable-profile-directory>
// This fixture NEVER calls setAccessibilitySupportEnabled or enables AX in-app.
// DOM effects / isTrusted are not raw AppKit input counts; the Swift receiver supplies those separately.
const { app, BrowserWindow, ipcMain } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const [logPath, profile] = process.argv.slice(2);
if (!logPath || !profile || !path.isAbsolute(logPath) || !path.isAbsolute(profile)) {
  throw new Error('Provide absolute log and disposable profile paths outside any checkout');
}
fs.mkdirSync(profile, { recursive: true });
app.setPath('userData', profile);
app.setPath('sessionData', profile);
const log = (event, fields = {}) => fs.appendFileSync(logPath, JSON.stringify({
  event, ...fields, pid: process.pid, monotonic_ns: process.hrtime.bigint().toString(),
}) + '\n');
const windows = [];
app.on('accessibility-support-changed', (_event, enabled) => log('ax_support_changed', { enabled }));
ipcMain.on('qa-dom', (event, payload) => {
  log('dom', { window: windows.findIndex(w => w.webContents === event.sender), ...payload });
});
app.whenReady().then(async () => {
  log('cold_start', { enabled: app.accessibilitySupportEnabled, electron: process.versions.electron });
  for (const index of [0, 1]) {
    const title = `AX Cold ${index === 0 ? 'primary' : 'secondary'}`;
    const win = new BrowserWindow({ title, x: 40, y: 60 + index * 300, width: 460, height: 260, show: false,
      // Synthetic inline page only: no navigation, preload, downloaded content or remote data.
      webPreferences: { nodeIntegration: true, contextIsolation: false },
    });
    windows.push(win);
    win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    win.webContents.on('will-navigate', event => event.preventDefault());
    for (const event of ['focus', 'blur']) win.on(event, () => log(event, { window: index }));
    await win.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(`<!doctype html>
      <html><head><title>${title}</title></head><body>
      <label>Amount <input id="amount" aria-label="Amount"></label>
      <button id="commit">Commit</button><output id="result">idle</output>
      <script>
      const { ipcRenderer } = require('electron');
      const field = document.getElementById('amount');
      const button = document.getElementById('commit');
      let count = 0;
      const record = e => ipcRenderer.send('qa-dom', { type: e.type, isTrusted: e.isTrusted,
        target: e.target.id, key: e.key, count, field: field.value });
      for (const name of ['pointerdown', 'pointerup', 'keydown', 'keyup', 'input']) {
        document.addEventListener(name, record, true);
      }
      button.addEventListener('click', e => {
        document.getElementById('result').textContent = 'count:' + (++count) + ' value:' + field.value;
        record(e);
      });
      </script></body></html>`));
    win.showInactive();
    const rect = await win.webContents.executeJavaScript(`(() => {
      const r = document.getElementById('commit').getBoundingClientRect();
      return [r.x, r.y, r.width, r.height];
    })()`);
    const content = win.getContentBounds();
    log('window_ready', { window: index, title, focused: win.isFocused(),
      commit_frame: [content.x + rect[0], content.y + rect[1], rect[2], rect[3]] });
  }
  log('cold_ready', { enabled: app.accessibilitySupportEnabled });
});
app.on('window-all-closed', () => app.quit());
process.stdin.setEncoding('utf8');
let input = '';
process.stdin.on('data', data => {
  input += data;
  const lines = input.split('\n');
  input = lines.pop();
  for (const line of lines) {
    if (line === 'quit') app.quit();
    else if (line.startsWith('mark')) log('mark', { label: line, enabled: app.accessibilitySupportEnabled,
      focused_windows: windows.map(w => w.isFocused()) });
  }
});
process.stdin.on('end', () => app.quit());
