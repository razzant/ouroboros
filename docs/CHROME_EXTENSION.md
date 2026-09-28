# Using your own Chrome through the Playwright Extension

> **Development candidate — do not enable for a personal profile.** This staged
> integration has only synthetic POSIX process-custody tests; discovery/Refresh
> still uses the SDK's uncustodied transport, and no real extension connection was
> verified. Instructions below
> describe intended configuration, not an installed or supported feature.

Ouroboros's built-in browser tools (`browse_page`, `browser_action`) always use a fresh headless Chromium with an empty private profile. They never see your cookies or logins. If you want a task to work in pages of **your own, already-open Chrome**, where you are already signed in, Ouroboros can use Microsoft's official [Playwright Extension](https://github.com/microsoft/playwright/tree/main/packages/extension) through the [Playwright MCP server](https://github.com/microsoft/playwright-mcp) (`@playwright/mcp --extension`).

This is opt-in and off by default. Ouroboros never installs a browser add-on. You install the extension yourself and add one MCP server entry. Ouroboros has no browser extension of its own.

## What you are granting

- A connected task can open, read and act in tabs of your Chrome profile **as you**. That includes any site where you are signed in: it can submit forms, send messages and confirm purchases. The extension has access to all sites and uses Chrome's `debugger` permission; there is no per-site prompt.
- Page text and snapshots are untrusted third-party data. Raw MCP browser tools do **not** pass through the built-in `browser_policy` URL/control checks, so task-scoped sessions alone cannot block navigation to local control surfaces or transactional clicks. A malicious page can try to persuade the model; neither a tab group nor the Safety Supervisor is a sandbox.
- The MCP client uses a separate session and names it `Ouroboros task <task id>` for each task. The extension groups tabs by connection, but this is **not** an enforced per-tab security boundary: its broad Chrome permissions and the MCP server's own behavior may expose other authorized tabs. Do not enable this entry for a task that must be restricted to one tab.

## Install and connect

1. In Chrome, install **Playwright Extension** from the [Chrome Web Store](https://chromewebstore.google.com/detail/playwright-extension/mmlmfjhmonkocbjadbfplnigmagldckm).
2. Make sure Node.js (`npx`) is available. The first run downloads `@playwright/mcp` from npm.
3. In the experimental candidate, open **Settings → Advanced → MCP Servers** and click **Chrome experiment**. This adds a disabled entry; **do not enable it for a personal profile**:

   ```json
   {
     "id": "chrome",
     "name": "My Chrome (Playwright Extension)",
     "enabled": false,
     "transport": "stdio",
     "command": "npx",
     "args": ["-y", "@playwright/mcp@0.0.82", "--extension"],
     "session_scope": "task"
   }
   ```

   Review it, tick **Enabled** and save only in an isolated test profile. The tools appear as `mcp_chrome__browser_*`. **Refresh** discovers them by opening and initializing an MCP transport; for an extension-backed server this may open a Chrome connection or approval page. It is not a passive check.
4. On the first browser tool call in a task, the extension opens its connect page in Chrome (Chrome starts if it is not running). Pick the tab to share and approve. Later calls in the same task reuse that connection.

Approve before the MCP **Per-tool timeout** runs out (60 s by default). Otherwise the call times out and closure is requested; a next call is refused until the session thread actually exits. Even then, detached process descendants are unproven. Raise the timeout if you need more time in an isolated test.

To use a Chrome profile other than the last one you used, add `"--profile-dir-name", "Profile 2"` to `args`. The directory name is the last part of "Profile Path" at `chrome://version`.

## Which tasks can connect

- Root tasks, including each direct-chat turn. Each chat message is its own task, so each turn opens its own connection.
- **Not** delegated child tasks: they never see these tools, and a call is refused before anything runs.
- **Not** background consciousness wakes: they are refused at dispatch.
- Never another task's connection. Sessions are keyed by task id; a second task opens its own and never reuses the first.

Without a token, **every new connection needs your approval in Chrome**. That approval is what keeps a second task from silently reaching your browser.

### The optional token (automatic connections)

The extension's status page (click the extension icon) shows a `PLAYWRIGHT_MCP_EXTENSION_TOKEN`. If you give it to the server, connections no longer ask. Store it as a secret setting and reference it with **Environment from settings**, e.g. `{"PLAYWRIGHT_MCP_EXTENSION_TOKEN": "MY_CHROME_TOKEN"}`.

With the token set, **every eligible task connects to your browser without asking**. It is a standing approval for all root tasks, not a per-task one. Ouroboros still refuses children and consciousness wakes, and still gives each task its own connection.

## Disconnect and Stop

| What you do | What happens |
|---|---|
| The task finishes, fails, or pauses on budget | Ouroboros requests closure at loop exit. A false close receipt retains local custody, but terminality is not proof of server-process death. |
| **Stop** on a task running in a worker | The task-held transport now requests a recorded process-group stop and checks exit in synthetic POSIX tests. This is not an installed Chrome disconnection proof; discovery/Refresh and Windows are not covered. Disconnect in Chrome and disable this entry if the stop is unconfirmed. |
| **Stop** on a direct-chat turn, or a Stop that lets the task wrap up | The loop requests session closure; an unconfirmed close does not prove disconnection. Panic requests and group settlement are present, but real hard-exit timing remains under independent review. An action already in flight may have taken effect. |
| A browser call fails or times out | Closure is requested. The same task cannot open another session while the prior thread remains alive; a verified thread exit permits a new attempt, but does not by itself establish the absence of detached descendants. |
| **Disconnect** on the extension's status page | The extension ends that connection. If the same task calls again, it may request another connection: it asks without a token and is automatic with one. To end access, disconnect in Chrome **and** disable the server. |
| Disable or change the server in Settings | Held sessions close when the process running the task next reads Settings (at the task's next tool round), not instantly. |
| Remove or disable the extension in Chrome | No connection is possible. |

Stopping never undoes an action already sent: a click or a submitted form may already have taken effect. Closing a connection does not close tabs. Tabs the agent opened stay in your Chrome.

The Playwright MCP server saves page snapshots as files in its working directory. Unless you set a **Working directory** on the entry, Ouroboros runs each task's server in a private temporary directory and deletes it when the session closes. An immediate Stop kills the process before that cleanup, so the directory can remain in your system's temporary folder, readable only by your user.

## Limits

- Ouroboros has no live "connected" indicator. Sessions live in the process running the task. The extension's status page is the truthful list of connections.
- A task's first call waits on your approval (or on the token handshake). The intended connection lifetime is that task. Synthetic POSIX tests exercise task-held process custody, but discovery/Refresh and real Chrome disconnection remain unproven; no installed-host guarantee follows.
- Tests exercise task sessions and process-group Stop/Panic logic against synthetic stateful stdio MCP servers; they do not exercise `@playwright/mcp`, the installed Chrome extension, a logged-in tab, or actual Panic process exit. The attempted isolated Chrome consumer probe was stopped after it repeatedly opened visible windows; it did not complete an attach/action/disconnect flow. This candidate must not be promoted as a safe personal-Chrome integration until these paths and the independent P1 custody findings are resolved.
