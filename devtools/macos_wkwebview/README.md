# System WKWebView composer clearance (`devtools/macos_wkwebview/`)

An optional macOS check that the Main and Project composers clear the newest message in the system WKWebView, the
engine the desktop app runs. The browser regression (`tests/test_chat_composer_reserve_browser.py`) runs Chromium and
Playwright's WebKit build, and that build did not reproduce the defect the transcript's flex spacer fixed: bottom
padding that the system WKWebView left outside a transcript's scrollable overflow, so a short history ended under the
composer. Run this check on a Mac when a change touches the transcript, its reserve or the composer:

```bash
uv sync --locked --extra desktop --no-install-project
.venv/bin/python -I -B devtools/macos_wkwebview/composer_clearance.py --out /tmp/composer-clearance
```

Any Python with the runtime dependencies and PyObjC's WebKit bindings works, the app's bundled Python included. The
check needs a logged-in desktop session, installs nothing and uses no credentials or real provider.

## What it does

`composer_clearance.py` copies this checkout's runtime inputs (the source entries the app bundle ships, from
`Ouroboros.spec`, without tests; uncommitted edits included) to a disposable root under `/tmp` and starts the real
`server.py` there with an isolated data root, a provider address on a loopback port with no listener and a `git` that
always fails. It saves a short exchange in Main and in a Project room, then opens one window holding a bare system
`WKWebView` with non-persistent website data. The desktop app's pywebview window adds bridge scripts but no stylesheet
(it passes `text_select=True`), so the layout is the app's. `composer_clearance.js`, injected at document start,
measures and acts as a user would and never edits the product.

At the desktop window's default width (1100 px, or the screen's width when that is narrower) and at 390 px, the same
engine's stack layout, the window is first sized so that each room's history fits its transcript while the newest row
would still end under the composer. Only the reserve then creates the needed scroll range, and every sample records
whether that held. The newest row is measured against the composer's top at the end of the scroll range: on first
open, after reopening (Main returns from Settings; the Project room is closed and opened again) and after an answer
arrives, each with an empty and a five-line composer. Main's return also checks that the reserve held while Main was
hidden and, when Main was left at its newest message, that the first frame back clears the composer as that frame is
presented: the first rendering update showing Main again, read at its ResizeObserver step after the update's own
callbacks and layout. The click back is its own task, as a user's click arrives between rendering updates. The system
WKWebView keeps no scroll position for a hidden size container (the transcript is one), and the room lands on its
newest message within that same update; the position at the start of the frame, before the app's own callbacks, is
recorded but not judged.

Two negatives run over the same short history: `old-padding` restores the padding layout the spacer replaced, and
`no-reserve` removes the reserve. `no-reserve` must overlap on any engine, or the run cannot see an overlap. A probe
records whether this engine scrolls a flex container's end padding. Where it does not, `old-padding` must overlap as
well. Where it does, the report states that this OS and engine no longer reproduce the historical defect. A negative
collapses the scroll range, so the engine clamps the position; each one hands back the position it started from, and
the run records that it did.

## Outcomes and evidence

| Verdict | Exit | Meaning |
|---|---|---|
| PASS | 0 | Every sample cleared the composer, its conditions held and both negatives behaved as required. |
| FAIL | 1 | A message ended under its composer, or Main lost its reserve behind another page. |
| INCONCLUSIVE | 3 | A condition was not established: the short history, a settled layout, served assets equal to the source, a detectable negative, a run that finished, a driver that exited normally, or children confirmed gone. |
| UNSUPPORTED | 4 | Not macOS, no PyObjC WebKit, or no display. |

`--out` (new or empty, outside the checkout) receives `report.json` with the verdict, macOS, WebKit and Safari versions, the source tree
digest and served asset hashes, `observation.json` with every measurement, `server.log`, `driver.log`, and
WKWebView snapshots in `screenshots/`. `--identity-only` prints the source identity, so a report can be matched to a
checkout. The disposable root is removed after its processes are proven gone; `--keep-root`, or a process that
outlived teardown, keeps it with `OUROBOROS_RETAINED.txt`.

Not covered: iOS, phone keyboards, the Telegram host, the desktop launcher itself and page zoom other than 100 %.
