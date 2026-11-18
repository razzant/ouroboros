# iPhone keyboard observation

This temporary kit runs only on pushes to the `probe/telegram-uikit-keyboard` branch.
It opens the checked-out web app in a
WKWebView on a newly created GitHub-hosted iPhone simulator. A loopback fixture
supplies ten saved rows per room and the Telegram fullscreen signals. XCTest
taps the control, taps one hittable letter on the software keyboard, waits for
input in the focused field and swipes the feed to dismiss the keyboard.

`run.py` owns the device and export, `fixture.py` serves the app, `MotionHost.swift` hosts WebKit, `KeyboardTests.swift` drives the touches, `Reading.js` locates the target, `recorder.js` records page geometry and `report.py` filters the results.

| Scenario | Where typing starts |
| --- | --- |
| `main-latest` | Reply on a folded Project result at the live edge of Main. |
| `main-reading` | The same Reply after scrolling back in Main, with more than 48 CSS px below. |
| `project-ready` | The Project composer after its saved rows have loaded. |
| `project-held` | The empty Project composer with history in flight. WebKit releases the response when it receives UIKit’s did-show notification. |

On coarse WebKit the timestamp can cover the centre of Main’s 24 px Reply.
`replyCenterAtTouch` reports what owned that point, including `msg-time`. The
probe taps a hittable point inside the same button and checks the trusted
touch target and the resulting reply strip.

## Reading the artifact

In `summary.json`, each case has its XCTest outcome, a short failure reason
when available, the keyboard overlap in points, and separate entry and exit
sampling. `trace.json` contains UIKit notifications in native uptime and their
deliveries to WebKit on the page clock. Native animation duration and curve
describe UIKit’s notification. Only numbered rows come from animation-frame
callbacks. Focus, input, notification and stop events have their own timestamps.

Each phase lists its **will** and **did** notification sequence numbers and
counts callbacks strictly between their deliveries. The largest gap includes both ends of that window. A
phase is marked `sampled` with at least two callbacks, gaps under 100 ms, and
native and delivery intervals that agree within 100 ms. A missing boundary
is `window-unavailable`. Each `classSwitches` entry gives the gap between the two
callbacks bracketing the change. A gap of 100 ms or more also limits the case’s
coverage. The report counts actual callbacks after hide.
These figures describe how much of the transition WebKit could observe.

`observed-with-sampling-gaps` means the native typing cycle was captured but
at least one phase, class change or the final frame was missed or sparsely sampled. It is a
usable baseline. `incomplete` points to missing scenario evidence in `checks`
or `xctestFailure`. A changed transcript revision or replaced anchor appears
in `historyStability`. After a timeout or cancellation the job attempts to
export already saved cases before deleting its simulator. A case needs its own
XCTest result to count as passed. Even a `complete` capture cannot establish display
smoothness, behaviour in the Telegram client or behaviour on a physical phone.

The job marks the swipe-typing introduction seen on its disposable simulator.
`tutorialPreferenceWritten` records the write outcome. Whether the tutorial
would have appeared is unknown. See [Chromium’s simulator setup](https://chromium.googlesource.com/chromium/src/+/93b31d4424cb0fddb7cb5a901f21eb9859270658/ios/build/bots/scripts/iossim_util.py).
