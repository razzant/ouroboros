---
name: unix_computer_use
description: Local and remote desktop observation and input, with macOS semantic actions, explicit foreground control, screenshot-bound coordinates and honest outcome receipts.
version: 0.9.0
type: extension
entry: plugin.py
plugin_api: "2.0"
runtime: python3
os: [darwin, linux]
permissions: [tool, subprocess, net]
env_from_settings: []
tools:
  - name: window_list
    effect: read
  - name: ax_tree
    effect: read
  - name: capabilities
    effect: read
  - name: cursor_position
    effect: read
  - name: list_connections
    effect: read
optional_mcp_servers: [{"id": "mac_window_session", "title": "Local Mac window capture (requires begin)", "runtime": "agent_python", "entry": "window_session_server.py", "task_session": true}]
when_to_use: The user asks Ouroboros to inspect a desktop, take a screenshot, act in a named application (click/type/press keys there), drag, or operate a local/explicitly configured remote GUI under human or benchmark supervision.
model_experience:
  what_model_sees: Desktop tools include bounded AX reads (hierarchy, separate title/description/value/url with a chosen text_limit), local macOS element_act press/set_value without activation or pointer input (optionally followed by a read of the same window in the same call), explicit foreground input, screenshot observation IDs, capture-only images of one exact local macOS window, capabilities and connections. Screenshots auto-attach as images; receipts separate action dispatch, readback, the read after the action and front observations. After the owner connects the window server, `mcp_mac_window_session__begin(app, window)` starts a capture-only stream of that window on the local Mac for this task. `frame` returns the latest MCP image with its sequence and ages. `release` stops the stream.
  token_effect: When loaded, all 25 tool schemas total about 16K characters; Nano can load tools on demand. Results are compact JSON without dropped fields. Input receipts grow with commands and output; screenshots add an image and capture metadata. Window lists and AX reads grow with returned entries and text (limit, app, max_nodes, text_limit); an element_act observation adds its own tree (default max_nodes 600; ax_tree 150). A receipt marked observed_effect also carries that tree and its window text into result review, increasing the material sent to the selected reviewers. The connected window session adds three MCP tool schemas (about 1.4K characters) and its server instructions; each delivered frame is one image of at most 1280x800 with a small JSON object, and `frame(after_seq=...)` sends no image when nothing newer exists.
---

# Unix Computer Use

This bundled extension exposes a reviewable computer-use substrate for local
macOS/Linux desktops and explicitly configured remote targets. **All actions are
for supervised, low-risk workflows under human or benchmark observation.** It is
offered and on by default on macOS and Linux (`os`); Windows desktops use
`windows_computer_use`. A previously enabled Unix remote skill stays available
on Windows through later OFF/ON. Actor-less legacy ON is retained too: manual
and early automatic choices cannot be distinguished. This adds no local Windows backend.

Tools:

- `capabilities` reports the platform, display-session type, who is in front
  (local and remote), the OS grants the local process holds (prompt-free
  preflights on macOS: Accessibility, Screen Recording, input events;
  `permission_state_verified` is true only when they ran; a remote machine's
  grants are not probed), whether `app=` targeting works in this session and
  what it is resolved from, and the detected screenshot/input backends plus
  the active connection.
- `list_connections`, `add_connection`, `test_connection`, `activate_connection`,
  and `use_local` manage the local/remote connection registry in skill state.
- `screenshot` captures the main display (or, on macOS with `app=`, the display
  holding that application's front window), downscales the image to fit WXGA
  (1280x800, configurable). When geometry and persistence succeed, it returns
  `mapped_input_available: true` with a saved image-to-input transform and observation ID.
  Otherwise the image remains usable for viewing, but `mapped_input_available: false`
  means no `coord_transform` or `observation_id` is supplied. SSH macOS uses the
  primary display capture and its matching CoreGraphics bounds; it does not map a
  single-screen image against the Finder desktop union. `display.selected_by` says how the
  display was chosen: `default`, `app`, or `fallback` when the named application
  was resolved but its window could not be located (`app_window_unread` says why)
  and the main display was captured instead; the result then carries no `app`.
  Once the display list is read, an `app=` not resolved to one running application
  (not running, ambiguous, or unknown because the helper gave no answer) is refused:
  nothing is captured. If the display list cannot be read, the application is not
  looked up and no app selection is established: the default capture stays
  capture-only (no mapping), with `selected_by: default`, `app_display_unread`
  saying why and no `app`. Linux and the remotes
  capture their one screen and ignore `app=` (the result says so). Pass
  coordinates read off the returned image together with its `observation_id`
  to the input tools; mapping is bound to that captured observation. The result carries the typed `auto_attach_image`
  field, so the host attaches the image to the conversation in the same round;
  an engine that cannot take images receives the path instead (host behaviour),
  so the text facts in every other tool are the way to check without a picture.
  Omit `window` for a display capture. With `window=` (a positive integer `window_id`
  or a nonblank title string, matched literally even if it is all digits)
  and `app=`, local macOS 14+ captures that one window through ScreenCaptureKit
  in a short-lived helper (`macos_capture.py`), also while other windows cover it,
  without activating anything: the window's own pixels, no cursor, no shadow, at
  its size (downscaled as above), under a new file name each call. Only a window
  ScreenCaptureKit lists for the resolved application's process qualifies; none
  or several (`no_window`, `window_not_found`, `window_ambiguous`) is refused,
  never replaced by another window or a display. `window=` without `app=`, on a
  remote connection or off macOS is refused before anything else runs. Without
  the Screen Recording grant (prompt-free preflight) nothing is asked or captured
  (`screen_recording_denied`). Both ScreenCaptureKit answers share one 5 s budget;
  the helper has 30 s in all, and only its normal end with one complete answer and
  a PNG that decodes at the reported size is a capture. The result is capture-only
  (`capture_kind: window`, `mapped_input_available: false`): no `observation_id`,
  transform or `display`; its `auto_attach_source` names that window, so a newer
  capture of it replaces only that window's older picture in what the model is sent. A window that is not on
  screen (`window.on_screen: false`: minimized, another Space) may come back blank.
- `click` (left/right/middle, double/triple), the thin aliases
  `double_click`/`triple_click` (models call these names unprompted), `move`,
  `left_click_drag`, `mouse_down`/`mouse_up`, `cursor_position`, `type_text`,
  `key`, `hold_key`, `scroll`, and `wait` execute input through platform tools
  when available (on a local or SSH Mac the skill's own CoreGraphics program, so
  nothing is installed there). Pointer coordinates pass through one normalizer that accepts
  the malformations models actually emit (the pair packed into `x` with `y`
  absent or duplicated) and fails loudly on anything ambiguous.
- `activate_app` brings a running application to the front (name, bundle id /
  WM_CLASS, or `pid:<n>` as `window_list` prints it; resolved from the running
  list, never launched) and reports who is in front afterwards.
- `window_list` lists running applications with their windows: who is in
  front, titles, bounds, pid, and on macOS whether the application has its own
  scripting commands (`scriptable`); `app=` filters, `limit` caps the rows.
- `ax_tree` reads one window as text through the Accessibility API, bounded:
  the frontmost application's window, or `app=`'s without bringing it forward;
  `window=` (a window ID or exact title) reads that window instead.
  On local macOS, `until_text="expected text"` waits for that literal,
  case-sensitive substring in a returned title, description, value or URL.
  It rereads the selected window under the same `timeout_ms` and returns the
  last tree with `until_text.status` (`matched`, `not_observed`, or `unknown`).
- `element_act` performs local macOS `press` or text `set_value` through the
  control's own Accessibility interface. It never activates, focuses, injects
  global input, writes the clipboard or falls back to pointer/keyboard input.
  The application itself may still raise a window: foreground observations
  are evidence, not a universal non-interference guarantee.
- `remote_exec` runs a shell command on the active remote backend only; it
  refuses on local backend. Use ordinary Ouroboros shell/file tools for local
  work.

## Window session (opt-in MCP tools)

Use `screenshot(app=..., window=...)` for one image; it needs no MCP connection.
For repeated observations, the owner can connect `mac_window_session` with **Connect tools**
on the Skills card. It watches a window on the **local Mac**, even if the other desktop tools
currently target a remote computer. Connecting or listing tools may start an idle server;
only `begin` starts capture. The host owns one server per task loop. Ending the loop closes
the server and its capture. The session is not restored after a Stop or restart.

- `mcp_mac_window_session__begin(app, window)` resolves `app` launch-free and selects
  exactly one of its windows as `screenshot(window=)` does (a `window_id` or an exact
  title; none or several is refused). Before starting, WindowServer must also confirm that
  window belongs to the selected PID. A missing window or failed lookup starts no stream;
  a later `begin` can try again. The stream binds the selected window once and does not
  resolve its title again. It starts one ScreenCaptureKit stream of that
  window alone (no cursor, no shadow, no audio; at most 1280x800, 5 frames per second,
  four queued surfaces) and returns facts, no image. One session per task: another
  `begin` is `session_exists` until `release`.
- `mcp_mac_window_session__frame(after_seq, wait_ms)` returns the latest complete image
  as an MCP image beside one JSON object. `seq` numbers the session's images and
  `frame_age_ms` is that image's own age (`frame_time_basis`: the stream's display
  time, else when it was received). `stream.last_report` with its age is separate
  evidence: `idle` says the window had not changed at that moment; it is not a newer
  image. Pass the last `seq` as `after_seq` to receive an image only when a newer one
  exists (`wait_ms`, at most 10000, waits for it); otherwise the answer is
  `image: false`, `no_newer_frame`, and the same picture is not sent again. Reset this
  cursor after a new `begin`: sequence numbers belong to that session, not the task's history.
- `mcp_mac_window_session__release()` ends the stream; without a session it is a
  no-op. `stop_unconfirmed` means the stream did not confirm its stop: call it again,
  since no other session begins until it is confirmed.

Besides `no_frame_yet` and `no_newer_frame`, `frame` can withhold an image for these reasons:

- `frame_withheld`: macOS supplied no usable pixels (`blank`, `suspended`, `stopped`,
  `invalid_sample`, `callback_error`, or `inactive` on macOS 15.2+). No earlier picture is
  substituted. A new complete frame from the same stream can still be delivered.
- `window_lookup_unavailable`: a window lookup that worked at `begin` now gives no answer.
  No frame is delivered on this call. A later call can check again.
- `window_gone`: WindowServer no longer lists the selected window for its process. The server
  checks during frame requests and while idle (`window_gone_check: window_server`). It permanently withholds frames from this session and asks macOS to
  stop capture, even if no one requests another frame. Reuse of the ID does not restart it.
- `stream_stopped`: the stream ended. See `stop.reason` and, if a native end arrived after
  delivery closed, `stream_end.reason`. Ask the owner before beginning again after `user_stopped`.

For `window_gone` and `stream_stopped`, `cleanup.confirmed` reports whether macOS confirmed
that capture is over. A missing window or image is not confirmation. `release` frees the
session slot, retrying an unconfirmed stop. `window.on_screen: false` by itself means neither
closure nor destruction. A Close button may only hide a window that macOS can still capture.

The images are capture-only, like `screenshot(window=)`: one window's pixels, no
`observation_id`, no display mapping (`mapped_input_available: false`), and no input
tool may use their coordinates; act through `element_act` or a mapped `screenshot`.
The observer does not retarget input tools. It requires macOS 14+ and the ordinary Screen
Recording grant. `begin` checks without prompting; without the grant it returns
`screen_recording_denied`. Other platforms or older macOS return `unsupported_platform` or
`unsupported_os`. macOS normally marks a running capture
with its own indicator; an owner who ends the capture there is reported as
`user_stopped`. The server holds only the latest frame; each image a task asked for
is kept by the host with that call's result.

## Choosing a control path

Prefer an application's own scripting commands when they express the task. For
background UI work, use `element_act(app=..., do="press"|"set_value",
select={identifier, role, label, nth}, window=..., text=...)`: name only the
selector fields needed, from a fresh `ax_tree`/`window_list`. `window` is a
window ID or exact title; `identifier` is AXIdentifier, `label` the full exact
title/description, and `nth` is one-based among current matching controls.
Tree IDs and coordinate marks are not reusable element handles. One helper
process resolves the application, finds and revalidates the element, acts and
reads it back; each call pays one helper startup and a bounded walk. Ambiguous
or incomplete selection refuses. `timeout_ms` is ONE budget from the start of
the helper's `element` operation (after its interpreter has started; the
application's resolution included), with or without `observe`: the search gets what is
left, and an element found and checked after it is spent is not acted on
(`budget_exhausted`, `dispatched: false`). The settle and readback after an
action take only what remains; a readback with nothing left is skipped and says
so, and the action's receipt stands. The budget is checked between steps, not a
wall-clock cap: a step already running finishes under its own bound (an
Accessibility call up to 1 s, an `lsappinfo` read up to 8 s), so a call can
outlast `timeout_ms`; the hard stop is the helper's process limit (about
`timeout_ms` + 10 s).

`observe=true` adds `observation`: the same window element read again after the
one action (or after a refusal, read-only), using `ax_tree`'s node shape and
`text_limit`; `window` is an object containing its title and available window ID.
The tree is bounded by this call's `max_nodes` (default 600; `ax_tree`: 150). It spends the same `timeout_ms`: the read gets what is left,
the window's facts included (`budget_exhausted` when nothing is). The
receipt is printed once complete (after the readback and `front_after`) and
before the read starts, so a read that fails, runs out or is
cut off by the helper's timeout (`helper_stopped`) leaves the action's receipt
as it was: the top-level outcome is always the action's. A helper stopped
before that line, with or without `observe`, leaves no receipt: the outcome is
unknown (the action may have run), as below. A call carries one
action at most. Several independent calls in one response are admitted one by
one; a step that depends on an earlier step's outcome belongs to the next round.

With `observe=true`, `until_text="expected text"` polls only that observation;
the action, target selection and AX exposure are each done once. The checkpoint
still precedes the first observation. Choose a positive piece of expected UI
text, such as part of the loaded thread, rather than treating a nonempty tree
as readiness. The match is a literal, case-sensitive substring within one
returned `title`, `description`, `value` or `url`; it never joins fields or
matches unseen text beyond `text_limit` (the legacy `label` is not searched).
Use `text_limit` large enough for the expected text and `include_web=true` when
it belongs under an AXWebArea. `matched` returns the read it was found in,
whose `truncated_by`, `errors`, and text length fields still disclose cuts. At
the deadline, `not_observed` means the last read was whole and did not show the
text, not that the app lacks it; any cut (skipped web content, `max_nodes`,
clipped text) leaves `unknown`. A read the deadline or a failed element cut
short, or one that returned nothing, never replaces an earlier read that
finished its walk: that tree comes back with `tree_read`, `tree_age_ms` and
the `latest` read's loss, as `unknown`. A lost window ends with `window_gone`
and no tree, a read that raised with `read_failed`. Without `until_text`, the
single-read behavior is unchanged.

On local macOS, `ax_tree` and `element_act` first request app-local accessibility
exposure with Electron's documented `AXManualAccessibility=true` attribute. The
one attempt runs under the AX messaging timeout before window selection, and
spends the action's existing time budget. `ax_exposure` records its status and
AX error code: `accepted` confirms the request only, `unsupported` is ordinary
for other applications, `unknown` means a timeout, and `failed` retains other
errors. Reads and actions still use their normal selection and verification;
there is no retry, readiness assumption, activation or input fallback. This
changes the target's accessibility exposure, not macOS accessibility settings.
See [Electron's client-side documentation](https://www.electronjs.org/docs/latest/tutorial/accessibility#macos).

The receipt separates `dispatched`, `action_outcome` and `element_after` from
`front_before`/`front_after`. A completed AX call is not proof of the goal;
verify the application's result. `CannotComplete`, a missing helper answer or
a timeout can follow an effect: re-observe, never blindly replay. A failed
readback does not erase an already completed action. Unsupported controls
require a deliberate choice of another method, never a hidden foreground fallback.

For foreground work, announce the handoff before activation. After the one-shot
`activate_app`, use targeted inputs with `raise_policy="never"`: switching to
another app then makes the next call refuse instead of taking the front back.
Wait for the owner's return and obtain a fresh observation before continuing.
Omitted policy remains `allow` for existing explicitly requested foreground
work. Untargeted input is global and belongs to the same announced handoff,
not background collaboration. Existing task Pause/Stop prevents later tool
starts; it cannot retract an admitted OS action. Front checks are not an
exclusive desktop lease, nor a guarantee against a switch during dispatch.

Acting in a named application (`app=`):

- `click`, `double_click`, `triple_click`, `left_click_drag`, `mouse_down`,
  `type_text`, `key`, `hold_key` and `scroll` take an optional `app`: a running
  application's name (case-insensitive), its bundle id or WM_CLASS, or
  `pid:<n>` copied from `window_list`; on X11 and OSWorld also a window's exact
  title or its `0x…` id. Resolution is exact: no prefix or substring guessing
  (on X11 and OSWorld a substring hit is returned as a candidate, so the exact
  name can be given; macOS and SSH answer with the running list; the one
  looser form is the WM_CLASS rule in the X11 row below, where `wmctrl` joins
  instance and class with a dot).
  On X11 and OSWorld the target is a set of windows: the one window for a
  title or id, every window of that WM_CLASS for a class (a process may host
  several classes, as LibreOffice's Writer and Calc do), every window of the
  process for `pid:`; "in front" means the active window is one of them. The
  tool resolves the name against the running list without launching anything,
  with `raise_policy="allow"` brings it forward if needed, then verifies the
  front before input. `never` refuses instead of activating and attempts no
  switch (`target_lost`, or `front_unread` when the front is unread). Two matches are
  a refusal that lists the candidates; an application that is not running is
  a refusal that lists the running applications; one that did not come to the
  front (macOS 14+ may refuse activation from a background process) is a
  refusal that names who is in front; a front that could not be read after
  the switch is the refusal `front_unread`.
  A refusal sends nothing and carries no effect mark (a switch it observed
  first keeps its own, `activation_done`). An empty `app` (what a voice model
  sends for an unused optional field) means "untargeted": the input goes to
  whatever is in front, as before.
- A `key` sequence (`'down down return'`) validates every step before input or
  activation, applies the raise policy once, and rechecks the front before
  every later step: when the named application
  (or window) has left the front (a `cmd+tab` in the sequence, the owner
  switching away, a sibling window of the same process coming forward for a
  title-named target, a window of the class that was not listed with a
  class-named target, such as a dialog a step opened) the remaining steps are
  not sent (`target_lost`,
  `steps_done` says how many ran); a switch that ran before a step that could
  not be sent keeps its mark (`activation_done`). Without `app=` every step
  goes to whatever is in front at that moment, and the receipt names each
  step's recipient when they differ (`steps_delivered_to`, `"unknown"` for a
  step whose front could not be read). `key` asks `get_stop_state` before a
  command that can raise the target and again before each chord. On macOS the
  activation helper also resolves the app. X11, OSWorld and SSH check after
  resolving it, just before requesting the switch. Each chord checks after
  the local front recheck or before the guest's own front probe. A refusal
  returns `status: stopped` with the host's reason. A confirmed switch keeps
  its receipt. The remaining chords stay unsent. Stop can arrive between
  a check and its command. A hold includes the release, and a drag stays whole. On a sequence that stopped, `dispatched`
  is the fact for the step that failed (`steps_done` counts the steps that ran
  before it; a step whose remote call did not answer is `null`, and the note
  says so, never "not sent"). A tool's registered budget covers one
  input's worst case (`type_text` 180 s, the other inputs 90 s, `move` and
  `mouse_up` 60 s), not a long
  sequence: when the host's budget ends a call, no receipt comes back.
- `move` and `mouse_up` never change which application is in front: a drag
  must end where it started.
- Keyboard focus follows the switch and stays there, as if the owner had
  clicked the application; nothing moves it back. Do this only when the owner
  asked for an action in that application (owner decision 27.10.2026).
- Where an application has its own commands (macOS: `scriptable: true` in
  `window_list`; AppleScript/JXA through `run_command`/`run_script`, e.g.
  `osascript -e 'tell application id "com.apple.iWork.Keynote" to show next'`
  while a presentation plays, or `set current slide of front document to
  slide 2` in the editor), prefer them: they do not depend on which window is
  in front and they can read the state back (`slide number of current slide`).
  A keystroke is the right instrument only where no such command exists, and
  the key that means "next" differs per application and per mode (Keynote
  advances on Right Arrow while presenting, Down Arrow in the editor's
  navigator).

Coordinate contract:

- Mapped input requires the screenshot's `observation_id` and coordinates in
  its image space. A missing, expired or mismatched mapping refuses before
  activation/input; there is no implicit raw-coordinate fallback. The binding
  covers the connection, resolved destination, backend and observed geometry,
  not unchanged page content: obtain a fresh observation after UI changes.
  `raw=true` explicitly selects measured native input coordinates and bypasses
  remapping. A coordinate-free `mouse_up` remains usable to release a hold.
- macOS input consumes LOGICAL points while screenshots are physical pixels;
  the stored transform folds the Retina scale in, computed from the captured
  display's own CoreGraphics bounds (a second display no longer stretches the
  scale), including a negative origin for a display left of or above the main
  one (the input events carry those absolute global points, locally and
  on a Mac over SSH; on a Mac negative raw coordinates are valid input, while
  an X11/Wayland desktop and an OSWorld guest, which have no screen space left
  of or above their origin, refuse them before any call, also when a stored
  transform maps a point there: a screenshot of another display or backend,
  so take a screenshot of this desktop first). Missing exact display bounds
  yield a capture-only result, not a guessed Finder-union mapping. SSH Mac
  capture and mapping use the same primary display; unavailable geometry also
  means capture-only. Raw coordinates require independently verified native points.

Input receipts (the same contract on every backend):

- A successful input call returns `observed_effect: true`: the input command
  ran and reported success. It proves dispatch, not that the screen changed or
  that the task's goal was met; the host's task result review reads it as
  work the task produced.
- Beside it an input receipt carries the place, each field where it is known
  (a refusal carries `target`, `reason` and the front it read, no
  `dispatched`: nothing ran): `delivered_to` (the
  application or window in front when the input was dispatched: app, pid,
  bundle id or class, window id and title where the backend knows them;
  `"unknown"` when no reader answered), `dispatched` (whether the input helper
  ran at all: `false` for a refused combo, a failed clipboard write or a
  Command shortcut its program refused before posting anything, `true` for any
  other helper that ran, whatever it reported, `null` with `transport_failed`
  when a remote call did not answer, so whether it ran is unknown), `front_before` (only when the
  tool switched applications first), `front_after` (who was in front when the tool
  read again after the input: present when that is another place than
  `delivered_to` (another window where the backend knows windows, else another
  application) or when `delivered_to` is unknown; an observation, not proof
  that the input caused a change), `target` (with `app=`: what was requested
  and resolved, `verified` in front before dispatch, `activated` when the front
  was read as another place before and as the target afterwards,
  `already_front`, `method`, `waited_ms` (on a remote: the host's round trip
  for the switch); `activated` is `null` when the previous front could not be
  read), `helper_stderr` when the input helper printed anything, and one
  `note` saying what this tool observed and did not observe (a dispatched input:
  no UI change; a withheld one: "not sent"; a lost remote answer: unknown).
- An input whose helper did not report success (non-zero exit, timeout) is
  `ok: false` without the input's mark; it carries `front_at_dispatch` instead
  of `delivered_to`, because whether anything reached the screen is unknown.
  An input that was never dispatched (a refused combo, the clipboard write
  before a paste failed: `reason: clipboard_failed`) says "not sent" in its
  note and carries `dispatched: false`. A `via=paste` receipt says whether the
  clipboard was replaced where that stage answered (a lost or cut-short reply
  says nothing about it: absence means unknown), also when the paste chord
  after it failed; a
  completed clipboard write is an observed effect of its own and keeps the
  mark (`clipboard_done`), as a switch does (`activation_done`). The fields a
  receipt carries follow what happened: a refusal has `target`, `reason` and
  who is in front; a failed helper has `front_at_dispatch`; a dispatched input
  has `delivered_to`.
- On a local or SSH Mac, the input program (`input_events.js`) and the paste
  chord program check before posting anything whether the process may post
  input events (`CGPreflightPostEventAccess`, prompt-free; the grant first) and
  whether every key a chord names exists in the active keyboard layout (a key
  sequence looks all its characters up in one call that posts nothing, after
  any switch to `app=` and before its first step, so in the target's own layout
  `key("return Ж")` on ABC sends no Return; a lookup that cannot be read is
  `layout_unread` and sends nothing, as is a layout without key data, for a
  single shortcut and the paste chord too; a switch
  before it keeps its mark, and the front is read again after the lookup, so a
  window the owner switched to meanwhile gets no key, and the receipt names who
  received the first key, or who was in front when it was refused, as read then); either refusal posts
  nothing, warns on stderr and exits 0. Such receipts are `ok: false` with
  `reason: permission_denied` / `untypable_characters`, no input mark and, for
  the input and Command chord programs, `dispatched: false` (a completed
  clipboard write or switch before it keeps its own, `clipboard_done` /
  `activation_done`); the error names what was not sent (the input, a shortcut,
  the paste chord). Every refusal for a missing grant carries one structured field
  (input, display and window screenshots, the window session, `ax_tree` and
  `element_act` without Accessibility alike),
  `refusal`: `{"capability_unavailable": true, "permission": "post_events" |
  "accessibility" | "screen_recording", "repair": "owner", "where": "System
  Settings > Privacy & Security > …"}` (`reason` stays the string code). Nothing
  asks the system to show a grant prompt. A successful answer names the
  permission the call proved it holds, `permissions_available`: `post_events`
  for input the program reports it posted (its count, or the chord program's
  key line; an exit 0 without either proves nothing), local or on the SSH Mac,
  a key sequence included, `screen_recording` for a live window session and a macOS
  screenshot whose preflight read `granted` (an unread preflight still captures
  but proves nothing) and for a window capture, `accessibility` for `ax_tree`
  and `element_act`. A window session the system ends for a withdrawn Screen
  Recording grant (-3801) answers its next frame with the same `refusal`. A local
  display `screenshot` without the Screen Recording grant (read with the display
  list) is refused the same way, `note: "nothing was captured"`, instead of
  returning the wallpaper. With `app=` the switch runs before the input program
  reads the grant, and keeps its mark.
- A macOS receipt names its method: `cg_events` (pointer, wheel, keys),
  `unicode_events` (typed text), `command_chord`, or `paste`. Typed text also
  says `chars_sent`, `chars_total` (graphemes) and `override_only`: characters
  no key of the active layout gives, sent on a neutral key with only their
  Unicode string; an application that ignores an event's string (some remote
  desktop clients, virtual machines, games) drops exactly those, so use
  `via=paste` there. Every input program receipt says how many events it
  `posted`: one that posted nothing (an empty text) carries no input mark, and
  one that stopped part-way (its own deadline, a failed post) keeps the mark
  with the counts of what went out. The receipt's `cmd` shows the operations as
  readable JSON, the text as typed.
- A switch that happened before a failed or dropped input keeps the mark
  (`activation_done`): the front changed, which is an observed effect.
- An input of several steps that fails after earlier steps succeeded returns
  `ok: false` with `steps_done` and keeps the mark. A failed or timed-out first
  step, a refusal, reads, `wait` and `remote_exec` carry no input mark (a
  switch or clipboard write that completed before keeps its own). A helper the
  OS could not start (`helper_not_launched`) never ran: `dispatched: false`. `activate_app`
  carries the mark only when the front actually changed to the target; already
  in front is `activated: false` without a mark.

Observation without images (`ax_tree`, `window_list`):

- macOS: a pure-ctypes Accessibility client runs in a short-lived helper
  process (`macos_native.py`), so a stuck application can be cut off and a
  native fault never reaches the host. Each node is read in one multi-attribute
  call under a messaging timeout set for the whole helper process (through the
  system-wide element, so it reaches every element the walk touches); the walk
  is breadth-first from the focused window (`root=focused` starts at the
  focused element), stops at `max_nodes` (default 150, ceiling 2000) or
  `timeout_ms` (default 3000, ceiling 15000) and names every cut in
  `truncated_by`: a node's children are requested up to the remaining node
  budget plus one, so a wide list is cut by `max_nodes` (named whenever the
  budget left children unrequested, whether or not the visited count reached
  it) and the node records `children_unread`; a node or a children list
  the application failed to answer is a cut too (`read_errors`: that subtree
  is not requested, `errors` counts the failures by attribute); web content (`AXWebArea`)
  is skipped unless `include_web=true`. `app=` is resolved in the same helper
  process. Nodes carry `id`, `depth`, `parent_id` (absent on the root), role,
  `label` (the title, else the description; at most 200 characters and no
  length of its own: the whole texts are `title` and `description`), `title` and
  `description` separately, `value`, `url` (a link's AXURL as an absolute
  string), bounds, `enabled: false` and `focused: true` (only those values), all
  from the same one read per node. `text_limit` (default 200, ceiling 100000)
  is the characters kept per title, description, value and url; a cut one of
  those four gives its whole length in `<field>_len`, so a larger `text_limit` (or
  `window=`/`root=focused` to narrow the read) brings the rest. An attribute the application failed to answer
  (an error other than ordinary absence) is counted in `errors` by attribute,
  and an element none of whose attributes could be read is a failed read, not
  an empty node; `marks` number the elements of the marked roles with centre
  points in input space (`raw=true`): a mark is a position, not proof that
  the element reacts to a click. A failed read is `ok: false` with its reason
  (`not_running`, `accessibility_untrusted`, `no_window`, a busy application,
  `accessibility_read_failed` when not one node could be read); it never
  answers with a process list. Measured on 27.10.2026: the old
  AppleScript `entire contents` walk took 18–31 s on a 37-node window; this
  reads it in 1.6 s cold and 50 ms warm.
- `window_list` never renders an unread fact as `false` or `[]`: a window list
  the Accessibility API refused says `windows_unread` (a window attribute it
  refused is listed in that window's `unread`, never shown as `false`; a window
  nothing of which could be read has `unread: {window: <error>}`), an
  unread front leaves the per-row `frontmost` out and says `front_unread`
  (on every backend), a listing that failed is `ok: false`. On X11 and OSWorld
  a row is `frontmost` when it is the active window, not merely a window of
  the active process.
- Linux: no element reader in this skill (`ok: false`, `reason: unsupported`,
  with the window rows); OSWorld remotes return the guest's whole accessibility
  tree (opt-in in the bridge; `app`, `root`, `max_nodes`, `timeout_ms` do not
  apply there); SSH Macs answer `unsupported`.
- Who is in front is read fresh on every call (macOS `lsappinfo`, X11
  `xdotool getactivewindow` matched to the `wmctrl` rows), never from a cached
  AppKit value.

Backend matrix (honest limitations):

- macOS: `screencapture` + `osascript`; `sips` for downscaling;
  `lsappinfo` for the frontmost application; the native helper for windows,
  activation, the element walk, grants and displays. Activation tries AX
  `AXFrontmost`, then `NSRunningApplication`, then System Events, accepting
  each only after a fresh read shows the target in front within `wait_ms` (a
  request the OS accepts and ignores, or completes slower than that, falls
  through to the next); the System Events step can raise a one-time
  Automation consent prompt. Pointer, wheel, keys and text are CoreGraphics
  events posted by the skill's own JavaScript for Automation program
  (`input_events.js` through `osascript -l JavaScript`, built by `mac_input.py`),
  under the grant to post input events (Accessibility; no Automation grant);
  every event carries the skill's mark in `kCGEventSourceUserData`. `click`
  takes left, right and middle with a real click count (double and triple
  register as such); `left_click_drag` posts the way between its ends;
  `mouse_down`/`mouse_up` take any button, and a `move` while a button is held
  is a drag; `cursor_position` reads the pointer without posting anything;
  `scroll` posts wheel lines at the pointer (up/down/left/right, one line per
  step). The window under the pointer receives them, whichever application
  `app=` verified in front: move the pointer over the window first (`move`).
  `hold_key` holds any key or modifier combo, and on an error or at its own
  deadline (below the helper's timeout) the program posts the prepared release
  of everything it pressed, in its `catch`, last pressed first. In `key` combos
  the base key is tapped while modifiers are held (`cmd+s` = hold cmd, tap s).
  On a local or SSH Mac and on X11 a letter's case under a modifier adds
  no Shift (`cmd+S` = `cmd+s`; Shift is named: `cmd+shift+s`), and a bare `A` is a
  capital. On local/SSH Macs, Command plus one printable character presses the
  key that gives that character under Command in the active layout. One only
  Command and Shift give adds Shift on the key an app reads as it: its
  Command+Shift character, or its Shift character where the layout ignores
  Shift under Command (`cmd+?` on ABC is Command, Shift and /). Name a
  shortcut by that character: under a Cyrillic layout `cmd+g`, while its own
  letters (`cmd+п`) are refused, never typed. A missing key or posting grant
  refuses before posting. Other combos with a character press the key that
  gives it in the active layout alone, with Shift, with Option or with both
  (those are then added; `å` on ABC is Option+A); a dead key is never that key,
  and a character the layout lacks is refused before anything is posted. A
  modifier alone (`key("shift")`) is pressed as a modifier, its flag on the way
  down. Named keys are layout-independent virtual key codes (`return`, `tab`,
  arrows, `home`, `end`, `page-up`/`page-down`, `f1`–`f16`, the keypad), posted
  with the class flags a keyboard gives them (arrows Fn and NumericPad, F-keys
  and the navigation block Fn, the keypad NumericPad), which Ctrl+arrow
  shortcuts need, never a modifier someone holds. `mute`, `volume-up`,
  `volume-down`, `brightness-up`, `brightness-down`, `play-pause`,
  `play-next`, `play-previous`, `keys-light-up`, `keys-light-down` and
  `keys-light-toggle` are the system key events a keyboard's media keys send.
  `type_text` sends each grapheme as one key-down/up pair carrying its own
  Unicode string, in any layout and script (Cyrillic under ABC, Latin under a
  Russian layout, emoji, accents), on the key that gives the character when
  the layout has one (a dead key never is: such a character goes on the
  neutral key with its string); a line break (`\n`, `\r` or CRLF) is Return.
  The receipt says how (see "Input receipts"). The program's own deadline and
  the helper's timeout are what the call has left of type_text's 180 s after
  the switch to `app=`, less 10 s for the receipt, so a long text cut short
  answers with `chars_sent` rather than the host's timeout. The skill's own
  POSIX commands reach an SSH Mac as UTF-8 bytes on ssh's stdin (`ssh -T`),
  read whole and run by `/bin/sh` whatever the account's shell (tcsh
  included). A `remote_exec` script is ssh's argument, for the account's own
  shell, while the host can start it so (Windows: the quoted command line up
  to 32,767 UTF-16 units; elsewhere an argument up to 128 KiB); a longer one
  goes on stdin, is read whole and then run by the account's own shell, not as
  a login, with its stdin closed, so it keeps its shell's syntax.
  `key(cmd+v)` sends only the shortcut, without changing the clipboard.
  `type_text` with `via=paste` writes the clipboard (`pbcopy`, given
  UTF-8 and told so whatever locale the process inherited, as over SSH) and
  pastes with Command and the key that gives `v` under Command in the active
  keyboard layout, with exactly the Command flag (an `osascript` JavaScript
  program, the same over SSH, posting under the same grant), after the target
  was verified in front (a refusal leaves the clipboard alone; the receipt says
  whether the clipboard was replaced, also when the paste chord after it
  failed); use it for long text or an application that ignores typed strings.
  Neither the layout's typed
  `v` nor a fixed US key is that key everywhere (measured on an AppKit Edit
  menu: Russian – PC, Dvorak – QWERTY ⌘, Dvorak); a layout with none
  (Turkmen) is `untypable_characters` and nothing is posted. `helper_stderr`
  names the key sent; a dispatched paste does not prove the focused element
  took it. `via` accepts `type` or `paste`; anything else is refused.
  Not yet measured on a real Mac (the events were checked only against inert
  stand-ins): whether natural scrolling flips the posted wheel direction, and
  which applications ignore an event's Unicode string.
- Linux X11: key/hold validation also requires loadable `libX11` (discovered by
  `ctypes.util.find_library`); missing validation refuses without sending a prefix.
  `xdotool` (input and the active window), `gnome-screenshot`/`scrot`
  (capture), `wmctrl` (window rows and `app=` activation through
  `wmctrl -i -a`, verified with `xdotool getactivewindow` matched to the rows,
  so no `getwindowclassname` is needed), ImageMagick (downscaling). Targeting
  needs both wmctrl and xdotool: without xdotool a switch cannot be verified
  and targeted inputs are refused. A name is matched exactly against the
  WM_CLASS as `wmctrl -x` prints it (`instance.Class`: the whole string, the
  part before the first dot or the part after a dot, so `Firefox` names
  `navigator.Firefox` and `Gimp-2.10` names `gimp-2.10.Gimp-2.10`; a fragment
  between dots never does), the window title or `pid:<n>`/`0x<id>`; a class
  is every window of that class, `pid:` every window of the process, a title
  or id that one window, and a window the X server has no pid for is itself.
  A class that is not in front is brought forward through its first-listed
  window (`wmctrl` lists the oldest first): name a window by its title to
  choose one. Window rows are parsed from `wmctrl` text, so a title that
  contains a newline can forge a row (a web page title cannot).
  Function keys and case-sensitive keysyms are supported (`f5`→`F5`,
  `XF86AudioPlay` as-is).
- Linux Wayland: `ydotool` (pointer + typing; requires a running `ydotoold`)
  or `wtype` (typing only), `grim` or `gnome-screenshot` (capture). `xdotool`
  does NOT work on Wayland. `key`, `hold_key`, `app=` targeting and
  `window_list` are **unsupported** on Wayland (the compositor exposes no
  window list or activation to other programs; with `wmctrl` present only
  XWayland windows are listed, and the result says so); the refusal names the
  session so the owner can bring the application forward.
- Remote OSWorld HTTP: `screenshot` uses `GET <target>/screenshot`; input and
  `remote_exec` use `POST <target>/execute`, matching OSWorld's in-VM
  `pyautogui` channel. The runner writes an `osworld_http` connection before
  each task. `app=` is resolved from the guest's `wmctrl -lpxG` rows (the same
  exact matching as local X11), activated by window id (`wmctrl -i -a`) and
  verified by polling `xdotool getactivewindow` up to `wait_ms`; every input's
  guest program reads the active window (`id|pid|title`, two `xdotool` calls
  per read, a third for a window without a pid, the title base64-encoded so that nothing a window is called can
  look like a marker) before and after the action in the same call and withholds a targeted input
  when the active window is not one of the target's windows (`target_lost`)
  or could not be read (`front_unread`); a switch observed earlier in the call
  keeps its mark. The probe cannot raise: for an untargeted input a probe
  failure leaves the place unknown and never fails the action. A targeted
  `type_text` makes one switch for the whole call; a withheld or refused paste
  is final (never retried by typing); a clipboard program that ran and failed
  (`pyperclip` or `xclip` absent) falls back to `typewrite` once with the same
  switch; a remote call that did not answer (`transport_failed`: the paste may
  have run) is final, nothing is retyped, with or without `app=`. With `app=""`
  the action, its exit-code reading and the clipboard/typewrite routing are
  otherwise unchanged, and two calls the base sent are refused before any
  call: a provided negative point for `mouse_down`/`mouse_up` (the base
  pressed at the pointer) and a point that a stored transform of another
  display maps outside the guest (the base clicked the scaled point); what changed on
  the benchmark path is the probe around the action (markers are stripped from
  the returned output), `activate_app` in the tool list, `app` in the input
  schemas and `window_list` answering parsed rows from a base64-encoded
  listing (the last line before its separator, so a login-shell greeting
  cannot hide the windows; `ok: false` when the guest printed no listing or
  one that does not decode); no text a window
  shows can reach the readers as a separator or a marker. Honesty guarantees on this
  backend: `scroll` clicks are 1:1 wheel detents (not multiplied); non-ASCII,
  angle brackets, multi-line or long `type_text` (or `via=paste`) is pasted via
  the in-VM clipboard (`pyperclip` + `ctrl+v`, ASCII `typewrite` fallback) so
  Unicode is not silently dropped; every action's `ok` reflects the guest
  process `returncode`/`status` (HTTP 200 alone is not success); `key`
  `backspace` presses BackSpace and `fwd-delete` presses forward Delete, and a
  letter reaches pyautogui as written (`ctrl+C` = `hotkey('ctrl', 'C')`);
  screenshots are capped at 20 MB and PNG-validated before use.
- Remote SSH macOS: `screenshot` uses remote `screencapture` + `scp`; input runs
  the same input program's source through the Mac's own `osascript -e` (Command-alone
  shortcuts/paste: the shared chord program), so nothing is installed there; wrapped
  so the same ssh call reads the frontmost
  application before and after and withholds a targeted input when the target
  is no longer in front. `app=` is resolved from the Mac's `lsappinfo list`
  (name, bundle id, `pid:<n>`; nothing is launched) and activated by pid
  through System Events (only the integer pid enters the remote command, never
  the name), verified by polling `lsappinfo front` up to `wait_ms`; an input
  the selected input helper cannot express is refused before any switch; a transport failure
  after a switch still reports the switch; an ssh call that ended without the
  wrapper's completion marker (a refused connection, a channel broken after
  the input may have run: `ssh_failed`) leaves the outcome unknown
  (`dispatched: null`, `transport_failed`; a clipboard marker that did come
  back is kept as a fact), never "not sent". An ssh that exits 0 while the
  remote shell printed none of the wrapper's lines ran no script (ssh gave it
  no stdin, as `StdinNull yes` for that host does): `remote_script_not_run`,
  `dispatched: false`. The programs' stderr is read the same
  way as locally, and the wheel, the middle button and `hold_key` work as locally. `type_text` with `via=paste` writes the Mac's clipboard
  (`pbcopy`, told the text is UTF-8 whatever locale ssh forwarded) and pastes
  with the same chord program through the Mac's own `osascript` in the same
  ssh call, and the receipt says whether the clipboard stage completed (a
  failed write sends nothing: `clipboard_failed`, `dispatched: false`).
  `window_list` lists applications and the frontmost one (no window
  titles over SSH); a listing that failed or named nothing is `apps_unread`,
  an unread front says `front_unread`. The remote Mac must have
  Screen Recording, Accessibility (input events, for the process the SSH session
  runs as) and (for activation) Automation grants configured by the owner;
  `test_connection` checks that `screencapture` and `osascript` are there. This skill does not install or store private SSH
  keys.

Connection registry:

- Connections live in `data/state/skills/unix_computer_use/connections.json`.
- Default active connection is `local`.
- `add_connection` can add `osworld_http` or `ssh_macos` connection metadata, but
  it never accepts private key material.
- For SSH, configure `~/.ssh/config`, `IdentityFile`, and/or `ssh-agent`
  yourself. If auth fails, `test_connection` tells you what to fix.
- Agents should add/test/activate connections only when the user or a benchmark
  runner provides an explicit target. They should not scan for arbitrary remote
  machines.
- A disabled active connection FAILS CLOSED: action tools return an error rather
  than silently falling back to the local desktop (an action aimed at a remote
  target must never land on the host).

Permissions note: the remote backends require the `net` permission. `net` needs
no owner grant (skill grants only gate `inject_chat`/`subscribe_event`), and the
launcher names this skill among the bundled seeds that start enabled anyway; the
network is used for configured connections, which the owner, a runner or the
agent can add via `add_connection`. Agents should use only explicitly provided
targets as described above. An owner who turns the skill off keeps it off across
updates. The reads declared in the manifest (`window_list`, `ax_tree`,
`capabilities`, `cursor_position`, `list_connections`) skip the per-call safety
check and are available to observe-only consciousness (owner decision
27.10.2026); `screenshot` and every input stay checked.

Layout: `plugin.py` (registration, coordinate contract, local input),
`place.py` (targets, activation, the receipt composer `_finish` shared by every
transport, window rows, element reads), `remote.py` (OSWorld and SSH
transports), `macos_native.py` (the helper process), `input_events.js` and
`mac_input.py` (macOS pointer, wheel, keys and text as CoreGraphics events, and
the structured grant refusal), `paste_chord.js` (the macOS paste chord, local and over SSH), `macos_stream.py` and
`window_session_server.py` (the window session and its MCP server). The skill intentionally
does not hide OS permission requirements, and missing backends produce explicit
errors with the capability report instead of guessing. The agent should prefer
semantic application APIs when available and should ask the human before
destructive or sensitive UI actions.
