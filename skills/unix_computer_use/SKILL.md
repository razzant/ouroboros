---
name: unix_computer_use
description: Local and remote desktop observation and input, with macOS semantic actions, explicit foreground control, screenshot-bound coordinates and honest outcome receipts.
version: 0.6.0
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
when_to_use: The user asks Ouroboros to inspect a desktop, take a screenshot, act in a named application (click/type/press keys there), drag, or operate a local/explicitly configured remote GUI under human or benchmark supervision.
model_experience:
  what_model_sees: Desktop tools include bounded AX reads, local macOS element_act press/set_value without activation or pointer input, explicit foreground input, screenshot observation IDs, capabilities and connections. Screenshots auto-attach as images; receipts separate action dispatch, readback and front observations.
  token_effect: The tool schemas add a moderate fixed cost to every round while the skill is enabled; each screenshot adds one image attachment (the dominant per-use cost), each input receipt about a hundred tokens of place facts, and an ax_tree read grows with the nodes it returns (bounded by max_nodes).
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
  display was chosen: `default`, `app`, or `fallback` when the named window
  could not be located (`app_window_unread` says why) and the main display was
  captured instead; the result then carries no `app`. Linux and the remotes
  capture their one screen and ignore `app=` (the result says so). Pass
  coordinates read off the returned image together with its `observation_id`
  to the input tools; mapping is bound to that captured observation. The result carries the typed `auto_attach_image`
  field, so the host attaches the image to the conversation in the same round;
  an engine that cannot take images receives the path instead (host behaviour),
  so the text facts in every other tool are the way to check without a picture.
- `click` (left/right/middle, double/triple), the thin aliases
  `double_click`/`triple_click` (models call these names unprompted), `move`,
  `left_click_drag`, `mouse_down`/`mouse_up`, `cursor_position`, `type_text`,
  `key`, `hold_key`, `scroll`, and `wait` execute input through platform tools
  when available. Pointer coordinates pass through one normalizer that accepts
  the malformations models actually emit (the pair packed into `x` with `y`
  absent or duplicated) and fails loudly on anything ambiguous.
- `activate_app` brings a running application to the front (name, bundle id /
  WM_CLASS, or `pid:<n>` as `window_list` prints it; resolved from the running
  list, never launched) and reports who is in front afterwards.
- `window_list` lists running applications with their windows: who is in
  front, titles, bounds, pid, and on macOS whether the application has its own
  scripting commands (`scriptable`); `app=` filters, `limit` caps the rows.
- `ax_tree` reads one window as text through the Accessibility API, bounded:
  the frontmost application's window, or `app=`'s without bringing it forward.
- `element_act` performs local macOS `press` or text `set_value` through the
  control's own Accessibility interface. It never activates, focuses, injects
  global input, writes the clipboard or falls back to pointer/keyboard input.
  The application itself may still raise a window: foreground observations
  are evidence, not a universal non-interference guarantee.
- `remote_exec` runs a shell command on the active remote backend only; it
  refuses on local backend. Use ordinary Ouroboros shell/file tools for local
  work.

## Choosing a control path

Prefer an application's own scripting commands when they express the task. For
background UI work, use `element_act(app=..., do="press"|"set_value",
select={identifier, role, label, nth}, window=..., text=...)`: name only the
selector fields needed, from a fresh `ax_tree`/`window_list`. `window` is a
window ID or exact title; `identifier` is AXIdentifier, `label` the full exact
title/description, and `nth` is one-based among current matching controls.
Tree IDs and coordinate marks are not reusable element handles. The helper
resolves, revalidates, acts and reads back in one process; each call pays helper
startup and a bounded walk. Ambiguous or incomplete selection refuses.

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
  step whose front could not be read). On a sequence that stopped, `dispatched`
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
  one (cliclick then receives `=`-prefixed absolute coordinates, locally and
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
  ran at all: `false` for a refused combo or a failed clipboard write, `true`
  for a helper that ran, whatever it reported, `null` with `transport_failed`
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
- On a local or SSH Mac, `cliclick` exits 0 and only warns on stderr when the
  posting process lacks the Accessibility grant (the OS drops the event) or
  when a character has no key code in the active keyboard layout (nothing is
  typed; measured 27.10.2026 under a Cyrillic layout). Such receipts are
  `ok: false` with `reason: permission_denied` / `untypable_characters` and no
  input mark (a completed clipboard write or switch before it keeps its own,
  `clipboard_done` / `activation_done`); the hint names the method that failed
  (typing, a shortcut, the paste chord).
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
  is skipped unless `include_web=true`. Nodes carry role, label, value (capped,
  with `value_len`), and bounds; an attribute the application failed to answer
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

- macOS: `screencapture` + `cliclick` + `osascript`; `sips` for downscaling;
  `lsappinfo` for the frontmost application; the native helper for windows,
  activation, the element walk, grants and displays. Activation tries AX
  `AXFrontmost`, then `NSRunningApplication`, then System Events, accepting
  each only after a fresh read shows the target in front within `wait_ms` (a
  request the OS accepts and ignores, or completes slower than that, falls
  through to the next); the System Events step can raise a one-time
  Automation consent prompt. `scroll` is **unsupported**
  (cliclick has no scroll-wheel command — page via `key` with
  `page-down`/`page-up`); `middle` click and non-left `mouse_down`/`mouse_up`
  are unsupported via cliclick. `hold_key` holds PURE-MODIFIER combos only
  (`cmd`, `cmd+shift`, ...): cliclick `kd:`/`ku:` accept only modifiers and
  `kp:` cannot hold — non-modifier holds report an honest error. In `key`
  combos the base key is tapped while modifiers are held (`cmd+s` = hold cmd,
  tap s). On a local or SSH Mac and on X11 a letter's case under a modifier adds
  no Shift (`cmd+S` = `cmd+s`; Shift is named: `cmd+shift+s`), and a bare `A` is a
  capital. `type_text` with `via=paste` writes the clipboard (`pbcopy`) and
  pastes with `cmd+v`, after the target was verified in front (a refusal leaves
  the clipboard alone; the receipt says whether the clipboard was replaced,
  also when the paste chord after it failed); use it for
  long or non-ASCII text, which `cliclick t:` types slowly. `via` accepts
  `type` or `paste`; anything else is refused.
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
- Remote SSH macOS: `screenshot` uses remote `screencapture` + `scp`; input uses
  `cliclick` on the Mac, wrapped so the same ssh call reads the frontmost
  application before and after and withholds a targeted input when the target
  is no longer in front. `app=` is resolved from the Mac's `lsappinfo list`
  (name, bundle id, `pid:<n>`; nothing is launched) and activated by pid
  through System Events (only the integer pid enters the remote command, never
  the name), verified by polling `lsappinfo front` up to `wait_ms`; an input
  cliclick cannot express is refused before any switch; a transport failure
  after a switch still reports the switch; an ssh call that ended without the
  wrapper's completion marker (a refused connection, a channel broken after
  the input may have run: `ssh_failed`) leaves the outcome unknown
  (`dispatched: null`, `transport_failed`; a clipboard marker that did come
  back is kept as a fact), never "not sent". cliclick's stderr is read the same
  way as locally. `type_text` with `via=paste` writes the Mac's clipboard
  (`pbcopy`) and pastes, and the receipt says whether the clipboard stage
  completed (a failed write sends nothing: `clipboard_failed`, `dispatched:
  false`). `window_list` lists applications and the frontmost one (no window
  titles over SSH); a listing that failed or named nothing is `apps_unread`,
  an unread front says `front_unread`. The remote Mac must have
  Screen Recording, Accessibility and (for activation) Automation grants
  configured by the owner. This skill does not install or store private SSH
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
transports), `macos_native.py` (the helper process). The skill intentionally
does not hide OS permission requirements, and missing backends produce explicit
errors with the capability report instead of guessing. The agent should prefer
semantic application APIs when available and should ask the human before
destructive or sensitive UI actions.
