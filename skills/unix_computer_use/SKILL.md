---
name: unix_computer_use
description: Local and remote desktop observation/input tools with coordinate normalization (local macOS/Linux by default; optional OSWorld HTTP and SSH Mac backends).
version: 0.4.1
type: extension
entry: plugin.py
plugin_api: "2.0"
runtime: python3
permissions: [tool, subprocess, net]
env_from_settings: []
when_to_use: The user asks Ouroboros to inspect a desktop, take a screenshot, click/type/press keys, drag, or operate a local/explicitly configured remote GUI under human or benchmark supervision.
model_experience:
  what_model_sees: Roughly two dozen desktop tools (screenshot, click/type/key/scroll input, connection registry) join the tool list, and each screenshot auto-attaches a WXGA-bounded image to the conversation in the same round.
  token_effect: The tool schemas add a moderate fixed cost to every round while the skill is enabled; each screenshot adds one image attachment (the dominant per-use cost), so prefer batching actions per capture.
---

# Unix Computer Use

This bundled extension exposes a reviewable computer-use substrate for local
macOS/Linux desktops and explicitly configured remote targets. **All actions are
for supervised, low-risk workflows under human or benchmark observation.** Windows
support is deferred to a separate backend/skill.

Tools:

- `capabilities` reports the platform, display-session type, and detected
  screenshot/input backends plus the active connection/backend.
- `list_connections`, `add_connection`, `test_connection`, `activate_connection`,
  and `use_local` manage the local/remote connection registry in skill state.
- `screenshot` captures the desktop, downscales the image to fit WXGA
  (1280x800, configurable) and persists the exact image-to-input coordinate
  transform. Pass coordinates read off the returned image directly to the
  input tools — they are remapped automatically. The result carries the typed
  `auto_attach_image` field, so the host attaches the image to the
  conversation in the same round (v6.81.1) — no `view_image` call is needed
  for a fresh capture; the returned `path` stays `view_image`-readable for
  re-viewing later.
- `click` (left/right/middle, double/triple), the thin aliases
  `double_click`/`triple_click` (models call these names unprompted), `move`,
  `left_click_drag`, `mouse_down`/`mouse_up`, `cursor_position`, `type_text`,
  `key`, `hold_key`, `scroll`, and `wait` execute input through platform tools
  when available. Pointer coordinates pass through one normalizer that accepts
  the malformations models actually emit (the pair packed into `x` with `y`
  absent or duplicated) and fails loudly on anything ambiguous.
- `window_list` lists visible windows/processes where the platform exposes a
  lightweight backend.
- `ax_tree` returns a set-of-marks accessibility snapshot of the frontmost
  window on macOS: numbered interactive elements (role, title, center
  coordinates in input space — click them with `raw=true`). It degrades
  honestly to a process/window list when the AX walk fails or on Linux.
- `remote_exec` runs a shell command on the active remote backend only; it
  refuses on local backend. Use ordinary Ouroboros shell/file tools for local
  work.

Coordinate contract:

- Input tools accept coordinates in the LAST screenshot's image space and
  remap them through the stored transform; `raw=true` bypasses remapping.
- macOS input consumes LOGICAL points while screenshots are physical pixels;
  the stored transform already folds the Retina scale in. On multi-display
  Macs the scale is approximate (`approx` flag) — prefer `ax_tree` marks.

Backend matrix (honest limitations):

- macOS: `screencapture` + `cliclick` + `osascript`; `sips` for downscaling.
  `scroll` is **unsupported** (cliclick has no scroll-wheel command — page via
  `key` with `page-down`/`page-up`); `middle` click and non-left
  `mouse_down`/`mouse_up` are unsupported via cliclick. `hold_key` holds
  PURE-MODIFIER combos only (`cmd`, `cmd+shift`, ...): cliclick `kd:`/`ku:`
  accept only modifiers and `kp:` cannot hold — non-modifier holds report an
  honest error. In `key` combos the base key is tapped while modifiers are
  held (`cmd+s` = hold cmd, tap s). TCC permission state (Screen Recording /
  Accessibility) is NOT verified: tools exit 0 even when denied
  (wallpaper-only capture / dropped input) — ensure grants.
- Linux X11: `xdotool` (input), `gnome-screenshot`/`scrot` (capture),
  `wmctrl` (windows), ImageMagick (downscaling). Function keys and
  case-sensitive keysyms are supported (`f5`→`F5`, `XF86AudioPlay` as-is).
- Linux Wayland: `ydotool` (pointer + typing; requires a running `ydotoold`)
  or `wtype` (typing only), `grim` or `gnome-screenshot` (capture). `xdotool`
  does NOT work on Wayland. `key` and `hold_key` are **unsupported** on
  Wayland (`ydotool key` takes raw keycodes only — combos would silently
  no-op, which this skill refuses to fake); use `type_text` for text.
  Pointer press/release uses ydotool button masks (`0x40`/`0x80`).
- Remote OSWorld HTTP: `screenshot` uses `GET <target>/screenshot`; input and
  `remote_exec` use `POST <target>/execute`, matching OSWorld's in-VM
  `pyautogui` channel. The runner writes an `osworld_http` connection before
  each task. Honesty guarantees on this backend: `scroll` clicks are 1:1 wheel
  detents (not multiplied); non-ASCII `type_text` is pasted via the in-VM
  clipboard (`pyperclip` + `ctrl+v`, ASCII `typewrite` fallback) so Unicode is
  not silently dropped; every action's `ok` reflects the guest process
  `returncode`/`status` (HTTP 200 alone is not success); `key` `backspace`
  presses BackSpace and `fwd-delete` presses forward Delete; screenshots are
  capped at 20 MB and PNG-validated before use.
- Remote SSH macOS: `screenshot` uses remote `screencapture` + `scp`; input uses
  a remote input helper path. The remote Mac must have Screen Recording and
  Accessibility grants configured by the owner. This skill does not install or
  store private SSH keys.

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
no owner grant (skill grants only gate `inject_chat`/`subscribe_event`), but it
does remove this skill from the launcher's native auto-enable class — so on a
fresh install the owner must ENABLE it explicitly (a benchmark runner does this
via `save_enabled`). Local-only computer use is unaffected once enabled.

The skill intentionally does not hide OS permission requirements, and missing
backends produce explicit errors with the capability report instead of
guessing. The agent should prefer semantic application APIs when available and
should ask the human before destructive or sensitive UI actions.
