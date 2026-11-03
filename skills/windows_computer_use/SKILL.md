---
name: windows_computer_use
description: Windows desktop reading and element actions through the UI Automation text tree (roles, labels, states, coordinates) - no screenshots, no coordinate clicks.
version: 0.1.1
type: extension
entry: plugin.py
plugin_api: "2.0"
runtime: python3
os: win32
permissions: [tool]
install:
  - kind: pip
    package: comtypes==1.4.17
when_to_use: The agent needs to read what a Windows window says (as text with roles and states), find an element by name/role/automation_id, or act on a named element (press a button, set a field, toggle, expand, select) so that the action survives window movement and DPI. It reads what the application exposes through UI Automation, not pixels.
model_experience:
  what_model_sees: Four tools - window_list (visible windows), window_read (a window as a text tree with roles/labels/states/rects), element_find (search the tree by selector), element_act (invoke/set_value/toggle/expand/collapse/select/scroll_into_view/focus on the element a selector names). No images; answers are compact JSON trees.
  token_effect: The tool schemas add a small fixed cost to every round while the skill is enabled; each window_read returns a bounded tree (default 400 nodes, ceiling 5000) that costs tokens proportional to the tree size actually read.
---

# Windows Computer Use (UI Automation)

Reads a Windows window **as text** - roles, labels, values, states, rectangles -
through UI Automation, and performs actions **on named elements** through their
UIA patterns. It ships with Ouroboros and is on by default on Windows only
(`os: win32`); macOS and Linux use the bundled `unix_computer_use`. It
deliberately does not take screenshots, click at coordinates or send general
keystrokes, so surfaces that expose no UI Automation elements (canvas-drawn
apps, remote desktops, some games) read as empty or partial.

Ported from Yegor Kosyrev's Praxis/Hélène `uia.rs`/`element.rs`, keeping
its contracts. See [NOTICE](NOTICE) for source links, the upstream Apache-2.0
licence and attribution; the contribution's licensing is a separate matter.

## Tools

- `window_list` - visible top-level windows that have a title (hwnd, title,
  class, pid, rect, foreground flag), front to back; untitled windows are left
  out. It stops at `limit` (default 100, at most 500, echoed back); a listing
  cut there says `truncated_by: [limit]` and `total_known: false`. Use it to
  pick an `hwnd` for the other tools.
- `window_read` - the whole window (or the one `hwnd` names) as a tree/flat
  list of elements: `role`, `name`, `value`, `automation_id`, `class`, `rect`
  with `center`, and a `state` block whose **missing keys mean the element does
  not expose them - not false**. Limits (`max_nodes`, `max_depth`,
  `max_children_per_node`, `max_text_chars`, `timeout_ms`) come back in the
  answer with their ceilings; every cut is named in `truncated_by` and `notes`,
  including `result_size` when a large reading is cut to what the host relays.
  `backend=win32` forces the plain Win32 reader (classic child windows only);
  `auto` uses UIA then Win32, naming `fallback_reason`. Each reader has its
  `timeout_ms` budget; the host allows 50 seconds for both plus setup.
- `element_find` - search the tree by selector (`automation_id`, `role`,
  `name`, `name_contains`, `value_contains`; AND semantics), returning every
  match up to `limit`. `timeout_ms=0` (default) means "look now"; a positive
  value waits for the first match.
- `element_act` - do something **to the element a selector names**:
  `invoke | set_value | toggle | expand | collapse | select |
  scroll_into_view | focus`. The action goes through the element's UIA
  pattern - there is **no fallback to clicking at coordinates**; if the
  element lacks the pattern the refusal lists the patterns it does support.
  `set_value` needs `text` as a string: `""` clears the field, null is refused.

## Selector contract (the important part)

- All given fields must match at once ("and", not "or"). `automation_id` is
  exact and case-sensitive; `role`/`name` fold case (Cyrillic included).
- **Two matches without `select.nth` is a refusal** with the match count and
  candidates (a display prefix if too large); the tool never picks for you.
- Supplied selector text fields must be non-blank strings. A number or null is
  rejected, not dropped; matching keeps the full original strings and whitespace.
- "Not found", "didn't finish reading the window" and "an element did not
  answer a field the selector names" are different answers
  (`searched_whole_window`, `reason`, `unreadable_candidates`) and are never
  confused: an unreadable candidate makes `element_act` refuse.
  On `element_act`, `stopped_before_timeout_ms` and `last_walk_ms` disclose an early wait stop.
- Ids index one answer only; re-find elements by `automation_id`/`hwnd`.

## Honesty rules

- Receipts say what was invoked and what the element looks like **after**;
  whether the goal was achieved stays your judgment. `action_outcome=completed`
  means the call returned, not that the goal succeeded. Failed after-reads retain
  that outcome with `after_read=partial|unavailable` and named unread fields.
  An exception inside an action is `action_outcome=unknown`, distinct from a
  pre-call refusal; a worker that did not answer in time also leaves
  `action_call_made=unknown`. A host timeout/kill may prevent that receipt entirely;
  it proves no outcome. Check the window before retrying, never retry blindly.
- A completed mutating act (all but `focus` and `scroll_into_view`) carries
  `observed_effect: true`: its pattern call returned on that window, which is
  not proof that the goal was met; unknown outcomes and refusals never carry it.
  `delivered_to` names that window (`hwnd`, `title`, `pid`) whenever the call was made.
- A field the window failed to report is listed in the node's `unread`, not
  shown as missing.
- All four tools fit the host's JSON-string relay. Oversized displays name
  `result_size`, list `shown`/`omitted` counts and per-object `clipped_fields`
  (original/shown UTF-16 units); these are display cuts, not unread fields.
  Text is capped at 4000 UTF-16 units only when the answer is too large, then
  lists keep a fitting prefix. A document's text in receipts and candidates
  shows its first 4000 units (`text_truncated` when longer); `value_contains`
  matches the whole text. Narrow the selector or read/filter an inner
  window for more; action facts and original match counts are never discarded.
- A hung window normally produces a partial answer or named fallback; a slow
  child setup or host stop can instead return a host error. The skill runs as an
  isolated-dependency extension: every call executes in a short-lived child
  process the host can kill. The host installs tested `comtypes==1.4.17` into the skill's own
  environment while the skill is on (at boot, or when it is turned on); a
  failed install stays the skill's dependency state and is retried at boot.

## Scope and safety

Windows-only, in the interactive session Ouroboros runs in: windows of
elevated processes may refuse an unelevated Ouroboros, and the secure desktop
is out of reach. All actions are meant for supervised, low-risk workflows under
human observation; prefer semantic application APIs when they exist, and ask
before destructive or sensitive UI actions. `focus` and `scroll_into_view` are
view-only (non-mutating); the rest change the target window. The owner can
turn the skill off on the Skills page; updates keep that choice.
