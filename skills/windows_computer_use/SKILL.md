---
name: windows_computer_use
description: Windows desktop reading and element actions through the UI Automation text tree (roles, labels, states, coordinates) - no screenshots, no coordinate clicks.
version: 0.1.0
type: extension
entry: plugin.py
plugin_api: "2.0"
runtime: python3
os: win32
permissions: [tool]
install:
  - kind: pip
    package: comtypes
when_to_use: The agent needs to read what a Windows window says (as text with roles and states), find an element by name/role/automation_id, or act on an element (click a button, set a field, toggle, expand, select) - on machines where screenshots are too slow, too large, or the goal is precision that survives window movement and DPI.
model_experience:
  what_model_sees: Four tools - window_list (visible windows), window_read (a window as a text tree with roles/labels/states/rects), element_find (search the tree by selector), element_act (invoke/set_value/toggle/expand/collapse/select/scroll_into_view/focus on the element a selector names). No images; answers are compact JSON trees.
  token_effect: The tool schemas add a small fixed cost to every round while the skill is enabled; each window_read returns a bounded tree (default 400 nodes, ceiling 5000) that costs tokens proportional to the tree size actually read.
---

# Windows Computer Use (UI Automation)

Reads a Windows window **as text** - roles, labels, values, states, rectangles -
through UI Automation, and performs actions **on named elements** through their
UIA patterns. This is the Windows slot the bundled `unix_computer_use` defers:
it does not duplicate macOS/Linux functionality, and it deliberately does not
take screenshots or click at coordinates.

Ported from Yegor Kosyrev's Praxis/Hélène `uia.rs`/`element.rs`, keeping
its contracts. See [NOTICE](NOTICE) for source links, the upstream Apache-2.0
licence and attribution; the contribution's licensing is a separate matter.

## Tools

- `window_list` - visible top-level windows (hwnd, title, class, pid, rect,
  foreground flag). Use it to pick an `hwnd` for the other tools.
- `window_read` - the whole window (or the one `hwnd` names) as a tree/flat
  list of elements: `role`, `name`, `value`, `automation_id`, `class`, `rect`
  with `center`, and a `state` block whose **missing keys mean the element does
  not expose them - not false**. Limits (`max_nodes`, `max_depth`,
  `max_children_per_node`, `max_text_chars`, `timeout_ms`) come back in the
  answer with their ceilings; every cut is named in `truncated_by` and `notes`.
  `backend=win32` forces the plain Win32 reader (classic child windows only);
  `auto` uses UIA and falls back to Win32 honestly, naming `fallback_reason`.
- `element_find` - search the tree by selector (`automation_id`, `role`,
  `name`, `name_contains`, `value_contains`; AND semantics), returning every
  match up to `limit`. `timeout_ms=0` (default) means "look now"; a positive
  value waits for the first match.
- `element_act` - do something **to the element a selector names**:
  `invoke | set_value | toggle | expand | collapse | select |
  scroll_into_view | focus`. The action goes through the element's UIA
  pattern - there is **no fallback to clicking at coordinates**; if the
  element lacks the pattern the refusal lists the patterns it does support.

## Selector contract (the important part)

- All given fields must match at once ("and", not "or"). `automation_id` is
  exact and case-sensitive; `role`/`name` fold case (Cyrillic included).
- **Two matches without `select.nth` is a refusal** listing all candidates -
  the tool never picks for you.
- "Not found" and "didn't finish reading the window" are different answers
  (`searched_whole_window`, `reason`) and are never confused.
- Ids index one answer only; re-find elements by `automation_id`/`hwnd`.

## Honesty rules

- Receipts say what was invoked and what the element looks like **after**;
  whether the goal was achieved stays your judgment.
- An unreadable competing candidate aborts the choice rather than guessing.
- A hung window produces a bounded, partial, honestly-marked answer (or a
  named fallback), never silence. Because COM can hang, the skill runs as an
  isolated-dependency extension: every call executes in a short-lived child
  process the host can kill.

## Scope and safety

Windows-only (`os: win32`). All actions are meant for supervised, low-risk
workflows under human observation; prefer semantic application APIs when they
exist, and ask before destructive or sensitive UI actions. `focus` and
`scroll_into_view` are view-only (non-mutating); the rest change the target
window.
