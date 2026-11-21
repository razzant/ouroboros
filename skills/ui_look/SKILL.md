---
name: ui_look
description: "One read-only tool that looks at Ouroboros's own interface on the live data: the places a change touches, at the owner's window and a phone, in both themes; measurements first, frames as files on request."
version: 0.1.0
type: extension
entry: plugin.py
plugin_api: "2.6"
runtime: python3
os: any
permissions: [tool, subprocess, net, fs]
tools:
  - name: look
    effect: read
when_to_use: "After changing how Ouroboros's own interface looks, or when asked to look it over: see the touched places as the owner sees them before judging, showing or merging the change."
model_experience:
  what_model_sees: "One tool, look. It answers JSON: what the change touches and why, the measurements per screen (engine, theme, viewport: covered text, small targets, overflow, cut text, uneven rows, long text), the panel widths used and, when frames are asked for, PNG paths in uploads/screenshots, each captioned with the engine that drew it. No image is attached; view_image opens a frame."
  token_effect: "The tool schema costs about two thousand characters per round while enabled. An answer lists only the screens with findings (at most 60 rows); the full report is a file beside the frames."
---

# Look at the interface

The eye of the `ui_design` skill, as a tool: `look` runs that skill's read-only stand
(`skills/ui_design/references/live_stand.py` of the running app) in a child process and
returns what it measured. Ouroboros calls it when its own judgement says a look helps; no
test, check, release step or schedule calls it, and it gives no verdict.

- **Where it looks.** By default the places a change touches: the diff between the draft
  `web/` (this task's body candidate, or `web_root`) and the served one, or the files in
  `changed`. A page whose module imports a changed module is looked at whole; a class only
  an app-level module draws, or a class of a changed CSS rule, is looked for on every screen.
  `web/ui.css`, `web/theme.js`, `web/app.js`, `web/index.html` or a changed token touch
  every place. `places` names places instead.
- **How it looks.** At the owner's window from the task's `owner_client` fact (its size and
  pixel ratio; a touch window measures 44 px targets) and at a 390 px phone, in both themes,
  with the owner's saved panel widths, which the page reads from the server itself. Engines:
  the pinned Chromium the browser tools use, and Playwright WebKit only where it is already
  installed, labelled as WebKit emulation: neither is the owner's macOS window. Nothing is
  downloaded; with no browser the tool says so and starts nothing.
- **Safety.** The stand answers every write locally and drops the page's socket frames, so
  pressing anything inside it changes nothing on the install.
- **What stays.** The report and, when asked, the frames are flat files in
  `uploads/screenshots` named after the work; the existing age-out of that folder removes
  them. A look with frames of 20 screens writes about 5–20 MB.
- **Time.** Minutes: `budget_sec` (default 480) bounds the run, the work's own deadline
  bounds the budget, and Stop ends the child at once; what was measured before stays in
  the report.
