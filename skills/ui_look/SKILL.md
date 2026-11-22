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
when_to_use: "When seeing Ouroboros's own interface helps a judgement: after changing how it looks, or when asked to look it over, to see the places a change touches at the owner's window, a phone and both themes."
model_experience:
  what_model_sees: "One tool, look. It answers JSON: what the change touches and why, the measurements per screen (engine, theme, viewport: covered text, small targets, overflow, cut text, uneven rows, long text), the panel widths used and, when frames are asked for, PNG paths in uploads/screenshots, each captioned with the engine and the version (served or draft) that drew it; with a draft and frames, what the draft adds is counted as new. No image is attached; view_image opens a frame."
  token_effect: "The tool schema costs about two thousand characters per round while enabled. An answer lists only the screens with findings (at most 60 rows); the full report is a file beside the frames."
---

# Look at the interface

The eye of the `ui_design` skill, as a tool: `look` runs that skill's read-only stand
(`skills/ui_design/references/live_stand.py` of the running app) in a child process and
returns what it measured. Ouroboros may call it when seeing the interface helps its
judgement, and it gives no verdict. The tool's own tests run in the UI browser lane; no
test, check, release step or schedule looks at the product's screens with it.

- **Where it looks.** By default the places a change touches: the diff between the draft
  `web/` (`web_root`, else this task's body candidate, else the system copy this work
  edits) and the served one, or the files in `changed`. A page whose module imports a
  changed module is looked at whole, with each of its views (tabs, a Dashboard view, a
  Settings or Extensions page); a class only an app-level module draws, or a class of a
  changed CSS rule (with its breakpoint), is looked for on every screen. `web/ui.css`,
  `web/theme.js`, `web/app.js`, `web/index.html`, a changed token or a new order of a
  stylesheet's rules touch every place. `places` names places instead; `rooms` opens the
  first Project rooms whatever the plan.
- **How it looks.** At the owner's window from the task's `owner_client` fact (its size and
  pixel ratio; a touch window measures 44 px targets, through the hit area the page adds)
  and at the other kind of window: a 390 px phone beside a desk, a desk beside a touch or
  narrow window. Both themes, with the owner's saved panel widths, which the page reads
  from the server itself. Engines: the pinned Chromium the browser tools use, with their
  launch arguments, and Playwright WebKit only where it is already installed, labelled as
  WebKit emulation: neither is the owner's macOS window. Nothing is downloaded; with no
  browser the tool says so and starts nothing.
- **Before and after.** With a draft and frames, a second page shows the served interface
  on the same data: every screen it also has is a pair (served, then draft) whose frames
  name their version, a changed class is looked for on both (so a part the draft hides
  still shows where it stood), and a finding the draft adds is counted as new. That takes
  about 20–40% longer and holds a second page in memory; without frames only the draft is
  measured, and new and earlier findings are not told apart.
- **Safety.** The stand answers every write locally and drops the page's socket frames, so
  pressing anything inside it changes nothing on the install.
- **What stays.** The report and, when asked, the frames are flat files in
  `uploads/screenshots` named after the work; the existing age-out of that folder removes
  them. A look with frames of 20 screens writes about 5–20 MB; `frames: all` keeps up to
  eight frames a screen (twice that in pairs), so every place at the 1500 s maximum can
  leave about 1 GB until the age-out.
- **Time.** Minutes: `budget_sec` (default 480) bounds the run, the work's deadline and
  its execution window bound the budget, and Stop (or a Stop that cannot be read) ends the
  child at once; what was measured before stays in the report. Every place on two engines
  needs a `budget_sec` near its 1500 s maximum.
