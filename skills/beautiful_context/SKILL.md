---
name: beautiful_context
version: 1.0.0
description: Inspect and compare local Beautiful Context model-request JSON reports in a sandboxed widget.
type: extension
entry: plugin.py
plugin_api: "2.0"
runtime: python3
os: any
icon: gauge
permissions:
  - widget
  - route
ui_tab:
  id: viewer
  title: Beautiful Context
  render:
    kind: iframe
    route: viewer
    start: auto
    height: 680
    span: 2
when_to_use: The owner wants to inspect or compare local Beautiful Context model-request JSON reports on the Widgets page.
model_experience:
  what_model_sees: Only the catalog entry; no new tools or messages.
  token_effect: A few catalog lines per request while enabled. Viewing reports makes no model calls.
---

# Beautiful Context

Optional widget for inspecting and comparing locally exported Beautiful Context
model-request JSON reports. Off by default; disabling this skill removes its card
from the Widgets board.

## What it does

- Opens one locally selected Beautiful Context JSON report at a time inside an
  opaque, bridge-free sandboxed frame (`web/context-report.html`).
- Shows how many leading UTF-8 bytes match, the first differing item, changed
  fields, and the token counts recorded for each attempt. The selected files
  stay in the frame.
- Reads no server archive, has no network permission, and registers no model
  tools.
- The frame follows the host's colour scheme; an older embedded browser that
  does not pass it on shows the report in the device's colours.

Leaving Widgets clears the open report unless the card's launch policy is
**Keep running**, which may retain it while you are on other pages; the
card's Stop or disabling the skill always clears it.

## Exporting a local report

Generate a report in a private terminal from a separate matching source checkout
(never the running checkout), to a fresh private non-synced destination
outside that checkout and outside Files. Substitute your directory and attempt ID below:

```sh
python -m ouroboros.context_attempt_reader --data-root DATA TASK_ID --attempt ATTEMPT_ID --report > "/path/to/private-directory/ATTEMPT_ID.json"
```

Use a shell that keeps stdout's bytes (a POSIX shell or `cmd.exe`) and give
each attempt its own fresh file — a report belongs to one attempt. Discard
failed exports. Then open the file in the Beautiful Context widget with
**Open local report**.

## Attribution

Originally authored by Egor Kosyrev (`@josephsteuerjr`; PR #3763, PR #4098).
