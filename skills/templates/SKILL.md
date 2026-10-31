---
name: Templates
version: 1.0.0
description: Starter buttons for the composer and instructions for making your own.
type: instruction
when_to_use: The owner asks to make, change, or remove a composer button.
model_experience:
  what_model_sees: This short instruction body when helping the owner make a button.
  token_effect: Small instruction context only when used; no tools or idle calls.
contributes:
  - id: explain_simpler
    placement: composer.plus
    order: 1
    label: Explain simpler
    action: {verb: draft.insert, text: "Explain your last answer more simply, with an example."}
  - id: summarize_chat
    placement: composer.plus
    order: 2
    label: Summarize this chat
    action: {verb: draft.insert, text: "Summarize what we decided in this chat and what is still open."}
  - id: check_text
    placement: composer.plus
    order: 3
    label: Check my text
    action: {verb: draft.insert, text: "Check the text below for mistakes and unclear places, then show a corrected version:\n\n"}
---

# Your own buttons

When the owner asks “make me a button that …”, create or edit `my-buttons/SKILL.md`
in the installation's data-plane external skills directory. Resolve that directory
from the current skill environment; do not put user skills in the source repository.
Never edit this bundled `templates` skill: updates reseed it.

Keep one contribution per button and preserve the owner's other entries. Use a
unique stable id, `placement: composer.plus`, a short label (at most 40 characters),
and `action.verb: draft.insert` with the full requested text. These buttons insert
an editable draft; only the owner's Send submits it.

Example user manifest:

```yaml
---
name: my-buttons
version: 1.0.0
description: My composer buttons.
type: instruction
contributes:
  - id: next_steps
    placement: composer.plus
    label: Next steps
    action:
      verb: draft.insert
      text: "Suggest the next steps for what we are working on."
---
My personal composer shortcuts.
```

Run the normal skill lifecycle: preflight → review → enable. Tell the owner the
button appears in the composer's “+” menu under `my-buttons` once enabled. For a
change or removal, edit that entry and repeat the lifecycle.
