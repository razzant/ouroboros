---
name: Device voice
description: Read Ouroboros replies aloud using the voices available on your device.
version: 1.0.0
type: extension
entry: plugin.py
plugin_api: "2.0"
runtime: python3
os: any
permissions: [page_module]
when_to_use: The owner wants an Ouroboros reply read aloud.
model_experience:
  what_model_sees: No new tools or messages; reading is an interface action.
  token_effect: No model tokens for reading or while idle.
contributes:
  - id: device_voice
    provides: tts
    label: Device voice
    runtime: client
    destination: device
    render: {kind: host, entry: ui/tts.js}
---

# Device voice

Enabled with the bundled skills. “Read aloud” reads a reply; “Stop” ends it.
Local voices appear first. The voice menu says “On this device” only when the
browser marks that voice local; other voices say “Online voice” and may send text
to the operating system's speech service. Voice availability and languages depend
on this device. No model is called and no model tokens are used.
