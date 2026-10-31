---
name: Browser speech
description: Free dictation using your browser's speech recognition. Audio may go to the browser vendor.
version: 1.0.0
type: extension
entry: plugin.py
plugin_api: "2.0"
runtime: python3
os: any
# net: the browser sends the owner's audio to its vendor. It is not on by default (skills/builtin.json).
permissions: [page_module, net]
when_to_use: The owner wants to dictate a draft using browser speech recognition.
model_experience:
  what_model_sees: Only the draft text the owner chooses to send; no new tools.
  token_effect: No model tokens for recognition or while idle.
contributes:
  - id: speech
    provides: stt
    label: Browser speech
    title: "Free. Your browser recognizes speech: Chrome sends audio to Google, Edge to Microsoft, Safari to Apple (on device when available)."
    runtime: client
    destination: vendor:Browser speech service
    render: {kind: host, entry: ui/stt.js}
---

# Browser speech

Find this engine in the composer's “+” → “Add dictation”: it is not installed by
default, and no audio leaves the page until it is chosen there as the dictation
engine. Dictation writes an editable draft; only Send submits it.

Chrome recognition sends audio to Google, Edge to Microsoft. Safari uses Apple's
speech service, on device when available. Other engines may use an online service;
the browser controls processing, availability and language support. This skill
makes no promise of local recognition. No model is called and no model tokens are
used. A browser without speech recognition offers another engine instead.
