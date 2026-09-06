---
name: telegram
description: Owner-only Telegram text bridge and Mini App gateway for the existing Ouroboros interface.
version: 1.2.0
type: extension
entry: plugin.py
plugin_api: "2.0"
runtime: python3
os: any
permissions: [net, read_settings, widget, route, supervised_task, subscribe_event, inject_chat, subprocess, companion_process]
env_from_settings: [TELEGRAM_BOT_TOKEN]
subscribe_events: [chat.outbound, chat.typing, chat.photo, chat.video, chat.document, chat.links, chat.quiz]
conflicts: [telegram-bridge, telegram-miniapp-poc]
when_to_use: The owner wants to communicate with and control Ouroboros through Telegram.
model_experience:
  what_model_sees: No new tools; the skill relays owner messages, photos and files (documents, video, audio, voice) from Telegram into the normal chat and mirrors replies back, so conversation turns may originate from Telegram without any visible difference. An owner's answer to a quiz card tapped or replied to in Telegram arrives through the same decision ingress as a web-card answer.
  token_effect: Near-zero while idle — no per-round schema cost; incoming Telegram media arrive as ordinary attachments and cost what any chat attachment costs.
timeout_sec: 60
companion_processes:
  - name: miniapp_gateway
    command: [python3, scripts/companion.py]
    runtime: python3
    restart_policy: on_failure
    max_restarts: 5
---

# Telegram

One owner-only Telegram integration provides both the established bot bridge
and the optional Mini App. The first positive private Telegram chat binds as
the sole owner. Text, photos and files (documents, video, audio and voice notes within
this integration's 10 MiB download cap) can be sent to Ouroboros, with or without a
caption; replies, photos, videos, documents, typing state, subagent cards, quiz
cards, and opt-in notifications are mirrored back to that owner. A quiz card is
answered by tapping an option or by replying to the card with a free-form
answer; both reach the same host decision ingress as the web UI. Slash commands
keep their ordinary command-mode rules even in replies; quote a literal command
as code or include it inside an explanation to send it as a quiz answer.

Version 1.1 adds richer Telegram formatting, native MP3/M4A playback, and
inline link keyboards. Version 1.1.1 fixes the task-done push, which read the
lifecycle axis as a whole object and warned on every finished task. Version 1.2
adds inbound files, answerable quiz cards, a truthful `degraded` bridge status
while token validation is deferred by a dead network, and a task-done push whose
word and icon follow the host's task phase (done, done with warnings, failed,
cancelled).

The Mini App exposes the unchanged Ouroboros SPA through the established
owner-authenticated sidecar and a pinned Cloudflare Quick Tunnel. It is enabled
by default after owner binding and can be turned off independently without
stopping the text bridge. Disabling the skill destroys process-memory Mini App
sessions, stops public exposure, and best-effort restores the prior Telegram
menu button. Rotate the bot token only while the skill is disabled, then
re-enable it.

Set `TELEGRAM_BOT_TOKEN` in Settings, grant it to this skill, enable the skill,
and send the bot a private message to bind the owner. No legacy Telegram skill
state is copied or changed. Installations that use `telegram-bridge` or
`telegram-miniapp-poc` must disable or remove those skills before enabling this
one.

The Mini App supports macOS arm64/x86_64, Linux arm64/x86_64, and Windows
x86_64. Only the explicit unsupported OS/architecture case degrades
independently: the text bridge remains available while Mini App status reports
that no pinned cloudflared asset exists. Invalid host runtime, unsafe state, or
companion registration errors fail the skill load instead of claiming a partial
healthy installation.

The Mini App is Beta. Its best-effort Cloudflare Quick Tunnel has no SLA and
does not support Server-Sent Events (SSE). It targets native Telegram clients;
Telegram WebA/WebK are not supported.
