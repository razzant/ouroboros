---
name: telegram
description: Owner-only Telegram text bridge and Mini App gateway for the existing Ouroboros interface.
version: 1.2.14
type: extension
entry: plugin.py
plugin_api: "2.0"
runtime: python3
os: any
permissions: [net, read_settings, widget, route, supervised_task, subscribe_event, inject_chat, subprocess, companion_process]
env_from_settings: [TELEGRAM_BOT_TOKEN]
subscribe_events: [chat.outbound, chat.typing, chat.photo, chat.video, chat.document, chat.links, chat.quiz, chat.quiz_state]
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
answer; both reach the same host decision ingress as the web UI. The bridge speaks
the install's interface language (Settings → Appearance → Language, or `/language`
here): its own lines are English source translated by the install's translation
memory, so the bot ships no dictionary and follows any language the owner chooses. Slash commands
keep their ordinary command-mode rules even in replies; quote a literal command
as code or include it inside an explanation to send it as a quiz answer.

Version 1.1 adds richer Telegram formatting, native MP3/M4A playback, and
inline link keyboards. Version 1.1.1 fixes the task-done push, which read the
lifecycle axis as a whole object and warned on every finished task. Version 1.2
adds inbound files, answerable quiz cards, a truthful `degraded` bridge status
while token validation is deferred by a dead network, and a task-done push whose
word and icon follow the host's task phase (done, done with warnings, failed,
cancelled).

Version 1.2.1 marks questions that wait for an owner answer and clears the
waiting line after an answer submitted through Telegram.
Version 1.2.3 serves the stored settings on GET settings/save so the Settings
form shows saved values before Save, and shortens the four long option labels.
Version 1.2.4 always sends one short line when a task does not finish cleanly, with
the same status word and reason sentence the task card shows; the task-completion
toggle now only adds the clean finishes.
Version 1.2.5 shows the whole quiz card: its project, the host's facts about the
asking task, and every option's detail, with localized field names; a card too
long for one Telegram message arrives as ordered parts followed by the keyboard
message, and nothing authored is cut.
Version 1.2.6 edits a sent question card on its lifecycle (`chat.quiz_state`): a
web answer settles it, a closed wait drops the waiting line yet keeps buttons,
a finished task says a late answer still counts, and (1.2.13) a question closed
unanswered states that and its reason without buttons. Answers and closures are final.
An open question (no options) is the same whole card without buttons; it asks for
a reply in your own words.

Version 1.2.9 routes every Telegram API call through the optional skill-local
`TELEGRAM_PROXY` setting, including polling, sends, downloads, notifications,
and the companion's menu button lifecycle. The Settings form masks the proxy
and never reads its stored credentials back into the browser.

Version 1.2.10 also forwards the owner's notes and skill notices (`reminder` and
`skill_notice` System rows, written for the owner) to the pinned chat when the
mirror mode is `telegram_only`; mode `all` already mirrored them.

Version 1.2.11 speaks the install's interface language: `/language` lists English
and the languages the install already has, `/language <name or tag>` chooses one
for the whole installation through the host's one writer, and the bridge's own
menu, buttons, card hints and notifications are read through the install's
translation memory (a line it lacks is generated like a browser miss). The
retired bridge-only `TELEGRAM_LANGUAGE` setting is migrated once.

Version 1.2.12 starts the Mini App when the Ouroboros server is a container's
PID 1 without an init. PID 1 counts as the server only when the host named
itself so; a companion adopted by init or a subreaper still stops.

The Mini App exposes the unchanged Ouroboros SPA through the established
owner-authenticated sidecar and a pinned Cloudflare Quick Tunnel. It is enabled
by default after owner binding and can be turned off independently without
stopping the text bridge. Disabling the skill destroys process-memory Mini App
sessions, stops public exposure, and best-effort restores the prior Telegram
menu button. Rotate the bot token only while the skill is disabled, then
re-enable it.

Delegated task cards lead with the executor's latest words (marked earlier
when retained), followed by problems and compact activity counts. These are
attributed observations, not the supervising task's narration or a completion
receipt; journal gaps and preview omissions remain visible.

Set `TELEGRAM_BOT_TOKEN` in Settings, grant it to this skill, enable the skill,
and send the bot a private message to bind the owner. No legacy Telegram skill
state is copied or changed. Installations that use `telegram-bridge` or
`telegram-miniapp-poc` must disable or remove those skills before enabling this
one.

If Telegram requires a proxy, open this skill's Telegram settings and enter
`TELEGRAM_PROXY` as `scheme://[user:password@]host[:port]` (`socks5`, `socks5h`,
`http` or `https`; SOCKS needs an explicit port). Leave the masked field empty
to keep a saved value, or use **Clear saved Telegram proxy** to remove it.
Disable and re-enable the skill after changing the proxy so its poller and
companion also pick up the change. No additional Secrets grant is needed.

To route Bot API traffic through a self-hosted `telegram-bot-api` server
instead of `https://api.telegram.org`, set `TELEGRAM_API_BASE` to its root URL
(`scheme://host[:port]`, no path, query or fragment; e.g.
`http://192.168.1.48:8081`). All Bot API calls — polling, sends, downloads,
notifications, and the companion/menu egress — go to that base; transport
security is whatever the chosen origin provides, so an `http://` base is
unencrypted between this installation and the server (fine on a trusted LAN,
not fine over the internet). A local server
serves downloaded files from its own disk instead of HTTP (tdlib/
telegram-bot-api#26): point `TELEGRAM_API_FILES_DIR` at the server's file
store. **The store must be shared by path**: the directory is mounted (bind
mount/volume) at the *same absolute path* for both the server and this
installation, so the absolute `file_path` the server returns in `getFile` is
directly readable here and stays confined to the store (not the filesystem
root). File downloads fall back to reading from that store. Disable and
re-enable the skill after
changing either setting. Both settings live in this skill's Settings form;
invalid values are rejected with an error naming the key. With neither set,
behavior is unchanged: the default `https://api.telegram.org` and HTTP
downloads.

With the default base the Bot API stays `https://api.telegram.org` with TLS
through the proxy and no redirects; a custom `TELEGRAM_API_BASE` origin
replaces the target, and transport security is then whatever that origin
provides (see above). Proxy credentials follow the selected protocol: HTTP and
SOCKS do
not encrypt the connection to the proxy; HTTPS does. Invalid proxy settings
produce an error naming the key without its value. With no skill proxy, the
bridge retains its existing direct/ambient-proxy behavior and the companion's
Telegram calls stay direct. This setting does not change the application proxy,
Cloudflare tunnel, Mini App web traffic, or local host requests.

The Mini App supports macOS arm64/x86_64, Linux arm64/x86_64, and Windows
x86_64. Only the explicit unsupported OS/architecture case degrades
independently: the text bridge remains available while Mini App status reports
that no pinned cloudflared asset exists. Invalid host runtime, unsafe state, or
companion registration errors fail the skill load instead of claiming a partial
healthy installation.

Registration, pinned cloudflared selection, and runtime status share one
architecture helper. Only when `platform.machine()` is empty on Windows does
it fall back to `sysconfig.get_platform()`: `win-amd64` selects the pinned
Windows asset; ARM64, 32-bit, and unknown builds remain unsupported in this
fallback. A nonempty machine value takes precedence over the interpreter build;
macOS and Linux use only the machine value.

The Mini App is Beta. Its best-effort Cloudflare Quick Tunnel has no SLA and
does not support Server-Sent Events (SSE). It targets native Telegram clients;
Telegram WebA/WebK are not supported.
