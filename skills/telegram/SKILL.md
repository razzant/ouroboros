---
name: telegram
description: Owner-only Telegram text bridge and Mini App gateway for the existing Ouroboros interface.
version: 1.4.0
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
  what_model_sees: No new tools; Telegram owner text, photos and files enter normal chat. Replies to mirrored messages carry host reply references for the mind to interpret, never routing instructions. Quiz answers use the web card's decision ingress. Answers to earlier owner messages use native Telegram replies.
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
cards, and opt-in notifications are mirrored back to that owner. Telegram's typing
indicator stays on while a reply you asked for, mirrored to this chat, is actively
being prepared by Ouroboros or a working helper; it stops for Pause, Stop now and
after the answer, and a question waiting for you, a model-access wait or background work
never keep it on (DESIGN §9). A quiz card is
answered by tapping an option or by replying to the card with a free-form
answer; both reach the same host decision ingress as the web UI. The bridge speaks
the install's interface language (Settings → Appearance → Language, or `/language`
here): its own lines are English source translated by the install's translation
memory, so the bot ships no dictionary and follows any language the owner chooses. Owner `/panic`
is available in every command mode, including strict: it sends once through the emergency door,
never queues or replays. Telegram dates before the host's whole-second import time are refused;
their previous outcome is unknown. This assumes aligned clocks: a host ahead rejects fresh Panic
until Telegram reaches that second; a host behind can accept retained commands. Dates in the
boundary second are accepted. Other slash commands keep their command-mode rules even in replies; quote a literal command
as code or include it inside an explanation to send it as a quiz answer.

Version 1.3 remembers every mirrored part, including media and completion
notices, in a bounded durable message map. Replying passes its host row or work
reference through normal chat; Ouroboros decides what it means. Unknown or
evicted ids add nothing. Answers to an earlier owner message from this Telegram
chat use a native reply; answers to its latest owner message stay unquoted.
Silent mode replaces a progress note only within one task episode and reply
origin; Continue episodes remain separate messages. Relay success stays silent; a host-reported
failure arrives as one line. A relayed row quotes the owner's words verbatim as
plain text, under where he wrote them and with the file names, above the note; a
note between works that quotes none of his words stays in its room.
Quiz buttons and replies keep their decision path; earlier parts of a long
question carry ordinary references.

Version 1.3.2 warns when the derived reference map cannot be saved, without
blocking an owner message or later outbound parts. Completion notifications
reference their work, not an authored answer shown only in Main.
Version 1.3.3 gives a finished root task's notice the task card's cost words for
its whole tree: `$3.42` when final, `up to $3.42` while costs may still arrive,
`Tracked:` when some step has no price; an unknown cost adds nothing.
Version 1.3.6 sends Wrap up's saved-work receipt, which the host types as a terminal
receipt, even with progress mirroring off; other card rows still follow that toggle.
Version 1.3.7 sends no finish notification for a run that finished quietly.
Version 1.3.8 paints the Mini App's connecting page in the interface's light or dark:
the Mini App's saved Light or Dark, else Telegram's own, as the page that follows it does.
Version 1.3.9 makes Silent mode mean no sound: replies, the final answer and progress,
Ouroboros's own technical notes included, arrive without sound, every authored message
stays its own message, and only a one-message progress note is replaced by the next one
of its episode; questions, files and problems keep their sound. A question card is the
answer's receipt: it shows the answer Ouroboros recorded, never the words or option sent
from here, and whether its delivery is saved for the task, pending or unknown; no line
says the task read it. A tap also gets a toast. A tap or a text answer gets one message
only for a conflict, which names the recorded answer, a closed question, an error, an
unconfirmed answer, or a card that could not be updated.
A card goes out as one message only while its answered form still fits in the install's
language. A request Ouroboros refused is told in one message replying to it, even to the
latest message: its target and the host's reason, then the choice of target when one is
needed, then the files that were not attached. The System line the web shows for the same
refusal comes here only when no acknowledgement carried the reason, and an accepted
request stays silent.

Version 1.4.0 can use a Bot API server you already run instead of the cloud
(`TELEGRAM_API_BASE`, `TELEGRAM_API_FILES_DIR`, below). The polling position
belongs to one server: a newly chosen server is read from its oldest pending
update, and a batch already handled is not handled again.

Version 1.2.1 marks questions that wait for an owner answer and clears the
waiting line after an answer submitted through Telegram.
Version 1.2.3 serves the stored settings on GET settings/save so the Settings
form shows saved values before Save, and shortens the four long option labels.
Version 1.2.4 introduced a non-clean completion notice even with the task toggle off;
version 1.2.14 replaces that policy with a true opt-out (below).
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

Version 1.2.14 makes `TELEGRAM_NOTIFY_TASKS=off` suppress every additional task
completion notice, failures and cancellations included. On sends completion
notices with the host's material causes; pure review provenance and missing
host-verification receipts stay in task details, without declaring success.
Required, unresolved or unknown review obligations remain disclosed. Previously
stored Off values now mean no extra notices; no setting is migrated or added.
The status-only Main mirror follows this toggle too; preserved intermediate
content still arrives. Ordinary replies, files and questions are independent, as is
`TELEGRAM_SILENT_MODE` (replies without sound). IDs observed while Off
remain seen, so enabling does not replay that backlog. Older summary rows without
the host's compact cause projection retain their original reason text when On.

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

A Bot API server you already run (`tdlib/telegram-bot-api`) can replace
`https://api.telegram.org`. `TELEGRAM_API_BASE` names its origin: `http://` or
`https://`, a host and an optional port, with no path, query or fragment (for
example `http://127.0.0.1:8081`). Every Telegram call of this skill then goes
there: polling, replies, downloads, notifications and the Mini App menu button.
Moving the bot to a different server follows the procedure with Ouroboros quit,
below. Another address of the same running server can be saved live in this
skill's Telegram settings on the Settings page of the computer running
Ouroboros (the Mini App and remote clients cannot change it, as with the owner
binding). A Save applies to the next send; polling moves after its current long
poll or retry wait (up to about a minute). An empty value means
`https://api.telegram.org`, and the form refuses an invalid value with an
error naming the key. The server receives the bot token in every request path
and becomes the source of the bot's updates, so prefer a loopback address or an
`https://` origin: a plain `http://` origin on another machine carries the
token unencrypted. A saved `TELEGRAM_PROXY` carries these calls too, and so
does an environment or system proxy the install already uses; to reach a
loopback origin directly, list its host in `NO_PROXY`. If the server refuses
the token for good, polling waits, sending it nothing more, until the token,
the base or the files directory changes. This skill neither installs nor runs
the server; it keeps polling (no webhook) and keeps its 10 MiB download and
50 MiB upload limits.

A server started with `--local` answers `getFile` with an absolute path in its
own file store and does not serve that file over HTTP. Set
`TELEGRAM_API_FILES_DIR` to that store: the same absolute path must exist on
this computer and be readable by Ouroboros (for a container, mount the store at
the identical path). Downloads then read the file from disk, only inside that
directory; a path outside it, including one reached through a symlink, is
refused. An emptied field keeps downloads on HTTP.

Moving the bot to a different server is manual, and Ouroboros never calls
`logOut` or `close` for you. Do it with Ouroboros quit: `telegram-bot-api` launches
a logged-out bot again on any request to that server, so a running bridge or Mini
App companion would undo the call with its next poll.

1. Have the new server running if it is yours (with `--local`, at the store path
   you share), and let the bridge answer pending messages. Updates the old server
   holds and Ouroboros has not read do not move with the bot: treat them as lost.
2. Copy `<data>/state/skills/telegram/settings.json` to a backup beside it, as
   private as the original, since it can hold proxy credentials. `<data>` is the
   data folder, `~/Ouroboros/data` by default.
3. Quit Ouroboros so that none of its processes remain; closing or hiding the
   window is not enough. Stop every other client of this token as well, such as
   a Presence connector for Telegram.
4. Call one method on the old server, then send it nothing more. The token is
   part of the request path: keep it out of shell history, logs and shared text.
   - Cloud to your server: `logOut` on `https://api.telegram.org`. Without it
     Telegram does not guarantee that updates reach your server. After a
     successful `logOut` the bot cannot log in to the cloud again for 10 minutes.
   - Between two of your servers: `close` on the old one. It returns error 429
     during the first 10 minutes after the bot was launched there; wait and repeat.
   - Your server back to the cloud, at least 10 minutes after the cloud `logOut`:
     `logOut` (or `close`) on your server.
5. With Ouroboros still quit, edit only these two keys in the skill's
   `settings.json`, adding them if absent and keeping every other key. The global
   settings document `<data>/settings.json` holds neither of them, so leave it alone.

   ```json
   "TELEGRAM_API_BASE": "http://127.0.0.1:8081",
   "TELEGRAM_API_FILES_DIR": "/absolute/path/to/server-file-store"
   ```

   Use `""` for `TELEGRAM_API_FILES_DIR` on a server without `--local`, and `""`
   for both to return to the cloud.
6. Start Ouroboros. It reads the new server from its oldest pending update. If it
   reports invalid settings, quit and restore the backup or correct the two keys;
   the backup does not undo the `logOut` or `close`.

Poll each bot token from one place only: another program polling the same token
takes updates this bridge would then miss. While a competing poll holds the token
(error 409), the bridge logs one warning and retries with its usual backoff, so it
recovers by itself when the other poll stops.

With the default base the Bot API stays `https://api.telegram.org` with TLS
through the proxy and no redirects. Proxy credentials follow the selected
protocol: HTTP and SOCKS do not encrypt the connection to the proxy; HTTPS
does. Invalid proxy settings
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
