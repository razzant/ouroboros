---
name: zoom_meeting
description: Joins Zoom meetings by voice through the Zoom Meeting SDK on a Linux server, as one Presence call.
version: 0.1.0
type: extension
entry: plugin.py
plugin_api: "2.0"
runtime: python3
os: linux
permissions: [tool, presence, call_peer, companion_process, net, read_settings]
env_from_settings: [ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET]
when_to_use: Someone asks this install to join a Zoom meeting by link or meeting ID and take part by voice.
model_experience:
  what_model_sees: >-
    Three tools (zoom_join, zoom_leave, zoom_status). A joined meeting becomes one Presence call, a
    live voice turn under the chosen binding's behavior and ceiling, in which the meeting audio
    arrives as the people speaking and the meeting chat, a host mute and the participants Zoom is
    delivering audio for arrive as peer-observed facts. The people in the meeting are correspondents,
    never the owner.
  token_effect: >-
    Three small schemas in every round while enabled. A meeting runs a realtime voice session, billed
    by the call engine for as long as the call lasts; OUROBOROS_PRESENCE_CALL_MAX_MINUTES is the
    owner's ceiling on its length.
timeout_sec: 60
companion_processes:
  - name: zoom_bot
    command: [python3, scripts/zoom_bot.py]
    runtime: python3
    restart_policy: on_failure
    max_restarts: 5
install_specs:
  - kind: pip
    package: PyJWT==2.13.0
    platforms: [linux-x86_64]
  - kind: pip
    package: zoom-meeting-sdk==0.0.27
    platforms: [linux-x86_64]
    check:
      argv: [python3, -c, "import jwt, zoom_meeting_sdk"]
---

# Zoom meeting

This install joins Zoom meetings by voice through the Zoom Meeting SDK and takes part
in each one as a Presence call: a single live voice turn governed by a Presence
binding's behavior skill and capability ceiling. The people in the meeting are
correspondents of that binding, never the owner. Any Linux x86_64 install with its own
Zoom Meeting SDK app can use it.

## What a join does

1. `zoom_join` checks the link or meeting ID, the Zoom app credentials and the Presence
   binding, then hands the join to the `zoom_bot` companion and reports Zoom's first
   answer.
2. The companion starts the SDK, authenticates the app with a short-lived JWT built
   from its credentials, and joins under the display name (default: this install's
   name) with video off and audio on.
3. In the meeting it joins computer audio, attaches its virtual microphone and asks the
   meeting host for access to the meeting audio (Zoom's local-recording privilege). The
   host approves this in Zoom on every join. Until then `zoom_status` says
   "in the meeting, waiting for the host to allow <name>'s audio". A declined request,
   or one the host leaves unanswered until Zoom's own timeout, ends the join and says
   so.
4. When meeting audio flows, the companion asks the Host Service to open the Presence
   call for the binding and attaches to it as the call's `meeting` peer. The meeting's
   mixed audio (32 kHz mono) goes up at the rate Zoom delivers; the call's voice comes
   down into the virtual microphone in exact 20 ms frames, in real time, and an
   interruption drops whatever voice was still queued. Zoom chat lines, a host mute and
   the participants Zoom is delivering audio for reach the call as peer-observed facts.
   "Delivering audio for" means Zoom is sending that participant's own audio stream; it
   is a platform fact, not a voice detector.
5. `zoom_leave`, the end of the meeting, removal by the host, or the end of the call
   (the owner's Stop, the spend limit or `OUROBOROS_PRESENCE_CALL_MAX_MINUTES`) end
   both: the companion says goodbye on the call socket, leaves the meeting and cleans
   the SDK up. A companion that crashed or restarted never re-joins by itself; it starts
   idle and reports what the previous run was doing.

## Setup

- Linux x86_64. The `zoom-meeting-sdk` binding ships Linux x86_64 wheels only, and the
  Zoom Linux SDK loads system libraries, GLib among them. The dependency check imports
  the binding, so a missing library shows as a failed check rather than as a broken
  join.
- Raw meeting audio on Linux needs a PulseAudio server (PipeWire's pulse service
  counts) for the user Ouroboros runs as, and `~/.config/zoomus.conf` containing
  `system.audio.type=default`; without them the SDK joins but delivers no audio.
  Source: Zoom Developer Docs, Meeting SDK for Linux, "Use raw data",
  https://developers.zoom.us/docs/meeting-sdk/linux/custom-ui/advanced-features/raw-data/.
  The companion finds the server through `PULSE_SERVER` or
  `$XDG_RUNTIME_DIR/pulse/native`; Ouroboros passes both variables on from its own
  environment, and the companion uses `/run/user/<uid>` when `XDG_RUNTIME_DIR` is
  unset. A service user without a login session provides a server in one of two
  ways, neither of which the skill does for it: a lingering user manager
  (`loginctl enable-linger <user>`, an administrator's command) creates
  `/run/user/<uid>` and runs the user's PipeWire or PulseAudio service there; or a
  PulseAudio server started as that user, named by `PULSE_SERVER` (for example
  `unix:/path/to/native`) in the environment Ouroboros starts with. `zoom_status`
  says exactly what the companion found missing.
- A Zoom Meeting SDK app of the install's own (Zoom App Marketplace, with the Meeting
  SDK enabled). Save its client id and secret as `ZOOM_CLIENT_ID` and
  `ZOOM_CLIENT_SECRET` in Settings → Secrets and grant them to this skill. The
  companion reads them when it starts; after changing them, turn the skill off and on.
- A Presence binding whose transport is this skill: transport `zoom`, the Presence
  account `zoom_status` shows (`zoom-sdk-app-` plus a digest of the client id, never
  the secret), conversation `*` for any meeting or one meeting number, a behavior skill,
  and call mode on so the meeting runs as one live call. With several Zoom bindings,
  `zoom_join` needs `binding_id`.

## Meetings hosted outside the app's account

Since March 2, 2026, Zoom requires a Meeting SDK app that joins a meeting hosted by
another Zoom account to be authorized for it: an On Behalf Of (OBF) token, or a ZAK
when it joins as a Zoom user. Meetings in the app owner's own account need only the app
JWT. An OBF token comes from Zoom's REST API
(`GET https://api.zoom.us/v2/users/me/token?type=onbehalf&meeting_id=<number>`) with a
user OAuth token that holds the `user:read:token` scope. The user who authorized it must
already be in the meeting, and the bot's session ends when that user leaves. Pass it as
`zoom_join(on_behalf_token=...)`; the companion sets it as the join's `onBehalfToken`,
which the zoom-meeting-sdk binding exposes. A join made before the authorizing user
arrives is refused with `MEETING_FAIL_AUTHORIZED_USER_NOT_INMEETING`; join again once
they are in.

Sources: Zoom Developer Docs, "FAQ - Updates to Meeting SDK authorization",
https://developers.zoom.us/docs/meeting-sdk/obf-faq/; Zoom Developer Blog,
"Transitioning to On Behalf Of (OBF) tokens in Meeting SDK apps",
https://developers.zoom.us/blog/transition-to-obf-token-meetingsdk-apps/.

## Policies that apply

- The Zoom Meeting SDK terms and the app's Zoom App Marketplace terms apply to every
  join; they are the app owner's agreement with Zoom.
- The meeting host approves the bot's access to the meeting audio on every join.
- Participants see the bot in the participant list under its display name (default:
  this install's name, for example "Aika") and hear its voice; it does not join hidden.

## Feasibility probe

On the server, run the probe with the skill's isolated Python, which holds the binding:
`<skill>/.ouroboros_env/python/bin/python <skill>/scripts/probe.py`, with
`ZOOM_CLIENT_ID` and `ZOOM_CLIENT_SECRET` in the environment or with
`--settings <path to settings.json>`. It checks the platform, imports the binding, finds
GLib, looks for the raw-audio prerequisites above, starts the SDK and authenticates the
app without joining any meeting, and prints PASS, FAIL or NOT_RUN for each step, naming
each missing prerequisite. It never prints a secret; `--json` gives the same result as
one JSON object. The probe reads the shell it runs in; the companion runs with the
environment Ouroboros starts it with, and `zoom_status` shows its own result of the same
audio check.

## Limits

- One meeting at a time per install; voice only (no video, no screen sharing).
- Personal link names (`zoom.us/my/<name>`) carry no meeting number: pass the numeric
  meeting ID. ZoomGov meetings need a different SDK domain and are not supported.
- The SDK runs on the companion's main thread under a GLib main loop, through PyGObject
  when it is importable and otherwise through libglib directly.
