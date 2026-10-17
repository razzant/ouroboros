---
name: meeting_browser
description: Experimental. Ouroboros's own Chrome window joins a Zoom web, Google Meet or Microsoft Teams meeting as a participant during a live call, heard through a script-replaced microphone instead of any audio driver.
version: 0.1.0
type: extension
entry: plugin.py
plugin_api: "2.0"
runtime: python3
os: any
permissions: [tool, call_peer, companion_process, subprocess, net]
env_from_settings: []
when_to_use: During a live call the owner wants Ouroboros itself to take part in an online meeting, heard by the other participants (experimental; until the service's participant probe passes, the join says so).
model_experience:
  what_model_sees: Three tools (join, leave, status). While joined, the meeting's audio arrives in the call as meeting audio and your speech also reaches the meeting.
  token_effect: Small fixed schema cost per round while enabled; meeting audio is billed as ordinary call input audio.
timeout_sec: 120
install_specs:
  - kind: pip
    package: playwright==1.61.0
    check: {argv: [python3, -c, "import playwright.sync_api"]}
companion_processes:
  - name: meeting_participant
    command: [python3, scripts/meeting_participant.py]
    runtime: python3
    restart_policy: on_failure
    max_restarts: 5
---

# Meeting browser participant (experimental)

Ouroboros joins an online meeting as itself: its own visible Chrome window opens
the meeting link, and a page script replaces that window's microphone with the
live call's voice and taps the meeting's incoming audio. No audio driver is
installed and the owner's microphone and camera are never used. Everything
happens inside a **live call**: the owner presses Call, then asks Ouroboros to
join; when the call ends, the window leaves the meeting and closes. Outside a call
`meeting_join` refuses and asks the owner to press Call.

## Which services work

Every service is offered and stays **experimental** until its participant probe
passed, because a page may refuse a replaced microphone and only a real meeting
shows it. The shipped record (`vendor_probes.json`) and an owner-run probe on this
computer say what is known; `meeting_join` reports it in its result:

| Service | Shipped probe | Known before probing |
|---|---|---|
| Zoom web (`*.zoom.us`) | NOT_RUN | Zoom's web client shows guests a CAPTCHA since 2026-07-22 |
| Google Meet (`meet.google.com`) | NOT_RUN | Meet holds unknown guests in a queue until the host admits them |
| Microsoft Teams (`teams.microsoft.com`, `teams.live.com`) | NOT_RUN | |

Until a service passes, the join result carries an `experimental` note: if the
page refuses the replaced microphone, the participant only listens and says so,
and the `meeting_audio` skill can still listen to the owner's own meeting. To
record a probe, the owner runs, with a real meeting that someone else is in:

```text
python3 scripts/probe_vendor.py --state-dir <data>/state/skills/meeting_browser --url <meeting link> --seconds 90
```

joins from the window that opens (sign in, click Join, get admitted) and says
something. The probe records PASS only when the page asked for the microphone,
sent it over WebRTC, audio bytes left, and remote meeting audio arrived; otherwise
FAIL with the missing facts. `--report` prints the table.

## How a join works

1. `meeting_join(url)`: the service comes from the link's host. The companion
   attaches to the call as a `meeting` peer with a speaker (the meeting hears the
   call's voice through this window only), then opens the link in Ouroboros's
   window. Microphone permission is granted only to that service's origins.
2. The page may show its own join or sign-in screen. The owner signs Ouroboros's
   own Zoom, Google or Microsoft account into this window once (the profile is
   kept in `data/state/skills/meeting_browser/chrome-profile`, never the owner's
   browser profile). Clicking **Join** is done in the visible window, for example
   with the computer-use tools (`screenshot`, then `click`).
3. `meeting_status` reports from what the page itself observed: **speaking**
   once the page sent Ouroboros's microphone over WebRTC and it is live,
   otherwise **listen-only** with the reason (not joined yet, muted by the page, or
   the page has not taken the replaced microphone). A service that refuses the
   replaced microphone therefore shows as listen-only, never as speaking.
4. `meeting_leave` (or the end of the call, or closing the window) leaves and
   closes the window.

The window's own sound is muted so the meeting does not play from the speakers
(and echo into any microphone in the room); Ouroboros hears it through the tap.
Video requests receive a plain "Ouroboros" card, never a camera.

## Requirements and limits

- Google Chrome installed (Playwright's `chrome` channel; no browser download).
  Without it the refusal is `MEETING_BROWSER_CHROME_MISSING`.
- Playwright is this skill's isolated dependency, installed by the host after review.
- Page content policies that refuse the audio worklet fall back to the same
  arithmetic on the page's main thread (`audio_mode` in the status).
- One meeting at a time; a new join leaves the previous one.

## State and troubleshooting

`data/state/skills/meeting_browser/`: `control.json` (private control channel),
`chrome-profile/` (Ouroboros's browser profile), `vendor_probes.json` (owner-run
probe results).

- `MEETING_NO_CALL`: not inside a live call; the owner presses Call first.
- `MEETING_VENDOR_UNKNOWN` / `MEETING_LINK_INVALID`: not a Zoom, Meet or Teams https link.
- `MEETING_BROWSER_PROFILE_BUSY`: Ouroboros's browser window is already open elsewhere.
- `MEETING_CALL_GONE`: the call ended before the participant could join.
- `MEETING_COMPANION_UNAVAILABLE`: the skill is disabled, its dependency is still
  installing, or the companion is restarting.
