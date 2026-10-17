---
name: meeting_audio
description: Lets Ouroboros hear the owner's meeting (Zoom, Teams, a browser meeting) from inside a live call through system audio capture with nothing to install by hand, and on Linux speak into it through a software microphone.
version: 0.1.0
type: extension
entry: plugin.py
plugin_api: "2.0"
runtime: python3
os: any
permissions: [tool, call_peer, companion_process, subprocess, net]
env_from_settings: []
when_to_use: During a live call the owner wants Ouroboros to listen to a meeting running on this computer, to whisper hints or take notes, or (Linux) to speak in it.
model_experience:
  what_model_sees: Five meeting tools (listen start/stop, speak start/stop, status). While listening, the meeting's sound arrives in the call as meeting audio beside the owner's voice, with a context note whenever that audio can contain your own voice.
  token_effect: Small fixed schema cost per round while enabled; meeting audio is billed as ordinary call input audio while listening is on.
timeout_sec: 60
install_specs:
  - kind: pip
    package: pyobjc-framework-ScreenCaptureKit==12.2.2
    platforms: [darwin]
    check: {argv: [python3, -c, "import ScreenCaptureKit, CoreMedia, AppKit"]}
  - kind: pip
    package: PyAudioWPatch==0.2.12.9
    platforms: [win32]
    check: {argv: [python3, -c, "import pyaudiowpatch"]}
companion_processes:
  - name: meeting_capture
    command: [python3, scripts/meeting_capture.py]
    runtime: python3
    restart_policy: on_failure
    max_restarts: 5
---

# Meeting audio

Ouroboros listens to the owner's meeting and whispers hints or writes notes; on
Linux it can also speak in it. Everything happens inside a **live call**: the
owner presses Call in a room, then asks Ouroboros to listen. The companion
captures the meeting application's sound and joins that same call as a
`meeting` peer through the Host Service door (`call_peer`), so the meeting is
heard beside the owner's voice. Ouroboros's spoken hints reach only the owner's
own speaker; the meeting hears nothing unless speaking mode is on (Linux) or the
`meeting_browser` participant is used. Outside a call the tools refuse in plain
words and ask the owner to press Call; this skill never opens a call itself.

Nothing is installed by hand and no audio driver is ever added (no BlackHole, no
VB-Cable). The only installs are this skill's isolated Python packages on macOS
and Windows, which the host installs after review.

## Tools

- `meeting_listen_start(app="")`: listen to the named application (`zoom`,
  `Microsoft Teams`, `Google Chrome` for a browser meeting, ...). Empty listens to
  the meeting applications that are running (Zoom, Teams, Webex, Slack, Discord,
  FaceTime, Skype, Telegram); a browser is never chosen by default because the
  Ouroboros page itself may play in it. A name that matches nothing, or several
  different applications, is refused with the running candidates.
- `meeting_listen_stop()`: stop listening (and speaking).
- `meeting_speak_start()` / `meeting_speak_stop()`: Linux only, see below.
- `meeting_status()`: idle, listening or speaking, what is captured and how, and
  the last reason listening stopped.

Results are short and plain, for example `listening to Zoom`, `waiting for the
macOS permission prompt`, `the call is no longer open`. Every refusal carries a
typed `code`.

## How each platform listens

- **macOS 13+**: ScreenCaptureKit, audio only, with a filter that includes only
  the meeting application, so the whisper played by the owner's surface is not
  captured. The first use asks the owner for **Screen & System Audio Recording**
  (System Settings > Privacy & Security). macOS attributes that permission to the
  responsible process, normally the Ouroboros app; the companion records which
  process it is in `capture_attribution.json` in its state directory without
  prompting, and `scripts/probe_capture.py` reports the same facts (its
  `--capture` mode is owner-run because it can show the prompt).
- **Windows 10 build 20348+ / Windows 11**: WASAPI process loopback of the meeting
  application's process tree, so only that program is heard. On older builds, or
  when Windows refuses process loopback, it falls back to the loopback of the
  default output device (PyAudioWPatch). **That fallback hears everything the
  computer plays, including Ouroboros's own whispered hints.** The status says so
  and the call receives a context note so the mind knows its own voice may be in
  the meeting audio; no timing or loudness rule tries to cut it out.
- **Linux**: PipeWire links the meeting application's own playback streams to a
  private capture node (`pw-record`, `pw-dump`, `pw-link`); new streams of that
  application are linked as they appear. Without `pw-link`, or on PulseAudio
  (`parec`, `pactl`), it records the monitor of the output device the
  application plays to, which also carries everything else playing there; the
  status and a context note say so. No root, no installs; a missing tool is a
  refusal that names it.

**Same browser.** When the meeting runs in a browser, the capture follows that
browser as a whole: tabs of one browser cannot be separated. If the Ouroboros page
is open in the same browser, Ouroboros's whispers are in the meeting audio; the
call is told this. Use a different browser (or the desktop app) for Ouroboros, or
a native meeting app, to keep them apart.

## Speaking mode (Linux only)

`meeting_speak_start` creates a user-space virtual microphone: `pactl` loads a
null sink `ouroboros_voice` and a remapped source `ouroboros_mic` (shown as
**Ouroboros**) over its monitor, and the call's output is played into it with
`pacat` (or `pw-play`). The owner picks **Ouroboros** as the microphone in the
meeting application once. It works under PulseAudio and PipeWire's
pipewire-pulse. The module ids loaded are recorded in `speak_modules.json`
before use, and exactly those ids are unloaded on stop, when the call ends, when
the companion exits, and at the next start after a crash or a refused unload; a
record is dropped only once its module is gone, never while the audio server is
not answering. A device of the same name that this skill did not create is left
untouched and reported. If the player stops, speaking mode ends and the status
says so; listening continues.

On macOS and Windows speaking mode is not offered: it would need an audio driver.
The `meeting_browser` participant (Ouroboros's own browser window joining the
meeting) is the driver-free way to be heard there.

## Operating the meeting

Joining through the desktop app, sharing a window or turning slides uses the
existing `unix_computer_use` tools; this skill adds no operation tools.

1. Join: open the meeting link with the platform opener (`open <link>` on macOS,
   `xdg-open <link>` on Linux) through a shell tool, then `screenshot` and `click`
   the application's Join button; confirm with another `screenshot`.
2. Share a window: `key` the application's share shortcut (Zoom: `cmd+shift+s` on
   macOS, `alt+s` on Linux), pick the window in the picker with `click`, confirm.
3. Next slide: `click` the presenting window to focus it, then `key` `right` or
   `page-down`; previous slide `left` or `page-up`.

Take a `screenshot` before each action and ask the owner before anything visible
to other participants (sharing, unmuting).

## State and troubleshooting

State lives in `data/state/skills/meeting_audio/`: `control.json` (the private
control channel: loopback port and token, owner-only), `speak_modules.json`
(Linux modules this companion loaded), `capture_attribution.json` and
`permission_requested.json` (macOS: who the permission belongs to, and that its
one-time prompt was shown), `process_loopback.json` (Windows: process loopback
turned off and why). Audio is never written to disk by this skill; it goes to
the call only.

- `MEETING_NO_CALL`: not inside a live call; the owner presses Call first.
- `MEETING_PERMISSION_PENDING`: macOS is showing its one-time prompt now.
  `MEETING_PERMISSION_DENIED`: it was asked before and is still not allowed; the
  owner turns it on in System Settings (macOS may ask to reopen the app).
- `MEETING_APP_NOT_FOUND` / `MEETING_APP_AMBIGUOUS`: name one of the listed
  applications with `app`.
- `MEETING_COMPANION_UNAVAILABLE`: the skill is disabled, its dependencies are
  still installing, or the companion is restarting.
- `MEETING_CALL_GONE`: the call ended before the meeting peer could join.
