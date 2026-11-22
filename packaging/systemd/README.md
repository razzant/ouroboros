# Running native Linux packages under systemd

The native `.deb` and `.rpm` packages ship an optional systemd user unit with a
stable name. It is an alternative way to launch Ouroboros when an operator wants
to use `systemctl --user`; the ordinary desktop entry still starts Ouroboros
directly and is not controlled by this unit.

Choose one launch path for an instance. If Ouroboros is already running from the
desktop entry, close it before starting the unit; the single-instance lock is
shared by both paths.

## Scope

The packaged unit is supported only for the native `.deb` and `.rpm` layout. It
starts the release-reviewed launcher at `/opt/ouroboros/Ouroboros`, the same
entry point as the packaged desktop file. Source checkouts, AppImages, and
tarballs have different locations and are deliberately outside this unit's
contract; containers use [`packaging/docker/compose.yaml`](../docker/compose.yaml)
([`docs/DEPLOYMENT.md`](../../docs/DEPLOYMENT.md)).

The package installs the unit (tracked here as `ouroboros.service.in`: macOS types a
`.service` file as an application) at:

```text
/usr/lib/systemd/user/ouroboros.service
```

Installation never enables or starts it. Starting a desktop agent remains an
explicit user decision.

## Use

Start or stop the packaged runtime explicitly:

```bash
systemctl --user start ouroboros
systemctl --user stop ouroboros
systemctl --user restart ouroboros
systemctl --user status ouroboros
```

After Panic, automatic startup and manual `systemctl --user start` or `restart`
leave Ouroboros stopped; the journal reports "Ouroboros is stopped. Use Start to resume."
Resume explicitly through the desktop entry or `ouroboros run --start ...`.

To start it automatically with the user session:

```bash
systemctl --user enable --now ouroboros
```

Undo that choice with:

```bash
systemctl --user disable --now ouroboros
```

`loginctl enable-linger "$USER"` is an additional, system-level owner choice
when the user service must survive logout. Without linger, the user manager and
its services end with the last session.

## Readiness

`systemctl --user is-active ouroboros` proves only that the launcher process is
running. `/api/state` reporting `supervisor_ready: true` is necessary before
task admission, but it is not sufficient: the worker pool has a separate
admission state and may still return a typed HTTP 503.

A successful task admission is the authoritative proof that work was accepted.
Automation may use a bounded retry policy for the specific typed 503 condition
it supports; it must not treat process activity or HTTP reachability alone as
proof that a task entered the queue. `ouroboros status` remains useful for
checking the loaded branch, SHA, and worker projection.

## Logs

The journal shows user-service lifecycle and launcher output:

```bash
journalctl --user -u ouroboros -f
```

Ouroboros also keeps its normal application logs under
`~/Ouroboros/data/logs/`, including `launcher.log` and `agent_stdout.log`.

## Lifecycle ownership

The unit intentionally has no systemd restart policy. The launcher already owns
managed restart, its bounded crash fuse, panic semantics, and final cleanup.
Adding a second restart owner would let systemd undo a panic stop or restart a
runtime that the launcher deliberately stopped.

`OOMPolicy=continue` (systemd 243 and later; below it systemd does not react to a child's OOM at all)
keeps the service up when the kernel kills a process. `Delegate=memory` and `DelegateSubgroup=runtime`
(systemd 254 and later) put the launcher, server and workers in a `runtime` subgroup beside a
`commands` group: in sessions started by the unit, each local command gets its own memory leaf there
(`ouroboros/command_memory.py`) and no swap, so an OOM at its limit kills that command alone. The `commands` group's
limit is the host total less the 10% reserve and the memory held outside the commands (one snapshot, the `runtime`
subgroup read directly); a new leaf gets all the room that leaves free, and running leaves are only raised. Every
leaf, not only the command's own, is recomputed when any command starts or finishes, and when a running command's
`memory.events` `max` count moves; that last refresh is a poll at the command's wait slice, so an allocation fast
enough reaches its leaf's OOM before it. This holds only for the gated foreground tools (`run_command`, `run_script`,
`verify_and_record` and the local executor) in a unit with systemd 254+ on cgroup v2. If two commands grow together
and fill the shared `commands` limit, a command dies, even a healthy sibling, and it is meant to be the largest: every
process of the command holding most gets a best-effort `oom_score_adj` of 1000, the others their command's share
(only raised, never below 0), refreshed at every start, pressure refresh, finish and wait slice. That choice of
victim is best effort, an accepted residual: the kernel picks a process, not a group, so a command that was largest
keeps its 1000 after it shrinks, and between two commands at 1000 (or before a lagging refresh) the one with the
bigger single process can die even if it is the smaller command. Only commands die either way; the service never. `start_service`,
`skill_exec`, MCP servers, extension processes and Claudexor runs bypass the gate: they stay in `runtime` and are
covered only by `OOMPolicy=continue`, so one of them can still take the whole group's room. A command's output is
still collected whole in its worker, as before this change, so a command streaming gigabytes grows the worker, not
its leaf. While free RAM is under 10%, new gated commands do not start; swap is not counted. This is delivered by the
package, not by Update: a unit already loaded needs `systemctl --user daemon-reload` and a restart, and an older
package, systemd below 254, the desktop entry and AppImage keep the old behaviour; `/status` says on Linux whether
commands are isolated, and why not.

`KillMode=control-group` sends the stop signal to the complete process tree
started by the unit, including the launcher, server, and workers.
`TimeoutStopSec=120` does not defer that signal and does not promise that an
in-flight tool call will finish. It is the upper bound systemd waits before
escalating to `SIGKILL` while remaining cgroup processes follow their own
shutdown paths.

The cgroup signal also stops a shared Claudexor daemon spawned inside this unit,
even in a separate process session; other clients using that daemon lose their
runs. When the launcher runs in browser mode (no usable display or GUI backend),
a service stop is a Quit: its accepted stop signal stops this installation's own
daemon wherever it runs (ARCHITECTURE §9); a daemon of another home is never
stopped. A windowed launcher (the startup probe found a usable display and GUI
backend; a set but unusable `DISPLAY` falls back to browser mode) has no such
handler, so the signal ends it without a Quit, and a daemon running
outside the unit keeps running, as it does when the signal precedes the handler.
Ouroboros's own Restart and Update keep the daemon unless the code they start
pins a different engine that is not older, which then replaces it.

## Why this is a user unit

Ouroboros state lives in the invoking user's `~/Ouroboros` directory and the
desktop launcher uses that user's session. A system unit would run under a
different identity or require a second state/permission contract, so the native
packages ship only this opt-in user service.
