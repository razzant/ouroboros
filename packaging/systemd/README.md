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

`KillMode=control-group` sends the stop signal to the complete process tree
started by the unit, including the launcher, server, and workers.
`TimeoutStopSec=120` does not defer that signal and does not promise that an
in-flight tool call will finish. It is the upper bound systemd waits before
escalating to `SIGKILL` while remaining cgroup processes follow their own
shutdown paths.

This also stops a shared Claudexor daemon if it was spawned inside this unit's
cgroup, even if it started a separate process session; other clients using that
daemon lose it too. A daemon already running outside the unit is unaffected.
Use the ordinary desktop launch path when the daemon must survive an Ouroboros
service stop.

## Why this is a user unit

Ouroboros state lives in the invoking user's `~/Ouroboros` directory and the
desktop launcher uses that user's session. A system unit would run under a
different identity or require a second state/permission contract, so the native
packages ship only this opt-in user service.
