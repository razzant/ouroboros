# Ship Ouroboros record metadata with a log collector

Verified on 2026-10-06 with Vector 0.58.0 (macOS arm64) against the Ouroboros source of the pull request
that added this example: `vector validate` and `vector test` on this file; a running Vector over a scratch
data root that Ouroboros's own writer filled and its own rotator rotated (at a lower size), stopped and
restarted midway; and a running Vector over the data root of an isolated Ouroboros server. The results are
in that pull request.

Ouroboros sends no telemetry. This example is for a deployment that wants its own monitoring: a collector
next to Ouroboros (a sidecar container, a host agent) reads the local JSONL records and forwards only
their metadata to the deployment's own log system. Nothing in Ouroboros changes, and nothing is sent
until you point the sink at a destination.

## What it reads

The supported records, their fields and their rules are declared in
[`ouroboros/contracts/record_contract.py`](../../../ouroboros/contracts/record_contract.py) (the record
passport). The collector follows it:

- `logs/events.jsonl`, `logs/tools.jsonl` and `logs/supervisor.jsonl` under the data root. Ouroboros
  renames each one into `archive/` at about 800 KB; the `device_and_inode` fingerprint lets Vector
  finish the renamed file and pick up the new one.
- `archive/{events,tools,supervisor}_*.jsonl`, so that generations rotated while Vector was stopped are
  still read; a generation Vector already read keeps its inode and checkpoint. `ignore_older_secs`
  (one day) keeps the tail off older history: to ship that, run a one-time backfill over the archive.
- `state/headless_tasks/*/data/logs/events.jsonl`: the model rounds, errors and task starts of subagents
  and forked-memory tasks are written there, not to the main `events.jsonl`. These drives are deleted
  once the task is finished and older than `OUROBOROS_GC_RETENTION_DAYS` (default 7), or at once for a
  cancelled subagent, so the collector must be running to keep them. Their `tools.jsonl` is skipped on
  purpose: the main `tools.jsonl` holds the same rows.

## What it forwards

Only the passport's anchor rows (the `anchors` list), and of them only metadata: identifiers, types,
statuses, durations, token counts. The `allowed` list drops everything else, including the
content-bearing fields the passport names (`args`, `result_preview`, `task`, `outcome_axes`,
`artifact_bundle`, `error`, `traceback`, ...). Extend it only with fields the passport lists for the row
types you need; a field it does not list may change without notice. A `task_received` row keeps its type,
time and `task_id`, taken from `task.id`; the rest of `task` is content. Each row also gets `log` (events,
tools or supervisor) and `child_drive` (true for a row read from a subagent's or forked-memory task's own
drive). If you add the drives' `tools.jsonl`, drop duplicate tools rows by `(invocation_id, type)`.
The example is a starting point, not a delivery guarantee: a collector that is stopped past retention or
past `ignore_older_secs` misses rows.

Money fields are left out on purpose. Spend comes only from Ouroboros's accounting views
(`GET /api/cost-breakdown`, `GET /api/state` `accounting`); `llm_usage.cost` is often null and is a
projection, so a sum of shipped rows is wrong.

## Run it

The paths assume the Compose deployment in `docs/DEPLOYMENT.md`, which mounts Ouroboros's data volume at
`/data`: give the collector the same volume read-only at `/data` and a writable `/var/lib/vector` for its
read positions, or replace both paths for a host install. Vector 0.58 does not expand `${VARIABLES}` in a
configuration file unless started with `--dangerously-allow-env-var-interpolation`, so the paths are
written out.

```bash
vector validate --no-environment vector.yaml && vector test vector.yaml && vector --config vector.yaml
```

Replace the `file` sink with your destination. Vector keeps its read positions in `/var/lib/vector`,
so a restart resumes where it stopped.

## Collector defaults that lose Ouroboros rows

- Vector keeps one file open for every file the globs match, old archive generations that
  `ignore_older_secs` skips included, and Ouroboros never deletes the archive. Give the collector an
  open-file limit well above the number of archive files (`LimitNOFILE` in a systemd unit,
  `--ulimit nofile=` for a container), or narrow the archive globs once the backlog is shipped; at the
  limit Vector cannot open the new generations.
- Vector discards lines longer than `max_line_bytes` (102,400 by default). Ouroboros rows reach about
  300 KB, so the example raises it.
- Fluent Bit's `tail` input stops monitoring a file at a line longer than `buffer_max_size` (32 KB by
  default) unless `skip_long_lines` is on. Raise the buffer.
- The OpenTelemetry Collector's `filelog` receiver splits rows longer than `max_log_size` (1 MiB), keeps
  the raw line in `body` after `retain` (add a `remove` of `body`), and keeps read positions across
  restarts only with the `file_storage` extension.
