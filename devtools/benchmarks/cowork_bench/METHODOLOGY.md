# Cowork Bench methodology

## Protocol and comparison

The dataset, runner and evaluator are pinned to
[`0717376/cowork_bench@d943e75bc0fc8e3b27141979300cd8cbcd1e890d`](https://github.com/0717376/cowork_bench/tree/d943e75bc0fc8e3b27141979300cd8cbcd1e890d).
A full run measures pass@1 over **all 496 tasks**, with denominator 496 even when
some tasks fail or cannot start. A smoke subset is qualification evidence only;
report its selected IDs and count separately. This document defines the setup
and does not claim a completed Ouroboros score.

The pinned upstream README reports Kimi K3 at **363/496 (73.2%)**, using its
`parallel` runner. This campaign does not run a paired reference agent. The
published row does not identify the exact serving provider or quantization, so
a difference against it cannot establish a causal improvement from Ouroboros's
agent loop alone.

Each task uses the official `run_parallel.sh` lifecycle: fresh PostgreSQL state,
an agent container and a separate evaluator phase. The runner script and task
evaluators are unchanged. A run-scoped `BASH_ENV` script changes only Docker
executable discovery so the official script uses the resource wrapper despite
resetting `PATH`. The run manifest records this environment and applied limits.
The task prompt is the benchmark's system prompt plus task text with workspace
substitution; the adapter adds no task-specific answer hints.

## Agent configuration

The accepted campaign configuration is:

| Setting | Value |
|---|---|
| Model | `moonshotai/kimi-k3` in all model and review slots |
| Provider routing | OpenRouter default routing; no provider pin |
| Reasoning | High effort |
| Agent loop | Single agent, no scheduled subagents or external coding delegation |
| Acceptance | Required, blocking, three same-model reviewer slots; at most two paid review panels, each followed by author reaction within ordinary task limits; Blocking still requires fresh approval to accept corrected work |
| Round bound | 100 Ouroboros rounds; not a promise of identical tool-call counts to other engines |
| Workspace and memory | External task workspace, empty task memory |
| Runtime / safety | `pro`; LLM safety pass `off` in the disposable benchmark environment |
| Native web | Disabled, including native browser/search tools; benchmark-required MCP tools remain available |
| Post-task evolution | Disabled |
| Qualification timeout | 3600 seconds per agent phase; final full-run timeout chosen after smoke |

Acceptance semantics in the table describe this source revision. The Kimi K3
and Qwen3.8-27B campaigns pinned to Ouroboros seed
`484d241cfdb6d3f731f2307672625927746075aa` retain that seed's earlier rule:
two review cycles allow one rework. Interpret each run using its recorded
`run_manifest.json` source SHA and immutable image, not a later revision of
this document. Updating the document does not change existing runs or results.

The pinned upstream runner keeps its agent container alive with `sleep 7200`.
That inherited two-hour lifetime can end a long task regardless of a larger
`--task-timeout`; retain it in the deferred full-run timeout decision. Qualification
uses 3600 seconds.

The acceptance panel is part of the measured agent, not the official scorer.
Safety-off is a benchmark-specific departure from the usual light-mode template;
it avoids adding a separate safety-model request to the mock office operations.
These settings do not alter the user's live installation. The committed model
roster is serialized for provenance; disabled subagent scheduling means that its
presence is not evidence that subagents ran.

The benchmark's MCP tools are exposed with Ouroboros server prefixes. Native
shell/file/context tools perform the roles of the reference engine's local
Python and context helpers; the adapter does not emulate those four helpers as
identically named tools. Document this capability difference when comparing
engines.

## Container and dependency disclosures

Ouroboros starts its ordinary server inside the task container from a clean
committed seed. The launcher resolves and records an immutable Docker image ID,
uses that ID for execution, and requires the same ID for compatible recovery;
matching labels on a rebuilt mutable tag alone do not prove an identical image.
Its dependency environment is separate from the benchmark's.
`mcp-proxy==0.12.0` with `mcp==1.30.0` holds the task's stdio MCP sessions alive
behind local HTTP endpoints. This preserves presentation and browser state
across Ouroboros calls without changing its core MCP client.

The derived image also repairs reproducibility failures in the pinned upstream
build. Vendored servers which resolve incompatible MCP 2.x are repinned to
`mcp==1.26.0`; `psycopg2-binary==2.9.10` is installed in the two finance-server
environments whose local PostgreSQL shims require it. Chromium is installed at
the revision selected by the vendored Playwright dependency and its presence is
checked during build. When running as root inside the disposable container,
Playwright receives `--no-sandbox`; the outer Docker limits remain in force.
These dependency and launch changes must accompany any result report.

Every container created through the official runner, including database,
evaluator and helper containers, receives a limit of **4 CPUs, 16 GiB memory,
no swap and 512 PIDs**. These are per-container limits, not an aggregate run
quota. Exact run labels scope cleanup. Image building is a separate preparation
step and is not covered by these CPU/memory/PID limits. The launcher monitors
the same heavy-storage free-space reserve during its owned build and task run
(default 200 GiB); task admission checks it before new containers are created.
The supervisor also checks the root filesystem (default 40 GiB). Another user's
writes can still consume shared storage between checks.

## Spending and run custody

A paid invocation requires a shared campaign file. Its spending is the selected
OpenRouter key's cumulative usage minus one durable baseline, plus any recorded
prior spending. The same file spans qualification, smoke, the full run and
infrastructure retries. Concurrent ownership is locked. A changed key, changed
ceiling, decreased usage counter or unsettled prior run requires reconciliation.
Unrelated spending on the same key counts conservatively toward the campaign.

The qualification example uses a total ceiling of **$1000** and an invocation's
default spending bound of **$150**. The supervisor preserves
the explicit nonnegative `--budget-reserve-usd` allowance (default $100) for
unsettled provider charges. It is independent of concurrency and of the per-task
lifetime bound (default $25), which remains unchanged. Select the allowance for
billing delay and work actually in flight; multiplying full task budgets would
prematurely stop a 32-task, $2000 campaign at $1200 spent. With a $100 reserve,
the campaign stop boundary is $1900 spent instead. The supervisor stops when the
campaign remainder reaches that allowance, when the invocation bound is reached,
when the disk reserve becomes unavailable, or when the meter stays blind for the
recorded `--meter-blindness-sec` bound (default 30 seconds).

Blindness counts from the request time of the last reading that was accepted
and saved to the campaign file. Read time, retry pauses and ledger/monitor writes
all count; a rejected, non-finite, lower or late value never restarts the bound.
A healthy meter is read every 15 seconds, or sooner when one read would
otherwise not fit. Each read gets at most 5 seconds and requests cache
revalidation; a failed or rejected read is retried after about 3 seconds within
the same bound, so one slow read cannot consume it. Only a finite, nonnegative,
nondecreasing value may update the campaign. Rejected values and errors remain in
the monitor and unit log, including after shutdown. The bound does not rebase
spending, change the ceiling, reserve or invocation bound, or ignore known
budget exhaustion. At the bound, the runnable supervisor fences admission and
stops the runner group and exact-label resources. Synchronous filesystem calls
cannot be preempted: an overlong save or diagnostic write is checked immediately
on return, never credited as a fresh window. Cleanup itself can take time.
A final reading, bounded by the same
interval, happens only after cleanup and settles accounting; it cannot admit work.

If the campaign file cannot be saved, the run stops with
`campaign_persistence_failed`; if settlement still cannot be saved, the campaign
keeps its unsettled `active_run`. A failed ledger or monitor write is reported on
stderr and in later records, without stopping work while the campaign record
and meter remain valid. An unexpected supervisor failure is recorded as
`supervisor_error`, never as a finished runner or exhausted budget.
These are configurable operator bounds. Billing can be delayed and paid calls may already be in flight;
the monitor is **not a provider-enforced hard dollar cap**.

Start qualification at concurrency 1 and choose full-run concurrency after
measuring resource use. If smoke projects the full dataset above the
remaining campaign budget, pause for an owner decision rather than changing the
model, effort, configuration or budget. Reconcile delayed charges before another
paid phase. Preserve every run in a new directory outside the source and live
runtime data, including aborted runs. No score is inferred from launcher exit 0.

## Outcomes and evidence

The adapter waits for pending/finalizing task artifacts within the existing outer
agent deadline. Explicitly partial cost on a completed result receives the CLI's
bounded finality wait (up to 60 seconds within that deadline). The summary retains
`accounted_upper_bound_usd` and the canonical cost openness/finality fields;
unknown or unfinished accounting is not presented as a final paid receipt.

A voluntarily completed Ouroboros task maps to the reference engine's `success`;
only the official evaluator decides pass or fail. Runtime round, budget and
deadline termination remain disclosed truncations. Provider/transport failures
and adapter setup failures are infrastructure outcomes. A wall-clock timeout
after model work is a genuine failed attempt, not a new attempt entitlement.
The result ledger retains every selected ID, including `not_attempted` entries.
A timeout before task submission is an infrastructure failure. After submission,
missing token telemetry does not prove that no paid/model work happened.
`not_attempted` means neither a runner row nor adapter start evidence exists; the
benchmark's pre-created dump and empty workspace are not evidence. The evaluator's
`traj_log.json` or `eval_res.json` alone does not prove an agent start, though the
receipt remains visible independently. A started task
without an adapter summary is `infra_failed` with reason `interrupted:<cause>`:
the run's stop reason, or `runner_exited` when the runner ended without a supervisor
stop. A live snapshot retains the legacy `infra_failed`/`missing_adapter_summary`
diagnostic bucket with `provisional: true`; it asserts no interruption or settled
outcome and must not be scored as a finished result. Its
`paid_activity` is `observed` only when a copied checkpoint holds a token-bearing
`llm_usage` record and is otherwise `unknown`; no task cost is inferred. A runner
row without a summary receives the same interruption cause after the run ends.
Infrastructure recovery uses new roots and the identical configuration, seed and
immutable image. With no explicit new selection, it preserves the original task
selection, and always retains cumulative ancestry. At most two recovery passes
with remaining work are permitted; settled successes and genuine failures
from every ancestor are skipped, never repeated for best-of selection. Any final
scoring overlay must retain provenance to the original attempts.

Every ledger row, whatever its execution outcome, carries the official
evaluator's receipt: the exact `eval_res.json` path, byte count and SHA-256,
the literal verdict and a bounded cause. The launcher never writes or re-runs
that file, never copies it into the receipt, and never infers a verdict from
runner output. As before, only a scored row also keeps the parsed evaluator
result in its details. `official_eval_status` is `completed` only for a literal JSON boolean.
`declined` means `pass: null` that exactly matches the evaluator's status gate
text, with its linked `traj_log.json` recording a non-success status. Any other
null is `unknown`. A missing or non-boolean `pass` is `invalid`. An I/O,
UTF-8, JSON or non-object failure is `unreadable`. No file is `unreported`, not
proof that the evaluator never ran. Only a successful agent phase with a
literal boolean is scored. `false` with evaluator output remains a genuine
verdict. Any other result there stays an infrastructure row, never
`bool(value)`. A verdict on an agent- or infrastructure-failed row is disclosed
by the ledger and audit but never promoted into the score. The shared ledger
default is `unreported`; `not_run` appears only when an adapter asserts it.
The audit's `official_pass` reports the literal receipt verdict independently
of the unchanged execution/scoring classification. Runtime stop disclosure is
read only from the summary-named exported task result whose embedded identity
matches; missing or mismatched sources remain explicit gaps, never a glob-selected
neighbour's outcome.

Ledgers written before this revision recorded `not_run` for every row without
a completed verdict, including rows where the evaluator ran and declined at its
status gate. The planning report dated 2026-09-24 records a prior SHA-256 match
between reconstructed gate receipts and the `audit_eval_sha256` values in
`combined-index-20260922T102550Z.jsonl` for all 22 affected `agent_failed` rows:
14 `deadline_local` and 8 `wall_clock_timeout`. This implementation did not
re-read those historical receipts or establish their current availability;
the prior hash comparison is not a fresh file-access check. The planning report
also records that the tasks' PostgreSQL state was destroyed, so historical
re-evaluation is unavailable. No historical index or score is rewritten and no
checker is rerun. For new receipts, the gate link is a fixed path plus exact gate
text; the evaluator records no hash of the log it read.

Each eval phase publishes one exclusive claim with a random attempt ID in the
task dump, outside the agent's `workspace/`, before the official
`evaluate_from_log_file` call. The exact `traj_log.json`, run config, upstream
evaluator and entrypoint bytes are bindings of that attempt, not retry keys; the
mutable workspace is never hashed. Its terminal receipt is written once,
atomically, and keeps the returned value apart from the `eval_res.json` state
(absent, unreadable, or present with byte count and SHA-256). A reentry with the
same bindings replays the recorded official lines and exit code without calling
the evaluator, with or without the diagnostic below. An unfinished claim (in
flight, killed, or with an unpublished receipt), changed bindings or an
unreadable claim never cause another official effect. An `eval_res.json` present
before any claim is preserved byte for byte: the evaluator and the diagnostic do
not run, and the ledger records `not_run` (`official_eval_not_run`) instead of
scoring that file. A current run may also be refused before its eval entrypoint creates
any claim. The launcher reads `run_manifest.json`: only a readable `applied_config`
without the diagnostic flag identifies a pre-protocol legacy run whose unclaimed
receipt may retain its old scoring behavior. The flag is recorded for every new
run, even when false. A missing or unreadable manifest cannot authenticate an
unclaimed `pass:true` as this attempt's verdict; it stays disclosed but unscored.
The launcher checkpoints the applied config before runner execution, so live
ledger snapshots and later audits read the same protocol provenance.
Upstream `TaskConfig` construction deletes an existing file, so the claim check
precedes it. Claims and receipts share the dump directory that a
lingering agent process could write before its container is removed, as it could
write `traj_log.json`; the audit flags tool arguments naming them.

Phase-aware mounts omit the task's evaluator and ground-truth workspace from the
agent's task view. Ouroboros settings and provider credentials remain outside the
shared dump directory; the run-local credential file is mode 0600 and is cleared
on launcher completion. The existing isolated-benchmark sentinel is created
before the server starts, suppressing runtime log rotation so the collector keeps
the full task-local event/tool history. Sanitize and inspect artifacts before
publication.
Task dumps are shared across the run, and native shell/Python can potentially
access PostgreSQL directly instead of using MCP, as can reference agents. Native
web is disabled because benchmark answers are public, but this is not proof of
complete network isolation or absence of contamination.

The offline audit reports token-bearing usage records, MCP activity, reported
capability omissions, known versus unknown cost, and argument references to
answer sources, evaluator artifacts or direct database clients. Findings contain
log coordinates for manual inspection, never copied answers, and do not change
scores. No findings are not proof of a clean trace. Missing logs and prices remain
unknown. `llm_usage` amounts are compatibility accounting evidence; the campaign
meter is the spending check. A billing provider such as `openrouter` does not
identify the upstream endpoint. `response_provider` observations are reported
only when present, with incomplete coverage disclosed; successful-call endpoint
evidence may be unavailable in the copied logs. Selected scrubbed logs are
checkpointed before each task-status request and at finalization; a slow request
can delay the next checkpoint. An interrupted run may therefore retain partial
evidence, while raw request/response blobs remain unexported.

A result report therefore needs the exact seed and image/benchmark pins, selected
IDs, applied settings, official evaluator outputs, complete denominator,
infrastructure and truncation disclosures, audit disposition, measured cost and
duration, and these protocol differences. Inventory or build success alone does
not demonstrate an end-to-end benchmark result.

## Residual-state diagnostic (opt-in, audit only)

`--diagnostic-eval-on-truncation` sets `diagnostic_eval_on_truncation` in the
applied config; the default is `false`. It is part of the recovery
configuration: runs recorded before the flag existed resume only with it off, and
any other difference is refused. It changes no official verdict, exit code,
`eval_res.json`, `summary.csv` row, ledger status or score.

After the official lines are final, the same eval phase may rerun the task's
unchanged checker once. Eligibility needs all of: the same attempt's official call
returned the upstream status-gate decline and `eval_res.json` still holds exactly
that payload; the adapter summary for that log names `deadline_local`,
`wall_clock_timeout`, `round_limit` or `budget_exhausted` (never owner stops,
unabsorbed children, finalization grace or infrastructure codes) with a submitted
task; observed model activity (`false` and unknown activity are reported
separately); and a daemon listing, made by the host shim before the eval exec,
without the exact agent container. A separate Python process whose stdout and
stderr already point to its own log rebuilds the pinned evaluator's command from
the saved task config (`TaskConfig.from_dict`, the `Evaluation.build` fallback,
the first two `launch_time` tokens, the same shell, working directory and
environment). Without a command the outcome is `unavailable`, never a pass. It
never calls `evaluate_from_log_file`, never edits the status or log, and never
writes into the shared runner log that the summary greps for `Status:` and
`Pass:` lines.

The checker sees the state left after the agent stopped and after the official
evaluator ran, possibly minutes past the limit and after that run's own side
effects: a residual-state observation, not an exact-deadline verdict. Coverage is
incomplete by construction. The run is capped at 900 seconds, a ceiling rather
than a promised window inside the evaluator container's `sleep 1800`; at the cap
the owned process group is terminated and its death verified, otherwise the
outcome is `unknown`. A process that leaves that group is covered only by
container removal. Enabling the diagnostic lengthens eligible eval phases and can
change which tasks a budget-stopped campaign reaches, so campaigns with and
without it are not promised identical results. Outcomes appear only in the
audit's `eval_attempts` block, with the declined, eligible and checked
denominators; they are never written to `result_index.jsonl`, `summary.csv`, a
combined index or a score.

A campaign stop is unchanged and has no drain: new containers are refused and the
run's containers are removed. The shim keeps host-only evidence in the run root's
`eval_admission/`, which no container mounts: the task each runner container
served and every refused eval-container creation. A task-specific refusal to
create an eval container is `unavailable` in the audit. It names
`campaign_stopped_before_eval` only when final cleanup, the first durable stop
cause and the monitor's campaign money/meter stop agree with the refusal. Disk failures
and generic stop-file refusals retain their own `eval_creation_refused` cause.
A claim without a terminal receipt stays `unknown` even after final
`monitor.json` proves exact-label cleanup: the claim precedes evaluator
preparation. An evaluator exception captured in a terminal receipt, followed by
proven cleanup, may be called `interrupted`; a successful linked returned verdict
is `evaluated`. A diagnostic claim likewise predates eligibility/spawn and stays
unknown without its terminal receipt. A stop file alone proves neither.
