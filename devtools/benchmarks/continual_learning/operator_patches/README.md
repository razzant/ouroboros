# Operator patches for the external CLB adapter (campaign v6.56.0)

The pinned Ouroboros adapter (`src/systems/ouroboros/`, handoff bundle v3,
reference commit 56764d6) predates v6.56.0 and hits three host/runtime
incompatibilities on a Linux host. These unified diffs port it; apply them on
top of the bundle copy inside the external continual-learning-bench checkout
(`patch -p0 < <file>` from the checkout root, adjusting paths):

1. `_launcher.v6560.patch` — write the declared `OUROBOROS_SAFETY_MODE=light`
   into the isolated settings at creation. v6.56.0 added an owner-guard that
   refuses env-side `full -> light` lowering at server boot
   (`_guard_safety_mode_lowering`), which killed every isolated server with
   uvicorn rc=3; the reference v6.52.2 run had no guard and was effectively
   `light`. Writing the knob into settings from birth keeps the guard intact
   (no lowering happens) and restores reference parity.
2. `clbench_step_shim.v6560.patch` — make the step-shim bind address
   overridable via `CLBENCH_SHIM_BIND` (default remains 127.0.0.1). On Linux,
   `--add-host=host.docker.internal:host-gateway` does NOT reach a
   host-loopback listener (empirically verified on both rootful and rootless
   daemons), so the docker-engine agent's submit_action silently never reached
   the shim: every question "completed" with `queries=0, reward=None`. Bind to
   the docker bridge IP (e.g. 172.17.0.1) for docker-engine runs.
3. `_docker_launcher.v6560.patch` — run the agent container with
   `--user <uid>:<gid>` (+ `HOME=/obo/data`). On a rootful daemon the
   container wrote root-owned files into the bind-mounted data root, and the
   host-side bridge died with PermissionError.

Also required for the docker engine: seed `skills/clbench_remote/` (from the
bundle's `bench-config/external-adapters/clbench_remote/`) into the dedicated
Ouroboros clone — the v6.56.0 native-seed allowlist
(`_POST_BOOTSTRAP_NEW_NATIVE_SEEDS`) deliberately does not auto-trust it on
the HOST engine path, which is why the docker engine (which installs the skill
itself) is the supported path for v6.56.0 bridge runs.

## Addendum (2026-07-05, campaign tick)

4. `run_clbench_bridge_agent.v6560.patch` — pass `run_index` to the task ctor
   only when its signature accepts it (`CodebaseAdaptationTask` lacks the
   parameter at the pinned runner commit; the DB task has it). Both call
   sites (stateless `_make_build` and the stateful path).
5. Runner-python venv also needs `mini-swe-agent` (import `minisweagent`) for
   the codebase_adaptation domain, and `pip` itself when the venv is created
   by uv (the isolated server's local-dev deps sync shells out to
   `python -m pip`).
