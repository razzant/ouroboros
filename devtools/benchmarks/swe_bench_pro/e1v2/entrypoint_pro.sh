#!/usr/bin/env bash
set -uo pipefail
OBO_PY=/opt/miniconda3/envs/oboros/bin/python
export HOME=/
export OUROBOROS_APP_ROOT=/obo-data/app
export OUROBOROS_DATA_DIR=/obo-data
export OUROBOROS_REPO_DIR=/obo-repo
export OUROBOROS_SETTINGS_PATH=/obo-data/settings.json
export OUROBOROS_RETURN_REASONING=true
export PYTHONPATH=/obo-repo
export PYTHONDONTWRITEBYTECODE=1
export OUROBOROS_DATA_DIR=/obo-data
export NO_PROXY=127.0.0.1,localhost,::1 ; export no_proxy="$NO_PROXY"
mkdir -p /obo-data /out
IID="${OBO_INSTANCE_ID:-task}"
WORK="${OBO_WORKDIR:-/app}"

[ -e /obo-repo/.git ] || { echo "[pro] seed /obo-repo" >&2; cp -a /opt/ouroboros-ro/. /obo-repo/; }
git -C /obo-repo config user.name  "Ouroboros"          2>/dev/null || true
git -C /obo-repo config user.email "ouroboros@local.mac" 2>/dev/null || true
cp /opt/oboros-settings-ro.json /obo-data/settings.json

# --- Transport: install-in-image (musl/Alpine task images that have no prebuilt
# 'oboros-env-musl' conda volume). Use the task image's system python rather than
# upgrading it: upgrading python without the exact expat runtime broke pyexpat on
# real Alpine images. Browser tools are disabled for this benchmark, so remove
# Playwright from musl requirements instead of forcing unsupported musl wheels.
if [ "${OBO_INSTALL_IN_IMAGE:-0}" = "1" ]; then
  echo "[pro] install-in-image transport: installing Ouroboros into the task image" >&2
  if [ -z "${OBO_INSTALL_READY:-}" ]; then
    {
      if command -v apk >/dev/null 2>&1; then
        PY_BEFORE="$(python3 -V 2>&1 || true)"
        apk add --no-cache py3-pip expat git curl bash build-base libffi-dev openssl-dev rust cargo
      elif command -v apt-get >/dev/null 2>&1; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip git curl bash build-essential
      fi
      PYBIN="$(command -v python3 || command -v python)"
      PY_AFTER="$("$PYBIN" -V 2>&1 || true)"
      echo "install-in-image: python before apk: ${PY_BEFORE:-unknown}" >&2
      echo "install-in-image: python after apk:  ${PY_AFTER:-unknown}" >&2
      "$PYBIN" -c "import xml.parsers.expat, pyexpat" || { echo "SOLVE_INFRA_SUSPECT reason=pyexpat_abi_mismatch" >&2; exit 87; }
      "$PYBIN" -m pip --version || { echo "SOLVE_INFRA_SUSPECT reason=pip_bootstrap_failed" >&2; exit 87; }
      grep -ivE '^(playwright|playwright-stealth)([<=>[:space:]].*)?$' /opt/ouroboros-ro/requirements.txt > /tmp/reqs_musl.txt
      if ! "$PYBIN" -m pip install --break-system-packages -r /tmp/reqs_musl.txt; then
        echo "install-in-image: musl requirements failed; retrying without tree-sitter" >&2
        "$PYBIN" -m pip --version || { echo "SOLVE_INFRA_SUSPECT reason=pip_bootstrap_failed" >&2; exit 87; }
        grep -ivE 'tree[-_]sitter' /tmp/reqs_musl.txt > /tmp/reqs_musl_no_ts.txt
        "$PYBIN" -m pip install --break-system-packages -r /tmp/reqs_musl_no_ts.txt || { echo "SOLVE_INFRA_SUSPECT reason=pip_bootstrap_failed" >&2; exit 87; }
      fi
      OBO_INSTALL_READY=1
    } >/out/install.log 2>&1
  fi
  PYBIN="$(command -v python3 || command -v python)"
  if PYTHONPATH=/obo-repo "$PYBIN" -c "import server" 2>>/out/install.log; then
    OBO_PY="$PYBIN"
    echo "[pro] install-in-image OK: OBO_PY=$OBO_PY ($("$OBO_PY" --version 2>&1))" >&2
  else
    echo "SOLVE_INFRA_SUSPECT reason=server_import_failed" >&2
    echo "[pro] install-in-image FAILED (server import) — see /out/install.log" >&2
    exit 87
  fi
fi

touch /obo-data/.ouroboros_isolated_benchmark
# Seed owner_chat_id BEFORE the budget reset. reset_per_task_budget() does a
# load-modify-write that creates state.json with ONLY the zeroed budget keys on
# a fresh volume; if the seed ran after it, the "[ ! -f ]" guard would be false
# and owner_chat_id would never be set -> post_task_evolution drops every cycle
# and E1v2 silently degrades to E0. Seeding first (then letting the reset
# preserve all non-budget keys) keeps native evolution active on fresh runs.
if [ ! -f /obo-data/state/state.json ]; then
  mkdir -p /obo-data/state
  printf '{"owner_chat_id": 1}' > /obo-data/state/state.json
  echo "[pro] seeded owner_chat_id (fresh state.json)" >&2
fi
"$OBO_PY" - <<'PYEOF' 2>/dev/null || true
from supervisor.state import reset_per_task_budget
reset_per_task_budget("/obo-data", confirm_isolated=True)
PYEOF
echo "[pro] budget ledger reset requested through guarded isolated helper" >&2

# --- Option A: at task start, close a dangling committed evolution transaction
# left by the previous cycle. The native in-cycle restart (request_restart ->
# execvpe -> restart-verify) is unreliable: the agent inconsistently calls
# request_restart / makes extra commits, leaving the transaction active with
# commit_sha + restart_verified=False -> a poison-pill that wedges
# enqueue_evolution_task_if_needed for ALL subsequent tasks (E1v2 degrades to
# E1). But between tasks the container FULLY restarts on /obo-repo with the
# already-committed evolution code, and the previous task's health-gate verified
# its import -> the commit IS absorbed, just verified by the container boundary
# rather than an in-cycle execvpe. Mirror the verified path of
# record_evolution_cycle: restart_verified=True + absorbed_cycles_done++ + move
# to transaction_history + pop active_transaction -> gate cleared, counters
# intact. GUARD: if commit_sha is NOT reachable from /obo-repo HEAD (health-gate
# rolled the self-edit back -> commit lost), do NOT mark absorbed; ABANDON the
# transaction instead (still clears the poison-pill, without incrementing the
# counter). With a core that performs its own boot reconciliation this is a
# harmless no-op (no active_transaction left to heal).
"$OBO_PY" - >&2 2>/dev/null <<'PYEOF' || true
import json, subprocess, time
CAMP = "/obo-data/state/evolution_campaign.json"
try:
    c = json.load(open(CAMP))
except Exception:
    c = None
if isinstance(c, dict):
    at = c.get("active_transaction")
    if isinstance(at, dict):
        sha = str(at.get("commit_sha") or "").strip()
        if sha and not at.get("restart_verified"):
            rc = subprocess.run(["git", "-C", "/obo-repo", "merge-base", "--is-ancestor", sha, "HEAD"],
                                capture_output=True).returncode
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            th = c.get("transaction_history")
            if not isinstance(th, list):
                th = []
            if rc == 0:   # commit reachable from HEAD -> really present -> absorbed
                at["restart_verified"] = True
                at["restart_required"] = False
                at["restart_verified_at"] = now
                at["restart_verified_by"] = "harness_option_a_container_boundary"
                th.append(at); c["transaction_history"] = th[-50:]
                c["absorbed_cycles_done"] = int(c.get("absorbed_cycles_done") or 0) + 1
                c.pop("active_transaction", None)
                print(f"[pro] Option A: healed dangling evolution tx {sha[:8]} as restart-verified (container boundary)")
            else:         # commit lost (rollback) -> abandon, but clear the poison-pill
                at["abandoned"] = True
                at["abandoned_at"] = now
                at["abandoned_reason"] = "commit_not_reachable_after_health_gate_or_rollback"
                th.append(at); c["transaction_history"] = th[-50:]
                c.pop("active_transaction", None)
                print(f"[pro] Option A: abandoned dangling evolution tx {sha[:8]} (commit not reachable from HEAD — rolled back)")
            json.dump(c, open(CAMP, "w"))
PYEOF

git -C "$WORK" -c advice.detachedHead=false checkout -q "$OBO_BASE_COMMIT" 2>/dev/null || true
git -C "$WORK" reset -q --hard "$OBO_BASE_COMMIT" || { echo "[pro] FATAL: reset $WORK failed" >&2; exit 1; }

# --- Gold-history strip (SWE-bench Pro issue #93, OPEN/unpatched): the public
# jefzda images carry FUTURE git history, so `git show <fix>` / `git log --all` /
# tags can recover the gold solution. Strip it (warn-only) via the shared helper
# before the agent starts. OBO_STRIP_GOLD_HISTORY=0 disables it (debugging only).
if [ "${OBO_STRIP_GOLD_HISTORY:-1}" = "1" ]; then
  bash /opt/ouroboros-ro/devtools/benchmarks/swe_bench_pro/strip_gold_history.sh "$WORK" "$OBO_BASE_COMMIT" || true
fi
git -C "$WORK" ls-files --others --exclude-standard -z > /out/base_untracked.snapshot 2>/dev/null || true

REPO_HEAD0="$(git -C /obo-repo rev-parse HEAD 2>/dev/null)"

export OUROBOROS_SERVER_HOST=127.0.0.1
export OUROBOROS_SERVER_PORT=8765
"$OBO_PY" /obo-repo/server.py >>/out/server.log 2>&1 &
SRV=$!
ready_probe() {
  "$OBO_PY" - <<'PYEOF' 2>/dev/null
import urllib.request, json, sys
try:
    urllib.request.urlopen("http://127.0.0.1:8765/api/health", timeout=5).read()
    s = urllib.request.urlopen("http://127.0.0.1:8765/api/state", timeout=5).read().decode()
    sys.exit(0 if json.loads(s).get("supervisor_ready") else 1)
except Exception:
    sys.exit(1)
PYEOF
}
READY_MAX="${OBO_READY_MAX:-900}"; R=0; T0=$(date +%s)
while [ $(( $(date +%s) - T0 )) -lt "$READY_MAX" ]; do
  if ready_probe; then R=1; break; fi
  kill -0 "$SRV" 2>/dev/null || { echo "[pro] server died" >&2; tail -30 /out/server.log >&2; exit 1; }
  sleep 3
done
[ "$R" = 1 ] || { echo "[pro] not ready after ${READY_MAX}s" >&2; tail -30 /out/server.log >&2; kill "$SRV" 2>/dev/null; exit 1; }
echo "[pro] server ready in $(( $(date +%s) - T0 ))s" >&2

"$OBO_PY" -m ouroboros.cli --url http://127.0.0.1:8765 evolve stop >/dev/null 2>&1 || true

cp /opt/oboros-settings-ro.json /obo-data/settings.json   # Close the short window where the model could be overwritten in settings.
# OPTIONAL pre-task self-evolution phase (default OFF). Runs ONLY when the driver
# passed --pretask-evolution (OBO_PRETASK_EVOLUTION=1); the baseline path skips
# this entire block and behaves exactly as before.
if [ "${OBO_PRETASK_EVOLUTION:-0}" = "1" ]; then
  git config --global --add safe.directory /obo-repo 2>/dev/null || true
  git config --global --add safe.directory "$WORK" 2>/dev/null || true
  echo "[pro] PRETASK-EVOLUTION $IID start (task-specific; repo=/app; wait_max=${OBO_PRETASK_EVOLUTION_WAIT_MAX:-0}s; solve_steps=${OBO_SOLVE_STEP_BUDGET:-0})" >&2
  PRE_EVO_HEAD0="$(git -C /obo-repo rev-parse HEAD 2>/dev/null || echo '')"
  "$OBO_PY" -m ouroboros.cli --url http://127.0.0.1:8765 evolve start "$(cat /opt/pretask_evolution_prompt.txt)" \
    >/out/pretask_evolution_start.json 2>/out/pretask_evolution_start.stderr || true
  "$OBO_PY" - "${OBO_PRETASK_EVOLUTION_WAIT_MAX:-0}" "$PRE_EVO_HEAD0" >/out/pretask_evolution.json 2>/out/pretask_evolution_wait.stderr <<'PYEOF' || printf '{"absorbed":false,"reason":"error","cycles":0}' >/out/pretask_evolution.json
import json, os, subprocess, sys, time, urllib.request
MAX = int(sys.argv[1] or 0)
SHA0 = str(sys.argv[2] or "")
IDLE_GRACE = 240
URL = "http://127.0.0.1:8765/api/state"
CAMP = "/obo-data/state/evolution_campaign.json"
def camp():
    try: return json.load(open(CAMP))
    except Exception: return {}
def absorbed():
    try: return int(camp().get("absorbed_cycles_done") or 0)
    except Exception: return 0
def cycles_done():
    try: return int(camp().get("cycles_done") or 0)
    except Exception: return 0
def head():
    try: return subprocess.run(["git", "-C", "/obo-repo", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception: return ""
def state():
    try:
        with urllib.request.urlopen(URL, timeout=5) as r: return json.loads(r.read().decode())
    except Exception: return {}
def pending_restart():
    at = camp().get("active_transaction") or {}
    return bool(str(at.get("commit_sha") or "").strip() and not at.get("restart_verified"))
def is_idle(st):
    return bool(st and st.get("supervisor_ready")
                and int(st.get("pending_count") or 0) == 0
                and int(st.get("running_count") or 0) == 0)
EVO0 = absorbed()
CYC0 = cycles_done()
t0 = time.time()
reason = "timeout"
while MAX <= 0 or time.time() - t0 < MAX:
    c = absorbed()
    sha = head()
    if c > EVO0 and sha and sha != SHA0:
        d2 = time.time() + 240
        while time.time() < d2:
            st = state()
            if st.get("supervisor_ready") and int(st.get("workers_total") or 0) > 0:
                break
            time.sleep(2)
        reason = "absorbed"
        break
    # A no-commit/degraded evolution cycle may complete without increasing the
    # absorbed counter and the supervisor can immediately enqueue the same
    # objective again. For pre-task evolution, one completed cycle is enough:
    # stop and move to solve instead of spending the whole benchmark repeating.
    if cycles_done() > CYC0:
        reason = "degraded" if pending_restart() else ("cycle_completed" if sha and sha != SHA0 else "no_commit_cycle")
        break
    st = state()
    if time.time() - t0 > IDLE_GRACE and is_idle(st):
        if pending_restart():
            time.sleep(8)
            if is_idle(state()) and pending_restart() and absorbed() == EVO0:
                reason = "degraded"
                break
        else:
            reason = "no_promotion"
            break
    time.sleep(5)
at = camp().get("active_transaction") or {}
print(json.dumps({
    "absorbed": reason == "absorbed",
    "reason": reason,
    "cycles_before": EVO0,
    "cycles_after": absorbed(),
    "campaign_cycles_before": CYC0,
    "campaign_cycles_after": cycles_done(),
    "degraded": pending_restart(),
    "active_tx_commit": str(at.get("commit_sha") or "")[:8],
    "sha_before": SHA0,
    "sha_after": head(),
    "elapsed_sec": round(time.time() - t0, 3),
    "wait_max_sec": MAX,
}))
PYEOF
  "$OBO_PY" -m ouroboros.cli --url http://127.0.0.1:8765 evolve stop >/out/pretask_evolution_stop.json 2>/out/pretask_evolution_stop.stderr || true
  "$OBO_PY" - <<'PYEOF' >&2 2>/dev/null || true
import json
try: d = json.load(open('/out/pretask_evolution.json'))
except Exception: d = {}
print(f"[pro] pretask-evolution: absorbed={d.get('absorbed',False)} cycles={d.get('cycles_after',0)} reason={d.get('reason','?')} degraded={d.get('degraded',False)} elapsed={d.get('elapsed_sec','?')}")
PYEOF
  if ! PYTHONPATH=/obo-repo "$OBO_PY" -c "import ouroboros.cli, ouroboros.agent, ouroboros.loop, ouroboros.config, ouroboros.subagent_worktrees, ouroboros.tools.subagent_integration, ouroboros.workspace_executor, ouroboros.contracts.task_constraint, ouroboros.retention, server, supervisor.queue" 2>/out/pretask_health.err; then
    echo "[pro] PRETASK HEALTH-GATE FAILED - rollback self-edit to $PRE_EVO_HEAD0" >&2
    sed 's/^/[pro]   pretask health: /' /out/pretask_health.err >&2 2>/dev/null | head -8
    git -C /obo-repo diff "$PRE_EVO_HEAD0" > /out/pretask_rejected_self_edit.diff 2>/dev/null || true
    git -C /obo-repo reset -q --hard "$PRE_EVO_HEAD0" 2>/dev/null || true
    git -C /obo-repo clean -qfd 2>/dev/null || true
    kill "$SRV" 2>/dev/null || true
    "$OBO_PY" /obo-repo/server.py >>/out/server.log 2>&1 &
    SRV=$!
    T0=$(date +%s); R=0
    while [ $(( $(date +%s) - T0 )) -lt "$READY_MAX" ]; do
      if ready_probe; then R=1; break; fi
      kill -0 "$SRV" 2>/dev/null || { echo "[pro] server died after pretask rollback" >&2; tail -30 /out/server.log >&2; exit 1; }
      sleep 3
    done
    [ "$R" = 1 ] || { echo "[pro] not ready after pretask rollback" >&2; tail -30 /out/server.log >&2; exit 1; }
  else
    echo "[pro] pretask health-gate OK" >&2
  fi
fi
if [ "${OBO_SELFIMPROVE:-0}" = "1" ]; then
  echo "[pro] ROOT-RUN $IID (self_modification; /app as active external workspace; post-task evolution=native)" >&2
else
  echo "[pro] ROOT-RUN $IID (self_modification; /app as active external workspace; post-task evolution=disabled baseline)" >&2
fi
# Tool denylist + per-task memory mode are passthrough knobs (run_pro --disable-tools / --memory-mode).
# Defaults preserve the original tool behavior (full web/browser/vision + claude_code_edit disabled).
OBO_DISABLE_TOOLS="${OBO_DISABLE_TOOLS:-web_search,browse_page,browser_action,analyze_screenshot,vlm_query,view_image,claude_code_edit}"
# Benchmark default is a FRESH child memory drive (v6.56.0): the measured artifact
# is the harness on this task, not memory accreted across tasks. Explicitly export
# OBO_MEMORY_MODE=shared/forked to opt back into carried memory.
OBO_MEMORY_MODE="${OBO_MEMORY_MODE:-empty}"
# The solve runs with /app as the ACTIVE EXTERNAL WORKSPACE (v6.56.0): contextual
# repo tools resolve against the task repo and the runtime captures a workspace
# patch artifact, instead of the legacy rootless "dig /app via user_files" mode.
# An explicitly exported EMPTY OBO_SOLVE_WORKSPACE_ROOT restores the legacy mode.
OBO_SOLVE_WORKSPACE_ROOT="${OBO_SOLVE_WORKSPACE_ROOT-/app}"
SOLVE_ARGS=(
  --jsonl --result-json-out /out/solve_result.json --timeout "${OBO_SOLVE_TIMEOUT:-3000}"
  --disable-tools "$OBO_DISABLE_TOOLS"
  --memory-mode "$OBO_MEMORY_MODE"
  --task-metadata-json '{"budget_profile": {"improvement_policy": "until_deadline", "cost_hard_stop_pct": 0}}'
)
[ -n "$OBO_SOLVE_WORKSPACE_ROOT" ] && SOLVE_ARGS+=(--workspace "$OBO_SOLVE_WORKSPACE_ROOT")
echo "[pro] solve tools-disabled=[$OBO_DISABLE_TOOLS] memory=[$OBO_MEMORY_MODE] workspace=[${OBO_SOLVE_WORKSPACE_ROOT:-none}]" >&2
"$OBO_PY" -m ouroboros.cli --url http://127.0.0.1:8765 run \
  "${SOLVE_ARGS[@]}" \
  "$(cat /opt/problem_statement.txt)" >/out/solve_events.jsonl 2>/out/solve.stderr || true
bash /opt/ouroboros-ro/devtools/benchmarks/swe_bench_pro/capture_patch.sh "$WORK" "$OBO_BASE_COMMIT" /out/patch.diff /out/base_untracked.snapshot 2>/out/capture_patch.stderr || true
cp /out/patch.status.txt /out/app_status.txt 2>/dev/null || true     # ARCHIVE: what the agent left in /app
git -C "$WORK" reset -q 2>/dev/null || true                                    # restore the index (leave the working tree untouched)
git -C "$WORK" diff --binary "$OBO_BASE_COMMIT" >/out/patch_tracked_only.diff 2>/dev/null || true
[ "${OBO_ARCHIVE_APP:-0}" = "1" ] && tar czf /out/app_state.tgz -C "$WORK" --exclude=.git --exclude=node_modules . 2>/dev/null || true
ROOT_TID="$("$OBO_PY" -c "import json;print(json.load(open('/out/solve_result.json')).get('task_id',''))" 2>/dev/null || echo '')"
SOLVE_EVENTS="$(wc -l < /out/solve_events.jsonl 2>/dev/null || echo 0)"
echo "[pro] ROOT-RUN patch=$(wc -c < /out/patch.diff)B events=$SOLVE_EVENTS task_id=$ROOT_TID" >&2
[ "$SOLVE_EVENTS" -lt 2 ] && echo "[pro] WARNING: SOLVE_INFRA_SUSPECT (too few events - possible server/network failure?)" >&2 || true

ABSORB_MAX="${OBO_ABSORB_MAX:-1800}"
if [ "${OBO_SELFIMPROVE:-0}" = "1" ]; then
  echo "[pro] wait-for-absorb: max=${ABSORB_MAX}s (native post-task evolution)" >&2
  "$OBO_PY" - "$ABSORB_MAX" >/out/absorb.json 2>/dev/null <<'PYEOF' || printf '{"absorbed":false,"reason":"error","cycles":0}' >/out/absorb.json
import json, os, subprocess, sys, time, urllib.request
MAX = int(sys.argv[1]); IDLE_GRACE = 180; URL = "http://127.0.0.1:8765/api/state"
CAMP = "/obo-data/state/evolution_campaign.json"
REQ  = "/obo-data/state/post_task_evolution_request.json"
def camp():
    try: return json.load(open(CAMP))
    except Exception: return {}
def absorbed():
    try: return int(camp().get("absorbed_cycles_done") or 0)
    except Exception: return 0
def head():
    try: return subprocess.run(["git","-C","/obo-repo","rev-parse","HEAD"],capture_output=True,text=True,timeout=15).stdout.strip()
    except Exception: return ""
def state():
    try:
        with urllib.request.urlopen(URL, timeout=5) as r: return json.loads(r.read().decode())
    except Exception: return {}
def pending_restart():
    at = camp().get("active_transaction") or {}
    return bool(str(at.get("commit_sha") or "").strip() and not at.get("restart_verified"))
def is_idle(st):
    return bool(st and st.get("supervisor_ready")
                and int(st.get("pending_count") or 0) == 0
                and int(st.get("running_count") or 0) == 0)
EVO0 = absorbed(); SHA0 = head(); t0 = time.time(); reason = "timeout"
while time.time() - t0 < MAX:
    c = absorbed(); sha = head()
    if c > EVO0 and sha and sha != SHA0:
        d2 = time.time() + 180   # Wait until the server is alive again after execvpe.
        while time.time() < d2:
            st = state()
            if st.get("supervisor_ready") and int(st.get("workers_total") or 0) > 0: break
            time.sleep(2)
        reason = "absorbed"; break
    if time.time() - t0 > IDLE_GRACE and c == EVO0 and is_idle(state()) and not os.path.exists(REQ):
        if pending_restart():
            time.sleep(6)
            if is_idle(state()) and pending_restart() and absorbed() == EVO0:
                reason = "degraded"; break
        else:
            reason = "no_promotion"; break
    time.sleep(5)
at = (camp().get("active_transaction") or {})
degraded = pending_restart()   # commit_sha exists AND restart_verified=False (same criterion as no_promotion)
print(json.dumps({"absorbed": reason == "absorbed", "reason": reason, "cycles": absorbed(),
                  "degraded": degraded, "active_tx_commit": str(at.get("commit_sha") or "")[:8],
                  "evo_before": EVO0, "sha_before": SHA0, "sha_after": head()}))
PYEOF
else
  printf '{"absorbed":false,"reason":"evolution_disabled","cycles":0,"degraded":false}\n' >/out/absorb.json
fi
"$OBO_PY" - <<'PYEOF' >&2 2>/dev/null || true
import json
try: d = json.load(open('/out/absorb.json'))
except Exception: d = {}
print(f"[pro] evolution: absorbed={d.get('absorbed',False)} cycles={d.get('cycles',0)} reason={d.get('reason','?')} degraded={d.get('degraded',False)}")
if d.get('degraded'):
    print(f"[pro] EVOLUTION_DEGRADED_RECOVERABLE: cycle committed (tx={d.get('active_tx_commit','')}) but in-cycle restart verification did not pass. "
          f"Core boot reconciliation / supervisor auto-restart will recover the transaction at the next server boundary; the run continues.")
PYEOF

SI_TID="$ROOT_TID"

if ! PYTHONPATH=/obo-repo "$OBO_PY" -c "import ouroboros.cli, ouroboros.agent, ouroboros.loop, ouroboros.config, ouroboros.subagent_worktrees, ouroboros.tools.subagent_integration, ouroboros.workspace_executor, ouroboros.contracts.task_constraint, ouroboros.retention, server, supervisor.queue" 2>/out/health.err; then
  echo "[pro] HEALTH-GATE FAILED - rollback self-edit to $REPO_HEAD0" >&2
  sed 's/^/[pro]   health: /' /out/health.err >&2 2>/dev/null | head -8
  git -C /obo-repo diff "$REPO_HEAD0" > /out/rejected_self_edit.diff 2>/dev/null || true
  git -C /obo-repo reset -q --hard "$REPO_HEAD0" 2>/dev/null || true
  git -C /obo-repo clean -qfd 2>/dev/null || true
  echo "[pro] HEALTH_GATE_ROLLBACK done -> /out/rejected_self_edit.diff" >&2
else
  echo "[pro] health-gate OK" >&2
fi

REPO_HEAD1="$(git -C /obo-repo rev-parse HEAD 2>/dev/null || echo '')"
SI_ROLLBACK=0; [ -s /out/rejected_self_edit.diff ] && SI_ROLLBACK=1 || true
"$OBO_PY" - "$REPO_HEAD0" "$REPO_HEAD1" "${SI_TID:-}" "$SI_ROLLBACK" <<'PYEOF' > /out/selfedit.json 2>/dev/null || printf '{}' > /out/selfedit.json
import json, subprocess, sys, glob, re
h0, h1, si_tid, rb = (sys.argv[1] or ""), (sys.argv[2] or ""), (sys.argv[3] or ""), sys.argv[4]
def git(*a):
    try: return subprocess.run(["git","-C","/obo-repo",*a], capture_output=True, text=True, timeout=30).stdout
    except Exception: return ""
grew = bool(h0 and h1 and h0 != h1)
commits_n = len([l for l in git("rev-list", f"{h0}..{h1}").splitlines() if l.strip()]) if grew else 0
files = [l for l in git("diff","--name-only",h0,h1).splitlines() if l.strip()] if grew else []
ss = git("diff","--shortstat",h0,h1) if grew else ""
mi = re.search(r'(\d+) insertion', ss); ins = int(mi.group(1)) if mi else 0
md = re.search(r'(\d+) deletion', ss); dele = int(md.group(1)) if md else 0
tools = [f for f in files if f.startswith("ouroboros/tools/") or "/skills/" in f]
verdicts = {}
for p in glob.glob("/obo-data/**/subagent_patch_verdict_*.json", recursive=True):
    try: o = json.load(open(p))
    except Exception: continue
    if si_tid and str(o.get("parent_task_id","")) != si_tid: continue
    k = str(o.get("outcome","")); verdicts[k] = verdicts.get(k, 0) + 1
print(json.dumps({"repo_head_before": h0, "repo_head_after": h1, "health_rollback": bool(int(rb)),
  "commits_added": commits_n, "files_changed": len(files), "file_list": files[:50],
  "loc_added": ins, "loc_removed": dele, "tools_added": tools, "verdicts": verdicts}))
PYEOF
"$OBO_PY" - <<'PYEOF' >&2 2>/dev/null || true
import json
try: d = json.load(open('/out/selfedit.json'))
except Exception: d = {}
print(f"[pro] selfedit: commits={d.get('commits_added',0)} loc=+{d.get('loc_added',0)}/-{d.get('loc_removed',0)} "
      f"tools={len(d.get('tools_added',[]))} verdicts={d.get('verdicts',{})} rollback={d.get('health_rollback',False)}")
PYEOF

echo "[pro] self-edit (obo-repo): HEAD before=$REPO_HEAD0 after=$(git -C /obo-repo rev-parse HEAD 2>/dev/null)" >&2
git -C /obo-repo status --porcelain 2>/dev/null | head -20 | sed 's/^/[pro]   /' >&2
echo "[pro] knowledge files: $(find /obo-data/memory/knowledge -type f 2>/dev/null | wc -l | tr -d ' ')" >&2

# B3 (strictly diagnostic, log-only — never affects solve/grading/evolution):
# detect the web-shadow class deterministically. With PYTHONPATH=/obo-repo, a
# target top-level package whose name also exists in the Ouroboros repo (`web`,
# `server`, `tests`, ...) gets shadowed — `import web` in $WORK resolves into
# Ouroboros, which surfaces as a pytest collection/usage error (exit 4) on the
# target's own tests. R2 fixes the ROOT (it scrubs /obo-repo from PYTHONPATH for
# target/user_files commands at runtime); this note just makes the class visible.
"$OBO_PY" - "$WORK" <<'PYEOF' >&2 2>/dev/null || true
import pathlib, sys
work = pathlib.Path(sys.argv[1]); repo = pathlib.Path("/obo-repo")
def has(base, name):
    return (base / name).is_dir() or (base / (name + ".py")).exists()
shadowed = [n for n in ("web", "server", "ouroboros", "supervisor", "tools", "tests", "docs", "scripts")
            if has(work, n) and has(repo, n)]
if shadowed:
    print(f"[pro] WEB_SHADOW_DIAGNOSTIC: target {work} and /obo-repo share top-level name(s) {shadowed}; "
          f"with /obo-repo on PYTHONPATH an `import {shadowed[0]}` in the target resolves into Ouroboros "
          f"(manifests as pytest exit=4 / collection error on the target's tests). R2 scrubs /obo-repo from "
          f"PYTHONPATH for target (user_files) commands so the agent sees the target's own module.")
else:
    print(f"[pro] WEB_SHADOW_DIAGNOSTIC: no top-level name collision between {work} and /obo-repo.")
PYEOF

kill "$SRV" 2>/dev/null || true
