#!/bin/bash
# Guest side of a requalification of the unified Update (test equipment, not product). Adapted from an earlier lab
# guest script. What changed:
#   - folders and labels: qb1, runs b1a (venv of the app's Python, 7.7.0 base), b1e (venv of an external CPython,
#     7.7.0 base), b1s (the default packaged user site, 7.8.0 base, stock 7.8.0 updater first), b1p (a bare external
#     CPython running the source launcher --no-ui), b1r (b1p's kind, no owner work; unqualified readiness mechanics);
#     one requested port per run;
#   - the candidate is bound by <namespace>/pkg/bind.env (digest-checked by the host) instead of literals;
#   - the official remote is served by a run-local bare mirror through the run home's own ~/.gitconfig insteadOf;
#     GitHub release assets and PyPI are NOT substituted: the live server fetches them itself (granted NAT only);
#   - preparation is the live server's own POST /api/update/apply {"strategy":"prepare"} (no prep_lab.py, no fetch
#     seam), followed through /api/update/status; the stock 7.8.0 update flow (check, preflight, apply auto_merge);
#     an owner restart through POST /api/command {"cmd":"/restart"}; a lab successor commit (mkplus) for the one
#     dependency-changing transition; package-identity facts in obs.
# Runs inside the lab VM as the guest user through `qa.py q LABEL SECONDS qb1.sh CMD RUN ...` (= `anka run <uuid>
# /bin/bash /Users/anka/oulab/<namespace>/bin/qb1.sh ...`, never -E). One qualification lives in one namespace folder
# Q = /Users/anka/oulab/<namespace> (qa.py qns binds it; default qb1): its bin, pkg, candsrc, runs and evidence, with tmp
# roots /private/tmp/<namespace>-RUN, so a fresh namespace never writes into an earlier one. Every evidence folder is
# create-only under $EV/RUN-STEP (or $EV/<namespace>-LABEL). Earlier lab runs ($ROOT/runs, older namespaces) and q7b4/
# are read, never written.
#   facts LABEL | selftest LABEL | netcheck LABEL      guest facts / observer selftest / reachability (granted NAT only)
#   install RUN                  fresh run root, its home (.bash_profile copy, .gitconfig insteadOf -> run mirror);
#                                the official app of the run's base version from the cached DMG (not for b1p)
#   mirror RUN                   run-local bare mirror: the base seed + the candidate bundle (main/ouroboros/
#                                ouroboros-stable = CAND; tags v7.7.0 v7.7.1 v7.8.0)
#   candsrc RUN                  once: clean checkout of CAND at qb1/candsrc (venvprep's source)
#   extpy RUN                    complete relocated copy of the official runtime (b1e: run's 7.7.0 app; b1p: 7.8.0 DMG)
#   venvprep RUN                 b1a/b1e: the owner's earlier venv by the candidate's own functions, wheels from the
#                                retained q7b4/wheelhouse (same lock bytes; DISCLOSED initial-state fixture)
#   barepip RUN                  b1p: the owner's earlier source install: ensurepip + the lock from the same wheelhouse
#   seed RUN                     b1p: the source-launcher seed tree at CAND (scripts/build_repo_bundle.py, lab tag)
#   open|openw RUN STEP [SECS] [ENV=VALUE...]   LaunchServices open (app runs) / spawn (b1p) with procwatch
#   obs RUN STEP | quit RUN STEP [SECS] | compose RUN | lscanary RUN STEP | effects RUN STEP
#   localfn RUN STEP [YAML_VERSION] [venv|private|base]   the owner's helper on the live server's own interpreter
#   api RUN STEP METHOD PATH [BODY_FILE] | status RUN STEP | settings RUN STEP KEY VALUE
#   prepare RUN STEP [SECS] [ready|failed=REASON|settled] [full|selected]   the live server's own preparation
#   activate RUN STEP [SECS]     the real activation request, procwatch armed, until that adoption closes
#   restart RUN STEP [SECS]      the owner's restart (POST /api/command /restart), procwatch armed
#   legacy RUN STEP [SECS]       the stock 7.8.0 updater: check, preflight, apply auto_merge (b1s)
#   mkplus RUN                   the lab successor P = CAND + one lock line, on the mirror's `ouroboros` branch
#   retarget RUN                 the mirror's `ouroboros` CAND -> the bound PUB_O1 (mechanics only; qualifies nothing)
#   wait RUN STEP SECS EXPR | pack LABEL
# Verdicts: an operational step writes OUT/verdict.txt (verdict=PASS|FAIL|UNKNOWN expected=... detail=...) and returns
# 0 PASS, 1 FAIL (the expected outcome was measured not to hold), 3 UNKNOWN (not established: unsettled, unreadable or
# an incomplete observation); 2 is a refusal before the step's effect. An exit status alone is never the verdict.
set -u
GH=/Users/anka
ROOT="$GH/oulab"
Q=$(cd "$(/usr/bin/dirname "$0")/.." 2>/dev/null && /bin/pwd -P)   # the namespace folder this verified copy runs from
NS=$(/usr/bin/basename "${Q:-/none}")
EV="$Q/evidence"
CLT="$ROOT/tools/clt-26.6-r2/Payload/Library/Developer/CommandLineTools"
GIT="$CLT/usr/bin/git"
OBSPY="$ROOT/pkg/7.8.0/Ouroboros.app/Contents/Frameworks/python-standalone/bin/python3"
PW="$Q/bin/procwatch.py"
VL="$Q/bin/venv_lab.py"
WHEELS="$ROOT/q7b4/wheelhouse"
BIND="$Q/pkg/bind.env"
EXTNAME=cpython-ext
PATH=/usr/bin:/bin:/usr/sbin:/sbin
export PATH
# The guest's /bin/bash is 3.2: no associative arrays.
dmg_of() { echo "$ROOT/packages/Ouroboros-$1.dmg"; }
dmgsha_of() { case "$1" in 7.7.0) echo de1387593eda0f5ea5a162e23002f116eabdd7ea974406946d9d4623a5d9bdef ;;
                           7.8.0) echo 41b9a8901d29c9a9b5dfdda8780d9317c441ba8c2941d949041c15abae0eff57 ;; esac; }
official_tag_of() { case "$1" in v7.7.0) echo e41596c389c3c33c3155637aaf67f8bebbf4934d ;;   # the official tags' commits
                                 v7.7.1) echo fc8cd8146671fdcb71d336b9b49970b6f9893a0a ;;
                                 v7.8.0) echo 6f3ddff05ff3e6c6f8d007ff96358df6af872e83 ;; esac; }
CAND= CTREE= CLOCK= CBUNDLE_NAME= CBUNDLE_SHA= PLUS_FROM= PLUS_TO= PLUS_DATE= PUB_O0= PUB_O1= VTAG_OFFICIAL=
# CAND CTREE CLOCK CBUNDLE_NAME CBUNDLE_SHA PLUS_FROM PLUS_TO PLUS_DATE [PUB_O0 PUB_O1] VTAG_OFFICIAL (host-checked digest)
[ -f "$BIND" ] && . "$BIND"
CBUNDLE="$Q/pkg/${CBUNDLE_NAME:-none}"
CANDSRC="$Q/candsrc"

die() { echo "qb1: $*" >&2; exit 2; }
namespace_ok() {   # this copy runs from $ROOT/<qb namespace>/bin; any other location is refused before an effect
    case "$NS" in qb|*[!a-z0-9-]*) die "invalid qualification namespace: $NS" ;; qb*) ;; *) die "invalid qualification namespace: $NS" ;; esac
    [ "$(/usr/bin/dirname "$Q")" = "$ROOT" ] || die "qb1.sh runs only from $ROOT/<namespace>/bin, not from $Q"
}
newdir() { namespace_ok; [ ! -e "$1" ] || die "already recorded: $1"; /bin/mkdir -p "$1" || die "cannot create $1"; echo "$1"; }
rc_line() { echo "exit_code=$1"; }
utc() { /bin/date -u '+%Y-%m-%dT%H:%M:%SZ'; }
verdict() {   # verdict OUT PASS|FAIL|UNKNOWN EXPECTED DETAIL...: the step's adjudicated outcome; returns 0, 1 or 3
    local out="$1" v="$2" expected="$3"; shift 3
    echo "verdict=$v expected=$expected detail=$*" > "$out/verdict.txt"
    case "$v" in PASS) return 0 ;; FAIL) return 1 ;; *) return 3 ;; esac
}
unobserved() {   # the caller's V=PASS rests on procwatch: UNKNOWN unless the watcher ended cleanly (disarm's WATCH_RC)
    if [ "$V" = PASS ] && [ "${WATCH_RC:-1}" != 0 ]; then V=UNKNOWN; WHY="$WHY; procwatch exit ${WATCH_RC:-unset}, observation incomplete"; fi
}
jget() {   # jget FILE KEY.PATH: one value of a JSON file ("" when absent); dicts/lists as JSON
    "$OBSPY" -I -S -B -c 'import json, sys
try:
    v = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
for k in sys.argv[2].split("."):
    v = v.get(k) if isinstance(v, dict) else None
print("" if v is None else (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v))' "$1" "$2" 2>/dev/null
}
run_root() {
    case "${1:-}" in
        b1a) KIND=app; BASEVER=7.7.0; SPORT=28821 ;;
        b1e) KIND=ext; BASEVER=7.7.0; SPORT=28822 ;;
        b1s) KIND=site; BASEVER=7.8.0; SPORT=28823 ;;
        b1p) KIND=bare; BASEVER=7.8.0; SPORT=28824 ;;
        b1r) KIND=bare; BASEVER=7.8.0; SPORT=28825 ;;   # a clean installation for the published-range readiness
        *) die "invalid run label (b1a|b1e|b1s|b1p|b1r)" ;;
    esac
    RUN="$1"; R="$Q/runs/$RUN"; APP="$R/Applications/Ouroboros.app"
    APPROOT="$R/app"; REPO="$APPROOT/repo"; DATA="$APPROOT/data"
    HOMEDIR="$R/home"; SETTINGS="$R/conf/settings.json"; PIDF="$R/locks/ouroboros.pid"; PORTF="$R/locks/server_port"
    TMPD="/private/tmp/$NS-$RUN/"; VENV="$R/venv"; MIRROR="$R/official.git"; SEED="$R/seed"
    EXTROOT="$R/ext-r2"; EXTPY="$EXTROOT/$EXTNAME"; LPIDF="$R/launcher.pid"
    case "$KIND" in
        app|site) BASE="$APP/Contents/Frameworks/python-standalone/bin/python3" ;;
        ext|bare) BASE="$EXTPY/bin/python3" ;;
    esac
    PRIVATE="$DATA/state/python-private"
}
step_dir() { case "${1:-}" in ''|*[!a-z0-9-]*) die "invalid step" ;; esac; newdir "$EV/$RUN-$1"; }
launch_pairs() {   # this run's launch environment as NAME=VALUE lines (the earlier lab's custom configuration, this run's port)
    printf '%s\n' "OUROBOROS_APP_ROOT=$APPROOT" "OUROBOROS_REPO_DIR=$REPO" "OUROBOROS_DATA_DIR=$DATA" \
        "OUROBOROS_SETTINGS_PATH=$SETTINGS" "OUROBOROS_PID_FILE=$PIDF" "TMPDIR=$TMPD" "OUROBOROS_PORT_FILE=$PORTF" \
        "HOME=$HOMEDIR" "CFFIXED_USER_HOME=$HOMEDIR" "OUROBOROS_SERVER_PORT=$SPORT" "OUROBOROS_HOST_SERVICE_PORT=$((SPORT + 100))" "OUROBOROS_SERVER_HOST=127.0.0.1" \
        "DEVELOPER_DIR=$CLT" "QB1_OPERATOR_CANARY=$RUN-operator-input-1"
    local extra; for extra in "${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}"; do echo "$extra"; done
}
open_argv() {   # the owner's open: the installed path, this run's environment, its explicit runtime venv (venv runs)
    OPEN_ARGV=(/usr/bin/open "$APP")
    while IFS= read -r line; do OPEN_ARGV+=(--env "$line"); done < <(launch_pairs)
    case "$KIND" in app|ext) OPEN_ARGV+=(--args --runtime-venv "$VENV") ;; esac
}
g() { HOME="$HOMEDIR" GIT_OPTIONAL_LOCKS=0 "$GIT" -C "$REPO" "$@"; }
run_rows() {   # ps rows naming the run root or this app; observers excluded
    /usr/bin/grep -F -e "$R/" -e "$APP" | /usr/bin/grep -v -F -e "qb1.sh" -e "procwatch.py" -e "$Q/bin/" -e "grep -F" -e "/usr/bin/grep "
}
pids_of_run() {   # every process naming the run root or this app (pid ppid pgid lstart command); observers excluded
    /bin/ps -axww -o pid=,ppid=,pgid=,lstart=,command= | run_rows
}
ps_run() {   # ps_run FILE: pids_of_run's rows into FILE; status 3 when ps itself failed (absence not established)
    local rows; rows=$(/bin/ps -axww -o pid=,ppid=,pgid=,lstart=,command=) || return 3
    printf '%s\n' "$rows" | run_rows > "$1"; return 0
}
server_rows() { pids_of_run | /usr/bin/grep -F "$REPO/server.py"; }   # identity rows (pid ppid pgid lstart command)
server_pids() { server_rows | /usr/bin/awk '{print $1}'; }
launcher_running() { pids_of_run | /usr/bin/grep -q -F -e "$APP/Contents/MacOS/" -e "$SEED/launcher.py" -e "$REPO/launcher.py"; }
bundle_facts() {   # bundle_facts APP: identity (device inode), version, entries, signature, quarantine
    local app="$1" f
    echo "== $app"; /usr/bin/stat -f 'device=%d inode=%i' "$app" "$app/Contents" 2>&1
    /usr/bin/plutil -extract CFBundleShortVersionString raw "$app/Contents/Info.plist" 2>&1
    /bin/ls -la "$app/Contents/MacOS" 2>&1
    /usr/bin/codesign --verify --deep --strict "$app" 2>&1; rc_line $?
    /usr/bin/codesign -dv --verbose=4 "$app" 2>&1 | /usr/bin/grep -E '^(Identifier|CDHash|TeamIdentifier|Authority)=' | /usr/bin/head -6
    for f in "$app"/Contents/MacOS/*; do echo "-- $f $(/usr/bin/shasum -a 256 "$f" | /usr/bin/cut -c1-64)"; done
    /usr/bin/xattr -l "$app" 2>&1
}
port_now() { /bin/cat "$PORTF" 2>/dev/null | /usr/bin/tr -dc 0-9; }
http_code() { local port; port=$(port_now); [ -n "$port" ] && /usr/bin/curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/" || echo -; }
cold_state() {   # cold_state OUT: no process of this run and no TCP listener on its requested, host-service or last
                 # recorded port (OUT/cold-*.txt); status 0 cold, 1 not cold, 3 not established
    local out="$1" port rc=0
    ps_run "$out/cold-ps.txt" || return 3
    [ -s "$out/cold-ps.txt" ] && rc=1
    : > "$out/cold-listen.txt"; : > "$out/cold-lsof.stderr"
    for port in $SPORT $((SPORT + 100)) $(port_now); do
        /usr/sbin/lsof -w -nP -iTCP:"$port" -sTCP:LISTEN >> "$out/cold-listen.txt" 2>> "$out/cold-lsof.stderr"
        case $? in 0) rc=1 ;; 1) ;; *) return 3 ;; esac
    done
    [ "$rc" = 0 ] && [ -s "$out/cold-lsof.stderr" ] && return 3   # lsof's "none" and its errors share status 1
    return "$rc"
}
arm() {   # arm OUT SECS: procwatch on this run (run root, app, TMPDIR, seed); sets J. Armed only when every process of
          # this run already running is registered for its exit and exit status; otherwise the step is refused
    "$OBSPY" -I -S -B "$PW" watch "$1/proc" "$R" "$APP" "$2" "$TMPD" "$SEED/" > "$1/procwatch.stdout" 2> "$1/procwatch.stderr" &
    J=$!
    local i; for i in $(/usr/bin/seq 1 50); do
        [ -f "$1/proc/armed" ] && return 0
        [ -f "$1/proc/arm-failed" ] && break
        /bin/sleep 0.2
    done
    /usr/bin/touch "$1/proc/stop" 2>/dev/null; wait "$J"; echo "procwatch_exit=$?" >> "$1/result.txt"
    die "procwatch did not arm (see $1/proc and $1/procwatch.stderr)"
}
disarm() { /usr/bin/touch "$1/proc/stop"; wait "$J"; WATCH_RC=$?; echo "procwatch_exit=$WATCH_RC" >> "$1/result.txt"; }
keep_if_changed() {   # keep_if_changed FILE DEST_PREFIX: a numbered copy whenever FILE's bytes changed
    local f="$1" p="$2" h last
    [ -f "$f" ] || return 0
    h=$(/usr/bin/shasum -a 256 "$f" 2>/dev/null | /usr/bin/cut -c1-64)
    last=$(/bin/cat "$p.last" 2>/dev/null)
    [ "$h" = "$last" ] && return 0
    echo "$h" > "$p.last"; KEPT=$((KEPT + 1)); /bin/cp "$f" "$p-$KEPT-$(/bin/date -u +%H%M%S)"
}
get_status() {   # get_status FILE: GET /api/update/status into FILE; prints the HTTP code
    local port; port=$(port_now); [ -n "$port" ] || { echo -; return; }
    /usr/bin/curl -sS -m 60 -o "$1" -w '%{http_code}' "http://127.0.0.1:$port/api/update/status" 2>/dev/null || echo -
}
closed_count() {   # one integer: the supervisor log's body_adoption_closed rows; "-" (status 3) when it is unreadable
    local f="$DATA/logs/supervisor.jsonl" n
    [ -r "$f" ] || { echo -; return 3; }
    n=$(/usr/bin/grep -c '"body_adoption_closed"' "$f")
    case $? in 0|1) echo "$n" ;; *) echo -; return 3 ;; esac   # 1 is grep's "no row", already printed as 0
}
closure_outcome() {   # closure_outcome ADOPTION_ID: that adoption's own body_adoption_closed outcome, "" while it is open
    [ -n "$1" ] || return 0
    "$OBSPY" -I -S -B -c 'import json, sys
try:
    lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
except OSError:
    sys.exit(0)
found = ""
for line in lines:
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if isinstance(row, dict) and row.get("type") == "body_adoption_closed" and row.get("adoption_id") == sys.argv[2]:
        found = str(row.get("outcome") or "-")
print(found)' "$DATA/logs/supervisor.jsonl" "$1" 2>/dev/null
}
exit_of() {   # exit_of EVENTS_JSONL PID: that pid's exit as procwatch observed it (exit_code=N|signal=N|unreported), else ""
    "$OBSPY" -I -S -B -c 'import json, sys
found = ""
try:
    lines = open(sys.argv[1], errors="replace").read().splitlines()
except OSError:
    lines = []
for line in lines:
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if row.get("event") == "exit" and str(row.get("pid")) == sys.argv[2]:
        found = ("exit_code=%s" % row.get("exit_code") if row.get("exited") else
                 "signal=%s" % row.get("signal") if row.get("exited") is False else "unreported")
print(found)' "$1" "$2" 2>/dev/null
}

facts() {
    local out f; RUN="$NS"; out=$(step_dir "facts-$1") || exit 2
    utc > "$out/utc.txt"
    /bin/df -k "$GH" > "$out/df.txt" 2>&1
    /bin/ls -la "$ROOT" "$ROOT/runs" "$ROOT/packages" "$ROOT/pkg" "$Q" "$Q/bin" "$Q/pkg" "$Q/runs" "$WHEELS" > "$out/ls.txt" 2>&1
    for f in "$ROOT"/runs/*/Applications "$Q"/runs/*/Applications; do echo "== $f"; /bin/ls -la "$f" 2>&1; done > "$out/prior-runs-apps.txt"
    /bin/ps -axww -o pid,ppid,pgid,user,lstart,command | /usr/bin/grep -i -E 'ouroboros|python|osascript|app_install' \
        | /usr/bin/grep -v -F -e qb1.sh -e 'grep -i' > "$out/ps-ouroboros.txt" 2>&1
    /usr/sbin/lsof -nP -a -iTCP -sTCP:LISTEN -u "$(/usr/bin/id -u)" > "$out/listen.txt" 2>&1
    { for f in "$GH/Library/WebKit/com.ouroboros.agent" "$GH/Library/Preferences/com.ouroboros.agent.plist" "$GH/Ouroboros" \
               "$GH/.gitconfig"; do echo "== $f"; /bin/ls -ld -T "$f" 2>&1; done; } > "$out/profile.txt" 2>&1
    /usr/bin/shasum -a 256 "$OBSPY" "$Q"/bin/* "$BIND" "$CBUNDLE" "$GIT" "$(dmg_of 7.7.0)" "$(dmg_of 7.8.0)" \
        "$WHEELS/SHA256SUMS" "$WHEELS/wheelhouse-report.json" > "$out/inputs-sha256.txt" 2>&1
    { echo "bind:"; /bin/cat "$BIND" 2>&1; echo "wheelhouse lock_sha256: $(jget "$WHEELS/wheelhouse-report.json" lock_sha256)"
      echo "wheels: $(/bin/ls "$WHEELS"/*.whl 2>/dev/null | /usr/bin/wc -l | /usr/bin/tr -d ' ')"; } > "$out/bind.txt" 2>&1
    "$OBSPY" -I -S -B -c 'import sys, select; print(sys.version); print(hasattr(select, "kqueue"))' > "$out/observer-python.txt" 2>&1
    rc_line $? >> "$out/observer-python.txt"
    "$GIT" --version > "$out/git.txt" 2>&1
    /usr/bin/sw_vers > "$out/sw_vers.txt" 2>&1
    /usr/sbin/sysctl -n hw.ncpu hw.memsize > "$out/hw.txt" 2>&1
    echo "$out"
}

selftest() {
    local out rc; RUN="$NS"; out=$(step_dir "selftest-$1") || exit 2
    "$OBSPY" -I -S -B "$PW" selftest "$out/selftest.json" > "$out/stdout.txt" 2> "$out/stderr.txt"; rc=$?; rc_line "$rc" > "$out/rc.txt"
    echo "$out"
    if [ "$rc" = 0 ]; then verdict "$out" PASS method_ok "the exit status 7 of a re-parented process was observed"
    elif [ -f "$out/selftest.json" ]; then verdict "$out" FAIL method_ok "selftest exit $rc (selftest.json)"
    else verdict "$out" UNKNOWN method_ok "no selftest record (exit $rc)"; fi
}

netcheck() {   # only under a granted NAT: which public endpoints answer (no body kept), and the host stays unreachable
    local out u; RUN="$NS"; out=$(step_dir "netcheck-$1") || exit 2
    for u in https://pypi.org/simple/pyyaml/ https://files.pythonhosted.org/ \
             https://github.com/razzant/ouroboros/releases/download/v7.8.0/SHA256SUMS; do
        echo "$u $(/usr/bin/curl -sS -I -m 20 -o /dev/null -w '%{http_code} %{remote_ip}' "$u" 2>&1)"; done > "$out/public.txt"
    /usr/sbin/netstat -rn -f inet > "$out/routes.txt" 2>&1
    /usr/sbin/scutil --dns > "$out/dns.txt" 2>&1
    echo "$out"
}

install() {   # PASS: the official app copied whole, its signature and Gatekeeper assessment accepted, version BASEVER
    run_root "$1"; local out mnt dmg copied=- signed=- assessed=- version=
    [ ! -e "$R" ] || die "run root exists: $R"
    [ ! -e "${TMPD%/}" ] || die "tmp root exists: ${TMPD%/}"
    out=$(step_dir install) || exit 2
    /bin/mkdir -p -m 700 "$R" "$R/Applications" "${TMPD%/}" "$HOMEDIR" "$R/conf" "$R/locks" || die "run root"
    /usr/bin/touch "$R/.start"
    /bin/cp "$GH/.bash_profile" "$HOMEDIR/.bash_profile" || die "home profile"
    # The owner's own git configuration in this home: the official URL is served by the run's mirror (DISCLOSED).
    { printf '[url "%s"]\n' "$MIRROR"
      printf '\tinsteadOf = https://github.com/razzant/ouroboros\n\tinsteadOf = https://github.com/razzant/ouroboros.git\n'
    } > "$HOMEDIR/.gitconfig"
    { echo "KIND=$KIND BASEVER=$BASEVER SPORT=$SPORT VENV=$VENV BASE=$BASE MIRROR=$MIRROR"; launch_pairs; } > "$out/launch-env.txt"
    /bin/cp "$HOMEDIR/.gitconfig" "$out/gitconfig"
    if [ "$KIND" != bare ]; then
        dmg=$(dmg_of "$BASEVER")
        { /usr/bin/shasum -a 256 "$dmg"; /usr/bin/xattr -l "$dmg"; } > "$out/dmg-identity.txt" 2>&1
        /usr/bin/shasum -a 256 "$dmg" | /usr/bin/grep -q "^$(dmgsha_of "$BASEVER") " || die "$BASEVER DMG digest mismatch"
        mnt="$R/.mnt-$BASEVER"; /bin/mkdir -p "$mnt"
        /usr/bin/hdiutil attach -readonly -nobrowse -noautoopen -mountpoint "$mnt" "$dmg" > "$out/attach.txt" 2>&1 || die "attach failed"
        /usr/bin/ditto "$mnt/Ouroboros.app" "$APP" > "$out/ditto.txt" 2>&1; copied=$?; rc_line "$copied" >> "$out/ditto.txt"
        /usr/bin/hdiutil detach "$mnt" > "$out/detach.txt" 2>&1; rc_line $? >> "$out/detach.txt"
        /bin/rmdir "$mnt"
        bundle_facts "$APP" > "$out/bundle.txt" 2>&1
        /usr/bin/codesign --verify --deep --strict "$APP" > "$out/codesign.txt" 2>&1; signed=$?; rc_line "$signed" >> "$out/codesign.txt"
        /usr/sbin/spctl -a -vv -t exec "$APP" > "$out/spctl.txt" 2>&1; assessed=$?; rc_line "$assessed" >> "$out/spctl.txt"
        version=$(/usr/bin/plutil -extract CFBundleShortVersionString raw "$APP/Contents/Info.plist" 2>/dev/null)
        /usr/bin/du -sk "$APP" > "$out/du.txt" 2>&1
    fi
    /bin/ls -laR "$R" 2>&1 | /usr/bin/head -60 > "$out/run-root.txt"
    echo "$out"
    [ "$KIND" = bare ] && { verdict "$out" PASS "run root" "b1p: run root and home only (no app)"; return; }
    if [ "$copied" = 0 ] && [ "$signed" = 0 ] && [ "$assessed" = 0 ] && [ "$version" = "$BASEVER" ]; then
        verdict "$out" PASS "official $BASEVER app" "copied, signature and Gatekeeper accepted"
    else
        verdict "$out" FAIL "official $BASEVER app" "ditto=$copied codesign=$signed spctl=$assessed version=${version:--}"
    fi
}

with_seed() {   # with_seed CALLBACK: the base release's own repo.bundle (the run's app, else the read-only 7.8.0 DMG)
    local seed mnt=""
    seed="$APP/Contents/Resources/repo.bundle"; [ -f "$seed" ] || seed="$APP/Contents/Frameworks/repo.bundle"
    if [ ! -f "$seed" ]; then
        mnt="$R/.mnt-seed"; /bin/mkdir -p "$mnt"
        /usr/bin/hdiutil attach -readonly -nobrowse -noautoopen -mountpoint "$mnt" "$(dmg_of 7.8.0)" >&2 || die "attach failed"
        seed="$mnt/Ouroboros.app/Contents/Resources/repo.bundle"
        [ -f "$seed" ] || seed="$mnt/Ouroboros.app/Contents/Frameworks/repo.bundle"
    fi
    "$1" "$seed"; local rc=$?
    [ -n "$mnt" ] && { /usr/bin/hdiutil detach "$mnt" >&2; /bin/rmdir "$mnt"; }
    return $rc
}

mirror_step() {   # mirror_step NAME CMD...: one mirror step, its own status recorded and returned (never left to errexit)
    local name="$1" rc; shift
    "$@"; rc=$?; echo "step=$name exit_code=$rc"; return "$rc"
}

mirror_from() {   # clone, config, bundle verify, fetch and ref steps in order; the first failing step's status is returned
    local seed="$1" ref
    echo "seed=$seed"; /usr/bin/shasum -a 256 "$seed" "$CBUNDLE"
    set -- "refs/heads/qb1-candidate:refs/heads/main" "refs/tags/v7.7.0:refs/tags/v7.7.0" \
        "refs/tags/v7.7.1:refs/tags/v7.7.1" "refs/tags/v7.8.0:refs/tags/v7.8.0"
    [ -z "$PUB_O1" ] || set -- "$@" "refs/heads/qb1-published-o1:refs/lab/published-o1"
    mirror_step clone "$GIT" clone -q --bare "$seed" "$MIRROR" || return
    mirror_step config "$GIT" -C "$MIRROR" config gc.auto 0 || return
    mirror_step verify "$GIT" -C "$MIRROR" bundle verify "$CBUNDLE" || return
    mirror_step fetch "$GIT" -C "$MIRROR" fetch -q "$CBUNDLE" "$@" || return
    for ref in ouroboros ouroboros-stable; do
        mirror_step "ref-$ref" "$GIT" -C "$MIRROR" update-ref "refs/heads/$ref" "$CAND" || return
    done
    echo "main=$("$GIT" -C "$MIRROR" rev-parse main) tree=$("$GIT" -C "$MIRROR" rev-parse 'main^{tree}') expected=$CAND/$CTREE"
    "$GIT" -C "$MIRROR" for-each-ref --format='%(refname) %(objectname) %(objecttype)'
}

mirror_check() {   # "ok ..." when the mirror holds exactly the bound refs, else the first difference (status 1)
    local ref t want got ver vt
    for ref in main ouroboros ouroboros-stable; do
        got=$("$GIT" -C "$MIRROR" rev-parse -q --verify "refs/heads/$ref^{commit}")
        [ "$got" = "$CAND" ] || { echo "refs/heads/$ref is ${got:-absent}, not CAND $CAND"; return 1; }
    done
    got=$("$GIT" -C "$MIRROR" rev-parse -q --verify "$CAND^{tree}")
    [ "$got" = "$CTREE" ] || { echo "the CAND tree is ${got:-absent}, not CTREE $CTREE"; return 1; }
    got=$("$GIT" -C "$MIRROR" cat-file blob "$CAND:requirements-runtime.lock" | /usr/bin/shasum -a 256 | /usr/bin/cut -c1-64)
    [ -z "$CLOCK" ] || [ "$got" = "$CLOCK" ] || { echo "CAND's runtime lock is $got, not CLOCK $CLOCK"; return 1; }
    for t in v7.7.0 v7.7.1 v7.8.0; do
        want=$(official_tag_of "$t"); got=$("$GIT" -C "$MIRROR" rev-parse -q --verify "refs/tags/$t^{commit}")
        [ "$got" = "$want" ] || { echo "tag $t names ${got:-nothing}, not the official $want"; return 1; }
    done
    if [ -n "$PUB_O1" ]; then
        got=$("$GIT" -C "$MIRROR" rev-parse -q --verify "refs/lab/published-o1^{commit}")
        [ "$got" = "$PUB_O1" ] || { echo "refs/lab/published-o1 is ${got:-absent}, not PUB_O1 $PUB_O1"; return 1; }
        got=$("$GIT" -C "$MIRROR" merge-base "$CAND" "$PUB_O1")
        [ "$got" = "$PUB_O0" ] || { echo "merge-base(CAND, PUB_O1) is ${got:-none}, not PUB_O0 $PUB_O0"; return 1; }
    fi
    ver=$("$GIT" -C "$MIRROR" cat-file blob "$CAND:VERSION" | /usr/bin/tr -d ' \n')
    vt=$("$GIT" -C "$MIRROR" rev-parse -q --verify "refs/tags/v$ver^{commit}")
    if [ -n "$vt" ] && [ "$vt" != "$CAND" ]; then
        echo "the mirror's v$ver names $vt, not CAND: the seed's lab provenance tag would collide"; return 1
    fi
    echo "ok main/ouroboros/ouroboros-stable=CAND tree=CTREE official tags pinned${PUB_O1:+ published-o1=$PUB_O1}; VERSION=$ver" \
         "v$ver in the mirror: ${vt:-absent}; official v$ver (host): ${VTAG_OFFICIAL:-none}"
}

mirror() {
    run_root "$1"; local out why rc
    [ -n "$CAND" ] && [ -f "$CBUNDLE" ] || die "no bound candidate bundle"
    [ ! -e "$MIRROR" ] || die "exists: $MIRROR"
    out=$(step_dir mirror) || exit 2
    with_seed mirror_from > "$out/mirror.txt" 2>&1; rc=$?   # not a condition: its status is the first failing step's
    if [ "$rc" != 0 ]; then
        echo "$out"; verdict "$out" FAIL "the exact bound mirror" "a clone, config, verify, fetch or ref step failed (exit $rc, mirror.txt)"
        return
    fi
    why=$(mirror_check); echo "$why" > "$out/refs-check.txt"
    echo "$out"
    case "$why" in ok*) verdict "$out" PASS "the exact bound mirror" "$why" ;; *) verdict "$out" FAIL "the exact bound mirror" "$why" ;; esac
}

candsrc() {   # once: the frozen candidate, checked out cleanly beside the runs (never a run's checkout)
    run_root "$1"; local out head tree dirty
    [ ! -e "$CANDSRC" ] || die "exists: $CANDSRC"
    out=$(newdir "$EV/$NS-candsrc") || exit 2
    {
        "$GIT" clone -q --no-checkout "$MIRROR" "$CANDSRC"; rc_line $?
        "$GIT" -C "$CANDSRC" checkout -q --detach "$CAND"; rc_line $?
        echo "HEAD=$("$GIT" -C "$CANDSRC" rev-parse HEAD) TREE=$("$GIT" -C "$CANDSRC" rev-parse 'HEAD^{tree}') expected_tree=$CTREE"
        echo "status_lines=$("$GIT" -C "$CANDSRC" status --porcelain | /usr/bin/wc -l | /usr/bin/tr -d ' ')"
        /usr/bin/shasum -a 256 "$CANDSRC/requirements-runtime.lock" "$CANDSRC/ouroboros/runtime_venv.py" \
            "$CANDSRC/ouroboros/launcher_bootstrap.py"
    } > "$out/candsrc.txt" 2>&1
    head=$("$GIT" -C "$CANDSRC" rev-parse HEAD 2>/dev/null); tree=$("$GIT" -C "$CANDSRC" rev-parse 'HEAD^{tree}' 2>/dev/null)
    dirty=$("$GIT" -C "$CANDSRC" status --porcelain 2>&1)
    echo "$out"
    if [ "$head" = "$CAND" ] && [ "$tree" = "$CTREE" ] && [ -z "$dirty" ]; then verdict "$out" PASS "clean CAND checkout" "HEAD=$head"
    else verdict "$out" FAIL "clean CAND checkout" "HEAD=${head:--} TREE=${tree:--} status_lines=$(printf '%s' "$dirty" | /usr/bin/grep -c .)"; fi
}

extpy_from() {   # extpy_from APP: the earlier lab's method (complete relocated copy; symlinks followed; empty site-packages);
                 # its last line counts the copy steps that failed
    local fw="$1/Contents/Frameworks" ps e f n=0 rc
    ps="$fw/python-standalone"
    /bin/mkdir -p "$EXTPY/bin" "$EXTPY/lib/python3.10/site-packages"; rc=$?; rc_line "$rc"; [ "$rc" = 0 ] || n=$((n + 1))
    for f in python3 python3.10; do /bin/cp -c -L "$ps/bin/$f" "$EXTPY/bin/$f"; rc=$?; rc_line "$rc"; [ "$rc" = 0 ] || n=$((n + 1)); done
    /bin/cp -c -L "$ps/lib/libpython3.10.dylib" "$EXTPY/lib/libpython3.10.dylib"; rc=$?; rc_line "$rc"; [ "$rc" = 0 ] || n=$((n + 1))
    for e in "$ps"/lib/python3.10/*; do
        [ "$(/usr/bin/basename "$e")" = site-packages ] && continue
        /bin/cp -c -R -L "$e" "$EXTPY/lib/python3.10/$(/usr/bin/basename "$e")" || { echo "copy failed: $e"; n=$((n + 1)); }
    done
    for f in "$fw"/*.dylib; do [ -L "$f" ] && continue; /bin/cp -c "$f" "$EXTROOT/$(/usr/bin/basename "$f")"; rc=$?
        echo "$(/usr/bin/basename "$f") rc=$rc"; [ "$rc" = 0 ] || n=$((n + 1)); done
    echo "copy_failures=$n"
}

extpy() {   # b1e: from the run's installed 7.7.0 app; b1p: from the read-only 7.8.0 DMG (both DISCLOSED copies)
            # PASS: every copy step succeeded, no symlink leads into an app, and the copy runs its probe
    run_root "$1"; local out l t rp mnt probe=- fails into
    case "$KIND" in ext|bare) ;; *) die "extpy is for b1e/b1p" ;; esac
    [ ! -e "$EXTROOT" ] || die "exists: $EXTROOT"
    out=$(step_dir extpy) || exit 2
    if [ "$KIND" = ext ]; then
        extpy_from "$APP" > "$out/copy.txt" 2>&1
    else
        mnt="$R/.mnt-7.8.0"; /bin/mkdir -p "$mnt"
        /usr/bin/hdiutil attach -readonly -nobrowse -noautoopen -mountpoint "$mnt" "$(dmg_of 7.8.0)" > "$out/attach.txt" 2>&1 || die "attach failed"
        extpy_from "$mnt/Ouroboros.app" > "$out/copy.txt" 2>&1
        /usr/bin/hdiutil detach "$mnt" > "$out/detach.txt" 2>&1; /bin/rmdir "$mnt"
    fi
    {
        echo "external_root=$EXTROOT external_python=$EXTPY/bin/python3"
        echo "-- symlinks under $EXTROOT (none may lead into an app)"
        /usr/bin/find "$EXTROOT" -type l 2>/dev/null | while IFS= read -r l; do
            t=$(/usr/bin/readlink "$l"); rp=$(/bin/realpath "$l" 2>/dev/null || echo "?")
            case "$rp" in *.app/*) echo "INTO_APP $l -> $t ($rp)" ;; "$EXTROOT"/*) echo "INSIDE $l -> $t" ;; *) echo "OTHER $l -> $t ($rp)" ;; esac
        done
        echo "symlink_count=$(/usr/bin/find "$EXTROOT" -type l | /usr/bin/wc -l | /usr/bin/tr -d ' ') regular_files=$(/usr/bin/find "$EXTROOT" -type f | /usr/bin/wc -l | /usr/bin/tr -d ' ')"
        /usr/bin/codesign --verify --verbose=2 "$EXTPY/bin/python3" 2>&1; rc_line $?
        "$EXTPY/bin/python3" -I -S -B -c 'import ctypes, json, os, sqlite3, ssl, sys, ensurepip, venv
libc = ctypes.CDLL(None)
libc._dyld_image_count.restype = ctypes.c_uint32
libc._dyld_get_image_name.restype = ctypes.c_char_p
images = [libc._dyld_get_image_name(i).decode() for i in range(libc._dyld_image_count())]
print(json.dumps({"version": sys.version, "sqlite": sqlite3.sqlite_version, "openssl": ssl.OPENSSL_VERSION,
    "executable": sys.executable, "real": os.path.realpath(sys.executable), "prefix": sys.prefix,
    "nonsystem_images": [n for n in images if not n.startswith(("/usr/lib/", "/System/"))]}, indent=1))' 2>&1
        probe=$?; rc_line "$probe"
    } > "$out/extpy.txt" 2>&1
    /usr/bin/du -sk "$EXTROOT" > "$out/du.txt" 2>&1
    fails=$(/usr/bin/sed -n 's/^copy_failures=//p' "$out/copy.txt"); into=$(/usr/bin/grep -c '^INTO_APP ' "$out/extpy.txt")
    echo "$out"
    if [ "$fails" = 0 ] && [ "$into" = 0 ] && [ "$probe" = 0 ]; then verdict "$out" PASS "complete relocated runtime" "runs its probe"
    else verdict "$out" FAIL "complete relocated runtime" "copy_failures=${fails:-unknown} into_app=$into probe=$probe"; fi
}

wheels_match() {   # the retained wheelhouse was built for exactly this lock's bytes
    [ "$(jget "$WHEELS/wheelhouse-report.json" lock_sha256)" = "$(/usr/bin/shasum -a 256 "$1" | /usr/bin/cut -c1-64)" ]
}

venvprep() {   # the owner's earlier venv preparation, through the candidate's own functions (DISCLOSED fixture seam)
               # PASS: the preparer exited 0 with a ready result and the venv facts were read
    run_root "$1"; local out prep facts_rc ready
    case "$KIND" in app|ext) ;; *) die "venvprep is for b1a/b1e" ;; esac
    [ ! -e "$VENV" ] || die "exists: $VENV"
    [ -d "$CANDSRC" ] && [ -d "$WHEELS" ] || die "candsrc and the retained wheelhouse first"
    wheels_match "$CANDSRC/requirements-runtime.lock" || die "the retained wheelhouse was built for another lock"
    [ -x "$BASE" ] || die "no base Python $BASE"
    out=$(step_dir venvprep) || exit 2
    /bin/mkdir -p "$R/venvprep/home" "$R/venvprep/data"
    echo "KIND=$KIND BASE=$BASE VENV=$VENV WHEELS=$WHEELS" > "$out/inputs.txt"
    /usr/bin/shasum -a 256 "$VL" "$BASE" "$WHEELS/SHA256SUMS" >> "$out/inputs.txt" 2>&1
    utc > "$out/utc-before.txt"
    /usr/bin/env -i HOME="$R/venvprep/home" PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="$TMPD" \
        PIP_NO_INDEX=1 PIP_FIND_LINKS="$WHEELS" PIP_DISABLE_PIP_VERSION_CHECK=1 \
        "$BASE" -I -B "$VL" prepare "$VENV" "$BASE" "$CANDSRC" "$APP" "$R/venvprep/data" "$out/prep" \
        > "$out/stdout.txt" 2> "$out/stderr.txt"
    prep=$?; rc_line "$prep" > "$out/rc.txt"; utc > "$out/utc-after.txt"
    "$OBSPY" -I -S -B "$VL" facts "$VENV" "$APP" "$CANDSRC" "$out/venv-facts.json" > /dev/null 2> "$out/venv-facts.stderr"
    facts_rc=$?
    /usr/bin/du -sk "$VENV" > "$out/du.txt" 2>&1
    ready=$(jget "$out/prep/result.json" final.ready)
    echo "$out"
    if [ "$prep" != 0 ]; then verdict "$out" FAIL "ready venv" "venv_lab prepare exit $prep (stderr.txt, prep/)"
    elif [ "$ready" != True ]; then verdict "$out" UNKNOWN "ready venv" "exit 0 without a ready prep/result.json"
    elif [ "$facts_rc" != 0 ]; then verdict "$out" UNKNOWN "ready venv" "the venv facts were not read (exit $facts_rc)"
    else verdict "$out" PASS "ready venv" "prepared and recorded ready"; fi
}

barepip() {   # b1p: the owner's earlier source install into the bare interpreter itself (DISCLOSED initial-state fixture)
    run_root "$1"; local out
    [ "$KIND" = bare ] || die "barepip is for b1p"
    [ -x "$BASE" ] && [ -d "$SEED" ] || die "extpy and seed first"
    wheels_match "$SEED/requirements-runtime.lock" || die "the retained wheelhouse was built for another lock"
    out=$(step_dir barepip) || exit 2
    /bin/mkdir -p "$R/barepip-home"
    (
        set -euo pipefail
        /usr/bin/env -i HOME="$R/barepip-home" PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="$TMPD" "$BASE" -I -m ensurepip; rc_line $?
        /usr/bin/env -i HOME="$R/barepip-home" PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="$TMPD" PIP_NO_INDEX=1 \
            PIP_FIND_LINKS="$WHEELS" PIP_DISABLE_PIP_VERSION_CHECK=1 "$BASE" -I -m pip install -r "$SEED/requirements-runtime.lock"
        rc_line $?
        "$BASE" -I -m pip list --format=freeze 2>&1
    ) > "$out/barepip.txt" 2>&1
    local rc=$?; echo "exit_code=$rc" > "$out/result.txt"
    echo "$out"; return "$rc"
}

seed() {   # b1p: the source launcher's seed tree at CAND, its bundle built by the product's own builder (lab tag)
    run_root "$1"; local out
    [ "$KIND" = bare ] || die "seed is for b1p"
    [ ! -e "$SEED" ] || die "exists: $SEED"
    [ -x "$BASE" ] || die "extpy first"
    out=$(step_dir seed) || exit 2
    (
        set -euo pipefail
        "$GIT" -c gc.auto=0 clone -q --no-local "$MIRROR" "$SEED"; rc_line $?
        "$GIT" -C "$SEED" checkout -q -B ouroboros "$CAND"; rc_line $?
        # The builder's provenance needs VERSION's tag on HEAD: a lab tag in the seed clone only (never the mirror).
        # A v$VERSION the clone already holds elsewhere is refused, never moved; the official name it shadows is disclosed.
        ver=$(/bin/cat "$SEED/VERSION"); vt=$("$GIT" -C "$SEED" rev-parse -q --verify "refs/tags/v$ver^{commit}" || true)
        if [ -n "$vt" ] && [ "$vt" != "$CAND" ]; then echo "refusing: v$ver already names $vt in the seed clone"; exit 3; fi
        echo "lab provenance tag v$ver -> CAND in the seed clone only; official v$ver (host): ${VTAG_OFFICIAL:-none}"
        if [ -z "$vt" ]; then
            "$GIT" -C "$SEED" -c user.name=qb1-lab -c user.email=lab@lab.invalid tag -a "v$ver" \
                -m "qb1 lab seed tag (seed clone only)" "$CAND"; rc_line $?
        fi
        /usr/bin/env -i HOME="$HOMEDIR" PATH=/usr/bin:/bin:/usr/sbin:/sbin DEVELOPER_DIR="$CLT" TMPDIR="$TMPD" \
            OUROBOROS_RELEASE_TAG="v$(/bin/cat "$SEED/VERSION")" "$BASE" -I -B "$SEED/scripts/build_repo_bundle.py" \
            --repo-root "$SEED" --output-bundle "$SEED/repo.bundle" --output-manifest "$SEED/repo_bundle_manifest.json"
        rc_line $?
        /usr/bin/shasum -a 256 "$SEED/repo.bundle" "$SEED/repo_bundle_manifest.json"; /bin/cat "$SEED/repo_bundle_manifest.json"
        echo "HEAD=$("$GIT" -C "$SEED" rev-parse HEAD) TREE=$("$GIT" -C "$SEED" rev-parse 'HEAD^{tree}') expected=$CTREE"
        echo "status: $("$GIT" -C "$SEED" status --porcelain --untracked-files=all | /usr/bin/head -5)"
    ) > "$out/seed.txt" 2>&1
    local rc=$?; echo "exit_code=$rc" > "$out/result.txt"
    echo "$out"; return "$rc"
}

spawn_bare() {   # spawn_bare OUT: the unchanged source-launcher command, detached in its own session (b1p)
    local envfile="$1/launch-env.txt"
    launch_pairs > "$envfile"; echo "PATH=/usr/bin:/bin:/usr/sbin:/sbin" >> "$envfile"
    "$OBSPY" -I -S -B -c 'import json, os, runpy, sys
out, envfile, abi, argv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
kernel_birth = runpy.run_path(abi)["kernel_birth"]
env = dict(line.rstrip("\n").split("=", 1) for line in open(envfile) if "=" in line)
first = os.fork()
if first == 0:
    os.setsid()
    second = os.fork()
    if second == 0:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        stdout, stderr = os.open(out + "/launcher.stdout", flags, 0o600), os.open(out + "/launcher.stderr", flags, 0o600)
        os.dup2(os.open("/dev/null", os.O_RDONLY), 0); os.dup2(stdout, 1); os.dup2(stderr, 2)
        os.chdir(env["HOME"])
        os.execve(argv[0], argv, env)
    birth = kernel_birth(second)
    with open(out + "/launcher.pid", "x") as handle:
        handle.write("%d\n" % second)
    with open(out + "/launcher.birth.json", "x") as handle:
        json.dump({"pid": second, "birth": "darwin-start:%d.%06d" % birth[:2] if birth else None}, handle)
    os._exit(0)
_, status = os.waitpid(first, 0)
sys.exit(os.waitstatus_to_exitcode(status))' "$1" "$envfile" "$Q/bin/platform_darwin_abi.py" "$BASE" "$SEED/launcher.py" --no-ui
    local rc=$?; [ "$rc" = 0 ] || return "$rc"
    /bin/cp "$1/launcher.pid" "$LPIDF.$(/bin/date -u +%H%M%S)"; /bin/cp "$1/launcher.pid" "$LPIDF.current"
    /bin/cp "$1/launcher.birth.json" "$LPIDF.current.birth.json"
}

open_app() {   # open RUN STEP [ENV=VALUE...]: without the observer; returns only the open request's own status (no verdict)
    run_root "$1"; local out step="$2" opened; shift 2; EXTRA_ENV=("$@")
    out=$(step_dir "$step") || exit 2
    utc > "$out/utc-before.txt"
    if [ "$KIND" = bare ]; then spawn_bare "$out"; opened=$?; rc_line "$opened" > "$out/rc.txt"
    else open_argv; printf '%s\n' "${OPEN_ARGV[@]}" > "$out/argv.txt"
         "${OPEN_ARGV[@]}" > "$out/stdout.txt" 2> "$out/stderr.txt"; opened=$?; rc_line "$opened" > "$out/rc.txt"; fi
    utc > "$out/utc-after.txt"
    echo "$out"; return "$opened"
}

openw() {   # openw RUN STEP SECONDS [ENV=VALUE...]: from a cold run (cold_state, else refused), procwatch first, then the
            # open, until launcher + server serve. PASS: served within SECONDS; FAIL: the open failed or nothing of the
            # run is left; UNKNOWN: running but not serving, or an incomplete observation
    run_root "$1"; local out step="$2" secs="${3:-300}" i code opened served=0 left V WHY; shift 3; EXTRA_ENV=("$@")
    out=$(step_dir "$step") || exit 2
    cold_state "$out"
    case $? in 0) ;; 1) die "not a cold start: a process of $RUN or a listener on its ports remains ($out/cold-*.txt)" ;;
               *) die "the cold state was not established ($out/cold-*.txt)" ;; esac
    arm "$out" "$secs"
    utc > "$out/utc-before.txt"
    if [ "$KIND" = bare ]; then spawn_bare "$out"; opened=$?; rc_line "$opened" > "$out/rc.txt"
    else open_argv; printf '%s\n' "${OPEN_ARGV[@]}" > "$out/argv.txt"
         "${OPEN_ARGV[@]}" > "$out/stdout.txt" 2> "$out/stderr.txt"; opened=$?; rc_line "$opened" > "$out/rc.txt"; fi
    if [ "$opened" = 0 ]; then
        for i in $(/usr/bin/seq 1 "$secs"); do
            code=$(http_code)
            echo "$(utc) port=$(port_now) http=$code servers=$(server_pids | /usr/bin/tr '\n' ',')" >> "$out/looks.txt"
            if [ "$code" = 200 ] && launcher_running && [ -n "$(server_pids)" ]; then served=1; break; fi
            /bin/sleep 1
        done
        /bin/sleep 3
    fi
    utc > "$out/utc-after.txt"
    ps_run "$out/ps-after.txt"; left=$?
    disarm "$out"
    if [ "$opened" != 0 ]; then V=FAIL; WHY="the open exited $opened"
    elif [ "$served" = 1 ]; then V=PASS; WHY="launcher and server serve HTTP 200"
    elif [ "$left" != 0 ]; then V=UNKNOWN; WHY="not serving within ${secs}s; ps failed"
    elif [ ! -s "$out/ps-after.txt" ]; then V=FAIL; WHY="nothing of $RUN runs ${secs}s after the open"
    else V=UNKNOWN; WHY="running but not serving within ${secs}s"; fi
    unobserved
    echo "$out"; verdict "$out" "$V" "served within ${secs}s from cold" "$WHY"
}

work_lines() { /usr/bin/grep -E '^(H_blob=|I_index_entry=|I_sha256=|W_hash_object=|U |U_count=)' "$1"; }   # H I W U of a bytes file
head_of() { /usr/bin/sed -n 's/^HEAD=\([0-9a-f]*\) .*/\1/p' "$1"; }
witness_scope() { echo "OWNER_NOTES.md in HEAD, index and worktree; $(/usr/bin/sed -n 's/^U_count=//p' "$1") untracked file(s)"; }
checkout_bytes() {   # H, I, W of OWNER_NOTES.md and every untracked file, as bytes (sha256) and git object ids: the selected
                     # witnesses (with HEAD, tree, branch and the index listing), never every tracked file. Status 0 and a
                     # last line READS=complete only when every read succeeded; "absent" is such a read that found nothing
                     # (clean geometry; U_count=0). A failed read is named in READS=incomplete and returns 3, never data.
    local f v line failed="" rc
    v=$(g rev-parse --verify -q 'HEAD^{commit}') || { v=unread; failed="$failed HEAD"; }; line="HEAD=$v"
    v=$(g rev-parse --verify -q 'HEAD^{tree}') || { v=unread; failed="$failed TREE"; }; line="$line TREE=$v"
    v=$(g symbolic-ref -q HEAD); rc=$?; [ "$rc" -le 1 ] || { v=unread; failed="$failed BRANCH"; }   # 1: detached
    echo "$line BRANCH=$v"
    v=$(set -o pipefail; g ls-files -s | /usr/bin/shasum -a 256 | /usr/bin/cut -c1-64) || { v=unread; failed="$failed INDEX"; }
    echo "INDEX_LS_FILES_SHA256=$v"
    if ! line=$(g ls-tree HEAD -- OWNER_NOTES.md); then echo "H_blob=unread H_sha256=unread"; failed="$failed H"
    elif [ -z "$line" ]; then echo "H_blob=absent H_sha256=absent"
    else line=$(echo "$line" | /usr/bin/awk '{print $3}')
         v=$(set -o pipefail; g cat-file blob "$line" | /usr/bin/shasum -a 256 | /usr/bin/cut -c1-64) || { v=unread; failed="$failed H"; }
         echo "H_blob=$line H_sha256=$v"; fi
    if ! line=$(g ls-files -s -- OWNER_NOTES.md); then echo "I_index_entry=unread"; echo "I_sha256=unread"; failed="$failed I"
    elif [ -z "$line" ]; then echo "I_index_entry=absent"; echo "I_sha256=absent"
    else echo "I_index_entry=$line"
         v=$(set -o pipefail; g cat-file blob :OWNER_NOTES.md | /usr/bin/shasum -a 256 | /usr/bin/cut -c1-64) || { v=unread; failed="$failed I"; }
         echo "I_sha256=$v"; fi
    f="$REPO/OWNER_NOTES.md"
    if [ -e "$f" ] || [ -L "$f" ]; then
        line=$(g hash-object -- "$f") || line=unread
        v=$(set -o pipefail; /usr/bin/shasum -a 256 < "$f" | /usr/bin/cut -c1-64) || v=unread
        rc=$(/usr/bin/stat -f %z "$f") || rc=unread
        case "$line $v $rc" in *unread*) failed="$failed W" ;; esac
        echo "W_hash_object=$line W_sha256=$v W_bytes=$rc"
    else echo "W_hash_object=absent W_sha256=absent W_bytes=absent"; fi
    echo "-- untracked (git ls-files --others --exclude-standard): sha256 bytes mode path"
    # The listing's own exit status rides behind its NUL-separated paths as a record no relative path can be. Its
    # stderr is read as well: Git warns about a directory it cannot open, omits it and still exits 0, so a listing
    # that wrote anything there is unread, never complete (the U lines still go to the record, through fd 3).
    { v=$( { (cd "$REPO" && { "$GIT" ls-files --others --exclude-standard -z; printf '/qb1-listing-exit=%s\0' "$?"; }) | {
        n=0; listed=; bad=0
        while IFS= read -r -d '' f; do
            case "$f" in   # "(pattern)": bash 3.2 reads a bare pattern's ")" inside $( ) as its end
                (/qb1-listing-exit=*) listed=${f#/qb1-listing-exit=} ;;
                (*) n=$((n + 1))
                    v=$(set -o pipefail; /usr/bin/shasum -a 256 < "$REPO/$f" | /usr/bin/cut -c1-64) || { v=unread; bad=1; }
                    line=$(/usr/bin/stat -f '%z %Sp' "$REPO/$f") || { line="unread unread"; bad=1; }
                    echo "U $v $line $f" ;;
            esac
        done
        echo "U_count=$n"; [ "$listed" = 0 ] || echo "U_listing=unread (exit ${listed:-none})"
        [ "$listed" = 0 ] && [ "$bad" = 0 ]; }; } 2>&1 >&3 ); rc=$?; } 3>&1   # the whole pipeline's stderr: Git's own
    [ -z "$v" ] || { echo "U_listing=unread (stderr: $(printf '%s' "$v" | /usr/bin/tr '\n' ' ' | /usr/bin/cut -c1-300))"; rc=1; }
    [ "$rc" = 0 ] || failed="$failed U"
    [ -z "$failed" ] || { echo "READS=incomplete failed=${failed# }"; return 3; }
    echo "READS=complete"
}

package_roots() {   # every package root this run's body or launcher may load from
    case "$KIND" in
        app|ext) /bin/ls -d "$VENV"/lib/python3.*/site-packages 2>/dev/null ;;
        site) /bin/ls -d "$DATA"/state/python-userbase/lib/python/site-packages "$APP"/Contents/Frameworks/python-standalone/lib/python3.*/site-packages 2>/dev/null ;;
        bare) /bin/ls -d "$PRIVATE"/lib/python3.*/site-packages "$EXTPY"/lib/python3.*/site-packages 2>/dev/null ;;
    esac
}

package_facts() {   # identities: the directory (device inode) and each dist-info's version, installer, origin, RECORD digest
    local root d
    for root in $(package_roots); do
        echo "== $root $(/usr/bin/stat -f 'device=%d inode=%i' "$root" 2>&1)"
        for d in "$root"/*.dist-info; do
            [ -d "$d" ] || continue
            echo "$(/usr/bin/basename "$d") installer=$(/bin/cat "$d/INSTALLER" 2>/dev/null | /usr/bin/head -1) direct_url=$([ -f "$d/direct_url.json" ] && echo yes || echo no) record_sha256=$(/usr/bin/shasum -a 256 "$d/RECORD" 2>/dev/null | /usr/bin/cut -c1-16)"
        done
    done
    echo "== environment directories beside the serving ones"
    /bin/ls -la -d "$R"/venv* "$DATA"/state/python-userbase* "$DATA"/state/python-private* 2>&1
}

obs() {
    run_root "$1"; local out f pids port
    out=$(step_dir "$2") || exit 2
    utc > "$out/utc.txt"
    pids_of_run > "$out/ps-run.txt" 2>&1
    /bin/ps -axww -o pid,ppid,pgid,user,lstart,command | /usr/bin/grep -i -E 'ouroboros|python|osascript|app_install|pip' \
        | /usr/bin/grep -v -F -e qb1.sh -e procwatch.py -e 'grep -i' > "$out/ps-ouroboros.txt" 2>&1
    pids=$(/usr/bin/awk '{print $1}' "$out/ps-run.txt")
    [ -n "$pids" ] && "$OBSPY" -I -S -B "$PW" snap "$out/env-snap.json" $pids > /dev/null 2> "$out/env-snap.stderr"
    /usr/sbin/lsof -nP -a -iTCP -sTCP:LISTEN -u "$(/usr/bin/id -u)" > "$out/listen.txt" 2>&1
    [ -n "$pids" ] && /usr/sbin/lsof -nP -a -i -p "$(echo $pids | /usr/bin/tr ' ' ',')" > "$out/sockets.txt" 2>&1
    { /bin/ls -la "$R/Applications"; for f in "$R"/Applications/*.app "$R"/Applications/.*.app; do [ -d "$f" ] && bundle_facts "$f"; done
      /bin/ls -laR "$R"/Applications/.Ouroboros-update-* 2>/dev/null | /usr/bin/head -40; } > "$out/bundles.txt" 2>&1
    for f in "$R"/Applications/.Ouroboros-update-*/install.jsonl; do [ -f "$f" ] && /bin/cp "$f" "$out/install.jsonl"; done
    if [ -d "$REPO/.git" ]; then
        { checkout_bytes; echo "-- status"; g status --porcelain=v2 --branch --untracked-files=all
          echo "-- diff (W vs I)"; g diff --stat; echo "-- diff --cached (I vs H)"; g diff --cached --stat; echo "-- log"; g log --oneline -6
          echo "-- refs"; g for-each-ref --format='%(refname) %(objectname)' refs/heads refs/ouroboros refs/remotes | /usr/bin/head -60
          echo "-- remotes"; g remote -v; echo "-- local config"; g config --local --list
          echo "-- stash"; g stash list; echo "-- worktrees"; g worktree list; } > "$out/checkout.txt" 2>&1
        checkout_bytes > "$out/bytes.txt" 2>&1
        { g diff; echo "== cached"; g diff --cached; } > "$out/checkout-diff.txt" 2>&1
        /bin/cat "$REPO/.git/ouroboros-managed.json" > "$out/managed-meta.json" 2>/dev/null
    fi
    package_facts > "$out/packages.txt" 2>&1
    { echo "KIND=$KIND BASEVER=$BASEVER VENV=$VENV BASE=$BASE requested_port=$SPORT"
      echo "SETTINGS=$SETTINGS"; /bin/ls -la "$SETTINGS" 2>&1; /usr/bin/shasum -a 256 "$SETTINGS" 2>&1
      echo "PIDF=$PIDF"; /bin/ls -la "$PIDF" 2>&1; echo "pid_file_content=$(/bin/cat "$PIDF" 2>&1 | /usr/bin/head -c 200)"
      echo "PORTF=$PORTF"; /bin/ls -la "$PORTF" 2>&1; echo "port_file_content=$(/bin/cat "$PORTF" 2>&1 | /usr/bin/head -c 50)"
      echo "-- default locations (expected absent in custom mode)"
      for f in "$DATA/settings.json" "$APPROOT/ouroboros.pid" "$DATA/state/server_port" "$HOMEDIR/Ouroboros" "$GH/Ouroboros/data"; do
          echo "$f: $( [ -e "$f" ] && echo present || echo absent)"; done
      echo "-- run root"; /bin/ls -la "$R" "$R/conf" "$R/locks" 2>&1
    } > "$out/config.txt" 2>&1
    [ -f "$SETTINGS" ] && /bin/cp "$SETTINGS" "$out/settings.json"
    [ -d "$VENV" ] && "$OBSPY" -I -S -B "$VL" facts "$VENV" "$APP" "$CANDSRC" "$out/venv-facts.json" > /dev/null 2> "$out/venv-facts.stderr"
    if [ -d "$DATA" ]; then
        /bin/ls -la "$DATA" "$DATA/state" "$DATA/logs" > "$out/data-ls.txt" 2>&1
        [ -d "$DATA/state/body_adoption" ] && /bin/cp -R "$DATA/state/body_adoption" "$out/body_adoption"
        /bin/mkdir -p "$out/records"; /bin/cp "$DATA"/state/private_copies/*.json "$out/records/" 2>/dev/null
        for f in update_activation.json process_ledger.jsonl server_port server_process.json owned_processes.json \
                 installation_scope.json install.json server_port.bindings.json; do
            [ -f "$DATA/state/$f" ] && /bin/cp "$DATA/state/$f" "$out/"; done
        /bin/mkdir -p "$out/logs"
        # This is a point-in-time observation; a live source may continue growing after the copy.
        for f in "$DATA"/logs/*; do [ -f "$f" ] && /bin/cp "$f" "$out/logs/$(/usr/bin/basename "$f")"; done
        (cd "$out/logs" && /usr/bin/shasum -a 256 ./*) > "$out/logs-copied-sha256.txt" 2>&1
    fi
    port=$(port_now)
    [ -n "$port" ] && /usr/bin/curl -sS -m 60 -o "$out/update-status.json" "http://127.0.0.1:$port/api/update/status" 2> "$out/update-status.stderr"
    /bin/ls -la "$APPROOT" > "$out/approot-ls.txt" 2>&1
    { for f in "$HOMEDIR/Library/WebKit/com.ouroboros.agent" "$HOMEDIR/Library/Preferences/com.ouroboros.agent.plist" \
               "$GH/Library/WebKit/com.ouroboros.agent" "$GH/Ouroboros"; do
          echo "== $f"; /usr/bin/find "$f" -maxdepth 6 -exec /bin/ls -ld -T {} + 2>&1 | /usr/bin/head -80; done; } > "$out/profile.txt" 2>&1
    profile_store > "$out/profile-store.txt" 2>&1
    echo "$out"
}

profile_store() {   # the private WebKit store of this run's home: file canary and every localStorage database (copies)
    local wk="$HOMEDIR/Library/WebKit/com.ouroboros.agent" db tmp f
    echo "store=$wk"
    echo "file_canary: $(/usr/bin/shasum -a 256 "$wk/qb1-profile-canary.txt" 2>&1)"
    /usr/bin/find "$wk" \( -name 'localstorage.sqlite3' -o -name '*.localstorage' \) 2>/dev/null | while IFS= read -r db; do
        echo "== $db"; /bin/ls -la -T "$db" "$db-wal" "$db-shm" 2>&1
        tmp=$(/usr/bin/mktemp -d "$TMPD/qb1-ls.XXXXXX") || continue
        /bin/cp "$db" "$tmp/db.sqlite3"; for f in wal shm; do [ -f "$db-$f" ] && /bin/cp "$db-$f" "$tmp/db.sqlite3-$f"; done
        /usr/bin/sqlite3 "$tmp/db.sqlite3" "SELECT typeof(key), key, length(value), CASE WHEN length(value) <= 96 THEN hex(value) ELSE 'long' END FROM ItemTable;" 2>&1
        /bin/rm -rf "$tmp"
    done
}

lscanary() {
    run_root "$1"; local out wk db hexval n=0 bad=0 r
    out=$(step_dir "$2") || exit 2
    cold_state "$out"
    case $? in 0) ;; 1) die "the app is running; quit it first ($out/cold-*.txt)" ;;
               *) die "whether the app runs is not established ($out/cold-*.txt)" ;; esac
    wk="$HOMEDIR/Library/WebKit/com.ouroboros.agent"
    [ -d "$wk" ] || { echo "no private WebKit store at $wk" > "$out/unsupported.txt"; echo "$out"
                      verdict "$out" UNKNOWN "canaries planted" "no private WebKit store at $wk"; return; }
    printf '%s private profile canary\nwritten %s while the app was quit\n' "$RUN" "$(utc)" > "$wk/qb1-profile-canary.txt"
    /usr/bin/shasum -a 256 "$wk/qb1-profile-canary.txt" > "$out/file-canary.txt"
    hexval=$("$OBSPY" -I -S -B -c 'import sys; print((sys.argv[1] + "-ls-canary-1").encode("utf-16-le").hex())' "$RUN")
    profile_store > "$out/before.txt" 2>&1
    while IFS= read -r db; do
        n=$((n + 1))
        /usr/bin/sqlite3 "$db" "INSERT INTO ItemTable(key, value) VALUES('$RUN-ls-canary', X'$hexval');" > "$out/insert-$n.txt" 2>&1
        r=$?; rc_line "$r" >> "$out/insert-$n.txt"; echo "$db" >> "$out/insert-$n.txt"; [ "$r" = 0 ] || bad=$((bad + 1))
    done < <(/usr/bin/find "$wk" -name 'localstorage.sqlite3' 2>/dev/null)
    echo "databases=$n failed=$bad value_hex=$hexval" > "$out/summary.txt"
    profile_store > "$out/after.txt" 2>&1
    echo "$out"
    if [ "$bad" != 0 ]; then verdict "$out" FAIL "canaries planted" "$bad of $n inserts failed"
    elif [ "$n" = 0 ]; then verdict "$out" UNKNOWN "canaries planted" "file canary only: no localStorage database"
    else verdict "$out" PASS "canaries planted" "file canary and $n localStorage database(s)"; fi
}

quit_app() {   # the app's own Quit (Apple Event by exact path); b1p: SIGINT to the recorded launcher pid. PASS: within SECS
               # no process of this run and no listener on its ports (cold_state); FAIL: still running or listening
    run_root "$1"; local out secs="${3:-240}" t0 i asked=0 V WHY
    out=$(step_dir "$2") || exit 2
    pids_of_run > "$out/ps-before.txt" 2>&1
    utc > "$out/utc-before.txt"; t0=$(/bin/date +%s)
    if [ "$KIND" = bare ]; then
        # Kernel birth survives the launcher's exec; PID reuse or unknown identity never selects a process.
        "$OBSPY" -I -S -B "$Q/bin/quit_bare.py" "$LPIDF.current.birth.json" "$SEED" "$REPO" > "$out/signal.txt" 2>&1
        local signal_rc=$?; rc_line "$signal_rc" >> "$out/signal.txt"
        [ "$signal_rc" = 0 ] || die "the recorded launcher identity is not proven; nothing was signalled ($out/signal.txt)"
    else
        /usr/bin/osascript -e "tell application \"$APP\" to quit" > "$out/osascript.txt" 2>&1; asked=$?; rc_line "$asked" >> "$out/osascript.txt"
    fi
    for i in $(/usr/bin/seq 1 "$secs"); do
        pids_of_run > "$out/ps-now.txt" 2>&1
        /usr/bin/grep -q -F -e "$APP/Contents/MacOS/" -e "$REPO/" -e "$SEED/launcher.py" "$out/ps-now.txt" || break
        /bin/sleep 1
    done
    echo "waited_seconds=$(( $(/bin/date +%s) - t0 ))" >> "$out/result.txt"
    /bin/mv "$out/ps-now.txt" "$out/ps-after.txt"; utc > "$out/utc-after.txt"
    cold_state "$out"
    case $? in 0) V=PASS; WHY="no process of $RUN and no listener on its ports" ;;
               1) V=FAIL; WHY="a process of $RUN or a listener on its ports remains after ${secs}s (cold-*.txt)" ;;
               *) V=UNKNOWN; WHY="absence not established (cold-*.txt)" ;; esac
    [ "$asked" = 0 ] || WHY="$WHY; the quit request exited $asked"
    echo "$out"; verdict "$out" "$V" "quit to a cold state within ${secs}s" "$WHY"
}

compose() {   # the owner's local work: H, I, W and U (b1a/b1e/b1p at CAND, DISCLOSED manual checkout for b1a/b1e;
              # b1s on its stock 7.8.0 checkout, which the stock updater then delivers)
    # PASS: the base (CAND; b1s: v7.8.0's stock commit) under exactly one owner commit H, I and W on OWNER_NOTES.md, U present
    run_root "$1"; local out base want r f fails=0 why="" footprint
    out=$(step_dir compose) || exit 2
    [ -d "$REPO/.git" ] || die "no managed checkout at $REPO"
    cold_state "$out"
    case $? in 0) ;; 1) die "the app is running; quit it first ($out/cold-*.txt)" ;;
               *) die "whether the app runs is not established ($out/cold-*.txt)" ;; esac
    {
        echo "-- before"; g status --porcelain=v2 --branch; g log --oneline -2; g remote -v
        g fetch managed; r=$?; rc_line "$r"; [ "$r" = 0 ] || fails=$((fails + 1))
        case "$KIND" in
            app|ext) g checkout -B "$(g symbolic-ref -q --short HEAD)" "$CAND"; r=$?; rc_line "$r"; [ "$r" = 0 ] || fails=$((fails + 1))
                     echo "candidate_tree=$(g rev-parse 'HEAD^{tree}') expected=$CTREE" ;;
            bare)    echo "candidate_tree=$(g rev-parse 'HEAD^{tree}') expected=$CTREE (seed bootstrap)" ;;
            site)    echo "stock_tree=$(g rev-parse 'HEAD^{tree}') (v7.8.0's own checkout)" ;;
        esac
        base=$(g rev-parse HEAD)
        printf '# Owner note (%s local H)\nThis line is a local commit of the owner.\n' "$RUN" > "$REPO/OWNER_NOTES.md"
        g add OWNER_NOTES.md && g -c user.name=Owner -c user.email=owner@lab.invalid commit -q -m "owner: local note ($RUN H)"
        r=$?; rc_line "$r"; [ "$r" = 0 ] || fails=$((fails + 1))
        printf 'Staged line (%s I).\n' "$RUN" >> "$REPO/OWNER_NOTES.md"; g add OWNER_NOTES.md; r=$?; rc_line "$r"; [ "$r" = 0 ] || fails=$((fails + 1))
        printf 'Unstaged line (%s W).\n' "$RUN" >> "$REPO/OWNER_NOTES.md"
        printf 'untracked owner scratch (%s U)\n' "$RUN" > "$REPO/owner-scratch.txt"
        /bin/mkdir -p "$REPO/owner-untracked/nested"
        /bin/dd if=/dev/urandom of="$REPO/owner-untracked/nested/blob.bin" bs=4096 count=16 2>/dev/null; r=$?; rc_line "$r"
        [ "$r" = 0 ] || fails=$((fails + 1))
        { echo '"""The owner'"'"'s own local helper (untracked): it runs with the packages of the interpreter it is run with."""'
          echo ''
          echo ''
          echo 'def owner_local_function():'
          echo '    import starlette'
          echo '    import yaml'
          echo ''
          echo "    return {\"token\": \"$RUN-owner-local-function-1\", \"yaml\": yaml.__version__, \"yaml_file\": yaml.__file__,"
          echo '            "starlette": starlette.__version__}'
        } > "$REPO/owner-untracked/owner_local_function.py"
        echo "-- after"; echo "HEAD=$(g rev-parse HEAD) TREE=$(g rev-parse 'HEAD^{tree}')"
        g status --porcelain=v2 --branch --untracked-files=all; g log --oneline -3
    } > "$out/compose.txt" 2>&1
    checkout_bytes > "$out/bytes.txt" 2>&1; footprint=$?
    /bin/cp "$REPO/owner-untracked/owner_local_function.py" "$out/owner_local_function.py"
    case "$KIND" in site) want=$(official_tag_of v7.8.0) ;; *) want="$CAND" ;; esac
    [ "$base" = "$want" ] || why="$why base ${base:--} is not $want;"
    [ "$(g rev-parse -q --verify 'HEAD^')" = "$base" ] || why="$why H is not one commit on the base;"
    [ "$(g diff --cached --name-only)" = OWNER_NOTES.md ] || why="$why I is not exactly OWNER_NOTES.md;"
    [ "$(g diff --name-only)" = OWNER_NOTES.md ] || why="$why W is not exactly OWNER_NOTES.md;"
    for f in owner-scratch.txt owner-untracked/nested/blob.bin owner-untracked/owner_local_function.py; do
        [ "$(g ls-files --others --exclude-standard -- "$f")" = "$f" ] || why="$why U lacks $f;"
    done
    [ "$fails" = 0 ] || why="$why $fails composition step(s) failed (compose.txt);"
    echo "$out"
    if [ -n "$why" ]; then verdict "$out" FAIL "H I W U on the base" "$why"
    elif [ "$footprint" != 0 ]; then verdict "$out" UNKNOWN "H I W U on the base" "the footprint record is incomplete (bytes.txt READS)"
    else verdict "$out" PASS "H I W U on the base" "base=$base; recorded: $(witness_scope "$out/bytes.txt")"; fi
}

localfn() {   # localfn RUN STEP [YAML_VERSION] [venv|private|base]: the owner's untracked local function on the interpreter
              # the live server runs (its exec argv[0], read from the kernel), with that interpreter's packages. PASS: the
              # helper's token, yaml (YAML_VERSION when given) loaded from inside that interpreter's prefix (b1s: or the
              # user base the server's own observed environment names, with which the helper runs) and, when given, that
              # serving prefix. b1p also runs the external base, kept apart: it shows the base unchanged and never stands
              # in for the body's result.
    run_root "$1"; local out pid n py want_yaml="${3:-}" want_prefix="" userbase="" body_rc base_rc=- checked V WHY penv line
    case "${4:-}" in '') ;; venv) want_prefix="$VENV" ;; private) want_prefix="$PRIVATE" ;; base) want_prefix="${BASE%/bin/*}" ;;
                     *) die "ENV is venv, private or base" ;; esac
    out=$(step_dir "$2") || exit 2
    /usr/bin/shasum -a 256 "$REPO/owner-untracked/owner_local_function.py" > "$out/file-sha256.txt" 2>&1
    local code='import json, os, runpy, sys
ns = runpy.run_path(sys.argv[1])
print(json.dumps({"result": ns["owner_local_function"](), "executable": sys.executable,
                  "real": os.path.realpath(sys.executable), "version": sys.version.split()[0],
                  "prefix": sys.prefix, "base_prefix": sys.base_prefix}, sort_keys=True))'
    # b1s: the serving process's own package-loading environment as observed (server-snap.json): HOME, VIRTUAL_ENV,
    # __PYVENV_LAUNCHER__ and every PYTHON* name it holds, each recorded in plain; otherwise not established
    local penv_py='import json, re, sys
try:
    with open(sys.argv[1]) as handle:
        env = json.load(handle)[0]["env"]
    keys, plain = env["keys"], env["plain"]
except (OSError, ValueError, LookupError, TypeError) as exc:
    sys.exit("no environment record (%s)" % type(exc).__name__)
names = sorted(k for k in keys if k in ("HOME", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__") or k.startswith("PYTHON"))
bad = [k for k in names if k not in plain or "\n" in plain[k] or re.search(r"\.\.\.\[\d+ chars\]$", plain[k])]
if bad:
    sys.exit("not recorded in plain: %s" % " ".join(bad))
for k in names:
    print("%s=%s" % (k, plain[k]))'
    server_rows > "$out/server.txt"; n=$(/usr/bin/grep -c . "$out/server.txt")
    if [ "$n" != 1 ]; then
        echo "$out"; verdict "$out" UNKNOWN "the helper on the serving interpreter" "$n serving server processes (server.txt)"; return
    fi
    pid=$(/usr/bin/awk '{print $1}' "$out/server.txt")
    "$OBSPY" -I -S -B "$PW" snap "$out/server-snap.json" "$pid" > /dev/null 2> "$out/server-snap.stderr"
    py=$("$OBSPY" -I -S -B -c 'import json, sys
rows = json.load(open(sys.argv[1]))
print(((rows[0].get("argv") or [""]) if rows else [""])[0])' "$out/server-snap.json" 2>/dev/null)
    echo "$py" > "$out/interpreter.txt"
    case "$py" in /*) ;; *) echo "$out"; verdict "$out" UNKNOWN "the helper on the serving interpreter" \
                               "the serving interpreter of pid $pid is not readable (server-snap.json)"; return ;; esac
    case "$KIND" in
        site) if ! "$OBSPY" -I -S -B -c "$penv_py" "$out/server-snap.json" > "$out/package-env.txt" 2> "$out/package-env.stderr"
              then echo "$out"; verdict "$out" UNKNOWN "the helper on the serving interpreter" \
                       "the serving package environment of pid $pid is not established (package-env.stderr)"; return
              fi
              userbase=$(/usr/bin/sed -n 's/^PYTHONUSERBASE=//p' "$out/package-env.txt")
              penv=(); while IFS= read -r line; do penv+=("$line"); done < "$out/package-env.txt"
              /usr/bin/env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 "${penv[@]+"${penv[@]}"}" \
                  "$py" -c "$code" "$REPO/owner-untracked/owner_local_function.py" > "$out/localfn.json" 2> "$out/stderr.txt" ;;
        *) "$py" -I -B -c "$code" "$REPO/owner-untracked/owner_local_function.py" > "$out/localfn.json" 2> "$out/stderr.txt" ;;
    esac
    body_rc=$?; rc_line "$body_rc" > "$out/rc.txt"
    if [ "$KIND" = bare ]; then
        "$BASE" -I -B -c "$code" "$REPO/owner-untracked/owner_local_function.py" > "$out/localfn-bare-interpreter.json" 2> "$out/stderr-base.txt"
        base_rc=$?; rc_line "$base_rc" > "$out/rc-base.txt"
    fi
    WHY=$("$OBSPY" -I -S -B -c 'import json, os, sys
path, token, yaml_want, prefix_want, userbase = sys.argv[1:6]
try:
    fact = json.load(open(path))
    result = fact["result"]
except (OSError, ValueError, KeyError, TypeError) as exc:
    print("no helper result (%s)" % type(exc).__name__)
    sys.exit(1)
real = lambda p: os.path.realpath(p) if p else ""
prefix = real(fact.get("prefix"))
roots = [root for root in (prefix, real(userbase)) if root]
problems = []
if result.get("token") != token:
    problems.append("token %r, not %r" % (result.get("token"), token))
if yaml_want and result.get("yaml") != yaml_want:
    problems.append("yaml %s, not %s" % (result.get("yaml"), yaml_want))
if not any(real(result.get("yaml_file")).startswith(root + os.sep) for root in roots):
    problems.append("yaml loaded from %s, outside %s" % (result.get("yaml_file"), roots))
if prefix_want and prefix != real(prefix_want):
    problems.append("serving prefix %s, not %s" % (prefix, real(prefix_want)))
print("; ".join(problems) if problems else "token ok, yaml %s from %s, prefix %s" % (
    result.get("yaml"), result.get("yaml_file"), prefix))
sys.exit(1 if problems else 0)' "$out/localfn.json" "$RUN-owner-local-function-1" "$want_yaml" "$want_prefix" "$userbase" 2>&1)
    checked=$?
    if [ "$body_rc" != 0 ]; then V=FAIL; WHY="the helper exited $body_rc on the serving interpreter $py (stderr.txt)"
    elif [ "$checked" != 0 ]; then V=FAIL; WHY="on $py: $WHY"
    else V=PASS; WHY="on $py: $WHY"; fi
    [ "$KIND" = bare ] && WHY="$WHY; external base exit $base_rc (localfn-bare-interpreter.json, recorded only)"
    echo "$out"; verdict "$out" "$V" "the helper on the serving interpreter${want_yaml:+ with yaml $want_yaml}" "$WHY"
}

api() {
    run_root "$1"; local out port method="$3" path="$4" body="${5:-}"
    out=$(step_dir "$2") || exit 2
    port=$(port_now); [ -n "$port" ] || die "no port in $PORTF"
    echo "$method http://127.0.0.1:$port$path body=$body requested_port=$SPORT" > "$out/request.txt"
    if [ -n "$body" ]; then
        /bin/cp "$body" "$out/body.json"
        /usr/bin/curl -sS -m 300 -X "$method" -H 'Content-Type: application/json' --data-binary "@$body" \
            -o "$out/response.json" -w '%{http_code}\n' "http://127.0.0.1:$port$path" > "$out/status.txt" 2> "$out/stderr.txt"
    else
        /usr/bin/curl -sS -m 300 -X "$method" -o "$out/response.json" -w '%{http_code}\n' \
            "http://127.0.0.1:$port$path" > "$out/status.txt" 2> "$out/stderr.txt"
    fi
    local sent=$? code; rc_line "$sent" >> "$out/status.txt"; code=$(/usr/bin/head -1 "$out/status.txt")
    echo "$out"
    if [ "$sent" != 0 ]; then verdict "$out" UNKNOWN "HTTP 2xx" "curl exit $sent, http=${code:--}"
    else case "$code" in 2??) verdict "$out" PASS "HTTP 2xx" "http=$code" ;; *) verdict "$out" FAIL "HTTP 2xx" "http=$code" ;; esac; fi
}

status() {
    run_root "$1"; local out
    out=$(step_dir "$2") || exit 2
    get_status "$out/response.json" > "$out/status.txt"
    echo "state=$(jget "$out/response.json" candidate.state) candidate=$(jget "$out/response.json" candidate.candidate_id)" >> "$out/status.txt"
    echo "$out"
}

settings() {   # the owner's Settings save of one key (the Settings page's POST /api/settings), then read back
    # PASS: the save answered 2xx and the read-back value is the one sent (letter case aside: JSON booleans read back)
    run_root "$1"; local out port posted read got code lower
    out=$(step_dir "$2") || exit 2
    port=$(port_now); [ -n "$port" ] || die "no port in $PORTF"
    "$OBSPY" -I -S -B -c 'import json, sys; print(json.dumps({sys.argv[1]: sys.argv[2]}))' "$3" "$4" > "$out/body.json"
    /usr/bin/curl -sS -m 120 -X POST -H 'Content-Type: application/json' --data-binary "@$out/body.json" \
        -o "$out/post.json" -w '%{http_code}\n' "http://127.0.0.1:$port/api/settings" > "$out/status.txt" 2> "$out/stderr.txt"
    posted=$?
    /usr/bin/curl -sS -m 60 -o "$out/get.json" -w '%{http_code}\n' "http://127.0.0.1:$port/api/settings" >> "$out/status.txt" 2>> "$out/stderr.txt"
    read=$?
    got=$(jget "$out/get.json" "$3"); echo "readback $3=$got" >> "$out/status.txt"
    code=$(/usr/bin/head -1 "$out/status.txt")
    lower() { printf '%s' "$1" | /usr/bin/tr '[:upper:]' '[:lower:]'; }
    echo "$out"
    if [ "$posted" != 0 ] || [ "$read" != 0 ]; then verdict "$out" UNKNOWN "saved and read back" "curl exits $posted/$read"
    else case "$code" in
        2??) if [ "$(lower "$got")" = "$(lower "$4")" ]; then verdict "$out" PASS "saved and read back" "$3=$got"
             else verdict "$out" FAIL "saved and read back" "read back ${got:-nothing}, not $4"; fi ;;
        *) verdict "$out" FAIL "saved and read back" "POST http=$code" ;; esac; fi
}

prepare() {   # prepare RUN STEP [SECS] [EXPECT] [TIER]: the live server's own preparation, followed through /api/update/status
              # until a candidate other than the one before the request settles. EXPECT: ready (default); failed=REASON
              # (the named negative: failed, its reason starting with REASON); settled (ready with passing checks, or a
              # named non-ready result; on the bound PUB_O1 when one is bound). TIER full|selected: a ready candidate's
              # selection must be that tier (full on the native local cutovers rev6 names); omitted, the tier is recorded
              # exactly as selected. Every expectation also needs the serving server and the owner's H/I/W/U unchanged.
              # outcome.txt is the observed product outcome; readiness.json and evidence.txt are its evidence receipt;
              # verdict.txt is PASS only for a PASS outcome whose evidence is complete (else UNKNOWN; a FAIL stays FAIL).
    run_root "$1"; local out secs="${3:-5400}" expect="${4:-ready}" tier="${5:-}" port t0 i state= sp id refusal http sent \
        settled=0 before_id before_refusal before_state reason ok got_tier af ps_after bytes_after evidence rc V WHY
    case "$expect" in ready|settled|failed=?*) ;; *) die "EXPECT is ready, settled or failed=REASON" ;; esac
    case "$tier" in ''|full|selected) ;; *) die "TIER is full or selected" ;; esac
    out=$(step_dir "$2") || exit 2
    port=$(port_now); [ -n "$port" ] || die "no port in $PORTF"
    echo "$expect${tier:+ tier=$tier}" > "$out/expect.txt"
    get_status "$out/status-before.json" > "$out/status-before-http.txt"
    before_id=$(jget "$out/status-before.json" candidate.candidate_id)
    before_refusal=$(jget "$out/status-before.json" candidate.refusal.at)
    ps_run "$out/ps-before.txt" || die "the process table is unreadable before the request ($out)"
    /usr/bin/grep -F "$REPO/server.py" "$out/ps-before.txt" > "$out/server-before.txt"
    checkout_bytes > "$out/bytes-before.txt" 2>&1 || die "the owner's H/I/W/U are unreadable before the request ($out/bytes-before.txt)"
    # Only a readable, settled status tells this request's candidate from one that was already there (the 202 answer
    # names none): no candidate, or a settled one with its id. Anything else is refused before the request is sent.
    before_state=$(jget "$out/status-before.json" candidate.state)
    case "$(/bin/cat "$out/status-before-http.txt" 2>/dev/null):$before_state" in
        200:none) [ -z "$before_id" ] ;;
        200:ready|200:failed|200:needs_decision|200:needs_adaptation|200:discarded|200:superseded|200:stale|200:paused_budget)
            [ -n "$before_id" ] ;;
        *) false ;;
    esac || die "the update status before the request is unreadable or not settled (http $(/bin/cat "$out/status-before-http.txt" 2>/dev/null), state ${before_state:-none read}, candidate ${before_id:-none}): nothing sent ($out)"
    arm "$out" "$((secs + 60))"
    echo '{"strategy": "prepare"}' > "$out/body.json"
    utc > "$out/utc-request.txt"
    /usr/bin/curl -sS -m 120 -X POST -H 'Content-Type: application/json' --data-binary "@$out/body.json" \
        -o "$out/response.json" -w '%{http_code}\n' "http://127.0.0.1:$port/api/update/apply" > "$out/status.txt" 2> "$out/stderr.txt"
    sent=$?; rc_line "$sent" >> "$out/status.txt"; http=$(/usr/bin/head -1 "$out/status.txt")
    t0=$(/bin/date +%s); KEPT=0; /bin/mkdir -p "$out/kept"
    case "$http" in 202|000|'') ;; *) settled=refused; get_status "$out/status-now.json" > /dev/null ;; esac
    if [ "$settled" = 0 ]; then
        for i in $(/usr/bin/seq 1 "$((secs / 5))"); do
            get_status "$out/status-now.json" > /dev/null
            state=$(jget "$out/status-now.json" candidate.state); id=$(jget "$out/status-now.json" candidate.candidate_id)
            refusal=$(jget "$out/status-now.json" candidate.refusal.at)
            keep_if_changed "$out/status-now.json" "$out/kept/status.json"
            sp=$(server_pids | /usr/bin/tr '\n' ',')
            echo "$(utc) +$(( $(/bin/date +%s) - t0 ))s state=${state:--} candidate=${id:--} servers=$sp children=$(pids_of_run | /usr/bin/grep -v -F -e "$REPO/server.py" -e "launcher.py" | /usr/bin/grep -c -E 'pip|pytest|git') http=$(http_code)" >> "$out/timeline.txt"
            if [ -n "$id" ] && [ "$id" != "$before_id" ]; then
                case "$state" in ready|failed|needs_decision|needs_adaptation|discarded|superseded|stale|paused_budget) settled=1; break ;; esac
            elif [ "$state" = none ] && [ -n "$refusal" ] && [ "$refusal" != "$before_refusal" ]; then
                state=refused; settled=1; break   # refused before any workspace: a named outcome of this request
            fi
            /bin/sleep 5
        done
    fi
    /bin/cp "$out/status-now.json" "$out/status-final.json" 2>/dev/null
    reason=$(jget "$out/status-final.json" candidate.reason); ok=$(jget "$out/status-final.json" candidate.checks.ok)
    got_tier=$(jget "$out/status-final.json" candidate.checks.tests.tier)
    af=$(jget "$out/status-final.json" candidate.checks.tests.already_failing); [ "$af" != "[]" ] || af=
    [ "$state" = refused ] && reason="$(jget "$out/status-final.json" candidate.refusal.code): $(jget "$out/status-final.json" candidate.refusal.text)"
    echo "final_state=${state:--} settled=$settled http=${http:--} tier=${got_tier:--} reason=${reason:--} seconds=$(( $(/bin/date +%s) - t0 ))" > "$out/result.txt"
    ps_run "$out/ps-after.txt"; ps_after=$?
    /usr/bin/grep -F "$REPO/server.py" "$out/ps-after.txt" > "$out/server-after.txt" 2>/dev/null
    checkout_bytes > "$out/bytes-after.txt" 2>&1; bytes_after=$?
    disarm "$out"
    if [ "$settled" = refused ]; then V=FAIL; WHY="the preparation was not admitted: http=$http $(jget "$out/response.json" reason)"
    elif [ "$settled" != 1 ]; then V=UNKNOWN; WHY="no new candidate settled within ${secs}s (state=${state:--}, curl exit $sent)"
    elif [ ! -s "$out/server-before.txt" ]; then V=UNKNOWN; WHY="no serving server was identified before the request"
    elif [ "$ps_after" != 0 ]; then V=UNKNOWN; WHY="the process table was unreadable after the preparation"
    elif ! /usr/bin/cmp -s "$out/server-before.txt" "$out/server-after.txt"; then V=FAIL; WHY="the serving server changed (server-*.txt)"
    elif [ "$bytes_after" != 0 ]; then V=UNKNOWN; WHY="the owner's H/I/W/U were not read completely after the preparation (bytes-after.txt)"
    elif ! /usr/bin/cmp -s "$out/bytes-before.txt" "$out/bytes-after.txt"; then V=FAIL; WHY="the owner's H/I/W/U changed (bytes-*.txt)"
    else
        case "$expect" in
            ready) if [ "$state" != ready ] || [ "$ok" != True ]; then V=FAIL; WHY="$state, checks.ok=${ok:--}, reason=${reason:--}"
                   elif [ -n "$tier" ] && [ "$got_tier" != "$tier" ]; then V=FAIL; WHY="ready on tier ${got_tier:--}, not the required $tier"
                   else V=PASS; WHY="ready, checks ok, tier ${got_tier:--}${af:+; disclosed failures, already failing on W: $af}"; fi ;;
            failed=*) case "$reason" in
                          "${expect#failed=}"*) if [ "$state" = failed ]; then V=PASS; WHY="failed as expected: $reason"
                                                else V=FAIL; WHY="$state: $reason"; fi ;;
                          *) V=FAIL; WHY="$state, reason=${reason:--}; expected ${expect}" ;; esac ;;
            settled) if [ "$state" = ready ] && [ "$ok" != True ]; then V=FAIL; WHY="ready without passing checks (checks.ok=${ok:--})"
                     elif [ "$state" = ready ] && [ -n "$tier" ] && [ "$got_tier" != "$tier" ]; then
                         V=FAIL; WHY="ready on tier ${got_tier:--}, not the required $tier"
                     elif [ "$state" != ready ] && [ -z "$reason" ]; then V=FAIL; WHY="$state without a named reason"
                     elif [ -n "$PUB_O1" ] && [ "$(jget "$out/status-final.json" candidate.target_sha)" != "$PUB_O1" ]; then
                         V=FAIL; WHY="the target is $(jget "$out/status-final.json" candidate.target_sha), not the bound PUB_O1"
                     elif [ "$state" = ready ]; then V=PASS; WHY="settled ready${got_tier:+, tier $got_tier}${af:+; disclosed failures, already failing on W: $af}"
                     else V=PASS; WHY="settled $state${reason:+: $reason}, a named non-ready result, not a readiness qualification"; fi ;;
        esac
        WHY="$WHY; unchanged witnesses: HEAD, tree, branch, index listing, $(witness_scope "$out/bytes-after.txt")"
        [ "$http" = 202 ] || WHY="$WHY; the admission answer was lost (http ${http:--}): this candidate is told apart only by the settled status before the request (${before_id:-no candidate})"
    fi
    unobserved
    echo "outcome=$V expected=$expect${tier:+ tier=$tier} detail=$WHY" > "$out/outcome.txt"
    readiness "$out" "$V" "$state"; rc=$?
    evidence=$(/bin/cat "$out/evidence.txt" 2>/dev/null)
    [ -n "$evidence" ] || evidence="evidence=incomplete missing=the receipt itself (readiness exit $rc)"
    case "$V:$evidence" in
        PASS:evidence=complete) ;;
        PASS:*) V=UNKNOWN; WHY="the observed outcome is PASS ($WHY), but its evidence is incomplete: ${evidence#evidence=}" ;;
        *) WHY="$WHY; $evidence" ;;
    esac
    verdict "$out" "$V" "$expect${tier:+ tier=$tier}" "$WHY"; rc=$?
    echo "$out"; return "$rc"
}

readiness() {   # readiness OUT OUTCOME STATE: prepare's evidence receipt, read from the producer's own references
                # (supervisor/update_checks.py, bound at its final freeze: sha256 e9890caf..., 4e53a124 tree 08eab970).
                # A run's role is the checks result that names it, tests.new / tests.baseline / tests.target_run .evidence:
                # that run's junit-* folder, never a sampled process or the latest file. Its run.json, parallel.xml and
                # serial.xml and both lanes' full outputs are copied before the candidate's scratch is retired, and checked
                # against the exact candidate record (composition_commit, layers_base.k2, target.target_sha) and each
                # other: both lanes merged as one run by node id, an empty lane only as the producer allows it (exit 5, no
                # cases). A run whose lanes read a pytest argument file (run.json targets_file, argv @FILE) needs that
                # file in its own folder, read by every lane and no other, its lines exactly run.json targets; it is copied
                # too. evidence.txt says complete or names what the observed outcome lacks; readiness.json also says
                # whether this is a readiness qualification (a PASS ready backed by that evidence), and green only for a
                # qualification without any failure (new run, already failing, blocking, target failing, any run read).
    local lock
    lock=$(g show HEAD:requirements-runtime.lock 2>/dev/null | /usr/bin/shasum -a 256 | /usr/bin/cut -c1-64)
    /bin/mkdir -p "$1/records"; /bin/cp "$DATA"/state/private_copies/*.json "$1/records/" 2>/dev/null
    "$OBSPY" -I -S -B -c 'import hashlib, json, os, shutil, sys
from xml.etree import ElementTree
out, outcome, state, cand, ctree, o0, o1, lock, server_mark, records_dir = sys.argv[1:11]
PRODUCER = "e9890cafbf182c0d9de6f78f75b25900ccfd01f95b88b183deba7655c4e91637"
CRASH = "crashed while running"
LANES = ("parallel", "serial")
TARGETS = "targets.txt"   # the argument file of the lanes (run.json targets_file): pytest @FILE, argparse one argument per line
NAMES = ("run.json", TARGETS) + tuple("%s.%s" % (lane, kind) for lane in LANES for kind in ("xml", "stdout.txt", "stderr.txt"))
def text(name):
    try:
        with open(os.path.join(out, name), errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return ""
def digest(path):
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None
def rows(name):
    found = []
    for line in text(name).splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            found.append(row)
    return found
def nodeid(case):   # update_checks._nodeid
    path, classname = str(case.get("file") or ""), str(case.get("classname") or "")
    module = path[:-3].replace("/", ".") if path.endswith(".py") else ""
    if not path:
        path, classes = classname.replace(".", "/") + ".py", []
    elif module and classname.startswith(module + "."):
        classes = classname[len(module) + 1:].split(".")
    else:
        classes = []
    return "::".join([path, *classes, str(case.get("name") or "")])
def cases(report):   # update_checks._cases: {node id: failure signature, or "" for a pass or skip}
    found = {}
    for case in ElementTree.parse(report).getroot().iter("testcase"):
        failure = case.find("failure") if case.find("failure") is not None else case.find("error")
        found[nodeid(case)] = "" if failure is None else "%s:%s:%s" % (
            failure.tag, failure.get("type") or "", (failure.get("message") or "").strip().splitlines()[:1])
    return found
part = lambda value, key: value.get(key) if isinstance(value.get(key), dict) else {}
missing = []
try:
    status = json.loads(text("status-final.json") or "null").get("candidate")
except (ValueError, AttributeError):
    status = None
if not isinstance(status, dict):
    status, missing = {}, ["the candidate in status-final.json"]
checks = part(status, "checks"); tests = part(checks, "tests")
records, exact, folder = [], [], os.path.join(out, "records")
for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
    try:
        with open(os.path.join(folder, name), "rb") as handle:
            blob = handle.read()
        record = json.loads(blob)
    except (OSError, ValueError):
        continue
    if isinstance(record, dict) and record.get("candidate_id") == status.get("candidate_id"):
        exact.append(record)
        records.append({"file": "records/" + name, "origin": os.path.join(records_dir, name),
                        "sha256": hashlib.sha256(blob).hexdigest(), "checks_equal_status": record.get("checks") == checks})
record = exact[0] if len(exact) == 1 else {}
if tests and len(exact) != 1:
    missing.append("the exact candidate record (%d found for %s)" % (len(exact), status.get("candidate_id")))
elif tests and record.get("checks") != checks:
    missing.append("the candidate record and the status name different checks")
if tests and checks.get("ok") != bool(tests.get("ok")):
    missing.append("checks.ok differs from tests.ok")
expected = {"new": record.get("composition_commit"), "baseline": part(record, "layers_base").get("k2"),
            "target_run": part(record, "target").get("target_sha")}
def check_run(role, summary, manifest, copied, copy, view):
    problems = []
    if not expected[role] or manifest.get("commit") != expected[role]:
        problems.append("run.json commit %s, not the %s commit %s of the candidate record" % (
            manifest.get("commit"), role, expected[role]))
    lanes = manifest.get("lanes")
    if not isinstance(lanes, list) or [lane.get("lane") if isinstance(lane, dict) else None for lane in lanes] != list(LANES):
        return problems + ["run.json lanes are not the parallel and the serial lane"]
    merged = {}
    for lane in lanes:
        label = lane["lane"]
        absent = [key for key in ("argv", "returncode", "cases", "seconds", "output") if key not in lane]
        if absent:
            problems.append("the %s lane lacks %s" % (label, ", ".join(absent)))
        for stream in ("stdout", "stderr"):
            if "%s.%s.txt" % (label, stream) not in copied:
                problems.append("%s.%s.txt, that lane full output" % (label, stream))
        report = "%s.xml" % label
        if report in copied:
            try:
                found = cases(os.path.join(copy, report))
            except ElementTree.ParseError as exc:
                found = None
                problems.append("%s does not parse (%s)" % (report, exc))
        elif lane.get("returncode") == 5 and lane.get("cases") == 0:
            found = {}   # a lane none of the selected cases belongs to, as the producer allows it
        else:   # a lane that never started (output none: refused or past the deadline) has no report: not evidence
            found = None
            problems.append(report if lane.get("output") != "none" else "%s absent: that lane never started (exit %s, output "
                            "none), so nothing of it can be read" % (report, lane.get("returncode")))
        if found is not None:
            if len(found) != lane.get("cases"):
                problems.append("%s holds %d case(s), its lane names %s" % (report, len(found), lane.get("cases")))
            merged.update(found)
    failed = {key: value for key, value in merged.items() if value}
    crashed = sorted(key for key, value in failed.items() if CRASH in value)
    complete = not crashed and bool(merged) and all(
        lane.get("returncode") in (0, 1) and lane.get("cases") or lane.get("returncode") == 5 and not lane.get("cases")
        for lane in lanes)
    returncode = next((lane.get("returncode") for lane in lanes if lane.get("returncode") not in (0, 5)), 0)
    view["reports"] = {"cases": len(merged), "failed": failed, "crashed": crashed, "complete": complete,
                       "returncode": returncode}
    for key, value in (("cases", len(merged)), ("failed", failed), ("crashed", crashed), ("complete", complete),
                       ("returncode", returncode)):
        if manifest.get(key) != value:
            problems.append("run.json %s differs from what its reports show" % key)
    for key in ("complete", "cases", "returncode", "seconds", "crashed", "failed"):
        if key in summary and summary[key] != manifest.get(key):
            problems.append("the checks name %s %r, run.json %r" % (key, summary[key], manifest.get(key)))
    if "lanes" in summary and summary["lanes"] != [{key: lane.get(key) for key in ("lane", "returncode", "cases", "seconds")}
                                                   for lane in lanes]:
        problems.append("the checks name other lanes than run.json")
    if complete and any(lane.get("output") != "complete" for lane in lanes):
        problems.append("a complete run whose lane output is not complete")
    named = manifest.get("targets_file")   # what the lanes ran: argparse reads every @FILE argument, one argument per line
    argfiles = [[token for token in (lane.get("argv") if isinstance(lane.get("argv"), list) else [])[1:]
                 if isinstance(token, str) and token.startswith("@")] for lane in lanes]
    if named is None:
        if any(argfiles):
            problems.append("a lane argv reads an argument file that run.json does not name")
    else:
        own = "/update-checks/%s/%s" % (os.path.basename(str(view["reference"])), TARGETS)
        if not isinstance(named, str) or not os.path.isabs(named) or os.path.normpath(named) != named \
                or not named.endswith(own):
            problems.append("run.json targets_file %r is not the %s of this run folder" % (named, TARGETS))
        if any(tokens != ["@%s" % named] for tokens in argfiles):
            problems.append("a lane argv does not read exactly the argument file that run.json names")
        if TARGETS not in copied:
            problems.append("%s, the argument file of the lanes" % TARGETS)
        else:
            with open(os.path.join(copy, TARGETS), "rb") as handle:
                blob = handle.read()
            try:
                lines = blob.decode("utf-8").splitlines()
            except UnicodeDecodeError:
                lines = None
            if lines != manifest.get("targets"):
                problems.append("%s does not hold exactly the run.json targets, one per line" % TARGETS)
            elif any(line.startswith("@") for line in lines):
                problems.append("%s names a further argument file" % TARGETS)
    if role == "new":
        absent = [key for key in ("selection", "footprint", "local_tests", "dropped_local_tests", "targets")
                  if key not in manifest]
        if absent:
            problems.append("run.json lacks %s: only the new run carries them, a baseline or target run cannot stand in"
                            % ", ".join(absent))
        else:
            selection = manifest["selection"] if isinstance(manifest["selection"], dict) else {}
            for key, mine in (("tier", selection.get("tier", tests.get("tier"))), ("file_count", len(manifest["targets"])),
                              ("changed", len(manifest["footprint"])), ("local_tests", len(manifest["local_tests"])),
                              ("dropped_local_tests", manifest["dropped_local_tests"])):
                if key in tests and mine != tests[key]:
                    problems.append("run.json %s %r, the checks %r" % (key, mine, tests[key]))
        if "failed" in tests and tests["failed"] != manifest.get("failed"):
            problems.append("the checks failed cases differ from run.json")
    else:   # a baseline or target run carries no selection record and runs exactly the cases its role names
        if "selection" in manifest:
            problems.append("run.json carries the new run selection record: a new run cannot stand in for the %s" % role)
        named = sorted(part(tests, "failed")) if role == "baseline" else sorted(tests.get("blocking") or [])
        targets = manifest.get("targets") if isinstance(manifest.get("targets"), list) else None
        if role == "baseline" and targets != named:
            problems.append("run.json targets are not exactly the new run failures %s" % named)
        if role == "target_run" and (not targets or not set(targets) <= set(named)):
            problems.append("run.json targets are not blocking cases")
    return problems
runs, folders = {}, {}
os.makedirs(os.path.join(out, "readiness"))
for role in ("new", "baseline", "target_run"):
    summary = tests.get(role)
    if not isinstance(summary, dict):
        continue
    reference, problems = summary.get("evidence"), []
    view = runs[role] = {"reference": reference, "expected_commit": expected[role]}
    if role == "target_run" and not reference and summary.get("cases") == 0:
        view["note"] = "no blocking case has its file in the target: the producer ran nothing there"
        continue
    if not reference:
        problems.append("the checks name no evidence folder")
    elif not os.path.basename(str(reference)).startswith("junit-") or \
            os.path.basename(os.path.dirname(str(reference))) != "update-checks":
        problems.append("%s is not a junit-* folder of update-checks" % reference)
    elif os.path.islink(reference) or not os.path.isdir(reference):
        problems.append("%s is absent or a link" % reference)
    elif os.path.realpath(reference) in folders:
        problems.append("%s is the %s run folder" % (reference, folders[os.path.realpath(reference)]))
    if not problems:
        folders[os.path.realpath(reference)] = role
        copy = os.path.join(out, "readiness", role)
        os.makedirs(copy)
        copied = {}
        for name in NAMES:
            source = os.path.join(reference, name)
            if os.path.isfile(source) and not os.path.islink(source):
                shutil.copyfile(source, os.path.join(copy, name))
                copied[name] = digest(os.path.join(copy, name))
        view.update(copied=copied, other_files=sorted(set(os.listdir(reference)) - set(NAMES)))
        try:
            with open(os.path.join(copy, "run.json")) as handle:
                manifest = json.load(handle)
        except (OSError, ValueError):
            manifest = None
        if isinstance(manifest, dict):
            problems.extend(check_run(role, summary, manifest, copied, copy, view))
        else:
            problems.append("run.json")
    view["missing"] = problems
    missing.extend("%s run: %s" % (role, item) for item in problems)
ready_claim = outcome == "PASS" and state == "ready"
if ready_claim:   # a ready claim is backed only by its own complete new run (and, for each failure, the baseline)
    reports = (runs.get("new") or {}).get("reports") or {}
    if not tests:
        missing.append("checks.tests")
    elif "new" not in runs:
        missing.append("the new run (tests.new)")
    elif reports.get("complete") is not True:
        missing.append("a complete new run in its own reports")
    elif tests.get("ok") is not True or tests.get("unverified"):
        missing.append("tests.ok without unverified")
    elif reports.get("failed"):   # ready although the new run failed: each failure must already fail on W
        prior = (runs.get("baseline") or {}).get("reports") or {}
        if prior.get("complete") is not True:
            missing.append("a complete baseline run behind the new run failures")
        elif any((prior.get("failed") or {}).get(key) != value for key, value in reports["failed"].items()):
            missing.append("a baseline failing each new failure the same way")
servers = [{"pid": row.get("pid"), "argv0": (row.get("argv") or [""])[0], "image": row.get("image")}
           for row in rows("proc/snapshots.jsonl") if server_mark in str(row.get("command") or "")]
new_failed = sorted(((runs.get("new") or {}).get("reports") or {}).get("failed") or part(tests, "failed"))
failures = {"new_run_failed": new_failed, "already_failing": sorted(tests.get("already_failing") or []),
            "blocking": sorted(tests.get("blocking") or []), "target_failing": sorted(tests.get("target_failing") or [])}
qualified = ready_claim and not missing
receipt = {
    "outcome": text("outcome.txt"), "expect": text("expect.txt"), "evidence": {"complete": not missing, "missing": missing},
    "readiness_qualified": qualified,   # green: that qualification, and no failure named by the checks or read in any run
    "green": qualified and not any(failures.values()) and not any(
        ((view.get("reports") or {}).get("failed")) for view in runs.values()), "failures": failures,
    "producer": {"source": "supervisor/update_checks.py", "sha256": PRODUCER,
                 "status": "the final freeze (4e53a124 tree 08eab970, no commit), read against its retained actual sample"},
    "bound": {"cand": cand, "ctree": ctree, "published_o0": o0 or None, "published_o1": o1 or None,
              "serving_lock_sha256": lock or None},
    "range": {key: status.get(key) for key in ("target_sha", "target_ref", "target_version", "channel", "base_sha",
                                               "base_version")},
    "target_is_bound_o1": bool(o1) and status.get("target_sha") == o1,
    "candidate": {key: status.get(key) for key in ("candidate_id", "state", "reason", "merge_kind", "local_changes",
                                                   "local_commits", "without_local_changes", "environment_same")},
    "selection": {key: tests.get(key) for key in ("tier", "file_count", "changed", "local_tests", "dropped_local_tests",
                                                  "uncovered_count")},
    "checks": checks, "runs": runs, "candidate_records": records, "serving_server": servers[:4],
    "sources": {"status_final": {"file": "status-final.json", "sha256": digest(os.path.join(out, "status-final.json")),
                                 "origin": "GET /api/update/status, its candidate projection"},
                "procwatch": {"snapshots": digest(os.path.join(out, "proc/snapshots.jsonl")),
                              "events": digest(os.path.join(out, "proc/events.jsonl"))}},
    "gaps": [
        "this reader is bound to the final frozen producer supervisor/update_checks.py sha256 " + PRODUCER +
        " (tree 08eab970): it does not read the serving checkout own copy; any other producer bytes need a rebind",
        "selection.record() is test_selection, outside that producer: only its tier is checked, with the producer own "
        "targets, footprint, local_tests and dropped_local_tests; its other fields and the floor are kept as found",
        "the run folders live in the candidate scratch and are retired with its checkout: they are copied here at "
        "settle, and nothing later can recover them",
    ],
    "request_utc": text("utc-request.txt"), "result": text("result.txt"),
    "local_footprint": {"before": text("bytes-before.txt"), "after": text("bytes-after.txt")},
}
with open(os.path.join(out, "readiness.json"), "x") as handle:
    json.dump(receipt, handle, indent=1, sort_keys=True)
with open(os.path.join(out, "evidence.txt"), "x") as handle:
    handle.write("evidence=complete\n" if not missing else "evidence=incomplete missing=%s\n" % "; ".join(missing))
print("readiness: roles %s, new run %s case(s), evidence %s" % (",".join(runs) or "none",
      ((runs.get("new") or {}).get("reports") or {}).get("cases"), "complete" if not missing else "incomplete"))' \
        "$1" "$2" "$3" "$CAND" "$CTREE" "$PUB_O0" "$PUB_O1" "$lock" "$REPO/server.py" "$DATA/state/private_copies" \
        > "$1/readiness.stdout" 2>&1
}

activate() {   # activate RUN STEP [SECS]: the real activation request (the body the Updates panel sends), procwatch armed.
               # PASS: a 200 answer naming this candidate activating with its adoption id, that adoption's own
               # body_adoption_closed row with outcome adopted, update_activation.json retaining this candidate adopted,
               # a launcher and a new server serving, the owner's H/I/W/U bytes kept and the previous HEAD contained in
               # the new one. FAIL: refused, closed otherwise, not serving, or local work lost. UNKNOWN: no adoption id
               # or no closure within SECS, or an incomplete observation
    run_root "$1"; local out secs="${3:-900}" cand port t0 i h phase last ident code ended=0 after=0 http sent aid outcome= \
        ps_after bytes_after V WHY
    out=$(step_dir "$2") || exit 2
    port=$(port_now); [ -n "$port" ] || die "no port in $PORTF"
    get_status "$out/status-before.json" > /dev/null
    cand=$(jget "$out/status-before.json" candidate.candidate_id)
    [ "$(jget "$out/status-before.json" candidate.state)" = ready ] && [ -n "$cand" ] || die "no ready candidate"
    printf '{"strategy": "activate", "candidate_id": "%s"}\n' "$cand" > "$out/body.json"
    ps_run "$out/ps-before.txt" || die "the process table is unreadable before the request ($out)"
    /usr/bin/grep -F "$REPO/server.py" "$out/ps-before.txt" > "$out/server-before.txt"
    closed_count > "$out/closed-before.txt"
    checkout_bytes > "$out/bytes-before.txt" 2>&1 || die "the owner's H/I/W/U are unreadable before the request ($out/bytes-before.txt)"
    arm "$out" "$secs"
    echo "POST http://127.0.0.1:$port/api/update/apply body=$out/body.json" > "$out/request.txt"
    utc > "$out/utc-request.txt"
    /usr/bin/curl -sS -m 120 -X POST -H 'Content-Type: application/json' --data-binary "@$out/body.json" \
        -o "$out/response.json" -w '%{http_code}\n' "http://127.0.0.1:$port/api/update/apply" > "$out/status.txt" 2> "$out/stderr.txt"
    sent=$?; rc_line "$sent" >> "$out/status.txt"; utc > "$out/utc-response.txt"
    http=$(/usr/bin/head -1 "$out/status.txt")
    aid=$(jget "$out/response.json" candidate.activation.adoption_id)
    case "$http" in 200|000|'') ;; *) ended=refused ;; esac   # answered otherwise: not applied, nothing restarts
    h="$DATA/state/body_adoption/handoff.json"; t0=$(/bin/date +%s); KEPT=0
    /bin/mkdir -p "$out/kept"
    if [ "$ended" = 0 ]; then
        for i in $(/usr/bin/seq 1 "$secs"); do
            phase=$(jget "$h" phase); ident=$(/usr/bin/stat -f '%i' "$APP" 2>/dev/null || echo -)
            last=$(/usr/bin/tail -n 1 "$R"/Applications/.Ouroboros-update-*/install.jsonl 2>/dev/null | /usr/bin/cut -c1-160)
            code=$(http_code)
            # A lost answer: the adoption id is read from this candidate's own handoff.
            [ -n "$aid" ] || { [ "$(jget "$h" candidate_id)" = "$cand" ] && aid=$(jget "$h" id); }
            outcome=$(closure_outcome "$aid")
            echo "$(utc) +$(( $(/bin/date +%s) - t0 ))s phase=${phase:--} inode=$ident port=$(port_now) http=$code servers=$(server_pids | /usr/bin/tr '\n' ',') closed=$(closed_count) adoption=${aid:--} outcome=${outcome:--} | $last" >> "$out/timeline.txt"
            for f in "$R"/Applications/.Ouroboros-update-*/install.jsonl; do keep_if_changed "$f" "$out/kept/install.jsonl"; done
            keep_if_changed "$h" "$out/kept/handoff.json"
            keep_if_changed "$DATA/state/update_activation.json" "$out/kept/update_activation.json"
            keep_if_changed "$DATA/state/process_ledger.jsonl" "$out/kept/process_ledger.jsonl"
            if [ "$ended" = 0 ] && [ -n "$outcome" ]; then ended=1; fi
            if [ "$ended" = 1 ]; then
                after=$((after + 1))
                [ "$outcome" = adopted ] || break   # closed otherwise: nothing more to wait for
                if [ "$code" = 200 ] && launcher_running && [ -n "$(server_pids)" ] && [ "$after" -ge 20 ]; then break; fi
                [ "$after" -ge 300 ] && break
            fi
            /bin/sleep 1
        done
    fi
    echo "ended=$ended adoption_id=${aid:--} outcome=${outcome:--} http=${http:--} seconds=$(( $(/bin/date +%s) - t0 ))" > "$out/result.txt"
    /usr/bin/grep '"body_adoption_closed"' "$DATA/logs/supervisor.jsonl" > "$out/closed-rows.jsonl" 2>/dev/null
    /bin/cp "$DATA/state/update_activation.json" "$out/activation-after.json" 2>/dev/null
    ps_run "$out/ps-after.txt"; ps_after=$?
    /usr/bin/grep -F "$REPO/server.py" "$out/ps-after.txt" > "$out/server-after.txt" 2>/dev/null
    checkout_bytes > "$out/bytes-after.txt" 2>&1; bytes_after=$?
    code=$(http_code)
    disarm "$out"
    if [ "$ended" = refused ]; then V=FAIL; WHY="answered http=$http: $(jget "$out/response.json" outcome) $(jget "$out/response.json" reason)"
    elif [ -z "$aid" ]; then V=UNKNOWN; WHY="no adoption id (answer http=${http:--}, curl exit $sent) and no handoff for $cand"
    elif [ -z "$outcome" ]; then V=UNKNOWN; WHY="adoption $aid did not close within ${secs}s"
    elif [ "$outcome" != adopted ]; then V=FAIL; WHY="adoption $aid closed $outcome"
    elif [ "$http" = 200 ] && { [ "$(jget "$out/response.json" candidate.candidate_id)" != "$cand" ] ||
                                [ "$(jget "$out/response.json" candidate.state)" != activating ]; }; then
        V=FAIL; WHY="the answer did not name $cand activating (response.json)"
    elif [ ! -f "$out/activation-after.json" ]; then V=UNKNOWN; WHY="adopted; update_activation.json not retained"
    elif [ "$(jget "$out/activation-after.json" candidate_id)" != "$cand" ] ||
         [ "$(jget "$out/activation-after.json" outcome)" != adopted ]; then
        V=FAIL; WHY="adopted, but update_activation.json retains $(jget "$out/activation-after.json" candidate_id) $(jget "$out/activation-after.json" outcome)"
    elif [ "$ps_after" != 0 ]; then V=UNKNOWN; WHY="adoption $aid closed adopted; the process table was unreadable afterwards"
    elif [ "$code" != 200 ] || [ ! -s "$out/server-after.txt" ] ||
         ! /usr/bin/grep -q -F -e "$APP/Contents/MacOS/" -e "$SEED/launcher.py" -e "$REPO/launcher.py" "$out/ps-after.txt"; then
        V=FAIL; WHY="adoption $aid closed adopted, but no launcher and server serve (http=$code)"
    elif /usr/bin/cmp -s "$out/server-before.txt" "$out/server-after.txt"; then V=FAIL; WHY="adopted, but the server is the same process"
    elif [ "$bytes_after" != 0 ]; then
        V=UNKNOWN; WHY="adoption $aid closed adopted and a new server serves; the owner's H/I/W/U were not read completely afterwards (bytes-after.txt)"
    elif [ "$(work_lines "$out/bytes-before.txt")" != "$(work_lines "$out/bytes-after.txt")" ]; then
        V=FAIL; WHY="adopted, but the owner's H/I/W/U bytes changed (bytes-*.txt)"
    elif ! g merge-base --is-ancestor "$(head_of "$out/bytes-before.txt")" "$(head_of "$out/bytes-after.txt")" 2>/dev/null; then
        V=FAIL; WHY="adopted, but the new HEAD does not contain the previous one (bytes-*.txt)"
    else V=PASS; WHY="adoption $aid closed adopted; $cand retained adopted; a new server serves; H/I/W/U kept ($(witness_scope "$out/bytes-after.txt")); the new HEAD contains the previous one"; fi
    unobserved
    echo "$out"; verdict "$out" "$V" "adopted, serving" "$WHY"
}

restart() {   # the owner's restart (POST /api/command {"cmd": "/restart"}), procwatch armed: who acts first after exit 42.
              # PASS: answered 2xx, every old server's exit 42 observed, a new server serving with a launcher running
    run_root "$1"; local out secs="${3:-600}" port t0 i old code lines sent http pid ex served=0 ps_after V=PASS WHY=""
    out=$(step_dir "$2") || exit 2
    port=$(port_now); [ -n "$port" ] || die "no port in $PORTF"
    ps_run "$out/ps-before.txt" || die "the process table is unreadable before the request ($out)"
    old=$(/usr/bin/grep -F "$REPO/server.py" "$out/ps-before.txt" | /usr/bin/awk '{print $1}' | /usr/bin/tr '\n' ',')
    echo "$old" > "$out/server-before.txt"
    lines=$(/usr/bin/wc -l < "$DATA/logs/launcher.log" 2>/dev/null | /usr/bin/tr -d ' '); echo "${lines:-0}" > "$out/launcher-log-lines-before.txt"
    arm "$out" "$secs"
    echo '{"cmd": "/restart"}' > "$out/body.json"; utc > "$out/utc-request.txt"
    /usr/bin/curl -sS -m 60 -X POST -H 'Content-Type: application/json' --data-binary "@$out/body.json" \
        -o "$out/response.json" -w '%{http_code}\n' "http://127.0.0.1:$port/api/command" > "$out/status.txt" 2> "$out/stderr.txt"
    sent=$?; rc_line "$sent" >> "$out/status.txt"; http=$(/usr/bin/head -1 "$out/status.txt")
    t0=$(/bin/date +%s)
    for i in $(/usr/bin/seq 1 "$secs"); do
        code=$(http_code)
        echo "$(utc) +$(( $(/bin/date +%s) - t0 ))s http=$code servers=$(server_pids | /usr/bin/tr '\n' ',')" >> "$out/timeline.txt"
        if [ "$i" -ge 10 ] && [ "$code" = 200 ] && [ -n "$(server_pids)" ] && [ "$(server_pids | /usr/bin/tr '\n' ',')" != "$old" ]; then
            served=1; /bin/sleep 15; break; fi
        /bin/sleep 1
    done
    /usr/bin/tail -n +"$(( ${lines:-0} + 1 ))" "$DATA/logs/launcher.log" > "$out/launcher-log-new.txt" 2>&1
    ps_run "$out/ps-after.txt"; ps_after=$?
    disarm "$out"
    [ -n "$old" ] || { V=UNKNOWN; WHY="no server before the request;"; }
    for pid in $(echo "$old" | /usr/bin/tr ',' ' '); do   # an exit status is claimed only from its observed event
        ex=$(exit_of "$out/proc/events.jsonl" "$pid"); echo "$pid ${ex:-no_exit_observed}" >> "$out/old-server-exits.txt"
        case "$ex" in
            exit_code=42) ;;
            '') if [ "$ps_after" = 0 ] && /usr/bin/awk -v p="$pid" '$1 == p' "$out/ps-after.txt" | /usr/bin/grep -q -F "$REPO/server.py"; then
                    V=FAIL; WHY="$WHY old server $pid still runs;"
                else [ "$V" = FAIL ] || V=UNKNOWN; WHY="$WHY old server $pid: exit not observed;"; fi ;;
            *) V=FAIL; WHY="$WHY old server $pid $ex, not exit 42;" ;;
        esac
    done
    if [ "$sent" != 0 ]; then [ "$V" = FAIL ] || V=UNKNOWN; WHY="$WHY /restart curl exit $sent;"
    else case "$http" in 2??) ;; *) V=FAIL; WHY="$WHY /restart answered http=$http;" ;; esac; fi
    if [ "$ps_after" != 0 ]; then [ "$V" = FAIL ] || V=UNKNOWN; WHY="$WHY the process table was unreadable afterwards;"
    else
        [ "$served" = 1 ] || { V=FAIL; WHY="$WHY no new server served within ${secs}s;"; }
        /usr/bin/grep -q -F -e "$APP/Contents/MacOS/" -e "$SEED/launcher.py" -e "$REPO/launcher.py" "$out/ps-after.txt" ||
            { V=FAIL; WHY="$WHY no launcher runs afterwards;"; }
    fi
    [ -n "$WHY" ] || WHY="old server exit 42; a new server serves; the launcher runs"
    unobserved
    echo "$out"; verdict "$out" "$V" "ordinary restart" "$WHY"
}

legacy() {   # the stock 7.8.0 updater as its Updates panel drives it: check, preflight, apply (auto_merge plan).
             # PASS: applied (2xx), HEAD moved to a commit containing CAND, a new server serving; not applied is FAIL
    run_root "$1"; local out secs="${3:-1800}" port t0 i strategy code old head_before head served=0 applied V WHY
    out=$(step_dir "$2") || exit 2
    port=$(port_now); [ -n "$port" ] || die "no port in $PORTF"
    old=$(server_pids | /usr/bin/tr '\n' ','); pids_of_run > "$out/ps-before.txt" 2>&1
    arm "$out" "$secs"
    /usr/bin/curl -sS -m 600 -X POST -H 'Content-Type: application/json' --data-binary '{}' -o "$out/check.json" \
        -w '%{http_code}\n' "http://127.0.0.1:$port/api/update/check" > "$out/check-status.txt" 2>&1
    /usr/bin/curl -sS -m 900 -X POST -H 'Content-Type: application/json' --data-binary '{}' -o "$out/preflight.json" \
        -w '%{http_code}\n' "http://127.0.0.1:$port/api/update/preflight" > "$out/preflight-status.txt" 2>&1
    # 7.8.0's own rule (web/modules/api_client.js updateStrategyForPlan): available + kind clean -> auto_merge.
    strategy=none
    [ "$(jget "$out/preflight.json" merge_plan.available)" = True ] && case "$(jget "$out/preflight.json" merge_plan.kind)" in
        clean) strategy=auto_merge ;; conflicting) strategy=assisted ;; esac
    "$OBSPY" -I -S -B -c 'import json, sys
plan = json.load(open(sys.argv[1])).get("merge_plan") or {}
print(json.dumps({"strategy": sys.argv[2], "expected_base_sha": str(plan.get("base_sha") or ""),
                  "expected_target_sha": str(plan.get("target_sha") or "")}))' "$out/preflight.json" "$strategy" > "$out/apply-body.json"
    if [ "$strategy" != auto_merge ] || [ -z "$(jget "$out/apply-body.json" expected_target_sha)" ]; then
        echo "not applied: strategy=$strategy (the stock plan is not a direct merge; nothing sent)" > "$out/result.txt"
        disarm "$out"; echo "$out"
        verdict "$out" FAIL "stock auto_merge delivery of CAND" "not applied: strategy=$strategy; nothing sent"; return
    fi
    head_before=$(g rev-parse HEAD 2>/dev/null)
    utc > "$out/utc-apply.txt"
    /usr/bin/curl -sS -m 900 -X POST -H 'Content-Type: application/json' --data-binary "@$out/apply-body.json" \
        -o "$out/apply.json" -w '%{http_code}\n' "http://127.0.0.1:$port/api/update/apply" > "$out/apply-status.txt" 2>&1
    applied=$(/usr/bin/head -1 "$out/apply-status.txt")
    t0=$(/bin/date +%s)
    for i in $(/usr/bin/seq 1 "$secs"); do
        code=$(http_code)
        echo "$(utc) +$(( $(/bin/date +%s) - t0 ))s http=$code servers=$(server_pids | /usr/bin/tr '\n' ',') head=$(g rev-parse --short HEAD 2>/dev/null)" >> "$out/timeline.txt"
        if [ "$i" -ge 10 ] && [ "$code" = 200 ] && [ -n "$(server_pids)" ] && [ "$(server_pids | /usr/bin/tr '\n' ',')" != "$old" ]; then
            served=1; /bin/sleep 20; break; fi
        /bin/sleep 1
    done
    echo "strategy=$strategy seconds=$(( $(/bin/date +%s) - t0 ))" > "$out/result.txt"
    disarm "$out"
    head=$(g rev-parse HEAD 2>/dev/null)
    case "$applied" in 2??) V=PASS; WHY="" ;; *) V=FAIL; WHY="apply answered http=${applied:--};" ;; esac
    [ "$served" = 1 ] || { V=FAIL; WHY="$WHY no new server served within ${secs}s;"; }
    [ -n "$head" ] && [ "$head" != "$head_before" ] || { V=FAIL; WHY="$WHY HEAD did not move;"; }
    g merge-base --is-ancestor "$CAND" "$head" 2>/dev/null || { V=FAIL; WHY="$WHY HEAD ${head:--} does not contain CAND;"; }
    [ -n "$WHY" ] || WHY="HEAD $head contains CAND; a new server serves"
    unobserved
    echo "$out"; verdict "$out" "$V" "stock auto_merge delivery of CAND" "$WHY"
}

mkplus() {   # the lab successor P = CAND + one lock line (PLUS_FROM -> PLUS_TO), deterministic, on `ouroboros` (DISCLOSED)
    run_root "$1"; local out idx blob tree plus
    [ -n "$PLUS_FROM" ] && [ -n "$PLUS_TO" ] && [ -n "$PLUS_DATE" ] || die "no bound lock change"
    out=$(step_dir mkplus) || exit 2
    local idxdir rc
    idxdir=$(/usr/bin/mktemp -d "$TMPD/qb1-index.XXXXXX") || return 2
    idx="$idxdir/index"   # absent, not an empty file Git would reject
    (
        set -euo pipefail
        [ "$("$GIT" -C "$MIRROR" cat-file blob "$CAND:requirements-runtime.lock" | /usr/bin/grep -c -x -F "$PLUS_FROM")" = 1 ]
        blob=$("$GIT" -C "$MIRROR" cat-file blob "$CAND:requirements-runtime.lock" \
            | /usr/bin/sed "s/^$(printf '%s' "$PLUS_FROM" | /usr/bin/sed 's/[.]/[.]/g')\$/$PLUS_TO/" | "$GIT" -C "$MIRROR" hash-object -w --stdin)
        GIT_INDEX_FILE="$idx" "$GIT" -C "$MIRROR" read-tree "$CAND"; rc_line $?
        GIT_INDEX_FILE="$idx" "$GIT" -C "$MIRROR" update-index --cacheinfo "100644,$blob,requirements-runtime.lock"; rc_line $?
        tree=$(GIT_INDEX_FILE="$idx" "$GIT" -C "$MIRROR" write-tree)
        plus=$(GIT_AUTHOR_NAME=qb1-lab GIT_AUTHOR_EMAIL=lab@lab.invalid GIT_AUTHOR_DATE="$PLUS_DATE" \
               GIT_COMMITTER_NAME=qb1-lab GIT_COMMITTER_EMAIL=lab@lab.invalid GIT_COMMITTER_DATE="$PLUS_DATE" \
               "$GIT" -C "$MIRROR" commit-tree "$tree" -p "$CAND" -m "lab: $PLUS_FROM -> $PLUS_TO (qb1 successor fixture)")
        "$GIT" -C "$MIRROR" update-ref refs/heads/ouroboros "$plus" "$CAND"; rc_line $?
        echo "PLUS=$plus TREE=$tree LOCK_BLOB=$blob"
        "$GIT" -C "$MIRROR" diff "$CAND" "$plus"
        "$GIT" -C "$MIRROR" for-each-ref --format='%(refname) %(objectname)' refs/heads
    ) > "$out/mkplus.txt" 2>&1
    rc=$?; echo "exit_code=$rc index=$idx" > "$out/result.txt"
    echo "$out"; return "$rc"
}

retarget() {   # retarget RUN: the mirror's `ouroboros` moves CAND -> the bound PUB_O1 (compare-and-swap). Equipment mechanics
               # only: whether a range qualifies as published official readiness is bound by the parent, not here
    run_root "$1"; local out rc=0 got
    [ -n "$PUB_O0" ] && [ -n "$PUB_O1" ] || die "no bound published range (PUB_O0, PUB_O1 in bind.env)"
    out=$(step_dir retarget) || exit 2
    {
        got=$("$GIT" -C "$MIRROR" rev-parse -q --verify "refs/lab/published-o1^{commit}")
        if [ "$got" != "$PUB_O1" ]; then echo "refs/lab/published-o1 is ${got:-absent}, not PUB_O1 $PUB_O1"; rc=1
        elif got=$("$GIT" -C "$MIRROR" merge-base "$CAND" "$PUB_O1"); [ "$got" != "$PUB_O0" ]; then
            echo "merge-base(CAND, PUB_O1) is ${got:-none}, not PUB_O0 $PUB_O0"; rc=1
        elif "$GIT" -C "$MIRROR" merge-base --is-ancestor "$PUB_O1" "$CAND"; then echo "PUB_O1 is already in CAND"; rc=3
        else "$GIT" -C "$MIRROR" update-ref refs/heads/ouroboros "$PUB_O1" "$CAND"; rc=$?; rc_line "$rc"
        fi
        if [ "$rc" = 0 ]; then
            echo "range PUB_O0=$PUB_O0 PUB_O1=$PUB_O1 official_commits=$("$GIT" -C "$MIRROR" rev-list --count "$PUB_O0..$PUB_O1")"
            "$GIT" -C "$MIRROR" for-each-ref --format='%(refname) %(objectname)' refs/heads refs/lab
        fi
    } > "$out/retarget.txt" 2>&1
    echo "exit_code=$rc" > "$out/result.txt"
    echo "$out"; return "$rc"
}

wait_for() {
    run_root "$1"; local out secs="$3" expr="$4" t0 i ok=1
    out=$(step_dir "$2") || exit 2
    echo "$expr" > "$out/expr.txt"; t0=$(/bin/date +%s)
    for i in $(/usr/bin/seq 1 "$secs"); do
        if R="$R" APP="$APP" REPO="$REPO" DATA="$DATA" PORTF="$PORTF" VENV="$VENV" /bin/bash -c "$expr" >> "$out/looks.txt" 2>&1; then ok=0; break; fi
        /bin/sleep 1
    done
    echo "result=$ok seconds=$(( $(/bin/date +%s) - t0 ))" > "$out/result.txt"
    pids_of_run > "$out/ps-run.txt" 2>&1
    echo "$out"; return "$ok"
}

effects() {
    run_root "$1"; local out
    out=$(step_dir "$2") || exit 2
    /usr/bin/touch "$out/.end"
    /usr/bin/find "$GH" /private/tmp /private/var/folders -xdev \( -path "$EV" -prune \) -o \( -path "$ROOT/runs" -prune \) \
        -o \( -path "$Q" -prune \) -o \( -newer "$R/.start" ! -newer "$out/.end" -print \) 2>/dev/null > "$out/outside-runs.txt"
    /usr/bin/find "$R" -newer "$R/.start" ! -newer "$out/.end" 2>/dev/null | /usr/bin/wc -l > "$out/inside-run-count.txt"
    echo "$out"
}

pack() {
    local out; RUN="$NS"; out=$(step_dir "pack-$1") || exit 2
    (cd "$EV" && /usr/bin/tar -cf "$out/$NS-guest-evidence.tar" $(/bin/ls -d b1a-* b1e-* b1s-* b1p-* b1r-* "$NS"-* 2>/dev/null | /usr/bin/grep -v "^$NS-pack-$1\$"))
    local rc=$?; rc_line "$rc" > "$out/rc.txt"
    /usr/bin/shasum -a 256 "$out/$NS-guest-evidence.tar" > "$out/SHA256"
    /usr/bin/tar -tf "$out/$NS-guest-evidence.tar" | /usr/bin/wc -l > "$out/members.txt"
    echo "$out"; return "$rc"
}

case "${1:-}" in
    facts) facts "${2:-x}" ;;
    selftest) selftest "${2:-x}" ;;
    netcheck) netcheck "${2:-x}" ;;
    install) install "${2:-}" ;;
    mirror) mirror "${2:-}" ;;
    candsrc) candsrc "${2:-}" ;;
    extpy) extpy "${2:-}" ;;
    venvprep) venvprep "${2:-}" ;;
    barepip) barepip "${2:-}" ;;
    seed) seed "${2:-}" ;;
    open) open_app "${2:-}" "${3:-}" "${@:4}" ;;
    openw) openw "${2:-}" "${3:-}" "${4:-300}" "${@:5}" ;;
    obs) obs "${2:-}" "${3:-}" ;;
    quit) quit_app "${2:-}" "${3:-}" "${4:-240}" ;;
    compose) compose "${2:-}" ;;
    localfn) localfn "${2:-}" "${3:-}" "${4:-}" "${5:-}" ;;
    api) api "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}" ;;
    status) status "${2:-}" "${3:-}" ;;
    settings) settings "${2:-}" "${3:-}" "${4:-}" "${5:-}" ;;
    prepare) prepare "${2:-}" "${3:-}" "${4:-5400}" "${5:-ready}" "${6:-}" ;;
    activate) activate "${2:-}" "${3:-}" "${4:-900}" ;;
    restart) restart "${2:-}" "${3:-}" "${4:-600}" ;;
    legacy) legacy "${2:-}" "${3:-}" "${4:-1800}" ;;
    mkplus) mkplus "${2:-}" ;;
    retarget) retarget "${2:-}" ;;
    wait) wait_for "${2:-}" "${3:-}" "${4:-60}" "${5:-false}" ;;
    lscanary) lscanary "${2:-}" "${3:-}" ;;
    effects) effects "${2:-}" "${3:-}" ;;
    pack) pack "${2:-x}" ;;
    *) die "usage: qb1.sh facts|selftest|netcheck|install|mirror|candsrc|extpy|venvprep|barepip|seed|open|openw|obs|quit|compose|localfn|api|status|settings|prepare|activate|restart|legacy|mkplus|retarget|wait|lscanary|effects|pack ..." ;;
esac
