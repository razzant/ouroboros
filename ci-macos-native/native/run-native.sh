#!/bin/bash
# One bounded native case of the unified Update qualification on a GitHub-hosted macOS arm64 runner (test equipment,
# not product): A (b1a, a venv of the 7.7.0 app's own Python) or b1e (a venv of an external copy of that Python). The
# official Ouroboros 7.7.0 app, copied from its pinned public DMG, is the owner's existing installation: Gatekeeper
# rejects that fresh copy (Unnotarized Developer ID; kept as the install claim's measured FAIL, a declared known
# negative), so it is used only as an existing copy, opened through LaunchServices and quit through its own Apple Event.
# It serves product C from a checkout composed with the owner's local work, on the venv the owner prepared earlier, whose
# packages come from a public wheel folder pinned by C's own uv.lock. C's own Update then prepares (its real checks) and
# activates the official 7.8.0 app in place of 7.7.0; the app is quit, reopened cold, and checked for its roots, runtime,
# profile and the owner's local function.
# Reused byte-identical from the hosted b1s probe: procwatch.py, mkbundle.sh, capture.py; qb1.sh from it with its
# refusal receipt and profile record repaired (README); venv_lab.py from the lab equipment with a docstring-only change.
# This runner is that probe's run-probe.sh adapted to the two cases.
#   run-native.sh OUT_ROOT CASE   CASE is b1a or b1e. OUT_ROOT must not exist; OUT_ROOT/out is the upload (raw public
#                                 receipts), OUT_ROOT/work is scratch. The run's fixtures live under /Users/anka/oulab,
#                                 created here, create-only; qb1's evidence folder there is a link to
#                                 OUT_ROOT/out/guest-evidence, so it is uploaded with OUT_ROOT/out. A pre-existing
#                                 /Users/anka is refused and nothing under it is touched.
# Outcome: OUT_ROOT/out/RESULT.txt and claims.tsv. Exit 0 PASS (every claim PASS, the declared known negative aside),
# 1 FAIL, 3 UNKNOWN (a claim not established, refused or not run), 2 a procedure error. A step's exit status alone is
# never its verdict.
set -u
umask 022
PATH=/usr/bin:/bin:/usr/sbin:/sbin
export PATH
# /bin/bash is 3.2 here: no associative arrays, and no case statement written inside $( ).

# Product C (C7), kept apart from the execution commit (GITHUB_SHA = C plus this overlay, nothing else).
C=b87e41411d9172846b09570d46aaa67539355e1a
CT=20789d1554e5be6c61a9ffd4c99846b96aff9a6c
CLOCK_PIN=68e739bde5b7fbf90d2663eb01939f52c5c4b0b087097390e05807e143e8673d
V770=e41596c389c3c33c3155637aaf67f8bebbf4934d
V780=6f3ddff05ff3e6c6f8d007ff96358df6af872e83
# Public inputs, each by exact size and sha256. URLs by C's own release rule (release_sync.release_asset_download_url).
REL780=https://github.com/razzant/ouroboros/releases/download/v7.8.0
DMG780_BYTES=754985809
DMG780_SHA=41b9a8901d29c9a9b5dfdda8780d9317c441ba8c2941d949041c15abae0eff57
SUMS780_BYTES=2642
SUMS780_SHA=c3c1c197760efa4a4a6f6b791f097c5fc288343f2ca708137180401d84dec049
REL770=https://github.com/razzant/ouroboros/releases/download/v7.7.0
DMG770_BYTES=751126575
DMG770_SHA=de1387593eda0f5ea5a162e23002f116eabdd7ea974406946d9d4623a5d9bdef
WORKFLOW=.github/workflows/qualification-update-native-macos.yml
# qb1's fixed geometry for the case (namespace qbci). Every path below is created by this run.
GH=/Users/anka
ROOT=$GH/oulab
NS=qbci
Q=$ROOT/$NS
EV=$Q/evidence
CLT=$ROOT/tools/clt-26.6-r2/Payload/Library/Developer/CommandLineTools
GIT=$CLT/usr/bin/git
OBSPY=$ROOT/pkg/7.8.0/Ouroboros.app/Contents/Frameworks/python-standalone/bin/python3
OFFICIAL780=$ROOT/pkg/7.8.0/Ouroboros.app
WHEELS=$ROOT/q7b4/wheelhouse
OWNER_PATHS='^(OWNER_NOTES\.md|owner-scratch\.txt|owner-untracked/.*)$'
# Bounds: no qb1 step starts unless its whole window fits before T0 + BUDGET_S - RESERVE_S (otherwise UNKNOWN
# time_budget and the chain stops; no window is shortened). The finish (an ordinary quit of what this run started) keeps
# RESERVE_S. The setup before the first step (downloads, each bounded only by curl --max-time; copies; mkbundle) is not
# under this bound: only the workflow's step and job timeouts bound everything. Disk: 25 GiB free to start, 8 GiB before
# each step.
BUDGET_S=20100
RESERVE_S=1200
# C's preparation is observed for one shared readiness-check budget: C's runtime_limits.UPDATE_CHECKS_TIMEOUT_SEC,
# 7200 s by default; qb1's launch environment names no OUROBOROS_PREFLIGHT_TIMEOUT_SEC, and the checks record the budget
# they used. It bounds the new run W', any comparison runs on W and T, their materializations and both lanes of each.
# Before it: 1800 s for the native app's downloads and staging, the generation's clone and pip, W' materialized once
# outside it and the prerequisite probes; after it: 600 s for W' restored (Git's own 300 s bound) and the record
# settled. qb1 follows the status that long; the client's window adds 2100 s for qb1's reads and copies after the
# settle. These are observation allowances sized from the default, never a product deadline: prepare's verdict reads
# the deadlines the checks recorded.
PREP_BUDGET_S=7200
PREP_GUEST_S=$((PREP_BUDGET_S + 1800 + 600))
PREP_WINDOW_S=$((PREP_GUEST_S + 2100))
START_FREE_KIB=26214400
STEP_FREE_KIB=8388608

[ $# = 2 ] || { echo "usage: run-native.sh OUT_ROOT b1a|b1e" >&2; exit 2; }
case "$1" in /*) ;; *) echo "OUT_ROOT must be absolute: $1" >&2; exit 2 ;; esac
CASE=$2
case "$CASE" in
    b1a) KIND=app ;;   # the venv of the installed app's own Python
    b1e) KIND=ext ;;   # the venv of an external copy of that Python, the owner's chosen interpreter
    *) echo "CASE is b1a or b1e: $CASE" >&2; exit 2 ;;
esac
R=$Q/runs/$CASE
APP=$R/Applications/Ouroboros.app
REPO=$R/app/repo
DATA=$R/app/data
HOMEDIR=$R/home
PORTF=$R/locks/server_port
TMPD=/private/tmp/$NS-$CASE/
MIRROR=$R/official.git
SEED=$R/seed
VENV=$R/venv
EXTROOT=$R/ext-r2
OUTROOT=$1
/bin/mkdir "$OUTROOT" || { echo "OUT_ROOT exists or cannot be made: $OUTROOT" >&2; exit 2; }
OUT=$OUTROOT/out
WORK=$OUTROOT/work
/bin/mkdir "$OUT" "$WORK" "$OUT/receipts" "$OUT/runner" "$OUT/inputs" "$OUT/product" "$OUT/checks" "$OUT/watch" \
    "$OUT/guest-evidence" || exit 2
HERE=$(cd "$(/usr/bin/dirname "$0")" && /bin/pwd -P) || exit 2   # <checkout>/ci-macos-native/native
OVL=$(/usr/bin/dirname "$HERE")
W=$(/usr/bin/dirname "$OVL")
ESHA=${GITHUB_SHA:-}
RHOME=${HOME:-}
U=$(/usr/bin/id -un)
UTMP=${TMPDIR:-/private/tmp}
T0=$(/bin/date +%s)
SEQ=0 N=0 STOPPED= PROC_ERR= FINISHING= SESSION=- H_OWNER= CLTSRC= HGIT= BUNDLE= RESULT_CODE=2
WJ= WDIR= WRC=none STEP_RAN=0
OWNED=   # 1 only once this run's own mkdir made /Users/anka: nothing under it is touched otherwise
SUMMARY=   # 1 once seal has put this run's result into the step summary (a later stop then amends it)
printf 'seq\tclaim\tverdict\tname\tdetail\n' > "$OUT/claims.tsv"

PY_EXITS='import json, sys
path, app, repo, wrc = sys.argv[1:5]
rows = []
try:
    with open(path, errors="replace") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
except OSError as exc:
    print("events unreadable: %s" % exc)
    sys.exit(1)
cmds, how, exits = {}, {}, {}
for row in rows:
    event, pid = row.get("event"), row.get("pid")
    if event in ("seen", "exec_changed"):
        cmds.setdefault(pid, []).append(row.get("command") or "")
    elif event == "registered":
        how[pid] = row.get("how") or ""
    elif event == "exit":
        exits[pid] = row


def named(pid, mark):
    return any(mark in command for command in cmds.get(pid, []))


def outcome(pid):
    row = exits.get(pid)
    if not row:
        return "%s no exit (%s)" % (pid, how.get(pid, "unregistered"))
    if not row.get("status_reported"):
        return "%s exit without status" % pid
    if row.get("exited"):
        return "%s exit_code=%s" % (pid, row.get("exit_code"))
    return "%s signal=%s" % (pid, row.get("signal"))


launchers = sorted(pid for pid in cmds if named(pid, app + "/Contents/MacOS/"))
servers = sorted(pid for pid in cmds if named(pid, repo + "/server.py"))
others = sorted(pid for pid in how if pid not in launchers and pid not in servers)
problems = []
if not launchers:
    problems.append("no launcher seen")
if not servers:
    problems.append("no server seen")
for pid in launchers + servers:
    if how.get(pid) != "exit+status":
        problems.append("%s registered %s" % (pid, how.get(pid, "never")))
for pid in sorted(how):
    if how[pid] == "exit+status" and not (exits.get(pid) or {}).get("status_reported"):
        problems.append("no exit status for %s" % pid)
if wrc != "0":
    problems.append("watcher exit %s" % wrc)
print("launcher %s; server %s; other %s%s" % (
    ", ".join(outcome(pid) for pid in launchers) or "-",
    ", ".join(outcome(pid) for pid in servers) or "-",
    ", ".join(outcome(pid) for pid in others) or "-",
    ("; " + "; ".join(problems)) if problems else ""))
sys.exit(1 if problems else 0)'

# PY_SERVER SNAPSHOT ARGV_TXT USERBASE: the server's kernel image and argv[0], then its own recorded environment
# (procwatch snap: every name, a sha256 prefix of every value, plain values for allowlisted names) against the exact
# --env tuple qb1 passed to that open (argv.txt) and, when USERBASE is given, the run's expected user base (empty for
# a venv run: a venv without system site packages turns the user site off, so the name is recorded, not judged).
# Exit 0 same, 1 a value differs, 3 not established (no environment record, or a requested name absent).
PY_SERVER='import hashlib, json, os, sys
snap, argv_file, userbase_want = sys.argv[1:4]
JUDGED = ("OUROBOROS_APP_ROOT", "OUROBOROS_REPO_DIR", "OUROBOROS_DATA_DIR", "OUROBOROS_SETTINGS_PATH",
          "OUROBOROS_PID_FILE", "OUROBOROS_PORT_FILE", "OUROBOROS_SERVER_PORT", "OUROBOROS_HOST_SERVICE_PORT",
          "OUROBOROS_SERVER_HOST", "HOME", "TMPDIR") + (("PYTHONUSERBASE",) if userbase_want else ())
PATHS = {"OUROBOROS_APP_ROOT", "OUROBOROS_REPO_DIR", "OUROBOROS_DATA_DIR", "OUROBOROS_SETTINGS_PATH",
         "OUROBOROS_PID_FILE", "OUROBOROS_PORT_FILE", "HOME", "TMPDIR", "PYTHONUSERBASE"}
RECORDED = ("CFFIXED_USER_HOME", "DEVELOPER_DIR", "QB1_OPERATOR_CANARY", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__")
try:
    with open(snap) as handle:
        rows = json.load(handle)
except (OSError, ValueError):
    rows = []
row = rows[0] if rows else {}
image = row.get("image") or ""
argv = row.get("argv") or [""]
print("image=%s" % (os.path.realpath(image) if image.startswith("/") else ""))
print("argv0=%s" % (argv[0] if argv else ""))
try:
    with open(argv_file) as handle:
        words = handle.read().splitlines()
except OSError:
    words = []
want = {}
for i in range(len(words) - 1):
    if words[i] == "--env" and "=" in words[i + 1]:
        name, value = words[i + 1].split("=", 1)
        want[name] = value
if userbase_want:
    want["PYTHONUSERBASE"] = userbase_want
env = row.get("env") or {}
keys, plain, sha16 = set(env.get("keys") or ()), env.get("plain") or {}, env.get("sha16") or {}
differs, missing = [], []
if not env:
    missing.append("environment-record")
for name in JUDGED:
    if name not in want:
        missing.append(name + "(not-requested)")
    elif env and name not in keys:
        missing.append(name)
        print("env %s requested=%s observed=absent" % (name, want[name]))
    elif env:
        if name in plain:
            got = plain[name]
            same = os.path.realpath(got) == os.path.realpath(want[name]) if name in PATHS else got == want[name]
        else:
            got = "sha16:%s" % sha16.get(name)
            same = sha16.get(name) == hashlib.sha256(want[name].encode("utf-8", "surrogateescape")).hexdigest()[:16]
        print("env %s requested=%s observed=%s %s" % (name, want[name], got, "same" if same else "DIFFERS"))
        if not same:
            differs.append(name)
for name in sorted(keys):
    if (name.startswith("PYTHON") and name not in JUDGED) or name in RECORDED:
        print("recorded %s=%s" % (name, plain.get(name, "sha16:%s" % sha16.get(name))))
print("differs=%s" % " ".join(differs))
print("missing=%s" % " ".join(missing))
sys.exit(1 if differs else 3 if missing else 0)'
PY_LOOKUP='import os, sys
try:
    os.stat(sys.argv[1])
except (FileNotFoundError, NotADirectoryError):
    print("absent")
except OSError as exc:
    print("lookup failed: %r" % exc)
else:
    print("present")'

utc() { /bin/date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "$(utc) $*" >> "$OUT/probe.log"; }
lower() { printf '%s' "$1" | /usr/bin/tr '[:upper:]' '[:lower:]'; }
oneline() { printf '%s' "$*" | /usr/bin/tr '\t\n' '  ' | /usr/bin/cut -c1-700; }
free_kib() { /bin/df -k "$1" 2>/dev/null | /usr/bin/awk 'NR == 2 {print $4}'; }
space() { echo "$(utc) $1 users_free_kib=$(free_kib /Users) out_free_kib=$(free_kib "$OUTROOT")" >> "$OUT/space.log"; }
clean() { /usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME="$RHOME" USER="$U" LOGNAME="$U" TMPDIR="$UTMP" "$@"; }
g0() { clean GIT_OPTIONAL_LOCKS=0 "$HGIT" -C "$W" "$@"; }   # the execution checkout, read only
gr() { /usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME="$HOMEDIR" GIT_OPTIONAL_LOCKS=0 "$GIT" -C "$REPO" "$@"; }

claim() {   # claim NAME VERDICT REASON DETAIL...: one line (PASS FAIL UNKNOWN REFUSED NOT_RUN RECORDED ERROR)
    local name="$1" v="$2" reason="$3"
    shift 3
    N=$((N + 1))
    printf '%02d\t%s\t%s\t%s\t%s\n' "$N" "$name" "$v" "$reason" "$(oneline "$*")" >> "$OUT/claims.tsv"
    log "claim $name $v $reason"
}
stop_chain() { [ -n "$STOPPED" ] || STOPPED=$1; log "chain stopped at $STOPPED"; }
gate() { claim "$@"; [ "$2" = PASS ] || stop_chain "$1"; [ "$2" = PASS ]; }   # a claim the next steps rest on
pending() { [ -z "$STOPPED" ] && return 0; claim "$1" NOT_RUN chain_stopped "after $STOPPED"; return 1; }
procedure_error() { PROC_ERR=$1; claim "$1" ERROR procedure_error "${2:-}"; stop_chain "$1"; return 1; }

home_facts() {   # the runner user's real home, where the app's defaults would land if the run's roots were ignored
    local f
    for f in "$RHOME/Library/WebKit/com.ouroboros.agent" "$RHOME/Library/Preferences/com.ouroboros.agent.plist" \
             "$RHOME/Ouroboros"; do
        echo "== $f"; /bin/ls -ld -T "$f" 2>&1
    done
}

runner_facts() {   # reads only; no environment dump (named variables only)
    local d="$OUT/runner"
    { echo "utc=$(utc)"; /usr/bin/uname -a
      echo "uname_m=$(/usr/bin/uname -m) hw.optional.arm64=$(/usr/sbin/sysctl -n hw.optional.arm64 2>&1) proc_translated=$(/usr/sbin/sysctl -n sysctl.proc_translated 2>&1)"
      echo "cpu=$(/usr/sbin/sysctl -n machdep.cpu.brand_string 2>&1) ncpu=$(/usr/sbin/sysctl -n hw.ncpu 2>&1) memsize=$(/usr/sbin/sysctl -n hw.memsize 2>&1)"
      /usr/bin/sw_vers
      echo "ImageOS=${ImageOS:-} ImageVersion=${ImageVersion:-} RUNNER_OS=${RUNNER_OS:-} RUNNER_ARCH=${RUNNER_ARCH:-}"
      echo "GITHUB_REPOSITORY=${GITHUB_REPOSITORY:-} GITHUB_REF=${GITHUB_REF:-} GITHUB_SHA=${GITHUB_SHA:-}"
      echo "GITHUB_RUN_ID=${GITHUB_RUN_ID:-} GITHUB_RUN_ATTEMPT=${GITHUB_RUN_ATTEMPT:-} GITHUB_WORKFLOW=${GITHUB_WORKFLOW:-}"
      echo "case=$CASE kind=$KIND user=$U uid=$(/usr/bin/id -u) gid=$(/usr/bin/id -g) home=$RHOME tmpdir=$UTMP"
    } > "$d/system.txt" 2>&1
    SESSION=$(/bin/launchctl managername 2>&1 | /usr/bin/head -1)
    { echo "launchctl_managername=$SESSION"
      echo "console_owner=$(/usr/bin/stat -f %Su /dev/console 2>&1)"
      /bin/launchctl print "gui/$(/usr/bin/id -u)" > /dev/null 2>&1
      echo "launchctl_print_gui_exit=$?"
      echo "-- who"; /usr/bin/who
      echo "-- session processes (pid user executable)"
      /bin/ps -axww -o pid=,user=,comm= | /usr/bin/grep -E 'WindowServer|loginwindow|/Dock\.app/|/Finder\.app/'
    } > "$d/gui-session.txt" 2>&1
    { echo "spctl_status=$(/usr/sbin/spctl --status 2>&1)"
      echo "csrutil=$(/usr/bin/csrutil status 2>&1)"
      echo "xcode_select=$(/usr/bin/xcode-select -p 2>&1)"
      echo "stapler=$(clean /usr/bin/xcrun --find stapler 2>&1)"
      echo "application_firewall=$(/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate 2>&1)"
    } > "$d/security.txt" 2>&1
    /bin/df -k /Users "$OUTROOT" > "$d/df.txt" 2>&1
    home_facts > "$d/real-home-before.txt" 2>&1
}

arch_gate() {   # from the machine itself, never from the job label
    local m a t
    m=$(/usr/bin/uname -m); a=$(/usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null)
    t=$(/usr/sbin/sysctl -n sysctl.proc_translated 2>/dev/null)
    if [ "$m" = arm64 ] && [ "$a" = 1 ] && [ "$t" = 0 ]; then
        gate runner-arm64 PASS arm64 "uname -m $m, hw.optional.arm64 $a, proc_translated $t; image ${ImageOS:-?} ${ImageVersion:-?}; macOS $(/usr/bin/sw_vers -productVersion 2>&1); session $SESSION; ncpu $(/usr/sbin/sysctl -n hw.ncpu 2>&1), memsize $(/usr/sbin/sysctl -n hw.memsize 2>&1)"
    else gate runner-arm64 FAIL not_arm64 "uname -m $m, hw.optional.arm64 ${a:-unread}, proc_translated ${t:-unread}"; fi
}

disk_gate() {
    local f
    pending disk-start || return 1
    f=$(free_kib /Users)
    case "$f" in ''|*[!0-9]*) gate disk-start UNKNOWN disk_unread "df -k /Users: ${f:-nothing}"; return 1 ;; esac
    if [ "$f" -ge "$START_FREE_KIB" ]; then gate disk-start PASS enough_free "free $f KiB (>= $START_FREE_KIB)"
    else gate disk-start FAIL low_disk "free $f KiB (< $START_FREE_KIB)"; fi
}

overlay_gate() {   # the checkout holds exactly the authored overlay, and the placed workflow is its copy
    pending overlay-integrity || return 1
    if (cd "$OVL" && /usr/bin/shasum -a 256 -c SHA256SUMS) > "$OUT/checks/overlay-sha256.txt" 2>&1 \
        && (cd "$HERE" && /usr/bin/shasum -a 256 -c OVERLAY-SHA256) >> "$OUT/checks/overlay-sha256.txt" 2>&1 \
        && /usr/bin/cmp "$OVL/workflow/qualification-update-native-macos.yml" "$W/$WORKFLOW" >> "$OUT/checks/overlay-sha256.txt" 2>&1
    then gate overlay-integrity PASS verified "SHA256SUMS and OVERLAY-SHA256 hold; $WORKFLOW equals the overlay copy"
    else procedure_error overlay-integrity "the checkout does not hold the authored overlay (checks/overlay-sha256.txt)"; fi
}

git_gate() {   # the runner's own Command Line Tools git (qb1 reaches it through its fixed CLT path, linked below)
    local v=none
    pending clt-git || return 1
    if [ -x /Library/Developer/CommandLineTools/usr/bin/git ]; then CLTSRC=/Library/Developer/CommandLineTools
    else CLTSRC=$(/usr/bin/xcode-select -p 2>/dev/null); fi
    HGIT=$CLTSRC/usr/bin/git
    if [ -n "$CLTSRC" ] && [ -x "$HGIT" ]; then v=$("$HGIT" --version 2>&1) || v=none; fi
    echo "clt_source=${CLTSRC:-none} git=$HGIT version=$v" > "$OUT/checks/git.txt"
    if [ "$v" != none ]; then gate clt-git PASS runner_git "$(/bin/cat "$OUT/checks/git.txt")"
    else gate clt-git FAIL clt_git_missing "no working git under ${CLTSRC:-xcode-select -p}"; fi
}

pin_gate() {   # C by commit and tree; the execution commit has exactly one parent, C, and only adds exactly the public
               # inventory: SHA256SUMS, every file it lists, and the placed workflow. The parent checks the same before
               # its push; this repeats it on the commit that actually ran.
    local d="$OUT/product" head t anc parents drc changed outside
    pending execution-pin || return 1
    [ -n "$ESHA" ] || { procedure_error execution-pin "GITHUB_SHA is unset"; return 1; }
    head=$(g0 rev-parse -q --verify 'HEAD^{commit}' 2>/dev/null)
    t=$(g0 rev-parse -q --verify "$C^{tree}" 2>/dev/null)
    g0 merge-base --is-ancestor "$C" "$ESHA" > /dev/null 2>&1; anc=$?
    parents=$(g0 rev-list --parents -n 1 "$ESHA" 2>/dev/null)
    g0 diff --no-renames --name-status "$C" "$ESHA" > "$d/execution-diff.txt" 2> "$d/execution-diff.stderr"; drc=$?
    { echo "$WORKFLOW"; echo ci-macos-native/SHA256SUMS
      /usr/bin/sed -n 's|^[0-9a-f]\{64\}  |ci-macos-native/|p' "$OVL/SHA256SUMS"; } | /usr/bin/sort > "$d/inventory-expected.txt"
    /usr/bin/awk -F '\t' '{print $2}' "$d/execution-diff.txt" | /usr/bin/sort > "$d/inventory-changed.txt"
    changed=$(/usr/bin/grep -v '^A[[:space:]]' "$d/execution-diff.txt")
    outside=$(/usr/bin/diff "$d/inventory-expected.txt" "$d/inventory-changed.txt")
    { echo "execution_sha=$ESHA checkout_head=${head:-unread} parents: ${parents:-unread}"
      echo "C=$C tree=${t:-unread} expected_tree=$CT ancestor_exit=$anc diff_exit=$drc"
      echo "commits C..execution: $(g0 rev-list --count "$C..$ESHA" 2>&1)"
      echo "changes other than additions: $(oneline "$changed")"
      echo "inventory (expected <, changed >): $(oneline "$outside")"; } > "$d/pins.txt"
    if [ "$head" = "$ESHA" ] && [ "$t" = "$CT" ] && [ "$anc" = 0 ] && [ "$parents" = "$ESHA $C" ] && [ "$drc" = 0 ] \
        && [ -z "$changed" ] && [ -z "$outside" ] && [ -s "$d/inventory-changed.txt" ]; then
        gate execution-pin PASS c_plus_inventory "execution $ESHA: one parent, C $C (tree $CT); it adds exactly the $(/usr/bin/grep -c . "$d/inventory-expected.txt") inventory paths"
    else gate execution-pin FAIL not_c_plus_inventory "$(oneline "$(/bin/cat "$d/pins.txt")")"; fi
}

fixture_gate() {   # /Users/anka must not exist; it is made once (sudo -n mkdir), handed to this user, then filled.
                   # Ownership is this run's own successful mkdir (OWNED), never inferred later from a pathname.
    local d="$OUT/checks/fixture.txt" p
    pending fixture-root || return 1
    if [ -e "$GH" ] || [ -L "$GH" ]; then gate fixture-root REFUSED fixture_root_exists "$GH exists; nothing is created in, read under or uploaded from it"; return 1; fi
    /usr/bin/sudo -n /bin/mkdir -m 0700 "$GH" > "$d" 2>&1 \
        || { gate fixture-root REFUSED cannot_create_fixture_root "sudo -n mkdir $GH failed (checks/fixture.txt)"; return 1; }
    OWNED=1
    /usr/bin/sudo -n /usr/sbin/chown "$(/usr/bin/id -u):$(/usr/bin/id -g)" "$GH" >> "$d" 2>&1 \
        || { gate fixture-root FAIL fixture_root_not_owned "sudo -n chown of $GH failed (checks/fixture.txt)"; return 1; }
    [ "$(/usr/bin/stat -f %u "$GH")" = "$(/usr/bin/id -u)" ] || { gate fixture-root FAIL fixture_root_not_owned "$(/bin/ls -ld "$GH" 2>&1)"; return 1; }
    (set -C; : > "$GH/.bash_profile") 2>> "$d" || { gate fixture-root FAIL fixture_create_failed "$GH/.bash_profile"; return 1; }
    for p in "$ROOT" "$ROOT/packages" "$ROOT/pkg" "$ROOT/pkg/7.8.0" "$ROOT/q7b4" "$ROOT/tools" "$ROOT/tools/clt-26.6-r2" \
             "$ROOT/tools/clt-26.6-r2/Payload" "$ROOT/tools/clt-26.6-r2/Payload/Library" \
             "$ROOT/tools/clt-26.6-r2/Payload/Library/Developer" "$Q" "$Q/bin" "$Q/pkg"; do
        /bin/mkdir -m 0700 "$p" 2>> "$d" || { gate fixture-root FAIL fixture_create_failed "$p"; return 1; }
    done
    /bin/ln -s "$CLTSRC" "$CLT" 2>> "$d" || { gate fixture-root FAIL fixture_create_failed "$CLT"; return 1; }
    # qb1's evidence folder: a link into the run's own output root, so the live evidence is uploaded with it
    /bin/ln -s "$OUT/guest-evidence" "$EV" 2>> "$d" || { gate fixture-root FAIL fixture_create_failed "$EV"; return 1; }
    { /bin/ls -ld "$GH" "$GH/.bash_profile"; /bin/ls -la "$ROOT" "$Q"; /bin/ls -l "$CLT"; } >> "$d" 2>&1
    gate fixture-root PASS created "$GH made for $U (create-only); empty .bash_profile; CLT link -> $CLTSRC; namespace $Q; evidence link -> $OUT/guest-evidence"
}

fetch() {   # fetch NAME URL DEST BYTES SHA256: create-only download; size and digest hold before DEST exists
    local name="$1" url="$2" dest="$3" bytes="$4" sha="$5" part size got rc
    part="$(/usr/bin/dirname "$dest")/.$(/usr/bin/basename "$dest").part"
    if [ -e "$dest" ] || [ -e "$part" ]; then gate "input-$name" REFUSED input_exists "$dest"; return 1; fi
    clean /usr/bin/curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' --tlsv1.2 \
        --retry 3 --retry-delay 5 --connect-timeout 30 --max-time 1800 \
        -w 'http_code=%{http_code} redirects=%{num_redirects} bytes=%{size_download} seconds=%{time_total}\n' \
        -o "$part" "$url" > "$OUT/inputs/$name.curl.txt" 2>&1 &
    wait "$!"; rc=$?   # waited on in the background: a stop signal is handled at once (on_signal)
    echo "curl_exit=$rc url=$url" >> "$OUT/inputs/$name.curl.txt"
    [ "$rc" = 0 ] || { gate "input-$name" UNKNOWN download_failed "curl exit $rc for $url"; return 1; }
    size=$(/usr/bin/stat -f %z "$part" 2>/dev/null); got=$(/usr/bin/shasum -a 256 "$part" | /usr/bin/cut -c1-64)
    echo "bytes=$size sha256=$got expected_bytes=$bytes expected_sha256=$sha" > "$OUT/inputs/$name.identity.txt"
    if [ "$size" != "$bytes" ] || [ "$got" != "$sha" ]; then gate "input-$name" FAIL digest_mismatch "bytes $size sha256 $got"; return 1; fi
    /bin/mv -n "$part" "$dest" && [ -f "$dest" ] && [ ! -e "$part" ] \
        || { gate "input-$name" UNKNOWN cannot_place_input "$dest"; return 1; }
    gate "input-$name" PASS pinned "$bytes bytes, sha256 $sha"
}

inputs_gate() {   # 7.8.0 (the observer, and the release C's Update stages) and 7.7.0 (the owner's existing app)
    pending input-sums-7.8.0 || return 1
    fetch sums-7.8.0 "$REL780/SHA256SUMS" "$OUT/inputs/SHA256SUMS-7.8.0" "$SUMS780_BYTES" "$SUMS780_SHA" || return 1
    if /usr/bin/grep -q -x -F "$DMG780_SHA  Ouroboros-7.8.0.dmg" "$OUT/inputs/SHA256SUMS-7.8.0"; then
        gate sums-names-dmg-7.8.0 PASS listed "SHA256SUMS lists Ouroboros-7.8.0.dmg as $DMG780_SHA"
    else gate sums-names-dmg-7.8.0 FAIL sums_line_missing "no exact Ouroboros-7.8.0.dmg line"; return 1; fi
    fetch dmg-7.8.0 "$REL780/Ouroboros-7.8.0.dmg" "$ROOT/packages/Ouroboros-7.8.0.dmg" "$DMG780_BYTES" "$DMG780_SHA" || return 1
    fetch dmg-7.7.0 "$REL770/Ouroboros-7.7.0.dmg" "$ROOT/packages/Ouroboros-7.7.0.dmg" "$DMG770_BYTES" "$DMG770_SHA"
}

capsule_770() {   # release-capsule-7.7.0: 7.7.0's own published SHA256SUMS (its bytes are not pinned) names the pinned
                  # DMG. C's Update proves the installed app's origin against exactly that capsule, so a capsule that
                  # names another digest is FAIL; one that is unreadable or names none is UNKNOWN. Not a gate: C's own
                  # preparation is left to show what it does with it.
    local dest="$OUT/inputs/SHA256SUMS-7.7.0" part="$OUT/inputs/.SHA256SUMS-7.7.0.part" rc size got line
    pending release-capsule-7.7.0 || return 1
    clean /usr/bin/curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' --tlsv1.2 \
        --retry 3 --retry-delay 5 --connect-timeout 30 --max-time 300 \
        -w 'http_code=%{http_code} redirects=%{num_redirects} bytes=%{size_download} seconds=%{time_total}\n' \
        -o "$part" "$REL770/SHA256SUMS" > "$OUT/inputs/sums-7.7.0.curl.txt" 2>&1 &
    wait "$!"; rc=$?
    echo "curl_exit=$rc url=$REL770/SHA256SUMS" >> "$OUT/inputs/sums-7.7.0.curl.txt"
    if [ "$rc" != 0 ] || ! /bin/mv -n "$part" "$dest" || [ ! -f "$dest" ]; then
        claim release-capsule-7.7.0 UNKNOWN capsule_unread "curl exit $rc for $REL770/SHA256SUMS"; return 0
    fi
    size=$(/usr/bin/stat -f %z "$dest" 2>/dev/null); got=$(/usr/bin/shasum -a 256 "$dest" | /usr/bin/cut -c1-64)
    echo "bytes=$size sha256=$got (not pinned)" > "$OUT/inputs/sums-7.7.0.identity.txt"
    line=$(/usr/bin/grep -E '^[0-9a-f]{64}  Ouroboros-7\.7\.0\.dmg$' "$dest")
    if /usr/bin/grep -q -x -F "$DMG770_SHA  Ouroboros-7.7.0.dmg" "$dest"; then
        claim release-capsule-7.7.0 PASS capsule_names_pinned_dmg "$size bytes, sha256 $got (recorded); it lists Ouroboros-7.7.0.dmg as $DMG770_SHA"
    elif [ -n "$line" ]; then claim release-capsule-7.7.0 FAIL capsule_names_other_dmg "$(oneline "$line")"
    else claim release-capsule-7.7.0 UNKNOWN capsule_lacks_dmg "$size bytes, sha256 $got: no Ouroboros-7.7.0.dmg line"; fi
}

observer_gate() {   # qb1's observer interpreter at its fixed path: a whole, verified copy of the pinned 7.8.0 app (not the
                    # run's); its code directory hash is also the reference of the official 7.8.0 app (native-target)
    local mnt="$WORK/mnt-observer" dst="$OFFICIAL780" d="$OUT/checks" ra rx rd rs py
    pending observer || return 1
    /bin/mkdir "$mnt" || { procedure_error observer "cannot make $mnt"; return 1; }
    clean /usr/bin/hdiutil attach -readonly -nobrowse -noautoopen -mountpoint "$mnt" "$ROOT/packages/Ouroboros-7.8.0.dmg" \
        > "$d/observer-attach.txt" 2>&1; ra=$?
    [ "$ra" = 0 ] || { gate observer UNKNOWN dmg_attach_failed "hdiutil attach exit $ra"; return 1; }
    clean /usr/bin/ditto "$mnt/Ouroboros.app" "$dst" > "$d/observer-ditto.txt" 2>&1; rx=$?
    clean /usr/bin/hdiutil detach "$mnt" > "$d/observer-detach.txt" 2>&1; rd=$?
    clean /usr/bin/codesign --verify --deep --strict "$dst" > "$d/observer-codesign.txt" 2>&1; rs=$?
    py=$("$OBSPY" -I -S -B -c 'import sys, select; print(sys.version.split()[0], hasattr(select, "kqueue"))' 2>&1)
    echo "ditto=$rx detach=$rd codesign=$rs python=$py" > "$d/observer.txt"
    if [ "$rx" = 0 ] && [ "$rs" = 0 ] && [ "${py##* }" = True ]; then
        gate observer PASS verified "Python $py; ditto and codesign --deep --strict ok; detach exit $rd"
    else gate observer FAIL observer_invalid "ditto $rx, codesign $rs, python ${py:-none}"; fi
}

wheel_pins_gate() {   # the declared wheel set (wheelhouse.pins) is what wheelhouse.py derives from C's own lock and uv.lock
                      # in this checkout, byte for byte, and that lock is C's
    local d="$OUT/checks/wheelhouse" lock rc
    pending wheel-pins || return 1
    /bin/mkdir "$d" || { procedure_error wheel-pins "cannot make $d"; return 1; }
    lock=$(/usr/bin/shasum -a 256 "$W/requirements-runtime.lock" 2>/dev/null | /usr/bin/cut -c1-64)
    "$OBSPY" -I -S -B "$HERE/wheelhouse.py" check "$W/requirements-runtime.lock" "$W/uv.lock" "$HERE/wheelhouse.pins" \
        > "$d/pins-check.txt" 2>&1; rc=$?
    echo "lock_sha256=${lock:-unread} uv_lock_sha256=$(/usr/bin/shasum -a 256 "$W/uv.lock" 2>/dev/null | /usr/bin/cut -c1-64) check_exit=$rc" >> "$d/pins-check.txt"
    if [ "$rc" = 0 ] && [ "$lock" = "$CLOCK_PIN" ]; then
        gate wheel-pins PASS reproduced "$(/usr/bin/grep -c -v '^#' "$HERE/wheelhouse.pins") wheels derived again from C's requirements-runtime.lock ($lock) and uv.lock"
    else gate wheel-pins FAIL pins_not_reproduced "$(oneline "$(/bin/cat "$d/pins-check.txt")")"; fi
}

wheelhouse_gate() {   # the public wheel folder qb1's venvprep reads at its fixed path, create-only: each pinned wheel from
                      # PyPI by its size and sha256 before its final name exists, then wheelhouse.py report writes the two
                      # files qb1 reads (lock_sha256 for wheels_match, SHA256SUMS)
    local d="$OUT/checks/wheelhouse" name version file size sha url part got rc n=0 tab
    pending wheelhouse || return 1
    /bin/mkdir -m 0755 "$WHEELS" 2>> "$d/stage.txt" || { gate wheelhouse REFUSED wheelhouse_exists "$WHEELS"; return 1; }
    tab=$(printf '\t')
    while IFS="$tab" read -r name version file size sha url; do
        case "$name" in '#'*|'') continue ;; esac
        part="$WHEELS/.$file.part"
        clean /usr/bin/curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' --tlsv1.2 \
            --retry 3 --retry-delay 5 --connect-timeout 30 --max-time 600 \
            -w "$file http_code=%{http_code} redirects=%{num_redirects} bytes=%{size_download} seconds=%{time_total}\n" \
            -o "$part" "$url" >> "$d/fetch.txt" 2>&1 &
        wait "$!"; rc=$?
        [ "$rc" = 0 ] || { gate wheelhouse UNKNOWN download_failed "curl exit $rc for $file ($n placed before it)"; return 1; }
        got=$(/usr/bin/shasum -a 256 "$part" | /usr/bin/cut -c1-64)
        if [ "$(/usr/bin/stat -f %z "$part")" != "$size" ] || [ "$got" != "$sha" ]; then
            gate wheelhouse FAIL digest_mismatch "$file: sha256 $got, pinned $sha"; return 1
        fi
        /bin/mv -n "$part" "$WHEELS/$file" && [ ! -e "$part" ] || { gate wheelhouse UNKNOWN cannot_place_input "$file"; return 1; }
        n=$((n + 1))
    done < "$HERE/wheelhouse.pins"
    if "$OBSPY" -I -S -B "$HERE/wheelhouse.py" report "$HERE/wheelhouse.pins" "$W/requirements-runtime.lock" "$WHEELS" \
        > "$d/report.txt" 2>&1; then
        /bin/cp "$WHEELS/SHA256SUMS" "$d/SHA256SUMS"; /bin/cp "$WHEELS/wheelhouse-report.json" "$d/wheelhouse-report.json"
        gate wheelhouse PASS staged "$n pinned PyPI wheels at $WHEELS; $(oneline "$(/bin/cat "$d/report.txt")")"
    else gate wheelhouse FAIL wheelhouse_invalid "$(oneline "$(/bin/cat "$d/report.txt")")"; fi
}

bundle_gate() {   # mkbundle.sh, unchanged, on the checkout: C's bundle and bind file (bundle bytes are not pinned)
    local mk="$WORK/mkbundle" d="$OUT/product" rc bc bt bl bn bs got
    pending product-bundle || return 1
    /usr/bin/env -i PATH="$CLTSRC/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin" HOME="$RHOME" TMPDIR="$UTMP" \
        OUROBOROS_QA_PRODUCT_REPO="$W" /bin/bash "$HERE/equipment/host/mkbundle.sh" commit "$C" "$mk" \
        > "$d/mkbundle.stdout" 2> "$d/mkbundle.stderr"
    rc=$?
    [ "$rc" = 0 ] || { gate product-bundle FAIL mkbundle_failed "mkbundle.sh exit $rc (product/mkbundle.stderr)"; return 1; }
    /bin/cp "$mk/bind.env" "$mk/mkbundle.json" "$d/"; /bin/cp "$mk"/cand-*.bundle.verify.txt "$d/" 2>/dev/null
    bc=$(/usr/bin/sed -n 's/^CAND=//p' "$mk/bind.env"); bt=$(/usr/bin/sed -n 's/^CTREE=//p' "$mk/bind.env")
    bl=$(/usr/bin/sed -n 's/^CLOCK=//p' "$mk/bind.env"); bn=$(/usr/bin/sed -n 's/^CBUNDLE_NAME=//p' "$mk/bind.env")
    bs=$(/usr/bin/sed -n 's/^CBUNDLE_SHA=//p' "$mk/bind.env")
    got=$(/usr/bin/shasum -a 256 "$mk/$bn" 2>/dev/null | /usr/bin/cut -c1-64)
    if [ "$bc" = "$C" ] && [ "$bt" = "$CT" ] && [ "$bl" = "$CLOCK_PIN" ] && [ "$bn" = "cand-${CT:0:8}.bundle" ] \
        && [ -n "$bs" ] && [ "$got" = "$bs" ]; then
        BUNDLE=$bn
        gate product-bundle PASS bound "CAND $bc, tree $bt, lock $bl; $bn sha256 $bs (built here)"
    else gate product-bundle FAIL bundle_bind_mismatch "CAND ${bc:-?} CTREE ${bt:-?} CLOCK ${bl:-?} name ${bn:-?} sha ${bs:-?}/${got:-?}"; fi
}

equip_gate() {   # the namespace's bin and pkg, create-only; their digests are re-checked before every step
    local f
    pending equipment || return 1
    for f in qb1.sh procwatch.py venv_lab.py; do
        [ ! -e "$Q/bin/$f" ] && /bin/cp "$HERE/equipment/guest/$f" "$Q/bin/$f" \
            || { gate equipment FAIL equipment_copy_failed "$Q/bin/$f"; return 1; }
    done
    for f in bind.env "$BUNDLE"; do
        [ ! -e "$Q/pkg/$f" ] && /bin/cp "$WORK/mkbundle/$f" "$Q/pkg/$f" \
            || { gate equipment FAIL equipment_copy_failed "$Q/pkg/$f"; return 1; }
    done
    (cd "$Q" && /usr/bin/shasum -a 256 bin/qb1.sh bin/procwatch.py bin/venv_lab.py pkg/bind.env "pkg/$BUNDLE") \
        > "$WORK/expected-bin-pkg.sha256" || { gate equipment FAIL equipment_copy_failed "digest list"; return 1; }
    /bin/cp "$WORK/expected-bin-pkg.sha256" "$OUT/checks/"
    for f in qb1.sh procwatch.py venv_lab.py; do
        /usr/bin/cmp -s "$Q/bin/$f" "$HERE/equipment/guest/$f" || { gate equipment FAIL equipment_copy_failed "$Q/bin/$f differs from the overlay"; return 1; }
    done
    gate equipment PASS placed "$(oneline "$(/bin/cat "$WORK/expected-bin-pkg.sha256")")"
}

name_of() {   # name_of CMD VERDICT DETAIL EVDIR: a short public name for a step's verdict
    local o
    if [ "$2" = PASS ]; then echo pass; return; fi
    case "$1" in
        install) # the declared known negative, only in its exact measured shape (qb1 install's own detail and spctl text)
                 if [ "$2" = FAIL ] && [ "$3" = "ditto=0 codesign=0 spctl=3 version=7.7.0" ] \
                     && /usr/bin/grep -q -x 'source=Unnotarized Developer ID' "$4/spctl.txt" 2>/dev/null \
                     && /usr/bin/grep -q ': rejected$' "$4/spctl.txt" 2>/dev/null; then echo known_negative_unnotarized
                 else echo "install_$(lower "$2")"; fi ;;
        openw) case "$3" in
                   *"the open exited"*) echo launchservices_open_failed ;;
                   *"nothing of"*) echo app_not_running_after_open ;;
                   *"running but not serving"*) echo not_serving ;;
                   *"procwatch exit"*) echo exit_observer_incomplete ;;
                   *) echo "open_$(lower "$2")" ;;
               esac ;;
        quit) o=$(/bin/cat "$4/osascript.txt" 2>/dev/null)
              case "$o" in
                  *-1743*) echo tcc_automation_refused ;;
                  *-1712*) echo apple_event_timed_out ;;
                  *-600*) echo apple_event_target_not_running ;;
                  *) if [ "$2" = FAIL ]; then echo quit_not_cold; else echo quit_absence_unknown; fi ;;
              esac ;;
        lscanary) case "$3" in
                      *"no private WebKit store"*) echo no_private_webkit_store ;;
                      *"no localStorage database"*) echo no_localstorage_db ;;
                      *) echo "canary_$(lower "$2")" ;;
                  esac ;;
        localfn) case "$3" in
                     *"package environment"*) echo package_env_not_established ;;
                     *"serving server processes"*) echo server_count_not_one ;;
                     *) echo "localfn_$(lower "$2")" ;;
                 esac ;;
        prepare) case "$3" in   # the owner's work changed first: qb1 keeps it FAIL beside any other outcome
                     *"H/I/W/U changed"*) echo local_work_changed ;;
                     *"evidence is incomplete"*) echo evidence_incomplete ;;
                     *"procwatch exit"*) echo exit_observer_incomplete ;;
                     *"was not admitted"*) echo prepare_not_admitted ;;
                     *"no new candidate settled"*) echo not_settled ;;
                     *"no serving server was identified"*) echo serving_not_identified ;;
                     *"process table was unreadable"*) echo process_table_unread ;;
                     *"serving server changed"*) echo serving_changed ;;
                     *"were not read completely"*) echo local_work_unread ;;
                     *"not the required"*) echo tier_mismatch ;;
                     *"a prerequisite of the checks is missing"*) echo prerequisite_unverified ;;   # e.g. no usable Node
                     *) echo "not_ready_$(lower "$2")" ;;
                 esac ;;
        activate) case "$3" in   # the owner's work changed first: qb1 keeps it FAIL beside any other outcome
                      *"bytes changed"*) echo local_work_changed ;;
                      *"procwatch exit"*) echo exit_observer_incomplete ;;
                      *"answered http="*) echo activation_refused ;;
                      *"no adoption id"*) echo no_adoption_id ;;
                      *"did not close within"*) echo adoption_not_closed ;;
                      *"no launcher and server serve"*) echo not_serving_after_adoption ;;
                      *"did not name"*) echo answer_not_activating ;;
                      *"not retained"*) echo activation_record_missing ;;
                      *"update_activation.json retains"*) echo activation_record_differs ;;
                      *"the same process"*) echo same_server ;;
                      *"were not read completely"*) echo local_work_unread ;;
                      *"does not contain the previous"*) echo head_not_contained ;;
                      *"process table was unreadable"*) echo process_table_unread ;;
                      *" closed "*) echo adoption_closed_otherwise ;;
                      *) echo "activate_$(lower "$2")" ;;
                  esac ;;
        *) echo "${1}_$(lower "$2")" ;;
    esac
}

step() {   # step KIND NAME WINDOW EVDIR QB1_ARGS...: one qb1 step through capture.py (KIND gate|claim|rec)
    local kind="$1" name="$2" win="$3" evd="$4" deadline left f label rc v reason detail vf
    shift 4
    STEP_RAN=0
    if [ -z "$FINISHING" ]; then pending "$name" || return 1; deadline=$((T0 + BUDGET_S - RESERVE_S))
    else deadline=$((T0 + BUDGET_S)); fi
    left=$((deadline - $(/bin/date +%s)))
    if [ "$left" -lt "$win" ]; then   # never shortened: a window that does not fit is not started
        claim "$name" UNKNOWN time_budget "its ${win}s window does not fit the ${left}s left of the budget; not started"
        stop_chain time_budget; return 1
    fi
    f=$(free_kib /Users)
    case "$f" in ''|*[!0-9]*) claim "$name" UNKNOWN disk_unread "df -k /Users"; stop_chain disk_unread; return 1 ;; esac
    if [ "$f" -lt "$STEP_FREE_KIB" ]; then claim "$name" UNKNOWN low_disk "free $f KiB"; stop_chain low_disk; return 1; fi
    if ! (cd "$Q" && /usr/bin/shasum -a 256 -c "$WORK/expected-bin-pkg.sha256") > /dev/null 2>&1; then
        procedure_error equipment-changed "bin or pkg differ from expected-bin-pkg.sha256 before $name"; return 1
    fi
    SEQ=$((SEQ + 1)); label=$(printf '%02d-%s' "$SEQ" "$name")
    space "before $label"
    STEP_RAN=1
    LAB_EVIDENCE="$OUT/receipts" "$OBSPY" -I -S -B "$HERE/equipment/scripts/capture.py" "$label" "$win" \
        /usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME="$GH" USER="$U" LOGNAME="$U" SHELL=/bin/bash TMPDIR="$UTMP" \
        /bin/bash "$Q/bin/qb1.sh" "$@" > "$OUT/receipts/$label.console" 2>&1 &
    wait "$!"; rc=$?   # waited on in the background: a stop signal is handled at once; the step is never signalled
    space "after $label exit=$rc"
    vf="$evd/verdict.txt"
    if [ "$rc" = 124 ]; then v=UNKNOWN; reason=capture_timeout; detail="the ${win}s window ended the client; the step's outcome is unknown"
    elif [ "$rc" = 125 ]; then v=UNKNOWN; reason=capture_not_started; detail="capture.py could not start the step"
    elif [ "$rc" = 2 ] && [ ! -f "$vf" ]; then
        detail=$(/usr/bin/tail -1 "$OUT/receipts/$label.stderr" 2>/dev/null)
        case "$detail" in   # only qb1's own refusal line (die, before the step's effect) is a pre-effect receipt
            "qb1: refused before effect: "*) v=REFUSED; reason=refused_before_effect ;;
            *) v=UNKNOWN; reason=step_aborted
               detail="exit 2 without a verdict or qb1's pre-effect refusal line, so effects are not excluded: ${detail:-no stderr}" ;;
        esac
    elif [ "$kind" = rec ]; then
        if [ "$rc" = 0 ]; then v=RECORDED; reason=recorded; else v=FAIL; reason=recorder_failed; fi
        detail="exit $rc; $evd"
    elif [ -f "$vf" ]; then
        v=$(/usr/bin/sed -n '1s/^verdict=\([A-Z]*\) .*/\1/p' "$vf"); detail=$(/usr/bin/sed -n '1s/.* detail=//p' "$vf")
        if [ "$rc:$v" = 0:PASS ] || [ "$rc:$v" = 1:FAIL ] || [ "$rc:$v" = 3:UNKNOWN ]; then reason=$(name_of "$1" "$v" "$detail" "$evd")
        else reason=verdict_exit_mismatch; detail="exit $rc, verdict ${v:-none}: $detail"; v=UNKNOWN; fi
    else v=UNKNOWN; reason=verdict_missing; detail="exit $rc without $vf"; fi
    case "$1:$v" in openw:FAIL|openw:UNKNOWN|openw:REFUSED|quit:FAIL|quit:UNKNOWN) detail="$detail; session $SESSION" ;; esac
    if [ "$kind" = gate ]; then gate "$name" "$v" "$reason" "$detail"; else claim "$name" "$v" "$reason" "$detail"; fi
    [ "$v" = PASS ] || [ "$v" = RECORDED ]
}

existing_copy() {   # existing-copy: the official 7.7.0 app, copied whole from the pinned DMG, is the owner's existing
                    # installation. Its fresh copy's Gatekeeper assessment stays the install claim's measured verdict;
                    # only that declared known negative (Unnotarized Developer ID) or an acceptance lets the copy go on,
                    # and then only through ordinary LaunchServices: never a fresh-install PASS
    local d="$EV/$CASE-install" s why="" gk
    pending existing-copy || return 1
    [ -f "$d/verdict.txt" ] || { gate existing-copy UNKNOWN install_unread "no install verdict in $d"; return 1; }
    /usr/bin/grep -q -x 'exit_code=0' "$d/ditto.txt" 2>/dev/null || why="$why ditto did not exit 0;"
    /usr/bin/grep -q -x 'exit_code=0' "$d/codesign.txt" 2>/dev/null || why="$why codesign --verify --deep --strict did not exit 0;"
    /usr/bin/grep -q "^$DMG770_SHA " "$d/dmg-identity.txt" 2>/dev/null || why="$why the image is not the pinned 7.7.0 DMG;"
    [ "$(/usr/bin/plutil -extract CFBundleShortVersionString raw "$APP/Contents/Info.plist" 2>/dev/null)" = 7.7.0 ] \
        || why="$why the copy is not version 7.7.0;"
    s=$(/usr/bin/sed -n 's/^exit_code=//p' "$d/spctl.txt" 2>/dev/null | /usr/bin/tail -1)
    if [ "$s" = 0 ]; then gk="Gatekeeper accepts this copy (the known negative did not reproduce)"
    elif [ "$s" = 3 ] && /usr/bin/grep -q -x 'source=Unnotarized Developer ID' "$d/spctl.txt" \
        && /usr/bin/grep -q ': rejected$' "$d/spctl.txt"; then
        gk="Gatekeeper rejects this fresh copy as Unnotarized Developer ID (the declared known negative, install/spctl.txt)"
    else gk="Gatekeeper answers otherwise (exit ${s:-unread})"; why="$why $gk;"; fi
    if [ -z "$why" ]; then gate existing-copy PASS official_existing_copy "the official 7.7.0 app copied whole from the pinned DMG (ditto 0, codesign --deep --strict 0, version 7.7.0); $gk; it is used only as an existing copy opened through LaunchServices, not as a fresh install"
    else gate existing-copy FAIL not_the_official_copy "$why"; fi
}

app_identity() {   # app_identity APP FILE: one bundle's version, bundle id, folder identity, code directory hash, team and
                   # strict signature (reads only)
    { echo "path=$1"
      echo "version=$(/usr/bin/plutil -extract CFBundleShortVersionString raw "$1/Contents/Info.plist" 2>&1)"
      echo "bundle_id=$(/usr/bin/plutil -extract CFBundleIdentifier raw "$1/Contents/Info.plist" 2>&1)"
      echo "folder=$(/usr/bin/stat -f 'device=%d inode=%i' "$1" 2>&1) contents=$(/usr/bin/stat -f 'inode=%i' "$1/Contents" 2>&1)"
      /usr/bin/codesign -dv --verbose=4 "$1" 2>&1 | /usr/bin/grep -E '^(Identifier|CDHash|TeamIdentifier)='
      /usr/bin/codesign --verify --deep --strict "$1" > /dev/null 2>&1; echo "codesign_strict=$?"
    } > "$2" 2>&1
}

native_target() {   # native-target: after the activation, the app at the installed path is the official 7.8.0 app: the
                    # same code directory hash and team as the whole copy of the pinned 7.8.0 DMG's app, a strict
                    # signature, and Gatekeeper's acceptance as Notarized Developer ID (the release smoke's own check);
                    # before it, that path held 7.7.0. The folder identities, the previous app's backup and the staging
                    # folders are recorded, not judged.
    local d="$OUT/checks/native" bv av ah oh at ot ac s why="" unk=""
    pending native-target || return 1
    app_identity "$APP" "$d/after.txt"
    app_identity "$OFFICIAL780" "$d/official-7.8.0.txt"
    clean /usr/sbin/spctl --assess --type exec -vv "$APP" > "$d/spctl-after.txt" 2>&1; s=$?
    /bin/ls -la "$R/Applications" > "$d/applications-after.txt" 2>&1
    bv=$(/usr/bin/sed -n 's/^version=//p' "$d/before.txt" 2>/dev/null); av=$(/usr/bin/sed -n 's/^version=//p' "$d/after.txt")
    ah=$(/usr/bin/sed -n 's/^CDHash=//p' "$d/after.txt"); oh=$(/usr/bin/sed -n 's/^CDHash=//p' "$d/official-7.8.0.txt")
    at=$(/usr/bin/sed -n 's/^TeamIdentifier=//p' "$d/after.txt"); ot=$(/usr/bin/sed -n 's/^TeamIdentifier=//p' "$d/official-7.8.0.txt")
    ac=$(/usr/bin/sed -n 's/^codesign_strict=//p' "$d/after.txt")
    [ "$bv" = 7.7.0 ] || unk="$unk the version before the preparation reads ${bv:-unread};"
    if [ -z "$av" ] || [ -z "$ah" ] || [ -z "$oh" ]; then unk="$unk the app's version or a code directory hash is unread;"
    else
        [ "$av" != 7.7.0 ] || why="$why the installed path still holds 7.7.0;"
        [ "$av" = 7.8.0 ] || why="$why version $av, not 7.8.0;"
        [ "$ah" = "$oh" ] || why="$why code directory hash $ah, not the official 7.8.0's $oh;"
        [ -n "$at" ] && [ "$at" = "$ot" ] || why="$why team ${at:-unread}, not ${ot:-unread};"
        [ "$ac" = 0 ] || why="$why codesign --verify --deep --strict exit ${ac:-unread};"
        [ "$s" = 0 ] && /usr/bin/grep -q 'source=Notarized Developer ID' "$d/spctl-after.txt" \
            || why="$why Gatekeeper does not accept it as Notarized Developer ID (exit $s);"
    fi
    if [ -n "$why" ]; then claim native-target FAIL native_target_mismatch "$why${unk:+ Not established:$unk}"
    elif [ -n "$unk" ]; then claim native-target UNKNOWN native_target_unread "$unk"
    else claim native-target PASS official_7_8_0 "$APP: version $bv before, $av now; code directory hash $ah and team $at equal the pinned 7.8.0 DMG's app; strict signature; Gatekeeper accepts it as Notarized Developer ID (checks/native)"; fi
}

watch_start() {   # watch_start NAME SECS: procwatch on the run with qb1's own marks; armed only with every exit registered
    local i
    WDIR="$OUT/watch/$1"; WJ=
    "$OBSPY" -I -S -B "$Q/bin/procwatch.py" watch --need exit+status "$WDIR" "$R" "$APP" "$2" "$TMPD" "$SEED/" \
        > "$OUT/watch/$1.stdout" 2> "$OUT/watch/$1.stderr" &
    WJ=$!
    for i in $(/usr/bin/seq 1 50); do
        [ -f "$WDIR/armed" ] && return 0
        [ -f "$WDIR/arm-failed" ] && return 1
        /bin/sleep 0.2
    done
    return 1
}

watch_stop() {   # the watcher's own stop file, then its exit status (WRC); never a signal
    local i
    WRC=none
    [ -n "$WJ" ] || return 0
    for i in $(/usr/bin/seq 1 50); do [ -d "$WDIR" ] && break; /bin/sleep 0.2; done
    /usr/bin/touch "$WDIR/stop" 2>/dev/null
    wait "$WJ"; WRC=$?; WJ=
}

quit_request() {   # quit_request NAME KIND EVDIR: the Apple Event Quit request's own outcome, from qb1's osascript.txt.
                   # PASS only on a positive, successful request (osascript exit 0); a denial (-1743) or a timeout
                   # (-1712) stays its own named non-PASS. The app's own exit code is not judged here.
    local name="$1-request" kind="$2" f="$3/osascript.txt" rc v reason detail
    rc=$(/usr/bin/sed -n 's/^exit_code=//p' "$f" 2>/dev/null | /usr/bin/tail -1)
    detail="osascript: $(oneline "$(/bin/cat "$f" 2>/dev/null)")"
    case "$rc" in
        '') v=UNKNOWN; reason=apple_event_request_unread; detail="no osascript exit status in $f" ;;
        0) v=PASS; reason=apple_event_request_ok ;;
        *) case "$(/bin/cat "$f")" in
               *-1743*) v=FAIL; reason=tcc_automation_refused ;;
               *-1712*) v=UNKNOWN; reason=apple_event_timed_out ;;
               *) v=FAIL; reason=apple_event_request_failed ;;
           esac ;;
    esac
    if [ "$kind" = gate ]; then gate "$name" "$v" "$reason" "$detail"; else claim "$name" "$v" "$reason" "$detail"; fi
}

quit_observed() {   # quit_observed NAME KIND: an extra procwatch armed first, qb1's own ordinary quit, then three claims:
                    # NAME (qb1: the cold state), NAME-request (the Apple Event request itself succeeded) and NAME-exits
                    # (the launcher's and the server's own exits observed). A normal Quit needs all three.
    local name="$1" kind="$2" armed=0 ok res rc
    if [ -z "$FINISHING" ] && [ -n "$STOPPED" ]; then
        claim "$name" NOT_RUN chain_stopped "after $STOPPED"; claim "$name-request" NOT_RUN chain_stopped "after $STOPPED"
        claim "$name-exits" NOT_RUN chain_stopped "after $STOPPED"; return 1
    fi
    watch_start "$name" 600 && armed=1
    step "$kind" "$name" 480 "$EV/$CASE-$name" quit "$CASE" "$name" 240; ok=$?
    watch_stop
    if [ "$STEP_RAN" != 1 ]; then
        claim "$name-request" NOT_RUN quit_not_run "the quit step did not run"
        claim "$name-exits" NOT_RUN quit_not_run "the quit step did not run"; return 1
    fi
    quit_request "$name" "$kind" "$EV/$CASE-$name" || ok=1
    if [ "$armed" != 1 ]; then
        claim "$name-exits" UNKNOWN exit_watch_not_armed "$(oneline "$(/bin/cat "$WDIR/arm-failed" "$OUT/watch/$name.stderr" 2>/dev/null)")"
    else
        res=$("$OBSPY" -I -S -B -c "$PY_EXITS" "$WDIR/events.jsonl" "$APP" "$REPO" "$WRC" 2>&1); rc=$?
        if [ "$rc" = 0 ]; then claim "$name-exits" PASS exits_observed "$res (exit codes recorded, not judged)"
        else claim "$name-exits" UNKNOWN exit_unobserved "$res"; fi
    fi
    return "$ok"
}

identity() {   # identity NAME MODE ARGV: after an open (or the activation's own restart) that served, the one server's exact
               # source, runtime and launch roots: its kernel image and argv[0] (the owner's runtime venv runs the body:
               # b1a on the app's own Python, b1e on the external copy), the app's version, HEAD, tracked changes,
               # loopback answer, its own environment against the requested tuple of that open (ARGV: qb1's argv.txt)
               # and qb1's default-root witnesses
    local name="$1" mode="$2" argvf="$3" d="$OUT/checks/identity-$1" n pid img argv0 appr extr runr ver head parent bad \
        other port code detail envrc envsum f v rc defaults="" unread="" why="" unk="" moved="" want_ver=7.7.0
    pending "identity-$name" || return 1
    /bin/mkdir "$d"
    /bin/ps -axww -o pid=,ppid=,lstart=,command= > "$d/ps.txt" 2>&1
    /usr/bin/grep -F "$REPO/server.py" "$d/ps.txt" | /usr/bin/grep -v -F -e "$Q/bin/" -e procwatch.py > "$d/server.txt"
    n=$(/usr/bin/grep -c . "$d/server.txt")
    if [ "$n" != 1 ]; then claim "identity-$name" UNKNOWN "server_count_$n" "$(oneline "$(/bin/cat "$d/server.txt")")"; return 1; fi
    pid=$(/usr/bin/awk '{print $1}' "$d/server.txt")
    "$OBSPY" -I -S -B "$Q/bin/procwatch.py" snap "$d/server-snap.json" "$pid" > /dev/null 2> "$d/server-snap.stderr"
    "$OBSPY" -I -S -B -c "$PY_SERVER" "$d/server-snap.json" "$argvf" "" > "$d/server-env.txt" 2>&1
    envrc=$?
    img=$(/usr/bin/sed -n 's/^image=//p' "$d/server-env.txt"); argv0=$(/usr/bin/sed -n 's/^argv0=//p' "$d/server-env.txt")
    case "$envrc" in
        0) envsum="the server's environment holds the requested tuple" ;;
        1) envsum="the server's environment DIFFERS in: $(/usr/bin/sed -n 's/^differs=//p' "$d/server-env.txt")"
           why="$why $envsum;" ;;
        *) envsum="the server's environment is not established for: $(/usr/bin/sed -n 's/^missing=//p' "$d/server-env.txt")"
           unk="$unk $envsum;" ;;
    esac
    # qb1 obs's default locations, absent when the requested roots are used. test -e carries no errno, so a location it
    # does not find gets one stat by the observer interpreter (following links, as test -e does): absent only when that
    # lookup answered that nothing is there (ENOENT, or ENOTDIR); found then, it is in use; any other error, or a reader
    # that did not answer, leaves it unread, never absent. A location in use stays FAIL beside an unread one.
    for f in "$DATA/settings.json" "$R/app/ouroboros.pid" "$DATA/state/server_port" "$HOMEDIR/Ouroboros" "$GH/Ouroboros/data"; do
        if [ -e "$f" ]; then v=present
        else v=$("$OBSPY" -I -S -B -c "$PY_LOOKUP" "$f" 2>&1); rc=$?
             [ "$rc" = 0 ] && { [ "$v" = absent ] || [ "$v" = present ]; } \
                 || v="unread (exit $rc: $(printf '%s' "$v" | /usr/bin/tr '\n' ' ' | /usr/bin/cut -c1-300))"; fi
        echo "$f: $v" >> "$d/default-roots.txt"
        if [ "$v" = present ]; then defaults="$defaults $f"; elif [ "$v" != absent ]; then unread="$unread $f"; fi
    done
    [ -z "$defaults" ] || why="$why default roots in use:$defaults;"
    [ -z "$unread" ] || unk="$unk default roots unread:$unread (default-roots.txt);"
    appr=$(cd "$APP" 2>/dev/null && /bin/pwd -P); runr=$(cd "$R" 2>/dev/null && /bin/pwd -P)
    extr=$(cd "$EXTROOT" 2>/dev/null && /bin/pwd -P)
    ver=$(/usr/bin/plutil -extract CFBundleShortVersionString raw "$APP/Contents/Info.plist" 2>/dev/null)
    head=$(gr rev-parse -q --verify 'HEAD^{commit}' 2>/dev/null); parent=$(gr rev-parse -q --verify 'HEAD^1^{commit}' 2>/dev/null)
    gr diff --name-only HEAD > "$d/tracked-changes.txt" 2>&1
    gr ls-files --others --exclude-standard > "$d/untracked.txt" 2>&1
    gr status --porcelain=v2 --branch > "$d/status.txt" 2>&1
    bad=$(/usr/bin/grep -v -E "$OWNER_PATHS" "$d/tracked-changes.txt")
    other=$(/usr/bin/grep -v -E "$OWNER_PATHS" "$d/untracked.txt" | /usr/bin/grep -c .)
    port=$(/bin/cat "$PORTF" 2>/dev/null | /usr/bin/tr -dc 0-9); code=-
    if [ -n "$port" ]; then
        code=$(/usr/bin/curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/")
        /usr/sbin/lsof -nP -a -iTCP:"$port" -sTCP:LISTEN > "$d/listen.txt" 2>&1
    fi
    if [ -z "$img" ] || [ -z "$appr" ] || [ -z "$runr" ] || [ -z "$argv0" ]; then unk="$unk the server's image, argv[0] or a run path is unread;"
    else
        case "$argv0" in "$VENV"/bin/*) ;; *) why="$why argv[0] $argv0 is not the runtime venv's interpreter;" ;; esac
        case "$KIND" in
            app) case "$img" in "$appr"/*) ;; *) why="$why runtime $img is outside the app;" ;; esac ;;
            ext) if [ -z "$extr" ]; then unk="$unk the external root is unread;"
                 else case "$img" in "$extr"/*) ;; *) why="$why runtime $img is not the external Python;" ;; esac; fi ;;
        esac
    fi
    [ "$mode" != native ] || want_ver=7.8.0
    [ "$ver" = "$want_ver" ] || why="$why app version ${ver:-unread}, not $want_ver;"
    if [ -z "$head" ]; then unk="$unk HEAD is unread;"
    else
        case "$mode" in
            stock) [ "$head" = "$V770" ] || why="$why HEAD $head is not v7.7.0 $V770;"
                   [ ! -s "$d/tracked-changes.txt" ] || why="$why tracked changes on the stock checkout;" ;;
            owner) [ "$head" = "$H_OWNER" ] || why="$why HEAD $head is not the composed ${H_OWNER:-unread};"
                   [ "$parent" = "$C" ] || why="$why HEAD^1 ${parent:-unread} is not C;"
                   [ -z "$bad" ] || why="$why tracked changes outside the owner's paths;" ;;
            native) gr merge-base --is-ancestor "$C" HEAD > /dev/null 2>&1 || why="$why HEAD does not contain C;"
                    gr merge-base --is-ancestor "${H_OWNER:-none}" HEAD > /dev/null 2>&1 || why="$why HEAD does not contain the composed ${H_OWNER:-unread};"
                    gr diff --name-only "$C" HEAD > "$d/diff-names-C-HEAD.txt" 2>&1
                    [ -z "$(/usr/bin/grep -v -E "$OWNER_PATHS" "$d/diff-names-C-HEAD.txt")" ] || why="$why HEAD differs from C outside the owner's paths;"
                    [ -z "$bad" ] || why="$why tracked changes outside the owner's paths;"
                    [ "$(gr rev-parse -q --verify "$C^{tree}" 2>/dev/null)" = "$CT" ] || why="$why C's tree here is not $CT;"
                    gr cat-file -e "$ESHA^{commit}" > /dev/null 2>&1 && why="$why the execution commit is in the run's checkout;"
                    /usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin GIT_OPTIONAL_LOCKS=0 "$GIT" -C "$MIRROR" cat-file -e \
                        "$ESHA^{commit}" > /dev/null 2>&1 && why="$why the execution commit is in the run mirror;"
                    [ "$head" = "$H_OWNER" ] || moved=" (moved from the composed ${H_OWNER:-unread})" ;;
        esac
    fi
    [ "$code" = 200 ] || why="$why loopback http $code on port ${port:-none};"
    detail="server $pid runtime ${img:-unread} (argv0 ${argv0:-?}); app ${ver:-?}; HEAD ${head:-unread}$moved; 127.0.0.1:${port:-?} http $code; $envsum (server-env.txt); untracked outside the owner's paths: $other (recorded, not judged)"
    if [ -n "$why" ]; then claim "identity-$name" FAIL identity_mismatch "$why${unk:+ Not established:$unk} $detail"
    elif [ -n "$unk" ]; then claim "identity-$name" UNKNOWN identity_not_established "$unk $detail"
    else claim "identity-$name" PASS identified "$detail"; fi
}

work_part() {   # work_part FILE H|I|W|U: one witness of qb1's work_lines, by its line prefixes: H, I and W of
                # OWNER_NOTES.md (HEAD's blob, the index entry, the worktree file), U every untracked file and their count
    case "$2" in
        H) /usr/bin/grep -E '^H_blob=' "$1" ;;
        I) /usr/bin/grep -E '^(I_index_entry=|I_sha256=)' "$1" ;;
        W) /usr/bin/grep -E '^W_hash_object=' "$1" ;;
        U) /usr/bin/grep -E '^(U |U_count=)' "$1" ;;
    esac
}

work_read() {   # work_read FILE H|I|W|U: checkout_bytes read that witness completely: the record's own last line is
                # READS=complete, or READS=incomplete with a failed list that does not name it
    local last
    last=$(/usr/bin/tail -1 "$1" 2>/dev/null)
    case "$last" in
        READS=complete) return 0 ;;
        "READS=incomplete failed="*) case " ${last#READS=incomplete failed=} " in *" $2 "*) return 1 ;; esac; return 0 ;;
    esac
    return 1
}

work_kept() {   # work_kept NAME STEP: the owner's selected witnesses as compose recorded them (its bytes.txt) are the same
                # bytes in obs STEP's bytes.txt. A witness counts only when both records read it completely; one read
                # completely that differs is FAIL, kept beside any unread one; otherwise an unread witness is UNKNOWN.
                # HEAD itself is identity-*'s (ancestry); prepare and activate keep their own comparisons.
    local name="$1" a="$EV/$CASE-compose/bytes.txt" b="$EV/$CASE-$2/bytes.txt" p changed="" diffs="" unk=""
    pending "$name" || return 1
    for p in H I W U; do
        if ! work_read "$a" "$p"; then unk="$unk $p not read completely in compose/bytes.txt;"
        elif ! work_read "$b" "$p"; then unk="$unk $p not read completely in $2/bytes.txt;"
        elif [ "$(work_part "$a" "$p")" != "$(work_part "$b" "$p")" ]; then changed="$changed $p"; fi
    done
    for p in $changed; do diffs="$diffs $(/usr/bin/diff <(work_part "$a" "$p") <(work_part "$b" "$p") | /usr/bin/grep '^[<>]')"; done
    if [ -n "$changed" ]; then
        claim "$name" FAIL local_work_changed "the owner's$changed differ from compose's (< compose/bytes.txt, > $2/bytes.txt):$diffs${unk:+ Not established:$unk}"
    elif [ -n "$unk" ]; then claim "$name" UNKNOWN local_work_unread "$unk"
    else claim "$name" PASS local_work_kept "H, I, W and U as composed in $2/bytes.txt (complete reads): OWNER_NOTES.md in HEAD, index and worktree; $(/usr/bin/sed -n 's/^U_count=//p' "$b") untracked file(s)"; fi
}

sha256_like() { printf '%s' "$1" | /usr/bin/grep -q -x -E '[0-9a-f]{64}'; }

profile() {   # profile-survives: both canaries planted while quit are on disk in the run's private store after the
              # activation and the cold reopen (a file and SQLite witness only, not proof that the live WebView used it):
              # the planted file's sha256, and a row whose key is exactly <case>-ls-canary (qb1 compares it in SQL)
              # with exactly the planted length and whole hex value. Each canary is judged only when its planting is
              # established: the file canary by its own planted sha256, the localStorage canary by lscanary's PASS and
              # its planted value; a planting not established is never read as a loss. A completed read that misses or
              # changes a canary is FAIL, kept beside any unread or unestablished part, which its detail still names;
              # otherwise such a part is UNKNOWN; PASS needs every read complete (PROFILE_READS=complete).
    local f="$EV/$CASE-cold/profile-store.txt" fb fa hex rows other ls reads lost="" why="" unk="" name=""
    pending profile-survives || return 1
    fb=$(/usr/bin/awk 'NR == 1 {print $1}' "$EV/$CASE-canary/file-canary.txt" 2>/dev/null)
    hex=$(/usr/bin/sed -n 's/.* value_hex=\([0-9a-f]*\)$/\1/p' "$EV/$CASE-canary/summary.txt" 2>/dev/null)
    if ! /usr/bin/grep -q '^verdict=PASS ' "$EV/$CASE-canary/verdict.txt" 2>/dev/null; then
        name=canary_not_planted; hex=; unk="$unk lscanary did not PASS: the localStorage canary is not established as planted;"
    elif [ -z "$hex" ] || [ $(( ${#hex} % 2 )) != 0 ]; then
        name=canary_unread; hex=; unk="$unk the planted record is unread (summary.txt value_hex: nothing);"
    fi
    sha256_like "$fb" || { name=${name:-canary_unread}; unk="$unk the planted record is unread (canary/file-canary.txt: ${fb:-nothing});"; fb=; }
    [ -f "$f" ] || { claim profile-survives UNKNOWN "${name:-cold_store_unread}" "${unk:+${unk# } }no obs cold profile-store.txt"; return 1; }
    fa=$(/usr/bin/sed -n 's/^file_canary=//p' "$f")
    if [ -z "$fb" ] || [ "$fa" = "$fb" ]; then :
    elif [ "$fa" = absent ]; then lost=1; why="$why the file canary is absent after the reopen ($fb planted);"
    elif sha256_like "$fa"; then why="$why the file canary reads $fa after the reopen, $fb planted;"
    else unk="$unk the file canary is unread (${fa:-no file_canary line});"; fi
    if [ -n "$hex" ]; then
        rows=$(/usr/bin/awk -F '|' -v n="$(( ${#hex} / 2 ))" -v v="$hex" 'NF == 6 && $1 == "item" && $3 == "1" {
            print (($5 == n && toupper($6) == toupper(v)) ? "exact" : "other length " $5 " hex " $6)}' "$f")
        other=$(printf '%s\n' "$rows" | /usr/bin/grep '^other ')
        [ -z "$other" ] || why="$why localStorage $CASE-ls-canary holds another value after the reopen ($(oneline "$other"); planted length $(( ${#hex} / 2 )) hex $hex);"
        ls=$(/usr/bin/sed -n 's/^localstorage_reads=//p' "$f")
        [ "$ls" = complete ] || unk="$unk localStorage is not read completely (${ls:-no localstorage_reads line});"
        if ! printf '%s\n' "$rows" | /usr/bin/grep -q -x exact && [ -z "$other" ] && [ "$ls" = complete ]; then
            lost=1; why="$why localStorage $CASE-ls-canary is absent after the reopen ($(/usr/bin/sed -n 's/^databases=//p' "$f") database(s), each read completely);"
        fi
    fi
    reads=$(/usr/bin/tail -1 "$f")
    [ -n "$unk" ] || [ "$reads" = PROFILE_READS=complete ] || unk="$unk the store record is incomplete (${reads:-empty});"
    if [ -n "$why" ] && [ -n "$lost" ]; then claim profile-survives FAIL canary_lost "$why${unk:+ Not established:$unk}"
    elif [ -n "$why" ]; then claim profile-survives FAIL canary_changed "$why${unk:+ Not established:$unk}"
    elif [ -n "$unk" ]; then claim profile-survives UNKNOWN "${name:-canary_unread}" "$unk"
    else claim profile-survives PASS canaries_present "file canary $fb and localStorage $CASE-ls-canary with exactly its planted value on disk in the run's private store after the activation and the cold reopen (every read complete; an on-disk witness only)"; fi
}

owned_rows() {   # owned_rows FILE: rows whose command line names this run's root or app (observers excluded), from one
                 # ps read. Status 3 when ps or a filter failed: absence is then not established. Rows show only what
                 # command lines name; a descendant whose command line names neither is never selected.
    local rc
    /bin/ps -axww -o pid=,ppid=,lstart=,command= > "$1.ps" 2> "$1.ps.stderr" || return 3
    /usr/bin/grep -F -e "$R/" -e "$APP" "$1.ps" > "$1.match"; rc=$?; [ "$rc" -le 1 ] || return 3
    /usr/bin/grep -v -F -e "$Q/bin/" -e procwatch.py "$1.match" > "$1"; rc=$?; [ "$rc" -le 1 ] || return 3
    return 0
}

finish() {   # always: stop leftover watchers by their stop file, quit what this run started (ordinarily), and list the
             # live guest evidence (under OUT, uploaded with it; not packed or copied here). Only with OWNED: a
             # refused /Users/anka is never written to, listed, sized or quit into.
    local p rb ra
    FINISHING=1
    log "finish; chain stopped at ${STOPPED:-nothing}; fixture owned by this run: ${OWNED:-no}"
    [ "$OWNED" = 1 ] && for p in "$EV"/*/proc; do [ -d "$p" ] && [ ! -e "$p/stop" ] && /usr/bin/touch "$p/stop"; done
    if [ "$OWNED" = 1 ] && [ -e "$R/.start" ] && [ -x "$OBSPY" ]; then
        owned_rows "$OUT/checks/finish-owned-before.txt"; rb=$?
        if [ "$rb" = 0 ] && [ -s "$OUT/checks/finish-owned-before.txt" ]; then quit_observed quit-final claim; fi
        owned_rows "$OUT/checks/finish-owned-after.txt"; ra=$?
        if [ "$ra" != 0 ]; then
            claim owned-processes-after UNKNOWN process_table_unread "ps or its filter failed (checks/finish-owned-*.ps.stderr; before: status $rb); absence is not established"
        elif [ -s "$OUT/checks/finish-owned-after.txt" ]; then
            claim owned-processes-after UNKNOWN processes_remain "$(oneline "$(/usr/bin/awk '{print $1}' "$OUT/checks/finish-owned-after.txt")") remain; nothing is signalled; the provider's disposal of the runner is not measured"
        else claim owned-processes-after PASS no_matching_rows "no process whose command line names the run root or app; a descendant whose command line names neither is not covered (launcher and server exits: quit-*-exits)"; fi
    fi
    if [ "$OWNED" = 1 ]; then
        (cd "$OUT/guest-evidence" && /usr/bin/find . -type f -print | /usr/bin/sort | while IFS= read -r p; do /usr/bin/shasum -a 256 "$p"; done) \
            > "$OUT/guest-evidence-SHA256SUMS" 2>&1
        claim guest-evidence RECORDED listed "$(/usr/bin/grep -c . "$OUT/guest-evidence-SHA256SUMS") files of the live guest evidence ($EV -> out/guest-evidence) listed at the finish (guest-evidence-SHA256SUMS); uploaded with out/"
        /usr/bin/du -sk "$ROOT" "$R" "$ROOT/pkg" "$ROOT/packages" "$WHEELS" "$WORK" "$OUT" > "$OUT/checks/du.txt" 2>&1
    else /usr/bin/du -sk "$WORK" "$OUT" > "$OUT/checks/du.txt" 2>&1; fi
    home_facts > "$OUT/runner/real-home-after.txt" 2>&1
    space end
}

result() {   # ERROR (procedure) > FAIL > UNKNOWN (UNKNOWN, REFUSED, NOT_RUN) > PASS. The one declared known negative (the
             # install claim's own FAIL named known_negative_unnotarized) is listed apart and decides nothing.
    local fails unknowns known overall first
    fails=$(/usr/bin/awk -F '\t' 'NR > 1 && $3 == "FAIL" && !($2 == "install" && $4 == "known_negative_unnotarized")' "$OUT/claims.tsv" | /usr/bin/grep -c .)
    known=$(/usr/bin/awk -F '\t' 'NR > 1 && $3 == "FAIL" && $2 == "install" && $4 == "known_negative_unnotarized"' "$OUT/claims.tsv" | /usr/bin/grep -c .)
    unknowns=$(/usr/bin/awk -F '\t' 'NR > 1 && ($3 == "UNKNOWN" || $3 == "REFUSED" || $3 == "NOT_RUN")' "$OUT/claims.tsv" | /usr/bin/grep -c .)
    if [ -n "$PROC_ERR" ]; then overall=ERROR; RESULT_CODE=2
    elif [ "$fails" != 0 ]; then overall=FAIL; RESULT_CODE=1
    elif [ "$unknowns" != 0 ]; then overall=UNKNOWN; RESULT_CODE=3
    else overall=PASS; RESULT_CODE=0; fi
    first=$(/usr/bin/awk -F '\t' 'NR > 1 && $3 != "PASS" && $3 != "RECORDED" && !($2 == "install" && $4 == "known_negative_unnotarized") {print $2 " " $3 " " $4; exit}' "$OUT/claims.tsv")
    { echo "result=$overall"
      echo "first_non_pass=${first:--}"
      echo "chain_stopped_at=${STOPPED:--}"
      echo "claims=$N fail=$fails unknown_refused_or_not_run=$unknowns known_negative=$known"
      if [ "$known" != 0 ]; then
          echo "known_negative=install FAIL known_negative_unnotarized: Gatekeeper rejects the fresh 7.7.0 copy (Unnotarized Developer ID); measured, kept, not qualified (README.md)"
      else echo "known_negative=none observed"; fi
      echo "product_c=$C tree=$CT execution_sha=${ESHA:--}"
      echo "case=$CASE kind=$KIND"
      echo "runner=$(/usr/bin/uname -m) ${ImageOS:-?} ${ImageVersion:-?} macOS $(/usr/bin/sw_vers -productVersion 2>&1) session=$SESSION"
      echo "elapsed_s=$(( $(/bin/date +%s) - T0 )) budget_s=$BUDGET_S"
      echo "scope=one native case ($CASE): the official 7.7.0 app as the owner's existing copy, C's own Update preparing (real checks) and activating the official 7.8.0 app (README.md: Not qualified)"
    } > "$OUT/RESULT.txt"
}

seal() {   # the step summary (public) and the digest list of every uploaded file
    local f
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        { echo '```'; /bin/cat "$OUT/RESULT.txt"; echo; /usr/bin/cut -f1-4 "$OUT/claims.tsv"; echo '```'; } >> "$GITHUB_STEP_SUMMARY"
        SUMMARY=1
    fi
    (cd "$OUT" && /usr/bin/find . -type f ! -path ./SHA256SUMS -print | /usr/bin/sort | while IFS= read -r f; do
        /usr/bin/shasum -a 256 "$f"; done) > "$WORK/SHA256SUMS.out" 2>&1
    /bin/cp "$WORK/SHA256SUMS.out" "$OUT/SHA256SUMS"
}

on_signal() {   # a stop signal: the watchers get their own stop file, one short fact is written, and the runner exits at
                # once. No finish, collection or digests here; the workflow's upload step runs on its own, and the
                # provider's teardown of what this run started is not measured. A result the run already wrote is
                # kept as completed_* lines under the final overall: a measured FAIL or ERROR stays, anything else
                # becomes UNKNOWN. A step summary that already shows it is amended; the exit code follows the overall.
    local p prev overall code when note=
    trap - INT TERM
    [ "$OWNED" = 1 ] && for p in "$EV"/*/proc; do [ -d "$p" ] && [ ! -e "$p/stop" ] && /usr/bin/touch "$p/stop"; done
    for p in "$OUT"/watch/*; do [ -d "$p" ] && [ ! -e "$p/stop" ] && /usr/bin/touch "$p/stop"; done
    prev=$(/usr/bin/sed -n '1s/^result=//p' "$OUT/RESULT.txt" 2>/dev/null)
    case "$prev" in FAIL) overall=FAIL; code=1 ;; ERROR) overall=ERROR; code=2 ;; *) overall=UNKNOWN; code=3 ;; esac
    if [ -n "$prev" ]; then when="after the overall result (result=$prev) was written"; else when="before the overall result"; fi
    [ -e "$OUT/SHA256SUMS" ] && note=" (SHA256SUMS predates this interruption)"
    echo "interrupted $(utc) at ${STOPPED:-the running step}, $when: a stop signal ended the runner; the remaining finish, export and digests were skipped$note; overall $overall; the live guest evidence stays in out/guest-evidence; processes it started may remain until the provider disposes of the runner (not measured)" > "$OUT/INTERRUPTED.txt"
    printf '%02d\tinterrupted\tUNKNOWN\tjob_stop_signal\t%s; overall %s (INTERRUPTED.txt)\n' "$((N + 1))" "$when" "$overall" >> "$OUT/claims.tsv"
    if [ -z "$prev" ]; then
        printf 'result=UNKNOWN\nfirst_non_pass=interrupted UNKNOWN job_stop_signal (INTERRUPTED.txt)\n' > "$OUT/RESULT.txt"
    else
        { echo "result=$overall"; echo "interrupted=$when; overall $overall (INTERRUPTED.txt)"
          /usr/bin/sed 's/^/completed_/' "$OUT/RESULT.txt"; } > "$OUT/RESULT.txt.new" && /bin/mv "$OUT/RESULT.txt.new" "$OUT/RESULT.txt"
    fi
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        if [ -n "$SUMMARY" ]; then
            echo "**Interrupted** $when: the result block above is the completed measurement's own record; the overall verdict is **$overall** (RESULT.txt, INTERRUPTED.txt), and the claims above stand as measured."
        else echo "**Interrupted** $when: the overall verdict is **$overall** (RESULT.txt, INTERRUPTED.txt)."; fi >> "$GITHUB_STEP_SUMMARY"
    fi
    exit "$code"
}
trap on_signal INT TERM

log "native $CASE start: execution ${ESHA:-unset}; product C $C; out $OUT"
space start
runner_facts
arch_gate
disk_gate
overlay_gate
git_gate
pin_gate
fixture_gate
inputs_gate
capsule_770
observer_gate
wheel_pins_gate
wheelhouse_gate
bundle_gate
equip_gate
step rec facts 300 "$EV/$NS-facts-x" facts x
step gate selftest 120 "$EV/$NS-selftest-x" selftest x
step claim install 1800 "$EV/$CASE-install" install "$CASE"   # its own verdict as measured: the fresh copy's Gatekeeper answer
existing_copy
step gate mirror 1800 "$EV/$CASE-mirror" mirror "$CASE"
step gate candsrc 900 "$EV/$NS-candsrc" candsrc "$CASE"
[ "$KIND" = ext ] && step gate extpy 1800 "$EV/$CASE-extpy" extpy "$CASE"
step gate venvprep 3600 "$EV/$CASE-venvprep" venvprep "$CASE"
step gate open-1 600 "$EV/$CASE-open-1" openw "$CASE" open-1 300
identity open-1 stock "$EV/$CASE-open-1/argv.txt"
quit_observed quit-1 gate
step claim canary 300 "$EV/$CASE-canary" lscanary "$CASE" canary
step gate compose 600 "$EV/$CASE-compose" compose "$CASE" && H_OWNER=$(gr rev-parse -q --verify 'HEAD^{commit}' 2>/dev/null)
step gate open-2 600 "$EV/$CASE-open-2" openw "$CASE" open-2 300
identity open-2 owner "$EV/$CASE-open-2/argv.txt"
step rec obs-pre 600 "$EV/$CASE-pre" obs "$CASE" pre
work_kept work-pre pre   # compose's witnesses after the first composed launch (identity-open-2: HEAD is the composed one)
step claim fn-pre 300 "$EV/$CASE-fn-pre" localfn "$CASE" fn-pre 6.0.3 venv
[ -n "$STOPPED" ] || { /bin/mkdir "$OUT/checks/native" && app_identity "$APP" "$OUT/checks/native/before.txt"; }
# C's own preparation, unstubbed: the expected tier is C's selector's own answer for this geometry (README: full)
step gate prepare "$PREP_WINDOW_S" "$EV/$CASE-prep" prepare "$CASE" prep "$PREP_GUEST_S" ready full
step gate activate 1200 "$EV/$CASE-act" activate "$CASE" act 900
native_target
identity post-activate native "$EV/$CASE-open-2/argv.txt"   # the server the activation's own launch started
step rec obs-post 600 "$EV/$CASE-post" obs "$CASE" post
step claim fn-post 300 "$EV/$CASE-fn-post" localfn "$CASE" fn-post 6.0.3 venv
quit_observed quit-2 gate
step gate reopen 600 "$EV/$CASE-reopen" openw "$CASE" reopen 300
identity reopen native "$EV/$CASE-reopen/argv.txt"
step rec obs-cold 600 "$EV/$CASE-cold" obs "$CASE" cold
work_kept work-cold cold   # compose's witnesses after the whole transition and the cold reopen (identity-reopen: ancestry)
profile
step claim fn-cold 300 "$EV/$CASE-fn-cold" localfn "$CASE" fn-cold 6.0.3 venv
step rec effects 600 "$EV/$CASE-fx" effects "$CASE" fx
finish
result
seal
exit "$RESULT_CODE"
