#!/bin/bash
# Linux platform wrapper of the O0 -> O1 published-range readiness instrument (test equipment, not product). It stands
# in for the macOS lab wrapper o0o1.sh (sha256 057e257d...) with that wrapper's fixture semantics; the driver
# o0o1_readiness.py runs as pinned (DRIVER_SHA256 below). One hosted Ubuntu job per root runs it as the runner user
# from $ROOT/<namespace>/bin (a qu* namespace), where bin holds exactly this script, o0o1_readiness.py and
# o0o1_witness_linux.py, in this order:
#   facts LABEL              what this runner offers: kernel, CPUs, memory, disk, limits, Git, the helper Python
#   roots clean|evolved      the readiness mirror, then the witness root and that root. The mirror holds only the
#                            official head under test (ouroboros = O1, fetched from the official URL by its id) and the
#                            fixture's official tags: no later head, no implementation object.
#                            A root is the managed clone O0's own launcher bootstrap makes
#                            (launcher_bootstrap._configure_managed_clone at e00fec02, its meta as
#                            _normalize_bundle_manifest writes it), its official URL served by the mirror through the
#                            root home's insteadOf: DISCLOSED equivalents (no O0 launcher ran, no O0 bundle was built).
#                            witness and clean are O0; evolved is O0 + H, I, W, U.
#   runtime IMPL UV          the bare runtime, installed separately: the implementation's own
#                            scripts/download_python_standalone.sh (CPython 3.10.22, python-build-standalone 20261009,
#                            its pinned SHA-256), without INTERPRETER_ONLY, so it installs O0's complete
#                            requirements-runtime.lock into the interpreter itself, with uv UV (0.12.1). No venv, no
#                            editable install, no development or test group.
#   node NODE                the installed runner Node (absolute; the workflow installs it as the product's CI does):
#                            the launch PATH becomes the fixed one plus NODE's directory, appended, and that PATH must
#                            select NODE, at or above the product's Node floor. Recorded once in $EV/<namespace>-node;
#                            witness and readiness read it there, check it still selects the same Node, and refuse
#                            without it. Nothing is placed in any root, checkout or runtime.
#   witness                  the serving witness: O0's own server.py from the witness root, on that runtime, observed
#                            serving from the kernel and stopped (o0o1_witness_linux.py): the driver's interpreter
#                            evidence
#   readiness clean|evolved IMPL   the driver on that root, on the interpreter the witness saw serving, with the
#                            implementation checkout IMPL
#   pack LABEL               the candidate scratch the checks leave beside the root (update-checks/junit-*), as a tar
# Exit status: a setup step's first failing step (and result.txt); witness is the witness's and readiness the driver's
# (0 PASS, 1 FAIL, 3 UNKNOWN, 2 refused before any effect).
set -u
# One fixed environment whatever the caller's: a hosted job exports XDG_CONFIG_HOME, RUNNER_*, GITHUB_* and more, which
# Git and the helper would otherwise read. Re-executed once under env -i.
if [ "${O0O1_LINUX_ENV:-}" != fixed ]; then
    exec /usr/bin/env -i O0O1_LINUX_ENV=fixed HOME=/home/runner/oulab PATH=/usr/bin:/bin:/usr/sbin:/sbin \
        LC_ALL=C.UTF-8 /bin/bash "$0" "$@"
fi
GH=/home/runner
ROOT="$GH/oulab"
Q=$(cd "$(dirname "$0")/.." 2>/dev/null && pwd -P)   # the namespace folder this verified copy runs from
NS=$(basename "${Q:-/none}")
EV="$Q/evidence"
GIT=/usr/bin/git
HELPY=/usr/bin/python3   # the stdlib helper (any stdlib Python): never the runtime under test
# Not ".../python-standalone": the product reads that name as a packaged app's runtime (platform_layer.
# interpreter_is_embedded); this one is an owner's bare interpreter, as the macOS geometry's b1p cpython-ext.
RUNTIME="$Q/runtime/cpython-bare"
PY="$RUNTIME/bin/python3"
MIRROR="$Q/runs/o0o1-mirror.git"
WITNESS_PORT=28831
WITNESS_SECONDS=300
BASE_PATH=/usr/bin:/bin:/usr/sbin:/sbin   # this script's own PATH; the launch PATH extends it (node)
PATH=$BASE_PATH
export PATH
NODE_EV="$EV/$NS-node"   # the bound Node: node.txt (its facts) and launch-path.txt (the one launch PATH)
# The product's Node floor (preflight_node.NODE_MIN_VERSION, (20, 11), read from C7 b87e4141): below it, a full
# tier over browser-module tests is unverified before any test runs (update_checks._node). Re-read on a rebind.
NODE_FLOOR_MAJOR=20
NODE_FLOOR_MINOR=11
# The parent-accepted published range; any other range needs a rebind.
O0=e00fec02e69ef07d65d180f5b27c6660e5177a38
O1=28a6ed8f9714c9ce7e279e99b5656d2c4c453b68
O1_COMMITS=61
LOCK_SHA256=68e739bde5b7fbf90d2663eb01939f52c5c4b0b087097390e05807e143e8673d
# ---- THE PRODUCT PIN: the implementation under test (C7), set here and nowhere else. The workflow reads IMPL_COMMIT,
# IMPL_TREE and DRIVER_SHA256 from these exact lines before it checks anything out; the bind's digests are IMPL_SHA256.
# A rebind edits this block (DOWNLOADER_SHA256 only if the downloader changed) and renames the workflow file and its
# branch pattern, whose 12 hex digits are names, not checks.
IMPL_COMMIT=b87e41411d9172846b09570d46aaa67539355e1a
IMPL_TREE=20789d1554e5be6c61a9ffd4c99846b96aff9a6c
IMPL_SHA256="supervisor/update_checks.py=34aa0f518a93907812d0a444e4148c985570b71fed1a8661d7694b29f491e9bf
ouroboros/test_selection.py=421686b56543545c3766f04a2f4b92874942e1a460adc92ae2e2655a295e453f
supervisor/update_prepare.py=9e300fa23a9200c550b66b48689d2749884d0407b9a4187097e4cdcc3ae5922b
supervisor/git_ops.py=583b68f3795a64b9133243c95e975459bcfaeac4ccb92d988132660dcaf2ee11
supervisor/update_native.py=7c311c961686fb4e03190dae739584544e0677d424f9e216775df902c3461f8e
ouroboros/__init__.py=eabfc4c4aa5859b4d9b6503a0c055b33e9c8d853861257ec352b105e17714795
ouroboros/test_environment.py=3b8580ff5022958056197e0c9568344d797ec6db6ffc2ffb34d8c436494ba9f0
ouroboros/launch_roots.py=33abc9650d5521437884d9019a676b2300059aab948ee02e1fbffe40f58245c8"
DOWNLOADER_SHA256=9cbb4414284c151fe34aaf805a0a2e291b42f87b79c62a597d6203e984e2e5ad   # C7's scripts/download_python_standalone.sh (as C6's and C2's)
# ---- end of the product pin. The equipment's own driver digest:
DRIVER_SHA256=02d7826550a3d1a8e763f0be26ff45ff9e663b3ed03b353b053cbcacc53976c2
PY_VERSION=3.10.22
SQLITE_VERSION=3.53.1
UV_VERSION=0.12.1
# O1's required floor (one module-level FLOOR_TESTS literal in 28a6ed8f:ouroboros/test_selection.py, read 2026-10-30).
FLOOR="tests/test_contracts.py=frozen ABI and tool/context protocols
tests/test_gateway_parity.py=HTTP/WS wire schemas and router parity
tests/test_packaging_sync.py=release identity and public install carriers
tests/test_process_custody.py=every long-lived child has a lifetime owner
tests/test_hermetic_data_root.py=tests cannot bind the live data roots
tests/test_preflight_process_containment.py=a leaked test process cannot pass the gate
tests/test_smoke.py=the shipped Python modules import
tests/test_safety_policy.py=every built-in tool has an explicit safety policy"
OFFICIAL_URL=https://github.com/razzant/ouroboros
BRANCH=ouroboros
# The fixture's official tags (the ones o0o1.sh fetched, all below O0): name, tag object, commit. The development
# channel reads none of them (update_source reads release tags for the stable channel only).
TAGS="v7.7.0 a773f0db6bcbec4a8b5e718137a1919537e93a79 e41596c389c3c33c3155637aaf67f8bebbf4934d
v7.7.1 4cc0f3939cff47e23972a24d5155e86d8e06bcf2 fc8cd8146671fdcb71d336b9b49970b6f9893a0a
v7.8.0 80a38cb91450a0c6bb0c17c7a980b8213e0a5830 6f3ddff05ff3e6c6f8d007ff96358df6af872e83"
CUTOVER=ouroboros/body_switch.py   # a cutover path the official O0..O1 range leaves unchanged
MARKER="# o0o1 readiness lab: an owner's local edit of the cutover machinery (comment only)"
LOCAL_TEST=tests/test_o0o1_local_probe.py
NOTES=OWNER_NOTES.md
UNTRACKED="owner-scratch.txt owner-untracked/nested/blob.bin"

die() { echo "o0o1-linux: $*" >&2; exit 2; }
namespace_ok() {   # this copy runs from $ROOT/<qu namespace>/bin; any other location is refused before an effect
    case "$NS" in qu|*[!a-z0-9-]*) die "invalid qualification namespace: $NS" ;; qu*) ;; *) die "invalid qualification namespace: $NS" ;; esac
    [ "$(dirname "$Q")" = "$ROOT" ] || die "o0o1_linux.sh runs only from $ROOT/<namespace>/bin, not from $Q"
}
label_ok() { case "${1:-}" in ''|*[!a-z0-9-]*) die "invalid label: ${1:-}" ;; esac; }
newdir() { namespace_ok; [ ! -e "$1" ] || die "already recorded: $1"; mkdir -p "$1" || die "cannot create $1"; echo "$1"; }
utc() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
step() { local name="$1" rc; shift; "$@"; rc=$?; echo "step=$name exit_code=$rc"; return "$rc"; }   # its own status, kept
sha256() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
gq() { GIT_OPTIONAL_LOCKS=0 "$GIT" "$@"; }   # a read
impl_ok() {   # impl_ok DIR: the implementation checkout is exactly C7 and clean, ignored files included (the driver reads
              # it again, its digests too)
    case "${1:-}" in /*) ;; *) die "IMPL is an absolute path, not ${1:-}" ;; esac
    [ "$(gq -C "$1" rev-parse -q --verify 'HEAD^{commit}')" = "$IMPL_COMMIT" ] || die "$1 is not $IMPL_COMMIT"
    [ "$(gq -C "$1" rev-parse -q --verify 'HEAD^{tree}')" = "$IMPL_TREE" ] || die "$1 does not hold tree $IMPL_TREE"
    [ -z "$(gq -C "$1" status --porcelain=v1 --untracked-files=all --ignored=matching)" ] || die "$1 is not clean"
}

mirror_build() {   # the official remote as published: its head under test, by id, and the fixture's tags, nothing else
    local home="$Q/runs/.mirror-home" name tagobj commit spec
    spec=("+$O1:refs/heads/$BRANCH")
    while read -r name tagobj commit; do
        [ -n "$name" ] && spec+=("+refs/tags/$name:refs/tags/$name")
    done <<EOF
$TAGS
EOF
    mkdir -p -m 700 "$home" || return
    step init "$GIT" -c init.defaultBranch="$BRANCH" init -q --bare "$MIRROR" || return
    step config "$GIT" -C "$MIRROR" config gc.auto 0 || return
    # The one network read: from the official URL itself, with a fresh home, no tag following, no prompt.
    step fetch env HOME="$home" GIT_TERMINAL_PROMPT=0 "$GIT" -C "$MIRROR" fetch --no-tags "$OFFICIAL_URL" "${spec[@]}" \
        || return
    step head "$GIT" -C "$MIRROR" symbolic-ref HEAD "refs/heads/$BRANCH" || return
}

mirror_check() {   # "ok ..." when the mirror serves exactly O1 on its one head, O0 below it, the pinned tags, nothing else
    local got want refs n name tagobj commit
    [ ! -f "$MIRROR/shallow" ] || { echo "the mirror is shallow"; return 1; }
    got=$(gq -C "$MIRROR" rev-parse -q --verify "refs/heads/$BRANCH^{commit}")
    [ "$got" = "$O1" ] || { echo "refs/heads/$BRANCH is ${got:-absent}, not O1 $O1"; return 1; }
    [ "$(gq -C "$MIRROR" symbolic-ref HEAD)" = "refs/heads/$BRANCH" ] || { echo "HEAD is not refs/heads/$BRANCH"; return 1; }
    want="refs/heads/$BRANCH"
    while read -r name tagobj commit; do
        [ -n "$name" ] || continue
        want="$want refs/tags/$name"
        [ "$(gq -C "$MIRROR" rev-parse -q --verify "refs/tags/$name")" = "$tagobj" ] || { echo "$name is not the tag object $tagobj"; return 1; }
        [ "$(gq -C "$MIRROR" cat-file -t "$tagobj")" = tag ] || { echo "$name is not an annotated tag"; return 1; }
        [ "$(gq -C "$MIRROR" rev-parse -q --verify "refs/tags/$name^{commit}")" = "$commit" ] || { echo "$name is not $commit"; return 1; }
        gq -C "$MIRROR" merge-base --is-ancestor "$commit" "$O0" || { echo "$name is not below O0"; return 1; }
    done <<EOF
$TAGS
EOF
    refs=$(gq -C "$MIRROR" for-each-ref --format='%(refname)' | sort | tr '\n' ' ')
    [ "$refs" = "$(printf '%s\n' $want | sort | tr '\n' ' ')" ] || { echo "the mirror holds other refs: $refs"; return 1; }
    gq -C "$MIRROR" merge-base --is-ancestor "$O0" "$O1" || { echo "O0 is not an ancestor of O1"; return 1; }
    n=$(gq -C "$MIRROR" rev-list --count "$O0..$O1")
    [ "$n" = "$O1_COMMITS" ] || { echo "O0..O1 holds $n commits, not $O1_COMMITS"; return 1; }
    if gq -C "$MIRROR" cat-file -e "$IMPL_COMMIT^{commit}" 2>/dev/null; then
        echo "the implementation $IMPL_COMMIT is in the mirror"; return 1
    fi
    echo "ok $BRANCH=O1 $O1, O0..O1 $n commits, refs $refs, no implementation object"
}

meta() {   # meta PATH VERSION: the normalized bundle manifest O0's launcher_bootstrap writes as the managed repo meta
    "$HELPY" -I -S -B -c 'import json, sys
path, o0, url, branch, version = sys.argv[1:6]
meta = {"schema_version": 1, "bundle_file": "repo.bundle", "app_version": version, "source_sha": o0, "release_tag": "",
        "bundle_sha256": "", "source_branch": branch, "managed_remote_name": "managed", "managed_remote_url": url,
        "managed_remote_branch": branch, "managed_local_branch": branch, "managed_local_stable_branch": "ouroboros-stable",
        "managed_remote_stable_branch": "ouroboros-stable"}
with open(path, "x", encoding="utf-8") as handle:
    handle.write(json.dumps(meta, indent=2) + "\n")' "$1" "$O0" "$OFFICIAL_URL" "$BRANCH" "$2"
}

local_test() { cat <<'EOF'
"""The owner's local readiness probe (o0o1 evolved root; lab work, not product).

It passes only where the ``ouroboros`` package the test process imports is the checkout beside this file and carries
this installation's local cutover marker: run by the Update's checks, that checkout is the materialized W'.
"""
import pathlib


def test_the_tests_import_the_checkout_they_run_in():
    import ouroboros

    here = pathlib.Path(__file__).resolve().parents[1]
    package = pathlib.Path(ouroboros.__file__).resolve().parent
    assert package == here / "ouroboros", (package, here)
    marker = "o0o1 readiness lab: an owner's local edit of the cutover machinery"
    assert marker in (package / "body_switch.py").read_text(encoding="utf-8")
EOF
}

owner_work() {   # owner_work REPO: H (one local commit on O0: the local test, the cutover comment, a note), then I, W, U
    local repo="$1" f
    mkdir -p "$repo/tests" || return
    local_test > "$repo/$LOCAL_TEST" || return
    printf '\n%s\n' "$MARKER" >> "$repo/$CUTOVER" || return
    printf '# Owner note (o0o1 evolved H)\nThis line is a local commit of the owner.\n' > "$repo/$NOTES" || return
    step h-add rg add -- "$LOCAL_TEST" "$CUTOVER" "$NOTES" || return
    step h-commit rg -c user.name=Owner -c user.email=owner@lab.invalid commit -q \
        -m "owner: local readiness probe and cutover note (o0o1 H)" || return
    printf 'Staged line (o0o1 I).\n' >> "$repo/$NOTES" || return
    step i-add rg add -- "$NOTES" || return
    printf 'Unstaged line (o0o1 W).\n' >> "$repo/$NOTES" || return
    printf 'untracked owner scratch (o0o1 U)\n' > "$repo/owner-scratch.txt" || return
    mkdir -p "$repo/owner-untracked/nested" || return
    step u-blob dd if=/dev/urandom of="$repo/owner-untracked/nested/blob.bin" bs=4096 count=16 2>/dev/null || return
    for f in $UNTRACKED; do [ -f "$repo/$f" ] || { echo "no untracked $f"; return 1; }; done
}

mkroot() {   # mkroot witness|clean|evolved: the managed clone at O0 (_configure_managed_clone's steps), then its kind's work
    local kind="$1" R="$Q/runs/o0o1-$1"
    repo="$R/app/repo"; RH="$R/home"
    [ ! -e "$R" ] || { echo "exists: $R"; return 1; }
    mkdir -p -m 700 "$R" "$RH" "$R/app" "$R/conf" "$R/locks" "$R/work" || return
    printf '[url "%s"]\n\tinsteadOf = %s\n\tinsteadOf = %s.git\n' "$MIRROR" "$OFFICIAL_URL" "$OFFICIAL_URL" > "$RH/.gitconfig" \
        || return
    step "clone-$kind" env HOME="$RH" "$GIT" clone -q --no-checkout "$OFFICIAL_URL" "$repo" || return
    step "checkout-$kind" rg checkout -q -B "$BRANCH" "$O0" || return
    step "stable-$kind" rg branch -f ouroboros-stable "$O0" || return
    step "origin-$kind" rg remote remove origin || return
    step "managed-$kind" rg remote add managed "$OFFICIAL_URL" || return
    step "name-$kind" rg config user.name Ouroboros || return
    step "email-$kind" rg config user.email ouroboros@local.mac || return
    step "meta-$kind" meta "$repo/.git/ouroboros-managed.json" "$(rg cat-file blob "$O0:VERSION" | tr -d ' \n')" \
        || return
    if [ "$kind" = evolved ]; then owner_work "$repo" || return; fi
    echo "-- $kind root: HEAD=$(rg rev-parse HEAD) branch=$(rg symbolic-ref -q HEAD) remotes=$(rg remote | tr '\n' ' ')"
    rg status --porcelain=v2 --branch --untracked-files=all
    rg log --format='%H %P %s' -3
}
rg() { env HOME="$RH" GIT_OPTIONAL_LOCKS=0 "$GIT" -C "$repo" "$@"; }   # the root being made, its own home

roots() {
    local kind="${1:-}" out rc why k
    namespace_ok
    case "$kind" in clean|evolved) ;; *) die "roots clean|evolved" ;; esac
    [ ! -e "$MIRROR" ] || die "exists: $MIRROR"
    out=$(newdir "$EV/$NS-roots") || exit 2
    utc > "$out/utc-start.txt"
    mirror_build > "$out/mirror.txt" 2>&1; rc=$?   # its status is the first failing step's
    if [ "$rc" = 0 ]; then why=$(mirror_check); echo "$why" > "$out/mirror-check.txt"; case "$why" in ok*) ;; *) rc=1 ;; esac; fi
    for k in witness "$kind"; do
        [ "$rc" = 0 ] || break
        mkroot "$k" > "$out/root-$k.txt" 2>&1; rc=$?
    done
    gq -C "$MIRROR" for-each-ref --format='%(refname) %(objectname) %(objecttype)' > "$out/mirror-refs.txt" 2>&1
    echo "exit_code=$rc" > "$out/result.txt"; utc > "$out/utc-end.txt"
    echo "$out"; return "$rc"
}

runtime_check() {   # the installed runtime, read with -I -B (no bytecode, no user site): FACTS, then its checks
    "$PY" -I -B -c 'import json, os, sqlite3, sys
from importlib import metadata
print(json.dumps({"executable": sys.executable, "real": os.path.realpath(sys.executable), "prefix": sys.prefix,
                  "base_prefix": sys.base_prefix, "version": sys.version, "sqlite": sqlite3.sqlite_version,
                  "flags": {name: getattr(sys.flags, name) for name in ("isolated", "no_site", "no_user_site")},
                  "pyvenv_cfg": os.path.exists(os.path.join(sys.prefix, "pyvenv.cfg")),
                  "distributions": sorted("%s==%s" % (dist.metadata["Name"], dist.version)
                                          for dist in metadata.distributions())}, indent=1, sort_keys=True))' > "$1" \
        || { echo "the runtime did not run its probe"; return 1; }
    "$HELPY" -I -S -B -c 'import json, os, sys
facts, runtime, version, sqlite = json.load(open(sys.argv[1], encoding="utf-8")), sys.argv[2], sys.argv[3], sys.argv[4]
problems = [text for ok, text in (
    (facts["version"].split()[0] == version, "Python %s, not %s" % (facts["version"].split()[0], version)),
    (facts["sqlite"] == sqlite, "SQLite %s, not %s" % (facts["sqlite"], sqlite)),
    (facts["prefix"] == facts["base_prefix"], "a virtual environment"),
    (not facts["pyvenv_cfg"], "a pyvenv.cfg beside the prefix"),
    (os.path.realpath(facts["prefix"]) == os.path.realpath(runtime), "prefix %s, not %s" % (facts["prefix"], runtime)),
    ("python-standalone" not in os.path.realpath(facts["executable"]).split(os.sep), "a packaged-runtime path"))
    if not ok]
print("; ".join(problems) or "ok Python %s, SQLite %s, bare prefix %s, %d distributions" % (
    version, sqlite, facts["prefix"], len(facts["distributions"])))
sys.exit(1 if problems else 0)' "$1" "$RUNTIME" "$PY_VERSION" "$SQLITE_VERSION"
}

runtime() {   # runtime IMPL UV: the implementation's own downloader, pinned asset and lock install, not INTERPRETER_ONLY
    local impl="${1:-}" uv="${2:-}" out work rc
    namespace_ok
    impl_ok "$impl"
    case "$uv" in /*) ;; *) die "UV is an absolute path, not $uv" ;; esac
    [ -x "$uv" ] || die "no uv at $uv"
    case "$("$uv" --version 2>/dev/null)" in "uv $UV_VERSION"|"uv $UV_VERSION "*) ;; *) die "$uv is not uv $UV_VERSION" ;; esac
    [ -d "$MIRROR" ] || die "no mirror at $MIRROR (roots first): O0's lock is read from it"
    [ "$(sha256 "$impl/scripts/download_python_standalone.sh")" = "$DOWNLOADER_SHA256" ] || die "the downloader is not $DOWNLOADER_SHA256"
    [ ! -e "$Q/runtime" ] || die "exists: $Q/runtime"
    work="$Q/runs/runtime-install"
    [ ! -e "$work" ] || die "exists: $work"
    out=$(newdir "$EV/$NS-runtime") || exit 2
    mkdir -p -m 700 "$Q/runtime" "$work" "$work/home" "$work/tmp" || die "cannot create $work"
    gq -C "$MIRROR" cat-file blob "$O0:requirements-runtime.lock" > "$work/requirements-runtime.lock" || die "O0's lock is unread"
    [ "$(sha256 "$work/requirements-runtime.lock")" = "$LOCK_SHA256" ] || die "O0's lock is not $LOCK_SHA256"
    { echo "impl=$impl uv=$uv ($("$uv" --version 2>&1)) dest=$RUNTIME"
      sha256sum "$impl/scripts/download_python_standalone.sh" "$work/requirements-runtime.lock" "$uv"; } > "$out/inputs.txt" 2>&1
    utc > "$out/utc-start.txt"
    # From a scratch folder holding exactly O0's lock (the script installs ./requirements-runtime.lock), in a fixed
    # environment: the script's own curl, tar and uv, uv's cache in the scratch home.
    (cd "$work" && exec env -i HOME="$work/home" TMPDIR="$work/tmp" PATH="$(dirname "$uv"):/usr/bin:/bin" LC_ALL=C.UTF-8 \
        PYTHON_STANDALONE_DEST="$RUNTIME" /bin/bash "$impl/scripts/download_python_standalone.sh") > "$out/install.txt" 2>&1
    rc=$?; echo "exit_code=$rc" > "$out/result.txt"; utc > "$out/utc-end.txt"
    if [ "$rc" = 0 ]; then runtime_check "$out/facts.json" > "$out/check.txt" 2>&1; rc=$?; echo "check_exit_code=$rc" >> "$out/result.txt"; fi
    du -sk "$RUNTIME" "$work" > "$out/du.txt" 2>&1
    echo "$out"; return "$rc"
}

launch_pairs() {   # launch_pairs ROOT TMPDIR PATH: an installation's launch environment in ROOT, the bind's env. o0o1.sh's,
                   # without macOS's CFFIXED_USER_HOME and DEVELOPER_DIR, and LC_CTYPE=C.UTF-8: under the C locale Python
                   # coerces LC_CTYPE into the environment at start, and the driver refuses any change to it. PATH is the
                   # bound launch PATH (bound_path), the same for the witness and the driver
    printf '%s\n' "OUROBOROS_APP_ROOT=$1/app" "OUROBOROS_REPO_DIR=$1/app/repo" "OUROBOROS_DATA_DIR=$1/app/data" \
        "OUROBOROS_SETTINGS_PATH=$1/conf/settings.json" "OUROBOROS_PID_FILE=$1/locks/ouroboros.pid" \
        "OUROBOROS_PORT_FILE=$1/locks/server_port" "OUROBOROS_SUBAGENT_WORKTREE_ROOT=$1/work/subagent_worktrees" \
        "OUROBOROS_SUBAGENT_PROJECTS_ROOT=$1/work/projects" "OUROBOROS_DELIVERABLES_ROOT=$1/work/Deliverables" \
        "OUROBOROS_UPDATE_CHANNEL=development" "HOME=$1/home" "TMPDIR=$2" "PATH=$3" \
        "LC_CTYPE=C.UTF-8" "PYTHONDONTWRITEBYTECODE=1"
}

node_facts() {   # node_facts LAUNCH_PATH: the Node that PATH selects: its first executable file named node, as the
                 # checks' shutil.which finds it; its version is what it prints on stdout, as their probe reads it
    local found real version
    found=$(IFS=:; for dir in $1; do [ -f "$dir/node" ] && [ -x "$dir/node" ] && { echo "$dir/node"; break; }; done)
    case "$found" in /*) ;; *) echo "no node on $1"; return 1 ;; esac
    real=$(readlink -f "$found") || { echo "$found does not resolve"; return 1; }
    version=$(env -i PATH="$1" "$found" --version 2>/dev/null) || { echo "$found --version failed (exit $?)"; return 1; }
    printf 'node=%s\nreal=%s\nsha256=%s\nversion=%s\n' "$found" "$real" "$(sha256 "$real")" "$version"
}

node_bind() {   # node NODE: the installed runner Node the installation is offered, through its launch PATH only
    local given="${1:-}" dir lp facts version v major minor out
    namespace_ok
    case "$given" in /*) ;; *) die "NODE is an absolute path, not ${given:-nothing}" ;; esac
    [ -f "$given" ] && [ -x "$given" ] || die "no executable Node at $given"
    dir=$(dirname "$given")
    case "$dir" in *[!A-Za-z0-9._/+-]*) die "the Node directory $dir is not a plain PATH entry" ;; esac
    case "$dir/" in "$ROOT"/*|"$GH"/work/*) die "the Node directory $dir lies in the lab or a checkout, not an installed runner Node" ;; esac
    lp="$BASE_PATH:$dir"
    facts=$(node_facts "$lp") || die "$facts"
    [ "$(echo "$facts" | sed -n 's/^node=//p')" = "$given" ] || die "on $lp the first node is not $given: $facts"
    version=$(echo "$facts" | sed -n 's/^version=//p')
    v=${version#v}
    case "$v" in [0-9]*.[0-9]*) ;; *) die "$given answers no version: $version" ;; esac
    major=${v%%.*}; minor=${v#*.}; minor=${minor%%.*}
    case "$major.$minor" in *[!0-9.]*) die "$given answers no version: $version" ;; esac
    if [ "$major" -lt "$NODE_FLOOR_MAJOR" ] || { [ "$major" -eq "$NODE_FLOOR_MAJOR" ] && [ "$minor" -lt "$NODE_FLOOR_MINOR" ]; }; then
        die "$given is $version, below the product's Node floor v$NODE_FLOOR_MAJOR.$NODE_FLOOR_MINOR"
    fi
    out=$(newdir "$NODE_EV") || exit 2
    echo "$facts" > "$out/node.txt"
    echo "$lp" > "$out/launch-path.txt"
    { echo "given=$given floor=v$NODE_FLOOR_MAJOR.$NODE_FLOOR_MINOR"; ls -l "$dir"; } > "$out/directory.txt" 2>&1
    cat "$out/node.txt" "$out/launch-path.txt"
}

bound_path() {   # the launch PATH node bound, once it is checked to select the same Node still
    local lp
    [ -f "$NODE_EV/node.txt" ] && [ -f "$NODE_EV/launch-path.txt" ] || die "no bound Node in $NODE_EV (node first)"
    lp=$(cat "$NODE_EV/launch-path.txt")
    case "$lp" in "$BASE_PATH":/*) ;; *) die "the bound PATH $lp does not extend $BASE_PATH" ;; esac
    [ "$(node_facts "$lp")" = "$(cat "$NODE_EV/node.txt")" ] || die "$lp no longer selects the Node bound in $NODE_EV/node.txt"
    echo "$lp"
}

witness() {   # O0's own server.py from the witness root, on the runtime: observed serving, then stopped
    local R="$Q/runs/o0o1-witness" out stepd tmpd version rc lp
    namespace_ok
    [ -x "$PY" ] || die "no runtime at $RUNTIME (runtime first)"
    [ -f "$R/app/repo/.git/ouroboros-managed.json" ] || die "no witness root at $R (roots first)"
    [ "$(gq -C "$R/app/repo" rev-parse -q --verify 'HEAD^{commit}')" = "$O0" ] || die "the witness root is not at O0"
    lp=$(bound_path) || exit 2
    version=$(gq -C "$R/app/repo" cat-file blob "$O0:VERSION" | tr -d ' \n')
    tmpd="/tmp/$NS-o0o1-witness"
    [ ! -e "$tmpd" ] || die "tmp root exists: $tmpd"
    out="$EV/o0o1-witness"
    [ ! -e "$out" ] || die "already recorded: $out"
    stepd=$(newdir "$EV/$NS-witness") || exit 2
    mkdir -p -m 700 "$tmpd" || die "tmp root $tmpd"
    { launch_pairs "$R" "$tmpd/" "$lp"; echo "OUROBOROS_SERVER_HOST=127.0.0.1"; echo "OUROBOROS_SERVER_PORT=$WITNESS_PORT"; } \
        > "$stepd/launch-env.txt"
    cp "$NODE_EV/node.txt" "$stepd/node.txt"
    sha256sum "$Q/bin/o0o1_witness_linux.py" "$stepd/launch-env.txt" "$R/app/repo/server.py" "$(readlink -f "$PY")" \
        > "$stepd/inputs-sha256.txt" 2>&1
    utc > "$stepd/utc-start.txt"
    "$HELPY" -I -S -B "$Q/bin/o0o1_witness_linux.py" "$PY" "$RUNTIME" "$R/app/repo/server.py" "$stepd/launch-env.txt" \
        "$version" "$out" "$WITNESS_SECONDS" > "$stepd/witness.stdout" 2> "$stepd/witness.stderr"
    rc=$?; echo "exit_code=$rc" > "$stepd/rc.txt"; utc > "$stepd/utc-end.txt"
    echo "$out"; return "$rc"
}

write_bind() {   # write_bind KIND ROOT INTERPRETER EVIDENCE ENVFILE OUT IMPL: the driver's bind, from this script's bytes
    "$HELPY" -I -S -B -c 'import hashlib, json, os, sys
(kind, root, python, evidence, envfile, out, git, impl, o0, o1, tree, lock, url, branch, cutover, marker,
 local_test, notes, untracked, impl_sha, floor) = sys.argv[1:22]
repo = os.path.join(root, "app", "repo")
pairs = lambda text: dict(line.split("=", 1) for line in text.splitlines() if line)
meta = json.load(open(os.path.join(repo, ".git", "ouroboros-managed.json"), encoding="utf-8"))
fact = json.load(open(os.path.join(evidence, "localfn.json"), encoding="utf-8"))
env = pairs(open(envfile, encoding="utf-8").read())
bind = {"schema": "o0o1-readiness-bind/1", "label": "o0o1-" + kind, "kind": kind, "range": {"o0": o0, "o1": o1},
        "git": git, "implementation": {"dir": impl, "tree": tree, "sha256": pairs(impl_sha)},
        "interpreter": {"path": python, "prefix": fact["prefix"],
                        "flags": {"isolated": 0, "ignore_environment": 0, "no_site": 0, "no_user_site": 0,
                                  "dont_write_bytecode": 1},
                        "evidence": {"dir": evidence}},
        "root": {"app_root": os.path.join(root, "app"), "repo": repo, "data": env["OUROBOROS_DATA_DIR"],
                 "home": env["HOME"], "official_url": url, "branch": branch, "meta": meta},
        "env": env,
        "expect": {"tier": "selected" if kind == "clean" else "full", "target": o1, "floor": pairs(floor),
                   "lock_sha256": lock}}
if kind == "evolved":
    with open(os.path.join(repo, local_test), "rb") as handle:
        probe = hashlib.sha256(handle.read()).hexdigest()
    bind["root"]["evolved"] = {"local_test": local_test, "local_test_sha256": probe, "cutover": cutover,
                               "marker": marker, "notes": notes, "untracked": untracked.split()}
with open(out, "x", encoding="utf-8") as handle:
    handle.write(json.dumps(bind, indent=1, sort_keys=True) + "\n")
print("bind %s" % out)' "$1" "$2" "$3" "$4" "$5" "$6" "$GIT" "$7" "$O0" "$O1" "$IMPL_TREE" "$LOCK_SHA256" \
        "$OFFICIAL_URL" "$BRANCH" "$CUTOVER" "$MARKER" "$LOCAL_TEST" "$NOTES" "$UNTRACKED" "$IMPL_SHA256" "$FLOOR"
}

post_fetch() {   # post_fetch REPO: what the production fetch left in the root (read only): the managed target first
    echo "managed/$BRANCH=$(gq -C "$1" rev-parse -q --verify "refs/remotes/managed/$BRANCH^{commit}" || echo absent) (O1 $O1)"
    gq -C "$1" for-each-ref --format='%(refname) %(objectname)' refs/heads refs/remotes refs/tags refs/ouroboros-managed
}

readiness() {
    local kind="${1:-}" impl="${2:-}" R out chosen tmpd rc line lp pairs=()
    namespace_ok
    case "$kind" in clean|evolved) ;; *) die "readiness clean|evolved IMPL" ;; esac
    impl_ok "$impl"
    R="$Q/runs/o0o1-$kind"
    [ -f "$R/app/repo/.git/ouroboros-managed.json" ] || die "no $kind root at $R (roots first)"
    chosen="$EV/o0o1-witness"
    [ -x "$PY" ] && [ -f "$chosen/localfn.json" ] && [ -f "$chosen/interpreter.txt" ] || die "no witnessed serving interpreter in $chosen (witness first)"
    [ "$(sha256 "$Q/bin/o0o1_readiness.py")" = "$DRIVER_SHA256" ] || die "the driver is not $DRIVER_SHA256"
    lp=$(bound_path) || exit 2
    # The PATH the witness saw O0 serve under is the one the driver runs under.
    [ "$(grep '^PATH=' "$EV/$NS-witness/launch-env.txt" 2>/dev/null)" = "PATH=$lp" ] \
        || die "the witness did not serve under the bound PATH $lp ($EV/$NS-witness/launch-env.txt)"
    tmpd="/tmp/$NS-o0o1-$kind"
    [ ! -e "$tmpd" ] || die "tmp root exists: $tmpd"
    out=$(newdir "$EV/o0o1-$kind-readiness") || exit 2
    mkdir -p -m 700 "$tmpd" || die "tmp root $tmpd"
    launch_pairs "$R" "$tmpd/" "$lp" > "$out/env.txt"
    cp "$NODE_EV/node.txt" "$out/node.txt"
    write_bind "$kind" "$R" "$PY" "$chosen" "$out/env.txt" "$out/bind.json" "$impl" > "$out/bind.txt" 2>&1 \
        || { echo "$out"; die "the bind was not written ($out/bind.txt)"; }
    sha256sum "$out/bind.json" "$Q/bin/o0o1_readiness.py" "$(readlink -f "$PY")" "$out/node.txt" > "$out/inputs-sha256.txt" 2>&1
    while IFS= read -r -d '' line; do pairs+=("$line"); done < <("$HELPY" -I -S -B -c 'import json, sys
for name, value in sorted(json.load(open(sys.argv[1], encoding="utf-8"))["env"].items()):
    sys.stdout.write("%s=%s\0" % (name, value))' "$out/bind.json")
    [ "${#pairs[@]}" -gt 0 ] || die "the bind holds no environment"
    echo "$R/home" > "$out/cwd.txt"
    utc > "$out/utc-start.txt"
    # Exactly the bind's environment and nothing inherited; the interpreter as the installation runs it (no -I/-S),
    # from the root's home.
    (cd "$R/home" && exec env -i "${pairs[@]}" "$PY" "$Q/bin/o0o1_readiness.py" run "$out/bind.json" "$out/driver") \
        > "$out/driver.stdout" 2> "$out/driver.stderr"
    rc=$?; echo "exit_code=$rc" > "$out/rc.txt"; utc > "$out/utc-end.txt"
    cp "$out/driver/verdict.txt" "$out/verdict.txt" 2>/dev/null \
        || echo "verdict=UNKNOWN expected=$kind detail=the driver wrote no verdict (exit $rc)" > "$out/verdict.txt"
    post_fetch "$R/app/repo" > "$out/post-fetch.txt" 2>&1
    echo "$out"; return "$rc"
}

pack() {   # pack LABEL: the candidate scratch beside each root (unnamed runs too), as one tar; the disk afterwards
    local out list rc=0
    namespace_ok
    out=$(newdir "$EV/$NS-pack-$1") || exit 2
    list=$(cd "$Q/runs" 2>/dev/null && ls -d o0o1-*/work/subagent_worktrees/copies.noindex/*.env/update-checks 2>/dev/null)
    echo "$list" > "$out/scratch-paths.txt"
    if [ -n "$list" ]; then tar -C "$Q/runs" -cf "$out/candidate-scratch.tar" $list; rc=$?; fi
    echo "exit_code=$rc" > "$out/rc.txt"
    { du -sk "$Q"/runs/* "$Q/runtime" 2>&1; df -k / /tmp 2>&1; } > "$out/disk.txt"
    echo "$out"; return "$rc"
}

facts() {
    local out
    out=$(newdir "$EV/$NS-facts-$1") || exit 2
    { utc; uname -a; echo "nproc=$(nproc)"; grep -m1 'model name' /proc/cpuinfo; free -m; df -h / /tmp "$ROOT"
      cat /etc/os-release; ulimit -a; "$GIT" --version; "$HELPY" -V; id
      # Observation only: the Node this script's fixed PATH offers, and the runner's usual places for one outside it.
      # The launch PATH is this one plus the bound Node's directory (node); facts adds nothing to any PATH.
      echo "node on PATH=$PATH: $(command -v node || echo absent)"; ls -l /usr/local/bin/node /opt/hostedtoolcache/node
      for f in /proc/pressure/cpu /proc/pressure/memory /proc/pressure/io; do echo "== $f"; cat "$f"; done; } \
        > "$out/facts.txt" 2>&1
    echo "$out"
}

case "${1:-}" in
    facts) label_ok "${2:-}"; facts "$2" ;;
    roots) roots "${2:-}" ;;
    runtime) runtime "${2:-}" "${3:-}" ;;
    node) node_bind "${2:-}" ;;
    witness) witness ;;
    readiness) readiness "${2:-}" "${3:-}" ;;
    pack) label_ok "${2:-}"; pack "$2" ;;
    *) die "facts LABEL | roots clean|evolved | runtime IMPL UV | node NODE | witness | readiness clean|evolved IMPL | pack LABEL" ;;
esac
