#!/usr/bin/env bash
# OB#1234 chain probe (v3) for ONE disposable x86_64 Linux host with sudo and podman
# (GitHub-hosted ubuntu-24.04). Stages, each writing evidence under $WORK/out:
#   run.sh prepare | image | start | arm A | arm B | collect | stop
# No --privileged, default seccomp/AppArmor, no SELinux change, no secrets.
# Root is used only by podman and by the passive /proc sampler.
set -euo pipefail
PROBE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PUBLIC_URL=${OB1234_PUBLIC_URL-}
RESERVE=${OB1234_RESERVE_BYTES:-3000000000}   # kept free for evidence at every stage; no sufficiency claim
QA_UID=1500
CTR=ob1234
ARM_BUDGET=840          # podman exec bound per arm: driver deadline 700 s + start probes and teardown (about 120 s)
SAMPLER_TTL=900         # sampler always ends by itself
WORK_FILE=/tmp/ob1234-work-dir

work() { cat "$WORK_FILE"; }
podman_() { sudo podman --root "$(work)/podman-root" --runroot /run/ob1234-podman "$@"; }
podman_t() { local t=$1; shift; timeout -k 5 "$t" sudo podman --root "$(work)/podman-root" --runroot /run/ob1234-podman "$@"; }
lock() { python3 -I -c "import json;d=json.load(open('$PROBE/sources.lock.json'));print(d['arms']['$1']['$2'])"; }
disk() { df -B1 --output=target,avail,size / /mnt "${RUNNER_TEMP:-/tmp}" 2>&1 | sort -u || echo "df rc=$?"; }

room() {
  # Record free space at a stage start; stop the stage (so its dependents) rather than eat the
  # evidence reserve. $2: bytes the stage is KNOWN to need (an asset size), otherwise 0 = unknown.
  local W avail need=${2:-0}; W=$(work)
  avail=$(df -B1 --output=avail "$W" | tail -1)
  { date -u +%FT%TZ; disk; } > "$W/out/facts/disk-$1.txt"
  if [ "$avail" -lt $((RESERVE + need)) ]; then
    printf '{"stage":"%s","available_bytes":%s,"reserve_bytes":%s,"known_stage_bytes":%s}\n' \
      "$1" "$avail" "$RESERVE" "$need" > "$W/out/facts/disk-stop-$1.json"
    echo "disk: stage $1 not started: $avail < reserve $RESERVE + known $need"; exit 3
  fi
}

prepare() {
  [ "$(uname -m)" = x86_64 ]
  # Pick the mount with the most measured free space; record every candidate.
  local best='' best_avail=0 dir avail
  for dir in /mnt "${RUNNER_TEMP:-}" "$HOME"; do
    [ -n "$dir" ] && [ -d "$dir" ] || continue
    avail=$(df -B1 --output=avail "$dir" | tail -1)
    if [ "$avail" -gt "$best_avail" ]; then best=$dir; best_avail=$avail; fi
  done
  sudo mkdir -p "$best/ob1234" && sudo chown "$(id -u):$(id -g)" "$best/ob1234"
  echo "$best/ob1234" > "$WORK_FILE"
  local W; W=$(work)
  mkdir -p "$W"/{dl,arms/A,arms/B,out/facts}; chmod 1777 "$W/out"
  disk > "$W/out/facts/disk-prepare.txt"
  { uname -a; cat /etc/os-release; nproc; free -b; echo "core_pattern=$(cat /proc/sys/kernel/core_pattern)"
    podman version; python3 -V; sudo aa-status --enabled && echo apparmor=enabled; } > "$W/out/facts/host.txt" 2>&1 || true
  # Check the exact registry object before spending time/space on payload downloads.
  local image_ref; image_ref=$(awk '$1 == "FROM" {print $2}' "$PROBE/Containerfile")
  skopeo inspect --raw "docker://$image_ref" > "$W/out/facts/userspace-manifest.json" 2> "$W/out/facts/registry-check.log"
  # Ubuntu's CNI plugins are an optional network-stack provider. The earlier warning
  # did not establish which backend was active; prepare its standard binaries, not a
  # new backend or a relaxed network policy, and retain the resulting backend facts.
  {
    dpkg-query -W -f='${Package} ${Version} ${db:Status-Abbrev}\n' podman netavark containernetworking-plugins iptables || true
    if [ ! -x /usr/lib/cni/bridge ] || [ ! -x /usr/lib/cni/portmap ] || [ ! -x /usr/lib/cni/firewall ] || [ ! -x /usr/lib/cni/tuning ]; then
      sudo apt-get update
      sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends containernetworking-plugins
    fi
    test -x /usr/lib/cni/bridge -a -x /usr/lib/cni/portmap -a -x /usr/lib/cni/firewall -a -x /usr/lib/cni/tuning
    dpkg-query -W -f='${Package} ${Version} ${db:Status-Abbrev}\n' podman netavark containernetworking-plugins iptables || true
  } > "$W/out/facts/network-dependencies.log" 2>&1
  podman_ info > "$W/out/facts/podman-info.txt" 2>&1 || true
  # One payload at a time: download, verify against the pinned digest AND the release
  # SHA256SUMS, extract, then delete the verified tarball (disposable runner, no duplicate copy).
  local arm url sha name p
  for arm in A B; do
    url=$(lock "$arm" url); sha=$(lock "$arm" sha256); name=$(lock "$arm" asset)
    room "before-download-$arm" "$(lock "$arm" size)"
    curl -fL --retry 3 -o "$W/dl/$name" "$url"
    curl -fL --retry 3 -o "$W/out/facts/SHA256SUMS-$arm" "${url%/*}/SHA256SUMS"
    echo "$sha  $W/dl/$name" | sha256sum -c - | tee "$W/out/facts/sha256-$arm.txt"
    grep -F "$sha  $name" "$W/out/facts/SHA256SUMS-$arm" >> "$W/out/facts/sha256-$arm.txt"
    tar -xzf "$W/dl/$name" -C "$W/arms/$arm"
    rm -f "$W/dl/$name"
    p="$W/arms/$arm/Ouroboros"
    { file "$p/Ouroboros"; du -sb "$p"
      echo '--- top-level _internal shared objects'; find "$p/_internal" -maxdepth 1 -name '*.so*' -printf '%f %s\n' | sort
      echo '--- linux-gtk'; ls -d "$p/_internal/linux-gtk" 2>&1 || true
      echo '--- bundled browsers'; find "$p/_internal" -maxdepth 9 -path '*/.local-browsers/*' -type f \
        \( -name chrome -o -name chrome-headless-shell -o -name headless_shell \) -printf '%p %s\n'
    } > "$W/out/facts/payload-$arm.txt" 2>&1
    disk > "$W/out/facts/disk-after-$arm.txt"
  done
}

image() {
  local W; W=$(work)
  room before-image
  podman_ build --pull=missing -t ob1234-sb43 -f "$PROBE/Containerfile" "$PROBE" > "$W/out/facts/image-build.log" 2>&1
  podman_ image inspect ob1234-sb43 > "$W/out/facts/image-inspect.json"
  for f in before after added; do podman_ run --rm ob1234-sb43 cat "/usr/share/ob1234-rpm-$f.txt" > "$W/out/facts/rpm-$f.txt"; done
  disk > "$W/out/facts/disk-after-image.txt"
}

start() {
  local W; W=$(work)
  room before-start
  podman_ run -d --name "$CTR" --systemd=always --shm-size=1g \
    -v "$W/arms:/opt/ob1234:ro" -v "$W/out:/out" -v "$PROBE:/probe:ro" ob1234-sb43 /sbin/init
  timeout 180 sudo podman --root "$W/podman-root" --runroot /run/ob1234-podman exec "$CTR" \
    systemctl is-system-running --wait > "$W/out/facts/systemd-state.txt" 2>&1 || true
  podman_ exec "$CTR" sh -c 'cat /etc/os-release; rpm -q glibc nss mesa-libEGL dbus-broker git-core;
    systemctl --failed --no-pager; command -v getenforce && getenforce; ulimit -c; id qa' \
    > "$W/out/facts/container.txt" 2>&1 || true
}

ctr_pid() { podman_t 30 inspect -f '{{.State.Pid}}' "$CTR"; }

sampler_start() {
  local W dir; W=$(work); dir="$W/out/$1/sampler"; mkdir -p "$dir"; chmod 1777 "$W/out/$1"
  sudo python3 -I "$PROBE/observer.py" --ns-pid "$(ctr_pid)" --uid "$QA_UID" --out "$dir" \
    --stop-file "$dir/STOP" --ttl "$SAMPLER_TTL" > "$dir/sampler.log" 2>&1 &
  for _ in $(seq 100); do [ -f "$dir/sampler.identity.json" ] && return 0; sleep 0.1; done
  echo 'sampler did not start' >> "$dir/sampler.log"; return 4
}

sampler_stop() {
  # Stop file first; signal only the process whose PID AND starttime match its own record.
  # Non-zero when no complete snapshot exists: an absent snapshot is a gap, never an empty observation.
  local W dir pid start now; W=$(work); dir="$W/out/$1/sampler"
  [ -f "$dir/sampler.identity.json" ] || return 1
  sudo touch "$dir/STOP"
  for _ in $(seq 100); do [ -f "$dir/sampler-meta.json" ] && return 0; sleep 0.1; done
  read -r pid start < <(python3 -I -c "import json;d=json.load(open('$dir/sampler.identity.json'));print(d['pid'],d['starttime'])")
  now=$(sudo awk '{sub(/.*\) /,""); split($0,f," "); print f[20]}' "/proc/$pid/stat" 2>/dev/null || true)
  if [ -n "$start" ] && [ "$now" = "$start" ]; then sudo kill -TERM "$pid"; echo "sampler TERM pid=$pid" >> "$dir/sampler.log"; fi
  for _ in $(seq 200); do [ -f "$dir/sampler-meta.json" ] && return 0; sleep 0.1; done
  return 1
}

arm_cleanup() {
  # Evidence first, then stop every probe-owned process. Idempotent. Every step is bounded and
  # records its own failure in cleanup-errors.txt; any entry makes the arm non-zero.
  local arm=$1 home=$2 W err rc sig; W=$(work); err="$W/out/$arm/cleanup-errors.txt"
  if [ -f "$W/out/$arm/.cleaned" ]; then [ ! -s "$err" ]; return $?; fi
  podman_t 15 exec "$CTR" ps -u qa -o pid=,ppid=,stat=,lstart=,args= > "$W/out/$arm/ps-before-sweep.txt" 2>&1 \
    || echo "ps-before-sweep rc=$?" >> "$err"
  podman_t 45 exec --user qa "$CTR" cp -a "/var/home/qa/$home/Ouroboros/data/logs" "/out/$arm/logs-at-cleanup" \
    || echo "logs-at-cleanup rc=$?" >> "$err"
  for sig in TERM KILL; do
    rc=0; podman_t 15 exec "$CTR" pkill "-$sig" -u qa || rc=$?
    [ "$rc" -le 1 ] || echo "pkill-$sig rc=$rc" >> "$err"   # 1: nothing left to signal
    [ "$sig" = KILL ] || sleep 5
  done
  sampler_stop "$arm" || echo "sampler-stop: no complete snapshot" >> "$err"
  podman_t 30 exec --user qa "$CTR" du -b --max-depth=2 "/var/home/qa/$home" > "$W/out/$arm/home-du.txt" 2>&1 \
    || echo "home-du rc=$?" >> "$err"
  disk > "$W/out/facts/disk-after-arm-$arm.txt" || echo "disk rc=$?" >> "$err"
  touch "$W/out/$arm/.cleaned"
  [ ! -s "$err" ]
}

arm() {
  local arm=$1 payload home W rc crc; W=$(work)
  case "$arm" in A) payload=A; home=state-A ;; B) payload=B; home=state-B ;; *) echo 'arms: A or B'; exit 2 ;; esac
  room "before-arm-$arm"
  mkdir -p "$W/out/$arm"
  trap "arm_cleanup $arm $home" EXIT
  trap "arm_cleanup $arm $home; exit 143" INT TERM
  sampler_start "$arm"
  set +e
  timeout --kill-after=30 "$ARM_BUDGET" sudo podman --root "$W/podman-root" --runroot /run/ob1234-podman \
    exec --user qa --workdir /var/home/qa "$CTR" \
    dbus-run-session -- python3 -I /probe/driver.py --arm "$arm" --payload "/opt/ob1234/$payload/Ouroboros" \
    --home "/var/home/qa/$home" --out "/out/$arm" --public-url "$PUBLIC_URL" --deadline 700 \
    > "$W/out/$arm/driver-console.log" 2>&1
  rc=$?
  echo "driver rc=$rc" >> "$W/out/$arm/driver-console.log"
  arm_cleanup "$arm" "$home"; crc=$?
  set -e
  trap - EXIT INT TERM
  echo "arm $arm: driver rc=$rc, cleanup rc=$crc (evidence in $W/out/$arm)"
  [ "$rc" = 0 ] && [ "$crc" = 0 ]
}

collect() {
  local W; W=$(work)
  sudo python3 -I "$PROBE/analyze.py" --out "$W/out" > "$W/out/collect-summary.json"
}

stop() {
  local W; W=$(work) || return 0
  podman_ rm -f "$CTR" > /dev/null 2>&1 || true
  disk > "$W/out/facts/disk-end.txt" || true
  sudo chown -R "$(id -u):$(id -g)" "$W/out" || true
  echo "$W/out"
}

case "${1:-}" in
  prepare|image|start|collect|stop) "$1" ;;
  arm) arm "$2" ;;
  *) echo "usage: $0 prepare|image|start|arm A|arm B|collect|stop"; exit 2 ;;
esac
