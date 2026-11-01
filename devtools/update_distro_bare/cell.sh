#!/usr/bin/env bash
# One cell on a disposable GitHub-hosted runner: the real apt state as a frozen image, then fresh containers of it.
#   cell.sh CELL SUBJECT_CHECKOUT RUN_ROOT      expects jq, docker and uv (pinned by the workflow) on PATH.
# Fails closed on an unpinned image or a subject that is not the manifest's. Runs no retries.
set -euo pipefail
cell=$1
subject=$2
q=$3
here=$(cd "$(dirname "$0")" && pwd)
manifest=$here/manifest.json
get() { jq -er "$1" "$manifest"; }

image=$(get '.image.name')
digest=$(jq -r '.image.digest' "$manifest")
if ! [[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "manifest image.digest is unset: the parent pins it when it authorizes the run" >&2
  exit 78
fi
python3 -I -B "$here/subject_check.py" check "$manifest" "$subject"
[ "$(uv --version | cut -d' ' -f2)" = "$(get '.tools.uv')" ]
overlay_test=$(get '.overlay_test')

host_uid=$(id -u)
host_gid=$(id -g)
mkdir -p "$q"/{evidence,tools,wheels,uv-cache,driver-site,subject}
tar -C "$subject" --exclude=.git -cf - . | tar -C "$q/subject" -xf -
cp -R "$here" "$q/equipment"
cp "$here/$(basename "$overlay_test")" "$q/subject/$overlay_test"   # the disposable copy only; the Git subject is untouched
cp "$(command -v uv)" "$q/tools/uv"
chmod -R a+rwX "$q"

packages=$(get ".cells[\"$cell\"].apt | join(\" \")")
uid_mode=$(get ".cells[\"$cell\"].expect.uid")
expected_pip=$(get ".cells[\"$cell\"].expect.expected_pip")
expect=$(jq -ec ".cells[\"$cell\"].expect" "$manifest")
frozen=q-$cell:frozen

# The real apt state: --no-install-recommends, so what is absent is absent because Ubuntu's own packages say so.
docker pull "$image@$digest"
docker run --name "base-$cell" "$image@$digest" sh -euc "
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends python3 git ca-certificates $packages
  groupadd -o -g $host_gid qual
  useradd -u $host_uid -g $host_gid -m qual
  rm -rf /var/lib/apt/lists/*
" > "$q/evidence/apt.log" 2>&1
docker commit "base-$cell" "$frozen" > "$q/evidence/frozen_image_id.txt"
docker rm "base-$cell" > /dev/null
docker image inspect "$image@$digest" --format '{{json .RepoDigests}}' > "$q/evidence/image_digest.json"

user=()
home=/root
if [ "$uid_mode" = "nonzero" ]; then user=(--user "$host_uid:$host_gid"); home=/home/qual; fi

run_phase() {  # run_phase PHASE DOCKER_OPTION...
  local phase=$1 nodes
  shift
  nodes=$(jq -r ".cells[\"$cell\"].phases[\"$phase\"].nodes // [] | join(\" \")" "$manifest")
  docker run --rm "${user[@]}" "$@" -e "HOME=$home" -e "Q_EXPECT=$expect" -e "Q_EXPECTED_PIP=$expected_pip" \
    -e "Q_NODES=$nodes" -e "LANG=C" -v "$q:/q" "$frozen" bash /q/equipment/phase.sh "$phase"
}

status=0
for phase in $(jq -r ".cells[\"$cell\"].phases | keys_unsorted | .[]" "$manifest"); do
  network=$(jq -r ".cells[\"$cell\"].phases[\"$phase\"].network" "$manifest")
  rc=0
  if [ "$network" = "none" ]; then run_phase "$phase" --network none || rc=$?; else run_phase "$phase" || rc=$?; fi
  echo "phase $phase rc=$rc" | tee -a "$q/evidence/phases.txt"
  [ "$rc" -eq 0 ] || status=1
  if [ "$rc" -eq 3 ] || [ "$rc" -eq 5 ]; then break; fi   # condition not met, or the builder failed: nothing more to say
done
docker run --rm -v "$q:/q" "$frozen" chown -R "$host_uid:$host_gid" /q/evidence || true

python3 -I -B "$here/verdict.py" "$manifest" "$cell" "$q/evidence" "$q/evidence/verdict.json" || status=1
exit "$status"
