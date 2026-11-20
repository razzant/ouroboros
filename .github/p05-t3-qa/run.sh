#!/usr/bin/env bash
# Only inside the workflow's disposable Ubuntu test unit. No direct local launch.
set -euo pipefail
root=${1:?no isolated QA staging root}
[[ "$root" =~ ^/var/lib/p05-[0-9]+-[0-9]+$ ]]
exec /usr/bin/python3 -I -S -B "$root/harness/p05_proof.py" run "$root"
