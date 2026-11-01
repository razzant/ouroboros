#!/usr/bin/env bash
# One phase inside one disposable container: census, (builder), the interpreter under test, census. See cell.sh.
#   phase.sh PHASE            PHASE is positive or no_index; /q is the volume cell.sh mounted.
# Exit 0 only when every step did; cell.sh records it in phases.txt and verdict.py reads it.
set -uo pipefail
phase=$1
target=/usr/bin/python3
evidence=/q/evidence/$phase
tools=/q/equipment
mkdir -p "$evidence"
census() { "$target" -I -B "$tools/census.py" snapshot "$evidence/census_$1.json"; }
status=0

census before || status=$?
case "$phase" in
  positive)
    # Before any builder or test: is this the condition the cell exists for? exit 3 stops the cell here.
    "$target" -I -B "$tools/probe.py" "$Q_EXPECT" /q/evidence/conditions.json
    rc=$?
    if [ "$rc" -ne 0 ]; then census after; "$target" -I -B "$tools/census.py" diff "$evidence/census_before.json" \
      "$evidence/census_after.json" "$evidence/census_diff.json"; exit "$rc"; fi
    # The builder is uv, not the interpreter under test: --target writes only the driver's directory, and works with no
    # pip, no ensurepip and without touching the marker. Native tags come from this interpreter.
    UV_CACHE_DIR=/q/uv-cache UV_PYTHON_DOWNLOADS=never /q/tools/uv pip install --no-config --no-progress \
      --python "$target" --target /q/driver-site -r /q/subject/requirements-runtime.lock \
      > "$evidence/builder.log" 2>&1 || { echo "builder failed: equipment, no product result" >&2; exit 5; }
    if [ "$Q_EXPECTED_PIP" = "interpreter_pip" ]; then
      "$target" -B -c 'import sys; sys.path[:0] = ["/q/subject", "/q/driver-site"]
from ouroboros.runtime_venv import fetch_pip_wheel
print(fetch_pip_wheel(sys.argv[1]))' /q/wheels > "$evidence/seed_pip_wheel.txt" || exit 5
      export Q_SEED_PIP_WHEEL
      Q_SEED_PIP_WHEEL=$(cat "$evidence/seed_pip_wheel.txt")
    fi
    ;;
esac

driver_rc=0
# shellcheck disable=SC2086  # the manifest's node list, one word each
Q_DRIVER_SITE=/q/driver-site Q_EVIDENCE=$evidence \
  "$target" -B "$tools/target_driver.py" /q/subject /q/driver-site "$evidence/junit.xml" $Q_NODES \
  > "$evidence/pytest.log" 2>&1 || driver_rc=$?
echo "$driver_rc" > "$evidence/driver_rc.txt"
[ "$driver_rc" -eq 0 ] || status=$driver_rc

census after || status=$?
"$target" -I -B "$tools/census.py" diff "$evidence/census_before.json" "$evidence/census_after.json" \
  "$evidence/census_diff.json" || status=$?
exit "$status"
