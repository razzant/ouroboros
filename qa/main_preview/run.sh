#!/usr/bin/env bash
# Fixed invocations from the temporary QA workflow; no dispatch-supplied commands.
set -euo pipefail
repo=$GITHUB_WORKSPACE
kit=$repo/qa/main_preview
python=$repo/.venv/bin/python
main_sha=e758a2190298d9890b1f26e632f63775d0b33d55
find_sha=54f57a5b5baf9c6448bc60a3e7788cf7290dd9be
mkdir -p "$QA_RAW" "$QA_OUTPUT" "$QA_OUTPUT/main"

stage_base() {
  local name=$1 sha=$2 dst=$RUNNER_TEMP/qa-$1-base
  git -C "$repo" cat-file -e "$sha^{commit}" 2>/dev/null ||
    git -C "$repo" -c credential.helper= fetch --no-tags --depth=1 origin "$sha"
  git -C "$repo" worktree add --detach "$dst" "$sha"
  test "$(git -C "$dst" rev-parse HEAD)" = "$sha"
  # Share the locked toolchain only while it describes the same dependencies.
  if ! cmp -s "$repo/uv.lock" "$dst/uv.lock" || ! cmp -s "$repo/pyproject.toml" "$dst/pyproject.toml"; then
    echo "The $name baseline and candidate dependency manifests differ." >&2
    return 2
  fi
  local package=$dst/tests/qa_main_preview
  mkdir "$package"
  install -m 0644 "$kit"/{__init__.py,scene.py,variant.py,own_targets.js,capture.py,test_preview.py} "$package/"
  if [[ $name == find ]]; then install -m 0644 "$kit/test_find.py" "$package/"; fi
  git -C "$dst" diff --exit-code HEAD -- web tests ouroboros
}

pytest_lane() (
  local lane=$1 bound=$2 expected=$3 directory=$4 code=0
  shift 4
  cd "$directory" || exit 2
  "$python" -I -S scripts/safe_test.py --temp-parent /tmp --timeout "$bound" \
    --evidence-dir "$QA_RAW/safe-$lane" -- "$python" -m pytest -o addopts='' -p no:cacheprovider \
    -q --ci-evidence-dir="$QA_RAW/$lane" "$@" >"$QA_RAW/$lane.log" 2>&1 || code=$?
  "$python" -I -S "$kit/receipt.py" record "$lane" pytest "$code" "$expected" "$@"
)

case "$1" in
  stage-main) stage_base main "$main_sha" ;;
  stage-find) stage_base find "$find_sha" ;;
  main|find)
    suite=$1
    failed=0
    for engine in chromium webkit; do
      export OUROBOROS_UI_BROWSER_ENGINE=$engine
      case "$suite" in
        main)
          export OUROBOROS_UI_EVIDENCE_DIR=$QA_OUTPUT/main/$engine
          pytest_lane "main-$engine" 540 30 "$RUNNER_TEMP/qa-main-base" \
            -m ui_browser tests/qa_main_preview/test_preview.py || failed=1
          ;;
        find)
          export OUROBOROS_UI_EVIDENCE_DIR=$QA_OUTPUT/find/$engine
          pytest_lane "find-$engine" 180 4 "$RUNNER_TEMP/qa-find-base" \
            -m ui_browser tests/qa_main_preview/test_find.py || failed=1
          ;;
      esac
    done
    exit "$failed"
    ;;
  *) exit 2 ;;
esac
