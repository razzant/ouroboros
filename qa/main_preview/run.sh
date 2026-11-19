#!/usr/bin/env bash
# Called only by the temporary branch workflow. No event-provided commands.
set -euo pipefail
kit="$GITHUB_WORKSPACE/qa/main_preview"
product=$QA_PRODUCT
ui_tree=$QA_UI
python=$QA_PYTHON
node=$QA_NODE
base=b581c570d0613479c1bbb8ae3d1c990183bf8c7f
# Bind the clean product independently of the runner overlay.
expected_tree=29c6469f05c19c2a3f658c05ea6b77322ebea79d

stage() {
  local tested tree path
  cd "$GITHUB_WORKSPACE"
  test "$(git rev-parse HEAD)" = "$GITHUB_SHA"
  # Read the first parent from the verified object: checkout deliberately leaves HEAD shallow.
  # Only that product commit and its fixed base need fetching; don't deepen the earlier preview.
  tested=$(git cat-file commit "$GITHUB_SHA" | awk '!NF {body=1} !body && $1=="parent" && !seen++ {print $2}')
  [[ $tested =~ ^[a-f0-9]{40}$ ]]
  git -c credential.helper= fetch --no-tags --depth=2 origin "$tested"
  test "$(git rev-parse "$tested^1")" = "$base"
  tree=$(git rev-parse "$tested^{tree}")
  test "$tree" = "$expected_tree"
  git diff --name-only --no-renames "$tested" "$GITHUB_SHA" -- > "$QA_RAW/overlay-paths"
  while IFS= read -r path; do
    case "$path" in
      .github/workflows/qa-costs-and-main-preview.yml|qa/main_preview/README.md|qa/main_preview/run.sh|qa/main_preview/receipt.py|qa/main_preview/test_visual_frames.py) ;;
      *) printf 'Unexpected path in temporary overlay.\n' >&2; return 2 ;;
    esac
  done < "$QA_RAW/overlay-paths"
  [[ ! -e $product && ! -e $ui_tree ]]
  git -c core.hooksPath=/dev/null worktree add --detach "$product" "$tested"
  git -c core.hooksPath=/dev/null worktree add --detach "$ui_tree" "$tested"
  test ! -e "$ui_tree/tests/test_qa_costs_main_visual_frames.py"
  install -m 0644 "$kit/test_visual_frames.py" "$ui_tree/tests/test_qa_costs_main_visual_frames.py"
  test -z "$(git -C "$product" status --porcelain=v1 --untracked-files=all)"
  printf '%s\n' "$tested" "$tree" "$base" > "$QA_RAW/product-identity"
}

safe() {
  local dir=$1 lane=$2 seconds=$3
  shift 3
  (cd "$dir" && "$python" -I -S scripts/safe_test.py --temp-parent /tmp --timeout "$seconds" \
    --evidence-dir "$QA_RAW/safe-$lane" -- "$@") > "$QA_RAW/$lane.log" 2>&1
}

pytest_lane() {
  local lane=$1 seconds=$2 dir=$3 marker=$4 code=0
  shift 4
  local -a prefix=()
  if [[ $lane == size ]]; then
    # These are deliberately scrubbed by safe_test; set only the nonsecret selectors inside.
    # The book gate ignores other branch push events. Use its local pairwise mode,
    # not a fictitious production ref or PR event.
    prefix=(env -u GITHUB_EVENT_NAME "OURO_SIZE_RATCHET_BASE_REF=$base" "OURO_BOOK_GROWTH_MODE=block")
  fi
  safe "$dir" "$lane" "$seconds" "${prefix[@]}" "$python" -m pytest -o addopts='' \
    -p no:cacheprovider -q --ci-evidence-dir="$QA_RAW/$lane" -m "$marker" "$@" || code=$?
  "$python" -I -S "$kit/receipt.py" record "$lane" pytest "$code" "$@"
}

visual_paths() {
  export OUROBOROS_UI_EVIDENCE_DIR="$QA_RAW/visual/$1"
  export OUROBOROS_BROWSER_EVIDENCE_OUT="$OUROBOROS_UI_EVIDENCE_DIR"
}

# One place for the parent to extend on integration. Add existing product files or exact
# nodeids below (not globs), and update the evidence allowlist only for reviewed synthetic captures.
main_browser=tests/test_room_work_browser.py::test_native_pill_reaches_each_source_without_acting
parked_browser=tests/test_evolution_browser.py::test_parked_cycle_shows_budget_until_its_explicit_resume
find_browser=(tests/test_history_reply_browser.py::test_ctrl_f_finds_saved_messages_of_the_room_in_front_and_lands_on_them
  tests/test_phone_keyboard_recovery_browser.py::test_the_find_sheet_stays_a_step_under_the_rooms_bar_with_the_keyboard_up)
more_python=(tests/test_main_decisions.py tests/test_main_state_projection.py tests/test_main_durable_edges.py
  tests/test_main_room_membership_agreement.py tests/test_usage_projection_contention.py::test_an_evolution_row_parked_at_exactly_zero_stays_parked)
more_node=(web/tests/evolution_history.test.js web/tests/evolution_wake_outcome.test.js)

frame_checkout() {
  local scene=$1 selector=$2
  if "$python" -I -S "$kit/receipt.py" instrument "$scene" "$selector" > "$QA_RAW/instrument-$scene.log" 2>&1; then
    printf 'instrumented\n' > "$QA_RAW/frame-stage-$scene"
    printf '%s\n' "$ui_tree"
  else
    printf 'capture_anchor_unavailable\n' > "$QA_RAW/frame-stage-$scene"
    printf '%s\n' "$product"
  fi
}

unit_tests=(
  tests/test_anthropic_native_custody.py tests/test_await_messages.py tests/test_builtin_refusal_results.py
  tests/test_cache_optimization.py tests/test_commit_gate_wave_admission.py tests/test_compatible_cache_hints.py
  tests/test_context_economy_window.py::test_a_window_bounded_chosen_read_survives_next_host_rows_and_source_shrink
  tests/test_context_cache_facts.py tests/test_context_drive_state.py tests/test_control_extraction.py
  tests/test_cost_period.py tests/test_cyber_network_settings.py tests/test_deepseek_pricing.py tests/test_deepseek_provider.py
  tests/test_delegate_wake_truthful_facts.py tests/test_design_skill_assets.py tests/test_design_skills_manifest.py
  tests/test_evolution_status.py tests/test_health_subtraction.py tests/test_known_spend_admission.py
  tests/test_llm_provider_golden.py tests/test_main_unread_counter.py tests/test_mcp_api.py tests/test_model_catalog_api.py
  tests/test_native_task_wait.py tests/test_openai_responses.py tests/test_openrouter_diagnostic.py
  tests/test_openrouter_transport_evidence.py tests/test_optional_task_bounds.py tests/test_packet_only_review_pool.py
  tests/test_pricing.py tests/test_provider_connection_api.py tests/test_provider_connections.py tests/test_provider_token_presence.py
  tests/test_query_data.py tests/test_query_data_sql.py tests/test_review_prompt_caching.py tests/test_runtime_mode_core.py
  tests/test_owner_settings_write_seam.py tests/test_settings_budget_hotreload.py tests/test_settings_read_seam.py tests/test_settings_secret_mask.py
  tests/test_subagent_handles.py tests/test_tool_result_delivery.py tests/test_transport_b_stream_deadlines.py
  tests/test_tree_cost_ceiling.py tests/test_typed_tool_refusals.py tests/test_usage_accounting.py tests/test_truthful_usage_events.py tests/test_usage_store.py
  tests/test_budget_paid_admission.py
)
node_tests=(
  web/tests/chat_markdown.test.js web/tests/claudexor_status_store.test.js web/tests/cost_period.test.js
  web/tests/dashboard_settings.test.js web/tests/main_waiting.test.js web/tests/message_share_card.test.js
  web/tests/message_share_file.test.js web/tests/message_share_model.test.js web/tests/model_catalog_store.test.js web/tests/settings_home.test.js
)
layout=tests/test_segmented_layout_browser.py
visual=tests/test_qa_costs_main_visual_frames.py
ordinary='not integration and not browser and not ui_browser and not ui_browser_docker and not skill_smoke and not size_ratchet and not portable_detail'

case "$1" in
  stage) stage ;;
  product-ui)
    failed=0
    for engine in chromium webkit; do
      export OUROBOROS_UI_BROWSER_ENGINE=$engine
      visual_paths "costs/$engine"
      pytest_lane "costs-$engine" 170 "$ui_tree" ui_browser \
        "$visual::test_costs_compares_zero_with_no_hourly_rate" || failed=1
    done
    unset OUROBOROS_UI_BROWSER_ENGINE
    visual_paths evolution
    pytest_lane evolution-layout 290 "$product" ui_browser \
      "$layout::test_every_strip_fits_its_row_and_shows_its_choice" \
      "$layout::test_evolution_measures_break_at_divider_with_wider_font" || failed=1
    parked_checkout=$(frame_checkout evolution "$parked_browser")
    for engine in chromium webkit; do
      export OUROBOROS_UI_BROWSER_ENGINE=$engine
      visual_paths "evolution-parked/$engine"
      pytest_lane "evolution-$engine" 115 "$parked_checkout" ui_browser "$parked_browser" || failed=1
    done
    unset OUROBOROS_UI_BROWSER_ENGINE
    visual_paths main
    main_checkout=$(frame_checkout main "$main_browser")
    pytest_lane main-a 210 "$main_checkout" ui_browser "$main_browser" || failed=1
    visual_paths find
    export CHAT_SEARCH_EVIDENCE_DIR="$OUROBOROS_UI_EVIDENCE_DIR"
    pytest_lane project-find 220 "$product" ui_browser "${find_browser[@]}" || failed=1
    exit "$failed"
    ;;
  share-design)
    failed=0
    for engine in chromium webkit; do
      export OUROBOROS_UI_BROWSER_ENGINE=$engine
      visual_paths "share/$engine"
      pytest_lane "share-$engine" 420 "$product" ui_browser \
        tests/test_message_share_browser.py tests/test_message_share_files_browser.py || failed=1
    done
    unset OUROBOROS_UI_BROWSER_ENGINE
    visual_paths comparison
    pytest_lane comparison 360 "$ui_tree" browser \
      tests/test_design_skill_assets_browser.py::test_compare_page_swaps_in_place_with_the_subject_anchored \
      tests/test_design_skill_assets_browser.py::test_compare_keeps_the_latest_variant_and_scroll_after_a_late_frame \
      tests/test_design_skill_assets_browser.py::test_compare_side_by_side_settle_cannot_repaint_a_newer_choice \
      "$visual::test_comparison_template_frame" || failed=1
    exit "$failed"
    ;;
  python) pytest_lane python 1240 "$product" "$ordinary" "${unit_tests[@]}" "${more_python[@]}" ;;
  node)
    code=0
    safe "$product" node 275 "$node" --test --test-reporter=tap "${node_tests[@]}" "${more_node[@]}" || code=$?
    "$python" -I -S "$kit/receipt.py" record node tap "$code" "${node_tests[@]}" "${more_node[@]}"
    ;;
  structure)
    failed=0
    test -z "$(git -C "$product" status --porcelain=v1 --untracked-files=all)"
    # Read the installed selector's literal floor, without importing the product outside safe_test.
    "$python" -I -S "$kit/receipt.py" floor > "$QA_RAW/floor" || exit 2
    mapfile -t floor < "$QA_RAW/floor"
    pytest_lane floor-structure 420 "$product" "$ordinary" "${floor[@]}" \
      tests/test_docs_sync.py tests/test_domain_manifest.py tests/test_domain_report.py \
      tests/test_architecture_facts.py tests/test_test_environment.py || failed=1
    # Exact size_ratchet marker homes on this public base; avoids collecting unrelated integration modules.
    pytest_lane size 360 "$product" size_ratchet \
      tests/test_reference_book_budgets.py tests/test_repo_health_smoke.py tests/test_smoke.py || failed=1
    code=0
    # context_gate.py itself enters safe_test. Give it time to forward TERM and reap before job timeout.
    (cd "$product" && timeout --signal=TERM --kill-after=45s 680s "$python" scripts/context_gate.py \
      --temp-parent /tmp --workers 2 --evidence-dir "$QA_RAW/context") > "$QA_RAW/context.log" 2>&1 || code=$?
    "$python" -I -S "$kit/receipt.py" record context context "$code" || failed=1
    exit "$failed"
    ;;
  *) exit 2 ;;
esac
