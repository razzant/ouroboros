#!/usr/bin/env python3
"""One-shot Ubuntu evidence for a detached Nano product; never exports raw reports."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen


def nodes(file, *names):
    return tuple(f"tests/{file}.py::{name}" for name in names)


COLD = nodes(
    "system_e2e/test_cold_return_window_park",
    "test_s47_a_returned_work_that_no_longer_fits_parks_then_resumes_once_it_fits",
    "test_s48_the_owner_ends_a_window_parks_attempt_and_the_ordinary_continue_starts_one_successor",
    "test_s49_a_window_park_offers_the_current_overview_and_only_the_owners_choice_takes_it",
    "test_s50_the_chosen_overview_outlives_an_ordinary_stop_after_its_grant_was_spent",
)
SIZE = nodes(
    "test_smoke", "test_no_oversized_modules", "test_size_ratchet_manifest_matches_live_tree",
    "test_size_ratchet_transition_against_explicit_base", "test_no_extremely_oversized_functions",
) + nodes(
    "test_repo_health_smoke", "test_checked_in_manifest_generator_check_is_exact",
) + nodes(
    "test_reference_book_budgets", "test_a_change_does_not_grow_a_reference_book",
    "test_a_change_does_not_grow_bible_or_the_prompts",
)
UNIT = nodes(
    "test_nano_activation_boundaries", "test_profile_change_invalidates_inflight_proposal_and_unsupported_route_does_not_count",
    "test_settings_ordinary_writer_keeps_old_text_then_saves_smaller_text",
    "test_protection_drops_on_changed_background_and_does_not_hide_root_pressure",
) + nodes(
    "test_nano_minimum", "test_publication_counts_the_complete_candidate_and_preserves_refused_growth",
    "test_work_text_already_in_a_measured_main_input_is_not_a_work_shortfall",
) + nodes(
    "test_nano_knowledge_chronicle_publication", "test_staged_overview_refusal_leaves_active_note_and_index_unchanged",
    "test_shared_broker_refuses_growth_and_retains_transaction_not_journal",
    "test_project_note_uses_no_installation_admission", "test_ordinary_scratch_appends_never_depend_on_a_nano_count",
) + nodes(
    "test_nano_settings_view", "test_a_non_local_main_reports_unsupported_route_without_reading_the_drive",
    "test_work_text_already_in_the_sample_leaves_the_work_floor_unconfirmed_not_short",
    "test_settings_publishes_the_scoped_fact_independently_of_the_measurement",
) + nodes(
    "test_local_server_logits", "test_the_server_entry_leaves_a_direct_command_line_its_own_choices",
) + nodes(
    "test_cold_return", "test_main_offers_resume_for_a_window_park_and_stays_quiet_for_a_pause_that_returns_by_itself",
    "test_stop_pause_time_and_money_decide_before_the_chosen_end",
    "test_only_an_exact_refusal_with_nothing_sent_since_the_return_is_a_parks_proof",
    "test_a_clients_view_of_the_hold_and_the_owners_notice_carry_its_numbers",
) + nodes("test_late_review_marks", "test_a_review_that_does_not_fit_remains_readable_and_owed") + nodes(
    "test_send_clock", "test_only_a_main_loop_opts_in_and_the_zone_is_the_validated_sender_zone",
    "test_the_local_lane_measures_and_sends_the_same_single_line",
) + nodes(
    "test_cold_return",
    *(f"test_a_chosen_overview_is_decided_by_one_note_read_under_the_writers_lock_and_never_fails_the_save[{case}]"
      for case in ("accepted", "moved", "lock_unavailable", "note_unreadable", "foreign_error")),
    *(f"test_a_real_window_park_decides_its_pins_where_its_state_is_built_whatever_the_wait_did[{case}]"
      for case in ("the_overview_is_written", "the_lock_is_lost", "nothing_changes")),
)

UNIT += (
    'tests/test_cold_return.py::test_only_the_scoped_reply_limiter_supplies_the_park_proof[80000-71809-0-nano-True-True]',
    'tests/test_cold_return.py::test_only_the_scoped_reply_limiter_supplies_the_park_proof[80000-70000-2000-nano-True-False]',
    'tests/test_cold_return.py::test_only_the_scoped_reply_limiter_supplies_the_park_proof[80000-71808-2000-nano-True-False]',
    'tests/test_cold_return.py::test_only_the_scoped_reply_limiter_supplies_the_park_proof[80000-71809-0-max-False-False]',
    'tests/test_cold_return.py::test_only_the_scoped_reply_limiter_supplies_the_park_proof[80000-71809-0-nano-False-False]',
    'tests/test_cold_return.py::test_only_the_scoped_reply_limiter_supplies_the_park_proof[120000-111809-0-nano-True-False]',
    'tests/test_cold_return.py::test_recount_compares_the_input_and_both_serving_bindings[None]',
    'tests/test_cold_return.py::test_recount_compares_the_input_and_both_serving_bindings[rows]',
    'tests/test_cold_return.py::test_recount_compares_the_input_and_both_serving_bindings[first_process]',
    'tests/test_cold_return.py::test_recount_compares_the_input_and_both_serving_bindings[second_formatter]',
    'tests/test_cold_return.py::test_recount_compares_the_input_and_both_serving_bindings[second_window]',
    'tests/test_cold_return.py::test_reprojected_root_frame_gives_floor_proof_to_queued_roots_only[main]',
    'tests/test_cold_return.py::test_reprojected_root_frame_gives_floor_proof_to_queued_roots_only[project]',
    'tests/test_cold_return.py::test_reprojected_root_frame_gives_floor_proof_to_queued_roots_only[direct_chat]',
    'tests/test_cold_return.py::test_reprojected_root_frame_gives_floor_proof_to_queued_roots_only[child]',
)

# Earlier cases are retained by behavior, including renamed lock checks.
UNIT += (
    'tests/test_cold_return.py::test_a_carried_choice_is_read_whole_or_not_at_all[True]',
    'tests/test_cold_return.py::test_a_carried_choice_is_read_whole_or_not_at_all[end]',
    'tests/test_cold_return.py::test_a_carried_choice_is_read_whole_or_not_at_all[held2]',
    'tests/test_cold_return.py::test_a_carried_choice_is_read_whole_or_not_at_all[held3]',
    'tests/test_cold_return.py::test_a_carried_choice_is_read_whole_or_not_at_all[held4]',
    'tests/test_cold_return.py::test_a_carried_choice_is_read_whole_or_not_at_all[held5]',
    'tests/test_cold_return.py::test_a_carried_choice_is_read_whole_or_not_at_all[held6]',
    'tests/test_cold_return.py::test_a_choice_is_offered_only_by_the_waiting_hold_it_names',
    'tests/test_cold_return.py::test_a_choice_leaves_its_grant_for_the_works_own_state_and_an_ordinary_restart_restores_it[action0]',
    'tests/test_cold_return.py::test_a_choice_leaves_its_grant_for_the_works_own_state_and_an_ordinary_restart_restores_it[action1]',
    'tests/test_cold_return.py::test_a_hold_that_did_not_take_place_or_a_technical_wait_keeps_the_chosen_end',
    'tests/test_cold_return.py::test_a_park_offers_the_current_overview_only_on_one_exact_form_where_it_fits',
    'tests/test_cold_return.py::test_a_park_takes_the_current_overview_only_for_the_very_revision_its_resume_chose',
    'tests/test_cold_return.py::test_a_restart_after_the_grant_was_spent_still_ends_the_attempt_the_owner_ended',
    'tests/test_cold_return.py::test_a_restored_overview_choice_or_a_lapsed_one_asks_the_model_as_a_plain_return',
    'tests/test_cold_return.py::test_a_return_without_that_choice_asks_the_model_as_before[armed0]',
    'tests/test_cold_return.py::test_a_return_without_that_choice_asks_the_model_as_before[armed1]',
    'tests/test_cold_return.py::test_an_exchange_needs_every_held_text_exactly_once',
    'tests/test_cold_return.py::test_an_outlined_index_is_exchanged_beside_the_address_of_its_own_full_text',
    'tests/test_cold_return.py::test_arming_is_per_start_and_reads_a_choice_only_from_the_grant_of_the_hold_it_names',
    'tests/test_cold_return.py::test_each_parks_notice_keeps_the_numbers_of_its_own_episode',
    'tests/test_cold_return.py::test_no_current_overview_is_offered_when_the_shelf_has_none_a_start_would_hold',
    'tests/test_cold_return.py::test_the_card_uses_the_parks_measured_overview_even_with_a_queue_marker',
    'tests/test_cold_return.py::test_the_choice_is_in_the_handover_checkpoint_before_the_grant_is_spent',
    'tests/test_cold_return.py::test_the_chosen_end_of_an_attempt_is_the_size_terminal_with_nothing_sent',
    'tests/test_cold_return.py::test_the_door_reads_a_choice_whole_or_refuses_it[body0-read0]',
    'tests/test_cold_return.py::test_the_door_reads_a_choice_whole_or_refuses_it[body1-read1]',
    'tests/test_cold_return.py::test_the_door_reads_a_choice_whole_or_refuses_it[body10-refused]',
    'tests/test_cold_return.py::test_the_door_reads_a_choice_whole_or_refuses_it[body11-refused]',
    'tests/test_cold_return.py::test_the_door_reads_a_choice_whole_or_refuses_it[body12-refused]',
    'tests/test_cold_return.py::test_the_door_reads_a_choice_whole_or_refuses_it[body13-refused]',
    'tests/test_cold_return.py::test_the_door_reads_a_choice_whole_or_refuses_it[body14-refused]',
    'tests/test_cold_return.py::test_the_door_reads_a_choice_whole_or_refuses_it[body15-refused]',
    'tests/test_cold_return.py::test_the_door_reads_a_choice_whole_or_refuses_it[body2-choice]',
    'tests/test_cold_return.py::test_the_door_reads_a_choice_whole_or_refuses_it[body3-choice]',
    'tests/test_cold_return.py::test_the_door_reads_a_choice_whole_or_refuses_it[body4-refused]',
    'tests/test_cold_return.py::test_the_door_reads_a_choice_whole_or_refuses_it[body5-refused]',
    'tests/test_cold_return.py::test_the_door_reads_a_choice_whole_or_refuses_it[body6-refused]',
    'tests/test_cold_return.py::test_the_door_reads_a_choice_whole_or_refuses_it[body7-refused]',
    'tests/test_cold_return.py::test_the_door_reads_a_choice_whole_or_refuses_it[body8-refused]',
    'tests/test_cold_return.py::test_the_door_reads_a_choice_whole_or_refuses_it[body9-refused]',
    'tests/test_cold_return.py::test_the_next_window_park_ends_the_choice_whether_or_not_it_could_take_it',
    'tests/test_cold_return.py::test_the_owners_own_pause_ends_a_chosen_end_and_its_resume_is_a_plain_return',
    'tests/test_cold_return.py::test_the_request_counted_with_the_current_overview_is_the_one_the_next_restore_renders[low]',
    'tests/test_cold_return.py::test_the_request_counted_with_the_current_overview_is_the_one_the_next_restore_renders[max]',
    'tests/test_cold_return.py::test_the_request_counted_with_the_current_overview_is_the_one_the_next_restore_renders[nano]',
    'tests/test_nano_activation_boundaries.py::test_direct_repair_edit_is_refused_when_the_base_moves_before_the_write[batch]',
    'tests/test_nano_activation_boundaries.py::test_direct_repair_edit_is_refused_when_the_base_moves_before_the_write[single]',
    'tests/test_nano_activation_boundaries.py::test_direct_repair_text_edit_is_refused_before_any_byte_changes[batch]',
    'tests/test_nano_activation_boundaries.py::test_direct_repair_text_edit_is_refused_before_any_byte_changes[single]',
    'tests/test_nano_activation_boundaries.py::test_enabling_an_ordinary_skill_counts_its_catalogue_row',
    'tests/test_nano_activation_boundaries.py::test_external_edit_inside_a_held_interval_is_never_accepted',
    'tests/test_nano_activation_boundaries.py::test_first_write_after_onboarding_uses_the_first_start_defaults',
    'tests/test_nano_activation_boundaries.py::test_forms_of_a_pending_release_are_counted_on_one_base',
    'tests/test_nano_activation_boundaries.py::test_kept_rollback_reserves_an_ordinary_catalogue_row',
    'tests/test_nano_activation_boundaries.py::test_known_absence_is_a_witness_value_and_a_read_error_is_not',
    'tests/test_nano_activation_boundaries.py::test_mixed_legacy_rollbacks_are_not_proved_by_two_fitting_endpoints',
    'tests/test_nano_activation_boundaries.py::test_new_off_skill_is_counted_by_its_rendered_line',
    'tests/test_nano_activation_boundaries.py::test_only_qualified_competing_replacement_waits_for_settlement[catalogue-only-ordinary-profile]',
    'tests/test_nano_activation_boundaries.py::test_only_qualified_competing_replacement_waits_for_settlement[catalogue-only-qualified-nano]',
    'tests/test_nano_activation_boundaries.py::test_only_qualified_competing_replacement_waits_for_settlement[resident-instructions-ordinary-profile]',
    'tests/test_nano_activation_boundaries.py::test_only_qualified_competing_replacement_waits_for_settlement[resident-instructions-qualified-nano]',
    'tests/test_nano_activation_boundaries.py::test_owner_edit_that_does_not_fit_stays_on_disk_without_acceptance',
    'tests/test_nano_activation_boundaries.py::test_pending_instruction_review_rechecks_before_projecting_its_result[False]',
    'tests/test_nano_activation_boundaries.py::test_pending_instruction_review_rechecks_before_projecting_its_result[True]',
    'tests/test_nano_activation_boundaries.py::test_pending_release_counts_what_a_fresh_main_reads_now_and_after_admission',
    'tests/test_nano_activation_boundaries.py::test_refused_landing_loads_again_what_it_unloaded',
    'tests/test_nano_activation_boundaries.py::test_reinstalling_a_name_whose_switch_stays_on_is_counted',
    'tests/test_nano_activation_boundaries.py::test_release_repairing_a_broken_enabled_instruction_is_counted',
    'tests/test_nano_activation_boundaries.py::test_switch_refusals_name_their_cause_and_only_capacity_speaks_of_size',
    'tests/test_nano_activation_boundaries.py::test_verdict_activating_corrected_files_is_held_and_a_later_one_on_acting_files_is_only_fenced',
    'tests/test_nano_publication_admission.py::test_a_count_without_its_background_is_unavailable_never_free',
    'tests/test_nano_publication_admission.py::test_a_fallback_refusal_receipt_withdraws_the_observation_without_a_count',
    'tests/test_nano_publication_admission.py::test_a_nano_refusal_is_a_typed_unsaved_settings_answer',
    'tests/test_nano_publication_admission.py::test_a_recorded_observation_keeps_the_whole_checked_capability',
    'tests/test_nano_publication_admission.py::test_a_refused_map_edit_keeps_the_map_its_measured_block_and_its_history',
    'tests/test_nano_publication_admission.py::test_a_registry_edit_is_shown_from_its_own_bytes_and_a_refusal_keeps_the_registry',
    'tests/test_nano_publication_admission.py::test_a_reset_or_a_lost_source_withdraws_the_witness',
    'tests/test_nano_publication_admission.py::test_a_rules_commit_carries_only_the_count_taken_for_exactly_these_rules',
    'tests/test_nano_publication_admission.py::test_a_rules_save_that_also_changes_other_settings_records_no_observation',
    'tests/test_nano_publication_admission.py::test_a_smaller_whole_request_does_not_license_background_into_the_reserve[appears]',
    'tests/test_nano_publication_admission.py::test_a_smaller_whole_request_does_not_license_background_into_the_reserve[grows]',
    'tests/test_nano_publication_admission.py::test_a_stopped_server_keeps_its_checked_capability_until_its_launch_values_change',
    'tests/test_nano_publication_admission.py::test_a_write_that_worsens_no_deficit_passes_on_a_crowded_profile[fold]',
    'tests/test_nano_publication_admission.py::test_a_write_that_worsens_no_deficit_passes_on_a_crowded_profile[memory-within-its-ceiling]',
    'tests/test_nano_publication_admission.py::test_a_write_that_worsens_no_deficit_passes_on_a_crowded_profile[smaller-background]',
    'tests/test_nano_publication_admission.py::test_an_edit_before_the_ordinary_start_waits_for_it_and_the_same_retry_then_counts[confirmed-counter]',
    'tests/test_nano_publication_admission.py::test_an_edit_before_the_ordinary_start_waits_for_it_and_the_same_retry_then_counts[not-yet-confirmed]',
    'tests/test_nano_publication_admission.py::test_an_incomplete_source_read_is_its_own_refusal_and_voids_the_old_proof[confirmed]',
    'tests/test_nano_publication_admission.py::test_an_incomplete_source_read_is_its_own_refusal_and_voids_the_old_proof[never-confirmed]',
    'tests/test_nano_publication_admission.py::test_an_unqualified_ticket_commits_after_an_ordinary_writer',
    'tests/test_nano_publication_admission.py::test_an_unqualified_ticket_is_stale_once_the_profile_qualifies',
    'tests/test_nano_publication_admission.py::test_background_may_grow_up_to_the_reserve_and_not_past_it',
    'tests/test_nano_publication_admission.py::test_blocks_that_exist_but_do_not_parse_are_an_incomplete_read[not-a-list]',
    'tests/test_nano_publication_admission.py::test_blocks_that_exist_but_do_not_parse_are_an_incomplete_read[torn]',
    'tests/test_nano_publication_admission.py::test_blocks_that_exist_but_do_not_parse_are_an_incomplete_read[undecodable]',
    'tests/test_nano_publication_admission.py::test_blocks_whose_journal_the_next_start_recreates_are_priced_with_its_reader',
    'tests/test_nano_publication_admission.py::test_both_shown_forms_of_the_catalogue_bind_the_observation',
    'tests/test_nano_publication_admission.py::test_equal_or_smaller_background_passes_even_when_it_already_crowds_the_reserve[36500]',
    'tests/test_nano_publication_admission.py::test_equal_or_smaller_background_passes_even_when_it_already_crowds_the_reserve[37000]',
    'tests/test_nano_publication_admission.py::test_growing_an_over_reserve_background_is_refused_without_an_earlier_proof',
    'tests/test_nano_publication_admission.py::test_invalidation_carries_the_checked_capability_and_a_new_profile_starts_clean',
    'tests/test_nano_publication_admission.py::test_only_a_cyber_save_that_authors_the_rules_is_counted_before_the_document_lock',
    'tests/test_nano_publication_admission.py::test_pending_saved_values_are_never_recorded_as_the_ones_that_serve',
    'tests/test_nano_publication_admission.py::test_profile_key_names_the_serving_launch_not_settings_saved_but_not_applied',
    'tests/test_nano_publication_admission.py::test_saved_values_name_the_serving_launch_only_when_they_start_it_again',
    'tests/test_nano_publication_admission.py::test_the_first_start_view_never_reads_an_unreadable_file_as_absent',
    'tests/test_nano_publication_admission.py::test_the_first_start_view_prices_only_bytes_every_start_writes_alike',
    'tests/test_nano_publication_admission.py::test_the_serving_launch_is_read_from_its_own_command_line_on_every_platform',
    'tests/test_nano_publication_admission.py::test_the_serving_launch_needs_the_binding_of_the_live_process',
    'tests/test_nano_publication_admission.py::test_the_settings_framing_leaves_out_the_rules_every_secret_value_and_explicit_defaults',
    'tests/test_nano_publication_admission.py::test_unqualified_writers_take_the_same_fence_and_exclude_a_snapshot',
    'tests/test_nano_publication_admission.py::test_without_a_binding_the_owner_report_keeps_unapplied_values_apart[pending0-pending:]',
    'tests/test_nano_publication_admission.py::test_without_a_binding_the_owner_report_keeps_unapplied_values_apart[pending1-settings:]',
    'tests/test_nano_reference_capacity.py::test_work_ablation_refuses_pointer_duplicates_or_wrong_author[rows0]',
    'tests/test_nano_reference_capacity.py::test_work_ablation_refuses_pointer_duplicates_or_wrong_author[rows1]',
    'tests/test_nano_reference_capacity.py::test_work_ablation_refuses_pointer_duplicates_or_wrong_author[rows2]',
    'tests/test_nano_reference_capacity.py::test_work_ablation_refuses_pointer_duplicates_or_wrong_author[rows3]',
    'tests/test_nano_startup_exact.py::test_known_empty_and_single_read_typed_failures',
    'tests/test_nano_startup_exact.py::test_scope_isolation_and_domain_boundary',
    'tests/test_nano_startup_exact.py::test_small_ceiling_input_cap_and_counter_unavailability_window_separation',
    'tests/test_nano_startup_exact.py::test_web_main_builder_to_prepared_payload_when_estimator_overcounts',
    'tests/test_nano_startup_exact.py::test_world_reader_after_identical_overflows_preserves_owner_words_and_clean_refusal',
)
# Reuse lifecycle cases already in the earlier selection.
UNIT += nodes(
    "test_nano_activation_boundaries",
    "test_qualified_nano_catalogue_counts_same_floor_after_real_extension_registration",
    "test_ordinary_catalogue_transition_rechecks_profile_before_write",
    "test_stale_replacement_keeps_active_extension_bytes_and_restores_registration",
    "test_starter_keeps_only_transient_nano_turn_on_for_retry[busy-pending]",
    "test_starter_keeps_only_transient_nano_turn_on_for_retry[stale-pending]",
    "test_starter_keeps_only_transient_nano_turn_on_for_retry[over_capacity-installed]",
    "test_pending_unadmitted_on_rollback_retains_its_catalogue_weight",
    "test_pending_rollback_reserves_room_across_another_memory_publication[ordinary-settle]",
    "test_pending_rollback_reserves_room_across_another_memory_publication[startup-recovery]",
    "test_writer_samples_web_runtime_once_without_freezing_the_edited_identity",
    "test_writer_repairs_an_incomplete_saved_identity_only_with_a_measured_fit",
) + nodes(
    "test_nano_publication_admission",
    "test_exact_repair_of_incomplete_baseline_has_to_fit_every_reserve[36808-20000-]",
    "test_exact_repair_of_incomplete_baseline_has_to_fit_every_reserve[36809-10000-over_capacity]",
    "test_exact_repair_of_incomplete_baseline_has_to_fit_every_reserve[30000-20001-over_capacity]",
    "test_block_reader_io_failure_is_not_a_repairable_parse_problem",
    "test_publication_does_not_activate_a_missing_chronicle",
) + nodes(
    "test_nano_startup_exact", *(
        f"test_start_count_waits_on_a_busy_slot_and_hands_its_control_to_the_loop[{control}]"
        for control in ("release", "pause", "stop", "deadline", "wrap_up")
    ),
) + nodes(
    "test_memory_fallback",
    "test_nano_publication_retries_the_same_paid_draft[busy]",
    "test_nano_publication_retries_the_same_paid_draft[stale]",
    "test_nano_publication_retries_the_same_paid_draft[over_capacity]",
    "test_unreadable_nano_paid_draft_does_not_redispatch",
) + nodes(
    "test_cold_return",
    "test_a_returned_round_excludes_a_direct_chat_in_either_service_representation",
    "test_current_window_offer_uses_the_painted_stop_identity_and_needs_readable_tree",
    "test_clean_window_retries_do_not_grow_their_transcript_or_edit_prior_words",
)

# Ordinary Web/Main start; the separate Chromium phase owns the actual composer click.
UNIT += nodes(
    "test_nano_web_main_entry",
    "test_routed_web_send_starts_exact_with_its_whole_input_and_its_row_said_once",
    "test_only_the_attested_row_yields_an_identical_earlier_message_stays_whole",
    "test_a_substituted_or_changed_input_keeps_the_row_whole",
    "test_a_web_reply_keeps_its_row_whole_with_what_it_replies_to",
    *(f"test_a_control_met_by_an_unproven_count_binds_before_the_estimate[{action}]"
      for action in ("pause", "stop", "deadline", "wrap_up")),
    "test_the_start_candidate_is_my_own_focus_on_the_qualified_local_main_route",
    *(f"test_a_control_met_by_the_last_count_binds_before_the_loop[{action}]"
      for action in ("pause", "stop", "deadline", "wrap_up")),
    "test_required_handbook_refusal_counts_the_whole_request_it_needs",
)

# Publication, reader fallback and retained skill rollback share the current minimum.
UNIT += (
    "tests/test_nano_activation_boundaries.py::test_writer_compares_complete_world_even_when_root_would_address_it",
    "tests/test_nano_minimum.py::test_world_reader_keeps_its_form_when_the_index_also_outlines_and_records_omitted_tokens",
    "tests/test_nano_publication_admission.py::test_world_growth_cannot_escape_into_its_reader_or_grow_behind_it_while_repairs_pass",
    "tests/test_nano_publication_admission.py::test_fresh_exact_count_clears_stale_counter_unavailable_without_reviving_protection",
    "tests/test_nano_activation_boundaries.py::test_off_skill_pending_rollback_is_priced_when_visible_and_skipped_when_unchanged",
    "tests/test_skill_update_retention.py::test_lock_contention_before_restore_keeps_update_snapshot_retryable_and_avoids_nested_lock[hub]",
    "tests/test_skill_update_retention.py::test_lock_contention_before_restore_keeps_update_snapshot_retryable_and_avoids_nested_lock[registry]",
    "tests/test_skill_update_retention.py::test_partial_restore_retains_evidence_and_never_replaces_payload_twice",
)

# Whole-index counting must not be replaced by the smaller start form. Keep the
# early formatter, count-frame drift and installation-identity controls beside it.
UNIT += nodes(
    "test_nano_reference_capacity",
    "test_reference_uses_the_exact_start_and_keeps_a_separate_catalogue[False]",
    "test_reference_uses_the_exact_start_and_keeps_a_separate_catalogue[True]",
) + nodes(
    "test_nano_minimum",
    "test_full_index_counts_for_publication_even_when_the_physical_floor_uses_its_outline[57000-whole-False]",
    "test_full_index_counts_for_publication_even_when_the_physical_floor_uses_its_outline[75000-outline-False]",
    "test_full_index_counts_for_publication_even_when_the_physical_floor_uses_its_outline[75000-whole-True]",
    "test_count_only_publication_reaches_formatter_despite_large_estimated_core",
    "test_publication_bracket_uses_one_mcp_map_despite_catalogue_drift",
    "test_publication_binds_one_bootstrapped_installation",
    "test_aba_of_active_memory_invalidates_an_inflight_capture",
) + nodes(
    "test_nano_startup_exact", "test_native_shortage_keeps_marks_history_and_full_index_despite_estimator",
) + nodes(
    "test_nano_activation_boundaries",
    "test_deferred_local_port_change_counts_only_the_dispatch_binding",
    "test_script_update_with_renamed_entry_reads_candidate_files_and_keeps_destination_bucket[update-ouroboroshub-trusted]",
    "test_script_update_with_renamed_entry_reads_candidate_files_and_keeps_destination_bucket[update-ouroboroshub-review]",
    "test_script_update_with_renamed_entry_reads_candidate_files_and_keeps_destination_bucket[replace-clawhub-trusted]",
    "test_missing_entry_in_candidate_or_rollback_is_counted_from_its_own_version_files",
) + nodes(
    "test_nano_publication_admission",
    "test_overlapping_max_scratchpad_writes_both_land_without_a_count",
    "test_name_tier_storage_preserves_max_settings_save_and_nano_serialization[enolck]",
    "test_name_tier_storage_preserves_max_settings_save_and_nano_serialization[eopnotsupp]",
    "test_unsupported_lock_errors_and_unreadable_identities_fail_closed",
    "test_nano_publication_tiers_exclude_on_one_name[False]",
    "test_nano_publication_tiers_exclude_on_one_name[True]",
)

# Use exact function selectors for these complete source families, and require every expected collected case.
UNIT_FAMILIES = {
    'tests/test_nano_publication_admission.py::test_a_smaller_whole_request_does_not_license_background_into_the_reserve': ('tests/test_nano_publication_admission.py::test_a_smaller_whole_request_does_not_license_background_into_the_reserve[grows]', 'tests/test_nano_publication_admission.py::test_a_smaller_whole_request_does_not_license_background_into_the_reserve[appears]'),
    'tests/test_nano_publication_admission.py::test_a_write_that_worsens_no_deficit_passes_on_a_crowded_profile': ('tests/test_nano_publication_admission.py::test_a_write_that_worsens_no_deficit_passes_on_a_crowded_profile[smaller-background]', 'tests/test_nano_publication_admission.py::test_a_write_that_worsens_no_deficit_passes_on_a_crowded_profile[fold]', 'tests/test_nano_publication_admission.py::test_a_write_that_worsens_no_deficit_passes_on_a_crowded_profile[memory-within-its-ceiling]'),
    'tests/test_nano_publication_admission.py::test_an_incomplete_source_read_is_its_own_refusal_and_voids_the_old_proof': ('tests/test_nano_publication_admission.py::test_an_incomplete_source_read_is_its_own_refusal_and_voids_the_old_proof[never-confirmed]', 'tests/test_nano_publication_admission.py::test_an_incomplete_source_read_is_its_own_refusal_and_voids_the_old_proof[confirmed]'),
    'tests/test_nano_publication_admission.py::test_blocks_that_exist_but_do_not_parse_are_an_incomplete_read': ('tests/test_nano_publication_admission.py::test_blocks_that_exist_but_do_not_parse_are_an_incomplete_read[torn]', 'tests/test_nano_publication_admission.py::test_blocks_that_exist_but_do_not_parse_are_an_incomplete_read[undecodable]', 'tests/test_nano_publication_admission.py::test_blocks_that_exist_but_do_not_parse_are_an_incomplete_read[not-a-list]'),
    'tests/test_nano_publication_admission.py::test_equal_or_smaller_background_passes_even_when_it_already_crowds_the_reserve': ('tests/test_nano_publication_admission.py::test_equal_or_smaller_background_passes_even_when_it_already_crowds_the_reserve[37000]', 'tests/test_nano_publication_admission.py::test_equal_or_smaller_background_passes_even_when_it_already_crowds_the_reserve[36500]'),
    'tests/test_nano_publication_admission.py::test_without_a_binding_the_owner_report_keeps_unapplied_values_apart': ('tests/test_nano_publication_admission.py::test_without_a_binding_the_owner_report_keeps_unapplied_values_apart[pending0-pending:]', 'tests/test_nano_publication_admission.py::test_without_a_binding_the_owner_report_keeps_unapplied_values_apart[pending1-settings:]'),
    'tests/test_nano_reference_capacity.py::test_work_ablation_refuses_pointer_duplicates_or_wrong_author': ('tests/test_nano_reference_capacity.py::test_work_ablation_refuses_pointer_duplicates_or_wrong_author[rows0]', 'tests/test_nano_reference_capacity.py::test_work_ablation_refuses_pointer_duplicates_or_wrong_author[rows1]', 'tests/test_nano_reference_capacity.py::test_work_ablation_refuses_pointer_duplicates_or_wrong_author[rows2]', 'tests/test_nano_reference_capacity.py::test_work_ablation_refuses_pointer_duplicates_or_wrong_author[rows3]'),
}
# Exact nodeids on the bound composition; changes must be declared, never silently deselected.
CONTRACTS = nodes(
    "test_persistence_inventory", "test_scan_is_populated_and_pinned", "test_every_scanned_path_has_an_inventory_row",
    "test_every_inventory_row_is_still_real", "test_actor_source_kinds_have_their_own_retention_row_or_known_debt",
) + nodes(
    "test_nano_reference_capacity", "test_work_ablation_removes_exactly_one_span_from_one_user_cell",
    "test_work_witness_is_not_charged_again_after_its_input_was_measured",
) + nodes(
    "test_memory_publication", "test_registry_refusal_exposes_the_private_proposal_reference",
) + nodes(
    "test_nano_activation_boundaries",
    "test_settings_nano_refusals_echo_owner_input_without_claiming_growth[busy-Try again when it is free.]",
    "test_settings_nano_refusals_echo_owner_input_without_claiming_growth[unavailable-Try again when it is available.]",
    "test_settings_nano_refusals_echo_owner_input_without_claiming_growth[stale-Check the current rules before retrying.]",
    "test_settings_nano_refusals_echo_owner_input_without_claiming_growth[bootstrap_pending-Start a chat, then save again.]",
    "test_settings_nano_refusals_echo_owner_input_without_claiming_growth[incomplete-Check the reported source before saving again.]",
) + nodes(
    "test_domain_manifest", "test_manifest_covers_the_tracked_population_exactly",
    "test_every_module_maps_to_a_known_domain",
    "test_strict_direction_matrix_matches_the_pinned_baseline",
    "test_lazy_and_dynamic_classification_matches_the_pin",
) + nodes(
    "test_architecture_facts", "test_owner_of_covers_every_manifest_module",
) + nodes(
    "test_docs_sync", "test_architecture_component_map_covers_every_live_runtime_module",
) + nodes(
    "test_generated_inventories", "test_data_layout_inventory_is_byte_identical",
    "test_facade_inventory_is_byte_identical",
) + nodes(
    "test_compact_context_schema", "test_schema_cost_and_all_typed_bindings",
    "test_compact_tool_stays_in_nano_and_low_max_keep_the_permitted_envelope",
) + nodes(
    "test_context_start_bound",
    "test_a_frame_the_window_cannot_hold_is_refused_before_the_memory_capture",
    "test_a_frame_just_under_the_window_reaches_the_ordinary_fit_unchanged",
    "test_no_arrived_answer_and_an_unknown_window_never_refuse_early",
    "test_the_connections_term_is_the_form_with_fewer_escaped_characters",
) + nodes(
    "test_context_minimum_view_unfit", "test_a_hopeless_nano_frame_skips_the_memory_render",
)

VIEWPORTS = ((1087, 740), (320, 640))
THEMES = ("dark", "light")
SETTINGS = nodes("test_settings_tools_models_browser", *(
    f"test_nano_context_projects_passive_facts_in_models[{theme}-{w}-{h}]"
    for theme in THEMES for w, h in VIEWPORTS
))
SETTINGS_FRAMES = {f"nano-context-{state}-{w}-{theme}.png": (w, h)
                   for state in ("summary", "measurement", "unconfirmed") for w, h in VIEWPORTS for theme in THEMES}
# Verify these declarations against the real-card fixture on the final product before releasing the overlay.
RESUME = nodes("test_cold_return_window_park_browser", "test_saved_work_card_uses_the_current_overview_then_resumes")
RESUME_SCENE = "cold-return-window-park"
RESUME_NAMES = (
    "01-held-does-not-fit.png", "02-dialog-three-steps.png", "03-choice-sent.png",
    "04-current-overview-held-after-reload.png", "05-dialog-can-continue.png", "06-continued.png",
)
# Must match the permanent fixture included in the exact tested product.
RESUME_VIEWPORT = (1440, 900)
RESUME_FRAMES = {name: RESUME_VIEWPORT for name in RESUME_NAMES}

PHASES = {"setup", "call", "teardown"}

TASK_CONTROL_CASES = (
    "a window hold offers its own steps, and each is one request naming the hold it was shown for",
    "a window hold the task's own read does not show sends nothing and says so",
    "the card's window hold is the accepted snapshot's: an older body answering late neither hides it nor sends a plain Resume",
)
MODEL_ROLES_CASES = ("Nano Settings distinguishes new memory checks from a dated count",)


def reference_pins():
    tree = ast.parse(Path("scripts/nano_reference_capacity.py").read_bytes())
    wanted = {"MODELS", "HUB_REVISION", "ANTI_SLOP", "SKILL_PIN", "ANTI_SLOP_BODY_SHA256", "REFERENCE_DEPS"}
    return {stmt.targets[0].id: ast.literal_eval(stmt.value) for stmt in tree.body
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id in wanted}


def stage_reference(cache):
    """Fetch only the five public data pins, without importing the candidate."""
    if not sys.platform.startswith("linux") or os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("Reference staging is for the approved ephemeral Ubuntu job")
    values = reference_pins()
    models = values["MODELS"]
    if set(models) != {"qwen35", "gemma4"}:
        raise SystemExit("The public pin set changed")
    pins = []
    for label, (repo, rev, _bos, assets) in models.items():
        if (not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)
                or not re.fullmatch(r"[0-9a-f]{40}", rev)
                or set(assets) != {"tokenizer.json", "chat_template.jinja"}):
            raise SystemExit("Unexpected model pin")
        pins.extend((cache / label / name, f"https://huggingface.co/{repo}/resolve/{rev}/{name}", pin)
                    for name, pin in assets.items())
    hub, name = values["HUB_REVISION"], values["ANTI_SLOP"]
    if not re.fullmatch(r"[0-9a-f]{40}", hub) or name != "anti-slop.SKILL.md":
        raise SystemExit("Unexpected instruction pin")
    pins.append((cache / name, f"https://raw.githubusercontent.com/razzant/OuroborosHub/{hub}/skills/anti-slop/SKILL.md",
                 values["SKILL_PIN"]))
    cache.mkdir()
    for dst, url, (length, digest) in pins:
        if not 0 < length <= 35_000_000 or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SystemExit("Invalid public file pin")
        with urlopen(url, timeout=45) as response:
            data = response.read(length + 1)
        if len(data) != length or hashlib.sha256(data).hexdigest() != digest:
            raise SystemExit(f"Pin integrity failure: {dst.name}")
        dst.parent.mkdir(exist_ok=True)
        with dst.open("xb") as handle:
            handle.write(data)


def node_proof(record, wanted, exit_code, identity):
    """Project only known node ids and terminal phase facts from the stock pytest reporter."""
    selected = record.get("selected_nodeids")
    report_rows = record.get("reports")
    expected = set(wanted)
    collected = (isinstance(selected, list) and all(isinstance(node, str) for node in selected)
                 and len(selected) == len(expected) and set(selected) == expected)
    observed = defaultdict(lambda: defaultdict(list))
    foreign_or_invalid = not isinstance(report_rows, list)
    for row in report_rows if isinstance(report_rows, list) else ():
        if not isinstance(row, dict):
            foreign_or_invalid = True
            continue
        node, phase, outcome = (row.get(key) for key in ("nodeid", "phase", "outcome"))
        if (not isinstance(node, str) or node not in expected or not isinstance(phase, str)
                or phase not in PHASES or outcome not in ("passed", "failed", "skipped")
                or row.get("wasxfail") is not None):
            foreign_or_invalid = True
            continue
        if row.get("is_subtest") is True:
            # A passing subtest is no substitute for call; only the ordinary node can qualify.
            foreign_or_invalid |= outcome != "passed"
            continue
        observed[node][phase].append(outcome)
    complete = {node for node, readings in observed.items()
                if set(readings) == PHASES and all(events == ["passed"] for events in readings.values())}
    environment = record.get("environment")
    bound = (record.get("format") == 1 and record.get("github") == identity
             and type(record.get("session_exit_code")) is int
             and isinstance(environment, dict) and environment.get("platform") == "Linux")
    accepted = (bool(wanted) and collected and not foreign_or_invalid and bound and exit_code == 0
                and record["session_exit_code"] == 0 and complete == expected
                and record.get("collection_failures") == record.get("collection_skips") == [])
    return {"accepted": accepted, "pytest_exit": exit_code, "session_bound": bound,
            "selection_matches": collected, "invalid_or_outside_reports": foreign_or_invalid,
            "selected_count": len(selected) if isinstance(selected, list) else None,
            "missing_from_selection": sorted(expected - {x for x in selected if isinstance(x, str)})
            if isinstance(selected, list) else sorted(expected),
            "foreign_selection_count": sum(not isinstance(x, str) or x not in expected for x in selected)
            if isinstance(selected, list) else None,
            "collection_skip_count": len(record["collection_skips"]) if isinstance(record.get("collection_skips"), list) else None,
            "collection_failure_count": len(record["collection_failures"]) if isinstance(record.get("collection_failures"), list) else None,
            "passed_nodes": sorted(complete), "incomplete_nodes": sorted(expected - complete),
            "observed_phases": {node: dict(sorted(observed.get(node, {}).items())) for node in sorted(expected)}}


def frame_proof(directory, expected):
    from PIL import Image

    if not expected or directory.is_symlink() or not directory.is_dir():
        return {}, False
    frames = {}
    actual = {p.name for p in directory.iterdir()}
    for name, size in sorted(expected.items()):
        if name not in actual or not re.fullmatch(r"[a-zA-Z0-9_.-]+\.png", name):
            continue
        path = directory / name
        if path.is_symlink() or not path.is_file():
            continue
        try:
            body = path.read_bytes()
            with Image.open(io.BytesIO(body), formats=["PNG"]) as image:
                if image.size != size:
                    continue
                image.verify()
            with Image.open(io.BytesIO(body), formats=["PNG"]) as image:
                image.load()
        except (OSError, SyntaxError, ValueError):
            continue
        frames[name] = (body, {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "size": size})
    return frames, set(frames) == actual == set(expected)


def exact_selectors(wanted, families):
    """Expand no IDs locally: run each named function and reconcile its exact full matrix."""
    if not wanted or len(set(wanted)) != len(wanted):
        raise ValueError("Empty or duplicate exact case inventory")
    expected = set(wanted)
    extra = set()
    for selector, cases in families.items():
        if ("[" in selector or not cases or len(set(cases)) != len(cases)
                or any(not case.startswith(selector + "[") or not case.endswith("]") for case in cases)
                or selector in expected or set(cases) != {node for node in expected if node.startswith(selector + "[")}
                or extra.intersection(cases)):
            raise ValueError(f"The required parameter matrix is unbound: {selector}")
        extra.update(cases)
    return [node for node in wanted if node not in extra] + list(families)


def test_phase(name, wanted, scratch, upload, identity, base, runner, frames=None):
    if not wanted or frames is not None and not frames:
        return {"accepted": False, "state": "unbound_resume_card_not_run"}
    env = dict(os.environ)
    args = []
    shots = scratch / (name + "-frames")
    if name == "size":
        # Run the explicit local pair; a QA push is not an official-line push. No fake ref/event.
        env.pop("GITHUB_EVENT_NAME", None)
        env.update(OURO_SIZE_RATCHET_BASE_REF=base, OURO_BOOK_GROWTH_MODE="block")
        args = ["-m", "size_ratchet"]
    elif name in ("s47", "s48", "s49", "s50"):
        env["OUROBOROS_E2E_DEEP"] = "mock"
        args = ["-m", "integration"]
    elif name == "web-composer":
        env.pop("OUROBOROS_UI_BROWSER_EXECUTABLE", None)
        env.update(OUROBOROS_E2E_DEEP="mock", PLAYWRIGHT_BROWSERS_PATH=str(runner / "nano-chromium"))
        args = ["-m", "browser"]
    elif frames is not None:
        shots.mkdir()
        env.pop("OUROBOROS_UI_BROWSER_EXECUTABLE", None)
        env["PLAYWRIGHT_BROWSERS_PATH"] = str(runner / "nano-chromium")
        if name == "resume-card":
            env.update(OUROBOROS_E2E_DEEP="mock", OUROBOROS_BROWSER_EVIDENCE_OUT=str(shots))
            shots = shots / RESUME_SCENE
            args = ["-m", "browser"]
        else:
            env.update(OUROBOROS_RUN_UI_SMOKE="1", OUROBOROS_UI_BROWSER_ENGINE="chromium",
                       OUROBOROS_UI_EVIDENCE_DIR=str(shots))
            args = ["-m", "ui_browser"]
    families = UNIT_FAMILIES if name == "units" else {}
    argv_selectors = exact_selectors(wanted, families)
    evidence = scratch / name
    evidence.mkdir()
    code = subprocess.call([sys.executable, "-m", "pytest", "-o", "addopts=", "-p", "no:cacheprovider", "-q",
                            "--ci-evidence-dir", str(evidence), *args, *argv_selectors], env=env)
    try:
        record = json.loads((evidence / "results.json").read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            record = {}
    except (OSError, ValueError):
        record = {}
    proof = node_proof(record, wanted, code, identity)
    if families:
        proof["required_function_matrices"] = families
    if frames is not None:
        checked, complete = frame_proof(shots, frames)
        proof["accepted"] &= complete
        proof["expected_frames"] = sorted(frames)
        proof["frame_inventory_complete"] = complete
        proof["missing_or_invalid_frames"] = sorted(set(frames) - checked.keys())
        proof["frames"] = {file: meta for file, (_, meta) in checked.items()}
        if checked:
            target = upload / name
            target.mkdir()
            for file, (blob, _) in checked.items():
                (target / file).write_bytes(blob)
    return proof


def check_reference(path):
    """Validate the meaning of the positive public report, not just its exit code."""
    report = json.loads(Path(path).read_bytes())
    limits = report["memory_work_reply_reserved"]
    if (not isinstance(limits, list) or limits != [20_000, 15_000, 8_192]
            or any(type(n) is not int for n in limits) or report["frame_limit"] != 80_000 - sum(limits)):
        raise SystemExit("Reference report does not measure the 80k/20k/15k contract and one reply reserve")
    memory_limit, work_floor, reply = limits

    def digest(value):
        return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))

    # Match the already-bound script's public pins without importing it on the staging or proof path.
    pins = reference_pins()
    models = {name: {"repo": repo, "revision": rev, "asset_sha256": {path: pin[1] for path, pin in assets.items()}}
              for name, (repo, rev, _bos, assets) in pins["MODELS"].items()}
    if (report["models"] != models or report["reference_dependencies"] != pins["REFERENCE_DEPS"]
            or report["anti_slop_hub_revision"] != pins["HUB_REVISION"]
            or report["anti_slop_file_sha256"] != pins["SKILL_PIN"][1]):
        raise SystemExit("Reference model, dependency or instruction source drifted")
    if report["start_reference_kind"] != (
        "exact startup selector over pinned HF counts; synthetic queued Web Main, no serving formatter or request sent"
    ):
        raise SystemExit("Reference report does not claim the exact synthetic startup surface")
    if report["framing"] != {
        "add_generation_prompt": True, "enable_thinking": False, "eos_token": "",
        "bos_token_by_model": {name: pin[2] for name, pin in pins["MODELS"].items()},
        "add_special_tokens": False, "truncation": False, "padding": False,
        "clock": "one fixed trailing user row", "reply": "reserved, not tokenized",
        "work_occurrences_checked_in": ["prepared messages", "rendered template"],
    }:
        raise SystemExit("Reference framing changed")
    profiles = report["profiles"]
    if set(profiles) != {"fresh_bare", "fresh_recommended", "mature_recommended"}:
        raise SystemExit("Reference report profile selection changed")
    clock_rows = set()
    for name, paired in profiles.items():
        if set(paired) != {"qwen35", "gemma4"}:
            raise SystemExit(f"{name}: both tokenizers are required")
        qwen, gemma = paired["qwen35"], paired["gemma4"]
        for key in ("capture_sha256", "tool_schema_sha256", "source_sha256", "full_index_sha256",
                    "installed_anti_slop_catalogue_row_sha256", "extension_reader_sha256"):
            if not qwen[key] or qwen[key] != gemma[key]:
                raise SystemExit(f"{name}: the two tokenizers did not use the same captured source ({key})")
        if any(qwen["work_pair"][key] != gemma["work_pair"][key] for key in ("fixed_fields_sha256", "work_sha256")):
            raise SystemExit(f"{name}: the two tokenizers did not use the same fixed fields and work")
        for model, row in paired.items():
            pair = row["work_pair"]
            joint = row["joint_witness"]
            if not all(type(n) is int and n > 0 for n in (
                row["F"], row["F0"], row["B"], row["B0_frame"], row["WB0_frame"],
                *(pair[k] for k in ("W", "W0", "WB", "prepared_reply_cap_tokens", "allocated_reply_cap_tokens")),
            )):
                raise SystemExit(f"{name}/{model}: an exact count or reply cap is missing")
            memory = pair["W"] - pair["WB"]
            full_memory = row["F"] - row["B"]
            minimum_memory = memory_limit * 3 // 4 if name.startswith("mature_") else 1
            frame_left = report["frame_limit"] - row["B0_frame"] - row["unspent_document_reserve_tokens"]
            remaining = 80_000 - pair["W"]
            if (row["errors"] != [] or row["negative_control"] is not False
                    or row["installed_anti_slop_catalogue_row_observed"] is not (name != "fresh_bare")
                    or row["anti_slop_body_sha256"] != (None if name == "fresh_bare" else pins["ANTI_SLOP_BODY_SHA256"])
                    or row["extension_catalogue_observed"] is not True
                    or not all(digest(row[k]) for k in ("capture_sha256", "full_index_sha256", "extension_reader_sha256",
                                                        "installed_anti_slop_catalogue_row_sha256", "tool_schema_sha256"))
                    or pair["W0_kind"] != "diagnostic_ablation_only" or pair["WB_kind"] != "diagnostic_ablation_only"
                    or row["F_kind"] != "whole_knowledge_index" or pair["startup_probe_matches_W"] is not True
                    or not minimum_memory <= memory == row["memory_W_minus_WB"] <= memory_limit
                    or not minimum_memory <= full_memory == row["whole_index_memory_minus_B"] <= memory_limit
                    or [pair[f"occurrences_in_{k}"] for k in ("W", "W0", "WB")] != [1, 0, 1]
                    or not all(digest(pair[k]) for k in ("W_sha256", "W0_sha256", "WB_sha256", "clock_row_sha256",
                                                         "fixed_fields_sha256", "work_sha256"))
                    or len({pair[k] for k in ("W_sha256", "W0_sha256", "WB_sha256")}) != 3
                    or set(pair["rendered_prompt_sha256"]) != {"W", "W0", "WB"}
                    or not all(digest(d) for d in pair["rendered_prompt_sha256"].values())
                    or len(set(pair["rendered_prompt_sha256"].values())) != 3
                    or pair["work_sha256"] != report["work_sha256"]
                    or not re.fullmatch(r"messages\[[1-9]\d*\]\.content", pair["changed_field"])
                    or pair["memory_changed_field"] != "messages[0].content"
                    or not work_floor <= pair["W"] - pair["W0"] == row["work_W_minus_W0"]
                    or not 0 <= row["reply_floor_remaining_after_W"] == remaining - reply
                    or not reply <= pair["allocated_reply_cap_tokens"] <= min(pair["prepared_reply_cap_tokens"], remaining)
                    or not 0 <= frame_left == row["uncommitted_frame_tokens"]):
                raise SystemExit(f"{name}/{model}: the actual start and ablations did not qualify")
            work = row["F"] - row["F0"]
            free = 80_000 - row["F"] - reply
            rendered = joint["rendered_prompt_sha256"]
            if (joint["errors"] != [] or work < work_floor or free < 0
                    or work != joint["work_F_minus_F0"] or free != joint["reply_floor_remaining_after_F"]
                    or joint["work_occurrences_F_B_F0"] != [1, 1, 0]
                    or joint["clock_row_sha256"] != pair["clock_row_sha256"]
                    or joint["F0_kind"] != "diagnostic_ablation_only"
                    or not all(digest(joint[k]) for k in ("F_sha256", "B_sha256", "F0_sha256", "clock_row_sha256"))
                    or len({joint[k] for k in ("F_sha256", "F0_sha256", "B_sha256")}) != 3
                    or set(rendered) != {"F", "F0", "B"} or not all(digest(d) for d in rendered.values())
                    or len(set(rendered.values())) != 3
                    or row["W_is_shorter_than_F_in_tokens"] is not (pair["W"] < row["F"])):
                raise SystemExit(f"{name}/{model}: diagnostic whole-index witness did not qualify")
            same_bytes = pair["W_sha256"] == joint["F_sha256"]
            if (row["forms_W"]["knowledge_index_form"] != "whole" or row["forms_F"]["knowledge_index_form"] != "whole"
                    or row["forms_W"]["world_form"] not in {"whole", "reader"}
                    or row["forms_W"]["extension_catalogue_form"] not in {"whole", "reader"}
                    or row["W_matches_F_bytes"] is not same_bytes
                    or (same_bytes and (pair["W0_sha256"] != joint["F0_sha256"]
                                        or pair["W"] != row["F"] or pair["W0"] != row["F0"]))):
                raise SystemExit(f"{name}/{model}: selected W and measured F are mislabeled")
            clock_rows.add(pair["clock_row_sha256"])
    if len(clock_rows) != 1:
        raise SystemExit("Reference profiles did not use one clock row")


def node_tap_proof(module, required):
    """Keep only expected public test names and TAP summary, not raw output."""
    try:
        run = subprocess.run([
            "node", "--test", "--test-reporter=tap", f"web/tests/{module}.test.js",
        ], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"accepted": False, "state": type(error).__name__}
    counts = {}
    for key in ("tests", "pass", "fail", "cancelled", "skipped", "todo"):
        found = re.findall(rf"^# {key} ([0-9]+)$", run.stdout, re.MULTILINE)
        counts[key] = int(found[0]) if len(found) == 1 else None
    results = {name: re.findall(rf"^ok [0-9]+ - {re.escape(name)}$", run.stdout, re.MULTILINE)
               for name in required}
    passed = [name for name, lines in results.items() if len(lines) == 1]
    accepted = (run.returncode == 0 and len(passed) == len(required)
                and counts["tests"] == counts["pass"] and counts["pass"] not in (None, 0)
                and all(counts[key] == 0 for key in ("fail", "cancelled", "skipped", "todo")))
    return {"accepted": accepted, "node_exit": run.returncode, "module_counts": counts,
            "passed_required_cases": passed}


def main(argv):
    lane, runner_dir, base, base_tree, product, product_tree, overlay, run_id, attempt = argv
    runner = Path(runner_dir).resolve(strict=True)
    owned = Path(os.environ["OUROBOROS_TEST_TEMP_ROOT"]).resolve(strict=True)
    if any(not Path(os.environ[key]).resolve().is_relative_to(owned) for key in ("HOME", "OUROBOROS_DATA_DIR", "TMPDIR")):
        raise SystemExit("No isolated test environment")
    if (any(not re.fullmatch(r"[0-9a-f]{40}", x) for x in (base, base_tree, product, product_tree, overlay))
            or not re.fullmatch(r"[0-9]+", run_id) or attempt != "1" or not sys.platform.startswith("linux")):
        raise SystemExit("Unbound Ubuntu subject")
    git_env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_NO_LAZY_FETCH="1",
                   GIT_ALLOW_PROTOCOL="file", GIT_NO_REPLACE_OBJECTS="1")

    def git(*args):
        return subprocess.check_output(["git", *args], text=True, env=git_env).strip()

    def product_is_clean():
        return (git("rev-parse", "HEAD") == product and git("rev-parse", "HEAD^{tree}") == product_tree
                and not git("status", "--porcelain", "--untracked-files=all"))

    if (not product_is_clean() or git("show", "-s", "--format=%P", overlay) != product
            or git("show", "-s", "--format=%P", product) != base or git("rev-parse", f"{base}^{{tree}}") != base_tree):
        raise SystemExit("The actual checkout is not the clean, detached product")
    if lane in ("s47", "s48", "s49", "s50"):
        phases = ((lane, (COLD[int(lane[1:]) - 47],), None),)
    elif lane == "supplement":
        phases = (("size", SIZE, None), ("units", UNIT, None))
    elif lane == "browser":
        composer = nodes("test_cold_return_window_park_browser",
                         "test_web_main_composer_keeps_one_saved_origin_and_first_prepared_request")
        phases = (("web-composer", composer, None), ("settings", SETTINGS, SETTINGS_FRAMES),
                  ("resume-card", RESUME, RESUME_FRAMES))
    elif lane == "contracts":
        phases = (("contracts", CONTRACTS, None),)
    else:
        raise SystemExit("Unrecognized qualification lane")
    scratch, upload = owned / "nano-r5", runner / "nano-proof"
    scratch.mkdir()
    upload.mkdir(exist_ok=True)
    identity = {"sha": overlay, "run_id": run_id, "run_attempt": attempt}
    report = {"schema": 3, "kind": "synthetic_only", "lane": lane, "public_base": base, "base_tree": base_tree,
              "product": product, "product_tree": product_tree, "workflow_source": identity, "phases": {},
              "full_battery": "not_run", "native_formatter_and_generation": "not_run", "physical_qwen_0_8b": "not_run",
              "pinned_hf_reference": "see separate reference lane (tokenizer/template only)",
              "context_gate": "not part of this overlay; see standard Context capacity gate workflow",
              "phase_success_requires": "exact selected_nodeids and one setup/call/teardown passed per ordinary node; no xfails, skips or missing reports",
              "optional_native_tests": "not_selected; no skip counts as pass"}
    for name, wanted, frames in phases:
        report["phases"][name] = test_phase(name, wanted, scratch, upload, identity, base, runner, frames)
        report["accepted"] = False # No positive receipt until every lane has settled and checkout still matches.
        (upload / "proof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if lane == "browser":
        report["phases"]["task-control-node"] = node_tap_proof("task_control_menu", TASK_CONTROL_CASES)
        report["phases"]["model-roles-node"] = node_tap_proof("model_roles", MODEL_ROLES_CASES)
    report["checkout_unchanged"] = product_is_clean()
    report["accepted"] = report["checkout_unchanged"] and all(row["accepted"] for row in report["phases"].values())
    (upload / "proof.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "stage-reference":
        stage_reference(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "check-reference":
        check_reference(sys.argv[2])
    else:
        sys.exit(main(sys.argv[1:]))
