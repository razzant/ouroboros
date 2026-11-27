/** Browser-side projection of the latest tool verification. */

/**
 * A check's recorded criterion verdict (`verify_and_record`, #1585): on the live
 * `tool_call_finished` frame as `verification`, inside `tool_result_meta` on the settled
 * `tool_call` row, and on a replayed settled tool-evidence observation. The call's own
 * `status`/`is_error` stay its execution outcome; exit, signal and timeout stay process facts.
 * @typedef {Object} ToolVerificationFact
 * @property {"pass"|"fail"} status  // the durable receipt's verdict
 * @property {string} contract_kind  // visible_verifier | explicit_command | explicit_metric | artifact_observation
 * @property {string} expected_match  // substring | exact | exact_line | json_equals | bytes_equal
 * @property {boolean=} matched  // present only where a comparison was evaluated (never on timeout)
 * @property {boolean=} paths_named  // a failed artifact observation had at least one named path
 * @property {string} check  // the receipt's check rendering, bounded for transport
 * @property {string=} check_rendering
 * @property {string=} criterion_id
 * @property {string} criterion_source  // task_stated | agent_defined
 * @property {string} receipt_ts  // the receipt's own timestamp; no separate receipt id exists
 * @property {string=} detail  // bounded diagnostic from the producer, including missing paths
 */
