/** Browser-only views of history facts; the wire schema belongs to the gateway. */

/** Additive task-ending fields of a history row.
 * @typedef {Object} TaskOutcomeHistoryFields
 * @property {'working'|'done'|'warn'|'error'|'cancelled'=} outcome_phase
 * @property {boolean=} outcome_final
 * @property {number=} task_attempt
 * @property {(string|Object)=} task_outcome
 * @property {Object=} current_outcome  // its own attempt and terminal_time, never the publication's
 * @property {Object=} historical_terminal — Python contract: `ouroboros/gateway/history_contracts.py::HistoricalTerminal`.
 * @property {Object=} model_execution
 * @property {Object=} terminal_delivery
 * @property {{v: 1, occurred_at: ?string, source: 'executor_terminal'|'supervisor_cancel'|'unknown', attempt: Object}=} terminal_time
 */

/**
 * Additive /api/chat/history row fields (v6.73.0 Project origin projection).
 * A Project thread may synthesize its start message from the binding's own
 * durable copy when the canonical row left the bounded read window:
 * `origin_projected: true` marks such a synthesized user row (normal history
 * shape otherwise), and a `system_type: "origin_omission"` system row discloses
 * origins omitted past the synthesis cap. Both fields are additive and safely
 * ignorable by renderers.
 * @typedef {Object} ProjectOriginHistoryFields
 * @property {boolean=} origin_projected
 * @property {"origin_omission"=} system_type
 */

/** Additive history fact: a project-owned Main mirror cannot grant cancel authority.
 * @typedef {Object} ProjectMirrorHistoryFields
 * @property {boolean=} project_mirror
 */

/**
 * Additive /api/chat/history row fields on `system_type: "skill_review"` rows:
 * the exact-job reference the producer already writes into chat.jsonl. A row
 * carrying a non-empty `job_id` lets the Chat card lazily fetch the rendered
 * review via GET /api/skills/{skill}/review-history/{record_id or job_id} (the
 * review record when the row names one, else the job); rows without either
 * (legacy full-text rows) keep local expansion. All fields are additive and
 * safely ignorable by renderers.
 * @typedef {Object} SkillReviewHistoryRowFields
 * @property {string=} skill
 * @property {string=} status
 * @property {string=} content_hash
 * @property {string=} job_id
 * @property {string=} record_id
 * @property {number=} review_round
 * @property {number=} snapshot_attempt
 */
