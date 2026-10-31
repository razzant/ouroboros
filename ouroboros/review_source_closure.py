"""Retain a review request's named evidence before binding any paid reader.

The artifact/source promotion owners verify typed references and their owned
JSON closure. These are immutable snapshots, not a copy of an execution drive.
Opaque prose is never path-rewritten. The returned request carries both original
provenance and actual reader paths; missing named bytes refuse dispatch.

Snapshots live under ``source_handles/review_inputs/request-*/`` of the task
artifact store; a missing original reader root refuses with
``original_reader_root_unavailable``. Sources the retired task acceptance wrote
(applied panel records, tool trajectories, retrieval snapshots) keep their
discovery shapes in ``promote_source_payload``, so an old record promotes whole.
"""
from __future__ import annotations

import copy
import json
import pathlib
from typing import Any


# These are producer-owned edges, not patterns for recognizing arbitrary JSON.
# Unknown evidence, model prose and call arguments remain byte-preserving data.
_REF_FIELDS = frozenset({
    'source_ref', 'source_refs', 'result_source_ref', 'checkpoint_ref', 'spec_source_ref',
    'wave_artifact', 'applied_source_ref', 'tool_trajectory_source_ref', 'repo_diff_source_ref',
    'required_sources_ref', 'native_required_sources_ref', 'native_history_source',
    'exact_source_ref', 'producer_source_ref', 'request_ref', 'full_log_ref',
    'manifest_ref', 'full_payload_ref', 'redacted_projection_ref', 'trace_ref',
    'prompt_ref', 'response_ref', 'round_sources', 'attachment_manifest_ref', 'knowledge_previous_source',
})
_METADATA_FIELDS = frozenset({
    'trace_refs', 'llm_call_refs', 'tool_call_refs', 'entries', 'services', 'log_finalization',
    'refs', 'sources', 'review_evidence', 'review_projection', 'panels',
    'loop_outcome', 'completion_observations', 'verification_ledger', 'owner_wait',
    'root_phase_checkpoint', 'plan_review_state', 'waves', 'actors', 'usage',
    'read_receipts', 'view_receipt', 'view_changes', 'sent_view', 'capsule_refs',
    'restored_unit_refs', 'review_source_closure', 'native_required_sources',
    'required_sources', 'late_settlement', 'reviewer_outputs', 'emitted_answer', 'delivered',
    'tool_trajectory', 'tool_trajectory_selected', '__unresolved_partial_artifacts__',
    'owner_requirements_and_decisions', 'plan_claims_exhibit', 'acceptance_support_refs',
    'skill_lifecycle_history_coverage', 'evidence_manifest',
})


def source_carrier(value: dict, key: str, carrier: str) -> str:
    """Select host-owned edges at a trusted entry point; data cannot opt in."""
    if carrier == 'tool_result_metadata':
        return 'metadata' if key == 'knowledge_previous_source' else ''
    if carrier == 'tool_execution_metadata':
        return 'tool_result_metadata' if key == 'tool_result_meta' else ''
    if key == 'result_meta' and carrier == 'metadata':
        return 'tool_execution_metadata'
    if key == 'tool_result_meta' and carrier == 'metadata':
        return 'tool_result_metadata'
    if carrier == 'historical_inputs':
        return 'historical_anchor' if key == 'anchors' else ''
    if carrier == 'historical_anchor':
        return 'metadata' if key == 'source_ref' else ''
    if carrier == 'request':
        return {'evidence': 'evidence', 'policy': 'metadata'}.get(key, '')
    if carrier == 'contract':
        return {'predecessor_authority': 'task_result', 'attachment_manifest_ref': 'metadata'}.get(key, '')
    if carrier == 'task_result':
        return {'task_contract': 'contract', 'trace_refs': 'trace', 'plan_review_state': 'plan_state', 'plan_record': 'plan_record',
                'review_evidence': 'evidence', 'review_projection': 'metadata', 'loop_outcome': 'metadata',
                'completion_observations': 'metadata', 'owner_wait': 'metadata',
                'verification_ledger': 'metadata', 'root_phase_checkpoint': 'metadata'}.get(key, '')
    if carrier == 'trace':
        if key == 'timeout_repeat':
            return 'timeout_repeat'
        return 'response_ref' if key == 'response' else (
            'metadata' if key in {'request', 'llm_call_refs', 'tool_call_refs', 'log_finalization'} else '')
    if carrier == 'timeout_repeat':
        return 'metadata' if key in {'first', 'repeat'} else ''
    if carrier == 'call_response':
        return 'metadata' if key in {'usage', 'producer_outcome'} else ''
    if carrier == 'call_request' and key in {'kwargs', 'messages', 'send_messages'}:
        return 'call_request' if key == 'kwargs' else 'message'
    if carrier == 'plan_state':
        return {'waves': 'plan_wave', 'current_attempt': 'plan_attempt'}.get(key, '')
    # The planning organ's record: exact revision texts and each participant's retained
    # delivery/reply (``plan_record``), every one a task source the closure follows.
    if carrier == 'plan_record':
        return {'revisions': 'plan_revisions', 'asks': 'plan_asks'}.get(key, '')
    if carrier == 'plan_revisions':
        return 'metadata' if key == 'text_ref' else ''
    if carrier == 'plan_asks':
        return {'participants': 'plan_participants', 'dialogue_ref': 'plan_source_entry'}.get(key, '')
    if carrier == 'plan_participants':
        return {'delivery_ref': 'plan_source_entry', 'reply_ref': 'plan_source_entry',
                'prompt_ref': 'metadata', 'response_ref': 'metadata'}.get(key, '')
    if carrier == 'plan_source_entry':
        return 'metadata' if key == 'ref' else ''
    if carrier == 'plan_attempt':
        return 'metadata' if key in {'author_subject', 'submitted_subject'} else ''
    if carrier == 'plan_wave':
        return {'wave_artifact': 'metadata', 'previous_wave_artifact': 'metadata',
                'supersedes_wave_artifact': 'metadata', 'spec_source_ref': 'metadata',
                'dialogue_source_ref': 'metadata', 'actors': 'metadata', 'reviewer_outputs': 'metadata',
                'historical_supplements': 'metadata', 'evidence_manifest_full': 'plan_manifest'}.get(key, '')
    if carrier == 'plan_manifest':
        return 'metadata' if key == 'own_dialogue' else ''
    if carrier == 'plan_history':
        return 'metadata' if key in {'original_wave_artifact', 'result'} else ''
    if carrier == 'message':
        from ouroboros.context_compaction import _capsule_metadata
        return 'block' if key == 'content' and _capsule_metadata(value)[1] is not None else ''
    if carrier == 'block':
        return 'metadata' if key == '_context_capsule' else ''
    if carrier in {'checkpoint', 'metadata'} and key == 'messages':
        return 'message'
    if carrier == 'evidence':
        provenance = value.get('__provenance__') or {}
        if key == 'agent_supplied' or (isinstance(provenance, dict) and provenance.get(key) == 'agent_supplied'):
            return ''
        if key == 'historical_author_inputs':
            return 'historical_inputs'
        if key == 'service_finalization':
            # Host teardown rows: only their log_finalization refs are walked.
            return 'metadata' if isinstance(provenance, dict) and provenance.get(key) == 'host_attested' else ''
    if key == 'task_contract' and carrier in {'evidence', 'task_result'}:
        return 'contract'
    if key == 'request' and carrier == 'metadata':
        return 'request'
    if key == 'trace_refs':
        return 'trace'
    if key == 'plan_review_state':
        return 'plan_state'
    if key == 'plan_record':
        return 'plan_record'
    if key == 'response_ref':
        return 'response_ref'
    return 'metadata' if key in _REF_FIELDS | _METADATA_FIELDS else ''


def _owned_attachment_path(path: pathlib.Path, root: pathlib.Path, task_id: str) -> bool:
    """The task store's own ``attachments/``, or the review-input copy of one of them.

    ``retain_review_request_sources`` places a reviewer's attachment copy at
    ``source_handles/review_inputs/request-*/task_results/artifacts/<task>/attachments/<file>``
    under the same task root, so a persisted request names that placement.
    """
    resolved = path.resolve()
    if resolved.is_relative_to(root / 'attachments'):
        return True
    inputs = root / 'source_handles' / 'review_inputs'
    if not resolved.is_relative_to(inputs):
        return False
    rel = resolved.relative_to(inputs).parts
    return (len(rel) == 6 and rel[0].startswith('request-')
            and rel[1:5] == ('task_results', 'artifacts', task_id, 'attachments'))


def retain_contract(value: dict, source: pathlib.Path, custody: pathlib.Path, task_id: str, state: dict) -> dict:
    """Review inputs must already belong to this host-attested task's store.

    Legacy attachment inheritance elsewhere keeps its admission semantics; review
    closure never turns an inline absolute path into a new input authorization.
    """
    from ouroboros.artifacts import promote_task_attachment_refs, task_artifact_dir_path

    if 'attachment_manifest_ref' not in value:
        roots = [task_artifact_dir_path(root, task_id).resolve() for root in (source, custody)]
        for row in value.get('attachment_manifest') or []:
            if row.get('status') == 'rejected':
                continue
            path = pathlib.Path(row.get('abs_path') or '')
            if (not path.is_absolute() or path.is_symlink()
                    or not any(_owned_attachment_path(path, root, task_id) for root in roots)
                    or type(row.get('size')) is not int or len(str(row.get('sha256') or '')) != 64):
                from ouroboros.source_retention import InvalidRetainedSource

                raise InvalidRetainedSource('review attachment has no captured owner-bound source', path)
    wrapper = {'task_contract': copy.deepcopy(value)}
    promote_task_attachment_refs(custody, source, task_id, wrapper, state)
    return wrapper['task_contract']


def promote_source_payload(raw: bytes, *, source_id: str, extension: str, category: str,
                           parent_root: pathlib.Path, child_root: pathlib.Path,
                           task_id: str, state: dict) -> bytes:
    """Close the host-owned review/checkpoint JSON shapes; preserve opaque bytes."""
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.observability import _rewrite_child_ref_tree, _rewrite_service_payload

    plan_wave_source = source_id.startswith("plan-review-wave-")
    # ``acceptance_tool_trajectory``, ``review-retrieval-*`` and ``task_acceptance``
    # request records are sources the retired task acceptance wrote; their shapes
    # stay recognized so an old record's nested refs are still discovered.
    if extension == "json" and (category == "context_checkpoints" or source_id == "acceptance_tool_trajectory"):
        json_lines = source_id == 'review-retrieval-verification-receipts'
        try:
            payload = [json.loads(line) for line in raw.splitlines() if line.strip()] if json_lines else json.loads(raw)
        except (ValueError, UnicodeError):
            payload = None
        meta = payload.get("artifact_meta") if isinstance(payload, dict) else None
        plan_wave = plan_wave_source and isinstance(meta, dict) and meta.get("kind") == "plan_review_wave"
        plan_history = source_id.startswith('plan-review-late-') and isinstance(payload, dict) and payload.get('kind') == 'plan_review_historical_supplement' and payload.get('task_id') == task_id
        # Native history/round sources use the same typed refs, including view
        # receipts. Follow their host-owned JSON shape, never arbitrary prose.
        native_source = isinstance(payload, dict) and 'read_receipts' in payload and (
            'round_sources' in payload or ('round' in payload and 'messages' in payload)
            or ('required_sources_ref' in payload and 'read_provenance' in payload))
        checkpoint = isinstance(payload, dict) and all(key in payload for key in (
            'messages', 'selection_fingerprint', 'observed_view_revision', 'selected_unit_ids'))
        retained_review = source_id.startswith('review-retrieval-')
        if plan_wave or plan_history or native_source or checkpoint or retained_review or (isinstance(payload, list) and source_id == "acceptance_tool_trajectory") or (
            isinstance(payload, dict) and isinstance(payload.get("request"), dict)
            and payload["request"].get("surface") == "task_acceptance"
        ):
            role = 'plan_wave' if plan_wave else 'plan_history' if plan_history else 'task_result' if source_id == 'review-retrieval-task-result' else (
                'checkpoint' if checkpoint or native_source else 'metadata')
            if isinstance(payload, list) and source_id in {'acceptance_tool_trajectory', 'review-retrieval-tool-trajectory'}:
                rewritten = [_rewrite_service_payload(call, parent_root, child_root, task_id, state)
                             for call in payload]
            else:
                rewritten = _rewrite_child_ref_tree(payload, parent_root, child_root, task_id, state, carrier=role)
            if checkpoint and rewritten != payload:
                # Capsules bind raw messages/unit identities. Copying their
                # relative source closure may not rewrite the captured transcript.
                raise ValueError('context checkpoint closure requires transcript rebinding')
            if rewritten != payload:
                # Retain the captured digest as well as the re-addressed view.
                # Historical transcripts may still quote the original handle.
                store_actor_source_bytes(parent_root, task_id, category=category,
                    source_id=source_id, data=raw, extension=extension)
                raw = (("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
                        for row in rewritten) + "\n") if json_lines else
                       json.dumps(rewritten, ensure_ascii=False, sort_keys=True, default=str)).encode("utf-8")
    return raw


def source_owner(key: str, value: Any, task_id: str, *, carrier: str = '', root: Any = None) -> str:
    """Only an attested predecessor carrier changes a nested source's owner: a helper's
    ``predecessor_authority``, or a timeout row a new-ID recovery kept (#1886) whose owner
    the reaper's reciprocal retry links name; any other claim stays the walked task's."""
    if key == 'predecessor_authority' and isinstance(value, dict) and value.get('task_id'):
        from ouroboros.agent_startup_checks import valid_task_result_authority_source
        from ouroboros.artifacts import validate_task_id

        owner = validate_task_id(value['task_id'])
        if not valid_task_result_authority_source(value.get('source'), owner):
            raise ValueError('predecessor source has no task-bound authority')
        return owner
    if carrier == 'timeout_repeat' and isinstance(value, dict) and value.get('task_id') not in (None, '', task_id):
        from ouroboros.source_retention import retry_lineage_owns

        if root is not None and retry_lineage_owns(root, task_id, str(value['task_id'])):
            return str(value['task_id'])
    return task_id
