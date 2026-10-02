"""Retain a review request's named evidence before binding any paid reader.

The artifact/source promotion owners verify typed references and their owned
JSON closure. Retrieving acceptance additionally names a task record, receipt
union, task-filtered trajectory and artifact manifest. These are immutable
snapshots, not a copy of an execution drive. Opaque prose is never path-rewritten.
The returned request carries both original provenance and actual reader paths;
missing named bytes refuse dispatch, while an absent optional log is disclosed.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import pathlib
import tempfile
from typing import Any


# These are producer-owned edges, not patterns for recognizing arbitrary JSON.
# Unknown evidence, model prose and call arguments remain byte-preserving data.
_REF_FIELDS = frozenset({
    'source_ref', 'source_refs', 'result_source_ref', 'checkpoint_ref', 'spec_source_ref',
    'wave_artifact', 'applied_source_ref', 'tool_trajectory_source_ref', 'repo_diff_source_ref',
    'required_sources_ref', 'native_required_sources_ref', 'native_history_source',
    'exact_source_ref', 'producer_source_ref', 'request_ref', 'full_log_ref',
    'manifest_ref', 'full_payload_ref', 'redacted_projection_ref', 'trace_ref',
    'prompt_ref', 'response_ref', 'round_sources', 'attachment_manifest_ref',
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
    if carrier == 'request':
        return {'evidence': 'evidence', 'policy': 'metadata'}.get(key, '')
    if carrier == 'contract':
        return {'predecessor_authority': 'task_result', 'attachment_manifest_ref': 'metadata'}.get(key, '')
    if carrier == 'task_result':
        return {'task_contract': 'contract', 'trace_refs': 'trace', 'plan_review_state': 'plan_state',
                'review_evidence': 'evidence', 'review_projection': 'metadata', 'loop_outcome': 'metadata',
                'completion_observations': 'metadata', 'owner_wait': 'metadata',
                'verification_ledger': 'metadata', 'root_phase_checkpoint': 'metadata',
                'acceptance_debt': 'acceptance_debt'}.get(key, '')
    if carrier == 'acceptance_debt':
        return 'metadata' if key == 'source_ref' else ''
    if carrier == 'trace':
        return 'response_ref' if key == 'response' else (
            'metadata' if key in {'request', 'llm_call_refs', 'tool_call_refs', 'log_finalization'} else '')
    if carrier == 'call_response':
        return 'metadata' if key in {'usage', 'producer_outcome'} else ''
    if carrier == 'call_request' and key in {'kwargs', 'messages', 'send_messages'}:
        return 'call_request' if key == 'kwargs' else 'message'
    if carrier == 'plan_state':
        return {'waves': 'plan_wave', 'current_attempt': 'plan_attempt'}.get(key, '')
    if carrier == 'plan_attempt':
        return 'metadata' if key == 'author_subject' else ''
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
    if key == 'task_contract' and carrier in {'evidence', 'task_result'}:
        return 'contract'
    if key == 'request' and carrier == 'metadata':
        return 'request'
    if key == 'trace_refs':
        return 'trace'
    if key == 'plan_review_state':
        return 'plan_state'
    if key == 'response_ref':
        return 'response_ref'
    return 'metadata' if key in _REF_FIELDS | _METADATA_FIELDS else ''


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
                    or not any(path.resolve().is_relative_to(root / 'attachments') for root in roots)
                    or type(row.get('size')) is not int or len(str(row.get('sha256') or '')) != 64):
                raise ValueError('review attachment has no captured owner-bound source')
    wrapper = {'task_contract': copy.deepcopy(value)}
    promote_task_attachment_refs(custody, source, task_id, wrapper, state)
    return wrapper['task_contract']


def retain_review_refs(value: Any, source: pathlib.Path, custody: pathlib.Path, task_id: str,
                       *, carrier: str = 'metadata') -> Any:
    """Retain explicit host source carriers, never infer authority from data."""
    from ouroboros.observability import _rewrite_child_ref_tree, child_ref_promotion_scope
    from ouroboros.owner_mailbox import promote_owner_attachments

    facts = {'pending_refs': [], 'unavailable_refs': [], 'promoted_ref_count': 0,
             'promoted_source_handle_count': 0, 'status': 'complete'}
    with child_ref_promotion_scope():
        # Accepted owner follow-ups have their own task-bound manifest owner.
        promote_owner_attachments(custody, source, task_id, facts)
        retained = _rewrite_child_ref_tree(value, custody, source, task_id, facts, carrier=carrier)
    if facts['pending_refs'] or facts['unavailable_refs']:
        raise ValueError(json.dumps(facts, sort_keys=True))
    return retained


def _retain_named_sources(request: Any, source: pathlib.Path, custody: pathlib.Path, canonical: pathlib.Path) -> list[dict]:
    from ouroboros.artifacts import (collect_task_artifact_records, copy_artifact_file, read_actor_source_bytes, store_actor_source_bytes,
                                    stream_artifact_file, task_artifact_dir_path)
    from ouroboros.outcome_receipt_store import publish_verification_receipt_union, verification_receipts_path
    from ouroboros.task_results import load_task_result, task_result_path

    task_id, rows = request.task_id, []
    base = task_artifact_dir_path(source, task_id)
    target = task_artifact_dir_path(custody, task_id, create=True)

    def retain(name, original, raw):
        ref = store_actor_source_bytes(custody, task_id, category='context_checkpoints',
                                      source_id='review-retrieval-' + name, data=raw, extension='json')
        if read_actor_source_bytes(custody, task_id, ref) != raw:
            raise ValueError(f'{name}: source readback failed')
        ref = retain_review_refs(ref, source, custody, task_id)
        rows.append({'name': name, 'source_path': str(original), 'source_ref': ref,
                     'retained_path': str(target / ref['path']), 'status': 'retained'})

    result = load_task_result(source, task_id, strict=True)
    if not result:
        raise ValueError('task result source unavailable')
    # A declared artifact is copied through the verified file owner, preserving
    # its name and source path beside a digest-named immutable reader target.
    preview = request.evidence.get('artifacts') or []
    manifest = [row for row in preview if row.get('name') != '…']
    if any(row.get('source_error') for row in preview if row.get('name') == '…'):
        raise ValueError('artifact inventory source unavailable')
    for marker in (row for row in preview if row.get('name') == '…' and row.get('source_ref')):
        inventory = json.loads(read_actor_source_bytes(source, task_id, marker['source_ref']))
        if inventory.get('task_id') != task_id or not isinstance(inventory.get('artifacts'), list):
            raise ValueError('artifact inventory owner mismatch')
        manifest.extend(inventory['artifacts'])
    # The preview is never the inventory. Include unsampled readable files even
    # for historical previews without a full-set ref; captured declarations still
    # participate, so disappearance after capture cannot silently shrink the set.
    manifest.extend(result.get('artifacts') or [])
    manifest.extend(collect_task_artifact_records(source, task_id, measure=False, strict=True, require_registered=True))
    paths = {}
    for artifact in manifest:
        if not isinstance(artifact, dict):
            raise ValueError('invalid artifact manifest member')
        original = pathlib.Path(artifact.get('path') or base / str(artifact.get('relpath') or artifact.get('name') or ''))
        if not original.is_absolute():
            original = source / original
        if (original.is_symlink() or not original.resolve().is_relative_to(base.resolve())):
            raise ValueError(f'artifact escapes its task owner: {original}')
        if original == verification_receipts_path(source, task_id):
            continue  # receipts have their union owner below
        identity = stream_artifact_file(original, expected=artifact)
        if artifact.get('sha12') and artifact['sha12'] != identity['sha256'][:12]:
            raise ValueError(f'artifact preview digest mismatch: {original}')
        if str(original) in paths:
            continue
        artifact_rel = f"source_handles/context_checkpoints/review-artifact-{identity['sha256']}.bin"
        copy_artifact_file(original, target / artifact_rel, expected=identity)
        ref = {'kind': 'task_source', 'root': 'artifact_store', 'path': artifact_rel, **identity,
               'read': {'tool': 'read_file', 'arguments': {'root': 'artifact_store', 'path': artifact_rel}}}
        read_actor_source_bytes(custody, task_id, ref)
        paths[str(original)] = str(target / artifact_rel)
        if artifact.get('path'):
            paths[artifact['path']] = str(target / artifact_rel)
        rows.append({'name': 'artifact:' + str(artifact.get('name') or original.name),
                     'source_path': str(original), 'source_ref': ref, 'retained_path': str(target / artifact_rel),
                     'status': 'retained'})

    def bind_artifacts(value):
        # Only the host-owned manifest is a locator carrier. Arbitrary nested
        # evidence, trace arguments and captured prose remain exact data.
        value = copy.deepcopy(value)
        value['artifacts'] = [{**row, 'path': paths.get(str(pathlib.Path(row.get('path') or
            base / str(row.get('relpath') or row.get('name') or ''))), row.get('path'))}
            for row in value.get('artifacts') or []]
        for row in value['artifacts']:
            if row.get('path') and not pathlib.Path(row['path']).is_absolute():
                row['path'] = paths.get(str(source / row['path']), row['path'])
            if row.get('path') is None:
                row.pop('path', None)
        return value

    # A saved child row can already name canonical-published sources. Close over
    # both explicit producer roots before copying into the private reader root.
    result = retain_review_refs(bind_artifacts(result), source, canonical, task_id, carrier='task_result')
    result = retain_review_refs(result, canonical, custody, task_id, carrier='task_result')
    retain('task-result', task_result_path(source, task_id), json.dumps(result, ensure_ascii=False).encode())
    request.evidence = bind_artifacts(request.evidence)
    retain('artifact-inventory', base, json.dumps({'task_id': task_id, 'artifacts': [
        row for row in rows if row['name'].startswith('artifact:')]}, ensure_ascii=False).encode())
    for row in request.evidence.get('artifacts') or []:
        if row.get('name') == '…':
            row['source_ref'] = rows[-1]['source_ref']
    for issue in request.evidence.get('__unresolved_partial_artifacts__') or []:
        if issue.get('tool') == 'artifact_manifest':
            issue.update(source_ref=rows[-1]['source_ref'], status='not_materialized_for_reviewer',
                         reason='artifact_manifest_preview')

    receipt = verification_receipts_path(source, task_id)
    canonical_receipt = verification_receipts_path(custody, task_id)
    if receipt.exists() and not publish_verification_receipt_union(custody, task_id, source):
        raise ValueError('verification receipt union unavailable')
    if canonical_receipt.exists():
        retain('verification-receipts', receipt, canonical_receipt.read_bytes())
    else:
        rows.append({'name': 'verification-receipts', 'source_path': str(receipt), 'status': 'not_recorded'})
    trajectory = source / 'logs' / 'tools.jsonl'
    if trajectory.exists():
        with trajectory.open(encoding='utf-8') as stream:
            records = [row for line in stream if line.strip() if (row := json.loads(line)).get('task_id') == task_id]
        records = retain_review_refs(records, source, canonical, task_id)
        records = retain_review_refs(records, canonical, custody, task_id)
        retain('tool-trajectory', trajectory, json.dumps(records, ensure_ascii=False).encode())
    else:
        rows.append({'name': 'tool-trajectory', 'source_path': str(trajectory), 'status': 'not_recorded'})
    return rows


def _reader_bindings(value: Any, root: pathlib.Path, task_id: str) -> list[dict]:
    """Give each task-owned ref its actual address without rewriting opaque text."""
    from ouroboros.artifacts import task_artifact_dir_path

    rows = {}

    def visit(item, owner, carrier='request'):
        if isinstance(item, dict):
            if item.get('kind') == 'task_source':
                path = task_artifact_dir_path(root, owner) / item['path']
                rows[(owner, str(path))] = {'owner_task_id': owner, 'source_path': item['path'],
                    'sha256': item['sha256'], 'size': item['size'],
                    'retained_path': str(path), 'read': {'tool': 'read_file', 'arguments': {
                        'root': 'runtime_data', 'path': str(path.relative_to(root))}}}
            else:
                for key, child in item.items():
                    role = source_carrier(item, key, carrier)
                    if role:
                        visit(child, source_owner(key, child, owner), role)
        elif isinstance(item, list):
            for child in item:
                visit(child, owner, carrier)

    visit(value, task_id)
    return list(rows.values())


def _retain_historical_sources(request: Any, source: pathlib.Path, read_root: pathlib.Path, debt: dict) -> list[dict]:
    """Re-address only the frozen subject and its retained HOST carriers.

    No current result, artifact inventory, workspace or tool log substitutes
    for a historical source. Manifest-only entries remain the subject's gaps.
    """
    from ouroboros.acceptance_history import read_acceptance_history, historical_source_location_owned
    from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path
    from ouroboros.task_results import load_task_result

    current = load_task_result(source, request.task_id, strict=True) or {}
    if current.get('acceptance_debt') != debt:
        raise ValueError('historical debt is not the canonical pin')
    frozen = read_acceptance_history(source, request.task_id, debt)
    if request.subject != frozen['answer'] or request.task_attempt != frozen['task_attempt']:
        raise ValueError('historical request subject mismatch')
    rows = []
    for name, ref in [('historical-subject', debt['source_ref']), *[
            (item['location'], item['source_ref']) for item in frozen.get('sources', [])
            if historical_source_location_owned(str(item.get('location') or ''))]]:
        retained = retain_review_refs(ref, source, read_root, request.task_id)
        # Observability manifests/blobs keep their existing typed reader in the
        # re-addressed subject. Promotion already verifies their owned bytes.
        if retained.get('kind') != 'task_source':
            continue
        read_actor_source_bytes(read_root, request.task_id, retained)
        rows.append({'name': name, 'source_path': ref['path'], 'source_ref': retained,
                     'retained_path': str(task_artifact_dir_path(read_root, request.task_id) / retained['path']),
                     'status': 'retained'})
    return rows


def retain_review_request_sources(request: Any, *, source_root: Any, custody_root: Any,
                                  historical_debt: dict | None = None) -> None:
    """Bind typed request refs and named acceptance sources to durable custody.

    Call BEFORE review_operation_scope, serialization, prompt caching or dispatch.
    This is not rendered-packet custody: general typed source closure includes
    earlier loops and native continuation sources. Reuse verifies the original
    snapshots; it never refreshes a request underneath a live paid reader.
    A failure raises before the caller can stamp paid work.
    """
    from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path

    source, custody = pathlib.Path(source_root), pathlib.Path(custody_root)
    if not source.is_dir():
        raise ValueError('original_reader_root_unavailable: cannot retain named sources')
    retained = request.policy.get('review_source_closure')
    if retained:
        read_root = pathlib.Path(retained['read_root']).resolve()
        if (retained.get('task_id') != request.task_id
                or source.resolve() != read_root
                or not read_root.is_relative_to((task_artifact_dir_path(custody, request.task_id) / 'source_handles' / 'review_inputs').resolve())):
            raise ValueError('review source closure identity mismatch')
        for row in retained['sources']:
            if row['status'] == 'retained':
                read_actor_source_bytes(read_root, request.task_id, row['source_ref'])
        retain_review_refs(dataclasses.asdict(request), read_root, custody, request.task_id, carrier='request')
        return
    # Work on a copy: partial publication cannot leave the caller appearing bound.
    bound = dataclasses.replace(request, **{key: value for key, value in retain_review_refs(
        dataclasses.asdict(request), source, custody, request.task_id, carrier='request').items()})
    parent = task_artifact_dir_path(custody, request.task_id, create=True) / 'source_handles' / 'review_inputs'
    parent.mkdir(parents=True, exist_ok=True)
    read_root = pathlib.Path(tempfile.mkdtemp(prefix='request-', dir=parent))
    bound = dataclasses.replace(bound, **retain_review_refs(dataclasses.asdict(bound), custody, read_root, request.task_id, carrier='request'))
    named = (_retain_historical_sources(bound, source, read_root, historical_debt) if historical_debt is not None
             else _retain_named_sources(bound, source, read_root, custody) if request.surface == 'task_acceptance' else [])
    bindings = _reader_bindings(dataclasses.asdict(bound), read_root, request.task_id)
    bound.policy = {**bound.policy, 'native_data_root': str(read_root), 'review_source_closure': {
        'schema_version': 1, 'task_id': request.task_id, 'read_root': str(read_root),
        'sources': named, 'refmap': bindings}}
    retain_review_refs(dataclasses.asdict(bound), read_root, custody, request.task_id, carrier='request')
    for field in dataclasses.fields(request):
        setattr(request, field.name, getattr(bound, field.name))


def promote_source_payload(raw: bytes, *, source_id: str, extension: str, category: str,
                           parent_root: pathlib.Path, child_root: pathlib.Path,
                           task_id: str, state: dict) -> bytes:
    """Close the host-owned review/checkpoint JSON shapes; preserve opaque bytes."""
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.observability import _rewrite_child_ref_tree, _rewrite_service_payload

    plan_wave_source = source_id.startswith("plan-review-wave-")
    if extension == "json" and (category == "context_checkpoints" or source_id == "acceptance_tool_trajectory"):
        json_lines = source_id == 'review-retrieval-verification-receipts'
        try:
            payload = [json.loads(line) for line in raw.splitlines() if line.strip()] if json_lines else json.loads(raw)
        except (ValueError, UnicodeError):
            payload = None
        if source_id == 'focused_room' and isinstance(payload, dict) and payload.get('kind') == 'chronicle_room_source':
            # The host's room manifest owns these exact chunk edges. Chat rows
            # inside each chunk remain opaque data; quoted refs confer no custody.
            for ref in payload.get('source_chunks', []):
                _rewrite_child_ref_tree(ref, parent_root, child_root, task_id, state, carrier='response_ref')
            return raw  # Placement changes never rewrite the captured manifest.
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
        historical = source_id == 'acceptance_historical'
        if historical and (not isinstance(payload, dict) or payload.get('schema_version') != 1
                or payload.get('kind') != 'acceptance_historical_subject' or payload.get('task_id') != task_id):
            raise ValueError('historical acceptance source owner mismatch')
        if historical or plan_wave or plan_history or native_source or checkpoint or retained_review or (isinstance(payload, list) and source_id == "acceptance_tool_trajectory") or (
            isinstance(payload, dict) and isinstance(payload.get("request"), dict)
            and payload["request"].get("surface") == "task_acceptance"
        ):
            role = 'plan_wave' if plan_wave else 'plan_history' if plan_history else 'task_result' if source_id == 'review-retrieval-task-result' else (
                'checkpoint' if checkpoint or native_source else 'metadata')
            if historical:
                from ouroboros.acceptance_history import historical_source_location_owned
                # Only the history writer's locations carry authority. Answers,
                # owner corpus, contracts and old unowned refs stay literal data.
                rewritten = {**payload, 'sources': [{**item, 'source_ref':
                    _rewrite_child_ref_tree(item.get('source_ref'), parent_root, child_root, task_id, state)
                    if historical_source_location_owned(str(item.get('location') or '')) else
                    item.get('source_ref')}
                    for item in payload.get('sources', [])]}
            elif isinstance(payload, list) and source_id in {'acceptance_tool_trajectory', 'review-retrieval-tool-trajectory'}:
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


def source_owner(key: str, value: Any, task_id: str) -> str:
    """Only an attested predecessor carrier changes a nested source's owner."""
    if key == 'predecessor_authority' and isinstance(value, dict) and value.get('task_id'):
        from ouroboros.agent_startup_checks import valid_task_result_authority_source
        from ouroboros.artifacts import validate_task_id

        owner = validate_task_id(value['task_id'])
        if not valid_task_result_authority_source(value.get('source'), owner):
            raise ValueError('predecessor source has no task-bound authority')
        return owner
    return task_id
