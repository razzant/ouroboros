"""Pins the ``model-visible ⟺ logged`` invariant at the model_send seam.

Design contract: ``docs/MODEL_SEND_OBSERVABILITY.md``.
Forward — every physical attempt seals its exact send copy before dispatch; the
explicit audit reconstructs that durable record and byte-compares it, and a
mismatch is a typed durable fact, never a dispatch gate. Exclusions form a
CLOSED enum; delegated lanes disclose ``unobserved`` instead of sealing.
Reverse — the bounded reconciliation sweep joins seals and attempts both ways.
"""

from __future__ import annotations

import copy
import gzip
import json
import os
import pathlib
import sys
from typing import Any, Dict

import pytest

from ouroboros import model_send_seal as seal_mod
from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient, _canonical_candidate_bytes
from tests._context_reports import (
    _dispatch, _payload, _rejected_then_answered, _Response, _rows, _scope, _settle_abandoned, _target,
)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.delenv("OUROBOROS_OBSERVABILITY_KEEP_RAW", raising=False)
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *args, **kwargs: 0.01)
    (root / "state").mkdir(parents=True)
    yield root


def _manifest(final: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(pathlib.Path(final["candidate_manifest_ref"]["path"]).read_text())


def test_local_request_reader_selects_one_sealed_attempt(data_root, capsys, monkeypatch):
    from ouroboros import context_attempt_reader as reader
    from ouroboros.context_attempt_reader import main, read_attempt

    first = _dispatch(data_root, 'reader-test')
    payload = _payload()
    payload['messages'][0]['content'][0]['text'] = 'Different request'
    last = _dispatch(data_root, 'reader-test', payload=payload, category='evolution')
    assert last['category'] == 'evolution'
    foreign = _dispatch(data_root, 'foreign-test')
    args = ['--data-root', str(data_root), 'reader-test']
    assert main(args) == 0
    page = json.loads(capsys.readouterr().out)
    assert [row['attempt_id'] for row in page['attempts']] == [last['attempt_id'], first['attempt_id']]
    assert main([*args, '--attempt', first['attempt_id']]) == 0
    selected = capsys.readouterr()
    assert 'sk-secret-model-send-seal' not in selected.out and 'summarize this' in selected.out
    assert 'Different request' not in selected.out
    assert json.loads(selected.err)['provider_body'] == 'not_observed'
    assert main([*args, '--attempt', first['attempt_id'], '--report']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['format'] == 'ouroboros.context-report/1'
    assert report['stored_json'] == selected.out
    assert report['facts']['attempt_id'] == first['attempt_id']
    assert report['facts']['transport_observation'] == 'not_confirmed'
    # The report dates and routes its attempt as the list does; the plain read's facts stay as they were.
    listed, life = page['attempts'][1], ('model', 'provider', 'ts_reserved', 'ts_dispatched', 'ts_final')
    assert all(listed[key] for key in life) and {key: report['facts'][key] for key in life} == {key: listed[key] for key in life}
    assert not {'ts_reserved', 'transport_observation', 'failure'} & set(json.loads(selected.err))
    for flag in ('--attempt', '--after'):
        assert main([*args, flag, foreign['attempt_id']]) == 2
        assert not capsys.readouterr().out
    with pytest.raises(ValueError, match='does not belong'):
        read_attempt(data_root, 'reader-test', foreign['attempt_id'])
    assert main([*args, '--after', last['attempt_id']]) == 0
    assert [row['attempt_id'] for row in json.loads(capsys.readouterr().out)['attempts']] == [first['attempt_id']]
    bound = [*args, '--attempt', last['attempt_id']]
    # --max-bytes only raises the bounds: one byte still reads the request under the default.
    assert main([*bound, '--max-bytes', '1']) == 0
    assert 'Different request' in capsys.readouterr().out
    with monkeypatch.context() as smaller:  # a request over the (here lowered) default bound
        smaller.setattr(reader, '_BODY_LIMIT', 1)
        assert main(bound) == 2
        bounded = capsys.readouterr()
        details = json.loads(bounded.err)
        assert not bounded.out and details['body'] == 'over_limit'
        # The gap names the record by digest, size, kind and its place under the data root; no folder of this computer.
        ref = _manifest(last)['redacted_projection_ref']
        assert details['redacted_ref'] == {**{key: value for key, value in ref.items() if key != 'path'},
                                           'path': f"observability/blobs/{ref['sha256']}.json.gz"}
        # A report of that gap is still written whole, and the terminal names the gap and the one --max-bytes that
        # admits the request and its record (the record is under its 2 MiB default, so the request's size).
        assert main([*bound, '--report']) == 0
        gap = capsys.readouterr()
    assert json.loads(gap.out)['facts']['redacted_ref'] == details['redacted_ref']
    assert str(data_root) not in gap.out + bounded.err and str(data_root.resolve()) not in gap.out + bounded.err
    assert json.loads(gap.out)['stored_json'] is None and json.loads(gap.out)['facts']['body'] == 'over_limit'
    assert details['retry_max_bytes'] == details['recorded_bytes']
    assert f"over_limit; --max-bytes {details['recorded_bytes']}" in gap.err
    assert main([*bound, '--max-bytes', str(details['recorded_bytes'])]) == 0
    assert 'Different request' in capsys.readouterr().out
    # An otherwise admitted ref to truncated gzip must not print partial request bytes.
    blob = pathlib.Path(_manifest(first)['redacted_projection_ref']['path'])
    from ouroboros.evidence_packs import SHARED_PLACEMENT_SUFFIX

    if not blob.exists():
        blob = blob.with_name(blob.name.removesuffix('.gz') + SHARED_PLACEMENT_SUFFIX)
    blob.write_bytes(blob.read_bytes()[:12])
    assert main([*args, '--attempt', first['attempt_id']]) == 2
    failed = capsys.readouterr()
    assert not failed.out and 'unavailable' in failed.err


def test_local_request_report_names_a_failure_by_type_never_by_its_words(data_root, monkeypatch):
    """A failed send's report carries what the ledger typed about it; a provider's words stay in private evidence."""
    from ouroboros.context_attempt_reader import read_attempt

    failed, answered = (read_attempt(data_root, "failed-reader-test", attempt, report_facts=True)[0]
                        for attempt in _rejected_then_answered(data_root, "failed-reader-test", monkeypatch))
    assert failed["failure"] == {"stage": "raised_exception", "exception_type": "_Rejected", "provider_status_code": 400}
    assert "private words" not in json.dumps(failed) and "evidence_ref" not in json.dumps(failed)
    assert "failure" not in answered and answered["attempt_id"] != failed["attempt_id"]


def test_local_request_report_of_an_abandoned_settlement_names_no_money(data_root, monkeypatch):
    """A rejected send whose task ended is settled with no receipt and an unknown price.

    Its report says how the ledger closed the record and keeps the request it sent; it carries no cost, price or
    usage field, so nothing downstream can read "settled" as a final charge."""
    from ouroboros.context_attempt_reader import read_attempt

    failed, _answered = _rejected_then_answered(data_root, "abandoned-reader-test", monkeypatch)
    row = _settle_abandoned(data_root, failed)
    assert (row["state"], row["settle_reason"], row["cost_usd"], row["cost_final"]) == ("settled", "abandoned", None, False)
    facts, body = read_attempt(data_root, "abandoned-reader-test", failed, report_facts=True)
    assert body and b"summarize this" in body
    assert (facts["ledger_state"], facts["settle_reason"], facts["transport_observation"]) == (
        "settled", "abandoned", "not_confirmed")
    assert facts["failure"]["exception_type"] == "_Rejected"
    assert not [key for key in facts if any(word in key for word in ("cost", "usd", "price", "token", "usage"))]


def test_local_request_reader_declines_export_without_store_handoff(data_root, capsys, monkeypatch):
    from ouroboros import store, usage_store
    from ouroboros.context_attempt_reader import main

    attempt = _dispatch(data_root, 'export-reader-test')
    with usage_store.hold(data_root) as txn:
        txn.set_meta('export', {'test': 'a switch is preparing this store'})

    def unexpected_handoff(*args, **kwargs):
        pytest.fail('request inspection must not take part in exporting the usage store')

    monkeypatch.setattr(store, 'prepare_handoff', unexpected_handoff)
    args = ['--data-root', str(data_root), 'export-reader-test']
    for tail in ([], ['--attempt', attempt['attempt_id']]):
        assert main([*args, *tail]) == 2
        failed = capsys.readouterr()
        assert not failed.out and 'UsageExportInProgress' in failed.err


def test_usage_inspection_hold_refuses_to_write_migrate_or_export_before_opening_the_store(data_root):
    from ouroboros import usage_store

    for unsafe in ({}, {'write': False}, {'migrate': False}, {'write': False, 'migrate': False, 'exporting': True}):
        with pytest.raises(ValueError, match='cannot write, migrate or export'):
            with usage_store.hold(data_root, inspect_only=True, **unsafe):
                pytest.fail('an inspection hold must refuse before it opens the store')
    assert not (data_root / usage_store.STORE_REL).exists()


def test_local_request_reader_logical_bound_is_not_shared_skeleton_bound(data_root, capsys, monkeypatch):
    """A view's skeleton names each message by a token longer than a short message, so it can be larger than the
    request it spells. The bound is the request's: the --max-bytes a gap names reads the request on that run."""
    import hashlib

    from ouroboros import context_attempt_reader as reader, observability
    from ouroboros.evidence_packs import shared_placement_path

    attempt = _dispatch(data_root, 'short-reader-test', payload={
        'model': 'gpt-x', 'messages': [{'role': 'user', 'content': 'go'}],
    })
    ref = _manifest(attempt)['redacted_projection_ref']
    placement = shared_placement_path(data_root, ref['sha256'])
    assert len(gzip.decompress(placement.read_bytes())) > ref['size']
    raw = observability.read_blob_ref(data_root, ref, max_bytes=ref['size'], raw_bytes=True)
    assert hashlib.sha256(raw).hexdigest() == ref['sha256'] and len(raw) == ref['size']
    with pytest.raises(ValueError, match='byte limit'):
        observability.read_blob_ref(data_root, ref, max_bytes=ref['size'] - 1, raw_bytes=True)
    args = ['--data-root', str(data_root), 'short-reader-test', '--attempt', attempt['attempt_id']]
    monkeypatch.setattr(reader, '_BODY_LIMIT', 1)  # the request is over this (here lowered) default
    assert reader.main([*args, '--report']) == 0
    named = json.loads(capsys.readouterr().out)['facts']['retry_max_bytes']
    assert named == ref['size']
    assert reader.main([*args, '--max-bytes', str(named)]) == 0
    assert 'go' in capsys.readouterr().out


@pytest.mark.parametrize('damage', ['missing', 'damaged'])
def test_local_request_reader_reads_a_packed_request_from_its_task_pack(data_root, capsys, damage):
    """Once its task is packed, an attempt's request is read from the task's own pack: a missing or damaged shared
    evidence index neither stops the export nor is rebuilt or marked by it (the packing pass owns that index)."""
    from ouroboros import evidence_packer, evidence_packs, store
    from ouroboros.context_attempt_reader import main
    from ouroboros.utils import utc_now_iso

    attempt = _dispatch(data_root, 'packed-reader-test')
    with store.open_store(data_root, write=True) as txn:  # a finished task, which the packing pass packs
        txn.store_task({'task_id': 'packed-reader-test', 'status': 'completed', '_schema_version': 1,
                        'updated_at': utc_now_iso()})
    evidence_packs._READERS.clear()
    try:
        evidence_packer.run_pass(data_root)
        ref = _manifest(attempt)['redacted_projection_ref']
        assert not pathlib.Path(ref['path']).exists()
        assert not evidence_packs.shared_placement_path(data_root, ref['sha256']).exists()
        packs = data_root / 'observability' / 'packs'
        index = packs / evidence_packs.INDEX_NAME
        index.unlink() if damage == 'missing' else index.write_bytes(b'not a database at all')
        before = index.read_bytes() if index.exists() else None
        args = ['--data-root', str(data_root), 'packed-reader-test', '--attempt', attempt['attempt_id']]
        assert main(args) == 0
        assert 'summarize this' in capsys.readouterr().out
        assert main([*args, '--report']) == 0
        assert json.loads(capsys.readouterr().out)['facts']['body'] == 'redacted_host_candidate'
        assert (index.read_bytes() if index.exists() else None) == before
        assert not (packs / evidence_packs.DAMAGED_MARK).exists()
    finally:
        evidence_packs._READERS.clear()


def test_local_request_reader_names_a_record_it_cannot_locate_now(data_root, capsys, monkeypatch):
    """A packed record whose place cannot be learned now (a busy index) is a named gap, never "not found"."""
    from ouroboros import evidence_packs
    from ouroboros.context_attempt_reader import main

    attempt = _dispatch(data_root, 'busy-reader-test')
    wanted, real = _manifest(attempt)['redacted_projection_ref']['sha256'], evidence_packs.blob_bytes

    def busy(root, ref, **kwargs):
        if ref.get('sha256') == wanted:
            raise evidence_packs.EvidenceUnavailable('evidence_unavailable: busy', reason='busy')
        return real(root, ref, **kwargs)

    monkeypatch.setattr(evidence_packs, 'blob_bytes', busy)
    args = ['--data-root', str(data_root), 'busy-reader-test', '--attempt', attempt['attempt_id']]
    assert main([*args, '--report']) == 0
    gap = capsys.readouterr()
    facts = json.loads(gap.out)['facts']
    assert (facts['body'], facts['evidence_reason']) == ('evidence_unavailable', 'busy')
    assert 'evidence_unavailable' in gap.err and 'run again later' in gap.err
    assert main(args) == 2
    plain = capsys.readouterr()
    assert not plain.out and json.loads(plain.err)['body'] == 'evidence_unavailable'


def test_local_request_reader_lists_and_reads_only_physical_attempts(data_root, capsys):
    """One ID is one request only for a row of the ledger's own calls (``usage_store.ATTEMPT_ROW``), an older row
    without a kind included. A delegated session or a folded history group may stand for many sends: the list leaves
    it out, a cursor naming it advances nothing, and its exact selection says why no request belongs to it."""
    from ouroboros.context_attempt_reader import list_attempts, main, read_attempt
    from tests._usage_store_testing import write_compacted_journal

    task = 'mixed-reader-test'
    older = {'task_id': task, 'root_task_id': task, 'provider': 'openai', 'model': 'm', 'category': 'task',
             'source': 'journal'}
    write_compacted_journal(data_root, [{**older, 'attempt_id': 'folded-group', 'folded_attempt_count': 3,
                                         'cost_usd': '1', 'cost_final': True}], [
        *({**older, 'attempt_id': 'kindless', 'state': state, 'reservation_upper_bound_usd': '0.5',
           'pricing_known': True} for state in ('reserved', 'dispatched')),
        {**older, 'kind': 'subscription_session', 'attempt_id': 'session-run', 'state': 'settled',
         'provider': 'claudexor', 'subscription_route': 'route', 'cost_usd': 0.5, 'cost_final': True}])
    sent = _dispatch(data_root, task, category='evolution')
    assert {row['attempt_id']: row.get('kind') for row in _rows(data_root) if row.get('task_id') == task} == {
        'folded-group': 'usage_baseline_group', 'kindless': None, 'session-run': 'subscription_session',
        sent['attempt_id']: 'attempt'}
    assert [row['attempt_id'] for row in list_attempts(data_root, task)['attempts']] == [sent['attempt_id'], 'kindless']
    args = ['--data-root', str(data_root), task]
    for cursor in ('session-run', 'folded-group'):
        assert main([*args, '--after', cursor]) == 2
        assert not capsys.readouterr().out
    assert list_attempts(data_root, task, after=sent['attempt_id'])['attempts'][0]['attempt_id'] == 'kindless'
    unlinked = read_attempt(data_root, task, 'kindless')[0]
    # No record is opened for a row without a link, so its trace is unread, not unrecorded.
    assert (unlinked['body'], unlinked['file_trace']) == ('not_linked', 'not_read')
    for one, kind in (('session-run', 'subscription_session'), ('folded-group', 'usage_baseline_group')):
        facts, body = read_attempt(data_root, task, one, report_facts=True)
        assert (facts['body'], facts['ledger_kind'], body) == ('not_one_attempt', kind, None)
        assert main([*args, '--attempt', one, '--report']) == 0
        gap = capsys.readouterr()
        assert json.loads(gap.out)['facts']['body'] == 'not_one_attempt' and '(not_one_attempt)' in gap.err


def test_local_request_reader_names_an_oversized_sealed_record_and_reads_it_on_request(data_root, capsys):
    """The redactions of 16,000 token-bearing URL parameters make an attempt's sealed record larger than 2 MiB; its
    request is larger still, yet under the 16 MiB default. That is a gap naming the record's size, and the --max-bytes
    it names reads the request at once: the allowance raises the record's bound and never lowers the request's. The
    record admitted that way is still checked against its digest."""
    from ouroboros.context_attempt_reader import main

    payload = _payload()
    payload['messages'][0]['content'][0]['text'] = 'see https://example.test/a?' + '&'.join(
        f'token=t{index:05d}' for index in range(16_000))
    payload['messages'].append({'role': 'user', 'content': 'tide table ' * 500_000})
    attempt = _dispatch(data_root, 'record-reader-test', payload=payload)
    record = pathlib.Path(attempt['candidate_manifest_ref']['path'])
    size = record.stat().st_size
    assert 16 * 1024 * 1024 > _manifest(attempt)['redacted_projection_ref']['size'] > size > 2 * 1024 * 1024
    args = ['--data-root', str(data_root), 'record-reader-test', '--attempt', attempt['attempt_id']]
    assert main([*args, '--report']) == 0
    gap = capsys.readouterr()
    facts = json.loads(gap.out)['facts']
    assert (facts['body'], facts['recorded_bytes'], facts['max_bytes'], facts['retry_max_bytes']) == (
        'manifest_over_limit', size, 2 * 1024 * 1024, size)
    assert facts['file_trace'] == 'not_read'  # the trace lies in the record this run did not open
    assert f'manifest_over_limit; --max-bytes {size} opens the record, and a request larger than it' in gap.err
    assert main([*args, '--max-bytes', str(size)]) == 0
    assert 'example.test' in capsys.readouterr().out
    raw = record.read_bytes()
    record.write_bytes(raw[:-1] + (b' ' if raw[-1:] != b' ' else b'\n'))  # same size, valid JSON, another digest
    assert main([*args, '--max-bytes', str(size)]) == 2
    assert not capsys.readouterr().out


def _windows_argv(line: str) -> list:
    """How the Windows C runtime splits a command line into the arguments python.exe receives."""
    args, word, quoted, slashes, started = [], [], False, 0, False
    for ch in line:
        if ch == '\\':
            slashes, started = slashes + 1, True
            continue
        if ch == '"':
            word.append('\\' * (slashes // 2))
            if slashes % 2:
                word.append('"')
            else:
                quoted = not quoted
            slashes, started = 0, True
            continue
        word.append('\\' * slashes)
        slashes = 0
        if ch in ' \t' and not quoted:
            if started:
                args.append(''.join(word))
            word, started = [], False
            continue
        word.append(ch)
        started = True
    word.append('\\' * slashes)
    if started:
        args.append(''.join(word))
    return args


def _cmd_passes_on(line: str) -> bool:
    """cmd.exe hands the line to the program as written: no & | < > ^ ( ) outside double quotes, nothing it expands."""
    quoted = False
    for ch in line:
        if ch == '"':
            quoted = not quoted
        elif ch in '%!' or (not quoted and ch in '&|<>^()'):
            return False
    return not quoted


def _printed_command(err: str) -> str:
    import re

    return re.search(r'without --report: (python -m ouroboros\.context_attempt_reader .+) > request\.json\)', err).group(1)


def test_local_request_report_too_large_for_the_page_is_a_bounded_gap_whose_command_reads_it(
        data_root, capsys, monkeypatch):
    """Beautiful Context opens no report over its limit, so a request that would make one is reported as that gap:
    a bounded record of the attempt, never its exclusion list, which grows with every masked secret (16,000 here make
    the full report several times the request); a failure's type and the file trace are kept. The gap names the
    request's size, the report's (exact when it was built, else a floor: a request over the page is never built into
    a report) and the one --max-bytes admitting the request and its sealed record; the command the terminal gives with
    it writes the request."""
    import hashlib
    import shlex

    from ouroboros import context_attempt_reader as reader
    from ouroboros.physical_attempt_context import PhysicalAttemptContext

    parse = _windows_argv if os.name == 'nt' else shlex.split
    payload = _payload()
    payload['messages'][0]['content'][0]['text'] = 'see https://example.test/a?' + '&'.join(
        f'token=t{index:05d}' for index in range(16_000))
    attempt = _dispatch(data_root, 'page-reader-test', payload=payload)
    smaller_payload = _payload()
    smaller_payload['messages'].append({'role': 'user', 'content': 'tide table ' * 300})  # larger than its gap report
    small = _dispatch(data_root, 'page-reader-test', payload=smaller_payload)
    # A refused send of a Main capture: its failure and file trace belong to its gap as well.
    prompt = 'You answer from the tide table.'
    digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    physical = PhysicalAttemptContext('owner_max', 'max', 'cold_estimate', '', '', None, None, False, False,
                                      base_prompt_source=('prompts/SYSTEM.md', digest, len(prompt), digest))
    with ua.bind_physical_attempt_context(physical):
        failed, _answered = _rejected_then_answered(data_root, 'page-reader-test', monkeypatch, payload={
            'model': 'gpt-x', 'messages': [{'role': 'system', 'content': prompt},
                                            {'role': 'user', 'content': 'tide table ' * 300}], 'max_tokens': 32})
    body, record = _manifest(attempt)['redacted_projection_ref']['size'], pathlib.Path(
        attempt['candidate_manifest_ref']['path']).stat().st_size
    assert record > 2 * 1024 * 1024 > body
    kept = {'task_id', 'attempt_id', 'ledger_state', 'provider_body', 'transport_observation', 'model', 'provider',
            'ts_reserved', 'ts_dispatched', 'ts_final', 'settle_reason', 'failure', 'file_trace', 'sha256',
            'redacted_ref', 'body', 'recorded_bytes', 'report_bytes', 'report_bytes_at_least', 'page_limit_bytes',
            'retry_max_bytes'}
    args = ['--data-root', str(data_root), 'page-reader-test', '--attempt']
    small_body = _manifest(small)['redacted_projection_ref']['size']
    failed_body = _manifest(next(row for row in _rows(data_root) if row['attempt_id'] == failed))[
        'redacted_projection_ref']['size']
    trace = [{'source_path': 'prompts/SYSTEM.md', 'source_sha256': digest, 'contribution': 'leading_system_prefix',
              'text_sha256': digest}]
    # Read whole, its report too large for the page; over this export's bound as well as over the page's; and read
    # whole but itself over the page, so its report is never built.
    for chosen, limit, smaller, allowance, size, report, said, life in (
            (attempt['attempt_id'], body, None, record, body, 'report_bytes', 'example.test', 'not_recorded'),
            (small['attempt_id'], small_body - 1, 1, small_body, small_body, 'report_bytes_at_least', 'tide',
             'not_recorded'),
            (failed, failed_body - 1, None, failed_body, failed_body, 'report_bytes_at_least', 'tide', trace)):
        built = []
        with monkeypatch.context() as patch:
            patch.setattr(reader, 'PAGE_LIMIT', limit)
            patch.setattr(reader, '_document', lambda facts, stored, real=reader._document: built.append(
                stored is not None) or real(facts, stored))
            if smaller:
                patch.setattr(reader, '_BODY_LIMIT', smaller)
            extra = ['--max-bytes', str(record)] if chosen == attempt['attempt_id'] else []
            assert reader.main([*args, chosen, *extra, '--report']) == 0
            gap = capsys.readouterr()
        document = json.loads(gap.out)
        facts = document['facts']
        assert len(gap.out.encode('utf-8')) <= limit and document['stored_json'] is None and set(facts) <= kept
        assert (facts['body'], facts['recorded_bytes'], facts['page_limit_bytes'], facts['retry_max_bytes']) == (
            'over_page_limit', size, limit, allowance)
        assert facts['file_trace'] == life and (chosen != failed or facts['failure']['exception_type'] == '_Rejected')
        if report == 'report_bytes':  # the report was built with the request, and measured
            assert built == [True, False] and facts['report_bytes'] > limit and 'report_bytes_at_least' not in facts
            assert f"its report would be {facts['report_bytes']} bytes" in gap.err
        else:
            assert True not in built and facts['report_bytes_at_least'] == size and 'report_bytes' not in facts
            assert f'its report would be at least {size} bytes' in gap.err
        assert reader.main(parse(_printed_command(gap.err))[3:]) == 0
        assert said in capsys.readouterr().out


def test_local_request_reader_gives_windows_a_command_cmd_exe_passes_on(tmp_path, capsys, monkeypatch):
    """On Windows the command a page-limit gap gives is written for cmd.exe, the shell the documentation names: & ^
    ( ) and spaces stay inside double quotes, the program reads back the very folder, task and attempt, and a value
    cmd.exe would still expand stays a placeholder."""
    from types import SimpleNamespace

    from ouroboros import context_attempt_reader as reader

    root = tmp_path / 'R&D'
    monkeypatch.setenv('OUROBOROS_DATA_DIR', str(root))
    monkeypatch.setenv('OUROBOROS_SETTINGS_PATH', str(root / 'settings.json'))
    monkeypatch.setenv('TOTAL_BUDGET', '100')
    monkeypatch.setattr(ua, 'estimate_cost_optional', lambda *args, **kwargs: 0.01)
    (root / 'state').mkdir(parents=True)
    payload = _payload()
    payload['messages'].append({'role': 'user', 'content': 'tide table ' * 500})
    attempt = _dispatch(root, 'windows-reader-test', payload=payload)
    body = _manifest(attempt)['redacted_projection_ref']['size']
    monkeypatch.setattr(reader, 'os', SimpleNamespace(name='nt'))
    monkeypatch.setattr(reader, 'PAGE_LIMIT', body)
    assert reader.main(['--data-root', str(root), 'windows-reader-test', '--attempt', attempt['attempt_id'],
                        '--report']) == 0
    line = _printed_command(capsys.readouterr().err)
    assert _cmd_passes_on(line) and _windows_argv(line)[3:] == [
        '--data-root', str(root), 'windows-reader-test', '--attempt', attempt['attempt_id'], '--max-bytes', str(body)]
    words = ['python', '-m', 'ouroboros.context_attempt_reader', '--data-root']
    for folder in ('D:\\a\\temp\\data', 'D:\\R&D\\data', 'C:\\Program Files (x86)\\Ouroboros\\data', 'D:\\data\\'):
        line = reader._retry_command(SimpleNamespace(data_root=folder, task_id='t-1', attempt='a1'), 7)
        line = line.removesuffix(' > request.json')
        assert _cmd_passes_on(line) and _windows_argv(line) == [*words, folder, 't-1', '--attempt', 'a1', '--max-bytes', '7']
    line = reader._retry_command(SimpleNamespace(data_root='D:\\%TEMP%\\data', task_id='t 1', attempt='a1'), 7)
    assert '--data-root DATA "t 1" --attempt a1 ' in line and _cmd_passes_on(line.removesuffix(' > request.json'))


def test_local_request_reader_names_a_receipt_recorded_without_a_request(data_root, capsys):
    """A live conversation's responses and connected time enter the ledger from usage reported once the work is under
    way: each row one attempt, with no saved request. Its report names that instead of a lost link."""
    from ouroboros.context_attempt_reader import list_attempts, main

    receipt = ua.record_received_attempt(ua.AttemptRequest(
        model='openai/gpt-realtime', provider='openai', task_id='live-reader-test', drive_root=data_root,
        source='live_connection_usage'), {'prompt_tokens': 3}, cost_usd=0.001)
    assert [row['attempt_id'] for row in list_attempts(data_root, 'live-reader-test')['attempts']] == [
        receipt.attempt_id]
    assert main(['--data-root', str(data_root), 'live-reader-test', '--attempt', receipt.attempt_id, '--report']) == 0
    gap = capsys.readouterr()
    facts = json.loads(gap.out)['facts']
    assert (facts['body'], facts['file_trace']) == ('receipt_only', 'not_recorded') and '(receipt_only)' in gap.err


def test_local_request_reader_names_a_data_root_without_a_usage_store(tmp_path, capsys):
    from ouroboros.context_attempt_reader import main

    missing = tmp_path / 'not-a-data-folder'
    for tail in ([], ['--attempt', 'any', '--report']):
        assert main(['--data-root', str(missing), 'task', *tail]) == 2
        failed = capsys.readouterr()
        assert not failed.out and 'No usage store under --data-root' in failed.err
        assert str(missing.resolve() / 'state' / 'usage.sqlite') in failed.err
    assert not missing.exists()


@pytest.mark.parametrize('claudexor_shape', [False, True])
def test_local_request_reader_reports_selected_file_not_current_file(data_root, tmp_path, claudexor_shape):
    import hashlib
    from dataclasses import asdict, replace
    from ouroboros.capability_evidence import CapabilityEvidence
    from ouroboros.context import build_context_fit_plan
    from ouroboros.context_attempt_reader import read_attempt
    from ouroboros.physical_attempt_context import PhysicalAttemptContext
    from tests._memory_view_context import world

    env, mem, _rooms = world(tmp_path / 'capture')
    prompt = env.repo_path('prompts/SYSTEM.md')
    before = b'System revision seventeen.\r\n'
    prompt.write_bytes(before)

    def route(_task, *, allow_fetch):
        return {'provider': 'openai', 'model': 'openai/gpt-x'}, CapabilityEvidence(
            872000, 'confirmed', 'metadata', 'test-model')

    plan = build_context_fit_plan(env, mem, {'id': 'reader-test', 'type': 'task', 'chat_id': 1, 'text': 'hello'},
                                  preferred_mode='max', route_resolver=route)
    payload = _payload()
    payload['messages'] = plan.messages_for(plan.initial_mode)
    if claudexor_shape:
        from ouroboros.llm_claudexor import _request

        # Exercise the deployed route's system-prefix transform, without a provider call.
        ready = _request({'source': 'codex', 'resolved_model': 'gpt-x'}, payload['messages'],
                         payload['tools'], {'prospective': True})
        payload['messages'] = ready['messages']
    prompt.write_bytes(b'System revision eighteen.\r\n')
    witness = plan.core.base_prompt_source
    physical = PhysicalAttemptContext('owner_max', 'max', 'cold_estimate', '', '', None, None, False, False,
                                      base_prompt_source=witness)
    # The witness rides beside the record: a copy keeps it, and no serialized form of the record holds it. A record
    # without one holds just its fields, as the ledger writes them, and a copy of it still reads None.
    assert replace(physical, prefix={'departs': 'first'}).base_prompt_source == witness
    assert 'base_prompt_source' not in asdict(physical)
    bare = PhysicalAttemptContext('owner_max', 'max', 'cold_estimate', '', '', None, None, False, False)
    assert vars(bare) == asdict(bare) and replace(bare, prefix={'departs': 'first'}).base_prompt_source is None
    with ua.bind_physical_attempt_context(physical):
        first = _dispatch(data_root, 'reader-test', payload=payload)
        later = _dispatch(data_root, 'reader-test', payload=_payload())  # the same witness over another first message
    facts, body = read_attempt(data_root, 'reader-test', first['attempt_id'])
    assert body and b'System revision seventeen.' in body and b'System revision eighteen.' not in body
    assert facts['file_trace'][0]['source_sha256'] == witness[1] == hashlib.sha256(before).hexdigest()
    assert read_attempt(data_root, 'reader-test', later['attempt_id'])[0]['file_trace'] == 'not_recorded'

    def holders(digest: str) -> list:
        """Every file under the data root that holds this digest, gzip blobs opened."""
        found = []
        for path in sorted(item for item in data_root.rglob('*') if item.is_file()):
            data = path.read_bytes()
            if digest.encode() in (gzip.decompress(data) if data[:2] == b'\x1f\x8b' else data):
                found.append(path.resolve())
        return found

    # The proven trace of the first manifest is the witness's one durable copy: the send that could not prove
    # it stored nothing of it, and neither did an accounting row.
    proven = pathlib.Path(first['candidate_manifest_ref']['path']).resolve()
    for digest in (witness[1], witness[3]):
        assert holders(digest) == [proven] and json.dumps(_manifest(first)).count(digest) == 1
        assert digest not in json.dumps(_rows(data_root), default=str)


def _violation_files(root: pathlib.Path):
    calls = root / "observability" / "calls"
    return sorted(calls.glob("*/*.model_send_violation.json")) if calls.is_dir() else []


def _violation_events(root: pathlib.Path):
    path = root / "logs" / "events.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip() and json.loads(line).get("type") == seal_mod.VIOLATION_EVENT_TYPE
    ]


# ---------------------------------------------------------------------------
# Forward: seal + reconstruction on the ordinary call
# ---------------------------------------------------------------------------


def test_ordinary_call_seals_the_send_copy_and_reconstructs_clean(data_root):
    final = _dispatch(data_root, "task-seal-clean")
    ref = final["candidate_manifest_ref"]
    assert ref["model_send_seal_version"] == seal_mod.SEAL_VERSION

    seal = _manifest(final)["model_send_seal"]
    assert seal["seal_version"] == seal_mod.SEAL_VERSION
    assert seal["canonical_basis"] == seal_mod.CANONICAL_BASIS
    assert seal["pre_redaction_sha256"] == final["candidate_raw_sha256"]
    assert seal["size_bytes"] == final["candidate_raw_size_bytes"]
    assert seal["attempt_id"] == final["attempt_id"]

    classes = [row["class"] for row in seal["exclusions"]]
    # Every disclosed instance uses the closed vocabulary.
    assert set(classes) <= seal_mod.EXCLUSION_CLASSES
    # The secret in the payload fired a per-instance redaction row with a path.
    redaction_rows = [r for r in seal["exclusions"] if r["class"] == "secret_redaction"]
    assert redaction_rows and all(r.get("path") for r in redaction_rows)
    # Class-level structural rows are always disclosed (note §4): the transport
    # adds envelope below the seam and provider-side transforms are unobservable.
    assert "transport_envelope" in classes
    assert "provider_side_transform" in classes

    # The invariant held: no typed fact anywhere, dispatch settled.
    assert final["state"] == "settled"
    assert not _violation_files(data_root)
    assert not _violation_events(data_root)


def test_a_known_candidate_is_still_sealed_before_the_send_with_the_whole_walks_bytes(data_root):
    """A later attempt of the task finds each part's redaction fact (``redaction.part_fact``): its sealed record still
    lands before the send, holds the bytes the whole walk gives and discloses the same secret rows."""
    first = _manifest(_dispatch(data_root, "task-seal-known"))
    sent = []

    def send(**candidate):
        calls = data_root / "observability" / "calls" / "task-seal-known"
        sent.append((copy.deepcopy(candidate), {path.name for path in calls.glob("*.json")}))
        return _Response()

    client = LLMClient(api_key="unused")
    with ua.usage_scope(_scope(data_root, "task-seal-known")):
        client._create_chat_completion_with_retries(send, _payload(), _target())
    final = _rows(data_root)[-1]
    manifest = _manifest(final)
    (candidate, present), = sent
    assert pathlib.Path(final["candidate_manifest_ref"]["path"]).name in present, "sealed before the send"
    stored, intact = seal_mod._read_blob_bytes(manifest)
    assert intact and stored == seal_mod._reconstruct_blob_bytes(candidate)
    rows = [row for row in manifest["model_send_seal"]["exclusions"] if row["class"] == "secret_redaction"]
    assert rows and rows == [row for row in first["model_send_seal"]["exclusions"] if row["class"] == "secret_redaction"]
    assert not _violation_files(data_root)


def test_the_closed_exclusion_enum_is_exactly_the_four_note_classes():
    assert seal_mod.EXCLUSION_CLASSES == {
        "secret_redaction", "provider_native_custody",
        "transport_envelope", "provider_side_transform",
    }
    assert seal_mod.MODEL_SEND_SEAL_UNOBSERVED == "unobserved"


def test_provider_native_custody_exclusion_is_disclosed_per_item(data_root):
    from ouroboros.anthropic_native_custody import ANTHROPIC_NATIVE_RECEIPT_KEY

    payload = _payload()
    payload["messages"].insert(0, {
        "role": "assistant",
        "content": [{"type": "text", "text": "prior turn"}],
        ANTHROPIC_NATIVE_RECEIPT_KEY: {
            "content_json": "[]", "content_sha256": "a" * 64, "tool_use_ids": [],
        },
    })
    final = _dispatch(data_root, "task-seal-custody", payload=payload)
    seal = _manifest(final)["model_send_seal"]
    custody_rows = [
        row for row in seal["exclusions"] if row["class"] == "provider_native_custody"
    ]
    assert custody_rows, seal["exclusions"]
    assert any(row.get("opaque_sha256") == "a" * 64 for row in custody_rows)
    assert all(row.get("path", "").startswith("$.messages") for row in custody_rows)
    # Both sides of the compare apply the same custody projection: still clean.
    assert not _violation_files(data_root)


def test_delegated_lane_carries_the_disclosed_unobserved_limit(data_root):
    with ua.usage_scope(_scope(data_root, "task-delegated")):
        ua.record_subscription_session(
            "session-cpl5", route="some-route", model="model-x",
        )
    row = _rows(data_root)[-1]
    assert row["kind"] == "subscription_session"
    assert row["model_send_seal"] == seal_mod.MODEL_SEND_SEAL_UNOBSERVED


def test_retry_rungs_carry_their_own_seals(data_root, monkeypatch):
    """§5.3: the invariant is per PHYSICAL attempt — each ladder rung's product
    is a new candidate with its own seal and attempt id."""
    client = LLMClient(api_key="unused")
    sends = []

    class _Rejected(RuntimeError):
        status_code = 400
        body = {"error": {"code": "unsupported_parameter", "type": "invalid_request_error"}}

    def create(**candidate):
        sends.append(copy.deepcopy(candidate))
        if len(sends) == 1:
            raise _Rejected("temperature unsupported")
        return _Response()

    monkeypatch.setattr(client, "_openrouter_signature_retry_kwargs", lambda *args: None)
    payload = {**_payload(), "temperature": 0.7}
    with ua.usage_scope(_scope(data_root, "task-rungs")):
        client._create_chat_completion_with_retries(create, payload, _target())

    finals: Dict[str, Dict[str, Any]] = {}
    for row in _rows(data_root):
        finals[row["attempt_id"]] = row
    seals = [_manifest(row)["model_send_seal"] for row in finals.values()]
    assert len(seals) == 2
    assert seals[0]["attempt_id"] != seals[1]["attempt_id"]
    assert seals[0]["pre_redaction_sha256"] != seals[1]["pre_redaction_sha256"]
    for attempt_id, row in finals.items():
        seal = _manifest(row)["model_send_seal"]
        assert seal["attempt_id"] == attempt_id
        assert seal["pre_redaction_sha256"] == row["candidate_raw_sha256"]
    assert not _violation_files(data_root)


# ---------------------------------------------------------------------------
# Forward: deliberately damaged durable records → typed facts, never blocks
# ---------------------------------------------------------------------------


def _tampering_persist(monkeypatch, tamper):
    """Wrap the real persist so ``tamper(manifest_path)`` corrupts the durable
    record between the write and the seam's read-back verification."""
    from ouroboros import observability

    real = observability.persist_physical_candidate

    def wrapped(*args, **kwargs):
        persisted = real(*args, **kwargs)
        tamper(pathlib.Path(persisted["manifest_ref"]["path"]))
        return persisted

    monkeypatch.setattr(observability, "persist_physical_candidate", wrapped)


def test_corrupted_blob_yields_reconstruction_fact_and_does_not_block(data_root, monkeypatch):
    def corrupt_blob(manifest_path: pathlib.Path):
        from ouroboros.evidence_packs import shared_placement_path

        manifest = json.loads(manifest_path.read_text())
        # The view is stored once: its durable record is the skeleton placement, not the canonical loose name.
        blob = shared_placement_path(data_root, manifest["full_payload_ref"]["sha256"])
        assert blob.is_file() and not pathlib.Path(manifest["full_payload_ref"]["path"]).exists()
        with gzip.open(blob, "wb") as handle:
            handle.write(b'{"tampered": true}')

    _tampering_persist(monkeypatch, corrupt_blob)
    final = _dispatch(data_root, "task-corrupt-blob")

    # Observability invariant: the call itself was NOT blocked.
    assert final["state"] == "settled"
    assert not _violation_files(data_root)
    seal_mod.reconcile_model_send_seals(data_root)
    files = _violation_files(data_root)
    assert len(files) == 1
    fact = json.loads(files[0].read_text())
    assert fact["type"] == seal_mod.VIOLATION_EVENT_TYPE
    assert fact["kind"] == "reconstruction_divergence"
    # A view stored once is spliced and verified before any byte is handed back: a damaged
    # skeleton (or part) is a typed unreadable record, never wrong bytes presented as the view.
    assert fact["divergence_class"] == "record_unreadable"
    assert fact["attempt_id"] == final["attempt_id"]
    assert "first_divergent_offset" not in fact, "no bytes came back, so there is no offset to name"
    # Digests and offsets only — no payload/secret bytes in the fact.
    assert "sk-secret" not in json.dumps(fact)
    events = _violation_events(data_root)
    assert len(events) == 1 and events[0]["kind"] == "reconstruction_divergence"


def test_tampered_seal_digest_yields_content_divergence_fact(data_root, monkeypatch):
    def rewrite_seal(manifest_path: pathlib.Path):
        manifest = json.loads(manifest_path.read_text())
        manifest["model_send_seal"]["pre_redaction_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _tampering_persist(monkeypatch, rewrite_seal)
    final = _dispatch(data_root, "task-tampered-digest")
    assert final["state"] == "settled"
    assert not _violation_files(data_root)
    seal_mod.reconcile_model_send_seals(data_root)
    [file] = _violation_files(data_root)
    fact = json.loads(file.read_text())
    assert fact["kind"] == "content_divergence"
    assert fact["divergence_class"] == "sdk_mutation"
    assert fact["expected"]["sha256"] == "0" * 64
    assert fact["observed"]["sha256"] == final["candidate_raw_sha256"]


def test_missing_seal_block_yields_reconstruction_fact(data_root, monkeypatch):
    def drop_seal(manifest_path: pathlib.Path):
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("model_send_seal", None)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _tampering_persist(monkeypatch, drop_seal)
    final = _dispatch(data_root, "task-missing-seal")
    assert final["state"] == "settled"
    assert not _violation_files(data_root)
    seal_mod.reconcile_model_send_seals(data_root)
    [file] = _violation_files(data_root)
    fact = json.loads(file.read_text())
    assert (fact["kind"], fact["divergence_class"]) == (
        "unlogged_attempt", "missing_seal_record",
    )


def test_undisclosed_exclusion_class_is_a_violation_of_the_closed_enum(data_root, monkeypatch):
    """§4: anything outside the enum is IN the byte domain — an exclusion row
    claiming a class the enum does not know is undisclosed by definition."""
    def smuggle_class(manifest_path: pathlib.Path):
        manifest = json.loads(manifest_path.read_text())
        manifest["model_send_seal"]["exclusions"].append({"class": "quiet_new_transform"})
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _tampering_persist(monkeypatch, smuggle_class)
    final = _dispatch(data_root, "task-undisclosed")
    assert final["state"] == "settled"
    assert not _violation_files(data_root)
    seal_mod.reconcile_model_send_seals(data_root)
    [file] = _violation_files(data_root)
    fact = json.loads(file.read_text())
    assert (fact["kind"], fact["divergence_class"]) == (
        "reconstruction_divergence", "undisclosed_exclusion",
    )


def test_foreign_serializer_basis_is_never_reinterpreted(data_root, monkeypatch):
    """§5.4: ``canonical_json_v1`` is the only equality basis; a seal claiming a
    different basis is reported, not re-read under this one."""
    def rewrite_basis(manifest_path: pathlib.Path):
        manifest = json.loads(manifest_path.read_text())
        manifest["model_send_seal"]["canonical_basis"] = "canonical_json_v2"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _tampering_persist(monkeypatch, rewrite_basis)
    _dispatch(data_root, "task-basis")
    assert not _violation_files(data_root)
    seal_mod.reconcile_model_send_seals(data_root)
    [file] = _violation_files(data_root)
    fact = json.loads(file.read_text())
    assert (fact["kind"], fact["divergence_class"]) == (
        "reconstruction_divergence", "serializer_basis",
    )
    assert fact["expected"]["basis"] == "canonical_json_v2"
    assert fact["observed"]["basis"] == "canonical_json_v1"


# ---------------------------------------------------------------------------
# Cost shape: the compare must not re-serialize without need
# ---------------------------------------------------------------------------


def test_dispatch_persists_once_without_a_readback_projection(data_root, monkeypatch):
    """The note budgets ONE read-back and ONE projection per attempt: the raw
    candidate is never serialized again for the compare (the seam digests are
    reused), and the CAS-basis projection pipeline runs exactly once more than
    persist alone (persist: 1, verify: 1)."""
    from ouroboros import llm_attempt, observability

    canonical_calls = []
    real_canonical = llm_attempt._canonical_candidate_bytes
    monkeypatch.setattr(
        llm_attempt, "_canonical_candidate_bytes",
        lambda payload: canonical_calls.append(1) or real_canonical(payload),
    )
    redact_calls = []
    real_redact = observability.redact_projection
    monkeypatch.setattr(
        observability, "redact_projection",
        lambda value, **kwargs: redact_calls.append(1) or real_redact(value, **kwargs),
    )
    _dispatch(data_root, "task-cost-shape")
    # 2 for measurement (raw + context), 2 for the seam's identity re-check;
    # the seam builds no second AttemptRequest or token/clock/image observations.
    assert len(canonical_calls) == 4, len(canonical_calls)
    # persist_call redacts once for the CAS write; verification reconstructs
    # once. Nothing else may redact this candidate.
    assert len(redact_calls) == 1, len(redact_calls)
    assert not _violation_files(data_root)


# ---------------------------------------------------------------------------
# Reverse: the bounded reconciliation sweep
# ---------------------------------------------------------------------------


def test_sweep_reports_clean_join_after_ordinary_calls(data_root):
    _dispatch(data_root, "task-sweep-clean")
    report = seal_mod.reconcile_model_send_seals(data_root)
    assert report["seals"] == 1
    assert report["sealed_attempts"] == 1
    assert report["orphan_seals"] == report["unlogged_attempts"] == 0
    assert report["facts_written"] == 0


def test_sweep_flags_a_synthetic_orphan_seal(data_root):
    from ouroboros.observability import persist_physical_candidate

    raw = _canonical_candidate_bytes({"model": "gpt-x", "messages": []})
    import hashlib

    persist_physical_candidate(
        data_root,
        task_id="task-orphan",
        attempt_id="f" * 32,
        candidate={"model": "gpt-x", "messages": []},
        candidate_facts={
            "candidate_raw_sha256": hashlib.sha256(raw).hexdigest(),
            "candidate_raw_size_bytes": len(raw),
            "candidate_measurement_kind": "canonical_json_v1",
        },
    )
    report = seal_mod.reconcile_model_send_seals(data_root)
    assert report["orphan_seals"] == 1 and report["facts_written"] == 1
    [file] = _violation_files(data_root)
    fact = json.loads(file.read_text())
    assert fact["kind"] == "orphan_seal"
    assert fact["attempt_id"] == "f" * 32
    assert fact["divergence_class"] == "missing_attempt_row"
    # Facts, never repairs: the seal manifest is still there.
    assert list((data_root / "observability" / "calls" / "task-orphan").glob("*.json"))


def test_an_attempt_folded_before_the_store_is_recorded_history_not_an_orphan_seal(data_root):
    """The retired compaction folded terminal attempts out of the journal into
    archive segments; the store imported only their aggregate. Asking the
    store alone would make every such attempt a durable orphan_seal fact at
    every startup, for history that is perfectly well recorded: the audit
    consults the retained evidence (``docs/USAGE_STORE.md``)."""
    from tests.fixtures_usage_store import fold_into_archive

    attempt_ids = [_dispatch(data_root, "task-folded-%d" % index)["attempt_id"] for index in range(2)]
    assert seal_mod.reconcile_model_send_seals(data_root)["orphan_seals"] == 0

    fold_into_archive(data_root, attempt_ids)
    live_ids = {str(row.get("attempt_id")) for row in _rows(data_root)}
    assert not live_ids & set(attempt_ids), "the fold must remove them from the store"

    report = seal_mod.reconcile_model_send_seals(data_root)
    assert report["seals"] == len(attempt_ids)
    assert report["orphan_seals"] == 0 and report["facts_written"] == 0
    assert _violation_files(data_root) == []
    assert _violation_events(data_root) == []


@pytest.mark.skipif(sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="chmod-based unreadability needs a non-root POSIX user")
def test_an_unreadable_archive_is_unknown_not_an_orphan_accusation(data_root):
    """Retained evidence that cannot be read is the sweep's UNKNOWN case: the
    history question has no answer, and no answer may become a durable
    accusation."""
    from tests.fixtures_usage_store import ARCHIVE_SEGMENT_REL, fold_into_archive

    attempt_id = _dispatch(data_root, "task-archive-unknown")["attempt_id"]
    fold_into_archive(data_root, [attempt_id])
    segment = data_root / ARCHIVE_SEGMENT_REL
    segment.chmod(0)
    try:
        report = seal_mod.reconcile_model_send_seals(data_root)
    finally:
        segment.chmod(0o644)
    assert report["status"] == "unknown"
    assert report["orphan_seals"] == 0 and report["facts_written"] == 0
    assert _violation_files(data_root) == []



def test_a_record_whose_bytes_cannot_be_located_now_is_unknown_never_a_violation(data_root, monkeypatch):
    """The audit reads each sealed record back; bytes that cannot be located now (``EvidenceUnavailable``:
    the packs catalog cannot answer) are neither unreadable nor missing. The pass reports ``unknown`` and
    writes no violation fact, the write-once latch that would keep a false verdict for good; the next pass,
    the bytes readable, verifies the record clean. The read names the record's task."""
    from ouroboros import evidence_packs

    final = _dispatch(data_root, "task-unknown-place")
    digest, real, tasks = _manifest(final)["full_payload_ref"]["sha256"], evidence_packs.blob_bytes, []

    def unlocatable(root, ref, **kwargs):
        tasks.append(kwargs.get("task_id"))
        if ref.get("sha256") == digest:
            raise evidence_packs.EvidenceUnavailable(
                "evidence_unavailable: the packs catalog cannot answer now", reason="busy")
        return real(root, ref, **kwargs)

    with monkeypatch.context() as busy:
        busy.setattr(evidence_packs, "blob_bytes", unlocatable)
        report = seal_mod.reconcile_model_send_seals(data_root)
    assert report["status"] == "unknown" and report["facts_written"] == 0, report
    assert _violation_files(data_root) == [] and _violation_events(data_root) == []
    assert "task-unknown-place" in tasks, tasks
    report = seal_mod.reconcile_model_send_seals(data_root)
    assert report["status"] == "completed" and report["facts_written"] == 0, report
    assert _violation_files(data_root) == []


def test_a_seal_whose_version_cannot_be_located_now_is_unknown_never_unlogged(data_root, monkeypatch):
    """The reverse direction reads each dispatched attempt's seal, falling back to the manifest's stored
    version; a version that cannot be located now leaves that attempt unknown, never an unlogged one. With
    the catalog answering, the vanished seal is the durable ``unlogged_attempt`` fact as before."""
    from ouroboros import evidence_packs

    final = _dispatch(data_root, "task-unknown-seal")
    ref = final["candidate_manifest_ref"]
    pathlib.Path(ref["path"]).unlink()
    real = evidence_packs.blob_bytes

    def unlocatable(root, blob, **kwargs):
        if blob.get("sha256") == ref["sha256"]:
            raise evidence_packs.EvidenceUnavailable(
                "evidence_unavailable: the packs catalog cannot answer now", reason="busy")
        return real(root, blob, **kwargs)

    with monkeypatch.context() as busy:
        busy.setattr(evidence_packs, "blob_bytes", unlocatable)
        report = seal_mod.reconcile_model_send_seals(data_root)
    assert report["status"] == "unknown" and report["unlogged_attempts"] == 0, report
    assert _violation_files(data_root) == []
    report = seal_mod.reconcile_model_send_seals(data_root)
    assert report["unlogged_attempts"] == 1 and report["facts_written"] == 1, report

def test_sweep_flags_a_dispatched_attempt_whose_seal_vanished(data_root):
    final = _dispatch(data_root, "task-sweep-unlogged")
    pathlib.Path(final["candidate_manifest_ref"]["path"]).unlink()
    report = seal_mod.reconcile_model_send_seals(data_root)
    assert report["unlogged_attempts"] == 1 and report["facts_written"] == 1
    [file] = _violation_files(data_root)
    fact = json.loads(file.read_text())
    assert fact["kind"] == "unlogged_attempt"
    assert fact["attempt_id"] == final["attempt_id"]
    assert fact["expected"]["sha256"] == final["candidate_raw_sha256"]


def test_sweep_is_idempotent_and_never_floods_the_events_plane(data_root):
    final = _dispatch(data_root, "task-sweep-dedup")
    pathlib.Path(final["candidate_manifest_ref"]["path"]).unlink()
    first = seal_mod.reconcile_model_send_seals(data_root)
    second = seal_mod.reconcile_model_send_seals(data_root)
    assert first["facts_written"] == 1
    assert second["facts_written"] == 0
    assert len(_violation_events(data_root)) == 1


def test_sweep_skips_promoted_child_manifests(data_root):
    """A manifest promoted from a child drive joins the CHILD ledger — reading
    it as an orphan here would be a false accusation."""
    import hashlib

    from ouroboros.observability import persist_physical_candidate

    raw = _canonical_candidate_bytes({"model": "gpt-x", "messages": []})
    persisted = persist_physical_candidate(
        data_root,
        task_id="task-promoted",
        attempt_id="e" * 32,
        candidate={"model": "gpt-x", "messages": []},
        candidate_facts={
            "candidate_raw_sha256": hashlib.sha256(raw).hexdigest(),
            "candidate_raw_size_bytes": len(raw),
            "candidate_measurement_kind": "canonical_json_v1",
        },
    )
    path = pathlib.Path(persisted["manifest_ref"]["path"])
    manifest = json.loads(path.read_text())
    manifest["promoted_call_manifest"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = seal_mod.reconcile_model_send_seals(data_root)
    assert report["seals"] == 0
    assert report["orphan_seals"] == 0 and report["facts_written"] == 0


def test_promotion_marks_the_canonical_copy(data_root, tmp_path):
    """The promotion path itself stamps the provenance marker the sweep trusts."""
    import hashlib

    from ouroboros.observability import persist_physical_candidate, promote_call_manifest_ref

    child = tmp_path / "child-drive"
    child.mkdir()
    raw = _canonical_candidate_bytes({"model": "gpt-x", "messages": []})
    persisted = persist_physical_candidate(
        child,
        task_id="task-child",
        attempt_id="d" * 32,
        candidate={"model": "gpt-x", "messages": []},
        candidate_facts={
            "candidate_raw_sha256": hashlib.sha256(raw).hexdigest(),
            "candidate_raw_size_bytes": len(raw),
            "candidate_measurement_kind": "canonical_json_v1",
        },
    )
    source_ref = {
        key: value for key, value in persisted["manifest_ref"].items()
        if key in {"path", "call_id", "sha256"}
    }
    promoted = promote_call_manifest_ref(
        child, data_root, source_ref, task_id="task-child",
    )
    manifest = json.loads(pathlib.Path(promoted["path"]).read_text())
    assert manifest["promoted_call_manifest"] is True
    report = seal_mod.reconcile_model_send_seals(data_root)
    assert report["orphan_seals"] == 0 and report["facts_written"] == 0


def test_sweep_skips_every_conclusion_on_an_unreadable_ledger(data_root):
    from ouroboros import usage_store

    _dispatch(data_root, "task-sweep-corrupt-ledger")
    store = data_root / usage_store.STORE_REL
    store.write_bytes(b"not a database\n" * 512)  # a damaged store: unknown, never an accusation
    usage_store.forget(data_root)
    report = seal_mod.reconcile_model_send_seals(data_root)
    assert report == {
        "status": "unknown", "manifests_checked": 0,
        "seals": 0, "sealed_attempts": 0,
        "orphan_seals": 0, "unlogged_attempts": 0,
        "facts_written": 0, "truncated": False,
    }


def test_sweep_rides_the_startup_family(data_root, monkeypatch):
    import ouroboros.server_maintenance as maintenance

    monkeypatch.setattr(maintenance, "DATA_DIR", data_root)
    calls = []
    monkeypatch.setattr(
        seal_mod, "reconcile_model_send_seals",
        lambda root, **kwargs: calls.append(pathlib.Path(root)) or {},
    )
    # Neighbour sweep steps degrade fail-soft on this synthetic root.
    maintenance._startup_custody_sweep()
    # Historical reconciliation belongs to the explicit owner/rebuild job;
    # no startup path launches it.
    assert calls == []
