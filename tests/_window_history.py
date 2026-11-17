"""One observer's unbounded chronological oracle and a faultable production store.

The oracle knows when a serving *claim* changed, including while publication was
unavailable. Receipts and time do not change that identity. Its history is never
compacted; only the production side passes through the JSON frontier writer.
"""
from uuid import uuid4
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from ouroboros import capability_evidence as ce
from ouroboros.context_window import WindowLimit, record_observation, resolve_window, statement_key, window_authority


ALPHABET = ('read80', 'read120', 'fail80', 'fail120', 'save80', 'accept100', 'accept200',
            'refuse_input60', 'refuse_total60', 'tick', 'expire', 'cache', 'refresh', 'instance')


class ObserverHistory:
    def __init__(self, monkeypatch, root):
        self.root = root
        self.store = {}
        self.failure = ''
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        monkeypatch.setattr(ce, '_read', lambda root: deepcopy(self.store))
        monkeypatch.setattr(ce, '_load', lambda root: deepcopy(self.store))
        monkeypatch.setattr(ce, '_save', self.save)
        monkeypatch.setattr(ce, '_store_transaction', self.transaction)
        monkeypatch.setattr(ce, 'utc_now', lambda: self.now)
        monkeypatch.setattr(ce, 'utc_now_iso', lambda: self.now.isoformat())

    @contextmanager
    def transaction(self, root):
        if self.failure == 'lock':
            raise TimeoutError('publication unavailable')
        yield

    def save(self, root, data):
        if self.failure == 'save':
            return False
        self.store = deepcopy(data)
        return True

    def reset(self):
        from ouroboros import context_window
        getattr(context_window, "_READ_ATTEMPTS", {}).clear()
        self.store = {}
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.claim = None
        self.receipt = None
        self.observations = []
        self.step = 0
        self.faulted = False
        self.instance = 0
        self.lost_order = False
        self.publication_origins = {}
        self.observe('read80')

    def observe(self, action):
        self.step += 1
        self.failure = 'lock' if action.startswith('fail') else 'save' if action.startswith('save') else ''
        self.faulted |= bool(self.failure)
        if action in ('tick', 'expire'):
            self.now += timedelta(seconds=ce._CONFIRMED_TTL_SEC * (0.6 if action == 'tick' else 1.1))
        elif action.startswith('accept') or action.startswith('refuse'):
            accepted = action.startswith('accept')
            tokens = int(action.removeprefix('accept')) * 1000 if accepted else 60000
            kind = 'input' if accepted or 'input' in action else 'total'
            self.observations.append(WindowLimit(tokens, kind, 'accepted' if accepted else 'refusal',
                                                 self.now.isoformat(), sequence=self.step))
            record_observation(self.root, 'r', **({'accepted': tokens} if accepted else {'refused': tokens, 'kind': kind, 'transmitted': 60000 + len(self.observations)}))
        elif action not in ('cache', 'restart'):
            if action == 'instance':
                self.instance += 1
            tokens = 80000 if action.endswith('80') else 120000 if action.endswith('120') else self.claim.tokens
            reading = WindowLimit(tokens, 'total', 'server_metadata', self.now.isoformat(),
                                  sequence=self.step, instance_id=str(self.instance))
            if self.claim and statement_key(reading) == statement_key(self.claim):
                reading = replace(reading, sequence=self.claim.sequence)
            self.claim = reading
            self.receipt = ce.CapabilityEvidence(tokens, 'confirmed', 'server_metadata', 'r',
                provider='openai-compatible', ts=self.now.isoformat(), window_instance_id=str(self.instance), window_read_id=uuid4().hex)
            self.live = deepcopy(self.receipt)
        if action == 'restart':
            from ouroboros import context_window
            context_window._READ_ATTEMPTS.clear()
        if action in ('cache', 'restart'):
            self.live = ce._cached_evidence(self.receipt.to_json(), 'r', '', 'openai-compatible')
        self.live.stale = not self.fresh(self.claim)
        result = resolve_window(self.live, 8192, root=self.root)
        for frontier in self.store.get('main_windows', {}).get('r', {}).get('reads', []):
            self.publication_origins.setdefault(frontier['sequence'], self.step)
        seq = self.live.window_sequence
        self.lost_order = self.fresh(self.claim) and seq > 0 and self.publication_origins[seq] != self.claim.sequence
        self.failure = ''
        physical = self.claim if self.fresh(self.claim) else None
        rows = [row for row in self.observations if self.fresh(row) or (row.source == 'accepted' and not physical)]
        ordered = sorted([*rows, *([physical] if physical else [])], key=lambda row: row.sequence)
        expected, _ = window_authority(ordered)
        capacities = [e.tokens - (8192 if e.kind == 'total' else 0) for e in expected.values()]
        oracle = min(capacities, default=float('inf'))
        self.observable_oracle = oracle
        if self.lost_order:
            observable = replace(physical, sequence=self.publication_origins[seq])
            limits, _ = window_authority(sorted([*rows, observable], key=lambda row: row.sequence))
            self.observable_oracle = min((e.tokens - (8192 if e.kind == 'total' else 0)
                                          for e in limits.values()), default=float('inf'))
        actual = result.usable if result.usable is not None else float('inf')
        return actual, oracle

    def fresh(self, row):
        return (self.now - datetime.fromisoformat(row.observed_at)).total_seconds() <= ce._CONFIRMED_TTL_SEC
