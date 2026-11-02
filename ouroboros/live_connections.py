"""Connection-owned fit, credentials and accounting while a call prepares its successor.

The call owns canonical history and effects. A connection fits a private snapshot;
only adoption publishes its projection. Native callbacks retain that connection's
physical/quote/window state through close, even after a successor starts serving.
"""
from __future__ import annotations

import asyncio
import copy
import time
from collections import deque
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import replace

from websockets.exceptions import ConnectionClosed

from ouroboros.contracts.live_conversation import ContextUpdate
from ouroboros.contracts.realtime import RealtimeSetupError, canonical_json
from ouroboros.realtime_session import joined

CURRENT = ContextVar('live_connection', default=None)
FIELDS = ('plan', '_physical_context', '_renewal_window', '_renewal_facts', '_held_input', '_growth',
          '_history_at_receipt', '_serving_history_at_receipt', '_selected_text_estimate', '_selected_raw_estimate', '_session_instructions',
          '_density_input', '_connection_quote', '_connection_accounting', '_meter_watermark', 'connection_number',
          '_view_session_factory', '_view_session_options', '_prepared_view', '_schemas_mode',
          '_history_tokens', '_history_cached_version', '_fitted_frame_rows')


def connection_for(host):
    if getattr(host, '_connection_owner', None) is not None:
        return None
    current = CURRENT.get()
    if current is not None and current._connection_owner is host:
        return current
    view = getattr(host, '_connection_view', None)
    return view if view is not None and view._connection_accounting else None


def publish(host, view):
    for name in FIELDS:
        if hasattr(view, name):
            setattr(host, name, getattr(view, name))
    host.ctx.context_fit_plan = view.plan
    host.ctx.active_context_mode = view.plan.initial_mode


class WaitingReader:
    """Keep the native reader/meter alive until adoption, without dispatching events.

    Adoption takes the buffered batches and the one in-flight read before
    returning to direct reads. Cancelling a waiter never abandons that read.
    """

    def __init__(self, lease):
        self.lease = lease
        self.batches = deque()
        self.native_receive = lease.session.receive
        lease.session.receive = self.receive
        self.task = asyncio.create_task(self._read())

    async def _read(self):
        from ouroboros.realtime_session import transport_loss
        while not self.lease.activate.done():
            try:
                events = await self.native_receive()
            except (OSError, ConnectionClosed) as exc:
                if transport_loss(exc):
                    raise RealtimeSetupError('prepared_stale', 'prepared transport closed while waiting') from exc
                raise RealtimeSetupError('provider_closed', 'provider refused the prepared connection') from exc
            for event in events:
                if event.kind == 'closed':
                    code = 'prepared_stale' if event.data.get('transport_loss') else 'provider_closed'
                    raise RealtimeSetupError(code, event.data.get('detail') or 'prepared connection closed')
            if events:
                self.batches.append(events)
            await self.lease.host.check()  # the same ledger/limit, including this connection's meter

    async def receive(self):
        if not self.batches:
            await asyncio.shield(self.task)
        if self.batches:
            return self.batches.popleft()
        return await self.native_receive()

    async def stop(self):
        self.task.cancel()
        await joined(asyncio.gather(self.task, return_exceptions=True))


class PreparedConnection:
    """An opened, silent lease. run adopts it once; close cancels and joins it.

    The lifecycle task owns extension registration and cleanup. Its consumer's
    result is released before close, so the next ready connection can serve while
    the retired one drains its final accounting. Host.finish joins every lease.
    """

    def __init__(self, host, options, *, ahead=False):
        self.host, self.options, self.ahead = host, options, ahead
        loop = asyncio.get_running_loop()
        self.ready, self.activate, self.result = (loop.create_future() for _ in range(3))
        for future in (self.ready, self.result):
            future.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        self.caught_up = False
        self.task = asyncio.create_task(self._lifetime())
        self.task.add_done_callback(self._cleanup_result)
        host._connection_tasks.append(self.task)

    def _cleanup_result(self, task):
        # Retired close may outlive its consumer. Observe its outcome even when
        # the active successor, rather than the original waiter, ends the call.
        error = None if task.cancelled() else task.exception()
        if error is not None and hasattr(self, 'connection'):
            self.connection['cleanup_failure'] = type(error).__name__
            self.view.observe('host', canonical_json({'type': 'call.connection_cleanup_failed',
                                                     'error_class': type(error).__name__}))

    async def run(self, consume):
        await asyncio.shield(self.ready)
        self.activate.set_result(consume)
        try:
            return await asyncio.shield(self.result)
        except asyncio.CancelledError:
            await self.close()
            raise

    async def close(self):
        if not self.task.done():
            self.task.cancel()
        await joined(asyncio.gather(self.task, return_exceptions=True))

    async def _fit(self, credentials, *, rebound=False):
        from ouroboros.observability import persist_physical_candidate
        from ouroboros.realtime_setup import session_candidate
        view = self.view
        def fit():
            with credentials.activate():
                return session_candidate(view, self.provider['session'], self.options, observation_fp=credentials.scope)
        # The heavy ordinary ladder runs off the media loop on a private context.
        self.session, self.request, evidence = await joined(asyncio.to_thread(fit))
        from ouroboros.context_images import fresh_call_images
        view._fitted_frame_rows = fresh_call_images(view.messages, owner_mode=view.plan.preferred_mode)
        view._context_session = self.session
        if self.options.resume_handle:
            previous = getattr(getattr(self.host, '_connection_view', None), '_context_session', None)
            if previous is not None and hasattr(previous, 'held_context_rows') and hasattr(self.session, 'held_context_rows'):
                self.session.held_context_rows = copy.deepcopy(previous.held_context_rows)
        self.host.timing_event('fit_done')
        self.connection['context_ready_at'] = time.time()
        self.host.timing.setdefault('context_ready_at', self.connection['context_ready_at'])
        view._physical_context = self.request.physical_context
        view._held_input = view._growth = 0
        view._history_at_receipt = view._history_estimate()
        view._serving_history_at_receipt = view._serving_history_estimate()
        view._meter_watermark = 0.0
        receipt = await joined(asyncio.to_thread(persist_physical_candidate, view.ctx.drive_root,
            task_id=view.ctx.task_id, attempt_id=f'call-{view.turn.spec.call_id}-{view.connection_number}' + ('-rebound' if rebound else ''),
            candidate=self.session.wire, candidate_facts=evidence))
        view.observe('host', canonical_json({'type': 'call.prepared', **evidence, 'manifest_ref': receipt['manifest_ref']}))
        view._view_session_factory, view._view_session_options = self.provider['session'], self.options
        view._prepared_view = evidence, receipt['manifest_ref']

    async def _adopt(self):
        from ouroboros.live_context_view import record_session_view
        from ouroboros.realtime_setup import measure_session_wire, text_input_estimate, rows_for_engine
        from ouroboros.llm_attempt import main_context_fit_refusal
        host, view = self.host, self.view
        async with host._tools_lock:
            if self.ahead:
                from ouroboros.realtime_credentials import capture_session_credentials
                if capture_session_credentials(host.turn, self.provider).scope != self.credentials.scope:
                    raise RealtimeSetupError('prepared_stale', 'account changed while preparing')
            # An authored edit/revision to the fitted prefix needs a fresh fit;
            # append-only arrivals use the ordinary ordered context update lane.
            if host.messages[:len(self.source)] != self.source or host.schemas != self.source_schemas:
                raise RealtimeSetupError('prepared_stale', 'canonical prefix or tools changed while preparing')
            tail = copy.deepcopy(host.messages[len(self.source):])
            evidence, manifest = view._prepared_view
            if tail:
                _, request, evidence, _ = await joined(asyncio.to_thread(measure_session_wire,
                    view, view.plan, [*rows_for_engine(view.messages, vision=host.model.vision,
                        fresh=not self.options.resume_handle, owner_mode=view.plan.preferred_mode, schemas=view.schemas), *tail], view.schemas, self.provider['session'], self.options,
                    _room_fact=False, fresh=False))
                if main_context_fit_refusal(request) is not None:
                    raise RealtimeSetupError('prepared_stale', 'new canonical rows need the ordinary ladder again')
                measured = evidence['fit']['measurement']
                capacity = measured['capacity_total_tokens']
                remaining = (capacity - measured['estimated_input_tokens'] - measured['response_reserve_tokens']
                             if capacity is not None else None)
                evidence['work_room'] = {**view._prepared_view[0]['work_room'], 'remaining_tokens': remaining}
                evidence['catchup_measurement_wire_sha256'] = evidence['wire_sha256']
                evidence['wire_sha256'] = view._prepared_view[0]['wire_sha256']
                view._prepared_view = evidence, manifest
                view._physical_context = request.physical_context
                view._selected_raw_estimate = evidence['fit']['measurement']['raw_input_tokens']
                view._selected_text_estimate = text_input_estimate([*view.messages, *tail], view.schemas)
                view._density_input = 0  # the initial wire and catch-up are distinct text envelopes
                view.observe('host', canonical_json({'type': 'call.catchup', 'rows': tail, 'fit': evidence['fit']}))
            host.messages[:] = [*view.messages, *tail]
            if not self.options.resume_handle:
                from ouroboros.context_images import evict_stale_image_blocks
                evict_stale_image_blocks(host.messages, owner_mode=view.plan.preferred_mode, readable_only=True, reader_available=False)
            host.schemas[:] = view.schemas
            from ouroboros.tools.tool_discovery import bind_resident_schemas
            enabled = set(host.ctx._actor_loaded_tool_names) | set(view.ctx._actor_loaded_tool_names)
            host.ctx._actor_loaded_tool_names = bind_resident_schemas(host.tools, host.schemas)
            host.ctx._actor_loaded_tool_names.update(enabled)
            host._context_changed()
            view.messages, view.schemas = host.messages, host.schemas
            view._history_at_receipt = view._history_estimate()
            view._serving_history_at_receipt = view._serving_history_estimate()
            host._connection_view = view
            if host._prepared_successor is self:
                host._prepared_successor = None
            publish(host, view)
            record_session_view(host, evidence, manifest)
            self.host.timing_event('switched')
            self.connection['switched_at'] = time.time()
            if tail:
                await self.session.send(ContextUpdate(tuple(rows_for_engine(tail, vision=host.model.vision, schemas=view.schemas)), respond=False))
                view._fitted_frame_rows.extend(tail)  # these pixels reached the new connection before consume
                self.caught_up = True
            if not self.options.resume_handle:
                host.tool_results_delivered(host.messages)

    async def _lifetime(self):
        with ExitStack() as scope:
            await self._owned_lifetime(scope)

    async def _owned_lifetime(self, scope):
        from ouroboros.extension_registry_state import extension_work_scope
        from ouroboros.loop_model_call import _presend_context_fit
        from ouroboros.realtime_credentials import capture_session_credentials
        from ouroboros.realtime_setup import loaded_provider, _LoopWork
        from ouroboros.response_limits import positive_limit, resolve_response_limit
        from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed
        host = self.host
        self.session = self.reader = None
        try:
            await host.check()
            self.provider = loaded_provider(host.turn.provider)
            if self.provider['extension_generation'] != host.turn._generation:
                raise RealtimeSetupError('provider_changed', 'selected loaded generation changed; start a new call')
            work = _LoopWork(); work.task = asyncio.current_task()
            scope.enter_context(extension_work_scope(self.provider, work))
            from ouroboros.live_context_view import drain_exposure
            await drain_exposure(host)
            async with host._tools_lock:
                self.source_revision = host._history_version
                self.source, self.source_schemas = copy.deepcopy(host.messages), copy.deepcopy(host.schemas)
                view = self.view = copy.copy(host)
                view._connection_owner, view._connection_view = host, None
                view.ctx, view.tools = copy.copy(host.ctx), copy.copy(host.tools)
                view.ctx._last_context_observation = copy.deepcopy(getattr(host.ctx, '_last_context_observation', None))
                view.messages, view.schemas = copy.deepcopy(self.source), copy.deepcopy(self.source_schemas)
                view.ctx.messages, view.tools._ctx = view.messages, view.ctx
                view.tools._handler_overrides = dict(host.tools._handler_overrides)
                view.ctx._actor_loaded_tool_names = set(getattr(host.ctx, '_actor_loaded_tool_names', ()))
                host._connection_sequence += 1
                view.connection_number = host._connection_sequence
                view._connection_accounting = False
                view._connection_quote = None
                view._caption_dirty = False
                view._history_cached_version = -1
                self.connection = {'number': view.connection_number, 'prepared_ahead': self.ahead}
                host.timing['connections'].append(self.connection)
            limit = resolve_response_limit(host.ctx.drive_root, provider=host.turn.route.accounting_provider,
                model=host.turn.model, base_url=host.turn.route.endpoint, allow_fetch=False)
            bound = (positive_limit(limit.max_output_tokens) or None) if not limit.stale else None
            self.options = replace(self.options, model_output_bound_tokens=bound)
            credentials = capture_session_credentials(host.turn, self.provider)
            await self._fit(credentials)
            if self.ahead and not getattr(self.session, 'silent_context_updates', False):
                self.ready.set_result(None)
                self.result.set_result(None)
                return
            with _presend_context_fit() as admit:
                if not admit(self.request):
                    raise PhysicalAttemptPreconditionFailed('context_capacity')
                await host.check()
                latest = capture_session_credentials(host.turn, self.provider)
                if latest.scope != credentials.scope:
                    await joined(self.session.close())
                    credentials = latest
                    await self._fit(credentials, rebound=True)
                    if not admit(self.request):
                        raise PhysicalAttemptPreconditionFailed('context_capacity')
                    await host.check()
                view._connection_quote = copy.deepcopy(host.quote)
                view._connection_accounting = True
                self.credentials = credentials
                with credentials.activate():
                    await self.session.open()
                self.connection['open_at'] = time.time()
                host.timing.setdefault('session_open_at', self.connection['open_at'])
                if self.ahead:
                    host.timing_event('prepared_ready')
                    self.reader = WaitingReader(self)
                self.ready.set_result(self)
                if self.reader is not None:
                    await asyncio.wait([self.activate, self.reader.task], return_when=asyncio.FIRST_COMPLETED)
                    if self.reader.task.done():
                        await self.reader.task  # failure closes and settles immediately, even while waiting for quiet
                consume = await self.activate
                await host.check()
                try:
                    await self._adopt()
                except (OSError, ConnectionClosed) as exc:
                    from ouroboros.realtime_session import transport_loss
                    if self.ahead and transport_loss(exc):
                        raise RealtimeSetupError('prepared_stale', 'prepared transport closed before handoff') from exc
                    raise
                token = CURRENT.set(view)
                try:
                    result = await consume(self.session)
                finally:
                    CURRENT.reset(token)
                self.result.set_result(result)
        except BaseException as exc:
            if isinstance(exc, RealtimeSetupError) and exc.code == 'context_capacity':
                exc.source_revision = getattr(self, 'source_revision', None)
            for future in (self.ready, self.result):
                if not future.done():
                    future.set_exception(exc)
        finally:
            try:
                try:
                    if self.reader is not None:
                        await self.reader.stop()
                finally:
                    if self.session is not None:
                        await joined(self.session.close())
            finally:
                if hasattr(self, 'connection'):
                    self.connection['closed_at'] = time.time()
                if host._prepared_successor is self:
                    host._prepared_successor = None
                if hasattr(self, 'view'):
                    self.view._connection_accounting = False
                    self.view._connection_quote = None
                    if host._connection_view is self.view:
                        host._connection_accounting = False
                        host._connection_quote = None
                if host._caption_dirty:
                    await joined(asyncio.to_thread(host.checkpoint, 'ready'))
                    host._caption_dirty = False
