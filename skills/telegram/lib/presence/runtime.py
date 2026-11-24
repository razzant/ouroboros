"""Bounded Presence custody runtime driven by the unified Telegram poller."""

from __future__ import annotations

import asyncio, contextlib, inspect, json, pathlib, re, sqlite3, threading, time, uuid
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from ...scripts.telegram_settings import load_settings, telegram_api_base, telegram_files_dir, telegram_proxy
from ..telegram_poll_binding import InjectNotAccepted, update_key
from ..telegram_token import binding_for
from . import outbox_files
from .status import build_status_payload, build_status_snapshot
from .custody import CALL_STARTED, REACTION_WITHHELD, CustodyStore, InboxFullError, InboxLease, OutboxLease, recorded_calls
from .delivery import (
    REACTION_HISTORY_UNAVAILABLE, REACTION_REPORT_UNREADABLE, delivery_report,
    is_reaction_report, typed_reaction_report,
)
from .discovery import group_sightings
from .events import parse_telegram_update, too_large_marks, too_large_note
from .host import (
    PlaceBindingRevoked, PlaceBindingUnavailable, PresenceHostHTTPError, PresenceSubmission,
    PresenceWorkResult, bind_place_sync,
)
from .outbox_worker import (
    MAX_OUTBOX_ATTEMPTS, AlbumIncomplete, LeaseLost, PlaceHeld, ProviderCalls, ReactionDeliveryHelper,
    deliver_outbox, format_error, is_reaction, recorded_end, retry_delay, send_method,
)
from .transport import (
    CLOUD_DOWNLOAD_LIMIT, PresenceBotClient, TelegramApiError, TelegramFileTooLarge,
    TelegramFileUnavailable, TelegramRequestNotSent,
)

_BOT_ID = re.compile(r"[1-9][0-9]*")
_REACTION_KINDS = frozenset({"message_reaction", "message_reaction_count"})
_NOT_STAGED = frozenset({"too_large", "failed", "unsupported"})
_DOWNLOAD_FAILURES_BEFORE_FAILED = 6
_DOWNLOAD_FAILING_SECONDS = 1800.0
_POLL_EVIDENCE_SECONDS = 120.0
_TYPING_RENEW_SECONDS = 4.5
_TYPING_CAP_SECONDS = 120.0
_MAX_STAGING_BYTES = 256 * 1024 * 1024


class PresenceSubmitter(Protocol):
    async def submit(self, event: Dict[str, Any], staged_files: Sequence[pathlib.Path]) -> PresenceSubmission: ...
    async def poll(self, work_ref: str, binding_id: str) -> PresenceWorkResult: ...


class PresenceRuntime:
    """Durable Presence inbox/work/outbox/typing workers sharing the bridge's bot token and transport."""

    def __init__(
        self,
        *,
        api: Any,
        state_dir: pathlib.Path,
        submitter: PresenceSubmitter,
        trust_env: bool = False,
        client_factory: Optional[Callable[[], Optional[PresenceBotClient]]] = None,
        inbox_workers: int = 2,
    ):
        self.api = api
        self.state_dir = pathlib.Path(state_dir)
        self.store = CustodyStore.shared(self.state_dir / "custody.sqlite3")
        self.submitter = submitter
        self._trust_env = bool(trust_env)
        self._custom_client_factory = client_factory
        self.inbox_workers = max(1, min(8, int(inbox_workers)))
        self._stopping = False
        self._stop_event: Optional[asyncio.Event] = None
        self._stopped_event = threading.Event()
        self._stopped_event.set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._run_task: Optional[asyncio.Task[Any]] = None
        self._bot_id = self._bot_username = self._validated_token = self._validated_api_base = ""
        self._bot_label = "not connected"
        self._validation_gen = 0
        self._runtime_state = "not_started"
        self._last_error = ""
        self._provider_ok_at = 0.0
        self._next_discover_at = 0.0
        self._typing_sent_at: Dict[Tuple[str, str], float] = {}
        self._delivered_at: Dict[Tuple[str, str], float] = {}
        self._reactions = ReactionDeliveryHelper(
            self.store, ready_client=self._ready_client, resolved_reporting=self._resolved_reporting,
            check_place=lambda payload, thread_id: self._check_place(payload, thread_id=thread_id),
            note_error=self._set_last_error, check_account=self._check_row_account,
            current_gen=lambda: self._validation_gen,
        )

    def _set_last_error(self, message: str) -> None:
        self._last_error = message

    def _current_token(self) -> Tuple[str, int, Any]:
        binding = binding_for(self.api)
        if binding.state == "pending":
            binding.read_now(self.api)
        token, generation = binding.snapshot()
        return str(token or "").strip(), generation, binding

    def _matches_binding(self, validation_gen: int, api_server: str) -> bool:
        if self._validation_gen != validation_gen or not self._bot_id:
            return False
        try:
            live_base = telegram_api_base(self.state_dir)
            if self._validated_api_base and live_base != self._validated_api_base:
                return False
            return live_base == api_server
        except Exception:
            return False

    def _client_is_current(self, client: Any, validation_gen: int) -> bool:
        if self._validation_gen != validation_gen or not self._bot_id:
            return False
        check = getattr(client, "is_current", None)
        return True if not callable(check) else bool(check())

    def _build_client(self) -> Optional[PresenceBotClient]:
        if self._custom_client_factory is not None:
            return self._custom_client_factory()
        token, generation, binding = self._current_token()
        if not token or not binding.is_validated() or token != self._validated_token or not _BOT_ID.fullmatch(self._bot_id):
            return None
        api_server, val_gen = telegram_api_base(self.state_dir), self._validation_gen
        return PresenceBotClient(
            token, api_server=api_server, proxy=telegram_proxy(self.state_dir), trust_env=self._trust_env,
            files_dir=telegram_files_dir(self.state_dir), binding=binding, generation=generation,
            is_current=lambda gen=val_gen, srv=api_server: self._matches_binding(gen, srv),
        )

    def on_bot_validated(self, token: str, bot_id_value: str, username: str = "", *, api_base: str = "") -> None:
        """Record a successful ``getMe`` from the unified poller without making a second request."""
        clean_token, clean_id, clean_user = str(token or "").strip(), str(bot_id_value or "").strip(), str(username or "").strip()
        clean_base = str(api_base or "").strip() or telegram_api_base(self.state_dir)
        if clean_token and _BOT_ID.fullmatch(clean_id):
            with contextlib.suppress(sqlite3.Error, OSError):
                self.store.record_confirmed_bot(clean_token, clean_id, api_base=clean_base)
        elif clean_token:
            with contextlib.suppress(sqlite3.Error, OSError):
                self.store.forget_confirmed_bot(clean_token, api_base=clean_base)
        self._bot_id = clean_id if _BOT_ID.fullmatch(clean_id) else ""
        self._bot_username = clean_user
        self._bot_label = f"@{clean_user}" if clean_user else (self._bot_id or "not connected")
        self._validated_token, self._validated_api_base = clean_token, clean_base
        self._validation_gen += 1
        if self._bot_id:
            self.submitter.connector_identity = {"sends_as": {"id": self._bot_id, "label": self._bot_label, "kind": "bot"}}
        self._runtime_state, self._last_error = "polling", ""

    def _clear_validated_identity(self, *, label: str, state: str) -> None:
        self._bot_id = self._bot_username = self._validated_token = self._validated_api_base = ""
        self._bot_label, self._runtime_state = label, state
        self._validation_gen += 1
        self.submitter.connector_identity = {}
        with contextlib.suppress(sqlite3.Error, OSError):
            self.store.clear_typing()

    def on_token_cleared(self) -> None:
        self._clear_validated_identity(label="not configured", state="waiting_for_token")

    def on_token_changed(self) -> None:
        self.on_token_cleared() if not self._current_token()[0] else self._clear_validated_identity(label="not connected", state="connecting")

    def on_token_rejected(self, token: str, error: str = "", *, api_base: str = "") -> None:
        clean_token = str(token or "").strip()
        if clean_token:
            with contextlib.suppress(sqlite3.Error, OSError):
                self.store.forget_confirmed_bot(clean_token, api_base=str(api_base or "").strip() or telegram_api_base(self.state_dir))
        self._clear_validated_identity(label="not connected", state="provider_error")
        self._last_error = str(error or "Telegram rejected the bot token")[:500]

    def on_poll_ok(self) -> None:
        self._provider_ok_at = time.time()
        if self._runtime_state in {"provider_error", "not_started", "waiting_for_token"} and self._bot_id:
            self._runtime_state = "polling"

    def on_poll_error(self, error: str) -> None:
        self._runtime_state = "provider_error"
        self._last_error = str(error or "")[:500]

    async def ingest_update(
        self, update: Dict[str, Any], *, account_id: str, bot_username: str = "", record_only: bool = False
    ) -> bool:
        """Commit a polled update into Presence custody (and always update chat discovery sightings)."""
        update_id = _update_id(update)
        if update_id is None:
            return False
        account = str(account_id or self._bot_id or "").strip()
        if not account:
            return False
        sightings = group_sightings(update, account)
        raw_key = update_key(update)
        sighting_key = f"{account}:{raw_key}" if raw_key else ""
        if record_only:
            try:
                self.store.commit_update(update_id, None, sightings, sighting_key=sighting_key)
            except sqlite3.OperationalError:
                pass
            return False
        discover = getattr(self.submitter, "discover_delivery_support", None)
        status = getattr(self.submitter, "delivery_reporting_status", "unknown")
        if discover and status not in {"supported", "unsupported"}:
            now = time.monotonic()
            if status == "unknown" or now >= self._next_discover_at:
                await discover()
                status = getattr(self.submitter, "delivery_reporting_status", "unknown")
                if status not in {"supported", "unsupported"}:
                    self._next_discover_at = now + 5.0
        # Default to places=True while Host discovery is unavailable so startup updates never drop
        # channels, reactions, anonymous admins or facts; PresenceHostClient.submit() strips facts on legacy hosts.
        places = bool(getattr(self.submitter, "places_version", 0)) if status in {"supported", "unsupported"} else True
        settings = load_settings(self.state_dir)
        event = parse_telegram_update(
            update,
            bot_account_id=account,
            management_group_id=str(settings.get("management_group_id") or "").strip(),
            places=places,
            bot_username=str(bot_username or self._bot_username or "").strip(),
            words=settings.get("words") if isinstance(settings.get("words"), list) else [],
            download_limit=CLOUD_DOWNLOAD_LIMIT,
        )
        try:
            return self.store.commit_update(update_id, event, sightings, sighting_key=sighting_key)
        except InboxFullError as exc:
            self._last_error = str(exc)
            self._log("warning", f"telegram presence: {exc}; recording overflow for update {update_id}")
            with contextlib.suppress(sqlite3.OperationalError):
                self.store.record_inbox_overflow(
                    update_id, event.conversation_key if event is not None else "", sightings, sighting_key=sighting_key
                )
            return False
        except sqlite3.OperationalError as exc:
            raise InjectNotAccepted(f"Presence custody busy ({type(exc).__name__}); deferring update") from exc

    async def run(self) -> None:
        self._stopping = False
        self._stopped_event.clear()
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        self._run_task = asyncio.current_task()
        workers: List[asyncio.Task[Any]] = []
        try:
            self.store.register_owner()
            released = self.store.release_own_leases()
            if any(released.values()):
                self._log("info", f"telegram presence: released stale leases {released}")
            self.store.clear_typing()
            outbox_files.sweep_staging_tmp(self.state_dir)
            discover = getattr(self.submitter, "discover_delivery_support", None)
            if discover:
                await discover()
            if self._runtime_state in {"not_started", "stopped"}:
                if self._validated_token and self._bot_id:
                    self._runtime_state, self._last_error = "polling", ""
                else:
                    token, _, _ = self._current_token()
                    self._runtime_state = "connecting" if token else "waiting_for_token"
            for index in range(self.inbox_workers):
                workers.append(asyncio.create_task(self._inbox_worker(), name=f"telegram-presence-inbox-{index}"))
            for coro, wname in ((self._work_worker, "work"), (self._outbox_worker, "outbox"), (self._typing_worker, "typing")):
                workers.append(asyncio.create_task(coro(), name=f"telegram-presence-{wname}"))
            stop_wait = asyncio.create_task(self._stop_event.wait(), name="telegram-presence-stop")
            try:
                while not self._stop_event.is_set():
                    needs_discover = bool(
                        discover and getattr(self.submitter, "delivery_reporting_status", "unknown") not in {"supported", "unsupported"}
                    )
                    done, _ = await asyncio.wait(
                        [stop_wait, *workers], timeout=5.0 if needs_discover else None, return_when=asyncio.FIRST_COMPLETED
                    )
                    if not done:
                        if needs_discover and not self._stop_event.is_set():
                            with contextlib.suppress(Exception):
                                await discover()
                        continue
                    for finished in done:
                        if finished is stop_wait or self._stop_event.is_set():
                            continue
                        exc = finished.exception() if not finished.cancelled() else None
                        if exc is not None:
                            raise exc
                        raise RuntimeError(f"Presence worker {finished.get_name()} exited unexpectedly")
            finally:
                stop_wait.cancel()
                await asyncio.gather(stop_wait, return_exceptions=True)
        finally:
            self._stopping = True
            if self._stop_event is not None: self._stop_event.set()
            for task in workers: task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            self._runtime_state, self._run_task, self._loop = "stopped", None, None
            self.store.unregister_owner()
            self.store.close()
            self._stopped_event.set()

    def stop(self, *, wait_sec: float = 0.0) -> None:
        """Thread-safe stop; waits up to ``wait_sec`` off-loop so unload never returns mid-request."""
        self._stopping = True
        loop, stop_event, run_task = self._loop, self._stop_event, self._run_task

        def _signal_on_loop() -> None:
            if stop_event is not None:
                stop_event.set()
            if run_task is not None and not run_task.done():
                run_task.cancel()

        if loop is not None and loop.is_running():
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                _signal_on_loop()
            else:
                try:
                    loop.call_soon_threadsafe(_signal_on_loop)
                except RuntimeError:
                    self.store.close()
                    self._stopped_event.set()
                else:
                    if wait_sec > 0:
                        self._stopped_event.wait(timeout=wait_sec)
        elif run_task is None or run_task.done():
            self.store.close()
            self._stopped_event.set()

    def bind_place_sync(self, account_id: str, conversation_id: str, thread_id: str = "") -> Optional[Dict[str, Any]]:
        host_base, token_provider = getattr(self.submitter, "host_base", ""), getattr(self.submitter, "token_provider", None)
        if not host_base or not callable(token_provider):
            return None
        fresh_fn = getattr(self.submitter, "identity_is_fresh", None)
        fresh = bool(fresh_fn()) if callable(fresh_fn) else True
        status = str(getattr(self.submitter, "delivery_reporting_status", "unknown") or "unknown") if fresh else "unknown"
        return bind_place_sync(
            host_base=host_base, token_provider=token_provider, account_id=account_id,
            conversation_id=conversation_id, thread_id=thread_id,
            places_version=getattr(self.submitter, "places_version", None) if fresh else None,
            place_binding_version=getattr(self.submitter, "place_binding_version", None) if fresh else None,
            delivery_reporting_status=status,
        )

    def confirmed_bot(self) -> Tuple[str, str]:
        """The bot id validated for the current token and API base in this process, else ``("", reason)``."""
        if self._run_task is None:
            return "", "Telegram bot not running"
        token, _, _ = self._current_token()
        if not token:
            return "", "Bot token missing"
        if token != self._validated_token or not _BOT_ID.fullmatch(self._bot_id):
            return "", "Bot identity is unconfirmed; check the transport status"
        if self._validated_api_base and telegram_api_base(self.state_dir) != self._validated_api_base:
            return "", "Bot API server changed; awaiting re-validation"
        return self._bot_id, ""

    def recorded_bot(self) -> str:
        """The bot ``getMe`` last confirmed for the current token and API base, across processes."""
        binding = binding_for(self.api)
        if self._run_task is None or binding.state == "pending":
            binding.read_now(self.api)
        token, _ = binding.snapshot()
        token = str(token or "").strip()
        return self.store.confirmed_account(token, api_base=telegram_api_base(self.state_dir)) if token else ""

    async def place_readiness(self, conversation_id: str) -> Dict[str, Any]:
        """Query the Host for the current Place state of ``conversation_id`` (read-only)."""
        account = self._bot_id or self.recorded_bot()
        status_fn = getattr(self.submitter, "place_status", None)
        if not callable(status_fn) or not account or not conversation_id:
            return {"state": "unavailable"}
        return await status_fn(account, str(conversation_id))

    def status_snapshot(self, custody: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return build_status_snapshot(self, custody)

    def status_payload(
        self,
        bridge_status: Dict[str, Any],
        mini_app_status: Dict[str, Any],
        settings: Dict[str, Any],
        *,
        requested_account_id: str = "",
    ) -> Dict[str, Any]:
        """Full Circles-compatible ``GET status`` response combining Presence custody and bridge state."""
        return build_status_payload(
            self, bridge_status, mini_app_status, settings, requested_account_id=requested_account_id
        )

    def _check_row_account(self, payload: Dict[str, Any]) -> None:
        row_account = str(payload.get("account_id") or (payload.get("_reporting") or {}).get("account_id") or "").strip()
        if not row_account:
            raise PlaceHeld("Held: queued row has no confirmed Telegram bot account_id")
        if not self._bot_id:
            raise PlaceHeld("Held: waiting for validated Telegram bot token")
        if row_account != self._bot_id:
            raise PlaceHeld(f"Held: queued for bot {row_account}, active bot is {self._bot_id}")

    async def _ready_client(self) -> Optional[PresenceBotClient]:
        client = self._build_client()
        if client is None:
            token, _, _ = self._current_token()
            if not token:
                self._runtime_state = "waiting_for_token"
            return None
        return client

    async def _inbox_worker(self) -> None:
        while not self._stopping:
            try:
                if not await self.process_one_inbox():
                    await self._pause(0.5)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = format_error(exc)
                self._log("warning", f"telegram presence: inbox worker error: {self._last_error}")
                await self._pause(1.0)

    async def _ensure_inbox_binding(self, lease: InboxLease, pb: Optional[Dict[str, Any]], *, require: bool = False, stage: str = "staging") -> Optional[Dict[str, Any]]:
        binder = getattr(self.submitter, "bind_place", None)
        if pb is None and callable(binder) and (not require or getattr(self.submitter, "place_binding_version", 0) >= 1):
            pb = await binder(str(lease.payload.get("account_id") or self._bot_id), str(lease.payload.get("conversation_id") or ""), str(lease.payload.get("thread_id") or ""))
            if require and pb is None:
                raise PlaceBindingUnavailable(f"Place binding unavailable for {stage} on Places host")
        if isinstance(pb, dict) and lease.payload.get("_place_binding") != pb:
            lease.payload["_place_binding"] = pb
            if not self.store.checkpoint_inbox(lease.event_id, lease.payload, lease_token=lease.lease_token or None):
                raise LeaseLost(f"inbox lease lost before {stage}")
        return pb

    async def process_one_inbox(self) -> bool:
        lease = self.store.claim_inbox(lease_sec=1860.0)
        if lease is None:
            return False
        tok = lease.lease_token or None
        row_account = str(lease.payload.get("account_id") or "").strip()
        if row_account and (not self._bot_id or row_account != self._bot_id):
            reason = f"Held: queued for bot {row_account}, active bot is {self._bot_id}" if self._bot_id else f"Held: waiting for bot {row_account} validation"
            self.store.defer_inbox(lease.event_id, reason=reason, retry_after_sec=5.0, lease_token=tok)
            return True
        if str((lease.payload.get("message") or {}).get("event_kind") or "") in _REACTION_KINDS:
            verdict = await self._reaction_deliverable(lease.payload)
            if verdict == "defer":
                self.store.defer_inbox(lease.event_id, reason="presence host unavailable for reaction check", retry_after_sec=5.0, lease_token=tok)
                return True
            if verdict != "deliver":
                self.store.mark_inbox_failed(lease.event_id, reason=REACTION_WITHHELD, lease_token=tok)
                return True
        staged: List[pathlib.Path] = []
        try:
            pb = lease.payload.get("_place_binding") if isinstance(lease.payload.get("_place_binding"), dict) else None
            pb = await self._ensure_inbox_binding(lease, pb, stage="staging")
            staged = await self._stage_attachments(lease)
            if tok and not self.store.inbox_lease_valid(lease.event_id, tok):
                raise LeaseLost("inbox lease lost before submit")
            submit_event = {k: v for k, v in lease.payload.items() if k != "_place_binding"}
            try:
                result = await self.submitter.submit(submit_event, staged)
            except PresenceHostHTTPError as refusal:
                delay = refusal.busy_retry_after
                if delay is not None:
                    self.store.defer_inbox(lease.event_id, reason=format_error(refusal), retry_after_sec=delay, lease_token=tok)
                    self._remove_staged_files(staged)
                    return True
                if refusal.disposition == "retained":
                    pb = await self._ensure_inbox_binding(lease, pb, require=True, stage="retention")
                if not self._settle_refusal(lease, refusal):
                    raise
            else:
                if result.text or result.work_ref or result.continuation_ref:
                    pb = await self._ensure_inbox_binding(lease, pb, require=True, stage="reply")
                self.store.record_submission(
                    lease.event_id, binding_id=result.binding_id, turn_ref=result.turn_ref,
                    outcome=result.outcome, text=result.text, work_ref=result.work_ref,
                    delivery_reporting_version=result.delivery_reporting_version,
                    continuation_ref=result.continuation_ref, output_ref=result.output_ref,
                    reply_anchor=result.reply_anchor, lease_token=tok,
                )
            self._remove_staged_files(staged)
            return True
        except LeaseLost as exc:
            self._remove_staged_files(staged)
            self._last_error = format_error(exc)
            return True
        except PlaceBindingUnavailable as exc:
            self._remove_staged_files(staged)
            if exc.code == "place_not_connected":
                self.store.mark_inbox_failed(lease.event_id, reason=format_error(exc), lease_token=tok)
            else:
                self.store.release_inbox(lease.event_id, reason=format_error(exc), retry_after_sec=retry_delay(lease.attempts), lease_token=tok)
                if exc.code == "presence_auth_blocked":
                    await self._pause(2.0)
            self._last_error = format_error(exc)
            return True
        except asyncio.CancelledError:
            self._remove_staged_files(staged)
            self.store.release_inbox(lease.event_id, reason="worker_cancelled", retry_after_sec=0, lease_token=tok)
            raise
        except Exception as exc:
            self._remove_staged_files(staged)
            wait = getattr(exc, "retry_after", None) or 0.0
            self.store.release_inbox(lease.event_id, reason=format_error(exc), retry_after_sec=max(retry_delay(lease.attempts), wait), lease_token=tok)
            self._last_error = format_error(exc)
            if isinstance(exc, PresenceHostHTTPError) and str(exc.payload.get("code") or "") == "presence_auth_blocked":
                await self._pause(2.0)
            return True

    async def _reaction_deliverable(self, event: Dict[str, Any]) -> str:
        if "reaction" not in (event.get("message") or {}):
            return "withhold"
        discover = getattr(self.submitter, "discover_delivery_support", None)
        if discover:
            await discover()
        if getattr(self.submitter, "delivery_reporting_status", "unknown") not in {"supported", "unsupported"}:
            return "defer"
        return "deliver" if getattr(self.submitter, "presence_reaction_context", 0) == 1 else "withhold"

    def _settle_refusal(self, lease: InboxLease, refusal: PresenceHostHTTPError) -> bool:
        receipt, tok = {"http_status": refusal.status_code, "response": refusal.payload}, lease.lease_token or None
        code = str(refusal.payload.get("code") or "")
        if refusal.disposition == "retained":
            self.store.record_retention(lease.event_id, binding_id=refusal.binding_id, receipt=receipt, lease_token=tok)
        elif refusal.disposition == "rejected" or (
            refusal.disposition == "blocked" and code in {"place_not_connected", "place_unattributable"}
        ):
            self.store.mark_inbox_failed(lease.event_id, reason=format_error(refusal), receipt=receipt, lease_token=tok)
        else:
            return False
        self._last_error = format_error(refusal)
        return True

    def _remove_staged_files(self, staged: Sequence[pathlib.Path]) -> None:
        outbox_files.remove_staged_files(self.state_dir, staged, on_error=lambda m: self._log("warning", m))

    async def _stage_attachments(self, lease: InboxLease) -> List[pathlib.Path]:
        message = lease.payload.get("message") if isinstance(lease.payload.get("message"), dict) else {}
        attachments = message.get("attachments") or []
        if not attachments:
            return []
        client = await self._ready_client()
        if client is None:
            raise RuntimeError("telegram token is unavailable")
        limit, tok = client.download_limit, lease.lease_token or ""
        attempt_dir = f"{outbox_files.safe_component(lease.event_id)}--{tok[:16]}" if tok else outbox_files.safe_component(lease.event_id)
        event_dir = self.state_dir / "staging" / attempt_dir
        staged, settled, waiting = [], False, None
        try:
            for index, attachment in enumerate(attachments):
                if not isinstance(attachment, dict) or attachment.get("status") in _NOT_STAGED or not attachment.get("file_id"):
                    continue
                declared = outbox_files.declared_size(attachment)
                marks = too_large_marks(declared, limit)
                if marks:
                    attachment.update(marks)
                    settled = True
                    continue
                name = outbox_files.safe_filename(str(attachment.get("file_name") or f"attachment-{index}"))
                destination = event_dir / f"{index}-{name}"
                if not destination.is_file() or destination.stat().st_size == 0:
                    remaining = outbox_files.stage_budget_remaining(self.state_dir, _MAX_STAGING_BYTES)
                    if remaining <= 0 or declared > remaining:
                        attachment.update(status="failed", note="staging disk cap reached; attachment not downloaded")
                        settled = True
                        continue
                    tmp_dest = self.state_dir / "staging_tmp" / f"{uuid.uuid4().hex}-{index}-{name}"
                    try:
                        if "max_bytes" in inspect.signature(client.download_file).parameters:
                            await client.download_file(str(attachment["file_id"]), tmp_dest, max_bytes=remaining)
                        else:
                            await client.download_file(str(attachment["file_id"]), tmp_dest)
                        if tok and not self.store.inbox_lease_valid(lease.event_id, tok):
                            raise LeaseLost("inbox lease lost during attachment download")
                        outbox_files.commit_staged_file(self.state_dir, tmp_dest, destination, _MAX_STAGING_BYTES)
                    except outbox_files.StagingStorageFull:
                        attachment.update(status="failed", note="staging disk cap reached; attachment not downloaded")
                        settled = True
                        continue
                    except TelegramFileTooLarge as exc:
                        if not exc.error_code and remaining < limit and declared <= limit:
                            attachment.update(status="failed", note="staging disk cap reached; attachment not downloaded")
                        else:
                            attachment.update(status="too_large", note=too_large_note(declared, limit, exc.description if exc.error_code else ""))
                        settled = True
                        continue
                    except TelegramFileUnavailable as exc:
                        attachment.update(status="failed", note=f'Telegram answered "{exc.description}"')
                        settled = True
                        continue
                    except TelegramApiError as exc:
                        note = self._bounded_failure(lease.event_id, index, exc)
                        if note is None:
                            if waiting is None or (exc.retry_after or 0) > (waiting.retry_after or 0):
                                waiting = exc
                            if exc.error_code == 429:
                                break
                            continue
                        attachment.update(status="failed", note=note)
                        settled = True
                        continue
                    finally:
                        with contextlib.suppress(OSError):
                            tmp_dest.unlink(missing_ok=True)
                attachment["status"] = "staged"
                staged.append(destination)
            if settled and not self.store.checkpoint_inbox(lease.event_id, lease.payload, lease_token=tok or None):
                raise LeaseLost("inbox lease lost while checkpointing attachment status")
            if waiting is not None:
                raise waiting
            if tok and not self.store.inbox_lease_valid(lease.event_id, tok):
                raise LeaseLost("inbox lease lost after staging")
            self.store.forget_download_failures(lease.event_id)
            return staged
        except BaseException:
            self._remove_staged_files(staged)
            raise

    def _bounded_failure(self, event_id: str, ordinal: int, exc: TelegramApiError) -> Optional[str]:
        now = time.time()
        code = exc.error_code
        if code in (401, 429) or (code == 404 and exc.endpoint == "getFile"):
            counted = False
        elif code:
            counted = True
        else:
            previous = self.store.download_failure(event_id, ordinal)
            since = max(float(previous["last_failed_at"]) if previous else 0.0, now - _POLL_EVIDENCE_SECONDS)
            counted = self._provider_ok_at >= since
        record = self.store.note_download_failure(event_id, ordinal, error=format_error(exc), counted=counted, now=now)
        failing = now - float(record["first_failed_at"] or now)
        if not counted or record["own_failures"] < _DOWNLOAD_FAILURES_BEFORE_FAILED or failing < _DOWNLOAD_FAILING_SECONDS:
            return None
        last = f'Telegram answered "{exc.description}"' if code else f"no answer ({exc.description})"
        return f"not received after {record['own_failures']} attempts over {failing / 60:.0f} min; last: {last}"

    async def _work_worker(self) -> None:
        while not self._stopping:
            try:
                if not await self.process_one_work():
                    await self._pause(0.5)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = format_error(exc)
                self._log("warning", f"telegram presence: work worker error: {self._last_error}")
                await self._pause(1.0)

    async def process_one_work(self) -> bool:
        lease = self.store.claim_work(lease_sec=120.0)
        if lease is None:
            return False
        tok = lease.lease_token or None
        try:
            result = await self.submitter.poll(lease.work_ref, lease.binding_id)
            if result.outputs and not self.store.record_work_outputs(lease.work_ref, result.outputs, lease_token=tok):
                return True
            if result.terminal:
                self.store.complete_work(
                    lease.work_ref, status=result.status,
                    text=result.text if result.outcome in {"message", "deferred"} and not result.output_ref else "",
                    child_ref=result.child_work_ref, reply_anchor=result.reply_anchor, lease_token=tok,
                )
            elif result.status == "unresolved":
                self.store.release_work(
                    lease.work_ref, reason=f"presence_work_unresolved: {result.code}",
                    retry_after_sec=retry_delay(lease.attempts), child_ref=result.child_work_ref, lease_token=tok,
                )
            else:
                self.store.release_work(
                    lease.work_ref, reason="presence_work_pending",
                    retry_after_sec=5.0, child_ref=result.child_work_ref, lease_token=tok,
                )
            return True
        except asyncio.CancelledError:
            self.store.release_work(lease.work_ref, reason="worker_cancelled", retry_after_sec=0, lease_token=tok)
            raise
        except Exception as exc:
            self.store.release_work(lease.work_ref, reason=format_error(exc), retry_after_sec=retry_delay(lease.attempts), lease_token=tok)
            self._last_error = format_error(exc)
            return True

    async def _outbox_worker(self) -> None:
        reporting = None
        try:
            while not self._stopping:
                try:
                    if reporting is None or reporting.done():
                        if reporting is not None:
                            try:
                                await reporting
                            except asyncio.CancelledError:
                                raise
                            except Exception as exc:
                                self._last_error = format_error(exc)
                                self._log("warning", f"telegram presence: delivery report error: {self._last_error}")
                        reporting = asyncio.create_task(self.process_one_delivery_report())
                    if not await self.process_one_outbox():
                        await self._pause(0.5)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._last_error = format_error(exc)
                    self._log("warning", f"telegram presence: outbox worker error: {self._last_error}")
                    await self._pause(1.0)
        finally:
            if reporting is not None:
                reporting.cancel()
                await asyncio.gather(reporting, return_exceptions=True)

    async def process_one_delivery_report(self) -> bool:
        pending = self.store.next_delivery_report()
        if pending is None:
            return False
        delivery_id, index, payload = pending
        try:
            if is_reaction_report(payload):
                discover = getattr(self.submitter, "discover_delivery_support", None)
                if discover:
                    await discover()
                capable = getattr(self.submitter, "presence_explicit_completion", 0) == 1
                payload = typed_reaction_report(payload) if capable else None
                if payload is None:
                    self.store.finish_delivery_report(
                        delivery_id, index,
                        error=REACTION_REPORT_UNREADABLE if capable else REACTION_HISTORY_UNAVAILABLE,
                    )
                    return True
            await self.submitter.report_delivery(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.store.finish_delivery_report(delivery_id, index, error=format_error(exc))
        else:
            self.store.finish_delivery_report(delivery_id, index)
        return True

    async def process_one_outbox(self) -> bool:
        lease = self.store.claim_outbox(lease_sec=120.0)
        if lease is None:
            return False
        if is_reaction(lease.payload):
            await self._reactions.process(lease)
            return True
        if recorded_calls(lease.payload) and self._settled_by_record(lease):
            return True
        calls, client, val_gen = ProviderCalls(self.store, lease), None, self._validation_gen
        tok, verified_fds = calls.lease_token or None, {}
        try:
            client = await self._ready_client()
            if client is None:
                raise PlaceHeld("Held: waiting for validated Telegram bot token")
            val_gen = self._validation_gen
            self._check_row_account(lease.payload)
            lease = replace(lease, payload={**lease.payload, "_reporting": await self._resolved_reporting(lease.payload)})
            calls.checkpoint(lease.payload)

            def _ensure_current() -> None:
                self._check_row_account(lease.payload)
                if not self._client_is_current(client, val_gen):
                    raise PlaceHeld("Held: Telegram bot identity or server binding changed")
                if not self.store.outbox_lease_valid(lease.delivery_id, calls.fence, tok):
                    raise LeaseLost("this attempt no longer holds the outbox row; nothing was sent")

            async def before_send() -> None:
                _ensure_current()
                await self._check_place(lease.payload, thread_id=str(lease.payload.get("topic_id") or ""))
                try:
                    fresh_fds = await asyncio.to_thread(outbox_files.open_verified, lease.payload, state_dir=self.state_dir, require_for_media=True)
                except outbox_files.FrozenFileMissing as exc:
                    raise PlaceHeld(f"Held: {exc}") from exc
                outbox_files.close_verified(verified_fds)
                verified_fds.update(fresh_fds)
                _ensure_current()

            receipt = await deliver_outbox(
                client, lease, self.store, before_send=before_send, verified_fds=verified_fds,
                account_id=str(lease.payload.get("account_id") or self._bot_id), delivered=self._note_delivered, calls=calls,
            )
            if self.store.mark_delivered(lease.delivery_id, provider_receipt=receipt, attempts=calls.fence, lease_token=tok):
                outbox_files.discard(outbox_files.records(lease.payload), self.state_dir)
            return True
        except PlaceHeld as exc:
            self.store.hold_outbox(lease.delivery_id, str(exc), attempts=calls.fence, lease_token=tok)
            return True
        except PlaceBindingRevoked as exc:
            report = delivery_report(lease.delivery_id, lease.payload, part_id="status", state="failed", error=str(exc))
            if self.store.mark_outbox_failed(lease.delivery_id, reason=str(exc), report=report, attempts=calls.fence, lease_token=tok):
                outbox_files.discard(outbox_files.records(lease.payload), self.state_dir)
            return True
        except asyncio.CancelledError:
            self.store.release_outbox(lease.delivery_id, reason="worker_cancelled", retry_after_sec=0, attempts=calls.fence, lease_token=tok)
            raise
        except LeaseLost as exc:
            self._last_error = format_error(exc)
            if exc.message_ids:
                self._log("warning", f"telegram presence: {lease.delivery_id} attempt {lease.attempts}: Telegram confirmed message_id {exc.message_ids} in chat {lease.payload.get('chat_id')} after lease loss")
            return True
        except Exception as exc:
            if (
                isinstance(exc, AlbumIncomplete)
                or calls.open is not None
                or (lease.payload.get("kind") in {"operation", "moderation"} and isinstance(exc, TelegramApiError) and not isinstance(exc, TelegramRequestNotSent) and exc.error_code == 0)
            ):
                self._end_uncertain(lease.delivery_id, format_error(exc), attempts=calls.fence, lease_token=tok)
                return True
            if calls.recorded and self._settled_by_stored_record(lease, exc):
                return True
            if isinstance(exc, TelegramRequestNotSent) and not self._client_is_current(client, val_gen):
                self.store.hold_outbox(lease.delivery_id, f"Held: {format_error(exc)}", attempts=calls.fence, lease_token=tok)
                return True
            wait = max(retry_delay(lease.attempts), float(getattr(exc, "retry_after", None) or 0.0))
            if isinstance(exc, TelegramApiError) and exc.error_code == 429:
                self.store.hold_outbox(lease.delivery_id, f"Rate limited: {format_error(exc)}", retry_after_sec=wait, attempts=calls.fence, lease_token=tok)
                self._last_error = format_error(exc)
                return True
            if lease.attempts >= MAX_OUTBOX_ATTEMPTS:
                payload = self.store.outbox_payload(lease.delivery_id)
                state = "failed" if (recorded_calls(payload) or isinstance(exc, TelegramRequestNotSent) or (isinstance(exc, RuntimeError) and str(exc) == "telegram token is unavailable") or (isinstance(exc, TelegramApiError) and exc.error_code)) else "uncertain"
                report = delivery_report(lease.delivery_id, payload, part_id="status", state=state, error=format_error(exc))
                if self.store.mark_outbox_failed(lease.delivery_id, reason=format_error(exc), report=report, state=state, attempts=calls.fence, lease_token=tok) and state == "failed":
                    outbox_files.discard(outbox_files.records(payload), self.state_dir)
            else:
                self.store.release_outbox(lease.delivery_id, reason=format_error(exc), retry_after_sec=wait, attempts=calls.fence, lease_token=tok)
            self._last_error = format_error(exc)
            return True
        finally:
            outbox_files.close_verified(verified_fds)

    def _settled_by_stored_record(self, lease: OutboxLease, exc: Exception) -> bool:
        try:
            stored = self.store.outbox_payload(lease.delivery_id)
        except Exception as read_error:
            self._last_error = format_error(read_error)
            return True
        if not self._settled_by_record(replace(lease, payload=stored, unrecorded_attempt=False)):
            return False
        self._last_error = format_error(exc)
        return True

    def _settled_by_record(self, lease: OutboxLease) -> bool:
        payload, tok = lease.payload, lease.lease_token or None
        end, value = recorded_end(payload)
        if (CALL_STARTED in payload and end != "uncertain") or (not end and lease.unrecorded_attempt):
            marker = payload.get(CALL_STARTED)
            method = str(marker.get("method") or "") if isinstance(marker, dict) else send_method(payload)
            end, value = "uncertain", format_error(TelegramApiError(method or "send", "send_result_unrecorded"))
        if not end:
            return False
        try:
            if end == "delivered":
                if self.store.mark_delivered(lease.delivery_id, provider_receipt=value, attempts=lease.attempts, lease_token=tok):
                    outbox_files.discard(outbox_files.records(payload), self.state_dir)
            else:
                self._end_uncertain(lease.delivery_id, value, attempts=lease.attempts, lease_token=tok)
        except Exception as exc:
            self._last_error = format_error(exc)
        return True

    def _end_uncertain(self, delivery_id: str, error: str, *, attempts: Optional[int] = None, lease_token: Optional[str] = None) -> None:
        payload = self.store.outbox_payload(delivery_id)
        report = delivery_report(delivery_id, payload, part_id="status", state="uncertain", error=error)
        self.store.mark_outbox_failed(delivery_id, reason=error, report=report, state="uncertain", attempts=attempts, lease_token=lease_token)
        self._last_error = error

    def _note_delivered(self, chat_id: str, topic_id: Any) -> None:
        self._delivered_at[(str(chat_id), str(topic_id or ""))] = time.time()

    async def _typing_worker(self) -> None:
        while not self._stopping:
            try:
                await self.process_typing()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log("warning", f"telegram presence: typing indicator skipped: {format_error(exc)}")
            await self._pause(1.0)

    async def process_typing(self) -> int:
        requests = self.store.typing_requests()
        if not requests:
            self._typing_sent_at.clear()
            return 0

        def over(row: Dict[str, Any], key: Tuple[str, str], now_ts: float) -> bool:
            return (
                now_ts - float(row["requested_at"]) > _TYPING_CAP_SECONDS
                or self._delivered_at.get(key, 0.0) >= float(row["requested_at"])
                or not self.store.turn_live(row["source_event_id"], row["turn_ref"])
            )

        def _typing_account(row: Dict[str, Any]) -> str:
            acct = str(row.get("account_id") or "").strip()
            if acct:
                return acct
            parts = str(row.get("source_event_id") or "").split(":")
            return parts[1] if len(parts) >= 3 and parts[0] == "telegram" and parts[1].isdigit() else ""

        now = time.time()
        ended: List[Dict[str, Any]] = []
        live: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for row in requests:
            key = (str(row["chat_id"]), str(row["topic_id"] or ""))
            row_account = _typing_account(row)
            if not row_account or not self._bot_id or row_account != self._bot_id or over(row, key, now):
                ended.append(row)
            else:
                live.setdefault(key, []).append(row)
        sent = 0
        for key, rows in live.items():
            last = self._typing_sent_at.get(key)
            if last is not None and now - last < _TYPING_RENEW_SECONDS:
                continue
            try:
                client = await self._ready_client()
                if client is None:
                    raise RuntimeError("telegram token is unavailable")
                val_gen = self._validation_gen
                if any(_typing_account(row) != self._bot_id for row in rows):
                    ended.extend(rows)
                    continue
                for raw_pb in sorted({str(row.get("place_binding_json") or "").strip() for row in rows}) or [""]:
                    parsed_pb = json.loads(raw_pb) if raw_pb else None
                    if raw_pb and not isinstance(parsed_pb, dict):
                        raise PlaceBindingRevoked("malformed typing place binding")
                    await self._check_place({"chat_id": key[0], **({"_place_binding": parsed_pb} if parsed_pb else {})}, thread_id=key[1])
                if not self._client_is_current(client, val_gen) or any(_typing_account(row) != self._bot_id for row in rows):
                    ended.extend(rows)
                    continue
                at = time.time()
                stale = [row for row in rows if over(row, key, at)]
                if stale:
                    ended.extend(stale)
                    rows = [row for row in rows if row not in stale]
                    if not rows:
                        continue
                await client.send_chat_action(key[0], "typing", topic_id=int(key[1]) if key[1] else None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log("info", f"telegram presence: typing stopped for chat {key[0]}: {format_error(exc)}")
                ended.extend(rows)
                continue
            self._typing_sent_at[key] = at
            sent += 1
        if ended:
            self.store.forget_typing(ended)
            for key in {(str(row["chat_id"]), str(row["topic_id"] or "")) for row in ended} - set(live):
                self._typing_sent_at.pop(key, None)
        return sent

    async def _resolved_reporting(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        reporting = dict(payload.get("_reporting") or {"version": 0})
        discover = getattr(self.submitter, "discover_delivery_support", None)
        if reporting.get("version") == -1:
            reporting["version"] = await discover() if discover else 0
            reporting["status"] = getattr(self.submitter, "delivery_reporting_status", "unsupported")
        if reporting.get("version") == 1 and not reporting.get("account_id"):
            reporting["account_id"] = self._bot_id
        if reporting.get("version") == 1:
            if is_reaction(payload) and discover:
                await discover()
            reporting["explicit_completion"] = getattr(self.submitter, "presence_explicit_completion", 0)
        return reporting

    async def _check_place(self, payload: Dict[str, Any], *, thread_id: str = "") -> None:
        check = getattr(self.submitter, "place_hold", None)
        if not check:
            return
        kw: Dict[str, Any] = {"thread_id": thread_id} if thread_id else {}
        binding = payload.get("_place_binding")
        params = inspect.signature(check).parameters
        if isinstance(binding, dict) and ("binding" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())):
            kw["binding"] = binding
        reason = await check(self._bot_id, str(payload.get("chat_id") or ""), **kw)
        if reason:
            raise PlaceHeld(reason)

    async def _pause(self, seconds: float) -> None:
        if self._stop_event is None:
            await asyncio.sleep(max(0.01, seconds))
            return
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=max(0.01, seconds))
        except asyncio.TimeoutError:
            return

    def _log(self, level: str, message: str) -> None:
        callback = getattr(self.api, "log", None)
        if callable(callback):
            callback(level, message)


def _update_id(update: Dict[str, Any]) -> Optional[int]:
    try:
        return int(update.get("update_id"))
    except (AttributeError, TypeError, ValueError):
        return None
