"""Count a prospective Nano root before publishing one active memory revision.

The common revision orders cooperating writers across processes. Measurement is
outside both the revision fence and the individual store locks. This guards the
sampled root; a measured write does not qualify all future inputs or publishers.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
import errno
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping
from uuid import uuid4

from ouroboros.nano_minimum import COUNTER_BUSY, measure_minimum
from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

WORK_FLOOR = 15_000
MEMORY_CEILING = 20_000
QUALIFIED_WINDOW = 80_000
PUBLICATION_OPENING = "Work starts here."
INSTRUCTION_ENABLE = "always_in_context"
INSTRUCTION_RELEASE = "always_in_context_release"
PUBLICATION_REVISION = Path("memory/.nano-publication.revision")
_TEXT_PREVIEW: ContextVar[Mapping[Path, bytes]] = ContextVar("nano_text_preview", default={})
_BACKGROUND: ContextVar[dict | None] = ContextVar("nano_publication_background", default=None)
_LATEST_SAMPLE = Path("state/nano_minimum_latest.json")
_SAMPLE_NUMBERS = ("context_window", "memory_tokens", "reply_floor_tokens", "minimum_reply_allowance_tokens",
                   "minimum_work_headroom_at_reply_floor_tokens", "minimum_work_headroom_at_wire_allowance_tokens")
_LOG = logging.getLogger(__name__)


def publication_background(key, read):
    """Share an unchanged producer's observation within one comparison, never between writes."""
    background = _BACKGROUND.get()
    if background is None:
        return read()
    if key not in background:
        try:
            background[key] = (read(), None)
        except Exception as exc:
            background[key] = (None, exc)
    value, failure = background[key]
    if failure is not None:
        raise failure
    return value


def prospective_text(path: Path) -> bytes | None:
    """Canonical producers use these bytes only while capturing the prospective root."""
    preview = _TEXT_PREVIEW.get()
    return preview.get(path.resolve()) if preview else None


def source_bytes(path: Path) -> bytes | None:
    """Absence is a revision; read failures are not absence."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


_NAME_LOCK_TIMEOUT_SEC = 10.0


@contextmanager
def _publication_lock(root: Path):
    """One name for a snapshot, a commit and a writer that counts nothing.

    Both platform tiers acquire that same name before using any kernel lock.
    A differing per-process probe cannot separate the critical sections.
    Counts and retention stay outside; ordinary writers contend only at publication.
    """
    name_path = Path(root) / "memory" / ".nano-publication.name.lock"
    outcome: dict[str, Any] = {}
    fd = acquire_exclusive_file_lock(
        name_path, timeout_sec=_NAME_LOCK_TIMEOUT_SEC, stale_sec=90.0,
        owner_aware_stale=True, outcome=outcome,
    )
    if fd is None:
        if outcome.get("reason") == "contention":
            raise TimeoutError(f"Nano publication name lock busy: {name_path}")
        raise OSError(outcome.get("errno") or errno.EIO,
                      f"Nano publication name lock unavailable ({outcome.get('reason', 'unknown')}): {name_path}")
    try:
        yield Path(root) / PUBLICATION_REVISION
    finally:
        release_exclusive_file_lock(name_path, fd)


def _marker_field(raw: bytes | None, field: str) -> str | None:
    try:
        value = json.loads(raw) if raw else None
    except ValueError:
        return None
    found = value.get(field) if isinstance(value, dict) else None
    return found if isinstance(found, str) else None


def _checked_profile(raw: bytes | None) -> str | None:
    return _marker_field(raw, "checked_profile")


# The confirmed-counter record every marker rewrite carries: the serving profile that counted
# exactly, the formatter it counted with, and the saved launch values proven to start it again.
_CAPABILITY = ("checked_profile", "checked_formatter", "checked_saved")


def _capability(previous: bytes | None, checked_profile: str | None = None, **bindings: str | None) -> dict:
    """Keep the observed capability across an invalidation or an observation; a new profile keeps
    only the bindings given for it. A brief counter outage is not a new provider."""
    kept = _checked_profile(previous)
    row = ({field: value for field in _CAPABILITY if (value := _marker_field(previous, field))}
           if checked_profile in (None, kept) else {})
    row["checked_profile"] = checked_profile or kept
    row.update({field: value for field, value in bindings.items() if field in _CAPABILITY and value})
    return row


def _invalidate_publication_reads(revision_path: Path, *, checked_profile: str | None = None,
                                  counter_unavailable: bool = False, **bindings: str | None) -> bytes:
    # A writer invalidates the in-flight captures *before* it can publish. Even
    # if the writer then dies, it cannot publish on the old witness. Replacing
    # the marker atomically also preserves the last invalidation if one dies
    # while writing the next marker.
    from ouroboros.utils import atomic_write_json

    row = {"revision": uuid4().hex, **_capability(source_bytes(revision_path), checked_profile, **bindings)}
    if counter_unavailable:
        row["reason"] = "confirmed_counter_unavailable"
    atomic_write_json(revision_path, row)
    return revision_path.read_bytes()


def _formatter_digest(sample: Mapping[str, Any]) -> str | None:
    """The serving formatter an exact count reported (its provenance and windows), as a digest."""
    formatter = sample.get("formatter")
    if not isinstance(formatter, dict) or not formatter:
        return None
    bound = {"formatter": formatter, **{key: sample.get(key) for key in ("context_window", "raw_context_window")}}
    return sha256(json.dumps(bound, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@contextmanager
def publication_fence(root: Path):
    """Order: installation publication fence, then source lock, then active write.

    Take neither for counting. Receipt/retention work that cannot change the
    active bytes does not need this fence. Owner ingress and Panic never take it.
    """
    with _publication_lock(root) as marker:
        _invalidate_if_present(marker)
        yield


def _invalidate_if_present(marker: Path) -> None:
    """An ordinary write withdraws existing publication evidence, but creates none."""
    if source_bytes(marker) is not None:
        _invalidate_publication_reads(marker)


@dataclass(frozen=True)
class MemoryChange:
    path: Path | None
    before: bytes | None
    after: bytes | None
    # Multi-object stores provide a semantic candidate (e.g. a normalized
    # active mark or a note publication), not copies of their private stores.
    # Their own writer checks the local revision before commit.
    subject: str = ""


@dataclass(frozen=True)
class RootFloor:
    """A fresh root capture with the normal local formatter and count-only transport.

    A prospective capture may use an in-memory overlay of the edited source,
    never a temporary replacement of the active store. ``rows`` and ``actual``
    are the unchanged ordinary root input checking the serving binding; the
    tentative memory is rendered only in the count-only N/B pair.
    """
    plan: Any
    rows: list
    actual: dict
    caller_max: int
    prepare: Callable[[list], dict]
    measure: Callable[[dict], dict]
    variant: str = "active"

    def _shown(self, section_id: str) -> str:
        core = self.plan.core
        return shown_digest(next((row for row in (*core.semi_stable_sections, *core.dynamic_head_sections,
                                                  *core.dynamic_sections)
                                  if getattr(row, "section_id", "") == section_id), ""))

    def instruction_digest(self) -> str:
        return self._shown("always_on_skills")

    def catalogue_digest(self) -> str:
        """The Installed Skills section exactly as this capture showed it."""
        return self._shown("installed_skills")

    def world_digest(self) -> str:
        return self._shown("world")

    def count(self) -> dict:
        if (self.plan.preferred_mode != "nano" or self.plan.provider != "local"
                or self.plan.model_role != "main" or self.plan.view_kind != "owner_turn"):
            return {"status": "unknown", "reason": "outside_local_text_root"}
        return measure_minimum(SimpleNamespace(profile="owner_nano", nano_floor_plan=(self.plan, "publication_candidate")),
                               self.rows, self.actual, self.measure(self.actual), self.caller_max,
                               self.prepare, self.measure)


def shown_digest(section: Any) -> str:
    """Both forms a host section can be shown in (``HostSection``: its whole text and its operative view)."""
    return sha256(json.dumps([str(section or ""), getattr(section, "operative_view", None)]).encode("utf-8")).hexdigest()


class NanoPublicationRefused(RuntimeError):
    def __init__(self, reason: str, detail: str = "", *, reader: str = "", source_ref: dict | None = None) -> None:
        self.reason, self.detail, self.reader, self.source_ref = reason, detail, reader, source_ref
        super().__init__(f"{reason}: {detail}" + (f"; attempted source: {reader}" if reader else ""))

    def tool_detail(self) -> str:
        """Tell the author how to recover at the refusal, beside the actual proposal reader if one was kept."""
        repair = {
            "over_capacity": "Keep the proposal. Free room by folding memory, or revise the proposal before retrying.",
            "busy": "Keep the proposal and retry when the counter is free.",
            "unavailable": "Keep the proposal until the source and exact counter can be checked again.",
            "stale": "Read the current revision and reconcile the proposal before retrying.",
            "incomplete": "Keep the proposal until the unreadable source can be read whole again.",
            "bootstrap_pending": "Keep the proposal; the ordinary start writes this file, then retry the same edit.",
        }.get(self.reason)
        return f"{self}\n{repair}" if repair else str(self)


def _pressure(sample: dict) -> tuple[int, int]:
    """Price both independent limits. Do not count raw allocator padding as window.

    Working space here is the attainable headroom at the reply floor, not the
    simultaneously free space at a dispatched cap. Qualification of elasticity
    is the host's separate obligation; this code does not change reply sizing.
    """
    memory = sample.get("whole_memory_tokens", sample["memory_tokens"])
    remaining = sample["minimum_work_headroom_at_reply_floor_tokens"]
    return max(0, memory - MEMORY_CEILING), max(0, WORK_FLOOR - remaining)


def publication_status(*, provider: str, measurement: Mapping[str, Any] | None = None) -> dict:
    """Passive Settings model; report the measured sample separately from growth protection.

    The sample is a real Main input, so its headroom is room *beyond* the work text it already carries: below
    ``WORK_FLOOR`` it proves no work shortfall, only that the floor is unconfirmed. Writers keep ``_pressure`` on
    their neutral opening, where that headroom is the work budget.
    """
    exact = bool(measurement and measurement.get("status") == "exact")
    sample, overshoot = "unavailable", None
    if (exact and measurement.get("context_window") == QUALIFIED_WINDOW
            and type(measurement.get("memory_tokens")) is int
            and type(headroom := measurement.get("minimum_work_headroom_at_reply_floor_tokens")) is int):
        memory, overflow = _pressure(measurement)[0], max(0, -headroom)
        sample = ("outside_limits" if memory or overflow else
                  "within_limits" if headroom >= WORK_FLOOR else "work_floor_unconfirmed")
        overshoot = {"memory_excess_tokens": memory, "prepared_overflow_tokens": overflow}
    reason = ("unsupported_route" if provider != "local" else
              "exact_counter_unproven" if not exact else
              "publication_scope_unqualified" if measurement.get("context_window") == QUALIFIED_WINDOW else
              "outside_qualified_window")
    return {"status": "not_claimed", "reason": reason, "measurement": "exact" if exact else "unavailable",
            "current_sample": sample, "pressure": overshoot}


def record_root_sample(root: Path, floor: Mapping[str, Any]) -> None:
    """Settings describes one prepared fresh Main; other roots retain their attempt manifests."""
    from ouroboros.config import DATA_DIR
    from ouroboros.nano_host_publication import profile_stamp, serving_counter_key
    from ouroboros.utils import atomic_write_json, utc_now_iso

    scope = floor.get("scope")
    root = Path(root).resolve()
    if (root != Path(DATA_DIR).resolve() or scope != "cold_start" or floor.get("room") != "main"
            or floor.get("held_sections")):
        return
    if floor.get("counter_confirmed") and (key := serving_counter_key(floor)):
        bindings = {field: value for field, value in {"checked_formatter": _formatter_digest(floor),
                                                      **profile_stamp()}.items() if value}
        incomplete = floor.get("reason_code") == SOURCE_READ_INCOMPLETE
        recovered = floor.get("status") == "exact" or incomplete
        with _publication_lock(root) as marker:
            current = source_bytes(marker)
            if serving_counter_key(floor) == key and (
                    _checked_profile(current) != key or incomplete
                    or (recovered and _marker_field(current, "reason") == "confirmed_counter_unavailable")
                    or any(_marker_field(current, field) != value for field, value in bindings.items())):
                _invalidate_publication_reads(marker, checked_profile=key, **bindings)
    minimum = (floor.get("minimum") or {}).get("input_tokens")
    exact = (floor.get("status") == "exact" and type(minimum) is int
             and type(floor.get("process_id")) is int and all(type(floor.get(k)) is int for k in _SAMPLE_NUMBERS))
    sample = {"status": "exact" if exact else "unknown", "scope": scope, "observed_at": utc_now_iso()}
    if exact:
        actual = floor.get("actual") or {}
        sample.update({k: floor[k] for k in _SAMPLE_NUMBERS}, minimum_input_tokens=minimum,
                      _process_id=floor["process_id"], _port=floor.get("port"), _rules_sha256=floor.get("rules_sha256"))
        optional = {"sent_max_tokens": actual.get("sent_max_tokens"), "actual_input_tokens": actual.get("input_tokens"),
                    "whole_memory_tokens": floor.get("whole_memory_tokens"),
                    "actual_work_headroom_at_wire_allowance_tokens": floor.get("actual_work_headroom_at_wire_allowance_tokens")}
        sample.update({k: v for k, v in optional.items() if type(v) is int})
    path = root / _LATEST_SAMPLE
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, sample)


def read_root_sample(root: Path, *, provider: str) -> dict:
    """Read one small sample for the current serving instance without any inference or journal reads."""
    result = publication_status(provider=provider)
    if provider != "local":
        return result
    observation = _last_root_observation(root, result)
    from ouroboros.nano_host_publication import current_protection

    return {**observation, **current_protection(root)}


def _last_root_observation(root: Path, result: dict) -> dict:
    try:
        with (Path(root) / _LATEST_SAMPLE).open("rb") as handle:
            raw = handle.read(2049)
        sample = json.loads(raw) if len(raw) <= 2048 else None
        if not isinstance(sample, dict) or sample.get("scope") != "cold_start":
            return result
        public = {key: sample[key] for key in ("scope", "observed_at") if isinstance(sample.get(key), str)}
        public["observation"] = "prepared"
        if sample.get("status") != "exact":
            return {**result, **public}
        from ouroboros.local_model import get_manager
        from ouroboros.nano_host_publication import current_rules_digest

        serving = get_manager().serving_context_evidence(bind_endpoint=True)
        if (not serving.get("confirmed") or sample.get("_process_id") != serving.get("process_id")
                or not sample.get("_port") or sample["_port"] != serving.get("port")
                or sample.get("context_window") != serving.get("context_window")):
            return {**result, "reason": "serving_binding_changed", **public}
        if not sample.get("_rules_sha256") or sample["_rules_sha256"] != current_rules_digest():
            return {**result, "reason": "counted_rules_changed", **public}
        numbers = (*_SAMPLE_NUMBERS, "minimum_input_tokens", "sent_max_tokens", "actual_input_tokens",
                   "actual_work_headroom_at_wire_allowance_tokens", "whole_memory_tokens")
        public.update({key: sample[key] for key in numbers if type(sample.get(key)) is int})
        return {**publication_status(provider="local", measurement=sample), **public}
    except (OSError, ValueError, NanoPublicationRefused):
        return result


# A capture that could not read one of its sources whole (``nano_minimum.measure_minimum``): it is
# neither a lost counter nor a never-confirmed formatter, and no write passes on it.
SOURCE_READ_INCOMPLETE = "source_read_incomplete"


def _background_deficit(sample: dict) -> int:
    """How far B (the fresh root without the six budgeted sources) eats into the reserve for whole memory and work."""
    return max(0, sample["without_memory"]["input_tokens"] + sample["reply_floor_tokens"]
               + MEMORY_CEILING + WORK_FLOOR - QUALIFIED_WINDOW)


def _check_counts(before: dict, after: dict, *, compare_pressure=True) -> None:
    repair = (before.get("status") != "exact" and before.get("reason_code") == SOURCE_READ_INCOMPLETE
              and before.get("counter_confirmed") is True and before.get("context_window") == QUALIFIED_WINDOW)
    for label, measure in (("baseline", before), ("candidate", after)):
        if label == "baseline" and repair:
            continue
        if measure.get("status") != "exact" or measure.get("context_window") != QUALIFIED_WINDOW:
            code = measure.get("reason_code")
            reason = "busy" if code == COUNTER_BUSY else "incomplete" if code == SOURCE_READ_INCOMPLETE else "unavailable"
            raise NanoPublicationRefused(reason, f"{label}: {measure.get('reason') or 'unqualified window'}")
        if type((measure.get("without_memory") or {}).get("input_tokens")) is not int:
            raise NanoPublicationRefused("unavailable", f"{label}: the count has no exact background")
        if (measure.get("world_form") or "whole") != "whole":
            raise NanoPublicationRefused("over_capacity", f"{label}: a publication count must use whole WORLD")
    keys = ("process_id", "context_window", "raw_context_window", "formatter", "caller_max_tokens", "reply_floor_tokens")
    if any(before.get(key) is None or before[key] != after.get(key) for key in keys):
        raise NanoPublicationRefused("unavailable", "the two counts did not use the same serving binding and reply policy")
    if repair:
        if (not 0 <= after.get("whole_memory_tokens", after["memory_tokens"]) <= MEMORY_CEILING
                or any(_pressure(after)) or _background_deficit(after)):
            raise NanoPublicationRefused("over_capacity", "a repair of an incomplete baseline must fit all three reserves")
        return
    if not compare_pressure:
        return
    old, new = _pressure(before), _pressure(after)
    # Three independent deficits: whole memory over its ceiling, work under its floor, and B
    # inside the reserve for both. None may appear or grow; a repair that worsens none passes.
    # An already over-limit profile can make verified repairs. Reducing one
    # violation cannot buy a new violation or make another existing one worse.
    if any(new) and (any(a > b for a, b in zip(new, old))
                     or after.get("whole_memory_tokens", after["memory_tokens"]) > before.get("whole_memory_tokens", before["memory_tokens"])):
        raise NanoPublicationRefused("over_capacity", f"memory excess {new[0]} tokens; work shortfall {new[1]} tokens")
    # B is one quantity whichever writer moves it (an instruction, the catalogue, a history
    # selection or account, a receipt): equal or smaller B passes on any profile, and neither a
    # smaller whole request nor a missing earlier proof licenses B into the reserve.
    was, grown = _background_deficit(before), _background_deficit(after)
    if grown > was:
        raise NanoPublicationRefused("over_capacity", f"the background would take {grown} tokens of the room reserved "
                                                      f"for whole memory and work (it took {was})")


def _protection_proof(samples, pair) -> dict | None:
    """The strongest statement shared by all reachable results of this write.

    Some store operations legitimately return without publishing, and a
    multi-file writer can fail after a prefix. Never certify from just 'after'.
    An instruction change invalidates the old background and starts unqualified.
    """
    if any(sample.get("status") != "exact" or (sample.get("world_form") or "whole") != "whole" for sample in samples):
        return None
    background = max(sample["without_memory"]["input_tokens"] for sample in samples)
    reply = samples[0]["reply_floor_tokens"]
    digests = {floor.instruction_digest() for floor in pair}
    catalogues = {floor.catalogue_digest() for floor in pair}
    worlds = {floor.world_digest() if hasattr(floor, "world_digest") else None for floor in pair}
    if (len(digests) != 1 or len(catalogues) != 1 or len(worlds) != 1
            or background + MEMORY_CEILING + WORK_FLOOR + reply > QUALIFIED_WINDOW
            or any(not 0 <= sample.get("whole_memory_tokens", sample["memory_tokens"]) <= MEMORY_CEILING
                   or any(_pressure(sample)) for sample in samples)):
        return None
    world = worlds.pop()
    return {"background_tokens": background, "reply_floor_tokens": reply, "world_form": "whole",
            "instruction_digest": digests.pop(), "catalogue_digest": catalogues.pop(),
            **({"world_digest": world} if world is not None else {})}


@dataclass(frozen=True)
class NanoPublication:
    root: Path
    revision: Callable[[], Any]
    capture: Callable[[tuple[MemoryChange, ...]], tuple[RootFloor, ...] | None]
    retain: Callable[[tuple[MemoryChange, ...]], str | dict] | None
    required: Callable[[], bool] | None = None
    # Only the concrete cross-process host adapter may advertise protection.
    # This proof is an observation for Settings, never reused for admission.
    profile_key: Callable[[], str | None] | None = None
    # Further bindings of the checked profile its adapter can prove now (``_CAPABILITY``).
    profile_stamp: Callable[[], Mapping[str, str]] | None = None

    def _applicable(self) -> bool:
        if self.profile_key is not None:
            return self.profile_key() is not None
        return self.required is None or self.required()

    def _inapplicable_now(self, marker: Path) -> bool:
        try:
            return not self._applicable()
        except NanoPublicationRefused:
            # An unreadable profile does not newly qualify an installation.
            return _checked_profile(source_bytes(marker)) is None

    @contextmanager
    def ordinary(self):
        """Let an inapplicable writer keep its original source lock discipline.

        The decision and an ordinary write take the same short publication fence as a
        capture, checked commit or qualification.
        """
        with _publication_lock(self.root) as marker:
            if self._inapplicable_now(marker):
                if _checked_profile(source_bytes(marker)) is not None:
                    _invalidate_if_present(marker)
                yield True
                return
        yield False

    def prepare(self, changes: MemoryChange | tuple[MemoryChange, ...]):
        """Do the unlocked part now; enter the returned commit context at publication."""
        changes = (changes,) if isinstance(changes, MemoryChange) else changes
        if not changes:
            raise ValueError("a prospective publication must name at least one source")
        unchecked = generation = None
        previously_counted = False
        try:
            # An inapplicable profile never counts; the decision and later write use the same fence.
            with _publication_lock(self.root):
                if not self._applicable():
                    return _PublicationTicket(self, changes, None, ordinary=True)
            # A snapshot cannot begin between another writer's invalidation and
            # its active write; both ends read the marker under the same lock.
            with _publication_lock(self.root) as revision_path:
                generation = source_bytes(revision_path)
                profile = self.profile_key() if self.profile_key is not None else None
                if profile is None if self.profile_key is not None else not self._applicable():
                    return _PublicationTicket(self, changes, None, ordinary=True)
                previously_counted = profile is not None and _checked_profile(generation) == profile
                unchecked = (generation, profile) if profile is not None and not previously_counted else None
                revision = self.revision()
                if revision and generation is None:
                    generation = _invalidate_publication_reads(revision_path)
                    if unchecked is not None:
                        unchecked = (generation, profile)
            if not revision:
                raise NanoPublicationRefused("unavailable", "the active publication revision could not be established")
            if not _files_match(self.root, changes):
                raise NanoPublicationRefused("stale", "the source changed before its count; read it again")
            pair = self.capture(changes)
            if not pair or len(pair) < 2:
                raise NanoPublicationRefused("unavailable", "no complete root capture for this source")
            baseline = pair[0].count()
            if (profile is not None and (baseline.get("counter_confirmed") or baseline.get("status") == "exact")
                    and baseline.get("context_window") == QUALIFIED_WINDOW):
                unchecked = None  # Exact baseline evidence, even before the first successful edit.
                bindings = {"checked_formatter": _formatter_digest(baseline),
                            **(self.profile_stamp() if self.profile_stamp is not None else {})}
                bindings = {field: value for field, value in bindings.items() if value}
                incomplete = baseline.get("reason_code") == SOURCE_READ_INCOMPLETE
                recovered = baseline.get("status") == "exact" or incomplete
                if (not previously_counted or incomplete
                        or (recovered and _marker_field(generation, "reason") == "confirmed_counter_unavailable")
                        or any(_marker_field(generation, field) != value for field, value in bindings.items())
                        or (baseline.get("status") == "exact" and not _protection_proof((baseline,), pair[:1]))):
                    # A first confirmation, a new binding of the same profile, a background past its
                    # reserve or a source that could not be read whole: no earlier proof stands.
                    with _publication_lock(self.root) as marker:
                        if (source_bytes(marker), self.revision()) != (generation, revision):
                            raise NanoPublicationRefused("stale", "the initial exact count no longer names the active source")
                        generation = _invalidate_publication_reads(marker, checked_profile=profile, **bindings)
                previously_counted = True
            samples = (baseline, *(floor.count() for floor in pair[1:]))
            baselines = {getattr(pair[0], "variant", "active"): baseline}
            for floor, sample in zip(pair[1:], samples[1:]):
                variant = getattr(floor, "variant", "active")
                if variant not in baselines:
                    _check_counts(baseline, sample, compare_pressure=False)
                    baselines[variant] = sample
                    continue
                _check_counts(baselines[variant], sample)
            proof = _protection_proof(samples, pair) if profile is not None else None
        except NanoPublicationRefused as error:
            if unchecked is not None and error.reason == "unavailable":
                # Never-confirmed formatter: allow ordinary editing and claim
                # no protection. A known formatter does not get this fallback,
                # and an incomplete source read is not a formatter's outage.
                return _PublicationTicket(self, changes, None, unchecked=unchecked)
            if previously_counted and error.reason == "unavailable":
                self._record_counter_loss(generation, profile)
            raise self._retain(changes, error) from error
        except (OSError, ValueError) as error:
            if unchecked is not None:
                return _PublicationTicket(self, changes, None, unchecked=unchecked)
            raise self._retain(changes, NanoPublicationRefused("unavailable", type(error).__name__)) from error
        return _PublicationTicket(self, changes, (generation, revision), proof)

    def _record_counter_loss(self, generation: bytes | None, profile: str) -> None:
        try:
            with _publication_lock(self.root) as marker:
                if source_bytes(marker) == generation and self.profile_key() == profile:
                    _invalidate_publication_reads(marker, counter_unavailable=True)
        except (OSError, ValueError, NanoPublicationRefused):
            _LOG.warning("Could not record the Nano counter outage", exc_info=True)

    @contextmanager
    def publication(self, changes: MemoryChange | tuple[MemoryChange, ...]):
        """Use outside the source lock; check its revision and commit inside the fence."""
        ticket = self.prepare(changes)
        try:
            with ticket.commit():
                yield
        except NanoPublicationRefused as error:
            raise self._retain(ticket.changes, error) from error

    def _retain(self, changes: tuple[MemoryChange, ...], error: NanoPublicationRefused) -> NanoPublicationRefused:
        if self.retain is None or error.source_ref is not None or error.reader:
            return error
        try:
            kept = self.retain(changes)
        except (OSError, ValueError):
            kept = ""
        ref = kept if isinstance(kept, dict) else None
        reader = f"read_file(root='artifact_store', path={ref['path']!r})" if ref else kept
        return NanoPublicationRefused(error.reason, error.detail + ("" if reader else "; source retention unavailable"),
                                      reader=reader, source_ref=ref)


@dataclass(frozen=True)
class _PublicationTicket:
    admission: NanoPublication
    changes: tuple[MemoryChange, ...]
    witness: tuple | None
    proof: dict | None = None
    unchecked: tuple | None = None
    # An inapplicable profile's write: it commits as an ordinary writer does.
    ordinary: bool = False

    @contextmanager
    def commit(self):
        try:
            with self._commit():
                yield
        except NanoPublicationRefused as error:
            kept = self.admission._retain(self.changes, error)
            if kept is error:
                raise
            raise kept from error

    @contextmanager
    def _commit(self):
        """Only a cheap re-read and the write here; no counter or retention I/O."""
        admission = self.admission
        with ExitStack() as stack:
            try:
                marker = stack.enter_context(_publication_lock(admission.root))
                if self.ordinary:
                    if not admission._inapplicable_now(marker):
                        raise NanoPublicationRefused("stale", "the active profile changed; check the proposal again")
                    _invalidate_if_present(marker)
                elif self.unchecked is not None:
                    if (source_bytes(marker), admission.profile_key()) != self.unchecked:
                        raise NanoPublicationRefused("stale", "the profile or its counter evidence changed; check the proposal again")
                    if marker.is_file():
                        _invalidate_publication_reads(marker)
                else:
                    if ((source_bytes(marker), admission.revision()) != self.witness
                            or not _files_match(admission.root, self.changes)):
                        raise NanoPublicationRefused("stale", "an input changed during counting; read it again")
                    _invalidate_publication_reads(marker)
            except (OSError, ValueError) as error:
                raise NanoPublicationRefused("unavailable", type(error).__name__) from error
            yield
            if self.proof is not None:
                self._record_protection(marker)

    def _record_protection(self, marker: Path) -> None:
        from ouroboros.utils import atomic_write_json, utc_now_iso

        try:
            after = self.admission.revision()
            # Only a rules publication is expected to change the Settings document's identity (the
            # witness's first element); everything after it, including every other setting a fresh
            # root shows, must be what was counted. A mixed save records nothing: its invalidation stands.
            before = self.witness[1]
            if after is not None and (after == before or (
                    any(change.subject == "owner_rules" for change in self.changes) and after[1:] == before[1:])):
                # The observation carries the whole checked capability, never only its profile name.
                atomic_write_json(marker, {"revision": uuid4().hex, **_capability(source_bytes(marker)),
                                           "profile": repr(after), "observed_at": utc_now_iso(), "proof": self.proof})
        except (OSError, ValueError, NanoPublicationRefused):
            # The editor's bytes have landed already. The invalidation done
            # before its write stands, so a lost observation never hides them.
            _LOG.warning("Could not record the Nano publication observation", exc_info=True)


def _files_match(root: Path, changes: tuple[MemoryChange, ...]) -> bool:
    return all(change.path is None or (change.path.is_absolute() and change.path.resolve().is_relative_to(root)
                                      and source_bytes(change.path) == change.before) for change in changes)


def admission_for(ctx: Any, root: Path):
    """Resolve the same installation profile in a worker or in the server."""
    from ouroboros.nano_host_publication import host_admission

    return host_admission(Path(root).resolve(), author=ctx)


def _retain_proposal(ctx, changes: tuple[MemoryChange, ...]) -> dict:
    """Retain the new authored inputs only; never archive a baseline store or settings."""
    from base64 import b64encode

    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.tool_access_paths import canonical_data_root
    from ouroboros.utils import machine_json

    rows = []
    for change in changes:
        row = {"source": change.subject or (change.path.name if change.path else "memory")}
        try:
            row["text"] = change.after.decode("utf-8") if change.after is not None else None
        except UnicodeDecodeError:
            row["content_base64"] = b64encode(change.after).decode("ascii")
        rows.append(row)
    data = machine_json(rows).encode("utf-8")
    for root in dict.fromkeys((Path(ctx.drive_root).resolve(), canonical_data_root(ctx))):
        ref = store_actor_source_bytes(root, ctx.task_id, category="context_checkpoints",
                                       source_id="nano-proposed-memory", data=data, extension="json", fsync=True)
    return {**ref, "task_id": ctx.task_id}


def _preview_sources(root: Path, changes: tuple[MemoryChange, ...], overlays, stack: ExitStack) -> None:
    files = {(root / "memory" / name).resolve()
             for name in ("identity.md", "scratchpad_blocks.json", "WORLD.md", "registry.md")}
    preview = {change.path.resolve(): change.after for change in changes
               if change.path is not None and not change.subject and change.after is not None}
    remaining = [change for change in changes if change.subject or change.path is None or change.after is None]
    if not preview.keys() <= files or len(preview) + len(remaining) != len(changes):
        raise NanoPublicationRefused("unavailable", "unsupported direct memory preview")
    for subject in sorted({change.subject for change in remaining}):
        if subject not in overlays:
            raise NanoPublicationRefused("unavailable", "no renderer for this active source")
    for overlay in dict.fromkeys(overlays[change.subject] for change in remaining):
        stack.enter_context(overlay(root, changes))
    token = _TEXT_PREVIEW.set(preview)
    stack.callback(_TEXT_PREVIEW.reset, token)


def local_root_publication(env, memory, task, *, author_ctx, client, tool_schemas, caller_max,
                           scope_revision, overlays=None, root_ctx=None, tool_registry=None,
                           intermediate=None, alternate_view=None) -> NanoPublication:
    """Bind a qualified envelope to the ordinary root builder and built-in count transport.

    The host supplies the root view and publication-wide revision reader, not
    the writer's (possibly delegated) resident plan. The caller is responsible
    for qualifying membership, other authors and dynamic bounds. No attempt here
    bootstraps that qualification. ``overlays`` maps registered semantic previews
    to the domain's read-only capture context manager. An alternate-view
    factory prices a reachable background against its own baseline.
    """
    from copy import copy
    from dataclasses import replace

    from ouroboros.context import _context_fit_route, build_context_fit_plan
    from ouroboros.local_model import get_manager
    from ouroboros.memory import Memory

    root = Path(task.get("budget_drive_root") or getattr(env, "budget_drive_root", None) or memory.drive_root).resolve()
    canonical = Memory(drive_root=root)
    # This is a capacity probe for the installation's configured Main, never
    # an attempt to send the proposing work's text or inherit its route pin.
    # Keep the selected room/view but do not carry the writer's text, attachments,
    # model override or saved execution into the bracket. Count fixed protocol
    # framing once. The actual run still carries its owner words in full.
    recipe = {key: task[key] for key in ("project_id", "chat_id", "workspace_root", "metadata") if key in task}
    recipe.update(type="task", id=str(task.get("id") or ""), text=PUBLICATION_OPENING,
                  delegation_role="root", budget_drive_root=str(root))

    def revision():
        scope = scope_revision()
        evidence = get_manager().serving_context_evidence(bind_endpoint=True)
        if (not scope or not evidence.get("confirmed") or evidence.get("context_window") != QUALIFIED_WINDOW
                or not evidence.get("process_id") or not evidence.get("port")):
            return None
        from hashlib import sha256

        # A known absence is part of the revision; a read error raises and is never an absence.
        # The host counts first-start defaults through a view that writes nothing
        # (``nano_host_publication._first_start_view``).
        content = (canonical.identity_path(), canonical.scratchpad_path(), canonical.world_path(),
                   canonical.scratchpad_blocks_path())
        witnesses = tuple(sha256(raw).hexdigest() if (raw := source_bytes(file)) is not None else None for file in content)
        return repr((scope, evidence["process_id"], evidence["port"], witnesses))

    def capture(changes):
        if (task.get("type") != "task" or task.get("_is_direct_chat") or task.get("_presence_turn")
                or task.get("parent_task_id") or task.get("delegation_role") not in (None, "", "root") or
                getattr(root_ctx, "_held_note_capture", None) is not None):
            return None
        world = canonical.world_path()
        if not getattr(memory, "renders_world_preview", False):
            try:
                world.stat()
            except FileNotFoundError as exc:
                raise NanoPublicationRefused("bootstrap_pending", "WORLD is written by the ordinary start") from exc
            except OSError as exc:
                raise NanoPublicationRefused("incomplete", "WORLD could not be read") from exc
            if any(change.path is not None and change.path.resolve() == world for change in changes):
                return None  # only a memory that reads the prospective map can price it
        from ouroboros.chronicle_store import capture as chronicle_capture

        token = _BACKGROUND.set({})
        try:
            with chronicle_capture(root):
                return compare(changes)
        finally:
            _BACKGROUND.reset(token)

    def compare(changes):
        # The builder inherits actual route/settings selection. No discovery or
        # remote account lookup is needed on the protected built-in path.
        from datetime import datetime, timezone

        clock_at = datetime.now(timezone.utc)

        def build():
            return build_context_fit_plan(env, memory, recipe, preferred_mode="nano", ctx=copy(root_ctx),
                tool_schemas=tool_schemas, tool_registry=tool_registry, count_only=True,
                route_resolver=lambda task, **_: _context_fit_route(task, allow_fetch=False))

        before = build()
        if not before.core:
            return None
        if not changes:
            return (client._local_publication_floor(before, caller_max, clock_at=clock_at),)
        def alternate():
            factory = alternate_view(root) if alternate_view is not None else None
            if factory is not None:
                with factory():
                    return build()
            return None

        plans, alternatives = [before], [alternate()]
        for proposal in (changes, *(intermediate(changes) if intermediate else ())):
            with ExitStack() as stack:
                _preview_sources(root, proposal, overlays or {}, stack)
                if any(change.subject == "chronicle" for change in proposal):
                    from ouroboros.chronicle_store import capture as chronicle_capture

                    stack.enter_context(chronicle_capture(root))  # the prospective branch and rollback read one transaction
                plans.append(build())
                alternatives.append(alternate())
        if after_alternatives := tuple(item for item in alternatives[1:] if item is not None):
            fallback = (alternatives[0] or before, *after_alternatives)
        else:
            fallback = ()
        if any(not plan.core or before.route_fp != plan.route_fp for plan in (*plans, *fallback)):
            return None
        reference = client._local_publication_floor(before, caller_max, clock_at=clock_at)
        return tuple(replace(reference if plan is before else client._local_publication_floor(
                            plan, caller_max, reference=reference), variant=variant)
                     for variant, branch in (("active", plans), ("alternate", fallback)) for plan in branch)

    return NanoPublication(root, revision, capture, lambda changes: _retain_proposal(author_ctx, changes))
