"""The route-owned work order of one acceptance panel's RETRIEVING rows.

A packet row reviews what the host assembled; a retrieving row reads the task's
own sources. Both receive the same task, criteria and output contract — what
differs is what each delivery can actually use, which is this module's whole
reason to change. Extracted from ``loop_acceptance_review`` (size paydown): the
panel's orchestration and this delivery contract have separate reasons to change
and separate readers.

Source first (#1329): the packet ceiling sizes what a PACKET row is handed; it
never decides whether a retrieving row may review. A retrieving row whose whole
first send would not hold the packet — or whose packet overflowed the shared
budget — gets a compact first send and the complete packet as ONE exact,
write-once source at an address this row's own reader resolves: the task
artifact store for a native episode, the retained file's absolute path for a
session using its ordinary read-only file tools. Both use existing host-owned
custody, without workspace copies or attachment capability gates. Missing or
changed source bytes are a typed refusal; an actual session read failure remains
an explicit evidence gap, never proof that the source does not exist.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
from typing import Any, Dict, List, Optional


_RETRIEVING_ACCESS_DISCLOSURE = (
    "Access outside the task workspace is not guaranteed on this delivery: a refused or "
    "failed read is absence of evidence, not absence of the artifact — report it as a gap "
    "you could not verify instead of inferring the artifact does not exist."
)

# One durable source id per panel: the store is write-once and content-addressed.
PACKET_SOURCE_ID = "acceptance-packet"
# Owner text, answer and refs ride a compact first send inline up to this size;
# beyond it the send names the key to read in the packet source instead.
_COMPACT_INLINE_TEXT_MAX = 16_000


def _retrieving_packet_projection(evidence: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.review_dispatch import retrieving_acceptance_packet

    return retrieving_acceptance_packet(evidence)


def _packet_source_body(request: Any) -> Dict[str, Any]:
    """Everything a packet row is handed, as one canonical object (the source)."""
    body = {"goal": request.goal, "scope": request.scope, "checklist": request.checklist,
            "subject": request.subject, "evidence_refs": request.evidence_refs,
            "evidence": request.evidence}
    if request.policy.get("review_source_closure"):
        body["retrieval_sources"] = request.policy["review_source_closure"]
    return body


def _section_index(body: Dict[str, Any]) -> str:
    """Key, size and digest of every section — a map of the source, not its content."""
    def row(name: str, value: Any) -> str:
        text = json.dumps(value, ensure_ascii=False, default=str)
        return f"- {name}: {len(text):,} chars, sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"

    lines = [row(key, value) for key, value in body.items() if key != "evidence"]
    # The full (arbitrarily wide) exhibit index lives in the exact packet.
    # Even keys are variable source material, so the first send lists fixed sections.
    lines.append(row("evidence", body.get("evidence")))
    return "\n".join(lines)


def _bounded(text: str, key: str) -> str:
    text = str(text or "")
    if len(text) <= _COMPACT_INLINE_TEXT_MAX:
        return text
    return (f"({len(text):,} chars — not repeated here: read key `{key}` of the packet source "
            "in full before judging.)")


class _PacketSource:
    """The panel's one stored packet source, written at most once, on first need."""

    def __init__(self, request: Any, data_root: pathlib.Path) -> None:
        self._request, self._data_root = request, data_root
        self._result: Optional[Dict[str, Any]] = None

    def get(self) -> Dict[str, Any]:
        if self._result is None:
            from ouroboros.artifacts import store_actor_source_bytes

            raw = json.dumps(_packet_source_body(self._request), ensure_ascii=False, indent=2,
                             default=str).encode("utf-8")
            try:
                ref = store_actor_source_bytes(
                    self._data_root, str(self._request.task_id or ""), category="context_checkpoints",
                    source_id=PACKET_SOURCE_ID, data=raw, extension="json")
                self._result = {"ref": ref, "chars": len(raw.decode("utf-8"))}
            except Exception as exc:  # maintenance fence, unwritable store, unusual task id
                self._result = {"error": f"{type(exc).__name__}: {exc}"}
        return self._result


def _source_path(root: Any, task_id: str, ref: Dict[str, Any]) -> str:
    from ouroboros.artifacts import task_artifact_dir_path

    return str((task_artifact_dir_path(root, task_id) / ref["path"]).resolve())


def _session_address(path: str) -> str:
    """An exact file read under existing authority, not an extra-root grant."""
    return (f'read the immutable packet file at absolute path {json.dumps(path, ensure_ascii=False)} '
            'with your ordinary read-only file tools, in ranges. The host authorizes reading this exact '
            'source even when it is outside your workspace; keep the task workspace as your root')


def _first_send_fits(request: Any, slot: Any, text: str, *, session: bool) -> tuple[bool, int, int]:
    """Measure the wrapper; session preparation is preliminary, dispatch rechecks."""
    from ouroboros.review_native_episode import (
        native_episode_transcript_bound,
        native_first_send_chars,
        native_landing_at,
    )
    from ouroboros.tools.review_brief_coupling import SESSION_INLINE_DIFF_CEILING_CHARS

    if session:
        from ouroboros.review_execution import (
            SessionInvocation, review_session_output_schema, session_route_for_review_slot,
        )
        from ouroboros.review_session_preparation import prepare_review_session_request, render_review_session_prompt
        route = session_route_for_review_slot(slot)
        invocation = SessionInvocation(task_id=request.task_id, surface=request.surface, slot_id=slot.slot_id,
                                       timeout_sec=604800, output_schema=review_session_output_schema(request.surface))
        wire = prepare_review_session_request(invocation, route,
            prompt=render_review_session_prompt(request, slot, text), root=request.session_root,
            thread_id="", schema_asked=True)
        measured = len(json.dumps(wire, ensure_ascii=False))
        return measured < SESSION_INLINE_DIFF_CEILING_CHARS, measured, SESSION_INLINE_DIFF_CEILING_CHARS
    measured = native_first_send_chars(
        str(request.session_root or ""), surface=request.surface, role_hint=slot.role_hint,
        slot_id=slot.slot_id, session_task=text, output_contract=request.policy["output_contract"],
        task_id=str(request.task_id or ""))
    ceiling = native_landing_at(native_episode_transcript_bound(request, slot))
    return measured < ceiling, measured, ceiling


def _measure_delivery(request: Any, slot: Any, text: str, *, session: bool,
                      delivery: dict) -> Optional[tuple[bool, int, int]]:
    """Keep a row's preparation failure distinct from a measured size refusal."""
    try:
        return _first_send_fits(request, slot, text, session=session)
    except Exception as exc:
        code = str(getattr(exc, "code", "") or "review_source_preparation_failed")
        delivery.update(status="unavailable", reason=f"{code}: {exc}",
                        preparation_error={"code": code, "type": type(exc).__name__, "message": str(exc)})
        return None


_SESSION_INTRO = "You review as a read-only agent session in the task workspace."
_NATIVE_INTRO = ("You review as a bounded read-only native inspection episode; the host data root at the "
                 "pointers is readable.")


def _compact_work_order(request: Any, slot: Any, *, session: bool, pointer_rows: tuple, vocabulary: str,
                        why: str, stored: Dict[str, Any], address: str) -> str:
    """The first send of a paged row: task, criteria, answer and a map of the source."""
    from ouroboros.review_execution import _render_prompt_parts

    ref = stored["ref"]
    compact = dataclasses.replace(request, evidence={}, goal="Read `goal` in the exact packet source.",
                                  scope="Read `scope` in the exact packet source.",
                                  checklist="Read `checklist` in the exact packet source.",
                                  subject="", evidence_refs=[])
    _stable, task_stable, _dynamic = _render_prompt_parts(compact, slot)
    refs = json.dumps(request.evidence_refs, ensure_ascii=False, indent=2, default=str)
    pointers = "\n".join(pointer_rows)
    if request.policy.get("review_source_closure"):
        pointers = _bounded(pointers, "retrieval_sources")
    return "\n\n".join((
        _SESSION_INTRO + " " + _RETRIEVING_ACCESS_DISCLOSURE if session else _NATIVE_INTRO,
        "RETRIEVAL POINTERS (absolute paths):\n" + pointers,
        (f"COMPLETE EVIDENCE PACKET — one exact source instead of inline ({why}). It is complete and "
         f"untruncated: {stored['chars']:,} chars, sha256 {ref['sha256']}; {address}. It holds the owner's "
         "full request, the delivered answer, the evidence refs and every packet exhibit. MANDATORY: read it "
         "before judging, in as many ranges as it takes. If you cannot open it, answer DEGRADED and name the "
         "gap; never PASS a criterion you could not verify."),
        vocabulary,
        "Canonical source identity (retained by the host): " + json.dumps(ref, ensure_ascii=False),
        task_stable.rstrip(),
        "Subject (the delivered answer):\n" + _bounded(request.subject, "subject"),
        "Evidence refs:\n" + _bounded(refs, "evidence_refs"),
        "Packet source index (key, size, digest — the content is in the source):\n"
        + _section_index(_packet_source_body(request)),
    ))


def acceptance_retrieving_work_order(
    request: Any, slots: List[Any], *, session_root: str, data_root: pathlib.Path,
) -> None:
    """Attach the route-owned work order of ONE acceptance panel's retrieving
    rows (owner R1/R4/R5/R15, 2026-09-01) to ``request`` in place.

    Every retrieving row receives the same task, criteria and output contract
    as the packet rows — rendered by the same `_render_prompt_parts` — plus
    absolute retrieval pointers. A SESSION row gets the FULL packet (its run is
    unobserved by the host, so the packet is its only attested view) and the
    access disclosure; a NATIVE row gets the packet without its freely
    degradable tail and the real data root (R5), because its episode reads
    task results and artifacts itself. When that whole first send does not
    fit the row, or the packet overflowed, the row gets the compact send and
    the exact packet source instead (module doc); ``slot_source_delivery``
    records which. The FULL packet stays on ``request.evidence``: evidence_refs
    resolve against it, never against a rendered projection."""
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.outcome_receipt_store import verification_receipts_path
    from ouroboros.review_execution import ReviewRouteKind, _render_prompt_parts, review_output_contract

    request.session_root = session_root
    request.policy["output_contract"] = review_output_contract(request)
    request.policy["native_data_root"] = str(data_root)
    task_id = str(request.task_id or "")
    root = pathlib.Path(data_root)
    try:
        artifacts_dir, receipts = task_artifact_dir_path(root, task_id), verification_receipts_path(root, task_id)
    except Exception:  # an unusual task id: name the canonical layout instead of refusing the work order
        artifacts_dir = root / "task_results" / "artifacts" / task_id
        receipts = artifacts_dir / "verification_receipts.jsonl"
    pointer_rows = (
        f"- task workspace — the active tree the task worked in (your root): {session_root}",
        f"- task result record (contract, status, children): {root / 'task_results' / (task_id + '.json')}",
        f"- task artifacts named by the packet's `artifacts` manifest: {artifacts_dir}/",
        f"- host-attested verification receipts: {receipts}",
        f"- tool trajectory log (rows with task_id={task_id}; one call's start / settlement / wait-ended rows "
        f"share one invocation_id): {root / 'logs' / 'tools.jsonl'}",
    )
    closure = request.policy.get('review_source_closure')
    if closure:
        workspace_label = ('historical retained-source view; original workspace bytes may be unavailable'
                           if request.policy.get('historical_acceptance') else 'the active tree the task worked in (your root)')
        pointer_rows = (
            f'- task workspace — {workspace_label}: {session_root}',
            f'- artifact_store for task {task_id}: {artifacts_dir}/',
            *(f"- {row['name']}: {row.get('retained_path') or row['status']}" for row in closure['sources']),
            *(f"- source owned by task {row['owner_task_id']}: {row['retained_path']}"
              for row in closure.get('refmap', [])),
        )
    pointers = "\n".join((
        "RETRIEVAL POINTERS (immutable named snapshots; original paths are provenance):" if closure else
        "RETRIEVAL POINTERS (absolute paths; the packet below is the host's attested projection of these sources):",
        *pointer_rows,
    ))
    vocabulary = ("Every evidence_ref must be an EXACT member of the packet's host-attested exhibit vocabulary; "
                  "the FULL packet is the host's resolution authority whatever you read at the pointers.")
    overflow = bool((request.evidence if isinstance(request.evidence, dict) else {}).get("__immutable_core_overflow__"))
    source = _PacketSource(request, root)
    stored = source.get()  # canonical identity is checkpointed even for inline delivery
    native_packet: Optional[Dict[str, Any]] = None
    for slot in slots:
        session = getattr(slot, "route", None) is ReviewRouteKind.AGENT_SESSION
        if session:
            preamble = (
                _SESSION_INTRO + " The host's FULL evidence packet follows; verify its claims against the "
                "sources at the pointers with your own tools. " + _RETRIEVING_ACCESS_DISCLOSURE
            )
            packet = request.evidence
        else:
            preamble = (
                _NATIVE_INTRO + " The evidence packet follows WITHOUT its tool-trajectory rows and "
                "artifact previews — read those sources yourself at the pointers."
            )
            if native_packet is None:
                native_packet = _retrieving_packet_projection(request.evidence)
            packet = native_packet
        _stable, task_stable, dynamic = _render_prompt_parts(dataclasses.replace(request, evidence=packet), slot)
        slot_line = f"Slot: {slot.slot_id}"
        dynamic = dynamic.rstrip()  # the renderer's tail may grow a newline; the executor labels the slot itself
        if dynamic.endswith(slot_line):
            dynamic = dynamic[: -len(slot_line)].rstrip()
        inline = "\n\n".join((preamble, pointers, vocabulary, task_stable.rstrip() + "\n\n" + dynamic))
        request.slot_session_tasks[slot.slot_id] = inline
        delivery: Dict[str, Any] = {"status": "inline",
                                    "source": dict(stored.get("ref") or {}), "source_root": str(root),
                                    "preparation_measurement_basis": "synthetic_session_invocation_dispatch_rechecks" if session else "native_first_send_wrapper",
                                    "reader": "filesystem" if session else "artifact_store"}
        request.slot_source_delivery[slot.slot_id] = delivery
        measured_fit = _measure_delivery(request, slot, inline, session=session, delivery=delivery)
        if measured_fit is None:
            continue
        fits, measured, ceiling = measured_fit
        delivery.update(first_send_chars=measured, first_send_ceiling=ceiling)
        if stored.get("error"):
            request.slot_source_delivery[slot.slot_id] = {**delivery, "status": "unavailable", "reason": stored["error"]}
            continue
        if fits and not overflow:
            request.slot_source_delivery[slot.slot_id] = delivery
            continue
        why = ("the packet overflowed the shared evidence budget" if overflow
               else "the whole first send does not fit this row")
        address: Dict[str, Any] = {"error": stored.get("error", "")}
        if not stored.get("error"):
            address = ({"address": _session_address(_source_path(root, task_id, stored["ref"])),
                        "source_path": _source_path(root, task_id, stored["ref"])} if session else {
                "address": ('read it in ranges with read_file(root="artifact_store", path="'
                            f'{stored["ref"]["path"]}", start_line=A, max_lines=N)')})
        if address.get("error"):
            reason = f"{why}; no exact packet source this row can open: {address['error']}"
            request.slot_source_delivery[slot.slot_id] = {**delivery, "status": "unavailable", "reason": reason}
            continue
        compact = _compact_work_order(request, slot, session=session, pointer_rows=pointer_rows,
                                      vocabulary=vocabulary, why=why, stored=stored, address=address["address"])
        request.slot_session_tasks[slot.slot_id] = compact
        measured_fit = _measure_delivery(request, slot, compact, session=session, delivery=delivery)
        if measured_fit is None:
            continue
        compact_fits, compact_chars, _ceiling = measured_fit
        ref = stored["ref"]
        request.slot_source_delivery[slot.slot_id] = {
            **delivery, "status": "paged" if compact_fits else "unavailable",
            "reason": why if compact_fits else "complete compact first send exceeds this row's bound",
            "first_send_chars": compact_chars, "inline_first_send_chars": measured, "source": dict(ref),
            **({"source_path": address["source_path"]} if session else {})}


def retain_review_source(request: Any, slot_id: str, custody_root: pathlib.Path) -> None:
    """Retain the packet without relocating the request's external source readers.

    The coordinator calls this before it constructs/persists the executor prompt.
    Packet durability and reader authority are separate: retaining one JSON file
    does not close task results, receipts, trajectory or absolute work orders.
    Only the landed generic request-closure owner may rebind those readers.
    Operation/drive lifetime stays with the existing custody owner.
    """
    from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes
    from ouroboros.review_execution import ReviewRouteUnavailable

    delivery = request.slot_source_delivery.get(slot_id)
    if not delivery or delivery.get("status") == "unavailable":
        return
    try:
        source_root = delivery.get("custody_root") or delivery.get("source_root") or request.policy["native_data_root"]
        source_ref = delivery.get("custody_source") or delivery["source"]
        raw = read_actor_source_bytes(source_root, request.task_id, source_ref)
        ref = store_actor_source_bytes(custody_root, request.task_id, category="context_checkpoints",
                                      source_id=PACKET_SOURCE_ID, data=raw, extension="json")
        if ref != delivery["source"]:
            raise ValueError("retained packet identity differs from the prepared source")
        read_actor_source_bytes(custody_root, request.task_id, ref)
        path = _source_path(custody_root, request.task_id, ref)
        if delivery.get("reader") == "filesystem" and delivery.get("status") == "paged":
            task = request.slot_session_tasks[slot_id]
            old_address = _session_address(delivery["source_path"])
            if old_address not in task:
                raise ValueError("session work order has no exact source address")
            request.slot_session_tasks[slot_id] = task.replace(old_address, _session_address(path), 1)
        delivery.update(custody_source=ref, custody_root=str(custody_root), source_path=path)
        # SHARED policy still names the source plane used by the work order's
        # pointers, including inline rows. A retained packet is not proof that
        # canonical custody also contains those external inputs.
        reader_root = request.policy.get("native_data_root") or delivery.get("source_root")
        closure = request.policy.get("review_source_closure") or {}
        closed = closure.get("task_id") == request.task_id and closure.get("read_root") == str(reader_root)
        delivery.update(reader_root=str(reader_root or ""), external_ref_closure="retained" if closed else "not_established")
        if not reader_root or not pathlib.Path(reader_root).is_dir():
            delivery["reader_source_gap"] = "original_reader_root_unavailable"
            raise ValueError("original_reader_root_unavailable: packet custody does not retain external sources")
        if delivery.get("reader") == "artifact_store" and delivery.get("status") == "paged":
            # Native read_file resolves the packet at its unchanged reader root;
            # a canonical copy cannot silently repair an unavailable original.
            try:
                read_actor_source_bytes(reader_root, request.task_id, ref)
            except Exception:
                delivery["reader_source_gap"] = "native_packet_at_reader_root_unavailable"
                raise
    except Exception as exc:
        raise ReviewRouteUnavailable(f"degraded_source_unreachable: canonical source custody failed: {exc}",
                                     code="degraded_source_unreachable") from exc


def prepare_session_source(delivery: Optional[Dict[str, Any]], *, task_id: str) -> str:
    """Verify the retained source and return its ordinary filesystem read address.

    No engine calls, uploads, capability inference or filesystem grants. The
    selected session's actual reads may still fail; coverage is diagnostic and
    the reviewer must disclose that gap. Host byte availability is not a receipt
    proving that a model read or understood the packet.
    """
    if not delivery:
        return ""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.review_execution import ReviewRouteUnavailable

    try:
        ref, root = delivery["custody_source"], delivery["custody_root"]
        if ref != delivery["source"]:
            raise ValueError("retained packet identity differs from the prepared source")
        read_actor_source_bytes(root, task_id, ref)
        path = _source_path(root, task_id, ref)
        if path != delivery["source_path"]:
            raise ValueError("session address differs from the retained source")
        return path
    except Exception as exc:
        raise ReviewRouteUnavailable(
            f"degraded_source_unreachable: Exact session source unavailable: {type(exc).__name__}: {exc}",
            code="degraded_source_unreachable") from exc
