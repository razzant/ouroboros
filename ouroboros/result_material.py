"""What a task produced, for its result review (``review_change(subject="result")``).

Production, the completeness of its material and the review subject are different facts (owner, 8 Nov
2026). A host-observed write, a producer's effect receipt or a registered deliverable is production even
when its material cannot be captured; the gap is then named, never filled in. Admission, an output declared
but never created, input copies, checkpoints and the expected result a task was given are intent, not
production. The view reads producers that already exist — the task's registered artifacts, its changes in
the admitted folder against that folder's own baseline and the commits its attributed epochs record
(``mutation_attribution``), the native writes the loop trace observed (``outcomes.reviewable_effect_projection``), at
the targets each writer's typed receipt says its binding resolved when it wrote, and extension effect receipts
(``observed_effect``) — and adds no watcher, ledger or classifier: a shell
effect nothing observed stays unknown. Every listed file's exact bytes stay readable at a named copy (an
immutable registered artifact, else one copy by digest in the task's source store), so a reviewer reads the
version the subject names, not a live path that may have changed since; a packet seat reads bounded
excerpts of the text ones inline. A produced symbolic link is listed as itself, by its own path and its exact
link text, never as its target's path or bytes: what it points at was not produced by making the link.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import os
import pathlib
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from ouroboros import work_domain

RESULT_WORDS_SOURCE = "the produced files named above by path and content hash"
_EXCERPT_FLOOR = 1200  # characters one text file may take of the excerpt budget before the rest is divided
# The excerpts are a bounded look for a packet seat, never the whole material inline (FIX1, 4 Nov 2026): at most this
# many characters, within what the words section leaves of the panel's share.
_EXCERPT_TOTAL = 16_000
_WRITE_ROOTS = ("active_workspace", "system_repo")
_CAPTURED_ROOTS = ("artifact_store", "user_files", "deliverables")  # their writes are registered artifacts


def _output_path(path: Any) -> str:
    """The listed output's own path: its folder resolved (a work folder keeps its spelling, the container resolves
    it), its last part never followed (a link is an output itself)."""
    path = pathlib.Path(path)
    return str(work_domain.settle(path) if path.name in {"", ".", ".."} else work_domain.settle(path.parent) / path.name)


def _file_identity(path: pathlib.Path) -> Dict[str, Any]:
    """Size and sha256 of one listed file, read stably (``artifacts.stream_artifact_file``; a work file through the
    confined intake, never a service open of its pathname), else why not; a symbolic link's are of its exact link
    text (``readlink``), shown whole as ``link``, broken or relative alike. A work domain that could not be asked is
    unreadable, no evidence that the file is gone."""
    from ouroboros.artifacts import stream_artifact_file

    try:
        work = work_domain.is_work(path)
        if work_domain.kind(path) == "link" if work else path.is_symlink():
            text = os.fsencode(work_domain.readlink(path)) if work else os.readlink(os.fsencode(path))
            return {"size": len(text), "sha256": hashlib.sha256(text).hexdigest(),
                    "link": json.dumps(os.fsdecode(text), ensure_ascii=False)}
        return work_domain.measure(path) if work else stream_artifact_file(path)
    except (OSError, ValueError) as exc:
        gone = isinstance(exc, FileNotFoundError) and not getattr(exc, "process_not_started", False)
        return {"size": 0, "sha256": "deleted" if gone else f"unreadable ({type(exc).__name__})"}


def _same_file_kind(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    """Only deduplicate identities when both observations are links or both are regular-file bytes."""
    return ("link" in left) == ("link" in right)


def _retained_copy(root: Any, task_id: str, item: Dict[str, Any], registration: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Where the listed file's exact bytes stay readable: an immutable registered artifact is its own copy; any
    other file is copied once by digest into this task's source store (``artifacts.copy_artifact_file``: a
    block-sharing clone where the filesystem gives one, else a verified stream, at any size), so a later write to
    the live path never changes what its subject names; else why no copy exists. Its cost: on APFS, btrfs or XFS a
    clone shares the original's blocks; elsewhere (ext4, a Docker overlay, Windows) each distinct content takes its
    full size once more. The copy belongs to the task's records under ``source_handles/tool_results``: no age bound,
    ``record_compression`` shrinks it where the filesystem compresses, and nothing removes it apart from them."""
    from ouroboros.artifacts import actor_source_reference, copy_artifact_file, stream_artifact_file, task_artifact_dir_path

    sha = str(item.get("sha256") or "")
    if sha == "deleted" or sha.startswith("unreadable"):
        return {"retained_gap": "the file is gone" if sha == "deleted" else sha}
    identity = {"size": item["size"], "sha256": sha}
    if registration and registration.get("immutable") and registration.get("sha256") == sha:
        return {"retained": str(item["path"])}
    try:
        ref = actor_source_reference(category="tool_results", source_id=f"result-material-{sha[:16]}", data=b"",
                                     extension=pathlib.Path(item["path"]).suffix.lstrip(".") or "bin", measured=identity)
        target = task_artifact_dir_path(root, task_id, create=True).joinpath(*pathlib.PurePosixPath(ref["path"]).parts)
        try:  # write-once by digest: an intact copy of these bytes IS this one
            present = target.is_file() and not target.is_symlink() and bool(stream_artifact_file(target, expected=identity))
        except OSError:
            present = False
        if not present and work_domain.is_work(item["path"]):  # its bytes received from the work, then verified
            with tempfile.TemporaryFile() as spool:
                work_domain.measure(item["path"], spool)
                spool.flush()
                copy_artifact_file(spool, target, expected=identity, mode=0o444)
        elif not present:
            copy_artifact_file(pathlib.Path(item["path"]), target, expected=identity, mode=0o444)
    except Exception as exc:  # changed while it was copied, a full disk, a closed publication: a stated gap
        return {"retained_gap": f"its copy could not be retained ({type(exc).__name__}: {exc})"}
    return {"retained": str(target)}


def _written_paths(ctx: Any, root: Any, task_id: str, effect: Dict[str, Any]) -> Tuple[List[pathlib.Path], str]:
    """The files one observed native write changed, exactly as its writer's binding resolved them when it wrote (its
    typed receipt: ``changed_paths`` under ``resolved_root``, ``git_repo_edit._with_path_facts``), or why they are not
    listed. Its arguments are never resolved again here: a link retargeted since the write would name a file the call
    never touched (Astra, 10 Nov 2026); a target its writer found already holding those bytes is not in the receipt."""
    from ouroboros.outcomes import _ROOT_WRITE_TOOLS

    tool = str(effect.get("tool") or "")
    if tool not in _ROOT_WRITE_TOOLS:
        return [], ""  # a commit, an integration or a registered output: read by attribution and the artifacts
    args = effect.get("args") if isinstance(effect.get("args"), dict) else {}
    where = str(args.get("root") or "active_workspace")
    if where in _CAPTURED_ROOTS:
        return [], ""
    calls = (getattr(ctx, "_execution_trace", None) or {}).get("tool_calls") or []
    call = calls[effect["index"]] if 0 <= int(effect.get("index", -1)) < len(calls) else {}
    meta = call.get("tool_result_meta") if isinstance(call, dict) and isinstance(call.get("tool_result_meta"), dict) else {}
    if where in _WRITE_ROOTS and isinstance(meta.get("changed_paths"), list) and meta.get("resolved_root"):
        omitted = meta.get("changed_paths_omitted")
        return [pathlib.Path(str(meta["resolved_root"])) / str(rel) for rel in dict.fromkeys(meta["changed_paths"])], (
            f"{tool} changed {omitted} more path(s) than its typed receipt could name; those are not listed here. "
            "Registered captures and attributed Git changes are listed separately." if omitted else "")
    if isinstance(call, dict) and call:
        from ouroboros.artifacts import materialize_tool_args_source

        full, complete, _gap = materialize_tool_args_source(root, call)
        args = full if complete and isinstance(full, dict) else args
    named = [str(args["path"])] if isinstance(args.get("path"), str) and args["path"] else []
    for entries in (args.get("files"), args.get("edits")):
        named += [str(item["path"]) for item in entries if isinstance(item, dict) and item.get("path")] \
            if isinstance(entries, list) else []
    if tool == "apply_patch" and isinstance(args.get("patch"), str):
        from ouroboros.tools.edit_ops import normalize_patch_paths

        normalize_patch_paths(args["patch"], lambda path: named.append(path) or path)
    named_text = ", ".join(dict.fromkeys(named)) or "paths its retained arguments do not name"
    if where in _WRITE_ROOTS:  # a record without its writer's receipt: what it resolved then is unknown now
        return [], (f"{tool} wrote {named_text} under {where}, but its call record carries no receipt of the targets "
                    "it resolved; its arguments are not resolved again now, so those bytes are not listed here")
    return [], f"{tool} wrote {named_text} under {where}; those bytes are not listed here"


def _basis_note(ctx: Any, snapshot: Dict[str, Any]) -> str:
    """The admitted folder's work-start HEAD against the HEAD its admission recorded, which the final patch
    capture uses (``workspace_patch_capture``): a difference is said, never silently reconciled."""
    metadata = getattr(ctx, "task_metadata", None)
    preflight = metadata.get("workspace_preflight") if isinstance(metadata, dict) else None
    git = preflight.get("git") if isinstance(preflight, dict) and isinstance(preflight.get("git"), dict) else {}
    admitted, started = str(git.get("head") or ""), str(snapshot.get("base_commit") or "")
    if not admitted or not started or admitted == started:
        return ""
    return (f"this folder's HEAD at work start ({started[:12]}) differs from the HEAD its admission recorded "
            f"({admitted[:12]}), which its final patch capture uses: the two may cover different intervals")


def produced_material(ctx: Any) -> Dict[str, Any]:
    """What this task produced: its registered artifacts, its changes in the admitted folder against that
    folder's own baseline and its own attributed commits, the native writes the trace observed, and the
    effects an extension tool declared. ``observed`` lists every host-observed production (the Required floor
    reads it even when material capture failed). A capture that fails is named as a gap, never filled in."""
    out: Dict[str, Any] = {"files": [], "effects": [], "issues": [], "workspace": "", "observed": []}
    root, task_id = getattr(ctx, "drive_root", None), str(getattr(ctx, "task_id", "") or "")
    if root is None or not task_id:
        return out
    registrations: Dict[str, Dict[str, Any]] = {}
    covered: set = set()

    def add(path: pathlib.Path, kind: str) -> Optional[Dict[str, Any]]:
        """List one output once; ``None`` when it is already listed or a healthy capture stands for it."""
        key = _output_path(path)
        if key in covered:
            return None
        covered.add(key)
        out["files"].append({"path": key, "kind": kind, **_file_identity(pathlib.Path(key))})
        return out["files"][-1]

    try:
        from ouroboros.artifacts import collect_task_artifact_records, registered_task_artifact

        for row in collect_task_artifact_records(root, task_id, measure=False, strict=True, require_registered=True):
            name = str(row.get("relpath") or row.get("name") or "")
            # A row is a file present in the store, so a registered empty file is production too; an output
            # never created has no row and no registration.
            registration = registered_task_artifact(root, task_id, name) if name else None
            if not registration:
                continue
            item = add(pathlib.Path(str(row["path"])), "artifact")
            if item is None:
                continue
            registrations[item["path"]] = registration
            source = _output_path(str(registration.get("source_path") or ""))
            # The capture stands for its written original only while that original still holds the captured
            # bytes: a later rewrite the capture did not follow stays its own production, listed below.
            if registration.get("source_path"):
                source_identity = _file_identity(pathlib.Path(source))
                if (_same_file_kind(source_identity, item)
                        and source_identity.get("sha256") == item.get("sha256")):
                    covered.add(source)
    except Exception as exc:
        out["issues"].append(f"task artifacts could not be listed ({type(exc).__name__}: {exc})")
    try:
        from ouroboros.mutation_attribution import attributed_commit_paths, attributed_git_candidates

        getter = getattr(ctx, "active_repo_dir", None)
        repo = getter() if callable(getter) else getattr(ctx, "repo_dir", None)
        if repo:
            results_root = getattr(ctx, "budget_drive_root", None) or root
            snapshot = attributed_git_candidates(results_root, task_id, repo)
            candidates = [str(c.get("path") if isinstance(c, dict) else c) for c in snapshot.get("candidates") or []]
            for relative in candidates:
                add(pathlib.Path(str(repo)) / relative, "workspace change")
            try:
                for relative in attributed_commit_paths(results_root, task_id, repo):
                    add(pathlib.Path(str(repo)) / relative, "committed change")
            except Exception as exc:
                out["issues"].append(f"this task's own commits could not be read ({type(exc).__name__}: {exc})")
            if candidates:
                out["workspace"] = str(work_domain.settle(str(repo)))
                from ouroboros.review_words import ReviewWords, section_budget

                try:
                    out["words_section"] = _workspace_words(repo, snapshot, candidates, _result_words_budget())
                except Exception as exc:
                    out["words_section"] = ReviewWords(gaps=(
                        f"workspace words ({type(exc).__name__}: {exc})",)).render(section_budget(), RESULT_WORDS_SOURCE)
            blockers = [str(item) for item in snapshot.get("blockers") or []]
            if blockers and blockers != ["baseline_missing"]:
                out["issues"].append("workspace changes not attributed in full: " + ", ".join(blockers))
            if note := _basis_note(ctx, snapshot):
                out["issues"].append(note)
    except Exception as exc:
        out["issues"].append(f"workspace changes could not be attributed ({type(exc).__name__}: {exc})")
    from ouroboros.outcomes import reviewable_effect_projection

    for effect in reviewable_effect_projection(getattr(ctx, "_execution_trace", None) or {}, observed_only=True):
        out["observed"].append({"tool": effect["tool"], "index": effect["index"]})
        try:
            paths, gap = _written_paths(ctx, root, task_id, effect)
        except Exception as exc:
            paths, gap = [], f"{effect['tool']} wrote paths that could not be read back ({type(exc).__name__}: {exc})"
        for path in paths:
            add(path, "written file")
        if gap and gap not in out["issues"]:
            out["issues"].append(gap)
    out["effects"] = _declared_effects(ctx, root, task_id, out["issues"])
    for item in out["files"]:  # a link's whole content is its link text, named in its line
        item.update({} if "link" in item else _retained_copy(root, task_id, item, registrations.get(item["path"])))
    if any(item.get("retained") for item in out["files"]):
        budget = min(_EXCERPT_TOTAL, _result_words_budget() - len(out.get("words_section") or ""))
        if contents := _contents_section(out["files"], budget):
            out["contents_section"] = contents
    return out


def answer_text(*, reply: str, produced: Dict[str, Any], verification: str) -> str:
    """The answer half of a result subject (the reply, what the task produced, the verification): its digest
    is the subject's ``tree_sha``, the answer's own identity apart from what was asked."""
    files = [f"- {item['path']} ({item['kind']}; sha256 {item['sha256']}; {item['size']} bytes; "
             + (f"symbolic link, its exact link text {item['link']})" if "link" in item else
                f"exact copy: {item['retained']})" if item.get("retained") else
                f"no exact copy: {item.get('retained_gap') or 'not retained'})")
             for item in produced.get("files") or []]
    effects = [f"- effect declared by {item['tool']}: {item['receipt']}"
               + (f"\n  host notes on this call: {json.dumps(item['host_notes'], ensure_ascii=False)}"
                  if item.get("host_notes") else "")
               for item in produced.get("effects") or []]
    gaps = [f"- not captured: {issue}" for issue in produced.get("issues") or []]
    return "\n".join([
        "## The reply", "", reply.strip() or "(no reply text)", "",
        "## What the task produced", "",
        *(["Effect receipts are the tool producer's reports of execution, not proof of a UI change or goal success.", ""]
          if effects else []),
        *(files + effects + gaps or ["(nothing produced beyond the reply)"]), "",
        *([produced["words_section"], ""] if produced.get("words_section") else []),
        *([produced["contents_section"], ""] if produced.get("contents_section") else []),
        "## Host-attested verification", "", verification, "",
    ])


def _contents_section(files: List[Dict[str, Any]], budget: int) -> str:
    """Bounded exact excerpts of the listed text files, read from their retained copies, for a seat that reads only
    the packet: each file's first complete lines within its share, else (no line break in what fits, as in one-line
    JSON, or a first line longer than the share) its first characters cut on a character boundary, with what is
    omitted said. Each read is bounded by its file's share, and the files left once the budget is spent are named
    without being read. A binary file and one without a copy are only named: their copies (or paths) above are
    where they are read."""
    candidates = [item for item in files if item.get("retained")]
    blocks, unread = ["## Excerpts of the produced files", "",
                      "Exact bytes from each file's retained copy, its first lines within this packet's budget."], []
    remaining = max(0, int(budget))
    for position, item in enumerate(candidates):
        share = min(remaining, max(_EXCERPT_FLOOR, remaining // (len(candidates) - position)))
        if share <= 0:
            unread.append(str(item["path"]))
            continue
        try:
            with open(item["retained"], "rb") as handle:
                head = handle.read(share * 4 + 4)  # a character is at most 4 UTF-8 bytes
        except OSError:
            continue
        if b"\0" in head:
            continue
        whole = len(head) >= int(item.get("size") or 0)
        try:  # exact bytes or nothing: text that is not UTF-8 is named, never shown with bytes dropped; a character
            text = codecs.getincrementaldecoder("utf-8")().decode(head, final=whole)  # cut by the read stays unread
        except UnicodeDecodeError:
            blocks += ["", f"### {item['path']} (not shown: not UTF-8 text; it is in its retained copy)"]
            continue
        lines = text.splitlines(keepends=True)
        if not whole and lines and lines[-1].splitlines()[0] == lines[-1]:
            lines.pop()  # the read ended inside this line
        shown, used = [], 0
        for line in lines:
            if used + len(line) > share:
                break
            shown.append(line)
            used += len(line)
        excerpt = "".join(shown) or text[:share]
        remaining -= len(excerpt)
        fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", excerpt)), default=0))
        omitted = ("the whole file shown" if whole and excerpt == text else
                   f"its first {len(shown)} lines shown; the rest is in its retained copy" if shown else
                   f"its first {len(excerpt)} characters shown; the rest is in its retained copy")
        blocks += ["", f"### {item['path']} ({omitted})"] + ([fence, excerpt.rstrip("\n"), fence] if excerpt else [])
    if unread:
        blocks += ["", "Not shown: this packet's excerpt budget is spent; each is in its retained copy above: "
                   + ", ".join(unread)]
    return "\n".join(blocks) if len(blocks) > 3 else ""


def _result_words_budget() -> int:
    """The words share a change packet takes, sized against this panel's seats: the one subject reaches every seat,
    inline for a packet seat and in a retrieving seat's brief."""
    from ouroboros.review_records import ReviewSlot
    from ouroboros.review_words import section_budget
    from ouroboros.reviewer_slot_config import commit_triad_delivery
    from ouroboros.tools.review import _triad_governance_usable_window
    from ouroboros.tools.review_admission import seat_vectors

    try:
        plan = seat_vectors(commit_triad_delivery())
    except ValueError:  # the wave refuses this pool itself; the subject keeps the global bound
        return section_budget()
    slots = [ReviewSlot(slot_id=plan["slot_ids"][i], model=plan["models"][i],
                        session_profile=plan["session_profiles"][i], use_local=plan["use_local"][i])
             for i in range(len(plan.get("models") or []))]
    if not slots:
        return section_budget()
    return section_budget(_triad_governance_usable_window([slot.model for slot in slots], slots))


def _workspace_words(repo: Any, snapshot: dict, candidates: list[str], budget: int) -> str:
    """Extract bounded words from attributed changes; the diff is transient, never retained."""
    from ouroboros.review_words import has_words, words_for_diff
    from ouroboros.tools.review_subject import _HARDENED_DIFF_FLAGS, _decode_diff, _git_bytes

    base = snapshot.get("base_tree") or snapshot.get("base_commit")
    if not base:
        raise ValueError("workspace diff has no recorded baseline")
    paths = [f":(literal){path}" for path in candidates if has_words(path)]
    if not paths:
        return ""
    rc, raw, error = _git_bytes(repo, ["diff", *_HARDENED_DIFF_FLAGS, "--full-index", str(base), "--", *paths])
    if rc:
        raise ValueError(f"workspace diff capture failed: {error}")
    rc, untracked, error = _git_bytes(repo, ["ls-files", "--others", "--exclude-standard", "-z", "--", *paths])
    if rc:
        raise ValueError(f"workspace new files could not be listed: {error}")
    for path in untracked.decode("utf-8").split("\0"):
        if not path:
            continue
        rc, added, error = _git_bytes(repo, ["diff", "--no-index", *_HARDENED_DIFF_FLAGS, "--full-index", "--", "/dev/null", path])
        if rc not in (0, 1):
            raise ValueError(f"workspace new file diff failed: {error}")
        raw += added
    diff = _decode_diff(raw)
    return words_for_diff(repo, diff).render(budget, RESULT_WORDS_SOURCE)


def _declared_effects(ctx: Any, root: Any, task_id: str, issues: List[str]) -> List[Dict[str, Any]]:
    """Extension tools' own execution receipts (``PluginAPI.register_tool``): read exactly as the producer
    wrote them, never inferred from a method or backend name. A call whose producer reported its own failure
    (``TOOL_REPORTED_FAILURE``) may still declare what ran before it failed; a host-owned outcome (a timeout,
    a refusal, a raised handler) declares nothing. Host notes trail the payload the model saw, so a noted
    call is read from its retained producer payload, verified, and its notes ride beside the receipt."""
    from ouroboros.artifacts import materialize_tool_result_source, read_actor_source_bytes
    from ouroboros.extension_surface_names import parse_extension_surface_name
    from ouroboros.store import StoreError

    effects: List[Dict[str, Any]] = []
    trace = getattr(ctx, "_execution_trace", None) or {}
    for call in trace.get("tool_calls") or []:
        if (not isinstance(call, dict) or not parse_extension_surface_name(str(call.get("tool") or ""))
                or (call.get("is_error") and call.get("tool_result_code") != "TOOL_REPORTED_FAILURE")):
            continue
        tool = str(call.get("tool") or "")
        try:
            if "producer_source_ref" in call:
                if not call["producer_source_ref"]:
                    issues.append(f"effect receipt of {tool} unavailable (its producer payload was not retained)")
                    continue
                text = read_actor_source_bytes(root, task_id, call["producer_source_ref"]).decode("utf-8")
            else:
                text, complete, gap = materialize_tool_result_source(root, task_id, call)
                if not complete:
                    issues.append(f"effect receipt of {tool} unavailable ({gap})")
                    continue
            payload = json.loads(text) if isinstance(text, str) else {}
        except (OSError, ValueError, TypeError, StoreError) as exc:
            issues.append(f"effect receipt of {tool} unreadable ({type(exc).__name__}: {exc})")
            continue
        if isinstance(payload, dict) and payload.get("observed_effect") is True:
            notes = [str(note) for note in call.get("host_annotations") or []]
            effects.append({"tool": tool, "receipt": text, **({"host_notes": notes} if notes else {})})
    return effects
