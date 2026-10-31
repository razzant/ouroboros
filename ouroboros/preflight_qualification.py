"""Exact-commit test evidence for adoption and evolution publication.

Neither consumer runs tests. The battery runs where a long wait belongs (a
reviewed commit's post-commit gate, evolution's forced run); these consumers
accept only its evidence, for exactly the commit in question. Publication of the
serving line needs full evidence. Local adoption (owner choice A, 21.10.2026)
also accepts a selected run of the commit when it was selected against exactly
the serving HEAD it adopts onto and carried every widening its commits declared.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

_HOW = ("a run that also carried uncommitted or untracked files does not count; commit from a clean checkout so "
        "the post-commit run tests exactly this tree (selected over the whole candidate delta, or full_tests=true "
        "for verifier, cutover or broad changes), then request the restart again")


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=60)


def _commit_tree(repo, commit) -> str:
    tree = _git(repo, "rev-parse", "--verify", f"{commit}^{{tree}}")
    return "" if tree.returncode else tree.stdout.strip()


def _exact_proof(ctx, repo, commit, *, require_full):
    """The task's own green proof when it is a run of exactly ``commit`` (or of the live
    candidate, with no commit named) that still matches; otherwise ``None``."""
    from ouroboros.commit_admission import preflight_test_proof_matches

    proof = getattr(ctx, "_preflight_test_proof", None)
    try:
        # A named commit needs a run of exactly its tree: green uncommitted or untracked
        # bytes on top of it are another body.
        if (getattr(ctx, "_preflight_tests_passed", False) and proof is not None
                and (not commit or (proof.head == commit and proof.tree == proof.index_tree == _commit_tree(repo, commit)))
                and preflight_test_proof_matches(ctx, repo, require_full=require_full)):
            return proof
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        pass
    return None


def full_proof_error(ctx, repo, commit="") -> str:
    if _exact_proof(ctx, repo, commit, require_full=True) is not None:
        return ""
    return "a full green exact-tree test proof is required; selected, skipped or stale evidence cannot qualify"


def _scope_gap(repo, commit, scope, local) -> str:
    """Why a selected run's scope cannot qualify this local adoption, or ``""``."""
    from ouroboros.test_selection import policy_digest

    if scope.get("base") != local["onto"]:
        return f"it was selected against {str(scope.get('base'))[:12]}, not the serving HEAD {local['onto'][:12]}"
    if scope.get("policy") != policy_digest():
        return "it ran under another selection policy"
    missing = sorted(set(local["paths"]) - set(scope.get("changed") or ()))
    if missing:
        return "its selection did not cover " + ", ".join(missing[:5])
    owed = local["widening"]
    if owed.get("full_tests"):
        return "a commit in this delta declared broad impact (full_tests): " + str(owed.get("test_reason") or "")
    absent = sorted(set(owed.get("extra_tests") or ()) - set(scope.get("extra_tests") or ()))
    if absent:  # a declared consumer the commit no longer carries has nothing left to run
        listed = _git(repo, "ls-tree", "--name-only", commit, "--", *absent)
        if listed.returncode or listed.stdout.split():
            return "it did not run the declared indirect consumers " + ", ".join(absent[:5])
    return ""


def receipt_for_commit(data_dir, repo, commit, local=None, gaps=None) -> dict:
    """A coherent durable receipt of exactly ``commit``, or ``{}``: a complete full run, else
    (only for a ``local`` adoption) a selected one whose scope qualifies it; ``gaps``
    collects why the selected records found do not."""
    from ouroboros.preflight_receipts import RECEIPT_SCHEMA, RECEIPTS_DIR, SELECTED_SCHEMA, _record_problem

    tree = _commit_tree(repo, commit)
    if not tree:
        return {}
    # The run must have carried the Node lane exactly when this commit has web tests.
    node_ran = any(name.endswith(".test.js") for name in _git(repo, "ls-tree", "-r", "--name-only", commit, "--",
                                                               "web/tests").stdout.split())
    selected = {}
    for path in sorted((Path(data_dir) / "state" / RECEIPTS_DIR).glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            identity = document["identity"]
            if (identity["head"] != commit or not identity["tree"] == identity["index_tree"] == tree
                    or _record_problem(document, identity, node_ran)):
                continue
            found = {"source": "receipt", "key": path.stem, "head": commit, "tree": tree,
                     "created_at": document["created_at"], "phase": document["phase"]}
            if identity["schema"] == RECEIPT_SCHEMA:
                return {"tier": "full", **found}
            if identity["schema"] == SELECTED_SCHEMA and local is not None and not selected:
                gap = _scope_gap(repo, commit, identity["scope"], local)
                if not gap:
                    selected = {"tier": "selected", **found, **_scope_facts(identity["scope"], document.get("selection"))}
                elif gaps is not None:
                    gaps.append(f"its selected record {path.stem[:12]}: {gap}")
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return selected


def _scope_facts(scope, counts) -> dict:
    """What the handoff and the owner note show of a selected qualification: no lists of tests."""
    counts = counts if isinstance(counts, dict) else {}
    return {"base": scope["base"], "policy": scope["policy"], "changed": len(scope["changed"]),
            **{key: counts[key] for key in ("rule", "file_count", "omitted_files") if key in counts},
            "extra_tests": list(scope["extra_tests"]), "test_reason": scope["test_reason"]}


def adoption_test_evidence(data_dir, repo, commit, ctx=None, *, local=None) -> dict:
    """Existing evidence of exactly ``commit`` for its adoption; never a new run.

    Full evidence qualifies any adoption: the task's own proof, else a durable receipt.
    ``local`` — ``{"onto": serving HEAD, "paths": switch paths, "widening": what the
    delta's commits declared}``, given only when every one of them came through the commit
    gate — also admits a SELECTED run of exactly ``commit`` (the task's proof, else its
    durable record) selected against ``onto`` under the policy this process loaded,
    covering every switched path and every declared widening.
    """
    if ctx is not None and (proof := _exact_proof(ctx, repo, commit, require_full=True)) is not None:
        return {"tier": "full", "source": "proof", "head": proof.head, "tree": proof.tree}
    gaps: list = []
    receipt = receipt_for_commit(data_dir, repo, commit, local, gaps)
    if receipt.get("tier") == "full":
        return receipt
    proof = _exact_proof(ctx, repo, commit, require_full=False) if local is not None and ctx is not None else None
    if proof is not None:
        selection = getattr(proof, "selection", None)
        gap = "it has no selection" if selection is None else _scope_gap(repo, commit, selection.scope(), local)
        if not gap:
            return {"tier": "selected", "source": "proof", "head": proof.head, "tree": proof.tree,
                    **_scope_facts(selection.scope(), selection.record())}
        gaps.insert(0, f"the task's selected run: {gap}")
    if receipt:
        return receipt
    why = f" ({'; '.join(gaps[:3])})" if gaps else ""
    raise ValueError(f"no qualifying test run of exactly {str(commit)[:12]} is on record{why}; {_HOW}")
