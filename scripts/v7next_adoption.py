#!/usr/bin/env python3
"""Validate ADOPTION_v7next.md — the v7-side adoption manifest (Ф0 skeleton).

The manifest enumerates every v7-side delta that must be re-applied on top of
the v7next upstream base: the 18 approved semantic-delta families from the
frozen reference ledger (``ouroboros_v7_wip @ 9f691656`` —
``scripts/v7_migration.py::APPROVED_SEMANTIC_DELTAS`` minus ``"none"``) plus
the campaign-decision items of plan §6 (ABI package 7.0) and §7 (completeness)
and the plan §2 class returns.

Checks (plan §5.1, roast F2 — artifact/train-based manifest):

- the fixed 7-column table schema parses;
- ids are unique and well-formed;
- every required delta family D02–D38 is present as ``kind=semantic-delta``;
- ``kind`` / ``disposition`` / ``status`` / ``phase`` come from closed enums
  (dispositions per the plan §5.4 three-column rule: retain / re-prove /
  superseded-by-upstream, with ``pending-decision`` allowed only before
  release);
- every row carries a non-empty verification hook;
- every post-cutoff upstream train the campaign absorbed keeps its row, still
  naming its upstream tip and its campaign merge (both modes — a whole-file
  overwrite deleted the sync #2 row past a bar that only the default mode ran);
- a ``done`` row's hook RESOLVES: every repo path exists and every ``::nodeid``
  names something the file actually defines (read by AST);
- a ``done`` row does not say the work is open, unless it declares what stays
  open in an explicit ``residual:`` clause;
- a post-release row's recorded authority and its text tell one story: an
  OWNER deferral carries the owner's ``owner verbatim «…»`` quote in the row,
  an operator disclosure carries none;
- the prose outside the table (header, schema, Notes) names ids only as the
  table has them: every id-shaped token there resolves to a row unless the
  prose declares it on a ``No-row ids: …`` line, and a declared no-row id must
  not have a row (the Notes called W4-F3/W4-F4 rowless for two days after
  d348ea46 made them rows — a green bar both days, because nothing read the
  Notes). Disclosed residual: a rowless claim written as free English is not
  read — a word marker was tried and misfired on «No row carries
  pending-decision any more» — so the schema gives the claim its declared
  form and the id resolution is the check that does not depend on wording;
- the Notes' ``Deferral authorities: <id> <authority>, …`` declaration (the
  register's mirror) names exactly the ids ``DEFERRED_OUT_OF_V70`` records with
  exactly their authority — the Notes called W4-F4 operator-disclosed for a day
  after the register made it an owner deferral, and nothing read the Notes;
  free prose about authority is still not read, the declared form is;
- ``--release``: no ``pending-decision`` dispositions and every row ``done``
  ("no unresolved rows at release", plan §10), with post-release rows leaving
  the bar only through a recorded deferral (owner-authored for the required
  inventory).

Exit 0 when green, 1 with findings, 2 when the manifest itself is missing or
structurally unparseable.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys
from collections import Counter

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "ADOPTION_v7next.md"

HEADER = ["id", "kind", "what", "disposition", "status", "phase", "verification hook"]

# APPROVED_SEMANTIC_DELTAS of the frozen reference, minus "none".
REQUIRED_DELTAS = (
    "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09", "D11",
    "D13", "D18", "D31", "D33", "D34", "D35", "D36", "D37", "D38",
)
# Required non-delta inventory (F0 phase review F2): the ABI package and the
# compatibility retirements are release-gated too, not only the D-families.
REQUIRED_ABI = tuple(f"ABI-{n}" for n in range(1, 11))
REQUIRED_CPL = tuple(f"CPL-{n}" for n in range(1, 8))
# F0 review round 2: the phase of every required row is itself part of the
# owner-approved inventory — a required row silently rescheduled to another
# phase (or parked post-release without a recorded deferral) must turn the
# validator red. DEFERRED_OUT_OF_V70 is that record: every post-release row
# must appear here, and a row of the owner-approved required inventory
# (REQUIRED_PHASE) may only be parked with OWNER authority, so flipping a
# required row post-release still cannot bypass the release bar. Operator
# authority covers rows that are disclosures rather than owner decisions —
# a defect a wave found and named instead of fixing.
REQUIRED_PHASE = {
    # D02 F1->F3: owner-ratified F3 layout (2026-08-31) — the typed organ is
    # re-derived whole by the F3.1 lane A; seam commit updates row + pin together.
    # D03 F1->F6 (ADOPTION truth wave, 2026-09-01): F1 closed with the settings
    # seam's rows 913-917/1080-1081 still hot-deferred, so the pin named a dead
    # phase. F6 is the live phase. This is an OPERATOR scheduling correction, not
    # an owner decision — disclosed in the manifest row and the ledger so the
    # owner can overturn it. Sibling rows D04/D05/D06/D35 landed through their
    # owner-decided lanes and read done; their F1 pins were deliberately left
    # alone (one decision per class, and nobody has decided this one).
    "D02": "F3", "D03": "F6", "D04": "F1", "D05": "F1", "D06": "F1",
    "D07": "F2", "D08": "F2", "D09": "F1", "D11": "F1", "D13": "F1",
    "D18": "F1", "D31": "F2", "D33": "F1", "D34": "F2", "D35": "F1",
    "D36": "F2", "D37": "F2", "D38": "F1",
    "ABI-1": "F3", "ABI-2": "F3", "ABI-3": "F3", "ABI-4": "F3",
    "ABI-5": "F3", "ABI-6": "F3", "ABI-7": "F3", "ABI-8": "POST",
    "ABI-9": "F3", "ABI-10": "F3",
    "CPL-1": "F5", "CPL-2": "F5", "CPL-3": "F5", "CPL-4": "F5",
    "CPL-5": "F5", "CPL-6": "F5", "CPL-7": "F5",
    "DEFER-BROWSER": "POST",
}
OWNER, OPERATOR = "owner", "operator-disclosed"
DEFERRED_OUT_OF_V70 = {
    # Owner decisions: Q5=A kept the handler ABI out of the bundle and Q16=A
    # retired the «7.1» label into the post-release backlog (ABI-8); batch №9
    # №14=A put the browser wave after the release, with a green smoke — not a
    # green browser lane — as the condition on the tag (DEFER-BROWSER).
    "ABI-8": OWNER,
    "DEFER-BROWSER": OWNER,
    # (W4-F1 and W4-F2 — the two evolution crash windows the F4 wave-4 lane
    # disclosed instead of fixing — were pulled INTO 7.0 by owner batch №13
    # item 9 = B, so they are no longer deferrals: their rows read done.)
    # Owner-sanctioned deferrals that lived as prose inside done rows or in the
    # ledger until the stage-2 bookkeeping made them rows (quotes in each row):
    # batch №7 5=A (headless cancel receipts), batch №9 №12=A (two frozen modules),
    # batch №12 A (C6 residuals), batch №8 5=A (task_results eternal).
    # Operator disclosure WITHOUT an owner decision: the wave-4 observation
    # W4-F4 only. Batch №13 item 13(и) asked to ratify W4-F3/W4-F4 and the owner
    # answered that he had not read that item; the F3 owner batch of 2026-09-04
    # re-asked, and its item 5 = A pulled W4-F3 INTO 7.0 (the marker is always
    # written; that row reads done and is no longer a deferral). Owner-decided
    # since batch №13:
    # DEFER-E2E-PAID-LANE (item 2 = A ordered the paid lane RUN once — E1/E13
    # executed green; E2/E3 stay unexecuted for a structural reason, the real
    # Claudexor lane needs a logged-in Claude account, which is the owner's act
    # — so the quote covers the execution and the remainder is a disclosed
    # block, not a waiver) and DEFER-SPEC64-PATHS (item 8 = A). (Left this
    # record by the same batch: item 10 = B pulled DEFER-TYPED-PROC-5 into 7.0;
    # item 15 = B landed the mutating delegation scenarios S24/S25 —
    # DEFER-E2E-DELEG-MUT reads done at phase F4; item 7 = A closed F23 as
    # covered by the release bar.)
    "DEFER-HEADLESS-CANCEL": OWNER,
    "DEFER-FROZEN-2": OWNER,
    "DEFER-C6-RESIDUALS": OWNER,
    "DEFER-C19-RETENTION": OWNER,
    "W4-F4": OWNER,
    "DEFER-E2E-PAID-LANE": OWNER,  # batch №13 item 2 = A (the run order); E2/E3 blocked structurally
    "DEFER-SPEC64-PATHS": OWNER,  # batch №13 item 8 = A
}
# Post-cutoff upstream adoption trains: id -> (upstream tip, campaign merge).
# A frozen inventory rather than a git derivation, and the history is the
# reason. Each sync's absorb merge does take its upstream tip as the literal
# second parent (20850191<-8d13373b, b9ceed6e<-f3fbfdbb, f4abe0a5<-a76961de),
# but only f4abe0a5 sits on this branch's first-parent line: the other two were
# made on lane lines and reached mainline on the second-parent side of a lane
# integration merge over a CAMPAIGN commit (0aa74e9f over 816e7b82, 0f9a8daf
# over 4c32691e). So a rule walking --first-parent merges and reading second
# parents would police one train of three and stay blind to the other two —
# exactly the hole that lost TRAIN-F6b-f3fbfdbb; widened to "second parent
# descends from a recorded upstream tip" it would demand a train row for every
# lane merge made after a sync (35 / 15 / 6 merges on this tree for the three
# tips), the C6 lane merge 9faccf31 over 8fb08d44 included. Neither is honest,
# and both need a subprocess. Sync #1 is recorded by its mainline carrier
# 0aa74e9f and names absorb merge 20850191 in the row text too; syncs #2 and #3
# are recorded by the absorb merge itself. Adding a train here is the same edit
# as merging one, and a deleted row is red at once.
REQUIRED_TRAINS = {
    "TRAIN-F6-8d13373b": ("8d13373b", "0aa74e9f"),
    "TRAIN-F6b-f3fbfdbb": ("f3fbfdbb", "b9ceed6e"),
    "TRAIN-F6c-a76961de": ("a76961de", "f4abe0a5"),
}
KINDS = frozenset({"semantic-delta", "plan-item", "class-return"})
DISPOSITIONS = frozenset({"retain", "re-prove", "superseded-by-upstream",
                          "pending-decision", "post-release"})
STATUSES = frozenset({"pending", "in-progress", "done", "deferred"})
PHASES = frozenset({"F0", "F1", "F2", "F3", "F4", "F5", "F6", "POST"})
ID_RE = re.compile(r"^(D\d\d|ABI-\d+|CPL-\d+|R-[A-Z0-9]+|TRAIN-[A-Za-z0-9._-]+"
                   r"|DEFER-[A-Z0-9][A-Z0-9-]*|W\d-F\d+)$")  # DEFER ids may carry hyphenated tokens (DEFER-E2E-PAID-LANE)
# The same id grammar, unanchored, for the prose outside the table. The
# boundaries keep `D-14` (a plan decision) and `CPL4-C6` (a lane label) out;
# the TRAIN class admits dots, so a sentence-final one is stripped by the reader.
_PROSE_ID_RE = re.compile(r"(?<![\w-])" + ID_RE.pattern[1:-1] + r"(?![\w-])")
# The one declared form for an id the prose names without a row (a folded or
# withdrawn row): a line `No-row ids: A, B`, optionally as a Notes bullet.
_NO_ROW_DECL_RE = re.compile(r"^\s*(?:-\s*)?No-row ids:(.*)$", re.M)
# The manifest's quote convention for an owner decision, as every OWNER row
# already spells it — the marker the authority lint keys on.
_OWNER_QUOTE_MARKER = "owner verbatim «"
# The one declared form for the deferral authorities the Notes claim — a
# `Deferral authorities…: <id> <authority>, …` bullet (continuation lines
# indented) mirroring DEFERRED_OUT_OF_V70. Free prose about authority is not read.
_DEFERRAL_DECL_RE = re.compile(
    r"^\s*(?:-\s*)?Deferral authorities\b[^:]*:(?P<body>.*(?:\n[ \t]+(?!-\s).*)*)", re.M)
_DEFERRAL_PAIR_RE = re.compile(_PROSE_ID_RE.pattern + rf"\s+({OPERATOR}|{OWNER})\b")


def split_row(line: str) -> list[str]:
    """Split one markdown table row on unescaped pipes."""
    body = line.strip().strip("|")
    cells, cur, escaped = [], [], False
    for ch in body:
        if escaped:
            cur.append(ch)
            escaped = False
        elif ch == "\\":
            cur.append(ch)
            escaped = True
        elif ch == "|":
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur).strip())
    return cells


def parse_rows(text: str) -> tuple[list[dict[str, str]], list[str]]:
    errors: list[str] = []
    rows: list[dict[str, str]] = []
    lines = text.splitlines()
    header_at = None
    for i, line in enumerate(lines):
        if line.startswith("|") and [c.lower() for c in split_row(line)] == HEADER:
            header_at = i
            break
    if header_at is None:
        errors.append(f"table header not found; expected columns: {' | '.join(HEADER)}")
        return rows, errors
    for j in range(header_at + 2, len(lines)):
        line = lines[j]
        if not line.startswith("|"):
            break
        cells = split_row(line)
        if len(cells) != len(HEADER):
            errors.append(f"line {j + 1}: expected {len(HEADER)} cells, got {len(cells)}")
            continue
        rows.append(dict(zip(HEADER, cells)))
    return rows, errors


def manifest_prose(text: str) -> str:
    """Everything outside the table — the header, the schema, the Notes — by
    the rule ``parse_rows`` already uses: a table line starts with ``|``."""
    return "\n".join(line for line in text.splitlines() if not line.startswith("|"))


def _prose_id_errors(prose: str, by_id: dict[str, dict[str, str]]) -> list[str]:
    """The prose may name a row id only as the table has it. Tokens are read by
    the table's own id grammar, not by phrasing, so the check does not depend
    on how a sentence says 'gets no row': an id without a row must be declared
    on a ``No-row ids:`` line, and a declared id must not have a row."""
    declared: set[str] = set()
    for m in _NO_ROW_DECL_RE.finditer(prose):
        declared.update(t.rstrip(".-") for t in _PROSE_ID_RE.findall(m.group(1)))
    named = {t.rstrip(".-") for t in _PROSE_ID_RE.findall(prose)}
    errors: list[str] = []
    for rid in sorted(declared & by_id.keys()):
        errors.append(f"prose: {rid} is declared under 'No-row ids:' while the "
                      "table has its row — drop the declaration or the row")
    for rid in sorted(named - by_id.keys() - declared):
        errors.append(f"prose: {rid} is named outside the table but has no row "
                      "and no 'No-row ids:' declaration")
    return errors


def declared_deferral_authorities(prose: str) -> dict[str, str] | None:
    """The Notes' ``Deferral authorities:`` declaration as ``{id: authority}``;
    ``None`` when the prose carries no declaration."""
    m = _DEFERRAL_DECL_RE.search(prose)
    if m is None:
        return None
    return dict(_DEFERRAL_PAIR_RE.findall(m.group("body")))


def _deferral_declaration_errors(prose: str) -> list[str]:
    """A declared deferral-authority list must match ``DEFERRED_OUT_OF_V70``
    exactly — the same ids, each with its recorded authority — so the Notes
    cannot tell a different story than the register."""
    declared = declared_deferral_authorities(prose)
    if declared is None:
        return []
    errors: list[str] = []
    for rid in sorted(declared.keys() - DEFERRED_OUT_OF_V70.keys()):
        errors.append(f"prose: Deferral authorities declares {rid}, which "
                      "DEFERRED_OUT_OF_V70 does not record")
    for rid in sorted(DEFERRED_OUT_OF_V70.keys() - declared.keys()):
        errors.append(f"prose: Deferral authorities omits {rid} "
                      f"({DEFERRED_OUT_OF_V70[rid]} in DEFERRED_OUT_OF_V70) — declare every recorded deferral")
    for rid in sorted(declared.keys() & DEFERRED_OUT_OF_V70.keys()):
        if declared[rid] != DEFERRED_OUT_OF_V70[rid]:
            errors.append(f"prose: Deferral authorities says {rid} is {declared[rid]} while "
                          f"DEFERRED_OUT_OF_V70 records {DEFERRED_OUT_OF_V70[rid]}")
    return errors


def validate(rows: list[dict[str, str]], release: bool, prose: str = "") -> list[str]:
    errors: list[str] = []
    ids = [r["id"] for r in rows]
    for rid, n in Counter(ids).items():
        if n > 1:
            errors.append(f"duplicate id: {rid} ({n} rows)")
    for r in rows:
        rid = r["id"]
        if not ID_RE.match(rid):
            errors.append(f"{rid or '<empty>'}: malformed id")
        if r["kind"] not in KINDS:
            errors.append(f"{rid}: unknown kind {r['kind']!r}")
        if r["disposition"] not in DISPOSITIONS:
            errors.append(f"{rid}: unknown disposition {r['disposition']!r}")
        if r["status"] not in STATUSES:
            errors.append(f"{rid}: unknown status {r['status']!r}")
        if r["phase"] not in PHASES:
            errors.append(f"{rid}: unknown phase {r['phase']!r}")
        if not r["what"]:
            errors.append(f"{rid}: empty 'what'")
        if not r["verification hook"]:
            errors.append(f"{rid}: empty verification hook")
    by_id = {r["id"]: r for r in rows}
    for d in REQUIRED_DELTAS:
        row = by_id.get(d)
        if row is None:
            errors.append(f"required semantic delta {d} is missing")
        elif row["kind"] != "semantic-delta":
            errors.append(f"{d}: must be kind=semantic-delta, got {row['kind']!r}")
    # F0 phase review F2: the ABI package and compatibility retirements are part
    # of the release inventory too — deleting their rows must turn --release red.
    for rid in (*REQUIRED_ABI, *REQUIRED_CPL):
        row = by_id.get(rid)
        if row is None:
            errors.append(f"required row {rid} is missing")
        elif row["kind"] != "plan-item":
            errors.append(f"{rid}: must be kind=plan-item, got {row['kind']!r}")
    # Row-specific coupling: post-release is a single coherent state, not three
    # independent knobs (prevents e.g. disposition=post-release with status=done
    # quietly counting as shipped) — and it needs a recorded deferral in
    # DEFERRED_OUT_OF_V70, owner-authored for the required inventory, so
    # flipping a required row to post-release cannot bypass the release bar.
    for r in rows:
        post_bits = [r["disposition"] == "post-release", r["status"] == "deferred",
                     r["phase"] == "POST"]
        if any(post_bits) and not all(post_bits):
            errors.append(
                f"{r['id']}: post-release rows need disposition=post-release + "
                f"status=deferred + phase=POST together, got "
                f"{r['disposition']}/{r['status']}/{r['phase']}")
        if all(post_bits):
            authority = DEFERRED_OUT_OF_V70.get(r["id"])
            if authority is None:
                errors.append(
                    f"{r['id']}: post-release needs a recorded deferral in "
                    f"DEFERRED_OUT_OF_V70 (currently "
                    f"{sorted(DEFERRED_OUT_OF_V70)})")
            elif authority != OWNER and r["id"] in REQUIRED_PHASE:
                errors.append(
                    f"{r['id']}: a row of the required inventory can only be "
                    f"parked post-release by an owner decision, not by "
                    f"{authority}")
            # The record and the row tell one story: an owner deferral carries
            # the owner's quote, an operator disclosure carries none. The
            # comment block over the record drifted from its values once
            # (E2/E3 and spec §6.4 read operator-disclosed beside OWNER), and
            # a reader trusts the prose first.
            quoted = _OWNER_QUOTE_MARKER in r["what"]
            if authority == OWNER and not quoted:
                errors.append(
                    f"{r['id']}: recorded as an owner deferral but the row carries "
                    f"no '{_OWNER_QUOTE_MARKER}…»' quote — quote the decision or "
                    f"record the row as {OPERATOR}")
            elif authority == OPERATOR and quoted:
                errors.append(
                    f"{r['id']}: recorded as {OPERATOR} but the row carries an owner "
                    f"quote — record the row as {OWNER} or drop the quote")
    # Every upstream train the campaign absorbed must keep its row, and the row
    # must still name the tip and the merge it is a record of. Both modes: the
    # deletion in 285ab66d survived because only the default mode was run.
    for rid, (tip, merge) in REQUIRED_TRAINS.items():
        row = by_id.get(rid)
        if row is None:
            errors.append(f"required upstream train {rid} is missing — every "
                          "absorbed upstream train keeps a row")
            continue
        if row["kind"] != "plan-item":
            errors.append(f"{rid}: must be kind=plan-item, got {row['kind']!r}")
        text = f"{row['what']} {row['verification hook']}"
        for sha in (tip, merge):
            if sha not in text:
                errors.append(f"{rid}: row text no longer names {sha} "
                              "(upstream tip and campaign merge are what the "
                              "row records)")
    # A shipped row must not say it is unshipped. This is a text-vs-cell
    # consistency lint on an operator manifest — not a semantic gate on any
    # runtime decision — and the `residual:` clause is the explicit escape, so
    # a genuine disclosure on a shipped row stays sayable.
    for r in rows:
        errors.extend(_honesty_errors(r))
        # Hook resolution is a property of a shipped row, not of the release
        # invocation, so it runs in both modes and its messages say `hook:`.
        # The manifest's own Notes state the same rule for its readers.
        if r["status"] == "done":
            errors.extend(_hook_resolution_errors(r))
    # Phase pinning of the required inventory.
    for rid, want in REQUIRED_PHASE.items():
        row = by_id.get(rid)
        if row is not None and row["phase"] != want:
            errors.append(f"{rid}: phase {row['phase']!r} != pinned {want!r} "
                          "(rescheduling a required row needs a new owner decision "
                          "and an update to REQUIRED_PHASE)")
    if prose:
        errors.extend(_prose_id_errors(prose, by_id))
        errors.extend(_deferral_declaration_errors(prose))
    if release:
        for r in rows:
            if r["disposition"] == "pending-decision":
                errors.append(f"release: {r['id']} still pending-decision")
            if r["disposition"] == "post-release":
                continue  # explicitly deferred out of v7.0 by an owner decision
            if r["status"] != "done":
                errors.append(f"release: {r['id']} status {r['status']!r} != done")
    return errors


# Any-extension token, anchored on BOTH sides: the lookbehind stops
# `not-scripts/x.py` being misread as a scripts/ reference (round 4), the
# lookahead stops `scripts/x.py-not-real` matching by its existing `.py`
# prefix (round 5) — a partial token is prose, not a reference.
_HOOK_PATH_RE = re.compile(r"(?<![\w./-])(?:tests|scripts|docs)/[\w./-]+\.\w+(?![\w-])")


# A shipped row that says the work is open contradicts its own status cell.
# The escape is explicit and named, not a keyword exception list.
_NOT_DONE_MARKERS = ("not done", "open residual", "not integrated yet",
                     "still owed", "read pending")
_RESIDUAL_CLAUSE = "residual:"


def _honesty_errors(row: dict[str, str]) -> list[str]:
    """A `done` row may carry an open residual — that is what a `residual:`
    clause declares — but it may not say the work itself is not done."""
    if row["status"] != "done":
        return []
    text = f"{row['what']} {row['verification hook']}".lower()
    if _RESIDUAL_CLAUSE in text:
        return []
    hits = [m for m in _NOT_DONE_MARKERS if m in text]
    if not hits:
        return []
    return [f"{row['id']}: status is done while the text says {hits!r}; either "
            f"the status is wrong or the row needs an explicit "
            f"'{_RESIDUAL_CLAUSE}' clause naming what stays open"]


# A hook may name a pytest node id. The path half was already resolved; the
# `::name` half was free text until now, so a hook could point at a suite that
# exists and a pin that does not.
_HOOK_NODEID_RE = re.compile(
    r"(?<![\w./-])((?:tests|scripts)/[\w./-]+\.py)((?:::[A-Za-z_]\w*)+)")


def _defined_names(path: pathlib.Path) -> set[str]:
    """Every name a pytest node id could legitimately address in a file:
    functions and classes at any depth (``path::Class::method``) plus
    module-level bindings — a hook may name the closed inventory a pin drives,
    not only the pin (`tests/_shared.py::SETTINGS_WRITERS`)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else [])
        names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def _hook_nodeid_errors(row: dict[str, str], hook: str) -> list[str]:
    errors: list[str] = []
    for rel, tail in _HOOK_NODEID_RE.findall(hook):
        path = (REPO_ROOT / rel).resolve()
        if not path.is_file():
            continue  # the path half is reported by the resolver above
        try:
            defined = _defined_names(path)
        except SyntaxError as exc:  # unparseable file: say so, do not pass it
            errors.append(f"hook: {row['id']} hook file {rel} does not parse ({exc})")
            continue
        for part in tail.split("::"):
            if part and part not in defined:
                errors.append(f"hook: {row['id']} hook names {rel}::{part}, "
                              f"which {rel} does not define")
    return errors


def _hook_resolution_errors(row: dict[str, str]) -> list[str]:
    """Shipped-row hook contract (F0 review rounds 1-4): a shipped row's
    verification hook must RESOLVE — prose alone cannot pass. At least one
    repo-path reference must be present, EVERY referenced token must exist
    (any extension — a smuggled bogus reference next to a valid one is an
    error, not ignored), and the path must stay inside its top directory
    (`tests/../x` traversal is rejected). This runs for every `done` row in
    BOTH modes — it is a property of a shipped row, not of the --release
    invocation — so the messages are prefixed `hook:`, not `release:`. A row
    that is not yet `done` keeps a free-prose hook, naming the suite the work
    will land in."""
    hook = row["verification hook"]
    paths = _HOOK_PATH_RE.findall(hook.replace("\\|", "|"))
    errors: list[str] = []
    if not paths:
        errors.append(
            f"hook: {row['id']} hook has no resolvable repo-path reference "
            "(tests/, scripts/ or docs/ file) — prose-only hooks cannot ship")
    for p in paths:
        top = p.split("/", 1)[0]
        candidate = (REPO_ROOT / p).resolve()
        top_root = (REPO_ROOT / top).resolve()
        # pathlib containment, not string prefixing: portable across
        # separators (round 5: the "/"-suffix check broke on Windows).
        inside = candidate == top_root or top_root in candidate.parents
        if ".." in p.split("/") or not inside:
            errors.append(f"hook: {row['id']} hook path escapes {top}/: {p}")
        elif not candidate.is_file():
            errors.append(f"hook: {row['id']} hook references missing file {p}")
    errors.extend(_hook_nodeid_errors(row, hook))
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", action="store_true",
                    help="enforce the release bar: no pending-decision, all rows done")
    ap.add_argument("--manifest", type=pathlib.Path, default=MANIFEST)
    args = ap.parse_args()

    if not args.manifest.is_file():
        print(f"missing manifest: {args.manifest}", file=sys.stderr)
        return 2
    text = args.manifest.read_text(encoding="utf-8")
    rows, errors = parse_rows(text)
    if not rows and errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 2
    errors += validate(rows, args.release, prose=manifest_prose(text))

    by_phase = Counter(r["phase"] for r in rows)
    by_disp = Counter(r["disposition"] for r in rows)
    by_status = Counter(r["status"] for r in rows)
    by_kind = Counter(r["kind"] for r in rows)
    print(f"{args.manifest.name}: {len(rows)} rows")
    print(f"  kind:        {dict(sorted(by_kind.items()))}")
    print(f"  phase:       {dict(sorted(by_phase.items()))}")
    print(f"  disposition: {dict(sorted(by_disp.items()))}")
    print(f"  status:      {dict(sorted(by_status.items()))}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print("OK" + (" (release bar)" if args.release else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
