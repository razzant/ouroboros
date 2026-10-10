# Code navigation

`query_code` returns source evidence and local syntax facts for reading and
changing unfamiliar code. It keeps ten operations on one tool surface. The model
chooses which evidence to inspect and which names or files to follow; the tool
does not prescribe a reasoning workflow.

An occurrence carries a source location, the grammar's raw node/field context
when available, enclosing names and a source slice. Parser-qualified hints such
as `call?` describe syntax. A token match, an enclosing name or a unique import
path candidate does not establish semantic binding. There is no compiler,
cross-file binding engine or mandatory language server behind these replies.

## Operations and scope

| Operation | Query | Directory `path` | File `path` | Result |
|---|---|---|---|---|
| `symbols` | Optional exact name | Outline scope | That file | Local outline symbols, including qualified declarators |
| `definition` | Exact name | Search scope | That file | Outline definitions and separately labelled syntax candidates |
| `references` | Exact token | Search scope | Definition selector for ordering; occurrences searched from root | Token occurrences with syntax context or text evidence |
| `callers` | Exact token | Search scope | Definition selector for ordering; calls searched from root | Syntactic callee-position occurrences, including member calls |
| `callees` | Definition name | Definition scope | Definition in that file | Calls within matching outline ranges |
| `impact` | File path or symbol | Importer/occurrence scope | Matching file target, or symbol definition selector | Import candidate rows for a file; occurrence files and import candidates for a symbol |
| `relevant_files` | Task words | Filter before scoring | That file | Ranked files using path, symbol, import and route facts |
| `digest` | Empty | Scoped directory rollups and file entries | One file entry | Symbols, imports, routes, calls and disposition coverage |
| `structural` | AST/node type | Independent bounded walk | That file | Grammar node matches; Python has a stdlib-AST fallback |
| `architecture` | Fact and argument | Non-default path unsupported | Non-default path unsupported | Facts from the Ouroboros repository's pinned carriers |

For `impact`, the target comes from `query`. A file `path` may select a symbol's
definition or repeat the file target; a conflicting file target is an argument
error. A file selector for `references` or `callers` orders evidence rather than
filtering it into an asserted binding set. Other files with the same spelling
remain relevant evidence, including competing declarations.

`kind` filters `symbols` and `definition`. `lang` filters inventory operations
and `structural`, using normalized grammar names; `architecture` does not accept
it. `depth` applies only to `impact`, bounded to 1–5. Non-default parameters an
operation does not use are refused rather than silently ignored.

All operations use the existing `root`, `bucket` and `skill_name` resource
binding. `user_files` requires an explicit target directory or file. Architecture
facts require an `active_workspace` or `system_repo` carrying the repository's
inventories. Existing access checks apply before source reads, including for
subagents; source snippets retain the existing exact-source egress policy with
no added masking layer.

## Reading the evidence

An outline entry is a parser-qualified declaration fact. Exported `const`
declarators, including declarations inside namespaces, belong in the outline
when their syntax is supported; this also makes them available to digest and
symbol-based consumers. A generic `name` field alone is insufficient: it can
name a Java invocation or a Python keyword argument. Additional definition
candidates retain their raw syntax and a candidate label separately from the
outline.

The shared language mapping treats `.mjs`/`.cjs` as JavaScript and `.mts`/`.cts`
as TypeScript for outlines, calls and imports. Files without an available outline
parser report `structural_unavailable:<language>` instead of an indexed empty
outline; their text can still contribute occurrence evidence.

`references` can expose identifiers, strings and comments. Files without a
usable grammar can still contribute text evidence. Text evidence is line-level:
the first match on a line without syntax context anchors that line, so one long
line may hold further raw matches. `callers` selects recognized callee positions
and does not promote text matches to calls. A qualified callee keeps its final
name, so PHP `\A\B\foo()` is a `foo` call. Type arguments are not an operand, so
where a C# grammar loads, `Target<int>()` is a `Target` call. A computed callee such as
`getters[key]()` has no callee name; its receiver and index stay ordinary
references. Go spells type arguments with the same brackets, so syntax alone
cannot tell `handlers[i]()` from the generic call `Make[int]()`: tree-sitter-go
gives both one tree and both names stay `call?` candidates. A literal or
expression index is unambiguous; a one-argument `Make[T](x)` parses as a
generic-type conversion and stays a reference. `callees` retains the local call view within the chosen outline
ranges, which describe the bytes the inventory hashed in its line model. A file
changed since then, or a Python file whose bare CR makes AST outline lines differ
from parser rows, contributes no callee rows. The reply is marked incomplete when
the current parse still finds calls in such a file; when it finds none, nothing
remains to misattribute and the mismatch is not disclosed, although the listed
definition ranges still describe the inventory's bytes.
Grammar coverage and
syntax errors can limit those views; an empty caller list does not prove that a
symbol has no runtime callers.

Aliases, namespace imports and re-exports remain visible as source syntax.
For example, a search for `helper` can show `from a import helper as chosen`.
The use `chosen()` requires a query for `chosen`; a parameter with that same name
may shadow the import. Neither spelling search proves which value a call uses.
The same evidence can expose an event name in registration and emission code
without adding a dedicated event graph.

File impact compares import specifiers with current inventory paths using one
request-local path lookup. The import scan reuses complete local syntax facts
after validating their hash against the bytes read for the source anchor. It
parses a file only when those facts are unavailable, exceeded their storage bound,
or no longer match the source. A relative or suffix match
is a filesystem candidate. Multiple matches remain ambiguous,
with sibling candidates disclosed. Only unique candidates expand to another
requested depth; every later hop retains the candidate qualification. Project
configuration, package resolution, dynamic imports and runtime behavior can
disagree with these path matches. Symbol impact combines files containing the
token with import candidates for its definition files. Each occurrence-file
summary retains a representative source anchor, preferring syntactic call
evidence over an earlier comment or other text match. It is evidence to inspect,
not a complete change-impact analysis.

## Pages and limits

`limit` bounds the returned page to 1–200 entries; it is not a 200-entry collection
cutoff. `offset` applies after operation-specific selection and ordering. Counts
refer to the selected operation before pagination, so unrelated token occurrences
do not consume a caller page. When a scan stops early, the reply discloses that
its count is a lower bound. A page beyond the collected results distinguishes
past-end from incomplete collection rather than reporting that no matches exist.

Digest applies its scope before counting and paging. Each file block is one
entry; directory rollups summarize the requested subtree. Relevant-file ranking
also applies scope before scoring and paging. Each request reads the current
files again. Page order is deterministic for unchanged inputs, but edits between
requests can move, add or remove rows; an offset is not an immutable snapshot.

Inventory operations enumerate visible files, including nested Git repositories
under their own ignore rules. `structural` uses its separate bounded filesystem
walk and reports that method. Architecture reads its pinned carriers. These
operations do not promise identical file universes, and `search_code` has its own
enumeration and transport rules.

Replies state the actual method, scope and relevant coverage limits: skipped or
oversized files, missing grammars, partial syntax trees, and applicable file,
row, parse or time caps. Completeness is only within that declared coverage.
Unknown grammar, generated code, dynamic names and configured module resolution
remain explicit limits rather than evidence of absence.

The inventory bounds enumeration to 20,000 entries and 45 seconds, and local
fact reads to 2 MB per file. Occurrence and structural reads use the existing
search transport's 1 MB limit. Occurrence views have a separate 20,000 selected
row work limit; structural has the same row bound. Impact applies its selected
row bound to occurrence-file summaries and matching import-candidate evidence,
so unrelated intermediate import specifiers do not spend the output budget.
These bounds are disclosed when reached and are independent of page size.
Directory rollups show at most
50 immediate directories; their flat file entries remain pageable. Source
slices preserve the text around the match, with an ellipsis when clipped;
tree-sitter and Python AST import columns count UTF-8 bytes. Each slice uses the
line model of its anchor: tree-sitter rows and literal matches count LF only, so
a form feed or other Unicode line separator stays inside the raw line, while
Python AST lines also end at a bare CR. Python AST local-call
fallback supplies line anchors without inventing columns. It rechecks the hash
against the current source read; a mismatch omits stale calls and visibly marks
the result incomplete.

Import specifier extraction is exercised for Python, JS/TS, Go, Java and Rust;
other grammar shapes are marked unverified. Literal `require()` / `import()`
spellings are syntax evidence and can be shadowed. Computed imports, compiler
configuration and package export maps are not evaluated. `lang` limits impact
evidence rows, while candidate matching retains the current path universe so
that a sibling in another language cannot silently erase ambiguity.

## Local cache and consumers

`code_intelligence.py` keeps a schema-5, hash-keyed cache of local file facts:
outlines, display import facts, calls, routes, coverage and a separate bounded
import syntax projection. That projection stores specifiers, line/column anchors,
parser slots and enclosing names, with method/completeness metadata. Valid Python
uses its existing stdlib AST parse; other languages reuse the outline tree-sitter
parse. Invalid Python retains tree-sitter partial import syntax when available.
The display `imports` field remains unchanged: its relative Python names and
non-JS/TS first-line summaries are not normalized candidate specifiers.
The reserved `exports`
field is currently unpopulated; export syntax is visible through source evidence.
The cache stores no source bodies, general occurrence lists or resolved cross-file
import paths. Import storage is bounded to 2,000 facts and 128,000 characters of
specifier/slot/enclosing text per file. Overflow discards that optional projection
and requires a source parse under the existing query limits; it never certifies
an empty or truncated import set. Source snippets and token occurrences are
obtained at query time, and import joins are recomputed against current paths,
so adding or removing a target can change impact without editing its importer.
Older caches rebuild on first use; the cache remains disposable derived data.

Local calls continue to feed digest's `Calls:` view. Tree-sitter call facts use the
callee positions `callers` recognizes, so a computed callee such as `getters[key]()`
records no call; schema 5 discards caches written by the earlier extractor.
`symbol_definitions` remains
available to architecture ownership queries. A bare-symbol `owner_of` reads a
current inventory with the same file admission and exclusions as the other
inventory operations; population modules without an indexed outline (enumeration
or time limits, skipped or excluded files, syntax errors) are named as a coverage limit, so an
empty or short owner list is not reported as absence. Module-path and dotted
lookups read only the manifest. The existing import-text and
relative-import helpers remain available to their architecture/review consumers.
A restricted or partial resource view does not overwrite the shared inventory
cache. Retention and reset behavior are recorded in [PERSISTENCE](PERSISTENCE.md).
