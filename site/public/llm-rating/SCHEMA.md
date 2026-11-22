# Data schema

Schema version 3. The page is drawn from one JSON document per edition.

| File | What it is |
| --- | --- |
| `data/editions/<id>/edition.json` | one edition, frozen after publication |
| `data/latest.json` | a copy of the newest edition |
| `data/latest.js` | the same data as a script, so the page also opens from a local file |
| `data/editions/index.json` | the list of editions shown on the editions page |
| `data/editions.js` | the same list as a script |

Texts that a reader sees are objects with two keys, `en` and `ru`. Dates are ISO dates (`YYYY-MM-DD`). Counts are whole numbers; shares are percentages with one decimal.

## Edition

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | number | version of this schema |
| `edition` | string | edition id, year and month: `2026.11` |
| `status` | `draft` or `final` | `draft` until every case is verified and I have locked the letters |
| `method_version` | number | version of the method, see METHOD.md |
| `period` | `{from, to}` | first and last day of the logs counted for the edition; a case carries its own date, and a few lie just outside |
| `published` | date or null | publication date; null before publication |
| `built` | date | the day the files were built |
| `previous_edition` | string or null | the edition before this one |
| `totals` | object | see Totals |
| `scale` | array | `{id, name}` for each letter in use, best first: what the letter means |
| `use` | object | `bands` (`often`, `sometimes`, `rarely`) and the `rule` that sorts records into them, in words |
| `groups` | array | `{id, name, main, works[]}`: groups of work in table order; `main` is true for a column of the main table |
| `works` | array | see Work |
| `models` | array | see Model; the ranked models first, in the order of my ranking |
| `cells` | array | see Cell; one entry per ranked model and kind of work |
| `picks` | array | `{group, models[]}`: who holds the top letter in each column of the main table |
| `cases` | array | see Case |
| `notes` | object | `author`, `review_from`, `changes`, `changelog`, `postscript` |

Nothing in the data is computed from anything else in it: the order of the models and every letter are written by hand.

## Totals

| Field | Meaning |
| --- | --- |
| `models` | `{rated, earlier, trial, untried}`: how many models have each status |
| `work_types`, `work_groups`, `columns` | kinds of work, groups of work, and columns of the main table |
| `letters` | `{main, works, own, unrated}`: letters in the main table; letters for kinds of work; how many of those carry a note of their own; cells without a letter |
| `cases` | dated cases in the edition |
| `tokens` | `{status, read, lower_bound, read_display, written, since, cache_share, definition, period_read, period_read_note}` |

`tokens.status` is `pending` while the total is being reconciled; then only `status` is present and the page shows a placeholder. When it is `final`: `read` and `written` are whole token counts since `since`, the date of the first record; `lower_bound` says that `read` is a minimum; `read_display` is the short form shown on the page; `cache_share` is the part of `read` that came from cache, between 0 and 1; `definition` says what was counted; `period_read` and `period_read_note` give the tokens read inside the edition's period.

## Work

| Field | Meaning |
| --- | --- |
| `id` | stable identifier, for example `rev_code` |
| `group` | id of its group |
| `name`, `what` | short name and one-line description |
| `unrated` | optional: `seat` when some cells of this kind have no letter because the work is a seat set by configuration |
| `total` | records of this kind in the logs, all models together. A record is a hand-off or a review seat inside the period; Safety and Main mind also include safety-call or root-task records from the whole event log. Those two totals are mixed record counts, not counts of calls alone or distinct tasks |
| `models_tried` | how many models have at least one record |
| `split` | up to six entries `{model, n, share}`: who gets this work |
| `other` | `{n, share}` for everyone else, or null |

`total`, `models_tried`, `split` and `other` are telemetry of use. They move no letter.

## Model

| Field | Meaning |
| --- | --- |
| `id` | stable identifier, for example `claude-opus-5.5` |
| `name` | display name |
| `short` | short label used in the table |
| `family`, `maker` | model line and the company that makes it |
| `status` | `rated` (in the ranking), `earlier` (an earlier version, listed apart), `trial` (a newcomer, listed apart) or `untried` |
| `rank` | place in my ranking, from 1; only for `rated` |
| `one_liner` | one sentence about the model; only for `rated` |
| `verdict` | its portrait, a few sentences that continue from `one_liner`; only for `rated` |
| `trust`, `check` | what I trust the model with, and what I check after it; only for `rated` |
| `groups` | only for `rated`: per column of the main table, `{grade, reason, cases}`: the letter, why, and the ids of the cases told in this column |
| `picks` | only for `rated`: ids of the columns where this model is my first pick |
| `note` | for `earlier` and `trial`: what I can say about the model without a letter |
| `tried` | for `earlier` and `trial`: kinds of work with at least one record |
| `stats` | optional records for the details: `records`, `review {seats, crit}`, `plan {answers, objections}`, `routes[] {harness, n, ok, ok_pct, top[] {kind, n}}` |
| `cases` | ids of the cases that mention the model |

## Cell

One entry per ranked model and kind of work.

| Field | Meaning |
| --- | --- |
| `model`, `work` | ids |
| `grade` | the letter, or null when the cell has no letter on purpose |
| `unrated` | present when `grade` is null: `seat`, a seat set by configuration |
| `note` | optional: the reason of this letter in its own words. It is always present when the letter differs from the letter of the column |
| `use` | `{band, n, share, rank, of}`: how often the model gets this work. `band` is `often`, `sometimes`, `rarely` or null; `n` is records, as in Work; `share` is the model's part of this work in percent; `rank` is its place by `n` among the `of` models that have records. Telemetry: it moves no letter |
| `stats` | optional records for the details, see below |
| `cases` | ids of the cases told for this cell |

A graded cell without a `note` must take the letter of a main column, whose reason is in the model's `groups`. Every graded cell outside the main columns has its own bilingual `note`.

`stats` may hold: `review {seats, answered, crit, sole, sole_fixed, route {agent, api}, missed [a, b], eye [read, real, rare, optional, wrong]}`, `plan {answers, zero, objections, lifted, open, unparseable}`, `first_pass {chains, first, self_judged}`, `safety {calls}`, `lead {roots}`. These are records of what happened. None of them is a rate of correctness.

## Case

| Field | Meaning |
| --- | --- |
| `id` | number, stable inside the edition; ids of cases that were dropped are not reused |
| `from`, `to` | the day, or the first and last day |
| `models`, `works` | ids the case belongs to; `works` may be empty for a case about the model in general |
| `text` | what the task was, what the model did and what verified it, told without private detail |

## Notes

`notes.author` is who set the order and the letters. `notes.review_from` is the first day of the review log. `notes.changes` lists what moved since the previous edition: `{model, rank_from, rank}` for a place, `{model, group, from, to}` for a letter of the main table, `{model, new, rank}` and `{model, left}` for a model that enters or leaves the ranking. It is empty in a first edition and across a change of method. `notes.changelog` is the edition's entry in CHANGELOG.md. `notes.postscript` is an optional closing line of the page.

## Compatibility

New optional fields may appear without a new schema version. Renaming or removing a field, or changing what a field means, raises `schema`. Schemas 1 and 2 belonged to drafts that were never published.
