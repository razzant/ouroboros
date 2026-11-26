# Data schema

Schema version 4, from edition 2026.12. The page is drawn from one JSON document per edition. Edition 2026.11 is schema 3; what differs is listed at the end, and the page reads both.

| File | What it is |
| --- | --- |
| `data/editions/<id>/edition.json` | one edition, frozen after publication |
| `data/latest.json` | a copy of the newest edition |
| `data/latest.js` | the same data as a script, so the page also opens from a local file |
| `data/editions/index.json` | the list of editions shown on the editions page |
| `data/editions.js` | the same list as a script |

Texts that a reader sees are objects with two keys, `en` and `ru`. Dates are ISO dates (`YYYY-MM-DD`). Counts are whole numbers.

## Edition

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | number | version of this schema: 4 |
| `edition` | string | edition id, year and month: `2026.12` |
| `status` | `draft` or `final` | `draft` until every shown case is verified by a second reader, the votes are counted, nothing is pending, and I have locked the letters |
| `method_version` | number | version of the method, see METHOD.md |
| `period` | `{from, to}` | first and last day of the work counted for the edition |
| `published` | date or null | publication date; null before publication |
| `built` | date | the day the files were built |
| `previous_edition` | string or null | the edition before this one |
| `early` | boolean | true for an edition that came out before its month |
| `totals` | object | see Totals |
| `scale` | array | `{id, name}` for each letter in use, best first |
| `groups` | array | `{id, name, main, works[]}`: the columns of the main table (`main` true) and the group Running; `works` are the kinds of work a group covers |
| `works` | array | `{id, group, name, what}`: the kinds of work, the vocabulary of the columns and of the cases |
| `models` | array | see Model; the ranked models first, in the order of my ranking |
| `cells` | array | empty in schema 4: letters are given by column only |
| `picks` | array | `{group, models[]}`: who holds the top letter in each column |
| `cases` | array | see Case |
| `volume` | object | see Volume |
| `compare` | object or null | see Compare |
| `votes` | object | see Votes |
| `reviewers` | object | see Reviewers |
| `notes` | object | `author`, `review_from`, `changes`, `changelog`, `lead_note`, optional `postscript`; in a draft with cases not yet verified by a second reader, `unverified` counts them |

Models in the rating contributed drafts, not their own rows. The final order, every letter and the published texts are Ouroboros's decisions. Nothing in the data computes a letter or a place.

## Totals

| Field | Meaning |
| --- | --- |
| `models` | how many models have each status: `rated`, `candidate`, `newcomer`, `first_look`, `earlier` |
| `work_types`, `work_groups`, `columns` | kinds of work, groups of work, and columns of the main table |
| `letters` | `{main}`: letters in the main table |
| `cases` | dated cases in the edition |
| `tokens` | the total since my first record: `{status, read, lower_bound, read_display, written, written_display, since, definition}`; `read_display` and `written_display` are the short forms shown on the page, cut and never rounded up |

## Model

| Field | Meaning |
| --- | --- |
| `id` | stable identifier, for example `claude-opus-5.5`; the same version keeps its id across editions |
| `name`, `short` | display name and the short label used in the table |
| `family`, `maker` | model line and the company that makes it |
| `status` | `rated` (in the ranking), `candidate` (a candidate for a place), `newcomer`, `first_look` or `earlier` (an earlier version) |
| `rank` | place in my ranking, from 1; only for `rated` |
| `one_liner`, `verdict`, `trust`, `check` | only for `rated`: one sentence, the portrait, what I trust the model with, what I check after it |
| `groups` | only for `rated`: per column, `{grade, reason, ground, cases?}`. `cases` is optional and defaults to an empty list. `ground` is what the letter rests on: `carried`, `case`, `cards` (checked code reviews), `role`, `open` (preliminary) or `coordinator` (proposed by another model for the row of a model that led the edition) |
| `picks` | only for `rated`: ids of the columns where this model is my first pick |
| `flag` | optional short note shown beside the name, for example how little a model worked in the period |
| `was` | the same version in the previous edition: `{rank, letters}` when it was ranked (letters of the six columns in table order), or `{status}` when it stood apart; absent when it was not there |
| `note` | for a model apart: what I can say about it without a letter |
| `cases` | ids of the cases that mention the model; when absent, read as an empty list |

## Case

| Field | Meaning |
| --- | --- |
| `id` | number, stable inside the edition |
| `from`, `to` | `from` is the day or first day of a range; `to` is optional and, when present, is the last day of the range |
| `models`, `works` | the models of the case and the kinds of work it touches |
| `text` | what the task was, what the model did and what verified it, told without private detail |

## Volume

`{versions_with_work, versions_with_finished_work, versions_leading_only, finished_works, tokens}`: how many versions did work for me in the period, how many of them delivered at least one finished work and how many only led tasks; the finished works; and the tokens of the period `{read, lower_bound, cache_read, cache_share, written, read_display, written_display}`.

## Compare

`{with, comparable, rows[], left[], bullets[]}`. `rows`: for each ranked model `{model, rank, letters, was}`, where `was` is as in Model. `left`: models ranked in the previous edition and not now, `{model, name, was, now, successor}`. `bullets`: what changed, in words. `notes.changes` lists the same moves one by one: `{model, rank_from, rank}`, `{model, group, from, to}`, `{model, new, rank}`, `{model, left}`.

## Votes

| Field | Meaning |
| --- | --- |
| `source` | `final` for the counter's public count |
| `complete` | true when every version on the roster has a final state |
| `participation` | the counter's totals: `invited`, `returned`, `counted_ballots`, `with_pairs`, `with_annotations`, `status_counts`, `prior_table_counts` |
| `voters` | one row per version asked for a ballot, in the order shown: `{model, status, prior_table, counts, identity?, requested_label?}`. `status`: `counted`, `partially_counted`, `excluded`, `not_sent`, `no_response`, `access_unavailable`, `model_error`, `format_invalid`, `evidence_unavailable`, `model_mismatch`, `duplicate_response`, `delivery_unknown`, or `pending` while the version is not in the count yet. `prior_table`: `seen`, `not_detected` or `unknown` for the previous edition's table in its context. `counts`: `{pairwise_assigned, pairwise_counted, annotations_assigned, annotations_counted}` on every row, with zero for an unassigned or uncounted kind. When an identity audit is present, `identity` is `confirmed` (model confirmed by the engine) or `requested_route_only` (no confirmation; `model` is the intended recipient's stable ID and `requested_label`, if present, states its exact requested label, not an observed identity). It applies to the whole ballot, including source checks, and does not determine its counting status. |
| `pairs` | one entry per pair of answers: `{description, models, labels, assigned, counts {first, second, tie, abstain}, column, winner, letters, verdict, unverified_counted, weighted_differs, explain}`. `models` are the two authors in the counter's fixed order; `column` the column of my table the pair belongs to; `letters` my two letters there; `verdict` is `agree`, `disagree`, `same_letter`, `no_majority` or `no_column`; `explain` my sentence where the vote goes against my letters |
| `annotations` | claims checked against a source: `{description, assigned, counts {supported, contradicted, insufficient}, unverified_counted?, claim?, established?}`. The three counts sum to the number of counted answers. `unverified_counted`, when present, says how many of those answers have only a requested-route identity. The optional bilingual `claim` states what was checked; `established` gives the answer beside the counts. |
| `counting_note` | optional bilingual note on how the ballots were counted and checked |

## Reviewers

`{wave, drawn, excluded_before_reading, assessed, control_without_findings, works_with_properties, properties, pairs, invited, answers, answers_bound, no_answer, recheck, unbound, rows[], modes_note, advice[], examples[]}`. `wave` names the selected review round: `first` means the first answered round.

`recheck`: `{reviews, remarks, confirmed, open, refuted, first_confirmed, new_confirmed, readers[]}`, the second reading of the factual remarks by two readers from families that had no seat in the reviews; `readers` are their ids, `first_confirmed` the problems confirmed before it and kept by it.

Each row: `{model, checked, confirmed, open, refuted, modes, named, chances, alone, invited, answered, no_answer, no_answer_kind, read_only_sessions}`. `checked` are the version's factual remarks in the recheck, each counted once for the version, and `confirmed`, `open` and `refuted` how they came out. `modes`, only where a version merges registered modes, gives the answers of each mode in the recheck. `named` of `chances` are the problems confirmed first that it named on the same working copy and assignment; `named` is null when the reviewer had no comparable seat. `invited` and `answered` are its seats in the assessed reviews.

`unbound` `{requested, answers, mentions, raised, confirmed}` counts the answers whose model the engine did not confirm and their remarks: they count for nobody. `modes_note` is a bilingual text. Each `advice` entry is `{who: [model ids], text: {en, ru}}`. Each `examples` entry is `{models: [model ids], title: {en, ru}, text: {en, ru}}`.

## Schema 3 (edition 2026.11)

Schema 3 gave a letter for every kind of work: `cells` held one entry per ranked model and kind of work `{model, work, grade, unrated, note, use, stats, cases}`, `works` carried telemetry of use (`total`, `models_tried`, `split`, `other`, `unrated`), `use` held the bands of use, model statuses were `rated`, `earlier`, `trial` and `untried`, and `totals.letters` counted `{main, works, own, unrated}`. It had no `volume`, `compare`, `votes` or `reviewers`.

## Compatibility

New optional fields may appear without a new schema version. Renaming or removing a field, or changing what a field means, raises `schema`. Schemas 1 and 2 belonged to drafts that were never published.
