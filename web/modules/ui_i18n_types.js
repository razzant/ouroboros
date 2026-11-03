/** Dependency-free JSDoc mirror of `ouroboros.gateway.ui_i18n_contracts` — the `/api/ui/i18n*` envelopes
 *  `api_types.js` indexes; the memory file they describe is ouroboros/i18n_memory.py schema 1. */
/**
 * A language's profile inside its translation memory (ouroboros/i18n_memory.py).
 * @typedef {Object} UiI18nProfile
 * @property {string} label  // display name the owner typed or the generator chose
 * @property {string} instruction  // free-text brief for the generator (an invented language's description)
 * @property {'ltr'|'rtl'} direction
 * @property {string=} lexicon  // generator-written vocabulary/rules for rare or invented languages (file and export only)
 * @property {number=} lexicon_chars  // the GET carries the lexicon's size, not its text
 */

/**
 * One translation: `text`, or plural `forms` keyed by CLDR category.
 * @typedef {Object} UiI18nEntry
 * @property {string=} text
 * @property {Object.<string,string>=} forms
 * @property {'generated'|'owner'|'imported'} provenance
 * @property {string=} source_hash
 * @property {string=} model
 * @property {string=} at
 * @property {string=} attempt_id
 * @property {string=} pack
 * @property {string=} pack_version
 * @property {string=} context
 * @property {string=} source  // the English a code entry translated (the browser's reword check)
 */

/**
 * @typedef {Object} UiI18nStats
 * @property {number} entries
 * @property {number} generated
 * @property {number} owner
 * @property {number} imported
 * @property {number|null} stale  // null = the current English of code keys was not available
 * @property {number} pending  // queued misses awaiting the generator
 * @property {number} refused  // keys the generator gave up on (cleared by Regenerate)
 */

/**
 * One language present on disk, for the Settings select.
 * @typedef {Object} UiI18nLanguageSummary
 * @property {string} language
 * @property {string} label
 * @property {number} entries
 * @property {number} pending
 * @property {boolean} malformed
 */

/**
 * The translation generator's state in this server process (ouroboros/ui_translation.py).
 * @typedef {Object} UiI18nGenerator
 * @property {'idle'|'running'|'no_model'|'failed'} state
 * @property {string} error  // the last failure, secret-free; "" unless failed/no_model
 * @property {string} updated_at
 * @property {number} applied  // entries written since the process started
 * @property {string} language  // the language the worker last ran for ("" before any run)
 * @property {number} in_flight  // keys of the batch at the model right now (counted into stats.pending)
 */

/**
 * GET /api/ui/i18n and the body of a successful language POST.
 * @typedef {Object} UiI18nResponse
 * @property {boolean=} ok
 * @property {string} language  // BCP-47 tag; "" = not chosen (English source renders)
 * @property {boolean} chosen
 * @property {boolean} english  // not chosen or chosen English: entries are empty by construction
 * @property {number} revision
 * @property {UiI18nProfile|null} profile
 * @property {Object|null} plural_select  // {map: {"0": "other", …}, period: number|null}, written by the browser
 * @property {string[]|null} plural_categories
 * @property {Object.<string,UiI18nEntry>} entries
 * @property {UiI18nStats} stats
 * @property {string} updated_at
 * @property {string} memory_error  // nonempty when the stored file is malformed (English fallback in effect)
 * @property {UiI18nLanguageSummary[]} languages
 * @property {UiI18nGenerator} generator
 */

/**
 * POST /api/ui/i18n/language.
 * @typedef {Object} UiI18nLanguageRequest
 * @property {string} language  // BCP-47 tag ("" = not chosen, "en" = chosen English)
 * @property {string=} label
 * @property {Object=} profile
 * @property {Object=} plural_select  // Intl.PluralRules select(n) for 0..100 (+ period)
 * @property {string[]=} plural_categories
 */

/**
 * POST /api/ui/i18n/missing: strings the renderer could not translate.
 * @typedef {Object} UiI18nMissingRequest
 * @property {string} language  // must equal the install's current language (409 otherwise)
 * @property {Object[]} items  // {key, context?}; shape-filtered, at most 200 per call
 */

/**
 * @typedef {Object} UiI18nMissingResponse
 * @property {number} accepted
 * @property {number} dropped
 * @property {number} pending
 */

/**
 * A translation memory file (schema 1): what GET /api/ui/i18n/export downloads and POST
 * /api/ui/i18n/import accepts (needs a non-English `language`; `schema`, when sent, must be 1).
 * @typedef {Object} UiI18nMemoryDocument
 * @property {number=} schema
 * @property {string} language
 * @property {number=} revision
 * @property {string=} created_at
 * @property {string=} updated_at
 * @property {UiI18nProfile=} profile
 * @property {?Object=} plural_select
 * @property {?string[]=} plural_categories
 * @property {string=} glossary_hash
 * @property {Object.<string,UiI18nEntry>=} entries  // replaces the imported layer; owner entries shadow it
 * @property {Object.<string,UiI18nEntry>=} shadow
 * @property {Object.<string,Object>=} refused
 * @property {string=} pack
 * @property {string=} pack_version
 */

/**
 * @typedef {Object} UiI18nImportResult
 * @property {number} added
 * @property {number} replaced
 * @property {number} shadowed
 * @property {number} dropped
 */

/**
 * POST /api/ui/i18n/import: the memory's new revision.
 * @typedef {Object} UiI18nImportResponse
 * @property {boolean} ok
 * @property {string} language
 * @property {number} revision
 * @property {UiI18nImportResult} result
 * @property {UiI18nStats} stats
 */

/**
 * POST /api/ui/i18n/regenerate; an absent `language` means the current one.
 * @typedef {Object} UiI18nRegenerateRequest
 * @property {string=} language
 */

/**
 * @typedef {Object} UiI18nRegenerateResponse
 * @property {boolean} ok
 * @property {string} language
 * @property {number} revision
 * @property {number} removed
 * @property {UiI18nStats} stats
 */

/**
 * WS frame: the install's interface language was chosen (any client or transport); re-read GET /api/ui/i18n.
 * @typedef {Object} UiLanguageChangedOutbound
 * @property {'ui_language_changed'} type
 * @property {string} language  // the chosen BCP-47 tag ("" = not chosen)
 */

/**
 * WS frame: the translation memory of `language` reached `revision` (generated, imported, regenerated or an owner pin).
 * @typedef {Object} UiI18nUpdatedOutbound
 * @property {'ui_i18n_updated'} type
 * @property {string} language
 * @property {number} revision
 * @property {number=} applied  // entries the generator wrote in this batch; absent for import, regenerate and a pin
 */
