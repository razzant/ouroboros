/** Dependency-free JSDoc mirror of `ouroboros/gateway/project_folder_contracts.py`: a Project's working folder as
 *  `/api/projects/{project_id}/folder` reads and changes it (producer: `ouroboros/project_folder.py`). */
/**
 * @typedef {Object} ProjectFolderWorkItem
 * @property {string} task_id
 * @property {string} name
 * @property {boolean} system_repo  // its recorded intent is Ouroboros's own repository: a chosen folder would take it from there
 */

/**
 * The room's queued roots by what a folder change means for them.
 * @typedef {Object} ProjectFolderWork
 * @property {boolean} observed  // false: the queue or the bindings could not be read, so nothing is counted
 * @property {number} kept  // admitted work: keeps the resources it was admitted with, through Resume and Continue
 * @property {ProjectFolderWorkItem[]} moved  // moved here, not restarted since: takes the room's folder at its next start
 * @property {number} moved_total  // every such work; `moved` names the first 20
 */

/**
 * One room's folder choice and what one look at it found. `actions` is empty for a folder in place.
 * @typedef {Object} ProjectFolderView
 * @property {string} project_id
 * @property {string} name
 * @property {number} chat_id
 * @property {string} folder  // the registered address, '' when none
 * @property {'usable'|'missing'|'unusable'|'unset'|'none'} state  // unset: no choice made; none: the recorded No folder
 * @property {string} reason  // why a set folder is missing or unusable, else ''
 * @property {?{at: string, by: string}} no_folder  // the recorded choice
 * @property {string} provenance
 * @property {Array<'clear'|'choose'>} actions
 * @property {string} expected  // sent back with a change; a choice that moved meanwhile answers 409
 * @property {ProjectFolderWork} work
 */

/**
 * `POST`: `clear` records No folder; `choose` attaches the existing folder `path`.
 * @typedef {Object} ProjectFolderRequest
 * @property {'clear'|'choose'} action
 * @property {string} expected
 * @property {string=} path
 */

/**
 * The change as appended to `logs/events.jsonl`; `recorded` false when that append failed.
 * @typedef {Object} ProjectFolderReceipt
 * @property {string} ts
 * @property {string} type
 * @property {string} project_id
 * @property {string} actor  // owner or ouroboros:<task id>
 * @property {'clear'|'choose'} action
 * @property {Object} old  // working_dir, no_folder, state, reason
 * @property {Object} new  // working_dir, no_folder
 * @property {boolean} recorded
 * @property {string=} note
 */

/**
 * @typedef {Object} ProjectFolderResponse
 * @property {?ProjectFolderView} folder  // null only beside a receipt: changed, the look after it failed (folder_error)
 * @property {ProjectFolderReceipt=} receipt
 * @property {string=} folder_error  // why that look failed; the change stands: read the folder again, never re-send
 */

export {};
