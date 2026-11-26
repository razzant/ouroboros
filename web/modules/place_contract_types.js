/** Browser-only views of Place identity, status and origin. */

/**
 * The connector's declared account; the client derives owner from confirmed user accounts.
 * @typedef {Object} PlaceIdentityAccount
 * @property {string} id
 * @property {string} label
 * @property {'user'|'bot'} kind
 * @property {boolean=} owner
 */

/**
 * Optional accounts used to write and read; absent means undeclared.
 * @typedef {Object} PlaceIdentity
 * @property {PlaceIdentityAccount=} sends_as
 * @property {Array<PlaceIdentityAccount>=} reads_as
 */

/**
 * The declared source.url response; identity and labels are optional on older connectors.
 * @typedef {Object} PlaceStatusPayload
 * @property {string} text
 * @property {string} tone
 * @property {PlaceIdentity=} identity
 * @property {Object<string,string>=} labels
 */

/**
 * The host's `place_origin` on a work frame or census row. `wakes` lists matched kinds in place order;
 * `labels` gives their connector's words when known. `routed` means no kind matched and the conversation's
 * works were read instead. An absent or empty `actor` means the author could not be identified.
 * @typedef {Object} PlaceOrigin
 * @property {string} place_id
 * @property {string=} place_label
 * @property {string=} provider
 * @property {{id?:string,label?:string,owner?:boolean,bot?:boolean}=} actor
 * @property {string[]} wakes
 * @property {Object<string,string>=} labels
 * @property {boolean=} routed
 */
