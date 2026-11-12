// Browser twin of ouroboros/gateway/work_contracts.py: the /api/state work projection (v1).
// One row per work (a request with its Continue episodes and live members) and one status per room.
// Words come from the server's codes through tr(); a client never derives a state, mark or control.

/**
 * The owner's cited words, bounded for publication (CLIENT_CONTRACT §5); text stays authored.
 * @typedef {Object} CitedModelWords
 * @property {string} text
 * @property {boolean} clipped  // the durable citation is longer than this published span
 */
/**
 * A work's primary route and whose choice it is.
 * @typedef {Object} WorkPrimaryRoute
 * @property {string} model
 * @property {boolean} use_local
 * @property {CitedModelWords=} cited
 * @property {string} basis  // main | a role it follows | room | owner | owner_words | mind | fallback | place | circle
 */
/**
 * The route a work acts on and its chooser (`lead` on a work and on its activity row). A departure names the
 * primary it left: `basis` is then `fallback` or `mind`, with when (`since`) and why (`cause`).
 * @typedef {Object} WorkLead
 * @property {string} model
 * @property {boolean} use_local
 * @property {CitedModelWords=} cited
 * @property {string} basis
 * @property {string=} role  // a fallback's chain row (`fallback:<n>`, n from 0): tells apart two rows of one model
 * @property {WorkPrimaryRoute=} primary  // only when the acting route is not the primary
 * @property {string=} since  // an ISO instant in UTC
 * @property {string=} cause  // a short code or phrase: `switch_model`, a failure's code, `configured fallback`
 * @property {import('./client_contract_types.js').ChatModelRoute=} pending  // the route the next step uses, when it differs
 */
/**
 * One act on a work's model (POST /api/tasks/{task_id}/model): `set` gives it a model, `follow` returns it to Main;
 * `return`, `wait` and `keep` leave a fallback.
 * @typedef {Object} TaskModelRequest
 * @property {string} request_id  // a retry of the same act carries the same id
 * @property {"set"|"follow"|"return"|"wait"|"keep"} action
 * @property {string=} model  // `set` only
 * @property {boolean=} use_local  // `set` only
 */
/**
 * @typedef {Object} TaskModelResponse
 * @property {true} ok
 * @property {string} task_id
 * @property {"next_step"|"at_start"} applied  // a running work switches at its next step, a queued one starts on it
 * @property {WorkLead} lead
 */
/**
 * @typedef {Object} WorkReason
 * @property {string} code  // closed per release; an unknown code renders its text
 * @property {string} text  // the English sentence for this code
 * @property {number=} count
 * @property {Array<string>=} ids
 * @property {*=} since
 * @property {PauseAuthor=} by  // the declared author of a decided Pause
 * @property {Array<SleepWake>=} wake  // what a `sleeping` reason waits for: its clauses, joined by "or", time last
 */
/**
 * @typedef {Object} PauseAuthor
 * @property {string} actor  // owner | ouroboros
 * @property {string} text  // the words the press recorded, said verbatim; '' for a press without words
 * @property {*} at  // the press time
 */
/**
 * @typedef {Object} SleepWake
 * @property {string} kind  // task | task_or_writes | tasks | all_tasks | runs | service | reviewer | sender | mail | time
 * @property {string=} name  // a chosen task's or sender's work name, as the census holds it
 * @property {number=} count
 * @property {string=} at  // the chosen wake instant, said in the viewer's clock
 */
/**
 * @typedef {Object} LastResume
 * @property {*} at  // when it was asked
 * @property {{actor: string}} by  // owner | ouroboros | host (a Continue's own start), as its door bound it
 * @property {string} error  // the door's typed refusal
 * @property {string} text  // the door's own words after "Resume refused: "
 */
/**
 * @typedef {Object} WorkTail
 * @property {string} kind  // helper | run | review | model
 * @property {string} state  // live | unknown | held (a helper a sleeping episode chose, paused until its Resume)
 * @property {string} code
 * @property {string} text  // its English sentence, said beside the work's word, never as it
 * @property {number=} count
 * @property {Array<string>=} ids
 */
/**
 * @typedef {Object} WorkTurn  // what waits for the owner (gateway/work_contracts.py WorkTurn)
 * @property {string} kind  // question | open_question | action
 * @property {string=} work_id  // in a room's turn: the work that holds it; absent for a question no work holds
 * @property {string} task_id
 * @property {string=} quiz_id
 * @property {string=} wait_id
 * @property {string=} code  // open_question: delivery_unknown for an answer whose delivery is unknown
 */
/**
 * @typedef {Object} WorkControl
 * @property {string} action  // stop | stop_earlier | pause | resume | hurry | wrap_up | model
 * @property {string} target  // the exact task id the door acts on
 * @property {string=} scope  // single | cascade | work
 */
/**
 * @typedef {Object} WorkEpisode
 * @property {string} task_id
 * @property {boolean} live
 */
/**
 * @typedef {Object} WorkMembers
 * @property {Array<string>} ids  // every member task id (episode roots and helpers): a card of another id re-reads the state
 * @property {Object<string, Object>=} states  // the same server classification for each Activity row
 */
/**
 * @typedef {Object} WorkFocus
 * @property {string} text
 * @property {*} at
 */
/**
 * @typedef {Object} Work
 * @property {string} work_id
 * @property {*} room
 * @property {string} name
 * @property {string} kind  // direct | task
 * @property {string} project_id
 * @property {WorkEpisode} episode
 * @property {string} activity  // working | queued | waiting | paused | unknown
 * @property {string} transition  // none | finalizing | pausing
 * @property {string} call  // preparing | sending | thinking | working, when every working member agrees
 * @property {Array<WorkReason>} because  // the current episode's own reason (or, once it ended, nothing)
 * @property {Array<WorkTail>} tails  // what still runs beside or after it
 * @property {boolean} live  // something of the work actually executes now
 * @property {Array<WorkTurn>} turn
 * @property {string} release  // self | owner | mind | none
 * @property {WorkMembers} members
 * @property {Array<WorkControl>} controls
 * @property {*} since
 * @property {boolean} confirmed
 * @property {string} mark  // moving | still | pause | ring | none (docs/DESIGN.md "Status marks")
 * @property {Object=} stop  // current stop and any independent member stop
 * @property {Object=} refusal  // the continuation's last refusal, rechecked by Resume
 * @property {LastResume=} last_resume  // the latest refused Resume about its current control (also a member's)
 * @property {string=} outcome
 * @property {WorkFocus=} focus
 * @property {Array<Object>=} gaps
 * @property {string=} client_message_id
 * @property {WorkLead=} lead  // the model it runs on and whose choice that is
 */
/**
 * @typedef {Object} RoomStatus
 * @property {string} activity  // working | queued | waiting | paused | idle | unknown
 * @property {string} transition
 * @property {string} call
 * @property {Array<WorkTurn>} turn  // its works' items, and its own questions no work holds
 * @property {boolean} amber
 * @property {string} mark  // moving | still | pause | ring | none
 * @property {boolean} confirmed
 * @property {Array<string>} works
 */
/**
 * @typedef {Object} WorkSummary
 * @property {Object<string, number>} counts
 * @property {number} turn  // each work with a turn, and each standalone question no counted work carries
 * @property {boolean} confirmed
 */

export {};
