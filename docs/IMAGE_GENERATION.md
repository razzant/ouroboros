# Image creation and editing

`list_image_models` exposes source-qualified evidence; the model chooses a connected
route within the task budget. Explicit source/model choices stay on that route.
There is no image settings slot, conversational-model inheritance, deterministic
ranking, automatic retry or subscription-to-API fallback. Credentials and capability
docs/catalogs prove neither access, billing eligibility nor quota. Unfamiliar
explicit models remain unknown and may be submitted to their selected source.

`generate_image(prompt, model, images?, options?, credential_profile_id?)` writes
ordinary immutable task artifacts. `model` is `openai::<id>`, `openrouter::<id>`,
`zai::<id>`, or the subscription route spelling `claudexor::<source>=<id>`.
Each edit input is `{root, path}` using the existing file-resource roots; originals
are read with the same access and protected-file policy as ordinary reads. Inputs
allow 16 PNG/JPEG/WebP files, 32 MiB each and 64 MiB together. The provider
may impose narrower limits. Unspecified native options are omitted. Direct-API
options pass through unchanged except for tool-owned prompt/input and buffered-result
fields; the provider validates model-specific and new parameters.

Responses list paths, MIME, size and SHA-256, never inline base64. The API envelope is
an `image_response` artifact: download/decode recovery needs no paid resubmission.
Images are `generated_image` artifacts, collected at task finalization and copied to
the parent by the existing child-result consumer. Artifacts use the execution drive;
accounting uses the canonical budget drive. Generation calls no private photo-send
helper. Use `view_image` to inspect, then `send_photo` or `send_file` for wanted delivery.

An ordinary acting child can call discovery, generation and its own recovery. A readonly child can
only discover. Observe also blocks generation/recovery but keeps discovery.
Presence requires an exact tool grant and each edit input's resource
grant; generation grants no delivery tool. Network-disabled tasks cannot use the
image tools. Edit inputs use original file references from the existing attachment
or artifact flow; vision preprocessing is owned separately.

Direct routes resolve their base URL, credential and headers through the shared
LLM target resolver, and use the shared proxy/trust-aware transport.
Each submission uses `execute_physical_attempt` once. Provider-reported USD cost is
recorded when present; other image usage is retained as evidence, with cost unknown.
Unsettled API accounting adds a warning to retained outputs: use `recover_image`
with the original request id, without another generation.
Image tokens are never passed through text-model pricing. Existing task-tree budget
fences apply, including their existing treatment of unknown prices; no exact price
or upper-bound guarantee is invented. Timeouts and uncertain server failures produce
`IMAGE_OUTCOME_UNKNOWN` and a durable event. A 429 is `IMAGE_RATE_LIMITED`, with the
provider code and retry fact; this client never alters text-lane quota state.
Output/download/storage failure is `IMAGE_RESULT_INCOMPLETE`; retained files stay
available. The model decides the next step.
If private response custody fails after a direct provider response, ordinary
response/image artifacts are still retained and the custody failure is disclosed.
Access and OS permission refusals are `PATH_BLOCKED`, other file I/O faults are
`TOOL_ERROR`, and malformed inputs are `TOOL_ARG_ERROR`; none submits a request.

`recover_image(request_id)` reads only this task's recorded attempt. Direct recovery
uses the retained envelope for decoding/download/storage and original accounting.
Subscription recovery uses the existing consumer: exact operation GET and result/ACK,
no create, upload or reservation. It reconciles the original ledger revision, even
after administrative abandonment. Terminal maintenance uses the same receipt for
accounting and leaves image ACK to the artifact consumer. A lost create reply without
a recorded operation id stays unknown; no history scan guesses it.
After a bare `TOOL_TIMEOUT` or any call without a request_id, both tool schemas direct
`recover_image()` (no arguments) before any new generation. It lists the current task's
image attempts from the ledger only (`usage_store.Txn.task_attempt_page`: the task index,
newest first by first record, 20 per page; `next_after_request_id` continues with older
ones). The reservation precedes any possible dispatch, and the timed-out worker may still
retain, settle and ACK its response later. Earlier tasks are not listed and another task's
request_id grants nothing. Listing sends, settles and acknowledges nothing and picks no
attempt; an empty page says nothing about a still-running call, and an unreadable store
raises instead of listing nothing. An explicit request_id whose direct response is not
retained yet answers that the call may still be in flight; nothing is resent.
Both paths retain partial outputs and warnings. Registered artifacts count as durable
effects even when their enclosing tool result reports incomplete output. Public cost
fields distinguish `actual_cost_usd` from `estimated_cost_usd`; missing evidence is null.
Uncaught builtin `UsageAccountingError`s cross registry/executor boundaries: `BudgetExceeded`
uses `loop.py`'s budget rail; other accounting errors re-raise from
`loop_round_limits._loop_exit_after_exception` with loop evidence. Pause/Stop controls
retain their rails.

## Direct API contracts

Documentation checked on 2026-10-09; implementation checks use fixtures, not live
accounts. These anchors describe published interfaces, not tested account access.
Re-check each exact route before changing a contract: a stale positive gets
requests refused, a stale negative hides a capability.

| Source | Operation | Wire shape | Result |
| --- | --- | --- | --- |
| OpenAI | `POST /v1/images/generations` | JSON `model`, `prompt`, optional native options | `data[].b64_json`, optional `usage` |
| OpenAI | `POST /v1/images/edits` | JSON adds `images: [{image_url: dataURL}]` | Same |
| OpenRouter | `POST /api/v1/images` | JSON `model`, `prompt`; editing adds `input_references: [{type: "image_url", image_url: {url: dataURL}}]` | `data[].b64_json`, `media_type`, optional `usage.cost` |
| Z.ai | `POST /api/paas/v4/images/generations` | JSON `model`, `prompt`, optional `size`, `quality`; text only, one output | `data[0].url` |

OpenAI GPT Image uses base64 output without `response_format`. JSON edit inputs
retain the exact original bytes. `/v1/models` has no capability or pricing metadata;
the tool's documented GPT Image entries are positive evidence, not an allowlist.
Sources: [generation](https://developers.openai.com/api/reference/resources/images/methods/generate),
[editing](https://developers.openai.com/api/reference/resources/images/methods/edit),
[model catalog](https://developers.openai.com/api/reference/resources/models/methods/list),
[pricing units](https://developers.openai.com/api/docs/pricing).

OpenRouter's dedicated `/api/v1/images/models` catalog exposes supported parameters
and endpoint links. Endpoint-specific options pass through unchanged; only fields
that replace tool-owned inputs or buffered result custody are refused locally.
`input_references` is explicit edit evidence; absence of a
capability descriptor remains unknown. The general chat `/models` image modality
alone does not prove support for this dedicated endpoint. Output bytes may be PNG,
JPEG, WebP, GIF or SVG; file delivery and viewing retain their own format support.
A task file opened as SVG gets a `sandbox` CSP and `nosniff`: exact bytes and static
view stay, script and the application origin do not.
This policy does not block external SVG subresource fetches when opened.
Sources: [image API and discovery](https://openrouter.ai/docs/guides/overview/multimodal/image-generation),
[editing example](https://openrouter.ai/blog/tutorials/nano-banana/),
[error envelope](https://openrouter.ai/docs/api_reference/errors-and-debugging).

Z.ai documents `glm-image` and `cogview-4-250304`, with no image-edit input field.
Its temporary output URL is downloaded without forwarding the API credential.
Every destination, including each of at most five redirects, must resolve to public
addresses: the browser's shared address classifier excludes private, metadata,
link-local and reserved targets. This is a bounded result downloader, not a new
network-policy framework; DNS is checked before each hop but not pinned to the
transport's connection. Public HTTP(S) CDN URLs remain supported.
The image endpoint is PAYG even when `ZAI_PLAN=coding` is selected for text; discovery
labels this route as API billing. Quality is `hd|standard`. Cost remains unknown unless reported by the provider;
there is no hardcoded tariff. Sources: [image endpoint](https://docs.z.ai/api-reference/image/generate-image),
[GLM-Image](https://docs.z.ai/guides/image/glm-image).

## Subscription contract

The engine advertises the `requestKind=image` query on the existing
`POST /v2/model-operations` operation descriptor. Source capabilities come from
`GET /v2/model-sources?view=accounts`, under `imageGeneration`; its model hints and
provenance are preserved without implying account availability or entitlement.
The paired engine advertises `gpt-image-2` as an upstream-client presence hint and
a maximum of five edit inputs; neither turns the model hint into an allowlist.
Discovery passively reads the owned engine; explicit generation uses the same
owned-daemon ensure/admission seam as ordinary model calls. An unlisted explicitly
requested model is forwarded unchanged. Absent structural
support returns `CAPABILITY_UNAVAILABLE` before submission, without an API fallback.
Malformed advertised edit-limit/option/model fields also return unavailable,
not a guessed zero limit or a caller argument error.

`gateways/claudexor_images.py` mirrors the engine's `model-image.ts`:
`{source, model, account, prompt, images, options}` is uploaded through the existing
model-purpose resource flow. Edit rows carry exact `dataBase64`, MIME and decoded
`sha256`; options are `size`, `quality` (`low|medium|high|auto`), `background` and
integer `n` from 1 to 10. Account selection is Auto unless `credential_profile_id`
pins a managed account. Neither ordinary host credentials nor a text-model slot
is consulted.

The input payload is retained once in private CAS; its small manifest gains the
upload reference before the single create POST and the operation identity immediately
after acceptance. Polls retain only their small operation observations, not edit bytes.
An original worker's late status-only observation can overwrite a concurrently
recovered terminal private receipt. Ordinary response/image files survive and no
extra generation is sent, but repeated recovery may report unknown; use those files.
Status and result reads, cancellation and ACK use the existing model-operation
verbs. A failed control read after acceptance is unknown, including ConnectError;
only a terminal `dispatch.state=not_started` or the engine's pre-admission
`model_request_invalid` refusal releases the physical attempt. Lost replies never
create another generation. Task Stop/deadline requests cancel without claiming remote
completion. Tool-local expiry returns unknown with the original request_id; the task
can continue with recover_image.

Results carry `images[]`, `requestedModel`, `route`, usage and cost evidence. The
observed `route.model` stays null when the engine cannot observe it. Every ready
result is retrieved, including failed, cancelled and interrupted operations.
`outcome` is completed, failed or unknown; good siblings, usage and diagnostics,
including `response_rejected.context.rejectedIndexes`, survive partial failure.
The complete result bytes must match the engine custody size/digest. Each image's
MIME, size and SHA-256 are checked before artifact publication. ACK binds that
whole JSON digest only after private response custody, original ledger accounting
and ordinary image/response artifacts are durable. Storage, settlement or confirmed
not-started release failure withholds ACK and discloses pending accounting; explicit
recovery can finish the same operation. Unknown execution
with a ready receipt still exposes its unknown outcome: recording a receipt in the
ledger does not assert successful generation or invent a known cost.
ACK or image extraction failure never causes a new generation. Subscription cash
is unknown unless the engine supplies cost evidence; subscription is not priced as
free or valued through text token tariffs. Image admission and quota belong to the
engine's existing account resolver. Its quota projections use `requestKind=image`
(and optional model); omitted kind means text. Raw constraints may carry independent
`applies_to_request_kinds` and model scopes, so image refusals must not be converted
into text cooldowns. Core does not maintain a second quota cache or infer access
from text model catalogs. A provider 429 stays image-scoped; exact credential
identity and generation handling remain engine-owned.

Paired against engine contract SHA-256
`7b6104207c84d716ca3a94feb57eb0a88bb8310ad46573fed71728926ea5ae0c`,
request schema `898c4fcbe295fb0eca6ea6a149151a397ce044701cb8e11d4d4d13dd90798e75`
and result schema `76751e0c7024baf7d4a5a498b0cdd7f0bd6a230d7c1f5764eba6992a37da9f22`.

Engine release, runtime pinning and live account qualification are separate work;
local contract fixtures do not prove upstream availability.

## Fixture verification

`tests/test_imagegen_tool.py` covers exact generation/edit wire shapes, original
bytes, explicit routes, unknown capabilities, URL credential separation, partial
output and timeout behavior, real accounting/budget fences, registry restrictions,
Presence grants, and the normal acting-child-to-parent artifact flow.
`tests/test_imagegen_subscription.py` exercises the real gateway with synthetic
engine HTTP: digest-bound uploads, exact edit options, null observed model, ledger
settlement, one-operation unknown outcomes, cancellation and whole-result ACK.
`tests/test_imagegen_read_bounds.py` blocks real loopback engine, provider and URL reads
until the real executor times out, then finds each attempt with `recover_image()`.
Use the repository's `scripts/safe_test.py` launcher for these and related
registry, settings, artifact and vision suites. No live backend PASS is implied.
