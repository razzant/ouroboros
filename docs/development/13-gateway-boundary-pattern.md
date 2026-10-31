# Gateway Boundary Pattern

This chapter owns dependency direction at the browser boundary: inbound routes enter through `ouroboros/gateway/`, outbound provider and harness adapters live in `ouroboros/gateways/` and carry no domain policy, and every interface, the builtin one included, is a client of one published contract (why the facade exists: ARCHITECTURE §1 "Gateway Boundary v1").

Frontend calls go through `web/modules/api_client.js`. Do not copy policy into an outbound adapter, do not promote the `gateway/host_service.py` callback surface into a general owner/task API, and do not require a class where established function owners already preserve the boundary. Enforcement: CHECKLISTS item 23 (`gateway_parity`) and `tests/test_gateway_parity.py`.

Publish routes/WS types via `gateway/endpoint_index.py`, `HTTP_SHAPES`/`WS_SHAPES`, contract TypedDicts and field-identical JSDoc twins; `/api/contract` derives from them. Update client-visible meaning in CLIENT_CONTRACT with code. Applied `ServedUI` alone selects optional setup/login pages; auth and settings retain their writers. Builtin recovery assets and the author kit stay builtin; never mix fallback files. Enforcement: `tests/test_client_contract_coverage.py`, `tests/test_gateway_client_contract.py`, `tests/test_client_contract_bindings.py`, `tests/test_ui_serving.py`.

Named skill chat ingress writes through the message-bus canonical writer before enqueue and correlates reads and cancel by `operation_ref` (custody, replay and the 409 rules: ARCHITECTURE §12). A cancel or intent write addresses the same installation root as the operation being read, and operation reads and cancel verify the complete source against actual task or direct-turn ownership; presentation annotations are discovery hints only. Tests: `tests/test_host_service_operation_identity.py`, `tests/test_host_service_operations.py`.

Dialogue storage changes preserve the public projection: `tests/test_dialogue_reader_golden.py`
compares history, log tails, task events and memory recall with the captured conversation in
`tests/fixtures/dialogue_readers.json`, normalizing only paths, witnesses and the snapshot token.
