# Model selection and response on a helper card

Three WebKit captures from the exact tree merged in [PR #4090](https://github.com/razzant/ouroboros/pull/4090), `6da07ae5e9e3c7fef6e092b175ec3eee5c2a1e50`. The isolated server, queue, cancellation, Settings, history, and renderer were real. The coordinator and Codex endpoint used synthetic responses. The two copies of the scheduling notice came as log and chat frames for the same event.

[Queued](queued.png) · [Cancelled before dispatch, reopened](cancelled.png) · [Different model answered, reopened](answered.png)

[Root clock, separate renderer fixture](root-clock-replay.png): on the correction build a synthetic scheduling record with no observed start reads `since 17:26`. This fourth frame is not part of the continuous two-helper run.
