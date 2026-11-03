# A small independent client with its own setup and sign-in

This example implements the entry journey using only the published
[client contract](../../CLIENT_CONTRACT.md): sign in, read setup, configure a
keyless OpenAI-compatible endpoint, read core state and send a Main message over
WebSocket. It does not import the builtin SPA. It is a demonstration, not a full
replacement for its history, tasks, questions, files or reconnection handling.

Copy this directory to an absolute path outside the managed repository and data
directories. Select it with the existing `OUROBOROS_UI_DIR` setting and restart
the core. Open the core's usual address. With a network password and a nonlocal
visitor, the ordinary entry leads to this client's sign-in page; loopback keeps
its existing password bypass. The bootstrap page configures an endpoint the
operator supplies, not a bundled or automatically downloaded model.

The optional entry files are `onboarding.html` and `auth/index.html`.
`GET /onboarding` and `GET /auth/login` use those pages. Setup and Main code is
under the selected `/static/`; everything in `auth/` is intentionally public
through `/auth/`, including this example's common CSS. Keep secrets and private
files out of that directory. `login` and `logout` are reserved core operations.
Missing assets are errors, not requests to borrow similarly named builtin files.

The core remains the password/session and settings authority. The login page
posts JSON to the existing login operation and follows its sanitized `next`.
The setup page reads bootstrap without saving, then sends one completion request.
This deliberately minimal keyless example skips subscription discovery and
permits an empty review pool; that is not a recommendation for a production
configuration. Use the builtin wizard for the full configuration choices.

A successful completion hands the **actual receipt** to an available native
`onboarding_finished` bridge, or to the builtin/copied parent's same-origin
`ouroboros:onboarding-complete` convention. Other parent clients may choose their
own handoff. A plain browser navigates to Main only when no restart is required;
otherwise it shows that the mode still needs a restart. An unknown save outcome
does not cause an automatic second write. A later readiness read does not prove
that the earlier attempt saved a particular value.

`/onboarding?builtin=1` and `/auth/login?builtin=1` reach complete builtin
recovery pages. Their resources remain builtin. The standard macOS host also
offers its Help menu recovery action; this example does not manufacture native
menus or new recovery chrome for other shells.

For separate hosting, keep the topology in
[the separate-UI example](../separate_ui/README.md) distinct: with UI serving set
to `none`, the core has no selected directory from which to read these files.
The independent host owns its page routing and may use the same public APIs.
Do not claim that putting these files on another host changes core-owned routes.

For ordinary personal customization, editing the standard application's source
uses its existing self-development and managed Update path. Disjoint official
source additions can merge into that personal version; actual source or semantic
conflicts still need an author and verification. A separately released client
owns its own files and new bespoke screens. Neither arrangement needs a new UI
updater or a manually written change summary to preserve the published API.
