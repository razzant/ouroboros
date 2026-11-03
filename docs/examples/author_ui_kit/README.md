# Optional author UI controls

This extension contains two small recipes: a module tab and an independently
designed HTML route. Install/review/enable it through the ordinary extension
lifecycle. If you change its skill name, change the route prefix in widget.js.

The module asks its own `author-kit` route for the installed `web/ui.css` and
self-contained `web/modules/ui_primitives.js`: the core's builtin files, whatever
main interface `OUROBOROS_UI_DIR` selects. `OuroborosWidget.fetch` uses the
existing authenticated parent bridge. The application inserts the CSS as text
and imports the primitives through a frame-owned Blob URL, revoked after import.
The module's existing sandbox and CSP remain unchanged.

The HTML recipe reads the same assets from `request.app.state.repo_dir` for each
initial page request. It safely embeds JSON by escaping `<`, then loads its own
application and the primitives from local Blob URLs. Its nonce policy allows
these scripts/styles and the controls' data-URI icons without a `/static`
subrequest. It has no module bridge.
Adapt this template to your application's existing CSP instead of weakening it.

Both recipes use the existing `renderSafeField`, `collectSafeFieldValues`, and
`setInlineStatus` exports. Native buttons use `.btn.btn-default`; `.ouro-ui` opts
only the selected area into the common appearance. The helper also exports
`escapeHtmlAttr` and `normalizeTone`. No host shell or 26-type widget renderer is
loaded. Application layout, validation, data and operations remain author-owned.
The example adds its own padding and native `Canvas`/`CanvasText` surface so its
heading and loading/failure text remain readable even before the kit is available.

The kit is optional: omit it for a completely independent application, or add
your own CSS after it to override selected controls. A new mount receives the
currently installed styling; retained frames keep their mounted snapshot until
they are opened again. There is no theme polling or forced remount. A failed kit
request displays `Controls unavailable` in the application, leaving other widgets
unaffected. The example's Preview changes local status only and performs no write.

## Appearance boundary

Settings → Appearance controls the host document only. Module and route iframes
have independent roots: CSS variables and `ouroboros.theme` do not propagate.
The kit delivers a stylesheet snapshot, not the host's current choice; without
an author-set `data-theme="light"` it uses that stylesheet's default palette.
Modules may opt into the host's resolved palette with
`OuroborosWidget.onTheme(theme => { document.documentElement.dataset.theme = theme; })`.
Keep the returned unsubscribe in the module disposer. The bridge delivers
`light` or `dark` without applying styles or forcing a remount; route iframes
have no bridge. Fetching the kit again alone does not synchronize appearance.

Authors may set their own root's `data-theme`, track their document's media
query, or keep a fixed palette. `data-theme` selects only that palette: the
product's capsule button and chip recipes apply where the document root itself is
`.ouro-ui` (the app and its setup page), so an inner `.ouro-ui` area like this
example's keeps the base controls that later author CSS such as `.ouro-ui .btn`
overrides. Native `Canvas`/`CanvasText` colours follow the document's supported
`color-scheme`, not necessarily the host's pinned choice or the OS. Declare that
scheme deliberately if following the OS. The example keeps its own native-colour
surface readable before kit loading and on kit failure.
