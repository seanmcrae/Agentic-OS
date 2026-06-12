# Verification status

Honest accounting of what has and hasn't been verified, and the exact
checklist for the first build on a Mac.

## Context

This codebase was authored in a Linux container with **no Swift toolchain,
no Xcode, and no iOS simulator available**. Every file has been carefully
reviewed for API correctness against the iOS 26 SDK surface, but **nothing
here has been compiled or run yet**. Expect a short first-build fix pass —
the items most likely to need a touch are listed below, each with its
intended fix.

## Verified by construction

- One write path: agent tools and views both mutate through `DataServices`;
  `@Query`-driven views update automatically after any agent action.
- The zoom transition is driven by a single scalar (`AppModel.zoom` +
  in-flight drag delta) — properties cannot desynchronize.
- Backend fallback chain never leaves the user without a working agent:
  FoundationModels → (optional) MLX → deterministic intent parser.
- All Metal arithmetic keeps `half`/`float` types segregated (MSL does not
  implicitly convert in `mix`/vector ops).
- No spacing, radius, color, or font outside the token system.

## Known API hedges (check these first on Mac)

1. **FoundationModels `Tool.call` return type** — written as
   `async throws -> String`. If the SDK in use requires `ToolOutput`,
   wrap each return: `ToolOutput(detail)`. One-line change per tool
   (13 sites in `FoundationModelBackend.swift`).
2. **`LanguageModelSession(tools:instructions:)`** — instructions passed as
   `String`. If the initializer wants an `Instructions` builder, change to
   `LanguageModelSession(tools: t) { AgentVoice.instructions() }`.
3. **`streamResponse(to:)` element type** — treated as cumulative `String`
   snapshots via `String(describing:)`, which is robust either way; if the
   element exposes `.content`, prefer that.
4. **`@Generable` optional properties** (`let due: String?`) — supported per
   WWDC25/26 docs; if a seed rejects optionals, make them non-optional with
   "" sentinel and `@Guide` text "empty string if unknown".
5. **Self-referential SwiftData relationship** (`TaskItem.blockedBy` /
   `blocking` explicit inverse) — if the macro rejects the keypath, fall
   back to storing `blockedByIDs: [UUID]` and resolving via fetch.
6. **`Text.Layout` iteration** in `CondensationRenderer` follows the WWDC24
   "Create custom visual effects" sample (`layout → line → run → slice`).
7. **MLX backend** is opt-in and untested by definition (needs the package
   + `-D POCKETBRAINS_MLX`); the `mlx-swift-examples` generate API moves —
   treat `MLXBackend.generate` as the adaptation point.

## First-run checklist (simulator/device)

Functional:
- [ ] Seeded content appears; thread welcome shows; suggestions fire.
- [ ] "Remind me to send the invoice Friday" → tool card → task visible in
      Today with Friday chip (works even in fallback mode).
- [ ] "What's blocking the website redesign?" → blocker named ("Build
      homepage in Framer" waiting on copy + photography).
- [ ] "Summarize my notes from yesterday" → notesFrom tool fires.
- [ ] Completing a task in Today updates project progress ring.
- [ ] Knowledge: tap node → refocus animation; tap focus → note opens;
      pinch/pan; link created by agent appears as new edge.
- [ ] Airplane Mode: everything above still works.

Design (record simulator captures, step frame-by-frame):
- [ ] Horizon pull: thread recedes smoothly, no corner-radius pop at
      dock/undock boundaries; interrupt mid-gesture both directions.
- [ ] Condensation reveal: glyphs never flash at full opacity before
      settling; landing glow subtle; 60fps during streaming.
- [ ] Glass speculars track device tilt (hardware only); no shimmer strobe.
- [ ] Tool cards slide from the orb without layout jump in the scroll.
- [ ] Project card → dossier hero expansion holds the hue and radius
      concentricity; no double-render flash of the matched pair.
- [ ] TabView page swipe vs. constellation pan: if they fight, gate the
      pan gesture to `scale > 1.05` or swap TabView for a custom pager.

Performance:
- [ ] Many glass chips on screen (Today with 10+ tasks) holds 60fps on an
      A17-class device; if not, drop `GlassSheen` timeline to 10Hz for
      `depth < 0.5` surfaces.
