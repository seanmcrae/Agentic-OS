# PocketBrains — Architecture

One conversational surface, real structured data underneath, all intelligence on-device.

## Layer map

```
┌─────────────────────────────────────────────────────────┐
│ Presentation (SwiftUI + Metal)                          │
│   Thread · Today · Projects · Knowledge · Spaces shell  │
└──────────────▲──────────────────────────▲───────────────┘
               │ observes                 │ renders
┌──────────────┴───────────┐  ┌───────────┴───────────────┐
│ Agent layer              │  │ Design system             │
│   AgentOrchestrator      │  │   Theme/Type/Motion       │
│   AgentTool registry     │  │   GlassSurface (Metal)    │
└──────▲───────────▲───────┘  └───────────────────────────┘
       │ calls     │ generates with
┌──────┴─────┐  ┌──┴────────────────────────────┐
│ Data layer │  │ Intelligence                  │
│  SwiftData │  │  ModelBackend protocol        │
│  Services  │  │   ├ FoundationModelBackend    │
│  Semantic  │  │   ├ MLXBackend (opt-in flag)  │
│  index     │  │   └ IntentFallbackBackend     │
└────────────┘  └───────────────────────────────┘
```

## Data layer (`Sources/Data`)

SwiftData models: `TaskItem`, `Project`, `Milestone`, `Note`, `Tag`,
`KnowledgeLink` (typed edges between any two entities), `ChatMessage`
(persistent transcript), `EmbeddingRecord` (vector cache).

`Services.swift` holds small, synchronous CRUD services
(`TaskService`, `ProjectService`, `NoteService`, `GraphService`).
**Both the agent tools and the structured views go through the same
services** — there is exactly one write path, so anything the agent does is
instantly visible in Today/Projects/Knowledge and vice versa.

`TaskItem` intentionally avoids the name `Task` (Swift Concurrency clash).

## Intelligence (`Sources/Intelligence`)

`ModelBackend` is a tiny protocol: `respond(to:context:tools:) ->
AsyncThrowingStream<AgentEvent>`. Events: `.thinking`, `.token(String)`
(cumulative snapshot), `.toolStarted`, `.toolFinished`, `.completed`.

Backends, tried in order at launch:

1. **FoundationModelBackend** — Apple Foundation Models framework (iOS 26).
   On-device ~3B model, native `Tool` protocol, streaming snapshots.
   The app's `AgentTool`s are bridged into `FoundationModels.Tool`
   wrappers with `@Generable` argument structs.
2. **MLXBackend** — compiled only with `-D POCKETBRAINS_MLX`. Runs a
   quantized open model (default `mlx-community/Qwen3-4B-Instruct-4bit`)
   via MLX Swift for devices without Apple Intelligence or when the user
   wants a larger model. Tool calling implemented via JSON tool-call
   prompting + parser.
3. **IntentFallbackBackend** — deterministic NL intent parser
   (NaturalLanguage framework + date parsing). No generation, but every
   tool still works ("remind me to send the invoice Friday" → createTask).
   Guarantees the product functions on any device and in the simulator.

`SemanticIndex` — `NLContextualEmbedding` sentence vectors, cached in
`EmbeddingRecord`, cosine search in memory. Used by the `searchNotes`
tool and the Knowledge view's "related" suggestions.

## Agent layer (`Sources/Agent`)

`AgentTool` is the app's own tool abstraction (name, description, typed
parameter schema, `run(args:) -> ToolResult`). One registry; each backend
adapts it to its native calling convention. Tools:

| Tool | Purpose |
|---|---|
| createTask / updateTask / completeTask | task CRUD, due dates, priority, deps |
| queryTasks | filtered queries ("what's due today", "blocked items") |
| createProject / updateProject / addMilestone | project management |
| projectStatus | blockers, progress, recent activity for a project |
| createNote / appendNote | capture |
| searchNotes | semantic + keyword search |
| linkItems | knowledge-graph edges between any entities |
| agenda | composes the Today payload |
| recall | query past chat history + activity log |

`AgentOrchestrator` (@MainActor, Observable) owns the active backend,
persists the transcript, exposes `send(_:)`, and republishes
`AgentEvent`s as UI state (streaming text, tool activity cards).

## Presentation (`Sources/Features`)

- **RootShell / Spaces** — the thread is home. A vertical pull (or the
  horizon button) scrubs a single `zoomProgress` value that drives the
  thread↔spaces transition: thread recedes into a glass tile, the three
  spaces (Today / Projects / Knowledge) fan in. Fully gestural and
  interruptible; one value drives every property so it can never desync.
- **Thread** — streaming text drawn by a custom iOS 18 `TextRenderer`
  ("condensation" reveal: per-glyph blur→sharp with a specular sweep),
  tool activity rendered as inline glass cards, thinking state is a
  living aurora orb (Metal), not a spinner.
- **Today** — agenda assembled by the same `agenda` tool the model uses.
- **Projects** — magazine-style cards, serif display headlines, per-project
  identity hue, progress ring, activity sparkline.
- **Knowledge** — custom orbital constellation layout (not a force
  hairball): focused node center, rings by graph distance, springy
  re-layout on focus change, Canvas-rendered, pinch/pan.

## Design system (`Sources/DesignSystem`, `Sources/Shaders`)

See DESIGN.md. The glass material is a Metal `layerEffect`
(`LiquidGlass.metal`): rounded-rect SDF refraction, edge specular driven
by CoreMotion light direction, content-aware tint. All shaders degrade
gracefully (plain material) if Metal compilation is unavailable.
