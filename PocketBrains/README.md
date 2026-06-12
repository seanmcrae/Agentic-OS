# PocketBrains

Tasks, projects, notes and the knowledge that connects them — run by an
agent that lives entirely on your phone. One conversational surface, real
structured data underneath, zero network calls.

![iOS 26](https://img.shields.io/badge/iOS-26-black) ![On--device](https://img.shields.io/badge/inference-on--device-gold) ![Offline](https://img.shields.io/badge/works-offline-green)

## What it is

- **The thread is home.** Talk naturally — "remind me to send the invoice
  Friday", "what's blocking the website redesign", "summarize my notes from
  yesterday", "link this note to Q3" — and the agent does the work with
  thirteen tools against a local SwiftData store.
- **Pull the thread down** and the conversation recedes into a floating
  glass tile while three spaces fan in behind it: **Today** (ranked agenda),
  **Projects** (magazine-style dossiers), **Knowledge** (an orbital
  constellation of notes, links and ideas — not a force-directed hairball).
  One scalar drives the whole transition; it is gestural and interruptible.
- **Local AI only.** Apple Foundation Models (iOS 26) with native tool
  calling and streaming; optional MLX backend running quantized Qwen3 for
  devices without Apple Intelligence; a deterministic intent parser as the
  always-works floor. On-device sentence embeddings power semantic note
  search. Airplane Mode changes nothing.
- **Deep Glass design system.** Custom Metal shaders: SDF refraction with
  chromatic edge split, a specular bead that tracks device motion, aurora
  noise fields for the agent's presence, film grain on the ink backdrop.
  Streaming text *condenses* — per-glyph blur→sharp reveal with a landing
  glow, drawn by a custom iOS 18 `TextRenderer`.

## Build

Requires a Mac with Xcode 26 (iOS 26 SDK).

```bash
brew install xcodegen
cd PocketBrains
xcodegen generate
open PocketBrains.xcodeproj
```

Run on an iOS 26 simulator or device. On hardware with Apple Intelligence
the Foundation Models brain engages automatically; everywhere else the app
falls back gracefully (see `docs/ARCHITECTURE.md`). To enable the MLX/Qwen3
backend, follow the comments in `project.yml`.

## Documentation

| Doc | Contents |
|---|---|
| `docs/ARCHITECTURE.md` | Layer map, agent/tool design, backend fallback chain |
| `docs/DESIGN.md` | The full design system: color logic, type scale, motion language |
| `docs/VERIFICATION.md` | What has been verified, what to check on first build, known API hedges |

## Source layout

```
Sources/
  App/            entry, app model, the thread↔spaces zoom shell
  DesignSystem/   tokens, Deep Glass material, components, haptics, motion light
  Shaders/        LiquidGlass.metal — glass, aurora, sweep, grain
  Data/           SwiftData models, unified services (single write path), seed
  Intelligence/   ModelBackend protocol, FoundationModels / MLX / fallback, embeddings
  Agent/          ToolBox (13 tools), registry, orchestrator, date parsing
  Features/       Thread, Today, Projects, Knowledge, Privacy
```
