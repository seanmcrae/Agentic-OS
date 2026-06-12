# PocketBrains — Design System

The bar: instantly eligible for an Apple Design Award. Every value below is a
rule, not a suggestion — break it nowhere.

## Material: "Deep Glass"

Not frosted cards. The UI is a single dark volume ("Ink") with light living
inside it. Glass surfaces are Metal-shaded: they *refract* what is behind
them, carry a specular bead on their top edge that tracks device motion, and
tint themselves from their content's identity hue. `.ultraThinMaterial` is
used only as the compositing base under the shader, never alone.

## Color logic

| Token | Value | Role |
|---|---|---|
| Ink 0 | `#0B0C0F` | app background (warm graphite black) |
| Ink 1 | `#13151A` | recessed wells |
| Ink 2 | `#1C1F26` | glass plate base |
| Paper | `#F4F2EC` | primary text (warm white, never pure #FFF) |
| Paper-60 / 38 | opacity steps | secondary / tertiary text |
| Lumen | `#E4C56F` | the single brand accent — candlelight gold |

Domain hues (used *only* as identity tints inside glass, never as fills):
Tasks **Citrine** `#D9A441` · Notes **Moss** `#8FAE8B` · Knowledge
**Glacier** `#7FB4C9` · Projects pick from a curated 8-hue wheel (no two
adjacent projects share a hue). **No purple-blue gradients anywhere.**
Gradients only appear as light (speculars, auras), never as decoration.

## Type scale (SF Pro; New York for project display headlines only)

| Token | Spec |
|---|---|
| display | NY 34/38 semibold, tracking -0.5 |
| title | SF 28/34 bold |
| heading | SF 22/28 semibold |
| body | SF 17/24 regular |
| callout | SF 15/20 medium |
| caption | SF 13/18 medium, tracking +0.2 |
| micro | SF 11/14 semibold, tracking +0.6, uppercase |

## Space & shape

4pt grid. Allowed spacings: 4 8 12 16 20 24 32 40 56. Screen gutter 20.
Corner radii (always `.continuous`): chip 10 · control 16 · card 22 ·
sheet 28. A child surface's radius = parent radius − inset (concentric).

## Motion language

One spring family: `snap` (0.32/0.86) for controls, `glide` (0.48/0.84)
for cards and layout, `drift` (0.85/0.92) for ambient/aurora.
Rules: nothing fades without also moving; nothing moves without physics;
every gesture is interruptible; transitions are driven by one scalar.
Haptics: `.soft` on touch-down of glass, `.rigid` on commit, success
pattern on task completion.

## Streaming = the heartbeat

Model latency is choreography, not lag:
- **Thinking**: an aurora orb breathes (Metal noise, drift spring); its
  intensity maps to elapsed time so waiting feels alive.
- **Reveal**: tokens condense — each glyph arrives blurred and 2pt low,
  sharpens and settles over 240ms with a one-time specular sweep.
- **Tools**: each call materializes as a small glass card sliding out of
  the orb, ticking to a checkmark when done. Work is visible.

## Information design

The thread shows prose + at most one structured card per action. Spaces
show structure. Never both at full density. Counts over lists, names over
IDs, relative dates ("Friday") over timestamps. Privacy is celebrated:
the Privacy space shows the network-silence badge — "Nothing leaves this
device" — as a designed moment, not a settings row.
