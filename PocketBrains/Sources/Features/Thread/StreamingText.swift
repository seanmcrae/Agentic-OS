import SwiftUI

/// Records when each stretch of streamed text arrived, so the renderer can
/// age every glyph individually.
struct ChunkClock {
    /// (cumulative character count, arrival time) — monotonically increasing.
    private(set) var marks: [(chars: Int, at: TimeInterval)] = []

    mutating func record(count: Int) {
        let now = Date.timeIntervalSinceReferenceDate
        if count < (marks.last?.chars ?? 0) { marks.removeAll() } // new turn
        guard count > (marks.last?.chars ?? 0) else { return }
        marks.append((count, now))
    }

    func arrival(forGlyph index: Int) -> (at: TimeInterval, indexInChunk: Int) {
        var previousChars = 0
        for mark in marks {
            if index < mark.chars { return (mark.at, index - previousChars) }
            previousChars = mark.chars
        }
        return (marks.last?.at ?? 0, 0)
    }
}

/// The condensation reveal: each glyph arrives 3pt low and blurred, sharpens
/// and settles over ~240ms, and throws a brief specular glow as it lands.
/// Latency becomes choreography — thinking time reads as light, not lag.
struct CondensationRenderer: TextRenderer {
    var now: TimeInterval
    var clock: ChunkClock

    private let duration: TimeInterval = 0.24
    private let stagger: TimeInterval = 0.014

    func draw(layout: Text.Layout, in context: inout GraphicsContext) {
        var glyphIndex = 0
        for line in layout {
            for run in line {
                for glyph in run {
                    let (arrivedAt, chunkOffset) = clock.arrival(forGlyph: glyphIndex)
                    let age = now - arrivedAt - Double(chunkOffset) * stagger
                    let p = max(0, min(1, age / duration))
                    if p >= 1 {
                        context.draw(glyph)
                    } else if p > 0 {
                        var settling = context
                        settling.opacity = p
                        settling.translateBy(x: 0, y: (1 - p) * 3)
                        settling.addFilter(.blur(radius: (1 - p) * 3.5))
                        if p > 0.45 {
                            // One-time landing glow that fades as it sharpens.
                            settling.addFilter(.shadow(
                                color: .white.opacity((1 - p) * 0.9), radius: 2.5))
                        }
                        settling.draw(glyph)
                    }
                    glyphIndex += 1
                }
            }
        }
    }
}

/// Agent text. Live text animates through the condensation renderer at the
/// display's refresh rate; settled history renders as plain Text (zero cost).
struct StreamingText: View {
    let text: String
    var isLive: Bool = false

    @State private var clock = ChunkClock()

    var body: some View {
        Group {
            if isLive {
                TimelineView(.animation) { timeline in
                    Text(text)
                        .textRenderer(CondensationRenderer(
                            now: timeline.date.timeIntervalSinceReferenceDate,
                            clock: clock))
                }
            } else {
                Text(text)
            }
        }
        .font(Type.body)
        .lineSpacing(4)
        .foregroundStyle(Paper.primary)
        .onChange(of: text, initial: true) { _, newValue in
            clock.record(count: newValue.count)
        }
    }
}
