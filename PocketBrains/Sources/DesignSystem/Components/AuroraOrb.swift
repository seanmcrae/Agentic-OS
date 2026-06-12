import SwiftUI

/// The agent's presence: a breathing field of liquid light, shader-driven.
/// `energy` 0…1 maps idle → thinking. Time spent waiting on the model makes
/// the orb *more* alive, never a spinner.
struct AuroraOrb: View {
    var energy: Double = 0.3
    var tint: Color = lumen
    var size: CGFloat = 44

    @State private var bornAt = Date()

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 40.0)) { timeline in
            let t = timeline.date.timeIntervalSince(bornAt)
            // Breath: a slow sine whose depth scales with energy.
            let breath = 1.0 + sin(t * (1.2 + energy * 2.2)) * (0.03 + energy * 0.06)
            ZStack {
                Rectangle()
                    .fill(.clear)
                    .colorEffect(
                        ShaderLibrary.aurora(
                            .float2(CGSize(width: size, height: size)),
                            .float(t),
                            .color(tint),
                            .float(0.55 + energy * 0.65)
                        )
                    )
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [tint.opacity(0.5 + energy * 0.4), .clear],
                            center: .center, startRadius: 0, endRadius: size * 0.5
                        )
                    )
                    .blur(radius: 2)
                    .scaleEffect(breath)
            }
            .frame(width: size, height: size)
        }
    }
}
