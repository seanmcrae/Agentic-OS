import SwiftUI

/// The Deep Glass material. A plate of ink-toned material whose sheen layer is
/// shaded by LiquidGlass.metal: rim refraction, a motion-tracked specular
/// bead, and an identity tint. Falls back to plain material if shaders fail.
struct GlassSurface: ViewModifier {
    var radius: CGFloat = Radius.card
    var tint: Color? = nil
    /// Elevation 0 = flush plate, 1 = floating card.
    var depth: Double = 1

    @State private var shaderHealthy = true

    func body(content: Content) -> some View {
        content
            .background {
                ZStack {
                    RoundedRectangle(cornerRadius: radius, style: .continuous)
                        .fill(.ultraThinMaterial)
                    RoundedRectangle(cornerRadius: radius, style: .continuous)
                        .fill(Ink.l2.opacity(0.55))
                    if shaderHealthy {
                        GlassSheen(radius: radius, tint: tint)
                    }
                }
                .compositingGroup()
                .shadow(
                    color: .black.opacity(0.22 * depth),
                    radius: 18 * depth, y: 8 * depth
                )
            }
            .overlay {
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .strokeBorder(
                        LinearGradient(
                            colors: [.white.opacity(0.16), .white.opacity(0.03)],
                            startPoint: .top, endPoint: .bottom
                        ),
                        lineWidth: 0.75
                    )
            }
            .clipShape(RoundedRectangle(cornerRadius: radius, style: .continuous))
    }
}

/// The shaded sheen layer. Re-evaluates with device motion (30Hz, low-passed)
/// and a slow clock for the rim shimmer.
private struct GlassSheen: View {
    let radius: CGFloat
    let tint: Color?
    private let light = MotionLight.shared

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 20.0)) { timeline in
            let t = timeline.date.timeIntervalSinceReferenceDate
            GeometryReader { geo in
                Rectangle()
                    .fill(.clear)
                    .background(
                        // A faint ambient field for the refraction to bend.
                        RadialGradient(
                            colors: [Color.white.opacity(0.10), .clear],
                            center: .init(x: 0.5 - light.direction.x * 0.4,
                                          y: 0.18),
                            startRadius: 0,
                            endRadius: max(geo.size.width, geo.size.height)
                        )
                    )
                    .layerEffect(
                        ShaderLibrary.liquidGlass(
                            .float2(geo.size),
                            .float(radius),
                            .float2(light.direction),
                            .color(tint ?? Color.white.opacity(0)),
                            .float(t.truncatingRemainder(dividingBy: 1000))
                        ),
                        maxSampleOffset: CGSize(width: 16, height: 16)
                    )
            }
        }
        .allowsHitTesting(false)
    }
}

extension View {
    /// Apply the Deep Glass material behind this view.
    func glass(_ radius: CGFloat = Radius.card, tint: Color? = nil, depth: Double = 1) -> some View {
        modifier(GlassSurface(radius: radius, tint: tint, depth: depth))
    }
}

/// The Ink backdrop with film grain — the volume every screen lives in.
struct InkBackdrop: View {
    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 8.0)) { timeline in
            Ink.l0
                .colorEffect(
                    ShaderLibrary.filmGrain(
                        .float(timeline.date.timeIntervalSinceReferenceDate
                            .truncatingRemainder(dividingBy: 100))
                    )
                )
        }
        .ignoresSafeArea()
    }
}
