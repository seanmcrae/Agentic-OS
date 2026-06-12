import SwiftUI

/// The signature moment. The thread is home; pulling its horizon down (or
/// tapping the horizon button) recedes the conversation into a floating glass
/// tile while the three spaces fan in behind it. One scalar (`app.zoom`)
/// drives scale, blur, radius and opacity together — gestural, interruptible,
/// and impossible to desync.
struct RootShell: View {
    @Environment(AppModel.self) private var app
    @State private var dragOffset: CGFloat = 0

    /// Live progress: settled zoom plus the in-flight gesture scrub.
    private var progress: CGFloat {
        min(1, max(0, app.zoom + dragOffset))
    }

    var body: some View {
        GeometryReader { geo in
            let p = progress
            let tileScale = 1 - 0.62 * p
            let tileOffsetY = (geo.size.height * 0.5 - 110) * p

            ZStack {
                InkBackdrop()

                // ── Spaces, behind: sharpen and fan in as the thread recedes.
                SpacesView()
                    .scaleEffect(1.06 - 0.06 * p)
                    .blur(radius: (1 - p) * 14)
                    .opacity(Double(p) * 1.25)
                    .allowsHitTesting(p > 0.95)

                // ── Thread, in front: recedes into a docked glass tile.
                ThreadView(horizonProgress: p, horizonDrag: horizonDrag)
                    .clipShape(RoundedRectangle(cornerRadius: 12 + 26 * p, style: .continuous))
                    .overlay {
                        RoundedRectangle(cornerRadius: 12 + 26 * p, style: .continuous)
                            .strokeBorder(.white.opacity(0.18 * Double(p)), lineWidth: 1)
                    }
                    .shadow(color: .black.opacity(0.45 * p), radius: 30 * p, y: 14 * p)
                    .scaleEffect(tileScale)
                    .offset(y: tileOffsetY)
                    .allowsHitTesting(p < 0.5)

                // ── While docked, the tile is one big "come home" button.
                if p > 0.9 {
                    RoundedRectangle(cornerRadius: 38, style: .continuous)
                        .fill(.white.opacity(0.001))
                        .frame(width: geo.size.width * tileScale,
                               height: geo.size.height * tileScale)
                        .offset(y: tileOffsetY)
                        .onTapGesture { app.returnToThread() }
                        .gesture(horizonDrag) // flick up also brings it home
                }
            }
            .animation(Motion.glide, value: app.zoom)
        }
        .statusBarHidden(progress > 0.5)
    }

    /// Attached to the thread's horizon grabber: pull down to dock the
    /// conversation, flick up from the dock to bring it home.
    private var horizonDrag: AnyGesture<DragGesture.Value> {
        AnyGesture(
            DragGesture(minimumDistance: 4, coordinateSpace: .global)
                .onChanged { value in
                    let delta = value.translation.height / 420
                    dragOffset = app.zoom == 0 ? max(0, delta) : min(0, delta)
                }
                .onEnded { value in
                    let velocity = value.predictedEndTranslation.height - value.translation.height
                    let settled: CGFloat = if app.zoom == 0 {
                        (progress > 0.35 || velocity > 120) ? 1 : 0
                    } else {
                        (progress < 0.65 || velocity < -120) ? 0 : 1
                    }
                    dragOffset = 0
                    withAnimation(Motion.glide) { app.zoom = settled }
                    if settled == 1 { Haptics.commit() } else { Haptics.touch() }
                }
        )
    }
}
