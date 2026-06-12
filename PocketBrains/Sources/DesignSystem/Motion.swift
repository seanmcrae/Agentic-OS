import SwiftUI

/// One spring family for the whole app. Nothing animates outside these.
enum Motion {
    /// Controls: presses, toggles, chips.
    static let snap = Animation.spring(response: 0.32, dampingFraction: 0.86)
    /// Cards, layout, navigation.
    static let glide = Animation.spring(response: 0.48, dampingFraction: 0.84)
    /// Ambient: auras, breathing, idle drift.
    static let drift = Animation.spring(response: 0.85, dampingFraction: 0.92)
}

/// Scale-on-press with spring physics and a soft haptic on touch-down.
struct GlassPressStyle: ButtonStyle {
    var scale: CGFloat = 0.965

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? scale : 1)
            .opacity(configuration.isPressed ? 0.92 : 1)
            .animation(Motion.snap, value: configuration.isPressed)
            .onChange(of: configuration.isPressed) { _, pressed in
                if pressed { Haptics.touch() }
            }
    }
}
