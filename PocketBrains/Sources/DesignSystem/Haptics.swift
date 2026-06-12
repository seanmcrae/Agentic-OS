import UIKit
import CoreHaptics

/// Centralized haptic vocabulary. Glass is touched softly, commits are rigid,
/// completion gets a designed two-tap success pattern.
@MainActor
enum Haptics {
    private static let soft = UIImpactFeedbackGenerator(style: .soft)
    private static let rigid = UIImpactFeedbackGenerator(style: .rigid)
    private static let light = UIImpactFeedbackGenerator(style: .light)
    private static var engine: CHHapticEngine?

    static func touch() { soft.impactOccurred(intensity: 0.55) }
    static func commit() { rigid.impactOccurred(intensity: 0.8) }
    static func tick() { light.impactOccurred(intensity: 0.4) }

    /// Two ascending taps — task completion.
    static func success() {
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics else {
            rigid.impactOccurred(); return
        }
        do {
            if engine == nil { engine = try CHHapticEngine() }
            try engine?.start()
            let events = [
                CHHapticEvent(eventType: .hapticTransient, parameters: [
                    .init(parameterID: .hapticIntensity, value: 0.6),
                    .init(parameterID: .hapticSharpness, value: 0.4),
                ], relativeTime: 0),
                CHHapticEvent(eventType: .hapticTransient, parameters: [
                    .init(parameterID: .hapticIntensity, value: 1.0),
                    .init(parameterID: .hapticSharpness, value: 0.7),
                ], relativeTime: 0.09),
            ]
            let pattern = try CHHapticPattern(events: events, parameters: [])
            try engine?.makePlayer(with: pattern).start(atTime: 0)
        } catch {
            rigid.impactOccurred()
        }
    }
}
