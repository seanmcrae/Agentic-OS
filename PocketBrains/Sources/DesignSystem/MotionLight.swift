import SwiftUI
import CoreMotion

/// Streams device attitude as a 2D light direction for the glass shaders,
/// so speculars track how the phone is held. Low-pass filtered so the light
/// feels like liquid, not a sensor readout.
@MainActor
@Observable
final class MotionLight {
    static let shared = MotionLight()

    /// Unit-ish vector, x right / y down, magnitude ≤ 1.
    private(set) var direction: CGPoint = .init(x: 0.35, y: -0.65)

    private let manager = CMMotionManager()
    private var smoothed = CGPoint(x: 0.35, y: -0.65)

    private init() {}

    func start() {
        guard manager.isDeviceMotionAvailable, !manager.isDeviceMotionActive else { return }
        manager.deviceMotionUpdateInterval = 1.0 / 30.0
        manager.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
            guard let self, let g = motion?.gravity else { return }
            // Light comes from "above" the tilt: invert gravity, bias upward.
            let target = CGPoint(x: g.x * 0.8, y: min(-0.25, g.y * 0.8 - 0.3))
            let k = 0.12 // low-pass
            smoothed.x += (target.x - smoothed.x) * k
            smoothed.y += (target.y - smoothed.y) * k
            direction = smoothed
        }
    }

    func stop() { manager.stopDeviceMotionUpdates() }
}
