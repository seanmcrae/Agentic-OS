import SwiftUI

struct GraphNode: Identifiable, Hashable {
    let id: UUID
    let kind: EntityKind
    let title: String
}

struct GraphEdge: Hashable {
    let from: UUID
    let to: UUID
    let relation: String
}

/// Orbital constellation layout — not a force hairball. The focused node
/// holds the center; everything else sits on rings by graph distance, placed
/// angularly near its parent so clusters read as constellations.
enum OrbitalLayout {
    static func positions(nodes: [GraphNode], edges: [GraphEdge],
                          focus: UUID?, in size: CGSize) -> [UUID: CGPoint] {
        guard !nodes.isEmpty else { return [:] }
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        let ringStep = min(size.width, size.height) * 0.21

        var adjacency: [UUID: Set<UUID>] = [:]
        for edge in edges {
            adjacency[edge.from, default: []].insert(edge.to)
            adjacency[edge.to, default: []].insert(edge.from)
        }

        // Default focus: the best-connected node.
        let focusID = focus ?? nodes.max {
            (adjacency[$0.id]?.count ?? 0) < (adjacency[$1.id]?.count ?? 0)
        }!.id

        // BFS rings.
        var distance: [UUID: Int] = [focusID: 0]
        var parent: [UUID: UUID] = [:]
        var queue = [focusID]
        while !queue.isEmpty {
            let current = queue.removeFirst()
            for neighbor in adjacency[current] ?? [] where distance[neighbor] == nil {
                distance[neighbor] = distance[current]! + 1
                parent[neighbor] = current
                queue.append(neighbor)
            }
        }

        var positions: [UUID: CGPoint] = [focusID: center]
        var angles: [UUID: Double] = [focusID: 0]
        let maxRing = distance.values.max() ?? 0

        // Ring by ring, children fan out around their parent's bearing.
        for ring in 1...max(1, maxRing) {
            let members = nodes.filter { distance[$0.id] == ring }
            guard !members.isEmpty else { continue }
            let radius = ringStep * CGFloat(ring)

            // Group members under their parent, then spread each family.
            let families = Dictionary(grouping: members) { parent[$0.id] ?? focusID }
            for (parentID, children) in families.sorted(by: { $0.key.uuidString < $1.key.uuidString }) {
                let baseAngle = ring == 1
                    ? 0
                    : (angles[parentID] ?? 0)
                let spread = ring == 1 ? 2 * Double.pi : Double.pi / 2.2
                for (index, child) in children.sorted(by: { $0.title < $1.title }).enumerated() {
                    let fraction = children.count == 1
                        ? 0.5
                        : Double(index) / Double(children.count - 1)
                    let angle = ring == 1
                        ? baseAngle + (2 * .pi) * Double(index) / Double(children.count) - .pi / 2
                        : baseAngle + (fraction - 0.5) * spread
                    angles[child.id] = angle
                    positions[child.id] = CGPoint(
                        x: center.x + cos(angle) * radius,
                        y: center.y + sin(angle) * radius * 0.92) // slight oval, feels composed
                }
            }
        }

        // Unconnected satellites: a dim outermost ring.
        let orphans = nodes.filter { distance[$0.id] == nil }
        if !orphans.isEmpty {
            let radius = ringStep * CGFloat(maxRing + 1) + ringStep * 0.4
            for (index, orphan) in orphans.enumerated() {
                let angle = (2 * .pi) * Double(index) / Double(orphans.count) + .pi / 7
                positions[orphan.id] = CGPoint(
                    x: center.x + cos(angle) * radius,
                    y: center.y + sin(angle) * radius * 0.92)
            }
        }
        return positions
    }
}

/// Per-frame interpolation toward layout targets plus a touch of idle drift,
/// so the constellation breathes and edges stay glued to their nodes.
@MainActor
@Observable
final class GraphSimulation {
    private(set) var displayed: [UUID: CGPoint] = [:]
    var targets: [UUID: CGPoint] = [:]
    private var phases: [UUID: Double] = [:]

    func retarget(_ new: [UUID: CGPoint]) {
        targets = new
        for id in new.keys where phases[id] == nil {
            phases[id] = Double.random(in: 0..<(2 * .pi))
        }
        // New nodes are born at their target, slightly inward.
        for (id, point) in new where displayed[id] == nil {
            displayed[id] = point
        }
        displayed = displayed.filter { new[$0.key] != nil }
    }

    func tick(time: TimeInterval) {
        for (id, target) in targets {
            let phase = phases[id] ?? 0
            let drift = CGPoint(
                x: cos(time * 0.5 + phase) * 3.5,
                y: sin(time * 0.37 + phase * 1.3) * 3.5)
            let goal = CGPoint(x: target.x + drift.x, y: target.y + drift.y)
            let current = displayed[id] ?? goal
            // Exponential approach — organic, interruption-proof.
            displayed[id] = CGPoint(
                x: current.x + (goal.x - current.x) * 0.085,
                y: current.y + (goal.y - current.y) * 0.085)
        }
    }
}
