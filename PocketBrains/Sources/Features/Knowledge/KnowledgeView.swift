import SwiftUI
import SwiftData

/// The constellation: notes, projects and linked tasks orbiting the focused
/// idea under an aurora sky. Tap to refocus, tap again to open. Pinch and
/// pan to wander.
struct KnowledgeView: View {
    @Environment(AppModel.self) private var app
    @Query private var notes: [Note]
    @Query private var projects: [Project]
    @Query private var tasks: [TaskItem]
    @Query private var links: [KnowledgeLink]

    @State private var simulation = GraphSimulation()
    @State private var focus: UUID?
    @State private var scale: CGFloat = 1
    @State private var settledScale: CGFloat = 1
    @State private var pan: CGSize = .zero
    @State private var settledPan: CGSize = .zero

    private var graphNodes: [GraphNode] {
        var result = notes.map { GraphNode(id: $0.id, kind: .note, title: $0.title) }
        result += projects.map { GraphNode(id: $0.id, kind: .project, title: $0.name) }
        // Only tasks that participate in the graph earn a star.
        let linked = Set(links.flatMap { [$0.fromID, $0.toID] })
        result += tasks.filter { linked.contains($0.id) }
            .map { GraphNode(id: $0.id, kind: .task, title: $0.title) }
        return result
    }

    private var graphEdges: [GraphEdge] {
        links.map { GraphEdge(from: $0.fromID, to: $0.toID, relation: $0.relation) }
    }

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // Aurora sky — glacier light behind the constellation.
                TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { timeline in
                    Rectangle()
                        .fill(.clear)
                        .colorEffect(ShaderLibrary.aurora(
                            .float2(geo.size),
                            .float(timeline.date.timeIntervalSinceReferenceDate * 0.4),
                            .color(DomainHue.knowledge),
                            .float(0.35)))
                }
                .allowsHitTesting(false)

                if graphNodes.isEmpty {
                    EmptyState(icon: "circle.hexagongrid",
                               title: "Nothing linked yet",
                               message: "Capture notes and tell the thread to link them — the constellation grows itself.")
                } else {
                    constellation(in: geo.size)
                        .scaleEffect(scale)
                        .offset(pan)
                        .gesture(wander)
                }
            }
        }
        .onAppear { relayout() }
        .onChange(of: focus) { _, _ in relayout() }
        .onChange(of: links.count) { _, _ in relayout() }
        .onChange(of: notes.count) { _, _ in relayout() }
    }

    // MARK: Constellation

    private func relayout() {
        // Layout in a roomy virtual canvas; pan/zoom handles the viewport.
        let canvas = CGSize(width: 390, height: 560)
        simulation.retarget(OrbitalLayout.positions(
            nodes: graphNodes, edges: graphEdges, focus: focus, in: canvas))
    }

    @ViewBuilder
    private func constellation(in size: CGSize) -> some View {
        TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate
            ZStack {
                edgeCanvas(time: time)
                ForEach(graphNodes) { node in
                    if let point = simulation.displayed[node.id] {
                        NodeStar(node: node,
                                 isFocus: node.id == effectiveFocus,
                                 hue: hue(for: node))
                            .position(point)
                            .onTapGesture { tap(node) }
                    }
                }
            }
            .onChange(of: timeline.date) { _, _ in simulation.tick(time: time) }
        }
    }

    private func edgeCanvas(time: TimeInterval) -> some View {
        Canvas { context, _ in
            for edge in graphEdges {
                guard let a = simulation.displayed[edge.from],
                      let b = simulation.displayed[edge.to] else { continue }
                var path = Path()
                path.move(to: a)
                // A gentle arc bowing away from the midline — drawn light.
                let mid = CGPoint(x: (a.x + b.x) / 2, y: (a.y + b.y) / 2)
                let dx = b.x - a.x, dy = b.y - a.y
                let bow = CGPoint(x: mid.x - dy * 0.12, y: mid.y + dx * 0.12)
                path.addQuadCurve(to: b, control: bow)

                let pulse = 0.5 + 0.22 * sin(time * 1.1 + Double(edge.from.uuidString.hashValue % 7))
                context.stroke(
                    path,
                    with: .linearGradient(
                        Gradient(colors: [
                            DomainHue.knowledge.opacity(0.05),
                            DomainHue.knowledge.opacity(pulse * 0.5),
                            DomainHue.knowledge.opacity(0.05),
                        ]),
                        startPoint: a, endPoint: b),
                    lineWidth: 1)
            }
        }
        .allowsHitTesting(false)
    }

    private var effectiveFocus: UUID? {
        focus ?? graphNodes.max {
            degree(of: $0.id) < degree(of: $1.id)
        }?.id
    }

    private func degree(of id: UUID) -> Int {
        graphEdges.filter { $0.from == id || $0.to == id }.count
    }

    private func hue(for node: GraphNode) -> Color {
        switch node.kind {
        case .note: DomainHue.note
        case .task: DomainHue.task
        case .milestone: lumen
        case .project:
            projects.first { $0.id == node.id }?.hue.color ?? lumen
        }
    }

    private func tap(_ node: GraphNode) {
        if effectiveFocus == node.id {
            // Second tap opens the thing itself.
            Haptics.commit()
            switch node.kind {
            case .note:
                app.focusedNote = notes.first { $0.id == node.id }
            case .project:
                app.focusedProject = projects.first { $0.id == node.id }
                withAnimation(Motion.glide) { app.space = .projects }
            case .task, .milestone:
                withAnimation(Motion.glide) { app.space = .today }
            }
        } else {
            Haptics.touch()
            withAnimation(Motion.glide) { focus = node.id }
        }
    }

    // MARK: Wandering

    private var wander: some Gesture {
        SimultaneousGesture(
            MagnificationGesture()
                .onChanged { value in scale = min(2.2, max(0.6, settledScale * value)) }
                .onEnded { _ in settledScale = scale },
            DragGesture()
                .onChanged { value in
                    pan = CGSize(width: settledPan.width + value.translation.width,
                                 height: settledPan.height + value.translation.height)
                }
                .onEnded { _ in settledPan = pan }
        )
    }
}

/// One star: a glowing core sized by importance, labeled in micro type.
private struct NodeStar: View {
    let node: GraphNode
    let isFocus: Bool
    let hue: Color

    var body: some View {
        VStack(spacing: Space.xxs) {
            ZStack {
                Circle()
                    .fill(RadialGradient(
                        colors: [hue.opacity(isFocus ? 0.95 : 0.65), hue.opacity(0)],
                        center: .center, startRadius: 1,
                        endRadius: isFocus ? 26 : 13))
                    .frame(width: isFocus ? 52 : 26, height: isFocus ? 52 : 26)
                Circle()
                    .fill(Paper.primary)
                    .frame(width: isFocus ? 9 : 5.5, height: isFocus ? 9 : 5.5)
                    .shadow(color: hue, radius: isFocus ? 8 : 4)
            }
            Text(node.title)
                .font(Type.micro)
                .tracking(0.3)
                .foregroundStyle(isFocus ? Paper.primary : Paper.tertiary)
                .lineLimit(1)
                .frame(maxWidth: 110)
        }
        .animation(Motion.glide, value: isFocus)
        .contentShape(Circle().scale(2.2))
    }
}
