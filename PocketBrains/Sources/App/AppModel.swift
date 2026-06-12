import SwiftUI
import SwiftData

enum SpaceKind: Int, CaseIterable, Identifiable {
    case today, projects, knowledge

    var id: Int { rawValue }
    var title: String {
        switch self {
        case .today: return "Today"
        case .projects: return "Projects"
        case .knowledge: return "Knowledge"
        }
    }
    var icon: String {
        switch self {
        case .today: return "sun.horizon"
        case .projects: return "square.stack"
        case .knowledge: return "circle.hexagongrid"
        }
    }
    var hue: Color {
        switch self {
        case .today: return DomainHue.task
        case .projects: return lumen
        case .knowledge: return DomainHue.knowledge
        }
    }
}

/// App-wide state. One scalar — `zoom` — drives the entire thread↔spaces
/// transition, so the signature moment can never desynchronize.
@MainActor
@Observable
final class AppModel {
    // Core stack
    let services: DataServices
    let semanticIndex: SemanticIndex
    let toolbox: ToolBox
    let agent: AgentOrchestrator

    // Navigation
    /// 0 = thread is home, 1 = spaces. Scrubbed by gesture, settled by spring.
    var zoom: CGFloat = 0
    var space: SpaceKind = .today
    var focusedProject: Project?
    var focusedNote: Note?

    init(context: ModelContext) {
        let services = DataServices(context: context)
        let semanticIndex = SemanticIndex(context: context)
        let toolbox = ToolBox(services: services, semanticIndex: semanticIndex)
        self.services = services
        self.semanticIndex = semanticIndex
        self.toolbox = toolbox
        self.agent = AgentOrchestrator(context: context, toolbox: toolbox)
    }

    func bootstrap() {
        SeedData.seedIfNeeded(services)
        semanticIndex.reindexAll(notes: services.notes.all())
    }

    func openSpaces(_ kind: SpaceKind? = nil) {
        if let kind { space = kind }
        withAnimation(Motion.glide) { zoom = 1 }
        Haptics.commit()
    }

    func returnToThread() {
        withAnimation(Motion.glide) { zoom = 0 }
        focusedProject = nil
        focusedNote = nil
        Haptics.touch()
    }
}
