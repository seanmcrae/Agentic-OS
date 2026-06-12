import Foundation
import SwiftData

// MARK: - Tasks

enum TaskPriority: Int, Codable, CaseIterable {
    case low = 0, normal = 1, high = 2, urgent = 3

    var bars: Int { max(0, rawValue) }
    var label: String {
        switch self {
        case .low: return "Low"
        case .normal: return "Normal"
        case .high: return "High"
        case .urgent: return "Urgent"
        }
    }
}

/// Named `TaskItem` (not `Task`) to avoid the Swift Concurrency type.
@Model
final class TaskItem {
    @Attribute(.unique) var id: UUID
    var title: String
    var details: String
    var createdAt: Date
    var dueDate: Date?
    var completedAt: Date?
    var priorityRaw: Int
    var project: Project?
    /// Tasks that must finish before this one can start.
    var blockedBy: [TaskItem]?
    /// Inverse: tasks this one is blocking (explicit, because the
    /// relationship is self-referential and SwiftData can't infer it).
    @Relationship(inverse: \TaskItem.blockedBy)
    var blocking: [TaskItem]?

    var priority: TaskPriority {
        get { TaskPriority(rawValue: priorityRaw) ?? .normal }
        set { priorityRaw = newValue.rawValue }
    }
    var isDone: Bool { completedAt != nil }
    var isBlocked: Bool { blockedBy?.contains { !$0.isDone } ?? false }
    var isOverdue: Bool {
        guard let dueDate, !isDone else { return false }
        return dueDate < Calendar.current.startOfDay(for: .now)
    }

    init(title: String, details: String = "", dueDate: Date? = nil,
         priority: TaskPriority = .normal, project: Project? = nil) {
        self.id = UUID()
        self.title = title
        self.details = details
        self.createdAt = .now
        self.dueDate = dueDate
        self.priorityRaw = priority.rawValue
        self.project = project
    }
}

// MARK: - Projects

enum ProjectStatus: String, Codable, CaseIterable {
    case active, paused, done

    var label: String { rawValue.capitalized }
}

@Model
final class Project {
    @Attribute(.unique) var id: UUID
    var name: String
    var summary: String
    var statusRaw: String
    var hueRaw: Int
    var createdAt: Date
    @Relationship(deleteRule: .nullify, inverse: \TaskItem.project)
    var tasks: [TaskItem]?
    @Relationship(deleteRule: .cascade, inverse: \Milestone.project)
    var milestones: [Milestone]?

    var status: ProjectStatus {
        get { ProjectStatus(rawValue: statusRaw) ?? .active }
        set { statusRaw = newValue.rawValue }
    }
    var hue: ProjectHue {
        get { ProjectHue(rawValue: hueRaw) ?? .slate }
        set { hueRaw = newValue.rawValue }
    }
    var progress: Double {
        let all = tasks ?? []
        guard !all.isEmpty else { return 0 }
        return Double(all.filter(\.isDone).count) / Double(all.count)
    }
    var openTasks: [TaskItem] { (tasks ?? []).filter { !$0.isDone } }
    var blockers: [TaskItem] { openTasks.filter(\.isBlocked) }

    init(name: String, summary: String = "", hue: ProjectHue = .slate) {
        self.id = UUID()
        self.name = name
        self.summary = summary
        self.statusRaw = ProjectStatus.active.rawValue
        self.hueRaw = hue.rawValue
        self.createdAt = .now
    }
}

@Model
final class Milestone {
    @Attribute(.unique) var id: UUID
    var title: String
    var targetDate: Date?
    var reachedAt: Date?
    var project: Project?

    var isReached: Bool { reachedAt != nil }

    init(title: String, targetDate: Date? = nil, project: Project? = nil) {
        self.id = UUID()
        self.title = title
        self.targetDate = targetDate
        self.project = project
    }
}

// MARK: - Notes & knowledge

@Model
final class Note {
    @Attribute(.unique) var id: UUID
    var title: String
    var body: String
    var createdAt: Date
    var modifiedAt: Date
    var tags: [String]
    var project: Project?

    init(title: String, body: String = "", tags: [String] = [], project: Project? = nil) {
        self.id = UUID()
        self.title = title
        self.body = body
        self.createdAt = .now
        self.modifiedAt = .now
        self.tags = tags
        self.project = project
    }
}

/// What kind of entity an edge endpoint refers to.
enum EntityKind: String, Codable {
    case task, project, note, milestone
}

/// A typed edge in the knowledge graph. Endpoints are (kind, id) pairs so
/// any entity can link to any other.
@Model
final class KnowledgeLink {
    @Attribute(.unique) var id: UUID
    var fromKindRaw: String
    var fromID: UUID
    var toKindRaw: String
    var toID: UUID
    /// e.g. "references", "blocks", "inspired-by", "belongs-to"
    var relation: String
    var createdAt: Date

    var fromKind: EntityKind { EntityKind(rawValue: fromKindRaw) ?? .note }
    var toKind: EntityKind { EntityKind(rawValue: toKindRaw) ?? .note }

    init(from: (EntityKind, UUID), to: (EntityKind, UUID), relation: String) {
        self.id = UUID()
        self.fromKindRaw = from.0.rawValue
        self.fromID = from.1
        self.toKindRaw = to.0.rawValue
        self.toID = to.1
        self.relation = relation
        self.createdAt = .now
    }
}

// MARK: - Chat transcript

enum ChatRole: String, Codable {
    case user, agent
}

/// One persisted turn in the thread. Tool activity is stored as encoded
/// `ToolEventRecord`s so history replays with its cards intact.
@Model
final class ChatMessage {
    @Attribute(.unique) var id: UUID
    var roleRaw: String
    var text: String
    var createdAt: Date
    var toolEventsData: Data?

    var role: ChatRole { ChatRole(rawValue: roleRaw) ?? .agent }
    var toolEvents: [ToolEventRecord] {
        get {
            guard let toolEventsData else { return [] }
            return (try? JSONDecoder().decode([ToolEventRecord].self, from: toolEventsData)) ?? []
        }
        set { toolEventsData = try? JSONEncoder().encode(newValue) }
    }

    init(role: ChatRole, text: String) {
        self.id = UUID()
        self.roleRaw = role.rawValue
        self.text = text
        self.createdAt = .now
    }
}

struct ToolEventRecord: Codable, Identifiable, Hashable {
    var id: UUID = UUID()
    var toolName: String
    var summary: String
    var detail: String
    var succeeded: Bool
}

// MARK: - Semantic index cache

@Model
final class EmbeddingRecord {
    @Attribute(.unique) var entityID: UUID
    var kindRaw: String
    /// Packed [Float] vector.
    var vector: Data
    var contentHash: Int

    init(entityID: UUID, kind: EntityKind, vector: [Float], contentHash: Int) {
        self.entityID = entityID
        self.kindRaw = kind.rawValue
        self.vector = vector.withUnsafeBufferPointer { Data(buffer: $0) }
        self.contentHash = contentHash
    }

    var floats: [Float] {
        vector.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}
