import Foundation
import SwiftData

/// The single write path. Agent tools AND structured views both call these,
/// so every mutation is visible everywhere instantly.
@MainActor
struct DataServices {
    let context: ModelContext
    let tasks: TaskService
    let projects: ProjectService
    let notes: NoteService
    let graph: GraphService

    init(context: ModelContext) {
        self.context = context
        self.tasks = TaskService(context: context)
        self.projects = ProjectService(context: context)
        self.notes = NoteService(context: context)
        self.graph = GraphService(context: context)
    }
}

// MARK: - Tasks

@MainActor
struct TaskService {
    let context: ModelContext

    @discardableResult
    func create(title: String, details: String = "", due: Date? = nil,
                priority: TaskPriority = .normal, project: Project? = nil) -> TaskItem {
        let task = TaskItem(title: title, details: details, dueDate: due,
                            priority: priority, project: project)
        context.insert(task)
        try? context.save()
        return task
    }

    func complete(_ task: TaskItem) {
        task.completedAt = .now
        try? context.save()
    }

    func reopen(_ task: TaskItem) {
        task.completedAt = nil
        try? context.save()
    }

    func update(_ task: TaskItem, title: String? = nil, details: String? = nil,
                due: Date?? = nil, priority: TaskPriority? = nil, project: Project?? = nil) {
        if let title { task.title = title }
        if let details { task.details = details }
        if let due { task.dueDate = due }
        if let priority { task.priority = priority }
        if let project { task.project = project }
        try? context.save()
    }

    func addDependency(_ task: TaskItem, blockedBy blocker: TaskItem) {
        guard task.id != blocker.id else { return }
        var deps = task.blockedBy ?? []
        guard !deps.contains(where: { $0.id == blocker.id }) else { return }
        deps.append(blocker)
        task.blockedBy = deps
        try? context.save()
    }

    func all(includeDone: Bool = false) -> [TaskItem] {
        let items = context.fetchAll(TaskItem.self, sortBy: [.init(\.createdAt, order: .reverse)])
        return includeDone ? items : items.filter { !$0.isDone }
    }

    func find(matching query: String) -> TaskItem? {
        let q = query.lowercased()
        return all(includeDone: true)
            .first { $0.title.lowercased().contains(q) || q.contains($0.title.lowercased()) }
    }

    func dueToday() -> [TaskItem] {
        let cal = Calendar.current
        return all().filter { $0.dueDate.map { cal.isDateInToday($0) || $0 < .now } ?? false }
    }

    func dueWithin(days: Int) -> [TaskItem] {
        guard let horizon = Calendar.current.date(byAdding: .day, value: days, to: .now)
        else { return [] }
        return all().filter { $0.dueDate.map { $0 <= horizon } ?? false }
    }
}

// MARK: - Projects

@MainActor
struct ProjectService {
    let context: ModelContext

    @discardableResult
    func create(name: String, summary: String = "") -> Project {
        // Round-robin hue assignment, skipping the most recent project's hue.
        let existing = all()
        let used = existing.first.map(\.hue)
        var hue = ProjectHue.allCases[existing.count % ProjectHue.allCases.count]
        if hue == used { hue = ProjectHue.allCases[(existing.count + 1) % ProjectHue.allCases.count] }
        let project = Project(name: name, summary: summary, hue: hue)
        context.insert(project)
        try? context.save()
        return project
    }

    func all() -> [Project] {
        context.fetchAll(Project.self, sortBy: [.init(\.createdAt, order: .reverse)])
    }

    func find(matching query: String) -> Project? {
        let q = query.lowercased()
        return all().first { $0.name.lowercased().contains(q) || q.contains($0.name.lowercased()) }
    }

    @discardableResult
    func addMilestone(to project: Project, title: String, target: Date? = nil) -> Milestone {
        let m = Milestone(title: title, targetDate: target, project: project)
        context.insert(m)
        try? context.save()
        return m
    }

    func setStatus(_ project: Project, _ status: ProjectStatus) {
        project.status = status
        try? context.save()
    }

    /// Recent activity feed for a project — completions, new tasks, notes.
    func recentActivity(for project: Project, limit: Int = 5) -> [String] {
        var events: [(Date, String)] = []
        for t in project.tasks ?? [] {
            if let done = t.completedAt { events.append((done, "Completed “\(t.title)”")) }
            else { events.append((t.createdAt, "Added “\(t.title)”")) }
        }
        for m in project.milestones ?? [] where m.isReached {
            events.append((m.reachedAt!, "Milestone reached: \(m.title)"))
        }
        return events.sorted { $0.0 > $1.0 }.prefix(limit).map(\.1)
    }
}

// MARK: - Notes

@MainActor
struct NoteService {
    let context: ModelContext

    @discardableResult
    func create(title: String, body: String, tags: [String] = [], project: Project? = nil) -> Note {
        let note = Note(title: title, body: body, tags: tags, project: project)
        context.insert(note)
        try? context.save()
        return note
    }

    func append(_ note: Note, text: String) {
        note.body += (note.body.isEmpty ? "" : "\n\n") + text
        note.modifiedAt = .now
        try? context.save()
    }

    func all() -> [Note] {
        context.fetchAll(Note.self, sortBy: [.init(\.modifiedAt, order: .reverse)])
    }

    func find(matching query: String) -> Note? {
        let q = query.lowercased()
        return all().first { $0.title.lowercased().contains(q) || q.contains($0.title.lowercased()) }
    }

    /// Plain keyword search; semantic search lives in SemanticIndex and is
    /// merged by the searchNotes tool.
    func keywordSearch(_ query: String, limit: Int = 8) -> [Note] {
        let terms = query.lowercased().split(separator: " ").map(String.init)
        guard !terms.isEmpty else { return [] }
        return all()
            .map { note -> (Note, Int) in
                let hay = (note.title + " " + note.body + " " + note.tags.joined(separator: " ")).lowercased()
                return (note, terms.filter { hay.contains($0) }.count)
            }
            .filter { $0.1 > 0 }
            .sorted { $0.1 > $1.1 }
            .prefix(limit)
            .map(\.0)
    }

    func from(day: Date) -> [Note] {
        let cal = Calendar.current
        return all().filter { cal.isDate($0.createdAt, inSameDayAs: day) || cal.isDate($0.modifiedAt, inSameDayAs: day) }
    }
}

// MARK: - Knowledge graph

@MainActor
struct GraphService {
    let context: ModelContext

    @discardableResult
    func link(from: (EntityKind, UUID), to: (EntityKind, UUID), relation: String) -> KnowledgeLink {
        let edges = allLinks()
        if let existing = edges.first(where: {
            $0.fromID == from.1 && $0.toID == to.1 && $0.relation == relation
        }) { return existing }
        let link = KnowledgeLink(from: from, to: to, relation: relation)
        context.insert(link)
        try? context.save()
        return link
    }

    func allLinks() -> [KnowledgeLink] {
        context.fetchAll(KnowledgeLink.self)
    }

    func links(touching id: UUID) -> [KnowledgeLink] {
        allLinks().filter { $0.fromID == id || $0.toID == id }
    }

    /// Display name resolution for any edge endpoint.
    func title(of kind: EntityKind, id: UUID) -> String? {
        switch kind {
        case .task: context.fetchAll(TaskItem.self).first { $0.id == id }?.title
        case .project: context.fetchAll(Project.self).first { $0.id == id }?.name
        case .note: context.fetchAll(Note.self).first { $0.id == id }?.title
        case .milestone: context.fetchAll(Milestone.self).first { $0.id == id }?.title
        }
    }
}
