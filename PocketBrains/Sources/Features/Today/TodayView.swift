import SwiftUI
import SwiftData

/// The agenda: exactly what needs attention, ranked — overdue, due today,
/// blocked, then the week. Auto-updates the instant the agent acts.
struct TodayView: View {
    @Environment(AppModel.self) private var app
    @Query(sort: \TaskItem.createdAt, order: .reverse) private var allTasks: [TaskItem]

    private var open: [TaskItem] { allTasks.filter { !$0.isDone } }
    private var overdue: [TaskItem] { open.filter(\.isOverdue) }
    private var dueToday: [TaskItem] {
        open.filter { $0.dueDate.map { Calendar.current.isDateInToday($0) } ?? false && !$0.isOverdue }
    }
    private var blocked: [TaskItem] { open.filter { $0.isBlocked && !$0.isOverdue } }
    private var thisWeek: [TaskItem] {
        let horizon = Calendar.current.date(byAdding: .day, value: 7, to: .now)!
        return open.filter { task in
            guard let due = task.dueDate else { return false }
            return due <= horizon && !task.isOverdue
                && !Calendar.current.isDateInToday(due) && !task.isBlocked
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Space.l) {
                if overdue.isEmpty && dueToday.isEmpty && blocked.isEmpty && thisWeek.isEmpty {
                    clearRunway
                }
                section("Overdue", tasks: overdue, urgentTint: true)
                section("Today", tasks: dueToday)
                section("Blocked", tasks: blocked)
                section("This week", tasks: thisWeek)
            }
            .padding(.horizontal, Space.gutter)
            .padding(.top, Space.xs)
            .padding(.bottom, Space.xl)
        }
        .scrollIndicators(.hidden)
        .animation(Motion.glide, value: open.count)
    }

    @ViewBuilder
    private func section(_ title: String, tasks: [TaskItem], urgentTint: Bool = false) -> some View {
        if !tasks.isEmpty {
            VStack(alignment: .leading, spacing: Space.xs) {
                SectionHeader(title: title, detail: "\(tasks.count)")
                VStack(spacing: Space.xs) {
                    ForEach(tasks, id: \.id) { task in
                        TaskRow(task: task, alarm: urgentTint)
                    }
                }
            }
        }
    }

    private var clearRunway: some View {
        VStack(spacing: Space.m) {
            AuroraOrb(energy: 0.3, tint: DomainHue.note, size: 64)
            Text("Clear runway")
                .font(Type.heading)
                .foregroundStyle(Paper.primary)
            Text("Nothing is overdue, due, or blocked.\nA rare and beautiful state.")
                .font(Type.callout)
                .foregroundStyle(Paper.tertiary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, Space.vast)
    }
}

/// One task on glass: completion ring, title, context chips.
struct TaskRow: View {
    @Environment(AppModel.self) private var app
    let task: TaskItem
    var alarm: Bool = false

    var body: some View {
        HStack(spacing: Space.s) {
            CompletionToggle(done: task.isDone) {
                if task.isDone {
                    app.services.tasks.reopen(task)
                } else {
                    app.services.tasks.complete(task)
                }
            }

            VStack(alignment: .leading, spacing: 3) {
                Text(task.title)
                    .font(Type.bodyMedium)
                    .foregroundStyle(Paper.primary)
                    .strikethrough(task.isDone, color: Paper.tertiary)
                    .lineLimit(2)

                HStack(spacing: Space.xs) {
                    if let due = task.dueDate {
                        Label(NaturalDateParser.describe(due),
                              systemImage: "calendar")
                            .font(Type.micro)
                            .foregroundStyle(task.isOverdue ? DomainHue.task : Paper.tertiary)
                    }
                    if let project = task.project {
                        HStack(spacing: 3) {
                            Circle().fill(project.hue.color).frame(width: 5, height: 5)
                            Text(project.name)
                                .font(Type.micro)
                                .foregroundStyle(Paper.tertiary)
                        }
                    }
                    if task.isBlocked {
                        Label("Blocked", systemImage: "hourglass")
                            .font(Type.micro)
                            .foregroundStyle(Paper.tertiary)
                    }
                }
            }

            Spacer(minLength: Space.xs)
            PriorityGlyph(priority: task.priority)
        }
        .padding(.horizontal, Space.m)
        .padding(.vertical, Space.s)
        .glass(Radius.card, tint: alarm ? DomainHue.task : nil, depth: 0.5)
        .opacity(task.isDone ? 0.55 : 1)
        .transition(.asymmetric(
            insertion: .move(edge: .bottom).combined(with: .opacity),
            removal: .scale(scale: 0.95).combined(with: .opacity)))
    }
}
