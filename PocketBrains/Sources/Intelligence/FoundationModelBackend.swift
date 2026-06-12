#if canImport(FoundationModels)
import Foundation
import FoundationModels

/// Primary brain: Apple's on-device ~3B foundation model (iOS 26).
/// The app's ToolBox is bridged into native `Tool` conformances with
/// @Generable arguments so the model gets real guided generation.
@MainActor
final class FoundationModelBackend: ModelBackend {
    let displayName = "Apple Intelligence · on-device"

    private let toolbox: ToolBox
    private let sink = ToolEventSink()
    private var session: LanguageModelSession

    init?(toolbox: ToolBox) {
        guard case .available = SystemLanguageModel.default.availability else { return nil }
        self.toolbox = toolbox
        self.session = Self.makeSession(toolbox: toolbox, sink: sink)
    }

    private static func makeSession(toolbox: ToolBox, sink: ToolEventSink) -> LanguageModelSession {
        LanguageModelSession(
            tools: Self.tools(toolbox: toolbox, sink: sink),
            instructions: AgentVoice.instructions()
        )
    }

    func resetConversation() {
        session = Self.makeSession(toolbox: toolbox, sink: sink)
    }

    func reply(to prompt: String) -> AsyncThrowingStream<AgentEvent, Error> {
        AsyncThrowingStream { continuation in
            sink.emit = { continuation.yield($0) }
            let task = Task { @MainActor in
                do {
                    var lastText = ""
                    let stream = session.streamResponse(to: prompt)
                    for try await partial in stream {
                        lastText = String(describing: partial)
                        continuation.yield(.text(lastText))
                    }
                    continuation.yield(.done(finalText: lastText))
                    continuation.finish()
                } catch {
                    // Context overflow → start a fresh session and surface
                    // a graceful failure; the orchestrator will retry.
                    self.resetConversation()
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Tool bridge

    private static func tools(toolbox: ToolBox, sink: ToolEventSink) -> [any Tool] {
        [
            CreateTaskTool(toolbox: toolbox, sink: sink),
            CompleteTaskTool(toolbox: toolbox, sink: sink),
            UpdateTaskTool(toolbox: toolbox, sink: sink),
            QueryTasksTool(toolbox: toolbox, sink: sink),
            CreateProjectTool(toolbox: toolbox, sink: sink),
            ProjectStatusTool(toolbox: toolbox, sink: sink),
            AddMilestoneTool(toolbox: toolbox, sink: sink),
            CreateNoteTool(toolbox: toolbox, sink: sink),
            SearchNotesTool(toolbox: toolbox, sink: sink),
            NotesFromTool(toolbox: toolbox, sink: sink),
            LinkItemsTool(toolbox: toolbox, sink: sink),
            AgendaTool(toolbox: toolbox, sink: sink),
            RecallTool(toolbox: toolbox, sink: sink),
        ]
    }
}

/// Routes tool activity out of FoundationModels' tool calls and into the
/// backend's event stream, and persists the record for the transcript.
@MainActor
final class ToolEventSink {
    var emit: ((AgentEvent) -> Void)?

    func perform(_ name: String, _ activity: String,
                 _ op: @MainActor () -> ToolResult) -> String {
        emit?(.toolStarted(name: name, summary: activity))
        let result = op()
        emit?(.toolFinished(result.record(toolName: name)))
        return result.detail
    }
}

// MARK: - Tool conformances

private struct CreateTaskTool: Tool {
    let name = "createTask"
    let description = "Create a task or reminder for the user."
    let toolbox: ToolBox
    let sink: ToolEventSink

    @Generable
    struct Arguments {
        @Guide(description: "Short imperative task title")
        let title: String
        @Guide(description: "Due date in natural language, e.g. 'Friday', 'tomorrow', 'June 20'")
        let due: String?
        @Guide(description: "Priority: low, normal, high, or urgent")
        let priority: String?
        @Guide(description: "Name of the project to file this under")
        let project: String?
    }

    func call(arguments: Arguments) async throws -> String {
        await sink.perform(name, "Creating task") {
            toolbox.createTask(title: arguments.title, due: arguments.due,
                               priority: arguments.priority, projectName: arguments.project)
        }
    }
}

private struct CompleteTaskTool: Tool {
    let name = "completeTask"
    let description = "Mark a task as done, matched by words from its title."
    let toolbox: ToolBox
    let sink: ToolEventSink

    @Generable
    struct Arguments {
        @Guide(description: "Words from the task's title")
        let query: String
    }

    func call(arguments: Arguments) async throws -> String {
        await sink.perform(name, "Completing task") {
            toolbox.completeTask(query: arguments.query)
        }
    }
}

private struct UpdateTaskTool: Tool {
    let name = "updateTask"
    let description = "Change a task's due date, priority, or project."
    let toolbox: ToolBox
    let sink: ToolEventSink

    @Generable
    struct Arguments {
        @Guide(description: "Words from the task's title")
        let query: String
        @Guide(description: "New due date in natural language")
        let due: String?
        @Guide(description: "New priority: low, normal, high, or urgent")
        let priority: String?
        @Guide(description: "Project to move the task into")
        let project: String?
    }

    func call(arguments: Arguments) async throws -> String {
        await sink.perform(name, "Updating task") {
            toolbox.updateTask(query: arguments.query, due: arguments.due,
                               priority: arguments.priority, projectName: arguments.project)
        }
    }
}

private struct QueryTasksTool: Tool {
    let name = "queryTasks"
    let description = "List the user's tasks with a filter."
    let toolbox: ToolBox
    let sink: ToolEventSink

    @Generable
    struct Arguments {
        @Guide(description: "Filter: today, overdue, blocked, upcoming, or all")
        let filter: String
        @Guide(description: "Limit results to this project name")
        let project: String?
    }

    func call(arguments: Arguments) async throws -> String {
        await sink.perform(name, "Looking at tasks") {
            toolbox.queryTasks(filter: arguments.filter, projectName: arguments.project)
        }
    }
}

private struct CreateProjectTool: Tool {
    let name = "createProject"
    let description = "Create a new project."
    let toolbox: ToolBox
    let sink: ToolEventSink

    @Generable
    struct Arguments {
        @Guide(description: "Project name")
        let projectName: String
        @Guide(description: "One-line summary of the project")
        let summary: String?
    }

    func call(arguments: Arguments) async throws -> String {
        await sink.perform(name, "Creating project") {
            toolbox.createProject(name: arguments.projectName, summary: arguments.summary ?? "")
        }
    }
}

private struct ProjectStatusTool: Tool {
    let name = "projectStatus"
    let description = "Get progress, blockers, milestones and recent activity for a project."
    let toolbox: ToolBox
    let sink: ToolEventSink

    @Generable
    struct Arguments {
        @Guide(description: "Project name")
        let projectName: String
    }

    func call(arguments: Arguments) async throws -> String {
        await sink.perform(name, "Checking project") {
            toolbox.projectStatus(name: arguments.projectName)
        }
    }
}

private struct AddMilestoneTool: Tool {
    let name = "addMilestone"
    let description = "Add a milestone to a project."
    let toolbox: ToolBox
    let sink: ToolEventSink

    @Generable
    struct Arguments {
        @Guide(description: "Project name")
        let projectName: String
        @Guide(description: "Milestone title")
        let title: String
        @Guide(description: "Target date in natural language")
        let target: String?
    }

    func call(arguments: Arguments) async throws -> String {
        await sink.perform(name, "Adding milestone") {
            toolbox.addMilestone(projectName: arguments.projectName,
                                 title: arguments.title, target: arguments.target)
        }
    }
}

private struct CreateNoteTool: Tool {
    let name = "createNote"
    let description = "Capture a note for the user."
    let toolbox: ToolBox
    let sink: ToolEventSink

    @Generable
    struct Arguments {
        @Guide(description: "Note title")
        let title: String
        @Guide(description: "Note content")
        let body: String
        @Guide(description: "Project name to file it under")
        let project: String?
    }

    func call(arguments: Arguments) async throws -> String {
        await sink.perform(name, "Writing note") {
            toolbox.createNote(title: arguments.title, body: arguments.body,
                               projectName: arguments.project)
        }
    }
}

private struct SearchNotesTool: Tool {
    let name = "searchNotes"
    let description = "Semantic and keyword search across all the user's notes."
    let toolbox: ToolBox
    let sink: ToolEventSink

    @Generable
    struct Arguments {
        @Guide(description: "What to search for")
        let query: String
    }

    func call(arguments: Arguments) async throws -> String {
        await sink.perform(name, "Searching notes") {
            toolbox.searchNotes(query: arguments.query)
        }
    }
}

private struct NotesFromTool: Tool {
    let name = "notesFrom"
    let description = "Fetch full text of notes from a given day (e.g. 'yesterday', 'Monday')."
    let toolbox: ToolBox
    let sink: ToolEventSink

    @Generable
    struct Arguments {
        @Guide(description: "The day, in natural language")
        let day: String
    }

    func call(arguments: Arguments) async throws -> String {
        await sink.perform(name, "Reading notes") {
            toolbox.notesFrom(dayDescription: arguments.day)
        }
    }
}

private struct LinkItemsTool: Tool {
    let name = "linkItems"
    let description = "Create a knowledge-graph link between two items (notes, tasks, projects)."
    let toolbox: ToolBox
    let sink: ToolEventSink

    @Generable
    struct Arguments {
        @Guide(description: "Source item, referenced by name")
        let from: String
        @Guide(description: "Target item, referenced by name")
        let to: String
        @Guide(description: "Relation, e.g. references, inspired-by, blocks")
        let relation: String?
    }

    func call(arguments: Arguments) async throws -> String {
        await sink.perform(name, "Linking items") {
            toolbox.linkItems(from: arguments.from, to: arguments.to,
                              relation: arguments.relation ?? "references")
        }
    }
}

private struct AgendaTool: Tool {
    let name = "agenda"
    let description = "What needs the user's attention today: overdue, due today, blocked, this week."
    let toolbox: ToolBox
    let sink: ToolEventSink

    @Generable
    struct Arguments {}

    func call(arguments: Arguments) async throws -> String {
        await sink.perform(name, "Composing agenda") {
            toolbox.agenda()
        }
    }
}

private struct RecallTool: Tool {
    let name = "recall"
    let description = "Search past conversations and completed work."
    let toolbox: ToolBox
    let sink: ToolEventSink

    @Generable
    struct Arguments {
        @Guide(description: "What to remember")
        let query: String
    }

    func call(arguments: Arguments) async throws -> String {
        await sink.perform(name, "Remembering") {
            toolbox.recall(query: arguments.query)
        }
    }
}
#endif
