import Foundation

/// Backend-neutral tool descriptions. The Foundation Models backend bridges
/// these to native `Tool` conformances with @Generable arguments; the MLX
/// backend renders them into a JSON tool-calling prompt; the fallback parser
/// maps intents straight onto them.
struct AgentToolSpec {
    let name: String
    let description: String
    let parameters: [Param]
    let run: @MainActor ([String: String]) -> ToolResult

    struct Param {
        let name: String
        let description: String
        var required = false
    }
}

@MainActor
enum AgentToolRegistry {
    static func all(toolbox: ToolBox) -> [AgentToolSpec] {
        [
            .init(name: "createTask",
                  description: "Create a task. Use for reminders and todos.",
                  parameters: [
                    .init(name: "title", description: "Short imperative title", required: true),
                    .init(name: "due", description: "Natural due date, e.g. 'Friday', 'tomorrow'"),
                    .init(name: "priority", description: "low | normal | high | urgent"),
                    .init(name: "project", description: "Project name to file it under"),
                  ],
                  run: { toolbox.createTask(title: $0["title"] ?? "Untitled",
                                            due: $0["due"], priority: $0["priority"],
                                            projectName: $0["project"]) }),
            .init(name: "completeTask",
                  description: "Mark a task done, matched by words from its title.",
                  parameters: [.init(name: "query", description: "Words from the task title", required: true)],
                  run: { toolbox.completeTask(query: $0["query"] ?? "") }),
            .init(name: "updateTask",
                  description: "Change a task's due date, priority, or project.",
                  parameters: [
                    .init(name: "query", description: "Words from the task title", required: true),
                    .init(name: "due", description: "New natural due date"),
                    .init(name: "priority", description: "low | normal | high | urgent"),
                    .init(name: "project", description: "Project to move it to"),
                  ],
                  run: { toolbox.updateTask(query: $0["query"] ?? "", due: $0["due"],
                                            priority: $0["priority"], projectName: $0["project"]) }),
            .init(name: "queryTasks",
                  description: "List tasks. Filters: today, overdue, blocked, upcoming, all.",
                  parameters: [
                    .init(name: "filter", description: "today | overdue | blocked | upcoming | all", required: true),
                    .init(name: "project", description: "Limit to a project"),
                  ],
                  run: { toolbox.queryTasks(filter: $0["filter"] ?? "all", projectName: $0["project"]) }),
            .init(name: "createProject",
                  description: "Create a project.",
                  parameters: [
                    .init(name: "name", description: "Project name", required: true),
                    .init(name: "summary", description: "One-line description"),
                  ],
                  run: { toolbox.createProject(name: $0["name"] ?? "Untitled", summary: $0["summary"] ?? "") }),
            .init(name: "projectStatus",
                  description: "Progress, blockers, milestones and recent activity for a project.",
                  parameters: [.init(name: "name", description: "Project name", required: true)],
                  run: { toolbox.projectStatus(name: $0["name"] ?? "") }),
            .init(name: "addMilestone",
                  description: "Add a milestone to a project.",
                  parameters: [
                    .init(name: "project", description: "Project name", required: true),
                    .init(name: "title", description: "Milestone title", required: true),
                    .init(name: "target", description: "Target date, natural language"),
                  ],
                  run: { toolbox.addMilestone(projectName: $0["project"] ?? "",
                                              title: $0["title"] ?? "", target: $0["target"]) }),
            .init(name: "createNote",
                  description: "Capture a note.",
                  parameters: [
                    .init(name: "title", description: "Note title", required: true),
                    .init(name: "body", description: "Note content", required: true),
                    .init(name: "project", description: "Project to file it under"),
                  ],
                  run: { toolbox.createNote(title: $0["title"] ?? "Untitled",
                                            body: $0["body"] ?? "", projectName: $0["project"]) }),
            .init(name: "searchNotes",
                  description: "Semantic + keyword search across all notes.",
                  parameters: [.init(name: "query", description: "What to look for", required: true)],
                  run: { toolbox.searchNotes(query: $0["query"] ?? "") }),
            .init(name: "notesFrom",
                  description: "Fetch full text of notes from a given day, e.g. 'yesterday'.",
                  parameters: [.init(name: "day", description: "Day, natural language", required: true)],
                  run: { toolbox.notesFrom(dayDescription: $0["day"] ?? "yesterday") }),
            .init(name: "linkItems",
                  description: "Create a knowledge-graph link between two items (notes, tasks, projects).",
                  parameters: [
                    .init(name: "from", description: "Source item, by name", required: true),
                    .init(name: "to", description: "Target item, by name", required: true),
                    .init(name: "relation", description: "e.g. references, inspired-by, blocks"),
                  ],
                  run: { toolbox.linkItems(from: $0["from"] ?? "", to: $0["to"] ?? "",
                                           relation: $0["relation"] ?? "references") }),
            .init(name: "agenda",
                  description: "What needs attention today: overdue, due, blocked, this week.",
                  parameters: [],
                  run: { _ in toolbox.agenda() }),
            .init(name: "recall",
                  description: "Search past conversation history and completed work.",
                  parameters: [.init(name: "query", description: "What to remember", required: true)],
                  run: { toolbox.recall(query: $0["query"] ?? "") }),
        ]
    }
}
