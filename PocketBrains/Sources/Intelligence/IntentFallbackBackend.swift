import Foundation

/// Deterministic floor: a rule-based intent parser that drives the same
/// ToolBox. No generation, but "remind me to send the invoice Friday" still
/// creates the task. Guarantees the product works on any device, simulator
/// included, with zero model assets.
@MainActor
final class IntentFallbackBackend: ModelBackend {
    let displayName = "Quick intents · offline"

    private let toolbox: ToolBox

    init(toolbox: ToolBox) {
        self.toolbox = toolbox
    }

    func resetConversation() {}

    func reply(to prompt: String) -> AsyncThrowingStream<AgentEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @MainActor in
                let (toolName, activity, result, reply) = Self.route(prompt, toolbox: toolbox)
                if let toolName, let result {
                    continuation.yield(.toolStarted(name: toolName, summary: activity))
                    try? await Task.sleep(for: .milliseconds(350)) // let the card breathe
                    continuation.yield(.toolFinished(result.record(toolName: toolName)))
                }
                // Stream the reply in word chunks so the condensation reveal
                // behaves identically to a real model.
                var shown = ""
                for word in reply.split(separator: " ", omittingEmptySubsequences: false) {
                    shown += (shown.isEmpty ? "" : " ") + word
                    continuation.yield(.text(shown))
                    try? await Task.sleep(for: .milliseconds(24))
                }
                continuation.yield(.done(finalText: reply))
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Routing

    @MainActor
    private static func route(_ prompt: String, toolbox: ToolBox)
        -> (tool: String?, activity: String, result: ToolResult?, reply: String) {
        let lower = prompt.lowercased()

        func strip(_ text: String, prefixes: [String]) -> String {
            var out = text
            for p in prefixes where out.lowercased().hasPrefix(p) {
                out = String(out.dropFirst(p.count))
            }
            return out.trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
        }

        // Agenda
        if lower.contains("agenda") || lower.contains("what's today") || lower.contains("what do i")
            || lower.contains("needs my attention") || (lower.contains("today") && lower.contains("due")) {
            let r = toolbox.agenda()
            return ("agenda", "Composing agenda", r, r.detail)
        }

        // Blockers / project status
        if lower.contains("blocking") || lower.contains("blocked") || lower.contains("status of") {
            let name = strip(prompt, prefixes: ["what's blocking", "whats blocking", "what is blocking", "status of", "what's the status of"])
            if !name.isEmpty, let r = optional(toolbox.projectStatus(name: name)) {
                return ("projectStatus", "Checking project", r, r.detail)
            }
            let r = toolbox.queryTasks(filter: "blocked")
            return ("queryTasks", "Looking at blocked tasks", r, r.detail)
        }

        // Complete
        if let range = lower.range(of: #"(complete|finish|mark .* done|i did|check off) "#, options: .regularExpression) {
            let query = String(prompt[range.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let r = toolbox.completeTask(query: query)
            return ("completeTask", "Completing task", r, r.succeeded ? "Done — \(r.summary.lowercased())." : r.detail)
        }

        // Reminders / tasks
        if lower.hasPrefix("remind me") || lower.contains("add a task") || lower.contains("add task")
            || lower.hasPrefix("todo") || lower.hasPrefix("i need to") {
            var title = strip(prompt, prefixes: ["remind me to", "remind me", "add a task to", "add a task:", "add task", "todo:", "todo", "i need to"])
            let due = NaturalDateParser.parse(prompt).map { _ in dueText(in: prompt) }
            if let dueWords = due { title = removeSuffix(title, dueWords) }
            let r = toolbox.createTask(title: sentenceCase(title), due: due)
            return ("createTask", "Creating task", r, "Got it — \(r.summary.lowercased()).")
        }

        // Notes — capture
        if lower.hasPrefix("note") || lower.contains("take a note") || lower.hasPrefix("jot") {
            let body = strip(prompt, prefixes: ["note that", "note:", "note", "take a note:", "take a note", "jot down", "jot"])
            let title = String(body.prefix(48))
            let r = toolbox.createNote(title: sentenceCase(title), body: body)
            return ("createNote", "Writing note", r, "Noted.")
        }

        // Summaries of a day's notes
        if lower.contains("summarize") || lower.contains("summary") {
            let day = lower.contains("yesterday") ? "yesterday" : lower.contains("today") ? "today" : "yesterday"
            let r = toolbox.notesFrom(dayDescription: day)
            let extract = extractiveSummary(of: r.detail)
            return ("notesFrom", "Reading notes", r, extract)
        }

        // Link
        if lower.hasPrefix("link ") || lower.contains("connect ") {
            let body = strip(prompt, prefixes: ["link", "connect"])
            let parts = body.components(separatedBy: " to ")
            if parts.count == 2 {
                let r = toolbox.linkItems(from: parts[0].trimmingCharacters(in: .whitespaces),
                                          to: parts[1].trimmingCharacters(in: .whitespaces))
                return ("linkItems", "Linking items", r, r.succeeded ? "\(r.summary)." : r.detail)
            }
        }

        // Search
        if lower.contains("find") || lower.contains("search") || lower.contains("notes about")
            || lower.hasPrefix("what do my notes") {
            let query = strip(prompt, prefixes: ["find notes about", "find", "search notes for", "search for", "search", "notes about"])
            let r = toolbox.searchNotes(query: query)
            return ("searchNotes", "Searching notes", r, r.detail)
        }

        // Recall
        if lower.contains("did i") || lower.contains("when did") || lower.contains("remember") {
            let r = toolbox.recall(query: strip(prompt, prefixes: ["do you remember", "remember", "when did i", "did i"]))
            return ("recall", "Remembering", r, r.detail)
        }

        // New project
        if lower.contains("new project") || lower.contains("create a project") || lower.contains("start a project") {
            let name = strip(prompt, prefixes: ["new project called", "new project:", "new project", "create a project called", "create a project", "start a project called", "start a project"])
            let r = toolbox.createProject(name: sentenceCase(name))
            return ("createProject", "Creating project", r, "\(r.summary). It's waiting in your Projects space.")
        }

        // Default: be honest about the floor mode.
        let r = toolbox.agenda()
        return ("agenda", "Composing agenda", r,
                "I'm running in quick-intent mode on this device, so I keep to direct phrasing — try \"remind me to…\", \"what's blocking…\", or \"summarize my notes from yesterday\". Meanwhile, here's today:\n\(r.detail)")
    }

    private static func optional(_ r: ToolResult) -> ToolResult? { r.succeeded ? r : nil }

    private static func dueText(in prompt: String) -> String {
        let candidates = ["today", "tomorrow", "next week", "monday", "tuesday", "wednesday",
                          "thursday", "friday", "saturday", "sunday"]
        let lower = prompt.lowercased()
        for c in candidates where lower.contains(c) { return c }
        return prompt // NSDataDetector path will find the explicit date
    }

    private static func removeSuffix(_ text: String, _ words: String) -> String {
        var out = text
        for token in [" on \(words)", " by \(words)", " \(words)"] {
            if let range = out.lowercased().range(of: token) {
                out.removeSubrange(range)
                break
            }
        }
        return out.trimmingCharacters(in: .whitespaces)
    }

    private static func sentenceCase(_ text: String) -> String {
        guard let first = text.first else { return text }
        return first.uppercased() + text.dropFirst()
    }

    /// Cheap extractive summary: first sentence of each note section.
    private static func extractiveSummary(of detail: String) -> String {
        let sections = detail.components(separatedBy: "\n\n")
        let leads = sections.compactMap { section -> String? in
            let flat = section.replacingOccurrences(of: "\n", with: " ")
            guard let stop = flat.firstIndex(where: { ".!?".contains($0) }) else {
                return flat.isEmpty ? nil : String(flat.prefix(120))
            }
            return String(flat[...stop])
        }
        guard !leads.isEmpty else { return "Nothing to summarize from that day." }
        return leads.joined(separator: " ")
    }
}
