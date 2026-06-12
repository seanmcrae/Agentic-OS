import Foundation

/// Events a backend emits while answering. `text` carries the *cumulative*
/// response snapshot (Foundation Models streams snapshots, not deltas, and
/// the UI reveal renderer wants cumulative text anyway).
enum AgentEvent {
    case toolStarted(name: String, summary: String)
    case toolFinished(ToolEventRecord)
    case text(String)
    case done(finalText: String)
}

/// A brain. All three implementations stream the same event language.
@MainActor
protocol ModelBackend {
    var displayName: String { get }
    func reply(to prompt: String) -> AsyncThrowingStream<AgentEvent, Error>
    func resetConversation()
}

/// Shared system instructions, so every backend speaks with one voice.
enum AgentVoice {
    static func instructions(today: Date = .now) -> String {
        """
        You are the user's private chief of staff inside PocketBrains, an \
        offline personal workspace. Today is \(today.formatted(.dateTime.weekday(.wide).month(.wide).day().year())).

        Use the provided tools to act on tasks, projects, notes and the \
        knowledge graph — never claim an action you did not perform with a \
        tool. Prefer acting over asking; only ask when genuinely ambiguous.

        Style: warm, precise, brief. One to three sentences unless asked for \
        more. No markdown headers, no bullet spam, no exclamation marks. \
        Refer to dates the way a person would ("Friday", "tomorrow").
        """
    }

    /// Picks the backend: Foundation Models when Apple Intelligence is
    /// available, optional MLX build, deterministic parser otherwise.
    @MainActor
    static func selectBackend(toolbox: ToolBox) -> ModelBackend {
        #if canImport(FoundationModels)
        if let fm = FoundationModelBackend(toolbox: toolbox) { return fm }
        #endif
        #if POCKETBRAINS_MLX
        if let mlx = MLXBackend(toolbox: toolbox) { return mlx }
        #endif
        return IntentFallbackBackend(toolbox: toolbox)
    }
}
