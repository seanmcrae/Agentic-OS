import Foundation
import SwiftData
import SwiftUI

/// Owns the conversation: picks the brain, streams its events into UI state,
/// and persists every turn. The thread view renders only from this object.
@MainActor
@Observable
final class AgentOrchestrator {
    enum Phase: Equatable {
        case idle
        case thinking       // prompt sent, nothing back yet
        case acting         // a tool is running
        case streaming      // tokens flowing
    }

    private(set) var phase: Phase = .idle
    /// Cumulative text of the in-flight reply.
    private(set) var liveText: String = ""
    /// Tool cards for the in-flight reply (running + finished).
    private(set) var liveToolEvents: [ToolEventRecord] = []
    private(set) var runningToolName: String?
    private(set) var messages: [ChatMessage] = []
    private(set) var backendName: String = ""

    private let context: ModelContext
    private var backend: ModelBackend
    private var currentTurn: Task<Void, Never>?

    init(context: ModelContext, toolbox: ToolBox) {
        self.context = context
        self.backend = AgentVoice.selectBackend(toolbox: toolbox)
        self.backendName = backend.displayName
        self.messages = context.fetchAll(ChatMessage.self, sortBy: [.init(\.createdAt)])
    }

    var isBusy: Bool { phase != .idle }

    func send(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isBusy else { return }

        let userMessage = ChatMessage(role: .user, text: trimmed)
        context.insert(userMessage)
        try? context.save()
        messages.append(userMessage)

        phase = .thinking
        liveText = ""
        liveToolEvents = []
        runningToolName = nil

        currentTurn = Task { @MainActor in
            do {
                for try await event in backend.reply(to: trimmed) {
                    handle(event)
                }
            } catch {
                finishTurn(with: liveText.isEmpty
                    ? "I hit a snag generating that — try once more, or rephrase."
                    : liveText)
            }
        }
    }

    func cancel() {
        currentTurn?.cancel()
        if !liveText.isEmpty { finishTurn(with: liveText) } else { phase = .idle }
    }

    func clearHistory() {
        for message in messages { context.delete(message) }
        try? context.save()
        messages = []
        backend.resetConversation()
    }

    // MARK: - Event handling

    private func handle(_ event: AgentEvent) {
        switch event {
        case .toolStarted(let name, let summary):
            phase = .acting
            runningToolName = name
            liveToolEvents.append(.init(toolName: name, summary: summary,
                                        detail: "", succeeded: true))
        case .toolFinished(let record):
            runningToolName = nil
            if let index = liveToolEvents.lastIndex(where: { $0.toolName == record.toolName && $0.detail.isEmpty }) {
                liveToolEvents[index] = record
            } else {
                liveToolEvents.append(record)
            }
            Haptics.tick()
        case .text(let snapshot):
            phase = .streaming
            liveText = snapshot
        case .done(let finalText):
            finishTurn(with: finalText)
        }
    }

    private func finishTurn(with text: String) {
        let reply = ChatMessage(role: .agent, text: text)
        reply.toolEvents = liveToolEvents
        context.insert(reply)
        try? context.save()
        messages.append(reply)
        liveText = ""
        liveToolEvents = []
        runningToolName = nil
        phase = .idle
    }
}
