import SwiftUI

/// One settled turn from the transcript.
struct MessageRow: View {
    let message: ChatMessage

    var body: some View {
        if message.role == .user {
            UserBubble(text: message.text)
        } else {
            AgentTurn(text: message.text, toolEvents: message.toolEvents,
                      runningTool: nil, isLive: false)
        }
    }
}

struct UserBubble: View {
    let text: String

    var body: some View {
        HStack {
            Spacer(minLength: Space.vast)
            Text(text)
                .font(Type.body)
                .foregroundStyle(Paper.primary)
                .padding(.horizontal, Space.m)
                .padding(.vertical, Space.s)
                .glass(Radius.control, tint: lumen, depth: 0.5)
        }
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }
}

/// Agent reply: tool cards first (the work, visible), then the prose.
struct AgentTurn: View {
    let text: String
    let toolEvents: [ToolEventRecord]
    let runningTool: String?
    let isLive: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            ForEach(toolEvents) { event in
                ToolActivityCard(event: event,
                                 isRunning: isLive && event.toolName == runningTool && event.detail.isEmpty)
            }
            if !text.isEmpty {
                StreamingText(text: text, isLive: isLive)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.trailing, Space.xl)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// A tool call made visible: a small glass card that slides out of the orb,
/// shows what's happening, and ticks to a check when done.
struct ToolActivityCard: View {
    let event: ToolEventRecord
    var isRunning: Bool = false

    var body: some View {
        HStack(spacing: Space.s) {
            Image(systemName: Self.icon(for: event.toolName))
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Self.hue(for: event.toolName))
                .frame(width: 18)

            Text(event.summary)
                .font(Type.caption)
                .foregroundStyle(Paper.secondary)
                .lineLimit(2)

            Spacer(minLength: Space.xs)

            if isRunning {
                AuroraOrb(energy: 0.8, tint: Self.hue(for: event.toolName), size: 18)
            } else {
                Image(systemName: event.succeeded ? "checkmark" : "exclamationmark.triangle")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(event.succeeded ? DomainHue.note : DomainHue.task)
                    .transition(.scale.combined(with: .opacity))
            }
        }
        .padding(.horizontal, Space.s)
        .padding(.vertical, Space.xs + 2)
        .glass(Radius.control, tint: Self.hue(for: event.toolName), depth: 0.4)
        .frame(maxWidth: 320, alignment: .leading)
        .transition(.asymmetric(
            insertion: .scale(scale: 0.85, anchor: .leading).combined(with: .opacity),
            removal: .opacity))
        .animation(Motion.snap, value: isRunning)
    }

    static func icon(for tool: String) -> String {
        switch tool {
        case "createTask", "updateTask": "circle.badge.plus"
        case "completeTask": "checkmark.circle"
        case "queryTasks", "agenda": "sun.horizon"
        case "createProject", "projectStatus", "addMilestone": "square.stack"
        case "createNote", "appendNote": "square.and.pencil"
        case "searchNotes", "notesFrom": "magnifyingglass"
        case "linkItems": "link"
        case "recall": "clock.arrow.circlepath"
        default: "sparkle"
        }
    }

    static func hue(for tool: String) -> Color {
        switch tool {
        case "createTask", "updateTask", "completeTask", "queryTasks", "agenda": DomainHue.task
        case "createProject", "projectStatus", "addMilestone": lumen
        case "createNote", "appendNote", "searchNotes", "notesFrom": DomainHue.note
        case "linkItems", "recall": DomainHue.knowledge
        default: lumen
        }
    }
}
