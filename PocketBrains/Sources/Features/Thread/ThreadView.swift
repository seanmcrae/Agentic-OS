import SwiftUI

/// Home: the conversation. Header carries the horizon grabber (the handle of
/// the signature zoom), the privacy badge, and the agent's presence orb.
struct ThreadView: View {
    @Environment(AppModel.self) private var app
    var horizonProgress: CGFloat = 0
    var horizonDrag: AnyGesture<DragGesture.Value>? = nil

    var body: some View {
        ZStack {
            InkBackdrop()
            VStack(spacing: 0) {
                header
                conversation
                if app.agent.messages.isEmpty && !app.agent.isBusy {
                    SuggestionRail()
                        .padding(.horizontal, Space.gutter)
                        .padding(.bottom, Space.s)
                        .transition(.opacity)
                }
                Composer()
            }
        }
    }

    // MARK: Header

    private var header: some View {
        VStack(spacing: Space.xs) {
            // Horizon grabber: the visible invitation to pull the thread down.
            Capsule()
                .fill(Paper.faint)
                .frame(width: 44, height: 5)
                .padding(.top, Space.xs)
                .padding(.bottom, 2)

            HStack(spacing: Space.s) {
                AuroraOrb(energy: app.agent.isBusy ? 1 : 0.25, size: 30)

                VStack(alignment: .leading, spacing: 1) {
                    Text("PocketBrains")
                        .font(Type.callout)
                        .foregroundStyle(Paper.primary)
                    HStack(spacing: Space.xxs) {
                        Image(systemName: "lock.fill")
                            .font(.system(size: 8, weight: .bold))
                        Text(app.agent.backendName)
                            .font(Type.micro)
                            .tracking(0.4)
                    }
                    .foregroundStyle(Paper.tertiary)
                }

                Spacer()

                Button {
                    app.openSpaces()
                } label: {
                    Image(systemName: "square.grid.2x2")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(Paper.secondary)
                        .frame(width: 36, height: 36)
                        .glass(Radius.control, depth: 0.4)
                }
                .buttonStyle(GlassPressStyle())
            }
            .padding(.horizontal, Space.gutter)
            .padding(.bottom, Space.s)
        }
        .contentShape(Rectangle())
        .modifier(OptionalGesture(gesture: horizonDrag))
    }

    // MARK: Conversation

    private var conversation: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: Space.l) {
                if app.agent.messages.isEmpty && !app.agent.isBusy {
                    welcome
                }
                ForEach(app.agent.messages, id: \.id) { message in
                    MessageRow(message: message)
                }
                if app.agent.isBusy {
                    liveTurn
                }
            }
            .padding(.horizontal, Space.gutter)
            .padding(.vertical, Space.m)
            .scrollTargetLayout()
        }
        .defaultScrollAnchor(.bottom)
        .scrollDismissesKeyboard(.interactively)
        .animation(Motion.glide, value: app.agent.messages.count)
    }

    /// The in-flight reply: thinking orb → tool cards → condensing prose.
    private var liveTurn: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            if app.agent.phase == .thinking {
                HStack(spacing: Space.s) {
                    AuroraOrb(energy: 0.9, size: 26)
                    Text("Thinking…")
                        .font(Type.caption)
                        .foregroundStyle(Paper.tertiary)
                        .transition(.opacity)
                }
            }
            AgentTurn(text: app.agent.liveText,
                      toolEvents: app.agent.liveToolEvents,
                      runningTool: app.agent.runningToolName,
                      isLive: true)
        }
        .animation(Motion.snap, value: app.agent.liveToolEvents.count)
        .animation(Motion.snap, value: app.agent.phase)
    }

    private var welcome: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            AuroraOrb(energy: 0.4, size: 56)
            Text("Everything stays here.")
                .font(Type.heading)
                .foregroundStyle(Paper.primary)
            Text("Tasks, projects, notes and what connects them — managed in one conversation, by a model that lives on this phone. No cloud, no account, no telemetry.")
                .font(Type.body)
                .lineSpacing(4)
                .foregroundStyle(Paper.secondary)
        }
        .padding(.top, Space.huge)
        .padding(.bottom, Space.l)
    }
}

/// Applies a gesture only when one is provided (the docked tile passes none).
private struct OptionalGesture: ViewModifier {
    let gesture: AnyGesture<DragGesture.Value>?

    func body(content: Content) -> some View {
        if let gesture {
            content.gesture(gesture)
        } else {
            content
        }
    }
}
