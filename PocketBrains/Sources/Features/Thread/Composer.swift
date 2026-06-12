import SwiftUI

/// The input bar. A glass field with a send control that is the agent's orb:
/// idle it's a quiet arrow, while the agent works it becomes living light
/// (tap to cancel).
struct Composer: View {
    @Environment(AppModel.self) private var app
    @State private var draft = ""
    @FocusState private var focused: Bool

    var body: some View {
        HStack(alignment: .bottom, spacing: Space.s) {
            TextField("Ask, capture, or command…", text: $draft, axis: .vertical)
                .font(Type.body)
                .foregroundStyle(Paper.primary)
                .tint(lumen)
                .lineLimit(1...5)
                .focused($focused)
                .padding(.horizontal, Space.m)
                .padding(.vertical, Space.s + 2)
                .glass(Radius.control, depth: 0.6)
                .onSubmit(send)

            Button(action: app.agent.isBusy ? cancel : send) {
                ZStack {
                    Circle()
                        .fill(canSend || app.agent.isBusy ? lumen : Ink.l2)
                        .overlay {
                            Circle().strokeBorder(.white.opacity(0.14), lineWidth: 0.75)
                        }
                    if app.agent.isBusy {
                        AuroraOrb(energy: 1, tint: Ink.l0, size: 26)
                    } else {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 16, weight: .bold))
                            .foregroundStyle(canSend ? Ink.l0 : Paper.tertiary)
                    }
                }
                .frame(width: 40, height: 40)
                .animation(Motion.snap, value: canSend)
                .animation(Motion.snap, value: app.agent.isBusy)
            }
            .buttonStyle(GlassPressStyle(scale: 0.9))
            .disabled(!canSend && !app.agent.isBusy)
        }
        .padding(.horizontal, Space.gutter)
        .padding(.vertical, Space.s)
    }

    private var canSend: Bool {
        !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func send() {
        guard canSend, !app.agent.isBusy else { return }
        Haptics.commit()
        app.agent.send(draft)
        draft = ""
    }

    private func cancel() {
        Haptics.touch()
        app.agent.cancel()
    }
}

/// First-run invitations — three example commands that teach the grammar.
struct SuggestionRail: View {
    @Environment(AppModel.self) private var app

    private let suggestions = [
        "What needs my attention today?",
        "Remind me to send the invoice Friday",
        "What's blocking the website redesign?",
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: Space.xs) {
            ForEach(suggestions, id: \.self) { suggestion in
                Button {
                    app.agent.send(suggestion)
                } label: {
                    HStack(spacing: Space.xs) {
                        Image(systemName: "sparkle")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(lumen)
                        Text(suggestion)
                            .font(Type.callout)
                            .foregroundStyle(Paper.secondary)
                    }
                    .padding(.horizontal, Space.m)
                    .padding(.vertical, Space.s)
                    .glass(Radius.control, tint: lumen, depth: 0.35)
                }
                .buttonStyle(GlassPressStyle())
            }
        }
    }
}
