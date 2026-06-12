import SwiftUI

/// Privacy, celebrated as a designed moment: the receipt of what never
/// happened. Invite the user to prove it with Airplane Mode.
struct PrivacyStoryView: View {
    @Environment(AppModel.self) private var app
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            InkBackdrop()
            VStack(spacing: Space.l) {
                AuroraOrb(energy: 0.45, tint: DomainHue.note, size: 72)
                    .padding(.top, Space.xl)

                Text("Nothing leaves this device.")
                    .font(Type.heading)
                    .foregroundStyle(Paper.primary)

                VStack(spacing: Space.s) {
                    receiptRow(label: "Inference", value: app.agent.backendName)
                    receiptRow(label: "Storage", value: "This phone, encrypted at rest")
                    receiptRow(label: "Network calls", value: "Zero")
                    receiptRow(label: "Accounts", value: "None")
                    receiptRow(label: "Telemetry", value: "None")
                }
                .padding(Space.l)
                .glass(Radius.card, tint: DomainHue.note, depth: 0.7)
                .padding(.horizontal, Space.gutter)

                Text("Don't take our word for it — turn on Airplane Mode.\nEverything keeps working.")
                    .font(Type.callout)
                    .foregroundStyle(Paper.tertiary)
                    .multilineTextAlignment(.center)

                Spacer()
            }
        }
    }

    private func receiptRow(label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            MicroLabel(text: label)
            Spacer()
            Text(value)
                .font(Type.caption)
                .foregroundStyle(Paper.primary)
                .multilineTextAlignment(.trailing)
        }
    }
}
