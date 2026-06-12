import SwiftUI

/// A note, opened from the constellation or the thread: editable body on
/// glass, tags, backlinks, and semantically related notes.
struct NoteView: View {
    @Environment(AppModel.self) private var app
    @Environment(\.dismiss) private var dismiss
    let note: Note

    @State private var draft: String = ""

    var body: some View {
        ZStack {
            InkBackdrop()
            VStack(alignment: .leading, spacing: 0) {
                header
                ScrollView {
                    VStack(alignment: .leading, spacing: Space.l) {
                        editor
                        backlinks
                        related
                    }
                    .padding(.horizontal, Space.gutter)
                    .padding(.bottom, Space.xl)
                }
                .scrollIndicators(.hidden)
                .scrollDismissesKeyboard(.interactively)
            }
        }
        .onAppear { draft = note.body }
        .onDisappear { saveIfNeeded() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: Space.xs) {
            HStack {
                MicroLabel(text: note.modifiedAt.formatted(.dateTime.month(.wide).day()),
                           color: DomainHue.note.opacity(0.9))
                Spacer()
                Button {
                    saveIfNeeded()
                    dismiss()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(Paper.secondary)
                        .frame(width: 32, height: 32)
                        .glass(Radius.control, depth: 0.4)
                }
                .buttonStyle(GlassPressStyle())
            }
            Text(note.title)
                .font(Type.display)
                .tracking(-0.5)
                .foregroundStyle(Paper.primary)
            if !note.tags.isEmpty {
                HStack(spacing: Space.xs) {
                    ForEach(note.tags, id: \.self) { tag in
                        GlassChip(text: tag, tint: DomainHue.note)
                    }
                }
            }
        }
        .padding(.horizontal, Space.gutter)
        .padding(.top, Space.xl)
        .padding(.bottom, Space.m)
    }

    private var editor: some View {
        TextEditor(text: $draft)
            .font(Type.body)
            .lineSpacing(5)
            .foregroundStyle(Paper.primary)
            .tint(lumen)
            .scrollContentBackground(.hidden)
            .frame(minHeight: 180)
            .padding(Space.s)
            .glass(Radius.card, tint: DomainHue.note, depth: 0.5)
    }

    @ViewBuilder
    private var backlinks: some View {
        let links = app.services.graph.links(touching: note.id)
        if !links.isEmpty {
            VStack(alignment: .leading, spacing: Space.xs) {
                SectionHeader(title: "Connections", detail: "\(links.count)")
                ForEach(links, id: \.id) { link in
                    let outbound = link.fromID == note.id
                    let otherKind = outbound ? link.toKind : link.fromKind
                    let otherID = outbound ? link.toID : link.fromID
                    if let title = app.services.graph.title(of: otherKind, id: otherID) {
                        HStack(spacing: Space.s) {
                            Image(systemName: outbound ? "arrow.up.right" : "arrow.down.left")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundStyle(DomainHue.knowledge)
                            Text(title)
                                .font(Type.callout)
                                .foregroundStyle(Paper.primary)
                            Spacer()
                            Text(link.relation)
                                .font(Type.micro)
                                .foregroundStyle(Paper.tertiary)
                        }
                        .padding(.horizontal, Space.m)
                        .padding(.vertical, Space.s)
                        .glass(Radius.control, tint: DomainHue.knowledge, depth: 0.35)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var related: some View {
        let companions = app.semanticIndex.search(note.title + " " + note.body, limit: 4)
            .filter { $0.id != note.id }
        if !companions.isEmpty {
            VStack(alignment: .leading, spacing: Space.xs) {
                SectionHeader(title: "Resonates with")
                ForEach(companions, id: \.id) { companion in
                    HStack(spacing: Space.s) {
                        Image(systemName: "sparkle")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(DomainHue.note)
                        Text(companion.title)
                            .font(Type.callout)
                            .foregroundStyle(Paper.secondary)
                        Spacer()
                    }
                    .padding(.horizontal, Space.m)
                    .padding(.vertical, Space.s)
                    .glass(Radius.control, depth: 0.3)
                }
            }
        }
    }

    private func saveIfNeeded() {
        guard draft != note.body else { return }
        note.body = draft
        note.modifiedAt = .now
        try? app.services.context.save()
        app.semanticIndex.index(note: note)
    }
}
