import SwiftUI

/// The zoomed-out world behind the thread: Today, Projects, Knowledge.
/// Horizontal paging between spaces; the docked thread tile floats below.
struct SpacesView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var app = app
        VStack(spacing: 0) {
            header
            switcher
                .padding(.bottom, Space.s)

            TabView(selection: $app.space) {
                TodayView().tag(SpaceKind.today)
                ProjectsView().tag(SpaceKind.projects)
                KnowledgeView().tag(SpaceKind.knowledge)
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
            .animation(Motion.glide, value: app.space)

            // Breathing room for the docked thread tile.
            Spacer(minLength: 190)
        }
        .sheet(item: $app.focusedNote) { note in
            NoteView(note: note)
                .presentationDetents([.large])
                .presentationBackground(.clear)
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(app.space.title)
                .displayStyle()
                .contentTransition(.numericText())
                .animation(Motion.glide, value: app.space)
            Spacer()
            PrivacyBadge()
        }
        .padding(.horizontal, Space.gutter)
        .padding(.top, Space.vast)
        .padding(.bottom, Space.m)
    }

    private var switcher: some View {
        HStack(spacing: Space.xs) {
            ForEach(SpaceKind.allCases) { kind in
                Button {
                    withAnimation(Motion.glide) { app.space = kind }
                    Haptics.touch()
                } label: {
                    HStack(spacing: Space.xxs) {
                        Image(systemName: kind.icon)
                            .font(.system(size: 12, weight: .semibold))
                        if app.space == kind {
                            Text(kind.title)
                                .font(Type.caption)
                                .transition(.move(edge: .leading).combined(with: .opacity))
                        }
                    }
                    .foregroundStyle(app.space == kind ? kind.hue : Paper.tertiary)
                    .padding(.horizontal, app.space == kind ? Space.m : Space.s)
                    .padding(.vertical, Space.xs + 1)
                    .glass(Radius.control, tint: app.space == kind ? kind.hue : nil,
                           depth: app.space == kind ? 0.8 : 0.3)
                }
                .buttonStyle(GlassPressStyle())
            }
            Spacer()
        }
        .padding(.horizontal, Space.gutter)
        .animation(Motion.snap, value: app.space)
    }
}

/// Privacy as a designed moment, always visible at the top of the world.
struct PrivacyBadge: View {
    @State private var showStory = false

    var body: some View {
        Button {
            showStory = true
        } label: {
            HStack(spacing: Space.xxs) {
                Image(systemName: "lock.fill")
                    .font(.system(size: 10, weight: .bold))
                MicroLabel(text: "On device", color: Paper.secondary)
            }
            .foregroundStyle(Paper.secondary)
            .padding(.horizontal, Space.s)
            .padding(.vertical, Space.xs)
            .glass(Radius.chip, tint: DomainHue.note, depth: 0.4)
        }
        .buttonStyle(GlassPressStyle())
        .sheet(isPresented: $showStory) {
            PrivacyStoryView()
                .presentationDetents([.medium])
                .presentationBackground(.clear)
        }
    }
}
