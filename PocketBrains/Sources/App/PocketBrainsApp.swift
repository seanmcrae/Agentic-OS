import SwiftUI
import SwiftData

@main
struct PocketBrainsApp: App {
    private let container: ModelContainer
    @State private var app: AppModel

    init() {
        let container = Store.makeContainer()
        self.container = container
        self._app = State(initialValue: AppModel(context: container.mainContext))
    }

    var body: some Scene {
        WindowGroup {
            RootShell()
                .environment(app)
                .modelContainer(container)
                .preferredColorScheme(.dark)
                .task {
                    MotionLight.shared.start()
                    app.bootstrap()
                }
        }
    }
}
