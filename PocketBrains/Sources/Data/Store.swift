import Foundation
import SwiftData

/// Container bootstrap + the one shared ModelContext the app and agent use.
@MainActor
enum Store {
    static let schema = Schema([
        TaskItem.self, Project.self, Milestone.self, Note.self,
        KnowledgeLink.self, ChatMessage.self, EmbeddingRecord.self,
    ])

    static func makeContainer(inMemory: Bool = false) -> ModelContainer {
        let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: inMemory)
        do {
            return try ModelContainer(for: schema, configurations: [config])
        } catch {
            // Schema migration dead-end during development: fall back to a
            // fresh in-memory store rather than crashing the demo.
            return try! ModelContainer(
                for: schema,
                configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)]
            )
        }
    }
}

extension ModelContext {
    func fetchAll<T: PersistentModel>(_ type: T.Type,
                                      sortBy: [SortDescriptor<T>] = []) -> [T] {
        (try? fetch(FetchDescriptor<T>(sortBy: sortBy))) ?? []
    }
}
