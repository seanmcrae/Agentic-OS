import Foundation
import SwiftData

/// First-run content so the product feels alive the moment it opens.
/// Everything here is plausibly the user's own world and trivially deletable.
@MainActor
enum SeedData {
    static func seedIfNeeded(_ services: DataServices) {
        let defaults = UserDefaults.standard
        guard !defaults.bool(forKey: "pb.seeded") else { return }
        defaults.set(true, forKey: "pb.seeded")

        let cal = Calendar.current
        func days(_ n: Int) -> Date { cal.date(byAdding: .day, value: n, to: .now)! }

        // Projects
        let website = services.projects.create(
            name: "Website Redesign",
            summary: "New marketing site: brand refresh, faster pages, clearer story."
        )
        let q3 = services.projects.create(
            name: "Q3 Planning",
            summary: "Goals, budget and staffing for the third quarter."
        )

        services.projects.addMilestone(to: website, title: "Design locked", target: days(10))
        services.projects.addMilestone(to: website, title: "Launch", target: days(30))
        services.projects.addMilestone(to: q3, title: "Draft plan circulated", target: days(5))

        // Tasks
        let copy = services.tasks.create(
            title: "Finalize homepage copy", due: days(2), priority: .high, project: website)
        let hero = services.tasks.create(
            title: "Approve hero photography", due: days(1), priority: .normal, project: website)
        let build = services.tasks.create(
            title: "Build homepage in Framer", due: days(7), priority: .high, project: website)
        services.tasks.addDependency(build, blockedBy: copy)
        services.tasks.addDependency(build, blockedBy: hero)

        services.tasks.create(
            title: "Send Q2 invoice to Meridian", due: days(1), priority: .urgent)
        services.tasks.create(
            title: "Book flights for the offsite", due: days(4), priority: .normal)
        services.tasks.create(
            title: "Draft Q3 budget", due: days(3), priority: .high, project: q3)

        // Notes
        let brand = services.notes.create(
            title: "Brand voice principles",
            body: "Confident, never loud. Short sentences. We say what the product does, not what category it is in. Avoid superlatives — show, don't claim.\n\nReference: the Braun product copy archive.",
            tags: ["brand", "writing"], project: website)
        let meeting = services.notes.create(
            title: "Kickoff with the design studio",
            body: "They proposed three directions: Editorial, Instrument, and Atelier. We leaned Instrument — precise grid, generous whitespace, single accent color.\n\nNext step: they deliver moodboards Thursday. Sarah owns the feedback round.",
            tags: ["meeting", "design"], project: website)
        let reading = services.notes.create(
            title: "Notes on 'The Timeless Way of Building'",
            body: "Alexander's 'quality without a name' — places feel alive when patterns resolve real forces, not when they imitate other places. Applies directly to product: copy the force, not the form.",
            tags: ["reading", "ideas"])

        // Knowledge edges
        services.graph.link(from: (.note, meeting.id), to: (.project, website.id), relation: "belongs-to")
        services.graph.link(from: (.note, brand.id), to: (.project, website.id), relation: "belongs-to")
        services.graph.link(from: (.note, reading.id), to: (.note, brand.id), relation: "inspired-by")
        services.graph.link(from: (.task, copy.id), to: (.note, brand.id), relation: "references")
    }
}
