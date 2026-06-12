import Foundation

/// Turns "Friday", "tomorrow", "June 20", "in 3 days" into Dates.
/// Used by the fallback intent backend and to harden model tool arguments.
enum NaturalDateParser {
    static func parse(_ text: String, from reference: Date = .now) -> Date? {
        let lower = text.lowercased()
        let cal = Calendar.current

        if lower.contains("today") { return cal.startOfDay(for: reference) }
        if lower.contains("yesterday") { return cal.date(byAdding: .day, value: -1, to: cal.startOfDay(for: reference)) }
        if lower.contains("tomorrow") { return cal.date(byAdding: .day, value: 1, to: cal.startOfDay(for: reference)) }
        if lower.contains("next week") { return cal.date(byAdding: .day, value: 7, to: cal.startOfDay(for: reference)) }

        if let range = lower.range(of: #"in (\d+) days?"#, options: .regularExpression),
           let n = Int(lower[range].split(separator: " ")[1]) {
            return cal.date(byAdding: .day, value: n, to: cal.startOfDay(for: reference))
        }

        // Weekday names → the next occurrence.
        let weekdays = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
        for (index, name) in weekdays.enumerated() where lower.contains(name) {
            let target = index + 1 // Calendar weekday is 1-based
            var date = cal.startOfDay(for: reference)
            for _ in 0..<8 {
                date = cal.date(byAdding: .day, value: 1, to: date)!
                if cal.component(.weekday, from: date) == target { return date }
            }
        }

        // ISO and natural dates via the system detector.
        if let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.date.rawValue) {
            let match = detector.firstMatch(
                in: text, range: NSRange(text.startIndex..., in: text))
            if let date = match?.date { return date }
        }
        return nil
    }

    /// Human phrasing for the UI: "Today", "Tomorrow", "Friday", "Jun 24".
    static func describe(_ date: Date) -> String {
        let cal = Calendar.current
        if cal.isDateInToday(date) { return "Today" }
        if cal.isDateInTomorrow(date) { return "Tomorrow" }
        if date < .now { return date.formatted(.dateTime.month(.abbreviated).day()) + " · overdue" }
        if let days = cal.dateComponents([.day], from: cal.startOfDay(for: .now), to: date).day,
           days < 7 {
            return date.formatted(.dateTime.weekday(.wide))
        }
        return date.formatted(.dateTime.month(.abbreviated).day())
    }
}
