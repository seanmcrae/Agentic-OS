import SwiftUI

/// The full type scale. No view declares an ad-hoc font.
enum Type {
    /// New York display — project headlines only.
    static let display = Font.system(size: 34, weight: .semibold, design: .serif)
    static let title = Font.system(size: 28, weight: .bold)
    static let heading = Font.system(size: 22, weight: .semibold)
    static let body = Font.system(size: 17, weight: .regular)
    static let bodyMedium = Font.system(size: 17, weight: .medium)
    static let callout = Font.system(size: 15, weight: .medium)
    static let caption = Font.system(size: 13, weight: .medium)
    static let micro = Font.system(size: 11, weight: .semibold)
    static let mono = Font.system(size: 13, weight: .medium, design: .monospaced)
}

struct MicroLabel: View {
    let text: String
    var color: Color = Paper.tertiary

    var body: some View {
        Text(text.uppercased())
            .font(Type.micro)
            .tracking(0.6)
            .foregroundStyle(color)
    }
}

extension Text {
    func displayStyle() -> some View {
        self.font(Type.display).tracking(-0.5).foregroundStyle(Paper.primary)
    }
}
