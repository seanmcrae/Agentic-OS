import SwiftUI

/// 4pt grid. These are the only spacings used anywhere.
enum Space {
    static let xxs: CGFloat = 4
    static let xs: CGFloat = 8
    static let s: CGFloat = 12
    static let m: CGFloat = 16
    static let l: CGFloat = 20
    static let xl: CGFloat = 24
    static let xxl: CGFloat = 32
    static let huge: CGFloat = 40
    static let vast: CGFloat = 56
    /// Horizontal screen gutter.
    static let gutter: CGFloat = 20
}

/// Corner radii — always continuous.
enum Radius {
    static let chip: CGFloat = 10
    static let control: CGFloat = 16
    static let card: CGFloat = 22
    static let sheet: CGFloat = 28
}

extension RoundedRectangle {
    static func chip() -> RoundedRectangle { .init(cornerRadius: Radius.chip, style: .continuous) }
    static func control() -> RoundedRectangle { .init(cornerRadius: Radius.control, style: .continuous) }
    static func card() -> RoundedRectangle { .init(cornerRadius: Radius.card, style: .continuous) }
    static func sheet() -> RoundedRectangle { .init(cornerRadius: Radius.sheet, style: .continuous) }
}
