import SwiftUI

/// PocketBrains color system. See docs/DESIGN.md — these are the only colors
/// in the app. Hue is identity; light is the only gradient.
enum Ink {
    static let l0 = Color(hex: 0x0B0C0F)   // app background
    static let l1 = Color(hex: 0x13151A)   // recessed wells
    static let l2 = Color(hex: 0x1C1F26)   // glass plate base
    static let line = Color.white.opacity(0.08)
}

enum Paper {
    static let primary = Color(hex: 0xF4F2EC)
    static let secondary = Color(hex: 0xF4F2EC).opacity(0.60)
    static let tertiary = Color(hex: 0xF4F2EC).opacity(0.38)
    static let faint = Color(hex: 0xF4F2EC).opacity(0.16)
}

/// The single brand accent: candlelight gold.
let lumen = Color(hex: 0xE4C56F)

/// Domain identity hues — tints inside glass, never fills.
enum DomainHue {
    static let task = Color(hex: 0xD9A441)      // citrine
    static let note = Color(hex: 0x8FAE8B)      // moss
    static let knowledge = Color(hex: 0x7FB4C9) // glacier
}

/// Curated 8-hue wheel for project identity. Assigned round-robin at
/// creation so adjacent projects never share a hue.
enum ProjectHue: Int, CaseIterable, Codable {
    case ember, citrine, moss, jade, glacier, dusk, rosewood, slate

    var color: Color {
        switch self {
        case .ember:    Color(hex: 0xC96F4A)
        case .citrine:  Color(hex: 0xD9A441)
        case .moss:     Color(hex: 0x8FAE8B)
        case .jade:     Color(hex: 0x5FA68C)
        case .glacier:  Color(hex: 0x7FB4C9)
        case .dusk:     Color(hex: 0x8E8BB8)
        case .rosewood: Color(hex: 0xB87A8E)
        case .slate:    Color(hex: 0x96A1AD)
        }
    }

    var name: String {
        switch self {
        case .ember: return "Ember"
        case .citrine: return "Citrine"
        case .moss: return "Moss"
        case .jade: return "Jade"
        case .glacier: return "Glacier"
        case .dusk: return "Dusk"
        case .rosewood: return "Rosewood"
        case .slate: return "Slate"
        }
    }
}

extension Color {
    init(hex: UInt32) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255
        )
    }
}
