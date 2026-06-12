// PocketBrains — Deep Glass material.
// Real refraction against a rounded-rect SDF, motion-tracked edge specular,
// and content-identity tinting. Applied via SwiftUI .layerEffect.

#include <metal_stdlib>
#include <SwiftUI/SwiftUI_Metal.h>
using namespace metal;

// Signed distance to a rounded rectangle centered in `size`.
static float roundedRectSDF(float2 p, float2 size, float radius) {
    float2 half_ = size * 0.5;
    float2 q = abs(p - half_) - (half_ - radius);
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

static float2 sdfNormal(float2 p, float2 size, float radius) {
    const float e = 1.0;
    float dx = roundedRectSDF(p + float2(e, 0), size, radius)
             - roundedRectSDF(p - float2(e, 0), size, radius);
    float dy = roundedRectSDF(p + float2(0, e), size, radius)
             - roundedRectSDF(p - float2(0, e), size, radius);
    float2 n = float2(dx, dy);
    float len = length(n);
    return len > 1e-5 ? n / len : float2(0, -1);
}

/// The glass plate. `light` is the device-motion light direction.
[[ stitchable ]] half4 liquidGlass(float2 position,
                                   SwiftUI::Layer layer,
                                   float2 size,
                                   float radius,
                                   float2 light,
                                   half4 tint,
                                   float time) {
    float d = roundedRectSDF(position, size, radius);
    if (d > 0.5) { return half4(0); }

    float2 n = sdfNormal(position, size, radius);

    // --- Refraction: bend samples near the rim, as a thick lens edge would.
    float rim = 1.0 - smoothstep(0.0, min(28.0, radius * 1.6), -d);
    float bend = rim * rim * 9.0;
    float2 refr = position - n * bend;

    // Mild chromatic split on the bend for lensy realism.
    // (Float math throughout; convert to half once at the end.)
    float4 c;
    c.r = float(layer.sample(refr + n * 1.4).r);
    c.g = float(layer.sample(refr).g);
    c.b = float(layer.sample(refr - n * 1.4).b);
    c.a = float(layer.sample(refr).a);

    // --- Identity tint: breathes into the body of the glass, strongest low.
    float4 tintF = float4(tint);
    float depthGrad = position.y / max(size.y, 1.0);
    c.rgb = mix(c.rgb, tintF.rgb, tintF.a * (0.10 + 0.10 * depthGrad));

    // --- Specular bead: a bright thread along the rim facing the light.
    float facing = max(0.0, dot(n, -normalize(light + float2(1e-4))));
    float edge = smoothstep(2.0, 0.0, abs(d + 1.5));        // thin band at rim
    float spec = edge * pow(facing, 3.0);
    // A slow shimmer travels the bead so the glass feels alive even at rest.
    float shimmer = 0.85 + 0.15 * sin(time * 0.7 + (position.x + position.y) * 0.02);
    c.rgb += spec * 0.55 * shimmer;

    // --- Broad sheen: soft diagonal light across the face.
    float2 uv = position / size;
    float sheen = max(0.0, 1.0 - distance(uv, float2(0.5) - light * 0.45) * 1.4);
    c.rgb += sheen * sheen * 0.05;

    // Feathered anti-aliased rim.
    c.a *= smoothstep(0.5, -0.8, d);
    return half4(c);
}

/// Aurora field — the thinking orb and the Knowledge space atmosphere.
/// Domain-warped fbm noise rendered as drifting light, tinted by `tint`.
static float hash21(float2 p) {
    p = fract(p * float2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}

static float vnoise(float2 p) {
    float2 i = floor(p), f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + float2(1, 0));
    float c = hash21(i + float2(0, 1));
    float d = hash21(i + float2(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

static float fbm(float2 p) {
    float v = 0.0, amp = 0.5;
    for (int i = 0; i < 4; i++) {
        v += amp * vnoise(p);
        p = p * 2.03 + float2(13.7, 7.1);
        amp *= 0.5;
    }
    return v;
}

[[ stitchable ]] half4 aurora(float2 position,
                              half4 existing,
                              float2 size,
                              float time,
                              half4 tint,
                              float intensity) {
    float2 uv = position / max(size.x, size.y);
    float t = time * 0.12;
    // Domain warp: noise displaced by noise — liquid light.
    float2 warp = float2(fbm(uv * 3.0 + t), fbm(uv * 3.0 - t + 5.2));
    float field = fbm(uv * 2.2 + warp * 1.6 + float2(0.0, t * 0.6));
    float glow = smoothstep(0.35, 0.9, field) * intensity;

    // Radial falloff so the field sits as a presence, not a wallpaper.
    float r = distance(position / size, float2(0.5));
    glow *= smoothstep(0.72, 0.18, r);

    half3 col = tint.rgb * half(glow);
    // Warm core where the field is strongest.
    col += half3(0.89, 0.77, 0.43) * half(pow(glow, 3.0) * 0.6);
    // Composite the light over whatever the view already drew.
    half a = half(glow * tint.a);
    return half4(existing.rgb * (1.0h - a) + col, max(existing.a, a));
}

/// Specular sweep used by the streaming-text condensation reveal: a band of
/// light passes once across freshly arrived glyphs.
[[ stitchable ]] half4 specularSweep(float2 position,
                                     half4 color,
                                     float2 size,
                                     float progress) {
    if (color.a < 0.001) { return color; }
    float x = position.x / max(size.x, 1.0);
    float band = 1.0 - smoothstep(0.0, 0.18, abs(x - progress));
    half3 boosted = color.rgb + half3(band * band * 0.85);
    return half4(boosted, color.a);
}

/// Fine monochrome grain over the Ink background — kills banding, adds tooth.
[[ stitchable ]] half4 filmGrain(float2 position, half4 color, float time) {
    float g = hash21(position + fract(time) * 100.0) - 0.5;
    return half4(color.rgb + half3(g * 0.015), color.a);
}
