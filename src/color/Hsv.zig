//! A color in the [HSV](https://en.wikipedia.org/wiki/HSL_and_HSV) colorspace.
//! - h: Hue, in degrees (0-360, though often normalized to 0-359).
//! - s: Saturation, as a percentage (0-100).
//! - v: Value, as a percentage (0-100).

const std = @import("std");

const conversions = @import("conversions.zig");
const formatting = @import("formatting.zig");
const Hsl = @import("Hsl.zig");
const Lab = @import("Lab.zig");
const Lch = @import("Lch.zig");
const Lms = @import("Lms.zig");
const Oklab = @import("Oklab.zig");
const Oklch = @import("Oklch.zig");
const Rgb = @import("Rgb.zig");
const Rgba = @import("Rgba.zig").Rgba;
const Xyb = @import("Xyb.zig");
const Xyz = @import("Xyz.zig");
const Ycbcr = @import("Ycbcr.zig");

h: f64,
s: f64,
v: f64,

const Hsv = @This();

pub const black: Hsv = .{ .h = 0, .s = 0, .v = 0 };
pub const white: Hsv = .{ .h = 0, .s = 0, .v = 100 };

/// Default formatting with ANSI color output
pub fn format(self: Hsv, writer: *std.Io.Writer) !void {
    return formatting.formatColor(Hsv, self, writer);
}

/// Returns true if saturation is 0 (grayscale).
pub fn isGray(self: Hsv) bool {
    return self.s == 0;
}

/// Converts to grayscale using the value component.
pub fn toGray(self: Hsv) u8 {
    return @intFromFloat(@round(self.v / 100 * 255));
}

/// Converts HSV to RGB color space.
pub fn toRgb(self: Hsv) Rgb {
    return conversions.hsvToRgb(self);
}

/// Converts HSV to RGBA by first converting to RGB and adding alpha.
pub fn toRgba(self: Hsv, alpha: u8) Rgba {
    return self.toRgb().toRgba(alpha);
}

/// Converts HSV to HSL color space using direct conversion.
pub fn toHsl(self: Hsv) Hsl {
    return conversions.hsvToHsl(self);
}

/// Converts HSV to CIE XYZ color space via RGB intermediate conversion.
pub fn toXyz(self: Hsv) Xyz {
    return self.toRgb().toXyz();
}

/// Converts HSV to CIELAB color space using direct conversion.
pub fn toLab(self: Hsv) Lab {
    return conversions.hsvToLab(self);
}

/// Converts HSV to LCh color space via Lab intermediate conversion.
pub fn toLch(self: Hsv) Lch {
    return self.toLab().toLch();
}

/// Converts HSV to LMS cone response via RGB intermediate conversion.
pub fn toLms(self: Hsv) Lms {
    return self.toRgb().toLms();
}

/// Converts HSV to Oklab via RGB intermediate conversion.
pub fn toOklab(self: Hsv) Oklab {
    return self.toRgb().toOklab();
}

/// Converts HSV to Oklch via RGB intermediate conversion.
pub fn toOklch(self: Hsv) Oklch {
    return self.toRgb().toOklch();
}

/// Converts HSV to XYB via RGB intermediate conversion.
pub fn toXyb(self: Hsv) Xyb {
    return self.toRgb().toXyb();
}

/// Converts HSV to YCbCr via RGB.
pub fn toYcbcr(self: Hsv) Ycbcr {
    return self.toRgb().toYcbcr();
}

/// Alpha blends the given RGBA color onto this HSV color in-place.
pub fn blend(self: *Hsv, color: Rgba) void {
    var rgb = self.toRgb();
    rgb.blend(color);
    self.* = rgb.toHsv();
}
