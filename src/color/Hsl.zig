//! A color in the [HSL](https://en.wikipedia.org/wiki/HSL_and_HSV) colorspace.
//! - h: Hue, in degrees (0-360, though often normalized to 0-359).
//! - s: Saturation, as a percentage (0-100).
//! - l: Lightness, as a percentage (0-100).

const std = @import("std");

const conversions = @import("conversions.zig");
const formatting = @import("formatting.zig");

h: f64,
s: f64,
l: f64,

pub const black: @This() = .{ .h = 0, .s = 0, .l = 0 };
pub const white: @This() = .{ .h = 0, .s = 0, .l = 100 };

const Self = @This();

pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
    return formatting.formatColor(Self, self, fmt, options, writer);
}

pub fn isGray(self: Self) bool {
    return self.s == 0;
}

pub fn toGray(self: Self) u8 {
    return @intFromFloat(@round(self.l / 100 * 255));
}

pub fn toRgb(self: Self) @import("Rgb.zig") {
    return conversions.hslToRgb(self);
}

pub fn toHsv(self: Self) @import("Hsv.zig") {
    return conversions.hsvToHsl(self.toRgb().toHsv());
}

pub fn toXyz(self: Self) @import("Xyz.zig") {
    return self.toRgb().toXyz();
}

pub fn toLab(self: Self) @import("Lab.zig") {
    return conversions.hslToLab(self);
}

pub fn toLms(self: Self) @import("Lms.zig") {
    return self.toRgb().toLms();
}

pub fn toOklab(self: Self) @import("Oklab.zig") {
    return self.toRgb().toOklab();
}

pub fn toXyb(self: Self) @import("Xyb.zig") {
    return self.toRgb().toXyb();
}

pub fn blend(self: *Self, color: @import("Rgba.zig")) void {
    var rgb = self.toRgb();
    rgb.blend(color);
    self.* = rgb.toHsl();
}
