//! Color module - All color types and utilities
//!
//! This module provides a unified interface to all color types in the system.
//! Each color type is implemented as a separate file using Zig's file-as-struct pattern.

const std = @import("std");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectEqualDeep = std.testing.expectEqualDeep;
const clamp = std.math.clamp;
const lerp = std.math.lerp;
const pow = std.math.pow;

const blending = @import("color/blending.zig");
pub const Blending = blending.Blending;
pub const blendColors = blending.blendColors;
const conversions = @import("color/conversions.zig");
pub const convertColor = conversions.convertColor;
pub const isColor = conversions.isColor;
const getSimpleTypeName = @import("meta.zig").getSimpleTypeName;

pub const ColorSpace = enum {
    gray,
    hsl,
    hsv,
    lab,
    lch,
    lms,
    oklab,
    oklch,
    rgb,
    rgba,
    xyb,
    xyz,
    ycbcr,

    pub fn fromSource(comptime S: type, comptime T: type) ColorSpace {
        return switch (S) {
            Gray(T) => .gray,
            Hsl(T) => .hsl,
            Hsv(T) => .hsv,
            Lab(T) => .lab,
            Lch(T) => .lch,
            Lms(T) => .lms,
            Oklab(T) => .oklab,
            Oklch(T) => .oklch,
            Rgba(T) => .rgba,
            Rgb(T) => .rgb,
            Xyb(T) => .xyb,
            Xyz(T) => .xyz,
            Ycbcr(T) => .ycbcr,
            else => @compileError("Unknown color type " ++ @typeName(S)),
        };
    }

    pub fn Color(self: ColorSpace, comptime T: type) type {
        return switch (self) {
            .gray => Gray(T),
            .hsl => Hsl(T),
            .hsv => Hsv(T),
            .lab => Lab(T),
            .lch => Lch(T),
            .lms => Lms(T),
            .oklab => Oklab(T),
            .oklch => Oklch(T),
            .rgba => Rgba(T),
            .rgb => Rgb(T),
            .xyb => Xyb(T),
            .xyz => Xyz(T),
            .ycbcr => Ycbcr(T),
        };
    }

    pub fn convert(comptime self: ColorSpace, comptime T: type, color: anytype) self.Color(T) {
        const InputType = @TypeOf(color);
        const C = ComponentType(InputType);
        const input_space = ColorSpace.fromSource(InputType, C);

        const input_supports_int = switch (input_space) {
            .rgb, .rgba, .ycbcr, .gray => true,
            else => false,
        };
        const target_is_int = switch (@typeInfo(T)) {
            .int => true,
            else => false,
        };

        if (target_is_int and !input_supports_int) {
            // Input is float-only (e.g. HSV), Target is Int.
            // Must convert to TargetSpace (preserving float) then cast to Int.
            // This assumes TargetSpace supports float (which is true for all current spaces).
            return color.to(self).as(T);
        } else {
            // Input supports T, or T is float (all spaces support float).
            // Convert type first (for better precision if promoting to float), then space.
            return color.as(T).to(self);
        }
    }

    fn ComponentType(comptime C: type) type {
        return switch (@typeInfo(C)) {
            .@"struct" => |info| info.fields[0].type,
            else => @compileError("Unknown color type " ++ @typeName(C)),
        };
    }
};

/// A color in the [sRGB](https://en.wikipedia.org/wiki/SRGB) colorspace, with all components
/// within the range 0-255 when `T` is `u8` and within 0-1 when `T` is float.
pub fn Rgb(comptime T: type) type {
    switch (@typeInfo(T)) {
        .float => {},
        .int => |info| if (info.bits != 8 or info.signedness != .unsigned) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space"),
        else => @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space"),
    }
    return struct {
        r: T,
        g: T,
        b: T,

        pub fn withAlpha(self: Rgb(T), alpha: T) Rgba(T) {
            return .{ .r = self.r, .g = self.g, .b = self.b, .a = alpha };
        }

        pub fn to(self: Rgb(T), comptime color_space: ColorSpace) color_space.Color(T) {
            return switch (color_space) {
                .gray => rgbToGray(T, self),
                .hsl => rgbToHsl(T, self),
                .hsv => rgbToHsv(T, self),
                .lab => rgbToLab(T, self),
                .lch => labToLch(T, rgbToLab(T, self)),
                .lms => xyzToLms(T, rgbToXyz(T, self)),
                .oklab => lmsToOklab(T, xyzToLms(T, rgbToXyz(T, self))),
                .oklch => oklabToOklch(T, lmsToOklab(T, xyzToLms(T, rgbToXyz(T, self)))),
                .rgba => .{ .r = self.r, .g = self.g, .b = self.b, .a = if (T == u8) 255 else 1.0 },
                .rgb => self,
                .xyb => lmsToXyb(T, xyzToLms(T, rgbToXyz(T, self))),
                .xyz => rgbToXyz(T, self),
                .ycbcr => rgbToYcbcr(T, self),
            };
        }

        pub fn as(self: Rgb(T), comptime U: type) Rgb(U) {
            switch (@typeInfo(U)) {
                .float => {},
                .int => |info| if (info.bits != 8 or info.signedness != .unsigned) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space"),
                else => @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space"),
            }
            return switch (T) {
                u8 => switch (U) {
                    u8 => self,
                    else => .{
                        .r = @as(U, @floatFromInt(self.r)) / 255,
                        .g = @as(U, @floatFromInt(self.g)) / 255,
                        .b = @as(U, @floatFromInt(self.b)) / 255,
                    },
                },
                else => switch (U) {
                    u8 => .{
                        .r = @intFromFloat(@round(255 * clamp(self.r, 0, 1))),
                        .g = @intFromFloat(@round(255 * clamp(self.g, 0, 1))),
                        .b = @intFromFloat(@round(255 * clamp(self.b, 0, 1))),
                    },
                    else => .{
                        .r = @floatCast(self.r),
                        .g = @floatCast(self.g),
                        .b = @floatCast(self.b),
                    },
                },
            };
        }
    };
}

/// A color in the [sRGB](https://en.wikipedia.org/wiki/SRGB) colorspace, with all components
/// within the range 0-255 when `T` is `u8` and within 0-1 when `T` is float.
pub fn Rgba(comptime T: type) type {
    switch (@typeInfo(T)) {
        .float => {},
        .int => |info| if (info.bits != 8 or info.signedness != .unsigned) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space"),
        else => @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space"),
    }
    return packed struct {
        r: T,
        g: T,
        b: T,
        a: T,

        pub fn to(self: Rgba(T), comptime color_space: ColorSpace) color_space.Color(T) {
            return switch (color_space) {
                .gray => rgbToGray(T, self.to(.rgb)),
                .hsl => rgbToHsl(T, self.to(.rgb)),
                .hsv => rgbToHsv(T, self.to(.rgb)),
                .lab => rgbToLab(T, self.to(.rgb)),
                .lch => labToLch(T, rgbToLab(T, self.to(.rgb))),
                .lms => xyzToLms(T, rgbToXyz(T, self.to(.rgb))),
                .oklab => lmsToOklab(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb)))),
                .oklch => oklabToOklch(T, lmsToOklab(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb))))),
                .rgba => self,
                .rgb => .{ .r = self.r, .g = self.g, .b = self.b },
                .xyb => lmsToXyb(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb)))),
                .xyz => rgbToXyz(T, self.to(.rgb)),
                .ycbcr => rgbToYcbcr(T, self.to(.rgb)),
            };
        }

        pub fn as(self: Rgba(T), comptime U: type) Rgba(U) {
            switch (@typeInfo(U)) {
                .float => {},
                .int => |info| if (info.bits != 8 or info.signedness != .unsigned) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space"),
                else => @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space"),
            }
            return switch (T) {
                u8 => switch (U) {
                    u8 => self,
                    else => .{
                        .r = @as(U, @floatFromInt(self.r)) / 255,
                        .g = @as(U, @floatFromInt(self.g)) / 255,
                        .b = @as(U, @floatFromInt(self.b)) / 255,
                        .a = @as(U, @floatFromInt(self.a)) / 255,
                    },
                },
                else => switch (U) {
                    u8 => .{
                        .r = @intFromFloat(@round(255 * clamp(self.r, 0, 1))),
                        .g = @intFromFloat(@round(255 * clamp(self.g, 0, 1))),
                        .b = @intFromFloat(@round(255 * clamp(self.b, 0, 1))),
                        .a = @intFromFloat(@round(255 * clamp(self.a, 0, 1))),
                    },
                    else => .{
                        .r = @floatCast(self.r),
                        .g = @floatCast(self.g),
                        .b = @floatCast(self.b),
                        .a = @floatCast(self.a),
                    },
                },
            };
        }
    };
}

/// A grayscale color using the same luminance definition as YCbCr (BT.601):
/// Y = 0.299 R + 0.587 G + 0.114 B. Supports `u8` and float backings.
pub fn Gray(comptime T: type) type {
    switch (@typeInfo(T)) {
        .float => {},
        .int => |info| if (info.bits != 8 or info.signedness != .unsigned) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space"),
        else => @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space"),
    }

    return struct {
        y: T,

        pub fn to(self: Gray(T), comptime color_space: ColorSpace) color_space.Color(T) {
            return switch (color_space) {
                .gray => self,
                else => grayToRgb(T, self).to(color_space),
            };
        }

        pub fn as(self: Gray(T), comptime U: type) Gray(U) {
            switch (@typeInfo(U)) {
                .float => {},
                .int => |info| if (info.bits != 8 or info.signedness != .unsigned) @compileError("Unsupported backing type " ++ @typeName(U) ++ " for color space"),
                else => @compileError("Unsupported backing type " ++ @typeName(U) ++ " for color space"),
            }

            return switch (T) {
                u8 => switch (U) {
                    u8 => .{ .y = self.y },
                    else => .{ .y = @as(U, @floatFromInt(self.y)) / 255 },
                },
                else => switch (U) {
                    u8 => .{ .y = @intFromFloat(@round(255 * clamp(self.y, 0, 1))) },
                    else => .{ .y = @floatCast(self.y) },
                },
            };
        }

        pub fn withAlpha(self: Gray(T), alpha: T) Rgba(T) {
            return .{ .r = self.y, .g = self.y, .b = self.y, .a = alpha };
        }
    };
}

/// A color in the [HSV](https://en.wikipedia.org/wiki/HSL_and_HSV) colorspace.
/// - h: Hue, in degrees (0-360, though often normalized to 0-359).
/// - s: Saturation, as a percentage (0-100).
/// - v: Value, as a percentage (0-100).
pub fn Hsv(comptime T: type) type {
    if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
    return struct {
        h: T,
        s: T,
        v: T,

        pub fn to(self: Hsv(T), comptime color_space: ColorSpace) color_space.Color(T) {
            return switch (color_space) {
                .gray => rgbToGray(T, self.to(.rgb)),
                .hsl => rgbToHsl(T, self.to(.rgb)),
                .hsv => self,
                .lab => rgbToLab(T, self.to(.rgb)),
                .lch => labToLch(T, rgbToLab(T, self.to(.rgb))),
                .lms => xyzToLms(T, rgbToXyz(T, self.to(.rgb))),
                .oklab => lmsToOklab(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb)))),
                .oklch => oklabToOklch(T, lmsToOklab(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb))))),
                .rgba => self.to(.rgb).to(.rgba),
                .rgb => hsvToRgb(T, self),
                .xyb => lmsToXyb(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb)))),
                .xyz => rgbToXyz(T, self.to(.rgb)),
                .ycbcr => @compileError("YCbCr only supported for u8"),
            };
        }

        pub fn as(self: Hsv(T), comptime U: type) Hsv(U) {
            if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
            return .{
                .h = @floatCast(self.h),
                .s = @floatCast(self.s),
                .v = @floatCast(self.v),
            };
        }
    };
}

/// A color in the [HSL](https://en.wikipedia.org/wiki/HSL_and_HSV) colorspace.
/// - h: Hue, in degrees (0-360, though often normalized to 0-359).
/// - s: Saturation, as a percentage (0-100).
/// - l: Lightness, as a percentage (0-100).
pub fn Hsl(comptime T: type) type {
    if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
    return struct {
        h: T,
        s: T,
        l: T,

        pub fn to(self: Hsl(T), comptime color_space: ColorSpace) color_space.Color(T) {
            return switch (color_space) {
                .gray => rgbToGray(T, self.to(.rgb)),
                .hsl => self,
                .hsv => rgbToHsv(T, self.to(.rgb)),
                .lab => rgbToLab(T, self.to(.rgb)),
                .lch => labToLch(T, rgbToLab(T, self.to(.rgb))),
                .lms => xyzToLms(T, rgbToXyz(T, self.to(.rgb))),
                .oklab => lmsToOklab(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb)))),
                .oklch => oklabToOklch(T, lmsToOklab(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb))))),
                .rgba => self.to(.rgb).to(.rgba),
                .rgb => hslToRgb(T, self),
                .xyb => lmsToXyb(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb)))),
                .xyz => rgbToXyz(T, self.to(.rgb)),
                .ycbcr => @compileError("YCbCr only supported for u8"),
            };
        }

        pub fn as(self: Hsl(T), comptime U: type) Hsv(U) {
            if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
            return .{
                .h = @floatCast(self.h),
                .s = @floatCast(self.s),
                .l = @floatCast(self.l),
            };
        }
    };
}

/// A color in the [CIE 1931 XYZ color space](https://en.wikipedia.org/wiki/CIE_1931_color_space).
/// This is a device-independent space that covers the full gamut of human-perceptible colors
/// visible to the CIE 2° standard observer.
/// - x, y, z: Tristimulus values, typically non-negative. Y represents luminance.
///   The typical range for these values can vary depending on the reference white point (e.g. D65).
///   Often, Y is normalized to 100 for white.
pub fn Xyz(comptime T: type) type {
    if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
    return struct {
        x: T,
        y: T,
        z: T,

        pub fn to(self: Xyz(T), comptime color_space: ColorSpace) color_space.Color(T) {
            return switch (color_space) {
                .gray => rgbToGray(T, self.to(.rgb)),
                .hsl => rgbToHsl(T, self.to(.rgb)),
                .hsv => rgbToHsv(T, self.to(.rgb)),
                .lab => rgbToLab(T, self.to(.rgb)),
                .lch => labToLch(T, rgbToLab(T, self.to(.rgb))),
                .lms => xyzToLms(T, self),
                .oklab => lmsToOklab(T, xyzToLms(T, self)),
                .oklch => oklabToOklch(T, lmsToOklab(T, xyzToLms(T, self))),
                .rgba => self.to(.rgb).to(.rgba),
                .rgb => xyzToRgb(T, self),
                .xyb => lmsToXyb(T, xyzToLms(T, self)),
                .xyz => self,
                .ycbcr => @compileError("YCbCr only supported for u8"),
            };
        }

        pub fn as(self: Xyz(T), comptime U: type) Xyz(U) {
            if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
            return .{
                .x = @floatCast(self.x),
                .y = @floatCast(self.y),
                .z = @floatCast(self.z),
            };
        }
    };
}

/// A color in the [CIELAB color space](https://en.wikipedia.org/wiki/CIELAB_color_space) (also known as L*a*b*).
/// It expresses color as three values:
/// - l: Lightness (0 for black to 100 for white).
/// - a: Green-red axis (-128 for green to +127 for red).
/// - b: Blue-yellow axis (-128 for blue to +127 for yellow).
pub fn Lab(comptime T: type) type {
    if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
    return struct {
        l: T,
        a: T,
        b: T,

        pub fn to(self: Lab(T), comptime color_space: ColorSpace) color_space.Color(T) {
            return switch (color_space) {
                .gray => rgbToGray(T, self.to(.rgb)),
                .hsl => rgbToHsl(T, self.to(.rgb)),
                .hsv => rgbToHsv(T, self.to(.rgb)),
                .lab => self,
                .lch => labToLch(T, self),
                .lms => xyzToLms(T, rgbToXyz(T, self.to(.rgb))),
                .oklab => lmsToOklab(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb)))),
                .oklch => oklabToOklch(T, lmsToOklab(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb))))),
                .rgba => self.to(.rgb).to(.rgba),
                .rgb => labToRgb(T, self),
                .xyb => lmsToXyb(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb)))),
                .xyz => rgbToXyz(T, self.to(.rgb)),
                .ycbcr => @compileError("YCbCr only supported for u8"),
            };
        }

        pub fn as(self: Lab(T), comptime U: type) Lab(U) {
            if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
            return .{
                .l = @floatCast(self.l),
                .a = @floatCast(self.a),
                .b = @floatCast(self.b),
            };
        }
    };
}

/// A color in the [CIELCh color space](https://en.wikipedia.org/wiki/CIELAB_color_space#Cylindrical_model).
/// LCh is the cylindrical representation of the CIELAB color space.
/// - l: Lightness (0 for black to 100 for white).
/// - c: Chroma (chromatic intensity) (0 for achromatic, no upper bound).
/// - h: Hue angle in degrees (0-360).
pub fn Lch(comptime T: type) type {
    if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
    return struct {
        l: T,
        m: T,
        s: T,
        pub fn to(self: Lch(T), comptime color_space: ColorSpace) color_space.Color(T) {
            return switch (color_space) {
                .gray => rgbToGray(T, self.to(.rgb)),
                .hsl => rgbToHsl(T, self.to(.rgb)),
                .hsv => rgbToHsv(T, self.to(.rgb)),
                .lab => lchToLab(T, self),
                .lch => self,
                .lms => xyzToLms(T, rgbToXyz(T, self.to(.rgb))),
                .oklab => lmsToOklab(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb)))),
                .oklch => oklabToOklch(T, lmsToOklab(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb))))),
                .rgba => self.to(.rgb).to(.rgba),
                .rgb => labToRgb(T, lchToLab(T, self)),
                .xyb => lmsToXyb(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb)))),
                .xyz => rgbToXyz(T, self.to(.rgb)),
                .ycbcr => @compileError("YCbCr only supported for u8"),
            };
        }

        pub fn as(self: Lch(T), comptime U: type) Lch(U) {
            if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
            return .{
                .l = @floatCast(self.l),
                .c = @floatCast(self.c),
                .h = @floatCast(self.h),
            };
        }
    };
}

/// A color in the [LMS color space](https://en.wikipedia.org/wiki/LMS_color_space).
/// Represents the response of the three types of cones (Long, Medium, Short wavelength) in the human eye.
/// Values are typically positive and represent the stimulus for each cone type.
pub fn Lms(comptime T: type) type {
    if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
    return struct {
        l: T,
        m: T,
        s: T,

        pub fn to(self: Lms(T), comptime color_space: ColorSpace) color_space.Color(T) {
            return switch (color_space) {
                .gray => rgbToGray(T, self.to(.rgb)),
                .hsl => rgbToHsl(T, self.to(.rgb)),
                .hsv => rgbToHsv(T, self.to(.rgb)),
                .lab => rgbToLab(T, self.to(.rgb)),
                .lch => labToLch(T, rgbToLab(T, self.to(.rgb))),
                .lms => self,
                .oklab => lmsToOklab(T, self),
                .oklch => oklabToOklch(T, lmsToOklab(T, self)),
                .rgba => self.to(.rgb).to(.rgba),
                .rgb => xyzToRgb(T, lmsToXyz(T, self)),
                .xyb => lmsToXyb(T, self),
                .xyz => lmsToXyz(T, self),
                .ycbcr => @compileError("YCbCr only supported for u8"),
            };
        }

        pub fn as(self: Lms(T), comptime U: type) Lms(U) {
            if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
            return .{
                .l = @floatCast(self.l),
                .m = @floatCast(self.m),
                .s = @floatCast(self.s),
            };
        }
    };
}

/// A color in the [Oklab color space](https://bottosson.github.io/posts/oklab/).
/// Oklab is designed to be a perceptually uniform color space.
/// - l: Perceived lightness (0 for black to approximately 1 for white).
/// - a: Green-red axis (negative values towards green, positive towards red, typically around -0.4 to 0.4).
/// - b: Blue-yellow axis (negative values towards blue, positive towards yellow, typically around -0.4 to 0.4).
pub fn Oklab(comptime T: type) type {
    if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
    return struct {
        l: T,
        a: T,
        b: T,

        pub fn to(self: Oklab(T), comptime color_space: ColorSpace) color_space.Color(T) {
            return switch (color_space) {
                .gray => rgbToGray(T, self.to(.rgb)),
                .hsl => rgbToHsl(T, self.to(.rgb)),
                .hsv => rgbToHsv(T, self.to(.rgb)),
                .lab => rgbToLab(T, self.to(.rgb)),
                .lch => labToLch(T, rgbToLab(T, self.to(.rgb))),
                .lms => oklabToLms(T, self),
                .oklab => self,
                .oklch => oklabToOklch(T, self),
                .rgba => self.to(.rgb).to(.rgba),
                .rgb => xyzToRgb(T, lmsToXyz(T, oklabToLms(T, self))),
                .xyb => lmsToXyb(T, oklabToLms(T, self)),
                .xyz => lmsToXyz(T, oklabToLms(T, self)),
                .ycbcr => @compileError("YCbCr only supported for u8"),
            };
        }

        pub fn as(self: Oklab(T), comptime U: type) Oklab(U) {
            if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
            return .{
                .l = @floatCast(self.l),
                .a = @floatCast(self.a),
                .b = @floatCast(self.b),
            };
        }
    };
}

/// A color in the [Oklch color space](https://en.wikipedia.org/wiki/Oklab_color_space).
/// Oklch is the cylindrical representation of the Oklab color space.
/// - l: Perceived lightness (0 for black to approximately 1 for white).
/// - c: Chroma (chromatic intensity) (0 for achromatic to approximately 0.5 for pure colors).
/// - h: Hue angle in degrees (0-360).
pub fn Oklch(comptime T: type) type {
    if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
    return struct {
        l: T,
        c: T,
        h: T,

        pub fn to(self: Oklch(T), comptime color_space: ColorSpace) color_space.Color(T) {
            return switch (color_space) {
                .gray => rgbToGray(T, self.to(.rgb)),
                .hsl => rgbToHsl(T, self.to(.rgb)),
                .hsv => rgbToHsv(T, self.to(.rgb)),
                .lab => rgbToLab(T, self.to(.rgb)),
                .lch => labToLch(T, rgbToLab(T, self.to(.rgb))),
                .lms => oklabToLms(T, oklchToOklab(T, self)),
                .oklab => oklchToOklab(T, self),
                .oklch => self,
                .rgba => self.to(.rgb).to(.rgba),
                .rgb => xyzToRgb(T, lmsToXyz(T, oklabToLms(T, oklchToOklab(T, self)))),
                .xyb => lmsToXyb(T, oklabToLms(T, oklchToOklab(T, self))),
                .xyz => lmsToXyz(T, oklabToLms(T, oklchToOklab(T, self))),
                .ycbcr => @compileError("YCbCr only supported for u8"),
            };
        }

        pub fn as(self: Oklch(T), comptime U: type) Oklch(U) {
            if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
            return .{
                .l = @floatCast(self.l),
                .c = @floatCast(self.c),
                .h = @floatCast(self.h),
            };
        }
    };
}

/// A color in the [XYB color space](https://jpeg.org/jpegxl/documentation/xl-color-management.html#xyb)
/// used in JPEG XL. It's derived from LMS and designed for efficient image compression.
/// - x: X component (L-M, red-green opponent channel).
/// - y: Y component (L+M, luminance-like channel).
/// - b: B component (S, blue-yellow like channel, but often scaled S cone response).
/// Ranges can vary based on transformations, but often centered around 0 for x and b, and positive for y.
pub fn Xyb(comptime T: type) type {
    if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
    return struct {
        x: T,
        y: T,
        z: T,

        pub fn to(self: Xyb(T), comptime color_space: ColorSpace) color_space.Color(T) {
            return switch (color_space) {
                .gray => rgbToGray(T, self.to(.rgb)),
                .hsl => rgbToHsl(T, self.to(.rgb)),
                .hsv => rgbToHsv(T, self.to(.rgb)),
                .lab => rgbToLab(T, self.to(.rgb)),
                .lch => labToLch(T, rgbToLab(T, self.to(.rgb))),
                .lms => xybToLms(T, self),
                .oklab => lmsToOklab(T, xybToLms(T, self)),
                .oklch => oklabToOklch(T, lmsToOklab(T, xybToLms(T, self))),
                .rgba => self.to(.rgb).to(.rgba),
                .rgb => xyzToRgb(T, lmsToXyz(T, xybToLms(T, self))),
                .xyb => self,
                .xyz => lmsToXyz(T, xybToLms(T, self)),
                .ycbcr => @compileError("YCbCr only supported for u8"),
            };
        }

        pub fn as(self: Xyb(T), comptime U: type) Xyb(U) {
            if (@typeInfo(T) != .float) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space");
            return .{
                .x = @floatCast(self.x),
                .y = @floatCast(self.y),
                .b = @floatCast(self.b),
            };
        }
    };
}

/// Ycbcr (Y'CbCr) colorspace used in JPEG and video encoding.
/// Y is luma (brightness), Cb is blue-difference chroma, Cr is red-difference chroma.
/// Uses ITU-R BT.601 coefficients for conversion to/from RGB.
pub fn Ycbcr(comptime T: type) type {
    switch (@typeInfo(T)) {
        .float => {},
        .int => |info| if (info.bits != 8 or info.signedness != .unsigned) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space"),
        else => @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space"),
    }
    return struct {
        y: T,
        cb: T,
        cr: T,

        pub fn to(self: Ycbcr(T), comptime color_space: ColorSpace) color_space.Color(T) {
            return switch (color_space) {
                .gray => rgbToGray(T, self.to(.rgb)),
                .hsl => rgbToHsl(T, self.to(.rgb)),
                .hsv => rgbToHsv(T, self.to(.rgb)),
                .lab => rgbToLab(T, self.to(.rgb)),
                .lch => labToLch(T, rgbToLab(T, self.to(.rgb))),
                .lms => xyzToLms(T, rgbToXyz(T, self.to(.rgb))),
                .oklab => lmsToOklab(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb)))),
                .oklch => oklabToOklch(T, lmsToOklab(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb))))),
                .rgb => ycbcrToRgb(T, self),
                .rgba => self.to(.rgb).to(.rgba),
                .xyb => lmsToXyb(T, xyzToLms(T, rgbToXyz(T, self.to(.rgb)))),
                .xyz => rgbToXyz(T, self.to(.rgb)),
                .ycbcr => self,
            };
        }

        pub fn as(self: Ycbcr(T), comptime U: type) Ycbcr(U) {
            switch (@typeInfo(U)) {
                .float => {},
                .int => |info| if (info.bits != 8 or info.signedness != .unsigned) @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space"),
                else => @compileError("Unsupported backing type " ++ @typeName(T) ++ " for color space"),
            }
            return switch (T) {
                u8 => switch (U) {
                    u8 => self,
                    else => .{
                        .y = @as(U, @floatFromInt(self.y)) / 255,
                        .cb = (@as(U, @floatFromInt(self.cb)) - 128) / 255,
                        .cr = (@as(U, @floatFromInt(self.cr)) - 128) / 255,
                    },
                },
                else => switch (U) {
                    u8 => .{
                        .y = @intFromFloat(@round(255 * clamp(self.y, 0, 1))),
                        .cb = @intFromFloat(@round(255 * clamp(self.cb + 0.5, 0, 1))),
                        .cr = @intFromFloat(@round(255 * clamp(self.cr + 0.5, 0, 1))),
                    },
                    else => .{
                        .y = @floatCast(self.y),
                        .cb = @floatCast(self.cb),
                        .cr = @floatCast(self.cr),
                    },
                },
            };
        }
    };
}

/// Converts RGB to Ycbcr using ITU-R BT.601 coefficients.
/// All components in [0, 255] range for u8, with Cb/Cr having 128 as neutral.
/// Uses 16-bit fixed-point arithmetic for precision when T is u8.
fn rgbToYcbcr(comptime T: type, rgb: Rgb(T)) Ycbcr(T) {
    if (T == u8) {
        const r: i32 = rgb.r;
        const g: i32 = rgb.g;
        const b: i32 = rgb.b;

        // BT.601 coefficients scaled by 65536 (2^16) for fixed-point
        // Y = 0.299*R + 0.587*G + 0.114*B
        const y: u8 = @intCast(clamp((19595 * r + 38470 * g + 7471 * b + 32768) >> 16, 0, 255));

        // Cb = -0.169*R - 0.331*G + 0.5*B + 128
        const cb: u8 = @intCast(clamp(((-11059 * r - 21710 * g + 32768 * b + 32768) >> 16) + 128, 0, 255));

        // Cr = 0.5*R - 0.419*G - 0.081*B + 128
        const cr: u8 = @intCast(clamp(((32768 * r - 27439 * g - 5329 * b + 32768) >> 16) + 128, 0, 255));

        return .{
            .y = y,
            .cb = cb,
            .cr = cr,
        };
    } else {
        const y = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
        const cb = (rgb.b - y) / 1.772;
        const cr = (rgb.r - y) / 1.402;
        return .{
            .y = clamp(y, 0, 1),
            .cb = clamp(cb, -0.5, 0.5),
            .cr = clamp(cr, -0.5, 0.5),
        };
    }
}

/// Converts RGB to grayscale using BT.601 luminance coefficients, matching the Y component of YCbCr.
/// For `u8`, uses 16-bit fixed-point arithmetic for consistency with the YCbCr path.
pub fn rgbToGray(comptime T: type, rgb: Rgb(T)) Gray(T) {
    if (T == u8) {
        const r: i32 = rgb.r;
        const g: i32 = rgb.g;
        const b: i32 = rgb.b;
        return .{ .y = @intCast(clamp((19595 * r + 38470 * g + 7471 * b + 32768) >> 16, 0, 255)) };
    } else {
        const y = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
        return .{ .y = clamp(y, 0, 1) };
    }
}

/// Converts grayscale to RGB by replicating the Y component across channels.
pub fn grayToRgb(comptime T: type, gray: Gray(T)) Rgb(T) {
    return .{ .r = gray.y, .g = gray.y, .b = gray.y };
}

/// Converts Ycbcr to RGB using ITU-R BT.601 coefficients.
/// Expects all components in [0, 255] range for u8, with Cb/Cr having 128 as neutral.
/// Uses 16-bit fixed-point arithmetic for precision when T is u8.
pub fn ycbcrToRgb(comptime T: type, ycbcr: Ycbcr(T)) Rgb(T) {
    if (T == u8) {
        const y = @as(i32, ycbcr.y);
        const cb = @as(i32, ycbcr.cb) - 128;
        const cr = @as(i32, ycbcr.cr) - 128;

        // BT.601 inverse coefficients scaled by 65536 (2^16) for fixed-point
        // R = Y + 1.402 * Cr
        const r: u8 = @intCast(clamp(y + ((91881 * cr + 32768) >> 16), 0, 255));

        // G = Y - 0.344136 * Cb - 0.714136 * Cr
        const g: u8 = @intCast(clamp(y - ((22554 * cb + 46802 * cr + 32768) >> 16), 0, 255));

        // B = Y + 1.772 * Cb
        const b: u8 = @intCast(clamp(y + ((116130 * cb + 32768) >> 16), 0, 255));

        return .{
            .r = r,
            .g = g,
            .b = b,
        };
    } else {
        // Float implementation
        const y = ycbcr.y;
        const cb = ycbcr.cb;
        const cr = ycbcr.cr;

        // R = Y + 1.402 * Cr
        const r = y + 1.402 * cr;
        // G = Y - 0.344136 * Cb - 0.714136 * Cr
        const g = y - 0.344136 * cb - 0.714136 * cr;
        // B = Y + 1.772 * Cb
        const b = y + 1.772 * cb;

        return .{
            .r = clamp(r, 0, 1),
            .g = clamp(g, 0, 1),
            .b = clamp(b, 0, 1),
        };
    }
}

fn rgbToHsv(comptime T: type, rgb: Rgb(T)) Hsv(T) {
    const min = @min(rgb.r, @min(rgb.g, rgb.b));
    const max = @max(rgb.r, @max(rgb.g, rgb.b));
    const delta = max - min;

    return .{
        .h = if (delta == 0) 0 else blk: {
            if (max == rgb.r) {
                break :blk @mod((rgb.g - rgb.b) / delta * 60, 360);
            } else if (max == rgb.g) {
                break :blk @mod(120 + (rgb.b - rgb.r) / delta * 60, 360);
            } else {
                break :blk @mod(240 + (rgb.r - rgb.g) / delta * 60, 360);
            }
        },
        .s = if (max == 0) 0 else (delta / max) * 100,
        .v = max * 100,
    };
}

pub fn hslToRgb(comptime T: type, hsl: Hsl(T)) Rgb(T) {
    const h = @max(0, @min(360, hsl.h));
    const s = @max(0, @min(1, hsl.s / 100));
    const l = @max(0, @min(1, hsl.l / 100));

    const hue_sector = h / 60.0;
    const sector: usize = @intFromFloat(hue_sector);
    const fractional = hue_sector - @as(f64, @floatFromInt(sector));

    const hue_factors = [_][3]f64{
        .{ 1, fractional, 0 },
        .{ 1 - fractional, 1, 0 },
        .{ 0, 1, fractional },
        .{ 0, 1 - fractional, 1 },
        .{ fractional, 0, 1 },
        .{ 1, 0, 1 - fractional },
    };

    const index = @mod(sector, 6);
    const r = lerp(1, 2 * hue_factors[index][0], s);
    const g = lerp(1, 2 * hue_factors[index][1], s);
    const b = lerp(1, 2 * hue_factors[index][2], s);

    return if (l < 0.5)
        .{
            .r = r * l,
            .g = g * l,
            .b = b * l,
        }
    else
        .{
            .r = lerp(r, 2, l) - 1,
            .g = lerp(g, 2, l) - 1,
            .b = lerp(b, 2, l) - 1,
        };
}

fn rgbToHsl(comptime T: type, rgb: Rgb(T)) Hsl(T) {
    const min = @min(rgb.r, @min(rgb.g, rgb.b));
    const max = @max(rgb.r, @max(rgb.g, rgb.b));
    const delta = max - min;

    const hue = if (delta == 0) 0 else blk: {
        if (max == rgb.r) {
            break :blk (rgb.g - rgb.b) / delta;
        } else if (max == rgb.g) {
            break :blk 2 + (rgb.b - rgb.r) / delta;
        } else {
            break :blk 4 + (rgb.r - rgb.g) / delta;
        }
    };

    const l = (max + min) / 2.0;
    const s = if (delta == 0) 0 else if (l < 0.5) delta / (2 * l) else delta / (2 - 2 * l);

    return .{
        .h = @mod(hue * 60.0, 360.0),
        .s = @max(0, @min(1, s)) * 100.0,
        .l = @max(0, @min(1, l)) * 100.0,
    };
}

fn hsvToRgb(comptime T: type, hsv: Hsv(T)) Rgb(T) {
    const hue = @max(0, @min(1, hsv.h / 360));
    const sat = @max(0, @min(1, hsv.s / 100));
    const val = @max(0, @min(1, hsv.v / 100));

    if (sat == 0.0) {
        return .{ .r = val, .g = val, .b = val };
    }

    const sector = hue * 6;
    const index: i32 = @intFromFloat(sector);
    const fractional = sector - @as(T, @floatFromInt(index));
    const p = val * (1 - sat);
    const q = val * (1 - (sat * fractional));
    const t = val * (1 - sat * (1 - fractional));
    const colors = [_][3]T{
        .{ val, t, p },
        .{ q, val, p },
        .{ p, val, t },
        .{ p, q, val },
        .{ t, p, val },
        .{ val, p, q },
    };
    const idx: usize = @intCast(@mod(index, 6));

    return .{
        .r = colors[idx][0],
        .g = colors[idx][1],
        .b = colors[idx][2],
    };
}

fn linearToGamma(comptime T: type, c: T) T {
    return if (c > 0.0031308) 1.055 * pow(T, c, (1.0 / 2.4)) - 0.055 else c * 12.92;
}

fn gammaToLinear(comptime T: type, c: T) T {
    return if (c > 0.04045) pow(T, (c + 0.055) / 1.055, 2.4) else c / 12.92;
}

pub fn rgbToXyz(comptime T: type, rgb: Rgb(T)) Xyz(T) {
    const r = gammaToLinear(rgb.r);
    const g = gammaToLinear(rgb.g);
    const b = gammaToLinear(rgb.b);

    return .{
        .x = (r * 0.4124 + g * 0.3576 + b * 0.1805) * 100,
        .y = (r * 0.2126 + g * 0.7152 + b * 0.0722) * 100,
        .z = (r * 0.0193 + g * 0.1192 + b * 0.9505) * 100,
    };
}

pub fn xyzToRgb(comptime T: type, xyz: Xyz(T)) Rgb(T) {
    const r = (xyz.x * 3.2406 + xyz.y * -1.5372 + xyz.z * -0.4986) / 100;
    const g = (xyz.x * -0.9689 + xyz.y * 1.8758 + xyz.z * 0.0415) / 100;
    const b = (xyz.x * 0.0557 + xyz.y * -0.2040 + xyz.z * 1.0570) / 100;

    return .{
        .r = clamp(linearToGamma(r), 0, 1),
        .g = clamp(linearToGamma(g), 0, 1),
        .b = clamp(linearToGamma(b), 0, 1),
    };
}

pub fn rgbToLab(comptime T: type, rgb: Rgb(T)) Lab(T) {

    // Convert to XYZ first
    const r = gammaToLinear(rgb.r);
    const g = gammaToLinear(rgb.g);
    const b = gammaToLinear(rgb.b);

    const x = (r * 0.4124 + g * 0.3576 + b * 0.1805) * 100;
    const y = (r * 0.2126 + g * 0.7152 + b * 0.0722) * 100;
    const z = (r * 0.0193 + g * 0.1192 + b * 0.9505) * 100;

    // Convert XYZ to Lab
    var xn = x / 95.047;
    var yn = y / 100.000;
    var zn = z / 108.883;

    if (xn > 0.008856) {
        xn = pow(T, xn, 1.0 / 3.0);
    } else {
        xn = (7.787 * xn) + (16.0 / 116.0);
    }

    if (yn > 0.008856) {
        yn = pow(T, yn, 1.0 / 3.0);
    } else {
        yn = (7.787 * yn) + (16.0 / 116.0);
    }

    if (zn > 0.008856) {
        zn = pow(T, zn, 1.0 / 3.0);
    } else {
        zn = (7.787 * zn) + (16.0 / 116.0);
    }

    return .{
        .l = clamp(116.0 * yn - 16.0, 0, 100),
        .a = clamp(500.0 * (xn - yn), -128, 127),
        .b = clamp(200.0 * (yn - zn), -128, 127),
    };
}

pub fn labToRgb(comptime T: type, lab: Lab(T)) Rgb(T) {
    // Convert Lab to XYZ first
    var y: f64 = (@max(0, @min(100, lab.l)) + 16.0) / 116.0;
    var x: f64 = (@max(-128, @min(127, lab.a)) / 500.0) + y;
    var z: f64 = y - (@max(-128, @min(127, lab.b)) / 200.0);

    if (pow(f64, y, 3.0) > 0.008856) {
        y = pow(f64, y, 3.0);
    } else {
        y = (y - 16.0 / 116.0) / 7.787;
    }

    if (pow(f64, x, 3.0) > 0.008856) {
        x = pow(f64, x, 3.0);
    } else {
        x = (x - 16.0 / 116.0) / 7.787;
    }

    if (pow(f64, z, 3.0) > 0.008856) {
        z = pow(f64, z, 3.0);
    } else {
        z = (z - 16.0 / 116.0) / 7.787;
    }

    // Observer. = 2°, illuminant = D65.
    x *= 95.047;
    y *= 100.000;
    z *= 108.883;

    // Convert XYZ to RGB
    const r = (x * 3.2406 + y * -1.5372 + z * -0.4986) / 100;
    const g = (x * -0.9689 + y * 1.8758 + z * 0.0415) / 100;
    const b = (x * 0.0557 + y * -0.2040 + z * 1.0570) / 100;

    return .{
        .r = clamp(linearToGamma(r), 0, 1),
        .g = clamp(linearToGamma(g), 0, 1),
        .b = clamp(linearToGamma(b), 0, 1),
    };
}

fn labToLch(comptime T: type, lab: Lab(T)) Lch(T) {
    const c = @sqrt(lab.a * lab.a + lab.b * lab.b);
    var h = std.math.atan2(lab.b, lab.a) * 180.0 / std.math.pi;
    // Ensure hue is in range [0, 360)
    if (h < 0) {
        h += 360.0;
    }
    return .{
        .l = lab.l,
        .c = c,
        .h = h,
    };
}

fn lchToLab(comptime T: type, lch: Lch(T)) Lab(T) {
    const h_rad = lch.h * std.math.pi / 180.0;
    return .{
        .l = lch.l,
        .a = lch.c * @cos(h_rad),
        .b = lch.c * @sin(h_rad),
    };
}

fn xyzToLms(comptime T: type, xyz: Xyz(T)) Lms(T) {
    return .{
        .l = (0.8951 * xyz.x + 0.2664 * xyz.y - 0.1614 * xyz.z) / 100,
        .m = (-0.7502 * xyz.x + 1.7135 * xyz.y + 0.0367 * xyz.z) / 100,
        .s = (0.0389 * xyz.x - 0.0685 * xyz.y + 1.0296 * xyz.z) / 100,
    };
}

fn lmsToXyz(comptime T: type, lms: Lms(T)) Xyz(T) {
    return .{
        .x = 100 * (0.9869929 * lms.l - 0.1470543 * lms.m + 0.1599627 * lms.s),
        .y = 100 * (0.4323053 * lms.l + 0.5183603 * lms.m + 0.0492912 * lms.s),
        .z = 100 * (-0.0085287 * lms.l + 0.0400428 * lms.m + 0.9684867 * lms.s),
    };
}

fn lmsToOklab(comptime T: type, lms: Lms(T)) Oklab(T) {
    const lp = std.math.cbrt(lms.l);
    const mp = std.math.cbrt(lms.m);
    const sp = std.math.cbrt(lms.s);
    return .{
        .l = 0.2104542553 * lp + 0.7936177850 * mp - 0.0040720468 * sp,
        .a = 1.9779984951 * lp - 2.4285922050 * mp + 0.4505937099 * sp,
        .b = 0.0259040371 * lp + 0.7827717662 * mp - 0.8086757660 * sp,
    };
}

fn oklabToLms(comptime T: type, oklab: Oklab(T)) Lms(T) {
    return .{
        .l = std.math.pow(f64, 0.9999999985 * oklab.l + 0.3963377922 * oklab.a + 0.2158037581 * oklab.b, 3),
        .m = std.math.pow(f64, 1.000000009 * oklab.l - 0.1055613423 * oklab.a - 0.06385417477 * oklab.b, 3),
        .s = std.math.pow(f64, 1.000000055 * oklab.l - 0.08948418209 * oklab.a - 1.291485538 * oklab.b, 3),
    };
}

fn oklabToOklch(comptime T: type, oklab: Oklab(T)) Oklch(T) {
    const c = @sqrt(oklab.a * oklab.a + oklab.b * oklab.b);
    var h = std.math.atan2(oklab.b, oklab.a) * 180.0 / std.math.pi;

    // Ensure hue is in range [0, 360)
    if (h < 0) {
        h += 360.0;
    }

    return .{
        .l = oklab.l,
        .c = c,
        .h = h,
    };
}

fn oklchToOklab(comptime T: type, oklch: Oklch(T)) Oklab(T) {
    const h_rad = oklch.h * std.math.pi / 180.0;

    return .{
        .l = oklch.l,
        .a = oklch.c * @cos(h_rad),
        .b = oklch.c * @sin(h_rad),
    };
}

fn lmsToXyb(comptime T: type, lms: Lms(T)) Xyb(T) {
    return .{ .x = lms.l - lms.m, .y = lms.l + lms.m, .b = lms.s };
}

fn xybToLms(comptime T: type, xyb: Xyb(T)) Lms(T) {
    return .{
        .l = 0.5 * (xyb.x + xyb.y),
        .m = 0.5 * (xyb.y - xyb.x),
        .s = xyb.b,
    };
}

// ============================================================================
// TESTS
// ============================================================================

// Import tests from sub-modules
// test {
//     _ = @import("color/blending.zig");
// }

// Helper function for testing round-trip conversions
fn testColorConversion(from: Rgb(u8), to: anytype) !void {
    const converted = convertColor(@TypeOf(to), from);
    try expectEqualDeep(converted, to);
    const recovered = convertColor(Rgb, converted);
    try expectEqualDeep(recovered, from);
}

// test "convert grayscale" {
//     try expectEqual(convertColor(u8, Rgb{ .r = 128, .g = 128, .b = 128 }), 128);
//     try expectEqual(convertColor(u8, Hsl{ .h = 0, .s = 100, .l = 50 }), 54);
//     try expectEqual(convertColor(u8, Hsv{ .h = 0, .s = 100, .v = 50 }), 27);
//     try expectEqual(convertColor(u8, Lab{ .l = 50, .a = 0, .b = 0 }), 119);
// }

// test "Rgb fromHex and toHex" {
//     // Test fromHex with various colors
//     try expectEqualDeep(Rgb.fromHex(0x4e008e), Rgb{ .r = 78, .g = 0, .b = 142 });
//     try expectEqualDeep(Rgb.fromHex(0x000000), Rgb{ .r = 0, .g = 0, .b = 0 });
//     try expectEqualDeep(Rgb.fromHex(0xffffff), Rgb{ .r = 255, .g = 255, .b = 255 });
//     try expectEqualDeep(Rgb.fromHex(0xff0000), Rgb{ .r = 255, .g = 0, .b = 0 });
//     try expectEqualDeep(Rgb.fromHex(0x00ff00), Rgb{ .r = 0, .g = 255, .b = 0 });
//     try expectEqualDeep(Rgb.fromHex(0x0000ff), Rgb{ .r = 0, .g = 0, .b = 255 });
//     try expectEqualDeep(Rgb.fromHex(0x808080), Rgb{ .r = 128, .g = 128, .b = 128 });

//     // Test toHex converts back correctly
//     const purple = Rgb{ .r = 78, .g = 0, .b = 142 };
//     try expectEqual(purple.toHex(), 0x4e008e);

//     const black = Rgb{ .r = 0, .g = 0, .b = 0 };
//     try expectEqual(black.toHex(), 0x000000);

//     const white = Rgb{ .r = 255, .g = 255, .b = 255 };
//     try expectEqual(white.toHex(), 0xffffff);

//     const red = Rgb{ .r = 255, .g = 0, .b = 0 };
//     try expectEqual(red.toHex(), 0xff0000);

//     // Test round-trip conversion
//     const test_colors = [_]u24{ 0x123456, 0xabcdef, 0x987654, 0xfedcba, 0x111111, 0xeeeeee };
//     for (test_colors) |hex_color| {
//         const rgb = Rgb.fromHex(hex_color);
//         const converted_back = rgb.toHex();
//         try expectEqual(converted_back, hex_color);
//     }
// }

// test "Rgba fromHex and toHex" {
//     // Test fromHex with various colors (RGBA format)
//     try expectEqualDeep(Rgba.fromHex(0x4e008eff), Rgba{ .r = 78, .g = 0, .b = 142, .a = 255 });
//     try expectEqualDeep(Rgba.fromHex(0x000000ff), Rgba{ .r = 0, .g = 0, .b = 0, .a = 255 });
//     try expectEqualDeep(Rgba.fromHex(0xffffff00), Rgba{ .r = 255, .g = 255, .b = 255, .a = 0 });
//     try expectEqualDeep(Rgba.fromHex(0xff000080), Rgba{ .r = 255, .g = 0, .b = 0, .a = 128 });
//     try expectEqualDeep(Rgba.fromHex(0x00ff00c0), Rgba{ .r = 0, .g = 255, .b = 0, .a = 192 });
//     try expectEqualDeep(Rgba.fromHex(0x0000ff40), Rgba{ .r = 0, .g = 0, .b = 255, .a = 64 });

//     // Test toHex converts back correctly
//     const purple_alpha = Rgba{ .r = 78, .g = 0, .b = 142, .a = 255 };
//     try expectEqual(purple_alpha.toHex(), 0x4e008eff);

//     const transparent_white = Rgba{ .r = 255, .g = 255, .b = 255, .a = 0 };
//     try expectEqual(transparent_white.toHex(), 0xffffff00);

//     const semi_red = Rgba{ .r = 255, .g = 0, .b = 0, .a = 128 };
//     try expectEqual(semi_red.toHex(), 0xff000080);

//     // Test round-trip conversion
//     const test_colors = [_]u32{ 0x12345678, 0xabcdef90, 0x98765432, 0xfedcba01, 0x11111111, 0xeeeeeeee };
//     for (test_colors) |hex_color| {
//         const rgba = Rgba.fromHex(hex_color);
//         const converted_back = rgba.toHex();
//         try expectEqual(converted_back, hex_color);
//     }

//     // Test edge cases
//     try expectEqualDeep(Rgba.fromHex(0x00000000), Rgba.transparent);
//     try expectEqualDeep(Rgba.fromHex(0x000000ff), Rgba.black);
//     try expectEqualDeep(Rgba.fromHex(0xffffffff), Rgba.white);
// }

// test "primary colors" {
//     // red: 0xff0000
//     try testColorConversion(.{ .r = 255, .g = 0, .b = 0 }, Hsl{ .h = 0, .s = 100, .l = 50 });
//     try testColorConversion(.{ .r = 255, .g = 0, .b = 0 }, Hsv{ .h = 0, .s = 100, .v = 100 });
//     try testColorConversion(.{ .r = 255, .g = 0, .b = 0 }, Lab{ .l = 53.23288178584245, .a = 80.10930952982204, .b = 67.22006831026425 });
//     // green: 0x00ff00
//     try testColorConversion(.{ .r = 0, .g = 255, .b = 0 }, Hsl{ .h = 120, .s = 100, .l = 50 });
//     try testColorConversion(.{ .r = 0, .g = 255, .b = 0 }, Hsv{ .h = 120, .s = 100, .v = 100 });
//     try testColorConversion(.{ .r = 0, .g = 255, .b = 0 }, Lab{ .l = 87.73703347354422, .a = -86.1846364976253, .b = 83.18116474777855 });
//     // blue: 0x0000ff
//     try testColorConversion(.{ .r = 0, .g = 0, .b = 255 }, Hsl{ .h = 240, .s = 100, .l = 50 });
//     try testColorConversion(.{ .r = 0, .g = 0, .b = 255 }, Hsv{ .h = 240, .s = 100, .v = 100 });
//     try testColorConversion(.{ .r = 0, .g = 0, .b = 255 }, Lab{ .l = 32.302586667249486, .a = 79.19666178930935, .b = -107.86368104495168 });
// }

// test "secondary colors" {
//     // cyan: 0x00ffff
//     try testColorConversion(.{ .r = 0, .g = 255, .b = 255 }, Hsl{ .h = 180, .s = 100, .l = 50 });
//     try testColorConversion(.{ .r = 0, .g = 255, .b = 255 }, Hsv{ .h = 180, .s = 100, .v = 100 });
//     try testColorConversion(.{ .r = 0, .g = 255, .b = 255 }, Lab{ .l = 91.11652110946342, .a = -48.079618466228716, .b = -14.138127754846131 });
//     // magenta: 0xff00ff
//     try testColorConversion(.{ .r = 255, .g = 0, .b = 255 }, Hsl{ .h = 300, .s = 100, .l = 50 });
//     try testColorConversion(.{ .r = 255, .g = 0, .b = 255 }, Hsv{ .h = 300, .s = 100, .v = 100 });
//     try testColorConversion(.{ .r = 255, .g = 0, .b = 255 }, Lab{ .l = 60.319933664076004, .a = 98.25421868616108, .b = -60.84298422386232 });
//     // yellow: 0xffff00
//     try testColorConversion(.{ .r = 255, .g = 255, .b = 0 }, Hsl{ .h = 60, .s = 100, .l = 50 });
//     try testColorConversion(.{ .r = 255, .g = 255, .b = 0 }, Hsv{ .h = 60, .s = 100, .v = 100 });
//     try testColorConversion(.{ .r = 255, .g = 255, .b = 0 }, Lab{ .l = 97.13824698129729, .a = -21.555908334832285, .b = 94.48248544644461 });
// }

// test "complementary colors" {
//     // orange: 0xff8800
//     try testColorConversion(.{ .r = 255, .g = 136, .b = 0 }, Hsl{ .h = 32, .s = 100, .l = 50 });
//     try testColorConversion(.{ .r = 255, .g = 136, .b = 0 }, Hsv{ .h = 32, .s = 100, .v = 100 });
//     try testColorConversion(.{ .r = 255, .g = 136, .b = 0 }, Lab{ .l = 68.65577208167872, .a = 38.85052375564019, .b = 74.99022544139406 });
//     // purple: 0x800080
//     try testColorConversion(.{ .r = 128, .g = 0, .b = 128 }, Hsl{ .h = 300, .s = 100, .l = 25.098039215686274 });
//     try testColorConversion(.{ .r = 128, .g = 0, .b = 128 }, Hsv{ .h = 300, .s = 100, .v = 50.19607843137255 });
//     try testColorConversion(.{ .r = 128, .g = 0, .b = 128 }, Lab{ .l = 29.782100092098077, .a = 58.93983731904206, .b = -36.49792996282386 });
// }

// test "neutral colors" {
//     // white: 0xffffff
//     try testColorConversion(.{ .r = 255, .g = 255, .b = 255 }, Hsl{ .h = 0, .s = 0, .l = 100 });
//     try testColorConversion(.{ .r = 255, .g = 255, .b = 255 }, Hsv{ .h = 0, .s = 0, .v = 100 });
//     try testColorConversion(.{ .r = 255, .g = 255, .b = 255 }, Lab{ .l = 100, .a = 0.00526049995830391, .b = -0.010408184525267927 });
//     // gray: 0x808080
//     try testColorConversion(.{ .r = 128, .g = 128, .b = 128 }, Hsl{ .h = 0, .s = 0, .l = 50.19607843137255 });
//     try testColorConversion(.{ .r = 128, .g = 128, .b = 128 }, Hsv{ .h = 0, .s = 0, .v = 50.19607843137255 });
//     try testColorConversion(.{ .r = 128, .g = 128, .b = 128 }, Lab{ .l = 53.58501345216902, .a = 0.003155620347972121, .b = -0.006243566036268078 });
//     // black: 0x000000
//     try testColorConversion(.{ .r = 0, .g = 0, .b = 0 }, Hsl{ .h = 0, .s = 0, .l = 0 });
//     try testColorConversion(.{ .r = 0, .g = 0, .b = 0 }, Hsv{ .h = 0, .s = 0, .v = 0 });
//     try testColorConversion(.{ .r = 0, .g = 0, .b = 0 }, Lab{ .l = 0, .a = 0, .b = 0 });
// }

// test "pastel colors" {
//     // pale_pink: 0xffd3ba
//     try testColorConversion(.{ .r = 255, .g = 211, .b = 186 }, Hsl{ .h = 21.739130434782602, .s = 100, .l = 86.47058823529412 });
//     try testColorConversion(.{ .r = 255, .g = 211, .b = 186 }, Hsv{ .h = 21.739130434782602, .s = 27.058823529411768, .v = 100 });
//     try testColorConversion(.{ .r = 255, .g = 211, .b = 186 }, Lab{ .l = 87.67593388241974, .a = 11.843797404960165, .b = 18.16236917854479 });
//     // mint_green: 0x96fa96
//     try testColorConversion(.{ .r = 150, .g = 250, .b = 150 }, Hsl{ .h = 120, .s = 90.90909090909089, .l = 78.43137254901961 });
//     try testColorConversion(.{ .r = 150, .g = 250, .b = 150 }, Hsv{ .h = 120, .s = 40, .v = 98.0392156862745 });
//     try testColorConversion(.{ .r = 150, .g = 250, .b = 150 }, Lab{ .l = 90.34795996024553, .a = -48.75545372512652, .b = 38.96689290268498 });
//     // sky_blue: #8ad1ed
//     try testColorConversion(.{ .r = 138, .g = 209, .b = 237 }, Hsl{ .h = 196.96969696969697, .s = 73.33333333333336, .l = 73.52941176470588 });
//     try testColorConversion(.{ .r = 138, .g = 209, .b = 237 }, Hsv{ .h = 196.96969696969697, .s = 41.77215189873419, .v = 92.94117647058823 });
//     try testColorConversion(.{ .r = 138, .g = 209, .b = 237 }, Lab{ .l = 80.24627015828005, .a = -15.11865203941365, .b = -20.767024460106565 });
// }

// test "vivid colors" {
//     // hot_pink: #ff66b3
//     try testColorConversion(.{ .r = 255, .g = 102, .b = 179 }, Hsl{ .h = 329.80392156862746, .s = 99.99999999999997, .l = 70 });
//     try testColorConversion(.{ .r = 255, .g = 102, .b = 179 }, Hsv{ .h = 329.80392156862746, .s = 60, .v = 100 });
//     try testColorConversion(.{ .r = 255, .g = 102, .b = 179 }, Lab{ .l = 64.9763931162809, .a = 65.40669278373645, .b = -10.847761988977656 });
//     // lime_green:#31cc31
//     try testColorConversion(.{ .r = 49, .g = 204, .b = 49 }, Hsl{ .h = 120, .s = 61.26482213438735, .l = 49.6078431372549 });
//     try testColorConversion(.{ .r = 49, .g = 204, .b = 49 }, Hsv{ .h = 120, .s = 75.98039215686275, .v = 80 });
//     try testColorConversion(.{ .r = 49, .g = 204, .b = 49 }, Lab{ .l = 72.26888334336961, .a = -67.03378336285304, .b = 61.425460443480894 });
//     // electric_blue: #80dfff
//     try testColorConversion(.{ .r = 128, .g = 223, .b = 255 }, Hsl{ .h = 195.11811023622047, .s = 100, .l = 75.09803921568627 });
//     try testColorConversion(.{ .r = 128, .g = 223, .b = 255 }, Hsv{ .h = 195.11811023622047, .s = 49.80392156862745, .v = 100 });
//     try testColorConversion(.{ .r = 128, .g = 223, .b = 255 }, Lab{ .l = 84.26919487615707, .a = -19.773688316136685, .b = -24.252061008370738 });
// }

// test "color formatting" {
//     const red = Rgb{ .r = 255, .g = 0, .b = 0 };

//     // Test plain format with {any}
//     var plain_buffer: [100]u8 = undefined;
//     const plain_result = try std.fmt.bufPrint(&plain_buffer, "{any}", .{red});
//     try std.testing.expect(std.mem.indexOf(u8, plain_result, ".r = 255, .g = 0, .b = 0") != null);
//     try std.testing.expect(std.mem.indexOf(u8, plain_result, "\x1b[") == null); // No SGR codes

//     // Test colored format with {f}
//     var color_buffer: [200]u8 = undefined;
//     const color_result = try std.fmt.bufPrint(&color_buffer, "{f}", .{red});
//     try std.testing.expect(std.mem.indexOf(u8, color_result, "Rgb{ .r = 255, .g = 0, .b = 0 }") != null);
//     try std.testing.expect(std.mem.indexOf(u8, color_result, "\x1b[") != null); // Has SGR codes
// }

// test "100 random colors" {
//     const seed: u64 = std.crypto.random.int(u64);
//     var prng: std.Random.DefaultPrng = .init(seed);
//     var random = prng.random();
//     for (0..100) |_| {
//         const rgb: Rgb = .{ .r = random.int(u8), .g = random.int(u8), .b = random.int(u8) };
//         const rgb_from_hsl = rgb.toHsl().toRgb();
//         try expectEqualDeep(rgb, rgb_from_hsl);
//         const rgb_from_hsv = rgb.toHsv().toRgb();
//         try expectEqualDeep(rgb, rgb_from_hsv);
//         const rgb_from_xyz = rgb.toXyz().toRgb();
//         try expectEqualDeep(rgb, rgb_from_xyz);
//         const rgb_from_lab = rgb.toLab().toRgb();
//         try expectEqualDeep(rgb, rgb_from_lab);
//     }
// }

// test "color type validation" {
//     try expectEqual(isColor(u8), true);
//     try expectEqual(isColor(Rgb), true);
//     try expectEqual(isColor(Rgba), true);
//     try expectEqual(isColor(Hsl), true);
//     try expectEqual(isColor(Hsv), true);
//     try expectEqual(isColor(Lab), true);
//     try expectEqual(isColor(Xyz), true);
//     try expectEqual(isColor(Lms), true);
//     try expectEqual(isColor(Oklab), true);
//     try expectEqual(isColor(Oklch), true);
//     try expectEqual(isColor(Xyb), true);
//     try expectEqual(isColor(Ycbcr), true);
//     try expectEqual(isColor(f32), false);
//     try expectEqual(isColor(i32), false);
// }

// test "generic convert function" {
//     const red = Rgb{ .r = 255, .g = 0, .b = 0 };

//     // Test conversion to all color types
//     const red_rgba = convertColor(Rgba, red);
//     try expectEqualDeep(red_rgba, Rgba{ .r = 255, .g = 0, .b = 0, .a = 255 });

//     const red_hsl = convertColor(Hsl, red);
//     try expectEqualDeep(red_hsl, Hsl{ .h = 0, .s = 100, .l = 50 });

//     const red_hsv = convertColor(Hsv, red);
//     try expectEqualDeep(red_hsv, Hsv{ .h = 0, .s = 100, .v = 100 });

//     const gray = convertColor(u8, red);
//     try expectEqual(gray, 54); // Luma-based grayscale of red
// }

// test "extended color space round trips" {
//     const colors = [_]Rgb{
//         .{ .r = 255, .g = 0, .b = 0 }, // Red
//         .{ .r = 0, .g = 255, .b = 0 }, // Green
//         .{ .r = 0, .g = 0, .b = 255 }, // Blue
//         .{ .r = 255, .g = 255, .b = 255 }, // White
//         .{ .r = 128, .g = 128, .b = 128 }, // Gray
//     };

//     for (colors) |original| {
//         // Test all round-trip conversions
//         try expectEqualDeep(original, original.toXyz().toRgb());
//         try expectEqualDeep(original, original.toLms().toRgb());
//         try expectEqualDeep(original, original.toOklab().toRgb());
//         try expectEqualDeep(original, original.toOklch().toRgb());
//         try expectEqualDeep(original, original.toXyb().toRgb());
//     }
// }

// test "color invert matches RGB inversion" {
//     const samples = [_]Rgb{
//         .{ .r = 0, .g = 0, .b = 0 },
//         .{ .r = 255, .g = 255, .b = 255 },
//         .{ .r = 12, .g = 34, .b = 56 },
//         .{ .r = 128, .g = 64, .b = 32 },
//         .{ .r = 5, .g = 200, .b = 150 },
//     };

//     inline for (color_types) |ColorType| {
//         for (samples) |rgb| {
//             const typed: ColorType = if (comptime ColorType == Rgb) rgb else convertColor(ColorType, rgb);
//             const inverted = typed.invert();
//             const recovered_rgb: Rgb = if (comptime ColorType == Rgb) inverted else inverted.toRgb();
//             const expected = rgb.invert();

//             if (comptime ColorType == Ycbcr) {
//                 try expect(@abs(@as(i16, expected.r) - @as(i16, recovered_rgb.r)) <= 1);
//                 try expect(@abs(@as(i16, expected.g) - @as(i16, recovered_rgb.g)) <= 1);
//                 try expect(@abs(@as(i16, expected.b) - @as(i16, recovered_rgb.b)) <= 1);
//             } else {
//                 try expectEqualDeep(expected, recovered_rgb);
//             }

//             if (comptime ColorType == Rgba) {
//                 const original_rgba = convertColor(Rgba, rgb);
//                 try expectEqual(original_rgba.a, inverted.a);
//             }
//         }
//     }
// }

// test "Xyz blend matches RGB blend" {
//     const base_rgb = Rgb{ .r = 120, .g = 100, .b = 80 };
//     const overlay = Rgba{ .r = 200, .g = 50, .b = 150, .a = 128 };

//     const blended_xyz = base_rgb.toXyz().blend(overlay, Blending.normal);
//     const blended_rgb = base_rgb.blend(overlay, Blending.normal);

//     try expectEqualDeep(blended_rgb, blended_xyz.toRgb());
// }

// /// List of color types to test. This is the only thing to update when adding a new color space.
// const color_types = .{ Rgb, Rgba, Hsl, Hsv, Lab, Lch, Xyz, Lms, Oklab, Oklch, Xyb, Ycbcr };

// /// Generates the list of conversion methods based on the color type names.
// fn generateConversionMethods() [color_types.len][]const u8 {
//     var methods: [color_types.len][]const u8 = undefined;
//     inline for (color_types, 0..) |ColorType, i| {
//         const simple_name = getSimpleTypeName(ColorType);
//         methods[i] = "to" ++ simple_name;
//     }
//     return methods;
// }

// /// Skip self-conversion methods (e.g., Rgb.toRgb doesn't exist)
// fn shouldSkipMethod(comptime ColorType: type, comptime method_name: []const u8) bool {
//     if (method_name.len < 3 or !std.mem.startsWith(u8, method_name, "to")) return false;

//     const target_type_name = method_name[2..]; // Remove "to" prefix
//     const current_type_name = getSimpleTypeName(ColorType);

//     return std.mem.eql(u8, current_type_name, target_type_name);
// }

// test "comprehensive color conversion method validation and round-trip testing" {
//     @setEvalBranchQuota(10000);
//     // Test colors for round-trip validation
//     const test_colors = [_]Rgb{
//         .{ .r = 255, .g = 0, .b = 0 }, // Pure red
//         .{ .r = 0, .g = 255, .b = 0 }, // Pure green
//         .{ .r = 0, .g = 0, .b = 255 }, // Pure blue
//         .{ .r = 255, .g = 255, .b = 255 }, // White
//         .{ .r = 128, .g = 128, .b = 128 }, // Gray
//         .{ .r = 255, .g = 64, .b = 32 }, // Orange-red
//         .{ .r = 64, .g = 192, .b = 128 }, // Teal-green
//     };

//     // Use metaprogramming to verify all color types have expected conversion methods
//     // and test round-trip accuracy for each conversion
//     inline for (color_types) |ColorType| {
//         inline for (comptime generateConversionMethods()) |method_name| {
//             // Skip self-conversion and special cases
//             if (comptime shouldSkipMethod(ColorType, method_name)) continue;

//             // Verify method exists using @hasDecl
//             try expect(@hasDecl(ColorType, method_name));

//             // Test round-trip conversion accuracy for this specific method
//             if (comptime std.mem.eql(u8, method_name, "toRgb")) {
//                 // For toRgb methods, test conversion from each test color
//                 for (test_colors) |test_rgb| {
//                     const intermediate_color = convertColor(ColorType, test_rgb);
//                     const recovered_rgb = intermediate_color.toRgb();
//                     // Allow small differences for integer-based color spaces like YCbCr
//                     if (ColorType == Ycbcr) {
//                         // Integer YCbCr conversion can have rounding errors
//                         try expect(@abs(@as(i16, test_rgb.r) - @as(i16, recovered_rgb.r)) <= 1);
//                         try expect(@abs(@as(i16, test_rgb.g) - @as(i16, recovered_rgb.g)) <= 1);
//                         try expect(@abs(@as(i16, test_rgb.b) - @as(i16, recovered_rgb.b)) <= 1);
//                     } else {
//                         try expectEqualDeep(test_rgb, recovered_rgb);
//                     }
//                 }
//             }

//             // Special case: test RGBA round-trip
//             if (comptime ColorType == Rgba and std.mem.eql(u8, method_name, "toRgb")) {
//                 for (test_colors) |test_rgb| {
//                     const rgba = test_rgb.toRgba(255);
//                     try expectEqualDeep(test_rgb, rgba.toRgb());
//                 }
//             }
//         }
//     }
// }

test "ColorSpace.convert" {
    const red_hsv: Hsv(f32) = .{ .h = 0, .s = 100, .v = 100 };
    const red_rgb_u8: Rgb(u8) = red_hsv.to(.rgb).as(u8);
    try expectEqualDeep(red_rgb_u8, Rgb(u8){ .r = 255, .g = 0, .b = 0 });

    const red_rgb_f32: Rgb(f32) = red_hsv.to(.rgb);
    try expectEqualDeep(red_rgb_f32, Rgb(f32){ .r = 1, .g = 0, .b = 0 });

    const red_u8 = Rgb(u8){ .r = 255, .g = 0, .b = 0 };
    const red_hsv_recovered: Hsv(f32) = red_u8.as(f32).to(.hsv);
    try expectEqualDeep(red_hsv_recovered, red_hsv);
}

// test "color conversion accuracy with reference values" {
//     // Test with well-known reference values to verify conversion accuracy

//     // Pure red: RGB(255,0,0) should convert to specific known values
//     try expectEqualDeep(Hsl{ .h = 0, .s = 100, .l = 50 }, (Rgb{ .r = 255, .g = 0, .b = 0 }).toHsl());
//     try expectEqualDeep(Hsv{ .h = 0, .s = 100, .v = 100 }, (Rgb{ .r = 255, .g = 0, .b = 0 }).toHsv());

//     // Pure green: RGB(0,255,0) should have hue=120
//     try expectEqualDeep(Hsl{ .h = 120, .s = 100, .l = 50 }, (Rgb{ .r = 0, .g = 255, .b = 0 }).toHsl());
//     try expectEqualDeep(Hsv{ .h = 120, .s = 100, .v = 100 }, (Rgb{ .r = 0, .g = 255, .b = 0 }).toHsv());

//     // Pure blue: RGB(0,0,255) should have hue=240
//     try expectEqualDeep(Hsl{ .h = 240, .s = 100, .l = 50 }, (Rgb{ .r = 0, .g = 0, .b = 255 }).toHsl());
//     try expectEqualDeep(Hsv{ .h = 240, .s = 100, .v = 100 }, (Rgb{ .r = 0, .g = 0, .b = 255 }).toHsv());

//     // White should have L=100 in Lab space (with small tolerance for floating point)
//     const white_lab = (Rgb{ .r = 255, .g = 255, .b = 255 }).toLab();
//     try expectEqualDeep(Lab{ .l = 100, .a = 0.00526049995830391, .b = -0.010408184525267927 }, white_lab);

//     // Black should have L=0 in Lab space
//     try expectEqualDeep(Lab{ .l = 0, .a = 0, .b = 0 }, (Rgb{ .r = 0, .g = 0, .b = 0 }).toLab());

//     // Gray should have saturation=0 in HSL
//     try expectEqualDeep(Hsl{ .h = 0, .s = 0, .l = 50.19607843137255 }, (Rgb{ .r = 128, .g = 128, .b = 128 }).toHsl());

//     // Cyan: RGB(0,255,255) should have hue=180
//     try expectEqualDeep(Hsl{ .h = 180, .s = 100, .l = 50 }, (Rgb{ .r = 0, .g = 255, .b = 255 }).toHsl());

//     // Magenta: RGB(255,0,255) should have hue=300
//     try expectEqualDeep(Hsl{ .h = 300, .s = 100, .l = 50 }, (Rgb{ .r = 255, .g = 0, .b = 255 }).toHsl());

//     // Yellow: RGB(255,255,0) should have hue=60
//     try expectEqualDeep(Hsl{ .h = 60, .s = 100, .l = 50 }, (Rgb{ .r = 255, .g = 255, .b = 0 }).toHsl());
// }
