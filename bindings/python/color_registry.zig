// Centralized registry for all color type bindings
// This file defines which color types to expose to Python and their configurations

const std = @import("std");
const zignal = @import("zignal");

const c = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

/// Complete list of all color types available in the system
/// This serves as the single source of truth for auto-generation
pub const color_types = .{
    zignal.Rgb,
    zignal.Rgba,
    zignal.Hsl,
    zignal.Hsv,
    zignal.Lab,
    zignal.Lch,
    zignal.Lms,
    zignal.Oklab,
    zignal.Oklch,
    zignal.Xyb,
    zignal.Xyz,
    zignal.Ycbcr,
};

/// Check if a type is a supported color type
pub fn isSupportedColor(comptime T: type) bool {
    inline for (color_types) |ColorType| {
        if (T == ColorType) return true;
    }
    return false;
}

/// Generic color component validation using type introspection
/// This function determines validation rules based on the actual field types and semantics
pub fn validateColorComponent(comptime ColorType: type, field_name: []const u8, value: anytype) bool {
    const T = @TypeOf(value);

    // Apply validation rules grouped by color type families
    return switch (ColorType) {
        // RGB family: integer components 0-255
        zignal.Rgb, zignal.Rgba => switch (@typeInfo(T)) {
            .int => |value_info| {
                return if (value_info.signedness == .unsigned)
                    value <= 255
                else
                    value >= 0 and value <= 255;
            },
            else => true,
        },

        // HSV/HSL family: same validation rules (h: 0-360, s/v/l: 0-100)
        zignal.Hsv, zignal.Hsl => switch (@typeInfo(T)) {
            .float => {
                if (std.mem.eql(u8, field_name, "h")) {
                    return value >= 0.0 and value <= 360.0;
                } else {
                    // s, v, l all use 0-100 percentage
                    return value >= 0.0 and value <= 100.0;
                }
            },
            else => true,
        },

        // Lab: L: 0-100, a/b: -200 to 200
        zignal.Lab => switch (@typeInfo(T)) {
            .float => {
                if (std.mem.eql(u8, field_name, "l")) {
                    return value >= 0.0 and value <= 100.0;
                } else if (std.mem.eql(u8, field_name, "a") or std.mem.eql(u8, field_name, "b")) {
                    return value >= -200.0 and value <= 200.0;
                } else {
                    return value >= -1000.0 and value <= 1000.0;
                }
            },
            else => true,
        },

        // Oklab: L: 0-1, a/b: -0.5 to 0.5
        zignal.Oklab => switch (@typeInfo(T)) {
            .float => {
                if (std.mem.eql(u8, field_name, "l")) {
                    return value >= 0.0 and value <= 1.0;
                } else if (std.mem.eql(u8, field_name, "a") or std.mem.eql(u8, field_name, "b")) {
                    return value >= -0.5 and value <= 0.5;
                } else {
                    return value >= -1000.0 and value <= 1000.0;
                }
            },
            else => true,
        },

        // Oklch: L: 0-1, c: 0-0.5, h: 0-360
        zignal.Oklch => switch (@typeInfo(T)) {
            .float => {
                if (std.mem.eql(u8, field_name, "l")) {
                    return value >= 0.0 and value <= 1.0;
                } else if (std.mem.eql(u8, field_name, "c")) {
                    return value >= 0.0 and value <= 0.5;
                } else if (std.mem.eql(u8, field_name, "h")) {
                    return value >= 0.0 and value <= 360.0;
                } else {
                    return value >= -1000.0 and value <= 1000.0;
                }
            },
            else => true,
        },

        // Lch: L: 0-100, c: >=0, h: 0-360
        zignal.Lch => switch (@typeInfo(T)) {
            .float => {
                if (std.mem.eql(u8, field_name, "l")) {
                    return value >= 0.0 and value <= 100.0;
                } else if (std.mem.eql(u8, field_name, "c")) {
                    return value >= 0.0;
                } else if (std.mem.eql(u8, field_name, "h")) {
                    return value >= 0.0 and value <= 360.0;
                } else {
                    return value >= -1000.0 and value <= 1000.0;
                }
            },
            else => true,
        },

        // XYZ: 0-150 (can exceed 100)
        zignal.Xyz => switch (@typeInfo(T)) {
            .float => value >= 0.0 and value <= 150.0,
            else => true,
        },

        // YCbCr: 0-255
        zignal.Ycbcr => switch (@typeInfo(T)) {
            .float => value >= 0.0 and value <= 255.0,
            else => true,
        },

        // LMS, XYB: generic validation for other color spaces
        zignal.Lms, zignal.Xyb => switch (@typeInfo(T)) {
            .float => value >= -1000.0 and value <= 1000.0,
            else => true,
        },

        else => @compileError("Missing validation for color type '" ++ @typeName(ColorType) ++ "'. "),
    };
}

// ============================================================================
// TYPE-DRIVEN COLOR FUNCTIONS
// ============================================================================

/// Get the validation error message for a specific color type
pub fn getValidationErrorMessage(comptime ColorType: type) []const u8 {
    // Return appropriate error message based on color type
    return switch (ColorType) {
        zignal.Rgb, zignal.Rgba => "RGB values must be in range 0-255",
        zignal.Hsv => "HSV values must be in valid ranges (h: 0-360, s: 0-100, v: 0-100)",
        zignal.Hsl => "HSL values must be in valid ranges (h: 0-360, s: 0-100, l: 0-100)",
        zignal.Lab => "Lab values must be in valid ranges (L: 0-100, a/b: typically -128 to 127)",
        zignal.Xyz => "XYZ values must be non-negative",
        zignal.Oklab => "Oklab values must be in valid ranges (l: 0-1, a/b: typically -0.5 to 0.5)",
        zignal.Oklch => "Oklch values must be in valid ranges (l: 0-1, c: 0-0.5, h: 0-360)",
        zignal.Lch => "Lch values must be in valid ranges (l: 0-100, c: >=0, h: 0-360)",
        zignal.Lms => "LMS values should be non-negative cone responses",
        zignal.Xyb => "XYB values used in JPEG XL compression",
        zignal.Ycbcr => "YCbCr values must be in range 0-255",
        else => @compileError("Missing validation error message for color type '" ++ @typeName(ColorType) ++ "'. "),
    };
}

/// Get the documentation string for a specific color type
pub fn getDocumentationString(comptime ColorType: type) []const u8 {
    // Return appropriate documentation based on color type
    return switch (ColorType) {
        zignal.Rgb => "RGB color in sRGB colorspace with components in range 0-255",
        zignal.Rgba => "RGBA color with alpha channel, components in range 0-255",
        zignal.Hsv => "HSV (Hue-Saturation-Value) color representation",
        zignal.Hsl => "HSL (Hue-Saturation-Lightness) color representation",
        zignal.Lab => "CIELAB color space representation",
        zignal.Xyz => "CIE 1931 XYZ color space representation",
        zignal.Oklab => "Oklab perceptual color space representation",
        zignal.Oklch => "Oklch perceptual color space in cylindrical coordinates",
        zignal.Lch => "CIE LCH color space representation (cylindrical Lab)",
        zignal.Lms => "LMS color space representing Long, Medium, Short wavelength cone responses",
        zignal.Xyb => "XYB color space used in JPEG XL image compression",
        zignal.Ycbcr => "YCbCr color space used in JPEG and video encoding",
        else => @compileError("Missing documentation for color type '" ++ @typeName(ColorType) ++ "'. "),
    };
}
