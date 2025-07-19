// Centralized registry for all color type bindings
// This file defines which color types to expose to Python and their configurations

const std = @import("std");
const zignal = @import("zignal");

const c = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

// ============================================================================
// COLOR TYPES REGISTRY
// ============================================================================

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

// ============================================================================
// VALIDATION FUNCTIONS
// ============================================================================

/// Generic color component validation using type introspection
/// This function determines validation rules based on the actual field types and semantics
pub fn validateColorComponent(comptime ColorType: type, field_name: []const u8, value: anytype) bool {
    // Compile-time validation that ColorType is supported
    switch (ColorType) {
        zignal.Rgb,
        zignal.Rgba,
        zignal.Hsv,
        zignal.Hsl,
        zignal.Lab,
        zignal.Xyz,
        zignal.Oklab,
        zignal.Oklch,
        zignal.Lch,
        zignal.Lms,
        zignal.Xyb,
        zignal.Ycbcr,
        => {},
        else => @compileError("Missing validation for color type '" ++ @typeName(ColorType) ++ "'. "),
    }

    const T = @TypeOf(value);

    // Apply validation rules based on color type and field name semantics
    return switch (@typeInfo(T)) {
        // Integer validation (RGB components): 0-255 range
        .int => |value_info| {
            if (ColorType == zignal.Rgb or ColorType == zignal.Rgba) {
                return if (value_info.signedness == .unsigned)
                    value <= 255
                else
                    value >= 0 and value <= 255;
            }
            return true; // Other integer types accepted as-is
        },

        // Float validation: determine range by field name semantics and color type
        .float => {
            // Hue components (h): 0-360 degrees
            if (std.mem.eql(u8, field_name, "h")) {
                return value >= 0.0 and value <= 360.0;
            }
            // Saturation, Value, Lightness (s, v, l): depends on color space
            else if (std.mem.eql(u8, field_name, "s") or
                std.mem.eql(u8, field_name, "v") or
                std.mem.eql(u8, field_name, "l"))
            {
                // HSV/HSL: 0-100 percentage
                if (ColorType == zignal.Hsv or ColorType == zignal.Hsl) {
                    return value >= 0.0 and value <= 100.0;
                }
                // Lab L*: 0-100
                else if (ColorType == zignal.Lab and std.mem.eql(u8, field_name, "l")) {
                    return value >= 0.0 and value <= 100.0;
                }
                // Oklab/Oklch L: 0-1
                else if ((ColorType == zignal.Oklab or ColorType == zignal.Oklch) and std.mem.eql(u8, field_name, "l")) {
                    return value >= 0.0 and value <= 1.0;
                }
                // LCH L: 0-100
                else if (ColorType == zignal.Lch and std.mem.eql(u8, field_name, "l")) {
                    return value >= 0.0 and value <= 100.0;
                }
                // Default for unmatched s/v/l
                else {
                    return value >= 0.0 and value <= 100.0;
                }
            }
            // Chroma components (c): non-negative
            else if (std.mem.eql(u8, field_name, "c")) {
                if (ColorType == zignal.Oklch) {
                    return value >= 0.0 and value <= 0.5;
                } else {
                    return value >= 0.0; // LCH and others: just non-negative
                }
            }
            // Lab a*, b* components: wider range allowed
            else if ((std.mem.eql(u8, field_name, "a") or std.mem.eql(u8, field_name, "b")) and ColorType == zignal.Lab) {
                return value >= -200.0 and value <= 200.0;
            }
            // Oklab a*, b* components: -0.5 to 0.5
            else if ((std.mem.eql(u8, field_name, "a") or std.mem.eql(u8, field_name, "b")) and ColorType == zignal.Oklab) {
                return value >= -0.5 and value <= 0.5;
            }
            // XYZ components: 0-150 (can exceed 100)
            else if (ColorType == zignal.Xyz) {
                return value >= 0.0 and value <= 150.0;
            }
            // YCbCr components: 0-255
            else if (ColorType == zignal.Ycbcr) {
                return value >= 0.0 and value <= 255.0;
            }
            // Generic float validation: accept reasonable range
            else {
                return value >= -1000.0 and value <= 1000.0;
            }
        },

        else => true, // Accept other types as-is
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
