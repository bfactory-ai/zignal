// Centralized registry for all color type bindings
// This file defines which color types to expose to Python and their configurations

const std = @import("std");
const zignal = @import("zignal");

const c = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

// ============================================================================
// VALIDATION FUNCTIONS
// ============================================================================

/// Generic color component validation using type introspection
/// This function determines validation rules based on the actual field types and semantics
fn validateColorComponent(comptime ColorType: type, field_name: []const u8, value: anytype) bool {
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
                    std.mem.eql(u8, field_name, "l")) {
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

/// Type-specific validation wrapper functions
fn validateRgbComponent(field_name: []const u8, value: anytype) bool {
    return validateColorComponent(zignal.Rgb, field_name, value);
}

fn validateRgbaComponent(field_name: []const u8, value: anytype) bool {
    return validateColorComponent(zignal.Rgba, field_name, value);
}

fn validateHsvComponent(field_name: []const u8, value: anytype) bool {
    return validateColorComponent(zignal.Hsv, field_name, value);
}

fn validateHslComponent(field_name: []const u8, value: anytype) bool {
    return validateColorComponent(zignal.Hsl, field_name, value);
}

fn validateLabComponent(field_name: []const u8, value: anytype) bool {
    return validateColorComponent(zignal.Lab, field_name, value);
}

fn validateXyzComponent(field_name: []const u8, value: anytype) bool {
    return validateColorComponent(zignal.Xyz, field_name, value);
}

fn validateOklabComponent(field_name: []const u8, value: anytype) bool {
    return validateColorComponent(zignal.Oklab, field_name, value);
}

fn validateOklchComponent(field_name: []const u8, value: anytype) bool {
    return validateColorComponent(zignal.Oklch, field_name, value);
}

fn validateLchComponent(field_name: []const u8, value: anytype) bool {
    return validateColorComponent(zignal.Lch, field_name, value);
}

fn validateLmsComponent(field_name: []const u8, value: anytype) bool {
    return validateColorComponent(zignal.Lms, field_name, value);
}

fn validateXybComponent(field_name: []const u8, value: anytype) bool {
    return validateColorComponent(zignal.Xyb, field_name, value);
}

fn validateYcbcrComponent(field_name: []const u8, value: anytype) bool {
    return validateColorComponent(zignal.Ycbcr, field_name, value);
}


// ============================================================================
// COLOR BINDING REGISTRY
// ============================================================================

pub const ColorBindingEntry = struct {
    name: []const u8,
    zig_type: type,
    validation_fn: ?*const fn ([]const u8, anytype) bool,
    validation_error: []const u8,
    doc: []const u8,
};

// Define all color types we want to expose to Python
pub const ColorRegistry = [_]ColorBindingEntry{
    .{
        .name = "Rgb",
        .zig_type = zignal.Rgb,
        .validation_fn = validateRgbComponent,
        .validation_error = "RGB values must be in range 0-255",
        .doc = "RGB color in sRGB colorspace with components in range 0-255",
    },
    .{
        .name = "Rgba",
        .zig_type = zignal.Rgba,
        .validation_fn = validateRgbaComponent,
        .validation_error = "RGBA values must be in range 0-255",
        .doc = "RGBA color with alpha channel, components in range 0-255",
    },
    .{
        .name = "Hsv",
        .zig_type = zignal.Hsv,
        .validation_fn = validateHsvComponent,
        .validation_error = "HSV values must be in valid ranges (h: 0-360, s: 0-100, v: 0-100)",
        .doc = "HSV (Hue-Saturation-Value) color representation",
    },
    .{
        .name = "Hsl",
        .zig_type = zignal.Hsl,
        .validation_fn = validateHslComponent,
        .validation_error = "HSL values must be in valid ranges (h: 0-360, s: 0-100, l: 0-100)",
        .doc = "HSL (Hue-Saturation-Lightness) color representation",
    },
    .{
        .name = "Lab",
        .zig_type = zignal.Lab,
        .validation_fn = validateLabComponent,
        .validation_error = "Lab values must be in valid ranges (L: 0-100, a/b: typically -128 to 127)",
        .doc = "CIELAB color space representation",
    },
    .{
        .name = "Xyz",
        .zig_type = zignal.Xyz,
        .validation_fn = validateXyzComponent,
        .validation_error = "XYZ values must be non-negative",
        .doc = "CIE 1931 XYZ color space representation",
    },
    .{
        .name = "Oklab",
        .zig_type = zignal.Oklab,
        .validation_fn = validateOklabComponent,
        .validation_error = "Oklab values must be in valid ranges (l: 0-1, a/b: typically -0.5 to 0.5)",
        .doc = "Oklab perceptual color space representation",
    },
    .{
        .name = "Oklch",
        .zig_type = zignal.Oklch,
        .validation_fn = validateOklchComponent,
        .validation_error = "Oklch values must be in valid ranges (l: 0-1, c: 0-0.5, h: 0-360)",
        .doc = "Oklch perceptual color space in cylindrical coordinates",
    },
    .{
        .name = "Lch",
        .zig_type = zignal.Lch,
        .validation_fn = validateLchComponent,
        .validation_error = "Lch values must be in valid ranges (l: 0-100, c: >=0, h: 0-360)",
        .doc = "CIE LCH color space representation (cylindrical Lab)",
    },
    .{
        .name = "Lms",
        .zig_type = zignal.Lms,
        .validation_fn = validateLmsComponent,
        .validation_error = "LMS values should be non-negative cone responses",
        .doc = "LMS color space representing Long, Medium, Short wavelength cone responses",
    },
    .{
        .name = "Xyb",
        .zig_type = zignal.Xyb,
        .validation_fn = validateXybComponent,
        .validation_error = "XYB values used in JPEG XL compression",
        .doc = "XYB color space used in JPEG XL image compression",
    },
    .{
        .name = "Ycbcr",
        .zig_type = zignal.Ycbcr,
        .validation_fn = validateYcbcrComponent,
        .validation_error = "YCbCr values must be in range 0-255",
        .doc = "YCbCr color space used in JPEG and video encoding",
    },
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Get the binding configuration for a specific color type
pub fn getBindingConfig(comptime T: type) ?ColorBindingEntry {
    inline for (ColorRegistry) |entry| {
        if (entry.zig_type == T) {
            return entry;
        }
    }
    return null;
}

/// Check if a type is registered in the color registry
pub fn isRegisteredColorType(comptime T: type) bool {
    return getBindingConfig(T) != null;
}
