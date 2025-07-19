// Centralized registry for all color type bindings
// This file defines which color types to expose to Python and their configurations

const std = @import("std");
const zignal = @import("zignal");
const color_factory = @import("color_factory.zig");

const c = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

// ============================================================================
// VALIDATION FUNCTIONS
// ============================================================================

// RGB validation: components must be 0-255
fn validateRgbComponent(field_name: []const u8, value: anytype) bool {
    _ = field_name;
    const T = @TypeOf(value);
    return switch (@typeInfo(T)) {
        .int => |info| if (info.signedness == .unsigned)
            value <= 255
        else
            value >= 0 and value <= 255,
        else => false,
    };
}

// HSV/HSL validation: h=0-360, s/v/l=0-100
fn validateHsvHslComponent(field_name: []const u8, value: anytype) bool {
    const T = @TypeOf(value);
    return switch (@typeInfo(T)) {
        .float => {
            if (std.mem.eql(u8, field_name, "h")) {
                return value >= 0.0 and value <= 360.0;
            } else if (std.mem.eql(u8, field_name, "s") or 
                      std.mem.eql(u8, field_name, "v") or 
                      std.mem.eql(u8, field_name, "l")) {
                return value >= 0.0 and value <= 100.0;
            } else {
                return false;
            }
        },
        else => false,
    };
}

// Lab validation: L=0-100, a/b typically -128 to 127 but can exceed
fn validateLabComponent(field_name: []const u8, value: anytype) bool {
    const T = @TypeOf(value);
    return switch (@typeInfo(T)) {
        .float => {
            if (std.mem.eql(u8, field_name, "l")) {
                return value >= 0.0 and value <= 100.0;
            } else {
                // a and b components can have wider ranges
                return value >= -200.0 and value <= 200.0;
            }
        },
        else => false,
    };
}

// XYZ validation: all components typically 0-100 but can exceed
fn validateXyzComponent(field_name: []const u8, value: anytype) bool {
    _ = field_name;
    const T = @TypeOf(value);
    return switch (@typeInfo(T)) {
        .float => value >= 0.0 and value <= 150.0, // XYZ can exceed 100
        else => false,
    };
}

// Oklab/Oklch validation
fn validateOklabComponent(field_name: []const u8, value: anytype) bool {
    const T = @TypeOf(value);
    return switch (@typeInfo(T)) {
        .float => {
            if (std.mem.eql(u8, field_name, "l")) {
                return value >= 0.0 and value <= 1.0;
            } else if (std.mem.eql(u8, field_name, "c")) {
                return value >= 0.0 and value <= 0.5; // Chroma
            } else if (std.mem.eql(u8, field_name, "h")) {
                return value >= 0.0 and value <= 360.0; // Hue for Oklch
            } else {
                // a and b for Oklab
                return value >= -0.5 and value <= 0.5;
            }
        },
        else => false,
    };
}

// Generic validation for types without special constraints
fn validateGenericComponent(field_name: []const u8, value: anytype) bool {
    _ = field_name;
    _ = value;
    return true; // Accept any value
}

// Lch validation: L=0-100, C>=0, H=0-360
fn validateLchComponent(field_name: []const u8, value: anytype) bool {
    const T = @TypeOf(value);
    return switch (@typeInfo(T)) {
        .float => {
            if (std.mem.eql(u8, field_name, "l")) {
                return value >= 0.0 and value <= 100.0;
            } else if (std.mem.eql(u8, field_name, "c")) {
                return value >= 0.0; // Chroma can be unbounded, but should be non-negative
            } else if (std.mem.eql(u8, field_name, "h")) {
                return value >= 0.0 and value <= 360.0;
            }
            return true;
        },
        else => false,
    };
}

// YCbCr validation: Y=0-255, Cb=0-255, Cr=0-255 (f32 components)
fn validateYcbcrComponent(field_name: []const u8, value: anytype) bool {
    _ = field_name;
    const T = @TypeOf(value);
    return switch (@typeInfo(T)) {
        .float => value >= 0.0 and value <= 255.0,
        else => false,
    };
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
        .validation_fn = validateRgbComponent,
        .validation_error = "RGBA values must be in range 0-255",
        .doc = "RGBA color with alpha channel, components in range 0-255",
    },
    .{
        .name = "Hsv",
        .zig_type = zignal.Hsv,
        .validation_fn = validateHsvHslComponent,
        .validation_error = "HSV values must be in valid ranges (h: 0-360, s: 0-100, v: 0-100)",
        .doc = "HSV (Hue-Saturation-Value) color representation",
    },
    .{
        .name = "Hsl",
        .zig_type = zignal.Hsl,
        .validation_fn = validateHsvHslComponent,
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
        .validation_fn = validateOklabComponent,
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
        .validation_fn = validateGenericComponent,
        .validation_error = "LMS values should be non-negative cone responses",
        .doc = "LMS color space representing Long, Medium, Short wavelength cone responses",
    },
    .{
        .name = "Xyb",
        .zig_type = zignal.Xyb,
        .validation_fn = validateGenericComponent,
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