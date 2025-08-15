//! Image filtering and convolution operations

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const convertColor = @import("../color.zig").convertColor;
const meta = @import("../meta.zig");
const as = meta.as;
const isScalar = meta.isScalar;
const is4xu8Struct = meta.is4xu8Struct;
const Image = @import("Image.zig").Image;

/// Border handling modes for filter operations
pub const BorderMode = enum {
    /// Pad with zeros
    zero,
    /// Replicate edge pixels
    replicate,
    /// Mirror at edges
    mirror,
    /// Wrap around (circular)
    wrap,
};

/// Check if a struct type has an alpha channel (4th field named 'a' or 'alpha')
fn hasAlphaChannel(comptime T: type) bool {
    const fields = std.meta.fields(T);
    if (fields.len != 4) return false;
    const last_field = fields[3];
    return std.mem.eql(u8, last_field.name, "a") or std.mem.eql(u8, last_field.name, "alpha");
}

/// Filter operations for Image(T)
pub fn Filter(comptime T: type) type {
    return struct {
        const Self = Image(T);

        /// Computes a blurred version of `self` using a box blur algorithm, efficiently implemented
        /// using an integral image. The `radius` parameter determines the size of the box window.
        /// This function is optimized using SIMD instructions for performance where applicable.
        pub fn boxBlur(self: Self, allocator: std.mem.Allocator, blurred: *Self, radius: usize) !void {
            if (!self.hasSameShape(blurred.*)) {
                blurred.* = try .initAlloc(allocator, self.rows, self.cols);
            }
            if (radius == 0) {
                self.copy(blurred.*);
                return;
            }

            switch (@typeInfo(T)) {
                .int, .float => {
                    var sat: Image(f32) = undefined;
                    try self.integral(allocator, &sat);
                    defer sat.deinit(allocator);

                    const simd_len = std.simd.suggestVectorLength(f32) orelse 1;

                    // Process each row
                    for (0..self.rows) |r| {
                        const r1 = r -| radius;
                        const r2 = @min(r + radius, self.rows - 1);
                        const r1_offset = r1 * self.cols;
                        const r2_offset = r2 * self.cols;

                        var c: usize = 0;

                        // Process SIMD chunks where safe (away from borders)
                        const row_safe = r >= radius and r + radius < self.rows;
                        if (simd_len > 1 and self.cols > 2 * radius + simd_len and row_safe) {
                            // Skip left border
                            while (c < radius) : (c += 1) {
                                const c1 = c -| radius;
                                const c2 = @min(c + radius, self.cols - 1);
                                const area: f32 = @floatFromInt((r2 - r1) * (c2 - c1));
                                const sum = sat.at(r2, c2).* - sat.at(r2, c1).* -
                                    sat.at(r1, c2).* + sat.at(r1, c1).*;
                                blurred.at(r, c).* = if (@typeInfo(T) == .int)
                                    @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(sum / area))))
                                else
                                    as(T, sum / area);
                            }

                            // SIMD middle section (constant area when row is safe)
                            const safe_end = self.cols - radius - simd_len;
                            if (c <= safe_end) {
                                const const_area: f32 = @floatFromInt((r2 - r1) * 2 * radius);
                                const area_vec: @Vector(simd_len, f32) = @splat(const_area);

                                while (c <= safe_end) : (c += simd_len) {
                                    const c1 = c - radius;
                                    const c2 = c + radius;
                                    const int11: @Vector(simd_len, f32) = sat.data[r1_offset + c1 ..][0..simd_len].*;
                                    const int12: @Vector(simd_len, f32) = sat.data[r1_offset + c2 ..][0..simd_len].*;
                                    const int21: @Vector(simd_len, f32) = sat.data[r2_offset + c1 ..][0..simd_len].*;
                                    const int22: @Vector(simd_len, f32) = sat.data[r2_offset + c2 ..][0..simd_len].*;
                                    const sums = int22 - int21 - int12 + int11;
                                    const vals = sums / area_vec;

                                    for (0..simd_len) |i| {
                                        blurred.at(r, c + i).* = if (@typeInfo(T) == .int)
                                            @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(vals[i]))))
                                        else
                                            vals[i];
                                    }
                                }
                            }
                        }

                        // Process remaining pixels (right border and any leftover)
                        while (c < self.cols) : (c += 1) {
                            const c1 = c -| radius;
                            const c2 = @min(c + radius, self.cols - 1);
                            const area: f32 = @floatFromInt((r2 - r1) * (c2 - c1));
                            const sum = sat.at(r2, c2).* - sat.at(r2, c1).* - sat.at(r1, c2).* + sat.at(r1, c1).*;
                            blurred.at(r, c).* = if (@typeInfo(T) == .int)
                                @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(sum / area))))
                            else
                                as(T, sum / area);
                        }
                    }
                },
                .@"struct" => {
                    if (is4xu8Struct(T)) {
                        try boxBlur4xu8Simd(self, allocator, blurred, radius);
                    } else {
                        // Generic struct path for other color types
                        var sat: Image([Self.channels()]f32) = undefined;
                        try self.integral(allocator, &sat);
                        defer sat.deinit(allocator);

                        for (0..self.rows) |r| {
                            for (0..self.cols) |c| {
                                const r1 = r -| radius;
                                const c1 = c -| radius;
                                const r2 = @min(r + radius, self.rows - 1);
                                const c2 = @min(c + radius, self.cols - 1);
                                const area: f32 = @floatFromInt((r2 - r1) * (c2 - c1));

                                inline for (std.meta.fields(T), 0..) |f, i| {
                                    const sum = sat.at(r2, c2)[i] - sat.at(r2, c1)[i] -
                                        sat.at(r1, c2)[i] + sat.at(r1, c1)[i];
                                    @field(blurred.at(r, c).*, f.name) = switch (@typeInfo(f.type)) {
                                        .int => @intFromFloat(@max(std.math.minInt(f.type), @min(std.math.maxInt(f.type), @round(sum / area)))),
                                        .float => as(f.type, sum / area),
                                        else => @compileError("Can't compute the boxBlur image with struct fields of type " ++ @typeName(f.type) ++ "."),
                                    };
                                }
                            }
                        }
                    }
                },
                else => @compileError("Can't compute the boxBlur image of " ++ @typeName(T) ++ "."),
            }
        }

        /// Optimized box blur implementation for structs with 4 u8 fields using SIMD throughout.
        /// This is automatically called by boxBlur() when T has exactly 4 u8 fields (e.g., RGBA, BGRA, etc).
        fn boxBlur4xu8Simd(self: Self, allocator: std.mem.Allocator, blurred: *Self, radius: usize) !void {
            // Verify at compile time that this is a struct with 4 u8 fields
            comptime {
                const fields = std.meta.fields(T);
                assert(fields.len == 4);
                for (fields) |field| {
                    assert(field.type == u8);
                }
            }

            // Initialize output if needed
            if (!self.hasSameShape(blurred.*)) {
                blurred.* = try .initAlloc(allocator, self.rows, self.cols);
            }
            if (radius == 0) {
                self.copy(blurred.*);
                return;
            }

            const fields = std.meta.fields(T);
            const has_alpha = comptime hasAlphaChannel(T);
            const channels_to_process = if (has_alpha) 3 else 4;

            // Create integral image - only for channels we'll process
            var sat = try Image([4]f32).initAlloc(allocator, self.rows, self.cols);
            defer sat.deinit(allocator);

            // Build integral image - first pass: row-wise cumulative sums
            for (0..self.rows) |r| {
                var tmp: @Vector(4, f32) = @splat(0);
                const row_offset = r * self.stride;
                const out_offset = r * sat.cols;

                for (0..self.cols) |c| {
                    const pixel = self.data[row_offset + c];
                    var pixel_vec: @Vector(4, f32) = @splat(0); // Initialize all to 0
                    // Only accumulate channels we're processing
                    inline for (0..channels_to_process) |i| {
                        pixel_vec[i] = @floatFromInt(@field(pixel, fields[i].name));
                    }
                    tmp += pixel_vec;
                    sat.data[out_offset + c] = tmp;
                }
            }

            // Second pass: column-wise cumulative sums
            for (1..self.rows) |r| {
                const prev_row_offset = (r - 1) * sat.cols;
                const curr_row_offset = r * sat.cols;

                for (0..self.cols) |c| {
                    const prev_vec: @Vector(4, f32) = sat.data[prev_row_offset + c];
                    const curr_vec: @Vector(4, f32) = sat.data[curr_row_offset + c];
                    sat.data[curr_row_offset + c] = prev_vec + curr_vec;
                }
            }

            // Apply box blur with SIMD
            for (0..self.rows) |r| {
                for (0..self.cols) |c| {
                    const r1 = r -| radius;
                    const c1 = c -| radius;
                    const r2 = @min(r + radius, self.rows - 1);
                    const c2 = @min(c + radius, self.cols - 1);
                    const area: f32 = @floatFromInt((r2 - r1) * (c2 - c1));
                    const area_vec: @Vector(4, f32) = @splat(area);

                    // Use vectors for the box sum calculation
                    const v_r2c2: @Vector(4, f32) = sat.at(r2, c2).*;
                    const v_r2c1: @Vector(4, f32) = sat.at(r2, c1).*;
                    const v_r1c2: @Vector(4, f32) = sat.at(r1, c2).*;
                    const v_r1c1: @Vector(4, f32) = sat.at(r1, c1).*;

                    const sum_vec = v_r2c2 - v_r2c1 - v_r1c2 + v_r1c1;
                    const avg_vec = sum_vec / area_vec;

                    // Convert back to struct
                    var result: T = undefined;
                    // Process color channels
                    inline for (0..channels_to_process) |i| {
                        @field(result, fields[i].name) = @intFromFloat(@max(0, @min(255, @round(avg_vec[i]))));
                    }
                    // Preserve alpha if present
                    if (has_alpha) {
                        @field(result, fields[3].name) = @field(self.at(r, c).*, fields[3].name);
                    }
                    blurred.at(r, c).* = result;
                }
            }
        }

        /// Computes a sharpened version of `self` by enhancing edges.
        /// It uses the formula `sharpened = 2 * original - blurred`, where `blurred` is a box-blurred
        /// version of the original image (calculated efficiently using an integral image).
        /// The `radius` parameter controls the size of the blur. This operation effectively
        /// increases the contrast at edges. SIMD optimizations are used for performance where applicable.
        pub fn sharpen(self: Self, allocator: std.mem.Allocator, sharpened: *Self, radius: usize) !void {
            if (!self.hasSameShape(sharpened.*)) {
                sharpened.* = try .initAlloc(allocator, self.rows, self.cols);
            }
            if (radius == 0) {
                self.copy(sharpened.*);
                return;
            }

            switch (@typeInfo(T)) {
                .int, .float => {
                    var sat: Image(f32) = undefined;
                    defer sat.deinit(allocator);
                    try self.integral(allocator, &sat);

                    const simd_len = std.simd.suggestVectorLength(f32) orelse 1;

                    // Process each row
                    for (0..self.rows) |r| {
                        const r1 = r -| radius;
                        const r2 = @min(r + radius, self.rows - 1);
                        const r1_offset = r1 * self.cols;
                        const r2_offset = r2 * self.cols;

                        var c: usize = 0;

                        // Process SIMD chunks where safe (away from borders)
                        const row_safe = r >= radius and r + radius < self.rows;
                        if (simd_len > 1 and self.cols > 2 * radius + simd_len and row_safe) {
                            // Skip left border
                            while (c < radius) : (c += 1) {
                                const c1 = c -| radius;
                                const c2 = @min(c + radius, self.cols - 1);
                                const area: f32 = @floatFromInt((r2 - r1) * (c2 - c1));
                                const sum = sat.at(r2, c2).* - sat.at(r2, c1).* -
                                    sat.at(r1, c2).* + sat.at(r1, c1).*;
                                const blurred = sum / area;
                                sharpened.at(r, c).* = if (@typeInfo(T) == .int)
                                    @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(2 * as(f32, self.at(r, c).*) - blurred))))
                                else
                                    as(T, 2 * as(f32, self.at(r, c).*) - blurred);
                            }

                            // SIMD middle section (constant area when row is safe)
                            const safe_end = self.cols - radius - simd_len;
                            if (c <= safe_end) {
                                const const_area: f32 = @floatFromInt((2 * radius + 1) * (2 * radius + 1));
                                const area_vec: @Vector(simd_len, f32) = @splat(const_area);

                                while (c <= safe_end) : (c += simd_len) {
                                    const c1 = c - radius;
                                    const c2 = c + radius;
                                    const int11: @Vector(simd_len, f32) = sat.data[r1_offset + c1 ..][0..simd_len].*;
                                    const int12: @Vector(simd_len, f32) = sat.data[r1_offset + c2 ..][0..simd_len].*;
                                    const int21: @Vector(simd_len, f32) = sat.data[r2_offset + c1 ..][0..simd_len].*;
                                    const int22: @Vector(simd_len, f32) = sat.data[r2_offset + c2 ..][0..simd_len].*;
                                    const sums = int22 - int21 - int12 + int11;
                                    const blurred_vals = sums / area_vec;

                                    for (0..simd_len) |i| {
                                        const original = self.at(r, c + i).*;
                                        sharpened.at(r, c + i).* = if (@typeInfo(T) == .int)
                                            @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(2 * as(f32, original) - blurred_vals[i]))))
                                        else
                                            as(T, 2 * as(f32, original) - blurred_vals[i]);
                                    }
                                }
                            }
                        }

                        // Process remaining pixels (right border and any leftover)
                        while (c < self.cols) : (c += 1) {
                            const c1 = c -| radius;
                            const c2 = @min(c + radius, self.cols - 1);
                            const area: f32 = @floatFromInt((r2 - r1) * (c2 - c1));
                            const sum = sat.at(r2, c2).* - sat.at(r2, c1).* -
                                sat.at(r1, c2).* + sat.at(r1, c1).*;
                            const blurred = sum / area;
                            sharpened.at(r, c).* = if (@typeInfo(T) == .int)
                                @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(2 * as(f32, self.at(r, c).*) - blurred))))
                            else
                                as(T, 2 * as(f32, self.at(r, c).*) - blurred);
                        }
                    }
                },
                .@"struct" => {
                    if (is4xu8Struct(T)) {
                        try sharpen4xu8Simd(self, allocator, sharpened, radius);
                    } else {
                        // Generic struct path for other color types
                        var sat: Image([Self.channels()]f32) = undefined;
                        try self.integral(allocator, &sat);
                        defer sat.deinit(allocator);

                        for (0..self.rows) |r| {
                            for (0..self.cols) |c| {
                                const r1 = r -| radius;
                                const c1 = c -| radius;
                                const r2 = @min(r + radius, self.rows - 1);
                                const c2 = @min(c + radius, self.cols - 1);
                                const area: f32 = @floatFromInt((r2 - r1) * (c2 - c1));

                                inline for (std.meta.fields(T), 0..) |f, i| {
                                    const sum = sat.at(r2, c2)[i] - sat.at(r2, c1)[i] -
                                        sat.at(r1, c2)[i] + sat.at(r1, c1)[i];
                                    const blurred = sum / area;
                                    const original = @field(self.at(r, c).*, f.name);
                                    @field(sharpened.at(r, c).*, f.name) = switch (@typeInfo(f.type)) {
                                        .int => blk: {
                                            const sharpened_val = 2 * as(f32, original) - blurred;
                                            break :blk @intFromFloat(@max(std.math.minInt(f.type), @min(std.math.maxInt(f.type), @round(sharpened_val))));
                                        },
                                        .float => as(f.type, 2 * as(f32, original) - blurred),
                                        else => @compileError("Can't compute the sharpen image with struct fields of type " ++ @typeName(f.type) ++ "."),
                                    };
                                }
                            }
                        }
                    }
                },
                else => @compileError("Can't compute the sharpen image of " ++ @typeName(T) ++ "."),
            }
        }

        /// Optimized sharpen implementation for structs with 4 u8 fields using SIMD throughout.
        /// This is automatically called by sharpen() when T has exactly 4 u8 fields (e.g., RGBA, BGRA, etc).
        fn sharpen4xu8Simd(self: Self, allocator: std.mem.Allocator, sharpened: *Self, radius: usize) !void {
            // Verify at compile time that this is a struct with 4 u8 fields
            comptime {
                const fields = std.meta.fields(T);
                assert(fields.len == 4);
                for (fields) |field| {
                    assert(field.type == u8);
                }
            }

            // Initialize output if needed
            if (!self.hasSameShape(sharpened.*)) {
                sharpened.* = try .initAlloc(allocator, self.rows, self.cols);
            }
            if (radius == 0) {
                self.copy(sharpened.*);
                return;
            }

            const fields = std.meta.fields(T);
            const has_alpha = comptime hasAlphaChannel(T);
            const channels_to_process = if (has_alpha) 3 else 4;

            // Create integral image - only for channels we'll process
            var sat = try Image([4]f32).initAlloc(allocator, self.rows, self.cols);
            defer sat.deinit(allocator);

            // Build integral image - first pass: row-wise cumulative sums
            for (0..self.rows) |r| {
                var tmp: @Vector(4, f32) = @splat(0);
                const row_offset = r * self.stride;
                const out_offset = r * sat.cols;

                for (0..self.cols) |c| {
                    const pixel = self.data[row_offset + c];
                    var pixel_vec: @Vector(4, f32) = @splat(0); // Initialize all to 0
                    // Only accumulate channels we're processing
                    inline for (0..channels_to_process) |i| {
                        pixel_vec[i] = @floatFromInt(@field(pixel, fields[i].name));
                    }
                    tmp += pixel_vec;
                    sat.data[out_offset + c] = tmp;
                }
            }

            // Second pass: column-wise cumulative sums
            for (1..self.rows) |r| {
                const prev_row_offset = (r - 1) * sat.cols;
                const curr_row_offset = r * sat.cols;

                for (0..self.cols) |c| {
                    const prev_vec: @Vector(4, f32) = sat.data[prev_row_offset + c];
                    const curr_vec: @Vector(4, f32) = sat.data[curr_row_offset + c];
                    sat.data[curr_row_offset + c] = prev_vec + curr_vec;
                }
            }

            // Apply sharpen with SIMD: sharpened = 2 * original - blurred
            for (0..self.rows) |r| {
                for (0..self.cols) |c| {
                    const r1 = r -| radius;
                    const c1 = c -| radius;
                    const r2 = @min(r + radius, self.rows - 1);
                    const c2 = @min(c + radius, self.cols - 1);
                    const area: f32 = @floatFromInt((r2 - r1) * (c2 - c1));
                    const area_vec: @Vector(4, f32) = @splat(area);

                    // Use vectors for the box sum calculation (blur)
                    const v_r2c2: @Vector(4, f32) = sat.at(r2, c2).*;
                    const v_r2c1: @Vector(4, f32) = sat.at(r2, c1).*;
                    const v_r1c2: @Vector(4, f32) = sat.at(r1, c2).*;
                    const v_r1c1: @Vector(4, f32) = sat.at(r1, c1).*;

                    const sum_vec = v_r2c2 - v_r2c1 - v_r1c2 + v_r1c1;
                    const blurred_vec = sum_vec / area_vec;

                    // Get original pixel as vector
                    const original_pixel = self.data[r * self.stride + c];
                    var original_vec: @Vector(4, f32) = @splat(0);
                    // Only process color channels
                    inline for (0..channels_to_process) |i| {
                        original_vec[i] = @floatFromInt(@field(original_pixel, fields[i].name));
                    }

                    // Apply sharpening formula: 2 * original - blurred (only for color channels)
                    const sharpened_vec = @as(@Vector(4, f32), @splat(2.0)) * original_vec - blurred_vec;

                    // Convert back to struct with clamping
                    var result: T = undefined;
                    // Process color channels
                    inline for (0..channels_to_process) |i| {
                        @field(result, fields[i].name) = @intFromFloat(@max(0, @min(255, @round(sharpened_vec[i]))));
                    }
                    // Preserve alpha if present
                    if (has_alpha) {
                        @field(result, fields[3].name) = @field(original_pixel, fields[3].name);
                    }
                    sharpened.at(r, c).* = result;
                }
            }
        }

        /// Applies a 2D convolution with the given kernel to the image.
        ///
        /// Parameters:
        /// - `allocator`: The allocator to use if `out` needs to be (re)initialized.
        /// - `kernel`: A 2D array representing the convolution kernel.
        /// - `out`: An out-parameter pointer to an `Image(T)` that will be filled with the convolved image.
        /// - `border_mode`: How to handle pixels at the image borders.
        pub fn convolve(self: Self, allocator: Allocator, kernel: anytype, out: *Self, border_mode: BorderMode) !void {
            const kernel_info = @typeInfo(@TypeOf(kernel));
            if (kernel_info != .array) @compileError("Kernel must be a 2D array");
            const outer_array = kernel_info.array;
            if (@typeInfo(outer_array.child) != .array) @compileError("Kernel must be a 2D array");

            const kernel_height = outer_array.len;
            const kernel_width = @typeInfo(outer_array.child).array.len;

            if (!self.hasSameShape(out.*)) {
                out.* = try .initAlloc(allocator, self.rows, self.cols);
            }

            // Check for special optimized cases
            if (kernel_height == 3 and kernel_width == 3) {
                return convolve3x3(self, kernel, out.*, border_mode);
            }

            // General convolution
            const half_h = kernel_height / 2;
            const half_w = kernel_width / 2;

            switch (@typeInfo(T)) {
                .int, .float => {
                    // Scalar types - single channel convolution
                    for (0..self.rows) |r| {
                        for (0..self.cols) |c| {
                            var accumulator: f32 = 0;

                            const in_interior = (r >= half_h and r + half_h < self.rows and c >= half_w and c + half_w < self.cols);
                            if (in_interior) {
                                // Fast path: no border handling needed
                                const r0: usize = r - half_h;
                                const c0: usize = c - half_w;
                                for (0..kernel_height) |kr| {
                                    const rr = r0 + kr;
                                    for (0..kernel_width) |kc| {
                                        const cc = c0 + kc;
                                        const pixel_val = self.at(rr, cc).*;
                                        const kernel_val = kernel[kr][kc];
                                        accumulator += as(f32, pixel_val) * as(f32, kernel_val);
                                    }
                                }
                            } else {
                                // Border path: fetch with border handling
                                for (0..kernel_height) |kr| {
                                    for (0..kernel_width) |kc| {
                                        const src_r = @as(isize, @intCast(r)) + @as(isize, @intCast(kr)) - @as(isize, @intCast(half_h));
                                        const src_c = @as(isize, @intCast(c)) + @as(isize, @intCast(kc)) - @as(isize, @intCast(half_w));
                                        const pixel_val = getPixelWithBorder(self, src_r, src_c, border_mode);
                                        const kernel_val = kernel[kr][kc];
                                        accumulator += as(f32, pixel_val) * as(f32, kernel_val);
                                    }
                                }
                            }

                            out.at(r, c).* = switch (@typeInfo(T)) {
                                .int => @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(accumulator)))),
                                .float => as(T, accumulator),
                                else => unreachable,
                            };
                        }
                    }
                },
                .@"struct" => {
                    // Struct types (RGB, RGBA, etc.) - channel-wise convolution
                    const fields = std.meta.fields(T);
                    const has_alpha = comptime hasAlphaChannel(T);
                    const channels_to_process = if (has_alpha) 3 else fields.len;

                    for (0..self.rows) |r| {
                        for (0..self.cols) |c| {
                            var result_pixel: T = undefined;

                            // Process color channels (skip alpha if present)
                            inline for (0..channels_to_process) |field_idx| {
                                const field = fields[field_idx];
                                var accumulator: f32 = 0;
                                const in_interior = (r >= half_h and r + half_h < self.rows and c >= half_w and c + half_w < self.cols);
                                if (in_interior) {
                                    const r0: usize = r - half_h;
                                    const c0: usize = c - half_w;
                                    for (0..kernel_height) |kr| {
                                        const rr = r0 + kr;
                                        for (0..kernel_width) |kc| {
                                            const cc = c0 + kc;
                                            const pixel_val = self.at(rr, cc).*;
                                            const channel_val = @field(pixel_val, field.name);
                                            const kernel_val = kernel[kr][kc];
                                            accumulator += as(f32, channel_val) * as(f32, kernel_val);
                                        }
                                    }
                                } else {
                                    for (0..kernel_height) |kr| {
                                        for (0..kernel_width) |kc| {
                                            const src_r = @as(isize, @intCast(r)) + @as(isize, @intCast(kr)) - @as(isize, @intCast(half_h));
                                            const src_c = @as(isize, @intCast(c)) + @as(isize, @intCast(kc)) - @as(isize, @intCast(half_w));
                                            const pixel_val = getPixelWithBorder(self, src_r, src_c, border_mode);
                                            const channel_val = @field(pixel_val, field.name);
                                            const kernel_val = kernel[kr][kc];
                                            accumulator += as(f32, channel_val) * as(f32, kernel_val);
                                        }
                                    }
                                }

                                @field(result_pixel, field.name) = switch (@typeInfo(field.type)) {
                                    .int => @intFromFloat(@max(std.math.minInt(field.type), @min(std.math.maxInt(field.type), @round(accumulator)))),
                                    .float => as(field.type, accumulator),
                                    else => @compileError("Unsupported field type in struct"),
                                };
                            }

                            // Preserve alpha channel if present
                            if (has_alpha) {
                                @field(result_pixel, fields[3].name) = @field(self.at(r, c).*, fields[3].name);
                            }

                            out.at(r, c).* = result_pixel;
                        }
                    }
                },
                else => @compileError("Convolution not supported for type " ++ @typeName(T)),
            }
        }

        /// Optimized convolution for 3x3 kernels using SIMD where possible.
        fn convolve3x3(self: Self, kernel: anytype, out: Self, border_mode: BorderMode) void {
            const kr = [9]f32{
                as(f32, kernel[0][0]), as(f32, kernel[0][1]), as(f32, kernel[0][2]),
                as(f32, kernel[1][0]), as(f32, kernel[1][1]), as(f32, kernel[1][2]),
                as(f32, kernel[2][0]), as(f32, kernel[2][1]), as(f32, kernel[2][2]),
            };

            switch (@typeInfo(T)) {
                .int, .float => {
                    // Scalar types - single channel with SIMD optimization
                    const simd_len = std.simd.suggestVectorLength(f32) orelse 1;

                    // Create kernel vectors for SIMD processing
                    const kr0_vec: @Vector(simd_len, f32) = @splat(kr[0]);
                    const kr1_vec: @Vector(simd_len, f32) = @splat(kr[1]);
                    const kr2_vec: @Vector(simd_len, f32) = @splat(kr[2]);
                    const kr3_vec: @Vector(simd_len, f32) = @splat(kr[3]);
                    const kr4_vec: @Vector(simd_len, f32) = @splat(kr[4]);
                    const kr5_vec: @Vector(simd_len, f32) = @splat(kr[5]);
                    const kr6_vec: @Vector(simd_len, f32) = @splat(kr[6]);
                    const kr7_vec: @Vector(simd_len, f32) = @splat(kr[7]);
                    const kr8_vec: @Vector(simd_len, f32) = @splat(kr[8]);

                    for (0..self.rows) |r| {
                        var c: usize = 0;

                        // SIMD processing for interior pixels
                        if (r > 0 and r + 1 < self.rows and simd_len > 1 and self.cols > simd_len + 2) {
                            // Process interior pixels with SIMD (no border checks needed)
                            // We need at least simd_len + 2 columns to safely process simd_len pixels
                            // (1 border on each side plus simd_len pixels)
                            const safe_end = self.cols - 1;

                            // Skip first column (border)
                            c = 1;

                            // Only process SIMD if we have enough pixels to fill a vector
                            while (c + simd_len <= safe_end) : (c += simd_len) {
                                // Load 3x3 neighborhoods for simd_len pixels
                                var p00_vec: @Vector(simd_len, f32) = undefined;
                                var p01_vec: @Vector(simd_len, f32) = undefined;
                                var p02_vec: @Vector(simd_len, f32) = undefined;
                                var p10_vec: @Vector(simd_len, f32) = undefined;
                                var p11_vec: @Vector(simd_len, f32) = undefined;
                                var p12_vec: @Vector(simd_len, f32) = undefined;
                                var p20_vec: @Vector(simd_len, f32) = undefined;
                                var p21_vec: @Vector(simd_len, f32) = undefined;
                                var p22_vec: @Vector(simd_len, f32) = undefined;

                                for (0..simd_len) |i| {
                                    p00_vec[i] = as(f32, self.at(r - 1, c + i - 1).*);
                                    p01_vec[i] = as(f32, self.at(r - 1, c + i).*);
                                    p02_vec[i] = as(f32, self.at(r - 1, c + i + 1).*);
                                    p10_vec[i] = as(f32, self.at(r, c + i - 1).*);
                                    p11_vec[i] = as(f32, self.at(r, c + i).*);
                                    p12_vec[i] = as(f32, self.at(r, c + i + 1).*);
                                    p20_vec[i] = as(f32, self.at(r + 1, c + i - 1).*);
                                    p21_vec[i] = as(f32, self.at(r + 1, c + i).*);
                                    p22_vec[i] = as(f32, self.at(r + 1, c + i + 1).*);
                                }

                                // Perform convolution using SIMD
                                const result_vec =
                                    p00_vec * kr0_vec + p01_vec * kr1_vec + p02_vec * kr2_vec +
                                    p10_vec * kr3_vec + p11_vec * kr4_vec + p12_vec * kr5_vec +
                                    p20_vec * kr6_vec + p21_vec * kr7_vec + p22_vec * kr8_vec;

                                // Store results
                                for (0..simd_len) |i| {
                                    out.at(r, c + i).* = switch (@typeInfo(T)) {
                                        .int => @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(result_vec[i])))),
                                        .float => as(T, result_vec[i]),
                                        else => unreachable,
                                    };
                                }
                            }
                        }

                        // Process remaining pixels (borders and leftovers) with scalar code
                        while (c < self.cols) : (c += 1) {
                            var result: f32 = 0;
                            if (r > 0 and r + 1 < self.rows and c > 0 and c + 1 < self.cols) {
                                // Fast interior path
                                const p00 = self.at(r - 1, c - 1).*;
                                const p01 = self.at(r - 1, c + 0).*;
                                const p02 = self.at(r - 1, c + 1).*;
                                const p10 = self.at(r + 0, c - 1).*;
                                const p11 = self.at(r + 0, c + 0).*;
                                const p12 = self.at(r + 0, c + 1).*;
                                const p20 = self.at(r + 1, c - 1).*;
                                const p21 = self.at(r + 1, c + 0).*;
                                const p22 = self.at(r + 1, c + 1).*;

                                result =
                                    as(f32, p00) * kr[0] + as(f32, p01) * kr[1] + as(f32, p02) * kr[2] +
                                    as(f32, p10) * kr[3] + as(f32, p11) * kr[4] + as(f32, p12) * kr[5] +
                                    as(f32, p20) * kr[6] + as(f32, p21) * kr[7] + as(f32, p22) * kr[8];
                            } else {
                                const ir = @as(isize, @intCast(r));
                                const ic = @as(isize, @intCast(c));
                                const p00 = getPixelWithBorder(self, ir - 1, ic - 1, border_mode);
                                const p01 = getPixelWithBorder(self, ir - 1, ic, border_mode);
                                const p02 = getPixelWithBorder(self, ir - 1, ic + 1, border_mode);
                                const p10 = getPixelWithBorder(self, ir, ic - 1, border_mode);
                                const p11 = getPixelWithBorder(self, ir, ic, border_mode);
                                const p12 = getPixelWithBorder(self, ir, ic + 1, border_mode);
                                const p20 = getPixelWithBorder(self, ir + 1, ic - 1, border_mode);
                                const p21 = getPixelWithBorder(self, ir + 1, ic, border_mode);
                                const p22 = getPixelWithBorder(self, ir + 1, ic + 1, border_mode);

                                result =
                                    as(f32, p00) * kr[0] + as(f32, p01) * kr[1] + as(f32, p02) * kr[2] +
                                    as(f32, p10) * kr[3] + as(f32, p11) * kr[4] + as(f32, p12) * kr[5] +
                                    as(f32, p20) * kr[6] + as(f32, p21) * kr[7] + as(f32, p22) * kr[8];
                            }

                            out.at(r, c).* = switch (@typeInfo(T)) {
                                .int => @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(result)))),
                                .float => as(T, result),
                                else => unreachable,
                            };
                        }
                    }
                },
                .@"struct" => {
                    // Optimized path for u8 structs (RGB, RGBA, etc.)
                    const fields = std.meta.fields(T);
                    const all_u8 = comptime blk: {
                        for (fields) |field| {
                            if (field.type != u8) break :blk false;
                        }
                        break :blk true;
                    };

                    if (all_u8 and (fields.len == 3 or fields.len == 4)) {
                        // Determine how many channels to process (skip alpha if present)
                        const has_alpha = comptime hasAlphaChannel(T);
                        const channels_to_process = if (has_alpha) 3 else fields.len;

                        // Check if kernel is suitable for integer arithmetic
                        var kernel_sum: f32 = 0;
                        var all_positive = true;
                        for (kr) |k| {
                            kernel_sum += k;
                            if (k < 0) all_positive = false;
                        }
                        const is_gaussian_like = all_positive and @abs(kernel_sum - 1.0) < 0.1;

                        if (!is_gaussian_like) {
                            // Integer SIMD path for edge detection/sharpening kernels
                            const SCALE = 256;
                            const kr_int = [9]i32{
                                @intFromFloat(@round(kr[0] * SCALE)),
                                @intFromFloat(@round(kr[1] * SCALE)),
                                @intFromFloat(@round(kr[2] * SCALE)),
                                @intFromFloat(@round(kr[3] * SCALE)),
                                @intFromFloat(@round(kr[4] * SCALE)),
                                @intFromFloat(@round(kr[5] * SCALE)),
                                @intFromFloat(@round(kr[6] * SCALE)),
                                @intFromFloat(@round(kr[7] * SCALE)),
                                @intFromFloat(@round(kr[8] * SCALE)),
                            };

                            for (0..self.rows) |r| {
                                for (0..self.cols) |c| {
                                    if (r > 0 and r + 1 < self.rows and c > 0 and c + 1 < self.cols) {
                                        // Fast interior path
                                        const p00 = self.at(r - 1, c - 1).*;
                                        const p01 = self.at(r - 1, c + 0).*;
                                        const p02 = self.at(r - 1, c + 1).*;
                                        const p10 = self.at(r + 0, c - 1).*;
                                        const p11 = self.at(r + 0, c + 0).*;
                                        const p12 = self.at(r + 0, c + 1).*;
                                        const p20 = self.at(r + 1, c - 1).*;
                                        const p21 = self.at(r + 1, c + 0).*;
                                        const p22 = self.at(r + 1, c + 1).*;

                                        var result_pixel: T = undefined;

                                        // Process color channels
                                        inline for (0..channels_to_process) |i| {
                                            const field = fields[i];
                                            const val = @divTrunc((@as(i32, @field(p00, field.name)) * kr_int[0] +
                                                @as(i32, @field(p01, field.name)) * kr_int[1] +
                                                @as(i32, @field(p02, field.name)) * kr_int[2] +
                                                @as(i32, @field(p10, field.name)) * kr_int[3] +
                                                @as(i32, @field(p11, field.name)) * kr_int[4] +
                                                @as(i32, @field(p12, field.name)) * kr_int[5] +
                                                @as(i32, @field(p20, field.name)) * kr_int[6] +
                                                @as(i32, @field(p21, field.name)) * kr_int[7] +
                                                @as(i32, @field(p22, field.name)) * kr_int[8] + SCALE / 2), SCALE);
                                            @field(result_pixel, field.name) = @intCast(@max(0, @min(255, val)));
                                        }

                                        // Preserve alpha if present
                                        if (has_alpha) {
                                            @field(result_pixel, fields[3].name) = @field(p11, fields[3].name);
                                        }

                                        out.at(r, c).* = result_pixel;
                                    } else {
                                        // Border handling path
                                        const ir = @as(isize, @intCast(r));
                                        const ic = @as(isize, @intCast(c));
                                        const p00 = getPixelWithBorder(self, ir - 1, ic - 1, border_mode);
                                        const p01 = getPixelWithBorder(self, ir - 1, ic, border_mode);
                                        const p02 = getPixelWithBorder(self, ir - 1, ic + 1, border_mode);
                                        const p10 = getPixelWithBorder(self, ir, ic - 1, border_mode);
                                        const p11 = getPixelWithBorder(self, ir, ic, border_mode);
                                        const p12 = getPixelWithBorder(self, ir, ic + 1, border_mode);
                                        const p20 = getPixelWithBorder(self, ir + 1, ic - 1, border_mode);
                                        const p21 = getPixelWithBorder(self, ir + 1, ic, border_mode);
                                        const p22 = getPixelWithBorder(self, ir + 1, ic + 1, border_mode);

                                        var result_pixel: T = undefined;

                                        // Process color channels
                                        inline for (0..channels_to_process) |i| {
                                            const field = fields[i];
                                            const val = @divTrunc((@as(i32, @field(p00, field.name)) * kr_int[0] +
                                                @as(i32, @field(p01, field.name)) * kr_int[1] +
                                                @as(i32, @field(p02, field.name)) * kr_int[2] +
                                                @as(i32, @field(p10, field.name)) * kr_int[3] +
                                                @as(i32, @field(p11, field.name)) * kr_int[4] +
                                                @as(i32, @field(p12, field.name)) * kr_int[5] +
                                                @as(i32, @field(p20, field.name)) * kr_int[6] +
                                                @as(i32, @field(p21, field.name)) * kr_int[7] +
                                                @as(i32, @field(p22, field.name)) * kr_int[8] + SCALE / 2), SCALE);
                                            @field(result_pixel, field.name) = @intCast(@max(0, @min(255, val)));
                                        }

                                        // Preserve alpha if present
                                        if (has_alpha) {
                                            @field(result_pixel, fields[3].name) = @field(p11, fields[3].name);
                                        }

                                        out.at(r, c).* = result_pixel;
                                    }
                                }
                            }
                        } else {
                            // Float-based path for Gaussian-like kernels
                            for (0..self.rows) |r| {
                                for (0..self.cols) |c| {
                                    if (r > 0 and r + 1 < self.rows and c > 0 and c + 1 < self.cols) {
                                        // Fast interior path
                                        const p00 = self.at(r - 1, c - 1).*;
                                        const p01 = self.at(r - 1, c + 0).*;
                                        const p02 = self.at(r - 1, c + 1).*;
                                        const p10 = self.at(r + 0, c - 1).*;
                                        const p11 = self.at(r + 0, c + 0).*;
                                        const p12 = self.at(r + 0, c + 1).*;
                                        const p20 = self.at(r + 1, c - 1).*;
                                        const p21 = self.at(r + 1, c + 0).*;
                                        const p22 = self.at(r + 1, c + 1).*;

                                        var result_pixel: T = undefined;

                                        // Process color channels
                                        inline for (0..channels_to_process) |i| {
                                            const field = fields[i];
                                            const result =
                                                as(f32, @field(p00, field.name)) * kr[0] + as(f32, @field(p01, field.name)) * kr[1] + as(f32, @field(p02, field.name)) * kr[2] +
                                                as(f32, @field(p10, field.name)) * kr[3] + as(f32, @field(p11, field.name)) * kr[4] + as(f32, @field(p12, field.name)) * kr[5] +
                                                as(f32, @field(p20, field.name)) * kr[6] + as(f32, @field(p21, field.name)) * kr[7] + as(f32, @field(p22, field.name)) * kr[8];
                                            @field(result_pixel, field.name) = @intFromFloat(@max(0, @min(255, @round(result))));
                                        }

                                        // Preserve alpha if present
                                        if (has_alpha) {
                                            @field(result_pixel, fields[3].name) = @field(p11, fields[3].name);
                                        }

                                        out.at(r, c).* = result_pixel;
                                    } else {
                                        // Border handling path
                                        const ir = @as(isize, @intCast(r));
                                        const ic = @as(isize, @intCast(c));
                                        const p00 = getPixelWithBorder(self, ir - 1, ic - 1, border_mode);
                                        const p01 = getPixelWithBorder(self, ir - 1, ic, border_mode);
                                        const p02 = getPixelWithBorder(self, ir - 1, ic + 1, border_mode);
                                        const p10 = getPixelWithBorder(self, ir, ic - 1, border_mode);
                                        const p11 = getPixelWithBorder(self, ir, ic, border_mode);
                                        const p12 = getPixelWithBorder(self, ir, ic + 1, border_mode);
                                        const p20 = getPixelWithBorder(self, ir + 1, ic - 1, border_mode);
                                        const p21 = getPixelWithBorder(self, ir + 1, ic, border_mode);
                                        const p22 = getPixelWithBorder(self, ir + 1, ic + 1, border_mode);

                                        var result_pixel: T = undefined;

                                        // Process color channels
                                        inline for (0..channels_to_process) |i| {
                                            const field = fields[i];
                                            const result =
                                                as(f32, @field(p00, field.name)) * kr[0] + as(f32, @field(p01, field.name)) * kr[1] + as(f32, @field(p02, field.name)) * kr[2] +
                                                as(f32, @field(p10, field.name)) * kr[3] + as(f32, @field(p11, field.name)) * kr[4] + as(f32, @field(p12, field.name)) * kr[5] +
                                                as(f32, @field(p20, field.name)) * kr[6] + as(f32, @field(p21, field.name)) * kr[7] + as(f32, @field(p22, field.name)) * kr[8];
                                            @field(result_pixel, field.name) = @intFromFloat(@max(0, @min(255, @round(result))));
                                        }

                                        // Preserve alpha if present
                                        if (has_alpha) {
                                            @field(result_pixel, fields[3].name) = @field(p11, fields[3].name);
                                        }

                                        out.at(r, c).* = result_pixel;
                                    }
                                }
                            }
                        }
                    } else {
                        // Generic struct path for other color types
                        for (0..self.rows) |r| {
                            for (0..self.cols) |c| {
                                var result_pixel: T = undefined;
                                if (r > 0 and r + 1 < self.rows and c > 0 and c + 1 < self.cols) {
                                    const p00 = self.at(r - 1, c - 1).*;
                                    const p01 = self.at(r - 1, c + 0).*;
                                    const p02 = self.at(r - 1, c + 1).*;
                                    const p10 = self.at(r + 0, c - 1).*;
                                    const p11 = self.at(r + 0, c + 0).*;
                                    const p12 = self.at(r + 0, c + 1).*;
                                    const p20 = self.at(r + 1, c - 1).*;
                                    const p21 = self.at(r + 1, c + 0).*;
                                    const p22 = self.at(r + 1, c + 1).*;

                                    inline for (std.meta.fields(T)) |field| {
                                        const result =
                                            as(f32, @field(p00, field.name)) * kr[0] + as(f32, @field(p01, field.name)) * kr[1] + as(f32, @field(p02, field.name)) * kr[2] +
                                            as(f32, @field(p10, field.name)) * kr[3] + as(f32, @field(p11, field.name)) * kr[4] + as(f32, @field(p12, field.name)) * kr[5] +
                                            as(f32, @field(p20, field.name)) * kr[6] + as(f32, @field(p21, field.name)) * kr[7] + as(f32, @field(p22, field.name)) * kr[8];
                                        @field(result_pixel, field.name) = switch (@typeInfo(field.type)) {
                                            .int => @intFromFloat(@max(std.math.minInt(field.type), @min(std.math.maxInt(field.type), @round(result)))),
                                            .float => as(field.type, result),
                                            else => @compileError("Unsupported field type"),
                                        };
                                    }
                                } else {
                                    const ir = @as(isize, @intCast(r));
                                    const ic = @as(isize, @intCast(c));
                                    const p00 = getPixelWithBorder(self, ir - 1, ic - 1, border_mode);
                                    const p01 = getPixelWithBorder(self, ir - 1, ic, border_mode);
                                    const p02 = getPixelWithBorder(self, ir - 1, ic + 1, border_mode);
                                    const p10 = getPixelWithBorder(self, ir, ic - 1, border_mode);
                                    const p11 = getPixelWithBorder(self, ir, ic, border_mode);
                                    const p12 = getPixelWithBorder(self, ir, ic + 1, border_mode);
                                    const p20 = getPixelWithBorder(self, ir + 1, ic - 1, border_mode);
                                    const p21 = getPixelWithBorder(self, ir + 1, ic, border_mode);
                                    const p22 = getPixelWithBorder(self, ir + 1, ic + 1, border_mode);

                                    inline for (std.meta.fields(T)) |field| {
                                        const result =
                                            as(f32, @field(p00, field.name)) * kr[0] + as(f32, @field(p01, field.name)) * kr[1] + as(f32, @field(p02, field.name)) * kr[2] +
                                            as(f32, @field(p10, field.name)) * kr[3] + as(f32, @field(p11, field.name)) * kr[4] + as(f32, @field(p12, field.name)) * kr[5] +
                                            as(f32, @field(p20, field.name)) * kr[6] + as(f32, @field(p21, field.name)) * kr[7] + as(f32, @field(p22, field.name)) * kr[8];
                                        @field(result_pixel, field.name) = switch (@typeInfo(field.type)) {
                                            .int => @intFromFloat(@max(std.math.minInt(field.type), @min(std.math.maxInt(field.type), @round(result)))),
                                            .float => as(field.type, result),
                                            else => @compileError("Unsupported field type"),
                                        };
                                    }
                                }

                                out.at(r, c).* = result_pixel;
                            }
                        }
                    }
                },
                else => unreachable,
            }
        }

        /// Get pixel value with border handling.
        inline fn getPixelWithBorder(self: Self, row: isize, col: isize, border_mode: BorderMode) T {
            const irows = @as(isize, @intCast(self.rows));
            const icols = @as(isize, @intCast(self.cols));

            switch (border_mode) {
                .zero => {
                    if (row < 0 or col < 0 or row >= irows or col >= icols) {
                        return std.mem.zeroes(T);
                    }
                    return self.at(@intCast(row), @intCast(col)).*;
                },
                .replicate => {
                    const r = @max(0, @min(row, irows - 1));
                    const c = @max(0, @min(col, icols - 1));
                    return self.at(@intCast(r), @intCast(c)).*;
                },
                .mirror => {
                    // Reflect indices across borders with period 2*N
                    if (irows == 0 or icols == 0) return std.mem.zeroes(T);
                    var r = @mod(row, 2 * irows);
                    var c = @mod(col, 2 * icols);
                    if (r >= irows) r = 2 * irows - 1 - r;
                    if (c >= icols) c = 2 * icols - 1 - c;
                    return self.at(@intCast(r), @intCast(c)).*;
                },
                .wrap => {
                    const r = @mod(row, irows);
                    const c = @mod(col, icols);
                    return self.at(@intCast(r), @intCast(c)).*;
                },
            }
        }

        /// Performs separable convolution using two 1D kernels (horizontal and vertical).
        /// This is much more efficient for separable filters like Gaussian blur.
        ///
        /// Parameters:
        /// - `allocator`: The allocator to use for temporary buffers.
        /// - `kernel_x`: Horizontal (column) kernel.
        /// - `kernel_y`: Vertical (row) kernel.
        /// - `out`: Output image.
        /// - `border_mode`: How to handle image borders.
        pub fn convolveSeparable(self: Self, allocator: Allocator, kernel_x: []const f32, kernel_y: []const f32, out: *Self, border_mode: BorderMode) !void {
            if (!self.hasSameShape(out.*)) {
                out.* = try .initAlloc(allocator, self.rows, self.cols);
            }

            // Allocate temporary buffer for intermediate result
            var temp = try Self.initAlloc(allocator, self.rows, self.cols);
            defer temp.deinit(allocator);

            const half_x = kernel_x.len / 2;
            const half_y = kernel_y.len / 2;

            // Horizontal pass
            switch (@typeInfo(T)) {
                .int, .float => {
                    for (0..self.rows) |r| {
                        for (0..self.cols) |c| {
                            var sum: f32 = 0;
                            if (c >= half_x and c + half_x < self.cols) {
                                const c0: usize = c - half_x;
                                for (kernel_x, 0..) |k, i| {
                                    const cc = c0 + i;
                                    const pixel = self.at(r, cc).*;
                                    sum += as(f32, pixel) * k;
                                }
                            } else {
                                for (kernel_x, 0..) |k, i| {
                                    const src_c = @as(isize, @intCast(c)) + @as(isize, @intCast(i)) - @as(isize, @intCast(half_x));
                                    const pixel = getPixelWithBorder(self, @intCast(r), src_c, border_mode);
                                    sum += as(f32, pixel) * k;
                                }
                            }
                            temp.at(r, c).* = switch (@typeInfo(T)) {
                                .int => @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(sum)))),
                                .float => as(T, sum),
                                else => unreachable,
                            };
                        }
                    }
                },
                .@"struct" => {
                    for (0..self.rows) |r| {
                        for (0..self.cols) |c| {
                            var result_pixel: T = undefined;
                            inline for (std.meta.fields(T)) |field| {
                                var sum: f32 = 0;
                                if (c >= half_x and c + half_x < self.cols) {
                                    const c0: usize = c - half_x;
                                    for (kernel_x, 0..) |k, i| {
                                        const cc = c0 + i;
                                        const pixel = self.at(r, cc).*;
                                        sum += as(f32, @field(pixel, field.name)) * k;
                                    }
                                } else {
                                    for (kernel_x, 0..) |k, i| {
                                        const src_c = @as(isize, @intCast(c)) + @as(isize, @intCast(i)) - @as(isize, @intCast(half_x));
                                        const pixel = getPixelWithBorder(self, @intCast(r), src_c, border_mode);
                                        sum += as(f32, @field(pixel, field.name)) * k;
                                    }
                                }
                                @field(result_pixel, field.name) = switch (@typeInfo(field.type)) {
                                    .int => @intFromFloat(@max(std.math.minInt(field.type), @min(std.math.maxInt(field.type), @round(sum)))),
                                    .float => as(field.type, sum),
                                    else => @compileError("Unsupported field type"),
                                };
                            }
                            temp.at(r, c).* = result_pixel;
                        }
                    }
                },
                else => @compileError("Separable convolution not supported for type " ++ @typeName(T)),
            }

            // Vertical pass
            switch (@typeInfo(T)) {
                .int, .float => {
                    for (0..self.rows) |r| {
                        for (0..self.cols) |c| {
                            var sum: f32 = 0;
                            if (r >= half_y and r + half_y < self.rows) {
                                const r0: usize = r - half_y;
                                for (kernel_y, 0..) |k, i| {
                                    const rr = r0 + i;
                                    const pixel = temp.at(rr, c).*;
                                    sum += as(f32, pixel) * k;
                                }
                            } else {
                                for (kernel_y, 0..) |k, i| {
                                    const src_r = @as(isize, @intCast(r)) + @as(isize, @intCast(i)) - @as(isize, @intCast(half_y));
                                    const pixel = getPixelWithBorder(temp, src_r, @intCast(c), border_mode);
                                    sum += as(f32, pixel) * k;
                                }
                            }
                            out.at(r, c).* = switch (@typeInfo(T)) {
                                .int => @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(sum)))),
                                .float => as(T, sum),
                                else => unreachable,
                            };
                        }
                    }
                },
                .@"struct" => {
                    for (0..self.rows) |r| {
                        for (0..self.cols) |c| {
                            var result_pixel: T = undefined;
                            inline for (std.meta.fields(T)) |field| {
                                var sum: f32 = 0;
                                if (r >= half_y and r + half_y < self.rows) {
                                    const r0: usize = r - half_y;
                                    for (kernel_y, 0..) |k, i| {
                                        const rr = r0 + i;
                                        const pixel = temp.at(rr, c).*;
                                        sum += as(f32, @field(pixel, field.name)) * k;
                                    }
                                } else {
                                    for (kernel_y, 0..) |k, i| {
                                        const src_r = @as(isize, @intCast(r)) + @as(isize, @intCast(i)) - @as(isize, @intCast(half_y));
                                        const pixel = getPixelWithBorder(temp, src_r, @intCast(c), border_mode);
                                        sum += as(f32, @field(pixel, field.name)) * k;
                                    }
                                }
                                @field(result_pixel, field.name) = switch (@typeInfo(field.type)) {
                                    .int => @intFromFloat(@max(std.math.minInt(field.type), @min(std.math.maxInt(field.type), @round(sum)))),
                                    .float => as(field.type, sum),
                                    else => @compileError("Unsupported field type"),
                                };
                            }
                            out.at(r, c).* = result_pixel;
                        }
                    }
                },
                else => unreachable,
            }
        }

        /// Applies Gaussian blur to the image using separable convolution.
        ///
        /// Parameters:
        /// - `allocator`: The allocator to use for temporary buffers.
        /// - `sigma`: Standard deviation of the Gaussian kernel.
        /// - `out`: Output blurred image.
        pub fn blurGaussian(self: Self, allocator: Allocator, sigma: f32, out: *Self) !void {
            if (sigma <= 0) return error.InvalidSigma;

            // Calculate kernel size (3 sigma on each side)
            const radius = @as(usize, @intFromFloat(@ceil(3.0 * sigma)));
            const kernel_size = 2 * radius + 1;

            // Generate 1D Gaussian kernel
            var kernel = try allocator.alloc(f32, kernel_size);
            defer allocator.free(kernel);

            var sum: f32 = 0;
            for (0..kernel_size) |i| {
                const x = @as(f32, @floatFromInt(i)) - @as(f32, @floatFromInt(radius));
                kernel[i] = @exp(-(x * x) / (2.0 * sigma * sigma));
                sum += kernel[i];
            }

            // Normalize kernel
            for (kernel) |*k| {
                k.* /= sum;
            }

            // Apply separable convolution
            try convolveSeparable(self, allocator, kernel, kernel, out, .mirror);
        }

        /// Applies Difference of Gaussians (DoG) band-pass filter to the image.
        /// This efficiently computes the difference between two Gaussian blurs with different sigmas,
        /// which acts as a band-pass filter and is commonly used for edge detection and feature enhancement.
        ///
        /// Parameters:
        /// - `allocator`: The allocator to use for temporary buffers.
        /// - `sigma1`: Standard deviation of the first (typically smaller) Gaussian kernel.
        /// - `sigma2`: Standard deviation of the second (typically larger) Gaussian kernel.
        /// - `out`: Output image containing the difference.
        ///
        /// The result is computed as: gaussian_blur(sigma1) - gaussian_blur(sigma2)
        /// For edge detection, typically sigma2 ≈ 1.6 * sigma1
        pub fn differenceOfGaussians(self: Self, allocator: Allocator, sigma1: f32, sigma2: f32, out: *Self) !void {
            if (sigma1 <= 0 or sigma2 <= 0) return error.InvalidSigma;
            if (sigma1 == sigma2) return error.SigmasMustDiffer;

            // Ensure output is allocated
            if (!self.hasSameShape(out.*)) {
                out.* = try .initAlloc(allocator, self.rows, self.cols);
            }

            // Calculate kernel sizes for both sigmas
            const radius1 = @as(usize, @intFromFloat(@ceil(3.0 * sigma1)));
            const kernel_size1 = 2 * radius1 + 1;
            const radius2 = @as(usize, @intFromFloat(@ceil(3.0 * sigma2)));
            const kernel_size2 = 2 * radius2 + 1;

            // Generate both 1D Gaussian kernels
            var kernel1 = try allocator.alloc(f32, kernel_size1);
            defer allocator.free(kernel1);
            var kernel2 = try allocator.alloc(f32, kernel_size2);
            defer allocator.free(kernel2);

            // Generate first kernel
            var sum1: f32 = 0;
            for (0..kernel_size1) |i| {
                const x = @as(f32, @floatFromInt(i)) - @as(f32, @floatFromInt(radius1));
                kernel1[i] = @exp(-(x * x) / (2.0 * sigma1 * sigma1));
                sum1 += kernel1[i];
            }
            for (kernel1) |*k| {
                k.* /= sum1;
            }

            // Generate second kernel
            var sum2: f32 = 0;
            for (0..kernel_size2) |i| {
                const x = @as(f32, @floatFromInt(i)) - @as(f32, @floatFromInt(radius2));
                kernel2[i] = @exp(-(x * x) / (2.0 * sigma2 * sigma2));
                sum2 += kernel2[i];
            }
            for (kernel2) |*k| {
                k.* /= sum2;
            }

            // Allocate temporary buffers for the two blurred results
            var blur1 = try Self.initAlloc(allocator, self.rows, self.cols);
            defer blur1.deinit(allocator);
            var blur2 = try Self.initAlloc(allocator, self.rows, self.cols);
            defer blur2.deinit(allocator);

            // Apply both Gaussian blurs using separable convolution
            try convolveSeparable(self, allocator, kernel1, kernel1, &blur1, .mirror);
            try convolveSeparable(self, allocator, kernel2, kernel2, &blur2, .mirror);

            // Compute the difference: blur1 - blur2
            switch (@typeInfo(T)) {
                .int => {
                    // For integer types, handle underflow/overflow carefully
                    for (0..self.rows) |r| {
                        for (0..self.cols) |c| {
                            const val1 = as(f32, blur1.at(r, c).*);
                            const val2 = as(f32, blur2.at(r, c).*);
                            const diff = val1 - val2;

                            // For visualization, you might want to add an offset and scale
                            // For now, we'll clamp to the valid range
                            out.at(r, c).* = @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(diff))));
                        }
                    }
                },
                .float => {
                    // For float types, direct subtraction
                    for (0..self.rows) |r| {
                        for (0..self.cols) |c| {
                            out.at(r, c).* = blur1.at(r, c).* - blur2.at(r, c).*;
                        }
                    }
                },
                .@"struct" => {
                    // For struct types (RGB, RGBA, etc.), process each channel
                    for (0..self.rows) |r| {
                        for (0..self.cols) |c| {
                            var result_pixel: T = undefined;
                            const pixel1 = blur1.at(r, c).*;
                            const pixel2 = blur2.at(r, c).*;

                            inline for (std.meta.fields(T)) |field| {
                                const val1 = as(f32, @field(pixel1, field.name));
                                const val2 = as(f32, @field(pixel2, field.name));
                                const diff = val1 - val2;

                                @field(result_pixel, field.name) = switch (@typeInfo(field.type)) {
                                    .int => @intFromFloat(@max(std.math.minInt(field.type), @min(std.math.maxInt(field.type), @round(diff)))),
                                    .float => as(field.type, diff),
                                    else => @compileError("Unsupported field type in struct"),
                                };
                            }
                            out.at(r, c).* = result_pixel;
                        }
                    }
                },
                else => @compileError("Difference of Gaussians not supported for type " ++ @typeName(T)),
            }
        }

        /// Applies the Sobel filter to `self` to perform edge detection.
        /// The output is a grayscale image representing the magnitude of gradients at each pixel.
        ///
        /// Parameters:
        /// - `allocator`: The allocator to use if `out` needs to be (re)initialized.
        /// - `out`: An out-parameter pointer to an `Image(u8)` that will be filled with the Sobel magnitude image.
        pub fn sobel(self: Self, allocator: Allocator, out: *Image(u8)) !void {
            if (!self.hasSameShape(out.*)) {
                out.* = try .initAlloc(allocator, self.rows, self.cols);
            }

            // Sobel kernels for edge detection
            const sobel_x = [3][3]f32{
                .{ -1, 0, 1 },
                .{ -2, 0, 2 },
                .{ -1, 0, 1 },
            };
            const sobel_y = [3][3]f32{
                .{ -1, -2, -1 },
                .{ 0, 0, 0 },
                .{ 1, 2, 1 },
            };

            // Convert input to grayscale float if needed
            var gray_float: Image(f32) = undefined;
            const needs_conversion = !isScalar(T) or @typeInfo(T) != .float;
            if (needs_conversion) {
                gray_float = try .initAlloc(allocator, self.rows, self.cols);
                for (0..self.rows) |r| {
                    for (0..self.cols) |c| {
                        gray_float.at(r, c).* = as(f32, convertColor(u8, self.at(r, c).*));
                    }
                }
            } else {
                gray_float = self;
            }
            defer if (needs_conversion) gray_float.deinit(allocator);

            // Apply Sobel X and Y filters
            var grad_x = Image(f32).empty;
            var grad_y = Image(f32).empty;
            defer grad_x.deinit(allocator);
            defer grad_y.deinit(allocator);

            const GrayFilter = Filter(f32);
            try GrayFilter.convolve(gray_float, allocator, sobel_x, &grad_x, .zero);
            try GrayFilter.convolve(gray_float, allocator, sobel_y, &grad_y, .zero);

            // Compute gradient magnitude
            for (0..self.rows) |r| {
                for (0..self.cols) |c| {
                    const gx = grad_x.at(r, c).*;
                    const gy = grad_y.at(r, c).*;
                    const magnitude = @sqrt(gx * gx + gy * gy);
                    out.at(r, c).* = @intFromFloat(@max(0, @min(255, magnitude)));
                }
            }
        }
    };
}
