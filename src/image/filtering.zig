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

        /// Separate RGB channels from a struct image into individual planes.
        /// Allocates and fills 3 channel planes (r, g, b).
        fn separateRGBChannels(self: Self, allocator: std.mem.Allocator) ![3][]u8 {
            const fields = std.meta.fields(T);
            const plane_size = self.rows * self.cols;

            const r_channel = try allocator.alloc(u8, plane_size);
            errdefer allocator.free(r_channel);
            const g_channel = try allocator.alloc(u8, plane_size);
            errdefer allocator.free(g_channel);
            const b_channel = try allocator.alloc(u8, plane_size);
            errdefer allocator.free(b_channel);

            // Separate channels (single pass for cache efficiency)
            var idx: usize = 0;
            for (0..self.rows) |r| {
                for (0..self.cols) |c| {
                    const pixel = self.at(r, c).*;
                    r_channel[idx] = @field(pixel, fields[0].name);
                    g_channel[idx] = @field(pixel, fields[1].name);
                    b_channel[idx] = @field(pixel, fields[2].name);
                    idx += 1;
                }
            }

            return .{ r_channel, g_channel, b_channel };
        }

        /// Combine RGB channels back into struct image, optionally preserving alpha from original.
        fn combineRGBChannels(self: Self, r_out: []const u8, g_out: []const u8, b_out: []const u8, out: Self) void {
            const fields = std.meta.fields(T);
            const has_alpha = comptime hasAlphaChannel(T);

            // Recombine channels, preserving alpha
            var idx: usize = 0;
            for (0..self.rows) |r| {
                for (0..self.cols) |c| {
                    var result_pixel: T = undefined;
                    @field(result_pixel, fields[0].name) = r_out[idx];
                    @field(result_pixel, fields[1].name) = g_out[idx];
                    @field(result_pixel, fields[2].name) = b_out[idx];

                    // Preserve alpha if present
                    if (has_alpha) {
                        @field(result_pixel, fields[3].name) = @field(self.at(r, c).*, fields[3].name);
                    } else if (fields.len == 4) {
                        // For non-alpha 4th channel, also preserve it
                        @field(result_pixel, fields[3].name) = @field(self.at(r, c).*, fields[3].name);
                    }

                    out.at(r, c).* = result_pixel;
                    idx += 1;
                }
            }
        }

        /// Build a 4-channel integral image for structs with 4 u8 fields.
        /// Only processes the first `channels_to_process` channels (3 for RGBA to skip alpha, 4 for others).
        fn build4ChannelIntegralImage(self: Self, sat: Image([4]f32), comptime channels_to_process: usize) void {
            const fields = std.meta.fields(T);

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

            // Build the integral image using the helper function
            if (has_alpha) {
                build4ChannelIntegralImage(self, sat, 3);
            } else {
                build4ChannelIntegralImage(self, sat, 4);
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

            // Build the integral image using the helper function
            if (has_alpha) {
                build4ChannelIntegralImage(self, sat, 3);
            } else {
                build4ChannelIntegralImage(self, sat, 4);
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

        /// Comptime function generator for specialized convolution implementations.
        /// Generates optimized code for specific kernel dimensions at compile time.
        fn ConvolveKernel(comptime height: usize, comptime width: usize) type {
            return struct {
                const kernel_size = height * width;
                const half_h = height / 2;
                const half_w = width / 2;

                /// Optimized convolution for u8 planes with integer arithmetic.
                fn convolveU8Plane(
                    src: []const u8,
                    dst: []u8,
                    rows: usize,
                    cols: usize,
                    kernel: [kernel_size]i32,
                    border_mode: BorderMode,
                ) void {
                    const SCALE = 256;
                    const vec_len = comptime std.simd.suggestVectorLength(i32) orelse 8;

                    for (0..rows) |r| {
                        var c: usize = 0;

                        // SIMD path for interior pixels (only for reasonably small kernels)
                        if ((comptime (height <= 7 and width <= 7)) and r >= half_h and r + half_h < rows and cols > vec_len + width) {
                            c = half_w;
                            const safe_end = if (cols > vec_len + half_w) cols - vec_len - half_w else half_w;

                            while (c + vec_len <= safe_end) : (c += vec_len) {
                                var result_vec: @Vector(vec_len, i32) = @splat(0);

                                // Unroll kernel application for known sizes
                                inline for (0..height) |ky| {
                                    inline for (0..width) |kx| {
                                        const kernel_val = kernel[ky * width + kx];
                                        const kernel_vec: @Vector(vec_len, i32) = @splat(kernel_val);

                                        var pixel_vec: @Vector(vec_len, i32) = undefined;
                                        for (0..vec_len) |i| {
                                            const src_r = r + ky - half_h;
                                            const src_c = c + i + kx - half_w;
                                            pixel_vec[i] = src[src_r * cols + src_c];
                                        }

                                        result_vec += pixel_vec * kernel_vec;
                                    }
                                }

                                // Scale and store
                                const half_scale_vec: @Vector(vec_len, i32) = @splat(SCALE / 2);
                                const scale_vec: @Vector(vec_len, i32) = @splat(SCALE);
                                const rounded_vec = @divTrunc(result_vec + half_scale_vec, scale_vec);

                                for (0..vec_len) |i| {
                                    dst[r * cols + c + i] = @intCast(@max(0, @min(255, rounded_vec[i])));
                                }
                            }
                        }

                        // Scalar path for remaining pixels
                        while (c < cols) : (c += 1) {
                            if (r >= half_h and r + half_h < rows and c >= half_w and c + half_w < cols) {
                                // Fast path without border checks
                                var result: i32 = 0;
                                inline for (0..height) |ky| {
                                    inline for (0..width) |kx| {
                                        const src_r = r + ky - half_h;
                                        const src_c = c + kx - half_w;
                                        const pixel_val = @as(i32, src[src_r * cols + src_c]);
                                        result += pixel_val * kernel[ky * width + kx];
                                    }
                                }
                                const rounded = @divTrunc(result + SCALE / 2, SCALE);
                                dst[r * cols + c] = @intCast(@max(0, @min(255, rounded)));
                            } else {
                                // Border handling
                                const ir = @as(isize, @intCast(r));
                                const ic = @as(isize, @intCast(c));
                                var result: i32 = 0;
                                inline for (0..height) |ky| {
                                    inline for (0..width) |kx| {
                                        const iry = ir + @as(isize, @intCast(ky)) - @as(isize, @intCast(half_h));
                                        const icx = ic + @as(isize, @intCast(kx)) - @as(isize, @intCast(half_w));
                                        const pixel_val = if (iry >= 0 and iry < rows and icx >= 0 and icx < cols)
                                            @as(i32, src[@as(usize, @intCast(iry * @as(isize, @intCast(cols)) + icx))])
                                        else switch (border_mode) {
                                            .zero => 0,
                                            .replicate => blk: {
                                                const clamped_r = @max(0, @min(@as(isize, @intCast(rows - 1)), iry));
                                                const clamped_c = @max(0, @min(@as(isize, @intCast(cols - 1)), icx));
                                                break :blk @as(i32, src[@as(usize, @intCast(clamped_r * @as(isize, @intCast(cols)) + clamped_c))]);
                                            },
                                            .mirror => blk: {
                                                var rr = iry;
                                                var cc = icx;
                                                if (rr < 0) rr = -rr - 1;
                                                if (rr >= rows) rr = 2 * @as(isize, @intCast(rows)) - rr - 1;
                                                if (cc < 0) cc = -cc - 1;
                                                if (cc >= cols) cc = 2 * @as(isize, @intCast(cols)) - cc - 1;
                                                break :blk @as(i32, src[@as(usize, @intCast(rr * @as(isize, @intCast(cols)) + cc))]);
                                            },
                                            .wrap => blk: {
                                                const wrapped_r = @mod(iry, @as(isize, @intCast(rows)));
                                                const wrapped_c = @mod(icx, @as(isize, @intCast(cols)));
                                                break :blk @as(i32, src[@as(usize, @intCast(wrapped_r * @as(isize, @intCast(cols)) + wrapped_c))]);
                                            },
                                        };
                                        result += pixel_val * kernel[ky * width + kx];
                                    }
                                }
                                const rounded = @divTrunc(result + SCALE / 2, SCALE);
                                dst[r * cols + c] = @intCast(@max(0, @min(255, rounded)));
                            }
                        }
                    }
                }

                /// Optimized convolution for f32 planes with SIMD.
                fn convolveF32Plane(
                    src: []const f32,
                    dst: []f32,
                    rows: usize,
                    cols: usize,
                    kernel: [kernel_size]f32,
                    border_mode: BorderMode,
                ) void {
                    const vec_len = comptime std.simd.suggestVectorLength(f32) orelse 8;

                    // Pre-create kernel vectors for SIMD (only for small kernels)
                    const use_kernel_vecs = comptime (kernel_size <= 25);
                    var kernel_vecs: if (use_kernel_vecs) [kernel_size]@Vector(vec_len, f32) else void = undefined;

                    // Initialize kernel vectors if small enough
                    if (use_kernel_vecs) {
                        inline for (0..kernel_size) |i| {
                            kernel_vecs[i] = @splat(kernel[i]);
                        }
                    }

                    for (0..rows) |r| {
                        var c: usize = 0;

                        // SIMD path for interior pixels
                        if ((comptime (height <= 7 and width <= 7)) and r >= half_h and r + half_h < rows and cols > vec_len + width) {
                            c = half_w;
                            const safe_end = if (cols > vec_len + half_w) cols - vec_len - half_w else half_w;

                            while (c + vec_len <= safe_end) : (c += vec_len) {
                                var result_vec: @Vector(vec_len, f32) = @splat(0);

                                inline for (0..height) |ky| {
                                    inline for (0..width) |kx| {
                                        const kid = ky * width + kx;
                                        const kernel_vec = if (use_kernel_vecs)
                                            kernel_vecs[kid]
                                        else
                                            @as(@Vector(vec_len, f32), @splat(kernel[kid]));

                                        var pixel_vec: @Vector(vec_len, f32) = undefined;
                                        for (0..vec_len) |i| {
                                            const src_r = r + ky - half_h;
                                            const src_c = c + i + kx - half_w;
                                            pixel_vec[i] = src[src_r * cols + src_c];
                                        }

                                        result_vec += pixel_vec * kernel_vec;
                                    }
                                }

                                for (0..vec_len) |i| {
                                    dst[r * cols + c + i] = result_vec[i];
                                }
                            }
                        }

                        // Scalar path
                        while (c < cols) : (c += 1) {
                            if (r >= half_h and r + half_h < rows and c >= half_w and c + half_w < cols) {
                                var result: f32 = 0;
                                inline for (0..height) |ky| {
                                    inline for (0..width) |kx| {
                                        const src_r = r + ky - half_h;
                                        const src_c = c + kx - half_w;
                                        result += src[src_r * cols + src_c] * kernel[ky * width + kx];
                                    }
                                }
                                dst[r * cols + c] = result;
                            } else {
                                // Border handling
                                const ir = @as(isize, @intCast(r));
                                const ic = @as(isize, @intCast(c));
                                var result: f32 = 0;
                                inline for (0..height) |ky| {
                                    inline for (0..width) |kx| {
                                        const iry = ir + @as(isize, @intCast(ky)) - @as(isize, @intCast(half_h));
                                        const icx = ic + @as(isize, @intCast(kx)) - @as(isize, @intCast(half_w));
                                        const pixel_val = if (iry >= 0 and iry < rows and icx >= 0 and icx < cols)
                                            src[@as(usize, @intCast(iry * @as(isize, @intCast(cols)) + icx))]
                                        else switch (border_mode) {
                                            .zero => 0,
                                            .replicate => blk: {
                                                const clamped_r = @max(0, @min(@as(isize, @intCast(rows - 1)), iry));
                                                const clamped_c = @max(0, @min(@as(isize, @intCast(cols - 1)), icx));
                                                break :blk src[@as(usize, @intCast(clamped_r * @as(isize, @intCast(cols)) + clamped_c))];
                                            },
                                            .mirror => blk: {
                                                var rr = iry;
                                                var cc = icx;
                                                if (rr < 0) rr = -rr - 1;
                                                if (rr >= rows) rr = 2 * @as(isize, @intCast(rows)) - rr - 1;
                                                if (cc < 0) cc = -cc - 1;
                                                if (cc >= cols) cc = 2 * @as(isize, @intCast(cols)) - cc - 1;
                                                break :blk src[@as(usize, @intCast(rr * @as(isize, @intCast(cols)) + cc))];
                                            },
                                            .wrap => blk: {
                                                const wrapped_r = @mod(iry, @as(isize, @intCast(rows)));
                                                const wrapped_c = @mod(icx, @as(isize, @intCast(cols)));
                                                break :blk src[@as(usize, @intCast(wrapped_r * @as(isize, @intCast(cols)) + wrapped_c))];
                                            },
                                        };
                                        result += pixel_val * kernel[ky * width + kx];
                                    }
                                }
                                dst[r * cols + c] = result;
                            }
                        }
                    }
                }
            };
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

            // Generate specialized implementation for this kernel size
            const Kernel = ConvolveKernel(kernel_height, kernel_width);

            switch (@typeInfo(T)) {
                .int, .float => {
                    // Optimized path for u8 with integer arithmetic
                    if (T == u8) {
                        // Convert floating-point kernel to integer
                        const SCALE = 256;
                        var kernel_int: [Kernel.kernel_size]i32 = undefined;

                        // Flatten and scale the kernel
                        var idx: usize = 0;
                        inline for (0..kernel_height) |kr| {
                            inline for (0..kernel_width) |kc| {
                                kernel_int[idx] = @intFromFloat(@round(as(f32, kernel[kr][kc]) * SCALE));
                                idx += 1;
                            }
                        }

                        // Get data as slices
                        const src_data = self.data[0 .. self.rows * self.cols];
                        const dst_data = out.data[0 .. out.rows * out.cols];

                        Kernel.convolveU8Plane(src_data, dst_data, self.rows, self.cols, kernel_int, border_mode);
                    } else if (T == f32) {
                        // Optimized path for f32 with SIMD
                        var kernel_flat: [Kernel.kernel_size]f32 = undefined;

                        // Flatten the kernel
                        var idx: usize = 0;
                        inline for (0..kernel_height) |kr| {
                            inline for (0..kernel_width) |kc| {
                                kernel_flat[idx] = as(f32, kernel[kr][kc]);
                                idx += 1;
                            }
                        }

                        // Get data as slices
                        const src_data = self.data[0 .. self.rows * self.cols];
                        const dst_data = out.data[0 .. out.rows * out.cols];

                        Kernel.convolveF32Plane(src_data, dst_data, self.rows, self.cols, kernel_flat, border_mode);
                    } else {
                        // Generic scalar path for other types
                        const half_h = Kernel.half_h;
                        const half_w = Kernel.half_w;
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
                        // Channel separation approach for optimal performance

                        // Convert kernel to integer for faster arithmetic
                        const SCALE = 256;
                        var kernel_int: [Kernel.kernel_size]i32 = undefined;

                        // Flatten and scale the kernel
                        var idx: usize = 0;
                        inline for (0..kernel_height) |kr| {
                            inline for (0..kernel_width) |kc| {
                                kernel_int[idx] = @intFromFloat(@round(as(f32, kernel[kr][kc]) * SCALE));
                                idx += 1;
                            }
                        }

                        // Separate channels using helper
                        const channels = try separateRGBChannels(self, allocator);
                        defer allocator.free(channels[0]);
                        defer allocator.free(channels[1]);
                        defer allocator.free(channels[2]);

                        // Allocate output planes
                        const plane_size = self.rows * self.cols;
                        const r_out = try allocator.alloc(u8, plane_size);
                        defer allocator.free(r_out);
                        const g_out = try allocator.alloc(u8, plane_size);
                        defer allocator.free(g_out);
                        const b_out = try allocator.alloc(u8, plane_size);
                        defer allocator.free(b_out);

                        // Convolve each channel independently using the optimized u8 plane function
                        inline for (.{ channels[0], channels[1], channels[2] }, .{ r_out, g_out, b_out }) |src_channel, dst_channel| {
                            Kernel.convolveU8Plane(src_channel, dst_channel, self.rows, self.cols, kernel_int, border_mode);
                        }

                        // Recombine channels using helper
                        combineRGBChannels(self, r_out, g_out, b_out, out.*);
                    } else {
                        // Generic struct path for other color types
                        const has_alpha = comptime hasAlphaChannel(T);
                        const channels_to_process = if (has_alpha) 3 else fields.len;
                        const half_h = Kernel.half_h;
                        const half_w = Kernel.half_w;

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
                    }
                },
                else => @compileError("Convolution not supported for type " ++ @typeName(T)),
            }
        }

        /// Optimized convolution for u8 planes with integer arithmetic and SIMD for arbitrary kernel sizes.
        /// The kernel must be pre-scaled by 256 for integer arithmetic.
        fn convolveU8Plane(
            src: []const u8,
            dst: []u8,
            rows: usize,
            cols: usize,
            kernel_int: []const i32,
            kernel_height: usize,
            kernel_width: usize,
            border_mode: BorderMode,
        ) void {
            const SCALE = 256;
            const vec_len = comptime std.simd.suggestVectorLength(i32) orelse 8;
            const half_h = kernel_height / 2;
            const half_w = kernel_width / 2;

            for (0..rows) |r| {
                var c: usize = 0;

                // SIMD path for interior pixels
                if (r >= half_h and r + half_h < rows and cols > vec_len + kernel_width) {
                    c = half_w;
                    const safe_end = if (cols > vec_len + half_w) cols - vec_len - half_w else half_w;

                    while (c + vec_len <= safe_end) : (c += vec_len) {
                        // Process vec_len pixels at once
                        var result_vec: @Vector(vec_len, i32) = @splat(0);

                        // Accumulate convolution for each kernel position
                        for (0..kernel_height) |ky| {
                            for (0..kernel_width) |kx| {
                                const kernel_val = kernel_int[ky * kernel_width + kx];
                                const kernel_vec: @Vector(vec_len, i32) = @splat(kernel_val);

                                // Load vec_len pixels from the neighborhood
                                var pixel_vec: @Vector(vec_len, i32) = undefined;
                                for (0..vec_len) |i| {
                                    const src_r = r + ky - half_h;
                                    const src_c = c + i + kx - half_w;
                                    const src_idx = src_r * cols + src_c;
                                    pixel_vec[i] = src[src_idx];
                                }

                                result_vec += pixel_vec * kernel_vec;
                            }
                        }

                        // Scale back and clamp
                        const half_scale_vec: @Vector(vec_len, i32) = @splat(SCALE / 2);
                        const scale_vec: @Vector(vec_len, i32) = @splat(SCALE);
                        const rounded_vec = @divTrunc(result_vec + half_scale_vec, scale_vec);

                        // Store results with clamping
                        for (0..vec_len) |i| {
                            dst[r * cols + c + i] = @intCast(@max(0, @min(255, rounded_vec[i])));
                        }
                    }
                }

                // Scalar fallback for remaining pixels and borders
                while (c < cols) : (c += 1) {
                    if (r >= half_h and r + half_h < rows and c >= half_w and c + half_w < cols) {
                        // Fast path: no border handling needed
                        var result: i32 = 0;
                        for (0..kernel_height) |ky| {
                            for (0..kernel_width) |kx| {
                                const src_r = r + ky - half_h;
                                const src_c = c + kx - half_w;
                                const pixel_val = @as(i32, src[src_r * cols + src_c]);
                                result += pixel_val * kernel_int[ky * kernel_width + kx];
                            }
                        }
                        const rounded = @divTrunc(result + SCALE / 2, SCALE);
                        dst[r * cols + c] = @intCast(@max(0, @min(255, rounded)));
                    } else {
                        // Border handling with integer math
                        const ir = @as(isize, @intCast(r));
                        const ic = @as(isize, @intCast(c));
                        var result: i32 = 0;
                        for (0..kernel_height) |ky| {
                            for (0..kernel_width) |kx| {
                                const iry = ir + @as(isize, @intCast(ky)) - @as(isize, @intCast(half_h));
                                const icx = ic + @as(isize, @intCast(kx)) - @as(isize, @intCast(half_w));
                                const pixel_val = if (iry >= 0 and iry < rows and icx >= 0 and icx < cols)
                                    @as(i32, src[@intCast(iry * @as(isize, @intCast(cols)) + icx)])
                                else switch (border_mode) {
                                    .zero => 0,
                                    .replicate => blk: {
                                        const clamped_r = @max(0, @min(@as(isize, @intCast(rows - 1)), iry));
                                        const clamped_c = @max(0, @min(@as(isize, @intCast(cols - 1)), icx));
                                        break :blk @as(i32, src[@intCast(clamped_r * @as(isize, @intCast(cols)) + clamped_c)]);
                                    },
                                    .mirror => blk: {
                                        var rr = iry;
                                        var cc = icx;
                                        if (rr < 0) rr = -rr - 1;
                                        if (rr >= rows) rr = 2 * @as(isize, @intCast(rows)) - rr - 1;
                                        if (cc < 0) cc = -cc - 1;
                                        if (cc >= cols) cc = 2 * @as(isize, @intCast(cols)) - cc - 1;
                                        break :blk @as(i32, src[@intCast(rr * @as(isize, @intCast(cols)) + cc)]);
                                    },
                                    .wrap => blk: {
                                        const wrapped_r = @mod(iry, @as(isize, @intCast(rows)));
                                        const wrapped_c = @mod(icx, @as(isize, @intCast(cols)));
                                        break :blk @as(i32, src[@intCast(wrapped_r * @as(isize, @intCast(cols)) + wrapped_c)]);
                                    },
                                };
                                result += pixel_val * kernel_int[ky * kernel_width + kx];
                            }
                        }
                        const rounded = @divTrunc(result + SCALE / 2, SCALE);
                        dst[r * cols + c] = @intCast(@max(0, @min(255, rounded)));
                    }
                }
            }
        }

        /// Optimized convolution for f32 planes with SIMD for arbitrary kernel sizes.
        fn convolveF32Plane(
            src: []const f32,
            dst: []f32,
            rows: usize,
            cols: usize,
            kernel: []const f32,
            kernel_height: usize,
            kernel_width: usize,
            border_mode: BorderMode,
        ) void {
            const vec_len = comptime std.simd.suggestVectorLength(f32) orelse 8;
            const half_h = kernel_height / 2;
            const half_w = kernel_width / 2;

            for (0..rows) |r| {
                var c: usize = 0;

                // SIMD path for interior pixels
                if (r >= half_h and r + half_h < rows and cols > vec_len + kernel_width) {
                    c = half_w;
                    const safe_end = if (cols > vec_len + half_w) cols - vec_len - half_w else half_w;

                    while (c + vec_len <= safe_end) : (c += vec_len) {
                        // Process vec_len pixels at once
                        var result_vec: @Vector(vec_len, f32) = @splat(0);

                        // Accumulate convolution for each kernel position
                        for (0..kernel_height) |ky| {
                            for (0..kernel_width) |kx| {
                                const kernel_val = kernel[ky * kernel_width + kx];
                                const kernel_vec: @Vector(vec_len, f32) = @splat(kernel_val);

                                // Load vec_len pixels from the neighborhood
                                var pixel_vec: @Vector(vec_len, f32) = undefined;
                                for (0..vec_len) |i| {
                                    const src_r = r + ky - half_h;
                                    const src_c = c + i + kx - half_w;
                                    const src_idx = src_r * cols + src_c;
                                    pixel_vec[i] = src[src_idx];
                                }

                                result_vec += pixel_vec * kernel_vec;
                            }
                        }

                        // Store results
                        for (0..vec_len) |i| {
                            dst[r * cols + c + i] = result_vec[i];
                        }
                    }
                }

                // Scalar fallback for remaining pixels and borders
                while (c < cols) : (c += 1) {
                    if (r >= half_h and r + half_h < rows and c >= half_w and c + half_w < cols) {
                        // Fast path: no border handling needed
                        var result: f32 = 0;
                        for (0..kernel_height) |ky| {
                            for (0..kernel_width) |kx| {
                                const src_r = r + ky - half_h;
                                const src_c = c + kx - half_w;
                                const pixel_val = src[src_r * cols + src_c];
                                result += pixel_val * kernel[ky * kernel_width + kx];
                            }
                        }
                        dst[r * cols + c] = result;
                    } else {
                        // Border handling
                        const ir = @as(isize, @intCast(r));
                        const ic = @as(isize, @intCast(c));
                        var result: f32 = 0;
                        for (0..kernel_height) |ky| {
                            for (0..kernel_width) |kx| {
                                const iry = ir + @as(isize, @intCast(ky)) - @as(isize, @intCast(half_h));
                                const icx = ic + @as(isize, @intCast(kx)) - @as(isize, @intCast(half_w));
                                const pixel_val = if (iry >= 0 and iry < rows and icx >= 0 and icx < cols)
                                    src[@as(usize, @intCast(iry * @as(isize, @intCast(cols)) + icx))]
                                else switch (border_mode) {
                                    .zero => 0,
                                    .replicate => blk: {
                                        const clamped_r = @max(0, @min(@as(isize, @intCast(rows - 1)), iry));
                                        const clamped_c = @max(0, @min(@as(isize, @intCast(cols - 1)), icx));
                                        break :blk src[@as(usize, @intCast(clamped_r * @as(isize, @intCast(cols)) + clamped_c))];
                                    },
                                    .mirror => blk: {
                                        var rr = iry;
                                        var cc = icx;
                                        if (rr < 0) rr = -rr - 1;
                                        if (rr >= rows) rr = 2 * @as(isize, @intCast(rows)) - rr - 1;
                                        if (cc < 0) cc = -cc - 1;
                                        if (cc >= cols) cc = 2 * @as(isize, @intCast(cols)) - cc - 1;
                                        break :blk src[@as(usize, @intCast(rr * @as(isize, @intCast(cols)) + cc))];
                                    },
                                    .wrap => blk: {
                                        const wrapped_r = @mod(iry, @as(isize, @intCast(rows)));
                                        const wrapped_c = @mod(icx, @as(isize, @intCast(cols)));
                                        break :blk src[@as(usize, @intCast(wrapped_r * @as(isize, @intCast(cols)) + wrapped_c))];
                                    },
                                };
                                result += pixel_val * kernel[ky * kernel_width + kx];
                            }
                        }
                        dst[r * cols + c] = result;
                    }
                }
            }
        }

        /// Optimized separable convolution for u8 planes with integer arithmetic.
        /// The kernel must be pre-scaled by 256 for integer arithmetic.
        fn convolveSeparableU8Plane(
            src: []const u8,
            dst: []u8,
            temp: []u8,
            rows: usize,
            cols: usize,
            kernel_x_int: []const i32,
            kernel_y_int: []const i32,
            border_mode: BorderMode,
        ) void {
            const SCALE = 256;
            const half_x = kernel_x_int.len / 2;
            const half_y = kernel_y_int.len / 2;

            // Horizontal pass (src -> temp)
            for (0..rows) |r| {
                for (0..cols) |c| {
                    var result: i32 = 0;
                    if (c >= half_x and c + half_x < cols) {
                        // Fast path: no border handling needed
                        const c0 = c - half_x;
                        for (kernel_x_int, 0..) |k, i| {
                            const cc = c0 + i;
                            const pixel_val = @as(i32, src[r * cols + cc]);
                            result += pixel_val * k;
                        }
                    } else {
                        // Border handling
                        const ic = @as(isize, @intCast(c));
                        for (kernel_x_int, 0..) |k, i| {
                            const icx = ic + @as(isize, @intCast(i)) - @as(isize, @intCast(half_x));
                            const pixel_val = if (icx >= 0 and icx < cols)
                                @as(i32, src[r * cols + @as(usize, @intCast(icx))])
                            else switch (border_mode) {
                                .zero => 0,
                                .replicate => blk: {
                                    const clamped_c = @max(0, @min(@as(isize, @intCast(cols - 1)), icx));
                                    break :blk @as(i32, src[r * cols + @as(usize, @intCast(clamped_c))]);
                                },
                                .mirror => blk: {
                                    var cc = icx;
                                    if (cc < 0) cc = -cc - 1;
                                    if (cc >= cols) cc = 2 * @as(isize, @intCast(cols)) - cc - 1;
                                    break :blk @as(i32, src[r * cols + @as(usize, @intCast(cc))]);
                                },
                                .wrap => blk: {
                                    const wrapped_c = @mod(icx, @as(isize, @intCast(cols)));
                                    break :blk @as(i32, src[r * cols + @as(usize, @intCast(wrapped_c))]);
                                },
                            };
                            result += pixel_val * k;
                        }
                    }
                    // Store intermediate result with scaling
                    const rounded = @divTrunc(result + SCALE / 2, SCALE);
                    temp[r * cols + c] = @intCast(@max(0, @min(255, rounded)));
                }
            }

            // Vertical pass (temp -> dst)
            for (0..rows) |r| {
                for (0..cols) |c| {
                    var result: i32 = 0;
                    if (r >= half_y and r + half_y < rows) {
                        // Fast path: no border handling needed
                        const r0 = r - half_y;
                        for (kernel_y_int, 0..) |k, i| {
                            const rr = r0 + i;
                            const pixel_val = @as(i32, temp[rr * cols + c]);
                            result += pixel_val * k;
                        }
                    } else {
                        // Border handling
                        const ir = @as(isize, @intCast(r));
                        for (kernel_y_int, 0..) |k, i| {
                            const iry = ir + @as(isize, @intCast(i)) - @as(isize, @intCast(half_y));
                            const pixel_val = if (iry >= 0 and iry < rows)
                                @as(i32, temp[@as(usize, @intCast(iry)) * cols + c])
                            else switch (border_mode) {
                                .zero => 0,
                                .replicate => blk: {
                                    const clamped_r = @max(0, @min(@as(isize, @intCast(rows - 1)), iry));
                                    break :blk @as(i32, temp[@as(usize, @intCast(clamped_r)) * cols + c]);
                                },
                                .mirror => blk: {
                                    var rr = iry;
                                    if (rr < 0) rr = -rr - 1;
                                    if (rr >= rows) rr = 2 * @as(isize, @intCast(rows)) - rr - 1;
                                    break :blk @as(i32, temp[@as(usize, @intCast(rr)) * cols + c]);
                                },
                                .wrap => blk: {
                                    const wrapped_r = @mod(iry, @as(isize, @intCast(rows)));
                                    break :blk @as(i32, temp[@as(usize, @intCast(wrapped_r)) * cols + c]);
                                },
                            };
                            result += pixel_val * k;
                        }
                    }
                    // Store final result with scaling
                    const rounded = @divTrunc(result + SCALE / 2, SCALE);
                    dst[r * cols + c] = @intCast(@max(0, @min(255, rounded)));
                }
            }
        }

        /// Optimized convolution for scalar types (int/float) with SIMD.
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
                    // Optimized path for u8 with integer arithmetic
                    if (T == u8) {
                        // Convert kernels to integer
                        const SCALE = 256;
                        const kernel_x_int = try allocator.alloc(i32, kernel_x.len);
                        defer allocator.free(kernel_x_int);
                        const kernel_y_int = try allocator.alloc(i32, kernel_y.len);
                        defer allocator.free(kernel_y_int);

                        for (kernel_x, 0..) |k, i| {
                            kernel_x_int[i] = @intFromFloat(@round(k * SCALE));
                        }
                        for (kernel_y, 0..) |k, i| {
                            kernel_y_int[i] = @intFromFloat(@round(k * SCALE));
                        }

                        // Get data as slices
                        const src_data = self.data[0 .. self.rows * self.cols];
                        const dst_data = out.data[0 .. out.rows * out.cols];
                        const temp_data = temp.data[0 .. temp.rows * temp.cols];

                        convolveSeparableU8Plane(src_data, dst_data, temp_data, self.rows, self.cols, kernel_x_int, kernel_y_int, border_mode);
                        return; // Skip the rest of the function
                    }

                    // Generic path for other scalar types
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
                    // Optimized path for u8 structs (RGB, RGBA, etc.)
                    const fields = std.meta.fields(T);
                    const all_u8 = comptime blk: {
                        for (fields) |field| {
                            if (field.type != u8) break :blk false;
                        }
                        break :blk true;
                    };

                    if (all_u8 and (fields.len == 3 or fields.len == 4)) {
                        // Channel separation approach for optimal performance

                        // Convert kernels to integer
                        const SCALE = 256;
                        const kernel_x_int = try allocator.alloc(i32, kernel_x.len);
                        defer allocator.free(kernel_x_int);
                        const kernel_y_int = try allocator.alloc(i32, kernel_y.len);
                        defer allocator.free(kernel_y_int);

                        for (kernel_x, 0..) |k, i| {
                            kernel_x_int[i] = @intFromFloat(@round(k * SCALE));
                        }
                        for (kernel_y, 0..) |k, i| {
                            kernel_y_int[i] = @intFromFloat(@round(k * SCALE));
                        }

                        // Separate channels using helper
                        const channels = try separateRGBChannels(self, allocator);
                        defer allocator.free(channels[0]);
                        defer allocator.free(channels[1]);
                        defer allocator.free(channels[2]);

                        // Allocate output and temp planes
                        const plane_size = self.rows * self.cols;
                        const r_out = try allocator.alloc(u8, plane_size);
                        defer allocator.free(r_out);
                        const g_out = try allocator.alloc(u8, plane_size);
                        defer allocator.free(g_out);
                        const b_out = try allocator.alloc(u8, plane_size);
                        defer allocator.free(b_out);

                        const r_temp = try allocator.alloc(u8, plane_size);
                        defer allocator.free(r_temp);
                        const g_temp = try allocator.alloc(u8, plane_size);
                        defer allocator.free(g_temp);
                        const b_temp = try allocator.alloc(u8, plane_size);
                        defer allocator.free(b_temp);

                        // Convolve each channel independently using the optimized u8 plane function
                        inline for (.{ channels[0], channels[1], channels[2] }, .{ r_out, g_out, b_out }, .{ r_temp, g_temp, b_temp }) |src_channel, dst_channel, temp_channel| {
                            convolveSeparableU8Plane(src_channel, dst_channel, temp_channel, self.rows, self.cols, kernel_x_int, kernel_y_int, border_mode);
                        }

                        // Recombine channels using helper
                        combineRGBChannels(self, r_out, g_out, b_out, out.*);
                        return; // Skip the rest of the function
                    }

                    // Generic struct path for other color types
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

            // For now, use float path for all types to ensure correctness
            // TODO: Optimize with signed integer gradients for u8/RGB inputs
            {
                // Original float path for other types
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
        }
    };
}
