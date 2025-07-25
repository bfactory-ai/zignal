//! Image interpolation and resizing algorithms
//!
//! This module provides various interpolation methods for image resizing and
//! sampling, including nearest neighbor, bilinear, bicubic, Catmull-Rom,
//! Lanczos, and Mitchell-Netravali filters.

const std = @import("std");
const as = @import("../meta.zig").as;
const is4xu8Struct = @import("../meta.zig").is4xu8Struct;
const isScalar = @import("../meta.zig").isScalar;
const isStruct = @import("../meta.zig").isStruct;

/// Interpolation method for image resizing and sampling
///
/// Performance and quality comparison:
/// | Method      | Quality | Speed | Best Use Case       | Overshoot |
/// |-------------|---------|-------|---------------------|-----------|
/// | Nearest     | ★☆☆☆☆   | ★★★★★ | Pixel art, masks    | No        |
/// | Bilinear    | ★★☆☆☆   | ★★★★☆ | Real-time, preview  | No        |
/// | Bicubic     | ★★★☆☆   | ★★★☆☆ | General purpose     | Yes       |
/// | Catmull-Rom | ★★★★☆   | ★★★☆☆ | Natural images      | No        |
/// | Lanczos3    | ★★★★★   | ★★☆☆☆ | High-quality resize | Yes       |
/// | Mitchell    | ★★★★☆   | ★★☆☆☆ | Balanced quality    | Yes       |
pub const InterpolationMethod = union(enum) {
    nearest_neighbor,
    bilinear,
    bicubic,
    catmull_rom,
    lanczos,
    mitchell: struct {
        /// Blur parameter (controls blur vs sharpness)
        /// Common values: 1/3 (Mitchell), 1 (B-spline), 0 (Catmull-Rom-like)
        b: f32,
        /// Ringing parameter (controls ringing vs blur)
        /// Common values: 1/3 (Mitchell), 0 (B-spline), 0.5 (Catmull-Rom)
        c: f32,
        pub const default: @This() = .{ .b = 1 / 3, .c = 1 / 3 };
    },
};

/// Sample a single pixel from an image using the specified interpolation method
///
/// Parameters:
/// - T: The pixel type of the image
/// - self: The source image to sample from
/// - x: Horizontal coordinate (0 to cols-1)
/// - y: Vertical coordinate (0 to rows-1)
/// - method: The interpolation method to use
///
/// Returns the interpolated pixel value or null if the coordinates are out of bounds
pub fn interpolate(comptime T: type, self: anytype, x: f32, y: f32, method: InterpolationMethod) ?T {
    return switch (method) {
        .nearest_neighbor => interpolateNearestNeighbor(T, self, x, y),
        .bilinear => interpolateBilinear(T, self, x, y),
        .bicubic => interpolateBicubic(T, self, x, y),
        .catmull_rom => interpolateCatmullRom(T, self, x, y),
        .lanczos => interpolateLanczos(T, self, x, y),
        .mitchell => |m| interpolateMitchell(T, self, x, y, m.b, m.c),
    };
}

/// Resize an image using the specified interpolation method
///
/// Parameters:
/// - T: The pixel type of the image
/// - self: The source image
/// - out: The destination image (must be pre-allocated)
/// - method: The interpolation method to use
pub fn resize(comptime T: type, self: anytype, out: anytype, method: InterpolationMethod) void {
    // Check for scale = 1 (just copy)
    if (self.rows == out.rows and self.cols == out.cols) {
        // If dimensions match exactly, just copy the data
        if (self.data.ptr == out.data.ptr) {
            // Same buffer, nothing to do
            return;
        }
        // Different buffers, need to copy
        @memcpy(out.data[0..out.data.len], self.data[0..self.data.len]);
        return;
    }

    // SIMD optimizations for nearest neighbor
    if (method == .nearest_neighbor) {
        if (is4xu8Struct(T)) {
            return resizeNearestNeighbor4xu8(T, self, out);
        }
    }

    // SIMD optimizations for bilinear
    if (method == .bilinear) {
        if (is4xu8Struct(T)) {
            // Check for 2x upscaling special case
            if (out.rows == self.rows * 2 and out.cols == self.cols * 2) {
                return resize2xUpscale4xu8(T, self, out);
            }
            return resizeBilinear4xu8(T, self, out);
        }
    }

    // SIMD optimizations for bicubic
    if (method == .bicubic) {
        if (is4xu8Struct(T)) {
            return resizeBicubic4xu8(T, self, out);
        }
    }

    // SIMD optimizations for catmull_rom
    if (method == .catmull_rom) {
        if (is4xu8Struct(T)) {
            return resizeCatmullRom4xu8(T, self, out);
        }
    }

    // SIMD optimizations for lanczos
    if (method == .lanczos) {
        if (is4xu8Struct(T)) {
            return resizeLanczos4xu8(T, self, out);
        }
    }

    // SIMD optimizations for mitchell
    if (method == .mitchell) {
        if (is4xu8Struct(T)) {
            return resizeMitchell4xu8(T, self, out, method.mitchell.b, method.mitchell.c);
        }
    }

    // Fall back to generic implementation
    const x_scale = @as(f32, @floatFromInt(self.cols - 1)) / @as(f32, @floatFromInt(out.cols - 1));
    const y_scale = @as(f32, @floatFromInt(self.rows - 1)) / @as(f32, @floatFromInt(out.rows - 1));

    for (0..out.rows) |r| {
        const src_y = @as(f32, @floatFromInt(r)) * y_scale;
        for (0..out.cols) |c| {
            const src_x = @as(f32, @floatFromInt(c)) * x_scale;
            if (interpolate(T, self, src_x, src_y, method)) |val| {
                out.at(r, c).* = val;
            }
        }
    }
}

// ============================================================================
// Private implementation functions
// ============================================================================

fn interpolateNearestNeighbor(comptime T: type, self: anytype, x: f32, y: f32) ?T {
    const col: isize = @intFromFloat(@round(x));
    const row: isize = @intFromFloat(@round(y));

    if (col < 0 or row < 0 or col >= self.cols or row >= self.rows) return null;
    return self.at(@intCast(row), @intCast(col)).*;
}

fn interpolateBilinear(comptime T: type, self: anytype, x: f32, y: f32) ?T {
    const left: isize = @intFromFloat(@floor(x));
    const top: isize = @intFromFloat(@floor(y));
    const right = left + 1;
    const bottom = top + 1;

    if (!(left >= 0 and top >= 0 and right < self.cols and bottom < self.rows)) {
        return null;
    }

    const lr_frac: f32 = x - as(f32, left);
    const tb_frac: f32 = y - as(f32, top);

    const tl: T = self.at(@intCast(top), @intCast(left)).*;
    const tr: T = self.at(@intCast(top), @intCast(right)).*;
    const bl: T = self.at(@intCast(bottom), @intCast(left)).*;
    const br: T = self.at(@intCast(bottom), @intCast(right)).*;

    // Handle different pixel types
    var temp: T = undefined;
    switch (@typeInfo(T)) {
        .int, .float => {
            temp = as(T, (1 - tb_frac) * ((1 - lr_frac) * as(f32, tl) +
                lr_frac * as(f32, tr)) +
                tb_frac * ((1 - lr_frac) * as(f32, bl) +
                    lr_frac * as(f32, br)));
        },
        .@"struct" => {
            inline for (std.meta.fields(T)) |f| {
                @field(temp, f.name) = as(
                    f.type,
                    (1 - tb_frac) * ((1 - lr_frac) * as(f32, @field(tl, f.name)) +
                        lr_frac * as(f32, @field(tr, f.name))) +
                        tb_frac * ((1 - lr_frac) * as(f32, @field(bl, f.name)) +
                            lr_frac * as(f32, @field(br, f.name))),
                );
            }
        },
        else => @compileError("Image(" ++ @typeName(T) ++ ").interpolateBilinear: unsupported image type"),
    }
    return temp;
}

fn interpolateBicubic(comptime T: type, self: anytype, x: f32, y: f32) ?T {
    const ix: isize = @intFromFloat(@floor(x));
    const iy: isize = @intFromFloat(@floor(y));

    // Check bounds - need 4x4 neighborhood
    if (ix < 1 or iy < 1 or ix >= self.cols - 2 or iy >= self.rows - 2) {
        return null;
    }

    const fx = x - as(f32, ix);
    const fy = y - as(f32, iy);

    // Bicubic kernel function
    const cubic = struct {
        fn kernel(t: f32) f32 {
            const at = @abs(t);
            if (at <= 1) {
                return 1 - 2 * at * at + at * at * at;
            } else if (at <= 2) {
                return 4 - 8 * at + 5 * at * at - at * at * at;
            }
            return 0;
        }
    };

    var result: T = std.mem.zeroes(T);

    switch (@typeInfo(T)) {
        .int, .float => {
            var sum: f32 = 0.0;
            for (0..4) |j| {
                const y_idx = iy - 1 + @as(isize, @intCast(j));
                const wy = cubic.kernel(as(f32, @as(isize, @intCast(j)) - 1) - fy);

                for (0..4) |i| {
                    const x_idx = ix - 1 + @as(isize, @intCast(i));
                    const wx = cubic.kernel(as(f32, @as(isize, @intCast(i)) - 1) - fx);

                    sum += wx * wy * as(f32, self.at(@intCast(y_idx), @intCast(x_idx)).*);
                }
            }
            result = if (@typeInfo(T) == .int)
                @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(sum))))
            else
                as(T, sum);
        },
        .@"struct" => {
            inline for (std.meta.fields(T)) |f| {
                var sum: f32 = 0.0;
                for (0..4) |j| {
                    const y_idx = iy - 1 + @as(isize, @intCast(j));
                    const wy = cubic.kernel(as(f32, @as(isize, @intCast(j)) - 1) - fy);

                    for (0..4) |i| {
                        const x_idx = ix - 1 + @as(isize, @intCast(i));
                        const wx = cubic.kernel(as(f32, @as(isize, @intCast(i)) - 1) - fx);

                        sum += wx * wy * as(f32, @field(self.at(@intCast(y_idx), @intCast(x_idx)).*, f.name));
                    }
                }
                @field(result, f.name) = switch (@typeInfo(f.type)) {
                    .int => @intFromFloat(@max(std.math.minInt(f.type), @min(std.math.maxInt(f.type), @round(sum)))),
                    .float => as(f.type, sum),
                    else => @compileError("Unsupported field type for interpolation"),
                };
            }
        },
        else => @compileError("Image(" ++ @typeName(T) ++ ").interpolateBicubic: unsupported image type"),
    }

    return result;
}

fn catmullRomKernel(x: f32) f32 {
    const ax = @abs(x);
    if (ax <= 1) {
        return 1.5 * ax * ax * ax - 2.5 * ax * ax + 1;
    } else if (ax <= 2) {
        return -0.5 * ax * ax * ax + 2.5 * ax * ax - 4 * ax + 2;
    }
    return 0;
}

fn interpolateCatmullRom(comptime T: type, self: anytype, x: f32, y: f32) ?T {
    const ix: isize = @intFromFloat(@floor(x));
    const iy: isize = @intFromFloat(@floor(y));

    // Check bounds - need 4x4 neighborhood
    if (ix < 1 or iy < 1 or ix >= self.cols - 2 or iy >= self.rows - 2) {
        return null;
    }

    const fx = x - as(f32, ix);
    const fy = y - as(f32, iy);

    var result: T = std.mem.zeroes(T);

    switch (@typeInfo(T)) {
        .int, .float => {
            var sum: f32 = 0.0;
            for (0..4) |j| {
                const y_idx = iy - 1 + @as(isize, @intCast(j));
                const wy = catmullRomKernel(as(f32, @as(isize, @intCast(j)) - 1) - fy);

                for (0..4) |i| {
                    const x_idx = ix - 1 + @as(isize, @intCast(i));
                    const wx = catmullRomKernel(as(f32, @as(isize, @intCast(i)) - 1) - fx);

                    sum += wx * wy * as(f32, self.at(@intCast(y_idx), @intCast(x_idx)).*);
                }
            }
            result = if (@typeInfo(T) == .int)
                @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(sum))))
            else
                as(T, sum);
        },
        .@"struct" => {
            inline for (std.meta.fields(T)) |f| {
                var sum: f32 = 0.0;
                for (0..4) |j| {
                    const y_idx = iy - 1 + @as(isize, @intCast(j));
                    const wy = catmullRomKernel(as(f32, @as(isize, @intCast(j)) - 1) - fy);

                    for (0..4) |i| {
                        const x_idx = ix - 1 + @as(isize, @intCast(i));
                        const wx = catmullRomKernel(as(f32, @as(isize, @intCast(i)) - 1) - fx);

                        sum += wx * wy * as(f32, @field(self.at(@intCast(y_idx), @intCast(x_idx)).*, f.name));
                    }
                }
                @field(result, f.name) = switch (@typeInfo(f.type)) {
                    .int => @intFromFloat(@max(std.math.minInt(f.type), @min(std.math.maxInt(f.type), @round(sum)))),
                    .float => as(f.type, sum),
                    else => @compileError("Unsupported field type for interpolation"),
                };
            }
        },
        else => @compileError("Image(" ++ @typeName(T) ++ ").interpolateCatmullRom: unsupported image type"),
    }

    return result;
}

fn lanczosKernel(x: f32, a: f32) f32 {
    if (x == 0) return 1;
    if (@abs(x) >= a) return 0;

    const pi_x = std.math.pi * x;
    const pi_x_over_a = pi_x / a;
    return (a * @sin(pi_x) * @sin(pi_x_over_a)) / (pi_x * pi_x);
}

fn interpolateLanczos(comptime T: type, self: anytype, x: f32, y: f32) ?T {
    const ix: isize = @intFromFloat(@floor(x));
    const iy: isize = @intFromFloat(@floor(y));
    const a: f32 = 3.0; // Lanczos3

    // Check bounds - need 6x6 neighborhood for Lanczos3
    if (ix < 2 or iy < 2 or ix >= self.cols - 3 or iy >= self.rows - 3) {
        return null;
    }

    const fx = x - as(f32, ix);
    const fy = y - as(f32, iy);

    var result: T = std.mem.zeroes(T);

    switch (@typeInfo(T)) {
        .int, .float => {
            var sum: f32 = 0.0;
            var weight_sum: f32 = 0.0;

            for (0..6) |j| {
                const y_idx = iy - 2 + @as(isize, @intCast(j));
                const dy = as(f32, @as(isize, @intCast(j)) - 2) - fy;
                const wy = lanczosKernel(dy, a);

                for (0..6) |i| {
                    const x_idx = ix - 2 + @as(isize, @intCast(i));
                    const dx = as(f32, @as(isize, @intCast(i)) - 2) - fx;
                    const wx = lanczosKernel(dx, a);
                    const w = wx * wy;

                    sum += w * as(f32, self.at(@intCast(y_idx), @intCast(x_idx)).*);
                    weight_sum += w;
                }
            }
            const final_value = if (weight_sum != 0.0) sum / weight_sum else sum;
            result = if (@typeInfo(T) == .int)
                @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(final_value))))
            else
                as(T, final_value);
        },
        .@"struct" => {
            inline for (std.meta.fields(T)) |f| {
                var sum: f32 = 0.0;
                var weight_sum: f32 = 0.0;

                for (0..6) |j| {
                    const y_idx = iy - 2 + @as(isize, @intCast(j));
                    const dy = as(f32, @as(isize, @intCast(j)) - 2) - fy;
                    const wy = lanczosKernel(dy, a);

                    for (0..6) |i| {
                        const x_idx = ix - 2 + @as(isize, @intCast(i));
                        const dx = as(f32, @as(isize, @intCast(i)) - 2) - fx;
                        const wx = lanczosKernel(dx, a);
                        const w = wx * wy;

                        sum += w * as(f32, @field(self.at(@intCast(y_idx), @intCast(x_idx)).*, f.name));
                        weight_sum += w;
                    }
                }
                const final_value = if (weight_sum != 0.0) sum / weight_sum else sum;
                @field(result, f.name) = switch (@typeInfo(f.type)) {
                    .int => @intFromFloat(@max(std.math.minInt(f.type), @min(std.math.maxInt(f.type), @round(final_value)))),
                    .float => as(f.type, final_value),
                    else => @compileError("Unsupported field type for interpolation"),
                };
            }
        },
        else => @compileError("Image(" ++ @typeName(T) ++ ").interpolateLanczos: unsupported image type"),
    }

    return result;
}

fn mitchellKernel(x: f32, b: f32, c: f32) f32 {
    const ax = @abs(x);
    const ax2 = ax * ax;
    const ax3 = ax2 * ax;

    if (ax < 1) {
        return ((12 - 9 * b - 6 * c) * ax3 +
            (-18 + 12 * b + 6 * c) * ax2 +
            (6 - 2 * b)) / 6;
    } else if (ax < 2) {
        return ((-b - 6 * c) * ax3 +
            (6 * b + 30 * c) * ax2 +
            (-12 * b - 48 * c) * ax +
            (8 * b + 24 * c)) / 6;
    }
    return 0;
}

fn interpolateMitchell(comptime T: type, self: anytype, x: f32, y: f32, b: f32, c: f32) ?T {
    const ix: isize = @intFromFloat(@floor(x));
    const iy: isize = @intFromFloat(@floor(y));

    // Check bounds - need 4x4 neighborhood
    if (ix < 1 or iy < 1 or ix >= self.cols - 2 or iy >= self.rows - 2) {
        return null;
    }

    const fx = x - as(f32, ix);
    const fy = y - as(f32, iy);

    var result: T = std.mem.zeroes(T);

    switch (@typeInfo(T)) {
        .int, .float => {
            var sum: f32 = 0.0;
            for (0..4) |j| {
                const y_idx = iy - 1 + @as(isize, @intCast(j));
                const wy = mitchellKernel(as(f32, @as(isize, @intCast(j)) - 1) - fy, b, c);

                for (0..4) |i| {
                    const x_idx = ix - 1 + @as(isize, @intCast(i));
                    const wx = mitchellKernel(as(f32, @as(isize, @intCast(i)) - 1) - fx, b, c);

                    sum += wx * wy * as(f32, self.at(@intCast(y_idx), @intCast(x_idx)).*);
                }
            }
            result = if (@typeInfo(T) == .int)
                @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(sum))))
            else
                as(T, sum);
        },
        .@"struct" => {
            inline for (std.meta.fields(T)) |f| {
                var sum: f32 = 0.0;
                for (0..4) |j| {
                    const y_idx = iy - 1 + @as(isize, @intCast(j));
                    const wy = mitchellKernel(as(f32, @as(isize, @intCast(j)) - 1) - fy, b, c);

                    for (0..4) |i| {
                        const x_idx = ix - 1 + @as(isize, @intCast(i));
                        const wx = mitchellKernel(as(f32, @as(isize, @intCast(i)) - 1) - fx, b, c);

                        sum += wx * wy * as(f32, @field(self.at(@intCast(y_idx), @intCast(x_idx)).*, f.name));
                    }
                }
                @field(result, f.name) = switch (@typeInfo(f.type)) {
                    .int => @intFromFloat(@max(std.math.minInt(f.type), @min(std.math.maxInt(f.type), @round(sum)))),
                    .float => as(f.type, sum),
                    else => @compileError("Unsupported field type for interpolation"),
                };
            }
        },
        else => @compileError("Image(" ++ @typeName(T) ++ ").interpolateMitchell: unsupported image type"),
    }

    return result;
}

// ============================================================================
// SIMD optimized functions
// ============================================================================

/// Specialized 2x upscaling for 4xu8 images (RGBA/RGB) using SIMD
fn resize2xUpscale4xu8(comptime T: type, self: anytype, out: anytype) void {
    std.debug.assert(is4xu8Struct(T));
    std.debug.assert(out.rows == self.rows * 2);
    std.debug.assert(out.cols == self.cols * 2);

    // Process each row
    for (0..self.rows) |sr| {
        const dr = sr * 2;

        // Process each pixel
        for (0..self.cols) |sc| {
            const dc = sc * 2;
            const src_pixel = self.at(sr, sc).*;

            // Convert struct to vector for SIMD operations
            var src_vec: @Vector(4, u16) = undefined;
            inline for (std.meta.fields(T), 0..) |field, i| {
                src_vec[i] = @field(src_pixel, field.name);
            }

            // Write top-left pixel
            out.at(dr, dc).* = src_pixel;

            // Top-right pixel (horizontal interpolation)
            if (sc < self.cols - 1) {
                const next_pixel = self.at(sr, sc + 1).*;
                var next_vec: @Vector(4, u16) = undefined;
                inline for (std.meta.fields(T), 0..) |field, i| {
                    next_vec[i] = @field(next_pixel, field.name);
                }

                const horiz_avg = (src_vec + next_vec) / @as(@Vector(4, u16), @splat(2));
                var result: T = undefined;
                inline for (std.meta.fields(T), 0..) |field, i| {
                    @field(result, field.name) = @intCast(horiz_avg[i]);
                }
                out.at(dr, dc + 1).* = result;
            } else {
                out.at(dr, dc + 1).* = src_pixel;
            }

            // Bottom row interpolation
            if (sr < self.rows - 1) {
                const bottom_pixel = self.at(sr + 1, sc).*;
                var bottom_vec: @Vector(4, u16) = undefined;
                inline for (std.meta.fields(T), 0..) |field, i| {
                    bottom_vec[i] = @field(bottom_pixel, field.name);
                }

                // Bottom-left (vertical interpolation)
                const vert_avg = (src_vec + bottom_vec) / @as(@Vector(4, u16), @splat(2));
                var vert_result: T = undefined;
                inline for (std.meta.fields(T), 0..) |field, i| {
                    @field(vert_result, field.name) = @intCast(vert_avg[i]);
                }
                out.at(dr + 1, dc).* = vert_result;

                // Bottom-right (bilinear interpolation)
                if (sc < self.cols - 1) {
                    const br_pixel = self.at(sr + 1, sc + 1).*;
                    var br_vec: @Vector(4, u16) = undefined;
                    inline for (std.meta.fields(T), 0..) |field, i| {
                        br_vec[i] = @field(br_pixel, field.name);
                    }

                    const next_pixel = self.at(sr, sc + 1).*;
                    var next_vec: @Vector(4, u16) = undefined;
                    inline for (std.meta.fields(T), 0..) |field, i| {
                        next_vec[i] = @field(next_pixel, field.name);
                    }

                    const bilinear_avg = (src_vec + next_vec + bottom_vec + br_vec) / @as(@Vector(4, u16), @splat(4));
                    var bilinear_result: T = undefined;
                    inline for (std.meta.fields(T), 0..) |field, i| {
                        @field(bilinear_result, field.name) = @intCast(bilinear_avg[i]);
                    }
                    out.at(dr + 1, dc + 1).* = bilinear_result;
                } else {
                    out.at(dr + 1, dc + 1).* = vert_result;
                }
            } else {
                // Last row - copy from top row
                out.at(dr + 1, dc).* = src_pixel;
                out.at(dr + 1, dc + 1).* = out.at(dr, dc + 1).*;
            }
        }
    }
}

/// SIMD-optimized nearest neighbor resize for 4xu8 types (RGBA)
fn resizeNearestNeighbor4xu8(comptime T: type, self: anytype, out: anytype) void {
    std.debug.assert(is4xu8Struct(T));

    const x_scale = @as(f32, @floatFromInt(self.cols)) / @as(f32, @floatFromInt(out.cols));
    const y_scale = @as(f32, @floatFromInt(self.rows)) / @as(f32, @floatFromInt(out.rows));

    // Process 4 output pixels at a time using SIMD
    const vec_size = 4;

    for (0..out.rows) |r| {
        const src_y_f = @as(f32, @floatFromInt(r)) * y_scale;
        const src_y = @min(self.rows - 1, @as(usize, @intFromFloat(@round(src_y_f))));

        var c: usize = 0;

        // SIMD processing for groups of 4 pixels
        while (c + vec_size <= out.cols) : (c += vec_size) {
            // Calculate source X coordinates for 4 pixels
            const x_coords = @Vector(4, f32){
                @floatFromInt(c),
                @floatFromInt(c + 1),
                @floatFromInt(c + 2),
                @floatFromInt(c + 3),
            };

            const src_x_vec = x_coords * @as(@Vector(4, f32), @splat(x_scale));

            // Round to nearest and clamp
            const src_x_indices = @Vector(4, u32){
                @min(self.cols - 1, @as(u32, @intFromFloat(@round(src_x_vec[0])))),
                @min(self.cols - 1, @as(u32, @intFromFloat(@round(src_x_vec[1])))),
                @min(self.cols - 1, @as(u32, @intFromFloat(@round(src_x_vec[2])))),
                @min(self.cols - 1, @as(u32, @intFromFloat(@round(src_x_vec[3])))),
            };

            // Gather pixels from source
            inline for (0..vec_size) |i| {
                const src_pixel = self.at(src_y, src_x_indices[i]).*;
                out.at(r, c + i).* = src_pixel;
            }
        }

        // Handle remaining pixels
        while (c < out.cols) : (c += 1) {
            const src_x_f = @as(f32, @floatFromInt(c)) * x_scale;
            const src_x = @min(self.cols - 1, @as(usize, @intFromFloat(@round(src_x_f))));
            out.at(r, c).* = self.at(src_y, src_x).*;
        }
    }
}

/// SIMD-optimized bilinear resize for 4xu8 types (RGBA) - works for all scales
fn resizeBilinear4xu8(comptime T: type, self: anytype, out: anytype) void {
    comptime std.debug.assert(is4xu8Struct(T));

    const x_scale = @as(f32, @floatFromInt(self.cols - 1)) / @as(f32, @floatFromInt(out.cols - 1));
    const y_scale = @as(f32, @floatFromInt(self.rows - 1)) / @as(f32, @floatFromInt(out.rows - 1));

    for (0..out.rows) |r| {
        const src_y = @as(f32, @floatFromInt(r)) * y_scale;
        const y0 = @as(usize, @intFromFloat(@floor(src_y)));
        const y1 = @min(y0 + 1, self.rows - 1);
        const fy = src_y - @as(f32, @floatFromInt(y0));

        for (0..out.cols) |c| {
            const src_x = @as(f32, @floatFromInt(c)) * x_scale;
            const x0 = @as(usize, @intFromFloat(@floor(src_x)));
            const x1 = @min(x0 + 1, self.cols - 1);
            const fx = src_x - @as(f32, @floatFromInt(x0));

            // Load the 4 neighboring pixels
            const tl = self.at(y0, x0).*;
            const tr = self.at(y0, x1).*;
            const bl = self.at(y1, x0).*;
            const br = self.at(y1, x1).*;

            // Convert to f32 vectors - same pattern as boxBlur4xu8Simd
            var tl_vec: @Vector(4, f32) = undefined;
            var tr_vec: @Vector(4, f32) = undefined;
            var bl_vec: @Vector(4, f32) = undefined;
            var br_vec: @Vector(4, f32) = undefined;

            inline for (std.meta.fields(T), 0..) |field, i| {
                tl_vec[i] = @floatFromInt(@field(tl, field.name));
                tr_vec[i] = @floatFromInt(@field(tr, field.name));
                bl_vec[i] = @floatFromInt(@field(bl, field.name));
                br_vec[i] = @floatFromInt(@field(br, field.name));
            }

            // Bilinear interpolation using f32 vectors
            const fx_vec: @Vector(4, f32) = @splat(fx);
            const fy_vec: @Vector(4, f32) = @splat(fy);
            const one_minus_fx_vec: @Vector(4, f32) = @splat(1.0 - fx);
            const one_minus_fy_vec: @Vector(4, f32) = @splat(1.0 - fy);

            // Horizontal interpolation
            const top = tl_vec * one_minus_fx_vec + tr_vec * fx_vec;
            const bottom = bl_vec * one_minus_fx_vec + br_vec * fx_vec;

            // Vertical interpolation
            const result_vec = top * one_minus_fy_vec + bottom * fy_vec;

            // Convert back to struct with clamping
            var result: T = undefined;
            inline for (std.meta.fields(T), 0..) |field, i| {
                @field(result, field.name) = @intFromFloat(@max(0, @min(255, @round(result_vec[i]))));
            }

            out.at(r, c).* = result;
        }
    }
}

/// SIMD-optimized bicubic resize for 4xu8 types (RGBA)
fn resizeBicubic4xu8(comptime T: type, self: anytype, out: anytype) void {
    comptime std.debug.assert(is4xu8Struct(T));

    const x_scale = @as(f32, @floatFromInt(self.cols - 1)) / @as(f32, @floatFromInt(out.cols - 1));
    const y_scale = @as(f32, @floatFromInt(self.rows - 1)) / @as(f32, @floatFromInt(out.rows - 1));

    // Bicubic kernel function (inlined for performance)
    const bicubicKernel = struct {
        inline fn eval(t: f32) f32 {
            const at = @abs(t);
            if (at <= 1) {
                return 1 - 2 * at * at + at * at * at;
            } else if (at <= 2) {
                return 4 - 8 * at + 5 * at * at - at * at * at;
            }
            return 0;
        }
    };

    // Process each output pixel
    for (0..out.rows) |r| {
        const src_y = @as(f32, @floatFromInt(r)) * y_scale;
        const iy = @as(isize, @intFromFloat(@floor(src_y)));
        const fy = src_y - @as(f32, @floatFromInt(iy));

        // Skip if we can't get a 4x4 neighborhood
        if (iy < 1 or iy >= self.rows - 2) {
            // Fall back to nearest neighbor for edge pixels
            for (0..out.cols) |c| {
                const src_x = @as(f32, @floatFromInt(c)) * x_scale;
                const ix_clamped = @max(0, @min(self.cols - 1, @as(usize, @intFromFloat(@round(src_x)))));
                const iy_clamped = @max(0, @min(self.rows - 1, @as(usize, @intCast(@max(0, @min(@as(isize, @intCast(self.rows - 1)), iy))))));
                out.at(r, c).* = self.at(iy_clamped, ix_clamped).*;
            }
            continue;
        }

        // Pre-compute y weights
        const y_weights = @Vector(4, f32){
            bicubicKernel.eval(-1 - fy),
            bicubicKernel.eval(0 - fy),
            bicubicKernel.eval(1 - fy),
            bicubicKernel.eval(2 - fy),
        };

        for (0..out.cols) |c| {
            const src_x = @as(f32, @floatFromInt(c)) * x_scale;
            const ix = @as(isize, @intFromFloat(@floor(src_x)));
            const fx = src_x - @as(f32, @floatFromInt(ix));

            // Skip if we can't get a 4x4 neighborhood
            if (ix < 1 or ix >= self.cols - 2) {
                // Fall back to nearest neighbor for edge pixels
                const ix_clamped = @max(0, @min(self.cols - 1, @as(usize, @intFromFloat(@round(src_x)))));
                const iy_clamped = @as(usize, @intCast(iy));
                out.at(r, c).* = self.at(iy_clamped, ix_clamped).*;
                continue;
            }

            // Pre-compute x weights
            const x_weights = @Vector(4, f32){
                bicubicKernel.eval(-1 - fx),
                bicubicKernel.eval(0 - fx),
                bicubicKernel.eval(1 - fx),
                bicubicKernel.eval(2 - fx),
            };

            // Accumulate weighted sum for each channel using SIMD
            var sums = @Vector(4, f32){ 0, 0, 0, 0 }; // RGBA channels

            // Process the 4x4 neighborhood more efficiently
            inline for (0..4) |j| {
                const y_idx = @as(usize, @intCast(iy - 1 + @as(isize, @intCast(j))));
                const wy_vec: @Vector(4, f32) = @splat(y_weights[j]);

                // Process all 4 x positions for this row
                var row_sum = @Vector(4, f32){ 0, 0, 0, 0 };

                inline for (0..4) |i| {
                    const x_idx = @as(usize, @intCast(ix - 1 + @as(isize, @intCast(i))));
                    const pixel = self.at(y_idx, x_idx).*;

                    // Convert pixel to vector - same pattern as bilinear
                    var pixel_vec: @Vector(4, f32) = undefined;
                    inline for (std.meta.fields(T), 0..) |field, k| {
                        pixel_vec[k] = @floatFromInt(@field(pixel, field.name));
                    }

                    // Accumulate with x weight
                    const wx_vec: @Vector(4, f32) = @splat(x_weights[i]);
                    row_sum += pixel_vec * wx_vec;
                }

                // Apply y weight to entire row
                sums += row_sum * wy_vec;
            }

            // Clamp and convert back to struct
            var result: T = undefined;
            inline for (std.meta.fields(T), 0..) |field, k| {
                @field(result, field.name) = @intFromFloat(@max(0, @min(255, @round(sums[k]))));
            }

            out.at(r, c).* = result;
        }
    }
}

/// SIMD-optimized Catmull-Rom resize for 4xu8 types (RGBA)
fn resizeCatmullRom4xu8(comptime T: type, self: anytype, out: anytype) void {
    comptime std.debug.assert(is4xu8Struct(T));

    const x_scale = @as(f32, @floatFromInt(self.cols - 1)) / @as(f32, @floatFromInt(out.cols - 1));
    const y_scale = @as(f32, @floatFromInt(self.rows - 1)) / @as(f32, @floatFromInt(out.rows - 1));

    // Catmull-Rom kernel function (inlined for performance)
    const kernel = struct {
        inline fn eval(t: f32) f32 {
            const at = @abs(t);
            if (at <= 1) {
                return 1.5 * at * at * at - 2.5 * at * at + 1;
            } else if (at <= 2) {
                return -0.5 * at * at * at + 2.5 * at * at - 4 * at + 2;
            }
            return 0;
        }
    };

    // Process each output pixel
    for (0..out.rows) |r| {
        const src_y = @as(f32, @floatFromInt(r)) * y_scale;
        const iy = @as(isize, @intFromFloat(@floor(src_y)));
        const fy = src_y - @as(f32, @floatFromInt(iy));

        // Skip if we can't get a 4x4 neighborhood
        if (iy < 1 or iy >= self.rows - 2) {
            // Fall back to nearest neighbor for edge pixels
            for (0..out.cols) |c| {
                const src_x = @as(f32, @floatFromInt(c)) * x_scale;
                const ix_clamped = @max(0, @min(self.cols - 1, @as(usize, @intFromFloat(@round(src_x)))));
                const iy_clamped = @max(0, @min(self.rows - 1, @as(usize, @intCast(@max(0, @min(@as(isize, @intCast(self.rows - 1)), iy))))));
                out.at(r, c).* = self.at(iy_clamped, ix_clamped).*;
            }
            continue;
        }

        // Pre-compute y weights
        const y_weights = @Vector(4, f32){
            kernel.eval(-1 - fy),
            kernel.eval(0 - fy),
            kernel.eval(1 - fy),
            kernel.eval(2 - fy),
        };

        for (0..out.cols) |c| {
            const src_x = @as(f32, @floatFromInt(c)) * x_scale;
            const ix = @as(isize, @intFromFloat(@floor(src_x)));
            const fx = src_x - @as(f32, @floatFromInt(ix));

            // Skip if we can't get a 4x4 neighborhood
            if (ix < 1 or ix >= self.cols - 2) {
                // Fall back to nearest neighbor for edge pixels
                const ix_clamped = @max(0, @min(self.cols - 1, @as(usize, @intFromFloat(@round(src_x)))));
                const iy_clamped = @as(usize, @intCast(iy));
                out.at(r, c).* = self.at(iy_clamped, ix_clamped).*;
                continue;
            }

            // Pre-compute x weights
            const x_weights = @Vector(4, f32){
                kernel.eval(-1 - fx),
                kernel.eval(0 - fx),
                kernel.eval(1 - fx),
                kernel.eval(2 - fx),
            };

            // Accumulate weighted sum for each channel using SIMD
            var sums = @Vector(4, f32){ 0, 0, 0, 0 }; // RGBA channels

            // Process the 4x4 neighborhood more efficiently
            inline for (0..4) |j| {
                const y_idx = @as(usize, @intCast(iy - 1 + @as(isize, @intCast(j))));
                const wy_vec: @Vector(4, f32) = @splat(y_weights[j]);

                // Process all 4 x positions for this row
                var row_sum = @Vector(4, f32){ 0, 0, 0, 0 };

                inline for (0..4) |i| {
                    const x_idx = @as(usize, @intCast(ix - 1 + @as(isize, @intCast(i))));
                    const pixel = self.at(y_idx, x_idx).*;

                    // Convert pixel to vector - same pattern as bilinear
                    var pixel_vec: @Vector(4, f32) = undefined;
                    inline for (std.meta.fields(T), 0..) |field, k| {
                        pixel_vec[k] = @floatFromInt(@field(pixel, field.name));
                    }

                    // Accumulate with x weight
                    const wx_vec: @Vector(4, f32) = @splat(x_weights[i]);
                    row_sum += pixel_vec * wx_vec;
                }

                // Apply y weight to entire row
                sums += row_sum * wy_vec;
            }

            // Clamp and convert back to struct
            var result: T = undefined;
            inline for (std.meta.fields(T), 0..) |field, k| {
                @field(result, field.name) = @intFromFloat(@max(0, @min(255, @round(sums[k]))));
            }

            out.at(r, c).* = result;
        }
    }
}

/// SIMD-optimized Mitchell-Netravali resize for 4xu8 types (RGBA)
fn resizeMitchell4xu8(comptime T: type, self: anytype, out: anytype, b: f32, c: f32) void {
    comptime std.debug.assert(is4xu8Struct(T));

    const x_scale = @as(f32, @floatFromInt(self.cols - 1)) / @as(f32, @floatFromInt(out.cols - 1));
    const y_scale = @as(f32, @floatFromInt(self.rows - 1)) / @as(f32, @floatFromInt(out.rows - 1));

    // Mitchell kernel function (inlined for performance)
    const kernel = struct {
        inline fn eval(x: f32, b_param: f32, c_param: f32) f32 {
            const ax = @abs(x);
            const ax2 = ax * ax;
            const ax3 = ax2 * ax;

            if (ax < 1) {
                return ((12 - 9 * b_param - 6 * c_param) * ax3 +
                    (-18 + 12 * b_param + 6 * c_param) * ax2 +
                    (6 - 2 * b_param)) / 6;
            } else if (ax < 2) {
                return ((-b_param - 6 * c_param) * ax3 +
                    (6 * b_param + 30 * c_param) * ax2 +
                    (-12 * b_param - 48 * c_param) * ax +
                    (8 * b_param + 24 * c_param)) / 6;
            }
            return 0;
        }
    };

    // Process each output pixel
    for (0..out.rows) |r| {
        const src_y = @as(f32, @floatFromInt(r)) * y_scale;
        const iy = @as(isize, @intFromFloat(@floor(src_y)));
        const fy = src_y - @as(f32, @floatFromInt(iy));

        // Skip if we can't get a 4x4 neighborhood
        if (iy < 1 or iy >= self.rows - 2) {
            // Fall back to nearest neighbor for edge pixels
            for (0..out.cols) |c_col| {
                const src_x = @as(f32, @floatFromInt(c_col)) * x_scale;
                const ix_clamped = @max(0, @min(self.cols - 1, @as(usize, @intFromFloat(@round(src_x)))));
                const iy_clamped = @max(0, @min(self.rows - 1, @as(usize, @intCast(@max(0, @min(@as(isize, @intCast(self.rows - 1)), iy))))));
                out.at(r, c_col).* = self.at(iy_clamped, ix_clamped).*;
            }
            continue;
        }

        // Pre-compute y weights
        const y_weights = @Vector(4, f32){
            kernel.eval(-1 - fy, b, c),
            kernel.eval(0 - fy, b, c),
            kernel.eval(1 - fy, b, c),
            kernel.eval(2 - fy, b, c),
        };

        for (0..out.cols) |c_col| {
            const src_x = @as(f32, @floatFromInt(c_col)) * x_scale;
            const ix = @as(isize, @intFromFloat(@floor(src_x)));
            const fx = src_x - @as(f32, @floatFromInt(ix));

            // Skip if we can't get a 4x4 neighborhood
            if (ix < 1 or ix >= self.cols - 2) {
                // Fall back to nearest neighbor for edge pixels
                const ix_clamped = @max(0, @min(self.cols - 1, @as(usize, @intFromFloat(@round(src_x)))));
                const iy_clamped = @as(usize, @intCast(iy));
                out.at(r, c_col).* = self.at(iy_clamped, ix_clamped).*;
                continue;
            }

            // Pre-compute x weights
            const x_weights = @Vector(4, f32){
                kernel.eval(-1 - fx, b, c),
                kernel.eval(0 - fx, b, c),
                kernel.eval(1 - fx, b, c),
                kernel.eval(2 - fx, b, c),
            };

            // Accumulate weighted sum for each channel using SIMD
            var sums = @Vector(4, f32){ 0, 0, 0, 0 }; // RGBA channels

            // Process the 4x4 neighborhood more efficiently
            inline for (0..4) |j| {
                const y_idx = @as(usize, @intCast(iy - 1 + @as(isize, @intCast(j))));
                const wy_vec: @Vector(4, f32) = @splat(y_weights[j]);

                // Process all 4 x positions for this row
                var row_sum = @Vector(4, f32){ 0, 0, 0, 0 };

                inline for (0..4) |i| {
                    const x_idx = @as(usize, @intCast(ix - 1 + @as(isize, @intCast(i))));
                    const pixel = self.at(y_idx, x_idx).*;

                    // Convert pixel to vector - same pattern as bilinear
                    var pixel_vec: @Vector(4, f32) = undefined;
                    inline for (std.meta.fields(T), 0..) |field, k| {
                        pixel_vec[k] = @floatFromInt(@field(pixel, field.name));
                    }

                    // Accumulate with x weight
                    const wx_vec: @Vector(4, f32) = @splat(x_weights[i]);
                    row_sum += pixel_vec * wx_vec;
                }

                // Apply y weight to entire row
                sums += row_sum * wy_vec;
            }

            // Clamp and convert back to struct
            var result: T = undefined;
            inline for (std.meta.fields(T), 0..) |field, k| {
                @field(result, field.name) = @intFromFloat(@max(0, @min(255, @round(sums[k]))));
            }

            out.at(r, c_col).* = result;
        }
    }
}

/// SIMD-optimized Lanczos resize for 4xu8 types (RGBA)
fn resizeLanczos4xu8(comptime T: type, self: anytype, out: anytype) void {
    comptime std.debug.assert(is4xu8Struct(T));

    const x_scale = @as(f32, @floatFromInt(self.cols - 1)) / @as(f32, @floatFromInt(out.cols - 1));
    const y_scale = @as(f32, @floatFromInt(self.rows - 1)) / @as(f32, @floatFromInt(out.rows - 1));

    // Lanczos kernel function (inlined for performance)
    const kernel = struct {
        inline fn eval(x: f32, a: f32) f32 {
            if (x == 0) return 1;
            const ax = @abs(x);
            if (ax >= a) return 0;

            const pi_x = std.math.pi * x;
            const pi_x_over_a = pi_x / a;
            return (a * @sin(pi_x) * @sin(pi_x_over_a)) / (pi_x * pi_x);
        }
    };

    const a: f32 = 3.0; // Lanczos3

    // Process each output pixel
    for (0..out.rows) |r| {
        const src_y = @as(f32, @floatFromInt(r)) * y_scale;
        const iy = @as(isize, @intFromFloat(@floor(src_y)));
        const fy = src_y - @as(f32, @floatFromInt(iy));

        // Skip if we can't get a 6x6 neighborhood
        if (iy < 2 or iy >= self.rows - 3) {
            // Fall back to nearest neighbor for edge pixels
            for (0..out.cols) |c| {
                const src_x = @as(f32, @floatFromInt(c)) * x_scale;
                const ix_clamped = @max(0, @min(self.cols - 1, @as(usize, @intFromFloat(@round(src_x)))));
                const iy_clamped = @max(0, @min(self.rows - 1, @as(usize, @intCast(@max(0, @min(@as(isize, @intCast(self.rows - 1)), iy))))));
                out.at(r, c).* = self.at(iy_clamped, ix_clamped).*;
            }
            continue;
        }

        // Pre-compute y weights
        const y_weights = @Vector(6, f32){
            kernel.eval(-2 - fy, a),
            kernel.eval(-1 - fy, a),
            kernel.eval(0 - fy, a),
            kernel.eval(1 - fy, a),
            kernel.eval(2 - fy, a),
            kernel.eval(3 - fy, a),
        };

        for (0..out.cols) |c| {
            const src_x = @as(f32, @floatFromInt(c)) * x_scale;
            const ix = @as(isize, @intFromFloat(@floor(src_x)));
            const fx = src_x - @as(f32, @floatFromInt(ix));

            // Skip if we can't get a 6x6 neighborhood
            if (ix < 2 or ix >= self.cols - 3) {
                // Fall back to nearest neighbor for edge pixels
                const ix_clamped = @max(0, @min(self.cols - 1, @as(usize, @intFromFloat(@round(src_x)))));
                const iy_clamped = @as(usize, @intCast(iy));
                out.at(r, c).* = self.at(iy_clamped, ix_clamped).*;
                continue;
            }

            // Pre-compute x weights
            const x_weights = @Vector(6, f32){
                kernel.eval(-2 - fx, a),
                kernel.eval(-1 - fx, a),
                kernel.eval(0 - fx, a),
                kernel.eval(1 - fx, a),
                kernel.eval(2 - fx, a),
                kernel.eval(3 - fx, a),
            };

            // Accumulate weighted sum for each channel using SIMD
            var sums = @Vector(4, f32){ 0, 0, 0, 0 }; // RGBA channels
            var weight_sum: f32 = 0;

            // Process the 6x6 neighborhood more efficiently
            inline for (0..6) |j| {
                const y_idx = @as(usize, @intCast(iy - 2 + @as(isize, @intCast(j))));
                const wy = y_weights[j];

                // Process all 6 x positions for this row
                var row_sum = @Vector(4, f32){ 0, 0, 0, 0 };
                var row_weight: f32 = 0;

                inline for (0..6) |i| {
                    const x_idx = @as(usize, @intCast(ix - 2 + @as(isize, @intCast(i))));
                    const pixel = self.at(y_idx, x_idx).*;

                    // Convert pixel to vector - same pattern as bilinear
                    var pixel_vec: @Vector(4, f32) = undefined;
                    inline for (std.meta.fields(T), 0..) |field, k| {
                        pixel_vec[k] = @floatFromInt(@field(pixel, field.name));
                    }

                    // Calculate combined weight
                    const wx = x_weights[i];
                    const w = wx * wy;

                    // Accumulate
                    const w_vec: @Vector(4, f32) = @splat(w);
                    row_sum += pixel_vec * w_vec;
                    row_weight += w;
                }

                sums += row_sum;
                weight_sum += row_weight;
            }

            // Normalize by weight sum and convert back to struct
            var result: T = undefined;
            if (weight_sum != 0) {
                const inv_weight: @Vector(4, f32) = @splat(1.0 / weight_sum);
                const normalized = sums * inv_weight;
                inline for (std.meta.fields(T), 0..) |field, k| {
                    @field(result, field.name) = @intFromFloat(@max(0, @min(255, @round(normalized[k]))));
                }
            } else {
                // Fallback to center pixel if weights sum to zero
                const center_pixel = self.at(@intCast(iy), @intCast(ix)).*;
                result = center_pixel;
            }

            out.at(r, c).* = result;
        }
    }
}
