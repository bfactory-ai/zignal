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
/// | Method      | Quality | Speed | Use Case            | Separable |
/// |-------------|---------|-------|---------------------|-----------|
/// | Nearest     | ★☆☆☆☆   | ★★★★★ | Pixel art, speed    | N/A       |
/// | Bilinear    | ★★☆☆☆   | ★★★★☆ | Fast smooth resize  | Yes       |
/// | Bicubic     | ★★★☆☆   | ★★★☆☆ | Good quality        | Yes       |
/// | Catmull-Rom | ★★★★☆   | ★★☆☆☆ | Sharp details       | Yes       |
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
    // SIMD optimizations for bilinear
    if (method == .bilinear) {
        // Check for special case: exact 2x upscaling
        if (out.rows == self.rows * 2 and out.cols == self.cols * 2) {
            if (T == u8) {
                return resize2xUpscaleU8(T, self, out);
            } else if (is4xu8Struct(T)) {
                return resize2xUpscale4xu8(T, self, out);
            }
        }
        
        // Use separable bilinear for other scales
        if (T == u8 or T == f32) {
            return resizeBilinearSeparable(T, self, out);
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

/// Specialized 2x upscaling for u8 images using SIMD
fn resize2xUpscaleU8(comptime T: type, self: anytype, out: anytype) void {
    std.debug.assert(T == u8);
    std.debug.assert(out.rows == self.rows * 2);
    std.debug.assert(out.cols == self.cols * 2);
    
    // Process 8 pixels at a time using SIMD
    const VecSize = 8;
    const Vec = @Vector(VecSize, u16);
    
    for (0..self.rows) |sr| {
        const dr = sr * 2;
        var col: usize = 0;
        
        // SIMD path for blocks of 8 pixels
        while (col + VecSize <= self.cols) : (col += VecSize) {
            // Load 8 source pixels
            var src_pixels: [VecSize]u8 = undefined;
            for (0..VecSize) |i| {
                src_pixels[i] = self.at(sr, col + i).*;
            }
            
            // Convert to u16 for averaging
            const src_vec: Vec = src_pixels;
            
            // For horizontal interpolation: average adjacent pixels
            var horiz_avg: [VecSize - 1]u8 = undefined;
            for (0..VecSize - 1) |i| {
                horiz_avg[i] = @intCast((src_vec[i] + src_vec[i + 1]) / 2);
            }
            
            // Write output pixels in 2x2 blocks
            for (0..VecSize) |i| {
                const dst_col = (col + i) * 2;
                // Top-left pixel
                out.at(dr, dst_col).* = src_pixels[i];
                // Top-right pixel (interpolated if not last)
                if (i < VecSize - 1) {
                    out.at(dr, dst_col + 1).* = horiz_avg[i];
                } else if (col + i < self.cols - 1) {
                    const val = self.at(sr, col + i).*;
                    const next_val = self.at(sr, col + i + 1).*;
                    out.at(dr, dst_col + 1).* = @intCast((as(u16, val) + as(u16, next_val)) / 2);
                } else {
                    out.at(dr, dst_col + 1).* = src_pixels[i];
                }
            }
            
            // Vertical interpolation for bottom row
            if (sr < self.rows - 1) {
                for (0..VecSize) |i| {
                    const dst_col = (col + i) * 2;
                    const top_val = src_pixels[i];
                    const bottom_val = self.at(sr + 1, col + i).*;
                    const vert_avg = @as(u8, @intCast((as(u16, top_val) + as(u16, bottom_val)) / 2));
                    
                    // Bottom-left pixel
                    out.at(dr + 1, dst_col).* = vert_avg;
                    
                    // Bottom-right pixel (bilinear interpolation)
                    if (i < VecSize - 1 and col + i < self.cols - 1) {
                        const br_val = self.at(sr + 1, col + i + 1).*;
                        const diag_avg = @as(u8, @intCast((as(u16, top_val) + as(u16, src_pixels[i + 1]) + 
                                                           as(u16, bottom_val) + as(u16, br_val)) / 4));
                        out.at(dr + 1, dst_col + 1).* = diag_avg;
                    } else {
                        out.at(dr + 1, dst_col + 1).* = vert_avg;
                    }
                }
            } else {
                // Last row - just copy
                for (0..VecSize) |i| {
                    const dst_col = (col + i) * 2;
                    out.at(dr + 1, dst_col).* = src_pixels[i];
                    out.at(dr + 1, dst_col + 1).* = out.at(dr, dst_col + 1).*;
                }
            }
        }
        
        // Handle remaining pixels
        while (col < self.cols) : (col += 1) {
            const val = self.at(sr, col).*;
            const dst_col = col * 2;
            
            // Top-left
            out.at(dr, dst_col).* = val;
            
            // Top-right
            if (col < self.cols - 1) {
                const next_val = self.at(sr, col + 1).*;
                out.at(dr, dst_col + 1).* = @intCast((as(u16, val) + as(u16, next_val)) / 2);
            } else {
                out.at(dr, dst_col + 1).* = val;
            }
            
            // Bottom row
            if (sr < self.rows - 1) {
                const bottom_val = self.at(sr + 1, col).*;
                const vert_avg = @as(u8, @intCast((as(u16, val) + as(u16, bottom_val)) / 2));
                out.at(dr + 1, dst_col).* = vert_avg;
                
                if (col < self.cols - 1) {
                    const br_val = self.at(sr + 1, col + 1).*;
                    const next_val = self.at(sr, col + 1).*;
                    const diag_avg = @as(u8, @intCast((as(u16, val) + as(u16, next_val) + 
                                                      as(u16, bottom_val) + as(u16, br_val)) / 4));
                    out.at(dr + 1, dst_col + 1).* = diag_avg;
                } else {
                    out.at(dr + 1, dst_col + 1).* = vert_avg;
                }
            } else {
                out.at(dr + 1, dst_col).* = val;
                out.at(dr + 1, dst_col + 1).* = out.at(dr, dst_col + 1).*;
            }
        }
    }
}

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

/// Separable bilinear interpolation implementation for better cache efficiency
fn resizeBilinearSeparable(comptime T: type, self: anytype, out: anytype) void {
    const x_scale = @as(f32, @floatFromInt(self.cols)) / @as(f32, @floatFromInt(out.cols));
    const y_scale = @as(f32, @floatFromInt(self.rows)) / @as(f32, @floatFromInt(out.rows));
    
    // Allocate temporary buffer for horizontal pass
    // Try to use stack allocation for small buffers, heap for large
    const temp_size = out.cols * self.rows;
    const use_stack = temp_size <= 16384; // 64KB for f32
    
    if (use_stack) {
        var stack_buffer: [16384]f32 = undefined;
        const temp_data = stack_buffer[0..temp_size];
        resizeBilinearSeparableImpl(T, self, out, temp_data, x_scale, y_scale);
    } else {
        // For now, fall back to scalar implementation for large images
        // In a real implementation, we'd allocate from a provided allocator
        resizeBilinearScalar(T, self, out);
    }
}

fn resizeBilinearSeparableImpl(
    comptime T: type,
    self: anytype,
    out: anytype,
    temp_data: []f32,
    x_scale: f32,
    y_scale: f32,
) void {
    // Horizontal pass: resize each row from source to temp buffer
    for (0..self.rows) |r| {
        const row_offset = r * out.cols;
        
        for (0..out.cols) |c| {
            const src_x = @as(f32, @floatFromInt(c)) * x_scale;
            const x0 = @as(usize, @intFromFloat(@floor(src_x)));
            const x1 = @min(x0 + 1, self.cols - 1);
            const fx = src_x - @as(f32, @floatFromInt(x0));
            
            const p0 = self.at(r, x0).*;
            const p1 = self.at(r, x1).*;
            
            const val = as(f32, p0) * (1 - fx) + as(f32, p1) * fx;
            temp_data[row_offset + c] = val;
        }
    }
    
    // Vertical pass: resize each column from temp buffer to output
    for (0..out.rows) |r| {
        const src_y = @as(f32, @floatFromInt(r)) * y_scale;
        const y0 = @as(usize, @intFromFloat(@floor(src_y)));
        const y1 = @min(y0 + 1, self.rows - 1);
        const fy = src_y - @as(f32, @floatFromInt(y0));
        
        for (0..out.cols) |c| {
            const p0 = temp_data[y0 * out.cols + c];
            const p1 = temp_data[y1 * out.cols + c];
            
            const val = p0 * (1 - fy) + p1 * fy;
            out.at(r, c).* = if (T == f32) val else as(T, @round(val));
        }
    }
}

/// Fallback scalar bilinear implementation
fn resizeBilinearScalar(comptime T: type, self: anytype, out: anytype) void {
    const x_scale = @as(f32, @floatFromInt(self.cols - 1)) / @as(f32, @floatFromInt(out.cols - 1));
    const y_scale = @as(f32, @floatFromInt(self.rows - 1)) / @as(f32, @floatFromInt(out.rows - 1));

    for (0..out.rows) |r| {
        const src_y = @as(f32, @floatFromInt(r)) * y_scale;
        for (0..out.cols) |c| {
            const src_x = @as(f32, @floatFromInt(c)) * x_scale;
            if (interpolateBilinear(T, self, src_x, src_y)) |val| {
                out.at(r, c).* = val;
            }
        }
    }
}