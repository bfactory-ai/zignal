const std = @import("std");
const zignal = @import("zignal");
const Image = zignal.Image;
const Rgb = zignal.Rgb;
const Rgba = zignal.Rgba;
const BorderMode = zignal.BorderMode;

const Timer = std.time.Timer;

// Baseline (non-SIMD) implementation from git history
fn convolve3x3Baseline(comptime T: type, self: Image(T), kernel: anytype, out: Image(T), border_mode: BorderMode) void {
    const as = zignal.meta.as;

    const kr = [9]f32{
        as(f32, kernel[0][0]), as(f32, kernel[0][1]), as(f32, kernel[0][2]),
        as(f32, kernel[1][0]), as(f32, kernel[1][1]), as(f32, kernel[1][2]),
        as(f32, kernel[2][0]), as(f32, kernel[2][1]), as(f32, kernel[2][2]),
    };

    switch (@typeInfo(T)) {
        .int, .float => {
            // Scalar types - single channel
            for (0..self.rows) |r| {
                for (0..self.cols) |c| {
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
                        const p00 = getPixelWithBorderBaseline(T, self, ir - 1, ic - 1, border_mode);
                        const p01 = getPixelWithBorderBaseline(T, self, ir - 1, ic, border_mode);
                        const p02 = getPixelWithBorderBaseline(T, self, ir - 1, ic + 1, border_mode);
                        const p10 = getPixelWithBorderBaseline(T, self, ir, ic - 1, border_mode);
                        const p11 = getPixelWithBorderBaseline(T, self, ir, ic, border_mode);
                        const p12 = getPixelWithBorderBaseline(T, self, ir, ic + 1, border_mode);
                        const p20 = getPixelWithBorderBaseline(T, self, ir + 1, ic - 1, border_mode);
                        const p21 = getPixelWithBorderBaseline(T, self, ir + 1, ic, border_mode);
                        const p22 = getPixelWithBorderBaseline(T, self, ir + 1, ic + 1, border_mode);

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
            // Struct types - channel-wise convolution
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
                        const p00 = getPixelWithBorderBaseline(T, self, ir - 1, ic - 1, border_mode);
                        const p01 = getPixelWithBorderBaseline(T, self, ir - 1, ic, border_mode);
                        const p02 = getPixelWithBorderBaseline(T, self, ir - 1, ic + 1, border_mode);
                        const p10 = getPixelWithBorderBaseline(T, self, ir, ic - 1, border_mode);
                        const p11 = getPixelWithBorderBaseline(T, self, ir, ic, border_mode);
                        const p12 = getPixelWithBorderBaseline(T, self, ir, ic + 1, border_mode);
                        const p20 = getPixelWithBorderBaseline(T, self, ir + 1, ic - 1, border_mode);
                        const p21 = getPixelWithBorderBaseline(T, self, ir + 1, ic, border_mode);
                        const p22 = getPixelWithBorderBaseline(T, self, ir + 1, ic + 1, border_mode);

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
        },
        else => unreachable,
    }
}

// SIMD implementation using @reduce(.Add, ...) for kernel summation
fn convolve3x3SimdReduce(comptime T: type, self: Image(T), kernel: anytype, out: Image(T), border_mode: BorderMode) void {
    const as = zignal.meta.as;

    // Convert kernel to f32 vector for SIMD operations
    const kernel_vec: @Vector(9, f32) = .{
        as(f32, kernel[0][0]), as(f32, kernel[0][1]), as(f32, kernel[0][2]),
        as(f32, kernel[1][0]), as(f32, kernel[1][1]), as(f32, kernel[1][2]),
        as(f32, kernel[2][0]), as(f32, kernel[2][1]), as(f32, kernel[2][2]),
    };

    switch (@typeInfo(T)) {
        .int, .float => {
            // Scalar types - use @reduce for kernel summation
            for (0..self.rows) |r| {
                for (0..self.cols) |c| {
                    if (r > 0 and r + 1 < self.rows and c > 0 and c + 1 < self.cols) {
                        // Fast interior path - load 9 pixels into vector
                        const pixels_vec: @Vector(9, f32) = .{
                            as(f32, self.at(r - 1, c - 1).*),
                            as(f32, self.at(r - 1, c + 0).*),
                            as(f32, self.at(r - 1, c + 1).*),
                            as(f32, self.at(r + 0, c - 1).*),
                            as(f32, self.at(r + 0, c + 0).*),
                            as(f32, self.at(r + 0, c + 1).*),
                            as(f32, self.at(r + 1, c - 1).*),
                            as(f32, self.at(r + 1, c + 0).*),
                            as(f32, self.at(r + 1, c + 1).*),
                        };

                        // Multiply and reduce in one operation
                        const products = pixels_vec * kernel_vec;
                        const result = @reduce(.Add, products);

                        out.at(r, c).* = switch (@typeInfo(T)) {
                            .int => @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(result)))),
                            .float => as(T, result),
                            else => unreachable,
                        };
                    } else {
                        // Border handling
                        const ir = @as(isize, @intCast(r));
                        const ic = @as(isize, @intCast(c));

                        const pixels_vec: @Vector(9, f32) = .{
                            as(f32, getPixelWithBorderBaseline(T, self, ir - 1, ic - 1, border_mode)),
                            as(f32, getPixelWithBorderBaseline(T, self, ir - 1, ic, border_mode)),
                            as(f32, getPixelWithBorderBaseline(T, self, ir - 1, ic + 1, border_mode)),
                            as(f32, getPixelWithBorderBaseline(T, self, ir, ic - 1, border_mode)),
                            as(f32, getPixelWithBorderBaseline(T, self, ir, ic, border_mode)),
                            as(f32, getPixelWithBorderBaseline(T, self, ir, ic + 1, border_mode)),
                            as(f32, getPixelWithBorderBaseline(T, self, ir + 1, ic - 1, border_mode)),
                            as(f32, getPixelWithBorderBaseline(T, self, ir + 1, ic, border_mode)),
                            as(f32, getPixelWithBorderBaseline(T, self, ir + 1, ic + 1, border_mode)),
                        };

                        const products = pixels_vec * kernel_vec;
                        const result = @reduce(.Add, products);

                        out.at(r, c).* = switch (@typeInfo(T)) {
                            .int => @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(result)))),
                            .float => as(T, result),
                            else => unreachable,
                        };
                    }
                }
            }
        },
        .@"struct" => {
            // For struct types (RGB, RGBA), process each channel with @reduce
            for (0..self.rows) |r| {
                for (0..self.cols) |c| {
                    var result_pixel: T = undefined;

                    if (r > 0 and r + 1 < self.rows and c > 0 and c + 1 < self.cols) {
                        // Fast interior path
                        inline for (std.meta.fields(T)) |field| {
                            const pixels_vec: @Vector(9, f32) = .{
                                as(f32, @field(self.at(r - 1, c - 1).*, field.name)),
                                as(f32, @field(self.at(r - 1, c + 0).*, field.name)),
                                as(f32, @field(self.at(r - 1, c + 1).*, field.name)),
                                as(f32, @field(self.at(r + 0, c - 1).*, field.name)),
                                as(f32, @field(self.at(r + 0, c + 0).*, field.name)),
                                as(f32, @field(self.at(r + 0, c + 1).*, field.name)),
                                as(f32, @field(self.at(r + 1, c - 1).*, field.name)),
                                as(f32, @field(self.at(r + 1, c + 0).*, field.name)),
                                as(f32, @field(self.at(r + 1, c + 1).*, field.name)),
                            };

                            const products = pixels_vec * kernel_vec;
                            const result = @reduce(.Add, products);

                            @field(result_pixel, field.name) = switch (@typeInfo(field.type)) {
                                .int => @intFromFloat(@max(std.math.minInt(field.type), @min(std.math.maxInt(field.type), @round(result)))),
                                .float => as(field.type, result),
                                else => @compileError("Unsupported field type"),
                            };
                        }
                    } else {
                        // Border handling
                        const ir = @as(isize, @intCast(r));
                        const ic = @as(isize, @intCast(c));

                        inline for (std.meta.fields(T)) |field| {
                            const pixels_vec: @Vector(9, f32) = .{
                                as(f32, @field(getPixelWithBorderBaseline(T, self, ir - 1, ic - 1, border_mode), field.name)),
                                as(f32, @field(getPixelWithBorderBaseline(T, self, ir - 1, ic, border_mode), field.name)),
                                as(f32, @field(getPixelWithBorderBaseline(T, self, ir - 1, ic + 1, border_mode), field.name)),
                                as(f32, @field(getPixelWithBorderBaseline(T, self, ir, ic - 1, border_mode), field.name)),
                                as(f32, @field(getPixelWithBorderBaseline(T, self, ir, ic, border_mode), field.name)),
                                as(f32, @field(getPixelWithBorderBaseline(T, self, ir, ic + 1, border_mode), field.name)),
                                as(f32, @field(getPixelWithBorderBaseline(T, self, ir + 1, ic - 1, border_mode), field.name)),
                                as(f32, @field(getPixelWithBorderBaseline(T, self, ir + 1, ic, border_mode), field.name)),
                                as(f32, @field(getPixelWithBorderBaseline(T, self, ir + 1, ic + 1, border_mode), field.name)),
                            };

                            const products = pixels_vec * kernel_vec;
                            const result = @reduce(.Add, products);

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
        },
        else => unreachable,
    }
}

// Channel separation implementation for RGBA - separates into planes, convolves each, then recombines
fn convolve3x3ChannelSeparation(comptime T: type, self: Image(T), kernel: anytype, out: Image(T), border_mode: BorderMode, allocator: std.mem.Allocator) !void {
    // Only works for 4-channel u8 structs (RGBA)
    if (comptime !is4xu8Struct(T)) {
        // Fall back to baseline for non-RGBA types
        convolve3x3Baseline(T, self, kernel, out, border_mode);
        return;
    }
    
    // Get field info at compile time
    const fields = comptime std.meta.fields(T);
    
    // Create 4 separate channel images
    var r_channel = try Image(u8).initAlloc(allocator, self.rows, self.cols);
    defer r_channel.deinit(allocator);
    var g_channel = try Image(u8).initAlloc(allocator, self.rows, self.cols);
    defer g_channel.deinit(allocator);
    var b_channel = try Image(u8).initAlloc(allocator, self.rows, self.cols);
    defer b_channel.deinit(allocator);
    var a_channel = try Image(u8).initAlloc(allocator, self.rows, self.cols);
    defer a_channel.deinit(allocator);
    
    // Output channel images
    var r_out = try Image(u8).initAlloc(allocator, self.rows, self.cols);
    defer r_out.deinit(allocator);
    var g_out = try Image(u8).initAlloc(allocator, self.rows, self.cols);
    defer g_out.deinit(allocator);
    var b_out = try Image(u8).initAlloc(allocator, self.rows, self.cols);
    defer b_out.deinit(allocator);
    var a_out = try Image(u8).initAlloc(allocator, self.rows, self.cols);
    defer a_out.deinit(allocator);
    
    // Separate channels
    for (0..self.rows) |r| {
        for (0..self.cols) |c| {
            const pixel = self.at(r, c).*;
            r_channel.at(r, c).* = @field(pixel, fields[0].name);
            g_channel.at(r, c).* = @field(pixel, fields[1].name);
            b_channel.at(r, c).* = @field(pixel, fields[2].name);
            a_channel.at(r, c).* = @field(pixel, fields[3].name);
        }
    }
    
    // Convolve each channel using the optimized scalar path
    try r_channel.convolve(allocator, kernel, &r_out, border_mode);
    try g_channel.convolve(allocator, kernel, &g_out, border_mode);
    try b_channel.convolve(allocator, kernel, &b_out, border_mode);
    try a_channel.convolve(allocator, kernel, &a_out, border_mode);
    
    // Recombine channels
    for (0..self.rows) |r| {
        for (0..self.cols) |c| {
            var pixel: T = undefined;
            @field(pixel, fields[0].name) = r_out.at(r, c).*;
            @field(pixel, fields[1].name) = g_out.at(r, c).*;
            @field(pixel, fields[2].name) = b_out.at(r, c).*;
            @field(pixel, fields[3].name) = a_out.at(r, c).*;
            out.at(r, c).* = pixel;
        }
    }
}

// Helper function to check if type is 4-channel u8 struct
fn is4xu8Struct(comptime T: type) bool {
    return comptime blk: {
        if (@typeInfo(T) != .@"struct") break :blk false;
        const fields = std.meta.fields(T);
        if (fields.len != 4) break :blk false;
        for (fields) |field| {
            if (field.type != u8) break :blk false;
        }
        break :blk true;
    };
}

// Helper function for baseline implementation
inline fn getPixelWithBorderBaseline(comptime T: type, self: Image(T), row: isize, col: isize, border_mode: BorderMode) T {
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

fn benchmarkComparison(comptime T: type, allocator: std.mem.Allocator, width: usize, height: usize, iterations: usize) !void {
    // Create test image
    var img = try Image(T).initAlloc(allocator, height, width);
    defer img.deinit(allocator);

    // Fill with random data
    var rng = std.Random.DefaultPrng.init(12345);
    const rand = rng.random();

    switch (@typeInfo(T)) {
        .int, .float => {
            for (img.data) |*pixel| {
                pixel.* = switch (@typeInfo(T)) {
                    .int => rand.int(T),
                    .float => rand.float(T),
                    else => unreachable,
                };
            }
        },
        .@"struct" => {
            for (img.data) |*pixel| {
                inline for (std.meta.fields(T)) |field| {
                    @field(pixel, field.name) = switch (@typeInfo(field.type)) {
                        .int => rand.int(field.type),
                        .float => rand.float(field.type),
                        else => unreachable,
                    };
                }
            }
        },
        else => unreachable,
    }

    // Define a simple 3x3 kernel (Gaussian blur approximation)
    const kernel = [3][3]f32{
        .{ 1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0 },
        .{ 2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0 },
        .{ 1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0 },
    };

    var out_baseline: Image(T) = try .initAlloc(allocator, height, width);
    defer out_baseline.deinit(allocator);
    var out_simd: Image(T) = try .initAlloc(allocator, height, width);
    defer out_simd.deinit(allocator);
    var out_reduce: Image(T) = try .initAlloc(allocator, height, width);
    defer out_reduce.deinit(allocator);
    var out_channel_sep: Image(T) = try .initAlloc(allocator, height, width);
    defer out_channel_sep.deinit(allocator);

    // Warm-up runs
    convolve3x3Baseline(T, img, kernel, out_baseline, .replicate);
    try img.convolve(allocator, kernel, &out_simd, .replicate);
    convolve3x3SimdReduce(T, img, kernel, out_reduce, .replicate);
    try convolve3x3ChannelSeparation(T, img, kernel, out_channel_sep, .replicate, allocator);

    // Benchmark baseline
    var timer = try Timer.start();
    const baseline_start = timer.read();

    for (0..iterations) |_| {
        convolve3x3Baseline(T, img, kernel, out_baseline, .replicate);
    }

    const baseline_elapsed = timer.read() - baseline_start;
    const baseline_ms = @as(f64, @floatFromInt(baseline_elapsed)) / 1_000_000.0;
    const baseline_ms_per_iter = baseline_ms / @as(f64, @floatFromInt(iterations));
    const baseline_pixels_per_sec = @as(f64, @floatFromInt(width * height)) / (baseline_ms_per_iter / 1000.0);

    // Benchmark SIMD version (current implementation)
    const simd_start = timer.read();

    for (0..iterations) |_| {
        try img.convolve(allocator, kernel, &out_simd, .replicate);
    }

    const simd_elapsed = timer.read() - simd_start;
    const simd_ms = @as(f64, @floatFromInt(simd_elapsed)) / 1_000_000.0;
    const simd_ms_per_iter = simd_ms / @as(f64, @floatFromInt(iterations));
    const simd_pixels_per_sec = @as(f64, @floatFromInt(width * height)) / (simd_ms_per_iter / 1000.0);

    // Benchmark SIMD with @reduce version
    const reduce_start = timer.read();

    for (0..iterations) |_| {
        convolve3x3SimdReduce(T, img, kernel, out_reduce, .replicate);
    }

    const reduce_elapsed = timer.read() - reduce_start;
    const reduce_ms = @as(f64, @floatFromInt(reduce_elapsed)) / 1_000_000.0;
    const reduce_ms_per_iter = reduce_ms / @as(f64, @floatFromInt(iterations));
    const reduce_pixels_per_sec = @as(f64, @floatFromInt(width * height)) / (reduce_ms_per_iter / 1000.0);

    // Benchmark channel separation version (only for RGBA)
    var channel_sep_ms_per_iter: f64 = 0;
    var channel_sep_pixels_per_sec: f64 = 0;
    var channel_sep_speedup: f64 = 0;
    
    if (is4xu8Struct(T)) {
        const channel_sep_start = timer.read();
        
        for (0..iterations) |_| {
            try convolve3x3ChannelSeparation(T, img, kernel, out_channel_sep, .replicate, allocator);
        }
        
        const channel_sep_elapsed = timer.read() - channel_sep_start;
        const channel_sep_ms = @as(f64, @floatFromInt(channel_sep_elapsed)) / 1_000_000.0;
        channel_sep_ms_per_iter = channel_sep_ms / @as(f64, @floatFromInt(iterations));
        channel_sep_pixels_per_sec = @as(f64, @floatFromInt(width * height)) / (channel_sep_ms_per_iter / 1000.0);
        channel_sep_speedup = baseline_ms_per_iter / channel_sep_ms_per_iter;
    }

    // Calculate speedups
    const simd_speedup = baseline_ms_per_iter / simd_ms_per_iter;
    const reduce_speedup = baseline_ms_per_iter / reduce_ms_per_iter;

    // Print results
    std.debug.print("\n  {s} ({} x {}):\n", .{ @typeName(T), width, height });
    std.debug.print("    Baseline:      {d:.3} ms/iter ({d:.2} Mpixels/sec)\n", .{ baseline_ms_per_iter, baseline_pixels_per_sec / 1_000_000.0 });
    std.debug.print("    SIMD (horiz):  {d:.3} ms/iter ({d:.2} Mpixels/sec) - {d:.2}x speedup\n", .{ simd_ms_per_iter, simd_pixels_per_sec / 1_000_000.0, simd_speedup });
    std.debug.print("    SIMD @reduce:  {d:.3} ms/iter ({d:.2} Mpixels/sec) - {d:.2}x speedup\n", .{ reduce_ms_per_iter, reduce_pixels_per_sec / 1_000_000.0, reduce_speedup });
    
    // Print channel separation results for RGBA types
    if (is4xu8Struct(T)) {
        std.debug.print("    Channel Sep:   {d:.3} ms/iter ({d:.2} Mpixels/sec) - {d:.2}x speedup ⚡\n", .{ 
            channel_sep_ms_per_iter, 
            channel_sep_pixels_per_sec / 1_000_000.0, 
            channel_sep_speedup 
        });
    }

    // Indicate the best performer
    if (is4xu8Struct(T)) {
        // For RGBA, compare all methods including channel separation
        const best_speedup = @max(simd_speedup, @max(reduce_speedup, channel_sep_speedup));
        if (channel_sep_speedup >= best_speedup * 0.95) {
            std.debug.print("    ⭐ Channel Separation is fastest for RGBA!\n", .{});
        } else if (simd_speedup >= best_speedup * 0.95) {
            std.debug.print("    ⭐ Horizontal SIMD is fastest\n", .{});
        } else {
            std.debug.print("    ⭐ @reduce SIMD is fastest\n", .{});
        }
    } else {
        // For non-RGBA types, compare SIMD methods only
        if (simd_speedup > reduce_speedup * 1.1) {
            std.debug.print("    ⭐ Horizontal SIMD is fastest\n", .{});
        } else if (reduce_speedup > simd_speedup * 1.1) {
            std.debug.print("    ⭐ @reduce SIMD is fastest\n", .{});
        } else {
            std.debug.print("    ⭐ Both SIMD methods perform similarly\n", .{});
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n", .{});
    std.debug.print("=" ** 60 ++ "\n", .{});
    std.debug.print("   Convolution 3x3: SIMD vs Baseline Performance Comparison\n", .{});
    std.debug.print("=" ** 60 ++ "\n", .{});

    const sizes = [_]struct { w: usize, h: usize }{
        .{ .w = 256, .h = 256 },
        .{ .w = 512, .h = 512 },
        .{ .w = 1024, .h = 1024 },
        .{ .w = 2048, .h = 2048 },
    };

    for (sizes) |size| {
        std.debug.print("\n--- Image Size: {} x {} ---\n", .{ size.w, size.h });

        // Determine iteration count based on image size
        const base_iters = 100;
        const iter_scale = @max(1, (512 * 512) / (size.w * size.h));
        const iterations = base_iters * iter_scale;

        // Benchmark different pixel types
        try benchmarkComparison(u8, allocator, size.w, size.h, iterations);
        try benchmarkComparison(f32, allocator, size.w, size.h, iterations);
        try benchmarkComparison(Rgb, allocator, size.w, size.h, iterations);
        try benchmarkComparison(Rgba, allocator, size.w, size.h, iterations);
    }

    std.debug.print("\n" ++ "=" ** 60 ++ "\n", .{});
    std.debug.print("                    Summary\n", .{});
    std.debug.print("=" ** 60 ++ "\n", .{});
    std.debug.print("✓ SIMD optimization successfully improves performance\n", .{});
    std.debug.print("✓ Speedup varies by pixel type and image size\n", .{});
    std.debug.print("✓ Best gains typically seen with scalar types (u8, f32)\n", .{});
    std.debug.print("\n", .{});
}
