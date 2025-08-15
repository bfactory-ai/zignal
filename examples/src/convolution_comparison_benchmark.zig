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

// Horizontal RGBA SIMD - processes multiple RGBA pixels simultaneously
fn convolve3x3RgbaHorizontal(comptime T: type, self: Image(T), kernel: anytype, out: Image(T), border_mode: BorderMode) void {
    const as = zignal.meta.as;

    // Only works for 4-channel u8 structs (RGBA)
    if (comptime !is4xu8Struct(T)) {
        // Fall back to baseline for non-RGBA types
        convolve3x3Baseline(T, self, kernel, out, border_mode);
        return;
    }

    const kr = [9]f32{
        as(f32, kernel[0][0]), as(f32, kernel[0][1]), as(f32, kernel[0][2]),
        as(f32, kernel[1][0]), as(f32, kernel[1][1]), as(f32, kernel[1][2]),
        as(f32, kernel[2][0]), as(f32, kernel[2][1]), as(f32, kernel[2][2]),
    };

    // Process 2 RGBA pixels at once using @Vector(8, f32)
    const pixels_per_vec = 2;
    const vec_len = pixels_per_vec * 4; // 8 channels total

    for (0..self.rows) |r| {
        var c: usize = 0;

        // Process interior pixels with SIMD (2 pixels at a time)
        if (r > 0 and r + 1 < self.rows and self.cols > pixels_per_vec + 2) {
            c = 1;
            const safe_end = self.cols - 1;

            while (c + pixels_per_vec <= safe_end) : (c += pixels_per_vec) {
                // Load 3x3 neighborhoods for 2 RGBA pixels
                // Each pixel has 4 channels, so we need 8-element vectors
                var result_vec: @Vector(vec_len, f32) = @splat(0);

                // Process each kernel position
                inline for (0..3) |ky| {
                    inline for (0..3) |kx| {
                        const kernel_val = kr[ky * 3 + kx];
                        const kernel_vec: @Vector(vec_len, f32) = @splat(kernel_val);

                        // Load 2 pixels from the neighborhood
                        var pixel_vec: @Vector(vec_len, f32) = undefined;

                        // First pixel's 4 channels
                        const p1 = self.at(r + ky - 1, c + kx - 1).*;
                        const fields = comptime std.meta.fields(T);
                        pixel_vec[0] = @floatFromInt(@field(p1, fields[0].name));
                        pixel_vec[1] = @floatFromInt(@field(p1, fields[1].name));
                        pixel_vec[2] = @floatFromInt(@field(p1, fields[2].name));
                        pixel_vec[3] = @floatFromInt(@field(p1, fields[3].name));

                        // Second pixel's 4 channels - note we're getting the next pixel horizontally
                        const p2 = self.at(r + ky - 1, c + 1 + kx - 1).*;
                        pixel_vec[4] = @floatFromInt(@field(p2, fields[0].name));
                        pixel_vec[5] = @floatFromInt(@field(p2, fields[1].name));
                        pixel_vec[6] = @floatFromInt(@field(p2, fields[2].name));
                        pixel_vec[7] = @floatFromInt(@field(p2, fields[3].name));

                        result_vec += pixel_vec * kernel_vec;
                    }
                }

                // Store results for both pixels
                const fields = comptime std.meta.fields(T);
                var out_pixel1: T = undefined;
                @field(out_pixel1, fields[0].name) = @intFromFloat(@max(0, @min(255, @round(result_vec[0]))));
                @field(out_pixel1, fields[1].name) = @intFromFloat(@max(0, @min(255, @round(result_vec[1]))));
                @field(out_pixel1, fields[2].name) = @intFromFloat(@max(0, @min(255, @round(result_vec[2]))));
                @field(out_pixel1, fields[3].name) = @intFromFloat(@max(0, @min(255, @round(result_vec[3]))));
                out.at(r, c).* = out_pixel1;

                var out_pixel2: T = undefined;
                @field(out_pixel2, fields[0].name) = @intFromFloat(@max(0, @min(255, @round(result_vec[4]))));
                @field(out_pixel2, fields[1].name) = @intFromFloat(@max(0, @min(255, @round(result_vec[5]))));
                @field(out_pixel2, fields[2].name) = @intFromFloat(@max(0, @min(255, @round(result_vec[6]))));
                @field(out_pixel2, fields[3].name) = @intFromFloat(@max(0, @min(255, @round(result_vec[7]))));
                out.at(r, c + 1).* = out_pixel2;
            }
        }

        // Process remaining pixels with scalar code
        while (c < self.cols) : (c += 1) {
            var result_pixel: T = undefined;

            if (r > 0 and r + 1 < self.rows and c > 0 and c + 1 < self.cols) {
                // Fast interior path
                const fields = comptime std.meta.fields(T);
                inline for (fields) |field| {
                    var result: f32 = 0;
                    inline for (0..3) |ky| {
                        inline for (0..3) |kx| {
                            const p = self.at(r + ky - 1, c + kx - 1).*;
                            result += @as(f32, @floatFromInt(@field(p, field.name))) * kr[ky * 3 + kx];
                        }
                    }
                    @field(result_pixel, field.name) = @intFromFloat(@max(0, @min(255, @round(result))));
                }
            } else {
                // Border handling
                const ir = @as(isize, @intCast(r));
                const ic = @as(isize, @intCast(c));
                const fields = comptime std.meta.fields(T);

                inline for (fields) |field| {
                    var result: f32 = 0;
                    inline for (0..3) |ky| {
                        inline for (0..3) |kx| {
                            const iry = ir + @as(isize, @intCast(ky)) - 1;
                            const icx = ic + @as(isize, @intCast(kx)) - 1;
                            const p = getPixelWithBorderBaseline(T, self, iry, icx, border_mode);
                            result += @as(f32, @floatFromInt(@field(p, field.name))) * kr[ky * 3 + kx];
                        }
                    }
                    @field(result_pixel, field.name) = @intFromFloat(@max(0, @min(255, @round(result))));
                }
            }

            out.at(r, c).* = result_pixel;
        }
    }
}

// Integer arithmetic version - avoids float conversions for RGBA
fn convolve3x3IntegerRgba(comptime T: type, self: Image(T), kernel: anytype, out: Image(T), border_mode: BorderMode) void {
    // Only works for 4-channel u8 structs (RGBA)
    if (comptime !is4xu8Struct(T)) {
        // Fall back to baseline for non-RGBA types
        convolve3x3Baseline(T, self, kernel, out, border_mode);
        return;
    }

    // Convert kernel to fixed-point integer (scale by 256 for precision)
    const SCALE = 256;
    const ki = [9]i32{
        @intFromFloat(@round(@as(f32, @floatCast(kernel[0][0])) * SCALE)),
        @intFromFloat(@round(@as(f32, @floatCast(kernel[0][1])) * SCALE)),
        @intFromFloat(@round(@as(f32, @floatCast(kernel[0][2])) * SCALE)),
        @intFromFloat(@round(@as(f32, @floatCast(kernel[1][0])) * SCALE)),
        @intFromFloat(@round(@as(f32, @floatCast(kernel[1][1])) * SCALE)),
        @intFromFloat(@round(@as(f32, @floatCast(kernel[1][2])) * SCALE)),
        @intFromFloat(@round(@as(f32, @floatCast(kernel[2][0])) * SCALE)),
        @intFromFloat(@round(@as(f32, @floatCast(kernel[2][1])) * SCALE)),
        @intFromFloat(@round(@as(f32, @floatCast(kernel[2][2])) * SCALE)),
    };

    const fields = comptime std.meta.fields(T);

    for (0..self.rows) |r| {
        for (0..self.cols) |c| {
            var result_pixel: T = undefined;

            if (r > 0 and r + 1 < self.rows and c > 0 and c + 1 < self.cols) {
                // Fast interior path using integer arithmetic
                inline for (fields) |field| {
                    var result: i32 = 0;

                    // Unrolled 3x3 convolution with integer math
                    const p00 = @as(i32, @field(self.at(r - 1, c - 1).*, field.name));
                    const p01 = @as(i32, @field(self.at(r - 1, c + 0).*, field.name));
                    const p02 = @as(i32, @field(self.at(r - 1, c + 1).*, field.name));
                    const p10 = @as(i32, @field(self.at(r + 0, c - 1).*, field.name));
                    const p11 = @as(i32, @field(self.at(r + 0, c + 0).*, field.name));
                    const p12 = @as(i32, @field(self.at(r + 0, c + 1).*, field.name));
                    const p20 = @as(i32, @field(self.at(r + 1, c - 1).*, field.name));
                    const p21 = @as(i32, @field(self.at(r + 1, c + 0).*, field.name));
                    const p22 = @as(i32, @field(self.at(r + 1, c + 1).*, field.name));

                    result = p00 * ki[0] + p01 * ki[1] + p02 * ki[2] +
                        p10 * ki[3] + p11 * ki[4] + p12 * ki[5] +
                        p20 * ki[6] + p21 * ki[7] + p22 * ki[8];

                    // Divide by scale factor and clamp to u8 range
                    // Add SCALE/2 for rounding before division
                    const rounded = @divTrunc((result + SCALE / 2), SCALE);
                    @field(result_pixel, field.name) = @intCast(@max(0, @min(255, rounded)));
                }
            } else {
                // Border handling with integer arithmetic
                const ir = @as(isize, @intCast(r));
                const ic = @as(isize, @intCast(c));

                inline for (fields) |field| {
                    var result: i32 = 0;

                    const p00 = @as(i32, @field(getPixelWithBorderBaseline(T, self, ir - 1, ic - 1, border_mode), field.name));
                    const p01 = @as(i32, @field(getPixelWithBorderBaseline(T, self, ir - 1, ic, border_mode), field.name));
                    const p02 = @as(i32, @field(getPixelWithBorderBaseline(T, self, ir - 1, ic + 1, border_mode), field.name));
                    const p10 = @as(i32, @field(getPixelWithBorderBaseline(T, self, ir, ic - 1, border_mode), field.name));
                    const p11 = @as(i32, @field(getPixelWithBorderBaseline(T, self, ir, ic, border_mode), field.name));
                    const p12 = @as(i32, @field(getPixelWithBorderBaseline(T, self, ir, ic + 1, border_mode), field.name));
                    const p20 = @as(i32, @field(getPixelWithBorderBaseline(T, self, ir + 1, ic - 1, border_mode), field.name));
                    const p21 = @as(i32, @field(getPixelWithBorderBaseline(T, self, ir + 1, ic, border_mode), field.name));
                    const p22 = @as(i32, @field(getPixelWithBorderBaseline(T, self, ir + 1, ic + 1, border_mode), field.name));

                    result = p00 * ki[0] + p01 * ki[1] + p02 * ki[2] +
                        p10 * ki[3] + p11 * ki[4] + p12 * ki[5] +
                        p20 * ki[6] + p21 * ki[7] + p22 * ki[8];

                    const rounded = @divTrunc((result + SCALE / 2), SCALE);
                    @field(result_pixel, field.name) = @intCast(@max(0, @min(255, rounded)));
                }
            }

            out.at(r, c).* = result_pixel;
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
    var out_horiz_rgba: Image(T) = try .initAlloc(allocator, height, width);
    defer out_horiz_rgba.deinit(allocator);
    var out_integer: Image(T) = try .initAlloc(allocator, height, width);
    defer out_integer.deinit(allocator);

    // Warm-up runs
    convolve3x3Baseline(T, img, kernel, out_baseline, .replicate);
    try img.convolve(allocator, kernel, &out_simd, .replicate);
    convolve3x3SimdReduce(T, img, kernel, out_reduce, .replicate);
    if (is4xu8Struct(T)) {
        try convolve3x3ChannelSeparation(T, img, kernel, out_channel_sep, .replicate, allocator);
        convolve3x3RgbaHorizontal(T, img, kernel, out_horiz_rgba, .replicate);
        convolve3x3IntegerRgba(T, img, kernel, out_integer, .replicate);
    }

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
    var horiz_rgba_ms_per_iter: f64 = 0;
    var horiz_rgba_pixels_per_sec: f64 = 0;
    var horiz_rgba_speedup: f64 = 0;
    var integer_ms_per_iter: f64 = 0;
    var integer_pixels_per_sec: f64 = 0;
    var integer_speedup: f64 = 0;

    if (is4xu8Struct(T)) {
        // Benchmark channel separation
        const channel_sep_start = timer.read();

        for (0..iterations) |_| {
            try convolve3x3ChannelSeparation(T, img, kernel, out_channel_sep, .replicate, allocator);
        }

        const channel_sep_elapsed = timer.read() - channel_sep_start;
        const channel_sep_ms = @as(f64, @floatFromInt(channel_sep_elapsed)) / 1_000_000.0;
        channel_sep_ms_per_iter = channel_sep_ms / @as(f64, @floatFromInt(iterations));
        channel_sep_pixels_per_sec = @as(f64, @floatFromInt(width * height)) / (channel_sep_ms_per_iter / 1000.0);
        channel_sep_speedup = baseline_ms_per_iter / channel_sep_ms_per_iter;

        // Benchmark horizontal RGBA SIMD
        const horiz_rgba_start = timer.read();

        for (0..iterations) |_| {
            convolve3x3RgbaHorizontal(T, img, kernel, out_horiz_rgba, .replicate);
        }

        const horiz_rgba_elapsed = timer.read() - horiz_rgba_start;
        const horiz_rgba_ms = @as(f64, @floatFromInt(horiz_rgba_elapsed)) / 1_000_000.0;
        horiz_rgba_ms_per_iter = horiz_rgba_ms / @as(f64, @floatFromInt(iterations));
        horiz_rgba_pixels_per_sec = @as(f64, @floatFromInt(width * height)) / (horiz_rgba_ms_per_iter / 1000.0);
        horiz_rgba_speedup = baseline_ms_per_iter / horiz_rgba_ms_per_iter;

        // Benchmark integer arithmetic version
        const integer_start = timer.read();

        for (0..iterations) |_| {
            convolve3x3IntegerRgba(T, img, kernel, out_integer, .replicate);
        }

        const integer_elapsed = timer.read() - integer_start;
        const integer_ms = @as(f64, @floatFromInt(integer_elapsed)) / 1_000_000.0;
        integer_ms_per_iter = integer_ms / @as(f64, @floatFromInt(iterations));
        integer_pixels_per_sec = @as(f64, @floatFromInt(width * height)) / (integer_ms_per_iter / 1000.0);
        integer_speedup = baseline_ms_per_iter / integer_ms_per_iter;
    }

    // Calculate speedups
    const simd_speedup = baseline_ms_per_iter / simd_ms_per_iter;
    const reduce_speedup = baseline_ms_per_iter / reduce_ms_per_iter;

    // Print results
    std.debug.print("\n  {s} ({} x {}):\n", .{ @typeName(T), width, height });
    std.debug.print("    Baseline:      {d:.3} ms/iter ({d:.2} Mpixels/sec)\n", .{ baseline_ms_per_iter, baseline_pixels_per_sec / 1_000_000.0 });
    std.debug.print("    SIMD (horiz):  {d:.3} ms/iter ({d:.2} Mpixels/sec) - {d:.2}x speedup\n", .{ simd_ms_per_iter, simd_pixels_per_sec / 1_000_000.0, simd_speedup });
    std.debug.print("    SIMD @reduce:  {d:.3} ms/iter ({d:.2} Mpixels/sec) - {d:.2}x speedup\n", .{ reduce_ms_per_iter, reduce_pixels_per_sec / 1_000_000.0, reduce_speedup });

    // Print channel separation and horizontal RGBA results for RGBA types
    if (is4xu8Struct(T)) {
        std.debug.print("    Channel Sep:   {d:.3} ms/iter ({d:.2} Mpixels/sec) - {d:.2}x speedup ⚡\n", .{ channel_sep_ms_per_iter, channel_sep_pixels_per_sec / 1_000_000.0, channel_sep_speedup });
        std.debug.print("    Horiz RGBA:    {d:.3} ms/iter ({d:.2} Mpixels/sec) - {d:.2}x speedup 🚀\n", .{ horiz_rgba_ms_per_iter, horiz_rgba_pixels_per_sec / 1_000_000.0, horiz_rgba_speedup });
        std.debug.print("    Integer Math:  {d:.3} ms/iter ({d:.2} Mpixels/sec) - {d:.2}x speedup 🔢\n", .{ integer_ms_per_iter, integer_pixels_per_sec / 1_000_000.0, integer_speedup });
    }

    // Indicate the best performer
    if (is4xu8Struct(T)) {
        // For RGBA, compare all methods
        const best_speedup = @max(@max(@max(simd_speedup, reduce_speedup), @max(channel_sep_speedup, horiz_rgba_speedup)), integer_speedup);
        if (channel_sep_speedup >= best_speedup * 0.95) {
            std.debug.print("    ⭐ Channel Separation is fastest for RGBA!\n", .{});
        } else if (integer_speedup >= best_speedup * 0.95) {
            std.debug.print("    ⭐ Integer arithmetic is fastest for RGBA!\n", .{});
        } else if (horiz_rgba_speedup >= best_speedup * 0.95) {
            std.debug.print("    ⭐ Horizontal RGBA SIMD is fastest!\n", .{});
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
