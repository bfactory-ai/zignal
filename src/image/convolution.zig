const std = @import("std");
const Allocator = std.mem.Allocator;

const Gray = @import("../color.zig").Gray;
const Image = @import("../image.zig").Image;
const meta = @import("../meta.zig");
const as = meta.as;
const channel_ops = @import("channel_ops.zig");
const border = @import("border.zig");

/// Border handling modes for filter operations
pub const BorderMode = border.BorderMode;

fn absI64(x: i64) i64 {
    return if (x >= 0) x else -x;
}

fn sumAbsI64NoOverflow(values: anytype) struct { sum: i64, overflow: bool } {
    var sum: i64 = 0;
    for (values) |value| {
        const addend = absI64(@as(i64, value));
        const new_sum, const overflow = @addWithOverflow(sum, addend);
        if (overflow != 0) return .{ .sum = new_sum, .overflow = true };
        sum = new_sum;
    }
    return .{ .sum = sum, .overflow = false };
}

fn convolveStructU8(
    comptime T: type,
    comptime Scalar: type,
    comptime kernel_height: usize,
    comptime kernel_width: usize,
    self: Image(T),
    allocator: Allocator,
    kernel_flat: [kernel_height * kernel_width]Scalar,
    border_mode: BorderMode,
    out: Image(T),
) !void {
    const Kernel = ConvolutionKernel(u8, Scalar, kernel_height, kernel_width);

    var kernel_sum: Scalar = 0;
    for (kernel_flat) |weight| kernel_sum += weight;

    const Pixel = PixelIO(u8, Scalar, 1);
    const scale: Scalar = @as(Scalar, Pixel.scale);
    const plane_size = self.rows * self.cols;

    // Separate channels using helper
    const split = try channel_ops.splitChannelsWithUniform(T, self, allocator);
    const channels = split.channels;
    const uniforms = split.uniforms;
    defer for (channels) |channel| allocator.free(channel);

    const ChannelStrategy = enum { normalized, scaled, non_uniform };
    var strategies: [channels.len]ChannelStrategy = undefined;

    inline for (uniforms, 0..) |uniform_value, i| {
        if (uniform_value != null and border_mode != .zero) {
            strategies[i] = if (kernel_sum == scale) .normalized else .scaled;
        } else {
            strategies[i] = .non_uniform;
        }
    }

    // Allocate output planes for strategies that require storage
    var out_channels: [channels.len][]u8 = undefined;
    inline for (&out_channels, strategies, uniforms) |*out_ch, strategy, uniform_value| {
        switch (strategy) {
            .normalized => out_ch.* = &[_]u8{},
            .scaled, .non_uniform => out_ch.* = try allocator.alloc(u8, plane_size),
        }
        if (strategy == .scaled) {
            const value = uniform_value orelse unreachable;
            const accum = @as(Scalar, @intCast(value)) * kernel_sum;
            const stored = Pixel.store(accum);
            @memset(out_ch.*, stored);
        }
    }
    defer {
        inline for (out_channels, strategies) |ch, strategy| {
            switch (strategy) {
                .normalized => {},
                .scaled, .non_uniform => if (ch.len > 0) allocator.free(ch),
            }
        }
    }

    // Convolve only non-uniform channels
    inline for (channels, out_channels, strategies) |src_data, dst_data, strategy| {
        if (strategy == .non_uniform) {
            const src_plane: Image(u8) = .{ .rows = self.rows, .cols = self.cols, .stride = self.cols, .data = src_data };
            const dst_plane: Image(u8) = .{ .rows = self.rows, .cols = self.cols, .stride = self.cols, .data = dst_data };
            Kernel.convolve(src_plane, dst_plane, kernel_flat, border_mode);
        }
    }

    // Recombine channels, using original values for uniform channels
    var final_channels: [channels.len][]const u8 = undefined;
    inline for (strategies, out_channels, channels, 0..) |strategy, out_ch, src_ch, i| {
        switch (strategy) {
            .normalized => final_channels[i] = src_ch,
            .scaled, .non_uniform => final_channels[i] = out_ch,
        }
    }
    channel_ops.mergeChannels(T, final_channels, out);
}

fn rangesOverlapBytes(a_ptr: [*]const u8, a_len: usize, b_ptr: [*]const u8, b_len: usize) bool {
    const a_start = @intFromPtr(a_ptr);
    const b_start = @intFromPtr(b_ptr);
    const a_end, const a_overflow = @addWithOverflow(a_start, a_len);
    const b_end, const b_overflow = @addWithOverflow(b_start, b_len);
    if (a_overflow != 0 or b_overflow != 0) return true;
    return a_start < b_end and b_start < a_end;
}

fn imagesOverlap(comptime T: type, a: Image(T), b: Image(T)) bool {
    const byte_len_a = a.data.len * @sizeOf(T);
    const byte_len_b = b.data.len * @sizeOf(T);
    return rangesOverlapBytes(@ptrCast(a.data.ptr), byte_len_a, @ptrCast(b.data.ptr), byte_len_b);
}

/// Pixel I/O operations for type-specific convolution.
/// Provides unified load/store operations for both u8 (with integer scaling) and f32.
fn PixelIO(comptime T: type, comptime Scalar: type, comptime vec_len: usize) type {
    if (T != u8 and T != f32) @compileError("PixelIO only supports u8 and f32 pixel types");
    if (T == u8 and Scalar != i32 and Scalar != i64) @compileError("PixelIO(u8) only supports i32/i64 scalars");
    if (T == f32 and Scalar != f32) @compileError("PixelIO(f32) only supports f32 scalars");

    return struct {
        const scale = if (T == u8) 256 else 1;

        inline fn load(value: T) Scalar {
            return if (T == u8) @as(Scalar, value) else value;
        }

        inline fn loadVec(src: []const T, offset: usize) @Vector(vec_len, Scalar) {
            if (T == u8) {
                const u8_vec: @Vector(vec_len, u8) = src[offset..][0..vec_len].*;
                return @intCast(u8_vec);
            } else {
                return src[offset..][0..vec_len].*;
            }
        }

        inline fn store(accum: Scalar) T {
            if (T == u8) {
                const rounded = @divTrunc(accum + scale / 2, scale);
                return meta.clamp(u8, rounded);
            } else {
                return accum;
            }
        }

        inline fn storeVec(accum_vec: @Vector(vec_len, Scalar), dst: []T, offset: usize) void {
            if (T == u8) {
                const half_scale_vec: @Vector(vec_len, Scalar) = @splat(scale / 2);
                const scale_vec: @Vector(vec_len, Scalar) = @splat(scale);
                const rounded_vec = @divTrunc(accum_vec + half_scale_vec, scale_vec);

                inline for (0..vec_len) |i| {
                    dst[offset + i] = meta.clamp(u8, rounded_vec[i]);
                }
            } else {
                inline for (0..vec_len) |i| {
                    dst[offset + i] = accum_vec[i];
                }
            }
        }
    };
}

/// Comptime function generator for specialized convolution implementations.
/// Generates optimized code for specific kernel dimensions at compile time.
fn ConvolutionKernel(comptime T: type, comptime Scalar: type, comptime rows: usize, comptime cols: usize) type {
    if (T != u8 and T != f32) {
        @compileError("Unsupported kernel type: " ++ @typeName(T) ++ ". Only u8 and f32 are supported");
    }
    if (T == u8 and Scalar != i32 and Scalar != i64) {
        @compileError("ConvolutionKernel(u8) only supports i32/i64 scalars");
    }
    if (T == f32 and Scalar != f32) {
        @compileError("ConvolutionKernel(f32) only supports f32 scalars");
    }

    return struct {
        const size = rows * cols;
        const half_h = rows / 2;
        const half_w = cols / 2;

        const vec_len = std.simd.suggestVectorLength(Scalar) orelse 1;

        // Use the shared pixel I/O operations
        const Pixels = PixelIO(T, Scalar, vec_len);

        /// Flatten a 2D kernel to 1D array and scale to integer for u8.
        pub fn flatten(kernel: anytype) [size]Scalar {
            const kernel_info = @typeInfo(@TypeOf(kernel));
            const kernel_height = kernel_info.array.len;
            const kernel_width = @typeInfo(kernel_info.array.child).array.len;
            var result: [size]Scalar = undefined;
            var idx: usize = 0;
            inline for (0..kernel_height) |kr| {
                inline for (0..kernel_width) |kx| {
                    const val = as(f32, kernel[kr][kx]);
                    result[idx] = if (T == u8)
                        @intFromFloat(@round(val * 256.0))
                    else
                        val;
                    idx += 1;
                }
            }
            return result;
        }

        fn convolvePixelWithBorder(src: Image(T), dst: Image(T), r: usize, c: usize, kernel: [size]Scalar, border_mode: BorderMode) void {
            const ir = @as(isize, @intCast(r));
            const ic = @as(isize, @intCast(c));
            var result: Scalar = 0;
            inline for (0..rows) |ky| {
                inline for (0..cols) |kx| {
                    const iry = ir + @as(isize, @intCast(ky)) - @as(isize, @intCast(half_h));
                    const icx = ic + @as(isize, @intCast(kx)) - @as(isize, @intCast(half_w));
                    const pixel_val = getPixelScalar(T, Scalar, src, iry, icx, border_mode);
                    result += pixel_val * kernel[ky * cols + kx];
                }
            }
            dst.data[r * dst.stride + c] = Pixels.store(result);
        }

        fn convolve(src: Image(T), dst: Image(T), kernel: [size]Scalar, border_mode: BorderMode) void {
            // Pre-create kernel vectors for SIMD (for f32)
            var kernel_vecs: [size]@Vector(vec_len, Scalar) = undefined;
            if (T == f32) {
                inline for (0..size) |i| {
                    kernel_vecs[i] = @splat(kernel[i]);
                }
            }

            for (0..src.rows) |r| {
                var c: usize = 0;

                // SIMD path for interior pixels
                if (r >= half_h and r + half_h < src.rows and src.cols >= vec_len + 2 * half_w) {
                    // Process leading border pixels of this row
                    while (c < half_w) : (c += 1) {
                        convolvePixelWithBorder(src, dst, r, c, kernel, border_mode);
                    }

                    const safe_end = src.cols - half_w;

                    while (c + vec_len <= safe_end) : (c += vec_len) {
                        var result_vec: @Vector(vec_len, Scalar) = @splat(0);

                        // Unroll kernel application for known sizes
                        inline for (0..rows) |ky| {
                            inline for (0..cols) |kx| {
                                const kid = ky * cols + kx;

                                const kernel_vec = if (T == f32)
                                    kernel_vecs[kid]
                                else
                                    @as(@Vector(vec_len, Scalar), @splat(kernel[kid]));

                                const src_r = r + ky - half_h;
                                const src_c = c + kx - half_w;
                                const src_idx = src_r * src.stride + src_c;
                                const pixel_vec = Pixels.loadVec(src.data, src_idx);
                                result_vec += pixel_vec * kernel_vec;
                            }
                        }

                        Pixels.storeVec(result_vec, dst.data, r * dst.stride + c);
                    }
                }

                // Scalar path for remaining pixels
                while (c < src.cols) : (c += 1) {
                    if (r >= half_h and r + half_h < src.rows and c >= half_w and c + half_w < src.cols) {
                        // Interior pixel - no border handling needed
                        var result: Scalar = 0;
                        inline for (0..rows) |ky| {
                            inline for (0..cols) |kx| {
                                const src_r = r + ky - half_h;
                                const src_c = c + kx - half_w;
                                const pixel_val = Pixels.load(src.data[src_r * src.stride + src_c]);
                                result += pixel_val * kernel[ky * cols + kx];
                            }
                        }
                        dst.data[r * dst.stride + c] = Pixels.store(result);
                    } else {
                        // Border pixel - needs border handling
                        convolvePixelWithBorder(src, dst, r, c, kernel, border_mode);
                    }
                }
            }
        }
    };
}

/// Applies a 2D convolution with the given kernel to the image.
///
/// Parameters:
/// - `allocator`: Used for temporary buffers (e.g. when `self` and `out` overlap, or for struct channel splitting).
/// - `kernel`: A 2D array representing the convolution kernel.
/// - `out`: Destination image (must have the same shape as `self`).
/// - `border_mode`: How to handle pixels at the image borders.
pub fn convolve(comptime T: type, self: Image(T), allocator: Allocator, kernel: anytype, border_mode: BorderMode, out: Image(T)) !void {
    if (!self.hasSameShape(out)) return error.ShapeMismatch;
    if (imagesOverlap(T, self, out)) {
        var temp = try Image(T).initLike(allocator, out);
        defer temp.deinit(allocator);
        try convolve(T, self, allocator, kernel, border_mode, temp);
        temp.copy(out);
        return;
    }

    const kernel_info = @typeInfo(@TypeOf(kernel));
    if (kernel_info != .array) @compileError("Kernel must be a 2D array");
    const outer_array = kernel_info.array;
    if (@typeInfo(outer_array.child) != .array) @compileError("Kernel must be a 2D array");

    const kernel_height = outer_array.len;
    const kernel_width = @typeInfo(outer_array.child).array.len;
    if (kernel_height == 0 or kernel_width == 0) @compileError("Kernel must be non-empty");
    if (kernel_height % 2 == 0 or kernel_width % 2 == 0) @compileError("Kernel dimensions must be odd");

    switch (T) {
        u8 => {
            const Kernel64 = ConvolutionKernel(u8, i64, kernel_height, kernel_width);
            const flat64 = Kernel64.flatten(kernel);

            const sum_abs = sumAbsI64NoOverflow(flat64);
            const max_accum, const overflow = @mulWithOverflow(sum_abs.sum, 255);

            if (!sum_abs.overflow and overflow == 0 and max_accum <= std.math.maxInt(i32)) {
                const Kernel32 = ConvolutionKernel(u8, i32, kernel_height, kernel_width);
                var flat32: [kernel_height * kernel_width]i32 = undefined;
                for (flat64, 0..) |w, i| flat32[i] = @intCast(w);
                Kernel32.convolve(self, out, flat32, border_mode);
            } else {
                Kernel64.convolve(self, out, flat64, border_mode);
            }
        },
        f32 => {
            const Kernel = ConvolutionKernel(f32, f32, kernel_height, kernel_width);
            const flat_kernel = Kernel.flatten(kernel);
            Kernel.convolve(self, out, flat_kernel, border_mode);
        },
        else => switch (@typeInfo(T)) {
            .@"struct" => {
                // Optimized path for u8 structs (RGB, RGBA, etc.)
                if (comptime meta.allFieldsAreU8(T)) {
                    const Kernel64 = ConvolutionKernel(u8, i64, kernel_height, kernel_width);
                    const kernel64 = Kernel64.flatten(kernel);

                    const sum_abs = sumAbsI64NoOverflow(kernel64);
                    const max_accum, const overflow = @mulWithOverflow(sum_abs.sum, 255);
                    const use_i32 = !sum_abs.overflow and overflow == 0 and max_accum <= std.math.maxInt(i32);

                    if (use_i32) {
                        var kernel32: [kernel_height * kernel_width]i32 = undefined;
                        for (kernel64, 0..) |w, i| kernel32[i] = @intCast(w);
                        try convolveStructU8(T, i32, kernel_height, kernel_width, self, allocator, kernel32, border_mode, out);
                    } else {
                        try convolveStructU8(T, i64, kernel_height, kernel_width, self, allocator, kernel64, border_mode, out);
                    }
                } else {
                    @compileError("Convolution only supports structs where all fields are u8. Type " ++ @typeName(T) ++ " is not supported.");
                }
            },
            else => @compileError("Convolution only supports u8, f32, and structs with all u8 fields. Type " ++ @typeName(T) ++ " is not supported."),
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
pub fn convolveSeparable(
    comptime T: type,
    image: Image(T),
    allocator: Allocator,
    kernel_x: []const f32,
    kernel_y: []const f32,
    border_mode: BorderMode,
    out: Image(T),
) !void {
    if (!image.hasSameShape(out)) return error.ShapeMismatch;
    if (kernel_x.len == 0 or kernel_y.len == 0) return error.EmptyKernel;
    if (kernel_x.len % 2 == 0 or kernel_y.len % 2 == 0) return error.EvenKernelNotSupported;
    if (imagesOverlap(T, image, out)) {
        var temp_out = try Image(T).initLike(allocator, out);
        defer temp_out.deinit(allocator);
        try convolveSeparable(T, image, allocator, kernel_x, kernel_y, border_mode, temp_out);
        temp_out.copy(out);
        return;
    }

    // Process based on type
    switch (T) {
        u8 => {
            var temp = try Image(T).init(allocator, image.rows, image.cols);
            defer temp.deinit(allocator);

            // Convert kernels to integer
            const scale = 256;
            const kernel_x_int = try allocator.alloc(i32, kernel_x.len);
            defer allocator.free(kernel_x_int);
            const kernel_y_int = try allocator.alloc(i32, kernel_y.len);
            defer allocator.free(kernel_y_int);

            for (kernel_x, 0..) |k, i| {
                kernel_x_int[i] = @intFromFloat(@round(k * scale));
            }
            for (kernel_y, 0..) |k, i| {
                kernel_y_int[i] = @intFromFloat(@round(k * scale));
            }

            const sum_abs_x = sumAbsI64NoOverflow(kernel_x_int);
            const sum_abs_y = sumAbsI64NoOverflow(kernel_y_int);
            const max_x, const overflow_x = @mulWithOverflow(sum_abs_x.sum, 255);
            const max_y, const overflow_y = @mulWithOverflow(sum_abs_y.sum, 255);

            if (!sum_abs_x.overflow and !sum_abs_y.overflow and overflow_x == 0 and overflow_y == 0 and max_x <= std.math.maxInt(i32) and max_y <= std.math.maxInt(i32)) {
                convolveSeparablePlane(u8, i32, image, out, temp, kernel_x_int, kernel_y_int, border_mode);
            } else {
                convolveSeparablePlane(u8, i64, image, out, temp, kernel_x_int, kernel_y_int, border_mode);
            }
        },
        f32 => {
            var temp = try Image(T).init(allocator, image.rows, image.cols);
            defer temp.deinit(allocator);

            const src_plane: Image(f32) = .{ .rows = image.rows, .cols = image.cols, .stride = image.stride, .data = image.data };
            const dst_plane: Image(f32) = .{ .rows = out.rows, .cols = out.cols, .stride = out.stride, .data = out.data };
            const tmp_plane: Image(f32) = .{ .rows = temp.rows, .cols = temp.cols, .stride = temp.stride, .data = temp.data };
            convolveSeparablePlane(f32, f32, src_plane, dst_plane, tmp_plane, kernel_x, kernel_y, border_mode);
        },
        else => switch (@typeInfo(T)) {
            .@"struct" => {
                // Optimized path for u8 structs (RGB, RGBA, etc.)
                if (comptime meta.allFieldsAreU8(T)) {
                    // Channel separation approach for optimal performance
                    const SCALE = 256;
                    const plane_size = image.rows * image.cols;
                    const Pixel = PixelIO(u8, i64, 1);

                    // Convert kernels to integer
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
                    const split = try channel_ops.splitChannelsWithUniform(T, image, allocator);
                    const channels = split.channels;
                    const uniforms = split.uniforms;
                    defer for (channels) |channel| allocator.free(channel);

                    var kernel_x_sum: i64 = 0;
                    for (kernel_x_int) |k| kernel_x_sum += @as(i64, k);
                    var kernel_y_sum: i64 = 0;
                    for (kernel_y_int) |k| kernel_y_sum += @as(i64, k);

                    const ChannelStrategy = enum { normalized, scaled, non_uniform };
                    var strategies: [channels.len]ChannelStrategy = undefined;
                    inline for (uniforms, 0..) |uniform_value, i| {
                        if (uniform_value != null and border_mode != .zero) {
                            if (kernel_x_sum == SCALE and kernel_y_sum == SCALE) {
                                strategies[i] = .normalized;
                            } else {
                                strategies[i] = .scaled;
                            }
                        } else {
                            strategies[i] = .non_uniform;
                        }
                    }

                    // Allocate output planes and a shared temp buffer only for non-normalized channels
                    var out_channels: [channels.len][]u8 = undefined;
                    var temp_plane: []u8 = &[_]u8{};
                    defer if (temp_plane.len > 0) allocator.free(temp_plane);

                    inline for (&out_channels, strategies, uniforms) |*out_ch, strategy, uniform_value| {
                        switch (strategy) {
                            .normalized => out_ch.* = &[_]u8{},
                            .scaled, .non_uniform => out_ch.* = try allocator.alloc(u8, plane_size),
                        }

                        if (strategy == .scaled) {
                            const value = uniform_value orelse unreachable;
                            const accum_x: i64 = @as(i64, @intCast(value)) * kernel_x_sum;
                            const stored_x = Pixel.store(accum_x);
                            const accum_y: i64 = @as(i64, stored_x) * kernel_y_sum;
                            const stored_y = Pixel.store(accum_y);
                            @memset(out_ch.*, stored_y);
                        } else if (strategy == .non_uniform and temp_plane.len == 0) {
                            temp_plane = try allocator.alloc(u8, plane_size);
                        }
                    }
                    defer {
                        inline for (out_channels, strategies) |out_ch, strategy| {
                            switch (strategy) {
                                .normalized => {},
                                .scaled, .non_uniform => if (out_ch.len > 0) allocator.free(out_ch),
                            }
                        }
                    }

                    // Convolve only non-uniform channels, reusing the shared temp buffer
                    inline for (channels, out_channels, strategies) |src_data, dst_data, strategy| {
                        if (strategy == .non_uniform) {
                            const src_plane: Image(u8) = .{ .rows = image.rows, .cols = image.cols, .stride = image.cols, .data = src_data };
                            const dst_plane: Image(u8) = .{ .rows = image.rows, .cols = image.cols, .stride = image.cols, .data = dst_data };
                            const tmp_plane: Image(u8) = .{ .rows = image.rows, .cols = image.cols, .stride = image.cols, .data = temp_plane };
                            convolveSeparablePlane(u8, i64, src_plane, dst_plane, tmp_plane, kernel_x_int, kernel_y_int, border_mode);
                        }
                    }

                    // Recombine channels, using original values for uniform channels
                    var final_channels: [channels.len][]const u8 = undefined;
                    inline for (strategies, out_channels, channels, 0..) |strategy, out_ch, src_ch, i| {
                        switch (strategy) {
                            .normalized => final_channels[i] = src_ch,
                            .scaled, .non_uniform => final_channels[i] = out_ch,
                        }
                    }
                    channel_ops.mergeChannels(T, final_channels, out);
                } else {
                    @compileError("Separable convolution only supports structs where all fields are u8. Type " ++ @typeName(T) ++ " is not supported.");
                }
            },
            else => @compileError("Separable convolution only supports u8, f32, and structs with all u8 fields. Type " ++ @typeName(T) ++ " is not supported."),
        },
    }
}

/// Unified separable convolution for both u8 and f32 planes with SIMD.
/// For u8, kernels must be pre-scaled by 256 for integer arithmetic.
fn convolveSeparablePlane(
    comptime T: type,
    comptime Scalar: type,
    src_img: Image(T),
    dst_img: Image(T),
    temp_img: Image(T),
    kernel_x: anytype,
    kernel_y: anytype,
    border_mode: BorderMode,
) void {
    if (T == u8 and Scalar != i32 and Scalar != i64) @compileError("u8 separable convolution requires i32/i64 Scalar");
    if (T == f32 and Scalar != f32) @compileError("f32 separable convolution requires f32 Scalar");
    const half_x = kernel_x.len / 2;
    const half_y = kernel_y.len / 2;
    const rows = src_img.rows;
    const cols = src_img.cols;
    const vec_len = std.simd.suggestVectorLength(Scalar) orelse 8;

    // Use the shared pixel I/O operations
    const Pixels = PixelIO(T, Scalar, vec_len);

    // Horizontal pass (src -> temp) with SIMD
    for (0..rows) |r| {
        const row_offset = r * src_img.stride;
        const temp_offset = r * temp_img.stride;

        // Process interior pixels with SIMD (no border handling)
        if (cols > 2 * half_x) {
            var c: usize = half_x;
            const safe_end = cols - half_x;

            // SIMD processing for interior pixels
            while (c + vec_len <= safe_end) : (c += vec_len) {
                var acc: @Vector(vec_len, Scalar) = @splat(0);

                // Apply kernel
                for (kernel_x, 0..) |k, ki| {
                    if (k != 0) {
                        const src_idx = row_offset + c + ki - half_x;
                        const pixels_vec = Pixels.loadVec(src_img.data, src_idx);
                        const k_vec: @Vector(vec_len, Scalar) = @splat(@as(Scalar, k));
                        acc += pixels_vec * k_vec;
                    }
                }

                Pixels.storeVec(acc, temp_img.data, temp_offset + c);
            }

            // Handle remaining pixels with scalar code
            while (c < safe_end) : (c += 1) {
                var result: Scalar = 0;
                const c0 = c - half_x;
                for (kernel_x, 0..) |k, i| {
                    const cc = c0 + i;
                    const pixel_val = Pixels.load(src_img.data[row_offset + cc]);
                    result += pixel_val * @as(Scalar, k);
                }
                temp_img.data[temp_offset + c] = Pixels.store(result);
            }
        }

        // Handle borders with scalar code
        for (0..@min(half_x, cols)) |c| {
            var result: Scalar = 0;
            const ic = @as(isize, @intCast(c));
            for (kernel_x, 0..) |k, i| {
                const icx = ic + @as(isize, @intCast(i)) - @as(isize, @intCast(half_x));
                const pixel_val = getPixelScalar(T, Scalar, src_img, @as(isize, @intCast(r)), icx, border_mode);
                result += pixel_val * @as(Scalar, k);
            }
            temp_img.data[temp_offset + c] = Pixels.store(result);
        }

        if (cols > half_x) {
            for (cols - half_x..cols) |c| {
                var result: Scalar = 0;
                const ic = @as(isize, @intCast(c));
                for (kernel_x, 0..) |k, i| {
                    const icx = ic + @as(isize, @intCast(i)) - @as(isize, @intCast(half_x));
                    const pixel_val = getPixelScalar(T, Scalar, src_img, @as(isize, @intCast(r)), icx, border_mode);
                    result += pixel_val * @as(Scalar, k);
                }
                temp_img.data[temp_offset + c] = Pixels.store(result);
            }
        }
    }

    // Vertical pass (temp -> dst) with SIMD across columns
    // Interior rows: process r in [half_y, rows - half_y)
    if (rows > 2 * half_y) {
        const safe_end_r = rows - half_y;
        for (half_y..safe_end_r) |r| {
            var c: usize = 0;

            // SIMD processing across columns
            while (c + vec_len <= cols) : (c += vec_len) {
                var acc: @Vector(vec_len, Scalar) = @splat(0);

                // Apply kernel
                for (kernel_y, 0..) |k, ki| {
                    if (k != 0) {
                        const src_row = r + ki - half_y;
                        const src_off = src_row * temp_img.stride;
                        const pixels_vec = Pixels.loadVec(temp_img.data, src_off + c);
                        const k_vec: @Vector(vec_len, Scalar) = @splat(@as(Scalar, k));
                        acc += pixels_vec * k_vec;
                    }
                }

                Pixels.storeVec(acc, dst_img.data, r * dst_img.stride + c);
            }

            // Remaining columns (scalar)
            while (c < cols) : (c += 1) {
                var result: Scalar = 0;
                const r0 = r - half_y;
                for (kernel_y, 0..) |k, i| {
                    if (k == 0) continue;
                    const rr = r0 + i;
                    const pixel_val = Pixels.load(temp_img.data[rr * temp_img.stride + c]);
                    result += pixel_val * @as(Scalar, k);
                }
                dst_img.data[r * dst_img.stride + c] = Pixels.store(result);
            }
        }
    }

    // Handle top border rows (scalar across columns)
    for (0..@min(half_y, rows)) |r| {
        for (0..cols) |c| {
            var result: Scalar = 0;
            const ir = @as(isize, @intCast(r));
            for (kernel_y, 0..) |k, i| {
                const iry = ir + @as(isize, @intCast(i)) - @as(isize, @intCast(half_y));
                const pixel_val = getPixelScalar(T, Scalar, temp_img, iry, @as(isize, @intCast(c)), border_mode);
                result += pixel_val * @as(Scalar, k);
            }
            dst_img.data[r * dst_img.stride + c] = Pixels.store(result);
        }
    }

    // Handle bottom border rows (scalar across columns)
    if (rows > half_y) {
        for (rows - half_y..rows) |r| {
            for (0..cols) |c| {
                var result: Scalar = 0;
                const ir = @as(isize, @intCast(r));
                for (kernel_y, 0..) |k, i| {
                    const iry = ir + @as(isize, @intCast(i)) - @as(isize, @intCast(half_y));
                    const pixel_val = getPixelScalar(T, Scalar, temp_img, iry, @as(isize, @intCast(c)), border_mode);
                    result += pixel_val * @as(Scalar, k);
                }
                dst_img.data[r * dst_img.stride + c] = Pixels.store(result);
            }
        }
    }
}

fn getPixelScalar(comptime T: type, comptime Scalar: type, img: Image(T), row: isize, col: isize, border_mode: BorderMode) Scalar {
    if (T != u8 and T != f32) @compileError("getPixelScalar only works with u8 and f32 pixel types");
    if (T == u8 and Scalar != i32 and Scalar != i64) @compileError("getPixelScalar(u8) only supports i32/i64 Scalar");
    if (T == f32 and Scalar != f32) @compileError("getPixelScalar(f32) only supports f32 Scalar");

    const coords = border.computeCoords(row, col, @intCast(img.rows), @intCast(img.cols), border_mode);
    const pixel = if (coords) |c| img.at(c.row, c.col).* else 0;
    return if (T == u8) @as(Scalar, pixel) else pixel;
}
