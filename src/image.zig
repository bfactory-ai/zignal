//! This module defines a generic Image struct and provides a suite of image processing operations,
//! including initialization, manipulation (flipping, resizing, rotation, cropping),
//! filtering (integral image, box blur, sharpen, Sobel edge detection), and pixel access utilities.
const std = @import("std");
const assert = std.debug.assert;
const expectEqual = std.testing.expectEqualDeep;
const expectEqualDeep = std.testing.expectEqualDeep;
const Allocator = std.mem.Allocator;

const as = @import("meta.zig").as;
const convert = @import("colorspace.zig").convert;
const isScalar = @import("meta.zig").isScalar;
const isStruct = @import("meta.zig").isStruct;
const Rgba = @import("colorspace.zig").Rgba;

const Rectangle = @import("geometry.zig").Rectangle;
const Point2d = @import("point.zig").Point2d;

/// A simple image struct that encapsulates the size and the data.
pub fn Image(comptime T: type) type {
    return struct {
        rows: usize,
        cols: usize,
        data: []T,
        stride: usize,

        const Self = @This();
        /// Constructs an image of rows and cols size.  If the slice is owned by this image,
        /// deinit should also be called.
        pub fn init(rows: usize, cols: usize, data: []T) Image(T) {
            return .{ .rows = rows, .cols = cols, .data = data, .stride = cols };
        }

        /// Constructs an image of rows and cols size allocating its own memory.
        pub fn initAlloc(allocator: std.mem.Allocator, rows: usize, cols: usize) !Image(T) {
            var array: std.ArrayList(T) = .init(allocator);
            try array.resize(rows * cols);
            return .{ .rows = rows, .cols = cols, .data = try array.toOwnedSlice(), .stride = cols };
        }

        /// Constructs an image of `rows` and `cols` size by reinterpreting the provided slice of `bytes` as a slice of `T`.
        /// The length of the `bytes` slice must be exactly `rows * cols * @sizeOf(T)`.
        pub fn initFromBytes(rows: usize, cols: usize, bytes: []u8) Image(T) {
            assert(rows * cols * @sizeOf(T) == bytes.len);
            return .{
                .rows = rows,
                .cols = cols,
                .data = @as([*]T, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / @sizeOf(T)],
                .stride = cols,
            };
        }

        /// Returns the image data reinterpreted as a slice of bytes.
        /// Note: The image should not be a view; this is enforced by an assertion.
        pub fn asBytes(self: Self) []u8 {
            assert(self.rows * self.cols == self.data.len);
            assert(!self.isView());
            return @as([*]u8, @ptrCast(@alignCast(self.data.ptr)))[0 .. self.data.len * @sizeOf(T)];
        }

        /// Sets the image rows and cols to zero and frees the memory from the image.  It should
        /// only be called if the image owns the memory.
        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.rows = 0;
            self.cols = 0;
            self.stride = 0;
            allocator.free(self.data);
        }

        /// Returns the number of channels or depth of this image type.
        pub fn channels() usize {
            return comptime switch (@typeInfo(T)) {
                .int, .float => 1,
                .@"struct" => std.meta.fields(T).len,
                .array => |info| info.len,
                else => @compileError("Image(" ++ @typeName(T) ++ " is unsupported."),
            };
        }

        /// Returns true if and only if `self` and `other` have the same number of rows and columns.
        /// It does not compare pixel data or types.
        pub fn hasSameShape(self: Self, other: anytype) bool {
            return self.rows == other.rows and self.cols == other.cols;
        }

        /// Returns the bounding rectangle for the current image.
        pub fn getRectangle(self: Self) Rectangle(usize) {
            return .{ .l = 0, .t = 0, .r = self.cols - 1, .b = self.rows - 1 };
        }

        /// Returns an image view with boundaries defined by `rect` within the image boundaries.
        /// The returned image references the memory of `self`, so there are no allocations
        /// or copies.
        pub fn view(self: Self, rect: Rectangle(usize)) Image(T) {
            const bounded = Rectangle(usize){
                .l = rect.l,
                .t = rect.t,
                .r = @min(rect.r, self.cols - 1),
                .b = @min(rect.b, self.rows - 1),
            };
            return .{
                .rows = bounded.height(),
                .cols = bounded.width(),
                .data = self.data[bounded.t * self.stride + bounded.l .. bounded.b * self.stride + bounded.r + 1],
                .stride = self.cols,
            };
        }

        /// Returns true if, and only if, `self` is a view of another image.
        /// This is determined by checking if the `cols` field differs from the `stride` field.
        pub fn isView(self: Self) bool {
            return self.cols != self.stride;
        }

        /// Returns the value at position row, col.  It assumes the coordinates are in bounds and
        /// triggers safety-checked undefined behavior when they aren't.
        pub inline fn at(self: Self, row: usize, col: usize) *T {
            assert(row < self.rows);
            assert(col < self.cols);
            return &self.data[row * self.stride + col];
        }

        /// Returns the optional value at row, col in the image.
        pub fn atOrNull(self: Self, row: isize, col: isize) ?*T {
            const irows: isize = @intCast(self.rows);
            const icols: isize = @intCast(self.cols);
            if (row < 0 or col < 0 or row >= irows or col >= icols) {
                return null;
            } else {
                return self.at(@intCast(row), @intCast(col));
            }
        }

        /// Flips an image from left to right (mirror effect).
        pub fn flipLeftRight(self: Self) void {
            for (0..self.rows) |r| {
                for (0..self.cols / 2) |c| {
                    std.mem.swap(T, self.at(r, c), self.at(r, self.cols - c - 1));
                }
            }
        }

        /// Flips an image from top to bottom (upside down effect).
        pub fn flipTopBottom(self: Self) void {
            for (0..self.rows / 2) |r| {
                for (0..self.cols) |c| {
                    std.mem.swap(T, self.at(r, c), self.at(self.rows - r - 1, c));
                }
            }
        }

        /// Performs bilinear interpolation at position x, y.
        /// Returns `null` if the coordinates `(x, y)` are too close to the image border for valid interpolation.
        pub fn interpolateBilinear(self: Self, x: f32, y: f32) ?T {
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

        /// Resizes an image to fit in out, using bilinear interpolation.
        pub fn resize(self: Self, out: Self) void {
            const x_scale: f32 = as(f32, self.cols - 1) / as(f32, @max(out.cols - 1, 1));
            const y_scale: f32 = as(f32, self.rows - 1) / as(f32, @max(out.rows - 1, 1));
            var sy: f32 = -y_scale;
            for (0..out.rows) |r| {
                sy += y_scale;
                var sx: f32 = -x_scale;
                for (0..out.cols) |c| {
                    sx += x_scale;
                    out.at(r, c).* = if (self.interpolateBilinear(sx, sy)) |val| val else std.mem.zeroes(T);
                }
            }
        }

        /// Rotates the image by `angle` (in radians) around a specified `center` point.
        ///
        /// Parameters:
        /// - `allocator`: The allocator to use for the rotated image's data.
        /// - `center`: The `Point2d(f32)` around which to rotate.
        /// - `angle`: The rotation angle in radians.
        /// - `rotated`: An out-parameter pointer to an `Image(T)` that will be initialized by this function
        ///   with the rotated image data. The caller is responsible for deallocating `rotated.data`
        ///   if it was allocated by this function.
        pub fn rotateFrom(self: Self, allocator: Allocator, center: Point2d(f32), angle: f32, rotated: *Self) !void {
            var array: std.ArrayList(T) = .init(allocator);
            try array.resize(self.rows * self.cols);
            rotated.* = .init(self.rows, self.cols, try array.toOwnedSlice());
            const cos = @cos(angle);
            const sin = @sin(angle);
            for (0..self.rows) |r| {
                const y: f32 = @floatFromInt(r);
                for (0..self.cols) |c| {
                    const x: f32 = @floatFromInt(c);
                    const rx = cos * (x - center.x) - sin * (y - center.y) + center.x;
                    const ry = sin * (x - center.x) + cos * (y - center.y) + center.y;
                    rotated.at(r, c).* = if (self.interpolateBilinear(rx, ry)) |val| val else std.mem.zeroes(T);
                }
            }
        }

        /// Rotates the image by `angle` (in radians) around its center.
        /// The returned `Image(T)` has newly allocated data that the caller is responsible for deallocating.
        pub fn rotate(self: Self, allocator: Allocator, angle: f32) !Self {
            return try self.rotateFrom(allocator, .{ .x = self.cols / 2, .y = self.rows / 2 }, angle);
        }

        /// Crops a rectangular region from the image.
        /// If the specified `rectangle` is not fully contained within the image, the out-of-bounds
        /// areas in the output `chip` are filled with zeroed pixels (e.g., black/transparent).
        ///
        /// Parameters:
        /// - `allocator`: The allocator to use for the cropped image's data.
        /// - `rectangle`: The `Rectangle(f32)` defining the region to crop. Coordinates will be rounded.
        /// - `chip`: An out-parameter pointer to an `Image(T)` that will be initialized by this function
        ///   with the cropped image data. The caller is responsible for deallocating `chip.data`.
        pub fn crop(self: Self, allocator: Allocator, rectangle: Rectangle(f32), chip: *Self) !void {
            const chip_top: isize = @intFromFloat(@round(rectangle.t));
            const chip_left: isize = @intFromFloat(@round(rectangle.l));
            const chip_rows: usize = @intFromFloat(@round(rectangle.height()));
            const chip_cols: usize = @intFromFloat(@round(rectangle.width()));
            chip.* = try .initAlloc(allocator, chip_rows, chip_cols);
            for (0..chip_rows) |r| {
                const ir: isize = @intCast(r);
                for (0..chip_cols) |c| {
                    const ic: isize = @intCast(c);
                    chip.at(r, c).* = if (self.atOrNull(ir + chip_top, ic + chip_left)) |val| val.* else std.mem.zeroes(T);
                }
            }
        }

        /// Computes the integral image (also known as a summed-area table) of `self`.
        /// For multi-channel images (e.g., structs like `Rgba`), it computes a per-channel
        /// integral image, storing the result as an array of floats per pixel in the output `integral` image.
        pub fn integralImage(
            self: Self,
            allocator: Allocator,
            integral: *Image(if (isScalar(T)) f32 else [Self.channels()]f32),
        ) !void {
            if (!self.hasSameShape(integral.*)) {
                integral.* = try .initAlloc(allocator, self.rows, self.cols);
            }
            switch (@typeInfo(T)) {
                .int, .float => {
                    var tmp: f32 = 0;
                    for (0..self.cols) |c| {
                        tmp += as(f32, (self.at(0, c).*));
                        integral.at(0, c).* = tmp;
                    }
                    for (1..self.rows) |r| {
                        tmp = 0;
                        for (0..self.cols) |c| {
                            tmp += as(f32, self.at(r, c).*);
                            integral.at(r, c).* = tmp + integral.at(r - 1, c).*;
                        }
                    }
                },
                .@"struct" => {
                    var tmp = [_]f32{0} ** Self.channels();
                    for (0..self.cols) |c| {
                        inline for (std.meta.fields(T), 0..) |f, i| {
                            tmp[i] += as(f32, @field(self.at(0, c).*, f.name));
                            integral.at(0, c)[i] = tmp[i];
                        }
                    }
                    for (1..self.rows) |r| {
                        tmp = [_]f32{0} ** Self.channels();
                        for (0..self.cols) |c| {
                            inline for (std.meta.fields(T), 0..) |f, i| {
                                tmp[i] += as(f32, @field(self.at(r, c).*, f.name));
                                integral.at(r, c)[i] = tmp[i] + integral.at(r - 1, c)[i];
                            }
                        }
                    }
                },
                else => @compileError("Can't compute the integral image of " ++ @typeName(T) ++ "."),
            }
        }

        /// Computes the integral image using SIMD optimizations where possible.
        /// Uses a two-pass approach: first computing row-wise sums, then adding column-wise.
        pub fn integralImageSimd(
            self: Self,
            allocator: Allocator,
            integral: *Image(if (isScalar(T)) f32 else [Self.channels()]f32),
        ) !void {
            if (!self.hasSameShape(integral.*)) {
                integral.* = try .initAlloc(allocator, self.rows, self.cols);
            }
            switch (@typeInfo(T)) {
                .int, .float => {
                    // First pass: compute row-wise cumulative sums
                    for (0..self.rows) |r| {
                        var tmp: f32 = 0;
                        const row_offset = r * self.stride;
                        const out_offset = r * integral.cols;
                        for (0..self.cols) |c| {
                            tmp += as(f32, self.data[row_offset + c]);
                            integral.data[out_offset + c] = tmp;
                        }
                    }

                    // Second pass: add column-wise cumulative sums using SIMD
                    const simd_len = std.simd.suggestVectorLength(f32) orelse 1;
                    for (1..self.rows) |r| {
                        const prev_row_offset = (r - 1) * integral.cols;
                        const curr_row_offset = r * integral.cols;
                        var c: usize = 0;

                        // Process SIMD-width chunks
                        while (c + simd_len <= self.cols) : (c += simd_len) {
                            const prev_vals: @Vector(simd_len, f32) = integral.data[prev_row_offset + c ..][0..simd_len].*;
                            const curr_vals: @Vector(simd_len, f32) = integral.data[curr_row_offset + c ..][0..simd_len].*;
                            integral.data[curr_row_offset + c ..][0..simd_len].* = prev_vals + curr_vals;
                        }

                        // Handle remaining columns
                        while (c < self.cols) : (c += 1) {
                            integral.data[curr_row_offset + c] += integral.data[prev_row_offset + c];
                        }
                    }
                },
                .@"struct" => {
                    const num_channels = comptime Self.channels();

                    // First pass: compute row-wise cumulative sums for all channels
                    for (0..self.rows) |r| {
                        var tmp = [_]f32{0} ** num_channels;
                        const row_offset = r * self.stride;
                        const out_offset = r * integral.cols;
                        for (0..self.cols) |c| {
                            inline for (std.meta.fields(T), 0..) |f, i| {
                                tmp[i] += as(f32, @field(self.data[row_offset + c], f.name));
                                integral.data[out_offset + c][i] = tmp[i];
                            }
                        }
                    }

                    // Second pass: add column-wise cumulative sums
                    for (1..self.rows) |r| {
                        const prev_row_offset = (r - 1) * integral.cols;
                        const curr_row_offset = r * integral.cols;
                        for (0..self.cols) |c| {
                            inline for (0..num_channels) |i| {
                                integral.data[curr_row_offset + c][i] += integral.data[prev_row_offset + c][i];
                            }
                        }
                    }
                },
                else => @compileError("Can't compute the integral image of " ++ @typeName(T) ++ "."),
            }
        }

        /// Computes a blurred version of `self` using a box blur algorithm, efficiently implemented
        /// using an integral image. The `radius` parameter determines the size of the box window.
        /// This function is optimized using SIMD instructions for performance where applicable.
        pub fn boxBlur(self: Self, allocator: std.mem.Allocator, blurred: *Self, radius: usize) !void {
            if (!self.hasSameShape(blurred.*)) {
                blurred.* = try .initAlloc(allocator, self.rows, self.cols);
            }
            if (radius == 0 and &self != blurred) {
                for (self.data, blurred.data) |s, *b| b.* = s;
                return;
            }

            switch (@typeInfo(T)) {
                .int, .float => {
                    var integral: Image(f32) = undefined;
                    try self.integralImage(allocator, &integral);
                    defer integral.deinit(allocator);
                    const size = self.rows * self.cols;
                    var pos: usize = 0;
                    var rem: usize = size;
                    const simd_len = std.simd.suggestVectorLength(T) orelse 1;
                    while (pos < size) {
                        const r = pos / self.cols;
                        const c = pos % self.cols;
                        const r1 = r -| radius;
                        const r2 = @min(r + radius, self.rows - 1);
                        const r1_offset = r1 * self.cols;
                        const r2_offset = r2 * self.cols;
                        const r2_r1 = r2 - r1;
                        if (r1 >= radius and r2 <= self.rows - 1 - radius and
                            self.cols >= 1 + radius + simd_len and
                            c >= radius and c <= self.cols - 1 - radius - simd_len and
                            rem >= simd_len)
                        {
                            const c1 = c - radius;
                            const c2 = c + radius;
                            const int11s: @Vector(simd_len, f32) = integral.data[r1_offset + c1 ..][0..simd_len].*;
                            const int12s: @Vector(simd_len, f32) = integral.data[r1_offset + c2 ..][0..simd_len].*;
                            const int21s: @Vector(simd_len, f32) = integral.data[r2_offset + c1 ..][0..simd_len].*;
                            const int22s: @Vector(simd_len, f32) = integral.data[r2_offset + c2 ..][0..simd_len].*;
                            const areas: @Vector(simd_len, f32) = @splat(@as(f32, @floatFromInt(r2_r1 * 2 * radius)));
                            const sums = int22s - int21s - int12s + int11s;
                            const vals: [simd_len]f32 = if (@typeInfo(T) == .int) @round(sums / areas) else sums / areas;
                            for (vals, 0..) |val, i| {
                                if (@typeInfo(T) == .int) {
                                    blurred.at(r, c + i).* = @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), val)));
                                } else {
                                    blurred.at(r, c + i).* = val;
                                }
                            }
                            pos += simd_len;
                            rem -= simd_len;
                        } else {
                            const c1 = c -| radius;
                            const c2 = @min(c + radius, self.cols - 1);
                            const pos11 = r1_offset + c1;
                            const pos12 = r1_offset + c2;
                            const pos21 = r2_offset + c1;
                            const pos22 = r2_offset + c2;
                            const area: f32 = @floatFromInt((r2 - r1) * (c2 - c1));
                            const sum = integral.data[pos22] - integral.data[pos21] - integral.data[pos12] + integral.data[pos11];
                            blurred.at(r, c).* = switch (@typeInfo(T)) {
                                .int => @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), (@round(sum / area))))),
                                else => as(T, sum / area),
                            };
                            pos += 1;
                            rem -= 1;
                        }
                    }
                },
                .@"struct" => {
                    var integral: Image([Self.channels()]f32) = undefined;
                    try self.integralImage(allocator, &integral);
                    defer integral.deinit(allocator);
                    const size = self.rows * self.cols;
                    var pos: usize = 0;
                    var rem: usize = size;
                    while (pos < size) {
                        const r = pos / self.cols;
                        const c = pos % self.cols;
                        const r1 = r -| radius;
                        const c1 = c -| radius;
                        const r2 = @min(r + radius, self.rows - 1);
                        const c2 = @min(c + radius, self.cols - 1);
                        const r1_offset = r1 * self.cols;
                        const r2_offset = r2 * self.cols;
                        const pos11 = r1_offset + c1;
                        const pos12 = r1_offset + c2;
                        const pos21 = r2_offset + c1;
                        const pos22 = r2_offset + c2;
                        const area: f32 = @floatFromInt((r2 - r1) * (c2 - c1));
                        inline for (std.meta.fields(T), 0..) |f, i| {
                            const sum = integral.data[pos22][i] - integral.data[pos21][i] - integral.data[pos12][i] + integral.data[pos11][i];
                            @field(blurred.data[pos], f.name) = switch (@typeInfo(f.type)) {
                                .int => @intFromFloat(@max(std.math.minInt(f.type), @min(std.math.maxInt(f.type), @round(sum / area)))),
                                .float => as(f.type, sum / area),
                                else => @compileError("Can't compute the boxBlur image with struct fields of type " ++ @typeName(f.type) ++ "."),
                            };
                        }
                        pos += 1;
                        rem -= 1;
                    }
                },
                else => @compileError("Can't compute the boxBlur image of " ++ @typeName(T) ++ "."),
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
            if (radius == 0 and &self != sharpened) {
                for (self.data, sharpened.data) |s, *b| b.* = s;
                return;
            }

            switch (@typeInfo(T)) {
                .int, .float => {
                    var integral: Image(f32) = undefined;
                    defer integral.deinit(allocator);
                    try self.integralImage(allocator, &integral);
                    const size = self.rows * self.cols;
                    var pos: usize = 0;
                    var rem: usize = size;
                    const simd_len = std.simd.suggestVectorLength(T) orelse 1;
                    const box_areas: @Vector(simd_len, f32) = @splat(2 * radius * 2 * radius);
                    while (pos < size) {
                        const r = pos / self.cols;
                        const c = pos % self.cols;
                        const r1 = r -| radius;
                        const r2 = @min(r + radius, self.rows - 1);
                        const r1_offset = r1 * self.cols;
                        const r2_offset = r2 * self.cols;
                        if (r1 >= radius and r2 <= self.rows - 1 - radius and
                            c >= radius and c <= self.cols - 1 - radius - simd_len and
                            rem >= simd_len)
                        {
                            const c1 = c - radius;
                            const c2 = c + radius;
                            const int11s: @Vector(simd_len, f32) = integral.data[r1_offset + c1 ..][0..simd_len];
                            const int12s: @Vector(simd_len, f32) = integral.data[r1_offset + c2 ..][0..simd_len];
                            const int21s: @Vector(simd_len, f32) = integral.data[r2_offset + c1 ..][0..simd_len];
                            const int22s: @Vector(simd_len, f32) = integral.data[r2_offset + c2 ..][0..simd_len];
                            const sums = int22s - int21s - int12s + int11s;
                            const vals: [simd_len]f32 = @round(sums / box_areas);
                            for (vals, 0..) |val, i| {
                                if (@typeInfo(T) == .int) {
                                    const temp = @max(0, @min(std.math.maxInt(T), as(isize, self.data[pos]) * 2 - @as(isize, val)));
                                    sharpened.at(r, c + i).* = as(T, temp);
                                } else {
                                    sharpened.at(r, c + i).* = 2 * self.data[pos] - val;
                                }
                            }
                            pos += simd_len;
                            rem -= simd_len;
                        } else {
                            const c1 = c -| radius;
                            const c2 = @min(c + radius, self.cols - 1);
                            const pos11 = r1_offset + c1;
                            const pos12 = r1_offset + c2;
                            const pos21 = r2_offset + c1;
                            const pos22 = r2_offset + c2;
                            const area: f32 = @floatFromInt((r2 - r1) * (c2 - c1));
                            const sum = integral.data[pos22] - integral.data[pos21] - integral.data[pos12] + integral.data[pos11];
                            sharpened.at(r, c).* = switch (@typeInfo(T)) {
                                .int => @intFromFloat(@max(std.math.minInt(T), @min(std.math.maxInt(T), @round(sum / area)))),
                                else => as(T, sum / area),
                            };
                            pos += 1;
                            rem -= 1;
                        }
                    }
                },
                .@"struct" => {
                    var integral: Image([Self.channels()]f32) = undefined;
                    try self.integralImage(allocator, &integral);
                    defer integral.deinit(allocator);
                    const size = self.rows * self.cols;
                    var pos: usize = 0;
                    var rem: usize = size;
                    while (pos < size) {
                        const r = pos / self.cols;
                        const c = pos % self.cols;
                        const r1 = r -| radius;
                        const c1 = c -| radius;
                        const r2 = @min(r + radius, self.rows - 1);
                        const c2 = @min(c + radius, self.cols - 1);
                        const r1_offset = r1 * self.cols;
                        const r2_offset = r2 * self.cols;
                        const pos11 = r1_offset + c1;
                        const pos12 = r1_offset + c2;
                        const pos21 = r2_offset + c1;
                        const pos22 = r2_offset + c2;
                        const area: f32 = @floatFromInt((r2 - r1) * (c2 - c1));
                        inline for (std.meta.fields(T), 0..) |f, i| {
                            const sum = integral.data[pos22][i] - integral.data[pos21][i] - integral.data[pos12][i] + integral.data[pos11][i];
                            @field(sharpened.data[pos], f.name) = switch (@typeInfo(f.type)) {
                                .int => as(f.type, @max(0, @min(std.math.maxInt(f.type), as(isize, @field(self.data[pos], f.name)) * 2 - as(isize, @round(sum / area))))),
                                .float => as(f.type, 2 * @field(self.data[pos], f.name) - sum / area),
                                else => @compileError("Can't compute the sharpen image with struct fields of type " ++ @typeName(f.type) ++ "."),
                            };
                        }
                        pos += 1;
                        rem -= 1;
                    }
                },
                else => @compileError("Can't compute the sharpen image of " ++ @typeName(T) ++ "."),
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
            const vert_filter = [3][3]i32{
                .{ -1, -2, -1 },
                .{ 0, 0, 0 },
                .{ 1, 2, 1 },
            };
            const horz_filter = [3][3]i32{
                .{ -1, 0, 1 },
                .{ -2, 0, 2 },
                .{ -1, 0, 1 },
            };
            for (0..self.rows) |r| {
                for (0..self.cols) |c| {
                    const ir: isize = @intCast(r);
                    const ic: isize = @intCast(c);
                    var horz_temp: i32 = 0;
                    var vert_temp: i32 = 0;
                    for (0..vert_filter.len) |m| {
                        const py: isize = ir - 1 + @as(isize, @intCast(m));
                        for (0..vert_filter[0].len) |n| {
                            const px: isize = ic - 1 + @as(isize, @intCast(n));
                            if (self.atOrNull(py, px)) |val| {
                                const p: i32 = @intCast(convert(u8, val.*));
                                horz_temp += p * horz_filter[m][n];
                                vert_temp += p * vert_filter[m][n];
                            }
                        }
                    }
                    out.at(r, c).* = @intFromFloat(@max(0, @min(255, @sqrt(@as(f32, @floatFromInt(horz_temp * horz_temp + vert_temp * vert_temp))))));
                }
            }
        }
    };
}

test "integral image scalar" {
    var image: Image(u8) = try .initAlloc(std.testing.allocator, 21, 13);
    defer image.deinit(std.testing.allocator);
    for (image.data) |*i| i.* = 1;
    var integral: Image(f32) = undefined;
    try image.integralImage(std.testing.allocator, &integral);
    defer integral.deinit(std.testing.allocator);
    try expectEqual(image.rows, integral.rows);
    try expectEqual(image.cols, integral.cols);
    try expectEqual(image.data.len, integral.data.len);
    for (0..image.rows) |r| {
        for (0..image.cols) |c| {
            const area_at_pos: f32 = @floatFromInt((r + 1) * (c + 1));
            try expectEqual(area_at_pos, integral.at(r, c).*);
        }
    }
}

test "integral image view scalar" {
    var image: Image(u8) = try .initAlloc(std.testing.allocator, 21, 13);
    defer image.deinit(std.testing.allocator);
    for (image.data) |*i| i.* = 1;
    const view = image.view(.{ .l = 2, .t = 3, .r = 8, .b = 10 });
    var integral: Image(f32) = undefined;
    try view.integralImage(std.testing.allocator, &integral);
    defer integral.deinit(std.testing.allocator);
    try expectEqual(view.rows, integral.rows);
    try expectEqual(view.cols, integral.cols);
    for (0..view.rows) |r| {
        for (0..view.cols) |c| {
            const area_at_pos: f32 = @floatFromInt((r + 1) * (c + 1));
            try expectEqual(area_at_pos, integral.at(r, c).*);
        }
    }
}

test "integral image struct" {
    var image: Image(Rgba) = try .initAlloc(std.testing.allocator, 21, 13);
    defer image.deinit(std.testing.allocator);
    for (image.data) |*i| i.* = .{ .r = 1, .g = 1, .b = 1, .a = 1 };
    var integral: Image([4]f32) = undefined;
    try image.integralImage(std.testing.allocator, &integral);

    defer integral.deinit(std.testing.allocator);
    try expectEqual(image.rows, integral.rows);
    try expectEqual(image.cols, integral.cols);
    try expectEqual(image.data.len, integral.data.len);
    for (0..image.rows) |r| {
        for (0..image.cols) |c| {
            const area_at_pos: f32 = @floatFromInt((r + 1) * (c + 1));
            for (0..4) |i| {
                try expectEqual(area_at_pos, integral.at(r, c)[i]);
            }
        }
    }
}

test "integral image SIMD performance" {
    const Timer = std.time.Timer;
    const print = std.debug.print;

    // Create a larger image for meaningful timing
    var image: Image(u8) = try .initAlloc(std.testing.allocator, 1024, 1024);
    defer image.deinit(std.testing.allocator);

    // Fill with random-ish data
    for (image.data, 0..) |*pixel, i| {
        pixel.* = @intCast((i * 17 + 23) % 256);
    }

    var integral_regular: Image(f32) = undefined;
    var integral_simd: Image(f32) = undefined;

    // Benchmark regular implementation
    var timer = try Timer.start();
    try image.integralImage(std.testing.allocator, &integral_regular);
    const regular_time = timer.read();
    defer integral_regular.deinit(std.testing.allocator);

    // Benchmark SIMD implementation
    timer.reset();
    try image.integralImageSimd(std.testing.allocator, &integral_simd);
    const simd_time = timer.read();
    defer integral_simd.deinit(std.testing.allocator);

    // Verify results are identical
    for (0..image.rows) |r| {
        for (0..image.cols) |c| {
            try expectEqual(integral_regular.at(r, c).*, integral_simd.at(r, c).*);
        }
    }

    print("\nIntegral Image Benchmark (1024x1024 u8):\n", .{});
    print("Regular: {d:.2}ms\n", .{@as(f64, @floatFromInt(regular_time)) / 1e6});
    print("SIMD:    {d:.2}ms\n", .{@as(f64, @floatFromInt(simd_time)) / 1e6});
    print("Speedup: {d:.2}x\n", .{@as(f64, @floatFromInt(regular_time)) / @as(f64, @floatFromInt(simd_time))});
}

test "integral image SIMD vs regular" {
    // Test scalar type
    var image_scalar: Image(u8) = try .initAlloc(std.testing.allocator, 47, 33);
    defer image_scalar.deinit(std.testing.allocator);

    // Fill with random-ish data
    for (image_scalar.data, 0..) |*pixel, i| {
        pixel.* = @intCast((i * 17 + 23) % 256);
    }

    var integral_regular: Image(f32) = undefined;
    var integral_simd: Image(f32) = undefined;

    try image_scalar.integralImage(std.testing.allocator, &integral_regular);
    defer integral_regular.deinit(std.testing.allocator);

    try image_scalar.integralImageSimd(std.testing.allocator, &integral_simd);
    defer integral_simd.deinit(std.testing.allocator);

    // Verify both produce identical results
    for (0..image_scalar.rows) |r| {
        for (0..image_scalar.cols) |c| {
            try expectEqual(integral_regular.at(r, c).*, integral_simd.at(r, c).*);
        }
    }

    // Test struct type
    var image_struct: Image(Rgba) = try .initAlloc(std.testing.allocator, 31, 29);
    defer image_struct.deinit(std.testing.allocator);

    for (image_struct.data, 0..) |*pixel, i| {
        pixel.* = .{
            .r = @intCast((i * 7 + 11) % 256),
            .g = @intCast((i * 13 + 17) % 256),
            .b = @intCast((i * 19 + 23) % 256),
            .a = @intCast((i * 29 + 31) % 256),
        };
    }

    var integral_struct_regular: Image([4]f32) = undefined;
    var integral_struct_simd: Image([4]f32) = undefined;

    try image_struct.integralImage(std.testing.allocator, &integral_struct_regular);
    defer integral_struct_regular.deinit(std.testing.allocator);

    try image_struct.integralImageSimd(std.testing.allocator, &integral_struct_simd);
    defer integral_struct_simd.deinit(std.testing.allocator);

    // Verify both produce identical results for all channels
    for (0..image_struct.rows) |r| {
        for (0..image_struct.cols) |c| {
            for (0..4) |ch| {
                try expectEqual(integral_struct_regular.at(r, c)[ch], integral_struct_simd.at(r, c)[ch]);
            }
        }
    }
}

test "getRectangle" {
    var image: Image(Rgba) = try .initAlloc(std.testing.allocator, 21, 13);
    defer image.deinit(std.testing.allocator);
    const rect = image.getRectangle();
    try expectEqual(rect.width(), image.cols);
    try expectEqual(rect.height(), image.rows);
}

test "view" {
    var image: Image(Rgba) = try .initAlloc(std.testing.allocator, 21, 13);
    defer image.deinit(std.testing.allocator);
    const rect: Rectangle(usize) = .{ .l = 0, .t = 0, .r = 8, .b = 10 };
    const view = image.view(rect);
    try expectEqual(view.isView(), true);
    try expectEqual(image.isView(), false);
    try expectEqualDeep(rect, view.getRectangle());
}
