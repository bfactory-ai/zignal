const std = @import("std");
const Allocator = std.mem.Allocator;
const Rgba = @import("color.zig").Rgba;
const as = @import("meta.zig").as;
const isScalar = @import("meta.zig").isScalar;
const isStruct = @import("meta.zig").isStruct;
const Rectangle = @import("geometry.zig").Rectangle(f32);
const Point2d = @import("point.zig").Point2d(f32);

/// A simple image struct that encapsulates the size and the data.
pub fn Image(comptime T: type) type {
    return struct {
        rows: usize,
        cols: usize,
        data: []T,

        const Self = @This();
        /// Constructs an image of rows and cols size.  If the slice
        /// is owned by this image, deinit should also be called.
        pub fn init(rows: usize, cols: usize, data: []T) Self {
            return .{ .rows = rows, .cols = cols, .data = data };
        }

        pub fn initAlloc(allocator: std.mem.Allocator, rows: usize, cols: usize) !Self {
            var array = std.ArrayList(T).init(allocator);
            try array.resize(rows * cols);
            return .{ .rows = rows, .cols = cols, .data = try array.toOwnedSlice() };
        }

        /// Sets the image rows and cols to zero and frees the memory from the image.
        /// It should only be called if the image owns the memory.
        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.rows = 0;
            self.cols = 0;
            allocator.free(self.data);
        }

        /// Returns the optional value at row, col in the image.
        pub fn at(self: Self, row: isize, col: isize) ?*T {
            const irows: isize = @intCast(self.rows);
            const icols: isize = @intCast(self.cols);
            if (row < 0 or col < 0 or row >= irows or col >= icols) {
                return null;
            } else {
                const pos: usize = @intCast(row * icols + col);
                return &self.data[pos];
            }
        }

        /// Flips an image from left to right (mirror effect).
        pub fn flipLeftRight(self: Self) void {
            for (0..self.rows) |r| {
                for (0..self.cols / 2) |c| {
                    const left = r * self.cols + c;
                    const right = r * self.cols + self.cols - c - 1;
                    std.mem.swap(T, &self.data[left], &self.data[right]);
                }
            }
        }

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
            const tl: T = self.data[as(usize, top) * self.cols + as(usize, left)];
            const tr: T = self.data[as(usize, top) * self.cols + as(usize, right)];
            const bl: T = self.data[as(usize, bottom) * self.cols + as(usize, left)];
            const br: T = self.data[as(usize, bottom) * self.cols + as(usize, right)];
            var temp: T = undefined;
            switch (@typeInfo(T)) {
                .Int, .Float => {
                    temp = as(T, (1 - tb_frac) * ((1 - lr_frac) * as(f32, tl) +
                        lr_frac * as(f32, tr)) +
                        tb_frac * ((1 - lr_frac) * as(f32, bl) +
                        lr_frac * as(f32, br)));
                },
                .Struct => {
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
        pub fn resize(self: Self, out: *Self) void {
            const x_scale: f32 = as(f32, self.cols - 1) / as(f32, @max(out.cols - 1, 1));
            const y_scale: f32 = as(f32, self.rows - 1) / as(f32, @max(out.rows - 1, 1));
            var sy: f32 = -y_scale;
            for (0..out.rows) |r| {
                sy += y_scale;
                var sx: f32 = -x_scale;
                for (0..out.cols) |c| {
                    sx += x_scale;
                    out.data[r * out.cols + c] = if (self.interpolateBilinear(sx, sy)) |val| val else std.mem.zeroes(T);
                }
            }
        }

        pub fn rotateFrom(self: Self, allocator: Allocator, center: Point2d, angle: f32, rotated: *Self) !void {
            var array = std.ArrayList(T).init(allocator);
            try array.resize(self.rows * self.cols);
            rotated.* = Self.init(self.rows, self.cols, try array.toOwnedSlice());
            const cos = @cos(angle);
            const sin = @sin(angle);
            for (0..self.rows) |r| {
                const y: f32 = @floatFromInt(r);
                for (0..self.cols) |c| {
                    const x: f32 = @floatFromInt(c);
                    const rx = cos * (x - center.x) - sin * (y - center.y) + center.x;
                    const ry = sin * (x - center.x) + cos * (y - center.y) + center.y;
                    rotated.data[r * rotated.cols + c] = if (self.interpolateBilinear(rx, ry)) |val| val else std.mem.zeroes(T);
                }
            }
        }

        /// Rotates the image by angle (in radians) from the center.  It must be freed on the caller side.
        pub fn rotate(self: Self, allocator: Allocator, angle: f32) !Self {
            return try self.rotateFrom(allocator, .{ .x = self.cols / 2, .y = self.rows / 2 }, angle);
        }

        /// Crops the rectangle out of the image.  If the rectangle is not fully contained in the image, that area
        /// is filled with black/transparent pixels.
        pub fn crop(self: Self, allocator: Allocator, rectangle: Rectangle, chip: *Self) !void {
            const chip_top: isize = @intFromFloat(@round(rectangle.t));
            const chip_left: isize = @intFromFloat(@round(rectangle.l));
            const chip_rows: usize = @intFromFloat(@round(rectangle.height()));
            const chip_cols: usize = @intFromFloat(@round(rectangle.width()));
            chip.* = try Image(T).initAlloc(allocator, chip_rows, chip_cols);
            for (0..chip_rows) |r| {
                const ir: isize = @intCast(r);
                for (0..chip_cols) |c| {
                    const ic: isize = @intCast(c);
                    chip.data[r * chip_cols + c] = if (self.at(@intCast(ir + chip_top), @intCast(ic + chip_left))) |val|
                        val.*
                    else
                        std.mem.zeroes(T);
                }
            }
        }

        /// Computes the integral image of self.
        pub fn integralImage(self: Self, allocator: Allocator) !(if (isScalar(T)) Image(f32) else if (isStruct(T)) Image([std.meta.fields(T).len]f32) else @compileError("Can't compute the integral image of " ++ @typeName(T) ++ ".")) {
            switch (@typeInfo(T)) {
                .ComptimeInt, .Int, .ComptimeFloat, .Float => {
                    var integral = try Image(f32).initAlloc(allocator, self.rows, self.cols);
                    var tmp: f32 = 0;
                    for (0..self.cols) |c| {
                        tmp += as(f32, (self.data[c]));
                        integral.data[c] = tmp;
                    }
                    for (1..self.rows) |r| {
                        tmp = 0;
                        for (0..self.cols) |c| {
                            const curr_pos = r * self.cols + c;
                            const prev_pos = (r - 1) * self.cols + c;
                            tmp += as(f32, self.data[curr_pos]);
                            integral.data[curr_pos] = tmp + integral.data[prev_pos];
                        }
                    }
                    return integral;
                },
                .Struct => {
                    var integral = try Image([std.meta.fields(T).len]f32).initAlloc(allocator, self.rows, self.cols);
                    var tmp: f32 = 0;
                    for (0..self.cols) |c| {
                        inline for (std.meta.fields(T), 0..) |f, i| {
                            tmp += as(f32, @field(self.data[c], f.name));
                            integral.data[c][i] = tmp;
                        }
                    }
                    for (1..self.rows) |r| {
                        tmp = 0;
                        for (0..self.cols) |c| {
                            const curr_pos = r * self.cols + c;
                            const prev_pos = (r - 1) * self.cols + c;
                            inline for (std.meta.fields(T), 0..) |f, i| {
                                tmp += as(f32, @field(self.data[curr_pos], f.name));
                                integral.data[curr_pos][i] = tmp + integral.data[prev_pos][i];
                            }
                        }
                    }
                    return integral;
                },
                else => @compileError("Can't compute the integral image of " ++ @typeName(T) ++ "."),
            }
        }
    };
}
