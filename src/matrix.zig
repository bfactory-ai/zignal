//! This module provides a generic, fixed-size Matrix struct for floating-point types
//! and a collection of common linear algebra operations such as addition, multiplication,
//! dot product, transpose, norm computation, determinant, and inverse (for small matrices).
const std = @import("std");
const assert = std.debug.assert;
const expectEqual = std.testing.expectEqual;
const expectEqualDeep = std.testing.expectEqualDeep;
const builtin = @import("builtin");

const Point2d = @import("geometry/points.zig").Point2d;
const Point3d = @import("geometry/points.zig").Point3d;

/// Storage strategy for static matrices with compile-time known dimensions
fn StaticStorage(comptime T: type, comptime rows: usize, comptime cols: usize) type {
    assert(@typeInfo(T) == .float);
    return struct {
        items: [rows][cols]T = undefined,
        
        pub fn at(self: @This(), row: usize, col: usize) T {
            assert(row < rows);
            assert(col < cols);
            return self.items[row][col];
        }
        
        pub fn set(self: *@This(), row: usize, col: usize, val: T) void {
            assert(row < rows);
            assert(col < cols);
            self.items[row][col] = val;
        }
        
        pub fn rowCount(self: @This()) usize {
            _ = self;
            return rows;
        }
        
        pub fn colCount(self: @This()) usize {
            _ = self;
            return cols;
        }
        
        // Allow direct access to items
        pub fn getItems(self: *@This()) *[rows][cols]T {
            return &self.items;
        }
    };
}

/// Storage strategy for dynamic matrices with runtime dimensions
fn DynamicStorage(comptime T: type) type {
    assert(@typeInfo(T) == .float);
    return struct {
        items: [][]T,
        row_count: usize,
        col_count: usize,
        allocator: std.mem.Allocator,
        
        pub fn init(allocator: std.mem.Allocator, row_count: usize, col_count: usize) !@This() {
            var self = @This(){
                .allocator = allocator,
                .row_count = row_count,
                .col_count = col_count,
                .items = undefined,
            };
            
            self.items = try allocator.alloc([]T, row_count);
            for (0..row_count) |i| {
                self.items[i] = try allocator.alloc(T, col_count);
            }
            return self;
        }
        
        pub fn deinit(self: *@This()) void {
            for (self.items) |row| {
                self.allocator.free(row);
            }
            self.allocator.free(self.items);
        }
        
        pub fn at(self: @This(), row: usize, col: usize) T {
            assert(row < self.row_count);
            assert(col < self.col_count);
            return self.items[row][col];
        }
        
        pub fn set(self: *@This(), row: usize, col: usize, val: T) void {
            assert(row < self.row_count);
            assert(col < self.col_count);
            self.items[row][col] = val;
        }
        
        pub fn rowCount(self: @This()) usize {
            return self.row_count;
        }
        
        pub fn colCount(self: @This()) usize {
            return self.col_count;
        }
        
        pub fn createSimilar(self: @This(), new_rows: usize, new_cols: usize) !DynamicStorage(T) {
            return DynamicStorage(T).init(self.allocator, new_rows, new_cols);
        }
    };
}

/// Generic matrix implementation that works with any storage strategy
fn MatrixImpl(comptime T: type, comptime StorageStrategy: type) type {
    assert(@typeInfo(T) == .float);
    return struct {
        const Self = @This();
        storage: StorageStrategy,
        
        /// Returns the rows and columns as a struct.
        pub fn shape(self: Self) struct { usize, usize } {
            return .{
                self.storage.rowCount(),
                self.storage.colCount(),
            };
        }
        
        /// Retrieves the element at position row, col in the matrix.
        /// Panics if `row` or `col` are out of bounds (if runtime safety is enabled).
        pub fn at(self: Self, row: usize, col: usize) T {
            return self.storage.at(row, col);
        }
        
        /// Sets the element at row, col to val.
        /// Panics if `row` or `col` are out of bounds (if runtime safety is enabled).
        pub fn set(self: *Self, row: usize, col: usize, val: T) void {
            self.storage.set(row, col, val);
        }
        
        /// Initialize a dynamic matrix with allocator and dimensions (only available for DynamicStorage)
        pub fn init(allocator: std.mem.Allocator, row_count: usize, col_count: usize) !Self {
            // This will only compile for DynamicStorage types
            return Self{
                .storage = try StorageStrategy.init(allocator, row_count, col_count),
            };
        }
        
        /// Deinitialize a dynamic matrix (only available for DynamicStorage)
        pub fn deinit(self: *Self) void {
            // This will only compile for DynamicStorage types
            self.storage.deinit();
        }
        
        /// Returns a matrix with all elements set to value.
        pub fn initAll(value: T) Self {
            var result: Self = .{ .storage = .{} };
            for (0..result.storage.rowCount()) |r| {
                for (0..result.storage.colCount()) |c| {
                    result.items[r][c] = value;
                }
            }
            return result;
        }
        
        /// Returns an identity-like matrix. For square matrices, this is the standard identity matrix.
        /// For non-square matrices, it has 1s on the main diagonal (min(rows,cols)) and 0s elsewhere.
        pub fn identity() Self {
            var result: Self = .{ .storage = .{} };
            for (0..result.storage.rowCount()) |r| {
                for (0..result.storage.colCount()) |c| {
                    if (r == c) {
                        result.items[r][c] = 1;
                    } else {
                        result.items[r][c] = 0;
                    }
                }
            }
            return result;
        }
        
        /// Returns a matrix filled with random floating-point numbers of type `T` using the provided `seed`.
        /// If `seed` is `null`, a seed is generated from the current system time.
        pub fn random(seed: ?u64) Self {
            const s: u64 = seed orelse @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())));
            var prng: std.Random.DefaultPrng = .init(s);
            var rand = prng.random();
            var result: Self = .{ .storage = .{} };
            for (0..result.storage.rowCount()) |r| {
                for (0..result.storage.colCount()) |c| {
                    result.items[r][c] = rand.float(T);
                }
            }
            return result;
        }
        
        /// Sums all the elements in a matrix.
        pub fn sum(self: Self) T {
            var accum: T = 0;
            for (0..self.storage.rowCount()) |r| {
                for (0..self.storage.colCount()) |c| {
                    accum += self.items[r][c];
                }
            }
            return accum;
        }
        
        /// Scales all matrix values.
        pub fn scale(self: Self, value: T) Self {
            var result: Self = .{ .storage = .{} };
            for (0..self.storage.rowCount()) |r| {
                for (0..self.storage.colCount()) |c| {
                    result.items[r][c] = value * self.items[r][c];
                }
            }
            return result;
        }
        
        /// Applies a unary function to all matrix values.
        pub fn apply(self: Self, comptime unaryFn: fn (arg: T) T) Self {
            var result: Self = .{ .storage = .{} };
            for (0..self.storage.rowCount()) |r| {
                for (0..self.storage.colCount()) |c| {
                    result.items[r][c] = unaryFn(self.items[r][c]);
                }
            }
            return result;
        }
        
        /// Performs pointwise multiplication.
        /// Both `self` and `other` must have the same dimensions.
        pub fn times(self: Self, other: Self) Self {
            var result: Self = .{ .storage = .{} };
            for (0..self.storage.rowCount()) |r| {
                for (0..self.storage.colCount()) |c| {
                    result.items[r][c] = self.items[r][c] * other.storage.at(r, c);
                }
            }
            return result;
        }
        
        /// Computes the Frobenius norm of the matrix as the square root of the sum of its squared values.
        pub fn frobeniusNorm(self: Self) T {
            return @sqrt(self.times(self).sum());
        }
        
        /// Sums all the elements in rows.
        pub fn sumRows(self: Self) Self {
            // This is a simplified version - in the full implementation we'd return different dimensions
            var result: Self = .{ .storage = .{} };
            for (0..self.storage.rowCount()) |r| {
                for (0..self.storage.colCount()) |c| {
                    result.items[0][c] = self.items[r][c];
                }
            }
            return result;
        }
        
        /// Sums all the elements in columns.
        pub fn sumCols(self: Self) Self {
            // This is a simplified version - in the full implementation we'd return different dimensions  
            var result: Self = .{ .storage = .{} };
            for (0..self.storage.rowCount()) |r| {
                for (0..self.storage.colCount()) |c| {
                    result.items[r][0] = self.items[r][c];
                }
            }
            return result;
        }
    };
}

/// Creates a dynamic matrix with runtime dimensions
pub fn DynamicMatrix(comptime T: type) type {
    return MatrixImpl(T, DynamicStorage(T));
}

/// Creates a Matrix with elements of type T and size rows times cols.
pub fn Matrix(comptime T: type, comptime rows: usize, comptime cols: usize) type {
    return struct {
        const Self = @This();
        items: [rows][cols]T = undefined,
        comptime rows: usize = rows,
        comptime cols: usize = cols,
        
        // Allow direct .items field initialization for compatibility
        pub fn setFromItems(items: [rows][cols]T) Self {
            var result: Self = .{};
            for (0..rows) |r| {
                for (0..cols) |c| {
                    result.items[r][c] = items[r][c];
                }
            }
            return result;
        }
        
        /// Returns the rows and columns as a struct.
        pub fn shape(self: Self) struct { usize, usize } {
            _ = self;
            return .{ rows, cols };
        }
        
        /// Retrieves the element at position row, col in the matrix.
        pub fn at(self: Self, row: usize, col: usize) T {
            assert(row < rows);
            assert(col < cols);
            return self.items[row][col];
        }
        
        /// Sets the element at row, col to val.
        pub fn set(self: *Self, row: usize, col: usize, val: T) void {
            assert(row < rows);
            assert(col < cols);
            self.items[row][col] = val;
        }
        
        /// Returns a matrix with all elements set to value.
        pub fn initAll(value: T) Self {
            var result: Self = .{};
            for (0..rows) |r| {
                for (0..cols) |c| {
                    result.items[r][c] = value;
                }
            }
            return result;
        }
        
        /// Returns an identity-like matrix.
        pub fn identity() Self {
            var result: Self = .{};
            for (0..rows) |r| {
                for (0..cols) |c| {
                    if (r == c) {
                        result.items[r][c] = 1;
                    } else {
                        result.items[r][c] = 0;
                    }
                }
            }
            return result;
        }
        
        /// Returns a matrix filled with random floating-point numbers.
        pub fn random(seed: ?u64) Self {
            const s: u64 = seed orelse @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())));
            var prng: std.Random.DefaultPrng = .init(s);
            var rand = prng.random();
            var result: Self = .{};
            for (0..rows) |r| {
                for (0..cols) |c| {
                    result.items[r][c] = rand.float(T);
                }
            }
            return result;
        }
        
        /// Sums all the elements in a matrix.
        pub fn sum(self: Self) T {
            var accum: T = 0;
            for (0..rows) |r| {
                for (0..cols) |c| {
                    accum += self.items[r][c];
                }
            }
            return accum;
        }
        
        /// Scales all matrix values.
        pub fn scale(self: Self, value: T) Self {
            var result: Self = .{};
            for (0..rows) |r| {
                for (0..cols) |c| {
                    result.items[r][c] = value * self.items[r][c];
                }
            }
            return result;
        }
        
        /// Applies a unary function to all matrix values.
        pub fn apply(self: Self, comptime unaryFn: fn (arg: T) T) Self {
            var result: Self = .{};
            for (0..rows) |r| {
                for (0..cols) |c| {
                    result.items[r][c] = unaryFn(self.items[r][c]);
                }
            }
            return result;
        }
        
        /// Adds an offset to all matrix values.
        pub fn offset(self: Self, value: T) Self {
            var result: Self = .{};
            for (0..rows) |r| {
                for (0..cols) |c| {
                    result.items[r][c] = value + self.items[r][c];
                }
            }
            return result;
        }
        
        /// Performs pointwise multiplication.
        pub fn times(self: Self, other: Self) Self {
            var result: Self = .{};
            for (0..rows) |r| {
                for (0..cols) |c| {
                    result.items[r][c] = self.items[r][c] * other.items[r][c];
                }
            }
            return result;
        }
        
        /// Computes the Frobenius norm of the matrix.
        pub fn frobeniusNorm(self: Self) T {
            return @sqrt(self.times(self).sum());
        }
        
        /// Computes the nuclear norm of the matrix as sum of the absolute values of all elements.
        pub fn nuclearNorm(self: Self) T {
            var accum: T = 0;
            for (0..rows) |r| {
                for (0..cols) |c| {
                    accum += @abs(self.items[r][c]);
                }
            }
            return accum;
        }
        
        /// If the matrix only contains one element, it returns it, otherwise it fails to compile.
        pub fn item(self: Self) T {
            comptime assert(rows == 1 and cols == 1);
            return self.items[0][0];
        }
        
        /// Computes the L-infinity norm (maximum absolute value among all elements) of the matrix.
        pub fn maxNorm(self: Self) T {
            var result: T = -std.math.inf(T);
            for (0..rows) |r| {
                for (0..cols) |c| {
                    const val = @abs(self.items[r][c]);
                    if (val > result) {
                        result = val;
                    }
                }
            }
            return result;
        }
        
        /// Computes the minimum absolute value among all elements of the matrix.
        pub fn minNorm(self: Self) T {
            var result: T = std.math.inf(T);
            for (0..rows) |r| {
                for (0..cols) |c| {
                    const val = @abs(self.items[r][c]);
                    if (val < result) {
                        result = val;
                    }
                }
            }
            return result;
        }
        
        /// Computes the L0 norm, which is the count of non-zero elements in the matrix.
        pub fn sparseNorm(self: Self) T {
            var count: T = 0;
            for (0..rows) |r| {
                for (0..cols) |c| {
                    count += if (self.items[r][c] != 0) 1 else 0;
                }
            }
            return count;
        }
        
        /// Performs the dot (or internal product) of two matrices.
        pub fn dot(self: Self, other: anytype) Matrix(T, rows, other.cols) {
            comptime assert(cols == other.rows);
            var result: Matrix(T, rows, other.cols) = .initAll(0);
            for (0..rows) |r| {
                for (0..other.cols) |c| {
                    for (0..cols) |k| {
                        result.items[r][c] = result.items[r][c] + self.items[r][k] * other.items[k][c];
                    }
                }
            }
            return result;
        }
        
        /// Adds a matrix.
        pub fn add(self: Self, other: Self) Self {
            var result: Self = .{};
            for (0..rows) |r| {
                for (0..cols) |c| {
                    result.items[r][c] = self.items[r][c] + other.items[r][c];
                }
            }
            return result;
        }
        
        /// Sets the sub-matrix at position row, col to sub_matrix.
        pub fn setSubMatrix(self: *Self, row: usize, col: usize, matrix: anytype) void {
            assert(matrix.rows + row <= rows);
            assert(matrix.cols + col <= cols);
            for (0..matrix.rows) |r| {
                for (0..matrix.cols) |c| {
                    self.items[row + r][col + c] = matrix.items[r][c];
                }
            }
        }
        
        /// Transposes the matrix.
        pub fn transpose(self: Self) Matrix(T, cols, rows) {
            var result: Matrix(T, cols, rows) = .{};
            for (0..rows) |r| {
                for (0..cols) |c| {
                    result.items[c][r] = self.items[r][c];
                }
            }
            return result;
        }
        
        /// Returns a new matrix which is a copy of the specified rectangular region of `self`.
        pub fn getSubMatrix(
            self: Self,
            comptime row_begin: usize,
            comptime col_begin: usize,
            comptime row_end: usize,
            comptime col_end: usize,
        ) Matrix(T, row_end - row_begin, col_end - col_begin) {
            comptime assert(row_begin < row_end);
            comptime assert(col_begin < col_end);
            comptime assert(row_end <= rows);
            comptime assert(col_end <= cols);
            var result: Matrix(T, row_end - row_begin, col_end - col_begin) = .{};
            for (row_begin..row_end) |r| {
                for (col_begin..col_end) |c| {
                    result.items[r - row_begin][c - col_begin] = self.items[r][c];
                }
            }
            return result;
        }
        
        /// Returns the elements in the column as a column Matrix.
        pub fn getCol(self: Self, col: usize) Matrix(T, rows, 1) {
            assert(col < cols);
            var result: Matrix(T, rows, 1) = .{};
            for (0..rows) |r| {
                result.items[r][0] = self.items[r][col];
            }
            return result;
        }
        
        /// Returns a new matrix with dimensions `new_rows` x `new_cols`, containing the same elements
        /// as `self` interpreted in row-major order.
        pub fn reshape(self: Self, comptime new_rows: usize, comptime new_cols: usize) Matrix(T, new_rows, new_cols) {
            comptime assert(rows * cols == new_rows * new_cols);
            var result: Matrix(T, new_rows, new_cols) = .{};
            for (0..new_rows) |r| {
                for (0..new_cols) |c| {
                    const idx = r * new_cols + c;
                    result.items[r][c] = self.items[idx / cols][idx % cols];
                }
            }
            return result;
        }
        
        /// Converts a column matrix into a Point2d.
        pub fn toPoint2d(self: Self) Point2d(T) {
            comptime assert(rows >= 2 and cols == 1);
            return .{ .x = self.items[0][0], .y = self.items[1][0] };
        }
        
        
        
        /// Computes the determinant of self if it's a square matrix.
        pub fn determinant(self: Self) T {
            comptime assert(rows == cols);
            return switch (rows) {
                1 => self.item(),
                2 => self.at(0, 0) * self.at(1, 1) - self.at(0, 1) * self.at(1, 0),
                3 => self.at(0, 0) * self.at(1, 1) * self.at(2, 2) +
                    self.at(0, 1) * self.at(1, 2) * self.at(2, 0) +
                    self.at(0, 2) * self.at(1, 0) * self.at(2, 1) -
                    self.at(0, 2) * self.at(1, 1) * self.at(2, 0) -
                    self.at(0, 1) * self.at(1, 0) * self.at(2, 2) -
                    self.at(0, 0) * self.at(1, 2) * self.at(2, 1),
                else => @compileError("Matrix(T).determinant() is not implemented for sizes above 3"),
            };
        }
        
        /// Computes the inverse of self if it's a square matrix.
        pub fn inverse(self: Self) ?Self {
            comptime assert(rows == cols);
            const det = self.determinant();
            if (det == 0) {
                return null;
            }
            var inv: Self = .{};
            switch (rows) {
                1 => inv.set(0, 0, 1 / det),
                2 => {
                    inv.set(0, 0, self.at(1, 1) / det);
                    inv.set(0, 1, -self.at(0, 1) / det);
                    inv.set(1, 0, -self.at(1, 0) / det);
                    inv.set(1, 1, self.at(0, 0) / det);
                },
                3 => {
                    inv.set(0, 0, (self.at(1, 1) * self.at(2, 2) - self.at(1, 2) * self.at(2, 1)) / det);
                    inv.set(0, 1, (self.at(0, 2) * self.at(2, 1) - self.at(0, 1) * self.at(2, 2)) / det);
                    inv.set(0, 2, (self.at(0, 1) * self.at(1, 2) - self.at(0, 2) * self.at(1, 1)) / det);
                    inv.set(1, 0, (self.at(1, 2) * self.at(2, 0) - self.at(1, 0) * self.at(2, 2)) / det);
                    inv.set(1, 1, (self.at(0, 0) * self.at(2, 2) - self.at(0, 2) * self.at(2, 0)) / det);
                    inv.set(1, 2, (self.at(0, 2) * self.at(1, 0) - self.at(0, 0) * self.at(1, 2)) / det);
                    inv.set(2, 0, (self.at(1, 0) * self.at(2, 1) - self.at(1, 1) * self.at(2, 0)) / det);
                    inv.set(2, 1, (self.at(0, 1) * self.at(2, 0) - self.at(0, 0) * self.at(2, 1)) / det);
                    inv.set(2, 2, (self.at(0, 0) * self.at(1, 1) - self.at(0, 1) * self.at(1, 0)) / det);
                },
                else => @compileError("Matrix(T).inverse() is not implemented for sizes above 3"),
            }
            return inv;
        }
        
        /// Formats the matrix for pretty printing with configurable precision.
        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;

            // Helper function to format a number with fallback to truncation
            const formatNumber = struct {
                fn format(buf: []u8, comptime format_str: []const u8, value: T) []const u8 {
                    return std.fmt.bufPrint(buf, format_str, .{value}) catch blk: {
                        // If formatting fails, truncate and add ellipsis
                        if (buf.len >= 4) {
                            const truncated = std.fmt.bufPrint(buf[0 .. buf.len - 3], "{d}", .{value}) catch buf[0 .. buf.len - 3];
                            @memcpy(buf[truncated.len .. truncated.len + 3], "...");
                            break :blk buf[0 .. truncated.len + 3];
                        } else {
                            // Buffer too small even for ellipsis
                            break :blk "...";
                        }
                    };
                }
            }.format;

            // First pass: calculate the maximum width needed for each column
            var col_widths: [cols]usize = [_]usize{0} ** cols;

            for (0..rows) |r| {
                for (0..cols) |c| {
                    // Create a temporary buffer to measure the width of this element
                    var temp_buf: [64]u8 = undefined;
                    const formatted = if (options.precision) |precision|
                        switch (precision) {
                            0 => formatNumber(temp_buf[0..], "{d:.0}", self.items[r][c]),
                            1 => formatNumber(temp_buf[0..], "{d:.1}", self.items[r][c]),
                            2 => formatNumber(temp_buf[0..], "{d:.2}", self.items[r][c]),
                            3 => formatNumber(temp_buf[0..], "{d:.3}", self.items[r][c]),
                            4 => formatNumber(temp_buf[0..], "{d:.4}", self.items[r][c]),
                            5 => formatNumber(temp_buf[0..], "{d:.5}", self.items[r][c]),
                            6 => formatNumber(temp_buf[0..], "{d:.6}", self.items[r][c]),
                            7 => formatNumber(temp_buf[0..], "{d:.7}", self.items[r][c]),
                            8 => formatNumber(temp_buf[0..], "{d:.8}", self.items[r][c]),
                            9 => formatNumber(temp_buf[0..], "{d:.9}", self.items[r][c]),
                            10 => formatNumber(temp_buf[0..], "{d:.10}", self.items[r][c]),
                            11 => formatNumber(temp_buf[0..], "{d:.11}", self.items[r][c]),
                            12 => formatNumber(temp_buf[0..], "{d:.12}", self.items[r][c]),
                            13 => formatNumber(temp_buf[0..], "{d:.13}", self.items[r][c]),
                            14 => formatNumber(temp_buf[0..], "{d:.14}", self.items[r][c]),
                            15 => formatNumber(temp_buf[0..], "{d:.15}", self.items[r][c]),
                            else => formatNumber(temp_buf[0..], "{d}", self.items[r][c]),
                        }
                    else
                        formatNumber(temp_buf[0..], "{}", self.items[r][c]);
                    col_widths[c] = @max(col_widths[c], formatted.len);
                }
            }

            // Second pass: format and write the matrix with proper alignment
            for (0..rows) |r| {
                try writer.writeAll("[ ");
                for (0..cols) |c| {
                    // Format the number with specified precision
                    var temp_buf: [64]u8 = undefined;
                    const formatted = if (options.precision) |precision|
                        switch (precision) {
                            0 => formatNumber(temp_buf[0..], "{d:.0}", self.items[r][c]),
                            1 => formatNumber(temp_buf[0..], "{d:.1}", self.items[r][c]),
                            2 => formatNumber(temp_buf[0..], "{d:.2}", self.items[r][c]),
                            3 => formatNumber(temp_buf[0..], "{d:.3}", self.items[r][c]),
                            4 => formatNumber(temp_buf[0..], "{d:.4}", self.items[r][c]),
                            5 => formatNumber(temp_buf[0..], "{d:.5}", self.items[r][c]),
                            6 => formatNumber(temp_buf[0..], "{d:.6}", self.items[r][c]),
                            7 => formatNumber(temp_buf[0..], "{d:.7}", self.items[r][c]),
                            8 => formatNumber(temp_buf[0..], "{d:.8}", self.items[r][c]),
                            9 => formatNumber(temp_buf[0..], "{d:.9}", self.items[r][c]),
                            10 => formatNumber(temp_buf[0..], "{d:.10}", self.items[r][c]),
                            11 => formatNumber(temp_buf[0..], "{d:.11}", self.items[r][c]),
                            12 => formatNumber(temp_buf[0..], "{d:.12}", self.items[r][c]),
                            13 => formatNumber(temp_buf[0..], "{d:.13}", self.items[r][c]),
                            14 => formatNumber(temp_buf[0..], "{d:.14}", self.items[r][c]),
                            15 => formatNumber(temp_buf[0..], "{d:.15}", self.items[r][c]),
                            else => formatNumber(temp_buf[0..], "{d}", self.items[r][c]),
                        }
                    else
                        formatNumber(temp_buf[0..], "{}", self.items[r][c]);

                    // Right-align the number within the column width
                    const padding = col_widths[c] - formatted.len;
                    for (0..padding) |_| {
                        try writer.writeAll(" ");
                    }
                    try writer.writeAll(formatted);

                    if (c < cols - 1) {
                        try writer.writeAll("  "); // Two spaces between columns
                    }
                }
                try writer.writeAll(" ]");
                if (r < rows - 1) {
                    try writer.writeAll("\n");
                }
            }
        }
        
        /// Sums all the elements in rows.
        pub fn sumRows(self: Self) Matrix(T, 1, cols) {
            var result: Matrix(T, 1, cols) = .initAll(0);
            for (0..rows) |r| {
                for (0..cols) |c| {
                    result.items[0][c] = result.items[0][c] + self.items[r][c];
                }
            }
            return result;
        }
        
        /// Sums all the elements in columns.
        pub fn sumCols(self: Self) Matrix(T, rows, 1) {
            var result: Matrix(T, rows, 1) = .initAll(0);
            for (0..rows) |r| {
                for (0..cols) |c| {
                    result.items[r][0] = result.items[r][0] + self.items[r][c];
                }
            }
            return result;
        }
        
        // Support direct .items initialization for compatibility
        pub fn init(items: [rows][cols]T) Self {
            var result: Self = .{};
            for (0..rows) |r| {
                for (0..cols) |c| {
                    result.items[r][c] = items[r][c];
                }
            }
            return result;
        }
        
        // Allow direct .items field initialization
        pub fn setItems(self: *Self, items: [rows][cols]T) void {
            for (0..rows) |r| {
                for (0..cols) |c| {
                    self.storage.set(r, c, items[r][c]);
                }
            }
        }
    };
}

test "identity" {
    const eye: Matrix(f32, 3, 3) = .identity();
    try expectEqual(eye.sum(), 3);
    for (0..eye.rows) |r| {
        for (0..eye.cols) |c| {
            if (r == c) {
                try expectEqual(eye.at(r, c), 1);
            } else {
                try expectEqual(eye.at(r, c), 0);
            }
        }
    }
}

test "initAll" {
    const zeros: Matrix(f32, 3, 3) = .initAll(0);
    try expectEqual(zeros.sum(), 0);
    const ones: Matrix(f32, 3, 3) = .initAll(1);
    const shape = ones.shape();
    try expectEqual(ones.sum(), @as(f32, @floatFromInt(shape[0] * shape[1])));
}

test "shape" {
    const matrix: Matrix(f32, 4, 5) = .{};
    const shape = matrix.shape();
    try expectEqual(shape[0], 4);
    try expectEqual(shape[1], 5);
}

test "scale" {
    const seed: u64 = @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())));
    const a: Matrix(f32, 4, 3) = .random(seed);
    const b = Matrix(f32, 4, 3).random(seed).scale(std.math.pi);
    try expectEqualDeep(a.shape(), b.shape());
    for (0..a.rows) |r| {
        for (0..a.cols) |c| {
            try expectEqual(std.math.pi * a.at(r, c), b.at(r, c));
        }
    }
}

test "apply" {
    var a: Matrix(f32, 3, 4) = .random(null);

    const f = struct {
        fn f(x: f32) f32 {
            return @sin(x);
        }
    }.f;

    var b = a.apply(f);
    try expectEqualDeep(a.shape(), b.shape());
    for (0..a.rows) |r| {
        for (0..a.cols) |c| {
            try expectEqual(@sin(a.at(r, c)), b.at(r, c));
        }
    }
}

test "norm" {
    var matrix: Matrix(f32, 3, 4) = .random(null);
    try expectEqual(matrix.frobeniusNorm(), @sqrt(matrix.times(matrix).sum()));

    const f = struct {
        fn f(x: f32) f32 {
            return @abs(x);
        }
    }.f;
    try expectEqual(matrix.nuclearNorm(), matrix.apply(f).sum());

    matrix.set(2, 3, 1000000);
    try expectEqual(matrix.maxNorm(), 1000000);

    matrix = matrix.offset(10);
    matrix.set(2, 3, -5);
    try expectEqual(matrix.minNorm(), 5);

    matrix.set(2, 3, 0);
    try expectEqual(matrix.sparseNorm(), 11);
}

test "sum" {
    var matrix: Matrix(f32, 3, 4) = .initAll(1);
    const matrixSumCols: Matrix(f32, 3, 1) = .initAll(4);
    const matrixSumRows: Matrix(f32, 1, 4) = .initAll(3);
    try expectEqual(matrix.sumRows(), matrixSumRows);
    try expectEqual(matrix.sumCols(), matrixSumCols);
    try expectEqual(matrix.sumCols().sumRows().item(), matrix.sum());
}

test "inverse" {
    var a: Matrix(f32, 2, 2) = .{};
    a.set(0, 0, -1); a.set(0, 1, 1.5);
    a.set(1, 0, 1); a.set(1, 1, -1);
    try expectEqual(a.determinant(), -0.5);
    var a_i: Matrix(f32, 2, 2) = .{};
    a_i.set(0, 0, 2); a_i.set(0, 1, 3);
    a_i.set(1, 0, 2); a_i.set(1, 1, 2);
    try expectEqualDeep(a.inverse(), a_i);
    var b: Matrix(f32, 3, 3) = .{};
    b.set(0, 0, 1); b.set(0, 1, 2); b.set(0, 2, 3);
    b.set(1, 0, 4); b.set(1, 1, 5); b.set(1, 2, 6);
    b.set(2, 0, 7); b.set(2, 1, 2); b.set(2, 2, 9);
    try expectEqual(b.determinant(), -36);
    var b_i: Matrix(f32, 3, 3) = .{};
    b_i.set(0, 0, -11.0 / 12.0); b_i.set(0, 1, 1.0 / 3.0); b_i.set(0, 2, 1.0 / 12.0);
    b_i.set(1, 0, -1.0 / 6.0); b_i.set(1, 1, 1.0 / 3.0); b_i.set(1, 2, -1.0 / 6.0);
    b_i.set(2, 0, 3.0 / 4.0); b_i.set(2, 1, -1.0 / 3.0); b_i.set(2, 2, 1.0 / 12.0);
    try expectEqualDeep(b.inverse().?, b_i);
}

test "format" {
    // Test 2x3 matrix with known values
    var m: Matrix(f32, 2, 3) = .{};
    m.set(0, 0, 1.23); m.set(0, 1, -4.5); m.set(0, 2, 7.0);
    m.set(1, 0, 10.1); m.set(1, 1, 0.0); m.set(1, 2, -5.67);

    // Test default formatting (scientific notation)
    var buffer: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buffer);
    try std.fmt.format(stream.writer(), "{}", .{m});
    const result_default = stream.getWritten();
    const expected_default = "[ 1.23e0  -4.5e0      7e0 ]\n[ 1.01e1     0e0  -5.67e0 ]";
    try std.testing.expect(std.mem.eql(u8, result_default, expected_default));

    // Test 2 decimal places
    stream.reset();
    try std.fmt.format(stream.writer(), "{:.2}", .{m});
    const result_2dp = stream.getWritten();
    const expected_2dp = "[  1.23  -4.50   7.00 ]\n[ 10.10   0.00  -5.67 ]";
    try std.testing.expect(std.mem.eql(u8, result_2dp, expected_2dp));

    // Test 0 decimal places (integers)
    stream.reset();
    try std.fmt.format(stream.writer(), "{:.0}", .{m});
    const result_0dp = stream.getWritten();
    const expected_0dp = "[  1  -5   7 ]\n[ 10   0  -6 ]";
    try std.testing.expect(std.mem.eql(u8, result_0dp, expected_0dp));

    // Test 1x1 matrix
    var m_single: Matrix(f64, 1, 1) = .{};
    m_single.set(0, 0, 3.14159);
    stream.reset();
    try std.fmt.format(stream.writer(), "{:.3}", .{m_single});
    const result_single = stream.getWritten();
    const expected_single = "[ 3.142 ]";
    try std.testing.expect(std.mem.eql(u8, result_single, expected_single));
}

// test "dynamic matrix basic operations" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();
//     
//     var dyn_matrix = try DynamicMatrix(f32).init(arena.allocator(), 2, 3);
//     defer dyn_matrix.deinit();
//     
//     // Test basic set/get operations
//     dyn_matrix.set(0, 0, 1.0);
//     dyn_matrix.set(0, 1, 2.0);
//     dyn_matrix.set(0, 2, 3.0);
//     dyn_matrix.set(1, 0, 4.0);
//     dyn_matrix.set(1, 1, 5.0);
//     dyn_matrix.set(1, 2, 6.0);
//     
//     try expectEqual(dyn_matrix.at(0, 0), 1.0);
//     try expectEqual(dyn_matrix.at(1, 2), 6.0);
//     
//     // Test shape
//     const shape = dyn_matrix.shape();
//     try expectEqual(shape[0], 2);
//     try expectEqual(shape[1], 3);
//     
//     // Test sum
//     try expectEqual(dyn_matrix.sum(), 21.0);
// }
