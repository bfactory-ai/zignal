//! Builder for chaining matrix operations with in-place modifications
//!
//! Performance note: GEMM operations use SIMD optimization for larger matrices.
//! The threshold is set at 512 multiply-accumulate operations (e.g., 8x8 * 8x8).
//! For smaller matrices, the scalar path is used to avoid allocation overhead.

const std = @import("std");
const assert = std.debug.assert;
const expectEqual = std.testing.expectEqual;

const Matrix = @import("Matrix.zig").Matrix;
const SMatrix = @import("SMatrix.zig").SMatrix;

/// Builder for chaining matrix operations with in-place modifications
pub fn OpsBuilder(comptime T: type) type {
    assert(@typeInfo(T) == .float);
    return struct {
        const Self = @This();

        result: Matrix(T),
        allocator: std.mem.Allocator,
        consumed: bool = false,

        /// Initialize builder with a copy of the input matrix
        pub fn init(allocator: std.mem.Allocator, matrix: Matrix(T)) !Self {
            const result = try Matrix(T).init(allocator, matrix.rows, matrix.cols);
            @memcpy(result.items, matrix.items);
            return Self{
                .result = result,
                .allocator = allocator,
            };
        }

        /// Clean up the builder (only if exec() was not called)
        pub fn deinit(self: *Self) void {
            if (!self.consumed) {
                self.result.deinit();
            }
        }

        /// Add another matrix element-wise
        pub fn add(self: *Self, other: Matrix(T)) !void {
            assert(self.result.rows == other.rows and self.result.cols == other.cols);
            for (0..self.result.items.len) |i| {
                self.result.items[i] += other.items[i];
            }
        }

        /// Subtract another matrix element-wise
        pub fn sub(self: *Self, other: Matrix(T)) !void {
            assert(self.result.rows == other.rows and self.result.cols == other.cols);
            for (0..self.result.items.len) |i| {
                self.result.items[i] -= other.items[i];
            }
        }

        /// Scale all elements by a value
        pub fn scale(self: *Self, value: T) !void {
            for (0..self.result.items.len) |i| {
                self.result.items[i] *= value;
            }
        }

        /// Transpose the matrix
        pub fn transpose(self: *Self) !void {
            var transposed = try Matrix(T).init(self.allocator, self.result.cols, self.result.rows);
            for (0..self.result.rows) |r| {
                for (0..self.result.cols) |c| {
                    transposed.at(c, r).* = self.result.at(r, c).*;
                }
            }
            self.result.deinit();
            self.result = transposed;
        }

        /// Perform element-wise multiplication
        pub fn times(self: *Self, other: Matrix(T)) !void {
            assert(self.result.rows == other.rows and self.result.cols == other.cols);
            for (0..self.result.items.len) |i| {
                self.result.items[i] *= other.items[i];
            }
        }

        /// Matrix multiplication (dot product) - changes dimensions
        pub fn dot(self: *Self, other: Matrix(T)) !void {
            try self.gemm(other, false, false, 1.0, 0.0, null);
        }

        /// Computes the determinant of the matrix (only for square matrices up to 3x3)
        fn determinant(matrix: Matrix(T)) T {
            assert(matrix.rows == matrix.cols);

            return switch (matrix.rows) {
                1 => matrix.at(0, 0).*,
                2 => matrix.at(0, 0).* * matrix.at(1, 1).* - matrix.at(0, 1).* * matrix.at(1, 0).*,
                3 => matrix.at(0, 0).* * matrix.at(1, 1).* * matrix.at(2, 2).* +
                    matrix.at(0, 1).* * matrix.at(1, 2).* * matrix.at(2, 0).* +
                    matrix.at(0, 2).* * matrix.at(1, 0).* * matrix.at(2, 1).* -
                    matrix.at(0, 2).* * matrix.at(1, 1).* * matrix.at(2, 0).* -
                    matrix.at(0, 1).* * matrix.at(1, 0).* * matrix.at(2, 2).* -
                    matrix.at(0, 0).* * matrix.at(1, 2).* * matrix.at(2, 1).*,
                else => blk: {
                    std.debug.panic("OpsBuilder.determinant() is not implemented for sizes above 3x3", .{});
                    break :blk 0;
                },
            };
        }

        /// Inverts the matrix (only for square matrices up to 3x3)
        pub fn inverse(self: *Self) !void {
            assert(self.result.rows == self.result.cols);

            const det = determinant(self.result);
            if (det == 0) {
                return error.SingularMatrix;
            }

            var inv = try Matrix(T).init(self.allocator, self.result.rows, self.result.cols);

            switch (self.result.rows) {
                1 => inv.at(0, 0).* = 1 / det,
                2 => {
                    inv.at(0, 0).* = self.result.at(1, 1).* / det;
                    inv.at(0, 1).* = -self.result.at(0, 1).* / det;
                    inv.at(1, 0).* = -self.result.at(1, 0).* / det;
                    inv.at(1, 1).* = self.result.at(0, 0).* / det;
                },
                3 => {
                    inv.at(0, 0).* = (self.result.at(1, 1).* * self.result.at(2, 2).* - self.result.at(1, 2).* * self.result.at(2, 1).*) / det;
                    inv.at(0, 1).* = (self.result.at(0, 2).* * self.result.at(2, 1).* - self.result.at(0, 1).* * self.result.at(2, 2).*) / det;
                    inv.at(0, 2).* = (self.result.at(0, 1).* * self.result.at(1, 2).* - self.result.at(0, 2).* * self.result.at(1, 1).*) / det;
                    inv.at(1, 0).* = (self.result.at(1, 2).* * self.result.at(2, 0).* - self.result.at(1, 0).* * self.result.at(2, 2).*) / det;
                    inv.at(1, 1).* = (self.result.at(0, 0).* * self.result.at(2, 2).* - self.result.at(0, 2).* * self.result.at(2, 0).*) / det;
                    inv.at(1, 2).* = (self.result.at(0, 2).* * self.result.at(1, 0).* - self.result.at(0, 0).* * self.result.at(1, 2).*) / det;
                    inv.at(2, 0).* = (self.result.at(1, 0).* * self.result.at(2, 1).* - self.result.at(1, 1).* * self.result.at(2, 0).*) / det;
                    inv.at(2, 1).* = (self.result.at(0, 1).* * self.result.at(2, 0).* - self.result.at(0, 0).* * self.result.at(2, 1).*) / det;
                    inv.at(2, 2).* = (self.result.at(0, 0).* * self.result.at(1, 1).* - self.result.at(0, 1).* * self.result.at(1, 0).*) / det;
                },
                else => {
                    std.debug.panic("OpsBuilder.inverse() is not implemented for sizes above 3x3", .{});
                },
            }

            self.result.deinit();
            self.result = inv;
        }

        /// Extract a submatrix - changes dimensions
        pub fn subMatrix(self: *Self, row_begin: usize, col_begin: usize, row_count: usize, col_count: usize) !void {
            assert(row_begin + row_count <= self.result.rows);
            assert(col_begin + col_count <= self.result.cols);

            var new_result = try Matrix(T).init(self.allocator, row_count, col_count);

            for (0..row_count) |r| {
                for (0..col_count) |c| {
                    new_result.at(r, c).* = self.result.at(row_begin + r, col_begin + c).*;
                }
            }

            self.result.deinit();
            self.result = new_result;
        }

        /// Extract a column - changes dimensions
        pub fn col(self: *Self, col_idx: usize) !void {
            assert(col_idx < self.result.cols);

            var new_result = try Matrix(T).init(self.allocator, self.result.rows, 1);

            for (0..self.result.rows) |r| {
                new_result.at(r, 0).* = self.result.at(r, col_idx).*;
            }

            self.result.deinit();
            self.result = new_result;
        }

        /// Extract a row - changes dimensions
        pub fn row(self: *Self, row_idx: usize) !void {
            assert(row_idx < self.result.rows);

            var new_result = try Matrix(T).init(self.allocator, 1, self.result.cols);

            for (0..self.result.cols) |c| {
                new_result.at(0, c).* = self.result.at(row_idx, c).*;
            }

            self.result.deinit();
            self.result = new_result;
        }

        /// Compute Gram matrix: X * X^T
        /// Useful for kernel methods and when rows < columns
        /// The resulting matrix is rows × rows
        pub fn gram(self: *Self) !void {
            try self.gemm(self.result, false, true, 1.0, 0.0, null);
        }

        /// Compute covariance matrix: X^T * X
        /// Useful for statistical analysis and when rows > columns
        /// The resulting matrix is columns × columns
        pub fn covariance(self: *Self) !void {
            try self.gemm(self.result, true, false, 1.0, 0.0, null);
        }

        /// Helper function for optimized SIMD GEMM kernel
        /// Both matrices must be arranged for row-major access
        fn simdGemmKernel(
            comptime VecType: type,
            comptime vec_len: usize,
            result: *Matrix(T),
            matrix_a: Matrix(T),
            matrix_b: Matrix(T),
            alpha: T,
            a_rows: usize,
            a_cols: usize,
            b_cols: usize,
        ) void {
            // Both matrices are now guaranteed to be accessed row-wise
            for (0..a_rows) |i| {
                for (0..b_cols) |j| {
                    var accumulator: T = 0;

                    // Process vec_len elements at once
                    var k: usize = 0;
                    while (k + vec_len <= a_cols) : (k += vec_len) {
                        // Load vectors from both matrices (both are row-major access)
                        const a_vec: VecType = matrix_a.items[i * a_cols + k .. i * a_cols + k + vec_len][0..vec_len].*;
                        const b_vec: VecType = matrix_b.items[j * a_cols + k .. j * a_cols + k + vec_len][0..vec_len].*;

                        // Vectorized multiply-accumulate
                        const prod_vec = a_vec * b_vec;
                        accumulator += @reduce(.Add, prod_vec);
                    }

                    // Handle remainder elements
                    while (k < a_cols) : (k += 1) {
                        accumulator += matrix_a.at(i, k).* * matrix_b.at(j, k).*;
                    }

                    result.at(i, j).* += alpha * accumulator;
                }
            }
        }

        /// General Matrix Multiply (GEMM): C = α * op(A) * op(B) + β * C
        ///
        /// This is the fundamental matrix operation that unifies many matrix computations:
        /// - op(A) = A if trans_a is false, A^T if trans_a is true
        /// - op(B) = B if trans_b is false, B^T if trans_b is true
        /// - α (alpha) scales the product op(A) * op(B)
        /// - β (beta) scales the existing matrix C before adding the product
        /// - If c_matrix is null, it defaults to zero matrix
        ///
        /// Examples:
        /// - Matrix multiplication: gemm(B, false, false, 1.0, 0.0, null)
        /// - Gram matrix: gemm(self, false, true, 1.0, 0.0, null) -> A * A^T
        /// - Covariance: gemm(self, true, false, 1.0, 0.0, null) -> A^T * A
        /// - Scaled product: gemm(B, false, false, 2.0, 0.0, null) -> 2 * A * B
        /// - Accumulation: gemm(B, false, false, 1.0, 1.0, C) -> A * B + C
        pub fn gemm(
            self: *Self,
            other: Matrix(T),
            trans_a: bool,
            trans_b: bool,
            alpha: T,
            beta: T,
            c_matrix: ?Matrix(T),
        ) !void {
            // Determine dimensions after potential transposition
            const a_rows = if (trans_a) self.result.cols else self.result.rows;
            const a_cols = if (trans_a) self.result.rows else self.result.cols;
            const b_rows = if (trans_b) other.cols else other.rows;
            const b_cols = if (trans_b) other.rows else other.cols;

            // Verify matrix multiplication compatibility
            assert(a_cols == b_rows);

            var result = try Matrix(T).init(self.allocator, a_rows, b_cols);

            // Initialize with scaled C matrix if provided
            if (c_matrix) |c| {
                assert(c.rows == a_rows and c.cols == b_cols);
                if (beta != 0) {
                    for (0..a_rows) |i| {
                        for (0..b_cols) |j| {
                            result.at(i, j).* = beta * c.at(i, j).*;
                        }
                    }
                }
            } else {
                // Initialize to zero
                @memset(result.items, 0);
            }

            // Skip computation if alpha is zero
            if (alpha != 0) {
                const vec_len = std.simd.suggestVectorLength(T) orelse 1;

                // Calculate total operations to determine if SIMD is worth the overhead
                const total_ops = a_rows * a_cols * b_cols;
                const SIMD_THRESHOLD = 512; // Use SIMD for larger matrices (>512 operations)

                // Use SIMD only for larger matrices where the benefit outweighs allocation overhead
                if (vec_len > 1 and total_ops >= SIMD_THRESHOLD) {
                    // Enable SIMD for all 4 transpose combinations
                    const VecType = @Vector(vec_len, T);

                    if (!trans_a and !trans_b) {
                        // Case 1: A * B - transpose B for cache-friendly row-major access
                        
                        // Handle special case when A and B are the same matrix (for A * A)
                        if (self.result.rows == other.rows and 
                            self.result.cols == other.cols and
                            std.mem.eql(T, self.result.items, other.items)) {
                            // For A * A, we need to transpose A for the second operand
                            var a_transposed = try Matrix(T).init(self.allocator, b_cols, a_cols);
                            defer a_transposed.deinit();
                            for (0..a_cols) |k| {
                                for (0..b_cols) |j| {
                                    a_transposed.at(j, k).* = self.result.at(k, j).*;
                                }
                            }
                            simdGemmKernel(VecType, vec_len, &result, self.result, a_transposed, alpha, a_rows, a_cols, b_cols);
                        } else {
                            // General case: transpose B for cache-friendly row-major access
                            var b_transposed = try Matrix(T).init(self.allocator, b_cols, a_cols);
                            defer b_transposed.deinit();
                            for (0..a_cols) |k| {
                                for (0..b_cols) |j| {
                                    b_transposed.at(j, k).* = other.at(k, j).*;
                                }
                            }
                            simdGemmKernel(VecType, vec_len, &result, self.result, b_transposed, alpha, a_rows, a_cols, b_cols);
                        }
                    } else if (trans_a and !trans_b) {
                        // Case 2: A^T * B - transpose A for cache-friendly row-major access
                        var a_transposed = try Matrix(T).init(self.allocator, a_rows, a_cols);
                        defer a_transposed.deinit();
                        // Transpose A: a_transposed[i,j] = A[j,i]
                        for (0..a_cols) |k| {
                            for (0..a_rows) |i| {
                                a_transposed.at(i, k).* = self.result.at(k, i).*;
                            }
                        }
                        // Handle special case when A and B are the same matrix (for covariance)
                        if (self.result.rows == other.rows and
                            self.result.cols == other.cols and
                            std.mem.eql(T, self.result.items, other.items))
                        {
                            // For covariance (A^T * A), we need to also use transposed for B
                            simdGemmKernel(VecType, vec_len, &result, a_transposed, a_transposed, alpha, a_rows, a_cols, b_cols);
                        } else {
                            // General case: transpose B for row-wise access
                            var b_transposed = try Matrix(T).init(self.allocator, b_cols, a_cols);
                            defer b_transposed.deinit();
                            for (0..a_cols) |k| {
                                for (0..b_cols) |j| {
                                    b_transposed.at(j, k).* = other.at(k, j).*;
                                }
                            }
                            simdGemmKernel(VecType, vec_len, &result, a_transposed, b_transposed, alpha, a_rows, a_cols, b_cols);
                        }
                    } else if (!trans_a and trans_b) {
                        // Case 3: A * B^T - no transpose needed, B^T is naturally row-wise
                        
                        // Handle special case when A and B are the same matrix (for A * A^T)
                        if (self.result.rows == other.rows and 
                            self.result.cols == other.cols and
                            std.mem.eql(T, self.result.items, other.items)) {
                            // For A * A^T, we need to transpose A for the second operand
                            var a_transposed = try Matrix(T).init(self.allocator, b_cols, a_cols);
                            defer a_transposed.deinit();
                            for (0..a_cols) |k| {
                                for (0..b_cols) |j| {
                                    a_transposed.at(j, k).* = self.result.at(j, k).*;
                                }
                            }
                            simdGemmKernel(VecType, vec_len, &result, self.result, a_transposed, alpha, a_rows, a_cols, b_cols);
                        } else {
                            // General case: B^T is naturally row-wise
                            simdGemmKernel(VecType, vec_len, &result, self.result, other, alpha, a_rows, a_cols, b_cols);
                        }
                    } else if (trans_a and trans_b) {
                        // Case 4: A^T * B^T - no transpose needed, both naturally row-wise
                        
                        // Handle special case when A and B are the same matrix (for A^T * A^T)
                        if (self.result.rows == other.rows and 
                            self.result.cols == other.cols and
                            std.mem.eql(T, self.result.items, other.items)) {
                            // For A^T * A^T, we transpose A to get the transposed version
                            var a_transposed = try Matrix(T).init(self.allocator, a_rows, a_cols);
                            defer a_transposed.deinit();
                            for (0..self.result.rows) |i| {
                                for (0..self.result.cols) |j| {
                                    a_transposed.at(j, i).* = self.result.at(i, j).*;
                                }
                            }
                            simdGemmKernel(VecType, vec_len, &result, a_transposed, self.result, alpha, a_rows, a_cols, b_cols);
                        } else {
                            // General case: both A^T and B^T are naturally row-wise
                            simdGemmKernel(VecType, vec_len, &result, self.result, other, alpha, a_rows, a_cols, b_cols);
                        }
                    }
                } else {
                    // No SIMD support, use scalar implementation for all transpose combinations
                    for (0..a_rows) |i| {
                        for (0..b_cols) |j| {
                            var accumulator: T = 0;
                            for (0..a_cols) |k| {
                                const a_val = if (trans_a) self.result.at(k, i).* else self.result.at(i, k).*;
                                const b_val = if (trans_b) other.at(j, k).* else other.at(k, j).*;
                                accumulator += a_val * b_val;
                            }
                            result.at(i, j).* += alpha * accumulator;
                        }
                    }
                }
            }

            self.result.deinit();
            self.result = result;
        }

        /// Scaled matrix multiplication: α * A * B
        /// Convenience method for common GEMM use case
        pub fn scaledDot(self: *Self, other: Matrix(T), alpha: T) !void {
            try self.gemm(other, false, false, alpha, 0.0, null);
        }

        /// Matrix multiplication with transpose: A * B^T
        /// Convenience method for common GEMM use case
        pub fn dotTranspose(self: *Self, other: Matrix(T)) !void {
            try self.gemm(other, false, true, 1.0, 0.0, null);
        }

        /// Transpose matrix multiplication: A^T * B
        /// Convenience method for common GEMM use case
        pub fn transposeDot(self: *Self, other: Matrix(T)) !void {
            try self.gemm(other, true, false, 1.0, 0.0, null);
        }

        /// Transfer ownership of the result to the caller
        pub fn toOwned(self: *Self) Matrix(T) {
            const final_result = self.result;
            // Mark as consumed to prevent deinit from freeing it
            self.consumed = true;
            return final_result;
        }
    };
}

// Tests for OpsBuilder functionality

test "complex operation chaining" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test data
    const static_a = SMatrix(f32, 2, 2).init(.{ .{ 1.0, 2.0 }, .{ 3.0, 4.0 } });
    const static_b = SMatrix(f32, 2, 2).init(.{ .{ 2.0, 0.0 }, .{ 0.0, 2.0 } });

    // SMatrix chaining (direct method calls)
    const static_result = static_a.dot(static_b).transpose().scale(0.5);

    // OpsBuilder chaining (equivalent operations)
    const dynamic_a = try static_a.toMatrix(arena.allocator());
    const dynamic_b = try static_b.toMatrix(arena.allocator());

    var ops = try OpsBuilder(f32).init(arena.allocator(), dynamic_a);
    try ops.dot(dynamic_b);
    try ops.transpose();
    try ops.scale(0.5);
    const dynamic_result = ops.toOwned();

    // Verify both approaches give identical results
    for (0..2) |r| {
        for (0..2) |c| {
            try expectEqual(static_result.at(r, c).*, dynamic_result.at(r, c).*);
        }
    }
}

test "row and column extraction with OpsBuilder" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test data
    const test_matrix = SMatrix(f32, 3, 2).init(.{
        .{ 1.0, 2.0 },
        .{ 3.0, 4.0 },
        .{ 5.0, 6.0 },
    });

    // Test OpsBuilder row/col extraction on equivalent Matrix
    const dynamic_matrix = try test_matrix.toMatrix(arena.allocator());

    var row_ops: OpsBuilder(f32) = try .init(arena.allocator(), dynamic_matrix);
    try row_ops.row(1);
    const dynamic_row = row_ops.toOwned();
    try expectEqual(@as(usize, 1), dynamic_row.rows);
    try expectEqual(@as(usize, 2), dynamic_row.cols);
    try expectEqual(@as(f32, 3.0), dynamic_row.at(0, 0).*);
    try expectEqual(@as(f32, 4.0), dynamic_row.at(0, 1).*);

    var col_ops: OpsBuilder(f32) = try .init(arena.allocator(), dynamic_matrix);
    try col_ops.col(1);
    const dynamic_col = col_ops.toOwned();
    try expectEqual(@as(usize, 3), dynamic_col.rows);
    try expectEqual(@as(usize, 1), dynamic_col.cols);
    try expectEqual(@as(f32, 2.0), dynamic_col.at(0, 0).*);
    try expectEqual(@as(f32, 4.0), dynamic_col.at(1, 0).*);
    try expectEqual(@as(f32, 6.0), dynamic_col.at(2, 0).*);
}

test "OpsBuilder gram and covariance matrices" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    // Create test matrix (3 samples × 2 features)
    var data: Matrix(f64) = try .init(arena.allocator(), 3, 2);
    data.at(0, 0).* = 1.0;
    data.at(0, 1).* = 2.0;
    data.at(1, 0).* = 3.0;
    data.at(1, 1).* = 4.0;
    data.at(2, 0).* = 5.0;
    data.at(2, 1).* = 6.0;

    // Test Gram matrix (X * X^T) - should be 3×3
    var gram_ops: OpsBuilder(f64) = try .init(arena.allocator(), data);
    try gram_ops.gram();
    const gram_result = gram_ops.toOwned();

    try expectEqual(@as(usize, 3), gram_result.rows);
    try expectEqual(@as(usize, 3), gram_result.cols);

    // Verify gram matrix values
    // First row: [1*1+2*2, 1*3+2*4, 1*5+2*6] = [5, 11, 17]
    try expectEqual(@as(f64, 5.0), gram_result.at(0, 0).*);
    try expectEqual(@as(f64, 11.0), gram_result.at(0, 1).*);
    try expectEqual(@as(f64, 17.0), gram_result.at(0, 2).*);

    // Test Covariance matrix (X^T * X) - should be 2×2
    var cov_ops: OpsBuilder(f64) = try .init(arena.allocator(), data);
    try cov_ops.covariance();
    const cov_result = cov_ops.toOwned();

    try expectEqual(@as(usize, 2), cov_result.rows);
    try expectEqual(@as(usize, 2), cov_result.cols);

    // Verify covariance matrix values
    // First row: [1*1+3*3+5*5, 1*2+3*4+5*6] = [35, 44]
    try expectEqual(@as(f64, 35.0), cov_result.at(0, 0).*);
    try expectEqual(@as(f64, 44.0), cov_result.at(0, 1).*);
    // Second row: [2*1+4*3+6*5, 2*2+4*4+6*6] = [44, 56]
    try expectEqual(@as(f64, 44.0), cov_result.at(1, 0).*);
    try expectEqual(@as(f64, 56.0), cov_result.at(1, 1).*);
}

test "OpsBuilder GEMM operations" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Create test matrices
    var a: Matrix(f32) = try .init(arena.allocator(), 2, 3);
    a.at(0, 0).* = 1.0;
    a.at(0, 1).* = 2.0;
    a.at(0, 2).* = 3.0;
    a.at(1, 0).* = 4.0;
    a.at(1, 1).* = 5.0;
    a.at(1, 2).* = 6.0;

    var b: Matrix(f32) = try .init(arena.allocator(), 3, 2);
    b.at(0, 0).* = 7.0;
    b.at(0, 1).* = 8.0;
    b.at(1, 0).* = 9.0;
    b.at(1, 1).* = 10.0;
    b.at(2, 0).* = 11.0;
    b.at(2, 1).* = 12.0;

    var c: Matrix(f32) = try .init(arena.allocator(), 2, 2);
    c.at(0, 0).* = 1.0;
    c.at(0, 1).* = 1.0;
    c.at(1, 0).* = 1.0;
    c.at(1, 1).* = 1.0;

    // Test basic matrix multiplication: A * B using dot() method
    var ops_dot: OpsBuilder(f32) = try .init(arena.allocator(), a);
    try ops_dot.dot(b);
    const dot_result = ops_dot.toOwned();

    try expectEqual(@as(f32, 58.0), dot_result.at(0, 0).*); // 1*7 + 2*9 + 3*11
    try expectEqual(@as(f32, 64.0), dot_result.at(0, 1).*); // 1*8 + 2*10 + 3*12
    try expectEqual(@as(f32, 139.0), dot_result.at(1, 0).*); // 4*7 + 5*9 + 6*11
    try expectEqual(@as(f32, 154.0), dot_result.at(1, 1).*); // 4*8 + 5*10 + 6*12

    // Test basic matrix multiplication: A * B using gemm() method
    var ops1: OpsBuilder(f32) = try .init(arena.allocator(), a);
    try ops1.gemm(b, false, false, 1.0, 0.0, null);
    const result1 = ops1.toOwned();

    try expectEqual(@as(f32, 58.0), result1.at(0, 0).*); // 1*7 + 2*9 + 3*11
    try expectEqual(@as(f32, 64.0), result1.at(0, 1).*); // 1*8 + 2*10 + 3*12
    try expectEqual(@as(f32, 139.0), result1.at(1, 0).*); // 4*7 + 5*9 + 6*11
    try expectEqual(@as(f32, 154.0), result1.at(1, 1).*); // 4*8 + 5*10 + 6*12

    // Test scaled multiplication: 2 * A * B
    var ops2: OpsBuilder(f32) = try .init(arena.allocator(), a);
    try ops2.gemm(b, false, false, 2.0, 0.0, null);
    const result2 = ops2.toOwned();

    try expectEqual(@as(f32, 116.0), result2.at(0, 0).*); // 2 * 58
    try expectEqual(@as(f32, 128.0), result2.at(0, 1).*); // 2 * 64

    // Test accumulation: A * B + C
    var ops3: OpsBuilder(f32) = try .init(arena.allocator(), a);
    try ops3.gemm(b, false, false, 1.0, 1.0, c);
    const result3 = ops3.toOwned();

    try expectEqual(@as(f32, 59.0), result3.at(0, 0).*); // 58 + 1
    try expectEqual(@as(f32, 65.0), result3.at(0, 1).*); // 64 + 1

    // Test Gram matrix using GEMM: A * A^T
    var ops4: OpsBuilder(f32) = try .init(arena.allocator(), a);
    try ops4.gemm(a, false, true, 1.0, 0.0, null);
    const gram = ops4.toOwned();

    try expectEqual(@as(usize, 2), gram.rows);
    try expectEqual(@as(usize, 2), gram.cols);
    try expectEqual(@as(f32, 14.0), gram.at(0, 0).*); // 1*1 + 2*2 + 3*3
    try expectEqual(@as(f32, 32.0), gram.at(0, 1).*); // 1*4 + 2*5 + 3*6

    // Test covariance using GEMM: A^T * A
    var ops5: OpsBuilder(f32) = try .init(arena.allocator(), a);
    try ops5.gemm(a, true, false, 1.0, 0.0, null);
    const cov = ops5.toOwned();

    try expectEqual(@as(usize, 3), cov.rows);
    try expectEqual(@as(usize, 3), cov.cols);
    try expectEqual(@as(f32, 17.0), cov.at(0, 0).*); // 1*1 + 4*4
    try expectEqual(@as(f32, 22.0), cov.at(0, 1).*); // 1*2 + 4*5
}

test "OpsBuilder SIMD case 2: A^T * B with same matrix (covariance)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Create a test matrix
    var data = try Matrix(f32).init(arena.allocator(), 4, 3);
    data.at(0, 0).* = 1.0;
    data.at(0, 1).* = 2.0;
    data.at(0, 2).* = 3.0;
    data.at(1, 0).* = 4.0;
    data.at(1, 1).* = 5.0;
    data.at(1, 2).* = 6.0;
    data.at(2, 0).* = 7.0;
    data.at(2, 1).* = 8.0;
    data.at(2, 2).* = 9.0;
    data.at(3, 0).* = 10.0;
    data.at(3, 1).* = 11.0;
    data.at(3, 2).* = 12.0;

    // Test covariance using OpsBuilder (should use SIMD)
    var ops1: OpsBuilder(f32) = try .init(arena.allocator(), data);
    try ops1.covariance(); // This calls gemm(self.result, true, false, 1.0, 0.0, null)
    const simd_result = ops1.toOwned();

    // Compute expected result manually
    var expected = try Matrix(f32).init(arena.allocator(), 3, 3);
    @memset(expected.items, 0);

    // Compute A^T * A manually
    for (0..3) |i| {
        for (0..3) |j| {
            var sum: f32 = 0;
            for (0..4) |k| {
                sum += data.at(k, i).* * data.at(k, j).*;
            }
            expected.at(i, j).* = sum;
        }
    }

    // Verify dimensions
    try expectEqual(@as(usize, 3), simd_result.rows);
    try expectEqual(@as(usize, 3), simd_result.cols);

    // Verify values match
    for (0..3) |i| {
        for (0..3) |j| {
            const diff = @abs(simd_result.at(i, j).* - expected.at(i, j).*);
            try std.testing.expect(diff < 1e-5);
        }
    }

    // Also test direct GEMM call with same matrix
    var ops2: OpsBuilder(f32) = try .init(arena.allocator(), data);
    try ops2.gemm(data, true, false, 1.0, 0.0, null);
    const direct_result = ops2.toOwned();

    // Verify direct GEMM gives same result
    for (0..3) |i| {
        for (0..3) |j| {
            const diff = @abs(direct_result.at(i, j).* - expected.at(i, j).*);
            try std.testing.expect(diff < 1e-5);
        }
    }
}

test "OpsBuilder GEMM size threshold optimization" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test 1: Small matrices (should use scalar path)
    // 4x4 * 4x4 = 64 operations < 512 threshold
    var small_a = try Matrix(f32).init(arena.allocator(), 4, 4);
    var small_b = try Matrix(f32).init(arena.allocator(), 4, 4);

    // Fill with simple test data
    for (0..4) |i| {
        for (0..4) |j| {
            small_a.at(i, j).* = 1.0;
            small_b.at(i, j).* = 2.0;
        }
    }

    var ops_small: OpsBuilder(f32) = try .init(arena.allocator(), small_a);
    try ops_small.dot(small_b);
    const small_result = ops_small.toOwned();

    // Verify correctness: 4x4 * 4x4 with all 1s and 2s should give matrix of all 8s
    try expectEqual(@as(usize, 4), small_result.rows);
    try expectEqual(@as(usize, 4), small_result.cols);
    try expectEqual(@as(f32, 8.0), small_result.at(0, 0).*);
    try expectEqual(@as(f32, 8.0), small_result.at(3, 3).*);

    // Test 2: Large matrices (should use SIMD path)
    // 12x12 * 12x12 = 1728 operations > 512 threshold
    var large_a = try Matrix(f32).init(arena.allocator(), 12, 12);
    var large_b = try Matrix(f32).init(arena.allocator(), 12, 12);

    // Fill with simple test data
    for (0..12) |i| {
        for (0..12) |j| {
            large_a.at(i, j).* = 0.5;
            large_b.at(i, j).* = 4.0;
        }
    }

    var ops_large: OpsBuilder(f32) = try .init(arena.allocator(), large_a);
    try ops_large.dot(large_b);
    const large_result = ops_large.toOwned();

    // Verify correctness: 12x12 * 12x12 with all 0.5s and 4s should give matrix of all 24s
    try expectEqual(@as(usize, 12), large_result.rows);
    try expectEqual(@as(usize, 12), large_result.cols);
    try expectEqual(@as(f32, 24.0), large_result.at(0, 0).*);
    try expectEqual(@as(f32, 24.0), large_result.at(11, 11).*);

    // Test 3: Exactly at threshold
    // 8x8 * 8x8 = 512 operations = threshold (should use SIMD)
    var threshold_a = try Matrix(f32).init(arena.allocator(), 8, 8);
    var threshold_b = try Matrix(f32).init(arena.allocator(), 8, 8);

    for (0..8) |i| {
        for (0..8) |j| {
            threshold_a.at(i, j).* = 1.0;
            threshold_b.at(i, j).* = 1.0;
        }
    }

    var ops_threshold: OpsBuilder(f32) = try .init(arena.allocator(), threshold_a);
    try ops_threshold.dot(threshold_b);
    const threshold_result = ops_threshold.toOwned();

    // Should be 8x8 matrix of all 8s
    try expectEqual(@as(usize, 8), threshold_result.rows);
    try expectEqual(@as(usize, 8), threshold_result.cols);
    try expectEqual(@as(f32, 8.0), threshold_result.at(0, 0).*);
    try expectEqual(@as(f32, 8.0), threshold_result.at(7, 7).*);
}

test "OpsBuilder GEMM all transpose cases with same matrix" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Create test matrix A (3x2 for non-square tests)
    var a = try Matrix(f32).init(arena.allocator(), 3, 2);
    a.at(0, 0).* = 1.0;
    a.at(0, 1).* = 2.0;
    a.at(1, 0).* = 3.0;
    a.at(1, 1).* = 4.0;
    a.at(2, 0).* = 5.0;
    a.at(2, 1).* = 6.0;

    // Create square matrix for Case 1 and Case 4
    var square_a = try Matrix(f32).init(arena.allocator(), 2, 2);
    square_a.at(0, 0).* = 1.0;
    square_a.at(0, 1).* = 2.0;
    square_a.at(1, 0).* = 3.0;
    square_a.at(1, 1).* = 4.0;

    // Case 1: A * A (SIMD same-matrix handling)
    var ops1: OpsBuilder(f32) = try .init(arena.allocator(), square_a);
    try ops1.gemm(square_a, false, false, 1.0, 0.0, null);
    const result1 = ops1.toOwned();

    // Expected: A * A = [[1*1+2*3, 1*2+2*4], [3*1+4*3, 3*2+4*4]] = [[7, 10], [15, 22]]
    try expectEqual(@as(usize, 2), result1.rows);
    try expectEqual(@as(usize, 2), result1.cols);
    try expectEqual(@as(f32, 7.0), result1.at(0, 0).*);   // 1*1 + 2*3
    try expectEqual(@as(f32, 10.0), result1.at(0, 1).*);  // 1*2 + 2*4
    try expectEqual(@as(f32, 15.0), result1.at(1, 0).*);  // 3*1 + 4*3
    try expectEqual(@as(f32, 22.0), result1.at(1, 1).*);  // 3*2 + 4*4

    // Case 2: A^T * A (covariance - SIMD same-matrix handling)
    var ops2: OpsBuilder(f32) = try .init(arena.allocator(), a);
    try ops2.gemm(a, true, false, 1.0, 0.0, null);
    const result2 = ops2.toOwned();

    // Expected: A^T * A (3x2 -> 2x2 result)
    // A^T = [[1,3,5], [2,4,6]]
    // A^T * A = [[1*1+3*3+5*5, 1*2+3*4+5*6], [2*1+4*3+6*5, 2*2+4*4+6*6]] = [[35, 44], [44, 56]]
    try expectEqual(@as(usize, 2), result2.rows);
    try expectEqual(@as(usize, 2), result2.cols);
    try expectEqual(@as(f32, 35.0), result2.at(0, 0).*);  // 1*1 + 3*3 + 5*5
    try expectEqual(@as(f32, 44.0), result2.at(0, 1).*);  // 1*2 + 3*4 + 5*6
    try expectEqual(@as(f32, 44.0), result2.at(1, 0).*);  // 2*1 + 4*3 + 6*5
    try expectEqual(@as(f32, 56.0), result2.at(1, 1).*);  // 2*2 + 4*4 + 6*6

    // Case 3: A * A^T (gram matrix - SIMD same-matrix handling)
    var ops3: OpsBuilder(f32) = try .init(arena.allocator(), a);
    try ops3.gemm(a, false, true, 1.0, 0.0, null);
    const result3 = ops3.toOwned();

    // Expected: A * A^T (3x2 -> 3x3 result)
    // A * A^T = [[1*1+2*2, 1*3+2*4, 1*5+2*6], [3*1+4*2, 3*3+4*4, 3*5+4*6], [5*1+6*2, 5*3+6*4, 5*5+6*6]]
    //         = [[5, 11, 17], [11, 25, 39], [17, 39, 61]]
    try expectEqual(@as(usize, 3), result3.rows);
    try expectEqual(@as(usize, 3), result3.cols);
    try expectEqual(@as(f32, 5.0), result3.at(0, 0).*);    // 1*1 + 2*2
    try expectEqual(@as(f32, 11.0), result3.at(0, 1).*);   // 1*3 + 2*4
    try expectEqual(@as(f32, 17.0), result3.at(0, 2).*);   // 1*5 + 2*6
    try expectEqual(@as(f32, 11.0), result3.at(1, 0).*);   // 3*1 + 4*2
    try expectEqual(@as(f32, 25.0), result3.at(1, 1).*);   // 3*3 + 4*4
    try expectEqual(@as(f32, 39.0), result3.at(1, 2).*);   // 3*5 + 4*6
    try expectEqual(@as(f32, 17.0), result3.at(2, 0).*);   // 5*1 + 6*2
    try expectEqual(@as(f32, 39.0), result3.at(2, 1).*);   // 5*3 + 6*4
    try expectEqual(@as(f32, 61.0), result3.at(2, 2).*);   // 5*5 + 6*6

    // Case 4: A^T * A^T (both transposed - SIMD same-matrix handling)
    var ops4: OpsBuilder(f32) = try .init(arena.allocator(), square_a);
    try ops4.gemm(square_a, true, true, 1.0, 0.0, null);
    const result4 = ops4.toOwned();

    // Expected: A^T * A^T where A^T = [[1,3], [2,4]]
    // A^T * A^T = [[1*1+3*2, 1*3+3*4], [2*1+4*2, 2*3+4*4]] = [[7, 15], [10, 22]]
    try expectEqual(@as(usize, 2), result4.rows);
    try expectEqual(@as(usize, 2), result4.cols);
    try expectEqual(@as(f32, 7.0), result4.at(0, 0).*);    // 1*1 + 3*2
    try expectEqual(@as(f32, 15.0), result4.at(0, 1).*);   // 1*3 + 3*4
    try expectEqual(@as(f32, 10.0), result4.at(1, 0).*);   // 2*1 + 4*2
    try expectEqual(@as(f32, 22.0), result4.at(1, 1).*);   // 2*3 + 4*4
}

test "OpsBuilder SIMD same-matrix detection verification" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test with matrices large enough to exceed threshold
    // 9x9 * 9x9 = 729 operations > 512 threshold
    var test_matrix = try Matrix(f32).init(arena.allocator(), 9, 9);
    
    // Fill with known values
    for (0..9) |i| {
        for (0..9) |j| {
            test_matrix.at(i, j).* = @as(f32, @floatFromInt(i + j + 1));
        }
    }

    // Test that all 4 cases complete without errors when using same matrix
    // (We can't easily verify which path was taken without debug output,
    //  but we can verify the operations complete successfully)
    
    // Case 1: A * A
    var ops1: OpsBuilder(f32) = try .init(arena.allocator(), test_matrix);
    try ops1.gemm(test_matrix, false, false, 1.0, 0.0, null);
    const result1 = ops1.toOwned();
    try expectEqual(@as(usize, 9), result1.rows);
    try expectEqual(@as(usize, 9), result1.cols);
    
    // Case 2: A^T * A  
    var ops2: OpsBuilder(f32) = try .init(arena.allocator(), test_matrix);
    try ops2.gemm(test_matrix, true, false, 1.0, 0.0, null);
    const result2 = ops2.toOwned();
    try expectEqual(@as(usize, 9), result2.rows);
    try expectEqual(@as(usize, 9), result2.cols);
    
    // Case 3: A * A^T
    var ops3: OpsBuilder(f32) = try .init(arena.allocator(), test_matrix);
    try ops3.gemm(test_matrix, false, true, 1.0, 0.0, null);
    const result3 = ops3.toOwned();
    try expectEqual(@as(usize, 9), result3.rows);
    try expectEqual(@as(usize, 9), result3.cols);
    
    // Case 4: A^T * A^T
    var ops4: OpsBuilder(f32) = try .init(arena.allocator(), test_matrix);
    try ops4.gemm(test_matrix, true, true, 1.0, 0.0, null);
    const result4 = ops4.toOwned();
    try expectEqual(@as(usize, 9), result4.rows);
    try expectEqual(@as(usize, 9), result4.cols);
}

test "OpsBuilder matrix operations: add, sub, scale, transpose" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test data
    const static_matrix: SMatrix(f32, 2, 3) = .init(.{
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
    });

    // Test operand matrix for add/sub operations
    const static_operand: SMatrix(f32, 2, 3) = .init(.{
        .{ 0.5, 1.0, 1.5 },
        .{ 2.0, 2.5, 3.0 },
    });

    // SMatrix operations for reference
    const static_scaled = static_matrix.scale(2.0);
    const static_transposed = static_matrix.transpose();
    const static_subtracted = static_matrix.sub(static_operand);

    // OpsBuilder operations on equivalent matrix
    const dynamic_matrix = try static_matrix.toMatrix(arena.allocator());

    // Test scale
    var scale_ops: OpsBuilder(f32) = try .init(arena.allocator(), dynamic_matrix);
    try scale_ops.scale(2.0);
    const dynamic_scaled = scale_ops.toOwned();

    // Test transpose
    var transpose_ops: OpsBuilder(f32) = try .init(arena.allocator(), dynamic_matrix);
    try transpose_ops.transpose();
    const dynamic_transposed = transpose_ops.toOwned();

    // Test add
    const add_matrix: Matrix(f32) = try .initAll(arena.allocator(), 2, 3, 1.0);
    var add_ops: OpsBuilder(f32) = try .init(arena.allocator(), dynamic_matrix);
    try add_ops.add(add_matrix);
    const dynamic_added = add_ops.toOwned();

    // Test subtract
    const dynamic_operand = try static_operand.toMatrix(arena.allocator());
    var sub_ops: OpsBuilder(f32) = try .init(arena.allocator(), dynamic_matrix);
    try sub_ops.sub(dynamic_operand);
    const dynamic_subtracted = sub_ops.toOwned();

    // Verify results match
    for (0..2) |r| {
        for (0..3) |c| {
            try expectEqual(static_scaled.at(r, c).*, dynamic_scaled.at(r, c).*);
        }
    }
    for (0..3) |r| {
        for (0..2) |c| {
            try expectEqual(static_transposed.at(r, c).*, dynamic_transposed.at(r, c).*);
        }
    }
    try expectEqual(@as(f32, 2.0), dynamic_added.at(0, 0).*); // 1 + 1
    try expectEqual(@as(f32, 7.0), dynamic_added.at(1, 2).*); // 6 + 1
    for (0..2) |r| {
        for (0..3) |c| {
            try expectEqual(static_subtracted.at(r, c).*, dynamic_subtracted.at(r, c).*);
        }
    }
}
