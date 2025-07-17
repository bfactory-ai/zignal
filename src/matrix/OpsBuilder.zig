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

        /// Computes the determinant of the matrix using analytical formulas for small matrices
        /// and LU decomposition for larger matrices
        pub fn determinant(self: *Self) !T {
            assert(self.result.rows == self.result.cols);

            const n = self.result.rows;

            // Use analytical formulas for small matrices (more efficient)
            if (n <= 3) {
                return switch (n) {
                    1 => self.result.at(0, 0).*,
                    2 => self.result.at(0, 0).* * self.result.at(1, 1).* -
                        self.result.at(0, 1).* * self.result.at(1, 0).*,
                    3 => self.result.at(0, 0).* * self.result.at(1, 1).* * self.result.at(2, 2).* +
                        self.result.at(0, 1).* * self.result.at(1, 2).* * self.result.at(2, 0).* +
                        self.result.at(0, 2).* * self.result.at(1, 0).* * self.result.at(2, 1).* -
                        self.result.at(0, 2).* * self.result.at(1, 1).* * self.result.at(2, 0).* -
                        self.result.at(0, 1).* * self.result.at(1, 0).* * self.result.at(2, 2).* -
                        self.result.at(0, 0).* * self.result.at(1, 2).* * self.result.at(2, 1).*,
                    else => unreachable,
                };
            }

            // Use LU decomposition for larger matrices
            var lu_result = try self.lu();
            defer lu_result.deinit();

            // det(A) = sign * product of diagonal elements of U
            var det = lu_result.sign;
            for (0..n) |i| {
                det *= lu_result.u.at(i, i).*;
            }

            return det;
        }


        /// Inverts the matrix using analytical formulas for small matrices (≤3x3)
        /// and Gauss-Jordan elimination for larger matrices
        pub fn inverse(self: *Self) !void {
            assert(self.result.rows == self.result.cols);

            const n = self.result.rows;

            // Use analytical formulas for small matrices (more efficient)
            if (n <= 3) {
                const det = try self.determinant();
                if (@abs(det) < std.math.floatEps(T)) {
                    return error.SingularMatrix;
                }

                var inv = try Matrix(T).init(self.allocator, n, n);

                switch (n) {
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
                    else => unreachable,
                }

                self.result.deinit();
                self.result = inv;
            } else {
                // Use Gauss-Jordan elimination for larger matrices
                try self.inverseGaussJordan();
            }
        }

        /// Inverts the matrix using Gauss-Jordan elimination with partial pivoting
        /// This is a general method that works for any size square matrix
        fn inverseGaussJordan(self: *Self) !void {
            const n = self.result.rows;

            // Create augmented matrix [A | I]
            var augmented = try Matrix(T).init(self.allocator, n, 2 * n);
            defer augmented.deinit();

            // Copy original matrix to left half and identity to right half
            for (0..n) |i| {
                for (0..n) |j| {
                    augmented.at(i, j).* = self.result.at(i, j).*;
                    augmented.at(i, n + j).* = if (i == j) 1.0 else 0.0;
                }
            }

            // Perform Gauss-Jordan elimination
            for (0..n) |pivot_col| {
                // Find pivot (partial pivoting for numerical stability)
                var max_row = pivot_col;
                var max_val = @abs(augmented.at(pivot_col, pivot_col).*);

                for (pivot_col + 1..n) |row_idx| {
                    const val = @abs(augmented.at(row_idx, pivot_col).*);
                    if (val > max_val) {
                        max_val = val;
                        max_row = row_idx;
                    }
                }

                // Check for singular matrix
                if (max_val < std.math.floatEps(T) * 10) {
                    return error.SingularMatrix;
                }

                // Swap rows if needed
                if (max_row != pivot_col) {
                    for (0..2 * n) |j| {
                        const temp = augmented.at(pivot_col, j).*;
                        augmented.at(pivot_col, j).* = augmented.at(max_row, j).*;
                        augmented.at(max_row, j).* = temp;
                    }
                }

                // Scale pivot row
                const pivot = augmented.at(pivot_col, pivot_col).*;
                for (0..2 * n) |j| {
                    augmented.at(pivot_col, j).* /= pivot;
                }

                // Eliminate column in all other rows
                for (0..n) |row_idx| {
                    if (row_idx != pivot_col) {
                        const factor = augmented.at(row_idx, pivot_col).*;
                        for (0..2 * n) |j| {
                            augmented.at(row_idx, j).* -= factor * augmented.at(pivot_col, j).*;
                        }
                    }
                }
            }

            // Extract inverse from right half of augmented matrix
            var inv = try Matrix(T).init(self.allocator, n, n);
            for (0..n) |i| {
                for (0..n) |j| {
                    inv.at(i, j).* = augmented.at(i, n + j).*;
                }
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
                const simd_threshold = 512; // Use SIMD for larger matrices (>512 operations)

                // Use SIMD only for larger matrices where the benefit outweighs allocation overhead
                if (vec_len > 1 and total_ops >= simd_threshold) {
                    // Enable SIMD for all 4 transpose combinations
                    const VecType = @Vector(vec_len, T);

                    if (!trans_a and !trans_b) {
                        // Case 1: A * B - transpose B for cache-friendly row-major access

                        // Handle special case when A and B are the same matrix (for A * A)
                        if (self.result.rows == other.rows and
                            self.result.cols == other.cols and
                            std.mem.eql(T, self.result.items, other.items))
                        {
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
                            std.mem.eql(T, self.result.items, other.items))
                        {
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
                            std.mem.eql(T, self.result.items, other.items))
                        {
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

        /// Result of LU decomposition
        pub const LUResult = struct {
            l: Matrix(T),      // Lower triangular matrix
            u: Matrix(T),      // Upper triangular matrix
            p: []usize,        // Permutation vector
            sign: T,           // Determinant sign (+1 or -1)
            allocator: std.mem.Allocator,

            pub fn deinit(self: *@This()) void {
                self.l.deinit();
                self.u.deinit();
                self.allocator.free(self.p);
            }
        };

        /// Compute LU decomposition with partial pivoting
        /// Returns L, U matrices and permutation vector such that PA = LU
        pub fn lu(self: *Self) !LUResult {
            const n = self.result.rows;
            assert(n == self.result.cols); // Must be square

            // Create working copy
            var work = try Matrix(T).init(self.allocator, n, n);
            defer work.deinit();
            @memcpy(work.items, self.result.items);

            // Initialize L as identity, U as zero
            var l = try Matrix(T).init(self.allocator, n, n);
            errdefer l.deinit();
            var u = try Matrix(T).init(self.allocator, n, n);
            errdefer u.deinit();

            // Initialize permutation vector
            var p = try self.allocator.alloc(usize, n);
            errdefer self.allocator.free(p);
            for (0..n) |i| {
                p[i] = i;
            }

            // Initialize matrices
            @memset(l.items, 0);
            @memset(u.items, 0);
            for (0..n) |i| {
                l.at(i, i).* = 1.0; // L starts as identity
            }

            var sign: T = 1.0;

            // Perform LU decomposition with partial pivoting
            for (0..n) |pivot_col| {
                // Find pivot
                var max_row = pivot_col;
                var max_val = @abs(work.at(pivot_col, pivot_col).*);

                for (pivot_col + 1..n) |row_idx| {
                    const val = @abs(work.at(row_idx, pivot_col).*);
                    if (val > max_val) {
                        max_val = val;
                        max_row = row_idx;
                    }
                }

                // Check for zero pivot (singular matrix)
                if (max_val < std.math.floatEps(T) * 10) {
                    // Continue with decomposition even if singular
                    // User can check if U has zeros on diagonal
                }

                // Swap rows if needed
                if (max_row != pivot_col) {
                    sign = -sign;
                    // Swap in permutation vector
                    const temp_p = p[pivot_col];
                    p[pivot_col] = p[max_row];
                    p[max_row] = temp_p;

                    // Swap rows in work matrix
                    for (0..n) |j| {
                        const temp = work.at(pivot_col, j).*;
                        work.at(pivot_col, j).* = work.at(max_row, j).*;
                        work.at(max_row, j).* = temp;
                    }

                    // Swap rows in L (only the part already computed)
                    for (0..pivot_col) |j| {
                        const temp = l.at(pivot_col, j).*;
                        l.at(pivot_col, j).* = l.at(max_row, j).*;
                        l.at(max_row, j).* = temp;
                    }
                }

                // Copy pivot row to U
                for (pivot_col..n) |j| {
                    u.at(pivot_col, j).* = work.at(pivot_col, j).*;
                }

                // Compute L column and eliminate
                for (pivot_col + 1..n) |row_idx| {
                    if (@abs(work.at(pivot_col, pivot_col).*) > std.math.floatEps(T)) {
                        const factor = work.at(row_idx, pivot_col).* / work.at(pivot_col, pivot_col).*;
                        l.at(row_idx, pivot_col).* = factor;

                        for (pivot_col + 1..n) |col_idx| {
                            work.at(row_idx, col_idx).* -= factor * work.at(pivot_col, col_idx).*;
                        }
                    }
                }
            }

            return LUResult{
                .l = l,
                .u = u,
                .p = p,
                .sign = sign,
                .allocator = self.allocator,
            };
        }

        pub const QRResult = struct {
            q: Matrix(T),      // Orthogonal matrix (Q^T Q = I)
            r: Matrix(T),      // Upper triangular matrix
            allocator: std.mem.Allocator,

            pub fn deinit(self: *@This()) void {
                self.q.deinit();
                self.r.deinit();
            }
        };

        /// Compute QR decomposition using Modified Gram-Schmidt algorithm
        /// Returns Q, R matrices such that A = QR where Q is orthogonal and R is upper triangular
        pub fn qr(self: *Self) !QRResult {
            const m = self.result.rows;
            const n = self.result.cols;

            // Initialize Q and R matrices
            var q = try Matrix(T).init(self.allocator, m, n);
            errdefer q.deinit();
            var r = try Matrix(T).init(self.allocator, n, n);
            errdefer r.deinit();

            // Copy A to Q (will be modified in-place)
            @memcpy(q.items, self.result.items);

            // Initialize R as zero
            @memset(r.items, 0);

            // Modified Gram-Schmidt algorithm
            for (0..n) |j| {
                // Compute R[j,j] = ||Q[:,j]||
                var norm_sq: T = 0;
                for (0..m) |i| {
                    const val = q.at(i, j).*;
                    norm_sq += val * val;
                }
                r.at(j, j).* = @sqrt(norm_sq);

                // Check for linear dependence
                if (r.at(j, j).* < std.math.floatEps(T) * 100) {
                    return error.LinearlyDependent;
                }

                // Normalize Q[:,j]
                const inv_norm = 1.0 / r.at(j, j).*;
                for (0..m) |i| {
                    q.at(i, j).* *= inv_norm;
                }

                // Orthogonalize remaining columns
                for (j + 1..n) |k| {
                    // Compute R[j,k] = Q[:,j]^T * Q[:,k]
                    var dot_product: T = 0;
                    for (0..m) |i| {
                        dot_product += q.at(i, j).* * q.at(i, k).*;
                    }
                    r.at(j, k).* = dot_product;

                    // Q[:,k] = Q[:,k] - R[j,k] * Q[:,j]
                    for (0..m) |i| {
                        q.at(i, k).* -= r.at(j, k).* * q.at(i, j).*;
                    }
                }
            }

            return QRResult{
                .q = q,
                .r = r,
                .allocator = self.allocator,
            };
        }

        /// Apply a function to all matrix elements with optional arguments
        pub fn apply(self: *Self, comptime func: anytype, args: anytype) !void {
            for (0..self.result.items.len) |i| {
                self.result.items[i] = @call(.auto, func, .{self.result.items[i]} ++ args);
            }
        }

        /// Sum of all elements
        pub fn sum(self: *Self) T {
            var total: T = 0;
            for (0..self.result.items.len) |i| {
                total += self.result.items[i];
            }
            return total;
        }

        /// Mean (average) of all elements
        pub fn mean(self: *Self) T {
            return self.sum() / @as(T, @floatFromInt(self.result.items.len));
        }

        /// Variance: E[(X - μ)²]
        pub fn variance(self: *Self) T {
            const mu = self.mean();
            var sum_sq_diff: T = 0;
            for (0..self.result.items.len) |i| {
                const diff = self.result.items[i] - mu;
                sum_sq_diff += diff * diff;
            }
            return sum_sq_diff / @as(T, @floatFromInt(self.result.items.len));
        }

        /// Standard deviation: sqrt(variance)
        pub fn stdDev(self: *Self) T {
            return @sqrt(self.variance());
        }

        /// Minimum element
        pub fn min(self: *Self) T {
            var min_val = self.result.items[0];
            for (1..self.result.items.len) |i| {
                if (self.result.items[i] < min_val) {
                    min_val = self.result.items[i];
                }
            }
            return min_val;
        }

        /// Maximum element
        pub fn max(self: *Self) T {
            var max_val = self.result.items[0];
            for (1..self.result.items.len) |i| {
                if (self.result.items[i] > max_val) {
                    max_val = self.result.items[i];
                }
            }
            return max_val;
        }

        /// Frobenius norm: sqrt(sum of squares of all elements)
        pub fn frobeniusNorm(self: *Self) T {
            // Using the SMatrix approach: sqrt(times(self).sum())
            var sum_squares: T = 0;
            for (0..self.result.items.len) |i| {
                sum_squares += self.result.items[i] * self.result.items[i];
            }
            return @sqrt(sum_squares);
        }

        /// L1 norm (nuclear norm): sum of absolute values of all elements
        pub fn l1Norm(self: *Self) T {
            var sum_abs: T = 0;
            for (0..self.result.items.len) |i| {
                sum_abs += @abs(self.result.items[i]);
            }
            return sum_abs;
        }

        /// Max norm (L-infinity): maximum absolute value
        pub fn maxNorm(self: *Self) T {
            var max_abs: T = 0;
            for (0..self.result.items.len) |i| {
                const abs_val = @abs(self.result.items[i]);
                if (abs_val > max_abs) {
                    max_abs = abs_val;
                }
            }
            return max_abs;
        }

        /// Trace: sum of diagonal elements (square matrices only)
        pub fn trace(self: *Self) T {
            assert(self.result.rows == self.result.cols);
            var sum_diag: T = 0;
            for (0..self.result.rows) |i| {
                sum_diag += self.result.at(i, i).*;
            }
            return sum_diag;
        }

        /// Add scalar to all elements
        pub fn offset(self: *Self, value: T) !void {
            for (0..self.result.items.len) |i| {
                self.result.items[i] += value;
            }
        }

        /// Raise all elements to power n (convenience method)
        pub fn pow(self: *Self, n: T) !void {
            const powN = struct {
                fn f(x: T, exponent: T) T {
                    return std.math.pow(T, x, exponent);
                }
            }.f;
            try self.apply(powN, .{n});
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
    try expectEqual(@as(f32, 7.0), result1.at(0, 0).*); // 1*1 + 2*3
    try expectEqual(@as(f32, 10.0), result1.at(0, 1).*); // 1*2 + 2*4
    try expectEqual(@as(f32, 15.0), result1.at(1, 0).*); // 3*1 + 4*3
    try expectEqual(@as(f32, 22.0), result1.at(1, 1).*); // 3*2 + 4*4

    // Case 2: A^T * A (covariance - SIMD same-matrix handling)
    var ops2: OpsBuilder(f32) = try .init(arena.allocator(), a);
    try ops2.gemm(a, true, false, 1.0, 0.0, null);
    const result2 = ops2.toOwned();

    // Expected: A^T * A (3x2 -> 2x2 result)
    // A^T = [[1,3,5], [2,4,6]]
    // A^T * A = [[1*1+3*3+5*5, 1*2+3*4+5*6], [2*1+4*3+6*5, 2*2+4*4+6*6]] = [[35, 44], [44, 56]]
    try expectEqual(@as(usize, 2), result2.rows);
    try expectEqual(@as(usize, 2), result2.cols);
    try expectEqual(@as(f32, 35.0), result2.at(0, 0).*); // 1*1 + 3*3 + 5*5
    try expectEqual(@as(f32, 44.0), result2.at(0, 1).*); // 1*2 + 3*4 + 5*6
    try expectEqual(@as(f32, 44.0), result2.at(1, 0).*); // 2*1 + 4*3 + 6*5
    try expectEqual(@as(f32, 56.0), result2.at(1, 1).*); // 2*2 + 4*4 + 6*6

    // Case 3: A * A^T (gram matrix - SIMD same-matrix handling)
    var ops3: OpsBuilder(f32) = try .init(arena.allocator(), a);
    try ops3.gemm(a, false, true, 1.0, 0.0, null);
    const result3 = ops3.toOwned();

    // Expected: A * A^T (3x2 -> 3x3 result)
    // A * A^T = [[1*1+2*2, 1*3+2*4, 1*5+2*6], [3*1+4*2, 3*3+4*4, 3*5+4*6], [5*1+6*2, 5*3+6*4, 5*5+6*6]]
    //         = [[5, 11, 17], [11, 25, 39], [17, 39, 61]]
    try expectEqual(@as(usize, 3), result3.rows);
    try expectEqual(@as(usize, 3), result3.cols);
    try expectEqual(@as(f32, 5.0), result3.at(0, 0).*); // 1*1 + 2*2
    try expectEqual(@as(f32, 11.0), result3.at(0, 1).*); // 1*3 + 2*4
    try expectEqual(@as(f32, 17.0), result3.at(0, 2).*); // 1*5 + 2*6
    try expectEqual(@as(f32, 11.0), result3.at(1, 0).*); // 3*1 + 4*2
    try expectEqual(@as(f32, 25.0), result3.at(1, 1).*); // 3*3 + 4*4
    try expectEqual(@as(f32, 39.0), result3.at(1, 2).*); // 3*5 + 4*6
    try expectEqual(@as(f32, 17.0), result3.at(2, 0).*); // 5*1 + 6*2
    try expectEqual(@as(f32, 39.0), result3.at(2, 1).*); // 5*3 + 6*4
    try expectEqual(@as(f32, 61.0), result3.at(2, 2).*); // 5*5 + 6*6

    // Case 4: A^T * A^T (both transposed - SIMD same-matrix handling)
    var ops4: OpsBuilder(f32) = try .init(arena.allocator(), square_a);
    try ops4.gemm(square_a, true, true, 1.0, 0.0, null);
    const result4 = ops4.toOwned();

    // Expected: A^T * A^T where A^T = [[1,3], [2,4]]
    // A^T * A^T = [[1*1+3*2, 1*3+3*4], [2*1+4*2, 2*3+4*4]] = [[7, 15], [10, 22]]
    try expectEqual(@as(usize, 2), result4.rows);
    try expectEqual(@as(usize, 2), result4.cols);
    try expectEqual(@as(f32, 7.0), result4.at(0, 0).*); // 1*1 + 3*2
    try expectEqual(@as(f32, 15.0), result4.at(0, 1).*); // 1*3 + 3*4
    try expectEqual(@as(f32, 10.0), result4.at(1, 0).*); // 2*1 + 4*2
    try expectEqual(@as(f32, 22.0), result4.at(1, 1).*); // 2*3 + 4*4
}

test "OpsBuilder SIMD 9x9 matrix with known values" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Create simple 9x9 matrix with predictable values (forces SIMD: 729 ops > 512)
    var test_matrix = try Matrix(f32).init(arena.allocator(), 9, 9);

    // Fill with simple pattern: A[i,j] = i + 1 (row number)
    for (0..9) |i| {
        for (0..9) |j| {
            test_matrix.at(i, j).* = @as(f32, @floatFromInt(i + 1));
        }
    }

    // Test Case 1: A * A (should use SIMD same-matrix optimization)
    var ops1: OpsBuilder(f32) = try .init(arena.allocator(), test_matrix);
    try ops1.gemm(test_matrix, false, false, 1.0, 0.0, null);
    const result1 = ops1.toOwned();

    // Verify Case 1: A * A (uses SIMD same-matrix optimization)
    try expectEqual(@as(usize, 9), result1.rows);
    try expectEqual(@as(usize, 9), result1.cols);
    try expectEqual(@as(f32, 45.0), result1.at(0, 0).*); // Row 0 * Col 0
    try expectEqual(@as(f32, 90.0), result1.at(1, 0).*); // Row 1 * Col 0
    try expectEqual(@as(f32, 405.0), result1.at(8, 8).*); // Row 8 * Col 8

    // Test Case 2: A^T * A (covariance)
    var ops2: OpsBuilder(f32) = try .init(arena.allocator(), test_matrix);
    try ops2.gemm(test_matrix, true, false, 1.0, 0.0, null);
    const result2 = ops2.toOwned();

    // Verify Case 2: A^T * A (covariance, uses SIMD same-matrix optimization)
    try expectEqual(@as(usize, 9), result2.rows);
    try expectEqual(@as(usize, 9), result2.cols);
    try expectEqual(@as(f32, 285.0), result2.at(0, 0).*); // Sum of squares: 1²+2²+...+9²
    try expectEqual(@as(f32, 285.0), result2.at(8, 8).*); // Same for all diagonal elements

    // Test Case 3: A * A^T (gram matrix)
    var ops3: OpsBuilder(f32) = try .init(arena.allocator(), test_matrix);
    try ops3.gemm(test_matrix, false, true, 1.0, 0.0, null);
    const result3 = ops3.toOwned();

    // Verify Case 3: A * A^T (gram matrix, uses SIMD same-matrix optimization)
    try expectEqual(@as(usize, 9), result3.rows);
    try expectEqual(@as(usize, 9), result3.cols);
    try expectEqual(@as(f32, 9.0), result3.at(0, 0).*); // 1² * 9 elements
    try expectEqual(@as(f32, 36.0), result3.at(1, 1).*); // 2² * 9 elements
    try expectEqual(@as(f32, 729.0), result3.at(8, 8).*); // 9² * 9 elements

    // Test Case 4: A^T * A^T
    var ops4: OpsBuilder(f32) = try .init(arena.allocator(), test_matrix);
    try ops4.gemm(test_matrix, true, true, 1.0, 0.0, null);
    const result4 = ops4.toOwned();

    // Verify Case 4: A^T * A^T (uses SIMD same-matrix optimization)
    try expectEqual(@as(usize, 9), result4.rows);
    try expectEqual(@as(usize, 9), result4.cols);
    try expectEqual(@as(f32, 45.0), result4.at(0, 0).*); // Corners same as case 1
    try expectEqual(@as(f32, 405.0), result4.at(0, 8).*);
    try expectEqual(@as(f32, 45.0), result4.at(8, 0).*);
    try expectEqual(@as(f32, 405.0), result4.at(8, 8).*);
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

test "OpsBuilder matrix inverse - small matrices" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test 2x2 matrix inverse
    var mat2 = try Matrix(f64).init(arena.allocator(), 2, 2);
    mat2.at(0, 0).* = 4.0;
    mat2.at(0, 1).* = 7.0;
    mat2.at(1, 0).* = 2.0;
    mat2.at(1, 1).* = 6.0;

    var ops2: OpsBuilder(f64) = try .init(arena.allocator(), mat2);
    try ops2.inverse();
    const inv2 = ops2.toOwned();

    // Verify A * A^(-1) = I
    var check2: OpsBuilder(f64) = try .init(arena.allocator(), mat2);
    try check2.dot(inv2);
    const identity2 = check2.toOwned();

    const eps = 1e-10;
    try std.testing.expect(@abs(identity2.at(0, 0).* - 1.0) < eps);
    try std.testing.expect(@abs(identity2.at(0, 1).* - 0.0) < eps);
    try std.testing.expect(@abs(identity2.at(1, 0).* - 0.0) < eps);
    try std.testing.expect(@abs(identity2.at(1, 1).* - 1.0) < eps);

    // Test 3x3 matrix inverse
    var mat3 = try Matrix(f64).init(arena.allocator(), 3, 3);
    mat3.at(0, 0).* = 1.0;
    mat3.at(0, 1).* = 2.0;
    mat3.at(0, 2).* = 3.0;
    mat3.at(1, 0).* = 0.0;
    mat3.at(1, 1).* = 1.0;
    mat3.at(1, 2).* = 4.0;
    mat3.at(2, 0).* = 5.0;
    mat3.at(2, 1).* = 6.0;
    mat3.at(2, 2).* = 0.0;

    var ops3: OpsBuilder(f64) = try .init(arena.allocator(), mat3);
    try ops3.inverse();
    const inv3 = ops3.toOwned();

    // Verify A * A^(-1) = I
    var check3: OpsBuilder(f64) = try .init(arena.allocator(), mat3);
    try check3.dot(inv3);
    const identity3 = check3.toOwned();

    for (0..3) |i| {
        for (0..3) |j| {
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try std.testing.expect(@abs(identity3.at(i, j).* - expected) < eps);
        }
    }
}

test "OpsBuilder matrix inverse - large matrices using Gauss-Jordan" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test 4x4 matrix inverse
    var mat4 = try Matrix(f64).init(arena.allocator(), 4, 4);
    // Create a well-conditioned matrix
    mat4.at(0, 0).* = 5.0;
    mat4.at(0, 1).* = 1.0;
    mat4.at(0, 2).* = 0.0;
    mat4.at(0, 3).* = 2.0;
    mat4.at(1, 0).* = 1.0;
    mat4.at(1, 1).* = 4.0;
    mat4.at(1, 2).* = 1.0;
    mat4.at(1, 3).* = 1.0;
    mat4.at(2, 0).* = 0.0;
    mat4.at(2, 1).* = 1.0;
    mat4.at(2, 2).* = 3.0;
    mat4.at(2, 3).* = 0.0;
    mat4.at(3, 0).* = 2.0;
    mat4.at(3, 1).* = 1.0;
    mat4.at(3, 2).* = 0.0;
    mat4.at(3, 3).* = 4.0;

    var ops4: OpsBuilder(f64) = try .init(arena.allocator(), mat4);
    try ops4.inverse();
    const inv4 = ops4.toOwned();

    // Verify A * A^(-1) = I
    var check4: OpsBuilder(f64) = try .init(arena.allocator(), mat4);
    try check4.dot(inv4);
    const identity4 = check4.toOwned();

    const eps = 1e-10;
    for (0..4) |i| {
        for (0..4) |j| {
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try std.testing.expect(@abs(identity4.at(i, j).* - expected) < eps);
        }
    }

    // Test 5x5 matrix inverse
    var mat5 = try Matrix(f64).init(arena.allocator(), 5, 5);
    // Create a diagonally dominant matrix (well-conditioned)
    for (0..5) |i| {
        for (0..5) |j| {
            if (i == j) {
                mat5.at(i, j).* = 10.0;
            } else {
                mat5.at(i, j).* = @as(f64, @floatFromInt(i + j)) * 0.5;
            }
        }
    }

    var ops5: OpsBuilder(f64) = try .init(arena.allocator(), mat5);
    try ops5.inverse();
    const inv5 = ops5.toOwned();

    // Verify A * A^(-1) = I
    var check5: OpsBuilder(f64) = try .init(arena.allocator(), mat5);
    try check5.dot(inv5);
    const identity5 = check5.toOwned();

    for (0..5) |i| {
        for (0..5) |j| {
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try std.testing.expect(@abs(identity5.at(i, j).* - expected) < eps);
        }
    }
}

test "OpsBuilder matrix inverse - singular matrix error" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test singular 2x2 matrix
    var sing2 = try Matrix(f64).init(arena.allocator(), 2, 2);
    sing2.at(0, 0).* = 1.0;
    sing2.at(0, 1).* = 2.0;
    sing2.at(1, 0).* = 2.0;
    sing2.at(1, 1).* = 4.0; // Second row is multiple of first

    var ops_sing2: OpsBuilder(f64) = try .init(arena.allocator(), sing2);
    try std.testing.expectError(error.SingularMatrix, ops_sing2.inverse());

    // Test singular 4x4 matrix (uses Gauss-Jordan)
    var sing4 = try Matrix(f64).init(arena.allocator(), 4, 4);
    // Make third row a linear combination of first two
    sing4.at(0, 0).* = 1.0;
    sing4.at(0, 1).* = 2.0;
    sing4.at(0, 2).* = 3.0;
    sing4.at(0, 3).* = 4.0;
    sing4.at(1, 0).* = 5.0;
    sing4.at(1, 1).* = 6.0;
    sing4.at(1, 2).* = 7.0;
    sing4.at(1, 3).* = 8.0;
    sing4.at(2, 0).* = 6.0; // row2 = row0 + row1
    sing4.at(2, 1).* = 8.0;
    sing4.at(2, 2).* = 10.0;
    sing4.at(2, 3).* = 12.0;
    sing4.at(3, 0).* = 9.0;
    sing4.at(3, 1).* = 10.0;
    sing4.at(3, 2).* = 11.0;
    sing4.at(3, 3).* = 12.0;

    var ops_sing4: OpsBuilder(f64) = try .init(arena.allocator(), sing4);
    defer ops_sing4.deinit();
    try std.testing.expectError(error.SingularMatrix, ops_sing4.inverse());
}

test "OpsBuilder determinant - small matrices" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test 1x1 matrix
    var mat1 = try Matrix(f64).init(arena.allocator(), 1, 1);
    mat1.at(0, 0).* = 5.0;
    var ops1 = try OpsBuilder(f64).init(arena.allocator(), mat1);
    defer ops1.deinit();
    try expectEqual(@as(f64, 5.0), try ops1.determinant());

    // Test 2x2 matrix
    var mat2 = try Matrix(f64).init(arena.allocator(), 2, 2);
    mat2.at(0, 0).* = 4.0;
    mat2.at(0, 1).* = 7.0;
    mat2.at(1, 0).* = 2.0;
    mat2.at(1, 1).* = 6.0;
    var ops2 = try OpsBuilder(f64).init(arena.allocator(), mat2);
    defer ops2.deinit();
    // det = 4*6 - 7*2 = 24 - 14 = 10
    try expectEqual(@as(f64, 10.0), try ops2.determinant());

    // Test 3x3 matrix
    var mat3 = try Matrix(f64).init(arena.allocator(), 3, 3);
    mat3.at(0, 0).* = 1.0;
    mat3.at(0, 1).* = 2.0;
    mat3.at(0, 2).* = 3.0;
    mat3.at(1, 0).* = 0.0;
    mat3.at(1, 1).* = 1.0;
    mat3.at(1, 2).* = 4.0;
    mat3.at(2, 0).* = 5.0;
    mat3.at(2, 1).* = 6.0;
    mat3.at(2, 2).* = 0.0;
    var ops3 = try OpsBuilder(f64).init(arena.allocator(), mat3);
    defer ops3.deinit();
    // det = 1*(1*0 - 4*6) - 2*(0*0 - 4*5) + 3*(0*6 - 1*5)
    //     = 1*(-24) - 2*(-20) + 3*(-5)
    //     = -24 + 40 - 15 = 1
    try expectEqual(@as(f64, 1.0), try ops3.determinant());
}

test "OpsBuilder determinant - large matrices using LU" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test 4x4 matrix
    var mat4 = try Matrix(f64).init(arena.allocator(), 4, 4);
    mat4.at(0, 0).* = 5.0;
    mat4.at(0, 1).* = 1.0;
    mat4.at(0, 2).* = 0.0;
    mat4.at(0, 3).* = 2.0;
    mat4.at(1, 0).* = 1.0;
    mat4.at(1, 1).* = 4.0;
    mat4.at(1, 2).* = 1.0;
    mat4.at(1, 3).* = 1.0;
    mat4.at(2, 0).* = 0.0;
    mat4.at(2, 1).* = 1.0;
    mat4.at(2, 2).* = 3.0;
    mat4.at(2, 3).* = 0.0;
    mat4.at(3, 0).* = 2.0;
    mat4.at(3, 1).* = 1.0;
    mat4.at(3, 2).* = 0.0;
    mat4.at(3, 3).* = 4.0;

    var ops4 = try OpsBuilder(f64).init(arena.allocator(), mat4);
    defer ops4.deinit();
    const det4 = try ops4.determinant();

    // This matrix should have a non-zero determinant
    try std.testing.expect(@abs(det4) > 1e-10);

    // Test singular matrix (determinant should be 0)
    var sing = try Matrix(f64).init(arena.allocator(), 4, 4);
    sing.at(0, 0).* = 1.0;
    sing.at(0, 1).* = 2.0;
    sing.at(0, 2).* = 3.0;
    sing.at(0, 3).* = 4.0;
    sing.at(1, 0).* = 2.0;
    sing.at(1, 1).* = 4.0;
    sing.at(1, 2).* = 6.0;
    sing.at(1, 3).* = 8.0; // Row 2 = 2 * Row 1
    sing.at(2, 0).* = 3.0;
    sing.at(2, 1).* = 5.0;
    sing.at(2, 2).* = 7.0;
    sing.at(2, 3).* = 9.0;
    sing.at(3, 0).* = 4.0;
    sing.at(3, 1).* = 6.0;
    sing.at(3, 2).* = 8.0;
    sing.at(3, 3).* = 10.0;

    var ops_sing = try OpsBuilder(f64).init(arena.allocator(), sing);
    defer ops_sing.deinit();
    const det_sing = try ops_sing.determinant();

    // Singular matrix should have determinant 0
    try std.testing.expect(@abs(det_sing) < 1e-10);

    // Test 5x5 identity matrix (determinant should be 1)
    var identity5 = try Matrix(f64).init(arena.allocator(), 5, 5);
    @memset(identity5.items, 0);
    for (0..5) |i| {
        identity5.at(i, i).* = 1.0;
    }

    var ops_id = try OpsBuilder(f64).init(arena.allocator(), identity5);
    defer ops_id.deinit();
    const det_id = try ops_id.determinant();

    try expectEqual(@as(f64, 1.0), det_id);
}

test "OpsBuilder apply method" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test matrix
    var mat = try Matrix(f64).init(arena.allocator(), 2, 3);
    mat.at(0, 0).* = 1.0;
    mat.at(0, 1).* = 4.0;
    mat.at(0, 2).* = 9.0;
    mat.at(1, 0).* = 16.0;
    mat.at(1, 1).* = 25.0;
    mat.at(1, 2).* = 36.0;

    // Test apply with no arguments (sqrt)
    var ops1 = try OpsBuilder(f64).init(arena.allocator(), mat);
    try ops1.apply(std.math.sqrt, .{});
    var result1 = ops1.toOwned();
    defer result1.deinit();

    try expectEqual(@as(f64, 1.0), result1.at(0, 0).*);
    try expectEqual(@as(f64, 2.0), result1.at(0, 1).*);
    try expectEqual(@as(f64, 3.0), result1.at(0, 2).*);
    try expectEqual(@as(f64, 4.0), result1.at(1, 0).*);
    try expectEqual(@as(f64, 5.0), result1.at(1, 1).*);
    try expectEqual(@as(f64, 6.0), result1.at(1, 2).*);

    // Test apply with arguments (pow)
    const pow2 = struct {
        fn f(x: f64, n: f64) f64 {
            return std.math.pow(f64, x, n);
        }
    }.f;
    var ops2 = try OpsBuilder(f64).init(arena.allocator(), result1);
    try ops2.apply(pow2, .{@as(f64, 2.0)});
    var result2 = ops2.toOwned();
    defer result2.deinit();

    try expectEqual(@as(f64, 1.0), result2.at(0, 0).*);
    try expectEqual(@as(f64, 4.0), result2.at(0, 1).*);
    try expectEqual(@as(f64, 9.0), result2.at(0, 2).*);
    try expectEqual(@as(f64, 16.0), result2.at(1, 0).*);
    try expectEqual(@as(f64, 25.0), result2.at(1, 1).*);
    try expectEqual(@as(f64, 36.0), result2.at(1, 2).*);

    // Test custom function
    const reciprocal = struct {
        fn f(x: f64) f64 {
            return 1.0 / x;
        }
    }.f;

    var ops3 = try OpsBuilder(f64).init(arena.allocator(), result1);
    try ops3.apply(reciprocal, .{});
    var result3 = ops3.toOwned();
    defer result3.deinit();

    try expectEqual(@as(f64, 1.0), result3.at(0, 0).*);
    try expectEqual(@as(f64, 0.5), result3.at(0, 1).*);
    try expectEqual(@as(f64, 1.0/3.0), result3.at(0, 2).*);
}

test "OpsBuilder statistical operations" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test matrix with known values
    var mat = try Matrix(f64).init(arena.allocator(), 2, 3);
    mat.at(0, 0).* = 1.0;
    mat.at(0, 1).* = 2.0;
    mat.at(0, 2).* = 3.0;
    mat.at(1, 0).* = 4.0;
    mat.at(1, 1).* = 5.0;
    mat.at(1, 2).* = 6.0;

    var ops = try OpsBuilder(f64).init(arena.allocator(), mat);
    defer ops.deinit();

    // Test sum: 1+2+3+4+5+6 = 21
    try expectEqual(@as(f64, 21.0), ops.sum());

    // Test mean: 21/6 = 3.5
    try expectEqual(@as(f64, 3.5), ops.mean());

    // Test min and max
    try expectEqual(@as(f64, 1.0), ops.min());
    try expectEqual(@as(f64, 6.0), ops.max());

    // Test variance: E[(X - 3.5)²]
    // Values: (1-3.5)² + (2-3.5)² + (3-3.5)² + (4-3.5)² + (5-3.5)² + (6-3.5)²
    //       = 6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25 = 17.5
    // Variance = 17.5 / 6 = 2.916666...
    const variance = ops.variance();
    try std.testing.expect(@abs(variance - 2.916666666666667) < 1e-10);

    // Test standard deviation: sqrt(variance)
    const std_dev = ops.stdDev();
    try std.testing.expect(@abs(std_dev - @sqrt(2.916666666666667)) < 1e-10);
}

test "OpsBuilder norms" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test matrix
    var mat = try Matrix(f64).init(arena.allocator(), 2, 2);
    mat.at(0, 0).* = 3.0;
    mat.at(0, 1).* = -4.0;
    mat.at(1, 0).* = -1.0;
    mat.at(1, 1).* = 2.0;

    var ops = try OpsBuilder(f64).init(arena.allocator(), mat);
    defer ops.deinit();

    // Test Frobenius norm: sqrt(9 + 16 + 1 + 4) = sqrt(30)
    const frob = ops.frobeniusNorm();
    try std.testing.expect(@abs(frob - @sqrt(30.0)) < 1e-10);

    // Test L1 norm: 3 + 4 + 1 + 2 = 10
    try expectEqual(@as(f64, 10.0), ops.l1Norm());

    // Test max norm: max(3, 4, 1, 2) = 4
    try expectEqual(@as(f64, 4.0), ops.maxNorm());

    // Test trace (diagonal sum): 3 + 2 = 5
    try expectEqual(@as(f64, 5.0), ops.trace());
}

test "OpsBuilder offset and pow" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test matrix
    var mat = try Matrix(f64).init(arena.allocator(), 2, 2);
    mat.at(0, 0).* = 1.0;
    mat.at(0, 1).* = 2.0;
    mat.at(1, 0).* = 3.0;
    mat.at(1, 1).* = 4.0;

    // Test offset
    var ops1 = try OpsBuilder(f64).init(arena.allocator(), mat);
    try ops1.offset(5.0);
    var result1 = ops1.toOwned();
    defer result1.deinit();

    try expectEqual(@as(f64, 6.0), result1.at(0, 0).*);
    try expectEqual(@as(f64, 7.0), result1.at(0, 1).*);
    try expectEqual(@as(f64, 8.0), result1.at(1, 0).*);
    try expectEqual(@as(f64, 9.0), result1.at(1, 1).*);

    // Test pow
    var ops2 = try OpsBuilder(f64).init(arena.allocator(), mat);
    try ops2.pow(2.0);
    var result2 = ops2.toOwned();
    defer result2.deinit();

    try expectEqual(@as(f64, 1.0), result2.at(0, 0).*);
    try expectEqual(@as(f64, 4.0), result2.at(0, 1).*);
    try expectEqual(@as(f64, 9.0), result2.at(1, 0).*);
    try expectEqual(@as(f64, 16.0), result2.at(1, 1).*);
}

test "OpsBuilder LU decomposition" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test 3x3 matrix
    var mat = try Matrix(f64).init(arena.allocator(), 3, 3);
    mat.at(0, 0).* = 2.0;
    mat.at(0, 1).* = 1.0;
    mat.at(0, 2).* = 1.0;
    mat.at(1, 0).* = 4.0;
    mat.at(1, 1).* = 3.0;
    mat.at(1, 2).* = 3.0;
    mat.at(2, 0).* = 8.0;
    mat.at(2, 1).* = 7.0;
    mat.at(2, 2).* = 9.0;

    var ops = try OpsBuilder(f64).init(arena.allocator(), mat);
    defer ops.deinit();

    // Compute LU decomposition
    var lu_result = try ops.lu();
    defer lu_result.deinit();

    // Verify dimensions
    try expectEqual(@as(usize, 3), lu_result.l.rows);
    try expectEqual(@as(usize, 3), lu_result.l.cols);
    try expectEqual(@as(usize, 3), lu_result.u.rows);
    try expectEqual(@as(usize, 3), lu_result.u.cols);

    // Verify L is lower triangular with 1s on diagonal
    try expectEqual(@as(f64, 1.0), lu_result.l.at(0, 0).*);
    try expectEqual(@as(f64, 1.0), lu_result.l.at(1, 1).*);
    try expectEqual(@as(f64, 1.0), lu_result.l.at(2, 2).*);
    try expectEqual(@as(f64, 0.0), lu_result.l.at(0, 1).*);
    try expectEqual(@as(f64, 0.0), lu_result.l.at(0, 2).*);
    try expectEqual(@as(f64, 0.0), lu_result.l.at(1, 2).*);

    // Verify U is upper triangular
    try expectEqual(@as(f64, 0.0), lu_result.u.at(1, 0).*);
    try expectEqual(@as(f64, 0.0), lu_result.u.at(2, 0).*);
    try expectEqual(@as(f64, 0.0), lu_result.u.at(2, 1).*);

    // Reconstruct PA = LU
    var pa = try Matrix(f64).init(arena.allocator(), 3, 3);
    defer pa.deinit();

    // Apply permutation: PA[i,j] = A[p[i],j]
    for (0..3) |i| {
        for (0..3) |j| {
            pa.at(i, j).* = mat.at(lu_result.p[i], j).*;
        }
    }

    // Compute L * U
    var lu_product = try Matrix(f64).init(arena.allocator(), 3, 3);
    defer lu_product.deinit();
    @memset(lu_product.items, 0);

    for (0..3) |i| {
        for (0..3) |j| {
            for (0..3) |k| {
                lu_product.at(i, j).* += lu_result.l.at(i, k).* * lu_result.u.at(k, j).*;
            }
        }
    }

    // Verify PA = LU (within numerical tolerance)
    const eps = 1e-10;
    for (0..3) |i| {
        for (0..3) |j| {
            const diff = @abs(pa.at(i, j).* - lu_product.at(i, j).*);
            try std.testing.expect(diff < eps);
        }
    }
}

test "OpsBuilder QR decomposition" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test 3x3 matrix
    var mat = try Matrix(f64).init(arena.allocator(), 3, 3);
    mat.at(0, 0).* = 12.0;
    mat.at(0, 1).* = -51.0;
    mat.at(0, 2).* = 4.0;
    mat.at(1, 0).* = 6.0;
    mat.at(1, 1).* = 167.0;
    mat.at(1, 2).* = -68.0;
    mat.at(2, 0).* = -4.0;
    mat.at(2, 1).* = 24.0;
    mat.at(2, 2).* = -41.0;

    var ops = try OpsBuilder(f64).init(arena.allocator(), mat);
    defer ops.deinit();

    // Compute QR decomposition
    var qr_result = try ops.qr();
    defer qr_result.deinit();

    // Verify dimensions
    try expectEqual(@as(usize, 3), qr_result.q.rows);
    try expectEqual(@as(usize, 3), qr_result.q.cols);
    try expectEqual(@as(usize, 3), qr_result.r.rows);
    try expectEqual(@as(usize, 3), qr_result.r.cols);

    // Verify R is upper triangular
    try expectEqual(@as(f64, 0.0), qr_result.r.at(1, 0).*);
    try expectEqual(@as(f64, 0.0), qr_result.r.at(2, 0).*);
    try expectEqual(@as(f64, 0.0), qr_result.r.at(2, 1).*);

    // Verify Q is orthogonal: Q^T * Q should be identity
    var qtq = try Matrix(f64).init(arena.allocator(), 3, 3);
    defer qtq.deinit();
    @memset(qtq.items, 0);

    for (0..3) |i| {
        for (0..3) |j| {
            for (0..3) |k| {
                qtq.at(i, j).* += qr_result.q.at(k, i).* * qr_result.q.at(k, j).*;
            }
        }
    }

    // Check that Q^T * Q is approximately identity
    const eps = 1e-10;
    for (0..3) |i| {
        for (0..3) |j| {
            const expected: f64 = if (i == j) 1.0 else 0.0;
            const diff = @abs(qtq.at(i, j).* - expected);
            try std.testing.expect(diff < eps);
        }
    }

    // Verify A = Q * R
    var qr_product = try Matrix(f64).init(arena.allocator(), 3, 3);
    defer qr_product.deinit();
    @memset(qr_product.items, 0);

    for (0..3) |i| {
        for (0..3) |j| {
            for (0..3) |k| {
                qr_product.at(i, j).* += qr_result.q.at(i, k).* * qr_result.r.at(k, j).*;
            }
        }
    }

    // Verify A = QR (within numerical tolerance)
    for (0..3) |i| {
        for (0..3) |j| {
            const diff = @abs(mat.at(i, j).* - qr_product.at(i, j).*);
            try std.testing.expect(diff < eps);
        }
    }

    // Test rectangular matrix (4x3) with linearly independent columns
    var rect_mat = try Matrix(f64).init(arena.allocator(), 4, 3);
    rect_mat.at(0, 0).* = 1.0;
    rect_mat.at(0, 1).* = 0.0;
    rect_mat.at(0, 2).* = 0.0;
    rect_mat.at(1, 0).* = 1.0;
    rect_mat.at(1, 1).* = 1.0;
    rect_mat.at(1, 2).* = 0.0;
    rect_mat.at(2, 0).* = 1.0;
    rect_mat.at(2, 1).* = 1.0;
    rect_mat.at(2, 2).* = 1.0;
    rect_mat.at(3, 0).* = 1.0;
    rect_mat.at(3, 1).* = 1.0;
    rect_mat.at(3, 2).* = 2.0;

    var rect_ops = try OpsBuilder(f64).init(arena.allocator(), rect_mat);
    defer rect_ops.deinit();

    var rect_qr = try rect_ops.qr();
    defer rect_qr.deinit();

    // Verify dimensions for rectangular matrix
    try expectEqual(@as(usize, 4), rect_qr.q.rows);
    try expectEqual(@as(usize, 3), rect_qr.q.cols);
    try expectEqual(@as(usize, 3), rect_qr.r.rows);
    try expectEqual(@as(usize, 3), rect_qr.r.cols);

    // Verify A = Q * R for rectangular matrix
    var rect_product = try Matrix(f64).init(arena.allocator(), 4, 3);
    defer rect_product.deinit();
    @memset(rect_product.items, 0);

    for (0..4) |i| {
        for (0..3) |j| {
            for (0..3) |k| {
                rect_product.at(i, j).* += rect_qr.q.at(i, k).* * rect_qr.r.at(k, j).*;
            }
        }
    }

    for (0..4) |i| {
        for (0..3) |j| {
            const diff = @abs(rect_mat.at(i, j).* - rect_product.at(i, j).*);
            try std.testing.expect(diff < eps);
        }
    }
}
