const std = @import("std");

const SMatrix = @import("matrix.zig").SMatrix;

/// Controls the size and computation of the left singular vectors matrix (U) in SVD.
/// This allows optimization of memory usage and computation time based on your needs.
pub const SvdMode = enum {
    /// Skip computation of U matrix entirely. Use when only singular values are needed.
    no_u,
    /// Compute only the first n columns of U (economy/thin SVD). Results in U being m×n.
    /// More memory efficient when m >> n.
    skinny_u,
    /// Compute the full m×m U matrix. Use when all left singular vectors are needed.
    full_u,
};

/// Internal state machine for the SVD algorithm's iterative process.
/// Based on the classical Golub-Reinsch algorithm.
const SvdState = enum {
    /// Test if the superdiagonal element can be set to zero (decoupling test)
    test_splitting,
    /// Cancel the superdiagonal element using Givens rotations
    cancellation,
    /// Check if the algorithm has converged for the current singular value
    test_convergence,
    /// Final convergence check and sign correction
    convergence_check,
};

/// Configuration options for SVD computation.
/// Allows fine-grained control over which matrices are computed.
pub const SvdOptions = struct {
    /// Whether to compute the left singular vectors (U matrix).
    /// Set to false when only singular values are needed.
    with_u: bool = true,
    /// Whether to compute the right singular vectors (V matrix).
    /// Set to false when only U and singular values are needed.
    with_v: bool = false,
    /// Controls the size of the U matrix when with_u is true.
    /// Ignored when with_u is false.
    mode: SvdMode = .full_u,
};

/// Result type for SVD decomposition: A = U × Σ × V^T
/// where A is the input matrix, U contains left singular vectors,
/// Σ is a diagonal matrix of singular values (stored as a vector),
/// and V contains right singular vectors.
///
/// The dimensions of matrices depend on the options used:
/// - U: m×m (full_u), m×n (skinny_u), or empty (no_u)
/// - s: n×1 vector of singular values in descending order
/// - V: n×n matrix (or empty if with_v=false)
pub fn SvdResult(
    comptime T: type,
    comptime rows: usize,
    comptime cols: usize,
    comptime options: SvdOptions,
) type {
    return struct {
        /// Left singular vectors matrix. Each column is a left singular vector.
        /// Dimensions: m×m (full_u) or m×n (skinny_u)
        u: SMatrix(T, rows, if (options.mode == .skinny_u) cols else rows),
        /// Singular values in descending order as a column vector.
        /// These are the diagonal elements of the Σ matrix.
        s: SMatrix(T, cols, 1),
        /// Right singular vectors matrix. Each column is a right singular vector.
        /// The matrix is orthogonal: V^T × V = I
        v: SMatrix(T, cols, cols),
        /// Convergence status: 0 if successful, k if failed at k-th singular value.
        /// Non-zero values indicate the iterative algorithm failed to converge.
        converged: usize,
    };
}

/// Performs singular value decompostion. Code adapted from dlib's svd4, which is, in turn,
/// translated to 'C' from the original Algol code in "Hanbook for Automatic Computation, vol. II,
/// Linear Algebra", Springer-Verlag.  Note that this published algorithm is considered to be
/// the best and numerically stable approach to computing the real-valued svd and is referenced
/// repeatedly in ieee journal papers, etc where the svd is used.
///
/// This is almost an exact translation from the original, except that an iteration counter
/// is added to prevent stalls.  This corresponds to similar changes in other translations.
/// Returns an error code = 0, if no errors and 'k' if a failure to converge at the 'kth'
/// singular value.
///
/// USAGE: given the singular value decomposition a = u * diagm(q) * trans(v) for an m*n
/// matrix a with m >= n.  After the svd call, u is an m x m matrix which is columnwise
/// orthogonal. q will be an n element vector consisting of singular values and v an n x n
/// orthogonal matrix. eps and tol are tolerance constants.  Suitable values are eps=1e-16
/// and tol=(1e-300)/eps if T == double.
///
/// If options.mode == .no_u, then u won't be computed and similarly if options.with_v == false
/// then v won't be computed.  If options.mode == .skinny_u then u will be m x n instead of m x m.
pub fn svd(
    comptime T: type,
    comptime rows: usize,
    comptime cols: usize,
    a: SMatrix(T, rows, cols),
    comptime options: SvdOptions,
) SvdResult(T, rows, cols, options) {
    comptime std.debug.assert(rows >= cols);
    var eps: T = std.math.floatEps(T);
    const tol: T = std.math.floatMin(T) / eps;
    const m = rows;
    const n = cols;
    var u = comptime if (options.mode == .full_u) SMatrix(T, m, m){} else SMatrix(T, m, n){};
    var v: SMatrix(T, n, n) = .{};
    var q: SMatrix(T, n, 1) = .{};
    var e: SMatrix(T, n, 1) = .{};
    var l: usize = 0;
    var retval: usize = 0;
    var c: T = undefined;
    var f: T = undefined;
    var g: T = undefined;
    var h: T = undefined;
    var s: T = undefined;
    var x: T = undefined;
    var y: T = undefined;
    var z: T = undefined;

    // Copy a to u.
    for (0..m) |i| {
        for (0..n) |j| {
            u.items[i][j] = a.items[i][j];
        }
    }

    // Householder's reduction to bidiagonal form.
    g = 0;
    x = 0;
    for (0..n) |i| {
        e.items[i][0] = g;
        s = 0;
        l = i + 1;

        for (i..m) |j| {
            s += u.items[j][i] * u.items[j][i];
        }

        if (s < tol) {
            g = 0;
        } else {
            f = u.items[i][i];
            g = if (f < 0) @sqrt(s) else -@sqrt(s);
            h = f * g - s;
            u.items[i][i] = f - g;

            for (l..n) |j| {
                s = 0;
                for (i..m) |k| {
                    s += u.items[k][i] * u.items[k][j];
                }
                f = s / h;

                for (i..m) |k| {
                    u.items[k][j] += f * u.items[k][i];
                }
            }
        }

        q.items[i][0] = g;
        s = 0;

        for (l..n) |j| {
            s += u.items[i][j] * u.items[i][j];
        }

        if (s < tol) {
            g = 0;
        } else {
            f = u.items[i][i + 1];
            g = if (f < 0) @sqrt(s) else -@sqrt(s);
            h = f * g - s;
            u.items[i][i + 1] = f - g;

            for (l..n) |j| {
                e.items[j][0] = u.items[i][j] / h;
            }

            for (l..m) |j| {
                s = 0;
                for (l..n) |k| {
                    s += u.items[j][k] * u.items[i][k];
                }
                for (l..n) |k| {
                    u.items[j][k] += s * e.items[k][0];
                }
            }
        }
        y = @abs(q.items[i][0]) + @abs(e.items[i][0]);
        if (y > x) {
            x = y;
        }
    }

    // Accumulation of right-hand transformations.
    if (options.with_v) {
        for (0..n) |ri| {
            const i = n - 1 - ri;
            if (g != 0) {
                h = u.items[i][i + 1] * g;
                for (l..n) |j| {
                    v.items[j][i] = u.items[i][j] / h;
                }
                for (l..n) |j| {
                    s = 0;
                    for (l..n) |k| {
                        s += u.items[i][k] * v.items[k][j];
                    }
                    for (l..n) |k| {
                        v.items[k][j] += s * v.items[k][i];
                    }
                }
            }
            for (l..n) |j| {
                v.items[i][j] = 0;
                v.items[j][i] = 0;
            }
            v.items[i][i] = 1;
            g = e.items[i][0];
            l = i;
        }
    }

    // Accumulation of left-hand transformations.
    if (options.mode != .no_u) {
        for (n..u.rows) |i| {
            for (n..u.cols) |j| {
                u.items[i][j] = 0;
            }
            if (i < u.cols) {
                u.items[i][i] = 1;
            }
        }
    }

    if (options.mode != .no_u) {
        for (0..n) |ri| {
            const i = n - 1 - ri;
            l = i + 1;
            g = q.items[i][0];

            for (l..u.cols) |j| {
                u.items[i][j] = 0;
            }
            if (g != 0) {
                h = u.items[i][i] * g;
                for (l..u.cols) |j| {
                    s = 0;
                    for (l..m) |k| {
                        s += u.items[k][i] * u.items[k][j];
                    }
                    f = s / h;
                    for (i..m) |k| {
                        u.items[k][j] += f * u.items[k][i];
                    }
                }
                for (i..m) |j| {
                    u.items[j][i] /= g;
                }
            } else {
                for (i..m) |j| {
                    u.items[j][i] = 0;
                }
            }
            u.items[i][i] += 1;
        }
    }

    // Diagonalization of the bidiagonal form.
    eps *= x;

    for (0..n) |rk| {
        const k = n - 1 - rk;
        var iter: usize = 0;

        svd_state: switch (SvdState.test_splitting) {
            .test_splitting => {
                for (0..k + 1) |rl| {
                    l = k - rl;
                    if (@abs(e.items[l][0]) <= eps) {
                        continue :svd_state .test_convergence;
                    }
                    if (@abs(q.items[l - 1][0]) <= eps) {
                        continue :svd_state .cancellation;
                    }
                }
                continue :svd_state .test_convergence;
            },

            .cancellation => {
                // Cancellation of e.items[l][0] if l > 0
                c = 0;
                s = 1;
                const l1 = l - 1;
                for (l..k + 1) |i| {
                    f = s * e.items[i][0];
                    e.items[i][0] *= c;

                    if (@abs(f) <= eps) {
                        continue :svd_state .test_convergence;
                    }
                    g = q.items[i][0];
                    h = @sqrt(f * f + g * g);
                    q.items[i][0] = h;
                    c = g / h;
                    s = -f / h;
                    if (options.mode != .no_u) {
                        for (0..m) |j| {
                            y = u.items[j][l1];
                            z = u.items[j][i];
                            u.items[j][l1] = y * c + z * s;
                            u.items[j][i] = -y * s + z * c;
                        }
                    }
                }
                continue :svd_state .test_convergence;
            },

            .test_convergence => {
                z = q.items[k][0];
                if (l == k) {
                    continue :svd_state .convergence_check;
                }
                // Shift from bottom 2x2 minor.
                iter += 1;
                if (iter > 300) {
                    retval = k;
                    break :svd_state;
                }
                x = q.items[l][0];
                y = q.items[k - 1][0];
                g = e.items[k - 1][0];
                h = e.items[k][0];
                f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y);
                g = @sqrt(f * f + 1.0);
                f = ((x - z) * (x + z) + h * (y / (if (f < 0) (f - g) else (f + g)) - h)) / x;

                // Next QR transformation.
                c = 1;
                s = 1;
                for (l + 1..k + 1) |i| {
                    g = e.items[i][0];
                    y = q.items[i][0];
                    h = s * g;
                    g *= c;
                    z = @sqrt(f * f + h * h);
                    e.items[i - 1][0] = z;
                    c = f / z;
                    s = h / z;
                    f = x * c + g * s;
                    g = -x * s + g * c;
                    h = y * s;
                    y *= c;
                    if (options.with_v) {
                        for (0..n) |j| {
                            x = v.items[j][i - 1];
                            z = v.items[j][i];
                            v.items[j][i - 1] = x * c + z * s;
                            v.items[j][i] = -x * s + z * c;
                        }
                    }
                    z = @sqrt(f * f + h * h);
                    q.items[i - 1][0] = z;
                    if (z != 0) {
                        c = f / z;
                        s = h / z;
                    }
                    f = c * g + s * y;
                    x = -s * g + c * y;
                    if (options.mode != .no_u) {
                        for (0..m) |j| {
                            y = u.items[j][i - 1];
                            z = u.items[j][i];
                            u.items[j][i - 1] = y * c + z * s;
                            u.items[j][i] = -y * s + z * c;
                        }
                    }
                }
                e.items[l][0] = 0;
                e.items[k][0] = f;
                q.items[k][0] = x;
                continue :svd_state .test_splitting;
            },

            .convergence_check => {
                if (z < 0) {
                    // q.items[k][0]s made non-negative
                    q.items[k][0] = -z;
                    if (options.with_v) {
                        for (0..n) |j| {
                            v.items[j][k] = -v.items[j][k];
                        }
                    }
                }
                break :svd_state;
            },
        }
    }
    return .{ .u = u, .s = q, .v = v, .converged = retval };
}

test "svd basic" {
    const m: usize = 5;
    const n: usize = 4;
    // Example matrix taken from Wikipedia
    const a: SMatrix(f64, m, n) = .init(.{
        .{ 1, 0, 0, 0 },
        .{ 0, 0, 0, 2 },
        .{ 0, 3, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 2, 0, 0, 0 },
    });
    const res = svd(f64, m, n, a, .{ .with_u = true, .with_v = true, .mode = .full_u });
    const u = &res.u;
    const s = &res.s;
    const v = &res.v;
    var w: SMatrix(f64, m, n) = .initAll(0);
    // build the diagonal matrix from s.
    for (0..s.rows) |i| {
        w.items[i][i] = s.items[i][0];
    }
    // check decomposition
    const tol = @sqrt(std.math.floatEps(f64));
    const b = u.dot(w.dot(v.transpose()));
    for (0..m) |i| {
        for (0..n) |j| {
            try std.testing.expectApproxEqRel(a.items[i][j], b.items[i][j], tol);
        }
    }
    // check for orhonormality of u and v
    const id_m: SMatrix(f64, m, m) = .identity();
    const uut = u.dot(u.transpose());
    for (u.items) |row| {
        const vec: @Vector(m, f64) = row;
        const norm = @reduce(.Add, vec * vec);
        try std.testing.expectApproxEqRel(norm, 1, tol);
    }
    for (0..m) |i| {
        for (0..m) |j| {
            try std.testing.expectApproxEqAbs(uut.items[i][j], id_m.items[i][j], 1e-15);
        }
    }
    const id_n: SMatrix(f64, n, n) = .identity();
    const vvt = v.dot(v.transpose());
    for (v.items) |row| {
        const vec: @Vector(n, f64) = row;
        const norm = @reduce(.Add, vec * vec);
        try std.testing.expectApproxEqRel(norm, 1, tol);
    }
    for (0..n) |i| {
        for (0..n) |j| {
            try std.testing.expectApproxEqAbs(vvt.items[i][j], id_n.items[i][j], 1e-15);
        }
    }
}

test "svd modes" {
    const m: usize = 4;
    const n: usize = 4;
    const a: SMatrix(f64, m, n) = .init(.{
        .{ 2, 1, 0, 0 },
        .{ 1, 2, 1, 0 },
        .{ 0, 1, 2, 1 },
        .{ 0, 0, 1, 2 },
    });

    // Test no_u mode
    const res_no_u = svd(f64, m, n, a, .{ .with_u = false, .with_v = true, .mode = .no_u });
    const s_no_u = &res_no_u.s;
    _ = res_no_u.v; // v_no_u

    // Test skinny_u mode
    const res_skinny = svd(f64, m, n, a, .{ .with_u = true, .with_v = false, .mode = .skinny_u });
    const u_skinny = &res_skinny.u;
    const s_skinny = &res_skinny.s;

    // Test full_u mode
    const res_full = svd(f64, m, n, a, .{ .with_u = true, .with_v = true, .mode = .full_u });
    const u_full = &res_full.u;
    const s_full = &res_full.s;
    _ = res_full.v; // v_full

    // Singular values should be the same across modes
    const tol = @sqrt(std.math.floatEps(f64));
    for (0..n) |i| {
        try std.testing.expectApproxEqRel(s_no_u.items[i][0], s_skinny.items[i][0], tol);
        try std.testing.expectApproxEqRel(s_skinny.items[i][0], s_full.items[i][0], tol);
    }

    // Check matrix dimensions
    try std.testing.expect(u_skinny.rows == m and u_skinny.cols == n);
    try std.testing.expect(u_full.rows == m and u_full.cols == m);
}

test "svd identity matrix" {
    const n: usize = 3;
    const a: SMatrix(f64, n, n) = .identity();

    const res = svd(f64, n, n, a, .{ .with_u = true, .with_v = true, .mode = .full_u });
    const s = &res.s;

    // Identity matrix should have all singular values equal to 1
    const tol = @sqrt(std.math.floatEps(f64));
    for (0..n) |i| {
        try std.testing.expectApproxEqRel(s.items[i][0], 1.0, tol);
    }
}

test "svd singular matrix" {
    const m: usize = 3;
    const n: usize = 3;
    const a: SMatrix(f64, m, n) = .init(.{
        .{ 1, 2, 3 },
        .{ 2, 4, 6 },
        .{ 1, 2, 3 },
    });

    const res = svd(f64, m, n, a, .{ .with_u = true, .with_v = true, .mode = .full_u });
    const s = &res.s;

    // This matrix has rank 1, so should have 2 zero singular values
    const tol = @sqrt(std.math.floatEps(f64));
    var zero_count: usize = 0;
    for (0..n) |i| {
        if (s.items[i][0] < tol) {
            zero_count += 1;
        }
    }
    try std.testing.expect(zero_count == 2);
}
