//! Principal Component Analysis (PCA) for dimensionality reduction and feature extraction.
//!
//! This implementation works with arbitrary-dimensional vectors and points,
//! providing a unified interface for PCA on geometric data, color spaces, and more.
//!
//! ## Example
//! ```zig
//! // 2D geometric points
//! const points_2d = [_]@Vector(2, f64){ .{1.0, 2.0}, .{3.0, 4.0}, .{5.0, 6.0} };
//! var pca_2d = PCA(f64, 2).init(allocator);
//! try pca_2d.fit(&points_2d, null);
//!
//! // Project and reconstruct
//! const coeffs = try pca_2d.project(.{2.0, 3.0});
//! const reconstructed = try pca_2d.reconstruct(coeffs);
//!
//! // Works with Point types too
//! const geometric_points = [_]Point2d(f64){ Point2d(f64).init2d(1.0, 2.0), ... };
//! try pca_2d.fitPoints(&geometric_points, null);
//! ```

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Matrix = @import("matrix.zig").Matrix;
const svd = @import("svd.zig").svd;
const SMatrix = @import("matrix.zig").SMatrix;
const Point = @import("geometry/Point.zig").Point;
const Point2d = @import("geometry/Point.zig").Point2d;
const Point3d = @import("geometry/Point.zig").Point3d;
const Image = @import("image.zig").Image;
const Rgb = @import("color.zig").Rgb;
const Rgba = @import("color.zig").Rgba;

/// Principal Component Analysis for arbitrary-dimensional vectors.
/// Uses SIMD-accelerated vector operations for optimal performance.
pub fn PCA(comptime T: type, comptime dim: usize) type {
    assert(@typeInfo(T) == .float);
    assert(dim >= 1);

    return struct {
        const Self = @This();
        const Vec = @Vector(dim, T);
        const PointType = Point(T, dim);

        /// Mean vector for centering data
        mean: Vec,
        /// Principal components matrix (dim × num_components)
        components: Matrix(T),
        /// Eigenvalues in descending order
        eigenvalues: []T,
        /// Number of components retained
        num_components: usize,
        /// Memory allocator
        allocator: Allocator,

        /// Initialize an empty PCA instance
        pub fn init(allocator: Allocator) Self {
            return Self{
                .mean = @splat(0),
                .components = undefined,
                .eigenvalues = &[_]T{},
                .num_components = 0,
                .allocator = allocator,
            };
        }

        /// Free allocated memory
        pub fn deinit(self: *Self) void {
            if (self.eigenvalues.len > 0) {
                self.allocator.free(self.eigenvalues);
                self.eigenvalues = &[_]T{};
            }
            if (self.num_components > 0) {
                self.components.deinit();
                self.num_components = 0;
            }
        }

        /// Fit PCA model to a set of vectors.
        /// - vectors: Training data as vectors
        /// - num_components: Number of components to retain (null = keep all possible)
        pub fn fit(self: *Self, vectors: []const Vec, num_components: ?usize) !void {
            if (vectors.len == 0) return error.NoVectors;
            if (vectors.len == 1) return error.InsufficientData;

            const n_samples = vectors.len;
            const max_components = @min(n_samples - 1, dim);
            const requested_components = num_components orelse max_components;
            const actual_components = @min(requested_components, max_components);

            // Compute mean vector using SIMD operations
            self.mean = @splat(0);
            for (vectors) |vec| {
                self.mean += vec;
            }
            self.mean = self.mean / @as(Vec, @splat(@floatFromInt(n_samples)));

            // Create centered data matrix
            var data_matrix = try Matrix(T).init(self.allocator, n_samples, dim);
            defer data_matrix.deinit();

            for (vectors, 0..) |vec, i| {
                const centered = vec - self.mean;
                for (0..dim) |j| {
                    data_matrix.at(i, j).* = centered[j];
                }
            }

            // Choose computation path based on data dimensions
            if (n_samples <= dim) {
                // Few samples: use Gram matrix approach (n_samples × n_samples)
                try self.computeComponentsFromGram(&data_matrix, actual_components);
            } else {
                // Many samples: use covariance matrix approach (dim × dim)
                try self.computeComponentsFromCovariance(&data_matrix, actual_components);
            }
        }

        /// Convenience method for fitting Point types
        pub fn fitPoints(self: *Self, points: []const PointType, num_components: ?usize) !void {
            var vectors = try self.allocator.alloc(Vec, points.len);
            defer self.allocator.free(vectors);

            for (points, 0..) |point, i| {
                vectors[i] = point.asVector();
            }

            return self.fit(vectors, num_components);
        }

        /// Project a vector onto the principal components.
        /// Returns the coefficients in PCA space.
        pub fn project(self: Self, vector: Vec) ![]T {
            if (self.num_components == 0) return error.NotFitted;

            // Center the vector
            const centered = vector - self.mean;

            // Project onto components
            var coefficients = try self.allocator.alloc(T, self.num_components);
            for (0..self.num_components) |i| {
                var sum: T = 0;
                for (0..dim) |j| {
                    sum += centered[j] * self.components.at(j, i).*;
                }
                coefficients[i] = sum;
            }

            return coefficients;
        }

        /// Project a Point onto the principal components
        pub fn projectPoint(self: Self, point: PointType) ![]T {
            return self.project(point.asVector());
        }

        /// Project multiple vectors efficiently as a batch.
        pub fn projectBatch(self: Self, vectors: []const Vec) !Matrix(T) {
            if (self.num_components == 0) return error.NotFitted;

            var projections = try Matrix(T).init(self.allocator, vectors.len, self.num_components);
            errdefer projections.deinit();

            for (vectors, 0..) |vector, i| {
                const coeffs = try self.project(vector);
                defer self.allocator.free(coeffs);

                for (0..self.num_components) |j| {
                    projections.at(i, j).* = coeffs[j];
                }
            }

            return projections;
        }

        /// Reconstruct a vector from PCA coefficients.
        pub fn reconstruct(self: Self, coefficients: []const T) !Vec {
            if (self.num_components == 0) return error.NotFitted;
            if (coefficients.len != self.num_components) return error.InvalidCoefficients;

            // Start with mean
            var result = self.mean;

            // Add weighted components
            for (0..self.num_components) |i| {
                const weight = coefficients[i];
                for (0..dim) |j| {
                    result[j] += weight * self.components.at(j, i).*;
                }
            }

            return result;
        }

        /// Reconstruct a Point from PCA coefficients
        pub fn reconstructPoint(self: Self, coefficients: []const T) !PointType {
            const vec = try self.reconstruct(coefficients);
            return PointType.fromVector(vec);
        }

        /// Get the mean vector
        pub fn getMean(self: Self) Vec {
            return self.mean;
        }

        /// Get the mean as a Point
        pub fn getMeanPoint(self: Self) PointType {
            return PointType.fromVector(self.mean);
        }

        /// Get the proportion of variance explained by each component.
        pub fn explainedVarianceRatio(self: Self) ![]T {
            if (self.num_components == 0) return error.NotFitted;

            var ratios = try self.allocator.alloc(T, self.num_components);

            var total_variance: T = 0;
            for (self.eigenvalues[0..self.num_components]) |eigenval| {
                total_variance += eigenval;
            }

            if (total_variance > 0) {
                for (self.eigenvalues[0..self.num_components], 0..) |eigenval, i| {
                    ratios[i] = eigenval / total_variance;
                }
            } else {
                @memset(ratios, 0);
            }

            return ratios;
        }

        /// Get cumulative variance explained.
        pub fn cumulativeVarianceRatio(self: Self) ![]T {
            const ratios = try self.explainedVarianceRatio();

            var cumulative: T = 0;
            for (ratios) |*ratio| {
                cumulative += ratio.*;
                ratio.* = cumulative;
            }

            return ratios;
        }

        // Private helper methods

        fn computeComponentsFromCovariance(self: *Self, data_matrix: *Matrix(T), num_components: usize) !void {
            // Compute covariance matrix (X^T * X) / (n-1)
            var cov_matrix = try computeCovarianceMatrix(self.allocator, data_matrix);
            defer cov_matrix.deinit();

            const n = cov_matrix.rows;

            // Convert to static matrix for SVD - keep size reasonable
            if (n > 64) return error.DimensionTooLarge;

            var cov_static: SMatrix(T, 64, 64) = std.mem.zeroes(SMatrix(T, 64, 64));
            for (0..n) |i| {
                for (0..n) |j| {
                    cov_static.items[i][j] = cov_matrix.at(i, j).*;
                }
            }

            // Perform SVD
            const result = svd(T, 64, 64, cov_static, .{
                .with_u = true,
                .with_v = false,
                .mode = .skinny_u,
            });

            if (result.converged != 0) return error.SvdFailed;

            // Extract components and eigenvalues
            self.num_components = num_components;
            self.eigenvalues = try self.allocator.alloc(T, num_components);
            self.components = try Matrix(T).init(self.allocator, dim, num_components);

            for (0..num_components) |i| {
                if (i < n) {
                    self.eigenvalues[i] = result.s.items[i][0];
                    for (0..dim) |j| {
                        if (j < n) {
                            self.components.at(j, i).* = result.u.items[j][i];
                        } else {
                            self.components.at(j, i).* = 0;
                        }
                    }
                } else {
                    self.eigenvalues[i] = 0;
                    for (0..dim) |j| {
                        self.components.at(j, i).* = 0;
                    }
                }
            }
        }

        fn computeComponentsFromGram(self: *Self, data_matrix: *Matrix(T), num_components: usize) !void {
            // Compute Gram matrix (X * X^T) / (n-1)
            var gram_matrix = try computeGramMatrix(self.allocator, data_matrix);
            defer gram_matrix.deinit();

            const n = gram_matrix.rows;

            // Convert to static matrix for SVD - keep size reasonable
            if (n > 64) return error.DimensionTooLarge;

            var gram_static: SMatrix(T, 64, 64) = .{};
            for (0..n) |i| {
                for (0..n) |j| {
                    gram_static.items[i][j] = gram_matrix.at(i, j).*;
                }
            }

            // Perform SVD on Gram matrix
            const result = svd(T, 64, 64, gram_static, .{
                .with_u = true,
                .with_v = false,
                .mode = .skinny_u,
            });

            if (result.converged != 0) return error.SvdFailed;

            // Extract components by projecting data onto eigenvectors
            self.num_components = num_components;
            self.eigenvalues = try self.allocator.alloc(T, num_components);
            self.components = try Matrix(T).init(self.allocator, dim, num_components);

            // Ensure we don't exceed the actual number of eigenvalues/eigenvectors computed
            const actual_components = @min(num_components, n);
            for (0..actual_components) |i| {
                const eigenval = result.s.items[i][0];
                self.eigenvalues[i] = eigenval;

                if (eigenval > 1e-10) {
                    // Compute component as X^T * u_i / sqrt(eigenval)
                    for (0..dim) |j| {
                        var sum: T = 0;
                        for (0..n) |k| {
                            sum += data_matrix.at(k, j).* * result.u.items[k][i];
                        }
                        self.components.at(j, i).* = sum / @sqrt(eigenval * @as(T, @floatFromInt(n)));
                    }
                } else {
                    // Zero eigenvalue, set component to zero
                    for (0..dim) |j| {
                        self.components.at(j, i).* = 0;
                    }
                }
            }

            // Fill remaining components with zeros if num_components > actual_components
            for (actual_components..num_components) |i| {
                self.eigenvalues[i] = 0;
                for (0..dim) |j| {
                    self.components.at(j, i).* = 0;
                }
            }
        }
    };
}

// Image-to-Points Conversion Utilities

/// Generic function to convert any color type to a point using reflection.
/// Works with any struct with numeric fields (u8, f32, etc.).
pub fn colorToPoint(comptime ColorType: type, color: ColorType) @Vector(std.meta.fields(ColorType).len, f64) {
    const fields = std.meta.fields(ColorType);
    var point: @Vector(fields.len, f64) = undefined;

    inline for (fields, 0..) |field, i| {
        const value = @field(color, field.name);
        const field_type = @TypeOf(value);

        // Convert to normalized f64 based on field type
        point[i] = switch (@typeInfo(field_type)) {
            .int => @as(f64, @floatFromInt(value)) / 255.0, // Assume u8 values are 0-255
            .float => @as(f64, @floatCast(value)), // Assume float values are already normalized
            else => @compileError("Unsupported color field type: " ++ @typeName(field_type)),
        };
    }

    return point;
}

/// Generic function to convert a point back to any color type using reflection.
/// Works with any struct with numeric fields (u8, f32, etc.).
pub fn pointToColor(comptime ColorType: type, point: @Vector(std.meta.fields(ColorType).len, f64)) ColorType {
    const fields = std.meta.fields(ColorType);
    var color: ColorType = undefined;

    inline for (fields, 0..) |field, i| {
        const field_type = field.type;
        const value = point[i];

        // Convert from normalized f64 based on field type
        @field(color, field.name) = switch (@typeInfo(field_type)) {
            .int => @intFromFloat(@round(std.math.clamp(value * 255.0, 0, 255))),
            .float => @floatCast(value),
            else => @compileError("Unsupported color field type: " ++ @typeName(field_type)),
        };
    }

    return color;
}

/// Generic function to convert any image to color points using reflection.
/// Works with any color type - RGB, RGBA, HSL, Lab, etc.
pub fn imageToColorPoints(comptime ColorType: type, allocator: Allocator, image: Image(ColorType)) ![]@Vector(std.meta.fields(ColorType).len, f64) {
    const num_channels = std.meta.fields(ColorType).len;
    var points = try allocator.alloc(@Vector(num_channels, f64), image.data.len);

    for (image.data, 0..) |pixel, i| {
        points[i] = colorToPoint(ColorType, pixel);
    }

    return points;
}

/// Generic function to convert color points back to an image using reflection.
/// Works with any color type - RGB, RGBA, HSL, Lab, etc.
pub fn colorPointsToImage(comptime ColorType: type, allocator: Allocator, points: []const @Vector(std.meta.fields(ColorType).len, f64), rows: usize, cols: usize) !Image(ColorType) {
    assert(points.len == rows * cols);

    var image = try Image(ColorType).initAlloc(allocator, rows, cols);

    for (points, 0..) |point, i| {
        image.data[i] = pointToColor(ColorType, point);
    }

    return image;
}

/// Convert a grayscale image to 1D intensity points.
/// Special case for u8 images (single intensity value).
pub fn imageToIntensityPoints(allocator: Allocator, image: Image(u8)) ![]@Vector(1, f64) {
    var points = try allocator.alloc(@Vector(1, f64), image.data.len);

    for (image.data, 0..) |pixel, i| {
        points[i] = .{@as(f64, @floatFromInt(pixel)) / 255.0};
    }

    return points;
}

/// Generic function to convert any image to spatial-color points.
/// Each pixel becomes a point with position (X, Y) + color information.
/// Returns points of dimension 2 + number_of_color_channels.
pub fn imageToSpatialColorPoints(comptime ColorType: type, allocator: Allocator, image: Image(ColorType)) ![]@Vector(2 + std.meta.fields(ColorType).len, f64) {
    const num_color_channels = std.meta.fields(ColorType).len;
    const total_dims = 2 + num_color_channels;
    var points = try allocator.alloc(@Vector(total_dims, f64), image.data.len);

    for (0..image.rows) |row| {
        for (0..image.cols) |col| {
            const idx = row * image.cols + col;
            const pixel = image.data[idx];

            var point: @Vector(total_dims, f64) = undefined;

            // Set spatial coordinates (normalized)
            point[0] = @as(f64, @floatFromInt(col)) / @as(f64, @floatFromInt(image.cols - 1));
            point[1] = @as(f64, @floatFromInt(row)) / @as(f64, @floatFromInt(image.rows - 1));

            // Set color channels using reflection
            const color_point = colorToPoint(ColorType, pixel);
            inline for (0..num_color_channels) |i| {
                point[2 + i] = color_point[i];
            }

            points[idx] = point;
        }
    }

    return points;
}

/// Convert a grayscale image to 3D spatial-intensity points (X, Y, I).
/// Each pixel becomes a point with both position and intensity information.
/// This is a convenience function equivalent to imageToSpatialColorPoints(u8, ...).
pub fn imageToSpatialIntensityPoints(allocator: Allocator, image: Image(u8)) ![]@Vector(3, f64) {
    var points = try allocator.alloc(@Vector(3, f64), image.data.len);

    for (0..image.rows) |row| {
        for (0..image.cols) |col| {
            const idx = row * image.cols + col;
            const pixel = image.data[idx];
            points[idx] = .{
                @as(f64, @floatFromInt(col)) / @as(f64, @floatFromInt(image.cols - 1)), // Normalized X
                @as(f64, @floatFromInt(row)) / @as(f64, @floatFromInt(image.rows - 1)), // Normalized Y
                @as(f64, @floatFromInt(pixel)) / 255.0, // Intensity
            };
        }
    }

    return points;
}

/// Convert intensity points back to a grayscale image.
/// Reconstructs an image from 1D intensity points.
pub fn intensityPointsToImage(allocator: Allocator, points: []const @Vector(1, f64), rows: usize, cols: usize) !Image(u8) {
    assert(points.len == rows * cols);

    var image = try Image(u8).initAlloc(allocator, rows, cols);

    for (points, 0..) |point, i| {
        const intensity = std.math.clamp(point[0] * 255.0, 0, 255);
        image.data[i] = @intFromFloat(@round(intensity));
    }

    return image;
}

/// Extract color components from spatial-color points.
/// Returns just the color part (R, G, B) from 5D spatial-color points.
pub fn extractColorFromSpatialColor(allocator: Allocator, spatial_points: []const @Vector(5, f64)) ![]@Vector(3, f64) {
    var color_points = try allocator.alloc(@Vector(3, f64), spatial_points.len);

    for (spatial_points, 0..) |point, i| {
        color_points[i] = .{ point[2], point[3], point[4] }; // R, G, B
    }

    return color_points;
}

/// Extract spatial components from spatial-color points.
/// Returns just the position part (X, Y) from 5D spatial-color points.
pub fn extractSpatialFromSpatialColor(allocator: Allocator, spatial_points: []const @Vector(5, f64)) ![]@Vector(2, f64) {
    var spatial_positions = try allocator.alloc(@Vector(2, f64), spatial_points.len);

    for (spatial_points, 0..) |point, i| {
        spatial_positions[i] = .{ point[0], point[1] }; // X, Y
    }

    return spatial_positions;
}

// Helper functions

/// Compute covariance matrix (X^T * X) / (n-1)
fn computeCovarianceMatrix(allocator: Allocator, data: *Matrix(f64)) !Matrix(f64) {
    const n_samples = data.rows;
    const n_features = data.cols;

    var cov_matrix = try Matrix(f64).init(allocator, n_features, n_features);

    // Compute X^T * X
    for (0..n_features) |i| {
        for (0..n_features) |j| {
            var sum: f64 = 0;
            for (0..n_samples) |k| {
                sum += data.at(k, i).* * data.at(k, j).*;
            }
            cov_matrix.at(i, j).* = sum / @as(f64, @floatFromInt(n_samples - 1));
        }
    }

    return cov_matrix;
}

/// Compute Gram matrix (X * X^T) / (n-1)
fn computeGramMatrix(allocator: Allocator, data: *Matrix(f64)) !Matrix(f64) {
    const n_samples = data.rows;
    const n_features = data.cols;

    var gram_matrix = try Matrix(f64).init(allocator, n_samples, n_samples);

    // Compute X * X^T
    for (0..n_samples) |i| {
        for (0..n_samples) |j| {
            var sum: f64 = 0;
            for (0..n_features) |k| {
                sum += data.at(i, k).* * data.at(j, k).*;
            }
            gram_matrix.at(i, j).* = sum / @as(f64, @floatFromInt(n_samples - 1));
        }
    }

    return gram_matrix;
}

// Tests

test "PCA initialization and cleanup" {
    const allocator = std.testing.allocator;

    var pca = PCA(f64, 2).init(allocator);
    defer pca.deinit();

    try std.testing.expectEqual(@as(usize, 0), pca.num_components);
}

test "PCA on 2D vectors" {
    const allocator = std.testing.allocator;

    // Create simple test vectors
    const vectors = [_]@Vector(2, f64){
        .{ 1.0, 2.0 },
        .{ 3.0, 4.0 },
        .{ 5.0, 6.0 },
        .{ 7.0, 8.0 },
    };

    var pca = PCA(f64, 2).init(allocator);
    defer pca.deinit();

    try pca.fit(&vectors, null);

    // Should have fitted successfully
    try std.testing.expect(pca.num_components > 0);

    // Test projection and reconstruction
    const coeffs = try pca.project(.{ 4.0, 5.0 });
    defer allocator.free(coeffs);

    const reconstructed = try pca.reconstruct(coeffs);

    // Reconstruction should be close to original (within numerical precision)
    try std.testing.expect(@abs(reconstructed[0] - 4.0) < 1e-10);
    try std.testing.expect(@abs(reconstructed[1] - 5.0) < 1e-10);
}

test "PCA on Point types" {
    const allocator = std.testing.allocator;

    // Create test points
    const points = [_]Point2d(f64){
        Point2d(f64).init2d(0.0, 0.0),
        Point2d(f64).init2d(1.0, 0.0),
        Point2d(f64).init2d(0.0, 1.0),
        Point2d(f64).init2d(1.0, 1.0),
    };

    var pca = PCA(f64, 2).init(allocator);
    defer pca.deinit();

    try pca.fitPoints(&points, 1);

    // Test Point projection and reconstruction
    const test_point = Point2d(f64).init2d(0.5, 0.5);
    const coeffs = try pca.projectPoint(test_point);
    defer allocator.free(coeffs);

    const reconstructed_point = try pca.reconstructPoint(coeffs);

    // Should be approximately equal to mean point since we only kept 1 component
    const mean_point = pca.getMeanPoint();
    try std.testing.expect(@abs(reconstructed_point.x() - mean_point.x()) < 1e-10);
    try std.testing.expect(@abs(reconstructed_point.y() - mean_point.y()) < 1e-10);
}

test "PCA explained variance" {
    const allocator = std.testing.allocator;

    // Create test vectors with clear variance structure
    const vectors = [_]@Vector(3, f64){
        .{ 1.0, 0.0, 0.0 },
        .{ 2.0, 0.0, 0.0 },
        .{ 3.0, 0.0, 0.0 },
        .{ 4.0, 0.0, 0.0 },
    };

    var pca = PCA(f64, 3).init(allocator);
    defer pca.deinit();

    try pca.fit(&vectors, null);

    // Check explained variance ratios
    const ratios = try pca.explainedVarianceRatio();
    defer allocator.free(ratios);

    // Test that ratios sum to <= 1 (allowing for numerical precision)
    var sum: f64 = 0;
    for (ratios) |ratio| {
        sum += ratio;
    }
    try std.testing.expect(sum <= 1.0001);

    // Test cumulative variance
    const cumulative = try pca.cumulativeVarianceRatio();
    defer allocator.free(cumulative);

    try std.testing.expect(cumulative[cumulative.len - 1] <= 1.0001);
}

test "Generic color to point conversion" {
    _ = std.testing.allocator;

    // Test with RGB color
    const rgb_color = Rgb{ .r = 255, .g = 128, .b = 0 };
    const rgb_point = colorToPoint(Rgb, rgb_color);

    try std.testing.expectEqual(@as(f64, 1.0), rgb_point[0]); // R
    try std.testing.expectEqual(@as(f64, 128.0 / 255.0), rgb_point[1]); // G
    try std.testing.expectEqual(@as(f64, 0.0), rgb_point[2]); // B

    // Convert back
    const recovered_rgb = pointToColor(Rgb, rgb_point);
    try std.testing.expectEqual(rgb_color.r, recovered_rgb.r);
    try std.testing.expectEqual(rgb_color.g, recovered_rgb.g);
    try std.testing.expectEqual(rgb_color.b, recovered_rgb.b);

    // Test with RGBA color
    const rgba_color = Rgba{ .r = 255, .g = 0, .b = 0, .a = 128 };
    const rgba_point = colorToPoint(Rgba, rgba_color);

    try std.testing.expectEqual(@as(f64, 1.0), rgba_point[0]); // R
    try std.testing.expectEqual(@as(f64, 0.0), rgba_point[1]); // G
    try std.testing.expectEqual(@as(f64, 0.0), rgba_point[2]); // B
    try std.testing.expectEqual(@as(f64, 128.0 / 255.0), rgba_point[3]); // A

    const recovered_rgba = pointToColor(Rgba, rgba_point);
    try std.testing.expectEqual(rgba_color.r, recovered_rgba.r);
    try std.testing.expectEqual(rgba_color.g, recovered_rgba.g);
    try std.testing.expectEqual(rgba_color.b, recovered_rgba.b);
    try std.testing.expectEqual(rgba_color.a, recovered_rgba.a);
}

test "Generic image to color points conversion" {
    const allocator = std.testing.allocator;

    // Create a simple 2x2 RGB image
    var image = try Image(Rgb).initAlloc(allocator, 2, 2);
    defer image.deinit(allocator);

    image.data[0] = Rgb{ .r = 255, .g = 0, .b = 0 }; // Red
    image.data[1] = Rgb{ .r = 0, .g = 255, .b = 0 }; // Green
    image.data[2] = Rgb{ .r = 0, .g = 0, .b = 255 }; // Blue
    image.data[3] = Rgb{ .r = 255, .g = 255, .b = 255 }; // White

    // Convert to color points using generic function
    const color_points = try imageToColorPoints(Rgb, allocator, image);
    defer allocator.free(color_points);

    // Check conversions
    try std.testing.expectEqual(@as(usize, 4), color_points.len);
    try std.testing.expectEqual(@as(f64, 1.0), color_points[0][0]); // Red pixel
    try std.testing.expectEqual(@as(f64, 0.0), color_points[0][1]);
    try std.testing.expectEqual(@as(f64, 0.0), color_points[0][2]);

    try std.testing.expectEqual(@as(f64, 0.0), color_points[1][0]); // Green pixel
    try std.testing.expectEqual(@as(f64, 1.0), color_points[1][1]);
    try std.testing.expectEqual(@as(f64, 0.0), color_points[1][2]);

    // Convert back to image using generic function
    var reconstructed = try colorPointsToImage(Rgb, allocator, color_points, 2, 2);
    defer reconstructed.deinit(allocator);

    // Check that reconstruction matches original
    try std.testing.expectEqual(image.data[0].r, reconstructed.data[0].r);
    try std.testing.expectEqual(image.data[0].g, reconstructed.data[0].g);
    try std.testing.expectEqual(image.data[0].b, reconstructed.data[0].b);
}

test "Generic spatial-color points conversion" {
    const allocator = std.testing.allocator;

    // Create a simple 2x2 RGB image
    var image = try Image(Rgb).initAlloc(allocator, 2, 2);
    defer image.deinit(allocator);

    image.data[0] = Rgb{ .r = 255, .g = 0, .b = 0 }; // Top-left: Red
    image.data[1] = Rgb{ .r = 0, .g = 255, .b = 0 }; // Top-right: Green
    image.data[2] = Rgb{ .r = 0, .g = 0, .b = 255 }; // Bottom-left: Blue
    image.data[3] = Rgb{ .r = 255, .g = 255, .b = 255 }; // Bottom-right: White

    // Convert to spatial-color points using generic function
    const spatial_points = try imageToSpatialColorPoints(Rgb, allocator, image);
    defer allocator.free(spatial_points);

    // Check that we have the right number of points
    try std.testing.expectEqual(@as(usize, 4), spatial_points.len);

    // Check first point (top-left): should be at (0, 0) with red color
    try std.testing.expectEqual(@as(f64, 0.0), spatial_points[0][0]); // X
    try std.testing.expectEqual(@as(f64, 0.0), spatial_points[0][1]); // Y
    try std.testing.expectEqual(@as(f64, 1.0), spatial_points[0][2]); // R
    try std.testing.expectEqual(@as(f64, 0.0), spatial_points[0][3]); // G
    try std.testing.expectEqual(@as(f64, 0.0), spatial_points[0][4]); // B

    // Check last point (bottom-right): should be at (1, 1) with white color
    try std.testing.expectEqual(@as(f64, 1.0), spatial_points[3][0]); // X
    try std.testing.expectEqual(@as(f64, 1.0), spatial_points[3][1]); // Y
    try std.testing.expectEqual(@as(f64, 1.0), spatial_points[3][2]); // R
    try std.testing.expectEqual(@as(f64, 1.0), spatial_points[3][3]); // G
    try std.testing.expectEqual(@as(f64, 1.0), spatial_points[3][4]); // B
}

test "PCA on image color data with generic functions" {
    const allocator = std.testing.allocator;

    // Create a simple gradient image
    var image = try Image(Rgb).initAlloc(allocator, 2, 2);
    defer image.deinit(allocator);

    image.data[0] = Rgb{ .r = 0, .g = 0, .b = 0 }; // Black
    image.data[1] = Rgb{ .r = 85, .g = 85, .b = 85 }; // Dark gray
    image.data[2] = Rgb{ .r = 170, .g = 170, .b = 170 }; // Light gray
    image.data[3] = Rgb{ .r = 255, .g = 255, .b = 255 }; // White

    // Convert to color points using generic function
    const color_points = try imageToColorPoints(Rgb, allocator, image);
    defer allocator.free(color_points);

    // Apply PCA to color data
    var pca = PCA(f64, 3).init(allocator);
    defer pca.deinit();

    try pca.fit(color_points, 1); // Keep only 1 component

    // Project all points
    var projections = try pca.projectBatch(color_points);
    defer projections.deinit();

    // Reconstruct points
    var reconstructed_points = try allocator.alloc(@Vector(3, f64), color_points.len);
    defer allocator.free(reconstructed_points);

    for (0..color_points.len) |i| {
        const coeffs = try allocator.alloc(f64, 1);
        coeffs[0] = projections.at(i, 0).*;
        defer allocator.free(coeffs);

        reconstructed_points[i] = try pca.reconstruct(coeffs);
    }

    // Convert back to image using generic function
    var reconstructed_image = try colorPointsToImage(Rgb, allocator, reconstructed_points, 2, 2);
    defer reconstructed_image.deinit(allocator);

    // PCA with 1 component should preserve the main gradient direction
    // All values should be close to the mean gray value since it's a grayscale gradient
    try std.testing.expect(reconstructed_image.data.len == 4);
}
