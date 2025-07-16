//! Principal Component Analysis (PCA) for dimensionality reduction and feature extraction.
//!
//! PCA is a statistical technique that identifies the principal axes of variation in data,
//! allowing for dimensionality reduction while preserving the most important information.
//! This implementation supports image data and follows Zignal's design patterns.
//!
//! ## Example
//! ```zig
//! var pca = PrincipalComponentAnalysis(Rgb).init(allocator);
//! defer pca.deinit();
//!
//! // Fit PCA to training images
//! try pca.fit(training_images, 50); // Keep top 50 components
//!
//! // Project new image to PCA space
//! const coefficients = try pca.project(test_image);
//! defer allocator.free(coefficients);
//!
//! // Reconstruct image from coefficients
//! var reconstructed = try pca.reconstruct(coefficients, test_image.rows, test_image.cols);
//! defer reconstructed.deinit(allocator);
//! ```

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Image = @import("image.zig").Image;
const Matrix = @import("matrix.zig").Matrix;
const OpsBuilder = @import("matrix.zig").OpsBuilder;
const svd = @import("svd.zig").svd;
const SMatrix = @import("matrix.zig").SMatrix;
const Rgb = @import("color.zig").Rgb;
const Rgba = @import("color.zig").Rgba;

/// Principal Component Analysis for image dimensionality reduction.
/// Supports RGB, RGBA, and grayscale images.
pub fn PrincipalComponentAnalysis(comptime T: type) type {
    comptime assert(T == Rgb or T == Rgba or T == u8);
    
    return struct {
        const Self = @This();
        
        /// Mean vector for centering data
        mean: []f64,
        /// Principal components (columns are eigenvectors)
        components: Matrix(f64),
        /// Eigenvalues in descending order
        eigenvalues: []f64,
        /// Number of components retained
        num_components: usize,
        /// Number of features (pixels × channels)
        num_features: usize,
        /// Memory allocator
        allocator: Allocator,
        /// Whether the data is grayscale
        is_grayscale: bool,
        
        /// Initialize an empty PCA instance
        pub fn init(allocator: Allocator) Self {
            return Self{
                .mean = &[_]f64{},
                .components = undefined,
                .eigenvalues = &[_]f64{},
                .num_components = 0,
                .num_features = 0,
                .allocator = allocator,
                .is_grayscale = (T == u8),
            };
        }
        
        /// Free allocated memory
        pub fn deinit(self: *Self) void {
            if (self.mean.len > 0) {
                self.allocator.free(self.mean);
                self.mean = &[_]f64{};
            }
            if (self.eigenvalues.len > 0) {
                self.allocator.free(self.eigenvalues);
                self.eigenvalues = &[_]f64{};
            }
            if (self.num_components > 0) {
                self.components.deinit();
                self.num_components = 0;
            }
        }
        
        /// Fit PCA model to a set of images.
        /// - images: Training images (must all have same dimensions)
        /// - num_components: Number of components to retain (null = keep all)
        pub fn fit(self: *Self, images: []const Image(T), num_components: ?usize) !void {
            if (images.len == 0) return error.NoImages;
            
            // Validate all images have same dimensions
            const rows = images[0].rows;
            const cols = images[0].cols;
            for (images[1..]) |img| {
                if (img.rows != rows or img.cols != cols) {
                    return error.InconsistentImageDimensions;
                }
            }
            
            // Determine number of features and channels
            const channels: usize = if (T == u8) 1 else 3;
            self.num_features = rows * cols * channels;
            
            // Convert images to data matrix (n_samples × n_features)
            var data_matrix = try imagesToMatrix(T, self.allocator, images, self.num_features);
            defer data_matrix.deinit();
            
            // For RGB/RGBA, detect if all images are actually grayscale
            if (T != u8) {
                self.is_grayscale = detectGrayscale(T, images);
            }
            
            // Compute mean and center the data
            self.mean = try computeMean(self.allocator, &data_matrix);
            centerData(&data_matrix, self.mean);
            
            // Compute covariance matrix efficiently
            const n_samples = images.len;
            const max_components = @min(n_samples - 1, self.num_features);
            const requested_components = num_components orelse max_components;
            
            // Clamp to available dimensions
            const actual_components = @min(requested_components, max_components);
            
            // Use economy SVD on the data matrix directly (more efficient than covariance)
            // X = U * S * V^T, where columns of V are the principal components
            if (n_samples < self.num_features) {
                // For n_samples < n_features, compute SVD of X * X^T instead
                var gram_matrix = try computeGramMatrix(self.allocator, &data_matrix);
                defer gram_matrix.deinit();
                
                // Perform eigendecomposition
                try self.computeComponentsFromGram(&data_matrix, &gram_matrix, actual_components);
            } else {
                // For n_samples >= n_features, compute SVD of X^T * X (covariance)
                var cov_matrix = try computeCovarianceMatrix(self.allocator, &data_matrix);
                defer cov_matrix.deinit();
                
                // Perform eigendecomposition
                try self.computeComponentsFromCovariance(&cov_matrix, actual_components);
            }
        }
        
        /// Project an image onto the principal components.
        /// Returns the coefficients in PCA space.
        pub fn project(self: Self, image: Image(T)) ![]f64 {
            if (self.num_components == 0) return error.NotFitted;
            if (image.rows * image.cols * (if (T == u8) 1 else 3) != self.num_features) {
                return error.DimensionMismatch;
            }
            
            // Convert image to feature vector
            var feature_vec = try imageToVector(T, self.allocator, image);
            defer self.allocator.free(feature_vec);
            
            // Center the data
            for (0..self.num_features) |i| {
                feature_vec[i] -= self.mean[i];
            }
            
            // Project onto components
            var coefficients = try self.allocator.alloc(f64, self.num_components);
            for (0..self.num_components) |i| {
                var sum: f64 = 0;
                for (0..self.num_features) |j| {
                    sum += feature_vec[j] * self.components.at(j, i).*;
                }
                coefficients[i] = sum;
            }
            
            return coefficients;
        }
        
        /// Reconstruct an image from PCA coefficients.
        pub fn reconstruct(self: Self, coefficients: []const f64, rows: usize, cols: usize) !Image(T) {
            if (self.num_components == 0) return error.NotFitted;
            if (coefficients.len != self.num_components) return error.InvalidCoefficients;
            if (rows * cols * (if (T == u8) 1 else 3) != self.num_features) {
                return error.DimensionMismatch;
            }
            
            // Reconstruct feature vector
            var feature_vec = try self.allocator.alloc(f64, self.num_features);
            defer self.allocator.free(feature_vec);
            
            // Initialize with mean
            @memcpy(feature_vec, self.mean);
            
            // Add weighted components
            for (0..self.num_components) |i| {
                for (0..self.num_features) |j| {
                    feature_vec[j] += coefficients[i] * self.components.at(j, i).*;
                }
            }
            
            // Convert back to image
            return vectorToImage(T, self.allocator, feature_vec, rows, cols, self.is_grayscale);
        }
        
        /// Project multiple images efficiently as a batch.
        pub fn projectBatch(self: Self, images: []const Image(T)) !Matrix(f64) {
            if (self.num_components == 0) return error.NotFitted;
            
            var projections = try Matrix(f64).init(self.allocator, images.len, self.num_components);
            errdefer projections.deinit();
            
            for (images, 0..) |image, i| {
                const coeffs = try self.project(image);
                defer self.allocator.free(coeffs);
                
                for (0..self.num_components) |j| {
                    projections.at(i, j).* = coeffs[j];
                }
            }
            
            return projections;
        }
        
        /// Get the proportion of variance explained by each component.
        pub fn explainedVarianceRatio(self: Self) ![]f64 {
            if (self.num_components == 0) return error.NotFitted;
            
            var ratios = try self.allocator.alloc(f64, self.num_components);
            
            var total_variance: f64 = 0;
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
        pub fn cumulativeVarianceRatio(self: Self) ![]f64 {
            const ratios = try self.explainedVarianceRatio();
            
            var cumulative: f64 = 0;
            for (ratios) |*ratio| {
                cumulative += ratio.*;
                ratio.* = cumulative;
            }
            
            return ratios;
        }
        
        // Private helper methods
        
        fn computeComponentsFromCovariance(self: *Self, cov_matrix: *Matrix(f64), num_components: usize) !void {
            const n = cov_matrix.rows;
            
            // Convert to static matrix for SVD - limit size to avoid stack overflow
            if (n > 64) return error.DimensionTooLarge;
            
            var cov_static: SMatrix(f64, 64, 64) = std.mem.zeroes(SMatrix(f64, 64, 64));
            for (0..n) |i| {
                for (0..n) |j| {
                    cov_static.items[i][j] = cov_matrix.at(i, j).*;
                }
            }
            
            // Perform SVD - use compile-time maximum dimension
            const result = svd(f64, 64, 64, cov_static, .{
                .with_u = true,
                .with_v = false,
                .mode = .skinny_u,
            });
            
            if (result.converged != 0) return error.SvdFailed;
            
            // Extract components and eigenvalues
            self.num_components = num_components;
            self.eigenvalues = try self.allocator.alloc(f64, num_components);
            self.components = try Matrix(f64).init(self.allocator, self.num_features, num_components);
            
            for (0..num_components) |i| {
                if (i < n) {
                    self.eigenvalues[i] = result.s.items[i][0];
                    for (0..self.num_features) |j| {
                        if (j < n) {
                            self.components.at(j, i).* = result.u.items[j][i];
                        } else {
                            self.components.at(j, i).* = 0;
                        }
                    }
                } else {
                    self.eigenvalues[i] = 0;
                    for (0..self.num_features) |j| {
                        self.components.at(j, i).* = 0;
                    }
                }
            }
        }
        
        fn computeComponentsFromGram(self: *Self, data_matrix: *Matrix(f64), gram_matrix: *Matrix(f64), num_components: usize) !void {
            const n = gram_matrix.rows;
            
            // Convert to static matrix for SVD - limit size to avoid stack overflow
            if (n > 64) return error.DimensionTooLarge;
            
            var gram_static: SMatrix(f64, 64, 64) = .{};
            for (0..n) |i| {
                for (0..n) |j| {
                    gram_static.items[i][j] = gram_matrix.at(i, j).*;
                }
            }
            
            // Perform SVD on Gram matrix - use compile-time maximum dimension
            const result = svd(f64, 64, 64, gram_static, .{
                .with_u = true,
                .with_v = false,
                .mode = .skinny_u,
            });
            
            if (result.converged != 0) return error.SvdFailed;
            
            // Extract components by projecting data onto eigenvectors
            self.num_components = num_components;
            self.eigenvalues = try self.allocator.alloc(f64, num_components);
            self.components = try Matrix(f64).init(self.allocator, self.num_features, num_components);
            
            // V = X^T * U * S^(-1)
            // Ensure we don't exceed the actual number of eigenvalues/eigenvectors computed
            const actual_components = @min(num_components, n);
            for (0..actual_components) |i| {
                const eigenval = result.s.items[i][0];
                self.eigenvalues[i] = eigenval;
                
                if (eigenval > 1e-10) {
                    // Compute component as X^T * u_i / sqrt(eigenval)
                    for (0..self.num_features) |j| {
                        var sum: f64 = 0;
                        for (0..n) |k| {
                            sum += data_matrix.at(k, j).* * result.u.items[k][i];
                        }
                        self.components.at(j, i).* = sum / @sqrt(eigenval * @as(f64, @floatFromInt(n)));
                    }
                } else {
                    // Zero eigenvalue, set component to zero
                    for (0..self.num_features) |j| {
                        self.components.at(j, i).* = 0;
                    }
                }
            }
            
            // Fill remaining components with zeros if num_components > actual_components
            for (actual_components..num_components) |i| {
                self.eigenvalues[i] = 0;
                for (0..self.num_features) |j| {
                    self.components.at(j, i).* = 0;
                }
            }
        }
    };
}

// Helper functions

/// Convert images to data matrix (n_samples × n_features)
fn imagesToMatrix(comptime T: type, allocator: Allocator, images: []const Image(T), num_features: usize) !Matrix(f64) {
    var matrix = try Matrix(f64).init(allocator, images.len, num_features);
    
    for (images, 0..) |image, i| {
        const vec = try imageToVector(T, allocator, image);
        defer allocator.free(vec);
        
        for (vec, 0..) |val, j| {
            matrix.at(i, j).* = val;
        }
    }
    
    return matrix;
}

/// Convert a single image to feature vector
fn imageToVector(comptime T: type, allocator: Allocator, image: Image(T)) ![]f64 {
    const channels: usize = if (T == u8) 1 else 3;
    const num_features = image.rows * image.cols * channels;
    
    var vec = try allocator.alloc(f64, num_features);
    
    if (T == u8) {
        // Grayscale
        for (image.data, 0..) |pixel, i| {
            vec[i] = @as(f64, @floatFromInt(pixel)) / 255.0;
        }
    } else {
        // RGB/RGBA
        var idx: usize = 0;
        for (image.data) |pixel| {
            vec[idx] = @as(f64, @floatFromInt(pixel.r)) / 255.0;
            vec[idx + 1] = @as(f64, @floatFromInt(pixel.g)) / 255.0;
            vec[idx + 2] = @as(f64, @floatFromInt(pixel.b)) / 255.0;
            idx += 3;
        }
    }
    
    return vec;
}

/// Convert feature vector back to image
fn vectorToImage(comptime T: type, allocator: Allocator, vec: []const f64, rows: usize, cols: usize, is_grayscale: bool) !Image(T) {
    var image = try Image(T).initAlloc(allocator, rows, cols);
    
    if (T == u8) {
        // Grayscale
        for (vec, 0..) |val, i| {
            const pixel_val = std.math.clamp(val * 255.0, 0, 255);
            image.data[i] = @intFromFloat(@round(pixel_val));
        }
    } else {
        // RGB/RGBA
        var idx: usize = 0;
        for (image.data) |*pixel| {
            if (is_grayscale) {
                // Reconstruct grayscale as RGB
                const gray_val = std.math.clamp(vec[idx] * 255.0, 0, 255);
                const gray = @as(u8, @intFromFloat(@round(gray_val)));
                pixel.r = gray;
                pixel.g = gray;
                pixel.b = gray;
                idx += 1;
            } else {
                const r = std.math.clamp(vec[idx] * 255.0, 0, 255);
                const g = std.math.clamp(vec[idx + 1] * 255.0, 0, 255);
                const b = std.math.clamp(vec[idx + 2] * 255.0, 0, 255);
                pixel.r = @intFromFloat(@round(r));
                pixel.g = @intFromFloat(@round(g));
                pixel.b = @intFromFloat(@round(b));
                idx += 3;
            }
        }
    }
    
    return image;
}

/// Compute mean of each feature
fn computeMean(allocator: Allocator, matrix: *Matrix(f64)) ![]f64 {
    const n_samples = matrix.rows;
    const n_features = matrix.cols;
    
    var mean = try allocator.alloc(f64, n_features);
    @memset(mean, 0);
    
    for (0..n_samples) |i| {
        for (0..n_features) |j| {
            mean[j] += matrix.at(i, j).*;
        }
    }
    
    const n: f64 = @floatFromInt(n_samples);
    for (mean) |*m| {
        m.* /= n;
    }
    
    return mean;
}

/// Center data by subtracting mean
fn centerData(matrix: *Matrix(f64), mean: []const f64) void {
    for (0..matrix.rows) |i| {
        for (0..matrix.cols) |j| {
            matrix.at(i, j).* -= mean[j];
        }
    }
}

/// Compute covariance matrix (X^T * X) / (n-1)
fn computeCovarianceMatrix(allocator: Allocator, data: *Matrix(f64)) !Matrix(f64) {
    const n_samples = data.rows;
    const n_features = data.cols;
    
    // Manual implementation to avoid OpsBuilder issues
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
    
    // Manual implementation to avoid OpsBuilder issues
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

/// Detect if RGB/RGBA images are actually grayscale
fn detectGrayscale(comptime T: type, images: []const Image(T)) bool {
    for (images) |image| {
        for (image.data) |pixel| {
            if (pixel.r != pixel.g or pixel.g != pixel.b) {
                return false;
            }
        }
    }
    return true;
}

// Tests

test "PCA initialization and cleanup" {
    const allocator = std.testing.allocator;
    
    var pca = PrincipalComponentAnalysis(Rgb).init(allocator);
    defer pca.deinit();
    
    try std.testing.expectEqual(pca.num_components, 0);
    try std.testing.expectEqual(pca.num_features, 0);
}

test "PCA matrix creation" {
    const allocator = std.testing.allocator;
    
    // Create simple test images with different patterns
    var images: [2]Image(u8) = undefined;
    
    // Image 0: all white
    images[0] = try Image(u8).initAlloc(allocator, 2, 2);
    @memset(images[0].data, 128);
    
    // Image 1: all black
    images[1] = try Image(u8).initAlloc(allocator, 2, 2);
    @memset(images[1].data, 64);
    
    defer for (&images) |*img| {
        img.deinit(allocator);
    };
    
    // Test matrix creation
    var data_matrix = try imagesToMatrix(u8, allocator, images[0..], 4);
    defer data_matrix.deinit();
    
    try std.testing.expectEqual(data_matrix.rows, 2);
    try std.testing.expectEqual(data_matrix.cols, 4);
    
    // Test mean computation
    const mean = try computeMean(allocator, &data_matrix);
    defer allocator.free(mean);
    
    try std.testing.expectEqual(mean.len, 4);
}

test "PCA covariance matrix" {
    const allocator = std.testing.allocator;
    
    // Create simple test images with different patterns
    var images: [2]Image(u8) = undefined;
    
    // Image 0: all white
    images[0] = try Image(u8).initAlloc(allocator, 2, 2);
    @memset(images[0].data, 200);
    
    // Image 1: all black
    images[1] = try Image(u8).initAlloc(allocator, 2, 2);
    @memset(images[1].data, 50);
    
    defer for (&images) |*img| {
        img.deinit(allocator);
    };
    
    // Test covariance matrix computation
    var data_matrix = try imagesToMatrix(u8, allocator, images[0..], 4);
    defer data_matrix.deinit();
    
    // Compute mean and center
    const mean = try computeMean(allocator, &data_matrix);
    defer allocator.free(mean);
    
    centerData(&data_matrix, mean);
    
    var cov_matrix = try computeCovarianceMatrix(allocator, &data_matrix);
    defer cov_matrix.deinit();
    
    try std.testing.expectEqual(cov_matrix.rows, 4);
    try std.testing.expectEqual(cov_matrix.cols, 4);
}

test "PCA explained variance" {
    const allocator = std.testing.allocator;
    
    // Create test images with clear variance structure
    var images: [4]Image(u8) = undefined;
    for (&images, 0..) |*img, i| {
        img.* = try Image(u8).initAlloc(allocator, 2, 2);
        // Create images with clear differences to ensure variance structure
        const base_val = @as(u8, @intCast(i * 60));
        for (img.data) |*pixel| {
            pixel.* = base_val;
        }
    }
    defer for (&images) |*img| {
        img.deinit(allocator);
    };
    
    var pca = PrincipalComponentAnalysis(u8).init(allocator);
    defer pca.deinit();
    
    const image_slice = images[0..];
    try pca.fit(image_slice, null);
    
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
    
    try std.testing.expect(cumulative[cumulative.len - 1] <= 1.0001); // Allow small numerical error
}

test "PCA batch projection" {
    const allocator = std.testing.allocator;
    
    // Create test images
    var images: [3]Image(Rgb) = undefined;
    for (&images, 0..) |*img, i| {
        img.* = try Image(Rgb).initAlloc(allocator, 2, 2);
        for (img.data) |*pixel| {
            pixel.* = Rgb{
                .r = @intCast(i * 50),
                .g = @intCast(i * 60),
                .b = @intCast(i * 70),
            };
        }
    }
    defer for (&images) |*img| {
        img.deinit(allocator);
    };
    
    var pca = PrincipalComponentAnalysis(Rgb).init(allocator);
    defer pca.deinit();
    
    const image_slice = images[0..];
    try pca.fit(image_slice, 2);
    
    // Test batch projection
    var projections = try pca.projectBatch(image_slice);
    defer projections.deinit();
    
    try std.testing.expectEqual(projections.rows, 3);
    try std.testing.expectEqual(projections.cols, 2);
}