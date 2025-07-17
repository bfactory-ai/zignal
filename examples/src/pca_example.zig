const std = @import("std");
const builtin = @import("builtin");

const zignal = @import("zignal");
const Image = zignal.Image;
const Rgba = zignal.Rgba;
const Canvas = zignal.Canvas;
const PCA = zignal.PrincipalComponentAnalysis;
const Point2d = zignal.Point2d;
const makePoint = zignal.makePoint;

pub const std_options: std.Options = .{
    .logFn = if (builtin.cpu.arch.isWasm()) @import("js.zig").logFn else std.log.defaultLog,
    .log_level = std.log.default_level,
};

pub fn panic(msg: []const u8, st: ?*std.builtin.StackTrace, addr: ?usize) noreturn {
    _ = st;
    _ = addr;
    std.log.err("panic: {s}", .{msg});
    @trap();
}

/// Bounds structure for point clouds
const Bounds = struct { min: @Vector(2, f64), max: @Vector(2, f64) };

/// Generate a 2D point cloud with a clear dominant direction
fn generatePointCloud(allocator: std.mem.Allocator, num_points: usize) ![]@Vector(2, f64) {
    const points = try allocator.alloc(@Vector(2, f64), num_points);

    // Use a deterministic random number generator for reproducible results
    var prng = std.Random.DefaultPrng.init(42);
    var rand = prng.random();

    // Generate points along a dominant diagonal direction with some noise
    // Main direction: (1, 0.6) - creates an elongated cloud
    const main_direction = @Vector(2, f64){ 1.0, 0.6 };
    const noise_strength = 0.3;

    for (points, 0..) |*point, i| {
        // Parameter along main direction
        const t = (rand.float(f64) - 0.5) * 4.0; // Range: -2 to 2

        // Point along main direction
        const main_point = main_direction * @as(@Vector(2, f64), @splat(t));

        // Add perpendicular noise
        const noise_x = (rand.float(f64) - 0.5) * noise_strength;
        const noise_y = (rand.float(f64) - 0.5) * noise_strength;
        const noise = @Vector(2, f64){ noise_x, noise_y };

        point.* = main_point + noise;

        // Add some outliers for more realistic data
        if (i < num_points / 20) {
            point.*[0] += (rand.float(f64) - 0.5) * 2.0;
            point.*[1] += (rand.float(f64) - 0.5) * 2.0;
        }
    }

    return points;
}

/// Convert world coordinates to canvas coordinates
/// Note: Y axis is flipped since image coordinates have Y=0 at top
fn worldToCanvas(world_point: @Vector(2, f64), canvas_size: f32, world_bounds: Bounds) Point2d(f32) {
    const world_size = world_bounds.max - world_bounds.min;
    const margin = canvas_size * 0.1; // 10% margin
    const draw_size = canvas_size - 2 * margin;

    // Normalize to [0, 1]
    const normalized = (world_point - world_bounds.min) / world_size;

    // Scale to canvas with margin
    // Flip Y coordinate: canvas Y=0 is at top, but we want mathematical Y=0 at bottom
    const canvas_x = normalized[0] * draw_size + margin;
    const canvas_y = (1.0 - normalized[1]) * draw_size + margin; // Flip Y axis

    return Point2d(f32).init2d(@floatCast(canvas_x), @floatCast(canvas_y));
}

/// Find bounding box of point cloud
fn findBounds(points: []const @Vector(2, f64)) Bounds {
    if (points.len == 0) return .{ .min = @splat(0), .max = @splat(1) };

    var min_point = points[0];
    var max_point = points[0];

    for (points[1..]) |point| {
        min_point = @min(min_point, point);
        max_point = @max(max_point, point);
    }

    // Add some padding
    const padding = (max_point - min_point) * @as(@Vector(2, f64), @splat(0.1));
    min_point -= padding;
    max_point += padding;

    return .{ .min = min_point, .max = max_point };
}

/// Draw points on canvas
fn drawPoints(canvas: Canvas(Rgba), points: []const @Vector(2, f64), bounds: Bounds, color: Rgba, radius: f32) void {
    const canvas_size = @as(f32, @floatFromInt(canvas.image.cols));

    for (points) |pt| {
        const canvas_point = worldToCanvas(pt, canvas_size, bounds);
        canvas.drawCircle(canvas_point, radius, color, 1, .soft);
    }
}

/// Draw PCA axes on canvas
fn drawPCAAxes(canvas: Canvas(Rgba), pca: PCA(f64, 2), bounds: Bounds) !void {
    const canvas_size = @as(f32, @floatFromInt(canvas.image.cols));
    const mean_vec = pca.getMean();
    const mean_canvas = worldToCanvas(mean_vec, canvas_size, bounds);

    // Draw mean point
    canvas.drawCircle(mean_canvas, 8, Rgba{ .r = 255, .g = 255, .b = 0, .a = 255 }, 2, .soft);

    // Get principal components (eigenvectors)
    const scale = 1.5; // Scale factor for drawing axes

    for (0..pca.num_components) |i| {
        var component: @Vector(2, f64) = undefined;
        for (0..2) |j| {
            component[j] = pca.components.at(j, i).*;
        }

        // Scale component for visualization
        const axis_end = mean_vec + component * @as(@Vector(2, f64), @splat(scale));
        const axis_start = mean_vec - component * @as(@Vector(2, f64), @splat(scale));

        const start_canvas = worldToCanvas(axis_start, canvas_size, bounds);
        const end_canvas = worldToCanvas(axis_end, canvas_size, bounds);

        // Use different colors for different components
        const axis_color = if (i == 0)
            Rgba{ .r = 255, .g = 0, .b = 0, .a = 255 } // Red for first PC
        else
            Rgba{ .r = 0, .g = 255, .b = 0, .a = 255 }; // Green for second PC

        canvas.drawLine(start_canvas, end_canvas, axis_color, 3, .soft);

        // Draw arrowhead at end
        const arrow_size = 10.0;
        const dir_x = end_canvas.x() - start_canvas.x();
        const dir_y = end_canvas.y() - start_canvas.y();
        const length = @sqrt(dir_x * dir_x + dir_y * dir_y);

        if (length > 0) {
            const norm_dir_x = dir_x / length;
            const norm_dir_y = dir_y / length;

            const arrow1 = Point2d(f32).init2d(end_canvas.x() - arrow_size * (norm_dir_x + norm_dir_y * 0.5), end_canvas.y() - arrow_size * (norm_dir_y - norm_dir_x * 0.5));
            const arrow2 = Point2d(f32).init2d(end_canvas.x() - arrow_size * (norm_dir_x - norm_dir_y * 0.5), end_canvas.y() - arrow_size * (norm_dir_y + norm_dir_x * 0.5));

            canvas.drawLine(end_canvas, arrow1, axis_color, 2, .soft);
            canvas.drawLine(end_canvas, arrow2, axis_color, 2, .soft);
        }
    }
}

/// Project points using PCA and create new aligned dataset
fn projectAndAlign(allocator: std.mem.Allocator, pca: PCA(f64, 2), original_points: []const @Vector(2, f64)) ![]@Vector(2, f64) {
    var aligned_points = try allocator.alloc(@Vector(2, f64), original_points.len);

    for (original_points, 0..) |point, i| {
        // Project to PCA space
        const coeffs = try pca.project(point);
        defer allocator.free(coeffs);

        // For visualization, we'll create new 2D coordinates where:
        // X = first principal component coefficient
        // Y = second principal component coefficient
        aligned_points[i] = @Vector(2, f64){ coeffs[0], coeffs[1] };
    }

    return aligned_points;
}

/// Main PCA demonstration function
pub fn demonstratePCA(allocator: std.mem.Allocator, canvas_size: u32) !struct { original_image: Image(Rgba), aligned_image: Image(Rgba), variance_ratios: []f64 } {
    const num_points = 200;

    // 1. Generate point cloud with dominant direction
    const original_points = try generatePointCloud(allocator, num_points);
    defer allocator.free(original_points);

    // 2. Create original visualization
    const original_image = try Image(Rgba).initAlloc(allocator, canvas_size, canvas_size);
    const original_canvas = Canvas(Rgba).init(allocator, original_image);
    original_canvas.fill(Rgba{ .r = 20, .g = 20, .b = 40, .a = 255 }); // Dark blue background

    const original_bounds = findBounds(original_points);

    // Draw original points
    drawPoints(original_canvas, original_points, original_bounds, Rgba{ .r = 100, .g = 150, .b = 255, .a = 255 }, 3.0); // Light blue points

    // 3. Compute PCA
    var pca = PCA(f64, 2).init(allocator);
    defer pca.deinit();

    try pca.fit(original_points, null); // Keep all components

    // Draw PCA axes on original image
    try drawPCAAxes(original_canvas, pca, original_bounds);

    // 4. Project points to PCA space and create aligned visualization
    const aligned_points = try projectAndAlign(allocator, pca, original_points);
    defer allocator.free(aligned_points);

    const aligned_image = try Image(Rgba).initAlloc(allocator, canvas_size, canvas_size);
    const aligned_canvas = Canvas(Rgba).init(allocator, aligned_image);
    aligned_canvas.fill(Rgba{ .r = 40, .g = 20, .b = 20, .a = 255 }); // Dark red background

    const aligned_bounds = findBounds(aligned_points);

    // Draw aligned points
    drawPoints(aligned_canvas, aligned_points, aligned_bounds, Rgba{ .r = 255, .g = 150, .b = 100, .a = 255 }, 3.0); // Light orange points

    // Draw coordinate axes in aligned space
    const zero_point = @Vector(2, f64){ 0.0, 0.0 };
    _ = zero_point; // Mark as used to avoid unused variable warning

    // X-axis (red) - corresponds to first principal component
    const x_axis_end = worldToCanvas(@Vector(2, f64){ aligned_bounds.max[0] * 0.8, 0.0 }, @floatFromInt(canvas_size), aligned_bounds);
    const x_axis_start = worldToCanvas(@Vector(2, f64){ aligned_bounds.min[0] * 0.8, 0.0 }, @floatFromInt(canvas_size), aligned_bounds);
    aligned_canvas.drawLine(x_axis_start, x_axis_end, Rgba{ .r = 255, .g = 0, .b = 0, .a = 255 }, 2, .soft);

    // Y-axis (green) - corresponds to second principal component
    const y_axis_end = worldToCanvas(@Vector(2, f64){ 0.0, aligned_bounds.max[1] * 0.8 }, @floatFromInt(canvas_size), aligned_bounds);
    const y_axis_start = worldToCanvas(@Vector(2, f64){ 0.0, aligned_bounds.min[1] * 0.8 }, @floatFromInt(canvas_size), aligned_bounds);
    aligned_canvas.drawLine(y_axis_start, y_axis_end, Rgba{ .r = 0, .g = 255, .b = 0, .a = 255 }, 2, .soft);

    // 5. Get variance explained ratios
    const variance_ratios = try pca.explainedVarianceRatio();

    return .{ .original_image = original_image, .aligned_image = aligned_image, .variance_ratios = variance_ratios };
}

// Export for WASM
export fn run_pca_demo() void {
    if (builtin.cpu.arch.isWasm()) {
        pcaDemo() catch |err| {
            std.log.err("PCA demo failed: {}", .{err});
        };
    }
}

pub fn pcaDemo() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const canvas_size = 400;

    // Run PCA demonstration
    var result = try demonstratePCA(allocator, canvas_size);
    defer result.original_image.deinit(allocator);
    defer result.aligned_image.deinit(allocator);
    defer allocator.free(result.variance_ratios);

    // Log results
    std.log.info("PCA Analysis Results:", .{});
    std.log.info("First PC explains {d:.1}% of variance", .{result.variance_ratios[0] * 100});
    std.log.info("Second PC explains {d:.1}% of variance", .{result.variance_ratios[1] * 100});
    std.log.info("Total variance explained: {d:.1}%", .{(result.variance_ratios[0] + result.variance_ratios[1]) * 100});

    // Save images if not running in WASM
    if (!builtin.cpu.arch.isWasm()) {
        try zignal.savePng(Rgba, allocator, result.original_image, "pca_original.png");
        try zignal.savePng(Rgba, allocator, result.aligned_image, "pca_aligned.png");
        std.log.info("Saved pca_original.png and pca_aligned.png", .{});
    }
}

pub fn main() !void {
    try pcaDemo();
}
