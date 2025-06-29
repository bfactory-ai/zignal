//! This module provides a Canvas for drawing various shapes and lines on images.

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;
const expectEqualStrings = testing.expectEqualStrings;

const as = @import("meta.zig").as;
const convert = @import("color.zig").convert;
const Hsv = @import("color.zig").Hsv;
const Image = @import("image.zig").Image;
const isColor = @import("color.zig").isColor;
const Point2d = @import("point.zig").Point2d;
const Rectangle = @import("geometry.zig").Rectangle;
const Rgba = @import("color.zig").Rgba;

/// Rendering quality mode for drawing operations
pub const FillMode = enum {
    /// Fast rendering - hard edges, maximum performance
    fast,
    /// Soft rendering - antialiased edges, better quality
    soft,
};

/// A drawing context for an image, providing methods to draw shapes and lines.
pub fn Canvas(comptime T: type) type {
    return struct {
        image: Image(T),
        allocator: std.mem.Allocator,

        const Self = @This();

        // Drawing-related constants
        /// Maximum number of line segments when tessellating Bézier curves for line drawing
        const bezier_max_segments_count = 200;
        /// Maximum number of line segments when tessellating spline polygons
        const spline_max_segments_count = 50;
        /// Minimum number of line segments for spline curves to ensure reasonable quality
        const spline_min_segments_count = 4;
        /// Minimum number of line segments for quadratic Bézier curves
        const quadratic_min_segments_count = 3;
        /// Target pixels per segment for smooth/antialiased rendering (higher quality, more segments)
        const pixels_per_segment_soft = 1.5;
        /// Target pixels per segment for solid/fast rendering (lower quality, fewer segments)
        const pixels_per_segment_fast = 3.0;
        /// Target pixels per segment specifically for quadratic Bézier curves
        const pixels_per_segment_quadratic = 2.0;
        /// Offset for antialiasing edge calculations (0.5 = pixel center alignment)
        const antialias_edge_offset = 0.5;
        /// Threshold for detecting horizontal/vertical lines in line drawing algorithms
        const horizontal_vertical_threshold = 0.001;
        /// Stack buffer size for polygon intersection calculations (avoids allocation for most cases)
        const polygon_intersection_stack_buffer_size = 64;
        /// Stack buffer size for spline polygon tessellation (avoids allocation for typical polygons)
        const spline_polygon_stack_buffer_size = 400;

        /// Creates a drawing canvas from an image, with an allocator for operations that need it.
        pub fn init(image: Image(T), allocator: std.mem.Allocator) Self {
            return .{ .image = image, .allocator = allocator };
        }

        /// Draws a colored straight line of a custom width between p1 and p2 on an image.
        /// Width=1 lines use fast Bresenham algorithm with no caps for precise pixel placement.
        /// Width>1 lines are rendered as rectangles with rounded caps for smooth appearance.
        /// Use FillMode.soft for anti-aliased lines or FillMode.fast for fast non-anti-aliased lines.
        /// If the `color` is of Rgba type, it alpha-blends it onto the image.
        pub fn drawLine(self: Self, p1: Point2d(f32), p2: Point2d(f32), color: anytype, width: usize, mode: FillMode) void {
            comptime assert(isColor(@TypeOf(color)));
            if (width == 0) return;

            switch (mode) {
                .fast => self.drawLineFast(p1, p2, width, color),
                .soft => self.drawLineSoft(p1, p2, width, color),
            }
        }

        /// Internal function for drawing solid (non-anti-aliased) lines.
        /// Uses Bresenham's algorithm for 1px lines and polygon-based approach for thick lines.
        fn drawLineFast(self: Self, p1: Point2d(f32), p2: Point2d(f32), width: usize, color: anytype) void {
            comptime assert(isColor(@TypeOf(color)));
            if (width == 0) return;
            if (width == 1) {
                // Use Bresenham's algorithm for 1px lines - fast and precise
                self.drawLineBresenham(p1, p2, color);
            } else {
                // Use polygon-based approach for thick lines
                self.drawLinePolygon(p1, p2, width, color);
            }
        }
        /// Internal function for drawing smooth (anti-aliased) lines.
        /// Uses Wu's algorithm for 1px lines (optimal antialiasing) and distance-based
        /// antialiasing for thick lines (better quality than polygon approach).
        fn drawLineSoft(self: Self, p1: Point2d(f32), p2: Point2d(f32), width: usize, color: anytype) void {
            if (width == 0) return;
            if (width == 1) {
                // Use Wu's algorithm for 1px lines - optimal antialiasing and performance
                self.drawLineWu(p1, p2, color);
            } else {
                // Use distance-based antialiasing for thick lines - better quality
                self.drawLineDistance(p1, p2, width, color);
            }
        }

        /// Bresenham's line algorithm for 1-pixel width solid lines.
        /// Fast integer-only algorithm that draws precise pixel-perfect lines.
        /// No antialiasing - draws hard edges for maximum performance.
        fn drawLineBresenham(self: Self, p1: Point2d(f32), p2: Point2d(f32), color: anytype) void {
            var x1: i32 = @intFromFloat(p1.x);
            var y1: i32 = @intFromFloat(p1.y);
            const x2: i32 = @intFromFloat(p2.x);
            const y2: i32 = @intFromFloat(p2.y);

            const dx: i32 = @intCast(@abs(x2 - x1));
            const dy: i32 = @intCast(@abs(y2 - y1));
            const sx: i32 = if (x1 < x2) 1 else -1;
            const sy: i32 = if (y1 < y2) 1 else -1;
            var err = dx - dy;

            while (true) {
                self.assignPixel(x1, y1, color, 1.0);

                if (x1 == x2 and y1 == y2) break;

                const e2 = 2 * err;
                if (e2 > -dy) {
                    err -= dy;
                    x1 += sx;
                }
                if (e2 < dx) {
                    err += dx;
                    y1 += sy;
                }
            }
        }

        /// Wu's anti-aliasing algorithm for 1-pixel width lines.
        /// Provides optimal antialiasing quality and performance for thin lines.
        fn drawLineWu(self: Self, p1: Point2d(f32), p2: Point2d(f32), color: anytype) void {
            const Float = @TypeOf(p1.x);
            const c2 = convert(Rgba, color);

            // Wu's algorithm for 1px lines
            var x1 = p1.x;
            var y1 = p1.y;
            var x2 = p2.x;
            var y2 = p2.y;

            const steep = @abs(y2 - y1) > @abs(x2 - x1);
            if (steep) {
                std.mem.swap(Float, &x1, &y1);
                std.mem.swap(Float, &x2, &y2);
            }
            if (x1 > x2) {
                std.mem.swap(Float, &x1, &x2);
                std.mem.swap(Float, &y1, &y2);
            }

            const dx = x2 - x1;
            const dy = y2 - y1;
            const gradient = if (dx == 0) 1.0 else dy / dx;

            // Handle first endpoint
            var x_end = @round(x1);
            var y_end = y1 + gradient * (x_end - x1);
            var x_gap = rfpart(x1 + 0.5);
            const x_px1: i32 = @intFromFloat(x_end);
            const y_px1: i32 = @intFromFloat(y_end);

            if (steep) {
                self.assignPixel(y_px1, x_px1, c2, rfpart(y_end) * x_gap);
                self.assignPixel(y_px1 + 1, x_px1, c2, fpart(y_end) * x_gap);
            } else {
                self.assignPixel(x_px1, y_px1, c2, rfpart(y_end) * x_gap);
                self.assignPixel(x_px1, y_px1 + 1, c2, fpart(y_end) * x_gap);
            }
            var intery = y_end + gradient;

            // Handle second endpoint
            x_end = @round(x2);
            y_end = y2 + gradient * (x_end - x2);
            x_gap = fpart(x2 + 0.5);
            const x_px2: i32 = @intFromFloat(x_end);
            const y_px2: i32 = @intFromFloat(y_end);

            if (steep) {
                self.assignPixel(y_px2, x_px2, c2, rfpart(y_end) * x_gap);
                self.assignPixel(y_px2 + 1, x_px2, c2, fpart(y_end) * x_gap);
            } else {
                self.assignPixel(x_px2, y_px2, c2, rfpart(y_end) * x_gap);
                self.assignPixel(x_px2, y_px2 + 1, c2, fpart(y_end) * x_gap);
            }

            // Main loop
            var x = x_px1 + 1;
            while (x < x_px2) : (x += 1) {
                if (steep) {
                    self.assignPixel(@as(i32, @intFromFloat(intery)), x, c2, rfpart(intery));
                    self.assignPixel(@as(i32, @intFromFloat(intery)) + 1, x, c2, fpart(intery));
                } else {
                    self.assignPixel(x, @as(i32, @intFromFloat(intery)), c2, rfpart(intery));
                    self.assignPixel(x, @as(i32, @intFromFloat(intery)) + 1, c2, fpart(intery));
                }
                intery += gradient;
            }
        }

        /// Polygon-based thick line drawing for solid (non-anti-aliased) lines.
        /// Creates a filled rectangle with circular end caps for thick line appearance.
        fn drawLinePolygon(self: Self, p1: Point2d(f32), p2: Point2d(f32), width: usize, color: anytype) void {
            const solid_color = convert(T, color);

            // For thick lines, draw as a filled rectangle
            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            const line_length = @sqrt(dx * dx + dy * dy);

            if (line_length == 0) {
                // Single point - draw a filled circle
                const half_width: f32 = @as(f32, @floatFromInt(width)) / 2.0;
                self.fillCircle(p1, half_width, color, .fast);
                return;
            }

            // Calculate perpendicular vector for thick line
            const half_width: f32 = @as(f32, @floatFromInt(width)) / 2.0;
            const perp_x = -dy / line_length * half_width;
            const perp_y = dx / line_length * half_width;

            // Create rectangle corners
            const corners = [_]Point2d(f32){
                .{ .x = p1.x - perp_x, .y = p1.y - perp_y },
                .{ .x = p1.x + perp_x, .y = p1.y + perp_y },
                .{ .x = p2.x + perp_x, .y = p2.y + perp_y },
                .{ .x = p2.x - perp_x, .y = p2.y - perp_y },
            };

            // Fill rectangle using scanline algorithm (no anti-aliasing)
            self.fillPolygon(&corners, solid_color, .fast) catch return;

            // Add rounded caps using solid circles
            self.fillCircle(p1, half_width, color, .fast);
            self.fillCircle(p2, half_width, color, .fast);
        }

        /// Distance-based anti-aliased line drawing for thick lines.
        /// Uses exact distance calculation from each pixel to the line for superior quality.
        fn drawLineDistance(self: Self, p1: Point2d(f32), p2: Point2d(f32), width: usize, color: anytype) void {
            const Float = @TypeOf(p1.x);
            const rows: Float = @floatFromInt(self.image.rows);
            const cols: Float = @floatFromInt(self.image.cols);
            const half_width: Float = @as(Float, @floatFromInt(width)) / 2.0;
            const c2 = convert(Rgba, color);

            // Calculate line direction vector
            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            const line_length = @sqrt(dx * dx + dy * dy);

            if (line_length == 0) {
                // Single point - draw a small circle
                self.fillCircle(p1, half_width, color, .soft);
                return;
            }

            // Special case for perfectly horizontal/vertical lines (faster rendering)
            if (@abs(dx) < horizontal_vertical_threshold) { // Vertical line
                const x1 = @round(p1.x);
                var y1 = @round(p1.y);
                var y2 = @round(p2.y);
                if (y1 > y2) std.mem.swap(Float, &y1, &y2);
                if (x1 < 0 or x1 >= cols) return;
                var y = y1;
                while (y <= y2) : (y += 1) {
                    if (y < 0 or y >= rows) continue;
                    var i = -half_width;
                    while (i <= half_width) : (i += 1) {
                        const px = x1 + i;
                        self.assignPixel(@intFromFloat(px), @intFromFloat(y), c2, 1.0);
                    }
                }
                // Add rounded caps
                self.fillCircle(p1, half_width, color, .soft);
                self.fillCircle(p2, half_width, color, .soft);
                return;
            } else if (@abs(dy) < horizontal_vertical_threshold) { // Horizontal line
                var x1 = @round(p1.x);
                var x2 = @round(p2.x);
                const y1 = @round(p1.y);
                if (x1 > x2) std.mem.swap(Float, &x1, &x2);
                if (y1 < 0 or y1 >= rows) return;
                var x = x1;
                while (x <= x2) : (x += 1) {
                    if (x < 0 or x >= cols) continue;
                    var i = -half_width;
                    while (i <= half_width) : (i += 1) {
                        const py = y1 + i;
                        self.assignPixel(@intFromFloat(x), @intFromFloat(py), c2, 1.0);
                    }
                }
                // Add rounded caps
                self.fillCircle(p1, half_width, color, .soft);
                self.fillCircle(p2, half_width, color, .soft);
                return;
            }

            // For diagonal lines, use optimized distance-based anti-aliasing
            // Calculate tighter bounding box
            const line_min_x = @min(p1.x, p2.x) - half_width;
            const line_max_x = @max(p1.x, p2.x) + half_width;
            const line_min_y = @min(p1.y, p2.y) - half_width;
            const line_max_y = @max(p1.y, p2.y) + half_width;

            const min_x = @max(0, @floor(line_min_x));
            const max_x = @min(cols - 1, @ceil(line_max_x));
            const min_y = @max(0, @floor(line_min_y));
            const max_y = @min(rows - 1, @ceil(line_max_y));

            // Precompute for distance calculation optimization
            const dx_sq = dx * dx;
            const dy_sq = dy * dy;
            const length_sq = dx_sq + dy_sq;
            const inv_length_sq = 1.0 / length_sq;

            // Iterate through pixels in bounding box
            var y: i32 = @as(i32, @intFromFloat(min_y));
            while (y <= @as(i32, @intFromFloat(max_y))) : (y += 1) {
                const py: Float = @floatFromInt(y);
                var x: i32 = @as(i32, @intFromFloat(min_x));
                while (x <= @as(i32, @intFromFloat(max_x))) : (x += 1) {
                    const px: Float = @floatFromInt(x);

                    // Optimized distance calculation
                    const dpx = px - p1.x;
                    const dpy = py - p1.y;
                    const t = @max(0, @min(1, (dpx * dx + dpy * dy) * inv_length_sq));
                    const closest_x = p1.x + t * dx;
                    const closest_y = p1.y + t * dy;
                    const dist_x = px - closest_x;
                    const dist_y = py - closest_y;
                    const dist = @sqrt(dist_x * dist_x + dist_y * dist_y);

                    // Anti-aliased coverage based on distance
                    if (dist <= half_width + antialias_edge_offset) {
                        var alpha: Float = 1.0;
                        if (dist > half_width - antialias_edge_offset) {
                            alpha = (half_width + antialias_edge_offset - dist);
                        }

                        if (alpha > 0) {
                            self.assignPixel(x, y, c2, alpha);
                        }
                    }
                }
            }
        }

        /// Assigns a color to a pixel at the given coordinates with alpha transparency.
        /// Uses optimized direct assignment for opaque colors (alpha >= 1.0) or blends when
        /// transparency is needed. Provides bounds checking and handles coordinate conversion.
        /// The alpha parameter (0.0-1.0) is multiplied with the color's alpha channel.
        pub fn assignPixel(self: Self, x: i32, y: i32, color: anytype, alpha: f32) void {
            if (self.image.atOrNull(y, x)) |pixel| {
                // No-op if color is already Rgba
                var src = convert(Rgba, color);

                // Optimize for full opacity - direct assignment when alpha is 1.0 and source is opaque
                if (alpha >= 1.0 and src.a == 255) {
                    pixel.* = convert(T, src);
                    return;
                }

                var dst = convert(Rgba, pixel.*);

                // Apply additional alpha factor
                if (alpha < 1.0) {
                    src.a = @intFromFloat(@as(f32, @floatFromInt(src.a)) * alpha);
                }

                dst.blend(src);
                pixel.* = convert(T, dst);
            }
        }

        /// Returns the fractional part of a floating-point number.
        /// Used in Wu's anti-aliasing algorithm to calculate pixel coverage.
        /// Example: fpart(3.7) = 0.7, fpart(-2.3) = 0.7
        fn fpart(x: f32) f32 {
            return x - @floor(x);
        }

        /// Returns the reverse fractional part (1 - fractional part).
        /// Used in Wu's anti-aliasing algorithm for complementary pixel coverage.
        /// Example: rfpart(3.7) = 0.3, rfpart(-2.3) = 0.3
        fn rfpart(x: f32) f32 {
            return 1 - fpart(x);
        }

        /// Draws the outline of a polygon on the given image.
        /// The polygon is defined by a sequence of vertices. Lines are drawn between consecutive
        /// vertices, and a final line is drawn from the last vertex to the first to close the shape.
        /// Round joints are added at vertices to ensure smooth connections.
        pub fn drawPolygon(self: Self, polygon: []const Point2d(f32), color: anytype, width: usize, mode: FillMode) void {
            comptime assert(isColor(@TypeOf(color)));
            if (width == 0) return;

            // Draw all line segments
            for (0..polygon.len) |i| {
                self.drawLine(polygon[i], polygon[@mod(i + 1, polygon.len)], color, width, mode);
            }
        }

        /// Draws the outline of a rectangle on the given image.
        pub fn drawRectangle(self: Self, rect: Rectangle(f32), color: anytype, width: usize, mode: FillMode) void {
            comptime assert(isColor(@TypeOf(color)));
            const points: []const Point2d(f32) = &.{
                .{ .x = rect.l, .y = rect.t },
                .{ .x = rect.r, .y = rect.t },
                .{ .x = rect.r, .y = rect.b },
                .{ .x = rect.l, .y = rect.b },
            };
            self.drawPolygon(points, color, width, mode);
        }

        /// Draws the outline of a circle on the given image.
        /// Use FillMode.soft for anti-aliased edges or FillMode.fast for fast aliased edges.
        pub fn drawCircle(self: Self, center: Point2d(f32), radius: f32, color: anytype, width: usize, mode: FillMode) void {
            comptime assert(isColor(@TypeOf(color)));
            if (radius <= 0 or width == 0) return;

            switch (mode) {
                .fast => self.drawCircleFast(center, radius, width, color),
                .soft => self.drawCircleSoft(center, radius, width, color),
            }
        }

        /// Internal function for drawing solid (aliased) circle outlines.
        fn drawCircleFast(self: Self, center: Point2d(f32), radius: f32, width: usize, color: anytype) void {
            if (width == 1) {
                // Use fast Bresenham for 1-pixel width
                const cx = @round(center.x);
                const cy = @round(center.y);
                const r = @round(radius);
                var x: f32 = r;
                var y: f32 = 0;
                var err: f32 = 0;
                while (x >= y) {
                    const points = [_]Point2d(f32){
                        .{ .x = cx + x, .y = cy + y },
                        .{ .x = cx - x, .y = cy + y },
                        .{ .x = cx + x, .y = cy - y },
                        .{ .x = cx - x, .y = cy - y },
                        .{ .x = cx + y, .y = cy + x },
                        .{ .x = cx - y, .y = cy + x },
                        .{ .x = cx + y, .y = cy - x },
                        .{ .x = cx - y, .y = cy - x },
                    };
                    for (points) |p| {
                        self.assignPixel(@intFromFloat(p.x), @intFromFloat(p.y), color, 1.0);
                    }
                    if (err <= 0) {
                        y += 1;
                        err += 2 * y + 1;
                    }
                    if (err > 0) {
                        x -= 1;
                        err -= 2 * x + 1;
                    }
                }
            } else {
                // Use ring filling for thick outlines
                const frows: f32 = @floatFromInt(self.image.rows);
                const fcols: f32 = @floatFromInt(self.image.cols);
                const line_width: f32 = @floatFromInt(width);
                const inner_radius = radius - line_width / 2.0;
                const outer_radius = radius + line_width / 2.0;
                const solid_color = convert(T, color);

                // Calculate bounding box
                const left: usize = @intFromFloat(@round(@max(0, center.x - outer_radius - 1)));
                const top: usize = @intFromFloat(@round(@max(0, center.y - outer_radius - 1)));
                const right: usize = @intFromFloat(@round(@min(fcols, center.x + outer_radius + 1)));
                const bottom: usize = @intFromFloat(@round(@min(frows, center.y + outer_radius + 1)));

                for (top..bottom) |r| {
                    const y = @as(f32, @floatFromInt(r)) - center.y;
                    for (left..right) |c| {
                        const x = @as(f32, @floatFromInt(c)) - center.x;
                        const dist_sq = x * x + y * y;
                        const inner_radius_sq = inner_radius * inner_radius;
                        const outer_radius_sq = outer_radius * outer_radius;

                        if (dist_sq >= inner_radius_sq and dist_sq <= outer_radius_sq) {
                            const pos = r * self.image.cols + c;
                            self.image.data[pos] = solid_color;
                        }
                    }
                }
            }
        }

        /// Internal function for drawing smooth (anti-aliased) circle outlines.
        fn drawCircleSoft(self: Self, center: Point2d(f32), radius: f32, width: usize, color: anytype) void {
            const frows: f32 = @floatFromInt(self.image.rows);
            const fcols: f32 = @floatFromInt(self.image.cols);
            const line_width: f32 = @floatFromInt(width);
            const inner_radius = radius - line_width / 2.0;
            const outer_radius = radius + line_width / 2.0;

            // Calculate bounding box
            const left: usize = @intFromFloat(@round(@max(0, center.x - outer_radius - 1)));
            const top: usize = @intFromFloat(@round(@max(0, center.y - outer_radius - 1)));
            const right: usize = @intFromFloat(@round(@min(fcols, center.x + outer_radius + 1)));
            const bottom: usize = @intFromFloat(@round(@min(frows, center.y + outer_radius + 1)));

            const c2 = convert(Rgba, color);

            for (top..bottom) |r| {
                const y = @as(f32, @floatFromInt(r)) - center.y;
                for (left..right) |c| {
                    const x = @as(f32, @floatFromInt(c)) - center.x;
                    const dist = @sqrt(x * x + y * y);

                    // Only draw if we're in the ring area
                    if (dist >= inner_radius - antialias_edge_offset and dist <= outer_radius + antialias_edge_offset) {
                        var alpha: f32 = 1.0;

                        // Smooth outer edge
                        if (dist > outer_radius - antialias_edge_offset) {
                            alpha = @min(alpha, outer_radius + antialias_edge_offset - dist);
                        }

                        // Smooth inner edge
                        if (dist < inner_radius + antialias_edge_offset) {
                            alpha = @min(alpha, dist - (inner_radius - antialias_edge_offset));
                        }

                        alpha = @max(0, @min(1, alpha));

                        if (alpha > 0) {
                            self.assignPixel(@intCast(c), @intCast(r), c2, alpha);
                        }
                    }
                }
            }
        }

        /// Fills the given polygon on an image using the scanline algorithm.
        /// The polygon is defined by an array of points (vertices).
        /// Use FillMode.fast for hard edges (fastest) or FillMode.soft for antialiased edges.
        pub fn fillPolygon(self: Self, polygon: []const Point2d(f32), color: anytype, mode: FillMode) !void {
            comptime assert(isColor(@TypeOf(color)));
            if (polygon.len < 3) return;

            const rows = self.image.rows;
            const cols = self.image.cols;

            // Find bounding box for optimization
            var min_y = polygon[0].y;
            var max_y = polygon[0].y;
            for (polygon) |p| {
                min_y = @min(min_y, p.y);
                max_y = @max(max_y, p.y);
            }

            const start_y = @max(0, @as(i32, @intFromFloat(@floor(min_y))));
            const end_y = @min(@as(i32, @intCast(rows)) - 1, @as(i32, @intFromFloat(@ceil(max_y))));

            // Use stack buffer for small polygons, fallback to heap for complex ones
            var stack_intersections: [polygon_intersection_stack_buffer_size]f32 = undefined;
            var heap_intersections: ?[]f32 = null;
            defer if (heap_intersections) |h| self.allocator.free(h);

            const c2 = convert(Rgba, color);

            var y = start_y;
            while (y <= end_y) : (y += 1) {
                const fy: f32 = @floatFromInt(y);
                var intersection_count: usize = 0;

                // Count intersections first
                for (0..polygon.len) |i| {
                    const p1 = polygon[i];
                    const p2 = polygon[(i + 1) % polygon.len];

                    if ((p1.y <= fy and p2.y > fy) or (p2.y <= fy and p1.y > fy)) {
                        intersection_count += 1;
                    }
                }

                // Get appropriate buffer
                var intersections: []f32 = undefined;
                if (intersection_count <= polygon_intersection_stack_buffer_size) {
                    intersections = stack_intersections[0..intersection_count];
                } else {
                    // Need heap allocation
                    if (heap_intersections == null or heap_intersections.?.len < intersection_count) {
                        if (heap_intersections) |h| self.allocator.free(h);
                        heap_intersections = try self.allocator.alloc(f32, intersection_count);
                    }
                    intersections = heap_intersections.?[0..intersection_count];
                }

                // Find actual intersections
                var idx: usize = 0;
                for (0..polygon.len) |i| {
                    const p1 = polygon[i];
                    const p2 = polygon[(i + 1) % polygon.len];

                    if ((p1.y <= fy and p2.y > fy) or (p2.y <= fy and p1.y > fy)) {
                        const intersection = p1.x + (fy - p1.y) * (p2.x - p1.x) / (p2.y - p1.y);
                        intersections[idx] = intersection;
                        idx += 1;
                    }
                }

                // Get intersection slice
                const intersection_slice = intersections;

                // Sort intersections
                if (intersection_slice.len > 1) {
                    std.mem.sort(f32, intersection_slice, {}, std.sort.asc(f32));
                }

                // Fill between pairs of intersections
                var i: usize = 0;
                while (i + 1 < intersection_slice.len) : (i += 2) {
                    const left_edge = intersection_slice[i];
                    const right_edge = intersection_slice[i + 1];

                    const x_start = @max(0, @as(i32, @intFromFloat(@floor(left_edge))));
                    const x_end = @min(@as(i32, @intCast(cols)) - 1, @as(i32, @intFromFloat(@ceil(right_edge))));

                    var x = x_start;
                    while (x <= x_end) : (x += 1) {
                        const pos = @as(usize, @intCast(y)) * cols + @as(usize, @intCast(x));

                        if (mode == .soft) {
                            // Apply antialiasing at edges
                            const fx = @as(f32, @floatFromInt(x));
                            var alpha: f32 = 1.0;
                            if (fx < left_edge + 1) {
                                alpha = @min(alpha, fx + antialias_edge_offset - left_edge);
                            }
                            if (fx > right_edge - 1) {
                                alpha = @min(alpha, right_edge - (fx - antialias_edge_offset));
                            }
                            alpha = @max(0, @min(1, alpha));

                            if (alpha > 0) {
                                self.assignPixel(x, @intCast(y), color, alpha);
                            }
                        } else {
                            // No antialiasing - direct pixel write
                            self.image.data[pos] = convert(T, c2);
                        }
                    }
                }
            }
        }
        /// Fills a circle on the given image.
        /// Use FillMode.soft for anti-aliased edges or FillMode.fast for hard edges.
        pub fn fillCircle(self: Self, center: Point2d(f32), radius: f32, color: anytype, mode: FillMode) void {
            comptime assert(isColor(@TypeOf(color)));
            if (radius <= 0) return;

            switch (mode) {
                .fast => self.fillCircleFast(center, radius, color),
                .soft => self.fillCircleSoft(center, radius, color),
            }
        }

        /// Internal function for filling smooth (anti-aliased) circles.
        fn fillCircleSoft(self: Self, center: Point2d(f32), radius: f32, color: anytype) void {
            const frows: f32 = @floatFromInt(self.image.rows);
            const fcols: f32 = @floatFromInt(self.image.cols);
            const left: usize = @intFromFloat(@round(@max(0, center.x - radius)));
            const top: usize = @intFromFloat(@round(@max(0, center.y - radius)));
            const right: usize = @intFromFloat(@round(@min(fcols, center.x + radius)));
            const bottom: usize = @intFromFloat(@round(@min(frows, center.y + radius)));

            for (top..bottom) |r| {
                const y = as(f32, r) - center.y;
                for (left..right) |c| {
                    const x = as(f32, c) - center.x;
                    const dist_sq = x * x + y * y;
                    if (dist_sq <= radius * radius) {
                        // Apply antialiasing at the edge
                        const dist = @sqrt(dist_sq);
                        if (dist > radius - 1) {
                            // Edge antialiasing
                            const edge_alpha = radius - dist;
                            self.assignPixel(@intCast(c), @intCast(r), color, edge_alpha);
                        } else {
                            // Full opacity in the center - direct assignment
                            self.assignPixel(@intCast(c), @intCast(r), color, 1.0);
                        }
                    }
                }
            }
        }

        /// Internal function for filling solid (non-anti-aliased) circles.
        fn fillCircleFast(self: Self, center: Point2d(f32), radius: f32, color: anytype) void {
            const solid_color = convert(T, color);
            const frows: f32 = @floatFromInt(self.image.rows);
            const fcols: f32 = @floatFromInt(self.image.cols);
            const left: usize = @intFromFloat(@round(@max(0, center.x - radius)));
            const top: usize = @intFromFloat(@round(@max(0, center.y - radius)));
            const right: usize = @intFromFloat(@round(@min(fcols, center.x + radius)));
            const bottom: usize = @intFromFloat(@round(@min(frows, center.y + radius)));

            const radius_sq = radius * radius;

            for (top..bottom) |r| {
                const y = as(f32, r) - center.y;
                for (left..right) |c| {
                    const x = as(f32, c) - center.x;
                    const dist_sq = x * x + y * y;
                    if (dist_sq <= radius_sq) {
                        const pos = r * self.image.cols + c;
                        self.image.data[pos] = solid_color;
                    }
                }
            }
        }

        /// Draws a quadratic Bézier curve with specified width and fill mode.
        pub fn drawQuadraticBezier(
            self: Self,
            p0: Point2d(f32),
            p1: Point2d(f32),
            p2: Point2d(f32),
            color: anytype,
            width: usize,
            mode: FillMode,
        ) void {
            comptime assert(isColor(@TypeOf(color)));
            if (width == 0) return;

            const estimated_length = estimateQuadraticBezierLength(p0, p1, p2);

            self.drawBezierTessellated(
                estimated_length,
                pixels_per_segment_quadratic,
                quadratic_min_segments_count,
                evalQuadraticBezier,
                .{ p0, p1, p2 },
                color,
                width,
                mode,
            );
        }

        /// Draws a cubic Bézier curve with specified width and fill mode.
        /// The curve is adaptively subdivided for optimal quality and performance.
        pub fn drawCubicBezier(
            self: Self,
            p0: Point2d(f32),
            p1: Point2d(f32),
            p2: Point2d(f32),
            p3: Point2d(f32),
            color: anytype,
            width: usize,
            mode: FillMode,
        ) void {
            comptime assert(isColor(@TypeOf(color)));
            if (width == 0) return;

            const estimated_length = estimateCubicBezierLength(p0, p1, p2, p3);
            const pixels_per_segment: f32 = if (mode == .soft or width > 2) pixels_per_segment_soft else pixels_per_segment_fast;

            self.drawBezierTessellated(
                estimated_length,
                pixels_per_segment,
                spline_min_segments_count,
                evalCubicBezier,
                .{ p0, p1, p2, p3 },
                color,
                width,
                mode,
            );
        }

        /// Draws a spline polygon outline with Bézier curves connecting vertices.
        /// The polygon's edges are rendered as cubic Bézier splines for smooth, curved appearance.
        /// Use tension to control curve smoothness: 0=sharp corners, 1=maximum smoothness.
        pub fn drawSplinePolygon(self: Self, polygon: []const Point2d(f32), color: anytype, width: usize, tension: f32, mode: FillMode) void {
            comptime assert(isColor(@TypeOf(color)));
            if (width == 0 or polygon.len < 3) return;

            for (0..polygon.len) |i| {
                const p0 = polygon[i];
                const p1 = polygon[(i + 1) % polygon.len];
                const p2 = polygon[(i + 2) % polygon.len];
                const control_points = calculateSmoothControlPoints(p0, p1, p2, tension);
                self.drawCubicBezier(p0, control_points.cp1, control_points.cp2, p1, color, width, mode);
            }
        }

        /// Fills a spline polygon with Bézier curves connecting vertices.
        /// The polygon's outline is defined by Bézier splines for smooth, curved edges.
        /// Use tension to control curve smoothness: 0=sharp corners, 1=maximum smoothness.
        pub fn fillSplinePolygon(self: Self, polygon: []const Point2d(f32), color: anytype, tension: f32, mode: FillMode) !void {
            comptime assert(isColor(@TypeOf(color)));
            if (polygon.len < 3) return;

            // Stack buffer for common cases (up to 50 segments per curve, 8 curves)
            var stack_buffer: [spline_polygon_stack_buffer_size]Point2d(f32) = undefined;
            var total_points: usize = 0;

            // First pass: calculate total points needed
            const pixels_per_segment = pixels_per_segment_fast; // Balance between quality and performance for filled shapes
            for (0..polygon.len) |i| {
                const p0 = polygon[i];
                const p1 = polygon[(i + 1) % polygon.len];
                const p2 = polygon[(i + 2) % polygon.len];
                const control_points = calculateSmoothControlPoints(p0, p1, p2, tension);
                const estimated_length = estimateCubicBezierLength(p0, control_points.cp1, control_points.cp2, p1);
                const segments = @max(spline_min_segments_count, @min(spline_max_segments_count, @as(usize, @intFromFloat(estimated_length / pixels_per_segment))));
                total_points += segments;
            }

            // Use stack buffer if possible, otherwise allocate
            var points_buffer: []Point2d(f32) = undefined;
            var heap_buffer: ?[]Point2d(f32) = null;
            defer if (heap_buffer) |h| self.allocator.free(h);

            if (total_points <= spline_polygon_stack_buffer_size) {
                points_buffer = stack_buffer[0..total_points];
            } else {
                heap_buffer = try self.allocator.alloc(Point2d(f32), total_points);
                points_buffer = heap_buffer.?;
            }

            // Second pass: tessellate curves into the buffer
            var write_idx: usize = 0;
            for (0..polygon.len) |i| {
                const p0 = polygon[i];
                const p1 = polygon[(i + 1) % polygon.len];
                const p2 = polygon[(i + 2) % polygon.len];
                const control_points = calculateSmoothControlPoints(p0, p1, p2, tension);

                const estimated_length = estimateCubicBezierLength(p0, control_points.cp1, control_points.cp2, p1);
                const segments = @max(spline_min_segments_count, @min(spline_max_segments_count, @as(usize, @intFromFloat(estimated_length / pixels_per_segment))));

                // Tessellate directly into our buffer
                const segment_buffer = points_buffer[write_idx .. write_idx + segments];
                const actual_segments = tessellateBezier(
                    estimated_length,
                    pixels_per_segment,
                    spline_min_segments_count, // min_segments for cubic
                    spline_max_segments_count, // max_segments
                    evalCubicBezier,
                    .{ p0, control_points.cp1, control_points.cp2, p1 },
                    segment_buffer,
                );
                write_idx += actual_segments;
            }

            try self.fillPolygon(points_buffer, color, mode);
        }

        /// Evaluates a quadratic Bézier curve at parameter t.
        /// Uses the standard quadratic Bézier formula: (1-t)²P₀ + 2t(1-t)P₁ + t²P₂
        /// Parameter t is in range [0, 1] where 0=start point, 1=end point.
        fn evalQuadraticBezier(p0: Point2d(f32), p1: Point2d(f32), p2: Point2d(f32), t: f32) Point2d(f32) {
            const u = 1 - t;
            const uu = u * u;
            const tt = t * t;
            return .{
                .x = uu * p0.x + 2 * u * t * p1.x + tt * p2.x,
                .y = uu * p0.y + 2 * u * t * p1.y + tt * p2.y,
            };
        }

        /// Evaluates a cubic Bézier curve at parameter t.
        /// Uses the standard cubic Bézier formula: (1-t)³P₀ + 3t(1-t)²P₁ + 3t²(1-t)P₂ + t³P₃
        /// Parameter t is in range [0, 1] where 0=start point, 1=end point.
        fn evalCubicBezier(p0: Point2d(f32), p1: Point2d(f32), p2: Point2d(f32), p3: Point2d(f32), t: f32) Point2d(f32) {
            const u = 1 - t;
            const uu = u * u;
            const uuu = uu * u;
            const tt = t * t;
            const ttt = tt * t;
            return .{
                .x = uuu * p0.x + 3 * uu * t * p1.x + 3 * u * tt * p2.x + ttt * p3.x,
                .y = uuu * p0.y + 3 * uu * t * p1.y + 3 * u * tt * p2.y + ttt * p3.y,
            };
        }

        /// Estimates the length of a quadratic Bézier curve segment.
        /// Uses chord + control polygon approximation for fast, reasonably accurate estimation.
        /// The estimate is (chord_length + control_polygon_length) / 2.
        fn estimateQuadraticBezierLength(p0: Point2d(f32), p1: Point2d(f32), p2: Point2d(f32)) f32 {
            // Use chord + control polygon approximation
            const chord = p0.distance(p2);
            const control_net = p0.distance(p1) + p1.distance(p2);
            return (chord + control_net) / 2.0;
        }

        /// Estimates the length of a cubic Bézier curve segment.
        /// Uses chord + control polygon approximation for fast, reasonably accurate estimation.
        /// The estimate is (chord_length + control_polygon_length) / 2.
        fn estimateCubicBezierLength(p0: Point2d(f32), p1: Point2d(f32), p2: Point2d(f32), p3: Point2d(f32)) f32 {
            // Use chord + control polygon approximation
            const chord = p0.distance(p3);
            const control_net = p0.distance(p1) + p1.distance(p2) + p2.distance(p3);
            return (chord + control_net) / 2.0;
        }

        /// Tessellates a Bézier curve into discrete points for line segment rendering.
        /// Adaptively determines segment count based on curve length and desired quality.
        ///
        /// Parameters:
        /// - estimated_length: Approximate curve length in pixels
        /// - pixels_per_segment: Target distance between tessellation points
        /// - evalFn: Function to evaluate curve at parameter t (e.g. evalCubicBezier)
        /// - evalArgs: Arguments to pass to evalFn before the t parameter
        ///
        /// Returns the number of points actually written to buffer.
        fn tessellateBezier(
            estimated_length: f32,
            pixels_per_segment: f32,
            min_segments: usize,
            max_segments: usize,
            comptime evalFn: anytype,
            evalArgs: anytype,
            buffer: []Point2d(f32),
        ) usize {
            const segments = @max(min_segments, @min(max_segments, @as(usize, @intFromFloat(estimated_length / pixels_per_segment))));
            const actual_segments = @min(segments, buffer.len);

            for (0..actual_segments) |i| {
                const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(actual_segments - 1));
                buffer[i] = @call(.auto, evalFn, evalArgs ++ .{t});
            }

            return actual_segments;
        }

        /// Draws a Bézier curve by tessellating it into line segments.
        fn drawBezierTessellated(
            self: Self,
            estimated_length: f32,
            pixels_per_segment: f32,
            min_segments: usize,
            comptime evalFn: anytype,
            evalArgs: anytype,
            color: anytype,
            width: usize,
            mode: FillMode,
        ) void {
            var stack_buffer: [bezier_max_segments_count]Point2d(f32) = undefined;

            const actual_segments = tessellateBezier(
                estimated_length,
                pixels_per_segment,
                min_segments,
                bezier_max_segments_count,
                evalFn,
                evalArgs,
                &stack_buffer,
            );

            // Draw lines between consecutive points
            for (1..actual_segments) |i| {
                self.drawLine(stack_buffer[i - 1], stack_buffer[i], color, width, mode);
            }
        }

        /// Calculates Bézier control points for smooth spline interpolation through three points.
        /// Creates control points that produce a smooth curve through p1, influenced by p0 and p2.
        ///
        /// Parameters:
        /// - p0: Previous point (influences incoming tangent)
        /// - p1: Current point (the vertex being processed)
        /// - p2: Next point (influences outgoing tangent)
        /// - tension: Curve tension (0=sharp corners, 1=maximum smoothness)
        ///
        /// Returns control points for cubic Bézier: cp1 (outgoing from p0), cp2 (incoming to p1).
        fn calculateSmoothControlPoints(p0: Point2d(f32), p1: Point2d(f32), p2: Point2d(f32), tension: f32) struct { cp1: Point2d(f32), cp2: Point2d(f32) } {
            const tension_factor = 1 - @max(0, @min(1, tension));
            return .{
                .cp1 = .{
                    .x = p0.x + (p1.x - p0.x) * tension_factor,
                    .y = p0.y + (p1.y - p0.y) * tension_factor,
                },
                .cp2 = .{
                    .x = p1.x - (p2.x - p1.x) * tension_factor,
                    .y = p1.y - (p2.y - p1.y) * tension_factor,
                },
            };
        }
    };
}

// Property-based tests for drawing functions

// MD5 checksum tests for pixel-perfect stability
//
// These tests ensure that drawing operations produce identical results across releases.
// To update the checksums after intentional changes to drawing algorithms:
//   zig build test -Dprint-md5sums=true
// Then copy the printed checksums into the md5_checksums array below.
const DrawTestCase = struct {
    name: []const u8,
    md5sum: []const u8,
    draw_fn: *const fn (canvas: Canvas(Rgba)) void,
};

// Golden MD5 checksums for drawing operations
const md5_checksums = [_]DrawTestCase{
    .{ .name = "line_horizontal", .md5sum = "96fc75d0d893373c0050e5fe76f5d7ea", .draw_fn = drawLineHorizontal },
    .{ .name = "line_vertical", .md5sum = "f7d52e274636af2b20b62172a408b446", .draw_fn = drawLineVertical },
    .{ .name = "line_diagonal", .md5sum = "1aee6bf80fd2e6a849e5520937566478", .draw_fn = drawLineDiagonal },
    .{ .name = "line_thick", .md5sum = "d8323d8d6580a34e724873701245f117", .draw_fn = drawLineThick },
    .{ .name = "circle_filled_solid", .md5sum = "efe2aa5419c9ffdead0dfddffb3b6a67", .draw_fn = drawCircleFilledSolid },
    .{ .name = "circle_filled_smooth", .md5sum = "4996924718641236276cdb1c166ae515", .draw_fn = drawCircleFilledSmooth },
    .{ .name = "circle_outline", .md5sum = "ae7e973d5644ff7bdde7338296e4ab40", .draw_fn = drawCircleOutline },
    .{ .name = "rectangle_filled", .md5sum = "3783f1119b7d5482b5a333f76c322c92", .draw_fn = drawRectangleFilled },
    .{ .name = "rectangle_outline", .md5sum = "033fdc24b89399af7b1810783e357b5f", .draw_fn = drawRectangleOutline },
    .{ .name = "triangle_filled", .md5sum = "283a9de3dd51dd00794559cc231ff5ac", .draw_fn = drawTriangleFilled },
    .{ .name = "bezier_cubic", .md5sum = "3a2b0d540a2353c817077729ee10007a", .draw_fn = drawBezierCubic },
    .{ .name = "bezier_quadratic", .md5sum = "c3286e308aaaef5b302129cf67b713c6", .draw_fn = drawBezierQuadratic },
    .{ .name = "polygon_complex", .md5sum = "da9b83426d2118ce99948eabebff91fb", .draw_fn = drawPolygonComplex },
    .{ .name = "spline_polygon", .md5sum = "6bae24f211c7fdd391cb5159dd4e8fd0", .draw_fn = drawSplinePolygon },
};

// Test drawing functions for MD5 checksums
fn drawLineHorizontal(canvas: Canvas(Rgba)) void {
    const color = Rgba{ .r = 255, .g = 0, .b = 0, .a = 255 };
    canvas.drawLine(.{ .x = 10, .y = 50 }, .{ .x = 90, .y = 50 }, color, 1, .fast);
}

fn drawLineVertical(canvas: Canvas(Rgba)) void {
    const color = Rgba{ .r = 0, .g = 255, .b = 0, .a = 255 };
    canvas.drawLine(.{ .x = 50, .y = 10 }, .{ .x = 50, .y = 90 }, color, 1, .fast);
}

fn drawLineDiagonal(canvas: Canvas(Rgba)) void {
    const color = Rgba{ .r = 0, .g = 0, .b = 255, .a = 255 };
    canvas.drawLine(.{ .x = 10, .y = 10 }, .{ .x = 90, .y = 90 }, color, 1, .fast);
}

fn drawLineThick(canvas: Canvas(Rgba)) void {
    const color = Rgba{ .r = 255, .g = 128, .b = 0, .a = 255 };
    canvas.drawLine(.{ .x = 20, .y = 20 }, .{ .x = 80, .y = 80 }, color, 5, .soft);
}

fn drawCircleFilledSolid(canvas: Canvas(Rgba)) void {
    const color = Rgba{ .r = 128, .g = 0, .b = 128, .a = 255 };
    canvas.fillCircle(.{ .x = 50, .y = 50 }, 30, color, .fast);
}

fn drawCircleFilledSmooth(canvas: Canvas(Rgba)) void {
    const color = Rgba{ .r = 0, .g = 128, .b = 128, .a = 255 };
    canvas.fillCircle(.{ .x = 50, .y = 50 }, 25, color, .soft);
}

fn drawCircleOutline(canvas: Canvas(Rgba)) void {
    const color = Rgba{ .r = 255, .g = 255, .b = 0, .a = 255 };
    canvas.drawCircle(.{ .x = 50, .y = 50 }, 35, color, 3, .soft);
}

fn drawRectangleFilled(canvas: Canvas(Rgba)) void {
    const color = Rgba{ .r = 64, .g = 128, .b = 192, .a = 255 };
    const rect = Rectangle(f32){ .l = 20, .t = 30, .r = 80, .b = 70 };
    const corners = [_]Point2d(f32){
        .{ .x = rect.l, .y = rect.t },
        .{ .x = rect.r, .y = rect.t },
        .{ .x = rect.r, .y = rect.b },
        .{ .x = rect.l, .y = rect.b },
    };
    canvas.fillPolygon(&corners, color, .fast) catch unreachable;
}

fn drawRectangleOutline(canvas: Canvas(Rgba)) void {
    const color = Rgba{ .r = 192, .g = 64, .b = 128, .a = 255 };
    const rect = Rectangle(f32){ .l = 15, .t = 25, .r = 85, .b = 75 };
    canvas.drawRectangle(rect, color, 2, .soft);
}

fn drawTriangleFilled(canvas: Canvas(Rgba)) void {
    const color = Rgba{ .r = 255, .g = 192, .b = 128, .a = 255 };
    const triangle = [_]Point2d(f32){
        .{ .x = 50, .y = 20 },
        .{ .x = 80, .y = 80 },
        .{ .x = 20, .y = 80 },
    };
    canvas.fillPolygon(&triangle, color, .soft) catch unreachable;
}

fn drawBezierCubic(canvas: Canvas(Rgba)) void {
    const color = Rgba{ .r = 128, .g = 192, .b = 255, .a = 255 };
    canvas.drawCubicBezier(
        .{ .x = 10, .y = 50 },
        .{ .x = 30, .y = 10 },
        .{ .x = 70, .y = 90 },
        .{ .x = 90, .y = 50 },
        color,
        2,
        .fast,
    );
}

fn drawBezierQuadratic(canvas: Canvas(Rgba)) void {
    const color = Rgba{ .r = 255, .g = 128, .b = 192, .a = 255 };
    canvas.drawQuadraticBezier(
        .{ .x = 20, .y = 80 },
        .{ .x = 50, .y = 20 },
        .{ .x = 80, .y = 80 },
        color,
        3,
        .soft,
    );
}

fn drawPolygonComplex(canvas: Canvas(Rgba)) void {
    const color = Rgba{ .r = 128, .g = 255, .b = 128, .a = 255 };
    const polygon = [_]Point2d(f32){
        .{ .x = 50, .y = 10 },
        .{ .x = 70, .y = 30 },
        .{ .x = 90, .y = 40 },
        .{ .x = 70, .y = 60 },
        .{ .x = 50, .y = 90 },
        .{ .x = 30, .y = 60 },
        .{ .x = 10, .y = 40 },
        .{ .x = 30, .y = 30 },
    };
    canvas.fillPolygon(&polygon, color, .soft) catch unreachable;
}

fn drawSplinePolygon(canvas: Canvas(Rgba)) void {
    const color = Rgba{ .r = 192, .g = 128, .b = 255, .a = 255 };
    const polygon = [_]Point2d(f32){
        .{ .x = 50, .y = 20 },
        .{ .x = 80, .y = 35 },
        .{ .x = 80, .y = 65 },
        .{ .x = 50, .y = 80 },
        .{ .x = 20, .y = 65 },
        .{ .x = 20, .y = 35 },
    };
    canvas.drawSplinePolygon(&polygon, color, 2, 0.5, .soft);
}

test "MD5 checksum regression tests" {
    const allocator = testing.allocator;
    const print_md5sums = @import("build_options").print_md5sums;

    // Fixed size for consistent checksums
    const width = 100;
    const height = 100;

    for (md5_checksums) |test_case| {
        var img = try Image(Rgba).initAlloc(allocator, width, height);
        defer img.deinit(allocator);

        // White background
        for (img.data) |*pixel| {
            pixel.* = Rgba{ .r = 255, .g = 255, .b = 255, .a = 255 };
        }

        const canvas = Canvas(Rgba).init(img, allocator);
        test_case.draw_fn(canvas);

        // Calculate MD5
        var md5sum: [std.crypto.hash.Md5.digest_length]u8 = undefined;
        std.crypto.hash.Md5.hash(img.asBytes(), &md5sum, .{});
        const hex_bytes = std.fmt.bytesToHex(md5sum, .lower);

        if (print_md5sums) {
            std.debug.print("    .{{ .name = \"{s}\", .md5sum = \"{s}\", .draw_fn = {s} }},\n", .{
                test_case.name,
                &hex_bytes,
                test_case.name,
            });
        } else {
            try expectEqualStrings(&hex_bytes, test_case.md5sum);
        }
    }
}

test "line endpoints are connected" {
    const allocator = testing.allocator;
    const width = 100;
    const height = 100;
    var img = try Image(Rgba).initAlloc(allocator, width, height);
    defer img.deinit(allocator);

    // Fill with white
    for (img.data) |*pixel| {
        pixel.* = Rgba{ .r = 255, .g = 255, .b = 255, .a = 255 };
    }

    const canvas = Canvas(Rgba).init(img, allocator);
    const color = Rgba{ .r = 0, .g = 0, .b = 0, .a = 255 };

    // Test various line directions
    const test_cases = [_]struct { p1: Point2d(f32), p2: Point2d(f32) }{
        .{ .p1 = .{ .x = 10, .y = 10 }, .p2 = .{ .x = 90, .y = 10 } }, // horizontal
        .{ .p1 = .{ .x = 10, .y = 10 }, .p2 = .{ .x = 10, .y = 90 } }, // vertical
        .{ .p1 = .{ .x = 10, .y = 10 }, .p2 = .{ .x = 90, .y = 90 } }, // diagonal
        .{ .p1 = .{ .x = 90, .y = 10 }, .p2 = .{ .x = 10, .y = 90 } }, // reverse diagonal
    };

    for (test_cases) |tc| {
        // Clear image
        for (img.data) |*pixel| {
            pixel.* = Rgba{ .r = 255, .g = 255, .b = 255, .a = 255 };
        }

        canvas.drawLine(tc.p1, tc.p2, color, 1, .fast);

        // Check that endpoints are set (or very close)
        // At least one pixel near each endpoint should be black
        var p1_found = false;
        var p2_found = false;

        // Check 3x3 area around endpoints
        for (0..3) |dy| {
            for (0..3) |dx| {
                const y1 = @as(i32, @intFromFloat(tc.p1.y)) + @as(i32, @intCast(dy)) - 1;
                const x1 = @as(i32, @intFromFloat(tc.p1.x)) + @as(i32, @intCast(dx)) - 1;
                const y2 = @as(i32, @intFromFloat(tc.p2.y)) + @as(i32, @intCast(dy)) - 1;
                const x2 = @as(i32, @intFromFloat(tc.p2.x)) + @as(i32, @intCast(dx)) - 1;

                if (y1 >= 0 and y1 < height and x1 >= 0 and x1 < width) {
                    const idx1 = @as(usize, @intCast(y1)) * width + @as(usize, @intCast(x1));
                    if (img.data[idx1].r == 0) p1_found = true;
                }

                if (y2 >= 0 and y2 < height and x2 >= 0 and x2 < width) {
                    const idx2 = @as(usize, @intCast(y2)) * width + @as(usize, @intCast(x2));
                    if (img.data[idx2].r == 0) p2_found = true;
                }
            }
        }

        try expect(p1_found);
        try expect(p2_found);
    }
}

test "thick lines have correct width" {
    const allocator = testing.allocator;
    const width = 200;
    const height = 200;
    var img = try Image(Rgba).initAlloc(allocator, width, height);
    defer img.deinit(allocator);

    const canvas = Canvas(Rgba).init(img, allocator);
    const color = Rgba{ .r = 0, .g = 0, .b = 0, .a = 255 };

    // Test different line widths
    const line_widths = [_]usize{ 1, 3, 5, 10, 20 };

    for (line_widths) |line_width| {
        // Clear image
        for (img.data) |*pixel| {
            pixel.* = Rgba{ .r = 255, .g = 255, .b = 255, .a = 255 };
        }

        // Draw horizontal line in the middle
        const y = @as(f32, @floatFromInt(height / 2));
        canvas.drawLine(.{ .x = 50, .y = y }, .{ .x = 150, .y = y }, color, line_width, .fast);

        // Measure actual width at several points along the line
        var measured_widths: [3]usize = .{ 0, 0, 0 };
        const x_positions = [_]usize{ 75, 100, 125 };

        for (x_positions, 0..) |x, i| {
            var min_y: usize = height;
            var max_y: usize = 0;

            for (0..height) |py| {
                const idx = py * width + x;
                if (img.data[idx].r == 0) {
                    min_y = @min(min_y, py);
                    max_y = @max(max_y, py);
                }
            }

            if (max_y >= min_y) {
                measured_widths[i] = max_y - min_y + 1;
            }
        }

        // Allow for some tolerance due to rounding
        for (measured_widths) |measured| {
            try expect(measured >= line_width - 1 and measured <= line_width + 1);
        }
    }
}

test "filled circle has correct radius" {
    const allocator = testing.allocator;
    const width = 200;
    const height = 200;
    var img = try Image(Rgba).initAlloc(allocator, width, height);
    defer img.deinit(allocator);

    const canvas = Canvas(Rgba).init(img, allocator);
    const color = Rgba{ .r = 0, .g = 0, .b = 0, .a = 255 };

    const test_radii = [_]f32{ 5, 10, 20, 30, 40 };
    const center = Point2d(f32){ .x = 100, .y = 100 };

    for (test_radii) |radius| {
        // Clear image
        for (img.data) |*pixel| {
            pixel.* = Rgba{ .r = 255, .g = 255, .b = 255, .a = 255 };
        }

        canvas.fillCircle(center, radius, color, .fast);

        // Check pixels at various distances from center
        var inside_count: usize = 0;
        var outside_count: usize = 0;
        var edge_count: usize = 0;

        for (0..height) |y| {
            for (0..width) |x| {
                const dx = @as(f32, @floatFromInt(x)) - center.x;
                const dy = @as(f32, @floatFromInt(y)) - center.y;
                const dist = @sqrt(dx * dx + dy * dy);
                const idx = y * width + x;
                const is_black = img.data[idx].r == 0;

                if (dist < radius - 1) {
                    // Should be inside
                    if (is_black) inside_count += 1;
                } else if (dist > radius + 1) {
                    // Should be outside
                    if (!is_black) outside_count += 1;
                } else {
                    // Edge region
                    edge_count += 1;
                }
            }
        }

        // Most pixels inside radius should be filled
        const inside_total = @as(usize, @intFromFloat(std.math.pi * (radius - 1) * (radius - 1)));
        // Allow 15% tolerance for small circles due to discretization
        const tolerance_factor: f32 = if (radius <= 10) 0.85 else 0.9;
        const expected_count = @as(usize, @intFromFloat(@as(f32, @floatFromInt(inside_total)) * tolerance_factor));
        try expect(inside_count >= expected_count);
    }
}

test "circle outline has correct thickness" {
    const allocator = testing.allocator;
    const width = 200;
    const height = 200;
    var img = try Image(Rgba).initAlloc(allocator, width, height);
    defer img.deinit(allocator);

    const canvas = Canvas(Rgba).init(img, allocator);
    const color = Rgba{ .r = 0, .g = 0, .b = 0, .a = 255 };

    const center = Point2d(f32){ .x = 100, .y = 100 };
    const radius: f32 = 40;
    const line_widths = [_]usize{ 1, 3, 5, 10 };

    for (line_widths) |line_width| {
        // Clear image
        for (img.data) |*pixel| {
            pixel.* = Rgba{ .r = 255, .g = 255, .b = 255, .a = 255 };
        }

        canvas.drawCircle(center, radius, color, line_width, .fast);

        // Sample along several radii to check thickness
        const angles = [_]f32{ 0, std.math.pi / @as(f32, 4), std.math.pi / @as(f32, 2), 3 * std.math.pi / @as(f32, 4) };

        for (angles) |angle| {
            var black_pixels: usize = 0;

            // Count black pixels along this radius
            var r: f32 = 0;
            while (r < radius * 2) : (r += 0.5) {
                const x = center.x + r * @cos(angle);
                const y = center.y + r * @sin(angle);

                if (x >= 0 and x < width and y >= 0 and y < height) {
                    const idx = @as(usize, @intFromFloat(y)) * width + @as(usize, @intFromFloat(x));
                    if (img.data[idx].r == 0) {
                        black_pixels += 1;
                    }
                }
            }

            // Should have approximately line_width black pixels
            try expect(black_pixels >= line_width / 2 and black_pixels <= line_width * 3);
        }
    }
}

test "filled rectangle has correct area" {
    const allocator = testing.allocator;
    const width = 200;
    const height = 200;
    var img = try Image(Rgba).initAlloc(allocator, width, height);
    defer img.deinit(allocator);

    const canvas = Canvas(Rgba).init(img, allocator);
    const color = Rgba{ .r = 0, .g = 0, .b = 0, .a = 255 };

    const rect = Rectangle(f32){ .l = 50, .t = 50, .r = 150, .b = 130 };
    const rect_width = rect.r - rect.l;
    const rect_height = rect.b - rect.t;
    const expected_area = rect_width * rect_height;

    // Clear and draw filled rectangle using polygon fill
    for (img.data) |*pixel| {
        pixel.* = Rgba{ .r = 255, .g = 255, .b = 255, .a = 255 };
    }

    const corners = [_]Point2d(f32){
        .{ .x = rect.l, .y = rect.t },
        .{ .x = rect.r, .y = rect.t },
        .{ .x = rect.r, .y = rect.b },
        .{ .x = rect.l, .y = rect.b },
    };
    try canvas.fillPolygon(&corners, color, .fast);

    // Count black pixels
    var black_pixels: usize = 0;
    for (img.data) |pixel| {
        if (pixel.r == 0) black_pixels += 1;
    }

    // Should match expected area closely
    const tolerance = expected_area * 0.01; // 1% tolerance
    const diff = @abs(@as(f32, @floatFromInt(black_pixels)) - expected_area);
    try expect(diff <= tolerance);
}

test "polygon fill respects convexity" {
    const allocator = testing.allocator;
    const width = 200;
    const height = 200;
    var img = try Image(Rgba).initAlloc(allocator, width, height);
    defer img.deinit(allocator);

    const canvas = Canvas(Rgba).init(img, allocator);
    const color = Rgba{ .r = 0, .g = 0, .b = 0, .a = 255 };

    // Test convex polygon (triangle)
    const triangle = [_]Point2d(f32){
        .{ .x = 100, .y = 30 },
        .{ .x = 170, .y = 150 },
        .{ .x = 30, .y = 150 },
    };

    for (img.data) |*pixel| {
        pixel.* = Rgba{ .r = 255, .g = 255, .b = 255, .a = 255 };
    }

    try canvas.fillPolygon(&triangle, color, .fast);

    // Check that points inside triangle are filled
    const test_points = [_]struct { p: Point2d(f32), inside: bool }{
        .{ .p = .{ .x = 100, .y = 100 }, .inside = true }, // centroid
        .{ .p = .{ .x = 100, .y = 50 }, .inside = true }, // near top
        .{ .p = .{ .x = 50, .y = 140 }, .inside = true }, // near bottom left
        .{ .p = .{ .x = 150, .y = 140 }, .inside = true }, // near bottom right
        .{ .p = .{ .x = 20, .y = 20 }, .inside = false }, // outside
        .{ .p = .{ .x = 180, .y = 180 }, .inside = false }, // outside
    };

    for (test_points) |tp| {
        const x = @as(usize, @intFromFloat(tp.p.x));
        const y = @as(usize, @intFromFloat(tp.p.y));
        if (x < width and y < height) {
            const idx = y * width + x;
            const is_black = img.data[idx].r == 0;
            try expectEqual(tp.inside, is_black);
        }
    }
}

test "antialiased vs solid fill coverage" {
    const allocator = testing.allocator;
    const width = 100;
    const height = 100;
    var img_solid = try Image(Rgba).initAlloc(allocator, width, height);
    defer img_solid.deinit(allocator);
    var img_smooth = try Image(Rgba).initAlloc(allocator, width, height);
    defer img_smooth.deinit(allocator);

    const canvas_solid = Canvas(Rgba).init(img_solid, allocator);
    const canvas_smooth = Canvas(Rgba).init(img_smooth, allocator);
    const color = Rgba{ .r = 0, .g = 0, .b = 0, .a = 255 };

    // Clear both images
    for (img_solid.data, img_smooth.data) |*p1, *p2| {
        p1.* = Rgba{ .r = 255, .g = 255, .b = 255, .a = 255 };
        p2.* = Rgba{ .r = 255, .g = 255, .b = 255, .a = 255 };
    }

    // Draw same circle with different modes
    const center = Point2d(f32){ .x = 50, .y = 50 };
    const radius: f32 = 20;

    canvas_solid.fillCircle(center, radius, color, .fast);
    canvas_smooth.fillCircle(center, radius, color, .soft);

    // Count coverage (sum of darkness)
    var solid_coverage: f32 = 0;
    var smooth_coverage: f32 = 0;

    for (img_solid.data, img_smooth.data) |p1, p2| {
        solid_coverage += @as(f32, @floatFromInt(255 - p1.r));
        smooth_coverage += @as(f32, @floatFromInt(255 - p2.r));
    }

    // Antialiased version should have similar total coverage
    // but slightly less due to edge smoothing
    try expect(smooth_coverage > solid_coverage * 0.9);
    try expect(smooth_coverage <= solid_coverage);
}

test "bezier curve smoothness" {
    const allocator = testing.allocator;
    const width = 200;
    const height = 200;
    var img = try Image(Rgba).initAlloc(allocator, width, height);
    defer img.deinit(allocator);

    const canvas = Canvas(Rgba).init(img, allocator);
    const color = Rgba{ .r = 0, .g = 0, .b = 0, .a = 255 };

    // Clear image
    for (img.data) |*pixel| {
        pixel.* = Rgba{ .r = 255, .g = 255, .b = 255, .a = 255 };
    }

    // Draw cubic bezier
    const p0 = Point2d(f32){ .x = 20, .y = 100 };
    const p1 = Point2d(f32){ .x = 60, .y = 20 };
    const p2 = Point2d(f32){ .x = 140, .y = 180 };
    const p3 = Point2d(f32){ .x = 180, .y = 100 };

    canvas.drawCubicBezier(p0, p1, p2, p3, color, 2, .fast);

    // Verify endpoints are connected
    var p0_found = false;
    var p3_found = false;

    // Check 3x3 area around endpoints
    for (0..3) |dy| {
        for (0..3) |dx| {
            const y0 = @as(i32, @intFromFloat(p0.y)) + @as(i32, @intCast(dy)) - 1;
            const x0 = @as(i32, @intFromFloat(p0.x)) + @as(i32, @intCast(dx)) - 1;
            const y3 = @as(i32, @intFromFloat(p3.y)) + @as(i32, @intCast(dy)) - 1;
            const x3 = @as(i32, @intFromFloat(p3.x)) + @as(i32, @intCast(dx)) - 1;

            if (y0 >= 0 and y0 < height and x0 >= 0 and x0 < width) {
                const idx0 = @as(usize, @intCast(y0)) * width + @as(usize, @intCast(x0));
                if (img.data[idx0].r == 0) p0_found = true;
            }

            if (y3 >= 0 and y3 < height and x3 >= 0 and x3 < width) {
                const idx3 = @as(usize, @intCast(y3)) * width + @as(usize, @intCast(x3));
                if (img.data[idx3].r == 0) p3_found = true;
            }
        }
    }

    try expect(p0_found);
    try expect(p3_found);

    // Verify curve has pixels (not empty)
    var black_pixel_count: usize = 0;
    for (img.data) |pixel| {
        if (pixel.r == 0) black_pixel_count += 1;
    }
    try expect(black_pixel_count > 50); // Should have a reasonable number of pixels
}
