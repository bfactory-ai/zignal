//! This module provides a Canvas for drawing various shapes and lines on images.
const std = @import("std");
const assert = std.debug.assert;

const Rgba = @import("color.zig").Rgba;
const Hsv = @import("color.zig").Hsv;

const as = @import("meta.zig").as;
const convert = @import("color.zig").convert;
const Image = @import("image.zig").Image;
const isColor = @import("color.zig").isColor;
const Point2d = @import("point.zig").Point2d;
const Rectangle = @import("geometry.zig").Rectangle;

/// A drawing context for an image, providing methods to draw shapes and lines.
pub fn Canvas(comptime T: type) type {
    return struct {
        image: Image(T),
        allocator: std.mem.Allocator,

        const Self = @This();

        /// Creates a drawing canvas from an image, with an optional allocator for operations that need it.
        pub fn init(image: Image(T), allocator: std.mem.Allocator) Self {
            return .{ .image = image, .allocator = allocator };
        }

        /// Draws a colored straight line of a custom width between p1 and p2 on an image.
        /// It uses Xiaolin Wu's line algorithm to perform anti-aliasing for diagonal lines.
        /// If the `color` is of Rgba type, it alpha-blends it onto the image.
        pub fn drawLine(self: Self, p1: Point2d(f32), p2: Point2d(f32), width: usize, color: anytype) void {
            comptime assert(isColor(@TypeOf(color)));
            if (width == 0) return;

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
                const x = @round(p1.x);
                const y = @round(p1.y);
                var i = -half_width;
                while (i <= half_width) : (i += 1) {
                    var j = -half_width;
                    while (j <= half_width) : (j += 1) {
                        const px = x + i;
                        const py = y + j;
                        if (px >= 0 and px < cols and py >= 0 and py < rows) {
                            if (i * i + j * j <= half_width * half_width) {
                                const pos = as(usize, py) * self.image.cols + as(usize, px);
                                var c1 = convert(Rgba, self.image.data[pos]);
                                c1.blend(c2);
                                self.image.data[pos] = convert(T, c1);
                            }
                        }
                    }
                }
                return;
            }

            // Normalize direction vector
            const dir_x = dx / line_length;
            const dir_y = dy / line_length;

            // Calculate perpendicular vector (rotated 90 degrees)
            const perp_x = -dir_y;
            const perp_y = dir_x;

            // Special case for perfectly horizontal/vertical lines (faster rendering)
            if (@abs(dx) < 0.001) { // Vertical line
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
                        if (px >= 0 and px < cols) {
                            const pos = as(usize, y) * self.image.cols + as(usize, px);
                            var c1 = convert(Rgba, self.image.data[pos]);
                            c1.blend(c2);
                            self.image.data[pos] = convert(T, c1);
                        }
                    }
                }
                return;
            } else if (@abs(dy) < 0.001) { // Horizontal line
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
                        if (py >= 0 and py < rows) {
                            const pos = as(usize, py) * self.image.cols + as(usize, x);
                            var c1 = convert(Rgba, self.image.data[pos]);
                            c1.blend(c2);
                            self.image.data[pos] = convert(T, c1);
                        }
                    }
                }
                return;
            }

            // For diagonal lines, use distance-based anti-aliasing
            // Calculate bounding box of the thick line
            const min_x = @floor(@min(@min(p1.x - half_width * @abs(perp_x), p1.x + half_width * @abs(perp_x)), @min(p2.x - half_width * @abs(perp_x), p2.x + half_width * @abs(perp_x))));
            const max_x = @ceil(@max(@max(p1.x - half_width * @abs(perp_x), p1.x + half_width * @abs(perp_x)), @max(p2.x - half_width * @abs(perp_x), p2.x + half_width * @abs(perp_x))));
            const min_y = @floor(@min(@min(p1.y - half_width * @abs(perp_y), p1.y + half_width * @abs(perp_y)), @min(p2.y - half_width * @abs(perp_y), p2.y + half_width * @abs(perp_y))));
            const max_y = @ceil(@max(@max(p1.y - half_width * @abs(perp_y), p1.y + half_width * @abs(perp_y)), @max(p2.y - half_width * @abs(perp_y), p2.y + half_width * @abs(perp_y))));

            // Iterate through pixels in bounding box
            var y = @max(0, @as(i32, @intFromFloat(min_y)));
            while (y <= @min(@as(i32, @intFromFloat(rows)) - 1, @as(i32, @intFromFloat(max_y)))) : (y += 1) {
                var x = @max(0, @as(i32, @intFromFloat(min_x)));
                while (x <= @min(@as(i32, @intFromFloat(cols)) - 1, @as(i32, @intFromFloat(max_x)))) : (x += 1) {
                    const px = @as(Float, @floatFromInt(x));
                    const py = @as(Float, @floatFromInt(y));

                    // Calculate distance from pixel to line segment
                    const dist = self.distanceToLineSegment(px, py, p1, p2);

                    // Anti-aliased coverage based on distance
                    if (dist <= half_width + 0.5) {
                        var alpha: Float = 1.0;
                        if (dist > half_width - 0.5) {
                            // Smooth falloff at edges
                            alpha = (half_width + 0.5 - dist);
                        }

                        if (alpha > 0) {
                            const pos = @as(usize, @intCast(y)) * self.image.cols + @as(usize, @intCast(x));
                            var c1 = convert(Rgba, self.image.data[pos]);
                            var c_blend = c2;
                            c_blend.a = @intFromFloat(alpha * @as(Float, @floatFromInt(c2.a)));
                            c1.blend(c_blend);
                            self.image.data[pos] = convert(T, c1);
                        }
                    }
                }
            }
        }

        /// Calculate the shortest distance from a point to a line segment
        fn distanceToLineSegment(self: Self, px: f32, py: f32, p1: Point2d(f32), p2: Point2d(f32)) f32 {
            _ = self;
            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            const length_sq = dx * dx + dy * dy;

            if (length_sq == 0) {
                // Line segment is a point
                const dpx = px - p1.x;
                const dpy = py - p1.y;
                return @sqrt(dpx * dpx + dpy * dpy);
            }

            // Calculate parameter t for the closest point on the line segment
            const t = @max(0, @min(1, ((px - p1.x) * dx + (py - p1.y) * dy) / length_sq));

            // Find the closest point on the line segment
            const closest_x = p1.x + t * dx;
            const closest_y = p1.y + t * dy;

            // Return distance to closest point
            const dist_x = px - closest_x;
            const dist_y = py - closest_y;
            return @sqrt(dist_x * dist_x + dist_y * dist_y);
        }

        /// Internal helper to draw a 1-pixel wide anti-aliased line using Wu's algorithm
        fn drawWuLine(self: Self, p1: Point2d(f32), p2: Point2d(f32), color: anytype) void {
            const Float = @TypeOf(p1.x);
            const rows: Float = @floatFromInt(self.image.rows);
            const cols: Float = @floatFromInt(self.image.cols);
            const c2 = convert(Rgba, color);
            const max_alpha: Float = @floatFromInt(c2.a);

            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;

            if (@abs(dx) > @abs(dy)) {
                // X-major line
                var x1 = p1.x;
                var y1 = p1.y;
                var x2 = p2.x;
                var y2 = p2.y;

                if (x1 > x2) {
                    std.mem.swap(Float, &x1, &x2);
                    std.mem.swap(Float, &y1, &y2);
                }

                const gradient = (y2 - y1) / (x2 - x1);
                var y = y1;

                var x = x1;
                while (x <= x2) : (x += 1.0) {
                    if (x >= 0 and x < cols) {
                        const y_floor = @floor(y);
                        const y_frac = y - y_floor;

                        // Upper pixel
                        if (y_floor >= 0 and y_floor < rows) {
                            const pos1 = @as(usize, @intFromFloat(y_floor)) * self.image.cols + @as(usize, @intFromFloat(x));
                            var c1 = convert(Rgba, self.image.data[pos1]);
                            var c_blend = c2;
                            c_blend.a = @intFromFloat((1.0 - y_frac) * max_alpha);
                            c1.blend(c_blend);
                            self.image.data[pos1] = convert(T, c1);
                        }

                        // Lower pixel
                        if (y_floor + 1 >= 0 and y_floor + 1 < rows) {
                            const pos2 = @as(usize, @intFromFloat(y_floor + 1)) * self.image.cols + @as(usize, @intFromFloat(x));
                            var c1 = convert(Rgba, self.image.data[pos2]);
                            var c_blend = c2;
                            c_blend.a = @intFromFloat(y_frac * max_alpha);
                            c1.blend(c_blend);
                            self.image.data[pos2] = convert(T, c1);
                        }
                    }
                    y += gradient;
                }
            } else {
                // Y-major line
                var x1 = p1.x;
                var y1 = p1.y;
                var x2 = p2.x;
                var y2 = p2.y;

                if (y1 > y2) {
                    std.mem.swap(Float, &x1, &x2);
                    std.mem.swap(Float, &y1, &y2);
                }

                const gradient = (x2 - x1) / (y2 - y1);
                var x = x1;

                var y = y1;
                while (y <= y2) : (y += 1.0) {
                    if (y >= 0 and y < rows) {
                        const x_floor = @floor(x);
                        const x_frac = x - x_floor;

                        // Left pixel
                        if (x_floor >= 0 and x_floor < cols) {
                            const pos1 = @as(usize, @intFromFloat(y)) * self.image.cols + @as(usize, @intFromFloat(x_floor));
                            var c1 = convert(Rgba, self.image.data[pos1]);
                            var c_blend = c2;
                            c_blend.a = @intFromFloat((1.0 - x_frac) * max_alpha);
                            c1.blend(c_blend);
                            self.image.data[pos1] = convert(T, c1);
                        }

                        // Right pixel
                        if (x_floor + 1 >= 0 and x_floor + 1 < cols) {
                            const pos2 = @as(usize, @intFromFloat(y)) * self.image.cols + @as(usize, @intFromFloat(x_floor + 1));
                            var c1 = convert(Rgba, self.image.data[pos2]);
                            var c_blend = c2;
                            c_blend.a = @intFromFloat(x_frac * max_alpha);
                            c1.blend(c_blend);
                            self.image.data[pos2] = convert(T, c1);
                        }
                    }
                    x += gradient;
                }
            }
        }

        /// Draws a colored straight line of a custom width between `p1` and `p2` on `image` using Bresenham's line algorithm.
        /// This function is faster than `drawLine` because it does not perform anti-aliasing.
        pub fn drawLineFast(self: Self, p1: Point2d(f32), p2: Point2d(f32), width: usize, color: T) void {
            if (width == 0) return;
            var x1: isize = @intFromFloat(p1.x);
            var y1: isize = @intFromFloat(p1.y);
            const x2: isize = @intFromFloat(p2.x);
            const y2: isize = @intFromFloat(p2.y);
            const sx: isize = if (x1 < x2) 1 else -1;
            const sy: isize = if (y1 < y2) 1 else -1;
            const dx: isize = @intCast(@abs(x2 - x1));
            const dy: isize = @intCast(@abs(y2 - y1));
            var err = dx - dy;
            while (x1 != x2 or y1 != y2) {
                const half_width: isize = @intCast(width / 2);
                var i = -half_width;
                while (i <= half_width) : (i += 1) {
                    var j = -half_width;
                    while (j <= half_width) : (j += 1) {
                        const px = x1 + i;
                        const py = y1 + j;
                        if (px >= 0 and px < self.image.cols and py >= 0 and py < self.image.rows) {
                            const pos = @as(usize, @intCast(py)) * self.image.cols + @as(usize, @intCast(px));
                            self.image.data[pos] = color;
                        }
                    }
                }
                const e2 = err * 2;
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

        /// Draws a cubic Bézier curve on the given image.
        pub fn drawBezierCurve(self: Self, points: [4]Point2d(f32), step: f32, color: T) void {
            comptime assert(isColor(T));
            assert(step >= 0);
            assert(step <= 1);
            var t: f32 = 0;
            while (t <= 1) : (t += step) {
                const b: Point2d(f32) = .{
                    .x = (1 - t) * (1 - t) * (1 - t) * points[0].x + 3 * (1 - t) * (1 - t) * t * points[1].x + 3 * (1 - t) * t * t * points[2].x + t * t * t * points[3].x,
                    .y = (1 - t) * (1 - t) * (1 - t) * points[0].y + 3 * (1 - t) * (1 - t) * t * points[1].y + 3 * (1 - t) * t * t * points[2].y + t * t * t * points[3].y,
                };
                const row: usize = @intFromFloat(@round(b.y));
                const col: usize = @intFromFloat(@round(b.x));
                if (row < self.image.rows and col < self.image.cols) {
                    self.image.at(row, col).* = color;
                }
            }
        }

        /// Draws a smooth polygon on the given image.
        /// The polygon's edges are rendered as cubic Bézier curves connecting the vertices,
        /// allowing for a curved and smooth appearance.
        pub fn drawSmoothPolygon(self: Self, polygon: []const Point2d(f32), color: T, tension: f32) void {
            assert(tension >= 0);
            assert(tension <= 1);
            for (0..polygon.len) |i| {
                const p0 = polygon[i];
                const p1 = polygon[(i + 1) % polygon.len];
                const p2 = polygon[(i + 2) % polygon.len];
                const cp1 = Point2d(f32){
                    .x = p0.x + (p1.x - p0.x) * (1 - tension),
                    .y = p0.y + (p1.y - p0.y) * (1 - tension),
                };
                const cp2 = Point2d(f32){
                    .x = p1.x - (p2.x - p1.x) * (1 - tension),
                    .y = p1.y - (p2.y - p1.y) * (1 - tension),
                };
                self.drawBezierCurve(.{ p0, cp1, cp2, p1 }, 0.01, color);
            }
        }

        /// Fills a smooth polygon on the given image.
        /// The polygon's outline is defined by Bézier curves connecting the vertices (similar to `drawSmoothPolygon`),
        /// and the resulting shape is then filled with the specified color using a scanline algorithm.
        pub fn fillSmoothPolygon(self: Self, polygon: []const Point2d(f32), color: T, tension: f32) !void {
            var points: std.ArrayList(Point2d(f32)) = .init(self.allocator);
            for (0..polygon.len) |i| {
                const p0 = polygon[i];
                const p1 = polygon[(i + 1) % polygon.len];
                const p2 = polygon[(i + 2) % polygon.len];
                const cp1 = Point2d(f32){
                    .x = p0.x + (p1.x - p0.x) * (1 - tension),
                    .y = p0.y + (p1.y - p0.y) * (1 - tension),
                };
                const cp2 = Point2d(f32){
                    .x = p1.x - (p2.x - p1.x) * (1 - tension),
                    .y = p1.y - (p2.y - p1.y) * (1 - tension),
                };
                const segment = try self.tessellateCurve(.{ p0, cp1, cp2, p1 }, 10);
                try points.appendSlice(segment);
            }
            try self.fillPolygon(points.items, color);
        }

        /// Draws the outline of a rectangle on the given image.
        pub fn drawRectangle(self: Self, rect: Rectangle(f32), width: usize, color: anytype) !void {
            comptime assert(isColor(@TypeOf(color)));
            const points: []const Point2d(f32) = &.{
                .{ .x = rect.l, .y = rect.t },
                .{ .x = rect.r, .y = rect.t },
                .{ .x = rect.r, .y = rect.b },
                .{ .x = rect.l, .y = rect.b },
            };
            self.drawPolygon(points, width, color);
        }

        /// Draws a cross shape (plus sign) on the given image at a specified center point.
        pub fn drawCross(self: Self, center: Point2d(f32), size: usize, color: T) void {
            comptime assert(isColor(T));
            if (size == 0) return;
            const x: usize = @intFromFloat(@round(@max(0, @min(as(f32, self.image.cols - 1), center.x))));
            const y: usize = @intFromFloat(@round(@max(0, @min(as(f32, self.image.rows - 1), center.y))));
            for (0..size) |i| {
                self.image.data[y * self.image.cols + x -| i] = color;
                self.image.data[(y -| i) * self.image.cols + x] = color;
                self.image.data[y * self.image.cols + @min(self.image.cols - 1, x + i)] = color;
                self.image.data[@min(self.image.rows - 1, y + i) * self.image.cols + x] = color;
            }
        }

        /// Draws the outline of a polygon on the given image.
        /// The polygon is defined by a sequence of vertices. Lines are drawn between consecutive
        /// vertices, and a final line is drawn from the last vertex to the first to close the shape.
        /// Round joints are added at vertices to ensure smooth connections.
        pub fn drawPolygon(self: Self, polygon: []const Point2d(f32), width: usize, color: anytype) void {
            comptime assert(isColor(@TypeOf(color)));
            if (width == 0) return;

            // Draw all line segments
            for (0..polygon.len) |i| {
                self.drawLine(polygon[i], polygon[@mod(i + 1, polygon.len)], width, color);
            }

            // Fill vertices with circles to create round joints
            const radius: f32 = @floatFromInt((width + 2) / 2);
            for (polygon) |vertex| {
                self.fillCircle(vertex, radius, color);
            }
        }

        /// Draws the outline of a circle on the given image.
        pub fn drawCircle(self: Self, center: Point2d(f32), radius: f32, color: anytype) void {
            comptime assert(isColor(@TypeOf(color)));
            if (radius <= 0) return;
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
                    const col = @as(usize, @intFromFloat(p.x));
                    const row = @as(usize, @intFromFloat(p.y));
                    if (row < self.image.rows and col < self.image.cols) {
                        const pos = row * self.image.cols + col;
                        var c1 = convert(Rgba, self.image.data[pos]);
                        const c2 = convert(Rgba, color);
                        c1.blend(c2);
                        self.image.data[pos] = convert(T, c1);
                    }
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
        }

        /// Fills a circle on the given image.
        pub fn fillCircle(self: Self, center: Point2d(f32), radius: f32, color: anytype) void {
            comptime assert(isColor(@TypeOf(color)));
            if (radius <= 0) return;
            const frows: f32 = @floatFromInt(self.image.rows);
            const fcols: f32 = @floatFromInt(self.image.cols);
            const left: usize = @intFromFloat(@round(@max(0, center.x - radius)));
            const top: usize = @intFromFloat(@round(@max(0, center.y - radius)));
            const right: usize = @intFromFloat(@round(@min(fcols, center.x + radius)));
            const bottom: usize = @intFromFloat(@round(@min(frows, center.y + radius)));

            var c2 = convert(Rgba, color);
            const max_alpha: f32 = @floatFromInt(c2.a);

            for (top..bottom) |r| {
                const y = as(f32, r) - center.y;
                for (left..right) |c| {
                    const x = as(f32, c) - center.x;
                    const dist_sq = x * x + y * y;
                    if (dist_sq <= radius * radius) {
                        const pos = r * self.image.cols + c;

                        // Apply antialiasing at the edge
                        const dist = @sqrt(dist_sq);
                        if (dist > radius - 1) {
                            // Edge antialiasing
                            const edge_alpha = radius - dist;
                            var c1 = convert(Rgba, self.image.data[pos]);
                            c2.a = @intFromFloat(edge_alpha * max_alpha);
                            c1.blend(c2);
                            self.image.data[pos] = convert(T, c1);
                        } else {
                            // Full opacity in the center
                            var c1 = convert(Rgba, self.image.data[pos]);
                            c2.a = @intFromFloat(max_alpha);
                            c1.blend(c2);
                            self.image.data[pos] = convert(T, c1);
                        }
                    }
                }
            }
        }

        /// Fills the given polygon on an image using the scanline algorithm.
        /// The polygon is defined by an array of points (vertices).
        pub fn fillPolygon(self: Self, polygon: []const Point2d(f32), color: anytype) !void {
            comptime assert(isColor(@TypeOf(color)));
            const rows = self.image.rows;
            const cols = self.image.cols;
            var intersections: std.ArrayList(f32) = .init(self.allocator);
            defer intersections.deinit();

            var c2 = convert(Rgba, color);
            const max_alpha: f32 = @floatFromInt(c2.a);

            for (0..rows) |r| {
                const y: f32 = @floatFromInt(r);
                intersections.clearRetainingCapacity();
                for (0..polygon.len) |i| {
                    const p1 = &polygon[i];
                    const p2 = &polygon[(i + 1) % polygon.len];
                    if (p1.y == p2.y) {
                        continue;
                    }
                    if ((y <= p1.y and y <= p2.y) or (y > p1.y and y > p2.y)) {
                        continue;
                    }
                    try intersections.append(p1.x + (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y));
                }
                std.mem.sort(f32, intersections.items, {}, std.sort.asc(f32));
                var i: usize = 1;
                while (i < intersections.items.len) : (i += 2) {
                    const left_edge = intersections.items[i - 1];
                    const right_edge = intersections.items[i];
                    var left_x: i32 = @intFromFloat(@floor(left_edge));
                    var right_x: i32 = @intFromFloat(@ceil(right_edge));
                    left_x = @max(0, left_x);
                    right_x = @min(as(i32, cols - 1), right_x);

                    var x: i32 = left_x;
                    while (x <= right_x) : (x += 1) {
                        const fx = as(f32, x);
                        const pos: usize = r * cols + as(usize, x);

                        // Apply antialiasing at edges
                        var alpha: f32 = 1.0;
                        if (fx < left_edge + 1) {
                            alpha = @min(alpha, fx + 0.5 - left_edge);
                        }
                        if (fx > right_edge - 1) {
                            alpha = @min(alpha, right_edge - (fx - 0.5));
                        }
                        alpha = @max(0, @min(1, alpha));

                        if (alpha > 0) {
                            var c1 = convert(Rgba, self.image.data[pos]);
                            c2.a = @intFromFloat(alpha * max_alpha);
                            c1.blend(c2);
                            self.image.data[pos] = convert(T, c1);
                        }
                    }
                }
            }
        }

        /// Tessellates a cubic Bézier curve into a series of line segments (points).
        fn tessellateCurve(self: Self, p: [4]Point2d(f32), segments: usize) anyerror![]const Point2d(f32) {
            var polygon: std.ArrayList(Point2d(f32)) = .init(self.allocator);
            for (0..segments) |i| {
                const t: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(segments));
                const u: f32 = 1 - t;
                const tt: f32 = t * t;
                const uu: f32 = u * u;
                const uuu: f32 = uu * u;
                const ttt: f32 = tt * t;
                try polygon.append(.{
                    .x = uuu * p[0].x + 3 * uu * t * p[1].x + 3 * u * tt * p[2].x + ttt * p[3].x,
                    .y = uuu * p[0].y + 3 * uu * t * p[1].y + 3 * u * tt * p[2].y + ttt * p[3].y,
                });
            }
            return try polygon.toOwnedSlice();
        }
    };
}
