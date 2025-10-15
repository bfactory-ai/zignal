//! This module provides a Canvas for drawing various shapes and lines on images.

const std = @import("std");
const assert = std.debug.assert;

const convertColor = @import("../color.zig").convertColor;
const isColor = @import("../color.zig").isColor;
const Rgba = @import("../color.zig").Rgba;
const Blending = @import("../color.zig").Blending;
const BitmapFont = @import("../font.zig").BitmapFont;
const Rectangle = @import("../geometry.zig").Rectangle;
const Point = @import("../geometry/Point.zig").Point;
const Image = @import("../image.zig").Image;
const assignPixel = @import("../image.zig").assignPixel;
const as = @import("../meta.zig").as;

/// Rendering quality mode for drawing operations
pub const DrawMode = enum {
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

        /// Creates a drawing canvas from an image.
        pub fn init(allocator: std.mem.Allocator, image: Image(T)) Self {
            return .{ .image = image, .allocator = allocator };
        }

        /// Clamps a floating-point coordinate to image bounds and converts to usize.
        /// Returns the clamped coordinate as a usize index.
        inline fn clampToImageBounds(coord: f32, max_size: usize) usize {
            return @intFromFloat(@max(0, @min(@as(f32, @floatFromInt(max_size)), coord)));
        }

        /// Clamps a rectangle to image bounds and returns integer pixel coordinates.
        /// Returns null if the rectangle is completely outside the image.
        inline fn clampRectToImage(self: Self, rect: Rectangle(f32)) ?Rectangle(usize) {
            const left = clampToImageBounds(rect.l, self.image.cols);
            const top = clampToImageBounds(rect.t, self.image.rows);
            const right = clampToImageBounds(rect.r, self.image.cols);
            const bottom = clampToImageBounds(rect.b, self.image.rows);

            // Check if rectangle is valid after clamping
            if (left >= right or top >= bottom) {
                return null;
            }

            return .{ .l = left, .t = top, .r = right, .b = bottom };
        }

        /// Fills the entire canvas with a solid color using @memset.
        pub fn fill(self: Self, color: anytype) void {
            self.image.fill(convertColor(T, color));
        }

        /// Gets a reference to the pixel at the given coordinates.
        /// Panics if coordinates are out of bounds.
        pub inline fn at(self: Self, row: usize, col: usize) *T {
            return self.image.at(row, col);
        }

        /// Gets a reference to the pixel at the given coordinates, or null if out of bounds.
        pub inline fn atOrNull(self: Self, row: isize, col: isize) ?*T {
            return self.image.atOrNull(row, col);
        }

        /// Returns the number of rows (height) in the canvas.
        pub inline fn rows(self: Self) usize {
            return self.image.rows;
        }

        /// Returns the number of columns (width) in the canvas.
        pub inline fn cols(self: Self) usize {
            return self.image.cols;
        }

        /// Returns the total number of pixels in the canvas (rows * cols).
        pub inline fn size(self: Self) usize {
            return self.image.size();
        }

        /// Returns true if and only if this canvas and `other` have the same number of rows and columns.
        /// It does not compare pixel data or types.
        pub inline fn hasSameShape(self: Self, other: anytype) bool {
            return self.image.hasSameShape(other.image);
        }

        /// Normalizes an angle to the [0, 2π] range.
        inline fn normalizeAngle(angle: f32) f32 {
            var normalized = @mod(angle, 2 * std.math.pi);
            if (normalized < 0) normalized += 2 * std.math.pi;
            return normalized;
        }

        /// Checks if an angle is within an arc's range.
        /// Arcs are drawn counter-clockwise from start_angle to end_angle.
        /// Handles wrapping around 2π correctly.
        inline fn isAngleInArc(angle: f32, start: f32, end: f32) bool {
            // Normalize all angles to [0, 2π] range
            const norm_angle = normalizeAngle(angle);
            const norm_start = normalizeAngle(start);
            var norm_end = normalizeAngle(end);

            // If the arc spans a full circle or more, include all angles
            if (@abs(end - start) >= 2 * std.math.pi) {
                return true;
            }

            // Ensure we go counter-clockwise from start to end
            // If end is less than start after normalization, it wraps around 0
            if (norm_end < norm_start) {
                norm_end += 2 * std.math.pi;
            }

            // Check if angle is in range, handling wrap-around
            if (norm_angle >= norm_start and norm_angle <= norm_end) {
                return true;
            }
            // Also check with angle shifted by 2π for wrap-around cases
            const shifted_angle = norm_angle + 2 * std.math.pi;
            return shifted_angle >= norm_start and shifted_angle <= norm_end;
        }

        /// Creates a view (sub-canvas) of this canvas within the specified rectangle.
        /// The view shares memory with the parent canvas - changes are reflected in both.
        /// Coordinates are automatically clipped to the canvas bounds.
        pub fn view(self: Self, rect: Rectangle(usize)) Self {
            return .{
                .image = self.image.view(rect),
                .allocator = self.allocator,
            };
        }

        /// Sets a horizontal span to `color` using @memset.
        pub fn setHorizontalSpan(self: Self, x1: f32, x2: f32, y: f32, color: T) void {
            const frows: f32 = @floatFromInt(self.image.rows);
            const fcols: f32 = @floatFromInt(self.image.cols);

            if (y < 0 or y >= frows) return;
            if (x2 < 0 or x1 >= fcols) return;

            const row: usize = @intFromFloat(y);
            const start: usize = @intFromFloat(@max(0, @floor(x1)));
            const end: usize = @intFromFloat(@min(fcols - 1, @ceil(x2)));

            if (start > end) return;

            const offset = row * self.image.stride + start;
            const len = end - start + 1;
            @memset(self.image.data[offset .. offset + len], color);
        }

        /// Draws a line between two points with configurable width and rendering quality.
        ///
        /// Algorithm selection:
        /// - Width 1, fast mode: Bresenham's algorithm (pixel-perfect, no antialiasing)
        /// - Width 1, soft mode: Xiaolin Wu's algorithm (smooth antialiasing)
        /// - Width >1, fast mode: Rectangle with circular caps (solid rendering)
        /// - Width >1, soft mode: Distance-based antialiasing (superior quality)
        ///
        /// Parameters:
        /// - p1, p2: Line endpoints in floating-point coordinates
        /// - color: Any color type (Rgba colors support alpha blending)
        /// - width: Line thickness in pixels (0 = no drawing)
        /// - mode: .fast (performance) or .soft (quality with antialiasing)
        pub fn drawLine(self: Self, p1: Point(2, f32), p2: Point(2, f32), color: anytype, width: usize, mode: DrawMode) void {
            comptime assert(isColor(@TypeOf(color)));
            if (width == 0) return;

            switch (mode) {
                .fast => self.drawLineFast(p1, p2, width, color),
                .soft => self.drawLineSoft(p1, p2, width, color),
            }
        }

        /// Internal dispatcher for fast (non-antialiased) line rendering.
        /// - Width 1: Uses Bresenham's algorithm for pixel-perfect precision
        /// - Width >1: Uses rectangle-based approach with circular end caps
        fn drawLineFast(self: Self, p1: Point(2, f32), p2: Point(2, f32), width: usize, color: anytype) void {
            if (width == 1) {
                // Use Bresenham's algorithm for 1px lines - fast and precise
                self.drawLineBresenham(p1, p2, color);
            } else {
                // Use polygon-based approach for thick lines
                self.drawLineRectangle(p1, p2, width, color);
            }
        }
        /// Internal dispatcher for soft (antialiased) line rendering.
        /// - Width 1: Uses Xiaolin Wu's algorithm for optimal thin line antialiasing
        /// - Width >1: Uses distance-based algorithm for superior thick line quality
        fn drawLineSoft(self: Self, p1: Point(2, f32), p2: Point(2, f32), width: usize, color: anytype) void {
            if (width == 1) {
                // Use Wu's algorithm for 1px lines - optimal antialiasing and performance
                self.drawLineXiaolinWu(p1, p2, color);
            } else {
                // Use distance-based antialiasing for thick lines - better quality
                self.drawLineDistance(p1, p2, width, color);
            }
        }

        /// Bresenham's line algorithm for 1-pixel width lines.
        /// Classic rasterization algorithm using integer arithmetic for maximum speed.
        /// Produces pixel-perfect lines with hard edges and no antialiasing.
        /// Optimal for grid-aligned graphics and when performance is critical.
        fn drawLineBresenham(self: Self, p1: Point(2, f32), p2: Point(2, f32), color: anytype) void {
            var x1: i32 = @intFromFloat(p1.x());
            var y1: i32 = @intFromFloat(p1.y());
            const x2: i32 = @intFromFloat(p2.x());
            const y2: i32 = @intFromFloat(p2.y());

            const pixel_color = convertColor(T, color);

            // Special case for horizontal lines - use fillHorizontalSpan for better performance
            if (y1 == y2) {
                const min_x = @min(x1, x2);
                const max_x = @max(x1, x2);
                self.setHorizontalSpan(@floatFromInt(min_x), @floatFromInt(max_x), @floatFromInt(y1), pixel_color);
                return;
            }

            // Special case for vertical lines - direct pixel access
            if (x1 == x2) {
                const min_y = @min(y1, y2);
                const max_y = @max(y1, y2);
                var y = min_y;
                while (y <= max_y) : (y += 1) {
                    if (self.atOrNull(y, x1)) |pixel| {
                        pixel.* = pixel_color;
                    }
                }
                return;
            }

            // General case - standard Bresenham algorithm
            const dx: i32 = @intCast(@abs(x2 - x1));
            const dy: i32 = @intCast(@abs(y2 - y1));
            const sx: i32 = if (x1 < x2) 1 else -1;
            const sy: i32 = if (y1 < y2) 1 else -1;
            var err = dx - dy;

            while (true) {
                if (self.atOrNull(y1, x1)) |pixel| {
                    pixel.* = pixel_color;
                }

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

        /// Xiaolin Wu's antialiasing algorithm for 1-pixel width lines.
        /// Uses fractional coverage to create smooth line edges with alpha blending.
        /// Handles steep vs. shallow lines optimally by swapping coordinates.
        /// Provides the best quality-to-performance ratio for thin antialiased lines.
        fn drawLineXiaolinWu(self: Self, p1: Point(2, f32), p2: Point(2, f32), color: anytype) void {
            const c2 = convertColor(Rgba, color);

            var x1 = p1.x();
            var y1 = p1.y();
            var x2 = p2.x();
            var y2 = p2.y();

            // Special case for perfectly horizontal lines
            if (@abs(y2 - y1) < 0.01) {
                const y = @round(y1);
                const min_x = @min(x1, x2);
                const max_x = @max(x1, x2);

                // Handle fractional endpoints with antialiasing
                const left_x = @floor(min_x);
                const right_x = @ceil(max_x);

                // Left endpoint antialiasing
                if (min_x > left_x) {
                    const alpha = min_x - left_x;
                    self.setPixel(.init(.{ left_x, y }), c2.fade(alpha));
                }

                // Middle solid part - use fillHorizontalSpan for performance
                const solid_start = @ceil(min_x);
                const solid_end = @floor(max_x);
                if (solid_end >= solid_start) {
                    self.setHorizontalSpan(solid_start, solid_end, y, convertColor(T, c2));
                }

                // Right endpoint antialiasing
                if (max_x < right_x) {
                    const alpha = right_x - max_x;
                    self.setPixel(.init(.{ right_x, y }), c2.fade(alpha));
                }

                return;
            }

            const steep = @abs(y2 - y1) > @abs(x2 - x1);
            if (steep) {
                std.mem.swap(f32, &x1, &y1);
                std.mem.swap(f32, &x2, &y2);
            }
            if (x1 > x2) {
                std.mem.swap(f32, &x1, &x2);
                std.mem.swap(f32, &y1, &y2);
            }

            const dx = x2 - x1;
            const dy = y2 - y1;
            const gradient = if (dx == 0) 1.0 else dy / dx;

            // Handle endpoints
            const x_end = @round(x1);
            const y_end = y1 + gradient * (x_end - x1);

            // Draw the actual endpoint pixels at full intensity
            if (steep) {
                self.setPixel(.init(.{ @round(y1), @round(x1) }), c2);
            } else {
                self.setPixel(.init(.{ @round(x1), @round(y1) }), c2);
            }
            var intery = y_end + gradient;

            // Draw the actual endpoint pixels at full intensity
            if (steep) {
                self.setPixel(.init(.{ @round(y2), @round(x2) }), c2);
            } else {
                self.setPixel(.init(.{ @round(x2), @round(y2) }), c2);
            }

            // Main loop
            const x_px1 = @round(x1);
            const x_px2 = @round(x2);
            var x = x_px1 + 1;
            while (x < x_px2) : (x += 1) {
                if (steep) {
                    self.setPixel(.init(.{ intery, x }), c2.fade(rfpart(intery)));
                    self.setPixel(.init(.{ @floor(intery) + 1, x }), c2.fade(fpart(intery)));
                } else {
                    self.setPixel(.init(.{ x, intery }), c2.fade(rfpart(intery)));
                    self.setPixel(.init(.{ x, @floor(intery) + 1 }), c2.fade(fpart(intery)));
                }
                intery += gradient;
            }
        }

        /// Rectangle-based thick line rendering for fast (non-antialiased) mode.
        /// Constructs a filled rectangle perpendicular to the line direction,
        /// then adds circular end caps for smooth line termination.
        /// Handles zero-length lines by drawing a single filled circle.
        fn drawLineRectangle(self: Self, p1: Point(2, f32), p2: Point(2, f32), width: usize, color: anytype) void {
            const solid_color = convertColor(T, color);

            // For thick lines, draw as a filled rectangle
            const dx = p2.x() - p1.x();
            const dy = p2.y() - p1.y();
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
            const corners = [_]Point(2, f32){
                .init(.{ p1.x() - perp_x, p1.y() - perp_y }),
                .init(.{ p1.x() + perp_x, p1.y() + perp_y }),
                .init(.{ p2.x() + perp_x, p2.y() + perp_y }),
                .init(.{ p2.x() - perp_x, p2.y() - perp_y }),
            };

            // Fill rectangle using scanline algorithm (no anti-aliasing)
            self.fillPolygon(&corners, solid_color, .fast) catch return;

            // Add rounded caps using solid circles
            self.fillCircle(p1, half_width, color, .fast);
            self.fillCircle(p2, half_width, color, .fast);
        }

        /// Distance-based antialiased rendering for thick lines.
        /// Calculates the exact perpendicular distance from each pixel to the line segment,
        /// applying smooth alpha falloff at edges for superior visual quality.
        /// Includes optimized paths for horizontal/vertical lines and handles end caps naturally.
        /// More expensive than rectangle-based approach but produces better results.
        fn drawLineDistance(self: Self, p1: Point(2, f32), p2: Point(2, f32), width: usize, color: anytype) void {
            const frows: f32 = @floatFromInt(self.image.rows);
            const fcols: f32 = @floatFromInt(self.image.cols);
            const half_width: f32 = @as(f32, @floatFromInt(width)) / 2.0;
            const c2 = convertColor(Rgba, color);

            // Calculate line direction vector
            const dx = p2.x() - p1.x();
            const dy = p2.y() - p1.y();
            const line_length = @sqrt(dx * dx + dy * dy);

            if (line_length == 0) {
                // Single point - draw a small circle
                self.fillCircle(p1, half_width, color, .soft);
                return;
            }

            // Special case for perfectly horizontal/vertical lines (faster rendering)
            if (@abs(dx) < horizontal_vertical_threshold) { // Vertical line
                const x1 = @round(p1.x());
                var y1 = @round(p1.y());
                var y2 = @round(p2.y());
                if (y1 > y2) std.mem.swap(f32, &y1, &y2);
                if (x1 < 0 or x1 >= fcols) return;

                const pixel_color = convertColor(T, c2);
                var y = y1;
                while (y <= y2) : (y += 1) {
                    if (y < 0 or y >= frows) continue;
                    // Use fillHorizontalSpan for each horizontal segment of the thick vertical line
                    const x_start = x1 - half_width;
                    const x_end = x1 + half_width;
                    self.setHorizontalSpan(x_start, x_end, y, pixel_color);
                }
                // Add rounded caps
                self.fillCircle(p1, half_width, color, .soft);
                self.fillCircle(p2, half_width, color, .soft);
                return;
            } else if (@abs(dy) < horizontal_vertical_threshold) { // Horizontal line
                var x1 = @round(p1.x());
                var x2 = @round(p2.x());
                const y1 = @round(p1.y());
                if (x1 > x2) std.mem.swap(f32, &x1, &x2);
                if (y1 < 0 or y1 >= frows) return;

                const pixel_color = convertColor(T, c2);

                // Draw horizontal spans for each row of the thick line
                var i = -half_width;
                while (i <= half_width) : (i += 1) {
                    const py = y1 + i;
                    if (py >= 0 and py < frows) {
                        // For soft mode, we need to handle edge pixels with alpha blending
                        if (i == -half_width or i == half_width) {
                            // Edge rows - use setPixel for alpha blending
                            var x = x1;
                            while (x <= x2) : (x += 1) {
                                if (x >= 0 and x < fcols) {
                                    self.setPixel(.init(.{ x, py }), c2);
                                }
                            }
                        } else {
                            // Middle rows - use fillHorizontalSpan for performance
                            self.setHorizontalSpan(x1, x2, py, pixel_color);
                        }
                    }
                }
                // Add rounded caps
                self.fillCircle(p1, half_width, color, .soft);
                self.fillCircle(p2, half_width, color, .soft);
                return;
            }

            // For diagonal lines, use optimized distance-based anti-aliasing
            // Calculate tighter bounding box
            const line_min_x = @min(p1.x(), p2.x()) - half_width;
            const line_max_x = @max(p1.x(), p2.x()) + half_width;
            const line_min_y = @min(p1.y(), p2.y()) - half_width;
            const line_max_y = @max(p1.y(), p2.y()) + half_width;

            const min_x = @max(0, @floor(line_min_x));
            const max_x = @min(fcols - 1, @ceil(line_max_x));
            const min_y = @max(0, @floor(line_min_y));
            const max_y = @min(frows - 1, @ceil(line_max_y));

            // Precompute for distance calculation optimization
            const dx_sq = dx * dx;
            const dy_sq = dy * dy;
            const length_sq = dx_sq + dy_sq;
            const inv_length_sq = 1.0 / length_sq;

            // Iterate through pixels in bounding box
            var py: f32 = min_y;
            while (py <= max_y) : (py += 1) {
                var px: f32 = min_x;
                while (px <= max_x) : (px += 1) {
                    // Optimized distance calculation
                    const dpx = px - p1.x();
                    const dpy = py - p1.y();
                    const t = @max(0, @min(1, (dpx * dx + dpy * dy) * inv_length_sq));
                    const closest_x = p1.x() + t * dx;
                    const closest_y = p1.y() + t * dy;
                    const dist_x = px - closest_x;
                    const dist_y = py - closest_y;
                    const dist = @sqrt(dist_x * dist_x + dist_y * dist_y);

                    // Anti-aliased coverage based on distance
                    if (dist <= half_width + antialias_edge_offset) {
                        var alpha: f32 = 1.0;
                        if (dist > half_width - antialias_edge_offset) {
                            alpha = (half_width + antialias_edge_offset - dist);
                        }

                        if (alpha > 0) {
                            self.setPixel(.init(.{ px, py }), c2.fade(alpha));
                        }
                    }
                }
            }
        }

        /// Sets a color to a pixel at the given coordinates with alpha transparency.
        /// Uses optimized direct assignment for opaque colors or blends when transparency is needed.
        /// Provides bounds checking and handles coordinate conversion.
        /// Coordinates are truncated to integers for pixel placement.
        /// For Rgba colors, uses the color's alpha channel; for other colors, treats as opaque.
        pub fn setPixel(self: Self, point: Point(2, f32), color: anytype) void {
            const ColorType = @TypeOf(color);
            comptime assert(isColor(ColorType));
            const row: isize = @intFromFloat(@floor(point.y()));
            const col: isize = @intFromFloat(@floor(point.x()));
            if (self.atOrNull(row, col)) |pixel| {
                if (comptime ColorType == Rgba) {
                    const mode: Blending = if (color.a == 255) .none else .normal;
                    assignPixel(pixel, color, mode);
                } else {
                    const converted = convertColor(T, color);
                    assignPixel(pixel, converted, .none);
                }
            }
        }

        /// Draws another image onto this canvas at the given top-left position.
        /// Supports alpha blending for RGBA images with the normal blend mode.
        /// For rotation, scaling, or custom blend modes, users should access the canvas's image field directly.
        pub fn drawImage(self: Self, source: anytype, position: Point(2, f32), source_rect_opt: ?Rectangle(usize), blend_mode: Blending) void {
            const SourcePixelType = std.meta.Child(@TypeOf(source.data));

            if (source.rows == 0 or source.cols == 0) return;

            const SourceRect = Rectangle(usize);
            const full_rect = SourceRect.init(0, 0, source.cols, source.rows);
            const requested = source_rect_opt orelse full_rect;
            const src_rect = full_rect.intersect(requested) orelse return;

            if (src_rect.isEmpty()) return;

            const origin_x = @as(isize, @intFromFloat(@round(position.x())));
            const origin_y = @as(isize, @intFromFloat(@round(position.y())));

            // Simple blit loop with type-based blending
            for (src_rect.t..src_rect.b) |src_r| {
                const row_offset = src_r - src_rect.t;
                const dest_y = origin_y + @as(isize, @intCast(row_offset));

                for (src_rect.l..src_rect.r) |src_c| {
                    const col_offset = src_c - src_rect.l;
                    const dest_x = origin_x + @as(isize, @intCast(col_offset));

                    if (self.atOrNull(dest_y, dest_x)) |dest_pixel| {
                        const src_pixel = source.at(src_r, src_c).*;
                        if (comptime SourcePixelType == Rgba) {
                            assignPixel(dest_pixel, src_pixel, blend_mode);
                        } else {
                            const converted = if (SourcePixelType == T)
                                src_pixel
                            else
                                convertColor(T, src_pixel);
                            assignPixel(dest_pixel, converted, .none);
                        }
                    }
                }
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
        pub fn drawPolygon(self: Self, polygon: []const Point(2, f32), color: anytype, width: usize, mode: DrawMode) void {
            comptime assert(isColor(@TypeOf(color)));
            if (width == 0) return;

            // Draw all line segments
            for (0..polygon.len) |i| {
                self.drawLine(polygon[i], polygon[@mod(i + 1, polygon.len)], color, width, mode);
            }
        }

        /// Draws the outline of a rectangle on the given image.
        pub fn drawRectangle(self: Self, rect: Rectangle(f32), color: anytype, width: usize, mode: DrawMode) void {
            comptime assert(isColor(@TypeOf(color)));
            // Rectangle has exclusive r,b bounds, but drawPolygon needs inclusive points
            // So we subtract 1 from r and b to get the actual corner positions
            const points: []const Point(2, f32) = &.{
                .init(.{ rect.l, rect.t }),
                .init(.{ rect.r - 1, rect.t }),
                .init(.{ rect.r - 1, rect.b - 1 }),
                .init(.{ rect.l, rect.b - 1 }),
            };
            self.drawPolygon(points, color, width, mode);
        }

        /// Fills a rectangle on the given image.
        /// The rectangle is defined using standard conventions where l,t are inclusive and r,b are exclusive.
        /// This means a rectangle from (0,0) to (10,10) will fill pixels at positions 0-9 in both dimensions.
        /// - **DrawMode.fast**: Uses @memset for optimal performance (no alpha blending)
        /// - **DrawMode.soft**: Supports alpha blending by using setPixel for each pixel
        pub fn fillRectangle(self: Self, rect: Rectangle(f32), color: anytype, mode: DrawMode) void {
            comptime assert(isColor(@TypeOf(color)));

            // Use helper to clamp rectangle to image bounds
            const bounds = self.clampRectToImage(rect) orelse return;

            switch (mode) {
                .fast => {
                    // Fast mode: Use @memset for optimal performance (no alpha blending)
                    const target_color = convertColor(T, color);
                    for (bounds.t..bounds.b) |row| {
                        const start_idx = row * self.image.stride + bounds.l;
                        const len = bounds.r - bounds.l;
                        @memset(self.image.data[start_idx .. start_idx + len], target_color);
                    }
                },
                .soft => {
                    // Soft mode: Support alpha blending by using setPixel for each pixel
                    for (bounds.t..bounds.b) |row| {
                        for (bounds.l..bounds.r) |col| {
                            self.setPixel(.init(.{ @as(f32, @floatFromInt(col)), @as(f32, @floatFromInt(row)) }), color);
                        }
                    }
                },
            }
        }

        /// Draws the outline of a circle on the given image.
        /// Use DrawMode.soft for anti-aliased edges or DrawMode.fast for fast aliased edges.
        pub fn drawCircle(self: Self, center: Point(2, f32), radius: f32, color: anytype, width: usize, mode: DrawMode) void {
            comptime assert(isColor(@TypeOf(color)));
            if (radius <= 0 or width == 0) return;

            switch (mode) {
                .fast => self.drawCircleFast(center, radius, width, color),
                .soft => self.drawCircleSoft(center, radius, width, color),
            }
        }

        /// Draws an arc (portion of a circle outline) with the specified parameters.
        ///
        /// **Parameters:**
        /// - `center`: The center point of the arc
        /// - `radius`: The radius of the arc (must be positive)
        /// - `start_angle`: Starting angle in radians (0 = positive X-axis)
        /// - `end_angle`: Ending angle in radians
        /// - `color`: The color to draw with (any color type)
        /// - `width`: Line thickness in pixels (0 = no drawing)
        /// - `mode`: DrawMode.fast for aliased or DrawMode.soft for anti-aliased
        ///
        /// **Angle Convention:**
        /// - Angles are measured from the positive X-axis
        /// - Positive angles rotate counter-clockwise
        /// - Angles can be negative or > 2π (automatically normalized)
        ///
        /// **Performance:**
        /// - Full circles (angle diff ≥ 2π) use optimized circle drawing
        /// - Fast mode: O(r) for thin lines, O(r²) for thick lines
        /// - Soft mode: Uses polygon tessellation, O(arc_length)
        ///
        /// **Example:**
        /// ```zig
        /// // Draw a red quarter arc from 0 to π/2 (90 degrees)
        /// try canvas.drawArc(center, 50, 0, std.math.pi / 2.0, Rgb.red, 2, .soft);
        /// ```
        pub fn drawArc(self: Self, center: Point(2, f32), radius: f32, start_angle: f32, end_angle: f32, color: anytype, width: usize, mode: DrawMode) !void {
            comptime assert(isColor(@TypeOf(color)));
            if (radius <= 0 or width == 0) return;

            // Validate angles are finite numbers
            if (!std.math.isFinite(start_angle) or !std.math.isFinite(end_angle)) {
                return;
            }

            // Check if this is a full circle (optimize for this case)
            const angle_diff = @abs(end_angle - start_angle);
            if (angle_diff >= 2 * std.math.pi) {
                // Full circle - use optimized circle drawing
                switch (mode) {
                    .fast => self.drawCircleFast(center, radius, width, color),
                    .soft => self.drawCircleSoft(center, radius, width, color),
                }
                return;
            }

            // Partial arc
            switch (mode) {
                .fast => self.drawArcFast(center, radius, start_angle, end_angle, width, color),
                .soft => try self.drawArcSoft(center, radius, start_angle, end_angle, width, color),
            }
        }

        /// Internal function for drawing solid (aliased) circle outlines.
        fn drawCircleFast(self: Self, center: Point(2, f32), radius: f32, width: usize, color: anytype) void {
            if (width == 1) {
                // Use fast Bresenham for 1-pixel width
                const cx = @round(center.x());
                const cy = @round(center.y());
                const r = @round(radius);
                var x: f32 = r;
                var y: f32 = 0;
                var err: f32 = 0;
                while (x >= y) {
                    const points = [_]Point(2, f32){
                        .init(.{ cx + x, cy + y }),
                        .init(.{ cx - x, cy + y }),
                        .init(.{ cx + x, cy - y }),
                        .init(.{ cx - x, cy - y }),
                        .init(.{ cx + y, cy + x }),
                        .init(.{ cx - y, cy + x }),
                        .init(.{ cx + y, cy - x }),
                        .init(.{ cx - y, cy - x }),
                    };
                    for (points) |p| {
                        self.setPixel(p, color);
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
                const solid_color = convertColor(T, color);

                // Calculate bounding box
                const left: usize = @intFromFloat(@round(@max(0, center.x() - outer_radius - 1)));
                const top: usize = @intFromFloat(@round(@max(0, center.y() - outer_radius - 1)));
                const right: usize = @intFromFloat(@round(@min(fcols, center.x() + outer_radius + 1)));
                const bottom: usize = @intFromFloat(@round(@min(frows, center.y() + outer_radius + 1)));

                for (top..bottom) |r| {
                    const y = @as(f32, @floatFromInt(r)) - center.y();
                    for (left..right) |c| {
                        const x = @as(f32, @floatFromInt(c)) - center.x();
                        const dist_sq = x * x + y * y;
                        const inner_radius_sq = inner_radius * inner_radius;
                        const outer_radius_sq = outer_radius * outer_radius;

                        if (dist_sq >= inner_radius_sq and dist_sq <= outer_radius_sq) {
                            const pos = r * self.image.stride + c;
                            self.image.data[pos] = solid_color;
                        }
                    }
                }
            }
        }

        /// Internal function for drawing smooth (anti-aliased) circle outlines.
        fn drawCircleSoft(self: Self, center: Point(2, f32), radius: f32, width: usize, color: anytype) void {
            const frows: f32 = @floatFromInt(self.image.rows);
            const fcols: f32 = @floatFromInt(self.image.cols);
            const line_width: f32 = @floatFromInt(width);
            const inner_radius = radius - line_width / 2.0;
            const outer_radius = radius + line_width / 2.0;

            // Calculate bounding box
            const left: usize = @intFromFloat(@round(@max(0, center.x() - outer_radius - 1)));
            const top: usize = @intFromFloat(@round(@max(0, center.y() - outer_radius - 1)));
            const right: usize = @intFromFloat(@round(@min(fcols, center.x() + outer_radius + 1)));
            const bottom: usize = @intFromFloat(@round(@min(frows, center.y() + outer_radius + 1)));

            const c2 = convertColor(Rgba, color);

            for (top..bottom) |r| {
                const y = @as(f32, @floatFromInt(r)) - center.y();
                for (left..right) |c| {
                    const x = @as(f32, @floatFromInt(c)) - center.x();
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
                            self.setPixel(.init(.{ @as(f32, @floatFromInt(c)), @as(f32, @floatFromInt(r)) }), c2.fade(alpha));
                        }
                    }
                }
            }
        }

        /// Internal function for drawing solid (aliased) arc outlines.
        fn drawArcFast(self: Self, center: Point(2, f32), radius: f32, start_angle: f32, end_angle: f32, width: usize, color: anytype) void {
            if (width == 1) {
                // Use modified Bresenham for 1-pixel width arcs
                const cx = @round(center.x());
                const cy = @round(center.y());
                const r = @round(radius);
                var x: f32 = r;
                var y: f32 = 0;
                var err: f32 = 0;

                while (x >= y) {
                    // Calculate points in all 8 octants
                    const points = [_]struct { px: f32, py: f32 }{
                        .{ .px = cx + x, .py = cy + y },
                        .{ .px = cx - x, .py = cy + y },
                        .{ .px = cx + x, .py = cy - y },
                        .{ .px = cx - x, .py = cy - y },
                        .{ .px = cx + y, .py = cy + x },
                        .{ .px = cx - y, .py = cy + x },
                        .{ .px = cx + y, .py = cy - x },
                        .{ .px = cx - y, .py = cy - x },
                    };

                    for (points) |pt| {
                        // Check if this point is within the arc's angle range
                        const angle = std.math.atan2(pt.py - cy, pt.px - cx);
                        if (isAngleInArc(angle, start_angle, end_angle)) {
                            self.setPixel(.init(.{ pt.px, pt.py }), color);
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
            } else {
                // Use ring filling for thick arc outlines
                const frows: f32 = @floatFromInt(self.image.rows);
                const fcols: f32 = @floatFromInt(self.image.cols);
                const line_width: f32 = @floatFromInt(width);
                const inner_radius = radius - line_width / 2.0;
                const outer_radius = radius + line_width / 2.0;

                // Calculate bounding box
                const left: usize = @intFromFloat(@round(@max(0, center.x() - outer_radius - 1)));
                const top: usize = @intFromFloat(@round(@max(0, center.y() - outer_radius - 1)));
                const right: usize = @intFromFloat(@round(@min(fcols, center.x() + outer_radius + 1)));
                const bottom: usize = @intFromFloat(@round(@min(frows, center.y() + outer_radius + 1)));

                if (left >= right or top >= bottom) return;

                for (top..bottom) |r| {
                    const y = @as(f32, @floatFromInt(r)) - center.y();
                    for (left..right) |c| {
                        const x = @as(f32, @floatFromInt(c)) - center.x();
                        const dist = @sqrt(x * x + y * y);

                        // Check if in ring and within arc angle
                        if (dist >= inner_radius and dist <= outer_radius) {
                            const angle = std.math.atan2(y, x);
                            if (isAngleInArc(angle, start_angle, end_angle)) {
                                self.setPixel(.init(.{ @as(f32, @floatFromInt(c)), @as(f32, @floatFromInt(r)) }), color);
                            }
                        }
                    }
                }
            }
        }

        /// Internal function for drawing smooth (anti-aliased) arc outlines.
        fn drawArcSoft(self: Self, center: Point(2, f32), radius: f32, start_angle: f32, end_angle: f32, width: usize, color: anytype) !void {
            // Generate polygon approximation of the arc
            const angle_span = end_angle - start_angle;
            const arc_length = @abs(angle_span) * radius;

            // Determine number of segments based on arc length
            const segments = @max(@as(usize, 8), @as(usize, @intFromFloat(@ceil(arc_length / 5.0))));
            const angle_step = angle_span / @as(f32, @floatFromInt(segments));

            // Stack allocation for reasonable arc sizes
            var stack_points: [256]Point(2, f32) = undefined;
            var points: []Point(2, f32) = undefined;

            const total_points = if (width > 1) (segments + 1) * 2 else segments + 1;

            if (total_points <= stack_points.len) {
                points = stack_points[0..total_points];
            } else {
                // For very large arcs, use heap allocation
                points = try self.allocator.alloc(Point(2, f32), total_points);
            }
            defer if (total_points > stack_points.len) self.allocator.free(points);

            if (width == 1) {
                // Generate points along the arc
                for (0..segments + 1) |i| {
                    const angle = start_angle + @as(f32, @floatFromInt(i)) * angle_step;
                    points[i] = .init(.{
                        center.x() + radius * @cos(angle),
                        center.y() + radius * @sin(angle),
                    });
                }

                // Draw as polyline
                for (0..segments) |i| {
                    self.drawLine(points[i], points[i + 1], color, 1, .soft);
                }
            } else {
                // Generate inner and outer arc points for thick line
                const line_width: f32 = @floatFromInt(width);
                const inner_radius = radius - line_width / 2.0;
                const outer_radius = radius + line_width / 2.0;

                // Generate outer arc (forward)
                for (0..segments + 1) |i| {
                    const angle = start_angle + @as(f32, @floatFromInt(i)) * angle_step;
                    points[i] = .init(.{
                        center.x() + outer_radius * @cos(angle),
                        center.y() + outer_radius * @sin(angle),
                    });
                }

                // Generate inner arc (backward)
                for (0..segments + 1) |i| {
                    const angle = end_angle - @as(f32, @floatFromInt(i)) * angle_step;
                    points[segments + 1 + i] = .init(.{
                        center.x() + inner_radius * @cos(angle),
                        center.y() + inner_radius * @sin(angle),
                    });
                }

                // Draw as filled polygon
                try self.fillPolygon(points, color, .soft);
            }
        }

        /// Fills the given polygon on an image using the scanline algorithm with @memset optimization.
        /// The polygon is defined by an array of points (vertices).
        ///
        /// **Rendering Modes:**
        /// - **DrawMode.fast**: Hard edges, maximum performance with @memset optimization
        /// - **DrawMode.soft**: Antialiased edges, uses alpha blending (no @memset)
        pub fn fillPolygon(self: Self, polygon: []const Point(2, f32), color: anytype, mode: DrawMode) !void {
            comptime assert(isColor(@TypeOf(color)));
            if (polygon.len < 3) return;

            const frows: f32 = @floatFromInt(self.image.rows);
            const fcols: f32 = @floatFromInt(self.image.cols);

            // Find bounding box for optimization
            var min_y = polygon[0].y();
            var max_y = polygon[0].y();
            for (polygon) |p| {
                min_y = @min(min_y, p.y());
                max_y = @max(max_y, p.y());
            }

            const start_y = @max(0, @floor(min_y));
            const end_y = @min(frows - 1, @ceil(max_y));

            // Use stack buffer for small polygons, fallback to heap for complex ones
            var stack_intersections: [polygon_intersection_stack_buffer_size]f32 = undefined;
            var heap_intersections: ?[]f32 = null;
            defer if (heap_intersections) |h| self.allocator.free(h);

            const c2 = convertColor(Rgba, color);
            const solid_color = convertColor(T, c2);

            var y = start_y;
            while (y <= end_y) : (y += 1) {
                var intersection_count: usize = 0;

                // Count intersections first
                for (0..polygon.len) |i| {
                    const p1 = polygon[i];
                    const p2 = polygon[(i + 1) % polygon.len];

                    if ((p1.y() <= y and p2.y() > y) or (p2.y() <= y and p1.y() > y)) {
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

                    if ((p1.y() <= y and p2.y() > y) or (p2.y() <= y and p1.y() > y)) {
                        const intersection = p1.x() + (y - p1.y()) * (p2.x() - p1.x()) / (p2.y() - p1.y());
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

                    const x_start = @max(0, @floor(left_edge));
                    const x_end = @min(fcols - 1, @ceil(right_edge));

                    switch (mode) {
                        .soft => {
                            var x = x_start;
                            while (x <= x_end) : (x += 1) {
                                // Apply antialiasing at edges
                                var alpha: f32 = 1.0;
                                if (x < left_edge + 1) {
                                    alpha = @min(alpha, x + antialias_edge_offset - left_edge);
                                }
                                if (x > right_edge - 1) {
                                    alpha = @min(alpha, right_edge - (x - antialias_edge_offset));
                                }
                                alpha = @max(0, @min(1, alpha));

                                if (alpha > 0) {
                                    self.setPixel(.init(.{ x, y }), c2.fade(alpha));
                                }
                            }
                        },
                        .fast => {
                            // Fast mode - use @memset for optimal span filling
                            self.setHorizontalSpan(left_edge, right_edge, y, solid_color);
                        },
                    }
                }
            }
        }

        /// Fills a circle on the given image.
        /// Use DrawMode.soft for anti-aliased edges or DrawMode.fast for hard edges.
        pub fn fillCircle(self: Self, center: Point(2, f32), radius: f32, color: anytype, mode: DrawMode) void {
            comptime assert(isColor(@TypeOf(color)));
            if (radius <= 0) return;

            switch (mode) {
                .fast => self.fillCircleFast(center, radius, color),
                .soft => self.fillCircleSoft(center, radius, color),
            }
        }

        /// Fills an arc (pie slice) on the given image.
        ///
        /// **Parameters:**
        /// - `center`: The center point of the arc
        /// - `radius`: The radius of the arc (must be positive)
        /// - `start_angle`: Starting angle in radians (0 = positive X-axis)
        /// - `end_angle`: Ending angle in radians
        /// - `color`: The fill color (any color type)
        /// - `mode`: DrawMode.fast for aliased or DrawMode.soft for anti-aliased edges
        ///
        /// **Angle Convention:**
        /// - Angles are measured from the positive X-axis
        /// - Positive angles rotate counter-clockwise
        /// - The filled region includes the center point (pie slice)
        ///
        /// **Performance:**
        /// - Full circles (angle diff ≥ 2π) use optimized circle filling
        /// - Fast mode: O(r²) with optimized scanline filling
        /// - Soft mode: Uses polygon tessellation for smooth edges
        ///
        /// **Example:**
        /// ```zig
        /// // Fill a green pie slice from π/4 to 3π/4 (45° to 135°)
        /// try canvas.fillArc(center, 60, std.math.pi / 4.0, 3.0 * std.math.pi / 4.0, Rgb.green, .soft);
        /// ```
        pub fn fillArc(self: Self, center: Point(2, f32), radius: f32, start_angle: f32, end_angle: f32, color: anytype, mode: DrawMode) !void {
            comptime assert(isColor(@TypeOf(color)));
            if (radius <= 0) return;

            // Validate angles are finite numbers
            if (!std.math.isFinite(start_angle) or !std.math.isFinite(end_angle)) {
                return;
            }

            // Check if this is a full circle (optimize for this case)
            const angle_diff = @abs(end_angle - start_angle);
            if (angle_diff >= 2 * std.math.pi) {
                // Full circle - use optimized circle filling
                switch (mode) {
                    .fast => self.fillCircleFast(center, radius, color),
                    .soft => self.fillCircleSoft(center, radius, color),
                }
                return;
            }

            // Partial arc
            switch (mode) {
                .fast => self.fillArcFast(center, radius, start_angle, end_angle, color),
                .soft => try self.fillArcSoft(center, radius, start_angle, end_angle, color),
            }
        }

        /// Internal function for filling smooth (anti-aliased) circles.
        fn fillCircleSoft(self: Self, center: Point(2, f32), radius: f32, color: anytype) void {
            const frows: f32 = @floatFromInt(self.image.rows);
            const fcols: f32 = @floatFromInt(self.image.cols);
            const left: usize = @intFromFloat(@round(@max(0, center.x() - radius)));
            const top: usize = @intFromFloat(@round(@max(0, center.y() - radius)));
            const right: usize = @intFromFloat(@round(@min(fcols, center.x() + radius)));
            const bottom: usize = @intFromFloat(@round(@min(frows, center.y() + radius)));

            for (top..bottom) |r| {
                const y = as(f32, r) - center.y();
                for (left..right) |c| {
                    const x = as(f32, c) - center.x();
                    const dist_sq = x * x + y * y;
                    if (dist_sq <= radius * radius) {
                        // Apply antialiasing at the edge
                        const dist = @sqrt(dist_sq);
                        if (dist > radius - 1) {
                            // Edge antialiasing
                            const edge_alpha = radius - dist;
                            const rgba_color = convertColor(Rgba, color);
                            self.setPixel(.init(.{ @as(f32, @floatFromInt(c)), @as(f32, @floatFromInt(r)) }), rgba_color.fade(edge_alpha));
                        } else {
                            // Full opacity in the center - direct assignment
                            self.setPixel(.init(.{ @as(f32, @floatFromInt(c)), @as(f32, @floatFromInt(r)) }), color);
                        }
                    }
                }
            }
        }

        /// Internal function for filling solid (non-anti-aliased) circles.
        fn fillCircleFast(self: Self, center: Point(2, f32), radius: f32, color: anytype) void {
            const solid_color = convertColor(T, color);
            const frows: f32 = @floatFromInt(self.image.rows);
            const top = @max(0, center.y() - radius);
            const bottom = @min(frows - 1, center.y() + radius);

            var y = top;
            while (y <= bottom) : (y += 1) {
                const dy = y - center.y();
                const dx = @sqrt(@max(0, radius * radius - dy * dy));

                if (dx > 0) {
                    const x1 = center.x() - dx;
                    const x2 = center.x() + dx;
                    self.setHorizontalSpan(x1, x2, y, solid_color);
                }
            }
        }

        /// Internal function for filling solid (non-anti-aliased) arcs.
        /// Fills a pie slice (arc + lines to center).
        fn fillArcFast(self: Self, center: Point(2, f32), radius: f32, start_angle: f32, end_angle: f32, color: anytype) void {
            const solid_color = convertColor(T, color);
            const frows: f32 = @floatFromInt(self.image.rows);
            const fcols: f32 = @floatFromInt(self.image.cols);

            // Calculate bounding box
            const top = @max(0, center.y() - radius);
            const bottom = @min(frows - 1, center.y() + radius);
            const left = @max(0, center.x() - radius);
            const right = @min(fcols - 1, center.x() + radius);

            // For each scanline in the bounding box
            var y = top;
            while (y <= bottom) : (y += 1) {
                const dy = y - center.y();

                // Calculate the x-range where the circle intersects this scanline
                const dx_max_sq = radius * radius - dy * dy;
                if (dx_max_sq <= 0) continue;

                const dx_max = @sqrt(dx_max_sq);
                const circle_left = center.x() - dx_max;
                const circle_right = center.x() + dx_max;

                // Clamp to image bounds
                const scan_left = @max(left, circle_left);
                const scan_right = @min(right, circle_right);

                // For pie slices, we need to check each pixel individually
                // This is more accurate than the previous sampling approach
                var x = scan_left;
                while (x <= scan_right) : (x += 1) {
                    const dx = x - center.x();

                    // Verify we're inside the circle
                    if (dx * dx + dy * dy <= radius * radius) {
                        // Check if this angle is within the arc
                        const angle = std.math.atan2(dy, dx);
                        if (isAngleInArc(angle, start_angle, end_angle)) {
                            // Find the continuous span of pixels in the arc
                            var span_end = x;
                            while (span_end < scan_right) : (span_end += 1) {
                                const next_dx = span_end + 1 - center.x();
                                if (next_dx * next_dx + dy * dy > radius * radius) break;

                                const next_angle = std.math.atan2(dy, next_dx);
                                if (!isAngleInArc(next_angle, start_angle, end_angle)) break;
                            }

                            // Fill the continuous span
                            self.setHorizontalSpan(x, span_end, y, solid_color);
                            x = span_end; // Skip to end of span
                        }
                    }
                }
            }
        }

        /// Helper: Calculate antialiased coverage for arc boundaries
        inline fn calculateArcCoverage(dist: f32, radius: f32, in_arc: bool, start_cross_product: f32, end_cross_product: f32) f32 {
            const start_cross = @abs(start_cross_product);
            const end_cross = @abs(end_cross_product);

            // Circular boundary coverage
            const circ_coverage = if (dist <= radius - 1.0)
                1.0
            else if (dist < radius + 1.0)
                @max(0, @min(1, radius - dist + 0.5))
            else
                0.0;

            if (!in_arc) {
                // Outside arc - apply edge antialiasing
                var edge_coverage: f32 = 0;
                if (start_cross < 1.0 and start_cross_product < 0) edge_coverage = @max(edge_coverage, 1.0 - start_cross);
                if (end_cross < 1.0 and end_cross_product > 0) edge_coverage = @max(edge_coverage, 1.0 - end_cross);
                return circ_coverage * edge_coverage;
            } else {
                // Inside arc - reduce coverage near edges
                var coverage = circ_coverage;
                if (start_cross < 1.0 and start_cross_product >= 0) coverage = @min(coverage, start_cross);
                if (end_cross < 1.0 and end_cross_product <= 0) coverage = @min(coverage, end_cross);
                return coverage;
            }
        }

        /// Internal function for filling smooth (anti-aliased) arcs.
        fn fillArcSoft(self: Self, center: Point(2, f32), radius: f32, start_angle: f32, end_angle: f32, color: anytype) !void {
            // For full circles, use fillCircle for better quality
            if (@abs(end_angle - start_angle) >= 2 * std.math.pi) {
                self.fillCircle(center, radius, color, .soft);
                return;
            }

            // Precompute edge vectors
            const start_edge = .{ .x = @cos(start_angle), .y = @sin(start_angle) };
            const end_edge = .{ .x = @cos(end_angle), .y = @sin(end_angle) };

            // Calculate bounding box
            const frows: f32 = @floatFromInt(self.image.rows);
            const fcols: f32 = @floatFromInt(self.image.cols);
            const bounds: Rectangle(usize) = .{
                .l = @intFromFloat(@round(@max(0, center.x() - radius - 1))),
                .t = @intFromFloat(@round(@max(0, center.y() - radius - 1))),
                .r = @intFromFloat(@round(@min(fcols, center.x() + radius + 1))),
                .b = @intFromFloat(@round(@min(frows, center.y() + radius + 1))),
            };
            if (bounds.isEmpty()) return;

            const rgba_color = convertColor(Rgba, color);

            // Process each pixel in bounding box
            for (bounds.t..bounds.b) |r| {
                const py = as(f32, r);
                const y = py - center.y();

                for (bounds.l..bounds.r) |c| {
                    const px = as(f32, c);
                    const x = px - center.x();

                    // Quick rejection for pixels far outside circle
                    const dist_sq = x * x + y * y;
                    if (dist_sq > (radius + 1) * (radius + 1)) continue;

                    // Check angle first (before expensive sqrt)
                    const angle = std.math.atan2(y, x);
                    const in_arc = isAngleInArc(angle, start_angle, end_angle);

                    // Calculate cross products for edge proximity (cheap)
                    const start_cross_product = x * start_edge.y - y * start_edge.x;
                    const end_cross_product = x * end_edge.y - y * end_edge.x;

                    // Early reject if outside arc and not near edges
                    if (!in_arc) {
                        const near_start = @abs(start_cross_product) < 1.0 and start_cross_product < 0;
                        const near_end = @abs(end_cross_product) < 1.0 and end_cross_product > 0;
                        if (!near_start and !near_end) continue;
                    }

                    // Now calculate expensive sqrt only for pixels we'll actually render
                    const dist = @sqrt(dist_sq);
                    const coverage = calculateArcCoverage(dist, radius, in_arc, start_cross_product, end_cross_product);
                    if (coverage > 0) {
                        self.setPixel(.init(.{ px, py }), rgba_color.fade(coverage));
                    }
                }
            }
        }

        /// Draws a quadratic Bézier curve with specified width and fill mode.
        pub fn drawQuadraticBezier(
            self: Self,
            p0: Point(2, f32),
            p1: Point(2, f32),
            p2: Point(2, f32),
            color: anytype,
            width: usize,
            mode: DrawMode,
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
            p0: Point(2, f32),
            p1: Point(2, f32),
            p2: Point(2, f32),
            p3: Point(2, f32),
            color: anytype,
            width: usize,
            mode: DrawMode,
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
        pub fn drawSplinePolygon(self: Self, polygon: []const Point(2, f32), color: anytype, width: usize, tension: f32, mode: DrawMode) void {
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
        pub fn fillSplinePolygon(self: Self, polygon: []const Point(2, f32), color: anytype, tension: f32, mode: DrawMode) !void {
            comptime assert(isColor(@TypeOf(color)));
            if (polygon.len < 3) return;

            // Stack buffer for common cases (up to 50 segments per curve, 8 curves)
            var stack_buffer: [spline_polygon_stack_buffer_size]Point(2, f32) = undefined;
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
            var points_buffer: []Point(2, f32) = undefined;
            var heap_buffer: ?[]Point(2, f32) = null;
            defer if (heap_buffer) |h| self.allocator.free(h);

            if (total_points <= spline_polygon_stack_buffer_size) {
                points_buffer = stack_buffer[0..total_points];
            } else {
                heap_buffer = try self.allocator.alloc(Point(2, f32), total_points);
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
        fn evalQuadraticBezier(p0: Point(2, f32), p1: Point(2, f32), p2: Point(2, f32), t: f32) Point(2, f32) {
            const u = 1 - t;
            const uu = u * u;
            const tt = t * t;
            return .init(.{
                uu * p0.x() + 2 * u * t * p1.x() + tt * p2.x(),
                uu * p0.y() + 2 * u * t * p1.y() + tt * p2.y(),
            });
        }

        /// Evaluates a cubic Bézier curve at parameter t.
        /// Uses the standard cubic Bézier formula: (1-t)³P₀ + 3t(1-t)²P₁ + 3t²(1-t)P₂ + t³P₃
        /// Parameter t is in range [0, 1] where 0=start point, 1=end point.
        fn evalCubicBezier(p0: Point(2, f32), p1: Point(2, f32), p2: Point(2, f32), p3: Point(2, f32), t: f32) Point(2, f32) {
            const u = 1 - t;
            const uu = u * u;
            const uuu = uu * u;
            const tt = t * t;
            const ttt = tt * t;
            return .init(.{
                uuu * p0.x() + 3 * uu * t * p1.x() + 3 * u * tt * p2.x() + ttt * p3.x(),
                uuu * p0.y() + 3 * uu * t * p1.y() + 3 * u * tt * p2.y() + ttt * p3.y(),
            });
        }

        /// Estimates the length of a quadratic Bézier curve segment.
        /// Uses chord + control polygon approximation for fast, reasonably accurate estimation.
        /// The estimate is (chord_length + control_polygon_length) / 2.
        fn estimateQuadraticBezierLength(p0: Point(2, f32), p1: Point(2, f32), p2: Point(2, f32)) f32 {
            // Use chord + control polygon approximation
            const chord = p0.distance(p2);
            const control_net = p0.distance(p1) + p1.distance(p2);
            return (chord + control_net) / 2.0;
        }

        /// Estimates the length of a cubic Bézier curve segment.
        /// Uses chord + control polygon approximation for fast, reasonably accurate estimation.
        /// The estimate is (chord_length + control_polygon_length) / 2.
        fn estimateCubicBezierLength(p0: Point(2, f32), p1: Point(2, f32), p2: Point(2, f32), p3: Point(2, f32)) f32 {
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
            buffer: []Point(2, f32),
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
            mode: DrawMode,
        ) void {
            var stack_buffer: [bezier_max_segments_count]Point(2, f32) = undefined;

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
        fn calculateSmoothControlPoints(p0: Point(2, f32), p1: Point(2, f32), p2: Point(2, f32), tension: f32) struct { cp1: Point(2, f32), cp2: Point(2, f32) } {
            const tension_factor = 1 - @max(0, @min(1, tension));
            return .{
                .cp1 = .init(.{
                    p0.x() + (p1.x() - p0.x()) * tension_factor,
                    p0.y() + (p1.y() - p0.y()) * tension_factor,
                }),
                .cp2 = .init(.{
                    p1.x() - (p2.x() - p1.x()) * tension_factor,
                    p1.y() - (p2.y() - p1.y()) * tension_factor,
                }),
            };
        }

        /// Helper function to get a bit value from glyph bitmap data.
        /// Returns 1 if the bit is set, 0 otherwise.
        inline fn getGlyphBit(char_data: []const u8, row: usize, col: usize, bytes_per_row: usize) u1 {
            const byte_idx = col / 8;
            const bit_idx = col % 8;
            const row_byte_offset = row * bytes_per_row + byte_idx;
            if (row_byte_offset >= char_data.len) return 0;
            return @intCast((char_data[row_byte_offset] >> @intCast(bit_idx)) & 1);
        }

        /// Helper function to calculate bytes per row for a glyph.
        /// Handles both fixed-width and variable-width fonts.
        inline fn calculateGlyphBytesPerRow(glyph_info: anytype, font: anytype) usize {
            // Variable-width fonts use glyph-specific width, fixed-width fonts use font-wide stride
            return if (font.glyph_map != null)
                (@as(usize, glyph_info.width) + 7) / 8
            else
                font.bytesPerRow();
        }

        /// Draws text at the specified position using a bitmap font.
        /// The position specifies the top-left corner of the text.
        /// Supports newlines for multi-line text.
        pub fn drawText(self: Self, text: []const u8, position: Point(2, f32), color: anytype, font: BitmapFont, scale: f32, mode: DrawMode) void {
            comptime assert(isColor(@TypeOf(color)));
            if (scale <= 0) return;

            // Compute text bounding box and early exit if outside image
            const text_bounds = font.getTextBounds(text, scale);
            const text_rect = Rectangle(f32){
                .l = position.x() + text_bounds.l,
                .t = position.y() + text_bounds.t,
                .r = position.x() + text_bounds.r,
                .b = position.y() + text_bounds.b,
            };
            const image_rect: Rectangle(f32) = .{
                .l = 0,
                .t = 0,
                .r = @floatFromInt(self.cols()),
                .b = @floatFromInt(self.rows()),
            };
            const clip_rect = text_rect.intersect(image_rect) orelse return;

            var x = position.x();
            var y = position.y();
            const start_x = x;

            // For scale 1.0, use simple pixel-by-pixel drawing
            if (scale == 1.0) {
                var utf8_iter = std.unicode.Utf8Iterator{ .bytes = text, .i = 0 };
                while (utf8_iter.nextCodepoint()) |codepoint| {
                    if (codepoint == '\n') {
                        x = start_x;
                        y += @floatFromInt(font.char_height);
                        continue;
                    }

                    if (font.getGlyphInfo(codepoint)) |glyph_info| {
                        if (font.getCharData(codepoint)) |char_data| {
                            const bitmap_bytes_per_row = calculateGlyphBytesPerRow(glyph_info, font);
                            const render_height = if (font.glyph_map == null) font.char_height else glyph_info.height;

                            // Draw the character bitmap
                            for (0..render_height) |row| {
                                for (0..glyph_info.width) |col| {
                                    if (getGlyphBit(char_data, row, col, bitmap_bytes_per_row) != 0) {
                                        const px = x + @as(f32, @floatFromInt(col)) + @as(f32, @floatFromInt(glyph_info.x_offset));
                                        const py = y + @as(f32, @floatFromInt(row)) + @as(f32, @floatFromInt(glyph_info.y_offset));
                                        self.setPixel(.init(.{ px, py }), color);
                                    }
                                }
                            }
                        }
                    }
                    // Use character-specific advance width if available
                    const advance = font.getCharAdvanceWidth(codepoint);
                    x += @floatFromInt(advance);
                }
                return;
            }

            // Scaled rendering
            const char_height_scaled = @as(f32, @floatFromInt(font.char_height)) * scale;
            const rgba_color = if (mode == .soft) convertColor(Rgba, color) else undefined;

            var utf8_iter = std.unicode.Utf8Iterator{ .bytes = text, .i = 0 };
            while (utf8_iter.nextCodepoint()) |codepoint| {
                if (codepoint == '\n') {
                    x = start_x;
                    y += char_height_scaled;
                    continue;
                }

                if (font.getGlyphInfo(codepoint)) |glyph_info| {
                    if (font.getCharData(codepoint)) |char_data| {
                        const glyph_bytes_per_row = calculateGlyphBytesPerRow(glyph_info, font);

                        switch (mode) {
                            .fast => {
                                // Fast mode: nearest-neighbor scaling
                                for (0..glyph_info.height) |row| {
                                    for (0..glyph_info.width) |col| {
                                        if (getGlyphBit(char_data, row, col, glyph_bytes_per_row) != 0) {
                                            // Draw a scaled pixel block
                                            const base_x = x + (@as(f32, @floatFromInt(col)) + @as(f32, @floatFromInt(glyph_info.x_offset))) * scale;
                                            const base_y = y + (@as(f32, @floatFromInt(row)) + @as(f32, @floatFromInt(glyph_info.y_offset))) * scale;

                                            // Calculate the integer bounds of the scaled pixel
                                            const x_start_f = @floor(base_x);
                                            const y_start_f = @floor(base_y);
                                            const x_end_f = @ceil(base_x + scale);
                                            const y_end_f = @ceil(base_y + scale);

                                            // Clip to the valid rectangle
                                            const x_start = @as(usize, @intFromFloat(@max(x_start_f, clip_rect.l)));
                                            const y_start = @as(usize, @intFromFloat(@max(y_start_f, clip_rect.t)));
                                            const x_end = @as(usize, @intFromFloat(@min(x_end_f, clip_rect.r)));
                                            const y_end = @as(usize, @intFromFloat(@min(y_end_f, clip_rect.b)));

                                            // Fill the pixel block
                                            if (x_start < x_end and y_start < y_end) {
                                                for (y_start..y_end) |py| {
                                                    for (x_start..x_end) |px| {
                                                        self.setPixel(.init(.{ @as(f32, @floatFromInt(px)), @as(f32, @floatFromInt(py)) }), color);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            .soft => {
                                // Soft mode: antialiased scaling with box filtering
                                const glyph_width_f = @as(f32, @floatFromInt(glyph_info.width));
                                const glyph_height_f = @as(f32, @floatFromInt(glyph_info.height));
                                const dest_width = @ceil(@as(f32, @floatFromInt(glyph_info.width)) * scale);
                                const dest_height = @ceil(@as(f32, @floatFromInt(glyph_info.height)) * scale);

                                var dy: f32 = 0;
                                while (dy < dest_height) : (dy += 1) {
                                    var dx: f32 = 0;
                                    while (dx < dest_width) : (dx += 1) {
                                        const dest_x = x + dx + @as(f32, @floatFromInt(glyph_info.x_offset)) * scale;
                                        const dest_y = y + dy + @as(f32, @floatFromInt(glyph_info.y_offset)) * scale;

                                        if (self.atOrNull(@intFromFloat(dest_y), @intFromFloat(dest_x))) |_| {
                                            // Calculate which part of the source we're sampling
                                            const src_x = dx / scale;
                                            const src_y = dy / scale;

                                            // Box filter: sample a box around the source position
                                            const sample_radius = 0.5 / scale;
                                            const x0 = src_x - sample_radius;
                                            const x1 = src_x + sample_radius;
                                            const y0 = src_y - sample_radius;
                                            const y1 = src_y + sample_radius;

                                            var total_coverage: f32 = 0;

                                            // Sample the font bitmap
                                            const row_start_f = @max(0, @floor(y0));
                                            const row_end_f = @min(glyph_height_f - 1, @ceil(y1));
                                            const col_start_f = @max(0, @floor(x0));
                                            const col_end_f = @min(glyph_width_f - 1, @ceil(x1));

                                            var row_f = row_start_f;
                                            while (row_f <= row_end_f) : (row_f += 1) {
                                                var col_f = col_start_f;
                                                while (col_f <= col_end_f) : (col_f += 1) {
                                                    const row_idx = @as(usize, @intFromFloat(row_f));
                                                    const col_idx = @as(usize, @intFromFloat(col_f));

                                                    if (getGlyphBit(char_data, row_idx, col_idx, glyph_bytes_per_row) != 0) {
                                                        // Calculate how much this pixel contributes
                                                        const px0 = col_f;
                                                        const px1 = col_f + 1;
                                                        const py0 = row_f;
                                                        const py1 = row_f + 1;

                                                        const overlap_x = @max(0, @min(x1, px1) - @max(x0, px0));
                                                        const overlap_y = @max(0, @min(y1, py1) - @max(y0, py0));
                                                        total_coverage += overlap_x * overlap_y;
                                                    }
                                                }
                                            }

                                            // Normalize coverage by the box area
                                            const box_area = (x1 - x0) * (y1 - y0);
                                            const normalized_coverage = total_coverage / box_area;

                                            if (normalized_coverage > 0) {
                                                self.setPixel(.init(.{ dest_x, dest_y }), rgba_color.fade(normalized_coverage));
                                            }
                                        }
                                    }
                                }
                            },
                        }
                    }
                }
                // Use character-specific advance width if available
                const advance = font.getCharAdvanceWidth(codepoint);
                x += @as(f32, @floatFromInt(advance)) * scale;
            }
        }
    };
}
