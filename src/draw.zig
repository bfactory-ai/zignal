const std = @import("std");
const assert = std.debug.assert;

const as = @import("meta.zig").as;
const colorspace = @import("colorspace.zig");
const Image = @import("image.zig").Image;
const Rgba = @import("colorspace.zig").Rgba;

const Point2d = @import("point.zig").Point2d;
const Rectangle = @import("geometry.zig").Rectangle;

/// Draws a colored straight of a custom width between p1 and p2 on image.  It uses Xialin
/// Wu's line algorithm to perform antialiasing along the diagonal.  Moreover, if color is of
/// Rgba type, it alpha-blends it on top of the image.
pub fn drawLine(
    comptime T: type,
    image: Image(T),
    p1: Point2d(f32),
    p2: Point2d(f32),
    width: usize,
    color: anytype,
) void {
    comptime assert(colorspace.isColor(@TypeOf(color)));
    if (width == 0) return;
    // To avoid casting all the time, perform all operations using the underlying type of p1 and p2.
    const Float = @TypeOf(p1.x);
    var x1 = @round(p1.x);
    var y1 = @round(p1.y);
    var x2 = @round(p2.x);
    var y2 = @round(p2.y);
    const rows: Float = @floatFromInt(image.rows);
    const cols: Float = @floatFromInt(image.cols);
    const half_width: Float = @floatFromInt(width / 2);
    var c2 = colorspace.convert(Rgba, color);

    if (x1 == x2) {
        if (y1 > y2) std.mem.swap(Float, &y1, &y2);
        if (x1 < 0 or x1 >= cols) return;
        var y = y1;
        while (y <= y2) : (y += 1) {
            if (y < 0 or y >= rows) continue;
            var i = -half_width;
            while (i <= half_width) : (i += 1) {
                const px = x1 + i;
                if (px >= 0 and px < cols) {
                    const pos = as(usize, y) * image.cols + as(usize, px);
                    var c1 = colorspace.convert(Rgba, image.data[pos]);
                    c1.blend(c2);
                    image.data[pos] = colorspace.convert(T, c1);
                }
            }
        }
    } else if (y1 == y2) {
        if (x1 > x2) std.mem.swap(Float, &x1, &x2);
        if (y1 < 0 or y1 > rows) return;
        var x = x1;
        while (x <= x2) : (x += 1) {
            if (x < 0 or x >= cols) continue;
            var i = -half_width;
            while (i <= half_width) : (i += 1) {
                const py = y1 + i;
                if (py >= 0 and py < rows) {
                    const pos = as(usize, py) * image.cols + as(usize, x);
                    var c1 = colorspace.convert(Rgba, image.data[pos]);
                    c1.blend(c2);
                    image.data[pos] = colorspace.convert(T, c1);
                }
            }
        }
    } else {
        // This part is a little more complicated because we are going to perform alpha blending
        // so the diagonal lines look nice.
        const max_alpha: Float = @floatFromInt(c2.a);
        const rise = y2 - y1;
        const run = x2 - x1;
        if (@abs(rise) < @abs(run)) {
            const slope = rise / run;
            const first = if (x1 > x2) @max(x2, 0) else @max(x1, 0);
            const last = if (x1 > x2) @min(x1, cols - 1) else @min(x2, cols - 1);
            var i = first;
            while (i <= last) : (i += 1) {
                const dy = slope * (i - x1) + y1;
                const dx = i;
                const y = @floor(dy);
                const x = @floor(dx);
                if (y >= 0 and y <= rows - 1) {
                    var j = -half_width;
                    while (j <= half_width) : (j += 1) {
                        const py = @max(0, y + j);
                        const pos = as(usize, py) * image.cols + as(usize, x);
                        if (py >= 0 and py < rows) {
                            var c1: Rgba = colorspace.convert(Rgba, image.data[pos]);
                            if (j == -half_width or j == half_width) {
                                c2.a = @intFromFloat((1 - (dy - y)) * max_alpha);
                                c1.blend(c2);
                                image.data[pos] = colorspace.convert(T, c1);
                            } else {
                                c1.blend(c2);
                                image.data[pos] = colorspace.convert(T, c1);
                            }
                        }
                    }
                }
                if (y + 1 >= 0 and y + 1 <= rows - 1) {
                    var j = -half_width;
                    while (j <= half_width) : (j += 1) {
                        const py = @max(0, y + 1 + j);
                        if (py >= 0 and py < rows) {
                            const pos = as(usize, py) * image.cols + as(usize, x);
                            var c1: Rgba = colorspace.convert(Rgba, image.data[pos]);
                            if (j == -half_width or j == half_width) {
                                c2.a = @intFromFloat((dy - y) * max_alpha);
                                c1.blend(c2);
                                image.data[pos] = colorspace.convert(T, c1);
                            } else {
                                c1.blend(c2);
                                image.data[pos] = colorspace.convert(T, c1);
                            }
                        }
                    }
                }
            }
        } else {
            const slope = run / rise;
            const first = if (y1 > y2) @max(y2, 0) else @max(y1, 0);
            const last = if (y1 > y2) @min(y1, rows - 1) else @min(y2, rows - 1);
            var i = first;
            while (i <= last) : (i += 1) {
                const dx = slope * (i - y1) + x1;
                const dy = i;
                const y = @floor(dy);
                const x = @floor(dx);
                if (x >= 0 and x <= cols - 1) {
                    var j = -half_width;
                    while (j <= half_width) : (j += 1) {
                        const px = @max(0, x + j);
                        const pos = as(usize, y) * image.cols + as(usize, px);
                        if (px >= 0 and px < cols) {
                            var c1: Rgba = colorspace.convert(Rgba, image.data[pos]);
                            if (j == -half_width or j == half_width) {
                                c2.a = @intFromFloat((1 - (dx - x)) * max_alpha);
                                c1.blend(c2);
                                image.data[pos] = colorspace.convert(T, c1);
                            } else {
                                c1.blend(c2);
                                image.data[pos] = colorspace.convert(T, c1);
                            }
                        }
                    }
                }
                if (x + 1 >= 0 and x + 1 <= cols - 1) {
                    c2.a = @intFromFloat((dx - x) * max_alpha);
                    var j = -half_width;
                    while (j <= half_width) : (j += 1) {
                        const px = @max(0, x + 1 + j);
                        const pos = as(usize, y) * image.cols + as(usize, px);
                        if (px >= 0 and px < cols) {
                            var c1: Rgba = colorspace.convert(Rgba, image.data[pos]);
                            if (j == -half_width or j == half_width) {
                                c1.blend(c2);
                                image.data[pos] = colorspace.convert(T, c1);
                            } else {
                                c1.blend(c2);
                                image.data[pos] = colorspace.convert(T, c1);
                            }
                        }
                    }
                }
            }
        }
    }
}
/// Draws a colored straight line of a custom width between p1 and p2 on image, using Bresenham's line algorithm.
pub fn drawLineFast(
    comptime T: type,
    image: Image(T),
    p1: Point2d(f32),
    p2: Point2d(f32),
    width: usize,
    color: T,
) void {
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
                if (px >= 0 and px < image.cols and py >= 0 and py < image.rows) {
                    const pos = @as(usize, @intCast(py)) * image.cols + @as(usize, @intCast(px));
                    image.data[pos] = color;
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
///
/// - **T**: The type of color used in the image, must be a color type.
/// - **image**: The `Image` object where the curve will be drawn.
/// - **points**: An array of 4 `Point2d` representing the control points of the Bézier curve.
///   The order is [start, first control, second control, end].
/// - **step**: The step size for t in the range [0, 1] for drawing the curve.
/// - **color**: The color to use for drawing the curve, of type `T`.
///
/// The function calculates points along the Bézier curve using the given control points and
/// draws them on the image. The curve's resolution is determined by the `step` parameter,
/// where smaller steps result in a smoother but more computationally intensive curve.
pub fn drawBezierCurve(
    comptime T: type,
    image: Image(T),
    points: [4]Point2d(f32),
    step: f32,
    color: T,
) void {
    comptime assert(colorspace.isColor(T));
    assert(step >= 0);
    assert(step <= 1);
    var t: f32 = 0;
    while (t <= 1) : (t += step) {
        const b: Point2d(f32) = .{
            .x = (1 - t) * (1 - t) * (1 - t) * points[0].x +
                3 * (1 - t) * (1 - t) * t * points[1].x +
                3 * (1 - t) * t * t * points[2].x +
                t * t * t * points[3].x,
            .y = (1 - t) * (1 - t) * (1 - t) * points[0].y +
                3 * (1 - t) * (1 - t) * t * points[1].y +
                3 * (1 - t) * t * t * points[2].y +
                t * t * t * points[3].y,
        };
        const row: usize = @intFromFloat(@round(b.y));
        const col: usize = @intFromFloat(@round(b.x));
        image.data[row * image.cols + col] = color;
    }
}

/// Tessellates a cubic Bézier curve into a series of points.
///
/// - **allocator**: An allocator for memory management.
/// - **points**: An array of 4 `Point2d` representing the control points of the Bézier curve.
///   The order is [start, first control, second control, end].
/// - **segments**: Number of segments to divide the curve into, affecting the resolution of the tessellation.
///
/// The caller owns the resulting slice.
fn tessellateCurve(
    allocator: std.mem.Allocator,
    p: [4]Point2d(f32),
    segments: usize,
) error.OutOfMemory![]const Point2d(f32) {
    var polygon: std.ArrayList(Point2d(f32)) = .init(allocator);
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

/// Draws a smooth polygon on the given image, using Bézier curves to connect points for a curved effect.
///
/// - **T**: The pixel type used in the image, must be a color type.
/// - **image**: The `Image` object where the polygon will be drawn.
/// - **polygon**: A slice of `Point2d` representing the vertices of the polygon.
/// - **color**: The color to use for drawing the polygon's edges, of type `T`.
/// - **tension**: A float value [0, 1] that controls how much the curve 'tenses' or straightens between vertices.
///   - 0 results in straight lines between points.
///   - 1 results in the maximum curve smoothness.
pub fn drawSmoothPolygon(
    comptime T: type,
    image: Image(T),
    polygon: []const Point2d(f32),
    color: T,
    tension: f32,
) void {
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
        drawBezierCurve(T, image, .{ p0, cp1, cp2, p1 }, 0.01, color);
    }
}

/// Fills a smooth polygon on the given image, using tessellated Bézier curves to create a curved outline before filling.
///
/// - **allocator**: An allocator for memory management.
/// - **T**: The pixel type used in the image, must be a color type.
/// - **image**: The `Image` object where the polygon will be filled.
/// - **polygon**: A slice of `Point2d` representing the vertices of the polygon to be filled.
/// - **color**: The color to use for filling the polygon, of type `T`.
/// - **tension**: A float value [0, 1] that controls the curvature of the polygon's edges:
///   - 0 results in straight lines between points.
///   - 1 results in the maximum curve smoothness.
pub fn fillSmoothPolygon(
    allocator: std.mem.Allocator,
    comptime T: type,
    image: Image(T),
    polygon: []const Point2d(f32),
    color: T,
    tension: f32,
) !void {
    var points: std.ArrayList(Point2d(f32)) = .init(allocator);
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
        const segment = try tessellateCurve(allocator, .{ p0, cp1, cp2, p1 }, 10);
        try points.appendSlice(segment);
    }
    fillPolygon(allocator, T, image, points.items, color);
}

/// Draws the given rectangle with the specified width and color.
pub fn drawRectangle(
    comptime T: type,
    image: Image(T),
    rect: Rectangle(f32),
    width: usize,
    color: anytype,
) void {
    comptime assert(colorspace.isColor(@TypeOf(color)));
    const points: []const Point2d(f32) = &.{
        .{ .x = rect.l, .y = rect.t },
        .{ .x = rect.r, .y = rect.t },
        .{ .x = rect.r, .y = rect.b },
        .{ .x = rect.l, .y = rect.b },
    };
    drawPolygon(T, image, points, width, color);
}

/// Draws a cross shape on the given image at a specified center point.
///
/// - **T**: The pixel type used in the image, must be a color type.
/// - **image**: The `Image` object where the cross will be drawn.
/// - **center**: A `Point2d` which defines the center of the cross.
/// - **size**: The length of each arm of the cross measured in pixels.
/// - **color**: The color to use for drawing the cross.
pub fn drawCross(
    comptime T: type,
    image: Image(T),
    center: Point2d(f32),
    size: usize,
    color: T,
) void {
    comptime assert(colorspace.isColor(T));
    if (size == 0) return;
    const x: usize = @intFromFloat(@round(@max(0, @min(as(f32, image.cols - 1), center.x))));
    const y: usize = @intFromFloat(@round(@max(0, @min(as(f32, image.rows - 1), center.y))));
    for (0..size) |i| {
        image.data[y * image.cols + x -| i] = color; // max(0, x - i)
        image.data[(y -| i) * image.cols + x] = color;
        image.data[y * image.cols + @min(image.cols - 1, x + i)] = color;
        image.data[@min(image.rows - 1, y + i) * image.cols + x] = color;
    }
}

/// Draws the given polygon defined as an array of points.
pub fn drawPolygon(
    comptime T: type,
    image: Image(T),
    polygon: []const Point2d(f32),
    width: usize,
    color: anytype,
) void {
    comptime assert(colorspace.isColor(@TypeOf(color)));
    if (width == 0) return;
    for (0..polygon.len) |i| {
        drawLine(T, image, polygon[i], polygon[@mod(i + 1, polygon.len)], width, color);
    }
}

/// Draws the circle defined by its center and radius.
pub fn drawCircle(
    comptime T: type,
    image: Image(T),
    center: Point2d(f32),
    radius: f32,
    color: T,
) void {
    if (radius <= 0) return;
    const frows: f32 = @floatFromInt(image.rows);
    const fcols: f32 = @floatFromInt(image.cols);
    if (center.x - radius < 0 or
        center.x + radius >= fcols or
        center.y - radius < 0 or
        center.y + radius >= frows or
        radius < 1)
    {
        return;
    }

    const first_x: f32 = @round(@max(0, center.x - radius));
    const last_x: f32 = @round(@min(fcols, center.x + radius));
    const rs: f32 = radius * radius;

    var top: f32 = @round(@sqrt(@max(rs - (first_x - center.x - 0.5) * (first_x - center.x - 0.5), 0)));
    top += center.y;
    var last = top;

    // Draw the left half of the circle
    var middle = @min(@round(center.x) - 1, last_x);
    var i: f32 = first_x;
    while (i <= middle) : (i += 1) {
        const a = i - center.x + 0.5;
        top = @round(@sqrt(@max(rs - a * a, 0)));
        top += center.y;
        const temp = top;
        while (top >= last) : (top -= 1) {
            const bottom = @round(center.y - top + center.y);
            drawLine(T, image, .{ .x = i, .y = top }, .{ .x = i, .y = bottom }, 1, color);
        }
        last = temp;
    }

    top = @round(@sqrt(@max(rs - (last_x - center.x + 0.5) * (last_x - center.x + 0.5), 0)));
    top += center.y;
    last = top;

    // Draw the right half of the circle
    middle = @max(@round(center.x), first_x);
    i = last_x;
    while (i >= middle) : (i -= 1) {
        const a = i - center.x - 0.5;
        top = @round(@sqrt(@max(rs - a * a, 0)));
        top += center.y;
        const temp = top;
        while (top >= last) : (top -= 1) {
            const bottom = @round(center.y - top + center.y);
            drawLine(T, image, .{ .x = i, .y = top }, .{ .x = i, .y = bottom }, 1, color);
        }
        last = temp;
    }
}

/// Draws the circle defined by its center and radius using a fast, but less accurate algorithm.
pub fn drawCircleFast(
    comptime T: type,
    image: Image(T),
    center: Point2d(f32),
    radius: f32,
    color: T,
) void {
    if (radius <= 0) return;
    const frows: f32 = @floatFromInt(image.rows);
    const fcols: f32 = @floatFromInt(image.cols);
    const left: usize = @intFromFloat(@round(@max(0, center.x - radius)));
    const top: usize = @intFromFloat(@round(@max(0, center.y - radius)));
    const right: usize = @intFromFloat(@round(@min(fcols, center.x + radius)));
    const bottom: usize = @intFromFloat(@round(@min(frows, center.y + radius)));
    for (top..bottom) |r| {
        const y = as(f32, r) - center.y;
        for (left..right) |c| {
            const x = as(f32, c) - center.x;
            if (x * x + y * y <= radius * radius) {
                image.data[r * image.cols + c] = color;
            }
        }
    }
}

/// Fills the given polygon defined as an array of points on image using the scanline algorithm.
pub fn fillPolygon(
    allocator: std.mem.Allocator,
    comptime T: type,
    image: Image(T),
    polygon: []const Point2d(f32),
    color: T,
) !void {
    const rows = image.rows;
    const cols = image.cols;
    var intersections: std.ArrayList(f32) = .init(allocator);
    defer intersections.deinit();
    for (0..rows) |r| {
        const y: f32 = @floatFromInt(r);
        intersections.clearRetainingCapacity();
        const frows: f32 = @floatFromInt(rows);
        for (0..polygon.len) |i| {
            const p1 = &polygon[i];
            const p2 = &polygon[(i + 1) % polygon.len];

            // Skip points above / below the image
            if (p1.y < 0 or p1.y < 0 or p1.y >= frows or p2.y >= frows) {
                continue;
            }

            // Skip horizontal lines
            if (p1.y == p2.y) {
                continue;
            }

            // Skip if there are no intersections at this position
            if ((y <= p1.y and y <= p2.y) or (y > p1.y and y > p2.y)) {
                continue;
            }

            // Add x coordinates of intersecting points
            try intersections.append(p1.x + (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y));
        }

        std.mem.sort(f32, intersections.items, {}, std.sort.asc(f32));

        // Paint the inside of the polygon
        var i: usize = 1;
        while (i < intersections.items.len) : (i += 2) {
            var left_x: i32 = @intFromFloat(@ceil(intersections.items[i - 1]));
            var right_x: i32 = @intFromFloat(@floor(intersections.items[i]));
            left_x = @max(0, left_x);
            right_x = @min(as(i32, cols), right_x);
            while (left_x <= right_x) : (left_x += 1) {
                const pos: usize = r * cols + as(usize, left_x);
                if (T == Rgba) {
                    image.data[pos].blend(color);
                } else {
                    image.data[pos] = color;
                }
            }
        }
    }
}
